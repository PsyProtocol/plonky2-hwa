#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::{fri_proof_with, FriValues};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, ProverCompute};
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::reverse_bits;
use crate::util::timing::TimingTree;
/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<Self> {
        C::Compute::compute_from_values(
            timing,
            values,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
        /*
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )*/
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<Self> {
        C::Compute::compute_from_coeffs(
            timing,
            polynomials,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha_value = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha_value);

        // On the GPU: the final polynomial from the proof's device LDEs, its values in the first
        // commit-phase layer's order, and that layer's tree.
        #[cfg(feature = "cuda")]
        let first_arity_bits = fri_params.reduction_arity_bits.first().copied();
        #[cfg(feature = "cuda")]
        let gpu = crate::util::stage_log::acc("fri_gpu_final", || {
            crate::gpu::fri::final_poly::<F, C, D>(
                instance,
                oracles,
                alpha_value,
                fri_params.config.rate_bits,
                first_arity_bits,
                fri_params.config.cap_height,
            )
        });
        #[cfg(feature = "cuda")]
        let (lde_final_poly, lde_final_values) = match gpu {
            Some(g) if !crate::gpu::fri::check_enabled() => (g.coeffs, FriValues::LeafOrder(g.leaf_values, g.first_tree)),
            gpu => {
                if gpu.is_none() {
                    crate::gpu::fri::record_cpu();
                }
                let (coeffs, values) = Self::final_poly_cpu(instance, oracles, &mut alpha, fri_params, timing);
                if let Some(g) = gpu {
                    let mut leaf = values.values.clone();
                    crate::util::reverse_index_bits_in_place(&mut leaf);
                    let tree_ok = match (&g.first_tree, first_arity_bits) {
                        (Some(t), Some(a)) => crate::gpu::fri::tree_matches(t, &g.leaf_values, a, fri_params.config.cap_height),
                        _ => true,
                    };
                    if g.coeffs.coeffs != coeffs.coeffs || g.leaf_values != leaf || !tree_ok {
                        crate::gpu::fri::record_mismatch();
                    }
                }
                (coeffs, FriValues::Natural(values))
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (lde_final_poly, lde_final_values) = {
            let (coeffs, values) = Self::final_poly_cpu(instance, oracles, &mut alpha, fri_params, timing);
            (coeffs, FriValues::Natural(values))
        };

        let fri_proof = fri_proof_with::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }

    /// The final polynomial's LDE coefficients and values: every opened polynomial reduced with
    /// powers of alpha, divided by `X - z` per batch, summed and extended.
    #[allow(clippy::type_complexity)]
    fn final_poly_cpu(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        alpha: &mut ReducingFactor<F::Extension>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> (PolynomialCoeffs<F::Extension>, PolynomialValues<F::Extension>) {
        let final_poly = crate::util::stage_log::acc("fri_reduce_polys", || {
            let mut final_poly = PolynomialCoeffs::empty();
            for FriBatchInfo { point, polynomials } in &instance.batches {
                let polys_coeff = polynomials.iter().map(|fri_poly| {
                    &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
                });
                let composition_poly = timed!(
                    timing,
                    &format!("reduce batch of {} polynomials", polynomials.len()),
                    alpha.reduce_polys_base(polys_coeff)
                );
                let mut quotient = composition_poly.divide_by_linear(*point);
                quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
                alpha.shift_poly(&mut final_poly);
                final_poly += quotient;
            }
            final_poly
        });

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            crate::util::stage_log::acc("fri_final_fft", || lde_final_poly
                .coset_fft(F::coset_shift().into()))
        );
        (lde_final_poly, lde_final_values)
    }
}
