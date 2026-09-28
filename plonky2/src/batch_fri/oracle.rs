#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::extension::Extendable;
use plonky2_field::fft::FftRootTable;
use plonky2_field::packed::PackedField;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use plonky2_util::{log2_strict, reverse_index_bits_in_place};

use crate::batch_fri::prover::batch_fri_proof;
use crate::fri::proof::FriProof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::batch_merkle_tree::BatchMerkleTree;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{reverse_bits, transpose};

/// Represents a batch FRI oracle, i.e. a batch of polynomials with different degrees which have
/// been Merkle-ized in a [`BatchMerkleTree`].
#[derive(Eq, PartialEq, Debug)]
pub struct BatchFriOracle<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub batch_merkle_tree: BatchMerkleTree<F, C::Hasher>,
    // The degree bits of each polynomial group.
    pub degree_bits: Vec<usize>,
    pub rate_bits: usize,
    pub blinding: bool,
}

const SALT_SIZE: usize = 4;

/// Computes the LDE of a set of polynomials.
fn lde_values<F: RichField + Extendable<D>, const D: usize>(
    polynomials: &[PolynomialCoeffs<F>],
    rate_bits: usize,
    blinding: bool,
    fft_root_table: Option<&FftRootTable<F>>,
) -> Vec<Vec<F>> {
    let degree = polynomials[0].len();
    let salt_size = if blinding { SALT_SIZE } else { 0 };

    polynomials
        .par_iter()
        .map(|p| {
            assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
            p.lde(rate_bits)
                .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                .values
        })
        .chain(
            (0..salt_size)
                .into_par_iter()
                .map(|_| F::rand_vec(degree << rate_bits)),
        )
        .collect()
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BatchFriOracle<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
    ) -> Self {
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
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
    ) -> Self {
        let mut degree_bits = polynomials
            .iter()
            .map(|p| log2_strict(p.len()))
            .collect_vec();
        assert!(degree_bits.windows(2).all(|pair| { pair[0] >= pair[1] }));

        let num_polynomials = polynomials.len();
        let mut group_start = 0;
        let mut leaves = Vec::new();

        for (i, d) in degree_bits.iter().enumerate() {
            if i == num_polynomials - 1 || *d > degree_bits[i + 1] {
                let lde_values = timed!(
                    timing,
                    "FFT + blinding",
                    lde_values::<F, D>(
                        &polynomials[group_start..i + 1],
                        rate_bits,
                        blinding,
                        fft_root_table[i]
                    )
                );

                let mut leaf_group = timed!(timing, "transpose LDEs", transpose(&lde_values));
                reverse_index_bits_in_place(&mut leaf_group);
                leaves.push(leaf_group);

                group_start = i + 1;
            }
        }

        let batch_merkle_tree = timed!(
            timing,
            "build Field Merkle tree",
            BatchMerkleTree::new(leaves, cap_height)
        );

        degree_bits.sort_unstable();
        degree_bits.dedup();
        degree_bits.reverse();
        assert_eq!(batch_merkle_tree.leaves.len(), degree_bits.len());
        Self {
            polynomials,
            batch_merkle_tree,
            degree_bits,
            rate_bits,
            blinding,
        }
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        degree_bits: &[usize],
        instances: &[FriInstanceInfo<F, D>],
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert_eq!(degree_bits.len(), instances.len());
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        let mut final_lde_polynomial_coeff = Vec::with_capacity(instances.len());
        let mut final_lde_polynomial_values = Vec::with_capacity(instances.len());
        for (i, instance) in instances.iter().enumerate() {
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

            assert_eq!(final_poly.len(), 1 << degree_bits[i]);
            let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
            let lde_final_values = timed!(
                timing,
                &format!("perform final FFT {}", lde_final_poly.len()),
                lde_final_poly.coset_fft(F::coset_shift().into())
            );
            final_lde_polynomial_coeff.push(lde_final_poly);
            final_lde_polynomial_values.push(lde_final_values);
        }

        batch_fri_proof::<F, C, D>(
            &oracles
                .iter()
                .map(|o| &o.batch_merkle_tree)
                .collect::<Vec<_>>(),
            final_lde_polynomial_coeff[0].clone(),
            &final_lde_polynomial_values,
            challenger,
            fri_params,
            timing,
        )
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(
        &self,
        degree_bits_index: usize,
        index: usize,
        step: usize,
        slice_start: usize,
        slice_len: usize,
    ) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_bits[degree_bits_index] + self.rate_bits);
        let slice = &self.batch_merkle_tree.leaves[degree_bits_index][index];
        &slice[slice_start..slice_start + slice_len]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(
        &self,
        degree_bits_index: usize,
        index_start: usize,
        step: usize,
        slice_start: usize,
        slice_len: usize,
    ) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| {
                self.get_lde_values(
                    degree_bits_index,
                    index_start + i,
                    step,
                    slice_start,
                    slice_len,
                )
            })
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
}
