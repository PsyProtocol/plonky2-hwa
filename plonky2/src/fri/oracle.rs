#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::{fri_proof, FriInitialTree};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::reverse_bits;
use crate::plonk::config::ProverCompute;
/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

#[derive(Debug)]
pub enum LdeValues<'a, F> {
    Borrowed(&'a [F]),
    Owned(Vec<F>),
}

impl<'a, F> core::ops::Deref for LdeValues<'a, F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub lde_values: Option<Vec<F>>,
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
            lde_values: None,
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
    pub fn get_lde_values(&self, index: usize, step: usize) -> LdeValues<'_, F> {
        let index = index * step;
        if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            let leaf_length = lde_values.len() / leaf_count;
            let value_length = leaf_length - if self.blinding { SALT_SIZE } else { 0 };
            LdeValues::Owned(
                (0..value_length)
                    .map(|poly_index| lde_values[poly_index * leaf_count + index])
                    .collect(),
            )
        } else {
            let index = reverse_bits(index, self.degree_log + self.rate_bits);
            let slice = &self.merkle_tree.leaves[index];
            LdeValues::Borrowed(&slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }])
        }
    }

    pub fn has_poly_major_lde_values(&self) -> bool {
        self.lde_values.is_some()
    }

    pub fn get_lde_values_range(
        &self,
        index: usize,
        step: usize,
        range: impl core::ops::RangeBounds<usize>,
    ) -> LdeValues<'_, F> {
        let index = index * step;
        let leaf_length = if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            lde_values.len() / leaf_count
        } else {
            let leaf_index = reverse_bits(index, self.degree_log + self.rate_bits);
            self.merkle_tree.leaves[leaf_index].len()
        };
        let start = match range.start_bound() {
            core::ops::Bound::Included(&start) => start,
            core::ops::Bound::Excluded(&start) => start + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(&end) => end + 1,
            core::ops::Bound::Excluded(&end) => end,
            core::ops::Bound::Unbounded => leaf_length - if self.blinding { SALT_SIZE } else { 0 },
        };
        if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            LdeValues::Owned(
                (start..end)
                    .map(|poly_index| lde_values[poly_index * leaf_count + index])
                    .collect(),
            )
        } else {
            let index = reverse_bits(index, self.degree_log + self.rate_bits);
            let slice = &self.merkle_tree.leaves[index];
            LdeValues::Borrowed(&slice[start..end])
        }
    }

    pub fn for_each_lde_values_range(
        &self,
        index: usize,
        step: usize,
        range: impl core::ops::RangeBounds<usize>,
        mut f: impl FnMut(usize, F),
    ) {
        let index = index * step;
        let leaf_length = if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            lde_values.len() / leaf_count
        } else {
            let leaf_index = reverse_bits(index, self.degree_log + self.rate_bits);
            self.merkle_tree.leaves[leaf_index].len()
        };
        let start = match range.start_bound() {
            core::ops::Bound::Included(&start) => start,
            core::ops::Bound::Excluded(&start) => start + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(&end) => end + 1,
            core::ops::Bound::Excluded(&end) => end,
            core::ops::Bound::Unbounded => leaf_length - if self.blinding { SALT_SIZE } else { 0 },
        };
        if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            for (j, poly_index) in (start..end).enumerate() {
                f(j, lde_values[poly_index * leaf_count + index]);
            }
        } else {
            let index = reverse_bits(index, self.degree_log + self.rate_bits);
            let slice = &self.merkle_tree.leaves[index];
            for (j, &value) in slice[start..end].iter().enumerate() {
                f(j, value);
            }
        }
    }

    pub fn get_leaf_values(&self, index: usize) -> Vec<F> {
        if let Some(lde_values) = &self.lde_values {
            let leaf_count = 1 << (self.degree_log + self.rate_bits);
            let leaf_length = lde_values.len() / leaf_count;
            let point_index = reverse_bits(index, self.degree_log + self.rate_bits);
            (0..leaf_length)
                .map(|poly_index| lde_values[poly_index * leaf_count + point_index])
                .collect()
        } else {
            self.merkle_tree.get(index).to_vec()
        }
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
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

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

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let initial_oracles = oracles
            .iter()
            .map(|&c| c as &dyn FriInitialTree<F, C::Hasher>)
            .collect::<Vec<_>>();

        let fri_proof = fri_proof::<F, C, D>(
            &initial_oracles,
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
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    FriInitialTree<F, C::Hasher> for PolynomialBatch<F, C, D>
{
    fn leaf(&self, index: usize) -> Vec<F> {
        self.get_leaf_values(index)
    }

    fn prove(&self, index: usize) -> crate::hash::merkle_proofs::MerkleProof<F, C::Hasher> {
        self.merkle_tree.prove(index)
    }
}
