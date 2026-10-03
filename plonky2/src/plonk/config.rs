//! Hashing configuration to be used when building a circuit.
//!
//! This module defines a [`Hasher`] trait as well as its recursive
//! counterpart [`AlgebraicHasher`] for in-circuit hashing. It also
//! provides concrete configurations, one fully recursive leveraging
//! the Poseidon hash function both internally and natively, and one
//! mixing Poseidon internally and truncated Keccak externally.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use plonky2_field::polynomial::PolynomialValues;
use plonky2_maybe_rayon::*;
use plonky2_util::log2_strict;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::fft::FftRootTable;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::PolynomialCoeffs;
use crate::fri::oracle::PolynomialBatch;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::keccak::KeccakHash;
use crate::hash::merkle_tree::MerkleTree;
use crate::hash::poseidon::PoseidonHash;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{reverse_index_bits_in_place, transpose};

pub trait GenericHashOut<F: RichField>:
    Copy + Clone + Debug + Eq + PartialEq + Send + Sync + Serialize + DeserializeOwned
{
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;

    fn to_vec(&self) -> Vec<F>;
}

/// Trait for hash functions.
pub trait Hasher<F: RichField>:
    'static + Sized + Copy + Debug + Eq + PartialEq + Send + Sync
{
    /// Size of `Hash` in bytes.
    const HASH_SIZE: usize;

    /// Hash Output
    type Hash: GenericHashOut<F>;

    /// Permutation used in the sponge construction.
    type Permutation: PlonkyPermutation<F>;

    /// Hash a message without any padding step. Note that this can enable length-extension attacks.
    /// However, it is still collision-resistant in cases where the input has a fixed length.
    fn hash_no_pad(input: &[F]) -> Self::Hash;

    /// Pad the message using the `pad10*1` rule, then hash it.
    fn hash_pad(input: &[F]) -> Self::Hash {
        let mut padded_input = input.to_vec();
        padded_input.push(F::ONE);
        while (padded_input.len() + 1) % Self::Permutation::RATE != 0 {
            padded_input.push(F::ZERO);
        }
        padded_input.push(F::ONE);
        Self::hash_no_pad(&padded_input)
    }

    /// Hash the slice if necessary to reduce its length to ~256 bits. If it already fits, this is a
    /// no-op.
    fn hash_or_noop(inputs: &[F]) -> Self::Hash {
        if inputs.len() * 8 <= Self::HASH_SIZE {
            let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];
            for i in 0..inputs.len() {
                inputs_bytes[i * 8..(i + 1) * 8]
                    .copy_from_slice(&inputs[i].to_canonical_u64().to_le_bytes());
            }
            Self::Hash::from_bytes(&inputs_bytes)
        } else {
            Self::hash_no_pad(inputs)
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash;

    /// Number of independent inputs that [`Hasher::hash_or_noop_batch`] and
    /// [`Hasher::two_to_one_batch`] process at once. When it is greater than 1, Merkle tree
    /// construction routes leaf and node hashing through the batch methods; otherwise it keeps the
    /// one-at-a-time path.
    const BATCH_WIDTH: usize = 1;

    /// Hashes every input with [`Hasher::hash_or_noop`].
    ///
    /// Implementations must return exactly `inputs.iter().map(|x| Self::hash_or_noop(x))`, down
    /// to the in-memory representation of every digest.
    fn hash_or_noop_batch(inputs: &[Vec<F>]) -> Vec<Self::Hash> {
        inputs.iter().map(|x| Self::hash_or_noop(x)).collect()
    }

    /// Compresses consecutive pairs: output `i` is
    /// `Self::two_to_one(inputs[2 * i], inputs[2 * i + 1])`. `inputs.len()` must be even.
    ///
    /// Implementations must match [`Hasher::two_to_one`] exactly, down to the in-memory
    /// representation of every digest.
    fn two_to_one_batch(inputs: &[Self::Hash]) -> Vec<Self::Hash> {
        assert_eq!(
            inputs.len() % 2,
            0,
            "two_to_one_batch needs an even number of inputs"
        );
        inputs
            .chunks_exact(2)
            .map(|pair| Self::two_to_one(pair[0], pair[1]))
            .collect()
    }
}

/// Trait for algebraic hash functions, built from a permutation using the sponge construction.
pub trait AlgebraicHasher<F: RichField>: 'static + Hasher<F, Hash = HashOut<F>> {
    type AlgebraicPermutation: PlonkyPermutation<Target>;

    /// Circuit to conditionally swap two chunks of the inputs (useful in verifying Merkle proofs),
    /// then apply the permutation.
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>;
}

/// Trait to abstract heavy prover computations (FFT + Merkle Hashing).
/// An external crate can implement this to provide GPU acceleration.
pub trait ProverCompute<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>:
    'static + Debug + Clone + Send + Sync
{
    /// Takes polynomial coefficients, performs ifft, then performs LDE (FFT), and commits to the Merkle tree.

    fn transpose_and_compute_from_coeffs(
        timing: &mut TimingTree,
        pre_transposed_quotient_polys: Vec<Vec<F>>,
        quotient_degree: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>>;

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>>;

    /// Takes polynomial coefficients, performs LDE (FFT), and commits to the Merkle tree.
    fn compute_from_coeffs(
        timing: &mut TimingTree,
        polynomials: Vec<PolynomialCoeffs<F>>,
        cap_height: usize,
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>>;

    fn build_merkle_tree(
        timing: &mut TimingTree,
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> anyhow::Result<MerkleTree<F, C::Hasher>>;
}

/// Default CPU implementation of `ProverCompute`.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CpuProverCompute;
impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverCompute<F, C, D> for CpuProverCompute
{
    fn compute_from_coeffs(
        timing: &mut TimingTree,
        polynomials: Vec<PolynomialCoeffs<F>>,
        cap_height: usize,
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
        //println!("CPU ProverCompute::compute_from_coeffs called");
        let commit_start = crate::util::stage_log::generators_start();
        let lg_n = polynomials.first().map_or(0, |p| log2_strict(p.len()));
        let cols = polynomials.len();
        #[cfg(feature = "cuda")]
        let cpu_reason = match crate::gpu::try_commit::<F, C::Hasher>(&polynomials, rate_bits, blinding, cap_height) {
            Ok(merkle_tree) => {
                return Ok(PolynomialBatch {
                    polynomials,
                    merkle_tree,
                    degree_log: lg_n,
                    rate_bits,
                    blinding,
                });
            }
            Err(reason) => reason,
        };
        #[cfg(not(feature = "cuda"))]
        let cpu_reason = "gpu_not_built";
        const SALT_SIZE: usize = 4;
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let degree = polynomials[0].len();
        let degree_log = log2_strict(degree);

        let lde_values: Vec<Vec<F>> = timed!(
            timing,
            "generate LDE values",
            crate::util::stage_log::acc("lde", || polynomials
                .par_iter()
                .map(|p| {
                    p.lde(rate_bits)
                        .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                        .values
                })
                .chain(
                    (0..salt_size)
                        .into_par_iter()
                        .map(|_| F::rand_vec(degree << rate_bits)),
                )
                .collect())
        );

        let leaves = crate::util::stage_log::acc("transpose_bitrev", || {
            let mut leaves = transpose(&lde_values);
            reverse_index_bits_in_place(&mut leaves);
            leaves
        });

        let merkle_tree = timed!(
            timing,
            "Merkle tree",
            crate::util::stage_log::acc("merkle", || {
                <Self as ProverCompute<F, C, D>>::build_merkle_tree(timing, leaves, cap_height)
            })?
        );
        #[cfg(feature = "cuda")]
        crate::gpu::count_cpu_commit();
        if let Some(t) = commit_start {
            let s = t.elapsed().as_secs_f64();
            crate::util::stage_log::commit_event("cpu", cpu_reason, lg_n, rate_bits, cols, 0.0, 0.0, s, 0.0, s);
        }
        Ok(PolynomialBatch {
            polynomials,
            merkle_tree,
            degree_log,
            rate_bits,
            blinding,
        })
    }

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
        //println!("CPU ProverCompute::compute_from_values called");
        #[cfg(feature = "cuda")]
        if let Ok((merkle_tree, polynomials)) =
            crate::gpu::try_commit_values::<F, C::Hasher>(&values, rate_bits, blinding, cap_height)
        {
            let degree_log = log2_strict(polynomials[0].len());
            return Ok(PolynomialBatch { polynomials, merkle_tree, degree_log, rate_bits, blinding });
        }
        let coeffs = timed!(
            timing,
            "IFFT",
            crate::util::stage_log::acc("ifft", || values
                .into_par_iter()
                .map(|v| v.ifft())
                .collect::<Vec<_>>())
        );
        Self::compute_from_coeffs(
            timing,
            coeffs,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    fn transpose_and_compute_from_coeffs(
        timing: &mut TimingTree,
        pre_transposed_quotient_polys: Vec<Vec<F>>,
        quotient_degree: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
        //println!("CPU ProverCompute::transpose_and_compute_from_coeffs called");
        let quotient_polys: Vec<PolynomialCoeffs<F>> = timed!(
            timing,
            "coset IFFT quotient polys",
            crate::util::stage_log::acc("quotient_coset_ifft", || transpose(
                &pre_transposed_quotient_polys
            )
            .into_par_iter()
            .map(PolynomialValues::new)
            .map(|values| values.coset_ifft(F::coset_shift()))
            .collect::<Vec<_>>())
        );

        let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
            timing,
            "split up quotient polys",
            quotient_polys
                .into_par_iter()
                .flat_map(|mut quotient_poly| {
                    quotient_poly.trim_to_len(quotient_degree).expect(
                        "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                    );
                    // Split quotient into degree-n chunks.
                    quotient_poly.chunks(degree)
                })
                .collect()
        );

        let quotient_polys_commitment = timed!(
            timing,
            "commit to quotient polys",
            Self::compute_from_coeffs(
                timing,
                all_quotient_poly_chunks,
                cap_height,
                rate_bits,
                blinding,
                fft_root_table,
            )?
        );
        Ok(quotient_polys_commitment)
    }

    fn build_merkle_tree(
        timing: &mut TimingTree,
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> anyhow::Result<MerkleTree<F, C::Hasher>> {
        //println!("CPU ProverCompute::build_merkle_tree called");
        let merkle_tree = timed!(timing, "Merkle tree", MerkleTree::new(leaves, cap_height));
        Ok(merkle_tree)
    }
}

/// Generic configuration trait.
pub trait GenericConfig<const D: usize>:
    'static + Debug + Clone + Sync + Sized + Send + Eq + PartialEq
{
    /// Main field.
    type F: RichField + Extendable<D, Extension = Self::FE>;
    /// Field extension of degree D of the main field.
    type FE: FieldExtension<D, BaseField = Self::F>;
    /// Hash function used for building Merkle trees.
    type Hasher: Hasher<Self::F>;
    /// Algebraic hash function used for the challenger and hashing public inputs.
    type InnerHasher: AlgebraicHasher<Self::F>;
    /// The engine used for FFTs and Merkle Trees.
    type Compute: ProverCompute<Self::F, Self, D>;
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksConfig;
impl GenericConfig<2> for PoseidonGoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
    type Compute = CpuProverCompute;
}

/// Configuration using truncated Keccak over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct KeccakGoldilocksConfig;
impl GenericConfig<2> for KeccakGoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = KeccakHash<25>;
    type InnerHasher = PoseidonHash;
    type Compute = CpuProverCompute;
}
