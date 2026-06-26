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
use core::marker::PhantomData;

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
#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::field::types::Field;
#[cfg(feature = "metal")]
use crate::field::types::Sample;
use crate::fri::oracle::PolynomialBatch;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::keccak::KeccakHash;
#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::hash::merkle_tree::MerkleCap;
use crate::hash::merkle_tree::MerkleTree;
use crate::hash::poseidon::PoseidonHash;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{reverse_index_bits_in_place, transpose};

#[cfg(feature = "metal")]
fn fill_goldilocks_rand(dst: &mut [GoldilocksField]) {
    let mut rng = rand::rngs::OsRng;
    for value in dst {
        *value = GoldilocksField::sample(&mut rng);
    }
}

#[cfg(feature = "metal")]
fn goldilocks_vec_for_overwrite(len: usize) -> Vec<GoldilocksField> {
    let mut values = Vec::with_capacity(len);
    // SAFETY: Each caller immediately overwrites the whole slice before it is read. This helper is
    // only sound for full-slice writes.
    unsafe {
        values.set_len(len);
    }
    values
}

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

fn compute_from_coeffs_with_prover_compute<
    PC: ProverCompute<F, C, D>,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    timing: &mut TimingTree,
    polynomials: Vec<PolynomialCoeffs<F>>,
    cap_height: usize,
    rate_bits: usize,
    blinding: bool,
    fft_root_table: Option<&FftRootTable<F>>,
) -> anyhow::Result<PolynomialBatch<F, C, D>> {
    const SALT_SIZE: usize = 4;
    let salt_size = if blinding { SALT_SIZE } else { 0 };

    let degree = polynomials[0].len();
    let degree_log = log2_strict(degree);

    let lde_values: Vec<Vec<F>> = timed!(
        timing,
        "generate LDE values",
        polynomials
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
            .collect()
    );

    let mut leaves = transpose(&lde_values);
    reverse_index_bits_in_place(&mut leaves);

    let merkle_tree = timed!(
        timing,
        "Merkle tree",
        PC::build_merkle_tree(timing, leaves, cap_height)?
    );
    Ok(PolynomialBatch {
        polynomials,
        merkle_tree,
        lde_values: None,
        degree_log,
        rate_bits,
        blinding,
    })
}

fn compute_from_values_with_prover_compute<
    PC: ProverCompute<F, C, D>,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    timing: &mut TimingTree,
    values: Vec<PolynomialValues<F>>,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> anyhow::Result<PolynomialBatch<F, C, D>> {
    let coeffs = timed!(
        timing,
        "IFFT",
        values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
    );
    PC::compute_from_coeffs(
        timing,
        coeffs,
        cap_height,
        rate_bits,
        blinding,
        fft_root_table,
    )
}

fn transpose_and_compute_from_coeffs_with_prover_compute<
    PC: ProverCompute<F, C, D>,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    timing: &mut TimingTree,
    pre_transposed_quotient_polys: Vec<Vec<F>>,
    quotient_degree: usize,
    degree: usize,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> anyhow::Result<PolynomialBatch<F, C, D>> {
    let quotient_polys: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "coset IFFT quotient polys",
        transpose(&pre_transposed_quotient_polys)
            .into_par_iter()
            .map(PolynomialValues::new)
            .map(|values| values.coset_ifft(F::coset_shift()))
            .collect::<Vec<_>>()
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
        PC::compute_from_coeffs(
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
        compute_from_coeffs_with_prover_compute::<Self, F, C, D>(
            timing,
            polynomials,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
        compute_from_values_with_prover_compute::<Self, F, C, D>(
            timing,
            values,
            rate_bits,
            blinding,
            cap_height,
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
        transpose_and_compute_from_coeffs_with_prover_compute::<Self, F, C, D>(
            timing,
            pre_transposed_quotient_polys,
            quotient_degree,
            degree,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
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

/// Marker for the placeholder CUDA backend.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct CudaBackend;

#[cfg(any(feature = "metal", feature = "cuda"))]
fn get_node_hash_index_in_digests(
    num_layers: usize,
    tree_length: usize,
    level: usize,
    index: usize,
) -> usize {
    let leaf_index = index << level;
    let tree_index = index >> (num_layers - level);
    let pair_index = ((leaf_index) & ((1 << num_layers) - 1)) >> level;
    let parity = pair_index & 1;
    let pair_index = pair_index >> 1;

    let siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    let d_index = 2 * siblings_index + parity;
    d_index + (tree_index * tree_length)
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn gpu_merkle_tree_from_digests(
    leaves: Vec<Vec<GoldilocksField>>,
    leaf_count: usize,
    digests: Vec<HashOut<GoldilocksField>>,
    cap_height: usize,
) -> MerkleTree<GoldilocksField, PoseidonHash> {
    let tree_height = log2_strict(leaf_count);
    let num_caps = 1usize << cap_height;
    let total_tree_hashes = leaf_count * 2 - 1;
    let total_digests = total_tree_hashes - (num_caps * 2 - 1);
    let num_layers = tree_height - cap_height;
    let tree_length = total_digests >> cap_height;

    let mut caps = Vec::with_capacity(num_caps);
    let cap_child_level = tree_height - cap_height - 1;
    for i in 0..num_caps {
        let left_index =
            get_node_hash_index_in_digests(num_layers, tree_length, cap_child_level, i * 2);
        caps.push(<PoseidonHash as Hasher<GoldilocksField>>::two_to_one(
            digests[left_index],
            digests[left_index + 1],
        ));
    }

    MerkleTree {
        leaves,
        digests,
        cap: MerkleCap(caps),
    }
}

/// Metal-backed prover compute implementation.
///
/// Currently this accelerates Poseidon Goldilocks Merkle tree construction for large trees and
/// delegates FFTs and small trees to the CPU path.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct MetalProverCompute;

impl<C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash>>
    ProverCompute<GoldilocksField, C, 2> for MetalProverCompute
{
    fn compute_from_coeffs(
        timing: &mut TimingTree,
        polynomials: Vec<PolynomialCoeffs<GoldilocksField>>,
        cap_height: usize,
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        #[cfg(feature = "metal")]
        {
            const SALT_SIZE: usize = 4;
            let salt_size = if blinding { SALT_SIZE } else { 0 };
            let degree = polynomials[0].len();
            let degree_log = log2_strict(degree);
            let leaf_count = degree << rate_bits;
            let leaf_length = polynomials.len() + salt_size;
            let tree_height = log2_strict(leaf_count);

            if cap_height < tree_height && tree_height >= 13 && plonky2_metal::is_available() {
                let lde_values = timed!(timing, "generate LDE values", {
                    if plonky2_metal::lde_enabled() {
                        let mut coeffs = goldilocks_vec_for_overwrite(polynomials.len() * degree);
                        coeffs
                            .par_chunks_mut(degree)
                            .zip(polynomials.par_iter())
                            .for_each(|(dst, polynomial)| {
                                dst.copy_from_slice(&polynomial.coeffs);
                            });
                        let metal_values = plonky2_metal::MetalRuntime::get()
                            .coset_lde_values_poly_major(
                                &coeffs,
                                polynomials.len(),
                                degree,
                                rate_bits,
                            );
                        let mut lde_values = metal_values;
                        let total_len = leaf_count * leaf_length;
                        if lde_values.len() < total_len {
                            lde_values.reserve(total_len - lde_values.len());
                            // SAFETY: The added salt chunks are fully overwritten below.
                            unsafe {
                                lde_values.set_len(total_len);
                            }
                        }
                        lde_values
                            .par_chunks_mut(leaf_count)
                            .skip(polynomials.len())
                            .for_each(fill_goldilocks_rand);
                        lde_values
                    } else {
                        let mut lde_values = goldilocks_vec_for_overwrite(leaf_count * leaf_length);
                        lde_values
                            .par_chunks_mut(leaf_count)
                            .enumerate()
                            .for_each(|(i, dst)| {
                                if i < polynomials.len() {
                                    let values = polynomials[i]
                                        .lde(rate_bits)
                                        .coset_fft_with_options(
                                            GoldilocksField::coset_shift(),
                                            Some(rate_bits),
                                            fft_root_table,
                                        )
                                        .values;
                                    dst.copy_from_slice(&values);
                                } else {
                                    fill_goldilocks_rand(dst);
                                }
                            });
                        lde_values
                    }
                });

                let digests = timed!(
                    timing,
                    "Merkle tree",
                    plonky2_metal::MetalRuntime::hash_merkle_tree_poly_major_digests(
                        &lde_values,
                        leaf_count,
                        leaf_length,
                        cap_height,
                    )
                    .into_iter()
                    .map(HashOut::from)
                    .collect::<Vec<_>>()
                );
                let merkle_tree =
                    gpu_merkle_tree_from_digests(Vec::new(), leaf_count, digests, cap_height);

                return Ok(PolynomialBatch {
                    polynomials,
                    merkle_tree,
                    lde_values: Some(lde_values),
                    degree_log,
                    rate_bits,
                    blinding,
                });
            }
        }

        compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            polynomials,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<GoldilocksField>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        compute_from_values_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            values,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn transpose_and_compute_from_coeffs(
        timing: &mut TimingTree,
        pre_transposed_quotient_polys: Vec<Vec<GoldilocksField>>,
        quotient_degree: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        transpose_and_compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            pre_transposed_quotient_polys,
            quotient_degree,
            degree,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn build_merkle_tree(
        timing: &mut TimingTree,
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> anyhow::Result<MerkleTree<GoldilocksField, C::Hasher>> {
        let tree_height = log2_strict(leaves.len());
        if cap_height == tree_height || tree_height < 13 {
            return <CpuProverCompute as ProverCompute<GoldilocksField, C, 2>>::build_merkle_tree(
                timing, leaves, cap_height,
            );
        }

        #[cfg(feature = "metal")]
        {
            if plonky2_metal::is_available() {
                let digests =
                    plonky2_metal::MetalRuntime::hash_merkle_tree_digests(&leaves, cap_height)
                        .into_iter()
                        .map(HashOut::from)
                        .collect::<Vec<_>>();

                let leaf_count = leaves.len();
                return Ok(gpu_merkle_tree_from_digests(
                    leaves, leaf_count, digests, cap_height,
                ));
            }
        }

        <CpuProverCompute as ProverCompute<GoldilocksField, C, 2>>::build_merkle_tree(
            timing, leaves, cap_height,
        )
    }
}

/// Placeholder for a GPU-backed prover compute implementation.
///
/// This currently delegates to the CPU implementation. The backend marker keeps future CUDA kernel
/// wiring isolated behind this `ProverCompute` implementation.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct PlaceholderGpuProverCompute<B>(PhantomData<B>);

/// CUDA-backed prover compute.
///
/// Phase 1 accelerates the LDE + Poseidon-Merkle commitment (`compute_from_coeffs`) for large trees
/// on the GPU and delegates everything else (and the no-`cuda`-feature build) to the CPU path.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct CudaProverCompute;

/// Reinterpret a Goldilocks slice as raw `u64`s. Sound because `GoldilocksField` is
/// `#[repr(transparent)]` over `u64`.
#[cfg(feature = "cuda")]
fn cuda_as_u64(values: &[GoldilocksField]) -> &[u64] {
    unsafe { core::slice::from_raw_parts(values.as_ptr() as *const u64, values.len()) }
}

impl<C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash>>
    ProverCompute<GoldilocksField, C, 2> for CudaProverCompute
{
    fn compute_from_coeffs(
        timing: &mut TimingTree,
        polynomials: Vec<PolynomialCoeffs<GoldilocksField>>,
        cap_height: usize,
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        #[cfg(feature = "cuda")]
        {
            const SALT_SIZE: usize = 4;
            let salt_size = if blinding { SALT_SIZE } else { 0 };
            let degree = polynomials[0].len();
            let degree_log = log2_strict(degree);
            let leaf_count = degree << rate_bits;
            let tree_height = log2_strict(leaf_count);

            if cap_height < tree_height && tree_height >= 13 && plonky2_cuda::is_available() {
                let poly_num = polynomials.len();

                // Poly-major coefficients as raw u64.
                let mut coeffs = vec![0u64; poly_num * degree];
                coeffs
                    .par_chunks_mut(degree)
                    .zip(polynomials.par_iter())
                    .for_each(|(dst, poly)| dst.copy_from_slice(cuda_as_u64(&poly.coeffs)));

                // Twiddle tables / coset-shift powers (see plonky2_cuda::runtime docs).
                let root_table = crate::field::fft::fft_root_table::<GoldilocksField>(degree).concat();
                let root_table2 =
                    crate::field::fft::fft_root_table::<GoldilocksField>(leaf_count).concat();
                let shift_powers: Vec<GoldilocksField> = GoldilocksField::coset_shift()
                    .powers()
                    .take(degree)
                    .collect();

                let result = timed!(
                    timing,
                    "CUDA LDE + Merkle tree",
                    plonky2_cuda::CudaRuntime::get()
                        .commit_from_coeffs(
                            &coeffs,
                            poly_num,
                            degree,
                            rate_bits,
                            cap_height,
                            salt_size,
                            cuda_as_u64(&root_table),
                            cuda_as_u64(&root_table2),
                            cuda_as_u64(&shift_powers),
                        )
                        .map_err(|e| anyhow::anyhow!("CUDA commit_from_coeffs failed: {e}"))?
                );

                // The GPU buffer is `[num_digests internal/leaf digests | len_cap caps]`; plonky2's
                // `merkle_tree.digests` holds only the `num_digests` part (caps are rebuilt below).
                let len_cap = 1usize << cap_height;
                let num_digests = 2 * (leaf_count - len_cap);
                let digests = result
                    .digests_and_caps
                    .into_iter()
                    .take(num_digests)
                    .map(|c| HashOut {
                        elements: [
                            GoldilocksField(c[0]),
                            GoldilocksField(c[1]),
                            GoldilocksField(c[2]),
                            GoldilocksField(c[3]),
                        ],
                    })
                    .collect::<Vec<_>>();
                // The GPU returns leaf-major values in the CPU Merkle tree's leaf order, so populate
                // `merkle_tree.leaves` directly (leaf length = polys + salt) and leave
                // `lde_values` empty.
                let leaf_len = poly_num + salt_size;
                let leaves = result
                    .leaves_flat
                    .chunks_exact(leaf_len)
                    .map(|leaf| leaf.iter().map(|&v| GoldilocksField(v)).collect())
                    .collect::<Vec<_>>();
                let merkle_tree =
                    gpu_merkle_tree_from_digests(leaves, leaf_count, digests, cap_height);

                return Ok(PolynomialBatch {
                    polynomials,
                    merkle_tree,
                    lde_values: None,
                    degree_log,
                    rate_bits,
                    blinding,
                });
            }
        }

        compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            polynomials,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<GoldilocksField>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        compute_from_values_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            values,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn transpose_and_compute_from_coeffs(
        timing: &mut TimingTree,
        pre_transposed_quotient_polys: Vec<Vec<GoldilocksField>>,
        quotient_degree: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        transpose_and_compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            pre_transposed_quotient_polys,
            quotient_degree,
            degree,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn build_merkle_tree(
        timing: &mut TimingTree,
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> anyhow::Result<MerkleTree<GoldilocksField, C::Hasher>> {
        <CpuProverCompute as ProverCompute<GoldilocksField, C, 2>>::build_merkle_tree(
            timing, leaves, cap_height,
        )
    }
}

impl<
        B: 'static + Debug + Clone + Send + Sync,
        C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash>,
    > ProverCompute<GoldilocksField, C, 2> for PlaceholderGpuProverCompute<B>
{
    fn compute_from_coeffs(
        timing: &mut TimingTree,
        polynomials: Vec<PolynomialCoeffs<GoldilocksField>>,
        cap_height: usize,
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            polynomials,
            cap_height,
            rate_bits,
            blinding,
            fft_root_table,
        )
    }

    fn compute_from_values(
        timing: &mut TimingTree,
        values: Vec<PolynomialValues<GoldilocksField>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        compute_from_values_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            values,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn transpose_and_compute_from_coeffs(
        timing: &mut TimingTree,
        pre_transposed_quotient_polys: Vec<Vec<GoldilocksField>>,
        quotient_degree: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        fft_root_table: Option<&FftRootTable<GoldilocksField>>,
    ) -> anyhow::Result<PolynomialBatch<GoldilocksField, C, 2>> {
        transpose_and_compute_from_coeffs_with_prover_compute::<Self, GoldilocksField, C, 2>(
            timing,
            pre_transposed_quotient_polys,
            quotient_degree,
            degree,
            rate_bits,
            blinding,
            cap_height,
            fft_root_table,
        )
    }

    fn build_merkle_tree(
        timing: &mut TimingTree,
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> anyhow::Result<MerkleTree<GoldilocksField, C::Hasher>> {
        <CpuProverCompute as ProverCompute<GoldilocksField, C, 2>>::build_merkle_tree(
            timing, leaves, cap_height,
        )
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

/// Configuration using Poseidon over the Goldilocks field with a Metal compute backend.
///
/// The backend is a CPU fallback placeholder until Metal kernels are wired in.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksMetalConfig;
impl GenericConfig<2> for PoseidonGoldilocksMetalConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
    type Compute = MetalProverCompute;
}

/// Configuration using Poseidon over the Goldilocks field with a CUDA compute backend.
///
/// The backend is a CPU fallback placeholder until CUDA kernels are wired in.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksCudaConfig;
impl GenericConfig<2> for PoseidonGoldilocksCudaConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
    type Compute = CudaProverCompute;
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

#[cfg(test)]
mod tests {
    use log::Level;

    use super::*;
    use crate::field::types::Field;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;

    fn test_leaves() -> Vec<Vec<GoldilocksField>> {
        (0..8)
            .map(|row| {
                (0..6)
                    .map(|col| GoldilocksField::from_canonical_usize(row * 17 + col * 3))
                    .collect()
            })
            .collect()
    }

    fn assert_backend_merkle_matches_cpu<C>()
    where
        C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash>,
    {
        let leaves = test_leaves();
        let expected = MerkleTree::<GoldilocksField, PoseidonHash>::new(leaves.clone(), 2);
        let mut timing = TimingTree::new("test backend merkle", Level::Debug);
        let actual = <C::Compute as ProverCompute<GoldilocksField, C, 2>>::build_merkle_tree(
            &mut timing,
            leaves,
            2,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    fn prove_small_addition<C>()
    where
        C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash, InnerHasher = PoseidonHash>,
    {
        let mut builder =
            CircuitBuilder::<GoldilocksField, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let sum = builder.add(a, b);
        builder.register_public_input(sum);

        let mut pw = PartialWitness::new();
        pw.set_target(a, GoldilocksField::from_canonical_u64(3))
            .unwrap();
        pw.set_target(b, GoldilocksField::from_canonical_u64(4))
            .unwrap();

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        assert_eq!(
            proof.public_inputs,
            vec![GoldilocksField::from_canonical_u64(7)]
        );
        data.verify(proof).unwrap();
    }

    #[test]
    fn metal_merkle_backend_matches_cpu_for_small_tree() {
        assert_backend_merkle_matches_cpu::<PoseidonGoldilocksMetalConfig>();
    }

    #[test]
    fn cuda_merkle_backend_matches_cpu_for_small_tree() {
        assert_backend_merkle_matches_cpu::<PoseidonGoldilocksCudaConfig>();
    }

    #[test]
    fn metal_config_proves_and_verifies_small_circuit() {
        prove_small_addition::<PoseidonGoldilocksMetalConfig>();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_compute_from_coeffs_matches_cpu_for_large_tree() {
        if !plonky2_metal::is_available() {
            eprintln!("skipping Metal compute_from_coeffs test: no Metal device available");
            return;
        }

        let polynomials = (0..6)
            .map(|poly| {
                PolynomialCoeffs::new(
                    (0..1024)
                        .map(|i| GoldilocksField::from_canonical_usize(poly * 31 + i * 17 + 5))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let mut cpu_timing = TimingTree::new("cpu", Level::Debug);
        let mut metal_timing = TimingTree::new("metal", Level::Debug);
        let cpu = PolynomialBatch::<GoldilocksField, PoseidonGoldilocksConfig, 2>::from_coeffs(
            polynomials.clone(),
            3,
            false,
            4,
            &mut cpu_timing,
            None,
        )
        .unwrap();
        let metal =
            PolynomialBatch::<GoldilocksField, PoseidonGoldilocksMetalConfig, 2>::from_coeffs(
                polynomials,
                3,
                false,
                4,
                &mut metal_timing,
                None,
            )
            .unwrap();

        assert!(metal.merkle_tree.leaves.is_empty());
        assert_eq!(metal.merkle_tree.digests, cpu.merkle_tree.digests);
        assert_eq!(metal.merkle_tree.cap, cpu.merkle_tree.cap);
        for &index in &[0, 1, 17, 4096, 8191] {
            assert_eq!(metal.get_leaf_values(index), cpu.get_leaf_values(index));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_compute_from_coeffs_matches_cpu_for_large_tree() {
        if !plonky2_cuda::is_available() {
            eprintln!("skipping CUDA compute_from_coeffs test: no CUDA device available");
            return;
        }

        // degree 1024, rate_bits 3 => leaf_count 8192 (tree_height 13) so the GPU path triggers.
        let polynomials = (0..6)
            .map(|poly| {
                PolynomialCoeffs::new(
                    (0..1024)
                        .map(|i| GoldilocksField::from_canonical_usize(poly * 31 + i * 17 + 5))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let mut cpu_timing = TimingTree::new("cpu", Level::Debug);
        let mut cuda_timing = TimingTree::new("cuda", Level::Debug);
        let cpu = PolynomialBatch::<GoldilocksField, PoseidonGoldilocksConfig, 2>::from_coeffs(
            polynomials.clone(),
            3,
            false,
            4,
            &mut cpu_timing,
            None,
        )
        .unwrap();
        let cuda =
            PolynomialBatch::<GoldilocksField, PoseidonGoldilocksCudaConfig, 2>::from_coeffs(
                polynomials,
                3,
                false,
                4,
                &mut cuda_timing,
                None,
            )
            .unwrap();

        // Check from the bottom up so a failure pinpoints the layer:
        // leaves (LDE correctness) -> cap (hashing) -> full digest array (layout).
        for &index in &[0, 1, 17, 4096, 8191] {
            assert_eq!(
                cuda.get_leaf_values(index),
                cpu.get_leaf_values(index),
                "leaf {index} mismatch (LDE / root-table convention)"
            );
        }
        assert_eq!(cuda.merkle_tree.cap, cpu.merkle_tree.cap, "cap mismatch");
        assert_eq!(
            cuda.merkle_tree.digests, cpu.merkle_tree.digests,
            "digest array mismatch (GPU digest ordering vs plonky2 layout)"
        );
    }

    /// Wall-clock comparison of `from_coeffs` (the Phase 1 accelerated path) on CPU vs CUDA for a
    /// large tree. Run with: `cargo test --features cuda cuda_from_coeffs_timing -- --ignored
    /// --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn cuda_from_coeffs_timing() {
        use std::time::Instant;

        if !plonky2_cuda::is_available() {
            eprintln!("skipping CUDA timing: no CUDA device available");
            return;
        }

        let degree_bits = 15;
        let degree = 1usize << degree_bits;
        let poly_num = 88; // ~ num_constants + num_routed_wires for a recursion circuit
        let rate_bits = 3;
        let cap_height = 4;
        let polynomials = (0..poly_num)
            .map(|p| {
                PolynomialCoeffs::new(
                    (0..degree)
                        .map(|i| GoldilocksField::from_canonical_usize(p * 31 + i * 17 + 5))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();

        let run_cpu = || {
            let mut t = TimingTree::new("cpu", Level::Debug);
            PolynomialBatch::<GoldilocksField, PoseidonGoldilocksConfig, 2>::from_coeffs(
                polynomials.clone(),
                rate_bits,
                false,
                cap_height,
                &mut t,
                None,
            )
            .unwrap()
        };
        let run_cuda = || {
            let mut t = TimingTree::new("cuda", Level::Debug);
            PolynomialBatch::<GoldilocksField, PoseidonGoldilocksCudaConfig, 2>::from_coeffs(
                polynomials.clone(),
                rate_bits,
                false,
                cap_height,
                &mut t,
                None,
            )
            .unwrap()
        };

        // Warm up (CUDA context creation + JIT, allocator warmup).
        let _ = run_cpu();
        let _ = run_cuda();

        let runs = 5;
        let mut cpu_ms = vec![];
        let mut cuda_ms = vec![];
        for _ in 0..runs {
            let s = Instant::now();
            let _ = run_cpu();
            cpu_ms.push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            let _ = run_cuda();
            cuda_ms.push(s.elapsed().as_secs_f64() * 1e3);
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let c = median(cpu_ms);
        let g = median(cuda_ms);
        eprintln!(
            "[from_coeffs timing] degree=2^{degree_bits} polys={poly_num} rate_bits={rate_bits} \
             leaf_count=2^{} | CPU={c:.1}ms CUDA={g:.1}ms speedup={:.2}x",
            degree_bits + rate_bits,
            c / g,
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_lde_compute_from_coeffs_matches_cpu_for_large_tree() {
        if !plonky2_metal::is_available() {
            eprintln!("skipping Metal LDE compute_from_coeffs test: no Metal device available");
            return;
        }

        let polynomials = (0..6)
            .map(|poly| {
                PolynomialCoeffs::new(
                    (0..1024)
                        .map(|i| GoldilocksField::from_canonical_usize(poly * 31 + i * 17 + 5))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let mut cpu_timing = TimingTree::new("cpu", Level::Debug);
        let mut metal_timing = TimingTree::new("metal lde", Level::Debug);
        let cpu = PolynomialBatch::<GoldilocksField, PoseidonGoldilocksConfig, 2>::from_coeffs(
            polynomials.clone(),
            3,
            false,
            4,
            &mut cpu_timing,
            None,
        )
        .unwrap();

        plonky2_metal::set_lde_enabled(true);
        let metal =
            PolynomialBatch::<GoldilocksField, PoseidonGoldilocksMetalConfig, 2>::from_coeffs(
                polynomials,
                3,
                false,
                4,
                &mut metal_timing,
                None,
            )
            .unwrap();
        plonky2_metal::set_lde_enabled(false);

        assert!(metal.merkle_tree.leaves.is_empty());
        assert_eq!(metal.merkle_tree.digests, cpu.merkle_tree.digests);
        assert_eq!(metal.merkle_tree.cap, cpu.merkle_tree.cap);
        for &index in &[0, 1, 17, 4096, 8191] {
            assert_eq!(metal.get_leaf_values(index), cpu.get_leaf_values(index));
        }
    }

    #[test]
    fn cuda_config_proves_and_verifies_small_circuit() {
        prove_small_addition::<PoseidonGoldilocksCudaConfig>();
    }
}
