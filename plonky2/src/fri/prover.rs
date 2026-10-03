#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::any::TypeId;

use plonky2_field::packable::Packable;
use plonky2_field::packed::PackedField;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::{flatten, unflatten, Extendable};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::MerkleTree;
use crate::hash::poseidon::{PoseidonHash, SPONGE_WIDTH};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::reverse_index_bits_in_place;
use crate::util::timing::TimingTree;

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    fri_proof_with::<F, C, D>(
        initial_merkle_trees,
        lde_polynomial_coeffs,
        FriValues::Natural(lde_polynomial_values),
        challenger,
        fri_params,
        final_poly_coeff_len,
        max_num_query_steps,
        timing,
    )
}

/// The polynomial's evaluations on the large domain, as `fri_proof_with` receives them.
pub(crate) enum FriValues<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// Natural order.
    Natural(PolynomialValues<F::Extension>),
    /// Index bits reversed (the first commit-phase layer's order), with that layer's Merkle tree
    /// when it is already built (the values may then be empty: the commit phase does not read
    /// them).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    LeafOrder(Vec<F::Extension>, Option<MerkleTree<F, H>>),
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> FriValues<F, H, D> {
    /// The number of values, or `None` when they were not materialised (the first tree is built).
    fn len(&self) -> Option<usize> {
        match self {
            Self::Natural(v) => Some(v.len()),
            Self::LeafOrder(v, Some(_)) if v.is_empty() => None,
            Self::LeafOrder(v, _) => Some(v.len()),
        }
    }
}

/// `fri_proof` with the evaluations in either order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fri_proof_with<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    lde_polynomial_values: FriValues<F, C::Hasher, D>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    let n = lde_polynomial_coeffs.len();
    if let Some(len) = lde_polynomial_values.len() {
        assert_eq!(len, n);
    }

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        crate::util::stage_log::acc("fri_commit_phase", || fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
        ))
    );

    // PoW phase
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        crate::util::stage_log::acc("fri_pow", || fri_proof_of_work::<F, C, D>(
            challenger,
            &fri_params.config
        ))
    );

    // Query phase
    let query_round_proofs = crate::util::stage_log::acc("fri_queries", || {
        fri_prover_query_rounds::<F, C, D>(initial_merkle_trees, &trees, challenger, n, fri_params)
    });

    FriProof {
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    }
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

pub fn final_poly_coeff_len(mut degree_bits: usize, reduction_arity_bits: &Vec<usize>) -> usize {
    for arity_bits in reduction_arity_bits {
        degree_bits -= *arity_bits;
    }
    1 << degree_bits
}

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    values: FriValues<F, C::Hasher, D>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    // The first layer's values may arrive in leaf order already (Some), with its tree (Some(Some)).
    let (mut values, mut first) = match values {
        FriValues::Natural(v) => (v, None),
        FriValues::LeafOrder(v, tree) => (PolynomialValues::new(v), Some(tree)),
    };
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        let tree = match first.take() {
            Some(Some(tree)) => tree,
            leaf_order => {
                if leaf_order.is_none() {
                    reverse_index_bits_in_place(&mut values.values);
                }
                let chunked_values = values
                    .values
                    .par_chunks(arity)
                    .map(|chunk: &[F::Extension]| flatten(chunk))
                    .collect();
                crate::util::stage_log::acc("fri_commit_merkle", || {
                    MerkleTree::<F, C::Hasher>::new(chunked_values, fri_params.config.cap_height)
                })
            }
        };

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        coeffs = PolynomialCoeffs::new(
            coeffs
                .coeffs
                .par_chunks_exact(arity)
                .map(|chunk| reduce_with_powers(chunk, beta))
                .collect::<Vec<_>>(),
        );
        shift = shift.exp_u64(arity as u64);
        values = crate::util::stage_log::acc("fri_commit_fft", || coeffs.coset_fft(shift.into()))
    }

    // When verifying this proof in a circuit with a different number of query steps,
    // we need the challenger to stay in sync with the verifier. Therefore, the challenger
    // must observe the additional hash caps and generate dummy challenges.
    if let Some(step_count) = max_num_query_steps {
        let cap_len = (1 << fri_params.config.cap_height) * NUM_HASH_OUT_ELTS;
        let zero_cap = vec![F::ZERO; cap_len];
        for _ in fri_params.reduction_arity_bits.len()..step_count {
            challenger.observe_elements(&zero_cap);
            challenger.get_extension_challenge::<D>();
        }
    }

    // The coefficients being removed here should always be zero.
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);

    challenger.observe_extension_elements(&coeffs.coeffs);
    // When verifying this proof in a circuit with a different final polynomial length,
    // the challenger needs to observe the full length of the final polynomial.
    if let Some(len) = final_poly_coeff_len {
        let current_len = coeffs.coeffs.len();
        for _ in current_len..len {
            challenger.observe_extension_element(&F::Extension::ZERO);
        }
    }

    (trees, coeffs)
}

/// Performs the proof-of-work (a.k.a. grinding) step of the FRI protocol. Returns the PoW witness.
pub(crate) fn fri_proof_of_work<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> F {
    let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;

    // The easiest implementation would be repeatedly clone our Challenger. With each clone, we'd
    // observe an incrementing PoW witness, then get the PoW response. If it contained sufficient
    // leading zeros, we'd end the search, and store this clone as our new challenger.
    //
    // However, performance is critical here. We want to avoid cloning Challenger, particularly
    // since it stores vectors, which means allocations. We'd like a more compact state to clone.
    //
    // We know that a duplex will be performed right after we send the PoW witness, so we can ignore
    // any output_buffer, which will be invalidated. We also know
    // input_buffer.len() < H::Permutation::WIDTH, an invariant of Challenger.
    //
    // We separate the duplex operation into two steps, one which can be performed now, and the
    // other which depends on the PoW witness candidate. The first step is the overwrite our sponge
    // state with any inputs (excluding the PoW witness candidate). The second step is to overwrite
    // one more element of our sponge state with the candidate, then apply the permutation,
    // obtaining our duplex's post-state which contains the PoW response.
    let mut duplex_intermediate_state = challenger.sponge_state;
    let witness_input_pos = challenger.input_buffer.len();
    duplex_intermediate_state.set_from_iter(challenger.input_buffer.clone(), 0);

    let first = crate::util::stage_log::pow_deterministic();
    // On the GPU (`PLONKY2_GPU_POW=1`): the smallest accepting candidate, like `find_first`.
    #[cfg(feature = "cuda")]
    let gpu = if TypeId::of::<C::Hasher>() == TypeId::of::<PoseidonHash>()
        && TypeId::of::<F>() == TypeId::of::<plonky2_field::goldilocks_field::GoldilocksField>()
    {
        let state: [u64; SPONGE_WIDTH] =
            core::array::from_fn(|i| duplex_intermediate_state.as_ref()[i].to_canonical_u64());
        crate::gpu::pow::search(
            &state,
            witness_input_pos,
            <<C::Hasher as Hasher<F>>::Permutation as PlonkyPermutation<F>>::RATE - 1,
            min_leading_zeros,
        )
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let gpu: Option<u64> = None;
    let pow_witness = if let Some(w) = gpu {
        Some(w)
    } else if <<F as Packable>::Packing as PackedField>::WIDTH > 1
        && TypeId::of::<C::Hasher>() == TypeId::of::<PoseidonHash>()
        && !crate::util::stage_log::pow_scalar()
    {
        // `PoseidonPermutation::permute` is `F::poseidon`, which `poseidon_packed` matches lane by
        // lane, so both searches accept exactly the same candidates.
        pow_search_packed::<F, <F as Packable>::Packing>(
            duplex_intermediate_state.as_ref(),
            witness_input_pos,
            <<C::Hasher as Hasher<F>>::Permutation as PlonkyPermutation<F>>::RATE,
            min_leading_zeros,
            first,
        )
    } else {
        pow_search_scalar::<F, C::Hasher>(
            duplex_intermediate_state,
            witness_input_pos,
            min_leading_zeros,
            first,
        )
    }
    .map(F::from_canonical_u64)
    .expect("Proof of work failed. This is highly unlikely!");

    // Recompute pow_response using our normal Challenger code, and make sure it matches.
    challenger.observe_element(pow_witness);
    let pow_response = challenger.get_challenge();
    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
    assert!(leading_zeros >= min_leading_zeros);
    pow_witness
}

/// Searches PoW witness candidates one permutation at a time. With `first`, returns the smallest
/// accepting candidate; otherwise any accepting candidate.
pub(crate) fn pow_search_scalar<F: RichField, H: Hasher<F>>(
    duplex_intermediate_state: H::Permutation,
    witness_input_pos: usize,
    min_leading_zeros: u32,
    first: bool,
) -> Option<u64> {
    let accept = |&candidate: &u64| {
        let mut duplex_state = duplex_intermediate_state;
        duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
        duplex_state.permute();
        let pow_response = duplex_state.squeeze().iter().last().unwrap();
        let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
        leading_zeros >= min_leading_zeros
    };
    let candidates = (0..=F::NEG_ONE.to_canonical_u64()).into_par_iter();
    if first {
        candidates.find_first(accept)
    } else {
        candidates.find_any(accept)
    }
}

/// Same search and result as `pow_search_scalar` for a `PoseidonPermutation` sponge, testing
/// `P::WIDTH` consecutive candidates per lane-parallel permutation. `state` is the duplex state
/// before the candidate is written at `witness_input_pos`; the response is the last element of
/// the `rate`-element squeeze.
fn pow_search_packed<F: RichField, P: PackedField<Scalar = F>>(
    state: &[F],
    witness_input_pos: usize,
    rate: usize,
    min_leading_zeros: u32,
    first: bool,
) -> Option<u64> {
    assert_eq!(state.len(), SPONGE_WIDTH);
    assert!(witness_input_pos < rate && rate <= SPONGE_WIDTH);
    let width = P::WIDTH as u64;
    let base: [P; SPONGE_WIDTH] = core::array::from_fn(|i| P::from(state[i]));
    // The smallest accepting candidate in batch `b` (candidates `b * width ..= b * width + width
    // - 1`), so the first accepting batch holds the smallest accepting candidate overall.
    let try_batch = |batch: u64| -> Option<u64> {
        let start = batch * width;
        let mut duplex_state = base;
        for (lane, x) in duplex_state[witness_input_pos]
            .as_slice_mut()
            .iter_mut()
            .enumerate()
        {
            *x = F::from_canonical_u64(start + lane as u64);
        }
        let out = F::poseidon_packed(duplex_state);
        out[rate - 1]
            .as_slice()
            .iter()
            .position(|r| r.to_canonical_u64().leading_zeros() >= min_leading_zeros)
            .map(|lane| start + lane as u64)
    };
    // Whole batches only: the last candidate, `num_batches * width - 1`, stays below the order,
    // like the scalar search's range. Neither search gets anywhere near it in practice.
    let num_batches = F::NEG_ONE.to_canonical_u64() / width;
    let batches = (0..num_batches).into_par_iter();
    if first {
        batches.find_map_first(try_batch)
    } else {
        batches.find_map_any(try_batch)
    }
}

fn fri_prover_query_rounds<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    challenger: &mut Challenger<F, C::Hasher>,
    n: usize,
    fri_params: &FriParams,
) -> Vec<FriQueryRound<F, C::Hasher, D>> {
    challenger
        .get_n_challenges(fri_params.config.num_query_rounds)
        .into_par_iter()
        .map(|rand| {
            let x_index = rand.to_canonical_u64() as usize % n;
            fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
        })
        .collect()
}

fn fri_prover_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    mut x_index: usize,
    fri_params: &FriParams,
) -> FriQueryRound<F, C::Hasher, D> {
    let mut query_steps = Vec::new();
    let initial_proof = initial_merkle_trees
        .iter()
        .map(|t| (t.get(x_index).to_vec(), t.prove(x_index)))
        .collect::<Vec<_>>();
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i];
        let evals = unflatten(tree.get(x_index >> arity_bits));
        let merkle_proof = tree.prove(x_index >> arity_bits);

        query_steps.push(FriQueryStep {
            evals,
            merkle_proof,
        });

        x_index >>= arity_bits;
    }
    FriQueryRound {
        initial_trees_proof: FriInitialTreeProof {
            evals_proofs: initial_proof,
        },
        steps: query_steps,
    }
}

#[cfg(test)]
mod tests {
    use plonky2_field::packable::Packable;
    use plonky2_field::packed::PackedField;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::{pow_search_packed, pow_search_scalar};
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, PrimeField64};
    use crate::hash::hashing::PlonkyPermutation;
    use crate::hash::poseidon::{PoseidonHash, PoseidonPermutation, SPONGE_RATE, SPONGE_WIDTH};
    use crate::hash::poseidon_goldilocks::packed_tests::ArrayPacked;

    fn random_state(rng: &mut ChaCha8Rng) -> [F; SPONGE_WIDTH] {
        core::array::from_fn(|_| F::from_noncanonical_u64(rng.gen()))
    }

    fn accepts(
        state: &[F; SPONGE_WIDTH],
        pos: usize,
        candidate: u64,
        min_leading_zeros: u32,
    ) -> bool {
        let mut perm = PoseidonPermutation::new(state.iter().copied());
        perm.set_elt(F::from_canonical_u64(candidate), pos);
        perm.permute();
        perm.squeeze()[SPONGE_RATE - 1]
            .to_canonical_u64()
            .leading_zeros()
            >= min_leading_zeros
    }

    fn check_matches_scalar<P: PackedField<Scalar = F>>() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9047_0001);
        for case in 0..24 {
            let state = random_state(&mut rng);
            let pos = case % SPONGE_RATE;
            let bits = [1, 4, 7, 9][case % 4];
            let perm = PoseidonPermutation::new(state.iter().copied());
            let scalar = pow_search_scalar::<F, PoseidonHash>(perm, pos, bits, true).unwrap();
            let packed = pow_search_packed::<F, P>(&state, pos, SPONGE_RATE, bits, true).unwrap();
            assert_eq!(
                packed,
                scalar,
                "width {} case {case}: first accepting candidate",
                P::WIDTH
            );
            let any = pow_search_packed::<F, P>(&state, pos, SPONGE_RATE, bits, false).unwrap();
            assert!(
                accepts(&state, pos, any, bits),
                "width {} case {case}: any",
                P::WIDTH
            );
        }
    }

    #[test]
    fn pow_search_packed_matches_scalar_native_width() {
        check_matches_scalar::<<F as Packable>::Packing>();
    }

    #[test]
    fn pow_search_packed_matches_scalar_width_8() {
        check_matches_scalar::<ArrayPacked<8>>();
    }

    #[test]
    fn pow_search_packed_matches_scalar_width_2() {
        check_matches_scalar::<ArrayPacked<2>>();
    }

    /// Single-thread cost per search: `RAYON_NUM_THREADS=1 cargo test --release -p plonky2 --lib
    /// pow_search_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn pow_search_speed() {
        use std::time::Instant;
        let mut rng = ChaCha8Rng::seed_from_u64(0x9047_0002);
        let (mut t_scalar, mut t_packed) = (0.0, 0.0);
        for _ in 0..32 {
            let state = random_state(&mut rng);
            let perm = PoseidonPermutation::new(state.iter().copied());
            let t = Instant::now();
            let a = pow_search_scalar::<F, PoseidonHash>(perm, 3, 16, true).unwrap();
            t_scalar += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b =
                pow_search_packed::<F, <F as Packable>::Packing>(&state, 3, SPONGE_RATE, 16, true)
                    .unwrap();
            t_packed += t.elapsed().as_secs_f64();
            assert_eq!(a, b);
        }
        println!(
            "pow_bits 16, 32 searches (find_first): scalar {:.1} ms, packed (width {}) {:.1} ms",
            1e3 * t_scalar,
            <<F as Packable>::Packing as PackedField>::WIDTH,
            1e3 * t_packed
        );
        for w in [8usize, 16] {
            let mut rng = ChaCha8Rng::seed_from_u64(0x9047_0002);
            let t = Instant::now();
            for _ in 0..32 {
                let state = random_state(&mut rng);
                if w == 8 {
                    pow_search_packed::<F, ArrayPacked<8>>(&state, 3, SPONGE_RATE, 16, true)
                        .unwrap();
                } else {
                    pow_search_packed::<F, ArrayPacked<16>>(&state, 3, SPONGE_RATE, 16, true)
                        .unwrap();
                }
            }
            println!(
                "  portable ArrayPacked<{w}>: {:.1} ms",
                1e3 * t.elapsed().as_secs_f64()
            );
        }
    }
}
