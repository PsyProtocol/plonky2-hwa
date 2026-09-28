//! plonky2 verifier implementation.

use anyhow::{ensure, Result};
use plonky2_field::extension::flatten;
use plonky2_util::log2_strict;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::fri::proof::{FriChallenges, FriInitialTreeProof, FriProof, FriQueryRound};
use crate::fri::structure::{FriInstanceInfo, FriOpenings};
use crate::fri::validate_shape::validate_fri_proof_shape;
use crate::fri::verifier::{
    compute_evaluation, fri_combine_initial, fri_verify_proof_of_work, PrecomputedReducedOpenings,
};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierOnlyCircuitData};
use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher};
use crate::plonk::plonk_common::reduce_with_powers;
use crate::plonk::proof::{Proof, ProofChallenges, ProofWithPublicInputs};
use crate::plonk::validate_shape::validate_proof_with_pis_shape;
use crate::plonk::vanishing_poly::eval_vanishing_poly;
use crate::plonk::vars::EvaluationVars;
use crate::util::reverse_bits;

pub fn verify_proof_borrowed<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    proof_with_pis: &ProofWithPublicInputs<F, C, D>,
    verifier_data: &VerifierOnlyCircuitData<C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Result<()> {
    validate_proof_with_pis_shape(&proof_with_pis, common_data)?;

    let public_inputs_hash = proof_with_pis.get_public_inputs_hash();
    let challenges = proof_with_pis.get_challenges(
        public_inputs_hash,
        &verifier_data.circuit_digest,
        common_data,
    )?;

    verify_with_challenges_borrowed_proof::<F, C, D>(
        &proof_with_pis.proof,
        public_inputs_hash,
        challenges,
        verifier_data,
        common_data,
    )
}

fn verify_with_challenges_borrowed_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    proof: &Proof<F, C, D>,
    public_inputs_hash: <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    challenges: ProofChallenges<F, D>,
    verifier_data: &VerifierOnlyCircuitData<C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Result<()> {
    let local_constants = &proof.openings.constants;
    let local_wires = &proof.openings.wires;
    let vars = EvaluationVars {
        local_constants,
        local_wires,
        public_inputs_hash: &public_inputs_hash,
    };
    let local_zs = &proof.openings.plonk_zs;
    let next_zs = &proof.openings.plonk_zs_next;
    let local_lookup_zs = &proof.openings.lookup_zs;
    let next_lookup_zs = &proof.openings.lookup_zs_next;
    let s_sigmas = &proof.openings.plonk_sigmas;
    let partial_products = &proof.openings.partial_products;

    // Evaluate the vanishing polynomial at our challenge point, zeta.
    let vanishing_polys_zeta = eval_vanishing_poly::<F, D>(
        common_data,
        challenges.plonk_zeta,
        vars,
        local_zs,
        next_zs,
        local_lookup_zs,
        next_lookup_zs,
        partial_products,
        s_sigmas,
        &challenges.plonk_betas,
        &challenges.plonk_gammas,
        &challenges.plonk_alphas,
        &challenges.plonk_deltas,
    );

    // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
    let quotient_polys_zeta = &proof.openings.quotient_polys;
    let zeta_pow_deg = challenges
        .plonk_zeta
        .exp_power_of_2(common_data.degree_bits());
    let z_h_zeta = zeta_pow_deg - F::Extension::ONE;
    // `quotient_polys_zeta` holds `num_challenges * quotient_degree_factor` evaluations.
    // Each chunk of `quotient_degree_factor` holds the evaluations of `t_0(zeta),...,t_{quotient_degree_factor-1}(zeta)`
    // where the "real" quotient polynomial is `t(X) = t_0(X) + t_1(X)*X^n + t_2(X)*X^{2n} + ...`.
    // So to reconstruct `t(zeta)` we can compute `reduce_with_powers(chunk, zeta^n)` for each
    // `quotient_degree_factor`-sized chunk of the original evaluations.
    for (i, chunk) in quotient_polys_zeta
        .chunks(common_data.quotient_degree_factor)
        .enumerate()
    {
        ensure!(vanishing_polys_zeta[i] == z_h_zeta * reduce_with_powers(chunk, zeta_pow_deg));
    }

    let merkle_caps = &[
        &verifier_data.constants_sigmas_cap,
        &proof.wires_cap,
        // In the lookup case, `plonk_zs_partial_products_cap` should also include the lookup commitment.
        &proof.plonk_zs_partial_products_cap,
        &proof.quotient_polys_cap,
    ];

    verify_fri_proof::<F, C, D>(
        &common_data.get_fri_instance(challenges.plonk_zeta),
        &proof.openings.to_fri_openings(),
        &challenges.fri_challenges,
        merkle_caps,
        &proof.opening_proof,
        &common_data.fri_params,
    )?;

    Ok(())
}

/// Verifies that the given leaf data is present at the given index in the Merkle tree with the
/// given cap.
fn verify_merkle_proof_to_cap<F: RichField, H: Hasher<F>>(
    leaf_data: &[F],
    leaf_index: usize,
    merkle_cap: &MerkleCap<F, H>,
    proof: &MerkleProof<F, H>,
) -> Result<()> {
    verify_batch_merkle_proof_to_cap(
        &[leaf_data],
        &[proof.siblings.len()],
        leaf_index,
        merkle_cap,
        proof,
    )
}

/// Verifies that the given leaf data is present at the given index in the Field Merkle tree with the
/// given cap.
fn verify_batch_merkle_proof_to_cap<F: RichField, H: Hasher<F>>(
    leaf_data: &[&[F]],
    leaf_heights: &[usize],
    mut leaf_index: usize,
    merkle_cap: &MerkleCap<F, H>,
    proof: &MerkleProof<F, H>,
) -> Result<()> {
    assert_eq!(leaf_data.len(), leaf_heights.len());
    let mut current_digest = H::hash_or_noop(&leaf_data[0]);
    let mut current_height = leaf_heights[0];
    let mut leaf_data_index = 1;
    for &sibling_digest in &proof.siblings {
        let bit = leaf_index & 1;
        leaf_index >>= 1;
        current_digest = if bit == 1 {
            H::two_to_one(sibling_digest, current_digest)
        } else {
            H::two_to_one(current_digest, sibling_digest)
        };
        current_height -= 1;

        if leaf_data_index < leaf_heights.len() && current_height == leaf_heights[leaf_data_index] {
            let mut new_leaves = current_digest.to_vec();
            new_leaves.extend_from_slice(&leaf_data[leaf_data_index]);
            current_digest = H::hash_or_noop(&new_leaves);
            leaf_data_index += 1;
        }
    }
    assert_eq!(leaf_data_index, leaf_data.len());
    ensure!(
        current_digest == merkle_cap.0[leaf_index],
        "Invalid Merkle proof."
    );

    Ok(())
}

fn fri_verify_initial_proof<F: RichField, H: Hasher<F>>(
    x_index: usize,
    proof: &FriInitialTreeProof<F, H>,
    initial_merkle_caps: &[&MerkleCap<F, H>],
) -> Result<()> {
    for ((evals, merkle_proof), cap) in proof.evals_proofs.iter().zip(initial_merkle_caps) {
        verify_merkle_proof_to_cap::<F, H>(&evals, x_index, cap, merkle_proof)?;
    }

    Ok(())
}

fn fri_verifier_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    challenges: &FriChallenges<F, D>,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    initial_merkle_caps: &[&MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    mut x_index: usize,
    n: usize,
    round_proof: &FriQueryRound<F, C::Hasher, D>,
    params: &FriParams,
) -> Result<()> {
    fri_verify_initial_proof::<F, C::Hasher>(
        x_index,
        &round_proof.initial_trees_proof,
        initial_merkle_caps,
    )?;
    // `subgroup_x` is `subgroup[x_index]`, i.e., the actual field element in the domain.
    let log_n = log2_strict(n);
    let mut subgroup_x = F::MULTIPLICATIVE_GROUP_GENERATOR
        * F::primitive_root_of_unity(log_n).exp_u64(reverse_bits(x_index, log_n) as u64);

    // old_eval is the last derived evaluation; it will be checked for consistency with its
    // committed "parent" value in the next iteration.
    let mut old_eval = fri_combine_initial::<F, C, D>(
        instance,
        &round_proof.initial_trees_proof,
        challenges.fri_alpha,
        subgroup_x,
        precomputed_reduced_evals,
        params,
    );

    for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;
        let evals = &round_proof.steps[i].evals;

        // Split x_index into the index of the coset x is in, and the index of x within that coset.
        let coset_index = x_index >> arity_bits;
        let x_index_within_coset = x_index & (arity - 1);

        // Check consistency with our old evaluation from the previous round.
        ensure!(evals[x_index_within_coset] == old_eval);

        // Infer P(y) from {P(x)}_{x^arity=y}.
        old_eval = compute_evaluation(
            subgroup_x,
            x_index_within_coset,
            arity_bits,
            evals,
            challenges.fri_betas[i],
        );

        verify_merkle_proof_to_cap::<F, C::Hasher>(
            &flatten(evals),
            coset_index,
            &proof.commit_phase_merkle_caps[i],
            &round_proof.steps[i].merkle_proof,
        )?;

        // Update the point x to x^arity.
        subgroup_x = subgroup_x.exp_power_of_2(arity_bits);

        x_index = coset_index;
    }

    // Final check of FRI. After all the reductions, we check that the final polynomial is equal
    // to the one sent by the prover.
    ensure!(
        proof.final_poly.eval(subgroup_x.into()) == old_eval,
        "Final polynomial evaluation is invalid."
    );

    Ok(())
}

fn verify_fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    instance: &FriInstanceInfo<F, D>,
    openings: &FriOpenings<F, D>,
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[&MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> Result<()> {
    validate_fri_proof_shape::<F, C, D>(proof, instance, params)?;

    // Size of the LDE domain.
    let n = params.lde_size();

    // Check PoW.
    fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)?;

    // Check that parameters are coherent.
    ensure!(
        params.config.num_query_rounds == proof.query_round_proofs.len(),
        "Number of query rounds does not match config."
    );

    let precomputed_reduced_evals =
        PrecomputedReducedOpenings::from_os_and_alpha(openings, challenges.fri_alpha);
    for (&x_index, round_proof) in challenges
        .fri_query_indices
        .iter()
        .zip(&proof.query_round_proofs)
    {
        fri_verifier_query_round::<F, C, D>(
            instance,
            challenges,
            &precomputed_reduced_evals,
            initial_merkle_caps,
            proof,
            x_index,
            n,
            round_proof,
            params,
        )?;
    }

    Ok(())
}
