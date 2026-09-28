//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;
use core::mem::swap;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
#[cfg(feature = "metal")]
use crate::field::types::PrimeField64;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher, ProverCompute};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{eval_vanishing_poly_base_batch, get_lut_poly};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::partial_products::{partial_products_and_z_gx, quotient_chunk_products};
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil, transpose};

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)?
    );

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    set_lookup_wires(prover_data, common_data, &mut partition_witness)?;

    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);

    let witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );

    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )?
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);

    let deltas = if has_lookup {
        let mut delts = Vec::with_capacity(2 * num_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        delts.extend(&betas);
        delts.extend(&gammas);
        delts.extend(additional);
        delts
    } else {
        vec![]
    };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(&witness, &betas, &gammas, prover_data, common_data)
    );

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();

    // All lookup polys: RE and partial SLDCs.
    let lookup_polys =
        compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    let zs_partial_products_lookups = if has_lookup {
        [zs_partial_products, lookup_polys].concat()
    } else {
        zs_partial_products
    };

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products_lookups,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )?
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    let pre_transposed_quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys_pre_transpose::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &deltas,
            &alphas,
        )
    );

    let quotient_polys_commitment = timed!(
        timing,
        "commit to quotient polys",
        C::Compute::transpose_and_compute_from_coeffs(
            timing,
            pre_transposed_quotient_polys,
            quotient_degree,
            degree,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            prover_data.fft_root_table.as_ref(),
        )?
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            None,
            None,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap,
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment.merkle_tree.cap,
        quotient_polys_cap: quotient_polys_commitment.merkle_tree.cap,
        openings,
        opening_proof,
    };
    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let k_is = &common_data.k_is;
    let num_prods = common_data.num_partial_products;
    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i];
            let numerators = (0..common_data.config.num_routed_wires).map(|j| {
                let wire_value = witness.get_wire(i, j);
                let k_i = k_is[j];
                let s_id = k_i * x;
                wire_value + beta * s_id + gamma
            });
            let denominators = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j);
                    let s_sigma = s_sigmas[j];
                    wire_value + beta * s_sigma + gamma
                })
                .collect::<Vec<_>>();
            let denominator_invs = F::batch_multiplicative_inverse(&denominators);
            let quotient_values = numerators
                .zip(denominator_invs)
                .map(|(num, den_inv)| num * den_inv)
                .collect::<Vec<_>>();

            quotient_chunk_products(&quotient_values, degree)
        })
        .collect::<Vec<_>>();

    let mut z_x = F::ONE;
    let mut all_partial_products_and_zs = Vec::with_capacity(all_quotient_chunk_products.len());
    for quotient_chunk_products in all_quotient_chunk_products {
        let mut partial_products_and_z_gx =
            partial_products_and_z_gx(z_x, &quotient_chunk_products);
        // The last term is Z(gx), but we replace it with Z(x), otherwise Z would end up shifted.
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }

    transpose(&all_partial_products_and_zs)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F; 4],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree();
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups);

    // First poly is RE, the rest are partial SLDCs.
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
    }

    for LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in prover_data.lookup_rows.clone()
    {
        // Set values for partial Sums and RE.
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // Get combos for Sum.
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // Get 1/(alpha - combo).
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // Get lookup combos, used to check the well formation of the LUT.
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
                })
                .collect();

            // Compute next row's first value of RE.
            // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // Set values for partial LDCs.
        for row in (last_lu_row..last_lut_row).rev() {
            // Get looking combos.
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // Get 1 / (alpha - combo).
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // Valid at _any_ row, even `first_lu_row`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.concat()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 32;

/// Optional sub-phase profiling for `compute_quotient_polys_pre_transpose`.
///
/// Accumulates per-phase time (summed across rayon threads, so totals exceed wall-clock; the
/// *ratios* are what matter) and prints a breakdown when the `QUOTIENT_PROFILE` env var is set.
/// Counters are `core` atomics so this compiles under `no_std`; timing/printing is `std`-only.
pub(crate) mod qp_profile {
    use core::sync::atomic::{AtomicU64, Ordering};

    pub static GATHER_NS: AtomicU64 = AtomicU64::new(0);
    pub static EVAL_NS: AtomicU64 = AtomicU64::new(0);
    pub static GATE_NS: AtomicU64 = AtomicU64::new(0);
    pub static REDUCE_NS: AtomicU64 = AtomicU64::new(0);
    pub static DIV_NS: AtomicU64 = AtomicU64::new(0);
    pub static METAL_PERM_NS: AtomicU64 = AtomicU64::new(0);

    #[allow(unused)]
    pub(crate) static COUNTERS: [&AtomicU64; 6] = [
        &GATHER_NS,
        &EVAL_NS,
        &GATE_NS,
        &REDUCE_NS,
        &DIV_NS,
        &METAL_PERM_NS,
    ];

    #[cfg(feature = "std")]
    fn enabled() -> bool {
        use std::sync::OnceLock;
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| std::env::var("QUOTIENT_PROFILE").is_ok())
    }

    #[cfg(feature = "std")]
    pub fn timed<R>(counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        let start = std::time::Instant::now();
        let r = f();
        counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    #[cfg(not(feature = "std"))]
    pub fn timed<R>(_counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
        f()
    }

    #[cfg(feature = "std")]
    pub fn reset() {
        if enabled() {
            for c in COUNTERS {
                c.store(0, Ordering::Relaxed);
            }
        }
    }

    #[cfg(not(feature = "std"))]
    pub fn reset() {}

    #[cfg(feature = "std")]
    pub fn report() {
        if !enabled() {
            return;
        }
        let g = GATHER_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let e = EVAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let gate = GATE_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let reduce = REDUCE_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let metal_perm = METAL_PERM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        // perm-arithmetic = whole eval minus gate eval minus the alpha reduction.
        let perm_arith = (e - gate - reduce).max(0.0);
        let d = DIV_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let total = (g + e + d + metal_perm).max(1e-9);
        eprintln!(
            "[quotient-profile] thread-summed (ratios matter): gather={g:.0}ms eval={e:.0}ms \
             (gate={gate:.0}ms perm_arith={perm_arith:.0}ms reduce={reduce:.0}ms) \
             metal_perm={metal_perm:.0}ms div={d:.0}ms | \
             gather={:.0}% gate={:.0}% perm_arith={:.0}% reduce={:.0}% metal_perm={:.0}% div={:.0}%",
            100.0 * g / total,
            100.0 * gate / total,
            100.0 * perm_arith / total,
            100.0 * reduce / total,
            100.0 * metal_perm / total,
            100.0 * d / total,
        );
    }

    #[cfg(not(feature = "std"))]
    pub fn report() {}
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn goldilocks_slice_from_field_slice<F: Field>(values: &[F]) -> &[GoldilocksField] {
    assert_eq!(
        core::any::type_name::<F>(),
        core::any::type_name::<GoldilocksField>()
    );
    assert_eq!(
        core::mem::size_of::<F>(),
        core::mem::size_of::<GoldilocksField>()
    );
    assert_eq!(
        core::mem::align_of::<F>(),
        core::mem::align_of::<GoldilocksField>()
    );
    unsafe { core::slice::from_raw_parts(values.as_ptr() as *const GoldilocksField, values.len()) }
}

#[cfg(feature = "cuda")]
fn u64_slice_from_field_slice<F: Field>(values: &[F]) -> &[u64] {
    let g = goldilocks_slice_from_field_slice(values);
    // SAFETY: GoldilocksField is #[repr(transparent)] over u64.
    unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, g.len()) }
}

#[cfg(feature = "cuda")]
fn flatten_leaves_u64<F: Field>(leaves: &[Vec<F>]) -> Vec<u64> {
    let leaf_len = leaves.first().map_or(0, |l| l.len());
    let mut out = Vec::with_capacity(leaves.len() * leaf_len);
    for leaf in leaves {
        out.extend_from_slice(u64_slice_from_field_slice(leaf));
    }
    out
}

#[cfg(feature = "cuda")]
fn polynomial_batch_leaf_len<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    batch: &PolynomialBatch<F, C, D>,
) -> usize {
    batch.polynomials.len()
        + if batch.blinding {
            crate::fri::oracle::SALT_SIZE
        } else {
            0
        }
}

/// Whether to offload the quotient permutation argument to CUDA. OFF by default: on a discrete GPU
/// the per-call leaf upload (no device residency yet) makes it a net loss vs the CPU path. Enable
/// with `CUDA_QUOTIENT_PERM=1` to experiment (and for the eventual device-resident path).
#[cfg(all(feature = "cuda", feature = "std"))]
fn cuda_quotient_perm_enabled() -> bool {
    std::env::var_os("CUDA_QUOTIENT_PERM").is_some()
}

#[cfg(all(feature = "cuda", feature = "std"))]
static CUDA_QUOTIENT_PERM_RESIDENT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(feature = "cuda", feature = "std", test))]
pub(crate) fn reset_cuda_quotient_perm_resident_calls() {
    CUDA_QUOTIENT_PERM_RESIDENT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(all(feature = "cuda", feature = "std", test))]
pub(crate) fn cuda_quotient_perm_resident_calls() -> usize {
    CUDA_QUOTIENT_PERM_RESIDENT_CALLS.load(std::sync::atomic::Ordering::SeqCst)
}

/// Cheap predicate (no uploads) mirroring `try_cuda_quotient_perm_contribution`'s guards, so the
/// caller can decide to overlap the GPU kernel with CPU gate evaluation.
#[cfg(all(feature = "cuda", feature = "std"))]
fn cuda_quotient_perm_applicable<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &PolynomialBatch<F, C, D>,
    alphas: &[F],
) -> bool {
    cuda_quotient_perm_enabled()
        && core::any::type_name::<F>() == core::any::type_name::<GoldilocksField>()
        && D == 2
        && common_data.num_lookup_polys == 0
        && alphas.len() <= 8
        && plonky2_cuda::is_available()
        && {
            let force_upload = std::env::var_os("CUDA_QUOTIENT_PERM_FORCE_UPLOAD").is_some();
            let resident = prover_data
                .constants_sigmas_commitment
                .cuda_leaves
                .as_ref()
                .map_or(false, |h| !h.is_empty())
                && wires_commitment
                    .cuda_leaves
                    .as_ref()
                    .map_or(false, |h| !h.is_empty())
                && zs_partial_products_and_lookup_commitment
                    .cuda_leaves
                    .as_ref()
                    .map_or(false, |h| !h.is_empty());
            let host = !prover_data
                .constants_sigmas_commitment
                .merkle_tree
                .leaves
                .is_empty()
                && !wires_commitment.merkle_tree.leaves.is_empty()
                && !zs_partial_products_and_lookup_commitment
                    .merkle_tree
                    .leaves
                    .is_empty();
            if force_upload {
                host
            } else {
                resident || host
            }
        }
}

/// CUDA analogue of `try_metal_quotient_perm_contribution`. Computes the (un-Z_H-divided) permutation
/// contribution on the GPU from the commitments' leaf-major Merkle leaves, returning a flat
/// `[point * num_challenges + challenge]` vector to feed `run_quotient_loop`'s precomputed-perm path.
/// Phase B1: uploads all leaves each call (correctness first).
#[cfg(all(feature = "cuda", feature = "std"))]
#[allow(clippy::too_many_arguments)]
fn try_cuda_quotient_perm_contribution<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &PolynomialBatch<F, C, D>,
    points: &[F],
    step: usize,
    next_step: usize,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    z_h_on_coset: &ZeroPolyOnCoset<F>,
) -> Option<Vec<F>> {
    if core::any::type_name::<F>() != core::any::type_name::<GoldilocksField>()
        || D != 2
        || common_data.num_lookup_polys != 0
        || alphas.len() > 8
        || !plonky2_cuda::is_available()
    {
        return None;
    }

    let shifted_xs: Vec<F> = points.iter().map(|&x| F::coset_shift() * x).collect();
    let l0_values: Vec<F> = shifted_xs
        .iter()
        .enumerate()
        .map(|(i, &x)| z_h_on_coset.eval_l_0(i, x))
        .collect();

    let cs_leaf_len = polynomial_batch_leaf_len(&prover_data.constants_sigmas_commitment);
    let wires_leaf_len = polynomial_batch_leaf_len(wires_commitment);
    let zs_leaf_len = polynomial_batch_leaf_len(zs_partial_products_and_lookup_commitment);

    let run_cuda_quotient_perm_upload = || -> Option<Vec<u64>> {
        let cs_leaves = &prover_data.constants_sigmas_commitment.merkle_tree.leaves;
        let wires_leaves = &wires_commitment.merkle_tree.leaves;
        let zs_leaves = &zs_partial_products_and_lookup_commitment.merkle_tree.leaves;
        if cs_leaves.is_empty() || wires_leaves.is_empty() || zs_leaves.is_empty() {
            return None;
        }

        let cs_flat = flatten_leaves_u64(cs_leaves);
        let wires_flat = flatten_leaves_u64(wires_leaves);
        let zs_flat = flatten_leaves_u64(zs_leaves);

        Some(
            plonky2_cuda::CudaRuntime::get()
                .quotient_perm(
                    common_data.degree_bits(),
                    common_data.config.fri_config.rate_bits,
                    step,
                    next_step,
                    u64_slice_from_field_slice(&shifted_xs),
                    u64_slice_from_field_slice(&l0_values),
                    &cs_flat,
                    cs_leaf_len,
                    &zs_flat,
                    zs_leaf_len,
                    &wires_flat,
                    wires_leaf_len,
                    common_data.num_constants,
                    common_data.config.num_routed_wires,
                    common_data.config.num_challenges,
                    common_data.quotient_degree_factor,
                    common_data.num_partial_products,
                    u64_slice_from_field_slice(&common_data.k_is),
                    u64_slice_from_field_slice(alphas),
                    u64_slice_from_field_slice(betas),
                    u64_slice_from_field_slice(gammas),
                )
                .expect("cuda quotient_perm failed"),
        )
    };

    let force_upload = std::env::var_os("CUDA_QUOTIENT_PERM_FORCE_UPLOAD").is_some();
    let resident_handles = (
        prover_data.constants_sigmas_commitment.cuda_leaves.as_ref(),
        zs_partial_products_and_lookup_commitment
            .cuda_leaves
            .as_ref(),
        wires_commitment.cuda_leaves.as_ref(),
    );
    let perm_u64 = if force_upload {
        run_cuda_quotient_perm_upload()?
    } else if let (Some(cs_h), Some(zs_h), Some(wires_h)) = resident_handles {
        CUDA_QUOTIENT_PERM_RESIDENT_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let resident = plonky2_cuda::CudaRuntime::get()
            .quotient_perm_resident(
                common_data.degree_bits(),
                common_data.config.fri_config.rate_bits,
                step,
                next_step,
                u64_slice_from_field_slice(&shifted_xs),
                u64_slice_from_field_slice(&l0_values),
                cs_h,
                cs_leaf_len,
                zs_h,
                zs_leaf_len,
                wires_h,
                wires_leaf_len,
                common_data.num_constants,
                common_data.config.num_routed_wires,
                common_data.config.num_challenges,
                common_data.quotient_degree_factor,
                common_data.num_partial_products,
                u64_slice_from_field_slice(&common_data.k_is),
                u64_slice_from_field_slice(alphas),
                u64_slice_from_field_slice(betas),
                u64_slice_from_field_slice(gammas),
            )
            .expect("cuda quotient_perm_resident failed");

        if std::env::var_os("CUDA_QUOTIENT_PERM_VERIFY_UPLOAD").is_some() {
            let upload = run_cuda_quotient_perm_upload()?;
            if resident != upload {
                let alpha_len = alphas.len();
                let (idx, (got, expected)) = resident
                    .iter()
                    .zip(&upload)
                    .enumerate()
                    .find(|(_, (got, expected))| got != expected)
                    .expect("resident/upload length mismatch without differing prefix");
                panic!(
                    "CUDA resident quotient mismatch at point {}, challenge {}: resident={:#x}, upload={:#x}",
                    idx / alpha_len,
                    idx % alpha_len,
                    got,
                    expected
                );
            }
        }
        resident
    } else {
        run_cuda_quotient_perm_upload()?
    };

    // SAFETY: F == GoldilocksField (checked above), #[repr(transparent)] over u64.
    Some(
        perm_u64
            .into_iter()
            .map(|v| unsafe { core::mem::transmute_copy::<u64, F>(&v) })
            .collect(),
    )
}

/// Cheap predicate mirroring `try_metal_quotient_perm_contribution`'s guards, so the caller can
/// decide to overlap the GPU kernel with CPU work knowing the kernel will run (returns `Some`).
#[cfg(feature = "metal")]
fn metal_quotient_perm_applicable<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &PolynomialBatch<F, C, D>,
    alphas: &[F],
) -> bool {
    if core::any::type_name::<F>() != core::any::type_name::<GoldilocksField>()
        || D != 2
        || common_data.num_lookup_polys != 0
        || alphas.len() > 64
        || !plonky2_metal::is_available()
    {
        return false;
    }
    let (Some(cs), Some(w), Some(z)) = (
        prover_data.constants_sigmas_commitment.lde_values.as_ref(),
        wires_commitment.lde_values.as_ref(),
        zs_partial_products_and_lookup_commitment
            .lde_values
            .as_ref(),
    ) else {
        return false;
    };
    let leaf_count = 1usize << (wires_commitment.degree_log + wires_commitment.rate_bits);
    cs.len() % leaf_count == 0 && w.len() % leaf_count == 0 && z.len() % leaf_count == 0
}

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn try_metal_quotient_perm_contribution<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &PolynomialBatch<F, C, D>,
    points: &[F],
    step: usize,
    next_step: usize,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    z_h_on_coset: &ZeroPolyOnCoset<F>,
) -> Option<Vec<F>> {
    if core::any::type_name::<F>() != core::any::type_name::<GoldilocksField>()
        || D != 2
        || common_data.num_lookup_polys != 0
        || alphas.len() > 64
        || !plonky2_metal::is_available()
    {
        return None;
    }

    let constants_sigmas_lde = prover_data
        .constants_sigmas_commitment
        .lde_values
        .as_ref()?;
    let wires_lde = wires_commitment.lde_values.as_ref()?;
    let zpp_lde = zs_partial_products_and_lookup_commitment
        .lde_values
        .as_ref()?;
    let leaf_count = 1usize << (wires_commitment.degree_log + wires_commitment.rate_bits);
    if constants_sigmas_lde.len() % leaf_count != 0
        || wires_lde.len() % leaf_count != 0
        || zpp_lde.len() % leaf_count != 0
    {
        return None;
    }

    let shifted_xs = points
        .iter()
        .map(|&x| F::coset_shift() * x)
        .collect::<Vec<_>>();
    let l0_values = points
        .iter()
        .enumerate()
        .map(|(i, &x)| z_h_on_coset.eval_l_0(i, F::coset_shift() * x))
        .collect::<Vec<_>>();

    let output = plonky2_metal::MetalRuntime::get().quotient_perm_reduce(
        goldilocks_slice_from_field_slice(wires_lde),
        goldilocks_slice_from_field_slice(constants_sigmas_lde),
        goldilocks_slice_from_field_slice(zpp_lde),
        goldilocks_slice_from_field_slice(&shifted_xs),
        goldilocks_slice_from_field_slice(&l0_values),
        goldilocks_slice_from_field_slice(&common_data.k_is),
        goldilocks_slice_from_field_slice(betas),
        goldilocks_slice_from_field_slice(gammas),
        goldilocks_slice_from_field_slice(alphas),
        points.len(),
        leaf_count,
        step,
        next_step,
        common_data.num_constants,
        common_data.config.num_routed_wires,
        common_data.config.num_challenges,
        common_data.num_partial_products,
        common_data.quotient_degree_factor,
    );

    let output = output
        .into_iter()
        .map(|value| F::from_canonical_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();

    #[cfg(feature = "std")]
    if std::env::var("VERIFY_METAL_QUOTIENT_PERM").is_ok() {
        verify_metal_quotient_perm_contribution(
            common_data,
            constants_sigmas_lde,
            wires_lde,
            zpp_lde,
            &shifted_xs,
            &l0_values,
            step,
            next_step,
            betas,
            gammas,
            alphas,
            leaf_count,
            &output,
        );
    }

    Some(output)
}

#[cfg(all(feature = "metal", feature = "std"))]
#[allow(clippy::too_many_arguments)]
fn verify_metal_quotient_perm_contribution<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    constants_sigmas_lde: &[F],
    wires_lde: &[F],
    zpp_lde: &[F],
    shifted_xs: &[F],
    l0_values: &[F],
    step: usize,
    next_step: usize,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    leaf_count: usize,
    metal_output: &[F],
) {
    let point_count = shifted_xs.len();
    let num_challenges = common_data.config.num_challenges;
    let num_routed_wires = common_data.config.num_routed_wires;
    let num_prods = common_data.num_partial_products;
    let max_degree = common_data.quotient_degree_factor;

    for point in 0..point_count {
        let lde_point = point * step;
        let next_lde_point = ((point + next_step) % point_count) * step;
        let x = shifted_xs[point];
        let l_0_x = l0_values[point];
        let mut accum = vec![F::ZERO; alphas.len()];
        let mut powers = vec![F::ONE; alphas.len()];
        let mut add_term = |term: F| {
            for ((acc, power), &alpha) in accum.iter_mut().zip(&mut powers).zip(alphas) {
                *acc += term * *power;
                *power *= alpha;
            }
        };

        for challenge in 0..num_challenges {
            let z_x = zpp_lde[challenge * leaf_count + lde_point];
            add_term(l_0_x * z_x.sub_one());
        }

        for challenge in 0..num_challenges {
            let beta = betas[challenge];
            let gamma = gammas[challenge];
            let z_x = zpp_lde[challenge * leaf_count + lde_point];
            let z_gx = zpp_lde[challenge * leaf_count + next_lde_point];

            for chunk in 0..=num_prods {
                let start_wire = chunk * max_degree;
                let end_wire = (start_wire + max_degree).min(num_routed_wires);
                let mut numerator_product = F::ONE;
                let mut denominator_product = F::ONE;
                for wire in start_wire..end_wire {
                    let wire_value = wires_lde[wire * leaf_count + lde_point];
                    let s_id = common_data.k_is[wire] * x;
                    let s_sigma = constants_sigmas_lde
                        [(common_data.num_constants + wire) * leaf_count + lde_point];
                    numerator_product *= wire_value + beta * s_id + gamma;
                    denominator_product *= wire_value + beta * s_sigma + gamma;
                }

                let prev_acc = if chunk == 0 {
                    z_x
                } else {
                    zpp_lde[(num_challenges + challenge * num_prods + chunk - 1) * leaf_count
                        + lde_point]
                };
                let next_acc = if chunk < num_prods {
                    zpp_lde
                        [(num_challenges + challenge * num_prods + chunk) * leaf_count + lde_point]
                } else {
                    z_gx
                };
                add_term(prev_acc * numerator_product - next_acc * denominator_product);
            }
        }

        let start = point * alphas.len();
        for (alpha_i, (&expected, &actual)) in accum
            .iter()
            .zip(&metal_output[start..start + alphas.len()])
            .enumerate()
        {
            if expected != actual {
                panic!(
                    "Metal quotient permutation mismatch at point {point}, alpha {alpha_i}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }
}

#[cfg(not(feature = "metal"))]
#[allow(clippy::too_many_arguments, dead_code)]
fn try_metal_quotient_perm_contribution<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    _common_data: &CommonCircuitData<F, D>,
    _prover_data: &ProverOnlyCircuitData<F, C, D>,
    _wires_commitment: &PolynomialBatch<F, C, D>,
    _zs_partial_products_and_lookup_commitment: &PolynomialBatch<F, C, D>,
    _points: &[F],
    _step: usize,
    _next_step: usize,
    _betas: &[F],
    _gammas: &[F],
    _alphas: &[F],
    _z_h_on_coset: &ZeroPolyOnCoset<F>,
) -> Option<Vec<F>> {
    None
}

fn compute_quotient_polys_pre_transpose<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    deltas: &[F],
    alphas: &[F],
) -> Vec<Vec<F>> {
    let num_challenges = common_data.config.num_challenges;
    let has_lookup = common_data.num_lookup_polys != 0;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);

    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet."
    );

    qp_profile::reset();

    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();
    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);

    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        (0..num_challenges)
            .map(|i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];
                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_index = r - LookupSelectors::StartEnd as usize;
                        let lut_row_number =
                            common_data.luts[lut_index].len().div_ceil(num_lut_slots);
                        let degree = num_lut_slots * lut_row_number;
                        get_lut_poly(common_data, lut_index, cur_deltas, degree)
                            .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };
    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect();

    // The per-point quotient loop, parameterized by the (optional) precomputed permutation
    // contribution. With `Some(..)` the loop only evaluates gate constraints and combines them with
    // the supplied permutation terms; with `None` it computes the permutation argument on the CPU.
    let run_quotient_loop = |metal_perm_contribution: Option<&[F]>| -> Vec<Vec<F>> {
        points
            .par_chunks(BATCH_SIZE)
            .enumerate()
            .flat_map(|(batch_i, xs_batch)| {
                let n = xs_batch.len();
                let indices_batch: Vec<usize> =
                    (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + n).collect();

                let mut shifted_xs_batch = Vec::with_capacity(n);
                let mut local_zs_batch_vecs = Vec::with_capacity(n);
                let mut next_zs_batch_vecs = Vec::with_capacity(n);
                let mut local_lookup_batch_vecs = Vec::with_capacity(n);
                let mut next_lookup_batch_vecs = Vec::with_capacity(n);
                let mut partial_products_batch_vecs = Vec::with_capacity(n);
                let mut s_sigmas_batch_vecs = Vec::with_capacity(n);

                let mut local_constants_batch = vec![F::ZERO; n * common_data.num_constants];
                let mut local_wires_batch = vec![F::ZERO; n * common_data.config.num_wires];
                let use_poly_major_lde = prover_data
                    .constants_sigmas_commitment
                    .has_poly_major_lde_values()
                    || wires_commitment.has_poly_major_lde_values()
                    || zs_partial_products_and_lookup_commitment.has_poly_major_lde_values();

                qp_profile::timed(&qp_profile::GATHER_NS, || {
                    if use_poly_major_lde {
                        let next_indices_batch = indices_batch
                            .iter()
                            .map(|&i| (i + next_step) % lde_size)
                            .collect::<Vec<_>>();

                        s_sigmas_batch_vecs = prover_data
                            .constants_sigmas_commitment
                            .get_lde_values_batch(&indices_batch, step, common_data.sigmas_range());
                        local_zs_batch_vecs = zs_partial_products_and_lookup_commitment
                            .get_lde_values_batch(&indices_batch, step, common_data.zs_range());
                        next_zs_batch_vecs = zs_partial_products_and_lookup_commitment
                            .get_lde_values_batch(
                                &next_indices_batch,
                                step,
                                common_data.zs_range(),
                            );
                        partial_products_batch_vecs = zs_partial_products_and_lookup_commitment
                            .get_lde_values_batch(
                                &indices_batch,
                                step,
                                common_data.partial_products_range(),
                            );
                        if has_lookup {
                            local_lookup_batch_vecs = zs_partial_products_and_lookup_commitment
                                .get_lde_values_batch(
                                    &indices_batch,
                                    step,
                                    common_data.lookup_range(),
                                );
                            next_lookup_batch_vecs = zs_partial_products_and_lookup_commitment
                                .get_lde_values_batch(
                                    &next_indices_batch,
                                    step,
                                    common_data.lookup_range(),
                                );
                        }

                        prover_data
                            .constants_sigmas_commitment
                            .fill_lde_values_batch(
                                &indices_batch,
                                step,
                                common_data.constants_range(),
                                &mut local_constants_batch,
                            );
                        wires_commitment.fill_lde_values_batch(
                            &indices_batch,
                            step,
                            0..common_data.config.num_wires,
                            &mut local_wires_batch,
                        );
                    }

                    for (k, (&i, &x)) in indices_batch.iter().zip(xs_batch).enumerate() {
                        shifted_xs_batch.push(F::coset_shift() * x);

                        let i_next = (i + next_step) % lde_size;
                        if !use_poly_major_lde {
                            let local_constants_sigmas = prover_data
                                .constants_sigmas_commitment
                                .get_lde_values(i, step);
                            let local_wires = wires_commitment.get_lde_values(i, step);
                            let local_zs_partial_and_lookup =
                                zs_partial_products_and_lookup_commitment.get_lde_values(i, step);
                            let next_zs_partial_and_lookup =
                                zs_partial_products_and_lookup_commitment
                                    .get_lde_values(i_next, step);

                            s_sigmas_batch_vecs
                                .push(local_constants_sigmas[common_data.sigmas_range()].to_vec());
                            local_zs_batch_vecs
                                .push(local_zs_partial_and_lookup[common_data.zs_range()].to_vec());
                            next_zs_batch_vecs
                                .push(next_zs_partial_and_lookup[common_data.zs_range()].to_vec());
                            partial_products_batch_vecs.push(
                                local_zs_partial_and_lookup[common_data.partial_products_range()]
                                    .to_vec(),
                            );
                            if has_lookup {
                                local_lookup_batch_vecs.push(
                                    local_zs_partial_and_lookup[common_data.lookup_range()]
                                        .to_vec(),
                                );
                                next_lookup_batch_vecs.push(
                                    next_zs_partial_and_lookup[common_data.lookup_range()].to_vec(),
                                );
                            }

                            for (j, &c) in local_constants_sigmas[common_data.constants_range()]
                                .iter()
                                .enumerate()
                            {
                                local_constants_batch[j * n + k] = c;
                            }
                            for (j, &w) in local_wires.iter().enumerate() {
                                local_wires_batch[j * n + k] = w;
                            }
                        }
                    }
                });

                let local_zs_batch: Vec<&[F]> =
                    local_zs_batch_vecs.iter().map(|v| v.as_slice()).collect();
                let next_zs_batch: Vec<&[F]> =
                    next_zs_batch_vecs.iter().map(|v| v.as_slice()).collect();
                let local_lookup_batch: Vec<&[F]> = local_lookup_batch_vecs
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                let next_lookup_batch: Vec<&[F]> = next_lookup_batch_vecs
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                let partial_products_batch: Vec<&[F]> = partial_products_batch_vecs
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                let s_sigmas_batch: Vec<&[F]> =
                    s_sigmas_batch_vecs.iter().map(|v| v.as_slice()).collect();
                let precomputed_perm_batch_vecs = metal_perm_contribution.map(|perm| {
                    indices_batch
                        .iter()
                        .map(|&i| &perm[i * alphas.len()..(i + 1) * alphas.len()])
                        .collect::<Vec<_>>()
                });

                let vars_batch = EvaluationVarsBaseBatch::new(
                    n,
                    &local_constants_batch,
                    &local_wires_batch,
                    public_inputs_hash,
                );

                let mut quotient_values_batch = qp_profile::timed(&qp_profile::EVAL_NS, || {
                    eval_vanishing_poly_base_batch::<F, D>(
                        common_data,
                        &indices_batch,
                        &shifted_xs_batch,
                        vars_batch,
                        &local_zs_batch,
                        &next_zs_batch,
                        &local_lookup_batch,
                        &next_lookup_batch,
                        &partial_products_batch,
                        &s_sigmas_batch,
                        betas,
                        gammas,
                        deltas,
                        alphas,
                        &z_h_on_coset,
                        &lut_re_poly_evals_refs,
                        precomputed_perm_batch_vecs.as_deref(),
                    )
                });

                qp_profile::timed(&qp_profile::DIV_NS, || {
                    for (&i, quotient_values) in
                        indices_batch.iter().zip(quotient_values_batch.iter_mut())
                    {
                        let denominator_inv = z_h_on_coset.eval_inverse(i);
                        quotient_values
                            .iter_mut()
                            .for_each(|v| *v *= denominator_inv);
                    }
                });
                quotient_values_batch
            })
            .collect()
    };

    // GPU↔CPU overlap: when the Metal permutation kernel is applicable, run it on a background
    // thread while the CPU evaluates gate constraints (fed a zero permutation contribution), then
    // fold the GPU's already-reduced permutation terms in and divide by Z_H. This hides the GPU
    // latency behind the CPU gate work instead of running them serially.
    #[cfg(all(feature = "metal", feature = "std"))]
    let quotient_values = if metal_quotient_perm_applicable(
        common_data,
        prover_data,
        wires_commitment,
        zs_partial_products_and_lookup_commitment,
        alphas,
    ) {
        let alpha_len = alphas.len();
        let zero_perm = vec![F::ZERO; points.len() * alpha_len];
        let (mut quotient_values, perm) = std::thread::scope(|scope| {
            let gpu = scope.spawn(|| {
                qp_profile::timed(&qp_profile::METAL_PERM_NS, || {
                    try_metal_quotient_perm_contribution(
                        common_data,
                        prover_data,
                        wires_commitment,
                        zs_partial_products_and_lookup_commitment,
                        &points,
                        step,
                        next_step,
                        betas,
                        gammas,
                        alphas,
                        &z_h_on_coset,
                    )
                    .expect("metal quotient perm applicable but returned None")
                })
            });
            let quotient_values = run_quotient_loop(Some(&zero_perm));
            let perm = gpu.join().expect("metal quotient perm thread panicked");
            (quotient_values, perm)
        });
        quotient_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, qs)| {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                for (c, q) in qs.iter_mut().enumerate() {
                    *q += perm[i * alpha_len + c] * denominator_inv;
                }
            });
        quotient_values
    } else {
        run_quotient_loop(None)
    };

    // CUDA path (Phase B1.5): run the GPU permutation kernel on a background thread while the CPU
    // evaluates gate constraints (fed a zero perm), then fold the GPU's perm terms in and divide by
    // Z_H -- hiding the GPU latency (and the leaf upload) behind the CPU gate work.
    #[cfg(all(feature = "cuda", feature = "std", not(feature = "metal")))]
    let quotient_values = if cuda_quotient_perm_applicable(
        common_data,
        prover_data,
        wires_commitment,
        zs_partial_products_and_lookup_commitment,
        alphas,
    ) {
        let alpha_len = alphas.len();
        let zero_perm = vec![F::ZERO; points.len() * alpha_len];
        let (mut quotient_values, perm) = std::thread::scope(|scope| {
            let gpu = scope.spawn(|| {
                qp_profile::timed(&qp_profile::METAL_PERM_NS, || {
                    try_cuda_quotient_perm_contribution(
                        common_data,
                        prover_data,
                        wires_commitment,
                        zs_partial_products_and_lookup_commitment,
                        &points,
                        step,
                        next_step,
                        betas,
                        gammas,
                        alphas,
                        &z_h_on_coset,
                    )
                    .expect("cuda quotient perm applicable but returned None")
                })
            });
            let quotient_values = run_quotient_loop(Some(&zero_perm));
            let perm = gpu.join().expect("cuda quotient perm thread panicked");
            (quotient_values, perm)
        });
        quotient_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, qs)| {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                for (c, q) in qs.iter_mut().enumerate() {
                    *q += perm[i * alpha_len + c] * denominator_inv;
                }
            });
        quotient_values
    } else {
        run_quotient_loop(None)
    };

    #[cfg(not(any(
        all(feature = "metal", feature = "std"),
        all(feature = "cuda", feature = "std", not(feature = "metal"))
    )))]
    let quotient_values = run_quotient_loop(None);

    qp_profile::report();
    quotient_values
}
