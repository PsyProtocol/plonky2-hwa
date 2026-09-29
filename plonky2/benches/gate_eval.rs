//! Compares row-by-row base-field gate evaluation (the default `eval_unfiltered_base_batch` loop
//! over `eval_unfiltered_base_one`) with the gate's own `eval_unfiltered_base_batch`.
//!
//! The workload mirrors quotient polynomial computation: 2^14 rows evaluated in batches of 32.
//! Both variants run interleaved on the same inputs; the minimum and median over all rounds are
//! reported, which is robust against other load on the machine.
//!
//! Run with `cargo bench -p plonky2 --bench gate_eval`.

mod allocator;

use std::hint::black_box;
use std::time::{Duration, Instant};

use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Sample;
use plonky2::gates::gate::Gate;
use plonky2::gates::poseidon::PoseidonGate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::plonk::vars::EvaluationVarsBaseBatch;

const NUM_ROWS: usize = 1 << 14;
const BATCH_SIZE: usize = 32;
const ROUNDS: usize = 30;

fn eval_row_by_row<F: RichField + Extendable<D>, G: Gate<F, D>, const D: usize>(
    gate: &G,
    vars_base: EvaluationVarsBaseBatch<F>,
) -> Vec<F> {
    let mut res = vec![F::ZERO; vars_base.len() * gate.num_constraints()];
    for (i, vars_base_one) in vars_base.iter().enumerate() {
        gate.eval_unfiltered_base_one(
            vars_base_one,
            StridedConstraintConsumer::new(&mut res, vars_base.len(), i),
        );
    }
    res
}

fn summarize(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort();
    (samples[0], samples[samples.len() / 2])
}

fn bench_gate<F: RichField + Extendable<D>, G: Gate<F, D>, const D: usize>(name: &str, gate: G) {
    let batches: Vec<(Vec<F>, Vec<F>)> = (0..NUM_ROWS / BATCH_SIZE)
        .map(|_| {
            (
                F::rand_vec(gate.num_constants() * BATCH_SIZE),
                F::rand_vec(gate.num_wires() * BATCH_SIZE),
            )
        })
        .collect();
    let public_inputs_hash = HashOut::<F>::rand();
    let run = |packed: bool| {
        let start = Instant::now();
        for (constants, wires) in &batches {
            let vars =
                EvaluationVarsBaseBatch::new(BATCH_SIZE, constants, wires, &public_inputs_hash);
            if packed {
                black_box(gate.eval_unfiltered_base_batch(black_box(vars)));
            } else {
                black_box(eval_row_by_row(&gate, black_box(vars)));
            }
        }
        start.elapsed()
    };

    // Warm up both paths.
    run(false);
    run(true);
    let mut row_by_row = Vec::with_capacity(ROUNDS);
    let mut batch = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        row_by_row.push(run(false));
        batch.push(run(true));
    }
    let (rbr_min, rbr_med) = summarize(row_by_row);
    let (b_min, b_med) = summarize(batch);
    println!(
        "{name}: {NUM_ROWS} rows in batches of {BATCH_SIZE}, packing width {}",
        <F as Packable>::Packing::WIDTH
    );
    println!("  row_by_row                 min {rbr_min:>10.3?}  median {rbr_med:>10.3?}");
    println!("  eval_unfiltered_base_batch min {b_min:>10.3?}  median {b_med:>10.3?}");
    println!(
        "  speedup (min) {:.2}x  (median) {:.2}x",
        rbr_min.as_secs_f64() / b_min.as_secs_f64(),
        rbr_med.as_secs_f64() / b_med.as_secs_f64()
    );
}

fn main() {
    // `cargo test --benches` passes `--test`-style arguments; only run under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    bench_gate("PoseidonGate", PoseidonGate::<GoldilocksField, 2>::new());
}
