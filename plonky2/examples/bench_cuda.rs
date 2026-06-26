//! End-to-end prove benchmark comparing the CPU config against the CUDA config on a
//! Poseidon-chain circuit.
//!
//! Run on a CUDA host with:
//! ```bash
//! CUDA_ARCH=sm_80 cargo run -p plonky2 --example bench_cuda --features cuda --release -- 16384 3
//! ```
//! Args: `<num_iters>` (Poseidon hashes in the chain, default 1<<14) and `<rounds>` (default 1).
//! Add a third arg `profile` to print the per-stage `TimingTree`.

use std::time::{Duration, Instant};

use anyhow::Result;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Sample};
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{
    AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig, PoseidonGoldilocksCudaConfig,
};
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::plonk::prover::prove as prove_with_timing;
use plonky2::util::timing::TimingTree;

struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}
static STDERR_LOGGER: StderrLogger = StderrLogger;

struct BenchCircuit<const D: usize, C: GenericConfig<D>> {
    data: CircuitData<C::F, C, D>,
    seed_target: HashOutTarget,
}

impl<const D: usize, C: GenericConfig<D>> BenchCircuit<D, C> {
    fn build<H: AlgebraicHasher<C::F>>(num_iters: usize) -> Self {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<C::F, D>::new(config);
        let seed_target = builder.add_virtual_hash();
        let mut cur = seed_target;

        for i in 0..num_iters {
            let i_constant = builder.constant(C::F::from_canonical_usize(i));
            cur = builder.hash_n_to_hash_no_pad::<H>(vec![
                i_constant,
                i_constant,
                i_constant,
                i_constant,
                cur.elements[0],
                cur.elements[1],
                cur.elements[2],
                cur.elements[3],
            ]);
        }
        builder.register_public_inputs(&cur.elements);

        let num_gates = builder.num_gates();
        let build_start = Instant::now();
        let data = builder.build::<C>();
        println!(
            "build: gates={} degree={} took {:?}",
            num_gates,
            data.common.degree(),
            build_start.elapsed()
        );

        Self { data, seed_target }
    }

    fn prove(
        &self,
        seed_hash: HashOut<C::F>,
        profile_prover: bool,
    ) -> Result<(ProofWithPublicInputs<C::F, C, D>, Duration)>
    where
        C::Hasher: Hasher<C::F>,
        C::InnerHasher: Hasher<C::F>,
    {
        let mut pw = PartialWitness::new();
        pw.set_hash_target(self.seed_target, seed_hash)?;
        let start = Instant::now();
        let proof = if profile_prover {
            let mut timing = TimingTree::new("prove", log::Level::Debug);
            let proof = prove_with_timing::<C::F, C, D>(
                &self.data.prover_only,
                &self.data.common,
                pw,
                &mut timing,
            )?;
            timing.filter(Duration::from_millis(1)).print();
            proof
        } else {
            self.data.prove(pw)?
        };
        Ok((proof, start.elapsed()))
    }
}

fn bench<C>(
    label: &str,
    num_iters: usize,
    seed_hash: HashOut<GoldilocksField>,
    profile_prover: bool,
) -> Result<Duration>
where
    C: GenericConfig<2, F = GoldilocksField, Hasher = PoseidonHash, InnerHasher = PoseidonHash>,
{
    let circuit = BenchCircuit::<2, C>::build::<PoseidonHash>(num_iters);
    let (proof, prove_time) = circuit.prove(seed_hash, profile_prover)?;

    let verify_start = Instant::now();
    circuit.data.verify(proof)?;
    let verify_time = verify_start.elapsed();

    println!("{label}: prove={prove_time:?} verify={verify_time:?}");
    Ok(prove_time)
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    plonky2_cuda::is_available()
}
#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

fn main() -> Result<()> {
    let num_iters = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 14);
    let rounds = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let profile_prover = std::env::args().any(|a| a == "profile" || a == "detail");
    if profile_prover {
        let _ = log::set_logger(&STDERR_LOGGER);
        log::set_max_level(log::LevelFilter::Debug);
    }

    println!("BENCH_ITERS={num_iters}");
    println!("BENCH_ROUNDS={rounds}");
    println!("cuda device available: {}", cuda_available());
    if !cuda_available() {
        println!("cuda config will use CPU fallback in this environment");
    }

    let seed_hash = HashOut {
        elements: GoldilocksField::rand_array::<4>(),
    };

    // Warm up the CUDA context/JIT so it does not skew round 1.
    if cuda_available() {
        let start = Instant::now();
        let _ = bench::<PoseidonGoldilocksCudaConfig>("cuda warmup", num_iters, seed_hash, false)?;
        println!("cuda warm-up took {:?}", start.elapsed());
    }

    let mut cpu_total = 0.0;
    let mut cuda_total = 0.0;
    for round in 0..rounds {
        println!("=== round {} ===", round + 1);
        let cpu = bench::<PoseidonGoldilocksConfig>("cpu", num_iters, seed_hash, profile_prover)?;
        let cuda =
            bench::<PoseidonGoldilocksCudaConfig>("cuda", num_iters, seed_hash, profile_prover)?;
        cpu_total += cpu.as_secs_f64();
        cuda_total += cuda.as_secs_f64();
    }

    let cpu_avg = cpu_total / rounds as f64;
    let cuda_avg = cuda_total / rounds as f64;
    println!("cpu avg prove:  {:.6}s", cpu_avg);
    println!("cuda avg prove: {:.6}s", cuda_avg);
    println!("cuda speedup vs cpu: {:.2}x", cpu_avg / cuda_avg);
    Ok(())
}
