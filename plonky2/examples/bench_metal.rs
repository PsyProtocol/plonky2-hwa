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
    AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig, PoseidonGoldilocksMetalConfig,
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
    println!("== {label} ==");
    let circuit = BenchCircuit::<2, C>::build::<PoseidonHash>(num_iters);
    let (proof, prove_time) = circuit.prove(seed_hash, profile_prover)?;

    let verify_start = Instant::now();
    circuit.data.verify(proof)?;
    let verify_time = verify_start.elapsed();

    println!("{label}: prove={prove_time:?} verify={verify_time:?}");
    Ok(prove_time)
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
    let profile_mode = std::env::args().nth(3);
    let profile_prover = matches!(profile_mode.as_deref(), Some("profile-detail") | Some("detail"));
    if profile_prover {
        init_profile_logger();
    }
    configure_metal_profile(profile_mode.as_deref());
    let metal_available = metal_available();
    println!("BENCH_ITERS={num_iters}");
    println!("BENCH_ROUNDS={rounds}");
    if let Some(profile_mode) = profile_mode {
        println!("BENCH_PROFILE={profile_mode}");
    }
    println!("metal device available: {metal_available}");
    if !metal_available {
        println!("metal config will use CPU fallback in this environment");
    }
    warm_up_metal();

    let seed_hash = HashOut {
        elements: GoldilocksField::rand_array::<4>(),
    };

    let mut cpu_total = 0.0;
    let mut metal_total = 0.0;
    for round in 0..rounds {
        println!("=== round {} ===", round + 1);
        let cpu = bench::<PoseidonGoldilocksConfig>("cpu", num_iters, seed_hash, profile_prover)?;
        let metal =
            bench::<PoseidonGoldilocksMetalConfig>("metal", num_iters, seed_hash, profile_prover)?;
        cpu_total += cpu.as_secs_f64();
        metal_total += metal.as_secs_f64();
    }

    let cpu_avg = cpu_total / rounds as f64;
    let metal_avg = metal_total / rounds as f64;
    println!("cpu avg prove: {:.6}s", cpu_avg);
    println!("metal avg prove: {:.6}s", metal_avg);
    println!(
        "metal avg speedup vs cpu: {:.2}%",
        (cpu_avg / metal_avg - 1.0) * 100.0
    );

    Ok(())
}

fn init_profile_logger() {
    let _ = log::set_logger(&STDERR_LOGGER);
    log::set_max_level(log::LevelFilter::Debug);
}

#[cfg(feature = "metal")]
fn metal_available() -> bool {
    plonky2_metal::is_available()
}

#[cfg(feature = "metal")]
fn warm_up_metal() {
    if plonky2_metal::is_available() {
        let start = Instant::now();
        let _ = plonky2_metal::MetalRuntime::get().warm_up();
        println!("metal runtime warm-up took {:?}", start.elapsed());
    }
}

#[cfg(feature = "metal")]
fn configure_metal_profile(profile_mode: Option<&str>) {
    match profile_mode {
        Some("profile-detail") | Some("detail") => {
            plonky2_metal::set_profile_detail_enabled(true);
        }
        Some("profile") => {
            plonky2_metal::set_profile_enabled(true);
        }
        _ => {}
    }
}

#[cfg(not(feature = "metal"))]
fn metal_available() -> bool {
    false
}

#[cfg(not(feature = "metal"))]
fn warm_up_metal() {}

#[cfg(not(feature = "metal"))]
fn configure_metal_profile(_profile_mode: Option<&str>) {}
