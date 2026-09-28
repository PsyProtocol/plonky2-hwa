// HACK: Ideally this would live in `benches/`, but `cargo bench` doesn't allow
// custom CLI argument parsing (even with harness disabled). We could also have
// put it in `src/bin/`, but then we wouldn't have access to
// `[dev-dependencies]`.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
use core::num::ParseIntError;
use core::ops::RangeInclusive;
use core::str::FromStr;

use anyhow::{Context as _, Result};
use log::{info, Level, LevelFilter};
use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::gate::GateRef;
use plonky2::gates::multiplication_extension::MulExtensionGate;
use plonky2::gates::noop::NoopGate;
use plonky2::gates::poseidon::PoseidonGate;
use plonky2::gates::poseidon_mds::PoseidonMdsGate;
use plonky2::gates::random_access::RandomAccessGate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::plonk::verifier_helper::verify_proof_borrowed;
use plonky2::util::timing::TimingTree;
use plonky2_field::extension::Extendable;
use plonky2_field::interpolation::barycentric_weights;
use plonky2_field::types::Sample;
use plonky2_maybe_rayon::rayon;
use rand::rngs::OsRng;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use structopt::StructOpt;

pub fn new_coset_gate_with_max_degree<F: RichField + Extendable<D>, const D: usize>(
    subgroup_bits: usize,
    max_degree: usize,
) -> CosetInterpolationGate<F, D> {
    let mut base_gate = CosetInterpolationGate::<F, D>::default();

    assert!(max_degree > 1, "need at least quadratic constraints");

    let n_points = 1 << subgroup_bits;

    // Number of intermediate values required to compute interpolation with degree bound
    let n_intermediates = (n_points - 2) / (max_degree - 1);

    // Find minimum degree such that (n_points - 2) / (degree - 1) < n_intermediates + 1
    // Minimizing the degree this way allows the gate to be in a larger selector group
    let degree = (n_points - 2) / (n_intermediates + 1) + 2;

    let barycentric_weights = barycentric_weights(
        &F::two_adic_subgroup(subgroup_bits)
            .into_iter()
            .map(|x| (x, F::ZERO))
            .collect::<Vec<_>>(),
    );
    base_gate.subgroup_bits = subgroup_bits;
    base_gate.degree = degree;
    base_gate.barycentric_weights = barycentric_weights;
    base_gate
}

pub fn pad_circuit_degree<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    target_degree: usize,
) {
    while builder.num_gates() < (1u64 << (target_degree as u64)) as usize {
        builder.add_gate(NoopGate, vec![]);
    }
}

pub trait CircuitBuilderQEDCommonGates<F: RichField + Extendable<D>, const D: usize> {
    fn add_qed_type_a_common_gates(&mut self, coset_gate: Option<GateRef<F, D>>);
    fn add_qed_type_a_common_gates_with_coset(&mut self, subgroup_bits: usize, max_degree: usize);
    fn add_qed_type_b_common_gates(&mut self);
    fn add_qed_type_c_common_gates(&mut self);
    fn add_qed_type_d_common_gates(&mut self);
    fn add_qed_type_e_common_gates(&mut self);
    fn add_qed_type_f_common_gates(&mut self);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderQEDCommonGates<F, D>
    for CircuitBuilder<F, D>
{
    fn add_qed_type_a_common_gates(&mut self, coset_gate: Option<GateRef<F, D>>) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        // self.add_gate_to_gate_set(GateRef::new(ComparisonGate::new(32, 16)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));
        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
        if coset_gate.is_some() {
            self.add_gate_to_gate_set(coset_gate.unwrap());
        }
    }

    fn add_qed_type_a_common_gates_with_coset(&mut self, subgroup_bits: usize, max_degree: usize) {
        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(
            subgroup_bits,
            max_degree,
        ));
        self.add_qed_type_a_common_gates(Some(coset_gate));
    }

    fn add_qed_type_b_common_gates(&mut self) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        // self.add_gate_to_gate_set(GateRef::new(ComparisonGate::new(32, 16)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));

        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
        self.add_gate_to_gate_set(coset_gate);

        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
    }

    fn add_qed_type_c_common_gates(&mut self) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));

        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
        self.add_gate_to_gate_set(coset_gate);

        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
    }

    fn add_qed_type_d_common_gates(&mut self) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        // self.add_gate_to_gate_set(GateRef::new(ComparisonGate::new(32, 16)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));

        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
        self.add_gate_to_gate_set(coset_gate);

        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
    }

    fn add_qed_type_e_common_gates(&mut self) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));

        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
        self.add_gate_to_gate_set(coset_gate);

        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
    }

    fn add_qed_type_f_common_gates(&mut self) {
        self.add_gate_to_gate_set(GateRef::new(ConstantGate::new(self.config.num_constants)));
        self.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
            &self.config,
            4,
        )));

        let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
        self.add_gate_to_gate_set(coset_gate);

        self.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
        self.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
        self.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&self.config)));
        self.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
            &self.config,
        )));
        self.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
            &self.config,
        )));
    }
}

pub struct DummyPsyTypeCCircuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub dummy_public_inputs: HashOutTarget,
    pub circuit_data: CircuitData<F, C, D>,
}
impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    DummyPsyTypeCCircuit<F, C, D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    pub fn new(config: &CircuitConfig) -> Self {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let dummy_public_inputs = builder.add_virtual_hash();
        let mut res_linear = builder.hash_n_to_hash_no_pad::<C::Hasher>(
            vec![dummy_public_inputs.elements, dummy_public_inputs.elements].concat(),
        );
        for _ in 0..8000 {
            res_linear = builder.hash_n_to_hash_no_pad::<C::Hasher>(
                vec![res_linear.elements, dummy_public_inputs.elements].concat(),
            );
        }

        let zero = builder.zero();
        let is_first_zero = builder.is_equal(res_linear.elements[0], zero);
        let is_second_zero = builder.is_equal(res_linear.elements[1], zero);
        let are_both_zero = builder.and(is_first_zero, is_second_zero);
        let is_one_non_zero = builder.not(are_both_zero);
        let one = builder.one();
        builder.connect(is_one_non_zero.target, one);
        builder.register_public_inputs(&dummy_public_inputs.elements);
        builder.add_qed_type_c_common_gates();
        pad_circuit_degree(&mut builder, 12);

        let circuit_data = builder.build::<C>();
        println!("common_data: {:?}", circuit_data.common);
        DummyPsyTypeCCircuit {
            dummy_public_inputs,
            circuit_data,
        }
    }
    pub fn get_verifier_data(&self) -> &VerifierOnlyCircuitData<C, D> {
        &self.circuit_data.verifier_only
    }
    pub fn get_common_data(&self) -> &CommonCircuitData<F, D> {
        &self.circuit_data.common
    }
    pub fn prove(&self, dummy_public_inputs: HashOut<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut pw = PartialWitness::new();
        pw.set_hash_target(self.dummy_public_inputs, dummy_public_inputs)?;
        println!("Starting to prove dummy psy type c...");
        let mut timing = TimingTree::new("prove dummy psy type c", Level::Debug);
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        )?;
        timing.print();
        Ok(proof)
    }
}
pub struct DummyPsyTypeCRecursiveVerifierCircuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof_a_target: ProofWithPublicInputsTarget<D>,
    pub proof_a_verifier_data_target: VerifierCircuitTarget,

    pub proof_b_target: ProofWithPublicInputsTarget<D>,
    pub proof_b_verifier_data_target: VerifierCircuitTarget,

    pub circuit_data: CircuitData<F, C, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    DummyPsyTypeCRecursiveVerifierCircuit<F, C, D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    pub fn new(
        config: &CircuitConfig,
        child_cap_height: usize,
        child_common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let proof_a_target = builder.add_virtual_proof_with_pis(child_common_data);
        let proof_a_verifier_data_target = builder.add_virtual_verifier_data(child_cap_height);

        let proof_b_target = builder.add_virtual_proof_with_pis(child_common_data);
        let proof_b_verifier_data_target = builder.add_virtual_verifier_data(child_cap_height);

        builder.verify_proof::<C>(
            &proof_a_target,
            &proof_a_verifier_data_target,
            child_common_data,
        );
        builder.verify_proof::<C>(
            &proof_b_target,
            &proof_b_verifier_data_target,
            child_common_data,
        );

        let public_inputs_output = builder.hash_n_to_hash_no_pad::<C::Hasher>(
            [
                proof_a_target.public_inputs.clone(),
                proof_b_target.public_inputs.clone(),
            ]
            .concat(),
        );
        builder.register_public_inputs(&public_inputs_output.elements);

        builder.add_qed_type_c_common_gates();
        let circuit_data = builder.build::<C>();
        DummyPsyTypeCRecursiveVerifierCircuit {
            proof_a_target,
            proof_a_verifier_data_target,
            proof_b_target,
            proof_b_verifier_data_target,
            circuit_data,
        }
    }
    pub fn get_verifier_data(&self) -> &VerifierOnlyCircuitData<C, D> {
        &self.circuit_data.verifier_only
    }
    pub fn get_common_data(&self) -> &CommonCircuitData<F, D> {
        &self.circuit_data.common
    }

    pub fn prove(
        &self,
        left_proof: &ProofWithPublicInputs<F, C, D>,
        right_proof: &ProofWithPublicInputs<F, C, D>,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut pw = PartialWitness::new();
        pw.set_verifier_data_target(&self.proof_a_verifier_data_target, verifier_data)?;
        pw.set_verifier_data_target(&self.proof_b_verifier_data_target, verifier_data)?;
        pw.set_proof_with_pis_target(&self.proof_a_target, left_proof)?;
        pw.set_proof_with_pis_target(&self.proof_b_target, right_proof)?;
        let mut timing = TimingTree::new("prove dummy psy type c", Level::Trace);
        println!("Starting to prove dummy psy type c recursive...");
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        )?;
        timing.print();

        Ok(proof)
    }
}

#[derive(Clone, StructOpt, Debug)]
#[structopt(name = "bench_recursion")]
struct Options {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Apply an env_filter compatible log filter
    #[structopt(long, env, default_value)]
    log_filter: String,

    /// Random seed for deterministic runs.
    /// If not specified a new seed is generated from OS entropy.
    #[structopt(long, parse(try_from_str = parse_hex_u64))]
    seed: Option<u64>,

    /// Number of compute threads to use. Defaults to number of cores. Can be a single
    /// value or a rust style range.
    #[structopt(long, parse(try_from_str = parse_range_usize))]
    threads: Option<RangeInclusive<usize>>,
}

fn run_psy_bench_recursion() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    let config = CircuitConfig::standard_recursion_config();
    let dummy_inner = DummyPsyTypeCCircuit::<F, C, D>::new(&config);
    let dummy_recursive = DummyPsyTypeCRecursiveVerifierCircuit::<F, C, D>::new(
        &config,
        dummy_inner
            .circuit_data
            .verifier_only
            .constants_sigmas_cap
            .height(),
        dummy_inner.get_common_data(),
    );
    let dummy_proof_1 = dummy_inner.prove(HashOut::rand())?;
    let dummy_proof_2 = dummy_inner.prove(HashOut::rand())?;

    let start_time = std::time::Instant::now();
    let recursive_proof = dummy_recursive.prove(
        &dummy_proof_1,
        &dummy_proof_2,
        dummy_inner.get_verifier_data(),
    )?;

    println!("Recursive proof time: {:?}", start_time.elapsed());
    dummy_inner.circuit_data.verify(dummy_proof_1.clone())?;
    dummy_inner.circuit_data.verify(dummy_proof_2.clone())?;
    dummy_recursive
        .circuit_data
        .verify(recursive_proof.clone())?;

    let p = recursive_proof.clone();
    let mut ctr = 0;
    let start = std::time::Instant::now();
    for i in 0..2000 {
        let res = verify_proof_borrowed(
            &p,
            &dummy_recursive.circuit_data.verifier_only,
            &dummy_recursive.circuit_data.common,
        );
        if res.is_err() {
            println!("Failed to verify at iteration {}", i);
            break;
        }
        ctr += 1;
    }
    println!("Verified {} times successfully", ctr);
    let duration = start.elapsed();
    let avg_time = duration.as_secs_f64() / ctr as f64;
    println!(
        "verying {} proofs took {:?}, avg time per proof: {:.6} ms",
        ctr,
        duration,
        avg_time * 1000.0
    );
    let clones: [_; 100] = core::array::from_fn(|_| recursive_proof.clone());
    let start = std::time::Instant::now();
    for p in clones {
        dummy_recursive.circuit_data.verify(p)?;
    }
    println!("Verified [cd] {} times successfully", 100);
    let duration = start.elapsed();
    let avg_time = duration.as_secs_f64() / 100 as f64;
    println!(
        "verying [cd] {} proofs took {:?}, avg time per proof: {:.6} ms",
        100,
        duration,
        avg_time * 1000.0
    );

    println!("Psy benchmark recursion proof verified successfully");
    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments, see `--help` for details.
    let options = Options::from_args_safe()?;
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.parse_filters(&options.log_filter);
    builder.format_timestamp(None);
    match options.verbose {
        0 => &mut builder,
        1 => builder.filter_level(LevelFilter::Info),
        2 => builder.filter_level(LevelFilter::Debug),
        _ => builder.filter_level(LevelFilter::Trace),
    };
    builder.try_init()?;

    // Initialize randomness source
    let rng_seed = options.seed.unwrap_or_else(|| OsRng.next_u64());
    info!("Using random seed {rng_seed:16x}");
    let _rng = ChaCha8Rng::seed_from_u64(rng_seed);
    // TODO: Use `rng` to create deterministic runs

    let num_cpus = num_cpus::get();
    let threads = options.threads.unwrap_or(num_cpus..=num_cpus);

    // Since the `size` is most likely to be an unbounded range we make that the outer iterator.
    for threads in threads.clone() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("Failed to build thread pool.")?
            .install(|| {
                info!(
                    "Using {} compute threads on {} cores",
                    rayon::current_num_threads(),
                    num_cpus
                );
                // Run the benchmark. `options.lookup_type` determines which benchmark to run.
                //benchmark_function(&config, log2_inner_size, options.lookup_type)

                run_psy_bench_recursion()
            })?;
    }

    Ok(())
}

fn parse_hex_u64(src: &str) -> Result<u64, ParseIntError> {
    let src = src.strip_prefix("0x").unwrap_or(src);
    u64::from_str_radix(src, 16)
}

fn parse_range_usize(src: &str) -> Result<RangeInclusive<usize>, ParseIntError> {
    if let Some((left, right)) = src.split_once("..=") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            usize::from_str(right)?,
        ))
    } else if let Some((left, right)) = src.split_once("..") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            if right.is_empty() {
                usize::MAX
            } else {
                usize::from_str(right)?.saturating_sub(1)
            },
        ))
    } else {
        let value = usize::from_str(src)?;
        Ok(RangeInclusive::new(value, value))
    }
}
