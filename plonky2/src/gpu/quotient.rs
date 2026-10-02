//! Quotient polynomial evaluation on the GPU (lab, `PLONKY2_GPU_QUOTIENT=1`).
//!
//! The kernel (`cuda/quotient.cu`) evaluates, at every point of the quotient domain, the same
//! terms as `eval_vanishing_poly_base_batch` and returns the same field values: the inputs are
//! the coset LDEs of the constants/sigmas, wires and Z/partial-product polynomials, computed on
//! the device from the prover's coefficients. Circuits are eligible when every gate has a device
//! implementation (looked up by downcasting), there are no lookups, no zero knowledge, two
//! challenges, and the quotient domain equals the LDE domain (`rate_bits` = quotient degree bits).
//! Anything else stays on the CPU.
//!
//! `PLONKY2_GPU_QUOTIENT_CHECK=1` (and every test build) also computes the CPU result and compares
//! them; on a mismatch the CPU result is used and the first mismatching point is diagnosed gate by
//! gate.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use super::*;
use crate::field::extension::Extendable;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
use crate::gates::arithmetic_base::ArithmeticGate;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::base_sum::BaseSumGate;
use crate::gates::constant::ConstantGate;
use crate::gates::coset_interpolation::CosetInterpolationGate;
use crate::gates::exponentiation::ExponentiationGate;
use crate::gates::multiplication_extension::MulExtensionGate;
use crate::gates::noop::NoopGate;
use crate::gates::poseidon::PoseidonGate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::gates::public_input::PublicInputGate;
use crate::gates::random_access::RandomAccessGate;
use crate::gates::reducing::ReducingGate;
use crate::gates::reducing_extension::ReducingExtensionGate;
use crate::hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS, N_ROUNDS, SPONGE_WIDTH};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::util::log2_ceil;

/// Gate kinds; must match `GateKind` in `cuda/quotient.cu`.
mod kind {
    pub const NOOP: u32 = 0;
    pub const CONSTANT: u32 = 1;
    pub const PUBLIC_INPUT: u32 = 2;
    pub const ARITHMETIC: u32 = 3;
    pub const BASE_SUM: u32 = 4;
    pub const ARITH_EXT: u32 = 5;
    pub const MUL_EXT: u32 = 6;
    pub const REDUCING: u32 = 7;
    pub const REDUCING_EXT: u32 = 8;
    pub const RANDOM_ACCESS: u32 = 9;
    pub const POSEIDON_MDS: u32 = 10;
    pub const POSEIDON: u32 = 11;
    pub const COSET_INTERP: u32 = 12;
    pub const EXPONENTIATION: u32 = 13;
    pub const COMPARISON: u32 = 14;
    pub const U32_ADD_MANY: u32 = 15;
    pub const U32_ARITHMETIC: u32 = 16;
    pub const U32_SUBTRACTION: u32 = 17;
    pub const U32_INTERLEAVE: u32 = 18;
    pub const UNINTERLEAVE_B32: u32 = 19;
    pub const UNINTERLEAVE_U32: u32 = 20;
    pub const U32_RANGE_CHECK: u32 = 21;
}

/// Number of challenges the kernel is compiled for (`NC` in the kernel).
const NC: usize = 2;
const MAX_RA_BITS: usize = 5;
const MAX_CI_BITS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GateDesc {
    kind: u32,
    row: u32,
    selector: u32,
    group_lo: u32,
    group_hi: u32,
    p0: u32,
    p1: u32,
    p2: u32,
    data_off: u32,
    num_constraints: u32,
}

/// Mirrors `QuotientArgs` in `cuda/quotient.cu`.
#[repr(C)]
struct QuotientArgs {
    cs: *const u64,
    wires: *const u64,
    zs: *const u64,
    aux: *const u64,
    gates: *const GateDesc,
    out: *mut u64,
    debug: *mut u64,
    coset_shift: u64,
    omega: u64,
    n_field: u64,
    zh_evals: [u64; 8],
    zh_invs: [u64; 8],
    pih: [u64; 4],
    lg_lde: u32,
    rate_mask: u32,
    next_step: u32,
    num_constants: u32,
    const_prefix: u32,
    many_selectors: u32,
    num_routed: u32,
    num_wires: u32,
    num_prods: u32,
    max_degree: u32,
    off_kis: u32,
    off_betas: u32,
    off_gammas: u32,
    off_alpha: u32,
    num_terms: u32,
    off_gate_data: u32,
    num_gates: u32,
    debug_stride: u32,
    debug_point: i32,
}

extern "C" {
    fn plonky2_quotient_init(poseidon_constants: *const u64, len: u32) -> i32;
    fn plonky2_quotient_poseidon_constants_len() -> u32;
    fn plonky2_quotient_launch(args: *const QuotientArgs) -> i32;
}

/// Device buffers of the quotient evaluation (only touched under the GPU lock).
pub(super) struct QuotientBufs {
    input: Pinned,
    out: Pinned,
    d_in: DevBuf,
    d_cs: DevBuf,
    d_wires: DevBuf,
    d_zs: DevBuf,
    d_aux: DevBuf,
    d_gates: DevBuf,
    d_out: DevBuf,
    d_debug: DevBuf,
}

impl QuotientBufs {
    pub(super) const fn new() -> Self {
        Self {
            input: Pinned::empty(),
            out: Pinned::empty(),
            d_in: DevBuf::empty(),
            d_cs: DevBuf::empty(),
            d_wires: DevBuf::empty(),
            d_zs: DevBuf::empty(),
            d_aux: DevBuf::empty(),
            d_gates: DevBuf::empty(),
            d_out: DevBuf::empty(),
            d_debug: DevBuf::empty(),
        }
    }
}

static Q_GPU: AtomicU64 = AtomicU64::new(0);
static Q_CPU: AtomicU64 = AtomicU64::new(0);
static Q_MISMATCH: AtomicU64 = AtomicU64::new(0);
static Q_EXEC_NS: AtomicU64 = AtomicU64::new(0);
static Q_LOCK_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static Q_FAILED: AtomicBool = AtomicBool::new(false);
static Q_OOM: AtomicU64 = AtomicU64::new(0);

/// (GPU evaluations, CPU fallbacks, check mismatches, device ns, lock wait ns).
pub fn quotient_stats() -> (u64, u64, u64, u64, u64) {
    (
        Q_GPU.load(Ordering::Relaxed),
        Q_CPU.load(Ordering::Relaxed),
        Q_MISMATCH.load(Ordering::Relaxed),
        Q_EXEC_NS.load(Ordering::Relaxed),
        Q_LOCK_WAIT_NS.load(Ordering::Relaxed),
    )
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_QUOTIENT").ok().as_deref() == Some("1"))
}

/// `PLONKY2_GPU_QUOTIENT_CHECK=1`: compare every GPU quotient with the CPU one (lab).
pub(crate) fn check_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    cfg!(test) || *ON.get_or_init(|| std::env::var("PLONKY2_GPU_QUOTIENT_CHECK").ok().as_deref() == Some("1"))
}

/// Largest LDE (all columns) evaluated on the device; larger circuits stay on the CPU.
fn max_lde_bytes() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_QUOTIENT_MAX_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 30)
    })
}

pub(crate) fn record_cpu() {
    Q_CPU.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_mismatch() {
    Q_MISMATCH.fetch_add(1, Ordering::Relaxed);
}

fn poseidon_constants() -> Vec<u64> {
    type G = GoldilocksField;
    let mut v = Vec::new();
    v.extend_from_slice(&ALL_ROUND_CONSTANTS[..SPONGE_WIDTH * N_ROUNDS]);
    v.extend_from_slice(&<G as Poseidon>::MDS_MATRIX_CIRC);
    v.extend_from_slice(&<G as Poseidon>::MDS_MATRIX_DIAG);
    v.extend_from_slice(&<G as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT);
    v.extend_from_slice(&<G as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS);
    for row in <G as Poseidon>::FAST_PARTIAL_ROUND_VS {
        v.extend_from_slice(&row);
    }
    for row in <G as Poseidon>::FAST_PARTIAL_ROUND_W_HATS {
        v.extend_from_slice(&row);
    }
    for row in <G as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX {
        v.extend_from_slice(&row);
    }
    v
}

/// Uploads the Poseidon constants once (call under the GPU lock).
fn init_once() -> GpuResult<()> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let c = poseidon_constants();
        let expected = unsafe { plonky2_quotient_poseidon_constants_len() } as usize;
        if c.len() != expected {
            return Err(format!("Poseidon constants: {} values, kernel expects {expected}", c.len()));
        }
        cuda(unsafe { plonky2_quotient_init(c.as_ptr(), c.len() as u32) }, "quotient constants")
    })
    .clone()
}

/// A circuit's gates as kernel descriptors, plus per-gate constant data.
struct Plan {
    gates: Vec<GateDesc>,
    data: Vec<u64>,
    max_constraints: usize,
    /// Some gates were recognised by id rather than by type.
    by_id: bool,
    /// Quotients still to be compared with the CPU before the plan is trusted.
    probation: AtomicU32,
    /// Set after a mismatch: the circuit stays on the CPU.
    disabled: AtomicBool,
}

fn plan<F: RichField + Extendable<D>, const D: usize>(common: &CommonCircuitData<F, D>) -> Result<Plan, String> {
    let mut gates = Vec::with_capacity(common.gates.len());
    let mut data: Vec<u64> = Vec::new();
    let mut max_constraints = 0;
    let mut by_id = false;
    for (i, g) in common.gates.iter().enumerate() {
        let any = g.0.as_any();
        let selector = common.selectors_info.selector_indices[i];
        let group = common.selectors_info.groups[selector].clone();
        let mut d = GateDesc {
            row: i as u32,
            selector: selector as u32,
            group_lo: group.start as u32,
            group_hi: group.end as u32,
            num_constraints: g.0.num_constraints() as u32,
            ..GateDesc::default()
        };
        max_constraints = max_constraints.max(g.0.num_constraints());
        if any.is::<NoopGate>() {
            d.kind = kind::NOOP;
        } else if let Some(x) = any.downcast_ref::<ConstantGate>() {
            d.kind = kind::CONSTANT;
            d.p0 = x.num_consts as u32;
        } else if any.is::<PublicInputGate>() {
            d.kind = kind::PUBLIC_INPUT;
        } else if let Some(x) = any.downcast_ref::<ArithmeticGate>() {
            d.kind = kind::ARITHMETIC;
            d.p0 = x.num_ops as u32;
        } else if let Some((limbs, base)) = base_sum_gate(any) {
            d.kind = kind::BASE_SUM;
            d.p0 = limbs as u32;
            d.p1 = base as u32;
        } else if let Some(x) = any.downcast_ref::<ArithmeticExtensionGate<D>>() {
            d.kind = kind::ARITH_EXT;
            d.p0 = x.num_ops as u32;
        } else if let Some(x) = any.downcast_ref::<MulExtensionGate<D>>() {
            d.kind = kind::MUL_EXT;
            d.p0 = x.num_ops as u32;
        } else if let Some(x) = any.downcast_ref::<ReducingGate<D>>() {
            d.kind = kind::REDUCING;
            d.p0 = x.num_coeffs as u32;
        } else if let Some(x) = any.downcast_ref::<ReducingExtensionGate<D>>() {
            d.kind = kind::REDUCING_EXT;
            d.p0 = x.num_coeffs as u32;
        } else if let Some(x) = any.downcast_ref::<RandomAccessGate<F, D>>() {
            if x.bits > MAX_RA_BITS {
                return Err(format!("RandomAccessGate with {} bits", x.bits));
            }
            d.kind = kind::RANDOM_ACCESS;
            d.p0 = x.bits as u32;
            d.p1 = x.num_copies as u32;
            d.p2 = x.num_extra_constants as u32;
        } else if any.is::<PoseidonMdsGate<F, D>>() {
            d.kind = kind::POSEIDON_MDS;
        } else if any.is::<PoseidonGate<F, D>>() {
            d.kind = kind::POSEIDON;
        } else if let Some(x) = any.downcast_ref::<CosetInterpolationGate<F, D>>() {
            if x.subgroup_bits > MAX_CI_BITS || x.degree < 2 {
                return Err(format!("CosetInterpolationGate with {} subgroup bits", x.subgroup_bits));
            }
            d.kind = kind::COSET_INTERP;
            d.p0 = x.subgroup_bits as u32;
            d.p1 = x.degree as u32;
            d.data_off = data.len() as u32;
            data.extend(F::two_adic_subgroup(x.subgroup_bits).iter().map(|v| v.to_canonical_u64()));
            data.extend(x.barycentric_weights.iter().map(|v| v.to_canonical_u64()));
        } else if let Some(x) = any.downcast_ref::<ExponentiationGate<F, D>>() {
            d.kind = kind::EXPONENTIATION;
            d.p0 = x.num_power_bits as u32;
        } else if let Some((k, p0, p1, p2)) = psy_u32_gate(&g.0.id(), g.0.num_constraints()) {
            // Gates defined outside plonky2 are recognised by id; the plan is then on probation
            // (its first quotient is compared with the CPU's).
            d.kind = k;
            d.p0 = p0;
            d.p1 = p1;
            d.p2 = p2;
            by_id = true;
        } else {
            return Err(format!("no device implementation for {}", g.0.id()));
        }
        gates.push(d);
    }
    Ok(Plan { gates, data, max_constraints, by_id, probation: AtomicU32::new(probation()), disabled: AtomicBool::new(false) })
}

/// `BaseSumGate<B>` for the bases plonky2 circuits use: (number of limbs, base).
fn base_sum_gate(any: &dyn core::any::Any) -> Option<(usize, usize)> {
    macro_rules! try_base {
        ($($b:literal),*) => {
            $(if let Some(x) = any.downcast_ref::<BaseSumGate<$b>>() {
                return Some((x.num_limbs, $b));
            })*
        };
    }
    try_base!(2, 3, 4, 8, 16);
    None
}

/// Parses `name { key: value, ... }` gate ids of the psy u32 gates (both crates define the same
/// constraints). The constraint count must match the layout the kernel assumes.
fn psy_u32_gate(id: &str, num_constraints: usize) -> Option<(u32, u32, u32, u32)> {
    let name = id.split([' ', '{', '(', '<']).next()?;
    let param = |key: &str| -> Option<u32> {
        let rest = &id[id.find(&format!("{key}: "))? + key.len() + 2..];
        rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    let n = num_constraints as u32;
    match name {
        "ComparisonGate" => {
            let (bits, chunks) = (param("num_bits")?, param("num_chunks")?);
            let chunk_bits = bits.div_ceil(chunks);
            (chunks > 0 && chunk_bits <= 8 && n == 2 + 5 * chunks + 1 + (chunk_bits + 1) + 2)
                .then_some((kind::COMPARISON, bits, chunks, 0))
        }
        "U32AddManyGate" => {
            let (addends, ops) = (param("num_addends")?, param("num_ops")?);
            (ops > 0 && n % ops == 0 && n / ops >= 3 + 16).then_some((kind::U32_ADD_MANY, addends, ops, n / ops - 3))
        }
        "U32ArithmeticGate" => {
            let ops = param("num_ops")?;
            (n == 36 * ops).then_some((kind::U32_ARITHMETIC, ops, 0, 0))
        }
        "U32SubtractionGate" => {
            let ops = param("num_ops")?;
            (n == 19 * ops).then_some((kind::U32_SUBTRACTION, ops, 0, 0))
        }
        "U32InterleaveGate" => {
            let ops = param("num_ops")?;
            (n == 34 * ops).then_some((kind::U32_INTERLEAVE, ops, 0, 0))
        }
        "UninterleaveToB32Gate" => {
            let ops = param("num_ops")?;
            (n == 67 * ops).then_some((kind::UNINTERLEAVE_B32, ops, 0, 0))
        }
        "UninterleaveToU32Gate" => {
            let ops = param("num_ops")?;
            (n == 67 * ops).then_some((kind::UNINTERLEAVE_U32, ops, 0, 0))
        }
        "U32RangeCheckGate" => {
            let limbs = param("num_input_limbs")?;
            (n == 17 * limbs).then_some((kind::U32_RANGE_CHECK, limbs, 0, 0))
        }
        _ => None,
    }
}

/// `PLONKY2_GPU_QUOTIENT_PROBATION` (default 1): how many quotients of each circuit are compared
/// with the CPU's before the GPU result is trusted. Any mismatch moves the circuit to the CPU.
fn probation() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_QUOTIENT_PROBATION").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
    })
}

fn cached_plan<F: RichField + Extendable<D>, const D: usize>(
    common: &CommonCircuitData<F, D>,
    circuit_digest: u64,
) -> Arc<Result<Plan, String>> {
    static PLANS: OnceLock<Mutex<HashMap<u64, Arc<Result<Plan, String>>>>> = OnceLock::new();
    let plans = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = plans.lock().unwrap_or_else(|e| e.into_inner()).get(&circuit_digest) {
        return p.clone();
    }
    let p = Arc::new(plan(common));
    if let Err(e) = p.as_ref() {
        log::info!("plonky2 GPU quotient: circuit {circuit_digest:016x} stays on the CPU: {e}");
    }
    plans.lock().unwrap_or_else(|e| e.into_inner()).insert(circuit_digest, p.clone());
    p
}

/// At the start of a proof: when its quotient will run on the GPU, keep the LDEs of the wires and
/// Z commitments on the device for it.
pub(crate) fn begin_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
) {
    if !enabled() || Q_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return;
    }
    let config = &common.config;
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || D != 2
        || config.num_challenges != NC
        || common.num_lookup_polys != 0
        || config.zero_knowledge
        || log2_ceil(common.quotient_degree_factor) != config.fri_config.rate_bits
    {
        return;
    }
    let digest = crate::plonk::config::GenericHashOut::to_vec(&prover_data.circuit_digest)[0].to_canonical_u64();
    let plan = cached_plan(common, digest);
    if matches!(plan.as_ref(), Ok(p) if !p.disabled.load(Ordering::Relaxed)) {
        super::begin_capture();
    }
}

/// Evaluates the quotient on the GPU. `Err` gives the reason it stayed on the CPU.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_compute<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &[F],
    wires: &PolynomialBatch<F, C, D>,
    zs: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
) -> Result<GpuQuotient<F>, String> {
    let (out, n_points, plan) = run(common, prover_data, public_inputs_hash, wires, zs, betas, gammas, alphas, None)?;
    let values = (0..n_points).map(|i| (0..NC).map(|c| F::from_canonical_u64(out[c * n_points + i])).collect()).collect();
    Ok(GpuQuotient { values, plan })
}

/// A GPU quotient and the circuit plan that produced it.
pub(crate) struct GpuQuotient<F> {
    pub values: Vec<Vec<F>>,
    plan: Arc<Result<Plan, String>>,
}

impl<F> GpuQuotient<F> {
    fn plan(&self) -> &Plan {
        self.plan.as_ref().as_ref().expect("only successful plans produce results")
    }

    /// The result must be compared with the CPU's before use: check mode, or the circuit is
    /// still on probation.
    pub fn needs_check(&self) -> bool {
        check_enabled() || self.plan().probation.load(Ordering::Relaxed) > 0
    }

    /// The comparison matched.
    pub fn passed(&self) {
        let _ = self.plan().probation.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1));
    }

    /// The comparison failed: the circuit stays on the CPU from now on.
    pub fn failed(&self) {
        self.plan().disabled.store(true, Ordering::Relaxed);
        log::error!(
            "plonky2 GPU quotient: circuit moved to the CPU after a mismatch (gates recognised by id: {})",
            self.plan().by_id
        );
    }
}

/// Runs the evaluation; with `debug_point`, returns instead every gate's unfiltered constraints
/// at that point (`[gate][max constraints]`).
#[allow(clippy::too_many_arguments)]
fn run<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &[F],
    wires: &PolynomialBatch<F, C, D>,
    zs: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    debug_point: Option<usize>,
) -> Result<(Vec<u64>, usize, Arc<Result<Plan, String>>), String> {
    if !enabled() {
        return Err("disabled".into());
    }
    if Q_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return Err("gpu_failed_earlier".into());
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() || D != 2 {
        return Err("unsupported field".into());
    }
    let b = backend().ok_or("gpu_disabled")?;
    let config = &common.config;
    let nc = config.num_challenges;
    let quotient_degree_bits = log2_ceil(common.quotient_degree_factor);
    let rate_bits = config.fri_config.rate_bits;
    if nc != NC {
        return Err(format!("{nc} challenges"));
    }
    if common.num_lookup_polys != 0 || config.zero_knowledge {
        return Err("lookups or zero knowledge".into());
    }
    if quotient_degree_bits != rate_bits || quotient_degree_bits > 3 {
        return Err(format!("quotient degree bits {quotient_degree_bits}, rate bits {rate_bits}"));
    }
    let lg_n = common.degree_bits();
    let lg_lde = lg_n + rate_bits;
    if lg_lde < 7 || lg_lde > MAX_LG {
        return Err(format!("LDE size 2^{lg_lde}"));
    }
    let n = 1usize << lg_n;
    let big_n = 1usize << lg_lde;
    let num_routed = config.num_routed_wires;
    let num_prods = common.num_partial_products;
    let max_degree = common.quotient_degree_factor;
    if num_routed.div_ceil(max_degree) != num_prods + 1 {
        return Err("partial product layout".into());
    }
    let cs_polys = &prover_data.constants_sigmas_commitment.polynomials;
    let cols = [cs_polys.len(), wires.polynomials.len(), zs.polynomials.len()];
    if cols[0] != common.num_constants + num_routed || cols[1] != config.num_wires || cols[2] != nc * (1 + num_prods) {
        return Err(format!("column counts {cols:?}"));
    }
    let total_cols: usize = cols.iter().sum();
    if total_cols * big_n * 8 > max_lde_bytes() {
        return Err(format!("LDE of {} bytes", total_cols * big_n * 8));
    }
    if cs_polys.iter().chain(&wires.polynomials).chain(&zs.polynomials).any(|p| p.len() != n) {
        return Err("polynomial lengths".into());
    }
    let digest = crate::plonk::config::GenericHashOut::to_vec(&prover_data.circuit_digest)[0].to_canonical_u64();
    let plan_ref = cached_plan(common, digest);
    let plan = plan_ref.as_ref().as_ref().map_err(|e| e.clone())?;
    if plan.disabled.load(Ordering::Relaxed) {
        return Err("disabled after a mismatch".into());
    }

    // Auxiliary data: k_is | betas | gammas | alpha powers [NC][num_terms] | gate data.
    let num_chunks = num_prods + 1;
    let num_terms = nc + nc * num_chunks + common.num_gate_constraints;
    let canon = |v: &F| v.to_canonical_u64();
    let mut aux: Vec<u64> = Vec::new();
    let off_kis = aux.len();
    aux.extend(common.k_is.iter().map(canon));
    let off_betas = aux.len();
    aux.extend(betas.iter().map(canon));
    let off_gammas = aux.len();
    aux.extend(gammas.iter().map(canon));
    let off_alpha = aux.len();
    for &alpha in alphas {
        let mut p = F::ONE;
        for _ in 0..num_terms {
            aux.push(p.to_canonical_u64());
            p *= alpha;
        }
    }
    let off_gate_data = aux.len();
    aux.extend_from_slice(&plan.data);

    let z_h = ZeroPolyOnCoset::<F>::new(lg_n, quotient_degree_bits);
    let rate = 1usize << quotient_degree_bits;
    let mut zh_evals = [0u64; 8];
    let mut zh_invs = [0u64; 8];
    for r in 0..rate {
        zh_evals[r] = z_h.eval(r).to_canonical_u64();
        zh_invs[r] = z_h.eval_inverse(r).to_canonical_u64();
    }
    let mut pih = [0u64; 4];
    for (d, s) in pih.iter_mut().zip(public_inputs_hash) {
        *d = s.to_canonical_u64();
    }
    let mut args = QuotientArgs {
        cs: core::ptr::null(),
        wires: core::ptr::null(),
        zs: core::ptr::null(),
        aux: core::ptr::null(),
        gates: core::ptr::null(),
        out: core::ptr::null_mut(),
        debug: core::ptr::null_mut(),
        coset_shift: F::coset_shift().to_canonical_u64(),
        omega: F::primitive_root_of_unity(lg_lde).to_canonical_u64(),
        n_field: F::from_canonical_usize(n).to_canonical_u64(),
        zh_evals,
        zh_invs,
        pih,
        lg_lde: lg_lde as u32,
        rate_mask: (rate - 1) as u32,
        next_step: rate as u32,
        num_constants: common.num_constants as u32,
        const_prefix: (common.selectors_info.num_selectors() + common.num_lookup_selectors) as u32,
        many_selectors: (common.selectors_info.num_selectors() > 1) as u32,
        num_routed: num_routed as u32,
        num_wires: config.num_wires as u32,
        num_prods: num_prods as u32,
        max_degree: max_degree as u32,
        off_kis: off_kis as u32,
        off_betas: off_betas as u32,
        off_gammas: off_gammas as u32,
        off_alpha: off_alpha as u32,
        num_terms: num_terms as u32,
        off_gate_data: off_gate_data as u32,
        num_gates: plan.gates.len() as u32,
        debug_stride: plan.max_constraints.max(1) as u32,
        debug_point: debug_point.map_or(-1, |p| p as i32),
    };

    let total = Instant::now();
    let wait = Instant::now();
    let mut state = range("gpu:quotient_lock_wait", || b.state.lock().unwrap_or_else(|e| e.into_inner()));
    Q_LOCK_WAIT_NS.fetch_add(wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let exec = Instant::now();
    // The wires and Z LDEs kept by this proof's commitments, when their shapes match.
    let kept = super::resident();
    let kept_lde = |i: usize, cols: usize| {
        kept.get(i).filter(|r| r.cols == cols && r.lg_rows == lg_lde).map(|r| r.ptr as *const u64)
    };
    let wires_lde = kept_lde(0, cols[1]).map_or(Lde::Upload(&wires.polynomials), Lde::Device);
    let zs_lde = kept_lde(1, cols[2]).map_or(Lde::Upload(&zs.polynomials), Lde::Device);
    if matches!((&wires_lde, &zs_lde), (Lde::Device(_), Lde::Device(_))) && debug_point.is_none() {
        Q_RESIDENT_USED.fetch_add(1, Ordering::Relaxed);
    }
    let mut result = range("gpu:quotient", || {
        device_run(
            &mut state.q,
            &mut args,
            digest,
            cs_polys,
            wires_lde,
            zs_lde,
            n,
            big_n,
            lg_n,
            rate_bits,
            &aux,
            &plan.gates,
        )
    });
    // Running out of device memory is not a device fault: drop the caches and use the CPU for
    // this quotient.
    let oom = matches!(&result, Err(e) if e.contains("out of memory"));
    if oom {
        release_device_caches(&mut state.q);
        Q_OOM.fetch_add(1, Ordering::Relaxed);
        result = Err("oom".into());
    }
    drop(state);
    if debug_point.is_none() {
        // The commitments' LDEs are not needed after the quotient.
        super::release_resident();
    }
    match result {
        Ok(out) => {
            if debug_point.is_none() {
                Q_GPU.fetch_add(1, Ordering::Relaxed);
                Q_EXEC_NS.fetch_add(exec.elapsed().as_nanos() as u64, Ordering::Relaxed);
                stage_log::quotient_event("gpu", "", exec.elapsed().as_secs_f64(), total.elapsed().as_secs_f64());
            }
            Ok((out, big_n, plan_ref.clone()))
        }
        Err(e) if e == "oom" => {
            log::warn!("plonky2 GPU quotient: out of device memory, caches dropped, this quotient on the CPU");
            Err("gpu_oom".into())
        }
        Err(e) => {
            Q_FAILED.store(true, Ordering::Relaxed);
            log::error!("plonky2 GPU quotient failed, using the CPU from now on: {e}");
            Err("gpu_error".into())
        }
    }
}

/// A batch's LDE: already on the device, or to be uploaded and extended.
enum Lde<'a, F: RichField> {
    Device(*const u64),
    Upload(&'a [PolynomialCoeffs<F>]),
}

/// Packs `polys` (canonical, column after column), uploads them and extends them into `dst`.
#[allow(clippy::too_many_arguments)]
fn upload_lde<F: RichField>(
    input: &mut Pinned,
    d_in: &mut DevBuf,
    polys: &[PolynomialCoeffs<F>],
    n: usize,
    lg_n: usize,
    rate_bits: usize,
    dst: *mut c_void,
    lap: &mut impl FnMut(&'static str),
) -> GpuResult<()> {
    let cols = polys.len();
    let host = input.ensure(cols * n)?;
    range("gpu:quotient_pack", || {
        host.par_chunks_mut(n).zip(polys.par_iter()).for_each(|(dst, p)| {
            for (d, c) in dst.iter_mut().zip(&p.coeffs) {
                *d = c.to_canonical_u64();
            }
        })
    });
    lap("q_pack");
    let d_in = d_in.ensure(cols * n)?;
    cuda(unsafe { ffi::cudaMemcpy(d_in, host.as_ptr() as *const c_void, cols * n * 8, ffi::H2D) }, "quotient H2D")?;
    lap("q_h2d");
    let cfg = ffi::NttConfig {
        batches: cols as u32,
        order: ffi::Order::NN,
        ntt_type: ffi::NttType::Standard,
        extension_rate_bits: rate_bits as u32,
        are_inputs_on_device: true,
        are_outputs_on_device: true,
        with_coset: true,
        is_multi_gpu: false,
        salt_size: 0,
    };
    range("gpu:quotient_lde", || {
        zeknox(unsafe { ffi::compute_batched_lde(0, dst, d_in, lg_n as u32, ffi::Direction::Forward, cfg) }, "quotient lde")
    })?;
    sync("quotient lde")?;
    lap("q_lde");
    Ok(())
}

/// Per-circuit constants/sigmas LDEs kept on the device (least recently used evicted), at most
/// `PLONKY2_GPU_CONST_CACHE_BYTES` (default 2 GiB). Only used under the GPU lock.
struct ConstCache {
    entries: HashMap<u64, (usize, usize, u64)>, // digest -> (ptr, bytes, last use)
    bytes: usize,
    tick: u64,
}

fn const_budget() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_CONST_CACHE_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(2 << 30)
    })
}

static CONST_CACHE: OnceLock<Mutex<ConstCache>> = OnceLock::new();
static Q_CONST_HITS: AtomicU64 = AtomicU64::new(0);
static Q_RESIDENT_USED: AtomicU64 = AtomicU64::new(0);

/// (constants cache hits, wires+Z LDEs taken from the commitments).
pub fn reuse_stats() -> (u64, u64) {
    (Q_CONST_HITS.load(Ordering::Relaxed), Q_RESIDENT_USED.load(Ordering::Relaxed))
}

/// The constants/sigmas LDE of circuit `digest`: cached, or extended now (into the cache when it
/// fits, else into `fallback`).
#[allow(clippy::too_many_arguments)]
fn constants_lde<F: RichField>(
    q: &mut QuotientBufs,
    digest: u64,
    polys: &[PolynomialCoeffs<F>],
    n: usize,
    big_n: usize,
    lg_n: usize,
    rate_bits: usize,
    lap: &mut impl FnMut(&'static str),
) -> GpuResult<*const u64> {
    let cache = CONST_CACHE.get_or_init(|| Mutex::new(ConstCache { entries: HashMap::new(), bytes: 0, tick: 0 }));
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    c.tick += 1;
    let tick = c.tick;
    if let Some(e) = c.entries.get_mut(&digest) {
        e.2 = tick;
        Q_CONST_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(e.0 as *const u64);
    }
    let bytes = polys.len() * big_n * 8;
    while !c.entries.is_empty() && c.bytes + bytes <= const_budget() && !super::device_room(bytes) {
        evict_lru(&mut c);
    }
    if bytes > const_budget() || !super::device_room(bytes) {
        let dst = q.d_cs.ensure(polys.len() * big_n)?;
        upload_lde(&mut q.input, &mut q.d_in, polys, n, lg_n, rate_bits, dst, lap)?;
        return Ok(dst as *const u64);
    }
    while c.bytes + bytes > const_budget() {
        evict_lru(&mut c);
    }
    let mut p = core::ptr::null_mut();
    cuda(unsafe { ffi::cudaMalloc(&mut p, bytes) }, "cudaMalloc (constants cache)")?;
    if let Err(e) = upload_lde(&mut q.input, &mut q.d_in, polys, n, lg_n, rate_bits, p, lap) {
        unsafe { ffi::cudaFree(p) };
        return Err(e);
    }
    c.entries.insert(digest, (p as usize, bytes, tick));
    c.bytes += bytes;
    Ok(p as *const u64)
}

fn evict_lru(c: &mut ConstCache) {
    if let Some((&old, &(ptr, b, _))) = c.entries.iter().min_by_key(|(_, e)| e.2) {
        unsafe { ffi::cudaFree(ptr as *mut c_void) };
        c.entries.remove(&old);
        c.bytes -= b;
    }
}

/// After running out of device memory: frees every cache the quotient keeps (call under the GPU
/// lock) and clears the error.
fn release_device_caches(q: &mut QuotientBufs) {
    if let Some(cache) = CONST_CACHE.get() {
        let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
        while !c.entries.is_empty() {
            evict_lru(&mut c);
        }
    }
    for b in [&mut q.d_in, &mut q.d_cs, &mut q.d_wires, &mut q.d_zs] {
        b.release();
    }
    super::trim_resident_pool();
    unsafe { ffi::cudaGetLastError() };
}

#[allow(clippy::too_many_arguments)]
fn device_run<F: RichField>(
    q: &mut QuotientBufs,
    args: &mut QuotientArgs,
    digest: u64,
    cs: &[PolynomialCoeffs<F>],
    wires: Lde<F>,
    zs: Lde<F>,
    n: usize,
    big_n: usize,
    lg_n: usize,
    rate_bits: usize,
    aux: &[u64],
    gates: &[GateDesc],
) -> GpuResult<Vec<u64>> {
    init_once()?;
    let mut t = Instant::now();
    let mut lap = |name: &'static str| {
        stage_log::detail(name, t.elapsed().as_secs_f64());
        t = Instant::now();
    };
    args.cs = constants_lde(q, digest, cs, n, big_n, lg_n, rate_bits, &mut lap)?;
    args.wires = match wires {
        Lde::Device(p) => p,
        Lde::Upload(polys) => {
            let dst = q.d_wires.ensure(polys.len() * big_n)?;
            upload_lde(&mut q.input, &mut q.d_in, polys, n, lg_n, rate_bits, dst, &mut lap)?;
            dst as *const u64
        }
    };
    args.zs = match zs {
        Lde::Device(p) => p,
        Lde::Upload(polys) => {
            let dst = q.d_zs.ensure(polys.len() * big_n)?;
            upload_lde(&mut q.input, &mut q.d_in, polys, n, lg_n, rate_bits, dst, &mut lap)?;
            dst as *const u64
        }
    };

    let d_aux = q.d_aux.ensure(aux.len())?;
    cuda(unsafe { ffi::cudaMemcpy(d_aux, aux.as_ptr() as *const c_void, aux.len() * 8, ffi::H2D) }, "quotient aux")?;
    let gate_bytes = core::mem::size_of_val(gates);
    let d_gates = q.d_gates.ensure(gate_bytes.div_ceil(8))?;
    cuda(unsafe { ffi::cudaMemcpy(d_gates, gates.as_ptr() as *const c_void, gate_bytes, ffi::H2D) }, "quotient gates")?;
    let d_out = q.d_out.ensure(NC * big_n)?;
    let debug_len = gates.len() * args.debug_stride as usize;
    let d_debug = if args.debug_point >= 0 { q.d_debug.ensure(debug_len)? } else { core::ptr::null_mut() };
    args.aux = d_aux as *const u64;
    args.gates = d_gates as *const GateDesc;
    args.out = d_out as *mut u64;
    args.debug = d_debug as *mut u64;
    range("gpu:quotient_kernel", || -> GpuResult<()> {
        cuda(unsafe { plonky2_quotient_launch(args) }, "quotient kernel launch")?;
        sync("quotient kernel")
    })?;
    lap("q_kernel");
    let (src, len) = if args.debug_point >= 0 { (d_debug, debug_len) } else { (d_out, NC * big_n) };
    let out = q.out.ensure(len)?;
    cuda(unsafe { ffi::cudaMemcpy(out.as_mut_ptr() as *mut c_void, src, len * 8, ffi::D2H) }, "quotient D2H")?;
    let out = out.to_vec();
    lap("q_d2h");
    Ok(out)
}

/// At LDE point `k`, compares every gate's unfiltered constraints computed by the kernel with the
/// CPU's and describes the differences (for check-mode mismatches).
#[allow(clippy::too_many_arguments)]
pub(crate) fn diagnose<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &[F],
    wires: &PolynomialBatch<F, C, D>,
    zs: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    k: usize,
) -> String {
    let dbg = match run(common, prover_data, public_inputs_hash, wires, zs, betas, gammas, alphas, Some(k)) {
        Ok((v, _, _)) => v,
        Err(e) => return format!("debug run failed: {e}"),
    };
    let stride = common.gates.iter().map(|g| g.0.num_constraints()).max().unwrap_or(1).max(1);
    let cs_row = prover_data.constants_sigmas_commitment.get_lde_values(k, 1);
    let prefix = common.selectors_info.num_selectors() + common.num_lookup_selectors;
    let pih = HashOut::<F>::from_partial(public_inputs_hash);
    let vars = EvaluationVarsBaseBatch::new(1, &cs_row[prefix..common.num_constants], wires.get_lde_values(k, 1), &pih);
    let mut report = Vec::new();
    for (gi, g) in common.gates.iter().enumerate() {
        let cpu = g.0.eval_unfiltered_base_batch(vars);
        let gpu = &dbg[gi * stride..gi * stride + cpu.len()];
        let bad: Vec<usize> = (0..cpu.len()).filter(|&c| cpu[c].to_canonical_u64() != gpu[c]).collect();
        if !bad.is_empty() {
            report.push(format!(
                "{}: {} of {} constraints differ, first #{} cpu {} gpu {}",
                g.0.id().chars().take(60).collect::<String>(),
                bad.len(),
                cpu.len(),
                bad[0],
                cpu[bad[0]].to_canonical_u64(),
                gpu[bad[0]]
            ));
        }
    }
    let _ = zs;
    if report.is_empty() {
        format!("point {k}: every gate's constraints match; the difference is in the vanishing terms")
    } else {
        format!("point {k}: {}", report.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::types::Field;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    /// A recursive verifier circuit uses every standard recursion gate; proving it with
    /// `PLONKY2_GPU=1 PLONKY2_GPU_QUOTIENT=1` runs the kernel, and the test build compares every
    /// GPU quotient with the CPU's (a mismatch panics with a per-gate diagnosis).
    #[test]
    fn recursive_circuit_quotient_on_gpu() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = GoldilocksField;
        if !enabled() || std::env::var("PLONKY2_GPU").ok().as_deref() != Some("1") {
            eprintln!("skipped: set PLONKY2_GPU=1 PLONKY2_GPU_QUOTIENT=1");
            return Ok(());
        }
        let config = CircuitConfig::standard_recursion_config();

        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        for _ in 0..4_000 {
            builder.add_gate(NoopGate, vec![]);
        }
        let inner = builder.build::<C>();
        let inner_proof = inner.prove(PartialWitness::new())?;

        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(&inner.common);
        pw.set_proof_with_pis_target(&pt, &inner_proof)?;
        let vd = builder.add_virtual_verifier_data(inner.common.config.fri_config.cap_height);
        pw.set_cap_target(&vd.constants_sigmas_cap, &inner.verifier_only.constants_sigmas_cap)?;
        pw.set_hash_target(vd.circuit_digest, inner.verifier_only.circuit_digest)?;
        builder.verify_proof::<C>(&pt, &vd, &inner.common);
        let outer = builder.build::<C>();
        plan(&outer.common).map_err(anyhow::Error::msg)?;

        let before = Q_GPU.load(Ordering::Relaxed);
        let reused_before = Q_RESIDENT_USED.load(Ordering::Relaxed);
        let proof = outer.prove(pw)?;
        outer.verify(proof)?;
        assert!(Q_GPU.load(Ordering::Relaxed) > before, "the outer proof's quotient did not run on the GPU");
        // With every commitment on the GPU, the quotient reads the commitments' kept LDEs.
        if backend().is_some_and(|b| b.min_lde_rows <= 1 << 10) {
            assert!(Q_RESIDENT_USED.load(Ordering::Relaxed) > reused_before, "kept LDEs were not used");
        }
        Ok(())
    }

    /// ExponentiationGate (with Arithmetic, Constant and PublicInput gates).
    #[test]
    fn exponentiation_quotient_on_gpu() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = GoldilocksField;
        if !enabled() || std::env::var("PLONKY2_GPU").ok().as_deref() != Some("1") {
            eprintln!("skipped: set PLONKY2_GPU=1 PLONKY2_GPU_QUOTIENT=1");
            return Ok(());
        }
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::new();
        for k in 0..40u64 {
            let base = builder.add_virtual_target();
            pw.set_target(base, F::from_canonical_u64(3 + k))?;
            let power = builder.exp_u64(base, 0x1234_5678_9abc + k);
            let sum = builder.add(power, base);
            builder.register_public_input(sum);
        }
        let data = builder.build::<C>();
        assert!(data.common.gates.iter().any(|g| g.0.as_any().is::<ExponentiationGate<F, D>>()));
        plan(&data.common).map_err(anyhow::Error::msg)?;
        let before = Q_GPU.load(Ordering::Relaxed);
        let proof = data.prove(pw)?;
        data.verify(proof)?;
        assert!(Q_GPU.load(Ordering::Relaxed) > before, "the quotient did not run on the GPU");
        Ok(())
    }

    #[test]
    fn psy_u32_gate_ids_parse() {
        let cmp = "ComparisonGate { num_bits: 32, num_chunks: 16, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>";
        assert_eq!(psy_u32_gate(cmp, 88), Some((kind::COMPARISON, 32, 16, 0)));
        assert_eq!(psy_u32_gate(cmp, 87), None);
        let add = "U32AddManyGate { num_addends: 3, num_ops: 5, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }";
        assert_eq!(psy_u32_gate(add, 105), Some((kind::U32_ADD_MANY, 3, 5, 18)));
        assert_eq!(psy_u32_gate("U32InterleaveGate { num_ops: 4 }", 136), Some((kind::U32_INTERLEAVE, 4, 0, 0)));
        assert_eq!(psy_u32_gate("UninterleaveToU32Gate { num_ops: 2 }", 134), Some((kind::UNINTERLEAVE_U32, 2, 0, 0)));
        assert_eq!(psy_u32_gate("U32RangeCheckGate { num_input_limbs: 8, _phantom: PhantomData<F> }", 136), Some((kind::U32_RANGE_CHECK, 8, 0, 0)));
        assert_eq!(psy_u32_gate("U32RangeCheckGate { num_input_limbs: 8 }", 135), None);
        assert_eq!(psy_u32_gate("SomeOtherGate { num_ops: 4 }", 136), None);
    }
}
