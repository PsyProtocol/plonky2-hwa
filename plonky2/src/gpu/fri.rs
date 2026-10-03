//! FRI's final polynomial on the GPU (lab, `PLONKY2_GPU_FRI=1`).
//!
//! `prove_openings` reduces every opened polynomial of batch b with powers of alpha
//! (C_b = sum_j alpha^j p_j), divides by X - z_b, sums the batches (earlier ones scaled by
//! alpha^(L_b), L_b = polynomials in later batches) and extends the sum to the LDE domain with an
//! FFT. With the proof's LDEs on the device (wires, Z and quotient kept from their commitments,
//! constants from the cache), `fri_values_kernel` evaluates the same polynomial at every LDE point:
//!   V(x) = sum_b alpha^(L_b) (C_b(x) - C_b(z_b)) / (x - z_b),
//! with C_b(z_b) from the opening values. Field arithmetic is exact, so the values equal the FFT's.
//! The coefficients (degree < n) come from an inverse FFT of the first coset on the host. The
//! values come out in leaf order, which is the first commit-phase layer's order: that layer's
//! Merkle tree (`plonky2_merkle`, leaves of `arity` values) is built on the device as well.

use core::cell::RefCell;

use plonky2_field::extension::{Extendable, FieldExtension};
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::types::{Field, PrimeField64};
use plonky2_util::{log2_strict, reverse_index_bits_in_place};

use super::*;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::structure::{FriInstanceInfo, FriOpenings};
use crate::plonk::config::GenericConfig;

/// The FRI final polynomial from the GPU: LDE coefficients, values in leaf order (index bits
/// reversed), and the first commit-phase layer's tree when it was built.
pub(crate) struct GpuFinal<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub coeffs: PolynomialCoeffs<F::Extension>,
    pub leaf_values: Vec<F::Extension>,
    pub first_tree: Option<MerkleTree<F, H>>,
}

#[repr(C)]
struct FriArgs {
    cols: *const *const u64,
    coef: *const u64,
    out: *mut u64,
    shift: u64,
    omega: u64,
    z: [[u64; 2]; 2],
    cz: [[u64; 2]; 2],
    scale: [[u64; 2]; 2],
    start: [u32; 3],
    lg_lde: u32,
    num_batches: u32,
}

extern "C" {
    fn plonky2_fri_values_launch(args: *const FriArgs, stream: *mut c_void) -> i32;
}

static FRI_FAILED: AtomicBool = AtomicBool::new(false);
static FRI_GPU: AtomicU64 = AtomicU64::new(0);
static FRI_CPU: AtomicU64 = AtomicU64::new(0);
static FRI_MISMATCH: AtomicU64 = AtomicU64::new(0);
static FRI_NS: AtomicU64 = AtomicU64::new(0);

/// (final polynomials from the GPU, from the CPU while enabled, check mismatches, GPU nanoseconds).
pub fn fri_stats() -> (u64, u64, u64, u64) {
    (
        FRI_GPU.load(Ordering::Relaxed),
        FRI_CPU.load(Ordering::Relaxed),
        FRI_MISMATCH.load(Ordering::Relaxed),
        FRI_NS.load(Ordering::Relaxed),
    )
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_FRI").ok().as_deref() == Some("1"))
}

/// `PLONKY2_GPU_FRI_CHECK=1` (always in tests): also compute the CPU result and compare.
pub(crate) fn check_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    cfg!(test) || *ON.get_or_init(|| std::env::var("PLONKY2_GPU_FRI_CHECK").ok().as_deref() == Some("1"))
}

pub(crate) fn record_mismatch() {
    FRI_MISMATCH.fetch_add(1, Ordering::Relaxed);
    FRI_FAILED.store(true, Ordering::Relaxed);
    log::error!("plonky2 GPU FRI final polynomial differs from the CPU, using the CPU from now on");
}

std::thread_local! {
    /// The current proof's circuit digest (the constants cache key) and opening values, canonical
    /// (a, b) per value in FRI order; set by the prover before `prove_openings`.
    static CONTEXT: RefCell<Option<(u64, Vec<Vec<[u64; 2]>>)>> = const { RefCell::new(None) };
}

/// Records what `final_poly` needs from the prover for the next `prove_openings` on this thread.
pub(crate) fn set_context<F: RichField + Extendable<D>, const D: usize>(digest: u64, openings: &FriOpenings<F, D>) {
    if !enabled() || D != 2 {
        return;
    }
    let values = openings
        .batches
        .iter()
        .map(|b| {
            b.values
                .iter()
                .map(|v| {
                    let a = v.to_basefield_array();
                    [a[0].to_canonical_u64(), a[1].to_canonical_u64()]
                })
                .collect()
        })
        .collect();
    CONTEXT.with(|c| *c.borrow_mut() = Some((digest, values)));
}

/// Goldilocks quadratic extension (X^2 = 7) on canonical pairs, for the host-side constants.
fn emul(x: [u64; 2], y: [u64; 2]) -> [u64; 2] {
    let f = |v: u64| GoldilocksField::from_canonical_u64(v);
    let (a, b, c, d) = (f(x[0]), f(x[1]), f(y[0]), f(y[1]));
    let w = GoldilocksField::from_canonical_u64(7);
    [(a * c + w * b * d).to_canonical_u64(), (a * d + b * c).to_canonical_u64()]
}

fn eadd(x: [u64; 2], y: [u64; 2]) -> [u64; 2] {
    let f = |v: u64| GoldilocksField::from_canonical_u64(v);
    [(f(x[0]) + f(y[0])).to_canonical_u64(), (f(x[1]) + f(y[1])).to_canonical_u64()]
}

/// The FRI final polynomial as `prove_openings` computes it, or `None` when the GPU does not
/// compute it. `first_arity_bits` and `cap_height` describe the first commit-phase layer, whose
/// tree is built too when its leaves are hashed (more than 4 elements).
pub(crate) fn final_poly<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    instance: &FriInstanceInfo<F, D>,
    oracles: &[&PolynomialBatch<F, C, D>],
    alpha: F::Extension,
    rate_bits: usize,
    first_arity_bits: Option<usize>,
    cap_height: usize,
) -> Option<GpuFinal<F, C::Hasher, D>> {
    let ctx = CONTEXT.with(|c| c.borrow_mut().take());
    if !enabled() || FRI_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    let (digest, openings) = ctx?;
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() || D != 2 || oracles.len() != 4 {
        return None;
    }
    backend()?;
    let n = oracles[0].polynomials.first()?.len();
    if !n.is_power_of_two() || oracles.iter().any(|o| o.polynomials.iter().any(|p| p.len() != n)) {
        return None;
    }
    let lg_n = log2_strict(n);
    let lg_lde = lg_n + rate_bits;
    let big_n = n << rate_bits;
    if !own_lde(lg_n, rate_bits) || instance.batches.is_empty() || instance.batches.len() > 2 {
        return None;
    }
    // Kept LDEs of the wires, Z and quotient batches (oracles 1..4); constants come later.
    let roles = [ROLE_WIRES, ROLE_ZS, ROLE_QUOTIENT];
    let mut base = [0usize; 4];
    for (k, &role) in roles.iter().enumerate() {
        base[k + 1] = kept(role, oracles[k + 1].polynomials.len(), lg_lde, true)?.ptr;
    }
    if openings.len() != instance.batches.len()
        || openings.iter().zip(&instance.batches).any(|(o, b)| o.len() != b.polynomials.len())
    {
        return None;
    }
    let canon = |v: F::Extension| {
        let a = v.to_basefield_array();
        [a[0].to_canonical_u64(), a[1].to_canonical_u64()]
    };
    let alpha_c = canon(alpha);
    let max_len = instance.batches.iter().map(|b| b.polynomials.len()).max().unwrap_or(0);
    let mut pows = Vec::with_capacity(max_len + 1);
    let mut p = [1u64, 0];
    for _ in 0..=max_len {
        pows.push(p);
        p = emul(p, alpha_c);
    }
    let pow = |e: usize| {
        let mut r = [1u64, 0];
        for _ in 0..e {
            r = emul(r, alpha_c);
        }
        r
    };

    let mut args = FriArgs {
        cols: core::ptr::null(),
        coef: core::ptr::null(),
        out: core::ptr::null_mut(),
        shift: GoldilocksField::coset_shift().to_canonical_u64(),
        omega: GoldilocksField::primitive_root_of_unity(lg_lde).to_canonical_u64(),
        z: [[0; 2]; 2],
        cz: [[0; 2]; 2],
        scale: [[0; 2]; 2],
        start: [0; 3],
        lg_lde: lg_lde as u32,
        num_batches: instance.batches.len() as u32,
    };
    // Terms: (oracle, column) per opened polynomial, with alpha^j; C_b(z_b) and alpha^(L_b).
    let mut terms: Vec<(usize, usize)> = Vec::new();
    let mut coef: Vec<u64> = Vec::new();
    let mut later = 0usize;
    for (b, batch) in instance.batches.iter().enumerate().rev() {
        args.scale[b] = pow(later);
        later += batch.polynomials.len();
    }
    for (b, batch) in instance.batches.iter().enumerate() {
        let z = canon(batch.point);
        if z[1] == 0 {
            return None; // x - z could vanish on the base-field domain
        }
        args.z[b] = z;
        args.start[b] = terms.len() as u32;
        let mut cz = [0u64, 0];
        for (j, info) in batch.polynomials.iter().enumerate() {
            if info.oracle_index >= 4 || info.polynomial_index >= oracles[info.oracle_index].polynomials.len() {
                return None;
            }
            terms.push((info.oracle_index, info.polynomial_index));
            coef.extend_from_slice(&pows[j]);
            cz = eadd(cz, emul(pows[j], openings[b][j]));
        }
        args.cz[b] = cz;
    }
    args.start[instance.batches.len()] = terms.len() as u32;
    for b in instance.batches.len() + 1..3 {
        args.start[b] = terms.len() as u32;
    }

    // The first layer's tree: rows of `arity` values (2 arity elements), hashed when longer than 4.
    let poseidon = TypeId::of::<C::Hasher>() == TypeId::of::<PoseidonHash>();
    let tree_shape = first_arity_bits
        .filter(|&a| poseidon && 2usize << a > 4 && a <= lg_lde && cap_height <= lg_lde - a)
        .map(|a| (lg_lde - a, 2usize << a, cap_height));

    let t = Instant::now();
    let result = range("gpu:fri", || {
        super::quotient::with_constants(digest, &oracles[0].polynomials, lg_n, rate_bits, |cs| {
            base[0] = cs as usize;
            let cols: Vec<u64> = terms.iter().map(|&(o, c)| (base[o] + c * big_n * 8) as u64).collect();
            run(&mut args, &cols, &coef, big_n, tree_shape)
        })
    });
    let (out, digests) = match result {
        Ok(v) => v,
        Err(e) if e.contains("out of memory") => {
            unsafe { ffi::cudaGetLastError() };
            log::warn!("plonky2 GPU FRI: out of device memory, this final polynomial on the CPU");
            return None;
        }
        Err(e) => {
            FRI_FAILED.store(true, Ordering::Relaxed);
            log::error!("plonky2 GPU FRI failed, using the CPU from now on: {e}");
            return None;
        }
    };
    let ext = |k: usize| {
        F::Extension::from_basefield_array(core::array::from_fn(|d| F::from_canonical_u64(out[2 * k + d])))
    };
    let leaf_values: Vec<F::Extension> = (0..big_n).into_par_iter().map(ext).collect();
    // Leaf positions 0..n are the first coset, x = shift w_n^m at position bitrev(m): it
    // determines the polynomial (degree < n), so its inverse FFT gives the coefficients.
    let mut coset0 = leaf_values[..n].to_vec();
    reverse_index_bits_in_place(&mut coset0);
    let coeffs = PolynomialValues::new(coset0).coset_ifft(F::coset_shift().into());
    let first_tree = tree_shape.map(|(lg_rows, row_len, cap_height)| {
        let rows = 1usize << lg_rows;
        let digests_len = 2 * (rows - (1 << cap_height)) * HASH_U64;
        let (digests, cap) = rebuild_digests(&digests, digests_len, lg_rows, cap_height);
        // SAFETY: F == GoldilocksField (checked above) is repr(transparent) over u64, and H ==
        // PoseidonHash is required by the caller, so H::Hash == HashOut<GoldilocksField>.
        unsafe {
            let leaves: Vec<GoldilocksField> = cast_vec_u64(out);
            MerkleTree {
                leaves: crate::hash::leaf_matrix::LeafMatrix::from_flat(cast_vec::<GoldilocksField, F>(leaves), row_len),
                digests: cast_vec::<HashOut<GoldilocksField>, <C::Hasher as Hasher<F>>::Hash>(digests),
                cap: MerkleCap(cast_vec::<HashOut<GoldilocksField>, <C::Hasher as Hasher<F>>::Hash>(cap)),
            }
        }
    });
    FRI_GPU.fetch_add(1, Ordering::Relaxed);
    FRI_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Some(GpuFinal { coeffs: coeffs.lde(rate_bits), leaf_values, first_tree })
}

/// Reinterprets canonical u64 values as field elements without copying.
///
/// # Safety
/// Every value must be canonical.
unsafe fn cast_vec_u64(v: Vec<u64>) -> Vec<GoldilocksField> {
    let mut v = core::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(v.as_mut_ptr() as *mut GoldilocksField, v.len(), v.capacity())
}

/// Check mode: the GPU's first-layer tree against one built on the CPU from the same values.
pub(crate) fn tree_matches<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
    tree: &MerkleTree<F, H>,
    leaf_values: &[F::Extension],
    arity_bits: usize,
    cap_height: usize,
) -> bool {
    let chunked: Vec<Vec<F>> = leaf_values.chunks(1 << arity_bits).map(|c| plonky2_field::extension::flatten(c)).collect();
    let cpu = MerkleTree::<F, H>::new(chunked, cap_height);
    cpu.cap == tree.cap && cpu.digests == tree.digests
}

pub(crate) fn record_cpu() {
    if enabled() {
        FRI_CPU.fetch_add(1, Ordering::Relaxed);
    }
}

/// A non-blocking stream with pinned staging and device buffers, reused across proofs.
struct FriCtx {
    stream: usize,
    staging: Pinned,
    out: Pinned,
    digests: Pinned,
    d_terms: DevBuf,
    d_out: DevBuf,
    d_digests: DevBuf,
}

// The raw pointers are only used by the thread that took the context from the pool.
unsafe impl Send for FriCtx {}

static FRI_CTXS: Mutex<Vec<FriCtx>> = Mutex::new(Vec::new());

/// Runs the kernel (and the first layer's tree for `tree`: lg rows, row length, cap height);
/// returns the values (leaf order) and the tree's digests followed by its cap.
fn run(
    args: &mut FriArgs,
    cols: &[u64],
    coef: &[u64],
    big_n: usize,
    tree: Option<(usize, usize, usize)>,
) -> GpuResult<(Vec<u64>, Vec<u64>)> {
    super::quotient::init_once()?;
    let mut ctx = match FRI_CTXS.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        Some(c) => c,
        None => {
            let mut stream = core::ptr::null_mut();
            cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 1) }, "fri stream")?;
            FriCtx {
                stream: stream as usize,
                staging: Pinned::empty(),
                out: Pinned::empty(),
                digests: Pinned::empty(),
                d_terms: DevBuf::empty(),
                d_out: DevBuf::empty(),
                d_digests: DevBuf::empty(),
            }
        }
    };
    let result = ctx_run(&mut ctx, args, cols, coef, big_n, tree);
    if result.is_ok() {
        // After an error the stream may still be busy; such a context is dropped.
        FRI_CTXS.lock().unwrap_or_else(|e| e.into_inner()).push(ctx);
    }
    result
}

fn ctx_run(
    ctx: &mut FriCtx,
    args: &mut FriArgs,
    cols: &[u64],
    coef: &[u64],
    big_n: usize,
    tree: Option<(usize, usize, usize)>,
) -> GpuResult<(Vec<u64>, Vec<u64>)> {
    let stream = ctx.stream as *mut c_void;
    let len = cols.len() + coef.len();
    let host = ctx.staging.ensure(len)?;
    host[..cols.len()].copy_from_slice(cols);
    host[cols.len()..len].copy_from_slice(coef);
    let d_terms = ctx.d_terms.ensure(len)?;
    let d_out = ctx.d_out.ensure(2 * big_n)?;
    cuda(unsafe { ffi::cudaMemcpyAsync(d_terms, host.as_ptr() as *const c_void, len * 8, ffi::H2D, stream) }, "fri terms")?;
    args.cols = d_terms as *const *const u64;
    args.coef = unsafe { (d_terms as *const u64).add(cols.len()) };
    args.out = d_out as *mut u64;
    cuda(unsafe { plonky2_fri_values_launch(args, stream) }, "fri kernel launch")?;
    let out = ctx.out.ensure(2 * big_n)?;
    cuda(unsafe { ffi::cudaMemcpyAsync(out.as_mut_ptr() as *mut c_void, d_out, 2 * big_n * 8, ffi::D2H, stream) }, "fri D2H")?;
    let mut dig_len = 0;
    if let Some((lg_rows, row_len, cap_height)) = tree {
        // Rows of `row_len` elements are consecutive in the leaf-order output.
        let rows = 1usize << lg_rows;
        let caps = 1usize << cap_height;
        let digests_len = 2 * (rows - caps) * HASH_U64;
        dig_len = digests_len + caps * HASH_U64;
        let d_dig = ctx.d_digests.ensure(dig_len)? as *mut u64;
        let d_cap = unsafe { d_dig.add(digests_len) };
        cuda(
            unsafe { plonky2_merkle(d_dig, d_cap, d_out as *const u64, lg_rows as u32, row_len as u32, cap_height as u32, stream) },
            "fri merkle",
        )?;
        let h_dig = ctx.digests.ensure(dig_len)?;
        cuda(unsafe { ffi::cudaMemcpyAsync(h_dig.as_mut_ptr() as *mut c_void, d_dig as *const c_void, dig_len * 8, ffi::D2H, stream) }, "fri digests D2H")?;
    }
    cuda(unsafe { ffi::cudaStreamSynchronize(stream) }, "fri sync")?;
    let digests = if dig_len > 0 { ctx.digests.as_slice()[..dig_len].to_vec() } else { Vec::new() };
    Ok((out[..2 * big_n].to_vec(), digests))
}
