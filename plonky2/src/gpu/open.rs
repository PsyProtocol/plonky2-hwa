//! The proof's opening set on the GPU (lab, `PLONKY2_GPU_OPEN=1`).
//!
//! `OpeningSet::new` evaluates every committed polynomial at zeta (and the Z polynomials at
//! g zeta) from its coefficients. With the proof's LDEs on the device (wires, Z and quotient kept
//! from their commitments, constants from the cache), `plonky2_openings` evaluates them from the
//! first coset of each leaf-order LDE instead: Lagrange interpolation on the coset sH,
//!   p(z) = sum_t p(x_t) W_t,   W_t = (z^n - s^n) / (n s^n) * x_t / (z - x_t),
//! one set of weights per point and one dot product per polynomial. Field arithmetic is exact, so
//! the values equal the CPU's.

use plonky2_field::extension::{Extendable, FieldExtension};
use plonky2_field::types::{Field, PrimeField64};

use super::*;
use crate::fri::oracle::PolynomialBatch;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::GenericConfig;
use crate::plonk::proof::OpeningSet;

#[repr(C)]
struct OpenArgs {
    cols: *const *const u64,
    point: *const u32,
    weights: *mut u64,
    out: *mut u64,
    z: [[u64; 2]; 2],
    c: [[u64; 2]; 2],
    shift: u64,
    omega: u64,
    lg_n: u32,
    num_points: u32,
    num_terms: u32,
}

extern "C" {
    fn plonky2_openings(args: *const OpenArgs, stream: *mut c_void) -> i32;
}

static OPEN_FAILED: AtomicBool = AtomicBool::new(false);
static OPEN_GPU: AtomicU64 = AtomicU64::new(0);

pub(crate) fn open_count() -> u64 {
    OPEN_GPU.load(Ordering::Relaxed)
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_OPEN").ok().as_deref() == Some("1"))
}

/// The opening set as `OpeningSet::new` computes it, or `None` to use the CPU.
#[allow(clippy::too_many_arguments)]
pub(crate) fn openings<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    zeta: F::Extension,
    g: F::Extension,
    constants_sigmas: &PolynomialBatch<F, C, D>,
    wires: &PolynomialBatch<F, C, D>,
    zs: &PolynomialBatch<F, C, D>,
    quotient: &PolynomialBatch<F, C, D>,
    common: &CommonCircuitData<F, D>,
    digest: u64,
) -> Option<OpeningSet<F, D>> {
    if !enabled() || OPEN_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() || D != 2 || common.num_lookup_polys != 0 {
        return None;
    }
    backend()?;
    let batches = [constants_sigmas, wires, zs, quotient];
    let n = constants_sigmas.polynomials.first()?.len();
    if !n.is_power_of_two() || batches.iter().any(|b| b.polynomials.iter().any(|p| p.len() != n)) {
        return None;
    }
    let lg_n = n.trailing_zeros() as usize;
    let rate_bits = constants_sigmas.rate_bits;
    let lg_lde = lg_n + rate_bits;
    let big_n = n << rate_bits;
    if !own_lde(lg_n, rate_bits) || batches.iter().any(|b| b.rate_bits != rate_bits) {
        return None;
    }
    let mut base = [0usize; 4];
    for (k, role) in [ROLE_WIRES, ROLE_ZS, ROLE_QUOTIENT].into_iter().enumerate() {
        base[k + 1] = kept(role, batches[k + 1].polynomials.len(), lg_lde, true)?.ptr;
    }
    let canon = |v: F::Extension| {
        let a = v.to_basefield_array();
        [a[0].to_canonical_u64(), a[1].to_canonical_u64()]
    };
    let next = g * zeta;
    let points = [zeta, next];
    if points.iter().any(|p| canon(*p)[1] == 0) {
        return None; // a base-field point could lie on the coset
    }
    // (z^n - s^n) / (n s^n) per point.
    let sn = F::coset_shift().exp_power_of_2(lg_n);
    let scale = (F::from_canonical_usize(n) * sn).inverse();
    let mut args = OpenArgs {
        cols: core::ptr::null(),
        point: core::ptr::null(),
        weights: core::ptr::null_mut(),
        out: core::ptr::null_mut(),
        z: [canon(zeta), canon(next)],
        c: core::array::from_fn(|p| canon((points[p].exp_power_of_2(lg_n) - sn.into()) * scale.into())),
        shift: GoldilocksField::coset_shift().to_canonical_u64(),
        omega: GoldilocksField::primitive_root_of_unity(lg_n).to_canonical_u64(),
        lg_n: lg_n as u32,
        num_points: 2,
        num_terms: 0,
    };
    // Terms: every polynomial of every batch at zeta, then the Z polynomials at g zeta.
    let zs_range = common.zs_range();
    let mut terms: Vec<(usize, usize, u32)> = Vec::new();
    for (b, batch) in batches.iter().enumerate() {
        terms.extend((0..batch.polynomials.len()).map(|c| (b, c, 0)));
    }
    terms.extend(zs_range.clone().map(|c| (2, c, 1)));
    args.num_terms = terms.len() as u32;

    let result = range("gpu:openings", || {
        super::quotient::with_constants(digest, &constants_sigmas.polynomials, lg_n, rate_bits, |cs| {
            base[0] = cs as usize;
            let cols: Vec<u64> = terms.iter().map(|&(b, c, _)| (base[b] + c * big_n * 8) as u64).collect();
            let pts: Vec<u32> = terms.iter().map(|&(_, _, p)| p).collect();
            run(&mut args, &cols, &pts, n)
        })
    });
    let out = match result {
        Ok(v) => v,
        Err(e) if e.contains("out of memory") => {
            unsafe { ffi::cudaGetLastError() };
            log::warn!("plonky2 GPU openings: out of device memory, these openings on the CPU");
            return None;
        }
        Err(e) => {
            OPEN_FAILED.store(true, Ordering::Relaxed);
            log::error!("plonky2 GPU openings failed, using the CPU from now on: {e}");
            return None;
        }
    };
    let ext = |k: usize| {
        F::Extension::from_basefield_array(core::array::from_fn(|d| F::from_canonical_u64(out[2 * k + d])))
    };
    let mut k = 0usize;
    let mut take = |len: usize| {
        let v: Vec<F::Extension> = (k..k + len).map(ext).collect();
        k += len;
        v
    };
    let cs_eval = take(constants_sigmas.polynomials.len());
    let wires_eval = take(wires.polynomials.len());
    let zs_eval = take(zs.polynomials.len());
    let quotient_eval = take(quotient.polynomials.len());
    let zs_next = take(zs_range.len());
    OPEN_GPU.fetch_add(1, Ordering::Relaxed);
    Some(OpeningSet {
        constants: cs_eval[common.constants_range()].to_vec(),
        plonk_sigmas: cs_eval[common.sigmas_range()].to_vec(),
        wires: wires_eval,
        plonk_zs: zs_eval[zs_range].to_vec(),
        plonk_zs_next: zs_next,
        partial_products: zs_eval[common.partial_products_range()].to_vec(),
        quotient_polys: quotient_eval,
        lookup_zs: zs_eval[common.lookup_range()].to_vec(),
        lookup_zs_next: Vec::new(),
    })
}

/// A non-blocking stream with pinned staging and device buffers, reused across proofs.
struct OpenCtx {
    stream: usize,
    staging: Pinned,
    out: Pinned,
    d: DevBuf,
}

// The raw pointers are only used by the thread that took the context from the pool.
unsafe impl Send for OpenCtx {}

static OPEN_CTXS: Mutex<Vec<OpenCtx>> = Mutex::new(Vec::new());

fn run(args: &mut OpenArgs, cols: &[u64], pts: &[u32], n: usize) -> GpuResult<Vec<u64>> {
    super::quotient::init_once()?;
    let mut ctx = match OPEN_CTXS.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        Some(c) => c,
        None => {
            let mut stream = core::ptr::null_mut();
            cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 1) }, "openings stream")?;
            OpenCtx { stream: stream as usize, staging: Pinned::empty(), out: Pinned::empty(), d: DevBuf::empty() }
        }
    };
    let result = ctx_run(&mut ctx, args, cols, pts, n);
    if result.is_ok() {
        // After an error the stream may still be busy; such a context is dropped.
        OPEN_CTXS.lock().unwrap_or_else(|e| e.into_inner()).push(ctx);
    }
    result
}

fn ctx_run(ctx: &mut OpenCtx, args: &mut OpenArgs, cols: &[u64], pts: &[u32], n: usize) -> GpuResult<Vec<u64>> {
    let stream = ctx.stream as *mut c_void;
    let terms = cols.len();
    let pts_u64 = terms.div_ceil(2);
    // Staging: column pointers, then the point indices (u32, packed); device: the same, then the
    // weights ([2][n][2]) and the output ([terms][2]).
    let host = ctx.staging.ensure(terms + pts_u64)?;
    host[..terms].copy_from_slice(cols);
    host[terms..terms + pts_u64].fill(0);
    unsafe { core::ptr::copy_nonoverlapping(pts.as_ptr(), host[terms..].as_mut_ptr() as *mut u32, terms) };
    let up = terms + pts_u64;
    let base = ctx.d.ensure(up + 4 * n + 2 * terms)? as *mut u64;
    cuda(unsafe { ffi::cudaMemcpyAsync(base as *mut c_void, host.as_ptr() as *const c_void, up * 8, ffi::H2D, stream) }, "openings H2D")?;
    args.cols = base as *const *const u64;
    args.point = unsafe { base.add(terms) } as *const u32;
    args.weights = unsafe { base.add(up) };
    args.out = unsafe { base.add(up + 4 * n) };
    cuda(unsafe { plonky2_openings(args, stream) }, "openings launch")?;
    let out = ctx.out.ensure(2 * terms)?;
    cuda(unsafe { ffi::cudaMemcpyAsync(out.as_mut_ptr() as *mut c_void, args.out as *const c_void, 2 * terms * 8, ffi::D2H, stream) }, "openings D2H")?;
    cuda(unsafe { ffi::cudaStreamSynchronize(stream) }, "openings sync")?;
    Ok(out[..2 * terms].to_vec())
}
