//! Permutation partial products and Z on the GPU (lab, `PLONKY2_GPU_PP=1`).
//!
//! `plonky2_partial_products` (cuda/quotient.cu) evaluates `wires_permutation_partial_products_and_zs`
//! for both challenges: per row, the chunk quotients (one inversion per row with Montgomery's
//! trick) and their product; Z as the exclusive prefix product of the row products (a scan); then
//! the partial products. Field arithmetic is exact, so the values equal the CPU's. The routed wire
//! values are read from the wires commitment's kept values when it interpolated them on the device,
//! else uploaded; each circuit's sigmas are uploaded once and kept.

use std::collections::HashMap;

use plonky2_field::extension::Extendable;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::types::{Field, PrimeField64};

use super::*;
use crate::iop::witness::MatrixWitness;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;

#[repr(C)]
struct PpArgs {
    wires: *const u64,
    sigmas: *const u64,
    k_is: *const u64,
    q: *mut u64,
    row: *mut u64,
    block: *mut u64,
    out: *mut u64,
    beta: [u64; 2],
    gamma: [u64; 2],
    omega: u64,
    lg_n: u32,
    num_routed: u32,
    degree: u32,
    num_chunks: u32,
    num_prods: u32,
}

extern "C" {
    fn plonky2_partial_products(args: *const PpArgs, stream: *mut c_void) -> i32;
}

/// Rows per block of the device scan (`PP_SCAN`).
const PP_SCAN: usize = 1024;
const PP_MAX_CHUNKS: usize = 32;

static PP_FAILED: AtomicBool = AtomicBool::new(false);
static PP_GPU: AtomicU64 = AtomicU64::new(0);
static PP_NO_ROOM: AtomicU64 = AtomicU64::new(0);
static PP_KEPT_VALUES: AtomicU64 = AtomicU64::new(0);

/// Partial products that read the wires commitment's kept values instead of uploading them.
pub(crate) fn kept_values_count() -> u64 {
    PP_KEPT_VALUES.load(Ordering::Relaxed)
}

pub(crate) fn pp_count() -> u64 {
    PP_GPU.load(Ordering::Relaxed)
}

/// Proofs that stayed on the CPU because the sigmas did not fit.
#[cfg(test)]
pub(crate) fn no_room_count() -> u64 {
    PP_NO_ROOM.load(Ordering::Relaxed)
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_PP").ok().as_deref() == Some("1"))
}

fn sigma_budget() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_SIGMA_CACHE_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(512 << 20)
    })
}

/// Each circuit's sigmas ([n][num_routed], canonical) on the device, never evicted; at most
/// `PLONKY2_GPU_SIGMA_CACHE_BYTES` (default 512 MiB), beyond which circuits stay on the CPU.
struct SigmaCache {
    entries: HashMap<u64, usize>,
    bytes: usize,
}

static SIGMAS: Mutex<Option<SigmaCache>> = Mutex::new(None);

fn sigmas_on_device<F: RichField>(digest: u64, sigmas: &[Vec<F>], num_routed: usize) -> GpuResult<Option<*const u64>> {
    let mut guard = SIGMAS.lock().unwrap_or_else(|e| e.into_inner());
    let cache = guard.get_or_insert_with(|| SigmaCache { entries: HashMap::new(), bytes: 0 });
    if let Some(&p) = cache.entries.get(&digest) {
        return Ok(Some(p as *const u64));
    }
    let bytes = sigmas.len() * num_routed * 8;
    if cache.bytes + bytes > sigma_budget() || !device_room(bytes) {
        PP_NO_ROOM.fetch_add(1, Ordering::Relaxed);
        return Ok(None);
    }
    let host: Vec<u64> = sigmas.iter().flat_map(|row| row[..num_routed].iter().map(|s| s.to_canonical_u64())).collect();
    let mut p = core::ptr::null_mut();
    cuda(unsafe { ffi::cudaMalloc(&mut p, bytes) }, "cudaMalloc (sigmas)")?;
    // From pageable memory the copy may still be in flight when cudaMemcpy returns, and the
    // partial products read the sigmas on a non-blocking stream: wait for it.
    let copied = cuda(unsafe { ffi::cudaMemcpy(p, host.as_ptr() as *const c_void, bytes, ffi::H2D) }, "sigmas H2D")
        .and_then(|()| cuda(unsafe { ffi::cudaDeviceSynchronize() }, "sigmas sync"));
    if let Err(e) = copied {
        unsafe { ffi::cudaFree(p) };
        return Err(e);
    }
    cache.entries.insert(digest, p as usize);
    cache.bytes += bytes;
    Ok(Some(p as *const u64))
}

/// The partial products and Z of every challenge, as `all_wires_permutation_partial_products`
/// returns them (per challenge: the partial products, then Z), or `None` to use the CPU.
pub(crate) fn partial_products<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
) -> Option<Vec<Vec<PolynomialValues<F>>>> {
    if !enabled() || PP_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() || common.config.num_challenges != 2 {
        return None;
    }
    backend()?;
    let n = prover_data.subgroup.len();
    let num_routed = common.config.num_routed_wires;
    let degree = common.quotient_degree_factor;
    let num_chunks = num_routed.div_ceil(degree.max(1));
    let num_prods = common.num_partial_products;
    if !n.is_power_of_two()
        || n < 2
        || n > PP_SCAN * PP_SCAN
        || degree == 0
        || num_chunks > PP_MAX_CHUNKS
        || num_prods + 1 != num_chunks
        || witness.wire_values.len() < num_routed
        || witness.wire_values[..num_routed].iter().any(|w| w.len() != n)
        || prover_data.sigmas.len() != n
        || prover_data.sigmas.iter().any(|s| s.len() < num_routed)
        || common.k_is.len() < num_routed
    {
        return None;
    }
    let lg_n = n.trailing_zeros() as usize;
    let digest = crate::plonk::config::GenericHashOut::to_vec(&prover_data.circuit_digest)[0].to_canonical_u64();
    let result = range("gpu:partial_products", || -> GpuResult<Option<Vec<u64>>> {
        let Some(sigmas) = sigmas_on_device(digest, &prover_data.sigmas, num_routed)? else {
            return Ok(None);
        };
        let wires_dev = kept(ROLE_WIRE_VALUES, common.config.num_wires, lg_n, false).map(|r| r.ptr as *const u64);
        if wires_dev.is_some() {
            PP_KEPT_VALUES.fetch_add(1, Ordering::Relaxed);
        }
        let mut ctx = PpCtx::take()?;
        let r = ctx.run(witness, wires_dev, betas, gammas, &common.k_is[..num_routed], sigmas, n, lg_n, num_routed, degree, num_chunks, num_prods);
        if r.is_ok() {
            ctx.give();
        }
        r.map(Some)
    });
    let out = match result {
        Ok(Some(v)) => v,
        Ok(None) => return None,
        Err(e) if e.contains("out of memory") => {
            unsafe { ffi::cudaGetLastError() };
            log::warn!("plonky2 GPU partial products: out of device memory, this proof on the CPU");
            return None;
        }
        Err(e) => {
            PP_FAILED.store(true, Ordering::Relaxed);
            log::error!("plonky2 GPU partial products failed, using the CPU from now on: {e}");
            return None;
        }
    };
    // Device columns: Z of each challenge, then each challenge's partial products.
    let col = |k: usize| PolynomialValues::new(out[k * n..(k + 1) * n].iter().map(|&x| F::from_canonical_u64(x)).collect());
    let res = (0..2)
        .map(|c| {
            let mut v: Vec<PolynomialValues<F>> = (0..num_prods).map(|k| col(2 + c * num_prods + k)).collect();
            v.push(col(c));
            v
        })
        .collect();
    PP_GPU.fetch_add(1, Ordering::Relaxed);
    Some(res)
}

/// A non-blocking stream with pinned staging and device scratch, reused across proofs.
struct PpCtx {
    stream: usize,
    staging: Pinned,
    out: Pinned,
    d: DevBuf,
}

// The raw pointers are only used by the thread that took the context from the pool.
unsafe impl Send for PpCtx {}

static PP_CTXS: Mutex<Vec<PpCtx>> = Mutex::new(Vec::new());

impl PpCtx {
    fn take() -> GpuResult<Self> {
        if let Some(c) = PP_CTXS.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(c);
        }
        let mut stream = core::ptr::null_mut();
        cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 1) }, "partial products stream")?;
        Ok(Self { stream: stream as usize, staging: Pinned::empty(), out: Pinned::empty(), d: DevBuf::empty() })
    }

    /// Returns the context for reuse (only after success: after an error its stream may still be
    /// busy, so it is dropped).
    fn give(self) {
        PP_CTXS.lock().unwrap_or_else(|e| e.into_inner()).push(self);
    }

    #[allow(clippy::too_many_arguments)]
    fn run<F: RichField>(
        &mut self,
        witness: &MatrixWitness<F>,
        wires_dev: Option<*const u64>,
        betas: &[F],
        gammas: &[F],
        k_is: &[F],
        sigmas: *const u64,
        n: usize,
        lg_n: usize,
        num_routed: usize,
        degree: usize,
        num_chunks: usize,
        num_prods: usize,
    ) -> GpuResult<Vec<u64>> {
        let stream = self.stream as *mut c_void;
        let blocks = n.div_ceil(PP_SCAN);
        let out_cols = 2 + 2 * num_prods;
        // Staging: the routed wire values (unless already on the device) then k_is; device: the
        // same, then the scratch and output.
        let wire_len = if wires_dev.is_some() { 0 } else { num_routed * n };
        let host = self.staging.ensure(wire_len + num_routed)?;
        if wires_dev.is_none() {
            host[..wire_len].par_chunks_mut(n).zip(witness.wire_values[..num_routed].par_iter()).for_each(|(dst, w)| {
                for (d, v) in dst.iter_mut().zip(w) {
                    *d = v.to_canonical_u64();
                }
            });
        }
        for (d, k) in host[wire_len..].iter_mut().zip(k_is) {
            *d = k.to_canonical_u64();
        }
        let up = wire_len + num_routed;
        let total = up + 2 * num_chunks * n + 2 * n + 2 * blocks + out_cols * n;
        let base = self.d.ensure(total)? as *mut u64;
        cuda(unsafe { ffi::cudaMemcpyAsync(base as *mut c_void, host.as_ptr() as *const c_void, up * 8, ffi::H2D, stream) }, "partial products H2D")?;
        let at = |off: usize| unsafe { base.add(off) };
        let args = PpArgs {
            wires: wires_dev.unwrap_or(at(0)),
            sigmas,
            k_is: at(wire_len),
            q: at(up),
            row: at(up + 2 * num_chunks * n),
            block: at(up + 2 * num_chunks * n + 2 * n),
            out: at(up + 2 * num_chunks * n + 2 * n + 2 * blocks),
            beta: [betas[0].to_canonical_u64(), betas[1].to_canonical_u64()],
            gamma: [gammas[0].to_canonical_u64(), gammas[1].to_canonical_u64()],
            omega: GoldilocksField::primitive_root_of_unity(lg_n).to_canonical_u64(),
            lg_n: lg_n as u32,
            num_routed: num_routed as u32,
            degree: degree as u32,
            num_chunks: num_chunks as u32,
            num_prods: num_prods as u32,
        };
        cuda(unsafe { plonky2_partial_products(&args, stream) }, "partial products launch")?;
        let out = self.out.ensure(out_cols * n)?;
        cuda(unsafe { ffi::cudaMemcpyAsync(out.as_mut_ptr() as *mut c_void, args.out as *const c_void, out_cols * n * 8, ffi::D2H, stream) }, "partial products D2H")?;
        cuda(unsafe { ffi::cudaStreamSynchronize(stream) }, "partial products sync")?;
        Ok(out[..out_cols * n].to_vec())
    }
}
