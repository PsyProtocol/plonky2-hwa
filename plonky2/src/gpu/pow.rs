//! FRI proof-of-work search on the GPU (lab, `PLONKY2_GPU_POW=1`).
//!
//! `pow_kernel` (in `cuda/quotient.cu`, which holds the device Poseidon constants) tries a batch of
//! consecutive candidates per launch and keeps the smallest accepting one, so the result equals
//! the CPU's ordered search (`find_first`). Batches continue from where the previous one ended;
//! after `MAX_BATCHES` without a hit, or after a device error, the CPU searches instead.
//!
//! The search does not take the GPU lock: each search borrows a context (its own non-blocking
//! stream, a device word and a pinned host word) from a small pool, so it runs alongside the
//! commitments instead of queueing behind them.

use super::*;

#[repr(C)]
struct PowArgs {
    state: [u64; 12],
    base: u64,
    best: *mut u64,
    pos: u32,
    response: u32,
    min_leading_zeros: u32,
    count: u32,
}

extern "C" {
    fn plonky2_pow_launch(args: *const PowArgs, stream: *mut c_void) -> i32;
    #[cfg(test)]
    fn plonky2_poseidon_permute_states(states: *mut u64, n: u32) -> i32;
}

/// Candidates per launch: 2^16 is the expected number of tries at 16 PoW bits.
const BATCH: u32 = 1 << 16;
const MAX_BATCHES: u64 = 1 << 10;

static POW_FAILED: AtomicBool = AtomicBool::new(false);
static POW_GPU: AtomicU64 = AtomicU64::new(0);
static POW_NS: AtomicU64 = AtomicU64::new(0);

/// (searches done on the GPU, nanoseconds spent).
pub fn pow_stats() -> (u64, u64) {
    (POW_GPU.load(Ordering::Relaxed), POW_NS.load(Ordering::Relaxed))
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_POW").ok().as_deref() == Some("1"))
}

/// The smallest candidate `c` such that permuting `state` with `c` at `pos` leaves at least
/// `min_leading_zeros` leading zeros in element `response`; `None` when the GPU does not search.
/// `state` must be canonical (Poseidon over Goldilocks, 12 elements).
pub(crate) fn search(state: &[u64; 12], pos: usize, response: usize, min_leading_zeros: u32) -> Option<u64> {
    if !enabled() || POW_FAILED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    backend()?;
    let t = Instant::now();
    let result = range("gpu:pow", || -> GpuResult<Option<u64>> {
        super::quotient::init_once()?;
        let ctx = PowCtx::take()?;
        let r = device_search(&ctx, state, pos, response, min_leading_zeros);
        if r.is_ok() {
            ctx.give();
        }
        r
    });
    match result {
        Ok(found) => {
            if found.is_some() {
                POW_GPU.fetch_add(1, Ordering::Relaxed);
                POW_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            found
        }
        Err(e) => {
            POW_FAILED.store(true, Ordering::Relaxed);
            log::error!("plonky2 GPU proof-of-work failed, using the CPU from now on: {e}");
            None
        }
    }
}

/// A non-blocking stream with a device word for the result and a pinned host word to read it.
struct PowCtx {
    stream: usize,
    d_best: usize,
    h_best: usize,
}

static POW_CTXS: Mutex<Vec<PowCtx>> = Mutex::new(Vec::new());

impl PowCtx {
    fn take() -> GpuResult<Self> {
        if let Some(c) = POW_CTXS.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(c);
        }
        let (mut stream, mut d, mut h) = (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null_mut());
        cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 1) }, "pow stream")?;
        cuda(unsafe { ffi::cudaMalloc(&mut d, 8) }, "pow device word")?;
        cuda(unsafe { ffi::cudaMallocHost(&mut h, 8) }, "pow host word")?;
        Ok(Self { stream: stream as usize, d_best: d as usize, h_best: h as usize })
    }

    /// Returns the context for reuse (only after a successful search: after an error its stream
    /// may still be busy, so it is dropped with the process).
    fn give(self) {
        POW_CTXS.lock().unwrap_or_else(|e| e.into_inner()).push(self);
    }
}

fn device_search(
    ctx: &PowCtx,
    state: &[u64; 12],
    pos: usize,
    response: usize,
    min_leading_zeros: u32,
) -> GpuResult<Option<u64>> {
    let stream = ctx.stream as *mut c_void;
    let h_best = ctx.h_best as *mut u64;
    let d_best = ctx.d_best as *mut c_void;
    unsafe { *h_best = u64::MAX };
    cuda(unsafe { ffi::cudaMemcpyAsync(d_best, h_best as *const c_void, 8, ffi::H2D, stream) }, "pow init")?;
    for batch in 0..MAX_BATCHES {
        let args = PowArgs {
            state: *state,
            base: batch * BATCH as u64,
            best: d_best as *mut u64,
            pos: pos as u32,
            response: response as u32,
            min_leading_zeros,
            count: BATCH,
        };
        cuda(unsafe { plonky2_pow_launch(&args, stream) }, "pow launch")?;
        cuda(unsafe { ffi::cudaMemcpyAsync(h_best as *mut c_void, d_best, 8, ffi::D2H, stream) }, "pow result")?;
        cuda(unsafe { ffi::cudaStreamSynchronize(stream) }, "pow sync")?;
        let best = unsafe { *h_best };
        if best != u64::MAX {
            return Ok(Some(best));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use plonky2_field::types::{Field, PrimeField64};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::hash::hashing::PlonkyPermutation;
    use crate::hash::poseidon::{Poseidon, PoseidonHash, PoseidonPermutation, SPONGE_RATE};

    fn gpu_ready() -> bool {
        enabled() && backend().is_some()
    }

    #[test]
    fn device_poseidon_matches_cpu() {
        if !gpu_ready() {
            eprintln!("skipped: set PLONKY2_GPU=1 PLONKY2_GPU_POW=1");
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9047_0101);
        let n = 64;
        let inputs: Vec<[GoldilocksField; 12]> =
            (0..n).map(|_| core::array::from_fn(|_| GoldilocksField::from_noncanonical_u64(rng.gen()))).collect();
        let flat: Vec<u64> = inputs.iter().flat_map(|s| s.iter().map(|x| x.to_canonical_u64())).collect();
        super::super::quotient::init_once().unwrap();
        let buf = DeviceBuffer::alloc(flat.len()).unwrap();
        let d = buf.ptr;
        unsafe {
            assert_eq!(ffi::cudaMemcpy(d, flat.as_ptr() as *const c_void, flat.len() * 8, ffi::H2D), 0);
            assert_eq!(plonky2_poseidon_permute_states(d as *mut u64, n as u32), 0);
            assert_eq!(ffi::cudaDeviceSynchronize(), 0);
        }
        let mut out = vec![0u64; flat.len()];
        unsafe { assert_eq!(ffi::cudaMemcpy(out.as_mut_ptr() as *mut c_void, d, out.len() * 8, ffi::D2H), 0) };
        drop(buf);
        for (k, input) in inputs.iter().enumerate() {
            let expected = GoldilocksField::poseidon(*input);
            for i in 0..12 {
                assert_eq!(out[12 * k + i], expected[i].to_canonical_u64(), "state {k} element {i}");
            }
        }
    }

    #[test]
    fn device_search_matches_cpu_first() {
        if !gpu_ready() {
            eprintln!("skipped: set PLONKY2_GPU=1 PLONKY2_GPU_POW=1");
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9047_0102);
        for case in 0..12 {
            let state: [GoldilocksField; 12] = core::array::from_fn(|_| GoldilocksField::from_noncanonical_u64(rng.gen()));
            let pos = case % SPONGE_RATE;
            let bits = [4, 8, 12, 16][case % 4];
            let canonical: [u64; 12] = core::array::from_fn(|i| state[i].to_canonical_u64());
            let gpu = search(&canonical, pos, SPONGE_RATE - 1, bits).expect("GPU search");
            let perm = PoseidonPermutation::new(state.iter().copied());
            let cpu = crate::fri::prover::pow_search_scalar::<GoldilocksField, PoseidonHash>(perm, pos, bits, true).unwrap();
            assert_eq!(gpu, cpu, "case {case}: {bits} bits");
        }
    }
}
