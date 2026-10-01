//! Optional CUDA backend for polynomial commitments (coset LDE -> transpose + bit reversal ->
//! Poseidon Merkle tree), built on the zeknox kernels (feature `cuda`).
//!
//! Runtime control (all off unless `PLONKY2_GPU=1`):
//! - `PLONKY2_GPU=1` enables the backend; anything else keeps every commitment on the CPU.
//! - `PLONKY2_GPU_MIN_LDE_ROWS` (default 2^20): a commitment goes to the GPU only if its LDE
//!   (extended) row count `2^(degree_bits + rate_bits)` is at least this many rows.
//! - Device selection is `CUDA_VISIBLE_DEVICES`; the backend always uses device 0 of what is
//!   visible.
//!
//! Correctness contract (tests in the GPU lab): outputs are bit-identical to the CPU path. Inputs
//! are canonicalised on the CPU before upload because zeknox's NTT mis-reduces values >= p.
//!
//! Concurrency: a commitment first claims an exclusive output buffer set (pinned leaves + digests)
//! from a bounded pool (`PLONKY2_GPU_OUTPUT_BUFFERS`, default 2), then takes the process-wide GPU
//! lock for upload, kernels and the synchronous download into its own set. The lock is released
//! once the device work is complete; the plonky2 `MerkleTree` is rebuilt outside the lock from the
//! claimed set, which returns to the pool only after the rebuild (a guard also returns it on
//! panic). Later commitments therefore never overwrite data still being rebuilt, and pinned host
//! memory is bounded by pool size x the largest commitment seen. The lock does not coordinate with
//! other processes (another proxy, or the Go Groth16 backend) sharing the same GPU.

use std::string::String;
use std::vec::Vec;
use core::any::TypeId;
use core::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Instant;

use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::hash::poseidon::PoseidonHash;
use crate::plonk::config::Hasher;
use crate::util::stage_log;

const P: u64 = 0xffff_ffff_0000_0001;
const HASH_U64: usize = 4;
/// Twiddles are precomputed up to this NTT size; larger commitments stay on the CPU.
const MAX_LG: usize = 21;

mod ffi {
    use super::*;

    #[repr(C)]
    pub struct RustError {
        pub code: i32,
        pub message: *mut c_char,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum Order {
        NN = 0,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum Direction {
        Forward = 0,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum NttType {
        Standard = 0,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct NttConfig {
        pub batches: u32,
        pub order: Order,
        pub ntt_type: NttType,
        pub extension_rate_bits: u32,
        pub are_inputs_on_device: bool,
        pub are_outputs_on_device: bool,
        pub with_coset: bool,
        pub is_multi_gpu: bool,
        pub salt_size: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct TransposeConfig {
        pub batches: u32,
        pub are_inputs_on_device: bool,
        pub are_outputs_on_device: bool,
    }

    extern "C" {
        pub fn init_twiddle_factors(device_id: usize, lg_n: usize) -> RustError;
        pub fn init_coset(device_id: usize, lg_n: usize, coset_gen: u64) -> RustError;
        pub fn compute_batched_lde(
            device_id: usize,
            output: *mut c_void,
            input: *mut c_void,
            lg_domain_size: u32,
            direction: Direction,
            cfg: NttConfig,
        ) -> RustError;
        pub fn compute_transpose_rev(
            device_id: usize,
            output: *mut c_void,
            input: *mut c_void,
            lg_n: u32,
            cfg: TransposeConfig,
        ) -> RustError;
        pub fn fill_digests_buf_linear_gpu_with_gpu_ptr(
            digests_buf_gpu_ptr: *mut c_void,
            cap_buf_gpu_ptr: *mut c_void,
            leaves_buf_gpu_ptr: *mut c_void,
            digests_buf_size: u64,
            cap_buf_size: u64,
            leaves_buf_size: u64,
            leaf_size: u64,
            cap_height: u64,
            hash_type: u64,
            gpu_id: u64,
        );

        pub fn cudaSetDevice(device: i32) -> i32;
        pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaGetLastError() -> i32;
        pub fn cudaGetErrorString(err: i32) -> *const c_char;
        pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
    }

    pub const H2D: i32 = 1;
    pub const D2H: i32 = 2;
}

type GpuResult<T> = Result<T, String>;

extern "C" {
    fn plonky2_nvtx_push(name: *const c_char);
    fn plonky2_nvtx_pop();
}

/// `PLONKY2_NVTX=1`: emit NVTX ranges (for Nsight Systems) around prover stages and GPU steps.
pub fn nvtx_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_NVTX").ok().as_deref() == Some("1"))
}

/// Runs `f` inside an NVTX range named `name` when NVTX is enabled.
pub fn range<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !nvtx_enabled() {
        return f();
    }
    std::thread_local! {
        static NAMES: core::cell::RefCell<std::collections::HashMap<&'static str, std::ffi::CString>> =
            core::cell::RefCell::new(std::collections::HashMap::new());
    }
    NAMES.with(|n| {
        let mut n = n.borrow_mut();
        let c = n.entry(name).or_insert_with(|| std::ffi::CString::new(name).unwrap());
        unsafe { plonky2_nvtx_push(c.as_ptr()) };
    });
    let r = f();
    unsafe { plonky2_nvtx_pop() };
    r
}

fn cuda(code: i32, what: &str) -> GpuResult<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(code)) }.to_string_lossy();
    Err(format!("{what}: CUDA error {code}: {msg}"))
}

fn zeknox(err: ffi::RustError, what: &str) -> GpuResult<()> {
    if err.code == 0 {
        return Ok(());
    }
    let msg = if err.message.is_null() {
        String::new()
    } else {
        let m = unsafe { CStr::from_ptr(err.message) }.to_string_lossy().into_owned();
        unsafe { libc_free(err.message as *mut c_void) };
        m
    };
    Err(format!("{what}: zeknox error {}: {msg}", err.code))
}

extern "C" {
    #[link_name = "free"]
    fn libc_free(p: *mut c_void);
}

/// Lab-only fault injection (feature `gpu-fault-injection`): `PLONKY2_GPU_FAULT=run_error` makes
/// the device phase fail after an output set was claimed; `rebuild_panic` panics during the
/// rebuild outside the GPU lock. `PLONKY2_GPU_FAULT_COUNT` (default 1) limits how many times.
#[cfg(feature = "gpu-fault-injection")]
fn inject(kind: &str) -> bool {
    static LEFT: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
    if std::env::var("PLONKY2_GPU_FAULT").ok().as_deref() != Some(kind) {
        return false;
    }
    let left = LEFT.get_or_init(|| {
        std::sync::atomic::AtomicI64::new(std::env::var("PLONKY2_GPU_FAULT_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
    });
    left.fetch_sub(1, Ordering::SeqCst) > 0
}

fn sync(what: &str) -> GpuResult<()> {
    cuda(unsafe { ffi::cudaDeviceSynchronize() }, what)?;
    cuda(unsafe { ffi::cudaGetLastError() }, what)
}

struct DeviceBuffer {
    ptr: *mut c_void,
}

impl DeviceBuffer {
    fn alloc(len_u64: usize) -> GpuResult<Self> {
        let mut ptr = core::ptr::null_mut();
        cuda(unsafe { ffi::cudaMalloc(&mut ptr, len_u64.max(1) * 8) }, "cudaMalloc")?;
        Ok(Self { ptr })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.ptr) };
    }
}

/// Growable page-locked host buffer.
struct Pinned {
    ptr: *mut u64,
    len: usize,
}

// Only touched while holding the backend lock.
unsafe impl Send for Pinned {}

impl Pinned {
    const fn empty() -> Self {
        Self { ptr: core::ptr::null_mut(), len: 0 }
    }

    fn ensure(&mut self, len: usize) -> GpuResult<&mut [u64]> {
        if self.len < len {
            self.release();
            let mut p = core::ptr::null_mut();
            cuda(unsafe { ffi::cudaMallocHost(&mut p, len * 8) }, "cudaMallocHost")?;
            self.ptr = p as *mut u64;
            self.len = len;
            let now = PINNED_BYTES.fetch_add(len as u64 * 8, Ordering::Relaxed) + len as u64 * 8;
            PINNED_PEAK.fetch_max(now, Ordering::Relaxed);
            log::info!("plonky2 GPU pinned host buffers now {now} bytes (grew one buffer to {} bytes)", len * 8);
        }
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr, len) })
    }

    fn release(&mut self) {
        if !self.ptr.is_null() {
            PINNED_BYTES.fetch_sub(self.len as u64 * 8, Ordering::Relaxed);
            unsafe { ffi::cudaFreeHost(self.ptr as *mut c_void) };
            self.ptr = core::ptr::null_mut();
            self.len = 0;
        }
    }
}

/// Pinned input staging, used only under the GPU lock.
struct State {
    input: Pinned,
}

/// One commitment's pinned output: row-major leaves and (digests || cap).
struct OutputSet {
    leaves: Pinned,
    digests: Pinned,
}

/// Bounded pool of output sets; a claimed set is owned by one commitment until it is rebuilt.
struct Pool {
    size: usize,
    free: Mutex<Vec<OutputSet>>,
    returned: Condvar,
}

struct OutputGuard {
    set: Option<OutputSet>,
    pool: &'static Pool,
}

impl Pool {
    fn acquire(&'static self) -> OutputGuard {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(set) = free.pop() {
                return OutputGuard { set: Some(set), pool: self };
            }
            free = self.returned.wait(free).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for OutputGuard {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            self.pool.free.lock().unwrap_or_else(|e| e.into_inner()).push(set);
            self.pool.returned.notify_one();
        }
    }
}

struct Backend {
    min_lde_rows: usize,
    state: Mutex<State>,
    pool: Pool,
}

static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();
/// Set after any device error; the rest of the process stays on the CPU.
static FAILED: AtomicBool = AtomicBool::new(false);

static GPU_COMMITS: AtomicU64 = AtomicU64::new(0);
static CPU_COMMITS: AtomicU64 = AtomicU64::new(0);
static GPU_ERRORS: AtomicU64 = AtomicU64::new(0);
static LOCK_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static BUF_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static PINNED_BYTES: AtomicU64 = AtomicU64::new(0);
static PINNED_PEAK: AtomicU64 = AtomicU64::new(0);
static EXEC_NS: AtomicU64 = AtomicU64::new(0);
static REBUILD_NS: AtomicU64 = AtomicU64::new(0);

/// Process-wide commitment counters of the CUDA backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuStats {
    pub gpu_commits: u64,
    pub cpu_commits: u64,
    pub gpu_errors: u64,
    pub buffer_wait_ns: u64,
    pub lock_wait_ns: u64,
    pub exec_ns: u64,
    pub rebuild_ns: u64,
    /// Page-locked host memory currently held by the backend, and its peak.
    pub pinned_bytes: u64,
    pub pinned_peak_bytes: u64,
    /// Output buffer sets: pool size and how many are free right now.
    pub pool_size: u64,
    pub pool_free: u64,
}

pub fn stats() -> GpuStats {
    GpuStats {
        gpu_commits: GPU_COMMITS.load(Ordering::Relaxed),
        cpu_commits: CPU_COMMITS.load(Ordering::Relaxed),
        gpu_errors: GPU_ERRORS.load(Ordering::Relaxed),
        buffer_wait_ns: BUF_WAIT_NS.load(Ordering::Relaxed),
        lock_wait_ns: LOCK_WAIT_NS.load(Ordering::Relaxed),
        exec_ns: EXEC_NS.load(Ordering::Relaxed),
        rebuild_ns: REBUILD_NS.load(Ordering::Relaxed),
        pinned_bytes: PINNED_BYTES.load(Ordering::Relaxed),
        pinned_peak_bytes: PINNED_PEAK.load(Ordering::Relaxed),
        pool_size: BACKEND.get().and_then(|b| b.as_ref()).map_or(0, |b| b.pool.size as u64),
        pool_free: BACKEND.get().and_then(|b| b.as_ref()).map_or(0, |b| b.pool.free.lock().map_or(0, |f| f.len() as u64)),
    }
}

pub(crate) fn count_cpu_commit() {
    CPU_COMMITS.fetch_add(1, Ordering::Relaxed);
}

fn backend() -> Option<&'static Backend> {
    BACKEND
        .get_or_init(|| {
            if std::env::var("PLONKY2_GPU").ok().as_deref() != Some("1") {
                return None;
            }
            let min_lde_rows = std::env::var("PLONKY2_GPU_MIN_LDE_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize << 20);
            let buffers = std::env::var("PLONKY2_GPU_OUTPUT_BUFFERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n: &usize| n >= 1)
                .unwrap_or(2);
            match init_device() {
                Ok(()) => {
                    log::info!(
                        "plonky2 GPU backend enabled: device 0 of CUDA_VISIBLE_DEVICES={:?}, min LDE rows {min_lde_rows}, \
                         {buffers} output buffer sets, twiddles up to 2^{MAX_LG}",
                        std::env::var("CUDA_VISIBLE_DEVICES").ok()
                    );
                    Some(Backend {
                        min_lde_rows,
                        state: Mutex::new(State { input: Pinned::empty() }),
                        pool: Pool {
                            size: buffers,
                            free: Mutex::new(
                                (0..buffers).map(|_| OutputSet { leaves: Pinned::empty(), digests: Pinned::empty() }).collect(),
                            ),
                            returned: Condvar::new(),
                        },
                    })
                }
                Err(e) => {
                    log::error!("plonky2 GPU backend disabled, initialisation failed: {e}");
                    None
                }
            }
        })
        .as_ref()
}

fn init_device() -> GpuResult<()> {
    use plonky2_field::types::{Field, PrimeField64};
    cuda(unsafe { ffi::cudaSetDevice(0) }, "cudaSetDevice")?;
    for lg in 2..=MAX_LG {
        zeknox(unsafe { ffi::init_twiddle_factors(0, lg) }, "init_twiddle_factors")?;
    }
    zeknox(unsafe { ffi::init_coset(0, MAX_LG, GoldilocksField::coset_shift().to_canonical_u64()) }, "init_coset")?;
    sync("init")
}

/// Why a commitment stayed on the CPU (recorded in the stage log).
pub(crate) fn cpu_reason<F: RichField, H: Hasher<F>>(lg_n: usize, rate_bits: usize, cols: usize, blinding: bool) -> &'static str {
    if FAILED.load(Ordering::Relaxed) {
        return "gpu_failed_earlier";
    }
    let Some(b) = backend() else { return "gpu_disabled" };
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() || TypeId::of::<H>() != TypeId::of::<PoseidonHash>() {
        return "unsupported_field_or_hash";
    }
    if blinding {
        return "blinding_salt";
    }
    if cols <= HASH_U64 {
        return "leaf_too_small";
    }
    if lg_n < 2 || lg_n + rate_bits > MAX_LG {
        return "size_out_of_range";
    }
    if (1usize << (lg_n + rate_bits)) < b.min_lde_rows {
        return "below_threshold";
    }
    ""
}

/// Commits to `polynomials` on the GPU when eligible. `Err(reason)` means the caller must use the
/// CPU path; the reason is recorded in the stage log by the caller.
pub(crate) fn try_commit<F: RichField, H: Hasher<F>>(
    polynomials: &[PolynomialCoeffs<F>],
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
) -> Result<MerkleTree<F, H>, &'static str> {
    let total = Instant::now();
    let cols = polynomials.len();
    let n = polynomials.first().map_or(0, |p| p.len());
    if n == 0 || !n.is_power_of_two() || polynomials.iter().any(|p| p.len() != n) {
        return Err("irregular_input");
    }
    let lg_n = n.trailing_zeros() as usize;
    let reason = cpu_reason::<F, H>(lg_n, rate_bits, cols, blinding);
    if !reason.is_empty() {
        return Err(reason);
    }
    if cap_height > lg_n + rate_bits {
        return Err("cap_too_high");
    }
    let b: &'static Backend = backend().expect("checked by cpu_reason");

    // 1. Claim an exclusive output set (bounded pool).
    let buf_wait_start = Instant::now();
    let mut out = range("gpu:buffer_wait", || b.pool.acquire());
    let buf_wait = buf_wait_start.elapsed().as_secs_f64();

    // 2. GPU lock: upload, kernels, synchronous download into the claimed set.
    let wait = Instant::now();
    let mut state = range("gpu:lock_wait", || b.state.lock().unwrap_or_else(|e| e.into_inner()));
    let lock_wait = wait.elapsed().as_secs_f64();
    let result = range("gpu:device", || run(&mut state.input, out.set.as_mut().expect("claimed"), polynomials, lg_n, rate_bits, cap_height));
    // 3. Device work is complete (synchronous copies + device sync); the set is ours alone.
    drop(state);

    match result {
        Ok((leaves_flat_len, digests_len, cap_len, exec_s)) => {
            // 4. Rebuild outside the GPU lock; the set returns to the pool when `out` drops.
            let rebuild = Instant::now();
            let tree = range("gpu:rebuild", || {
                rebuild_tree::<F, H>(out.set.as_ref().expect("claimed"), cols, lg_n + rate_bits, cap_height, leaves_flat_len, digests_len, cap_len)
            });
            drop(out);
            let rebuild_s = rebuild.elapsed().as_secs_f64();
            stage_log::commit_event("gpu", "", lg_n, rate_bits, cols, buf_wait, lock_wait, exec_s, rebuild_s, total.elapsed().as_secs_f64());
            BUF_WAIT_NS.fetch_add((buf_wait * 1e9) as u64, Ordering::Relaxed);
            GPU_COMMITS.fetch_add(1, Ordering::Relaxed);
            LOCK_WAIT_NS.fetch_add((lock_wait * 1e9) as u64, Ordering::Relaxed);
            EXEC_NS.fetch_add((exec_s * 1e9) as u64, Ordering::Relaxed);
            REBUILD_NS.fetch_add((rebuild_s * 1e9) as u64, Ordering::Relaxed);
            Ok(tree)
        }
        Err(e) => {
            drop(out);
            FAILED.store(true, Ordering::Relaxed);
            GPU_ERRORS.fetch_add(1, Ordering::Relaxed);
            log::error!("plonky2 GPU commitment failed, falling back to CPU for the rest of the process: {e}");
            Err("gpu_error")
        }
    }
}

/// Upload, LDE, transpose, Merkle, download into the pinned buffers. Returns the element counts
/// written (leaves, digests, cap, in u64) and the execution time.
fn run<F: RichField>(
    input_staging: &mut Pinned,
    out: &mut OutputSet,
    polynomials: &[PolynomialCoeffs<F>],
    lg_n: usize,
    rate_bits: usize,
    cap_height: usize,
) -> GpuResult<(usize, usize, usize, f64)> {
    let t = Instant::now();
    let cols = polynomials.len();
    let n = 1usize << lg_n;
    let lg_rows = lg_n + rate_bits;
    let rows = 1usize << lg_rows;
    let caps = 1usize << cap_height;
    let num_digests = 2 * (rows - caps);

    #[cfg(feature = "gpu-fault-injection")]
    if inject("run_error") {
        return Err("injected run_error".into());
    }

    // Canonicalise into the pinned input buffer (CPU, see module docs).
    let input = input_staging.ensure(cols * n)?;
    range("gpu:canonicalize", || {
        input.par_chunks_mut(n).zip(polynomials.par_iter()).for_each(|(dst, p)| {
            for (d, c) in dst.iter_mut().zip(&p.coeffs) {
                let x = plonky2_field::types::PrimeField64::to_noncanonical_u64(c);
                *d = if x >= P { x - P } else { x };
            }
        })
    });

    let d_in = DeviceBuffer::alloc(cols * n)?;
    let d_lde = DeviceBuffer::alloc(cols * rows)?;
    range("gpu:h2d", || cuda(unsafe { ffi::cudaMemcpy(d_in.ptr, input.as_ptr() as *const c_void, cols * n * 8, ffi::H2D) }, "H2D"))?;

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
    range("gpu:lde", || {
        zeknox(unsafe { ffi::compute_batched_lde(0, d_lde.ptr, d_in.ptr, lg_n as u32, ffi::Direction::Forward, cfg) }, "lde")?;
        sync("lde")
    })?;
    drop(d_in);

    let d_leaves = DeviceBuffer::alloc(cols * rows)?;
    let tcfg = ffi::TransposeConfig { batches: cols as u32, are_inputs_on_device: true, are_outputs_on_device: true };
    range("gpu:transpose", || {
        zeknox(unsafe { ffi::compute_transpose_rev(0, d_leaves.ptr, d_lde.ptr, lg_rows as u32, tcfg) }, "transpose")?;
        sync("transpose")
    })?;
    drop(d_lde);

    let d_digests = DeviceBuffer::alloc(num_digests * HASH_U64)?;
    let d_cap = DeviceBuffer::alloc(caps * HASH_U64)?;
    range("gpu:merkle", || {
        unsafe {
            ffi::fill_digests_buf_linear_gpu_with_gpu_ptr(
                d_digests.ptr,
                d_cap.ptr,
                d_leaves.ptr,
                num_digests as u64,
                caps as u64,
                rows as u64,
                cols as u64,
                cap_height as u64,
                0,
                0,
            );
        }
        sync("merkle")
    })?;

    let _d2h = NvtxScope::new("gpu:d2h");
    let leaves = out.leaves.ensure(cols * rows)?;
    cuda(unsafe { ffi::cudaMemcpy(leaves.as_mut_ptr() as *mut c_void, d_leaves.ptr, cols * rows * 8, ffi::D2H) }, "D2H leaves")?;
    let digests = out.digests.ensure((num_digests + caps) * HASH_U64)?;
    let (dg, cp) = digests.split_at_mut(num_digests * HASH_U64);
    if num_digests > 0 {
        cuda(unsafe { ffi::cudaMemcpy(dg.as_mut_ptr() as *mut c_void, d_digests.ptr, num_digests * HASH_U64 * 8, ffi::D2H) }, "D2H digests")?;
    }
    cuda(unsafe { ffi::cudaMemcpy(cp.as_mut_ptr() as *mut c_void, d_cap.ptr, caps * HASH_U64 * 8, ffi::D2H) }, "D2H cap")?;
    sync("download")?;
    Ok((cols * rows, num_digests * HASH_U64, caps * HASH_U64, t.elapsed().as_secs_f64()))
}

/// NVTX range for the rest of the enclosing scope.
struct NvtxScope(bool);

impl NvtxScope {
    fn new(name: &'static str) -> Self {
        if !nvtx_enabled() {
            return Self(false);
        }
        let c = std::ffi::CString::new(name).unwrap();
        unsafe { plonky2_nvtx_push(c.as_ptr()) };
        Self(true)
    }
}

impl Drop for NvtxScope {
    fn drop(&mut self) {
        if self.0 {
            unsafe { plonky2_nvtx_pop() };
        }
    }
}

/// Builds the plonky2 `MerkleTree` from the pinned buffers: row-major leaves, zeknox's linear
/// digest layout (per cap subtree, node k has children 2k+2 and 2k+3) converted to plonky2's
/// recursive layout, and the cap.
fn rebuild_tree<F: RichField, H: Hasher<F>>(
    state: &OutputSet,
    cols: usize,
    lg_rows: usize,
    cap_height: usize,
    leaves_len: usize,
    digests_len: usize,
    cap_len: usize,
) -> MerkleTree<F, H> {
    debug_assert_eq!(TypeId::of::<F>(), TypeId::of::<GoldilocksField>());
    debug_assert_eq!(TypeId::of::<H>(), TypeId::of::<PoseidonHash>());
    #[cfg(feature = "gpu-fault-injection")]
    if inject("rebuild_panic") {
        panic!("injected rebuild_panic");
    }
    let leaves_flat = unsafe { core::slice::from_raw_parts(state.leaves.ptr, leaves_len) };
    let digest_flat = unsafe { core::slice::from_raw_parts(state.digests.ptr, digests_len + cap_len) };
    let (lin_flat, cap_flat) = digest_flat.split_at(digests_len);

    let to_hash = |c: &[u64]| -> HashOut<GoldilocksField> {
        HashOut { elements: [GoldilocksField(c[0]), GoldilocksField(c[1]), GoldilocksField(c[2]), GoldilocksField(c[3])] }
    };

    let leaves: Vec<Vec<GoldilocksField>> =
        leaves_flat.par_chunks(cols).map(|r| r.iter().map(|&x| GoldilocksField(x)).collect()).collect();

    let caps = 1usize << cap_height;
    let subtree_leaves = 1usize << (lg_rows - cap_height);
    let per_subtree = if subtree_leaves > 1 { 2 * subtree_leaves - 2 } else { 0 };
    let mut digests = std::vec![HashOut::<GoldilocksField>::default(); caps * per_subtree];
    if per_subtree > 0 {
        digests.par_chunks_mut(per_subtree).enumerate().for_each(|(s, out)| {
            let lin = &lin_flat[s * per_subtree * HASH_U64..(s + 1) * per_subtree * HASH_U64];
            linear_to_recursive(out, lin, 0, &to_hash);
        });
    }
    let cap: Vec<HashOut<GoldilocksField>> = cap_flat.chunks_exact(HASH_U64).map(to_hash).collect();

    // SAFETY: F == GoldilocksField and H == PoseidonHash (checked by `cpu_reason`), so
    // H::Hash == HashOut<GoldilocksField>; the casts are identities.
    unsafe {
        MerkleTree {
            leaves: cast_vec::<Vec<GoldilocksField>, Vec<F>>(leaves),
            digests: cast_vec::<HashOut<GoldilocksField>, H::Hash>(digests),
            cap: MerkleCap(cast_vec::<HashOut<GoldilocksField>, H::Hash>(cap)),
        }
    }
}

/// `out` is one subtree in plonky2's layout (left subtree || left digest || right digest || right
/// subtree), `lin` the same subtree in zeknox's heap layout, `left` the heap index of the left
/// child of the subtree's root.
fn linear_to_recursive<T: Send + Copy>(out: &mut [T], lin: &[u64], left: usize, to_hash: &(impl Fn(&[u64]) -> T + Sync)) {
    let half = out.len() / 2;
    out[half - 1] = to_hash(&lin[left * HASH_U64..(left + 1) * HASH_U64]);
    out[half] = to_hash(&lin[(left + 1) * HASH_U64..(left + 2) * HASH_U64]);
    if out.len() > 2 {
        let (lo, rest) = out.split_at_mut(half - 1);
        let hi = &mut rest[2..];
        if lo.len() >= 1 << 12 {
            join(
                || linear_to_recursive(lo, lin, 2 * left + 2, to_hash),
                || linear_to_recursive(hi, lin, 2 * (left + 1) + 2, to_hash),
            );
        } else {
            linear_to_recursive(lo, lin, 2 * left + 2, to_hash);
            linear_to_recursive(hi, lin, 2 * (left + 1) + 2, to_hash);
        }
    }
}

/// Reinterprets a `Vec<A>` as `Vec<B>` for identical types `A == B`.
unsafe fn cast_vec<A: 'static, B: 'static>(v: Vec<A>) -> Vec<B> {
    assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
    let mut v = core::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(v.as_mut_ptr() as *mut B, v.len(), v.capacity())
}

#[cfg(test)]
mod pool_tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    fn pool(n: usize) -> &'static Pool {
        Box::leak(Box::new(Pool {
            size: n,
            free: Mutex::new((0..n).map(|_| OutputSet { leaves: Pinned::empty(), digests: Pinned::empty() }).collect()),
            returned: Condvar::new(),
        }))
    }

    fn free(p: &Pool) -> usize {
        p.free.lock().unwrap().len()
    }

    #[test]
    fn capacity_is_bounded_and_waiter_is_woken_on_return() {
        let p = pool(2);
        let a = p.acquire();
        let b = p.acquire();
        assert_eq!(free(p), 0);
        let (tx, rx) = mpsc::channel();
        let t = std::thread::spawn(move || {
            let _c = p.acquire();
            tx.send(()).unwrap();
        });
        // The third claimer must wait while both sets are out.
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
        drop(a);
        rx.recv_timeout(Duration::from_secs(5)).expect("waiter woken after a return");
        t.join().unwrap();
        drop(b);
        assert_eq!(free(p), 2);
    }

    #[test]
    fn panic_while_holding_a_set_returns_it_and_wakes_waiter() {
        let p = pool(1);
        let (tx, rx) = mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _g = p.acquire();
            tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(100));
            panic!("simulated rebuild panic");
        });
        rx.recv().unwrap();
        let waiter = std::thread::spawn(move || {
            let _g = p.acquire();
        });
        assert!(holder.join().is_err(), "holder panicked");
        waiter.join().expect("waiter got the set after the panic");
        assert_eq!(free(p), 1);
    }

    #[test]
    fn returning_a_set_does_not_need_the_gpu_lock() {
        let p = pool(1);
        let gpu_lock: &'static Mutex<()> = Box::leak(Box::new(Mutex::new(())));
        let g = p.acquire();
        let held = gpu_lock.lock().unwrap();
        // Return the set from another thread while the "GPU lock" is held here.
        let t = std::thread::spawn(move || drop(g));
        t.join().unwrap();
        drop(held);
        assert_eq!(free(p), 1);
    }
}
