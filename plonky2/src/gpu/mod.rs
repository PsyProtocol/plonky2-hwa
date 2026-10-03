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
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::hash::poseidon::PoseidonHash;
use crate::plonk::config::Hasher;
use crate::util::stage_log;

pub(crate) mod fri;
mod leaf_pool;
pub(crate) mod open;
pub(crate) mod perm;
pub(crate) mod pow;
pub(crate) mod quotient;

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
        pub fn cudaMemcpy2D(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaGetLastError() -> i32;
        pub fn cudaGetErrorString(err: i32) -> *const c_char;
        pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
        pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        pub fn cudaSetDeviceFlags(flags: u32) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    }

    pub const H2D: i32 = 1;
    pub const D2H: i32 = 2;
}

type GpuResult<T> = Result<T, String>;

/// `PLONKY2_GPU_MERKLE=zeknox`: build commitment Merkle trees with zeknox's kernels (heap layout,
/// reordered afterwards) instead of `plonky2_merkle` (about 1.9x faster, same digests).
fn zeknox_merkle() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("PLONKY2_GPU_MERKLE").ok().as_deref() == Some("zeknox"))
}

/// Coset LDEs with `plonky2_coset_lde` (column-major, in leaf order; with the transpose 3.5-4x
/// faster than zeknox's LDE plus `compute_transpose_rev`) for `lg_n <= 18`, unless
/// `PLONKY2_GPU_LDE=zeknox`. Larger LDEs, and all with the switch, use zeknox (column-major,
/// natural order). Every LDE of a circuit has the same `lg_n`, so a quotient's inputs all share
/// one layout.
pub(crate) fn own_lde(lg_n: usize, rate_bits: usize) -> bool {
    static ZEKNOX: OnceLock<bool> = OnceLock::new();
    let zeknox = *ZEKNOX.get_or_init(|| std::env::var("PLONKY2_GPU_LDE").ok().as_deref() == Some("zeknox"));
    !zeknox && (1..=18).contains(&lg_n) && rate_bits <= 4
}

/// Extends `cols` canonical coefficient columns (column-major, `2^lg_n` each, on the device) into
/// `dst` with the layout `own_lde` selects. Synchronous.
pub(crate) fn coset_lde(dst: *mut c_void, src: *mut c_void, lg_n: usize, rate_bits: usize, cols: usize) -> GpuResult<()> {
    coset_lde_on(dst, src, lg_n, rate_bits, cols, core::ptr::null_mut())?;
    sync("coset lde")
}

/// `coset_lde` queued on `stream` (ours), or synchronous after `stream` (zeknox's, which runs on
/// the legacy stream).
fn coset_lde_on(dst: *mut c_void, src: *mut c_void, lg_n: usize, rate_bits: usize, cols: usize, stream: *mut c_void) -> GpuResult<()> {
    use plonky2_field::types::{Field, PrimeField64};
    if own_lde(lg_n, rate_bits) {
        let omega = GoldilocksField::primitive_root_of_unity(lg_n + rate_bits).to_canonical_u64();
        let shift = GoldilocksField::coset_shift().to_canonical_u64();
        return cuda(
            unsafe { plonky2_coset_lde(dst as *mut u64, src as *const u64, lg_n as u32, rate_bits as u32, cols as u32, omega, shift, stream) },
            "coset lde",
        );
    }
    stream_sync(stream, "before lde")?;
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
    // zeknox's LDE synchronises internally.
    zeknox(unsafe { ffi::compute_batched_lde(0, dst, src, lg_n as u32, ffi::Direction::Forward, cfg) }, "lde")?;
    sync("lde")
}

extern "C" {
    fn plonky2_merkle(digests: *mut u64, cap: *mut u64, leaves: *const u64, lg_rows: u32, cols: u32, cap_height: u32, stream: *mut c_void) -> i32;
    fn plonky2_merkle_cols(digests: *mut u64, cap: *mut u64, leaves: *const u64, lg_rows: u32, cols: u32, cap_height: u32, stream: *mut c_void) -> i32;
    fn plonky2_digest_layout(dst: *mut u64, src: *const u64, per_subtree: u32, lg_leaves: u32, num_subtrees: u32) -> i32;
    fn plonky2_coset_lde(dst: *mut u64, src: *const u64, lg_n: u32, rate_bits: u32, cols: u32, omega: u64, shift: u64, stream: *mut c_void) -> i32;
    fn plonky2_transpose(dst: *mut u64, src: *const u64, rows: u32, cols: u32, stream: *mut c_void) -> i32;
    fn plonky2_intt(dst: *mut u64, src: *const u64, lg_n: u32, cols: u32, omega_n: u64, stream: *mut c_void) -> i32;
}

/// `PLONKY2_GPU_INTT=1`: commitments from values (wires, Z and partial products) upload the
/// values and interpolate them on the device (`plonky2_intt`), instead of an IFFT per column on
/// the CPU; the coefficients come back with the tree.
static INTT_COMMITS: AtomicU64 = AtomicU64::new(0);
static DEVICE_LEAF_COMMITS: AtomicU64 = AtomicU64::new(0);

/// `PLONKY2_GPU_DEVICE_LEAVES=1`: a commitment whose LDE is kept on the device for the proof
/// leaves its Merkle leaves there too. The tree hashes the column-major LDE directly (no
/// transpose), nothing is downloaded, and the leaves are rows of the kept LDE copied out when
/// read (the FRI queries read a few dozen; a CPU fallback reading many copies them all once).
pub(crate) fn device_leaves_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_DEVICE_LEAVES").ok().as_deref() == Some("1"))
}

pub(crate) fn intt_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_INTT").ok().as_deref() == Some("1"))
}

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

fn stream_sync(stream: *mut c_void, what: &str) -> GpuResult<()> {
    cuda(unsafe { ffi::cudaStreamSynchronize(stream) }, what)?;
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

    fn as_slice(&self) -> &[u64] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
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

/// What the GPU lock guards: lane 0 and the quotient's device buffers.
struct State {
    /// Lane 0: large commitments, and every commitment with one lane.
    lane: Lane,
    q: quotient::QuotientBufs,
}

/// What one commitment needs on the device at a time: a non-blocking stream for its kernels and
/// copies, a second one for the overlapped leaf download (ordered by `event`), pinned staging and
/// a device workspace. `PLONKY2_GPU_COMMIT_LANES=n` runs up to n commitments at once.
struct Lane {
    stream: usize,
    copy_stream: usize,
    event: usize,
    input: Pinned,
    ws: Workspace,
}

// The raw pointers are only used by the thread holding the lane.
unsafe impl Send for Lane {}

impl Lane {
    const fn empty() -> Self {
        Self { stream: 0, copy_stream: 0, event: 0, input: Pinned::empty(), ws: Workspace::new() }
    }

    /// Creates the streams and the event on first use.
    fn ready(&mut self) -> GpuResult<()> {
        if self.stream == 0 {
            let (mut s, mut c, mut e) = (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null_mut());
            cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut s, 1) }, "lane stream")?;
            cuda(unsafe { ffi::cudaStreamCreateWithFlags(&mut c, 1) }, "lane copy stream")?;
            cuda(unsafe { ffi::cudaEventCreateWithFlags(&mut e, 2) }, "lane event")?; // cudaEventDisableTiming
            (self.stream, self.copy_stream, self.event) = (s as usize, c as usize, e as usize);
        }
        Ok(())
    }
}

/// `PLONKY2_GPU_COMMIT_LANES` (default 1): commitments on the device at once.
fn commit_lanes() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| std::env::var("PLONKY2_GPU_COMMIT_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1))
}

/// Commitments with an LDE above this size (secp) run on lane 0 only, so the other lanes' device
/// workspaces stay small.
const LANE_BIG_BYTES: usize = 256 << 20;

/// Lanes 1.. (created on demand, at most `commit_lanes() - 1`), idle ones in `free`.
struct ExtraLanes {
    free: Vec<Lane>,
    created: usize,
}

static EXTRA_LANES: Mutex<ExtraLanes> = Mutex::new(ExtraLanes { free: Vec::new(), created: 0 });
static LANE_FREED: Condvar = Condvar::new();

/// A lane held by one commitment.
enum HeldLane {
    Main(std::sync::MutexGuard<'static, State>),
    Extra(Option<Lane>),
}

impl HeldLane {
    fn lane(&mut self) -> &mut Lane {
        match self {
            Self::Main(s) => &mut s.lane,
            Self::Extra(l) => l.as_mut().expect("held"),
        }
    }
}

impl Drop for HeldLane {
    fn drop(&mut self) {
        if let Self::Extra(l) = self {
            if let Some(l) = l.take() {
                EXTRA_LANES.lock().unwrap_or_else(|e| e.into_inner()).free.push(l);
                LANE_FREED.notify_one();
            }
        }
    }
}

/// Lane 0 for large commitments (or with one lane); otherwise any free lane, lane 0 included.
fn acquire_lane(b: &'static Backend, large: bool) -> HeldLane {
    let lanes = commit_lanes();
    if lanes <= 1 || large {
        return HeldLane::Main(b.state.lock().unwrap_or_else(|e| e.into_inner()));
    }
    loop {
        {
            let mut x = EXTRA_LANES.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(l) = x.free.pop() {
                return HeldLane::Extra(Some(l));
            }
            if x.created < lanes - 1 {
                x.created += 1;
                return HeldLane::Extra(Some(Lane::empty()));
            }
        }
        match b.state.try_lock() {
            Ok(g) => return HeldLane::Main(g),
            Err(std::sync::TryLockError::Poisoned(e)) => return HeldLane::Main(e.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        let x = EXTRA_LANES.lock().unwrap_or_else(|e| e.into_inner());
        // Lane 0 frees without a notification: poll it every few hundred microseconds.
        let _ = LANE_FREED.wait_timeout(x, std::time::Duration::from_micros(300));
    }
}

/// Growable device buffers reused across commitments (only touched by the lane's holder).
struct Workspace {
    bufs: [DevBuf; 7],
}

impl Workspace {
    const fn new() -> Self {
        Self { bufs: [DevBuf::empty(), DevBuf::empty(), DevBuf::empty(), DevBuf::empty(), DevBuf::empty(), DevBuf::empty(), DevBuf::empty()] }
    }
}

const WS_IN: usize = 0;
const WS_LDE: usize = 1;
const WS_LEAVES: usize = 2;
const WS_DIGESTS: usize = 3;
const WS_CAP: usize = 4;
const WS_DIGESTS_PLONKY2: usize = 5;
/// Values uploaded for an inverse NTT (commitments from values).
const WS_VALUES: usize = 6;

struct DevBuf {
    ptr: *mut c_void,
    cap_u64: usize,
}

impl DevBuf {
    const fn empty() -> Self {
        Self { ptr: core::ptr::null_mut(), cap_u64: 0 }
    }

    /// Frees the buffer (call under the GPU lock).
    fn release(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cudaFree(self.ptr) };
            DEVICE_WS_BYTES.fetch_sub(self.cap_u64 as u64 * 8, Ordering::Relaxed);
            self.ptr = core::ptr::null_mut();
            self.cap_u64 = 0;
        }
    }

    fn ensure(&mut self, len_u64: usize) -> GpuResult<*mut c_void> {
        if self.cap_u64 < len_u64.max(1) {
            if !self.ptr.is_null() {
                range("gpu:ws_free", || unsafe { ffi::cudaFree(self.ptr) });
                DEVICE_WS_BYTES.fetch_sub(self.cap_u64 as u64 * 8, Ordering::Relaxed);
                self.ptr = core::ptr::null_mut();
                self.cap_u64 = 0;
            }
            let mut p = core::ptr::null_mut();
            range("gpu:ws_alloc", || cuda(unsafe { ffi::cudaMalloc(&mut p, len_u64.max(1) * 8) }, "cudaMalloc (workspace)"))?;
            self.ptr = p;
            self.cap_u64 = len_u64.max(1);
            DEVICE_WS_ALLOCS.fetch_add(1, Ordering::Relaxed);
            let now = DEVICE_WS_BYTES.fetch_add(self.cap_u64 as u64 * 8, Ordering::Relaxed) + self.cap_u64 as u64 * 8;
            DEVICE_WS_PEAK.fetch_max(now, Ordering::Relaxed);
        }
        Ok(self.ptr)
    }
}

// The raw stream handle is only used while holding the backend lock.
unsafe impl Send for State {}

/// One commitment's pinned output: row-major leaves and (digests || cap).
struct OutputSet {
    leaves: Pinned,
    digests: Pinned,
    /// Canonicalised coefficients when canonicalising outside the GPU lock.
    input: Pinned,
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
    /// `PLONKY2_GPU_LEAF_LEASE=1`: leaves are downloaded into leased pinned buffers that the tree
    /// keeps (no copy); budget `PLONKY2_GPU_LEAF_POOL_BYTES` (default 4 GiB, at most 32 GiB).
    leaf_pool: Option<std::sync::Arc<leaf_pool::LeafPool>>,
}

std::thread_local! {
    /// Depth of `prove_with_partition_witness` calls on this thread.
    static IN_PROOF: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Marks the current thread as proving until dropped. Leaf leases are only granted inside a
/// proof: their lifetime is bounded by the proof's `PolynomialBatch` values. Commitments made
/// elsewhere (circuit preprocessing, whose trees live as long as the circuit data) copy their
/// leaves instead, so they can never hold leases indefinitely.
#[derive(Debug)]
pub struct ProofScope(());

pub fn proof_scope() -> ProofScope {
    IN_PROOF.with(|d| d.set(d.get() + 1));
    ProofScope(())
}

impl Drop for ProofScope {
    fn drop(&mut self) {
        let depth = IN_PROOF.with(|d| {
            let v = d.get().saturating_sub(1);
            d.set(v);
            v
        });
        if depth == 0 {
            CAPTURE_ROLES.with(|c| c.set(0));
            release_resident();
        }
    }
}

std::thread_local! {
    /// How many of the current proof's commitments, counted from its first, keep their LDE on
    /// the device (0: none).
    static CAPTURE_ROLES: core::cell::Cell<u8> = const { core::cell::Cell::new(0) };
    /// Commitments the current proof has made so far (on the GPU or not).
    static COMMIT_SEQ: core::cell::Cell<u8> = const { core::cell::Cell::new(0) };
    /// LDEs kept on the device for the current proof.
    static RESIDENT: core::cell::RefCell<Vec<Resident>> = const { core::cell::RefCell::new(Vec::new()) };
    /// Validity of the device leaves backed by those LDEs, cleared when the LDEs are released.
    static LEAF_VIEWS: core::cell::RefCell<Vec<std::sync::Arc<AtomicBool>>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// A proof's commitments in order (`prove`): wires, Z and partial products, quotient. A kept LDE
/// is tagged with its commitment's position, so one that stayed on the CPU cannot shift the
/// others into the wrong role.
pub(crate) const ROLE_WIRES: u8 = 0;
pub(crate) const ROLE_ZS: u8 = 1;
pub(crate) const ROLE_QUOTIENT: u8 = 2;
/// Not a commitment: the wires' subgroup values, kept by the wires commitment when it
/// interpolates them on the device, for the partial products.
pub(crate) const ROLE_WIRE_VALUES: u8 = 3;

/// A commitment's LDE (column-major; in leaf order when `leaf_order`, else natural order) kept on
/// the device for the quotient and FRI.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Resident {
    pub ptr: usize,
    pub bytes: usize,
    pub cols: usize,
    pub lg_rows: usize,
    pub leaf_order: bool,
    pub role: u8,
}

/// Keep the LDEs of this proof's first `roles` commitments on the device.
pub(crate) fn begin_capture(roles: u8) {
    release_resident();
    CAPTURE_ROLES.with(|c| c.set(roles));
    COMMIT_SEQ.with(|c| c.set(0));
}

/// The role of the commitment starting now, if its LDE is to be kept.
fn next_capture_role() -> Option<u8> {
    if !in_proof() {
        return None;
    }
    let seq = COMMIT_SEQ.with(|c| {
        let v = c.get();
        c.set(v.saturating_add(1));
        v
    });
    (seq < CAPTURE_ROLES.with(|c| c.get())).then_some(seq)
}

/// The kept LDE of `role`, if it has the expected shape and layout.
pub(crate) fn kept(role: u8, cols: usize, lg_rows: usize, leaf_order: bool) -> Option<Resident> {
    RESIDENT.with(|r| {
        r.borrow().iter().copied().find(|k| k.role == role && k.cols == cols && k.lg_rows == lg_rows && k.leaf_order == leaf_order)
    })
}

/// Returns this thread's kept LDEs to the pool.
pub(crate) fn release_resident() {
    for v in LEAF_VIEWS.with(|v| v.borrow_mut().drain(..).collect::<Vec<_>>()) {
        v.store(false, Ordering::SeqCst);
    }
    let kept: Vec<Resident> = RESIDENT.with(|r| r.borrow_mut().drain(..).collect());
    for r in kept {
        RESIDENT_POOL.give(r.ptr, r.bytes);
    }
}

/// Device buffers for kept LDEs, reused by size; at most `PLONKY2_GPU_RESIDENT_BYTES` (default
/// 4 GiB) allocated. Taking needs the GPU lock (it may cudaMalloc / cudaFree); giving does not.
struct DevPool {
    free: Mutex<Vec<(usize, usize)>>,
    held: AtomicU64,
    refused: AtomicU64,
    taken: AtomicU64,
}

static RESIDENT_POOL: DevPool =
    DevPool { free: Mutex::new(Vec::new()), held: AtomicU64::new(0), refused: AtomicU64::new(0), taken: AtomicU64::new(0) };

fn resident_budget() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_RESIDENT_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(4 << 30)
    })
}

/// Device memory left free for the commitments' workspace when caching LDEs for the quotient
/// (`PLONKY2_GPU_RESERVE_BYTES`, default 3 GiB: a 2^17-row commitment's workspace).
fn device_reserve() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PLONKY2_GPU_RESERVE_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(3 << 30)
    })
}

/// Whether `bytes` more device memory can be allocated while keeping the reserve free.
/// The device's free memory (0 if unknown).
fn device_free() -> usize {
    let (mut free, mut total) = (0usize, 0usize);
    if unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) } == 0 {
        free
    } else {
        0
    }
}

pub(crate) fn device_room(bytes: usize) -> bool {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) == 0 && free >= bytes + device_reserve() }
}

impl DevPool {
    fn take(&self, bytes: usize) -> Option<*mut c_void> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = free.iter().position(|&(b, _)| b == bytes) {
            self.taken.fetch_add(1, Ordering::Relaxed);
            return Some(free.swap_remove(i).1 as *mut c_void);
        }
        // Free idle buffers of other sizes while over the budget.
        while self.held.load(Ordering::Relaxed) + bytes as u64 > resident_budget() {
            let Some((b, ptr)) = free.pop() else { break };
            unsafe { ffi::cudaFree(ptr as *mut c_void) };
            self.held.fetch_sub(b as u64, Ordering::Relaxed);
        }
        // Idle buffers count against the device's free memory too: free them for room, but only
        // when that can make enough (otherwise keep them for reuse).
        if !device_room(bytes) {
            let idle: usize = free.iter().map(|&(b, _)| b).sum();
            if device_free() + idle >= bytes + device_reserve() {
                while !device_room(bytes) {
                    let Some((b, ptr)) = free.pop() else { break };
                    unsafe { ffi::cudaFree(ptr as *mut c_void) };
                    self.held.fetch_sub(b as u64, Ordering::Relaxed);
                }
            }
        }
        if self.held.load(Ordering::Relaxed) + bytes as u64 > resident_budget() || !device_room(bytes) {
            self.refused.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut p = core::ptr::null_mut();
        if unsafe { ffi::cudaMalloc(&mut p, bytes) } != 0 {
            unsafe { ffi::cudaGetLastError() };
            self.refused.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.held.fetch_add(bytes as u64, Ordering::Relaxed);
        self.taken.fetch_add(1, Ordering::Relaxed);
        Some(p)
    }

    fn give(&self, ptr: usize, bytes: usize) {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push((bytes, ptr));
    }

    /// Frees the idle buffers (call under the GPU lock).
    fn trim(&self) {
        let idle: Vec<(usize, usize)> = self.free.lock().unwrap_or_else(|e| e.into_inner()).drain(..).collect();
        for (b, ptr) in idle {
            unsafe { ffi::cudaFree(ptr as *mut c_void) };
            self.held.fetch_sub(b as u64, Ordering::Relaxed);
        }
    }
}

/// Frees idle kept-LDE buffers (call under the GPU lock).
pub(crate) fn trim_resident_pool() {
    RESIDENT_POOL.trim();
}

/// (kept-LDE buffers handed out, refused for the budget, bytes allocated).
pub fn resident_stats() -> (u64, u64, u64) {
    (
        RESIDENT_POOL.taken.load(Ordering::Relaxed),
        RESIDENT_POOL.refused.load(Ordering::Relaxed),
        RESIDENT_POOL.held.load(Ordering::Relaxed),
    )
}

fn in_proof() -> bool {
    IN_PROOF.with(|d| d.get() > 0)
}

struct CudaPinned;

impl leaf_pool::HostAlloc for CudaPinned {
    fn alloc(&self, bytes: usize) -> Option<*mut u8> {
        let mut p = core::ptr::null_mut();
        let ok = unsafe { ffi::cudaMallocHost(&mut p, bytes) } == 0;
        if ok {
            log::info!("plonky2 GPU leaf pool: pinned {bytes} bytes");
        }
        ok.then_some(p as *mut u8)
    }

    unsafe fn free(&self, ptr: *mut u8, _bytes: usize) {
        ffi::cudaFreeHost(ptr as *mut c_void);
    }
}

fn make_leaf_pool() -> Option<std::sync::Arc<leaf_pool::LeafPool>> {
    if std::env::var("PLONKY2_GPU_LEAF_LEASE").ok().as_deref() != Some("1") {
        return None;
    }
    let mut budget = std::env::var("PLONKY2_GPU_LEAF_POOL_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(leaf_pool::DEFAULT_BUDGET);
    if budget > leaf_pool::MAX_BUDGET {
        log::warn!("PLONKY2_GPU_LEAF_POOL_BYTES={budget} exceeds the 32 GiB cap; using 32 GiB");
        budget = leaf_pool::MAX_BUDGET;
    }
    Some(leaf_pool::LeafPool::new(budget, Box::new(CudaPinned)))
}

/// `PLONKY2_GPU_CANON_OUTSIDE_LOCK=1`: canonicalise into the claimed output set's own input
/// staging before taking the GPU lock (the set is not reused before its upload completed).
fn canon_outside_lock() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_CANON_OUTSIDE_LOCK").ok().as_deref() == Some("1"))
}

/// `PLONKY2_GPU_DEVICE_WORKSPACE=1`: one device workspace, protected by the GPU lock, grown on
/// demand and reused, instead of cudaMalloc / cudaFree per commitment.
fn device_workspace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_DEVICE_WORKSPACE").ok().as_deref() == Some("1"))
}

/// `PLONKY2_GPU_OVERLAP_D2H=1`: download the leaves on a separate non-blocking stream while the
/// Merkle kernels run (both only read the device leaves), instead of after them.
fn overlap_d2h() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_OVERLAP_D2H").ok().as_deref() == Some("1"))
}

static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();
/// Set after any device error; the rest of the process stays on the CPU.
static FAILED: AtomicBool = AtomicBool::new(false);

static GPU_COMMITS: AtomicU64 = AtomicU64::new(0);
static CPU_COMMITS: AtomicU64 = AtomicU64::new(0);
static GPU_ERRORS: AtomicU64 = AtomicU64::new(0);
static OOM_RETRIES: AtomicU64 = AtomicU64::new(0);
static OOM_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static QUARANTINED: AtomicU64 = AtomicU64::new(0);
static LOCK_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static BUF_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static PINNED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVICE_WS_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVICE_WS_PEAK: AtomicU64 = AtomicU64::new(0);
static DEVICE_WS_ALLOCS: AtomicU64 = AtomicU64::new(0);
static DEVICE_CALL_ALLOCS: AtomicU64 = AtomicU64::new(0);
static PINNED_PEAK: AtomicU64 = AtomicU64::new(0);
static EXEC_NS: AtomicU64 = AtomicU64::new(0);
static REBUILD_NS: AtomicU64 = AtomicU64::new(0);

/// Process-wide commitment counters of the CUDA backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuStats {
    pub gpu_commits: u64,
    pub cpu_commits: u64,
    pub gpu_errors: u64,
    /// Errors after which the device could not be confirmed idle; their buffers were leaked.
    pub quarantined: u64,
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
    /// Leaf leases (PLONKY2_GPU_LEAF_LEASE=1): granted, refused (copied instead), leased bytes now
    /// and at peak, pinned bytes held by the leaf pool at peak.
    pub leases_granted: u64,
    pub leases_refused: u64,
    pub leased_bytes: u64,
    pub leased_peak_bytes: u64,
    pub leaf_pool_peak_bytes: u64,
    /// Pinned bytes the leaf pool holds right now (leased + idle cache), at most its budget.
    pub leaf_pool_bytes: u64,
    /// Leaf pool pinned allocations and frees (reclaims of idle buffers to fit a new class).
    pub leaf_pool_allocs: u64,
    pub leaf_pool_frees: u64,
    /// Device workspace (PLONKY2_GPU_DEVICE_WORKSPACE=1): bytes now / peak, cudaMalloc calls; and
    /// per-commitment cudaMalloc calls when the workspace is off.
    pub device_ws_bytes: u64,
    pub device_ws_peak_bytes: u64,
    pub device_ws_allocs: u64,
    pub device_call_allocs: u64,
    /// Quotient evaluation (PLONKY2_GPU_QUOTIENT=1): on the GPU, on the CPU, check mismatches,
    /// device time and GPU lock wait.
    pub quotient_gpu: u64,
    pub quotient_cpu: u64,
    pub quotient_mismatch: u64,
    pub quotient_exec_ns: u64,
    pub quotient_lock_wait_ns: u64,
    /// Quotients that read the commitments' kept wires and Z LDEs; constants cache hits; kept-LDE
    /// buffers handed out, refused for the budget, and bytes allocated.
    pub quotient_reused_lde: u64,
    pub quotient_const_hits: u64,
    pub quotient_unlocked: u64,
    pub kept_lde_taken: u64,
    pub kept_lde_refused: u64,
    pub kept_lde_bytes: u64,
    /// FRI proof-of-work searches done on the GPU (PLONKY2_GPU_POW=1) and their wall time.
    pub pow_gpu: u64,
    pub pow_ns: u64,
    /// Commitments that ran out of device memory: retried after dropping the quotient caches, and
    /// computed on the CPU after all.
    pub oom_retries: u64,
    pub oom_fallbacks: u64,
    /// FRI final polynomials (PLONKY2_GPU_FRI=1) from the GPU, from the CPU while enabled, check
    /// mismatches, and GPU wall time.
    pub fri_gpu: u64,
    pub fri_cpu: u64,
    pub fri_mismatch: u64,
    pub fri_ns: u64,
    /// Commitments from values interpolated on the device (PLONKY2_GPU_INTT=1).
    pub intt_commits: u64,
    /// Commitments whose leaves stayed on the device (PLONKY2_GPU_DEVICE_LEAVES=1).
    pub device_leaf_commits: u64,
    /// Proofs whose permutation partial products ran on the GPU (PLONKY2_GPU_PP=1).
    pub pp_gpu: u64,
    /// Opening sets computed on the GPU (PLONKY2_GPU_OPEN=1).
    pub open_gpu: u64,
}

pub fn stats() -> GpuStats {
    GpuStats {
        gpu_commits: GPU_COMMITS.load(Ordering::Relaxed),
        cpu_commits: CPU_COMMITS.load(Ordering::Relaxed),
        gpu_errors: GPU_ERRORS.load(Ordering::Relaxed),
        quarantined: QUARANTINED.load(Ordering::Relaxed),
        buffer_wait_ns: BUF_WAIT_NS.load(Ordering::Relaxed),
        lock_wait_ns: LOCK_WAIT_NS.load(Ordering::Relaxed),
        exec_ns: EXEC_NS.load(Ordering::Relaxed),
        rebuild_ns: REBUILD_NS.load(Ordering::Relaxed),
        pinned_bytes: PINNED_BYTES.load(Ordering::Relaxed),
        pinned_peak_bytes: PINNED_PEAK.load(Ordering::Relaxed),
        pool_size: BACKEND.get().and_then(|b| b.as_ref()).map_or(0, |b| b.pool.size as u64),
        pool_free: BACKEND.get().and_then(|b| b.as_ref()).map_or(0, |b| b.pool.free.lock().map_or(0, |f| f.len() as u64)),
        leases_granted: leaf_stat(|p| p.granted.load(Ordering::Relaxed)),
        leases_refused: leaf_stat(|p| p.refused.load(Ordering::Relaxed)),
        leased_bytes: leaf_stat(|p| p.leased_bytes.load(Ordering::Relaxed)),
        leased_peak_bytes: leaf_stat(|p| p.leased_peak.load(Ordering::Relaxed)),
        leaf_pool_peak_bytes: leaf_stat(|p| p.allocated_peak.load(Ordering::Relaxed)),
        leaf_pool_bytes: leaf_stat(|p| p.allocated() as u64),
        leaf_pool_allocs: leaf_stat(|p| p.allocs.load(Ordering::Relaxed)),
        leaf_pool_frees: leaf_stat(|p| p.frees.load(Ordering::Relaxed)),
        device_ws_bytes: DEVICE_WS_BYTES.load(Ordering::Relaxed),
        device_ws_peak_bytes: DEVICE_WS_PEAK.load(Ordering::Relaxed),
        device_ws_allocs: DEVICE_WS_ALLOCS.load(Ordering::Relaxed),
        device_call_allocs: DEVICE_CALL_ALLOCS.load(Ordering::Relaxed),
        quotient_gpu: quotient::quotient_stats().0,
        quotient_cpu: quotient::quotient_stats().1,
        quotient_mismatch: quotient::quotient_stats().2,
        quotient_exec_ns: quotient::quotient_stats().3,
        quotient_lock_wait_ns: quotient::quotient_stats().4,
        quotient_reused_lde: quotient::reuse_stats().1,
        quotient_const_hits: quotient::reuse_stats().0,
        quotient_unlocked: quotient::reuse_stats().2,
        kept_lde_taken: resident_stats().0,
        kept_lde_refused: resident_stats().1,
        kept_lde_bytes: resident_stats().2,
        pow_gpu: pow::pow_stats().0,
        pow_ns: pow::pow_stats().1,
        oom_retries: OOM_RETRIES.load(Ordering::Relaxed),
        oom_fallbacks: OOM_FALLBACKS.load(Ordering::Relaxed),
        fri_gpu: fri::fri_stats().0,
        fri_cpu: fri::fri_stats().1,
        fri_mismatch: fri::fri_stats().2,
        fri_ns: fri::fri_stats().3,
        intt_commits: INTT_COMMITS.load(Ordering::Relaxed),
        device_leaf_commits: DEVICE_LEAF_COMMITS.load(Ordering::Relaxed),
        pp_gpu: perm::pp_count(),
        open_gpu: open::open_count(),
    }
}

fn leaf_stat(f: impl Fn(&leaf_pool::LeafPool) -> u64) -> u64 {
    BACKEND.get().and_then(|b| b.as_ref()).and_then(|b| b.leaf_pool.as_deref()).map_or(0, f)
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
                    let leaf_pool = make_leaf_pool();
                    if let Some(secs) = std::env::var("PLONKY2_GPU_STATS_LOG_SECS").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&s| s > 0) {
                        std::thread::spawn(move || loop {
                            std::thread::sleep(std::time::Duration::from_secs(secs));
                            log::info!("plonky2 GPU stats {:?}", stats());
                        });
                    }
                    log::info!(
                        "plonky2 GPU leaf leases: {}",
                        leaf_pool.as_ref().map_or("off".to_string(), |p| format!("on, budget {} bytes", p.budget()))
                    );
                    Some(Backend {
                        leaf_pool,
                        min_lde_rows,
                        state: Mutex::new(State { lane: Lane::empty(), q: quotient::QuotientBufs::new() }),
                        pool: Pool {
                            size: buffers,
                            free: Mutex::new(
                                (0..buffers).map(|_| OutputSet { leaves: Pinned::empty(), digests: Pinned::empty(), input: Pinned::empty() }).collect(),
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

/// `PLONKY2_GPU_SYNC`: how host threads wait for the device. `spin` (CUDA's choice for this
/// setup; burns a core while waiting), `yield`, or `block` (sleep until the work completes).
fn sync_flags() -> u32 {
    match std::env::var("PLONKY2_GPU_SYNC").ok().as_deref() {
        Some("block") => 4, // cudaDeviceScheduleBlockingSync
        Some("yield") => 2, // cudaDeviceScheduleYield
        Some("spin") => 1,  // cudaDeviceScheduleSpin
        _ => 0,             // cudaDeviceScheduleAuto
    }
}

fn init_device() -> GpuResult<()> {
    use plonky2_field::types::{Field, PrimeField64};
    let flags = sync_flags();
    if flags != 0 {
        // Must precede the first use of the device in this process.
        cuda(unsafe { ffi::cudaSetDeviceFlags(flags) }, "cudaSetDeviceFlags")?;
        log::info!("plonky2 GPU: device schedule flags {flags}");
    }
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
    let columns: Vec<&[F]> = polynomials.iter().map(|p| p.coeffs.as_slice()).collect();
    commit_columns::<F, H>(&columns, false, rate_bits, blinding, cap_height).map(|(tree, _)| tree)
}

/// Commits to polynomials given by their values on the subgroup (`PLONKY2_GPU_INTT=1`): returns
/// the tree and the coefficients. On `Err` nothing is counted, so the caller can interpolate on the
/// CPU and commit the coefficients instead.
pub(crate) fn try_commit_values<F: RichField, H: Hasher<F>>(
    values: &[PolynomialValues<F>],
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
) -> Result<(MerkleTree<F, H>, Vec<PolynomialCoeffs<F>>), &'static str> {
    let columns: Vec<&[F]> = values.iter().map(|v| v.values.as_slice()).collect();
    try_commit_value_columns::<F, H>(&columns, rate_bits, blinding, cap_height)
}

/// `try_commit_values` on borrowed columns (the prover's witness, without copying it).
pub(crate) fn try_commit_value_columns<F: RichField, H: Hasher<F>>(
    columns: &[&[F]],
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
) -> Result<(MerkleTree<F, H>, Vec<PolynomialCoeffs<F>>), &'static str> {
    if !intt_enabled() {
        return Err("intt_disabled");
    }
    let n = columns.first().map_or(0, |v| v.len());
    if n == 0 || !n.is_power_of_two() || !own_lde(n.trailing_zeros() as usize, rate_bits) {
        return Err("intt_unsupported_size");
    }
    let seq = COMMIT_SEQ.with(|c| c.get());
    let result = commit_columns::<F, H>(columns, true, rate_bits, blinding, cap_height);
    match result {
        Ok((tree, Some(coeffs))) => {
            #[cfg(test)]
            for (v, c) in columns.iter().zip(&coeffs) {
                assert_eq!(PolynomialValues::new(v.to_vec()).ifft().coeffs, c.coeffs, "device interpolation differs from the CPU IFFT");
            }
            INTT_COMMITS.fetch_add(1, Ordering::Relaxed);
            Ok((tree, coeffs))
        }
        Ok((_, None)) => unreachable!("values commitments return coefficients"),
        Err(e) => {
            // The coefficient commitment that follows counts this commitment again.
            COMMIT_SEQ.with(|c| c.set(seq));
            Err(e)
        }
    }
}

/// Commits to `columns`: coefficients, or values on the subgroup when `values_in` (interpolated on
/// the device; their coefficients are returned too).
#[allow(clippy::type_complexity)]
fn commit_columns<F: RichField, H: Hasher<F>>(
    columns: &[&[F]],
    values_in: bool,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
) -> Result<(MerkleTree<F, H>, Option<Vec<PolynomialCoeffs<F>>>), &'static str> {
    let total = Instant::now();
    let cols = columns.len();
    let n = columns.first().map_or(0, |c| c.len());
    if n == 0 || !n.is_power_of_two() || columns.iter().any(|c| c.len() != n) {
        return Err("irregular_input");
    }
    let lg_n = n.trailing_zeros() as usize;
    // Counted before any CPU decision, so the proof's commitment positions stay aligned.
    let capture_role = next_capture_role();
    let reason = cpu_reason::<F, H>(lg_n, rate_bits, cols, blinding);
    if !reason.is_empty() {
        return Err(reason);
    }
    if cap_height > lg_n + rate_bits {
        return Err("cap_too_high");
    }
    let b: &'static Backend = backend().expect("checked by cpu_reason");

    // 0. The commitments of a proof whose quotient (and FRI) run on the GPU extend straight into a
    // kept buffer, which those read instead of recomputing the LDE.
    let lde_bytes = cols * (n << rate_bits) * 8;
    let mut kept = KeptGuard(if capture_role.is_some() { RESIDENT_POOL.take(lde_bytes).map(|p| (p as usize, lde_bytes)) } else { None });
    // Lease leaf storage (non-blocking; None -> leaves are copied out of the output set). Leaves
    // that stay on the device in the kept buffer need no host storage.
    let device_leaves = kept.0.is_some() && device_leaves_enabled() && own_lde(lg_n, rate_bits) && !zeknox_merkle();
    let mut lease = if in_proof() && !device_leaves {
        b.leaf_pool.as_ref().and_then(|p| p.try_lease(cols * (n << rate_bits) * 8))
    } else {
        None
    };
    let lease_ptr = lease.as_mut().map(|l| l.as_mut_ptr() as *mut u64);

    // 1. Claim an exclusive output set (bounded pool).
    let buf_wait_start = Instant::now();
    let mut out = range("gpu:buffer_wait", || b.pool.acquire());
    let buf_wait = buf_wait_start.elapsed().as_secs_f64();

    // 1b. Optionally canonicalise into the set's own staging before taking the GPU lock.
    let pre_canon = if canon_outside_lock() {
        let set = out.set.as_mut().expect("claimed");
        match canonicalize_into(&mut set.input, columns, own_lde(lg_n, rate_bits)) {
            Ok(()) => true,
            Err(e) => {
                drop(out);
                drop(lease);
                FAILED.store(true, Ordering::Relaxed);
                GPU_ERRORS.fetch_add(1, Ordering::Relaxed);
                log::error!("plonky2 GPU staging failed, falling back to CPU for the rest of the process: {e}");
                return Err("gpu_error");
            }
        }
    } else {
        false
    };

    // 2. A lane (lane 0 under the GPU lock for large or zeknox commitments): upload, kernels,
    // download into the claimed set, all on the lane's streams.
    let large = cols * (n << rate_bits) * 8 > LANE_BIG_BYTES || !own_lde(lg_n, rate_bits) || zeknox_merkle();
    let wait = Instant::now();
    let mut held = range("gpu:lock_wait", || acquire_lane(b, large));
    let lock_wait = wait.elapsed().as_secs_f64();
    let mut result = range("gpu:device", || {
        run(held.lane(), out.set.as_mut().expect("claimed"), &mut kept, pre_canon, lease_ptr, columns, values_in, lg_n, rate_bits, cap_height, capture_role)
    });
    // Out of device memory while growing buffers (nothing was launched yet): drop the quotient's
    // caches and try once more.
    if matches!(&result, Err(e) if e.contains("out of memory")) {
        unsafe { ffi::cudaGetLastError() };
        match &mut held {
            HeldLane::Main(state) => quotient::release_device_caches(&mut state.q),
            HeldLane::Extra(_) => quotient::release_device_caches(&mut b.state.lock().unwrap_or_else(|e| e.into_inner()).q),
        }
        OOM_RETRIES.fetch_add(1, Ordering::Relaxed);
        result = range("gpu:device", || {
            run(held.lane(), out.set.as_mut().expect("claimed"), &mut kept, pre_canon, lease_ptr, columns, values_in, lg_n, rate_bits, cap_height, capture_role)
        });
    }
    // 3. Device work is complete (the lane's streams synchronised); the set is ours alone.
    drop(held);

    match result {
        Ok((leaves_flat_len, digests_len, cap_len, exec_s, device)) => {
            // 4. Rebuild outside the GPU lock; the set returns to the pool when `out` drops.
            let rebuild = Instant::now();
            let leased = lease.is_some();
            let tree = range("gpu:rebuild", || {
                rebuild_tree::<F, H>(out.set.as_ref().expect("claimed"), lease, cols, lg_n + rate_bits, cap_height, leaves_flat_len, digests_len, cap_len, device)
            });
            // The interpolated coefficients were downloaded into the set's staging.
            let coeffs = values_in.then(|| {
                range("gpu:coeffs", || {
                    out.set.as_ref().expect("claimed").input.as_slice()[..cols * n]
                        .par_chunks(n)
                        .map(|c| PolynomialCoeffs::new(c.iter().map(|&x| F::from_canonical_u64(x)).collect()))
                        .collect::<Vec<_>>()
                })
            });
            drop(out);
            let rebuild_s = rebuild.elapsed().as_secs_f64();
            let mode = if leased {
                "leaves:lease"
            } else if b.leaf_pool.is_none() {
                "leaves:copy"
            } else if !in_proof() {
                "leaves:copy_preprocessing"
            } else {
                "leaves:copy_refused"
            };
            stage_log::commit_event("gpu", mode, lg_n, rate_bits, cols, buf_wait, lock_wait, exec_s, rebuild_s, total.elapsed().as_secs_f64());
            BUF_WAIT_NS.fetch_add((buf_wait * 1e9) as u64, Ordering::Relaxed);
            GPU_COMMITS.fetch_add(1, Ordering::Relaxed);
            LOCK_WAIT_NS.fetch_add((lock_wait * 1e9) as u64, Ordering::Relaxed);
            EXEC_NS.fetch_add((exec_s * 1e9) as u64, Ordering::Relaxed);
            REBUILD_NS.fetch_add((rebuild_s * 1e9) as u64, Ordering::Relaxed);
            Ok((tree, coeffs))
        }
        Err(e) if e.contains("out of memory") && confirm_device_idle_after_error() => {
            // Still no room after dropping the caches: this commitment uses the CPU; the GPU stays
            // on for the next ones.
            unsafe { ffi::cudaGetLastError() };
            drop(out);
            drop(lease);
            OOM_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            log::warn!("plonky2 GPU commitment: out of device memory, this commitment on the CPU: {e}");
            Err("gpu_oom")
        }
        Err(e) => {
            // The failed run may have left an async download in flight into `out` / `lease`.
            // Return them only once the device confirms all work is done; otherwise quarantine
            // (leak) them so no later commitment can reuse memory the DMA may still write.
            if confirm_device_idle_after_error() {
                drop(out);
                drop(lease);
            } else {
                QUARANTINED.fetch_add(1, Ordering::Relaxed);
                log::error!("plonky2 GPU: could not confirm device idle after an error; quarantining its buffers");
                core::mem::forget(out);
                core::mem::forget(lease);
            }
            FAILED.store(true, Ordering::Relaxed);
            GPU_ERRORS.fetch_add(1, Ordering::Relaxed);
            log::error!("plonky2 GPU commitment failed, falling back to CPU for the rest of the process: {e}");
            Err("gpu_error")
        }
    }
}

fn confirm_device_idle_after_error() -> bool {
    #[cfg(feature = "gpu-fault-injection")]
    if std::env::var("PLONKY2_GPU_FAULT").ok().as_deref() == Some("d2h_inflight_sync_fail") {
        return false;
    }
    unsafe { ffi::cudaDeviceSynchronize() == 0 }
}

/// Upload, LDE, transpose, Merkle, download into the pinned buffers. Returns the element counts
/// written (leaves, digests, cap, in u64) and the execution time.
/// Canonicalises the coefficients (column after column) into `staging`.
/// With `raw` (the own-LDE path, whose kernels canonicalise on load) the elements are copied as
/// they are.
fn canonicalize_into<F: RichField>(staging: &mut Pinned, columns: &[&[F]], raw: bool) -> GpuResult<()> {
    let n = columns[0].len();
    let input = staging.ensure(columns.len() * n)?;
    if raw && TypeId::of::<F>() == TypeId::of::<GoldilocksField>() {
        range("gpu:stage_raw", || {
            input.par_chunks_mut(n).zip(columns.par_iter()).for_each(|(dst, p)| {
                // SAFETY: F == GoldilocksField, repr(transparent) over u64.
                dst.copy_from_slice(unsafe { core::slice::from_raw_parts(p.as_ptr() as *const u64, p.len()) });
            })
        });
        return Ok(());
    }
    range("gpu:canonicalize", || {
        input.par_chunks_mut(n).zip(columns.par_iter()).for_each(|(dst, p)| {
            for (d, c) in dst.iter_mut().zip(p.iter()) {
                let x = plonky2_field::types::PrimeField64::to_noncanonical_u64(c);
                *d = if x >= P { x - P } else { x };
            }
        })
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run<F: RichField>(
    lane: &mut Lane,
    out: &mut OutputSet,
    kept_guard: &mut KeptGuard,
    pre_canon: bool,
    lease_ptr: Option<*mut u64>,
    columns: &[&[F]],
    values_in: bool,
    lg_n: usize,
    rate_bits: usize,
    cap_height: usize,
    capture_role: Option<u8>,
) -> GpuResult<(usize, usize, usize, f64, Option<DeviceLeafView>)> {
    let t = Instant::now();
    let cols = columns.len();
    let n = 1usize << lg_n;
    let lg_rows = lg_n + rate_bits;
    let rows = 1usize << lg_rows;
    let caps = 1usize << cap_height;
    let num_digests = 2 * (rows - caps);

    #[cfg(feature = "gpu-fault-injection")]
    if inject("run_error") {
        return Err("injected run_error".into());
    }

    let kept = kept_guard.0.map(|(p, _)| p as *mut c_void);
    let leaf_order = own_lde(lg_n, rate_bits);
    let own_merkle = !zeknox_merkle();
    // The tree's leaves stay in the kept LDE (`PLONKY2_GPU_DEVICE_LEAVES=1`).
    let device_leaves = kept.is_some() && leaf_order && own_merkle && device_leaves_enabled();

    // Grow the pinned outputs first: cudaMallocHost may synchronise and must not split the
    // overlapped download from the Merkle kernels below.
    let mut leaves: Option<&mut [u64]> = if device_leaves {
        None
    } else {
        Some(match lease_ptr {
            // SAFETY: the lease holds at least `cols * rows` u64 and is exclusively ours.
            Some(p) => unsafe { core::slice::from_raw_parts_mut(p, cols * rows) },
            None => out.leaves.ensure(cols * rows)?,
        })
    };
    let digests = out.digests.ensure((num_digests + caps) * HASH_U64)?;
    // Interpolated coefficients go to the set's staging (after the values left it).
    let coeffs_host: Option<*mut u64> = if values_in { Some(out.input.ensure(cols * n)?.as_mut_ptr()) } else { None };
    let overlap = overlap_d2h();

    lane.ready()?;
    let stream = lane.stream as *mut c_void;
    let copy_stream = lane.copy_stream as *mut c_void;
    let event = lane.event as *mut c_void;
    // Canonicalised input: already in the set's staging, or canonicalise now (holding the lane).
    let input: &[u64] = if pre_canon {
        &out.input.as_slice()[..cols * n]
    } else {
        canonicalize_into(&mut lane.input, columns, own_lde(lg_n, rate_bits))?;
        &lane.input.as_slice()[..cols * n]
    };

    // Device memory: the reusable workspace, or per-commitment allocations freed when done.
    let ws_on = device_workspace();
    let mut owned: Vec<DeviceBuffer> = Vec::new();
    let mut dev = |slot: usize, len: usize| -> GpuResult<*mut c_void> {
        if ws_on {
            lane.ws.bufs[slot].ensure(len)
        } else {
            let b = range("gpu:alloc", || DeviceBuffer::alloc(len))?;
            DEVICE_CALL_ALLOCS.fetch_add(1, Ordering::Relaxed);
            let p = b.ptr;
            owned.push(b);
            Ok(p)
        }
    };
    let d_in = dev(WS_IN, cols * n)?;

    let d_lde = match kept {
        Some(p) => p,
        None => dev(WS_LDE, cols * rows)?,
    };
    let d_leaves = if device_leaves { core::ptr::null_mut() } else { dev(WS_LEAVES, cols * rows)? };
    let d_digests = dev(WS_DIGESTS, num_digests * HASH_U64)?;
    let d_cap = dev(WS_CAP, caps * HASH_U64)?;
    // Values are uploaded beside the coefficient buffer and interpolated into it. The wires'
    // values go to a kept buffer when the partial products will read them on the device.
    let keep_values = values_in && capture_role == Some(ROLE_WIRES) && perm::enabled();
    let kept_values = if keep_values { RESIDENT_POOL.take(cols * n * 8) } else { None };
    let values_guard = KeptGuard(kept_values.map(|p| (p as usize, cols * n * 8)));
    let d_up = match kept_values {
        Some(p) => p,
        None if values_in => dev(WS_VALUES, cols * n)?,
        None => d_in,
    };
    // On an early return, wait for what this run queued before its buffers go back.
    struct Idle([usize; 2], bool);
    impl Drop for Idle {
        fn drop(&mut self) {
            if !self.1 {
                for s in self.0 {
                    unsafe { ffi::cudaStreamSynchronize(s as *mut c_void) };
                }
            }
        }
    }
    let mut idle = Idle([lane.stream, lane.copy_stream], false);
    // From pinned staging, ordered on the lane's stream.
    range("gpu:h2d", || cuda(unsafe { ffi::cudaMemcpyAsync(d_up, input.as_ptr() as *const c_void, cols * n * 8, ffi::H2D, stream) }, "H2D"))?;
    if values_in {
        use plonky2_field::types::{Field, PrimeField64};
        let omega_n = GoldilocksField::primitive_root_of_unity(lg_n).to_canonical_u64();
        range("gpu:intt", || {
            cuda(
                unsafe { plonky2_intt(d_in as *mut u64, d_up as *const u64, lg_n as u32, cols as u32, omega_n, stream) },
                "intt",
            )
        })?;
    }

    range("gpu:lde", || coset_lde_on(d_lde, d_in, lg_n, rate_bits, cols, stream))?;
    if let Some(host) = coeffs_host {
        // Into pinned staging after the interpolation, ordered on the lane's stream.
        range("gpu:d2h_coeffs", || {
            cuda(unsafe { ffi::cudaMemcpyAsync(host as *mut c_void, d_in, cols * n * 8, ffi::D2H, stream) }, "D2H coefficients")
        })?;
    }
    if device_leaves {
        // The tree hashes the column-major LDE; no row-major copy is needed.
    } else if leaf_order {
        // Already in leaf order: a plain transpose to row-major leaves.
        range("gpu:transpose", || {
            cuda(unsafe { plonky2_transpose(d_leaves as *mut u64, d_lde as *const u64, rows as u32, cols as u32, stream) }, "transpose")?;
            // The overlapped leaf download (copy stream) starts after the transpose.
            cuda(unsafe { ffi::cudaEventRecord(event, stream) }, "transpose event")
        })?;
    } else {
        // zeknox's transpose synchronises internally, so the submit range includes the kernel
        // time; the separate sync range then only catches what is left.
        let tcfg = ffi::TransposeConfig { batches: cols as u32, are_inputs_on_device: true, are_outputs_on_device: true };
        range("gpu:transpose:submit", || {
            zeknox(unsafe { ffi::compute_transpose_rev(0, d_leaves, d_lde, lg_rows as u32, tcfg) }, "transpose")
        })?;
        range("gpu:transpose:sync", || sync("transpose"))?;
    }

    if let (true, Some(leaves)) = (overlap, leaves.as_deref_mut()) {
        // Starts once the leaves are complete (zeknox's transpose synchronised the device; ours
        // recorded the event), alongside the Merkle kernels; the final syncs wait for it.
        if leaf_order {
            cuda(unsafe { ffi::cudaStreamWaitEvent(copy_stream, event, 0) }, "leaves wait")?;
        }
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(leaves.as_mut_ptr() as *mut c_void, d_leaves, cols * rows * 8, ffi::D2H, copy_stream)
            },
            "D2H leaves (async)",
        )?;
        // Lab: fail while the async download is submitted but not yet confirmed complete.
        #[cfg(feature = "gpu-fault-injection")]
        if inject("d2h_inflight_error") || inject("d2h_inflight_sync_fail") {
            return Err("injected failure with the leaf download in flight".into());
        }
    }
    if device_leaves {
        quotient::init_once()?;
        range("gpu:merkle:submit", || {
            cuda(
                unsafe {
                    plonky2_merkle_cols(d_digests as *mut u64, d_cap as *mut u64, d_lde as *const u64, lg_rows as u32, cols as u32, cap_height as u32, stream)
                },
                "merkle (columns)",
            )
        })?;
    } else if own_merkle {
        // Digests straight into plonky2's layout (cuda/quotient.cu); needs the Poseidon constants.
        quotient::init_once()?;
        range("gpu:merkle:submit", || {
            cuda(
                unsafe {
                    plonky2_merkle(d_digests as *mut u64, d_cap as *mut u64, d_leaves as *const u64, lg_rows as u32, cols as u32, cap_height as u32, stream)
                },
                "merkle",
            )
        })?;
    } else {
        // zeknox's Merkle runs on the legacy stream: the leaves must be complete first.
        stream_sync(stream, "before merkle")?;
        range("gpu:merkle:submit", || unsafe {
            ffi::fill_digests_buf_linear_gpu_with_gpu_ptr(
                d_digests,
                d_cap,
                d_leaves,
                num_digests as u64,
                caps as u64,
                rows as u64,
                cols as u64,
                cap_height as u64,
                0,
                0,
            )
        });
    }
    let (dg, cp) = digests.split_at_mut(num_digests * HASH_U64);
    if own_merkle {
        // The downloads queue behind the Merkle kernels on the lane's stream; one wait for both
        // streams at the end.
        range("gpu:d2h:submit", || -> GpuResult<()> {
            if let (false, Some(leaves)) = (overlap, leaves) {
                cuda(unsafe { ffi::cudaMemcpyAsync(leaves.as_mut_ptr() as *mut c_void, d_leaves, cols * rows * 8, ffi::D2H, stream) }, "D2H leaves")?;
            }
            if num_digests > 0 {
                cuda(
                    unsafe { ffi::cudaMemcpyAsync(dg.as_mut_ptr() as *mut c_void, d_digests, num_digests * HASH_U64 * 8, ffi::D2H, stream) },
                    "D2H digests",
                )?;
            }
            cuda(unsafe { ffi::cudaMemcpyAsync(cp.as_mut_ptr() as *mut c_void, d_cap, caps * HASH_U64 * 8, ffi::D2H, stream) }, "D2H cap")
        })?;
        range("gpu:merkle+d2h:sync", || {
            stream_sync(stream, "download")?;
            stream_sync(copy_stream, "leaf download")
        })?;
        idle.1 = true;
        return Ok(finish(kept_guard, values_guard, capture_role, cols, lg_n, lg_rows, leaf_order, device_leaves, num_digests, caps, rows, t, owned));
    }
    // zeknox's Merkle (legacy stream): wait for the kernels and the overlapped leaf download.
    range(if overlap { "gpu:merkle+d2h_leaves:sync" } else { "gpu:merkle:sync" }, || sync("merkle"))?;

    let _d2h = NvtxScope::new("gpu:d2h");
    if let (false, Some(leaves)) = (overlap, leaves) {
        cuda(unsafe { ffi::cudaMemcpy(leaves.as_mut_ptr() as *mut c_void, d_leaves, cols * rows * 8, ffi::D2H) }, "D2H leaves")?;
    }
    if num_digests > 0 {
        // Reorder zeknox's digests into plonky2's layout on the device; the host then only copies.
        let d_plonky2 = dev(WS_DIGESTS_PLONKY2, num_digests * HASH_U64)?;
        let per_subtree = num_digests / caps;
        cuda(
            unsafe { plonky2_digest_layout(d_plonky2 as *mut u64, d_digests as *const u64, per_subtree as u32, (lg_rows - cap_height) as u32, caps as u32) },
            "digest layout",
        )?;
        cuda(unsafe { ffi::cudaMemcpy(dg.as_mut_ptr() as *mut c_void, d_plonky2, num_digests * HASH_U64 * 8, ffi::D2H) }, "D2H digests")?;
    }
    cuda(unsafe { ffi::cudaMemcpy(cp.as_mut_ptr() as *mut c_void, d_cap, caps * HASH_U64 * 8, ffi::D2H) }, "D2H cap")?;
    sync("download")?;
    drop(_d2h);
    idle.1 = true;
    Ok(finish(kept_guard, values_guard, capture_role, cols, lg_n, lg_rows, leaf_order, device_leaves, num_digests, caps, rows, t, owned))
}

/// The end of a successful `run`: frees per-call device memory and records the kept buffers.
#[allow(clippy::too_many_arguments)]
fn finish(
    kept_guard: &mut KeptGuard,
    mut values_guard: KeptGuard,
    capture_role: Option<u8>,
    cols: usize,
    lg_n: usize,
    lg_rows: usize,
    leaf_order: bool,
    device_leaves: bool,
    num_digests: usize,
    caps: usize,
    rows: usize,
    t: Instant,
    owned: Vec<DeviceBuffer>,
) -> (usize, usize, usize, f64, Option<DeviceLeafView>) {
    range("gpu:free", || drop(owned));
    let mut view = None;
    if let Some((p, bytes)) = kept_guard.0.take() {
        let role = capture_role.expect("kept only with a role");
        RESIDENT.with(|r| r.borrow_mut().push(Resident { ptr: p, bytes, cols, lg_rows, leaf_order, role }));
        if device_leaves {
            let alive = std::sync::Arc::new(AtomicBool::new(true));
            LEAF_VIEWS.with(|v| v.borrow_mut().push(alive.clone()));
            DEVICE_LEAF_COMMITS.fetch_add(1, Ordering::Relaxed);
            view = Some(DeviceLeafView { ptr: p, alive });
        }
    }
    if let Some((p, bytes)) = values_guard.0.take() {
        // Canonical subgroup values, column-major; `lg_rows` is the subgroup's size here.
        RESIDENT.with(|r| r.borrow_mut().push(Resident { ptr: p, bytes, cols, lg_rows: lg_n, leaf_order: false, role: ROLE_WIRE_VALUES }));
    }
    (if view.is_some() { 0 } else { cols * rows }, num_digests * HASH_U64, caps * HASH_U64, t.elapsed().as_secs_f64(), view)
}

/// Returns a kept buffer to the pool if its run fails before it is recorded.
struct KeptGuard(Option<(usize, usize)>);

impl Drop for KeptGuard {
    fn drop(&mut self) {
        if let Some((p, b)) = self.0.take() {
            RESIDENT_POOL.give(p, b);
        }
    }
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
#[allow(clippy::too_many_arguments)]
/// A commitment's leaves left on the device: its kept LDE (column-major, leaf order) and whether
/// that buffer still belongs to the proof.
pub(crate) struct DeviceLeafView {
    ptr: usize,
    alive: std::sync::Arc<AtomicBool>,
}

/// Leaves read from a kept LDE: row `r` is element `r` of every column.
fn device_leaves<F: RichField>(view: DeviceLeafView, cols: usize, rows: usize) -> crate::hash::leaf_matrix::DeviceLeaves<F> {
    let (ptr, alive) = (view.ptr, view.alive);
    let alive2 = alive.clone();
    let check = move |a: &AtomicBool| {
        assert!(a.load(Ordering::SeqCst), "device leaves read after the proof released its kept LDEs");
    };
    let fetch_row = move |r: usize| -> Vec<F> {
        check(&alive);
        let mut v = vec![0u64; cols];
        // One element per column: source pitch is a column, destination pitch one element.
        let rc = unsafe {
            ffi::cudaMemcpy2D(v.as_mut_ptr() as *mut c_void, 8, (ptr + r * 8) as *const c_void, rows * 8, 8, cols, ffi::D2H)
        };
        assert_eq!(rc, 0, "device leaves: row copy failed ({rc})");
        v.into_iter().map(F::from_canonical_u64).collect()
    };
    let fetch_all = move || -> Vec<F> {
        check(&alive2);
        let mut col_major = vec![0u64; cols * rows];
        let rc = unsafe { ffi::cudaMemcpy(col_major.as_mut_ptr() as *mut c_void, ptr as *const c_void, cols * rows * 8, ffi::D2H) };
        assert_eq!(rc, 0, "device leaves: copy failed ({rc})");
        let mut out = vec![F::ZERO; cols * rows];
        out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
            for (c, x) in row.iter_mut().enumerate() {
                *x = F::from_canonical_u64(col_major[c * rows + r]);
            }
        });
        out
    };
    crate::hash::leaf_matrix::DeviceLeaves::new(rows, cols, Box::new(fetch_row), Box::new(fetch_all))
}

/// Device leaf commitments (`PLONKY2_GPU_DEVICE_LEAVES=1`).
pub fn device_leaf_commits() -> u64 {
    DEVICE_LEAF_COMMITS.load(Ordering::Relaxed)
}

#[allow(clippy::too_many_arguments)]
fn rebuild_tree<F: RichField, H: Hasher<F>>(
    state: &OutputSet,
    lease: Option<leaf_pool::Lease>,
    cols: usize,
    lg_rows: usize,
    cap_height: usize,
    leaves_len: usize,
    digests_len: usize,
    cap_len: usize,
    device: Option<DeviceLeafView>,
) -> MerkleTree<F, H> {
    debug_assert_eq!(TypeId::of::<F>(), TypeId::of::<GoldilocksField>());
    debug_assert_eq!(TypeId::of::<H>(), TypeId::of::<PoseidonHash>());
    if let Some(view) = device {
        let digest_flat = unsafe { core::slice::from_raw_parts(state.digests.ptr, digests_len + cap_len) };
        let (digests, cap) = rebuild_digests(digest_flat, digests_len, lg_rows, cap_height);
        // SAFETY: F == GoldilocksField and H == PoseidonHash (checked by `cpu_reason`).
        return unsafe {
            MerkleTree {
                leaves: crate::hash::leaf_matrix::LeafMatrix::from_device(device_leaves::<F>(view, cols, 1 << lg_rows)),
                digests: cast_vec::<HashOut<GoldilocksField>, H::Hash>(digests),
                cap: MerkleCap(cast_vec::<HashOut<GoldilocksField>, H::Hash>(cap)),
            }
        };
    }
    #[cfg(feature = "gpu-fault-injection")]
    if inject("rebuild_panic") {
        panic!("injected rebuild_panic");
    }
    if let Some(mut lease) = lease {
        // The leaves are already in the leased buffer: hand it to the tree, no copy.
        let ptr = lease.as_mut_ptr() as *const F;
        let digest_flat = unsafe { core::slice::from_raw_parts(state.digests.ptr, digests_len + cap_len) };
        let (digests, cap) = rebuild_digests(digest_flat, digests_len, lg_rows, cap_height);
        // SAFETY: F == GoldilocksField (repr(transparent) u64); `leaves_len` elements were
        // downloaded into the lease, which `ExternalLeaves` keeps alive and never writes again.
        let leaves = unsafe { crate::hash::leaf_matrix::ExternalLeaves::new(ptr, leaves_len, Box::new(lease)) };
        return unsafe {
            MerkleTree {
                leaves: crate::hash::leaf_matrix::LeafMatrix::from_external(leaves, cols),
                digests: cast_vec::<HashOut<GoldilocksField>, H::Hash>(digests),
                cap: MerkleCap(cast_vec::<HashOut<GoldilocksField>, H::Hash>(cap)),
            }
        };
    }
    let leaves_flat = unsafe { core::slice::from_raw_parts(state.leaves.ptr, leaves_len) };
    let digest_flat = unsafe { core::slice::from_raw_parts(state.digests.ptr, digests_len + cap_len) };

    // One contiguous row-major copy out of the pinned buffer (no per-row allocation).
    let mut leaves: Vec<GoldilocksField> = Vec::with_capacity(leaves_flat.len());
    {
        let spare = &mut leaves.spare_capacity_mut()[..leaves_flat.len()];
        spare.par_chunks_mut(1 << 16).zip(leaves_flat.par_chunks(1 << 16)).for_each(|(dst, src)| {
            for (d, &x) in dst.iter_mut().zip(src) {
                d.write(GoldilocksField(x));
            }
        });
    }
    // SAFETY: every element up to `leaves_flat.len()` was written above.
    unsafe { leaves.set_len(leaves_flat.len()) };
    let (digests, cap) = rebuild_digests(digest_flat, digests_len, lg_rows, cap_height);

    // SAFETY: F == GoldilocksField and H == PoseidonHash (checked by `cpu_reason`), so
    // H::Hash == HashOut<GoldilocksField>; the casts are identities.
    unsafe {
        MerkleTree {
            leaves: crate::hash::leaf_matrix::LeafMatrix::from_flat(cast_vec::<GoldilocksField, F>(leaves), cols),
            digests: cast_vec::<HashOut<GoldilocksField>, H::Hash>(digests),
            cap: MerkleCap(cast_vec::<HashOut<GoldilocksField>, H::Hash>(cap)),
        }
    }
}

/// zeknox's linear digests (per cap subtree, node k has children 2k+2 and 2k+3) followed by the
/// cap, converted to plonky2's recursive digest layout and cap.
fn rebuild_digests(
    digest_flat: &[u64],
    digests_len: usize,
    lg_rows: usize,
    cap_height: usize,
) -> (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>) {
    let (lin_flat, cap_flat) = digest_flat.split_at(digests_len);
    let to_hash = |c: &[u64]| -> HashOut<GoldilocksField> {
        HashOut { elements: [GoldilocksField(c[0]), GoldilocksField(c[1]), GoldilocksField(c[2]), GoldilocksField(c[3])] }
    };
    let caps = 1usize << cap_height;
    let subtree_leaves = 1usize << (lg_rows - cap_height);
    let per_subtree = if subtree_leaves > 1 { 2 * subtree_leaves - 2 } else { 0 };
    // Already in plonky2's layout (reordered on the device by `plonky2_digest_layout`).
    debug_assert_eq!(lin_flat.len(), caps * per_subtree * HASH_U64);
    let digests = lin_flat.chunks_exact(HASH_U64).map(to_hash).collect();
    let cap = cap_flat.chunks_exact(HASH_U64).map(to_hash).collect();
    (digests, cap)
}

/// Position in plonky2's subtree layout of the digest at heap index `k` (see
/// `plonky2_digest_layout` in cuda/quotient.cu).
#[cfg(test)]
fn heap_to_plonky2_index(k: usize, lg_leaves: usize) -> usize {
    let d = (usize::BITS - 1 - (k + 2).leading_zeros()) as usize;
    let index = k + 2 - (1 << d);
    let layer = lg_leaves - d;
    2 * (((index >> 1) << (layer + 1)) + (1 << layer) - 1) + (index & 1)
}

/// `out` is one subtree in plonky2's layout (left subtree || left digest || right digest || right
/// subtree), `lin` the same subtree in zeknox's heap layout, `left` the heap index of the left
/// child of the subtree's root. The reference for the device reordering.
#[cfg(test)]
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
mod layout_tests {
    use super::*;

    #[test]
    fn heap_index_mapping_matches_linear_to_recursive() {
        for lg in 1..=12 {
            let per = (1usize << (lg + 1)) - 2;
            let lin: Vec<u64> = (0..per as u64).flat_map(|k| [k, 0, 0, 0]).collect();
            let mut out = vec![u64::MAX; per];
            linear_to_recursive(&mut out, &lin, 0, &|c: &[u64]| c[0]);
            for k in 0..per {
                assert_eq!(out[heap_to_plonky2_index(k, lg)], k as u64, "lg {lg} heap index {k}");
            }
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    fn pool(n: usize) -> &'static Pool {
        Box::leak(Box::new(Pool {
            size: n,
            free: Mutex::new((0..n).map(|_| OutputSet { leaves: Pinned::empty(), digests: Pinned::empty(), input: Pinned::empty() }).collect()),
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
