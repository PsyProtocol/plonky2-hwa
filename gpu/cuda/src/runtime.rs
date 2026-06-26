//! High-level, host-side CUDA runtime for plonky2 commitments.
//!
//! This wraps the low-level device-pointer FFI (`crate::merkle_tree_from_coeffs`) behind a
//! Metal-style host-in / host-out API: callers pass host slices, the runtime owns the CUDA
//! context, streams and device memory, and returns host `Vec`s. It is only compiled with the
//! `runtime` feature (which also builds the native kernels).
//!
//! Phase 1 scope: the LDE + Poseidon-Merkle commitment for `GoldilocksField` / `PoseidonHash`.
//! All sizes are runtime parameters, so it is circuit-general.
//!
//! ## Conventions that must be validated on-device (see cuda-plan.md §8)
//! - Device buffer layout mirrors the reference `from_coeffs_with_gpu`:
//!   `[coeffs | (pad) ext-poly-major | leaf-major-LDE@0 after transpose | digests@2*ext]`.
//! - `root_table`  = `fft_root_table(degree).concat()`            (ifft over the base domain).
//! - `root_table2` = `fft_root_table(degree << rate_bits).concat()` (coset fft over the LDE domain).
//! - `shift_powers` = coset-shift powers `g^i`, `i in 0..degree`.
//! - Returned digests/caps ordering matches the CPU Merkle layout after the same remap the Metal
//!   backend performs (`get_node_hash_index_in_digests`).

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::Device;
use rustacuda::memory::{CopyDestination, DeviceBuffer};
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::CudaFlags;

/// C-ABI context handed to the kernels. Must match `struct CudaInvContext { cudaStream_t stream;
/// cudaStream_t stream2; }` in `plonky2_gpu.cu`. We rely on `rustacuda::Stream` being a transparent
/// wrapper around a raw `CUstream` (a single pointer), exactly as the reference implementation does.
#[repr(C)]
struct CudaInnerContext {
    stream: Stream,
    stream2: Stream,
}

/// Twiddle tables / coset-shift powers cached on-device, keyed by `(degree, rate_bits)`. Within a
/// proof every commitment shares the same base degree, so these upload once and are then reused.
struct TableCache {
    degree: usize,
    rate_bits: usize,
    root_table: DeviceBuffer<u64>,
    root_table2: DeviceBuffer<u64>,
    shift_powers: DeviceBuffer<u64>,
}

struct Inner {
    // Field order matters: `context` must outlive the buffers/streams, and is dropped last.
    inner_ctx: CudaInnerContext,
    /// Resident scratch buffer, grown on demand and reused across commitments to avoid a large
    /// allocation per call.
    cache: Option<DeviceBuffer<u64>>,
    cache_len: usize,
    tables: Option<TableCache>,
    context: Context,
}

// SAFETY: every access goes through the `Mutex` in `CudaRuntime`, and we make the context current
// on the accessing thread before issuing any device work.
unsafe impl Send for Inner {}

/// Process-wide CUDA runtime. Obtain via [`CudaRuntime::get`].
pub struct CudaRuntime {
    inner: Mutex<Inner>,
}

/// Output of [`CudaRuntime::commit_from_coeffs`].
pub struct CommitResult {
    /// `num_digests_and_caps` hashes, each 4 `u64` limbs, in the kernel's native ordering.
    pub digests_and_caps: Vec<[u64; 4]>,
    /// Leaf-major LDE values, `leaf_count` leaves of `ext_poly_num` limbs each, in the same leaf
    /// order as the CPU Merkle tree. Reconstruct `MerkleTree::leaves` by chunking on `ext_poly_num`.
    pub leaves_flat: Vec<u64>,
}

const SALT_SIZE: usize = 4;

fn build_runtime() -> Option<CudaRuntime> {
    rustacuda::init(CudaFlags::empty()).ok()?;
    let device = Device::get_device(0).ok()?;
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).ok()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
    let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
    Some(CudaRuntime {
        inner: Mutex::new(Inner {
            inner_ctx: CudaInnerContext { stream, stream2 },
            cache: None,
            cache_len: 0,
            tables: None,
            context,
        }),
    })
}

fn runtime_slot() -> &'static Option<CudaRuntime> {
    static RUNTIME: OnceLock<Option<CudaRuntime>> = OnceLock::new();
    RUNTIME.get_or_init(build_runtime)
}

/// Returns `true` if a CUDA device is present and a context could be created.
pub fn is_available() -> bool {
    runtime_slot().is_some()
}

impl CudaRuntime {
    /// Returns the process-wide runtime, or panics if no device is available. Call
    /// [`is_available`] first.
    pub fn get() -> &'static CudaRuntime {
        runtime_slot()
            .as_ref()
            .expect("CudaRuntime::get called but no CUDA device is available")
    }

    /// Runs LDE + Poseidon-Merkle on the GPU for poly-major coefficients.
    ///
    /// - `coeffs_poly_major`: `poly_num * degree` limbs, polynomial `p` occupying
    ///   `[p*degree .. (p+1)*degree)`.
    /// - `degree` must be a power of two; `salt_size` is `SALT_SIZE` when blinding else `0`.
    /// - `root_table`, `root_table2`, `shift_powers`: see module docs.
    pub fn commit_from_coeffs(
        &self,
        coeffs_poly_major: &[u64],
        poly_num: usize,
        degree: usize,
        rate_bits: usize,
        cap_height: usize,
        salt_size: usize,
        root_table: &[u64],
        root_table2: &[u64],
        shift_powers: &[u64],
    ) -> Result<CommitResult, String> {
        assert!(degree.is_power_of_two(), "degree must be a power of two");
        assert_eq!(coeffs_poly_major.len(), poly_num * degree);
        assert!(salt_size == 0 || salt_size == SALT_SIZE);

        let log_len = degree.trailing_zeros() as i32;
        let ext_poly_num = poly_num + salt_size;
        let values_flatten_len = poly_num * degree;
        let ext_values_flatten_len = ext_poly_num * degree * (1 << rate_bits);
        let pad_extvalues_len = ext_values_flatten_len;

        let leaf_count = degree << rate_bits;
        let len_cap = 1usize << cap_height;
        let num_digests = 2 * (leaf_count - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        // Layout: [0, ext)=leaf-major LDE (after transpose), [ext, 2*ext)=scratch, then digests.
        let digests_offset = 2 * ext_values_flatten_len;
        let cache_len = digests_offset + num_digests_and_caps * 4;

        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        CurrentContext::set_current(&inner.context).map_err(|e| format!("{e:?}"))?;

        // Grow the resident scratch buffer on demand; reuse it otherwise (no per-call alloc).
        if inner.cache_len < cache_len {
            inner.cache = Some(
                unsafe { DeviceBuffer::uninitialized(cache_len) }.map_err(|e| format!("{e:?}"))?,
            );
            inner.cache_len = cache_len;
        }

        // Upload twiddle tables only when the (degree, rate_bits) changes (once per proof).
        let tables_stale = match &inner.tables {
            Some(t) => t.degree != degree || t.rate_bits != rate_bits,
            None => true,
        };
        if tables_stale {
            inner.tables = Some(TableCache {
                degree,
                rate_bits,
                root_table: DeviceBuffer::from_slice(root_table).map_err(|e| format!("{e:?}"))?,
                root_table2: DeviceBuffer::from_slice(root_table2).map_err(|e| format!("{e:?}"))?,
                shift_powers: DeviceBuffer::from_slice(shift_powers)
                    .map_err(|e| format!("{e:?}"))?,
            });
        }

        // Upload coefficients into the head of the resident buffer.
        let cache = inner.cache.as_mut().unwrap();
        cache
            .split_at_mut(values_flatten_len)
            .0
            .copy_from(coeffs_poly_major)
            .map_err(|e| format!("{e:?}"))?;
        let base_ptr = cache.as_device_ptr().as_raw_mut();

        let tables = inner.tables.as_mut().unwrap();
        let rt = tables.root_table.as_device_ptr().as_raw();
        let rt2 = tables.root_table2.as_device_ptr().as_raw();
        let sp = tables.shift_powers.as_device_ptr().as_raw();

        // Kernel pipeline (ifft -> lde -> coset fft -> reverse bits -> hash -> reduce -> transpose).
        let err = unsafe {
            crate::merkle_tree_from_coeffs(
                base_ptr,
                base_ptr,
                poly_num as i32,
                degree as i32,
                log_len,
                rt,
                rt2,
                sp,
                rate_bits as i32,
                salt_size as i32,
                cap_height as i32,
                pad_extvalues_len as i32,
                &mut inner.inner_ctx as *mut CudaInnerContext as *mut c_void,
            )
        };
        if err.code != 0 {
            return Err(String::from(err));
        }

        // Read back the leaf-major leaves ([0, ext)) and the digests/caps (at `digests_offset`).
        let cache = inner.cache.as_ref().unwrap();
        let mut leaves_flat = vec![0u64; ext_values_flatten_len];
        cache
            .split_at(ext_values_flatten_len)
            .0
            .copy_to(&mut leaves_flat[..])
            .map_err(|e| format!("{e:?}"))?;

        let mut digests_flat = vec![0u64; num_digests_and_caps * 4];
        cache
            .split_at(digests_offset)
            .1
            .split_at(digests_flat.len())
            .0
            .copy_to(&mut digests_flat[..])
            .map_err(|e| format!("{e:?}"))?;
        let digests_and_caps = digests_flat
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();

        Ok(CommitResult {
            digests_and_caps,
            leaves_flat,
        })
    }
}
