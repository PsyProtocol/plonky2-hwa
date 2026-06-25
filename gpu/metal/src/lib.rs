use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lazy_static::lazy_static;
use metal::objc::rc::autoreleasepool;
use metal::*;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_maybe_rayon::*;

pub type GoldilocksDigest = [GoldilocksField; 4];

static METAL_PROFILE_LEVEL: AtomicU8 = AtomicU8::new(0);

#[derive(Default)]
struct MetalHashProfile {
    encode: Duration,
    wait: Duration,
    leaf_encode: Duration,
    leaf_wait: Duration,
    tree_encode: Duration,
    tree_wait: Duration,
    readback: Duration,
}

fn metal_profile_enabled() -> bool {
    METAL_PROFILE_LEVEL.load(Ordering::Relaxed) != 0
}

fn metal_profile_detail_enabled() -> bool {
    METAL_PROFILE_LEVEL.load(Ordering::Relaxed) >= 2
}

pub fn set_profile_enabled(enabled: bool) {
    METAL_PROFILE_LEVEL.store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
}

pub fn set_profile_detail_enabled(enabled: bool) {
    METAL_PROFILE_LEVEL.store(if enabled { 2 } else { 0 }, Ordering::Relaxed);
}

fn flatten_leaves_parallel(
    leaves: &[Vec<GoldilocksField>],
    leaf_length: usize,
) -> Vec<GoldilocksField> {
    assert!(leaves.iter().all(|leaf| leaf.len() == leaf_length));

    let mut leaf_u64s = Vec::with_capacity(leaves.len() * leaf_length);
    // SAFETY: Every chunk is initialized exactly once below, and all leaf lengths are checked
    // before exposing the initialized Vec to callers.
    unsafe {
        leaf_u64s.set_len(leaves.len() * leaf_length);
    }
    leaf_u64s
        .par_chunks_mut(leaf_length)
        .zip(leaves.par_iter())
        .for_each(|(dst, leaf)| dst.copy_from_slice(leaf));
    leaf_u64s
}

fn get_size_for_count(count: usize) -> MTLSize {
    const MAX_DIM: usize = 32768;
    if count <= MAX_DIM {
        MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        }
    } else {
        MTLSize {
            width: MAX_DIM as u64,
            height: count.div_ceil(MAX_DIM) as u64,
            depth: 1,
        }
    }
}

const SHADERLIB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher.metallib"
));

fn shader_source() -> String {
    const PARTS: &[&str] = &[
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/u128.h.metal")),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/goldilocks.metal"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/poseidon_goldilocks_mds.metal"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/poseidon_goldilocks.metal"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/poseidon_merkle_hasher.metal"
        )),
    ];

    let mut source = String::from("#include <metal_stdlib>\n");
    for part in PARTS {
        for line in part.lines() {
            if line.trim_start().starts_with("#include") {
                continue;
            }
            source.push_str(line);
            source.push('\n');
        }
    }
    source
}

pub fn is_available() -> bool {
    Device::system_default().is_some()
}

pub struct MetalRuntime {
    device: Mutex<Device>,
    pipeline_poseidon_hash_leaves: ComputePipelineState,
    pipeline_poseidon_hash_leaves_poly_major: ComputePipelineState,
    pipeline_poseidon_hash_tree_level: ComputePipelineState,
    command_queue: CommandQueue,
    init_time: u128,
}

unsafe impl Sync for MetalRuntime {}

lazy_static! {
    static ref RUNTIME: MetalRuntime = {
        let device = Device::system_default().expect("Metal device is not available");
        let lib = device
            .new_library_with_source(&shader_source(), &CompileOptions::new())
            .or_else(|source_error| {
                device.new_library_with_data(SHADERLIB).map_err(|data_error| {
                    format!(
                        "failed to compile Metal shader source: {source_error}; failed to load bundled metallib: {data_error}"
                    )
                })
            })
            .unwrap();
        let command_queue = device.new_command_queue();

        let f_poseidon_hash_leaves = lib.get_function("poseidon_hash_leaves", None).unwrap();
        let f_poseidon_hash_leaves_poly_major = lib
            .get_function("poseidon_hash_leaves_poly_major", None)
            .unwrap();
        let f_poseidon_hash_tree_level =
            lib.get_function("poseidon_hash_tree_level", None).unwrap();
        let pipeline_poseidon_hash_leaves = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_leaves)
            .unwrap();
        let pipeline_poseidon_hash_leaves_poly_major = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_leaves_poly_major)
            .unwrap();
        let pipeline_poseidon_hash_tree_level = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_tree_level)
            .unwrap();

        MetalRuntime {
            device: Mutex::new(device),
            command_queue,
            pipeline_poseidon_hash_leaves,
            pipeline_poseidon_hash_leaves_poly_major,
            pipeline_poseidon_hash_tree_level,
            init_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros(),
        }
    };
}

impl MetalRuntime {
    pub fn get() -> &'static MetalRuntime {
        &RUNTIME
    }

    pub fn warm_up(&self) -> u128 {
        self.init_time
    }

    pub fn hash_merkle_tree_digests(
        leaves: &[Vec<GoldilocksField>],
        cap_height: usize,
    ) -> Vec<GoldilocksDigest> {
        let profile = metal_profile_enabled();
        let total_start = Instant::now();
        let leaf_length = leaves[0].len();
        let leaf_count = leaves.len();
        let tree_height = leaf_count.ilog2() as usize;

        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;
        let total_digests = total_tree_hashes - (num_caps * 2 - 1);

        let flatten_start = Instant::now();
        let leaf_u64s = flatten_leaves_parallel(leaves, leaf_length);
        let flatten_time = flatten_start.elapsed();
        let mut input_buffer_time = Duration::ZERO;
        let mut hash_profile = MetalHashProfile::default();
        let digests = autoreleasepool::<Vec<GoldilocksDigest>, _>(|| {
            let input_buffer_start = Instant::now();
            let leaves_buffer = Self::get()
                .device
                .lock()
                .unwrap()
                .new_buffer_with_bytes_no_copy(
                    unsafe { std::mem::transmute(leaf_u64s.as_ptr()) },
                    (leaf_length * leaf_count * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
            input_buffer_time = input_buffer_start.elapsed();
            let (result, current_hash_profile) = Self::get().hash_merkle_tree_buf_profiled(
                leaves_buffer,
                tree_height,
                leaf_length,
                cap_height,
                total_digests,
                false,
            );
            hash_profile = current_hash_profile;
            result
        });

        if profile {
            eprintln!(
                "[metal-profile] leaves={} leaf_len={} cap_height={} flatten={:?} input_buffer={:?} encode={:?} wait={:?} readback={:?} total={:?}",
                leaf_count,
                leaf_length,
                cap_height,
                flatten_time,
                input_buffer_time,
                hash_profile.encode,
                hash_profile.wait,
                hash_profile.readback,
                total_start.elapsed(),
            );
            if metal_profile_detail_enabled() {
                eprintln!(
                    "[metal-profile-detail] leaves={} leaf_len={} leaf_encode={:?} leaf_wait={:?} tree_encode={:?} tree_wait={:?}",
                    leaf_count,
                    leaf_length,
                    hash_profile.leaf_encode,
                    hash_profile.leaf_wait,
                    hash_profile.tree_encode,
                    hash_profile.tree_wait,
                );
            }
        }

        digests
    }

    pub fn hash_merkle_tree_poly_major_digests(
        poly_values: &[GoldilocksField],
        leaf_count: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> Vec<GoldilocksDigest> {
        let profile = metal_profile_enabled();
        let total_start = Instant::now();
        assert_eq!(poly_values.len(), leaf_count * leaf_length);
        let tree_height = leaf_count.ilog2() as usize;

        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;
        let total_digests = total_tree_hashes - (num_caps * 2 - 1);

        let mut input_buffer_time = Duration::ZERO;
        let mut hash_profile = MetalHashProfile::default();
        let digests = autoreleasepool::<Vec<GoldilocksDigest>, _>(|| {
            let input_buffer_start = Instant::now();
            let values_buffer = Self::get()
                .device
                .lock()
                .unwrap()
                .new_buffer_with_bytes_no_copy(
                    unsafe { std::mem::transmute(poly_values.as_ptr()) },
                    (poly_values.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
            input_buffer_time = input_buffer_start.elapsed();
            let (result, current_hash_profile) = Self::get().hash_merkle_tree_buf_profiled(
                values_buffer,
                tree_height,
                leaf_length,
                cap_height,
                total_digests,
                true,
            );
            hash_profile = current_hash_profile;
            result
        });

        if profile {
            eprintln!(
                "[metal-profile] layout=poly-major leaves={} leaf_len={} cap_height={} flatten=0ns input_buffer={:?} encode={:?} wait={:?} readback={:?} total={:?}",
                leaf_count,
                leaf_length,
                cap_height,
                input_buffer_time,
                hash_profile.encode,
                hash_profile.wait,
                hash_profile.readback,
                total_start.elapsed(),
            );
            if metal_profile_detail_enabled() {
                eprintln!(
                    "[metal-profile-detail] layout=poly-major leaves={} leaf_len={} leaf_encode={:?} leaf_wait={:?} tree_encode={:?} tree_wait={:?}",
                    leaf_count,
                    leaf_length,
                    hash_profile.leaf_encode,
                    hash_profile.leaf_wait,
                    hash_profile.tree_encode,
                    hash_profile.tree_wait,
                );
            }
        }

        digests
    }

    fn hash_merkle_tree_buf_profiled(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
        total_digests: usize,
        poly_major: bool,
    ) -> (Vec<GoldilocksDigest>, MetalHashProfile) {
        let leaf_count = 1usize << tree_height;
        assert!(
            cap_height < tree_height,
            "cap height must be less than tree height"
        );

        let num_layers = tree_height - cap_height;
        let tree_length = total_digests >> cap_height;
        let mut uniforms: [u32; 5] = [
            0,
            tree_length as u32,
            num_layers as u32,
            leaf_length as u32,
            tree_height as u32,
        ];

        let mut digests = Vec::<GoldilocksDigest>::with_capacity(total_digests);
        // SAFETY: The no-copy Metal buffer below covers the whole Vec backing allocation, and the
        // command buffer is waited on before the Vec is read.
        unsafe {
            digests.set_len(total_digests);
        }
        let digests_buffer = self.device.lock().unwrap().new_buffer_with_bytes_no_copy(
            digests.as_mut_ptr() as *const core::ffi::c_void,
            (total_digests * 4 * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        if metal_profile_detail_enabled() {
            return self.hash_merkle_tree_buf_profile_detail(
                leaves_buffer,
                digests,
                digests_buffer,
                uniforms,
                leaf_count,
                num_layers,
                poly_major,
            );
        }

        let encode_start = Instant::now();
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        let leaf_pipeline = if poly_major {
            &self.pipeline_poseidon_hash_leaves_poly_major
        } else {
            &self.pipeline_poseidon_hash_leaves
        };
        encoder.set_compute_pipeline_state(leaf_pipeline);
        encoder.set_buffer(0, Some(&leaves_buffer), 0);
        encoder.set_buffer(1, Some(&digests_buffer), 0);

        let uniforms_buffer = self.device.lock().unwrap().new_buffer_with_data(
            unsafe { std::mem::transmute(uniforms.as_ptr()) },
            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeWriteCombined,
        );
        encoder.set_buffer(2, Some(&uniforms_buffer), 0);

        let threads_per_grid = get_size_for_count(leaf_count);
        let thread_group_size = MTLSize {
            width: leaf_pipeline.thread_execution_width(),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, thread_group_size);
        if num_layers > 1 {
            encoder.memory_barrier_with_resources(&[digests_buffer.as_ref()]);
        }

        for i in 1..num_layers {
            encoder.set_compute_pipeline_state(&self.pipeline_poseidon_hash_tree_level);

            let work_count = leaf_count >> i;
            uniforms[0] = i as u32;

            encoder.set_bytes(
                1,
                (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                uniforms.as_ptr() as *mut core::ffi::c_void,
            );
            encoder.set_buffer(0, Some(&digests_buffer), 0);

            let threads_per_grid = get_size_for_count(work_count);
            let thread_group_size = MTLSize {
                width: self
                    .pipeline_poseidon_hash_tree_level
                    .thread_execution_width(),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(threads_per_grid, thread_group_size);
            if i + 1 < num_layers {
                encoder.memory_barrier_with_resources(&[digests_buffer.as_ref()]);
            }
        }

        encoder.end_encoding();
        let encode_time = encode_start.elapsed();

        command_buffer.commit();
        let wait_start = Instant::now();
        command_buffer.wait_until_completed();
        let wait_time = wait_start.elapsed();

        (
            digests,
            MetalHashProfile {
                encode: encode_time,
                wait: wait_time,
                readback: Duration::ZERO,
                ..MetalHashProfile::default()
            },
        )
    }

    fn hash_merkle_tree_buf_profile_detail(
        &self,
        leaves_buffer: Buffer,
        digests: Vec<GoldilocksDigest>,
        digests_buffer: Buffer,
        mut uniforms: [u32; 5],
        leaf_count: usize,
        num_layers: usize,
        poly_major: bool,
    ) -> (Vec<GoldilocksDigest>, MetalHashProfile) {
        let leaf_encode_start = Instant::now();
        let leaf_command_buffer = self.command_queue.new_command_buffer();
        let leaf_compute_pass_descriptor = ComputePassDescriptor::new();
        let leaf_encoder = leaf_command_buffer
            .compute_command_encoder_with_descriptor(leaf_compute_pass_descriptor);

        let leaf_pipeline = if poly_major {
            &self.pipeline_poseidon_hash_leaves_poly_major
        } else {
            &self.pipeline_poseidon_hash_leaves
        };
        leaf_encoder.set_compute_pipeline_state(leaf_pipeline);
        leaf_encoder.set_buffer(0, Some(&leaves_buffer), 0);
        leaf_encoder.set_buffer(1, Some(&digests_buffer), 0);

        let uniforms_buffer = self.device.lock().unwrap().new_buffer_with_data(
            unsafe { std::mem::transmute(uniforms.as_ptr()) },
            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeWriteCombined,
        );
        leaf_encoder.set_buffer(2, Some(&uniforms_buffer), 0);

        let threads_per_grid = get_size_for_count(leaf_count);
        let thread_group_size = MTLSize {
            width: leaf_pipeline.thread_execution_width(),
            height: 1,
            depth: 1,
        };
        leaf_encoder.dispatch_threads(threads_per_grid, thread_group_size);
        leaf_encoder.end_encoding();
        let leaf_encode_time = leaf_encode_start.elapsed();

        leaf_command_buffer.commit();
        let leaf_wait_start = Instant::now();
        leaf_command_buffer.wait_until_completed();
        let leaf_wait_time = leaf_wait_start.elapsed();

        let tree_encode_start = Instant::now();
        let tree_command_buffer = self.command_queue.new_command_buffer();
        let tree_compute_pass_descriptor = ComputePassDescriptor::new();
        let tree_encoder = tree_command_buffer
            .compute_command_encoder_with_descriptor(tree_compute_pass_descriptor);

        for i in 1..num_layers {
            tree_encoder.set_compute_pipeline_state(&self.pipeline_poseidon_hash_tree_level);

            let work_count = leaf_count >> i;
            uniforms[0] = i as u32;

            tree_encoder.set_bytes(
                1,
                (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                uniforms.as_ptr() as *mut core::ffi::c_void,
            );
            tree_encoder.set_buffer(0, Some(&digests_buffer), 0);

            let threads_per_grid = get_size_for_count(work_count);
            let thread_group_size = MTLSize {
                width: self
                    .pipeline_poseidon_hash_tree_level
                    .thread_execution_width(),
                height: 1,
                depth: 1,
            };

            tree_encoder.dispatch_threads(threads_per_grid, thread_group_size);
            if i + 1 < num_layers {
                tree_encoder.memory_barrier_with_resources(&[digests_buffer.as_ref()]);
            }
        }

        tree_encoder.end_encoding();
        let tree_encode_time = tree_encode_start.elapsed();

        tree_command_buffer.commit();
        let tree_wait_start = Instant::now();
        tree_command_buffer.wait_until_completed();
        let tree_wait_time = tree_wait_start.elapsed();

        (
            digests,
            MetalHashProfile {
                encode: leaf_encode_time + tree_encode_time,
                wait: leaf_wait_time + tree_wait_time,
                leaf_encode: leaf_encode_time,
                leaf_wait: leaf_wait_time,
                tree_encode: tree_encode_time,
                tree_wait: tree_wait_time,
                readback: Duration::ZERO,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reverse_bits(value: usize, num_bits: usize) -> usize {
        value
            .reverse_bits()
            .overflowing_shr(usize::BITS - num_bits as u32)
            .0
    }

    #[test]
    fn metal_runtime_hashes_large_merkle_tree_when_available() {
        if !is_available() {
            eprintln!("skipping Metal runtime smoke test: no Metal device available");
            return;
        }

        let leaves = (0..(1usize << 13))
            .map(|row| {
                (0..8)
                    .map(|col| GoldilocksField((row * 17 + col * 3) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let cap_height = 4;
        let digests = MetalRuntime::hash_merkle_tree_digests(&leaves, cap_height);

        assert_eq!(digests.len(), 2 * (leaves.len() - (1usize << cap_height)));
        assert_ne!(digests[0], [GoldilocksField(0); 4]);
    }

    #[test]
    fn poly_major_hash_matches_row_major_hash_when_available() {
        if !is_available() {
            eprintln!("skipping Metal poly-major test: no Metal device available");
            return;
        }

        let leaf_count = 1usize << 13;
        let leaf_length = 20;
        let tree_height = leaf_count.ilog2() as usize;
        let cap_height = 4;
        let poly_values = (0..leaf_length)
            .flat_map(|poly| {
                (0..leaf_count).map(move |point| {
                    GoldilocksField(((poly * 31 + point * 17 + 7) as u64) & 0xffff_ffff)
                })
            })
            .collect::<Vec<_>>();
        let leaves = (0..leaf_count)
            .map(|leaf_index| {
                let point_index = reverse_bits(leaf_index, tree_height);
                (0..leaf_length)
                    .map(|poly_index| poly_values[poly_index * leaf_count + point_index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let row_major = MetalRuntime::hash_merkle_tree_digests(&leaves, cap_height);
        let poly_major = MetalRuntime::hash_merkle_tree_poly_major_digests(
            &poly_values,
            leaf_count,
            leaf_length,
            cap_height,
        );

        assert_eq!(poly_major, row_major);
    }
}
