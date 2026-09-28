use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lazy_static::lazy_static;
use metal::objc::rc::autoreleasepool;
use metal::*;
use plonky2_field::fft::fft_root_table;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

pub type GoldilocksDigest = [GoldilocksField; 4];

static METAL_PROFILE_LEVEL: AtomicU8 = AtomicU8::new(0);
static METAL_LDE_ENABLED: AtomicBool = AtomicBool::new(cfg!(feature = "lde"));
static METAL_LEAF_THREADGROUP_WIDTH: AtomicU64 = AtomicU64::new(0);

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

pub fn set_lde_enabled(enabled: bool) {
    METAL_LDE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn lde_enabled() -> bool {
    METAL_LDE_ENABLED.load(Ordering::Relaxed)
}

pub fn set_leaf_threadgroup_width(width: u64) {
    METAL_LEAF_THREADGROUP_WIDTH.store(width, Ordering::Relaxed);
}

pub fn leaf_threadgroup_width() -> u64 {
    METAL_LEAF_THREADGROUP_WIDTH.load(Ordering::Relaxed)
}

fn leaf_threadgroup_size(pipeline: &ComputePipelineState) -> MTLSize {
    let requested = leaf_threadgroup_width();
    let max_supported = pipeline.max_total_threads_per_threadgroup();
    let default_width = pipeline.thread_execution_width();
    let width = if requested == 0 {
        default_width
    } else {
        requested.min(max_supported).max(default_width)
    };
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
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
    assert!(
        count <= u32::MAX as usize,
        "Metal shader thread index uses u32; count must fit in u32"
    );
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

fn goldilocks_vec_for_gpu_overwrite(len: usize) -> Vec<GoldilocksField> {
    let mut output = Vec::with_capacity(len);
    // SAFETY: Callers bind this Vec as a Metal output buffer and wait for the command buffer before
    // returning it. This helper is only sound for kernels that write every element before any Rust
    // read; it must not be used with partial-output kernels.
    unsafe {
        output.set_len(len);
    }
    output
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
    pipeline_poseidon_hash_leaves_poly_major_16: ComputePipelineState,
    pipeline_poseidon_hash_tree_level: ComputePipelineState,
    pipeline_fft_init_reverse_zero: ComputePipelineState,
    pipeline_fft_init_coset_lde: ComputePipelineState,
    pipeline_fft_coset_lde_single: ComputePipelineState,
    pipeline_fft_init_coset_lde_local: ComputePipelineState,
    pipeline_fft_stage: ComputePipelineState,
    pipeline_fft_stage_pair: ComputePipelineState,
    pipeline_fft_stage_triple: ComputePipelineState,
    pipeline_quotient_perm_reduce: ComputePipelineState,
    fft_root_buffers: Mutex<HashMap<usize, Vec<Buffer>>>,
    fft_flat_root_buffers: Mutex<HashMap<usize, Buffer>>,
    coset_shift_buffers: Mutex<HashMap<usize, Buffer>>,
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
                eprintln!("failed to compile Metal shader source: {source_error}");
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
        let f_poseidon_hash_leaves_poly_major_16 = lib
            .get_function("poseidon_hash_leaves_poly_major_16", None)
            .unwrap();
        let f_poseidon_hash_tree_level =
            lib.get_function("poseidon_hash_tree_level", None).unwrap();
        let f_fft_init_reverse_zero = lib.get_function("fft_init_reverse_zero", None).unwrap();
        let f_fft_init_coset_lde = lib.get_function("fft_init_coset_lde", None).unwrap();
        let f_fft_coset_lde_single = lib.get_function("fft_coset_lde_single", None).unwrap();
        let f_fft_init_coset_lde_local =
            lib.get_function("fft_init_coset_lde_local", None).unwrap();
        let f_fft_stage = lib.get_function("fft_stage", None).unwrap();
        let f_fft_stage_pair = lib.get_function("fft_stage_pair", None).unwrap();
        let f_fft_stage_triple = lib.get_function("fft_stage_triple", None).unwrap();
        let f_quotient_perm_reduce = lib.get_function("quotient_perm_reduce", None).unwrap();
        let pipeline_poseidon_hash_leaves = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_leaves)
            .unwrap();
        let pipeline_poseidon_hash_leaves_poly_major = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_leaves_poly_major)
            .unwrap();
        let pipeline_poseidon_hash_leaves_poly_major_16 = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_leaves_poly_major_16)
            .unwrap();
        let pipeline_poseidon_hash_tree_level = device
            .new_compute_pipeline_state_with_function(&f_poseidon_hash_tree_level)
            .unwrap();
        let pipeline_fft_init_reverse_zero = device
            .new_compute_pipeline_state_with_function(&f_fft_init_reverse_zero)
            .unwrap();
        let pipeline_fft_init_coset_lde = device
            .new_compute_pipeline_state_with_function(&f_fft_init_coset_lde)
            .unwrap();
        let pipeline_fft_coset_lde_single = device
            .new_compute_pipeline_state_with_function(&f_fft_coset_lde_single)
            .unwrap();
        let pipeline_fft_init_coset_lde_local = device
            .new_compute_pipeline_state_with_function(&f_fft_init_coset_lde_local)
            .unwrap();
        let pipeline_fft_stage = device
            .new_compute_pipeline_state_with_function(&f_fft_stage)
            .unwrap();
        let pipeline_fft_stage_pair = device
            .new_compute_pipeline_state_with_function(&f_fft_stage_pair)
            .unwrap();
        let pipeline_fft_stage_triple = device
            .new_compute_pipeline_state_with_function(&f_fft_stage_triple)
            .unwrap();
        let pipeline_quotient_perm_reduce = device
            .new_compute_pipeline_state_with_function(&f_quotient_perm_reduce)
            .unwrap();

        MetalRuntime {
            device: Mutex::new(device),
            command_queue,
            pipeline_poseidon_hash_leaves,
            pipeline_poseidon_hash_leaves_poly_major,
            pipeline_poseidon_hash_leaves_poly_major_16,
            pipeline_poseidon_hash_tree_level,
            pipeline_fft_init_reverse_zero,
            pipeline_fft_init_coset_lde,
            pipeline_fft_coset_lde_single,
            pipeline_fft_init_coset_lde_local,
            pipeline_fft_stage,
            pipeline_fft_stage_pair,
            pipeline_fft_stage_triple,
            pipeline_quotient_perm_reduce,
            fft_root_buffers: Mutex::new(HashMap::new()),
            fft_flat_root_buffers: Mutex::new(HashMap::new()),
            coset_shift_buffers: Mutex::new(HashMap::new()),
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

    fn root_buffers_for_len(&self, fft_len: usize) -> Vec<Buffer> {
        let mut cache = self.fft_root_buffers.lock().unwrap();
        if let Some(root_buffers) = cache.get(&fft_len) {
            return root_buffers.to_vec();
        }

        let root_table = fft_root_table::<GoldilocksField>(fft_len);
        let device = self.device.lock().unwrap();
        let root_buffers = root_table
            .iter()
            .map(|row| {
                device.new_buffer_with_data(
                    row.as_ptr() as *const core::ffi::c_void,
                    (row.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::CPUCacheModeWriteCombined,
                )
            })
            .collect::<Vec<_>>();
        cache.insert(fft_len, root_buffers.clone());
        root_buffers
    }

    fn flat_root_buffer_for_len(&self, fft_len: usize) -> Buffer {
        let mut cache = self.fft_flat_root_buffers.lock().unwrap();
        if let Some(buffer) = cache.get(&fft_len) {
            return buffer.clone();
        }

        let root_table = fft_root_table::<GoldilocksField>(fft_len);
        let mut roots = Vec::with_capacity(fft_len);
        for row in root_table {
            roots.extend(row);
        }
        debug_assert_eq!(roots.len(), fft_len);

        let buffer = self.device.lock().unwrap().new_buffer_with_data(
            roots.as_ptr() as *const core::ffi::c_void,
            (roots.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::CPUCacheModeWriteCombined,
        );
        cache.insert(fft_len, buffer.clone());
        buffer
    }

    fn coset_shift_buffer_for_len(&self, coeff_len: usize) -> Buffer {
        let mut cache = self.coset_shift_buffers.lock().unwrap();
        if let Some(buffer) = cache.get(&coeff_len) {
            return buffer.clone();
        }

        let mut shift_powers = vec![GoldilocksField::ZERO; coeff_len];
        let coset_shift = GoldilocksField::coset_shift();
        let mut shift_power = GoldilocksField::ONE;
        for power in &mut shift_powers {
            *power = shift_power;
            shift_power *= coset_shift;
        }

        let buffer = self.device.lock().unwrap().new_buffer_with_data(
            shift_powers.as_ptr() as *const core::ffi::c_void,
            (shift_powers.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::CPUCacheModeWriteCombined,
        );
        cache.insert(coeff_len, buffer.clone());
        buffer
    }

    pub fn coset_lde_values_poly_major(
        &self,
        coeffs: &[GoldilocksField],
        poly_count: usize,
        coeff_len: usize,
        rate_bits: usize,
    ) -> Vec<GoldilocksField> {
        let coeff_count = poly_count
            .checked_mul(coeff_len)
            .expect("Metal LDE coefficient count overflow");
        assert_eq!(coeffs.len(), coeff_count);
        assert!(coeff_len.is_power_of_two());

        let fft_len = coeff_len << rate_bits;
        let log_fft_len = fft_len.ilog2() as usize;
        assert!(log_fft_len <= 32);
        assert!(rate_bits <= log_fft_len);
        let value_count = poly_count
            .checked_mul(fft_len)
            .expect("Metal LDE value count overflow");
        assert!(
            value_count <= u32::MAX as usize,
            "Metal LDE value count must fit in u32 shader indices"
        );

        let root_buffers = self.root_buffers_for_len(fft_len);
        let flat_roots_buffer = self.flat_root_buffer_for_len(fft_len);
        let shift_powers_buffer = self.coset_shift_buffer_for_len(coeff_len);
        let mut output = goldilocks_vec_for_gpu_overwrite(value_count);
        let local_tile_log = log_fft_len.min(11);
        let use_local_init = rate_bits < local_tile_log;

        autoreleasepool::<(), _>(|| {
            let device = self.device.lock().unwrap();
            let coeffs_buffer = device.new_buffer_with_bytes_no_copy(
                coeffs.as_ptr() as *const core::ffi::c_void,
                (coeffs.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let output_buffer = device.new_buffer_with_bytes_no_copy(
                output.as_mut_ptr() as *const core::ffi::c_void,
                (output.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            drop(device);

            let command_buffer = self.command_queue.new_command_buffer();
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            // This CUDA-style path is correct but under-utilizes Apple GPUs for large FFTs:
            // one threadgroup owns an entire polynomial, so dispatch overhead drops but occupancy
            // is too low for the large global-memory stage loops.
            let use_single_kernel = false;
            if use_single_kernel {
                let thread_count = self
                    .pipeline_fft_coset_lde_single
                    .max_total_threads_per_threadgroup()
                    .min(1024);
                let uniforms = [
                    poly_count as u32,
                    coeff_len as u32,
                    fft_len as u32,
                    log_fft_len as u32,
                    rate_bits as u32,
                    thread_count as u32,
                ];

                encoder.set_compute_pipeline_state(&self.pipeline_fft_coset_lde_single);
                encoder.set_buffer(0, Some(&coeffs_buffer), 0);
                encoder.set_buffer(1, Some(&shift_powers_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_buffer(3, Some(&flat_roots_buffer), 0);
                encoder.set_bytes(
                    4,
                    (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                    uniforms.as_ptr() as *mut core::ffi::c_void,
                );
                let threadgroups_per_grid = get_size_for_count(poly_count);
                let thread_group_size = MTLSize {
                    width: thread_count,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(threadgroups_per_grid, thread_group_size);
            } else {
                let first_global_stage = if use_local_init {
                    let uniforms = [
                        poly_count as u32,
                        coeff_len as u32,
                        fft_len as u32,
                        log_fft_len as u32,
                        rate_bits as u32,
                        local_tile_log as u32,
                    ];
                    let tile_len = 1usize << local_tile_log;
                    let tile_count = fft_len >> local_tile_log;

                    encoder.set_compute_pipeline_state(&self.pipeline_fft_init_coset_lde_local);
                    encoder.set_buffer(0, Some(&coeffs_buffer), 0);
                    encoder.set_buffer(1, Some(&shift_powers_buffer), 0);
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                    encoder.set_buffer(3, Some(&flat_roots_buffer), 0);
                    encoder.set_bytes(
                        4,
                        (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                        uniforms.as_ptr() as *mut core::ffi::c_void,
                    );
                    encoder.set_threadgroup_memory_length(
                        0,
                        (tile_len * std::mem::size_of::<u64>()) as u64,
                    );
                    let threadgroups_per_grid = get_size_for_count(poly_count * tile_count);
                    let thread_group_size = MTLSize {
                        width: (tile_len >> 1) as u64,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatch_thread_groups(threadgroups_per_grid, thread_group_size);
                    local_tile_log
                } else {
                    let uniforms = [
                        poly_count as u32,
                        coeff_len as u32,
                        fft_len as u32,
                        log_fft_len as u32,
                        rate_bits as u32,
                    ];
                    encoder.set_compute_pipeline_state(&self.pipeline_fft_init_coset_lde);
                    encoder.set_buffer(0, Some(&coeffs_buffer), 0);
                    encoder.set_buffer(1, Some(&shift_powers_buffer), 0);
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                    encoder.set_bytes(
                        3,
                        (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                        uniforms.as_ptr() as *mut core::ffi::c_void,
                    );
                    let threads_per_grid = get_size_for_count(output.len());
                    let thread_group_size = MTLSize {
                        width: self.pipeline_fft_init_coset_lde.thread_execution_width(),
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatch_threads(threads_per_grid, thread_group_size);
                    rate_bits
                };

                if first_global_stage < log_fft_len {
                    encoder.memory_barrier_with_resources(&[output_buffer.as_ref()]);
                }

                let mut stage = first_global_stage;
                while stage < log_fft_len {
                    // Three-stage fusion is correct but was slightly slower in the prover
                    // benchmark, likely due to higher register pressure. Keep the kernel around
                    // for experiments, but use two-stage fusion in the default Metal LDE path.
                    let use_triple_fusion = false;
                    if use_triple_fusion && stage + 2 < log_fft_len {
                        let half_m = 1usize << stage;
                        let uniforms = [poly_count as u32, fft_len as u32, half_m as u32];

                        encoder.set_compute_pipeline_state(&self.pipeline_fft_stage_triple);
                        encoder.set_buffer(0, Some(&output_buffer), 0);
                        encoder.set_buffer(1, Some(&root_buffers[stage]), 0);
                        encoder.set_buffer(2, Some(&root_buffers[stage + 1]), 0);
                        encoder.set_buffer(3, Some(&root_buffers[stage + 2]), 0);
                        encoder.set_bytes(
                            4,
                            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                            uniforms.as_ptr() as *mut core::ffi::c_void,
                        );
                        let threads_per_grid = get_size_for_count(poly_count * (fft_len >> 3));
                        let thread_group_size = MTLSize {
                            width: self.pipeline_fft_stage_triple.thread_execution_width(),
                            height: 1,
                            depth: 1,
                        };
                        encoder.dispatch_threads(threads_per_grid, thread_group_size);
                        stage += 3;
                    } else if stage + 1 < log_fft_len {
                        let half_m = 1usize << stage;
                        let uniforms = [poly_count as u32, fft_len as u32, half_m as u32];

                        encoder.set_compute_pipeline_state(&self.pipeline_fft_stage_pair);
                        encoder.set_buffer(0, Some(&output_buffer), 0);
                        encoder.set_buffer(1, Some(&root_buffers[stage]), 0);
                        encoder.set_buffer(2, Some(&root_buffers[stage + 1]), 0);
                        encoder.set_bytes(
                            3,
                            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                            uniforms.as_ptr() as *mut core::ffi::c_void,
                        );
                        let threads_per_grid = get_size_for_count(poly_count * (fft_len >> 2));
                        let thread_group_size = MTLSize {
                            width: self.pipeline_fft_stage_pair.thread_execution_width(),
                            height: 1,
                            depth: 1,
                        };
                        encoder.dispatch_threads(threads_per_grid, thread_group_size);
                        stage += 2;
                    } else {
                        let half_m = 1usize << stage;
                        let uniforms = [
                            poly_count as u32,
                            fft_len as u32,
                            log_fft_len as u32,
                            rate_bits as u32,
                            half_m as u32,
                        ];

                        encoder.set_compute_pipeline_state(&self.pipeline_fft_stage);
                        encoder.set_buffer(0, Some(&output_buffer), 0);
                        encoder.set_buffer(1, Some(&root_buffers[stage]), 0);
                        encoder.set_bytes(
                            2,
                            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                            uniforms.as_ptr() as *mut core::ffi::c_void,
                        );
                        let threads_per_grid = get_size_for_count(poly_count * (fft_len >> 1));
                        let thread_group_size = MTLSize {
                            width: self.pipeline_fft_stage.thread_execution_width(),
                            height: 1,
                            depth: 1,
                        };
                        encoder.dispatch_threads(threads_per_grid, thread_group_size);
                        stage += 1;
                    }

                    if stage < log_fft_len {
                        encoder.memory_barrier_with_resources(&[output_buffer.as_ref()]);
                    }
                }
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        output
    }

    pub fn fft_values_poly_major(
        &self,
        values: &[GoldilocksField],
        poly_count: usize,
        fft_len: usize,
        zero_factor: usize,
    ) -> Vec<GoldilocksField> {
        let value_count = poly_count
            .checked_mul(fft_len)
            .expect("Metal FFT value count overflow");
        assert_eq!(values.len(), value_count);
        assert!(fft_len.is_power_of_two());
        let log_fft_len = fft_len.ilog2() as usize;
        assert!(log_fft_len <= 32);
        assert!(zero_factor <= log_fft_len);
        assert!(
            value_count <= u32::MAX as usize,
            "Metal FFT value count must fit in u32 shader indices"
        );

        let root_buffers = self.root_buffers_for_len(fft_len);
        let mut output = goldilocks_vec_for_gpu_overwrite(values.len());

        autoreleasepool::<(), _>(|| {
            let device = self.device.lock().unwrap();
            let input_buffer = device.new_buffer_with_bytes_no_copy(
                values.as_ptr() as *const core::ffi::c_void,
                (values.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let output_buffer = device.new_buffer_with_bytes_no_copy(
                output.as_mut_ptr() as *const core::ffi::c_void,
                (output.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            drop(device);

            let command_buffer = self.command_queue.new_command_buffer();
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            let uniforms = [
                poly_count as u32,
                fft_len as u32,
                log_fft_len as u32,
                zero_factor as u32,
                0u32,
            ];
            encoder.set_compute_pipeline_state(&self.pipeline_fft_init_reverse_zero);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                uniforms.as_ptr() as *mut core::ffi::c_void,
            );
            let threads_per_grid = get_size_for_count(values.len());
            let thread_group_size = MTLSize {
                width: self.pipeline_fft_init_reverse_zero.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(threads_per_grid, thread_group_size);

            if zero_factor < log_fft_len {
                encoder.memory_barrier_with_resources(&[output_buffer.as_ref()]);
            }

            for (stage, roots_buffer) in root_buffers
                .iter()
                .enumerate()
                .take(log_fft_len)
                .skip(zero_factor)
            {
                let half_m = 1usize << stage;
                let uniforms = [
                    poly_count as u32,
                    fft_len as u32,
                    log_fft_len as u32,
                    zero_factor as u32,
                    half_m as u32,
                ];

                encoder.set_compute_pipeline_state(&self.pipeline_fft_stage);
                encoder.set_buffer(0, Some(&output_buffer), 0);
                encoder.set_buffer(1, Some(roots_buffer), 0);
                encoder.set_bytes(
                    2,
                    (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                    uniforms.as_ptr() as *mut core::ffi::c_void,
                );
                let threads_per_grid = get_size_for_count(poly_count * (fft_len >> 1));
                let thread_group_size = MTLSize {
                    width: self.pipeline_fft_stage.thread_execution_width(),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(threads_per_grid, thread_group_size);
                if stage + 1 < log_fft_len {
                    encoder.memory_barrier_with_resources(&[output_buffer.as_ref()]);
                }
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        output
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quotient_perm_reduce(
        &self,
        wires_lde: &[GoldilocksField],
        constants_sigmas_lde: &[GoldilocksField],
        zpp_lde: &[GoldilocksField],
        shifted_xs: &[GoldilocksField],
        l0_values: &[GoldilocksField],
        k_is: &[GoldilocksField],
        betas: &[GoldilocksField],
        gammas: &[GoldilocksField],
        alphas: &[GoldilocksField],
        point_count: usize,
        leaf_count: usize,
        step: usize,
        next_step: usize,
        num_constants: usize,
        num_routed_wires: usize,
        num_challenges: usize,
        num_partial_products: usize,
        max_degree: usize,
    ) -> Vec<GoldilocksField> {
        assert_eq!(shifted_xs.len(), point_count);
        assert_eq!(l0_values.len(), point_count);
        assert_eq!(k_is.len(), num_routed_wires);
        assert_eq!(betas.len(), num_challenges);
        assert_eq!(gammas.len(), num_challenges);
        assert!(alphas.len() <= 64);
        assert!(point_count <= u32::MAX as usize);
        assert!(leaf_count <= u32::MAX as usize);
        assert!(step <= u32::MAX as usize);
        assert!(next_step <= u32::MAX as usize);

        let output_len = point_count
            .checked_mul(alphas.len())
            .expect("Metal quotient output length overflow");
        let mut output = goldilocks_vec_for_gpu_overwrite(output_len);

        autoreleasepool::<(), _>(|| {
            let device = self.device.lock().unwrap();
            let wires_buffer = device.new_buffer_with_bytes_no_copy(
                wires_lde.as_ptr() as *const core::ffi::c_void,
                (wires_lde.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let constants_sigmas_buffer = device.new_buffer_with_bytes_no_copy(
                constants_sigmas_lde.as_ptr() as *const core::ffi::c_void,
                (constants_sigmas_lde.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let zpp_buffer = device.new_buffer_with_bytes_no_copy(
                zpp_lde.as_ptr() as *const core::ffi::c_void,
                (zpp_lde.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let shifted_xs_buffer = device.new_buffer_with_bytes_no_copy(
                shifted_xs.as_ptr() as *const core::ffi::c_void,
                (shifted_xs.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let l0_buffer = device.new_buffer_with_bytes_no_copy(
                l0_values.as_ptr() as *const core::ffi::c_void,
                (l0_values.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let k_is_buffer = device.new_buffer_with_bytes_no_copy(
                k_is.as_ptr() as *const core::ffi::c_void,
                (k_is.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let betas_buffer = device.new_buffer_with_bytes_no_copy(
                betas.as_ptr() as *const core::ffi::c_void,
                (betas.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let gammas_buffer = device.new_buffer_with_bytes_no_copy(
                gammas.as_ptr() as *const core::ffi::c_void,
                (gammas.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let alphas_buffer = device.new_buffer_with_bytes_no_copy(
                alphas.as_ptr() as *const core::ffi::c_void,
                (alphas.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let output_buffer = device.new_buffer_with_bytes_no_copy(
                output.as_mut_ptr() as *const core::ffi::c_void,
                (output.len() * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            drop(device);

            let uniforms = [
                point_count as u32,
                leaf_count as u32,
                step as u32,
                next_step as u32,
                num_constants as u32,
                num_routed_wires as u32,
                num_challenges as u32,
                num_partial_products as u32,
                max_degree as u32,
                alphas.len() as u32,
            ];

            let command_buffer = self.command_queue.new_command_buffer();
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);
            encoder.set_compute_pipeline_state(&self.pipeline_quotient_perm_reduce);
            encoder.set_buffer(0, Some(&wires_buffer), 0);
            encoder.set_buffer(1, Some(&constants_sigmas_buffer), 0);
            encoder.set_buffer(2, Some(&zpp_buffer), 0);
            encoder.set_buffer(3, Some(&shifted_xs_buffer), 0);
            encoder.set_buffer(4, Some(&l0_buffer), 0);
            encoder.set_buffer(5, Some(&k_is_buffer), 0);
            encoder.set_buffer(6, Some(&betas_buffer), 0);
            encoder.set_buffer(7, Some(&gammas_buffer), 0);
            encoder.set_buffer(8, Some(&alphas_buffer), 0);
            encoder.set_buffer(9, Some(&output_buffer), 0);
            encoder.set_bytes(
                10,
                (uniforms.len() * std::mem::size_of::<u32>()) as u64,
                uniforms.as_ptr() as *mut core::ffi::c_void,
            );
            let threads_per_grid = get_size_for_count(point_count);
            let thread_group_size = MTLSize {
                width: self.pipeline_quotient_perm_reduce.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(threads_per_grid, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        output
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
        let input_len = leaf_count
            .checked_mul(leaf_length)
            .expect("Metal Merkle input length overflow");
        assert!(
            input_len <= u32::MAX as usize,
            "Metal Merkle input length must fit in u32 shader indices"
        );

        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;
        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        assert!(
            total_digests <= u32::MAX as usize,
            "Metal Merkle digest length must fit in u32 shader indices"
        );

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
        let input_len = leaf_count
            .checked_mul(leaf_length)
            .expect("Metal poly-major Merkle input length overflow");
        assert_eq!(poly_values.len(), input_len);
        assert!(
            input_len <= u32::MAX as usize,
            "Metal poly-major Merkle input length must fit in u32 shader indices"
        );
        let tree_height = leaf_count.ilog2() as usize;

        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;
        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        assert!(
            total_digests <= u32::MAX as usize,
            "Metal poly-major Merkle digest length must fit in u32 shader indices"
        );

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

    fn leaf_hash_pipeline(&self, poly_major: bool, leaf_length: usize) -> &ComputePipelineState {
        if !poly_major {
            return &self.pipeline_poseidon_hash_leaves;
        }

        match leaf_length {
            16 => &self.pipeline_poseidon_hash_leaves_poly_major_16,
            _ => &self.pipeline_poseidon_hash_leaves_poly_major,
        }
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
                leaf_length,
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

        let leaf_pipeline = self.leaf_hash_pipeline(poly_major, leaf_length);
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
        let thread_group_size = leaf_threadgroup_size(leaf_pipeline);
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
        leaf_length: usize,
        leaf_count: usize,
        num_layers: usize,
        poly_major: bool,
    ) -> (Vec<GoldilocksDigest>, MetalHashProfile) {
        let leaf_encode_start = Instant::now();
        let leaf_command_buffer = self.command_queue.new_command_buffer();
        let leaf_compute_pass_descriptor = ComputePassDescriptor::new();
        let leaf_encoder = leaf_command_buffer
            .compute_command_encoder_with_descriptor(leaf_compute_pass_descriptor);

        let leaf_pipeline = self.leaf_hash_pipeline(poly_major, leaf_length);
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
        let thread_group_size = leaf_threadgroup_size(leaf_pipeline);
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
    use plonky2_field::polynomial::PolynomialCoeffs;

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

        for leaf_length in [16, 20, 33, 84, 135] {
            let leaf_count = 1usize << 9;
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

            assert_eq!(poly_major, row_major, "leaf_length={leaf_length}");
        }
    }

    #[test]
    fn fft_poly_major_matches_cpu_when_available() {
        if !is_available() {
            eprintln!("skipping Metal FFT test: no Metal device available");
            return;
        }

        let poly_count = 3;
        let fft_len = 64;
        let zero_factor = 2;
        let live_coeff_len = fft_len >> zero_factor;
        let mut values = vec![GoldilocksField::ZERO; poly_count * fft_len];
        for poly in 0..poly_count {
            for i in 0..live_coeff_len {
                values[poly * fft_len + i] =
                    GoldilocksField::from_canonical_u64((poly * 97 + i * 13 + 5) as u64);
            }
        }

        let metal =
            MetalRuntime::get().fft_values_poly_major(&values, poly_count, fft_len, zero_factor);
        for poly in 0..poly_count {
            let start = poly * fft_len;
            let cpu = PolynomialCoeffs::new(values[start..start + fft_len].to_vec())
                .fft_with_options(Some(zero_factor), None)
                .values;
            assert_eq!(&metal[start..start + fft_len], cpu.as_slice());
        }
    }

    #[test]
    fn coset_lde_poly_major_matches_cpu_when_available() {
        if !is_available() {
            eprintln!("skipping Metal coset LDE test: no Metal device available");
            return;
        }

        let poly_count = 2;
        let coeff_len = 32;
        let rate_bits = 3;
        let fft_len = coeff_len << rate_bits;
        let coeffs = (0..poly_count)
            .flat_map(|poly| {
                (0..coeff_len).map(move |i| {
                    GoldilocksField::from_canonical_u64((poly * 131 + i * 17 + 11) as u64)
                })
            })
            .collect::<Vec<_>>();

        let metal = MetalRuntime::get()
            .coset_lde_values_poly_major(&coeffs, poly_count, coeff_len, rate_bits);
        for poly in 0..poly_count {
            let coeff_start = poly * coeff_len;
            let eval_start = poly * fft_len;
            let cpu = PolynomialCoeffs::new(coeffs[coeff_start..coeff_start + coeff_len].to_vec())
                .lde(rate_bits)
                .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
                .values;
            assert_eq!(&metal[eval_start..eval_start + fft_len], cpu.as_slice());
        }
    }
}
