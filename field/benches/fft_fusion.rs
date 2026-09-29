//! Fused FFT kernel vs the original layer-by-layer kernel (`fft_reference`).
//!
//!   cargo bench -p plonky2_field --features fft-reference --bench fft_fusion
//!   cargo bench -p plonky2_field --features fft-reference,fft-prefetch --bench fft_fusion
//!
//! The "fused" rows are labelled "fused+prefetch" when the `fft-prefetch` feature is on; the
//! "old" rows are the same code in both builds and serve as the control.
//! `FFT_BENCH_THREADS` sets the rayon pool size of the LDE scenario (default 24).

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use plonky2_field::fft::{fft_root_table, fft_with_options};
use plonky2_field::fft_reference;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::Sample;
use rayon::prelude::*;

type F = GoldilocksField;

const FUSED: &str = if cfg!(feature = "fft-prefetch") {
    "fused+prefetch"
} else {
    "fused"
};

fn bench_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_single");
    group.sample_size(20);
    for lg_n in [16, 18, 20, 22] {
        let n = 1usize << lg_n;
        let table = fft_root_table::<F>(n);
        let input = F::rand_vec(n);
        group.bench_with_input(BenchmarkId::new("old", lg_n), &lg_n, |b, _| {
            b.iter_batched(
                || PolynomialCoeffs::new(input.clone()),
                |p| fft_reference::fft_with_options(p, None, Some(&table)),
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new(FUSED, lg_n), &lg_n, |b, _| {
            b.iter_batched(
                || PolynomialCoeffs::new(input.clone()),
                |p| fft_with_options(p, None, Some(&table)),
                BatchSize::LargeInput,
            )
        });
    }
}

/// LDE-like load: `POLYS` polynomials of degree 2^17 blown up by 2^3 to 2^20 points
/// (`zero_factor = 3`), one polynomial per rayon task, as in `PolynomialBatch::lde_values`.
fn bench_lde(c: &mut Criterion) {
    const POLYS: usize = 64;
    const LG_N: usize = 20;
    const RATE_BITS: usize = 3;
    let threads = std::env::var("FFT_BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let n = 1usize << LG_N;
    let table = fft_root_table::<F>(n);
    let polys: Vec<PolynomialCoeffs<F>> = (0..POLYS)
        .map(|_| PolynomialCoeffs::new(F::rand_vec(n >> RATE_BITS)).lde(RATE_BITS))
        .collect();

    let mut group = c.benchmark_group(format!("lde_{POLYS}x2^{LG_N}_t{threads}"));
    group.sample_size(10);
    group.bench_function("old", |b| {
        b.iter_batched(
            || polys.clone(),
            |ps| {
                pool.install(|| {
                    ps.into_par_iter()
                        .map(|p| fft_reference::fft_with_options(p, Some(RATE_BITS), Some(&table)))
                        .collect::<Vec<_>>()
                })
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function(FUSED, |b| {
        b.iter_batched(
            || polys.clone(),
            |ps| {
                pool.install(|| {
                    ps.into_par_iter()
                        .map(|p| fft_with_options(p, Some(RATE_BITS), Some(&table)))
                        .collect::<Vec<_>>()
                })
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_single, bench_lde);
criterion_main!(benches);
