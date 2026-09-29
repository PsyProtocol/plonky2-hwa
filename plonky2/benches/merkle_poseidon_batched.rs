//! Compares one-at-a-time and batched (SIMD) Poseidon for Merkle tree construction.
//!
//! "scalar" uses a hasher that only differs from `PoseidonHash` in keeping the default
//! `BATCH_WIDTH = 1`, so `MerkleTree::new` takes the original path. "batched" is `PoseidonHash`,
//! which batches `<GoldilocksField as Packable>::Packing::WIDTH` permutations per vector: 8 with
//! AVX-512 target features, 4 with AVX2, 1 (no batching) otherwise. Build with e.g.
//! `RUSTFLAGS="-C target-cpu=native"` to enable them.
//!
//! Set `RAYON_NUM_THREADS` to limit the thread pool. Run with
//! `cargo bench -p plonky2 --bench merkle_poseidon_batched`.

mod allocator;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Sample;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::{Poseidon, PoseidonHash, PoseidonPermutation, SPONGE_WIDTH};
use plonky2::plonk::config::Hasher;

type F = GoldilocksField;
type Packed = <F as Packable>::Packing;

/// `PoseidonHash` without batching: `MerkleTree::new` hashes one leaf / node at a time.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct ScalarPoseidonHash;

impl Hasher<F> for ScalarPoseidonHash {
    const HASH_SIZE: usize = <PoseidonHash as Hasher<F>>::HASH_SIZE;
    type Hash = HashOut<F>;
    type Permutation = PoseidonPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        PoseidonHash::hash_no_pad(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        PoseidonHash::two_to_one(left, right)
    }
}

const CAP_HEIGHT: usize = 4;

fn bench_merkle<H: Hasher<F>>(c: &mut Criterion, name: &str, log_leaves: usize, leaf_len: usize) {
    let mut group = c.benchmark_group(format!("merkle-poseidon/{leaf_len}-elem-leaves"));
    group.sample_size(10);
    let n = 1usize << log_leaves;
    group.throughput(Throughput::Elements(n as u64));
    let leaves: Vec<Vec<F>> = (0..n).map(|_| F::rand_vec(leaf_len)).collect();
    group.bench_function(BenchmarkId::new(name, format!("2^{log_leaves}")), |b| {
        b.iter_batched(
            || leaves.clone(),
            |leaves| MerkleTree::<F, H>::new(leaves, CAP_HEIGHT),
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn bench_permutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon-permutation-single-thread");
    group.throughput(Throughput::Elements(Packed::WIDTH as u64));
    let states: Vec<[F; SPONGE_WIDTH]> = (0..Packed::WIDTH)
        .map(|_| F::rand_array::<SPONGE_WIDTH>())
        .collect();
    group.bench_function(format!("scalar x{}", Packed::WIDTH), |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| F::poseidon(core::hint::black_box(*s)))
                .collect::<Vec<_>>()
        })
    });
    let mut packed = [Packed::ZEROS; SPONGE_WIDTH];
    for (lane, s) in states.iter().enumerate() {
        for i in 0..SPONGE_WIDTH {
            packed[i].as_slice_mut()[lane] = s[i];
        }
    }
    group.bench_function(format!("packed width {}", Packed::WIDTH), |b| {
        b.iter(|| F::poseidon_packed(core::hint::black_box(packed)))
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    println!(
        "packing width {}, RAYON_NUM_THREADS={}",
        Packed::WIDTH,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset (all cores)".into())
    );

    bench_permutation(c);
    for (log_leaves, leaf_len) in [(18, 135), (20, 12)] {
        bench_merkle::<ScalarPoseidonHash>(c, "scalar", log_leaves, leaf_len);
        bench_merkle::<PoseidonHash>(c, "batched", log_leaves, leaf_len);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
