#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
use crate::hash::leaf_matrix::LeafMatrix;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
#[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
#[repr(transparent)]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
#[repr(C)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree (row `i` is `leaves[i]`).
    pub leaves: LeafMatrix<F>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: LeafMatrix::default(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

/// Subtrees with at most this many leaves are filled serially by `fill_subtree_batched` when the
/// hasher batches (`H::BATCH_WIDTH > 1`). Larger subtrees are split in parallel as before.
const BATCH_SUBTREE_MAX_LEAVES: usize = 256;

/// Number of leaves hashed per task when the whole tree is cap (`digests_buf` is empty).
const BATCH_CAP_CHUNK: usize = 256;

/// Index, within the digest buffer of a subtree, of the digest of node `index` on `layer`
/// (layer 0 holds the leaf digests). This is the layout `fill_subtree` produces and
/// `merkle_tree_prove` reads: siblings are adjacent, and the pair containing the node sits at
/// pair index `(index >> 1) * 2^(layer + 1) + 2^layer - 1`.
#[inline]
const fn subtree_digest_index(layer: usize, index: usize) -> usize {
    2 * (((index >> 1) << (layer + 1)) + (1 << layer) - 1) + (index & 1)
}

/// Same result and layout as the recursive `fill_subtree`, computed one layer at a time so that
/// every layer is hashed with `H::hash_or_noop_batch` / `H::two_to_one_batch`.
fn fill_subtree_batched<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    debug_assert!(leaves.len().is_power_of_two());

    // Every slot of `digests_buf` is written exactly once: layer `k` has `n / 2^k` nodes, and
    // over layers 0..log2(n) they fill all `2n - 2` slots. The length checks keep a misbehaving
    // batch implementation from leaving slots uninitialized.
    let mut layer_digests = H::hash_or_noop_batch(leaves);
    assert_eq!(layer_digests.len(), leaves.len());
    let mut layer = 0;
    while layer_digests.len() > 1 {
        for (index, &digest) in layer_digests.iter().enumerate() {
            digests_buf[subtree_digest_index(layer, index)].write(digest);
        }
        let next = H::two_to_one_batch(&layer_digests);
        assert_eq!(next.len(), layer_digests.len() / 2);
        layer_digests = next;
        layer += 1;
    }
    layer_digests[0]
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if H::BATCH_WIDTH > 1 && leaves.len() <= BATCH_SUBTREE_MAX_LEAVES {
        fill_subtree_batched::<F, H>(digests_buf, leaves)
    } else if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        if H::BATCH_WIDTH > 1 {
            cap_buf
                .par_chunks_mut(BATCH_CAP_CHUNK)
                .zip(leaves.par_chunks(BATCH_CAP_CHUNK))
                .for_each(|(cap_chunk, leaves_chunk)| {
                    let digests = H::hash_or_noop_batch(leaves_chunk);
                    assert_eq!(digests.len(), cap_chunk.len());
                    for (cap_buf, digest) in cap_chunk.iter_mut().zip(digests) {
                        cap_buf.write(digest);
                    }
                });
        } else {
            cap_buf
                .par_iter_mut()
                .zip(leaves)
                .for_each(|(cap_buf, leaf)| {
                    cap_buf.write(H::hash_or_noop(leaf));
                });
        }
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves: LeafMatrix::from(leaves),
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::hash_types::HashOut;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::poseidon::{
        hash_or_noop_batch_packed, two_to_one_batch_packed, PoseidonHash, PoseidonPermutation,
    };
    use crate::hash::poseidon_goldilocks::packed_tests::{random_raw_vec, ArrayPacked};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    /// Verbatim copy of the one-at-a-time construction before batching was introduced. It is
    /// the reference that the batched construction must reproduce bit for bit.
    mod reference {
        use super::*;

        fn fill_subtree<F: RichField, H: Hasher<F>>(
            digests_buf: &mut [MaybeUninit<H::Hash>],
            leaves: &[Vec<F>],
        ) -> H::Hash {
            assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
            if digests_buf.is_empty() {
                H::hash_or_noop(&leaves[0])
            } else {
                let (left_digests_buf, right_digests_buf) =
                    digests_buf.split_at_mut(digests_buf.len() / 2);
                let (left_digest_mem, left_digests_buf) =
                    left_digests_buf.split_last_mut().unwrap();
                let (right_digest_mem, right_digests_buf) =
                    right_digests_buf.split_first_mut().unwrap();
                let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

                let (left_digest, right_digest) = plonky2_maybe_rayon::join(
                    || fill_subtree::<F, H>(left_digests_buf, left_leaves),
                    || fill_subtree::<F, H>(right_digests_buf, right_leaves),
                );

                left_digest_mem.write(left_digest);
                right_digest_mem.write(right_digest);
                H::two_to_one(left_digest, right_digest)
            }
        }

        fn fill_digests_buf<F: RichField, H: Hasher<F>>(
            digests_buf: &mut [MaybeUninit<H::Hash>],
            cap_buf: &mut [MaybeUninit<H::Hash>],
            leaves: &[Vec<F>],
            cap_height: usize,
        ) {
            if digests_buf.is_empty() {
                debug_assert_eq!(cap_buf.len(), leaves.len());
                cap_buf
                    .par_iter_mut()
                    .zip(leaves)
                    .for_each(|(cap_buf, leaf)| {
                        cap_buf.write(H::hash_or_noop(leaf));
                    });
                return;
            }

            let subtree_digests_len = digests_buf.len() >> cap_height;
            let subtree_leaves_len = leaves.len() >> cap_height;
            let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
            let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
            assert_eq!(digests_chunks.len(), cap_buf.len());
            assert_eq!(digests_chunks.len(), leaves_chunks.len());
            digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
                |((subtree_digests, subtree_cap), subtree_leaves)| {
                    subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
                },
            );
        }

        /// Returns `(digests, cap)` as the original `MerkleTree::new` computed them.
        pub(crate) fn merkle_tree<F: RichField, H: Hasher<F>>(
            leaves: &[Vec<F>],
            cap_height: usize,
        ) -> (Vec<H::Hash>, Vec<H::Hash>) {
            let num_digests = 2 * (leaves.len() - (1 << cap_height));
            let mut digests = Vec::with_capacity(num_digests);
            let len_cap = 1 << cap_height;
            let mut cap = Vec::with_capacity(len_cap);
            let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
            let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
            fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height);
            unsafe {
                digests.set_len(num_digests);
                cap.set_len(len_cap);
            }
            (digests, cap)
        }
    }

    /// Poseidon with the batched Merkle path forced on and the width-1 packed permutation, so
    /// the batched layout and sponge code run even without SIMD target features.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct ForcedBatchPoseidon;

    impl Hasher<GoldilocksField> for ForcedBatchPoseidon {
        const HASH_SIZE: usize = <PoseidonHash as Hasher<GoldilocksField>>::HASH_SIZE;
        type Hash = HashOut<GoldilocksField>;
        type Permutation = PoseidonPermutation<GoldilocksField>;

        fn hash_no_pad(input: &[GoldilocksField]) -> Self::Hash {
            PoseidonHash::hash_no_pad(input)
        }

        fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
            PoseidonHash::two_to_one(left, right)
        }

        const BATCH_WIDTH: usize = 2;

        fn hash_or_noop_batch(inputs: &[Vec<GoldilocksField>]) -> Vec<Self::Hash> {
            hash_or_noop_batch_packed::<GoldilocksField, GoldilocksField>(inputs)
        }

        fn two_to_one_batch(inputs: &[Self::Hash]) -> Vec<Self::Hash> {
            two_to_one_batch_packed::<GoldilocksField, GoldilocksField>(inputs)
        }
    }

    /// Poseidon with the batched Merkle path at width 8 (the AVX-512 width), using a portable
    /// packed type so it runs on any machine.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct Width8BatchPoseidon;

    impl Hasher<GoldilocksField> for Width8BatchPoseidon {
        const HASH_SIZE: usize = <PoseidonHash as Hasher<GoldilocksField>>::HASH_SIZE;
        type Hash = HashOut<GoldilocksField>;
        type Permutation = PoseidonPermutation<GoldilocksField>;

        fn hash_no_pad(input: &[GoldilocksField]) -> Self::Hash {
            PoseidonHash::hash_no_pad(input)
        }

        fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
            PoseidonHash::two_to_one(left, right)
        }

        const BATCH_WIDTH: usize = 8;

        fn hash_or_noop_batch(inputs: &[Vec<GoldilocksField>]) -> Vec<Self::Hash> {
            hash_or_noop_batch_packed::<GoldilocksField, ArrayPacked<8>>(inputs)
        }

        fn two_to_one_batch(inputs: &[Self::Hash]) -> Vec<Self::Hash> {
            two_to_one_batch_packed::<GoldilocksField, ArrayPacked<8>>(inputs)
        }
    }

    fn raw_digests(digests: &[HashOut<GoldilocksField>]) -> Vec<[u64; 4]> {
        digests.iter().map(|h| h.elements.map(|x| x.0)).collect()
    }

    fn check_against_reference<H: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>>(
        leaves: &[Vec<GoldilocksField>],
        cap_height: usize,
        expected: &(Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>),
    ) {
        let tree = MerkleTree::<GoldilocksField, H>::new(leaves.to_vec(), cap_height);
        let what = format!(
            "{} leaves of {} elements, cap height {cap_height}, hasher {:?}",
            leaves.len(),
            leaves[0].len(),
            core::any::type_name::<H>()
        );
        assert_eq!(
            raw_digests(&tree.digests),
            raw_digests(&expected.0),
            "digests: {what}"
        );
        assert_eq!(
            raw_digests(&tree.cap.0),
            raw_digests(&expected.1),
            "cap: {what}"
        );
        assert_eq!(tree.leaves, leaves, "leaves: {what}");
    }

    #[test]
    fn batched_construction_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_0101);
        for log_n in 0..=12 {
            for leaf_len in [1, 4, 5, 8, 12, 135] {
                let leaves: Vec<Vec<GoldilocksField>> = (0..1 << log_n)
                    .map(|_| random_raw_vec(&mut rng, leaf_len))
                    .collect();
                for cap_height in 0..=log_n {
                    let expected = reference::merkle_tree::<GoldilocksField, PoseidonHash>(
                        &leaves, cap_height,
                    );
                    check_against_reference::<PoseidonHash>(&leaves, cap_height, &expected);
                    check_against_reference::<ForcedBatchPoseidon>(&leaves, cap_height, &expected);
                    check_against_reference::<Width8BatchPoseidon>(&leaves, cap_height, &expected);
                }
            }
        }
    }

    #[test]
    fn subtree_digest_index_matches_prove_layout() {
        // Every slot of a subtree's digest buffer is addressed exactly once.
        for log_n in 0..=10 {
            let n = 1usize << log_n;
            let mut seen = vec![false; 2 * n - 2];
            for layer in 0..log_n {
                for index in 0..(n >> layer) {
                    let i = subtree_digest_index(layer, index);
                    assert!(!seen[i], "slot {i} written twice (n = {n})");
                    seen[i] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "unwritten slot (n = {n})");
        }
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
