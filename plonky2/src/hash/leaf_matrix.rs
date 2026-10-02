//! Storage for Merkle tree leaves.
//!
//! `LeafMatrix` holds the leaves either as one contiguous row-major buffer (every row the same
//! width) or in the original nested form. Both index as row slices (`leaves[i]: [F]`), report the
//! row count with `len()` and iterate rows, so existing readers keep working. Equality compares
//! rows, so the two forms of the same leaves are equal. The serialized form (speedy) is the nested
//! `Vec<Vec<F>>` encoding regardless of the in-memory form, so stored data is unchanged.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
use core::ops::Index;

/// Read-only contiguous leaves owned by something else (e.g. a pinned buffer leased from a
/// pool); `owner` keeps the memory alive and is dropped (returning the buffer) with the matrix.
pub struct ExternalLeaves<F> {
    data: *const F,
    len: usize,
    _owner: Box<dyn core::any::Any + Send + Sync>,
}

// SAFETY: the data is never written after construction and lives as long as `_owner`.
unsafe impl<F: Sync> Send for ExternalLeaves<F> {}
unsafe impl<F: Sync> Sync for ExternalLeaves<F> {}

impl<F> core::fmt::Debug for ExternalLeaves<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ExternalLeaves {{ len: {} }}", self.len)
    }
}

impl<F> ExternalLeaves<F> {
    /// # Safety
    /// `data` must point to `len` initialised elements that stay valid and unmodified for as long
    /// as `owner` is alive.
    pub unsafe fn new(data: *const F, len: usize, owner: Box<dyn core::any::Any + Send + Sync>) -> Self {
        Self { data, len, _owner: owner }
    }

    fn as_slice(&self) -> &[F] {
        // SAFETY: see `new`.
        unsafe { core::slice::from_raw_parts(self.data, self.len) }
    }
}

#[derive(Debug)]
pub enum LeafMatrix<F> {
    /// `rows * width` elements, row `i` at `data[i * width..(i + 1) * width]`; `width > 0`.
    Flat { data: Vec<F>, width: usize },
    Nested(Vec<Vec<F>>),
    /// Contiguous like `Flat`, but the storage belongs to `leaves`' owner.
    External { leaves: ExternalLeaves<F>, width: usize },
}

impl<F: Clone> Clone for LeafMatrix<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Flat { data, width } => Self::Flat { data: data.clone(), width: *width },
            Self::Nested(v) => Self::Nested(v.clone()),
            // A clone never shares the external owner: copy into a flat matrix.
            Self::External { leaves, width } => Self::Flat { data: leaves.as_slice().to_vec(), width: *width },
        }
    }
}

impl<F> Default for LeafMatrix<F> {
    fn default() -> Self {
        Self::Nested(Vec::new())
    }
}

impl<F> LeafMatrix<F> {
    /// Contiguous leaves of `width` elements each.
    pub fn from_flat(data: Vec<F>, width: usize) -> Self {
        assert!(width > 0, "flat leaves need a non-zero width");
        assert_eq!(data.len() % width, 0, "flat leaves length must be a multiple of the width");
        Self::Flat { data, width }
    }

    /// Contiguous leaves of `width` elements each in external storage.
    pub fn from_external(leaves: ExternalLeaves<F>, width: usize) -> Self {
        assert!(width > 0, "external leaves need a non-zero width");
        assert_eq!(leaves.len % width, 0, "external leaves length must be a multiple of the width");
        Self::External { leaves, width }
    }

    /// Number of leaves (rows).
    pub fn len(&self) -> usize {
        match self {
            Self::Flat { data, width } => data.len() / width,
            Self::Nested(v) => v.len(),
            Self::External { leaves, width } => leaves.len / width,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<&[F]> {
        match self {
            Self::Flat { data, width } => data.get(i * width..(i + 1) * width),
            Self::Nested(v) => v.get(i).map(|r| r.as_slice()),
            Self::External { leaves, width } => leaves.as_slice().get(i * width..(i + 1) * width),
        }
    }

    pub fn iter(&self) -> LeafRows<'_, F> {
        LeafRows { m: self, next: 0 }
    }

    pub const fn is_flat(&self) -> bool {
        matches!(self, Self::Flat { .. })
    }

    pub const fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }

    /// The nested form (copies when the storage is flat).
    pub fn to_nested(&self) -> Vec<Vec<F>>
    where
        F: Clone,
    {
        self.iter().map(|r| r.to_vec()).collect()
    }
}

impl<F> From<Vec<Vec<F>>> for LeafMatrix<F> {
    fn from(v: Vec<Vec<F>>) -> Self {
        Self::Nested(v)
    }
}

impl<F> Index<usize> for LeafMatrix<F> {
    type Output = [F];

    fn index(&self, i: usize) -> &[F] {
        match self {
            Self::Flat { data, width } => &data[i * width..(i + 1) * width],
            Self::Nested(v) => &v[i],
            Self::External { leaves, width } => &leaves.as_slice()[i * width..(i + 1) * width],
        }
    }
}

#[derive(Debug)]
pub struct LeafRows<'a, F> {
    m: &'a LeafMatrix<F>,
    next: usize,
}

impl<'a, F> Iterator for LeafRows<'a, F> {
    type Item = &'a [F];

    fn next(&mut self) -> Option<&'a [F]> {
        let r = self.m.get(self.next)?;
        self.next += 1;
        Some(r)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.m.len() - self.next;
        (n, Some(n))
    }
}

impl<F> ExactSizeIterator for LeafRows<'_, F> {}

impl<'a, F> IntoIterator for &'a LeafMatrix<F> {
    type Item = &'a [F];
    type IntoIter = LeafRows<'a, F>;

    fn into_iter(self) -> LeafRows<'a, F> {
        self.iter()
    }
}

impl<F: PartialEq> PartialEq for LeafMatrix<F> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<F: Eq> Eq for LeafMatrix<F> {}

impl<F: PartialEq> PartialEq<Vec<Vec<F>>> for LeafMatrix<F> {
    fn eq(&self, other: &Vec<Vec<F>>) -> bool {
        self == &other[..]
    }
}

impl<F: PartialEq> PartialEq<[Vec<F>]> for LeafMatrix<F> {
    fn eq(&self, other: &[Vec<F>]) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b.as_slice())
    }
}

impl<F: PartialEq> PartialEq<&[Vec<F>]> for LeafMatrix<F> {
    fn eq(&self, other: &&[Vec<F>]) -> bool {
        self == *other
    }
}

#[cfg(feature = "serialize_speedy")]
impl<'a, C: speedy::Context, F: speedy::Readable<'a, C>> speedy::Readable<'a, C> for LeafMatrix<F> {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(Self::Nested(Vec::<Vec<F>>::read_from(reader)?))
    }
}

#[cfg(feature = "serialize_speedy")]
impl<C: speedy::Context, F: speedy::Writable<C> + Clone> speedy::Writable<C> for LeafMatrix<F> {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        match self {
            Self::Nested(v) => v.write_to(writer),
            Self::Flat { .. } | Self::External { .. } => self.to_nested().write_to(writer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_and_nested_agree() {
        let nested: Vec<Vec<u64>> = (0..8).map(|i| (0..5).map(|j| i * 10 + j).collect()).collect();
        let flat = LeafMatrix::from_flat(nested.concat(), 5);
        let n = LeafMatrix::from(nested.clone());
        assert_eq!(flat.len(), 8);
        assert_eq!(&flat[3], &nested[3][..]);
        assert!(flat == n && n == flat && flat == nested);
        assert_eq!(flat.iter().map(|r| r.len()).sum::<usize>(), 40);
        assert_eq!(flat.to_nested(), nested);
        assert_eq!(flat.get(8), None);
    }

    #[test]
    fn external_leaves_behave_like_flat_and_drop_their_owner() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        struct Owner(Vec<u64>, Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.1.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let nested: Vec<Vec<u64>> = (0..6).map(|i| vec![i, i + 1, i + 2]).collect();
        let owner = Box::new(Owner(nested.concat(), dropped.clone()));
        let ptr = owner.0.as_ptr();
        let m = LeafMatrix::from_external(unsafe { ExternalLeaves::new(ptr, 18, owner) }, 3);
        assert!(m == nested && m.len() == 6 && &m[2] == &[2, 3, 4][..]);
        let c = m.clone();
        assert!(!c.is_external() && c == m);
        drop(m);
        assert_eq!(dropped.load(Ordering::SeqCst), 1, "owner dropped with the matrix");
        assert!(c == nested, "clone owns its data");
    }

    #[cfg(feature = "serialize_speedy")]
    #[test]
    fn speedy_encoding_is_the_nested_encoding() {
        use speedy::{Readable, Writable};
        let nested: Vec<Vec<u64>> = (0..4).map(|i| vec![i; 3]).collect();
        let want = nested.write_to_vec().unwrap();
        assert_eq!(LeafMatrix::from_flat(nested.concat(), 3).write_to_vec().unwrap(), want);
        assert_eq!(LeafMatrix::from(nested.clone()).write_to_vec().unwrap(), want);
        let back = LeafMatrix::<u64>::read_from_buffer(&want).unwrap();
        assert!(back == nested);
    }
}
