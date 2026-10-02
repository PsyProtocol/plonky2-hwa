//! Bounded pool of pinned host buffers that GPU commitments lease as Merkle leaf storage.
//!
//! A lease is taken before the device work, the leaves are downloaded straight into it, and the
//! resulting `MerkleTree` owns it (`LeafMatrix::External`); dropping the tree returns the buffer.
//! Leasing never blocks: when the budget is exhausted `try_lease` returns `None` and the caller
//! copies the leaves into ordinary memory instead, so correctness and liveness do not depend on
//! the budget. Buffers are kept per size class and reused; idle buffers of other classes are freed
//! when a new class does not fit the budget.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const MIB: usize = 1 << 20;
/// Hard cap on the configurable budget.
pub(crate) const MAX_BUDGET: usize = 32 << 30;
pub(crate) const DEFAULT_BUDGET: usize = 4 << 30;

/// Size class: 1 MiB granules below 64 MiB, 16 MiB granules above.
pub(crate) fn class_of(bytes: usize) -> usize {
    let g = if bytes < 64 * MIB { MIB } else { 16 * MIB };
    bytes.max(1).div_ceil(g) * g
}

/// Host allocation backend (pinned CUDA memory in production, heap memory in tests).
pub(crate) trait HostAlloc: Send + Sync + 'static {
    fn alloc(&self, bytes: usize) -> Option<*mut u8>;
    /// # Safety
    /// `ptr` must come from `alloc(bytes)` on this allocator and not be used afterwards.
    unsafe fn free(&self, ptr: *mut u8, bytes: usize);
}

struct Buf {
    ptr: *mut u8,
    class: usize,
}

// SAFETY: a `Buf` is owned by exactly one place at a time (pool or lease).
unsafe impl Send for Buf {}

struct Inner {
    free: BTreeMap<usize, Vec<Buf>>,
    /// Bytes of all buffers this pool holds (free + leased).
    allocated: usize,
}

pub(crate) struct LeafPool {
    budget: usize,
    alloc: Box<dyn HostAlloc>,
    inner: Mutex<Inner>,
    pub(crate) granted: AtomicU64,
    pub(crate) refused: AtomicU64,
    pub(crate) leased_bytes: AtomicU64,
    pub(crate) leased_peak: AtomicU64,
    pub(crate) allocated_peak: AtomicU64,
    pub(crate) allocs: AtomicU64,
    pub(crate) frees: AtomicU64,
}

/// An exclusive lease on one buffer; returns it to the pool when dropped.
pub(crate) struct Lease {
    buf: Option<Buf>,
    pool: Arc<LeafPool>,
}

impl Lease {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_ref().expect("live lease").ptr
    }
}

// SAFETY: see `Buf`.
unsafe impl Send for Lease {}
unsafe impl Sync for Lease {}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.leased_bytes.fetch_sub(buf.class as u64, Ordering::Relaxed);
            let mut inner = self.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.free.entry(buf.class).or_default().push(buf);
        }
    }
}

impl LeafPool {
    pub(crate) fn new(budget: usize, alloc: Box<dyn HostAlloc>) -> Arc<Self> {
        Arc::new(Self {
            budget: budget.min(MAX_BUDGET),
            alloc,
            inner: Mutex::new(Inner { free: BTreeMap::new(), allocated: 0 }),
            granted: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            leased_bytes: AtomicU64::new(0),
            leased_peak: AtomicU64::new(0),
            allocated_peak: AtomicU64::new(0),
            allocs: AtomicU64::new(0),
            frees: AtomicU64::new(0),
        })
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// Bytes of all buffers the pool holds (leased + idle).
    pub(crate) fn allocated(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).allocated
    }

    /// Non-blocking: a buffer of at least `bytes`, or `None` if the budget cannot fit one.
    pub(crate) fn try_lease(self: &Arc<Self>, bytes: usize) -> Option<Lease> {
        let class = class_of(bytes);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let buf = match inner.free.get_mut(&class).and_then(Vec::pop) {
            Some(buf) => buf,
            None => {
                if class > self.budget {
                    drop(inner);
                    self.refused.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                // Make room by freeing idle buffers of other classes, largest first.
                while inner.allocated + class > self.budget {
                    let Some((&c, _)) = inner.free.iter().rev().find(|(_, v)| !v.is_empty()) else { break };
                    let victim = inner.free.get_mut(&c).and_then(Vec::pop).expect("non-empty class");
                    // SAFETY: the buffer came from this allocator and is idle.
                    unsafe { self.alloc.free(victim.ptr, victim.class) };
                    inner.allocated -= victim.class;
                    self.frees.fetch_add(1, Ordering::Relaxed);
                }
                if inner.allocated + class > self.budget {
                    drop(inner);
                    self.refused.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                let Some(ptr) = self.alloc.alloc(class) else {
                    drop(inner);
                    self.refused.fetch_add(1, Ordering::Relaxed);
                    return None;
                };
                inner.allocated += class;
                self.allocs.fetch_add(1, Ordering::Relaxed);
                self.allocated_peak.fetch_max(inner.allocated as u64, Ordering::Relaxed);
                Buf { ptr, class }
            }
        };
        drop(inner);
        self.granted.fetch_add(1, Ordering::Relaxed);
        let now = self.leased_bytes.fetch_add(buf.class as u64, Ordering::Relaxed) + buf.class as u64;
        self.leased_peak.fetch_max(now, Ordering::Relaxed);
        Some(Lease { buf: Some(buf), pool: self.clone() })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// Heap allocator that counts live bytes.
    struct Heap(Arc<AtomicUsize>);

    impl HostAlloc for Heap {
        fn alloc(&self, bytes: usize) -> Option<*mut u8> {
            self.0.fetch_add(bytes, Ordering::SeqCst);
            let v = vec![0u8; bytes].into_boxed_slice();
            Some(Box::into_raw(v) as *mut u8)
        }

        unsafe fn free(&self, ptr: *mut u8, bytes: usize) {
            self.0.fetch_sub(bytes, Ordering::SeqCst);
            drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, bytes)));
        }
    }

    fn pool(budget: usize) -> (Arc<LeafPool>, Arc<AtomicUsize>) {
        let live = Arc::new(AtomicUsize::new(0));
        (LeafPool::new(budget, Box::new(Heap(live.clone()))), live)
    }

    #[test]
    fn classes() {
        assert_eq!(class_of(1), MIB);
        assert_eq!(class_of(MIB + 1), 2 * MIB);
        assert_eq!(class_of(35 * MIB + 7), 36 * MIB);
        assert_eq!(class_of(1_140_850_688), 1_140_850_688usize.div_ceil(16 * MIB) * 16 * MIB);
    }

    #[test]
    fn budget_is_never_exceeded_and_refusal_does_not_block() {
        let (p, live) = pool(10 * MIB);
        let a = p.try_lease(4 * MIB).unwrap();
        let b = p.try_lease(4 * MIB).unwrap();
        assert!(p.try_lease(4 * MIB).is_none(), "third 4 MiB lease exceeds the 10 MiB budget");
        assert!(live.load(Ordering::SeqCst) <= 10 * MIB);
        assert_eq!(p.refused.load(Ordering::Relaxed), 1);
        drop(a);
        let c = p.try_lease(4 * MIB).expect("returned buffer is reused");
        assert_eq!(live.load(Ordering::SeqCst), 8 * MIB, "reuse, no new allocation");
        drop((b, c));
        assert_eq!(p.leased_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn idle_buffers_of_other_classes_are_freed_to_fit_a_new_class() {
        let (p, live) = pool(10 * MIB);
        drop((p.try_lease(4 * MIB).unwrap(), p.try_lease(4 * MIB).unwrap()));
        assert_eq!(live.load(Ordering::SeqCst), 8 * MIB, "kept for reuse");
        let big = p.try_lease(9 * MIB).expect("idle 4 MiB buffers are freed to make room");
        assert_eq!(live.load(Ordering::SeqCst), 9 * MIB);
        assert!(p.try_lease(2 * MIB).is_none(), "the 9 MiB lease is busy; nothing idle to free");
        drop(big);
    }

    #[test]
    fn a_class_larger_than_the_budget_is_refused() {
        let (p, live) = pool(4 * MIB);
        assert!(p.try_lease(5 * MIB).is_none());
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn panic_while_holding_a_lease_returns_it() {
        let (p, _) = pool(4 * MIB);
        let p2 = p.clone();
        let r = std::thread::spawn(move || {
            let _l = p2.try_lease(4 * MIB).unwrap();
            panic!("simulated failure while holding a lease");
        })
        .join();
        assert!(r.is_err());
        assert_eq!(p.leased_bytes.load(Ordering::Relaxed), 0);
        assert!(p.try_lease(4 * MIB).is_some(), "lease returned on unwind");
    }

    #[test]
    fn budget_is_capped() {
        let (p, _) = pool(usize::MAX);
        assert_eq!(p.budget(), MAX_BUDGET);
    }
}
