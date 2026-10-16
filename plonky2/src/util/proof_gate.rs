//! Bounds how many large proofs run at once (`PLONKY2_LARGE_PROOF_SLOTS`).
//!
//! A proof's host memory grows with its circuit: a 2^17 proof holds about 1.2 GB while it runs
//! (witness, coefficients, Merkle leaves). Under a burst of large requests every one of them
//! starts at once and the peak is their sum, although their commitments then queue on the GPU
//! anyway. With `PLONKY2_LARGE_PROOF_SLOTS=n`, proofs of circuits with at least
//! `PLONKY2_LARGE_PROOF_MIN_DEGREE_BITS` (default 16) degree bits wait for one of `n` slots before
//! generating their witness. Unset or 0: no limit.

#[cfg(feature = "std")]
mod imp {
    use core::cell::Cell;
    use std::sync::{Condvar, Mutex, OnceLock};
    use std::time::Instant;

    fn slots() -> usize {
        static V: OnceLock<usize> = OnceLock::new();
        *V.get_or_init(|| std::env::var("PLONKY2_LARGE_PROOF_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
    }

    fn min_degree_bits() -> usize {
        static V: OnceLock<usize> = OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("PLONKY2_LARGE_PROOF_MIN_DEGREE_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
        })
    }

    /// Slots in use and a condition variable to wait for one.
    struct Gate {
        in_use: Mutex<usize>,
        freed: Condvar,
    }

    static GATE: Gate = Gate { in_use: Mutex::new(0), freed: Condvar::new() };

    std::thread_local! {
        /// This thread already holds a slot (a nested prove call must not wait for a second one).
        static HELD: Cell<bool> = const { Cell::new(false) };
        /// Seconds this thread's last acquisition waited, for the stage log.
        static WAITED: Cell<f64> = const { Cell::new(0.0) };
    }

    /// Takes the wait of this thread's last acquisition (0 once taken).
    pub fn take_wait() -> f64 {
        WAITED.with(|w| w.replace(0.0))
    }

    /// Holds a slot of its gate until dropped.
    pub struct LargeProofSlot(Option<&'static Gate>, Option<crate::util::observation::Span>);

    impl core::fmt::Debug for LargeProofSlot {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_tuple("LargeProofSlot").field(&self.0.is_some()).finish()
        }
    }

    impl Drop for LargeProofSlot {
        fn drop(&mut self) {
            if let Some(g) = self.0 {
                HELD.with(|h| h.set(false));
                *g.in_use.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
                g.freed.notify_one();
                self.1.take(); // Record release only after the permit was returned.
            }
        }
    }

    /// Waits for a slot if the circuit is large and slots are limited; returns the guard and the
    /// time waited.
    pub fn acquire(degree_bits: usize) -> (LargeProofSlot, f64) {
        acquire_with(&GATE, slots(), min_degree_bits(), degree_bits)
    }

    fn acquire_with(gate: &'static Gate, n: usize, min_bits: usize, degree_bits: usize) -> (LargeProofSlot, f64) {
        if n == 0 || degree_bits < min_bits || HELD.with(|h| h.get()) {
            return (LargeProofSlot(None, None), 0.0);
        }
        let start = Instant::now();
        let observation_wait = crate::util::observation::enter(crate::util::observation::Phase::LargeProofWait);
        let mut in_use = gate.in_use.lock().unwrap_or_else(|e| e.into_inner());
        while *in_use >= n {
            in_use = gate.freed.wait(in_use).unwrap_or_else(|e| e.into_inner());
        }
        *in_use += 1;
        drop(in_use);
        HELD.with(|h| h.set(true));
        // Own the permit before any user-supplied hook or guard destructor runs.
        let mut slot = LargeProofSlot(Some(gate), None);
        let waited = start.elapsed().as_secs_f64();
        WAITED.with(|w| w.set(waited));
        drop(observation_wait);
        slot.1 = Some(crate::util::observation::enter(crate::util::observation::Phase::LargeProofHeld));
        (slot, waited)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Gates of their own, so proofs of other tests running in parallel do not count.
        static G1: Gate = Gate { in_use: Mutex::new(0), freed: Condvar::new() };
        static G2: Gate = Gate { in_use: Mutex::new(0), freed: Condvar::new() };

        #[test]
        fn hook_panic_returns_permit_and_clears_thread_state() {
            use crate::util::observation::{with_hook, Hook, Phase};
            static G3: Gate = Gate { in_use: Mutex::new(0), freed: Condvar::new() };
            struct End(bool);
            impl Drop for End {
                fn drop(&mut self) { if self.0 { panic!("wait span destructor fixture"); } }
            }
            for fail_wait_drop in [false, true] {
                let hook: Hook = Arc::new(move |phase| {
                    if !fail_wait_drop && phase == Phase::LargeProofHeld { panic!("held callback fixture"); }
                    Box::new(End(fail_wait_drop && phase == Phase::LargeProofWait))
                });
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_hook(hook, || { let _slot = acquire_with(&G3, 1, 16, 17); });
                }));
                assert!(result.is_err());
                assert_eq!(*G3.in_use.lock().unwrap(), 0);
                assert!(!HELD.with(|h| h.get()));
                let (slot, _) = acquire_with(&G3, 1, 16, 17);
                assert!(slot.0.is_some());
                drop(slot);
                assert_eq!(*G3.in_use.lock().unwrap(), 0);
            }
        }

        #[test]
        fn small_circuits_and_nested_calls_do_not_wait() {
            let (none, w) = acquire_with(&G1, 0, 16, 20);
            assert!(none.0.is_none() && w == 0.0, "no limit without slots");
            let (outer, _) = acquire_with(&G1, 1, 16, 17);
            let (nested, w) = acquire_with(&G1, 1, 16, 17);
            assert!(nested.0.is_none() && w == 0.0, "a nested call must not wait for a second slot");
            let (small, _) = acquire_with(&G1, 1, 16, 12);
            assert!(small.0.is_none());
            drop((nested, small, outer));
            assert_eq!(*G1.in_use.lock().unwrap(), 0);
        }

        #[test]
        fn at_most_n_large_proofs_hold_slots() {
            let (now, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let (now, peak) = (now.clone(), peak.clone());
                    std::thread::spawn(move || {
                        let (slot, _) = acquire_with(&G2, 3, 16, 17);
                        let k = now.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(k, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        now.fetch_sub(1, Ordering::SeqCst);
                        drop(slot);
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            assert_eq!(peak.load(Ordering::SeqCst), 3);
        }
    }
}

#[cfg(feature = "std")]
pub use imp::{acquire, take_wait, LargeProofSlot};

/// Without `std` there is no limit.
#[cfg(not(feature = "std"))]
#[derive(Debug)]
pub struct LargeProofSlot;

#[cfg(not(feature = "std"))]
pub fn acquire(_degree_bits: usize) -> (LargeProofSlot, f64) {
    (LargeProofSlot, 0.0)
}

#[cfg(not(feature = "std"))]
pub fn take_wait() -> f64 {
    0.0
}
