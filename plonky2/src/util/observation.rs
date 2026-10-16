//! Optional synchronous phase hook. No logger, global lock, allocation or clock
//! read when no hook is installed. Callers own bounded storage and must not block.
//! The hook follows the proving thread, not Rayon workers; a phase measures the
//! caller's wall time including any wait for its parallel work.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Witness,
    WiresCommit,
    PartialProducts,
    ZsCommit,
    Quotient,
    QuotientCommit,
    Openings,
    Fri,
    LargeProofWait,
    LargeProofHeld,
    GpuBufferWait,
    GpuLaneWait,
    GpuLaneHeld,
    GpuDevice,
    GpuRebuild,
    GpuQuotientWait,
    GpuFriWait,
}

pub fn named(name: &'static str) -> Option<Phase> {
    Some(match name {
        "prove:generators" => Phase::Witness,
        "commit:wires" => Phase::WiresCommit,
        "prove:partial_products" => Phase::PartialProducts,
        "commit:zs" => Phase::ZsCommit,
        "prove:quotient_polys" => Phase::Quotient,
        "commit:quotient" => Phase::QuotientCommit,
        "prove:openings" => Phase::Openings,
        "prove:fri" => Phase::Fri,
        "gpu:buffer_wait" => Phase::GpuBufferWait,
        "gpu:lock_wait" => Phase::GpuLaneWait,
        "gpu:device" => Phase::GpuDevice,
        "gpu:rebuild" => Phase::GpuRebuild,
        "gpu:quotient_lock_wait" => Phase::GpuQuotientWait,
        "gpu:fri_lock_wait" => Phase::GpuFriWait,
        _ => return None,
    })
}

#[cfg(feature = "std")]
mod imp {
    use std::cell::RefCell;
    use std::sync::Arc;

    use super::Phase;

    // A guard's destruction ends the phase, including on unwind.
    pub type Hook = Arc<dyn Fn(Phase) -> Box<dyn Send + Sync> + Send + Sync>;
    std::thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }
    pub struct Span(Option<Box<dyn Send + Sync>>);
    impl core::fmt::Debug for Span {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Span")
                .field("active", &self.0.is_some())
                .finish()
        }
    }
    pub fn enter(phase: Phase) -> Span {
        let hook = HOOK.with(|h| h.borrow().clone());
        Span(hook.map(|f| f(phase)))
    }
    pub fn with_hook<R>(hook: Hook, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<Hook>);
        impl Drop for Restore {
            fn drop(&mut self) {
                HOOK.with(|h| *h.borrow_mut() = self.0.take());
            }
        }
        let _restore = Restore(HOOK.with(|h| h.replace(Some(hook))));
        f()
    }
}
#[cfg(feature = "std")]
pub use imp::*;

#[cfg(not(feature = "std"))]
#[derive(Debug)]
pub struct Span;
#[cfg(not(feature = "std"))]
pub fn enter(_phase: Phase) -> Span {
    Span
}

pub fn enter_named(name: &'static str) -> Option<Span> {
    named(name).map(enter)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    #[test]
    fn nesting_panic_and_thread_isolation() {
        struct End(Arc<Mutex<Vec<&'static str>>>);
        impl Drop for End {
            fn drop(&mut self) {
                self.0.lock().unwrap().push("end");
            }
        }
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();
        let hook: Hook = Arc::new(move |_| {
            capture.lock().unwrap().push("begin");
            Box::new(End(capture.clone()))
        });
        let _off = enter(Phase::Fri);
        with_hook(hook.clone(), || {
            let _outer = enter(Phase::Fri);
            std::thread::spawn(|| {
                let _unrelated = enter(Phase::Fri);
            })
            .join()
            .unwrap();
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_hook(hook.clone(), || {
                    let _inner = enter(Phase::Quotient);
                    panic!("fixture");
                });
            }));
            assert!(failed.is_err());
            let _after = enter(Phase::Witness);
        });
        let _off_again = enter(Phase::Fri);
        assert_eq!(
            *events.lock().unwrap(),
            ["begin", "begin", "end", "begin", "end", "end"]
        );
    }
}
