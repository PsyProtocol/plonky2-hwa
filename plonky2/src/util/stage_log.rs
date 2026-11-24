//! Opt-in per-proof stage log for profiling (lab use).
//!
//! When the environment variable `PLONKY2_STAGE_LOG` names a file, every call to
//! `prove_with_partition_witness` appends one JSON line with the circuit shape and the wall time
//! of each top-level prover stage. When the variable is unset the cost is one cached lookup.

#[cfg(feature = "std")]
mod imp {
    use std::cell::Cell;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::string::String;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use web_time::Instant;
    use std::vec::Vec;

    use crate::field::extension::Extendable;
    use crate::hash::hash_types::RichField;
    use crate::plonk::circuit_data::CommonCircuitData;

    fn path() -> Option<&'static str> {
        static PATH: OnceLock<Option<String>> = OnceLock::new();
        PATH.get_or_init(|| {
            std::env::var("PLONKY2_STAGE_LOG")
                .ok()
                .filter(|p| !p.is_empty())
        })
        .as_deref()
    }

    std::thread_local! {
        /// Witness generation time measured by `prove`, consumed by the next stage log.
        static GENERATORS_SECONDS: Cell<Option<f64>> = const { Cell::new(None) };
        /// Sub-stage seconds accumulated on this thread since the current stage log started.
        static DETAIL: core::cell::RefCell<Vec<(&'static str, f64)>> = const { core::cell::RefCell::new(Vec::new()) };
    }

    /// Runs `f`, adding its wall time to sub-stage `name` of the current proof's stage log.
    #[inline]
    pub fn acc<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let s = t.elapsed().as_secs_f64();
        DETAIL.with(|d| {
            let mut d = d.borrow_mut();
            match d.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += s,
                None => d.push((name, s)),
            }
        });
        r
    }

    /// Sub-stage seconds summed over every thread (CPU time, not wall time) since the current
    /// stage log started; for code that runs on the rayon pool.
    static SHARED: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

    /// An interned copy of `name` (leaked once per distinct name), for `add_shared`.
    pub fn static_name(name: &str) -> &'static str {
        static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        let mut names = NAMES.lock().unwrap();
        if let Some(found) = names.iter().find(|n| **n == name) {
            return found;
        }
        let leaked: &'static str = std::boxed::Box::leak(String::from(name).into_boxed_str());
        names.push(leaked);
        leaked
    }

    /// A start time for `add_shared`, or `None` when the log is off.
    #[inline]
    pub fn timer() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    /// Adds the time since `start` to shared sub-stage `name`.
    #[inline]
    pub fn add_shared(name: &'static str, start: Option<Instant>) {
        let Some(start) = start else { return };
        let s = start.elapsed().as_secs_f64();
        if let Ok(mut d) = SHARED.lock() {
            match d.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += s,
                None => d.push((name, s)),
            }
        }
    }

    /// Like `acc`, but callable from any thread; the times of all threads are added up.
    #[inline]
    pub fn acc_shared<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let s = t.elapsed().as_secs_f64();
        if let Ok(mut d) = SHARED.lock() {
            match d.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += s,
                None => d.push((name, s)),
            }
        }
        r
    }

    /// Turned on by `set_enabled` (the only way in a browser, which has no environment).
    static ENABLED: AtomicBool = AtomicBool::new(false);
    /// Finished lines kept in memory for `take`.
    static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub fn set_enabled(enabled: bool) {
        ENABLED.store(enabled, Ordering::Relaxed);
    }

    /// Milliseconds since the first call in this process: one clock for every timeline line.
    pub fn now_ms() -> f64 {
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    }

    /// Adds a caller's own JSON line (an RPC, a phase) to the lines `take` returns.
    pub fn note(line: impl FnOnce() -> String) {
        if ENABLED.load(Ordering::Relaxed) {
            if let Ok(mut lines) = LINES.lock() {
                lines.push(line());
            }
        }
    }

    /// The stage lines recorded since the last call, one JSON object per proof.
    pub fn take() -> Vec<String> {
        LINES.lock().map(|mut lines| core::mem::take(&mut *lines)).unwrap_or_default()
    }

    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed) || path().is_some()
    }

    pub fn generators_start() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    pub fn generators_end(start: Option<Instant>) {
        if let Some(t) = start {
            GENERATORS_SECONDS.with(|c| c.set(Some(t.elapsed().as_secs_f64())));
        }
    }

    #[derive(Debug)]
    pub struct StageLog {
        start: Instant,
        last: Instant,
        stages: Vec<(&'static str, f64)>,
        shape: String,
    }

    impl StageLog {
        pub fn start<F: RichField + Extendable<D>, const D: usize>(
            common: &CommonCircuitData<F, D>,
            circuit_digest: u64,
        ) -> Option<Self> {
            if !enabled() {
                return None;
            }
            let config = &common.config;
            let fri = &common.fri_params;
            let shape = format!(
                concat!(
                    "\"circuit\":\"{:016x}\",\"degree_bits\":{},\"num_wires\":{},\"num_routed_wires\":{},",
                    "\"num_constants\":{},\"num_gates\":{},\"quotient_degree_factor\":{},",
                    "\"num_partial_products\":{},\"num_lookup_polys\":{},\"num_challenges\":{},",
                    "\"num_public_inputs\":{},\"rate_bits\":{},\"cap_height\":{},\"zero_knowledge\":{},",
                    "\"reduction_arity_bits\":{:?},\"num_query_rounds\":{},\"pow_bits\":{}"
                ),
                circuit_digest,
                common.degree_bits(),
                config.num_wires,
                config.num_routed_wires,
                common.num_constants,
                common.gates.len(),
                common.quotient_degree_factor,
                common.num_partial_products,
                common.num_all_lookup_polys(),
                config.num_challenges,
                common.num_public_inputs,
                config.fri_config.rate_bits,
                config.fri_config.cap_height,
                config.zero_knowledge,
                fri.reduction_arity_bits,
                config.fri_config.num_query_rounds,
                config.fri_config.proof_of_work_bits,
            );
            DETAIL.with(|d| d.borrow_mut().clear());
            let now = Instant::now();
            Some(Self {
                start: now,
                last: now,
                stages: Vec::new(),
                shape,
            })
        }

        pub fn mark(&mut self, stage: &'static str) {
            let now = Instant::now();
            self.stages.push((stage, (now - self.last).as_secs_f64()));
            self.last = now;
        }

        pub fn finish(self) {
            let generators = GENERATORS_SECONDS.with(|c| c.take());
            let stages: Vec<String> = self
                .stages
                .iter()
                .map(|(name, s)| format!("\"{name}\":{s:.6}"))
                .collect();
            let mut detail: Vec<String> = DETAIL.with(|d| {
                d.borrow_mut()
                    .drain(..)
                    .map(|(name, s)| format!("\"{name}\":{s:.6}"))
                    .collect()
            });
            if let Ok(mut shared) = SHARED.lock() {
                detail.extend(shared.drain(..).map(|(name, s)| format!("\"cpu_{name}\":{s:.6}")));
            }
            let line = format!(
                "{{\"kind\":\"proof\",\"end_ms\":{:.1},{},\"generators_s\":{},\"prove_s\":{:.6},\"stages\":{{{}}},\"detail\":{{{}}}}}\n",
                now_ms(),
                self.shape,
                generators.map_or("null".into(), |g| format!("{g:.6}")),
                (self.last - self.start).as_secs_f64(),
                stages.join(","),
                detail.join(","),
            );
            if ENABLED.load(Ordering::Relaxed) {
                if let Ok(mut lines) = LINES.lock() {
                    lines.push(line.trim_end().into());
                }
            }
            let Some(path) = path() else { return };
            // One write call per line keeps concurrent appends from interleaving.
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    use crate::field::extension::Extendable;
    use crate::hash::hash_types::RichField;
    use crate::plonk::circuit_data::CommonCircuitData;

    pub fn enabled() -> bool {
        false
    }
    pub fn set_enabled(_enabled: bool) {}
    pub fn now_ms() -> f64 {
        0.0
    }
    pub fn note(_line: impl FnOnce() -> alloc::string::String) {}
    #[inline]
    pub fn acc_shared<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }
    pub fn static_name(_name: &str) -> &'static str {
        ""
    }
    #[inline]
    pub fn timer() -> Option<()> {
        None
    }
    #[inline]
    pub fn add_shared(_name: &'static str, _start: Option<()>) {}
    pub fn take() -> alloc::vec::Vec<alloc::string::String> {
        alloc::vec::Vec::new()
    }
    pub fn generators_start() -> Option<()> {
        None
    }
    pub fn generators_end(_start: Option<()>) {}
    #[inline]
    pub fn acc<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }
    #[derive(Debug)]
    pub struct StageLog;
    impl StageLog {
        pub fn start<F: RichField + Extendable<D>, const D: usize>(
            _common: &CommonCircuitData<F, D>,
            _circuit_digest: u64,
        ) -> Option<Self> {
            None
        }
        pub fn mark(&mut self, _stage: &'static str) {}
        pub fn finish(self) {}
    }
}

pub use imp::*;
