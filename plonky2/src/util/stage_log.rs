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
    use std::time::Instant;
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
        /// One JSON object per polynomial commitment made on this thread during the current proof.
        static COMMITS: core::cell::RefCell<Vec<String>> = const { core::cell::RefCell::new(Vec::new()) };
    }

    /// Records one polynomial commitment: which backend ran it and why, its shape, and (GPU) the
    /// time spent waiting for the device lock, executing, and rebuilding the host-side tree.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_event(
        backend: &'static str,
        reason: &'static str,
        lg_n: usize,
        rate_bits: usize,
        cols: usize,
        buffer_wait_s: f64,
        lock_wait_s: f64,
        exec_s: f64,
        rebuild_s: f64,
        total_s: f64,
    ) {
        if !enabled() {
            return;
        }
        let e = format!(
            "{{\"backend\":\"{backend}\",\"reason\":\"{reason}\",\"rows\":{},\"lde_rows\":{},\"cols\":{cols},\"buffer_wait_s\":{buffer_wait_s:.6},\"lock_wait_s\":{lock_wait_s:.6},\"exec_s\":{exec_s:.6},\"rebuild_s\":{rebuild_s:.6},\"total_s\":{total_s:.6}}}",
            1usize << lg_n,
            1usize << (lg_n + rate_bits),
        );
        COMMITS.with(|c| c.borrow_mut().push(e));
    }

    /// Runs `f`, adding its wall time to sub-stage `name` of the current proof's stage log.
    /// NVTX range only (no stage-log accounting); a no-op without the `cuda` feature.
    #[inline]
    pub fn range<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        // CUDA range installs the same hook; avoid entering it twice.
        #[cfg(not(feature = "cuda"))]
        let _observation = crate::util::observation::enter_named(name);
        #[cfg(feature = "cuda")]
        {
            crate::gpu::range(name, f)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = name;
            f()
        }
    }

    #[inline]
    pub fn acc<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return range(name, f);
        }
        let t = Instant::now();
        let r = range(name, f);
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

    pub fn enabled() -> bool {
        path().is_some()
    }

    /// Adds `seconds` to the current proof's detail entry `name`.
    pub fn detail(name: &'static str, seconds: f64) {
        if !enabled() {
            return;
        }
        DETAIL.with(|d| d.borrow_mut().push((name, seconds)));
    }

    /// Records a quotient evaluation on `backend` (device seconds and total seconds) in the
    /// current proof's detail.
    pub fn quotient_event(backend: &'static str, _reason: &str, exec_s: f64, total_s: f64) {
        if !enabled() {
            return;
        }
        DETAIL.with(|d| {
            let mut d = d.borrow_mut();
            d.push(if backend == "gpu" { ("quotient_gpu_exec", exec_s) } else { ("quotient_cpu", exec_s) });
            d.push(("quotient_total", total_s));
        });
    }

    pub fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
        let t = Instant::now();
        let r = f();
        (r, t.elapsed().as_secs_f64())
    }

    /// `PLONKY2_VERIFY_AFTER_PROVE=1`: `CircuitData::prove` verifies every proof it returns.
    pub fn verify_after_prove() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("PLONKY2_VERIFY_AFTER_PROVE").ok().as_deref() == Some("1"))
    }

    /// `PLONKY2_POW_DETERMINISTIC=1`: the FRI proof-of-work search returns the smallest valid
    /// witness instead of any, so proofs are reproducible run to run.
    pub fn pow_deterministic() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("PLONKY2_POW_DETERMINISTIC").ok().as_deref() == Some("1"))
    }

    /// `PLONKY2_POW_SCALAR=1`: the FRI proof-of-work search uses the one-at-a-time scalar
    /// permutation instead of the lane-parallel one (same result; for A/B measurements).
    pub fn pow_scalar() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("PLONKY2_POW_SCALAR").ok().as_deref() == Some("1"))
    }

    /// `PLONKY2_QUOTIENT_REFERENCE=1`: the quotient polynomials are evaluated one point at a time
    /// by the original code instead of the packed path (same result; for A/B measurements).
    pub fn quotient_reference() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("PLONKY2_QUOTIENT_REFERENCE").ok().as_deref() == Some("1"))
    }

    /// `PLONKY2_DETERMINISTIC_RANDOM=1`: `RandomValueGenerator` (unused public-input-gate wires,
    /// ZK blinding rows) uses a fixed function of the target instead of fresh randomness.
    /// Correctness comparisons only; never for production (it removes the blinding).
    pub fn deterministic_random() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("PLONKY2_DETERMINISTIC_RANDOM").ok().as_deref() == Some("1"))
    }

    static VERIFY_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static VERIFY_FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// (verified ok, verification failed) since process start.
    pub fn verify_counts() -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (VERIFY_OK.load(Relaxed), VERIFY_FAILED.load(Relaxed))
    }

    /// Records the result of verifying a just-produced proof: a counter, an error log on failure,
    /// and (with the stage log on) one `{"verify":...}` line.
    pub fn verify_event(circuit_digest: u64, error: Option<String>, seconds: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        match &error {
            None => VERIFY_OK.fetch_add(1, Relaxed),
            Some(e) => {
                log::error!("proof verification failed right after proving (circuit {circuit_digest:016x}): {e}");
                VERIFY_FAILED.fetch_add(1, Relaxed)
            }
        };
        let Some(path) = path() else { return };
        let line = format!(
            "{{\"verify\":{},\"circuit\":\"{circuit_digest:016x}\",\"verify_s\":{seconds:.6}}}\n",
            error.is_none()
        );
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
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

    /// The first time a circuit is proven, appends one `{"circuit_gates":...}` line listing its
    /// gates (id, selector, group, constraint count, degree) and constraint layout.
    fn log_circuit_gates_once<F: RichField + Extendable<D>, const D: usize>(
        common: &CommonCircuitData<F, D>,
        circuit_digest: u64,
    ) {
        static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
        {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if seen.contains(&circuit_digest) {
                return;
            }
            seen.push(circuit_digest);
        }
        let Some(path) = path() else { return };
        let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let gates: Vec<String> = common
            .gates
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let s = common.selectors_info.selector_indices[i];
                let r = &common.selectors_info.groups[s];
                format!(
                    "{{\"id\":\"{}\",\"selector\":{s},\"group\":[{},{}],\"constraints\":{},\"degree\":{}}}",
                    escape(&g.0.id()),
                    r.start,
                    r.end,
                    g.0.num_constraints(),
                    g.0.degree()
                )
            })
            .collect();
        let line = format!(
            "{{\"circuit_gates\":\"{circuit_digest:016x}\",\"degree_bits\":{},\"num_constants\":{},\"num_selectors\":{},\"num_lookup_selectors\":{},\"num_gate_constraints\":{},\"num_wires\":{},\"gates\":[{}]}}\n",
            common.degree_bits(),
            common.num_constants,
            common.selectors_info.num_selectors(),
            common.num_lookup_selectors,
            common.num_gate_constraints,
            common.config.num_wires,
            gates.join(",")
        );
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    impl StageLog {
        pub fn start<F: RichField + Extendable<D>, const D: usize>(
            common: &CommonCircuitData<F, D>,
            circuit_digest: u64,
        ) -> Option<Self> {
            path()?;
            log_circuit_gates_once(common, circuit_digest);
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
            COMMITS.with(|c| c.borrow_mut().clear());
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
            let Some(path) = path() else { return };
            let generators = GENERATORS_SECONDS.with(|c| c.take());
            let stages: Vec<String> = self
                .stages
                .iter()
                .map(|(name, s)| format!("\"{name}\":{s:.6}"))
                .collect();
            let detail: Vec<String> = DETAIL.with(|d| {
                d.borrow_mut()
                    .drain(..)
                    .map(|(name, s)| format!("\"{name}\":{s:.6}"))
                    .collect()
            });
            let commits: Vec<String> = COMMITS.with(|c| c.borrow_mut().drain(..).collect());
            let thread = std::thread::current();
            let line = format!(
                "{{\"t\":{:.3},{},\"generators_s\":{},\"prove_s\":{:.6},\"stages\":{{{}}},\"detail\":{{{}}},\"commits\":[{}],\"thread\":\"{}\"}}\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
                self.shape,
                generators.map_or("null".into(), |g| format!("{g:.6}")),
                (self.last - self.start).as_secs_f64(),
                stages.join(","),
                detail.join(","),
                commits.join(","),
                thread.name().unwrap_or("?").replace('"', "'"),
            );
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
    pub fn generators_start() -> Option<()> {
        None
    }
    pub fn generators_end(_start: Option<()>) {}
    pub fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
        (f(), 0.0)
    }
    pub fn quotient_event(_backend: &'static str, _reason: &str, _exec_s: f64, _total_s: f64) {}
    pub fn detail(_name: &'static str, _seconds: f64) {}
    pub fn verify_after_prove() -> bool {
        false
    }
    pub fn pow_deterministic() -> bool {
        false
    }
    pub fn pow_scalar() -> bool {
        false
    }
    pub fn quotient_reference() -> bool {
        false
    }
    pub fn deterministic_random() -> bool {
        false
    }
    pub fn verify_event(_circuit_digest: u64, _error: Option<std::string::String>, _seconds: f64) {}
    #[inline]
    pub fn acc<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }
    #[inline]
    pub fn range<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn commit_event(
        _backend: &'static str,
        _reason: &'static str,
        _lg_n: usize,
        _rate_bits: usize,
        _cols: usize,
        _buffer_wait_s: f64,
        _lock_wait_s: f64,
        _exec_s: f64,
        _rebuild_s: f64,
        _total_s: f64,
    ) {
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
