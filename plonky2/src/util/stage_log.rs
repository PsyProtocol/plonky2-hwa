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

    pub fn enabled() -> bool {
        path().is_some()
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
            path()?;
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
            let thread = std::thread::current();
            let line = format!(
                "{{\"t\":{:.3},{},\"generators_s\":{},\"prove_s\":{:.6},\"stages\":{{{}}},\"detail\":{{{}}},\"thread\":\"{}\"}}\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
                self.shape,
                generators.map_or("null".into(), |g| format!("{g:.6}")),
                (self.last - self.start).as_secs_f64(),
                stages.join(","),
                detail.join(","),
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
