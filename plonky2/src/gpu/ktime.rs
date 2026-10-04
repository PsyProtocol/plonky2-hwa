//! Device time per GPU phase (lab, `PLONKY2_GPU_KTIME=1`).
//!
//! A stage records a CUDA event on its stream before its first operation (`start`) and after each
//! phase (`lap`), then, once it has synchronised that stream, adds the time between consecutive
//! events to a per-phase total (`collect`). The events live in a per-thread pool: every stage
//! runs its stream synchronously on one thread. With several lanes or streams in flight a phase's
//! time includes sharing the device, so a breakdown of device work is taken with one lane.
//! Off by default; then every call returns at once.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::*;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PLONKY2_GPU_KTIME").ok().as_deref() == Some("1"))
}

struct Timer {
    events: Vec<usize>,
    names: Vec<&'static str>,
    used: usize,
}

std::thread_local! {
    static TIMER: core::cell::RefCell<Timer> = const { core::cell::RefCell::new(Timer { events: Vec::new(), names: Vec::new(), used: 0 }) };
}

/// Phase -> (milliseconds, count).
static TOTALS: Mutex<BTreeMap<&'static str, (f64, u64)>> = Mutex::new(BTreeMap::new());

fn record(t: &mut Timer, stream: *mut c_void, name: &'static str) {
    if t.used == t.events.len() {
        let mut e = core::ptr::null_mut();
        if unsafe { ffi::cudaEventCreateWithFlags(&mut e, 0) } != 0 {
            unsafe { ffi::cudaGetLastError() };
            t.used = 0;
            return;
        }
        t.events.push(e as usize);
        t.names.push("");
    }
    if unsafe { ffi::cudaEventRecord(t.events[t.used] as *mut c_void, stream) } != 0 {
        unsafe { ffi::cudaGetLastError() };
        t.used = 0;
        return;
    }
    t.names[t.used] = name;
    t.used += 1;
}

/// Starts a sequence on `stream` (drops any sequence left by an earlier failure).
pub(crate) fn start(stream: *mut c_void) {
    if enabled() {
        TIMER.with(|t| {
            let mut t = t.borrow_mut();
            t.used = 0;
            record(&mut t, stream, "");
        });
    }
}

/// Ends phase `name` on `stream`, the stream of the sequence's `start`.
pub(crate) fn lap(stream: *mut c_void, name: &'static str) {
    if enabled() {
        TIMER.with(|t| {
            let mut t = t.borrow_mut();
            if t.used > 0 {
                record(&mut t, stream, name);
            }
        });
    }
}

/// Adds the sequence's phases to the totals. Call after synchronising its stream.
pub(crate) fn collect() {
    if !enabled() {
        return;
    }
    TIMER.with(|t| {
        let mut t = t.borrow_mut();
        let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        for i in 1..t.used {
            let mut ms = 0f32;
            if unsafe { cudaEventElapsedTime(&mut ms, t.events[i - 1] as *mut c_void, t.events[i] as *mut c_void) } == 0 {
                let e = totals.entry(t.names[i]).or_insert((0.0, 0));
                e.0 += ms as f64;
                e.1 += 1;
            } else {
                unsafe { ffi::cudaGetLastError() };
            }
        }
        t.used = 0;
    });
}

/// `phase=total_ms/count ...`, largest total first.
pub(crate) fn summary() -> String {
    let totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<_> = totals.iter().collect();
    v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    v.iter().map(|(k, (ms, n))| format!("{k}={ms:.0}/{n}")).collect::<Vec<_>>().join(" ")
}

extern "C" {
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}
