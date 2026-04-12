//! `TamBackend` trait and the `NullBackend` smoke-test implementation (campsite 4.1).
//!
//! ## The trait contract
//!
//! Every backend that the harness tests implements `TamBackend`.  The trait is
//! intentionally minimal — it matches what the test oracle needs, not what a full
//! execution runtime would expose.  This keeps the trait stable even as Peak 1
//! (IR), Peak 3 (PTX), Peak 5 (CPU), and Peak 7 (Vulkan) land their backends.
//!
//! ## Plugging in a real backend
//!
//! When a backend is ready (e.g., `CpuInterpreterBackend` from Peak 5):
//!
//! ```rust,ignore
//! use tambear_tam_test_harness::{TamBackend, TamProgram, Inputs, Outputs};
//!
//! struct CpuInterpreterBackend { /* ... */ }
//!
//! impl TamBackend for CpuInterpreterBackend {
//!     fn name(&self) -> String { "cpu-interpreter".into() }
//!
//!     fn run(&self, program: &TamProgram, inputs: &Inputs) -> Outputs {
//!         // interpret program.ir using inputs ...
//!     }
//!
//!     fn is_available(&self) -> bool { true }
//! }
//! ```
//!
//! Then register it with the harness:
//!
//! ```rust,ignore
//! let mut registry = BackendRegistry::new();
//! registry.push(Box::new(CpuInterpreterBackend::new()));
//! let results = run_all_backends(&registry, &program, &inputs);
//! ```

use crate::{TamProgram, Inputs, Outputs};

/// The uniform interface every backend must implement.
///
/// `name()` uniquely identifies the backend in reports.  Two backends with the
/// same name are an error — the harness detects this at registration.
pub trait TamBackend: Send + Sync {
    /// Human-readable unique identifier, e.g. `"cpu-interpreter"`,
    /// `"cuda-ptx-raw"`, `"vulkan-spirv"`, `"cuda-nvrtc-legacy"`.
    fn name(&self) -> String;

    /// Execute `program` on `inputs` and return the output slots.
    ///
    /// The backend must produce the same number of output slots as the program
    /// declares.  Returning an empty `Outputs` when the program expects results
    /// will cause the harness to flag a dimension mismatch error rather than a
    /// numerical disagreement.
    fn run(&self, program: &TamProgram, inputs: &Inputs) -> Outputs;

    /// Whether this backend is currently usable on this machine.
    ///
    /// A backend that returns `false` is silently skipped by the harness.
    /// This allows GPU backends to be compiled into the harness unconditionally
    /// but gracefully absent when the hardware or driver is missing.
    ///
    /// Returning `false` never hides a real disagreement — it just means "no
    /// data point from this backend on this run."  The harness logs skipped
    /// backends in the report.
    fn is_available(&self) -> bool {
        true
    }
}

/// A registry of backends that the harness will run.
///
/// Populated by tests; each backend is stored by index.  The harness enforces
/// that all names are unique at registration time.
#[derive(Default)]
pub struct BackendRegistry {
    backends: Vec<Box<dyn TamBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self { Self::default() }

    /// Register a backend.  Panics if a backend with the same name already exists.
    pub fn push(&mut self, backend: Box<dyn TamBackend>) {
        let name = backend.name();
        let conflict = self.backends.iter().any(|b| b.name() == name);
        assert!(!conflict, "duplicate backend name: {name:?}");
        self.backends.push(backend);
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn TamBackend> {
        self.backends.iter().map(|b| b.as_ref())
    }

    pub fn len(&self) -> usize { self.backends.len() }
    pub fn is_empty(&self) -> bool { self.backends.is_empty() }
}

// ---------------------------------------------------------------------------
// NullBackend — smoke-test only
// ---------------------------------------------------------------------------

/// A backend that always returns zero-filled outputs.
///
/// Used exclusively for smoke-testing that the harness infrastructure compiles
/// and links correctly.  It must NEVER be registered alongside real backends in
/// a real test, because it will agree bit-exactly with any other backend that
/// happens to produce zeros.
///
/// # Why this exists
///
/// The harness trait shape must be reviewable and testable independently of any
/// real IR or execution engine.  `NullBackend` gives us that.  It is not a
/// shortcut for "I'll implement the real backend later" — the real backends live
/// in their own crates and are plugged in when they land.
pub struct NullBackend {
    name: String,
    n_output_slots: usize,
}

impl NullBackend {
    /// Create a null backend that returns `n_output_slots` zeros.
    pub fn new(n_output_slots: usize) -> Self {
        Self { name: "null".into(), n_output_slots }
    }

    pub fn named(name: impl Into<String>, n_output_slots: usize) -> Self {
        Self { name: name.into(), n_output_slots }
    }
}

impl TamBackend for NullBackend {
    fn name(&self) -> String { self.name.clone() }

    fn run(&self, _program: &TamProgram, _inputs: &Inputs) -> Outputs {
        Outputs::from_slots(vec![0.0; self.n_output_slots])
    }

    fn is_available(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_returns_zeros() {
        let backend = NullBackend::new(3);
        let program = TamProgram::pure("smoke");
        let inputs = Inputs::new().with_buf("x", vec![1.0, 2.0, 3.0]);
        let out = backend.run(&program, &inputs);
        assert_eq!(out.slots.len(), 3);
        assert!(out.slots.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn null_backend_is_available() {
        let backend = NullBackend::new(1);
        assert!(backend.is_available());
    }

    #[test]
    fn null_backend_name() {
        let backend = NullBackend::named("test-null", 0);
        assert_eq!(backend.name(), "test-null");
    }

    #[test]
    fn registry_deduplicates_names() {
        let result = std::panic::catch_unwind(|| {
            let mut reg = BackendRegistry::new();
            reg.push(Box::new(NullBackend::named("same-name", 1)));
            reg.push(Box::new(NullBackend::named("same-name", 1)));
        });
        assert!(result.is_err(), "should panic on duplicate name");
    }

    #[test]
    fn registry_accepts_unique_names() {
        let mut reg = BackendRegistry::new();
        reg.push(Box::new(NullBackend::named("a", 1)));
        reg.push(Box::new(NullBackend::named("b", 1)));
        assert_eq!(reg.len(), 2);
    }
}
