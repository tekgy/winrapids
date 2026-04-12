//! # tambear-tam-test-harness
//!
//! The test oracle for the Bit-Exact Trek.
//!
//! This crate owns:
//! - [`TamBackend`] — the uniform interface every backend must implement (campsite 4.1)
//! - [`TamProgram`] / [`Inputs`] / [`Outputs`] — placeholder types; IR Architect will
//!   replace them with real types from `tambear-tam-ir` once that crate lands.
//! - [`ToleranceSpec`] — `bit_exact` and `within_ulp` tolerance policies (campsite 4.4/4.5)
//! - [`assert_cross_backend_agreement`] — runs all backends, diffs every pair (campsite 4.3)
//! - [`run_all_backends`] — collects `(name, Outputs)` from every backend in a registry
//!
//! ## Invariants enforced here
//!
//! - **I5**: reduction tests that are non-deterministic before Peak 6 are marked
//!   `#[ignore = "xfail_nondeterministic — remove after Peak 6"]`.  No reduction test
//!   may assert bit-exactness until the `@xfail_nondeterministic` annotation is lifted.
//! - **I9**: mpmath is the oracle.  When R and Python disagree, we call `ulp_distance`
//!   against the mpmath reference, not against each other.
//! - **I10**: every backend that joins the harness makes the cross-backend diff
//!   *stricter*, never weaker.  Tolerances only widen when a libm function is in the
//!   chain and the ULP bound is documented.

pub mod tolerance;
pub mod backend;
pub mod harness;
pub mod hard_cases;

pub use tolerance::ToleranceSpec;
pub use backend::{TamBackend, NullBackend};
pub use harness::{run_all_backends, assert_cross_backend_agreement, AgreementReport};

// ---------------------------------------------------------------------------
// Placeholder IR types
//
// The IR Architect (Peak 1) will provide real types from `tambear-tam-ir`.
// These placeholders let the harness compile, get reviewed, and accept
// plugged-in backends TODAY without waiting for Peak 1 to finish.
//
// When `tambear-tam-ir` lands:
//   1. Replace these type aliases with `pub use tambear_tam_ir::{TamProgram, ...}`.
//   2. Update every backend impl to accept the real types.
//   3. No other changes needed — the trait signatures don't change.
// ---------------------------------------------------------------------------

/// A compiled `.tam` program ready for execution.
///
/// Placeholder until `tambear-tam-ir::Program` exists (Peak 1, campsite 1.2).
#[derive(Debug, Clone)]
pub struct TamProgram {
    /// Human-readable name for error messages and reports.
    pub name: String,
    /// Whether any transcendental ops are present.
    ///
    /// Drives tolerance selection: pure-arithmetic programs require `bit_exact`;
    /// programs with transcendentals may use `within_ulp`.
    pub has_transcendentals: bool,
}

impl TamProgram {
    pub fn pure(name: impl Into<String>) -> Self {
        Self { name: name.into(), has_transcendentals: false }
    }

    pub fn with_transcendentals(name: impl Into<String>) -> Self {
        Self { name: name.into(), has_transcendentals: true }
    }
}

/// Inputs to a kernel — one or more named f64 buffers.
///
/// Placeholder until `tambear-tam-ir::Inputs` exists.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub buffers: Vec<(String, Vec<f64>)>,
}

impl Inputs {
    pub fn new() -> Self { Self::default() }

    pub fn with_buf(mut self, name: impl Into<String>, data: Vec<f64>) -> Self {
        self.buffers.push((name.into(), data));
        self
    }
}

/// Outputs from a kernel — named f64 result slots.
///
/// Placeholder until `tambear-tam-ir::Outputs` exists.
#[derive(Debug, Clone)]
pub struct Outputs {
    pub slots: Vec<f64>,
}

impl Outputs {
    pub fn from_slots(slots: Vec<f64>) -> Self { Self { slots } }
    pub fn empty() -> Self { Self { slots: vec![] } }
}
