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
pub mod cpu_backend;

pub use tolerance::ToleranceSpec;
pub use backend::{TamBackend, NullBackend};
pub use harness::{run_all_backends, assert_cross_backend_agreement, AgreementReport};
pub use cpu_backend::CpuInterpreterBackend;

// ---------------------------------------------------------------------------
// IR types — now backed by tambear-tam-ir (Peak 1 complete)
//
// `TamProgram` wraps the real `tambear_tam_ir::Program` so that:
// - `CpuInterpreterBackend` (and future backends) can execute it.
// - NullBackend smoke tests still compile unchanged (they use `TamProgram::pure`).
// - The `TamBackend::run` signature doesn't change.
// ---------------------------------------------------------------------------

/// A `.tam` program ready for harness execution.
///
/// Constructed from a `tambear_tam_ir::Program` via `TamProgram::from_ir`,
/// or as a placeholder via `TamProgram::pure` (for NullBackend smoke tests).
#[derive(Debug, Clone)]
pub struct TamProgram {
    /// Human-readable name for error messages and reports.
    pub name: String,
    /// Whether any transcendental ops are present.
    ///
    /// Drives tolerance selection: pure-arithmetic programs require `bit_exact`;
    /// programs with transcendentals may use `within_ulp`.
    pub has_transcendentals: bool,
    /// The real IR program.  `None` for NullBackend smoke tests only.
    pub ir: Option<tambear_tam_ir::ast::Program>,
}

impl TamProgram {
    /// Placeholder — no IR, for NullBackend smoke tests.
    pub fn pure(name: impl Into<String>) -> Self {
        Self { name: name.into(), has_transcendentals: false, ir: None }
    }

    /// Placeholder with transcendentals flag set.
    pub fn with_transcendentals(name: impl Into<String>) -> Self {
        Self { name: name.into(), has_transcendentals: true, ir: None }
    }

    /// Wrap a real IR program.  Transcendental detection is automatic.
    pub fn from_ir(prog: tambear_tam_ir::ast::Program) -> Self {
        use tambear_tam_ir::ast::{Op, Stmt};
        fn op_is_transcendental(op: &Op) -> bool {
            matches!(op,
                Op::TamExp { .. } | Op::TamLn { .. } | Op::TamSin { .. } |
                Op::TamCos { .. } | Op::TamPow { .. }
            )
        }
        let has_transcendentals = prog.kernels.iter().any(|k| {
            k.body.iter().any(|stmt| match stmt {
                Stmt::Op(op) => op_is_transcendental(op),
                Stmt::Loop(lp) => lp.body.iter().any(op_is_transcendental),
            })
        }) || prog.funcs.iter().any(|f| {
            f.body.iter().any(op_is_transcendental)
        });
        let name = prog.kernels.first().map(|k| k.name.clone()).unwrap_or_default();
        Self { name, has_transcendentals, ir: Some(prog) }
    }

    /// Parse a `.tam` source string and wrap it.
    ///
    /// Returns an error string if parsing or verification fails.
    pub fn from_source(src: &str) -> Result<Self, String> {
        let prog = tambear_tam_ir::parse::parse_program(src)
            .map_err(|e| format!("parse error: {e}"))?;
        let errors = tambear_tam_ir::verify::verify(&prog);
        if !errors.is_empty() {
            let msg: Vec<String> = errors.iter()
                .map(|e| format!("[{}] {}", e.context, e.message))
                .collect();
            return Err(format!("verify errors:\n{}", msg.join("\n")));
        }
        Ok(Self::from_ir(prog))
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
