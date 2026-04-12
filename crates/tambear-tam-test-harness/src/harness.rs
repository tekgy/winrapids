//! Cross-backend agreement assertion (campsite 4.3).
//!
//! The harness runs a program through every registered backend and compares
//! all pairwise results under a tolerance policy.  A single disagreement
//! produces a readable report — not just a test failure.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tambear_tam_test_harness::{
//!     TamProgram, Inputs, ToleranceSpec,
//!     run_all_backends, assert_cross_backend_agreement,
//! };
//! use tambear_tam_test_harness::backend::{BackendRegistry, NullBackend};
//!
//! let mut registry = BackendRegistry::new();
//! registry.push(Box::new(NullBackend::new(2)));
//!
//! let program = TamProgram::pure("sum");
//! let inputs  = Inputs::new().with_buf("x", vec![1.0, 2.0, 3.0]);
//!
//! let results = run_all_backends(&registry, &program, &inputs);
//! assert_cross_backend_agreement(&results, ToleranceSpec::bit_exact());
//! ```

use crate::{TamProgram, Inputs, Outputs, ToleranceSpec};
use crate::backend::BackendRegistry;

/// One backend's output for a specific program/input combination.
#[derive(Debug, Clone)]
pub struct BackendResult {
    /// Backend name (from [`TamBackend::name`]).
    pub backend_name: String,
    /// Output slots.  `None` if the backend was unavailable and skipped.
    pub outputs: Option<Outputs>,
}

impl BackendResult {
    pub fn available(name: impl Into<String>, outputs: Outputs) -> Self {
        Self { backend_name: name.into(), outputs: Some(outputs) }
    }

    pub fn skipped(name: impl Into<String>) -> Self {
        Self { backend_name: name.into(), outputs: None }
    }
}

/// Run `program` with `inputs` through every backend in the registry.
///
/// Backends that return `is_available() == false` produce a `skipped` result
/// rather than an error.  The caller (usually a test) decides whether to skip
/// or fail when too few backends are available.
pub fn run_all_backends(
    registry: &BackendRegistry,
    program: &TamProgram,
    inputs: &Inputs,
) -> Vec<BackendResult> {
    registry.iter().map(|b| {
        if !b.is_available() {
            BackendResult::skipped(b.name())
        } else {
            let out = b.run(program, inputs);
            BackendResult::available(b.name(), out)
        }
    }).collect()
}

/// Summary of a cross-backend comparison.
#[derive(Debug)]
pub struct AgreementReport {
    pub program_name: String,
    pub tolerance: ToleranceSpec,
    pub violations: Vec<ViolationEntry>,
    pub skipped_backends: Vec<String>,
    pub compared_backends: Vec<String>,
}

/// A single disagreement between two backends on one output slot.
#[derive(Debug)]
pub struct ViolationEntry {
    pub backend_a: String,
    pub backend_b: String,
    pub slot_index: usize,
    pub value_a: f64,
    pub value_b: f64,
    pub description: String,
}

impl AgreementReport {
    /// Returns true if all compared backends agreed on all slots.
    pub fn all_agree(&self) -> bool {
        self.violations.is_empty()
    }

    /// Produce a human-readable failure message.
    pub fn failure_message(&self) -> String {
        if self.all_agree() {
            return format!(
                "PASS: {} ({} backends agree, {} skipped)",
                self.program_name,
                self.compared_backends.len(),
                self.skipped_backends.len()
            );
        }
        let mut msg = format!(
            "FAIL: {} — {} violation(s) under {:?}\n",
            self.program_name,
            self.violations.len(),
            self.tolerance
        );
        for v in &self.violations {
            msg.push_str(&format!(
                "  slot[{}]: {} vs {}: {}\n",
                v.slot_index, v.backend_a, v.backend_b, v.description
            ));
        }
        if !self.skipped_backends.is_empty() {
            msg.push_str(&format!(
                "  skipped: {}\n",
                self.skipped_backends.join(", ")
            ));
        }
        msg
    }
}

/// Compare all pairs of available backends and return a report.
///
/// Does not panic — the caller decides whether to `assert!(report.all_agree())`.
/// This allows tests to print the full report before panicking.
pub fn compare_backends(
    results: &[BackendResult],
    program_name: &str,
    tolerance: ToleranceSpec,
) -> AgreementReport {
    let mut compared = Vec::new();
    let mut skipped = Vec::new();
    let mut violations = Vec::new();

    // Separate available from skipped
    let available: Vec<&BackendResult> = results.iter()
        .filter_map(|r| {
            if r.outputs.is_some() {
                compared.push(r.backend_name.clone());
                Some(r)
            } else {
                skipped.push(r.backend_name.clone());
                None
            }
        })
        .collect();

    // Compare the first available backend against all others.
    // This is transitive: if A==B and A==C, then B==C.
    if available.len() >= 2 {
        let reference = &available[0];
        let ref_slots = reference.outputs.as_ref().unwrap();

        for other in &available[1..] {
            let other_slots = other.outputs.as_ref().unwrap();

            // Dimension mismatch is always a hard error regardless of tolerance.
            if ref_slots.slots.len() != other_slots.slots.len() {
                violations.push(ViolationEntry {
                    backend_a: reference.backend_name.clone(),
                    backend_b: other.backend_name.clone(),
                    slot_index: 0,
                    value_a: ref_slots.slots.len() as f64,
                    value_b: other_slots.slots.len() as f64,
                    description: format!(
                        "output slot count mismatch: {} vs {}",
                        ref_slots.slots.len(), other_slots.slots.len()
                    ),
                });
                continue;
            }

            for (idx, (&va, &vb)) in ref_slots.slots.iter().zip(other_slots.slots.iter()).enumerate() {
                if let Some(desc) = tolerance.describe_violation(va, vb) {
                    violations.push(ViolationEntry {
                        backend_a: reference.backend_name.clone(),
                        backend_b: other.backend_name.clone(),
                        slot_index: idx,
                        value_a: va,
                        value_b: vb,
                        description: desc,
                    });
                }
            }
        }
    }

    AgreementReport {
        program_name: program_name.to_string(),
        tolerance,
        violations,
        skipped_backends: skipped,
        compared_backends: compared,
    }
}

/// Assert cross-backend agreement, panicking with a full report if any pair disagrees.
///
/// This is the main entry point for tests.
pub fn assert_cross_backend_agreement(results: &[BackendResult], tolerance: ToleranceSpec) {
    let program_name = "unknown";
    let report = compare_backends(results, program_name, tolerance);
    assert!(report.all_agree(), "\n{}", report.failure_message());
}

/// Assert cross-backend agreement with a named program for better error messages.
pub fn assert_cross_backend_agreement_named(
    results: &[BackendResult],
    program_name: &str,
    tolerance: ToleranceSpec,
) {
    let report = compare_backends(results, program_name, tolerance);
    assert!(report.all_agree(), "\n{}", report.failure_message());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendRegistry, NullBackend};

    fn two_null_results(slots: Vec<f64>) -> Vec<BackendResult> {
        vec![
            BackendResult::available("backend-a", Outputs::from_slots(slots.clone())),
            BackendResult::available("backend-b", Outputs::from_slots(slots)),
        ]
    }

    #[test]
    fn identical_backends_agree_bit_exact() {
        let results = two_null_results(vec![1.0, 2.0, 3.0]);
        let report = compare_backends(&results, "test", ToleranceSpec::bit_exact());
        assert!(report.all_agree());
    }

    #[test]
    fn differing_backends_disagree_bit_exact() {
        let mut results = two_null_results(vec![1.0]);
        if let Some(ref mut out) = results[1].outputs {
            out.slots[0] = f64::from_bits(1.0_f64.to_bits() + 1);
        }
        let report = compare_backends(&results, "test", ToleranceSpec::bit_exact());
        assert!(!report.all_agree());
        assert_eq!(report.violations.len(), 1);
    }

    #[test]
    fn within_ulp_tolerates_small_diff() {
        let x = 1.0_f64;
        let y = f64::from_bits(x.to_bits() + 2);
        let results = vec![
            BackendResult::available("a", Outputs::from_slots(vec![x])),
            BackendResult::available("b", Outputs::from_slots(vec![y])),
        ];
        let report = compare_backends(&results, "test", ToleranceSpec::within_ulp(3));
        assert!(report.all_agree());
    }

    #[test]
    fn within_ulp_fails_large_diff() {
        let x = 1.0_f64;
        let y = f64::from_bits(x.to_bits() + 5);
        let results = vec![
            BackendResult::available("a", Outputs::from_slots(vec![x])),
            BackendResult::available("b", Outputs::from_slots(vec![y])),
        ];
        let report = compare_backends(&results, "test", ToleranceSpec::within_ulp(3));
        assert!(!report.all_agree());
    }

    #[test]
    fn skipped_backend_does_not_affect_agreement() {
        let results = vec![
            BackendResult::available("a", Outputs::from_slots(vec![1.0])),
            BackendResult::skipped("b"),
            BackendResult::available("c", Outputs::from_slots(vec![1.0])),
        ];
        let report = compare_backends(&results, "test", ToleranceSpec::bit_exact());
        assert!(report.all_agree());
        assert_eq!(report.skipped_backends, vec!["b"]);
        assert_eq!(report.compared_backends, vec!["a", "c"]);
    }

    #[test]
    fn slot_count_mismatch_is_violation() {
        let results = vec![
            BackendResult::available("a", Outputs::from_slots(vec![1.0, 2.0])),
            BackendResult::available("b", Outputs::from_slots(vec![1.0])),
        ];
        let report = compare_backends(&results, "test", ToleranceSpec::bit_exact());
        assert!(!report.all_agree());
        assert!(report.violations[0].description.contains("slot count mismatch"));
    }

    #[test]
    fn run_all_backends_smoke() {
        let mut registry = BackendRegistry::new();
        registry.push(Box::new(NullBackend::new(3)));
        let program = TamProgram::pure("smoke");
        let inputs = Inputs::new().with_buf("x", vec![1.0, 2.0, 3.0]);
        let results = run_all_backends(&registry, &program, &inputs);
        assert_eq!(results.len(), 1);
        assert!(results[0].outputs.is_some());
    }

    #[test]
    fn two_null_backends_agree_bit_exact() {
        let mut registry = BackendRegistry::new();
        registry.push(Box::new(NullBackend::named("null-a", 3)));
        registry.push(Box::new(NullBackend::named("null-b", 3)));
        let program = TamProgram::pure("smoke");
        let inputs = Inputs::new().with_buf("x", vec![1.0, 2.0, 3.0]);
        let results = run_all_backends(&registry, &program, &inputs);
        assert_cross_backend_agreement(&results, ToleranceSpec::bit_exact());
    }
}
