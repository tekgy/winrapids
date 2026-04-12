//! `CpuInterpreterBackend` — wires `tambear-tam-ir`'s CPU interpreter into
//! the harness `TamBackend` trait.
//!
//! This is the first real backend to join the harness (campsite 4.6).
//! It executes `.tam` programs using the reference CPU interpreter from Peak 1.
//!
//! ## How it works
//!
//! The interpreter requires named buffers: `[("data0", &mut data), ("out", &mut out)]`.
//! `CpuInterpreterBackend::run` maps the `Inputs` named buffers to that interface,
//! allocates the output buffer (using the same slot-scanning heuristic as the CLI),
//! runs the first kernel in the program, and returns the output slots.
//!
//! ## Invariants enforced
//!
//! - I3/I4: inherited from the interpreter (no FMA, program-order execution).
//! - I1/I8: transcendental stubs panic — any test that reaches a `tam_exp` etc.
//!   will fail until `tambear-libm` is wired in at campsite 5.2.

use crate::{TamBackend, TamProgram, Inputs, Outputs};
use tambear_tam_ir::ast::{Op, Stmt};
use tambear_tam_ir::interp::Interpreter;

/// The CPU interpreter backend.
///
/// Implements `TamBackend` using `tambear_tam_ir::interp::Interpreter`.
/// Always available (no hardware requirement).
pub struct CpuInterpreterBackend;

impl CpuInterpreterBackend {
    pub fn new() -> Self { CpuInterpreterBackend }
}

impl Default for CpuInterpreterBackend {
    fn default() -> Self { Self::new() }
}

impl TamBackend for CpuInterpreterBackend {
    fn name(&self) -> String { "cpu-interpreter".into() }

    fn run(&self, program: &TamProgram, inputs: &Inputs) -> Outputs {
        let ir = match &program.ir {
            Some(p) => p,
            None => panic!("CpuInterpreterBackend requires TamProgram::from_ir — use TamProgram::from_ir or TamProgram::from_source"),
        };

        if ir.kernels.is_empty() {
            return Outputs::empty();
        }

        let kernel = &ir.kernels[0];
        let out_size = output_slot_count(kernel);
        let out = vec![0.0_f64; out_size];

        // Build the named buffer list: all inputs from Inputs, plus "out".
        // The Inputs may contain multiple named buffers (data0, data1, ...).
        let mut owned_bufs: Vec<(String, Vec<f64>)> = inputs.buffers.clone();
        owned_bufs.push(("out".into(), out.clone()));

        // Convert to the form the interpreter expects: Vec<(&str, &mut Vec<f64>)>.
        // We need to re-borrow from `owned_bufs` after the out slot is tracked.
        let mut buf_storage: Vec<(String, Vec<f64>)> = owned_bufs;

        // Run via the interpreter.
        let interp = Interpreter::new(ir);
        {
            let mut bufs_ref: Vec<(&str, &mut Vec<f64>)> = buf_storage
                .iter_mut()
                .map(|(name, data)| (name.as_str(), data))
                .collect();
            interp.run_kernel(&kernel.name, &mut bufs_ref)
                .unwrap_or_else(|e| panic!("cpu-interpreter: {e}"));
        }

        // Extract the output buffer.
        let result_slots = buf_storage.into_iter()
            .find(|(name, _)| name == "out")
            .map(|(_, data)| data)
            .unwrap_or_default();

        Outputs::from_slots(result_slots)
    }

    fn is_available(&self) -> bool { true }
}

/// Determine the output slot count by scanning the kernel for ReduceBlockAdd ops.
/// Returns max slot index + 1, or 16 as a fallback (same logic as the CLI).
fn output_slot_count(kernel: &tambear_tam_ir::ast::KernelDef) -> usize {
    let mut const_i32: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut max_slot: Option<i32> = None;

    for stmt in &kernel.body {
        match stmt {
            Stmt::Op(Op::ConstI32 { dst, value }) => {
                const_i32.insert(dst.name.clone(), *value);
            }
            Stmt::Op(Op::ReduceBlockAdd { slot_idx, .. }) => {
                if let Some(&slot) = const_i32.get(&slot_idx.name) {
                    max_slot = Some(match max_slot {
                        None => slot,
                        Some(prev) => prev.max(slot),
                    });
                }
            }
            _ => {}
        }
    }

    match max_slot {
        Some(s) if s >= 0 => (s as usize) + 1,
        _ => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToleranceSpec;
    use crate::backend::{BackendRegistry, NullBackend};
    use crate::harness::{run_all_backends, compare_backends};

    fn sum_all_add_program() -> TamProgram {
        TamProgram::from_source(include_str!(
            "../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/sum_all_add.tam"
        )).expect("sum_all_add.tam should parse and verify")
    }

    fn variance_pass_program() -> TamProgram {
        TamProgram::from_source(include_str!(
            "../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/variance_pass.tam"
        )).expect("variance_pass.tam should parse and verify")
    }

    #[test]
    fn cpu_backend_sum_all_add_empty() {
        let backend = CpuInterpreterBackend::new();
        let program = sum_all_add_program();
        let inputs = Inputs::new().with_buf("data", vec![]);
        let out = backend.run(&program, &inputs);
        assert_eq!(out.slots[0], 0.0);
    }

    #[test]
    fn cpu_backend_sum_all_add_1_to_10() {
        let backend = CpuInterpreterBackend::new();
        let program = sum_all_add_program();
        let inputs = Inputs::new().with_buf("data", (1..=10).map(|x| x as f64).collect());
        let out = backend.run(&program, &inputs);
        assert_eq!(out.slots[0], 55.0);
    }

    #[test]
    fn cpu_backend_variance_pass() {
        let backend = CpuInterpreterBackend::new();
        let program = variance_pass_program();
        let inputs = Inputs::new()
            .with_buf("data", vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let out = backend.run(&program, &inputs);
        assert_eq!(out.slots[0], 40.0,  "sum");
        assert_eq!(out.slots[1], 232.0, "sum_sq");
        assert_eq!(out.slots[2], 8.0,   "count");
    }

    #[test]
    fn cpu_backend_is_always_available() {
        assert!(CpuInterpreterBackend::new().is_available());
    }

    #[test]
    fn cpu_backend_name() {
        assert_eq!(CpuInterpreterBackend::new().name(), "cpu-interpreter");
    }

    /// Campsite 4.6 acceptance test: CPU interpreter backend registered in the
    /// harness, runs sum_all_add, produces bit-exact results compared against
    /// itself (single-backend smoke; real cross-backend diff arrives in Peak 3/7).
    #[test]
    fn harness_with_cpu_backend_bit_exact_self() {
        let mut registry = BackendRegistry::new();
        registry.push(Box::new(CpuInterpreterBackend::new()));
        let program = sum_all_add_program();
        let inputs = Inputs::new().with_buf("data", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let results = run_all_backends(&registry, &program, &inputs);
        // Single backend — nothing to diff, but harness must not panic.
        assert_eq!(results.len(), 1);
        assert!(results[0].outputs.is_some());
        assert_eq!(results[0].outputs.as_ref().unwrap().slots[0], 15.0);
    }

    /// Dummy second backend (NullBackend) disagrees with CPU interpreter for
    /// non-zero inputs — confirms the harness detects the disagreement.
    #[test]
    fn harness_detects_cpu_vs_null_disagreement() {
        let mut registry = BackendRegistry::new();
        registry.push(Box::new(CpuInterpreterBackend::new()));
        registry.push(Box::new(NullBackend::new(1)));
        let program = sum_all_add_program();
        let inputs = Inputs::new().with_buf("data", vec![1.0, 2.0, 3.0]);
        let results = run_all_backends(&registry, &program, &inputs);
        let report = compare_backends(&results, "sum_all_add", ToleranceSpec::bit_exact());
        // CPU gives 6.0; NullBackend gives 0.0 — must disagree.
        assert!(!report.all_agree(), "CPU vs Null should disagree on non-zero inputs");
    }
}
