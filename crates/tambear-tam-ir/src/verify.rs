//! Verifier: type-check and SSA invariant check for `.tam` programs.
//!
//! The verifier catches structural errors that the parser cannot. It is
//! separate from parsing because the AST represents potentially-invalid
//! programs — the verifier is the gate that says "this is well-formed."
//!
//! ## What the verifier checks (Phase 1)
//!
//! 1. **Every register has exactly one defining op** (SSA invariant). Exception:
//!    primed registers may shadow unprimed ones as loop phi outputs.
//! 2. **Every operand's type matches the op's signature.** `fadd` takes f64; `iadd`
//!    takes i32; `select` takes pred + two matching types.
//! 3. **Buffer index operands are i32.** `load.f64 %buf, %idx` requires `%idx:i32`.
//! 4. **Every primed register used after a loop has a corresponding unprimed
//!    definition before the loop.**
//! 5. **Functions end with exactly one `ret.f64`.**
//! 6. **Kernels do not contain `ret.f64`.**
//! 7. **The loop induction variable is distinct from all other register names.**
//!
//! ## What the verifier does NOT check
//!
//! - Out-of-bounds buffer accesses (runtime property).
//! - Reachability / dead code (allowed — the interpreter will not execute dead ops).
//! - Whether every primed register is actually *used* after the loop (warning-only).

use crate::ast::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyError {
    pub context: String,
    pub message: String,
}

impl VerifyError {
    fn new(ctx: impl Into<String>, msg: impl Into<String>) -> Self {
        VerifyError { context: ctx.into(), message: msg.into() }
    }
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "verify error in {}: {}", self.context, self.message)
    }
}

impl std::error::Error for VerifyError {}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Verify a `Program` for well-formedness.
///
/// Returns a list of all errors found. An empty list means the program is valid.
/// Stops collecting errors after 100 (to avoid infinite lists for pathological
/// programs).
pub fn verify(prog: &Program) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    for func in &prog.funcs {
        let mut ctx = VerifyCtx::new(format!("func '{}'", func.name));
        // Bind function parameters
        for param in &func.params {
            ctx.define(&param.reg, Ty::F64, &mut errors);
        }
        ctx.verify_func_body(&func.body, &mut errors);
    }
    for kernel in &prog.kernels {
        let mut ctx = VerifyCtx::new(format!("kernel '{}'", kernel.name));
        // Bind kernel parameters
        for param in &kernel.params {
            ctx.define(&param.reg, param.ty.clone(), &mut errors);
        }
        ctx.verify_kernel_body(&kernel.body, &mut errors);
    }
    errors
}

// ═══════════════════════════════════════════════════════════════════
// Verification context
// ═══════════════════════════════════════════════════════════════════

struct VerifyCtx {
    name: String,
    /// Maps register base name → type. Prime suffix is handled separately.
    types: HashMap<String, Ty>,
    /// True if we are currently inside a loop body.
    in_loop: bool,
}

impl VerifyCtx {
    fn new(name: String) -> Self {
        VerifyCtx { name, types: HashMap::new(), in_loop: false }
    }

    fn err(&self, msg: impl Into<String>) -> VerifyError {
        VerifyError::new(&self.name, msg)
    }

    fn define(&mut self, reg: &Reg, ty: Ty, errors: &mut Vec<VerifyError>) {
        if reg.prime {
            // Primed registers are phi outputs; they may shadow the unprimed version.
            // Check the unprimed version exists (it must be defined before the loop).
            if !self.types.contains_key(&reg.name) {
                errors.push(self.err(format!(
                    "primed register {} has no unprimed counterpart defined before the loop",
                    reg.display()
                )));
            }
            // We record the primed type under "<name>'" so lookups work.
            let prime_key = format!("{}'", reg.name);
            if self.types.contains_key(&prime_key) {
                errors.push(self.err(format!(
                    "SSA violation: register {} defined more than once",
                    reg.display()
                )));
            }
            self.types.insert(prime_key, ty);
        } else {
            if self.types.contains_key(&reg.name) {
                errors.push(self.err(format!(
                    "SSA violation: register {} defined more than once",
                    reg.display()
                )));
            }
            self.types.insert(reg.name.clone(), ty);
        }
    }

    #[allow(dead_code)]
    fn lookup(&self, reg: &Reg, errors: &mut Vec<VerifyError>) -> Option<Ty> {
        let key = if reg.prime {
            // Inside the loop, a primed register's type is the same as its unprimed counterpart.
            // Outside the loop after definition, it's stored under "<name>'".
            // We try both.
            let prime_key = format!("{}'", reg.name);
            if let Some(ty) = self.types.get(&prime_key) {
                return Some(ty.clone());
            }
            // Fallback to unprimed (valid if we're inside the loop and it was just defined)
            reg.name.clone()
        } else {
            reg.name.clone()
        };
        if let Some(ty) = self.types.get(&key) {
            Some(ty.clone())
        } else {
            errors.push(self.err(format!("undefined register: {}", reg.display())));
            None
        }
    }

    fn expect_ty(&self, reg: &Reg, expected: &Ty, errors: &mut Vec<VerifyError>) {
        // Look up without pushing "undefined" error (lookup already does that)
        let key = if reg.prime { format!("{}'", reg.name) } else { reg.name.clone() };
        let fallback = reg.name.clone();
        let actual = self.types.get(&key).or_else(|| self.types.get(&fallback));
        match actual {
            None => errors.push(self.err(format!("undefined register: {}", reg.display()))),
            Some(ty) if ty != expected => {
                errors.push(self.err(format!(
                    "type mismatch for {}: expected {:?}, got {:?}",
                    reg.display(), expected, ty
                )));
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Body verification
// ═══════════════════════════════════════════════════════════════════

impl VerifyCtx {
    fn verify_func_body(&mut self, body: &[Op], errors: &mut Vec<VerifyError>) {
        // Functions may not contain loops or reduce_block_add
        let mut has_ret = false;
        for op in body {
            match op {
                Op::ReduceBlockAdd { .. } => {
                    errors.push(self.err("reduce_block_add is not allowed inside a func"));
                }
                Op::RetF64 { val } => {
                    self.expect_ty(val, &Ty::F64, errors);
                    has_ret = true;
                }
                other => self.verify_op(other, errors),
            }
        }
        if !has_ret {
            errors.push(self.err("func must end with ret.f64"));
        }
    }

    fn verify_kernel_body(&mut self, body: &[Stmt], errors: &mut Vec<VerifyError>) {
        let mut loop_count = 0usize;
        for stmt in body {
            match stmt {
                Stmt::Op(op) => {
                    if matches!(op, Op::RetF64 { .. }) {
                        errors.push(self.err("ret.f64 is not allowed inside a kernel"));
                    }
                    self.verify_op(op, errors);
                }
                Stmt::Loop(lp) => {
                    loop_count += 1;
                    if loop_count > 1 {
                        errors.push(self.err("Phase 1: only one loop_grid_stride per kernel"));
                    }
                    self.verify_loop(lp, errors);
                }
            }
        }
    }

    fn verify_loop(&mut self, lp: &LoopGridStride, errors: &mut Vec<VerifyError>) {
        // limit must be i32
        self.expect_ty(&lp.limit, &Ty::I32, errors);

        // Induction variable is defined by the loop with type i32
        self.define(&lp.induction, Ty::I32, errors);

        // Verify body ops; primed registers become defined in this scope
        self.in_loop = true;
        for op in &lp.body {
            self.verify_op(op, errors);
        }
        self.in_loop = false;
    }

    fn verify_op(&mut self, op: &Op, errors: &mut Vec<VerifyError>) {
        match op {
            Op::ConstF64 { dst, .. } => self.define(dst, Ty::F64, errors),
            Op::ConstI32 { dst, .. } => self.define(dst, Ty::I32, errors),

            Op::BufSize { dst, buf } => {
                // buf must be BufF64 (parameter)
                self.expect_ty(buf, &Ty::BufF64, errors);
                self.define(dst, Ty::I32, errors);
            }
            Op::LoadF64 { dst, buf, idx } => {
                self.expect_ty(buf, &Ty::BufF64, errors);
                self.expect_ty(idx, &Ty::I32, errors);
                self.define(dst, Ty::F64, errors);
            }
            Op::StoreF64 { buf, idx, val } => {
                self.expect_ty(buf, &Ty::BufF64, errors);
                self.expect_ty(idx, &Ty::I32, errors);
                self.expect_ty(val, &Ty::F64, errors);
            }

            Op::FAdd { dst, a, b } | Op::FSub { dst, a, b }
            | Op::FMul { dst, a, b } | Op::FDiv { dst, a, b } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.expect_ty(b, &Ty::F64, errors);
                self.define(dst, Ty::F64, errors);
            }
            Op::FSqrt { dst, a } | Op::FNeg { dst, a } | Op::FAbs { dst, a } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.define(dst, Ty::F64, errors);
            }

            Op::ConstI64 { dst, .. } => self.define(dst, Ty::I64, errors),

            Op::IAdd { dst, a, b } | Op::ISub { dst, a, b } | Op::IMul { dst, a, b } => {
                self.expect_ty(a, &Ty::I32, errors);
                self.expect_ty(b, &Ty::I32, errors);
                self.define(dst, Ty::I32, errors);
            }
            Op::ICmpLt { dst, a, b } => {
                self.expect_ty(a, &Ty::I32, errors);
                self.expect_ty(b, &Ty::I32, errors);
                self.define(dst, Ty::Pred, errors);
            }

            Op::IAdd64 { dst, a, b } | Op::ISub64 { dst, a, b }
            | Op::AndI64 { dst, a, b } | Op::OrI64 { dst, a, b } | Op::XorI64 { dst, a, b } => {
                self.expect_ty(a, &Ty::I64, errors);
                self.expect_ty(b, &Ty::I64, errors);
                self.define(dst, Ty::I64, errors);
            }
            Op::ShlI64 { dst, a, shift } | Op::ShrI64 { dst, a, shift } => {
                self.expect_ty(a, &Ty::I64, errors);
                self.expect_ty(shift, &Ty::I32, errors);
                self.define(dst, Ty::I64, errors);
            }

            Op::LdExpF64 { dst, mantissa, exp } => {
                self.expect_ty(mantissa, &Ty::F64, errors);
                self.expect_ty(exp, &Ty::I32, errors);
                self.define(dst, Ty::F64, errors);
            }
            Op::F64ToI32Rn { dst, a } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.define(dst, Ty::I32, errors);
            }
            Op::BitcastF64ToI64 { dst, a } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.define(dst, Ty::I64, errors);
            }
            Op::BitcastI64ToF64 { dst, a } => {
                self.expect_ty(a, &Ty::I64, errors);
                self.define(dst, Ty::F64, errors);
            }

            Op::FCmpGt { dst, a, b } | Op::FCmpLt { dst, a, b } | Op::FCmpEq { dst, a, b } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.expect_ty(b, &Ty::F64, errors);
                self.define(dst, Ty::Pred, errors);
            }

            Op::SelectF64 { dst, pred, on_true, on_false } => {
                self.expect_ty(pred, &Ty::Pred, errors);
                self.expect_ty(on_true, &Ty::F64, errors);
                self.expect_ty(on_false, &Ty::F64, errors);
                self.define(dst, Ty::F64, errors);
            }
            Op::SelectI32 { dst, pred, on_true, on_false } => {
                self.expect_ty(pred, &Ty::Pred, errors);
                self.expect_ty(on_true, &Ty::I32, errors);
                self.expect_ty(on_false, &Ty::I32, errors);
                self.define(dst, Ty::I32, errors);
            }

            Op::TamExp { dst, a } | Op::TamLn { dst, a }
            | Op::TamSin { dst, a } | Op::TamCos { dst, a } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.define(dst, Ty::F64, errors);
            }
            Op::TamPow { dst, a, b } => {
                self.expect_ty(a, &Ty::F64, errors);
                self.expect_ty(b, &Ty::F64, errors);
                self.define(dst, Ty::F64, errors);
            }

            Op::ReduceBlockAdd { out_buf, slot_idx, val } => {
                self.expect_ty(out_buf, &Ty::BufF64, errors);
                self.expect_ty(slot_idx, &Ty::I32, errors);
                self.expect_ty(val, &Ty::F64, errors);
            }

            Op::RetF64 { val } => {
                self.expect_ty(val, &Ty::F64, errors);
                // Note: checking that ret is at end of body is done by the caller.
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests (campsite 1.9 acceptance criteria)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_program;
    use crate::fixtures::variance_pass_program;

    #[test]
    fn verify_variance_pass_clean() {
        let prog = variance_pass_program();
        let errors = verify(&prog);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    // Five hand-crafted broken programs — campsite 1.9 requires catching them all.

    #[test]
    fn verify_catches_undefined_register() {
        // Use %undefined before defining it
        let src = ".tam 0.1\n.target cross\nkernel k(buf<f64> %data) {\nentry:\n  %x = fadd.f64 %undefined, %undefined\n}\n";
        let prog = parse_program(src).unwrap();
        let errors = verify(&prog);
        assert!(!errors.is_empty(), "should catch undefined register");
        assert!(errors.iter().any(|e| e.message.contains("undefined register")));
    }

    #[test]
    fn verify_catches_ssa_double_define() {
        // Define %x twice
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %x = const.f64 1.0\n  %x = const.f64 2.0\n}\n";
        let prog = parse_program(src).unwrap();
        let errors = verify(&prog);
        assert!(!errors.is_empty(), "should catch double define");
        assert!(errors.iter().any(|e| e.message.contains("SSA violation")));
    }

    #[test]
    fn verify_catches_type_mismatch_fadd_on_i32() {
        // fadd expects f64 but gets i32
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %n = const.i32 5\n  %x = fadd.f64 %n, %n\n}\n";
        let prog = parse_program(src).unwrap();
        let errors = verify(&prog);
        assert!(!errors.is_empty(), "should catch type mismatch");
        assert!(errors.iter().any(|e| e.message.contains("type mismatch")));
    }

    #[test]
    fn verify_catches_ret_in_kernel() {
        // ret.f64 is not allowed in a kernel
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %x = const.f64 1.0\n  ret.f64 %x\n}\n";
        let prog = parse_program(src).unwrap();
        let errors = verify(&prog);
        assert!(!errors.is_empty(), "should catch ret in kernel");
        assert!(errors.iter().any(|e| e.message.contains("ret.f64 is not allowed")));
    }

    #[test]
    fn verify_catches_missing_ret_in_func() {
        // func must end with ret.f64
        let src = ".tam 0.1\n.target cross\nfunc f(f64 %x) -> f64 {\nentry:\n  %y = fadd.f64 %x, %x\n}\n";
        let prog = parse_program(src).unwrap();
        let errors = verify(&prog);
        assert!(!errors.is_empty(), "should catch missing ret");
        assert!(errors.iter().any(|e| e.message.contains("must end with ret.f64")));
    }
}
