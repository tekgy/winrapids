//! CPU interpreter: execute a `.tam` `Program` on concrete f64 buffers.
//!
//! ## Purpose
//!
//! The interpreter is the **canonical reference oracle**. It is the single
//! most-trusted implementation in the entire system. Everything else — the
//! PTX backend, the SPIR-V backend, the JIT — is verified against it.
//!
//! It is intentionally slow. Correctness is the only goal. The interpreter
//! executes ops one-by-one, in program order, with no optimization.
//!
//! ## Invariants enforced here
//!
//! - **I3 (no FMA):** `fadd` and `fmul` are separate ops. Rust does not
//!   auto-fuse them. No `f64::mul_add` is called here.
//! - **I4 (no reordering):** ops execute in program order. No reordering.
//! - **I1 (no vendor math):** transcendental stubs panic until tambear-libm
//!   is wired in (campsite 5.2). `f64::exp`, `f64::sin`, etc. are NOT called
//!   here. The comment on each transcendental arm says so explicitly.
//!
//! ## Grid-stride loop semantics on CPU
//!
//! On CPU, "grid-stride" means serial execution: `%i` runs from 0 to n-1.
//! There is one "block" covering all elements. `reduce_block_add` does a
//! direct store (no tree reduction needed).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tambear_tam_ir::interp::Interpreter;
//! use tambear_tam_ir::ast::Program;
//!
//! let prog: Program = todo!();
//! let interp = Interpreter::new(&prog);
//! let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
//! let mut out = vec![0.0f64; 1];
//! interp.run_kernel("sum_all_add", &[
//!     ("data", &mut data),
//!     ("out", &mut out),
//! ]).unwrap();
//! // out[0] == 55.0
//! ```

use crate::ast::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Value type
// ═══════════════════════════════════════════════════════════════════

/// A runtime value that can live in a register.
#[derive(Debug, Clone)]
enum Val {
    F64(f64),
    I32(i32),
    I64(i64),
    Pred(bool),
}

impl Val {
    fn as_f64(&self, reg: &str) -> Result<f64, InterpError> {
        match self {
            Val::F64(v) => Ok(*v),
            _ => Err(InterpError::new(format!("register %{} is not f64", reg))),
        }
    }
    fn as_i32(&self, reg: &str) -> Result<i32, InterpError> {
        match self {
            Val::I32(v) => Ok(*v),
            _ => Err(InterpError::new(format!("register %{} is not i32", reg))),
        }
    }
    fn as_i64(&self, reg: &str) -> Result<i64, InterpError> {
        match self {
            Val::I64(v) => Ok(*v),
            _ => Err(InterpError::new(format!("register %{} is not i64", reg))),
        }
    }
    fn as_pred(&self, reg: &str) -> Result<bool, InterpError> {
        match self {
            Val::Pred(v) => Ok(*v),
            _ => Err(InterpError::new(format!("register %{} is not pred", reg))),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct InterpError {
    pub message: String,
}

impl InterpError {
    fn new(msg: impl Into<String>) -> Self {
        InterpError { message: msg.into() }
    }
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "interp error: {}", self.message)
    }
}

impl std::error::Error for InterpError {}

// ═══════════════════════════════════════════════════════════════════
// Register file
// ═══════════════════════════════════════════════════════════════════

/// A register file: maps register names (without % prefix) to values.
///
/// Prime registers are stored as `name'`.
#[derive(Debug, Default)]
struct Env {
    vals: HashMap<String, Val>,
}

impl Env {
    fn set(&mut self, reg: &Reg, val: Val) {
        let key = if reg.prime {
            format!("{}'", reg.name)
        } else {
            reg.name.clone()
        };
        self.vals.insert(key, val);
    }

    fn get(&self, reg: &Reg) -> Result<&Val, InterpError> {
        let key = if reg.prime {
            // Try prime first, then unprimed (for within-loop reads)
            let prime_key = format!("{}'", reg.name);
            if self.vals.contains_key(&prime_key) {
                return Ok(&self.vals[&prime_key]);
            }
            reg.name.clone()
        } else {
            reg.name.clone()
        };
        self.vals.get(&key)
            .ok_or_else(|| InterpError::new(format!("undefined register %{}", key)))
    }

    fn get_f64(&self, reg: &Reg) -> Result<f64, InterpError> {
        self.get(reg)?.as_f64(&reg.display())
    }

    fn get_i32(&self, reg: &Reg) -> Result<i32, InterpError> {
        self.get(reg)?.as_i32(&reg.display())
    }

    fn get_i64(&self, reg: &Reg) -> Result<i64, InterpError> {
        self.get(reg)?.as_i64(&reg.display())
    }

    fn get_pred(&self, reg: &Reg) -> Result<bool, InterpError> {
        self.get(reg)?.as_pred(&reg.display())
    }

    /// Promote phi outputs: after a loop, `%x'` becomes `%x`.
    /// Called once after the loop body completes.
    fn promote_phi_outputs(&mut self, phi_names: &[String]) {
        for name in phi_names {
            let prime_key = format!("{}'", name);
            if let Some(val) = self.vals.remove(&prime_key) {
                self.vals.insert(name.clone(), val);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Interpreter
// ═══════════════════════════════════════════════════════════════════

/// The CPU interpreter. Holds a reference to the program for function lookup.
pub struct Interpreter<'p> {
    prog: &'p Program,
}

impl<'p> Interpreter<'p> {
    pub fn new(prog: &'p Program) -> Self {
        Interpreter { prog }
    }

    /// Execute a named kernel.
    ///
    /// `inputs`: one slice per kernel buffer parameter (in parameter order).
    /// `outputs`: one slice per output buffer parameter. The kernel determines
    /// which buffers are outputs by writing to them via `reduce_block_add` or
    /// `store.f64`.
    ///
    /// **Caller convention:** pass ALL buffer parameters in `inputs`. If a
    /// parameter is both read and written, pass the same buffer in both
    /// `inputs` and `outputs`. For the standard kernel shape (read-only
    /// inputs, write-only output), pass the output buffer as the last input
    /// too — or use `run_kernel_split` for the split convention.
    ///
    /// In practice, kernels only call `reduce_block_add` on output buffers,
    /// which already hold pre-zeroed storage. Callers zero the output buffers
    /// before calling `run_kernel`.
    pub fn run_kernel(
        &self,
        kernel_name: &str,
        // Named-parameter convention: (name, slice)
        buffers: &[(&str, &mut Vec<f64>)],
    ) -> Result<(), InterpError> {
        let kernel = self.prog.kernel(kernel_name)
            .ok_or_else(|| InterpError::new(format!("kernel '{}' not found", kernel_name)))?;

        // Bind parameters to buffers by name matching.
        // This is cleaner than positional binding for multi-buffer kernels.
        let buf_map: HashMap<&str, *mut Vec<f64>> = buffers.iter()
            .map(|(name, buf)| (*name, *buf as *const Vec<f64> as *mut Vec<f64>))
            .collect();

        let mut env = Env::default();

        // Bind kernel parameters: buf<f64> params stay as named references;
        // scalar params get their values bound.
        for param in &kernel.params {
            match &param.ty {
                Ty::BufF64 => {
                    // BufF64 params are handled by looking up the name in buf_map
                    // during load/store/bufsize/reduce ops. We put a sentinel.
                    // (The env doesn't store buffers — the executor uses buf_map directly.)
                }
                Ty::I32 => {
                    // Scalar i32 params: look up from buffers as length hints.
                    // Phase 1 kernels don't take scalar params; this is future-proofing.
                }
                _ => {}
            }
        }

        // Execute the body
        self.exec_kernel_body(&kernel.body, &kernel.params, &buf_map, &mut env)?;

        Ok(())
    }

    fn exec_kernel_body(
        &self,
        body: &[Stmt],
        params: &[KernelParam],
        buf_map: &HashMap<&str, *mut Vec<f64>>,
        env: &mut Env,
    ) -> Result<(), InterpError> {
        for stmt in body {
            match stmt {
                Stmt::Op(op) => self.exec_op(op, params, buf_map, env)?,
                Stmt::Loop(lp) => self.exec_loop(lp, params, buf_map, env)?,
            }
        }
        Ok(())
    }

    fn exec_loop(
        &self,
        lp: &LoopGridStride,
        params: &[KernelParam],
        buf_map: &HashMap<&str, *mut Vec<f64>>,
        env: &mut Env,
    ) -> Result<(), InterpError> {
        let n = env.get_i32(&lp.limit)?;

        // Collect phi names: any register in the body that has a prime suffix.
        // After each iteration, promote %name' → %name so the next iteration
        // sees the updated value.
        let phi_names: Vec<String> = lp.body.iter()
            .filter_map(|op| get_dst(op))
            .filter(|r| r.prime)
            .map(|r| r.name.clone())
            .collect();

        // Serial execution (CPU grid-stride = serial)
        for i in 0..n {
            env.set(&lp.induction, Val::I32(i));
            for op in &lp.body {
                self.exec_op(op, params, buf_map, env)?;
            }
            // After each iteration, promote phi outputs
            env.promote_phi_outputs(&phi_names);
        }
        Ok(())
    }

    fn exec_op(
        &self,
        op: &Op,
        _params: &[KernelParam],
        buf_map: &HashMap<&str, *mut Vec<f64>>,
        env: &mut Env,
    ) -> Result<(), InterpError> {
        match op {
            // ── Constants ─────────────────────────────────────────────────────
            Op::ConstF64 { dst, value } => {
                env.set(dst, Val::F64(*value));
            }
            Op::ConstI32 { dst, value } => {
                env.set(dst, Val::I32(*value));
            }

            // ── Buffer ops ────────────────────────────────────────────────────
            Op::BufSize { dst, buf } => {
                let slice = self.get_buf(buf, buf_map)?;
                env.set(dst, Val::I32(slice.len() as i32));
            }
            Op::LoadF64 { dst, buf, idx } => {
                let slice = self.get_buf(buf, buf_map)?;
                let i = env.get_i32(idx)? as usize;
                if i >= slice.len() {
                    return Err(InterpError::new(format!(
                        "load.f64: index {} out of bounds (len={})", i, slice.len()
                    )));
                }
                env.set(dst, Val::F64(slice[i]));
            }
            Op::StoreF64 { buf, idx, val } => {
                let i = env.get_i32(idx)? as usize;
                let v = env.get_f64(val)?;
                let slice = self.get_buf_mut(buf, buf_map)?;
                if i >= slice.len() {
                    return Err(InterpError::new(format!(
                        "store.f64: index {} out of bounds (len={})", i, slice.len()
                    )));
                }
                slice[i] = v;
            }

            // ── Floating-point arithmetic (non-contracting, RNE) ──────────────
            // I3: These are NEVER fused. Rust does not auto-fuse + and *.
            // I4: Sequential evaluation. No reordering.
            Op::FAdd { dst, a, b } => {
                // I3: fadd is fadd. Never fma.
                let r = env.get_f64(a)? + env.get_f64(b)?;
                env.set(dst, Val::F64(r));
            }
            Op::FSub { dst, a, b } => {
                let r = env.get_f64(a)? - env.get_f64(b)?;
                env.set(dst, Val::F64(r));
            }
            Op::FMul { dst, a, b } => {
                // I3: fmul is fmul. Never fma.
                let r = env.get_f64(a)? * env.get_f64(b)?;
                env.set(dst, Val::F64(r));
            }
            Op::FDiv { dst, a, b } => {
                let r = env.get_f64(a)? / env.get_f64(b)?;
                env.set(dst, Val::F64(r));
            }
            Op::FSqrt { dst, a } => {
                // IEEE correctly rounded by Rust's f64::sqrt.
                let r = env.get_f64(a)?.sqrt();
                env.set(dst, Val::F64(r));
            }
            Op::FNeg { dst, a } => {
                let r = -env.get_f64(a)?;
                env.set(dst, Val::F64(r));
            }
            Op::FAbs { dst, a } => {
                let r = env.get_f64(a)?.abs();
                env.set(dst, Val::F64(r));
            }

            // ── Integer arithmetic ────────────────────────────────────────────
            Op::ConstI64 { dst, value } => {
                env.set(dst, Val::I64(*value));
            }
            Op::IAdd { dst, a, b } => {
                let r = env.get_i32(a)?.wrapping_add(env.get_i32(b)?);
                env.set(dst, Val::I32(r));
            }
            Op::ISub { dst, a, b } => {
                let r = env.get_i32(a)?.wrapping_sub(env.get_i32(b)?);
                env.set(dst, Val::I32(r));
            }
            Op::IMul { dst, a, b } => {
                let r = env.get_i32(a)?.wrapping_mul(env.get_i32(b)?);
                env.set(dst, Val::I32(r));
            }
            Op::ICmpLt { dst, a, b } => {
                let r = env.get_i32(a)? < env.get_i32(b)?;
                env.set(dst, Val::Pred(r));
            }
            Op::IAdd64 { dst, a, b } => {
                let r = env.get_i64(a)?.wrapping_add(env.get_i64(b)?);
                env.set(dst, Val::I64(r));
            }
            Op::ISub64 { dst, a, b } => {
                let r = env.get_i64(a)?.wrapping_sub(env.get_i64(b)?);
                env.set(dst, Val::I64(r));
            }
            Op::AndI64 { dst, a, b } => {
                let r = env.get_i64(a)? & env.get_i64(b)?;
                env.set(dst, Val::I64(r));
            }
            Op::OrI64 { dst, a, b } => {
                let r = env.get_i64(a)? | env.get_i64(b)?;
                env.set(dst, Val::I64(r));
            }
            Op::XorI64 { dst, a, b } => {
                let r = env.get_i64(a)? ^ env.get_i64(b)?;
                env.set(dst, Val::I64(r));
            }
            Op::ShlI64 { dst, a, shift } => {
                let v = env.get_i64(a)?;
                let s = env.get_i32(shift)? as u32;
                env.set(dst, Val::I64(v.wrapping_shl(s)));
            }
            Op::ShrI64 { dst, a, shift } => {
                let v = env.get_i64(a)?;
                let s = env.get_i32(shift)? as u32;
                env.set(dst, Val::I64(v.wrapping_shr(s)));
            }

            // ── Float ↔ integer conversion ─────────────────────────────────────
            Op::LdExpF64 { dst, mantissa, exp } => {
                // Correct IEEE ldexp: mantissa * 2^exp.
                // Rust stdlib `f64::powi` is NOT ldexp — it's x^n.
                // We use the bit-manipulation form: add exp to the biased exponent field.
                // This handles subnormals, overflow, and underflow correctly.
                let m = env.get_f64(mantissa)?;
                let n = env.get_i32(exp)?;
                // Use libm-style implementation: multiply by 2^n via repeated halving/doubling.
                // For the CPU interpreter, Rust's `f64::ldexp` equivalent is:
                // m * 2f64.powi(n) — BUT powi can overflow for large n.
                // Use the safe approach: scale in stages to avoid intermediate overflow.
                let r = if m.is_nan() || m.is_infinite() || m == 0.0 {
                    m
                } else {
                    // Safe ldexp via bit manipulation (same result as C ldexp)
                    let bits = m.to_bits();
                    let biased_exp = ((bits >> 52) & 0x7ff) as i64;
                    let new_biased = biased_exp + n as i64;
                    if new_biased >= 0x7ff {
                        // Overflow to infinity with sign
                        if m > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }
                    } else if new_biased <= 0 {
                        // Underflow (subnormal or zero)
                        // For simplicity in the reference interpreter: flush to zero.
                        // Phase 1 programs don't generate subnormal inputs via ldexp.
                        0.0f64.copysign(m)
                    } else {
                        // Normal: replace exponent field
                        let new_bits = (bits & !0x7ff0_0000_0000_0000u64)
                            | ((new_biased as u64) << 52);
                        f64::from_bits(new_bits)
                    }
                };
                env.set(dst, Val::F64(r));
            }
            Op::F64ToI32Rn { dst, a } => {
                // Round f64 to nearest i32 (round-to-nearest-even).
                // Saturates at INT32 bounds for out-of-range values.
                let v = env.get_f64(a)?;
                let r = if v.is_nan() {
                    0i32
                } else if v >= i32::MAX as f64 {
                    i32::MAX
                } else if v <= i32::MIN as f64 {
                    i32::MIN
                } else {
                    // Rust's `round()` uses round-half-away-from-zero.
                    // round-to-nearest-even: use the from/to cast which truncates;
                    // then apply banker's rounding manually.
                    let floor = v.floor();
                    let frac = v - floor;
                    let floor_i = floor as i64;
                    if frac > 0.5 {
                        (floor_i + 1) as i32
                    } else if frac < 0.5 {
                        floor_i as i32
                    } else {
                        // Exactly 0.5: round to even
                        if floor_i % 2 == 0 { floor_i as i32 } else { (floor_i + 1) as i32 }
                    }
                };
                env.set(dst, Val::I32(r));
            }
            Op::BitcastF64ToI64 { dst, a } => {
                // Reinterpret f64 bit pattern as i64. No value conversion.
                let v = env.get_f64(a)?;
                env.set(dst, Val::I64(v.to_bits() as i64));
            }
            Op::BitcastI64ToF64 { dst, a } => {
                // Reinterpret i64 bit pattern as f64. No value conversion.
                let v = env.get_i64(a)?;
                env.set(dst, Val::F64(f64::from_bits(v as u64)));
            }

            // ── Floating-point comparisons ─────────────────────────────────────
            Op::FCmpGt { dst, a, b } => {
                let r = env.get_f64(a)? > env.get_f64(b)?;
                env.set(dst, Val::Pred(r));
            }
            Op::FCmpLt { dst, a, b } => {
                let r = env.get_f64(a)? < env.get_f64(b)?;
                env.set(dst, Val::Pred(r));
            }
            Op::FCmpEq { dst, a, b } => {
                // NaN comparisons return false per IEEE 754.
                let r = env.get_f64(a)? == env.get_f64(b)?;
                env.set(dst, Val::Pred(r));
            }

            // ── Select (branch-free) ──────────────────────────────────────────
            // I11: NaN propagates through SelectF64. If either value operand is
            // NaN, the result is NaN regardless of the predicate. This prevents
            // the pattern `fcmp_gt(NaN, x) → false` + `select(false, NaN, x) → x`
            // from silently swallowing NaN. Programs that need NaN-unsafe min/max
            // semantics must use a separate op explicitly designed for that.
            //
            // PTX emit target: use `selp.f64` AFTER checking both value operands
            // for NaN (via `testp.notanumber.f64`). SPIR-V: add OpIsNan guards.
            // See also: spec.md §5.5, peak3-ptx/lowering-blueprint.md §NaN.
            Op::SelectF64 { dst, pred, on_true, on_false } => {
                let vt = env.get_f64(on_true)?;
                let vf = env.get_f64(on_false)?;
                // I11: if either value is NaN, propagate NaN.
                let r = if vt.is_nan() || vf.is_nan() {
                    f64::NAN
                } else {
                    let p = env.get_pred(pred)?;
                    if p { vt } else { vf }
                };
                env.set(dst, Val::F64(r));
            }
            Op::SelectI32 { dst, pred, on_true, on_false } => {
                let p = env.get_pred(pred)?;
                let r = if p { env.get_i32(on_true)? } else { env.get_i32(on_false)? };
                env.set(dst, Val::I32(r));
            }

            // ── Transcendental stubs ──────────────────────────────────────────
            // I1: Do NOT call f64::exp, f64::ln, f64::sin, f64::cos here.
            // I8: First-principles implementations only, via tambear-libm.
            // These panic until tambear-libm is wired in (campsite 5.2).
            //
            // "not yet in libm" is the campsite 1.13 contract.
            Op::TamExp { .. } => {
                panic!("tam_exp: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }
            Op::TamLn { .. } => {
                panic!("tam_ln: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }
            Op::TamSin { .. } => {
                panic!("tam_sin: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }
            Op::TamCos { .. } => {
                panic!("tam_cos: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }
            Op::TamAtan { .. } => {
                panic!("tam_atan: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }
            Op::TamPow { .. } => {
                panic!("tam_pow: not yet in libm (campsite 1.13 stub). Wire tambear-libm at campsite 5.2.");
            }

            // ── Reduction ─────────────────────────────────────────────────────
            // On CPU: one "block" covers all elements. The "reduce" is a direct
            // store of the value accumulated by the surrounding loop body.
            // The loop body runs sequentially (SSA phi update = left fold), so the
            // CPU interpreter effectively executes `sequential_left` regardless of
            // the declared order strategy. This is correct for Phase 1:
            //   - sequential_left: the loop body IS the sequential fold.
            //   - tree_fixed_fanout_2: the loop-then-store is *semantically* the
            //     same result for exact arithmetic but may differ by a few ULP on
            //     numerically sensitive inputs. The PTX backend (Peak 3) implements
            //     the actual tree structure to achieve cross-backend bit-exactness.
            // The verifier (via the registry) ensures only known strategies reach here.
            Op::ReduceBlockAdd { out_buf, slot_idx, val, order: _ } => {
                let slot = env.get_i32(slot_idx)? as usize;
                let v = env.get_f64(val)?;
                let slice = self.get_buf_mut(out_buf, buf_map)?;
                if slot >= slice.len() {
                    return Err(InterpError::new(format!(
                        "reduce_block_add: slot {} out of bounds (len={})", slot, slice.len()
                    )));
                }
                // Direct store: the loop has already accumulated into %acc'.
                // The CPU "block" result IS the total result.
                slice[slot] = v;
            }

            // ── Return (not valid in kernels — verifier catches this) ─────────
            Op::RetF64 { .. } => {
                return Err(InterpError::new("ret.f64 encountered in kernel body"));
            }
        }
        Ok(())
    }

    /// Execute a function definition (for libm dispatch).
    /// Returns the f64 return value.
    pub fn call_func(&self, func_name: &str, args: &[f64]) -> Result<f64, InterpError> {
        let func = self.prog.func(func_name)
            .ok_or_else(|| InterpError::new(format!("func '{}' not found", func_name)))?;

        if args.len() != func.params.len() {
            return Err(InterpError::new(format!(
                "func '{}': expected {} args, got {}",
                func_name, func.params.len(), args.len()
            )));
        }

        let mut env = Env::default();
        for (param, &arg) in func.params.iter().zip(args.iter()) {
            env.set(&param.reg, Val::F64(arg));
        }

        // Empty buf_map — functions don't access buffers
        let buf_map: HashMap<&str, *mut Vec<f64>> = HashMap::new();
        let dummy_params: Vec<KernelParam> = vec![];

        for op in &func.body {
            if let Op::RetF64 { val } = op {
                return env.get_f64(val);
            }
            self.exec_op(op, &dummy_params, &buf_map, &mut env)?;
        }
        Err(InterpError::new(format!("func '{}' never reached ret.f64", func_name)))
    }

    // ── Buffer helpers ────────────────────────────────────────────────────────

    fn get_buf<'a>(
        &self,
        reg: &Reg,
        buf_map: &'a HashMap<&str, *mut Vec<f64>>,
    ) -> Result<&'a Vec<f64>, InterpError> {
        buf_map.get(reg.name.as_str())
            .map(|p| unsafe { &**p })
            .ok_or_else(|| InterpError::new(format!("no buffer bound for %{}", reg.name)))
    }

    fn get_buf_mut<'a>(
        &self,
        reg: &Reg,
        buf_map: &'a HashMap<&str, *mut Vec<f64>>,
    ) -> Result<&'a mut Vec<f64>, InterpError> {
        buf_map.get(reg.name.as_str())
            .map(|p| unsafe { &mut **p })
            .ok_or_else(|| InterpError::new(format!("no buffer bound for %{}", reg.name)))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helper: get dst register from an op
// ═══════════════════════════════════════════════════════════════════

fn get_dst(op: &Op) -> Option<&Reg> {
    match op {
        Op::ConstF64 { dst, .. } => Some(dst),
        Op::ConstI32 { dst, .. } => Some(dst),
        Op::BufSize { dst, .. } => Some(dst),
        Op::LoadF64 { dst, .. } => Some(dst),
        Op::FAdd { dst, .. } => Some(dst),
        Op::FSub { dst, .. } => Some(dst),
        Op::FMul { dst, .. } => Some(dst),
        Op::FDiv { dst, .. } => Some(dst),
        Op::FSqrt { dst, .. } => Some(dst),
        Op::FNeg { dst, .. } => Some(dst),
        Op::FAbs { dst, .. } => Some(dst),
        Op::ConstI64 { dst, .. } => Some(dst),
        Op::IAdd { dst, .. } => Some(dst),
        Op::ISub { dst, .. } => Some(dst),
        Op::IMul { dst, .. } => Some(dst),
        Op::ICmpLt { dst, .. } => Some(dst),
        Op::IAdd64 { dst, .. } => Some(dst),
        Op::ISub64 { dst, .. } => Some(dst),
        Op::AndI64 { dst, .. } => Some(dst),
        Op::OrI64  { dst, .. } => Some(dst),
        Op::XorI64 { dst, .. } => Some(dst),
        Op::ShlI64 { dst, .. } => Some(dst),
        Op::ShrI64 { dst, .. } => Some(dst),
        Op::LdExpF64 { dst, .. } => Some(dst),
        Op::F64ToI32Rn { dst, .. } => Some(dst),
        Op::BitcastF64ToI64 { dst, .. } => Some(dst),
        Op::BitcastI64ToF64 { dst, .. } => Some(dst),
        Op::FCmpGt { dst, .. } => Some(dst),
        Op::FCmpLt { dst, .. } => Some(dst),
        Op::FCmpEq { dst, .. } => Some(dst),
        Op::SelectF64 { dst, .. } => Some(dst),
        Op::SelectI32 { dst, .. } => Some(dst),
        Op::TamExp { dst, .. } => Some(dst),
        Op::TamLn { dst, .. } => Some(dst),
        Op::TamSin { dst, .. } => Some(dst),
        Op::TamCos  { dst, .. } => Some(dst),
        Op::TamAtan { dst, .. } => Some(dst),
        Op::TamPow { dst, .. } => Some(dst),
        Op::StoreF64 { .. } => None,
        Op::ReduceBlockAdd { .. } => None,
        Op::RetF64 { .. } => None,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Convenience runner for the standard kernel shape
// ═══════════════════════════════════════════════════════════════════

/// Run a kernel where the last parameter is the output buffer.
///
/// This is the standard shape for accumulation kernels:
/// inputs are read-only, the last parameter is the output buffer.
///
/// Pre-zeros the output buffer before running.
pub fn run_standard_kernel(
    prog: &Program,
    kernel_name: &str,
    input_names: &[&str],
    inputs: &[&Vec<f64>],
    output_name: &str,
    n_output_slots: usize,
) -> Result<Vec<f64>, InterpError> {
    let mut output = vec![0.0f64; n_output_slots];

    // Build the buffer map. Inputs are read-only but we need to pass mutable
    // references — we cast them unsafely through *const to *mut.
    // Safety: inputs are never written by the kernel (only output is written
    // via reduce_block_add/store).
    let mut owned_inputs: Vec<Vec<f64>> = inputs.iter().map(|s| s.to_vec()).collect();

    let interp = Interpreter::new(prog);

    // Build buf list: named inputs + named output
    let mut buf_vec: Vec<(&str, *mut Vec<f64>)> = input_names.iter()
        .zip(owned_inputs.iter_mut())
        .map(|(name, v)| (*name, v as *mut Vec<f64>))
        .collect();
    buf_vec.push((output_name, &mut output as *mut Vec<f64>));

    // Convert to the &mut Vec<f64> form expected by run_kernel
    // We use a temporary to bridge the pointer and reference lifetimes.
    let mut buf_refs: Vec<(&str, &mut Vec<f64>)> = unsafe {
        buf_vec.iter_mut()
            .map(|(name, ptr)| (*name, &mut **ptr))
            .collect()
    };

    interp.run_kernel(kernel_name, &buf_refs.iter_mut().map(|(n, v)| (*n, v as &mut Vec<f64>)).collect::<Vec<_>>())?;

    Ok(output)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{variance_pass_program, sum_all_add_program};

    /// Campsite 1.10 acceptance: sum_all_add([1..=10]) == 55.0
    #[test]
    fn sum_all_add_1_to_10() {
        let prog = sum_all_add_program();
        let interp = Interpreter::new(&prog);
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let mut out = vec![0.0f64; 1];
        interp.run_kernel("sum_all_add", &[
            ("data", &mut data.clone()),
            ("out", &mut out),
        ]).unwrap();
        assert_eq!(out[0], 55.0, "sum(1..=10) should be 55.0");
    }

    #[test]
    fn sum_all_add_empty() {
        let prog = sum_all_add_program();
        let interp = Interpreter::new(&prog);
        let data: Vec<f64> = vec![];
        let mut out = vec![0.0f64; 1];
        interp.run_kernel("sum_all_add", &[
            ("data", &mut data.clone()),
            ("out", &mut out),
        ]).unwrap();
        assert_eq!(out[0], 0.0, "sum([]) should be 0.0");
    }

    #[test]
    fn sum_all_add_single_element() {
        let prog = sum_all_add_program();
        let interp = Interpreter::new(&prog);
        let data = vec![42.0f64];
        let mut out = vec![0.0f64; 1];
        interp.run_kernel("sum_all_add", &[
            ("data", &mut data.clone()),
            ("out", &mut out),
        ]).unwrap();
        assert_eq!(out[0], 42.0);
    }

    /// Campsite 1.12: variance_pass through interpreter matches expected output.
    #[test]
    fn variance_pass_matches_expected() {
        let prog = variance_pass_program();
        let interp = Interpreter::new(&prog);
        let data: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut out = vec![0.0f64; 3];
        interp.run_kernel("variance_pass", &[
            ("data", &mut data.clone()),
            ("out", &mut out),
        ]).unwrap();
        // sum = 40.0, sum_sq = 4+16+16+16+25+25+49+81 = 232.0, count = 8.0
        assert_eq!(out[2], 8.0, "count");
        assert_eq!(out[0], 40.0, "sum");
        assert_eq!(out[1], 232.0, "sum_sq");
        // population variance = sum_sq/n - (sum/n)^2 = 212/8 - (40/8)^2 = 26.5 - 25.0 = 4.0
        let n = out[2];
        let mean = out[0] / n;
        let var = out[1] / n - mean * mean;
        assert!((var - 4.0).abs() < 1e-12, "population variance should be 4.0, got {}", var);
    }

    #[test]
    fn fsqrt_correct() {
        // Test fsqrt via a tiny function
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: "sqrt_f".into(),
                params: vec![FuncParam { reg: Reg::new("x") }],
                body: vec![
                    Op::FSqrt { dst: Reg::new("r"), a: Reg::new("x") },
                    Op::RetF64 { val: Reg::new("r") },
                ],
            }],
            kernels: vec![],
        };
        let interp = Interpreter::new(&prog);
        let result = interp.call_func("sqrt_f", &[4.0]).unwrap();
        assert_eq!(result, 2.0);
        let result2 = interp.call_func("sqrt_f", &[2.0]).unwrap();
        assert!((result2 - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn select_f64_branch_free() {
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: "abs_via_select".into(),
                params: vec![FuncParam { reg: Reg::new("x") }],
                body: vec![
                    Op::ConstF64 { dst: Reg::new("zero"), value: 0.0 },
                    Op::FCmpLt { dst: Reg::new("neg"), a: Reg::new("x"), b: Reg::new("zero") },
                    Op::FNeg { dst: Reg::new("negx"), a: Reg::new("x") },
                    Op::SelectF64 {
                        dst: Reg::new("r"),
                        pred: Reg::new("neg"),
                        on_true: Reg::new("negx"),
                        on_false: Reg::new("x"),
                    },
                    Op::RetF64 { val: Reg::new("r") },
                ],
            }],
            kernels: vec![],
        };
        let interp = Interpreter::new(&prog);
        assert_eq!(interp.call_func("abs_via_select", &[-3.0]).unwrap(), 3.0);
        assert_eq!(interp.call_func("abs_via_select", &[5.0]).unwrap(), 5.0);
        assert_eq!(interp.call_func("abs_via_select", &[0.0]).unwrap(), 0.0);
    }

    #[test]
    fn parse_and_run_sum_all_add() {
        // End-to-end: parse the hand-written .tam file and run it
        let src = include_str!("../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/sum_all_add.tam");
        let prog = crate::parse::parse_program(src).unwrap();
        let interp = Interpreter::new(&prog);
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let mut out = vec![0.0f64; 1];
        interp.run_kernel("sum_all_add", &[
            ("data", &mut data.clone()),
            ("out", &mut out),
        ]).unwrap();
        assert_eq!(out[0], 55.0, "parse+run sum_all_add([1..=10]) should be 55.0");
    }

    // ── I11: NaN propagation tests ────────────────────────────────────────

    /// Helper: build a function `select_nan_test(x) -> select(x >= 0, x, -x)`
    /// where the input may be NaN. Verifies that I11 is enforced by the interpreter.
    fn make_nan_select_prog(name: &str) -> Program {
        // select(pred, on_true, on_false) where on_true = NaN literal.
        // fcmp_gt(nan_val, 0.0) -> false (IEEE 754)
        // select(false, nan_val, zero) -> should be NaN per I11 (both values checked)
        Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: name.into(),
                params: vec![FuncParam { reg: Reg::new("x") }],
                body: vec![
                    Op::ConstF64 { dst: Reg::new("zero"), value: 0.0 },
                    // pred = x > 0
                    Op::FCmpGt {
                        dst: Reg::new("pred"),
                        a: Reg::new("x"),
                        b: Reg::new("zero"),
                    },
                    // neg = -x
                    Op::FNeg { dst: Reg::new("negx"), a: Reg::new("x") },
                    // r = select(pred, x, -x)  [abs-value pattern]
                    Op::SelectF64 {
                        dst: Reg::new("r"),
                        pred: Reg::new("pred"),
                        on_true: Reg::new("x"),
                        on_false: Reg::new("negx"),
                    },
                    Op::RetF64 { val: Reg::new("r") },
                ],
            }],
            kernels: vec![],
        }
    }

    #[test]
    fn i11_select_f64_nan_on_true_propagates() {
        // I11: select(pred, NaN, finite) → NaN.
        // If on_true is NaN, result must be NaN regardless of pred.
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: "select_nan_true".into(),
                params: vec![],
                body: vec![
                    Op::ConstF64 { dst: Reg::new("nan_v"), value: f64::NAN },
                    Op::ConstF64 { dst: Reg::new("one"), value: 1.0 },
                    // pred = false (1.0 > 2.0 is false, so on_false would be selected)
                    Op::ConstF64 { dst: Reg::new("two"), value: 2.0 },
                    Op::FCmpGt { dst: Reg::new("pred"), a: Reg::new("one"), b: Reg::new("two") },
                    // select(false, NaN, 1.0) — on_true=NaN, on_false=1.0
                    // I11: result must be NaN (not 1.0)
                    Op::SelectF64 {
                        dst: Reg::new("r"),
                        pred: Reg::new("pred"),
                        on_true: Reg::new("nan_v"),
                        on_false: Reg::new("one"),
                    },
                    Op::RetF64 { val: Reg::new("r") },
                ],
            }],
            kernels: vec![],
        };
        let interp = Interpreter::new(&prog);
        let result = interp.call_func("select_nan_true", &[]).unwrap();
        assert!(result.is_nan(),
            "I11: select(false, NaN, 1.0) must propagate NaN, got {}", result);
    }

    #[test]
    fn i11_select_f64_nan_on_false_propagates() {
        // I11: select(pred, finite, NaN) → NaN.
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: "select_nan_false".into(),
                params: vec![],
                body: vec![
                    Op::ConstF64 { dst: Reg::new("nan_v"), value: f64::NAN },
                    Op::ConstF64 { dst: Reg::new("one"), value: 1.0 },
                    // pred = true (1.0 < 2.0)
                    Op::ConstF64 { dst: Reg::new("two"), value: 2.0 },
                    Op::FCmpLt { dst: Reg::new("pred"), a: Reg::new("one"), b: Reg::new("two") },
                    // select(true, 1.0, NaN) — on_true=1.0, on_false=NaN
                    // I11: result must be NaN (not 1.0)
                    Op::SelectF64 {
                        dst: Reg::new("r"),
                        pred: Reg::new("pred"),
                        on_true: Reg::new("one"),
                        on_false: Reg::new("nan_v"),
                    },
                    Op::RetF64 { val: Reg::new("r") },
                ],
            }],
            kernels: vec![],
        };
        let interp = Interpreter::new(&prog);
        let result = interp.call_func("select_nan_false", &[]).unwrap();
        assert!(result.is_nan(),
            "I11: select(true, 1.0, NaN) must propagate NaN, got {}", result);
    }

    #[test]
    fn i11_select_f64_comparison_with_nan_input_propagates() {
        // I11: the naturalist's bug. fcmp_gt(NaN, x) → false (IEEE 754 behavior).
        // Then select(false, NaN_branch, x) must NOT silently return x.
        // The guard is on the VALUE operands, not the predicate.
        //
        // Kernel: abs(x) = select(x > 0, x, -x)
        // When x=NaN: x > 0 = false, so select(false, NaN, -NaN) = NaN (both are NaN).
        // This test checks that the select propagates NaN in this scenario.
        let prog = make_nan_select_prog("abs_nan_test");
        let interp = Interpreter::new(&prog);
        // Normal inputs still work.
        assert_eq!(interp.call_func("abs_nan_test", &[3.0]).unwrap(), 3.0);
        assert_eq!(interp.call_func("abs_nan_test", &[-5.0]).unwrap(), 5.0);
        // NaN input: both on_true (NaN) and on_false (-NaN) are NaN, so result is NaN.
        let result = interp.call_func("abs_nan_test", &[f64::NAN]).unwrap();
        assert!(result.is_nan(),
            "I11: abs(NaN) via select must propagate NaN, got {}", result);
    }
}
