//! AST types for the `.tam` IR.
//!
//! These are the "shapes" — what kinds of things exist. No parsing, no printing,
//! no execution lives here. Just the types and their equality / debug impls.
//!
//! Design constraints:
//! - Zero dependencies outside `std`.
//! - Every type derives `Debug`, `Clone`, `PartialEq` so tests can `assert_eq!`.
//! - Types are closed over Phase 1 scope. Ops beyond Phase 1 do not appear here;
//!   they go in `peak1-tam-ir/future-ops.md` first.
//! - The SSA invariant is a verifier property, not a type property. The AST
//!   represents potentially-invalid programs; the verifier catches them.

// ═══════════════════════════════════════════════════════════════════
// Metadata
// ═══════════════════════════════════════════════════════════════════

/// The version declared in the `.tam` file header.
///
/// Phase 1 only recognizes `TamVersion { major: 0, minor: 1 }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TamVersion {
    pub major: u32,
    pub minor: u32,
}

impl TamVersion {
    pub const PHASE1: TamVersion = TamVersion { major: 0, minor: 1 };
}

/// The target declared in the `.tam` file header.
///
/// Phase 1 only emits `Target::Cross`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// Hardware-agnostic module. All backends can lower this.
    Cross,
    /// Future: backend-specific modules (PTX, SPIR-V, etc.).
    /// Phase 1 parsers must reject these.
    Other(String),
}

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

/// The type system for `.tam` Phase 1.
///
/// Five types total. `BufF64` is a parameter type only — it cannot be a
/// register type. `Pred` is a result type only — it cannot be an arithmetic
/// operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// 32-bit signed integer. Used for indices, counts, and slot numbers.
    I32,
    /// 64-bit signed integer. For future large-N index support.
    I64,
    /// 64-bit IEEE 754 binary64. The math type.
    F64,
    /// Boolean predicate. Result of comparison ops only.
    Pred,
    /// Flat buffer of f64 values with a runtime length. Parameter type only.
    BufF64,
}

// ═══════════════════════════════════════════════════════════════════
// Registers
// ═══════════════════════════════════════════════════════════════════

/// A register name, e.g. `%acc`, `%v2`, `%acc'`.
///
/// The `prime` field records whether this name carries a `'` suffix. In the
/// SSA convention, `prime == false` is the value entering a loop iteration;
/// `prime == true` is the value exiting (the phi-node output).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reg {
    /// The base name (without `%` prefix and without `'` suffix).
    pub name: String,
    /// Whether this register has a prime suffix (`'`).
    pub prime: bool,
}

impl Reg {
    pub fn new(name: impl Into<String>) -> Self {
        Reg { name: name.into(), prime: false }
    }
    pub fn prime(name: impl Into<String>) -> Self {
        Reg { name: name.into(), prime: true }
    }
    /// The canonical display form: `%name` or `%name'`.
    pub fn display(&self) -> String {
        if self.prime {
            format!("%{}'", self.name)
        } else {
            format!("%{}", self.name)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Ops
// ═══════════════════════════════════════════════════════════════════

/// A single instruction in the `.tam` IR.
///
/// Every op that produces a value has a `dst: Reg` field. Ops that don't
/// produce a value (`store.f64`, `reduce_block_add.f64`, `ret.f64`) have
/// no `dst`.
///
/// All floating-point arithmetic ops are non-contracting (I3). There is no
/// `fma.f64` in Phase 1. FMA is never emitted by a backend unless it comes
/// from an explicit future `fma` op.
///
/// All fp ops follow IEEE 754 round-to-nearest-even (I4). No flush-to-zero,
/// no denormal suppression, no fast-math at the IR level.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // ── Constants ────────────────────────────────────────────────────────────

    /// `const.f64 %dst = <value>`
    ConstF64 { dst: Reg, value: f64 },

    /// `const.i32 %dst = <value>`
    ConstI32 { dst: Reg, value: i32 },

    // ── Buffer ops ───────────────────────────────────────────────────────────

    /// `%dst:i32 = bufsize %buf`
    ///
    /// Yields the number of elements in the buffer. Returns i32 (Phase 1 limit:
    /// N ≤ 2^31 - 1). See `future-ops.md` for the i64 upgrade path.
    BufSize { dst: Reg, buf: Reg },

    /// `%dst:f64 = load.f64 %buf, %idx:i32`
    LoadF64 { dst: Reg, buf: Reg, idx: Reg },

    /// `store.f64 %buf, %idx:i32, %val:f64`  (no dst)
    StoreF64 { buf: Reg, idx: Reg, val: Reg },

    // ── Floating-point arithmetic (non-contracting, RNE) ─────────────────────

    /// `fadd.f64 %dst = %a, %b`  — I3: never becomes FMA
    FAdd { dst: Reg, a: Reg, b: Reg },

    /// `fsub.f64 %dst = %a, %b`
    FSub { dst: Reg, a: Reg, b: Reg },

    /// `fmul.f64 %dst = %a, %b`  — I3: never fused with adjacent add
    FMul { dst: Reg, a: Reg, b: Reg },

    /// `fdiv.f64 %dst = %a, %b`
    FDiv { dst: Reg, a: Reg, b: Reg },

    /// `fsqrt.f64 %dst = %a`  — IEEE correctly rounded by spec
    FSqrt { dst: Reg, a: Reg },

    /// `fneg.f64 %dst = %a`
    FNeg { dst: Reg, a: Reg },

    /// `fabs.f64 %dst = %a`
    FAbs { dst: Reg, a: Reg },

    // ── Integer arithmetic ────────────────────────────────────────────────────

    /// `iadd.i32 %dst = %a, %b`
    IAdd { dst: Reg, a: Reg, b: Reg },

    /// `isub.i32 %dst = %a, %b`
    ISub { dst: Reg, a: Reg, b: Reg },

    /// `imul.i32 %dst = %a, %b`
    IMul { dst: Reg, a: Reg, b: Reg },

    /// `icmp_lt %dst:pred = %a:i32, %b:i32`
    ICmpLt { dst: Reg, a: Reg, b: Reg },

    // ── Floating-point comparisons (produce Pred) ─────────────────────────────

    /// `fcmp_gt.f64 %dst:pred = %a, %b`
    FCmpGt { dst: Reg, a: Reg, b: Reg },

    /// `fcmp_lt.f64 %dst:pred = %a, %b`
    FCmpLt { dst: Reg, a: Reg, b: Reg },

    /// `fcmp_eq.f64 %dst:pred = %a, %b`
    FCmpEq { dst: Reg, a: Reg, b: Reg },

    // ── Select (branch-free conditional) ─────────────────────────────────────

    /// `select.f64 %dst = %pred, %t, %f`
    SelectF64 { dst: Reg, pred: Reg, on_true: Reg, on_false: Reg },

    /// `select.i32 %dst = %pred, %t, %f`
    SelectI32 { dst: Reg, pred: Reg, on_true: Reg, on_false: Reg },

    // ── Transcendental stubs ─────────────────────────────────────────────────
    //
    // These are opcodes, not function calls. Every backend inlines the
    // tambear-libm implementation of the function at the call site.
    // Before tambear-libm exists (campsite 1.13 stubs them) these panic.
    //
    // I1: no backend may implement these by calling a vendor math library.
    // I8: first-principles implementations only.

    /// `tam_exp.f64 %dst = %a`  — e^a
    TamExp { dst: Reg, a: Reg },

    /// `tam_ln.f64 %dst = %a`  — natural log
    TamLn { dst: Reg, a: Reg },

    /// `tam_sin.f64 %dst = %a`
    TamSin { dst: Reg, a: Reg },

    /// `tam_cos.f64 %dst = %a`
    TamCos { dst: Reg, a: Reg },

    /// `tam_pow.f64 %dst = %a, %b`  — a^b
    TamPow { dst: Reg, a: Reg, b: Reg },

    // ── Reduction ────────────────────────────────────────────────────────────

    /// `reduce_block_add.f64 %out_buf, %slot_idx:i32, %val:f64`
    ///
    /// Semantic: the accumulated value of `%val` over this block's slice is
    /// written into `%out_buf[%slot_idx]`. CPU interpreter: direct store (one
    /// "block" = all elements). PTX backend: shared-memory tree reduce + block-
    /// zero write. Host folds all block partials after dispatch.
    ///
    /// Phase 6 makes this deterministic across GPU backends. Until then it is
    /// tagged `@xfail_nondeterministic` in the replay harness.
    ReduceBlockAdd { out_buf: Reg, slot_idx: Reg, val: Reg },

    // ── Function return ───────────────────────────────────────────────────────

    /// `ret.f64 %result`  — exits a `func` definition.
    /// Not valid inside a `kernel`.
    RetF64 { val: Reg },
}

// ═══════════════════════════════════════════════════════════════════
// Loop
// ═══════════════════════════════════════════════════════════════════

/// A `loop_grid_stride` block.
///
/// ```text
/// loop_grid_stride %i in [0, %n) {
///   ; body ops
/// }
/// ```
///
/// The induction variable `%i` is defined by the loop, type `i32`.
/// `limit` is the exclusive upper bound — must be an `i32` register.
///
/// Phi pairs: any register `%x` used inside the body that was defined
/// before the loop, and whose updated form `%x'` is assigned inside the body,
/// constitutes a phi pair. The verifier checks that every prime-suffix register
/// used after the loop has a corresponding un-primed definition before it.
///
/// Nesting: NOT allowed in Phase 1. One level only.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopGridStride {
    /// The induction variable (always type `i32`).
    pub induction: Reg,
    /// The exclusive upper bound (`%n`), must resolve to an `i32` register.
    pub limit: Reg,
    /// The body of the loop: flat list of ops.
    /// May NOT contain a nested `LoopGridStride`.
    pub body: Vec<Op>,
}

// ═══════════════════════════════════════════════════════════════════
// Kernel and function body
// ═══════════════════════════════════════════════════════════════════

/// A statement in a kernel body: either a plain op or a loop block.
///
/// Kernels may contain at most one `loop_grid_stride` in Phase 1.
/// The loop appears between setup ops (before it) and reduction ops (after it).
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Op(Op),
    Loop(LoopGridStride),
}

/// A kernel parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelParam {
    /// Parameter type. Must be `BufF64` or `I32`.
    pub ty: Ty,
    /// The register name this parameter binds to in the body.
    pub reg: Reg,
}

/// A function parameter (libm functions only — always `f64`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncParam {
    pub reg: Reg,
}

/// A kernel definition.
///
/// ```text
/// kernel <name>(<params>) {
/// entry:
///   <stmts>
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<KernelParam>,
    /// The body, in order. The verifier checks structural constraints.
    pub body: Vec<Stmt>,
}

/// A function definition (for tambear-libm).
///
/// ```text
/// func <name>(f64 %a, ...) -> f64 {
/// entry:
///   <ops>
///   ret.f64 %result
/// }
/// ```
///
/// Functions may NOT contain loops or `reduce_block_add` ops.
/// They MUST end with exactly one `ret.f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub name: String,
    pub params: Vec<FuncParam>,
    /// The body ops, in order. Must end with `Op::RetF64`.
    pub body: Vec<Op>,
}

// ═══════════════════════════════════════════════════════════════════
// Top-level program
// ═══════════════════════════════════════════════════════════════════

/// A complete `.tam` module.
///
/// A module has a header (version + target), zero or more function definitions
/// (libm), and zero or more kernel definitions (compute entry points).
///
/// Phase 1 only accepts `version == TamVersion::PHASE1` and `target == Target::Cross`.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub version: TamVersion,
    pub target: Target,
    pub funcs: Vec<FuncDef>,
    pub kernels: Vec<KernelDef>,
}

impl Program {
    /// Convenience: get a kernel by name.
    pub fn kernel(&self, name: &str) -> Option<&KernelDef> {
        self.kernels.iter().find(|k| k.name == name)
    }

    /// Convenience: get a function by name.
    pub fn func(&self, name: &str) -> Option<&FuncDef> {
        self.funcs.iter().find(|f| f.name == name)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_display_unprimed() {
        assert_eq!(Reg::new("acc").display(), "%acc");
    }

    #[test]
    fn reg_display_primed() {
        assert_eq!(Reg::prime("acc").display(), "%acc'");
    }

    #[test]
    fn reg_equality_prime_matters() {
        assert_ne!(Reg::new("acc"), Reg::prime("acc"));
        assert_eq!(Reg::prime("x"), Reg::prime("x"));
    }

    #[test]
    fn tam_version_phase1_constant() {
        let v = TamVersion::PHASE1;
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
    }

    #[test]
    fn program_lookup_kernel() {
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![],
            kernels: vec![KernelDef {
                name: "sum_all_add".into(),
                params: vec![],
                body: vec![],
            }],
        };
        assert!(prog.kernel("sum_all_add").is_some());
        assert!(prog.kernel("nope").is_none());
    }

    #[test]
    fn op_const_f64_partialeq() {
        let a = Op::ConstF64 { dst: Reg::new("x"), value: 1.0 };
        let b = Op::ConstF64 { dst: Reg::new("x"), value: 1.0 };
        assert_eq!(a, b);
    }

    #[test]
    fn op_const_f64_nan_identity() {
        // NaN != NaN in IEEE, but we need structural equality for AST comparison.
        // PartialEq on Op::ConstF64 delegates to f64::eq, so nan != nan.
        // This is the correct behavior for a structural AST — don't fight it.
        let a = Op::ConstF64 { dst: Reg::new("x"), value: f64::NAN };
        let b = Op::ConstF64 { dst: Reg::new("x"), value: f64::NAN };
        // Structural inequality for NaN is expected. The parser round-trip test
        // (campsite 1.8) uses bit-exact representation, not semantic equality.
        assert_ne!(a, b);
    }
}
