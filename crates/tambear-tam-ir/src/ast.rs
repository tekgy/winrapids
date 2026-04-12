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
// OrderStrategyRef
// ═══════════════════════════════════════════════════════════════════

/// A named reference into the OrderStrategy registry.
///
/// Programs declare their reduction order by name, e.g. `@order(sequential_left)`.
/// The name is validated against the registry at verify time. Unknown names are
/// rejected by the verifier (I7 compliance — "backend chooses" is not a total order).
///
/// This is an open reference type, not a closed enum. New strategies are added
/// to the registry in `order_strategy.rs` without changing this type or any
/// existing code that pattern-matches on op variants. The AST stays stable;
/// the registry grows.
///
/// The full engineering contract for each strategy (formal spec, reference
/// implementation, bit-exact test vectors, fusion-compatibility metadata)
/// lives in `order_strategy.rs`. This type is just the name carried by an op.
///
/// **I7 compliance:** The verifier calls `order_strategy::is_known(&ref_.0)` and
/// rejects programs that name an unregistered strategy. This prevents programs
/// from naming strategies that no backend can implement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderStrategyRef(pub String);

impl OrderStrategyRef {
    /// Construct from a static string (for use in tests and fixtures).
    pub fn new(name: impl Into<String>) -> Self {
        OrderStrategyRef(name.into())
    }

    /// The strategy name as a str reference.
    pub fn name(&self) -> &str {
        &self.0
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

    /// `const.i64 %dst = <value>`
    ConstI64 { dst: Reg, value: i64 },

    /// `iadd.i32 %dst = %a, %b`
    IAdd { dst: Reg, a: Reg, b: Reg },

    /// `isub.i32 %dst = %a, %b`
    ISub { dst: Reg, a: Reg, b: Reg },

    /// `imul.i32 %dst = %a, %b`
    IMul { dst: Reg, a: Reg, b: Reg },

    /// `icmp_lt %dst:pred = %a:i32, %b:i32`
    ICmpLt { dst: Reg, a: Reg, b: Reg },

    /// `iadd.i64 %dst = %a:i64, %b:i64`
    IAdd64 { dst: Reg, a: Reg, b: Reg },

    /// `isub.i64 %dst = %a:i64, %b:i64`
    ISub64 { dst: Reg, a: Reg, b: Reg },

    /// `and.i64 %dst = %a:i64, %b:i64`
    AndI64 { dst: Reg, a: Reg, b: Reg },

    /// `or.i64 %dst = %a:i64, %b:i64`  — needed for RFA exponent extraction
    OrI64 { dst: Reg, a: Reg, b: Reg },

    /// `xor.i64 %dst = %a:i64, %b:i64`
    XorI64 { dst: Reg, a: Reg, b: Reg },

    /// `shl.i64 %dst = %a:i64, %shift:i32`  — logical shift left
    ShlI64 { dst: Reg, a: Reg, shift: Reg },

    /// `shr.i64 %dst = %a:i64, %shift:i32`  — arithmetic shift right
    ShrI64 { dst: Reg, a: Reg, shift: Reg },

    // ── Float ↔ integer conversion and reinterpretation ───────────────────────

    /// `ldexp.f64 %dst = %mantissa:f64, %exp:i32`  — %dst = %mantissa * 2^%exp
    ///
    /// Correct IEEE 754 semantics: handles subnormals, overflow to infinity,
    /// and underflow to zero. Required for tam_exp range reconstruction.
    /// Equivalent to C `ldexp(mantissa, exp)`.
    LdExpF64 { dst: Reg, mantissa: Reg, exp: Reg },

    /// `f64_to_i32_rn %dst:i32 = %a:f64`  — f64 → i32, round-to-nearest-even
    ///
    /// Required for tam_exp argument reduction (computing n = round(x / ln2)).
    /// Saturates at INT32_MIN / INT32_MAX for out-of-range inputs.
    F64ToI32Rn { dst: Reg, a: Reg },

    /// `bitcast.f64.i64 %dst:i64 = %a:f64`  — reinterpret f64 bits as i64
    ///
    /// No value conversion. The bit pattern of %a (IEEE 754 double) is
    /// reinterpreted directly as a signed 64-bit integer. Used for exponent
    /// and mantissa extraction in tam_ln and RFA bin-index computation.
    BitcastF64ToI64 { dst: Reg, a: Reg },

    /// `bitcast.i64.f64 %dst:f64 = %a:i64`  — reinterpret i64 bits as f64
    ///
    /// Inverse of `bitcast.f64.i64`. Assembles an f64 from a bit pattern.
    /// Used to reconstruct f64 values with manipulated exponent fields.
    BitcastI64ToF64 { dst: Reg, a: Reg },

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

    /// `tam_atan.f64 %dst = %a`  — arctan(a), result in radians.
    ///
    /// Range: `(-π/2, π/2)` for all finite inputs. `tam_atan(±∞) = ±π/2`.
    /// `tam_atan(NaN) = NaN`. `tam_atan(0) = 0` (sign preserved).
    ///
    /// Campsite 1.18 stub: added 2026-04-12 to unblock Phase 2 atan implementation.
    /// Implementation pending tambear-libm (campsite 2.8 or similar).
    TamAtan { dst: Reg, a: Reg },

    // ── Reduction ────────────────────────────────────────────────────────────

    /// `reduce_block_add.f64 %out_buf, %slot_idx:i32, %val:f64 @order(<strategy>)`
    ///
    /// Semantic: the accumulated value of `%val` over this block's slice is
    /// written into `%out_buf[%slot_idx]`. CPU interpreter: direct store (one
    /// "block" = all elements). PTX backend: shared-memory tree reduce + block-
    /// zero write. Host folds all block partials after dispatch.
    ///
    /// The `order` field is a named reference into the OrderStrategy registry
    /// (see `order_strategy.rs`). The verifier rejects unknown strategy names
    /// to satisfy I7 (total order enables correctness). New strategies are added
    /// to the registry without changing this type.
    ///
    /// Phase 6 delivers deterministic cross-backend results by making the order
    /// explicit and consistent across CPU, PTX, and SPIR-V backends.
    ReduceBlockAdd { out_buf: Reg, slot_idx: Reg, val: Reg, order: OrderStrategyRef },

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
///
/// Optional attributes go in the `attrs` field. Phase 1 defines two attributes:
/// `accumulator_state_size` (bytes of shared memory per accumulator, for RFA) and
/// `default_order_strategy` (the kernel-level default reduction order, per I7).
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<KernelParam>,
    /// Optional kernel attributes. See `KernelAttr`.
    pub attrs: Vec<KernelAttr>,
    /// The body, in order. The verifier checks structural constraints.
    pub body: Vec<Stmt>,
}

/// A kernel attribute — key/value metadata on a kernel definition.
///
/// Phase 1 attributes:
///
/// - `accumulator_state_size: <bytes>` — size in bytes of the per-thread
///   accumulator state buffer needed by the kernel (e.g. 52 for RFA fold=3).
///   GPU backends use this to allocate shared memory. CPU backend ignores it.
///   Parser syntax: `@accumulator_state_size(<decimal>)` before `kernel`.
///
/// - `default_order_strategy: <name>` — the default OrderStrategy for all
///   `reduce_block_add` ops in this kernel. Per-op `@order(...)` overrides this.
///   Parser syntax: `@default_order_strategy(sequential_left)` before `kernel`.
///
///   This is campsite 1.17's contribution. A kernel declares its default at the
///   top, and individual ops override when needed. The verifier enforces that
///   every `reduce_block_add` op has an effective order strategy — either from
///   the per-op `@order(...)` or from the kernel's `@default_order_strategy(...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelAttr {
    /// `@accumulator_state_size(<bytes>)`
    AccumulatorStateSize(usize),
    /// `@default_order_strategy(<name>)` — default reduction order for this kernel.
    DefaultOrderStrategy(OrderStrategyRef),
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
                attrs: vec![],
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
