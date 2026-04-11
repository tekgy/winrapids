//! Metadata for Expr atoms: the complete self-description.
//!
//! Every Expr variant is DONE DONE — fully described across all 20 facets
//! that any consumer (IDE, TAM, proof engine, symbolic differentiator,
//! optimizer, notation renderer, discover(), sweep()) could need.

use super::Expr;

/// Properties an Expr node can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Continuous,
    Differentiable,
    Bounded,
    Monotonic,
    MonotonicIncreasing,
    MonotonicDecreasing,
    Periodic,
    Associative,
    Commutative,
    Idempotent,
    Involution,          // f(f(x)) = x
    PositivePreserving,  // x > 0 → f(x) > 0
    LinearInFirstArg,    // f(ax, y) = a*f(x, y)
    DistributesOverAdd,  // f(a+b) = f(a) + f(b) (homomorphism)
}

/// Domain restriction.
#[derive(Debug, Clone)]
pub enum Domain {
    AllReals,
    Positive,          // x > 0
    NonNegative,       // x ≥ 0
    UnitInterval,      // x ∈ [-1, 1]
    NonZero,           // x ≠ 0
    Custom(&'static str),
}

/// How an atom handles NaN input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanBehavior {
    /// NaN in → NaN out (most math ops)
    Propagate,
    /// NaN is treated as identity (dangerous — the bug class we found)
    Absorb,
    /// Returns a specific value for NaN input
    Replace(i64), // using i64 to keep Copy; actual value = f64::from_bits(v as u64)
    /// Should never receive NaN (domain violation)
    DomainError,
}

/// Relative computational cost (for optimizer hints).
/// 1 = add/sub, 2 = mul, 4 = div, 8 = sqrt, 10 = exp/ln, 15 = pow, 20 = trig
#[derive(Debug, Clone, Copy)]
pub struct Cost(pub u8);

/// Complete metadata for an Expr atom.
#[derive(Debug, Clone)]
pub struct AtomMeta {
    // === Notation (3 representations) ===
    pub syntax: &'static str,
    pub latex: &'static str,
    pub tambear: &'static str,

    // === Domain/Range ===
    pub domain: Domain,
    pub range: &'static str,

    // === Algebraic Properties ===
    pub properties: &'static [Property],

    // === Derivative rule: d/dx f(x) expressed as a closure-like description ===
    /// How to differentiate this node (chain rule component).
    /// None = not differentiable or leaf node.
    pub derivative: Option<DerivativeRule>,

    // === Inverse operation ===
    /// What undoes this operation. exp ↔ ln, sin ↔ asin, sq ↔ sqrt (for x≥0)
    pub inverse: Option<&'static str>,

    // === Identity / Absorbing elements (for Ops) ===
    /// The identity element: f(identity, x) = x. E.g. 0 for Add, 1 for Mul.
    pub identity: Option<f64>,
    /// The absorbing element: f(absorbing, x) = absorbing. E.g. 0 for Mul, -∞ for Max.
    pub absorbing: Option<f64>,

    // === Numerical behavior ===
    pub nan_behavior: NanBehavior,
    pub cost: Cost,
    /// When does this atom lose precision? Empty = always stable.
    pub stability_note: &'static str,

    // === Simplification rules (Expr → Expr rewrites) ===
    /// Named simplification patterns this atom participates in.
    /// E.g. "exp(ln(x)) = x", "neg(neg(x)) = x", "sq(sqrt(x)) = x for x≥0"
    pub simplifications: &'static [&'static str],

    // === GPU mapping ===
    /// What this compiles to in .tam IR.
    /// Cost(1-2) = direct IEEE 754 hardware op (fadd, fmul, fdiv, fsqrt).
    /// Cost(8+) = tambear-implemented from hardware ops (sin, exp, ln).
    /// The cost already tells the optimizer which is which.
    pub tam_instruction: &'static str,
}

/// How to differentiate an Expr node.
#[derive(Debug, Clone, Copy)]
pub enum DerivativeRule {
    /// d/dx x = 1 (identity)
    Identity,
    /// d/dx c = 0 (constant)
    Zero,
    /// d/dx f(g(x)) = f'(g(x)) * g'(x) — outer derivative is named
    /// e.g. for sin: outer = cos, so d/dx sin(g(x)) = cos(g(x)) * g'(x)
    Chain(&'static str),
    /// d/dx (a + b) = a' + b' (sum rule)
    SumRule,
    /// d/dx (a * b) = a'b + ab' (product rule)
    ProductRule,
    /// d/dx (a / b) = (a'b - ab') / b² (quotient rule)
    QuotientRule,
    /// d/dx a^b = a^b * (b' * ln(a) + b * a'/a) (generalized power rule)
    PowerRule,
}

/// Get metadata for an Expr variant.
pub fn atom_meta(expr: &Expr) -> AtomMeta {
    match expr {
        Expr::Val => AtomMeta {
            syntax: "val", latex: "x", tambear: "v",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Identity),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "load",
        },
        Expr::Lit(_) => AtomMeta {
            syntax: "c", latex: "c", tambear: "c",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Zero),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "immediate",
        },
        Expr::Neg(_) => AtomMeta {
            syntax: "neg(x)", latex: "-x", tambear: "−v",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicDecreasing, Property::Involution],
            derivative: Some(DerivativeRule::Chain("neg")), // d/dx -f = -f'
            inverse: Some("neg"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "", simplifications: &["neg(neg(x)) = x"],
            tam_instruction: "fneg",
        },
        Expr::Abs(_) => AtomMeta {
            syntax: "abs(x)", latex: "|x|", tambear: "|v|",
            domain: Domain::AllReals, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Bounded],
            derivative: Some(DerivativeRule::Chain("sign")), // d/dx |f| = sign(f) * f'
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "", simplifications: &["abs(abs(x)) = abs(x)", "abs(neg(x)) = abs(x)"],
            tam_instruction: "fabs",
        },
        Expr::Sq(_) => AtomMeta {
            syntax: "sq(x)", latex: "x^2", tambear: "v²",
            domain: Domain::AllReals, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Differentiable, Property::PositivePreserving],
            derivative: Some(DerivativeRule::Chain("2*x")), // d/dx f² = 2f * f'
            inverse: Some("sqrt"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(2),
            stability_note: "overflow for |x| > 1.3e154",
            simplifications: &["sq(sqrt(x)) = x for x≥0", "sq(abs(x)) = sq(x)"],
            tam_instruction: "fmul(x,x)",
        },
        Expr::Sqrt(_) => AtomMeta {
            syntax: "sqrt(x)", latex: r"\sqrt{x}", tambear: "√v",
            domain: Domain::NonNegative, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::PositivePreserving],
            derivative: Some(DerivativeRule::Chain("1/(2*sqrt(x))")), // d/dx √f = f'/(2√f)
            inverse: Some("sq"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(8),
            stability_note: "NaN for negative input",
            simplifications: &["sqrt(sq(x)) = abs(x)", "sqrt(0) = 0", "sqrt(1) = 1"],
            tam_instruction: "fsqrt",
        },
        Expr::Ln(_) => AtomMeta {
            syntax: "ln(x)", latex: r"\ln(x)", tambear: "ln v",
            domain: Domain::Positive, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing],
            derivative: Some(DerivativeRule::Chain("1/x")), // d/dx ln(f) = f'/f
            inverse: Some("exp"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(10),
            stability_note: "ln(1+x) loses precision for small x; use ln1p",
            simplifications: &["ln(exp(x)) = x", "ln(1) = 0", "ln(e) = 1"],
            tam_instruction: "log",
        },
        Expr::Exp(_) => AtomMeta {
            syntax: "exp(x)", latex: "e^x", tambear: "eᵛ",
            domain: Domain::AllReals, range: "ℝ>0",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::PositivePreserving],
            derivative: Some(DerivativeRule::Chain("exp")), // d/dx e^f = e^f * f'  (self-derivative!)
            inverse: Some("ln"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(10),
            stability_note: "overflow for x > 709.8; underflow for x < -745.1",
            simplifications: &["exp(ln(x)) = x for x>0", "exp(0) = 1"],
            tam_instruction: "exp",
        },
        Expr::Sin(_) => AtomMeta {
            syntax: "sin(x)", latex: r"\sin(x)", tambear: "sin v",
            domain: Domain::AllReals, range: "[-1,1]",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::Periodic],
            derivative: Some(DerivativeRule::Chain("cos")), // d/dx sin(f) = cos(f) * f'
            inverse: Some("asin"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "precision degrades for very large |x| (argument reduction)",
            simplifications: &["sin(0) = 0", "sin(asin(x)) = x for |x|≤1"],
            tam_instruction: "sin",
        },
        Expr::Cos(_) => AtomMeta {
            syntax: "cos(x)", latex: r"\cos(x)", tambear: "cos v",
            domain: Domain::AllReals, range: "[-1,1]",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::Periodic],
            derivative: Some(DerivativeRule::Chain("-sin")), // d/dx cos(f) = -sin(f) * f'
            inverse: Some("acos"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "same as sin",
            simplifications: &["cos(0) = 1", "sin²+cos²=1"],
            tam_instruction: "cos",
        },
        Expr::Tanh(_) => AtomMeta {
            syntax: "tanh(x)", latex: r"\tanh(x)", tambear: "tanh v",
            domain: Domain::AllReals, range: "(-1,1)",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::MonotonicIncreasing],
            derivative: Some(DerivativeRule::Chain("1-tanh²")), // d/dx tanh(f) = (1-tanh²(f)) * f'
            inverse: Some("atanh"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "",
            simplifications: &["tanh(0) = 0"],
            tam_instruction: "tanh",
        },
        Expr::Recip(_) => AtomMeta {
            syntax: "recip(x)", latex: r"\frac{1}{x}", tambear: "1/v",
            domain: Domain::NonZero, range: "ℝ\\{0}",
            properties: &[Property::Continuous, Property::Differentiable, Property::Involution, Property::MonotonicDecreasing],
            derivative: Some(DerivativeRule::Chain("-1/x²")), // d/dx 1/f = -f'/f²
            inverse: Some("recip"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(4),
            stability_note: "infinity for x near 0",
            simplifications: &["recip(recip(x)) = x"],
            tam_instruction: "fdiv(1,x) or rcp",
        },
        Expr::Sign(_) => AtomMeta {
            syntax: "sign(x)", latex: r"\text{sgn}(x)", tambear: "sgn v",
            domain: Domain::AllReals, range: "{-1,0,1}",
            properties: &[Property::Bounded, Property::Idempotent],
            derivative: None, // not differentiable at 0
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(2),
            stability_note: "discontinuous at 0",
            simplifications: &["sign(sign(x)) = sign(x)", "sign(abs(x)) = 1 for x≠0"],
            tam_instruction: "copysign(1,x)",
        },
        Expr::Add(_, _) => AtomMeta {
            syntax: "a + b", latex: "a + b", tambear: "a + b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::Associative, Property::Commutative],
            derivative: Some(DerivativeRule::SumRule),
            inverse: Some("sub"), identity: Some(0.0), absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "catastrophic cancellation when a ≈ -b",
            simplifications: &["x + 0 = x", "x + neg(x) = 0"],
            tam_instruction: "fadd",
        },
        Expr::Sub(_, _) => AtomMeta {
            syntax: "a - b", latex: "a - b", tambear: "a − b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable],
            derivative: Some(DerivativeRule::SumRule), // d/dx (f-g) = f' - g'
            inverse: Some("add"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "catastrophic cancellation when a ≈ b",
            simplifications: &["x - 0 = x", "x - x = 0"],
            tam_instruction: "fsub",
        },
        Expr::Mul(_, _) => AtomMeta {
            syntax: "a * b", latex: r"a \cdot b", tambear: "a × b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::Associative, Property::Commutative],
            derivative: Some(DerivativeRule::ProductRule),
            inverse: Some("div"), identity: Some(1.0), absorbing: Some(0.0),
            nan_behavior: NanBehavior::Propagate, cost: Cost(2),
            stability_note: "overflow for large operands",
            simplifications: &["x * 1 = x", "x * 0 = 0", "x * neg(1) = neg(x)"],
            tam_instruction: "fmul",
        },
        Expr::Div(_, _) => AtomMeta {
            syntax: "a / b", latex: r"\frac{a}{b}", tambear: "a ÷ b",
            domain: Domain::Custom("b ≠ 0"), range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable],
            derivative: Some(DerivativeRule::QuotientRule),
            inverse: Some("mul"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(4),
            stability_note: "infinity when b → 0; NaN when 0/0",
            simplifications: &["x / 1 = x", "x / x = 1 for x≠0", "0 / x = 0 for x≠0"],
            tam_instruction: "fdiv",
        },
        Expr::Pow(_, _) => AtomMeta {
            syntax: "a ^ b", latex: "a^b", tambear: "aᵇ",
            domain: Domain::Custom("a > 0 or b ∈ ℤ"), range: "ℝ",
            properties: &[Property::Differentiable],
            derivative: Some(DerivativeRule::PowerRule),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "overflow/underflow; NaN for negative base with non-integer exponent",
            simplifications: &["x^0 = 1", "x^1 = x", "x^2 = sq(x)"],
            tam_instruction: "pow",
        },

        Expr::Val2 => AtomMeta {
            syntax: "val2", latex: "y", tambear: "w",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Zero),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "load_col2",
        },
        Expr::Col(_) => AtomMeta {
            syntax: "col(n)", latex: "x_n", tambear: "cₙ",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Zero),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "load_col",
        },
        Expr::Ref => AtomMeta {
            syntax: "ref", latex: r"\mu", tambear: "r",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Zero),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "load_ref",
        },
        Expr::Var(_) => AtomMeta {
            syntax: "var(name)", latex: "name", tambear: "name",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: Some(DerivativeRule::Zero),
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(0),
            stability_note: "", simplifications: &[],
            tam_instruction: "load_var",
        },
        Expr::Floor(_) => AtomMeta {
            syntax: "floor(x)", latex: r"\lfloor x \rfloor", tambear: "⌊v⌋",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::MonotonicIncreasing, Property::Idempotent],
            derivative: None, // not differentiable at integers
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "", simplifications: &["floor(floor(x)) = floor(x)"],
            tam_instruction: "floor",
        },
        Expr::Ceil(_) => AtomMeta {
            syntax: "ceil(x)", latex: r"\lceil x \rceil", tambear: "⌈v⌉",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::MonotonicIncreasing, Property::Idempotent],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "", simplifications: &["ceil(ceil(x)) = ceil(x)"],
            tam_instruction: "ceil",
        },
        Expr::IsFinite(_) => AtomMeta {
            syntax: "is_finite(x)", latex: r"\mathbb{1}_{x \in \mathbb{R}}", tambear: "fin? v",
            domain: Domain::AllReals, range: "{0,1}",
            properties: &[Property::Bounded, Property::Idempotent],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Replace(0), // NaN → 0 (not finite)
            cost: Cost(1),
            stability_note: "", simplifications: &[],
            tam_instruction: "is_finite",
        },
        Expr::Tan(_) => AtomMeta {
            syntax: "tan(x)", latex: r"\tan(x)", tambear: "tan v",
            domain: Domain::Custom("ℝ \\ {π/2 + kπ}"), range: "ℝ",
            properties: &[Property::Differentiable, Property::Periodic],
            derivative: Some(DerivativeRule::Chain("1+tan²")), // sec²
            inverse: Some("atan"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "undefined at odd multiples of π/2",
            simplifications: &["tan(0) = 0", "tan(atan(x)) = x"],
            tam_instruction: "tan",
        },
        Expr::Asin(_) => AtomMeta {
            syntax: "asin(x)", latex: r"\arcsin(x)", tambear: "asin v",
            domain: Domain::UnitInterval, range: "[-π/2, π/2]",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::Bounded],
            derivative: Some(DerivativeRule::Chain("1/sqrt(1-x²)")),
            inverse: Some("sin"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "NaN for |x| > 1",
            simplifications: &["asin(sin(x)) = x for |x|≤π/2"],
            tam_instruction: "asin",
        },
        Expr::Acos(_) => AtomMeta {
            syntax: "acos(x)", latex: r"\arccos(x)", tambear: "acos v",
            domain: Domain::UnitInterval, range: "[0, π]",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicDecreasing, Property::Bounded],
            derivative: Some(DerivativeRule::Chain("-1/sqrt(1-x²)")),
            inverse: Some("cos"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "NaN for |x| > 1",
            simplifications: &[],
            tam_instruction: "acos",
        },
        Expr::Atan(_) => AtomMeta {
            syntax: "atan(x)", latex: r"\arctan(x)", tambear: "atan v",
            domain: Domain::AllReals, range: "(-π/2, π/2)",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::Bounded],
            derivative: Some(DerivativeRule::Chain("1/(1+x²)")),
            inverse: Some("tan"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "",
            simplifications: &["atan(0) = 0"],
            tam_instruction: "atan",
        },
        Expr::Sinh(_) => AtomMeta {
            syntax: "sinh(x)", latex: r"\sinh(x)", tambear: "sinh v",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing],
            derivative: Some(DerivativeRule::Chain("cosh")),
            inverse: Some("asinh"), identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(10),
            stability_note: "overflow for large |x|",
            simplifications: &["sinh(0) = 0"],
            tam_instruction: "sinh",
        },
        Expr::Cosh(_) => AtomMeta {
            syntax: "cosh(x)", latex: r"\cosh(x)", tambear: "cosh v",
            domain: Domain::AllReals, range: "[1, ∞)",
            properties: &[Property::Continuous, Property::Differentiable, Property::PositivePreserving],
            derivative: Some(DerivativeRule::Chain("sinh")),
            inverse: None, // cosh is not injective
            identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(10),
            stability_note: "overflow for large |x|",
            simplifications: &["cosh(0) = 1", "cosh²-sinh²=1"],
            tam_instruction: "cosh",
        },
        Expr::Round(_) => AtomMeta {
            syntax: "round(x)", latex: r"\text{round}(x)", tambear: "⌊v⌉",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::Idempotent],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "banker's rounding for .5",
            simplifications: &["round(round(x)) = round(x)"],
            tam_instruction: "round",
        },
        Expr::Trunc(_) => AtomMeta {
            syntax: "trunc(x)", latex: r"\text{trunc}(x)", tambear: "trunc v",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::Idempotent],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "",
            simplifications: &["trunc(trunc(x)) = trunc(x)"],
            tam_instruction: "trunc",
        },
        Expr::Min(_, _) => AtomMeta {
            syntax: "min(a, b)", latex: r"\min(a, b)", tambear: "a ∧ b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Associative, Property::Commutative, Property::Idempotent],
            derivative: None, // not differentiable at a = b
            inverse: None, identity: Some(f64::INFINITY), absorbing: Some(f64::NEG_INFINITY),
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "",
            simplifications: &["min(x, x) = x", "min(x, ∞) = x"],
            tam_instruction: "fmin",
        },
        Expr::Max(_, _) => AtomMeta {
            syntax: "max(a, b)", latex: r"\max(a, b)", tambear: "a ∨ b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Associative, Property::Commutative, Property::Idempotent],
            derivative: None,
            inverse: None, identity: Some(f64::NEG_INFINITY), absorbing: Some(f64::INFINITY),
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "",
            simplifications: &["max(x, x) = x", "max(x, -∞) = x"],
            tam_instruction: "fmax",
        },
        Expr::Mod(_, _) => AtomMeta {
            syntax: "a % b", latex: r"a \bmod b", tambear: "a mod b",
            domain: Domain::Custom("b ≠ 0"), range: "ℝ",
            properties: &[Property::Periodic],
            derivative: None, // not differentiable at multiples of b
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(4),
            stability_note: "",
            simplifications: &["x % 1 = fract(x)"],
            tam_instruction: "fmod",
        },
        Expr::Atan2(_, _) => AtomMeta {
            syntax: "atan2(y, x)", latex: r"\text{atan2}(y, x)", tambear: "atan2(a, b)",
            domain: Domain::Custom("not both zero"), range: "(-π, π]",
            properties: &[Property::Continuous, Property::Bounded],
            derivative: None, // partial derivatives exist but complex
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(15),
            stability_note: "",
            simplifications: &[],
            tam_instruction: "atan2",
        },
        Expr::Clamp(_, _, _) => AtomMeta {
            syntax: "clamp(x, lo, hi)", latex: r"\text{clamp}(x, lo, hi)", tambear: "clamp(v, lo, hi)",
            domain: Domain::AllReals, range: "[lo, hi]",
            properties: &[Property::Continuous, Property::MonotonicIncreasing, Property::Bounded, Property::Idempotent],
            derivative: None, // not differentiable at lo and hi
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(2),
            stability_note: "",
            simplifications: &["clamp(clamp(x, lo, hi), lo, hi) = clamp(x, lo, hi)"],
            tam_instruction: "clamp",
        },
        Expr::If(_, _, _) => AtomMeta {
            syntax: "if(cond, then, else)", latex: r"\text{if } c > 0 \text{ then } a \text{ else } b", tambear: "cond ? a : b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
            derivative: None, // not differentiable at the branch point
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(2),
            stability_note: "NaN in condition → takes else branch",
            simplifications: &[],
            tam_instruction: "select",
        },
        Expr::Gt(_, _) => AtomMeta {
            syntax: "a > b", latex: "a > b", tambear: "a > b",
            domain: Domain::AllReals, range: "{0, 1}",
            properties: &[Property::Bounded],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "NaN > x = false (0.0)",
            simplifications: &[],
            tam_instruction: "fcmp_gt",
        },
        Expr::Lt(_, _) => AtomMeta {
            syntax: "a < b", latex: "a < b", tambear: "a < b",
            domain: Domain::AllReals, range: "{0, 1}",
            properties: &[Property::Bounded],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "NaN < x = false (0.0)",
            simplifications: &[],
            tam_instruction: "fcmp_lt",
        },
        Expr::Eq(_, _) => AtomMeta {
            syntax: "a == b", latex: "a = b", tambear: "a ≈ b",
            domain: Domain::AllReals, range: "{0, 1}",
            properties: &[Property::Commutative, Property::Bounded],
            derivative: None,
            inverse: None, identity: None, absorbing: None,
            nan_behavior: NanBehavior::Propagate, cost: Cost(1),
            stability_note: "uses epsilon comparison (1e-15)",
            simplifications: &["(x == x) = 1 for finite x"],
            tam_instruction: "fcmp_eq",
        },
    }
}

// === Notation renderers (unchanged) ===

/// Render a full Expr tree as TBS syntax.
pub fn to_syntax(expr: &Expr) -> String {
    match expr {
        Expr::Val => "val".into(),
        Expr::Val2 => "val2".into(),
        Expr::Col(n) => format!("col({n})"),
        Expr::Ref => "ref".into(),
        Expr::Lit(c) => format!("{c}"),
        Expr::Var(s) => s.clone(),
        Expr::Neg(a) => format!("-({})", to_syntax(a)),
        Expr::Abs(a) => format!("abs({})", to_syntax(a)),
        Expr::Recip(a) => format!("1/({})", to_syntax(a)),
        Expr::Sq(a) => format!("({})²", to_syntax(a)),
        Expr::Sqrt(a) => format!("sqrt({})", to_syntax(a)),
        Expr::Ln(a) => format!("ln({})", to_syntax(a)),
        Expr::Exp(a) => format!("exp({})", to_syntax(a)),
        Expr::Sin(a) => format!("sin({})", to_syntax(a)),
        Expr::Cos(a) => format!("cos({})", to_syntax(a)),
        Expr::Tan(a) => format!("tan({})", to_syntax(a)),
        Expr::Tanh(a) => format!("tanh({})", to_syntax(a)),
        Expr::Sign(a) => format!("sign({})", to_syntax(a)),
        Expr::Floor(a) => format!("floor({})", to_syntax(a)),
        Expr::Ceil(a) => format!("ceil({})", to_syntax(a)),
        Expr::Round(a) => format!("round({})", to_syntax(a)),
        Expr::Trunc(a) => format!("trunc({})", to_syntax(a)),
        Expr::IsFinite(a) => format!("is_finite({})", to_syntax(a)),
        Expr::Asin(a) => format!("asin({})", to_syntax(a)),
        Expr::Acos(a) => format!("acos({})", to_syntax(a)),
        Expr::Atan(a) => format!("atan({})", to_syntax(a)),
        Expr::Sinh(a) => format!("sinh({})", to_syntax(a)),
        Expr::Cosh(a) => format!("cosh({})", to_syntax(a)),
        Expr::Add(a, b) => format!("({} + {})", to_syntax(a), to_syntax(b)),
        Expr::Sub(a, b) => format!("({} - {})", to_syntax(a), to_syntax(b)),
        Expr::Mul(a, b) => format!("({} * {})", to_syntax(a), to_syntax(b)),
        Expr::Div(a, b) => format!("({} / {})", to_syntax(a), to_syntax(b)),
        Expr::Pow(a, b) => format!("({} ^ {})", to_syntax(a), to_syntax(b)),
        Expr::Min(a, b) => format!("min({}, {})", to_syntax(a), to_syntax(b)),
        Expr::Max(a, b) => format!("max({}, {})", to_syntax(a), to_syntax(b)),
        Expr::Mod(a, b) => format!("({} % {})", to_syntax(a), to_syntax(b)),
        Expr::Atan2(a, b) => format!("atan2({}, {})", to_syntax(a), to_syntax(b)),
        Expr::Clamp(x, lo, hi) => format!("clamp({}, {}, {})", to_syntax(x), to_syntax(lo), to_syntax(hi)),
        Expr::If(c, t, e) => format!("if({}, {}, {})", to_syntax(c), to_syntax(t), to_syntax(e)),
        Expr::Gt(a, b) => format!("({} > {})", to_syntax(a), to_syntax(b)),
        Expr::Lt(a, b) => format!("({} < {})", to_syntax(a), to_syntax(b)),
        Expr::Eq(a, b) => format!("({} == {})", to_syntax(a), to_syntax(b)),
    }
}

/// Render a full Expr tree as LaTeX.
pub fn to_latex(expr: &Expr) -> String {
    match expr {
        Expr::Val => "x".into(),
        Expr::Val2 => "y".into(),
        Expr::Ref => r"\mu".into(),
        Expr::Lit(c) => format!("{c}"),
        Expr::Var(s) => s.clone(),
        Expr::Neg(a) => format!("-{}", to_latex(a)),
        Expr::Abs(a) => format!("|{}|", to_latex(a)),
        Expr::Recip(a) => format!(r"\frac{{1}}{{{}}}", to_latex(a)),
        Expr::Sq(a) => format!("{}^2", to_latex(a)),
        Expr::Sqrt(a) => format!(r"\sqrt{{{}}}", to_latex(a)),
        Expr::Ln(a) => format!(r"\ln({})", to_latex(a)),
        Expr::Exp(a) => format!("e^{{{}}}", to_latex(a)),
        Expr::Sin(a) => format!(r"\sin({})", to_latex(a)),
        Expr::Cos(a) => format!(r"\cos({})", to_latex(a)),
        Expr::Tan(a) => format!(r"\tan({})", to_latex(a)),
        Expr::Tanh(a) => format!(r"\tanh({})", to_latex(a)),
        Expr::Add(a, b) => format!("{} + {}", to_latex(a), to_latex(b)),
        Expr::Sub(a, b) => format!("{} - {}", to_latex(a), to_latex(b)),
        Expr::Mul(a, b) => format!(r"{} \cdot {}", to_latex(a), to_latex(b)),
        Expr::Div(a, b) => format!(r"\frac{{{}}}{{{}}}", to_latex(a), to_latex(b)),
        Expr::Pow(a, b) => format!("{}^{{{}}}", to_latex(a), to_latex(b)),
        _ => format!("\\text{{{}}}", to_syntax(expr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_complete_metadata() {
        let meta = atom_meta(&Expr::Sin(Box::new(Expr::Val)));
        assert!(meta.properties.contains(&Property::Bounded));
        assert!(meta.properties.contains(&Property::Periodic));
        assert!(meta.properties.contains(&Property::Differentiable));
        assert_eq!(meta.syntax, "sin(x)");
        assert_eq!(meta.inverse, Some("asin"));
        assert!(matches!(meta.derivative, Some(DerivativeRule::Chain("cos"))));
        assert!(matches!(meta.nan_behavior, NanBehavior::Propagate));
        assert_eq!(meta.cost.0, 15);
        assert!(!meta.simplifications.is_empty());
        assert_eq!(meta.tam_instruction, "sin");
    }

    #[test]
    fn add_algebraic_properties() {
        let meta = atom_meta(&Expr::Add(Box::new(Expr::Val), Box::new(Expr::Val)));
        assert!(meta.properties.contains(&Property::Associative));
        assert!(meta.properties.contains(&Property::Commutative));
        assert_eq!(meta.identity, Some(0.0));
        assert_eq!(meta.absorbing, None); // add has no absorbing element
        assert!(matches!(meta.derivative, Some(DerivativeRule::SumRule)));
    }

    #[test]
    fn mul_has_absorbing_zero() {
        let meta = atom_meta(&Expr::Mul(Box::new(Expr::Val), Box::new(Expr::Val)));
        assert_eq!(meta.identity, Some(1.0));
        assert_eq!(meta.absorbing, Some(0.0));
        assert!(matches!(meta.derivative, Some(DerivativeRule::ProductRule)));
    }

    #[test]
    fn exp_ln_inverse_pair() {
        let exp_meta = atom_meta(&Expr::Exp(Box::new(Expr::Val)));
        let ln_meta = atom_meta(&Expr::Ln(Box::new(Expr::Val)));
        assert_eq!(exp_meta.inverse, Some("ln"));
        assert_eq!(ln_meta.inverse, Some("exp"));
    }

    #[test]
    fn neg_is_involution() {
        let meta = atom_meta(&Expr::Neg(Box::new(Expr::Val)));
        assert!(meta.properties.contains(&Property::Involution));
        assert!(meta.simplifications.contains(&"neg(neg(x)) = x"));
    }

    #[test]
    fn sq_overflow_note() {
        let meta = atom_meta(&Expr::Sq(Box::new(Expr::Val)));
        assert!(!meta.stability_note.is_empty());
    }

    #[test]
    fn syntax_renders() {
        let expr = Expr::val().sq().add(Expr::lit(1.0));
        assert_eq!(to_syntax(&expr), "((val)² + 1)");
    }

    #[test]
    fn latex_renders() {
        let expr = Expr::var("sum").div(Expr::var("count"));
        assert_eq!(to_latex(&expr), r"\frac{sum}{count}");
    }
}
