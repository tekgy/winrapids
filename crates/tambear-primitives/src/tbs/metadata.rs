//! Metadata for Expr atoms: syntax, notation, properties, derivatives.
//!
//! Each Expr variant is self-describing. The metadata is what the IDE,
//! the proof engine, the notation renderer, and the symbolic differentiator
//! use to reason about expressions.

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
}

/// Domain restriction for an Expr node.
#[derive(Debug, Clone)]
pub enum Domain {
    /// All reals
    AllReals,
    /// x > 0 (strictly positive)
    Positive,
    /// x ≥ 0
    NonNegative,
    /// x ∈ [-1, 1]
    UnitInterval,
    /// x ≠ 0
    NonZero,
    /// Arbitrary constraint description
    Custom(&'static str),
}

/// Metadata for a single Expr variant.
#[derive(Debug, Clone)]
pub struct AtomMeta {
    /// TBS syntax: how you type it
    pub syntax: &'static str,
    /// LaTeX notation: how it renders in math
    pub latex: &'static str,
    /// Tambear formalism: how the compiler sees it
    pub tambear: &'static str,
    /// Input domain restriction
    pub domain: Domain,
    /// Output range description
    pub range: &'static str,
    /// Algebraic/analytic properties
    pub properties: &'static [Property],
}

/// Get metadata for an Expr variant.
/// Only handles the node type, not the full tree.
pub fn atom_meta(expr: &Expr) -> AtomMeta {
    match expr {
        // === Leaves ===
        Expr::Val => AtomMeta {
            syntax: "val", latex: "x", tambear: "v",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
        },
        Expr::Val2 => AtomMeta {
            syntax: "val2", latex: "y", tambear: "w",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
        },
        Expr::Ref => AtomMeta {
            syntax: "ref", latex: r"\mu", tambear: "r",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
        },
        Expr::Lit(_) => AtomMeta {
            syntax: "lit(c)", latex: "c", tambear: "c",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
        },
        Expr::Var(_) => AtomMeta {
            syntax: "var(name)", latex: "name", tambear: "name",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[],
        },

        // === Unary ===
        Expr::Neg(_) => AtomMeta {
            syntax: "neg(x)", latex: "-x", tambear: "−v",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicDecreasing, Property::Involution],
        },
        Expr::Abs(_) => AtomMeta {
            syntax: "abs(x)", latex: "|x|", tambear: "|v|",
            domain: Domain::AllReals, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Bounded],
        },
        Expr::Recip(_) => AtomMeta {
            syntax: "recip(x)", latex: r"\frac{1}{x}", tambear: "1/v",
            domain: Domain::NonZero, range: "ℝ\\{0}",
            properties: &[Property::Continuous, Property::Differentiable, Property::Involution],
        },
        Expr::Sq(_) => AtomMeta {
            syntax: "sq(x)", latex: "x^2", tambear: "v²",
            domain: Domain::AllReals, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Differentiable, Property::PositivePreserving],
        },
        Expr::Sqrt(_) => AtomMeta {
            syntax: "sqrt(x)", latex: r"\sqrt{x}", tambear: "√v",
            domain: Domain::NonNegative, range: "ℝ≥0",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::PositivePreserving],
        },
        Expr::Ln(_) => AtomMeta {
            syntax: "ln(x)", latex: r"\ln(x)", tambear: "ln v",
            domain: Domain::Positive, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing],
        },
        Expr::Exp(_) => AtomMeta {
            syntax: "exp(x)", latex: "e^x", tambear: "eᵛ",
            domain: Domain::AllReals, range: "ℝ>0",
            properties: &[Property::Continuous, Property::Differentiable, Property::MonotonicIncreasing, Property::PositivePreserving],
        },
        Expr::Sin(_) => AtomMeta {
            syntax: "sin(x)", latex: r"\sin(x)", tambear: "sin v",
            domain: Domain::AllReals, range: "[-1,1]",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::Periodic],
        },
        Expr::Cos(_) => AtomMeta {
            syntax: "cos(x)", latex: r"\cos(x)", tambear: "cos v",
            domain: Domain::AllReals, range: "[-1,1]",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::Periodic],
        },
        Expr::Tan(_) => AtomMeta {
            syntax: "tan(x)", latex: r"\tan(x)", tambear: "tan v",
            domain: Domain::Custom("ℝ \\ {π/2 + kπ}"), range: "ℝ",
            properties: &[Property::Differentiable, Property::Periodic],
        },
        Expr::Tanh(_) => AtomMeta {
            syntax: "tanh(x)", latex: r"\tanh(x)", tambear: "tanh v",
            domain: Domain::AllReals, range: "(-1,1)",
            properties: &[Property::Continuous, Property::Differentiable, Property::Bounded, Property::MonotonicIncreasing],
        },
        Expr::Sign(_) => AtomMeta {
            syntax: "sign(x)", latex: r"\text{sgn}(x)", tambear: "sgn v",
            domain: Domain::AllReals, range: "{-1,0,1}",
            properties: &[Property::Bounded, Property::Idempotent],
        },
        Expr::Floor(_) => AtomMeta {
            syntax: "floor(x)", latex: r"\lfloor x \rfloor", tambear: "⌊v⌋",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::MonotonicIncreasing, Property::Idempotent],
        },
        Expr::Ceil(_) => AtomMeta {
            syntax: "ceil(x)", latex: r"\lceil x \rceil", tambear: "⌈v⌉",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::MonotonicIncreasing, Property::Idempotent],
        },
        Expr::Round(_) => AtomMeta {
            syntax: "round(x)", latex: r"\text{round}(x)", tambear: "⌊v⌉",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::Idempotent],
        },
        Expr::Trunc(_) => AtomMeta {
            syntax: "trunc(x)", latex: r"\text{trunc}(x)", tambear: "trunc v",
            domain: Domain::AllReals, range: "ℤ",
            properties: &[Property::Idempotent],
        },
        Expr::IsFinite(_) => AtomMeta {
            syntax: "is_finite(x)", latex: r"\mathbb{1}_{x \in \mathbb{R}}", tambear: "fin? v",
            domain: Domain::AllReals, range: "{0,1}",
            properties: &[Property::Bounded, Property::Idempotent],
        },

        // === Binary ===
        Expr::Add(_, _) => AtomMeta {
            syntax: "a + b", latex: "a + b", tambear: "a + b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::Associative, Property::Commutative],
        },
        Expr::Sub(_, _) => AtomMeta {
            syntax: "a - b", latex: "a - b", tambear: "a − b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable],
        },
        Expr::Mul(_, _) => AtomMeta {
            syntax: "a * b", latex: r"a \cdot b", tambear: "a × b",
            domain: Domain::AllReals, range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable, Property::Associative, Property::Commutative],
        },
        Expr::Div(_, _) => AtomMeta {
            syntax: "a / b", latex: r"\frac{a}{b}", tambear: "a ÷ b",
            domain: Domain::Custom("b ≠ 0"), range: "ℝ",
            properties: &[Property::Continuous, Property::Differentiable],
        },
        Expr::Pow(_, _) => AtomMeta {
            syntax: "a ^ b", latex: "a^b", tambear: "aᵇ",
            domain: Domain::Custom("a > 0 or b ∈ ℤ"), range: "ℝ",
            properties: &[Property::Differentiable],
        },
        Expr::Mod(_, _) => AtomMeta {
            syntax: "a % b", latex: r"a \bmod b", tambear: "a mod b",
            domain: Domain::Custom("b ≠ 0"), range: "ℝ",
            properties: &[Property::Periodic],
        },
        Expr::Atan2(_, _) => AtomMeta {
            syntax: "atan2(y, x)", latex: r"\text{atan2}(y, x)", tambear: "atan2(a, b)",
            domain: Domain::Custom("not both zero"), range: "(-π, π]",
            properties: &[Property::Continuous, Property::Bounded],
        },

        // Remaining variants get a default
        _ => AtomMeta {
            syntax: "?", latex: "?", tambear: "?",
            domain: Domain::AllReals, range: "?",
            properties: &[],
        },
    }
}

/// Render a full Expr tree as TBS syntax.
pub fn to_syntax(expr: &Expr) -> String {
    match expr {
        Expr::Val => "val".into(),
        Expr::Val2 => "val2".into(),
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
        _ => format!("?({:?})", expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_properties() {
        let meta = atom_meta(&Expr::Sin(Box::new(Expr::Val)));
        assert!(meta.properties.contains(&Property::Bounded));
        assert!(meta.properties.contains(&Property::Periodic));
        assert!(meta.properties.contains(&Property::Differentiable));
        assert_eq!(meta.syntax, "sin(x)");
    }

    #[test]
    fn add_is_associative_commutative() {
        let meta = atom_meta(&Expr::Add(Box::new(Expr::Val), Box::new(Expr::Val)));
        assert!(meta.properties.contains(&Property::Associative));
        assert!(meta.properties.contains(&Property::Commutative));
    }

    #[test]
    fn syntax_renders() {
        let expr = Expr::val().sq().add(Expr::lit(1.0));
        let syntax = to_syntax(&expr);
        assert_eq!(syntax, "((val)² + 1)");
    }

    #[test]
    fn latex_renders() {
        let expr = Expr::var("sum").div(Expr::var("count"));
        let latex = to_latex(&expr);
        assert_eq!(latex, r"\frac{sum}{count}");
    }

    #[test]
    fn variance_latex() {
        let expr = Expr::var("sum_sq")
            .sub(Expr::var("sum").sq().div(Expr::var("n")))
            .div(Expr::var("n").sub(Expr::lit(1.0)));
        let latex = to_latex(&expr);
        // Should render as a fraction
        assert!(latex.contains(r"\frac"));
    }

    #[test]
    fn relu_syntax() {
        let relu = Expr::If(
            Box::new(Expr::Gt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)))),
            Box::new(Expr::val()),
            Box::new(Expr::lit(0.0)),
        );
        let syntax = to_syntax(&relu);
        assert_eq!(syntax, "if((val > 0), val, 0)");
    }

    #[test]
    fn ln_domain_positive() {
        let meta = atom_meta(&Expr::Ln(Box::new(Expr::Val)));
        assert!(matches!(meta.domain, Domain::Positive));
    }
}
