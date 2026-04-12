//! TBS: the expression language for all of tambear.
//!
//! One AST type. Used everywhere:
//! - As the φ (transform) before accumulate
//! - As the gather formula after accumulate
//! - As the user's script in the IDE
//! - As the input to the .tam compiler
//!
//! TBS expressions compile to .tam IR, which dispatches through
//! vendor driver kits to any ALU. No CUDA strings. No WGSL.
//! No vendor language anywhere. Our language, our compiler, our IR.

pub mod metadata;
pub mod shape;

/// A TBS expression: the universal AST node.
///
/// Every mathematical expression in tambear is a tree of these nodes.
/// The tree is what the compiler optimizes, fuses, and emits as
/// machine code for whatever ALU TAM chooses.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // === Leaf nodes ===

    /// The current element value (sugar for Col(0)).
    Val,
    /// A second column value (sugar for Col(1)).
    Val2,
    /// Any column by index. Col(0) = Val, Col(1) = Val2, Col(n) = nth column.
    /// Allows expressions over arbitrary numbers of columns.
    Col(usize),
    /// A reference value (per-group mean, running value, etc).
    /// Set by the accumulate context — e.g. group mean for centering.
    Ref,
    /// A literal constant.
    Lit(f64),
    /// A named variable (reference to a prior accumulate output).
    Var(String),

    // === Unary operations ===

    /// Negation: -x
    Neg(Box<Expr>),
    /// Absolute value: |x|
    Abs(Box<Expr>),
    /// Reciprocal: 1/x
    Recip(Box<Expr>),
    /// Square: x²
    Sq(Box<Expr>),
    /// Square root: √x
    Sqrt(Box<Expr>),
    /// Natural log: ln(x)
    Ln(Box<Expr>),
    /// Exponential: eˣ
    Exp(Box<Expr>),
    /// Floor: ⌊x⌋
    Floor(Box<Expr>),
    /// Ceil: ⌈x⌉
    Ceil(Box<Expr>),
    /// Sign: -1, 0, or 1
    Sign(Box<Expr>),
    /// Is finite: 1.0 if finite, 0.0 otherwise
    IsFinite(Box<Expr>),

    // === Trigonometric ===

    /// Sine
    Sin(Box<Expr>),
    /// Cosine
    Cos(Box<Expr>),
    /// Tangent
    Tan(Box<Expr>),
    /// Arcsine
    Asin(Box<Expr>),
    /// Arccosine
    Acos(Box<Expr>),
    /// Arctangent
    Atan(Box<Expr>),
    /// Hyperbolic sine
    Sinh(Box<Expr>),
    /// Hyperbolic cosine
    Cosh(Box<Expr>),
    /// Hyperbolic tangent
    Tanh(Box<Expr>),

    // === Rounding / Modular ===

    /// Round to nearest integer
    Round(Box<Expr>),
    /// Truncate toward zero
    Trunc(Box<Expr>),

    // === Binary operations ===

    /// Addition: a + b
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: a - b
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: a * b
    Mul(Box<Expr>, Box<Expr>),
    /// Division: a / b
    Div(Box<Expr>, Box<Expr>),
    /// Power: a^b
    Pow(Box<Expr>, Box<Expr>),
    /// Min: min(a, b)
    Min(Box<Expr>, Box<Expr>),
    /// Max: max(a, b)
    Max(Box<Expr>, Box<Expr>),
    /// Clamp: clamp(x, lo, hi)
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Two-argument arctangent: atan2(y, x)
    Atan2(Box<Expr>, Box<Expr>),
    /// Modulo: a % b
    Mod(Box<Expr>, Box<Expr>),
    /// Conditional: if cond > 0 then a else b
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    // === Comparison (returns 0.0 or 1.0, NaN propagates) ===

    /// a > b → 1.0, else 0.0. NaN in either operand → NaN.
    Gt(Box<Expr>, Box<Expr>),
    /// a < b → 1.0, else 0.0. NaN in either operand → NaN.
    Lt(Box<Expr>, Box<Expr>),
    /// a == b → 1.0, else 0.0. NaN in either operand → NaN.
    Eq(Box<Expr>, Box<Expr>),
}

// === Convenience constructors ===

impl Expr {
    /// val (the input element, sugar for col(0))
    pub fn val() -> Self { Expr::Val }
    /// val2 (second column, sugar for col(1))
    pub fn val2() -> Self { Expr::Val2 }
    /// nth column reference
    pub fn col(n: usize) -> Self { Expr::Col(n) }
    /// reference value
    pub fn r#ref() -> Self { Expr::Ref }
    /// literal constant
    pub fn lit(c: f64) -> Self { Expr::Lit(c) }
    /// named variable
    pub fn var(name: &str) -> Self { Expr::Var(name.to_string()) }

    /// x²
    pub fn sq(self) -> Self { Expr::Sq(Box::new(self)) }
    /// √x
    pub fn sqrt(self) -> Self { Expr::Sqrt(Box::new(self)) }
    /// ln(x)
    pub fn ln(self) -> Self { Expr::Ln(Box::new(self)) }
    /// eˣ
    pub fn exp(self) -> Self { Expr::Exp(Box::new(self)) }
    /// |x|
    pub fn abs(self) -> Self { Expr::Abs(Box::new(self)) }
    /// 1/x
    pub fn recip(self) -> Self { Expr::Recip(Box::new(self)) }
    /// -x
    pub fn neg(self) -> Self { Expr::Neg(Box::new(self)) }

    /// a + b
    pub fn add(self, other: Expr) -> Self { Expr::Add(Box::new(self), Box::new(other)) }
    /// a - b
    pub fn sub(self, other: Expr) -> Self { Expr::Sub(Box::new(self), Box::new(other)) }
    /// a * b
    pub fn mul(self, other: Expr) -> Self { Expr::Mul(Box::new(self), Box::new(other)) }
    /// a / b
    pub fn div(self, other: Expr) -> Self { Expr::Div(Box::new(self), Box::new(other)) }
    /// a^b
    pub fn pow(self, other: Expr) -> Self { Expr::Pow(Box::new(self), Box::new(other)) }
    /// sin(x)
    pub fn sin(self) -> Self { Expr::Sin(Box::new(self)) }
    /// cos(x)
    pub fn cos(self) -> Self { Expr::Cos(Box::new(self)) }
    /// tan(x)
    pub fn tan(self) -> Self { Expr::Tan(Box::new(self)) }
    /// asin(x)
    pub fn asin(self) -> Self { Expr::Asin(Box::new(self)) }
    /// acos(x)
    pub fn acos(self) -> Self { Expr::Acos(Box::new(self)) }
    /// atan(x)
    pub fn atan(self) -> Self { Expr::Atan(Box::new(self)) }
    /// sinh(x)
    pub fn sinh(self) -> Self { Expr::Sinh(Box::new(self)) }
    /// cosh(x)
    pub fn cosh(self) -> Self { Expr::Cosh(Box::new(self)) }
    /// tanh(x)
    pub fn tanh(self) -> Self { Expr::Tanh(Box::new(self)) }
    /// round(x)
    pub fn round(self) -> Self { Expr::Round(Box::new(self)) }
    /// trunc(x)
    pub fn trunc(self) -> Self { Expr::Trunc(Box::new(self)) }
    /// atan2(y, x)
    pub fn atan2(self, other: Expr) -> Self { Expr::Atan2(Box::new(self), Box::new(other)) }
    /// a % b
    pub fn modulo(self, other: Expr) -> Self { Expr::Mod(Box::new(self), Box::new(other)) }
}

// === Evaluation ===

use std::collections::HashMap;

/// Evaluate a TBS expression given variable bindings.
/// `val` = column 0, `val2` = column 1. For Col(n) with n >= 2,
/// pass additional columns via the `cols` parameter in eval_multi.
pub fn eval(expr: &Expr, val: f64, val2: f64, reference: f64, vars: &HashMap<String, f64>) -> f64 {
    match expr {
        Expr::Val => val,
        Expr::Val2 => val2,
        Expr::Col(0) => val,
        Expr::Col(1) => val2,
        Expr::Col(_) => f64::NAN, // use eval_multi for 3+ columns
        Expr::Ref => reference,
        Expr::Lit(c) => *c,
        Expr::Var(name) => *vars.get(name.as_str()).unwrap_or(&f64::NAN),

        Expr::Neg(a) => -eval(a, val, val2, reference, vars),
        Expr::Abs(a) => eval(a, val, val2, reference, vars).abs(),
        Expr::Recip(a) => 1.0 / eval(a, val, val2, reference, vars),
        Expr::Sq(a) => { let v = eval(a, val, val2, reference, vars); v * v },
        Expr::Sqrt(a) => eval(a, val, val2, reference, vars).sqrt(),
        Expr::Ln(a) => eval(a, val, val2, reference, vars).ln(),
        Expr::Exp(a) => eval(a, val, val2, reference, vars).exp(),
        Expr::Floor(a) => eval(a, val, val2, reference, vars).floor(),
        Expr::Ceil(a) => eval(a, val, val2, reference, vars).ceil(),
        Expr::Sign(a) => {
            let v = eval(a, val, val2, reference, vars);
            if v.is_nan() { v } else if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
        }
        Expr::IsFinite(a) => if eval(a, val, val2, reference, vars).is_finite() { 1.0 } else { 0.0 },

        Expr::Sin(a)  => eval(a, val, val2, reference, vars).sin(),
        Expr::Cos(a)  => eval(a, val, val2, reference, vars).cos(),
        Expr::Tan(a)  => eval(a, val, val2, reference, vars).tan(),
        Expr::Asin(a) => eval(a, val, val2, reference, vars).asin(),
        Expr::Acos(a) => eval(a, val, val2, reference, vars).acos(),
        Expr::Atan(a) => eval(a, val, val2, reference, vars).atan(),
        Expr::Sinh(a) => eval(a, val, val2, reference, vars).sinh(),
        Expr::Cosh(a) => eval(a, val, val2, reference, vars).cosh(),
        Expr::Tanh(a) => eval(a, val, val2, reference, vars).tanh(),
        Expr::Round(a) => eval(a, val, val2, reference, vars).round(),
        Expr::Trunc(a) => eval(a, val, val2, reference, vars).trunc(),

        Expr::Add(a, b) => eval(a, val, val2, reference, vars) + eval(b, val, val2, reference, vars),
        Expr::Sub(a, b) => eval(a, val, val2, reference, vars) - eval(b, val, val2, reference, vars),
        Expr::Mul(a, b) => eval(a, val, val2, reference, vars) * eval(b, val, val2, reference, vars),
        Expr::Div(a, b) => eval(a, val, val2, reference, vars) / eval(b, val, val2, reference, vars),
        Expr::Pow(a, b) => eval(a, val, val2, reference, vars).powf(eval(b, val, val2, reference, vars)),
        Expr::Min(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va.is_nan() || vb.is_nan() { f64::NAN } else if va <= vb { va } else { vb }
        }
        Expr::Max(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va.is_nan() || vb.is_nan() { f64::NAN } else if va >= vb { va } else { vb }
        }
        Expr::Clamp(x, lo, hi) => {
            let vx = eval(x, val, val2, reference, vars);
            let vlo = eval(lo, val, val2, reference, vars);
            let vhi = eval(hi, val, val2, reference, vars);
            vx.max(vlo).min(vhi)
        }

        Expr::Atan2(a, b) => eval(a, val, val2, reference, vars).atan2(eval(b, val, val2, reference, vars)),
        Expr::Mod(a, b) => eval(a, val, val2, reference, vars) % eval(b, val, val2, reference, vars),
        Expr::If(cond, then, els) => {
            if eval(cond, val, val2, reference, vars) > 0.0 {
                eval(then, val, val2, reference, vars)
            } else {
                eval(els, val, val2, reference, vars)
            }
        }

        Expr::Gt(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va.is_nan() || vb.is_nan() { f64::NAN } else if va > vb { 1.0 } else { 0.0 }
        }
        Expr::Lt(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va.is_nan() || vb.is_nan() { f64::NAN } else if va < vb { 1.0 } else { 0.0 }
        }
        Expr::Eq(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va.is_nan() || vb.is_nan() { f64::NAN } else if va == vb { 1.0 } else { 0.0 }
        }
    }
}

/// Evaluate with arbitrary number of input columns.
/// `cols[0]` = val, `cols[1]` = val2, `cols[n]` = Col(n).
/// Use this when expressions reference 3+ columns.
pub fn eval_multi(expr: &Expr, cols: &[f64], reference: f64, vars: &HashMap<String, f64>) -> f64 {
    let get_col = |n: usize| cols.get(n).copied().unwrap_or(f64::NAN);
    let val = get_col(0);
    let val2 = get_col(1);

    match expr {
        Expr::Col(n) => get_col(*n),
        // All recursive calls also need multi-column context
        Expr::Neg(a) => -eval_multi(a, cols, reference, vars),
        Expr::Abs(a) => eval_multi(a, cols, reference, vars).abs(),
        Expr::Sq(a) => { let v = eval_multi(a, cols, reference, vars); v * v },
        Expr::Mul(a, b) => eval_multi(a, cols, reference, vars) * eval_multi(b, cols, reference, vars),
        Expr::Add(a, b) => eval_multi(a, cols, reference, vars) + eval_multi(b, cols, reference, vars),
        Expr::Sub(a, b) => eval_multi(a, cols, reference, vars) - eval_multi(b, cols, reference, vars),
        Expr::Div(a, b) => eval_multi(a, cols, reference, vars) / eval_multi(b, cols, reference, vars),
        // For non-recursive leaf/simple cases, delegate to eval
        _ => eval(expr, val, val2, reference, vars),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_vars() -> HashMap<String, f64> { HashMap::new() }

    #[test]
    fn val_returns_input() {
        assert_eq!(eval(&Expr::Val, 42.0, 0.0, 0.0, &empty_vars()), 42.0);
    }

    #[test]
    fn literal() {
        assert_eq!(eval(&Expr::lit(3.14), 0.0, 0.0, 0.0, &empty_vars()), 3.14);
    }

    #[test]
    fn square_of_val() {
        // val.sq() = val²
        assert_eq!(eval(&Expr::val().sq(), 5.0, 0.0, 0.0, &empty_vars()), 25.0);
    }

    #[test]
    fn sum_of_two_columns() {
        // val * val2 (cross product)
        let expr = Expr::val().mul(Expr::val2());
        assert_eq!(eval(&expr, 3.0, 4.0, 0.0, &empty_vars()), 12.0);
    }

    #[test]
    fn deviation_squared() {
        // (val - ref)²
        let expr = Expr::val().sub(Expr::r#ref()).sq();
        assert_eq!(eval(&expr, 5.0, 0.0, 3.0, &empty_vars()), 4.0);
    }

    #[test]
    fn variance_gather_formula() {
        // (sum_sq - sum * sum / count) / (count - 1)
        let mut vars = HashMap::new();
        vars.insert("sum".to_string(), 15.0);
        vars.insert("sum_sq".to_string(), 55.0);
        vars.insert("count".to_string(), 5.0);

        let expr = Expr::var("sum_sq")
            .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
            .div(Expr::var("count").sub(Expr::lit(1.0)));

        let result = eval(&expr, 0.0, 0.0, 0.0, &vars);
        assert!((result - 2.5).abs() < 1e-14, "variance = {result}");
    }

    #[test]
    fn geometric_mean_gather() {
        // exp(log_sum / count)
        let mut vars = HashMap::new();
        vars.insert("log_sum".to_string(), 3.0_f64.ln() * 3.0);
        vars.insert("count".to_string(), 3.0);

        let expr = Expr::var("log_sum").div(Expr::var("count")).exp();
        let result = eval(&expr, 0.0, 0.0, 0.0, &vars);
        assert!((result - 3.0).abs() < 1e-14);
    }

    #[test]
    fn chain_transforms() {
        // |ln(val)| — composable, no special enum variant needed
        let expr = Expr::val().ln().abs();
        let result = eval(&expr, 0.5, 0.0, 0.0, &empty_vars());
        assert!((result - 0.5_f64.ln().abs()).abs() < 1e-14);
    }

    #[test]
    fn clamp_works() {
        let expr = Expr::Clamp(
            Box::new(Expr::val()),
            Box::new(Expr::lit(0.0)),
            Box::new(Expr::lit(1.0)),
        );
        assert_eq!(eval(&expr, -0.5, 0.0, 0.0, &empty_vars()), 0.0);
        assert_eq!(eval(&expr, 0.5, 0.0, 0.0, &empty_vars()), 0.5);
        assert_eq!(eval(&expr, 1.5, 0.0, 0.0, &empty_vars()), 1.0);
    }

    #[test]
    fn sin_cos_identity() {
        // sin²(x) + cos²(x) = 1
        let x = 1.7;
        let s = eval(&Expr::val().sin(), x, 0.0, 0.0, &empty_vars());
        let c = eval(&Expr::val().cos(), x, 0.0, 0.0, &empty_vars());
        assert!((s * s + c * c - 1.0).abs() < 1e-14);
    }

    #[test]
    fn tanh_range() {
        // tanh is in (-1, 1)
        let result = eval(&Expr::val().tanh(), 100.0, 0.0, 0.0, &empty_vars());
        assert!((result - 1.0).abs() < 1e-10);
        let result = eval(&Expr::val().tanh(), -100.0, 0.0, 0.0, &empty_vars());
        assert!((result + 1.0).abs() < 1e-10);
    }

    #[test]
    fn atan2_quadrants() {
        let pi = std::f64::consts::PI;
        let r = eval(&Expr::val().atan2(Expr::val2()), 1.0, 1.0, 0.0, &empty_vars());
        assert!((r - pi / 4.0).abs() < 1e-14); // 45 degrees
    }

    #[test]
    fn modulo() {
        let r = eval(&Expr::val().modulo(Expr::lit(3.0)), 7.0, 0.0, 0.0, &empty_vars());
        assert!((r - 1.0).abs() < 1e-14); // 7 % 3 = 1
    }

    #[test]
    fn conditional_if() {
        // ReLU: if x > 0 then x else 0
        let relu = Expr::If(
            Box::new(Expr::Gt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)))),
            Box::new(Expr::val()),
            Box::new(Expr::lit(0.0)),
        );
        assert_eq!(eval(&relu, 5.0, 0.0, 0.0, &empty_vars()), 5.0);
        assert_eq!(eval(&relu, -3.0, 0.0, 0.0, &empty_vars()), 0.0);
    }

    #[test]
    fn round_and_trunc() {
        assert_eq!(eval(&Expr::val().round(), 2.7, 0.0, 0.0, &empty_vars()), 3.0);
        assert_eq!(eval(&Expr::val().trunc(), 2.7, 0.0, 0.0, &empty_vars()), 2.0);
        assert_eq!(eval(&Expr::val().trunc(), -2.7, 0.0, 0.0, &empty_vars()), -2.0);
    }

    #[test]
    fn indicator_gt() {
        let expr = Expr::Gt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)));
        assert_eq!(eval(&expr, 1.0, 0.0, 0.0, &empty_vars()), 1.0);
        assert_eq!(eval(&expr, -1.0, 0.0, 0.0, &empty_vars()), 0.0);
    }

    #[test]
    fn count_positives() {
        // For each element: 1.0 if x > 0, else 0.0. Then accumulate with Add = count_positive.
        let expr = Expr::Gt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)));
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let count_pos: f64 = data.iter()
            .map(|&x| eval(&expr, x, 0.0, 0.0, &empty_vars()))
            .sum();
        assert_eq!(count_pos, 2.0);
    }

    // Oracle tests for comparison NaN propagation (P16/P18 verification).
    // IEEE 754 §6.2: NaN is "not a number" — any operation that receives NaN
    // as input and returns a numeric result must propagate NaN. Returning 0.0
    // (false) for Gt/Lt/Eq when an operand is NaN would silently swallow the
    // NaN and let the downstream If-branch execute as if the comparison was
    // valid. This is a correctness violation: NaN means "unknown", not "false".
    //
    // Note: Gt/Lt/Eq are NOT IEEE 754 comparison predicates (which return
    // boolean unordered). They are TBS numeric predicates returning f64 values
    // 0.0 or 1.0. In this context, propagating NaN is correct because the NaN
    // carries the "unknown" signal through the expression tree.

    #[test]
    fn gt_nan_propagates() {
        let expr = Expr::Gt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)));
        assert!(eval(&expr, f64::NAN, 0.0, 0.0, &empty_vars()).is_nan(),
            "Gt(NaN, 0) must return NaN, not 0.0");
        let expr2 = Expr::Gt(Box::new(Expr::lit(1.0)), Box::new(Expr::val()));
        assert!(eval(&expr2, f64::NAN, 0.0, 0.0, &empty_vars()).is_nan(),
            "Gt(1, NaN) must return NaN, not 0.0");
    }

    #[test]
    fn lt_nan_propagates() {
        let expr = Expr::Lt(Box::new(Expr::val()), Box::new(Expr::lit(0.0)));
        assert!(eval(&expr, f64::NAN, 0.0, 0.0, &empty_vars()).is_nan(),
            "Lt(NaN, 0) must return NaN, not 0.0");
    }

    #[test]
    fn eq_nan_propagates() {
        // Eq(NaN, NaN) must return NaN, not 1.0 (would require to_bits() equality, which is wrong
        // semantics — two NaN values from different operations are not "equal") and not 0.0
        // (would swallow the NaN signal).
        let expr = Expr::Eq(Box::new(Expr::val()), Box::new(Expr::val()));
        assert!(eval(&expr, f64::NAN, 0.0, 0.0, &empty_vars()).is_nan(),
            "Eq(NaN, NaN) must return NaN, not 0.0 or 1.0");
        let expr2 = Expr::Eq(Box::new(Expr::val()), Box::new(Expr::lit(5.0)));
        assert!(eval(&expr2, f64::NAN, 0.0, 0.0, &empty_vars()).is_nan(),
            "Eq(NaN, 5) must return NaN, not 0.0");
    }

    #[test]
    fn eq_exact_for_nearby_floats() {
        // Eq uses exact equality (==), not epsilon. Two floats 1 ULP apart are NOT equal.
        // This is the fix for P18: the old 1e-15 epsilon made 1-ULP differences compare as equal.
        let x: f64 = 1.0;
        let x_plus_1ulp = f64::from_bits(x.to_bits() + 1);
        assert!(x_plus_1ulp > x, "sanity: next float after 1.0 is larger");
        let expr = Expr::Eq(Box::new(Expr::val()), Box::new(Expr::lit(x)));
        assert_eq!(eval(&expr, x_plus_1ulp, 0.0, 0.0, &empty_vars()), 0.0,
            "Eq must use exact equality: 1 ULP apart is not equal");
        assert_eq!(eval(&expr, x, 0.0, 0.0, &empty_vars()), 1.0,
            "Eq(x, x) must be 1.0");
    }
}
