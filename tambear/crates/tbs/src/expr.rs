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

/// A TBS expression: the universal AST node.
///
/// Every mathematical expression in tambear is a tree of these nodes.
/// The tree is what the compiler optimizes, fuses, and emits as
/// machine code for whatever ALU TAM chooses.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // === Leaf nodes ===

    /// The current element value (the implicit input to a transform).
    /// In a scatter context, this is the value being scattered.
    Val,
    /// A second column value (for binary/cross-column operations).
    /// In Pearson r: Val is x, Val2 is y.
    Val2,
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

    // === Comparison (returns 0.0 or 1.0) ===

    /// a > b → 1.0, else 0.0
    Gt(Box<Expr>, Box<Expr>),
    /// a < b → 1.0, else 0.0
    Lt(Box<Expr>, Box<Expr>),
    /// a == b → 1.0, else 0.0 (within epsilon)
    Eq(Box<Expr>, Box<Expr>),
}

// === Convenience constructors ===

impl Expr {
    /// val (the input element)
    pub fn val() -> Self { Expr::Val }
    /// val2 (second column)
    pub fn val2() -> Self { Expr::Val2 }
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
}

// === Evaluation ===

use std::collections::HashMap;

/// Evaluate a TBS expression given variable bindings.
pub fn eval(expr: &Expr, val: f64, val2: f64, reference: f64, vars: &HashMap<String, f64>) -> f64 {
    match expr {
        Expr::Val => val,
        Expr::Val2 => val2,
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
            if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
        }
        Expr::IsFinite(a) => if eval(a, val, val2, reference, vars).is_finite() { 1.0 } else { 0.0 },

        Expr::Add(a, b) => eval(a, val, val2, reference, vars) + eval(b, val, val2, reference, vars),
        Expr::Sub(a, b) => eval(a, val, val2, reference, vars) - eval(b, val, val2, reference, vars),
        Expr::Mul(a, b) => eval(a, val, val2, reference, vars) * eval(b, val, val2, reference, vars),
        Expr::Div(a, b) => eval(a, val, val2, reference, vars) / eval(b, val, val2, reference, vars),
        Expr::Pow(a, b) => eval(a, val, val2, reference, vars).powf(eval(b, val, val2, reference, vars)),
        Expr::Min(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va <= vb { va } else { vb }
        }
        Expr::Max(a, b) => {
            let va = eval(a, val, val2, reference, vars);
            let vb = eval(b, val, val2, reference, vars);
            if va >= vb { va } else { vb }
        }
        Expr::Clamp(x, lo, hi) => {
            let vx = eval(x, val, val2, reference, vars);
            let vlo = eval(lo, val, val2, reference, vars);
            let vhi = eval(hi, val, val2, reference, vars);
            vx.max(vlo).min(vhi)
        }

        Expr::Gt(a, b) => if eval(a, val, val2, reference, vars) > eval(b, val, val2, reference, vars) { 1.0 } else { 0.0 },
        Expr::Lt(a, b) => if eval(a, val, val2, reference, vars) < eval(b, val, val2, reference, vars) { 1.0 } else { 0.0 },
        Expr::Eq(a, b) => {
            let diff = (eval(a, val, val2, reference, vars) - eval(b, val, val2, reference, vars)).abs();
            if diff < 1e-15 { 1.0 } else { 0.0 }
        }
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
}
