//! Transforms: element-wise operations applied before accumulation.
//!
//! These are the φ (phi) in `scatter_phi`. Each transform takes one
//! or two input values and produces one output value. Transforms fuse
//! into the accumulate kernel — they don't cost a separate pass.
//!
//! On GPU: the transform becomes an inline expression in the kernel.
//! On CPU: the transform is a closure applied per element in the loop.

/// A transform applied to each element before accumulation.
/// This is the complete alphabet of element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    // === Unary: f(x) → y ===

    /// Identity: y = x. Passthrough.
    Identity,
    /// Constant: y = c. Ignores input. Used for counting (c=1.0).
    Const(f64),
    /// Square: y = x².
    Square,
    /// Cube: y = x³.
    Cube,
    /// Fourth power: y = x⁴.
    Fourth,
    /// General power: y = x^p.
    Pow(f64),
    /// Absolute value: y = |x|.
    Abs,
    /// Negation: y = -x.
    Neg,
    /// Reciprocal: y = 1/x.
    Reciprocal,
    /// Natural log: y = ln(x).
    Ln,
    /// Exponential: y = eˣ.
    Exp,
    /// Square root: y = √x.
    Sqrt,
    /// Sign: y = signum(x) ∈ {-1, 0, 1}.
    Sign,
    /// Floor: y = ⌊x⌋.
    Floor,
    /// Ceil: y = ⌈x⌉.
    Ceil,
    /// Clamp: y = clamp(x, lo, hi).
    Clamp(f64, f64),
    /// Indicator: y = 1.0 if x > threshold, else 0.0.
    GreaterThan(f64),
    /// Indicator: y = 1.0 if x < threshold, else 0.0.
    LessThan(f64),
    /// Is finite: y = 1.0 if x is finite, else 0.0.
    IsFinite,
    /// Is NaN: y = 1.0 if x is NaN, else 0.0.
    IsNaN,

    // === Binary: f(x, y) → z  (two-column operations) ===

    /// Multiply pair: z = x * y. For cross-products, covariance.
    MulPair,
    /// Subtract pair: z = x - y. For differences, residuals.
    SubPair,
    /// Add pair: z = x + y.
    AddPair,
    /// Squared difference: z = (x - y)². For distances, MSE.
    SqDiffPair,
    /// Absolute difference: z = |x - y|. For MAE, L1 distance.
    AbsDiffPair,

    // === Centered (requires a reference value, e.g. mean) ===

    /// Deviation from reference: y = x - ref.
    Deviation,
    /// Squared deviation: y = (x - ref)².
    SqDeviation,
    /// Absolute deviation: y = |x - ref|.
    AbsDeviation,
    /// Cubed deviation: y = (x - ref)³.
    CubedDeviation,
    /// Fourth deviation: y = (x - ref)⁴.
    FourthDeviation,
}

impl Transform {
    /// Apply this transform to a single value.
    /// `reference` is used by Deviation/SqDeviation/etc.
    /// `second` is used by binary transforms (MulPair, SubPair, etc).
    #[inline]
    pub fn apply(&self, x: f64, reference: f64, second: f64) -> f64 {
        match self {
            Transform::Identity     => x,
            Transform::Const(c)     => *c,
            Transform::Square       => x * x,
            Transform::Cube         => x * x * x,
            Transform::Fourth       => { let x2 = x * x; x2 * x2 },
            Transform::Pow(p)       => x.powf(*p),
            Transform::Abs          => x.abs(),
            Transform::Neg          => -x,
            Transform::Reciprocal   => 1.0 / x,
            Transform::Ln           => x.ln(),
            Transform::Exp          => x.exp(),
            Transform::Sqrt         => x.sqrt(),
            Transform::Sign         => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
            Transform::Floor        => x.floor(),
            Transform::Ceil         => x.ceil(),
            Transform::Clamp(lo, hi) => x.max(*lo).min(*hi),
            Transform::GreaterThan(t) => if x > *t { 1.0 } else { 0.0 },
            Transform::LessThan(t)  => if x < *t { 1.0 } else { 0.0 },
            Transform::IsFinite     => if x.is_finite() { 1.0 } else { 0.0 },
            Transform::IsNaN        => if x.is_nan() { 1.0 } else { 0.0 },

            Transform::MulPair      => x * second,
            Transform::SubPair      => x - second,
            Transform::AddPair      => x + second,
            Transform::SqDiffPair   => { let d = x - second; d * d },
            Transform::AbsDiffPair  => (x - second).abs(),

            Transform::Deviation      => x - reference,
            Transform::SqDeviation    => { let d = x - reference; d * d },
            Transform::AbsDeviation   => (x - reference).abs(),
            Transform::CubedDeviation => { let d = x - reference; d * d * d },
            Transform::FourthDeviation => { let d = x - reference; let d2 = d * d; d2 * d2 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() { assert_eq!(Transform::Identity.apply(3.0, 0.0, 0.0), 3.0); }

    #[test]
    fn const_one() { assert_eq!(Transform::Const(1.0).apply(999.0, 0.0, 0.0), 1.0); }

    #[test]
    fn square() { assert_eq!(Transform::Square.apply(3.0, 0.0, 0.0), 9.0); }

    #[test]
    fn ln_exp_roundtrip() {
        let x = 2.5;
        let result = Transform::Exp.apply(Transform::Ln.apply(x, 0.0, 0.0), 0.0, 0.0);
        assert!((result - x).abs() < 1e-14);
    }

    #[test]
    fn mul_pair() { assert_eq!(Transform::MulPair.apply(3.0, 0.0, 4.0), 12.0); }

    #[test]
    fn sq_deviation() {
        assert_eq!(Transform::SqDeviation.apply(5.0, 3.0, 0.0), 4.0); // (5-3)² = 4
    }

    #[test]
    fn clamp() {
        assert_eq!(Transform::Clamp(0.0, 1.0).apply(-0.5, 0.0, 0.0), 0.0);
        assert_eq!(Transform::Clamp(0.0, 1.0).apply(0.5, 0.0, 0.0), 0.5);
        assert_eq!(Transform::Clamp(0.0, 1.0).apply(1.5, 0.0, 0.0), 1.0);
    }

    #[test]
    fn indicator() {
        assert_eq!(Transform::GreaterThan(0.0).apply(1.0, 0.0, 0.0), 1.0);
        assert_eq!(Transform::GreaterThan(0.0).apply(-1.0, 0.0, 0.0), 0.0);
    }
}
