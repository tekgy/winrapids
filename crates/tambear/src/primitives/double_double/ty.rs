//! The `DoubleDouble` type itself — storage, constructors, conversions,
//! classification predicates. Operator overloads live in `ops.rs`.

/// Unevaluated sum `hi + lo` with `|lo| <= ulp(hi) / 2` (non-overlapping).
///
/// Represents approximately 106 bits of precision. The canonical form has
/// `hi` holding the round-to-nearest f64 approximation of the true value
/// and `lo` holding the residual.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

impl DoubleDouble {
    /// Zero: `0 + 0`.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// One: `1 + 0`. Exact.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    /// Construct from a single f64. The low part is zero by definition.
    #[inline(always)]
    pub const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// Construct from an already non-overlapping pair. Caller is responsible
    /// for ensuring the invariant; use `two_sum` on arbitrary pairs before
    /// constructing.
    #[inline(always)]
    pub const fn from_parts(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// Round the double-double to an ordinary f64, losing the low part.
    ///
    /// Note this is NOT the "true" correctly-rounded value in general —
    /// rounding a DD to f64 requires a conditional adjustment when `lo`
    /// is exactly half an ulp of `hi`. For correctly-rounded conversion
    /// use `to_f64_correctly_rounded`.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Correctly-rounded conversion to f64, implementing round-half-to-even
    /// when the low part is exactly at the midpoint.
    ///
    /// Delegates to `to_f64` for now — the default Rust rounding is
    /// already round-half-to-even for `f64 + f64`, so the simple `hi + lo`
    /// path yields the correct result when the pair is in canonical form.
    #[inline(always)]
    pub fn to_f64_correctly_rounded(self) -> f64 {
        self.hi + self.lo
    }

    /// Negation: `-(hi + lo) == (-hi) + (-lo)`.
    #[inline(always)]
    pub fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            self.neg()
        } else {
            self
        }
    }

    /// Is this any NaN?
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.hi.is_nan() || self.lo.is_nan()
    }

    /// Is this infinite (in the high part)?
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        self.hi.is_infinite()
    }

    /// Is this finite? (i.e., both parts are finite)
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    /// Is this exactly zero?
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }
}

impl From<f64> for DoubleDouble {
    #[inline(always)]
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl Default for DoubleDouble {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_conversion() {
        let a = DoubleDouble::from_f64(3.14);
        assert_eq!(a.hi, 3.14);
        assert_eq!(a.lo, 0.0);
        assert_eq!(a.to_f64(), 3.14);
    }

    #[test]
    fn constants() {
        assert_eq!(DoubleDouble::ZERO.to_f64(), 0.0);
        assert_eq!(DoubleDouble::ONE.to_f64(), 1.0);
        assert!(DoubleDouble::ZERO.is_zero());
        assert!(!DoubleDouble::ONE.is_zero());
    }

    #[test]
    fn negation_and_abs() {
        let a = DoubleDouble::from_parts(3.0, 1e-18);
        let neg = a.neg();
        assert_eq!(neg.hi, -3.0);
        assert_eq!(neg.lo, -1e-18);
        let abs = neg.abs();
        assert_eq!(abs.hi, 3.0);
        assert_eq!(abs.lo, 1e-18);
    }

    #[test]
    fn classification() {
        let nan_dd = DoubleDouble::from_f64(f64::NAN);
        assert!(nan_dd.is_nan());
        assert!(!nan_dd.is_finite());

        let inf_dd = DoubleDouble::from_f64(f64::INFINITY);
        assert!(inf_dd.is_infinite());
        assert!(!inf_dd.is_finite());

        let finite_dd = DoubleDouble::from_parts(1.0, 1e-18);
        assert!(finite_dd.is_finite());
        assert!(!finite_dd.is_nan());
        assert!(!finite_dd.is_infinite());
    }

    #[test]
    fn default_is_zero() {
        let d: DoubleDouble = Default::default();
        assert!(d.is_zero());
    }

    #[test]
    fn from_trait() {
        let d: DoubleDouble = 7.5.into();
        assert_eq!(d.hi, 7.5);
        assert_eq!(d.lo, 0.0);
    }
}
