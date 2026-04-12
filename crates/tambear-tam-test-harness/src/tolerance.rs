//! Tolerance policy for cross-backend comparison (campsites 4.4 and 4.5).
//!
//! ## The two policies
//!
//! - [`ToleranceSpec::bit_exact()`]: `a.to_bits() == b.to_bits()`.  No tolerance.
//!   Used for all pure-arithmetic kernels (no transcendentals).
//!
//! - [`ToleranceSpec::within_ulp(bound)`]: computes the ULP distance between `a`
//!   and `b` and asserts it is ≤ `bound`.  Used only when a libm function is in
//!   the chain, and only at the ULP bound documented for that function (I9).
//!
//! ## Why bit-exact is the default
//!
//! Two backends executing the same sequence of `fadd`/`fmul`/`fsub`/`fdiv`/`fsqrt`
//! with the same input bits must produce the same output bits.  IEEE 754 mandates
//! correctly-rounded results for these operations.  Any divergence is a backend
//! bug — not a reason to widen the tolerance.
//!
//! ## Raising tolerances
//!
//! Raising a tolerance is a **red flag** (per trek invariants).  Before any call
//! to `within_ulp`, the caller must root-cause why bit-exactness is impossible.
//! The only valid reason is a documented libm ULP bound.

/// Comparison policy between two f64 outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToleranceSpec {
    /// Bit-level equality: `a.to_bits() == b.to_bits()`.
    ///
    /// Use for pure-arithmetic kernels.  Two backends with the same IEEE 754
    /// operations MUST agree here — divergence means a backend bug.
    BitExact,

    /// ULP-bounded: `ulp_distance(a, b) <= bound`.
    ///
    /// Use only when a libm transcendental is in the chain.
    /// `bound` must come from the function's documented ULP guarantee.
    WithinUlp { bound: u64 },
}

impl ToleranceSpec {
    /// Bit-exact policy — the default and the strictest.
    pub fn bit_exact() -> Self {
        Self::BitExact
    }

    /// ULP-bounded policy — only valid when libm transcendentals are in chain.
    ///
    /// `bound` should come directly from the relevant tambear-libm function's
    /// accuracy docs.  Never guess; never over-estimate.
    pub fn within_ulp(bound: u64) -> Self {
        Self::WithinUlp { bound }
    }

    /// Check whether `a` and `b` satisfy this tolerance.
    pub fn check(&self, a: f64, b: f64) -> bool {
        match self {
            Self::BitExact => a.to_bits() == b.to_bits(),
            Self::WithinUlp { bound } => ulp_distance(a, b) <= *bound,
        }
    }

    /// Describe the violation if `a` and `b` do not satisfy this tolerance.
    /// Returns `None` if they agree.
    pub fn describe_violation(&self, a: f64, b: f64) -> Option<String> {
        if self.check(a, b) {
            return None;
        }
        match self {
            Self::BitExact => Some(format!(
                "bit-exact violation: a={a:?} (0x{:016x}) vs b={b:?} (0x{:016x}), ulp_distance={}",
                a.to_bits(), b.to_bits(), ulp_distance(a, b)
            )),
            Self::WithinUlp { bound } => Some(format!(
                "ulp violation: a={a:?} vs b={b:?}, ulp_distance={}, bound={bound}",
                ulp_distance(a, b)
            )),
        }
    }
}

/// Compute the ULP (unit in the last place) distance between two f64 values.
///
/// ## Algorithm
///
/// IEEE 754 binary64 values are laid out such that adjacent representable values
/// differ by exactly one in their bit pattern when interpreted as a signed integer
/// (with the sign bit flipping the order for negatives).  The standard trick is:
///
/// 1. Reinterpret both bits as i64.
/// 2. Flip the sign bit if the value is negative, so the bit pattern is
///    monotonically increasing with the real-valued ordering.
/// 3. Subtract and take the absolute value.
///
/// ## Special cases
///
/// - `NaN` vs anything (including another `NaN`): returns `u64::MAX` — NaN
///   comparisons are always "maximally wrong" so tests never silently pass NaN.
/// - `+0.0` vs `-0.0`: returns 0 — they are the same value by IEEE 754.
/// - `+inf` vs `+inf`, `-inf` vs `-inf`: returns 0 — identical.
/// - `+inf` vs `-inf`: returns a very large number (not `u64::MAX`, but the
///   actual bit distance across the entire positive/negative range).
pub fn ulp_distance(a: f64, b: f64) -> u64 {
    // NaN is never equal to anything, including itself.
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }

    // Canonicalize -0.0 → +0.0 so that +0.0 and -0.0 compare as zero distance.
    let a = if a == 0.0 { 0.0_f64 } else { a };
    let b = if b == 0.0 { 0.0_f64 } else { b };

    let ai = ordered_bits(a);
    let bi = ordered_bits(b);
    ai.abs_diff(bi)
}

/// Map an f64 to its "signed integer order" so subtraction gives ULP distance.
///
/// IEEE 754 binary64 values have the property that their bit patterns, when
/// interpreted as sign-magnitude integers, are monotonically ordered with the
/// real-valued ordering.  We convert to two's complement here so that simple
/// subtraction gives the correct ULP distance.
fn ordered_bits(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        // Negative float: flip all bits except the sign bit to make the ordering
        // monotone (most-negative float → smallest integer).
        bits ^ i64::MAX
    } else {
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_exact_identical() {
        let spec = ToleranceSpec::bit_exact();
        assert!(spec.check(1.0_f64, 1.0_f64));
        assert!(spec.check(-3.14_f64, -3.14_f64));
        assert!(spec.check(f64::INFINITY, f64::INFINITY));
        assert!(spec.check(f64::NEG_INFINITY, f64::NEG_INFINITY));
    }

    #[test]
    fn bit_exact_differs_by_one_ulp() {
        let x = 1.0_f64;
        let y = f64::from_bits(x.to_bits() + 1);
        let spec = ToleranceSpec::bit_exact();
        assert!(!spec.check(x, y));
        assert!(spec.describe_violation(x, y).is_some());
    }

    #[test]
    fn zero_sign_is_irrelevant() {
        // +0.0 and -0.0 must have distance 0 (same value, different encoding)
        assert_eq!(ulp_distance(0.0_f64, -0.0_f64), 0);
    }

    #[test]
    fn nan_is_max_distance() {
        assert_eq!(ulp_distance(f64::NAN, 1.0), u64::MAX);
        assert_eq!(ulp_distance(1.0, f64::NAN), u64::MAX);
        assert_eq!(ulp_distance(f64::NAN, f64::NAN), u64::MAX);
    }

    #[test]
    fn adjacent_f64_ulp_one() {
        let x = 1.0_f64;
        let next = f64::from_bits(x.to_bits() + 1);
        assert_eq!(ulp_distance(x, next), 1);
    }

    #[test]
    fn within_ulp_policy() {
        let x = 1.0_f64;
        let next2 = f64::from_bits(x.to_bits() + 2);
        let spec = ToleranceSpec::within_ulp(2);
        assert!(spec.check(x, next2));

        let next3 = f64::from_bits(x.to_bits() + 3);
        assert!(!spec.check(x, next3));
    }

    #[test]
    fn ulp_symmetric() {
        let x = 2.718281828_f64;
        let y = f64::from_bits(x.to_bits() + 5);
        assert_eq!(ulp_distance(x, y), ulp_distance(y, x));
    }

    #[test]
    fn negative_values_ordered_correctly() {
        // -1.0 should be exactly 1 ULP from -nextafter(-1.0, 0)
        let a = -1.0_f64;
        let b = f64::from_bits(a.to_bits() - 1); // adjacent negative (less negative)
        assert_eq!(ulp_distance(a, b), 1);
    }

    #[test]
    fn identical_inf_zero_distance() {
        assert_eq!(ulp_distance(f64::INFINITY, f64::INFINITY), 0);
        assert_eq!(ulp_distance(f64::NEG_INFINITY, f64::NEG_INFINITY), 0);
    }
}
