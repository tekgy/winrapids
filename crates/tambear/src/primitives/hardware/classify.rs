//! Floating-point classification predicates.
//!
//! These return `bool` indicating what kind of value an f64 is: NaN, infinity,
//! finite, signed. They compile to single hardware instructions (usually a
//! bitwise check against the exponent and mantissa bits).
//!
//! All four are **total** — they always return a definite bool, never NaN.
//! They operate on the bit pattern and don't perform arithmetic.

/// Returns `true` if `x` is any NaN.
///
/// # Note
/// IEEE 754 allows many distinct NaN bit patterns (different payloads, signaling
/// vs quiet). This primitive returns `true` for all of them. Recipes almost
/// never care about the distinction.
#[inline(always)]
pub fn is_nan(x: f64) -> bool {
    x.is_nan()
}

/// Returns `true` if `x` is either positive or negative infinity.
///
/// Does not include NaN or any finite value.
#[inline(always)]
pub fn is_inf(x: f64) -> bool {
    x.is_infinite()
}

/// Returns `true` if `x` is a finite number (not NaN, not ±infinity).
///
/// Includes zero, subnormals, and all normal values.
///
/// # Usage
/// The preferred predicate for input validation. Instead of writing
/// `!x.is_nan() && !x.is_infinite()`, write `is_finite(x)`.
#[inline(always)]
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

/// Returns `true` if the sign bit of `x` is set (i.e., `x` is negative,
/// including `-0.0` and `-NaN`).
///
/// # Note
/// This checks the sign bit, not the mathematical sign. `signbit(-0.0)` is
/// `true` even though `-0.0 == 0.0`. `signbit(NaN)` depends on the bit pattern
/// of the specific NaN.
#[inline(always)]
pub fn signbit(x: f64) -> bool {
    x.is_sign_negative()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_nan_matches_nan() {
        assert!(is_nan(f64::NAN));
        assert!(!is_nan(1.0));
        assert!(!is_nan(0.0));
        assert!(!is_nan(-0.0));
        assert!(!is_nan(f64::INFINITY));
        assert!(!is_nan(f64::NEG_INFINITY));
    }

    #[test]
    fn is_inf_matches_infinities() {
        assert!(is_inf(f64::INFINITY));
        assert!(is_inf(f64::NEG_INFINITY));
        assert!(!is_inf(1.0));
        assert!(!is_inf(f64::NAN));
        assert!(!is_inf(f64::MAX));
    }

    #[test]
    fn is_finite_excludes_nan_and_inf() {
        assert!(is_finite(0.0));
        assert!(is_finite(-0.0));
        assert!(is_finite(1.0));
        assert!(is_finite(f64::MAX));
        assert!(is_finite(f64::MIN_POSITIVE));
        assert!(!is_finite(f64::NAN));
        assert!(!is_finite(f64::INFINITY));
        assert!(!is_finite(f64::NEG_INFINITY));
    }

    #[test]
    fn signbit_detects_negative() {
        assert!(!signbit(0.0));
        assert!(signbit(-0.0));
        assert!(!signbit(1.0));
        assert!(signbit(-1.0));
        assert!(!signbit(f64::INFINITY));
        assert!(signbit(f64::NEG_INFINITY));
    }
}
