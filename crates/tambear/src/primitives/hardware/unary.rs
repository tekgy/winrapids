//! Unary floating-point primitives: absolute value, negation, sign copy.
//!
//! All three of these are single hardware instructions on every target.
//! They are bit-level operations — they flip or copy the sign bit of the
//! IEEE 754 representation and do not perform any arithmetic. They cannot
//! generate NaN from a non-NaN input.

/// Absolute value: clears the sign bit.
///
/// # Properties
/// - `fabs(NaN)` is NaN (any NaN — the payload may or may not change depending
///   on hardware, but the result is still NaN).
/// - `fabs(-0.0) == 0.0` (positive zero).
/// - `fabs(-inf) == inf`.
/// - Never generates NaN from a non-NaN input.
/// - `fabs(fabs(x)) == fabs(x)` (idempotent).
#[inline(always)]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Unary negation: flips the sign bit.
///
/// # Properties
/// - `fneg(NaN)` is NaN with flipped sign bit (IEEE 754 preserves payload,
///   flips sign). From the numerical consumer's perspective, it's still NaN.
/// - `fneg(0.0) == -0.0` and `fneg(-0.0) == 0.0`.
/// - `fneg(inf) == -inf`, `fneg(-inf) == inf`.
/// - `fneg(fneg(x)) == x` bitwise exactly (involution).
/// - Distinct from `fsub(0.0, x)` in the edge case: `fsub(0.0, 0.0) == 0.0`,
///   but `fneg(0.0) == -0.0`. This matters for some compensated arithmetic
///   patterns that care about the sign of zero.
#[inline(always)]
pub fn fneg(x: f64) -> f64 {
    -x
}

/// Copy sign: result has the magnitude of `magnitude` and the sign of `sign`.
///
/// `fcopysign(magnitude, sign)` returns `|magnitude|` with the sign bit taken
/// from `sign`.
///
/// # Properties
/// - `fcopysign(x, 1.0) == fabs(x)`.
/// - `fcopysign(x, -1.0) == -fabs(x)`.
/// - Works with NaN: `fcopysign(NaN, -1.0)` returns a NaN with the negative
///   sign bit set. (This is IEEE 754-defined even though NaN is "unsigned"
///   mathematically.)
/// - Works with ±0: `fcopysign(0.0, -1.0) == -0.0`.
///
/// # Consumers
/// Sign-preserving normalization, conditional negation without branching
/// (which is important for SIMD vectorization), implementation of `atan2`
/// and other quadrant-sensitive functions.
#[inline(always)]
pub fn fcopysign(magnitude: f64, sign: f64) -> f64 {
    magnitude.copysign(sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabs_basic() {
        assert_eq!(fabs(3.0), 3.0);
        assert_eq!(fabs(-3.0), 3.0);
        assert_eq!(fabs(0.0), 0.0);
    }

    #[test]
    fn fabs_negative_zero() {
        let result = fabs(-0.0);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_positive());
    }

    #[test]
    fn fabs_infinity() {
        assert_eq!(fabs(f64::INFINITY), f64::INFINITY);
        assert_eq!(fabs(f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn fabs_nan_stays_nan() {
        assert!(fabs(f64::NAN).is_nan());
    }

    #[test]
    fn fabs_idempotent() {
        let x = -3.14;
        assert_eq!(fabs(fabs(x)), fabs(x));
    }

    #[test]
    fn fneg_basic() {
        assert_eq!(fneg(3.0), -3.0);
        assert_eq!(fneg(-3.0), 3.0);
    }

    #[test]
    fn fneg_zero_signs() {
        // fneg flips the sign bit, even on zero.
        let pos_zero = 0.0_f64;
        let neg_result = fneg(pos_zero);
        assert_eq!(neg_result, 0.0);
        assert!(neg_result.is_sign_negative());

        let neg_zero = -0.0_f64;
        let pos_result = fneg(neg_zero);
        assert_eq!(pos_result, 0.0);
        assert!(pos_result.is_sign_positive());
    }

    #[test]
    fn fneg_infinity() {
        assert_eq!(fneg(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(fneg(f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn fneg_involution() {
        let values = [3.14, -2.71, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1e-300, 1e300];
        for &x in &values {
            assert_eq!(fneg(fneg(x)).to_bits(), x.to_bits(),
                "fneg(fneg({x:e})) should be bitwise-equal to {x:e}");
        }
    }

    #[test]
    fn fcopysign_basic() {
        assert_eq!(fcopysign(3.0, 1.0), 3.0);
        assert_eq!(fcopysign(3.0, -1.0), -3.0);
        assert_eq!(fcopysign(-3.0, 1.0), 3.0);
        assert_eq!(fcopysign(-3.0, -1.0), -3.0);
    }

    #[test]
    fn fcopysign_with_zero_magnitude() {
        // Magnitude zero, negative sign → negative zero.
        let result = fcopysign(0.0, -1.0);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn fcopysign_with_negative_zero_sign() {
        // Sign comes from -0.0, which has negative sign bit.
        let result = fcopysign(3.0, -0.0);
        assert_eq!(result, -3.0);
    }

    #[test]
    fn fcopysign_with_infinity() {
        assert_eq!(fcopysign(f64::INFINITY, -1.0), f64::NEG_INFINITY);
        assert_eq!(fcopysign(f64::NEG_INFINITY, 1.0), f64::INFINITY);
    }
}
