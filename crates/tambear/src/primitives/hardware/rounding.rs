//! Floating-point rounding primitives.
//!
//! These convert an f64 to the nearest representable integer value (still
//! stored as f64), using different rounding modes.
//!
//! # Rounding modes
//!
//! - `frint`: round to nearest, ties-to-even. This is IEEE 754's default
//!   "round half to even" (also called banker's rounding). Matches the
//!   rounding used by `fadd`/`fmul`/etc.
//! - `ffloor`: round toward negative infinity.
//! - `fceil`: round toward positive infinity.
//! - `ftrunc`: round toward zero.
//!
//! All four preserve NaN (NaN in → NaN out) and infinity (±inf in → ±inf out).

/// Round to nearest integer, ties-to-even (IEEE 754 default rounding).
///
/// `frint(0.5) == 0.0` (round half to even), `frint(1.5) == 2.0`,
/// `frint(2.5) == 2.0`, `frint(3.5) == 4.0`, `frint(-0.5) == 0.0`.
#[inline(always)]
pub fn frint(x: f64) -> f64 {
    x.round_ties_even()
}

/// Round toward negative infinity.
///
/// `ffloor(1.5) == 1.0`, `ffloor(-1.5) == -2.0`, `ffloor(0.0) == 0.0`.
#[inline(always)]
pub fn ffloor(x: f64) -> f64 {
    x.floor()
}

/// Round toward positive infinity.
///
/// `fceil(1.5) == 2.0`, `fceil(-1.5) == -1.0`, `fceil(0.0) == 0.0`.
#[inline(always)]
pub fn fceil(x: f64) -> f64 {
    x.ceil()
}

/// Round toward zero (truncate the fractional part).
///
/// `ftrunc(1.5) == 1.0`, `ftrunc(-1.5) == -1.0`, `ftrunc(0.0) == 0.0`.
#[inline(always)]
pub fn ftrunc(x: f64) -> f64 {
    x.trunc()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frint_ties_to_even() {
        assert_eq!(frint(0.5), 0.0);
        assert_eq!(frint(1.5), 2.0);
        assert_eq!(frint(2.5), 2.0);
        assert_eq!(frint(3.5), 4.0);
        assert_eq!(frint(-0.5), 0.0);
        assert_eq!(frint(-1.5), -2.0);
        assert_eq!(frint(-2.5), -2.0);
    }

    #[test]
    fn frint_non_ties() {
        assert_eq!(frint(0.4), 0.0);
        assert_eq!(frint(0.6), 1.0);
        assert_eq!(frint(-0.4), 0.0);
        assert_eq!(frint(-0.6), -1.0);
    }

    #[test]
    fn ffloor_basic() {
        assert_eq!(ffloor(1.5), 1.0);
        assert_eq!(ffloor(1.0), 1.0);
        assert_eq!(ffloor(0.5), 0.0);
        assert_eq!(ffloor(-0.5), -1.0);
        assert_eq!(ffloor(-1.5), -2.0);
    }

    #[test]
    fn fceil_basic() {
        assert_eq!(fceil(1.5), 2.0);
        assert_eq!(fceil(1.0), 1.0);
        assert_eq!(fceil(0.5), 1.0);
        assert_eq!(fceil(-0.5), 0.0);
        assert_eq!(fceil(-1.5), -1.0);
    }

    #[test]
    fn ftrunc_basic() {
        assert_eq!(ftrunc(1.5), 1.0);
        assert_eq!(ftrunc(1.9), 1.0);
        assert_eq!(ftrunc(-1.5), -1.0);
        assert_eq!(ftrunc(-1.9), -1.0);
        assert_eq!(ftrunc(0.0), 0.0);
    }

    #[test]
    fn rounding_preserves_nan() {
        assert!(frint(f64::NAN).is_nan());
        assert!(ffloor(f64::NAN).is_nan());
        assert!(fceil(f64::NAN).is_nan());
        assert!(ftrunc(f64::NAN).is_nan());
    }

    #[test]
    fn rounding_preserves_infinity() {
        assert_eq!(frint(f64::INFINITY), f64::INFINITY);
        assert_eq!(ffloor(f64::INFINITY), f64::INFINITY);
        assert_eq!(fceil(f64::INFINITY), f64::INFINITY);
        assert_eq!(ftrunc(f64::INFINITY), f64::INFINITY);

        assert_eq!(frint(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(ffloor(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(fceil(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(ftrunc(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn ftrunc_differs_from_ffloor_on_negatives() {
        // trunc toward zero, floor toward -inf
        assert_ne!(ftrunc(-1.5), ffloor(-1.5));
        assert_eq!(ftrunc(-1.5), -1.0);
        assert_eq!(ffloor(-1.5), -2.0);
    }
}
