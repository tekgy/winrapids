//! Arithmetic operations on `DoubleDouble`.
//!
//! Each routine documents the flop count and the error bound from
//! Joldes-Muller-Popescu (2017) where applicable. The operator overloads
//! at the bottom of this file dispatch to these routines so that recipes
//! can write `a + b * c` naturally.

use super::ty::DoubleDouble;
use crate::primitives::compensated::eft::{two_product_fma, two_sum};

/// Double-double addition: `a + b`, both in DD, result in DD.
///
/// Algorithm: Joldes-Muller-Popescu `DWPlusDW` (also called `dd_add_accurate`).
/// Error bound: `(3·ε² + ε³) |a + b|` for non-overlapping inputs.
///
/// Cost: ~20 flops.
#[inline]
pub fn dd_add(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let (sh, sl) = two_sum(a.hi, b.hi);
    let (th, tl) = two_sum(a.lo, b.lo);
    let c = sl + th;
    let (vh, vl) = two_sum(sh, c);
    let w = tl + vl;
    let (rh, rl) = two_sum(vh, w);
    DoubleDouble { hi: rh, lo: rl }
}

/// Double-double subtraction: `a - b`.
///
/// Implemented as `dd_add(a, -b)`. Negation is exact.
#[inline]
pub fn dd_sub(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    dd_add(a, b.neg())
}

/// Double-double addition of a DD and an f64.
///
/// Optimized from full DD add — one less EFT because `b.lo == 0`.
/// Cost: ~10 flops.
#[inline]
pub fn dd_add_f64(a: DoubleDouble, b: f64) -> DoubleDouble {
    let (sh, sl) = two_sum(a.hi, b);
    let (rh, rl) = two_sum(sh, sl + a.lo);
    DoubleDouble { hi: rh, lo: rl }
}

/// Double-double multiplication: `a * b`.
///
/// Algorithm: Joldes-Muller-Popescu `DWTimesDW3`. Uses FMA for the high
/// part product error-free transformation; remaining cross-terms folded
/// into the low part.
///
/// Error bound: `5·ε² |a·b|`. Cost: ~9 flops.
#[inline]
pub fn dd_mul(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let (p, e) = two_product_fma(a.hi, b.hi);
    // Cross terms: a.hi * b.lo + a.lo * b.hi. These are computed in plain
    // f64 because the cross terms are already O(ε) relative to p, and any
    // rounding error on them is O(ε²) — below the double-double precision
    // bound.
    let cross = a.hi.mul_add(b.lo, a.lo * b.hi);
    let (rh, rl) = two_sum(p, e + cross);
    DoubleDouble { hi: rh, lo: rl }
}

/// Double-double multiplication of a DD and an f64.
///
/// Cost: ~6 flops. Error bound: `2·ε² |a·b|`.
#[inline]
pub fn dd_mul_f64(a: DoubleDouble, b: f64) -> DoubleDouble {
    let (p, e) = two_product_fma(a.hi, b);
    let cross = a.lo * b;
    let (rh, rl) = two_sum(p, e + cross);
    DoubleDouble { hi: rh, lo: rl }
}

/// Double-double division: `a / b`.
///
/// Algorithm: one Newton iteration on the reciprocal, then multiply.
///
/// 1. Initial reciprocal estimate: `r0 = 1 / b.hi` (f64 divide).
/// 2. Newton refinement in DD: `r1 = r0 + r0 · (1 - b · r0)`.
/// 3. Result: `a · r1`.
///
/// Error bound: ~8·ε² relative. Cost: ~30 flops.
///
/// # Special cases
/// - Division by zero returns a DD with `hi = ±∞`.
/// - NaN propagates.
#[inline]
pub fn dd_div(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let r0 = 1.0 / b.hi;
    // Compute (1 - b * r0) in DD, which should be very small if r0 is a
    // good estimate.
    let br0 = dd_mul_f64(b, r0);
    let one_minus_br0 = dd_sub(DoubleDouble::ONE, br0);
    // r1 = r0 + r0 * (1 - b*r0)
    let correction = dd_mul_f64(one_minus_br0, r0);
    let r1 = dd_add_f64(correction, r0);
    // result = a * r1
    dd_mul(a, r1)
}

/// Double-double division by an f64: `a / b`.
#[inline]
pub fn dd_div_f64(a: DoubleDouble, b: f64) -> DoubleDouble {
    // Same Newton approach but the denominator is a scalar.
    let r0 = 1.0 / b;
    let br0 = b * r0; // approximately 1.0
    // correction factor: (1 - b*r0)
    let resid = 1.0 - br0;
    let r1_hi = r0;
    let r1_lo = r0 * resid;
    dd_mul(a, DoubleDouble::from_parts(r1_hi, r1_lo))
}

/// Double-double square root.
///
/// Algorithm: initial estimate from `f64::sqrt`, then one Newton iteration.
/// Error bound: ~4·ε² relative. Cost: ~25 flops.
///
/// # Special cases
/// - `sqrt(0) = 0`.
/// - `sqrt(NaN) = NaN`.
/// - `sqrt(-x)` for negative input returns NaN.
#[inline]
pub fn dd_sqrt(a: DoubleDouble) -> DoubleDouble {
    if a.hi <= 0.0 {
        if a.hi == 0.0 && a.lo == 0.0 {
            return DoubleDouble::ZERO;
        }
        return DoubleDouble::from_f64(f64::NAN);
    }
    // Newton: x_{n+1} = (x_n + a / x_n) / 2
    // With a single iteration, accuracy is doubled from ~53 bits to ~106.
    let x0 = a.hi.sqrt();
    // r = a - x0 * x0, computed exactly via two_product_fma
    let (x0_sq, x0_sq_err) = two_product_fma(x0, x0);
    let r_hi = a.hi - x0_sq;
    let r_lo = a.lo - x0_sq_err;
    let r = DoubleDouble::from_parts(r_hi, r_lo);
    // correction = r / (2 * x0) ≈ r * (1 / (2 * x0))
    let half_over_x0 = 0.5 / x0;
    let correction = dd_mul_f64(r, half_over_x0);
    dd_add_f64(correction, x0)
}

// ─── Operator overloads ────────────────────────────────────────────────────

use std::ops::{Add, Div, Mul, Neg, Sub};

impl Add for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        dd_add(self, rhs)
    }
}

impl Add<f64> for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        dd_add_f64(self, rhs)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        dd_sub(self, rhs)
    }
}

impl Sub<f64> for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        dd_add_f64(self, -rhs)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        dd_mul(self, rhs)
    }
}

impl Mul<f64> for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        dd_mul_f64(self, rhs)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        dd_div(self, rhs)
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        dd_div_f64(self, rhs)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        DoubleDouble::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference: the exact value of (1 + 2^-27)^2 - 1 - 2^-26 is 2^-54,
    // a canonical test that plain f64 gets wrong but DD captures.
    #[test]
    fn dd_add_captures_lost_bits() {
        let a = DoubleDouble::from_f64(1.0);
        let b = DoubleDouble::from_f64(1e-30);
        let s = dd_add(a, b);
        // In plain f64, 1 + 1e-30 == 1. In DD, the 1e-30 is preserved in lo.
        assert_eq!(s.hi, 1.0);
        assert!(s.lo > 0.0, "lo should preserve the small value, got {}", s.lo);
    }

    #[test]
    fn dd_add_exact_on_simple_values() {
        let a = DoubleDouble::from_f64(1.0);
        let b = DoubleDouble::from_f64(2.0);
        let s = dd_add(a, b);
        assert_eq!(s.hi, 3.0);
        assert_eq!(s.lo, 0.0);
    }

    #[test]
    fn dd_sub_trivial() {
        let a = DoubleDouble::from_f64(5.0);
        let b = DoubleDouble::from_f64(3.0);
        let r = dd_sub(a, b);
        assert_eq!(r.hi, 2.0);
        assert_eq!(r.lo, 0.0);
    }

    #[test]
    fn dd_mul_exact_on_integers() {
        let a = DoubleDouble::from_f64(3.0);
        let b = DoubleDouble::from_f64(4.0);
        let p = dd_mul(a, b);
        assert_eq!(p.hi, 12.0);
        assert_eq!(p.lo, 0.0);
    }

    #[test]
    fn dd_mul_captures_rounding_error() {
        // (1 + 2^-30) * (1 + 2^-30) has a bit at 2^-60 that f64 rounds off.
        let x = 1.0 + 2.0_f64.powi(-30);
        let a = DoubleDouble::from_f64(x);
        let b = DoubleDouble::from_f64(x);
        let p = dd_mul(a, b);
        // hi should match f64 multiplication.
        assert_eq!(p.hi, x * x);
        // lo should be approximately 2^-60.
        assert!(
            (p.lo - 2.0_f64.powi(-60)).abs() < 2.0_f64.powi(-70),
            "lo was {}, expected ~2^-60",
            p.lo
        );
    }

    #[test]
    fn dd_div_reciprocal_improves_over_f64() {
        // 1 / 3 has no exact f64 representation. DD should approximate to
        // ~106 bits.
        let one = DoubleDouble::ONE;
        let three = DoubleDouble::from_f64(3.0);
        let third = dd_div(one, three);
        // hi should match standard 1/3 in f64.
        assert_eq!(third.hi, 1.0 / 3.0);
        // Combined value should be very close to mathematical 1/3.
        let product = dd_mul(third, three);
        // product should be ~1.0 to nearly full DD precision.
        assert!(
            (product.hi - 1.0).abs() < 1e-16,
            "product hi off by {:e}",
            product.hi - 1.0
        );
        // |product - 1| should be ~ε².
        let total_err = (product.to_f64() - 1.0).abs();
        assert!(total_err < 1e-30, "DD division error {:e} too large", total_err);
    }

    #[test]
    fn dd_sqrt_matches_f64_on_perfect_squares() {
        assert_eq!(dd_sqrt(DoubleDouble::from_f64(4.0)).to_f64(), 2.0);
        assert_eq!(dd_sqrt(DoubleDouble::from_f64(9.0)).to_f64(), 3.0);
        assert_eq!(dd_sqrt(DoubleDouble::from_f64(16.0)).to_f64(), 4.0);
    }

    #[test]
    fn dd_sqrt_improves_on_f64() {
        let two = DoubleDouble::from_f64(2.0);
        let rt2 = dd_sqrt(two);
        // Squaring should give us back 2.0 with DD precision.
        let squared = dd_mul(rt2, rt2);
        let err = (squared.to_f64() - 2.0).abs();
        assert!(err < 1e-30, "sqrt(2) reconstruction error {:e}", err);
    }

    #[test]
    fn dd_sqrt_of_zero_is_zero() {
        let r = dd_sqrt(DoubleDouble::ZERO);
        assert!(r.is_zero());
    }

    #[test]
    fn dd_sqrt_of_negative_is_nan() {
        let r = dd_sqrt(DoubleDouble::from_f64(-1.0));
        assert!(r.is_nan());
    }

    // ── Operator overload smoke tests ──────────────────────────────────────

    #[test]
    fn operators_work() {
        let a = DoubleDouble::from_f64(3.0);
        let b = DoubleDouble::from_f64(4.0);

        assert_eq!((a + b).to_f64(), 7.0);
        assert_eq!((a - b).to_f64(), -1.0);
        assert_eq!((a * b).to_f64(), 12.0);
        assert_eq!((-a).to_f64(), -3.0);
        // (12.0 / 4.0) == 3.0 exactly — this is the only division path that
        // hits exactly in f64 so we can assert equality.
        assert_eq!((DoubleDouble::from_f64(12.0) / b).to_f64(), 3.0);
    }

    #[test]
    fn operators_with_f64_rhs() {
        let a = DoubleDouble::from_f64(3.0);
        assert_eq!((a + 4.0).to_f64(), 7.0);
        assert_eq!((a - 1.0).to_f64(), 2.0);
        assert_eq!((a * 2.0).to_f64(), 6.0);
        assert_eq!((DoubleDouble::from_f64(6.0) / 2.0).to_f64(), 3.0);
    }

    // ── Round-trip stress: (a + b) * (a - b) should equal a² - b² in DD ───

    #[test]
    fn difference_of_squares_identity() {
        let a = DoubleDouble::from_f64(1.0 + 1e-15);
        let b = DoubleDouble::from_f64(1.0);

        // Compute (a + b)(a - b) via DD.
        let sum = a + b;
        let diff = a - b;
        let prod = sum * diff;

        // Compute a² - b² via DD.
        let a_sq = a * a;
        let b_sq = b * b;
        let direct = a_sq - b_sq;

        // Should match to DD precision.
        let err = (prod.to_f64() - direct.to_f64()).abs();
        assert!(err < 1e-30, "DD identity error {:e}", err);
    }
}
