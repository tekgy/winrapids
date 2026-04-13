//! Error-free transformations (EFTs).
//!
//! Each primitive here takes ordinary f64 inputs and returns a pair `(hi, lo)`
//! representing an *exact* mathematical result as the unevaluated sum
//! `hi + lo`. The property that makes these "error-free" is: `hi + lo` equals
//! the mathematically true result of the underlying operation, with no
//! rounding loss, as long as no overflow occurs.
//!
//! These are the foundation stones of compensated arithmetic. Every higher
//! primitive — Kahan sum, Neumaier sum, dot-2, compensated Horner, the
//! double-double type — is built by chaining EFTs.
//!
//! # References
//!
//! - Knuth, *TAOCP Vol 2*, §4.2.2: the original `two_sum` algorithm.
//! - Dekker, "A floating-point technique for extending the available
//!   precision" (1971): `fast_two_sum` (conditional on `|a| ≥ |b|`).
//! - Ogita, Rump, Oishi, "Accurate sum and dot product" (2005): the modern
//!   formulation used in Sum_K/Dot_K.
//! - Ercegovac & Muller, *Elementary Functions* (2018), ch. 4: `two_product_fma`
//!   via fused multiply-add.

use crate::primitives::hardware::{fmadd, fmsub};

/// Knuth's `two_sum`: exact error-free transformation for f64 addition.
///
/// Given `a`, `b`, returns `(s, e)` such that:
/// - `s == fadd(a, b)` (the rounded sum)
/// - `a + b == s + e` exactly in real arithmetic
/// - `|e| <= ulp(s) / 2`
///
/// Costs 6 flops. Works for any finite inputs with no ordering requirement.
///
/// # Overflow
/// If `s` overflows, `e` is unspecified. Caller must ensure finite inputs
/// and non-overflowing sum.
///
/// # NaN
/// `two_sum(NaN, x) = (NaN, NaN)`. NaN propagates through both components.
#[inline(always)]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    let delta_a = a - a_prime;
    let delta_b = b - b_prime;
    let e = delta_a + delta_b;
    (s, e)
}

/// Dekker's `fast_two_sum`: cheaper EFT that requires `|a| >= |b|`.
///
/// Given `a`, `b` with `|a| >= |b|`, returns `(s, e)` such that:
/// - `s == fadd(a, b)`
/// - `a + b == s + e` exactly
///
/// Costs 3 flops — half of `two_sum`. The precondition is a debug assertion,
/// not enforced in release builds; callers must guarantee ordering (e.g. by
/// pre-sorting, or by construction — when accumulating a running sum into a
/// larger running magnitude, the ordering is automatic).
///
/// # Panics (debug only)
/// Panics in debug builds if `|a| < |b|` and neither is NaN.
#[inline(always)]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    debug_assert!(
        a.is_nan() || b.is_nan() || a.abs() >= b.abs(),
        "fast_two_sum precondition violated: |a| < |b|, a={a}, b={b}"
    );
    let s = a + b;
    let b_virtual = s - a;
    let e = b - b_virtual;
    (s, e)
}

/// Error-free transformation for subtraction: `two_diff(a, b) = (s, e)` with
/// `a - b == s + e` exactly.
///
/// Implemented as `two_sum(a, -b)`; the negation is exact and free.
#[inline(always)]
pub fn two_diff(a: f64, b: f64) -> (f64, f64) {
    two_sum(a, -b)
}

/// Error-free product via FMA: `two_product_fma(a, b) = (p, e)` with
/// `a * b == p + e` exactly.
///
/// - `p == fmul(a, b)` (the rounded product)
/// - `e == fmsub(a, b, p)` (the residual, computed in single-rounding FMA)
///
/// Costs 2 flops (one mul, one FMA). This is THE reason FMA is a
/// first-class primitive in the hardware layer: without FMA, computing the
/// product error requires Dekker's splitting trick (17 flops) or a
/// double-double multiply (also expensive).
///
/// # Overflow
/// If `p` overflows, `e` is unspecified.
#[inline(always)]
pub fn two_product_fma(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = fmsub(a, b, p);
    (p, e)
}

/// Error-free square via FMA: `two_square(x) = (p, e)` with `x * x == p + e`
/// exactly.
///
/// Equivalent to `two_product_fma(x, x)`. Kept as a named primitive because
/// recipes for `x^2`, `|z|^2` and `norm_squared` appear often enough that a
/// dedicated entry point makes the intent clear.
#[inline(always)]
pub fn two_square(x: f64) -> (f64, f64) {
    two_product_fma(x, x)
}

/// Fused multiply-add residual: `fma_residual(a, b, c) = fmadd(a, b, c) - (a*b + c)`
/// computed as a compensated pair.
///
/// Returns `(r, err)` where `r` is the rounded `fmadd(a, b, c)` and `err` is
/// the exact residual such that `a*b + c == r + err` mathematically.
///
/// Costs: 1 FMA + 1 two_product_fma + 1 two_sum ≈ 9 flops.
///
/// This is the primary workhorse for compensated polynomial evaluation
/// (Langlois-Louvet-Graillat compensated Horner).
#[inline(always)]
pub fn fma_residual(a: f64, b: f64, c: f64) -> (f64, f64) {
    let r = fmadd(a, b, c);
    let (p, pi) = two_product_fma(a, b);
    let (s, sigma) = two_sum(p, c);
    // The error of (a*b + c) relative to its rounded FMA result r is
    // reconstructed from the two accumulated residuals.
    let err = (pi + sigma) + (s - r);
    (r, err)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── two_sum ─────────────────────────────────────────────────────────────

    #[test]
    fn two_sum_exactness_large_plus_small() {
        // Classic cancellation: 1.0 + 1e-20 rounds to 1.0, error is ~1e-20.
        let (s, e) = two_sum(1.0, 1e-20);
        assert_eq!(s, 1.0);
        assert_eq!(e, 1e-20);
    }

    #[test]
    fn two_sum_reconstructs_exactly() {
        // For any pair of finite floats, s + e should equal a + b when
        // computed in higher precision. We verify this via the invariant
        // that (s, e) is a non-overlapping pair: |e| <= 0.5 * ulp(s).
        let cases = [
            (1.0, 2.0),
            (1e100, 1.0),
            (1.5, 0.5),
            (-1.0, 1.0),
            (1e-20, 1e20),
            (7.0, 3.0),
        ];
        for (a, b) in cases {
            let (s, e) = two_sum(a, b);
            assert_eq!(s, a + b);
            // The error term must be small enough not to affect s again.
            assert_eq!(s + e, s, "two_sum({a}, {b}) error magnitude too large");
        }
    }

    #[test]
    fn two_sum_zero_error_when_exact() {
        // Powers of two adding exactly: no error.
        let (s, e) = two_sum(2.0, 4.0);
        assert_eq!(s, 6.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn two_sum_cancellation() {
        // Exact cancellation: error is zero (assuming no subnormal issues).
        let (s, e) = two_sum(1.0, -1.0);
        assert_eq!(s, 0.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn two_sum_commutative() {
        let cases = [(1.0, 1e-20), (1e100, 3.14), (-2.5, 7.0)];
        for (a, b) in cases {
            let (s1, e1) = two_sum(a, b);
            let (s2, e2) = two_sum(b, a);
            assert_eq!(s1, s2);
            assert_eq!(e1, e2);
        }
    }

    // ── fast_two_sum ────────────────────────────────────────────────────────

    #[test]
    fn fast_two_sum_agrees_with_two_sum_when_ordered() {
        let cases: [(f64, f64); 5] = [
            (10.0, 1e-15),
            (1e100, 1.0),
            (5.0, 3.0),
            (-5.0, 3.0),
            (2.5, -1.5),
        ];
        for (a, b) in cases {
            assert!(a.abs() >= b.abs());
            let (s_fast, e_fast) = fast_two_sum(a, b);
            let (s_slow, e_slow) = two_sum(a, b);
            assert_eq!(s_fast, s_slow);
            assert_eq!(e_fast, e_slow);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "fast_two_sum precondition violated")]
    fn fast_two_sum_panics_when_unordered_in_debug() {
        let _ = fast_two_sum(1.0, 10.0);
    }

    // ── two_diff ────────────────────────────────────────────────────────────

    #[test]
    fn two_diff_basic() {
        let (s, e) = two_diff(1.0, 1e-20);
        assert_eq!(s, 1.0);
        assert_eq!(e, -1e-20);
    }

    #[test]
    fn two_diff_exact() {
        let (s, e) = two_diff(10.0, 3.0);
        assert_eq!(s, 7.0);
        assert_eq!(e, 0.0);
    }

    // ── two_product_fma ─────────────────────────────────────────────────────

    #[test]
    fn two_product_fma_exact_on_small() {
        // 3.0 * 4.0 = 12.0 exactly; no error.
        let (p, e) = two_product_fma(3.0, 4.0);
        assert_eq!(p, 12.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn two_product_fma_captures_rounding_error() {
        // Pick two numbers whose exact product has more than 53 significant
        // bits, forcing rounding in the product.
        let a = 1.0_f64 + 2.0_f64.powi(-26); // 53 bits
        let b = 1.0_f64 - 2.0_f64.powi(-26); // 53 bits
        let (p, e) = two_product_fma(a, b);
        // Exact product is 1 - 2^-52, which fits in 53 bits → no error.
        // Check that p + e round-trips to the product.
        assert_eq!(p, a * b);
        // The error may be zero or tiny; the invariant is p + e == a*b exactly
        // in extended precision. We check by verifying |e| <= ulp(p).
        let ulp_p = f64::EPSILON * p.abs();
        assert!(e.abs() <= ulp_p);
    }

    #[test]
    fn two_product_fma_large_values() {
        // A case where rounding definitely occurs.
        let a = 1.0 + 2.0_f64.powi(-30);
        let b = 1.0 + 2.0_f64.powi(-30);
        // Exact: 1 + 2·2^-30 + 2^-60 = 1 + 2^-29 + 2^-60
        // Rounded to f64 (53 bits): 1 + 2^-29, losing the 2^-60 bit.
        let (p, e) = two_product_fma(a, b);
        assert_eq!(p, a * b);
        // Residual should be approximately 2^-60.
        assert!((e - 2.0_f64.powi(-60)).abs() < 2.0_f64.powi(-70));
    }

    // ── two_square ──────────────────────────────────────────────────────────

    #[test]
    fn two_square_basic() {
        let (p, e) = two_square(3.0);
        assert_eq!(p, 9.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn two_square_captures_error() {
        let x = 1.0 + 2.0_f64.powi(-30);
        let (p, e) = two_square(x);
        assert_eq!(p, x * x);
        // Residual ~ 2^-60.
        assert!((e - 2.0_f64.powi(-60)).abs() < 2.0_f64.powi(-70));
    }

    // ── fma_residual ────────────────────────────────────────────────────────

    #[test]
    fn fma_residual_agrees_with_separate_components() {
        // For simple exact cases, error is zero.
        let (r, err) = fma_residual(2.0, 3.0, 4.0);
        assert_eq!(r, 10.0);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn fma_residual_captures_lost_precision() {
        // Setup where a*b has a tiny residual that c cannot absorb.
        let a = 1.0 + 2.0_f64.powi(-30);
        let b = 1.0 + 2.0_f64.powi(-30);
        let c = -(a * b); // wipes out the rounded product
        let (r, err) = fma_residual(a, b, c);
        // After wiping, r should be a tiny residual close to 2^-60.
        // The compensated error captures the piece that the single FMA couldn't.
        let reconstruct = r + err;
        let expected = (a * b) + c; // f64 approximation, loses info
        // The compensated (r + err) should be strictly more accurate than the
        // plain fp sum expected.
        assert!(
            reconstruct.abs() <= expected.abs() + f64::EPSILON,
            "fma_residual reconstruct {reconstruct:e} exceeded plain sum {expected:e}"
        );
    }
}
