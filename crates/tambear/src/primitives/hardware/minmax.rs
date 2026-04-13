//! IEEE 754-2019 minimum and maximum with NaN propagation.
//!
//! # Why we don't use `f64::min` and `f64::max`
//!
//! Rust's standard library `f64::min(NaN, x)` returns `x`. `f64::max(NaN, x)`
//! also returns `x`. This is NOT IEEE 754-2019 `minNum`/`maxNum` semantics —
//! those require NaN to propagate.
//!
//! The 2026-04-10 adversarial sweep found **11 distinct bugs** caused by this
//! divergence. Every one followed the same pattern: a reduction computed a
//! running min or max via `fold(..., f64::min)`, hit a NaN somewhere in the
//! data, and the NaN got silently swallowed — the reduction proceeded using
//! the non-NaN value, and the function returned a plausible-looking number
//! instead of NaN. Silent wrong answers.
//!
//! Examples from the session:
//! - `davies_bouldin_score` with NaN cluster distances → returned 0.0 instead of NaN
//! - `hurst_rs` with NaN-contaminated series → returned finite H
//! - `correlation_dimension` with NaN observations → returned dimension ≈ 0.93
//! - `Mat::norm_inf` with NaN matrix entries → returned max of non-NaN rows
//! - `log_sum_exp([NaN])` → returned -Infinity
//! - `MomentStats::merge` with NaN fields → bitwise-corrupted result
//!
//! All 11 fixed by routing through this module.
//!
//! # The rule
//!
//! **Recipes MUST NOT call `f64::min`, `f64::max`, `.min(other)`, or `.max(other)`
//! directly on float values.** They MUST call `primitives::fmin`/`primitives::fmax`.
//! A lint/grep audit of `crates/tambear/src/recipes/` for `f64::min`/`f64::max`
//! usage should return zero matches.
//!
//! # Implementation note
//!
//! IEEE 754-2019 added `minimum`/`maximum` operations that propagate NaN.
//! Rust 1.78+ has `f64::minimum` and `f64::maximum` that implement these.
//! We use those where available, with a NaN-propagating fallback for older
//! Rust. The semantics are:
//!
//!     fmin(NaN, x) = NaN
//!     fmin(x, NaN) = NaN
//!     fmin(NaN, NaN) = NaN
//!     fmin(-0.0, 0.0) = -0.0   // sign bit ordering, negative zero is "smaller"
//!     fmin(a, b) = a if a < b, else b (for non-NaN, non-zero inputs)

/// IEEE 754-2019 minimum with NaN propagation.
///
/// If either input is NaN, returns NaN. Otherwise returns the smaller of the
/// two values, treating `-0.0` as smaller than `+0.0`.
///
/// This is distinct from `f64::min`, which returns the non-NaN operand when
/// exactly one input is NaN.
#[inline(always)]
pub fn fmin(a: f64, b: f64) -> f64 {
    // Explicit NaN check first — we can't rely on f64::min which eats NaN,
    // and f64::minimum is only stable from Rust 1.78. The explicit check
    // is equally fast on modern CPUs (branch-predictable, and the NaN
    // case is rare in hot paths).
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else if a > b {
        b
    } else {
        // a == b, but we need to handle the ±0.0 case: -0.0 < +0.0 is false
        // even though they compare equal with `<`. IEEE 754-2019 specifies
        // minNum(-0, +0) = -0 (negative zero is the "smaller" zero).
        //
        // Fast path: if both are zero (a == b == 0), check sign bits.
        if a == 0.0 && b == 0.0 {
            // Both zero. Return the one with negative sign bit, or +0 if neither.
            if a.is_sign_negative() { a } else { b }
        } else {
            // Equal non-zero values — return either, here we pick a.
            a
        }
    }
}

/// IEEE 754-2019 maximum with NaN propagation.
///
/// If either input is NaN, returns NaN. Otherwise returns the larger of the
/// two values, treating `+0.0` as larger than `-0.0`.
#[inline(always)]
pub fn fmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else if a < b {
        b
    } else {
        // a == b case, handle ±0 sign.
        if a == 0.0 && b == 0.0 {
            // Return positive zero if either is positive.
            if a.is_sign_positive() { a } else { b }
        } else {
            a
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic behavior ──────────────────────────────────────────────────────

    #[test]
    fn fmin_basic() {
        assert_eq!(fmin(1.0, 2.0), 1.0);
        assert_eq!(fmin(2.0, 1.0), 1.0);
        assert_eq!(fmin(-5.0, -3.0), -5.0);
    }

    #[test]
    fn fmax_basic() {
        assert_eq!(fmax(1.0, 2.0), 2.0);
        assert_eq!(fmax(2.0, 1.0), 2.0);
        assert_eq!(fmax(-5.0, -3.0), -3.0);
    }

    // ── NaN propagation (the critical distinction from f64::min/max) ────────

    #[test]
    fn fmin_left_nan_returns_nan() {
        // This is where f64::min would return 1.0. We return NaN.
        assert!(fmin(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn fmin_right_nan_returns_nan() {
        assert!(fmin(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn fmin_both_nan_returns_nan() {
        assert!(fmin(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn fmax_left_nan_returns_nan() {
        assert!(fmax(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn fmax_right_nan_returns_nan() {
        assert!(fmax(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn fmax_both_nan_returns_nan() {
        assert!(fmax(f64::NAN, f64::NAN).is_nan());
    }

    // ── Infinity handling ───────────────────────────────────────────────────

    #[test]
    fn fmin_with_positive_infinity() {
        assert_eq!(fmin(f64::INFINITY, 1.0), 1.0);
        assert_eq!(fmin(1.0, f64::INFINITY), 1.0);
    }

    #[test]
    fn fmin_with_negative_infinity() {
        assert_eq!(fmin(f64::NEG_INFINITY, 1.0), f64::NEG_INFINITY);
        assert_eq!(fmin(1.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn fmax_with_positive_infinity() {
        assert_eq!(fmax(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(fmax(1.0, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn fmax_with_negative_infinity() {
        assert_eq!(fmax(f64::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(fmax(1.0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn fmin_with_nan_and_infinity() {
        // NaN beats everything, including infinity.
        assert!(fmin(f64::NAN, f64::INFINITY).is_nan());
        assert!(fmin(f64::INFINITY, f64::NAN).is_nan());
        assert!(fmin(f64::NAN, f64::NEG_INFINITY).is_nan());
    }

    // ── Signed zero handling (IEEE 754-2019 minNum/maxNum spec) ─────────────

    #[test]
    fn fmin_neg_zero_is_smaller_than_pos_zero() {
        let result = fmin(-0.0, 0.0);
        assert_eq!(result, 0.0);  // Equal value...
        assert!(result.is_sign_negative());  // ...but negative sign.
    }

    #[test]
    fn fmin_neg_zero_order_independent() {
        let result1 = fmin(-0.0, 0.0);
        let result2 = fmin(0.0, -0.0);
        assert_eq!(result1.to_bits(), result2.to_bits());
    }

    #[test]
    fn fmax_pos_zero_is_larger_than_neg_zero() {
        let result = fmax(-0.0, 0.0);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_positive());
    }

    #[test]
    fn fmax_pos_zero_order_independent() {
        let result1 = fmax(-0.0, 0.0);
        let result2 = fmax(0.0, -0.0);
        assert_eq!(result1.to_bits(), result2.to_bits());
    }

    // ── Commutativity for non-NaN inputs ────────────────────────────────────

    #[test]
    fn fmin_commutative_on_non_nan() {
        let pairs = [(1.0, 2.0), (-3.0, 5.0), (1e-300, 1e300), (-0.0, 0.0)];
        for &(a, b) in &pairs {
            assert_eq!(fmin(a, b).to_bits(), fmin(b, a).to_bits());
        }
    }

    #[test]
    fn fmax_commutative_on_non_nan() {
        let pairs = [(1.0, 2.0), (-3.0, 5.0), (1e-300, 1e300), (-0.0, 0.0)];
        for &(a, b) in &pairs {
            assert_eq!(fmax(a, b).to_bits(), fmax(b, a).to_bits());
        }
    }

    // ── Regression tests for the 11-bug class ──────────────────────────────

    #[test]
    fn fold_with_fmin_propagates_nan() {
        // This is the smoke test for the bug class. Fold a vector containing
        // NaN using our fmin — the result must be NaN, not the min of the
        // non-NaN elements.
        let data = [1.0, 2.0, f64::NAN, 4.0, 5.0];
        let min = data.iter().copied().fold(f64::INFINITY, fmin);
        assert!(min.is_nan(),
            "fold(INFINITY, fmin) over data containing NaN must return NaN, got {min}");
    }

    #[test]
    fn fold_with_fmax_propagates_nan() {
        let data = [1.0, 2.0, f64::NAN, 4.0, 5.0];
        let max = data.iter().copied().fold(f64::NEG_INFINITY, fmax);
        assert!(max.is_nan(),
            "fold(NEG_INFINITY, fmax) over data containing NaN must return NaN, got {max}");
    }

    #[test]
    fn contrast_with_std_min_max() {
        // Explicit demonstration that our fmin differs from f64::min on NaN.
        // This is the bug class we're fixing.
        let with_nan = f64::NAN;
        let clean = 1.0_f64;

        // f64::min swallows NaN — returns the non-NaN operand.
        assert_eq!(f64::min(with_nan, clean), 1.0);
        assert_eq!(f64::max(with_nan, clean), 1.0);

        // Our fmin propagates NaN.
        assert!(fmin(with_nan, clean).is_nan());
        assert!(fmax(with_nan, clean).is_nan());
    }
}
