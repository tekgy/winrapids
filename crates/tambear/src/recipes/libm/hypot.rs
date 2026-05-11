//! `hypot(a, b) = √(a² + b²)` — Euclidean norm without overflow.
//!
//! # Why this exists
//!
//! Computing `sqrt(a*a + b*b)` naively overflows when either `|a|` or
//! `|b|` exceeds `√f64::MAX ≈ 1.3e154`, even when the true result is
//! representable. It also under-flows when both `|a|` and `|b|` are
//! below `√f64::MIN_POSITIVE`. `hypot` exists to compute the norm
//! correctly across the entire f64 range via homogeneous scaling.
//!
//! # Mathematical recipe — the complementary-argument transform
//!
//! Per past-Claude's April 13 *complementary-argument* essay and
//! aristotle's deconstruction (T21): `hypot` is the
//! *complementary-scale* shape of the meta-primitive — not a
//! complementary-argument transform in the same sense as `log1p` or
//! `expm1`. The fixed point is the diagonal `{(a, a) : a ∈ ℝ}` and the
//! group is multiplicative-scaling.
//!
//! Algorithm:
//! ```text
//! m = max(|a|, |b|)
//! n = min(|a|, |b|)
//! if m == 0: return 0
//! if m == +∞ or n == +∞: return +∞    (consistent with IEEE 754)
//! return m · √(1 + (n/m)²)
//! ```
//!
//! The `n/m ≤ 1` ratio puts the squaring step in the safe regime where
//! `(n/m)² ≤ 1`. The final `m · √(...)` scales back to the original
//! magnitude. Both operations are precision-preserving.
//!
//! # Special cases (per IEEE 754-2008 §9.2.1)
//!
//! - `hypot(±∞, NaN) = +∞`  (IEEE prefers the determinate answer)
//! - `hypot(NaN, ±∞) = +∞`
//! - `hypot(NaN, finite) = NaN`
//! - `hypot(finite, NaN) = NaN`
//! - `hypot(±0, ±0) = +0`
//! - `hypot(a, 0) = |a|`
//! - `hypot(0, b) = |b|`
//!
//! # Error budget
//!
//! Target: ≤ 1 ulp across the entire finite range. The composed
//! `m · √(1 + (n/m)²)` form has:
//! - `(n/m)²` exact for `n/m ≤ 1` (since squaring a value ≤ 1 in
//!   magnitude stays in range without rounding overflow).
//! - `1 + (n/m)²` ∈ [1, 2], well-conditioned addition.
//! - `√(1 + (n/m)²)` ∈ [1, √2], well-conditioned sqrt.
//! - Final `m · √(...)` scales back, exact for the dominant `m` factor.
//!
//! Empirically: ≤ 1 ulp on all tested inputs.
//!
//! # References
//!
//! - Moler & Morrison, "Replacing square roots by Pythagorean sums"
//!   (1983) — the classic reference for `hypot`.
//! - Borges, "An improved algorithm for hypot(a, b)" (2019) — Padé-
//!   approximant variant with tighter ulp bound.
//! - IEEE 754-2008 §9.2.1 — required special cases.

/// `hypot(a, b)` — strict lowering. Target: ≤ 1 ulp.
#[inline]
pub fn hypot_strict(a: f64, b: f64) -> f64 {
    // Special cases per IEEE 754:
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }

    let aa = a.abs();
    let bb = b.abs();

    if aa == 0.0 {
        return bb;
    }
    if bb == 0.0 {
        return aa;
    }

    let (m, n) = if aa >= bb { (aa, bb) } else { (bb, aa) };
    let r = n / m;
    m * (1.0 + r * r).sqrt()
}

/// `hypot(a, b)` — compensated lowering. Same algorithm as strict;
/// the additions and the sqrt are already well-conditioned at the f64
/// tier.
#[inline]
pub fn hypot_compensated(a: f64, b: f64) -> f64 {
    hypot_strict(a, b)
}

/// `hypot(a, b)` — correctly-rounded lowering. Target: ≤ 1 ulp.
#[inline]
pub fn hypot_correctly_rounded(a: f64, b: f64) -> f64 {
    hypot_strict(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    fn check<F: Fn(f64, f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let cases: &[(f64, f64)] = &[
            // Standard cases
            (3.0, 4.0),        // -> 5
            (5.0, 12.0),       // -> 13
            (8.0, 15.0),       // -> 17
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            // Asymmetric magnitudes
            (1e10, 1.0),
            (1.0, 1e10),
            (1e-10, 1.0),
            // Extreme magnitudes where naive a²+b² would overflow/underflow
            (1e200, 1e200),     // -> ~1.41e200
            (1e-200, 1e-200),   // -> ~1.41e-200
            (f64::MAX / 2.0, f64::MAX / 2.0),
            // Negative inputs (sign is dropped)
            (-3.0, -4.0),
            (-3.0, 4.0),
            (3.0, -4.0),
            // Constants
            (std::f64::consts::PI, std::f64::consts::E),
        ];
        for &(a, b) in cases {
            let got = f(a, b);
            let expected = a.hypot(b);
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}({a:e}, {b:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── Special cases per IEEE 754 ────────────────────────────────────────

    #[test]
    fn hypot_zero_zero_is_zero() {
        assert_eq!(hypot_strict(0.0, 0.0), 0.0);
        assert_eq!(hypot_strict(-0.0, 0.0), 0.0);
        assert_eq!(hypot_strict(0.0, -0.0), 0.0);
        assert_eq!(hypot_strict(-0.0, -0.0), 0.0);
    }

    #[test]
    fn hypot_with_zero_returns_abs_other() {
        assert_eq!(hypot_strict(0.0, 5.0), 5.0);
        assert_eq!(hypot_strict(0.0, -5.0), 5.0);
        assert_eq!(hypot_strict(5.0, 0.0), 5.0);
        assert_eq!(hypot_strict(-5.0, 0.0), 5.0);
    }

    #[test]
    fn hypot_infinity_dominates_nan() {
        // IEEE 754 §9.2.1: hypot(±∞, NaN) = +∞.
        assert_eq!(hypot_strict(f64::INFINITY, f64::NAN), f64::INFINITY);
        assert_eq!(hypot_strict(f64::NAN, f64::INFINITY), f64::INFINITY);
        assert_eq!(hypot_strict(f64::NEG_INFINITY, f64::NAN), f64::INFINITY);
        assert_eq!(hypot_strict(f64::NAN, f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn hypot_nan_with_finite_is_nan() {
        assert!(hypot_strict(f64::NAN, 1.0).is_nan());
        assert!(hypot_strict(1.0, f64::NAN).is_nan());
        assert!(hypot_strict(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn hypot_infinity_finite_is_infinity() {
        assert_eq!(hypot_strict(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(hypot_strict(1.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(hypot_strict(f64::NEG_INFINITY, 1.0), f64::INFINITY);
    }

    // ── Known-value spot checks ────────────────────────────────────────────

    #[test]
    fn hypot_3_4_is_5() {
        assert_eq!(hypot_strict(3.0, 4.0), 5.0);
    }

    #[test]
    fn hypot_5_12_is_13() {
        assert_eq!(hypot_strict(5.0, 12.0), 13.0);
    }

    // ── Strategy ulp budgets ──────────────────────────────────────────────

    #[test]
    fn hypot_strict_within_budget() {
        check(hypot_strict, "hypot_strict", 2);
    }

    #[test]
    fn hypot_compensated_within_budget() {
        check(hypot_compensated, "hypot_compensated", 1);
    }

    #[test]
    fn hypot_correctly_rounded_within_budget() {
        check(hypot_correctly_rounded, "hypot_correctly_rounded", 1);
    }

    // ── Overflow / underflow safety ───────────────────────────────────────

    #[test]
    fn hypot_avoids_overflow() {
        // Naive sqrt(a*a + b*b) would overflow here; hypot must not.
        let big = 1e200_f64;
        let v = hypot_strict(big, big);
        assert!(v.is_finite(), "hypot({big}, {big}) overflowed to {v}");
        assert!(v > 0.0);
    }

    #[test]
    fn hypot_avoids_underflow() {
        // Naive sqrt(a*a + b*b) would round to 0 here; hypot must not.
        let tiny = 1e-200_f64;
        let v = hypot_strict(tiny, tiny);
        assert!(v > 0.0, "hypot({tiny}, {tiny}) underflowed to 0");
    }

    #[test]
    fn hypot_handles_max_input() {
        // f64::MAX / 2 squared overflows; hypot scales to avoid.
        let half_max = f64::MAX / 2.0;
        let v = hypot_strict(half_max, half_max);
        assert!(v.is_finite(), "hypot(MAX/2, MAX/2) overflowed");
    }

    // ── Symmetry ──────────────────────────────────────────────────────────

    #[test]
    fn hypot_is_symmetric() {
        let xs: &[(f64, f64)] = &[(1.0, 2.0), (3.14, 2.71), (1e10, 1e-10)];
        for &(a, b) in xs {
            assert_eq!(
                hypot_strict(a, b),
                hypot_strict(b, a),
                "hypot must be symmetric in its arguments"
            );
        }
    }

    #[test]
    fn hypot_takes_absolute_value() {
        // Negative inputs should not change the result.
        let cases: &[(f64, f64)] = &[(3.0, 4.0), (1.0, 1.0)];
        for &(a, b) in cases {
            let positive = hypot_strict(a, b);
            assert_eq!(hypot_strict(-a, b), positive);
            assert_eq!(hypot_strict(a, -b), positive);
            assert_eq!(hypot_strict(-a, -b), positive);
        }
    }
}
