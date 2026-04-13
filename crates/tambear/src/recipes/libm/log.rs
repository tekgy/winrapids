//! `log(x)` — natural logarithm.
//!
//! # Mathematical recipe
//!
//! For any positive finite `x`, write
//!
//! ```text
//! x = 2^k · m,   m ∈ [1, 2)    (via frexp)
//! ```
//!
//! Then `ln(x) = k · ln(2) + ln(m)`. We reduce `m` further into
//! `[√2/2, √2)` by adjusting `k` when `m > √2`, so the approximation
//! polynomial always operates near 1. With `f = m - 1`:
//!
//! ```text
//! ln(m) = ln(1 + f) ≈ f · P(f) / Q(f)   (minimax rational approx)
//! ```
//!
//! For the Taylor/Padé approach we use instead:
//!
//! ```text
//! ln(1 + f) = 2·s + 2·s³/3 + 2·s⁵/5 + ...   where s = f / (2 + f)
//! ```
//!
//! This `s`-based series converges much faster than the direct Taylor
//! series for `ln(1+f)` because `|s| ≤ |f|/2`, halving the magnitude.
//!
//! # Special cases (all three entry points)
//!
//! - `log(NaN) = NaN`
//! - `log(+∞) = +∞`
//! - `log(-∞) = NaN`
//! - `log(0) = -∞`
//! - `log(negative) = NaN`
//! - `log(1) = 0` exactly
//!
//! # References
//!
//! - Cody & Waite, *Software Manual for the Elementary Functions* (1980)
//! - Sun fdlibm `__ieee754_log` — the canonical reference implementation
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11

use crate::primitives::compensated::dot::{compensated_horner, horner};
use crate::primitives::constants::{LN_2_DD, SQRT_2_F64};
use crate::primitives::double_double::{ops::dd_add_f64, ops::dd_mul_f64, DoubleDouble};
use crate::primitives::hardware::frexp;

/// Coefficients for the odd-power series `s + s^3·L1 + s^5·L2 + ...`
/// where `s = f / (2 + f)` and `f = m - 1`. These are the Lg1..Lg7
/// constants from Sun fdlibm's `__ieee754_log`, which are a minimax
/// fit to `(ln(1+f) - 2s) / s` on `[√2/2 - 1, √2 - 1]`.
const LOG_COEFFS: [f64; 7] = [
    6.666_666_666_666_735_13e-01, // Lg1
    3.999_999_999_940_941_91e-01, // Lg2
    2.857_142_874_366_239_15e-01, // Lg3
    2.222_219_843_214_978_40e-01, // Lg4
    1.818_357_216_161_805_01e-01, // Lg5
    1.531_383_769_920_937_33e-01, // Lg6
    1.479_819_860_511_658_59e-01, // Lg7
];

/// Coefficients for the correctly-rounded path. Uses the same 7
/// fdlibm coefficients — a future pass with a proper Remez solver will
/// extend to degree 10+ with minimax-optimized tails. For now, the
/// correctly-rounded path uses the same polynomial as compensated but
/// with DD reconstruction, giving ~2 ulps instead of the target 1.
const LOG_COEFFS_LONG: [f64; 7] = LOG_COEFFS;

// ── Entry points ────────────────────────────────────────────────────────────

/// `log(x)` — strict lowering. Target: ≤ 4 ulps.
#[inline]
pub fn log_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    let (k, f) = reduce(x);
    let s = f / (2.0 + f);
    let s2 = s * s;
    // R(z) = Lg1·z + Lg2·z² + ... where z = s². This is the correction
    // to the leading `2s` term in the ln(1+f) = 2s + 2s³/3 + ... series.
    // The fdlibm formula: ln(1+f) = f - hfsq + s·(hfsq + R(z))
    let r = s2 * horner(&LOG_COEFFS, s2);
    let hfsq = 0.5 * f * f;
    let result = f - hfsq + s * (hfsq + r);
    result + (k as f64) * LN_2_DD.hi
}

/// `log(x)` — compensated lowering. Target: ≤ 2 ulps.
///
/// Uses Cody-Waite reconstruction via `LN_2_DD` for the `k · ln(2)` term
/// and compensated polynomial evaluation.
#[inline]
pub fn log_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    let (k, f) = reduce(x);
    let s = f / (2.0 + f);
    let s2 = s * s;
    let r = s2 * compensated_horner(&LOG_COEFFS, s2);
    let hfsq = 0.5 * f * f;
    let poly_part = f - hfsq + s * (hfsq + r);
    let kf = k as f64;
    let k_ln2_hi = kf * LN_2_DD.hi;
    let k_ln2_lo = kf * LN_2_DD.lo;
    poly_part + k_ln2_lo + k_ln2_hi
}

/// `log(x)` — correctly-rounded lowering. Target: ≤ 1 ulp.
///
/// Uses DD working precision for the polynomial and the reconstruction.
#[inline]
pub fn log_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    let (k, f) = reduce(x);
    let s = f / (2.0 + f);
    let s2 = s * s;

    // DD polynomial: R(z) = z · P(z) where z = s².
    let poly_dd = eval_odd_poly_dd(&LOG_COEFFS_LONG, s2);
    let r = s2 * poly_dd.to_f64();

    let hfsq = 0.5 * f * f;
    let poly = f - hfsq + s * (hfsq + r);

    let k_ln2 = dd_mul_f64(LN_2_DD, k as f64);
    let result_dd = dd_add_f64(k_ln2, poly);
    result_dd.to_f64()
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn special_case(x: f64) -> Option<f64> {
    if x.is_nan() || x < 0.0 {
        return Some(f64::NAN);
    }
    if x == f64::INFINITY {
        return Some(f64::INFINITY);
    }
    if x == 0.0 {
        return Some(f64::NEG_INFINITY);
    }
    None
}

/// Range reduction: decompose x into `2^k · (1 + f)` with `f ∈ [√2/2 - 1, √2 - 1)`.
///
/// Returns `(k, f)` where `f = m - 1` and `m ∈ [√2/2, √2)`.
#[inline]
fn reduce(x: f64) -> (i32, f64) {
    let (mut m, mut k) = frexp(x);
    // frexp returns m ∈ [0.5, 1.0), exponent such that x = m · 2^k.
    // We want m ∈ [1, 2) so multiply m by 2 and decrement k.
    m *= 2.0;
    k -= 1;

    // If m > √2, adjust so the polynomial argument is centered near 0.
    if m > SQRT_2_F64 {
        m *= 0.5;
        k += 1;
    }
    let f = m - 1.0;
    (k, f)
}

/// Evaluate the odd-power correction polynomial:
/// `L1·s² + L2·s⁴ + L3·s⁶ + ...` — plain Horner on the coefficients
/// with `z = s²` as the variable.
#[inline]
fn eval_odd_poly(coeffs: &[f64], z: f64) -> f64 {
    horner(coeffs, z)
}

/// Compensated Horner on the odd-power polynomial.
#[inline]
fn eval_odd_poly_compensated(coeffs: &[f64], z: f64) -> f64 {
    compensated_horner(coeffs, z)
}

/// DD-valued Horner on the odd-power polynomial.
#[inline]
fn eval_odd_poly_dd(coeffs: &[f64], z: f64) -> DoubleDouble {
    let n = coeffs.len();
    let mut acc = DoubleDouble::from_f64(coeffs[n - 1]);
    for i in (0..n - 1).rev() {
        acc = dd_mul_f64(acc, z);
        acc = dd_add_f64(acc, coeffs[i]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            0.5,
            1.0,
            std::f64::consts::E,
            2.0,
            10.0,
            100.0,
            1000.0,
            1e10,
            1e100,
            1e-10,
            1e-100,
            0.1,
            0.01,
            0.001,
            std::f64::consts::PI,
            std::f64::consts::LN_2,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.ln();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── Boundary semantics ────────────────────────────────────────────────

    #[test]
    fn log_of_one_is_zero() {
        assert_eq!(log_strict(1.0), 0.0);
        assert_eq!(log_compensated(1.0), 0.0);
        assert_eq!(log_correctly_rounded(1.0), 0.0);
    }

    #[test]
    fn log_of_nan_is_nan() {
        assert!(log_strict(f64::NAN).is_nan());
        assert!(log_compensated(f64::NAN).is_nan());
        assert!(log_correctly_rounded(f64::NAN).is_nan());
    }

    #[test]
    fn log_of_pos_inf_is_pos_inf() {
        assert_eq!(log_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn log_of_zero_is_neg_inf() {
        assert_eq!(log_strict(0.0), f64::NEG_INFINITY);
        assert_eq!(log_compensated(0.0), f64::NEG_INFINITY);
        assert_eq!(log_correctly_rounded(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_of_negative_is_nan() {
        assert!(log_strict(-1.0).is_nan());
        assert!(log_compensated(-1.0).is_nan());
        assert!(log_correctly_rounded(-1.0).is_nan());
    }

    // ── Known-value spot checks ────────────────────────────────────────────

    #[test]
    fn log_of_e_is_one() {
        let e = std::f64::consts::E;
        assert_within_ulps(log_strict(e), 1.0, 4, "log_strict(e)");
        assert_within_ulps(log_compensated(e), 1.0, 2, "log_compensated(e)");
        assert_within_ulps(log_correctly_rounded(e), 1.0, 2, "log_correctly_rounded(e)");
    }

    #[test]
    fn log_of_2_is_ln2() {
        let ln2 = std::f64::consts::LN_2;
        assert_within_ulps(log_strict(2.0), ln2, 4, "log_strict(2)");
        assert_within_ulps(log_compensated(2.0), ln2, 2, "log_compensated(2)");
        assert_within_ulps(log_correctly_rounded(2.0), ln2, 2, "log_correctly_rounded(2)");
    }

    // ── Strategy ulp budgets ──────────────────────────────────────────────

    #[test]
    fn log_strict_within_budget() {
        check_strategy(log_strict, "log_strict", 4);
    }

    #[test]
    fn log_compensated_within_budget() {
        check_strategy(log_compensated, "log_compensated", 2);
    }

    #[test]
    fn log_correctly_rounded_within_budget() {
        // Currently 2 ulps — a Remez refit of the polynomial tails
        // will bring this to 1 ulp. Using the same 7-coefficient table
        // as the other strategies but with DD reconstruction.
        check_strategy(log_correctly_rounded, "log_correctly_rounded", 2);
    }

    // ── Mathematical identity tests ────────────────────────────────────────

    #[test]
    fn exp_log_roundtrip() {
        use crate::recipes::libm::exp::exp_correctly_rounded;
        let xs: &[f64] = &[0.5, 1.0, 2.0, 10.0, 100.0];
        for &x in xs {
            let rt = exp_correctly_rounded(log_correctly_rounded(x));
            let dist = ulps_between(rt, x);
            assert!(dist <= 3, "exp(log({x})) = {rt}, {dist} ulps from {x}");
        }
    }

    #[test]
    fn log_product_rule() {
        let a = 7.0;
        let b = 11.0;
        let lhs = log_correctly_rounded(a * b);
        let rhs = log_correctly_rounded(a) + log_correctly_rounded(b);
        let dist = ulps_between(lhs, rhs);
        assert!(dist <= 4, "log product rule: {dist} ulps");
    }

    #[test]
    fn log_monotone_on_positive_range() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let mut prev = log_correctly_rounded(xs[0]);
        for &x in &xs[1..] {
            let y = log_correctly_rounded(x);
            assert!(y >= prev, "log not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
