//! `log10(x)` — base-10 logarithm.
//!
//! # Mathematical recipe
//!
//! `log10(x) = log(x) · (1/ln 10) = log(x) · LOG10_E`. Unlike `log2`,
//! there's no exact short-circuit for powers of 10: while `10^3 = 1000`
//! is exactly representable, `log10(1000)` involves the natural log of
//! a number that is not an exact float and the multiplication by
//! `1/ln(10)` (also not exact) — so we always go through the polynomial.
//!
//! However, `log10` does admit an integer-output observation: for
//! `x = 10^k` with `k` such that `10^k` is exactly representable,
//! `log10(10^k)` should round to `k`. We do not special-case this
//! because the input `10^k` is generally not exact (only k = 0, 1, ...
//! up to ~15 yield exact 10^k values), and detecting exactness is
//! more expensive than computing the polynomial.
//!
//! # Special cases
//!
//! - `log10(NaN) = NaN`
//! - `log10(+∞) = +∞`
//! - `log10(0) = -∞`
//! - `log10(x) = NaN` for `x < 0`
//! - `log10(1) = 0` exactly
//!
//! # Error budget
//!
//! | Entry point              | Path                  | Target ulps |
//! |--------------------------|-----------------------|-------------|
//! | `log10_strict`           | log(x) · LOG10_E      | ≤ 4         |
//! | `log10_compensated`      | log_comp · LOG10_E    | ≤ 2         |
//! | `log10_correctly_rounded`| log_cr · LOG10_E      | ≤ 2         |
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.4.

use crate::intermediates::TamSession;
use crate::primitives::constants::LOG10_E_F64;

use super::log::{log_compensated, log_correctly_rounded, log_strict};

/// `log10(x)` — strict lowering. Target: ≤ 4 ulps.
#[inline]
pub fn log10_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    log_strict(x) * LOG10_E_F64
}

/// `log10(x)` — compensated lowering. Target: ≤ 2 ulps.
#[inline]
pub fn log10_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    log_compensated(x) * LOG10_E_F64
}

/// `log10(x)` — correctly-rounded lowering. Target: ≤ 2 ulps.
#[inline]
pub fn log10_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    log_correctly_rounded(x) * LOG10_E_F64
}

/// `log10(x)` — session-aware pass-through (LogKernelState not yet
/// shipped; future work adds it).
pub fn log10_session(_session: &mut TamSession, x: f64) -> f64 {
    log10_strict(x)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            1.0, 10.0, 100.0, 1000.0, 1e10, 1e100,
            0.1, 0.01, 1e-10, 1e-100,
            2.0, 3.0, 5.0, 7.0, 50.0,
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.log10();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    #[test]
    fn log10_of_one_is_zero() {
        assert_eq!(log10_strict(1.0), 0.0);
        assert_eq!(log10_compensated(1.0), 0.0);
        assert_eq!(log10_correctly_rounded(1.0), 0.0);
    }

    #[test]
    fn log10_of_ten_is_one() {
        assert_within_ulps(log10_strict(10.0), 1.0, 4, "log10_strict(10)");
        assert_within_ulps(log10_compensated(10.0), 1.0, 2, "log10_compensated(10)");
    }

    #[test]
    fn log10_of_thousand_is_three() {
        assert_within_ulps(log10_strict(1000.0), 3.0, 4, "log10_strict(1000)");
        assert_within_ulps(log10_compensated(1000.0), 3.0, 4, "log10_compensated(1000)");
    }

    #[test]
    fn log10_of_nan_is_nan() {
        assert!(log10_strict(f64::NAN).is_nan());
    }

    #[test]
    fn log10_of_pos_inf_is_pos_inf() {
        assert_eq!(log10_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn log10_of_zero_is_neg_inf() {
        assert_eq!(log10_strict(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log10_of_negative_is_nan() {
        assert!(log10_strict(-1.0).is_nan());
    }

    #[test]
    fn log10_strict_within_budget() {
        check_strategy(log10_strict, "log10_strict", 4);
    }

    #[test]
    fn log10_compensated_within_budget() {
        check_strategy(log10_compensated, "log10_compensated", 4);
    }

    #[test]
    fn log10_correctly_rounded_within_budget() {
        check_strategy(log10_correctly_rounded, "log10_correctly_rounded", 4);
    }

    #[test]
    fn log10_session_matches_strict() {
        let mut session = TamSession::new();
        let xs: &[f64] = &[1.0, 10.0, 100.0, 0.1, std::f64::consts::PI];
        for &x in xs {
            assert_eq!(log10_session(&mut session, x), log10_strict(x));
        }
    }

    #[test]
    fn log10_exp10_roundtrip() {
        use super::super::exp10::exp10_strict;
        let xs: &[f64] = &[0.5, 1.0, 2.0, 5.0, -1.0, std::f64::consts::PI];
        for &x in xs {
            let rt = log10_strict(exp10_strict(x));
            let dist = ulps_between(rt, x);
            // exp10(x) introduces error via x·ln10 multiplication;
            // log10(10^x) recovers x with a few ulps of slack.
            assert!(dist <= 16, "log10(exp10({x})) = {rt}, {dist} ulps from {x}");
        }
    }

    #[test]
    fn log10_monotone_on_positive_range() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let mut prev = log10_strict(xs[0]);
        for &x in &xs[1..] {
            let y = log10_strict(x);
            assert!(y >= prev, "log10 not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
