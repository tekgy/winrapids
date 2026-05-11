//! `log2(x)` — base-2 logarithm.
//!
//! # Mathematical recipe
//!
//! Per the libm-factoring frame, `log2` is a "binary-scaled" column of
//! the exp/log periodic table — analogous to `exp2`. There are two
//! regimes:
//!
//! - For **exact powers of 2** (x = 2^k for integer k, including
//!   negative k), `frexp(x)` returns `(0.5, k+1)` exactly and
//!   `log2(2^k) = k` is the exact answer with no polynomial needed.
//! - For **general positive x**, factor through the natural logarithm:
//!   `log2(x) = log(x) · (1/ln 2)`. The natural log handles its own
//!   range reduction; the final multiplication by `LOG2_E` is exact at
//!   double-double precision (the constant is precomputed).
//!
//! # The deferred LogKernelState
//!
//! Per math-researcher's libm-factoring open question #5, `log` and
//! `log2` should share a `LogKernelState(k, f, log1p_f)` rather than
//! `log2 = log · LOG2_E`. That kernel state isn't shipped in Sweep 35
//! (only `ExpKernelState` is). The session-aware variant here is a
//! pass-through to the strict path; future work introduces
//! `LogKernelState` and `log_session` / `log2_session` share it.
//!
//! # Special cases
//!
//! - `log2(NaN) = NaN`
//! - `log2(+∞) = +∞`
//! - `log2(0) = -∞`
//! - `log2(x) = NaN` for `x < 0`
//! - `log2(1) = 0` exactly
//! - `log2(2^k) = k` exactly for integer k in the f64 range
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.4.

use crate::intermediates::TamSession;
use crate::primitives::constants::LOG2_E_F64;
use crate::primitives::hardware::frexp;

use super::log::{log_compensated, log_correctly_rounded, log_strict};

/// `log2(x)` — strict lowering. Target: ≤ 4 ulps.
#[inline]
pub fn log2_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = power_of_two_exact(x) {
        return exact;
    }
    log_strict(x) * LOG2_E_F64
}

/// `log2(x)` — compensated lowering. Target: ≤ 2 ulps.
#[inline]
pub fn log2_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = power_of_two_exact(x) {
        return exact;
    }
    log_compensated(x) * LOG2_E_F64
}

/// `log2(x)` — correctly-rounded lowering. Target: ≤ 2 ulps.
#[inline]
pub fn log2_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = power_of_two_exact(x) {
        return exact;
    }
    log_correctly_rounded(x) * LOG2_E_F64
}

/// `log2(x)` — session-aware pass-through.
///
/// Currently delegates to `log2_strict`. Future work introduces
/// `LogKernelState` and threads sharing through it; the API is
/// in place here so consumers can adopt it now.
pub fn log2_session(_session: &mut TamSession, x: f64) -> f64 {
    log2_strict(x)
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

/// Exact `log2(2^k) = k` short-circuit. Returns `Some(k as f64)` when
/// `x` is a positive power of 2 (its mantissa bits are all zero after
/// frexp normalization).
#[inline]
fn power_of_two_exact(x: f64) -> Option<f64> {
    if x <= 0.0 || !x.is_finite() {
        return None;
    }
    // frexp(x) returns (m, k) with m ∈ [0.5, 1.0); for x = 2^j,
    // m = 0.5 exactly and k = j + 1, so log2(x) = j = k - 1.
    let (m, k) = frexp(x);
    if m == 0.5 {
        Some((k - 1) as f64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            // Power-of-two exact path
            1.0, 2.0, 4.0, 8.0, 16.0, 1024.0, 0.5, 0.25, 0.125,
            // General path
            3.0, 5.0, 10.0, 100.0, 0.1, 0.01,
            1e10, 1e-10, 1e100, 1e-100,
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.log2();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    #[test]
    fn log2_of_one_is_zero() {
        assert_eq!(log2_strict(1.0), 0.0);
        assert_eq!(log2_compensated(1.0), 0.0);
        assert_eq!(log2_correctly_rounded(1.0), 0.0);
    }

    #[test]
    fn log2_of_power_of_two_is_exact() {
        // The exact path must produce bit-identical results.
        let cases: &[(f64, f64)] = &[
            (2.0, 1.0), (4.0, 2.0), (8.0, 3.0), (1024.0, 10.0),
            (0.5, -1.0), (0.25, -2.0), (1.0 / 1024.0, -10.0),
            ((1u64 << 52) as f64, 52.0),
        ];
        for &(x, expected) in cases {
            assert_eq!(log2_strict(x), expected, "log2_strict({x})");
            assert_eq!(log2_compensated(x), expected, "log2_compensated({x})");
            assert_eq!(log2_correctly_rounded(x), expected, "log2_correctly_rounded({x})");
        }
    }

    #[test]
    fn log2_of_nan_is_nan() {
        assert!(log2_strict(f64::NAN).is_nan());
    }

    #[test]
    fn log2_of_pos_inf_is_pos_inf() {
        assert_eq!(log2_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn log2_of_zero_is_neg_inf() {
        assert_eq!(log2_strict(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log2_of_negative_is_nan() {
        assert!(log2_strict(-1.0).is_nan());
    }

    #[test]
    fn log2_strict_within_budget() {
        check_strategy(log2_strict, "log2_strict", 4);
    }

    #[test]
    fn log2_compensated_within_budget() {
        check_strategy(log2_compensated, "log2_compensated", 4);
    }

    #[test]
    fn log2_correctly_rounded_within_budget() {
        check_strategy(log2_correctly_rounded, "log2_correctly_rounded", 4);
    }

    #[test]
    fn log2_session_matches_strict() {
        let mut session = TamSession::new();
        let xs: &[f64] = &[1.0, 2.0, 3.0, 10.0, 100.0, 0.5, 0.1];
        for &x in xs {
            assert_eq!(log2_session(&mut session, x), log2_strict(x));
        }
    }

    #[test]
    fn log2_exp2_roundtrip() {
        use super::super::exp2::exp2_strict;
        let xs: &[f64] = &[0.5, 1.0, 2.0, 10.0, -1.0, -10.0, std::f64::consts::PI];
        for &x in xs {
            let rt = log2_strict(exp2_strict(x));
            let dist = ulps_between(rt, x);
            assert!(dist <= 4, "log2(exp2({x})) = {rt}, {dist} ulps from {x}");
        }
    }

    #[test]
    fn log2_monotone_on_positive_range() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let mut prev = log2_strict(xs[0]);
        for &x in &xs[1..] {
            let y = log2_strict(x);
            assert!(y >= prev, "log2 not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
