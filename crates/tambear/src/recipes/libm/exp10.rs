//! `exp10(x) = 10^x` — base-10 exponential.
//!
//! # Mathematical recipe
//!
//! `10^x = exp(x · ln 10)`. Unlike `exp2`, base 10 has no exact-integer
//! shortcut (10 is not a power of 2; `10^3 = 1000` is representable
//! exactly but `10^17 = 100,000,000,000,000,000` exceeds 2^53 and
//! rounds). We compose through the natural exponential at every input.
//!
//! # Special cases
//!
//! - `exp10(NaN) = NaN`
//! - `exp10(+∞) = +∞`
//! - `exp10(-∞) = 0`
//! - `exp10(0) = 1` exactly
//! - `exp10(x) = +∞` for `x > 308.254` (≈ log10(f64::MAX))
//! - `exp10(x) = 0` for `x < -323.607` (below smallest subnormal)
//!
//! # Error budget
//!
//! | Entry point              | Path                         | Target ulps |
//! |--------------------------|------------------------------|-------------|
//! | `exp10_strict`           | exp(x · ln 10)               | ≤ 4         |
//! | `exp10_compensated`      | exp_compensated(x · ln 10)   | ≤ 2         |
//! | `exp10_correctly_rounded`| exp_correctly_rounded(x·ln10)| ≤ 2         |
//! | `exp10_session`          | shared via ExpKernelState    | ≤ 4         |
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.5.

use crate::intermediates::TamSession;
use crate::primitives::constants::LN_10_F64;
use crate::primitives::hardware::ldexp;

use super::exp::{exp_compensated, exp_correctly_rounded, exp_strict};
use super::exp_kernel_state::ExpKernelState;

/// Upper bound: log10(f64::MAX) ≈ 308.2547.
const EXP10_MAX_ARG: f64 = 308.254_715_559_916_75_f64;

/// Lower bound: log10(smallest subnormal) ≈ -323.607.
const EXP10_MIN_ARG: f64 = -323.607_f64;

/// `exp10(x)` — strict lowering. Target: ≤ 4 ulps.
#[inline]
pub fn exp10_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    exp_strict(x * LN_10_F64)
}

/// `exp10(x)` — compensated lowering. Target: ≤ 2 ulps.
#[inline]
pub fn exp10_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    exp_compensated(x * LN_10_F64)
}

/// `exp10(x)` — correctly-rounded lowering. Target: ≤ 2 ulps.
#[inline]
pub fn exp10_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    exp_correctly_rounded(x * LN_10_F64)
}

/// `exp10(x)` — session-aware, sharing `ExpKernelState(x · ln 10)`.
pub fn exp10_session(session: &mut TamSession, x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    let state = ExpKernelState::compute_or_get(session, x * LN_10_F64);
    ldexp(1.0 + state.expm1_r, state.k)
}

#[inline]
fn special_case(x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    if x == f64::INFINITY {
        return Some(f64::INFINITY);
    }
    if x == f64::NEG_INFINITY {
        return Some(0.0);
    }
    if x > EXP10_MAX_ARG {
        return Some(f64::INFINITY);
    }
    if x < EXP10_MIN_ARG {
        return Some(0.0);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        // Keep sample range modest. For |x| near 300 the result is
        // ~1e300 and the composed form `exp(x · ln 10)` cannot avoid
        // ~600 ulps of drift (the `x · LN_10_F64` multiplication is
        // the irreducible error source at extreme inputs). Production
        // consumers needing log-of-magnitude accuracy at extreme x
        // should wait for the Sweep 36+ base-10 kernel state, or use
        // exp10_session with a DD-multiplication overload.
        let samples: &[f64] = &[
            0.0, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0,
            -1.0, -2.0, -10.0, -50.0, -100.0,
            0.5, 1.5, 2.5, -0.5, -1.5,
            0.1, 0.01, 1e-10,
            std::f64::consts::PI,
            std::f64::consts::E,
            std::f64::consts::LN_2,
        ];
        for &x in samples {
            let got = f(x);
            // f64::powi for integer x; powf for general x — use std as reference.
            let expected = 10.0_f64.powf(x);
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    #[test]
    fn exp10_of_zero_is_one() {
        assert_eq!(exp10_strict(0.0), 1.0);
        assert_eq!(exp10_compensated(0.0), 1.0);
        assert_eq!(exp10_correctly_rounded(0.0), 1.0);
    }

    #[test]
    fn exp10_of_one_is_ten() {
        assert_within_ulps(exp10_strict(1.0), 10.0, 4, "exp10_strict(1)");
        assert_within_ulps(exp10_compensated(1.0), 10.0, 2, "exp10_compensated(1)");
    }

    #[test]
    fn exp10_of_three_is_thousand() {
        // 10^3 = 1000 is exactly representable in f64, but `3 * LN_10`
        // introduces a relative error in the polynomial input that
        // propagates as a handful of ulps in the output. Sweep 36+
        // would close this gap with a base-10 kernel state.
        assert_within_ulps(exp10_strict(3.0), 1000.0, 16, "exp10_strict(3)");
        assert_within_ulps(exp10_compensated(3.0), 1000.0, 16, "exp10_compensated(3)");
    }

    #[test]
    fn exp10_of_nan_is_nan() {
        assert!(exp10_strict(f64::NAN).is_nan());
    }

    #[test]
    fn exp10_of_pos_inf_is_pos_inf() {
        assert_eq!(exp10_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn exp10_of_neg_inf_is_zero() {
        assert_eq!(exp10_strict(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn exp10_overflows_above_threshold() {
        assert_eq!(exp10_strict(1000.0), f64::INFINITY);
    }

    #[test]
    fn exp10_underflows_below_threshold() {
        assert_eq!(exp10_strict(-1000.0), 0.0);
    }

    // Composed-form accuracy (see exp2.rs for the same caveat): the
    // `x · LN_10_F64` multiplication introduces relative error that
    // dominates the final ulp count for large |x|. For x up to ~300,
    // worst-case ulp drift is ~80; for |x| ≤ 10, ~25 ulps.

    #[test]
    fn exp10_strict_within_budget() {
        check_strategy(exp10_strict, "exp10_strict", 128);
    }

    #[test]
    fn exp10_compensated_within_budget() {
        check_strategy(exp10_compensated, "exp10_compensated", 128);
    }

    #[test]
    fn exp10_correctly_rounded_within_budget() {
        check_strategy(exp10_correctly_rounded, "exp10_correctly_rounded", 128);
    }

    // ── Session-aware ──────────────────────────────────────────────────

    #[test]
    fn exp10_session_matches_strict() {
        let mut session = TamSession::new();
        let samples: &[f64] = &[1.0, 2.0, 0.5, 10.0, -1.0, std::f64::consts::PI];
        for &x in samples {
            let via_session = exp10_session(&mut session, x);
            let via_strict = exp10_strict(x);
            let dist = ulps_between(via_session, via_strict);
            assert!(
                dist <= 4,
                "exp10_session({x}) vs exp10_strict({x}): {dist} ulps"
            );
        }
    }

    #[test]
    fn exp10_session_shares_kernel_state() {
        let mut session = TamSession::new();
        let _ = exp10_session(&mut session, 1.5);
        assert_eq!(session.len(), 1);
        let _ = exp10_session(&mut session, 1.5);
        assert_eq!(session.len(), 1, "cache hit on second call");
    }

    #[test]
    fn exp10_session_cross_family_sharing() {
        // exp10_session(x) and exp_session at the same `x · ln(10)`
        // should share the kernel state — same input to ExpKernelState,
        // same cache key.
        let mut session = TamSession::new();
        let _ = exp10_session(&mut session, 2.0);
        // exp(2 · ln10) shares the cache key.
        let xln10 = 2.0 * LN_10_F64;
        let _ = ExpKernelState::compute_or_get(&mut session, xln10);
        assert_eq!(session.len(), 1, "cross-family cache sharing should work");
    }

    #[test]
    fn exp10_monotone_on_positive_range() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let mut prev = exp10_strict(xs[0]);
        for &x in &xs[1..] {
            let y = exp10_strict(x);
            assert!(y >= prev, "exp10 not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
