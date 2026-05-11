//! `exp2(x) = 2^x` — base-2 exponential.
//!
//! # Mathematical recipe
//!
//! Per the libm-factoring frame, `exp2` is a "binary-scaled" column of
//! the exp/log periodic table — analogous to `sinpi` for trig:
//!
//! - For **integer** `x` in the f64-representable integer range
//!   (`-1074 ≤ x ≤ 1023`), `2^x` is an **exact** result via `ldexp`.
//!   No polynomial, no reduction, no rounding.
//! - For **general** `x`, factor through the natural exponential:
//!   `2^x = exp(x · ln 2)`. This pulls the standard `ExpKernelState`
//!   for the reduced argument `x · ln 2`.
//!
//! # Special cases
//!
//! - `exp2(NaN) = NaN`
//! - `exp2(+∞) = +∞`
//! - `exp2(-∞) = 0`
//! - `exp2(0) = 1` exactly
//! - `exp2(x) = +∞` for `x > 1023` (overflow)
//! - `exp2(x) = 0` for `x < -1074` (below smallest subnormal)
//! - `exp2(integer)` returns the exact power of 2 (no polynomial)
//!
//! # Error budget
//!
//! Composed form `exp(x · ln 2)` inherits an `x · LN_2_F64` multiplication
//! whose relative error scales with `|x|`. For |x| ≤ 100, worst-case
//! drift is ~16 ulps; for integer x, the exact-path skips this entirely.
//! A dedicated `exp2_kernel_state` (math-researcher's open question #5)
//! would close the gap to ≤ 2 ulps by keeping the reduced argument in
//! base-2 throughout; deferred to Sweep 36+.
//!
//! | Entry point             | Path                         | Achievable ulps |
//! |-------------------------|------------------------------|----------------|
//! | `exp2_strict`           | integer-exact OR exp(x·ln2)  | ≤ 32           |
//! | `exp2_compensated`      | integer-exact OR exp_comp    | ≤ 32           |
//! | `exp2_correctly_rounded`| integer-exact OR exp_cr      | ≤ 32           |
//! | `exp2_session`          | shared via ExpKernelState    | ≤ 32           |
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.5.
//! - Tang, "Table-driven implementation of the exponential function in
//!   IEEE floating-point arithmetic" (1989) — base-2 reduction is the
//!   natural form for exp2.

use std::sync::Arc;

use crate::intermediates::TamSession;
use crate::primitives::constants::LN_2_F64;
use crate::primitives::hardware::ldexp;

use super::exp::{exp_compensated, exp_correctly_rounded, exp_strict};
use super::exp_kernel_state::ExpKernelState;

/// Upper bound for finite `exp2` output: any x > 1024 overflows.
const EXP2_MAX_ARG: f64 = 1024.0;

/// Lower bound for non-zero `exp2` output (`2^-1074` is the smallest
/// representable subnormal; below that the result rounds to 0).
const EXP2_MIN_ARG: f64 = -1075.0;

/// `exp2(x)` — strict lowering. Target: ≤ 2 ulps.
#[inline]
pub fn exp2_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = integer_exact(x) {
        return exact;
    }
    exp_strict(x * LN_2_F64)
}

/// `exp2(x)` — compensated lowering. Target: ≤ 1 ulp.
#[inline]
pub fn exp2_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = integer_exact(x) {
        return exact;
    }
    exp_compensated(x * LN_2_F64)
}

/// `exp2(x)` — correctly-rounded lowering. Target: ≤ 1 ulp.
#[inline]
pub fn exp2_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = integer_exact(x) {
        return exact;
    }
    exp_correctly_rounded(x * LN_2_F64)
}

/// `exp2(x)` — session-aware, sharing `ExpKernelState(x · ln 2)` via
/// the supplied `TamSession`. First call computes; subsequent calls
/// (with the same input and session) pull from cache.
///
/// Returns the f64 result by reconstructing `exp(r) · 2^k` from the
/// cached `(k, r, expm1_r)`. The reconstruction is `(1 + expm1_r) · 2^k`
/// via `ldexp`, which is exact for the `2^k` step.
pub fn exp2_session(session: &mut TamSession, x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }
    if let Some(exact) = integer_exact(x) {
        return exact;
    }
    // Form the natural-log-equivalent input and pull the shared state.
    // Note: `x * LN_2_F64` introduces error in the multiplication; for
    // session-aware exp2 to be bit-equivalent to exp2_strict, this
    // recipe-layer step is the irreducible source of relative error
    // (per math-researcher's open question #5: exp2 wants its own
    // kernel state at the binary-scaled column, deferred to Sweep 36).
    let xln2 = x * LN_2_F64;
    let state = ExpKernelState::compute_or_get(session, xln2);
    reconstruct_exp(&state)
}

/// `exp2(x)` for integer `x` in the f64 integer range — exact result
/// via `ldexp(1.0, x_as_int)`. Returns `None` if `x` is not an integer
/// or is outside the safe integer range for `ldexp`.
#[inline]
fn integer_exact(x: f64) -> Option<f64> {
    // x must be an integer and within ldexp's safe range. f64 can
    // represent integers exactly up to 2^53; ldexp uses i32 so the
    // shift count must fit. For exp2 the meaningful integer range is
    // [-1074, 1023] (below underflow, above overflow); we tighten to
    // [-1023, 1023] for the exact path because ldexp normal-result
    // boundary is 2^-1022 (subnormal exp2 results need full
    // computation to avoid losing precision in the polynomial).
    if x.fract() != 0.0 {
        return None;
    }
    if !(-1023.0..=1023.0).contains(&x) {
        return None;
    }
    let n = x as i32;
    Some(ldexp(1.0, n))
}

/// Reconstruct `exp(r) · 2^k = (1 + expm1_r) · 2^k` from the kernel state.
#[inline]
fn reconstruct_exp(state: &ExpKernelState) -> f64 {
    // The bit-shift via `ldexp` is exact for any `k` in the normal range.
    // `1 + expm1_r` is well-conditioned because |expm1_r| ≤ 0.414 (the
    // bound from |r| ≤ ln(2)/2), so `1 + expm1_r ∈ [0.586, 1.414]` —
    // no cancellation, no overflow.
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
    if x > EXP2_MAX_ARG {
        return Some(f64::INFINITY);
    }
    if x < EXP2_MIN_ARG {
        return Some(0.0);
    }
    None
}

/// Drop unused-import silencer for `Arc` — used only when the session
/// path is exercised; keep import visible for clarity.
#[allow(dead_code)]
fn _arc_witness(_: Arc<u8>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            // Integer-exact path
            0.0, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 500.0, 1000.0,
            -1.0, -2.0, -10.0, -50.0, -100.0, -500.0, -1000.0,
            // General path
            0.5, 1.5, 2.5, 3.14, 10.5, 100.5,
            -0.5, -1.5, -100.5,
            0.1, 0.001, 1e-10,
            // Constants
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.exp2();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    #[test]
    fn exp2_of_zero_is_one() {
        assert_eq!(exp2_strict(0.0), 1.0);
        assert_eq!(exp2_compensated(0.0), 1.0);
        assert_eq!(exp2_correctly_rounded(0.0), 1.0);
    }

    #[test]
    fn exp2_of_integer_is_exact_power_of_two() {
        // The integer-exact path must produce bit-identical results.
        let cases: &[(f64, f64)] = &[
            (1.0, 2.0), (2.0, 4.0), (3.0, 8.0), (10.0, 1024.0),
            (-1.0, 0.5), (-2.0, 0.25), (-10.0, 1.0 / 1024.0),
            (52.0, (1u64 << 52) as f64),
        ];
        for &(x, expected) in cases {
            assert_eq!(exp2_strict(x), expected, "exp2_strict({x})");
            assert_eq!(exp2_compensated(x), expected, "exp2_compensated({x})");
            assert_eq!(exp2_correctly_rounded(x), expected, "exp2_correctly_rounded({x})");
        }
    }

    #[test]
    fn exp2_of_nan_is_nan() {
        assert!(exp2_strict(f64::NAN).is_nan());
    }

    #[test]
    fn exp2_of_pos_inf_is_pos_inf() {
        assert_eq!(exp2_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn exp2_of_neg_inf_is_zero() {
        assert_eq!(exp2_strict(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn exp2_overflows_above_1024() {
        assert_eq!(exp2_strict(2000.0), f64::INFINITY);
    }

    #[test]
    fn exp2_underflows_below_neg_1075() {
        assert_eq!(exp2_strict(-2000.0), 0.0);
    }

    #[test]
    fn exp2_of_half_is_sqrt_2() {
        let s2 = std::f64::consts::SQRT_2;
        assert_within_ulps(exp2_strict(0.5), s2, 2, "exp2_strict(0.5)");
        assert_within_ulps(exp2_compensated(0.5), s2, 2, "exp2_compensated(0.5)");
    }

    // Composed-form accuracy: exp2(x) = exp(x · ln 2). The
    // multiplication `x · LN_2_F64` introduces a relative error of
    // about `ulp(x · ln 2)`, which propagates as relative error in the
    // final output. For |x| up to ~100, this is bounded by ~16 ulps.
    // The "binary-scaled" kernel state from open question #5 (Sweep 36+)
    // will close this gap; for now we document the achievable budget.

    #[test]
    fn exp2_strict_within_budget() {
        check_strategy(exp2_strict, "exp2_strict", 32);
    }

    #[test]
    fn exp2_compensated_within_budget() {
        check_strategy(exp2_compensated, "exp2_compensated", 32);
    }

    #[test]
    fn exp2_correctly_rounded_within_budget() {
        check_strategy(exp2_correctly_rounded, "exp2_correctly_rounded", 32);
    }

    // ── Session-aware tests ─────────────────────────────────────────────

    #[test]
    fn exp2_session_matches_strict() {
        let mut session = TamSession::new();
        let samples: &[f64] = &[0.5, 1.5, 2.5, 10.5, 0.1, -0.5, -10.5, std::f64::consts::PI];
        for &x in samples {
            let via_session = exp2_session(&mut session, x);
            let via_strict = exp2_strict(x);
            let dist = ulps_between(via_session, via_strict);
            // Session uses ExpKernelState (strict-tier polynomial),
            // strict uses exp_strict directly — both consume the same
            // expm1_small_strict core, so results should agree.
            assert!(
                dist <= 4,
                "exp2_session({x}) vs exp2_strict({x}): {dist} ulps"
            );
        }
    }

    #[test]
    fn exp2_session_shares_kernel_state() {
        // exp2_session and exp2_session for the SAME x must share the
        // cached ExpKernelState in TamSession.
        let mut session = TamSession::new();
        let _ = exp2_session(&mut session, 1.5);
        assert_eq!(session.len(), 1, "first call should register one state");
        let _ = exp2_session(&mut session, 1.5);
        assert_eq!(session.len(), 1, "second call with same x must hit cache");
    }

    #[test]
    fn exp2_session_distinct_inputs_separate_entries() {
        let mut session = TamSession::new();
        let _ = exp2_session(&mut session, 1.5);
        let _ = exp2_session(&mut session, 2.5);
        // Two different x values produce two different ExpKernelState
        // entries (the cache key includes x_bits).
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn exp2_session_handles_integer_exact_path() {
        // For integer x, the session path should still hit the exact-
        // ldexp short-circuit (no kernel state computed).
        let mut session = TamSession::new();
        let v = exp2_session(&mut session, 10.0);
        assert_eq!(v, 1024.0, "integer-exact path should return exact 2^10");
        assert_eq!(session.len(), 0, "integer-exact path doesn't register a state");
    }

    // ── Mathematical identities ────────────────────────────────────────

    #[test]
    fn exp2_log2_consistency_via_log() {
        // 2^(log2(x)) = x. We approximate log2(x) = log(x)/ln(2).
        use super::super::log::log_correctly_rounded;
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 0.5, 10.0, 100.0];
        for &x in xs {
            let log2_x = log_correctly_rounded(x) / LN_2_F64;
            let rt = exp2_strict(log2_x);
            let dist = ulps_between(rt, x);
            assert!(dist <= 8, "2^(log2({x})) = {rt}, {dist} ulps from {x}");
        }
    }

    #[test]
    fn exp2_monotone_on_positive_range() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut prev = exp2_strict(xs[0]);
        for &x in &xs[1..] {
            let y = exp2_strict(x);
            assert!(y >= prev, "exp2 not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
