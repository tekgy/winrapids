//! Roll (1984) implied bid–ask spread from return autocovariance.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! Kulisch-backed autocovariance accumulation + scalar closed form. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Under Roll's (1984) assumption of a constant half-spread `s/2` and
//! uncorrelated underlying price innovations, the first-order
//! autocovariance of observed returns equals `−(s/2)²`. Inverting:
//!
//! ```text
//! γ₁    =  Cov(r[t], r[t−1])
//! s     =  2 · √(−γ₁)              if γ₁ < 0
//! s     =  0                       if γ₁ ≥ 0        (no spread signal)
//! ```
//!
//! Non-negative autocovariance indicates either no bid–ask bounce or a
//! positive-autocorrelation microstructure that violates the Roll model
//! — in both cases the estimator is reported as 0 per convention.
//!
//! The sample autocovariance uses the common mean of the full return
//! series for both `r[t]` and `r[t−1]`, which is the usual Roll
//! formulation. With `μ = (1/n) Σ r[t]` and `m = n−1` paired terms:
//!
//! ```text
//! γ̂₁  =  (1/(m−1)) · Σ (r[t] − μ) · (r[t−1] − μ)
//! ```
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! bucket `roll_spread` header field is defined on log returns:
//!
//! ```text
//! roll_spread = 2 · √(max(0, −Cov(log_ret[i], log_ret[i−1])))
//! ```
//!
//! # Reference
//!
//! Roll, R. (1984). A simple implicit measure of the effective bid–ask
//! spread in an efficient market. *Journal of Finance* 39(4):
//! 1127–1139.
//!
//! # Composition
//!
//! - **Kulisch-backed mean** over returns — one accumulation. Lowers to
//!   `accumulate(rets, Grouping::All, Op::Add, expr=Value)` +
//!   normalization, via `math::mean`.
//! - **Kulisch-backed autocovariance sum** over `(r[t]−μ)·(r[t−1]−μ)` —
//!   one accumulation. Lowers to `accumulate(rets, Grouping::All,
//!   Op::Add, expr=Custom("(v−μ)·(gather(rets,−1)−μ)"))` once the
//!   lag-gather pattern is wired into the atom layer; today computed
//!   via a local `KulischAccumulator`.
//! - **Scalar closed form** — `2 · sqrt(−γ₁)` with a clamp to 0 on
//!   non-negative autocovariance.
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element, and the estimator returns NaN when the data
//! cannot identify it.
//!
//! - Empty, single-return, or fewer than two finite adjacent pairs
//!   after filtering → returns NaN (the autocovariance cannot be
//!   estimated, consistent with `var → NaN` when count=0).
//! - Any non-finite return → that return and its adjacent pair are
//!   skipped at the autocovariance step. The mean uses `math::mean`
//!   which also filters non-finite values.
//! - All returns equal → autocovariance is 0, result is 0 (the Roll
//!   estimator's "no spread detected" output — distinct from "no
//!   data").
//! - Non-negative autocovariance → returns 0 (Roll's fit-but-no-signal
//!   output — also distinct from "no data").
//!
//! The distinction matters for SIP consumers: NaN means "bucket too
//! empty to estimate"; 0.0 means "estimator ran and returned zero."
//!
//! # Default parameters
//!
//! None. Roll's estimator is parameter-free at this level; the upstream
//! choice of return type (log vs arithmetic, mid vs trade price) is the
//! policy decision.

/// Roll's implied spread from a return series.
///
/// Returns the implied full spread `s` in the same units as the returns
/// (i.e. log-return units if `returns` are log returns).
///
/// - **NaN** if the series is empty, too short, or has fewer than two
///   finite adjacent pairs after filtering — the autocovariance cannot
///   be estimated.
/// - **0.0** when the estimator runs but detects no spread (non-
///   negative first-order autocovariance).
/// - Otherwise `2·√(−γ₁)` in return units.
pub fn roll_spread(returns: &[f64]) -> f64 {
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    if returns.len() < 3 {
        return f64::NAN;
    }

    // Mean over finite returns (same convention as math::mean).
    let mu = crate::math::mean(returns);
    if !mu.is_finite() {
        return f64::NAN;
    }

    // Autocovariance of adjacent pairs, Kulisch-exact sum. Skip any pair
    // where either return is non-finite.
    let mut cov_acc = KulischAccumulator::new();
    let mut n_pairs: usize = 0;
    for i in 1..returns.len() {
        let a = returns[i - 1];
        let b = returns[i];
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        cov_acc.add_f64((a - mu) * (b - mu));
        n_pairs += 1;
    }

    if n_pairs < 2 {
        return f64::NAN;
    }

    let gamma1 = cov_acc.to_f64() / (n_pairs - 1) as f64;
    roll_spread_from_autocov(gamma1)
}

/// Roll's implied spread from a pre-computed first-order return
/// autocovariance.
///
/// For the per-bucket SIP pipeline the autocovariance `γ₁` can be
/// produced directly by the accumulate layer:
///
/// ```text
/// accumulate(log_ret, ByKey{bucket}, Op::Add, expr=v · gather(log_ret, −1))
///    ... normalized by (n − 1) after subtracting the mean product.
/// ```
///
/// This thin wrapper applies the closed form `s = 2·√(−γ₁)` with the
/// non-negative clamp. Returns 0.0 if `γ₁` is non-finite or `≥ 0`.
#[inline]
pub fn roll_spread_from_autocov(gamma1: f64) -> f64 {
    if !gamma1.is_finite() || gamma1 >= 0.0 {
        return 0.0;
    }
    2.0 * (-gamma1).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_short_is_nan() {
        // Per the NaN contract, "no data" → NaN (estimator undefined),
        // distinct from 0.0 which means "estimator ran, no spread".
        assert!(roll_spread(&[]).is_nan());
        assert!(roll_spread(&[0.01]).is_nan());
        assert!(roll_spread(&[0.01, 0.02]).is_nan());
    }

    #[test]
    fn all_nan_input_is_nan() {
        // No finite pairs survive → NaN.
        let rets = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        assert!(roll_spread(&rets).is_nan());
    }

    #[test]
    fn flat_returns_zero_spread() {
        // All returns identical → autocovariance is 0 → spread 0 (the
        // estimator ran and detected no spread — not "no data").
        let rets = vec![0.001; 100];
        assert_eq!(roll_spread(&rets), 0.0);
    }

    #[test]
    fn bid_ask_bounce_positive_spread() {
        // Alternating returns around 0 generate strong negative first-
        // order autocovariance, which is the signature Roll exploits.
        //
        //   +0.01, -0.01, +0.01, -0.01, ...
        //
        // Mean ≈ 0, and each pair (r[t-1] - 0)·(r[t] - 0) = -1e-4.
        let rets: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let s = roll_spread(&rets);
        // γ₁ ≈ -1e-4, so s ≈ 2·√(1e-4) = 2·0.01 = 0.02.
        assert!((s - 0.02).abs() < 1e-3, "got {s}");
    }

    #[test]
    fn positive_autocov_returns_zero() {
        // Monotonic-ish ramp → positive autocovariance → reported as 0.
        let rets: Vec<f64> = (0..100).map(|i| i as f64 * 1e-4).collect();
        let s = roll_spread(&rets);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn from_autocov_closed_form() {
        // γ₁ = -0.0025 → s = 2·√0.0025 = 0.1
        assert!((roll_spread_from_autocov(-0.0025) - 0.1).abs() < 1e-12);
        // γ₁ = 0.0 → 0
        assert_eq!(roll_spread_from_autocov(0.0), 0.0);
        // γ₁ = 0.5 → 0 (non-negative clamp)
        assert_eq!(roll_spread_from_autocov(0.5), 0.0);
        // Non-finite γ₁ → 0
        assert_eq!(roll_spread_from_autocov(f64::NAN), 0.0);
        assert_eq!(roll_spread_from_autocov(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn skips_non_finite_pairs() {
        // A single NaN inside a bid-ask-bounce series should not poison
        // the overall spread — its two adjacent pairs are skipped.
        let mut rets: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        rets[50] = f64::NAN;
        let s = roll_spread(&rets);
        // Still dominated by the bounce signal; should be close to 0.02.
        assert!((s - 0.02).abs() < 5e-3, "got {s}");
    }

    #[test]
    fn scales_with_sqrt_autocov() {
        // Doubling |γ₁| should multiply the spread by √2.
        let s1 = roll_spread_from_autocov(-0.01);
        let s2 = roll_spread_from_autocov(-0.02);
        assert!((s2 / s1 - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn non_negative_output() {
        // For any input, roll_spread must be non-negative.
        let cases: [&[f64]; 4] = [
            &[0.01, -0.01, 0.01, -0.01, 0.01],
            &[0.0; 10],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[-5.0, 5.0, -5.0, 5.0, -5.0],
        ];
        for rets in cases {
            let s = roll_spread(rets);
            assert!(s >= 0.0, "negative spread {s} for {rets:?}");
        }
    }
}
