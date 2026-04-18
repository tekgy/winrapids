//! Realized volatility — annualized, computed from sub-sampled returns.
//!
//! Locked vocabulary: this is a Tier 4 recipe — pure composition over
//! a sparse-stride pick + log-return computation + Kulisch-backed
//! sum-of-squares + sqrt + annualization. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Given a per-bucket close-price series `closes[0..n]` and a sub-sampling
//! `stride` (in buckets), realized volatility on the sub-sampled returns:
//!
//! ```text
//! sub_closes  =  closes[0], closes[stride], closes[2*stride], ..., closes[k*stride]
//! sub_rets[i] =  ln(sub_closes[i+1] / sub_closes[i])
//! rv          =  sqrt(Σ sub_rets[i]²)
//! rv_annual   =  rv · sqrt(annualization_factor)
//! ```
//!
//! This is the standard Andersen-Bollerslev-Diebold-Labys realized
//! volatility, computed at a chosen sampling frequency. Sub-sampling at
//! lower frequency than the bucket grid reduces microstructure noise
//! contamination at the cost of fewer samples.
//!
//! # Annualization factor
//!
//! For per-bucket SIP data:
//!
//! - 100ms buckets, `stride=3000` → 5-minute returns. With 252 trading
//!   days × 24 hours × 12 five-minute periods per hour = 72,576 → set
//!   `annualization_factor = 72_576.0 / k` where `k` is the count of
//!   sub-returns observed. (For SIP's hourly file, k = 11 sub-returns,
//!   so factor = 72,576 / 11.)
//!
//! Caller supplies the annualization factor explicitly. We don't bake
//! a calendar assumption into the recipe.
//!
//! # Composition
//!
//! - **Sparse-stride pick** — gather (locked-Tier-3 atom) every
//!   `stride`-th element. Today written as a direct Rust slice index;
//!   will lower to `gather(closes, Addressing::Strided{stride})`.
//! - **Log-return computation** — pointwise `ln(close[i+1] / close[i])`.
//! - **Sum of squares** — `tambear::math::sum_sq` (Kulisch-exact).
//! - **Sqrt + multiply** — primitive arithmetic.
//!
//! # NaN/Inf policy
//!
//! - Empty / single-close input → returns 0.0 (no return observed).
//! - Any sub-sampled close is non-positive → returns NaN (log-return
//!   undefined).
//! - Any sub-sampled close is non-finite → returns NaN.
//!
//! # Default parameters
//!
//! - `stride` — caller-provided. SIP at 100ms buckets uses 3000 (5 min).
//! - `annualization_factor` — caller-provided. See above for SIP value.

/// Realized volatility from a sparsely-sampled price series, annualized.
///
/// `closes` is the per-bucket close price series.
/// `stride` is the sub-sampling step in buckets (≥ 1).
/// `annualization_factor` multiplies the inside-sqrt sum to produce an
/// annualized volatility (use `1.0` for raw realized volatility on the
/// sub-sample frequency).
///
/// # Panics
///
/// - If `stride == 0`.
/// - If `annualization_factor` is non-finite or negative.
pub fn realized_vol_subsampled(
    closes: &[f64],
    stride: usize,
    annualization_factor: f64,
) -> f64 {
    assert!(stride > 0, "realized_vol_subsampled: stride must be > 0");
    assert!(
        annualization_factor.is_finite() && annualization_factor >= 0.0,
        "realized_vol_subsampled: annualization_factor must be finite and non-negative, got {annualization_factor}"
    );

    if closes.len() < 2 {
        return 0.0;
    }

    // Sparse-stride pick.
    let sub_closes: Vec<f64> = closes.iter().step_by(stride).copied().collect();
    if sub_closes.len() < 2 {
        return 0.0;
    }

    // Compute log-returns; bail out on any non-positive or non-finite close.
    let mut log_rets: Vec<f64> = Vec::with_capacity(sub_closes.len() - 1);
    for w in sub_closes.windows(2) {
        let (a, b) = (w[0], w[1]);
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
            return f64::NAN;
        }
        log_rets.push((b / a).ln());
    }

    let sum_sq = crate::math::sum_sq(&log_rets);
    (sum_sq * annualization_factor).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_single_returns_zero() {
        assert_eq!(realized_vol_subsampled(&[], 1, 1.0), 0.0);
        assert_eq!(realized_vol_subsampled(&[100.0], 1, 1.0), 0.0);
    }

    #[test]
    fn flat_prices_zero_volatility() {
        let p = vec![100.0; 10];
        assert_eq!(realized_vol_subsampled(&p, 1, 1.0), 0.0);
    }

    #[test]
    fn known_value_two_returns() {
        // Two log-returns of magnitude 0.05 each.
        // sub_rets = [ln(105/100), ln(110.25/105)] = [0.04879, 0.04879]
        // sum_sq = 2 · (0.04879)² = 0.004763
        // sqrt = 0.06902
        let p = vec![100.0, 105.0, 110.25];
        let rv = realized_vol_subsampled(&p, 1, 1.0);
        assert!((rv - 0.06902).abs() < 1e-4, "got {rv}");
    }

    #[test]
    fn stride_skips_intermediate_samples() {
        // With stride=2, every other close is picked.
        let p = vec![100.0, 999.0, 105.0, 999.0, 110.25];
        // Picks: [100, 105, 110.25] → same as known_value_two_returns.
        let rv = realized_vol_subsampled(&p, 2, 1.0);
        assert!((rv - 0.06902).abs() < 1e-4, "got {rv}");
    }

    #[test]
    fn annualization_scales_by_sqrt() {
        // Doubling the annualization factor should multiply RV by sqrt(2).
        let p = vec![100.0, 105.0, 110.25];
        let r1 = realized_vol_subsampled(&p, 1, 1.0);
        let r2 = realized_vol_subsampled(&p, 1, 2.0);
        let ratio = r2 / r1;
        assert!((ratio - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn nan_on_non_positive_close() {
        let p = vec![100.0, 0.0, 100.0];
        assert!(realized_vol_subsampled(&p, 1, 1.0).is_nan());
        let p = vec![100.0, -50.0, 100.0];
        assert!(realized_vol_subsampled(&p, 1, 1.0).is_nan());
    }

    #[test]
    fn nan_on_non_finite_close() {
        let p = vec![100.0, f64::NAN, 100.0];
        assert!(realized_vol_subsampled(&p, 1, 1.0).is_nan());
        let p = vec![100.0, f64::INFINITY, 100.0];
        assert!(realized_vol_subsampled(&p, 1, 1.0).is_nan());
    }

    #[test]
    fn stride_too_large_returns_zero() {
        // Only one sub-sample selected → no return observable.
        let p = vec![100.0, 110.0, 120.0];
        assert_eq!(realized_vol_subsampled(&p, 10, 1.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn panics_on_zero_stride() {
        let _ = realized_vol_subsampled(&[100.0, 105.0], 0, 1.0);
    }

    #[test]
    #[should_panic(expected = "annualization_factor")]
    fn panics_on_negative_annualization() {
        let _ = realized_vol_subsampled(&[100.0, 105.0], 1, -1.0);
    }
}
