//! Maximum drawdown — worst peak-to-trough decline as a fraction of peak.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! prefix-max scan + pointwise ratio. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Given a price (or wealth) series `p[0..n]`:
//!
//! ```text
//! running_max[t]  =  max(p[0..=t])
//! drawdown[t]     =  (running_max[t] − p[t])  /  running_max[t]
//! max_drawdown    =  max(drawdown[0..n])
//! ```
//!
//! Returned as a non-negative fraction in `[0, 1]`. A return of 0 means
//! no drawdown was observed (price never declined from its running peak);
//! 0.5 means the worst observed drawdown was 50% from a peak.
//!
//! # Composition
//!
//! - **prefix-max** over the price series — Kingdom A scan with `Op::Max`.
//!   Will lower to `accumulate(prices, Grouping::Prefix, Op::Max)` once
//!   the prefix-Max atom path lands; today computed via a sequential
//!   running-max loop in this recipe.
//! - **pointwise ratio** for the per-step drawdown.
//! - **reduce-max** over the drawdown series — Kingdom A reduction with
//!   `Op::Max`. Lowers to `accumulate(drawdowns, Grouping::All, Op::Max)`.
//!
//! # NaN/Inf policy
//!
//! - Empty input → returns 0.0 (no drawdown).
//! - Any non-finite price → returns NaN (a non-finite price is a data
//!   error; emit honestly rather than silently fix).
//! - All non-positive prices → returns NaN (drawdown denominator
//!   undefined).
//!
//! # Default parameters
//!
//! None. Drawdown is parameter-free; the worst-decline definition is
//! universal.

/// Maximum drawdown over a price series.
///
/// Returns a non-negative fraction in `[0, 1]`. See module docs for
/// the math and edge cases.
pub fn max_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    // Reject any non-finite or non-positive price up front — drawdown is only
    // meaningful for a strictly positive price series, and a non-finite price
    // is a data error worth surfacing.
    for &p in prices {
        if !p.is_finite() || p <= 0.0 {
            return f64::NAN;
        }
    }

    let mut running_max = prices[0];
    let mut worst = 0.0_f64;
    for &p in &prices[1..] {
        if p > running_max {
            running_max = p;
        } else {
            // Pointwise drawdown vs the peak so far.
            let dd = (running_max - p) / running_max;
            if dd > worst {
                worst = dd;
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_series_no_drawdown() {
        assert_eq!(max_drawdown(&[]), 0.0);
    }

    #[test]
    fn single_price_no_drawdown() {
        assert_eq!(max_drawdown(&[100.0]), 0.0);
    }

    #[test]
    fn monotonically_increasing_no_drawdown() {
        let p = vec![100.0, 101.0, 105.0, 110.0, 200.0];
        assert_eq!(max_drawdown(&p), 0.0);
    }

    #[test]
    fn simple_50pct_drawdown() {
        // Peak 100, trough 50 → drawdown = 0.5
        let p = vec![100.0, 50.0];
        assert!((max_drawdown(&p) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn drawdown_picks_worst_not_last() {
        // Sequence: 100 → 50 (50%) → 75 → 25 (from peak 100 = 75% drawdown)
        // Wait: at t=3 (price=25), running_max is still 100. So drawdown = 75/100 = 0.75.
        let p = vec![100.0, 50.0, 75.0, 25.0];
        assert!((max_drawdown(&p) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn drawdown_picks_worst_against_running_peak() {
        // 100 → 60 (40%) → 200 (new peak) → 100 (50% from 200) → 110.
        // Worst = 50% (from peak of 200 to 100).
        let p = vec![100.0, 60.0, 200.0, 100.0, 110.0];
        assert!((max_drawdown(&p) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn drawdown_handles_recovery() {
        // 100 → 80 → 100 — peak/trough/peak. Worst drawdown 20%.
        let p = vec![100.0, 80.0, 100.0];
        assert!((max_drawdown(&p) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn drawdown_nan_on_non_finite_price() {
        let p = vec![100.0, f64::NAN, 50.0];
        assert!(max_drawdown(&p).is_nan());
        let p = vec![100.0, f64::INFINITY, 50.0];
        assert!(max_drawdown(&p).is_nan());
    }

    #[test]
    fn drawdown_nan_on_non_positive_price() {
        let p = vec![100.0, 0.0, 50.0];
        assert!(max_drawdown(&p).is_nan());
        let p = vec![100.0, -50.0];
        assert!(max_drawdown(&p).is_nan());
    }

    #[test]
    fn drawdown_in_unit_interval() {
        // Constructed series: prices in [50, 200]. All drawdowns must be in [0, 1].
        let p = vec![100.0, 80.0, 120.0, 90.0, 200.0, 150.0, 75.0, 180.0];
        let dd = max_drawdown(&p);
        assert!((0.0..=1.0).contains(&dd));
    }
}
