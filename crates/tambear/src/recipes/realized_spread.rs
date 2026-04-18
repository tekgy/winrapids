//! Realized spread — market-maker profitability after price reversion.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! lookforward gather + pointwise arithmetic + Kulisch-backed averaging.
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! For each trade at time `t` with execution price `p` and bid-ask
//! midpoint `m_t`, the **realized spread** measures price reversion
//! after a lookforward window `Δ`:
//!
//! ```text
//! signed_distance     =  side · (p − m_t)
//! reversion           =  side · (m_t − m_{t+Δ})
//! realized_spread_t   =  2 · (signed_distance + reversion)
//!                     =  2 · side · (p − m_{t+Δ})
//! ```
//!
//! where `side` is `+1` for buys and `−1` for sells. Equivalently, the
//! realized spread is twice the signed gap between trade price and the
//! midpoint Δ later — capturing how much of the effective spread the
//! market maker keeps after the price drifts.
//!
//! Per-bucket realized spread is the average of `realized_spread_t`
//! over trades within the bucket whose lookforward midpoint exists.
//!
//! # Reference
//!
//! Huang, R. D., & Stoll, H. R. (1996). Dealer versus auction markets:
//! A paired comparison of execution costs on NASDAQ and the NYSE.
//! *Journal of Financial Economics* 41(3): 313–357.
//!
//! # Composition
//!
//! - **Lookforward gather** — for each trade index `i`, retrieve the
//!   midpoint at index `i + lookforward_steps`. Lowers to `gather(mids,
//!   Addressing::Offset{+lookforward_steps})` once that addressing is
//!   wired; today computed via direct slice indexing.
//! - **Pointwise arithmetic** — `2 · side · (p − future_mid)` per trade.
//! - **Kulisch-backed average** — `math::mean` over the per-trade
//!   realized spreads.
//!
//! # NaN/Inf policy
//!
//! - Trades for which the lookforward midpoint is unavailable
//!   (insufficient future data) contribute nothing to the average.
//! - Trades with non-finite price, midpoint, or future midpoint are
//!   skipped via `math::mean`'s `is_finite` filter.
//! - Empty input or no usable trades → returns NaN.
//!
//! # Default parameters
//!
//! - `lookforward_steps` — caller-supplied. SIP per signal-compute-spec
//!   uses 10 buckets (1 second at 100ms cadence).

/// Realized spread averaged across trades in a window.
///
/// Inputs are aligned slices of length `n` over the window's trades:
/// - `prices[i]` — execution price of trade i
/// - `sides[i]` — `+1` (buy) or `-1` (sell); other values skip the trade
/// - `midpoints[i]` — bid-ask midpoint at the time of trade i
///
/// `lookforward_steps` is the offset (in trades) at which to read the
/// "future" midpoint for reversion. Trades for which `i +
/// lookforward_steps >= n` are dropped from the average.
///
/// Returns the per-trade-averaged realized spread (in price units).
/// Returns NaN if no eligible trades exist or any required input is
/// non-finite.
///
/// # Panics
///
/// Panics if the four input slices have mismatched lengths or if
/// `lookforward_steps == 0`.
pub fn realized_spread(
    prices: &[f64],
    sides: &[i8],
    midpoints: &[f64],
    lookforward_steps: usize,
) -> f64 {
    assert_eq!(
        prices.len(),
        sides.len(),
        "realized_spread: prices and sides must match length"
    );
    assert_eq!(
        prices.len(),
        midpoints.len(),
        "realized_spread: prices and midpoints must match length"
    );
    assert!(
        lookforward_steps > 0,
        "realized_spread: lookforward_steps must be > 0"
    );

    let n = prices.len();
    if n <= lookforward_steps {
        return f64::NAN;
    }

    // Per-trade realized spread, dropping any with non-finite inputs or
    // sides that are not ±1.
    let mut per_trade: Vec<f64> = Vec::with_capacity(n - lookforward_steps);
    for i in 0..(n - lookforward_steps) {
        let p = prices[i];
        let m_now = midpoints[i];
        let m_future = midpoints[i + lookforward_steps];
        let side = sides[i];
        if !p.is_finite() || !m_now.is_finite() || !m_future.is_finite() {
            continue;
        }
        let side_f = match side {
            1 => 1.0_f64,
            -1 => -1.0_f64,
            _ => continue,
        };
        per_trade.push(2.0 * side_f * (p - m_future));
    }

    crate::math::mean(&per_trade)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_too_short_returns_nan() {
        assert!(realized_spread(&[], &[], &[], 1).is_nan());
        // 3 trades, lookforward 5 → not enough future to evaluate any.
        let p = vec![100.0, 101.0, 102.0];
        let s = vec![1_i8, -1, 1];
        let m = vec![100.0, 101.0, 102.0];
        assert!(realized_spread(&p, &s, &m, 5).is_nan());
    }

    #[test]
    fn no_drift_zero_realized_spread() {
        // Midpoint constant, trades at midpoint → realized spread is 0.
        let p = vec![100.0; 10];
        let s = vec![1_i8; 10];
        let m = vec![100.0; 10];
        let rs = realized_spread(&p, &s, &m, 1);
        assert!((rs - 0.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    fn buy_at_premium_no_drift_positive_realized() {
        // Buy at 100.5 when mid=100, mid stays 100 → realized = 2·1·(100.5 - 100) = 1.0
        let p = vec![100.5; 10];
        let s = vec![1_i8; 10];
        let m = vec![100.0; 10];
        let rs = realized_spread(&p, &s, &m, 1);
        assert!((rs - 1.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    fn sell_at_discount_no_drift_positive_realized() {
        // Sell at 99.5 when mid=100, mid stays 100 → realized = 2·(-1)·(99.5 - 100) = 2·(-1)·(-0.5) = 1.0
        let p = vec![99.5; 10];
        let s = vec![-1_i8; 10];
        let m = vec![100.0; 10];
        let rs = realized_spread(&p, &s, &m, 1);
        assert!((rs - 1.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    fn skips_invalid_sides() {
        // Sides of 0 (unknown) → trade is dropped; remaining ones average.
        let p = vec![100.5, 100.5, 100.5];
        let s = vec![0_i8, 1, 0];
        let m = vec![100.0, 100.0, 100.0];
        let rs = realized_spread(&p, &s, &m, 1);
        // i=0 dropped (side=0), i=1 contributes 2·1·(100.5-100) = 1.0,
        // i=2 has no future. So mean of [1.0] = 1.0.
        assert!((rs - 1.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    fn skips_non_finite_inputs() {
        let p = vec![100.5, f64::NAN, 100.5];
        let s = vec![1_i8, 1, 1];
        let m = vec![100.0, 100.0, 100.0];
        let rs = realized_spread(&p, &s, &m, 1);
        // i=1 dropped (NaN price), i=0 has future, i=2 has no future.
        // i=0 contributes 2·1·(100.5 - 100) = 1.0. Mean of [1.0] = 1.0.
        assert!((rs - 1.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    fn lookforward_window_picks_correct_future() {
        // 5 trades. With lookforward=2, only trades 0..2 are eligible.
        let p = vec![100.5, 100.5, 100.5, 100.5, 100.5];
        let s = vec![1_i8; 5];
        let m = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let rs = realized_spread(&p, &s, &m, 2);
        // All eligible trades give realized = 1.0.
        assert!((rs - 1.0).abs() < 1e-12, "got {rs}");
    }

    #[test]
    #[should_panic(expected = "match length")]
    fn panics_on_mismatched_lengths() {
        let _ = realized_spread(&[100.0, 101.0], &[1_i8], &[100.0, 101.0], 1);
    }

    #[test]
    #[should_panic(expected = "lookforward_steps")]
    fn panics_on_zero_lookforward() {
        let _ = realized_spread(&[100.0], &[1_i8], &[100.0], 0);
    }
}
