//! Maximum drawdown — worst peak-to-trough decline and its location in
//! the price series.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! prefix-max scan + pointwise ratio + three argmax reductions. See
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
//!
//! trough_index    =  argmax_t drawdown[t]
//! peak_index      =  argmax_{t ≤ trough_index} p[t]       // peak leading into MDD
//! recovery_index  =  min{t > trough_index : p[t] ≥ p[peak_index]}
//!                    = None (u32::MAX)  if no such t exists
//! ```
//!
//! `max_drawdown` is returned as a non-negative fraction in `[0, 1]`.
//! A return of 0 means no drawdown was observed (price never declined
//! from its running peak); 0.5 means the worst observed drawdown was
//! 50% from a peak.
//!
//! The three indices together tell the full story of the worst
//! drawdown event:
//!
//! - `peak_index` — when the run-up topped out.
//! - `trough_index` — when the decline bottomed out.
//! - `recovery_index` — when (if at all) the price returned to the
//!   prior peak level.
//!
//! For a series that never recovers within the window,
//! `recovery_index` is the sentinel `u32::MAX`.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` (v2,
//! 2026-04-22) the per-hour `max_drawdown` kernel is a four-output
//! multi-output:
//!
//! - `max_drawdown_pct: f32`            → `[904:908)`
//! - `max_drawdown_start_bucket: u32`   → `[1875:1879)`  (peak)
//! - `max_drawdown_end_bucket: u32`     → `[1879:1883)`  (trough)
//! - `max_drawdown_recovery_bucket: u32` → `[1883:1887)`
//!
//! All four come from the same three passes over the price series —
//! the recipe contributes them together.
//!
//! # Composition
//!
//! - **prefix-max** over the price series — Kingdom A scan with
//!   `Op::Max`. Lowers to `accumulate(prices, Grouping::Prefix,
//!   Op::Max)` once the prefix-Max atom path lands; today a sequential
//!   running-max loop.
//! - **pointwise ratio** for per-step drawdown.
//! - **argmax-reduce** over the drawdown series — identifies
//!   `trough_index` via `accumulate(drawdowns, All, Op::MaxIdx)`.
//! - **argmax-reduce** over the prefix-restricted price — identifies
//!   `peak_index` via `accumulate(prices[..=trough_index], All,
//!   Op::MaxIdx)`.
//! - **sequential scan** for `recovery_index` — no atomic equivalent,
//!   small `O(n − trough_index)` postprocess.
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element.
//!
//! - Empty input → `pct = 0.0`, all indices `u32::MAX`.
//! - Any non-finite or non-positive price → `pct = NaN`, all indices
//!   `u32::MAX`. A non-finite price is a data error worth surfacing
//!   rather than silently filtering; drawdown's multiplicative nature
//!   makes skip semantics unclean (a gap in the series breaks the
//!   peak-to-trough relationship).
//! - Series with no peak-to-trough decline → `pct = 0.0`, `peak_index
//!   = 0`, `trough_index = 0`, `recovery_index = 0` (trivially
//!   recovered at the very first observation, since the "drawdown"
//!   was zero).
//!
//! # Default parameters
//!
//! None. Drawdown is parameter-free; the worst-decline definition is
//! universal.

/// Result of the maximum-drawdown analysis over a price series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxDrawdownResult {
    /// Worst peak-to-trough decline as a non-negative fraction in
    /// `[0, 1]`. NaN on invalid input.
    pub pct: f64,
    /// Index of the running-peak price leading into the worst
    /// drawdown. `u32::MAX` on invalid input.
    pub peak_index: u32,
    /// Index of the worst drawdown's trough. `u32::MAX` on invalid
    /// input.
    pub trough_index: u32,
    /// Index of the first observation after the trough at which the
    /// price matches or exceeds the prior peak. `u32::MAX` when the
    /// series never recovers (or is empty / invalid).
    pub recovery_index: u32,
}

impl MaxDrawdownResult {
    #[inline]
    pub fn nan() -> Self {
        Self {
            pct: f64::NAN,
            peak_index: u32::MAX,
            trough_index: u32::MAX,
            recovery_index: u32::MAX,
        }
    }

    /// Result for the degenerate case of an empty input — no drawdown
    /// observed, no indices to report.
    #[inline]
    pub fn empty() -> Self {
        Self {
            pct: 0.0,
            peak_index: u32::MAX,
            trough_index: u32::MAX,
            recovery_index: u32::MAX,
        }
    }
}

/// Maximum drawdown over a price series, with peak / trough / recovery
/// indices.
///
/// See module docs for the math and edge cases. The returned struct's
/// `pct` field matches the scalar result of the previous scalar-only
/// form of this recipe.
pub fn max_drawdown(prices: &[f64]) -> MaxDrawdownResult {
    if prices.is_empty() {
        return MaxDrawdownResult::empty();
    }
    // Reject any non-finite or non-positive price up front — drawdown
    // is only meaningful for a strictly-positive price series, and a
    // non-finite price is a data error worth surfacing rather than
    // silently skipping.
    for &p in prices {
        if !p.is_finite() || p <= 0.0 {
            return MaxDrawdownResult::nan();
        }
    }

    // Pass 1 — track running max, worst drawdown, and the indices.
    let mut running_max = prices[0];
    let mut running_max_index: usize = 0;
    let mut worst: f64 = 0.0;
    let mut trough_index: usize = 0;
    let mut peak_for_worst: usize = 0;
    for (i, &p) in prices.iter().enumerate().skip(1) {
        if p > running_max {
            running_max = p;
            running_max_index = i;
        } else {
            let dd = (running_max - p) / running_max;
            if dd > worst {
                worst = dd;
                trough_index = i;
                peak_for_worst = running_max_index;
            }
        }
    }

    // Pass 2 — find recovery_index: first t > trough_index where
    // p[t] >= prices[peak_for_worst]. If never, u32::MAX.
    let peak_price = prices[peak_for_worst];
    let mut recovery_index: u32 = u32::MAX;
    for i in (trough_index + 1)..prices.len() {
        if prices[i] >= peak_price {
            recovery_index = i as u32;
            break;
        }
    }

    // If worst is 0.0 (no drawdown ever observed), the "peak" and
    // "trough" degenerate to the first observation and the series
    // trivially never left the peak — report index 0 for both and
    // flag recovery at 0 (already recovered).
    if worst == 0.0 {
        return MaxDrawdownResult {
            pct: 0.0,
            peak_index: 0,
            trough_index: 0,
            recovery_index: 0,
        };
    }

    MaxDrawdownResult {
        pct: worst,
        peak_index: peak_for_worst as u32,
        trough_index: trough_index as u32,
        recovery_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_series_zero_drawdown() {
        let r = max_drawdown(&[]);
        assert_eq!(r.pct, 0.0);
        assert_eq!(r.peak_index, u32::MAX);
        assert_eq!(r.trough_index, u32::MAX);
        assert_eq!(r.recovery_index, u32::MAX);
    }

    #[test]
    fn single_price_no_drawdown() {
        let r = max_drawdown(&[100.0]);
        assert_eq!(r.pct, 0.0);
    }

    #[test]
    fn monotonically_increasing_no_drawdown() {
        let p = vec![100.0, 101.0, 105.0, 110.0, 200.0];
        let r = max_drawdown(&p);
        assert_eq!(r.pct, 0.0);
        // No drawdown ever observed → all indices trivial.
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 0);
        assert_eq!(r.recovery_index, 0);
    }

    #[test]
    fn simple_50pct_drawdown_reports_indices() {
        // Peak 100 at index 0 → trough 50 at index 1 → no recovery.
        let p = vec![100.0, 50.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.5).abs() < 1e-12);
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 1);
        assert_eq!(r.recovery_index, u32::MAX);
    }

    #[test]
    fn drawdown_with_recovery() {
        // 100 → 50 → 100: peak at 0, trough at 1, recovery at 2.
        let p = vec![100.0, 50.0, 100.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.5).abs() < 1e-12);
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 1);
        assert_eq!(r.recovery_index, 2);
    }

    #[test]
    fn drawdown_partial_recovery_no_full_recovery() {
        // 100 → 50 → 90 → 99 → never hits 100.
        let p = vec![100.0, 50.0, 90.0, 99.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.5).abs() < 1e-12);
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 1);
        assert_eq!(r.recovery_index, u32::MAX);
    }

    #[test]
    fn drawdown_picks_worst_against_running_peak() {
        // 100 → 60 (40%) → 200 (new peak) → 100 (50% from 200) → 110.
        // Worst = 50% from peak 200 (at index 2) to trough 100 (at 3).
        // No recovery to 200 before series ends.
        let p = vec![100.0, 60.0, 200.0, 100.0, 110.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.5).abs() < 1e-12);
        assert_eq!(r.peak_index, 2);
        assert_eq!(r.trough_index, 3);
        assert_eq!(r.recovery_index, u32::MAX);
    }

    #[test]
    fn drawdown_recovers_from_worst() {
        // 100 → 60 → 50 (worst, 50%) → 80 → 110 (full recovery).
        let p = vec![100.0, 60.0, 50.0, 80.0, 110.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.5).abs() < 1e-12);
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 2);
        assert_eq!(r.recovery_index, 4);
    }

    #[test]
    fn drawdown_nan_on_non_finite_price() {
        let p = vec![100.0, f64::NAN, 50.0];
        let r = max_drawdown(&p);
        assert!(r.pct.is_nan());
        assert_eq!(r.peak_index, u32::MAX);
    }

    #[test]
    fn drawdown_nan_on_non_positive_price() {
        let p = vec![100.0, 0.0, 50.0];
        assert!(max_drawdown(&p).pct.is_nan());
        let p = vec![100.0, -50.0];
        assert!(max_drawdown(&p).pct.is_nan());
    }

    #[test]
    fn drawdown_in_unit_interval() {
        let p = vec![100.0, 80.0, 120.0, 90.0, 200.0, 150.0, 75.0, 180.0];
        let r = max_drawdown(&p);
        assert!((0.0..=1.0).contains(&r.pct));
        // peak_index should point at a bar with price 200 (index 4),
        // trough at the 75-valued bar (index 6). Recovery at the
        // 180-valued bar — never hits 200 again.
        assert_eq!(r.peak_index, 4);
        assert_eq!(r.trough_index, 6);
        assert_eq!(r.recovery_index, u32::MAX);
    }

    #[test]
    fn drawdown_handles_recovery_at_equality() {
        // Price touches the peak exactly on recovery — the `>=` test
        // must fire at the equality.
        let p = vec![100.0, 80.0, 100.0];
        let r = max_drawdown(&p);
        assert!((r.pct - 0.2).abs() < 1e-12);
        assert_eq!(r.peak_index, 0);
        assert_eq!(r.trough_index, 1);
        assert_eq!(r.recovery_index, 2);
    }
}
