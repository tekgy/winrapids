//! Micro-regime classifier — 7-code per-bucket market state.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over
//! pointwise comparisons against documented thresholds. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The regime classifier maps a bucket's four high-level signals —
//! event count, effective spread, realized volatility, and lag-1
//! autocorrelation of returns — into one of seven ordinal regimes:
//!
//! ```text
//! 0 = quiet      — few events, tight spread, low vol, near-zero autocorr
//! 1 = normal     — baseline micro-state (the fall-through bucket)
//! 2 = active     — elevated event count without volatility spike
//! 3 = volatile   — elevated volatility, non-trending
//! 4 = trending   — strong positive autocorrelation of returns
//! 5 = stressed   — volatility + wide spread (liquidity withdrawal)
//! 6 = illiquid   — wide spread without volume to back it up
//! ```
//!
//! The decision tree is a documented heuristic, not an estimated model.
//! Thresholds are exposed as parameters with defaults chosen to match
//! the SIP per-signal-compute-spec intent; domain experts override
//! them per-deployment.
//!
//! ```text
//! if n_events == 0:                                   → 0  (quiet, empty bucket)
//! if spread > thresholds.illiquid_spread
//!         and n_events < thresholds.illiquid_events:  → 6  (illiquid)
//! if vol    > thresholds.stressed_vol
//!         and spread > thresholds.stressed_spread:    → 5  (stressed)
//! if autocorr > thresholds.trending_autocorr:         → 4  (trending)
//! if vol    > thresholds.volatile_vol:                → 3  (volatile)
//! if n_events > thresholds.active_events:             → 2  (active)
//! if n_events < thresholds.quiet_events
//!         and vol < thresholds.quiet_vol
//!         and spread < thresholds.quiet_spread:       → 0  (quiet)
//! else:                                                → 1  (normal)
//! ```
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! bucket `regime` header field is one byte (u8) encoding the regime
//! code. This recipe is computed as a CPU post-process on the per-
//! bucket signals — no new accumulation, no new gather, just a cascade
//! of pointwise comparisons.
//!
//! # Composition
//!
//! - **Seven pointwise comparisons**, each lowering to
//!   `accumulate(..., Elementwise, Op::Cmp, ...)` or evaluated directly
//!   on the bucket's scalar aggregates. No cross-bucket dependency.
//! - **Fall-through** to the "normal" code when no rule fires.
//!
//! # NaN/Inf policy
//!
//! - Any non-finite input → returns code `1` (normal) by fall-through.
//!   Classifier rules only fire on strictly-finite comparisons, so
//!   missing data maps to the neutral state rather than one of the
//!   outlier regimes.
//! - Negative inputs for `n_events`, `spread`, or `vol` fall through
//!   to normal for the same reason.
//!
//! # Default parameters
//!
//! See `RegimeThresholds::default` for the full default set. They are
//! calibrated for the SIP signal-compute-spec's per-bucket cadence
//! (100 ms, crypto-tick data in USD units). Override per deployment
//! via `regime_classify_with_thresholds`.

/// Decision thresholds for the regime classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeThresholds {
    /// Event count above which a bucket is "active" (code 2).
    pub active_events: u64,
    /// Event count below which a bucket is eligible for "quiet" (code 0).
    pub quiet_events: u64,
    /// Realized-volatility threshold below which a bucket is eligible
    /// for "quiet" (code 0).
    pub quiet_vol: f64,
    /// Spread threshold below which a bucket is eligible for "quiet"
    /// (code 0).
    pub quiet_spread: f64,
    /// Realized-volatility threshold above which a bucket is "volatile"
    /// (code 3) in the absence of a spread spike.
    pub volatile_vol: f64,
    /// Autocorrelation threshold above which a bucket is "trending"
    /// (code 4).
    pub trending_autocorr: f64,
    /// Realized-volatility threshold for the "stressed" regime (code 5).
    /// Must be ≥ `volatile_vol` to preserve the cascade ordering.
    pub stressed_vol: f64,
    /// Spread threshold for the "stressed" regime (code 5).
    pub stressed_spread: f64,
    /// Spread threshold for the "illiquid" regime (code 6) — wide
    /// spread without the volume to back it up.
    pub illiquid_spread: f64,
    /// Event-count ceiling for the "illiquid" regime — if `n_events`
    /// is above this and the spread is wide, we treat it as stressed
    /// (supply + demand, just wide) rather than illiquid (no flow).
    pub illiquid_events: u64,
}

impl Default for RegimeThresholds {
    /// SIP-spec defaults. Volatility thresholds are in the same units
    /// as the bucket's `realized_vol` field; spread thresholds are in
    /// the same units as the bucket's `roll_spread` (return units).
    /// Event-count thresholds are integer counts of trades per 100 ms
    /// bucket.
    fn default() -> Self {
        Self {
            // "Active" fires when the bucket has an elevated event rate.
            // At 100 ms cadence, crypto majors average 1–5 trades per
            // bucket; 20+ is elevated.
            active_events: 20,
            // "Quiet" requires essentially no trading activity AND calm
            // microstructure.
            quiet_events: 2,
            quiet_vol: 1e-4,     // 1 bp return stdev within the bucket
            quiet_spread: 1e-4,  // 1 bp implied spread
            // "Volatile" — a vol spike on its own.
            volatile_vol: 2e-3,  // 20 bps within-bucket return stdev
            // "Trending" — strong directional autocorrelation.
            trending_autocorr: 0.5,
            // "Stressed" — vol spike with spread widening simultaneously.
            stressed_vol: 5e-3,    // 50 bps
            stressed_spread: 5e-4, // 5 bps
            // "Illiquid" — wide spread with starved flow.
            illiquid_spread: 5e-4, // 5 bps
            illiquid_events: 2,
        }
    }
}

/// Regime classifier with SIP-default thresholds.
///
/// See `RegimeThresholds::default` and the module docs for the decision
/// tree.
#[inline]
pub fn regime_classify(n_events: u64, spread: f64, realized_vol: f64, autocorr: f64) -> u8 {
    regime_classify_with_thresholds(
        n_events,
        spread,
        realized_vol,
        autocorr,
        &RegimeThresholds::default(),
    )
}

/// Regime classifier with a caller-supplied threshold set.
pub fn regime_classify_with_thresholds(
    n_events: u64,
    spread: f64,
    realized_vol: f64,
    autocorr: f64,
    t: &RegimeThresholds,
) -> u8 {
    // Empty bucket is a distinct regime from "normal" — no trading
    // happened at all.
    if n_events == 0 {
        return 0;
    }

    // Illiquid: wide spread with starved flow.
    if spread.is_finite()
        && spread > t.illiquid_spread
        && n_events < t.illiquid_events
    {
        return 6;
    }

    // Stressed: simultaneous vol spike + wide spread.
    if realized_vol.is_finite()
        && spread.is_finite()
        && realized_vol > t.stressed_vol
        && spread > t.stressed_spread
    {
        return 5;
    }

    // Trending: directional autocorrelation.
    if autocorr.is_finite() && autocorr > t.trending_autocorr {
        return 4;
    }

    // Volatile: vol spike on its own.
    if realized_vol.is_finite() && realized_vol > t.volatile_vol {
        return 3;
    }

    // Active: elevated flow.
    if n_events > t.active_events {
        return 2;
    }

    // Quiet: nothing-is-happening state (requires all three signals
    // below their respective quiet thresholds and few events).
    if n_events < t.quiet_events
        && realized_vol.is_finite()
        && realized_vol < t.quiet_vol
        && spread.is_finite()
        && spread < t.quiet_spread
    {
        return 0;
    }

    // Fall-through — baseline bucket.
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bucket_is_quiet() {
        assert_eq!(regime_classify(0, 1e-3, 1e-3, 0.0), 0);
    }

    #[test]
    fn normal_fallthrough() {
        // Moderate everything → 1 (normal).
        assert_eq!(regime_classify(10, 3e-4, 5e-4, 0.1), 1);
    }

    #[test]
    fn active_regime_on_high_event_count() {
        // Lots of events but otherwise calm → 2 (active).
        assert_eq!(regime_classify(100, 1e-4, 5e-4, 0.1), 2);
    }

    #[test]
    fn volatile_regime_on_vol_spike() {
        assert_eq!(regime_classify(10, 1e-4, 5e-3, 0.1), 3);
    }

    #[test]
    fn trending_regime_on_high_autocorr() {
        assert_eq!(regime_classify(10, 1e-4, 1e-4, 0.8), 4);
    }

    #[test]
    fn stressed_beats_volatile_when_spread_also_wide() {
        // High vol + wide spread → stressed (5) wins over volatile (3).
        assert_eq!(regime_classify(10, 1e-3, 1e-2, 0.1), 5);
    }

    #[test]
    fn illiquid_on_wide_spread_low_events() {
        // Wide spread + 1 trade → illiquid (6).
        assert_eq!(regime_classify(1, 1e-3, 1e-4, 0.0), 6);
    }

    #[test]
    fn wide_spread_with_activity_is_stressed_not_illiquid() {
        // Wide spread but plenty of flow → stressed if vol is high,
        // normal if vol is low — NOT illiquid.
        let code = regime_classify(50, 1e-3, 1e-4, 0.0);
        assert_ne!(code, 6, "shouldn't be illiquid with 50 events");
    }

    #[test]
    fn quiet_requires_all_three_calm_signals() {
        // Quiet spread + quiet vol + 1 event → 0 (quiet).
        assert_eq!(regime_classify(1, 5e-5, 5e-5, 0.0), 0);

        // Same but with vol above quiet threshold → not quiet.
        let code = regime_classify(1, 5e-5, 5e-4, 0.0);
        assert_ne!(code, 0);
    }

    #[test]
    fn non_finite_inputs_fall_through_to_normal() {
        // All-NaN comparisons should not fire — bucket is classified
        // by its finite fields only.
        let code = regime_classify(10, f64::NAN, f64::NAN, f64::NAN);
        // No comparisons match → falls through to "normal" (1).
        assert_eq!(code, 1);
    }

    #[test]
    fn threshold_override_changes_outcome() {
        // With default thresholds, (events=10, vol=3e-3) is volatile (3).
        assert_eq!(regime_classify(10, 1e-4, 3e-3, 0.1), 3);

        // Raise the volatile threshold above 3e-3 → same inputs drop
        // to normal (1).
        let mut t = RegimeThresholds::default();
        t.volatile_vol = 1e-2;
        assert_eq!(
            regime_classify_with_thresholds(10, 1e-4, 3e-3, 0.1, &t),
            1
        );
    }

    #[test]
    fn cascade_ordering_respected() {
        // When multiple rules could fire, the earlier (more specific)
        // one wins. Illiquid beats stressed when events are starved.
        let t = RegimeThresholds::default();
        // n_events=1 (below illiquid_events=2), spread=1e-3 (above
        // illiquid_spread=5e-4), vol=1e-2 (above stressed_vol=5e-3),
        // autocorr=0.8 (above trending_autocorr=0.5).
        // All four rules technically fire; illiquid (6) should win.
        let code = regime_classify_with_thresholds(1, 1e-3, 1e-2, 0.8, &t);
        assert_eq!(code, 6);
    }

    #[test]
    fn regime_output_in_valid_range() {
        // For any plausible input the regime code must be in 0..=6.
        let inputs: Vec<(u64, f64, f64, f64)> = vec![
            (0, 0.0, 0.0, 0.0),
            (100, 1e-5, 1e-5, 0.0),
            (1, 1e-3, 1e-5, 0.0),
            (50, 5e-4, 1e-2, 0.2),
            (10, 1e-4, 1e-4, 0.9),
            (10, 1e-4, 3e-3, 0.1),
            (5, 1e-4, 1e-4, 0.1),
        ];
        for (n, sp, v, ac) in inputs {
            let c = regime_classify(n, sp, v, ac);
            assert!(c <= 6, "regime code out of range: {c}");
        }
    }
}
