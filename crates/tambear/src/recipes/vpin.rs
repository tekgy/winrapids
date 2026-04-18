//! VPIN — Volume-synchronized Probability of Informed trading.
//!
//! Locked vocabulary: this is a Tier 4 recipe — pure composition over
//! pre-computed bucket aggregates. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Per-bucket VPIN as the SIP column-graph defines it (column-graph.md
//! row D3):
//!
//! ```text
//! vpin  =  |buy_qty − sell_qty|  /  (buy_qty + sell_qty)
//!       =  |C11c − C11d|  /  (C11c + C11d)
//! ```
//!
//! where `buy_qty` and `sell_qty` are pre-computed per-bucket sums of
//! signed quantity (Kulisch-backed, deterministic). VPIN ∈ [0, 1]:
//!
//! - 0.0 = perfectly balanced flow (no directional pressure)
//! - 1.0 = entirely one-sided flow (full directional pressure)
//!
//! # Distinction from `vpin_bvc` in volatility.rs
//!
//! This is the **classical per-bucket VPIN** SIP needs as a header
//! field. The `vpin_bvc` recipe in `volatility.rs` is the
//! Bulk-Volume-Classification (BVC) variant from Easley-López de Prado-
//! O'Hara (2012) which uses a Φ(ΔP/σ) classifier across volume
//! buckets rather than tick-level signed quantity. Both are valid
//! VPIN definitions; SIP uses the classical form because tick-level
//! buy/sell classification is available from the venue data.
//!
//! # Reference
//!
//! Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow toxicity
//! and liquidity in a high-frequency world. *Review of Financial
//! Studies* 25(5): 1457–1493.
//!
//! # Composition
//!
//! - One absolute-value, one subtraction, one addition, one division.
//!   Leaf recipe — bottoms out at primitive arithmetic. The buy/sell
//!   sums are themselves Kulisch-backed accumulations done upstream
//!   (locked-tier-3 atom calls), so this recipe inherits exact
//!   summation transitively without doing any accumulation itself.
//!
//! # Default parameters
//!
//! None. The classical VPIN ratio has no tunable parameters at this
//! level; the upstream classification of trades into buy/sell volume
//! is the policy choice (per-trade tick rule vs BVC vs other).

/// Per-bucket VPIN from pre-computed buy and sell volume sums.
///
/// `buy_qty` and `sell_qty` should be the Kulisch-backed sums of base-
/// asset quantity for ticks classified as buys and sells respectively
/// in this bucket. NaN-skip / Inf-skip semantics live at the upstream
/// summation step; this recipe assumes the inputs are finite.
///
/// Returns:
/// - VPIN ∈ [0, 1] when `buy_qty + sell_qty > 0`
/// - 0.0 when both inputs are exactly 0 (empty bucket — no directional
///   pressure observable)
/// - NaN if either input is non-finite or negative (data error upstream)
///
/// The empty-bucket case returns 0.0 rather than NaN to preserve the
/// prefix-sum flatline property: an empty bucket's VPIN can be
/// included in a running average without poisoning the result.
#[inline]
pub fn vpin(buy_qty: f64, sell_qty: f64) -> f64 {
    if !buy_qty.is_finite() || !sell_qty.is_finite() || buy_qty < 0.0 || sell_qty < 0.0 {
        return f64::NAN;
    }
    let total = buy_qty + sell_qty;
    if total == 0.0 {
        return 0.0;
    }
    (buy_qty - sell_qty).abs() / total
}

/// Per-bucket VPIN from a pre-computed net signed volume and total volume.
///
/// Convenience form when the upstream pipeline carries `net_signed_vol = Σ
/// (qty · side_sign)` and `total_vol = Σ qty` directly (e.g. when the bucket
/// was accumulated via `accumulate(signed_vol, ByKey, Add)` and
/// `accumulate(qty, ByKey, Add)` rather than via masked buy/sell sums).
///
/// Equivalent to `vpin(buy_qty, sell_qty)` since
/// `|buy − sell| / (buy + sell) = |net| / total` algebraically.
#[inline]
pub fn vpin_from_signed(net_signed_vol: f64, total_vol: f64) -> f64 {
    if !net_signed_vol.is_finite() || !total_vol.is_finite() || total_vol < 0.0 {
        return f64::NAN;
    }
    if total_vol == 0.0 {
        return 0.0;
    }
    net_signed_vol.abs() / total_vol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vpin_balanced_flow_is_zero() {
        assert_eq!(vpin(100.0, 100.0), 0.0);
        assert_eq!(vpin_from_signed(0.0, 200.0), 0.0);
    }

    #[test]
    fn vpin_all_buys_is_one() {
        assert_eq!(vpin(100.0, 0.0), 1.0);
        assert_eq!(vpin_from_signed(100.0, 100.0), 1.0);
    }

    #[test]
    fn vpin_all_sells_is_one() {
        assert_eq!(vpin(0.0, 100.0), 1.0);
        assert_eq!(vpin_from_signed(-100.0, 100.0), 1.0);
    }

    #[test]
    fn vpin_empty_bucket_is_zero() {
        // Both qty zero → empty bucket, return 0 (flatline-safe).
        assert_eq!(vpin(0.0, 0.0), 0.0);
        assert_eq!(vpin_from_signed(0.0, 0.0), 0.0);
    }

    #[test]
    fn vpin_quarter_imbalance() {
        // buy=75, sell=25 → |75-25|/(75+25) = 50/100 = 0.5
        assert_eq!(vpin(75.0, 25.0), 0.5);
        // Equivalently: net=50, total=100
        assert_eq!(vpin_from_signed(50.0, 100.0), 0.5);
    }

    #[test]
    fn vpin_two_paths_agree() {
        // For arbitrary buy/sell, vpin(b,s) must equal vpin_from_signed(b-s, b+s)
        let cases = [
            (123.4, 56.7),
            (0.001, 999.0),
            (10.0, 10.0001),
            (1e10, 1.5e10),
        ];
        for &(b, s) in &cases {
            let a = vpin(b, s);
            let c = vpin_from_signed(b - s, b + s);
            assert!((a - c).abs() < 1e-12, "diverged for ({b}, {s}): {a} vs {c}");
        }
    }

    #[test]
    fn vpin_nan_on_invalid_inputs() {
        assert!(vpin(f64::NAN, 100.0).is_nan());
        assert!(vpin(100.0, f64::INFINITY).is_nan());
        assert!(vpin(-1.0, 100.0).is_nan()); // negative qty = data error
        assert!(vpin_from_signed(f64::NAN, 100.0).is_nan());
        assert!(vpin_from_signed(50.0, -100.0).is_nan()); // negative total
    }

    #[test]
    fn vpin_in_unit_interval() {
        // For any non-negative buy/sell, vpin must be in [0, 1].
        for &(b, s) in &[(0.5, 0.5), (1.0, 0.0), (3.14, 2.71), (1e15, 1.0)] {
            let v = vpin(b, s);
            assert!((0.0..=1.0).contains(&v), "out of [0,1] for ({b}, {s}): {v}");
        }
    }
}
