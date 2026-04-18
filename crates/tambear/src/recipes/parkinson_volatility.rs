//! Parkinson (1980) range-based volatility estimator — per-bucket form.
//!
//! Locked vocabulary: this is a Tier 4 recipe. Pure composition; no inline math
//! beyond literal Parkinson formula constants. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Parkinson's range-based variance estimator on log scale:
//!
//! ```text
//! σ²_P  =  (ln(H/L))²  /  (4 · ln 2)
//! σ_P   =  sqrt(σ²_P)
//! ```
//!
//! For pre-computed `hl_log_range_sq = (ln(H/L))²`:
//!
//! ```text
//! σ_P  =  sqrt(hl_log_range_sq / (4 · ln 2))
//! ```
//!
//! ~5x more efficient than close-to-close volatility under the no-drift
//! assumption.
//!
//! # Reference
//!
//! Parkinson, M. (1980). The extreme value method for estimating the variance
//! of the rate of return. *Journal of Business* 53(1): 61–65.
//!
//! # Composition
//!
//! - One sqrt over a constant-divided value. No accumulation, no atom call.
//!   This is a leaf recipe — its sub-tree bottoms out immediately at primitive
//!   arithmetic.
//!
//! # Default parameters
//!
//! None. The estimator has no tunable parameters; the constant `4 · ln 2`
//! is the closed-form Parkinson coefficient.

/// Parkinson volatility from a pre-computed `hl_log_range_sq = (ln(H/L))²`.
///
/// For SIP buckets this is typically the C12 column from the column-graph
/// (`hl_log_range2`). Computing `(ln(H/L))²` from raw H/L is done by the
/// `parkinson_variance` recipe in `volatility.rs`; this recipe consumes the
/// already-computed squared-log-range and produces the volatility (not
/// variance) in the same units as a returns standard deviation.
///
/// Returns NaN if `hl_log_range_sq` is non-finite or negative (which would
/// indicate H < L in the upstream calculation — a data error).
#[inline]
pub fn parkinson_volatility(hl_log_range_sq: f64) -> f64 {
    if !hl_log_range_sq.is_finite() || hl_log_range_sq < 0.0 {
        return f64::NAN;
    }
    let four_ln2 = 4.0 * std::f64::consts::LN_2;
    (hl_log_range_sq / four_ln2).sqrt()
}

/// Parkinson volatility from raw high and low prices.
///
/// Convenience wrapper for the case where the bucket's H/L is on hand
/// instead of the precomputed squared log range. Equivalent to:
/// `parkinson_volatility((ln(H/L))²)`.
///
/// Returns NaN if either price is non-positive or H < L.
#[inline]
pub fn parkinson_volatility_from_hl(high: f64, low: f64) -> f64 {
    if high <= 0.0 || low <= 0.0 || high < low || !high.is_finite() || !low.is_finite() {
        return f64::NAN;
    }
    let lnhl = (high / low).ln();
    parkinson_volatility(lnhl * lnhl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parkinson_zero_range_is_zero() {
        // H == L → no range → zero volatility
        assert_eq!(parkinson_volatility(0.0), 0.0);
        assert_eq!(parkinson_volatility_from_hl(100.0, 100.0), 0.0);
    }

    #[test]
    fn parkinson_known_value() {
        // H=110, L=90 → ln(110/90) = ln(11/9) ≈ 0.20067
        // squared ≈ 0.04027
        // / (4·ln2) ≈ / 2.7726 ≈ 0.014523
        // sqrt ≈ 0.12051
        let v = parkinson_volatility_from_hl(110.0, 90.0);
        assert!((v - 0.12051).abs() < 1e-4, "got {v}");
    }

    #[test]
    fn parkinson_matches_two_paths() {
        // Squared-range path and raw-HL path must agree.
        let h: f64 = 105.0;
        let l: f64 = 95.0;
        let lnhl = (h / l).ln();
        let direct = parkinson_volatility(lnhl * lnhl);
        let from_hl = parkinson_volatility_from_hl(h, l);
        assert_eq!(direct.to_bits(), from_hl.to_bits());
    }

    #[test]
    fn parkinson_nan_on_invalid_inputs() {
        assert!(parkinson_volatility(f64::NAN).is_nan());
        assert!(parkinson_volatility(f64::INFINITY).is_nan());
        assert!(parkinson_volatility(-1.0).is_nan());
        assert!(parkinson_volatility_from_hl(0.0, 0.0).is_nan());
        assert!(parkinson_volatility_from_hl(50.0, 100.0).is_nan()); // H < L
        assert!(parkinson_volatility_from_hl(-1.0, 50.0).is_nan());
    }

    #[test]
    fn parkinson_scales_with_log_range() {
        // Doubling the squared log range should multiply volatility by sqrt(2).
        let v1 = parkinson_volatility(0.04);
        let v2 = parkinson_volatility(0.08);
        let ratio = v2 / v1;
        assert!((ratio - std::f64::consts::SQRT_2).abs() < 1e-12);
    }
}
