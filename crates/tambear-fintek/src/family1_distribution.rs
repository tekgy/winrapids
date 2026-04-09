//! Family 1 — Distribution / Moments / Normality.
//!
//! Covers fintek leaves: `distribution`, `normality`, `shannon_entropy`,
//! `spectral_entropy`, `tail_field`.
//!
//! All are DIRECT or COMPOSE mappings over tambear primitives.

use tambear::descriptive::{moments_ungrouped, quantile, QuantileMethod};
use tambear::nonparametric::{shapiro_wilk, NonparametricResult};
use tambear::hypothesis::jarque_bera;
use tambear::volatility::{realized_variance, bipower_variation, jump_test_bns};

/// Output of the `distribution` leaf.
///
/// Fintek columns (K02P01C01):
/// - `do01` = mean
/// - `do02` = std
/// - `do03` = skewness
/// - `do04` = excess kurtosis
/// - `do05` = median (q50)
/// - `do06` = q25
/// - `do07` = q75
/// - `do08` = realized variance
/// - `do09` = bipower variation
/// - `do10` = BNS jump statistic
#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub mean: f64,
    pub std: f64,
    pub skew: f64,
    pub kurt_excess: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub realized_var: f64,
    pub bipower_var: f64,
    pub bns_jump: f64,
}

impl DistributionResult {
    /// NaN result for empty or invalid bins.
    pub fn nan() -> Self {
        Self {
            mean: f64::NAN, std: f64::NAN, skew: f64::NAN, kurt_excess: f64::NAN,
            median: f64::NAN, q25: f64::NAN, q75: f64::NAN,
            realized_var: f64::NAN, bipower_var: f64::NAN, bns_jump: f64::NAN,
        }
    }
}

/// Compute distribution features for a single bin.
///
/// Equivalent to `distribution.rs` in fintek.
///
/// `returns`: pre-computed log returns within the bin (not raw prices).
/// If you have raw prices, convert first via `family2_transforms::log_returns`.
pub fn distribution(returns: &[f64]) -> DistributionResult {
    if returns.len() < 2 {
        return DistributionResult::nan();
    }

    let stats = moments_ungrouped(returns);
    let mean = stats.mean();
    let std = stats.std(1); // unbiased
    let skew = stats.skewness(false);
    let kurt_excess = stats.kurtosis(true, false); // excess kurtosis

    // Sorted copy for quantiles (linear interpolation method)
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = quantile(&sorted, 0.5, QuantileMethod::Linear);
    let q25 = quantile(&sorted, 0.25, QuantileMethod::Linear);
    let q75 = quantile(&sorted, 0.75, QuantileMethod::Linear);

    let realized_var = realized_variance(returns);
    let bipower_var = bipower_variation(returns);
    let bns_jump = jump_test_bns(returns);

    DistributionResult {
        mean, std, skew, kurt_excess, median, q25, q75,
        realized_var, bipower_var, bns_jump,
    }
}

/// Output of the `normality` leaf.
///
/// - `do01` = Jarque-Bera statistic
/// - `do02` = Jarque-Bera p-value
/// - `do03` = Shapiro-Wilk W
/// - `do04` = Shapiro-Wilk p-value
#[derive(Debug, Clone)]
pub struct NormalityResult {
    pub jb_statistic: f64,
    pub jb_p_value: f64,
    pub sw_statistic: f64,
    pub sw_p_value: f64,
}

impl NormalityResult {
    pub fn nan() -> Self {
        Self {
            jb_statistic: f64::NAN, jb_p_value: f64::NAN,
            sw_statistic: f64::NAN, sw_p_value: f64::NAN,
        }
    }
}

/// Normality tests on bin-level returns.
///
/// Shapiro-Wilk is more sensitive for small n (< 5000); Jarque-Bera is
/// asymptotic and works for any n ≥ 3.
pub fn normality(returns: &[f64]) -> NormalityResult {
    if returns.len() < 3 {
        return NormalityResult::nan();
    }
    let stats = moments_ungrouped(returns);
    let jb = jarque_bera(&stats);
    let sw: NonparametricResult = shapiro_wilk(returns);
    NormalityResult {
        jb_statistic: jb.statistic,
        jb_p_value: jb.p_value,
        sw_statistic: sw.statistic,
        sw_p_value: sw.p_value,
    }
}

/// Shannon entropy of histogram-binned returns.
///
/// Equivalent to fintek's `shannon_entropy.rs`. Uses equal-width binning
/// (via Freedman-Diaconis rule) by default.
///
/// `returns`: bin-level returns.
/// Returns Shannon entropy in bits (log base 2).
pub fn shannon_entropy_of_returns(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    // Use Freedman-Diaconis for the bin count
    let k = tambear::nonparametric::freedman_diaconis_bins(returns).max(2);
    let hist = tambear::nonparametric::histogram_auto(
        returns,
        tambear::nonparametric::BinRule::Fixed(k),
    );

    // Normalize counts → probabilities → Shannon entropy
    let total: u64 = hist.counts.iter().sum();
    if total == 0 { return f64::NAN; }
    let probs: Vec<f64> = hist.counts.iter().map(|&c| c as f64 / total as f64).collect();
    // Convert to log2 (Shannon's entropy is defined in bits)
    let mut h = 0.0;
    for p in probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

/// Tail concentration metric: quintile binning + chi-square against uniform.
///
/// Equivalent to fintek's `tail_field.rs`. Returns the chi² statistic.
pub fn tail_field_chi2(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 10 {
        return f64::NAN;
    }
    // Split into quintiles
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut bounds = [0.0; 4];
    for (i, &q) in [0.2, 0.4, 0.6, 0.8].iter().enumerate() {
        bounds[i] = quantile(&sorted, q, QuantileMethod::Linear);
    }
    // Count observations per quintile
    let mut counts = [0.0_f64; 5];
    for &r in returns {
        let bin = if r < bounds[0] { 0 }
                  else if r < bounds[1] { 1 }
                  else if r < bounds[2] { 2 }
                  else if r < bounds[3] { 3 }
                  else { 4 };
        counts[bin] += 1.0;
    }
    // Expected under uniform quintile distribution
    let expected = [n as f64 / 5.0; 5];
    let result = tambear::hypothesis::chi2_goodness_of_fit(&counts, &expected);
    result.statistic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_basic() {
        // Standard small return series
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let r = distribution(&returns);
        assert!(r.mean.is_finite());
        assert!(r.std > 0.0);
        assert!(r.skew.is_finite());
        assert!(r.median.is_finite());
        assert!(r.q25 < r.median && r.median < r.q75,
            "Q25 < median < Q75: {} {} {}", r.q25, r.median, r.q75);
        assert!(r.realized_var >= 0.0);
        assert!(r.bipower_var >= 0.0);
    }

    #[test]
    fn distribution_degenerate() {
        let r = distribution(&[]);
        assert!(r.mean.is_nan());
        let r = distribution(&[1.0]);
        assert!(r.mean.is_nan());
    }

    #[test]
    fn normality_on_normal_data() {
        // Gaussian noise via Xoshiro
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = normality(&data);
        assert!(r.jb_p_value > 0.05, "JB should not reject normal: p={}", r.jb_p_value);
    }

    #[test]
    fn normality_on_uniform() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
        let r = normality(&data);
        assert!(r.jb_p_value < 0.05, "JB should reject uniform: p={}", r.jb_p_value);
    }

    #[test]
    fn shannon_entropy_constant() {
        // Single bin → entropy = 0
        let h = shannon_entropy_of_returns(&[0.5; 50]);
        assert!((h - 0.0).abs() < 1e-10 || h.is_nan());
    }

    #[test]
    fn shannon_entropy_uniform() {
        // Uniform data → entropy near log₂(k)
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let h = shannon_entropy_of_returns(&data);
        assert!(h > 3.0, "Uniform should have moderate-high entropy, got {}", h);
    }

    #[test]
    fn shannon_entropy_too_few() {
        assert!(shannon_entropy_of_returns(&[1.0]).is_nan());
        assert!(shannon_entropy_of_returns(&[]).is_nan());
    }

    #[test]
    fn tail_field_uniform() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let chi2 = tail_field_chi2(&data);
        // Uniform data → chi² near 0 (counts match expected)
        assert!(chi2 < 5.0, "Uniform chi² should be small, got {}", chi2);
    }

    #[test]
    fn tail_field_too_few() {
        assert!(tail_field_chi2(&[1.0, 2.0, 3.0]).is_nan());
    }
}
