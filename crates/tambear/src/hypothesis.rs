//! Hypothesis testing — t-tests, ANOVA, chi-square, proportions, effect sizes.
//!
//! ## Architecture
//!
//! Every test consumes `MomentStats` or `GroupedMomentStats` from descriptive.rs.
//! No re-scanning. The two-pass scatter already happened. Tests are O(k) CPU
//! where k = number of groups — negligible.
//!
//! The p-value pipeline:
//! 1. **MomentStats** (from scatter) → test statistic (CPU arithmetic)
//! 2. Test statistic → p-value (via special_functions CDFs)
//!
//! ## Tests implemented
//!
//! **t-tests**: one-sample, two-sample (equal variance), paired, Welch's
//! **ANOVA**: one-way (between-groups F-test)
//! **Chi-square**: goodness of fit, independence
//! **Proportions**: one-sample z-test, two-sample z-test
//! **Effect sizes**: Cohen's d (all variants), eta-squared, omega-squared,
//!   Cramér's V, odds ratio, Glass's delta
//! **Multiple comparison**: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)
//!
//! ## .tbs integration
//!
//! ```text
//! t_test(x, mu=0)                # one-sample
//! t_test(x, y)                   # two-sample (Welch's)
//! anova(groups)                  # one-way F-test
//! chi2_test(observed, expected)  # goodness of fit
//! ```

use crate::descriptive::MomentStats;
use crate::special_functions::{
    normal_two_tail_p, t_two_tail_p,
    f_right_tail_p, chi2_right_tail_p,
};

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a hypothesis test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Name of the test (e.g., "One-sample t-test").
    pub test_name: &'static str,
    /// Test statistic value (t, F, χ², z).
    pub statistic: f64,
    /// Degrees of freedom (NaN if not applicable, e.g., z-test).
    pub df: f64,
    /// Two-tailed p-value (or right-tail for F, χ²).
    pub p_value: f64,
    /// Effect size (Cohen's d, eta², etc.). NaN if not computed.
    pub effect_size: f64,
    /// Name of effect size measure.
    pub effect_size_name: &'static str,
}

impl TestResult {
    /// Is the result significant at the given alpha level?
    pub fn significant_at(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Result of an ANOVA test with group-level detail.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F-statistic.
    pub f_statistic: f64,
    /// Between-groups degrees of freedom (k - 1).
    pub df_between: f64,
    /// Within-groups degrees of freedom (N - k).
    pub df_within: f64,
    /// p-value (right-tail of F distribution).
    pub p_value: f64,
    /// Sum of squares between groups.
    pub ss_between: f64,
    /// Sum of squares within groups.
    pub ss_within: f64,
    /// Total sum of squares.
    pub ss_total: f64,
    /// Mean square between.
    pub ms_between: f64,
    /// Mean square within.
    pub ms_within: f64,
    /// Eta-squared: SS_between / SS_total.
    pub eta_squared: f64,
    /// Omega-squared: (SS_between - (k-1)*MS_within) / (SS_total + MS_within).
    pub omega_squared: f64,
}

/// Result of a chi-square test.
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub df: f64,
    /// p-value (right-tail).
    pub p_value: f64,
    /// Cramér's V (for independence tests). NaN for goodness of fit.
    pub cramers_v: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// t-tests
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample t-test: test whether population mean equals `mu`.
///
/// H₀: μ = mu
/// H₁: μ ≠ mu (two-tailed)
///
/// t = (x̄ - μ₀) / (s / √n), df = n - 1
///
/// Consumes MomentStats from scatter. Zero re-scanning.
pub fn one_sample_t(stats: &MomentStats, mu: f64) -> TestResult {
    let n = stats.count;
    if n < 2.0 {
        return TestResult {
            test_name: "One-sample t-test",
            statistic: f64::NAN, df: f64::NAN, p_value: f64::NAN,
            effect_size: f64::NAN, effect_size_name: "Cohen's d",
        };
    }
    let mean = stats.mean();
    let se = stats.sem();
    let t = (mean - mu) / se;
    let df = n - 1.0;
    let p = t_two_tail_p(t, df);
    let d = (mean - mu) / stats.std(1); // Cohen's d

    TestResult {
        test_name: "One-sample t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
    }
}

/// Two-sample t-test (equal variances assumed — Student's t-test).
///
/// H₀: μ₁ = μ₂
/// H₁: μ₁ ≠ μ₂
///
/// Pooled variance: s²_p = (m2_1 + m2_2) / (n₁ + n₂ - 2)
/// t = (x̄₁ - x̄₂) / (s_p √(1/n₁ + 1/n₂)), df = n₁ + n₂ - 2
pub fn two_sample_t(stats1: &MomentStats, stats2: &MomentStats) -> TestResult {
    let n1 = stats1.count;
    let n2 = stats2.count;
    if n1 < 1.0 || n2 < 1.0 || (n1 + n2) < 3.0 {
        return TestResult {
            test_name: "Two-sample t-test",
            statistic: f64::NAN, df: f64::NAN, p_value: f64::NAN,
            effect_size: f64::NAN, effect_size_name: "Cohen's d",
        };
    }
    let mean1 = stats1.mean();
    let mean2 = stats2.mean();
    let df = n1 + n2 - 2.0;
    let pooled_var = (stats1.m2 + stats2.m2) / df;
    let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
    let t = (mean1 - mean2) / se;
    let p = t_two_tail_p(t, df);
    let d = (mean1 - mean2) / pooled_var.sqrt(); // Cohen's d (pooled)

    TestResult {
        test_name: "Two-sample t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
    }
}

/// Welch's t-test (unequal variances — the default in most modern software).
///
/// Does NOT assume equal variances. Welch-Satterthwaite approximation for df.
///
/// s²ᵢ = m2ᵢ / (nᵢ - 1)  (sample variance)
/// t = (x̄₁ - x̄₂) / √(s²₁/n₁ + s²₂/n₂)
/// df = (s²₁/n₁ + s²₂/n₂)² / [(s²₁/n₁)²/(n₁-1) + (s²₂/n₂)²/(n₂-1)]
pub fn welch_t(stats1: &MomentStats, stats2: &MomentStats) -> TestResult {
    let n1 = stats1.count;
    let n2 = stats2.count;
    if n1 < 2.0 || n2 < 2.0 {
        return TestResult {
            test_name: "Welch's t-test",
            statistic: f64::NAN, df: f64::NAN, p_value: f64::NAN,
            effect_size: f64::NAN, effect_size_name: "Cohen's d",
        };
    }
    let var1 = stats1.variance(1);
    let var2 = stats2.variance(1);
    let vn1 = var1 / n1;
    let vn2 = var2 / n2;
    let se = (vn1 + vn2).sqrt();

    let t = (stats1.mean() - stats2.mean()) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (vn1 + vn2).powi(2);
    let denom = vn1 * vn1 / (n1 - 1.0) + vn2 * vn2 / (n2 - 1.0);
    let df = num / denom;

    let p = t_two_tail_p(t, df);
    // Cohen's d using pooled SD (even for Welch, this is the standard effect size)
    let pooled_sd = ((var1 + var2) / 2.0).sqrt();
    let d = (stats1.mean() - stats2.mean()) / pooled_sd;

    TestResult {
        test_name: "Welch's t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
    }
}

/// Paired t-test: test whether mean difference is zero.
///
/// H₀: μ_d = 0  where d = x - y
/// H₁: μ_d ≠ 0
///
/// Caller computes `diff_stats` = MomentStats of (x₁-y₁, x₂-y₂, ..., xₙ-yₙ).
/// Then it's just a one-sample t-test with mu = 0.
pub fn paired_t(diff_stats: &MomentStats) -> TestResult {
    let mut result = one_sample_t(diff_stats, 0.0);
    result.test_name = "Paired t-test";
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// One-way ANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// One-way ANOVA from per-group MomentStats.
///
/// SS_between = Σ nᵢ (x̄ᵢ - x̄)²
/// SS_within = Σ m2ᵢ
/// F = MS_between / MS_within, df₁ = k-1, df₂ = N-k
///
/// This is pure CPU arithmetic on the 7 accumulators per group.
/// No re-scanning. The scatter already happened.
pub fn one_way_anova(groups: &[MomentStats]) -> AnovaResult {
    let k = groups.len() as f64;
    let total_n: f64 = groups.iter().map(|g| g.count).sum();
    let total_sum: f64 = groups.iter().map(|g| g.sum).sum();
    let grand_mean = if total_n > 0.0 { total_sum / total_n } else { f64::NAN };

    // SS_between = Σ nᵢ (x̄ᵢ - x̄)²
    let ss_between: f64 = groups.iter().map(|g| {
        if g.count > 0.0 {
            let dev = g.mean() - grand_mean;
            g.count * dev * dev
        } else {
            0.0
        }
    }).sum();

    // SS_within = Σ m2ᵢ  (sum of centered squared deviations within each group)
    let ss_within: f64 = groups.iter().map(|g| g.m2).sum();

    let ss_total = ss_between + ss_within;

    let df_between = k - 1.0;
    let df_within = total_n - k;

    let ms_between = if df_between > 0.0 { ss_between / df_between } else { f64::NAN };
    let ms_within = if df_within > 0.0 { ss_within / df_within } else { f64::NAN };

    let f_stat = if ms_within > 0.0 { ms_between / ms_within } else { f64::NAN };
    let p = if f_stat.is_nan() { f64::NAN } else { f_right_tail_p(f_stat, df_between, df_within) };

    let eta_sq = if ss_total > 0.0 { ss_between / ss_total } else { f64::NAN };
    let omega_sq = if ss_total + ms_within > 0.0 {
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
    } else {
        f64::NAN
    };

    AnovaResult {
        f_statistic: f_stat,
        df_between, df_within,
        p_value: p,
        ss_between, ss_within, ss_total,
        ms_between, ms_within,
        eta_squared: eta_sq,
        omega_squared: omega_sq,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Chi-square tests
// ═══════════════════════════════════════════════════════════════════════════

/// Chi-square goodness-of-fit test.
///
/// H₀: observed data follow expected distribution
/// χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ, df = k - 1
///
/// `expected`: expected counts (not proportions). Must sum to same total as observed.
pub fn chi2_goodness_of_fit(observed: &[f64], expected: &[f64]) -> ChiSquareResult {
    assert_eq!(observed.len(), expected.len(), "observed and expected must have same length");
    let k = observed.len();
    let chi2: f64 = observed.iter().zip(expected.iter())
        .map(|(&o, &e)| if e > 0.0 { (o - e) * (o - e) / e } else { 0.0 })
        .sum();
    let df = (k as f64) - 1.0;
    let p = chi2_right_tail_p(chi2, df);

    ChiSquareResult { statistic: chi2, df, p_value: p, cramers_v: f64::NAN }
}

/// Chi-square test of independence from a contingency table.
///
/// Input: `table` is a flattened row-major contingency table with `n_rows` rows.
/// χ² = Σ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ where Eᵢⱼ = (row_total × col_total) / N
/// df = (r-1)(c-1)
///
/// The contingency table can come from `information_theory::joint_histogram`.
pub fn chi2_independence(table: &[f64], n_rows: usize) -> ChiSquareResult {
    let n_cols = table.len() / n_rows;
    assert_eq!(table.len(), n_rows * n_cols, "table size must be n_rows × n_cols");

    let n: f64 = table.iter().sum();
    if n == 0.0 {
        return ChiSquareResult {
            statistic: 0.0, df: 0.0, p_value: 1.0, cramers_v: 0.0,
        };
    }

    // Row and column totals
    let mut row_totals = vec![0.0; n_rows];
    let mut col_totals = vec![0.0; n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let v = table[r * n_cols + c];
            row_totals[r] += v;
            col_totals[c] += v;
        }
    }

    // χ² statistic
    let mut chi2 = 0.0;
    for r in 0..n_rows {
        for c in 0..n_cols {
            let expected = row_totals[r] * col_totals[c] / n;
            if expected > 0.0 {
                let diff = table[r * n_cols + c] - expected;
                chi2 += diff * diff / expected;
            }
        }
    }

    let df = ((n_rows - 1) * (n_cols - 1)) as f64;
    let p = if df > 0.0 { chi2_right_tail_p(chi2, df) } else { f64::NAN };

    // Cramér's V = √(χ²/(N × min(r-1, c-1)))
    let min_dim = ((n_rows - 1).min(n_cols - 1)) as f64;
    let cramers_v = if min_dim > 0.0 && n > 0.0 {
        (chi2 / (n * min_dim)).sqrt()
    } else {
        f64::NAN
    };

    ChiSquareResult { statistic: chi2, df, p_value: p, cramers_v }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proportion tests (z-tests)
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample proportion z-test.
///
/// H₀: p = p₀
/// z = (p̂ - p₀) / √(p₀(1-p₀)/n)
pub fn one_proportion_z(successes: f64, n: f64, p0: f64) -> TestResult {
    if n < 1.0 || p0 <= 0.0 || p0 >= 1.0 {
        return TestResult {
            test_name: "One-proportion z-test",
            statistic: f64::NAN, df: f64::NAN, p_value: f64::NAN,
            effect_size: f64::NAN, effect_size_name: "Cohen's h",
        };
    }
    let p_hat = successes / n;
    let se = (p0 * (1.0 - p0) / n).sqrt();
    let z = (p_hat - p0) / se;
    let p = normal_two_tail_p(z);
    // Cohen's h = 2 arcsin(√p̂) - 2 arcsin(√p₀)
    let h = 2.0 * p_hat.sqrt().asin() - 2.0 * p0.sqrt().asin();

    TestResult {
        test_name: "One-proportion z-test",
        statistic: z, df: f64::NAN, p_value: p,
        effect_size: h, effect_size_name: "Cohen's h",
    }
}

/// Two-sample proportion z-test.
///
/// H₀: p₁ = p₂
/// Pooled: p̂ = (x₁ + x₂) / (n₁ + n₂)
/// z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))
pub fn two_proportion_z(successes1: f64, n1: f64, successes2: f64, n2: f64) -> TestResult {
    if n1 < 1.0 || n2 < 1.0 {
        return TestResult {
            test_name: "Two-proportion z-test",
            statistic: f64::NAN, df: f64::NAN, p_value: f64::NAN,
            effect_size: f64::NAN, effect_size_name: "Cohen's h",
        };
    }
    let p1 = successes1 / n1;
    let p2 = successes2 / n2;
    let p_pooled = (successes1 + successes2) / (n1 + n2);
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
    let z = if se > 0.0 { (p1 - p2) / se } else { f64::NAN };
    let p = normal_two_tail_p(z);
    let h = 2.0 * p1.sqrt().asin() - 2.0 * p2.sqrt().asin();

    TestResult {
        test_name: "Two-proportion z-test",
        statistic: z, df: f64::NAN, p_value: p,
        effect_size: h, effect_size_name: "Cohen's h",
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Effect sizes (standalone)
// ═══════════════════════════════════════════════════════════════════════════

/// Cohen's d for two independent samples (pooled SD).
///
/// d = (x̄₁ - x̄₂) / s_pooled
/// s_pooled = √((m2₁ + m2₂) / (n₁ + n₂ - 2))
pub fn cohens_d(stats1: &MomentStats, stats2: &MomentStats) -> f64 {
    let df = stats1.count + stats2.count - 2.0;
    if df <= 0.0 { return f64::NAN; }
    let pooled_sd = ((stats1.m2 + stats2.m2) / df).sqrt();
    if pooled_sd == 0.0 { return f64::NAN; }
    (stats1.mean() - stats2.mean()) / pooled_sd
}

/// Glass's delta: uses only the control group's SD.
///
/// Δ = (x̄₁ - x̄₂) / s₂
/// (stats2 is the control group)
pub fn glass_delta(stats_treatment: &MomentStats, stats_control: &MomentStats) -> f64 {
    let sd_ctrl = stats_control.std(1);
    if sd_ctrl == 0.0 { return f64::NAN; }
    (stats_treatment.mean() - stats_control.mean()) / sd_ctrl
}

/// Hedges' g: bias-corrected Cohen's d for small samples.
///
/// g = d × (1 - 3/(4(n₁+n₂) - 9))
pub fn hedges_g(stats1: &MomentStats, stats2: &MomentStats) -> f64 {
    let d = cohens_d(stats1, stats2);
    let n = stats1.count + stats2.count;
    if n < 4.0 { return d; }
    let correction = 1.0 - 3.0 / (4.0 * n - 9.0);
    d * correction
}

/// Point-biserial correlation from two groups (equivalent to Pearson r
/// between dichotomous IV and continuous DV).
///
/// r_pb = d / √(d² + (n₁+n₂)²/(n₁n₂))
pub fn point_biserial_r(stats1: &MomentStats, stats2: &MomentStats) -> f64 {
    let d = cohens_d(stats1, stats2);
    if d.is_nan() { return f64::NAN; }
    let n1 = stats1.count;
    let n2 = stats2.count;
    let a = (n1 + n2) * (n1 + n2) / (n1 * n2);
    d / (d * d + a).sqrt()
}

/// Odds ratio from a 2×2 contingency table [a, b, c, d].
///
/// OR = (a × d) / (b × c)
pub fn odds_ratio(table_2x2: &[f64; 4]) -> f64 {
    let [a, b, c, d] = *table_2x2;
    if b * c == 0.0 { return f64::INFINITY; }
    (a * d) / (b * c)
}

/// Log odds ratio (more useful for inference — normally distributed).
pub fn log_odds_ratio(table_2x2: &[f64; 4]) -> f64 {
    odds_ratio(table_2x2).ln()
}

/// Standard error of log odds ratio.
///
/// SE(ln OR) = √(1/a + 1/b + 1/c + 1/d)
pub fn log_odds_ratio_se(table_2x2: &[f64; 4]) -> f64 {
    let [a, b, c, d] = *table_2x2;
    (1.0/a + 1.0/b + 1.0/c + 1.0/d).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiple comparison corrections
// ═══════════════════════════════════════════════════════════════════════════

/// Bonferroni correction: multiply each p-value by number of tests.
///
/// Returns adjusted p-values (capped at 1.0).
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len() as f64;
    p_values.iter().map(|&p| (p * m).min(1.0)).collect()
}

/// Holm-Bonferroni step-down correction.
///
/// Stronger than Bonferroni: controls FWER but less conservative.
/// Sort ascending, multiply p[i] by (m - i), enforce monotonicity.
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    // Sort indices by p-value
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0; m];
    let mut running_max: f64 = 0.0;
    for (rank, &idx) in order.iter().enumerate() {
        let adj = p_values[idx] * (m - rank) as f64;
        running_max = running_max.max(adj);
        adjusted[idx] = running_max.min(1.0);
    }
    adjusted
}

/// Benjamini-Hochberg FDR correction.
///
/// Controls false discovery rate at level α. Less conservative than FWER methods.
/// Sort ascending, multiply p[i] by m/rank, enforce monotonicity (step-up).
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0; m];
    let mut running_min: f64 = 1.0;
    // Step-up: process from largest to smallest
    for (rank_rev, &idx) in order.iter().enumerate().rev() {
        let rank = rank_rev + 1; // 1-based
        let adj = p_values[idx] * m as f64 / rank as f64;
        running_min = running_min.min(adj);
        adjusted[idx] = running_min.min(1.0);
    }
    adjusted
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: HypothesisEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Engine that wraps a ComputeEngine for hypothesis tests on raw data.
///
/// For tests that only need MomentStats, use the free functions directly.
/// The engine provides convenience methods that compute MomentStats from
/// raw data via scatter, then run the test.
pub struct HypothesisEngine {
    desc: crate::descriptive::DescriptiveEngine,
}

impl HypothesisEngine {
    /// Create a new HypothesisEngine using auto-detected backend.
    pub fn new() -> Self {
        Self { desc: crate::descriptive::DescriptiveEngine::new() }
    }

    /// One-sample t-test from raw data.
    pub fn one_sample_t_test(&mut self, data: &[f64], mu: f64) -> TestResult {
        let stats = crate::descriptive::moments_ungrouped(data);
        one_sample_t(&stats, mu)
    }

    /// Welch's t-test from two raw data vectors.
    pub fn welch_t_test(&mut self, x: &[f64], y: &[f64]) -> TestResult {
        let s1 = crate::descriptive::moments_ungrouped(x);
        let s2 = crate::descriptive::moments_ungrouped(y);
        welch_t(&s1, &s2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::{MomentStats, moments_ungrouped};

    const TOL: f64 = 1e-3;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        if a.is_infinite() && b.is_infinite() { return a.signum() == b.signum(); }
        (a - b).abs() < tol
    }

    // ── One-sample t ─────────────────────────────────────────────────────

    #[test]
    fn one_sample_t_basic() {
        // Data with mean ≈ 5.0, testing against mu=5 should give non-significant
        let data = [4.5, 5.1, 5.3, 4.8, 5.0, 4.9, 5.2, 5.1];
        let stats = moments_ungrouped(&data);
        let r = one_sample_t(&stats, 5.0);
        assert!(r.p_value > 0.05, "Should not reject H0 that mean=5.0, p={}", r.p_value);
    }

    #[test]
    fn one_sample_t_significant() {
        // Data clearly above 0
        let data = [3.0, 4.0, 5.0, 6.0, 7.0];
        let stats = moments_ungrouped(&data);
        let r = one_sample_t(&stats, 0.0);
        assert!(r.p_value < 0.01, "Should strongly reject H0 that mean=0, p={}", r.p_value);
        assert!(r.statistic > 0.0, "t should be positive");
        assert!(r.df == 4.0, "df should be n-1=4");
    }

    #[test]
    fn one_sample_t_edge_empty() {
        let stats = MomentStats::empty();
        let r = one_sample_t(&stats, 0.0);
        assert!(r.p_value.is_nan());
    }

    #[test]
    fn one_sample_t_edge_single() {
        let stats = moments_ungrouped(&[42.0]);
        let r = one_sample_t(&stats, 0.0);
        assert!(r.p_value.is_nan()); // df=0, can't test
    }

    // ── Two-sample t (equal variance) ────────────────────────────────────

    #[test]
    fn two_sample_t_equal_groups() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.2, 2.1, 3.1, 3.9, 5.1];
        let s1 = moments_ungrouped(&x);
        let s2 = moments_ungrouped(&y);
        let r = two_sample_t(&s1, &s2);
        // Very similar distributions → non-significant
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn two_sample_t_different_groups() {
        let x = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s1 = moments_ungrouped(&x);
        let s2 = moments_ungrouped(&y);
        let r = two_sample_t(&s1, &s2);
        assert!(r.p_value < 0.001, "Should be highly significant, p={}", r.p_value);
        assert!(r.statistic > 0.0, "t > 0 since group 1 > group 2");
    }

    // ── Welch's t ────────────────────────────────────────────────────────

    #[test]
    fn welch_t_basic() {
        let x = [10.0, 12.0, 14.0, 11.0, 13.0];
        let y = [20.0, 22.0, 24.0, 21.0, 23.0];
        let s1 = moments_ungrouped(&x);
        let s2 = moments_ungrouped(&y);
        let r = welch_t(&s1, &s2);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
        assert!(r.test_name == "Welch's t-test");
    }

    #[test]
    fn welch_t_unequal_variance() {
        // Group 1: low variance, Group 2: high variance
        let x = [5.0, 5.1, 4.9, 5.0, 5.0];
        let y = [4.0, 6.0, 3.0, 7.0, 5.0];
        let s1 = moments_ungrouped(&x);
        let s2 = moments_ungrouped(&y);
        let r = welch_t(&s1, &s2);
        // Same mean (5.0), different variance → non-significant
        assert!(r.p_value > 0.5, "p={}", r.p_value);
        // Welch's df should be less than pooled df (n1+n2-2=8) due to unequal variance
        assert!(r.df < 8.0, "Welch df={} should be < 8", r.df);
    }

    // ── Paired t ─────────────────────────────────────────────────────────

    #[test]
    fn paired_t_basic() {
        let before = [10.0, 12.0, 14.0, 11.0, 13.0];
        let after = [12.0, 14.0, 16.0, 13.0, 15.0];
        let diffs: Vec<f64> = before.iter().zip(after.iter()).map(|(&b, &a)| b - a).collect();
        let diff_stats = moments_ungrouped(&diffs);
        let r = paired_t(&diff_stats);
        assert!(r.p_value < 0.001, "Treatment effect should be significant, p={}", r.p_value);
        assert!(r.test_name == "Paired t-test");
        assert!(r.statistic < 0.0, "t < 0 since before < after");
    }

    // ── One-way ANOVA ────────────────────────────────────────────────────

    #[test]
    fn anova_identical_groups() {
        let g1 = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let g2 = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let g3 = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let r = one_way_anova(&[g1, g2, g3]);
        assert!(approx(r.f_statistic, 0.0, 1e-10), "F should be 0 for identical groups");
        assert!(r.p_value > 0.99, "p should be ~1.0");
        assert!(approx(r.eta_squared, 0.0, 1e-10), "η² should be 0");
    }

    #[test]
    fn anova_different_groups() {
        let g1 = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let g2 = moments_ungrouped(&[10.0, 11.0, 12.0]);
        let g3 = moments_ungrouped(&[20.0, 21.0, 22.0]);
        let r = one_way_anova(&[g1, g2, g3]);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
        assert!(r.f_statistic > 50.0, "F={}", r.f_statistic);
        assert!(r.eta_squared > 0.95, "η²={}", r.eta_squared);
        // Verify SS decomposition: SS_total = SS_between + SS_within
        assert!(approx(r.ss_total, r.ss_between + r.ss_within, 1e-10));
    }

    #[test]
    fn anova_two_groups_matches_t_test() {
        // ANOVA with k=2 should give F = t², same p-value as two-sample t-test
        let g1 = moments_ungrouped(&[2.0, 3.0, 4.0, 5.0]);
        let g2 = moments_ungrouped(&[6.0, 7.0, 8.0, 9.0]);
        let anova_r = one_way_anova(&[g1, g2]);
        let t_r = two_sample_t(&g1, &g2);
        assert!(approx(anova_r.f_statistic, t_r.statistic * t_r.statistic, 0.01),
            "F={} should ≈ t²={}", anova_r.f_statistic, t_r.statistic.powi(2));
        assert!(approx(anova_r.p_value, t_r.p_value, 0.01),
            "ANOVA p={} should ≈ t-test p={}", anova_r.p_value, t_r.p_value);
    }

    #[test]
    fn anova_omega_squared() {
        // Omega² should be less than eta² (less biased)
        let g1 = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let g2 = moments_ungrouped(&[4.0, 5.0, 6.0]);
        let r = one_way_anova(&[g1, g2]);
        assert!(r.omega_squared < r.eta_squared, "ω²={} should be < η²={}", r.omega_squared, r.eta_squared);
        assert!(r.omega_squared > 0.0, "ω² should be positive for truly different groups");
    }

    // ── Chi-square goodness of fit ───────────────────────────────────────

    #[test]
    fn chi2_gof_uniform() {
        // Observed matches expected perfectly
        let obs = [25.0, 25.0, 25.0, 25.0];
        let exp = [25.0, 25.0, 25.0, 25.0];
        let r = chi2_goodness_of_fit(&obs, &exp);
        assert!(approx(r.statistic, 0.0, 1e-10), "χ² should be 0");
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn chi2_gof_significant() {
        // Heavy departure from expected
        let obs = [90.0, 5.0, 3.0, 2.0];
        let exp = [25.0, 25.0, 25.0, 25.0];
        let r = chi2_goodness_of_fit(&obs, &exp);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
        assert_eq!(r.df, 3.0);
    }

    // ── Chi-square independence ──────────────────────────────────────────

    #[test]
    fn chi2_independence_perfect() {
        // Perfect independence: row and column proportions are constant
        let table = [10.0, 20.0, 20.0, 40.0]; // 2×2, proportional
        let r = chi2_independence(&table, 2);
        assert!(approx(r.statistic, 0.0, 1e-10), "χ²={}", r.statistic);
        assert!(r.p_value > 0.99);
        assert!(approx(r.cramers_v, 0.0, 1e-6));
    }

    #[test]
    fn chi2_independence_associated() {
        // Strong association
        let table = [50.0, 5.0, 5.0, 50.0]; // 2×2
        let r = chi2_independence(&table, 2);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
        assert!(r.cramers_v > 0.7, "V={}", r.cramers_v);
    }

    #[test]
    fn chi2_independence_3x3() {
        // 3×3 table
        let table = [
            30.0, 5.0, 5.0,
            5.0, 30.0, 5.0,
            5.0, 5.0, 30.0,
        ];
        let r = chi2_independence(&table, 3);
        assert!(r.p_value < 0.001);
        assert_eq!(r.df, 4.0); // (3-1)(3-1)
        assert!(r.cramers_v > 0.5);
    }

    // ── Proportion tests ─────────────────────────────────────────────────

    #[test]
    fn one_proportion_fair_coin() {
        // 52 heads out of 100 — should NOT reject H0: p = 0.5
        let r = one_proportion_z(52.0, 100.0, 0.5);
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn one_proportion_biased_coin() {
        // 80 heads out of 100 — should reject H0: p = 0.5
        let r = one_proportion_z(80.0, 100.0, 0.5);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
    }

    #[test]
    fn two_proportion_equal() {
        // Similar proportions
        let r = two_proportion_z(50.0, 100.0, 48.0, 100.0);
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn two_proportion_different() {
        // Very different proportions
        let r = two_proportion_z(80.0, 100.0, 30.0, 100.0);
        assert!(r.p_value < 0.001, "p={}", r.p_value);
    }

    // ── Effect sizes ─────────────────────────────────────────────────────

    #[test]
    fn cohens_d_known() {
        // Exact computation: groups differ by 1 SD
        let s1 = moments_ungrouped(&[0.0, 1.0, 2.0]);
        let s2 = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let d = cohens_d(&s1, &s2);
        assert!(approx(d, -1.0, 0.01), "d={} should be ≈ -1.0", d);
    }

    #[test]
    fn hedges_g_correction() {
        let s1 = moments_ungrouped(&[0.0, 1.0, 2.0]);
        let s2 = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let d = cohens_d(&s1, &s2);
        let g = hedges_g(&s1, &s2);
        // Hedges' g is slightly smaller than Cohen's d (correction for small n)
        assert!(g.abs() < d.abs(), "g={} should be smaller than d={}", g, d);
    }

    #[test]
    fn point_biserial_range() {
        let s1 = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let s2 = moments_ungrouped(&[4.0, 5.0, 6.0]);
        let r = point_biserial_r(&s1, &s2);
        assert!(r.abs() <= 1.0, "r_pb must be in [-1,1], got {}", r);
    }

    #[test]
    fn odds_ratio_basic() {
        // Classic 2×2: [a=10, b=5, c=5, d=10]
        let table = [10.0, 5.0, 5.0, 10.0];
        let or = odds_ratio(&table);
        assert!(approx(or, 4.0, 1e-10), "OR={}", or);
    }

    #[test]
    fn log_odds_ratio_se_basic() {
        let table = [10.0, 5.0, 5.0, 10.0];
        let se = log_odds_ratio_se(&table);
        // SE = √(1/10 + 1/5 + 1/5 + 1/10) = √0.6 ≈ 0.7746
        assert!(approx(se, 0.6_f64.sqrt(), 1e-10));
    }

    // ── Multiple comparison ──────────────────────────────────────────────

    #[test]
    fn bonferroni_basic() {
        let ps = [0.01, 0.04, 0.03, 0.5];
        let adj = bonferroni(&ps);
        assert!(approx(adj[0], 0.04, 1e-10));
        assert!(approx(adj[1], 0.16, 1e-10));
        assert!(approx(adj[2], 0.12, 1e-10));
        assert!(approx(adj[3], 1.0, 1e-10)); // capped at 1
    }

    #[test]
    fn holm_basic() {
        let ps = [0.01, 0.04, 0.03, 0.5];
        let adj = holm(&ps);
        // Sorted: 0.01, 0.03, 0.04, 0.5
        // Holm: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08→max(0.09)=0.09, 0.5*1=0.5
        assert!(approx(adj[0], 0.04, 1e-10));
        assert!(approx(adj[2], 0.09, 1e-10)); // p=0.03 is rank 2
        assert!(adj[1] >= adj[2]); // monotonicity: later ranks ≥ earlier
    }

    #[test]
    fn bh_fdr_basic() {
        let ps = [0.01, 0.04, 0.03, 0.5];
        let adj = benjamini_hochberg(&ps);
        // All adjusted p-values should be ≥ raw p-values
        for (raw, adjusted) in ps.iter().zip(adj.iter()) {
            assert!(*adjusted >= *raw - 1e-10, "adj {} < raw {}", adjusted, raw);
        }
        // BH is less conservative than Bonferroni
        let bonf = bonferroni(&ps);
        for (bh_p, bonf_p) in adj.iter().zip(bonf.iter()) {
            assert!(*bh_p <= *bonf_p + 1e-10, "BH {} > Bonf {}", bh_p, bonf_p);
        }
    }

    // ── Integration ──────────────────────────────────────────────────────

    #[test]
    fn test_result_significant_at() {
        let r = TestResult {
            test_name: "test", statistic: 2.0, df: 10.0, p_value: 0.03,
            effect_size: 0.5, effect_size_name: "d",
        };
        assert!(r.significant_at(0.05));
        assert!(!r.significant_at(0.01));
    }
}
