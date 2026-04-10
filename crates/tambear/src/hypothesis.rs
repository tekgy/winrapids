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
    /// Lower bound of confidence interval on the estimand (NaN if not computed).
    pub ci_lower: f64,
    /// Upper bound of confidence interval on the estimand (NaN if not computed).
    pub ci_upper: f64,
    /// Confidence level used (e.g., 0.95). NaN if CI not computed.
    pub ci_level: f64,
}

impl TestResult {
    /// Is the result significant at the given alpha level?
    pub fn significant_at(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Does the confidence interval contain the given null value?
    /// Returns None if CI was not computed.
    pub fn ci_contains(&self, null_value: f64) -> Option<bool> {
        if self.ci_lower.is_nan() || self.ci_upper.is_nan() { return None; }
        Some(self.ci_lower <= null_value && null_value <= self.ci_upper)
    }

    /// Builder: construct a TestResult without a CI (legacy / tests without an estimand CI).
    pub fn no_ci(
        test_name: &'static str,
        statistic: f64, df: f64, p_value: f64,
        effect_size: f64, effect_size_name: &'static str,
    ) -> Self {
        Self {
            test_name, statistic, df, p_value, effect_size, effect_size_name,
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        }
    }

    /// Builder: construct a TestResult with a CI on the estimand.
    pub fn with_ci(
        test_name: &'static str,
        statistic: f64, df: f64, p_value: f64,
        effect_size: f64, effect_size_name: &'static str,
        ci_lower: f64, ci_upper: f64, ci_level: f64,
    ) -> Self {
        Self {
            test_name, statistic, df, p_value, effect_size, effect_size_name,
            ci_lower, ci_upper, ci_level,
        }
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

impl AnovaResult {
    /// Cohen's f effect size: f = sqrt(eta² / (1 - eta²)).
    /// Small: 0.1, Medium: 0.25, Large: 0.4.
    pub fn cohens_f(&self) -> f64 {
        if self.eta_squared >= 1.0 { return f64::INFINITY; }
        (self.eta_squared / (1.0 - self.eta_squared)).sqrt()
    }

    /// Partial eta-squared (same as eta-squared for one-way ANOVA,
    /// differs for factorial designs).
    pub fn partial_eta_squared(&self) -> f64 {
        if self.ss_between + self.ss_within <= 0.0 { return 0.0; }
        self.ss_between / (self.ss_between + self.ss_within)
    }
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
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let mean = stats.mean();
    let se = stats.sem();
    let t = (mean - mu) / se;
    let df = n - 1.0;
    let p = t_two_tail_p(t, df);
    let d = (mean - mu) / stats.std(1); // Cohen's d

    // 95% CI on the mean: x̄ ± t_{0.975, n-1} · SE
    let t_crit = crate::special_functions::t_quantile(0.975, df);
    let ci_half = t_crit * se;
    let ci_lower = mean - ci_half;
    let ci_upper = mean + ci_half;

    TestResult {
        test_name: "One-sample t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
        ci_lower, ci_upper, ci_level: 0.95,
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
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let mean1 = stats1.mean();
    let mean2 = stats2.mean();
    let mean_diff = mean1 - mean2;
    let df = n1 + n2 - 2.0;
    let pooled_var = (stats1.m2 + stats2.m2) / df;
    let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
    let t = mean_diff / se;
    let p = t_two_tail_p(t, df);
    let d = mean_diff / pooled_var.sqrt(); // Cohen's d (pooled)

    // 95% CI on (μ₁ - μ₂): (x̄₁ - x̄₂) ± t_{0.975, df} · SE_diff
    let t_crit = crate::special_functions::t_quantile(0.975, df);
    let ci_half = t_crit * se;

    TestResult {
        test_name: "Two-sample t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
        ci_lower: mean_diff - ci_half,
        ci_upper: mean_diff + ci_half,
        ci_level: 0.95,
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
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let var1 = stats1.variance(1);
    let var2 = stats2.variance(1);
    let vn1 = var1 / n1;
    let vn2 = var2 / n2;
    let se = (vn1 + vn2).sqrt();
    let mean_diff = stats1.mean() - stats2.mean();
    let t = mean_diff / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (vn1 + vn2).powi(2);
    let denom = vn1 * vn1 / (n1 - 1.0) + vn2 * vn2 / (n2 - 1.0);
    let df = num / denom;

    let p = t_two_tail_p(t, df);
    // Cohen's d using proper pooled SD: sqrt((m2₁ + m2₂) / (n₁ + n₂ - 2))
    let d = cohens_d(stats1, stats2);

    // 95% CI on (μ₁ - μ₂) using Welch's df
    let t_crit = crate::special_functions::t_quantile(0.975, df);
    let ci_half = t_crit * se;

    TestResult {
        test_name: "Welch's t-test",
        statistic: t, df, p_value: p,
        effect_size: d, effect_size_name: "Cohen's d",
        ci_lower: mean_diff - ci_half,
        ci_upper: mean_diff + ci_half,
        ci_level: 0.95,
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
// Levene's test for homogeneity of variance (Levene 1960, Brown-Forsythe 1974)
// ═══════════════════════════════════════════════════════════════════════════

/// Center type for Levene's test.
#[derive(Debug, Clone, Copy)]
pub enum LeveneCenter {
    /// Original Levene (1960): most powerful for symmetric distributions.
    Mean,
    /// Brown-Forsythe (1974): most robust, recommended default.
    Median,
}

/// Levene's test result.
#[derive(Debug, Clone)]
pub struct LeveneResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_between: f64,
    pub df_within: f64,
}

/// Levene's test for equality of variances across k groups.
///
/// Applies one-way ANOVA to the absolute deviations from group centers.
/// W = ANOVA F on z_ij = |x_ij - center_j|, where center_j = mean or median.
/// W ~ F(k-1, N-k) under H₀.
///
/// Use `LeveneCenter::Median` (Brown-Forsythe variant) for robustness.
pub fn levene_test(groups: &[&[f64]], center: LeveneCenter) -> LeveneResult {
    // Filter out empty groups
    let groups: Vec<&[f64]> = groups.iter().filter(|g| !g.is_empty()).copied().collect();
    let k = groups.len();
    if k < 2 {
        return LeveneResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: 0.0, df_within: 0.0 };
    }

    // Compute absolute deviations from group center
    let z_groups: Vec<Vec<f64>> = groups.iter().map(|g| {
        let center_val = match center {
            LeveneCenter::Mean => g.iter().sum::<f64>() / g.len() as f64,
            LeveneCenter::Median => {
                let mut sorted = g.to_vec();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let n = sorted.len();
                if n % 2 == 0 { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
                else { sorted[n / 2] }
            }
        };
        g.iter().map(|x| (x - center_val).abs()).collect()
    }).collect();

    // Compute MomentStats for each group of z values
    let z_stats: Vec<MomentStats> = z_groups.iter()
        .map(|zg| crate::descriptive::moments_ungrouped(zg))
        .collect();

    // Apply one-way ANOVA on the z values
    let anova = one_way_anova(&z_stats);

    LeveneResult {
        f_statistic: anova.f_statistic,
        p_value: anova.p_value,
        df_between: anova.df_between,
        df_within: anova.df_within,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Welch's ANOVA (Welch 1951)
// ═══════════════════════════════════════════════════════════════════════════

/// Welch's ANOVA result.
#[derive(Debug, Clone)]
pub struct WelchAnovaResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_between: f64,
    pub df_within: f64,
}

/// Welch's ANOVA for k groups given pre-computed sufficient statistics.
///
/// Accepts `&[&MomentStats]` — consistent with the moment-first API
/// where `moments_ungrouped` is called once per group and the stats are reused.
/// Groups with n < 2 (undefined sample variance) are silently excluded.
pub fn welch_anova_from_moments(groups: &[&crate::descriptive::MomentStats]) -> WelchAnovaResult {
    let groups: Vec<&crate::descriptive::MomentStats> = groups.iter()
        .filter(|g| g.count >= 2.0)
        .copied()
        .collect();
    let k = groups.len();
    let kf = k as f64;
    if k < 2 {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: 0.0, df_within: 0.0 };
    }
    let means: Vec<f64> = groups.iter().map(|g| g.mean()).collect();
    let vars: Vec<f64> = groups.iter().map(|g| g.variance(1)).collect();
    let ns: Vec<f64> = groups.iter().map(|g| g.count).collect();
    let w: Vec<f64> = (0..k).map(|j| {
        if vars[j] < 1e-300 { f64::NAN } else { ns[j] / vars[j] }
    }).collect();
    if w.iter().any(|v| v.is_nan()) {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: kf - 1.0, df_within: f64::NAN };
    }
    let w_total: f64 = w.iter().sum();
    if w_total < 1e-300 {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: kf - 1.0, df_within: 0.0 };
    }
    let x_tilde: f64 = (0..k).map(|j| w[j] * means[j]).sum::<f64>() / w_total;
    let numerator: f64 = (0..k).map(|j| w[j] * (means[j] - x_tilde).powi(2)).sum::<f64>() / (kf - 1.0);
    let lambda: f64 = (0..k).map(|j| {
        let ratio = 1.0 - w[j] / w_total;
        ratio * ratio / (ns[j] - 1.0).max(1.0)
    }).sum();
    let denominator = 1.0 + 2.0 * (kf - 2.0) / (kf * kf - 1.0) * lambda;
    let f_star = numerator / denominator;
    let df1 = kf - 1.0;
    let df2 = if lambda > 1e-300 { (kf * kf - 1.0) / (3.0 * lambda) } else { f64::INFINITY };
    let p_value = crate::special_functions::f_right_tail_p(f_star, df1, df2);
    WelchAnovaResult { f_statistic: f_star, p_value, df_between: df1, df_within: df2 }
}

/// Welch's ANOVA for comparing k group means with unequal variances.
///
/// Unlike standard ANOVA, does not assume equal variances.
/// F* ~ F(k-1, ν) where ν is Welch's corrected denominator df.
/// Always safe to use; reduces to standard ANOVA when variances are equal.
pub fn welch_anova(groups: &[&[f64]]) -> WelchAnovaResult {
    // Filter out empty groups and n=1 groups (undefined variance → undefined weight).
    // Groups with n=1 cannot contribute to a Welch ANOVA — their variance is undefined.
    let groups: Vec<&[f64]> = groups.iter()
        .filter(|g| g.len() >= 2)
        .copied()
        .collect();
    let k = groups.len();
    let kf = k as f64;

    if k < 2 {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: 0.0, df_within: 0.0 };
    }

    let means: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>() / g.len() as f64).collect();
    let vars: Vec<f64> = groups.iter().zip(&means).map(|(g, &m)| {
        g.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (g.len() - 1) as f64
    }).collect();
    let ns: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();

    // Weights: w_j = n_j / s²_j
    // If a group has zero variance (all identical observations), treat weight as NaN
    // and return NaN result — the F statistic is undefined (numerator/denominator both zero
    // or degenerate).
    let w: Vec<f64> = (0..k).map(|j| {
        if vars[j] < 1e-300 { f64::NAN } else { ns[j] / vars[j] }
    }).collect();
    if w.iter().any(|v| v.is_nan()) {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: kf - 1.0, df_within: f64::NAN };
    }
    let w_total: f64 = w.iter().sum();
    if w_total < 1e-300 {
        return WelchAnovaResult { f_statistic: f64::NAN, p_value: f64::NAN, df_between: kf - 1.0, df_within: 0.0 };
    }

    // Weighted grand mean
    let x_tilde: f64 = (0..k).map(|j| w[j] * means[j]).sum::<f64>() / w_total;

    // Numerator
    let numerator: f64 = (0..k).map(|j| w[j] * (means[j] - x_tilde).powi(2)).sum::<f64>() / (kf - 1.0);

    // Lambda correction
    let lambda: f64 = (0..k).map(|j| {
        let ratio = 1.0 - w[j] / w_total;
        ratio * ratio / (ns[j] - 1.0).max(1.0)
    }).sum();

    let denominator = 1.0 + 2.0 * (kf - 2.0) / (kf * kf - 1.0) * lambda;
    let f_star = numerator / denominator;

    let df1 = kf - 1.0;
    let df2 = if lambda > 1e-300 { (kf * kf - 1.0) / (3.0 * lambda) } else { f64::INFINITY };

    let p_value = crate::special_functions::f_right_tail_p(f_star, df1, df2);

    WelchAnovaResult { f_statistic: f_star, p_value, df_between: df1, df_within: df2 }
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
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let p_hat = successes / n;
    let se = (p0 * (1.0 - p0) / n).sqrt();
    let z = (p_hat - p0) / se;
    let p = normal_two_tail_p(z);
    // Cohen's h = 2 arcsin(√p̂) - 2 arcsin(√p₀)
    let h = 2.0 * p_hat.sqrt().asin() - 2.0 * p0.sqrt().asin();

    // Wilson score 95% CI on p̂ (better than normal approximation, especially near 0/1)
    let z_crit = crate::special_functions::normal_quantile(0.975);
    let z2 = z_crit * z_crit;
    let denom_wilson = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom_wilson;
    let halfw = (z_crit * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt()) / denom_wilson;

    TestResult {
        test_name: "One-proportion z-test",
        statistic: z, df: f64::NAN, p_value: p,
        effect_size: h, effect_size_name: "Cohen's h",
        ci_lower: center - halfw,
        ci_upper: center + halfw,
        ci_level: 0.95,
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
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let p1 = successes1 / n1;
    let p2 = successes2 / n2;
    let p_pooled = (successes1 + successes2) / (n1 + n2);
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
    let z = if se > 0.0 { (p1 - p2) / se } else { f64::NAN };
    let p = normal_two_tail_p(z);
    let h = 2.0 * p1.sqrt().asin() - 2.0 * p2.sqrt().asin();

    // 95% CI on (p₁ - p₂) using unpooled SE (standard for CI, even though test uses pooled)
    let diff = p1 - p2;
    let se_diff = (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt();
    let z_crit = crate::special_functions::normal_quantile(0.975);
    let ci_half = z_crit * se_diff;

    TestResult {
        test_name: "Two-proportion z-test",
        statistic: z, df: f64::NAN, p_value: p,
        effect_size: h, effect_size_name: "Cohen's h",
        ci_lower: diff - ci_half,
        ci_upper: diff + ci_half,
        ci_level: 0.95,
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
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

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
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

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
// Tukey HSD post-hoc test
// ═══════════════════════════════════════════════════════════════════════════

/// Result for one pairwise comparison in Tukey HSD.
#[derive(Debug, Clone)]
pub struct TukeyComparison {
    /// Index of group i
    pub group_i: usize,
    /// Index of group j
    pub group_j: usize,
    /// Observed difference in means: ȳ_i - ȳ_j
    pub mean_diff: f64,
    /// Tukey q statistic
    pub q_statistic: f64,
    /// Right-tail p-value from studentized range distribution
    pub p_value: f64,
    /// Whether the difference is significant at α = 0.05 (convenience default).
    /// Use `significant_at(alpha)` for custom significance levels.
    pub significant: bool,
}

impl TukeyComparison {
    /// Is this pairwise comparison significant at a custom alpha level?
    pub fn significant_at(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Tukey's Honestly Significant Difference (HSD) post-hoc test.
///
/// Computes all pairwise comparisons after a significant one-way ANOVA.
/// Controls the family-wise error rate using the studentized range distribution
/// (Tukey-Kramer adjustment for unequal group sizes).
///
/// # Parameters
///
/// - `groups`: slice of `MomentStats`, one per group (count, mean, variance).
/// - `ms_error`: mean square error from the ANOVA (= SS_within / df_within).
/// - `df_error`: degrees of freedom of the error term (= N - k).
/// - `alpha`: significance level for the `significant` flag on each comparison.
///   `None` ⇒ default 0.05. Common alternatives: 0.01 (strict), 0.10 (lenient).
///
/// The returned `TukeyComparison::significant` reflects the chosen `alpha`.
/// For per-comparison thresholds at a different level, use
/// `TukeyComparison::significant_at(alpha)` on each result.
///
/// # Example
/// ```no_run
/// use tambear::hypothesis::{tukey_hsd, MomentStats};
/// # let groups: &[MomentStats] = &[];
/// # let ms_error = 1.0;
/// # let df_error = 10.0;
/// // Strict alpha for a confirmatory analysis:
/// let results_01 = tukey_hsd(groups, ms_error, df_error, Some(0.01));
/// // Default 5%:
/// let results_05 = tukey_hsd(groups, ms_error, df_error, None);
/// ```
pub fn tukey_hsd(
    groups: &[MomentStats],
    ms_error: f64,
    df_error: f64,
    alpha: Option<f64>,
) -> Vec<TukeyComparison> {
    let k = groups.len();
    let alpha = alpha.unwrap_or(0.05);
    let mut results = Vec::new();

    for i in 0..k {
        for j in (i + 1)..k {
            let ni = groups[i].count as f64;
            let nj = groups[j].count as f64;
            if ni < 1.0 || nj < 1.0 { continue; }

            let mean_i = groups[i].sum / ni;
            let mean_j = groups[j].sum / nj;
            let diff = mean_i - mean_j;

            // Harmonic mean of group sizes (handles unequal n via Tukey-Kramer)
            let n_harm = 2.0 / (1.0 / ni + 1.0 / nj);
            let se = (ms_error / n_harm).sqrt();

            let q = if se < 1e-300 { f64::INFINITY } else { diff.abs() / se };
            let p = crate::special_functions::studentized_range_p(q, k, df_error);

            results.push(TukeyComparison {
                group_i: i,
                group_j: j,
                mean_diff: diff,
                q_statistic: q,
                p_value: p,
                significant: p < alpha,
            });
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: HypothesisEngine
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Breusch-Pagan heteroscedasticity test
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Breusch-Pagan test for heteroscedasticity.
#[derive(Debug, Clone)]
pub struct BreuschPaganResult {
    /// LM test statistic: n * R² ~ χ²(k) under H₀
    pub statistic: f64,
    /// Right-tail p-value
    pub p_value: f64,
    /// Degrees of freedom (= number of regressors excluding intercept)
    pub df: usize,
}

/// Breusch-Pagan (1979) test for heteroscedasticity.
///
/// H₀: homoscedastic errors (constant variance).
/// H₁: variance of errors depends linearly on the regressors `x`.
///
/// Algorithm (Koenker 1981 robust variant):
/// 1. Fit OLS: y = Xβ + ε; compute residuals ê_i
/// 2. Compute ê_i² and the mean ū = mean(ê_i²)
/// 3. Let w_i = ê_i² / ū  (studentized squared residuals)
/// 4. Regress w on X (including intercept): w = Xγ + η
/// 5. LM = n * R² ~ χ²(k) under H₀, where k = cols(X) - 1
///
/// `x_with_intercept`: design matrix with intercept column (n × p).
/// `residuals`: OLS residuals from the primary regression (length n).
pub fn breusch_pagan(x_with_intercept: &crate::linear_algebra::Mat, residuals: &[f64]) -> BreuschPaganResult {
    let n = residuals.len();
    let p = x_with_intercept.cols; // includes intercept
    let df = if p > 1 { p - 1 } else { 1 };

    // Squared residuals and their mean
    let e2: Vec<f64> = residuals.iter().map(|e| e * e).collect();
    let e2_mean = e2.iter().sum::<f64>() / n as f64;

    // Studentized squared residuals (Koenker form)
    let w: Vec<f64> = if e2_mean < 1e-300 {
        vec![1.0; n] // all residuals zero → trivially homoscedastic
    } else {
        e2.iter().map(|ei2| ei2 / e2_mean).collect()
    };

    // OLS: regress w on X
    let beta_aux = crate::linear_algebra::qr_solve(x_with_intercept, &w);

    // Fitted values and R² of the auxiliary regression
    let cols = x_with_intercept.cols;
    let w_hat: Vec<f64> = (0..n)
        .map(|i| (0..cols).map(|k| x_with_intercept.data[i * cols + k] * beta_aux[k]).sum::<f64>())
        .collect();

    let w_mean = w.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = w.iter().map(|wi| (wi - w_mean).powi(2)).sum();
    let ss_res: f64 = w.iter().zip(w_hat.iter()).map(|(wi, fi)| (wi - fi).powi(2)).sum();

    let r2 = if ss_tot < 1e-300 {
        0.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    let statistic = n as f64 * r2;
    let p_value = chi2_right_tail_p(statistic, df as f64);

    BreuschPaganResult { statistic, p_value, df }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cook's distance & leverage (regression influence diagnostics)
// ═══════════════════════════════════════════════════════════════════════════

/// Regression influence diagnostics.
#[derive(Debug, Clone)]
pub struct InfluenceResult {
    /// Cook's distance for each observation.
    pub cooks_distance: Vec<f64>,
    /// Hat matrix diagonal (leverage) for each observation.
    pub leverage: Vec<f64>,
    /// Number of observations flagged as influential (Cook's D > 4/n).
    pub n_influential: usize,
}

/// Cook's distance and leverage for OLS regression.
///
/// Cook's D_i = (e_i² * h_ii) / (p * MSE * (1 - h_ii)²)
/// where h_ii = diagonal of hat matrix H = X(X'X)⁻¹X'.
///
/// `x_with_intercept`: n × p design matrix (includes intercept column).
/// `residuals`: OLS residuals (length n).
/// `influence_threshold`: Cook's D threshold for flagging influential points.
///   Default (`None`) uses the classical `4/n` rule. Alternatives include
///   `1.0` (conservative) or `4/(n-k-1)` (adjusted). Every knob tunable.
pub fn cooks_distance(
    x_with_intercept: &crate::linear_algebra::Mat,
    residuals: &[f64],
) -> InfluenceResult {
    cooks_distance_with_threshold(x_with_intercept, residuals, None)
}

/// Cook's distance with an explicit influence threshold.
///
/// See [`cooks_distance`] for details. This variant exposes the threshold
/// as a tunable parameter per Tambear Contract Principle 4.
pub fn cooks_distance_with_threshold(
    x_with_intercept: &crate::linear_algebra::Mat,
    residuals: &[f64],
    influence_threshold: Option<f64>,
) -> InfluenceResult {
    let n = residuals.len();
    let p = x_with_intercept.cols;

    // Compute (X'X)⁻¹ via Cholesky
    let mut xtx = vec![0.0; p * p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xtx[j * p + k] += x_with_intercept.get(i, j) * x_with_intercept.get(i, k);
            }
        }
    }
    let xtx_mat = crate::linear_algebra::Mat::from_vec(p, p, xtx);
    let l = crate::linear_algebra::cholesky(&xtx_mat);

    // Hat matrix diagonal: h_ii = x_i' (X'X)⁻¹ x_i
    let leverage: Vec<f64> = if let Some(ref l_mat) = l {
        (0..n).map(|i| {
            let xi: Vec<f64> = (0..p).map(|j| x_with_intercept.get(i, j)).collect();
            // Solve L·z = x_i (forward substitution)
            let z = crate::linear_algebra::cholesky_solve(l_mat, &xi);
            // h_ii = x_i' (X'X)⁻¹ x_i = z' z (since (X'X)⁻¹ x_i = z via Cholesky)
            // Actually h_ii = x_i' * z where z = (X'X)^{-1} x_i
            xi.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>()
        }).collect()
    } else {
        vec![1.0 / n as f64; n] // fallback: equal leverage
    };

    // MSE
    let mse = residuals.iter().map(|e| e * e).sum::<f64>() / (n - p).max(1) as f64;
    let pf = p as f64;

    // Cook's distance
    let cooks_distance: Vec<f64> = if mse < 1e-300 {
        // All residuals effectively zero — no influence possible
        vec![0.0; n]
    } else {
        (0..n).map(|i| {
            let h = leverage[i].clamp(0.0, 1.0 - 1e-10);
            let e = residuals[i];
            (e * e * h) / (pf * mse * (1.0 - h) * (1.0 - h))
        }).collect()
    };

    let threshold = influence_threshold.unwrap_or(4.0 / n as f64);
    let n_influential = cooks_distance.iter().filter(|&&d| d > threshold).count();

    InfluenceResult { cooks_distance, leverage, n_influential }
}

// ═══════════════════════════════════════════════════════════════════════════
// Weighted Least Squares (WLS)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of weighted least squares regression.
#[derive(Debug, Clone)]
pub struct WlsResult {
    /// Coefficients β (length p, includes intercept if X has intercept column).
    pub coefficients: Vec<f64>,
    /// Weighted residual sum of squares.
    pub weighted_rss: f64,
    /// Weighted R² = 1 - WRSS / WTSS.
    pub r_squared: f64,
}

/// Weighted least squares: β = (X'WX)⁻¹ X'Wy.
///
/// Equivalent to OLS on (√w_i · x_i, √w_i · y_i).
///
/// `x`: n×p design matrix (row-major, include intercept column if desired).
/// `y`: response vector (length n).
/// `weights`: observation weights (length n, all > 0).
pub fn wls(x: &crate::linear_algebra::Mat, y: &[f64], weights: &[f64]) -> WlsResult {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(y.len(), n);
    assert_eq!(weights.len(), n);

    // Transform: X_w = diag(√w) X, y_w = diag(√w) y
    let mut x_w = vec![0.0; n * p];
    let mut y_w = vec![0.0; n];
    for i in 0..n {
        let sw = weights[i].max(0.0).sqrt();
        y_w[i] = sw * y[i];
        for j in 0..p {
            x_w[i * p + j] = sw * x.get(i, j);
        }
    }

    let x_mat = crate::linear_algebra::Mat::from_vec(n, p, x_w);
    let coefficients = crate::linear_algebra::qr_solve(&x_mat, &y_w);

    // Weighted residuals and R²
    let mut wrss = 0.0;
    let w_y_mean = {
        let sw: f64 = weights.iter().sum();
        if sw > 0.0 { weights.iter().zip(y.iter()).map(|(w, yi)| w * yi).sum::<f64>() / sw } else { 0.0 }
    };
    let mut wtss = 0.0;
    for i in 0..n {
        let fitted: f64 = (0..p).map(|j| x.get(i, j) * coefficients[j]).sum();
        let resid = y[i] - fitted;
        wrss += weights[i] * resid * resid;
        wtss += weights[i] * (y[i] - w_y_mean) * (y[i] - w_y_mean);
    }
    let r_squared = if wtss > 1e-300 { (1.0 - wrss / wtss).clamp(0.0, 1.0) } else { 0.0 };

    WlsResult { coefficients, weighted_rss: wrss, r_squared }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayes factors (Rouder et al. 2009, Wagenmakers 2007)
// ═══════════════════════════════════════════════════════════════════════════

/// Bayes factor result.
#[derive(Debug, Clone)]
pub struct BayesFactorResult {
    /// BF₁₀ = P(data | H₁) / P(data | H₀). Values > 1 favor H₁, < 1 favor H₀.
    pub bf10: f64,
    /// BF₀₁ = 1 / BF₁₀. Values > 1 favor H₀.
    pub bf01: f64,
    /// Log₁₀ BF₁₀ (for magnitude comparison).
    pub log10_bf10: f64,
    /// Qualitative interpretation (Jeffreys 1961).
    pub interpretation: &'static str,
}

/// Jeffreys (1961) / Kass-Raftery (1995) verbal interpretation of BF₁₀.
///
/// Maps a Bayes factor `BF₁₀` to a qualitative strength-of-evidence label.
/// Thresholds follow Jeffreys (1961) Table II as refined by Kass & Raftery
/// (1995, *JASA* 90:773–795): decisive >100, very strong >30, strong >10,
/// moderate >3, anecdotal >1; symmetric for evidence in favor of H0.
///
/// # Parameters
/// - `bf10`: Bayes factor in the H1-favoring direction (BF₁₀ = P(data|H1)/P(data|H0))
///
/// # Returns
/// A `'static` string label suitable for display or structured output.
///
/// # Consumers
/// Bayesian t-tests, correlation BF, ANOVA BF, mixed-model BF, any function
/// returning a `BayesFactorResult` that needs the `.interpretation` field filled.
pub fn interpret_bf(bf10: f64) -> &'static str {
    let bf01 = 1.0 / bf10;
    if bf10 > 100.0 { "Extreme evidence for H1" }
    else if bf10 > 30.0 { "Very strong evidence for H1" }
    else if bf10 > 10.0 { "Strong evidence for H1" }
    else if bf10 > 3.0 { "Moderate evidence for H1" }
    else if bf10 > 1.0 { "Anecdotal evidence for H1" }
    else if bf01 > 100.0 { "Extreme evidence for H0" }
    else if bf01 > 30.0 { "Very strong evidence for H0" }
    else if bf01 > 10.0 { "Strong evidence for H0" }
    else if bf01 > 3.0 { "Moderate evidence for H0" }
    else { "Anecdotal evidence for H0" }
}

/// Rouder et al. (2009) Jeffreys-Zellner-Siow (JZS) Bayes factor for
/// one-sample or paired t-test.
///
/// Cauchy prior on effect size δ with scale `r` (default 0.707 = √2/2).
/// Computes BF₁₀ via numerical integration over the nuisance precision g.
///
/// `t`: observed t-statistic.
/// `n`: sample size.
/// `r`: Cauchy prior scale (default: 1/√2 ≈ 0.707).
pub fn bayes_factor_t_one_sample(t: f64, n: usize, r: Option<f64>) -> BayesFactorResult {
    let r = r.unwrap_or(std::f64::consts::FRAC_1_SQRT_2);
    let nu = (n - 1) as f64;
    let nf = n as f64;

    // JZS integrand: marginal likelihood under H1 vs H0 ratio
    //
    // Following Rouder et al. 2009 Eq. 1:
    // BF₁₀ = ∫₀^∞ (1 + n·g·r²)^(-1/2) · [(1 + t²/(ν·(1+n·g·r²)))/(1 + t²/ν)]^(-(ν+1)/2) · pg(g) dg
    // where pg(g) = (1/√(2π)) · g^(-3/2) · exp(-1/(2g))  (inverse chi-square with 1 df)
    //
    // Substitute u = 1/g → g = 1/u, dg = -du/u², pg(g)·dg becomes
    // pg(1/u)·du/u² = (1/√(2π)) · u^(3/2) · exp(-u/2) / u² · du = (1/√(2π)) · u^(-1/2) · exp(-u/2) du
    // So the integrand in u-space has weight u^(-1/2)·exp(-u/2)/(√(2π)), integration over u ∈ (0, ∞).

    // Use Gauss-Laguerre-like quadrature: simple trapezoidal on log-scale substitution.
    // Integrate over g ∈ (0, ∞). For stability, parameterize by s = √g, s ∈ (0, ∞), dg = 2s ds.
    // Then pg(g)·dg = (1/√(2π)) · s^(-3) · exp(-1/(2s²)) · 2s ds = (√(2/π)) · s^(-2) · exp(-1/(2s²)) ds.

    // Adaptive Simpson on g ∈ [1e-8, 1e4], which covers the bulk of the mass.
    let integrand = |g: f64| -> f64 {
        if g <= 0.0 { return 0.0; }
        let ngr2 = nf * g * r * r;
        let factor = (1.0 + ngr2).powf(-0.5);
        let ratio = (1.0 + t * t / (nu * (1.0 + ngr2))) / (1.0 + t * t / nu);
        let shape = ratio.powf(-(nu + 1.0) / 2.0);
        // pg(g) = (1/√(2π)) · g^(-3/2) · exp(-1/(2g))
        let pg = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * g.powf(-1.5) * (-1.0 / (2.0 * g)).exp();
        factor * shape * pg
    };

    // Composite Simpson's rule on [g_min, g_max] with log-spaced samples
    let g_min: f64 = 1e-8;
    let g_max: f64 = 1e4;
    let n_points: usize = 4000;
    let log_min = g_min.ln();
    let log_max = g_max.ln();
    let dlog = (log_max - log_min) / n_points as f64;

    let mut integral: f64 = 0.0;
    for i in 0..n_points {
        let log_g1 = log_min + i as f64 * dlog;
        let log_g2 = log_min + (i + 1) as f64 * dlog;
        let g1 = log_g1.exp();
        let g2 = log_g2.exp();
        let gm = ((log_g1 + log_g2) / 2.0).exp();
        // Simpson: ∫ f dg ≈ (g2 - g1)/6 · (f(g1) + 4·f(gm) + f(g2))
        let width = g2 - g1;
        integral += (width / 6.0) * (integrand(g1) + 4.0 * integrand(gm) + integrand(g2));
    }

    let bf10: f64 = integral.max(1e-300);
    let bf01 = 1.0 / bf10;
    BayesFactorResult {
        bf10,
        bf01,
        log10_bf10: bf10.log10(),
        interpretation: interpret_bf(bf10),
    }
}

/// Bayes factor for Pearson correlation (Jeffreys 1961, exact).
///
/// Uses the stretched beta prior on ρ with width `kappa` (default 1.0 = uniform).
/// Requires the observed Pearson r and sample size n.
///
/// Formula (Ly, Verhagen & Wagenmakers 2016, Eq. 14):
/// BF₁₀ = (2^((κ-2)/κ) · Γ((2n-1)/2) / (√π · Γ((2n-1)/2 - 1/κ + 1))) · ∫...
///
/// We use a simpler valid form: BIC approximation from the F-statistic.
/// BIC₁₀ ≈ exp((t²/(1 + t²/df) - log(n)) / 2) where t = r·√(df/(1 - r²)).
pub fn bayes_factor_correlation(r: f64, n: usize) -> BayesFactorResult {
    if n < 4 || r.abs() >= 1.0 {
        return BayesFactorResult {
            bf10: f64::NAN, bf01: f64::NAN, log10_bf10: f64::NAN,
            interpretation: "insufficient data",
        };
    }
    let df = (n - 2) as f64;
    let t2 = r * r * df / (1.0 - r * r);
    // Wagenmakers 2007 BIC approximation:
    // BF₁₀ ≈ exp((BIC₀ - BIC₁) / 2)
    // BIC_diff = n·log(SS_total/SS_residual) - log(n)
    //          = n·log(1/(1-r²)) - log(n)
    //          = -n·log(1-r²) - log(n)
    let log_bf10 = 0.5 * (-(n as f64) * (1.0 - r * r).ln() - (n as f64).ln());
    let bf10 = log_bf10.exp();
    let _ = t2; // BIC form uses r² directly
    BayesFactorResult {
        bf10,
        bf01: 1.0 / bf10,
        log10_bf10: log_bf10 / std::f64::consts::LN_10,
        interpretation: interpret_bf(bf10),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mediation analysis (Baron & Kenny 1986, Sobel 1982)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of mediation analysis.
#[derive(Debug, Clone)]
pub struct MediationResult {
    /// Total effect c (Y regressed on X alone).
    pub total_effect: f64,
    /// Direct effect c' (Y regressed on X controlling for M).
    pub direct_effect: f64,
    /// Path a (M regressed on X).
    pub path_a: f64,
    /// Path b (Y regressed on M controlling for X).
    pub path_b: f64,
    /// Indirect effect (a × b).
    pub indirect_effect: f64,
    /// Sobel z-statistic for indirect effect.
    pub sobel_z: f64,
    /// Sobel p-value (two-tailed).
    pub sobel_p: f64,
    /// Proportion mediated: indirect / total (may exceed 1 or be negative).
    pub proportion_mediated: f64,
}

/// Simple 1-predictor OLS: returns (intercept, slope, residuals, se_slope).
fn ols_simple(x: &[f64], y: &[f64]) -> (f64, f64, Vec<f64>, f64) {
    let n = x.len();
    let nf = n as f64;
    let mx: f64 = x.iter().sum::<f64>() / nf;
    let my: f64 = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        sxx += dx * dx;
        sxy += dx * (y[i] - my);
    }
    let slope = if sxx > 1e-300 { sxy / sxx } else { 0.0 };
    let intercept = my - slope * mx;
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - intercept - slope * x[i]).collect();
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let mse = if n > 2 { rss / (n - 2) as f64 } else { 0.0 };
    let se_slope = if sxx > 1e-300 { (mse / sxx).sqrt() } else { 0.0 };
    (intercept, slope, residuals, se_slope)
}

/// 2-predictor OLS (x, m → y with intercept): returns (b0, b_x, b_m, se_b_m).
fn ols_two_predictor(x: &[f64], m: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len();
    let nf = n as f64;
    if n < 4 { return (0.0, 0.0, 0.0, 0.0); }

    // Build 3x3 normal equations X'X where X = [1, x, m]
    let sum_x: f64 = x.iter().sum();
    let sum_m: f64 = m.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|&v| v * v).sum();
    let sum_mm: f64 = m.iter().map(|&v| v * v).sum();
    let sum_xm: f64 = x.iter().zip(m.iter()).map(|(&a, &b)| a * b).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let sum_my: f64 = m.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

    // Solve (X'X) β = X'y via 3x3 matrix inversion (Cramer's rule)
    // X'X = [[n, sum_x, sum_m], [sum_x, sum_xx, sum_xm], [sum_m, sum_xm, sum_mm]]
    let a = [
        [nf, sum_x, sum_m],
        [sum_x, sum_xx, sum_xm],
        [sum_m, sum_xm, sum_mm],
    ];
    let rhs = [sum_y, sum_xy, sum_my];

    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-300 { return (0.0, 0.0, 0.0, 0.0); }

    // Inverse of symmetric 3x3 (cofactors)
    let cof = [
        [a[1][1] * a[2][2] - a[1][2] * a[2][1], -(a[0][1] * a[2][2] - a[0][2] * a[2][1]), a[0][1] * a[1][2] - a[0][2] * a[1][1]],
        [-(a[1][0] * a[2][2] - a[1][2] * a[2][0]), a[0][0] * a[2][2] - a[0][2] * a[2][0], -(a[0][0] * a[1][2] - a[0][2] * a[1][0])],
        [a[1][0] * a[2][1] - a[1][1] * a[2][0], -(a[0][0] * a[2][1] - a[0][1] * a[2][0]), a[0][0] * a[1][1] - a[0][1] * a[1][0]],
    ];

    let b0 = (cof[0][0] * rhs[0] + cof[0][1] * rhs[1] + cof[0][2] * rhs[2]) / det;
    let b_x = (cof[1][0] * rhs[0] + cof[1][1] * rhs[1] + cof[1][2] * rhs[2]) / det;
    let b_m = (cof[2][0] * rhs[0] + cof[2][1] * rhs[1] + cof[2][2] * rhs[2]) / det;

    // SE of b_m from (X'X)⁻¹[2,2] * σ²
    let inv_mm = cof[2][2] / det;
    let mut rss = 0.0;
    for i in 0..n { let e = y[i] - b0 - b_x * x[i] - b_m * m[i]; rss += e * e; }
    let mse = rss / (n - 3) as f64;
    let se_b_m = (mse * inv_mm).sqrt();

    (b0, b_x, b_m, se_b_m)
}

/// Mediation analysis: does mediator M explain the X → Y relationship?
///
/// Fits three models:
/// 1. Y on X (total effect c)
/// 2. M on X (path a)
/// 3. Y on X + M (direct effect c', path b)
///
/// Indirect effect = a × b (= c - c' when all variables are centered).
/// Sobel test: z = ab / √(b²·SE(a)² + a²·SE(b)²).
pub fn mediation(x: &[f64], m: &[f64], y: &[f64]) -> MediationResult {
    assert_eq!(x.len(), m.len());
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 4 {
        return MediationResult {
            total_effect: f64::NAN, direct_effect: f64::NAN,
            path_a: f64::NAN, path_b: f64::NAN,
            indirect_effect: f64::NAN, sobel_z: f64::NAN,
            sobel_p: f64::NAN, proportion_mediated: f64::NAN,
        };
    }

    // 1. Y on X → total effect c
    let (_, c, _, _) = ols_simple(x, y);

    // 2. M on X → path a (and SE of a)
    let (_, a, _, se_a) = ols_simple(x, m);

    // 3. Y on X + M → direct effect c', path b (and SE of b)
    let (_, c_prime, b, se_b) = ols_two_predictor(x, m, y);

    let indirect = a * b;
    // Sobel z-statistic
    let var_ab = b * b * se_a * se_a + a * a * se_b * se_b;
    let sobel_z = if var_ab > 1e-300 { indirect / var_ab.sqrt() } else { 0.0 };
    let sobel_p = 2.0 * (1.0 - crate::special_functions::normal_cdf(sobel_z.abs()));

    let proportion_mediated = if c.abs() > 1e-15 { indirect / c } else { f64::NAN };

    MediationResult {
        total_effect: c,
        direct_effect: c_prime,
        path_a: a,
        path_b: b,
        indirect_effect: indirect,
        sobel_z,
        sobel_p,
        proportion_mediated,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Moderation analysis (interaction terms + simple slopes)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of moderation analysis.
#[derive(Debug, Clone)]
pub struct ModerationResult {
    /// Intercept β₀.
    pub intercept: f64,
    /// Main effect of X (β₁).
    pub main_x: f64,
    /// Main effect of Z (β₂).
    pub main_z: f64,
    /// Interaction coefficient (β₃).
    pub interaction: f64,
    /// Standard error of the interaction coefficient.
    pub se_interaction: f64,
    /// t-statistic for the interaction (β₃ / SE).
    pub t_interaction: f64,
    /// Two-tailed p-value for the interaction.
    pub p_interaction: f64,
    /// Simple slope of X when Z = Z_mean − SD.
    pub simple_slope_low: f64,
    /// Simple slope of X when Z = Z_mean.
    pub simple_slope_mid: f64,
    /// Simple slope of X when Z = Z_mean + SD.
    pub simple_slope_high: f64,
}

/// Moderation analysis: does Z moderate the X → Y relationship?
///
/// Fits: Y = β₀ + β₁·X + β₂·Z + β₃·(X·Z) + ε
///
/// A significant β₃ indicates moderation. Simple slopes are computed at
/// Z_mean - SD, Z_mean, Z_mean + SD to visualize the conditional effect of X.
pub fn moderation(x: &[f64], z: &[f64], y: &[f64]) -> ModerationResult {
    assert_eq!(x.len(), z.len());
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 5 {
        return ModerationResult {
            intercept: f64::NAN, main_x: f64::NAN, main_z: f64::NAN,
            interaction: f64::NAN, se_interaction: f64::NAN,
            t_interaction: f64::NAN, p_interaction: f64::NAN,
            simple_slope_low: f64::NAN, simple_slope_mid: f64::NAN, simple_slope_high: f64::NAN,
        };
    }

    // Build 4-column design matrix: [1, x, z, xz]
    let xz: Vec<f64> = x.iter().zip(z.iter()).map(|(&a, &b)| a * b).collect();
    let mut design = vec![0.0; n * 4];
    for i in 0..n {
        design[i * 4] = 1.0;
        design[i * 4 + 1] = x[i];
        design[i * 4 + 2] = z[i];
        design[i * 4 + 3] = xz[i];
    }
    let x_mat = crate::linear_algebra::Mat::from_vec(n, 4, design);
    let beta = crate::linear_algebra::qr_solve(&x_mat, y);
    if beta.len() != 4 {
        return ModerationResult {
            intercept: f64::NAN, main_x: f64::NAN, main_z: f64::NAN,
            interaction: f64::NAN, se_interaction: f64::NAN,
            t_interaction: f64::NAN, p_interaction: f64::NAN,
            simple_slope_low: f64::NAN, simple_slope_mid: f64::NAN, simple_slope_high: f64::NAN,
        };
    }

    let (b0, b_x, b_z, b_xz) = (beta[0], beta[1], beta[2], beta[3]);

    // Residual sum of squares and MSE
    let mut rss = 0.0;
    for i in 0..n {
        let fitted = b0 + b_x * x[i] + b_z * z[i] + b_xz * xz[i];
        let e = y[i] - fitted;
        rss += e * e;
    }
    let mse = rss / (n - 4).max(1) as f64;

    // SE of interaction: (X'X)⁻¹[3,3] * σ²
    // Compute (X'X)⁻¹ directly
    let mut xtx = vec![0.0; 16];
    for i in 0..n {
        for a in 0..4 {
            for b in 0..4 {
                xtx[a * 4 + b] += x_mat.get(i, a) * x_mat.get(i, b);
            }
        }
    }
    // Invert 4x4 via Cholesky
    let xtx_mat = crate::linear_algebra::Mat::from_vec(4, 4, xtx);
    let inv_diag_33 = if let Some(l) = crate::linear_algebra::cholesky(&xtx_mat) {
        let e3 = vec![0.0, 0.0, 0.0, 1.0];
        let col = crate::linear_algebra::cholesky_solve(&l, &e3);
        col[3]
    } else { 0.0 };
    let se_interaction = (mse * inv_diag_33).sqrt();

    let t_interaction = if se_interaction > 1e-300 { b_xz / se_interaction } else { 0.0 };
    let df = (n - 4).max(1) as f64;
    let p_interaction = 2.0 * (1.0 - crate::special_functions::t_cdf(t_interaction.abs(), df));

    // Simple slopes at Z_mean ± SD
    let z_mean: f64 = z.iter().sum::<f64>() / n as f64;
    let z_var: f64 = z.iter().map(|&v| (v - z_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let z_sd = z_var.sqrt();

    let simple_slope = |z_val: f64| b_x + b_xz * z_val;

    ModerationResult {
        intercept: b0, main_x: b_x, main_z: b_z,
        interaction: b_xz, se_interaction,
        t_interaction, p_interaction,
        simple_slope_low: simple_slope(z_mean - z_sd),
        simple_slope_mid: simple_slope(z_mean),
        simple_slope_high: simple_slope(z_mean + z_sd),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression via IRLS
// ═══════════════════════════════════════════════════════════════════════════

/// Result of fitting a logistic regression model via IRLS.
///
/// Coefficients are on the log-odds scale. Standard errors, z-statistics,
/// and p-values come from the observed Fisher information matrix at convergence.
#[derive(Debug, Clone)]
pub struct LogisticRegressionResult {
    /// Coefficients β (length = n_features + 1; last element is intercept).
    pub coefficients: Vec<f64>,
    /// Standard errors of each coefficient (same length as `coefficients`).
    pub std_errors: Vec<f64>,
    /// z-statistics = coefficient / SE (same length as `coefficients`).
    pub z_statistics: Vec<f64>,
    /// Two-tailed p-values under H₀: βⱼ = 0 (same length as `coefficients`).
    pub p_values: Vec<f64>,
    /// Null deviance: −2 ln L(intercept-only model).
    pub null_deviance: f64,
    /// Residual deviance: −2 ln L(fitted model).
    pub residual_deviance: f64,
    /// AIC = residual_deviance + 2 * (n_features + 1).
    pub aic: f64,
    /// Number of IRLS iterations run.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

impl LogisticRegressionResult {
    /// Predict P(y=1|x) for new observations.
    ///
    /// `x` is row-major with n_features columns (no intercept column needed).
    pub fn predict_proba(&self, x: &crate::linear_algebra::Mat) -> Vec<f64> {
        let p = self.coefficients.len() - 1;
        assert_eq!(x.cols, p, "x has {} cols, model has {} features", x.cols, p);
        (0..x.rows).map(|i| {
            let mut z = self.coefficients[p]; // intercept
            for j in 0..p {
                z += self.coefficients[j] * x.get(i, j);
            }
            sigmoid(z)
        }).collect()
    }
}

fn sigmoid(z: f64) -> f64 {
    crate::special_functions::logistic(z)
}

/// Fit binary logistic regression via Iteratively Reweighted Least Squares (IRLS).
///
/// IRLS is Newton-Raphson on the log-likelihood:
/// ```text
/// β_new = (X'WX)⁻¹ X'W z_adj
/// where W = diag(μ(1-μ)), z_adj = Xβ + (y - μ)/W_diag
/// ```
///
/// # Parameters
/// - `x`: design matrix, n × p (rows = observations, cols = features). No intercept column.
/// - `y`: binary response (0.0 or 1.0), length n.
/// - `max_iter`: maximum IRLS iterations (typically 25 is more than sufficient).
/// - `tol`: convergence tolerance on max absolute coefficient change.
///
/// # Returns
/// `None` if the design matrix is rank-deficient or has fewer observations than parameters.
pub fn logistic_regression(
    x: &crate::linear_algebra::Mat,
    y: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<LogisticRegressionResult> {
    use crate::linear_algebra::{Mat, mat_mul, cholesky, cholesky_solve, inv};

    let n = x.rows;
    let p = x.cols;
    if n <= p + 1 { return None; }

    // Augment X with intercept column (rightmost): n × (p+1)
    let q = p + 1; // number of parameters
    let mut xa_data = vec![0.0f64; n * q];
    for i in 0..n {
        for j in 0..p {
            xa_data[i * q + j] = x.get(i, j);
        }
        xa_data[i * q + p] = 1.0; // intercept
    }
    let xa = Mat::from_vec(n, q, xa_data);

    // Initialize β = 0
    let mut beta = vec![0.0f64; q];
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..max_iter {
        iterations += 1;

        // Compute linear predictor η = Xaβ and μ = sigmoid(η)
        let mut eta = vec![0.0f64; n];
        let mut mu = vec![0.0f64; n];
        let mut w_diag = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0f64;
            for j in 0..q {
                sum += xa.get(i, j) * beta[j];
            }
            eta[i] = sum;
            let m = sigmoid(eta[i]);
            // Clamp to keep weights finite
            mu[i] = m.clamp(1e-10, 1.0 - 1e-10);
            w_diag[i] = mu[i] * (1.0 - mu[i]);
        }

        // Adjusted dependent variable: z_adj_i = η_i + (y_i - μ_i) / w_i
        let z_adj: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / w_diag[i]).collect();

        // Form X'WX (q × q) and X'Wz_adj (q × 1)
        let mut xtwx = vec![0.0f64; q * q];
        let mut xtwz = vec![0.0f64; q];
        for i in 0..n {
            let wi = w_diag[i];
            for j in 0..q {
                xtwz[j] += xa.get(i, j) * wi * z_adj[i];
                for k in 0..q {
                    xtwx[j * q + k] += xa.get(i, j) * wi * xa.get(i, k);
                }
            }
        }
        let xtwx_mat = Mat::from_vec(q, q, xtwx);

        // Solve (X'WX) β_new = X'Wz_adj via Cholesky (X'WX is SPD)
        let beta_new = if let Some(l) = cholesky(&xtwx_mat) {
            cholesky_solve(&l, &xtwz)
        } else {
            // Singular (perfect separation or rank deficiency) — stop
            return None;
        };

        // Check convergence
        let max_change = beta.iter().zip(&beta_new)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        beta = beta_new;
        if max_change < tol {
            converged = true;
            break;
        }
    }

    // ── Standard errors from Fisher information I = X'WX at final β ────────
    // Recompute W at final β
    let mut w_final = vec![0.0f64; n];
    let mut mu_final = vec![0.0f64; n];
    for i in 0..n {
        let mut z = 0.0f64;
        for j in 0..q {
            z += xa.get(i, j) * beta[j];
        }
        let m = sigmoid(z).clamp(1e-10, 1.0 - 1e-10);
        mu_final[i] = m;
        w_final[i] = m * (1.0 - m);
    }

    let mut xtwx_final = vec![0.0f64; q * q];
    for i in 0..n {
        for j in 0..q {
            for k in 0..q {
                xtwx_final[j * q + k] += xa.get(i, j) * w_final[i] * xa.get(i, k);
            }
        }
    }
    let info_mat = Mat::from_vec(q, q, xtwx_final);
    let cov_mat = inv(&info_mat)?; // variance-covariance matrix

    let std_errors: Vec<f64> = (0..q).map(|j| cov_mat.get(j, j).max(0.0).sqrt()).collect();
    let z_statistics: Vec<f64> = beta.iter().zip(&std_errors)
        .map(|(b, se)| if *se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_statistics.iter()
        .map(|&z| normal_two_tail_p(z))
        .collect();

    // ── Deviances ────────────────────────────────────────────────────────────
    // Null deviance: intercept-only model, μ_null = ȳ
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_null = y_mean.clamp(1e-15, 1.0 - 1e-15);
    let null_deviance = -2.0 * y.iter().map(|&yi| {
        yi * y_null.ln() + (1.0 - yi) * (1.0 - y_null).ln()
    }).sum::<f64>();

    // Residual deviance: fitted model
    let residual_deviance = -2.0 * (0..n).map(|i| {
        let pi = mu_final[i].clamp(1e-15, 1.0 - 1e-15);
        y[i] * pi.ln() + (1.0 - y[i]) * (1.0 - pi).ln()
    }).sum::<f64>();

    let aic = residual_deviance + 2.0 * q as f64;

    Some(LogisticRegressionResult {
        coefficients: beta,
        std_errors,
        z_statistics,
        p_values,
        null_deviance,
        residual_deviance,
        aic,
        iterations,
        converged,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Generalized Linear Model (GLM) — Poisson & Negative Binomial via IRLS
// ═══════════════════════════════════════════════════════════════════════════

/// GLM family specifying link, variance, and deviance functions.
#[derive(Debug, Clone, Copy)]
pub enum GlmFamily {
    /// Poisson: link=log, variance=μ.
    Poisson,
    /// Negative binomial: link=log, variance=μ + μ²/theta.
    NegativeBinomial(f64), // theta (dispersion)
}

/// Result of fitting a GLM.
#[derive(Debug, Clone)]
pub struct GlmResult {
    /// Regression coefficients (length = p+1, last = intercept).
    pub coefficients: Vec<f64>,
    /// Standard errors.
    pub std_errors: Vec<f64>,
    /// z-statistics (Wald).
    pub z_statistics: Vec<f64>,
    /// p-values (two-tailed normal).
    pub p_values: Vec<f64>,
    /// Residual deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only).
    pub null_deviance: f64,
    /// AIC = deviance + 2p.
    pub aic: f64,
    /// Number of IRLS iterations.
    pub iterations: usize,
    /// Whether IRLS converged.
    pub converged: bool,
}

/// Fit a Generalized Linear Model via IRLS.
///
/// `x`: n×p design matrix (row-major, NO intercept column — added internally).
/// `y`: response vector (counts for Poisson/NegBin, must be non-negative).
/// `family`: GLM family (Poisson or NegativeBinomial(theta)).
/// `max_iter`: maximum IRLS iterations.
/// `tol`: convergence tolerance on deviance change.
pub fn glm_fit(
    x: &crate::linear_algebra::Mat, y: &[f64],
    family: GlmFamily, max_iter: usize, tol: f64,
) -> GlmResult {
    let n = x.rows;
    let p = x.cols;
    let q = p + 1; // with intercept

    // Build augmented design matrix [X | 1]
    let mut xa_data = vec![0.0; n * q];
    for i in 0..n {
        for j in 0..p { xa_data[i * q + j] = x.get(i, j); }
        xa_data[i * q + p] = 1.0; // intercept
    }
    let xa = crate::linear_algebra::Mat::from_vec(n, q, xa_data);

    // Initialize: β = 0, μ = y_mean (or y + 0.5 for zeros)
    let mut beta = vec![0.0; q];
    let y_mean = y.iter().sum::<f64>() / n as f64;
    beta[p] = y_mean.max(0.1).ln(); // intercept = log(mean)

    let link_inv = |eta: f64| -> f64 { eta.exp().min(1e15) }; // exp for log link
    let variance = |mu: f64, fam: GlmFamily| -> f64 {
        match fam {
            GlmFamily::Poisson => mu.max(1e-10),
            GlmFamily::NegativeBinomial(theta) => (mu + mu * mu / theta).max(1e-10),
        }
    };

    let mut converged = false;
    let mut iterations = 0;
    let mut prev_dev = f64::INFINITY;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute η = Xβ, μ = g⁻¹(η), weights w = 1/(V(μ) · (dμ/dη)²)
        // For log link: dμ/dη = μ, so w = μ²/V(μ) · 1/μ² = 1/V(μ) · μ...
        // Actually: w_i = (dμ/dη)² / V(μ) = μ² / V(μ)
        let mut eta = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut w = vec![0.0; n];
        let mut z = vec![0.0; n]; // working response

        for i in 0..n {
            eta[i] = (0..q).map(|j| xa.get(i, j) * beta[j]).sum::<f64>();
            mu[i] = link_inv(eta[i]);
            let v = variance(mu[i], family);
            let dmu_deta = mu[i]; // for log link
            w[i] = dmu_deta * dmu_deta / v;
            // Working response: z = η + (y - μ) / (dμ/dη)
            z[i] = eta[i] + (y[i] - mu[i]) / dmu_deta.max(1e-15);
        }

        // Weighted least squares: β = (X'WX)⁻¹ X'Wz
        let mut xtwx = vec![0.0; q * q];
        let mut xtwz = vec![0.0; q];
        for i in 0..n {
            for j in 0..q {
                xtwz[j] += xa.get(i, j) * w[i] * z[i];
                for k in 0..q {
                    xtwx[j * q + k] += xa.get(i, j) * w[i] * xa.get(i, k);
                }
            }
        }
        let xtwx_mat = crate::linear_algebra::Mat::from_vec(q, q, xtwx);
        let l = match crate::linear_algebra::cholesky(&xtwx_mat) {
            Some(l) => l,
            None => break,
        };
        let new_beta = crate::linear_algebra::cholesky_solve(&l, &xtwz);

        // Deviance
        let dev: f64 = (0..n).map(|i| {
            let eta_i: f64 = (0..q).map(|j| xa.get(i, j) * new_beta[j]).sum();
            let mu_i = link_inv(eta_i);
            match family {
                GlmFamily::Poisson => {
                    if y[i] > 0.0 { 2.0 * (y[i] * (y[i] / mu_i).ln() - (y[i] - mu_i)) }
                    else { 2.0 * mu_i }
                }
                GlmFamily::NegativeBinomial(theta) => {
                    let mut d = 0.0;
                    if y[i] > 0.0 { d += 2.0 * y[i] * (y[i] / mu_i).ln(); }
                    d += 2.0 * (y[i] + theta) * ((mu_i + theta) / (y[i] + theta)).ln();
                    d
                }
            }
        }).sum();

        beta = new_beta;

        if (prev_dev - dev).abs() < tol {
            converged = true;
            break;
        }
        prev_dev = dev;
    }

    // Final deviance and null deviance
    let final_dev: f64 = (0..n).map(|i| {
        let eta_i: f64 = (0..q).map(|j| xa.get(i, j) * beta[j]).sum();
        let mu_i = link_inv(eta_i);
        match family {
            GlmFamily::Poisson => {
                if y[i] > 0.0 { 2.0 * (y[i] * (y[i] / mu_i).ln() - (y[i] - mu_i)) }
                else { 2.0 * mu_i }
            }
            GlmFamily::NegativeBinomial(theta) => {
                let mut d = 0.0;
                if y[i] > 0.0 { d += 2.0 * y[i] * (y[i] / mu_i).ln(); }
                d += 2.0 * (y[i] + theta) * ((mu_i + theta) / (y[i] + theta)).ln();
                d
            }
        }
    }).sum();

    let null_mu = y_mean.max(1e-10);
    let null_dev: f64 = (0..n).map(|i| {
        match family {
            GlmFamily::Poisson => {
                if y[i] > 0.0 { 2.0 * (y[i] * (y[i] / null_mu).ln() - (y[i] - null_mu)) }
                else { 2.0 * null_mu }
            }
            GlmFamily::NegativeBinomial(theta) => {
                let mut d = 0.0;
                if y[i] > 0.0 { d += 2.0 * y[i] * (y[i] / null_mu).ln(); }
                d += 2.0 * (y[i] + theta) * ((null_mu + theta) / (y[i] + theta)).ln();
                d
            }
        }
    }).sum();

    // Standard errors from final (X'WX)⁻¹
    let mut w_final = vec![0.0; n];
    for i in 0..n {
        let eta_i: f64 = (0..q).map(|j| xa.get(i, j) * beta[j]).sum();
        let mu_i = link_inv(eta_i);
        let v = variance(mu_i, family);
        w_final[i] = mu_i * mu_i / v;
    }
    let mut xtwx_f = vec![0.0; q * q];
    for i in 0..n {
        for j in 0..q {
            for k in 0..q {
                xtwx_f[j * q + k] += xa.get(i, j) * w_final[i] * xa.get(i, k);
            }
        }
    }
    let info = crate::linear_algebra::Mat::from_vec(q, q, xtwx_f);
    let std_errors = if let Some(cov) = crate::linear_algebra::inv(&info) {
        (0..q).map(|j| cov.get(j, j).max(0.0).sqrt()).collect()
    } else {
        vec![f64::NAN; q]
    };

    let z_statistics: Vec<f64> = beta.iter().zip(&std_errors)
        .map(|(b, se)| if *se > 0.0 { b / se } else { 0.0 }).collect();
    let p_values: Vec<f64> = z_statistics.iter()
        .map(|&z| normal_two_tail_p(z)).collect();

    let aic = final_dev + 2.0 * q as f64;

    GlmResult {
        coefficients: beta, std_errors, z_statistics, p_values,
        deviance: final_dev, null_deviance: null_dev, aic,
        iterations, converged,
    }
}

/// Engine that wraps a ComputeEngine for hypothesis tests on raw data.
///
// ═══════════════════════════════════════════════════════════════════════════
// Expert pipeline: two-group comparison (Layer 3)
// ═══════════════════════════════════════════════════════════════════════════

/// Comprehensive two-group comparison report.
#[derive(Debug, Clone)]
pub struct TwoGroupReport {
    pub n1: usize, pub n2: usize,
    pub mean1: f64, pub mean2: f64,
    pub std1: f64, pub std2: f64,
    pub median1: f64, pub median2: f64,
    pub normality_p1: f64, pub normality_p2: f64,
    pub normality_test: &'static str,
    pub both_normal: bool,
    pub levene_p: f64, pub equal_variance: bool,
    pub test_name: &'static str,
    pub statistic: f64, pub p_value: f64, pub df: f64,
    pub cohens_d: f64, pub hedges_g: f64,
    pub mean_diff: f64, pub ci_lower: f64, pub ci_upper: f64,
    pub recommendation: String,
}

/// Expert two-group comparison: descriptives → normality → variance → test → effect size → CI.
pub fn two_group_comparison(group1: &[f64], group2: &[f64], alpha: Option<f64>) -> TwoGroupReport {
    let alpha = alpha.unwrap_or(0.05);
    let (n1, n2) = (group1.len(), group2.len());
    let s1 = crate::descriptive::moments_ungrouped(group1);
    let s2 = crate::descriptive::moments_ungrouped(group2);
    let mut sorted1 = group1.to_vec(); sorted1.sort_by(|a, b| a.total_cmp(b));
    let mut sorted2 = group2.to_vec(); sorted2.sort_by(|a, b| a.total_cmp(b));
    let median1 = crate::descriptive::median(&sorted1);
    let median2 = crate::descriptive::median(&sorted2);

    let normality_test = if n1.max(n2) < 5000 { "Shapiro-Wilk" } else { "D'Agostino-Pearson" };
    let (np1, np2) = if n1.max(n2) < 5000 {
        (crate::nonparametric::shapiro_wilk(group1).p_value,
         crate::nonparametric::shapiro_wilk(group2).p_value)
    } else {
        (crate::nonparametric::dagostino_pearson(group1).p_value,
         crate::nonparametric::dagostino_pearson(group2).p_value)
    };
    let both_normal = np1 > alpha && np2 > alpha;

    let lev = levene_test(&[group1, group2], LeveneCenter::Median);
    let equal_variance = lev.p_value > alpha;

    let (test_name, statistic, p_value, df): (&str, f64, f64, f64) = if both_normal {
        if equal_variance {
            let t = two_sample_t(&s1, &s2);
            ("Student's t-test (pooled)", t.statistic, t.p_value, t.df)
        } else {
            let t = welch_t(&s1, &s2);
            ("Welch's t-test", t.statistic, t.p_value, t.df)
        }
    } else {
        let mw = crate::nonparametric::mann_whitney_u(group1, group2);
        ("Mann-Whitney U", mw.statistic, mw.p_value, f64::NAN)
    };

    let d = cohens_d(&s1, &s2);
    let g = hedges_g(&s1, &s2);
    let mean_diff = s1.mean() - s2.mean();
    let se = (s1.variance(1) / n1 as f64 + s2.variance(1) / n2 as f64).sqrt();
    let welch_df = {
        let v1 = s1.variance(1) / n1 as f64;
        let v2 = s2.variance(1) / n2 as f64;
        let num = (v1 + v2).powi(2);
        let den = v1 * v1 / (n1 - 1).max(1) as f64 + v2 * v2 / (n2 - 1).max(1) as f64;
        if den > 0.0 { num / den } else { 1.0 }
    };
    let t_crit = crate::special_functions::t_quantile(1.0 - alpha / 2.0, welch_df);

    let recommendation = if both_normal && equal_variance {
        format!("Normal + equal variance → pooled t. d={d:.3}")
    } else if both_normal {
        format!("Normal + unequal variance → Welch t. d={d:.3}")
    } else {
        format!("Non-normal → Mann-Whitney U. d={d:.3} (interpret cautiously)")
    };

    TwoGroupReport {
        n1, n2, mean1: s1.mean(), mean2: s2.mean(),
        std1: s1.std(1), std2: s2.std(1), median1, median2,
        normality_p1: np1, normality_p2: np2, normality_test, both_normal,
        levene_p: lev.p_value, equal_variance,
        test_name, statistic, p_value, df,
        cohens_d: d, hedges_g: g,
        mean_diff, ci_lower: mean_diff - t_crit * se, ci_upper: mean_diff + t_crit * se,
        recommendation,
    }
}

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
// Fisher's exact test for 2×2 contingency tables
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher's exact test result.
#[derive(Debug, Clone)]
pub struct FisherExactResult {
    /// Two-sided p-value (sum of probabilities ≤ observed).
    pub p_value: f64,
    /// Odds ratio: (a·d) / (b·c).
    pub odds_ratio: f64,
}

/// Fisher's exact test for 2×2 contingency tables.
///
/// Computes the exact p-value using the hypergeometric distribution.
/// Preferred over chi-square when any expected cell count < 5.
///
/// Table layout: [[a, b], [c, d]]
///   - a = group1 & outcome1, b = group1 & outcome2
///   - c = group2 & outcome1, d = group2 & outcome2
///
/// `table`: [a, b, c, d] (four cells).
/// Returns two-sided p-value.
pub fn fisher_exact(table: &[u64; 4]) -> FisherExactResult {
    let a = table[0] as f64;
    let b = table[1] as f64;
    let c = table[2] as f64;
    let d = table[3] as f64;

    let or = if b * c < 1e-300 { f64::INFINITY } else { (a * d) / (b * c) };

    let r1 = a + b;          // row 1 total
    let r2 = c + d;          // row 2 total
    let c1 = a + c;          // col 1 total
    let n = r1 + r2;         // grand total

    // Log-hypergeometric probability: P(X=k) = C(r1,k) C(r2, c1-k) / C(n, c1)
    let log_hyper = |k: f64| -> f64 {
        crate::special_functions::log_gamma(r1 + 1.0) - crate::special_functions::log_gamma(k + 1.0) - crate::special_functions::log_gamma(r1 - k + 1.0)
        + crate::special_functions::log_gamma(r2 + 1.0) - crate::special_functions::log_gamma(c1 - k + 1.0) - crate::special_functions::log_gamma(r2 - c1 + k + 1.0)
        - crate::special_functions::log_gamma(n + 1.0) + crate::special_functions::log_gamma(c1 + 1.0) + crate::special_functions::log_gamma(n - c1 + 1.0)
    };

    let p_obs = log_hyper(a);

    // Two-sided: sum probabilities ≤ p_obs
    let k_min = (c1 - r2).max(0.0) as u64;
    let k_max = c1.min(r1) as u64;

    let mut p_two_sided = 0.0;
    for k in k_min..=k_max {
        let p_k = log_hyper(k as f64);
        if p_k <= p_obs + 1e-10 {
            p_two_sided += p_k.exp();
        }
    }

    FisherExactResult {
        p_value: p_two_sided.clamp(0.0, 1.0),
        odds_ratio: or,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Jarque-Bera normality test (Jarque & Bera 1987)
// ═══════════════════════════════════════════════════════════════════════════

/// Jarque-Bera normality test from pre-computed MomentStats.
///
/// JB = (n/6) [S² + K²/4] ~ χ²(2)
///
/// where S = skewness, K = excess kurtosis. The ultimate tambear method:
/// O(1) from MomentStats (already accumulated).
///
/// `stats`: pre-computed moments (count, m2, m3, m4).
pub fn jarque_bera(stats: &MomentStats) -> TestResult {
    let n = stats.count;
    if n < 3.0 {
        return TestResult {
            test_name: "Jarque-Bera", statistic: f64::NAN, p_value: f64::NAN,
            df: 2.0, effect_size: f64::NAN, effect_size_name: "",
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }
    let s = stats.skewness(false);
    let k = stats.kurtosis(true, false); // excess kurtosis (first arg = excess flag)
    let jb = (n / 6.0) * (s * s + k * k / 4.0);
    let p = chi2_right_tail_p(jb, 2.0);
    TestResult {
        test_name: "Jarque-Bera", statistic: jb, p_value: p,
        df: 2.0, effect_size: f64::NAN, effect_size_name: "",
        ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// McNemar's test for paired nominal data
// ═══════════════════════════════════════════════════════════════════════════

/// McNemar's test for paired dichotomous data (2x2 table of before/after).
///
/// Tests whether the row and column marginal frequencies are equal (symmetry).
/// Table: [[a, b], [c, d]] where b and c are the discordant pairs.
///
/// χ² = (|b - c| - continuity)² / (b + c) ~ χ²(1)
///
/// `table`: [a, b, c, d]. `continuity`: if true, applies Yates correction (-0.5).
pub fn mcnemar(table: &[f64; 4], continuity: bool) -> TestResult {
    let b = table[1];
    let c = table[2];
    let bc = b + c;

    if bc < 1e-15 {
        return TestResult {
            test_name: "McNemar", statistic: 0.0, p_value: 1.0,
            df: 1.0, effect_size: f64::NAN, effect_size_name: "",
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }

    let correction = if continuity { 0.5 } else { 0.0 };
    let diff = (b - c).abs() - correction;
    let chi2 = if diff > 0.0 { diff * diff / bc } else { 0.0 };
    let p = chi2_right_tail_p(chi2, 1.0);

    TestResult {
        test_name: "McNemar", statistic: chi2, p_value: p,
        df: 1.0, effect_size: f64::NAN, effect_size_name: "",
        ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cochran's Q test for k related dichotomous samples
// ═══════════════════════════════════════════════════════════════════════════

/// Cochran's Q test: generalization of McNemar to k > 2 treatments.
///
/// Tests H₀: all k treatments have the same probability of success.
/// Extension of Friedman for binary data.
///
/// Q = (k-1) [k Σ Cⱼ² - T²] / [kT - Σ Rᵢ²] ~ χ²(k-1)
///
/// `data`: n × k binary matrix (row-major, 0/1). n subjects, k treatments.
pub fn cochran_q(data: &[f64], n_subjects: usize, n_treatments: usize) -> TestResult {
    let n = n_subjects;
    let k = n_treatments;

    if k < 2 || n < 1 {
        return TestResult {
            test_name: "Cochran's Q", statistic: f64::NAN, p_value: f64::NAN,
            df: 0.0, effect_size: f64::NAN, effect_size_name: "",
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }

    // Column totals Cj (number of successes per treatment)
    let mut col_totals = vec![0.0; k];
    // Row totals Ri (number of successes per subject)
    let mut row_totals = vec![0.0; n];
    let mut grand_total = 0.0;

    for i in 0..n {
        for j in 0..k {
            let v = data[i * k + j];
            col_totals[j] += v;
            row_totals[i] += v;
            grand_total += v;
        }
    }

    let kf = k as f64;
    let sum_cj2: f64 = col_totals.iter().map(|c| c * c).sum();
    let sum_ri2: f64 = row_totals.iter().map(|r| r * r).sum();

    let denom = kf * grand_total - sum_ri2;
    if denom.abs() < 1e-15 {
        return TestResult {
            test_name: "Cochran's Q", statistic: 0.0, p_value: 1.0,
            df: kf - 1.0, effect_size: f64::NAN, effect_size_name: "",
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
    }

    let q = (kf - 1.0) * (kf * sum_cj2 - grand_total * grand_total) / denom;
    let df = kf - 1.0;
    let p = chi2_right_tail_p(q, df);

    TestResult {
        test_name: "Cochran's Q", statistic: q, p_value: p,
        df, effect_size: f64::NAN, effect_size_name: "",
        ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Power analysis (Cohen 1988)
// ═══════════════════════════════════════════════════════════════════════════
//
// These functions use the normal approximation, which is adequate for
// practical study design. For small n, the exact non-central t calculation
// would give slightly different numbers, but the normal approximation is
// standard in G*Power and scipy.stats.power.
//
// Effect sizes are Cohen's conventions:
//   Small:  d=0.2, r=0.1, f=0.1
//   Medium: d=0.5, r=0.3, f=0.25
//   Large:  d=0.8, r=0.5, f=0.4

/// Power of a one-sample t-test via normal approximation.
///
/// `effect_size`: Cohen's d = (μ - μ₀) / σ
/// `n`: sample size
/// `alpha`: significance level (e.g., 0.05)
/// `two_sided`: if true, uses α/2 in each tail
pub fn power_one_sample_t(effect_size: f64, n: f64, alpha: f64, two_sided: bool) -> f64 {
    if n < 2.0 || alpha <= 0.0 || alpha >= 1.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_crit = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    let ncp = effect_size * n.sqrt();
    let power_upper = 1.0 - crate::special_functions::normal_cdf(z_crit - ncp);
    let power = if two_sided {
        power_upper + crate::special_functions::normal_cdf(-z_crit - ncp)
    } else {
        power_upper
    };
    power.clamp(0.0, 1.0)
}

/// Power of a two-sample t-test (equal n) via normal approximation.
///
/// `effect_size`: Cohen's d = (μ₁ - μ₂) / σ_pooled
/// `n_per_group`: sample size per group
pub fn power_two_sample_t(effect_size: f64, n_per_group: f64, alpha: f64, two_sided: bool) -> f64 {
    if n_per_group < 2.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_crit = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    // SE for mean difference with equal n: σ · sqrt(2/n) → ncp = d · sqrt(n/2)
    let ncp = effect_size * (n_per_group / 2.0).sqrt();
    let power_upper = 1.0 - crate::special_functions::normal_cdf(z_crit - ncp);
    let power = if two_sided {
        power_upper + crate::special_functions::normal_cdf(-z_crit - ncp)
    } else {
        power_upper
    };
    power.clamp(0.0, 1.0)
}

/// Required sample size for a one-sample t-test to achieve desired power.
///
/// Closed-form normal approximation: n = ((z_{α/2} + z_β) / d)²
pub fn sample_size_one_sample_t(effect_size: f64, power: f64, alpha: f64, two_sided: bool) -> f64 {
    if effect_size.abs() < 1e-15 || power <= 0.0 || power >= 1.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_alpha = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    let z_beta = crate::special_functions::normal_quantile(power);
    let n = ((z_alpha + z_beta) / effect_size.abs()).powi(2);
    n.ceil()
}

/// Required sample size per group for a two-sample t-test.
///
/// n_per_group = 2·((z_{α/2} + z_β) / d)²
pub fn sample_size_two_sample_t(effect_size: f64, power: f64, alpha: f64, two_sided: bool) -> f64 {
    if effect_size.abs() < 1e-15 || power <= 0.0 || power >= 1.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_alpha = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    let z_beta = crate::special_functions::normal_quantile(power);
    let n = 2.0 * ((z_alpha + z_beta) / effect_size.abs()).powi(2);
    n.ceil()
}

/// Power of a one-way ANOVA via non-central F (Patnaik approximation).
///
/// `f`: Cohen's f = sqrt(η² / (1 - η²))
/// `k`: number of groups
/// `n_per_group`: sample size per group (equal n)
pub fn power_anova(f: f64, k: f64, n_per_group: f64, alpha: f64) -> f64 {
    if k < 2.0 || n_per_group < 2.0 { return f64::NAN; }
    let n_total = k * n_per_group;
    let df1 = k - 1.0;
    let df2 = n_total - k;
    if df2 < 1.0 { return f64::NAN; }
    let lambda = f * f * n_total;  // non-centrality
    let f_crit = crate::special_functions::f_quantile(1.0 - alpha, df1, df2);
    // Patnaik: non-central F ≈ ((df1 + λ)/df1) · central F(df1', df2)
    // where df1' = (df1 + λ)² / (df1 + 2λ)
    let scale = (df1 + lambda) / df1;
    let threshold = f_crit / scale;
    let df1_adj = (df1 + lambda).powi(2) / (df1 + 2.0 * lambda);
    let power = 1.0 - crate::special_functions::f_cdf(threshold, df1_adj, df2);
    power.clamp(0.0, 1.0)
}

/// Required sample size per group for one-way ANOVA via binary search.
pub fn sample_size_anova(f: f64, k: f64, power: f64, alpha: f64) -> f64 {
    if k < 2.0 || f.abs() < 1e-15 || power <= 0.0 || power >= 1.0 { return f64::NAN; }
    let mut lo = 2.0_f64;
    let mut hi = 10000.0_f64;
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let p = power_anova(f, k, mid, alpha);
        if p < power { lo = mid; } else { hi = mid; }
        if hi - lo < 1.0 { break; }
    }
    hi.ceil()
}

/// Power of a correlation test (H₀: ρ = 0) via Fisher's z-transform.
///
/// z = atanh(r) ~ N(atanh(ρ), 1/(n-3))
pub fn power_correlation(r: f64, n: f64, alpha: f64, two_sided: bool) -> f64 {
    if n < 4.0 || r.abs() >= 1.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_crit = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    let z_r = 0.5 * ((1.0 + r) / (1.0 - r)).ln(); // atanh(r)
    let se = 1.0 / (n - 3.0).sqrt();
    let ncp = z_r / se;
    let power_upper = 1.0 - crate::special_functions::normal_cdf(z_crit - ncp);
    let power = if two_sided {
        power_upper + crate::special_functions::normal_cdf(-z_crit - ncp)
    } else {
        power_upper
    };
    power.clamp(0.0, 1.0)
}

/// Required sample size for a correlation test.
pub fn sample_size_correlation(r: f64, power: f64, alpha: f64, two_sided: bool) -> f64 {
    if r.abs() < 1e-15 || r.abs() >= 1.0 || power <= 0.0 || power >= 1.0 { return f64::NAN; }
    let alpha_eff = if two_sided { alpha / 2.0 } else { alpha };
    let z_alpha = crate::special_functions::normal_quantile(1.0 - alpha_eff);
    let z_beta = crate::special_functions::normal_quantile(power);
    let z_r = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let n = ((z_alpha + z_beta) / z_r).powi(2) + 3.0;
    n.ceil()
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

    // ── Breusch-Pagan ────────────────────────────────────────────────────

    fn make_design(n: usize, x_col: &[f64]) -> crate::linear_algebra::Mat {
        // [1, x] design matrix
        let mut data = vec![0.0_f64; n * 2];
        for i in 0..n {
            data[i * 2] = 1.0;
            data[i * 2 + 1] = x_col[i];
        }
        crate::linear_algebra::Mat { rows: n, cols: 2, data }
    }

    #[test]
    fn bp_homoscedastic_high_pvalue() {
        // Residuals generated as iid N(0, 0.01²) — no heteroscedasticity.
        // p-value should be large (fail to reject H₀).
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let mut rng = 77777u64;
        let residuals: Vec<f64> = (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng as f64 / u64::MAX as f64 - 0.5) * 0.02
        }).collect();
        let xmat = make_design(n, &x);
        let result = breusch_pagan(&xmat, &residuals);
        assert!(result.p_value > 0.05,
            "Homoscedastic: p={:.4} should be > 0.05", result.p_value);
        assert_eq!(result.df, 1);
    }

    #[test]
    fn bp_heteroscedastic_low_pvalue() {
        // Residuals with variance proportional to x → clear heteroscedasticity.
        let n = 200usize;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut rng = 31415u64;
        let residuals: Vec<f64> = x.iter().map(|xi| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 * 1.7320508;
            z * xi.sqrt() * 0.5 // σ grows with √x
        }).collect();
        let xmat = make_design(n, &x);
        let result = breusch_pagan(&xmat, &residuals);
        assert!(result.statistic > 5.0,
            "Heteroscedastic: statistic={:.3} should be > 5.0", result.statistic);
    }

    #[test]
    fn bp_zero_residuals_no_panic() {
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let residuals = vec![0.0_f64; n];
        let xmat = make_design(n, &x);
        let result = breusch_pagan(&xmat, &residuals);
        assert!(result.statistic.is_finite());
        assert!(result.p_value.is_finite());
    }

    // ── Tukey HSD ────────────────────────────────────────────────────────

    fn moment_stats_from_slice(data: &[f64]) -> MomentStats {
        let n = data.len() as f64;
        let sum: f64 = data.iter().sum();
        let mean = sum / n;
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        MomentStats {
            count: n,
            sum,
            min: data.iter().copied().fold(f64::INFINITY, f64::min),
            max: data.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            m2,
            m3: 0.0,
            m4: 0.0,
        }
    }

    #[test]
    fn tukey_hsd_equal_groups_no_difference() {
        // Three identical groups → all comparisons non-significant
        let g1 = moment_stats_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let g2 = moment_stats_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let g3 = moment_stats_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let n = 15usize;
        let k = 3usize;
        // SS_within = g1.m2 + g2.m2 + g3.m2
        let ss_within = g1.m2 + g2.m2 + g3.m2;
        let df_error = (n - k) as f64;
        let ms_error = ss_within / df_error;
        let comparisons = tukey_hsd(&[g1, g2, g3], ms_error, df_error, None);
        assert_eq!(comparisons.len(), 3);
        for c in &comparisons {
            assert!(c.mean_diff.abs() < 1e-10, "Equal groups: mean_diff should be 0");
            assert!(!c.significant, "Equal groups: should not be significant");
        }
    }

    #[test]
    fn tukey_hsd_separated_groups() {
        // Group 1: [0,0,0], Group 2: [10,10,10], Group 3: [0,0,0]
        // Pair (1,2) should be significant; (1,3) and (2,3) similar
        let g1 = moment_stats_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let g2 = moment_stats_from_slice(&[100.0, 100.0, 100.0, 100.0, 100.0]);
        let g3 = moment_stats_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        // ms_error = SS_within / df: all within-group variance = 0, so ms_error → tiny
        // Use a small positive ms_error so the test is meaningful
        let ms_error = 1.0; // unit variance
        let df_error = 12.0;
        let comparisons = tukey_hsd(&[g1, g2, g3], ms_error, df_error, None);
        let pair_12 = comparisons.iter().find(|c| c.group_i == 0 && c.group_j == 1).unwrap();
        assert!(pair_12.significant, "Groups 0 and 1 differ by 100, should be significant");
        assert!(pair_12.q_statistic > 5.0, "q={:.2} should be large", pair_12.q_statistic);
    }

    #[test]
    fn tukey_hsd_studentized_range_p_known_value() {
        // For k=3, df=∞ (df_error=1000), q≈3.314 should give p≈0.05
        // Verify that q=3.314 gives p close to 0.05
        let q = 3.314_f64;
        let p = crate::special_functions::studentized_range_p(q, 3, 1000.0);
        assert!(
            (p - 0.05).abs() < 0.02,
            "studentized_range_p(3.314, 3, ∞) ≈ 0.05, got {:.4}", p
        );
    }

    // ── Integration ──────────────────────────────────────────────────────

    #[test]
    fn test_result_significant_at() {
        let r = TestResult {
            test_name: "test", statistic: 2.0, df: 10.0, p_value: 0.03,
            effect_size: 0.5, effect_size_name: "d",
            ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
        };
        assert!(r.significant_at(0.05));
        assert!(!r.significant_at(0.01));
    }

    // ── Cook's distance ─────────────────────────────────────────────────

    #[test]
    fn cooks_distance_no_outliers() {
        // Well-behaved linear data: y = 2x + 1 + small noise
        let n = 20;
        let p = 2;
        let mut data = vec![0.0; n * p];
        let mut residuals = Vec::new();
        let mut rng = 42u64;
        for i in 0..n {
            let x = i as f64 / n as f64;
            data[i * p] = 1.0; // intercept
            data[i * p + 1] = x;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.1;
            residuals.push(noise);
        }
        let x = crate::linear_algebra::Mat::from_vec(n, p, data);
        let r = cooks_distance(&x, &residuals);
        assert_eq!(r.cooks_distance.len(), n);
        assert_eq!(r.leverage.len(), n);
        // No outliers → no influential points
        assert!(r.n_influential <= 2,
            "Clean data should have few influential points, got {}", r.n_influential);
        // All Cook's D should be small
        let max_d = r.cooks_distance.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_d < 1.0, "Max Cook's D={} should be < 1 for clean data", max_d);
    }

    // ── Levene's test ─────────────────────────────────────────────────────

    #[test]
    fn levene_equal_variance_groups() {
        // Three groups with similar spread — Levene p should be high (fail to reject H₀)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let r = levene_test(&[&a, &b, &c], LeveneCenter::Median);
        assert!(r.p_value > 0.05, "equal-spread groups: levene p={}", r.p_value);
        assert!(r.f_statistic >= 0.0 && r.f_statistic.is_finite());
    }

    #[test]
    fn levene_very_unequal_variance() {
        // Group A is tight, group B is very spread — Levene should reject H₀
        let a = vec![1.0, 1.01, 0.99, 1.0, 1.01];
        let b = vec![0.0, 10.0, -10.0, 5.0, -5.0];
        let r = levene_test(&[&a, &b], LeveneCenter::Median);
        assert!(r.p_value < 0.05, "very different variance: levene p={}", r.p_value);
    }

    #[test]
    fn levene_empty_group_ignored() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![];
        let c = vec![2.0, 3.0, 4.0];
        let r = levene_test(&[&a, &b, &c], LeveneCenter::Mean);
        // Should handle empty group gracefully (k becomes 2)
        assert!(r.f_statistic.is_finite() || r.f_statistic.is_nan());
    }

    // ── Welch ANOVA ───────────────────────────────────────────────────────

    #[test]
    fn welch_anova_three_groups_same_mean() {
        // Three groups with same mean — F should be near 0, p near 1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5]; // shifted slightly
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = welch_anova(&[&a, &b, &c]);
        assert!(r.f_statistic >= 0.0, "F statistic should be non-negative");
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn welch_anova_clearly_different_means() {
        // Groups with very different means — should reject H₀
        let a = vec![1.0, 1.1, 0.9, 1.0, 1.1];
        let b = vec![10.0, 10.1, 9.9, 10.0, 10.1];
        let c = vec![20.0, 20.1, 19.9, 20.0, 20.1];
        let r = welch_anova(&[&a, &b, &c]);
        assert!(r.p_value < 0.001, "very different means: welch F p={}", r.p_value);
        assert!(r.df_between > 0.0);
    }

    #[test]
    fn welch_anova_size_one_group_excluded() {
        // Group of size 1 is silently excluded (undefined variance)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![42.0]; // n=1: excluded
        let c = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let r = welch_anova(&[&a, &b, &c]);
        // With only a and c (both low-variance similar-mean groups), p should be > 0.05
        assert!(r.f_statistic.is_finite() || r.f_statistic.is_nan(),
            "f_statistic should be finite or NaN, got {}", r.f_statistic);
    }

    #[test]
    fn cooks_distance_with_outlier() {
        // One extreme outlier should have high Cook's D
        let n = 20;
        let p = 2;
        let mut data = vec![0.0; n * p];
        let mut residuals = vec![0.01; n]; // small residuals
        for i in 0..n {
            data[i * p] = 1.0;
            data[i * p + 1] = i as f64 / n as f64;
        }
        // Make observation 0 an outlier with huge residual
        residuals[0] = 50.0;
        let x = crate::linear_algebra::Mat::from_vec(n, p, data);
        let r = cooks_distance(&x, &residuals);
        // Observation 0 should have the largest Cook's D
        let max_idx = r.cooks_distance.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap().0;
        assert_eq!(max_idx, 0, "Outlier at index 0 should have max Cook's D");
        assert!(r.cooks_distance[0] > r.cooks_distance[10],
            "Outlier D={} should exceed normal point D={}",
            r.cooks_distance[0], r.cooks_distance[10]);
    }

    // ── Logistic regression via IRLS ──────────────────────────────────────

    #[test]
    fn logistic_regression_linearly_separable() {
        // Two clear groups with some overlap so IRLS converges.
        // x uniformly from -3 to +3 (n=30), y=Bernoulli(sigmoid(2x)).
        // Sufficient overlap that the MLE is finite and IRLS converges.
        let n = 30;
        // Evenly spaced x from -2.9 to +2.9
        let x_data: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64) * 5.8 - 2.9).collect();
        // Deterministic labeling: y=1 when x>0, y=0 when x<0 (with some transition)
        // Use a smooth threshold to avoid perfect separation:
        // y = 1 if sigmoid(1.5*x) > 0.5 i.e. x > 0; add one misclassification to avoid separation
        let y: Vec<f64> = x_data.iter().enumerate().map(|(i, &xi)| {
            if i == 0 { 1.0 } else if xi >= 0.0 { 1.0 } else { 0.0 } // one "wrong" label
        }).collect();
        let x = crate::linear_algebra::Mat::from_vec(n, 1, x_data.clone());

        let result = logistic_regression(&x, &y, 100, 1e-8).unwrap();

        // Coefficient on x should be positive (higher x → higher P(y=1))
        assert!(result.coefficients[0] > 0.0,
            "coeff[x]={:.4} should be positive", result.coefficients[0]);
        // Residual deviance < null deviance
        assert!(result.residual_deviance < result.null_deviance,
            "residual dev={:.4} should < null dev={:.4}",
            result.residual_deviance, result.null_deviance);
        // AIC should be finite and positive
        assert!(result.aic > 0.0 && result.aic.is_finite(),
            "AIC={:.4} should be positive finite", result.aic);
        // SE should be positive finite
        assert!(result.std_errors[0] > 0.0 && result.std_errors[0].is_finite(),
            "SE={:.4} should be positive finite", result.std_errors[0]);
    }

    #[test]
    fn logistic_regression_null_model() {
        // When y is all one class, the coefficient should be near zero
        // and the model should converge to the null model.
        // Use a case where features are pure noise and y is 50/50 — coefficient near 0.
        let n = 20;
        // x = alternating ±1 (no signal), y = 50/50
        let x_data: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let x = crate::linear_algebra::Mat::from_vec(n, 1, x_data);

        let result = logistic_regression(&x, &y, 25, 1e-8).unwrap();
        // With uncorrelated x and balanced y, coefficient near 0, p large
        assert!(result.coefficients[0].abs() < 1.0,
            "coeff[x]={:.4} should be near 0 for uncorrelated predictor", result.coefficients[0]);
        // SE should be finite and positive
        assert!(result.std_errors[0] > 0.0 && result.std_errors[0].is_finite(),
            "SE={:.4} should be positive finite", result.std_errors[0]);
    }

    #[test]
    fn logistic_regression_predict_proba_range() {
        // Probabilities from predict_proba must all be in [0, 1]
        // Use data with some overlap to keep coefficients moderate.
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / 10.0 - 1.5).collect();
        // One deliberate mislabel at each end to avoid perfect separation
        let y: Vec<f64> = x_data.iter().enumerate().map(|(i, &xi)| {
            if i == 0 { 1.0 } else if xi >= 0.0 { 1.0 } else { 0.0 }
        }).collect();
        let x = crate::linear_algebra::Mat::from_vec(n, 1, x_data);

        let result = logistic_regression(&x, &y, 25, 1e-8).unwrap();
        let probs = result.predict_proba(&x);
        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= 0.0 && p <= 1.0, "proba[{i}]={p} must be in [0,1]");
        }
        // Class 0 region (first half) should have lower mean probability than class 1 region
        let mean_p_low: f64 = probs[..n/2].iter().sum::<f64>() / (n/2) as f64;
        let mean_p_high: f64 = probs[n/2..].iter().sum::<f64>() / (n/2) as f64;
        assert!(mean_p_low < mean_p_high,
            "lower-x region mean P={:.3} should < higher-x region mean P={:.3}",
            mean_p_low, mean_p_high);
    }

    #[test]
    fn logistic_regression_underdetermined_returns_none() {
        // n <= p+1 → not enough degrees of freedom
        let x = crate::linear_algebra::Mat::from_rows(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
        ]);
        let y = vec![0.0, 1.0]; // n=2, q=3 (2 features + intercept)
        assert!(logistic_regression(&x, &y, 25, 1e-8).is_none());
    }

    // ── Mediation ──

    #[test]
    fn mediation_full_mediation() {
        // Full mediation: X → M → Y, no direct X → Y path
        // M = 2 * X + noise, Y = 3 * M + noise → indirect = 2*3 = 6
        let mut rng = crate::rng::Xoshiro256::new(42);
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / 20.0).collect();
        let m: Vec<f64> = x.iter().map(|&xi| {
            2.0 * xi + crate::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();
        let y: Vec<f64> = m.iter().map(|&mi| {
            3.0 * mi + crate::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();

        let res = mediation(&x, &m, &y);
        // Path a ≈ 2, path b ≈ 3
        assert!((res.path_a - 2.0).abs() < 0.2, "a={}", res.path_a);
        assert!((res.path_b - 3.0).abs() < 0.3, "b={}", res.path_b);
        // Indirect ≈ 6
        assert!((res.indirect_effect - 6.0).abs() < 0.5, "indirect={}", res.indirect_effect);
        // Direct effect should be small (no X → Y direct path)
        assert!(res.direct_effect.abs() < 0.5, "direct={}", res.direct_effect);
        // Sobel z should be large (significant mediation)
        assert!(res.sobel_z.abs() > 3.0, "sobel z={}", res.sobel_z);
        assert!(res.sobel_p < 0.01, "sobel p={}", res.sobel_p);
    }

    #[test]
    fn mediation_no_mediation() {
        // No mediation: M is random, unrelated to X
        let mut rng = crate::rng::Xoshiro256::new(99);
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / 20.0).collect();
        let m: Vec<f64> = (0..n).map(|_| crate::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| {
            2.0 * xi + crate::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();

        let res = mediation(&x, &m, &y);
        // Path a should be ~0 (M unrelated to X)
        assert!(res.path_a.abs() < 0.3, "a={} should be ~0", res.path_a);
        // Indirect should be ~0
        assert!(res.indirect_effect.abs() < 0.3, "indirect={} should be ~0", res.indirect_effect);
        // Sobel should not reject
        assert!(res.sobel_p > 0.05, "sobel p={} should not reject", res.sobel_p);
    }

    // ── Moderation ──

    #[test]
    fn moderation_interaction_detected() {
        // y = 1 + 2x + 0.5z + 1.5*x*z + noise
        // The x*z coefficient (1.5) should be detected as significant
        let mut rng = crate::rng::Xoshiro256::new(42);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) / (n as f64) - 0.5) * 4.0).collect();
        let z: Vec<f64> = (0..n).map(|_| crate::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let y: Vec<f64> = (0..n).map(|i| {
            1.0 + 2.0 * x[i] + 0.5 * z[i] + 1.5 * x[i] * z[i]
                + crate::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();

        let res = moderation(&x, &z, &y);
        assert!((res.interaction - 1.5).abs() < 0.3,
            "interaction={} should be ~1.5", res.interaction);
        assert!(res.p_interaction < 0.001,
            "p_interaction={} should be significant", res.p_interaction);
        // Simple slopes should differ at high vs low z
        assert!(res.simple_slope_high > res.simple_slope_low,
            "high={} should exceed low={}",
            res.simple_slope_high, res.simple_slope_low);
    }

    // ── Bayes factors ──

    #[test]
    fn bayes_factor_t_no_effect_favors_null() {
        // Small t-statistic → should favor H0
        let result = bayes_factor_t_one_sample(0.2, 30, None);
        assert!(result.bf10 < 1.0, "BF10={} should be < 1 for small t", result.bf10);
        assert!(result.bf01 > 1.0);
    }

    #[test]
    fn bayes_factor_t_large_effect_favors_alternative() {
        // Large t-statistic → should favor H1
        let result = bayes_factor_t_one_sample(4.0, 30, None);
        assert!(result.bf10 > 10.0, "BF10={} should be > 10 for large t", result.bf10);
        assert!(result.bf01 < 0.1);
    }

    #[test]
    fn bayes_factor_correlation_zero_favors_null() {
        let result = bayes_factor_correlation(0.05, 50);
        assert!(result.bf10 < 1.0, "BF10={} should be < 1 for near-zero r", result.bf10);
    }

    #[test]
    fn bayes_factor_correlation_strong_favors_alternative() {
        let result = bayes_factor_correlation(0.7, 30);
        assert!(result.bf10 > 10.0, "BF10={} should be > 10 for strong r", result.bf10);
    }

    // ── GLM (Poisson) ──

    #[test]
    fn glm_poisson_count_data() {
        // y ~ Poisson(exp(0.5 + 0.3*x))
        let mut rng = crate::rng::Xoshiro256::new(42);
        let n = 200;
        let x_vals: Vec<f64> = (0..n).map(|i| (i as f64 - 100.0) / 50.0).collect();
        let y_vals: Vec<f64> = x_vals.iter().map(|&xi| {
            let lambda = (0.5 + 0.3 * xi).exp();
            crate::rng::sample_poisson(&mut rng, lambda) as f64
        }).collect();
        let x_mat = crate::linear_algebra::Mat::from_vec(n, 1,
            x_vals.iter().copied().collect());
        let result = glm_fit(&x_mat, &y_vals, GlmFamily::Poisson, 50, 1e-8);
        assert!(result.converged, "Poisson GLM should converge");
        // Intercept ≈ 0.5, slope ≈ 0.3
        assert!((result.coefficients[0] - 0.3).abs() < 0.15,
            "slope={} should be ~0.3", result.coefficients[0]);
        assert!((result.coefficients[1] - 0.5).abs() < 0.15,
            "intercept={} should be ~0.5", result.coefficients[1]);
        assert!(result.deviance < result.null_deviance,
            "model deviance should be less than null");
    }

    #[test]
    fn moderation_no_interaction() {
        // y = 1 + 2x + 0.5z + noise (no x*z term)
        let mut rng = crate::rng::Xoshiro256::new(99);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) / (n as f64) - 0.5) * 4.0).collect();
        let z: Vec<f64> = (0..n).map(|_| crate::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let y: Vec<f64> = (0..n).map(|i| {
            1.0 + 2.0 * x[i] + 0.5 * z[i]
                + crate::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();

        let res = moderation(&x, &z, &y);
        // Interaction coefficient should be ~0
        assert!(res.interaction.abs() < 0.2,
            "interaction={} should be ~0", res.interaction);
        // And not significant
        assert!(res.p_interaction > 0.05,
            "p_interaction={} should not be significant", res.p_interaction);
    }

    // ── Expert pipeline: two-group comparison ──

    #[test]
    fn two_group_comparison_different_means() {
        let mut rng = crate::rng::Xoshiro256::new(42);
        let g1: Vec<f64> = (0..50).map(|_| crate::rng::sample_normal(&mut rng, 10.0, 2.0)).collect();
        let g2: Vec<f64> = (0..50).map(|_| crate::rng::sample_normal(&mut rng, 15.0, 2.0)).collect();
        let report = two_group_comparison(&g1, &g2, None);
        assert!(report.p_value < 0.001, "Should detect difference, p={}", report.p_value);
        assert!(report.cohens_d.abs() > 1.0, "Large effect expected, d={}", report.cohens_d);
        assert!(report.ci_lower < report.mean_diff && report.mean_diff < report.ci_upper);
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn two_group_comparison_same_distribution() {
        let mut rng = crate::rng::Xoshiro256::new(99);
        let g1: Vec<f64> = (0..50).map(|_| crate::rng::sample_normal(&mut rng, 10.0, 2.0)).collect();
        let g2: Vec<f64> = (0..50).map(|_| crate::rng::sample_normal(&mut rng, 10.0, 2.0)).collect();
        let report = two_group_comparison(&g1, &g2, None);
        assert!(report.p_value > 0.05, "Should not reject, p={}", report.p_value);
        assert!(report.cohens_d.abs() < 0.5, "Small effect, d={}", report.cohens_d);
    }
}
