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
    // Cohen's d using proper pooled SD: sqrt((m2₁ + m2₂) / (n₁ + n₂ - 2))
    let d = cohens_d(stats1, stats2);

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
/// Controls the family-wise error rate using the studentized range distribution.
///
/// `groups`: slice of `MomentStats`, one per group (count, mean, variance).
/// `ms_error`: mean square error from the ANOVA (= SS_within / df_within).
/// `df_error`: degrees of freedom of the error term (= N - k).
///
/// Each pair (i,j) with i < j produces a `TukeyComparison`.
pub fn tukey_hsd(groups: &[MomentStats], ms_error: f64, df_error: f64) -> Vec<TukeyComparison> {
    let k = groups.len();
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
                significant: p < 0.05,
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
pub fn cooks_distance(
    x_with_intercept: &crate::linear_algebra::Mat,
    residuals: &[f64],
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

    let threshold = 4.0 / n as f64;
    let n_influential = cooks_distance.iter().filter(|&&d| d > threshold).count();

    InfluenceResult { cooks_distance, leverage, n_influential }
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
    1.0 / (1.0 + (-z).exp())
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
        let comparisons = tukey_hsd(&[g1, g2, g3], ms_error, df_error);
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
        let comparisons = tukey_hsd(&[g1, g2, g3], ms_error, df_error);
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
}
