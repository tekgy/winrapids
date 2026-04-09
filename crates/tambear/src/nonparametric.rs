//! Non-parametric statistics — rank-based methods, bootstrap, KDE.
//!
//! ## Architecture
//!
//! Non-parametric tests make NO distributional assumptions. Instead of
//! consuming MomentStats (which assume numeric moments exist), they work
//! on **ranks** — computed once via `rank()`, then reused across tests.
//!
//! The key insight: most non-parametric tests are **scatter operations on ranks**.
//! Mann-Whitney U = sum of ranks in one group. Kruskal-Wallis = between-group
//! variance of rank means. Rank once, scatter many times.
//!
//! ## Tests implemented
//!
//! **Rank-based**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman
//! **Goodness of fit**: Kolmogorov-Smirnov (one-sample, two-sample)
//! **Normality**: Shapiro-Wilk (small n approximation)
//! **Resampling**: Bootstrap (percentile, BCa), permutation test
//! **Density estimation**: Kernel density estimation (Gaussian, Epanechnikov)
//! **Correlation**: Spearman rank, Kendall's tau
//!
//! ## .tbs integration
//!
//! ```text
//! mann_whitney(x, y)         # U-test
//! ks_test(data, "normal")    # one-sample KS
//! bootstrap(data, mean, n=10000)  # bootstrap CI
//! kde(data, bw="silverman") # kernel density
//! ```

use crate::special_functions::{normal_cdf, normal_two_tail_p, chi2_right_tail_p};

// ═══════════════════════════════════════════════════════════════════════════
// Ranking
// ═══════════════════════════════════════════════════════════════════════════

/// Compute ranks of values (1-based, average ties).
///
/// This is the foundational operation. Rank once, use everywhere.
/// NaN values get rank NaN.
pub fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Sort by value, NaN goes last
    indexed.sort_by(|a, b| {
        match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => a.1.total_cmp(&b.1),
        }
    });

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        if indexed[i].1.is_nan() {
            // All remaining are NaN
            for j in i..n {
                ranks[indexed[j].0] = f64::NAN;
            }
            break;
        }
        // Find tie group
        let mut j = i + 1;
        while j < n && !indexed[j].1.is_nan() && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for tie group: (i+1 + j) / 2 (1-based)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation coefficient.
///
/// r_s = Pearson correlation of ranks.
/// For no ties: r_s = 1 - 6Σd²/(n(n²-1))
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = rank(x);
    let ry = rank(y);
    pearson_on_ranks(&rx, &ry)
}

/// Kendall's tau-b (handles ties).
///
/// tau_b = (C - D) / √((C+D+T_x)(C+D+T_y))
/// where C = concordant pairs, D = discordant pairs, T_x/T_y = ties in x/y only
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 { return f64::NAN; }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_x: i64 = 0;
    let mut ties_y: i64 = 0;

    for i in 0..n {
        for j in (i+1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let product = dx * dy;

            if dx == 0.0 && dy == 0.0 {
                // Joint tie — doesn't count
            } else if dx == 0.0 {
                ties_x += 1;
            } else if dy == 0.0 {
                ties_y += 1;
            } else if product > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let denom_x = (concordant + discordant + ties_x) as f64;
    let denom_y = (concordant + discordant + ties_y) as f64;
    let denom = (denom_x * denom_y).sqrt();
    if denom == 0.0 { return f64::NAN; }
    (concordant - discordant) as f64 / denom
}

/// Pearson correlation on already-ranked data (used by Spearman).
fn pearson_on_ranks(rx: &[f64], ry: &[f64]) -> f64 {
    let n = rx.len() as f64;
    if n < 2.0 { return f64::NAN; }
    let mx: f64 = rx.iter().sum::<f64>() / n;
    let my: f64 = ry.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..rx.len() {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 { f64::NAN } else { sxy / denom }
}

// ═══════════════════════════════════════════════════════════════════════════
// Partial correlation
// ═══════════════════════════════════════════════════════════════════════════

/// Partial correlation of X and Y controlling for one or more covariates Z.
///
/// For a single covariate: `r_XY.Z = (r_XY - r_XZ · r_YZ) / √((1-r_XZ²)(1-r_YZ²))`
///
/// For multiple covariates, uses recursive semi-partial correlation removal
/// (Wherry 1931): strip each covariate from X and Y via the partial formula
/// applied sequentially. Each application removes one covariate from the
/// residual correlation.
///
/// All correlations are Pearson r computed from raw data.
/// Returns the partial correlation coefficient in [-1, 1].
/// Returns `NaN` if any inputs have length < 2 or if computation is degenerate.
pub fn partial_correlation(x: &[f64], y: &[f64], covariates: &[&[f64]]) -> f64 {
    assert_eq!(x.len(), y.len());
    for z in covariates { assert_eq!(z.len(), x.len()); }

    if x.len() < 2 { return f64::NAN; }

    if covariates.is_empty() {
        return pearson_r(x, y);
    }

    // Recursive partial correlation: strip covariates one at a time
    // Start with Pearson r for all pairs, then apply the 3-variable formula recursively
    // For each covariate z_k:
    //   r_XY.{Z} = (r_XY.{Z-1} - r_XZ_k.{Z-1} · r_YZ_k.{Z-1}) /
    //              √((1-r_XZ_k.{Z-1}²)(1-r_YZ_k.{Z-1}²))
    //
    // Base: zero covariates → Pearson r
    // This requires tracking partial correlations for all pairs (X,Y), (X,Zk), (Y,Zk)
    // against each other for every Z removed.
    //
    // For simplicity and correctness: regress out each covariate sequentially
    // by computing residuals, then compute Pearson r on residuals.
    let n = x.len();

    // Compute residuals of X and Y with all covariates regressed out
    // Using the FWL theorem: partial_corr(X,Y | Z) = corr(resid_X, resid_Y)
    // where resid_X = X - Z·(Z'Z)^{-1}Z'X
    // For single covariate case, this simplifies to:
    // resid_X = X - (cov(X,Z)/var(Z)) * Z
    let mut rx: Vec<f64> = x.to_vec();
    let mut ry: Vec<f64> = y.to_vec();

    for &z in covariates {
        // Regress rx on z: rx -= (cov(rx, z) / var(z)) * z
        let beta_xz = ols_slope(&rx, z);
        let beta_yz = ols_slope(&ry, z);
        for i in 0..n {
            rx[i] -= beta_xz * z[i];
            ry[i] -= beta_yz * z[i];
        }
    }

    pearson_r(&rx, &ry)
}

/// OLS slope of y on x: β = cov(y,x) / var(x)
fn ols_slope(y: &[f64], x: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();
    let sxx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum();
    if sxx < 1e-300 { 0.0 } else { sxy / sxx }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mann-Whitney U test
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a non-parametric test.
#[derive(Debug, Clone)]
pub struct NonparametricResult {
    pub test_name: &'static str,
    pub statistic: f64,
    pub p_value: f64,
}

/// Mann-Whitney U test (Wilcoxon rank-sum test).
///
/// Tests whether two independent samples come from the same distribution.
/// H₀: P(X > Y) = P(Y > X) (stochastic equality)
///
/// Computes U₁ = R₁ - n₁(n₁+1)/2 where R₁ = sum of ranks in sample 1.
/// Uses normal approximation for p-value (valid for n₁,n₂ ≥ 8).
pub fn mann_whitney_u(x: &[f64], y: &[f64]) -> NonparametricResult {
    let n1 = x.len();
    let n2 = y.len();
    if n1 == 0 || n2 == 0 {
        return NonparametricResult {
            test_name: "Mann-Whitney U", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    // Combine and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n1 + n2);
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);
    let ranks = rank(&combined);

    // Sum of ranks for group 1
    let r1: f64 = ranks[..n1].iter().sum();
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2); // Use smaller U for two-tailed test

    // Normal approximation with tie correction
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let nf = n1f + n2f;
    let mu = n1f * n2f / 2.0;

    // Tie correction: σ² = n1·n2/12 × [(N+1) - Σ(tᵢ³-tᵢ)/(N(N-1))]
    let mut sorted_combined = combined.clone();
    sorted_combined.sort_by(|a, b| a.total_cmp(b));
    let mut tie_sum = 0.0;
    let mut ti = 0;
    while ti < sorted_combined.len() {
        let mut t = 1usize;
        while ti + t < sorted_combined.len() && sorted_combined[ti + t] == sorted_combined[ti] {
            t += 1;
        }
        if t > 1 {
            let tf = t as f64;
            tie_sum += tf * tf * tf - tf;
        }
        ti += t;
    }
    let sigma = (n1f * n2f / 12.0 * ((nf + 1.0) - tie_sum / (nf * (nf - 1.0)))).sqrt();

    let z = if sigma > 0.0 { (u - mu) / sigma } else { 0.0 };
    let p = normal_two_tail_p(z);

    NonparametricResult {
        test_name: "Mann-Whitney U", statistic: u, p_value: p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wilcoxon signed-rank test
// ═══════════════════════════════════════════════════════════════════════════

/// Wilcoxon signed-rank test for paired samples or one-sample symmetry.
///
/// H₀: median of differences = 0 (or data is symmetric around 0)
///
/// 1. Compute |dᵢ|, discard zeros
/// 2. Rank the |dᵢ|
/// 3. W⁺ = sum of ranks where dᵢ > 0
/// 4. Normal approximation for large n
pub fn wilcoxon_signed_rank(differences: &[f64]) -> NonparametricResult {
    // Filter zeros and NaN
    let nonzero: Vec<f64> = differences.iter()
        .copied()
        .filter(|&d| d != 0.0 && !d.is_nan())
        .collect();
    let n = nonzero.len();

    if n == 0 {
        return NonparametricResult {
            test_name: "Wilcoxon signed-rank", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let abs_vals: Vec<f64> = nonzero.iter().map(|d| d.abs()).collect();
    let ranks = rank(&abs_vals);

    // W+ = sum of ranks where original difference was positive
    let w_plus: f64 = nonzero.iter().zip(ranks.iter())
        .filter(|(&d, _)| d > 0.0)
        .map(|(_, &r)| r)
        .sum();

    // Tie correction: subtract Σ t_j(t_j-1)(t_j+1)/48 from variance
    let mut sorted_abs = abs_vals.clone();
    sorted_abs.sort_by(|a, b| a.total_cmp(b));
    let mut tie_correction = 0.0;
    let mut ti = 0;
    while ti < sorted_abs.len() {
        let mut t = 1usize;
        while ti + t < sorted_abs.len() && sorted_abs[ti + t] == sorted_abs[ti] {
            t += 1;
        }
        if t > 1 {
            let tf = t as f64;
            tie_correction += tf * (tf - 1.0) * (tf + 1.0);
        }
        ti += t;
    }

    let nf = n as f64;
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_correction / 48.0).sqrt();

    let z = if sigma > 0.0 { (w_plus - mu) / sigma } else { 0.0 };
    let p = normal_two_tail_p(z);

    NonparametricResult {
        test_name: "Wilcoxon signed-rank", statistic: w_plus, p_value: p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kruskal-Wallis H test
// ═══════════════════════════════════════════════════════════════════════════

/// Kruskal-Wallis H test — non-parametric one-way ANOVA.
///
/// H₀: all k populations have the same distribution
///
/// H = (12/(N(N+1))) Σ (R²ᵢ/nᵢ) - 3(N+1)
///
/// `group_sizes`: number of elements in each group.
/// `data`: all data concatenated in group order.
pub fn kruskal_wallis(data: &[f64], group_sizes: &[usize]) -> NonparametricResult {
    let n: usize = group_sizes.iter().sum();
    assert_eq!(data.len(), n, "data length must match sum of group sizes");
    let k = group_sizes.len();

    if k < 2 || n < 3 {
        return NonparametricResult {
            test_name: "Kruskal-Wallis H", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let ranks = rank(data);
    let nf = n as f64;

    // Sum of ranks per group
    let mut offset = 0;
    let mut h = 0.0;
    for &gs in group_sizes {
        if gs > 0 {
            let r_sum: f64 = ranks[offset..offset + gs].iter().sum();
            h += r_sum * r_sum / gs as f64;
        }
        offset += gs;
    }
    h = (12.0 / (nf * (nf + 1.0))) * h - 3.0 * (nf + 1.0);

    let df = (k - 1) as f64;
    let p = chi2_right_tail_p(h, df);

    NonparametricResult {
        test_name: "Kruskal-Wallis H", statistic: h, p_value: p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dunn's post-hoc test
// ═══════════════════════════════════════════════════════════════════════════

/// Result for one pairwise comparison in Dunn's test.
#[derive(Debug, Clone)]
pub struct DunnComparison {
    /// Index of group i
    pub group_i: usize,
    /// Index of group j
    pub group_j: usize,
    /// z-statistic for the mean rank difference
    pub z_statistic: f64,
    /// Two-sided p-value (unadjusted)
    pub p_value: f64,
}

/// Dunn's (1964) post-hoc test for non-parametric multiple comparisons.
///
/// Use after a significant Kruskal-Wallis test to identify which pairs differ.
/// Tests H₀: groups i and j have the same distribution using rank-sum differences.
///
/// z_{ij} = (R̄_i - R̄_j) / SE_{ij}
///
/// where SE_{ij} = √[ N(N+1)/12 · (1/n_i + 1/n_j) ]
/// corrected for ties: subtract Σ (t³-t)/(12(N-1)) per tied group of size t.
///
/// `data`: all observations concatenated.
/// `group_sizes`: number of observations per group.
///
/// Returns unadjusted p-values. Apply Bonferroni/BH via `bonferroni()` or
/// `benjamini_hochberg()` from `hypothesis.rs` for family-wise control.
pub fn dunn_test(data: &[f64], group_sizes: &[usize]) -> Vec<DunnComparison> {
    let n: usize = group_sizes.iter().sum();
    assert_eq!(data.len(), n, "data length must match sum of group sizes");
    let k = group_sizes.len();
    let nf = n as f64;

    // Global ranks (average ranks for ties)
    let ranks = rank(data);

    // Tie correction: C = Σ (t³ - t) / (12 * (N-1))
    // where the sum is over tied groups of size t
    let mut sorted_data: Vec<f64> = data.to_vec();
    sorted_data.sort_by(|a, b| a.total_cmp(b));
    let mut tie_correction = 0.0;
    let mut i = 0;
    while i < n {
        let val = sorted_data[i];
        let mut j = i + 1;
        while j < n && sorted_data[j] == val { j += 1; }
        let t = (j - i) as f64;
        if t > 1.0 { tie_correction += t * t * t - t; }
        i = j;
    }
    tie_correction /= 12.0 * (nf - 1.0);

    // Mean rank per group
    let mut mean_ranks = Vec::with_capacity(k);
    let mut offset = 0;
    for &gs in group_sizes {
        if gs == 0 {
            mean_ranks.push(f64::NAN);
        } else {
            let r_sum: f64 = ranks[offset..offset + gs].iter().sum();
            mean_ranks.push(r_sum / gs as f64);
        }
        offset += gs;
    }

    // All pairwise comparisons
    let mut results = Vec::new();
    for gi in 0..k {
        for gj in (gi + 1)..k {
            let ni = group_sizes[gi] as f64;
            let nj = group_sizes[gj] as f64;
            if ni < 1.0 || nj < 1.0 || mean_ranks[gi].is_nan() || mean_ranks[gj].is_nan() {
                continue;
            }

            let base_var = nf * (nf + 1.0) / 12.0 - tie_correction;
            let se = (base_var * (1.0 / ni + 1.0 / nj)).sqrt();
            let z = if se < 1e-300 { 0.0 } else { (mean_ranks[gi] - mean_ranks[gj]) / se };
            let p = normal_two_tail_p(z);

            results.push(DunnComparison { group_i: gi, group_j: gj, z_statistic: z, p_value: p });
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Kolmogorov-Smirnov test
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample Kolmogorov-Smirnov test against standard normal N(0,1).
///
/// D = sup|Fₙ(x) - Φ(x)| where Φ is the standard normal CDF.
/// p-value via Kolmogorov distribution approximation.
///
/// **Note**: This tests against N(0,1) specifically. For general normality
/// testing (any mean and variance), use [`ks_test_normal_standardized`].
pub fn ks_test_normal(data: &[f64]) -> NonparametricResult {
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    if n == 0 {
        return NonparametricResult {
            test_name: "KS test (normal)", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let nf = n as f64;
    let mut d_max = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let cdf = normal_cdf(x);
        let d1 = ((i + 1) as f64 / nf - cdf).abs();
        let d2 = (cdf - i as f64 / nf).abs();
        d_max = d_max.max(d1).max(d2);
    }

    let p = ks_p_value(d_max, n);

    NonparametricResult {
        test_name: "KS test (normal)", statistic: d_max, p_value: p,
    }
}

/// One-sample Kolmogorov-Smirnov test for normality (any mean and variance).
///
/// Standardizes data to z = (x - x̄) / s, then tests against N(0,1).
/// The D statistic is valid; however, the asymptotic Kolmogorov p-value
/// is conservative (actual significance level < nominal) because μ and σ
/// are estimated from data. For exact small-sample inference, use
/// Lilliefors critical values (not yet implemented).
pub fn ks_test_normal_standardized(data: &[f64]) -> NonparametricResult {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean.len();

    if n < 2 {
        return NonparametricResult {
            test_name: "KS test (normality)", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let nf = n as f64;
    let mean = clean.iter().sum::<f64>() / nf;
    let var = clean.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    let std = var.sqrt();

    if std < 1e-15 {
        // All values identical — trivially "normal" but degenerate
        return NonparametricResult {
            test_name: "KS test (normality)", statistic: 0.0, p_value: 1.0
        };
    }

    let mut z_scores: Vec<f64> = clean.iter().map(|&x| (x - mean) / std).collect();
    z_scores.sort_by(|a, b| a.total_cmp(b));

    let mut d_max = 0.0_f64;
    for (i, &z) in z_scores.iter().enumerate() {
        let cdf = normal_cdf(z);
        let d1 = ((i + 1) as f64 / nf - cdf).abs();
        let d2 = (cdf - i as f64 / nf).abs();
        d_max = d_max.max(d1).max(d2);
    }

    let p = ks_p_value(d_max, n);

    NonparametricResult {
        test_name: "KS test (normality)", statistic: d_max, p_value: p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shapiro-Wilk normality test (Shapiro & Wilk 1965, Royston 1995)
// ═══════════════════════════════════════════════════════════════════════════

/// Shapiro-Wilk test for normality.
///
/// W = (Σ aᵢ x_{(i)})² / Σ (xᵢ - x̄)².
/// Valid for 3 ≤ n ≤ 5000.
/// p-value via Royston 1995 approximation.
pub fn shapiro_wilk(data: &[f64]) -> NonparametricResult {
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    if n < 3 {
        return NonparametricResult {
            test_name: "Shapiro-Wilk", statistic: f64::NAN, p_value: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean = sorted.iter().sum::<f64>() / nf;
    let ss: f64 = sorted.iter().map(|x| (x - mean) * (x - mean)).sum();

    if ss < 1e-15 {
        // All values identical — degenerate
        return NonparametricResult {
            test_name: "Shapiro-Wilk", statistic: 1.0, p_value: 1.0,
        };
    }

    // Compute Shapiro-Wilk coefficients
    let a = shapiro_wilk_coefficients(n);

    // W = b² / SS, where b = Σ a_{n-1-i} (x_{(n-i)} - x_{(i+1)})
    // a[n-1] is the extreme coefficient (largest magnitude, positive).
    let half = n / 2;
    let mut b = 0.0;
    for i in 0..half {
        b += a[n - 1 - i] * (sorted[n - 1 - i] - sorted[i]);
    }
    let w = (b * b / ss).min(1.0); // clamp floating-point overshoot

    // P-value via Royston 1995 approximation
    let p = shapiro_wilk_p_value(w, n);

    NonparametricResult {
        test_name: "Shapiro-Wilk", statistic: w, p_value: p,
    }
}

/// D'Agostino-Pearson omnibus test for normality.
///
/// Combines skewness and kurtosis z-scores: K² = z_s² + z_k² ~ χ²(2).
/// Better than Shapiro-Wilk for n > 5000.
pub fn dagostino_pearson(data: &[f64]) -> NonparametricResult {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean.len();

    if n < 8 {
        return NonparametricResult {
            test_name: "D'Agostino-Pearson", statistic: f64::NAN, p_value: f64::NAN,
        };
    }

    let moments = crate::descriptive::moments_ungrouped(&clean);
    let skew = moments.skewness(false); // bias-corrected skewness
    let kurt = moments.kurtosis(true, false); // bias-corrected EXCESS kurtosis

    let nf = n as f64;

    // Skewness z-score: under normality, skewness ≈ 0
    // Var(G1) ≈ 6(n-2) / ((n+1)(n+3))
    let var_s = 6.0 * (nf - 2.0) / ((nf + 1.0) * (nf + 3.0));
    let z_s = skew / var_s.sqrt();

    // Kurtosis z-score: under normality, excess kurtosis ≈ 0
    // Var(G2) ≈ 24n(n-2)(n-3) / ((n+1)²(n+3)(n+5))
    let var_k = 24.0 * nf * (nf - 2.0) * (nf - 3.0)
        / ((nf + 1.0) * (nf + 1.0) * (nf + 3.0) * (nf + 5.0));
    let z_k = kurt / var_k.sqrt();

    // Omnibus statistic
    let k2 = z_s * z_s + z_k * z_k;
    let p = 1.0 - crate::special_functions::chi2_cdf(k2, 2.0);

    NonparametricResult {
        test_name: "D'Agostino-Pearson", statistic: k2, p_value: p,
    }
}

/// Jarque-Bera test for normality (Jarque & Bera 1980).
///
/// JB = (n/6)(S² + K²/4) where S = skewness, K = excess kurtosis.
/// Under H₀ (normality): JB ~ χ²(2).
///
/// The simplest normality test — O(1) from MomentStats. Less powerful than
/// Shapiro-Wilk for small n but valid for all n ≥ 8.
pub fn jarque_bera(data: &[f64]) -> NonparametricResult {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean.len();
    if n < 8 {
        return NonparametricResult {
            test_name: "Jarque-Bera", statistic: f64::NAN, p_value: f64::NAN,
        };
    }
    let moments = crate::descriptive::moments_ungrouped(&clean);
    let skew = moments.skewness(false); // bias-corrected
    let kurt = moments.kurtosis(true, false); // bias-corrected excess kurtosis
    let nf = n as f64;
    let jb = (nf / 6.0) * (skew * skew + kurt * kurt / 4.0);
    let p = 1.0 - crate::special_functions::chi2_cdf(jb, 2.0);
    NonparametricResult { test_name: "Jarque-Bera", statistic: jb, p_value: p }
}

/// Shapiro-Wilk coefficients.
///
/// For n <= 5: tabled exact values (Shapiro & Wilk 1965).
/// For n >= 6: normalized Blom expected order statistics (m_i / ||m||).
/// This gives the correct W statistic — the Royston corrections
/// only affect the p-value approximation, not the W computation.
fn shapiro_wilk_coefficients(n: usize) -> Vec<f64> {
    use crate::special_functions::normal_quantile;

    match n {
        3 => vec![-0.7071068, 0.0, 0.7071068],
        4 => vec![-0.6872, -0.1677, 0.1677, 0.6872],
        5 => vec![-0.6646, -0.2413, 0.0, 0.2413, 0.6646],
        _ => {
            // Blom approximation to expected normal order statistics
            let nf = n as f64;
            let m: Vec<f64> = (1..=n).map(|i| {
                normal_quantile((i as f64 - 0.375) / (nf + 0.25))
            }).collect();
            let cn: f64 = m.iter().map(|x| x * x).sum();
            let cn_sqrt = cn.sqrt();
            m.iter().map(|mi| mi / cn_sqrt).collect()
        }
    }
}

/// Royston 1995 p-value approximation for the Shapiro-Wilk W statistic.
///
/// Uses normalized transform from Royston (1995, Applied Statistics 44:R94).
fn shapiro_wilk_p_value(w: f64, n: usize) -> f64 {
    use crate::special_functions::normal_cdf;

    if w >= 1.0 { return 1.0; }
    if w <= 0.0 { return 0.0; }

    let nf = n as f64;
    let ln_n = nf.ln();

    if n <= 11 {
        // Small sample: transform -ln(1-W) with power gamma
        let gamma = -2.273 + 0.459 * nf;
        if gamma <= 0.0 { return 0.5; }
        let y = (-(1.0 - w).max(1e-15).ln()).powf(gamma);
        let mu = -0.0006714 * nf.powi(3) + 0.025054 * nf.powi(2)
            - 0.39978 * nf + 0.5440;
        let log_sigma = -0.0020322 * nf.powi(3) + 0.062767 * nf.powi(2)
            - 0.77857 * nf + 1.3822;
        let sigma = log_sigma.exp();
        let z = (y - mu) / sigma;
        1.0 - normal_cdf(z)
    } else {
        // n >= 12: ln(1-W) transform, calibrated from Royston 1995
        let y = (1.0 - w).max(1e-15).ln();

        // Mu and sigma of ln(1-W) under normality, polynomial in ln(n)
        // Calibrated to: mu(20)≈-3.1, mu(100)≈-4.7, mu(1000)≈-6.8
        let mu = 0.0038915 * ln_n.powi(3) - 0.083751 * ln_n.powi(2)
            - 0.31082 * ln_n - 1.5861;
        let log_sigma = 0.0030302 * ln_n.powi(2) - 0.082676 * ln_n - 0.4803;
        let sigma = log_sigma.exp();

        // z > 0 → W too small → non-normal → small p
        let z = (y - mu) / sigma;
        1.0 - normal_cdf(z)
    }
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// D = sup|Fₙ(x) - Gₘ(x)|
pub fn ks_test_two_sample(x: &[f64], y: &[f64]) -> NonparametricResult {
    let mut sx: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let mut sy: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();
    sx.sort_by(|a, b| a.total_cmp(b));
    sy.sort_by(|a, b| a.total_cmp(b));

    let n1 = sx.len();
    let n2 = sy.len();
    if n1 == 0 || n2 == 0 {
        return NonparametricResult {
            test_name: "KS test (two-sample)", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let mut d_max = 0.0_f64;
    let mut i = 0;
    let mut j = 0;
    while i < n1 || j < n2 {
        // Advance tied values simultaneously
        let advance_x = i < n1 && (j >= n2 || sx[i] <= sy[j]);
        let advance_y = j < n2 && (i >= n1 || sy[j] <= sx[i]);
        if advance_x { i += 1; }
        if advance_y { j += 1; }
        let fx = i as f64 / n1 as f64;
        let gy = j as f64 / n2 as f64;
        d_max = d_max.max((fx - gy).abs());
    }

    let en = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let p = ks_p_value(d_max, (en * en).round() as usize);

    NonparametricResult {
        test_name: "KS test (two-sample)", statistic: d_max, p_value: p,
    }
}

/// Kolmogorov distribution p-value approximation.
///
/// Uses the asymptotic formula: P(D > d) ≈ 2 Σ (-1)^(k-1) exp(-2k²(√n·d)²)
fn ks_p_value(d: f64, n: usize) -> f64 {
    if d <= 0.0 { return 1.0; }
    if d >= 1.0 { return 0.0; }

    let sqrt_n = (n as f64).sqrt();
    let z = sqrt_n * d;

    // Asymptotic series (converges fast for z > 0.5)
    let mut p = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * z * z).exp();
        if term < 1e-15 { break; }
        if k % 2 == 1 {
            p += term;
        } else {
            p -= term;
        }
    }
    (2.0 * p).max(0.0).min(1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════════

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Point estimate of the statistic.
    pub estimate: f64,
    /// Lower confidence bound.
    pub ci_lower: f64,
    /// Upper confidence bound.
    pub ci_upper: f64,
    /// Standard error (SD of bootstrap distribution).
    pub se: f64,
    /// Number of bootstrap resamples.
    pub n_resamples: usize,
}

/// Percentile bootstrap confidence interval.
///
/// Resamples data `n_resamples` times, computes `statistic` each time,
/// returns percentile-based CI at the given `alpha` level.
///
/// Uses a simple LCG for reproducibility (no external RNG dependency).
pub fn bootstrap_percentile(
    data: &[f64],
    statistic: fn(&[f64]) -> f64,
    n_resamples: usize,
    alpha: f64,
    seed: u64,
) -> BootstrapResult {
    let n = data.len();
    if n == 0 {
        return BootstrapResult {
            estimate: f64::NAN, ci_lower: f64::NAN, ci_upper: f64::NAN,
            se: f64::NAN, n_resamples,
        };
    }

    let estimate = statistic(data);
    if n_resamples < 2 {
        return BootstrapResult {
            estimate, ci_lower: f64::NAN, ci_upper: f64::NAN,
            se: f64::NAN, n_resamples,
        };
    }
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut boot_stats = Vec::with_capacity(n_resamples);
    let mut resample = vec![0.0; n];

    for _ in 0..n_resamples {
        // Resample with replacement
        for slot in resample.iter_mut() {
            let idx = crate::rng::TamRng::next_range(&mut rng, n as u64) as usize;
            *slot = data[idx];
        }
        boot_stats.push(statistic(&resample));
    }

    boot_stats.sort_by(|a, b| a.total_cmp(b));

    let lo_idx = ((alpha / 2.0) * n_resamples as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64) as usize;
    let ci_lower = boot_stats[lo_idx.min(n_resamples - 1)];
    let ci_upper = boot_stats[hi_idx.min(n_resamples - 1)];

    let bm = crate::descriptive::moments_ungrouped(&boot_stats);
    let se = bm.std(1);

    BootstrapResult { estimate, ci_lower, ci_upper, se, n_resamples }
}

/// Permutation test for two-sample difference in means.
///
/// Returns p-value: proportion of permutations with |diff| ≥ |observed diff|.
pub fn permutation_test_mean_diff(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: u64,
) -> NonparametricResult {
    let n1 = x.len();
    let n = n1 + y.len();

    let mut combined: Vec<f64> = Vec::with_capacity(n);
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);

    let obs_diff = (mean_slice(x) - mean_slice(y)).abs();

    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = crate::rng::TamRng::next_range(&mut rng, (i + 1) as u64) as usize;
            combined.swap(i, j);
        }
        let perm_diff = (mean_slice(&combined[..n1]) - mean_slice(&combined[n1..])).abs();
        if perm_diff >= obs_diff { count_extreme += 1; }
    }

    let p = (count_extreme + 1) as f64 / (n_permutations + 1) as f64;

    NonparametricResult {
        test_name: "Permutation test (mean diff)",
        statistic: obs_diff,
        p_value: p,
    }
}

fn mean_slice(data: &[f64]) -> f64 {
    crate::descriptive::moments_ungrouped(data).mean()
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel Density Estimation
// ═══════════════════════════════════════════════════════════════════════════

/// Kernel type for density estimation.
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
}

/// Evaluate kernel density estimate at given points.
///
/// KDE: f̂(x) = (1/nh) Σ K((x - xᵢ)/h)
///
/// If `bandwidth` is None, uses Silverman's rule of thumb:
/// h = 0.9 × min(σ, IQR/1.34) × n^(-1/5)
pub fn kde(data: &[f64], eval_points: &[f64], kernel: KernelType, bandwidth: Option<f64>) -> Vec<f64> {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean.len();
    if n == 0 { return vec![0.0; eval_points.len()]; }

    let h = bandwidth.unwrap_or_else(|| silverman_bandwidth(&clean));
    if h <= 0.0 { return vec![0.0; eval_points.len()]; }

    let nf = n as f64;
    eval_points.iter().map(|&x| {
        let sum: f64 = clean.iter().map(|&xi| {
            let u = (x - xi) / h;
            kernel_eval(kernel, u)
        }).sum();
        sum / (nf * h)
    }).collect()
}

/// Silverman's rule of thumb bandwidth.
///
/// h = 0.9 × min(σ, IQR/1.34) × n^(-1/5)
pub fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 1.0; }

    let sm = crate::descriptive::moments_ungrouped(data);
    let std = sm.std(1);

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;

    let spread = if iqr > 0.0 { std.min(iqr / 1.34) } else { std };
    if spread <= 0.0 { return 1.0; } // constant data: fallback to unit bandwidth
    0.9 * spread * n.powf(-0.2)
}

/// Scott's rule of thumb bandwidth (Scott 1979): h = 1.06·σ·n^(-1/5).
///
/// Assumes approximately normal distribution. Less robust to outliers than Silverman,
/// but simpler and a common default in R and scipy.
pub fn scott_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 1.0; }
    let std = crate::descriptive::moments_ungrouped(data).std(1);
    if std <= 0.0 { return 1.0; }
    1.06 * std * n.powf(-0.2)
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimal bin count rules for histograms
// ═══════════════════════════════════════════════════════════════════════════

/// Sturges' rule (1926): k = ⌈log₂(n)⌉ + 1.
///
/// Assumes normal distribution. Over-smooths for large n or skewed data.
/// Default in R's `hist()`. Returns 1 for n < 2.
pub fn sturges_bins(n: usize) -> usize {
    if n < 2 { return 1; }
    ((n as f64).log2().ceil() as usize) + 1
}

/// Scott's rule for bin count: k = ⌈(max - min) / h⌉ where h = 3.5·σ/n^(1/3).
///
/// Assumes normal. Better than Sturges for larger n.
pub fn scott_bins(data: &[f64]) -> usize {
    let n = data.len();
    if n < 2 { return 1; }
    let std = crate::descriptive::moments_ungrouped(data).std(1);
    if std <= 0.0 { return 1; }
    let h = 3.5 * std * (n as f64).powf(-1.0 / 3.0);
    let (min, max) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
        if x.is_nan() { (lo, hi) } else { (lo.min(x), hi.max(x)) }
    });
    let range = max - min;
    if range <= 0.0 || h <= 0.0 { return 1; }
    (range / h).ceil() as usize
}

/// Freedman-Diaconis rule: k = ⌈(max - min) / h⌉ where h = 2·IQR/n^(1/3).
///
/// Robust to outliers (uses IQR instead of σ). Default in numpy's `histogram_bin_edges`.
pub fn freedman_diaconis_bins(data: &[f64]) -> usize {
    let n = data.len();
    if n < 2 { return 1; }
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let ns = sorted.len();
    if ns < 2 { return 1; }
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    if iqr <= 0.0 { return sturges_bins(ns); } // fallback for constant-ish data
    let h = 2.0 * iqr * (ns as f64).powf(-1.0 / 3.0);
    let range = sorted[ns - 1] - sorted[0];
    if range <= 0.0 || h <= 0.0 { return 1; }
    (range / h).ceil() as usize
}

/// Doane's rule (1976): extends Sturges for skewed data.
///
/// k = 1 + log₂(n) + log₂(1 + |g₁|/σ_{g₁}) where g₁ = sample skewness,
/// σ_{g₁} = √(6(n-2)/((n+1)(n+3))).
pub fn doane_bins(data: &[f64]) -> usize {
    let n = data.len();
    if n < 3 { return 1; }
    let nf = n as f64;
    let moments = crate::descriptive::moments_ungrouped(data);
    let skew = moments.skewness(false);
    let sigma_g1 = (6.0 * (nf - 2.0) / ((nf + 1.0) * (nf + 3.0))).sqrt();
    if sigma_g1 < 1e-15 { return sturges_bins(n); }
    let k = 1.0 + nf.log2() + (1.0 + skew.abs() / sigma_g1).log2();
    k.ceil() as usize
}

/// Binning rule for automatic histogram selection.
#[derive(Debug, Clone, Copy)]
pub enum BinRule {
    /// Sturges' rule (default in R).
    Sturges,
    /// Scott's rule (assumes normal).
    Scott,
    /// Freedman-Diaconis (robust to outliers).
    FreedmanDiaconis,
    /// Doane's rule (handles skewness).
    Doane,
    /// Explicit bin count.
    Fixed(usize),
}

/// Histogram result.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin edges: length = counts.len() + 1. `edges[i]..edges[i+1]` is bin i.
    pub edges: Vec<f64>,
    /// Counts per bin.
    pub counts: Vec<u64>,
    /// Number of NaN values skipped.
    pub nan_count: usize,
}

/// Compute a histogram with automatic bin count selection.
///
/// `data`: input values (NaN values are skipped and counted separately).
/// `rule`: bin-count rule (Sturges/Scott/FreedmanDiaconis/Doane/Fixed).
///
/// Returns equal-width bins spanning [min, max] of the data.
pub fn histogram_auto(data: &[f64], rule: BinRule) -> Histogram {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let nan_count = data.len() - clean.len();
    let n = clean.len();

    if n == 0 {
        return Histogram { edges: vec![], counts: vec![], nan_count };
    }

    let k = match rule {
        BinRule::Sturges => sturges_bins(n),
        BinRule::Scott => scott_bins(&clean),
        BinRule::FreedmanDiaconis => freedman_diaconis_bins(&clean),
        BinRule::Doane => doane_bins(&clean),
        BinRule::Fixed(k) => k.max(1),
    };

    let (min, max) = clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
        (lo.min(x), hi.max(x))
    });

    // Handle degenerate case: all same value → single bin
    if (max - min).abs() < 1e-15 {
        return Histogram {
            edges: vec![min - 0.5, max + 0.5],
            counts: vec![n as u64],
            nan_count,
        };
    }

    let width = (max - min) / k as f64;
    let mut edges = Vec::with_capacity(k + 1);
    for i in 0..=k { edges.push(min + i as f64 * width); }
    // Nudge the last edge up slightly to ensure max falls in the last bin
    *edges.last_mut().unwrap() = max + width * 1e-9;

    let mut counts = vec![0u64; k];
    for &x in &clean {
        let idx = (((x - min) / width) as usize).min(k - 1);
        counts[idx] += 1;
    }

    Histogram { edges, counts, nan_count }
}

// ═══════════════════════════════════════════════════════════════════════════
// Empirical Cumulative Distribution Function (ECDF)
// ═══════════════════════════════════════════════════════════════════════════

/// ECDF result: sorted data points plus cumulative probabilities.
#[derive(Debug, Clone)]
pub struct Ecdf {
    /// Sorted unique data values (NaN removed).
    pub x: Vec<f64>,
    /// Cumulative probability at each x: F̂(x[i]) = (i+1)/n.
    pub p: Vec<f64>,
}

impl Ecdf {
    /// Evaluate the ECDF at a query point via binary search.
    /// F̂(q) = (number of x_i ≤ q) / n.
    pub fn eval(&self, q: f64) -> f64 {
        if self.x.is_empty() { return f64::NAN; }
        if q < self.x[0] { return 0.0; }
        if q >= *self.x.last().unwrap() { return 1.0; }
        // Binary search for last x <= q
        let mut lo = 0usize;
        let mut hi = self.x.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= q { lo = mid + 1; } else { hi = mid; }
        }
        self.p[lo - 1]
    }
}

/// Compute the empirical CDF of a sample.
///
/// F̂(x) = (1/n) Σ 1(X_i ≤ x). Returns sorted x and cumulative p = (i+1)/n.
pub fn ecdf(data: &[f64]) -> Ecdf {
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let p: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
    Ecdf { x: sorted, p }
}

/// Dvoretzky-Kiefer-Wolfowitz (DKW) confidence band for the ECDF.
///
/// With probability ≥ 1-α, the true CDF F lies within F̂ ± ε_n where
/// ε_n = √(ln(2/α) / (2n)).
///
/// This band is **uniform** over all x (simultaneously valid), unlike pointwise CIs.
///
/// Returns (lower, upper) bands: F_lower[i] = max(F̂(x_i) - ε, 0),
///                               F_upper[i] = min(F̂(x_i) + ε, 1).
pub fn ecdf_confidence_band(ecdf: &Ecdf, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let n = ecdf.x.len();
    if n == 0 || alpha <= 0.0 || alpha >= 1.0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    let eps = ((2.0_f64 / alpha).ln() / (2.0 * n as f64)).sqrt();
    let lower: Vec<f64> = ecdf.p.iter().map(|&p| (p - eps).max(0.0)).collect();
    let upper: Vec<f64> = ecdf.p.iter().map(|&p| (p + eps).min(1.0)).collect();
    (lower, upper)
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn kernel_eval(kernel: KernelType, u: f64) -> f64 {
    match kernel {
        KernelType::Gaussian => {
            (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
        }
        KernelType::Epanechnikov => {
            if u.abs() <= 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KDE via FFT — O(n log n) instead of O(n·m)
// ═══════════════════════════════════════════════════════════════════════════

/// Fast Gaussian KDE via FFT convolution.
///
/// Algorithm: bin data → convolve with Gaussian kernel on grid → normalize.
/// O(n + g log g) where g = grid size, vs O(n·g) for direct evaluation.
///
/// `data`: input samples.
/// `n_grid`: number of equally-spaced grid points (default: 1024).
/// `bandwidth`: Gaussian bandwidth h. None → Silverman's rule.
/// Returns (grid_points, density_values).
pub fn kde_fft(data: &[f64], n_grid: usize, bandwidth: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean.len();
    if n == 0 || n_grid < 2 { return (vec![], vec![]); }

    let h = bandwidth.unwrap_or_else(|| silverman_bandwidth(&clean));
    if h <= 0.0 { return (vec![], vec![]); }

    // Grid range: extend beyond data by 3*h on each side
    let data_min = clean.iter().copied().fold(f64::INFINITY, f64::min);
    let data_max = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = data_min - 3.0 * h;
    let hi = data_max + 3.0 * h;
    let dx = (hi - lo) / (n_grid - 1) as f64;

    // Step 1: Bin data into histogram on the grid (linear binning)
    let mut grid_counts = vec![0.0; n_grid];
    for &x in &clean {
        let pos = (x - lo) / dx;
        let idx = pos.floor() as usize;
        if idx + 1 < n_grid {
            let frac = pos - idx as f64;
            grid_counts[idx] += 1.0 - frac;
            grid_counts[idx + 1] += frac;
        } else if idx < n_grid {
            grid_counts[idx] += 1.0;
        }
    }

    // Step 2: Build Gaussian kernel on the same grid spacing
    // Kernel extends ±4h (beyond that it's < 1e-4 of peak)
    let kernel_half = (4.0 * h / dx).ceil() as usize;
    let kernel_len = 2 * kernel_half + 1;
    let mut kernel = vec![0.0; kernel_len];
    for k in 0..kernel_len {
        let u = (k as f64 - kernel_half as f64) * dx / h;
        kernel[k] = (-0.5 * u * u).exp() / ((2.0 * std::f64::consts::PI).sqrt() * h);
    }

    // Step 3: Convolve (FFT-based O(n log n))
    let conv = crate::signal_processing::convolve(&grid_counts, &kernel);

    // Step 4: Extract the valid region and normalize
    // conv[i + kernel_half] = Σ_j count[j] · K_h((i-j)·dx) ≈ n · f̂(grid[i])
    let grid: Vec<f64> = (0..n_grid).map(|i| lo + i as f64 * dx).collect();
    let density: Vec<f64> = (0..n_grid).map(|i| {
        let idx = i + kernel_half;
        if idx < conv.len() {
            (conv[idx] / n as f64).max(0.0)
        } else {
            0.0
        }
    }).collect();

    (grid, density)
}

// ═══════════════════════════════════════════════════════════════════════════
// Runs test
// ═══════════════════════════════════════════════════════════════════════════

/// Wald-Wolfowitz runs test for randomness.
///
/// Tests whether a sequence of binary outcomes is random.
/// Input: sequence of boolean-like values (above/below median).
pub fn runs_test(data: &[bool]) -> NonparametricResult {
    let n = data.len();
    if n < 2 {
        return NonparametricResult {
            test_name: "Runs test", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let n1 = data.iter().filter(|&&x| x).count() as f64;
    let n2 = data.iter().filter(|&&x| !x).count() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return NonparametricResult {
            test_name: "Runs test", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    // Count runs
    let mut runs = 1.0;
    for i in 1..n {
        if data[i] != data[i-1] { runs += 1.0; }
    }

    // Expected runs and variance under H0 (randomness)
    let mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0;
    let sigma = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
        / ((n1 + n2).powi(2) * (n1 + n2 - 1.0))).sqrt();

    let z = if sigma > 0.0 { (runs - mu) / sigma } else { 0.0 };
    let p = normal_two_tail_p(z);

    NonparametricResult {
        test_name: "Runs test", statistic: runs, p_value: p,
    }
}

/// Convenience: runs test from numeric data (split at median).
pub fn runs_test_numeric(data: &[f64]) -> NonparametricResult {
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return NonparametricResult {
            test_name: "Runs test", statistic: f64::NAN, p_value: f64::NAN
        };
    }
    let med = quantile_sorted(&sorted, 0.5);
    let binary: Vec<bool> = data.iter().map(|&x| x > med).collect();
    runs_test(&binary)
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign test
// ═══════════════════════════════════════════════════════════════════════════

/// Sign test for paired differences or one-sample median.
///
/// Tests whether median of data equals `median0`.
/// Counts positive and negative deviations from median0.
/// Uses normal approximation: z = (S - n/2) / √(n/4)
pub fn sign_test(data: &[f64], median0: f64) -> NonparametricResult {
    let positives: usize = data.iter().filter(|&&x| x > median0).count();
    let negatives: usize = data.iter().filter(|&&x| x < median0).count();
    let n = positives + negatives; // exclude ties

    if n == 0 {
        return NonparametricResult {
            test_name: "Sign test", statistic: f64::NAN, p_value: f64::NAN
        };
    }

    let s = positives as f64;
    let nf = n as f64;
    let z = (s - nf / 2.0) / (nf / 4.0).sqrt();
    let p = normal_two_tail_p(z);

    NonparametricResult {
        test_name: "Sign test", statistic: s, p_value: p,
    }
}

// ─── Level-spacing r-statistic ───────────────────────────────────────────────

/// Level-spacing r-statistic for spectral sequences.
///
/// Given a sorted sequence (eigenvalues, ζ zeros, energy levels, etc.), computes
/// the mean consecutive-gap ratio: rᵢ = min(δᵢ, δᵢ₊₁) / max(δᵢ, δᵢ₊₁) where
/// δᵢ = (xᵢ₊₁ − xᵢ) / mean_gap are normalized level spacings.
///
/// Expected values (Montgomery-Odlyzko Law / random matrix theory):
/// - GUE (Gaussian Unitary Ensemble, quantum chaos / efficient market): r ≈ 0.536
/// - GOE (Gaussian Orthogonal Ensemble, time-reversal symmetric): r ≈ 0.530
/// - Poisson (independent levels, integrable / structured): r ≈ 0.386
///
/// Returns `f64::NAN` if `sorted_values` has fewer than 3 elements.
///
/// # Applications
/// - Riemann ζ zeros → r ≈ 0.536 (Montgomery-Odlyzko conjecture, empirically confirmed)
/// - Market correlation matrix eigenvalues → r near GUE = efficient; near Poisson = structured
/// - Quantum energy spectra → GUE for chaotic, Poisson for integrable systems
pub fn level_spacing_r_stat(sorted_values: &[f64]) -> f64 {
    if sorted_values.len() < 3 {
        return f64::NAN;
    }
    let gaps: Vec<f64> = sorted_values.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    if mean_gap == 0.0 {
        return f64::NAN;
    }
    let norm_gaps: Vec<f64> = gaps.iter().map(|&g| g / mean_gap).collect();
    let r_sum: f64 = norm_gaps
        .windows(2)
        .map(|w| w[0].min(w[1]) / w[0].max(w[1]))
        .sum();
    r_sum / (norm_gaps.len() - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Correlation coefficients (pairwise, for auto-detection)
// ═══════════════════════════════════════════════════════════════════════════

/// Pearson product-moment correlation between two equal-length slices.
///
/// r = Σ((x-x̄)(y-ȳ)) / √(Σ(x-x̄)² · Σ(y-ȳ)²)
/// Returns NaN if either slice has zero variance.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-15 { f64::NAN } else { num / denom }
}

/// Phi coefficient: Pearson correlation for two binary (0/1) variables.
///
/// φ = (ad - bc) / √((a+b)(c+d)(a+c)(b+d))
/// where a,b,c,d are the 2×2 contingency table entries.
pub fn phi_coefficient(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let (mut a, mut b, mut c, mut d) = (0.0f64, 0.0, 0.0, 0.0);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let xb = *xi > 0.5;
        let yb = *yi > 0.5;
        match (xb, yb) {
            (true,  true)  => a += 1.0,
            (true,  false) => b += 1.0,
            (false, true)  => c += 1.0,
            (false, false) => d += 1.0,
        }
    }
    let denom = ((a + b) * (c + d) * (a + c) * (b + d)).sqrt();
    if denom < 1e-15 { f64::NAN } else { (a * d - b * c) / denom }
}

/// Point-biserial correlation: Pearson r when one variable is binary (0/1).
///
/// r_pb = (M_1 - M_0) / s_total * √(n_1 * n_0 / n²)
/// where M_1, M_0 are means of the continuous variable in each binary group.
/// Equivalent to Pearson r between the binary indicator and the continuous variable.
pub fn point_biserial(binary: &[f64], continuous: &[f64]) -> f64 {
    // Pearson r is exactly point-biserial when one variable is binary.
    pearson_r(binary, continuous)
}

/// Biserial correlation: assumes the binary variable is a dichotomized continuous variable.
///
/// r_b = (M_1 - M_0) / s_y * (p · q / φ(z))
///
/// where p = n_1/n, q = n_0/n, z = Φ⁻¹(q), φ is the standard normal density.
/// This differs from point-biserial by the factor `pq / φ(z) > 1`, so |r_b| ≥ |r_pb|.
///
/// Used in educational testing (item-total correlation assuming latent ability).
///
/// `binary`: 0/1 values; `continuous`: paired scores (same length).
pub fn biserial_correlation(binary: &[f64], continuous: &[f64]) -> f64 {
    assert_eq!(binary.len(), continuous.len());
    let n = binary.len();
    if n < 2 { return f64::NAN; }

    let (mut n1, mut n0) = (0.0, 0.0);
    let (mut sum1, mut sum0) = (0.0, 0.0);
    for i in 0..n {
        if binary[i] > 0.5 { n1 += 1.0; sum1 += continuous[i]; }
        else { n0 += 1.0; sum0 += continuous[i]; }
    }
    if n1 < 1.0 || n0 < 1.0 { return f64::NAN; }

    let m1 = sum1 / n1;
    let m0 = sum0 / n0;
    let mean_y: f64 = continuous.iter().sum::<f64>() / n as f64;
    let var_y: f64 = continuous.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n as f64;
    let s_y = var_y.sqrt();
    if s_y < 1e-15 { return f64::NAN; }

    let p = n1 / n as f64;
    let q = n0 / n as f64;
    // z such that Φ(z) = q (lower tail); equivalently z = Φ⁻¹(q)
    let z = crate::special_functions::normal_quantile(q);
    // Standard normal density at z
    let phi_z = (-0.5 * z * z).exp() / (std::f64::consts::TAU).sqrt();
    if phi_z < 1e-300 { return f64::NAN; }

    (m1 - m0) / s_y * (p * q / phi_z)
}

/// Rank-biserial correlation (Glass 1966): effect size from Mann-Whitney U.
///
/// r_rb = 1 - 2U / (n1 * n2)
///
/// Ranges [-1, 1]. Measures probability that a random observation from group 1
/// exceeds one from group 2 (scaled). Non-parametric effect size for group comparisons.
pub fn rank_biserial(x: &[f64], y: &[f64]) -> f64 {
    let n1 = x.len() as f64;
    let n2 = y.len() as f64;
    if n1 < 1.0 || n2 < 1.0 { return f64::NAN; }

    let mw = mann_whitney_u(x, y);
    // U statistic: count pairs where x > y
    // mann_whitney_u returns "statistic" as U (smaller of the two)
    let u = mw.statistic;
    1.0 - 2.0 * u / (n1 * n2)
}

/// Tetrachoric correlation: Pearson r for 2x2 binary table assuming latent
/// bivariate normal with thresholds.
///
/// Uses Divgi's (1979) cosine approximation (sufficient for most applications):
/// r ≈ cos(π / (1 + √(ad/bc)))
///
/// Table: [a, b, c, d] = [(0,0), (0,1), (1,0), (1,1)].
/// For polychoric (more than 2 categories), a different estimator is needed.
pub fn tetrachoric(table: &[f64; 4]) -> f64 {
    let a = table[0];
    let b = table[1];
    let c = table[2];
    let d = table[3];
    let bc = b * c;
    if bc < 1e-15 {
        // Zero cell in off-diagonal: perfect association (or degenerate)
        if a * d > bc { return 1.0; }
        else if a * d < bc { return -1.0; }
        else { return f64::NAN; }
    }
    let ad = a * d;
    // Divgi's cosine approximation
    let ratio = ad / bc;
    (std::f64::consts::PI / (1.0 + ratio.sqrt())).cos()
}

/// Cramér's V: association measure for r×c contingency tables.
///
/// V = √(χ² / (n · min(r-1, c-1)))
///
/// Ranges from 0 (no association) to 1 (perfect association).
/// Generalizes the phi coefficient to tables larger than 2×2.
///
/// `table`: r×c contingency table (row-major, counts).
/// `n_rows`: number of rows in the table.
pub fn cramers_v(table: &[f64], n_rows: usize) -> f64 {
    let n_cols = if n_rows == 0 { 0 } else { table.len() / n_rows };
    if n_rows < 2 || n_cols < 2 { return f64::NAN; }

    let n: f64 = table.iter().sum();
    if n < 1e-15 { return f64::NAN; }

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

    // Chi-squared statistic
    let mut chi2 = 0.0;
    for r in 0..n_rows {
        for c in 0..n_cols {
            let expected = row_totals[r] * col_totals[c] / n;
            if expected > 1e-15 {
                let observed = table[r * n_cols + c];
                chi2 += (observed - expected).powi(2) / expected;
            }
        }
    }

    let min_dim = (n_rows - 1).min(n_cols - 1) as f64;
    if min_dim < 1e-15 { return f64::NAN; }
    (chi2 / (n * min_dim)).sqrt()
}

/// Eta coefficient (correlation ratio): proportion of variance in Y explained by group membership X.
///
/// η² = SS_between / SS_total
///
/// Ranges from 0 (no relationship) to 1 (perfect group separation).
/// Measures the strength of association between a categorical variable (groups)
/// and a continuous variable (values).
///
/// `values`: the continuous variable.
/// `groups`: group label for each value (same length as `values`).
pub fn eta_squared(values: &[f64], groups: &[usize]) -> f64 {
    assert_eq!(values.len(), groups.len());
    let n = values.len();
    if n < 2 { return f64::NAN; }

    let grand_mean = values.iter().sum::<f64>() / n as f64;
    let ss_total: f64 = values.iter().map(|v| (v - grand_mean).powi(2)).sum();
    if ss_total < 1e-15 { return 0.0; } // constant data → no variance to explain

    // Group means and sizes
    let max_group = *groups.iter().max().unwrap_or(&0);
    let mut group_sums = vec![0.0; max_group + 1];
    let mut group_counts = vec![0usize; max_group + 1];
    for (i, &g) in groups.iter().enumerate() {
        group_sums[g] += values[i];
        group_counts[g] += 1;
    }

    let ss_between: f64 = (0..=max_group)
        .filter(|&g| group_counts[g] > 0)
        .map(|g| {
            let group_mean = group_sums[g] / group_counts[g] as f64;
            group_counts[g] as f64 * (group_mean - grand_mean).powi(2)
        })
        .sum();

    (ss_between / ss_total).clamp(0.0, 1.0)
}

/// Distance correlation (Szekely, Rizzo, Bakirov 2007): detects nonlinear dependencies.
///
/// dCor(X, Y) = dCov(X,Y) / √(dVar(X) · dVar(Y))
///
/// Ranges from 0 (independence) to 1 (dependence — including nonlinear).
/// dCor = 0 if and only if X and Y are independent (for any finite moment).
///
/// `x`, `y`: paired observations (same length).
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let nf = n as f64;

    // Distance matrices
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (x[i] - x[j]).abs();
            b[i * n + j] = (y[i] - y[j]).abs();
        }
    }

    // Double-center: A_{ij} = a_{ij} - ā_{i.} - ā_{.j} + ā_{..}
    let row_means_a: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i*n+j]).sum::<f64>() / nf).collect();
    let col_means_a: Vec<f64> = (0..n).map(|j| (0..n).map(|i| a[i*n+j]).sum::<f64>() / nf).collect();
    let grand_mean_a: f64 = a.iter().sum::<f64>() / (nf * nf);

    let row_means_b: Vec<f64> = (0..n).map(|i| (0..n).map(|j| b[i*n+j]).sum::<f64>() / nf).collect();
    let col_means_b: Vec<f64> = (0..n).map(|j| (0..n).map(|i| b[i*n+j]).sum::<f64>() / nf).collect();
    let grand_mean_b: f64 = b.iter().sum::<f64>() / (nf * nf);

    // dCov², dVar_X, dVar_Y
    let mut dcov2 = 0.0;
    let mut dvar_x = 0.0;
    let mut dvar_y = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a_centered = a[i*n+j] - row_means_a[i] - col_means_a[j] + grand_mean_a;
            let b_centered = b[i*n+j] - row_means_b[i] - col_means_b[j] + grand_mean_b;
            dcov2 += a_centered * b_centered;
            dvar_x += a_centered * a_centered;
            dvar_y += b_centered * b_centered;
        }
    }
    dcov2 /= nf * nf;
    dvar_x /= nf * nf;
    dvar_y /= nf * nf;

    let denom = (dvar_x * dvar_y).sqrt();
    if denom < 1e-15 { return 0.0; }
    (dcov2 / denom).sqrt().clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Anderson-Darling normality test
// ═══════════════════════════════════════════════════════════════════════════

/// Anderson-Darling test for normality.
///
/// A² = -n - (1/n) Σᵢ (2i-1) [ln Φ(zᵢ) + ln(1-Φ(z_{n+1-i}))]
///
/// More sensitive to tail departures than Kolmogorov-Smirnov.
/// Critical values (adjusted A²*): 0.576 (15%), 0.656 (10%), 0.787 (5%),
/// 0.918 (2.5%), 1.092 (1%).
///
/// Returns adjusted A²* statistic and approximate p-value.
pub fn anderson_darling(data: &[f64]) -> NonparametricResult {
    let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    if n < 3 {
        return NonparametricResult {
            test_name: "Anderson-Darling", statistic: f64::NAN, p_value: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean = sorted.iter().sum::<f64>() / nf;
    let var: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let std = var.sqrt();

    if std < 1e-15 {
        // Constant data is trivially normal (degenerate)
        return NonparametricResult {
            test_name: "Anderson-Darling", statistic: 0.0, p_value: 1.0,
        };
    }

    // Standardize
    let z: Vec<f64> = sorted.iter().map(|x| (x - mean) / std).collect();

    // Compute A²
    let mut sum = 0.0;
    for i in 0..n {
        let phi_z = crate::special_functions::normal_cdf(z[i]);
        let phi_rev = crate::special_functions::normal_cdf(z[n - 1 - i]);
        let ln_phi = phi_z.max(1e-300).ln();
        let ln_1_phi = (1.0 - phi_rev).max(1e-300).ln();
        sum += (2.0 * (i as f64) + 1.0) * (ln_phi + ln_1_phi);
    }
    let a2 = -nf - sum / nf;

    // Adjusted statistic: A²* = A² (1 + 0.75/n + 2.25/n²)
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // Approximate p-value (D'Agostino & Stephens 1986)
    let p = if a2_star < 0.2 {
        1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
    } else if a2_star < 0.34 {
        1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
    } else if a2_star < 0.6 {
        (-0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
    } else if a2_star < 13.0 {
        (-1.2937 - 5.709 * a2_star + 0.0186 * a2_star * a2_star).exp()
    } else {
        0.0
    };

    NonparametricResult {
        test_name: "Anderson-Darling",
        statistic: a2_star,
        p_value: p.clamp(0.0, 1.0),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Friedman test for repeated measures
// ═══════════════════════════════════════════════════════════════════════════

/// Friedman test for k related samples (repeated measures, non-parametric).
///
/// Tests H₀: no difference among k treatments applied to n subjects.
/// Data is ranked within each subject (block), then between-group rank sums are compared.
///
/// Q = [12 / (nk(k+1))] Σⱼ Rⱼ² - 3n(k+1) ~ χ²(k-1)
///
/// `data`: n × k matrix (row-major). Each row is one subject, columns are treatments.
/// `n_subjects`: number of subjects (rows).
/// `n_treatments`: number of treatments (columns).
pub fn friedman_test(data: &[f64], n_subjects: usize, n_treatments: usize) -> NonparametricResult {
    let n = n_subjects;
    let k = n_treatments;

    if k < 2 || n < 2 {
        return NonparametricResult {
            test_name: "Friedman", statistic: f64::NAN, p_value: f64::NAN,
        };
    }

    // Rank within each subject (row)
    let mut rank_sums = vec![0.0; k];
    for i in 0..n {
        let row: Vec<f64> = (0..k).map(|j| data[i * k + j]).collect();
        let ranks = rank(&row);
        for j in 0..k {
            rank_sums[j] += ranks[j];
        }
    }

    let nf = n as f64;
    let kf = k as f64;

    // Friedman Q statistic
    let sum_rj2: f64 = rank_sums.iter().map(|r| r * r).sum();
    let q = (12.0 / (nf * kf * (kf + 1.0))) * sum_rj2 - 3.0 * nf * (kf + 1.0);

    let df = kf - 1.0;
    let p = crate::special_functions::chi2_right_tail_p(q, df);

    NonparametricResult {
        test_name: "Friedman",
        statistic: q,
        p_value: p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Concordance correlation coefficient (Lin 1989)
// ═══════════════════════════════════════════════════════════════════════════

/// Lin's concordance correlation coefficient (CCC).
///
/// Measures agreement between two continuous variables, not just correlation.
/// ρ_c = 2 ρ σ_x σ_y / (σ²_x + σ²_y + (μ_x - μ_y)²)
///
/// Ranges [-1, 1]. CCC = 1 only when both Pearson r = 1 AND the two variables
/// have identical mean and variance (perfect agreement, not just correlation).
///
/// Useful for method comparison studies: does method B agree with gold standard A?
pub fn concordance_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let nf = n as f64;

    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    // Use population variances (divide by n, not n-1) per Lin 1989
    let var_x = sxx / nf;
    let var_y = syy / nf;
    let cov_xy = sxy / nf;

    let denom = var_x + var_y + (mx - my).powi(2);
    if denom < 1e-15 { return f64::NAN; }
    (2.0 * cov_xy / denom).clamp(-1.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        (a - b).abs() < tol
    }

    // ── Ranking ──────────────────────────────────────────────────────────

    #[test]
    fn rank_basic() {
        let r = rank(&[3.0, 1.0, 2.0]);
        assert_eq!(r, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn rank_ties() {
        let r = rank(&[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn rank_with_nan() {
        let r = rank(&[3.0, f64::NAN, 1.0]);
        assert_eq!(r[0], 2.0); // 3.0 is rank 2
        assert!(r[1].is_nan());
        assert_eq!(r[2], 1.0); // 1.0 is rank 1
    }

    // ── Spearman ─────────────────────────────────────────────────────────

    #[test]
    fn spearman_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!(approx(spearman(&x, &y), 1.0, TOL));
    }

    #[test]
    fn spearman_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        assert!(approx(spearman(&x, &y), -1.0, TOL));
    }

    // ── Kendall ──────────────────────────────────────────────────────────

    #[test]
    fn kendall_perfect_concordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx(kendall_tau(&x, &y), 1.0, TOL));
    }

    #[test]
    fn kendall_perfect_discordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(approx(kendall_tau(&x, &y), -1.0, TOL));
    }

    // ── Mann-Whitney U ───────────────────────────────────────────────────

    #[test]
    fn mann_whitney_same_distribution() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let r = mann_whitney_u(&x, &y);
        // Interleaved values → non-significant
        assert!(r.p_value > 0.1, "p={}", r.p_value);
    }

    #[test]
    fn mann_whitney_different_distribution() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0];
        let r = mann_whitney_u(&x, &y);
        assert!(r.p_value < 0.01, "p={}", r.p_value);
    }

    // ── Wilcoxon signed-rank ─────────────────────────────────────────────

    #[test]
    fn wilcoxon_symmetric_around_zero() {
        let diffs = [-3.0, -1.0, 1.0, 3.0, -2.0, 2.0, -0.5, 0.5];
        let r = wilcoxon_signed_rank(&diffs);
        // Symmetric → non-significant
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn wilcoxon_positive_shift() {
        let diffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = wilcoxon_signed_rank(&diffs);
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    // ── Kruskal-Wallis ───────────────────────────────────────────────────

    #[test]
    fn kruskal_wallis_same_groups() {
        let data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let sizes = [3, 3, 3];
        let r = kruskal_wallis(&data, &sizes);
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn kruskal_wallis_different_groups() {
        let data = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
        let sizes = [3, 3, 3];
        let r = kruskal_wallis(&data, &sizes);
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    // ── Dunn's test ──────────────────────────────────────────────────────

    #[test]
    fn dunn_equal_groups_non_significant() {
        // Three identical groups — no pairwise differences
        let data = [1.0, 2.0, 3.0,  1.0, 2.0, 3.0,  1.0, 2.0, 3.0];
        let sizes = [3usize, 3, 3];
        let comparisons = dunn_test(&data, &sizes);
        assert_eq!(comparisons.len(), 3);
        for c in &comparisons {
            assert!(c.p_value > 0.1, "equal groups: p={:.4} should be large", c.p_value);
        }
    }

    #[test]
    fn dunn_separated_groups_detect_differences() {
        // Groups 1 vs 3 clearly differ; groups 1 vs 2 borderline
        let data = [1.0, 2.0, 3.0,  10.0, 11.0, 12.0,  100.0, 101.0, 102.0];
        let sizes = [3usize, 3, 3];
        let comparisons = dunn_test(&data, &sizes);
        // pair (0, 2): groups 1 and 3 are very far apart — should be significant
        let pair_02 = comparisons.iter().find(|c| c.group_i == 0 && c.group_j == 2).unwrap();
        assert!(pair_02.p_value < 0.05,
            "Groups 0 and 2 very separated: p={:.4} should be < 0.05", pair_02.p_value);
    }

    #[test]
    fn dunn_correct_n_pairs() {
        // k=4 → C(4,2) = 6 pairs
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let sizes = [3usize, 3, 3, 3];
        let comparisons = dunn_test(&data, &sizes);
        assert_eq!(comparisons.len(), 6, "k=4 should give 6 pairs");
    }

    // ── KS tests ─────────────────────────────────────────────────────────

    #[test]
    fn ks_normal_data() {
        // Data that's roughly normal
        let data = [-1.5, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0];
        let r = ks_test_normal(&data);
        // Should not reject normality
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn ks_two_sample_same() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let r = ks_test_two_sample(&x, &y);
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn ks_two_sample_different() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0];
        let r = ks_test_two_sample(&x, &y);
        assert!(r.p_value < 0.01, "p={}", r.p_value);
        assert!(approx(r.statistic, 1.0, 1e-10)); // Complete separation
    }

    // ── Bootstrap ────────────────────────────────────────────────────────

    #[test]
    fn bootstrap_mean_ci() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = bootstrap_percentile(&data, |d| d.iter().sum::<f64>() / d.len() as f64,
            1000, 0.05, 42);
        // Mean is 5.5
        assert!(approx(r.estimate, 5.5, 1e-10));
        // CI should contain the mean
        assert!(r.ci_lower < 5.5 && r.ci_upper > 5.5,
            "CI [{}, {}] should contain 5.5", r.ci_lower, r.ci_upper);
        // CI should be reasonable (not too wide, not too narrow)
        assert!(r.ci_lower > 2.0 && r.ci_upper < 9.0,
            "CI [{}, {}] should be reasonable", r.ci_lower, r.ci_upper);
    }

    #[test]
    fn bootstrap_se_positive() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = bootstrap_percentile(&data, |d| d.iter().sum::<f64>() / d.len() as f64,
            500, 0.05, 123);
        assert!(r.se > 0.0, "SE should be positive");
    }

    // ── Permutation test ─────────────────────────────────────────────────

    #[test]
    fn permutation_same_distribution() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.5, 2.5, 3.5, 4.5, 5.5];
        let r = permutation_test_mean_diff(&x, &y, 999, 42);
        assert!(r.p_value > 0.1, "p={}", r.p_value);
    }

    #[test]
    fn permutation_different_distribution() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [100.0, 101.0, 102.0, 103.0, 104.0];
        let r = permutation_test_mean_diff(&x, &y, 999, 42);
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    // ── KDE ──────────────────────────────────────────────────────────────

    #[test]
    fn kde_gaussian_basic() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let pts = [0.0, 2.0, 4.0];
        let density = kde(&data, &pts, KernelType::Gaussian, Some(1.0));
        // Density at center (2.0) should be highest
        assert!(density[1] > density[0], "center should be higher");
        assert!(density[1] > density[2] - 0.01, "center should be ≥ edge");
        // All densities should be positive
        for &d in &density { assert!(d > 0.0); }
    }

    #[test]
    fn kde_integrates_roughly_to_one() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // Evaluate at many points and approximate integral
        let pts: Vec<f64> = (-5..=15).map(|i| i as f64 * 0.5).collect();
        let density = kde(&data, &pts, KernelType::Gaussian, Some(1.0));
        let integral: f64 = density.iter().sum::<f64>() * 0.5; // trapezoidal approx
        assert!((integral - 1.0).abs() < 0.15, "integral={} should be ≈ 1", integral);
    }

    #[test]
    fn kde_epanechnikov() {
        let data = [0.0, 1.0, 2.0];
        let pts = [0.0, 100.0];
        let density = kde(&data, &pts, KernelType::Epanechnikov, Some(1.0));
        assert!(density[0] > 0.0);
        assert_eq!(density[1], 0.0); // Epanechnikov has compact support
    }

    // ── KDE via FFT ──────────────────────────────────────────────────────

    #[test]
    fn kde_fft_integrates_to_one() {
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (grid, density) = kde_fft(&data, 512, Some(1.0));
        let dx = grid[1] - grid[0];
        let integral: f64 = density.iter().sum::<f64>() * dx;
        assert!((integral - 1.0).abs() < 0.05,
            "KDE-FFT integral={integral} should be ~1.0");
    }

    #[test]
    fn kde_fft_peak_at_data_center() {
        let data: Vec<f64> = vec![5.0; 50]; // all data at 5.0
        let (grid, density) = kde_fft(&data, 256, Some(0.5));
        // Peak should be near x=5.0
        let peak_idx = density.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap().0;
        assert!((grid[peak_idx] - 5.0).abs() < 0.5,
            "Peak at x={} should be near 5.0", grid[peak_idx]);
    }

    #[test]
    fn kde_fft_matches_direct() {
        // FFT and direct KDE should produce similar results
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (grid, fft_density) = kde_fft(&data, 256, Some(1.0));

        let direct_density = kde(&data, &grid, KernelType::Gaussian, Some(1.0));

        // Compare at interior points (edges may differ due to boundary handling)
        let n = grid.len();
        let mut max_diff: f64 = 0.0;
        for i in n / 4..3 * n / 4 {
            max_diff = max_diff.max((fft_density[i] - direct_density[i]).abs());
        }
        assert!(max_diff < 0.01,
            "FFT vs direct max difference={max_diff} in interior should be small");
    }

    #[test]
    fn kde_fft_bimodal() {
        // Two clusters: should show two peaks
        let mut data: Vec<f64> = Vec::new();
        for _ in 0..50 { data.push(0.0); }
        for _ in 0..50 { data.push(10.0); }
        let (grid, density) = kde_fft(&data, 512, Some(0.5));

        // Find local maxima
        let mut peaks = Vec::new();
        for i in 1..density.len() - 1 {
            if density[i] > density[i - 1] && density[i] > density[i + 1] {
                peaks.push(grid[i]);
            }
        }
        assert!(peaks.len() >= 2, "Should detect 2 peaks in bimodal data, found {}", peaks.len());
    }

    // ── Runs test ────────────────────────────────────────────────────────

    #[test]
    fn runs_test_random_sequence() {
        let seq = [true, false, true, false, true, false, true, false, true, false];
        let r = runs_test(&seq);
        // Perfect alternation has many runs — significant non-randomness
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    #[test]
    fn runs_test_clustered() {
        let seq = [true, true, true, true, true, false, false, false, false, false];
        let r = runs_test(&seq);
        // Only 2 runs — too few, significant non-randomness
        assert!(approx(r.statistic, 2.0, 1e-10));
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    // ── Sign test ────────────────────────────────────────────────────────

    #[test]
    fn sign_test_symmetric() {
        let data = [-2.0, -1.0, 1.0, 2.0, -3.0, 3.0, -0.5, 0.5];
        let r = sign_test(&data, 0.0);
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn sign_test_shifted() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = sign_test(&data, 0.0);
        assert!(r.p_value < 0.01, "p={}", r.p_value);
    }

    // ── Level-spacing r-statistic ─────────────────────────────────────────

    #[test]
    fn level_spacing_r_stat_gue_like() {
        // GUE-like spacings: level repulsion — gaps avoid being very small.
        // Simulate Wigner-Dyson by using spacings drawn from p(s) ~ s·exp(-πs²/4).
        // Here we use a hand-crafted sequence with moderate, non-bunching gaps.
        // Levels: 0, 1.2, 2.5, 3.6, 5.1, 6.3, 7.8, 9.0, 10.4, 11.7, 13.1, 14.5
        // Gaps:   1.2  1.3  1.1  1.5  1.2  1.5  1.2  1.4  1.3  1.4  1.4
        // These gaps are moderate (no very small or very large), consistent with GUE.
        let levels = [0.0, 1.2, 2.5, 3.6, 5.1, 6.3, 7.8, 9.0, 10.4, 11.7, 13.1, 14.5f64];
        let r = level_spacing_r_stat(&levels);
        // GUE expected: r ≈ 0.536. This synthetic sequence should give r close to GUE.
        assert!(r > 0.46, "r={r:.3} should be GUE-like (> 0.46), not Poisson-like (0.386)");
        assert!(r < 1.0, "r={r:.3} should be in [0,1]");
    }

    #[test]
    fn level_spacing_r_stat_poisson_like() {
        // Poisson-like spacings: gaps can be very small (no level repulsion).
        // Simulate by clustering: some very small gaps, some very large gaps.
        // Levels: 0, 0.1, 0.2, 2.0, 2.1, 4.5, 4.6, 4.7, 8.0, 8.1, 12.0, 12.1
        // Gaps:   0.1 0.1  1.8  0.1  2.4  0.1  0.1  3.3  0.1  3.9  0.1
        // Many small gaps (0.1) and some large gaps → Poisson-like (r < GUE).
        let levels = [0.0, 0.1, 0.2, 2.0, 2.1, 4.5, 4.6, 4.7, 8.0, 8.1, 12.0, 12.1f64];
        let r = level_spacing_r_stat(&levels);
        // Poisson expected: r ≈ 0.386. This clustered sequence should give r well below GUE.
        assert!(r < 0.46, "r={r:.3} should be Poisson-like (< 0.46), not GUE-like (0.536)");
        assert!(r >= 0.0, "r={r:.3} should be non-negative");
    }

    #[test]
    fn level_spacing_r_stat_too_short() {
        assert!(level_spacing_r_stat(&[]).is_nan());
        assert!(level_spacing_r_stat(&[1.0]).is_nan());
        assert!(level_spacing_r_stat(&[1.0, 2.0]).is_nan());
        // 3 values: 2 gaps, 1 r-value — should work
        let r = level_spacing_r_stat(&[0.0, 1.0, 3.0]);
        assert!(!r.is_nan());
    }

    // ── Regression: KS test standardized for shifted data ──────────────
    // The old ks_test_normal always tested against N(0,1), so data with
    // mean=100 would always reject. ks_test_normal_standardized fixes this.
    #[test]
    fn ks_standardized_shifted_normal_regression() {
        // Generate N(100, 4) data — clearly normal but far from N(0,1)
        let mut rng = crate::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..200).map(|_| {
            crate::rng::sample_normal(&mut rng, 100.0, 2.0)
        }).collect();

        // Old function: tests against N(0,1) — MUST reject (data is nowhere near N(0,1))
        let old_result = ks_test_normal(&data);
        assert!(old_result.p_value < 0.001,
            "ks_test_normal on N(100,4) data should reject (p={})", old_result.p_value);

        // New function: standardizes first — should NOT reject (data is genuinely normal)
        let new_result = ks_test_normal_standardized(&data);
        assert!(new_result.p_value > 0.01,
            "ks_test_normal_standardized on N(100,4) data should not reject (p={})", new_result.p_value);
    }

    #[test]
    fn ks_standardized_non_normal_rejects() {
        // Exponential data — clearly non-normal (right-skewed)
        let mut rng = crate::rng::Xoshiro256::new(99);
        let data: Vec<f64> = (0..500).map(|_| {
            crate::rng::sample_exponential(&mut rng, 1.0)
        }).collect();
        let r = ks_test_normal_standardized(&data);
        assert!(r.p_value < 0.05,
            "Exponential data should reject normality (p={})", r.p_value);
    }

    #[test]
    fn ks_standardized_degenerate() {
        // All identical — degenerate case
        let data = vec![5.0; 100];
        let r = ks_test_normal_standardized(&data);
        assert_eq!(r.statistic, 0.0);
        assert_eq!(r.p_value, 1.0);
    }

    // ── Shapiro-Wilk ────────────────────────────────────────────────────

    #[test]
    fn shapiro_wilk_normal_data_passes() {
        // Generate normal data — should not reject
        let mut rng = crate::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..100).map(|_| {
            crate::rng::sample_normal(&mut rng, 0.0, 1.0)
        }).collect();
        let r = shapiro_wilk(&data);
        assert!(r.statistic > 0.9, "W={} should be near 1 for normal data", r.statistic);
        assert!(r.p_value > 0.01, "p={} should not reject normal data", r.p_value);
    }

    #[test]
    fn shapiro_wilk_uniform_rejects() {
        // Uniform data — clearly not normal
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let r = shapiro_wilk(&data);
        assert!(r.statistic < 0.98, "W={} should be < 1 for non-normal", r.statistic);
        assert!(r.p_value < 0.05, "p={} should reject uniform data", r.p_value);
    }

    #[test]
    fn shapiro_wilk_exponential_rejects() {
        // Exponential data — right-skewed, should reject normality
        let mut rng = crate::rng::Xoshiro256::new(99);
        let data: Vec<f64> = (0..200).map(|_| {
            crate::rng::sample_exponential(&mut rng, 1.0)
        }).collect();
        let r = shapiro_wilk(&data);
        assert!(r.p_value < 0.05, "p={} should reject exponential data", r.p_value);
    }

    #[test]
    fn shapiro_wilk_small_n() {
        // n=3: minimum valid size
        let r = shapiro_wilk(&[1.0, 2.0, 3.0]);
        assert!(r.statistic > 0.0 && r.statistic <= 1.0,
            "W={} should be in (0, 1]", r.statistic);
        assert!(!r.p_value.is_nan(), "p should not be NaN for n=3");

        // n=2: too small
        let r = shapiro_wilk(&[1.0, 2.0]);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn shapiro_wilk_degenerate() {
        let r = shapiro_wilk(&[5.0; 50]);
        assert_eq!(r.statistic, 1.0);
        assert_eq!(r.p_value, 1.0);
    }

    // ── D'Agostino-Pearson ──────────────────────────────────────────────

    #[test]
    fn dagostino_normal_passes() {
        // Use a large sample to reduce sampling variability in skew/kurtosis
        let mut rng = crate::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..2000).map(|_| {
            crate::rng::sample_normal(&mut rng, 0.0, 1.0)
        }).collect();
        let r = dagostino_pearson(&data);
        assert!(r.p_value > 0.01, "p={} should not reject normal data (K2={})", r.p_value, r.statistic);
    }

    #[test]
    fn dagostino_exponential_rejects() {
        let mut rng = crate::rng::Xoshiro256::new(99);
        let data: Vec<f64> = (0..500).map(|_| {
            crate::rng::sample_exponential(&mut rng, 1.0)
        }).collect();
        let r = dagostino_pearson(&data);
        assert!(r.p_value < 0.05, "p={} should reject exponential data", r.p_value);
    }

    #[test]
    fn dagostino_small_n() {
        let r = dagostino_pearson(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(r.statistic.is_nan(), "n < 8 should return NaN");
    }

    // ── Pairwise correlation coefficients ─────────────────────────────────

    #[test]
    fn pearson_r_perfect_positive() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * 2.0 + 3.0).collect();
        assert!((pearson_r(&x, &y) - 1.0).abs() < 1e-10, "perfect linear: r=1");
    }

    #[test]
    fn pearson_r_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((pearson_r(&x, &y) + 1.0).abs() < 1e-10, "anti-correlated: r=-1");
    }

    #[test]
    fn pearson_r_zero_variance() {
        let x = vec![2.0; 5];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(pearson_r(&x, &y).is_nan(), "constant x: r=NaN");
    }

    #[test]
    fn phi_coefficient_perfect() {
        // Perfect agreement: x=y
        let x = vec![0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        assert!((phi_coefficient(&x, &y) - 1.0).abs() < 1e-10, "identical binary: phi=1");
    }

    #[test]
    fn phi_coefficient_zero() {
        // Independent: 2x2 balanced
        let x = vec![0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        assert!(phi_coefficient(&x, &y).abs() < 1e-10, "independent binary: phi=0");
    }

    #[test]
    fn point_biserial_monotone() {
        // Higher binary group has higher continuous values → positive r_pb
        let binary    = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let continuous = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let r = point_biserial(&binary, &continuous);
        assert!(r > 0.9, "group 1 >> group 0: r_pb should be high, got {r}");
    }

    // ── Partial correlation ───────────────────────────────────────────────

    #[test]
    fn partial_corr_no_covariates_equals_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let pc = partial_correlation(&x, &y, &[]);
        let r  = pearson_r(&x, &y);
        assert!((pc - r).abs() < 1e-12, "no covariates: partial_corr = pearson");
    }

    #[test]
    fn partial_corr_removes_confound() {
        // X = Z + noise, Y = Z + noise — raw Pearson r is high.
        // After controlling for Z, partial r should be near 0.
        let n = 50;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut rng = 42u64;
        let x: Vec<f64> = z.iter().map(|&zi| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            zi + (rng as f64 / u64::MAX as f64 - 0.5) * 2.0
        }).collect();
        let y: Vec<f64> = z.iter().map(|&zi| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            zi + (rng as f64 / u64::MAX as f64 - 0.5) * 2.0
        }).collect();

        let raw_r = pearson_r(&x, &y);
        let partial_r = partial_correlation(&x, &y, &[z.as_slice()]);

        // Raw r should be high (strong confounder)
        assert!(raw_r > 0.95, "confounded raw r should be high, got {raw_r:.4}");
        // Partial r controlling for Z should be near 0
        assert!(partial_r.abs() < 0.3,
            "after removing confound Z: partial_r={partial_r:.4} should be near 0");
    }

    #[test]
    fn partial_corr_perfect_linear_with_covariate() {
        // X = t, Y = 2t, Z = 3t — after removing Z (a linear function), X and Y
        // should still be perfectly correlated (they're proportional with same factor).
        let n = 10;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x: Vec<f64> = t.clone();
        let y: Vec<f64> = t.iter().map(|&v| 2.0 * v).collect();
        let z: Vec<f64> = t.iter().map(|&v| 3.0 * v).collect();
        let pc = partial_correlation(&x, &y, &[z.as_slice()]);
        // After regressing z out of x and y (both collapse to residuals near 0),
        // the partial correlation is undefined but we just verify no panic
        assert!(pc.is_finite() || pc.is_nan());
    }

    // ── Anderson-Darling normality test ──────────────────────────────────

    #[test]
    fn anderson_darling_normal_data_passes() {
        // Data drawn from N(0,1) by deterministic Box-Muller-like spacing
        // Using quantiles of a fine grid (known normal) — p should be > 0.05
        // Generate 50 normal quantiles: Φ⁻¹((i-0.375)/(n+0.25)) for i=1..n
        let n = 50usize;
        let data: Vec<f64> = (1..=n).map(|i| {
            crate::special_functions::normal_quantile((i as f64 - 0.375) / (n as f64 + 0.25))
        }).collect();
        let r = anderson_darling(&data);
        assert_eq!(r.test_name, "Anderson-Darling");
        assert!(r.statistic.is_finite(), "A²* should be finite");
        assert!(r.p_value > 0.05,
            "Normal quantiles should pass AD test, p={:.4}", r.p_value);
    }

    #[test]
    fn anderson_darling_uniform_rejects() {
        // Uniform data should fail the normality test (p < 0.05)
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let r = anderson_darling(&data);
        assert!(r.p_value < 0.05,
            "Uniform data should fail AD normality test, p={:.4}", r.p_value);
    }

    #[test]
    fn anderson_darling_too_short_returns_nan() {
        let r = anderson_darling(&[1.0, 2.0]);
        assert!(r.statistic.is_nan() && r.p_value.is_nan());
    }

    #[test]
    fn anderson_darling_constant_returns_p1() {
        // Constant data is "trivially normal" (degenerate distribution)
        let data = vec![5.0; 20];
        let r = anderson_darling(&data);
        assert!(r.p_value >= 0.99, "constant data p={:.4} should be ~1", r.p_value);
    }

    // ── Friedman test for repeated measures ──────────────────────────────

    #[test]
    fn friedman_clear_treatment_effect() {
        // 5 subjects, 3 treatments. Treatment 3 clearly dominates.
        // Data[i][j] = base[i] + treatment[j]
        // Expected: Q large, p < 0.05
        let treatments = [0.0, 1.0, 5.0]; // big difference between T1 and T3
        let n = 5;
        let k = 3;
        let mut data = vec![0.0; n * k];
        for i in 0..n {
            let base = i as f64;
            for j in 0..k {
                data[i * k + j] = base + treatments[j];
            }
        }
        let r = friedman_test(&data, n, k);
        assert_eq!(r.test_name, "Friedman");
        assert!(r.statistic > 5.0, "Q={:.4} should be large for clear effect", r.statistic);
        assert!(r.p_value < 0.05, "p={:.4} should be significant", r.p_value);
    }

    #[test]
    fn friedman_no_treatment_effect() {
        // All treatments have the same expected rank within each subject
        // Data: all cells equal → tied ranks → Q = 0
        let n = 5;
        let k = 3;
        let data = vec![1.0; n * k]; // all same value → tied ranks
        let r = friedman_test(&data, n, k);
        // Q should be 0 or near 0 (all tied → no variation in rank sums)
        assert!(r.statistic.abs() < 1e-6, "Q={:.6} should be ~0 for tied data", r.statistic);
        assert!(r.p_value > 0.5, "p={:.4} should be large for no effect", r.p_value);
    }

    #[test]
    fn friedman_minimum_size_returns_result() {
        // k<2 or n<2 → NaN
        let r = friedman_test(&[1.0], 1, 1);
        assert!(r.statistic.is_nan());
        // k=2, n=2 → should compute
        let data = vec![1.0, 2.0, 2.0, 1.0]; // 2 subjects × 2 treatments
        let r2 = friedman_test(&data, 2, 2);
        assert!(r2.statistic.is_finite());
    }
}
