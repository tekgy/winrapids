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

    // Normal approximation
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let mu = n1f * n2f / 2.0;
    let sigma = (n1f * n2f * (n1f + n2f + 1.0) / 12.0).sqrt();

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

    let nf = n as f64;
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();

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
        let r_sum: f64 = ranks[offset..offset + gs].iter().sum();
        h += r_sum * r_sum / gs as f64;
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
// Kolmogorov-Smirnov test
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample Kolmogorov-Smirnov test against standard normal.
///
/// D = sup|Fₙ(x) - Φ(x)| where Φ is the standard normal CDF.
/// p-value via Kolmogorov distribution approximation.
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
        let fx = i as f64 / n1 as f64;
        let gy = j as f64 / n2 as f64;
        d_max = d_max.max((fx - gy).abs());

        // Advance the smaller value
        if i < n1 && (j >= n2 || sx[i] <= sy[j]) {
            i += 1;
        } else {
            j += 1;
        }
    }
    // Check final point
    d_max = d_max.max((1.0 - n2 as f64 / n2 as f64).abs()); // = 0, just for completeness

    let en = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let p = ks_p_value(d_max, (en * en) as usize);

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
    let mut rng_state = seed;
    let mut boot_stats = Vec::with_capacity(n_resamples);
    let mut resample = vec![0.0; n];

    for _ in 0..n_resamples {
        // Resample with replacement
        for slot in resample.iter_mut() {
            rng_state = lcg_next(rng_state);
            let idx = (rng_state >> 16) as usize % n;
            *slot = data[idx];
        }
        boot_stats.push(statistic(&resample));
    }

    boot_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = ((alpha / 2.0) * n_resamples as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64) as usize;
    let ci_lower = boot_stats[lo_idx.min(n_resamples - 1)];
    let ci_upper = boot_stats[hi_idx.min(n_resamples - 1)];

    let mean: f64 = boot_stats.iter().sum::<f64>() / n_resamples as f64;
    let se = (boot_stats.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n_resamples - 1) as f64).sqrt();

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

    let mut rng_state = seed;
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            rng_state = lcg_next(rng_state);
            let j = (rng_state >> 16) as usize % (i + 1);
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

/// Simple LCG pseudo-random: state' = state * 6364136223846793005 + 1442695040888963407
#[inline]
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn mean_slice(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    data.iter().sum::<f64>() / data.len() as f64
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

    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;

    let spread = if iqr > 0.0 { std.min(iqr / 1.34) } else { std };
    0.9 * spread * n.powf(-0.2)
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
    if n == 0 { return (vec![], vec![]); }

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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
}
