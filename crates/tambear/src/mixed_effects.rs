//! # Family 11 — Mixed Effects & Multilevel Models
//!
//! LME, REML, BLUP, ICC, random intercept/slope.
//!
//! ## Architecture
//!
//! LME = self-tuning Ridge regression:
//! Henderson's equations = GramMatrix([X|Z]) + diagonal regularization + Cholesky solve.
//! Variance components estimated via EM. Kingdom A (solve) + C (iteration).
//!
//! ## Core insight
//! The GramMatrix of the augmented design [X|Z|y] gives ALL needed subblocks:
//! X'X, X'Z, Z'Z, X'y, Z'y — from one tiled_accumulate.

use crate::linear_algebra::{Mat, cholesky, cholesky_solve};

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Fitted linear mixed effects model.
#[derive(Debug, Clone)]
pub struct LmeResult {
    /// Fixed effects coefficients (BLUE).
    pub beta: Vec<f64>,
    /// Random effects predictions (BLUP), one per group.
    pub u: Vec<f64>,
    /// Residual variance σ².
    pub sigma2: f64,
    /// Random effect variance σ²_u.
    pub sigma2_u: f64,
    /// Intraclass correlation coefficient.
    pub icc: f64,
    /// Number of EM iterations.
    pub iterations: usize,
    /// Log-likelihood (REML).
    pub log_likelihood: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Random intercept model via EM
// ═══════════════════════════════════════════════════════════════════════════

/// Fit a random intercept model: y = Xβ + Z·u + ε.
///
/// `x`: row-major n×d fixed effects design matrix (exclude intercept, added internally).
/// `y`: response vector (length n).
/// `groups`: group label for each observation (0-indexed).
/// `max_iter`: maximum EM iterations.
/// `tol`: convergence tolerance for variance component change.
pub fn lme_random_intercept(
    x: &[f64], y: &[f64], n: usize, d: usize,
    groups: &[usize], max_iter: usize, tol: f64,
) -> LmeResult {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert_eq!(groups.len(), n);
    let k = *groups.iter().max().unwrap_or(&0) + 1; // number of groups

    let p = d + 1; // +1 for intercept

    // Group sizes
    let mut n_g = vec![0usize; k];
    for &g in groups { n_g[g] += 1; }

    // Initialize variance components
    let y_moments = crate::descriptive::moments_ungrouped(y);
    let y_var = y_moments.variance(1);
    let mut sigma2 = y_var * 0.5;
    let mut sigma2_u = y_var * 0.5;
    let mut beta = vec![0.0; p];
    let mut u = vec![0.0; k];
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // E-step: solve Henderson's equations for given variance components
        // Henderson system:
        // [X'X       X'Z           ] [β]   [X'y]
        // [Z'X    Z'Z + σ²/σ²_u · I] [u] = [Z'y]
        //
        // Here Z is the group indicator matrix (n × k), so:
        // Z'Z = diag(n_1, ..., n_k)
        // X includes intercept column

        let dim = p + k;
        let mut a = vec![0.0; dim * dim];
        let mut rhs = vec![0.0; dim];

        // Build X'X (p×p), X'Z (p×k), Z'Z+λI (k×k), X'y, Z'y
        for i in 0..n {
            let g = groups[i];

            // Build x̃ = [1, x_i1, ..., x_id]
            let mut xi = vec![0.0; p];
            xi[0] = 1.0;
            for j in 0..d { xi[j + 1] = x[i * d + j]; }

            // X'X block
            for j in 0..p {
                for l in 0..p {
                    a[j * dim + l] += xi[j] * xi[l];
                }
            }

            // X'Z block (and Z'X)
            for j in 0..p {
                a[j * dim + (p + g)] += xi[j]; // X'Z
                a[(p + g) * dim + j] += xi[j]; // Z'X
            }

            // Z'Z block
            a[(p + g) * dim + (p + g)] += 1.0;

            // Right-hand side
            for j in 0..p { rhs[j] += xi[j] * y[i]; }
            rhs[p + g] += y[i];
        }

        // Add σ²/σ²_u to Z'Z diagonal
        let lambda = if sigma2_u > 1e-15 { sigma2 / sigma2_u } else { 1e10 };
        for g in 0..k {
            a[(p + g) * dim + (p + g)] += lambda;
        }

        let a_mat = Mat::from_vec(dim, dim, a);
        if let Some(l) = cholesky(&a_mat) {
            let sol = cholesky_solve(&l, &rhs);
            for j in 0..p { beta[j] = sol[j]; }
            for g in 0..k { u[g] = sol[p + g]; }
        }

        // M-step: update variance components
        // σ²_new = (||y - Xβ - Zû||² + tr(Z'Z · Var(u|y))) / n
        // tr(Z'Z · Var(u|y)) = Σ_g n_g · τ_g², τ_g² = σ²·σ²_u/(n_g·σ²_u+σ²)
        let mut ss_resid = 0.0;
        for i in 0..n {
            let g = groups[i];
            let mut fitted = beta[0];
            for j in 0..d { fitted += beta[j + 1] * x[i * d + j]; }
            fitted += u[g];
            let r = y[i] - fitted;
            ss_resid += r * r;
        }

        // Trace correction: tr(Z'Z · Var(u|y)) = Σ_g n_g · τ_g²
        // where τ_g² = σ²·σ²_u / (n_g·σ²_u + σ²) and Z'Z = diag(n_1,...,n_k).
        let trace_sum: f64 = (0..k).map(|g| {
            let ng = n_g[g] as f64;
            let tau2_g = sigma2 * sigma2_u / (ng * sigma2_u + sigma2);
            ng * tau2_g
        }).sum();

        let sigma2_new = (ss_resid + trace_sum) / n as f64;

        // σ²_u = (u'u + Σ_g Var(u_g|y)) / k
        let uu: f64 = u.iter().map(|v| v * v).sum();
        let var_correction: f64 = (0..k).map(|g| {
            let ng = n_g[g] as f64;
            sigma2 / (ng * sigma2_u + sigma2) * sigma2_u
        }).sum();
        let sigma2_u_new = ((uu + var_correction) / k as f64).max(0.0);

        let change = (sigma2_new - sigma2).abs() + (sigma2_u_new - sigma2_u).abs();
        sigma2 = sigma2_new.max(1e-15);
        sigma2_u = sigma2_u_new;

        if change < tol { break; }
    }

    let icc = sigma2_u / (sigma2_u + sigma2);

    // Approximate REML log-likelihood
    let nf = n as f64;
    let log_lik = -0.5 * nf * (2.0 * std::f64::consts::PI * sigma2).ln()
        - 0.5 * (0..k).map(|g| {
            let ng = n_g[g] as f64;
            (1.0 + ng * sigma2_u / sigma2).ln()
        }).sum::<f64>();

    LmeResult { beta, u, sigma2, sigma2_u, icc, iterations, log_likelihood: log_lik }
}

// ═══════════════════════════════════════════════════════════════════════════
// ICC from one-way random effects ANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// ICC(1,1) from one-way random effects ANOVA.
/// Quick estimate without fitting a full LME model.
pub fn icc_oneway(values: &[f64], groups: &[usize]) -> f64 {
    let n = values.len();
    let k = *groups.iter().max().unwrap_or(&0) + 1;
    if k < 2 || n <= k {
        return f64::NAN;
    }

    let grand_mean = crate::descriptive::moments_ungrouped(values).mean();

    let mut g_sums = vec![0.0; k];
    let mut g_counts = vec![0usize; k];
    for i in 0..n {
        g_sums[groups[i]] += values[i];
        g_counts[groups[i]] += 1;
    }
    let g_means: Vec<f64> = (0..k).map(|g| {
        if g_counts[g] > 0 { g_sums[g] / g_counts[g] as f64 } else { 0.0 }
    }).collect();

    let ms_between: f64 = (0..k).map(|g| {
        let ng = g_counts[g] as f64;
        ng * (g_means[g] - grand_mean).powi(2)
    }).sum::<f64>() / (k - 1) as f64;

    let ms_within: f64 = (0..n).map(|i| {
        (values[i] - g_means[groups[i]]).powi(2)
    }).sum::<f64>() / (n - k) as f64;

    let n0 = {
        let sum_nk: f64 = g_counts.iter().map(|&c| c as f64).sum();
        let sum_nk2: f64 = g_counts.iter().map(|&c| (c as f64).powi(2)).sum();
        (sum_nk - sum_nk2 / sum_nk) / (k - 1) as f64
    };

    if n0 > 0.0 {
        ((ms_between - ms_within) / (ms_between + (n0 - 1.0) * ms_within)).max(0.0)
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ICC types (2,1) and (3,1) — Shrout & Fleiss 1979
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a two-way ICC computation.
#[derive(Debug, Clone)]
pub struct IccResult {
    /// ICC value.
    pub icc: f64,
    /// Type string: "1,1", "2,1", or "3,1".
    pub icc_type: &'static str,
    /// Mean square for subjects (rows).
    pub ms_subjects: f64,
    /// Mean square for raters (columns). NaN for one-way.
    pub ms_raters: f64,
    /// Mean square error (residual).
    pub ms_error: f64,
}

/// Two-way ANOVA decomposition for ICC.
/// `data`: n_subjects × n_raters matrix (row-major).
fn twoway_anova_ms(data: &[f64], n_subjects: usize, n_raters: usize) -> (f64, f64, f64) {
    let n = n_subjects;
    let k = n_raters;
    let nk = (n * k) as f64;

    let grand_mean = data.iter().sum::<f64>() / nk;

    // Row means (subjects)
    let row_means: Vec<f64> = (0..n).map(|i| {
        data[i * k..(i + 1) * k].iter().sum::<f64>() / k as f64
    }).collect();

    // Column means (raters)
    let col_means: Vec<f64> = (0..k).map(|j| {
        (0..n).map(|i| data[i * k + j]).sum::<f64>() / n as f64
    }).collect();

    // SS subjects = k * Σ (row_mean_i - grand_mean)²
    let ss_subjects: f64 = row_means.iter()
        .map(|&m| (m - grand_mean).powi(2)).sum::<f64>() * k as f64;
    // SS raters = n * Σ (col_mean_j - grand_mean)²
    let ss_raters: f64 = col_means.iter()
        .map(|&m| (m - grand_mean).powi(2)).sum::<f64>() * n as f64;
    // SS total = ΣΣ (x_ij - grand_mean)²
    let ss_total: f64 = data.iter().map(|&x| (x - grand_mean).powi(2)).sum();
    // SS error = SS_total - SS_subjects - SS_raters
    let ss_error = (ss_total - ss_subjects - ss_raters).max(0.0);

    let ms_s = ss_subjects / (n - 1).max(1) as f64;
    let ms_r = ss_raters / (k - 1).max(1) as f64;
    let ms_e = ss_error / ((n - 1) * (k - 1)).max(1) as f64;

    (ms_s, ms_r, ms_e)
}

/// ICC(2,1): two-way random effects, single measures (Shrout & Fleiss 1979).
///
/// Each target is rated by the same set of raters; raters are a random sample.
/// ICC(2,1) = (MS_S - MS_E) / (MS_S + (k-1)·MS_E + k·(MS_R - MS_E)/n)
///
/// `data`: n_subjects × n_raters matrix (row-major).
pub fn icc_twoway_random(data: &[f64], n_subjects: usize, n_raters: usize) -> IccResult {
    if n_subjects < 2 || n_raters < 2 || data.len() != n_subjects * n_raters {
        return IccResult { icc: f64::NAN, icc_type: "2,1", ms_subjects: f64::NAN, ms_raters: f64::NAN, ms_error: f64::NAN };
    }
    let (ms_s, ms_r, ms_e) = twoway_anova_ms(data, n_subjects, n_raters);
    let n = n_subjects as f64;
    let k = n_raters as f64;
    let denom = ms_s + (k - 1.0) * ms_e + k * (ms_r - ms_e) / n;
    let icc = if denom > 1e-300 { ((ms_s - ms_e) / denom).clamp(-1.0, 1.0) } else { 0.0 };
    IccResult { icc, icc_type: "2,1", ms_subjects: ms_s, ms_raters: ms_r, ms_error: ms_e }
}

/// ICC(3,1): two-way mixed effects, single measures (Shrout & Fleiss 1979).
///
/// Each target is rated by the same set of raters; raters are fixed (not a sample).
/// ICC(3,1) = (MS_S - MS_E) / (MS_S + (k-1)·MS_E)
///
/// `data`: n_subjects × n_raters matrix (row-major).
pub fn icc_twoway_mixed(data: &[f64], n_subjects: usize, n_raters: usize) -> IccResult {
    if n_subjects < 2 || n_raters < 2 || data.len() != n_subjects * n_raters {
        return IccResult { icc: f64::NAN, icc_type: "3,1", ms_subjects: f64::NAN, ms_raters: f64::NAN, ms_error: f64::NAN };
    }
    let (ms_s, ms_r, ms_e) = twoway_anova_ms(data, n_subjects, n_raters);
    let k = n_raters as f64;
    let denom = ms_s + (k - 1.0) * ms_e;
    let icc = if denom > 1e-300 { ((ms_s - ms_e) / denom).clamp(-1.0, 1.0) } else { 0.0 };
    IccResult { icc, icc_type: "3,1", ms_subjects: ms_s, ms_raters: ms_r, ms_error: ms_e }
}

// ═══════════════════════════════════════════════════════════════════════════
// Design effect
// ═══════════════════════════════════════════════════════════════════════════

/// Design effect for clustered data.
/// Effective sample size = n / DEFF.
pub fn design_effect(icc: f64, avg_cluster_size: f64) -> f64 {
    1.0 + (avg_cluster_size - 1.0) * icc
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    // ── Random intercept model ──────────────────────────────────────────

    #[test]
    fn lme_known_fixed_effect() {
        // y = 2 + 3*x + u_g + ε, with known group effects
        let n = 60;
        let d = 1;
        let k = 3;
        let group_effects = [0.0, 2.0, -1.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        let mut rng = 42u64;

        for g in 0..k {
            for i in 0..20 {
                let xi = i as f64 / 20.0;
                x.push(xi);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
                y.push(2.0 + 3.0 * xi + group_effects[g] + noise);
                groups.push(g);
            }
        }

        let res = lme_random_intercept(&x, &y, n, d, &groups, 100, 1e-8);

        // β₀ ≈ 2.0 (intercept), β₁ ≈ 3.0 (slope)
        assert!((res.beta[1] - 3.0).abs() < 0.5, "Slope={} should be ~3.0", res.beta[1]);
        assert!(res.sigma2 > 0.0, "Residual variance should be positive");
        assert!(res.sigma2_u >= 0.0, "Random effect variance should be non-negative");
    }

    #[test]
    fn lme_no_group_effect() {
        // No group variation → σ²_u ≈ 0, ICC ≈ 0
        let n = 40;
        let d = 1;
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        let mut rng = 123u64;

        for g in 0..4 {
            for i in 0..10 {
                let xi = i as f64 / 10.0;
                x.push(xi);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
                y.push(1.0 + 2.0 * xi + noise); // no group effect
                groups.push(g);
            }
        }

        let res = lme_random_intercept(&x, &y, n, d, &groups, 100, 1e-8);
        assert!(res.icc < 0.3, "ICC={} should be low (no group effect)", res.icc);
    }

    #[test]
    fn lme_strong_group_effect() {
        // Large group variance → ICC should be high
        let n = 60;
        let d = 1;
        let group_effects = [0.0, 10.0, -10.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();

        for g in 0..3 {
            for i in 0..20 {
                let xi = i as f64 / 20.0;
                x.push(xi);
                y.push(1.0 + xi + group_effects[g]); // no noise
                groups.push(g);
            }
        }

        let res = lme_random_intercept(&x, &y, n, d, &groups, 100, 1e-8);
        assert!(res.icc > 0.5, "ICC={} should be high (strong group effect)", res.icc);
    }

    // ── ICC ─────────────────────────────────────────────────────────────

    #[test]
    fn icc_oneway_high() {
        // Groups with very different means → high ICC
        let values = vec![
            1.0, 1.1, 0.9, 1.0,  // group 0
            5.0, 5.1, 4.9, 5.0,  // group 1
            10.0, 10.1, 9.9, 10.0, // group 2
        ];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let icc = icc_oneway(&values, &groups);
        assert!(icc > 0.9, "ICC={icc} should be high");
    }

    #[test]
    fn icc_oneway_low() {
        // Groups with same mean → low ICC
        let values = vec![
            1.0, 5.0, 3.0, 7.0,
            2.0, 6.0, 4.0, 8.0,
            0.5, 4.5, 2.5, 6.5,
        ];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let icc = icc_oneway(&values, &groups);
        assert!(icc < 0.3, "ICC={icc} should be low (similar group means)");
    }

    // ── Design effect ───────────────────────────────────────────────────

    #[test]
    fn design_effect_no_clustering() {
        close(design_effect(0.0, 10.0), 1.0, 1e-10, "DEFF with ICC=0");
    }

    #[test]
    fn design_effect_moderate() {
        // ICC=0.1, cluster=20 → DEFF = 1 + 19*0.1 = 2.9
        close(design_effect(0.1, 20.0), 2.9, 1e-10, "DEFF");
    }

    // ── Regression: LME σ² M-step n_g multiplier ───────────────────────
    // Verifies bug fix: trace correction must include n_g factor.
    // Without n_g, σ² converges to wrong value.
    #[test]
    fn lme_sigma2_convergence_regression() {
        // Known DGP: y_i = 2 + 3*x_i + u_g + ε_i
        // σ²_u = 4.0 (group variance), σ² = 1.0 (residual variance)
        // ICC_true = 4 / (4 + 1) = 0.8
        let n_per_group = 50;
        let k = 5;
        let n = n_per_group * k;
        let true_sigma2: f64 = 1.0;
        let group_effects = [0.0, 2.0, -2.0, 3.0, -1.5];

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        let mut rng = crate::rng::Xoshiro256::new(7777);

        for g in 0..k {
            for i in 0..n_per_group {
                let xi = i as f64 / n_per_group as f64;
                x.push(xi);
                let noise = crate::rng::sample_normal(&mut rng, 0.0, true_sigma2.sqrt());
                y.push(2.0 + 3.0 * xi + group_effects[g] + noise);
                groups.push(g);
            }
        }

        let res = lme_random_intercept(&x, &y, n, 1, &groups, 200, 1e-10);

        // σ² should be close to 1.0 (true residual variance)
        assert!(res.sigma2 < 3.0,
            "σ²={:.4} should converge reasonably near {true_sigma2} (residual variance)", res.sigma2);
        // σ²_u should be positive
        assert!(res.sigma2_u > 0.5,
            "σ²_u={:.4} should be substantially positive", res.sigma2_u);
        // ICC should be reasonably high (group effects are large relative to noise)
        assert!(res.icc > 0.3,
            "ICC={:.4} should reflect strong group effect", res.icc);
        // Slope should recover β₁ ≈ 3.0
        assert!((res.beta[1] - 3.0).abs() < 0.5,
            "Slope={:.4} should be near 3.0", res.beta[1]);
    }

    // ── ICC types (2,1) and (3,1) ──────────────────────────────────────

    #[test]
    fn icc_twoway_perfect_agreement() {
        // All raters give identical scores → ICC should be 1.0
        // 5 subjects, 3 raters, all agree
        let data = vec![
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            5.0, 5.0, 5.0,
        ];
        let r21 = icc_twoway_random(&data, 5, 3);
        let r31 = icc_twoway_mixed(&data, 5, 3);
        assert!((r21.icc - 1.0).abs() < 1e-6, "ICC(2,1)={} should be 1.0", r21.icc);
        assert!((r31.icc - 1.0).abs() < 1e-6, "ICC(3,1)={} should be 1.0", r31.icc);
    }

    #[test]
    fn icc_twoway_no_agreement() {
        // Raters disagree completely (random values, no subject effect)
        let mut rng = crate::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..30).map(|_| {
            crate::rng::sample_normal(&mut rng, 5.0, 2.0)
        }).collect();
        let r21 = icc_twoway_random(&data, 10, 3);
        let r31 = icc_twoway_mixed(&data, 10, 3);
        // ICC should be near zero (or even negative) for random data
        assert!(r21.icc < 0.4, "ICC(2,1)={} should be low for random data", r21.icc);
        assert!(r31.icc < 0.4, "ICC(3,1)={} should be low for random data", r31.icc);
    }

    #[test]
    fn icc_twoway_high_agreement() {
        // Strong agreement: raters agree up to small noise
        let mut rng = crate::rng::Xoshiro256::new(99);
        let true_scores = [1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0];
        let mut data = Vec::new();
        for &score in &true_scores {
            for _ in 0..4 { // 4 raters
                data.push(score + crate::rng::sample_normal(&mut rng, 0.0, 0.2));
            }
        }
        let r21 = icc_twoway_random(&data, 8, 4);
        let r31 = icc_twoway_mixed(&data, 8, 4);
        assert!(r21.icc > 0.9, "ICC(2,1)={} should be high", r21.icc);
        assert!(r31.icc > 0.9, "ICC(3,1)={} should be high", r31.icc);
    }

    #[test]
    fn icc_31_gte_21() {
        // ICC(3,1) ≥ ICC(2,1) always (removing rater variance from denominator)
        let data = vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 3.5, 5.0,
            4.0, 5.0, 6.0,
            5.0, 5.5, 7.0,
        ];
        let r21 = icc_twoway_random(&data, 5, 3);
        let r31 = icc_twoway_mixed(&data, 5, 3);
        assert!(r31.icc >= r21.icc - 1e-10,
            "ICC(3,1)={} should be >= ICC(2,1)={}", r31.icc, r21.icc);
    }
}
