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
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
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
        // σ² = ||y - Xβ - Zu||² / n + σ² Σ_g (1 - σ⁻² n_g σ²_u / (n_g σ²_u + σ²)) / n
        let mut ss_resid = 0.0;
        for i in 0..n {
            let g = groups[i];
            let mut fitted = beta[0];
            for j in 0..d { fitted += beta[j + 1] * x[i * d + j]; }
            fitted += u[g];
            let r = y[i] - fitted;
            ss_resid += r * r;
        }

        let trace_correction: f64 = (0..k).map(|g| {
            let ng = n_g[g] as f64;
            1.0 - ng * sigma2_u / (ng * sigma2_u + sigma2)
        }).sum();

        let sigma2_new = ss_resid / n as f64 + sigma2 * trace_correction / n as f64;

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

    let grand_mean = values.iter().sum::<f64>() / n as f64;

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
}
