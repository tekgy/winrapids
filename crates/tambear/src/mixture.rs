//! # Family 16 — Mixture & Latent Class Models
//!
//! GMM (EM algorithm), LCA (categorical EM), HMM (Baum-Welch).
//!
//! ## Architecture
//!
//! EM = alternating E-step (soft assignment) + M-step (weighted accumulate).
//! Kingdom C (iterative). Each M-step's weighted mean/covariance = Kingdom A.

use crate::linear_algebra::{Mat, cholesky};

// ═══════════════════════════════════════════════════════════════════════════
// Gaussian Mixture Model (EM)
// ═══════════════════════════════════════════════════════════════════════════

/// GMM result.
#[derive(Debug, Clone)]
pub struct GmmResult {
    /// Mixing weights (sum to 1).
    pub weights: Vec<f64>,
    /// Component means (k × d).
    pub means: Vec<Vec<f64>>,
    /// Component covariance matrices (k items, each d×d).
    pub covariances: Vec<Mat>,
    /// Cluster assignments (hard: argmax of responsibilities).
    pub labels: Vec<usize>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations.
    pub iterations: usize,
}

/// Fit Gaussian Mixture Model via EM algorithm.
/// `data`: row-major n×d. `k`: number of components.
pub fn gmm_em(data: &[f64], n: usize, d: usize, k: usize, max_iter: usize, tol: f64) -> GmmResult {
    assert_eq!(data.len(), n * d);
    assert!(k > 0 && k <= n);

    // Initialize: K-means++ style seeding for means, uniform weights, identity covariance
    let mut means: Vec<Vec<f64>> = Vec::with_capacity(k);
    // First center = first data point
    means.push((0..d).map(|j| data[j]).collect());
    let mut rng = 42u64;
    for _ in 1..k {
        // Pick point with probability proportional to squared distance to nearest center
        let mut dists = vec![f64::MAX; n];
        for i in 0..n {
            for m in &means {
                let dist: f64 = (0..d).map(|j| (data[i * d + j] - m[j]).powi(2)).sum();
                dists[i] = dists[i].min(dist);
            }
        }
        let total: f64 = dists.iter().sum();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut target = (rng as f64 / u64::MAX as f64) * total;
        let mut chosen = n - 1;
        for i in 0..n {
            target -= dists[i];
            if target <= 0.0 { chosen = i; break; }
        }
        means.push((0..d).map(|j| data[chosen * d + j]).collect());
    }

    let mut weights = vec![1.0 / k as f64; k];
    let mut covariances: Vec<Mat> = (0..k).map(|_| Mat::eye(d)).collect();
    let mut resp = vec![0.0; n * k]; // responsibilities
    let mut log_likelihood = f64::NEG_INFINITY;

    let mut iterations = 0;
    for iter in 0..max_iter {
        iterations = iter + 1;

        // ── E-step: compute responsibilities ──
        for i in 0..n {
            let mut max_log = f64::NEG_INFINITY;
            for c in 0..k {
                let log_p = log_gaussian_pdf(data, i, d, &means[c], &covariances[c])
                    + weights[c].ln();
                resp[i * k + c] = log_p;
                if log_p > max_log { max_log = log_p; }
            }
            // Log-sum-exp for numerical stability
            let mut sum = 0.0;
            for c in 0..k {
                resp[i * k + c] = (resp[i * k + c] - max_log).exp();
                sum += resp[i * k + c];
            }
            for c in 0..k { resp[i * k + c] /= sum; }
        }

        // ── M-step: update parameters ──
        let mut nk = vec![0.0; k];
        for i in 0..n {
            for c in 0..k { nk[c] += resp[i * k + c]; }
        }

        // Update means
        for c in 0..k {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..n { s += resp[i * k + c] * data[i * d + j]; }
                means[c][j] = s / nk[c].max(1e-15);
            }
        }

        // Update covariances (with regularization)
        for c in 0..k {
            let mut cov = Mat::zeros(d, d);
            for i in 0..n {
                let w = resp[i * k + c];
                for j in 0..d {
                    let dj = data[i * d + j] - means[c][j];
                    for l in j..d {
                        let dl = data[i * d + l] - means[c][l];
                        let v = cov.get(j, l) + w * dj * dl;
                        cov.set(j, l, v);
                        if l != j { cov.set(l, j, v); }
                    }
                }
            }
            for j in 0..d {
                for l in 0..d {
                    cov.set(j, l, cov.get(j, l) / nk[c].max(1e-15));
                }
                // Regularization to prevent singularity
                cov.set(j, j, cov.get(j, j) + 1e-6);
            }
            covariances[c] = cov;
        }

        // Update weights
        for c in 0..k { weights[c] = nk[c] / n as f64; }

        // Compute log-likelihood
        let mut ll = 0.0;
        for i in 0..n {
            let mut sum = 0.0;
            for c in 0..k {
                let log_p = log_gaussian_pdf(data, i, d, &means[c], &covariances[c]);
                sum += weights[c] * log_p.exp();
            }
            ll += sum.max(1e-300).ln();
        }

        if (ll - log_likelihood).abs() < tol { log_likelihood = ll; break; }
        log_likelihood = ll;
    }

    // Hard assignments
    let labels: Vec<usize> = (0..n).map(|i| {
        let mut best = 0;
        let mut best_r = resp[i * k];
        for c in 1..k {
            if resp[i * k + c] > best_r {
                best_r = resp[i * k + c];
                best = c;
            }
        }
        best
    }).collect();

    GmmResult { weights, means, covariances, labels, log_likelihood, iterations }
}

/// Log of multivariate Gaussian PDF (unnormalized sufficient for EM).
fn log_gaussian_pdf(data: &[f64], i: usize, d: usize, mean: &[f64], cov: &Mat) -> f64 {
    let l = match cholesky(cov) {
        Some(l) => l,
        None => return f64::NEG_INFINITY,
    };

    // Mahalanobis distance: (x-μ)' Σ⁻¹ (x-μ)
    let diff: Vec<f64> = (0..d).map(|j| data[i * d + j] - mean[j]).collect();
    let solved = crate::linear_algebra::cholesky_solve(&l, &diff);
    let maha: f64 = diff.iter().zip(&solved).map(|(a, b)| a * b).sum();

    // log|Σ| = 2 Σ log(L_ii)
    let log_det: f64 = (0..d).map(|j| l.get(j, j).ln()).sum::<f64>() * 2.0;

    -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + maha)
}

// ═══════════════════════════════════════════════════════════════════════════
// BIC / AIC for model selection
// ═══════════════════════════════════════════════════════════════════════════

/// BIC for GMM: -2·logL + k·log(n).
pub fn gmm_bic(log_likelihood: f64, n: usize, d: usize, k: usize) -> f64 {
    let n_params = k * d + k * d * (d + 1) / 2 + k - 1; // means + covariances + weights
    -2.0 * log_likelihood + n_params as f64 * (n as f64).ln()
}

/// AIC for GMM: -2·logL + 2k.
pub fn gmm_aic(log_likelihood: f64, d: usize, k: usize) -> f64 {
    let n_params = k * d + k * d * (d + 1) / 2 + k - 1;
    -2.0 * log_likelihood + 2.0 * n_params as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── GMM-EM ──────────────────────────────────────────────────────────

    #[test]
    fn gmm_two_clusters_1d() {
        // Two well-separated 1D clusters
        let mut data = Vec::new();
        for _ in 0..30 { data.push(0.0); }
        for _ in 0..30 { data.push(10.0); }
        // Add small noise
        let mut rng = 42u64;
        for v in data.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
        }

        let res = gmm_em(&data, 60, 1, 2, 100, 1e-6);

        // Means should be near 0 and 10
        let mut sorted_means: Vec<f64> = res.means.iter().map(|m| m[0]).collect();
        sorted_means.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted_means[0] - 0.0).abs() < 1.0, "Mean 1={}", sorted_means[0]);
        assert!((sorted_means[1] - 10.0).abs() < 1.0, "Mean 2={}", sorted_means[1]);
    }

    #[test]
    fn gmm_two_clusters_2d() {
        // Two clusters in 2D
        let mut data = Vec::new();
        let mut rng = 123u64;
        for _ in 0..25 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.3;
            data.push(0.0 + noise);
            data.push(0.0 + noise);
        }
        for _ in 0..25 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.3;
            data.push(5.0 + noise);
            data.push(5.0 + noise);
        }

        let res = gmm_em(&data, 50, 2, 2, 100, 1e-6);
        // Check that labels separate the two groups
        let g1: Vec<usize> = res.labels[..25].to_vec();
        let g2: Vec<usize> = res.labels[25..].to_vec();
        let g1_mode = if g1.iter().filter(|&&l| l == 0).count() > 12 { 0 } else { 1 };
        let g2_mode = if g2.iter().filter(|&&l| l == 0).count() > 12 { 0 } else { 1 };
        assert_ne!(g1_mode, g2_mode, "Two clusters should get different labels");
    }

    #[test]
    fn gmm_bic_penalizes_complexity() {
        let ll = -100.0;
        let bic_2 = gmm_bic(ll, 100, 2, 2);
        let bic_5 = gmm_bic(ll, 100, 2, 5);
        assert!(bic_5 > bic_2, "BIC should penalize more components");
    }

    #[test]
    fn gmm_weights_sum_to_one() {
        let data: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { 10.0 }).collect();
        let res = gmm_em(&data, 40, 1, 2, 50, 1e-6);
        let w_sum: f64 = res.weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-10, "Weights sum={w_sum} should be 1.0");
    }
}
