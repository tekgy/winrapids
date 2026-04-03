//! # Family 22 — Dimensionality Reduction
//!
//! PCA, t-SNE, classical MDS, NMF.
//!
//! ## Architecture
//!
//! PCA = SVD of centered data (Kingdom A).
//! t-SNE = KL divergence optimization (Kingdom C).
//! MDS = eigendecomposition of double-centered distance matrix (Kingdom A).
//! NMF = multiplicative updates (Kingdom C).

use crate::linear_algebra::{Mat, svd, sym_eigen, mat_mul};

// ═══════════════════════════════════════════════════════════════════════════
// PCA
// ═══════════════════════════════════════════════════════════════════════════

/// PCA result.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal components (rows = original variables, cols = components).
    pub components: Mat,
    /// Singular values (descending).
    pub singular_values: Vec<f64>,
    /// Explained variance ratio per component.
    pub explained_variance_ratio: Vec<f64>,
    /// Projected data (n × n_components).
    pub transformed: Mat,
}

/// PCA via SVD of centered data.
/// `data`: n×d matrix (row-major). `n_components`: number of PCs to keep.
pub fn pca(data: &[f64], n: usize, d: usize, n_components: usize) -> PcaResult {
    assert_eq!(data.len(), n * d);
    let k = n_components.min(d).min(n);

    // Center
    let mut means = vec![0.0; d];
    for i in 0..n {
        for j in 0..d { means[j] += data[i * d + j]; }
    }
    for j in 0..d { means[j] /= n as f64; }

    let mut centered = Mat::zeros(n, d);
    for i in 0..n {
        for j in 0..d {
            centered.set(i, j, data[i * d + j] - means[j]);
        }
    }

    // SVD
    let svd_res = svd(&centered);

    // Components: V' rows (first k)
    let mut components = Mat::zeros(d, k);
    for j in 0..d {
        for c in 0..k {
            components.set(j, c, svd_res.vt.get(c, j));
        }
    }

    // Explained variance ratio
    let total_var: f64 = svd_res.sigma.iter().map(|s| s * s).sum();
    let explained_variance_ratio: Vec<f64> = svd_res.sigma[..k].iter()
        .map(|s| s * s / total_var.max(1e-15))
        .collect();

    // Project: X_centered · V
    let mut transformed = Mat::zeros(n, k);
    for i in 0..n {
        for c in 0..k {
            let mut val = 0.0;
            for j in 0..d { val += centered.get(i, j) * components.get(j, c); }
            transformed.set(i, c, val);
        }
    }

    PcaResult {
        components,
        singular_values: svd_res.sigma[..k].to_vec(),
        explained_variance_ratio,
        transformed,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Classical MDS
// ═══════════════════════════════════════════════════════════════════════════

/// Classical MDS result.
#[derive(Debug, Clone)]
pub struct MdsResult {
    /// Embedded coordinates (n × n_components).
    pub embedding: Mat,
    /// Eigenvalues used.
    pub eigenvalues: Vec<f64>,
    /// Stress (normalized).
    pub stress: f64,
}

/// Classical (metric) MDS via eigendecomposition of double-centered distance matrix.
/// `dist`: n×n distance matrix (symmetric, row-major).
pub fn classical_mds(dist: &[f64], n: usize, n_components: usize) -> MdsResult {
    assert_eq!(dist.len(), n * n);
    let k = n_components.min(n);

    // Squared distances
    let mut d2 = vec![0.0; n * n];
    for i in 0..n * n { d2[i] = dist[i] * dist[i]; }

    // Double centering: B = -0.5 · J · D² · J where J = I - (1/n)·11'
    let mut b = Mat::zeros(n, n);
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += d2[i * n + j];
            col_means[j] += d2[i * n + j];
            grand_mean += d2[i * n + j];
        }
    }
    for i in 0..n { row_means[i] /= n as f64; }
    for j in 0..n { col_means[j] /= n as f64; }
    grand_mean /= (n * n) as f64;

    for i in 0..n {
        for j in 0..n {
            b.set(i, j, -0.5 * (d2[i * n + j] - row_means[i] - col_means[j] + grand_mean));
        }
    }

    // Eigendecomposition
    let (evals, evecs) = sym_eigen(&b);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b_idx| evals[b_idx].total_cmp(&evals[a]));

    let mut embedding = Mat::zeros(n, k);
    let mut eigenvalues = Vec::with_capacity(k);
    for c in 0..k {
        let ci = idx[c];
        let lam = evals[ci].max(0.0);
        eigenvalues.push(lam);
        let scale = lam.sqrt();
        for i in 0..n {
            embedding.set(i, c, scale * evecs.get(i, ci));
        }
    }

    // Compute stress
    let mut stress_num = 0.0;
    let mut stress_den = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_orig = dist[i * n + j];
            let mut d_embed = 0.0;
            for c in 0..k {
                let diff = embedding.get(i, c) - embedding.get(j, c);
                d_embed += diff * diff;
            }
            d_embed = d_embed.sqrt();
            stress_num += (d_orig - d_embed).powi(2);
            stress_den += d_orig * d_orig;
        }
    }
    let stress = if stress_den > 0.0 { (stress_num / stress_den).sqrt() } else { 0.0 };

    MdsResult { embedding, eigenvalues, stress }
}

// ═══════════════════════════════════════════════════════════════════════════
// t-SNE
// ═══════════════════════════════════════════════════════════════════════════

/// t-SNE result.
#[derive(Debug, Clone)]
pub struct TsneResult {
    /// Embedded coordinates (n × 2).
    pub embedding: Mat,
    /// Final KL divergence.
    pub kl_divergence: f64,
    /// Number of iterations.
    pub iterations: usize,
}

/// t-SNE embedding to 2D.
/// `data`: n×d input, row-major. `perplexity`: target perplexity (~5-50).
pub fn tsne(data: &[f64], n: usize, d: usize, perplexity: f64, max_iter: usize, lr: f64) -> TsneResult {
    assert_eq!(data.len(), n * d);
    let out_dim = 2;

    // Pairwise distances
    let mut dist2 = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..d { d2 += (data[i * d + k] - data[j * d + k]).powi(2); }
            dist2[i * n + j] = d2;
            dist2[j * n + i] = d2;
        }
    }

    // Compute joint probabilities P using binary search for sigma
    let mut p = vec![0.0; n * n];
    let target_entropy = perplexity.ln();

    for i in 0..n {
        let mut sigma_lo = 1e-10_f64;
        let mut sigma_hi = 1e4_f64;

        for _ in 0..50 {
            let sigma = (sigma_lo + sigma_hi) / 2.0;
            let beta = 1.0 / (2.0 * sigma * sigma);

            let mut sum = 0.0;
            for j in 0..n {
                if j == i { continue; }
                let w = (-beta * dist2[i * n + j]).exp();
                p[i * n + j] = w;
                sum += w;
            }
            if sum < 1e-300 { sigma_lo = sigma; continue; }
            for j in 0..n {
                if j == i { continue; }
                p[i * n + j] /= sum;
            }

            let entropy: f64 = -(0..n)
                .filter(|&j| j != i && p[i * n + j] > 1e-300)
                .map(|j| p[i * n + j] * p[i * n + j].ln())
                .sum::<f64>();

            if entropy > target_entropy { sigma_hi = sigma; }
            else { sigma_lo = sigma; }

            if (sigma_hi - sigma_lo) < 1e-10 { break; }
        }

        // Explicit row normalization after binary search
        let mut row_sum = 0.0;
        for j in 0..n {
            if j == i { continue; }
            row_sum += p[i * n + j];
        }
        if row_sum > 1e-300 {
            for j in 0..n {
                if j == i { continue; }
                p[i * n + j] /= row_sum;
            }
        }
    }

    // Symmetrize: P_ij = (P_i|j + P_j|i) / 2n
    for i in 0..n {
        for j in (i + 1)..n {
            let sym = (p[i * n + j] + p[j * n + i]) / (2.0 * n as f64);
            p[i * n + j] = sym.max(1e-12);
            p[j * n + i] = sym.max(1e-12);
        }
    }

    // Initialize Y randomly
    let mut y = Mat::zeros(n, out_dim);
    let mut rng = 42u64;
    for i in 0..n {
        for c in 0..out_dim {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            y.set(i, c, (rng as f64 / u64::MAX as f64 - 0.5) * 0.01);
        }
    }

    let mut y_prev = y.clone();
    let momentum = 0.5;
    let mut kl = 0.0;

    for iter in 0..max_iter {
        // Compute q (Student-t with 1 dof)
        let mut q_unnorm = vec![0.0; n * n];
        let mut q_sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d2 = 0.0;
                for c in 0..out_dim { d2 += (y.get(i, c) - y.get(j, c)).powi(2); }
                let w = 1.0 / (1.0 + d2);
                q_unnorm[i * n + j] = w;
                q_unnorm[j * n + i] = w;
                q_sum += 2.0 * w;
            }
        }
        q_sum = q_sum.max(1e-300);

        // Gradient (Jacobi: compute ALL gradients first, THEN apply)
        let mom = if iter < 250 { momentum } else { 0.8 };
        // Early exaggeration: multiply P by 4.0 for first 250 iterations
        let exag = if iter < 250 { 4.0 } else { 1.0 };

        let mut grad_buf = Mat::zeros(n, out_dim);
        for i in 0..n {
            for c in 0..out_dim {
                let mut grad = 0.0;
                for j in 0..n {
                    if j == i { continue; }
                    let q_ij = q_unnorm[i * n + j] / q_sum;
                    let pq_diff = exag * p[i * n + j] - q_ij;
                    grad += 4.0 * pq_diff * q_unnorm[i * n + j] * (y.get(i, c) - y.get(j, c));
                }
                grad_buf.set(i, c, grad);
            }
        }

        let y_old = y.clone();
        for i in 0..n {
            for c in 0..out_dim {
                let new_val = y.get(i, c) - lr * grad_buf.get(i, c) + mom * (y.get(i, c) - y_prev.get(i, c));
                y.set(i, c, new_val);
            }
        }
        y_prev = y_old;

        // KL divergence
        if iter == max_iter - 1 || iter % 100 == 0 {
            kl = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if j == i { continue; }
                    let q_ij = (q_unnorm[i * n + j] / q_sum).max(1e-12);
                    if p[i * n + j] > 1e-12 {
                        kl += p[i * n + j] * (p[i * n + j] / q_ij).ln();
                    }
                }
            }
        }
    }

    TsneResult { embedding: y, kl_divergence: kl, iterations: max_iter }
}

// ═══════════════════════════════════════════════════════════════════════════
// NMF (Non-negative Matrix Factorization)
// ═══════════════════════════════════════════════════════════════════════════

/// NMF result: V ≈ W·H.
#[derive(Debug, Clone)]
pub struct NmfResult {
    /// Basis matrix W (n × k).
    pub w: Mat,
    /// Coefficient matrix H (k × m).
    pub h: Mat,
    /// Reconstruction error (Frobenius norm).
    pub error: f64,
    /// Iterations.
    pub iterations: usize,
}

/// Non-negative Matrix Factorization via multiplicative updates (Lee & Seung).
/// `v`: n×m non-negative matrix (row-major). `k`: rank.
pub fn nmf(v: &[f64], n: usize, m: usize, k: usize, max_iter: usize) -> NmfResult {
    assert_eq!(v.len(), n * m);

    // Initialize W and H with random non-negative values
    let mut rng = 42u64;
    let mut w = Mat::zeros(n, k);
    let mut h = Mat::zeros(k, m);

    for i in 0..n {
        for c in 0..k {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            w.set(i, c, (rng as f64 / u64::MAX as f64) * 0.1 + 0.01);
        }
    }
    for c in 0..k {
        for j in 0..m {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            h.set(c, j, (rng as f64 / u64::MAX as f64) * 0.1 + 0.01);
        }
    }

    let v_mat = Mat::from_vec(n, m, v.to_vec());
    let eps = 1e-10;

    for iter in 0..max_iter {
        // Update H: H ← H .* (W'V) ./ (W'WH + eps)
        let wt_v = {
            let wt = transpose(&w);
            mat_mul(&wt, &v_mat)
        };
        let wt_w = {
            let wt = transpose(&w);
            mat_mul(&wt, &w)
        };
        let wt_wh = mat_mul(&wt_w, &h);
        for i in 0..k {
            for j in 0..m {
                h.set(i, j, h.get(i, j) * wt_v.get(i, j) / (wt_wh.get(i, j) + eps));
            }
        }

        // Update W: W ← W .* (VH') ./ (WHH' + eps)
        let ht = transpose(&h);
        let v_ht = mat_mul(&v_mat, &ht);
        let wh = mat_mul(&w, &h);
        let wh_ht = mat_mul(&wh, &ht);
        for i in 0..n {
            for j in 0..k {
                w.set(i, j, w.get(i, j) * v_ht.get(i, j) / (wh_ht.get(i, j) + eps));
            }
        }
    }

    // Reconstruction error
    let approx = mat_mul(&w, &h);
    let mut err = 0.0;
    for i in 0..n {
        for j in 0..m {
            err += (v_mat.get(i, j) - approx.get(i, j)).powi(2);
        }
    }
    let error = err.sqrt();

    NmfResult { w, h, error, iterations: max_iter }
}

fn transpose(a: &Mat) -> Mat {
    let mut t = Mat::zeros(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            t.set(j, i, a.get(i, j));
        }
    }
    t
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

    #[test]
    fn pca_variance_explained_sums_to_one() {
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 5.0, 7.0,
            4.0, 6.0, 8.0,
            5.0, 7.0, 9.0,
        ];
        let res = pca(&data, 5, 3, 3);
        let total: f64 = res.explained_variance_ratio.iter().sum();
        close(total, 1.0, 1e-6, "Total explained variance");
    }

    #[test]
    fn pca_reduces_dimension() {
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 5.0, 7.0,
            4.0, 6.0, 8.0,
        ];
        let res = pca(&data, 4, 3, 2);
        assert_eq!(res.transformed.rows, 4);
        assert_eq!(res.transformed.cols, 2);
        assert_eq!(res.singular_values.len(), 2);
    }

    #[test]
    fn pca_first_component_most_variance() {
        let mut data = Vec::new();
        let mut rng = 42u64;
        for _ in 0..50 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = rng as f64 / u64::MAX as f64 * 10.0;
            data.push(x);
            data.push(x + (rng as f64 / u64::MAX as f64 - 0.5) * 0.1); // nearly collinear
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push(rng as f64 / u64::MAX as f64); // independent
        }
        let res = pca(&data, 50, 3, 3);
        assert!(res.explained_variance_ratio[0] > res.explained_variance_ratio[1],
            "First PC should explain more variance");
    }

    #[test]
    fn mds_preserves_distances_2d() {
        // 4 points forming a square
        let dist = vec![
            0.0, 1.0, 2.0_f64.sqrt(), 1.0,
            1.0, 0.0, 1.0, 2.0_f64.sqrt(),
            2.0_f64.sqrt(), 1.0, 0.0, 1.0,
            1.0, 2.0_f64.sqrt(), 1.0, 0.0,
        ];
        let res = classical_mds(&dist, 4, 2);
        assert_eq!(res.embedding.rows, 4);
        assert_eq!(res.embedding.cols, 2);
        assert!(res.stress < 0.3, "Stress={} should be low for perfect 2D embedding", res.stress);
    }

    #[test]
    fn tsne_separates_clusters() {
        // Two simple clusters in 3D
        let mut data = Vec::new();
        for _ in 0..15 { data.extend_from_slice(&[0.0, 0.0, 0.0]); }
        for _ in 0..15 { data.extend_from_slice(&[10.0, 10.0, 10.0]); }

        let res = tsne(&data, 30, 3, 5.0, 300, 50.0);
        assert_eq!(res.embedding.rows, 30);
        assert_eq!(res.embedding.cols, 2);

        // Check cluster separation: centroid distance in embedding should be positive
        let mut c1 = [0.0, 0.0];
        let mut c2 = [0.0, 0.0];
        for i in 0..15 {
            c1[0] += res.embedding.get(i, 0);
            c1[1] += res.embedding.get(i, 1);
        }
        for i in 15..30 {
            c2[0] += res.embedding.get(i, 0);
            c2[1] += res.embedding.get(i, 1);
        }
        for v in &mut c1 { *v /= 15.0; }
        for v in &mut c2 { *v /= 15.0; }
        let dist = ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2)).sqrt();
        assert!(dist > 0.1, "t-SNE cluster centroids should be separated, dist={dist}");
    }

    #[test]
    fn nmf_reduces_error() {
        let v: Vec<f64> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let res = nmf(&v, 4, 3, 2, 200);
        // With rank-2 approximation of a nearly rank-2 matrix, error should be small
        assert!(res.error < 5.0, "NMF error={} should be moderate", res.error);
    }

    #[test]
    fn nmf_nonnegative() {
        let v: Vec<f64> = vec![
            1.0, 0.5, 0.0,
            0.0, 1.0, 0.5,
            0.5, 0.0, 1.0,
        ];
        let res = nmf(&v, 3, 3, 2, 100);
        for i in 0..res.w.rows {
            for j in 0..res.w.cols {
                assert!(res.w.get(i, j) >= 0.0, "W[{i},{j}]={} should be non-negative", res.w.get(i, j));
            }
        }
        for i in 0..res.h.rows {
            for j in 0..res.h.cols {
                assert!(res.h.get(i, j) >= 0.0, "H[{i},{j}]={} should be non-negative", res.h.get(i, j));
            }
        }
    }
}
