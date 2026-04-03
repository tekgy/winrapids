//! # Family 33 — Multivariate Analysis
//!
//! MANOVA, CCA, Hotelling's T², LDA, multivariate normality tests.
//!
//! ## Architecture
//!
//! Every method in this family follows one pattern:
//! 1. Compute cross-product matrices from data (GramMatrix subblocks)
//! 2. Form a ratio or product of these matrices
//! 3. Eigendecompose the result
//! 4. Extract test statistics / canonical variates from eigenvalues/eigenvectors
//!
//! Kingdom A (Commutative): pure GramMatrix → eigendecomposition → extraction.

use crate::linear_algebra::{
    Mat, mat_mul, mat_add, mat_scale, cholesky, cholesky_solve, sym_eigen, svd,
};
use crate::special_functions::{f_right_tail_p, chi2_right_tail_p, normal_two_tail_p};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: Covariance matrices from data
// ═══════════════════════════════════════════════════════════════════════════

/// Compute sample covariance matrix (Bessel-corrected) from n×p data matrix.
/// Each row is an observation, each column is a variable.
fn covariance_matrix(x: &Mat) -> Mat {
    let n = x.rows;
    let p = x.cols;
    // Column means
    let mut means = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            means[j] += x.get(i, j);
        }
    }
    for j in 0..p { means[j] /= n as f64; }

    // Centered cross-products
    let mut cov = Mat::zeros(p, p);
    for i in 0..n {
        for j in 0..p {
            let dj = x.get(i, j) - means[j];
            for k in j..p {
                let dk = x.get(i, k) - means[k];
                let v = cov.get(j, k) + dj * dk;
                cov.set(j, k, v);
                if k != j { cov.set(k, j, v); }
            }
        }
    }
    let denom = (n - 1) as f64;
    for j in 0..p {
        for k in 0..p {
            cov.set(j, k, cov.get(j, k) / denom);
        }
    }
    cov
}

/// Column means of an n×p matrix.
fn col_means(x: &Mat) -> Vec<f64> {
    let n = x.rows as f64;
    let mut m = vec![0.0; x.cols];
    for i in 0..x.rows {
        for j in 0..x.cols { m[j] += x.get(i, j); }
    }
    for j in 0..x.cols { m[j] /= n; }
    m
}

/// Within-group and between-group SSCP matrices for MANOVA/LDA.
/// `groups[i]` = group label (0-indexed) for observation i.
/// Returns (W, H, group_means, group_counts, grand_mean).
fn sscp_matrices(x: &Mat, groups: &[usize]) -> (Mat, Mat, Vec<Vec<f64>>, Vec<usize>, Vec<f64>) {
    let n = x.rows;
    let p = x.cols;
    let k = *groups.iter().max().unwrap_or(&0) + 1;

    // Grand mean + group means + group counts
    let grand_mean = col_means(x);
    let mut g_sums = vec![vec![0.0; p]; k];
    let mut g_counts = vec![0usize; k];
    for i in 0..n {
        let g = groups[i];
        g_counts[g] += 1;
        for j in 0..p { g_sums[g][j] += x.get(i, j); }
    }
    let g_means: Vec<Vec<f64>> = (0..k).map(|g| {
        if g_counts[g] == 0 { vec![0.0; p] }
        else { g_sums[g].iter().map(|&s| s / g_counts[g] as f64).collect() }
    }).collect();

    // Within-group SSCP: W = Σ_g Σ_{i∈g} (x_i - x̄_g)(x_i - x̄_g)'
    let mut w = Mat::zeros(p, p);
    for i in 0..n {
        let g = groups[i];
        for j in 0..p {
            let dj = x.get(i, j) - g_means[g][j];
            for l in j..p {
                let dl = x.get(i, l) - g_means[g][l];
                let v = w.get(j, l) + dj * dl;
                w.set(j, l, v);
                if l != j { w.set(l, j, v); }
            }
        }
    }

    // Between-group SSCP: H = Σ_g n_g (x̄_g - x̄)(x̄_g - x̄)'
    let mut h = Mat::zeros(p, p);
    for g in 0..k {
        let ng = g_counts[g] as f64;
        if ng == 0.0 { continue; }
        for j in 0..p {
            let dj = g_means[g][j] - grand_mean[j];
            for l in j..p {
                let dl = g_means[g][l] - grand_mean[l];
                let v = h.get(j, l) + ng * dj * dl;
                h.set(j, l, v);
                if l != j { h.set(l, j, v); }
            }
        }
    }

    (w, h, g_means, g_counts, grand_mean)
}

// ═══════════════════════════════════════════════════════════════════════════
// Hotelling's T²
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Hotelling's T² test.
#[derive(Debug, Clone)]
pub struct HotellingResult {
    pub t2: f64,
    pub f_statistic: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
}

/// One-sample Hotelling's T² test: H₀: μ = μ₀.
/// `x` is n×p data matrix, `mu0` is the hypothesized mean vector.
pub fn hotelling_one_sample(x: &Mat, mu0: &[f64]) -> HotellingResult {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(mu0.len(), p);

    let means = col_means(x);
    let cov = covariance_matrix(x);
    let l = cholesky(&cov).expect("covariance not positive definite");

    // d = x̄ - μ₀
    let d: Vec<f64> = means.iter().zip(mu0).map(|(&m, &m0)| m - m0).collect();
    // T² = n · d' S⁻¹ d
    let s_inv_d = cholesky_solve(&l, &d);
    let t2 = n as f64 * d.iter().zip(&s_inv_d).map(|(a, b)| a * b).sum::<f64>();

    let nf = n as f64;
    let pf = p as f64;
    let f_stat = (nf - pf) / ((nf - 1.0) * pf) * t2;
    let df1 = pf;
    let df2 = nf - pf;
    let p_value = f_right_tail_p(f_stat, df1, df2);

    HotellingResult { t2, f_statistic: f_stat, df1, df2, p_value }
}

/// Two-sample Hotelling's T² test: H₀: μ₁ = μ₂.
pub fn hotelling_two_sample(x1: &Mat, x2: &Mat) -> HotellingResult {
    let n1 = x1.rows;
    let n2 = x2.rows;
    let p = x1.cols;
    assert_eq!(x2.cols, p);

    let m1 = col_means(x1);
    let m2 = col_means(x2);
    let s1 = covariance_matrix(x1);
    let s2 = covariance_matrix(x2);

    // Pooled covariance: S_p = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    let sp = {
        let a = mat_scale((n1 - 1) as f64, &s1);
        let b = mat_scale((n2 - 1) as f64, &s2);
        let sum = mat_add(&a, &b);
        mat_scale(1.0 / (n1 + n2 - 2) as f64, &sum)
    };
    let l = cholesky(&sp).expect("pooled covariance not positive definite");

    let d: Vec<f64> = m1.iter().zip(&m2).map(|(a, b)| a - b).collect();
    let s_inv_d = cholesky_solve(&l, &d);
    let scale = (n1 as f64 * n2 as f64) / (n1 + n2) as f64;
    let t2 = scale * d.iter().zip(&s_inv_d).map(|(a, b)| a * b).sum::<f64>();

    let n = (n1 + n2) as f64;
    let pf = p as f64;
    let f_stat = (n - pf - 1.0) / ((n - 2.0) * pf) * t2;
    let df1 = pf;
    let df2 = n - pf - 1.0;
    let p_value = f_right_tail_p(f_stat, df1, df2);

    HotellingResult { t2, f_statistic: f_stat, df1, df2, p_value }
}

// ═══════════════════════════════════════════════════════════════════════════
// MANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// MANOVA test statistics.
#[derive(Debug, Clone)]
pub struct ManovaResult {
    /// Wilks' Lambda: Λ = |W| / |T| = Π 1/(1+θᵢ)
    pub wilks_lambda: f64,
    /// Pillai's trace: V = Σ θᵢ/(1+θᵢ)
    pub pillai_trace: f64,
    /// Hotelling-Lawley trace: U = Σ θᵢ
    pub hotelling_lawley: f64,
    /// Roy's largest root: θ_max
    pub roy_largest_root: f64,
    /// Eigenvalues of H W⁻¹
    pub eigenvalues: Vec<f64>,
    /// Approximate F-statistic (from Pillai's trace)
    pub f_statistic: f64,
    /// Approximate p-value (from Pillai's trace — most robust)
    pub p_value: f64,
}

/// One-way MANOVA: test whether group means differ across p response variables.
/// `x` is n×p data, `groups[i]` is group index for observation i.
pub fn manova(x: &Mat, groups: &[usize]) -> ManovaResult {
    assert_eq!(x.rows, groups.len());
    let n = x.rows;
    let p = x.cols;
    let k = *groups.iter().max().unwrap_or(&0) + 1;

    let (w, h, _, _, _) = sscp_matrices(x, groups);

    // Eigenvalues of H·W⁻¹ via generalized eigenvalue → solve W⁻¹·H
    // Equivalent: eigenvalues of L⁻¹ H L⁻ᵀ where W = LLᵀ
    let l = cholesky(&w).expect("within-group SSCP not positive definite");

    // Solve L⁻¹ H: forward-substitute each column of H
    let mut lih = Mat::zeros(p, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| h.get(i, j)).collect();
        let solved = forward_solve(&l, &col);
        for i in 0..p { lih.set(i, j, solved[i]); }
    }
    // Form L⁻¹ H L⁻ᵀ = (L⁻¹ H)ᵀ solved again... actually form M = L⁻¹ H L⁻ᵀ
    // M = (L⁻¹ H) L⁻ᵀ. Let A = L⁻¹ H. Then M = A L⁻ᵀ = (L⁻¹ Aᵀ)ᵀ
    let liht = lih.t();
    let mut m = Mat::zeros(p, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| liht.get(i, j)).collect();
        let solved = forward_solve(&l, &col);
        for i in 0..p { m.set(j, i, solved[i]); } // transpose back
    }

    let (eigenvalues, _) = sym_eigen(&m);
    let s = p.min(k - 1);

    let wilks_lambda: f64 = eigenvalues.iter().take(s).map(|&e| 1.0 / (1.0 + e)).product();
    let pillai_trace: f64 = eigenvalues.iter().take(s).map(|&e| e / (1.0 + e)).sum();
    let hotelling_lawley: f64 = eigenvalues.iter().take(s).sum();
    let roy_largest_root = eigenvalues.first().copied().unwrap_or(0.0);

    // Approximate F from Pillai's trace (most robust)
    let sf = s as f64;
    let pf = p as f64;
    let kf = k as f64;
    let df1 = sf * pf;
    let df2 = sf * (n as f64 - kf);
    let f_stat = if df2 > 0.0 && sf > 0.0 {
        (pillai_trace / sf) / ((sf - pillai_trace) / sf) * (df2 / df1)
    } else { 0.0 };
    let p_value = if df1 > 0.0 && df2 > 0.0 { f_right_tail_p(f_stat, df1, df2) } else { 1.0 };

    ManovaResult {
        wilks_lambda, pillai_trace, hotelling_lawley,
        roy_largest_root, eigenvalues, f_statistic: f_stat, p_value,
    }
}

/// Forward substitution: solve L·x = b where L is lower-triangular.
fn forward_solve(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            x[i] -= l.get(i, j) * x[j];
        }
        x[i] /= l.get(i, i);
    }
    x
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear Discriminant Analysis (LDA)
// ═══════════════════════════════════════════════════════════════════════════

/// LDA result: discriminant axes and classification.
#[derive(Debug, Clone)]
pub struct LdaResult {
    /// Discriminant directions (p × d matrix, d = min(p, k-1)).
    pub axes: Mat,
    /// Eigenvalues (between/within ratio for each axis).
    pub eigenvalues: Vec<f64>,
    /// Group means in original space (k × p).
    pub group_means: Vec<Vec<f64>>,
    /// Group counts.
    pub group_counts: Vec<usize>,
}

impl LdaResult {
    /// Project data onto discriminant axes. Returns n×d projected coordinates.
    pub fn transform(&self, x: &Mat) -> Mat {
        mat_mul(x, &self.axes)
    }

    /// Classify new observations by nearest centroid in discriminant space.
    /// Returns predicted group labels.
    pub fn predict(&self, x: &Mat) -> Vec<usize> {
        let projected = self.transform(x);
        // Project group means
        let k = self.group_means.len();
        let d = self.axes.cols;
        let proj_means: Vec<Vec<f64>> = self.group_means.iter().map(|gm| {
            let gm_mat = Mat::from_vec(1, x.cols, gm.clone());
            let proj = mat_mul(&gm_mat, &self.axes);
            (0..d).map(|j| proj.get(0, j)).collect()
        }).collect();

        (0..projected.rows).map(|i| {
            let mut best_g = 0;
            let mut best_dist = f64::MAX;
            for g in 0..k {
                let dist: f64 = (0..d).map(|j| {
                    let diff = projected.get(i, j) - proj_means[g][j];
                    diff * diff
                }).sum();
                if dist < best_dist { best_dist = dist; best_g = g; }
            }
            best_g
        }).collect()
    }
}

/// Fisher's Linear Discriminant Analysis.
/// `x` is n×p data, `groups[i]` is group index for observation i.
pub fn lda(x: &Mat, groups: &[usize]) -> LdaResult {
    let p = x.cols;
    let k = *groups.iter().max().unwrap_or(&0) + 1;
    let d = p.min(k - 1); // number of discriminant dimensions

    let (w, h, g_means, g_counts, _) = sscp_matrices(x, groups);

    // Generalized eigenvalue problem: H·v = λ·W·v
    // Equivalent: sym_eigen(L⁻¹ H L⁻ᵀ) where W = LLᵀ
    let l = cholesky(&w).expect("within-group SSCP not positive definite");

    let mut lih = Mat::zeros(p, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| h.get(i, j)).collect();
        let solved = forward_solve(&l, &col);
        for i in 0..p { lih.set(i, j, solved[i]); }
    }
    let liht = lih.t();
    let mut m = Mat::zeros(p, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| liht.get(i, j)).collect();
        let solved = forward_solve(&l, &col);
        for i in 0..p { m.set(j, i, solved[i]); }
    }

    let (eigenvalues, eigenvectors) = sym_eigen(&m);

    // Back-transform eigenvectors: v = L⁻ᵀ · u
    let mut axes = Mat::zeros(p, d);
    for j in 0..d {
        let u: Vec<f64> = (0..p).map(|i| eigenvectors.get(i, j)).collect();
        let v = back_solve_transpose(&l, &u);
        for i in 0..p { axes.set(i, j, v[i]); }
    }

    LdaResult {
        axes,
        eigenvalues: eigenvalues[..d].to_vec(),
        group_means: g_means,
        group_counts: g_counts,
    }
}

/// Back-substitution for L^T x = b (solves with transpose of lower-triangular).
fn back_solve_transpose(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= l.get(j, i) * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] /= l.get(i, i);
    }
    x
}

// ═══════════════════════════════════════════════════════════════════════════
// Canonical Correlation Analysis (CCA)
// ═══════════════════════════════════════════════════════════════════════════

/// CCA result.
#[derive(Debug, Clone)]
pub struct CcaResult {
    /// Canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρ_s (s = min(p,q)).
    pub correlations: Vec<f64>,
    /// Canonical variates for X (p × s).
    pub x_weights: Mat,
    /// Canonical variates for Y (q × s).
    pub y_weights: Mat,
    /// Wilks' Lambda for significance test.
    pub wilks_lambda: f64,
    /// Approximate p-value (Bartlett's χ² approximation).
    pub p_value: f64,
}

/// Canonical Correlation Analysis between X (n×p) and Y (n×q).
pub fn cca(x: &Mat, y: &Mat) -> CcaResult {
    let n = x.rows;
    assert_eq!(y.rows, n);
    let p = x.cols;
    let q = y.cols;
    let s = p.min(q);

    // Covariance subblocks from the joint [X, Y] matrix
    let cov_xx = covariance_matrix(x);
    let cov_yy = covariance_matrix(y);

    // Cross-covariance Σ_XY
    let mx = col_means(x);
    let my = col_means(y);
    let mut cov_xy = Mat::zeros(p, q);
    for i in 0..n {
        for j in 0..p {
            let dj = x.get(i, j) - mx[j];
            for k in 0..q {
                let dk = y.get(i, k) - my[k];
                let v = cov_xy.get(j, k) + dj * dk;
                cov_xy.set(j, k, v);
            }
        }
    }
    let denom = (n - 1) as f64;
    for j in 0..p {
        for k in 0..q { cov_xy.set(j, k, cov_xy.get(j, k) / denom); }
    }

    // SVD of whitened cross-covariance: Σ_XX^(-1/2) Σ_XY Σ_YY^(-1/2)
    let lx = cholesky(&cov_xx).expect("Σ_XX not positive definite");
    let ly = cholesky(&cov_yy).expect("Σ_YY not positive definite");

    // Solve L_X⁻¹ Σ_XY
    let mut whitened = Mat::zeros(p, q);
    for j in 0..q {
        let col: Vec<f64> = (0..p).map(|i| cov_xy.get(i, j)).collect();
        let solved = forward_solve(&lx, &col);
        for i in 0..p { whitened.set(i, j, solved[i]); }
    }
    // Then multiply by L_Y⁻ᵀ on the right: (L_X⁻¹ Σ_XY) L_Y⁻ᵀ
    let wt = whitened.t(); // q × p
    let mut result_t = Mat::zeros(q, p);
    for j in 0..p {
        let col: Vec<f64> = (0..q).map(|i| wt.get(i, j)).collect();
        let solved = forward_solve(&ly, &col);
        for i in 0..q { result_t.set(i, j, solved[i]); }
    }
    let whitened_cov = result_t.t(); // p × q

    let svd_res = svd(&whitened_cov);
    let correlations: Vec<f64> = svd_res.sigma.iter().take(s).copied().collect();

    // Canonical weights: a = L_X⁻ᵀ u, b = L_Y⁻ᵀ v
    let mut x_weights = Mat::zeros(p, s);
    for j in 0..s {
        let u: Vec<f64> = (0..p).map(|i| svd_res.u.get(i, j)).collect();
        let a = back_solve_transpose(&lx, &u);
        for i in 0..p { x_weights.set(i, j, a[i]); }
    }
    let mut y_weights = Mat::zeros(q, s);
    for j in 0..s {
        let v: Vec<f64> = (0..q).map(|i| svd_res.vt.get(j, i)).collect();
        let b = back_solve_transpose(&ly, &v);
        for i in 0..q { y_weights.set(i, j, b[i]); }
    }

    // Wilks' Lambda = Π(1 - ρᵢ²)
    let wilks: f64 = correlations.iter().map(|r| 1.0 - r * r).product();

    // Bartlett's χ² approximation: -(n-1-(p+q+1)/2) ln(Λ)
    let nf = n as f64;
    let pf = p as f64;
    let qf = q as f64;
    let chi2 = -(nf - 1.0 - (pf + qf + 1.0) / 2.0) * wilks.ln();
    let df = pf * qf;
    let p_value = chi2_right_tail_p(chi2, df);

    CcaResult { correlations, x_weights, y_weights, wilks_lambda: wilks, p_value }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mardia's multivariate normality tests
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Mardia's multivariate normality tests.
#[derive(Debug, Clone)]
pub struct MardiaNormalityResult {
    /// Multivariate skewness statistic b₁,p.
    pub skewness: f64,
    /// p-value for skewness (χ² approximation).
    pub skewness_p: f64,
    /// Multivariate kurtosis statistic b₂,p.
    pub kurtosis: f64,
    /// p-value for kurtosis (normal approximation).
    pub kurtosis_p: f64,
}

/// Mardia's tests for multivariate normality.
pub fn mardia_normality(x: &Mat) -> MardiaNormalityResult {
    let n = x.rows;
    let p = x.cols;
    let nf = n as f64;
    let pf = p as f64;

    let means = col_means(x);
    let cov = covariance_matrix(x);
    let l = cholesky(&cov).expect("covariance not positive definite");

    // Compute Mahalanobis distances: d_ij = (x_i - x̄)' S⁻¹ (x_j - x̄)
    // First, compute S⁻¹(x_i - x̄) for all i
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let d: Vec<f64> = (0..p).map(|j| x.get(i, j) - means[j]).collect();
        z.push(cholesky_solve(&l, &d));
    }

    // Skewness: b₁,p = (1/n²) Σᵢ Σⱼ [d_ij]³ where d_ij = (x_i-x̄)'S⁻¹(x_j-x̄)
    // d_ij = z_i · (x_j - x̄) (since z_i = S⁻¹(x_i - x̄))
    let mut b1 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let dij: f64 = (0..p).map(|k| z[i][k] * (x.get(j, k) - means[k])).sum();
            b1 += dij * dij * dij;
        }
    }
    b1 /= nf * nf;

    // Kurtosis: b₂,p = (1/n) Σᵢ [d_ii]² where d_ii = (x_i-x̄)'S⁻¹(x_i-x̄)
    let mut b2 = 0.0;
    for i in 0..n {
        let dii: f64 = (0..p).map(|k| z[i][k] * (x.get(i, k) - means[k])).sum();
        b2 += dii * dii;
    }
    b2 /= nf;

    // Skewness test: n·b₁/6 ~ χ²(p(p+1)(p+2)/6)
    let skew_chi2 = nf * b1 / 6.0;
    let skew_df = pf * (pf + 1.0) * (pf + 2.0) / 6.0;
    let skewness_p = chi2_right_tail_p(skew_chi2, skew_df);

    // Kurtosis test: (b₂ - p(p+2)) / √(8p(p+2)/n) ~ N(0,1)
    let kurt_z = (b2 - pf * (pf + 2.0)) / (8.0 * pf * (pf + 2.0) / nf).sqrt();
    let kurtosis_p = normal_two_tail_p(kurt_z);

    MardiaNormalityResult { skewness: b1, skewness_p, kurtosis: b2, kurtosis_p }
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

    // ── Hotelling's T² ──────────────────────────────────────────────────

    #[test]
    fn hotelling_one_sample_known() {
        // 2D data centered near (1, 2) with known covariance
        let x = Mat::from_rows(&[
            &[1.2, 2.1], &[0.8, 1.9], &[1.1, 2.3], &[0.9, 1.7],
            &[1.3, 2.0], &[1.0, 2.2], &[0.7, 1.8], &[1.1, 2.1],
        ]);
        let res = hotelling_one_sample(&x, &[1.0, 2.0]);
        assert!(res.t2 >= 0.0, "T² should be non-negative");
        assert!(res.f_statistic >= 0.0, "F should be non-negative");
        assert!(res.p_value > 0.05, "Should not reject null (data is near μ₀)");
    }

    #[test]
    fn hotelling_one_sample_reject() {
        // Data clearly not centered at origin
        let x = Mat::from_rows(&[
            &[10.0, 20.0], &[11.0, 19.0], &[10.5, 20.5], &[9.5, 19.5],
            &[10.2, 20.2], &[10.8, 19.8], &[10.1, 20.1], &[10.3, 19.7],
        ]);
        let res = hotelling_one_sample(&x, &[0.0, 0.0]);
        assert!(res.p_value < 0.001, "Should reject null (data far from origin)");
    }

    #[test]
    fn hotelling_two_sample_same_population() {
        // Two samples from same distribution
        let x1 = Mat::from_rows(&[
            &[1.0, 2.0], &[1.1, 2.1], &[0.9, 1.9], &[1.0, 2.0], &[1.05, 1.95],
        ]);
        let x2 = Mat::from_rows(&[
            &[1.0, 2.0], &[0.95, 2.05], &[1.05, 1.95], &[0.9, 2.1], &[1.1, 1.9],
        ]);
        let res = hotelling_two_sample(&x1, &x2);
        assert!(res.p_value > 0.05, "Should not reject (same distribution)");
    }

    #[test]
    fn hotelling_reduces_to_t2() {
        // For p=1, Hotelling T² = t² (square of Student's t)
        let x = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let res = hotelling_one_sample(&x, &[0.0]);
        // t = (3 - 0) / (sqrt(2.5) / sqrt(5)) = 3 / 0.7071 ≈ 4.243
        // T² = t² ≈ 18
        assert!(res.t2 > 10.0, "T² should be large for data far from 0");
        close(res.df1, 1.0, 1e-10, "df1 for univariate");
    }

    // ── MANOVA ──────────────────────────────────────────────────────────

    #[test]
    fn manova_clear_separation() {
        // Three well-separated groups in 2D
        let x = Mat::from_rows(&[
            // Group 0: cluster at (0, 0)
            &[0.1, 0.2], &[-0.1, -0.1], &[0.2, 0.0], &[0.0, 0.1],
            // Group 1: cluster at (5, 5)
            &[5.1, 5.2], &[4.9, 4.9], &[5.2, 5.0], &[5.0, 5.1],
            // Group 2: cluster at (10, 0)
            &[10.1, 0.2], &[9.9, -0.1], &[10.2, 0.0], &[10.0, 0.1],
        ]);
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let res = manova(&x, &groups);
        assert!(res.wilks_lambda < 0.01, "Wilks should be near 0 for clear separation");
        assert!(res.pillai_trace > 1.5, "Pillai should be near 2.0 for 3 groups 2D");
        assert!(res.p_value < 0.001, "Should strongly reject H₀");
    }

    #[test]
    fn manova_no_difference() {
        // Two groups drawn from same distribution
        let x = Mat::from_rows(&[
            &[1.0, 2.0], &[1.1, 2.1], &[0.9, 1.9], &[1.05, 1.95], &[0.95, 2.05],
            &[1.0, 2.0], &[1.1, 1.9], &[0.9, 2.1], &[1.05, 2.05], &[0.95, 1.95],
        ]);
        let groups = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let res = manova(&x, &groups);
        assert!(res.wilks_lambda > 0.5, "Wilks should be near 1.0 for no difference");
        assert!(res.p_value > 0.05, "Should not reject H₀");
    }

    // ── LDA ─────────────────────────────────────────────────────────────

    #[test]
    fn lda_separates_groups() {
        // Two clearly separated 2D groups → 1 discriminant dimension
        let x = Mat::from_rows(&[
            &[0.0, 0.0], &[0.1, 0.1], &[-0.1, -0.1], &[0.0, 0.1],
            &[5.0, 5.0], &[5.1, 5.1], &[4.9, 4.9], &[5.0, 5.1],
        ]);
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let res = lda(&x, &groups);

        assert_eq!(res.eigenvalues.len(), 1, "k=2 → 1 discriminant dimension");
        assert!(res.eigenvalues[0] > 10.0, "Large eigenvalue for clear separation");

        // Classification should be perfect
        let preds = res.predict(&x);
        for (i, &pred) in preds.iter().enumerate() {
            assert_eq!(pred, groups[i], "Misclassified observation {i}");
        }
    }

    #[test]
    fn lda_three_groups() {
        // Three groups in 3D → 2 discriminant dimensions (need n_g > p for W to be non-singular)
        let x = Mat::from_rows(&[
            &[0.0, 0.0, 0.0], &[0.1, 0.2, 0.1], &[-0.1, 0.0, 0.2], &[0.2, -0.1, 0.0],
            &[5.0, 0.0, 0.0], &[5.1, 0.2, 0.1], &[4.9, 0.0, 0.2], &[5.2, -0.1, 0.0],
            &[0.0, 5.0, 0.0], &[0.1, 5.2, 0.1], &[-0.1, 5.0, 0.2], &[0.2, 4.9, 0.0],
        ]);
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let res = lda(&x, &groups);
        assert_eq!(res.eigenvalues.len(), 2, "k=3 → 2 discriminant dimensions");
        assert!(res.eigenvalues[0] > 1.0, "First axis should discriminate");
    }

    // ── CCA ─────────────────────────────────────────────────────────────

    #[test]
    fn cca_strong_correlation() {
        // X: two independent columns. Y ≈ linear transform of X + small noise.
        let x = Mat::from_rows(&[
            &[1.0, 5.0], &[2.0, 3.0], &[3.0, 7.0], &[4.0, 1.0],
            &[5.0, 6.0], &[6.0, 2.0], &[7.0, 8.0], &[8.0, 4.0],
            &[9.0, 9.0], &[10.0, 0.0],
        ]);
        let y = Mat::from_rows(&[
            &[2.1, 5.2], &[4.0, 3.1], &[6.1, 7.1], &[8.0, 1.1],
            &[10.1, 6.0], &[12.0, 2.1], &[14.1, 8.0], &[16.0, 4.1],
            &[18.1, 9.1], &[20.0, 0.1],
        ]);
        let res = cca(&x, &y);
        assert!(res.correlations[0] > 0.99, "First canonical correlation should be ~1.0, got {}", res.correlations[0]);
        assert!(res.wilks_lambda < 0.05, "Wilks should be near 0");
    }

    #[test]
    fn cca_independent() {
        // X and Y constructed to be uncorrelated
        let x = Mat::from_rows(&[
            &[1.0], &[2.0], &[3.0], &[4.0], &[5.0],
            &[6.0], &[7.0], &[8.0], &[9.0], &[10.0],
        ]);
        let y = Mat::from_rows(&[
            &[5.0], &[3.0], &[7.0], &[1.0], &[9.0],
            &[2.0], &[8.0], &[4.0], &[6.0], &[10.0],
        ]);
        let res = cca(&x, &y);
        // With reordered Y, correlation should be modest
        assert!(res.correlations[0] < 0.8, "Should not show strong canonical correlation");
    }

    // ── Mardia normality ────────────────────────────────────────────────

    #[test]
    fn mardia_normal_data() {
        // Approximately normal 2D data (grid around mean)
        let mut data = Vec::new();
        for i in 0..20 {
            let x = (i as f64 - 10.0) * 0.3;
            let y = x * 0.5 + (i as f64 * 1.23456).sin() * 0.3;
            data.push(x);
            data.push(y);
        }
        let x = Mat::from_vec(20, 2, data);
        let res = mardia_normality(&x);
        // Should not reject normality for approximately normal data
        assert!(res.skewness >= 0.0, "Skewness statistic should be non-negative");
        assert!(res.kurtosis > 0.0, "Kurtosis statistic should be positive");
    }
}
