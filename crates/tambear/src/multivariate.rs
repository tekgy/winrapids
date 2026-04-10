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
    Mat, mat_mul, mat_add, mat_scale, cholesky, cholesky_solve, sym_eigen, svd, qr_solve,
    forward_solve, back_solve_transpose,
};
use crate::special_functions::{f_right_tail_p, chi2_right_tail_p, normal_two_tail_p};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: Covariance matrices from data
// ═══════════════════════════════════════════════════════════════════════════

/// Compute sample covariance matrix (Bessel-corrected) from n×p data matrix.
/// Each row is an observation, each column is a variable.
/// Sample covariance matrix of an n×p data matrix.
///
/// Returns a p×p symmetric matrix where entry (j,k) is the covariance
/// between columns j and k. Uses Bessel correction by default (divides
/// by n-1 for unbiased estimation).
///
/// # Parameters
/// - `x`: n×p data matrix (n observations, p variables)
/// - `ddof`: delta degrees of freedom for the denominator.
///   `None` → default 1 (sample covariance, divides by n-1).
///   `Some(0)` → population covariance (divides by n).
///   `Some(k)` → divides by n-k.
///
/// # Consumers
/// PCA, factor analysis, LDA, Mahalanobis distance, mixed effects,
/// multivariate tests, CCA, portfolio optimization. Any method that
/// needs the second-moment structure of multivariate data.
///
/// Kingdom A: single-pass accumulate over rows.
pub fn covariance_matrix(x: &Mat, ddof: Option<usize>) -> Mat {
    let n = x.rows;
    let p = x.cols;
    let means = col_means(x);
    let ddof = ddof.unwrap_or(1);

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
    let denom = (n.saturating_sub(ddof)).max(1) as f64;
    for j in 0..p {
        for k in 0..p {
            cov.set(j, k, cov.get(j, k) / denom);
        }
    }
    cov
}

/// Column means of an n×p matrix.
///
/// Returns a vector of length p where entry j is the arithmetic mean
/// of column j. Standalone primitive — used by covariance_matrix,
/// centering, PCA preprocessing, any column-wise summary.
///
/// Kingdom A: single-pass accumulate over rows.
pub fn col_means(x: &Mat) -> Vec<f64> {
    let n = x.rows as f64;
    let mut m = vec![0.0; x.cols];
    for i in 0..x.rows {
        for j in 0..x.cols { m[j] += x.get(i, j); }
    }
    for j in 0..x.cols { m[j] /= n; }
    m
}

/// Within-group and between-group SSCP (sum-of-squares-and-cross-products) matrices.
///
/// Used by MANOVA, LDA, and any multivariate method that needs the decomposition
/// of total variation into within-group and between-group components.
///
/// # Parameters
/// - `x`: n×p data matrix (n observations, p variables)
/// - `groups`: group label (0-indexed) for each observation
///
/// # Returns
/// `(W, H, group_means, group_counts, grand_mean)` where:
/// - `W`: p×p within-group SSCP matrix
/// - `H`: p×p between-group SSCP matrix
/// - `group_means`: k×p matrix of per-group column means
/// - `group_counts`: number of observations per group
/// - `grand_mean`: p-vector of overall column means
///
/// # Consumers
/// MANOVA (Wilks, Pillai, Hotelling, Roy statistics), LDA (W⁻¹H eigendecomposition).
///
/// Kingdom A: single accumulate pass over rows.
pub fn sscp_matrices(x: &Mat, groups: &[usize]) -> (Mat, Mat, Vec<Vec<f64>>, Vec<usize>, Vec<f64>) {
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
    if n <= p {
        return HotellingResult { t2: f64::NAN, f_statistic: f64::NAN, df1: p as f64, df2: f64::NAN, p_value: f64::NAN };
    }

    let means = col_means(x);
    let cov = covariance_matrix(x, None);
    let l = match cholesky(&cov) {
        Some(l) => l,
        None => return HotellingResult { t2: f64::NAN, f_statistic: f64::NAN, df1: p as f64, df2: f64::NAN, p_value: f64::NAN },
    };

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
    if n1 + n2 <= p + 1 {
        return HotellingResult { t2: f64::NAN, f_statistic: f64::NAN, df1: p as f64, df2: f64::NAN, p_value: f64::NAN };
    }

    let m1 = col_means(x1);
    let m2 = col_means(x2);
    let s1 = covariance_matrix(x1, None);
    let s2 = covariance_matrix(x2, None);

    // Pooled covariance: S_p = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    let sp = {
        let a = mat_scale((n1 - 1) as f64, &s1);
        let b = mat_scale((n2 - 1) as f64, &s2);
        let sum = mat_add(&a, &b);
        mat_scale(1.0 / (n1 + n2 - 2) as f64, &sum)
    };
    let l = match cholesky(&sp) {
        Some(l) => l,
        None => return HotellingResult { t2: f64::NAN, f_statistic: f64::NAN, df1: p as f64, df2: f64::NAN, p_value: f64::NAN },
    };

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
    // df2 depends on whether s = p or s = k-1:
    //   s = p  (p <= k-1): df2 = s*(N-k)
    //   s = k-1 (k-1 < p): df2 = s*(N-p-1)
    let sf = s as f64;
    let pf = p as f64;
    let kf = k as f64;
    let nf = n as f64;
    let df1 = sf * pf;
    let df2 = if p <= k - 1 { sf * (nf - kf) } else { sf * (nf - pf - 1.0) };
    let f_stat = if df2 > 0.0 && sf > 0.0 {
        (pillai_trace / sf) / ((sf - pillai_trace) / sf) * (df2 / df1)
    } else { 0.0 };
    let p_value = if df1 > 0.0 && df2 > 0.0 { f_right_tail_p(f_stat, df1, df2) } else { 1.0 };

    ManovaResult {
        wilks_lambda, pillai_trace, hotelling_lawley,
        roy_largest_root, eigenvalues, f_statistic: f_stat, p_value,
    }
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
    let cov_xx = covariance_matrix(x, None);
    let cov_yy = covariance_matrix(y, None);

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
    let cov = covariance_matrix(x, None);
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
// Variance Inflation Factor (VIF) for multicollinearity detection
// ═══════════════════════════════════════════════════════════════════════════

/// Variance Inflation Factor for each predictor in a design matrix.
///
/// VIF_j = 1 / (1 - R²_j), where R²_j is the R² from regressing column j
/// on all other columns. Interpretation:
/// - VIF < 5: acceptable
/// - VIF 5–10: moderate multicollinearity
/// - VIF > 10: severe multicollinearity — consider removing or combining predictors
///
/// `x`: n×p design matrix (row-major, columns are predictors).
/// Returns a Vec of length p. VIF = Inf when a predictor is a perfect linear
/// combination of others (R² = 1). VIF = 1 when completely uncorrelated.
///
/// Note: intercept column (all ones) should be excluded — VIF for an intercept
/// is undefined and always inflated.
pub fn vif(x: &Mat) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;

    if p < 2 {
        // Single predictor: no collinearity possible
        return vec![1.0; p];
    }

    let mut result = Vec::with_capacity(p);

    for j in 0..p {
        // Build design matrix with column j as response, all others as predictors
        let mut x_other = Mat::zeros(n, p - 1);
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = x.data[i * p + j];
            let mut col = 0;
            for k in 0..p {
                if k == j { continue; }
                x_other.data[i * (p - 1) + col] = x.data[i * p + k];
                col += 1;
            }
        }

        // OLS: β = (X'X)⁻¹X'y via QR
        let beta = qr_solve(&x_other, &y);

        // Fitted values and SS_res
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot < 1e-300 {
            // Constant column: VIF undefined, return Inf
            result.push(f64::INFINITY);
            continue;
        }

        let mut ss_res = 0.0;
        for i in 0..n {
            let fitted: f64 = (0..p-1).map(|k| x_other.data[i * (p-1) + k] * beta[k]).sum();
            ss_res += (y[i] - fitted).powi(2);
        }

        let r2 = 1.0 - ss_res / ss_tot;
        let r2 = r2.clamp(0.0, 1.0 - 1e-15); // guard against perfect collinearity
        result.push(1.0 / (1.0 - r2));
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Mahalanobis distance for outlier detection
// ═══════════════════════════════════════════════════════════════════════════

/// Mahalanobis distance from each point to the sample mean, plus chi-squared p-values.
///
/// D²(xᵢ) = (xᵢ - x̄)ᵀ S⁻¹ (xᵢ - x̄)
///
/// where S is the sample covariance matrix. Under multivariate normality,
/// D²(xᵢ) ~ χ²(p) approximately, enabling outlier detection via p-values.
///
/// Returns `(d2, p_values)`:
/// - `d2[i]` = squared Mahalanobis distance for point i
/// - `p_values[i]` = right-tail p-value from χ²(p): small p → likely outlier
///
/// Returns `None` if n ≤ p (under-determined covariance) or Cholesky fails
/// (singular covariance matrix).
pub fn mahalanobis_distances(x: &Mat) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = x.rows;
    let p = x.cols;
    if n <= p { return None; }

    // Sample mean
    let mut mean = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p { mean[j] += x.data[i * p + j]; }
    }
    for j in 0..p { mean[j] /= n as f64; }

    // Sample covariance S = (1/(n-1)) Σ (xᵢ-x̄)(xᵢ-x̄)ᵀ
    let mut s_data = vec![0.0_f64; p * p];
    for i in 0..n {
        for r in 0..p {
            for c in 0..p {
                let dr = x.data[i * p + r] - mean[r];
                let dc = x.data[i * p + c] - mean[c];
                s_data[r * p + c] += dr * dc;
            }
        }
    }
    let scale = 1.0 / (n as f64 - 1.0);
    for v in s_data.iter_mut() { *v *= scale; }
    let s_mat = Mat { rows: p, cols: p, data: s_data };

    // Cholesky factorization of S (fails if singular)
    let l = cholesky(&s_mat)?;

    // D²(xᵢ) = ‖L⁻¹(xᵢ - x̄)‖² = (xᵢ-x̄)ᵀ S⁻¹ (xᵢ-x̄)
    let pf = p as f64;
    let mut d2 = Vec::with_capacity(n);
    let mut p_values = Vec::with_capacity(n);
    for i in 0..n {
        let diff: Vec<f64> = (0..p).map(|j| x.data[i * p + j] - mean[j]).collect();
        // D²(xᵢ) = diffᵀ S⁻¹ diff.
        // cholesky_solve returns S⁻¹·diff. Taking the dot product with diff gives D².
        let s_inv_diff = cholesky_solve(&l, &diff);
        let di2: f64 = diff.iter().zip(s_inv_diff.iter()).map(|(a, b)| a * b).sum();
        d2.push(di2);
        // chi2 right-tail p-value
        let pv = crate::special_functions::chi2_right_tail_p(di2, pf);
        p_values.push(pv);
    }

    Some((d2, p_values))
}

// ═══════════════════════════════════════════════════════════════════════════
// Regularized regression: Ridge, Lasso, Elastic Net
// ═══════════════════════════════════════════════════════════════════════════

/// Result of regularized regression.
#[derive(Debug, Clone)]
pub struct RegularizedResult {
    /// Coefficient vector (length p, does NOT include intercept).
    pub beta: Vec<f64>,
    /// Intercept term.
    pub intercept: f64,
    /// Residual sum of squares.
    pub rss: f64,
    /// R² on training data.
    pub r2: f64,
    /// Number of non-zero coefficients (for Lasso/Elastic Net sparsity).
    pub n_nonzero: usize,
    /// Number of iterations (for Lasso/Elastic Net coordinate descent).
    pub iterations: usize,
}

/// Ridge regression: OLS with L2 penalty.
///
/// Minimizes ‖y - Xβ - β₀‖² + λ‖β‖²
///
/// Closed-form solution: β = (X'X + λI)⁻¹ X'y (after centering).
/// Ridge never sets coefficients exactly to zero; it shrinks them toward zero.
///
/// `x`: n × p design matrix (row-major, NO intercept column).
/// `y`: response vector (length n).
/// `lambda`: L2 penalty strength (λ ≥ 0). λ=0 reduces to OLS.
pub fn ridge(x: &Mat, y: &[f64], lambda: f64) -> RegularizedResult {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(y.len(), n);

    // Center X and y
    let mut x_means = vec![0.0; p];
    let y_mean = y.iter().sum::<f64>() / n as f64;
    for j in 0..p {
        x_means[j] = (0..n).map(|i| x.data[i * p + j]).sum::<f64>() / n as f64;
    }

    // Build X'X + λI and X'y (centered)
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let yi = y[i] - y_mean;
        for j in 0..p {
            let xij = x.data[i * p + j] - x_means[j];
            xty[j] += xij * yi;
            for k in 0..p {
                let xik = x.data[i * p + k] - x_means[k];
                xtx[j * p + k] += xij * xik;
            }
        }
    }
    // Add ridge penalty
    for j in 0..p {
        xtx[j * p + j] += lambda;
    }

    // Solve via Cholesky (X'X + λI is always PD for λ > 0)
    let a = Mat::from_vec(p, p, xtx);
    let beta = match cholesky(&a) {
        Some(l) => cholesky_solve(&l, &xty),
        None => {
            // Fallback to QR if Cholesky fails (shouldn't happen with λ > 0)
            qr_solve(&a, &xty)
        }
    };

    // Intercept: β₀ = ȳ - Σ β_j · x̄_j
    let intercept = y_mean - beta.iter().zip(x_means.iter()).map(|(b, m)| b * m).sum::<f64>();

    // RSS and R²
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let mut rss = 0.0;
    for i in 0..n {
        let predicted = intercept + (0..p).map(|j| beta[j] * x.data[i * p + j]).sum::<f64>();
        rss += (y[i] - predicted).powi(2);
    }
    let r2 = if ss_tot < 1e-300 { 0.0 } else { 1.0 - rss / ss_tot };

    RegularizedResult {
        n_nonzero: beta.iter().filter(|&&b| b.abs() > 1e-10).count(),
        beta, intercept, rss, r2: r2.clamp(0.0, 1.0), iterations: 1,
    }
}

/// Lasso regression: OLS with L1 penalty via coordinate descent.
///
/// Minimizes ‖y - Xβ - β₀‖² / (2n) + λ‖β‖₁
///
/// Solved via cyclic coordinate descent with soft-thresholding.
/// Lasso can set coefficients exactly to zero (feature selection).
///
/// `x`: n × p design matrix (row-major, NO intercept column).
/// `y`: response vector (length n).
/// `lambda`: L1 penalty strength (λ ≥ 0). λ=0 reduces to OLS.
/// `max_iter`: maximum coordinate descent iterations.
/// `tol`: convergence tolerance on max coefficient change.
pub fn lasso(x: &Mat, y: &[f64], lambda: f64, max_iter: usize, tol: f64) -> RegularizedResult {
    elastic_net(x, y, lambda, 1.0, max_iter, tol)
}

/// Elastic Net regression: combined L1 + L2 penalty via coordinate descent.
///
/// Minimizes ‖y - Xβ - β₀‖² / (2n) + λ · [α‖β‖₁ + (1-α)‖β‖²/2]
///
/// `alpha`: mixing parameter. α=1 is Lasso, α=0 is Ridge.
/// `lambda`: total penalty strength.
/// `max_iter`: maximum coordinate descent iterations.
/// `tol`: convergence tolerance.
pub fn elastic_net(x: &Mat, y: &[f64], lambda: f64, alpha: f64, max_iter: usize, tol: f64) -> RegularizedResult {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(y.len(), n);
    let nf = n as f64;

    // Center X and y
    let mut x_means = vec![0.0; p];
    let y_mean = y.iter().sum::<f64>() / nf;
    for j in 0..p {
        x_means[j] = (0..n).map(|i| x.data[i * p + j]).sum::<f64>() / nf;
    }
    let y_centered: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();

    // Precompute X'X diagonal (for coordinate descent denominator)
    let mut x_col_sq = vec![0.0; p];
    for j in 0..p {
        for i in 0..n {
            let xij = x.data[i * p + j] - x_means[j];
            x_col_sq[j] += xij * xij;
        }
    }

    // Initialize beta = 0
    let mut beta = vec![0.0; p];
    let mut residual = y_centered.clone();
    let mut iterations = 0;

    // Soft-thresholding operator
    let soft_threshold = |z: f64, gamma: f64| -> f64 {
        if z > gamma { z - gamma }
        else if z < -gamma { z + gamma }
        else { 0.0 }
    };

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut max_change = 0.0_f64;

        for j in 0..p {
            // Add back current beta_j's contribution to residual
            if beta[j] != 0.0 {
                for i in 0..n {
                    residual[i] += beta[j] * (x.data[i * p + j] - x_means[j]);
                }
            }

            // Compute partial residual correlation
            let mut rho = 0.0;
            for i in 0..n {
                rho += (x.data[i * p + j] - x_means[j]) * residual[i];
            }

            // Update beta_j via soft-thresholding
            let denom = x_col_sq[j] + nf * lambda * (1.0 - alpha);
            let new_beta = if denom < 1e-300 {
                0.0
            } else {
                soft_threshold(rho, nf * lambda * alpha) / denom
            };

            let change = (new_beta - beta[j]).abs();
            if change > max_change { max_change = change; }
            beta[j] = new_beta;

            // Subtract new beta_j's contribution from residual
            if beta[j] != 0.0 {
                for i in 0..n {
                    residual[i] -= beta[j] * (x.data[i * p + j] - x_means[j]);
                }
            }
        }

        if max_change < tol { break; }
    }

    // Intercept
    let intercept = y_mean - beta.iter().zip(x_means.iter()).map(|(b, m)| b * m).sum::<f64>();

    // RSS and R²
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let mut rss = 0.0;
    for i in 0..n {
        let predicted = intercept + (0..p).map(|j| beta[j] * x.data[i * p + j]).sum::<f64>();
        rss += (y[i] - predicted).powi(2);
    }
    let r2 = if ss_tot < 1e-300 { 0.0 } else { 1.0 - rss / ss_tot };

    RegularizedResult {
        n_nonzero: beta.iter().filter(|&&b| b.abs() > 1e-10).count(),
        beta, intercept, rss, r2: r2.clamp(0.0, 1.0), iterations,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Weighted Least Squares (WLS)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of weighted least squares regression.
#[derive(Debug, Clone)]
pub struct WlsResult {
    /// Coefficient vector (length p).
    pub beta: Vec<f64>,
    /// Weighted residual sum of squares: Σ w_i (y_i - x_i'β)².
    pub wrss: f64,
    /// Weighted R².
    pub r2: f64,
}

/// Weighted least squares: β = (X'WX)⁻¹ X'Wy.
///
/// Minimizes Σ w_i (y_i - x_i'β)². Equivalent to OLS on the transformed
/// system √w_i · y_i = √w_i · x_i'β + ε_i.
///
/// Use when observations have known unequal variances: set w_i = 1/σ²_i.
///
/// `x`: n × p design matrix (row-major, include intercept column if desired).
/// `y`: response vector (length n).
/// `weights`: non-negative weights (length n). Zero weights exclude observations.
pub fn wls(x: &Mat, y: &[f64], weights: &[f64]) -> WlsResult {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(y.len(), n);
    assert_eq!(weights.len(), n);

    // X'WX and X'Wy
    let mut xtwx = vec![0.0; p * p];
    let mut xtwy = vec![0.0; p];
    for i in 0..n {
        let w = weights[i].max(0.0);
        for j in 0..p {
            xtwy[j] += w * x.data[i * p + j] * y[i];
            for k in 0..p {
                xtwx[j * p + k] += w * x.data[i * p + j] * x.data[i * p + k];
            }
        }
    }

    let a = Mat::from_vec(p, p, xtwx);
    let beta = match cholesky(&a) {
        Some(l) => cholesky_solve(&l, &xtwy),
        None => qr_solve(&a, &xtwy),
    };

    // Weighted RSS and R²
    let mut wrss = 0.0;
    let mut w_total = 0.0;
    let mut wy_sum = 0.0;
    for i in 0..n {
        let w = weights[i].max(0.0);
        let fitted: f64 = (0..p).map(|j| beta[j] * x.data[i * p + j]).sum();
        wrss += w * (y[i] - fitted).powi(2);
        w_total += w;
        wy_sum += w * y[i];
    }
    let wy_mean = if w_total > 1e-300 { wy_sum / w_total } else { 0.0 };
    let ss_tot: f64 = (0..n).map(|i| {
        let w = weights[i].max(0.0);
        w * (y[i] - wy_mean).powi(2)
    }).sum();
    let r2 = if ss_tot < 1e-300 { 0.0 } else { (1.0 - wrss / ss_tot).clamp(0.0, 1.0) };

    WlsResult { beta, wrss, r2 }
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

    // ── Regression: Pillai df2 when p > k-1 ────────────────────────────
    // Old code always used df2 = s*(N-k). When p > k-1, the correct
    // formula is df2 = s*(N-p-1). This test verifies the conditional.
    #[test]
    fn manova_pillai_df2_p_gt_k_minus_1_regression() {
        // 2 groups, 3 variables → p=3, k=2, k-1=1, s=min(p,k-1)=1
        // Since p > k-1, df2 should be s*(N-p-1) = 1*(N-4), not s*(N-k) = 1*(N-2)
        let x = Mat::from_rows(&[
            &[0.0, 0.0, 0.0], &[0.1, 0.1, 0.1], &[-0.1, -0.1, 0.0],
            &[0.05, -0.05, 0.1], &[0.0, 0.0, -0.1],
            &[5.0, 5.0, 5.0], &[5.1, 5.1, 5.1], &[4.9, 4.9, 5.0],
            &[5.05, 4.95, 5.1], &[5.0, 5.0, 4.9],
        ]);
        let groups = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let res = manova(&x, &groups);

        // With well-separated groups, should strongly reject
        assert!(res.p_value < 0.01,
            "MANOVA with p>k-1 should still detect clear separation, p={}", res.p_value);
        // Verify reasonable F statistic
        assert!(res.f_statistic > 0.0, "F should be positive");
    }

    // ── VIF ─────────────────────────────────────────────────────────────

    #[test]
    fn vif_orthogonal_predictors() {
        // Orthogonal predictors: no collinearity → all VIF = 1
        let x = Mat::from_rows(&[
            &[1.0, 0.0], &[-1.0, 0.0], &[0.0, 1.0], &[0.0, -1.0],
            &[1.0, 0.0], &[-1.0, 0.0], &[0.0, 1.0], &[0.0, -1.0],
        ]);
        let v = vif(&x);
        assert_eq!(v.len(), 2);
        for (j, &vi) in v.iter().enumerate() {
            assert!((vi - 1.0).abs() < 0.1, "VIF[{j}]={vi} should be ≈1 for orthogonal predictors");
        }
    }

    #[test]
    fn vif_high_collinearity() {
        // x2 ≈ x1 + small noise → strong multicollinearity → high VIF
        let n = 20;
        let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x2: Vec<f64> = x1.iter().map(|&v| v + 0.01 * ((v * 7.3) % 1.0)).collect();
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(x1[i]);
            data.push(x2[i]);
        }
        let x = Mat::from_vec(n, 2, data);
        let v = vif(&x);
        assert!(v[0] > 100.0, "VIF[0]={} should be very high (near-collinear)", v[0]);
        assert!(v[1] > 100.0, "VIF[1]={} should be very high (near-collinear)", v[1]);
    }

    #[test]
    fn vif_single_predictor_returns_one() {
        let x = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let v = vif(&x);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 1.0, "single predictor VIF should be exactly 1");
    }

    #[test]
    fn vif_three_uncorrelated_predictors() {
        // Three independent predictors → VIF all ≈ 1
        let x = Mat::from_rows(&[
            &[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0], &[0.0, -1.0, 0.0],
            &[0.0, 0.0, 1.0], &[0.0, 0.0, -1.0],
            &[1.0, 1.0, 0.0], &[-1.0, -1.0, 0.0],
        ]);
        let v = vif(&x);
        assert_eq!(v.len(), 3);
        for (j, &vi) in v.iter().enumerate() {
            assert!(vi < 5.0, "VIF[{j}]={vi} should be low for uncorrelated predictors");
        }
    }

    // ── Mahalanobis distance ──────────────────────────────────────────────

    #[test]
    fn mahalanobis_centroid_has_zero_distance() {
        // The centroid of the distribution has D²=0.
        // Use identity covariance: D² = Euclidean² from mean.
        // Points symmetrically placed so mean = (0,0).
        let data = Mat::from_rows(&[
            &[1.0, 0.0], &[-1.0, 0.0], &[0.0, 1.0], &[0.0, -1.0],
        ]);
        let (d2, _) = mahalanobis_distances(&data).unwrap();
        assert_eq!(d2.len(), 4);
        // All points are equidistant from mean (symmetric)
        let first = d2[0];
        for &di in &d2 {
            assert!((di - first).abs() < 1e-10, "symmetric data: all D² equal, got {di}");
        }
    }

    #[test]
    fn mahalanobis_outlier_has_large_distance() {
        // Use a large symmetric cluster so the mean stays near (0,0) and the
        // sample covariance stays near I₂, making D² ≈ squared Euclidean distance.
        // Cluster points are the 4 cardinal unit vectors repeated many times
        // (mean = 0, covariance ≈ I). Outlier at (5,0) should have D²≈25.
        //
        // With N=200 balanced points, one outlier at (5,0) can only shift the
        // mean by ~5/201 ≈ 0.025, and barely perturbs the covariance.
        let mut data_flat: Vec<f64> = Vec::new();
        // 50 copies of each of the 4 cardinal unit vectors → 200 points, mean=(0,0)
        for _ in 0..50 {
            data_flat.extend_from_slice(&[1.0, 0.0]);
            data_flat.extend_from_slice(&[-1.0, 0.0]);
            data_flat.extend_from_slice(&[0.0, 1.0]);
            data_flat.extend_from_slice(&[0.0, -1.0]);
        }
        // Outlier at (5, 0)
        let n_cluster = 200;
        data_flat.push(5.0);
        data_flat.push(0.0);
        let n_total = n_cluster + 1;
        let data = Mat { rows: n_total, cols: 2, data: data_flat };
        let (d2, _p_values) = mahalanobis_distances(&data).unwrap();
        let outlier_d2 = d2[n_cluster];
        let max_non_outlier = d2[..n_cluster].iter().cloned().fold(0.0f64, f64::max);
        assert!(outlier_d2 > max_non_outlier,
            "outlier D²={outlier_d2:.2} should be > max non-outlier D²={max_non_outlier:.2}");
        // Outlier at (5,0) from distribution with σ≈1 → D²≈25; cluster max is ≈1
        assert!(outlier_d2 > 5.0 * max_non_outlier,
            "outlier D²={outlier_d2:.2} should be >> cluster D²={max_non_outlier:.4}");
    }

    #[test]
    fn mahalanobis_underdetermined_returns_none() {
        // n <= p: singular covariance → should return None
        let data = Mat::from_rows(&[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
        ]);
        // n=2, p=3: under-determined
        assert!(mahalanobis_distances(&data).is_none());
    }

    #[test]
    fn mahalanobis_d2_chi2_distribution() {
        // For multivariate normal data, D²ᵢ ~ χ²(p) approximately
        // So the mean of D² values should be approximately p
        let n = 100;
        let p = 2;
        let mut rng = 0xdeadbeef_u64;
        let data_flat: Vec<f64> = (0..n * p).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 * 1.7320508 // uniform ≈ N(0,1)
        }).collect();
        let data = Mat { rows: n, cols: p, data: data_flat };
        let (d2, _) = mahalanobis_distances(&data).unwrap();
        let mean_d2: f64 = d2.iter().sum::<f64>() / n as f64;
        // E[D²] = p for χ²(p). Tolerance: ±p (very loose).
        assert!((mean_d2 - p as f64).abs() < p as f64,
            "mean D²={mean_d2:.2} should be near p={p}");
    }

    // ── Ridge regression ─────────────────────────────────────────────────

    #[test]
    fn ridge_recovers_ols_at_zero_lambda() {
        // With λ=0, ridge should match OLS on clean linear data y=2x+3
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x_data.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let result = ridge(&x, &y, 0.0);
        close(result.beta[0], 2.0, 1e-6, "ridge coeff ≈ 2 (OLS)");
        close(result.intercept, 3.0, 1e-4, "ridge intercept ≈ 3");
        close(result.r2, 1.0, 1e-6, "R² = 1 for perfect linear fit");
    }

    #[test]
    fn ridge_shrinks_toward_zero() {
        // With large λ, ridge coefficients shrink toward zero
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x_data.iter().map(|&xi| 5.0 * xi + 1.0).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let result_small = ridge(&x, &y, 0.01);
        let result_large = ridge(&x, &y, 1000.0);
        assert!(result_large.beta[0].abs() < result_small.beta[0].abs(),
            "larger λ should shrink: small={:.4}, large={:.4}",
            result_small.beta[0], result_large.beta[0]);
    }

    // ── Lasso regression ─────────────────────────────────────────────────

    #[test]
    fn lasso_recovers_signal() {
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x_data.iter().map(|&xi| 3.0 * xi + 0.5).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let result = lasso(&x, &y, 0.001, 1000, 1e-8);
        assert!(result.beta[0] > 1.0,
            "lasso coeff={:.4} should be substantial", result.beta[0]);
        assert!(result.r2 > 0.8, "R²={:.4} should be high", result.r2);
    }

    #[test]
    fn lasso_sparsity_large_lambda() {
        // With large λ, Lasso should zero out the coefficient (constant y)
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let result = lasso(&x, &y, 10.0, 1000, 1e-8);
        assert!(result.beta[0].abs() < 0.01,
            "lasso should zero coeff for constant y, got={:.6}", result.beta[0]);
        assert_eq!(result.n_nonzero, 0, "n_nonzero should be 0");
    }

    // ── ElasticNet regression ─────────────────────────────────────────────

    #[test]
    fn elastic_net_alpha1_is_lasso() {
        // alpha=1 → pure Lasso. Should match lasso().
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x_data.iter().map(|&xi| 2.0 * xi + 0.5).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let lasso_r = lasso(&x, &y, 0.01, 2000, 1e-10);
        let en_r = elastic_net(&x, &y, 0.01, 1.0, 2000, 1e-10);

        close(en_r.beta[0], lasso_r.beta[0], 1e-4, "ElasticNet(α=1) should match Lasso");
    }

    #[test]
    fn elastic_net_shrinks_with_regularization() {
        // ElasticNet with strong regularization should produce smaller coefficients
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x_data.iter().map(|&xi| 5.0 * xi + 1.0).collect();
        let x = Mat::from_vec(n, 1, x_data);

        let strong = elastic_net(&x, &y, 1.0, 0.5, 1000, 1e-8);
        let weak = elastic_net(&x, &y, 0.001, 0.5, 1000, 1e-8);

        assert!(strong.beta[0].abs() < weak.beta[0].abs(),
            "stronger regularization should shrink more: strong={:.4} weak={:.4}",
            strong.beta[0], weak.beta[0]);
    }
}
