//! # COPA — Centered Outer Product Accumulate
//!
//! One-pass, streaming, parallel-mergeable covariance computation.
//! The matrix generalization of scalar M₂ in `MomentStats::merge`.
//!
//! ## Core Idea
//!
//! COPA state = (n, μ, C) where C = Σᵢ(xᵢ-μ)(xᵢ-μ)ᵀ.
//! Merge: C = Cₐ + C_b + (nₐ·n_b/n)·ΔΔᵀ.
//! PCA = eigendecomp of C/(n-1). No SVD needed for tall matrices.
//!
//! ## Why not SVD?
//!
//! Standard PCA computes SVD of the n×d centered matrix — O(n·d²) with n×d memory.
//! COPA accumulates into a d×d matrix in one pass — O(n·d²) work but only O(d²) state.
//! For tall matrices (n >> d), this is the same work but far less memory.
//! More importantly: COPA states are mergeable (associative semigroup), enabling
//! parallel GPU tile reduction, streaming, and distributed computation.
//!
//! ## Structural Rhyme
//!
//! ```text
//! MomentStats::merge (scalar):  m2 = a.m2 + b.m2 + δ² × nₐ × n_b / n
//! CopaState::merge  (matrix):   C  = Cₐ   + C_b  + (nₐ n_b / n) ΔΔᵀ
//! ```
//! Same formula, higher rank. Same semigroup structure. Same liftability.

use crate::linear_algebra::{Mat, sym_eigen, mat_mul, power_iteration, cholesky_solve, cholesky};

// ═══════════════════════════════════════════════════════════════════════════
// COPA State
// ═══════════════════════════════════════════════════════════════════════════

/// Centered Outer Product Accumulate state.
///
/// Minimum sufficient representation for streaming covariance:
/// (n, μ, C) where C = Σᵢ(xᵢ - μ)(xᵢ - μ)ᵀ.
/// Total size: 1 + p + p(p+1)/2 scalars (symmetric C stored as full p×p).
#[derive(Debug, Clone)]
pub struct CopaState {
    /// Number of observations accumulated.
    pub n: usize,
    /// Dimensionality.
    pub p: usize,
    /// Running mean vector (length p).
    pub mean: Vec<f64>,
    /// Centered cross-product matrix C, p×p row-major (symmetric).
    /// C = Σᵢ(xᵢ - μ)(xᵢ - μ)ᵀ.
    pub c: Vec<f64>,
}

/// PCA result from COPA eigendecomposition.
#[derive(Debug, Clone)]
pub struct CopaPcaResult {
    /// Principal components (columns of p×k matrix, each column is a PC direction).
    pub components: Mat,
    /// Eigenvalues (descending) = variance along each PC.
    pub eigenvalues: Vec<f64>,
    /// Explained variance ratio per component.
    pub explained_variance_ratio: Vec<f64>,
    /// Projected data (n × k) — only available if `project()` is called with data.
    pub transformed: Option<Mat>,
}

impl CopaState {
    /// Create an empty COPA state for p-dimensional data.
    pub fn new(p: usize) -> Self {
        CopaState { n: 0, p, mean: vec![0.0; p], c: vec![0.0; p * p] }
    }

    /// Accumulate a single observation (Welford-style online update).
    pub fn add(&mut self, x: &[f64]) {
        assert_eq!(x.len(), self.p);
        self.n += 1;
        let n = self.n as f64;
        let p = self.p;

        // δ = x - μ_old
        let delta: Vec<f64> = (0..p).map(|j| x[j] - self.mean[j]).collect();

        // Update mean: μ_new = μ_old + δ/n
        for j in 0..p { self.mean[j] += delta[j] / n; }

        // δ' = x - μ_new
        let delta2: Vec<f64> = (0..p).map(|j| x[j] - self.mean[j]).collect();

        // Update C: C += δ ⊗ δ' (outer product of old and new deviations)
        for j in 0..p {
            for k in 0..p {
                self.c[j * p + k] += delta[j] * delta2[k];
            }
        }
    }

    /// Accumulate a batch of observations (n × p row-major).
    pub fn add_batch(&mut self, data: &[f64], n: usize) {
        assert_eq!(data.len(), n * self.p);
        for i in 0..n {
            self.add(&data[i * self.p..(i + 1) * self.p]);
        }
    }

    /// Merge two COPA states (parallel Welford for matrices).
    ///
    /// C = Cₐ + C_b + (nₐ · n_b / n) · ΔΔᵀ
    ///
    /// This is the same formula as MomentStats::merge but for matrices.
    /// Associative — valid as a parallel scan combiner.
    pub fn merge(a: &CopaState, b: &CopaState) -> CopaState {
        assert_eq!(a.p, b.p);
        let p = a.p;

        if a.n == 0 { return b.clone(); }
        if b.n == 0 { return a.clone(); }

        let na = a.n as f64;
        let nb = b.n as f64;
        let n = na + nb;

        // Δ = μ_b - μ_a
        let delta: Vec<f64> = (0..p).map(|j| b.mean[j] - a.mean[j]).collect();

        // Combined mean
        let mean: Vec<f64> = (0..p).map(|j| (na * a.mean[j] + nb * b.mean[j]) / n).collect();

        // C = Cₐ + C_b + (nₐ·n_b/n) · ΔΔᵀ
        let factor = na * nb / n;
        let mut c = vec![0.0; p * p];
        for j in 0..p {
            for k in 0..p {
                c[j * p + k] = a.c[j * p + k] + b.c[j * p + k] + factor * delta[j] * delta[k];
            }
        }

        CopaState { n: a.n + b.n, p, mean, c }
    }

    /// Extract the sample covariance matrix Σ = C / (n-1).
    pub fn covariance(&self) -> Mat {
        let p = self.p;
        if self.n < 2 { return Mat::zeros(p, p); }
        let scale = 1.0 / (self.n - 1) as f64;
        let data: Vec<f64> = self.c.iter().map(|&v| v * scale).collect();
        Mat::from_vec(p, p, data)
    }

    /// Extract the population covariance matrix Σ = C / n.
    pub fn covariance_population(&self) -> Mat {
        let p = self.p;
        if self.n == 0 { return Mat::zeros(p, p); }
        let scale = 1.0 / self.n as f64;
        let data: Vec<f64> = self.c.iter().map(|&v| v * scale).collect();
        Mat::from_vec(p, p, data)
    }

    /// Extract the correlation matrix from the covariance.
    pub fn correlation(&self) -> Mat {
        let p = self.p;
        let cov = self.covariance();
        let stds: Vec<f64> = (0..p).map(|j| cov.get(j, j).sqrt().max(1e-15)).collect();
        let mut corr = Mat::zeros(p, p);
        for j in 0..p {
            for k in 0..p {
                corr.set(j, k, cov.get(j, k) / (stds[j] * stds[k]));
            }
        }
        corr
    }

    /// Per-variable standard deviations (from diagonal of C/(n-1)).
    pub fn std_devs(&self) -> Vec<f64> {
        let p = self.p;
        if self.n < 2 { return vec![0.0; p]; }
        let scale = 1.0 / (self.n - 1) as f64;
        (0..p).map(|j| (self.c[j * p + j] * scale).sqrt()).collect()
    }

    /// PCA via eigendecomposition of the covariance matrix.
    ///
    /// For n >> d, this is far cheaper than SVD of the full n×d matrix.
    /// The covariance is d×d symmetric — eigen is O(d³), independent of n.
    pub fn pca(&self, n_components: usize) -> CopaPcaResult {
        let p = self.p;
        let k = n_components.min(p);
        let cov = self.covariance();

        // Symmetric eigendecomposition: cov = V·Λ·Vᵀ
        let (eigenvalues_unsorted, eigenvectors) = sym_eigen(&cov);

        // Sort by descending eigenvalue
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| eigenvalues_unsorted[b].partial_cmp(&eigenvalues_unsorted[a])
            .unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<f64> = idx[..k].iter().map(|&i| eigenvalues_unsorted[i].max(0.0)).collect();
        let total_var: f64 = eigenvalues_unsorted.iter().map(|v| v.max(0.0)).sum::<f64>().max(1e-15);
        let explained_variance_ratio: Vec<f64> = eigenvalues.iter().map(|v| v / total_var).collect();

        // Components: columns of V corresponding to top-k eigenvalues
        let mut components = Mat::zeros(p, k);
        for c in 0..k {
            let src_col = idx[c];
            for j in 0..p {
                components.set(j, c, eigenvectors.get(j, src_col));
            }
        }

        CopaPcaResult { components, eigenvalues, explained_variance_ratio, transformed: None }
    }

    /// PCA with projection of original data.
    pub fn pca_transform(&self, data: &[f64], n: usize, n_components: usize) -> CopaPcaResult {
        assert_eq!(data.len(), n * self.p);
        let mut result = self.pca(n_components);
        let p = self.p;
        let k = result.eigenvalues.len();

        // Project: (x - μ) · V
        let mut transformed = Mat::zeros(n, k);
        for i in 0..n {
            for c in 0..k {
                let mut val = 0.0;
                for j in 0..p {
                    val += (data[i * p + j] - self.mean[j]) * result.components.get(j, c);
                }
                transformed.set(i, c, val);
            }
        }
        result.transformed = Some(transformed);
        result
    }

    /// Mahalanobis distance of a point from the COPA distribution center.
    ///
    /// d_M(x) = √((x − μ)ᵀ Σ⁻¹ (x − μ))
    ///
    /// A super-Fock extraction: uses only the COPA state (μ, Σ), no data.
    /// Returns None if the covariance is singular (n < p+1).
    pub fn mahalanobis(&self, x: &[f64]) -> Option<f64> {
        assert_eq!(x.len(), self.p);
        if self.n < 2 { return None; }
        let cov = self.covariance();
        let l = cholesky(&cov)?;
        let delta: Vec<f64> = (0..self.p).map(|j| x[j] - self.mean[j]).collect();
        let z = cholesky_solve(&l, &delta);
        let d_sq: f64 = delta.iter().zip(z.iter()).map(|(d, zi)| d * zi).sum();
        Some(d_sq.max(0.0).sqrt())
    }

    /// Level 1 progressive approximation: top eigenvalue via power iteration.
    ///
    /// Returns (eigenvalue, eigenvector) of the covariance matrix Σ = C/(n-1).
    /// Cost: O(d² × iterations), much cheaper than full eigendecomp O(d³).
    /// Gives the dominant principal component and fraction of variance explained.
    pub fn top_eigenvalue(&self) -> (f64, Vec<f64>) {
        let cov = self.covariance();
        power_iteration(&cov, 200, 1e-12)
    }

    /// Fraction of total variance explained by the top principal component.
    ///
    /// Level 1 diagnostic: if this is close to 1.0, the data is nearly 1D
    /// and full PCA is unnecessary. A fast screening test from COPA state.
    pub fn explained_variance_top1(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        let (top_eval, _) = self.top_eigenvalue();
        let total_var: f64 = (0..self.p)
            .map(|j| self.c[j * self.p + j] / (self.n - 1) as f64)
            .sum::<f64>()
            .max(1e-15);
        (top_eval / total_var).clamp(0.0, 1.0)
    }

    /// Bures-Wasserstein distance between two COPA states.
    ///
    /// This is the Wasserstein-2 distance between Gaussian distributions
    /// N(μ₁, Σ₁) and N(μ₂, Σ₂):
    ///
    /// W₂² = ‖μ₁ − μ₂‖² + tr(Σ₁ + Σ₂ − 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
    ///
    /// Cost: O(d³), independent of n. A super-Fock extraction from two COPA
    /// states — no data re-reading required. This enables K04 cross-ticker
    /// distributional distance entirely from accumulated COPA states.
    pub fn bures_wasserstein(&self, other: &CopaState) -> f64 {
        assert_eq!(self.p, other.p);
        let p = self.p;

        // Mean distance: ‖μ₁ − μ₂‖²
        let mean_dist_sq: f64 = (0..p)
            .map(|j| (self.mean[j] - other.mean[j]).powi(2))
            .sum();

        if self.n < 2 || other.n < 2 {
            return mean_dist_sq.sqrt();
        }

        let sigma1 = self.covariance();
        let sigma2 = other.covariance();

        // Compute Σ₁^{1/2} via eigendecomposition
        let (evals1, evecs1) = sym_eigen(&sigma1);

        // Σ₁^{1/2} = V · diag(√λ) · Vᵀ
        let mut sqrt_sigma1 = Mat::zeros(p, p);
        for i in 0..p {
            let sqrt_lambda = evals1[i].max(0.0).sqrt();
            for r in 0..p {
                for c in 0..p {
                    sqrt_sigma1.set(r, c,
                        sqrt_sigma1.get(r, c) + sqrt_lambda * evecs1.get(r, i) * evecs1.get(c, i));
                }
            }
        }

        // M = Σ₁^{1/2} · Σ₂ · Σ₁^{1/2}
        let m = mat_mul(&mat_mul(&sqrt_sigma1, &sigma2), &sqrt_sigma1);

        // tr(M^{1/2}) via eigendecomposition of M
        let (evals_m, _) = sym_eigen(&m);
        let trace_sqrt_m: f64 = evals_m.iter().map(|&v| v.max(0.0).sqrt()).sum();

        // W₂² = ‖Δμ‖² + tr(Σ₁) + tr(Σ₂) − 2·tr(M^{1/2})
        let trace_sigma1: f64 = (0..p).map(|j| sigma1.get(j, j)).sum();
        let trace_sigma2: f64 = (0..p).map(|j| sigma2.get(j, j)).sum();

        let w2_sq = mean_dist_sq + trace_sigma1 + trace_sigma2 - 2.0 * trace_sqrt_m;
        w2_sq.max(0.0).sqrt()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience constructors
// ═══════════════════════════════════════════════════════════════════════════

/// Build a COPA state from a data matrix (n × p row-major) in one pass.
pub fn copa_from_data(data: &[f64], n: usize, p: usize) -> CopaState {
    assert_eq!(data.len(), n * p);
    let mut state = CopaState::new(p);
    state.add_batch(data, n);
    state
}

/// PCA via COPA: one-pass covariance accumulation + eigendecomposition.
/// Drop-in replacement for SVD-based PCA when n >> d.
pub fn copa_pca(data: &[f64], n: usize, d: usize, n_components: usize) -> CopaPcaResult {
    let state = copa_from_data(data, n, d);
    state.pca_transform(data, n, n_components)
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

    // ── Basic accumulation ─────────────────────────────────────────────

    #[test]
    fn copa_mean_correct() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2 matrix
        let state = copa_from_data(&data, 3, 2);
        assert_eq!(state.n, 3);
        close(state.mean[0], 3.0, 1e-10, "mean[0]"); // (1+3+5)/3
        close(state.mean[1], 4.0, 1e-10, "mean[1]"); // (2+4+6)/3
    }

    #[test]
    fn copa_covariance_known() {
        // x = [1, 2, 3], y = [2, 4, 6] → perfect correlation, cov(x,y) = 2.0
        let data = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]; // 3×2
        let state = copa_from_data(&data, 3, 2);
        let cov = state.covariance();
        close(cov.get(0, 0), 1.0, 1e-10, "var(x)");
        close(cov.get(1, 1), 4.0, 1e-10, "var(y)");
        close(cov.get(0, 1), 2.0, 1e-10, "cov(x,y)");
        close(cov.get(1, 0), 2.0, 1e-10, "cov(y,x) symmetry");
    }

    // ── Merge ──────────────────────────────────────────────────────────

    #[test]
    fn copa_merge_equals_batch() {
        // Split data into two halves, merge, should equal batch
        let data = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ]; // 6×2
        let batch = copa_from_data(&data, 6, 2);

        let a = copa_from_data(&data[..6], 3, 2); // first 3 rows
        let b = copa_from_data(&data[6..], 3, 2); // last 3 rows
        let merged = CopaState::merge(&a, &b);

        assert_eq!(merged.n, batch.n);
        for j in 0..2 {
            close(merged.mean[j], batch.mean[j], 1e-10, &format!("mean[{j}]"));
        }
        for j in 0..4 {
            close(merged.c[j], batch.c[j], 1e-10, &format!("c[{j}]"));
        }
    }

    #[test]
    fn copa_merge_associative() {
        // (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C) — semigroup law
        let data = [
            1.0, 5.0,
            2.0, 3.0,
            4.0, 1.0,
            3.0, 7.0,
            6.0, 2.0,
            5.0, 4.0,
        ]; // 6×2

        let a = copa_from_data(&data[..4], 2, 2);
        let b = copa_from_data(&data[4..8], 2, 2);
        let c = copa_from_data(&data[8..], 2, 2);

        let ab_c = CopaState::merge(&CopaState::merge(&a, &b), &c);
        let a_bc = CopaState::merge(&a, &CopaState::merge(&b, &c));

        for j in 0..4 {
            close(ab_c.c[j], a_bc.c[j], 1e-10, &format!("associativity c[{j}]"));
        }
    }

    #[test]
    fn copa_merge_empty() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let state = copa_from_data(&data, 2, 2);
        let empty = CopaState::new(2);

        let m1 = CopaState::merge(&state, &empty);
        let m2 = CopaState::merge(&empty, &state);
        assert_eq!(m1.n, state.n);
        assert_eq!(m2.n, state.n);
        for j in 0..4 {
            close(m1.c[j], state.c[j], 1e-10, "merge with empty left");
            close(m2.c[j], state.c[j], 1e-10, "merge with empty right");
        }
    }

    // ── PCA ────────────────────────────────────────────────────────────

    #[test]
    fn copa_pca_variance_explained_sums_to_one() {
        let data = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            2.0, 3.0, 1.0,
            5.0, 6.0, 4.0,
        ]; // 5×3
        let state = copa_from_data(&data, 5, 3);
        let pca = state.pca(3);
        let sum: f64 = pca.explained_variance_ratio.iter().sum();
        close(sum, 1.0, 1e-6, "variance ratio sum");
    }

    #[test]
    fn copa_pca_first_component_most_variance() {
        let data = [
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
            5.0, 10.0,
            1.5, 3.0,
            2.5, 5.1,
        ]; // 7×2
        let state = copa_from_data(&data, 7, 2);
        let pca = state.pca(2);
        assert!(pca.eigenvalues[0] >= pca.eigenvalues[1],
            "PC1 eigenvalue {} should be >= PC2 {}", pca.eigenvalues[0], pca.eigenvalues[1]);
        assert!(pca.explained_variance_ratio[0] > 0.9,
            "PC1 should explain >90% of variance, got {}", pca.explained_variance_ratio[0]);
    }

    #[test]
    fn copa_pca_matches_svd_pca() {
        // Verify COPA PCA gives same results as SVD-based PCA
        let data = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            2.0, 3.0, 1.0,
            5.0, 6.0, 4.0,
            3.0, 1.0, 2.0,
            8.0, 7.0, 5.0,
            6.0, 4.0, 3.0,
        ]; // 8×3
        let n = 8;
        let d = 3;
        let k = 2;

        // SVD-based PCA
        let svd_result = crate::dim_reduction::pca(&data, n, d, k);

        // COPA PCA
        let copa_result = copa_pca(&data, n, d, k);

        // Eigenvalues should match (up to SVD σ² = eigenvalue * (n-1))
        // SVD: singular_values² / (n-1) = eigenvalues
        for i in 0..k {
            let svd_var = svd_result.singular_values[i].powi(2) / (n - 1) as f64;
            close(copa_result.eigenvalues[i], svd_var, 0.1,
                &format!("eigenvalue[{i}] COPA vs SVD"));
        }

        // Explained variance ratios should match
        for i in 0..k {
            close(copa_result.explained_variance_ratio[i],
                  svd_result.explained_variance_ratio[i], 0.05,
                  &format!("explained_variance_ratio[{i}]"));
        }
    }

    // ── Numerical stability ────────────────────────────────────────────

    #[test]
    fn copa_high_mean_low_variance_stable() {
        // The catastrophic cancellation test from the proof.
        // Data near mean 1e8 with tiny variance — naive formula would fail.
        let base = 1e8;
        let n = 1000;
        let p = 2;
        let mut data = vec![0.0; n * p];
        let mut rng = 42u64;

        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise_x = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise_y = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            data[i * p] = base + noise_x;
            data[i * p + 1] = base + noise_y;
        }

        let state = copa_from_data(&data, n, p);
        let cov = state.covariance();

        // Variance should be ~1/3 (uniform on [-1,1] has var=1/3)
        assert!(cov.get(0, 0) > 0.1 && cov.get(0, 0) < 1.0,
            "var(x)={} should be ~0.33", cov.get(0, 0));
        assert!(cov.get(1, 1) > 0.1 && cov.get(1, 1) < 1.0,
            "var(y)={} should be ~0.33", cov.get(1, 1));
        // Cross-covariance should be ~0 (independent)
        assert!(cov.get(0, 1).abs() < 0.2,
            "cov(x,y)={} should be ~0", cov.get(0, 1));
    }

    #[test]
    fn copa_merge_high_offset_stable() {
        // Two batches with very different means — merge must not lose precision
        let n = 500;
        let p = 2;
        let mut data_a = vec![0.0; n * p];
        let mut data_b = vec![0.0; n * p];
        let mut rng = 77u64;

        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            data_a[i * p] = 1e10 + noise;
            data_a[i * p + 1] = 1e10 + noise * 0.5;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise2 = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            data_b[i * p] = -1e10 + noise2;
            data_b[i * p + 1] = -1e10 + noise2 * 0.5;
        }

        let a = copa_from_data(&data_a, n, p);
        let b = copa_from_data(&data_b, n, p);
        let merged = CopaState::merge(&a, &b);

        // Combined data has mean ~0, huge between-group variance
        assert!(merged.mean[0].abs() < 1e5, "mean should be ~0, got {}", merged.mean[0]);

        let cov = merged.covariance();
        // Total variance dominated by between-group component (1e10)² × factor
        assert!(cov.get(0, 0) > 1e15, "should have huge variance from offset, got {}", cov.get(0, 0));
    }

    // ── Correlation ────────────────────────────────────────────────────

    #[test]
    fn copa_correlation_perfect() {
        let data = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]; // 4×2, y=2x
        let state = copa_from_data(&data, 4, 2);
        let corr = state.correlation();
        close(corr.get(0, 0), 1.0, 1e-10, "corr(x,x)");
        close(corr.get(1, 1), 1.0, 1e-10, "corr(y,y)");
        close(corr.get(0, 1), 1.0, 1e-10, "corr(x,y) perfect");
    }

    // ── GPU-style tile-parallel merge ──────────────────────────────────

    #[test]
    fn copa_tile_parallel_reduction() {
        // Simulate GPU reduction: split 1000×5 data into 8 tiles,
        // accumulate each tile independently, then tree-reduce.
        // Result must match single-pass batch.
        let n = 1000;
        let p = 5;
        let tile_size = 125; // 1000 / 8
        let mut data = vec![0.0; n * p];
        let mut rng = 314u64;

        for i in 0..n * p {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data[i] = (rng as f64 / u64::MAX as f64 - 0.5) * 100.0;
        }

        // Single-pass (ground truth)
        let batch = copa_from_data(&data, n, p);

        // Tile-parallel accumulation
        let mut tiles: Vec<CopaState> = (0..8).map(|t| {
            let start = t * tile_size * p;
            let end = start + tile_size * p;
            copa_from_data(&data[start..end], tile_size, p)
        }).collect();

        // Tree reduction (log₂(8) = 3 levels)
        while tiles.len() > 1 {
            let mut next = Vec::new();
            for pair in tiles.chunks(2) {
                if pair.len() == 2 {
                    next.push(CopaState::merge(&pair[0], &pair[1]));
                } else {
                    next.push(pair[0].clone());
                }
            }
            tiles = next;
        }
        let reduced = &tiles[0];

        // Verify
        assert_eq!(reduced.n, batch.n);
        for j in 0..p {
            close(reduced.mean[j], batch.mean[j], 1e-10, &format!("tile mean[{j}]"));
        }
        for j in 0..p * p {
            close(reduced.c[j], batch.c[j], 1e-8,
                &format!("tile C[{},{}]", j / p, j % p));
        }

        // PCA from tile-reduced state should match batch PCA
        let batch_pca = batch.pca(3);
        let tile_pca = reduced.pca(3);
        for i in 0..3 {
            close(tile_pca.eigenvalues[i], batch_pca.eigenvalues[i], 1e-6,
                &format!("tile PCA eigenvalue[{i}]"));
        }
    }

    #[test]
    fn copa_tile_parallel_uneven() {
        // Uneven tiles — last tile smaller. Merge still exact.
        let n = 107; // not divisible by 4
        let p = 3;
        let mut data = vec![0.0; n * p];
        let mut rng = 271u64;
        for i in 0..n * p {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data[i] = (rng as f64 / u64::MAX as f64) * 50.0;
        }

        let batch = copa_from_data(&data, n, p);

        // 4 uneven tiles
        let tile_sizes = [27, 27, 27, 26];
        let mut offset = 0;
        let mut tiles = Vec::new();
        for &ts in &tile_sizes {
            let start = offset * p;
            tiles.push(copa_from_data(&data[start..start + ts * p], ts, p));
            offset += ts;
        }

        let merged = tiles.iter().skip(1).fold(tiles[0].clone(), |acc, t| CopaState::merge(&acc, t));

        assert_eq!(merged.n, batch.n);
        for j in 0..p * p {
            close(merged.c[j], batch.c[j], 1e-8, &format!("uneven C[{j}]"));
        }
    }

    // ── Descriptive stats from COPA (diagonal = MomentStats m2) ───────

    #[test]
    fn copa_diagonal_matches_moment_stats() {
        // Diagonal of C should equal per-variable m2 from MomentStats
        let data = [
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ]; // 5×2

        let copa = copa_from_data(&data, 5, 2);

        // Manual m2 for column 0: Σ(x - 3)² = 4+1+0+1+4 = 10
        close(copa.c[0 * 2 + 0], 10.0, 1e-10, "C[0,0] = m2 of col 0");
        // Manual m2 for column 1: Σ(x - 30)² = 400+100+0+100+400 = 1000
        close(copa.c[1 * 2 + 1], 1000.0, 1e-10, "C[1,1] = m2 of col 1");
        // Cross: Σ(x-3)(y-30) = (-2)(-20)+(-1)(-10)+0+1*10+2*20 = 40+10+10+40 = 100
        close(copa.c[0 * 2 + 1], 100.0, 1e-10, "C[0,1] = cross m2");

        // Variance = C/(n-1) matches known values
        let stds = copa.std_devs();
        close(stds[0], (10.0_f64 / 4.0).sqrt(), 1e-10, "std[0]");
        close(stds[1], (1000.0_f64 / 4.0).sqrt(), 1e-10, "std[1]");
    }

    // ── Mahalanobis distance ────────────────────────────────────────────

    #[test]
    fn copa_mahalanobis_at_mean_is_zero() {
        // Non-collinear data so covariance matrix is invertible
        let data = [
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]; // 3×2
        let state = copa_from_data(&data, 3, 2);
        let d = state.mahalanobis(&state.mean).unwrap();
        close(d, 0.0, 1e-10, "Mahalanobis at mean = 0");
    }

    #[test]
    fn copa_mahalanobis_scales_with_std() {
        // For uncorrelated data, Mahalanobis = Euclidean normalized by std
        let data = [
            0.0, 0.0,
            2.0, 0.0,
            0.0, 20.0,
            2.0, 20.0,
        ]; // 4×2, mean=(1, 10), var_x=4/3, var_y=400/3
        let state = copa_from_data(&data, 4, 2);
        // Point at (2, 10): 1 std in x, 0 std in y
        let d = state.mahalanobis(&[2.0, 10.0]).unwrap();
        // For uncorrelated: d_M = √((2-1)²/var_x + (10-10)²/var_y) = √(1/var_x)
        let var_x: f64 = 4.0 / 3.0; // sample variance = C[0,0]/(n-1) = (4/3)
        let expected = (1.0_f64 / var_x).sqrt();
        close(d, expected, 0.01, "Mahalanobis scales with std");
    }

    // ── Progressive approximation: top eigenvalue ──────────────────────

    #[test]
    fn copa_top_eigenvalue_matches_full_pca() {
        let data = [
            1.0, 0.0,
            2.0, 0.1,
            3.0, 0.0,
            4.0, -0.1,
            5.0, 0.1,
        ]; // 5×2, dominant variance along col 0
        let state = copa_from_data(&data, 5, 2);
        let (top_eval, _) = state.top_eigenvalue();
        let pca_result = state.pca(2);
        close(top_eval, pca_result.eigenvalues[0], 1e-8, "top eigenvalue matches PCA");
    }

    #[test]
    fn copa_explained_variance_top1_nearly_1d() {
        // Data almost perfectly 1D → top1 should explain ~100%
        let data = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]; // 4×2, perfect correlation
        let state = copa_from_data(&data, 4, 2);
        let ev = state.explained_variance_top1();
        assert!(ev > 0.99, "Nearly 1D data should have top1 > 0.99, got {ev}");
    }

    // ── Bures-Wasserstein distance ────────────────────────────────────

    #[test]
    fn bures_wasserstein_identical() {
        // W₂ between identical distributions = 0
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let state = copa_from_data(&data, 3, 2);
        let d = state.bures_wasserstein(&state);
        close(d, 0.0, 1e-10, "BW(X, X) = 0");
    }

    #[test]
    fn bures_wasserstein_shifted_mean() {
        // Two datasets with same covariance but different means.
        // W₂ should equal ‖μ₁ − μ₂‖.
        let data_a = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]; // 3×2, mean=(1/3, 1/3)
        let data_b = [10.0, 10.0, 11.0, 10.0, 10.0, 11.0]; // same shape, shifted by (10, 10)
        let a = copa_from_data(&data_a, 3, 2);
        let b = copa_from_data(&data_b, 3, 2);
        let d = a.bures_wasserstein(&b);
        // Mean shift: (10-1/3+10-1/3...) — means are (1/3,1/3) and (31/3,31/3)
        // Δμ = (10, 10), ‖Δμ‖ = 10√2 ≈ 14.142
        // Same covariance → W₂ = ‖Δμ‖ exactly
        let expected = (10.0_f64.powi(2) + 10.0_f64.powi(2)).sqrt();
        close(d, expected, 0.01, "BW with same cov = mean distance");
    }

    #[test]
    fn bures_wasserstein_different_spread() {
        // Same mean, different covariance → W₂ > 0
        let data_a = [0.0, 1.0, 2.0, 3.0]; // 2×2
        let data_b = [0.0, 10.0, 20.0, 30.0]; // 2×2, 10× wider
        let a = copa_from_data(&data_a, 2, 2);
        let b = copa_from_data(&data_b, 2, 2);
        let d = a.bures_wasserstein(&b);
        assert!(d > 0.0, "Different spreads should have positive BW distance");
    }

    #[test]
    fn bures_wasserstein_symmetric() {
        // W₂(A, B) = W₂(B, A)
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let data_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 3×2
        let a = copa_from_data(&data_a, 3, 2);
        let b = copa_from_data(&data_b, 3, 2);
        let d_ab = a.bures_wasserstein(&b);
        let d_ba = b.bures_wasserstein(&a);
        close(d_ab, d_ba, 1e-10, "BW symmetry");
    }
}
