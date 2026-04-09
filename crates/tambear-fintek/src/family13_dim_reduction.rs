//! Family 13 — Dimension Reduction / Multivariate.
//!
//! Covers fintek leaves: `pca` (delay embedding), `ssa`, `tick_compression`.
//! NOT covered: `ica`, `rmt`, `grassmannian`, `spectral_embedding`, `diff_geometry`
//! (partial GAPs — composition needed).

use tambear::dim_reduction::{pca, PcaResult};

/// PCA on a delay-embedded time series.
///
/// Builds the trajectory matrix via delay embedding (m dimensions at lag tau),
/// then runs tambear's PCA. Returns top-k eigenvalues and explained variance ratios.
#[derive(Debug, Clone)]
pub struct DelayPcaResult {
    pub eigenvalues: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    /// Effective rank = exp(Shannon entropy of normalized eigenvalues).
    pub effective_rank: f64,
}

impl DelayPcaResult {
    pub fn nan(k: usize) -> Self {
        Self {
            eigenvalues: vec![f64::NAN; k],
            explained_variance_ratio: vec![f64::NAN; k],
            effective_rank: f64::NAN,
        }
    }
}

/// Compute PCA on delay-embedded series.
///
/// `data`: bin-level series.
/// `m`: embedding dimension.
/// `tau`: embedding lag.
/// `n_components`: top components to return.
pub fn delay_pca(data: &[f64], m: usize, tau: usize, n_components: usize) -> DelayPcaResult {
    let n = data.len();
    let n_rows = n.saturating_sub((m - 1) * tau);
    if n_rows < m || m < 2 {
        return DelayPcaResult::nan(n_components);
    }
    // Build trajectory matrix: row i = [data[i], data[i+tau], ..., data[i+(m-1)*tau]]
    let mut traj = Vec::with_capacity(n_rows * m);
    for i in 0..n_rows {
        for j in 0..m {
            traj.push(data[i + j * tau]);
        }
    }
    let result: PcaResult = pca(&traj, n_rows, m, n_components);

    // Effective rank via Shannon entropy of normalized eigenvalues
    let total: f64 = result.singular_values.iter().map(|s| s * s).sum();
    let effective_rank = if total > 1e-15 {
        let mut h = 0.0;
        for s in &result.singular_values {
            let ev = s * s;
            if ev > 0.0 {
                let p = ev / total;
                h -= p * p.ln();
            }
        }
        h.exp()
    } else { f64::NAN };

    DelayPcaResult {
        eigenvalues: result.singular_values.iter().map(|s| s * s).collect(),
        explained_variance_ratio: result.explained_variance_ratio,
        effective_rank,
    }
}

/// Singular Spectrum Analysis: SVD on delay-embedded trajectory matrix.
///
/// Returns top-k singular values. Used for trend/noise decomposition.
pub fn ssa(data: &[f64], window: usize, n_components: usize) -> DelayPcaResult {
    // SSA is essentially PCA on Hankel matrix (delay embedding with tau=1)
    delay_pca(data, window, 1, n_components)
}

/// Tick compression: effective rank of delay-embedded series.
///
/// Returns just the effective rank (exp of Shannon entropy of eigenvalues).
pub fn tick_compression(data: &[f64], m: usize, tau: usize) -> f64 {
    delay_pca(data, m, tau, m).effective_rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_pca_sine() {
        // Pure sine should have rank 2 (sin and cos components in delay space)
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = delay_pca(&data, 5, 3, 5);
        assert!(r.effective_rank.is_finite());
        assert!(r.effective_rank < 3.0, "Sine delay PCA effective rank should be low, got {}", r.effective_rank);
    }

    #[test]
    fn delay_pca_too_short() {
        let r = delay_pca(&[1.0, 2.0], 5, 3, 3);
        assert!(r.effective_rank.is_nan());
    }

    #[test]
    fn tick_compression_random() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let er = tick_compression(&data, 5, 1);
        // Random noise should have high effective rank (near 5)
        assert!(er > 2.0, "Random effective rank should be moderately high, got {}", er);
    }

    #[test]
    fn ssa_basic() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).cos() + 0.1 * i as f64).collect();
        let r = ssa(&data, 10, 5);
        assert_eq!(r.eigenvalues.len(), 5);
        assert!(r.eigenvalues[0] >= r.eigenvalues[1]); // sorted descending
    }
}
