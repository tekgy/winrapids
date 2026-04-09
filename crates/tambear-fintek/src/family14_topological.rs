//! Family 14 — Topological / Geometric features.
//!
//! Covers fintek leaves: `persistent_homology`.
//! NOT covered: `nvg`, `hvg`, `tick_geometry` (GAPs — tasks #145).

use tambear::tda::{rips_h0, persistence_entropy, PersistenceDiagram};

/// Persistent homology summary statistics.
#[derive(Debug, Clone)]
pub struct PersistentHomologyResult {
    /// Number of H₀ components.
    pub n_components: f64,
    /// Persistence entropy: Shannon entropy of normalized persistence values.
    pub persistence_entropy: f64,
    /// Maximum persistence (longest-lived H₀ feature, excluding the immortal one).
    pub max_persistence: f64,
    /// Mean persistence across finite-lifetime pairs.
    pub mean_persistence: f64,
}

impl PersistentHomologyResult {
    pub fn nan() -> Self {
        Self {
            n_components: f64::NAN, persistence_entropy: f64::NAN,
            max_persistence: f64::NAN, mean_persistence: f64::NAN,
        }
    }
}

/// Compute H₀ persistence features on a 1D time series via pairwise distance matrix.
///
/// Treats each time point as a point in R^1 with the natural distance.
/// For richer analysis, pass a pre-computed distance matrix.
pub fn persistent_homology_1d(data: &[f64]) -> PersistentHomologyResult {
    let n = data.len();
    if n < 3 { return PersistentHomologyResult::nan(); }

    // Build pairwise |x_i - x_j| distance matrix
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (data[i] - data[j]).abs();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let diagram: PersistenceDiagram = rips_h0(&dist, n);
    let finite_pairs: Vec<_> = diagram.pairs.iter().filter(|p| p.death.is_finite()).collect();

    let n_components = diagram.pairs.len() as f64;
    let persistences: Vec<f64> = finite_pairs.iter().map(|p| p.persistence()).collect();

    let max_persistence = persistences.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_persistence = if persistences.is_empty() {
        f64::NAN
    } else {
        persistences.iter().sum::<f64>() / persistences.len() as f64
    };
    let p_entropy = persistence_entropy(&diagram.pairs);

    PersistentHomologyResult {
        n_components,
        persistence_entropy: p_entropy,
        max_persistence: if max_persistence.is_finite() { max_persistence } else { f64::NAN },
        mean_persistence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_homology_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0];
        let r = persistent_homology_1d(&data);
        assert!(r.n_components > 0.0);
        assert!(r.persistence_entropy.is_finite());
    }

    #[test]
    fn persistent_homology_too_short() {
        let r = persistent_homology_1d(&[1.0, 2.0]);
        assert!(r.n_components.is_nan());
    }
}
