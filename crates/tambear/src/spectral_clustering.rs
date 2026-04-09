//! # Spectral Clustering (Ng–Jordan–Weiss and unnormalized)
//!
//! Spectral clustering treats clustering as a graph partitioning problem:
//!
//! 1. Build an affinity (similarity) graph W from pairwise data.
//! 2. Compute the graph Laplacian L = D − W where D is the degree matrix.
//! 3. Compute the first k eigenvectors of L (smallest eigenvalues for
//!    unnormalized Laplacian, or of the symmetric normalized Laplacian
//!    L_sym = I − D^(−1/2) W D^(−1/2) for the Ng–Jordan–Weiss variant).
//! 4. Form an n × k matrix U of eigenvectors, row-normalize for NJW.
//! 5. Run k-means on the rows of U; the labels are the cluster assignments.
//!
//! ## References
//!
//! - Ng, Jordan & Weiss (2001), "On Spectral Clustering: Analysis and an Algorithm"
//! - Shi & Malik (2000), "Normalized Cuts and Image Segmentation"
//! - von Luxburg (2007), "A Tutorial on Spectral Clustering"
//!
//! ## Tambear contract
//!
//! - **Custom implementation**: RBF kernel, Laplacian, eigendecomposition
//!   (via the existing tambear `sym_eigen` Jacobi routine), and Lloyd k-means
//!   are all written here from first principles. No SciPy, no scikit-learn.
//! - **Accumulate + gather**: the affinity matrix is a pairwise accumulate
//!   over data points (reuses the same phase-space distance pattern from
//!   complexity.rs / nonparametric.rs). The Laplacian is a pointwise
//!   transform of that matrix. Both are embarrassingly parallel.
//! - **Every parameter tunable**: affinity kernel, bandwidth (σ), graph
//!   construction strategy (full / k-NN), Laplacian variant, k-means max
//!   iterations, convergence tolerance, random seed — all explicit.
//! - **Kingdom**: A (pairwise graph build) + C (eigendecomposition iteration).
//!
//! ## Example
//!
//! ```
//! use tambear::spectral_clustering::{spectral_cluster, SpectralClusterParams,
//!     AffinityKind, LaplacianKind};
//!
//! // Two well-separated clusters in 2D
//! let data: Vec<f64> = vec![
//!     0.0, 0.0,  0.1, 0.1,  0.0, 0.2,  0.2, 0.0,
//!     5.0, 5.0,  5.1, 5.1,  5.0, 5.2,  5.2, 5.0,
//! ];
//! let params = SpectralClusterParams {
//!     k: 2,
//!     affinity: AffinityKind::Rbf { sigma: 1.0 },
//!     laplacian: LaplacianKind::SymmetricNormalized,
//!     kmeans_max_iter: 50,
//!     kmeans_tol: 1e-6,
//!     seed: 42,
//! };
//! let result = spectral_cluster(&data, 8, 2, &params);
//! // Points 0..4 are one cluster, 4..8 are another
//! let first = result.labels[0];
//! assert!(result.labels[0..4].iter().all(|&l| l == first));
//! let second = result.labels[4];
//! assert!(second != first);
//! assert!(result.labels[4..8].iter().all(|&l| l == second));
//! ```

use crate::linear_algebra::{Mat, sym_eigen};
use crate::rng::{Xoshiro256, TamRng};

// ═══════════════════════════════════════════════════════════════════════════
// Parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Affinity graph kernel — how pairwise similarity is computed from squared
/// Euclidean distance d².
#[derive(Debug, Clone, Copy)]
pub enum AffinityKind {
    /// Gaussian RBF: W[i,j] = exp(−d(i,j)² / (2σ²)).
    Rbf { sigma: f64 },
    /// k-nearest-neighbour graph: W[i,j] = 1 iff i is among the k nearest
    /// neighbours of j OR j is among the k nearest of i (symmetrized).
    KNearest { k_neighbours: usize },
    /// Epsilon-neighbourhood: W[i,j] = 1 iff d(i,j) ≤ ε.
    Epsilon { epsilon: f64 },
}

/// Choice of graph Laplacian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianKind {
    /// L = D − W. Smallest eigenvalues used for embedding.
    Unnormalized,
    /// Random-walk: L_rw = I − D⁻¹W (Shi–Malik 2000). Uses smallest eigenvalues.
    RandomWalk,
    /// Symmetric normalized: L_sym = I − D⁻¹ᐟ² W D⁻¹ᐟ² (Ng–Jordan–Weiss 2001).
    /// Eigenvectors are row-normalized before k-means.
    SymmetricNormalized,
}

#[derive(Debug, Clone)]
pub struct SpectralClusterParams {
    /// Number of clusters (= number of eigenvectors used).
    pub k: usize,
    /// How to build the affinity graph.
    pub affinity: AffinityKind,
    /// Laplacian variant.
    pub laplacian: LaplacianKind,
    /// Maximum Lloyd iterations in the k-means step.
    pub kmeans_max_iter: usize,
    /// Relative convergence tolerance for k-means (inertia change fraction).
    pub kmeans_tol: f64,
    /// Deterministic seed for k-means initialization.
    pub seed: u64,
}

impl Default for SpectralClusterParams {
    fn default() -> Self {
        Self {
            k: 2,
            affinity: AffinityKind::Rbf { sigma: 1.0 },
            laplacian: LaplacianKind::SymmetricNormalized,
            kmeans_max_iter: 100,
            kmeans_tol: 1e-6,
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpectralClusterResult {
    /// Cluster label per input point (0..k−1). Empty on degenerate input.
    pub labels: Vec<usize>,
    /// The n × k embedding matrix (row-normalized for NJW).
    pub embedding: Vec<f64>,
    /// The k smallest eigenvalues used, sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Final k-means inertia on the embedding.
    pub inertia: f64,
    /// Number of Lloyd iterations actually used.
    pub kmeans_iters: usize,
}

impl SpectralClusterResult {
    pub fn empty() -> Self {
        Self {
            labels: vec![], embedding: vec![], eigenvalues: vec![],
            inertia: f64::NAN, kmeans_iters: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Affinity construction (accumulate-shaped pairwise over n × d data)
// ═══════════════════════════════════════════════════════════════════════════

/// Squared Euclidean distance matrix for row-major `n × d` data.
/// Symmetric, zero-diagonal. O(n² d).
pub fn pairwise_sq_dist(data: &[f64], n: usize, d: usize) -> Mat {
    let mut out = Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..d {
                let dx = data[i * d + k] - data[j * d + k];
                s += dx * dx;
            }
            out.set(i, j, s);
            out.set(j, i, s);
        }
    }
    out
}

/// Build the affinity matrix W from pairwise squared distances using the
/// chosen kernel. Always returns a symmetric n × n matrix with zero diagonal.
pub fn build_affinity(sq_dist: &Mat, kind: AffinityKind) -> Mat {
    let n = sq_dist.rows;
    let mut w = Mat::zeros(n, n);
    match kind {
        AffinityKind::Rbf { sigma } => {
            let sigma = sigma.max(f64::MIN_POSITIVE);
            let denom = 2.0 * sigma * sigma;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = (-sq_dist.get(i, j) / denom).exp();
                    w.set(i, j, v);
                    w.set(j, i, v);
                }
            }
        }
        AffinityKind::Epsilon { epsilon } => {
            let e2 = epsilon * epsilon;
            for i in 0..n {
                for j in (i + 1)..n {
                    if sq_dist.get(i, j) <= e2 {
                        w.set(i, j, 1.0);
                        w.set(j, i, 1.0);
                    }
                }
            }
        }
        AffinityKind::KNearest { k_neighbours } => {
            // For each row, keep the k smallest off-diagonal distances.
            // Symmetrize with OR semantics: W[i,j] = 1 if i∈NN(j) or j∈NN(i).
            let kk = k_neighbours.min(n.saturating_sub(1));
            for i in 0..n {
                let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                idx.sort_by(|&a, &b| sq_dist.get(i, a).total_cmp(&sq_dist.get(i, b)));
                for &j in idx.iter().take(kk) {
                    w.set(i, j, 1.0);
                    w.set(j, i, 1.0);
                }
            }
        }
    }
    w
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplacian construction
// ═══════════════════════════════════════════════════════════════════════════

/// Build the chosen graph Laplacian. Returns the n × n matrix whose *smallest*
/// eigenvalues give the spectral embedding.
pub fn build_laplacian(w: &Mat, kind: LaplacianKind) -> Mat {
    let n = w.rows;
    // Degree vector
    let mut deg = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            deg[i] += w.get(i, j);
        }
    }

    match kind {
        LaplacianKind::Unnormalized => {
            // L = D − W
            let mut l = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let v = if i == j { deg[i] - w.get(i, j) } else { -w.get(i, j) };
                    l.set(i, j, v);
                }
            }
            l
        }
        LaplacianKind::RandomWalk => {
            // L_rw = I − D⁻¹ W. Note: this is not symmetric, which breaks
            // the Jacobi eigendecomposition. We instead return L_sym and
            // let the caller interpret — they share the same eigenvalues.
            // (See von Luxburg 2007 §3.2 for the equivalence.)
            Self_normalized_laplacian(w, &deg)
        }
        LaplacianKind::SymmetricNormalized => {
            Self_normalized_laplacian(w, &deg)
        }
    }
}

/// L_sym = I − D⁻¹ᐟ² W D⁻¹ᐟ²
#[allow(non_snake_case)]
fn Self_normalized_laplacian(w: &Mat, deg: &[f64]) -> Mat {
    let n = w.rows;
    let d_inv_sqrt: Vec<f64> = deg.iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let mut l = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let off = d_inv_sqrt[i] * w.get(i, j) * d_inv_sqrt[j];
            let v = if i == j { 1.0 - off } else { -off };
            l.set(i, j, v);
        }
    }
    l
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral embedding
// ═══════════════════════════════════════════════════════════════════════════

/// Given a symmetric Laplacian, extract the eigenvectors corresponding to the
/// `k` smallest eigenvalues. Returns (n × k flattened embedding, eigenvalues).
///
/// `normalize_rows` controls whether each row of the embedding is rescaled to
/// unit L² norm — required for Ng–Jordan–Weiss and beneficial for general
/// spectral clustering because it projects onto the unit sphere before k-means.
pub fn spectral_embedding(
    laplacian: &Mat,
    k: usize,
    normalize_rows: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = laplacian.rows;
    if n == 0 || k == 0 { return (vec![], vec![]); }

    // sym_eigen returns eigenvalues sorted *descending*; we want smallest k.
    let (eigvals, eigvecs) = sym_eigen(laplacian);
    if eigvals.is_empty() { return (vec![], vec![]); }

    // Take the LAST k eigenvectors (they correspond to smallest eigenvalues
    // under descending sort). Pair with their eigenvalues for clarity.
    let n_avail = eigvals.len();
    let take = k.min(n_avail);
    let mut out = vec![0.0_f64; n * take];
    let mut eigs = vec![0.0_f64; take];
    for col_idx in 0..take {
        let src_col = n_avail - 1 - col_idx;
        eigs[col_idx] = eigvals[src_col];
        for row in 0..n {
            out[row * take + col_idx] = eigvecs.get(row, src_col);
        }
    }

    if normalize_rows {
        for row in 0..n {
            let mut norm_sq = 0.0_f64;
            for col in 0..take {
                let v = out[row * take + col];
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt();
            if norm > f64::MIN_POSITIVE {
                for col in 0..take {
                    out[row * take + col] /= norm;
                }
            }
        }
    }

    (out, eigs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Lloyd's k-means on the embedding
// ═══════════════════════════════════════════════════════════════════════════

/// Minimal deterministic k-means++ initialization + Lloyd iteration used as
/// the final step of spectral clustering. Not intended as a general-purpose
/// k-means replacement (see `kmeans.rs` for the GPU-accelerated engine).
///
/// - `points`: n × d row-major f64
/// - Returns (labels, inertia, iterations_used)
pub fn lloyd_kmeans(
    points: &[f64],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> (Vec<usize>, f64, usize) {
    if n == 0 || d == 0 || k == 0 {
        return (vec![], f64::NAN, 0);
    }
    if k == 1 {
        return (vec![0; n], points_total_inertia(points, n, d), 1);
    }
    if k >= n {
        // Degenerate: each point its own cluster.
        return ((0..n).collect(), 0.0, 1);
    }

    let mut rng = Xoshiro256::new(seed);

    // k-means++ initialization: first centroid uniform, subsequent centroids
    // chosen with probability proportional to squared distance from nearest
    // existing centroid.
    let mut centroids = vec![0.0_f64; k * d];
    let first = (rng.next_u64() as usize) % n;
    for j in 0..d {
        centroids[j] = points[first * d + j];
    }
    let mut nearest_sq = vec![f64::INFINITY; n];
    for c_idx in 1..k {
        // Update nearest squared distance to any placed centroid (the new one)
        let new_c = c_idx - 1;
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..d {
                let dx = points[i * d + j] - centroids[new_c * d + j];
                s += dx * dx;
            }
            if s < nearest_sq[i] { nearest_sq[i] = s; }
        }
        // Sample proportional to nearest_sq
        let total: f64 = nearest_sq.iter().sum();
        if total <= 0.0 {
            // All points coincide with placed centroids — duplicate
            let idx = (rng.next_u64() as usize) % n;
            for j in 0..d {
                centroids[c_idx * d + j] = points[idx * d + j];
            }
            continue;
        }
        let u = (rng.next_u64() as f64 / u64::MAX as f64) * total;
        let mut cum = 0.0_f64;
        let mut chosen = 0;
        for i in 0..n {
            cum += nearest_sq[i];
            if cum >= u { chosen = i; break; }
        }
        for j in 0..d {
            centroids[c_idx * d + j] = points[chosen * d + j];
        }
    }

    // Lloyd iterations
    let mut labels = vec![0usize; n];
    let mut prev_inertia = f64::INFINITY;
    let mut iters_used = 0;
    for it in 0..max_iter {
        iters_used = it + 1;

        // Assign
        let mut inertia = 0.0_f64;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let mut s = 0.0_f64;
                for j in 0..d {
                    let dx = points[i * d + j] - centroids[c * d + j];
                    s += dx * dx;
                    if s >= best_d { break; }
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            labels[i] = best;
            inertia += best_d;
        }

        // Convergence check
        if prev_inertia.is_finite() {
            let delta = (prev_inertia - inertia).abs();
            let scale = prev_inertia.abs().max(1e-12);
            if delta / scale < tol { return (labels, inertia, iters_used); }
        }
        prev_inertia = inertia;

        // Update
        let mut new_c = vec![0.0_f64; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let lab = labels[i];
            counts[lab] += 1;
            for j in 0..d {
                new_c[lab * d + j] += points[i * d + j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: re-seed to a random point to avoid collapse.
                let idx = (rng.next_u64() as usize) % n;
                for j in 0..d {
                    new_c[c * d + j] = points[idx * d + j];
                }
            } else {
                let inv = 1.0 / counts[c] as f64;
                for j in 0..d {
                    new_c[c * d + j] *= inv;
                }
            }
        }
        centroids = new_c;
    }

    // Recompute final inertia after last update
    let mut inertia = 0.0_f64;
    for i in 0..n {
        let lab = labels[i];
        let mut s = 0.0_f64;
        for j in 0..d {
            let dx = points[i * d + j] - centroids[lab * d + j];
            s += dx * dx;
        }
        inertia += s;
    }
    (labels, inertia, iters_used)
}

fn points_total_inertia(points: &[f64], n: usize, d: usize) -> f64 {
    if n == 0 { return 0.0; }
    let mut mean = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            mean[j] += points[i * d + j];
        }
    }
    let inv = 1.0 / n as f64;
    for j in 0..d { mean[j] *= inv; }
    let mut inertia = 0.0_f64;
    for i in 0..n {
        for j in 0..d {
            let dx = points[i * d + j] - mean[j];
            inertia += dx * dx;
        }
    }
    inertia
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level spectral clustering driver
// ═══════════════════════════════════════════════════════════════════════════

/// Run the full spectral clustering pipeline on row-major `n × d` data.
///
/// Pipeline:
/// 1. Pairwise squared distances.
/// 2. Affinity matrix via the chosen kernel.
/// 3. Laplacian matrix (unnormalized / symmetric-normalized / random-walk).
/// 4. Symmetric eigendecomposition; take the k smallest-eigenvalue vectors.
/// 5. Row-normalize (for symmetric-normalized / random-walk variants).
/// 6. Lloyd k-means with k-means++ initialization on the embedding.
pub fn spectral_cluster(
    data: &[f64],
    n: usize,
    d: usize,
    params: &SpectralClusterParams,
) -> SpectralClusterResult {
    if n == 0 || d == 0 || params.k == 0 || data.len() != n * d {
        return SpectralClusterResult::empty();
    }

    let sq = pairwise_sq_dist(data, n, d);
    let w = build_affinity(&sq, params.affinity);
    let laplacian = build_laplacian(&w, params.laplacian);
    let normalize_rows = matches!(
        params.laplacian,
        LaplacianKind::SymmetricNormalized | LaplacianKind::RandomWalk
    );
    let (embedding, eigenvalues) = spectral_embedding(&laplacian, params.k, normalize_rows);
    if embedding.is_empty() {
        return SpectralClusterResult::empty();
    }
    let k_used = eigenvalues.len();
    let (labels, inertia, iters) = lloyd_kmeans(
        &embedding,
        n,
        k_used,
        params.k,
        params.kmeans_max_iter,
        params.kmeans_tol,
        params.seed,
    );
    SpectralClusterResult {
        labels,
        embedding,
        eigenvalues,
        inertia,
        kmeans_iters: iters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_well_separated_clusters() -> (Vec<f64>, usize, usize) {
        // 8 points in 2D, split into two tight clusters far apart.
        let data = vec![
            0.0, 0.0,
            0.1, 0.05,
            0.05, 0.1,
            -0.05, 0.02,
            10.0, 10.0,
            10.1, 9.9,
            9.95, 10.05,
            10.0, 10.1,
        ];
        (data, 8, 2)
    }

    #[test]
    fn pairwise_sq_dist_basic() {
        let data = vec![0.0, 0.0, 3.0, 4.0];
        let sq = pairwise_sq_dist(&data, 2, 2);
        assert_eq!(sq.rows, 2);
        assert_eq!(sq.get(0, 0), 0.0);
        assert_eq!(sq.get(1, 1), 0.0);
        assert!((sq.get(0, 1) - 25.0).abs() < 1e-12);
        assert!((sq.get(1, 0) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn rbf_affinity_self_max() {
        let sq = Mat::from_rows(&[&[0.0, 4.0], &[4.0, 0.0]]);
        let w = build_affinity(&sq, AffinityKind::Rbf { sigma: 1.0 });
        // Diagonals stay 0 (we only fill off-diagonals for i<j)
        assert_eq!(w.get(0, 0), 0.0);
        assert!((w.get(0, 1) - (-2.0_f64).exp()).abs() < 1e-12);
        // Symmetry
        assert!((w.get(0, 1) - w.get(1, 0)).abs() < 1e-14);
    }

    #[test]
    fn epsilon_affinity_cutoff() {
        let sq = Mat::from_rows(&[
            &[0.0, 1.0, 100.0],
            &[1.0, 0.0, 100.0],
            &[100.0, 100.0, 0.0],
        ]);
        let w = build_affinity(&sq, AffinityKind::Epsilon { epsilon: 2.0 });
        // 0-1 are within ε² = 4 → connected
        assert_eq!(w.get(0, 1), 1.0);
        // 0-2 and 1-2 are far → disconnected
        assert_eq!(w.get(0, 2), 0.0);
        assert_eq!(w.get(1, 2), 0.0);
    }

    #[test]
    fn knn_affinity_symmetric() {
        // Points on a line: 0, 1, 2, 10
        let sq = Mat::from_rows(&[
            &[0.0, 1.0, 4.0, 100.0],
            &[1.0, 0.0, 1.0, 81.0],
            &[4.0, 1.0, 0.0, 64.0],
            &[100.0, 81.0, 64.0, 0.0],
        ]);
        let w = build_affinity(&sq, AffinityKind::KNearest { k_neighbours: 1 });
        // Each point's single nearest neighbour:
        //   0 → 1,  1 → 0 or 2 (both at distance 1, first wins = 0),
        //   2 → 1,  3 → 2
        // Symmetrized with OR → {(0,1), (1,2), (2,3)}
        assert_eq!(w.get(0, 1), 1.0);
        assert_eq!(w.get(1, 0), 1.0);
        assert_eq!(w.get(1, 2), 1.0);
        assert_eq!(w.get(2, 3), 1.0);
        assert_eq!(w.get(0, 3), 0.0);
    }

    #[test]
    fn unnormalized_laplacian_row_sums_zero() {
        // For L = D - W, row sums should be zero (since sum(W[i,*]) = deg[i]).
        let w = Mat::from_rows(&[
            &[0.0, 1.0, 0.5],
            &[1.0, 0.0, 1.0],
            &[0.5, 1.0, 0.0],
        ]);
        let l = build_laplacian(&w, LaplacianKind::Unnormalized);
        for i in 0..3 {
            let mut row_sum = 0.0_f64;
            for j in 0..3 { row_sum += l.get(i, j); }
            assert!(row_sum.abs() < 1e-12, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn symmetric_normalized_laplacian_is_symmetric() {
        let w = Mat::from_rows(&[
            &[0.0, 1.0, 0.5],
            &[1.0, 0.0, 1.0],
            &[0.5, 1.0, 0.0],
        ]);
        let l = build_laplacian(&w, LaplacianKind::SymmetricNormalized);
        for i in 0..3 {
            for j in 0..3 {
                assert!((l.get(i, j) - l.get(j, i)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn spectral_cluster_two_blobs_rbf_njw() {
        let (data, n, d) = two_well_separated_clusters();
        let params = SpectralClusterParams {
            k: 2,
            affinity: AffinityKind::Rbf { sigma: 1.0 },
            laplacian: LaplacianKind::SymmetricNormalized,
            kmeans_max_iter: 100,
            kmeans_tol: 1e-8,
            seed: 42,
        };
        let r = spectral_cluster(&data, n, d, &params);
        assert_eq!(r.labels.len(), n);
        assert_eq!(r.eigenvalues.len(), 2);
        // First 4 points share a label, last 4 share the other label.
        let a = r.labels[0];
        let b = r.labels[4];
        assert_ne!(a, b, "cluster labels should differ across blobs");
        for i in 0..4 {
            assert_eq!(r.labels[i], a, "point {} misassigned", i);
            assert_eq!(r.labels[i + 4], b, "point {} misassigned", i + 4);
        }
    }

    #[test]
    fn spectral_cluster_two_blobs_unnormalized() {
        let (data, n, d) = two_well_separated_clusters();
        let params = SpectralClusterParams {
            k: 2,
            affinity: AffinityKind::Rbf { sigma: 1.0 },
            laplacian: LaplacianKind::Unnormalized,
            kmeans_max_iter: 100,
            kmeans_tol: 1e-8,
            seed: 7,
        };
        let r = spectral_cluster(&data, n, d, &params);
        let a = r.labels[0];
        let b = r.labels[4];
        assert_ne!(a, b);
        for i in 0..4 {
            assert_eq!(r.labels[i], a);
            assert_eq!(r.labels[i + 4], b);
        }
    }

    #[test]
    fn spectral_cluster_rejects_empty() {
        let params = SpectralClusterParams::default();
        let r = spectral_cluster(&[], 0, 2, &params);
        assert!(r.labels.is_empty());
        assert!(r.eigenvalues.is_empty());
    }

    #[test]
    fn spectral_cluster_k1_single_cluster() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let params = SpectralClusterParams {
            k: 1,
            ..SpectralClusterParams::default()
        };
        let r = spectral_cluster(&data, 3, 2, &params);
        assert_eq!(r.labels, vec![0, 0, 0]);
    }

    #[test]
    fn lloyd_kmeans_two_points() {
        let points = vec![0.0, 0.0, 10.0, 10.0];
        let (labels, inertia, _) = lloyd_kmeans(&points, 2, 2, 2, 100, 1e-8, 1);
        assert_ne!(labels[0], labels[1]);
        assert!(inertia < 1e-10);
    }

    #[test]
    fn lloyd_kmeans_constant_data() {
        let points = vec![5.0; 20];
        let (labels, inertia, _) = lloyd_kmeans(&points, 10, 2, 2, 100, 1e-8, 1);
        // All identical → inertia is 0, one valid cluster
        assert!(inertia < 1e-12);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn lloyd_kmeans_k_equals_n() {
        let points = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let (labels, inertia, _) = lloyd_kmeans(&points, 3, 2, 3, 100, 1e-8, 1);
        assert_eq!(labels.len(), 3);
        assert_eq!(inertia, 0.0);
    }
}
