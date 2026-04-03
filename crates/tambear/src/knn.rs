//! K-Nearest Neighbors via shared distance matrix.
//!
//! KNN is the canonical CONSUMER of the sharing infrastructure. Given a
//! precomputed distance matrix (from DBSCAN, silhouette scoring, or any
//! distance-producing algorithm), KNN is O(n²k) CPU work — no GPU needed.
//!
//! Without sharing, KNN would recompute the O(n²d) distance matrix on GPU.
//! With sharing, the distance matrix comes from the session for free.
//!
//! ## Cross-algorithm sharing example
//!
//! ```no_run
//! use tambear::intermediates::TamSession;
//! use tambear::clustering::ClusteringEngine;
//! use tambear::knn::knn_session;
//!
//! let mut session = TamSession::new();
//! let mut engine = ClusteringEngine::new().unwrap();
//! let data = vec![0.0, 0.0,  1.0, 0.0,  5.0, 5.0,  5.5, 5.0f64];
//!
//! // DBSCAN: computes distance matrix, registers in session
//! let clusters = engine.dbscan_session(&mut session, &data, 4, 2, 1.0, 2).unwrap();
//!
//! // KNN: finds distance matrix in session — zero GPU cost
//! let neighbors = knn_session(&mut session, &data, 4, 2, 2).unwrap();
//! ```
//!
//! This is the sharing vision: two algorithms, one GPU computation.

use std::sync::Arc;

use tam_gpu::TamGpu;
use winrapids_tiled::{TiledEngine, DistanceOp, TiledOp};

use crate::intermediates::{DataId, DistanceMatrix, IntermediateTag, Metric, TamSession};

/// Per-point k-nearest neighbor result.
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// For each point i, `neighbors[i]` is a Vec of (neighbor_index, distance)
    /// sorted by distance (nearest first). Length k (or less if fewer points exist).
    pub neighbors: Vec<Vec<(usize, f64)>>,

    /// Number of points.
    pub n: usize,

    /// Number of neighbors per point.
    pub k: usize,
}

impl KnnResult {
    /// Get the k nearest neighbor indices for point i.
    pub fn neighbor_indices(&self, i: usize) -> Vec<usize> {
        self.neighbors[i].iter().map(|&(idx, _)| idx).collect()
    }

    /// Get the distance to the k-th nearest neighbor for point i.
    /// Useful for Local Outlier Factor (LOF) and DBSCAN parameter tuning.
    pub fn kth_distance(&self, i: usize) -> f64 {
        self.neighbors[i].last().map(|&(_, d)| d).unwrap_or(f64::INFINITY)
    }

    /// Build a k-nearest neighbor graph as an adjacency list.
    /// Returns `(row_indices, col_indices)` — edges from row to col.
    pub fn to_graph(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(self.n * self.k);
        let mut cols = Vec::with_capacity(self.n * self.k);
        for i in 0..self.n {
            for &(j, _) in &self.neighbors[i] {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }
}

// ---------------------------------------------------------------------------
// Core KNN from precomputed distance matrix
// ---------------------------------------------------------------------------

/// Compute k-nearest neighbors from a precomputed distance matrix.
///
/// For each point i, scan row i of the distance matrix and keep the k
/// smallest distances (excluding self-distance at diagonal).
///
/// Complexity: O(n² * k) — dominated by the partial sort per row.
/// For n=5000, k=10: ~250M comparisons, <100ms on CPU.
pub fn knn_from_distance(dist: &DistanceMatrix, k: usize) -> KnnResult {
    let n = dist.n;
    let k = k.min(n - 1); // can't have more neighbors than n-1

    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        let row = dist.row(i);

        // Partial sort: maintain a max-heap of size k (implemented as sorted vec
        // for simplicity — k is typically small, 5-50)
        let mut best: Vec<(usize, f64)> = Vec::with_capacity(k + 1);

        for j in 0..n {
            if i == j { continue; } // skip self
            let d = row[j];

            if best.len() < k {
                best.push((j, d));
                if best.len() == k {
                    best.sort_by(|a, b| a.1.total_cmp(&b.1));
                }
            } else if d < best[k - 1].1 {
                best[k - 1] = (j, d);
                // Re-sort (insertion sort would be better for nearly-sorted, but
                // k is small enough that this doesn't matter)
                best.sort_by(|a, b| a.1.total_cmp(&b.1));
            }
        }

        // Ensure sorted even if we never filled to k
        if best.len() < k {
            best.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        neighbors.push(best);
    }

    KnnResult { neighbors, n, k }
}

// ---------------------------------------------------------------------------
// Session-aware KNN
// ---------------------------------------------------------------------------

/// Session-aware KNN: check session for cached distance matrix, compute if missing.
///
/// This is the cross-algorithm sharing entry point. If DBSCAN (or any other
/// distance-producing algorithm) already ran on this data, the distance matrix
/// is in the session and KNN runs at zero GPU cost.
pub fn knn_session(
    session: &mut TamSession,
    data: &[f64],
    n: usize,
    d: usize,
    k: usize,
) -> Result<KnnResult, Box<dyn std::error::Error>> {
    knn_session_with_op(session, data, n, d, k, &DistanceOp)
}

/// Session-aware KNN with pluggable distance metric.
pub fn knn_session_with_op(
    session: &mut TamSession,
    data: &[f64],
    n: usize,
    d: usize,
    k: usize,
    distance_op: &dyn TiledOp,
) -> Result<KnnResult, Box<dyn std::error::Error>> {
    assert_eq!(data.len(), n * d);
    assert!(n >= 2);
    assert!(k >= 1);

    let data_id = DataId::from_f64(data);
    let manifold_name = distance_op.params_key();
    let tag = if manifold_name.is_empty() {
        IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id }
    } else {
        IntermediateTag::ManifoldDistanceMatrix { manifold_name, data_id }
    };

    // Check session for cached distance matrix
    let dist: Arc<DistanceMatrix> = if let Some(cached) = session.get(&tag) {
        eprintln!("[knn] session HIT: reusing distance matrix ({}x{})", n, n);
        cached
    } else {
        eprintln!("[knn] session MISS: computing distance matrix ({}x{}, {}d)", n, n, d);
        let tiled = TiledEngine::new(tam_gpu::detect());

        let data_t: Vec<f64> = (0..d)
            .flat_map(|k| (0..n).map(move |i| data[i * d + k]))
            .collect();
        let dist_data = tiled.run(distance_op, data, &data_t, n, n, d)?;
        let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, dist_data));
        session.register(tag, Arc::clone(&dm));
        dm
    };

    Ok(knn_from_distance(&dist, k))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::ClusteringEngine;

    #[test]
    fn knn_basic() {
        // 4 points in 1D (embedded in 2D): 0, 1, 5, 6
        let _data = vec![
            0.0, 0.0,
            1.0, 0.0,
            5.0, 0.0,
            6.0, 0.0,
        ];
        let dist = DistanceMatrix::from_vec(Metric::L2Sq, 4, vec![
            0.0, 1.0, 25.0, 36.0,
            1.0, 0.0, 16.0, 25.0,
            25.0, 16.0, 0.0, 1.0,
            36.0, 25.0, 1.0, 0.0,
        ]);
        let result = knn_from_distance(&dist, 2);

        assert_eq!(result.n, 4);
        assert_eq!(result.k, 2);

        // Point 0's nearest: point 1 (d=1), then point 2 (d=25)
        assert_eq!(result.neighbor_indices(0), vec![1, 2]);
        // Point 3's nearest: point 2 (d=1), then point 1 (d=25)
        assert_eq!(result.neighbor_indices(3), vec![2, 1]);
    }

    #[test]
    fn knn_kth_distance() {
        let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
            0.0, 4.0, 9.0,
            4.0, 0.0, 1.0,
            9.0, 1.0, 0.0,
        ]);
        let result = knn_from_distance(&dist, 1);
        // Point 0's nearest is point 1 at distance 4
        assert_eq!(result.kth_distance(0), 4.0);
        // Point 1's nearest is point 2 at distance 1
        assert_eq!(result.kth_distance(1), 1.0);
    }

    #[test]
    fn knn_graph() {
        let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
            0.0, 1.0, 4.0,
            1.0, 0.0, 1.0,
            4.0, 1.0, 0.0,
        ]);
        let result = knn_from_distance(&dist, 1);
        let (rows, cols) = result.to_graph();
        // 3 points, k=1 → 3 edges
        assert_eq!(rows.len(), 3);
        assert_eq!(cols.len(), 3);
        // 0→1, 1→0 or 1→2 (tie at d=1, first encountered wins), 2→1
        assert_eq!(rows[0], 0);
        assert_eq!(cols[0], 1);
    }

    /// The cross-algorithm sharing test: DBSCAN produces a distance matrix,
    /// KNN consumes it from the session. One GPU computation, two algorithms.
    #[test]
    fn cross_algorithm_sharing_dbscan_then_knn() {
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 4.9,
            4.9, 5.1,
        ];
        let n = 6;
        let d = 2;
        let mut session = TamSession::new();
        let mut engine = ClusteringEngine::new().unwrap();

        // DBSCAN: computes distance matrix, registers in session
        let clusters = engine.dbscan_session(&mut session, &data, n, d, 0.5, 2).unwrap();
        assert_eq!(clusters.n_clusters, 2);
        assert_eq!(session.len(), 1, "distance matrix registered");

        // KNN: finds distance matrix in session — zero GPU cost
        let neighbors = knn_session(&mut session, &data, n, d, 2).unwrap();
        assert_eq!(session.len(), 1, "no new intermediate — reused from DBSCAN");

        // Verify KNN results make sense
        // Point 0 (0,0): nearest should be point 1 (0.1,0.1) and point 2 (0.2,0)
        let n0 = neighbors.neighbor_indices(0);
        assert!(n0.contains(&1), "point 0's neighbors should include point 1");
        assert!(n0.contains(&2), "point 0's neighbors should include point 2");

        // Point 3 (5,5): nearest should be point 4 (5.1,4.9) and point 5 (4.9,5.1)
        let n3 = neighbors.neighbor_indices(3);
        assert!(n3.contains(&4), "point 3's neighbors should include point 4");
        assert!(n3.contains(&5), "point 3's neighbors should include point 5");

        println!("cross-algorithm sharing: DBSCAN + KNN, 1 GPU computation");
    }

    /// Euclidean and manifold KNN on the same data use separate session cache entries.
    /// Before the fix, both keyed to `DistanceMatrix { metric: L2Sq }` — a collision.
    #[test]
    fn knn_session_manifold_cache_key_no_collision() {
        use crate::manifold::{ManifoldDistanceOp, Manifold};

        let data = vec![0.1_f64, 0.0,  0.0, 0.1,  0.5, 0.5];
        let n = 3;
        let d = 2;
        let mut session = TamSession::new();

        // Run Euclidean KNN — registers under DistanceMatrix { L2Sq }
        knn_session(&mut session, &data, n, d, 1).unwrap();
        assert_eq!(session.len(), 1);

        // Run Poincaré KNN on the same data — must use a different cache key
        let poincare_op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        knn_session_with_op(&mut session, &data, n, d, 1, &poincare_op).unwrap();
        assert_eq!(session.len(), 2, "Poincaré distance is a separate cache entry from Euclidean");

        // Run Poincaré KNN again — must hit the cache, not add a third entry
        knn_session_with_op(&mut session, &data, n, d, 1, &poincare_op).unwrap();
        assert_eq!(session.len(), 2, "second Poincaré call is a cache hit");
    }

    /// KNN runs standalone (no prior algorithm in session).
    #[test]
    fn knn_session_standalone() {
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ];
        let mut session = TamSession::new();

        let result = knn_session(&mut session, &data, 3, 2, 1).unwrap();
        assert_eq!(session.len(), 1, "distance matrix registered for future consumers");

        // Point 0: nearest is either point 1 or point 2 (both at distance 1.0)
        let d = result.kth_distance(0);
        assert!((d - 1.0).abs() < 1e-10, "nearest distance should be 1.0, got {}", d);
    }
}
