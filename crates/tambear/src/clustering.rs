//! Density-based clustering — DBSCAN via GPU distance matrix.
//!
//! ## Algorithm
//!
//! Four primitives chained:
//!
//! 1. **Distance matrix** (GPU, via [`TiledEngine`]):
//!    `D[i,j] = distance(data[i], data[j])` for all pairs.
//!    The distance function is pluggable: pass any [`TiledOp`] as `distance_op`.
//!    Default: [`DistanceOp`] gives squared L2 (‖a − b‖²).
//!
//! 2. **Density estimation** (CPU):
//!    `density[i] = |{j ≠ i : D[i,j] ≤ epsilon_threshold}|`.
//!    Count neighbors within the distance threshold.
//!
//! 3. **Core identification** (CPU):
//!    `is_core[i] = density[i] >= min_samples`.
//!
//! 4. **Connected-component labeling** (CPU, union-find):
//!    Core points within epsilon of each other → same cluster.
//!    Border points (non-core with a core neighbor) → nearest core's cluster.
//!    Isolated non-core points → noise (label = -1).
//!
//! ## Complexity
//!
//! - GPU distance matrix: O(n² × d) — dominates, runs on GPU (tiled 16×16 blocks)
//! - CPU density + union-find + assignment: O(n²) — cheap after matrix is computed
//!
//! For n ≤ 5000, the full n×n distance matrix fits in GPU memory (200 MB for f64).
//! For larger n: approximate NN (future work).
//!
//! ## Example
//!
//! ```no_run
//! use tambear::clustering::{ClusteringEngine, ClusterResult};
//! use winrapids_tiled::DistanceOp;
//!
//! // 4 points in 2D: two tight clusters
//! let data = vec![
//!     0.0, 0.0,  // cluster A
//!     0.1, 0.1,  // cluster A
//!     5.0, 5.0,  // cluster B
//!     5.1, 4.9,  // cluster B
//! ];
//!
//! let mut engine = ClusteringEngine::new().unwrap();
//! let result = engine.discover_clusters(
//!     &data, 4, 2,
//!     0.5,         // epsilon threshold (for L2²: radius=~0.7)
//!     2,           // min_samples
//!     &DistanceOp, // distance metric
//! ).unwrap();
//!
//! assert_eq!(result.n_clusters, 2);
//! assert_eq!(result.n_noise, 0);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tam_gpu::TamGpu;
use winrapids_tiled::{TiledEngine, TiledOp, DistanceOp};

use crate::intermediates::{DataId, DistanceMatrix, IntermediateTag, Metric, TamSession};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Clustering result from [`ClusteringEngine::discover_clusters`].
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Per-point cluster assignment. `-1` = noise (no cluster).
    /// Values `0..n_clusters-1` are cluster indices.
    pub labels: Vec<i32>,

    /// Number of distinct clusters found (excluding noise).
    pub n_clusters: usize,

    /// Number of core points (density ≥ min_samples).
    pub n_core: usize,

    /// Number of noise points (no core neighbor within epsilon).
    pub n_noise: usize,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// GPU-accelerated density-based clustering engine.
///
/// Uses [`TiledEngine`] (any [`TamGpu`] backend) for the distance computation.
/// The rest of the algorithm runs on CPU — it's fast given the precomputed matrix.
pub struct ClusteringEngine {
    tiled: TiledEngine,
}

impl ClusteringEngine {
    /// Initialise on the best available GPU (CUDA if present, CPU fallback).
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { tiled: TiledEngine::new(tam_gpu::detect()) })
    }

    /// Initialise with a specific [`TamGpu`] backend.
    pub fn with_gpu(gpu: Arc<dyn TamGpu>) -> Self {
        Self { tiled: TiledEngine::new(gpu) }
    }

    /// Discover clusters in `data` (n × d, row-major, f64).
    ///
    /// # Parameters
    ///
    /// - `data`: input matrix, n rows × d columns, row-major
    /// - `n`, `d`: shape of `data`
    /// - `epsilon_threshold`: distance threshold for neighbourhood membership.
    ///   For [`DistanceOp`] this is squared L2 — pass `radius * radius`.
    ///   For custom metrics, pass the appropriate threshold for the op's output scale.
    /// - `min_samples`: minimum neighbours (within epsilon) for a point to be a core
    /// - `distance_op`: pluggable distance metric. Any [`TiledOp`] whose output is
    ///   a non-negative dissimilarity. Pass `&DistanceOp` for squared L2.
    ///
    /// # Returns
    ///
    /// [`ClusterResult`] with per-point labels: `-1` = noise, `0..k-1` = clusters.
    pub fn discover_clusters(
        &mut self,
        data: &[f64],
        n: usize,
        d: usize,
        epsilon_threshold: f64,
        min_samples: usize,
        distance_op: &dyn TiledOp,
    ) -> Result<ClusterResult, Box<dyn std::error::Error>> {
        assert_eq!(data.len(), n * d, "data must be n × d");
        assert!(n >= 2, "need at least 2 points");
        assert!(min_samples >= 1, "min_samples must be ≥ 1");

        // ── Step 1: GPU pairwise distance matrix ─────────────────────────────
        // TiledEngine computes: C[i,j] = op(A[i,:], B[:,j]) accumulating over K.
        // For self-distance: A = data (n×d) and B = data^T (d×n).
        //   A[i,k] = data[i*d + k]  — row i, dimension k
        //   B[k,j] = data_T[k*n+j] = data[j*d+k]  — dimension k of point j
        // DistanceOp gives: C[i,j] = sum_k (data[i,k] - data[j,k])² = L2²(i,j)
        let data_t: Vec<f64> = (0..d)
            .flat_map(|k| (0..n).map(move |i| data[i * d + k]))
            .collect();
        let dist = self.tiled.run(distance_op, data, &data_t, n, n, d)?;
        // dist has length n * n; CPU steps are shared with discover_clusters_from_distance
        Ok(clustering_from_distance(&dist, n, epsilon_threshold, min_samples))
    }

    /// Like `discover_clusters` but also returns the computed distance matrix.
    ///
    /// The distance matrix is an expensive intermediate — O(n²d) GPU work.
    /// By returning it as a byproduct, callers can pass it to subsequent algorithms
    /// (KNN, outlier detection, DBSCAN with different parameters) at zero GPU cost.
    ///
    /// This is the **producer** side of the sharing infrastructure. See
    /// [`discover_clusters_from_distance`] for the **consumer** side.
    pub fn discover_clusters_with_distance(
        &mut self,
        data: &[f64],
        n: usize,
        d: usize,
        epsilon_threshold: f64,
        min_samples: usize,
        distance_op: &dyn TiledOp,
    ) -> Result<(ClusterResult, Arc<DistanceMatrix>), Box<dyn std::error::Error>> {
        assert_eq!(data.len(), n * d, "data must be n × d");
        assert!(n >= 2, "need at least 2 points");
        assert!(min_samples >= 1, "min_samples must be ≥ 1");

        let data_t: Vec<f64> = (0..d)
            .flat_map(|k| (0..n).map(move |i| data[i * d + k]))
            .collect();
        let dist_data = self.tiled.run(distance_op, data, &data_t, n, n, d)?;

        // Wrap before consuming — Arc::clone is O(1) for passing to caller
        let dist = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, dist_data));

        let result = clustering_from_distance(&dist.data, n, epsilon_threshold, min_samples);
        Ok((result, dist))
    }

    /// Run clustering on a precomputed distance matrix.
    ///
    /// This is the **consumer** side of the sharing infrastructure. If another
    /// algorithm (e.g., a previous `discover_clusters_with_distance` call) already
    /// computed the n×n distance matrix, pass it here to avoid O(n²d) GPU recomputation.
    ///
    /// # Panics
    ///
    /// Panics if `distance.metric != Metric::L2Sq` — the density / epsilon semantics
    /// depend on the metric. Use the correct threshold for the metric in use.
    pub fn discover_clusters_from_distance(
        &mut self,
        distance: Arc<DistanceMatrix>,
        epsilon_threshold: f64,
        min_samples: usize,
    ) -> Result<ClusterResult, Box<dyn std::error::Error>> {
        assert!(
            distance.is_compatible_with(Metric::L2Sq),
            "discover_clusters_from_distance requires Metric::L2Sq; got {:?}",
            distance.metric,
        );
        assert!(min_samples >= 1, "min_samples must be ≥ 1");
        let n = distance.n;
        Ok(clustering_from_distance(&distance.data, n, epsilon_threshold, min_samples))
    }

    /// Run DBSCAN on a pre-combined (raw) distance slice, without metric assertion.
    ///
    /// Used when the distance matrix is a weighted mixture of multiple geometry matrices
    /// (via `ManifoldMixture::combine`). Since the metric is not a single `Metric` variant,
    /// the metric assertion is skipped.
    ///
    /// `epsilon_threshold` is used verbatim — for L2Sq mixtures this should be `epsilon²`.
    pub fn discover_clusters_from_combined(
        dist: &[f64],
        n: usize,
        epsilon_threshold: f64,
        min_samples: usize,
    ) -> ClusterResult {
        assert!(min_samples >= 1, "min_samples must be ≥ 1");
        clustering_from_distance(dist, n, epsilon_threshold, min_samples)
    }

    /// Convenience wrapper using squared L2 distance (default for most cases).
    ///
    /// Pass `epsilon_radius` — the radius of the neighbourhood ball.
    /// Internally converted to `epsilon_radius²` (since [`DistanceOp`] returns L2²).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tambear::clustering::ClusteringEngine;
    ///
    /// let mut engine = ClusteringEngine::new().unwrap();
    /// let data = vec![0.0, 0.0,  0.1, 0.1,  5.0, 5.0,  5.1, 4.9f64];
    /// let result = engine.dbscan(&data, 4, 2, 0.5, 2).unwrap();
    /// assert_eq!(result.n_clusters, 2);
    /// ```
    pub fn dbscan(
        &mut self,
        data: &[f64],
        n: usize,
        d: usize,
        epsilon_radius: f64,
        min_samples: usize,
    ) -> Result<ClusterResult, Box<dyn std::error::Error>> {
        self.discover_clusters(
            data, n, d,
            epsilon_radius * epsilon_radius,  // DistanceOp returns L2²
            min_samples,
            &DistanceOp,
        )
    }

    // -----------------------------------------------------------------------
    // Session-aware methods — automatic sharing via TamSession
    // -----------------------------------------------------------------------

    /// Session-aware clustering: check session for cached distance matrix,
    /// compute if missing, register for downstream reuse.
    ///
    /// This is the **compiler pattern** — the session matches producers to
    /// consumers automatically. Two algorithms on the same data with the same
    /// metric share the distance matrix without explicit wiring.
    ///
    /// # How it works
    ///
    /// 1. Compute `DataId` from the data (blake3 content hash)
    /// 2. Build `IntermediateTag::DistanceMatrix { metric, data_id }`
    /// 3. If session has it → reuse (zero GPU cost)
    /// 4. If not → compute on GPU, register in session, return result
    pub fn discover_clusters_session(
        &mut self,
        session: &mut TamSession,
        data: &[f64],
        n: usize,
        d: usize,
        epsilon_threshold: f64,
        min_samples: usize,
        distance_op: &dyn TiledOp,
    ) -> Result<ClusterResult, Box<dyn std::error::Error>> {
        assert_eq!(data.len(), n * d, "data must be n × d");
        assert!(n >= 2, "need at least 2 points");
        assert!(min_samples >= 1, "min_samples must be ≥ 1");

        let data_id = DataId::from_f64(data);
        let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id };

        // Check session for cached distance matrix
        if let Some(dist) = session.get::<DistanceMatrix>(&tag) {
            return self.discover_clusters_from_distance(dist, epsilon_threshold, min_samples);
        }

        // Compute + register
        let (result, dist) = self.discover_clusters_with_distance(
            data, n, d, epsilon_threshold, min_samples, distance_op,
        )?;
        session.register(tag, dist);
        Ok(result)
    }

    /// Session-aware convenience wrapper using squared L2 distance.
    ///
    /// Like [`dbscan`] but with automatic intermediate sharing via session.
    pub fn dbscan_session(
        &mut self,
        session: &mut TamSession,
        data: &[f64],
        n: usize,
        d: usize,
        epsilon_radius: f64,
        min_samples: usize,
    ) -> Result<ClusterResult, Box<dyn std::error::Error>> {
        self.discover_clusters_session(
            session, data, n, d,
            epsilon_radius * epsilon_radius,
            min_samples,
            &DistanceOp,
        )
    }
}

// ---------------------------------------------------------------------------
// CPU clustering on a precomputed distance slice
// ---------------------------------------------------------------------------

/// Run DBSCAN density estimation + union-find on a flat n×n distance matrix.
///
/// This is the pure CPU part of DBSCAN — steps 2-5 from `discover_clusters`.
/// Extracted so both `discover_clusters` (which computes distance) and
/// `discover_clusters_from_distance` (which reuses precomputed distance) can
/// share identical logic without duplication.
fn clustering_from_distance(
    dist: &[f64],
    n: usize,
    epsilon_threshold: f64,
    min_samples: usize,
) -> ClusterResult {
    // ── Step 2: Density (CPU, O(n²)) ─────────────────────────────────────
    let mut density = vec![0usize; n];
    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        for j in 0..n {
            if row[j] <= epsilon_threshold {
                density[i] += 1;
            }
        }
    }

    // ── Step 3: Core identification ───────────────────────────────────────
    let is_core: Vec<bool> = density.iter().map(|&c| c >= min_samples).collect();
    let n_core = is_core.iter().filter(|&&c| c).count();

    // ── Step 4a: Union-find over core-core connections ────────────────────
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if !is_core[i] { continue; }
        let row_i = &dist[i * n..(i + 1) * n];
        for j in (i + 1)..n {
            if is_core[j] && row_i[j] <= epsilon_threshold {
                let ri = uf_find(&mut parent, i);
                let rj = uf_find(&mut parent, j);
                if ri != rj { parent[ri] = rj; }
            }
        }
    }

    // ── Step 4b: Sequential cluster IDs ──────────────────────────────────
    let mut root_to_id: HashMap<usize, i32> = HashMap::new();
    let mut next_id: i32 = 0;
    let mut labels = vec![-1i32; n];
    for i in 0..n {
        if !is_core[i] { continue; }
        let root = uf_find(&mut parent, i);
        let id = match root_to_id.get(&root).copied() {
            Some(id) => id,
            None => {
                let id = next_id;
                root_to_id.insert(root, id);
                next_id += 1;
                id
            }
        };
        labels[i] = id;
    }

    // ── Step 5: Border-point assignment ──────────────────────────────────
    for i in 0..n {
        if is_core[i] { continue; }
        let row_i = &dist[i * n..(i + 1) * n];
        for j in 0..n {
            if is_core[j] && row_i[j] <= epsilon_threshold {
                labels[i] = labels[j];
                break;
            }
        }
    }

    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    ClusterResult { labels, n_clusters: next_id as usize, n_core, n_noise }
}

// ---------------------------------------------------------------------------
// Union-find helpers
// ---------------------------------------------------------------------------

/// Iterative path-halving find — no recursion, O(α(n)) amortized.
fn uf_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> ClusteringEngine {
        ClusteringEngine::new().expect("ClusteringEngine init failed")
    }

    #[test]
    fn two_tight_clusters() {
        // Two well-separated clusters in 2D, no noise
        // Cluster A: (0,0), (0.1,0.1), (0.2,0)
        // Cluster B: (5,5), (5.1,4.9), (4.9,5.1)
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 4.9,
            4.9, 5.1,
        ];
        let mut e = engine();
        // epsilon = 0.5 → epsilon_sq = 0.25; all within-cluster distances < 0.25
        let r = e.dbscan(&data, 6, 2, 0.5, 2).unwrap();
        assert_eq!(r.n_clusters, 2, "should find 2 clusters, got {}", r.n_clusters);
        assert_eq!(r.n_noise, 0, "no noise, got {} noise points", r.n_noise);
        // All 3 A-points have the same label
        assert_eq!(r.labels[0], r.labels[1]);
        assert_eq!(r.labels[1], r.labels[2]);
        // All 3 B-points have the same label
        assert_eq!(r.labels[3], r.labels[4]);
        assert_eq!(r.labels[4], r.labels[5]);
        // A and B have different labels
        assert_ne!(r.labels[0], r.labels[3]);
        println!("two_tight_clusters: labels={:?}", r.labels);
    }

    #[test]
    fn isolated_point_is_noise() {
        // One tight pair and one isolated outlier
        let data = vec![
            0.0, 0.0,
            0.1, 0.0,
            100.0, 100.0,  // isolated
        ];
        let mut e = engine();
        let r = e.dbscan(&data, 3, 2, 0.5, 2).unwrap();
        assert_eq!(r.n_clusters, 1, "should find 1 cluster");
        assert_eq!(r.n_noise, 1, "isolated point should be noise");
        assert_eq!(r.labels[2], -1, "point 2 should be noise");
        println!("isolated_point: labels={:?}", r.labels);
    }

    #[test]
    fn min_samples_determines_cores() {
        // 3 close points, min_samples=3 → all three are cores; min_samples=4 → all noise
        let data = vec![0.0, 0.0,  0.1, 0.0,  0.0, 0.1f64];
        let mut e = engine();

        let r3 = e.dbscan(&data, 3, 2, 0.5, 2).unwrap();
        assert_eq!(r3.n_clusters, 1);
        assert_eq!(r3.n_core, 3, "all 3 should be cores with min_samples=2");

        let r4 = e.dbscan(&data, 3, 2, 0.5, 4).unwrap();
        assert_eq!(r4.n_clusters, 0, "no cluster possible with min_samples=4");
        assert_eq!(r4.n_noise, 3, "all noise");
        println!("min_samples test: r3.n_core={}, r4.n_noise={}", r3.n_core, r4.n_noise);
    }

    #[test]
    fn custom_distance_op_plugs_in() {
        // Same geometry as two_tight_clusters, but pass DistanceOp explicitly
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.0f64,
            5.0, 5.0,  5.1, 4.9,  4.9, 5.1,
        ];
        let mut e = engine();
        // discover_clusters takes distance_op as parameter
        let r = e.discover_clusters(&data, 6, 2, 0.25, 2, &DistanceOp).unwrap();
        assert_eq!(r.n_clusters, 2);
        println!("custom_distance_op: n_clusters={}", r.n_clusters);
    }

    #[test]
    fn single_large_cluster() {
        // All points within epsilon of each other → 1 cluster
        let data = vec![0.0, 0.0,  0.1, 0.0,  0.2, 0.0,  0.3, 0.0f64];
        let mut e = engine();
        let r = e.dbscan(&data, 4, 2, 1.0, 2).unwrap();
        assert_eq!(r.n_clusters, 1);
        assert!(r.labels.iter().all(|&l| l == 0), "all in cluster 0: {:?}", r.labels);
        println!("single_cluster: labels={:?}", r.labels);
    }

    /// Distance matrix sharing: compute once, run DBSCAN twice with different epsilon.
    ///
    /// This demonstrates the producer → consumer intermediate sharing pattern.
    /// The n×n GPU distance computation runs ONCE; both DBSCAN variants use the
    /// cached matrix at zero GPU cost.
    #[test]
    fn shared_distance_matrix_between_calls() {
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 4.9,
            4.9, 5.1,
        ];
        let mut e = engine();

        // First call: tight epsilon → 2 clusters, and we get the distance matrix back
        let (r1, dist) = e.discover_clusters_with_distance(
            &data, 6, 2, 0.25, 2, &DistanceOp,
        ).unwrap();
        assert_eq!(r1.n_clusters, 2, "tight epsilon: 2 clusters");
        assert_eq!(dist.n, 6, "distance matrix is 6x6");

        // Second call: looser epsilon (> cross-cluster L2Sq ≈ 50), NO new GPU computation
        let r2 = e.discover_clusters_from_distance(
            std::sync::Arc::clone(&dist), 60.0, 2,
        ).unwrap();
        assert_eq!(r2.n_clusters, 1, "loose epsilon: all points in 1 cluster");

        // Verify: dist is L2Sq-compatible (no panic)
        assert!(dist.is_compatible_with(crate::intermediates::Metric::L2Sq));

        // Verify entries: D[0,0] = 0, D[0,3] should be large (~50)
        assert!((dist.entry(0, 0)).abs() < 1e-10, "diagonal is 0");
        assert!(dist.entry(0, 3) > 40.0, "cross-cluster distance large: {}", dist.entry(0, 3));

        println!("sharing test: r1.n_clusters={}, r2.n_clusters={}, dist.n={}",
                 r1.n_clusters, r2.n_clusters, dist.n);
    }

    /// Session-aware sharing: two DBSCAN runs, same data, different epsilon.
    ///
    /// The session automatically caches the distance matrix from the first run.
    /// The second run finds it in the session and skips GPU computation entirely.
    /// This is the compiler pattern: produce → register → match → reuse.
    #[test]
    fn session_automatic_sharing() {
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 4.9,
            4.9, 5.1,
        ];
        let mut e = engine();
        let mut session = TamSession::new();

        // First call: tight epsilon → 2 clusters. Session is empty, so GPU computes distance.
        assert_eq!(session.len(), 0, "session starts empty");
        let r1 = e.dbscan_session(&mut session, &data, 6, 2, 0.5, 2).unwrap();
        assert_eq!(r1.n_clusters, 2, "tight epsilon: 2 clusters");
        assert_eq!(session.len(), 1, "session now holds distance matrix");

        // Second call: loose epsilon → 1 cluster. Session HIT — no GPU work.
        let r2 = e.dbscan_session(&mut session, &data, 6, 2, 10.0, 2).unwrap();
        assert_eq!(r2.n_clusters, 1, "loose epsilon: 1 cluster");
        assert_eq!(session.len(), 1, "no new intermediates registered");

        // Third call: same tight epsilon again — still a session hit.
        let r3 = e.dbscan_session(&mut session, &data, 6, 2, 0.5, 2).unwrap();
        assert_eq!(r3.labels, r1.labels, "identical parameters → identical result");

        // Different data → session MISS → new distance matrix registered.
        let data2 = vec![
            1.0, 1.0,
            1.1, 1.1,
            1.2, 1.0,
            6.0, 6.0,
            6.1, 5.9,
            5.9, 6.1,
        ];
        let r4 = e.dbscan_session(&mut session, &data2, 6, 2, 0.5, 2).unwrap();
        assert_eq!(r4.n_clusters, 2, "new data: 2 clusters");
        assert_eq!(session.len(), 2, "session now holds 2 distance matrices");

        println!("session_automatic_sharing: 4 DBSCAN runs, 2 GPU computations");
    }

    /// Session clear frees cached intermediates.
    #[test]
    fn session_clear_forces_recompute() {
        let data = vec![0.0, 0.0,  0.1, 0.0,  0.0, 0.1f64];
        let mut e = engine();
        let mut session = TamSession::new();

        e.dbscan_session(&mut session, &data, 3, 2, 0.5, 2).unwrap();
        assert_eq!(session.len(), 1);

        session.clear();
        assert_eq!(session.len(), 0);

        // After clear, same data triggers fresh GPU computation + re-registration
        e.dbscan_session(&mut session, &data, 3, 2, 0.5, 2).unwrap();
        assert_eq!(session.len(), 1, "re-registered after clear");
    }

    // ── Manifold distance integration ────────────────────────────────────────

    /// DBSCAN with Poincaré ball distance.
    ///
    /// In the Poincaré ball, distances are hyperbolic: points near the unit-ball
    /// boundary are exponentially farther from each other than their Euclidean
    /// coordinates suggest.
    ///
    /// Setup:
    /// - p0=(0.1, 0.0), p1=(0.0, 0.1), p2=(0.0, 0.0) — near the origin
    /// - p3=(0.95, 0.0) — near the boundary
    ///
    /// Poincaré distances (κ=1):
    /// - d_H(p0,p1) ≈ 0.57, d_H(p0,p2) ≈ 0.40, d_H(p1,p2) ≈ 0.40 → all < 1.0
    /// - d_H(p3, any of p0-p2) ≈ 6–12 → all >> 1.0
    ///
    /// With epsilon=1.0, min_samples=2: {p0,p1,p2} form one cluster; p3 is noise.
    #[test]
    fn poincare_dbscan_boundary_isolation() {
        use crate::manifold::{ManifoldDistanceOp, Manifold};

        // 4 points: 3 near origin, 1 near boundary
        let data = vec![
            0.1, 0.0,  // p0
            0.0, 0.1,  // p1
            0.0, 0.0,  // p2 (origin)
            0.95, 0.0, // p3 (near ball boundary, ||p3||² ≈ 0.9025)
        ];

        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        let mut e = engine();

        // epsilon=1.0: p0–p1–p2 are within 1.0 of each other; p3 is not
        let r = e.discover_clusters(&data, 4, 2, 1.0, 2, &op).unwrap();

        assert_eq!(r.n_clusters, 1, "one hyperbolic cluster, got {}", r.n_clusters);
        assert_eq!(r.n_noise, 1, "boundary point is noise, got {} noise", r.n_noise);
        // p0, p1, p2 all in the same cluster
        assert_eq!(r.labels[0], r.labels[1], "p0 and p1 same cluster");
        assert_eq!(r.labels[1], r.labels[2], "p1 and p2 same cluster");
        assert_eq!(r.labels[0], 0, "cluster id is 0");
        // p3 is noise
        assert_eq!(r.labels[3], -1, "p3 (boundary) is noise, got label={}", r.labels[3]);
    }

    /// DBSCAN with spherical geodesic distance.
    ///
    /// For unit vectors on the sphere, geodesic distance is arc length (arccos of dot).
    /// Points that are "nearly parallel" (small angle) cluster together.
    ///
    /// p0=(1,0), p1≈(cos 0.3, sin 0.3) — 0.3 rad apart from p0
    /// p2=(0,1), p3≈(cos(π/2-0.2), sin(π/2-0.2)) — 0.2 rad apart from p2
    ///
    /// With epsilon=0.5 rad and min_samples=2:
    /// - {p0, p1} form one cluster (d≈0.3)
    /// - {p2, p3} form another cluster (d≈0.2)
    #[test]
    fn sphere_geodesic_dbscan_two_angular_clusters() {
        use crate::manifold::{ManifoldDistanceOp, Manifold};

        let theta1: f64 = 0.3_f64; // small angle from (1,0)
        let theta2: f64 = std::f64::consts::PI / 2.0 - 0.2; // near (0,1)

        let data = vec![
            1.0_f64, 0.0,               // p0: (1, 0)
            theta1.cos(), theta1.sin(), // p1: 0.3 rad from p0
            0.0_f64, 1.0,               // p2: (0, 1)
            theta2.cos(), theta2.sin(), // p3: 0.2 rad from p2
        ];

        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        let mut e = engine();

        // epsilon = 0.5 rad: captures tight angular clusters, not cross-cluster neighbors
        let r = e.discover_clusters(&data, 4, 2, 0.5, 2, &op).unwrap();

        assert_eq!(r.n_clusters, 2, "two angular clusters, got {}", r.n_clusters);
        assert_eq!(r.n_noise, 0, "no noise, got {} noise", r.n_noise);
        // p0 and p1 in the same cluster
        assert_eq!(r.labels[0], r.labels[1], "p0 and p1 (near (1,0)) same cluster");
        // p2 and p3 in the same cluster
        assert_eq!(r.labels[2], r.labels[3], "p2 and p3 (near (0,1)) same cluster");
        // The two clusters are different
        assert_ne!(r.labels[0], r.labels[2], "two distinct clusters");
    }
}
