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
/// This is the **pure CPU primitive** for DBSCAN — density estimation + core
/// identification + union-find component labeling + border-point assignment.
/// It operates on any precomputed n×n distance slice regardless of metric.
///
/// ## Parameters
///
/// - `dist`: flat n×n distance matrix (row-major). `dist[i*n+j]` = distance(i,j).
///   Distance values must be non-negative; the metric is caller-supplied.
/// - `n`: number of points.
/// - `epsilon_threshold`: neighborhood radius in the caller's distance units.
///   For L2² distances pass `radius²`; for L2 distances pass `radius`.
/// - `min_samples`: minimum neighbors (including self) for a point to be a core.
///
/// ## Returns
///
/// [`ClusterResult`] with per-point labels: `-1` = noise, `0..k-1` = clusters.
///
/// ## Why public
///
/// This function is math that exists independently of GPU infrastructure. Any
/// algorithm that builds a pairwise distance matrix (KNN graph, manifold
/// topology, spectral clustering, etc.) can reuse this CPU primitive directly
/// without instantiating a [`ClusteringEngine`]. The GPU engine is an
/// optimization wrapper; this is the core algorithm.
///
/// Kingdom A: all steps are accumulate+gather over the n×n distance slice.
pub fn clustering_from_distance(
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
// Union-find — public primitives
// ---------------------------------------------------------------------------

/// Allocate a fresh union-find structure for `n` elements.
///
/// Each element starts in its own component: `parent[i] == i`.
/// Pass the returned `Vec<usize>` to `uf_find` and `uf_union`.
///
/// Kingdom A: O(n) initialisation.
pub fn uf_new(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Iterative path-halving find — no recursion, O(α(n)) amortized.
///
/// Returns the root (representative) of the component containing `x`.
/// Mutates `parent` in-place to flatten the path (path-halving variant).
///
/// # Consumers
/// DBSCAN (core-point merging), Kruskal MST, connected-components,
/// any algorithm that needs online component merging.
pub fn uf_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

/// Union the components containing `a` and `b`.
///
/// After this call, `uf_find(parent, a) == uf_find(parent, b)`.
/// Uses union-by-index (roots point to the smaller root index) for
/// deterministic output; caller may use `parent[ra] = rb` directly
/// for union-by-arrival when determinism is not required.
pub fn uf_union(parent: &mut Vec<usize>, a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        // smaller root wins → deterministic component IDs
        if ra < rb { parent[rb] = ra; } else { parent[ra] = rb; }
    }
}

// ---------------------------------------------------------------------------
// Cluster validation metrics (CPU, pure Rust)
// ---------------------------------------------------------------------------

/// Cluster validation metrics computed from raw data and cluster labels.
#[derive(Debug, Clone)]
pub struct ClusterValidation {
    /// Silhouette coefficient (mean over all non-noise points). Range [-1, 1].
    /// Higher is better; > 0.5 indicates well-separated clusters.
    pub silhouette: f64,
    /// Calinski-Harabasz index (ratio of between/within cluster variance).
    /// Higher is better.
    pub calinski_harabasz: f64,
    /// Davies-Bouldin index. Lower is better; 0 = perfect clustering.
    pub davies_bouldin: f64,
}

// ── Shared centroid computation ───────────────────────────────────────────

/// Prepared cluster geometry: sizes, per-cluster centroids, and the compact
/// label-to-index map. Produced by `cluster_centroids` and consumed by the
/// three standalone validation primitives so the centroid pass runs only once.
///
/// This struct is the minimum sufficient representation shared by
/// `calinski_harabasz_score`, `davies_bouldin_score`, and `silhouette_score`.
#[derive(Debug, Clone)]
pub struct ClusterCentroids {
    /// Number of clusters k (noise-free labels only).
    pub k: usize,
    /// Number of dimensions d.
    pub n_dims: usize,
    /// Cluster sizes (length k).
    pub sizes: Vec<usize>,
    /// Row-major k×d centroid matrix (length k × n_dims).
    pub centroids: Vec<f64>,
    /// Map from original label value → compact index in [0, k).
    pub id_to_idx: std::collections::HashMap<i32, usize>,
}

/// Compute cluster centroids from raw data and integer labels.
///
/// Labels < 0 are treated as noise and ignored. Returns `None` if fewer than
/// 2 valid clusters exist.
///
/// **Primitive**: shared intermediate for all cluster validation metrics.
/// Called once per validation pass; all three score primitives consume the
/// result rather than recomputing centroids independently.
///
/// Kingdom A: single linear pass over data and labels, O(n·d).
pub fn cluster_centroids(data: &[f64], labels: &[i32], n_dims: usize) -> Option<ClusterCentroids> {
    let n = labels.len();
    assert_eq!(data.len(), n * n_dims);

    // Identify unique non-noise cluster IDs
    let mut cluster_ids: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    cluster_ids.sort_unstable();
    cluster_ids.dedup();
    let k = cluster_ids.len();
    if k < 2 { return None; }

    // Map cluster label → compact index
    let id_to_idx: std::collections::HashMap<i32, usize> =
        cluster_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Accumulate cluster sums → divide by size to get centroids
    let mut sizes = vec![0usize; k];
    let mut centroids = vec![0.0_f64; k * n_dims];
    for i in 0..n {
        if let Some(&ci) = id_to_idx.get(&labels[i]) {
            sizes[ci] += 1;
            for d in 0..n_dims {
                centroids[ci * n_dims + d] += data[i * n_dims + d];
            }
        }
    }
    for ci in 0..k {
        if sizes[ci] > 0 {
            for d in 0..n_dims { centroids[ci * n_dims + d] /= sizes[ci] as f64; }
        }
    }

    Some(ClusterCentroids { k, n_dims, sizes, centroids, id_to_idx })
}

// ── Standalone validation score primitives ───────────────────────────────

/// Calinski-Harabász index for a clustering.
///
/// CH = [SS_B / (k−1)] / [SS_W / (n−k)]
/// where SS_B = Σ_c nₓ ‖centroid_c − global_centroid‖² (between-cluster scatter)
/// and   SS_W = Σ_c Σ_{i in c} ‖xᵢ − centroid_c‖²     (within-cluster scatter)
///
/// Higher is better. Returns `f64::INFINITY` when SS_W = 0 (perfect clusters)
/// and NaN for degenerate input (< 2 clusters).
///
/// # Parameters
/// - `data`: n×d row-major data matrix
/// - `labels`: cluster labels (negative = noise, excluded)
/// - `n_dims`: d
/// - `cc`: pre-computed centroids from [`cluster_centroids`], or `None` to
///   compute internally (use `None` when calling this primitive standalone;
///   pass `Some` when sharing with other score primitives)
///
/// **Primitive**: exists independently of the validation bundle. Kingdom A.
pub fn calinski_harabasz_score(
    data: &[f64], labels: &[i32], n_dims: usize,
    cc: Option<&ClusterCentroids>,
) -> f64 {
    let owned;
    let cc = match cc {
        Some(c) => c,
        None => {
            owned = cluster_centroids(data, labels, n_dims);
            match owned.as_ref() { Some(c) => c, None => return f64::NAN }
        }
    };
    let ClusterCentroids { k, n_dims, sizes, centroids, id_to_idx } = cc;
    let n = labels.len();

    let sq_dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    };

    let n_clustered = sizes.iter().sum::<usize>() as f64;
    let mut global_centroid = vec![0.0_f64; *n_dims];
    for ci in 0..*k {
        for d in 0..*n_dims {
            global_centroid[d] += sizes[ci] as f64 * centroids[ci * n_dims + d];
        }
    }
    for d in 0..*n_dims { global_centroid[d] /= n_clustered; }

    let ss_b: f64 = (0..*k)
        .map(|ci| sizes[ci] as f64 * sq_dist(&centroids[ci*n_dims..(ci+1)*n_dims], &global_centroid))
        .sum();
    let mut ss_w = 0.0_f64;
    for i in 0..n {
        if let Some(&ci) = id_to_idx.get(&labels[i]) {
            ss_w += sq_dist(&data[i*n_dims..(i+1)*n_dims], &centroids[ci*n_dims..(ci+1)*n_dims]);
        }
    }
    let n_k = n_clustered - *k as f64;
    if ss_w < 1e-300 || n_k <= 0.0 {
        f64::INFINITY
    } else {
        (ss_b / (*k as f64 - 1.0)) / (ss_w / n_k)
    }
}

/// Davies-Bouldin index for a clustering.
///
/// DB = (1/k) Σᵢ max_{j≠i} (sᵢ + sⱼ) / d(cᵢ, cⱼ)
/// where sᵢ = mean intra-cluster Euclidean distance to cluster i's centroid.
///
/// Lower is better. Returns 0 for perfect (tight, well-separated) clusters.
///
/// # Parameters
/// Same as [`calinski_harabasz_score`] — pass `Some(&cc)` to share centroids.
///
/// **Primitive**: exists independently. Kingdom A.
pub fn davies_bouldin_score(
    data: &[f64], labels: &[i32], n_dims: usize,
    cc: Option<&ClusterCentroids>,
) -> f64 {
    let owned;
    let cc = match cc {
        Some(c) => c,
        None => {
            owned = cluster_centroids(data, labels, n_dims);
            match owned.as_ref() { Some(c) => c, None => return f64::NAN }
        }
    };
    let ClusterCentroids { k, n_dims, sizes, centroids, id_to_idx } = cc;
    let n = labels.len();

    let sq_dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    };

    let mut s = vec![0.0_f64; *k]; // mean intra-cluster distance to centroid
    for i in 0..n {
        if let Some(&ci) = id_to_idx.get(&labels[i]) {
            s[ci] += sq_dist(&data[i*n_dims..(i+1)*n_dims], &centroids[ci*n_dims..(ci+1)*n_dims]).sqrt();
        }
    }
    for ci in 0..*k { if sizes[ci] > 0 { s[ci] /= sizes[ci] as f64; } }

    let db_sum: f64 = (0..*k).map(|i| {
        (0..*k).filter(|&j| j != i).map(|j| {
            let d_ij = sq_dist(&centroids[i*n_dims..(i+1)*n_dims], &centroids[j*n_dims..(j+1)*n_dims]).sqrt();
            if d_ij < 1e-300 { 0.0 } else { (s[i] + s[j]) / d_ij }
        }).fold(0.0_f64, f64::max)
    }).sum();
    db_sum / *k as f64
}

/// Mean silhouette coefficient for a clustering.
///
/// s(i) = (b(i) − a(i)) / max(a(i), b(i))
/// where a(i) = mean distance to own cluster, b(i) = min mean distance to any
/// other cluster. Mean taken over all non-noise points in clusters of size ≥ 2.
///
/// Range [−1, 1]. Values > 0.5 indicate well-separated clusters.
/// Returns 0.0 if no valid points exist.
///
/// # Parameters
/// - `data`, `labels`, `n_dims`: raw inputs (centroids are not needed — silhouette
///   uses pairwise distances to actual points, not centroid approximations)
///
/// **Primitive**: exists independently. Kingdom B (pairwise O(n²) inner loop).
pub fn silhouette_score(data: &[f64], labels: &[i32], n_dims: usize) -> f64 {
    let n = labels.len();
    assert_eq!(data.len(), n * n_dims);

    // Build id_to_idx and sizes (minimal centroid computation — sizes only)
    let mut cluster_ids: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    cluster_ids.sort_unstable();
    cluster_ids.dedup();
    let k = cluster_ids.len();
    if k < 2 { return 0.0; }

    let id_to_idx: std::collections::HashMap<i32, usize> =
        cluster_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut sizes = vec![0usize; k];
    for &l in labels { if let Some(&ci) = id_to_idx.get(&l) { sizes[ci] += 1; } }

    let sq_dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    };

    let mut sil_sum = 0.0_f64;
    let mut sil_count = 0usize;

    for i in 0..n {
        let ci = match id_to_idx.get(&labels[i]) {
            Some(&ci) => ci,
            None => continue,
        };
        if sizes[ci] < 2 { continue; }

        // a(i): mean distance to other points in the same cluster
        let a = {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for j in 0..n {
                if j == i { continue; }
                if let Some(&cj) = id_to_idx.get(&labels[j]) {
                    if cj == ci {
                        sum += sq_dist(&data[i*n_dims..(i+1)*n_dims], &data[j*n_dims..(j+1)*n_dims]).sqrt();
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 { 0.0 } else { sum / cnt as f64 }
        };

        // b(i): min mean distance to points in any other cluster
        let b = {
            let mut min_b = f64::INFINITY;
            for cj in 0..k {
                if cj == ci { continue; }
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for j in 0..n {
                    if let Some(&cjj) = id_to_idx.get(&labels[j]) {
                        if cjj == cj {
                            sum += sq_dist(&data[i*n_dims..(i+1)*n_dims], &data[j*n_dims..(j+1)*n_dims]).sqrt();
                            cnt += 1;
                        }
                    }
                }
                if cnt > 0 { min_b = min_b.min(sum / cnt as f64); }
            }
            min_b
        };

        let max_ab = a.max(b);
        let sil_i = if max_ab < 1e-300 { 0.0 } else { (b - a) / max_ab };
        sil_sum += sil_i;
        sil_count += 1;
    }

    if sil_count == 0 { 0.0 } else { sil_sum / sil_count as f64 }
}

// ── Validation bundle ─────────────────────────────────────────────────────

/// Compute cluster validation metrics from raw data and cluster assignments.
///
/// Bundles three standalone score primitives — [`calinski_harabasz_score`],
/// [`davies_bouldin_score`], and [`silhouette_score`] — computing shared
/// centroids once via [`cluster_centroids`] and passing them to each primitive.
///
/// `data`: n×d row-major matrix (n points, d dimensions).
/// `labels`: length-n cluster labels (noise/outliers should use label = -1 or
///   any value < 0; these points are excluded from silhouette computation
///   but included in CH/DB).
/// `n_dims`: d (number of dimensions per point).
///
/// Returns `None` if < 2 clusters or insufficient data.
pub fn cluster_validation(data: &[f64], labels: &[i32], n_dims: usize) -> Option<ClusterValidation> {
    // Compute shared centroids once.
    let cc = cluster_centroids(data, labels, n_dims)?;

    // Each score primitive receives the shared ClusterCentroids.
    // Silhouette uses pairwise distances (not centroids), so it computes
    // its own id_to_idx + sizes from labels directly — no centroid needed.
    let calinski_harabasz = calinski_harabasz_score(data, labels, n_dims, Some(&cc));
    let davies_bouldin = davies_bouldin_score(data, labels, n_dims, Some(&cc));
    let silhouette = silhouette_score(data, labels, n_dims);

    Some(ClusterValidation { silhouette, calinski_harabasz, davies_bouldin })
}

// ---------------------------------------------------------------------------
// Hierarchical (agglomerative) clustering
// ---------------------------------------------------------------------------

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    /// Minimum distance between any pair (tends to produce elongated clusters).
    Single,
    /// Maximum distance between any pair (tends to produce compact clusters).
    Complete,
    /// Mean distance between all pairs.
    Average,
    /// Minimize total within-cluster variance (Ward 1963).
    Ward,
}

/// One merge step in the dendrogram.
#[derive(Debug, Clone)]
pub struct DendrogramStep {
    /// Index of the first cluster merged.
    pub cluster_a: usize,
    /// Index of the second cluster merged.
    pub cluster_b: usize,
    /// Distance at which the merge occurred.
    pub distance: f64,
    /// Size of the merged cluster.
    pub size: usize,
}

/// Result of hierarchical clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Merge history (n-1 steps for n observations).
    pub dendrogram: Vec<DendrogramStep>,
    /// Cluster labels when cut at k clusters.
    pub labels: Vec<i32>,
    /// Number of clusters requested.
    pub k: usize,
}

/// Agglomerative hierarchical clustering.
///
/// `data`: n x d row-major matrix. `n_dims`: d. `k`: number of clusters to produce.
/// `linkage`: merge criterion (Single, Complete, Average, Ward).
///
/// Uses O(n^2) distance matrix + Lance-Williams recurrence for O(n^2 log n) total.
pub fn hierarchical_clustering(
    data: &[f64], n: usize, n_dims: usize, k: usize, linkage: Linkage,
) -> HierarchicalResult {
    assert_eq!(data.len(), n * n_dims);
    let k = k.max(1).min(n);

    // Pairwise squared Euclidean distances
    let mut dist = vec![f64::INFINITY; n * n];
    for i in 0..n {
        dist[i * n + i] = 0.0;
        for j in (i + 1)..n {
            let d: f64 = (0..n_dims).map(|dim| {
                (data[i * n_dims + dim] - data[j * n_dims + dim]).powi(2)
            }).sum();
            // For Ward, store squared distance; for others, store Euclidean
            let d = match linkage { Linkage::Ward => d, _ => d.sqrt() };
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut sizes = vec![1usize; n];
    let mut active = vec![true; n]; // which clusters are still active
    let mut labels: Vec<usize> = (0..n).collect(); // each point starts as its own cluster
    let mut dendrogram = Vec::with_capacity(n.saturating_sub(1));

    for _step in 0..(n.saturating_sub(k)) {
        // Find closest pair among active clusters
        let mut best_d = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;
        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                if dist[i * n + j] < best_d {
                    best_d = dist[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best_d == f64::INFINITY { break; }

        // Merge j into i
        let ni = sizes[best_i] as f64;
        let nj = sizes[best_j] as f64;

        dendrogram.push(DendrogramStep {
            cluster_a: best_i, cluster_b: best_j,
            distance: best_d, size: sizes[best_i] + sizes[best_j],
        });

        // Update labels: everything in cluster best_j → best_i
        for lbl in labels.iter_mut() {
            if *lbl == best_j { *lbl = best_i; }
        }

        // Lance-Williams update: d(i∪j, m) for all active m
        for m in 0..n {
            if !active[m] || m == best_i || m == best_j { continue; }
            let nm = sizes[m] as f64;
            let di = dist[best_i * n + m];
            let dj = dist[best_j * n + m];
            let dij = dist[best_i * n + best_j];

            let new_d = match linkage {
                Linkage::Single => di.min(dj),
                Linkage::Complete => di.max(dj),
                Linkage::Average => (ni * di + nj * dj) / (ni + nj),
                Linkage::Ward => {
                    let total = ni + nj + nm;
                    ((ni + nm) * di + (nj + nm) * dj - nm * dij) / total
                }
            };
            dist[best_i * n + m] = new_d;
            dist[m * n + best_i] = new_d;
        }

        sizes[best_i] += sizes[best_j];
        active[best_j] = false;
    }

    // Compact labels to 0..k-1
    let active_clusters: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let mut final_labels = vec![0i32; n];
    for i in 0..n {
        let cluster_idx = active_clusters.iter().position(|&c| labels[i] == c).unwrap_or(0);
        final_labels[i] = cluster_idx as i32;
    }

    HierarchicalResult { dendrogram, labels: final_labels, k }
}

// ---------------------------------------------------------------------------
// Hopkins statistic (Hopkins & Skellam 1954)
// ---------------------------------------------------------------------------

/// Hopkins statistic for clustering tendency.
///
/// Tests whether a dataset has meaningful cluster structure vs. uniform random.
/// H ≈ 0.5: uniform (no clustering). H → 1: highly clustered.
/// Under H₀ (uniform), H ~ Beta(m, m).
///
/// `data`: n×d row-major matrix. `m`: sample size (typically min(n/10, 100)).
pub fn hopkins_statistic(data: &[f64], n: usize, d: usize, m: usize, seed: u64) -> f64 {
    if n < 2 || m == 0 || d == 0 { return 0.5; }
    let m = m.min(n);

    let mut rng = crate::rng::Xoshiro256::new(seed);

    // Bounding box per dimension
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for i in 0..n {
        for j in 0..d {
            let v = data[i * d + j];
            if v < mins[j] { mins[j] = v; }
            if v > maxs[j] { maxs[j] = v; }
        }
    }

    // w_sum: sum of squared distances from m uniform random points to nearest data neighbor
    let mut w_sum = 0.0;
    for _ in 0..m {
        let pt: Vec<f64> = (0..d).map(|j| {
            crate::rng::TamRng::next_f64_range(&mut rng, mins[j], maxs[j].max(mins[j] + 1e-15))
        }).collect();
        let mut min_dist = f64::INFINITY;
        for i in 0..n {
            let dist: f64 = (0..d).map(|j| (pt[j] - data[i * d + j]).powi(2)).sum();
            if dist < min_dist { min_dist = dist; }
        }
        w_sum += min_dist;
    }

    // u_sum: sum of squared distances from m random data points to nearest OTHER data neighbor
    let mut u_sum = 0.0;
    let indices = crate::rng::sample_without_replacement(&mut rng, n, m);
    for &idx in &indices {
        let mut min_dist = f64::INFINITY;
        for i in 0..n {
            if i == idx { continue; }
            let dist: f64 = (0..d).map(|j| {
                (data[idx * d + j] - data[i * d + j]).powi(2)
            }).sum();
            if dist < min_dist { min_dist = dist; }
        }
        u_sum += min_dist;
    }

    if w_sum + u_sum < 1e-300 { return 0.5; }
    w_sum / (w_sum + u_sum)
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

    // ── Cluster validation metrics ────────────────────────────────────────

    #[test]
    fn validation_perfect_clusters() {
        // Two tight, well-separated clusters in 2D
        // Cluster 0: points around (0, 0)
        // Cluster 1: points around (100, 100)
        let data = vec![
            0.0, 0.0,   0.1, 0.0,   0.0, 0.1,   0.1, 0.1,   // cluster 0
            100.0, 100.0, 100.1, 100.0, 100.0, 100.1, 100.1, 100.1, // cluster 1
        ];
        let labels = vec![0i32, 0, 0, 0, 1, 1, 1, 1];
        let r = cluster_validation(&data, &labels, 2).unwrap();

        // Silhouette should be close to 1 (very well separated)
        assert!(r.silhouette > 0.9,
            "well-separated clusters: silhouette={:.4} should be > 0.9", r.silhouette);

        // Davies-Bouldin should be very small (tight clusters, large inter-cluster distance)
        assert!(r.davies_bouldin < 0.01,
            "well-separated: DB={:.6} should be near 0", r.davies_bouldin);

        // Calinski-Harabasz should be very large
        assert!(r.calinski_harabasz > 100.0,
            "well-separated: CH={:.2} should be large", r.calinski_harabasz);
    }

    #[test]
    fn validation_overlapping_clusters_lower_silhouette() {
        // Two overlapping clusters — silhouette should be lower
        let data = vec![
            0.0, 0.0,  1.0, 0.0,  2.0, 0.0,  3.0, 0.0, // cluster 0
            2.5, 0.0,  3.5, 0.0,  4.5, 0.0,  5.5, 0.0, // cluster 1
        ];
        let labels = vec![0i32, 0, 0, 0, 1, 1, 1, 1];
        let tight = cluster_validation(&data, &labels, 2).unwrap();

        // Compare against the well-separated case — silhouette should be lower
        let separated_data = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
            100.0, 100.0, 100.1, 100.0, 100.0, 100.1, 100.1, 100.1,
        ];
        let separated_labels = vec![0i32, 0, 0, 0, 1, 1, 1, 1];
        let separated = cluster_validation(&separated_data, &separated_labels, 2).unwrap();

        assert!(tight.silhouette < separated.silhouette,
            "overlapping ({:.3}) should have lower silhouette than separated ({:.3})",
            tight.silhouette, separated.silhouette);
    }

    #[test]
    fn validation_noise_points_excluded_from_silhouette() {
        // One noise point (-1 label) — should be excluded from silhouette
        let data = vec![
            0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  // cluster 0
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // cluster 1
            5.0, 5.0,   // "noise"
        ];
        let labels = vec![0i32, 0, 0, 1, 1, 1, -1];
        // Should not panic, noise point excluded
        let r = cluster_validation(&data, &labels, 2).unwrap();
        assert!(r.silhouette.is_finite(), "silhouette should be finite");
    }

    #[test]
    fn validation_fewer_than_two_clusters_returns_none() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let labels = vec![0i32, 0, 0]; // only one cluster
        assert!(cluster_validation(&data, &labels, 2).is_none());

        // All noise
        let noise_labels = vec![-1i32, -1, -1];
        assert!(cluster_validation(&data, &noise_labels, 2).is_none());
    }

    // ── Hopkins statistic ─────────────────────────────────────────────

    #[test]
    fn hopkins_clustered_data_high() {
        // Two tight clusters in 2D — should give H > 0.5
        let mut data = Vec::new();
        for _ in 0..50 { data.extend_from_slice(&[0.0, 0.0]); } // cluster at origin
        for _ in 0..50 { data.extend_from_slice(&[10.0, 10.0]); } // cluster at (10,10)
        // Add small noise
        let mut rng = crate::rng::Xoshiro256::new(42);
        for v in data.iter_mut() {
            *v += crate::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let h = hopkins_statistic(&data, 100, 2, 10, 42);
        assert!(h > 0.5, "H={} should be > 0.5 for clustered data", h);
    }

    #[test]
    fn hopkins_regular_grid_low() {
        // Regular grid — MORE uniform than random → H < 0.5
        let mut data = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                data.push(i as f64);
                data.push(j as f64);
            }
        }
        let h = hopkins_statistic(&data, 100, 2, 10, 42);
        assert!(h < 0.5, "H={} should be < 0.5 for regular grid (anti-clustered)", h);
    }

    // ── Hierarchical clustering ──────────────────────────────────────

    #[test]
    fn hierarchical_two_clusters_ward() {
        // Two well-separated clusters in 2D
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.0, 0.1,  0.1, 0.0, // cluster A near origin
            10.0, 10.0, 10.1, 10.1, 10.0, 10.1, 10.1, 10.0, // cluster B
        ];
        let result = hierarchical_clustering(&data, 8, 2, 2, Linkage::Ward);
        assert_eq!(result.k, 2);
        assert_eq!(result.dendrogram.len(), 6); // n-k = 8-2 = 6 merges
        // First 4 points should be in one cluster, last 4 in the other
        let first_cluster = result.labels[0];
        let second_cluster = result.labels[4];
        assert_ne!(first_cluster, second_cluster, "Two well-separated groups should be in different clusters");
        for i in 0..4 {
            assert_eq!(result.labels[i], first_cluster, "First 4 points should share a cluster");
        }
        for i in 4..8 {
            assert_eq!(result.labels[i], second_cluster, "Last 4 points should share a cluster");
        }
    }

    #[test]
    fn hierarchical_single_linkage_chain() {
        // Chain of points: single linkage should group them as one cluster
        let data = vec![0.0, 0.0,  1.0, 0.0,  2.0, 0.0,  3.0, 0.0,  4.0, 0.0];
        let result = hierarchical_clustering(&data, 5, 2, 1, Linkage::Single);
        assert_eq!(result.k, 1);
        // All points should be in the same cluster
        for i in 1..5 {
            assert_eq!(result.labels[i], result.labels[0]);
        }
    }

    #[test]
    fn hierarchical_complete_linkage() {
        // Simple 3-cluster test with complete linkage
        let data = vec![
            0.0, 0.0,  0.5, 0.5,      // cluster 1
            5.0, 5.0,  5.5, 5.5,      // cluster 2
            10.0, 0.0, 10.5, 0.5,     // cluster 3
        ];
        let result = hierarchical_clustering(&data, 6, 2, 3, Linkage::Complete);
        assert_eq!(result.k, 3);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_eq!(result.labels[4], result.labels[5]);
    }

    #[test]
    fn hierarchical_dendrogram_distances_monotone_ward() {
        // Ward dendrogram distances should generally increase (merges become costlier)
        let data = vec![
            0.0, 0.0,  0.1, 0.0,
            5.0, 0.0,  5.1, 0.0,
            10.0, 0.0, 10.1, 0.0,
        ];
        let result = hierarchical_clustering(&data, 6, 2, 1, Linkage::Ward);
        assert_eq!(result.dendrogram.len(), 5);
        // Later merges should have larger distances than very first merge
        assert!(result.dendrogram.last().unwrap().distance >
                result.dendrogram[0].distance);
    }
}
