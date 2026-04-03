
use std::sync::Arc;
use crate::clustering::ClusteringEngine;
use crate::intermediates::{DataId, DistanceMatrix, IntermediateTag, Metric, TamSession};
use crate::kmeans::KMeansEngine;
use crate::knn::{self, KnnResult};
use crate::manifold::{ManifoldDistanceOp, ManifoldMixture};
use crate::train::linear::{self, LinearModel};
use crate::train::logistic::{self, LogisticModel};
use winrapids_tiled::{DistanceOp, TiledEngine};

// ---------------------------------------------------------------------------
// TamFrame — the value that flows through the chain
// ---------------------------------------------------------------------------

/// A row-major data frame: `n` points × `d` dimensions.
///
/// This is the type that flows through a `TamPipeline` chain. Each step
/// produces a new frame (or augments the current one). The frame is cheap to
/// move — data is held as an owned `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct TamFrame {
    /// Row-major feature data: `data[i * d + j]` = feature j of point i.
    pub data: Vec<f64>,

    /// Number of points (rows).
    pub n: usize,

    /// Number of dimensions (columns) in `data`.
    pub d: usize,

    /// Cluster labels from the most recent clustering step.
    /// `labels[i] == -1` → point i is noise (DBSCAN).
    pub labels: Option<Vec<i32>>,

    /// Number of clusters found by the most recent clustering step.
    pub n_clusters: Option<usize>,

    /// K-nearest neighbor result from the most recent KNN step.
    pub knn_result: Option<KnnResult>,
}

impl TamFrame {
    fn new(data: Vec<f64>, n: usize, d: usize) -> Self {
        assert_eq!(data.len(), n * d, "TamFrame: data.len() must equal n * d");
        Self { data, n, d, labels: None, n_clusters: None, knn_result: None }
    }
}

// ---------------------------------------------------------------------------
// DescribeResult — per-column statistics from .describe()
// ---------------------------------------------------------------------------

/// Per-column descriptive statistics from `.describe()`.
#[derive(Debug, Clone)]
pub struct DescribeResult {
    /// One entry per column.
    pub columns: Vec<ColumnDescribe>,
}

/// Statistics for a single column.
#[derive(Debug, Clone)]
pub struct ColumnDescribe {
    /// Column index (0-based).
    pub index: usize,
    /// Number of valid (non-NaN) observations.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (Bessel-corrected).
    pub std: f64,
    /// Minimum value.
    pub min: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Maximum value.
    pub max: f64,
    /// Interquartile range (Q3 − Q1).
    pub iqr: f64,
    /// Sample skewness (Fisher's adjusted).
    pub skewness: f64,
    /// Excess kurtosis (Fisher's definition, normal = 0).
    pub kurtosis: f64,
}

// ---------------------------------------------------------------------------
// TamPipeline — the fluent builder
// ---------------------------------------------------------------------------

/// Fluent pipeline over a `TamFrame`.
///
/// Holds engines (initialized once) and a `TamSession` (shared across all
/// steps). The caller never interacts with individual engines or the session
/// directly — the pipeline wires them automatically.
pub struct TamPipeline {
    frame: TamFrame,
    session: TamSession,
    clustering: ClusteringEngine,
    kmeans: KMeansEngine,
    tiled: TiledEngine,
}

impl TamPipeline {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Build a pipeline from an owned data buffer.
    ///
    /// `data` must be row-major: `data[i * d + j]` = feature j of point i.
    /// `n * d` must equal `data.len()`.
    pub fn from_slice(
        data: Vec<f64>,
        n: usize,
        d: usize,
    ) -> Self {
        // Defer GPU init — fail on first GPU step if no CUDA
        let clustering = ClusteringEngine::new().expect("ClusteringEngine::new failed");
        let kmeans = KMeansEngine::new().expect("KMeansEngine::new failed");
        let tiled = {
            #[cfg(feature = "wgpu")]
            { TiledEngine::new(Arc::from(tambear_wgpu::detect_wgpu())) }
            #[cfg(not(feature = "wgpu"))]
            { TiledEngine::new(tam_gpu::detect()) }
        };
        Self {
            frame: TamFrame::new(data, n, d),
            session: TamSession::new(),
            clustering,
            kmeans,
            tiled,
        }
    }

    // -----------------------------------------------------------------------
    // Preprocessing
    // -----------------------------------------------------------------------

    /// Z-score normalize each column independently.
    ///
    /// For each column j: `x'[i][j] = (x[i][j] - mean_j) / std_j`.
    /// Constant columns (std < 1e-15) are left unchanged (divide by 1).
    ///
    /// Produces a `SufficientStatistics` intermediate that is registered in
    /// the session. Downstream `train_linear()` can reuse these stats if it
    /// operates on the same normalized data.
    pub fn normalize(mut self) -> Self {
        let stats = linear::column_stats(&self.frame.data, self.frame.n, self.frame.d);
        let n = self.frame.n;
        let d = self.frame.d;

        for i in 0..n {
            for j in 0..d {
                let mean = stats.mean(j);
                #[allow(deprecated)]
                let std = {
                    let s = stats.std(j);
                    if s < 1e-15 { 1.0 } else { s }
                };
                self.frame.data[i * d + j] = (self.frame.data[i * d + j] - mean) / std;
            }
        }
        self
    }

    // -----------------------------------------------------------------------
    // Clustering
    // -----------------------------------------------------------------------

    /// Run DBSCAN clustering on the current frame.
    ///
    /// Uses the session for the n×n L2Sq distance matrix. If a prior step
    /// already computed the distance matrix for this data, the GPU kernel is
    /// skipped — zero additional GPU cost.
    ///
    /// After this step, `pipeline.frame().labels` contains per-point cluster
    /// assignments (−1 = noise).
    pub fn discover_clusters(mut self, epsilon: f64, min_samples: usize) -> Self {
        let result = self.clustering.discover_clusters_session(
            &mut self.session,
            &self.frame.data,
            self.frame.n,
            self.frame.d,
            epsilon,
            min_samples,
            &DistanceOp,
        ).expect("discover_clusters_session failed");

        self.frame.n_clusters = Some(result.n_clusters);
        self.frame.labels = Some(result.labels);
        self
    }

    /// Run DBSCAN on a manifold mixture distance matrix.
    ///
    /// For each component `(manifold, weight)` in `mix`:
    /// 1. Checks the session for a cached `ManifoldDistanceMatrix`
    /// 2. If missing: computes via `ManifoldDistanceOp + TiledEngine`, registers in session
    /// 3. Combines all component matrices via weighted sum (O(n²), not O(n²d))
    /// 4. Runs DBSCAN on the combined distance matrix
    ///
    /// The combined matrix is cached under `ManifoldMixtureDistance { mix_id, data_id }`.
    ///
    /// ## Notes on epsilon
    ///
    /// `epsilon_threshold` is used verbatim as the DBSCAN density radius.
    /// For Euclidean-only mixtures: use `epsilon_radius²`. For mixtures with
    /// cosine distances (range 0–2): calibrate by examining the combined distances.
    pub fn discover_clusters_mixture(
        mut self,
        mix: &ManifoldMixture,
        epsilon_threshold: f64,
        min_samples: usize,
    ) -> Self {
        let n = self.frame.n;
        let d = self.frame.d;
        let data_id = DataId::from_f64(&self.frame.data);

        // Transpose of data: d×n, for TiledEngine's B argument (K×N format)
        // Local reference first — avoids moving self.frame.data into the closure
        let data_t: Vec<f64> = {
            let data = &self.frame.data;
            (0..d).flat_map(|k| (0..n).map(move |i| data[i * d + k])).collect()
        };

        // Build component distance matrices (session-cached)
        let mut matrices: Vec<Arc<DistanceMatrix>> = Vec::new();
        for (manifold, _) in &mix.components {
            let tag = IntermediateTag::ManifoldDistanceMatrix {
                manifold_name: manifold.name(),
                data_id,
            };

            let dm = if let Some(cached) = self.session.get::<DistanceMatrix>(&tag) {
                cached
            } else {
                let dist_data = self.tiled.run(
                    &ManifoldDistanceOp::new(manifold.clone()),
                    &self.frame.data,
                    &data_t,
                    n, n, d,
                ).expect("ManifoldDistanceOp::run failed");
                // Note: Metric::L2Sq is a placeholder; the real metric is in the tag
                let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, dist_data));
                self.session.register(tag, Arc::clone(&dm));
                dm
            };
            matrices.push(dm);
        }

        // Combine: O(n²) weighted sum of cached matrices
        let combined = mix.combine(&matrices);

        // Cache the combined matrix
        let mix_id = mix.mix_id();
        let mix_tag = IntermediateTag::ManifoldMixtureDistance { mix_id, data_id };
        if self.session.get::<DistanceMatrix>(&mix_tag).is_none() {
            let combined_dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, combined.clone()));
            self.session.register(mix_tag, combined_dm);
        }

        // Run DBSCAN on combined distance matrix
        let result = ClusteringEngine::discover_clusters_from_combined(
            &combined, n, epsilon_threshold, min_samples,
        );
        self.frame.n_clusters = Some(result.n_clusters);
        self.frame.labels = Some(result.labels);
        self
    }

    /// Run KMeans clustering on the current frame.
    ///
    /// Converts data to f32 (KMeans uses f32 for GPU efficiency), runs `max_iter`
    /// iterations of Euclidean KMeans, and stores the result in
    /// `frame.labels` and `frame.n_clusters`.
    ///
    /// The centroids are registered in the session under
    /// `IntermediateTag::Centroids { data_id, k }`. Subsequent calls with the
    /// same data and k return the cached result.
    ///
    /// Labels are `0..k-1` (all points assigned; no noise concept in KMeans).
    pub fn kmeans(mut self, k: usize, max_iter: usize) -> Self {
        use std::sync::Arc;

        let data_id = DataId::from_f64(&self.frame.data);
        let tag = IntermediateTag::Centroids { data_id, k };

        // Cache hit: labels already known for this (data, k) pair
        if let Some(cached_labels) = self.session.get::<Vec<i32>>(&tag) {
            self.frame.labels = Some((*cached_labels).clone());
            self.frame.n_clusters = Some(k);
            return self;
        }

        let data_f32: Vec<f32> = self.frame.data.iter().map(|&x| x as f32).collect();
        let result = self.kmeans.fit(&data_f32, self.frame.n, self.frame.d, k, max_iter)
            .expect("KMeansEngine::fit failed");

        let labels: Vec<i32> = result.labels.iter().map(|&l| l as i32).collect();
        self.session.register(tag, Arc::new(labels.clone()));
        self.frame.labels = Some(labels);
        self.frame.n_clusters = Some(k);
        self
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Fit a linear model on the current frame.
    ///
    /// `y` must have `n` entries (one per point). Uses `fit_session` which
    /// registers per-column `SufficientStatistics` in the session and reuses
    /// them if already present.
    ///
    /// Returns `(pipeline, model)` so the pipeline can continue after fitting.
    pub fn train_linear(
        mut self,
        y: &[f64],
    ) -> Result<(Self, LinearModel), Box<dyn std::error::Error>> {
        let model = linear::fit_session(
            &mut self.session,
            &self.frame.data,
            y,
            self.frame.n,
            self.frame.d,
        )?;
        Ok((self, model))
    }

    /// Fit a logistic regression model on the current frame.
    ///
    /// `y` must have `n` entries of binary labels (0.0 or 1.0).
    /// Uses gradient descent with `TiledEngine::DotProduct` for both forward
    /// and backward passes (the gradient duality).
    ///
    /// Returns `(pipeline, model)` so the pipeline can continue after fitting.
    pub fn train_logistic(
        self,
        y: &[f64],
        lr: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<(Self, LogisticModel), Box<dyn std::error::Error>> {
        let model = logistic::fit(
            &self.frame.data,
            y,
            self.frame.n,
            self.frame.d,
            lr,
            max_iter,
            tol,
        )?;
        Ok((self, model))
    }

    // -----------------------------------------------------------------------
    // Neighbors
    // -----------------------------------------------------------------------

    /// Compute k-nearest neighbors for the current frame.
    ///
    /// Uses the session for the n×n distance matrix. If a prior step (e.g.,
    /// `discover_clusters`) already computed it, the GPU kernel is skipped.
    ///
    /// After this step, `pipeline.frame().knn_result` contains the neighbor
    /// assignments.
    pub fn knn(mut self, k: usize) -> Self {
        let result = knn::knn_session(
            &mut self.session,
            &self.frame.data,
            self.frame.n,
            self.frame.d,
            k,
        ).expect("knn_session failed");

        self.frame.knn_result = Some(result);
        self
    }

    // -----------------------------------------------------------------------
    // Descriptive statistics
    // -----------------------------------------------------------------------

    /// Compute per-column descriptive statistics for the current frame.
    ///
    /// Returns `(pipeline, DescribeResult)` so the pipeline can continue.
    /// Uses Welford's algorithm (via `moments_ungrouped`) for numerical
    /// stability — safe even on data with extreme range.
    pub fn describe(self) -> (Self, DescribeResult) {
        let n = self.frame.n;
        let d = self.frame.d;
        let mut columns = Vec::with_capacity(d);

        for j in 0..d {
            let col: Vec<f64> = (0..n).map(|i| self.frame.data[i * d + j]).collect();
            let moments = crate::descriptive::moments_ungrouped(&col);
            let sorted = crate::descriptive::sorted_nan_free(&col);
            let median = crate::descriptive::median(&sorted);
            let (q1, _, q3) = crate::descriptive::quartiles(&sorted);
            let iqr = crate::descriptive::iqr(&sorted);

            let count = moments.count;
            let mean = if count > 0.0 { moments.sum / count } else { f64::NAN };
            let variance = if count > 1.0 { moments.m2 / (count - 1.0) } else { 0.0 };
            let std = variance.sqrt();
            let skewness = if count > 2.0 && moments.m2 > 0.0 {
                (count * moments.m3) / ((count - 1.0) * (count - 2.0) * (moments.m2 / count).powf(1.5))
            } else {
                0.0
            };
            let kurtosis = if count > 3.0 && moments.m2 > 0.0 {
                let excess = (count * (count + 1.0) * moments.m4)
                    / ((count - 1.0) * (count - 2.0) * (count - 3.0) * (moments.m2 / count).powi(2))
                    - 3.0 * (count - 1.0).powi(2) / ((count - 2.0) * (count - 3.0));
                excess
            } else {
                0.0
            };

            columns.push(ColumnDescribe {
                index: j,
                count: count as usize,
                mean,
                std,
                min: moments.min,
                q1,
                median,
                q3,
                max: moments.max,
                iqr,
                skewness,
                kurtosis,
            });
        }

        (self, DescribeResult { columns })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current frame state.
    pub fn frame(&self) -> &TamFrame {
        &self.frame
    }

    /// Number of entries currently in the session cache.
    pub fn session_len(&self) -> usize {
        self.session.len()
    }

    /// Consume the pipeline and extract the session (for inspection or reuse).
    pub fn into_session(self) -> TamSession {
        self.session
    }

    /// Consume the pipeline and extract the frame.
    pub fn into_frame(self) -> TamFrame {
        self.frame
    }
}

// ---------------------------------------------------------------------------
// Superposition clustering — .discover()
// ---------------------------------------------------------------------------

/// Specifies one clustering algorithm and its parameters for a `.discover()` run.
#[derive(Debug, Clone)]
pub enum ClusterSpec {
    /// DBSCAN with radius `epsilon` (in L2 distance, not L2Sq) and minimum
    /// neighborhood size `min_samples`.
    Dbscan { epsilon: f64, min_samples: usize },
    /// KMeans with `k` clusters and up to `max_iter` Lloyd iterations.
    Kmeans { k: usize, max_iter: usize },
    /// DBSCAN on a manifold mixture distance matrix.
    ///
    /// Uses the `ManifoldMixture` to compute a combined distance (session-cached),
    /// then runs DBSCAN at `epsilon_threshold` (in combined-distance units).
    Mixture {
        mix: ManifoldMixture,
        epsilon_threshold: f64,
        min_samples: usize,
    },
}

impl ClusterSpec {
    fn name(&self) -> String {
        match self {
            ClusterSpec::Dbscan { epsilon, min_samples } =>
                format!("dbscan(ε={:.3}, m={})", epsilon, min_samples),
            ClusterSpec::Kmeans { k, .. } =>
                format!("kmeans(k={})", k),
            ClusterSpec::Mixture { mix, epsilon_threshold, min_samples } =>
                format!("mixture(components={}, ε={:.3}, m={})",
                    mix.components.len(), epsilon_threshold, min_samples),
        }
    }
}

/// One cluster view from `.discover()`: one algorithm run + structural fingerprints.
#[derive(Debug, Clone)]
pub struct ClusterView {
    /// Human-readable spec description, e.g. `"dbscan(ε=1.5, m=3)"`.
    pub name: String,
    /// The spec that produced this view.
    pub spec: ClusterSpec,
    /// Per-point cluster label. `−1` = noise (DBSCAN only; never appears in KMeans).
    pub labels: Vec<i32>,
    /// Number of distinct non-noise clusters.
    pub n_clusters: usize,
    /// Fraction of points assigned to noise (`−1`). Always 0.0 for KMeans.
    pub noise_fraction: f64,
    /// Mean squared distance from each non-noise point to its cluster centroid,
    /// normalized by the overall data variance. Lower = more compact clusters.
    /// `f64::NAN` if all points are noise or there is only one non-noise point.
    pub compactness: f64,
}

/// Result of `.discover()` or `.discover_with()`: all cluster views simultaneously,
/// without collapse.
///
/// The structurally determined assignments are those that agree across views —
/// `view_agreement` and `modal_k` identify the most stable structure.
#[derive(Debug)]
pub struct DiscoveryResult {
    /// All cluster views, in the order they were computed.
    pub views: Vec<ClusterView>,
    /// Average pairwise Rand Index across all view pairs.
    /// 1.0 = all views fully agree. 0.5 = chance level (random labelings).
    /// Computed on a sample of up to 500 points for scalability.
    pub view_agreement: f64,
    /// Number of distinct `n_clusters` values across all views.
    pub n_distinct_k: usize,
    /// The most common `n_clusters` value (mode). Points to the most-supported structure.
    pub modal_k: usize,
}

impl DiscoveryResult {
    fn build(views: Vec<ClusterView>, n: usize) -> Self {
        let view_agreement = pairwise_rand_index(&views, n);

        // Mode of n_clusters
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for v in &views { *counts.entry(v.n_clusters).or_insert(0) += 1; }
        let modal_k = counts.iter().max_by_key(|(_, c)| *c).map(|(k, _)| *k).unwrap_or(0);
        let n_distinct_k = counts.len();

        DiscoveryResult { views, view_agreement, n_distinct_k, modal_k }
    }
}

/// Compute pairwise Rand Index averaged over all view pairs.
/// Samples up to `max_sample` points for O(1) scalability in n.
fn pairwise_rand_index(views: &[ClusterView], n: usize) -> f64 {
    if views.len() < 2 { return 1.0; }
    let sample_n = n.min(500);
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..views.len() {
        for j in i+1..views.len() {
            sum += rand_index_sampled(&views[i].labels, &views[j].labels, sample_n);
            count += 1;
        }
    }
    if count == 0 { 1.0 } else { sum / count as f64 }
}

/// Rand Index between two labelings, computed on the first `sample_n` points.
fn rand_index_sampled(a: &[i32], b: &[i32], sample_n: usize) -> f64 {
    let n = a.len().min(sample_n);
    let mut agree = 0u64;
    let mut total = 0u64;
    for i in 0..n {
        for j in i+1..n {
            let same_a = a[i] == a[j];
            let same_b = b[i] == b[j];
            if same_a == same_b { agree += 1; }
            total += 1;
        }
    }
    if total == 0 { 1.0 } else { agree as f64 / total as f64 }
}

/// Compactness: mean squared distance from non-noise points to their cluster centroid,
/// normalized by overall data variance.
fn compute_compactness(data: &[f64], n: usize, d: usize, labels: &[i32]) -> f64 {
    // Cluster centroids
    let max_label = labels.iter().filter(|&&l| l >= 0).max().copied().unwrap_or(-1);
    if max_label < 0 { return f64::NAN; }
    let k = (max_label + 1) as usize;

    let mut centroids = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let lbl = labels[i];
        if lbl < 0 { continue; }
        let cl = lbl as usize;
        counts[cl] += 1;
        for j in 0..d {
            centroids[cl * d + j] += data[i * d + j];
        }
    }
    for cl in 0..k {
        if counts[cl] > 0 {
            for j in 0..d { centroids[cl * d + j] /= counts[cl] as f64; }
        }
    }

    // Mean squared distance to centroid
    let mut intra_sum = 0.0f64;
    let mut intra_count = 0usize;
    for i in 0..n {
        let lbl = labels[i];
        if lbl < 0 { continue; }
        let cl = lbl as usize;
        let dist2: f64 = (0..d).map(|j| (data[i*d+j] - centroids[cl*d+j]).powi(2)).sum();
        intra_sum += dist2;
        intra_count += 1;
    }
    if intra_count == 0 { return f64::NAN; }
    let intra_mean = intra_sum / intra_count as f64;

    // Overall data variance
    let overall_mean: Vec<f64> = (0..d).map(|j| {
        data.iter().skip(j).step_by(d).sum::<f64>() / n as f64
    }).collect();
    let total_var: f64 = (0..n).map(|i| {
        (0..d).map(|j| (data[i*d+j] - overall_mean[j]).powi(2)).sum::<f64>()
    }).sum::<f64>() / n as f64;

    if total_var < 1e-15 { return f64::NAN; }
    intra_mean / total_var
}

/// Percentile of a sorted slice (linear interpolation, p in [0,1]).
fn sorted_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

impl TamPipeline {
    // -----------------------------------------------------------------------
    // Superposition clustering
    // -----------------------------------------------------------------------

    /// Run multiple clustering algorithms simultaneously without collapsing.
    ///
    /// Auto-selects epsilon values for DBSCAN from the k-NN distance distribution
    /// (percentile sweep: p15, p50, p85 of 4-NN distances) plus KMeans with
    /// k = 2..=min(5, √n).
    ///
    /// The session-cached distance matrix is reused across all DBSCAN runs —
    /// no redundant GPU computation.
    ///
    /// Returns the pipeline (frame.labels set to the modal-k DBSCAN or KMeans view)
    /// and the full `DiscoveryResult` with all views and agreement metrics.
    pub fn discover(mut self) -> (Self, DiscoveryResult) {
        let n = self.frame.n;
        let d = self.frame.d;

        // Compute k-NN for epsilon selection.
        // k = min(4, n/3) ensures we stay within clusters for typical data.
        // For tiny n (< 9), k=2 avoids cross-cluster contamination.
        let k_nn = 4.min(n / 3).max(2).min(n.saturating_sub(1));
        let knn_result = knn::knn_session(
            &mut self.session,
            &self.frame.data,
            n,
            d,
            k_nn,
        ).expect("knn_session failed in discover");

        // Collect kth-NN distances (L2Sq → sqrt → L2) for epsilon selection.
        // We take the FURTHEST of the k_nn neighbors per point (the kth distance),
        // which represents the local neighborhood radius.
        let mut nn_dists: Vec<f64> = knn_result.neighbors.iter()
            .filter_map(|row| row.last().map(|(_, d2)| d2.sqrt()))
            .collect();
        nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let eps_tight  = sorted_percentile(&nn_dists, 0.15).max(1e-12);
        let eps_mid    = sorted_percentile(&nn_dists, 0.50).max(1e-12);
        let eps_loose  = sorted_percentile(&nn_dists, 0.85).max(1e-12);

        let k_max = 5.min(((n as f64).sqrt() as usize).max(2));

        // min_samples=2: allows core points in smallest clusters (3+ points
        // would require 3 mutual neighbors, which fails for 2-3 point clusters).
        let mut specs = vec![
            ClusterSpec::Dbscan { epsilon: eps_tight, min_samples: 2 },
            ClusterSpec::Dbscan { epsilon: eps_mid,   min_samples: 2 },
            ClusterSpec::Dbscan { epsilon: eps_loose, min_samples: 2 },
        ];
        for k in 2..=k_max {
            specs.push(ClusterSpec::Kmeans { k, max_iter: 100 });
        }

        self.frame.knn_result = Some(knn_result);
        self.run_discover(specs)
    }

    /// Run a caller-specified set of clustering specs simultaneously.
    ///
    /// Like `discover()` but the epsilon/k sweep is under caller control.
    /// The session distance matrix is shared across all DBSCAN specs.
    pub fn discover_with(mut self, specs: Vec<ClusterSpec>) -> (Self, DiscoveryResult) {
        self.run_discover(specs)
    }

    fn run_discover(mut self, specs: Vec<ClusterSpec>) -> (Self, DiscoveryResult) {
        let n = self.frame.n;
        let d = self.frame.d;
        let mut views = Vec::with_capacity(specs.len());

        for spec in specs {
            let (labels, n_clusters) = match &spec {
                ClusterSpec::Dbscan { epsilon, min_samples } => {
                    // epsilon is L2 distance; discover_clusters_session takes L2Sq epsilon
                    let eps_sq = epsilon * epsilon;
                    let result = self.clustering.discover_clusters_session(
                        &mut self.session,
                        &self.frame.data,
                        n,
                        d,
                        eps_sq,
                        *min_samples,
                        &DistanceOp,
                    ).expect("discover_clusters_session failed");
                    (result.labels, result.n_clusters)
                }
                ClusterSpec::Kmeans { k, max_iter } => {
                    let data_f32: Vec<f32> = self.frame.data.iter().map(|&x| x as f32).collect();
                    let result = self.kmeans.fit(&data_f32, n, d, *k, *max_iter)
                        .expect("KMeansEngine::fit failed");
                    let labels: Vec<i32> = result.labels.iter().map(|&l| l as i32).collect();
                    (labels, *k)
                }
                ClusterSpec::Mixture { mix, epsilon_threshold, min_samples } => {
                    // Reuse discover_clusters_mixture logic: compute manifold distances
                    // (session-cached), combine, then DBSCAN on combined matrix.
                    // We temporarily consume self then re-assign — save the frame data.
                    let data_id = DataId::from_f64(&self.frame.data);
                    let data_t: Vec<f64> = {
                        let data = &self.frame.data;
                        (0..d).flat_map(|k| (0..n).map(move |i| data[i * d + k])).collect()
                    };
                    let mut matrices: Vec<Arc<DistanceMatrix>> = Vec::new();
                    for (manifold, _) in &mix.components {
                        let tag = IntermediateTag::ManifoldDistanceMatrix {
                            manifold_name: manifold.name(),
                            data_id,
                        };
                        let dm = if let Some(cached) = self.session.get::<DistanceMatrix>(&tag) {
                            cached
                        } else {
                            let dist_data = self.tiled.run(
                                &ManifoldDistanceOp::new(manifold.clone()),
                                &self.frame.data,
                                &data_t,
                                n, n, d,
                            ).expect("ManifoldDistanceOp::run failed");
                            let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, dist_data));
                            self.session.register(tag, Arc::clone(&dm));
                            dm
                        };
                        matrices.push(dm);
                    }
                    let combined = mix.combine(&matrices);
                    let result = ClusteringEngine::discover_clusters_from_combined(
                        &combined, n, *epsilon_threshold, *min_samples,
                    );
                    (result.labels, result.n_clusters)
                }
            };

            let noise_count = labels.iter().filter(|&&l| l < 0).count();
            let noise_fraction = noise_count as f64 / n as f64;
            let compactness = compute_compactness(&self.frame.data, n, d, &labels);

            views.push(ClusterView {
                name: spec.name(),
                spec,
                labels,
                n_clusters,
                noise_fraction,
                compactness,
            });
        }

        let result = DiscoveryResult::build(views, n);

        // Set frame to the first view whose n_clusters == modal_k
        // (prefer DBSCAN over KMeans when both have the same k)
        let modal = result.modal_k;
        if let Some(view) = result.views.iter().find(|v| v.n_clusters == modal) {
            self.frame.labels = Some(view.labels.clone());
            self.frame.n_clusters = Some(view.n_clusters);
        }

        (self, result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_clusters() -> Vec<f64> {
        // 6 points in R², two clear clusters
        // Cluster A: (1,1), (1,2), (2,1)
        // Cluster B: (10,10), (10,11), (11,10)
        vec![
            1.0, 1.0,
            1.0, 2.0,
            2.0, 1.0,
            10.0, 10.0,
            10.0, 11.0,
            11.0, 10.0,
        ]
    }

    #[test]
    fn pipeline_from_slice_shape() {
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2);
        assert_eq!(p.frame().n, 6);
        assert_eq!(p.frame().d, 2);
        assert_eq!(p.frame().data.len(), 12);
    }

    #[test]
    fn pipeline_normalize_changes_data() {
        let data = make_two_clusters();
        let original_mean_col0: f64 = data.iter().step_by(2).sum::<f64>() / 6.0;

        let p = TamPipeline::from_slice(data, 6, 2).normalize();

        // After normalization, column 0 mean should be ~0
        let col0_mean: f64 = p.frame().data.iter().step_by(2).sum::<f64>() / 6.0;
        assert!(col0_mean.abs() < 1e-10, "normalized col0 mean = {col0_mean}");

        // Original mean was not zero
        assert!((original_mean_col0 - col0_mean).abs() > 1.0);
    }

    #[test]
    fn pipeline_discover_clusters_finds_two() {
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2)
            .discover_clusters(3.0, 1);

        let frame = p.frame();
        assert_eq!(frame.n_clusters, Some(2), "expected 2 clusters");
        assert!(frame.labels.is_some());
        let labels = frame.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 6);
        // All points should be assigned (no noise with epsilon=3)
        assert!(labels.iter().all(|&l| l >= 0), "unexpected noise points");
    }

    #[test]
    fn pipeline_session_caches_distance_matrix() {
        let data = make_two_clusters();
        let mut p = TamPipeline::from_slice(data, 6, 2);
        assert_eq!(p.session_len(), 0);

        p = p.discover_clusters(3.0, 1);
        // Session should now hold the distance matrix
        assert_eq!(p.session_len(), 1, "distance matrix should be in session");

        // Second clustering call on same data — session hit, still 1 entry
        p = p.discover_clusters(5.0, 2);
        assert_eq!(p.session_len(), 1, "session size should not grow on cache hit");
    }

    #[test]
    fn pipeline_train_linear_basic() {
        // Perfect linear relationship: y = 2*x0 + 3*x1
        let x: Vec<f64> = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            2.0, 0.0,
            0.0, 2.0,
        ];
        let y: Vec<f64> = vec![2.0, 3.0, 5.0, 4.0, 6.0];

        let (p, model) = TamPipeline::from_slice(x, 5, 2)
            .train_linear(&y)
            .unwrap();

        assert!(model.r_squared > 0.99, "R² should be near 1 for perfect linear data, got {}", model.r_squared);
        // Session should hold the SufficientStatistics
        assert!(p.session_len() >= 1);
    }

    #[test]
    fn pipeline_normalize_then_train() {
        // Same perfect linear data, normalized first
        let x: Vec<f64> = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            2.0, 0.0,
            0.0, 2.0,
        ];
        let y: Vec<f64> = vec![2.0, 3.0, 5.0, 4.0, 6.0];

        let (_, model) = TamPipeline::from_slice(x, 5, 2)
            .normalize()
            .train_linear(&y)
            .unwrap();

        // Should still fit well after normalization
        assert!(model.r_squared > 0.99, "R² after normalize = {}", model.r_squared);
    }

    #[test]
    fn pipeline_full_chain() {
        // Chain: normalize → cluster → train
        let x: Vec<f64> = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            2.0, 0.0,
            0.0, 2.0,
        ];
        let y: Vec<f64> = vec![2.0, 3.0, 5.0, 4.0, 6.0];

        let (pipeline, model) = TamPipeline::from_slice(x, 5, 2)
            .normalize()
            .discover_clusters(2.0, 1)
            .train_linear(&y)
            .unwrap();

        assert!(model.r_squared > 0.9, "R² = {}", model.r_squared);
        assert!(pipeline.frame().labels.is_some());
        // Session holds at least: distance matrix + column stats
        assert!(pipeline.session_len() >= 2, "session_len = {}", pipeline.session_len());
    }

    #[test]
    fn pipeline_train_logistic_basic() {
        // Binary classification: two clusters
        let n = 100;
        let d = 2;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let label = if i < n / 2 { 0.0 } else { 1.0 };
            let cx = if label == 0.0 { -2.0 } else { 2.0 };
            let offset = (i % (n / 2)) as f64 / (n as f64 / 2.0) - 0.5;
            x[i * d] = cx + offset * 0.5;
            x[i * d + 1] = cx + offset * 0.3;
            y[i] = label;
        }

        let (_, model) = TamPipeline::from_slice(x, n, d)
            .train_logistic(&y, 1.0, 500, 1e-8)
            .unwrap();

        assert!(model.accuracy > 0.9, "accuracy={:.1}%", model.accuracy * 100.0);
    }

    #[test]
    fn pipeline_knn_basic() {
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2).knn(2);

        let knn = p.frame().knn_result.as_ref().expect("expected knn_result");
        assert_eq!(knn.n, 6);
        assert_eq!(knn.k, 2);
        // Session should hold the distance matrix
        assert_eq!(p.session_len(), 1);
    }

    #[test]
    fn pipeline_cluster_then_knn_shares_distance() {
        // Cross-algorithm sharing: DBSCAN → KNN through pipeline session
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2)
            .discover_clusters(3.0, 1)
            .knn(2);

        assert_eq!(p.frame().n_clusters, Some(2));
        assert!(p.frame().knn_result.is_some());
        // Both ops share the same distance matrix — session holds exactly 1
        assert_eq!(p.session_len(), 1, "distance matrix shared, not duplicated");
    }

    #[test]
    fn pipeline_kmeans_finds_two_clusters() {
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2)
            .kmeans(2, 100);

        let frame = p.frame();
        assert_eq!(frame.n_clusters, Some(2));
        let labels = frame.labels.as_ref().expect("expected labels");
        assert_eq!(labels.len(), 6);
        // All points assigned (no noise in KMeans)
        assert!(labels.iter().all(|&l| l >= 0));
        // Two distinct label values
        let unique: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2, "expected exactly 2 label values");
    }

    #[test]
    fn pipeline_kmeans_session_caches_result() {
        let data = make_two_clusters();
        let mut p = TamPipeline::from_slice(data, 6, 2);
        assert_eq!(p.session_len(), 0);

        p = p.kmeans(2, 100);
        // Session should hold the centroids
        assert_eq!(p.session_len(), 1, "centroids should be in session after kmeans");

        let labels_first: Vec<i32> = p.frame().labels.clone().unwrap();

        // Second call on same data/k: session hit, no recompute
        p = p.kmeans(2, 100);
        assert_eq!(p.session_len(), 1, "session should not grow on cache hit");
        let labels_second: Vec<i32> = p.frame().labels.clone().unwrap();
        assert_eq!(labels_first, labels_second, "cached result must be identical");
    }

    #[test]
    fn pipeline_kmeans_different_k_misses_cache() {
        let data = make_two_clusters();
        let p = TamPipeline::from_slice(data, 6, 2)
            .kmeans(2, 100)
            .kmeans(3, 100);

        // k=2 and k=3 produce different tags → both in session
        assert_eq!(p.session_len(), 2, "k=2 and k=3 are different intermediates");
        assert_eq!(p.frame().n_clusters, Some(3));
    }

    // ── ManifoldMixture pipeline tests ───────────────────────────────────────

    #[test]
    fn pipeline_mixture_euclidean_only_matches_dbscan() {
        // A single-manifold mixture (Euclidean, w=1.0) should find the same clusters as plain DBSCAN
        use crate::manifold::{Manifold, ManifoldMixture};

        let data = make_two_clusters();
        let n = 6;
        let d = 2;

        // Plain DBSCAN (L2Sq, epsilon=9.0 = 3²)
        let p_plain = TamPipeline::from_slice(data.clone(), n, d)
            .discover_clusters(3.0, 1);

        // Mixture with Euclidean only (epsilon_threshold=9.0 since L2Sq)
        let mix = ManifoldMixture::single(Manifold::Euclidean);
        let p_mix = TamPipeline::from_slice(data, n, d)
            .discover_clusters_mixture(&mix, 9.0, 1);

        assert_eq!(
            p_plain.frame().n_clusters,
            p_mix.frame().n_clusters,
            "single-manifold mixture should match plain DBSCAN cluster count"
        );
    }

    #[test]
    fn pipeline_mixture_caches_component_matrices() {
        // Each component distance matrix should be in session after clustering
        use crate::manifold::{Manifold, ManifoldMixture};

        let data = make_two_clusters();
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::sphere(1.0),
        ]);

        let p = TamPipeline::from_slice(data, 6, 2)
            .discover_clusters_mixture(&mix, 2.0, 1);

        // Session should hold: 2 component matrices + 1 combined = 3 entries
        assert_eq!(p.session_len(), 3,
            "2 component + 1 combined = 3 session entries, got {}", p.session_len());
        assert!(p.frame().labels.is_some());
    }

    #[test]
    fn pipeline_mixture_second_call_hits_cache() {
        // Second call with same mix+data should hit all 3 session entries, not recompute
        use crate::manifold::{Manifold, ManifoldMixture};

        let data = make_two_clusters();
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::sphere(1.0),
        ]);

        let p = TamPipeline::from_slice(data, 6, 2)
            .discover_clusters_mixture(&mix, 2.0, 1)
            .discover_clusters_mixture(&mix, 3.0, 1); // different epsilon, same matrices

        // Session should still be 3 — same matrices reused, combined matrix overwritten
        // Actually: mix_id same, but combined matrix already cached — second call skips re-register
        assert_eq!(p.session_len(), 3, "no new session entries on second call");
    }

    // -----------------------------------------------------------------------
    // .discover() / .discover_with() tests
    // -----------------------------------------------------------------------

    #[test]
    fn discover_finds_two_clusters() {
        // Two well-separated clusters in R².
        let data = make_two_clusters(); // 6 points, 2 clear clusters
        let (p, result) = TamPipeline::from_slice(data, 6, 2).discover();

        assert!(!result.views.is_empty(), "discover should produce at least one view");

        // The modal k should be 2 (both DBSCAN and KMeans should agree)
        assert_eq!(result.modal_k, 2, "modal_k should be 2 for two-cluster data, got {}", result.modal_k);

        // Frame should be set to modal view
        assert_eq!(p.frame().n_clusters, Some(2));
        assert_eq!(p.frame().labels.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn discover_views_have_fingerprints() {
        let data = make_two_clusters();
        let (_, result) = TamPipeline::from_slice(data, 6, 2).discover();

        for view in &result.views {
            assert!(!view.name.is_empty());
            assert_eq!(view.labels.len(), 6);
            // noise_fraction ∈ [0,1]
            assert!(view.noise_fraction >= 0.0 && view.noise_fraction <= 1.0,
                "noise_fraction out of range: {}", view.noise_fraction);
        }

        // view_agreement: should be high (both methods find same structure)
        assert!(result.view_agreement >= 0.5,
            "view_agreement should be ≥ 0.5 for consistent data, got {}", result.view_agreement);
    }

    #[test]
    fn discover_session_shared_across_specs() {
        // DBSCAN and KMeans views share the distance matrix in session.
        // All DBSCAN specs after the first should hit the session cache.
        let data = make_two_clusters();
        let p0 = TamPipeline::from_slice(data.clone(), 6, 2);
        let session_before = p0.session_len();

        let (p_after, result) = TamPipeline::from_slice(data, 6, 2).discover();

        // discover() produces multiple views — session should have distance matrix + knn
        assert!(p_after.session_len() > session_before,
            "session should grow after discover");
        assert!(result.views.len() >= 3,
            "should have at least 3 views (3 DBSCAN + KMeans), got {}", result.views.len());
    }

    #[test]
    fn discover_with_user_specs() {
        let data = make_two_clusters();
        let specs = vec![
            ClusterSpec::Dbscan { epsilon: 2.0, min_samples: 1 },
            ClusterSpec::Kmeans { k: 2, max_iter: 50 },
            ClusterSpec::Kmeans { k: 3, max_iter: 50 },
        ];
        let (p, result) = TamPipeline::from_slice(data, 6, 2).discover_with(specs);

        assert_eq!(result.views.len(), 3);
        assert_eq!(result.views[0].name, "dbscan(ε=2.000, m=1)");
        assert_eq!(result.views[1].name, "kmeans(k=2)");
        assert_eq!(result.views[2].name, "kmeans(k=3)");

        // DBSCAN with epsilon=2.0 on the two-cluster data should find 2 clusters
        assert_eq!(result.views[0].n_clusters, 2,
            "DBSCAN ε=2 should find 2 clusters");

        // KMeans k=2 should find 2 clusters with 0 noise
        assert_eq!(result.views[1].n_clusters, 2);
        assert_eq!(result.views[1].noise_fraction, 0.0);

        // Frame set to modal view
        assert_eq!(p.frame().n_clusters, Some(2));
    }

    #[test]
    fn discover_compactness_well_separated() {
        // Well-separated clusters → low compactness (points close to centroids)
        let data = make_two_clusters();
        let specs = vec![ClusterSpec::Kmeans { k: 2, max_iter: 100 }];
        let (_, result) = TamPipeline::from_slice(data, 6, 2).discover_with(specs);

        let view = &result.views[0];
        assert!(!view.compactness.is_nan(), "compactness should not be NaN for 2 clean clusters");
        assert!(view.compactness < 0.5,
            "well-separated clusters should have low compactness, got {}", view.compactness);
    }

    #[test]
    fn discover_n_distinct_k() {
        let data = make_two_clusters();
        let specs = vec![
            ClusterSpec::Kmeans { k: 2, max_iter: 50 },
            ClusterSpec::Kmeans { k: 2, max_iter: 50 }, // duplicate k
            ClusterSpec::Kmeans { k: 3, max_iter: 50 }, // different k
        ];
        let (_, result) = TamPipeline::from_slice(data, 6, 2).discover_with(specs);

        assert_eq!(result.n_distinct_k, 2, "should have 2 distinct k values (2 and 3)");
        assert_eq!(result.modal_k, 2, "modal_k should be 2 (appears twice)");
    }
}
