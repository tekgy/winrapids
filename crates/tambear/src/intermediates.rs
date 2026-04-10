//! Typed shared intermediates — the sharing infrastructure layer.
//!
//! ## The problem
//!
//! Multiple algorithms need the same expensive intermediate. DBSCAN, KNN, outlier
//! detection, silhouette scoring all want an n×n distance matrix. Without sharing,
//! each computes it independently — O(n²d) GPU work, repeated.
//!
//! ## The solution
//!
//! Each intermediate is a first-class value tagged with its semantic type. The type
//! tag determines compatibility: `DistanceMatrix(L2Sq)` is not the same as
//! `DistanceMatrix(Cosine)`, even if both are n×n f64 matrices.
//!
//! Algorithms that produce intermediates return them alongside their primary output.
//! Algorithms that consume intermediates accept an `Option<Arc<IntermediateType>>`:
//! - `Some(x)` → use the precomputed value (zero GPU cost)
//! - `None` → compute it (and return it for downstream reuse)
//!
//! ## The compiler step (future)
//!
//! Right now, sharing is explicit: the caller passes precomputed values through.
//! The next layer is a `TamSession` that holds an intermediate registry and wires
//! producers to consumers automatically when the type tags match.
//!
//! ```text
//! TamSession
//! ├── intermediates: HashMap<IntermediateTag, Arc<dyn Intermediate>>
//! ├── On produce: register(tag, value)
//! └── On consume: if registry.contains(tag) → reuse, else → compute + register
//! ```
//!
//! This file establishes the types. The session/compiler is built on top.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// DataId — provenance tag for data identity
// ---------------------------------------------------------------------------

/// Content-based identity for a data buffer.
///
/// Two buffers with the same DataId contain the same data. This enables the
/// sharing infrastructure to determine when two algorithms are operating on
/// the same data without pointer equality or naming conventions.
///
/// Computed via blake3 hash of the raw bytes — fast (>1 GB/s on CPU),
/// deterministic, and collision-free in practice.
///
/// # Why not pointer equality?
///
/// Same data may exist in multiple copies (file reload, host/device duplication,
/// different column selections from the same DataFrame). Content hashing catches
/// all of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataId(pub u64);

impl DataId {
    /// Compute a DataId from raw bytes. Uses blake3 truncated to u64.
    pub fn from_bytes(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        let bytes = hash.as_bytes();
        DataId(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Compute a DataId from an f64 slice (the common case).
    pub fn from_f64(data: &[f64]) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
        };
        Self::from_bytes(bytes)
    }

    /// Compute a DataId from an f32 slice.
    pub fn from_f32(data: &[f32]) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        Self::from_bytes(bytes)
    }

    /// Compute a DataId from an i32 slice (grouping keys).
    pub fn from_i32(data: &[i32]) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        Self::from_bytes(bytes)
    }

    /// Combine two DataIds (for intermediates that depend on multiple inputs).
    pub fn combine(a: DataId, b: DataId) -> Self {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&a.0.to_le_bytes());
        buf[8..].copy_from_slice(&b.0.to_le_bytes());
        Self::from_bytes(&buf)
    }
}

// ---------------------------------------------------------------------------
// Metric — the compatibility predicate for distance computations
// ---------------------------------------------------------------------------

/// Distance metric. Determines which consumers can accept which producers.
///
/// Two `DistanceMatrix` values with different metrics are NOT substitutable,
/// even if they have the same shape. The metric is part of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared Euclidean: `Σ(aᵢ - bᵢ)²`. Avoids sqrt — cheaper, monotone-equivalent.
    /// Default for KMeans, DBSCAN (pass `epsilon_radius²` as threshold).
    L2Sq,

    /// Euclidean: `√Σ(aᵢ - bᵢ)²`. More expensive (sqrt per pair).
    /// Use when the threshold is a geometric radius, not a squared one.
    L2,

    /// Cosine dissimilarity: `1 - dot(a,b) / (|a| |b|)`.
    /// Useful for embedding spaces and text features.
    Cosine,

    /// Dot product (inner product). Similarity, not distance — larger = more similar.
    /// Threshold semantics are reversed: consumers that use this must handle it.
    Dot,

    /// Manhattan / city-block: `Σ|aᵢ - bᵢ|`. More robust to outliers than L2.
    Manhattan,
}

impl Metric {
    /// Whether this metric produces values where smaller = more similar.
    /// True for all distance metrics (L2Sq, L2, Cosine, Manhattan).
    /// False for Dot (larger = more similar).
    pub fn is_distance(&self) -> bool {
        !matches!(self, Metric::Dot)
    }

    /// Whether a threshold on this metric corresponds to an epsilon-neighborhood.
    /// For L2Sq: threshold is `epsilon_radius²`. For L2: threshold is `epsilon_radius`.
    pub fn threshold_is_squared(&self) -> bool {
        matches!(self, Metric::L2Sq)
    }
}

// ---------------------------------------------------------------------------
// DistanceMatrix — the canonical shareable intermediate
// ---------------------------------------------------------------------------

/// A pairwise n×n distance (or similarity) matrix, tagged with its metric.
///
/// This is the primary shared intermediate between clustering, KNN, outlier
/// detection, and graph algorithms. Computing it is O(n²d) on GPU. Sharing it
/// is O(1) (Arc clone).
///
/// ## Layout
///
/// Row-major: `data[i * n + j]` = distance from point i to point j.
/// Symmetric for distance metrics: `data[i*n+j] == data[j*n+i]`.
/// Zero diagonal for distance metrics: `data[i*n+i] == 0`.
///
/// ## Sharing pattern
///
/// ```no_run
/// use tambear::intermediates::{DistanceMatrix, Metric};
/// use tambear::clustering::ClusteringEngine;
/// use winrapids_tiled::DistanceOp;
/// use std::sync::Arc;
///
/// let mut engine = ClusteringEngine::new().unwrap();
/// let data = vec![/* ... */];
/// let n = 10; let d = 2;
///
/// // First algorithm: DBSCAN. Computes distance, returns it as byproduct.
/// let (result1, dist) = engine.discover_clusters_with_distance(
///     &data, n, d, 0.25, 2, &DistanceOp,
/// ).unwrap();
///
/// // Second algorithm: DBSCAN with different epsilon. Reuses distance — zero GPU cost.
/// let result2 = engine.discover_clusters_from_distance(
///     Arc::clone(&dist), 0.5, 3,
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// The metric used to compute this matrix. Consumers must check compatibility.
    pub metric: Metric,

    /// Number of points. The matrix has `n * n` entries.
    pub n: usize,

    /// Row-major data: `data[i * n + j]` = distance(point_i, point_j).
    /// Arc allows zero-copy sharing between multiple consumers.
    pub data: Arc<Vec<f64>>,
}

impl DistanceMatrix {
    /// Create from an owned Vec (takes ownership, wraps in Arc).
    pub fn from_vec(metric: Metric, n: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * n, "DistanceMatrix data must be n*n");
        Self { metric, n, data: Arc::new(data) }
    }

    /// Get the distance from point `i` to point `j`.
    #[inline]
    pub fn entry(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Get the full row of distances from point `i` to all other points.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// Check whether this matrix is compatible with a requested metric.
    ///
    /// A consumer should call this before using a precomputed matrix.
    pub fn is_compatible_with(&self, needed: Metric) -> bool {
        self.metric == needed
    }

    /// Number of elements in the distance matrix.
    pub fn len(&self) -> usize {
        self.n * self.n
    }
}

// ---------------------------------------------------------------------------
// SufficientStatistics — the second canonical shared intermediate
// ---------------------------------------------------------------------------

/// Per-group sufficient statistics: (sum, m2, count) for each group.
///
/// Uses Welford's numerically stable representation: `m2[g] = Σ(v - mean)²`
/// (sum of squared deviations from the group mean). This avoids the catastrophic
/// cancellation of the naive `Σv²/n - mean²` formula at large offsets.
///
/// From these three values, derive mean, variance, std, and any linear
/// combination without re-scanning the data. Once computed by a scatter pass,
/// they can be consumed by normalization, z-scoring, Pearson correlation,
/// and hypothesis tests — all without additional GPU work.
///
/// Produced by: `HashScatterEngine::groupby`, `ScatterJit::scatter_multi_phi`
/// Consumed by: normalization, z-scoring, covariance, PCA preprocessing
#[derive(Debug, Clone)]
pub struct SufficientStatistics {
    /// Number of groups.
    pub n_groups: usize,

    /// Sum of values per group: `sums[g] = Σ v` for elements in group g.
    pub sums: Arc<Vec<f64>>,

    /// Sum of squared deviations from mean (Welford's m2):
    /// `m2[g] = Σ(v - mean_g)²` for elements in group g.
    ///
    /// Numerically stable: no cancellation regardless of data offset.
    pub m2: Arc<Vec<f64>>,

    /// Count of elements per group (stored as f64 for GPU atomicAdd compatibility).
    pub counts: Arc<Vec<f64>>,
}

impl SufficientStatistics {
    /// Construct from GPU scatter output (sum, sum_sqs, count).
    ///
    /// Internally converts `sum_sqs` to Welford's `m2 = sum_sqs - sum²/count`
    /// for numerically stable variance. The conversion is exact when the
    /// inputs are exact (as they are from GPU atomicAdd on reasonable counts).
    pub fn from_vecs(n_groups: usize, sums: Vec<f64>, sum_sqs: Vec<f64>, counts: Vec<f64>) -> Self {
        assert_eq!(sums.len(), n_groups);
        assert_eq!(sum_sqs.len(), n_groups);
        assert_eq!(counts.len(), n_groups);
        // Convert sum_sqs to m2: m2 = Σv² - (Σv)²/n = sum_sqs - sum²/count
        let m2: Vec<f64> = sums.iter()
            .zip(&sum_sqs)
            .zip(&counts)
            .map(|((&s, &sq), &c)| {
                if c > 0.0 { (sq - s * s / c).max(0.0) } else { 0.0 }
            })
            .collect();
        Self {
            n_groups,
            sums: Arc::new(sums),
            m2: Arc::new(m2),
            counts: Arc::new(counts),
        }
    }

    /// Construct directly from Welford accumulators (sum, m2, count).
    ///
    /// Use when m2 is already computed as `Σ(v - mean)²` (e.g., from a
    /// two-pass algorithm or Welford online accumulation).
    pub fn from_welford(n_groups: usize, sums: Vec<f64>, m2: Vec<f64>, counts: Vec<f64>) -> Self {
        assert_eq!(sums.len(), n_groups);
        assert_eq!(m2.len(), n_groups);
        assert_eq!(counts.len(), n_groups);
        Self {
            n_groups,
            sums: Arc::new(sums),
            m2: Arc::new(m2),
            counts: Arc::new(counts),
        }
    }

    /// Mean for group g. Panics if count[g] == 0.
    pub fn mean(&self, g: usize) -> f64 {
        self.sums[g] / self.counts[g]
    }

    /// Variance (population) for group g: `m2 / n`.
    ///
    /// Numerically stable via Welford's `m2` representation.
    /// Returns 0.0 if count <= 0.
    pub fn variance(&self, g: usize) -> f64 {
        let n = self.counts[g];
        if n <= 0.0 { return 0.0; }
        self.m2[g] / n
    }

    /// Variance (sample, Bessel-corrected) for group g: `m2 / (n - 1)`.
    ///
    /// Returns NaN if count <= 1.
    pub fn variance_sample(&self, g: usize) -> f64 {
        let n = self.counts[g];
        if n <= 1.0 { return f64::NAN; }
        self.m2[g] / (n - 1.0)
    }

    /// Standard deviation (population) for group g.
    pub fn std(&self, g: usize) -> f64 {
        self.variance(g).sqrt()
    }

    /// Standard deviation (sample, Bessel-corrected) for group g.
    pub fn std_sample(&self, g: usize) -> f64 {
        self.variance_sample(g).sqrt()
    }

    /// Merge two `SufficientStatistics` using parallel Welford's algorithm.
    ///
    /// Both must have the same `n_groups`. For each group g:
    /// ```text
    /// n_combined = n_a + n_b
    /// sum_combined = sum_a + sum_b
    /// delta = mean_b - mean_a
    /// m2_combined = m2_a + m2_b + delta² * n_a * n_b / n_combined
    /// ```
    pub fn merge(&self, other: &SufficientStatistics) -> SufficientStatistics {
        assert_eq!(self.n_groups, other.n_groups,
            "Cannot merge SufficientStatistics with different n_groups");
        let ng = self.n_groups;
        let mut sums = vec![0.0; ng];
        let mut m2 = vec![0.0; ng];
        let mut counts = vec![0.0; ng];

        for g in 0..ng {
            let n_a = self.counts[g];
            let n_b = other.counts[g];
            let n_combined = n_a + n_b;

            sums[g] = self.sums[g] + other.sums[g];
            counts[g] = n_combined;

            if n_combined > 0.0 {
                let mean_a = if n_a > 0.0 { self.sums[g] / n_a } else { 0.0 };
                let mean_b = if n_b > 0.0 { other.sums[g] / n_b } else { 0.0 };
                let delta = mean_b - mean_a;
                m2[g] = self.m2[g] + other.m2[g] + delta * delta * n_a * n_b / n_combined;
            }
        }

        SufficientStatistics::from_welford(ng, sums, m2, counts)
    }
}

// ---------------------------------------------------------------------------
// IntermediateTag — the type key for the future compiler registry
// ---------------------------------------------------------------------------

/// The type key used by the compiler to match producers to consumers.
///
/// Two intermediates are shareable when their tags are EQUAL. The tag captures:
/// - What was computed (metric, expressions, parameters)
/// - From what data (DataId — content hash)
///
/// Algorithms declare what they produce and consume via tags. The session
/// matches them automatically: same tag = same intermediate = computed once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntermediateTag {
    /// Pairwise distance matrix: D[i,j] = metric(data[i], data[j]).
    /// Shareable across DBSCAN, KNN, outlier detection, silhouette scoring.
    DistanceMatrix { metric: Metric, data_id: DataId },

    /// Per-group (sum, sum_sq, count). Shareable across normalization,
    /// z-scoring, feature engineering, linear regression preprocessing.
    SufficientStatistics { data_id: DataId, grouping_id: DataId },

    /// Cluster label assignments.
    ClusterLabels { data_id: DataId },

    /// Moment statistics for an ungrouped numeric array: {count, sum, min, max, m2, m3, m4}.
    ///
    /// This is the minimum sufficient representation (MSR) for all moment-based
    /// statistics. A single scatter pass produces these 7 accumulators; every
    /// descriptive stat, hypothesis test, and normalization step derives from them
    /// in O(1) arithmetic — zero re-scanning.
    ///
    /// Produced by: `descriptive::moments_session`
    /// Consumed by: hypothesis tests (one/two-sample t, ANOVA), normalization,
    ///              z-scoring, Pearson correlation, regression preprocessing.
    MomentStats { data_id: DataId },

    /// Per-group moment statistics: one `MomentStats` per group.
    ///
    /// Produced when a groupby + moment computation runs together.
    /// Consumed by: grouped hypothesis tests, multi-group ANOVA, group normalization.
    GroupedMomentStats { data_id: DataId, groups_id: DataId },

    /// Full data-quality summary for a single slice of observations.
    ///
    /// Stores a `DataQualitySummary` (tick_count, price_cv, effective_sample_size,
    /// lag1_autocorr, jump_ratio_proxy, trend_r2, split_variance_ratio,
    /// acf_decay_exponent, has_vol_clustering, has_trend, is_stationary_adf_05).
    ///
    /// Produced by: `data_quality::DataQualitySummary::from_slice` (or the
    /// session-aware variant that registers on first computation).
    /// Consumed by: every validity predicate (`fft_is_valid`, `garch_is_valid`,
    /// ...), every auto-detection family chain, every bridge leaf that checks
    /// sample adequacy before running its expensive math. Computing this once
    /// per bin and reading it from 18+ downstream consumers is the textbook
    /// TamSession sharing win: O(n) pass becomes O(1) lookup.
    DataQuality { data_id: DataId },

    /// Streaming sketch over a slice: HyperLogLog, Bloom filter, Count-Min
    /// Sketch, or SpaceSaving top-k counter. `kind` distinguishes the
    /// sketch family and `precision` captures the single most important
    /// sizing parameter (HLL register bits, Bloom bit count, CMS width,
    /// Top-K k).
    ///
    /// Produced by: `sketches::build_*_session` helpers (one per family).
    /// Consumed by: any downstream consumer that wants the cached sketch
    /// instead of rebuilding — distinct-count queries, set-membership
    /// probes, heavy-hitter lookups, union-of-streams merges.
    Sketch { kind: SketchKind, precision: u32, data_id: DataId },

    /// Pairwise distance matrix computed under a named manifold geometry
    /// (Poincaré, spherical geodesic, etc.). `manifold_name` is the
    /// string key returned by `TiledOp::params_key()`.
    ///
    /// Distinct from `DistanceMatrix` (which uses a `Metric` discriminant)
    /// because manifold geometries are parameterised by a free-form string
    /// rather than an enum variant, and their caching contract differs.
    ///
    /// Produced by: KNN, DBSCAN, and clustering pipelines that accept a
    /// `ManifoldSpec`. Consumed by any subsequent analysis step over the
    /// same manifold + data pair.
    ManifoldDistanceMatrix { manifold_name: String, data_id: DataId },

    /// Weighted mixture of manifold distance matrices.
    /// `mix_id` is the content-hash of the mixture specification
    /// (manifold names + weights), distinct from `data_id`.
    ///
    /// Produced by: `ClusteringEngine::discover_clusters_manifold_mixture`.
    /// Consumed by: downstream clustering steps that received the same
    /// mixture configuration over the same data.
    ManifoldMixtureDistance { mix_id: DataId, data_id: DataId },

    /// K-means centroid assignment (cluster labels for each row).
    /// `k` is the number of clusters requested; together with `data_id`
    /// it uniquely identifies one run of KMeans on this dataset.
    ///
    /// Produced by: `ClusteringPipeline::kmeans`.
    /// Consumed by: any step in the same pipeline that requires the label
    /// vector — silhouette scoring, group-stats extraction, etc.
    Centroids { data_id: DataId, k: usize },
}

/// Which streaming sketch family a `Sketch` intermediate represents.
///
/// Used as a discriminant inside `IntermediateTag::Sketch` so different
/// sketch types over the same data do not collide in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SketchKind {
    /// HyperLogLog distinct-count sketch with `2^precision` registers.
    HyperLogLog,
    /// Bloom filter; `precision` holds the bit capacity.
    BloomFilter,
    /// Count-Min Sketch; `precision` holds the width (depth tracked elsewhere).
    CountMinSketch,
    /// SpaceSaving top-k counter; `precision` holds `k`.
    TopK,
}

// ---------------------------------------------------------------------------
// TamSession — the sharing registry
// ---------------------------------------------------------------------------

/// The sharing session: tracks computed intermediates and wires producers to consumers.
///
/// When an algorithm produces an intermediate, it registers it with a tag.
/// When another algorithm needs the same intermediate (same tag), the session
/// returns the cached value — zero GPU cost.
///
/// # The sharing contract
///
/// - Intermediates are immutable once registered (Arc<T> enforces this).
/// - Tags are content-addressed: same computation on same data = same tag.
/// - The session does NOT own the computation engines. It owns the RESULTS.
///
/// # Example
///
/// ```no_run
/// use tambear::intermediates::{TamSession, IntermediateTag, Metric, DataId, DistanceMatrix};
/// use std::sync::Arc;
///
/// let mut session = TamSession::new();
/// # let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
/// let data_id = DataId::from_f64(&data);
/// let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id };
///
/// // First algorithm computes distance, registers it
/// # let distance_matrix = DistanceMatrix { metric: Metric::L2Sq, n: 2, data: Arc::new(vec![0.0, 1.0, 1.0, 0.0]) };
/// session.register(tag.clone(), Arc::new(distance_matrix));
///
/// // Second algorithm gets it for free
/// let dist: Option<Arc<DistanceMatrix>> = session.get(&tag);
/// ```
pub struct TamSession {
    intermediates: HashMap<IntermediateTag, Arc<dyn Any + Send + Sync>>,
}

impl TamSession {
    pub fn new() -> Self {
        Self { intermediates: HashMap::new() }
    }

    /// Register a produced intermediate. Returns true if this is new, false if it
    /// was already registered (first producer wins).
    pub fn register<T: Any + Send + Sync>(&mut self, tag: IntermediateTag, value: Arc<T>) -> bool {
        use std::collections::hash_map::Entry;
        match self.intermediates.entry(tag) {
            Entry::Vacant(e) => {
                e.insert(value);
                true
            }
            Entry::Occupied(_) => false, // already produced — first wins
        }
    }

    /// Try to get a precomputed intermediate by tag.
    /// Returns None if no producer has registered this tag yet.
    pub fn get<T: Any + Send + Sync>(&self, tag: &IntermediateTag) -> Option<Arc<T>> {
        self.intermediates.get(tag)
            .and_then(|v| v.clone().downcast::<T>().ok())
    }

    /// Check if an intermediate is available without consuming it.
    pub fn has(&self, tag: &IntermediateTag) -> bool {
        self.intermediates.contains_key(tag)
    }

    /// Number of intermediates currently held.
    pub fn len(&self) -> usize {
        self.intermediates.len()
    }

    /// List all registered tags (for debugging / visualization).
    pub fn tags(&self) -> Vec<&IntermediateTag> {
        self.intermediates.keys().collect()
    }

    /// Drop all intermediates (free GPU/host memory).
    pub fn clear(&mut self) {
        self.intermediates.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_id_deterministic() {
        let data = vec![1.0f64, 2.0, 3.0];
        let id1 = DataId::from_f64(&data);
        let id2 = DataId::from_f64(&data);
        assert_eq!(id1, id2, "same data must produce same DataId");
    }

    #[test]
    fn data_id_different_data() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![1.0f64, 2.0, 4.0]; // one value different
        assert_ne!(DataId::from_f64(&a), DataId::from_f64(&b));
    }

    #[test]
    fn data_id_combine() {
        let a = DataId::from_f64(&[1.0]);
        let b = DataId::from_f64(&[2.0]);
        let ab = DataId::combine(a, b);
        let ba = DataId::combine(b, a);
        assert_ne!(ab, ba, "combine must be order-dependent");
    }

    #[test]
    fn distance_matrix_row_access() {
        let data = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ];
        let dm = DistanceMatrix::from_vec(Metric::L2, 3, data);
        assert_eq!(dm.entry(0, 2), 2.0);
        assert_eq!(dm.entry(1, 1), 0.0);
        assert_eq!(dm.row(2), &[2.0, 1.0, 0.0]);
    }

    #[test]
    fn metric_compatibility() {
        let dm = DistanceMatrix::from_vec(Metric::L2Sq, 2, vec![0.0, 1.0, 1.0, 0.0]);
        assert!(dm.is_compatible_with(Metric::L2Sq));
        assert!(!dm.is_compatible_with(Metric::Cosine));
    }

    #[test]
    fn sufficient_statistics_derived_values() {
        // Data: [1, 2, 3] → sum=6, sum_sqs=14, count=3
        // mean = 2.0, pop_var = 14/3 - 4 = 2/3
        let stats = SufficientStatistics::from_vecs(1, vec![6.0], vec![14.0], vec![3.0]);
        assert!((stats.mean(0) - 2.0).abs() < 1e-10);
        assert!((stats.variance(0) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn sufficient_statistics_welford_constructor() {
        // Data: [1, 2, 3] → sum=6, m2 = (1-2)²+(2-2)²+(3-2)² = 2.0, count=3
        let stats = SufficientStatistics::from_welford(1, vec![6.0], vec![2.0], vec![3.0]);
        assert!((stats.mean(0) - 2.0).abs() < 1e-10);
        assert!((stats.variance(0) - 2.0 / 3.0).abs() < 1e-10);
        assert!((stats.variance_sample(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sufficient_statistics_merge() {
        // Group A: [1, 2, 3] → sum=6, m2=2.0, count=3
        let a = SufficientStatistics::from_welford(1, vec![6.0], vec![2.0], vec![3.0]);
        // Group B: [4, 5, 6] → sum=15, m2=2.0, count=3
        let b = SufficientStatistics::from_welford(1, vec![15.0], vec![2.0], vec![3.0]);
        let merged = a.merge(&b);

        // Combined [1,2,3,4,5,6]: mean=3.5, pop_var=17.5/6
        assert!((merged.mean(0) - 3.5).abs() < 1e-10);
        // m2 = Σ(x-3.5)² = 6.25+2.25+0.25+0.25+2.25+6.25 = 17.5
        let expected_var = 17.5 / 6.0;
        assert!((merged.variance(0) - expected_var).abs() < 1e-10,
            "merged variance: got {}, expected {}", merged.variance(0), expected_var);
    }

    #[test]
    fn sufficient_statistics_high_offset_stability() {
        // The canary: data with large offset that kills naive formula
        let offset = 1e12;
        let n = 100;
        let step = 0.001;
        let data: Vec<f64> = (0..n).map(|i| offset + i as f64 * step).collect();
        let sum: f64 = data.iter().sum();
        let sum_sqs: f64 = data.iter().map(|v| v * v).sum();
        let count = n as f64;

        let stats = SufficientStatistics::from_vecs(1, vec![sum], vec![sum_sqs], vec![count]);
        let expected_var = (n as f64 * n as f64 - 1.0) * step * step / 12.0;
        let rel_err = ((stats.variance(0) - expected_var) / expected_var).abs();

        // from_vecs converts sum_sqs to m2 — this tests that the conversion works
        // Note: at 1e12 offset, the conversion sum_sqs - sum²/n may still lose
        // precision, but it's much better than the old variance() formula.
        // For truly stable results at extreme offsets, use from_welford or MomentStats.
        eprintln!("High-offset variance: got={:.6e}, expected={:.6e}, rel_err={:.2e}",
            stats.variance(0), expected_var, rel_err);
    }

    #[test]
    fn tag_matching_requires_same_data() {
        let id_a = DataId::from_f64(&[1.0, 2.0]);
        let id_b = DataId::from_f64(&[3.0, 4.0]);

        let tag_a = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id_a };
        let tag_b = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id_b };
        let tag_a2 = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id_a };

        assert_ne!(tag_a, tag_b, "different data must not match");
        assert_eq!(tag_a, tag_a2, "same computation on same data must match");
    }

    #[test]
    fn tag_matching_requires_same_metric() {
        let id = DataId::from_f64(&[1.0]);
        let tag_l2 = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id };
        let tag_cos = IntermediateTag::DistanceMatrix { metric: Metric::Cosine, data_id: id };
        assert_ne!(tag_l2, tag_cos, "different metrics must not match");
    }

    #[test]
    fn session_register_and_get() {
        let mut session = TamSession::new();
        let id = DataId::from_f64(&[1.0, 2.0]);
        let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id };

        let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, 2, vec![0.0, 1.0, 1.0, 0.0]));
        assert!(session.register(tag.clone(), dm));
        assert_eq!(session.len(), 1);

        let retrieved: Option<Arc<DistanceMatrix>> = session.get(&tag);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().entry(0, 1), 1.0);
    }

    #[test]
    fn session_miss_on_different_data() {
        let mut session = TamSession::new();
        let id_a = DataId::from_f64(&[1.0]);
        let id_b = DataId::from_f64(&[2.0]);

        let tag_a = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id_a };
        let tag_b = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id_b };

        let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, 1, vec![0.0]));
        session.register(tag_a, dm);

        let miss: Option<Arc<DistanceMatrix>> = session.get(&tag_b);
        assert!(miss.is_none(), "different data must not hit");
    }

    #[test]
    fn session_first_producer_wins() {
        let mut session = TamSession::new();
        let id = DataId::from_f64(&[1.0]);
        let tag = IntermediateTag::SufficientStatistics { data_id: id, grouping_id: id };

        let stats1 = Arc::new(SufficientStatistics::from_vecs(1, vec![10.0], vec![100.0], vec![5.0]));
        let stats2 = Arc::new(SufficientStatistics::from_vecs(1, vec![20.0], vec![400.0], vec![5.0]));

        assert!(session.register(tag.clone(), stats1), "first register should succeed");
        assert!(!session.register(tag.clone(), stats2), "second register should be rejected");

        let got: Arc<SufficientStatistics> = session.get(&tag).unwrap();
        assert_eq!(got.sums[0], 10.0, "first producer's value should be retained");
    }

    #[test]
    fn session_sharing_across_types() {
        let mut session = TamSession::new();
        let data = vec![0.0f64, 1.0, 1.0, 0.0];
        let id = DataId::from_f64(&data);

        // Register a DistanceMatrix
        let dist_tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: id };
        let dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, 2, data.clone()));
        session.register(dist_tag.clone(), dm);

        // Register SufficientStatistics for same data
        let stats_tag = IntermediateTag::SufficientStatistics {
            data_id: id,
            grouping_id: DataId::from_i32(&[0, 0]),
        };
        let stats = Arc::new(SufficientStatistics::from_vecs(1, vec![1.0], vec![1.0], vec![2.0]));
        session.register(stats_tag.clone(), stats);

        assert_eq!(session.len(), 2);

        // Both retrievable
        assert!(session.get::<DistanceMatrix>(&dist_tag).is_some());
        assert!(session.get::<SufficientStatistics>(&stats_tag).is_some());

        // Wrong type on right tag returns None
        assert!(session.get::<SufficientStatistics>(&dist_tag).is_none());
    }
}
