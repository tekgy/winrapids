//! Geometric manifold as a composable parameter for distance, mean, and gradient.
//!
//! ## The insight
//!
//! `Metric` tells you how to compute distance. `Manifold` tells you four things:
//!
//! 1. **Distance** — how to measure proximity (`tiled_dist_expr`)
//! 2. **Mean/centroid** — how to average points (`centroid_update_expr`)
//! 3. **Gradient scaling** — the conformal factor for optimization (`gradient_scale_expr`)
//! 4. **Projection** — how to keep points on the manifold after gradient steps (`project_expr`)
//!
//! Each of these is a JIT expression string. When `TiledEngine` is parameterized by `Manifold`,
//! the same KMeans/DBSCAN algorithm works in any geometry — the geometry is a parameter.
//!
//! ## Current status
//!
//! This module defines the type scaffolding. The JIT wiring (passing `Manifold::tiled_dist_expr()`
//! into TiledEngine) is the next layer — it requires extending TiledEngine's kernel generation.
//! The types and expression strings are correct and ready; the compiler plumbing comes later.
//!
//! ## Manifold hierarchy
//!
//! ```text
//! Manifold
//! ├── Euclidean            — standard R^d, L2 distance
//! ├── Poincare { c }       — hyperbolic, curvature c < 0
//! ├── Sphere { r }         — spherical, radius r > 0
//! ├── Learned { params }   — metric learned from data (future)
//! └── Bayesian { prior, posterior } — prior geometry + learned correction
//! ```
//!
//! ## Connection to Metric
//!
//! `Metric` is a restricted view of `Manifold` — distance only, no mean/gradient.
//! `Manifold::Euclidean` with L2Sq distance corresponds to `Metric::L2Sq`.
//! `Metric` is preserved for the current `IntermediateTag` keys; `Manifold` will replace it
//! when non-Euclidean algorithms are implemented.

use std::fmt;
use std::sync::Arc;
use crate::intermediates::{DataId, DistanceMatrix, Metric};
use winrapids_tiled::TiledOp;

// ---------------------------------------------------------------------------
// Manifold
// ---------------------------------------------------------------------------

/// A geometric space that parameterizes distance, mean, and gradient operations.
///
/// Each variant corresponds to a different geometry. The methods on this type
/// generate CUDA expression strings that JIT-compile into the appropriate kernels.
///
/// ## Hashing
///
/// `Manifold` implements `Hash + Eq` for use as an `IntermediateTag` key.
/// Floating-point parameters (curvature, radius) are stored as bit patterns to
/// enable deterministic hashing without floating-point comparison issues.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Manifold {
    /// Standard Euclidean space R^d. L2 or L2Sq distance, arithmetic mean.
    ///
    /// This is the default geometry. All existing tambear algorithms operate here.
    Euclidean,

    /// Poincaré ball model of hyperbolic space.
    ///
    /// Curvature c < 0 (typically c = -1.0). Points live in the open ball ||x|| < 1/√|c|.
    /// The Poincaré model is the natural embedding for hierarchical data (trees, knowledge graphs).
    ///
    /// `curvature_bits` stores the f64 curvature value as raw bits for `Hash`.
    Poincare {
        /// Curvature (negative). Stored as f64 bit pattern. Use `curvature()` to get the f64.
        curvature_bits: u64,
    },

    /// Spherical geometry S^(d-1) with radius r.
    ///
    /// Points live on the sphere ||x|| = r. Geodesic distance is arc length.
    /// The natural geometry for normalized embeddings (e.g., cosine-distance spaces).
    ///
    /// `radius_bits` stores the f64 radius as raw bits for `Hash`.
    Sphere {
        /// Radius (positive). Stored as f64 bit pattern. Use `radius()` to get the f64.
        radius_bits: u64,
    },

    /// Spherical geodesic distance (arc length) on S^(d-1) with radius r.
    ///
    /// Uses the Riemannian geodesic arc length `arccos(⟨x,y⟩ / (||x||·||y||))`,
    /// valid for any non-zero vectors (not just unit-normalized).
    ///
    /// Unlike `Sphere` (cosine distance, unit vectors only), this computes the
    /// true arc-length distance and uses the 3-field accumulator
    /// `{sq_norm_x, sq_norm_y, dot_prod}` — the same sufficient statistics as
    /// the Poincaré ball and Euclidean distance.
    ///
    /// `radius_bits` stores the f64 radius as raw bits for `Hash`.
    SphericalGeodesic {
        /// Radius (positive). Stored as f64 bit pattern. Use `radius()` to get the f64.
        radius_bits: u64,
    },

    /// Metric space learned from data.
    ///
    /// The metric is parameterized by a learned weight matrix or neural network,
    /// identified by content hash. The JIT expressions reference the params buffer.
    /// (Future: not yet implemented.)
    Learned {
        /// Content hash of the learned parameters buffer.
        params_id: DataId,
    },

    /// Geometry with a prior and a learned posterior correction.
    ///
    /// Starts from a base geometry (prior) and applies a learned deformation (posterior).
    /// This is exactly what baking learned constants into a JIT kernel does —
    /// the constants are the posterior correction to the base kernel expression.
    ///
    /// (Future: not yet implemented.)
    Bayesian {
        /// Base geometry before correction.
        prior: Box<Manifold>,
        /// Content hash of the posterior parameters.
        posterior_id: DataId,
    },
}

impl Manifold {
    // -----------------------------------------------------------------------
    // Constructors with f64 parameters
    // -----------------------------------------------------------------------

    /// Create a Poincaré ball manifold with the given curvature (must be < 0).
    pub fn poincare(curvature: f64) -> Self {
        assert!(curvature < 0.0, "Poincaré curvature must be negative, got {curvature}");
        Manifold::Poincare { curvature_bits: curvature.to_bits() }
    }

    /// Create a sphere manifold with the given radius (must be > 0).
    pub fn sphere(radius: f64) -> Self {
        assert!(radius > 0.0, "Sphere radius must be positive, got {radius}");
        Manifold::Sphere { radius_bits: radius.to_bits() }
    }

    /// Create a spherical geodesic manifold with the given radius (must be > 0).
    ///
    /// Computes true arc-length distance: `arccos(⟨x,y⟩ / (||x||·||y||))`.
    /// Valid for non-unit vectors; uses the 3-field `{sq_norm_x, sq_norm_y, dot_prod}`
    /// accumulator — no unit-normalization required by the caller.
    pub fn spherical_geodesic(radius: f64) -> Self {
        assert!(radius > 0.0, "SphericalGeodesic radius must be positive, got {radius}");
        Manifold::SphericalGeodesic { radius_bits: radius.to_bits() }
    }

    // -----------------------------------------------------------------------
    // Parameter accessors
    // -----------------------------------------------------------------------

    /// Curvature for Poincaré manifolds (returns None for others).
    pub fn curvature(&self) -> Option<f64> {
        match self {
            Manifold::Poincare { curvature_bits } => Some(f64::from_bits(*curvature_bits)),
            _ => None,
        }
    }

    /// Radius for sphere manifolds (returns None for others).
    pub fn radius(&self) -> Option<f64> {
        match self {
            Manifold::Sphere { radius_bits } |
            Manifold::SphericalGeodesic { radius_bits } => Some(f64::from_bits(*radius_bits)),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // JIT expression strings
    // -----------------------------------------------------------------------

    /// CUDA expression for the inner-loop distance kernel between two d-dimensional points.
    ///
    /// The expression is a string that computes the pairwise distance (or squared distance)
    /// between points `a` (pointer to d f64s) and `b` (pointer to d f64s), using
    /// loop variable `k` for the dimension index.
    ///
    /// This replaces TiledEngine's hardcoded `(a-b)^2` inner loop with a
    /// geometry-parameterized version.
    ///
    /// ## Output format
    ///
    /// The returned string is a CUDA expression that accumulates into `acc`:
    /// `acc += <expr>`. This maps to `tiled_reduce` with a custom phi_expr.
    pub fn tiled_dist_expr(&self) -> &'static str {
        match self {
            Manifold::Euclidean =>
                // Standard L2Sq: (a[k] - b[k])^2
                "(a[k] - b[k]) * (a[k] - b[k])",

            Manifold::Sphere { .. } =>
                // Dot product for spherical (use for cosine similarity; normalized → arc length)
                "a[k] * b[k]",

            Manifold::SphericalGeodesic { .. } =>
                // Accumulates dot product; full arccos(dot/sqrt(norms)) is in ManifoldDistanceOp.
                // This legacy expr doesn't carry norms — use ManifoldDistanceOp for the true geodesic.
                "a[k] * b[k]",

            Manifold::Poincare { .. } =>
                // Poincaré ball: mobius subtraction distance — computed as a whole,
                // not per-dimension. The per-dimension form would require a full kernel
                // rewrite. For now: Euclidean L2Sq as fallback.
                // TODO: requires whole-vector kernel (not per-dim accumulate)
                "(a[k] - b[k]) * (a[k] - b[k])",

            Manifold::Learned { .. } | Manifold::Bayesian { .. } =>
                // Learned metrics require a network call — not expressible as a simple
                // per-dimension accumulate. Fall back to Euclidean.
                // TODO: implement when learned metric JIT is ready
                "(a[k] - b[k]) * (a[k] - b[k])",
        }
    }

    /// CUDA expression for updating a centroid with a new data point (for KMeans).
    ///
    /// Returns an expression that computes the weighted Fréchet mean update:
    /// `centroid[k] = weighted_mean(centroid[k], point[k], count)`.
    ///
    /// For Euclidean: arithmetic mean (standard KMeans centroid step).
    /// For Poincaré: Möbius gyrovector centroid (future).
    /// For Sphere: spherical mean via normalization (future).
    pub fn centroid_update_expr(&self) -> &'static str {
        match self {
            // Euclidean: simple scatter-sum then divide by count
            Manifold::Euclidean =>
                "sum[k] / count",

            // Sphere / SphericalGeodesic: renormalize to the sphere after arithmetic mean
            // The normalization happens per-point after the scatter-sum step
            Manifold::Sphere { .. } | Manifold::SphericalGeodesic { .. } =>
                "sum[k] / norm(sum)",  // placeholder — requires norm helper

            // Poincaré: gyrovector mean (future)
            Manifold::Poincare { .. } | Manifold::Learned { .. } | Manifold::Bayesian { .. } =>
                "sum[k] / count",  // fallback to Euclidean until implemented
        }
    }

    /// CUDA expression for the conformal (Riemannian) gradient scaling factor.
    ///
    /// Riemannian SGD: `grad_manifold = scale_factor(x) * grad_euclidean`
    ///
    /// For Euclidean: factor = 1.0 (no scaling).
    /// For Poincaré with curvature c: factor = ((1 - c*||x||²) / 2)²
    /// For Sphere: factor = 1/r²
    ///
    /// Returns a CUDA expression in terms of `x_norm_sq` (squared norm of the current point).
    pub fn gradient_scale_expr(&self) -> String {
        match self {
            Manifold::Euclidean => "1.0".to_owned(),

            Manifold::Poincare { curvature_bits } => {
                let c = f64::from_bits(*curvature_bits);
                let abs_c = c.abs();
                // Conformal factor: ((1 - |c|*||x||²) / 2)²
                format!("pow((1.0 - {abs_c:.15} * x_norm_sq) / 2.0, 2.0)")
            }

            Manifold::Sphere { radius_bits } | Manifold::SphericalGeodesic { radius_bits } => {
                let r = f64::from_bits(*radius_bits);
                format!("{:.15}", 1.0 / (r * r))
            }

            Manifold::Learned { .. } | Manifold::Bayesian { .. } =>
                "1.0".to_owned(),  // fallback
        }
    }

    /// Whether this manifold needs a projection step after gradient updates.
    ///
    /// Euclidean: no projection needed (R^d is closed).
    /// Poincaré: yes — gradient steps may push points outside the ball.
    /// Sphere: yes — gradient steps may push points off the sphere.
    pub fn needs_projection(&self) -> bool {
        matches!(self, Manifold::Poincare { .. } | Manifold::Sphere { .. } | Manifold::SphericalGeodesic { .. })
    }

    /// CUDA expression for projecting a point back onto the manifold.
    ///
    /// Returns a per-dimension expression that projects `point[k]` back.
    pub fn project_expr(&self) -> String {
        match self {
            Manifold::Euclidean => "point[k]".to_owned(),  // identity

            Manifold::Poincare { curvature_bits } => {
                let c = f64::from_bits(*curvature_bits);
                let abs_c = c.abs();
                let limit = (1.0 / abs_c.sqrt()) * (1.0 - 1e-5);
                // Project inside the ball: if ||x|| >= 1/√|c|, rescale
                format!("(norm_sq >= {:.15}) ? point[k] * ({:.15} / sqrt(norm_sq)) : point[k]",
                    limit * limit, limit)
            }

            Manifold::Sphere { radius_bits } | Manifold::SphericalGeodesic { radius_bits } => {
                let r = f64::from_bits(*radius_bits);
                // Project onto sphere: rescale to radius r
                format!("point[k] * ({r:.15} / sqrt(norm_sq))")
            }

            Manifold::Learned { .. } | Manifold::Bayesian { .. } =>
                "point[k]".to_owned(),
        }
    }

    /// Convert from a `Metric` to the closest `Manifold` representation.
    ///
    /// This bridges the current `Metric`-based type system (used in `IntermediateTag`)
    /// with the `Manifold` type system. As `IntermediateTag` migrates to use `Manifold`,
    /// this conversion allows gradual migration.
    ///
    /// - `Metric::L2Sq` / `Metric::L2` → `Manifold::Euclidean`
    /// - `Metric::Cosine` → `Manifold::Sphere { radius: 1.0 }` (unit sphere)
    /// - `Metric::Dot` → `Manifold::Sphere { radius: 1.0 }` (treated as inner product on sphere)
    /// - `Metric::Manhattan` → `Manifold::Euclidean` (approximation; no exact manifold equivalent)
    pub fn from_metric(m: Metric) -> Self {
        match m {
            Metric::L2Sq | Metric::L2 | Metric::Manhattan => Manifold::Euclidean,
            Metric::Cosine | Metric::Dot => Manifold::sphere(1.0),
        }
    }

    /// Whether this manifold's distance is symmetric: dist(a,b) == dist(b,a).
    pub fn is_symmetric(&self) -> bool {
        true  // all current variants are symmetric
    }

    /// Whether smaller distance means more similar (false for dot-product / Sphere cosine).
    pub fn distance_is_dissimilarity(&self) -> bool {
        !matches!(self, Manifold::Sphere { .. })
    }

    /// Human-readable name for logging and session tags.
    pub fn name(&self) -> String {
        match self {
            Manifold::Euclidean => "euclidean".to_owned(),
            Manifold::Poincare { curvature_bits } =>
                format!("poincare(c={:.4})", f64::from_bits(*curvature_bits)),
            Manifold::Sphere { radius_bits } =>
                format!("sphere(r={:.4})", f64::from_bits(*radius_bits)),
            Manifold::SphericalGeodesic { radius_bits } =>
                format!("sphere_geodesic(r={:.4})", f64::from_bits(*radius_bits)),
            Manifold::Learned { params_id } =>
                format!("learned(params={:016x})", params_id.0),
            Manifold::Bayesian { prior, posterior_id } =>
                format!("bayesian({}, post={:016x})", prior.name(), posterior_id.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Manifold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ManifoldMixture — weighted superposition of geometric spaces
// ---------------------------------------------------------------------------

/// A weighted mixture of `Manifold` variants.
///
/// Instead of committing to one geometry, `ManifoldMixture` runs multiple geometries
/// simultaneously. The combination weights determine the final distance matrix:
/// `D_combined[i,j] = Σ w_k * D_k[i,j]`
///
/// The weights encode the model's answer to: "what geometry best explains this data?"
///
/// ## Session integration
///
/// Each component's `DistanceMatrix` is cached independently in the session under
/// `ManifoldDistanceMatrix { manifold, data_id }`. Combining them is O(n²) (weighted sum),
/// not O(n²d) (re-computing distances). The combined matrix is cached under
/// `ManifoldMixtureDistance { mix_id, data_id }`.
///
/// ## Example
///
/// ```
/// use tambear::manifold::{Manifold, ManifoldMixture};
///
/// let mix = ManifoldMixture::new(vec![
///     (Manifold::Euclidean, 0.7),
///     (Manifold::poincare(-1.0), 0.3),
/// ]);
/// assert!((mix.total_weight() - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ManifoldMixture {
    /// Component manifolds with associated weights.
    /// Weights should sum to 1.0 (enforced by `normalize()`, checked in `combine()`).
    pub components: Vec<(Manifold, f64)>,
}

impl ManifoldMixture {
    /// Create a new mixture. Weights need not sum to 1.0 yet — call `normalize()` if needed.
    pub fn new(components: Vec<(Manifold, f64)>) -> Self {
        assert!(!components.is_empty(), "ManifoldMixture: must have at least one component");
        Self { components }
    }

    /// Create a uniform mixture (equal weights, summing to 1.0).
    pub fn uniform(manifolds: Vec<Manifold>) -> Self {
        assert!(!manifolds.is_empty(), "ManifoldMixture::uniform: need at least one manifold");
        let n = manifolds.len();
        let w = 1.0 / n as f64;
        Self::new(manifolds.into_iter().map(|m| (m, w)).collect())
    }

    /// Create a single-manifold "mixture" (degenerate case, w=1.0).
    pub fn single(manifold: Manifold) -> Self {
        Self::new(vec![(manifold, 1.0)])
    }

    /// Normalize weights so they sum to 1.0.
    pub fn normalize(mut self) -> Self {
        let total: f64 = self.components.iter().map(|(_, w)| w).sum();
        assert!(total > 0.0, "ManifoldMixture::normalize: weights sum to zero");
        for (_, w) in &mut self.components {
            *w /= total;
        }
        self
    }

    /// Sum of all weights (should be 1.0 for a valid mixture).
    pub fn total_weight(&self) -> f64 {
        self.components.iter().map(|(_, w)| w).sum()
    }

    /// Stable content hash of the mixture specification (for session keys).
    ///
    /// The hash is over the serialized `(manifold_name, weight_bits)` pairs —
    /// two mixtures with the same manifolds in the same order with the same weights
    /// produce the same `mix_id`.
    pub fn mix_id(&self) -> DataId {
        let mut bytes: Vec<u8> = Vec::new();
        for (m, w) in &self.components {
            let name = m.name();
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&w.to_bits().to_le_bytes());
        }
        DataId::from_bytes(&bytes)
    }

    /// Combine pre-computed distance matrices from the session via weighted sum.
    ///
    /// `matrices[i]` must correspond to `self.components[i]`. Each `DistanceMatrix`
    /// must have the same `n` (number of points).
    ///
    /// Returns a flat `Vec<f64>` of size `n*n`, row-major:
    /// `result[i*n + j] = Σ_k w_k * matrices[k].data[i*n + j]`
    ///
    /// This is O(n²) per call (not O(n²d)) — the expensive per-dimension work happened
    /// when each component matrix was computed.
    pub fn combine(&self, matrices: &[Arc<DistanceMatrix>]) -> Vec<f64> {
        assert_eq!(
            matrices.len(), self.components.len(),
            "ManifoldMixture::combine: got {} matrices for {} components",
            matrices.len(), self.components.len()
        );
        assert!(!matrices.is_empty(), "ManifoldMixture::combine: no matrices provided");

        let n = matrices[0].n;
        for m in matrices.iter().skip(1) {
            assert_eq!(m.n, n, "ManifoldMixture::combine: all matrices must have the same n");
        }

        let mut result = vec![0.0f64; n * n];
        for ((_, w), matrix) in self.components.iter().zip(matrices.iter()) {
            for (r, d) in result.iter_mut().zip(matrix.data.iter()) {
                *r += w * d;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ManifoldDistanceOp — TiledOp bridge for manifold-parameterized distance
// ---------------------------------------------------------------------------

/// A [`TiledOp`] that computes pairwise distances under a given geometry.
///
/// This bridges `Manifold` geometry specifications with the `TiledEngine` kernel
/// generator. Pass this to `TiledEngine::run()` or `AccumulateEngine::accumulate()`
/// (via `Op::DotProduct` override) to compute distance matrices in non-Euclidean spaces.
///
/// ## Geometry notes
///
/// | Manifold    | Accumulate       | Extract            | Pre-req           |
/// |-------------|------------------|--------------------|-------------------|
/// | Euclidean   | Σ(a-b)²          | acc                | none              |
/// | Sphere      | Σ(a·b)           | 1.0 - acc          | unit-normalize A,B|
/// | Poincaré    | Σ(a-b)² (approx) | acc                | TODO: full-vector |
/// | Learned     | Σ(a-b)² (approx) | acc                | TODO              |
///
/// For `Manifold::Sphere`: the extracted value `1.0 - dot(a, b)` is cosine distance
/// only when both input vectors are unit-normalized. Use `Manifold::needs_prenormalization()`
/// to check, and pre-normalize before calling `TiledEngine::run()`.
pub struct ManifoldDistanceOp {
    /// The geometry to use for distance computation.
    pub manifold: Manifold,
}

impl ManifoldDistanceOp {
    pub fn new(manifold: Manifold) -> Self { Self { manifold } }
}

impl TiledOp for ManifoldDistanceOp {
    fn name(&self) -> &'static str { "manifold_distance" }

    fn params_key(&self) -> String { self.manifold.name() }

    fn cuda_acc_type(&self) -> String {
        match &self.manifold {
            Manifold::Poincare { .. } =>
                // 3-field struct: all three quantities decompose per-dimension.
                // d(x,y) = (2/√κ) * arccosh(1 + 2κ·sq_dist/((1-κ·sq_norm_x)(1-κ·sq_norm_y)))
                "struct PoincareAcc { double sq_dist; double sq_norm_x; double sq_norm_y; }".into(),
            Manifold::SphericalGeodesic { .. } =>
                // 3-field struct: same sufficient statistics as the mixture kernel.
                // d(x,y) = arccos(dot_prod / sqrt(sq_norm_x · sq_norm_y))
                "struct SphGeoAcc { double sq_norm_x; double sq_norm_y; double dot_prod; }".into(),
            _ =>
                "double".into(),
        }
    }

    fn cuda_identity(&self) -> String {
        match &self.manifold {
            Manifold::Poincare { .. } | Manifold::SphericalGeodesic { .. } => "{0.0, 0.0, 0.0}".into(),
            _ => "0.0".into(),
        }
    }

    fn cuda_accumulate_body(&self) -> String {
        match &self.manifold {
            Manifold::Euclidean =>
                "    double diff = a_val - b_val;\n    acc += diff * diff;".to_owned(),

            Manifold::Sphere { .. } =>
                // Dot product; extract converts to cosine distance.
                // Input vectors must be unit-normalized by the caller.
                "    acc += a_val * b_val;".to_owned(),

            Manifold::Poincare { .. } =>
                // All three quantities decompose per-dimension — no whole-vector needed.
                concat!(
                    "    double diff = a_val - b_val;\n",
                    "    acc.sq_dist   += diff * diff;\n",
                    "    acc.sq_norm_x += a_val * a_val;\n",
                    "    acc.sq_norm_y += b_val * b_val;"
                ).to_owned(),

            Manifold::SphericalGeodesic { .. } =>
                // Same 3 sufficient statistics as ManifoldMixtureOp's MixtureAcc.
                // Extract: arccos(dot_prod / sqrt(sq_norm_x · sq_norm_y))
                concat!(
                    "    acc.sq_norm_x += a_val * a_val;\n",
                    "    acc.sq_norm_y += b_val * b_val;\n",
                    "    acc.dot_prod  += a_val * b_val;"
                ).to_owned(),

            Manifold::Learned { .. } | Manifold::Bayesian { .. } =>
                "    double diff = a_val - b_val;\n    acc += diff * diff;".to_owned(),
        }
    }

    fn cuda_extract(&self) -> String {
        match &self.manifold {
            Manifold::Sphere { .. } =>
                "(1.0 - acc)".to_owned(),

            Manifold::Poincare { curvature_bits } => {
                let c     = f64::from_bits(*curvature_bits);
                let kappa = c.abs();
                let scale = 2.0 / kappa.sqrt();
                // Single-expression form: fmax/acosh are CUDA device math functions.
                // Clamp denom to ≥1e-15 (points near the ball boundary), arg to ≥1.0.
                format!(
                    "({scale:.15} * acosh(fmax(1.0, 1.0 + 2.0 * {kappa:.15} * acc.sq_dist \
                     / fmax(1e-15, (1.0 - {kappa:.15} * acc.sq_norm_x) \
                     * (1.0 - {kappa:.15} * acc.sq_norm_y)))))",
                )
            }

            Manifold::SphericalGeodesic { .. } =>
                // arccos(dot / sqrt(sq_norm_x · sq_norm_y))
                // Clamp argument to [-1, 1] to guard against float rounding past ±1.
                "acos(fmax(-1.0, fmin(1.0, acc.dot_prod / fmax(1e-30, sqrt(acc.sq_norm_x * acc.sq_norm_y)))))".to_owned(),

            _ => "acc".to_owned(),
        }
    }

    fn acc_byte_size(&self) -> usize {
        match &self.manifold {
            Manifold::Poincare { .. } | Manifold::SphericalGeodesic { .. } => 24,  // 3 × f64
            _ => 8,
        }
    }

    // ------------------------------------------------------------------
    // WGSL overrides — Poincaré falls back to Euclidean L2Sq.
    //
    // f32 precision (7 digits) is insufficient for the Poincaré boundary
    // denominator (1 - κ·||x||²) which vanishes near the ball edge.
    // Sphere and Euclidean work fine at f32 precision.
    // ------------------------------------------------------------------

    fn wgsl_identity(&self) -> String { "0.0".into() }  // always scalar f32

    fn wgsl_accumulate_body(&self) -> String {
        match &self.manifold {
            Manifold::Sphere { .. } =>
                "    acc += a_val * b_val;".to_owned(),
            _ =>
                // Euclidean L2Sq (Poincaré and SphericalGeodesic both fall back —
                // the 3-field struct accumulator isn't supported in the WGSL scalar path)
                "    var diff = a_val - b_val;\n    acc += diff * diff;".to_owned(),
        }
    }

    fn wgsl_extract(&self) -> String {
        match &self.manifold {
            Manifold::Sphere { .. } => "(1.0 - acc)".to_owned(),
            _                      => "acc".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// ManifoldMixtureOp — fused multi-manifold distance kernel
// ---------------------------------------------------------------------------

/// A fused tiled kernel that computes pairwise distances under ALL manifolds in a
/// [`ManifoldMixture`] in a single GPU pass.
///
/// ## Key insight
///
/// The sufficient statistics for ALL inner-product-based distances are:
/// - `sq_norm_x = ||x||² = Σxᵢ²`
/// - `sq_norm_y = ||y||² = Σyᵢ²`
/// - `dot_prod  = x·y   = Σxᵢyᵢ`
///
/// From these three fields (computed in one pass over K), any manifold's distance
/// can be extracted:
/// - Euclidean L2Sq: `sq_norm_x + sq_norm_y − 2·dot_prod`
/// - Poincaré ball:  `(2/√κ)·arccosh(1 + 2κ·sq_dist / denom)` where `denom = (1−κ·sq_norm_x)(1−κ·sq_norm_y)`
/// - Sphere cosine:  `1 − dot_prod` (for unit-normalized inputs)
///
/// **Cost**: 3 accumulator fields regardless of the number of manifolds.
/// **Convergence**: exponential in manifold count (Pith liftability guarantee).
///
/// ## Usage
///
/// Use [`ManifoldMixtureOp::run`] via a [`TiledEngine`]. Returns one distance
/// matrix per component manifold.
pub struct ManifoldMixtureOp {
    pub mixture: ManifoldMixture,
}

const MIXTURE_TILE: usize = 16;

impl ManifoldMixtureOp {
    pub fn new(mixture: ManifoldMixture) -> Self { Self { mixture } }

    pub fn num_manifolds(&self) -> usize { self.mixture.components.len() }

    /// Params key: pipe-separated manifold names, e.g. `"euclidean|poincare(c=-1.0000)"`.
    /// Used as both the CpuBackend dispatch key and the BLAKE3 hash input.
    pub fn params_key(&self) -> String {
        self.mixture.components.iter()
            .map(|(m, _)| m.name())
            .collect::<Vec<_>>()
            .join("|")
    }

    /// BLAKE3 cache key for the compiled kernel.
    pub fn mixture_cache_key(&self) -> String {
        let source = self.cuda_source();
        let mut h = blake3::Hasher::new();
        h.update(b"manifold_mixture");
        h.update(self.params_key().as_bytes());
        h.update(source.as_bytes());
        h.finalize().to_hex().to_string()
    }

    /// Generate the CUDA (f64) composite kernel source.
    ///
    /// Accumulator: `{sq_norm_x, sq_norm_y, dot_prod}` (3 × f64 = 24 bytes).
    /// Extract writes `n_manifolds` values per output position.
    pub fn cuda_source(&self) -> String {
        let n = self.num_manifolds();
        let extract_stmts = self.cuda_extract_stmts();
        let params_key = self.params_key();
        let tile = MIXTURE_TILE;

        format!(
r#"// Tiled accumulation kernel for operator: manifold_mixture
// params: {params_key}
// TILE_M={tile}, TILE_N={tile}, TILE_K={tile}
// Generated by winrapids-tiled (Rust)

struct MixtureAcc {{ double sq_norm_x; double sq_norm_y; double dot_prod; }};
typedef MixtureAcc acc_t;

__device__ __forceinline__ acc_t make_identity() {{
    acc_t acc = {{0.0, 0.0, 0.0}};
    return acc;
}}

// Tiled kernel: all manifold distances in one pass over K
// C has shape M×N×{n} (n_manifolds dimension varies fastest)
extern "C" __global__ void tiled_accumulate(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    const int* __restrict__ dims
) {{
    int M = dims[0]; int N = dims[1]; int K = dims[2];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * {tile} + ty;
    int col = bx * {tile} + tx;

    __shared__ double As[{tile}][{tile}];
    __shared__ double Bs[{tile}][{tile}];

    acc_t acc = make_identity();

    for (int t = 0; t < (K + {tile} - 1) / {tile}; t++) {{
        int a_col = t * {tile} + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        int b_row = t * {tile} + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < {tile}; k++) {{
            double a_val = As[ty][k];
            double b_val = Bs[k][tx];
            acc.sq_norm_x += a_val * a_val;
            acc.sq_norm_y += b_val * b_val;
            acc.dot_prod  += a_val * b_val;
        }}
        __syncthreads();
    }}

    if (row < M && col < N) {{
        int base_idx = (row * N + col) * {n};
        double sq_dist = acc.sq_norm_x + acc.sq_norm_y - 2.0 * acc.dot_prod;
{extract_stmts}    }}
}}
"#)
    }

    /// Generate per-manifold extract statements for the CUDA kernel.
    fn cuda_extract_stmts(&self) -> String {
        let mut stmts = String::new();
        for (k, (manifold, _)) in self.mixture.components.iter().enumerate() {
            stmts += &match manifold {
                Manifold::Euclidean =>
                    format!("        C[base_idx + {k}] = sq_dist;\n"),

                Manifold::Poincare { curvature_bits } => {
                    let kappa = f64::from_bits(*curvature_bits).abs();
                    let scale = 2.0 / kappa.sqrt();
                    let line0 = format!("        {{\n");
                    let line1 = format!("            double denom = fmax(1e-15, (1.0 - {kappa:.15}*acc.sq_norm_x)*(1.0 - {kappa:.15}*acc.sq_norm_y));\n");
                    let line2 = format!("            double arg   = fmax(1.0, 1.0 + 2.0*{kappa:.15}*sq_dist/denom);\n");
                    let line3 = format!("            C[base_idx + {k}] = {scale:.15}*acosh(arg);\n");
                    let line4 = "        }\n".to_owned();
                    line0 + &line1 + &line2 + &line3 + &line4
                }

                Manifold::Sphere { .. } =>
                    // Cosine distance = 1 − dot(x,y). Requires unit-normalized inputs.
                    format!("        C[base_idx + {k}] = 1.0 - acc.dot_prod;\n"),

                Manifold::SphericalGeodesic { .. } =>
                    // True geodesic arc length: arccos(dot / sqrt(sq_norm_x · sq_norm_y)).
                    // Clamp to [-1, 1] to guard against float rounding.
                    format!("        C[base_idx + {k}] = acos(fmax(-1.0, fmin(1.0, acc.dot_prod / fmax(1e-30, sqrt(acc.sq_norm_x * acc.sq_norm_y)))));\n"),

                // Learned / Bayesian: Euclidean L2Sq fallback
                _ =>
                    format!("        C[base_idx + {k}] = sq_dist;\n"),
            };
        }
        stmts
    }

    /// Execute the mixture kernel via `engine`.
    ///
    /// For CUDA and CPU backends: one fused composite kernel (3-field accumulator).
    /// For WGSL backends: N sequential single-manifold kernels (unfused fallback).
    ///
    /// Returns one Vec<f64> per manifold, each of shape M×N row-major.
    pub fn run(
        &self,
        engine: &winrapids_tiled::TiledEngine,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> tam_gpu::TamResult<Vec<Vec<f64>>> {
        use tam_gpu::ShaderLang;
        match engine.shader_lang() {
            ShaderLang::Cuda | ShaderLang::Cpu => {
                // Fused: one composite kernel, one GPU pass
                engine.run_raw_mixture(
                    &self.cuda_source(),
                    &self.mixture_cache_key(),
                    self.num_manifolds(),
                    a, b, m, n, k,
                )
            }
            _ => {
                // WGSL: N separate single-manifold kernels (no struct accumulator in WGSL yet)
                let mut results = Vec::with_capacity(self.num_manifolds());
                for (manifold, _) in &self.mixture.components {
                    let dist = engine.run(
                        &ManifoldDistanceOp::new(manifold.clone()),
                        a, b, m, n, k,
                    )?;
                    results.push(dist);
                }
                Ok(results)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_hash_eq() {
        assert_eq!(Manifold::Euclidean, Manifold::Euclidean);
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Manifold::Euclidean);
        assert!(set.contains(&Manifold::Euclidean));
    }

    #[test]
    fn poincare_same_curvature_eq() {
        let a = Manifold::poincare(-1.0);
        let b = Manifold::poincare(-1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn poincare_different_curvature_ne() {
        let a = Manifold::poincare(-1.0);
        let b = Manifold::poincare(-0.5);
        assert_ne!(a, b);
    }

    #[test]
    fn poincare_curvature_roundtrip() {
        let m = Manifold::poincare(-1.0);
        assert_eq!(m.curvature(), Some(-1.0));
    }

    #[test]
    fn sphere_radius_roundtrip() {
        let m = Manifold::sphere(2.0);
        assert_eq!(m.radius(), Some(2.0));
    }

    #[test]
    fn euclidean_dist_expr_contains_l2sq() {
        let expr = Manifold::Euclidean.tiled_dist_expr();
        assert!(expr.contains("a[k] - b[k]"), "expected L2Sq form, got: {expr}");
    }

    #[test]
    fn poincare_gradient_scale_contains_curvature() {
        let m = Manifold::poincare(-1.0);
        let expr = m.gradient_scale_expr();
        assert!(expr.contains("1.0"), "expected conformal factor expression, got: {expr}");
        assert!(expr.contains("x_norm_sq"), "expected x_norm_sq in expression, got: {expr}");
    }

    #[test]
    fn projection_needed() {
        assert!(!Manifold::Euclidean.needs_projection());
        assert!(Manifold::poincare(-1.0).needs_projection());
        assert!(Manifold::sphere(1.0).needs_projection());
    }

    #[test]
    fn bayesian_hashable() {
        use crate::intermediates::DataId;
        let prior = Manifold::Euclidean;
        let posterior_id = DataId(0xdeadbeef);
        let m = Manifold::Bayesian { prior: Box::new(prior), posterior_id };
        let name = m.name();
        assert!(name.contains("bayesian"), "got: {name}");
    }

    #[test]
    fn learned_hashable() {
        use crate::intermediates::DataId;
        let m = Manifold::Learned { params_id: DataId(0x1234) };
        let name = m.name();
        assert!(name.contains("learned"), "got: {name}");
    }

    #[test]
    fn display_names() {
        assert_eq!(Manifold::Euclidean.to_string(), "euclidean");
        assert!(Manifold::poincare(-1.0).to_string().contains("poincare"));
        assert!(Manifold::sphere(1.0).to_string().contains("sphere"));
    }

    #[test]
    fn euclidean_in_hashmap() {
        use std::collections::HashMap;
        let mut map: HashMap<Manifold, &str> = HashMap::new();
        map.insert(Manifold::Euclidean, "l2sq");
        map.insert(Manifold::poincare(-1.0), "hyperbolic");
        assert_eq!(map[&Manifold::Euclidean], "l2sq");
        assert_eq!(map[&Manifold::poincare(-1.0)], "hyperbolic");
    }

    // ── ManifoldMixture tests ────────────────────────────────────────────────

    fn make_distance_matrix(n: usize, data: Vec<f64>) -> Arc<DistanceMatrix> {
        Arc::new(DistanceMatrix {
            metric: crate::intermediates::Metric::L2Sq,
            n,
            data: Arc::new(data),
        })
    }

    #[test]
    fn mixture_uniform_weights_sum_to_one() {
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::poincare(-1.0),
            Manifold::sphere(1.0),
        ]);
        assert!((mix.total_weight() - 1.0).abs() < 1e-12);
        assert_eq!(mix.components.len(), 3);
        for (_, w) in &mix.components {
            assert!((w - 1.0/3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mixture_single_is_weight_one() {
        let mix = ManifoldMixture::single(Manifold::Euclidean);
        assert_eq!(mix.components.len(), 1);
        assert_eq!(mix.components[0].1, 1.0);
    }

    #[test]
    fn mixture_normalize_renormalizes() {
        let mix = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 3.0),
            (Manifold::poincare(-1.0), 1.0),
        ]).normalize();
        assert!((mix.total_weight() - 1.0).abs() < 1e-12);
        assert!((mix.components[0].1 - 0.75).abs() < 1e-12);
        assert!((mix.components[1].1 - 0.25).abs() < 1e-12);
    }

    #[test]
    fn mixture_combine_single_identity() {
        // combine with a single matrix and weight 1.0 should return the same data
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0]; // 3×3
        let m = make_distance_matrix(3, data.clone());
        let mix = ManifoldMixture::single(Manifold::Euclidean);
        let combined = mix.combine(&[m]);
        assert_eq!(combined, data);
    }

    #[test]
    fn mixture_combine_weighted_sum() {
        // Two 2×2 matrices: D1 = all 1s, D2 = all 2s
        // With weights 0.6 and 0.4: result = 0.6*1 + 0.4*2 = 1.4
        let n = 2;
        let d1 = make_distance_matrix(n, vec![1.0; 4]);
        let d2 = make_distance_matrix(n, vec![2.0; 4]);
        let mix = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.6),
            (Manifold::poincare(-1.0), 0.4),
        ]);
        let combined = mix.combine(&[d1, d2]);
        assert_eq!(combined.len(), 4);
        for &v in &combined {
            assert!((v - 1.4).abs() < 1e-12, "expected 1.4, got {v}");
        }
    }

    #[test]
    fn mixture_combine_non_symmetric_weights() {
        // 3×3 matrices: D1 = identity-ish, D2 = ones
        // Test that the combination respects row-major layout
        let n = 2;
        // D1[0,0]=0, D1[0,1]=5, D1[1,0]=5, D1[1,1]=0
        let d1 = make_distance_matrix(n, vec![0.0, 5.0, 5.0, 0.0]);
        // D2 = all 2s
        let d2 = make_distance_matrix(n, vec![2.0; 4]);
        let mix = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.5),
            (Manifold::sphere(1.0), 0.5),
        ]);
        let combined = mix.combine(&[d1, d2]);
        // [0,0] = 0.5*0 + 0.5*2 = 1.0
        assert!((combined[0] - 1.0).abs() < 1e-12, "combined[0,0]={}", combined[0]);
        // [0,1] = 0.5*5 + 0.5*2 = 3.5
        assert!((combined[1] - 3.5).abs() < 1e-12, "combined[0,1]={}", combined[1]);
    }

    #[test]
    fn mixture_mix_id_stable() {
        // Same mixture = same id
        let mix_a = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.7),
            (Manifold::poincare(-1.0), 0.3),
        ]);
        let mix_b = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.7),
            (Manifold::poincare(-1.0), 0.3),
        ]);
        assert_eq!(mix_a.mix_id(), mix_b.mix_id());
    }

    #[test]
    fn mixture_mix_id_differs_for_different_weights() {
        let mix_a = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.7),
            (Manifold::poincare(-1.0), 0.3),
        ]);
        let mix_b = ManifoldMixture::new(vec![
            (Manifold::Euclidean, 0.5),
            (Manifold::poincare(-1.0), 0.5),
        ]);
        assert_ne!(mix_a.mix_id(), mix_b.mix_id());
    }

    #[test]
    fn mixture_mix_id_differs_for_different_manifolds() {
        let mix_a = ManifoldMixture::new(vec![(Manifold::Euclidean, 1.0)]);
        let mix_b = ManifoldMixture::new(vec![(Manifold::poincare(-1.0), 1.0)]);
        assert_ne!(mix_a.mix_id(), mix_b.mix_id());
    }

    #[test]
    fn from_metric_euclidean() {
        assert_eq!(Manifold::from_metric(crate::intermediates::Metric::L2Sq), Manifold::Euclidean);
        assert_eq!(Manifold::from_metric(crate::intermediates::Metric::L2), Manifold::Euclidean);
        assert_eq!(Manifold::from_metric(crate::intermediates::Metric::Manhattan), Manifold::Euclidean);
    }

    #[test]
    fn from_metric_sphere() {
        let m = Manifold::from_metric(crate::intermediates::Metric::Cosine);
        assert!(matches!(m, Manifold::Sphere { .. }));
        assert_eq!(m.radius(), Some(1.0));
    }

    // ── ManifoldDistanceOp tests ─────────────────────────────────────────────

    #[test]
    fn manifold_distance_op_euclidean_via_tiled() {
        // 3 points in 2D: p0=(0,0), p1=(1,0), p2=(0,1)
        // L2Sq: d(p0,p1)=1, d(p0,p2)=1, d(p1,p2)=2
        //
        // TiledEngine::run(op, A, B, m, n, k) expects:
        //   A: M×K (n_points × n_dims)
        //   B: K×N (n_dims × n_points) — B is the TRANSPOSE of the point matrix
        //   C: M×N distance matrix
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        // A: 3 points × 2 dims, row-major
        let a = vec![0.0_f64, 0.0,   // p0 = (0,0)
                     1.0,     0.0,   // p1 = (1,0)
                     0.0,     1.0];  // p2 = (0,1)
        // B = A^T: 2 dims × 3 points, B[k,j] = point_j[k]
        let a_t = vec![0.0_f64, 1.0, 0.0,   // dim 0 of (p0, p1, p2)
                       0.0,     0.0, 1.0];  // dim 1 of (p0, p1, p2)

        let op = ManifoldDistanceOp::new(Manifold::Euclidean);
        let result = engine.run(&op, &a, &a_t, 3, 3, 2).unwrap();
        // result[i*3+j] = L2Sq(p_i, p_j)
        assert!((result[0] - 0.0).abs() < 1e-10, "d(0,0) = {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-10, "d(0,1) = {}", result[1]);
        assert!((result[2] - 1.0).abs() < 1e-10, "d(0,2) = {}", result[2]);
        assert!((result[4] - 0.0).abs() < 1e-10, "d(1,1) = {}", result[4]);
        assert!((result[5] - 2.0).abs() < 1e-10, "d(1,2) = {}", result[5]);
    }

    #[test]
    fn manifold_distance_op_sphere_cosine_unit_vectors() {
        // For unit vectors: cosine distance = 1 - dot(a, b)
        // p0 = (1,0), p1 = (0,1): dot=0, cosine_dist=1
        // p0 with itself: dot=1, cosine_dist=0
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let a = vec![1.0f64, 0.0,  0.0, 1.0]; // 2×2, unit vectors
        let op = ManifoldDistanceOp::new(Manifold::sphere(1.0));
        let result = engine.run(&op, &a, &a, 2, 2, 2).unwrap();
        // d(p0,p0) = 1 - 1 = 0
        assert!((result[0] - 0.0).abs() < 1e-10, "cosine_dist(p0,p0) = {}", result[0]);
        // d(p0,p1) = 1 - 0 = 1
        assert!((result[1] - 1.0).abs() < 1e-10, "cosine_dist(p0,p1) = {}", result[1]);
        // d(p1,p0) = 1 (symmetric)
        assert!((result[2] - 1.0).abs() < 1e-10, "cosine_dist(p1,p0) = {}", result[2]);
    }

    #[test]
    fn manifold_distance_op_params_key_unique_per_manifold() {
        let op_e = ManifoldDistanceOp::new(Manifold::Euclidean);
        let op_s = ManifoldDistanceOp::new(Manifold::sphere(1.0));
        let op_p = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        // Different params_key → different kernel cache entry
        assert_ne!(op_e.params_key(), op_s.params_key());
        assert_ne!(op_e.params_key(), op_p.params_key());
        assert_ne!(op_s.params_key(), op_p.params_key());
    }

    #[test]
    fn manifold_distance_op_accumulate_body_differs_by_geometry() {
        let euclidean_body = ManifoldDistanceOp::new(Manifold::Euclidean).cuda_accumulate_body();
        let sphere_body = ManifoldDistanceOp::new(Manifold::sphere(1.0)).cuda_accumulate_body();
        let poincare_body = ManifoldDistanceOp::new(Manifold::poincare(-1.0)).cuda_accumulate_body();
        assert!(euclidean_body.contains("diff * diff"), "euclidean: {euclidean_body}");
        assert!(sphere_body.contains("a_val * b_val"), "sphere: {sphere_body}");
        assert!(poincare_body.contains("sq_dist"), "poincare: {poincare_body}");
        assert!(poincare_body.contains("sq_norm_x"), "poincare: {poincare_body}");
        assert!(poincare_body.contains("sq_norm_y"), "poincare: {poincare_body}");
    }

    #[test]
    fn manifold_distance_op_poincare_acc_type_is_struct() {
        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        assert!(op.cuda_acc_type().contains("struct"), "expected struct, got: {}", op.cuda_acc_type());
        assert!(op.cuda_acc_type().contains("PoincareAcc"), "expected PoincareAcc");
        assert_eq!(op.acc_byte_size(), 24);
    }

    #[test]
    fn manifold_distance_op_poincare_wgsl_is_euclidean_fallback() {
        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        // WGSL fallback: scalar f32, Euclidean L2Sq body, plain acc extract
        assert_eq!(op.wgsl_acc_type(), "f32");
        assert_eq!(op.wgsl_identity(), "0.0");
        assert!(op.wgsl_accumulate_body().contains("diff * diff"), "{}", op.wgsl_accumulate_body());
        assert_eq!(op.wgsl_extract(), "acc");
    }

    #[test]
    fn manifold_distance_op_poincare_self_distance_is_zero() {
        // d(x, x) = 0 for any x in the Poincaré ball.
        // x = (0.3, 0.4) → ||x||² = 0.25, inside unit ball.
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        // A: 1 point × 2 dims = (0.3, 0.4)
        let a = vec![0.3_f64, 0.4];
        // B = A^T: 2 dims × 1 point
        let a_t = vec![0.3_f64, 0.4];
        let result = engine.run(&op, &a, &a_t, 1, 1, 2).unwrap();
        assert!(result[0].abs() < 1e-10, "d(x,x) should be 0, got {}", result[0]);
    }

    #[test]
    fn manifold_distance_op_poincare_distance_to_origin() {
        // d(origin, x) for x = (0.5, 0.0), kappa=1 (c=-1):
        // sq_dist = 0.25, sq_norm_x = 0.0, sq_norm_y = 0.25
        // denom = (1 - 0)(1 - 0.25) = 0.75
        // arg = 1 + 2*0.25/0.75 = 5/3
        // d = 2 * acosh(5/3)
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        // A: 1 point × 2 dims = origin (0, 0)
        let a   = vec![0.0_f64, 0.0];
        // B = x^T: 2 dims × 1 point = (0.5, 0.0)
        let b_t = vec![0.5_f64, 0.0];
        let result = engine.run(&op, &a, &b_t, 1, 1, 2).unwrap();

        let expected = 2.0 * (5.0_f64 / 3.0).acosh();
        assert!((result[0] - expected).abs() < 1e-8,
            "d(origin, (0.5,0)) = {}, expected {expected}", result[0]);
    }

    #[test]
    fn manifold_distance_op_poincare_symmetric() {
        // d(x, y) = d(y, x)
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        // A: 2 points × 2 dims: p0=(0.3,0.4), p1=(0.1,0.2)
        let a   = vec![0.3_f64, 0.4,  0.1, 0.2];
        // B = A^T: 2 dims × 2 points
        let a_t = vec![0.3_f64, 0.1,  0.4, 0.2];
        let result = engine.run(&op, &a, &a_t, 2, 2, 2).unwrap();
        // result[0*2+1] = d(p0, p1), result[1*2+0] = d(p1, p0)
        assert!((result[1] - result[2]).abs() < 1e-10,
            "d(p0,p1) = {} but d(p1,p0) = {}", result[1], result[2]);
        // Self-distances should be zero
        assert!(result[0].abs() < 1e-10, "d(p0,p0) = {}", result[0]);
        assert!(result[3].abs() < 1e-10, "d(p1,p1) = {}", result[3]);
    }

    #[test]
    fn manifold_distance_op_poincare_extract_contains_acosh() {
        let op = ManifoldDistanceOp::new(Manifold::poincare(-1.0));
        let extract = op.cuda_extract();
        assert!(extract.contains("acosh"), "extract: {extract}");
        assert!(extract.contains("sq_dist"), "extract: {extract}");
        assert!(extract.contains("sq_norm_x"), "extract: {extract}");
    }

    // ── SphericalGeodesic tests ─────────────────────────────────────────────

    #[test]
    fn spherical_geodesic_radius_accessor() {
        let m = Manifold::spherical_geodesic(2.0);
        assert_eq!(m.radius(), Some(2.0));
        assert!(m.needs_projection());
        assert!(m.name().contains("sphere_geodesic"));
    }

    #[test]
    fn spherical_geodesic_acc_type_is_struct() {
        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        assert!(op.cuda_acc_type().contains("SphGeoAcc"), "acc_type: {}", op.cuda_acc_type());
        assert!(op.cuda_acc_type().contains("sq_norm_x"), "acc_type: {}", op.cuda_acc_type());
        assert!(op.cuda_acc_type().contains("dot_prod"), "acc_type: {}", op.cuda_acc_type());
        assert_eq!(op.acc_byte_size(), 24);
    }

    #[test]
    fn spherical_geodesic_extract_contains_acos() {
        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        assert!(op.cuda_extract().contains("acos"), "extract: {}", op.cuda_extract());
        assert!(op.cuda_extract().contains("dot_prod"), "extract: {}", op.cuda_extract());
    }

    #[test]
    fn spherical_geodesic_self_distance_is_zero() {
        // d(x, x) = arccos(1) = 0 for any nonzero x.
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        let a   = vec![0.6_f64, 0.8];  // unit vector
        let a_t = vec![0.6_f64, 0.8];
        let result = engine.run(&op, &a, &a_t, 1, 1, 2).unwrap();
        assert!(result[0].abs() < 1e-10, "d(x,x) should be 0, got {}", result[0]);
    }

    #[test]
    fn spherical_geodesic_orthogonal_vectors_pi_over_2() {
        // d((1,0), (0,1)) = arccos(0) = π/2
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        let a   = vec![1.0_f64, 0.0,  0.0, 1.0]; // 2 unit vectors
        let a_t = vec![1.0_f64, 0.0,  0.0, 1.0]; // A^T
        let result = engine.run(&op, &a, &a_t, 2, 2, 2).unwrap();
        let half_pi = std::f64::consts::PI / 2.0;
        assert!((result[1] - half_pi).abs() < 1e-10, "d(p0,p1) = {}, expected π/2", result[1]);
        assert!(result[0].abs() < 1e-10, "d(p0,p0) = {}", result[0]);
    }

    #[test]
    fn spherical_geodesic_opposite_vectors_pi() {
        // d((1,0), (-1,0)) = arccos(-1) = π
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        let a   = vec![1.0_f64, 0.0];
        let b_t = vec![-1.0_f64, 0.0];
        let result = engine.run(&op, &a, &b_t, 1, 1, 2).unwrap();
        assert!((result[0] - std::f64::consts::PI).abs() < 1e-10,
            "d(x,-x) should be π, got {}", result[0]);
    }

    #[test]
    fn spherical_geodesic_non_unit_vectors_correct() {
        // Non-unit vectors: angle should be independent of scaling.
        // d(2*(1,0), 3*(0,1)) = arccos(0) = π/2
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let op = ManifoldDistanceOp::new(Manifold::spherical_geodesic(1.0));
        let a   = vec![2.0_f64, 0.0];
        let b_t = vec![0.0_f64, 3.0];
        let result = engine.run(&op, &a, &b_t, 1, 1, 2).unwrap();
        let half_pi = std::f64::consts::PI / 2.0;
        assert!((result[0] - half_pi).abs() < 1e-10,
            "d(2*(1,0), 3*(0,1)) should be π/2, got {}", result[0]);
    }

    #[test]
    fn mixture_op_sphere_geodesic_in_mixture() {
        // SphericalGeodesic in a mixture: orthogonal unit vectors → output ≈ π/2
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let a   = vec![1.0_f64, 0.0,  0.0, 1.0];
        let a_t = vec![1.0_f64, 0.0,  0.0, 1.0];
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::spherical_geodesic(1.0),
        ]);
        let op = ManifoldMixtureOp::new(mix);
        let results = op.run(&engine, &a, &a_t, 2, 2, 2).unwrap();

        assert_eq!(results.len(), 2);
        let geo = &results[1];
        let half_pi = std::f64::consts::PI / 2.0;
        assert!(geo[0].abs() < 1e-10, "d(p0,p0)={}", geo[0]);
        assert!((geo[1] - half_pi).abs() < 1e-10, "d(p0,p1)={} expected π/2", geo[1]);
    }

    // ── ManifoldMixtureOp tests ──────────────────────────────────────────────

    #[test]
    fn mixture_op_params_key_pipe_separated() {
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::poincare(-1.0),
            Manifold::sphere(1.0),
        ]);
        let op = ManifoldMixtureOp::new(mix);
        let key = op.params_key();
        assert!(key.contains('|'), "expected pipe separator: {key}");
        let parts: Vec<&str> = key.split('|').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].contains("euclidean"), "part[0]: {}", parts[0]);
        assert!(parts[1].contains("poincare"), "part[1]: {}", parts[1]);
        assert!(parts[2].contains("sphere"), "part[2]: {}", parts[2]);
    }

    #[test]
    fn mixture_op_num_manifolds() {
        let op2 = ManifoldMixtureOp::new(ManifoldMixture::uniform(vec![
            Manifold::Euclidean, Manifold::poincare(-1.0),
        ]));
        assert_eq!(op2.num_manifolds(), 2);

        let op3 = ManifoldMixtureOp::new(ManifoldMixture::uniform(vec![
            Manifold::Euclidean, Manifold::poincare(-1.0), Manifold::sphere(1.0),
        ]));
        assert_eq!(op3.num_manifolds(), 3);
    }

    #[test]
    fn mixture_op_cuda_source_has_mixture_acc_struct() {
        let mix = ManifoldMixture::uniform(vec![Manifold::Euclidean, Manifold::poincare(-1.0)]);
        let op = ManifoldMixtureOp::new(mix);
        let src = op.cuda_source();
        assert!(src.contains("MixtureAcc"), "source must define MixtureAcc: {src}");
        assert!(src.contains("sq_norm_x"), "source must have sq_norm_x");
        assert!(src.contains("sq_norm_y"), "source must have sq_norm_y");
        assert!(src.contains("dot_prod"), "source must have dot_prod");
        // Line 2 carries params key
        let line2 = src.lines().nth(1).unwrap_or("");
        assert!(line2.starts_with("// params:"), "line 2: {line2}");
    }

    #[test]
    fn mixture_op_single_euclidean_matches_distance_op() {
        // A single-manifold MixtureOp (Euclidean) must produce the same result
        // as ManifoldDistanceOp(Euclidean).
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        // 3 points in 2D
        let a   = vec![0.0_f64, 0.0,  1.0, 0.0,  0.0, 1.0];
        let a_t = vec![0.0_f64, 1.0, 0.0,  0.0, 0.0, 1.0];

        // Fused single-manifold run
        let mix_op = ManifoldMixtureOp::new(ManifoldMixture::single(Manifold::Euclidean));
        let fused = mix_op.run(&engine, &a, &a_t, 3, 3, 2).unwrap();
        assert_eq!(fused.len(), 1);

        // Reference single-op run
        let ref_op = ManifoldDistanceOp::new(Manifold::Euclidean);
        let reference = engine.run(&ref_op, &a, &a_t, 3, 3, 2).unwrap();

        assert_eq!(fused[0].len(), reference.len());
        for (i, (&got, &exp)) in fused[0].iter().zip(reference.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10, "[{i}] fused={got} ref={exp}");
        }
    }

    #[test]
    fn mixture_op_euclidean_output_correct() {
        // 2 points: p0=(0,0), p1=(1,0)  → L2Sq(p0,p1) = 1
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        let a   = vec![0.0_f64, 0.0,  1.0, 0.0]; // 2×2
        let a_t = vec![0.0_f64, 1.0,  0.0, 0.0]; // 2×2 (transpose)
        let mix = ManifoldMixture::uniform(vec![Manifold::Euclidean, Manifold::poincare(-1.0)]);
        let op = ManifoldMixtureOp::new(mix);
        let results = op.run(&engine, &a, &a_t, 2, 2, 2).unwrap();

        assert_eq!(results.len(), 2);
        // Euclidean output (index 0): diagonal = 0, off-diagonal = 1
        let euclid = &results[0];
        assert!(euclid[0].abs() < 1e-10, "L2Sq(p0,p0)={}", euclid[0]);
        assert!((euclid[1] - 1.0).abs() < 1e-10, "L2Sq(p0,p1)={}", euclid[1]);
        assert!((euclid[2] - 1.0).abs() < 1e-10, "L2Sq(p1,p0)={}", euclid[2]);
        assert!(euclid[3].abs() < 1e-10, "L2Sq(p1,p1)={}", euclid[3]);
    }

    #[test]
    fn mixture_op_poincare_self_distance_zero() {
        // All diagonal entries of the Poincaré output must be zero.
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);

        // 3 points inside the unit ball
        let a   = vec![0.3_f64, 0.4,   0.1, 0.2,   0.0, 0.5];
        let a_t = vec![0.3_f64, 0.1, 0.0,   0.4, 0.2, 0.5]; // A^T: 2×3
        let mix = ManifoldMixture::uniform(vec![Manifold::Euclidean, Manifold::poincare(-1.0)]);
        let op = ManifoldMixtureOp::new(mix);
        let results = op.run(&engine, &a, &a_t, 3, 3, 2).unwrap();

        let poincare = &results[1];
        // Diagonal: indices 0, 4, 8
        assert!(poincare[0].abs() < 1e-9, "d(p0,p0)={}", poincare[0]);
        assert!(poincare[4].abs() < 1e-9, "d(p1,p1)={}", poincare[4]);
        assert!(poincare[8].abs() < 1e-9, "d(p2,p2)={}", poincare[8]);
    }

    #[test]
    fn mixture_op_one_compiled_kernel_for_fused_run() {
        // The fused path must compile exactly one kernel (not N separate ones).
        use std::sync::Arc;
        use tam_gpu::detect;
        use winrapids_tiled::TiledEngine;
        let gpu = Arc::from(detect());
        let engine = TiledEngine::new(gpu);
        assert_eq!(engine.cache_len(), 0);

        let a   = vec![0.0_f64, 0.0];
        let a_t = vec![0.0_f64, 0.0];
        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean, Manifold::poincare(-1.0), Manifold::sphere(1.0),
        ]);
        let op = ManifoldMixtureOp::new(mix);
        op.run(&engine, &a, &a_t, 1, 1, 2).unwrap();

        // On CUDA/CPU: one compiled kernel for all three manifolds
        // On WGSL: one kernel per manifold (3 kernels)
        let cache_len = engine.cache_len();
        assert!(
            cache_len == 1 || cache_len == 3,
            "expected 1 (fused) or 3 (WGSL sequential), got {cache_len}"
        );
    }

    // ── CpuBackend manifold tests ───────────────────────────────────────────
    //
    // These prove manifold distances are mathematical properties of the
    // accumulate primitive, not artifacts of GPU computation — extending
    // the "structure not substrate" proof to composite geometry.

    fn cpu_engine() -> winrapids_tiled::TiledEngine {
        use std::sync::Arc;
        winrapids_tiled::TiledEngine::new(Arc::new(tam_gpu::CpuBackend::new()))
    }

    #[test]
    fn mixture_on_cpu_euclidean_matches_single_op() {
        let engine = cpu_engine();
        let a   = vec![0.0_f64, 0.0,  1.0, 0.0,  0.0, 1.0];
        let a_t = vec![0.0_f64, 1.0, 0.0,  0.0, 0.0, 1.0];

        let mix_op = ManifoldMixtureOp::new(ManifoldMixture::single(Manifold::Euclidean));
        let fused = mix_op.run(&engine, &a, &a_t, 3, 3, 2).unwrap();
        assert_eq!(fused.len(), 1);

        let ref_op = ManifoldDistanceOp::new(Manifold::Euclidean);
        let reference = engine.run(&ref_op, &a, &a_t, 3, 3, 2).unwrap();

        for (i, (&got, &exp)) in fused[0].iter().zip(reference.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10, "[{i}] fused={got} ref={exp}");
        }
    }

    #[test]
    fn mixture_on_cpu_three_manifolds() {
        let engine = cpu_engine();

        // 2 points inside unit ball: p0=(0.3,0.4), p1=(-0.2,0.1)
        // Both non-zero (meaningful cosine) and ||p||<1 (valid Poincaré)
        let a   = vec![0.3_f64, 0.4,  -0.2, 0.1];
        let a_t = vec![0.3_f64, -0.2,  0.4, 0.1]; // transposed: 2×2

        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::poincare(-1.0),
            Manifold::sphere(1.0),
        ]);
        let op = ManifoldMixtureOp::new(mix);
        let results = op.run(&engine, &a, &a_t, 2, 2, 2).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 4);

        // Self-distances are zero for Euclidean and Poincaré (diagonal: 0, 3)
        for (mk, name) in [(0, "Euclidean"), (1, "Poincaré")] {
            assert!(results[mk][0].abs() < 1e-9,
                "{name}: d(p0,p0) = {}", results[mk][0]);
            assert!(results[mk][3].abs() < 1e-9,
                "{name}: d(p1,p1) = {}", results[mk][3]);
        }
        // Sphere: d(p,p) = 1 - dot(p,p) = 1 - ||p||². Zero only for unit vectors.
        let norm_p0_sq = 0.3 * 0.3 + 0.4 * 0.4; // 0.25
        assert!((results[2][0] - (1.0 - norm_p0_sq)).abs() < 1e-10,
            "Sphere d(p0,p0) = {}, expected {}", results[2][0], 1.0 - norm_p0_sq);

        // Symmetry d(p0,p1) = d(p1,p0) for all manifolds
        for (mk, name) in [(0, "Euclidean"), (1, "Poincaré"), (2, "Sphere")] {
            assert!((results[mk][1] - results[mk][2]).abs() < 1e-9,
                "{name}: d(p0,p1)={} ≠ d(p1,p0)={}", results[mk][1], results[mk][2]);
        }

        // Euclidean: ||p0-p1||² = (0.5)² + (0.3)² = 0.34
        assert!((results[0][1] - 0.34).abs() < 1e-10,
            "Euclidean d(p0,p1) = {}, expected 0.34", results[0][1]);

        // Poincaré distance > Euclidean L2 (hyperbolic stretching)
        assert!(results[1][1] > results[0][1].sqrt(),
            "Poincaré {} should exceed Euclidean L2 {}", results[1][1], results[0][1].sqrt());
    }

    #[test]
    fn mixture_on_cpu_matches_detect_backend() {
        let cpu_eng = cpu_engine();
        let det_eng = winrapids_tiled::TiledEngine::new(tam_gpu::detect());

        let a   = vec![0.1_f64, 0.2,  0.3, 0.1,  0.2, 0.3];
        let a_t = vec![0.1_f64, 0.3, 0.2,  0.2, 0.1, 0.3];

        let mix = ManifoldMixture::uniform(vec![
            Manifold::Euclidean,
            Manifold::poincare(-1.0),
        ]);
        let op = ManifoldMixtureOp::new(mix);

        let cpu_res = op.run(&cpu_eng, &a, &a_t, 3, 3, 2).unwrap();
        let det_res = op.run(&det_eng, &a, &a_t, 3, 3, 2).unwrap();

        assert_eq!(cpu_res.len(), det_res.len());
        for mk in 0..cpu_res.len() {
            for (i, (&c, &d)) in cpu_res[mk].iter().zip(det_res[mk].iter()).enumerate() {
                assert!((c - d).abs() < 1e-9,
                    "manifold {mk}, index {i}: cpu={c}, detect={d}");
            }
        }
    }
}
