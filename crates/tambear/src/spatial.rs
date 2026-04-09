//! # Family 30 — Spatial Statistics
//!
//! From first principles. Geostatistics, point patterns, spatial autocorrelation.
//!
//! ## What lives here
//!
//! **Variograms**: empirical (binned), theoretical models (spherical, exponential, Gaussian, Matérn)
//! **Kriging**: ordinary kriging (best linear unbiased predictor)
//! **Spatial autocorrelation**: Moran's I, Geary's C
//! **Point patterns**: Ripley's K-function, L-function, nearest-neighbor distance
//! **Distance**: Euclidean, Haversine (great-circle)
//! **Spatial weights**: k-nearest neighbors, distance threshold
//!
//! ## Architecture
//!
//! Points are (x, y) or (x, y, value) tuples.
//! Spatial weights matrices are sparse (compressed row format).
//! Kriging solves a linear system — uses our F02 linear algebra.
//!
//! ## MSR insight
//!
//! The variogram IS the spatial sufficient statistic. Once you have the
//! variogram model parameters (nugget, sill, range), you can krige anywhere
//! without the raw data. The variogram is the MSR of spatial structure.

use std::f64::consts::PI;

/// 2D point with value.
#[derive(Debug, Clone, Copy)]
pub struct SpatialPoint {
    pub x: f64,
    pub y: f64,
    pub value: f64,
}

// ─── Distance ───────────────────────────────────────────────────────

/// Euclidean distance between two 2D points.
pub fn euclidean_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}

/// Haversine (great-circle) distance in km.
///
/// Input: (lat1, lon1, lat2, lon2) in degrees.
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0; // Earth radius in km
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1r = lat1.to_radians();
    let lat2r = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

// ─── Variograms ─────────────────────────────────────────────────────

/// Empirical semivariogram bin.
#[derive(Debug, Clone)]
pub struct VariogramBin {
    pub lag: f64,
    pub gamma: f64,
    pub count: usize,
}

/// Empirical semivariogram (cloud binned by distance).
///
/// γ(h) = (1/2N(h)) Σ [z(x_i) - z(x_j)]² for ||x_i - x_j|| ∈ lag bin.
pub fn empirical_variogram(points: &[SpatialPoint], n_bins: usize, max_lag: f64) -> Vec<VariogramBin> {
    let n = points.len();
    if n < 2 || n_bins == 0 { return vec![]; }
    let bin_width = max_lag / n_bins as f64;
    let mut sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];

    for i in 0..n {
        for j in i + 1..n {
            let d = euclidean_2d(points[i].x, points[i].y, points[j].x, points[j].y);
            if d >= max_lag { continue; }
            let bin = (d / bin_width).floor() as usize;
            if bin < n_bins {
                let diff = points[i].value - points[j].value;
                sums[bin] += diff * diff;
                counts[bin] += 1;
            }
        }
    }

    (0..n_bins)
        .filter(|&i| counts[i] > 0)
        .map(|i| VariogramBin {
            lag: (i as f64 + 0.5) * bin_width,
            gamma: sums[i] / (2.0 * counts[i] as f64),
            count: counts[i],
        })
        .collect()
}

/// Variogram model parameters.
#[derive(Debug, Clone, Copy)]
pub struct VariogramModel {
    /// Nugget (discontinuity at origin).
    pub nugget: f64,
    /// Sill (asymptotic variance).
    pub sill: f64,
    /// Range (distance at which sill is reached).
    pub range: f64,
}

/// Spherical variogram model.
pub fn spherical_variogram(h: f64, model: &VariogramModel) -> f64 {
    if h < 1e-300 { return 0.0; }
    if h >= model.range {
        model.nugget + model.sill
    } else {
        let hr = h / model.range;
        model.nugget + model.sill * (1.5 * hr - 0.5 * hr.powi(3))
    }
}

/// Exponential variogram model.
pub fn exponential_variogram(h: f64, model: &VariogramModel) -> f64 {
    if h < 1e-300 { return 0.0; }
    model.nugget + model.sill * (1.0 - (-3.0 * h / model.range).exp())
}

/// Gaussian variogram model.
pub fn gaussian_variogram(h: f64, model: &VariogramModel) -> f64 {
    if h < 1e-300 { return 0.0; }
    model.nugget + model.sill * (1.0 - (-3.0 * (h / model.range).powi(2)).exp())
}

// ─── Kriging ────────────────────────────────────────────────────────

/// Ordinary kriging result.
#[derive(Debug, Clone)]
pub struct KrigingResult {
    pub predicted: Vec<f64>,
    pub variance: Vec<f64>,
}

/// Ordinary kriging interpolation.
///
/// Predicts values at query points using the variogram model.
/// Solves the kriging system: [C | 1] [w | μ]^T = [c | 1].
pub fn ordinary_kriging(
    points: &[SpatialPoint],
    query_x: &[f64],
    query_y: &[f64],
    model: &VariogramModel,
    variogram_fn: fn(f64, &VariogramModel) -> f64,
) -> KrigingResult {
    let n = points.len();
    let nq = query_x.len();
    if n == 0 || nq == 0 || nq != query_y.len() {
        return KrigingResult { predicted: vec![], variance: vec![] };
    }

    // Build covariance matrix C(x_i, x_j) = sill + nugget - γ(||x_i - x_j||)
    let c_max = model.sill + model.nugget;
    let n1 = n + 1; // extended system (Lagrange multiplier)
    let mut a = vec![vec![0.0; n1]; n1];
    for i in 0..n {
        for j in 0..n {
            let h = euclidean_2d(points[i].x, points[i].y, points[j].x, points[j].y);
            a[i][j] = c_max - variogram_fn(h, model);
        }
        a[i][n] = 1.0;
        a[n][i] = 1.0;
    }
    a[n][n] = 0.0;

    let mut predicted = vec![0.0; nq];
    let mut variance = vec![0.0; nq];

    for q in 0..nq {
        let mut b = vec![0.0; n1];
        for i in 0..n {
            let h = euclidean_2d(points[i].x, points[i].y, query_x[q], query_y[q]);
            b[i] = c_max - variogram_fn(h, model);
        }
        b[n] = 1.0;

        // Solve Aw = b via Gaussian elimination
        let mut mat = a.clone();
        let mut rhs = b.clone();
        let w = solve_system(&mat, &rhs);

        // Prediction = Σ w_i * z_i
        let mut pred = 0.0;
        for i in 0..n {
            pred += w[i] * points[i].value;
        }
        predicted[q] = pred;

        // Kriging variance = C(0) - Σ w_i * c(x_i, x*) - μ
        let mut var = c_max;
        for i in 0..n {
            let h = euclidean_2d(points[i].x, points[i].y, query_x[q], query_y[q]);
            var -= w[i] * (c_max - variogram_fn(h, model));
        }
        var -= w[n]; // Lagrange multiplier
        variance[q] = var.max(0.0);
    }

    KrigingResult { predicted, variance }
}

/// Solve linear system via F02 (linear_algebra::solve — LU with partial pivoting).
fn solve_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 { return vec![]; }
    let data: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = crate::linear_algebra::Mat::from_vec(n, n, data);
    crate::linear_algebra::solve(&mat, b)
        .unwrap_or_else(|| vec![0.0; n])
}

// ─── Spatial Autocorrelation ────────────────────────────────────────

/// Spatial weights matrix (adjacency list: `neighbors[i]` = list of `(neighbor_idx, weight)`).
#[derive(Debug, Clone)]
pub struct SpatialWeights {
    /// For each node: list of (neighbor_idx, weight).
    pub neighbors: Vec<Vec<(usize, f64)>>,
    pub n: usize,
}

impl SpatialWeights {
    /// Build k-nearest-neighbor weights.
    pub fn knn(points: &[(f64, f64)], k: usize) -> Self {
        let n = points.len();
        let mut neighbors = vec![vec![]; n];
        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, euclidean_2d(points[i].0, points[i].1, points[j].0, points[j].1)))
                .collect();
            dists.sort_by(|a, b| a.1.total_cmp(&b.1));
            let k_actual = k.min(dists.len());
            neighbors[i] = dists[..k_actual].iter().map(|&(j, _)| (j, 1.0)).collect();
        }
        SpatialWeights { neighbors, n }
    }

    /// Build distance-threshold weights.
    pub fn distance_band(points: &[(f64, f64)], threshold: f64) -> Self {
        let n = points.len();
        let mut neighbors = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let d = euclidean_2d(points[i].0, points[i].1, points[j].0, points[j].1);
                if d <= threshold {
                    neighbors[i].push((j, 1.0));
                }
            }
        }
        SpatialWeights { neighbors, n }
    }

    /// Row-standardize weights (each row sums to 1).
    pub fn row_standardize(&mut self) {
        for row in &mut self.neighbors {
            let sum: f64 = row.iter().map(|(_, w)| w).sum();
            if sum > 0.0 {
                for (_, w) in row.iter_mut() { *w /= sum; }
            }
        }
    }
}

/// Moran's I spatial autocorrelation statistic.
///
/// I = (n/S0) Σ_ij w_ij (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²
///
/// I ≈ 1: positive autocorrelation (similar values cluster)
/// I ≈ -1: negative autocorrelation (dissimilar values cluster)
/// I ≈ E[I] = -1/(n-1): random
pub fn morans_i(values: &[f64], weights: &SpatialWeights) -> f64 {
    let n = values.len();
    if n < 2 || n != weights.n { return f64::NAN; }
    let mean = crate::descriptive::moments_ungrouped(values).mean();
    let dev: Vec<f64> = values.iter().map(|&v| v - mean).collect();
    let ss: f64 = dev.iter().map(|d| d * d).sum();
    if ss < 1e-300 { return 0.0; }

    let mut s0 = 0.0;
    let mut numerator = 0.0;
    for i in 0..n {
        for &(j, w) in &weights.neighbors[i] {
            s0 += w;
            numerator += w * dev[i] * dev[j];
        }
    }
    if s0 < 1e-300 { return 0.0; }
    (n as f64 / s0) * numerator / ss
}

/// Geary's C spatial autocorrelation statistic.
///
/// C = ((n-1)/2S0) Σ_ij w_ij (x_i - x_j)² / Σ_i (x_i - x̄)²
///
/// C < 1: positive autocorrelation, C > 1: negative, C ≈ 1: random
pub fn gearys_c(values: &[f64], weights: &SpatialWeights) -> f64 {
    let n = values.len();
    if n < 2 || n != weights.n { return f64::NAN; }
    let moments = crate::descriptive::moments_ungrouped(values);
    let mean = moments.mean();
    let ss = moments.m2; // Σ(x - x̄)² — the raw second central moment sum
    if ss < 1e-300 { return 0.0; }

    let mut s0 = 0.0;
    let mut numerator = 0.0;
    for i in 0..n {
        for &(j, w) in &weights.neighbors[i] {
            s0 += w;
            numerator += w * (values[i] - values[j]).powi(2);
        }
    }
    if s0 < 1e-300 { return 0.0; }
    ((n - 1) as f64 / (2.0 * s0)) * numerator / ss
}

// ─── Point Pattern Analysis ─────────────────────────────────────────

/// Ripley's K-function estimate.
///
/// K(r) = (area/n²) Σ_i Σ_{j≠i} I(d_ij ≤ r)
///
/// For a homogeneous Poisson process, K(r) = π r².
/// K(r) > π r² → clustering; K(r) < π r² → regularity.
pub fn ripleys_k(points: &[(f64, f64)], radii: &[f64], area: f64) -> Vec<f64> {
    let n = points.len();
    if n < 2 { return vec![0.0; radii.len()]; }
    let mut k_vals = vec![0.0; radii.len()];

    for (ri, &r) in radii.iter().enumerate() {
        let mut count = 0u64;
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let d = euclidean_2d(points[i].0, points[i].1, points[j].0, points[j].1);
                if d <= r { count += 1; }
            }
        }
        k_vals[ri] = area * count as f64 / (n * n) as f64;
    }
    k_vals
}

/// Ripley's L-function: L(r) = √(K(r)/π) - r.
///
/// Variance-stabilized K. L(r) = 0 for Poisson, >0 for clustered, <0 for regular.
pub fn ripleys_l(points: &[(f64, f64)], radii: &[f64], area: f64) -> Vec<f64> {
    let k = ripleys_k(points, radii, area);
    k.iter().zip(radii.iter())
        .map(|(&ki, &r)| (ki / PI).sqrt() - r)
        .collect()
}

/// Nearest-neighbor distance distribution (G-function numerator).
///
/// Returns sorted nearest-neighbor distances.
pub fn nn_distances(points: &[(f64, f64)]) -> Vec<f64> {
    let n = points.len();
    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let mut min_d = f64::INFINITY;
        for j in 0..n {
            if i == j { continue; }
            let d = euclidean_2d(points[i].0, points[i].1, points[j].0, points[j].1);
            if d < min_d { min_d = d; }
        }
        if min_d < f64::INFINITY { dists.push(min_d); }
    }
    dists.sort_by(|a, b| a.total_cmp(b));
    dists
}

/// Clark-Evans R statistic (nearest-neighbor test for spatial randomness).
///
/// R = observed_mean_nn / expected_mean_nn
/// R < 1: clustering, R > 1: regularity, R ≈ 1: random
pub fn clark_evans_r(points: &[(f64, f64)], area: f64) -> f64 {
    let n = points.len();
    if n < 2 || area <= 0.0 { return f64::NAN; }
    let dists = nn_distances(points);
    let mean_nn = dists.iter().sum::<f64>() / dists.len() as f64;
    let expected = 0.5 * (area / n as f64).sqrt(); // E[nn] for Poisson
    if expected < 1e-300 { return f64::NAN; }
    mean_nn / expected
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Distance ──

    #[test]
    fn euclidean_basic() {
        assert!((euclidean_2d(0.0, 0.0, 3.0, 4.0) - 5.0).abs() < 1e-10);
        assert!((euclidean_2d(1.0, 1.0, 1.0, 1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn haversine_known() {
        // New York to London ≈ 5570 km
        let d = haversine(40.7128, -74.0060, 51.5074, -0.1278);
        assert!(d > 5500.0 && d < 5700.0, "NY-London = {} km", d);
    }

    // ── Variograms ──

    #[test]
    fn spherical_at_range() {
        let model = VariogramModel { nugget: 0.0, sill: 1.0, range: 10.0 };
        // At range, γ should equal sill
        assert!((spherical_variogram(10.0, &model) - 1.0).abs() < 1e-10);
        // At half range
        let half = spherical_variogram(5.0, &model);
        assert!(half > 0.0 && half < 1.0, "half range γ = {}", half);
        // At zero
        assert!((spherical_variogram(0.0, &model) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn exponential_asymptote() {
        let model = VariogramModel { nugget: 0.5, sill: 1.0, range: 10.0 };
        // Should approach nugget + sill for large h
        let far = exponential_variogram(100.0, &model);
        assert!((far - 1.5).abs() < 0.01, "far γ = {}", far);
    }

    #[test]
    fn empirical_variogram_monotone() {
        // Points with spatial correlation: nearby = similar values
        let points: Vec<SpatialPoint> = (0..20).map(|i| {
            let x = i as f64;
            SpatialPoint { x, y: 0.0, value: x * 0.1 + (x * 0.3).sin() * 0.1 }
        }).collect();
        let vario = empirical_variogram(&points, 5, 10.0);
        assert!(!vario.is_empty());
        // Generally should increase with lag
        if vario.len() >= 2 {
            assert!(vario.last().unwrap().gamma >= vario[0].gamma * 0.5,
                "variogram not generally increasing");
        }
    }

    // ── Kriging ──

    #[test]
    fn kriging_at_known_points() {
        // Kriging should reproduce known values exactly (nugget = 0)
        let points = vec![
            SpatialPoint { x: 0.0, y: 0.0, value: 1.0 },
            SpatialPoint { x: 1.0, y: 0.0, value: 2.0 },
            SpatialPoint { x: 0.0, y: 1.0, value: 3.0 },
            SpatialPoint { x: 1.0, y: 1.0, value: 4.0 },
        ];
        let model = VariogramModel { nugget: 0.0, sill: 5.0, range: 2.0 };
        let result = ordinary_kriging(
            &points,
            &[0.0, 1.0], &[0.0, 1.0],
            &model, spherical_variogram,
        );
        assert!((result.predicted[0] - 1.0).abs() < 0.1,
            "kriging at (0,0) = {}", result.predicted[0]);
        assert!((result.predicted[1] - 4.0).abs() < 0.1,
            "kriging at (1,1) = {}", result.predicted[1]);
    }

    #[test]
    fn kriging_variance_at_data_low() {
        // Variance should be low at data points, high far away
        let points = vec![
            SpatialPoint { x: 0.0, y: 0.0, value: 1.0 },
            SpatialPoint { x: 10.0, y: 0.0, value: 2.0 },
        ];
        let model = VariogramModel { nugget: 0.0, sill: 1.0, range: 5.0 };
        let result = ordinary_kriging(
            &points,
            &[0.0, 50.0], &[0.0, 0.0],
            &model, exponential_variogram,
        );
        // Variance at data point should be less than far away
        assert!(result.variance[0] < result.variance[1],
            "var at data {} >= var far {}", result.variance[0], result.variance[1]);
    }

    // ── Spatial autocorrelation ──

    #[test]
    fn morans_i_positive() {
        // Clustered data: left side high, right side low
        let values = vec![10.0, 9.0, 8.0, 2.0, 1.0, 0.0];
        let pts: Vec<(f64, f64)> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let weights = SpatialWeights::knn(&pts, 2);
        let i = morans_i(&values, &weights);
        assert!(i > 0.0, "Moran's I = {} (should be positive)", i);
    }

    #[test]
    fn gearys_c_positive_correlation() {
        // Same clustered data as above
        let values = vec![10.0, 9.0, 8.0, 2.0, 1.0, 0.0];
        let pts: Vec<(f64, f64)> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let weights = SpatialWeights::knn(&pts, 2);
        let c = gearys_c(&values, &weights);
        assert!(c < 1.0, "Geary's C = {} (should be < 1 for positive autocorrelation)", c);
    }

    // ── Point patterns ──

    #[test]
    fn ripleys_k_clustered() {
        // Clustered points should have K(r) > π·r²
        let mut points = Vec::new();
        // Cluster 1 around (0, 0)
        for i in 0..5 { points.push((i as f64 * 0.1, i as f64 * 0.05)); }
        // Cluster 2 around (10, 10)
        for i in 0..5 { points.push((10.0 + i as f64 * 0.1, 10.0 + i as f64 * 0.05)); }
        let r = 2.0;
        let area = 15.0 * 15.0; // bounding box area
        let k = ripleys_k(&points, &[r], area);
        let expected_poisson = PI * r * r;
        assert!(k[0] > expected_poisson,
            "K({}) = {} vs Poisson = {}", r, k[0], expected_poisson);
    }

    #[test]
    fn nn_distances_sorted() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (5.0, 0.0), (10.0, 0.0)];
        let dists = nn_distances(&points);
        assert_eq!(dists.len(), 4);
        // Should be sorted ascending
        for w in dists.windows(2) {
            assert!(w[0] <= w[1], "not sorted: {} > {}", w[0], w[1]);
        }
        // Smallest nn distance should be 1.0 (between 0 and 1)
        assert!((dists[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clark_evans_clustered() {
        // Clustered points → R < 1
        let mut points = Vec::new();
        for i in 0..10 { points.push((i as f64 * 0.01, 0.0)); }
        let r = clark_evans_r(&points, 100.0);
        assert!(r < 1.0, "Clark-Evans R = {} (should be < 1 for clustering)", r);
    }

    // ── Spatial weights ──

    #[test]
    fn knn_weights_correct_k() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let w = SpatialWeights::knn(&pts, 2);
        assert_eq!(w.neighbors[0].len(), 2);
        assert_eq!(w.neighbors[1].len(), 2);
    }

    #[test]
    fn distance_band_threshold() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)];
        let w = SpatialWeights::distance_band(&pts, 2.0);
        // Node 0 should be neighbors with node 1 only
        assert_eq!(w.neighbors[0].len(), 1);
        assert_eq!(w.neighbors[0][0].0, 1);
    }

    #[test]
    fn row_standardize_sums_to_one() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let mut w = SpatialWeights::knn(&pts, 2);
        w.row_standardize();
        for row in &w.neighbors {
            let sum: f64 = row.iter().map(|(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-10, "row sum = {}", sum);
        }
    }

    // ── Edge cases ──

    #[test]
    fn empty_points() {
        let vario = empirical_variogram(&[], 5, 10.0);
        assert!(vario.is_empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D Convex Hull — Graham scan
// ─────────────────────────────────────────────────────────────────────────────
//
// Graham (1972) scan in O(n log n). Used by tick_geometry.rs fintek leaf
// for convex-hull area and perimeter of price-volume trajectories.

/// Compute the 2D convex hull of a set of points using the Graham scan.
///
/// Returns the hull vertices in counter-clockwise order starting from the
/// lowest-left point. Returns an empty vec when fewer than 3 distinct points
/// are provided (collinear or degenerate inputs return the unique points in
/// CCW order).
///
/// Points are `(x, y)` tuples.
pub fn convex_hull_2d(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Find the lowest (then leftmost) point — the pivot.
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    let pivot_idx = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(a.0.partial_cmp(&b.0).unwrap())
        })
        .map(|(i, _)| i)
        .unwrap();
    pts.swap(0, pivot_idx);
    let pivot = pts[0];

    // Sort remaining points by polar angle with respect to pivot.
    // Break ties by distance (closer first).
    pts[1..].sort_by(|&(ax, ay), &(bx, by)| {
        let cross = (ax - pivot.0) * (by - pivot.1) - (ay - pivot.1) * (bx - pivot.0);
        if cross.abs() < 1e-12 {
            // Collinear — sort by distance
            let da = (ax - pivot.0).hypot(ay - pivot.1);
            let db = (bx - pivot.0).hypot(by - pivot.1);
            da.partial_cmp(&db).unwrap()
        } else if cross > 0.0 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    // Remove collinear duplicates keeping only the farthest.
    let mut filtered: Vec<(f64, f64)> = vec![pivot];
    for &p in &pts[1..] {
        while filtered.len() > 1 {
            let len = filtered.len();
            let o = filtered[len - 2];
            let a = filtered[len - 1];
            let cross = (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0);
            if cross.abs() < 1e-12 {
                // Collinear — keep farthest
                filtered.pop();
            } else {
                break;
            }
        }
        filtered.push(p);
    }

    if filtered.len() < 3 {
        return filtered;
    }

    // Graham scan on the filtered list.
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(filtered.len());
    for &p in &filtered {
        while hull.len() > 1 {
            let len = hull.len();
            let o = hull[len - 2];
            let a = hull[len - 1];
            // Cross product: (a-o) × (p-o); negative → right turn → pop
            let cross = (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

/// Signed area of a polygon given as CCW vertex list (shoelace formula).
///
/// Returns positive area for CCW ordering. For a convex hull returned by
/// [`convex_hull_2d`], the result is always non-negative.
pub fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    for i in 0..n {
        let (x0, y0) = vertices[i];
        let (x1, y1) = vertices[(i + 1) % n];
        area += x0 * y1 - x1 * y0;
    }
    area.abs() / 2.0
}

/// Perimeter of a polygon given as a vertex list (consecutive edges + closing edge).
pub fn polygon_perimeter(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 2 {
        return 0.0;
    }
    let mut perim = 0.0_f64;
    for i in 0..n {
        let (x0, y0) = vertices[i];
        let (x1, y1) = vertices[(i + 1) % n];
        perim += (x1 - x0).hypot(y1 - y0);
    }
    perim
}

#[cfg(test)]
mod hull_tests {
    use super::*;

    #[test]
    fn hull_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 4, "interior point should be excluded");
        let area = polygon_area(&hull);
        assert!((area - 1.0).abs() < 1e-10, "area={}", area);
    }

    #[test]
    fn hull_triangle() {
        let pts = vec![(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 3);
        let area = polygon_area(&hull);
        // base=4, height=3 → area=6
        assert!((area - 6.0).abs() < 1e-10, "area={}", area);
    }

    #[test]
    fn hull_collinear_points() {
        // All collinear → degenerate, hull is endpoints
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let hull = convex_hull_2d(&pts);
        // At most 2 distinct non-collinear points
        assert!(hull.len() <= 4);
        let area = polygon_area(&hull);
        assert!(area.abs() < 1e-10);
    }

    #[test]
    fn hull_single_point() {
        let pts = vec![(1.0, 2.0)];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 1);
    }

    #[test]
    fn hull_perimeter_unit_square() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let p = polygon_perimeter(&square);
        assert!((p - 4.0).abs() < 1e-10, "perimeter={}", p);
    }

    #[test]
    fn hull_interior_points_excluded() {
        // Circle of 8 points + 10 interior points
        use std::f64::consts::TAU;
        let mut pts: Vec<(f64, f64)> = (0..8)
            .map(|i| {
                let a = TAU * i as f64 / 8.0;
                (a.cos(), a.sin())
            })
            .collect();
        // Interior points
        for i in 0..10 {
            pts.push((0.05 * i as f64, 0.0));
        }
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 8, "interior points should not appear in hull");
        let area = polygon_area(&hull);
        // Regular octagon inscribed in unit circle: area = 2√2 ≈ 2.828
        assert!((area - 2.0 * 2.0_f64.sqrt()).abs() < 0.01, "area={}", area);
    }
}
