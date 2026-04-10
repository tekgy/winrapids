//! Robust statistics — estimators resistant to outliers.
//!
//! ## Architecture
//!
//! Robust estimators replace the arithmetic mean and variance with estimators
//! that have high breakdown points (up to 50%) — meaning they remain valid
//! even when half the data is corrupted.
//!
//! Most methods here compose from existing primitives:
//! - MAD (already in descriptive.rs) as the workhorse scale estimator
//! - Sorting (for trimmed/winsorized methods) — the one place Tam sorts
//! - Iterative reweighting via scatter (for M-estimators)
//!
//! ## Methods
//!
//! **Location**: Huber M-estimator, bisquare (Tukey) M-estimator, Hampel M-estimator
//! **Scale**: MAD, Qn estimator, Sn estimator, tau-scale
//! **Regression**: LTS (least trimmed squares), LMS (least median squares)
//! **Multivariate**: MCD (minimum covariance determinant)
//! **Trimming**: Trimmed mean/variance (already in descriptive), interquartile mean
//!
//! ## .tbs integration
//!
//! ```text
//! huber_mean(data, k=1.345)        # Huber M-estimate of location
//! bisquare_mean(data, k=4.685)     # Tukey bisquare M-estimate
//! mad(data)                        # median absolute deviation (already in descriptive)
//! ```

use crate::descriptive::{sorted_nan_free, median};

// ═══════════════════════════════════════════════════════════════════════════
// Weight functions (ψ and w functions for M-estimators)
// ═══════════════════════════════════════════════════════════════════════════

/// Huber weight function: w(u) = min(1, k/|u|).
///
/// The Huber estimator is the most common M-estimator. It's like the mean
/// for small residuals and like the median for large ones.
/// k = 1.345 gives 95% efficiency at the normal distribution.
#[inline]
pub fn huber_weight(u: f64, k: f64) -> f64 {
    if u.abs() <= k { 1.0 } else { k / u.abs() }
}

/// Tukey bisquare weight function: w(u) = (1 - (u/k)²)² for |u| ≤ k, 0 otherwise.
///
/// Hard redescender: completely rejects outliers beyond k.
/// k = 4.685 gives 95% efficiency at the normal.
#[inline]
pub fn bisquare_weight(u: f64, k: f64) -> f64 {
    if u.abs() <= k {
        let t = 1.0 - (u / k).powi(2);
        t * t
    } else {
        0.0
    }
}

/// Hampel weight function: three-part redescender.
///
/// w(u) = 1 for |u| ≤ a
/// w(u) = a/|u| for a < |u| ≤ b
/// w(u) = a(c-|u|) / (|u|(c-b)) for b < |u| ≤ c
/// w(u) = 0 for |u| > c
///
/// Default thresholds: a=1.7, b=3.4, c=8.5 (from MASS library).
#[inline]
pub fn hampel_weight(u: f64, a: f64, b: f64, c: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u <= a {
        1.0
    } else if abs_u <= b {
        a / abs_u
    } else if abs_u <= c {
        a * (c - abs_u) / (abs_u * (c - b))
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// M-estimators of location
// ═══════════════════════════════════════════════════════════════════════════

/// M-estimator result.
#[derive(Debug, Clone)]
pub struct MEstimateResult {
    /// Converged location estimate.
    pub location: f64,
    /// Scale estimate used (MAD-based).
    pub scale: f64,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Huber M-estimate of location.
///
/// Iteratively reweighted least squares (IRLS):
/// 1. Start with median as initial estimate
/// 2. Compute standardized residuals: u = (x - μ) / σ̂
/// 3. Compute weights: w = huber_weight(u, k)
/// 4. Update: μ = Σ wᵢxᵢ / Σ wᵢ
/// 5. Repeat until convergence
///
/// k = 1.345 gives 95% asymptotic efficiency at the normal.
pub fn huber_m_estimate(data: &[f64], k: f64, max_iter: usize, tol: f64) -> MEstimateResult {
    m_estimate_irls(data, |u| huber_weight(u, k), max_iter, tol)
}

/// Tukey bisquare M-estimate of location.
///
/// Like Huber but with hard rejection of extreme outliers.
/// k = 4.685 gives 95% efficiency at the normal.
pub fn bisquare_m_estimate(data: &[f64], k: f64, max_iter: usize, tol: f64) -> MEstimateResult {
    m_estimate_irls(data, |u| bisquare_weight(u, k), max_iter, tol)
}

/// Hampel M-estimate of location.
///
/// Three-part redescender with smooth transition.
pub fn hampel_m_estimate(data: &[f64], a: f64, b: f64, c: f64, max_iter: usize, tol: f64) -> MEstimateResult {
    m_estimate_irls(data, |u| hampel_weight(u, a, b, c), max_iter, tol)
}

/// Generic IRLS M-estimation with any weight function.
fn m_estimate_irls(
    data: &[f64],
    weight_fn: impl Fn(f64) -> f64,
    max_iter: usize,
    tol: f64,
) -> MEstimateResult {
    let clean = sorted_nan_free(data);
    if clean.is_empty() {
        return MEstimateResult {
            location: f64::NAN, scale: f64::NAN, iterations: 0, converged: false
        };
    }
    if clean.len() == 1 {
        return MEstimateResult {
            location: clean[0], scale: 0.0, iterations: 0, converged: true
        };
    }

    let med = median(&clean);
    let mad_val = mad_from_sorted(&clean, med);
    // Scale: MAD * 1.4826 (consistency factor for normal)
    let scale = if mad_val > 0.0 { mad_val * 1.4826 } else { 1.0 };

    let mut mu = med;
    let mut converged = false;

    for iter in 0..max_iter {
        let mut w_sum = 0.0;
        let mut wx_sum = 0.0;
        for &x in &clean {
            let u = (x - mu) / scale;
            let w = weight_fn(u);
            w_sum += w;
            wx_sum += w * x;
        }
        let new_mu = if w_sum > 0.0 { wx_sum / w_sum } else { mu };
        if (new_mu - mu).abs() < tol * scale {
            converged = true;
            mu = new_mu;
            return MEstimateResult { location: mu, scale, iterations: iter + 1, converged };
        }
        mu = new_mu;
    }

    MEstimateResult { location: mu, scale, iterations: max_iter, converged }
}

/// MAD from already-sorted data.
fn mad_from_sorted(sorted: &[f64], med: f64) -> f64 {
    let mut deviations: Vec<f64> = sorted.iter().map(|&x| (x - med).abs()).collect();
    deviations.sort_by(|a, b| a.total_cmp(b));
    median(&deviations)
}

// ═══════════════════════════════════════════════════════════════════════════
// Robust scale estimators
// ═══════════════════════════════════════════════════════════════════════════

/// Qn scale estimator (Rousseeuw & Croux 1993).
///
/// Qn = 2.2191 × {|xᵢ - xⱼ|; i < j}_{(k)}
/// where k = ⌊(n choose 2) / 4⌋ + 1 (roughly the first quartile of pairwise distances).
///
/// Breakdown point = 50%. More efficient than MAD (82% vs 37% at normal).
/// O(n²) in this implementation — fine for moderate n.
pub fn qn_scale(data: &[f64]) -> f64 {
    let clean = sorted_nan_free(data);
    let n = clean.len();
    if n < 2 { return 0.0; }

    // Compute all pairwise |xᵢ - xⱼ| for i < j
    let m = n * (n - 1) / 2;
    let mut diffs = Vec::with_capacity(m);
    for i in 0..n {
        for j in (i+1)..n {
            diffs.push((clean[i] - clean[j]).abs());
        }
    }
    diffs.sort_by(|a, b| a.total_cmp(b));

    // Rousseeuw & Croux (1993): k = C(h, 2) where h = ⌊n/2⌋ + 1
    let h = n / 2 + 1;
    let k = h * (h - 1) / 2;
    let raw = diffs[(k - 1).min(m - 1)];

    // Consistency factor for normal: 2.2191
    // Finite-sample correction
    let correction = match n {
        2 => 0.399,
        3 => 0.994,
        4 => 0.512,
        5 => 0.844,
        6 => 0.611,
        7 => 0.857,
        8 => 0.669,
        9 => 0.872,
        _ => 1.0,
    };
    2.2191 * correction * raw
}

/// Sn scale estimator (Rousseeuw & Croux 1993).
///
/// Sn = 1.1926 × med_i { med_j |xᵢ - xⱼ| }
///
/// Breakdown point = 50%. Efficiency ≈ 58% at normal.
pub fn sn_scale(data: &[f64]) -> f64 {
    let clean = sorted_nan_free(data);
    let n = clean.len();
    if n < 2 { return 0.0; }

    let mut inner_medians = Vec::with_capacity(n);
    for i in 0..n {
        let mut diffs: Vec<f64> = clean.iter().map(|&x| (x - clean[i]).abs()).collect();
        diffs.sort_by(|a, b| a.total_cmp(b));
        inner_medians.push(median(&diffs));
    }
    inner_medians.sort_by(|a, b| a.total_cmp(b));

    // Consistency factor for normal: 1.1926
    1.1926 * median(&inner_medians)
}

/// Tau-scale estimator (default parameters: MAD consistency factor 1.4826,
/// bisquare k = 4.685 for 95% efficiency at the normal).
///
/// For full control, use [`tau_scale_with_params`].
pub fn tau_scale(data: &[f64]) -> f64 {
    tau_scale_with_params(data, 1.4826, 4.685)
}

/// Tau-scale estimator with tunable parameters.
///
/// τ² = s² × (1/n) Σ ρ((xᵢ - μ)/s)
/// where s = MAD × `mad_factor`, μ = median, ρ = bisquare rho with
/// tuning constant `bisquare_k`.
///
/// `mad_factor`: consistency factor for the scale estimator. 1.4826 is
///   correct for the normal distribution. 1.0 for the raw MAD.
/// `bisquare_k`: bisquare rho/weight tuning constant. 4.685 gives 95%
///   efficiency at the normal. Smaller values reject more aggressively.
///
/// Combines high breakdown (50%) with good efficiency.
pub fn tau_scale_with_params(data: &[f64], mad_factor: f64, bisquare_k: f64) -> f64 {
    let clean = sorted_nan_free(data);
    let n = clean.len();
    if n < 2 { return 0.0; }

    let med = median(&clean);
    let mad_val = mad_from_sorted(&clean, med);
    if mad_val == 0.0 { return 0.0; }

    let s = mad_val * mad_factor;
    let k = bisquare_k;

    let rho_sum: f64 = clean.iter().map(|&x| {
        let u = (x - med) / s;
        bisquare_rho(u, k)
    }).sum();

    (s * s * rho_sum / n as f64).sqrt()
}

/// Bisquare rho function: ρ(u) = (k²/6)(1 - (1-(u/k)²)³) for |u| ≤ k, k²/6 otherwise.
fn bisquare_rho(u: f64, k: f64) -> f64 {
    let k2_6 = k * k / 6.0;
    if u.abs() <= k {
        let t = 1.0 - (u / k).powi(2);
        k2_6 * (1.0 - t.powi(3))
    } else {
        k2_6
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Least Trimmed Squares (LTS) — simple linear regression
// ═══════════════════════════════════════════════════════════════════════════

/// LTS regression result (simple linear: y = a + bx).
#[derive(Debug, Clone)]
pub struct LtsResult {
    /// Intercept.
    pub intercept: f64,
    /// Slope.
    pub slope: f64,
    /// Number of points in the trimmed subset.
    pub h: usize,
    /// Sum of squared residuals for the trimmed subset.
    pub trimmed_ss: f64,
}

/// Least trimmed squares regression (simple linear).
///
/// Minimizes Σ_{i=1}^h r²_(i:n) where h ≈ ⌊n/2⌋ + 1.
/// The "LTS idea": fit is determined by the majority of the data,
/// even if up to ~50% are outliers.
///
/// This uses random subsampling (C-step algorithm, Rousseeuw & Van Driessen 2006).
/// For each trial: fit to random pair → compute residuals → keep h smallest → refit.
pub fn lts_simple(x: &[f64], y: &[f64], n_trials: usize, seed: u64) -> LtsResult {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 {
        return LtsResult { intercept: f64::NAN, slope: f64::NAN, h: 0, trimmed_ss: f64::INFINITY };
    }

    let h = n / 2 + 1; // Breakdown point ≈ 50%
    let mut best_ss = f64::INFINITY;
    let mut best_a = 0.0;
    let mut best_b = 0.0;
    let mut rng = crate::rng::Xoshiro256::new(seed);

    for _ in 0..n_trials {
        // Random pair
        let i1 = crate::rng::TamRng::next_range(&mut rng, n as u64) as usize;
        let mut i2 = crate::rng::TamRng::next_range(&mut rng, n as u64) as usize;
        if i2 == i1 { i2 = (i1 + 1) % n; }

        // Initial fit through two points
        let dx = x[i2] - x[i1];
        if dx.abs() < 1e-15 { continue; }
        let slope = (y[i2] - y[i1]) / dx;
        let intercept = y[i1] - slope * x[i1];

        // C-step: compute residuals, keep h smallest, refit
        let (a, b, ss) = c_step(x, y, intercept, slope, h);
        if ss < best_ss {
            best_ss = ss;
            best_a = a;
            best_b = b;
        }
    }

    LtsResult { intercept: best_a, slope: best_b, h, trimmed_ss: best_ss }
}

/// One C-step: compute residuals → keep h smallest → OLS on subset.
fn c_step(x: &[f64], y: &[f64], a: f64, b: f64, h: usize) -> (f64, f64, f64) {
    let n = x.len();
    let mut residuals: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, (y[i] - a - b * x[i]).powi(2)))
        .collect();
    residuals.sort_by(|a, b| a.1.total_cmp(&b.1));

    // OLS on h smallest residuals
    let subset: Vec<usize> = residuals[..h].iter().map(|&(i, _)| i).collect();
    let (new_a, new_b) = ols_subset(x, y, &subset);

    // Trimmed SS with new fit
    let mut new_residuals: Vec<f64> = (0..n)
        .map(|i| (y[i] - new_a - new_b * x[i]).powi(2))
        .collect();
    new_residuals.sort_by(|a, b| a.total_cmp(b));
    let ss: f64 = new_residuals[..h].iter().sum();

    (new_a, new_b, ss)
}

/// OLS on a subset of indices.
/// Centered formulation avoids catastrophic cancellation.
fn ols_subset(x: &[f64], y: &[f64], indices: &[usize]) -> (f64, f64) {
    let n = indices.len() as f64;
    let mean_x: f64 = indices.iter().map(|&i| x[i]).sum::<f64>() / n;
    let mean_y: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for &i in indices {
        let dx = x[i] - mean_x;
        sxy += dx * (y[i] - mean_y);
        sxx += dx * dx;
    }
    if sxx.abs() < 1e-15 {
        return (mean_y, 0.0);
    }
    let b = sxy / sxx;
    let a = mean_y - b * mean_x;
    (a, b)
}

// ═══════════════════════════════════════════════════════════════════════════
// Robust covariance (simplified MCD)
// ═══════════════════════════════════════════════════════════════════════════

/// Minimum Covariance Determinant result (for 2D data).
#[derive(Debug, Clone)]
pub struct McdResult2D {
    /// Robust location (center_x, center_y).
    pub center: (f64, f64),
    /// Robust covariance matrix [var_x, cov_xy, cov_xy, var_y].
    pub covariance: [f64; 4],
    /// Robust Mahalanobis distances for each point.
    pub distances: Vec<f64>,
    /// Indices of the h-subset.
    pub subset: Vec<usize>,
}

/// Simplified MCD for 2D data.
///
/// Full MCD is implemented via FAST-MCD (Rousseeuw & Van Driessen 1999).
/// This 2D version captures the core algorithm: subsample → concentrate → minimize det.
pub fn mcd_2d(x: &[f64], y: &[f64], n_trials: usize, seed: u64) -> McdResult2D {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let h = n / 2 + 1;

    let mut best_det = f64::INFINITY;
    let mut best_subset = Vec::new();
    let mut rng = crate::rng::Xoshiro256::new(seed);

    for _ in 0..n_trials {
        // Random 3-point initial subset (minimum for 2D)
        let mut indices: Vec<usize> = Vec::with_capacity(3);
        for _ in 0..30 {
            let idx = crate::rng::TamRng::next_range(&mut rng, n as u64) as usize;
            if !indices.contains(&idx) { indices.push(idx); }
            if indices.len() == 3 { break; }
        }
        if indices.len() < 3 { continue; }

        // C-step: compute distances, keep h closest, recompute
        for _ in 0..10 {
            let (cx, cy, cov) = subset_stats_2d(x, y, &indices);
            let det = cov[0] * cov[3] - cov[1] * cov[2];
            if det <= 0.0 { break; }

            // Compute Mahalanobis distances
            let inv_det = 1.0 / det;
            let mut dists: Vec<(usize, f64)> = (0..n).map(|i| {
                let dx = x[i] - cx;
                let dy = y[i] - cy;
                let d2 = inv_det * (cov[3] * dx * dx - 2.0 * cov[1] * dx * dy + cov[0] * dy * dy);
                (i, d2.max(0.0))
            }).collect();
            dists.sort_by(|a, b| a.1.total_cmp(&b.1));
            indices = dists[..h].iter().map(|&(i, _)| i).collect();
        }

        let (_, _, cov) = subset_stats_2d(x, y, &indices);
        let det = cov[0] * cov[3] - cov[1] * cov[2];
        if det > 0.0 && det < best_det {
            best_det = det;
            best_subset = indices.clone();
        }
    }

    // Final result from best subset
    if best_subset.is_empty() {
        return McdResult2D {
            center: (f64::NAN, f64::NAN),
            covariance: [f64::NAN; 4],
            distances: vec![f64::NAN; n],
            subset: vec![],
        };
    }

    let (cx, cy, cov) = subset_stats_2d(x, y, &best_subset);
    let det = cov[0] * cov[3] - cov[1] * cov[2];
    let inv_det = if det > 0.0 { 1.0 / det } else { f64::INFINITY };

    let distances = (0..n).map(|i| {
        let dx = x[i] - cx;
        let dy = y[i] - cy;
        (inv_det * (cov[3] * dx * dx - 2.0 * cov[1] * dx * dy + cov[0] * dy * dy)).max(0.0).sqrt()
    }).collect();

    McdResult2D {
        center: (cx, cy),
        covariance: cov,
        distances,
        subset: best_subset,
    }
}

/// Compute mean and covariance from a subset of 2D points.
fn subset_stats_2d(x: &[f64], y: &[f64], indices: &[usize]) -> (f64, f64, [f64; 4]) {
    let n = indices.len() as f64;
    let cx: f64 = indices.iter().map(|&i| x[i]).sum::<f64>() / n;
    let cy: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
    let mut vxx = 0.0;
    let mut vxy = 0.0;
    let mut vyy = 0.0;
    for &i in indices {
        let dx = x[i] - cx;
        let dy = y[i] - cy;
        vxx += dx * dx;
        vxy += dx * dy;
        vyy += dy * dy;
    }
    let denom = n - 1.0;
    (cx, cy, [vxx / denom, vxy / denom, vxy / denom, vyy / denom])
}

// ═══════════════════════════════════════════════════════════════════════════
// Medcouple (robust measure of skewness)
// ═══════════════════════════════════════════════════════════════════════════

/// Medcouple: robust measure of skewness.
///
/// MC = med { h(xᵢ, xⱼ) : xᵢ ≥ median ≥ xⱼ }
/// where h(xᵢ, xⱼ) = (xⱼ - median + median - xᵢ) / (xⱼ - xᵢ)
///                   = ((xⱼ - med) - (xᵢ - med)) / (xⱼ - xᵢ)
///
/// MC = 0 for symmetric. MC > 0 for right-skewed. MC < 0 for left-skewed.
/// Breakdown point = 25%.
pub fn medcouple(data: &[f64]) -> f64 {
    let sorted = sorted_nan_free(data);
    let n = sorted.len();
    if n < 2 { return f64::NAN; }

    let med = median(&sorted);

    // Find points above and below median
    let z_plus: Vec<f64> = sorted.iter().filter(|&&x| x >= med).copied().collect();
    let z_minus: Vec<f64> = sorted.iter().filter(|&&x| x <= med).copied().collect();

    // Compute all h(xᵢ, xⱼ) values
    let mut h_vals = Vec::new();
    for &zp in &z_plus {
        for &zm in &z_minus {
            let diff = zp - zm;
            if diff.abs() < 1e-15 {
                // Both equal to median
                h_vals.push(0.0);
            } else {
                h_vals.push(((zp - med) - (med - zm)) / diff);
            }
        }
    }

    if h_vals.is_empty() { return 0.0; }
    h_vals.sort_by(|a, b| a.total_cmp(b));
    median(&h_vals)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        (a - b).abs() < tol
    }

    // ── M-estimators of location ─────────────────────────────────────────

    #[test]
    fn huber_clean_data() {
        // Clean normal-ish data: Huber should be close to mean
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = huber_m_estimate(&data, 1.345, 50, 1e-6);
        assert!(r.converged);
        assert!(approx(r.location, 5.5, 0.3), "location={}", r.location);
    }

    #[test]
    fn huber_with_outlier() {
        // Data with outlier: Huber should resist it
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0];
        let r = huber_m_estimate(&data, 1.345, 50, 1e-6);
        // Mean = 104.5, but Huber should be close to 5.0
        assert!(r.location < 10.0, "location={} should resist outlier", r.location);
    }

    #[test]
    fn bisquare_hard_rejection() {
        // Bisquare should completely reject extreme outliers
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10000.0];
        let r = bisquare_m_estimate(&data, 4.685, 50, 1e-6);
        assert!(r.location < 10.0, "location={} should hard-reject outlier", r.location);
    }

    #[test]
    fn hampel_with_outliers() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0];
        let r = hampel_m_estimate(&data, 1.7, 3.4, 8.5, 50, 1e-6);
        assert!(r.location < 10.0, "location={}", r.location);
    }

    #[test]
    fn m_estimate_single_value() {
        let data = [42.0];
        let r = huber_m_estimate(&data, 1.345, 50, 1e-6);
        assert!(approx(r.location, 42.0, TOL));
    }

    #[test]
    fn m_estimate_empty() {
        let data: [f64; 0] = [];
        let r = huber_m_estimate(&data, 1.345, 50, 1e-6);
        assert!(r.location.is_nan());
    }

    // ── Scale estimators ─────────────────────────────────────────────────

    #[test]
    fn qn_scale_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let q = qn_scale(&data);
        assert!(q > 0.0, "Qn should be positive");
        // For uniform-ish data, Qn should be in the right ballpark
        assert!(q > 1.0 && q < 10.0, "Qn={}", q);
    }

    #[test]
    fn qn_scale_with_outlier() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0];
        let q = qn_scale(&data);
        // Should be robust to the outlier
        assert!(q < 50.0, "Qn={} should resist outlier", q);
    }

    #[test]
    fn sn_scale_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let s = sn_scale(&data);
        assert!(s > 0.0);
        assert!(s > 1.0 && s < 10.0, "Sn={}", s);
    }

    #[test]
    fn tau_scale_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let t = tau_scale(&data);
        assert!(t > 0.0);
    }

    // ── LTS regression ───────────────────────────────────────────────────

    #[test]
    fn lts_clean_linear() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let r = lts_simple(&x, &y, 100, 42);
        assert!(approx(r.slope, 2.0, 0.1), "slope={}", r.slope);
        assert!(approx(r.intercept, 1.0, 0.5), "intercept={}", r.intercept);
    }

    #[test]
    fn lts_with_outliers() {
        // 7 good points + 3 outliers
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        y[7] = 100.0; // outlier
        y[8] = -50.0; // outlier
        y[9] = 200.0; // outlier
        let r = lts_simple(&x, &y, 200, 42);
        // LTS should recover slope ≈ 2, ignoring outliers
        assert!(approx(r.slope, 2.0, 0.5), "slope={}", r.slope);
    }

    // ── MCD ──────────────────────────────────────────────────────────────

    #[test]
    fn mcd_clean_data() {
        // Non-collinear 2D data
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [2.1, 3.8, 6.2, 8.1, 9.5, 12.3, 13.8, 16.1, 17.9, 20.5];
        let r = mcd_2d(&x, &y, 100, 42);
        // Center should be near the data center
        assert!((r.center.0 - 5.5).abs() < 3.0, "cx={}", r.center.0);
        assert!((r.center.1 - 11.0).abs() < 5.0, "cy={}", r.center.1);
        assert_eq!(r.distances.len(), 10);
    }

    #[test]
    fn mcd_with_outlier() {
        // Non-collinear data with an outlier at index 9
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let y = [2.1, 3.8, 6.2, 8.1, 9.5, 12.3, 13.8, 16.1, 17.9, 200.0];
        let r = mcd_2d(&x, &y, 100, 42);
        // Outlier (100, 200) should have the largest Mahalanobis distance
        assert!(!r.subset.is_empty(), "Should find a valid subset");
        let max_idx = r.distances.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i).unwrap();
        assert_eq!(max_idx, 9, "Outlier should have largest distance, got idx={}", max_idx);
    }

    // ── Medcouple ────────────────────────────────────────────────────────

    #[test]
    fn medcouple_symmetric() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mc = medcouple(&data);
        assert!(mc.abs() < 0.1, "MC={} should be near 0 for symmetric data", mc);
    }

    #[test]
    fn medcouple_right_skewed() {
        // Exponential-like: heavy right tail, median = 5.0
        let data = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
        let mc = medcouple(&data);
        assert!(mc > 0.0, "MC={} should be positive for right-skewed", mc);
    }

    #[test]
    fn medcouple_left_skewed() {
        // Mirror of right-skewed
        let data = [-100.0, -50.0, -20.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.1];
        let mc = medcouple(&data);
        assert!(mc < 0.0, "MC={} should be negative for left-skewed", mc);
    }

    // ── Weight functions ─────────────────────────────────────────────────

    #[test]
    fn huber_weight_inside() {
        assert_eq!(huber_weight(0.5, 1.345), 1.0);
    }

    #[test]
    fn huber_weight_outside() {
        let w = huber_weight(3.0, 1.345);
        assert!(w < 1.0 && w > 0.0, "w={}", w);
        assert!(approx(w, 1.345 / 3.0, 1e-10));
    }

    #[test]
    fn bisquare_weight_inside() {
        let w = bisquare_weight(1.0, 4.685);
        assert!(w > 0.0 && w <= 1.0);
    }

    #[test]
    fn bisquare_weight_outside() {
        assert_eq!(bisquare_weight(5.0, 4.685), 0.0);
    }

    // ── Cancellation canary ─────────────────────────────────────────────

    #[test]
    fn ols_subset_cancellation_canary() {
        // Both x and y at 1e12: naive formula (n*sxx - sx*sx) produces garbage
        // because the ~2.5e27 terms cancel below ULP. Centered formula is exact
        // because dx = i - 24.5 and dy = 2i - 49 are small integers.
        let x: Vec<f64> = (0..50).map(|i| 1e12 + i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + 2.0 * xi).collect();
        let indices: Vec<usize> = (0..50).collect();
        let (a, b) = ols_subset(&x, &y, &indices);
        assert!((b - 2.0).abs() < 1e-6,
            "slope={b} should be 2.0 (cancellation canary)");
        assert!((a - 3.0).abs() < 1e-3,
            "intercept={a} should be 3.0 (cancellation canary)");
    }
}
