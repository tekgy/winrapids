//! Descriptive statistics — all flavors, from tambear primitives.
//!
//! ## Architecture
//!
//! Two-pass numerically stable algorithm:
//!
//! **Pass 1**: `scatter_multi_phi(["v", "1.0"])` → (sum, count) per group
//!            + `scatter_extremum(min, max)` → extrema per group
//!            + CPU: `means = sum / count`
//!
//! **Pass 2**: `scatter_multi_phi(["(v-r)²", "(v-r)³", "(v-r)⁴"])` with `refs = means`
//!            → centered moment sums (m2, m3, m4) per group
//!
//! From 7 accumulators `{count, sum, min, max, m2, m3, m4}`, every descriptive
//! statistic is derived on CPU in O(n_groups) time — negligible.
//!
//! ## NaN handling
//!
//! NaN values are excluded via packed u64 bitmask (mask-not-filter invariant).
//! The fast path (no NaN) uses fused `scatter_multi_phi` (1 kernel per pass).
//! The NaN path uses `scatter_phi_masked` (1 kernel per expression per pass).
//!
//! ## Single-group optimization
//!
//! For ungrouped statistics (one group), a direct CPU two-pass loop is used —
//! no scatter overhead, no GPU transfer, NaN-aware in-loop.
//!
//! ## Sharing
//!
//! `MomentStats` is the MSR (minimum sufficient representation). Register it
//! in `TamSession` to share across consumers: normalization, z-scoring,
//! hypothesis tests, regression preprocessing. All derive from these 7 fields.
//!
//! ## .tbs integration
//!
//! ```text
//! describe()                      # full descriptive stats
//! mean()                          # single stat
//! std(ddof=1)                     # with degrees of freedom
//! skewness(bias=false)            # adjusted Fisher-Pearson
//! quantile(0.25)                  # Q1
//! ```

use crate::compute_engine::ComputeEngine;

// ── Phi expression constants ──────────────────────────────────────────────

/// φ = (v - r)³ → centered sum of cubes (for skewness)
pub const PHI_CENTERED_CU: &str = "(v - r) * (v - r) * (v - r)";
/// φ = (v - r)⁴ → centered sum of fourth powers (for kurtosis)
pub const PHI_CENTERED_QU: &str = "(v - r) * (v - r) * (v - r) * (v - r)";

// Re-use existing phi constants from the crate
use crate::{PHI_SUM, PHI_COUNT, PHI_CENTERED_SUM_SQ};

// ── MomentStats ───────────────────────────────────────────────────────────

/// Sufficient statistics for moment-based descriptive statistics.
///
/// The MSR: 7 accumulators from which ALL descriptive stats derive in O(1).
/// Two scatter passes produce these for any number of groups.
///
/// These fields are what `TamSession` caches for cross-algorithm sharing.
/// Normalization, z-scoring, Pearson correlation, hypothesis tests — all
/// consume `MomentStats` without re-scanning the data.
#[derive(Debug, Clone, Copy)]
pub struct MomentStats {
    /// Number of valid (non-NaN) observations.
    pub count: f64,
    /// Sum of valid values: Σx.
    pub sum: f64,
    /// Minimum value (f64::INFINITY if count == 0).
    pub min: f64,
    /// Maximum value (f64::NEG_INFINITY if count == 0).
    pub max: f64,
    /// Second central moment sum: Σ(x - x̄)².
    pub m2: f64,
    /// Third central moment sum: Σ(x - x̄)³.
    pub m3: f64,
    /// Fourth central moment sum: Σ(x - x̄)⁴.
    pub m4: f64,
}

impl MomentStats {
    /// Empty stats (no valid observations).
    pub fn empty() -> Self {
        Self { count: 0.0, sum: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY,
               m2: 0.0, m3: 0.0, m4: 0.0 }
    }

    /// Number of valid observations as integer.
    pub fn n(&self) -> usize { self.count as usize }

    // ── Central tendency ──────────────────────────────────────────────

    /// Arithmetic mean. NaN if count == 0.
    pub fn mean(&self) -> f64 {
        if self.count == 0.0 { f64::NAN } else { self.sum / self.count }
    }

    // ── Dispersion ────────────────────────────────────────────────────

    /// Variance.
    ///
    /// `ddof`: delta degrees of freedom.
    /// - `ddof = 0` → population variance: m2 / n
    /// - `ddof = 1` → sample variance: m2 / (n - 1)  (Bessel's correction)
    ///
    /// Returns NaN if n <= ddof.
    pub fn variance(&self, ddof: u32) -> f64 {
        let denom = self.count - ddof as f64;
        if denom <= 0.0 { return f64::NAN; }
        self.m2 / denom
    }

    /// Standard deviation. See [`variance`](Self::variance) for `ddof`.
    pub fn std(&self, ddof: u32) -> f64 { self.variance(ddof).sqrt() }

    /// Range: max - min. NaN if count == 0.
    pub fn range(&self) -> f64 {
        if self.count == 0.0 { f64::NAN } else { self.max - self.min }
    }

    /// Coefficient of variation: std(ddof=1) / |mean|.
    /// NaN if mean == 0 or count < 2.
    pub fn cv(&self) -> f64 {
        let m = self.mean();
        if m == 0.0 || self.count < 2.0 { return f64::NAN; }
        self.std(1) / m.abs()
    }

    /// Standard error of the mean: std(ddof=1) / √n.
    pub fn sem(&self) -> f64 {
        if self.count < 2.0 { return f64::NAN; }
        self.std(1) / self.count.sqrt()
    }

    // ── Shape ─────────────────────────────────────────────────────────

    /// Skewness (Fisher's g₁).
    ///
    /// - `bias = true` → population (biased): g₁ = μ₃ / σ³
    /// - `bias = false` → sample (adjusted Fisher-Pearson):
    ///   G₁ = g₁ × √(n(n-1)) / (n-2)
    ///
    /// Returns NaN if n < 3 (sample), n < 1 (population), or variance == 0.
    pub fn skewness(&self, bias: bool) -> f64 {
        let n = self.count;
        if n < 1.0 || self.m2 == 0.0 { return f64::NAN; }

        // Population: g₁ = (m3/n) / (m2/n)^(3/2)
        let mu3 = self.m3 / n;
        let var = self.m2 / n;
        let g1 = mu3 / (var * var.sqrt());

        if bias { g1 } else {
            if n < 3.0 { return f64::NAN; }
            g1 * (n * (n - 1.0)).sqrt() / (n - 2.0)
        }
    }

    /// Kurtosis.
    ///
    /// - `excess = true` → subtract 3 (normal = 0)
    /// - `bias = true` → population (biased)
    /// - `bias = false` → sample (adjusted)
    ///
    /// Returns NaN if n < 4 (sample), n < 1 (population), or variance == 0.
    pub fn kurtosis(&self, excess: bool, bias: bool) -> f64 {
        let n = self.count;
        if n < 1.0 || self.m2 == 0.0 { return f64::NAN; }

        // Population: raw = (m4/n) / (m2/n)²
        let var = self.m2 / n;
        let raw = (self.m4 / n) / (var * var);
        let g2 = if excess { raw - 3.0 } else { raw };

        if bias { g2 } else {
            if n < 4.0 { return f64::NAN; }
            if excess {
                // G₂ = ((n-1)/((n-2)(n-3))) × ((n+1)×g₂_biased + 6)
                ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * (raw - 3.0) + 6.0)
            } else {
                ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * (raw - 3.0) + 6.0) + 3.0
            }
        }
    }

    /// All derived statistics as a [`DescriptiveResult`].
    pub fn describe(&self) -> DescriptiveResult {
        DescriptiveResult {
            count: self.count,
            mean: self.mean(),
            std_pop: self.std(0),
            std_sample: self.std(1),
            variance_pop: self.variance(0),
            variance_sample: self.variance(1),
            min: self.min,
            max: self.max,
            range: self.range(),
            skewness: self.skewness(false),
            kurtosis: self.kurtosis(true, false),
            cv: self.cv(),
            sem: self.sem(),
            sum: self.sum,
            m2: self.m2,
            m3: self.m3,
            m4: self.m4,
        }
    }

    /// Combine two MomentStats (for parallel / distributed computation).
    ///
    /// Uses the parallel merge formulas for central moments. This allows
    /// computing stats across partitions without re-reading the data.
    pub fn merge(a: &MomentStats, b: &MomentStats) -> MomentStats {
        let na = a.count;
        let nb = b.count;
        let n = na + nb;
        if n == 0.0 { return MomentStats::empty(); }
        if na == 0.0 { return *b; }
        if nb == 0.0 { return *a; }

        let sum = a.sum + b.sum;
        let delta = b.sum / nb - a.sum / na;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let m2 = a.m2 + b.m2 + delta2 * na * nb / n;
        let m3 = a.m3 + b.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * b.m2 - nb * a.m2) / n;
        let m4 = a.m4 + b.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * b.m2 + nb * nb * a.m2) / (n * n)
            + 4.0 * delta * (na * b.m3 - nb * a.m3) / n;

        MomentStats {
            count: n,
            sum,
            min: a.min.min(b.min),
            max: a.max.max(b.max),
            m2, m3, m4,
        }
    }
}

// ── DescriptiveResult ─────────────────────────────────────────────────────

/// Full derived descriptive statistics for one group or the entire array.
///
/// Every field is derived from [`MomentStats`] in O(1). The raw moment sums
/// (m2, m3, m4) are included for downstream sharing.
#[derive(Debug, Clone)]
pub struct DescriptiveResult {
    pub count: f64,
    pub mean: f64,
    pub std_pop: f64,
    pub std_sample: f64,
    pub variance_pop: f64,
    pub variance_sample: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    /// Sample-adjusted Fisher-Pearson skewness.
    pub skewness: f64,
    /// Sample-adjusted excess kurtosis.
    pub kurtosis: f64,
    /// Coefficient of variation (std_sample / |mean|).
    pub cv: f64,
    /// Standard error of the mean.
    pub sem: f64,
    pub sum: f64,
    /// Raw central moment sums (for sharing).
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
}

// ── GroupedMomentStats ────────────────────────────────────────────────────

/// Per-group moment statistics: parallel arrays, one entry per group.
///
/// Produced by `DescriptiveEngine::moments_grouped`. Each group's stats
/// can be extracted as a `MomentStats` for O(1) derived statistics.
#[derive(Debug, Clone)]
pub struct GroupedMomentStats {
    pub n_groups: usize,
    pub counts: Vec<f64>,
    pub sums: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub m2s: Vec<f64>,
    pub m3s: Vec<f64>,
    pub m4s: Vec<f64>,
}

impl GroupedMomentStats {
    /// Extract `MomentStats` for group `g`.
    pub fn group(&self, g: usize) -> MomentStats {
        MomentStats {
            count: self.counts[g], sum: self.sums[g],
            min: self.mins[g], max: self.maxs[g],
            m2: self.m2s[g], m3: self.m3s[g], m4: self.m4s[g],
        }
    }

    /// Describe all groups.
    pub fn describe_all(&self) -> Vec<DescriptiveResult> {
        (0..self.n_groups).map(|g| self.group(g).describe()).collect()
    }

    /// Per-group means (O(n_groups), no GPU).
    pub fn means(&self) -> Vec<f64> {
        self.sums.iter().zip(&self.counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { f64::NAN })
            .collect()
    }
}

// ── DescriptiveEngine ─────────────────────────────────────────────────────

/// Descriptive statistics engine.
///
/// For ungrouped (global) statistics: direct CPU two-pass loop (optimal).
/// For grouped statistics: `ComputeEngine` scatter primitives (GPU-accelerated).
///
/// # Example
/// ```no_run
/// use tambear::descriptive::DescriptiveEngine;
///
/// let mut engine = DescriptiveEngine::new();
///
/// // Ungrouped
/// let stats = engine.moments(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert!((stats.mean() - 3.0).abs() < 1e-10);
///
/// // Grouped (uses GPU scatter on CUDA, CPU fallback otherwise)
/// let keys = vec![0i32, 0, 0, 1, 1, 1];
/// let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let grouped = engine.moments_grouped(&vals, &keys, 2).unwrap();
/// assert!((grouped.group(0).mean() - 2.0).abs() < 1e-10);
/// assert!((grouped.group(1).mean() - 5.0).abs() < 1e-10);
/// ```
pub struct DescriptiveEngine {
    compute: ComputeEngine,
}

impl DescriptiveEngine {
    /// Create engine with auto-detected backend (CUDA → CPU fallback).
    pub fn new() -> Self {
        Self { compute: ComputeEngine::new(tam_gpu::detect()) }
    }

    /// Compute moment statistics for an ungrouped array (single group).
    ///
    /// Two-pass CPU loop: NaN-aware, numerically stable, O(2n).
    pub fn moments(&self, values: &[f64]) -> MomentStats {
        moments_ungrouped(values)
    }

    /// Full descriptive statistics for an ungrouped array.
    pub fn describe(&self, values: &[f64]) -> DescriptiveResult {
        self.moments(values).describe()
    }

    /// Compute grouped moment statistics using scatter primitives.
    ///
    /// Two scatter passes (+ extremum):
    /// 1. `scatter(sum, count)` + `extremum(min, max)` → basic accumulators
    /// 2. `scatter(centered_sq, centered_cu, centered_qu)` with refs=means → moments
    ///
    /// NaN values excluded via packed u64 bitmask. Fast path (no NaN) uses
    /// fused `scatter_multi_phi` for each pass (2 kernel launches + 2 extremum).
    pub fn moments_grouped(
        &mut self,
        values: &[f64],
        keys: &[i32],
        n_groups: usize,
    ) -> Result<GroupedMomentStats, Box<dyn std::error::Error>> {
        assert_eq!(values.len(), keys.len(), "values and keys must have same length");

        // NaN mask
        let (mask, n_nan) = nan_mask_count(values);
        let has_nan = n_nan > 0;

        // ── Pass 1: sum + count ─────────────────────────────────────────
        let (sums, counts) = if has_nan {
            let s = self.compute.scatter_phi_masked(PHI_SUM, keys, values, None, &mask, n_groups)?;
            let c = self.compute.scatter_phi_masked(PHI_COUNT, keys, values, None, &mask, n_groups)?;
            (s, c)
        } else {
            let r = self.compute.scatter_multi_phi(&[PHI_SUM, PHI_COUNT], keys, values, None, n_groups)?;
            (r[0].clone(), r[1].clone())
        };

        // ── Extrema ─────────────────────────────────────────────────────
        let (mins, maxs) = if has_nan {
            // scatter_extremum has no masked variant; CPU fallback for NaN case
            extremum_grouped_nan_safe(values, keys, n_groups)
        } else {
            let mins = self.compute.scatter_extremum(false, keys, values, n_groups)?;
            let maxs = self.compute.scatter_extremum(true, keys, values, n_groups)?;
            (mins, maxs)
        };

        // ── Compute means (CPU, O(n_groups)) ────────────────────────────
        let means: Vec<f64> = sums.iter().zip(&counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { 0.0 })
            .collect();

        // ── Pass 2: centered moments ────────────────────────────────────
        let (m2s, m3s, m4s) = if has_nan {
            let m2 = self.compute.scatter_phi_masked(PHI_CENTERED_SUM_SQ, keys, values, Some(&means), &mask, n_groups)?;
            let m3 = self.compute.scatter_phi_masked(PHI_CENTERED_CU, keys, values, Some(&means), &mask, n_groups)?;
            let m4 = self.compute.scatter_phi_masked(PHI_CENTERED_QU, keys, values, Some(&means), &mask, n_groups)?;
            (m2, m3, m4)
        } else {
            let r = self.compute.scatter_multi_phi(
                &[PHI_CENTERED_SUM_SQ, PHI_CENTERED_CU, PHI_CENTERED_QU],
                keys, values, Some(&means), n_groups,
            )?;
            (r[0].clone(), r[1].clone(), r[2].clone())
        };

        Ok(GroupedMomentStats { n_groups, counts, sums, mins, maxs, m2s, m3s, m4s })
    }

    /// Full descriptive statistics per group.
    pub fn describe_grouped(
        &mut self,
        values: &[f64],
        keys: &[i32],
        n_groups: usize,
    ) -> Result<Vec<DescriptiveResult>, Box<dyn std::error::Error>> {
        Ok(self.moments_grouped(values, keys, n_groups)?.describe_all())
    }

    // ── Individual stat shortcuts ─────────────────────────────────────

    /// Mean of ungrouped values. O(n) single pass.
    pub fn mean(&self, values: &[f64]) -> f64 {
        let (sum, count) = sum_count_nan_safe(values);
        if count == 0.0 { f64::NAN } else { sum / count }
    }

    /// Per-group means. One scatter pass.
    pub fn mean_grouped(
        &mut self,
        values: &[f64],
        keys: &[i32],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let (mask, n_nan) = nan_mask_count(values);
        let (sums, counts) = if n_nan > 0 {
            let s = self.compute.scatter_phi_masked(PHI_SUM, keys, values, None, &mask, n_groups)?;
            let c = self.compute.scatter_phi_masked(PHI_COUNT, keys, values, None, &mask, n_groups)?;
            (s, c)
        } else {
            let r = self.compute.scatter_multi_phi(&[PHI_SUM, PHI_COUNT], keys, values, None, n_groups)?;
            (r[0].clone(), r[1].clone())
        };
        Ok(sums.iter().zip(&counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { f64::NAN })
            .collect())
    }
}

// ── Ungrouped computation ─────────────────────────────────────────────────

/// Compute moment stats for an ungrouped array. Direct CPU two-pass loop.
///
/// NaN values are excluded. Numerically stable via centering in pass 2.
pub fn moments_ungrouped(values: &[f64]) -> MomentStats {
    if values.is_empty() { return MomentStats::empty(); }

    // Pass 1: sum, count, min, max
    let mut count = 0.0f64;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for &v in values {
        if v.is_nan() { continue; }
        count += 1.0;
        sum += v;
        if v < min { min = v; }
        if v > max { max = v; }
    }

    if count == 0.0 { return MomentStats::empty(); }

    let mean = sum / count;

    // Pass 2: centered moments (numerically stable — deviations from mean are small)
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;

    for &v in values {
        if v.is_nan() { continue; }
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    MomentStats { count, sum, min, max, m2, m3, m4 }
}

/// Session-aware variant of [`moments_ungrouped`].
///
/// Checks `session` for cached `MomentStats` for this exact data. If found,
/// returns the cached value at zero cost. If not, computes via `moments_ungrouped`,
/// registers in `session`, and returns the result.
///
/// # Cross-algorithm sharing
///
/// A pipeline that calls `moments_session` once makes its stats free for any
/// subsequent hypothesis test, normalization step, or correlation computation
/// operating on the same data:
///
/// ```no_run
/// use tambear::intermediates::TamSession;
/// use tambear::descriptive::moments_session;
/// use tambear::hypothesis::one_sample_t;
///
/// let mut session = TamSession::new();
/// let data = vec![2.0, 3.0, 5.0, 7.0, 11.0];
///
/// // First call: computes + registers.
/// let stats = moments_session(&mut session, &data);
///
/// // Hypothesis test consumes the same stats — zero re-scan.
/// let result = one_sample_t(&stats, 5.0);
/// ```
pub fn moments_session(session: &mut crate::intermediates::TamSession, values: &[f64]) -> MomentStats {
    use std::sync::Arc;
    use crate::intermediates::{DataId, IntermediateTag};

    let data_id = DataId::from_f64(values);
    let tag = IntermediateTag::MomentStats { data_id };

    if let Some(cached) = session.get::<MomentStats>(&tag) {
        return *cached;
    }

    let stats = moments_ungrouped(values);
    session.register(tag, Arc::new(stats));
    stats
}

// ── Helper functions ──────────────────────────────────────────────────────

fn sum_count_nan_safe(values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for &v in values {
        if !v.is_nan() { sum += v; count += 1.0; }
    }
    (sum, count)
}

/// Build NaN exclusion mask (packed u64 bitmask) and count NaN values.
fn nan_mask_count(values: &[f64]) -> (Vec<u64>, usize) {
    let n_words = (values.len() + 63) / 64;
    let mut mask = vec![0u64; n_words];
    let mut n_nan = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            n_nan += 1;
        } else {
            mask[i / 64] |= 1u64 << (i % 64);
        }
    }
    (mask, n_nan)
}

/// Per-group min/max with NaN exclusion (CPU fallback).
fn extremum_grouped_nan_safe(
    values: &[f64],
    keys: &[i32],
    n_groups: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::INFINITY; n_groups];
    let mut maxs = vec![f64::NEG_INFINITY; n_groups];
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() { continue; }
        let g = keys[i] as usize;
        if v < mins[g] { mins[g] = v; }
        if v > maxs[g] { maxs[g] = v; }
    }
    (mins, maxs)
}

// ── Quantile computation (sort-based) ─────────────────────────────────────

/// Interpolation method for quantiles.
///
/// These match R's `type` parameter in `quantile()` and NumPy/SciPy methods.
/// Method 7 (Linear) is the default in R, Python, and most software.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// R type 1: inverse CDF (discontinuous). Smallest observation ≥ p.
    InverseCdf,
    /// R type 4: linear interpolation of empirical CDF. p*n.
    Linear4,
    /// R type 5: Hazen. (p*n + 0.5).
    Hazen,
    /// R type 6: Weibull quantile. p*(n+1).
    Weibull,
    /// R type 7: linear interpolation (default). 1 + p*(n-1).
    Linear,
    /// R type 8: median-unbiased. p*(n+1/3) + 1/3.
    MedianUnbiased,
    /// R type 9: normal-unbiased. p*(n+1/4) + 3/8.
    NormalUnbiased,
}

impl Default for QuantileMethod {
    fn default() -> Self { Self::Linear }
}

/// Compute a single quantile from a **sorted, NaN-free** slice.
///
/// `q` must be in [0, 1]. Returns NaN if slice is empty.
pub fn quantile(sorted: &[f64], q: f64, method: QuantileMethod) -> f64 {
    assert!((0.0..=1.0).contains(&q), "quantile q must be in [0, 1], got {}", q);
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return sorted[0]; }
    let nf = n as f64;

    match method {
        QuantileMethod::InverseCdf => {
            if q == 0.0 { return sorted[0]; }
            let j = (q * nf).ceil() as usize;
            sorted[j.min(n).saturating_sub(1)]
        }
        _ => {
            let h = match method {
                QuantileMethod::Linear4 => q * nf,
                QuantileMethod::Hazen => q * nf + 0.5,
                QuantileMethod::Weibull => q * (nf + 1.0),
                QuantileMethod::Linear => 1.0 + q * (nf - 1.0),
                QuantileMethod::MedianUnbiased => q * (nf + 1.0 / 3.0) + 1.0 / 3.0,
                QuantileMethod::NormalUnbiased => q * (nf + 0.25) + 3.0 / 8.0,
                QuantileMethod::InverseCdf => unreachable!(),
            };
            // h is 1-indexed: h=1 → sorted[0], h=n → sorted[n-1]
            let h0 = (h - 1.0).clamp(0.0, (n - 1) as f64);
            let j = h0.floor() as usize;
            let g = h0 - j as f64;
            if j >= n - 1 {
                sorted[n - 1]
            } else {
                sorted[j] * (1.0 - g) + sorted[j + 1] * g
            }
        }
    }
}

/// Median of a **sorted, NaN-free** slice.
pub fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
}

/// Quartiles (Q1, Q2, Q3) of a **sorted, NaN-free** slice.
pub fn quartiles(sorted: &[f64]) -> (f64, f64, f64) {
    (
        quantile(sorted, 0.25, QuantileMethod::Linear),
        quantile(sorted, 0.50, QuantileMethod::Linear),
        quantile(sorted, 0.75, QuantileMethod::Linear),
    )
}

/// Interquartile range: Q3 - Q1.
pub fn iqr(sorted: &[f64]) -> f64 {
    let (q1, _, q3) = quartiles(sorted);
    q3 - q1
}

// ── Additional central tendency ───────────────────────────────────────────

/// Geometric mean: exp(Σlog(x) / n).
///
/// Only valid for positive values. Returns NaN if any value ≤ 0, is NaN,
/// or array is empty.
pub fn geometric_mean(values: &[f64]) -> f64 {
    let mut sum_log = 0.0f64;
    let mut n = 0.0f64;
    for &v in values {
        if v.is_nan() { continue; }
        if v <= 0.0 { return f64::NAN; }
        sum_log += v.ln();
        n += 1.0;
    }
    if n == 0.0 { return f64::NAN; }
    (sum_log / n).exp()
}

/// Harmonic mean: n / Σ(1/x).
///
/// Only valid for positive values. Returns NaN if any value ≤ 0, is NaN,
/// or array is empty.
pub fn harmonic_mean(values: &[f64]) -> f64 {
    let mut sum_recip = 0.0f64;
    let mut n = 0.0f64;
    for &v in values {
        if v.is_nan() { continue; }
        if v <= 0.0 { return f64::NAN; }
        sum_recip += 1.0 / v;
        n += 1.0;
    }
    if n == 0.0 || sum_recip == 0.0 { return f64::NAN; }
    n / sum_recip
}

/// Trimmed mean: mean of the middle portion after removing the lowest and
/// highest `fraction` of values.
///
/// `sorted` must be sorted ascending, NaN-free.
/// `fraction` in [0, 0.5) — proportion to trim from EACH end.
pub fn trimmed_mean(sorted: &[f64], fraction: f64) -> f64 {
    assert!((0.0..0.5).contains(&fraction), "trim fraction must be in [0, 0.5), got {}", fraction);
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    let trim = (n as f64 * fraction).floor() as usize;
    let trimmed = &sorted[trim..n - trim];
    if trimmed.is_empty() { return f64::NAN; }
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Winsorized mean: mean after replacing extremes at the trim boundaries.
///
/// `sorted` must be sorted ascending, NaN-free.
/// `fraction` in [0, 0.5) — proportion to winsorize from EACH end.
pub fn winsorized_mean(sorted: &[f64], fraction: f64) -> f64 {
    assert!((0.0..0.5).contains(&fraction), "winsorize fraction must be in [0, 0.5), got {}", fraction);
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    let trim = (n as f64 * fraction).floor() as usize;
    if trim == 0 { return sorted.iter().sum::<f64>() / n as f64; }
    let lo = sorted[trim];
    let hi = sorted[n - 1 - trim];
    let mut sum = 0.0f64;
    for (i, &v) in sorted.iter().enumerate() {
        sum += if i < trim { lo } else if i >= n - trim { hi } else { v };
    }
    sum / n as f64
}

/// Filter NaN values and sort. Returns sorted, NaN-free copy.
///
/// Use as preprocessing for quantile/median/trimmed/winsorized operations.
pub fn sorted_nan_free(values: &[f64]) -> Vec<f64> {
    let mut clean: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    clean.sort_unstable_by(|a, b| a.total_cmp(b));
    clean
}

// ── Additional skewness flavors ───────────────────────────────────────────

/// Pearson's first skewness coefficient: 3(mean - median) / std.
pub fn pearson_first_skewness(stats: &MomentStats, median_val: f64) -> f64 {
    let s = stats.std(1);
    if s == 0.0 { return f64::NAN; }
    3.0 * (stats.mean() - median_val) / s
}

/// Bowley (quartile) skewness: (Q3 + Q1 - 2·Q2) / (Q3 - Q1).
///
/// Robust to outliers. `sorted` must be sorted ascending, NaN-free.
pub fn bowley_skewness(sorted: &[f64]) -> f64 {
    let (q1, q2, q3) = quartiles(sorted);
    let denom = q3 - q1;
    if denom == 0.0 { return f64::NAN; }
    (q3 + q1 - 2.0 * q2) / denom
}

// ── Additional dispersion ─────────────────────────────────────────────────

/// Median Absolute Deviation: median(|x - median(x)|).
///
/// `sorted` must be sorted ascending, NaN-free.
/// Returns unscaled MAD. For normal data, MAD × 1.4826 ≈ std.
pub fn mad(sorted: &[f64]) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let med = median(sorted);
    let mut deviations: Vec<f64> = sorted.iter().map(|&v| (v - med).abs()).collect();
    deviations.sort_unstable_by(|a, b| a.total_cmp(b));
    median(&deviations)
}

/// Gini coefficient from **sorted** non-negative data.
///
/// Gini = (2·Σ(i·xᵢ)) / (n·Σxᵢ) - (n+1)/n
///
/// Range [0, 1]: 0 = perfect equality, 1 = maximum inequality.
pub fn gini(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let weighted_sum: f64 = sorted.iter().enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * v)
        .sum();
    2.0 * weighted_sum / (n as f64 * sum) - (n as f64 + 1.0) / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, msg: &str) {
        if a.is_nan() && b.is_nan() { return; }
        assert!((a - b).abs() < tol,
            "{}: expected {}, got {} (diff {})", msg, b, a, (a - b).abs());
    }

    // ── Session sharing ───────────────────────────────────────────────

    #[test]
    fn moments_session_hit_returns_same_value() {
        use crate::intermediates::TamSession;
        let mut session = TamSession::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // First call: computes and registers.
        let s1 = moments_session(&mut session, &data);
        assert_eq!(session.len(), 1, "session should hold 1 intermediate after first call");

        // Second call on same data: session hit, zero re-computation.
        let s2 = moments_session(&mut session, &data);
        assert_eq!(session.len(), 1, "session should not grow on cache hit");

        // Results must be identical.
        assert_eq!(s1.count, s2.count);
        assert_eq!(s1.sum, s2.sum);
        assert_eq!(s1.m2, s2.m2);
    }

    #[test]
    fn moments_session_different_data_registers_separately() {
        use crate::intermediates::TamSession;
        let mut session = TamSession::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        moments_session(&mut session, &a);
        moments_session(&mut session, &b);
        assert_eq!(session.len(), 2, "different data → different tags → 2 entries");
    }

    // ── Ungrouped moments ─────────────────────────────────────────────

    #[test]
    fn moments_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = moments_ungrouped(&v);
        assert_eq!(s.count, 5.0);
        close(s.mean(), 3.0, 1e-10, "mean");
        close(s.min, 1.0, 1e-10, "min");
        close(s.max, 5.0, 1e-10, "max");
        close(s.variance(0), 2.0, 1e-10, "var_pop");
        close(s.variance(1), 2.5, 1e-10, "var_sample");
        close(s.std(0), 2.0f64.sqrt(), 1e-10, "std_pop");
        close(s.range(), 4.0, 1e-10, "range");
    }

    #[test]
    fn skewness_symmetric() {
        // Symmetric distribution → skewness = 0
        let s = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        close(s.skewness(true), 0.0, 1e-10, "skew_biased");
        close(s.skewness(false), 0.0, 1e-10, "skew_unbiased");
    }

    #[test]
    fn skewness_right_skewed() {
        // [1, 1, 1, 1, 10]: right-skewed
        let s = moments_ungrouped(&[1.0, 1.0, 1.0, 1.0, 10.0]);
        assert!(s.skewness(true) > 0.0, "right-skewed: positive skewness");

        // Verify by hand:
        // mean = 2.8, m2 = 64.8, m3 = 349.92
        // g1 = (349.92/5) / (64.8/5)^1.5 = 69.984 / 46.656 = 1.5
        close(s.skewness(true), 1.5, 1e-10, "skew_biased_right");
    }

    #[test]
    fn kurtosis_uniform() {
        // [1,2,3,4,5]: platykurtic (flatter than normal)
        // m2 = 10, m4 = 34
        // g2 = (34/5)/(10/5)² - 3 = 6.8/4 - 3 = 1.7 - 3 = -1.3
        let s = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        close(s.kurtosis(true, true), -1.3, 1e-10, "kurt_excess_biased");
    }

    #[test]
    fn kurtosis_leptokurtic() {
        // [0, 0, 0, 0, 100]: leptokurtic (heavy tails / peaked)
        let s = moments_ungrouped(&[0.0, 0.0, 0.0, 0.0, 100.0]);
        assert!(s.kurtosis(true, true) > 0.0, "leptokurtic: positive excess kurtosis");
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn empty_array() {
        let s = moments_ungrouped(&[]);
        assert_eq!(s.count, 0.0);
        assert!(s.mean().is_nan());
        assert!(s.variance(0).is_nan());
    }

    #[test]
    fn single_element() {
        let s = moments_ungrouped(&[42.0]);
        assert_eq!(s.count, 1.0);
        close(s.mean(), 42.0, 1e-10, "single_mean");
        close(s.variance(0), 0.0, 1e-10, "single_var_pop");
        assert!(s.variance(1).is_nan(), "sample variance undefined for n=1");
        assert!(s.skewness(false).is_nan(), "skewness undefined for n=1");
    }

    #[test]
    fn all_same_values() {
        let s = moments_ungrouped(&[7.0, 7.0, 7.0, 7.0]);
        close(s.mean(), 7.0, 1e-10, "same_mean");
        close(s.variance(0), 0.0, 1e-10, "same_var");
        assert!(s.skewness(true).is_nan(), "same_skew (var=0)");
        assert!(s.kurtosis(true, true).is_nan(), "same_kurt (var=0)");
    }

    #[test]
    fn nan_handling() {
        let s = moments_ungrouped(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        assert_eq!(s.count, 3.0);
        close(s.mean(), 3.0, 1e-10, "nan_mean");
        close(s.min, 1.0, 1e-10, "nan_min");
        close(s.max, 5.0, 1e-10, "nan_max");
    }

    #[test]
    fn all_nan() {
        let s = moments_ungrouped(&[f64::NAN, f64::NAN]);
        assert_eq!(s.count, 0.0);
        assert!(s.mean().is_nan());
    }

    #[test]
    fn inf_values() {
        let s = moments_ungrouped(&[1.0, f64::INFINITY, 3.0]);
        assert_eq!(s.count, 3.0);
        assert_eq!(s.max, f64::INFINITY);
    }

    #[test]
    fn two_elements() {
        let s = moments_ungrouped(&[1.0, 3.0]);
        close(s.mean(), 2.0, 1e-10, "two_mean");
        close(s.variance(0), 1.0, 1e-10, "two_var_pop");
        close(s.variance(1), 2.0, 1e-10, "two_var_sample");
        // Skewness: symmetric, should be 0
        close(s.skewness(true), 0.0, 1e-10, "two_skew");
        // Sample skewness undefined for n=2 (n-2 = 0)
        assert!(s.skewness(false).is_nan(), "two_skew_sample");
    }

    // ── Merge ─────────────────────────────────────────────────────────

    #[test]
    fn merge_recovers_full() {
        let full = moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = moments_ungrouped(&[1.0, 2.0, 3.0]);
        let b = moments_ungrouped(&[4.0, 5.0, 6.0]);
        let merged = MomentStats::merge(&a, &b);

        close(merged.count, full.count, 1e-10, "merge_count");
        close(merged.sum, full.sum, 1e-10, "merge_sum");
        close(merged.min, full.min, 1e-10, "merge_min");
        close(merged.max, full.max, 1e-10, "merge_max");
        close(merged.m2, full.m2, 1e-8, "merge_m2");
        close(merged.m3, full.m3, 1e-8, "merge_m3");
        close(merged.m4, full.m4, 1e-7, "merge_m4");
    }

    // ── DescriptiveResult ─────────────────────────────────────────────

    #[test]
    fn describe_output() {
        let engine = DescriptiveEngine::new();
        let r = engine.describe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(r.count, 5.0);
        close(r.mean, 3.0, 1e-10, "desc_mean");
        close(r.min, 1.0, 1e-10, "desc_min");
        close(r.max, 5.0, 1e-10, "desc_max");
        close(r.sum, 15.0, 1e-10, "desc_sum");
        close(r.range, 4.0, 1e-10, "desc_range");
    }

    // ── Quantiles ─────────────────────────────────────────────────────

    #[test]
    fn quantile_linear() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        close(quantile(&s, 0.0, QuantileMethod::Linear), 1.0, 1e-10, "q0");
        close(quantile(&s, 0.25, QuantileMethod::Linear), 2.0, 1e-10, "q25");
        close(quantile(&s, 0.5, QuantileMethod::Linear), 3.0, 1e-10, "q50");
        close(quantile(&s, 0.75, QuantileMethod::Linear), 4.0, 1e-10, "q75");
        close(quantile(&s, 1.0, QuantileMethod::Linear), 5.0, 1e-10, "q100");
    }

    #[test]
    fn quantile_interpolation() {
        // 4 elements: Q1 with Linear method = 1 + 0.25*3 = 1.75 → sorted[0.75]
        // sorted[0] * 0.25 + sorted[1] * 0.75 = 1*0.25 + 2*0.75 = 1.75
        let s = vec![1.0, 2.0, 3.0, 4.0];
        close(quantile(&s, 0.25, QuantileMethod::Linear), 1.75, 1e-10, "q25_interp");
        close(quantile(&s, 0.5, QuantileMethod::Linear), 2.5, 1e-10, "q50_interp");
    }

    #[test]
    fn median_odd() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
    }

    #[test]
    fn median_even() {
        close(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, 1e-10, "median_even");
    }

    #[test]
    fn iqr_basic() {
        close(iqr(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.0, 1e-10, "iqr");
    }

    // ── Central tendency variants ─────────────────────────────────────

    #[test]
    fn geometric_mean_basic() {
        // GM(1,2,4,8) = (64)^(1/4) = 2√2
        close(geometric_mean(&[1.0, 2.0, 4.0, 8.0]), 64.0f64.powf(0.25), 1e-10, "geom_mean");
    }

    #[test]
    fn geometric_mean_negative() {
        assert!(geometric_mean(&[1.0, -1.0, 4.0]).is_nan());
    }

    #[test]
    fn harmonic_mean_basic() {
        // HM(1,2,4) = 3 / (1 + 0.5 + 0.25) = 3/1.75
        close(harmonic_mean(&[1.0, 2.0, 4.0]), 3.0 / 1.75, 1e-10, "harm_mean");
    }

    #[test]
    fn trimmed_mean_basic() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        // Trim 20% from each end: remove 1 element from each → [2,3,4], mean=3
        close(trimmed_mean(&s, 0.2), 3.0, 1e-10, "trimmed_20");
    }

    #[test]
    fn winsorized_mean_basic() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        // Winsorize 20%: [2,2,3,4,4], mean = 15/5 = 3.0
        close(winsorized_mean(&s, 0.2), 3.0, 1e-10, "winsorized_20");
    }

    // ── Dispersion variants ───────────────────────────────────────────

    #[test]
    fn mad_basic() {
        // [1,2,3,4,5]: median=3, deviations=[2,1,0,1,2], sorted=[0,1,1,2,2], median=1
        close(mad(&[1.0, 2.0, 3.0, 4.0, 5.0]), 1.0, 1e-10, "mad");
    }

    #[test]
    fn gini_equal() {
        close(gini(&[1.0, 1.0, 1.0, 1.0]), 0.0, 1e-10, "gini_equal");
    }

    #[test]
    fn gini_unequal() {
        // [0,0,0,100]: Gini = 2*(1*0+2*0+3*0+4*100)/(4*100) - 5/4
        // = 2*400/400 - 1.25 = 2.0 - 1.25 = 0.75
        close(gini(&[0.0, 0.0, 0.0, 100.0]), 0.75, 1e-10, "gini_unequal");
    }

    // ── Skewness variants ─────────────────────────────────────────────

    #[test]
    fn bowley_symmetric() {
        close(bowley_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]), 0.0, 1e-10, "bowley_sym");
    }

    #[test]
    fn pearson_first_symmetric() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = moments_ungrouped(&vals);
        let sorted = sorted_nan_free(&vals);
        let med = median(&sorted);
        close(pearson_first_skewness(&stats, med), 0.0, 1e-10, "pearson1_sym");
    }

    // ── Grouped (engine-based) ────────────────────────────────────────

    #[test]
    fn grouped_basic() {
        let mut engine = DescriptiveEngine::new();
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let keys = vec![0i32, 0, 0, 1, 1, 1];
        let stats = engine.moments_grouped(&vals, &keys, 2).unwrap();

        // Group 0: [1,2,3] → mean=2, var_pop = (1+0+1)/3 = 2/3
        let g0 = stats.group(0);
        close(g0.count, 3.0, 1e-10, "g0_count");
        close(g0.mean(), 2.0, 1e-10, "g0_mean");
        close(g0.variance(0), 2.0 / 3.0, 1e-10, "g0_var_pop");
        close(g0.min, 1.0, 1e-10, "g0_min");
        close(g0.max, 3.0, 1e-10, "g0_max");

        // Group 1: [4,5,6] → mean=5, var_pop = 2/3
        let g1 = stats.group(1);
        close(g1.count, 3.0, 1e-10, "g1_count");
        close(g1.mean(), 5.0, 1e-10, "g1_mean");
        close(g1.variance(0), 2.0 / 3.0, 1e-10, "g1_var_pop");
    }

    #[test]
    fn grouped_with_nan() {
        let mut engine = DescriptiveEngine::new();
        let vals = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN];
        let keys = vec![0i32, 0, 0, 1, 1, 1];
        let stats = engine.moments_grouped(&vals, &keys, 2).unwrap();

        // Group 0: [1, NaN, 3] → valid: [1,3], count=2, mean=2
        let g0 = stats.group(0);
        close(g0.count, 2.0, 1e-10, "g0_nan_count");
        close(g0.mean(), 2.0, 1e-10, "g0_nan_mean");

        // Group 1: [4, 5, NaN] → valid: [4,5], count=2, mean=4.5
        let g1 = stats.group(1);
        close(g1.count, 2.0, 1e-10, "g1_nan_count");
        close(g1.mean(), 4.5, 1e-10, "g1_nan_mean");
    }

    #[test]
    fn grouped_mean_shortcut() {
        let mut engine = DescriptiveEngine::new();
        let vals = vec![10.0, 20.0, 30.0, 40.0];
        let keys = vec![0i32, 1, 0, 1];
        let means = engine.mean_grouped(&vals, &keys, 2).unwrap();
        close(means[0], 20.0, 1e-10, "mean_g0"); // (10+30)/2
        close(means[1], 30.0, 1e-10, "mean_g1"); // (20+40)/2
    }

    #[test]
    fn grouped_higher_moments() {
        let mut engine = DescriptiveEngine::new();
        // Group 0: [1,2,3,4,5] → symmetric → skewness = 0
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0];
        let keys = vec![0i32, 0, 0, 0, 0, 1, 1, 1];
        let stats = engine.moments_grouped(&vals, &keys, 2).unwrap();

        let g0 = stats.group(0);
        close(g0.skewness(true), 0.0, 1e-10, "g0_skew");
        close(g0.kurtosis(true, true), -1.3, 1e-10, "g0_kurt");
    }

    // ── sorted_nan_free ───────────────────────────────────────────────

    #[test]
    fn sorted_nan_free_basic() {
        let sorted = sorted_nan_free(&[3.0, f64::NAN, 1.0, f64::NAN, 2.0]);
        assert_eq!(sorted, vec![1.0, 2.0, 3.0]);
    }

    // ── CV and SEM ────────────────────────────────────────────────────

    #[test]
    fn cv_and_sem() {
        let s = moments_ungrouped(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        // mean = 30, std_sample = sqrt(250) = 15.811..., cv = 15.811/30 = 0.5270
        close(s.cv(), s.std(1) / s.mean().abs(), 1e-10, "cv");
        close(s.sem(), s.std(1) / 5.0f64.sqrt(), 1e-10, "sem");
    }

    // ── Collatz Stopping Time Distribution (Prize Expedition, Exp 1) ──

    /// Collatz stopping time: steps until n reaches 1.
    fn collatz_stopping_time(mut n: u64) -> u32 {
        let mut steps = 0u32;
        while n > 1 {
            if n % 2 == 0 { n /= 2; } else { n = 3 * n + 1; }
            steps += 1;
        }
        steps
    }

    #[test]
    fn collatz_stopping_time_distribution() {
        let n_max: u64 = 100_000;

        // Compute stopping times for n = 2..N
        let stop_times: Vec<f64> = (2..=n_max)
            .map(|n| collatz_stopping_time(n) as f64)
            .collect();

        // Raw stopping time moments (Kingdom A: one-pass accumulate)
        let stats = moments_ungrouped(&stop_times);

        eprintln!("\n=== Collatz Stopping Time Distribution (n=2..{}) ===", n_max);
        eprintln!("  Count: {}", stats.n());
        eprintln!("  Mean:  {:.2}", stats.mean());
        eprintln!("  Std:   {:.2}", stats.std(1));
        eprintln!("  Min:   {:.0}", stats.min);
        eprintln!("  Max:   {:.0}", stats.max);
        eprintln!("  Skew:  {:.4}", stats.skewness(false));
        eprintln!("  Kurt:  {:.4}", stats.kurtosis(true, false));

        // Terras (1976): E[stopping_time] ≈ 9.48 · log₂(n) for typical n
        let terras_ratio = stats.mean() / (n_max as f64).log2();
        eprintln!("  Terras ratio: mean/log₂(N) = {:.4} (expected ≈ 9.48)", terras_ratio);

        // Log-transform: test lognormality
        let log_stop: Vec<f64> = stop_times.iter()
            .filter(|&&s| s > 0.0)
            .map(|&s| s.ln())
            .collect();
        let log_stats = moments_ungrouped(&log_stop);

        eprintln!("\n  Log-transformed:");
        eprintln!("  Skew(ln s): {:.4} (0 = Gaussian → lognormal stopping times)", log_stats.skewness(false));
        eprintln!("  Kurt(ln s): {:.4} (0 = Gaussian → lognormal stopping times)", log_stats.kurtosis(true, false));

        let is_approx_lognormal = log_stats.skewness(false).abs() < 0.5
            && log_stats.kurtosis(true, false).abs() < 1.0;
        eprintln!("  Lognormal: {}", if is_approx_lognormal { "approximately ✓" } else { "no" });

        // KS test: standardize log(s) and test against standard normal
        let mu = log_stats.mean();
        let sigma = log_stats.std(1);
        let z_scores: Vec<f64> = log_stop.iter().map(|&x| (x - mu) / sigma).collect();
        let ks = crate::nonparametric::ks_test_normal(&z_scores);
        eprintln!("\n  KS test (log-stopping times vs normal):");
        eprintln!("  D = {:.6}, p = {:.6}", ks.statistic, ks.p_value);
        if ks.p_value < 0.05 {
            eprintln!("  → Rejects normality at 5% level (not exactly lognormal)");
        } else {
            eprintln!("  → Fails to reject normality (consistent with lognormal)");
        }

        // Assertions
        assert!(stats.mean() > 50.0, "mean stopping time too small");
        assert!(stats.max > 200.0, "max stopping time too small");
        assert!(terras_ratio > 5.0 && terras_ratio < 15.0,
            "Terras ratio {} outside expected range", terras_ratio);
    }

    /// Run density diffusion at a given N, return (escaped_frac, |lambda_2|, spectral_gap).
    fn diffusion_spectral_gap(n_max: usize, steps: usize) -> (f64, f64, f64) {
        let mut density = vec![0.0f64; n_max + 1];
        for i in 2..=n_max {
            density[i] = 1.0 / (n_max - 1) as f64;
        }
        let mut mass_at_1 = vec![0.0f64; steps];
        let mut total_mass = vec![0.0f64; steps];

        for step in 0..steps {
            let mut next = vec![0.0f64; n_max + 1];
            for n in 2..=n_max {
                if density[n] == 0.0 { continue; }
                let target = if n % 2 == 0 { n / 2 } else { (3 * n + 1) / 2 };
                if target <= n_max {
                    next[target] += density[n];
                }
            }
            next[1] += density[1];
            density = next;
            mass_at_1[step] = density[1];
            total_mass[step] = density.iter().sum::<f64>();
        }

        let escaped = 1.0 - total_mass[steps - 1];

        // Fit exponential decay to residual (non-absorbed, non-escaped mass)
        // Use window where residual is between 1e-12 and 0.5 (skip transient + noise)
        let mut log_residuals: Vec<(f64, f64)> = Vec::new();
        for k in 10..steps {
            let r = total_mass[k] - mass_at_1[k];
            if r > 1e-12 && r < 0.5 {
                log_residuals.push((k as f64, r.ln()));
            }
        }

        if log_residuals.len() < 10 {
            return (escaped, f64::NAN, f64::NAN);
        }

        let n_pts = log_residuals.len() as f64;
        let sx: f64 = log_residuals.iter().map(|p| p.0).sum();
        let sy: f64 = log_residuals.iter().map(|p| p.1).sum();
        let sxy: f64 = log_residuals.iter().map(|p| p.0 * p.1).sum();
        let sx2: f64 = log_residuals.iter().map(|p| p.0 * p.0).sum();
        let slope = (n_pts * sxy - sx * sy) / (n_pts * sx2 - sx * sx);
        let lambda2 = slope.exp();
        (escaped, lambda2, 1.0 - lambda2)
    }

    #[test]
    fn collatz_density_diffusion() {
        eprintln!("\n=== Collatz Spectral Gap vs N ===");
        eprintln!("  {:>8}  {:>8}  {:>8}  {:>8}", "N", "escaped", "|λ₂|", "gap");

        for &n_max in &[1_000usize, 3_000, 10_000, 30_000, 100_000] {
            let steps = 500;
            let (escaped, lambda2, gap) = diffusion_spectral_gap(n_max, steps);
            eprintln!("  {:>8}  {:>8.4}  {:>8.6}  {:>8.6}", n_max, escaped, lambda2, gap);
        }

        // Primary assertion: spectral gap exists at N=10K
        let (_, lambda2, gap) = diffusion_spectral_gap(10_000, 300);
        assert!(gap > 0.01, "spectral gap should be positive, got {}", gap);
        assert!(lambda2 < 1.0, "|lambda2| should be < 1, got {}", lambda2);
    }

    /// Apply one step of the Collatz density operator T to a density vector.
    /// Returns a new density vector after one diffusion step.
    fn collatz_operator_apply(density: &[f64], n_max: usize) -> Vec<f64> {
        let mut next = vec![0.0f64; n_max + 1];
        for n in 2..=n_max {
            if density[n] == 0.0 { continue; }
            let target = if n % 2 == 0 { n / 2 } else { (3 * n + 1) / 2 };
            if target <= n_max {
                next[target] += density[n];
            }
        }
        next[1] += density[1]; // mass at 1 is absorbing
        next
    }

    /// Chebyshev polynomial T_d(x) evaluated via recurrence.
    fn chebyshev_poly(d: usize, x: f64) -> f64 {
        if d == 0 { return 1.0; }
        if d == 1 { return x; }
        let mut t_prev = 1.0;
        let mut t_curr = x;
        for _ in 2..=d {
            let t_next = 2.0 * x * t_curr - t_prev;
            t_prev = t_curr;
            t_curr = t_next;
        }
        t_curr
    }

    /// Apply degree-4 Chebyshev filter p(T) via direct polynomial evaluation.
    ///
    /// p(λ) = T_4((2λ/λ₂ - 1)) / T_4(2/λ₂ - 1)
    ///
    /// Expanding T_4(cλ-1) with c=2/λ₂:
    ///   p(T)·ρ = (8c⁴·T⁴ρ - 32c³·T³ρ + 40c²·T²ρ - 16c·Tρ + ρ) / T_4(c-1)
    ///
    /// Direct evaluation avoids the numerical instability of the three-term
    /// recurrence on this non-symmetric operator.
    fn chebyshev_filter_degree4(density: &[f64], n_max: usize, lambda2: f64) -> Vec<f64> {
        let c = 2.0 / lambda2;
        let c2 = c * c;
        let c3 = c2 * c;
        let c4 = c3 * c;
        let norm = chebyshev_poly(4, c - 1.0);

        let a0 = 1.0;
        let a1 = -16.0 * c;
        let a2 = 40.0 * c2;
        let a3 = -32.0 * c3;
        let a4 = 8.0 * c4;

        let t0 = density.to_vec();
        let t1 = collatz_operator_apply(&t0, n_max);
        let t2 = collatz_operator_apply(&t1, n_max);
        let t3 = collatz_operator_apply(&t2, n_max);
        let t4 = collatz_operator_apply(&t3, n_max);

        (0..=n_max).map(|i| {
            (a0 * t0[i] + a1 * t1[i] + a2 * t2[i] + a3 * t3[i] + a4 * t4[i]) / norm
        }).collect()
    }

    #[test]
    fn chebyshev_amplification() {
        // Chebyshev-accelerated density diffusion diagnostic.
        //
        // p(λ) = T_4((2λ/λ₂-1)) / T_4(2/λ₂-1) satisfies:
        //   p(1) = 1, |p(λ)| ≤ 1/T_4(σ(1)) ≈ 0.245 for λ ∈ [0, λ₂]
        //
        // If the operator has eigenvalues outside [0, λ₂], the filter will
        // amplify those components — this is itself a structural finding.

        let n_max = 10_000usize;
        let lambda2 = 0.935;
        let degree = 4usize;

        let sigma_1 = 2.0 / lambda2 - 1.0;
        let td = chebyshev_poly(degree, sigma_1);

        eprintln!("\n=== Chebyshev Amplification (N={}, λ₂={}) ===", n_max, lambda2);
        eprintln!("  σ(1) = {:.4}, T_4(σ(1)) = {:.4}", sigma_1, td);
        eprintln!("  Predicted decay per super-step: {:.4}", 1.0 / td);
        eprintln!("  Naive decay per 4 steps: {:.4}", lambda2.powi(degree as i32));
        eprintln!("  Predicted speedup: {:.2}×", lambda2.powi(degree as i32) * td);

        // Initial density: uniform on [2, n_max]
        let mut naive_rho = vec![0.0f64; n_max + 1];
        let mut cheb_rho = vec![0.0f64; n_max + 1];
        for i in 2..=n_max {
            naive_rho[i] = 1.0 / (n_max - 1) as f64;
            cheb_rho[i] = 1.0 / (n_max - 1) as f64;
        }

        let total_super_steps = 25; // 25 × 4 = 100 T-applications
        let total_naive_steps = total_super_steps * degree;

        let mut naive_residuals = Vec::new();
        let mut cheb_residuals = Vec::new();

        // Naive iteration
        for step in 0..total_naive_steps {
            naive_rho = collatz_operator_apply(&naive_rho, n_max);
            if (step + 1) % degree == 0 {
                let total: f64 = naive_rho.iter().sum();
                let at_1 = naive_rho[1];
                naive_residuals.push((step + 1, (total - at_1).max(0.0)));
            }
        }

        // Chebyshev iteration — pure linear operator, no clamping
        for ss in 0..total_super_steps {
            cheb_rho = chebyshev_filter_degree4(&cheb_rho, n_max, lambda2);
            let equiv_step = (ss + 1) * degree;
            let total: f64 = cheb_rho.iter().sum();
            let at_1 = cheb_rho[1];
            cheb_residuals.push((equiv_step, total - at_1));
        }

        eprintln!("\n  {:>6}  {:>14}  {:>14}  {:>10}",
            "step", "naive_resid", "cheb_resid", "cheb/naive");
        for (naive, cheb) in naive_residuals.iter().zip(cheb_residuals.iter()) {
            let ratio = if naive.1.abs() > 1e-20 { cheb.1 / naive.1 } else { f64::NAN };
            if naive.0 % 20 == 0 || naive.0 <= 8 {
                eprintln!("  {:>6}  {:>14.6e}  {:>14.6e}  {:>10.4}",
                    naive.0, naive.1, cheb.1, ratio);
            }
        }

        eprintln!("\n  Mass at n=1 after {} T-applications:", total_naive_steps);
        eprintln!("    Naive:     {:.6}", naive_rho[1]);
        eprintln!("    Chebyshev: {:.6}", cheb_rho[1]);

        let last_naive = naive_residuals.last().unwrap().1;
        let last_cheb = cheb_residuals.last().unwrap().1;
        eprintln!("\n  Final residuals:");
        eprintln!("    Naive:     {:.6e}", last_naive);
        eprintln!("    Chebyshev: {:.6e}", last_cheb);

        let neg_count = cheb_rho.iter().filter(|&&v| v < -1e-15).count();
        eprintln!("  Negative density components: {}", neg_count);

        if last_cheb.abs() < last_naive {
            eprintln!("  ✓ Chebyshev filter provides amplification");
        } else if last_cheb.abs() > 1e10 {
            eprintln!("  ✗ Filter DIVERGES — T has spectral components outside [0, λ₂]");
            eprintln!("    (Likely real NEGATIVE eigenvalues; see symmetric filter below)");
        } else {
            eprintln!("  ~ Filter not faster — spectrum partially outside [0, λ₂]");
        }

        // Part 2: Symmetric filter for [-λ₂, λ₂]
        // If the operator has real negative eigenvalues, a filter designed for
        // the symmetric interval [-λ₂, λ₂] should work. The affine map becomes
        // σ(λ) = λ/λ₂, and σ(1) = 1/λ₂ ≈ 1.070.
        //
        // The even-degree Chebyshev polynomial T_4(x) treats ±x symmetrically,
        // so it equally suppresses positive and negative eigenvalues.
        //
        // p_sym(T) = T_4(T/λ₂) / T_4(1/λ₂)
        // Expansion: T_4(λ/λ₂) = 8λ⁴/λ₂⁴ - 8λ²/λ₂² + 1

        let sym_sigma1 = 1.0 / lambda2;
        let sym_td = chebyshev_poly(degree, sym_sigma1);
        eprintln!("\n=== Symmetric Chebyshev Filter [-λ₂, λ₂] ===");
        eprintln!("  σ(1) = {:.4}, T_4(σ(1)) = {:.4}", sym_sigma1, sym_td);
        eprintln!("  Predicted speedup: {:.2}×", lambda2.powi(degree as i32) * sym_td);

        let c_sym = 1.0 / lambda2;
        let c2s = c_sym * c_sym;
        let c4s = c2s * c2s;
        let a0s = 1.0;
        let a2s = -8.0 * c2s;
        let a4s = 8.0 * c4s;

        // Re-initialize
        let mut sym_rho = vec![0.0f64; n_max + 1];
        for i in 2..=n_max {
            sym_rho[i] = 1.0 / (n_max - 1) as f64;
        }

        let mut sym_residuals = Vec::new();
        for ss in 0..total_super_steps {
            // p_sym(T)·ρ = (a4·T⁴ρ + a2·T²ρ + a0·ρ) / T_4(1/λ₂)
            // Note: T_4 is even, so T_4(x) = 8x⁴ - 8x² + 1. Only even powers of T.
            let t0 = sym_rho.clone();
            let t1 = collatz_operator_apply(&t0, n_max);
            let t2 = collatz_operator_apply(&t1, n_max);
            let t3 = collatz_operator_apply(&t2, n_max);
            let t4 = collatz_operator_apply(&t3, n_max);

            sym_rho = (0..=n_max).map(|i| {
                (a0s * t0[i] + a2s * t2[i] + a4s * t4[i]) / sym_td
            }).collect();

            let equiv_step = (ss + 1) * degree;
            let total: f64 = sym_rho.iter().sum();
            let at_1 = sym_rho[1];
            sym_residuals.push((equiv_step, total - at_1));
        }

        eprintln!("\n  {:>6}  {:>14}  {:>14}  {:>10}",
            "step", "naive_resid", "sym_resid", "sym/naive");
        for (naive, sym) in naive_residuals.iter().zip(sym_residuals.iter()) {
            let ratio = if naive.1.abs() > 1e-20 { sym.1 / naive.1 } else { f64::NAN };
            if naive.0 % 20 == 0 || naive.0 <= 8 {
                eprintln!("  {:>6}  {:>14.6e}  {:>14.6e}  {:>10.4}",
                    naive.0, naive.1, sym.1, ratio);
            }
        }

        let last_sym = sym_residuals.last().unwrap().1;
        eprintln!("\n  Symmetric filter final residual: {:.6e}", last_sym);
        if last_sym.abs() < last_naive && last_sym.abs() < 1e10 {
            eprintln!("  ✓ Symmetric filter WORKS — confirms negative eigenvalues");
            eprintln!("    (Asymmetric [0,λ₂] failed; symmetric [-λ₂,λ₂] succeeds)");
        } else if last_sym.abs() > 1e10 {
            eprintln!("  ✗ Symmetric filter also diverges — eigenvalues outside [-λ₂,λ₂]");
        } else {
            eprintln!("  ~ Symmetric filter comparable to naive");
        }
    }

    #[test]
    fn collatz_operator_complex_eigenvalue_probe() {
        // The Chebyshev experiment showed the Collatz density operator has spectral
        // components outside [0, λ₂]. If eigenvalues are complex (λ = r·e^{iθ}),
        // the step-by-step residual will OSCILLATE around the exponential decay trend.
        //
        // Method: compute log(residual_k) - slope·k = detrended log-residual.
        // If purely real spectrum: flat (no oscillation).
        // If complex spectrum: periodic oscillation with period 2π/θ.
        //
        // Detect oscillation via autocorrelation at lag > 0.

        let n_max = 5_000usize;
        let steps = 200;

        let mut density = vec![0.0f64; n_max + 1];
        for i in 2..=n_max {
            density[i] = 1.0 / (n_max - 1) as f64;
        }

        let mut residuals = Vec::new();
        for _step in 0..steps {
            density = collatz_operator_apply(&density, n_max);
            let total: f64 = density.iter().sum();
            let at_1 = density[1];
            let residual = total - at_1;
            if residual > 1e-15 {
                residuals.push(residual);
            }
        }

        // Compute log-residuals
        let log_res: Vec<f64> = residuals.iter()
            .map(|r| r.ln())
            .collect();

        // Fit linear trend: log_res ≈ a + b·k
        let n = log_res.len();
        let n_f = n as f64;
        let sx: f64 = (0..n).map(|i| i as f64).sum();
        let sy: f64 = log_res.iter().sum();
        let sxy: f64 = (0..n).map(|i| i as f64 * log_res[i]).sum();
        let sx2: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();
        let slope = (n_f * sxy - sx * sy) / (n_f * sx2 - sx * sx);
        let intercept = (sy - slope * sx) / n_f;

        // Detrend: remove exponential decay
        let detrended: Vec<f64> = (0..n)
            .map(|i| log_res[i] - (intercept + slope * i as f64))
            .collect();

        // Compute autocorrelation of detrended signal at various lags
        let mean_dt = detrended.iter().sum::<f64>() / n_f;
        let var_dt = detrended.iter().map(|x| (x - mean_dt).powi(2)).sum::<f64>() / n_f;

        eprintln!("\n=== Collatz Operator Complex Eigenvalue Probe (N={}) ===", n_max);
        eprintln!("  Fitted decay rate: |λ₂| ≈ {:.6}", slope.exp());
        eprintln!("  Detrended signal variance: {:.6e}", var_dt);

        eprintln!("\n  {:>4}  {:>10}", "lag", "autocorr");

        let mut max_corr_lag = 0;
        let mut max_corr_val = 0.0f64;

        for lag in 1..=30 {
            if lag >= n { break; }
            let acf: f64 = (0..n - lag)
                .map(|i| (detrended[i] - mean_dt) * (detrended[i + lag] - mean_dt))
                .sum::<f64>() / ((n - lag) as f64 * var_dt);
            eprintln!("  {:>4}  {:>10.4}", lag, acf);
            if acf.abs() > max_corr_val.abs() && lag > 1 {
                max_corr_val = acf;
                max_corr_lag = lag;
            }
        }

        eprintln!("\n  Peak autocorrelation at lag {}: {:.4}", max_corr_lag, max_corr_val);
        if max_corr_val.abs() > 0.3 {
            let period = max_corr_lag as f64;
            let theta = 2.0 * std::f64::consts::PI / period;
            eprintln!("  → OSCILLATION DETECTED: period ≈ {} steps", max_corr_lag);
            eprintln!("  → Complex eigenvalue angle θ ≈ {:.4} rad ({:.1}°)",
                theta, theta.to_degrees());
            eprintln!("  → λ₂ ≈ {:.4} · e^(i·{:.4})", slope.exp(), theta);
        } else {
            eprintln!("  → No significant oscillation (eigenvalues likely real or multi-modal)");
        }

        // First few detrended values for visual inspection
        eprintln!("\n  Detrended log-residual (first 20 steps):");
        for i in 0..20.min(n) {
            let bar_len = ((detrended[i] / 0.01).clamp(-30.0, 30.0)) as i32;
            let bar: String = if bar_len >= 0 {
                " ".repeat(30) + &"█".repeat(bar_len as usize)
            } else {
                " ".repeat((30 + bar_len).max(0) as usize)
                    + &"█".repeat((-bar_len) as usize)
            };
            eprintln!("  {:>3} {:+.6} {}", i, detrended[i], bar);
        }
    }

    #[test]
    fn collatz_all_ones_escape() {
        // Verify: starting from n = 2^k - 1, the trajectory takes exactly k steps
        // to escape the all-ones neighborhood (lose all trailing 1-bits).
        //
        // Prediction: n_j = 3^j · 2^{k-j} - 1 has exactly k-j trailing 1-bits.
        eprintln!("\n=== All-Ones Escape Verification ===");
        eprintln!("  {:>4}  {:>10}  {:>12}  {:>6}  {:>12}  {:>6}",
            "k", "n₀=2^k-1", "n_k=3^k-1", "steps", "growth", "predicted");

        for k in 2..=20u32 {
            let n0: u128 = (1u128 << k) - 1;
            let mut n = n0;
            let mut steps = 0u32;

            // Run Collatz until we lose all trailing 1-bits
            while n & 1 == 1 && steps < k + 5 {
                n = (3 * n + 1) / 2;  // compressed map
                steps += 1;
            }

            // Verify: should take exactly k steps
            let predicted_nk = 3u128.pow(k) - 1;
            let actual_growth = n0 as f64 / 1.0; // growth = n_k / n_0
            let growth = n as f64 / n0 as f64;
            let predicted_growth = (1.5f64).powi(k as i32);

            eprintln!("  {:>4}  {:>10}  {:>12}  {:>6}  {:>12.2}  {:>12.2}",
                k, n0, n, steps, growth, predicted_growth);

            if k <= 16 {
                assert_eq!(steps, k, "escape should take exactly k={} steps", k);
                // After exactly k steps of v₂=1, result should be 3^k - 1
                assert_eq!(n, predicted_nk, "n_k should be 3^{}-1={}", k, predicted_nk);
            }
        }
    }

    /// Kreiss constant computation for truncated Collatz density operator.
    ///
    /// The Kreiss constant K(N) = max_{k≥0} ||T^k||_∞ / ρ(T)^k measures the
    /// worst-case transient amplification relative to the spectral radius.
    /// For a normal operator, K = 1. For the Collatz operator (non-normal due
    /// to sub-stochastic truncation), K may be large.
    ///
    /// Key question: does K(N) grow as N^{0.585} (matching the all-ones bound)
    /// or faster? If faster, there are directions worse than all-ones.
    #[test]
    fn collatz_kreiss_constant() {
        eprintln!("\n=== Collatz Kreiss Constant K(N) ===");
        eprintln!("  {:>4}  {:>10}  {:>10}  {:>12}  {:>12}  {:>10}",
            "N", "ρ(T)≈", "||T||_∞", "K(N)", "N^0.585", "K/N^0.585");

        for &n_max in &[16, 32, 64, 128] {
            // Build the (n_max-1) × (n_max-1) transition matrix T on {2,...,n_max}.
            // T[i][j] = probability that state j+2 transitions to state i+2.
            // The Collatz map: even n → n/2, odd n → (3n+1)/2.
            // This is deterministic, so T is a permutation-like matrix (with truncation).
            let dim = n_max - 1; // states 2..=n_max, indexed 0..dim-1
            let mut t_mat = vec![0.0f64; dim * dim]; // row-major

            for col in 0..dim {
                let n = col + 2; // state n
                let target = if n % 2 == 0 { n / 2 } else { (3 * n + 1) / 2 };
                if target >= 2 && target <= n_max {
                    let row = target - 2;
                    t_mat[row * dim + col] = 1.0;
                }
                // If target < 2 (reached 1) or target > n_max (escaped), mass is lost.
                // This is the sub-stochastic truncation that creates non-normality.
            }

            // Compute ||T||_∞ (max row sum)
            let t_inf_norm = (0..dim).map(|r| {
                (0..dim).map(|c| t_mat[r * dim + c].abs()).sum::<f64>()
            }).fold(0.0f64, f64::max);

            // Compute T^k and ||T^k||_∞ for k = 1..60
            // T^k = T · T^{k-1} (matrix multiplication)
            let mut t_power = t_mat.clone(); // T^1
            let mut max_norm = t_inf_norm;   // ||T^1||_∞
            let mut kreiss_ratio = 1.0f64;

            // Estimate spectral radius from ||T^k||^{1/k} as k grows
            let mut spectral_radius_est = t_inf_norm;

            for k in 2..=60usize {
                // Compute T^k = T · T^{k-1}
                let mut next = vec![0.0f64; dim * dim];
                for i in 0..dim {
                    for j in 0..dim {
                        let mut sum = 0.0;
                        for p in 0..dim {
                            sum += t_mat[i * dim + p] * t_power[p * dim + j];
                        }
                        next[i * dim + j] = sum;
                    }
                }
                t_power = next;

                // ||T^k||_∞
                let norm_k = (0..dim).map(|r| {
                    (0..dim).map(|c| t_power[r * dim + c].abs()).sum::<f64>()
                }).fold(0.0f64, f64::max);

                if norm_k > max_norm { max_norm = norm_k; }

                // Update spectral radius estimate: ρ ≈ ||T^k||^{1/k}
                if norm_k > 0.0 {
                    let rho_k = norm_k.powf(1.0 / k as f64);
                    if k >= 10 { // stabilize after initial transient
                        spectral_radius_est = spectral_radius_est.min(rho_k);
                    }
                }

                // Kreiss ratio: ||T^k|| / ρ^k
                if spectral_radius_est > 1e-15 {
                    let ratio = norm_k / spectral_radius_est.powi(k as i32);
                    if ratio > kreiss_ratio { kreiss_ratio = ratio; }
                }
            }

            let n_bound = (n_max as f64).powf(0.585); // all-ones prediction

            eprintln!("  {:>4}  {:>10.4}  {:>10.1}  {:>12.2}  {:>12.2}  {:>10.3}",
                n_max, spectral_radius_est, t_inf_norm, kreiss_ratio, n_bound,
                kreiss_ratio / n_bound);
        }

        // Detailed norm decay for N=64
        eprintln!("\n=== Norm Decay Sequence for N=64 ===");
        let n_max = 64usize;
        let dim = n_max - 1;
        let mut t_mat = vec![0.0f64; dim * dim];
        for col in 0..dim {
            let n = col + 2;
            let target = if n % 2 == 0 { n / 2 } else { (3 * n + 1) / 2 };
            if target >= 2 && target <= n_max {
                t_mat[(target - 2) * dim + col] = 1.0;
            }
        }

        // Track ||T^k||_∞ and ||T^k||_1 (max column sum) for k=1..100
        let mut t_power = t_mat.clone();
        eprintln!("  {:>4}  {:>12}  {:>12}  {:>12}",
            "k", "||T^k||_∞", "||T^k||^{1/k}", "max_col_sum");
        for k in 1..=100usize {
            if k > 1 {
                let mut next = vec![0.0f64; dim * dim];
                for i in 0..dim {
                    for j in 0..dim {
                        let mut sum = 0.0;
                        for p in 0..dim {
                            sum += t_mat[i * dim + p] * t_power[p * dim + j];
                        }
                        next[i * dim + j] = sum;
                    }
                }
                t_power = next;
            }
            let inf_norm = (0..dim).map(|r| {
                (0..dim).map(|c| t_power[r * dim + c].abs()).sum::<f64>()
            }).fold(0.0f64, f64::max);
            let col_max = (0..dim).map(|c| {
                (0..dim).map(|r| t_power[r * dim + c].abs()).sum::<f64>()
            }).fold(0.0f64, f64::max);
            let rho_k = if inf_norm > 0.0 { inf_norm.powf(1.0 / k as f64) } else { 0.0 };

            if k <= 20 || k % 10 == 0 {
                eprintln!("  {:>4}  {:>12.4}  {:>12.6}  {:>12.4}", k, inf_norm, rho_k, col_max);
            }
        }

        assert!(true, "Kreiss constant diagnostic completed");
    }
}
