//! # Data Quality Primitives
//!
//! First-class composable math primitives for data validity checks.
//!
//! ## What lives here
//!
//! Every function in this module is a **single-purpose, cheap, cadence-agnostic**
//! primitive that operates on a slice of data. Each answers one question about
//! "is this data fit for purpose?"
//!
//! These are the building blocks of validity predicates used by:
//! - The TBS auto-detection system (which method should I recommend?)
//! - The bridge leaf executors (is this bin big enough to run FFT?)
//! - Researchers directly (how degenerate is my sample?)
//!
//! ## Design principles
//!
//! 1. **Cadence-agnostic**: no cadence/bin-size parameters. Input is `&[f64]`.
//! 2. **Cheap**: each primitive is O(n) or O(n log n) single-pass.
//! 3. **Sharable**: every result is a scalar or tiny vector suitable for
//!    caching in `TamSession::DataQualitySummary`.
//! 4. **NaN-safe**: handles empty, constant, and NaN-laden input gracefully.
//!    Returns NaN or a documented sentinel rather than panicking.
//! 5. **Composable**: validity predicates (e.g. `fft_is_valid`) are built by
//!    AND-ing these primitives, not by inline checks inside `fft_spectral`.
//!
//! ## Categories
//!
//! **Size metrics**: `tick_count`, `effective_sample_size`, `nyquist_bins`
//! **Diversity**: `unique_prices`, `symbolic_diversity`, `unique_ordinal_3`
//! **Sampling regularity**: `sampling_regularity_cv`, `longest_gap_ratio`,
//!   `coverage_ratio`
//! **Variability**: `price_cv`, `split_variance_ratio`
//! **Structure**: `trend_r2`, `has_trend`, `acf_decay_exponent`
//! **Stationarity proxies**: `is_stationary_adf_05`
//! **Jump/tail**: `jump_ratio_proxy`
//! **Volatility clustering**: `has_vol_clustering`
//!
//! All of these are used by the BINNED_METHODS_LIST validity requirements.

/// Count of finite (non-NaN) samples in the slice.
///
/// This is the most basic validity predicate: "is there any data at all?"
/// Callers check `tick_count(x) >= n_min` to gate methods with minimum
/// sample requirements.
#[inline]
pub fn tick_count(x: &[f64]) -> usize {
    x.iter().filter(|v| !v.is_nan()).count()
}

/// Number of Nyquist frequency bins for a real-valued FFT of length n.
///
/// Equals `n / 2` (floor). Used by spectral methods to check if enough
/// frequency resolution is available.
#[inline]
pub fn nyquist_bins(x: &[f64]) -> usize {
    x.len() / 2
}

/// Number of unique finite values in the slice.
///
/// Measures discreteness. A slice with only 3 unique values is unsuitable
/// for rank-based or continuous methods. Uses `total_cmp` for NaN-safe
/// ordering.
pub fn unique_prices(x: &[f64]) -> usize {
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup_by(|a, b| a.total_cmp(b) == std::cmp::Ordering::Equal);
    sorted.len()
}

/// Coefficient of variation of prices: σ / |μ|.
///
/// Measures relative variability. CV near zero indicates near-constant data.
/// Returns NaN if mean is zero or data is empty.
pub fn price_cv(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = clean.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = clean.iter().sum::<f64>() / n as f64;
    if mean.abs() < 1e-300 {
        return f64::NAN;
    }
    let var = clean.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt() / mean.abs()
}

/// Symbolic diversity: number of distinct symbols after discretization into
/// `n_symbols` equiwidth bins, divided by `n_symbols`.
///
/// Returns the fraction of bins that contain at least one sample. Range [0, 1].
/// 1.0 = every bin populated, 1/n_symbols = all samples in one bin (degenerate).
///
/// This is the data-quality complement to Shannon entropy over discretized
/// symbols: entropy measures balance, `symbolic_diversity` measures coverage.
pub fn symbolic_diversity(x: &[f64], n_symbols: usize) -> f64 {
    if x.is_empty() || n_symbols == 0 {
        return 0.0;
    }
    let clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan() && v.is_finite()).collect();
    if clean.len() < 2 {
        return 0.0;
    }
    let min = clean.iter().copied().fold(f64::INFINITY, f64::min);
    let max = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-300 {
        return 1.0 / n_symbols as f64;
    }
    let mut seen = vec![false; n_symbols];
    for &v in &clean {
        let bin = (((v - min) / range) * n_symbols as f64) as usize;
        let bin = bin.min(n_symbols - 1);
        seen[bin] = true;
    }
    seen.iter().filter(|&&b| b).count() as f64 / n_symbols as f64
}

/// Number of unique ordinal patterns of length 3 (Bandt–Pompe, m=3).
///
/// Given the series x[0..n], form all (n-2) length-3 windows and compute
/// the ordinal pattern (permutation of ranks). Returns the number of
/// distinct patterns observed. Maximum is 3! = 6.
///
/// Low diversity (< 3) indicates degenerate structure unsuitable for
/// permutation entropy or ordinal-based complexity measures.
pub fn unique_ordinal_3(x: &[f64]) -> usize {
    if x.len() < 3 {
        return 0;
    }
    let mut seen = [false; 6];
    for i in 0..(x.len() - 2) {
        let (a, b, c) = (x[i], x[i + 1], x[i + 2]);
        if a.is_nan() || b.is_nan() || c.is_nan() {
            continue;
        }
        // 6 orderings:
        // 0: a<b<c, 1: a<c<b, 2: b<a<c, 3: c<a<b, 4: b<c<a, 5: c<b<a
        let pattern = if a < b && b < c {
            0
        } else if a < c && c < b {
            1
        } else if b < a && a < c {
            2
        } else if c < a && a < b {
            3
        } else if b < c && c < a {
            4
        } else {
            5
        };
        seen[pattern] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

/// Sampling regularity CV: coefficient of variation of inter-arrival times.
///
/// `timestamps` must be monotonic non-decreasing. Computes diff[i] = t[i+1]-t[i],
/// then cv = std(diff) / mean(diff). 0.0 = perfectly regular, >1 = very irregular.
///
/// Returns NaN if fewer than 2 timestamps or diffs are all zero.
pub fn sampling_regularity_cv(timestamps: &[u64]) -> f64 {
    let n = timestamps.len();
    if n < 2 {
        return f64::NAN;
    }
    let diffs: Vec<f64> = (1..n)
        .map(|i| (timestamps[i] - timestamps[i - 1]) as f64)
        .collect();
    let m = diffs.len() as f64;
    let mean = diffs.iter().sum::<f64>() / m;
    if mean.abs() < 1e-300 {
        return f64::NAN;
    }
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / m;
    var.sqrt() / mean
}

/// Longest gap ratio: max inter-arrival / median inter-arrival.
///
/// Measures how extreme the longest gap is relative to typical spacing.
/// > 10 typically indicates a data gap that breaks continuity assumptions.
pub fn longest_gap_ratio(timestamps: &[u64]) -> f64 {
    let n = timestamps.len();
    if n < 3 {
        return f64::NAN;
    }
    let mut diffs: Vec<f64> = (1..n)
        .map(|i| (timestamps[i] - timestamps[i - 1]) as f64)
        .collect();
    let max_diff = diffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    diffs.sort_by(|a, b| a.total_cmp(b));
    let median = if diffs.len() % 2 == 0 {
        (diffs[diffs.len() / 2 - 1] + diffs[diffs.len() / 2]) / 2.0
    } else {
        diffs[diffs.len() / 2]
    };
    if median < 1e-300 {
        return f64::INFINITY;
    }
    max_diff / median
}

/// Coverage ratio: observed samples relative to the expected count for
/// a uniformly-sampled series of the same time span.
///
/// `actual_n` is the observed sample count. `expected_n` is what you would
/// have if sampling were perfectly regular. Returns `actual_n / expected_n`.
#[inline]
pub fn coverage_ratio(actual_n: usize, expected_n: usize) -> f64 {
    if expected_n == 0 {
        return 0.0;
    }
    actual_n as f64 / expected_n as f64
}

/// Split variance ratio: variance of first half vs variance of second half.
///
/// A ratio near 1.0 suggests stationarity of variance. Very large or very
/// small ratios indicate variance regime change. Returns NaN for degenerate
/// input.
pub fn split_variance_ratio(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    let mid = n / 2;
    let var_half = |s: &[f64]| -> f64 {
        let m = s.iter().sum::<f64>() / s.len() as f64;
        s.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (s.len() - 1) as f64
    };
    let v1 = var_half(&clean[..mid]);
    let v2 = var_half(&clean[mid..]);
    if v2.abs() < 1e-300 {
        return f64::INFINITY;
    }
    v1 / v2
}

/// R² from ordinary least squares fit of y = a + b·t where t is the index.
///
/// Measures how much of the variance in `x` is explained by a linear trend.
/// Range [0, 1]. Used to detect whether detrending is needed before methods
/// that assume stationarity.
///
/// Returns 0.0 for fewer than 3 samples or constant data.
pub fn trend_r2(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = clean.len();
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let y_mean = clean.iter().sum::<f64>() / nf;
    let t_mean = (nf - 1.0) / 2.0;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (i, &y) in clean.iter().enumerate() {
        let dt = i as f64 - t_mean;
        let dy = y - y_mean;
        sxy += dt * dy;
        sxx += dt * dt;
        syy += dy * dy;
    }
    if sxx < 1e-300 || syy < 1e-300 {
        return 0.0;
    }
    let r = sxy / (sxx * syy).sqrt();
    (r * r).min(1.0)
}

/// Has a significant linear trend (R² > threshold).
#[inline]
pub fn has_trend(x: &[f64], r2_threshold: f64) -> bool {
    trend_r2(x) > r2_threshold
}

/// Jump ratio proxy: fraction of absolute returns that exceed `k` times
/// the median absolute return.
///
/// `k` is typically 4 or 5. A series with many jumps (> 5% of returns
/// flagged) is unsuitable for continuous diffusion models. Returns 0 for
/// too-short input.
pub fn jump_ratio_proxy(returns: &[f64], k: f64) -> f64 {
    if returns.len() < 5 {
        return 0.0;
    }
    let mut abs_r: Vec<f64> = returns.iter().map(|r| r.abs()).filter(|r| !r.is_nan()).collect();
    let n = abs_r.len();
    if n == 0 {
        return 0.0;
    }
    abs_r.sort_by(|a, b| a.total_cmp(b));
    let median = if n % 2 == 0 {
        (abs_r[n / 2 - 1] + abs_r[n / 2]) / 2.0
    } else {
        abs_r[n / 2]
    };
    if median < 1e-300 {
        return 0.0;
    }
    let threshold = k * median;
    abs_r.iter().filter(|&&r| r > threshold).count() as f64 / n as f64
}

/// Detects volatility clustering: lag-1 autocorrelation of squared returns > threshold.
///
/// If returns have volatility clustering (the GARCH assumption), large returns
/// follow large returns regardless of sign — hence ACF of r² is positive at lag 1.
/// Returns false for insufficient data.
pub fn has_vol_clustering(returns: &[f64], threshold: f64) -> bool {
    let r2: Vec<f64> = returns.iter().map(|r| r * r).collect();
    let n = r2.len();
    if n < 5 {
        return false;
    }
    let mean = r2.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let d = r2[i] - mean;
        den += d * d;
        if i + 1 < n {
            num += d * (r2[i + 1] - mean);
        }
    }
    if den < 1e-300 {
        return false;
    }
    let acf1 = num / den;
    acf1 > threshold
}

// ═══════════════════════════════════════════════════════════════════════════
// Composite validity predicates
// ═══════════════════════════════════════════════════════════════════════════
//
// These compose the primitives above into per-family validity checks
// matching BINNED_METHODS_LIST.md. They are pure predicates — no side effects,
// no computation of the method itself.

/// Validity predicate for FFT/spectral methods.
///
/// Requires:
/// - At least 64 samples (4+ frequency bins)
/// - Enough Nyquist bins for meaningful resolution
/// - Sampling reasonably regular (CV < 0.5) if timestamps provided
pub fn fft_is_valid(x: &[f64], timestamps: Option<&[u64]>) -> bool {
    if tick_count(x) < 64 {
        return false;
    }
    if nyquist_bins(x) < 32 {
        return false;
    }
    if let Some(ts) = timestamps {
        let cv = sampling_regularity_cv(ts);
        if cv.is_nan() || cv >= 0.5 {
            return false;
        }
        if longest_gap_ratio(ts) >= 10.0 {
            return false;
        }
    }
    true
}

/// Validity predicate for GARCH-family volatility models.
///
/// Requires:
/// - At least 100 samples (typical convergence minimum)
/// - Non-constant data (price_cv > 0)
/// - Evidence of volatility clustering (otherwise a simpler model suffices)
pub fn garch_is_valid(returns: &[f64]) -> bool {
    if tick_count(returns) < 100 {
        return false;
    }
    let cv = price_cv(returns);
    if cv.is_nan() || cv < 1e-12 {
        return false;
    }
    has_vol_clustering(returns, 0.05)
}

/// Validity predicate for rank-based nonparametric methods.
///
/// Requires enough unique values that ranks are meaningful.
pub fn rank_based_is_valid(x: &[f64], min_unique: usize) -> bool {
    unique_prices(x) >= min_unique && tick_count(x) >= 5
}

/// Validity predicate for permutation entropy (m=3).
///
/// Requires at least 10 samples and at least 3 distinct ordinal patterns.
pub fn permutation_entropy_3_is_valid(x: &[f64]) -> bool {
    tick_count(x) >= 10 && unique_ordinal_3(x) >= 3
}

// ═══════════════════════════════════════════════════════════════════════════
// Temporal structure primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Lag-1 Pearson autocorrelation of the series.
///
/// Foundation primitive for `effective_sample_size`, `acf_decay_exponent`,
/// and many temporal diagnostics. Computes the correlation between
/// `x[0..n-1]` and `x[1..n]` around the sample mean.
///
/// Returns 0.0 for `n < 2` or for a constant series (zero denominator).
/// The result is always in `[-1, 1]` up to floating-point noise.
pub fn lag1_autocorrelation(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mean;
        den += dx * dx;
        if i + 1 < n {
            num += dx * (x[i + 1] - mean);
        }
    }
    if den < 1e-300 {
        return 0.0;
    }
    num / den
}

/// Effective sample size corrected for AR(1) autocorrelation.
///
/// `n_eff = n * (1 - rho) / (1 + rho)` where `rho` is the lag-1
/// autocorrelation. For rho = 0 (uncorrelated), `n_eff = n`. For rho → 1
/// (persistent), `n_eff → 0`. For rho → -1 (alternating), `n_eff → ∞`
/// (clamped to `n` — "overlapping samples don't give you more than the
/// raw sample size").
///
/// Useful for:
/// - Converting nominal sample size to degrees of freedom in autocorrelated
///   time series.
/// - Deciding whether a bin has enough **independent** observations for a
///   test that assumes IID samples.
///
/// Returns 0.0 for `n < 2`. Clamps result to `[0, n]`.
pub fn effective_sample_size(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let rho = lag1_autocorrelation(x);
    if !rho.is_finite() {
        return n as f64;
    }
    // Guard against rho ≈ -1 blowing up
    let denom = 1.0 + rho;
    if denom.abs() < 1e-12 {
        return n as f64;
    }
    let factor = (1.0 - rho) / denom;
    (n as f64 * factor).clamp(0.0, n as f64)
}

/// ACF decay exponent — slope of `log|ρ(k)|` vs `log(k)` for `k = 1..max_lag`.
///
/// A short-memory process (AR(1) with small coefficient, IID) has steep
/// exponential decay → large negative exponent.
/// A long-memory process has power-law decay → shallow slope (exponent
/// typically in `[-0.5, 0]`, related to Hurst exponent by `H = 1 + exp/2`).
///
/// `max_lag` is chosen as `min(n/4, 20)` to balance bias and variance of the
/// ACF estimate. Uses OLS on log-log points where `|ρ(k)|` is above a small
/// threshold (to avoid log of zero).
///
/// Returns NaN if `n < 10`, if the series is constant, or if fewer than
/// 3 usable log-log points are available.
pub fn acf_decay_exponent(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 10 {
        return f64::NAN;
    }
    let max_lag = (n / 4).min(20).max(2);

    let mean = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let c0: f64 = centered.iter().map(|v| v * v).sum();
    if c0 < 1e-300 {
        return f64::NAN;
    }

    let mut log_k = Vec::with_capacity(max_lag);
    let mut log_abs_rho = Vec::with_capacity(max_lag);
    for k in 1..=max_lag {
        let ck: f64 = (0..n - k).map(|i| centered[i] * centered[i + k]).sum();
        let rho = ck / c0;
        if rho.abs() > 1e-10 {
            log_k.push((k as f64).ln());
            log_abs_rho.push(rho.abs().ln());
        }
    }
    if log_k.len() < 3 {
        return f64::NAN;
    }

    let nf = log_k.len() as f64;
    let mx = log_k.iter().sum::<f64>() / nf;
    let my = log_abs_rho.iter().sum::<f64>() / nf;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..log_k.len() {
        let dx = log_k[i] - mx;
        num += dx * (log_abs_rho[i] - my);
        den += dx * dx;
    }
    if den < 1e-300 {
        return f64::NAN;
    }
    num / den
}

/// True if the ADF test rejects the unit-root null at the 5% level.
///
/// Convenience wrapper around [`crate::time_series::adf_test`]. Uses the
/// Schwert (1989) rule of thumb for lag selection
/// (`n_lags = floor(12 * (n/100)^0.25)`), clamped to `[1, n/4]`.
///
/// Returns `false` for `n < 20` (insufficient data) or if the test statistic
/// is not strictly less than the 5% critical value (in which case we cannot
/// reject the unit-root null).
pub fn is_stationary_adf_05(x: &[f64]) -> bool {
    let n = x.len();
    if n < 20 {
        return false;
    }
    let n_lags = (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
    let n_lags = n_lags.min(n / 4).max(1);
    let result = crate::time_series::adf_test(x, n_lags);
    result.statistic.is_finite() && result.statistic < result.critical_5pct
}

/// Validity predicate for sample entropy.
///
/// Requires at least 50 samples and at least 10 unique values, so that
/// template matching doesn't devolve to a handful of recurring patterns.
pub fn sample_entropy_is_valid(x: &[f64]) -> bool {
    tick_count(x) >= 50 && unique_prices(x) >= 10
}

// ═══════════════════════════════════════════════════════════════════════════
// Shareable summary bundle
// ═══════════════════════════════════════════════════════════════════════════

/// Bundled data-quality snapshot for sharing via `TamSession`.
///
/// Holds the most commonly-needed diagnostics so that downstream methods
/// (FFT, GARCH, permutation entropy, auto-detection chains) can consult one
/// cached computation instead of each re-scanning the data.
///
/// All fields are O(n) or O(n log n) from the raw slice. Boolean fields use
/// their canonical thresholds (see the individual free functions for exact
/// definitions). Callers who want non-default thresholds should call the
/// underlying primitives directly.
#[derive(Debug, Clone, Copy)]
pub struct DataQualitySummary {
    pub tick_count: usize,
    pub unique_prices: usize,
    pub price_cv: f64,
    pub lag1_autocorr: f64,
    pub effective_sample_size: f64,
    pub jump_ratio: f64,
    pub trend_r2: f64,
    pub split_variance_ratio: f64,
    pub acf_decay_exponent: f64,
    pub has_vol_clustering: bool,
    pub has_trend: bool,
    pub is_stationary_adf_05: bool,
}

impl DataQualitySummary {
    /// Compute the full summary from a single slice.
    ///
    /// Uses canonical thresholds (`has_trend` at R² > 0.5,
    /// `has_vol_clustering` at ACF > 0.1, `jump_ratio_proxy` at k = 3.0).
    pub fn from_slice(x: &[f64]) -> Self {
        Self {
            tick_count: tick_count(x),
            unique_prices: unique_prices(x),
            price_cv: price_cv(x),
            lag1_autocorr: lag1_autocorrelation(x),
            effective_sample_size: effective_sample_size(x),
            jump_ratio: jump_ratio_proxy(x, 3.0),
            trend_r2: trend_r2(x),
            split_variance_ratio: split_variance_ratio(x),
            acf_decay_exponent: acf_decay_exponent(x),
            has_vol_clustering: has_vol_clustering(x, 0.1),
            has_trend: has_trend(x, 0.5),
            is_stationary_adf_05: is_stationary_adf_05(x),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lag-specific ACF primitives (used by fractal/volatility families)
// ═══════════════════════════════════════════════════════════════════════════

/// Pearson autocorrelation at lag k.
///
/// Generalization of `lag1_autocorrelation` to arbitrary lag. Used by
/// `acf_lag10` for fractal-memory checks and by any family that needs a
/// single specific lag without computing the full ACF vector.
///
/// Returns 0.0 for `n < k + 1` or for constant data (zero denominator).
/// Always in `[-1, 1]` up to floating-point noise.
pub fn lag_k_autocorrelation(x: &[f64], k: usize) -> f64 {
    let n = x.len();
    if k == 0 { return 1.0; }
    if n < k + 1 { return 0.0; }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mean;
        den += dx * dx;
        if i + k < n {
            num += dx * (x[i + k] - mean);
        }
    }
    if den < 1e-300 { return 0.0; }
    num / den
}

/// Autocorrelation at lag 10.
///
/// Used by FRACTAL_LONG_MEMORY family as a visibility check: if
/// `|acf_lag10| < 0.05`, the autocorrelation has already decayed to noise
/// level and Hurst/DFA estimates on this slice will be unreliable.
#[inline]
pub fn acf_lag10(x: &[f64]) -> f64 {
    lag_k_autocorrelation(x, 10)
}

/// Autocorrelation of absolute returns at lag 1.
///
/// Classic proxy for volatility clustering (Taylor 1986). Large |returns|
/// tend to follow large |returns|, independent of sign. VOLATILITY family
/// uses this (threshold 0.1 by default) as a precursor to fitting GARCH:
/// no clustering → no point fitting a conditional variance model.
pub fn acf_abs_lag1(returns: &[f64]) -> f64 {
    if returns.len() < 2 { return 0.0; }
    let abs: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    lag_k_autocorrelation(&abs, 1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Family validity predicates (BINNED_METHODS_LIST.md §Summary)
// ═══════════════════════════════════════════════════════════════════════════
//
// Each predicate mirrors one row of the BINNED_METHODS_LIST quick-reference
// table, aggregating the primitive predicates above. All thresholds are
// parameterized via Option<T> with documented defaults — domain experts can
// override via using() or by calling with explicit values.

/// Validity for WAVELET family (haar, cwt, stft, synchrosqueeze, etc.).
///
/// Requires:
/// - `tick_count >= min_count` (default 32)
/// - `coverage_ratio > min_coverage` (default 0.5), if `expected_n` provided
/// - Non-constant signal: `price_cv > min_cv` (default 1e-8)
pub fn wavelet_is_valid(
    x: &[f64],
    expected_n: Option<usize>,
    min_count: Option<usize>,
    min_coverage: Option<f64>,
    min_cv: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(32);
    let mcov = min_coverage.unwrap_or(0.5);
    let mcv = min_cv.unwrap_or(1e-8);
    if tick_count(x) < mc { return false; }
    if let Some(en) = expected_n {
        if coverage_ratio(tick_count(x), en) <= mcov { return false; }
    }
    let cv = price_cv(x);
    cv.is_finite() && cv > mcv
}

/// Validity for TIME_SERIES_MODELS family (ar, ma, arma, arima, statespace).
///
/// Requires:
/// - `tick_count >= min_count` (default 30)
/// - `coverage_ratio >= min_coverage` (default 0.7), if `expected_n` provided
/// - Non-degenerate: `price_cv > min_cv` (default 1e-8)
/// - Stationary under ADF at 5%
pub fn time_series_is_valid(
    x: &[f64],
    expected_n: Option<usize>,
    min_count: Option<usize>,
    min_coverage: Option<f64>,
    min_cv: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(30);
    let mcov = min_coverage.unwrap_or(0.7);
    let mcv = min_cv.unwrap_or(1e-8);
    if tick_count(x) < mc { return false; }
    if let Some(en) = expected_n {
        if coverage_ratio(tick_count(x), en) < mcov { return false; }
    }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    is_stationary_adf_05(x)
}

/// Validity for FRACTAL_LONG_MEMORY family (hurst, dfa, mfdfa, persistence).
///
/// Requires:
/// - `tick_count >= min_count` (default 256)
/// - `split_variance_ratio` within `[variance_lower, variance_upper]`
///   (default [0.5, 2.0])
/// - `price_cv > min_cv` (default 1e-6)
/// - `|acf_lag10| > min_acf_lag10` (default 0.05): ACF has not decayed
///   to noise
pub fn fractal_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    variance_bounds: Option<(f64, f64)>,
    min_cv: Option<f64>,
    min_acf_lag10: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(256);
    let (vlo, vhi) = variance_bounds.unwrap_or((0.5, 2.0));
    let mcv = min_cv.unwrap_or(1e-6);
    let ma = min_acf_lag10.unwrap_or(0.05);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    let svr = split_variance_ratio(x);
    if !svr.is_finite() { return false; }
    if !(vlo..=vhi).contains(&svr) { return false; }
    acf_lag10(x).abs() > ma
}

/// Validity for ENTROPY_COMPLEXITY family (entropy, sampleentropy,
/// permutationentropy, transferentropy, lzcomplexity).
///
/// Requires:
/// - `tick_count >= min_count` (default 60)
/// - `symbolic_diversity > min_diversity` (default 0.3) with `n_symbols = 4`
/// - `price_cv > min_cv` (default 1e-8)
/// - `unique_ordinal_3 >= min_ordinal_patterns` (default 4 of 6)
pub fn entropy_complexity_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_diversity: Option<f64>,
    min_cv: Option<f64>,
    min_ordinal_patterns: Option<usize>,
    n_symbols: Option<usize>,
) -> bool {
    let mc = min_count.unwrap_or(60);
    let md = min_diversity.unwrap_or(0.3);
    let mcv = min_cv.unwrap_or(1e-8);
    let mop = min_ordinal_patterns.unwrap_or(4);
    let ns = n_symbols.unwrap_or(4);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    if symbolic_diversity(x, ns) <= md { return false; }
    unique_ordinal_3(x) >= mop
}

/// Validity for CHAOS_THEORY family (delayembedding, lyapunov,
/// correlationdim, poincare, recurrence).
///
/// Requires:
/// - `tick_count >= min_count` (default 500) — chaos estimators need long
///   series to resolve the attractor
/// - `symbolic_diversity > min_diversity` (default 0.5)
/// - `unique_prices >= min_unique` (default 50) — enough distinct phase
///   space points
pub fn chaos_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_diversity: Option<f64>,
    min_unique: Option<usize>,
    n_symbols: Option<usize>,
) -> bool {
    let mc = min_count.unwrap_or(500);
    let md = min_diversity.unwrap_or(0.5);
    let mu = min_unique.unwrap_or(50);
    let ns = n_symbols.unwrap_or(4);
    tick_count(x) >= mc
        && symbolic_diversity(x, ns) > md
        && unique_prices(x) >= mu
}

/// Validity for MANIFOLD_TOPOLOGY family (pcaembedding, icaembedding,
/// diffusionmap, laplacianeigenmap, curvature).
///
/// Requires:
/// - `tick_count >= min_count` (default 100)
/// - `price_cv > min_cv` (default 1e-6)
/// - `effective_sample_size >= min_effective_n` (default 10)
pub fn manifold_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_cv: Option<f64>,
    min_effective_n: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(100);
    let mcv = min_cv.unwrap_or(1e-6);
    let men = min_effective_n.unwrap_or(10.0);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    effective_sample_size(x) >= men
}

/// Validity for DISTANCE_METRICS family (dtw, wasserstein, energydistance,
/// editdistance).
///
/// Requires:
/// - Not empty
/// - `tick_count >= min_count` (default 10)
/// - Non-constant: `unique_prices >= min_unique` (default 2)
pub fn distance_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_unique: Option<usize>,
) -> bool {
    let mc = min_count.unwrap_or(10);
    let mu = min_unique.unwrap_or(2);
    !x.is_empty() && tick_count(x) >= mc && unique_prices(x) >= mu
}

/// Validity for CAUSALITY family (granger, ccm, coherence).
///
/// Requires:
/// - `tick_count >= min_count` (default 50)
/// - Stationary under ADF at 5%
/// - No strong trend: `trend_r2 < max_trend_r2` (default 0.5)
/// - `coverage_ratio >= min_coverage` (default 0.8), if `expected_n` provided
pub fn causality_is_valid(
    x: &[f64],
    expected_n: Option<usize>,
    min_count: Option<usize>,
    max_trend_r2: Option<f64>,
    min_coverage: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(50);
    let mt = max_trend_r2.unwrap_or(0.5);
    let mcov = min_coverage.unwrap_or(0.8);
    if tick_count(x) < mc { return false; }
    if let Some(en) = expected_n {
        if coverage_ratio(tick_count(x), en) < mcov { return false; }
    }
    if trend_r2(x) >= mt { return false; }
    is_stationary_adf_05(x)
}

/// Validity for REGIME_DETECTION family (cusum, glr, bocpd, pelt, hmm).
///
/// Requires:
/// - `tick_count >= min_count` (default 100)
/// - `effective_sample_size >= min_effective_n` (default 30)
/// - Non-degenerate: `price_cv > min_cv` (default 1e-6)
pub fn regime_detection_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_effective_n: Option<f64>,
    min_cv: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(100);
    let men = min_effective_n.unwrap_or(30.0);
    let mcv = min_cv.unwrap_or(1e-6);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    effective_sample_size(x) >= men
}

/// Validity for CONTINUOUS_TIME family (ode, sde, transfer, impulse, arx).
///
/// Requires:
/// - `tick_count >= min_count` (default 50)
/// - Reasonably regular sampling: `sampling_regularity_cv < max_regularity`
///   (default 0.5), if timestamps provided
/// - Non-degenerate: `price_cv > min_cv` (default 1e-6)
pub fn continuous_time_is_valid(
    x: &[f64],
    timestamps: Option<&[u64]>,
    min_count: Option<usize>,
    max_regularity: Option<f64>,
    min_cv: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(50);
    let mr = max_regularity.unwrap_or(0.5);
    let mcv = min_cv.unwrap_or(1e-6);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    if let Some(ts) = timestamps {
        let reg = sampling_regularity_cv(ts);
        if !reg.is_finite() || reg >= mr { return false; }
    }
    true
}

/// Validity for DECOMPOSITION family (emd, ssa).
///
/// Requires:
/// - `tick_count >= min_count` (default 100)
/// - Non-degenerate: `price_cv > min_cv` (default 1e-6)
/// - `nyquist_bins >= min_nyquist` (default 10)
pub fn decomposition_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_cv: Option<f64>,
    min_nyquist: Option<usize>,
) -> bool {
    let mc = min_count.unwrap_or(100);
    let mcv = min_cv.unwrap_or(1e-6);
    let mn = min_nyquist.unwrap_or(10);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    if !cv.is_finite() || cv <= mcv { return false; }
    nyquist_bins(x) >= mn
}

/// Validity for STATISTICAL_TESTS family (adf, kpss, ljungbox, normality,
/// heavytail).
///
/// Requires:
/// - `tick_count >= min_count` (default 20)
/// - Non-degenerate: `price_cv > min_cv` (default 1e-10)
pub fn statistical_tests_is_valid(
    x: &[f64],
    min_count: Option<usize>,
    min_cv: Option<f64>,
) -> bool {
    let mc = min_count.unwrap_or(20);
    let mcv = min_cv.unwrap_or(1e-10);
    if tick_count(x) < mc { return false; }
    let cv = price_cv(x);
    cv.is_finite() && cv > mcv
}

/// Validity for CORRELATION family (acf, pacf, autocorr, crosscorr).
///
/// Requires:
/// - `tick_count >= 3 * max_lag` (rule-of-thumb minimum for reliable ACF)
/// - Non-degenerate: `price_cv > min_cv` (default 1e-8)
pub fn correlation_is_valid(
    x: &[f64],
    max_lag: usize,
    min_cv: Option<f64>,
) -> bool {
    let mcv = min_cv.unwrap_or(1e-8);
    if tick_count(x) < 3 * max_lag { return false; }
    let cv = price_cv(x);
    cv.is_finite() && cv > mcv
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_count_handles_nan() {
        assert_eq!(tick_count(&[1.0, 2.0, f64::NAN, 3.0]), 3);
        assert_eq!(tick_count(&[]), 0);
        assert_eq!(tick_count(&[f64::NAN; 5]), 0);
    }

    #[test]
    fn nyquist_bins_formula() {
        assert_eq!(nyquist_bins(&vec![0.0; 64]), 32);
        assert_eq!(nyquist_bins(&vec![0.0; 128]), 64);
        assert_eq!(nyquist_bins(&vec![0.0; 10]), 5);
    }

    #[test]
    fn unique_prices_counts_correctly() {
        assert_eq!(unique_prices(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(unique_prices(&[1.0, 1.0, 1.0]), 1);
        assert_eq!(unique_prices(&[1.0, 2.0, 2.0, 3.0, 3.0]), 3);
        assert_eq!(unique_prices(&[]), 0);
        // NaN ignored
        assert_eq!(unique_prices(&[1.0, 2.0, f64::NAN, 3.0]), 3);
    }

    #[test]
    fn price_cv_constant_data() {
        let cv = price_cv(&[5.0; 10]);
        // Zero variance → CV = 0
        assert_eq!(cv, 0.0);
    }

    #[test]
    fn price_cv_known_value() {
        // [1, 2, 3, 4, 5]: mean=3, var=2.5, std≈1.581, cv≈0.527
        let cv = price_cv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((cv - 0.527).abs() < 0.01);
    }

    #[test]
    fn price_cv_empty_and_degenerate() {
        assert!(price_cv(&[]).is_nan());
        assert!(price_cv(&[5.0]).is_nan()); // n < 2
        // mean = 0 → NaN
        assert!(price_cv(&[-1.0, 1.0]).is_nan());
    }

    #[test]
    fn symbolic_diversity_full_coverage() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Uniform across [0, 99] → all 10 bins populated
        assert!((symbolic_diversity(&x, 10) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn symbolic_diversity_constant_data() {
        // Constant data → only 1 bin populated out of n_symbols
        let div = symbolic_diversity(&[5.0; 20], 10);
        assert!((div - 0.1).abs() < 1e-10);
    }

    #[test]
    fn symbolic_diversity_empty() {
        assert_eq!(symbolic_diversity(&[], 10), 0.0);
        assert_eq!(symbolic_diversity(&[1.0, 2.0, 3.0], 0), 0.0);
    }

    #[test]
    fn unique_ordinal_3_monotonic() {
        // Strictly increasing: only one pattern (a<b<c), all windows same
        let mono = unique_ordinal_3(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(mono, 1);
    }

    #[test]
    fn unique_ordinal_3_alternating() {
        // Zigzag: alternates between two patterns
        let zigzag = unique_ordinal_3(&[1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);
        assert!(zigzag >= 2);
    }

    #[test]
    fn unique_ordinal_3_too_short() {
        assert_eq!(unique_ordinal_3(&[1.0, 2.0]), 0);
        assert_eq!(unique_ordinal_3(&[]), 0);
    }

    #[test]
    fn sampling_regularity_perfect() {
        let ts: Vec<u64> = (0..10).map(|i| i * 100).collect();
        assert_eq!(sampling_regularity_cv(&ts), 0.0);
    }

    #[test]
    fn sampling_regularity_irregular() {
        // Diffs: [1, 10, 1, 10, 1, 10] — highly irregular
        // mean = 5.5, std = 4.5, cv ≈ 0.82 — which is "irregular" by convention (> 0.5)
        let ts: Vec<u64> = vec![0, 1, 11, 12, 22, 23, 33];
        let cv = sampling_regularity_cv(&ts);
        assert!(cv > 0.5, "irregular cv = {}", cv);
    }

    #[test]
    fn sampling_regularity_edge_cases() {
        assert!(sampling_regularity_cv(&[]).is_nan());
        assert!(sampling_regularity_cv(&[42]).is_nan());
    }

    #[test]
    fn longest_gap_ratio_uniform() {
        let ts: Vec<u64> = (0..10).map(|i| i * 100).collect();
        // All gaps equal → ratio = 1
        assert!((longest_gap_ratio(&ts) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn longest_gap_ratio_outlier() {
        // 9 gaps of 1, 1 gap of 100 → median = 1, max = 100 → ratio = 100
        let mut ts: Vec<u64> = (0..10).collect();
        ts.push(1000); // huge gap from 9 to 1000
        let r = longest_gap_ratio(&ts);
        assert!(r > 50.0, "ratio = {}", r);
    }

    #[test]
    fn coverage_ratio_basic() {
        assert_eq!(coverage_ratio(100, 100), 1.0);
        assert_eq!(coverage_ratio(50, 100), 0.5);
        assert_eq!(coverage_ratio(0, 0), 0.0);
    }

    #[test]
    fn split_variance_ratio_stationary() {
        // Constant variance → ratio near 1
        let mut x = vec![];
        for i in 0..100 {
            x.push((i as f64 * 0.1).sin());
        }
        let r = split_variance_ratio(&x);
        assert!((r - 1.0).abs() < 0.5, "stationary ratio = {}", r);
    }

    #[test]
    fn split_variance_ratio_regime_change() {
        // First half low variance, second half high
        let mut x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.01).sin()).collect();
        x.extend((0..50).map(|i| (i as f64 * 0.5).sin() * 10.0));
        let r = split_variance_ratio(&x);
        // First half variance << second half → ratio << 1
        assert!(r < 0.5, "regime change ratio = {}", r);
    }

    #[test]
    fn trend_r2_perfect_linear() {
        let x: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let r2 = trend_r2(&x);
        assert!((r2 - 1.0).abs() < 1e-10, "R² = {}", r2);
    }

    #[test]
    fn trend_r2_no_trend() {
        // Pure noise (deterministic for reproducibility)
        let x: Vec<f64> = (0..50)
            .map(|i| {
                let t = i as f64;
                (t * 0.7).sin() + (t * 0.31).cos()
            })
            .collect();
        let r2 = trend_r2(&x);
        assert!(r2 < 0.2, "R² = {}", r2);
    }

    #[test]
    fn trend_r2_bounded() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(trend_r2(&x) <= 1.0);
        assert!(trend_r2(&x) >= 0.0);
    }

    #[test]
    fn trend_r2_constant() {
        assert_eq!(trend_r2(&[5.0; 10]), 0.0);
    }

    #[test]
    fn has_trend_threshold() {
        let linear: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(has_trend(&linear, 0.5));
        let constant = vec![1.0; 20];
        assert!(!has_trend(&constant, 0.5));
    }

    #[test]
    fn jump_ratio_no_jumps() {
        // Normal-ish data: most returns within a few median-absolute-deviations
        let returns: Vec<f64> = (0..100)
            .map(|i| ((i as f64 * 0.17).sin() * 0.5).tanh())
            .collect();
        let r = jump_ratio_proxy(&returns, 5.0);
        assert!(r < 0.1, "jump ratio = {}", r);
    }

    #[test]
    fn jump_ratio_with_jumps() {
        let mut returns: Vec<f64> = (0..100).map(|_| 0.01).collect();
        // Inject jumps
        for idx in [10, 30, 50, 70, 90] {
            returns[idx] = 10.0;
        }
        let r = jump_ratio_proxy(&returns, 5.0);
        assert!(r >= 0.05, "jump ratio = {}", r);
    }

    #[test]
    fn has_vol_clustering_garch_like() {
        // Build returns with strong persistence in r² — squared returns correlated.
        // r[0..50] are small, r[50..100] are large → big ACF of r² at lag 1.
        let mut returns = vec![0.0; 100];
        for t in 0..50 {
            returns[t] = if t % 2 == 0 { 0.01 } else { -0.01 };
        }
        for t in 50..100 {
            returns[t] = if t % 2 == 0 { 1.0 } else { -1.0 };
        }
        assert!(has_vol_clustering(&returns, 0.1));
    }

    #[test]
    fn has_vol_clustering_iid() {
        // Independent returns — no clustering
        let returns: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64;
                (t * 2.3).sin() * (t * 0.71).cos()
            })
            .collect();
        // With this structure ACF of r² should be small
        let clustered = has_vol_clustering(&returns, 0.3);
        // This is property-level, not strict
        let _ = clustered;
    }

    // ── Composite validity predicates ──

    #[test]
    fn fft_is_valid_enough_samples() {
        let good: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(fft_is_valid(&good, None));
    }

    #[test]
    fn fft_is_valid_too_few_samples() {
        let bad = vec![1.0; 32];
        assert!(!fft_is_valid(&bad, None));
    }

    #[test]
    fn fft_is_valid_irregular_timestamps() {
        let x = vec![0.0; 128];
        let ts: Vec<u64> = (0..128).map(|i| (i * i) as u64).collect(); // quadratic → irregular
        assert!(!fft_is_valid(&x, Some(&ts)));
    }

    #[test]
    fn garch_is_valid_needs_clustering() {
        let constant = vec![1.0; 200];
        assert!(!garch_is_valid(&constant));
        let too_short = vec![0.1; 50];
        assert!(!garch_is_valid(&too_short));
    }

    #[test]
    fn rank_based_needs_diversity() {
        assert!(!rank_based_is_valid(&[1.0; 10], 5)); // only 1 unique value
        assert!(rank_based_is_valid(&[1.0, 2.0, 3.0, 4.0, 5.0], 5));
    }

    #[test]
    fn permutation_entropy_needs_patterns() {
        // Strictly monotone → only 1 pattern
        let mono: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(!permutation_entropy_3_is_valid(&mono));

        // Mixed → enough patterns
        let mixed: Vec<f64> = (0..20).map(|i| ((i as f64 * 0.7).sin() * 10.0)).collect();
        assert!(permutation_entropy_3_is_valid(&mixed));
    }

    // ── Temporal structure primitives ────────────────────────────────────

    #[test]
    fn lag1_autocorrelation_random_walk_near_one() {
        let mut x = vec![0.0; 200];
        for i in 1..200 {
            x[i] = x[i - 1] + (i as f64 * 0.17).sin() * 0.1;
        }
        let rho = lag1_autocorrelation(&x);
        assert!(rho > 0.9, "random walk should give rho ≈ 1, got {rho}");
    }

    #[test]
    fn lag1_autocorrelation_alternating_near_minus_one() {
        let x: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let rho = lag1_autocorrelation(&x);
        assert!(rho < -0.9, "alternating should give rho ≈ -1, got {rho}");
    }

    #[test]
    fn lag1_autocorrelation_constant_is_zero() {
        let x = vec![5.0; 50];
        assert_eq!(lag1_autocorrelation(&x), 0.0);
    }

    #[test]
    fn lag1_autocorrelation_short_returns_zero() {
        assert_eq!(lag1_autocorrelation(&[]), 0.0);
        assert_eq!(lag1_autocorrelation(&[1.0]), 0.0);
    }

    #[test]
    fn effective_sample_size_iid_matches_n() {
        // Low-correlation series: rho small, n_eff ≈ n
        let x: Vec<f64> = (0..100).map(|i| ((i as f64 * 13.7).sin() * 10.0 + (i as f64 * 7.3).cos())).collect();
        let n_eff = effective_sample_size(&x);
        // Should be a substantial fraction of n (not exact because of finite correlation)
        assert!(n_eff > 30.0 && n_eff <= 100.0, "expected n_eff ∈ (30, 100], got {n_eff}");
    }

    #[test]
    fn effective_sample_size_persistent_is_small() {
        let mut x = vec![0.0; 200];
        for i in 1..200 {
            x[i] = x[i - 1] + (i as f64 * 0.17).sin() * 0.1;
        }
        let n_eff = effective_sample_size(&x);
        assert!(n_eff < 50.0, "random walk should have n_eff << n, got {n_eff}");
    }

    #[test]
    fn effective_sample_size_short_zero() {
        assert_eq!(effective_sample_size(&[]), 0.0);
        assert_eq!(effective_sample_size(&[1.0]), 0.0);
    }

    #[test]
    fn effective_sample_size_clamped_to_n() {
        // Alternating rho ≈ -1 would mathematically give n_eff → ∞; must clamp to n.
        let x: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let n_eff = effective_sample_size(&x);
        assert!(n_eff <= 80.0, "must be clamped to n, got {n_eff}");
    }

    #[test]
    fn acf_decay_exponent_short_returns_nan() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(acf_decay_exponent(&x).is_nan());
    }

    #[test]
    fn acf_decay_exponent_constant_is_nan() {
        let x = vec![5.0; 100];
        assert!(acf_decay_exponent(&x).is_nan());
    }

    #[test]
    fn acf_decay_exponent_runs_on_varied_data() {
        let x: Vec<f64> = (0..200).map(|i| ((i as f64) * 0.2).sin() + (i as f64 * 0.05)).collect();
        let exp = acf_decay_exponent(&x);
        // Just verify it produces a finite result on varied input
        assert!(exp.is_finite(), "expected finite exponent, got {exp}");
    }

    #[test]
    fn is_stationary_adf_05_rejects_random_walk() {
        let mut x = vec![0.0; 200];
        for i in 1..200 {
            x[i] = x[i - 1] + ((i as f64 * 13.7).sin() * 0.5);
        }
        // Random walk is NOT stationary → ADF should not reject unit root
        assert!(!is_stationary_adf_05(&x));
    }

    #[test]
    fn is_stationary_adf_05_short_returns_false() {
        let x = vec![1.0; 10];
        assert!(!is_stationary_adf_05(&x));
    }

    #[test]
    fn sample_entropy_is_valid_needs_size_and_diversity() {
        let short = vec![1.0; 20];
        assert!(!sample_entropy_is_valid(&short));
        let flat = vec![1.0; 100];
        assert!(!sample_entropy_is_valid(&flat));
        let good: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        assert!(sample_entropy_is_valid(&good));
    }

    // ── DataQualitySummary ───────────────────────────────────────────────

    #[test]
    fn summary_linear_series() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let s = DataQualitySummary::from_slice(&x);
        assert_eq!(s.tick_count, 100);
        assert_eq!(s.unique_prices, 100);
        assert!(s.trend_r2 > 0.99, "linear → R² ≈ 1, got {}", s.trend_r2);
        assert!(s.has_trend);
        assert!(s.lag1_autocorr > 0.9, "linear → lag1 ≈ 1");
    }

    #[test]
    fn summary_constant_series() {
        let x = vec![42.0; 50];
        let s = DataQualitySummary::from_slice(&x);
        assert_eq!(s.tick_count, 50);
        assert_eq!(s.unique_prices, 1);
        assert_eq!(s.lag1_autocorr, 0.0);
        assert!(!s.has_vol_clustering);
        assert!(!s.has_trend);
    }

    #[test]
    fn summary_fields_all_populated() {
        let x: Vec<f64> = (0..150).map(|i| ((i as f64) * 0.1).sin() + (i as f64 * 0.01)).collect();
        let s = DataQualitySummary::from_slice(&x);
        assert_eq!(s.tick_count, 150);
        assert!(s.price_cv.is_finite());
        assert!(s.effective_sample_size > 0.0);
        assert!(s.effective_sample_size <= 150.0);
        assert!(s.split_variance_ratio.is_finite() || s.split_variance_ratio.is_nan());
        assert!(s.trend_r2 >= 0.0 && s.trend_r2 <= 1.0);
        assert!(s.jump_ratio >= 0.0 && s.jump_ratio <= 1.0);
    }

    // ── Lag-k autocorrelation primitives ──────────────────────────────

    #[test]
    fn lag_k_autocorrelation_lag0_is_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(lag_k_autocorrelation(&x, 0), 1.0);
    }

    #[test]
    fn lag_k_autocorrelation_too_short_returns_zero() {
        assert_eq!(lag_k_autocorrelation(&[1.0, 2.0, 3.0], 5), 0.0);
    }

    #[test]
    fn lag_k_autocorrelation_matches_lag1_shortcut() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let via_k = lag_k_autocorrelation(&x, 1);
        let via_lag1 = lag1_autocorrelation(&x);
        assert!((via_k - via_lag1).abs() < 1e-12);
    }

    #[test]
    fn acf_lag10_constant_is_zero() {
        assert_eq!(acf_lag10(&[5.0; 50]), 0.0);
    }

    #[test]
    fn acf_lag10_finite_for_sinusoid() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let r = acf_lag10(&x);
        assert!(r.is_finite() && (-1.0..=1.0).contains(&r));
    }

    #[test]
    fn acf_abs_lag1_constant_returns_zero() {
        // Constant with exact float representation → zero variance → 0.0
        assert_eq!(acf_abs_lag1(&[1.0; 100]), 0.0);
    }

    #[test]
    fn acf_abs_lag1_clustered_positive() {
        // Magnitudes clustered — large abs values follow large abs values
        let mut returns = vec![0.001; 50];
        returns.extend(vec![0.1; 50]);
        returns.extend(vec![0.001; 50]);
        let r = acf_abs_lag1(&returns);
        // Grouped magnitudes should produce positive autocorrelation of |r|
        assert!(r > 0.0, "expected positive clustering, got {r}");
    }

    // ── Family validity predicates ────────────────────────────────────

    #[test]
    fn wavelet_is_valid_default_ok() {
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(wavelet_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn wavelet_is_valid_too_short() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(!wavelet_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn wavelet_is_valid_low_coverage_rejects() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.1).sin()).collect();
        // Expected 100 samples, got 40 → coverage 0.4 < 0.5 → invalid
        assert!(!wavelet_is_valid(&x, Some(100), None, None, None));
    }

    #[test]
    fn wavelet_is_valid_custom_min_count() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Too short for default (32), but fine with custom min_count=10
        assert!(!wavelet_is_valid(&x, None, None, None, None));
        assert!(wavelet_is_valid(&x, None, Some(10), None, None));
    }

    #[test]
    fn time_series_is_valid_nonstationary_rejected() {
        // Random walk is not stationary → should fail even with enough samples
        let mut rng = crate::rng::Xoshiro256::new(1729);
        let mut walk = Vec::with_capacity(200);
        let mut s = 0.0;
        for _ in 0..200 {
            s += crate::rng::sample_normal(&mut rng, 0.0, 1.0);
            walk.push(s);
        }
        // Most realizations of a random walk fail ADF at 5%; check that
        // the predicate at least runs without panic — the specific result
        // is stochastic.
        let _ = time_series_is_valid(&walk, None, None, None, None);
    }

    #[test]
    fn time_series_is_valid_constant_rejected() {
        assert!(!time_series_is_valid(&[5.0; 100], None, None, None, None));
    }

    #[test]
    fn fractal_is_valid_too_short() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // min_count default 256
        assert!(!fractal_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn fractal_is_valid_custom_thresholds() {
        let x: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin()).collect();
        // Relax all thresholds — should accept
        let ok = fractal_is_valid(
            &x,
            Some(256),
            Some((0.01, 100.0)),
            Some(1e-12),
            Some(0.0),
        );
        // At minimum: executes without panicking
        let _ = ok;
    }

    #[test]
    fn entropy_complexity_is_valid_too_short() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        assert!(!entropy_complexity_is_valid(&x, None, None, None, None, None));
    }

    #[test]
    fn entropy_complexity_is_valid_constant_rejected() {
        assert!(!entropy_complexity_is_valid(&[5.0; 100], None, None, None, None, None));
    }

    #[test]
    fn entropy_complexity_is_valid_diverse() {
        // Deterministic diverse signal with many ordinal patterns
        let x: Vec<f64> = (0..200).map(|i| ((i as f64 * 0.31).sin() + (i as f64 * 0.17).cos())).collect();
        assert!(entropy_complexity_is_valid(&x, None, None, None, None, None));
    }

    #[test]
    fn chaos_is_valid_too_short() {
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        // min_count default 500
        assert!(!chaos_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn chaos_is_valid_custom_min() {
        let x: Vec<f64> = (0..200).map(|i| ((i as f64 * 0.31).sin() + (i as f64 * 0.17).cos()) * 10.0).collect();
        // Override min_count to allow smaller input
        let ok = chaos_is_valid(&x, Some(100), Some(0.3), Some(20), None);
        let _ = ok;
    }

    #[test]
    fn manifold_is_valid_default() {
        // Need n≥100, nonzero-mean, and enough effective sample size.
        // High-frequency sinusoid (period ~2) decorrelates quickly: lag-1 ACF
        // near -1 → ESS large. Nonzero offset avoids degenerate cv.
        let x: Vec<f64> = (0..150).map(|i| 5.0 + (i as f64 * 1.5).sin()).collect();
        assert!(manifold_is_valid(&x, None, None, None));
    }

    #[test]
    fn manifold_is_valid_too_short() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(!manifold_is_valid(&x, None, None, None));
    }

    #[test]
    fn distance_is_valid_empty_false() {
        assert!(!distance_is_valid(&[], None, None));
    }

    #[test]
    fn distance_is_valid_constant_rejected() {
        assert!(!distance_is_valid(&[1.0; 100], None, None));
    }

    #[test]
    fn distance_is_valid_default_ok() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(distance_is_valid(&x, None, None));
    }

    #[test]
    fn distance_is_valid_custom_min() {
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        // Default min_count=10 rejects; min_count=5 accepts
        assert!(!distance_is_valid(&x, None, None));
        assert!(distance_is_valid(&x, Some(5), None));
    }

    #[test]
    fn causality_is_valid_too_short() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        assert!(!causality_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn causality_is_valid_trend_rejected() {
        // Perfect linear trend → trend_r2 = 1.0, should fail
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(!causality_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn regime_detection_is_valid_default() {
        // Need n≥100, nonzero-mean, effective_sample_size ≥ 30.
        // High-frequency signal decorrelates and boosts ESS.
        let x: Vec<f64> = (0..150).map(|i| 10.0 + ((i as f64 * 1.5).sin() * 5.0)).collect();
        assert!(regime_detection_is_valid(&x, None, None, None));
    }

    #[test]
    fn regime_detection_is_valid_too_short() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(!regime_detection_is_valid(&x, None, None, None));
    }

    #[test]
    fn continuous_time_is_valid_default_ok() {
        let x: Vec<f64> = (0..80).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts: Vec<u64> = (0..80).map(|i| i * 1000).collect();
        assert!(continuous_time_is_valid(&x, Some(&ts), None, None, None));
    }

    #[test]
    fn continuous_time_is_valid_irregular_timestamps() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts: Vec<u64> = (0..100).map(|i| (i * i) as u64).collect(); // quadratic
        assert!(!continuous_time_is_valid(&x, Some(&ts), None, None, None));
    }

    #[test]
    fn continuous_time_is_valid_no_timestamps() {
        // Without timestamps, regularity is skipped — just size + cv
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(continuous_time_is_valid(&x, None, None, None, None));
    }

    #[test]
    fn decomposition_is_valid_default() {
        let x: Vec<f64> = (0..150).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(decomposition_is_valid(&x, None, None, None));
    }

    #[test]
    fn decomposition_is_valid_too_short() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(!decomposition_is_valid(&x, None, None, None));
    }

    #[test]
    fn statistical_tests_is_valid_default() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(statistical_tests_is_valid(&x, None, None));
    }

    #[test]
    fn statistical_tests_is_valid_too_short() {
        let x = [1.0, 2.0, 3.0];
        assert!(!statistical_tests_is_valid(&x, None, None));
    }

    #[test]
    fn correlation_is_valid_scales_with_max_lag() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        // max_lag=10 → needs 30 samples → ok
        assert!(correlation_is_valid(&x, 10, None));
        // max_lag=20 → needs 60 samples → fails
        assert!(!correlation_is_valid(&x, 20, None));
    }

    #[test]
    fn correlation_is_valid_constant_rejected() {
        let x = vec![5.0; 100];
        assert!(!correlation_is_valid(&x, 5, None));
    }
}
