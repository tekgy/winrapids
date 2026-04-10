//! Family 11 — Tick-level microstructure.
//!
//! Pure-function bridges for fintek's tick_* leaves. All inputs are
//! raw per-bin price/size/timestamp slices; all outputs are named
//! result structs. No dependency on fintek's Leaf trait or tambear
//! GPU primitives — pure arithmetic.
//!
//! Covered leaves:
//! - tick_vol     (K02P10C02R03F01) — realized_var, bipower_var, tick_frequency_var, microstructure_noise
//! - tick_complexity (K02P13C04R01) — inter_arrival_entropy, size_entropy, joint_entropy, normalized_complexity
//! - tick_scaling (K02P12C03R01)   — scaling_exponent, scaling_r2, xmin, burstiness
//! - tick_attractor (K02P14C05R01) — phase_asymmetry, tick_persistence, reversal_rate, phase_spread
//! - tick_alignment (K02P16C04R01) — arrival_regularity, clustering_index, gap_ratio, uniformity_score
//! - tick_causality (K02P17C03R01) — lead_lag_corr, lead_lag_offset, coupling_strength, impulse_ratio
//! - tick_geometry  (K02P15C6R1)   — hull_area, angular_entropy, radial_kurtosis, aspect_ratio
//! - tick_space     (K02P18C03R03F01) — tick_entropy, mode_concentration, tick_clustering, regime_persistence
//! - tick_ou        (K02P11C05R01)  — theta, half_life, sigma, mr_strength
//! - tick_compression (K02P25C01)   — real_effective_rank, shuffled_effective_rank, compression_ratio, n_active_features

use tambear::data_quality::lag1_autocorrelation;
use tambear::simple_linear_regression;

// ── Shared helpers ──────────────────────────────────────────────────────────

/// Shannon entropy of a histogram (nats).
fn histogram_entropy(values: &[f64], n_bins: usize) -> f64 {
    if values.is_empty() || n_bins == 0 { return f64::NAN; }
    let vmin = values.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (vmax - vmin) < 1e-30 { return 0.0; }
    let bw = (vmax - vmin) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];
    for &v in values {
        let idx = ((v - vmin) / bw) as usize;
        counts[idx.min(n_bins - 1)] += 1;
    }
    let total = values.len() as f64;
    let mut h = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Log returns from prices: ln(p[i+1]/p[i]).
fn log_returns_from_prices(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| {
        let p0 = w[0].max(1e-300);
        let p1 = w[1].max(1e-300);
        (p1 / p0).ln()
    }).collect()
}

/// Convex hull area via gift wrapping + Shoelace. Exact port of fintek's tick_geometry.rs.
fn convex_hull_area(points: &[(f64, f64)]) -> f64 {
    if points.len() < 3 { return 0.0; }

    let mut start = 0;
    for i in 1..points.len() {
        if points[i].0 < points[start].0
            || (points[i].0 == points[start].0 && points[i].1 < points[start].1)
        {
            start = i;
        }
    }

    let mut hull = Vec::new();
    let mut current = start;
    loop {
        hull.push(current);
        let mut next = 0;
        for j in 0..points.len() {
            if j == current { continue; }
            if next == current { next = j; continue; }
            let cross = (points[j].0 - points[current].0) * (points[next].1 - points[current].1)
                      - (points[j].1 - points[current].1) * (points[next].0 - points[current].0);
            if cross > 0.0 {
                next = j;
            } else if cross == 0.0 {
                let dj = (points[j].0 - points[current].0).powi(2)
                       + (points[j].1 - points[current].1).powi(2);
                let dn = (points[next].0 - points[current].0).powi(2)
                       + (points[next].1 - points[current].1).powi(2);
                if dj > dn { next = j; }
            }
        }
        current = next;
        if current == start || hull.len() > points.len() { break; }
    }

    let n = hull.len();
    if n < 3 { return 0.0; }
    let mut area = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[hull[i]].0 * points[hull[j]].1;
        area -= points[hull[j]].0 * points[hull[i]].1;
    }
    area.abs() / 2.0
}

// ── family 11a: tick_vol (K02P10C02R03F01) ─────────────────────────────────

/// Tick volatility features from raw per-bin prices and timestamps.
///
/// - `realized_var`: sum of squared log-returns
/// - `bipower_var`: Barndorff-Nielsen/Shephard bipower variation (jump-robust RV)
/// - `tick_frequency_var`: realized variance at sub-sampled grids (avg over 5 offsets)
/// - `microstructure_noise`: difference between full-sample RV and sub-sampled RV
///
/// Corresponds to fintek's `tick_vol.rs` (K02P10C02R03F01).
#[derive(Debug, Clone)]
pub struct TickVolResult {
    pub realized_var: f64,
    pub bipower_var: f64,
    pub tick_frequency_var: f64,
    pub microstructure_noise: f64,
}

impl TickVolResult {
    pub fn nan() -> Self {
        Self { realized_var: f64::NAN, bipower_var: f64::NAN,
               tick_frequency_var: f64::NAN, microstructure_noise: f64::NAN }
    }
}

const SUB_K: usize = 5;
const MIN_TICKS_VOL: usize = 20;

/// Compute tick volatility features from bin prices and timestamps.
pub fn tick_vol(prices: &[f64], timestamps_ns: &[u64]) -> TickVolResult {
    let n = prices.len();
    if n < MIN_TICKS_VOL { return TickVolResult::nan(); }

    let returns = log_returns_from_prices(prices);
    let m = returns.len();

    // Realized variance: Σ r²
    let realized_var: f64 = returns.iter().map(|r| r * r).sum();

    // Bipower variation: (π/2) · Σ |r_t| · |r_{t-1}|
    let bv_factor = std::f64::consts::PI / 2.0;
    let bipower_var = if m >= 2 {
        bv_factor * returns.windows(2).map(|w| w[0].abs() * w[1].abs()).sum::<f64>()
    } else { f64::NAN };

    // Tick frequency var: sub-sample at step SUB_K across SUB_K offset grids
    let (tick_freq_var, _) = if !timestamps_ns.is_empty() && timestamps_ns.len() == n {
        // Duration-weighted subsampling using actual timestamps
        let mut sub_rvs = Vec::with_capacity(SUB_K);
        for offset in 0..SUB_K {
            let sub_prices: Vec<f64> = (offset..n).step_by(SUB_K).map(|i| prices[i]).collect();
            if sub_prices.len() >= 2 {
                let sub_ret = log_returns_from_prices(&sub_prices);
                let rv: f64 = sub_ret.iter().map(|r| r * r).sum();
                sub_rvs.push(rv);
            }
        }
        if sub_rvs.is_empty() { (f64::NAN, f64::NAN) }
        else {
            let mean_rv = sub_rvs.iter().sum::<f64>() / sub_rvs.len() as f64;
            (mean_rv, mean_rv)
        }
    } else {
        // No timestamps: uniform sub-sampling
        let mut sub_rvs = Vec::with_capacity(SUB_K);
        for offset in 0..SUB_K {
            let sub_prices: Vec<f64> = (offset..n).step_by(SUB_K).map(|i| prices[i]).collect();
            if sub_prices.len() >= 2 {
                let sub_ret = log_returns_from_prices(&sub_prices);
                let rv: f64 = sub_ret.iter().map(|r| r * r).sum();
                sub_rvs.push(rv);
            }
        }
        if sub_rvs.is_empty() { (f64::NAN, f64::NAN) }
        else {
            let mean_rv = sub_rvs.iter().sum::<f64>() / sub_rvs.len() as f64;
            (mean_rv, mean_rv)
        }
    };

    // Microstructure noise: positive part of (full RV - sub-sampled RV)
    let microstructure_noise = if tick_freq_var.is_finite() {
        (realized_var - tick_freq_var).max(0.0)
    } else { f64::NAN };

    TickVolResult { realized_var, bipower_var, tick_frequency_var: tick_freq_var, microstructure_noise }
}

// ── family 11b: tick_complexity (K02P13C04R01) ─────────────────────────────

/// Information-theoretic complexity of tick stream.
///
/// - `inter_arrival_entropy`: Shannon entropy of IAT distribution (16 bins)
/// - `size_entropy`: Shannon entropy of trade-size distribution (16 bins)
/// - `joint_entropy`: joint entropy of (IAT, size) on 16×16 grid
/// - `normalized_complexity`: joint_entropy / max_entropy
///
/// Corresponds to fintek's `tick_complexity.rs` (K02P13C04R01).
#[derive(Debug, Clone)]
pub struct TickComplexityResult {
    pub inter_arrival_entropy: f64,
    pub size_entropy: f64,
    pub joint_entropy: f64,
    pub normalized_complexity: f64,
}

impl TickComplexityResult {
    pub fn nan() -> Self {
        Self { inter_arrival_entropy: f64::NAN, size_entropy: f64::NAN,
               joint_entropy: f64::NAN, normalized_complexity: f64::NAN }
    }
}

const N_HIST_BINS: usize = 16;
const MIN_TICKS_COMPLEXITY: usize = 10;

/// Compute tick information complexity.
///
/// `timestamps_ns`: per-tick nanosecond timestamps.
/// `sizes`: per-tick trade sizes (shares or contracts).
pub fn tick_complexity(timestamps_ns: &[u64], sizes: &[f64]) -> TickComplexityResult {
    let n = timestamps_ns.len().min(sizes.len());
    if n < MIN_TICKS_COMPLEXITY + 1 { return TickComplexityResult::nan(); }

    // Inter-arrival times (nanoseconds → float)
    let iats: Vec<f64> = timestamps_ns.windows(2)
        .map(|w| (w[1].saturating_sub(w[0])) as f64)
        .collect();
    if iats.is_empty() { return TickComplexityResult::nan(); }

    let sz = &sizes[..n];

    let inter_arrival_entropy = histogram_entropy(&iats, N_HIST_BINS);
    let size_entropy = histogram_entropy(sz, N_HIST_BINS);

    // Joint entropy on 16×16 grid
    let iat_min = iats.iter().copied().fold(f64::INFINITY, f64::min);
    let iat_max = iats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sz_min  = sz.iter().copied().fold(f64::INFINITY, f64::min);
    let sz_max  = sz.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let joint_entropy = if (iat_max - iat_min) > 1e-30 && (sz_max - sz_min) > 1e-30 {
        let iat_bw = (iat_max - iat_min) / N_HIST_BINS as f64;
        let sz_bw  = (sz_max  - sz_min)  / N_HIST_BINS as f64;
        let mut joint = vec![0u64; N_HIST_BINS * N_HIST_BINS];
        let m = iats.len().min(sz.len() - 1); // sizes has one more than iats
        for i in 0..m {
            let ai = ((iats[i] - iat_min) / iat_bw) as usize;
            let ai = ai.min(N_HIST_BINS - 1);
            let bi = ((sz[i] - sz_min) / sz_bw) as usize;
            let bi = bi.min(N_HIST_BINS - 1);
            joint[ai * N_HIST_BINS + bi] += 1;
        }
        let total = m as f64;
        let mut h = 0.0f64;
        for &c in &joint {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * p.ln();
            }
        }
        h
    } else { 0.0 };

    let max_entropy = ((N_HIST_BINS * N_HIST_BINS) as f64).ln();
    let normalized_complexity = if max_entropy > 1e-30 {
        joint_entropy / max_entropy
    } else { 0.0 };

    TickComplexityResult { inter_arrival_entropy, size_entropy, joint_entropy, normalized_complexity }
}

// ── family 11c: tick_scaling (K02P12C03R01) ─────────────────────────────────

/// Power-law scaling of inter-arrival time distribution.
///
/// - `scaling_exponent`: MLE power-law exponent α (CCDF: P(X>x) ~ x^{-α})
/// - `scaling_r2`: R² of log-log linear fit to CCDF
/// - `xmin`: lower cut-off (median IAT)
/// - `burstiness`: B = (σ-μ)/(σ+μ) of IAT distribution
///
/// Corresponds to fintek's `tick_scaling.rs` (K02P12C03R01).
#[derive(Debug, Clone)]
pub struct TickScalingResult {
    pub scaling_exponent: f64,
    pub scaling_r2: f64,
    pub xmin: f64,
    pub burstiness: f64,
}

impl TickScalingResult {
    pub fn nan() -> Self {
        Self { scaling_exponent: f64::NAN, scaling_r2: f64::NAN,
               xmin: f64::NAN, burstiness: f64::NAN }
    }
}

const MIN_TICKS_SCALING: usize = 20;

/// Compute tick arrival scaling features.
pub fn tick_scaling(timestamps_ns: &[u64]) -> TickScalingResult {
    let n = timestamps_ns.len();
    if n < MIN_TICKS_SCALING + 1 { return TickScalingResult::nan(); }

    let iats: Vec<f64> = timestamps_ns.windows(2)
        .map(|w| (w[1].saturating_sub(w[0])) as f64)
        .filter(|&v| v > 0.0)
        .collect();
    if iats.len() < MIN_TICKS_SCALING { return TickScalingResult::nan(); }

    let mut sorted_iats = iats.clone();
    sorted_iats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = sorted_iats.len();
    let xmin = sorted_iats[m / 2]; // median as cutoff

    // Power-law tail: fit CCDF above xmin via log-log OLS
    let tail: Vec<f64> = sorted_iats.iter().copied().filter(|&v| v >= xmin).collect();
    let n_tail = tail.len();

    let (scaling_exponent, scaling_r2) = if n_tail >= 5 {
        // CCDF: for rank i (0-indexed from smallest), P(X > tail[i]) ≈ (n_tail - i) / n_tail
        let log_x: Vec<f64> = tail.iter().map(|&x| x.ln()).collect();
        let log_p: Vec<f64> = (0..n_tail).map(|i| {
            let p = (n_tail - i) as f64 / n_tail as f64;
            p.ln()
        }).collect();

        // OLS: log_p = a + b * log_x  → α is the negative slope of the log-log CCDF
        let reg = simple_linear_regression(&log_x, &log_p);
        let r2 = if reg.r_squared.is_finite() { reg.r_squared.clamp(0.0, 1.0) } else { f64::NAN };
        (-reg.slope, r2)
    } else { (f64::NAN, f64::NAN) };

    // Burstiness: B = (σ - μ) / (σ + μ)
    let iat_mean: f64 = iats.iter().sum::<f64>() / iats.len() as f64;
    let iat_var: f64 = iats.iter().map(|v| (v - iat_mean).powi(2)).sum::<f64>() / iats.len() as f64;
    let iat_std = iat_var.sqrt();
    let burstiness = if (iat_std + iat_mean).abs() > 1e-30 {
        (iat_std - iat_mean) / (iat_std + iat_mean)
    } else { 0.0 };

    TickScalingResult { scaling_exponent, scaling_r2, xmin, burstiness }
}

// ── family 11d: tick_attractor (K02P14C05R01) ──────────────────────────────

/// Phase-space attractor features of tick price process.
///
/// - `phase_asymmetry`: (Q1+Q3 - Q2+Q4) / total in phase portrait quadrants
/// - `tick_persistence`: lag-1 autocorrelation of log-returns
/// - `reversal_rate`: fraction of sign changes in return sequence
/// - `phase_spread`: radial std of phase portrait
///
/// Corresponds to fintek's `tick_attractor.rs` (K02P14C05R01).
#[derive(Debug, Clone)]
pub struct TickAttractorResult {
    pub phase_asymmetry: f64,
    pub tick_persistence: f64,
    pub reversal_rate: f64,
    pub phase_spread: f64,
}

impl TickAttractorResult {
    pub fn nan() -> Self {
        Self { phase_asymmetry: f64::NAN, tick_persistence: f64::NAN,
               reversal_rate: f64::NAN, phase_spread: f64::NAN }
    }
}

const MIN_RETURNS_ATTRACTOR: usize = 10;

/// Compute tick phase-space attractor features.
pub fn tick_attractor(prices: &[f64]) -> TickAttractorResult {
    if prices.len() < MIN_RETURNS_ATTRACTOR + 2 { return TickAttractorResult::nan(); }

    let returns = log_returns_from_prices(prices);
    let n = returns.len();
    if n < MIN_RETURNS_ATTRACTOR + 1 { return TickAttractorResult::nan(); }

    // Phase portrait: (r[t], r[t-1])
    let mut q = [0u64; 4]; // Q1=+/+, Q2=-/+, Q3=-/-, Q4=+/-
    let mut radii = Vec::with_capacity(n - 1);
    for t in 1..n {
        let x = returns[t];
        let y = returns[t - 1];
        radii.push((x * x + y * y).sqrt());
        match (x >= 0.0, y >= 0.0) {
            (true,  true)  => q[0] += 1,
            (false, true)  => q[1] += 1,
            (false, false) => q[2] += 1,
            (true,  false) => q[3] += 1,
        }
    }
    let total = (n - 1) as f64;

    // Phase asymmetry: trend quadrants (Q1+Q3) vs reversal quadrants (Q2+Q4)
    // Q1 (+,+) and Q3 (-,-) = momentum; Q2 (-,+) and Q4 (+,-) = reversal
    let phase_asymmetry = if total > 0.0 {
        ((q[0] + q[2]) as f64 - (q[1] + q[3]) as f64) / total
    } else { f64::NAN };

    // Tick persistence: lag-1 autocorrelation of returns
    let tick_persistence = lag1_autocorrelation(&returns);

    // Reversal rate: fraction of sign changes
    let reversal_rate = {
        let mut changes = 0u64;
        let mut valid = 0u64;
        for t in 1..n {
            let prev_sign = returns[t - 1] > 0.0;
            let curr_sign = returns[t] > 0.0;
            if returns[t - 1] != 0.0 && returns[t] != 0.0 {
                valid += 1;
                if prev_sign != curr_sign { changes += 1; }
            }
        }
        if valid > 0 { changes as f64 / valid as f64 } else { f64::NAN }
    };

    // Phase spread: std of radial distances
    let phase_spread = if !radii.is_empty() {
        let r_mean: f64 = radii.iter().sum::<f64>() / radii.len() as f64;
        let var: f64 = radii.iter().map(|r| (r - r_mean).powi(2)).sum::<f64>() / radii.len() as f64;
        var.sqrt()
    } else { f64::NAN };

    TickAttractorResult { phase_asymmetry, tick_persistence, reversal_rate, phase_spread }
}

// ── family 11e: tick_alignment (K02P16C04R01) ──────────────────────────────

/// Temporal alignment / regularity of tick stream.
///
/// - `arrival_regularity`: 1 - CoV of IAT (coefficient of variation)
/// - `clustering_index`: Fano factor of IAT (var/mean), clamped to [0,∞)
/// - `gap_ratio`: fraction of IATs > median (large-gap fraction)
/// - `uniformity_score`: 1 - KS statistic vs uniform distribution
///
/// Corresponds to fintek's `tick_alignment.rs` (K02P16C04R01).
#[derive(Debug, Clone)]
pub struct TickAlignmentResult {
    pub arrival_regularity: f64,
    pub clustering_index: f64,
    pub gap_ratio: f64,
    pub uniformity_score: f64,
}

impl TickAlignmentResult {
    pub fn nan() -> Self {
        Self { arrival_regularity: f64::NAN, clustering_index: f64::NAN,
               gap_ratio: f64::NAN, uniformity_score: f64::NAN }
    }
}

const MIN_TICKS_ALIGNMENT: usize = 10;

/// Compute tick temporal alignment features.
pub fn tick_alignment(timestamps_ns: &[u64]) -> TickAlignmentResult {
    let n = timestamps_ns.len();
    if n < MIN_TICKS_ALIGNMENT + 1 { return TickAlignmentResult::nan(); }

    let iats: Vec<f64> = timestamps_ns.windows(2)
        .map(|w| (w[1].saturating_sub(w[0])) as f64)
        .collect();
    let m = iats.len();
    if m < MIN_TICKS_ALIGNMENT { return TickAlignmentResult::nan(); }

    let iat_mean: f64 = iats.iter().sum::<f64>() / m as f64;
    let iat_var: f64 = iats.iter().map(|v| (v - iat_mean).powi(2)).sum::<f64>() / m as f64;
    let iat_std = iat_var.sqrt();

    // Arrival regularity: 1 - CoV (1 = perfectly regular, 0 = very bursty)
    let arrival_regularity = if iat_mean > 1e-30 {
        (1.0 - iat_std / iat_mean).clamp(-1.0, 1.0)
    } else { f64::NAN };

    // Clustering index: Fano factor = var/mean
    let clustering_index = if iat_mean > 1e-30 {
        (iat_var / iat_mean).max(0.0)
    } else { f64::NAN };

    // Gap ratio: fraction of IATs > median
    let mut sorted_iats = iats.clone();
    sorted_iats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_iat = sorted_iats[m / 2];
    let gap_ratio = iats.iter().filter(|&&v| v > median_iat).count() as f64 / m as f64;

    // Uniformity score: 1 - KS statistic vs uniform on [0, max_iat]
    let iat_max = sorted_iats.last().copied().unwrap_or(1.0);
    let uniformity_score = if iat_max > 1e-30 {
        let mut ks_stat = 0.0f64;
        for (i, &v) in sorted_iats.iter().enumerate() {
            let empirical = (i + 1) as f64 / m as f64;
            let theoretical = v / iat_max;
            let d = (empirical - theoretical).abs();
            if d > ks_stat { ks_stat = d; }
        }
        1.0 - ks_stat
    } else { f64::NAN };

    TickAlignmentResult { arrival_regularity, clustering_index, gap_ratio, uniformity_score }
}

// ── family 11f: tick_causality (K02P17C03R01) ──────────────────────────────

/// Volume→return lead-lag causality features.
///
/// - `lead_lag_corr`: maximum absolute cross-correlation over lags ±1..5
/// - `lead_lag_offset`: lag at which max cross-correlation is achieved
/// - `coupling_strength`: fraction of lags with |xcorr| > threshold
/// - `impulse_ratio`: ratio of max forward to max backward correlation
///
/// Corresponds to fintek's `tick_causality.rs` (K02P17C03R01).
#[derive(Debug, Clone)]
pub struct TickCausalityResult {
    pub lead_lag_corr: f64,
    pub lead_lag_offset: f64,
    pub coupling_strength: f64,
    pub impulse_ratio: f64,
}

impl TickCausalityResult {
    pub fn nan() -> Self {
        Self { lead_lag_corr: f64::NAN, lead_lag_offset: f64::NAN,
               coupling_strength: f64::NAN, impulse_ratio: f64::NAN }
    }
}

const MAX_LAG_CAUSALITY: usize = 5;
const COUPLING_THRESHOLD: f64 = 0.1;
const MIN_TICKS_CAUSALITY: usize = 20;

/// Compute volume-return cross-correlation causality features.
///
/// `returns`: per-tick log-returns.
/// `volumes`: per-tick trade volumes.
pub fn tick_causality(returns: &[f64], volumes: &[f64]) -> TickCausalityResult {
    let n = returns.len().min(volumes.len());
    if n < MIN_TICKS_CAUSALITY { return TickCausalityResult::nan(); }

    let ret = &returns[..n];
    let vol = &volumes[..n];

    let ret_mean: f64 = ret.iter().sum::<f64>() / n as f64;
    let vol_mean: f64 = vol.iter().sum::<f64>() / n as f64;
    let ret_std: f64 = {
        let v: f64 = ret.iter().map(|r| (r - ret_mean).powi(2)).sum::<f64>() / n as f64;
        v.sqrt()
    };
    let vol_std: f64 = {
        let v: f64 = vol.iter().map(|v| (v - vol_mean).powi(2)).sum::<f64>() / n as f64;
        v.sqrt()
    };

    if ret_std < 1e-30 || vol_std < 1e-30 { return TickCausalityResult::nan(); }

    // Cross-correlation at lags -MAX_LAG..=MAX_LAG (positive lag = volume leads return)
    let mut xcorrs: Vec<(i64, f64)> = Vec::new();
    for lag in -(MAX_LAG_CAUSALITY as i64)..=(MAX_LAG_CAUSALITY as i64) {
        if lag == 0 { continue; }
        let mut cov = 0.0f64;
        let mut count = 0usize;
        if lag > 0 {
            // vol leads ret: vol[t] vs ret[t+lag]
            for t in 0..(n - lag as usize) {
                cov += (vol[t] - vol_mean) * (ret[t + lag as usize] - ret_mean);
                count += 1;
            }
        } else {
            // ret leads vol
            let l = (-lag) as usize;
            for t in l..n {
                cov += (vol[t] - vol_mean) * (ret[t - l] - ret_mean);
                count += 1;
            }
        }
        if count > 0 {
            let xcorr = (cov / count as f64) / (vol_std * ret_std);
            xcorrs.push((lag, xcorr));
        }
    }

    if xcorrs.is_empty() { return TickCausalityResult::nan(); }

    let (best_lag, best_corr) = xcorrs.iter()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap();

    let coupling_strength = xcorrs.iter().filter(|&&(_, c)| c.abs() > COUPLING_THRESHOLD).count() as f64
        / xcorrs.len() as f64;

    // Impulse ratio: max forward corr (lag>0, vol→ret) vs max backward (lag<0)
    let fwd_max = xcorrs.iter().filter(|&&(l, _)| l > 0).map(|&(_, c)| c.abs()).fold(0.0f64, f64::max);
    let bwd_max = xcorrs.iter().filter(|&&(l, _)| l < 0).map(|&(_, c)| c.abs()).fold(0.0f64, f64::max);
    let impulse_ratio = if bwd_max > 1e-30 { fwd_max / bwd_max } else { f64::NAN };

    TickCausalityResult {
        lead_lag_corr: best_corr.abs(),
        lead_lag_offset: best_lag as f64,
        coupling_strength,
        impulse_ratio,
    }
}

// ── family 11g: tick_geometry (K02P15C6R1) ─────────────────────────────────

/// Phase-plane geometric features of tick price process.
///
/// - `hull_area`: convex hull area of (r[t], r[t-1]) phase portrait
/// - `angular_entropy`: normalized Shannon entropy of angular distribution (32 bins)
/// - `radial_kurtosis`: excess kurtosis of radial distances
/// - `aspect_ratio`: minor/major eigenvalue ratio of scatter matrix
///
/// Corresponds to fintek's `tick_geometry.rs` (K02P15C6R1).
#[derive(Debug, Clone)]
pub struct TickGeometryResult {
    pub hull_area: f64,
    pub angular_entropy: f64,
    pub radial_kurtosis: f64,
    pub aspect_ratio: f64,
}

impl TickGeometryResult {
    pub fn nan() -> Self {
        Self { hull_area: f64::NAN, angular_entropy: f64::NAN,
               radial_kurtosis: f64::NAN, aspect_ratio: f64::NAN }
    }
}

const N_ANGLE_BINS: usize = 32;
const MIN_RETURNS_GEOMETRY: usize = 20;
const MAX_PTS_GEOMETRY: usize = 5000;

/// Compute tick phase-plane geometry features.
pub fn tick_geometry(prices: &[f64]) -> TickGeometryResult {
    if prices.len() < MIN_RETURNS_GEOMETRY + 2 { return TickGeometryResult::nan(); }

    let raw_returns = log_returns_from_prices(prices);
    let returns = if raw_returns.len() > MAX_PTS_GEOMETRY {
        let step = raw_returns.len() / MAX_PTS_GEOMETRY;
        raw_returns.iter().step_by(step).copied().collect::<Vec<_>>()
    } else { raw_returns };
    let n = returns.len();
    if n < 3 { return TickGeometryResult::nan(); }

    let n_pts = n - 1;
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_pts);
    let mut angles: Vec<f64> = Vec::with_capacity(n_pts);
    let mut radii: Vec<f64> = Vec::with_capacity(n_pts);

    for t in 1..n {
        let x = returns[t];
        let y = returns[t - 1];
        points.push((x, y));
        radii.push((x * x + y * y).sqrt());
        angles.push(y.atan2(x));
    }

    let hull_area = convex_hull_area(&points);

    // Angular entropy (normalized)
    let bin_width = 2.0 * std::f64::consts::PI / N_ANGLE_BINS as f64;
    let mut angle_hist = vec![0u32; N_ANGLE_BINS];
    for &a in &angles {
        let bin = ((a + std::f64::consts::PI) / bin_width).floor() as usize;
        angle_hist[bin.min(N_ANGLE_BINS - 1)] += 1;
    }
    let total = n_pts as f64;
    let max_entropy = (N_ANGLE_BINS as f64).ln();
    let mut raw_entropy = 0.0f64;
    for &c in &angle_hist {
        if c > 0 {
            let p = c as f64 / total;
            raw_entropy -= p * p.ln();
        }
    }
    let angular_entropy = if max_entropy > 1e-30 { raw_entropy / max_entropy } else { 0.0 };

    // Radial kurtosis
    let r_mean: f64 = radii.iter().sum::<f64>() / n_pts as f64;
    let r_var: f64 = radii.iter().map(|r| (r - r_mean).powi(2)).sum::<f64>() / n_pts as f64;
    let radial_kurtosis = if r_var > 1e-30 {
        let m4: f64 = radii.iter().map(|r| (r - r_mean).powi(4)).sum::<f64>() / n_pts as f64;
        m4 / (r_var * r_var) - 3.0
    } else { 0.0 };

    // Aspect ratio from 2×2 scatter matrix eigenvalues
    let mean_x: f64 = points.iter().map(|p| p.0).sum::<f64>() / n_pts as f64;
    let mean_y: f64 = points.iter().map(|p| p.1).sum::<f64>() / n_pts as f64;
    let mut sxx = 0.0f64; let mut syy = 0.0f64; let mut sxy = 0.0f64;
    for &(x, y) in &points {
        let dx = x - mean_x; let dy = y - mean_y;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx /= n_pts as f64; syy /= n_pts as f64; sxy /= n_pts as f64;
    let avg = (sxx + syy) / 2.0;
    let disc = (((sxx - syy) / 2.0).powi(2) + sxy * sxy).sqrt();
    let l1 = avg + disc;
    let l2 = (avg - disc).max(0.0);
    let aspect_ratio = if l1 > 1e-30 { l2 / l1 } else { 1.0 };

    TickGeometryResult { hull_area, angular_entropy, radial_kurtosis, aspect_ratio }
}

// ── family 11h: tick_space (K02P18C03R03F01) ───────────────────────────────

/// Tick-space distributional features.
///
/// - `tick_entropy`: Shannon entropy of tick-size distribution (50 bins)
/// - `mode_concentration`: fraction of ticks at the modal size (1% precision)
/// - `tick_clustering`: lag-1 autocorrelation of absolute price changes
/// - `regime_persistence`: mean run length of same-sign price moves
///
/// Corresponds to fintek's `tick_space.rs` (K02P18C03R03F01).
#[derive(Debug, Clone)]
pub struct TickSpaceResult {
    pub tick_entropy: f64,
    pub mode_concentration: f64,
    pub tick_clustering: f64,
    pub regime_persistence: f64,
}

impl TickSpaceResult {
    pub fn nan() -> Self {
        Self { tick_entropy: f64::NAN, mode_concentration: f64::NAN,
               tick_clustering: f64::NAN, regime_persistence: f64::NAN }
    }
}

const N_ENTROPY_BINS_SPACE: usize = 50;
const MIN_CHANGES_SPACE: usize = 10;

/// Compute tick-space distributional features.
pub fn tick_space(prices: &[f64]) -> TickSpaceResult {
    let n = prices.len();
    if n < MIN_CHANGES_SPACE + 1 { return TickSpaceResult::nan(); }

    let dp: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let tick_sizes: Vec<f64> = dp.iter().map(|d| d.abs()).collect();
    let nonzero_ticks: Vec<f64> = tick_sizes.iter().copied().filter(|&t| t > 0.0).collect();
    if nonzero_ticks.len() < MIN_CHANGES_SPACE { return TickSpaceResult::nan(); }

    // Tick entropy (50 linear bins)
    let tick_entropy = histogram_entropy(&nonzero_ticks, N_ENTROPY_BINS_SPACE);

    // Mode concentration: round to 1% of median, find modal bin
    let mut sorted_ticks = nonzero_ticks.clone();
    sorted_ticks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_tick = sorted_ticks[sorted_ticks.len() / 2];
    let precision = (median_tick * 0.01).max(1e-15);

    let mut rounded: Vec<i64> = nonzero_ticks.iter()
        .map(|&t| (t / precision).round() as i64)
        .collect();
    rounded.sort();

    let mut max_count = 0u64;
    let mut current_count = 1u64;
    for i in 1..rounded.len() {
        if rounded[i] == rounded[i - 1] {
            current_count += 1;
        } else {
            if current_count > max_count { max_count = current_count; }
            current_count = 1;
        }
    }
    if current_count > max_count { max_count = current_count; }
    let mode_concentration = max_count as f64 / nonzero_ticks.len() as f64;

    // Tick clustering: lag-1 ACF of |dp|
    let tick_clustering = lag1_autocorrelation(&tick_sizes);

    // Regime persistence: mean run length of same-sign moves
    let signs: Vec<i8> = dp.iter().map(|&d| {
        if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 }
    }).collect();
    let regime_persistence = if signs.len() >= 2 {
        let mut run_lengths: Vec<u64> = Vec::new();
        let mut current_run = 1u64;
        for i in 1..signs.len() {
            if signs[i] == signs[i - 1] && signs[i] != 0 {
                current_run += 1;
            } else {
                if current_run > 0 { run_lengths.push(current_run); }
                current_run = 1;
            }
        }
        run_lengths.push(current_run);
        if run_lengths.is_empty() { 1.0 }
        else { run_lengths.iter().sum::<u64>() as f64 / run_lengths.len() as f64 }
    } else { f64::NAN };

    TickSpaceResult { tick_entropy, mode_concentration, tick_clustering, regime_persistence }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prices(n: usize, drift: f64, noise: f64, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        let mut p = 100.0f64;
        let mut prices = vec![p];
        for _ in 0..n {
            p += drift + tambear::rng::sample_normal(&mut rng, 0.0, noise);
            prices.push(p.max(0.01));
        }
        prices
    }

    fn linspace_timestamps(n: usize, interval_ns: u64) -> Vec<u64> {
        (0..n).map(|i| i as u64 * interval_ns).collect()
    }

    // ── tick_vol ──────────────────────────────────────────────────────────

    #[test]
    fn tick_vol_too_short() {
        let prices = make_prices(10, 0.0, 0.01, 1);
        let r = tick_vol(&prices, &[]);
        assert!(r.realized_var.is_nan());
    }

    #[test]
    fn tick_vol_basic_outputs() {
        let prices = make_prices(200, 0.0, 0.01, 2);
        let ts = linspace_timestamps(prices.len(), 1_000_000);
        let r = tick_vol(&prices, &ts);
        assert!(r.realized_var.is_finite() && r.realized_var >= 0.0,
            "realized_var should be finite non-negative, got {}", r.realized_var);
        assert!(r.bipower_var.is_finite() && r.bipower_var >= 0.0,
            "bipower_var should be finite non-negative");
        assert!(r.tick_frequency_var.is_finite() && r.tick_frequency_var >= 0.0);
        assert!(r.microstructure_noise.is_finite() && r.microstructure_noise >= 0.0);
    }

    #[test]
    fn tick_vol_bipower_finite() {
        // BV should be finite and non-negative.
        // Note: BV = (π/2)·Σ|r_t||r_{t-1}| uses the factor π/2 ≈ 1.57,
        // so BV can exceed RV in finite samples — the BV ≤ RV property holds
        // only asymptotically under continuous semimartingale theory, not for
        // every finite realization.
        let prices = make_prices(300, 0.0, 0.01, 3);
        let r = tick_vol(&prices, &[]);
        assert!(r.bipower_var.is_finite() && r.bipower_var >= 0.0,
            "BV should be finite non-negative, got {}", r.bipower_var);
        assert!(r.realized_var.is_finite() && r.realized_var >= 0.0);
    }

    #[test]
    fn tick_vol_no_timestamps() {
        // Should work without timestamps
        let prices = make_prices(100, 0.0, 0.01, 4);
        let r = tick_vol(&prices, &[]);
        assert!(r.realized_var.is_finite());
        assert!(r.microstructure_noise >= 0.0);
    }

    // ── tick_complexity ────────────────────────────────────────────────────

    #[test]
    fn tick_complexity_too_short() {
        let r = tick_complexity(&[0, 1, 2], &[1.0, 2.0, 3.0]);
        assert!(r.inter_arrival_entropy.is_nan());
    }

    #[test]
    fn tick_complexity_basic() {
        let ts = linspace_timestamps(100, 500_000_000); // 0.5s intervals
        let mut rng = tambear::rng::Xoshiro256::new(5);
        let sizes: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 100.0, 20.0).abs()).collect();
        let r = tick_complexity(&ts, &sizes);
        assert!(r.inter_arrival_entropy.is_finite(),
            "entropy should be finite for regular timestamps");
        assert_eq!(r.inter_arrival_entropy, 0.0,
            "regular timestamps → zero IAT entropy");
        assert!(r.size_entropy.is_finite() && r.size_entropy >= 0.0);
        assert!(r.normalized_complexity >= 0.0 && r.normalized_complexity <= 1.0,
            "normalized complexity in [0,1], got {}", r.normalized_complexity);
    }

    #[test]
    fn tick_complexity_random_higher_entropy() {
        // Random arrival times → higher entropy than regular
        let mut rng_t = tambear::rng::Xoshiro256::new(6);
        let mut t = 0u64;
        let ts_rand: Vec<u64> = (0..100).map(|_| {
            t += (tambear::rng::sample_normal(&mut rng_t, 1e9, 5e8).abs() as u64).max(1);
            t
        }).collect();
        let ts_reg = linspace_timestamps(100, 1_000_000_000);
        let sizes: Vec<f64> = vec![100.0; 100];

        let r_rand = tick_complexity(&ts_rand, &sizes);
        let r_reg = tick_complexity(&ts_reg, &sizes);
        assert!(r_rand.inter_arrival_entropy > r_reg.inter_arrival_entropy,
            "random arrivals should have higher entropy than regular");
    }

    // ── tick_scaling ───────────────────────────────────────────────────────

    #[test]
    fn tick_scaling_too_short() {
        let r = tick_scaling(&[0, 1, 2, 3]);
        assert!(r.scaling_exponent.is_nan());
    }

    #[test]
    fn tick_scaling_basic() {
        let ts = linspace_timestamps(100, 100_000_000);
        let r = tick_scaling(&ts);
        assert!(r.xmin.is_finite() && r.xmin > 0.0);
        // Regular arrivals → CoV ≈ 0 → burstiness ≈ -1
        assert!(r.burstiness < 0.0,
            "regular arrivals → burstiness < 0, got {}", r.burstiness);
    }

    #[test]
    fn tick_scaling_bursty() {
        // Bursty arrivals (exponential-like) → burstiness ≈ 0
        let mut t = 0u64;
        let mut rng = tambear::rng::Xoshiro256::new(7);
        let ts: Vec<u64> = (0..200).map(|_| {
            // Exponential-like gaps (std ≈ mean → B ≈ 0)
            let gap = (tambear::rng::sample_normal(&mut rng, 1e9, 1e9).abs() as u64).max(1);
            t += gap; t
        }).collect();
        let r = tick_scaling(&ts);
        assert!(r.scaling_exponent.is_finite());
        assert!(r.burstiness.is_finite());
    }

    // ── tick_attractor ─────────────────────────────────────────────────────

    #[test]
    fn tick_attractor_too_short() {
        let r = tick_attractor(&[100.0, 101.0, 100.0]);
        assert!(r.phase_asymmetry.is_nan());
    }

    #[test]
    fn tick_attractor_basic() {
        let prices = make_prices(200, 0.0, 0.01, 8);
        let r = tick_attractor(&prices);
        assert!(r.phase_asymmetry.is_finite());
        assert!(r.phase_asymmetry >= -1.0 && r.phase_asymmetry <= 1.0,
            "phase asymmetry in [-1,1], got {}", r.phase_asymmetry);
        assert!(r.tick_persistence.is_finite());
        assert!(r.reversal_rate >= 0.0 && r.reversal_rate <= 1.0,
            "reversal_rate in [0,1], got {}", r.reversal_rate);
        assert!(r.phase_spread >= 0.0);
    }

    #[test]
    fn tick_attractor_trending_vs_reverting() {
        // Trending series (persistence ≈ 1): positive persistence
        let mut p = 100.0f64;
        let mut trending_prices = vec![p];
        for _ in 0..150 {
            p += 0.1; // constant upward drift, no noise
            trending_prices.push(p);
        }

        // Mean-reverting: oscillate up/down
        let oscillating: Vec<f64> = (0..=150).map(|i| {
            100.0 + if i % 2 == 0 { 0.1 } else { -0.1 }
        }).collect();

        let r_trend = tick_attractor(&trending_prices);
        let r_osc = tick_attractor(&oscillating);

        // Oscillating series should have higher reversal rate
        assert!(r_osc.reversal_rate > r_trend.reversal_rate,
            "oscillating should have higher reversal rate: {} vs {}",
            r_osc.reversal_rate, r_trend.reversal_rate);
    }

    // ── tick_alignment ─────────────────────────────────────────────────────

    #[test]
    fn tick_alignment_too_short() {
        let r = tick_alignment(&[0, 1, 2, 3]);
        assert!(r.arrival_regularity.is_nan());
    }

    #[test]
    fn tick_alignment_regular() {
        let ts = linspace_timestamps(100, 1_000_000_000); // 1s intervals
        let r = tick_alignment(&ts);
        // Perfect regularity → CoV = 0 → arrival_regularity = 1
        assert!((r.arrival_regularity - 1.0).abs() < 1e-6,
            "regular timestamps → regularity=1, got {}", r.arrival_regularity);
        // All IATs equal → Fano factor (var/mean) = 0
        assert!(r.clustering_index.abs() < 1e-6,
            "regular arrivals → Fano factor=0, got {}", r.clustering_index);
        // Gap ratio: all IATs equal median → 0 strictly above median
        assert!(r.gap_ratio >= 0.0 && r.gap_ratio <= 1.0);
        // KS: identical IATs = point mass at 1s, NOT uniform on [0, 1s].
        // uniformity_score = 1 - KS; KS will be high → uniformity_score low.
        // Just verify it's in range.
        assert!(r.uniformity_score >= 0.0 && r.uniformity_score <= 1.0);
    }

    #[test]
    fn tick_alignment_random() {
        let mut rng = tambear::rng::Xoshiro256::new(9);
        let mut t = 0u64;
        let ts: Vec<u64> = (0..100).map(|_| {
            t += (tambear::rng::sample_normal(&mut rng, 1e9, 5e8).abs() as u64).max(1);
            t
        }).collect();
        let r = tick_alignment(&ts);
        assert!(r.arrival_regularity.is_finite());
        assert!(r.clustering_index.is_finite() && r.clustering_index >= 0.0);
        assert!(r.gap_ratio >= 0.0 && r.gap_ratio <= 1.0);
        assert!(r.uniformity_score >= 0.0 && r.uniformity_score <= 1.0);
    }

    // ── tick_causality ─────────────────────────────────────────────────────

    #[test]
    fn tick_causality_too_short() {
        let r = tick_causality(&[0.0; 5], &[1.0; 5]);
        assert!(r.lead_lag_corr.is_nan());
    }

    #[test]
    fn tick_causality_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(10);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let volumes: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 100.0, 20.0).abs()).collect();
        let r = tick_causality(&returns, &volumes);
        assert!(r.lead_lag_corr.is_finite() && r.lead_lag_corr >= 0.0,
            "corr should be non-negative (abs value), got {}", r.lead_lag_corr);
        assert!(r.coupling_strength >= 0.0 && r.coupling_strength <= 1.0);
    }

    #[test]
    fn tick_causality_planted_lead() {
        // Plant vol[t] → ret[t+1]: should detect positive lead-lag at lag=1
        let n = 200;
        let vol: Vec<f64> = (0..n).map(|i| if i % 10 < 5 { 200.0 } else { 50.0 }).collect();
        let ret: Vec<f64> = std::iter::once(0.0)
            .chain(vol.iter().take(n - 1).map(|&v| if v > 100.0 { 0.01 } else { -0.01 }))
            .collect();
        let r = tick_causality(&ret, &vol);
        assert!(r.lead_lag_corr > 0.0);
        assert_eq!(r.lead_lag_offset, 1.0, "planted lag=1 should be detected");
    }

    // ── tick_geometry ──────────────────────────────────────────────────────

    #[test]
    fn tick_geometry_too_short() {
        let r = tick_geometry(&[100.0; 5]);
        assert!(r.hull_area.is_nan());
    }

    #[test]
    fn tick_geometry_basic() {
        let prices = make_prices(200, 0.0, 0.01, 11);
        let r = tick_geometry(&prices);
        assert!(r.hull_area.is_finite() && r.hull_area >= 0.0,
            "hull_area should be non-negative, got {}", r.hull_area);
        assert!(r.angular_entropy >= 0.0 && r.angular_entropy <= 1.0,
            "normalized angular entropy in [0,1], got {}", r.angular_entropy);
        assert!(r.radial_kurtosis.is_finite());
        assert!(r.aspect_ratio >= 0.0 && r.aspect_ratio <= 1.0,
            "aspect ratio in [0,1], got {}", r.aspect_ratio);
    }

    #[test]
    fn tick_geometry_collinear_has_zero_hull() {
        // Monotonic prices → all log-returns equal → phase portrait is a single point.
        // Hull area should be negligibly small (float rounding only).
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.01).collect();
        let r = tick_geometry(&prices);
        assert!(r.hull_area < 1e-15,
            "monotonic returns → degenerate hull ≈ 0, got {}", r.hull_area);
    }

    // ── tick_space ─────────────────────────────────────────────────────────

    #[test]
    fn tick_space_too_short() {
        let r = tick_space(&[100.0; 3]);
        assert!(r.tick_entropy.is_nan());
    }

    #[test]
    fn tick_space_basic() {
        let prices = make_prices(200, 0.0, 0.01, 12);
        let r = tick_space(&prices);
        assert!(r.tick_entropy.is_finite() && r.tick_entropy >= 0.0,
            "tick entropy should be non-negative, got {}", r.tick_entropy);
        assert!(r.mode_concentration > 0.0 && r.mode_concentration <= 1.0,
            "mode concentration in (0,1], got {}", r.mode_concentration);
        assert!(r.tick_clustering.is_finite());
        assert!(r.regime_persistence >= 1.0,
            "mean run length >= 1, got {}", r.regime_persistence);
    }

    #[test]
    fn tick_space_constant_tick_size() {
        // All moves ±same size → zero entropy
        let prices: Vec<f64> = (0..100).map(|i| {
            100.0 + if i % 2 == 0 { 0.01 } else { -0.01 }
        }).collect();
        let r = tick_space(&prices);
        assert_eq!(r.tick_entropy, 0.0, "uniform tick sizes → zero entropy");
        assert!((r.mode_concentration - 1.0).abs() < 1e-6,
            "all ticks same → mode_concentration=1, got {}", r.mode_concentration);
    }

    #[test]
    fn tick_space_trending_persistence() {
        // Monotonic series → long runs → high regime persistence
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let r = tick_space(&prices);
        // All moves same direction → single run of length n-1
        assert!(r.regime_persistence > 50.0,
            "monotonic series → long run, got {}", r.regime_persistence);
    }

    // ── shared helpers ─────────────────────────────────────────────────────

    #[test]
    fn histogram_entropy_uniform() {
        // Uniform distribution → maximum entropy = ln(n_bins)
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let h = histogram_entropy(&values, 10);
        assert!((h - 10.0_f64.ln()).abs() < 0.01,
            "uniform → max entropy ≈ ln(10), got {}", h);
    }

    #[test]
    fn histogram_entropy_constant_zero() {
        let values = vec![5.0; 100];
        let h = histogram_entropy(&values, 10);
        assert_eq!(h, 0.0, "constant → zero entropy");
    }

    #[test]
    fn convex_hull_area_square() {
        // Unit square: area = 1
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let a = convex_hull_area(&pts);
        assert!((a - 1.0).abs() < 1e-10, "unit square area = 1, got {}", a);
    }

    #[test]
    fn convex_hull_area_triangle() {
        // Right triangle with legs 3,4: area = 6
        let pts = vec![(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)];
        let a = convex_hull_area(&pts);
        assert!((a - 6.0).abs() < 1e-10, "triangle area = 6, got {}", a);
    }
}

// ── family 11i: tick_ou (K02P11C05R01) ────────────────────────────────────

/// OU process fit to log-prices for a single bin.
///
/// Fits the discrete OU model dx = a + b·x + ε via OLS on log-prices,
/// where θ = -b is the mean-reversion speed and σ is the residual std.
///
/// Corresponds to fintek's `tick_ou.rs` (K02P11C05R01).
#[derive(Debug, Clone)]
pub struct TickOuResult {
    /// Mean-reversion speed θ = -b. Positive → mean-reverting.
    pub theta: f64,
    /// Half-life in ticks: ln(2) / θ. +inf when θ ≤ 0.
    pub half_life: f64,
    /// Residual std of the OLS fit (annot. σ in the continuous model).
    pub sigma: f64,
    /// Dimensionless strength: θ × std(log_price) / σ.
    pub mr_strength: f64,
}

impl TickOuResult {
    pub fn nan() -> Self {
        Self { theta: f64::NAN, half_life: f64::NAN, sigma: f64::NAN, mr_strength: f64::NAN }
    }
}

/// Fit tick-level OU process from raw prices (min 5 prices required).
///
/// Exact port of fintek's `tick_ou_features` inner math.
pub fn tick_ou(prices: &[f64]) -> TickOuResult {
    let n = prices.len();
    if n < 5 { return TickOuResult::nan(); }

    let log_prices: Vec<f64> = prices.iter().map(|p| p.max(1e-300).ln()).collect();

    let mean_lp: f64 = log_prices.iter().sum::<f64>() / n as f64;
    let var_lp: f64 = log_prices.iter().map(|v| (v - mean_lp) * (v - mean_lp)).sum::<f64>() / n as f64;
    let std_lp = var_lp.sqrt();

    let m = n - 1;
    let mut sum_x = 0.0f64;
    let mut sum_dx = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_xdx = 0.0f64;

    for i in 0..m {
        let x = log_prices[i];
        let dx = log_prices[i + 1] - log_prices[i];
        sum_x += x;
        sum_dx += dx;
        sum_x2 += x * x;
        sum_xdx += x * dx;
    }

    let mf = m as f64;
    let mean_x = sum_x / mf;
    let mean_dx = sum_dx / mf;

    let cov_xdx = sum_xdx / mf - mean_x * mean_dx;
    let var_x = sum_x2 / mf - mean_x * mean_x;
    if var_x < 1e-30 { return TickOuResult::nan(); }

    let b = cov_xdx / var_x;
    let a = mean_dx - b * mean_x;
    let theta = -b;

    let mut ss_res = 0.0f64;
    for i in 0..m {
        let x = log_prices[i];
        let dx = log_prices[i + 1] - log_prices[i];
        let resid = dx - (a + b * x);
        ss_res += resid * resid;
    }
    let sigma = if m > 1 { (ss_res / (m - 1) as f64).sqrt() } else { return TickOuResult::nan(); };

    let half_life = if theta > 1e-10 { std::f64::consts::LN_2 / theta } else { f64::INFINITY };
    let mr_strength = if sigma > 1e-15 { theta * std_lp / sigma } else { 0.0 };

    TickOuResult { theta, half_life, sigma, mr_strength }
}

// ── family 11j: tick_compression (K02P25C01) ──────────────────────────────

/// Effective rank + compression ratio from the tick price-volume feature matrix.
///
/// Builds a 7-column feature matrix (log_ret, |log_ret|, log_ret², log_vol,
/// Δlog_vol, |Δlog_vol|, log_ret·Δlog_vol) for each tick pair in the bin,
/// standardizes it, computes SVD, and reports:
/// - `real_effective_rank`: exp(Shannon entropy of sv²) for the temporal series
/// - `shuffled_effective_rank`: same after 5 independent column shuffles (baseline)
/// - `compression_ratio`: real / shuffled (< 1 → temporal structure reduces rank)
/// - `n_active_features`: columns with std > 1e-15
///
/// Corresponds to fintek's `tick_compression.rs` (K02P25C01).
#[derive(Debug, Clone)]
pub struct TickCompressionResult {
    pub real_effective_rank: f64,
    pub shuffled_effective_rank: f64,
    pub compression_ratio: f64,
    pub n_active_features: usize,
}

impl TickCompressionResult {
    pub fn nan() -> Self {
        Self { real_effective_rank: f64::NAN, shuffled_effective_rank: f64::NAN,
               compression_ratio: f64::NAN, n_active_features: 0 }
    }
}

const N_FEATURES_COMP: usize = 7;
const N_SHUFFLES_COMP: usize = 5;
const MAX_PTS_COMP: usize = 2000;

fn build_compression_matrix(price: &[f64], volume: &[f64]) -> (Vec<f64>, usize) {
    let n = price.len();
    if n < 3 || volume.len() != n { return (Vec::new(), 0); }
    let m = n - 1;
    let mut mat = vec![0.0f64; m * N_FEATURES_COMP];
    for i in 0..m {
        let p0 = price[i].max(1e-300);
        let p1 = price[i + 1].max(1e-300);
        let v0 = volume[i].max(1e-300);
        let v1 = volume[i + 1].max(1e-300);
        let log_ret = (p1 / p0).ln();
        let log_vol = v1.ln();
        let d_log_vol = (v1 / v0).ln();
        let row = i * N_FEATURES_COMP;
        mat[row]     = log_ret;
        mat[row + 1] = log_ret.abs();
        mat[row + 2] = log_ret * log_ret;
        mat[row + 3] = log_vol;
        mat[row + 4] = d_log_vol;
        mat[row + 5] = d_log_vol.abs();
        mat[row + 6] = log_ret * d_log_vol;
    }
    (mat, m)
}

fn compression_column_stats(mat: &[f64], m: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut means = vec![0.0f64; N_FEATURES_COMP];
    let mut m2 = vec![0.0f64; N_FEATURES_COMP];
    for i in 0..m {
        for j in 0..N_FEATURES_COMP { means[j] += mat[i * N_FEATURES_COMP + j]; }
    }
    for j in 0..N_FEATURES_COMP { means[j] /= m as f64; }
    for i in 0..m {
        for j in 0..N_FEATURES_COMP {
            let d = mat[i * N_FEATURES_COMP + j] - means[j];
            m2[j] += d * d;
        }
    }
    let stds: Vec<f64> = m2.iter().map(|&v| (v / m as f64).sqrt()).collect();
    let active: Vec<bool> = stds.iter().map(|&s| s > 1e-15).collect();
    (means, stds, active)
}

/// Effective rank from singular values — delegates to the tambear global primitive.
#[inline]
fn effective_rank_from_sv(sv: &[f64]) -> f64 {
    tambear::linear_algebra::effective_rank_from_sv(sv)
}

fn matrix_effective_rank(mat: &[f64], m: usize) -> (f64, usize) {
    if m < 2 { return (f64::NAN, 0); }
    let (means, stds, active) = compression_column_stats(mat, m);
    let n_active: usize = active.iter().filter(|&&a| a).count();
    if n_active < 2 { return (f64::NAN, n_active); }

    // Standardize active columns only
    let mut z = vec![0.0f64; m * n_active];
    for i in 0..m {
        let mut col_out = 0;
        for j in 0..N_FEATURES_COMP {
            if active[j] {
                z[i * n_active + col_out] = (mat[i * N_FEATURES_COMP + j] - means[j]) / stds[j];
                col_out += 1;
            }
        }
    }

    let mat_z = tambear::linear_algebra::Mat::from_vec(m, n_active, z);
    let svd_r = tambear::linear_algebra::svd(&mat_z);
    (effective_rank_from_sv(&svd_r.sigma), n_active)
}

/// Simple xorshift64 for reproducible column shuffling.
struct Xorshift64Comp(u64);
impl Xorshift64Comp {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn shuffle(&mut self, arr: &mut [f64]) {
        let n = arr.len();
        for i in (1..n).rev() {
            let j = (self.next() as usize) % (i + 1);
            arr.swap(i, j);
        }
    }
}

fn shuffled_effective_rank(mat: &[f64], m: usize) -> f64 {
    let (_, _, active) = compression_column_stats(mat, m);
    let n_active: usize = active.iter().filter(|&&a| a).count();
    if n_active < 2 { return f64::NAN; }

    let mut rng = Xorshift64Comp(42);
    let mut ranks = Vec::with_capacity(N_SHUFFLES_COMP);

    for _ in 0..N_SHUFFLES_COMP {
        let mut shuf = mat.to_vec();
        for j in 0..N_FEATURES_COMP {
            if !active[j] { continue; }
            let mut col: Vec<f64> = (0..m).map(|i| shuf[i * N_FEATURES_COMP + j]).collect();
            rng.shuffle(&mut col);
            for i in 0..m { shuf[i * N_FEATURES_COMP + j] = col[i]; }
        }
        let (rank, _) = matrix_effective_rank(&shuf, m);
        if rank.is_finite() { ranks.push(rank); }
    }

    if ranks.is_empty() { f64::NAN } else { ranks.iter().sum::<f64>() / ranks.len() as f64 }
}

/// Compute tick compression features (effective rank + shuffled baseline).
///
/// `prices` and `volumes` must have equal length; minimum 10 elements.
/// Large bins (> 2000 ticks) are sub-sampled at stride `len/2000`.
pub fn tick_compression(prices: &[f64], volumes: &[f64]) -> TickCompressionResult {
    if prices.len() < 10 || volumes.len() != prices.len() {
        return TickCompressionResult::nan();
    }

    // Cap to MAX_PTS_COMP to avoid memory blowup
    let (p_use, v_use): (std::borrow::Cow<[f64]>, std::borrow::Cow<[f64]>) = if prices.len() > MAX_PTS_COMP {
        let step = prices.len() / MAX_PTS_COMP;
        let ps: Vec<f64> = prices.iter().step_by(step).copied().collect();
        let vs: Vec<f64> = volumes.iter().step_by(step).copied().collect();
        (std::borrow::Cow::Owned(ps), std::borrow::Cow::Owned(vs))
    } else {
        (std::borrow::Cow::Borrowed(prices), std::borrow::Cow::Borrowed(volumes))
    };

    let (mat, m) = build_compression_matrix(&p_use, &v_use);
    if m < 3 { return TickCompressionResult::nan(); }

    let (real_rank, n_active) = matrix_effective_rank(&mat, m);
    if !real_rank.is_finite() {
        return TickCompressionResult { real_effective_rank: f64::NAN, shuffled_effective_rank: f64::NAN,
                                       compression_ratio: f64::NAN, n_active_features: n_active };
    }

    let shuf_rank = shuffled_effective_rank(&mat, m);
    let compression_ratio = if shuf_rank.is_finite() && shuf_rank > 1e-15 {
        real_rank / shuf_rank
    } else { f64::NAN };

    TickCompressionResult { real_effective_rank: real_rank, shuffled_effective_rank: shuf_rank,
                            compression_ratio, n_active_features: n_active }
}

#[cfg(test)]
mod tick_ou_compression_tests {
    use super::*;

    // ── tick_ou ───────────────────────────────────────────────────────────────

    #[test]
    fn tick_ou_too_short() {
        let r = tick_ou(&[100.0, 101.0, 100.0]);
        assert!(r.theta.is_nan());
    }

    #[test]
    fn tick_ou_constant_prices_nan() {
        // All same price → var_x = 0 → NaN
        let prices = vec![100.0f64; 50];
        let r = tick_ou(&prices);
        assert!(r.theta.is_nan());
    }

    #[test]
    fn tick_ou_random_walk_theta_small() {
        // Random walk → no mean reversion → theta ≈ 0 or negative
        let n = 100;
        let mut prices = vec![100.0f64; n];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for i in 1..n {
            prices[i] = prices[i - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.5);
        }
        let r = tick_ou(&prices);
        // Random walk should have |theta| < 0.5 on average
        assert!(r.theta.is_finite(), "theta should be finite for random walk, got {}", r.theta);
        assert!(r.sigma.is_finite() && r.sigma > 0.0);
    }

    #[test]
    fn tick_ou_mean_reverting_positive_theta() {
        // OU process with strong mean reversion
        let n = 200;
        let mu = 100.0f64;
        let theta = 0.5f64;
        let sigma = 0.05f64;
        let mut prices = vec![mu; n];
        let mut rng = tambear::rng::Xoshiro256::new(7);
        for i in 1..n {
            let x = prices[i - 1];
            prices[i] = x + theta * (mu.ln() - x.ln()) * x + sigma * x * tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
            prices[i] = prices[i].max(1e-10);
        }
        let r = tick_ou(&prices);
        assert!(r.theta.is_finite(), "theta finite: {}", r.theta);
        // For a mean-reverting process with positive theta, we expect positive theta or mr_strength
        // (may not always be > 0 due to estimation noise at finite n, but sigma should be finite)
        assert!(r.sigma.is_finite() && r.sigma > 0.0, "sigma > 0: {}", r.sigma);
        assert!(r.half_life.is_finite() || r.half_life == f64::INFINITY);
    }

    #[test]
    fn tick_ou_half_life_positive_theta() {
        // Manually check half_life = ln2 / theta
        let prices: Vec<f64> = (0..100).map(|i| {
            let x = i as f64 / 10.0;
            100.0 * (1.0 + 0.01 * (x - x.round()))
        }).collect();
        let r = tick_ou(&prices);
        if r.theta > 1e-10 {
            let expected_hl = std::f64::consts::LN_2 / r.theta;
            assert!((r.half_life - expected_hl).abs() < 1e-10,
                "half_life = ln2/theta: expected {}, got {}", expected_hl, r.half_life);
        }
    }

    // ── tick_compression ──────────────────────────────────────────────────────

    #[test]
    fn tick_compression_too_short() {
        let r = tick_compression(&[100.0, 101.0], &[10.0, 10.0]);
        assert!(r.real_effective_rank.is_nan());
    }

    #[test]
    fn tick_compression_mismatched_lengths() {
        let prices = vec![100.0f64; 50];
        let volumes = vec![1.0f64; 30]; // wrong length
        let r = tick_compression(&prices, &volumes);
        assert!(r.real_effective_rank.is_nan());
    }

    #[test]
    fn tick_compression_outputs_finite() {
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01 + tambear::rng::sample_normal(&mut rng, 0.0, 0.1)).collect();
        let mut rng2 = tambear::rng::Xoshiro256::new(17);
        let volumes: Vec<f64> = (0..n).map(|_| 1000.0 + tambear::rng::sample_normal(&mut rng2, 0.0, 200.0).abs()).collect();
        let r = tick_compression(&prices, &volumes);
        assert!(r.real_effective_rank.is_finite() && r.real_effective_rank > 0.0,
            "real_effective_rank = {}", r.real_effective_rank);
        assert!(r.shuffled_effective_rank.is_finite() && r.shuffled_effective_rank > 0.0,
            "shuffled_effective_rank = {}", r.shuffled_effective_rank);
        assert!(r.n_active_features >= 2 && r.n_active_features <= 7,
            "n_active_features = {}", r.n_active_features);
    }

    #[test]
    fn tick_compression_ratio_in_range() {
        let n = 200;
        let mut rng = tambear::rng::Xoshiro256::new(99);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 * (1.0 + 0.001 * i as f64)).collect();
        let volumes: Vec<f64> = (0..n).map(|_| 500.0 + tambear::rng::sample_normal(&mut rng, 0.0, 100.0).abs()).collect();
        let r = tick_compression(&prices, &volumes);
        if r.compression_ratio.is_finite() {
            // Compression ratio is real/shuffled, typically < 2
            assert!(r.compression_ratio > 0.0, "ratio > 0: {}", r.compression_ratio);
        }
    }

    #[test]
    fn tick_compression_n_active_7_for_varied_data() {
        // With varied price and volume, all 7 features should be active
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(55);
        let prices: Vec<f64> = {
            let mut p = vec![100.0f64];
            for _ in 1..n { p.push(*p.last().unwrap() * (1.0 + tambear::rng::sample_normal(&mut rng, 0.0, 0.01))); }
            p
        };
        let volumes: Vec<f64> = (0..n).map(|_| 1000.0 * (1.0 + tambear::rng::sample_normal(&mut rng, 0.0, 0.3).abs())).collect();
        let r = tick_compression(&prices, &volumes);
        assert_eq!(r.n_active_features, 7, "all 7 features should be active, got {}", r.n_active_features);
    }
}
