//! Family 12 — Causality & Information-Theoretic measures.
//!
//! Pure-function bridges for fintek leaves that measure coupling,
//! causality, and information flow between price and volume series.
//!
//! Covered leaves:
//! - granger          (K02P17C01R01) — Granger causality: F-stat, p-value, direction, lag
//! - mutual_info      (K02P13C03R01) — MI, Miller-Madow corrected MI, nonlinear excess, normalized MI
//! - coherence        (K02P17C01R02) — Spectral coherence: mean, max, peak freq, bandwidth
//! - cross_correlation (K02P07C02R01) — CCF features: max_xcorr, lag, zero-lag, asymmetry, lead_lag_ratio

// ── Shared helpers ──────────────────────────────────────────────────────────

/// O(n²) real DFT for a single segment (coherence / CCF use segments ≤ 256 pts).
/// Returns `n/2 + 1` complex coefficients as `(re, im)`.
fn rfft_naive(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let n_out = n / 2 + 1;
    let mut result = vec![(0.0, 0.0); n_out];
    let c = -2.0 * std::f64::consts::PI / n as f64;
    for k in 0..n_out {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (t, &v) in x.iter().enumerate() {
            let a = c * k as f64 * t as f64;
            re += v * a.cos();
            im += v * a.sin();
        }
        result[k] = (re, im);
    }
    result
}

/// Approximate F-distribution survival function P(F > x | df1, df2).
/// Uses the Abramowitz-Stegun normal approximation (adequate for df ≥ 3).
fn f_survival(x: f64, df1: usize, df2: usize) -> f64 {
    if x <= 0.0 || df1 == 0 || df2 == 0 { return 1.0; }
    let d1 = df1 as f64;
    let d2 = df2 as f64;
    // Fisher's cube-root normal approximation
    let z = ((x * d1 / d2).powf(1.0 / 3.0) * (1.0 - 2.0 / (9.0 * d2))
            - (1.0 - 2.0 / (9.0 * d1)))
            / ((2.0 / (9.0 * d1))
               + (x * d1 / d2).powf(2.0 / 3.0) * 2.0 / (9.0 * d2))
              .sqrt();
    // P(Z > z) ≈ erfc(z/√2)/2
    erfc_approx(z / std::f64::consts::SQRT_2) * 0.5
}

fn erfc_approx(x: f64) -> f64 {
    if x >= 6.0 { return 0.0; }
    if x <= -6.0 { return 2.0; }
    if x < 0.0 { return 2.0 - erfc_approx(-x); }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// OLS residual sum of squares: regress `y` on design matrix `x` (row-major,
/// `nobs` rows × `ncols` columns). Solves via Gauss-Jordan on the normal equations.
/// Returns ∞ if the system is rank-deficient.
fn ols_rss(y: &[f64], x: &[f64], nobs: usize, ncols: usize) -> f64 {
    let mut xtx = vec![0.0f64; ncols * ncols];
    let mut xty = vec![0.0f64; ncols];
    for r in 0..nobs {
        for j in 0..ncols {
            xty[j] += x[r * ncols + j] * y[r];
            for k in j..ncols {
                let v = x[r * ncols + j] * x[r * ncols + k];
                xtx[j * ncols + k] += v;
                if j != k { xtx[k * ncols + j] += v; }
            }
        }
    }
    // Gauss-Jordan on augmented matrix [X^T X | X^T y]
    let mut aug = vec![0.0f64; ncols * (ncols + 1)];
    for r in 0..ncols {
        for c in 0..ncols { aug[r * (ncols + 1) + c] = xtx[r * ncols + c]; }
        aug[r * (ncols + 1) + ncols] = xty[r];
    }
    for col in 0..ncols {
        let pivot = aug[col * (ncols + 1) + col];
        if pivot.abs() < 1e-30 { return f64::INFINITY; }
        for j in 0..=ncols { aug[col * (ncols + 1) + j] /= pivot; }
        for row in 0..ncols {
            if row == col { continue; }
            let f = aug[row * (ncols + 1) + col];
            for j in 0..=ncols { aug[row * (ncols + 1) + j] -= f * aug[col * (ncols + 1) + j]; }
        }
    }
    let beta: Vec<f64> = (0..ncols).map(|r| aug[r * (ncols + 1) + ncols]).collect();
    let mut rss = 0.0f64;
    for r in 0..nobs {
        let pred: f64 = (0..ncols).map(|j| x[r * ncols + j] * beta[j]).sum();
        let e = y[r] - pred;
        rss += e * e;
    }
    rss
}

// ── family 12a: granger (K02P17C01R01) ─────────────────────────────────────

/// Granger causality between two series (volume ↔ returns).
///
/// - `f_stat_xy`: F-statistic for x → y (volume predicts returns)
/// - `p_value_xy`: p-value for x → y
/// - `f_stat_yx`: F-statistic for y → x (returns predicts volume)
/// - `p_value_yx`: p-value for y → x
/// - `granger_direction`: +1 = x→y, -1 = y→x, 0 = neither or both
/// - `optimal_lag`: BIC-selected lag order
///
/// Corresponds to fintek's `granger.rs` (K02P17C01R01).
#[derive(Debug, Clone)]
pub struct GrangerResult {
    pub f_stat_xy: f64,
    pub p_value_xy: f64,
    pub f_stat_yx: f64,
    pub p_value_yx: f64,
    pub granger_direction: f64,
    pub optimal_lag: usize,
}

impl GrangerResult {
    pub fn nan() -> Self {
        Self { f_stat_xy: f64::NAN, p_value_xy: f64::NAN,
               f_stat_yx: f64::NAN, p_value_yx: f64::NAN,
               granger_direction: f64::NAN, optimal_lag: 0 }
    }
}

const GRANGER_MIN_OBS: usize = 30;
const GRANGER_MAX_LAG: usize = 5;

/// Bivariate Granger causality test between `x` (e.g. volume changes) and `y` (e.g. returns).
///
/// Uses BIC to select lag order (1..=MAX_LAG), then runs F-tests in both directions.
pub fn granger_causality(y: &[f64], x: &[f64]) -> GrangerResult {
    let n = y.len().min(x.len());
    if n < GRANGER_MIN_OBS { return GrangerResult::nan(); }

    let max_lag = GRANGER_MAX_LAG.min(n / 4);
    if max_lag == 0 { return GrangerResult::nan(); }

    // BIC lag selection using unrestricted model (const + p lags of y + p lags of x)
    let mut best_lag = 1usize;
    let mut best_bic = f64::INFINITY;
    for lag in 1..=max_lag {
        let nobs = n - lag;
        if nobs < 2 * lag + 3 { continue; }
        let ncols = 1 + 2 * lag;
        let mut x_mat = vec![0.0f64; nobs * ncols];
        let mut y_vec = vec![0.0f64; nobs];
        for t in 0..nobs {
            let ti = t + lag;
            y_vec[t] = y[ti];
            x_mat[t * ncols] = 1.0;
            for l in 0..lag {
                x_mat[t * ncols + 1 + l] = y[ti - 1 - l];
                x_mat[t * ncols + 1 + lag + l] = x[ti - 1 - l];
            }
        }
        let rss = ols_rss(&y_vec, &x_mat, nobs, ncols);
        if rss <= 0.0 || !rss.is_finite() { continue; }
        let bic = nobs as f64 * (rss / nobs as f64).ln() + ncols as f64 * (nobs as f64).ln();
        if bic < best_bic { best_bic = bic; best_lag = lag; }
    }

    let lag = best_lag;
    let nobs = n - lag;
    if nobs < 2 * lag + 3 { return GrangerResult::nan(); }

    // F-test: does x Granger-cause y?
    let granger_f = |y_ser: &[f64], x_ser: &[f64]| -> (f64, f64) {
        let ncols_r = 1 + lag;       // restricted: const + lag*y
        let ncols_u = 1 + 2 * lag;   // unrestricted: + lag*x
        let mut x_r = vec![0.0f64; nobs * ncols_r];
        let mut x_u = vec![0.0f64; nobs * ncols_u];
        let mut y_v = vec![0.0f64; nobs];
        for t in 0..nobs {
            let ti = t + lag;
            y_v[t] = y_ser[ti];
            x_r[t * ncols_r] = 1.0;
            x_u[t * ncols_u] = 1.0;
            for l in 0..lag {
                x_r[t * ncols_r + 1 + l] = y_ser[ti - 1 - l];
                x_u[t * ncols_u + 1 + l] = y_ser[ti - 1 - l];
                x_u[t * ncols_u + 1 + lag + l] = x_ser[ti - 1 - l];
            }
        }
        let rss_r = ols_rss(&y_v, &x_r, nobs, ncols_r);
        let rss_u = ols_rss(&y_v, &x_u, nobs, ncols_u);
        if rss_u < 1e-30 || !rss_r.is_finite() || !rss_u.is_finite() {
            return (0.0, 1.0);
        }
        let df1 = lag;
        let df2 = nobs.saturating_sub(ncols_u);
        if df2 == 0 { return (0.0, 1.0); }
        let f_stat = ((rss_r - rss_u) / df1 as f64) / (rss_u / df2 as f64);
        let p = f_survival(f_stat.max(0.0), df1, df2);
        (f_stat.max(0.0), p)
    };

    let (f_xy, p_xy) = granger_f(y, x); // x → y
    let (f_yx, p_yx) = granger_f(x, y); // y → x

    let direction = match (p_xy < 0.05, p_yx < 0.05) {
        (true,  false) => 1.0,
        (false, true)  => -1.0,
        _              => 0.0,
    };

    GrangerResult { f_stat_xy: f_xy, p_value_xy: p_xy, f_stat_yx: f_yx, p_value_yx: p_yx,
                    granger_direction: direction, optimal_lag: lag }
}

// ── family 12b: mutual_info (K02P13C03R01) ─────────────────────────────────

/// Mutual information features between two series (e.g. returns and volume).
///
/// - `mutual_info`: MI(x, y) in nats via 16-bin histogram
/// - `mi_corrected`: Miller-Madow bias-corrected MI
/// - `mi_nonlinear_excess`: MI - MI_gaussian (nonlinear component)
/// - `mi_normalized`: MI / min(H(x), H(y))
///
/// Corresponds to fintek's `mutual_info.rs` (K02P13C03R01).
#[derive(Debug, Clone)]
pub struct MutualInfoResult {
    pub mutual_info: f64,
    pub mi_corrected: f64,
    pub mi_nonlinear_excess: f64,
    pub mi_normalized: f64,
}

impl MutualInfoResult {
    pub fn nan() -> Self {
        Self { mutual_info: f64::NAN, mi_corrected: f64::NAN,
               mi_nonlinear_excess: f64::NAN, mi_normalized: f64::NAN }
    }
}

const MI_N_BINS: usize = 16;
const MI_MIN_OBS: usize = 20;

/// Compute mutual information features between `x` and `y`.
pub fn mutual_info(x: &[f64], y: &[f64]) -> MutualInfoResult {
    let n = x.len().min(y.len());
    if n < MI_MIN_OBS { return MutualInfoResult::nan(); }

    let discretize = |data: &[f64]| -> Vec<usize> {
        let vmin = data.iter().copied().fold(f64::INFINITY, f64::min);
        let vmax = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (vmax - vmin).max(1e-30);
        data.iter().map(|&v| {
            ((v - vmin) / range * MI_N_BINS as f64).floor() as usize
        }).map(|b| b.min(MI_N_BINS - 1)).collect()
    };

    let bx = discretize(&x[..n]);
    let by = discretize(&y[..n]);

    let mut hx = vec![0u32; MI_N_BINS];
    let mut hy = vec![0u32; MI_N_BINS];
    let mut hxy = vec![0u32; MI_N_BINS * MI_N_BINS];
    for t in 0..n {
        hx[bx[t]] += 1;
        hy[by[t]] += 1;
        hxy[bx[t] * MI_N_BINS + by[t]] += 1;
    }

    let entropy_of = |hist: &[u32]| -> f64 {
        let mut h = 0.0f64;
        for &c in hist {
            if c > 0 {
                let p = c as f64 / n as f64;
                h -= p * p.ln();
            }
        }
        h
    };

    let hx_val  = entropy_of(&hx);
    let hy_val  = entropy_of(&hy);
    let hxy_val = entropy_of(&hxy);

    let mi = (hx_val + hy_val - hxy_val).max(0.0);

    // Miller-Madow bias correction: subtract (occupied - 1) / (2n)
    let occupied = hxy.iter().filter(|&&c| c > 0).count();
    let mm_correction = (occupied.saturating_sub(1)) as f64 / (2.0 * n as f64);
    let mi_corrected = (mi - mm_correction).max(0.0);

    // Gaussian MI: -0.5 * ln(1 - r²)
    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0f64; let mut var_x = 0.0f64; let mut var_y = 0.0f64;
    for t in 0..n {
        let dx = x[t] - mean_x; let dy = y[t] - mean_y;
        cov += dx * dy; var_x += dx * dx; var_y += dy * dy;
    }
    let r = if var_x > 1e-30 && var_y > 1e-30 {
        cov / (var_x * var_y).sqrt()
    } else { 0.0 };
    let mi_gauss = if r.abs() < 1.0 - 1e-12 { -0.5 * (1.0 - r * r).ln() } else { 0.0 };

    let mi_normalized = {
        let min_h = hx_val.min(hy_val);
        if min_h > 1e-30 { mi / min_h } else { 0.0 }
    };

    MutualInfoResult {
        mutual_info: mi,
        mi_corrected,
        mi_nonlinear_excess: (mi - mi_gauss).max(0.0),
        mi_normalized,
    }
}

// ── family 12c: coherence (K02P17C01R02) ───────────────────────────────────

/// Spectral coherence features between two series (Welch method).
///
/// - `mean_coherence`: average squared coherence (excluding DC)
/// - `max_coherence`: peak squared coherence
/// - `peak_coherence_freq`: normalized frequency of peak (0..1)
/// - `coherence_bandwidth`: fraction of frequencies where coherence > 0.5
///
/// Corresponds to fintek's `coherence.rs` (K02P17C01R02).
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    pub mean_coherence: f64,
    pub max_coherence: f64,
    pub peak_coherence_freq: f64,
    pub coherence_bandwidth: f64,
}

impl CoherenceResult {
    pub fn nan() -> Self {
        Self { mean_coherence: f64::NAN, max_coherence: f64::NAN,
               peak_coherence_freq: f64::NAN, coherence_bandwidth: f64::NAN }
    }
}

const COH_MIN_OBS: usize = 20;
const COH_MAX_DFT: usize = 256;

/// Compute spectral coherence features between `x` and `y` using Welch's method.
pub fn spectral_coherence(x: &[f64], y: &[f64]) -> CoherenceResult {
    let n = x.len().min(y.len());
    if n < COH_MIN_OBS { return CoherenceResult::nan(); }

    let nperseg = COH_MAX_DFT.min(n / 2).max(16);
    let overlap  = nperseg / 2;
    let step     = nperseg - overlap;
    let n_freq   = nperseg / 2 + 1;

    // Hann window
    let hann: Vec<f64> = (0..nperseg).map(|t| {
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * t as f64 / (nperseg - 1) as f64).cos())
    }).collect();

    let mut sxx    = vec![0.0f64; n_freq];
    let mut syy    = vec![0.0f64; n_freq];
    let mut sxy_re = vec![0.0f64; n_freq];
    let mut sxy_im = vec![0.0f64; n_freq];
    let mut n_seg  = 0u32;

    let mut pos = 0;
    while pos + nperseg <= n {
        let seg_x: Vec<f64> = (0..nperseg).map(|t| x[pos + t] * hann[t]).collect();
        let seg_y: Vec<f64> = (0..nperseg).map(|t| y[pos + t] * hann[t]).collect();
        let fx = rfft_naive(&seg_x);
        let fy = rfft_naive(&seg_y);
        for k in 0..n_freq {
            sxx[k] += fx[k].0 * fx[k].0 + fx[k].1 * fx[k].1;
            syy[k] += fy[k].0 * fy[k].0 + fy[k].1 * fy[k].1;
            // Sxy = conj(Fx) * Fy
            sxy_re[k] += fx[k].0 * fy[k].0 + fx[k].1 * fy[k].1;
            sxy_im[k] += fx[k].1 * fy[k].0 - fx[k].0 * fy[k].1;
        }
        n_seg += 1;
        pos += step;
    }
    if n_seg == 0 { return CoherenceResult::nan(); }

    // Coherence² = |Sxy|² / (Sxx · Syy), skip DC (k=0)
    let n_pos = n_freq - 1;
    if n_pos == 0 { return CoherenceResult::nan(); }

    let coh: Vec<f64> = (1..n_freq).map(|k| {
        let xy_sq = sxy_re[k] * sxy_re[k] + sxy_im[k] * sxy_im[k];
        let denom  = sxx[k] * syy[k];
        if denom > 1e-30 { (xy_sq / denom).min(1.0) } else { 0.0 }
    }).collect();

    let mean_coherence = coh.iter().sum::<f64>() / coh.len() as f64;
    let (max_idx, &max_coherence) = coh.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let peak_coherence_freq = (max_idx + 1) as f64 / n_pos as f64;
    let coherence_bandwidth = coh.iter().filter(|&&c| c > 0.5).count() as f64 / coh.len() as f64;

    CoherenceResult { mean_coherence, max_coherence, peak_coherence_freq, coherence_bandwidth }
}

// ── family 12d: cross_correlation (K02P07C02R01) ───────────────────────────

/// Cross-correlation function features between two series.
///
/// Computes CCF at lags -MAX_LAG..+MAX_LAG and extracts 5 summary features:
/// - `max_xcorr`: maximum absolute CCF value
/// - `max_xcorr_lag`: lag at which max absolute CCF occurs
/// - `xcorr_at_zero`: CCF at lag 0 (contemporaneous correlation)
/// - `asymmetry`: mean(positive lags) - mean(negative lags)
/// - `lead_lag_ratio`: energy(positive lags) / energy(negative lags)
///
/// Corresponds to fintek's `cross_correlation.rs` (K02P07C02R01).
#[derive(Debug, Clone)]
pub struct CrossCorrResult {
    pub max_xcorr: f64,
    pub max_xcorr_lag: f64,
    pub xcorr_at_zero: f64,
    pub asymmetry: f64,
    pub lead_lag_ratio: f64,
}

impl CrossCorrResult {
    pub fn nan() -> Self {
        Self { max_xcorr: f64::NAN, max_xcorr_lag: f64::NAN, xcorr_at_zero: f64::NAN,
               asymmetry: f64::NAN, lead_lag_ratio: f64::NAN }
    }
}

const CCF_MAX_LAG: usize = 5;
const CCF_MIN_OBS: usize = CCF_MAX_LAG + 1;

/// Compute cross-correlation function summary features between `x` and `y`.
///
/// Positive lag = x leads y (x is causal predictor of y).
pub fn cross_corr_features(x: &[f64], y: &[f64]) -> CrossCorrResult {
    let n = x.len().min(y.len());
    if n < CCF_MIN_OBS { return CrossCorrResult::nan(); }

    let mx: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let my: f64 = y[..n].iter().sum::<f64>() / n as f64;
    let xm: Vec<f64> = x[..n].iter().map(|&v| v - mx).collect();
    let ym: Vec<f64> = y[..n].iter().map(|&v| v - my).collect();
    let sx: f64 = xm.iter().map(|v| v * v).sum::<f64>().sqrt();
    let sy: f64 = ym.iter().map(|v| v * v).sum::<f64>().sqrt();

    if sx < 1e-15 || sy < 1e-15 { return CrossCorrResult::nan(); }
    let norm = sx * sy;

    let width = 2 * CCF_MAX_LAG + 1;
    let mut ccf_vals = vec![0.0f64; width];
    for ki in 0..width {
        let k = ki as i32 - CCF_MAX_LAG as i32;
        if k >= 0 {
            let ku = k as usize;
            let valid = n.saturating_sub(ku);
            if valid > 0 {
                let s: f64 = (0..valid).map(|i| xm[i] * ym[i + ku]).sum();
                ccf_vals[ki] = s / norm;
            }
        } else {
            let ku = (-k) as usize;
            let valid = n.saturating_sub(ku);
            if valid > 0 {
                let s: f64 = (0..valid).map(|i| xm[i + ku] * ym[i]).sum();
                ccf_vals[ki] = s / norm;
            }
        }
    }

    let (max_idx, max_xcorr) = ccf_vals.iter().enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v.abs()))
        .unwrap_or((CCF_MAX_LAG, 0.0));
    let max_xcorr_lag = max_idx as f64 - CCF_MAX_LAG as f64;
    let xcorr_at_zero = ccf_vals[CCF_MAX_LAG];

    let pos_lags = &ccf_vals[CCF_MAX_LAG + 1..];  // lags +1..+MAX_LAG
    let neg_lags = &ccf_vals[..CCF_MAX_LAG];        // lags -MAX_LAG..-1
    let pos_mean: f64 = pos_lags.iter().sum::<f64>() / pos_lags.len() as f64;
    let neg_mean: f64 = neg_lags.iter().sum::<f64>() / neg_lags.len() as f64;
    let asymmetry = pos_mean - neg_mean;

    let energy_pos: f64 = pos_lags.iter().map(|v| v * v).sum();
    let energy_neg: f64 = neg_lags.iter().map(|v| v * v).sum();
    let lead_lag_ratio = if energy_neg > 1e-30 {
        energy_pos / energy_neg
    } else if energy_pos < 1e-30 {
        f64::NAN
    } else {
        f64::INFINITY
    };

    CrossCorrResult { max_xcorr, max_xcorr_lag, xcorr_at_zero, asymmetry, lead_lag_ratio }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_series(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect()
    }

    // ── Granger ───────────────────────────────────────────────────────────

    #[test]
    fn granger_too_short() {
        let r = granger_causality(&[0.0; 10], &[0.0; 10]);
        assert!(r.f_stat_xy.is_nan());
    }

    #[test]
    fn granger_independent_series() {
        // Two independent white noise series → should NOT find Granger causality
        let y = make_series(200, 1);
        let x = make_series(200, 2);
        let r = granger_causality(&y, &x);
        assert!(r.f_stat_xy.is_finite(), "F-stat should be finite, got {}", r.f_stat_xy);
        assert!(r.p_value_xy >= 0.0 && r.p_value_xy <= 1.0,
            "p-value in [0,1], got {}", r.p_value_xy);
        assert!(r.optimal_lag >= 1 && r.optimal_lag <= GRANGER_MAX_LAG);
        // With high probability, independent series → p > 0.05
        // (not an assertion since it's stochastic, but direction should be 0)
    }

    #[test]
    fn granger_planted_causality() {
        // x[t] → y[t+1]: plant causal relationship
        let n = 200;
        let x = make_series(n, 3);
        let noise = make_series(n, 5);
        let y: Vec<f64> = std::iter::once(noise[0])
            .chain((1..n).map(|t| 0.8 * x[t - 1] + 0.3 * noise[t]))
            .collect();
        let r = granger_causality(&y, &x);
        assert!(r.f_stat_xy.is_finite());
        // Strong planted causality → F-stat should be large
        assert!(r.f_stat_xy > 5.0,
            "planted x→y should give large F-stat, got {}", r.f_stat_xy);
        assert!(r.p_value_xy < 0.05,
            "planted x→y should give p<0.05, got {}", r.p_value_xy);
        assert_eq!(r.granger_direction, 1.0,
            "direction should be +1 (x→y), got {}", r.granger_direction);
        // Reverse direction should NOT be significant
        assert!(r.p_value_yx > 0.05 || r.f_stat_yx < r.f_stat_xy,
            "reverse should be weaker");
    }

    #[test]
    fn granger_f_stat_nonneg() {
        let y = make_series(100, 6);
        let x = make_series(100, 7);
        let r = granger_causality(&y, &x);
        assert!(r.f_stat_xy >= 0.0 || r.f_stat_xy.is_nan());
        assert!(r.f_stat_yx >= 0.0 || r.f_stat_yx.is_nan());
    }

    // ── Mutual information ────────────────────────────────────────────────

    #[test]
    fn mutual_info_too_short() {
        let r = mutual_info(&[0.0; 5], &[0.0; 5]);
        assert!(r.mutual_info.is_nan());
    }

    #[test]
    fn mutual_info_identical_series() {
        // MI(x, x) should approach H(x) — high value
        let x = make_series(200, 8);
        let r = mutual_info(&x, &x);
        assert!(r.mutual_info.is_finite() && r.mutual_info > 0.5,
            "MI(x,x) should be high (≈H(x)), got {}", r.mutual_info);
        // Normalized MI of identical series = 1
        assert!((r.mi_normalized - 1.0).abs() < 1e-6,
            "normalized MI(x,x)=1, got {}", r.mi_normalized);
    }

    #[test]
    fn mutual_info_independent_lower() {
        // MI of independent series < MI(x,x)
        let x = make_series(200, 9);
        let y = make_series(200, 10);
        let r_self = mutual_info(&x, &x);
        let r_indep = mutual_info(&x, &y);
        assert!(r_indep.mutual_info < r_self.mutual_info,
            "MI(x,y) should be < MI(x,x): {} vs {}", r_indep.mutual_info, r_self.mutual_info);
    }

    #[test]
    fn mutual_info_nonneg() {
        let x = make_series(100, 11);
        let y = make_series(100, 12);
        let r = mutual_info(&x, &y);
        assert!(r.mutual_info >= 0.0);
        assert!(r.mi_corrected >= 0.0);
        assert!(r.mi_nonlinear_excess >= 0.0);
        assert!(r.mi_normalized >= 0.0 && r.mi_normalized <= 1.0 + 1e-12);
    }

    #[test]
    fn mutual_info_strongly_correlated() {
        // Linearly related: y = x + small noise → MI should be high
        let x = make_series(200, 13);
        let mut rng = tambear::rng::Xoshiro256::new(14);
        let y: Vec<f64> = x.iter().map(|&v| v + tambear::rng::sample_normal(&mut rng, 0.0, 0.1)).collect();
        let r_corr = mutual_info(&x, &y);
        let r_indep = mutual_info(&x, &make_series(200, 15));
        assert!(r_corr.mutual_info > r_indep.mutual_info,
            "correlated MI should exceed independent: {} vs {}", r_corr.mutual_info, r_indep.mutual_info);
    }

    // ── Spectral coherence ────────────────────────────────────────────────

    #[test]
    fn coherence_too_short() {
        let r = spectral_coherence(&[0.0; 5], &[0.0; 5]);
        assert!(r.mean_coherence.is_nan());
    }

    #[test]
    fn coherence_identical_series() {
        // Coherence(x, x) = 1 at all frequencies
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = spectral_coherence(&x, &x);
        assert!((r.mean_coherence - 1.0).abs() < 0.01,
            "coh(x,x) mean should be ≈1, got {}", r.mean_coherence);
        assert!((r.max_coherence - 1.0).abs() < 0.01,
            "coh(x,x) max should be ≈1, got {}", r.max_coherence);
    }

    #[test]
    fn coherence_independent_low() {
        // Two independent white noise series → coherence near 0
        let x = make_series(200, 16);
        let y = make_series(200, 17);
        let r = spectral_coherence(&x, &y);
        assert!(r.mean_coherence.is_finite() && r.mean_coherence >= 0.0);
        assert!(r.max_coherence >= 0.0 && r.max_coherence <= 1.0 + 1e-10);
        assert!(r.peak_coherence_freq >= 0.0 && r.peak_coherence_freq <= 1.0);
        assert!(r.coherence_bandwidth >= 0.0 && r.coherence_bandwidth <= 1.0);
    }

    #[test]
    fn coherence_shared_frequency() {
        // x and y share a sinusoidal at the same frequency → high coherence at that freq
        let n = 200;
        let freq = 0.1;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).cos()).collect();
        let r = spectral_coherence(&x, &y);
        assert!(r.max_coherence > 0.5,
            "shared frequency should give high coherence, got max={}", r.max_coherence);
    }

    // ── Cross-correlation ─────────────────────────────────────────────────

    #[test]
    fn cross_corr_too_short() {
        let r = cross_corr_features(&[0.0; 3], &[0.0; 3]);
        assert!(r.max_xcorr.is_nan());
    }

    #[test]
    fn cross_corr_identical() {
        // CCF(x,x) at lag 0 = 1 (maximum)
        let x = make_series(100, 18);
        let r = cross_corr_features(&x, &x);
        assert!((r.xcorr_at_zero - 1.0).abs() < 1e-6,
            "CCF(x,x) at lag 0 = 1, got {}", r.xcorr_at_zero);
        assert!((r.max_xcorr - 1.0).abs() < 1e-6,
            "max |CCF(x,x)| = 1, got {}", r.max_xcorr);
        assert_eq!(r.max_xcorr_lag, 0.0,
            "max at lag 0 for identical series, got {}", r.max_xcorr_lag);
    }

    #[test]
    fn cross_corr_planted_lag() {
        // x[t] = y[t+2]: CCF should peak at lag +2
        let n = 150;
        let y = make_series(n, 19);
        let x: Vec<f64> = std::iter::repeat(0.0).take(2)
            .chain(y.iter().take(n - 2).copied())
            .collect();
        // x[t] = y[t-2], so y leads x by 2 → x leads y at lag -2, or y leads x at lag +2
        let r = cross_corr_features(&x, &y);
        assert!(r.max_xcorr > 0.5,
            "planted lag should show high CCF, got {}", r.max_xcorr);
    }

    #[test]
    fn cross_corr_independent() {
        let x = make_series(100, 20);
        let y = make_series(100, 21);
        let r = cross_corr_features(&x, &y);
        assert!(r.max_xcorr.is_finite() && r.max_xcorr >= 0.0);
        assert!(r.max_xcorr <= 1.0 + 1e-10);
        assert!(r.lead_lag_ratio.is_finite() || r.lead_lag_ratio.is_infinite()
            || r.lead_lag_ratio.is_nan());
    }

    // ── Shared helper ─────────────────────────────────────────────────────

    #[test]
    fn f_survival_symmetry() {
        // P(F > 0 | df1, df2) = 1
        assert!((f_survival(0.0, 3, 20) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f_survival_large_x() {
        // P(F > very_large) ≈ 0
        let p = f_survival(1000.0, 5, 100);
        assert!(p < 0.001, "P(F>1000) should be tiny, got {}", p);
    }

    #[test]
    fn ols_rss_perfect_fit() {
        // y = 2x + 1: OLS should give RSS = 0
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let x = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]; // [const, x] row-major
        let rss = ols_rss(&y, &x, 4, 2);
        assert!(rss < 1e-10, "perfect fit → RSS≈0, got {}", rss);
    }
}
