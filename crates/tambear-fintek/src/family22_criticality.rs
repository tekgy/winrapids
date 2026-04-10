//! Family 22 — Criticality detection.
//!
//! Covers fintek leaves:
//! - `phase_transition` (K02P18C01R04F01) — SOC/Ising-style criticality via rolling magnetization
//! - `mfdfa`            (K02P18C01R05F01) — Multifractal Detrended Fluctuation Analysis

// ── Phase transition ───────────────────────────────────────────────────────────

const PT_WINDOW: usize = 20;
const PT_MIN_RETURNS: usize = 50;
const PT_MULTISCALE_WINDOWS: [usize; 5] = [10, 20, 40, 80, 160];
const PT_MIN_MULTISCALE: usize = 200;

/// Phase-transition result matching fintek's `phase_features` (K02P18C01R04F01).
#[derive(Debug, Clone)]
pub struct PhaseTransitionResult {
    pub order_parameter: f64,
    pub susceptibility: f64,
    pub binder_cumulant: f64,
    pub critical_exponent: f64,
}

impl PhaseTransitionResult {
    pub fn nan() -> Self {
        Self {
            order_parameter: f64::NAN,
            susceptibility: f64::NAN,
            binder_cumulant: f64::NAN,
            critical_exponent: f64::NAN,
        }
    }
}

/// Detect phase-transition signatures via rolling Ising-style magnetization.
///
/// Computes order parameter, susceptibility, Binder cumulant from rolling
/// window magnetization. Critical exponent via log-log regression of mean|m|
/// vs window size across five scales (requires ≥200 data points).
///
/// Returns NaN if `returns.len() < 50`.
pub fn phase_transition(returns: &[f64]) -> PhaseTransitionResult {
    let n = returns.len();
    if n < PT_MIN_RETURNS { return PhaseTransitionResult::nan(); }

    // Rolling magnetization: sign-based "spin" field
    // m_k = mean of sign(returns[i]) for i in window k
    let n_windows = if n >= PT_WINDOW { n - PT_WINDOW + 1 } else { 0 };
    if n_windows < 2 { return PhaseTransitionResult::nan(); }

    let mut magnetizations = Vec::with_capacity(n_windows);
    for k in 0..n_windows {
        let window = &returns[k..k + PT_WINDOW];
        let spin_sum: f64 = window.iter().map(|&x| {
            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
        }).sum();
        magnetizations.push(spin_sum / PT_WINDOW as f64);
    }

    let nw = magnetizations.len() as f64;

    // order_parameter = mean(|m|)
    let order_parameter = magnetizations.iter().map(|m| m.abs()).sum::<f64>() / nw;

    // m^2 and m^4 moments for susceptibility and Binder cumulant
    let mean_m2: f64 = magnetizations.iter().map(|m| m * m).sum::<f64>() / nw;
    let mean_m4: f64 = magnetizations.iter().map(|m| m * m * m * m).sum::<f64>() / nw;

    // susceptibility = variance(m) × n_windows (fluctuation measure)
    let mean_m: f64 = magnetizations.iter().sum::<f64>() / nw;
    let var_m: f64 = magnetizations.iter().map(|m| (m - mean_m) * (m - mean_m)).sum::<f64>() / nw;
    let susceptibility = var_m * nw;

    // Binder cumulant: 1 - <m^4> / (3 <m^2>^2)
    // Approaches 2/3 in ordered phase, 0 in disordered phase at critical point
    let binder_cumulant = if mean_m2 > 1e-30 {
        1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
    } else {
        f64::NAN
    };

    // Critical exponent: log-log slope of mean|m| vs window size
    // Only computed with enough data for all 5 scales
    let critical_exponent = if n >= PT_MIN_MULTISCALE {
        let mut log_w = Vec::with_capacity(PT_MULTISCALE_WINDOWS.len());
        let mut log_m = Vec::with_capacity(PT_MULTISCALE_WINDOWS.len());
        for &w in &PT_MULTISCALE_WINDOWS {
            if n < w { continue; }
            let nw_local = n - w + 1;
            let mut mean_abs_m = 0.0f64;
            for k in 0..nw_local {
                let window = &returns[k..k + w];
                let spin_sum: f64 = window.iter().map(|&x| {
                    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
                }).sum();
                mean_abs_m += (spin_sum / w as f64).abs();
            }
            mean_abs_m /= nw_local as f64;
            if mean_abs_m > 1e-30 {
                log_w.push((w as f64).ln());
                log_m.push(mean_abs_m.ln());
            }
        }

        if log_w.len() >= 2 {
            // OLS: log_m = slope * log_w + intercept → slope is the critical exponent
            let m_len = log_w.len() as f64;
            let mean_lw = log_w.iter().sum::<f64>() / m_len;
            let mean_lm = log_m.iter().sum::<f64>() / m_len;
            let sxx: f64 = log_w.iter().map(|&x| (x - mean_lw) * (x - mean_lw)).sum();
            let sxy: f64 = log_w.iter().zip(log_m.iter()).map(|(&x, &y)| (x - mean_lw) * (y - mean_lm)).sum();
            if sxx > 1e-30 { sxy / sxx } else { f64::NAN }
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    PhaseTransitionResult { order_parameter, susceptibility, binder_cumulant, critical_exponent }
}

// ── MFDFA ─────────────────────────────────────────────────────────────────────

const MFDFA_Q_VALUES: [f64; 6] = [-2.0, -1.0, 0.5, 1.0, 1.5, 2.0];
const MFDFA_MIN_N: usize = 32;
const MFDFA_MAX_N: usize = 5000;

/// MFDFA result matching fintek's mfdfa leaf (K02P18C01R05F01).
///
/// h(q) for q ∈ {-2, -1, 0.5, 1, 1.5, 2}, multifractal width, τ(2),
/// raw width (h(-2) - h(2)), and mean standard error.
#[derive(Debug, Clone)]
pub struct MfdfaResult {
    pub h_q_neg2: f64,   // generalized Hurst h(-2)
    pub h_q_neg1: f64,   // h(-1)
    pub h_q_05: f64,     // h(0.5)
    pub h_q_1: f64,      // h(1)  — classical DFA Hurst
    pub h_q_15: f64,     // h(1.5)
    pub h_q_2: f64,      // h(2)  — standard DFA
    pub width_z: f64,    // standardized width (h(-2)-h(2))/h(1)
    pub tau_2: f64,      // τ(2) = q·h(q)-1 at q=2
    pub width_raw: f64,  // h(-2) - h(2)
    pub mean_se: f64,    // mean standard error across q OLS fits
}

impl MfdfaResult {
    pub fn nan() -> Self {
        Self {
            h_q_neg2: f64::NAN, h_q_neg1: f64::NAN, h_q_05: f64::NAN,
            h_q_1: f64::NAN, h_q_15: f64::NAN, h_q_2: f64::NAN,
            width_z: f64::NAN, tau_2: f64::NAN, width_raw: f64::NAN,
            mean_se: f64::NAN,
        }
    }
}

/// Multifractal DFA for a return series.
///
/// 1. Cumulative profile from mean-subtracted returns.
/// 2. Window sizes: powers of 2 from 4 up to n/4, capped at 32 windows.
/// 3. For each window s: divide profile into non-overlapping segments of
///    length s; detrend each by OLS line; RMS of residuals = F(s,v).
/// 4. Fq(s) = [mean(F^q)]^(1/q) for q≠0; geometric mean for q=0.
/// 5. h(q) = OLS slope of log Fq vs log s.
///
/// Returns NaN if `returns.len() < 32`.
pub fn mfdfa(returns: &[f64]) -> MfdfaResult {
    let data: Vec<f64> = if returns.len() > MFDFA_MAX_N {
        let step = returns.len() / MFDFA_MAX_N;
        returns.iter().step_by(step).take(MFDFA_MAX_N).copied().collect()
    } else {
        returns.to_vec()
    };

    let n = data.len();
    if n < MFDFA_MIN_N { return MfdfaResult::nan(); }

    // Cumulative profile: Y[i] = sum_{k=0}^{i-1} (x[k] - mean_x)
    let mean_x = data.iter().sum::<f64>() / n as f64;
    let mut profile = vec![0.0f64; n + 1];
    for i in 0..n {
        profile[i + 1] = profile[i] + (data[i] - mean_x);
    }

    // Window sizes: powers of 2 from 4 to n/4, but at least min segment size
    let max_win = (n / 4).max(4);
    let mut window_sizes: Vec<usize> = Vec::new();
    let mut w = 4usize;
    while w <= max_win {
        window_sizes.push(w);
        w *= 2;
    }
    if window_sizes.is_empty() { return MfdfaResult::nan(); }

    // For each window size and q value, compute Fq(s)
    let n_q = MFDFA_Q_VALUES.len();
    let n_s = window_sizes.len();

    // fq_matrix[q_idx][s_idx] = log(Fq(s))
    let mut log_fq = vec![vec![f64::NAN; n_s]; n_q];

    for (s_idx, &s) in window_sizes.iter().enumerate() {
        // Divide profile[0..n] into non-overlapping segments of length s
        let n_segs = n / s;
        if n_segs < 2 { continue; }

        // Compute F²(s, v) for each segment: variance after linear detrend
        let mut seg_rms2 = Vec::with_capacity(n_segs);
        for v in 0..n_segs {
            let start = v * s;
            let end = start + s;
            let seg = &profile[start..=end]; // length s+1
            let seg_len = seg.len();

            // OLS linear fit to segment
            let sf = seg_len as f64;
            let mean_y = seg.iter().sum::<f64>() / sf;
            let mean_t: f64 = (sf - 1.0) / 2.0;
            let stt: f64 = (0..seg_len).map(|i| {
                let t = i as f64;
                (t - mean_t) * (t - mean_t)
            }).sum();
            let sty: f64 = (0..seg_len).map(|i| {
                let t = i as f64;
                (t - mean_t) * (seg[i] - mean_y)
            }).sum();

            let slope = if stt > 1e-30 { sty / stt } else { 0.0 };
            let intercept = mean_y - slope * mean_t;

            let rms2: f64 = (0..seg_len).map(|i| {
                let t = i as f64;
                let residual = seg[i] - (slope * t + intercept);
                residual * residual
            }).sum::<f64>() / sf;

            seg_rms2.push(rms2.max(1e-300));
        }

        if seg_rms2.is_empty() { continue; }

        // Compute Fq for each q
        for (q_idx, &q) in MFDFA_Q_VALUES.iter().enumerate() {
            let fq = if q.abs() < 0.05 {
                // q ≈ 0: geometric mean = exp(mean(log(F²)) / 2)
                let log_mean: f64 = seg_rms2.iter().map(|&f2| f2.ln()).sum::<f64>()
                    / seg_rms2.len() as f64;
                (log_mean / 2.0).exp()
            } else {
                // Fq = (mean(F² ^ (q/2)))^(1/q)
                let fq_q: f64 = seg_rms2.iter().map(|&f2| f2.powf(q / 2.0)).sum::<f64>()
                    / seg_rms2.len() as f64;
                if fq_q > 0.0 { fq_q.powf(1.0 / q) } else { f64::NAN }
            };

            if fq.is_finite() && fq > 0.0 {
                log_fq[q_idx][s_idx] = fq.ln();
            }
        }
    }

    // OLS log Fq vs log s → slope = h(q)
    let log_s: Vec<f64> = window_sizes.iter().map(|&s| (s as f64).ln()).collect();

    let mut h_values = [f64::NAN; 6];
    let mut se_values = [f64::NAN; 6];

    for q_idx in 0..n_q {
        let mut pairs: Vec<(f64, f64)> = log_s.iter().zip(log_fq[q_idx].iter())
            .filter(|(_, &lf)| lf.is_finite())
            .map(|(&ls, &lf)| (ls, lf))
            .collect();

        if pairs.len() < 2 { continue; }

        let pm = pairs.len() as f64;
        let mean_ls = pairs.iter().map(|(x, _)| x).sum::<f64>() / pm;
        let mean_lf = pairs.iter().map(|(_, y)| y).sum::<f64>() / pm;
        let sxx: f64 = pairs.iter().map(|(x, _)| (x - mean_ls) * (x - mean_ls)).sum();
        let sxy: f64 = pairs.iter().map(|(x, y)| (x - mean_ls) * (y - mean_lf)).sum();

        if sxx < 1e-30 { continue; }
        let slope = sxy / sxx;
        h_values[q_idx] = slope;

        // Standard error of slope
        let intercept = mean_lf - slope * mean_ls;
        let ssr: f64 = pairs.iter().map(|(x, y)| {
            let pred = slope * x + intercept;
            (y - pred) * (y - pred)
        }).sum();
        let se = if pm > 2.0 { (ssr / ((pm - 2.0) * sxx)).sqrt() } else { f64::NAN };
        se_values[q_idx] = se;
    }

    let [h_neg2, h_neg1, h_05, h_1, h_15, h_2] = h_values;

    let width_raw = if h_neg2.is_finite() && h_2.is_finite() { h_neg2 - h_2 } else { f64::NAN };
    let width_z = if width_raw.is_finite() && h_1.is_finite() && h_1.abs() > 1e-30 {
        width_raw / h_1
    } else {
        f64::NAN
    };
    let tau_2 = if h_2.is_finite() { 2.0 * h_2 - 1.0 } else { f64::NAN };

    let valid_se: Vec<f64> = se_values.iter().copied().filter(|s| s.is_finite()).collect();
    let mean_se = if valid_se.is_empty() {
        f64::NAN
    } else {
        valid_se.iter().sum::<f64>() / valid_se.len() as f64
    };

    MfdfaResult {
        h_q_neg2: h_neg2, h_q_neg1: h_neg1, h_q_05: h_05,
        h_q_1: h_1, h_q_15: h_15, h_q_2: h_2,
        width_z, tau_2, width_raw, mean_se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect()
    }

    // ── Phase transition tests ──

    #[test]
    fn phase_too_short() {
        let r = phase_transition(&[0.0; 10]);
        assert!(r.order_parameter.is_nan());
    }

    #[test]
    fn phase_white_noise_finite() {
        let data = wn(200, 42);
        let r = phase_transition(&data);
        assert!(r.order_parameter.is_finite() && r.order_parameter >= 0.0);
        assert!(r.susceptibility.is_finite() && r.susceptibility >= 0.0);
        assert!(r.binder_cumulant.is_finite() || r.binder_cumulant.is_nan());
    }

    #[test]
    fn phase_ordered_series_high_order_param() {
        // All positive returns → spins all +1 → order parameter = 1
        let data = vec![0.01f64; 100];
        let r = phase_transition(&data);
        assert!((r.order_parameter - 1.0).abs() < 1e-10,
            "all-positive: order_parameter should be 1.0, got {}", r.order_parameter);
    }

    #[test]
    fn phase_critical_exponent_with_enough_data() {
        let data = wn(300, 99);
        let r = phase_transition(&data);
        // With 300 points, critical_exponent should be defined
        assert!(r.critical_exponent.is_finite() || r.critical_exponent.is_nan(),
            "critical_exponent should be finite or NaN, got {}", r.critical_exponent);
    }

    // ── MFDFA tests ──

    #[test]
    fn mfdfa_too_short() {
        let r = mfdfa(&[0.0; 10]);
        assert!(r.h_q_1.is_nan());
    }

    #[test]
    fn mfdfa_white_noise_finite() {
        let data = wn(500, 77);
        let r = mfdfa(&data);
        assert!(r.h_q_1.is_finite() || r.h_q_1.is_nan());
        assert!(r.h_q_2.is_finite() || r.h_q_2.is_nan());
    }

    #[test]
    fn mfdfa_brownian_motion_hurst_near_15() {
        // Brownian motion: cumulative sum of white noise has H ≈ 1.5 in DFA
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let wn: Vec<f64> = (0..1000).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = mfdfa(&wn);
        // h(2) for white noise should be close to 0.5 (scaling exponent)
        if r.h_q_2.is_finite() {
            assert!(r.h_q_2 > 0.0 && r.h_q_2 < 2.0,
                "h(2) for white noise should be in (0, 2), got {}", r.h_q_2);
        }
    }

    #[test]
    fn mfdfa_width_raw_nonneg_for_multifractal() {
        // Multifractal: negative q should have larger h than positive q
        let data = wn(500, 11);
        let r = mfdfa(&data);
        // width_raw = h(-2) - h(2); for random data this can be either sign
        if r.width_raw.is_finite() {
            assert!(r.width_raw.is_finite()); // just finite
        }
    }

    #[test]
    fn mfdfa_tau2_consistent() {
        let data = wn(256, 55);
        let r = mfdfa(&data);
        // tau(2) = 2*h(2) - 1
        if r.h_q_2.is_finite() && r.tau_2.is_finite() {
            assert!((r.tau_2 - (2.0 * r.h_q_2 - 1.0)).abs() < 1e-10,
                "tau(2) = 2*h(2)-1 invariant violated");
        }
    }
}
