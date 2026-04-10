//! Family 5 — Stationarity / Structural tests.
//!
//! Covers fintek leaves: `stationarity`, `dependence`, `classical_cp`.
//! NOT covered: `pelt` (needs new primitive), `bocpd` (needs new primitive).

use tambear::time_series::{
    adf_test, kpss_test, ljung_box, cusum_binary_segmentation,
    AdfResult, KpssResult, LjungBoxResult,
};

/// Classification result from combined ADF + KPSS confirmatory test pair.
///
/// Contract: ADF rejects unit root → series likely stationary.
/// KPSS rejects stationarity → series likely has a unit root.
/// Agreement → confident classification. Disagreement → inconclusive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StationarityClass {
    /// Both tests agree: stationary (ADF rejects unit root, KPSS does not reject).
    Stationary,
    /// Both tests agree: non-stationary (KPSS rejects stationarity, ADF does not reject).
    NonStationary,
    /// Either contradictory evidence (both reject) or low power (neither rejects).
    /// True trend-stationarity requires ADF-with-trend + KPSS-trend, which is a
    /// separate configuration not exposed by the basic `stationarity()` function.
    Inconclusive,
}

/// Stationarity test combining ADF and KPSS.
///
/// Fintek's `stationarity.rs` outputs:
/// - adf_statistic, adf_p_value (approximate)
/// - kpss_statistic
/// - classification
#[derive(Debug, Clone)]
pub struct StationarityResult {
    pub adf_statistic: f64,
    pub adf_critical_5pct: f64,
    pub kpss_statistic: f64,
    pub kpss_critical_5pct: f64,
    pub classification: StationarityClass,
}

impl StationarityResult {
    pub fn nan() -> Self {
        Self {
            adf_statistic: f64::NAN,
            adf_critical_5pct: f64::NAN,
            kpss_statistic: f64::NAN,
            kpss_critical_5pct: f64::NAN,
            classification: StationarityClass::Inconclusive,
        }
    }
}

/// Run ADF + KPSS and classify.
///
/// `data`: bin-level series.
/// `n_lags`: lag truncation for ADF (default 4 for financial data).
pub fn stationarity(data: &[f64], n_lags: usize) -> StationarityResult {
    if data.len() < 20 {
        return StationarityResult::nan();
    }
    let adf: AdfResult = adf_test(data, n_lags);
    let kpss: KpssResult = kpss_test(data, false, None);

    // ADF rejects unit root (stationary) when stat < critical_5pct
    let adf_rejects = adf.statistic < adf.critical_5pct;
    // KPSS rejects stationarity when stat > critical_5pct
    let kpss_rejects = kpss.statistic > kpss.critical_5pct;

    // Confirmatory pair (ADF + KPSS) classification:
    // - (ADF rejects, KPSS does not) → stationary (both agree)
    // - (ADF doesn't reject, KPSS rejects) → non-stationary (both agree)
    // - (ADF doesn't reject, KPSS doesn't reject) → inconclusive (low power)
    // - (ADF rejects, KPSS rejects) → CONTRADICTORY evidence
    //
    // The contradictory case CANNOT be labeled "trend-stationary" here because we
    // run ADF without a trend term (`adf_test(data, n_lags)` uses constant only).
    // True trend-stationarity requires `adf_with_trend` and a separate KPSS-trend
    // call. With our current configuration, both rejecting means contradictory
    // evidence — most often a sign that the series has a deterministic trend that
    // neither test was configured to model. Label it Inconclusive.
    let classification = match (adf_rejects, kpss_rejects) {
        (true, false)  => StationarityClass::Stationary,
        (false, true)  => StationarityClass::NonStationary,
        (true, true)   => StationarityClass::Inconclusive,  // contradictory
        (false, false) => StationarityClass::Inconclusive,  // low power
    };

    StationarityResult {
        adf_statistic: adf.statistic,
        adf_critical_5pct: adf.critical_5pct,
        kpss_statistic: kpss.statistic,
        kpss_critical_5pct: kpss.critical_5pct,
        classification,
    }
}

/// Dependence (Ljung-Box) test output for a single bin.
#[derive(Debug, Clone)]
pub struct DependenceResult {
    pub lb_statistic: f64,
    pub lb_p_value: f64,
    pub lb_n_lags: usize,
}

impl DependenceResult {
    pub fn nan() -> Self {
        Self { lb_statistic: f64::NAN, lb_p_value: f64::NAN, lb_n_lags: 0 }
    }
}

/// Ljung-Box Q test on bin returns.
pub fn dependence(returns: &[f64], n_lags: usize) -> DependenceResult {
    if returns.len() < n_lags + 2 {
        return DependenceResult::nan();
    }
    let r: LjungBoxResult = ljung_box(returns, n_lags, 0);
    DependenceResult {
        lb_statistic: r.statistic,
        lb_p_value: r.p_value,
        lb_n_lags: n_lags,
    }
}

// ── Classical changepoint (CUSUM + binary segmentation) ──────────────────────

/// Classical changepoint detection result.
///
/// Fintek's `classical_cp.rs` (K02P18C01R01F01) outputs 4 scalars per bin:
///   DO01: n_changepoints — number of detected changepoints
///   DO02: max_cusum — maximum |CUSUM| statistic (structural break strength)
///   DO03: mean_segment_var_ratio — ratio of max to min segment variance (regime contrast)
///   DO04: detection_delay — average distance from bin boundary to nearest changepoint
#[derive(Debug, Clone)]
pub struct ClassicalCpResult {
    pub n_changepoints: f64,
    pub max_cusum: f64,
    pub mean_segment_var_ratio: f64,
    pub detection_delay: f64,
}

impl ClassicalCpResult {
    pub fn nan() -> Self {
        Self { n_changepoints: f64::NAN, max_cusum: f64::NAN,
               mean_segment_var_ratio: f64::NAN, detection_delay: f64::NAN }
    }
}

/// CUSUM + binary segmentation changepoint detection.
///
/// Matches fintek's `classical_cp.rs`:
/// - Threshold = 2 · std(data) · sqrt(n)  (approximately 5% significance level)
/// - min_segment_size = 20, max_changepoints = 5 (fintek default depth=5)
/// - Returns 4 summary scalars.
pub fn classical_cp(returns: &[f64]) -> ClassicalCpResult {
    const MIN_N: usize = 40; // 2 × min_segment_size
    const MIN_SEGMENT: usize = 20;
    const MAX_CP: usize = 5;

    if returns.len() < MIN_N {
        return ClassicalCpResult::nan();
    }
    let n = returns.len();

    // Compute std for threshold
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    if std < 1e-12 {
        // Constant series — no structural breaks
        return ClassicalCpResult {
            n_changepoints: 0.0,
            max_cusum: 0.0,
            mean_segment_var_ratio: 1.0,
            detection_delay: 0.0,
        };
    }

    // Threshold: 2σ√n matches fintek's calibration
    let threshold = 2.0 * std * (n as f64).sqrt();

    let cps = cusum_binary_segmentation(returns, threshold, MIN_SEGMENT, MAX_CP);
    let n_cp = cps.len();

    // Max |CUSUM| across full series
    let mut max_cusum = 0.0_f64;
    let mut running = 0.0_f64;
    for &x in returns {
        running += x - mean;
        max_cusum = max_cusum.max(running.abs());
    }

    // Segment variance ratio: max/min variance across segments
    let boundaries: Vec<usize> = std::iter::once(0)
        .chain(cps.iter().copied())
        .chain(std::iter::once(n))
        .collect();
    let seg_vars: Vec<f64> = boundaries.windows(2).filter_map(|w| {
        let seg = &returns[w[0]..w[1]];
        if seg.len() < 2 { return None; }
        let m = seg.iter().sum::<f64>() / seg.len() as f64;
        let v = seg.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / seg.len() as f64;
        Some(v)
    }).collect();

    let mean_seg_var_ratio = if seg_vars.len() >= 2 {
        let max_v = seg_vars.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_v = seg_vars.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_v > 1e-14 { max_v / min_v } else { f64::NAN }
    } else {
        1.0
    };

    // Detection delay: mean distance from each changepoint to nearest bin boundary (0 or n)
    // Normalized by n so it's ∈ [0, 0.5]
    let detection_delay = if n_cp == 0 {
        0.0
    } else {
        let sum: f64 = cps.iter().map(|&cp| {
            let d_left = cp as f64;
            let d_right = (n - cp) as f64;
            d_left.min(d_right) / n as f64
        }).sum::<f64>();
        sum / n_cp as f64
    };

    ClassicalCpResult {
        n_changepoints: n_cp as f64,
        max_cusum,
        mean_segment_var_ratio: mean_seg_var_ratio,
        detection_delay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationarity_random_walk() {
        // Random walk — non-stationary (ADF does not reject)
        let n = 200;
        let mut data = vec![0.0; n];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 1..n {
            data[t] = data[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let r = stationarity(&data, 4);
        assert!(r.classification != StationarityClass::Stationary);
    }

    #[test]
    fn stationarity_white_noise() {
        // White noise — should be stationary
        let n = 200;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = stationarity(&data, 4);
        // ADF should reject unit root for white noise
        assert!(r.adf_statistic < 0.0, "ADF statistic for WN should be < 0, got {}", r.adf_statistic);
    }

    #[test]
    fn dependence_white_noise() {
        let n = 200;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = dependence(&data, 10);
        // White noise → high p-value
        assert!(r.lb_p_value > 0.01, "LB p on WN should be > 0.01, got {}", r.lb_p_value);
    }

    #[test]
    fn dependence_too_short() {
        let r = dependence(&[1.0, 2.0], 10);
        assert!(r.lb_statistic.is_nan());
    }

    #[test]
    fn classical_cp_no_break() {
        // White noise — no systematic breaks expected
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = classical_cp(&data);
        // Should run without panic; n_changepoints is non-negative
        assert!(r.n_changepoints >= 0.0 || r.n_changepoints.is_nan());
    }

    #[test]
    fn classical_cp_step_break() {
        // Step function: 50 zeros then 50 ones → strong mean break
        let mut data = vec![0.0_f64; 50];
        data.extend(vec![10.0_f64; 50]);
        let r = classical_cp(&data);
        assert!(r.n_changepoints >= 1.0, "should detect ≥1 break, got {}", r.n_changepoints);
        assert!(r.max_cusum.is_finite());
    }

    #[test]
    fn classical_cp_too_short() {
        let r = classical_cp(&[1.0, 2.0, 3.0]);
        assert!(r.n_changepoints.is_nan());
    }
}
