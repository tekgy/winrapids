//! Family 5 — Stationarity / Structural tests.
//!
//! Covers fintek leaves: `stationarity`, `dependence`.
//! NOT covered: `struct_break`, `classical_cp`, `pelt`, `bocpd` (GAPS — tasks #143).

use tambear::time_series::{adf_test, kpss_test, ljung_box, AdfResult, KpssResult, LjungBoxResult};

/// Classification result from combined ADF + KPSS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StationarityClass {
    /// Both tests agree: stationary.
    Stationary,
    /// Both tests agree: non-stationary.
    NonStationary,
    /// ADF rejects unit root but KPSS rejects stationarity: trend-stationary after detrending.
    TrendStationary,
    /// Inconclusive.
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

    let classification = match (adf_rejects, kpss_rejects) {
        (true, false)  => StationarityClass::Stationary,
        (false, true)  => StationarityClass::NonStationary,
        (true, true)   => StationarityClass::TrendStationary,
        (false, false) => StationarityClass::Inconclusive,
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
}
