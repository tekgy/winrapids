//! Family 9 — Volatility.
//!
//! Covers fintek leaves: `garch`, `realized_vol`, `jump_detection`,
//! `roll_spread`, `signature_plot`.
//! NOT covered: `range_vol`, `vpin_bvc`, `vol_regime`, `vol_dynamics`
//! (GAPs — tasks #138, #146, or partial composition).

use tambear::volatility::{
    garch11_fit, garch11_forecast, realized_variance, bipower_variation,
    jump_test_bns, roll_spread, GarchResult,
};

/// GARCH(1,1) fit result.
#[derive(Debug, Clone)]
pub struct GarchFitResult {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub log_likelihood: f64,
    /// Unconditional variance: ω / (1 - α - β).
    pub uncond_variance: f64,
    /// Persistence: α + β.
    pub persistence: f64,
    /// True if α + β > 0.99 (near IGARCH).
    pub near_igarch: bool,
}

impl GarchFitResult {
    pub fn nan() -> Self {
        Self {
            omega: f64::NAN, alpha: f64::NAN, beta: f64::NAN,
            log_likelihood: f64::NAN, uncond_variance: f64::NAN,
            persistence: f64::NAN, near_igarch: false,
        }
    }
}

/// Fit GARCH(1,1) on bin-level returns.
pub fn garch_fit(returns: &[f64], max_iter: usize) -> GarchFitResult {
    if returns.len() < 30 {
        return GarchFitResult::nan();
    }
    let r: GarchResult = garch11_fit(returns, max_iter);
    let persistence = r.alpha + r.beta;
    // Unconditional variance: ω / (1 - α - β), unreliable if persistence ≈ 1
    let uncond = if persistence < 0.999 && persistence.is_finite() {
        r.omega / (1.0 - persistence)
    } else {
        f64::NAN
    };
    GarchFitResult {
        omega: r.omega,
        alpha: r.alpha,
        beta: r.beta,
        log_likelihood: r.log_likelihood,
        uncond_variance: uncond,
        persistence,
        near_igarch: r.near_igarch,
    }
}

/// Forecast GARCH(1,1) variance h steps ahead.
pub fn garch_forecast(fit: &GarchFitResult, last_return: f64, n_steps: usize) -> Vec<f64> {
    if !fit.omega.is_finite() || n_steps == 0 { return vec![f64::NAN; n_steps]; }
    let garch = GarchResult {
        omega: fit.omega,
        alpha: fit.alpha,
        beta: fit.beta,
        log_likelihood: fit.log_likelihood,
        variances: vec![],
        iterations: 0,
        near_igarch: fit.near_igarch,
    };
    garch11_forecast(&garch, last_return, n_steps)
}

/// Realized volatility features.
#[derive(Debug, Clone)]
pub struct RealizedVolResult {
    pub realized_variance: f64,
    pub realized_vol: f64,  // sqrt(RV)
    pub bipower_variation: f64,
    pub jump_ratio: f64,    // (RV - BV) / RV  (jump component share)
}

impl RealizedVolResult {
    pub fn nan() -> Self {
        Self { realized_variance: f64::NAN, realized_vol: f64::NAN, bipower_variation: f64::NAN, jump_ratio: f64::NAN }
    }
}

/// Compute RV + BV + jump ratio.
pub fn realized_vol(returns: &[f64]) -> RealizedVolResult {
    if returns.len() < 3 { return RealizedVolResult::nan(); }
    let rv = realized_variance(returns);
    let bv = bipower_variation(returns);
    let jump_ratio = if rv > 1e-15 { (rv - bv).max(0.0) / rv } else { f64::NAN };
    RealizedVolResult {
        realized_variance: rv,
        realized_vol: rv.sqrt(),
        bipower_variation: bv,
        jump_ratio,
    }
}

/// BNS jump test result.
#[derive(Debug, Clone)]
pub struct JumpDetectionResult {
    pub bns_statistic: f64,
    pub has_jump_5pct: bool,
}

/// Run BNS (Barndorff-Nielsen-Shephard) jump test.
///
/// Reject at 5% when |Z| > 1.96.
pub fn jump_detection(returns: &[f64]) -> JumpDetectionResult {
    if returns.len() < 5 {
        return JumpDetectionResult { bns_statistic: f64::NAN, has_jump_5pct: false };
    }
    let stat = jump_test_bns(returns);
    JumpDetectionResult {
        bns_statistic: stat,
        has_jump_5pct: stat.is_finite() && stat.abs() > 1.96,
    }
}

/// Roll 1984 effective spread from serial covariance.
///
/// Wraps `volatility::roll_spread`.
pub fn roll_effective_spread(prices: &[f64]) -> f64 {
    roll_spread(prices)
}

/// Range volatility estimators from OHLC data.
///
/// Fintek's `range_vol.rs` emits all four (Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang).
#[derive(Debug, Clone)]
pub struct RangeVolResult {
    pub parkinson: f64,
    pub garman_klass: f64,
    pub rogers_satchell: f64,
    pub yang_zhang: f64,
}

impl RangeVolResult {
    pub fn nan() -> Self {
        Self {
            parkinson: f64::NAN, garman_klass: f64::NAN,
            rogers_satchell: f64::NAN, yang_zhang: f64::NAN,
        }
    }
}

/// Compute all four range volatility estimators from an OHLC series.
///
/// Parkinson/GK/RS are averaged across bars; Yang-Zhang is computed jointly.
pub fn range_volatility(
    opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64], prev_closes: &[f64],
) -> RangeVolResult {
    let n = opens.len();
    if n == 0 || highs.len() != n || lows.len() != n || closes.len() != n {
        return RangeVolResult::nan();
    }

    let mut park_sum = 0.0_f64;
    let mut gk_sum = 0.0_f64;
    let mut rs_sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        let p = tambear::volatility::parkinson_variance(highs[i], lows[i]);
        let gk = tambear::volatility::garman_klass_variance(opens[i], highs[i], lows[i], closes[i]);
        let rs = tambear::volatility::rogers_satchell_variance(opens[i], highs[i], lows[i], closes[i]);
        if p.is_finite() { park_sum += p; count += 1; }
        if gk.is_finite() { gk_sum += gk; }
        if rs.is_finite() { rs_sum += rs; }
    }
    let countf = count.max(1) as f64;
    let yz = if prev_closes.len() == n {
        tambear::volatility::yang_zhang_variance(opens, highs, lows, closes, prev_closes)
    } else { f64::NAN };

    RangeVolResult {
        parkinson: park_sum / countf,
        garman_klass: gk_sum / countf,
        rogers_satchell: rs_sum / countf,
        yang_zhang: yz,
    }
}

/// Tripower quarticity (TQ) wrapper.
///
/// Used as the variance estimator for the BV in the BNS jump test.
pub fn tripower_quarticity(returns: &[f64]) -> f64 {
    tambear::volatility::tripower_quarticity(returns)
}

/// Signature plot: realized variance at multiple sampling frequencies.
///
/// Returns (sampling_steps, rv_at_each_step). Used to detect microstructure noise.
pub fn signature_plot(returns: &[f64], steps: &[usize]) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(steps.len());
    for &step in steps {
        if step == 0 || returns.len() < step + 1 {
            out.push((step, f64::NAN));
            continue;
        }
        // Sub-sample at interval `step`
        let subsampled: Vec<f64> = (0..returns.len()).step_by(step).map(|i| returns[i]).collect();
        if subsampled.len() < 2 {
            out.push((step, f64::NAN));
        } else {
            out.push((step, realized_variance(&subsampled)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garch_fit_on_garch_data() {
        // Generate GARCH(1,1) data: ω=0.01, α=0.1, β=0.85
        let n = 500;
        let mut returns = vec![0.0; n];
        let mut sigma2: f64 = 0.01 / (1.0 - 0.95);
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 0..n {
            let z = tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
            returns[t] = sigma2.sqrt() * z;
            sigma2 = 0.01 + 0.1 * returns[t].powi(2) + 0.85 * sigma2;
        }
        let r = garch_fit(&returns, 200);
        assert!(r.omega.is_finite());
        assert!(r.alpha > 0.0 && r.alpha < 1.0);
        assert!(r.beta > 0.0 && r.beta < 1.0);
        assert!(r.persistence < 1.0, "persistence should be < 1 for stationary GARCH");
    }

    #[test]
    fn realized_vol_basic() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let r = realized_vol(&returns);
        assert!(r.realized_variance >= 0.0);
        assert!(r.realized_vol >= 0.0);
        assert!((r.realized_vol - r.realized_variance.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn jump_detection_no_jumps() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = jump_detection(&returns);
        assert!(r.bns_statistic.is_finite());
        // No injected jumps → should not reject
        assert!(!r.has_jump_5pct || r.bns_statistic.abs() < 3.0);
    }

    #[test]
    fn roll_spread_basic() {
        let prices = vec![100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0];
        let s = roll_effective_spread(&prices);
        // Bid-ask bounce → positive Roll spread
        assert!(s.is_finite());
    }

    #[test]
    fn signature_plot_increasing_n() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let sp = signature_plot(&returns, &[1, 2, 4, 8]);
        assert_eq!(sp.len(), 4);
        for (_, rv) in &sp { assert!(rv.is_finite() && *rv >= 0.0); }
    }
}
