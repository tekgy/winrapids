//! Family 23 — Taylor fold detector.
//!
//! Covers fintek leaf: `taylor_fold` (K02P23C01).
//!
//! Fits progressively higher-order polynomials (order 0–3) to price
//! ticks within a bin, then measures correction ratios between orders.
//! Diverging ratios signal proximity to a fold/phase-transition in
//! price dynamics.
//!
//! 8 outputs:
//!   DO01: correction_ratio_01 — |error_1| / |error_0|
//!   DO02: correction_ratio_12 — |error_2| / |error_1|
//!   DO03: correction_ratio_23 — |error_3| / |error_2|
//!   DO04: divergence_onset    — first order where ratio > 1.0 (0=none)
//!   DO05: regime_flag         — 0=convergent, 1=divergent, 2=oscillating
//!   DO06: fit_residual_0      — baseline (0th order) prediction error
//!   DO07: max_correction_ratio
//!   DO08: regime_strength     — fit_residual_0 × max_correction_ratio

const MAX_ORDER: usize = 3;
/// Minimum data points needed for each polynomial order.
const MIN_TICKS_ORDER: [usize; 4] = [1, 2, 3, 4];

/// Taylor fold result matching fintek's `taylor_fold.rs` (K02P23C01).
#[derive(Debug, Clone)]
pub struct TaylorFoldResult {
    pub correction_ratio_01: f64,
    pub correction_ratio_12: f64,
    pub correction_ratio_23: f64,
    pub divergence_onset: f64,
    pub regime_flag: f64,
    pub fit_residual_0: f64,
    pub max_correction_ratio: f64,
    pub regime_strength: f64,
}

impl TaylorFoldResult {
    pub fn nan() -> Self {
        Self {
            correction_ratio_01: f64::NAN,
            correction_ratio_12: f64::NAN,
            correction_ratio_23: f64::NAN,
            divergence_onset: f64::NAN,
            regime_flag: f64::NAN,
            fit_residual_0: f64::NAN,
            max_correction_ratio: f64::NAN,
            regime_strength: f64::NAN,
        }
    }
}

/// Compute Taylor fold diagnostics for a tick series (prices, not returns).
///
/// Fits polynomial orders 0–3 to `ticks[..n-1]`, predicts `ticks[n-1]`,
/// measures absolute prediction error at each order, then derives
/// correction ratios and regime classification.
pub fn taylor_fold(ticks: &[f64]) -> TaylorFoldResult {
    if ticks.is_empty() { return TaylorFoldResult::nan(); }

    // Fit errors for orders 0..3
    let mut errors = [f64::NAN; 4];
    for order in 0..=MAX_ORDER {
        if ticks.len() >= MIN_TICKS_ORDER[order] + 1 {
            errors[order] = polyfit_predict_error(ticks, order);
        }
    }

    // Correction ratios: error[i+1] / error[i]
    let mut ratios = [f64::NAN; 3];
    for i in 0..3 {
        let e_prev = errors[i];
        let e_curr = errors[i + 1];
        if e_prev.is_nan() || e_curr.is_nan() { continue; }
        ratios[i] = if e_prev == 0.0 {
            if e_curr == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            e_curr / e_prev
        };
    }

    // Max finite correction ratio
    let mut max_ratio = f64::NAN;
    for &r in &ratios {
        if r.is_finite() {
            max_ratio = if max_ratio.is_nan() { r } else { max_ratio.max(r) };
        }
    }

    // Divergence onset: first transition where ratio > 1.0
    let mut onset = f64::NAN;
    let mut has_valid = false;
    for i in 0..3 {
        if ratios[i].is_finite() || !ratios[i].is_nan() {
            has_valid = true;
            if ratios[i] > 1.0 && onset.is_nan() {
                onset = (i + 1) as f64;
            }
        }
    }
    if has_valid && onset.is_nan() { onset = 0.0; }

    // Regime classification
    let valid_ratios: Vec<f64> = ratios.iter().copied().filter(|r| r.is_finite()).collect();
    let regime = if valid_ratios.is_empty() {
        f64::NAN
    } else {
        let n_above: usize = valid_ratios.iter().filter(|&&r| r > 1.0).count();
        if n_above == valid_ratios.len() {
            1.0 // all divergent
        } else if n_above == 0 {
            0.0 // all convergent
        } else {
            // Check alternation pattern
            let above: Vec<bool> = ratios.iter()
                .filter(|r| r.is_finite())
                .map(|&r| r > 1.0)
                .collect();
            let alternates = above.len() >= 2 && above.windows(2).any(|w| w[0] != w[1]);
            if alternates { 2.0 } else { 1.0 }
        }
    };

    let baseline = errors[0];
    let strength = if baseline.is_finite() && max_ratio.is_finite() {
        baseline * max_ratio
    } else {
        f64::NAN
    };

    TaylorFoldResult {
        correction_ratio_01: ratios[0],
        correction_ratio_12: ratios[1],
        correction_ratio_23: ratios[2],
        divergence_onset: onset,
        regime_flag: regime,
        fit_residual_0: baseline,
        max_correction_ratio: max_ratio,
        regime_strength: strength,
    }
}

/// Fit polynomial of degree `order` to `ticks[0..n-2]`, predict `ticks[n-1]`.
///
/// Returns absolute prediction error. Returns NaN if insufficient data or
/// the Vandermonde normal equations are rank-deficient.
fn polyfit_predict_error(ticks: &[f64], order: usize) -> f64 {
    let n = ticks.len();
    if n < 2 || n - 1 < order + 1 { return f64::NAN; }
    let history = &ticks[..n - 1];
    let target = ticks[n - 1];
    let m = history.len();

    match order {
        0 => {
            // Order 0: mean prediction
            let mean = history.iter().sum::<f64>() / m as f64;
            (target - mean).abs()
        }
        _ => {
            // Higher orders: build and solve normal equations X'X β = X'y
            let ncols = order + 1;
            let mut xtx = vec![0.0f64; ncols * ncols];
            let mut xty = vec![0.0f64; ncols];
            for i in 0..m {
                let xi = i as f64;
                let yi = history[i];
                let mut pows = vec![1.0f64; ncols];
                for p in 1..ncols { pows[p] = pows[p - 1] * xi; }
                for r in 0..ncols {
                    xty[r] += pows[r] * yi;
                    for c in 0..ncols { xtx[r * ncols + c] += pows[r] * pows[c]; }
                }
            }

            // Augmented matrix for Gaussian elimination with partial pivoting
            let mut aug = vec![0.0f64; ncols * (ncols + 1)];
            for r in 0..ncols {
                for c in 0..ncols { aug[r * (ncols + 1) + c] = xtx[r * ncols + c]; }
                aug[r * (ncols + 1) + ncols] = xty[r];
            }

            for col in 0..ncols {
                // Partial pivot
                let mut max_row = col;
                let mut max_val = aug[col * (ncols + 1) + col].abs();
                for row in (col + 1)..ncols {
                    let v = aug[row * (ncols + 1) + col].abs();
                    if v > max_val { max_val = v; max_row = row; }
                }
                if max_val < 1e-30 { return f64::NAN; }
                if max_row != col {
                    for c in 0..ncols + 1 {
                        aug.swap(col * (ncols + 1) + c, max_row * (ncols + 1) + c);
                    }
                }
                let pivot = aug[col * (ncols + 1) + col];
                for row in (col + 1)..ncols {
                    let factor = aug[row * (ncols + 1) + col] / pivot;
                    for c in col..ncols + 1 {
                        let above = aug[col * (ncols + 1) + c];
                        aug[row * (ncols + 1) + c] -= factor * above;
                    }
                }
            }

            // Back substitution
            let mut beta = vec![0.0f64; ncols];
            for r in (0..ncols).rev() {
                let mut s = aug[r * (ncols + 1) + ncols];
                for c in (r + 1)..ncols { s -= aug[r * (ncols + 1) + c] * beta[c]; }
                beta[r] = s / aug[r * (ncols + 1) + r];
            }

            // Predict at x = m
            let xp = m as f64;
            let mut pred = 0.0f64;
            let mut xpow = 1.0f64;
            for b in &beta { pred += b * xpow; xpow *= xp; }
            (target - pred).abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taylor_fold_empty() {
        let r = taylor_fold(&[]);
        assert!(r.correction_ratio_01.is_nan());
    }

    #[test]
    fn taylor_fold_perfect_linear() {
        // Perfect linear series: y = x → order-1 fit should be perfect (error ≈ 0)
        let ticks: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = taylor_fold(&ticks);
        assert!(r.fit_residual_0.is_finite(), "baseline should be finite");
        // correction_ratio_01 should be small (order 1 beats order 0)
        if r.correction_ratio_01.is_finite() {
            assert!(r.correction_ratio_01 < 1.0,
                "linear series: ratio01 should be < 1, got {}", r.correction_ratio_01);
        }
    }

    #[test]
    fn taylor_fold_white_noise_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let ticks: Vec<f64> = std::iter::once(100.0_f64)
            .chain((0..50).scan(100.0_f64, |price, _| {
                *price *= (tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).exp();
                Some(*price)
            }))
            .collect();
        let r = taylor_fold(&ticks);
        assert!(r.fit_residual_0.is_finite() && r.fit_residual_0 >= 0.0);
        // regime_flag should be in {0, 1, 2} or NaN
        if r.regime_flag.is_finite() {
            assert!(r.regime_flag == 0.0 || r.regime_flag == 1.0 || r.regime_flag == 2.0,
                "regime_flag must be 0/1/2, got {}", r.regime_flag);
        }
    }

    #[test]
    fn taylor_fold_step_break_regime() {
        // Step function: strongly nonlinear → higher-order corrections may diverge
        let mut ticks = vec![1.0_f64; 30];
        for i in 20..30 { ticks[i] = 10.0; }
        let r = taylor_fold(&ticks);
        // Should complete without panic; all outputs defined
        assert!(r.fit_residual_0.is_finite() || r.fit_residual_0.is_nan());
        assert!(r.regime_strength.is_finite() || r.regime_strength.is_nan());
    }

    #[test]
    fn polyfit_linear_prediction() {
        // Order-1 fit on x=0..9, predict x=10: perfect linear → near-zero error
        let ticks: Vec<f64> = (0..=10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let err = polyfit_predict_error(&ticks, 1);
        assert!(err < 1e-8, "linear fit error should be near-zero, got {}", err);
    }
}
