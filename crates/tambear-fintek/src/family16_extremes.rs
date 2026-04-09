//! Family 16 — Extremes / Heavy Tails.
//!
//! Covers fintek leaves: `heavy_tail` (Hill estimator).
//! NOT covered: `seismic` (requires Omori p + Bath ratio — compose later).

use tambear::volatility::{hill_estimator, hill_tail_alpha};

/// Heavy tail features for a single bin of returns.
///
/// Fintek's `heavy_tail.rs` outputs:
/// - `hill_xi`: Hill estimator ξ̂ = 1/α
/// - `hill_alpha`: tail index α
/// - `k_used`: number of order statistics used (tail depth)
#[derive(Debug, Clone)]
pub struct HeavyTailResult {
    pub hill_xi: f64,
    pub hill_alpha: f64,
    pub k_used: usize,
}

impl HeavyTailResult {
    pub fn nan() -> Self {
        Self { hill_xi: f64::NAN, hill_alpha: f64::NAN, k_used: 0 }
    }
}

/// Hill tail estimator with automatic k selection.
///
/// `data`: bin-level returns.
/// `k_fraction`: fraction of the tail to use (default 0.05-0.10).
/// Uses k = max(5, ⌊n · k_fraction⌋).
pub fn heavy_tail(data: &[f64], k_fraction: f64) -> HeavyTailResult {
    let n = data.len();
    if n < 20 || k_fraction <= 0.0 || k_fraction >= 1.0 {
        return HeavyTailResult::nan();
    }
    let k = ((n as f64 * k_fraction) as usize).max(5).min(n - 1);
    let xi = hill_estimator(data, k);
    let alpha = hill_tail_alpha(data, k);
    HeavyTailResult { hill_xi: xi, hill_alpha: alpha, k_used: k }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heavy_tail_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let n = 500;
        let alpha_true: f64 = 3.0;
        let data: Vec<f64> = (0..n).map(|_| {
            let u: f64 = tambear::rng::TamRng::next_f64(&mut rng).max(1e-300);
            u.powf(-1.0 / alpha_true)
        }).collect();
        let r = heavy_tail(&data, 0.05);
        assert!(r.hill_alpha.is_finite());
        assert!((r.hill_alpha - alpha_true).abs() < 1.5,
            "Hill α should recover true α: est={}, true={}", r.hill_alpha, alpha_true);
    }

    #[test]
    fn heavy_tail_too_short() {
        let r = heavy_tail(&[1.0, 2.0, 3.0], 0.1);
        assert!(r.hill_alpha.is_nan());
    }

    #[test]
    fn heavy_tail_bad_fraction() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(heavy_tail(&data, 0.0).hill_alpha.is_nan());
        assert!(heavy_tail(&data, 1.5).hill_alpha.is_nan());
    }
}
