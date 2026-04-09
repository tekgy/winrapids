//! Family 10 — Nonlinear Dynamics / Chaos.
//!
//! Covers fintek leaves: `sample_entropy`, `permutation_entropy`, `hurst_rs`,
//! `dfa`, `correlation_dim`, `lyapunov`, `poincare`.
//! NOT covered: `lz_complexity`, `rqa`, `embedding`, `mfdfa` (partial GAPs).

use tambear::complexity as cx;

/// Sample entropy at (m, r) = (embedding dim, tolerance).
pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    if data.len() < m + 2 { return f64::NAN; }
    cx::sample_entropy(data, m, r)
}

/// Permutation entropy at (m, tau).
pub fn permutation_entropy(data: &[f64], m: usize, tau: usize) -> f64 {
    if data.len() < m * tau + 1 { return f64::NAN; }
    cx::permutation_entropy(data, m, tau)
}

/// R/S Hurst exponent (Mandelbrot-Wallis).
pub fn hurst_rs(data: &[f64]) -> f64 {
    if data.len() < 20 { return f64::NAN; }
    cx::hurst_rs(data)
}

/// Detrended Fluctuation Analysis α exponent.
pub fn dfa(data: &[f64], min_box: usize, max_box: usize) -> f64 {
    if data.len() < max_box + 1 { return f64::NAN; }
    cx::dfa(data, min_box, max_box)
}

/// Grassberger-Procaccia correlation dimension at embedding (m, tau).
pub fn correlation_dimension(data: &[f64], m: usize, tau: usize) -> f64 {
    if data.len() < m * tau + 10 { return f64::NAN; }
    cx::correlation_dimension(data, m, tau)
}

/// Rosenstein largest Lyapunov exponent.
pub fn largest_lyapunov(data: &[f64], m: usize, tau: usize, dt: f64) -> f64 {
    if data.len() < m * tau + 20 { return f64::NAN; }
    cx::largest_lyapunov(data, m, tau, dt)
}

/// Poincaré plot SD1/SD2 from lag-1 return pairs.
///
/// Fintek's `poincare.rs`:
/// - SD1 = √((1/2) · Var(x_{t+1} - x_t))
/// - SD2 = √(2·Var(x) - SD1²)
#[derive(Debug, Clone)]
pub struct PoincareResult {
    pub sd1: f64,
    pub sd2: f64,
    pub ratio: f64, // SD1/SD2
}

impl PoincareResult {
    pub fn nan() -> Self { Self { sd1: f64::NAN, sd2: f64::NAN, ratio: f64::NAN } }
}

pub fn poincare(data: &[f64]) -> PoincareResult {
    let n = data.len();
    if n < 4 { return PoincareResult::nan(); }

    // Variance of x (sample)
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var_x: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Variance of (x_{t+1} - x_t)
    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let dmean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var_d: f64 = diffs.iter().map(|d| (d - dmean).powi(2)).sum::<f64>() / (diffs.len() - 1).max(1) as f64;

    let sd1_sq = 0.5 * var_d;
    let sd2_sq = 2.0 * var_x - sd1_sq;
    let sd1 = sd1_sq.max(0.0).sqrt();
    let sd2 = sd2_sq.max(0.0).sqrt();
    let ratio = if sd2 > 1e-15 { sd1 / sd2 } else { f64::NAN };

    PoincareResult { sd1, sd2, ratio }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_entropy_constant() {
        let se = sample_entropy(&[5.0; 100], 2, 0.2);
        // Constant data should have sample entropy near 0 (fully predictable)
        assert!(!se.is_nan());
    }

    #[test]
    fn permutation_entropy_random() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let pe = permutation_entropy(&data, 3, 1);
        assert!(pe.is_finite());
        // Random data has all orderings ~equally likely → high PE
        assert!(pe > 1.0);
    }

    #[test]
    fn hurst_rs_too_short() {
        assert!(hurst_rs(&[1.0, 2.0]).is_nan());
    }

    #[test]
    fn dfa_white_noise() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..500).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let alpha = dfa(&data, 4, 50);
        // White noise DFA α ≈ 0.5
        assert!(alpha > 0.2 && alpha < 0.8, "WN DFA α should be near 0.5, got {}", alpha);
    }

    #[test]
    fn poincare_basic() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = poincare(&data);
        assert!(r.sd1 >= 0.0 && r.sd2 >= 0.0);
        assert!(r.ratio.is_finite());
    }

    #[test]
    fn poincare_too_short() {
        assert!(poincare(&[1.0, 2.0]).sd1.is_nan());
    }
}
