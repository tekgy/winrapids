//! Lehmer mean: Σxᵢᵖ / Σxᵢᵖ⁻¹.
//!
//! A different one-parameter family from the power mean.
//! NOT the same as power_mean despite both having a p parameter.
//!
//! Special cases:
//! - p = 0 → harmonic mean (n / Σ(1/x))
//! - p = 1 → arithmetic mean (Σx / n)
//! - p = 2 → contraharmonic mean (Σx² / Σx)
//!
//! The Lehmer mean is always between min(x) and max(x).
//! For p > 1, it's ≥ arithmetic mean. For p < 1, it's ≤ arithmetic.
//!
//! # Requires
//! All values > 0 for general p (negative values cause complex numbers
//! for non-integer p).
//!
//! # Kingdom
//! A — two accumulates (Σxᵖ and Σxᵖ⁻¹), then divide.

/// Lehmer mean with parameter p. Returns NaN for empty or non-positive data.
pub fn lehmer_mean(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if p.is_nan() { return f64::NAN; }
    if data.iter().any(|&v| v <= 0.0 || v.is_nan()) { return f64::NAN; }

    let num: f64 = data.iter().map(|&x| x.powf(p)).sum();
    let den: f64 = data.iter().map(|&x| x.powf(p - 1.0)).sum();

    if den.abs() < 1e-300 { return f64::NAN; }
    num / den
}

/// Contraharmonic mean: Lehmer mean with p=2. Σx²/Σx.
#[inline]
pub fn mean_contraharmonic(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }
    let sum: f64 = data.iter().sum();
    if sum.abs() < 1e-300 { return f64::NAN; }
    let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
    sum_sq / sum
}

#[cfg(test)]
mod tests;
