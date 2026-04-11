//! Exponential moving average (EMA): s_t = α·x_t + (1-α)·s_{t-1}.
//!
//! A recurrence that looks sequential but IS Kingdom A — the map
//! (a, b) → (a·x_t, a·b + α·x_t) is an affine map that composes
//! associatively: (a₁,b₁) ∘ (a₂,b₂) = (a₁·a₂, a₁·b₂ + b₁).
//!
//! This means EMA admits a parallel prefix scan. The sequential
//! implementation here is O(n); the GPU version (via affine_prefix_scan)
//! would be O(log n).
//!
//! # Parameters
//! - `alpha`: smoothing factor in (0, 1). Higher = more weight on recent.
//!   Common conventions: alpha = 2/(period+1) for period-based EMA.
//!
//! # Kingdom
//! A — affine prefix scan over (1-α, α·x_t) maps.

/// Exponential moving average. Returns the full EMA series.
///
/// First value is initialized to x[0] (SMA-1 initialization).
pub fn mean_exponential_moving(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    if alpha <= 0.0 || alpha >= 1.0 || alpha.is_nan() { return vec![f64::NAN; data.len()]; }

    let mut result = Vec::with_capacity(data.len());
    let mut ema = data[0];
    result.push(ema);

    let one_minus_alpha = 1.0 - alpha;
    for &x in &data[1..] {
        ema = alpha * x + one_minus_alpha * ema;
        result.push(ema);
    }
    result
}

/// EMA from period: alpha = 2 / (period + 1).
pub fn mean_exponential_moving_period(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 { return vec![f64::NAN; data.len()]; }
    let alpha = 2.0 / (period as f64 + 1.0);
    mean_exponential_moving(data, alpha)
}

#[cfg(test)]
mod tests;
