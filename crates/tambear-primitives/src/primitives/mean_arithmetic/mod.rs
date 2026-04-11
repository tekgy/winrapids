//! Arithmetic mean: Σxᵢ / n.
//!
//! The simplest central tendency measure. No pow, no exp, no log —
//! just sum and divide.
//!
//! # Recipe (accumulate+gather decomposition)
//!
//! ```text
//! Accumulate(All, Value, Add) → "sum"
//! Accumulate(All, One,   Add) → "count"
//! Gather("sum / count")       → "mean"
//! ```
//!
//! Shares with: variance (sum + count), skewness (sum + count),
//! kurtosis (sum + count), and every other primitive that sums values.
//!
//! # Kingdom
//! A — two accumulates + one gather. All fuse into one data pass.

use crate::recipe::*;

/// The recipe for arithmetic mean. TAM compiles this.
pub const RECIPE: Recipe = MEAN_ARITHMETIC;

/// Arithmetic mean of a slice. Returns NaN for empty or NaN input.
///
/// This is the CPU reference implementation. TAM uses the RECIPE
/// above to compute the same result on any ALU.
#[inline]
pub fn mean_arithmetic(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }
    let n = data.len() as f64;
    data.iter().sum::<f64>() / n
}

/// Online/streaming arithmetic mean via Welford's method.
/// Feed one value at a time, query the mean at any point.
///
/// The Welford accumulator IS the parallel merge primitive for
/// this recipe — MeanAccumulator::merge is the associative combine
/// that makes distributed/parallel mean computation exact.
#[derive(Debug, Clone)]
pub struct MeanAccumulator {
    pub count: usize,
    pub mean: f64,
}

impl MeanAccumulator {
    pub fn new() -> Self { Self { count: 0, mean: 0.0 } }

    /// Add one observation.
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
    }

    /// Merge two accumulators (for parallel reduce).
    /// This is the associative combine operation.
    pub fn merge(&self, other: &Self) -> Self {
        let total = self.count + other.count;
        if total == 0 { return Self::new(); }
        Self {
            count: total,
            mean: (self.mean * self.count as f64 + other.mean * other.count as f64) / total as f64,
        }
    }

    pub fn value(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.mean }
    }
}

#[cfg(test)]
mod tests;
