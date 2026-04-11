//! Log-sum-exp: numerically stable computation of log(Σ exp(xᵢ)).
//!
//! The foundational primitive for working in log-probability space.
//! Used by: HMM forward, softmax, mixture log-likelihood, Bayesian
//! model evidence, partition functions, attention mechanisms.
//!
//! # Formula
//!
//! ```text
//! lse(x₁, ..., xₙ) = max(x) + ln(Σᵢ exp(xᵢ - max(x)))
//! ```
//!
//! The max-subtraction trick prevents overflow/underflow.
//!
//! # Kingdom
//!
//! Kingdom A. The pairwise combine `lse(a, b) = max(a,b) + ln(1 + exp(-|a-b|))`
//! is associative and commutative — this IS the LogSumExp semiring's addition.

/// Numerically stable log-sum-exp of a slice.
///
/// Returns `log(Σᵢ exp(xᵢ))` without overflow or catastrophic cancellation.
///
/// # Edge cases
/// - Empty slice → `-∞` (additive identity of LogSumExp semiring)
/// - All `-∞` → `-∞`
/// - Any `NaN` → `NaN` (propagates)
/// - Single element → that element (no-op)
#[inline]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    // NaN propagation
    if values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Pairwise log-sum-exp: the LogSumExp semiring's addition.
///
/// `lse2(a, b) = log(exp(a) + exp(b))` computed stably.
/// This is the binary operation used in prefix scans over the LogSumExp semiring.
#[inline]
pub fn log_sum_exp_pair(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let (max, min) = if a >= b { (a, b) } else { (b, a) };
    max + (1.0 + (min - max).exp()).ln()
}

#[cfg(test)]
mod tests;
