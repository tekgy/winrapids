//! Softmax and log-softmax: normalized exponential.
//!
//! Composes from `log_sum_exp` — softmax(x)ᵢ = exp(xᵢ - lse(x)).
//! This is a METHOD (composition of primitives), not an atom.
//! The atom is `log_sum_exp`. This is the formula on top.

use crate::log_sum_exp::log_sum_exp;

/// Softmax: normalized exponential distribution over a vector.
///
/// `softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ) = exp(xᵢ - lse(x))`
///
/// Numerically stable via log-sum-exp subtraction.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    let lse = log_sum_exp(x);
    if lse.is_nan() { return vec![f64::NAN; x.len()]; }
    x.iter().map(|&xi| (xi - lse).exp()).collect()
}

/// Log-softmax: log of the softmax output.
///
/// `log_softmax(x)ᵢ = xᵢ - lse(x)`
///
/// More numerically stable than `softmax(x).iter().map(|p| p.ln())`.
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    let lse = log_sum_exp(x);
    if lse.is_nan() { return vec![f64::NAN; x.len()]; }
    x.iter().map(|&xi| xi - lse).collect()
}

#[cfg(test)]
mod tests;
