//! Hazard Functions — Probability of a changepoint occurring given current run-length.
//!
//! Architecture:
//! These primitives define the "hazard" $H(r) = P(\text{changepoint at } t \mid r_{t-1} = r)$.
//! They are implemented from first principles and are fully parametric.

use crate::using::UsingBag;

/// Constant Hazard
/// $H(r) = \lambda$
///
/// The most common hazard function. A changepoint is equally likely at any step.
pub fn hazard_constant(_r: usize, using: &UsingBag) -> f64 {
    using.get_f64("hazard_lambda").unwrap_or(1.0 / 200.0)
}

/// Geometric Hazard
/// $H(r) = 1 - (1-p)^r$ (approximately)
///
/// Logic: The longer a run persists, the more likely it is to break.
pub fn hazard_geometric(r: usize, using: &UsingBag) -> f64 {
    let p = using.get_f64("hazard_p").unwrap_or(0.01);
    1.0 - (1.0 - p).powi(r as i32)
}

/// Power Law Hazard
/// $H(r) = \alpha \cdot r^\beta$
pub fn hazard_power_law(r: usize, using: &UsingBag) -> f64 {
    let alpha = using.get_f64("hazard_alpha").unwrap_or(1e-4);
    let beta = using.get_f64("hazard_beta").unwrap_or(0.5);
    (alpha * (r as f64).powf(beta)).min(1.0)
}
