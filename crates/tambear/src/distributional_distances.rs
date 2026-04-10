//! Distributional Distance Suite
//!
//! This module implements a set of primitives for measuring the distance
//! between probability distributions.
//!
//! Architecture:
//! - Kingdom A (Parallel): Point-wise reductions (TVD, Hellinger, KL, JS)
//! - Kingdom A (Sorted): Distance based on CDFs (KS, Wasserstein-1)
//! - Kingdom C (Iterative): Optimal transport (Wasserstein-p)
//!
//! All implementations follow the Tambear Contract: first-principles,
//! no vendor libraries, and fully parametric.

use crate::using::UsingBag;

/// Total Variation Distance (TVD)
/// TV(P, Q) = 0.5 * Σ |p_i - q_i|
///
/// Kingdom A: Point-wise reduction (sum of abs differences).
pub fn total_variation_distance(p: &[f64], q: &[f64], _using: &UsingBag) -> f64 {
    assert_eq!(p.len(), q.len(), "TVD: distributions must have same length");
    0.5 * p
        .iter()
        .zip(q)
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
}

/// Hellinger Distance
/// H^2(P, Q) = 1 - Σ √(p_i * q_i)
///
/// Kingdom A: Point-wise reduction (sum of root products).
pub fn hellinger_distance(p: &[f64], q: &[f64], _using: &UsingBag) -> f64 {
    assert_eq!(
        p.len(),
        q.len(),
        "Hellinger: distributions must have same length"
    );
    let sum_sqrt: f64 = p.iter().zip(q).map(|(&pi, &qi)| (pi * qi).sqrt()).sum();
    (1.0 - sum_sqrt).sqrt()
}

/// Kullback-Leibler Divergence (KL)
/// D_KL(P || Q) = Σ p_i log(p_i/q_i)
///
/// Kingdom A: Point-wise reduction.
pub fn kl_divergence(p: &[f64], q: &[f64], using: &UsingBag) -> f64 {
    assert_eq!(p.len(), q.len(), "KL: distributions must have same length");
    let epsilon = using.get_f64("epsilon").unwrap_or(1e-12);

    p.iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else if qi <= 0.0 {
                // Use epsilon smoothing if provided, otherwise infinity
                if epsilon > 0.0 {
                    pi * (pi / epsilon).ln()
                } else {
                    f64::INFINITY
                }
            } else {
                pi * (pi / qi).ln()
            }
        })
        .sum()
}

/// Jensen-Shannon Divergence (JS)
/// D_JS(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = (P + Q) / 2
///
/// Kingdom A: Composition of KL.
pub fn js_divergence(p: &[f64], q: &[f64], using: &UsingBag) -> f64 {
    assert_eq!(p.len(), q.len(), "JS: distributions must have same length");
    let m: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    0.5 * kl_divergence(p, &m, using) + 0.5 * kl_divergence(q, &m, using)
}
