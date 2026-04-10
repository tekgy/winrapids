//! Scoring and Thresholding Primitives for Changepoint Detection.
//!
//! These primitives are used at the end of a changepoint detection pipeline
//! to translate a run-length posterior distribution into discrete changepoint indices.

use crate::using::UsingBag;

/// Scoring: Maximum Posterior Drop
/// S(t) = max_p(t-1) - max_p(t)
///
/// Identifies changepoints where the confidence in the current run length collapses.
pub fn score_max_posterior_drop(max_posts: &[f64], t: usize) -> f64 {
    if t == 0 || t >= max_posts.len() {
        return 0.0;
    }
    (max_posts[t - 1] - max_posts[t]).max(0.0)
}

/// Scoring: Run-Length Zero Mass
/// S(t) = P(r_t = 0 | data)
///
/// Identifies changepoints by the probability that the run length has reset to zero.
pub fn score_rl0_mass(run_post: &[f64], t: usize) -> f64 {
    // This assumes the run_post passed is the posterior at time t.
    // If the slice is the posterior for time t, we just take the 0-th element.
    // The calling composition must handle the indexing.
    0.0 // Placeholder: in practice, the caller passes the specific posterior vector
}

/// Threshold: Fixed
/// Returns the fixed value provided in the using bag.
pub fn threshold_fixed(using: &UsingBag) -> f64 {
    using.get_f64("threshold").unwrap_or(0.5)
}

/// Threshold: Adaptive
/// Thresh = median(drops) + k * std(drops)
///
/// Prevents spurious detections in noisy data by scaling the threshold to the signal.
pub fn threshold_adaptive(drops: &[f64], using: &UsingBag) -> f64 {
    if drops.is_empty() {
        return 0.5;
    }

    let k = using.get_f64("threshold_k").unwrap_or(2.0);

    let mut sorted = drops.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = sorted[sorted.len() / 2];
    let mean = drops.iter().sum::<f64>() / drops.len() as f64;
    let var = drops.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / drops.len() as f64;

    (median + k * var.sqrt()).max(0.3)
}
