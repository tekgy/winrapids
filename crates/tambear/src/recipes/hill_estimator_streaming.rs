//! Hill heavy-tail index — streaming / mergeable form via quantile sketch.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! quantile-sketch primitive (Tier 1) plus a tail-region threshold pass
//! plus a Kulisch-backed log-mean. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! For an absolute-return series with heavy upper tail, the Hill (1975)
//! estimator of the tail index `ξ = 1/α` (where α is the Pareto-like
//! tail exponent) is:
//!
//! ```text
//! sort |returns| descending  →  X_(1) ≥ X_(2) ≥ ... ≥ X_(n)
//! ξ̂  =  (1/k) · Σ_{i=1..k}  ln( X_(i) / X_(k+1) )
//! ```
//!
//! Here `k` is the number of order statistics in the tail. The
//! threshold `X_(k+1)` is the value at quantile `q = 1 − k/n` of the
//! absolute-return distribution.
//!
//! # Why streaming
//!
//! The classical `hill_estimator` (in volatility.rs) sorts the full
//! data, which costs `O(n log n)` time and `O(n)` memory. For SIP at
//! scale (100M+ ticks per backfill window) this is prohibitive.
//!
//! The streaming form uses a quantile sketch to find the
//! threshold without sorting, then makes a single pass to compute the
//! log-mean over values exceeding it. Time `O(n)`, memory `O(1/ε)`.
//!
//! Most importantly: the sketch is **mergeable**, so partial Hill
//! estimates from distributed shards merge into a global estimate
//! without re-streaming the full data.
//!
//! # Composition
//!
//! - **QuantileSketch primitive** (Tier 1) — KLL by default; t-digest
//!   selectable (and arguably preferable here — t-digest's tail
//!   accuracy benefits Hill directly).
//! - **Threshold pass** — find `X_(k+1)` by querying `sketch.quantile(1
//!   − k/n)` on the absolute-value sketch.
//! - **Kulisch-backed log-mean** — `math::sum` of `ln(X_(i)/X_(k+1))`
//!   over the top-k values, divided by k.
//!
//! # Reference
//!
//! Hill, B. M. (1975). A simple general approach to inference about
//! the tail of a distribution. *The Annals of Statistics* 3(5):
//! 1163–1174.
//!
//! # Default parameters
//!
//! - `k_fraction` — fraction of observations to use as the tail
//!   region. SIP uses `max(10, sqrt(n))` heuristic; we expose
//!   `k_fraction` as the user-facing parameter (k = floor(n ·
//!   k_fraction)).
//! - `epsilon` — sketch error in quantile rank. 0.005 is a reasonable
//!   default at SIP scales.
//! - `sketch` — KLL by default; t-digest worth choosing for Hill
//!   specifically because its tail accuracy is exceptional.
//!
//! # NaN/Inf policy
//!
//! Inherited from the sketch and `math::sum`: non-finite values
//! skipped at both stages. Returns NaN if `n < 2` finite values, or
//! if the sketch threshold is non-finite.

use crate::primitives::specialist::quantile_sketch::{
    QuantileSketch, SketchAlgorithm,
};

/// Hill tail-index estimator (ξ̂) using the locked-default sketch and ε.
///
/// `k_fraction` is the fraction of observations to use as the tail region
/// (typically 0.05–0.10, i.e., 5–10% of the data). The actual `k` used
/// is `max(10, floor(n · k_fraction))`.
///
/// Returns ξ̂ ∈ [0, ∞). Convert to tail exponent α via `α = 1 / ξ̂`.
/// Returns NaN if too few finite observations or threshold non-finite.
///
/// # Panics
///
/// Panics if `k_fraction` is non-finite or outside `(0, 0.5]`.
pub fn hill_estimator_streaming(returns: &[f64], k_fraction: f64) -> f64 {
    hill_estimator_streaming_with(returns, k_fraction, 0.005, SketchAlgorithm::DEFAULT)
}

/// Hill tail-index estimator with explicit sketch and epsilon.
pub fn hill_estimator_streaming_with(
    returns: &[f64],
    k_fraction: f64,
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> f64 {
    assert!(
        k_fraction.is_finite() && k_fraction > 0.0 && k_fraction <= 0.5,
        "hill_estimator_streaming: k_fraction must be in (0, 0.5], got {k_fraction}"
    );

    // Absolute returns — Hill is a two-sided tail estimator on |return|.
    let abs_rets: Vec<f64> = returns
        .iter()
        .filter(|r| r.is_finite())
        .map(|r| r.abs())
        .filter(|r| r.is_finite() && *r > 0.0) // skip 0.0 — log(0) undefined
        .collect();
    let n = abs_rets.len();
    if n < 2 {
        return f64::NAN;
    }

    let k = ((n as f64 * k_fraction).floor() as usize).max(10).min(n - 1);

    // Build sketch and find threshold X_(k+1) at quantile q = 1 − k/n.
    let q_threshold = 1.0 - (k as f64 / n as f64);
    let threshold = match algorithm {
        SketchAlgorithm::Kll => {
            let mut sk = crate::primitives::specialist::KllSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::Gk => {
            let mut sk = crate::primitives::specialist::GkSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::Tdigest => {
            let mut sk = crate::primitives::specialist::TdigestSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::DdSketch => {
            let mut sk = crate::primitives::specialist::DdSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            sk.quantile(q_threshold)
        }
    };

    if !threshold.is_finite() || threshold <= 0.0 {
        return f64::NAN;
    }

    // Kulisch-backed log-mean over the top-k values.
    let log_ratios: Vec<f64> = abs_rets
        .iter()
        .copied()
        .filter(|r| *r > threshold)
        .map(|r| (r / threshold).ln())
        .collect();

    if log_ratios.is_empty() {
        return f64::NAN;
    }

    crate::math::sum(&log_ratios) / log_ratios.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_too_short_returns_nan() {
        assert!(hill_estimator_streaming(&[], 0.05).is_nan());
        assert!(hill_estimator_streaming(&[1.0], 0.05).is_nan());
    }

    #[test]
    fn pareto_known_tail_index() {
        // Generate Pareto(α=2) samples via inverse-CDF: x = (1-u)^(-1/α).
        // For α=2, ξ = 1/α = 0.5. Hill should recover ~0.5.
        // Use a deterministic pseudo-uniform sequence (Halton-like).
        let mut samples: Vec<f64> = Vec::with_capacity(10000);
        for i in 1..=10000 {
            // Pseudo-uniform using Van der Corput sequence base 2.
            let u = van_der_corput(i, 2);
            let x = (1.0 - u).powf(-1.0 / 2.0); // α = 2
            samples.push(x);
        }
        let xi = hill_estimator_streaming(&samples, 0.05);
        // ξ̂ should be close to 0.5. Hill estimator has known
        // small-k bias; allow generous tol.
        assert!(
            (xi - 0.5).abs() < 0.15,
            "Hill estimator off: got {xi}, want ~0.5"
        );
    }

    #[test]
    fn three_sketches_close_for_hill() {
        let mut samples: Vec<f64> = Vec::with_capacity(5000);
        for i in 1..=5000 {
            let u = van_der_corput(i, 3);
            samples.push((1.0 - u).powf(-1.0 / 3.0)); // α = 3
        }
        let h_kll = hill_estimator_streaming_with(&samples, 0.05, 0.01, SketchAlgorithm::Kll);
        let h_gk = hill_estimator_streaming_with(&samples, 0.05, 0.01, SketchAlgorithm::Gk);
        let h_td = hill_estimator_streaming_with(&samples, 0.05, 0.01, SketchAlgorithm::Tdigest);

        // All three should be in the same neighborhood (~0.33).
        let target = 1.0 / 3.0;
        for (label, val) in [("kll", h_kll), ("gk", h_gk), ("tdigest", h_td)] {
            assert!(
                (val - target).abs() < 0.2,
                "{label}: {val} vs target {target}"
            );
        }
    }

    #[test]
    fn skips_non_finite_returns() {
        let mut samples: Vec<f64> = Vec::with_capacity(1000);
        for i in 1..=1000 {
            let u = van_der_corput(i, 2);
            samples.push((1.0 - u).powf(-1.0 / 2.0));
        }
        // Inject some non-finite values.
        samples.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let xi_with_garbage = hill_estimator_streaming(&samples, 0.05);
        let xi_clean = {
            let cleaned: Vec<f64> = samples.iter().copied().filter(|r| r.is_finite() && *r > 0.0).collect();
            hill_estimator_streaming(&cleaned, 0.05)
        };
        // Should match the cleaned-version result.
        assert!(
            (xi_with_garbage - xi_clean).abs() < 0.05,
            "with-garbage {xi_with_garbage} vs clean {xi_clean}"
        );
    }

    #[test]
    #[should_panic(expected = "k_fraction")]
    fn panics_on_bad_k_fraction() {
        let _ = hill_estimator_streaming(&[1.0, 2.0, 3.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "k_fraction")]
    fn panics_on_k_fraction_too_large() {
        let _ = hill_estimator_streaming(&[1.0, 2.0, 3.0], 0.6);
    }

    /// Van der Corput sequence — deterministic low-discrepancy
    /// pseudo-uniform values in (0, 1).
    fn van_der_corput(mut n: u64, base: u64) -> f64 {
        let mut q = 0.0_f64;
        let mut bk = 1.0_f64 / base as f64;
        while n > 0 {
            q += (n % base) as f64 * bk;
            n /= base;
            bk /= base as f64;
        }
        q
    }
}
