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
//! # Hill stability — report three k values
//!
//! The Hill estimator is notoriously unstable as a function of `k`:
//! different choices of the tail cutoff produce different estimates
//! even on the same data. The conventional workaround is a **Hill
//! plot** — compute ξ̂ across many k values and look for a region of
//! stability.
//!
//! SIP's per-hour signal reports three Hill estimates at `k = n/10`,
//! `k = n/20`, and `k = n/50`. If the three agree, the tail estimate
//! is robust; if they diverge, the consumer knows to treat the
//! estimate with caution.
//!
//! # Why streaming
//!
//! The classical Hill estimator sorts the full data at `O(n log n)`
//! time and `O(n)` memory. For SIP at scale (100M+ ticks per backfill
//! window) this is prohibitive.
//!
//! The streaming form uses a quantile sketch to find the threshold
//! without sorting, then makes a single pass to compute the log-mean
//! over values exceeding it. Time `O(n)`, memory `O(1/ε)`.
//!
//! Most importantly: the sketch is **mergeable**, so partial Hill
//! estimates from distributed shards merge into a global estimate
//! without re-streaming the full data.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` (v2,
//! 2026-04-22) the per-hour `heavy_tail_index` kernel is a three-
//! output multi-output:
//!
//! - `heavy_tail_index: f32`   → `[896:900)` — primary, k = n/10
//! - `hill_index_k_n20: f32`   → `[1896:1900)`
//! - `hill_index_k_n50: f32`   → `[1900:1904)`
//!
//! All three derive from the same quantile sketch built once over the
//! hour's filtered absolute returns; the recipe contributes them
//! together.
//!
//! # Reference
//!
//! Hill, B. M. (1975). A simple general approach to inference about
//! the tail of a distribution. *The Annals of Statistics* 3(5):
//! 1163–1174.
//!
//! # Composition
//!
//! - **QuantileSketch primitive** (Tier 1) — built once over the
//!   filtered absolute returns. KLL by default; t-digest selectable
//!   (and arguably preferable for Hill because its tail accuracy is
//!   exceptional).
//! - **Threshold gather** — for each `k`, query `sketch.quantile(1 −
//!   k/n)` to obtain `X_(k+1)`.
//! - **Kulisch-backed log-mean** — for each `k`, sum `ln(X_(i) /
//!   X_(k+1))` over values exceeding the threshold.
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element. Also skips `x == 0` (log undefined) and `x < 0`
//! after the absolute-value transform (i.e., never — |x| ≥ 0 always).
//!
//! - Insufficient finite positive observations (`n < 2`, or fewer
//!   than `k + 1` above the threshold) → affected Hill value is NaN.
//! - Non-finite sketch threshold → affected Hill value is NaN.
//!
//! # Default parameters
//!
//! - Single-k form — `k_fraction` caller-supplied.
//! - Three-k form — fixed at `(0.10, 0.05, 0.02)` per SIP spec; the
//!   three correspond to `k = n/10, n/20, n/50`.
//! - `epsilon` — 0.005 default sketch error. Locked-good for SIP
//!   scales.
//! - `sketch` — KLL default.

use crate::primitives::specialist::quantile_sketch::{QuantileSketch, SketchAlgorithm};

/// Single-k Hill estimator result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HillSingleResult {
    /// Hill tail-index estimate ξ̂ ∈ [0, ∞). `α = 1 / ξ̂`.
    pub hill: f64,
    /// Threshold `X_(k+1)` used for the log-mean (in return units).
    pub threshold: f64,
    /// `k` actually used (after the `max(10, ·)` floor and `min(n−1,
    /// ·)` ceiling).
    pub k_used: u64,
    /// Number of finite positive observations after filtering.
    pub n_observations: u64,
}

impl HillSingleResult {
    #[inline]
    pub fn nan() -> Self {
        Self {
            hill: f64::NAN,
            threshold: f64::NAN,
            k_used: 0,
            n_observations: 0,
        }
    }
}

/// Three-k Hill estimator result — the SIP multi-output shape.
///
/// All three Hill values share the same underlying sketch, so they are
/// computed together in one pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HillThreeKResult {
    /// Primary Hill estimate at `k_fraction = 0.10` (k = n/10).
    pub hill_k_n10: f64,
    /// Hill estimate at `k_fraction = 0.05` (k = n/20) — intermediate
    /// stability check.
    pub hill_k_n20: f64,
    /// Hill estimate at `k_fraction = 0.02` (k = n/50) — deep-tail
    /// stability check.
    pub hill_k_n50: f64,
    /// Thresholds at the three cutoffs (same order as the hill fields).
    pub threshold_k_n10: f64,
    pub threshold_k_n20: f64,
    pub threshold_k_n50: f64,
    /// `k` actually used at each cutoff.
    pub k_n10_used: u64,
    pub k_n20_used: u64,
    pub k_n50_used: u64,
    /// Number of finite positive observations after filtering.
    pub n_observations: u64,
}

impl HillThreeKResult {
    #[inline]
    pub fn nan() -> Self {
        Self {
            hill_k_n10: f64::NAN,
            hill_k_n20: f64::NAN,
            hill_k_n50: f64::NAN,
            threshold_k_n10: f64::NAN,
            threshold_k_n20: f64::NAN,
            threshold_k_n50: f64::NAN,
            k_n10_used: 0,
            k_n20_used: 0,
            k_n50_used: 0,
            n_observations: 0,
        }
    }
}

/// Hill tail-index estimator (ξ̂) at a single `k_fraction`.
///
/// See `HillSingleResult` for the returned fields; the top-level
/// `hill` field is ξ̂ ∈ [0, ∞). Convert to tail exponent α via
/// `α = 1 / hill`.
///
/// # Panics
///
/// Panics if `k_fraction` is non-finite or outside `(0, 0.5]`.
pub fn hill_estimator_streaming(returns: &[f64], k_fraction: f64) -> HillSingleResult {
    hill_estimator_streaming_with(returns, k_fraction, 0.005, SketchAlgorithm::DEFAULT)
}

/// Hill tail-index estimator with explicit sketch choice and epsilon.
pub fn hill_estimator_streaming_with(
    returns: &[f64],
    k_fraction: f64,
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> HillSingleResult {
    assert!(
        k_fraction.is_finite() && k_fraction > 0.0 && k_fraction <= 0.5,
        "hill_estimator_streaming: k_fraction must be in (0, 0.5], got {k_fraction}"
    );

    let abs_rets = filter_positive_abs(returns);
    let n = abs_rets.len();
    if n < 2 {
        return HillSingleResult::nan();
    }

    let threshold = sketch_threshold_at(&abs_rets, k_fraction, epsilon, algorithm);
    finalize_single(&abs_rets, k_fraction, threshold)
}

/// Three-k Hill estimator — SIP spec's multi-output form.
///
/// Produces Hill estimates at `k = n/10`, `n/20`, `n/50` from a single
/// sketch build. `hill_k_n10` is the primary (SIP header's
/// `heavy_tail_index`); the other two are stability checks.
pub fn hill_estimator_three_k(returns: &[f64]) -> HillThreeKResult {
    hill_estimator_three_k_with(returns, 0.005, SketchAlgorithm::DEFAULT)
}

/// Three-k Hill estimator with explicit sketch choice and epsilon.
pub fn hill_estimator_three_k_with(
    returns: &[f64],
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> HillThreeKResult {
    let abs_rets = filter_positive_abs(returns);
    let n = abs_rets.len();
    if n < 2 {
        return HillThreeKResult::nan();
    }

    // Build sketch once and query three thresholds.
    let (t_n10, t_n20, t_n50) = match algorithm {
        SketchAlgorithm::Kll => {
            let mut sk = crate::primitives::specialist::KllSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            let t_n10 = sk.quantile(1.0 - 0.10);
            let t_n20 = sk.quantile(1.0 - 0.05);
            let t_n50 = sk.quantile(1.0 - 0.02);
            (t_n10, t_n20, t_n50)
        }
        SketchAlgorithm::Gk => {
            let mut sk = crate::primitives::specialist::GkSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            (
                sk.quantile(1.0 - 0.10),
                sk.quantile(1.0 - 0.05),
                sk.quantile(1.0 - 0.02),
            )
        }
        SketchAlgorithm::Tdigest => {
            let mut sk = crate::primitives::specialist::TdigestSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            (
                sk.quantile(1.0 - 0.10),
                sk.quantile(1.0 - 0.05),
                sk.quantile(1.0 - 0.02),
            )
        }
        SketchAlgorithm::DdSketch => {
            let mut sk = crate::primitives::specialist::DdSketch::new(epsilon);
            sk.add_slice(&abs_rets);
            (
                sk.quantile(1.0 - 0.10),
                sk.quantile(1.0 - 0.05),
                sk.quantile(1.0 - 0.02),
            )
        }
    };

    let r_n10 = finalize_single(&abs_rets, 0.10, t_n10);
    let r_n20 = finalize_single(&abs_rets, 0.05, t_n20);
    let r_n50 = finalize_single(&abs_rets, 0.02, t_n50);

    HillThreeKResult {
        hill_k_n10: r_n10.hill,
        hill_k_n20: r_n20.hill,
        hill_k_n50: r_n50.hill,
        threshold_k_n10: r_n10.threshold,
        threshold_k_n20: r_n20.threshold,
        threshold_k_n50: r_n50.threshold,
        k_n10_used: r_n10.k_used,
        k_n20_used: r_n20.k_used,
        k_n50_used: r_n50.k_used,
        n_observations: n as u64,
    }
}

// ─── Private helpers ──────────────────────────────────────────────

/// Filter input to the finite positive absolute-return subset.
/// Zero-valued returns are excluded because `ln(0)` is undefined in
/// the log-mean step.
fn filter_positive_abs(returns: &[f64]) -> Vec<f64> {
    returns
        .iter()
        .filter(|r| r.is_finite())
        .map(|r| r.abs())
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect()
}

/// Query a freshly-built sketch for the `1 − k_fraction` quantile with
/// the `max(10, …)` floor applied to `k`.
fn sketch_threshold_at(
    abs_rets: &[f64],
    k_fraction: f64,
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> f64 {
    let n = abs_rets.len();
    if n < 2 {
        return f64::NAN;
    }
    let k = ((n as f64 * k_fraction).floor() as usize).max(10).min(n - 1);
    let q_threshold = 1.0 - (k as f64 / n as f64);
    match algorithm {
        SketchAlgorithm::Kll => {
            let mut sk = crate::primitives::specialist::KllSketch::new(epsilon);
            sk.add_slice(abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::Gk => {
            let mut sk = crate::primitives::specialist::GkSketch::new(epsilon);
            sk.add_slice(abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::Tdigest => {
            let mut sk = crate::primitives::specialist::TdigestSketch::new(epsilon);
            sk.add_slice(abs_rets);
            sk.quantile(q_threshold)
        }
        SketchAlgorithm::DdSketch => {
            let mut sk = crate::primitives::specialist::DdSketch::new(epsilon);
            sk.add_slice(abs_rets);
            sk.quantile(q_threshold)
        }
    }
}

/// Finalize the log-mean over values exceeding `threshold`, given a
/// pre-filtered `abs_rets` slice and its nominal `k_fraction`.
fn finalize_single(abs_rets: &[f64], k_fraction: f64, threshold: f64) -> HillSingleResult {
    let n = abs_rets.len();
    if n < 2 || !threshold.is_finite() || threshold <= 0.0 {
        return HillSingleResult {
            hill: f64::NAN,
            threshold,
            k_used: 0,
            n_observations: n as u64,
        };
    }
    let k_used = ((n as f64 * k_fraction).floor() as usize).max(10).min(n - 1) as u64;

    let log_ratios: Vec<f64> = abs_rets
        .iter()
        .copied()
        .filter(|r| *r > threshold)
        .map(|r| (r / threshold).ln())
        .collect();

    if log_ratios.is_empty() {
        return HillSingleResult {
            hill: f64::NAN,
            threshold,
            k_used,
            n_observations: n as u64,
        };
    }

    let hill = crate::math::sum(&log_ratios) / log_ratios.len() as f64;
    HillSingleResult {
        hill,
        threshold,
        k_used,
        n_observations: n as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_too_short_returns_nan() {
        assert!(hill_estimator_streaming(&[], 0.05).hill.is_nan());
        assert!(hill_estimator_streaming(&[1.0], 0.05).hill.is_nan());
        let three = hill_estimator_three_k(&[]);
        assert!(three.hill_k_n10.is_nan());
        assert!(three.hill_k_n20.is_nan());
        assert!(three.hill_k_n50.is_nan());
    }

    #[test]
    fn single_k_reports_full_diagnostics() {
        let mut samples: Vec<f64> = Vec::with_capacity(10000);
        for i in 1..=10000 {
            let u = van_der_corput(i, 2);
            samples.push((1.0 - u).powf(-1.0 / 2.0));
        }
        let r = hill_estimator_streaming(&samples, 0.05);
        assert!(r.hill.is_finite());
        assert!(r.threshold > 0.0);
        assert!(r.k_used >= 10);
        assert_eq!(r.n_observations, 10000);
    }

    #[test]
    fn pareto_known_tail_index_single_k() {
        let mut samples: Vec<f64> = Vec::with_capacity(10000);
        for i in 1..=10000 {
            let u = van_der_corput(i, 2);
            let x = (1.0 - u).powf(-1.0 / 2.0); // α = 2, ξ = 0.5
            samples.push(x);
        }
        let r = hill_estimator_streaming(&samples, 0.05);
        assert!(
            (r.hill - 0.5).abs() < 0.15,
            "Hill estimator off: got {}, want ~0.5",
            r.hill
        );
    }

    #[test]
    fn three_k_form_matches_three_single_k_calls() {
        let mut samples: Vec<f64> = Vec::with_capacity(10000);
        for i in 1..=10000 {
            let u = van_der_corput(i, 2);
            samples.push((1.0 - u).powf(-1.0 / 2.0));
        }
        let three = hill_estimator_three_k(&samples);
        let single_n10 = hill_estimator_streaming(&samples, 0.10);
        let single_n20 = hill_estimator_streaming(&samples, 0.05);
        let single_n50 = hill_estimator_streaming(&samples, 0.02);

        assert!((three.hill_k_n10 - single_n10.hill).abs() < 1e-12);
        assert!((three.hill_k_n20 - single_n20.hill).abs() < 1e-12);
        assert!((three.hill_k_n50 - single_n50.hill).abs() < 1e-12);
        assert_eq!(three.n_observations, single_n10.n_observations);
        assert_eq!(three.k_n10_used, single_n10.k_used);
        assert_eq!(three.k_n20_used, single_n20.k_used);
        assert_eq!(three.k_n50_used, single_n50.k_used);
    }

    #[test]
    fn three_k_estimates_agree_for_pareto() {
        // For a true Pareto, the Hill estimator should be stable
        // across different k values. The three estimates should all
        // be within a reasonable band of the true ξ = 0.5.
        let mut samples: Vec<f64> = Vec::with_capacity(20000);
        for i in 1..=20000 {
            let u = van_der_corput(i, 2);
            samples.push((1.0 - u).powf(-1.0 / 2.0));
        }
        let three = hill_estimator_three_k(&samples);
        for (label, val) in [
            ("n10", three.hill_k_n10),
            ("n20", three.hill_k_n20),
            ("n50", three.hill_k_n50),
        ] {
            assert!(
                (val - 0.5).abs() < 0.2,
                "{label}: Hill = {val}, want ~0.5"
            );
        }
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

        let target = 1.0 / 3.0;
        for (label, val) in [("kll", h_kll.hill), ("gk", h_gk.hill), ("tdigest", h_td.hill)] {
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
        samples.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let r_dirty = hill_estimator_streaming(&samples, 0.05);
        let cleaned: Vec<f64> = samples
            .iter()
            .copied()
            .filter(|r| r.is_finite() && *r > 0.0)
            .collect();
        let r_clean = hill_estimator_streaming(&cleaned, 0.05);
        assert!(
            (r_dirty.hill - r_clean.hill).abs() < 0.05,
            "dirty {} vs clean {}",
            r_dirty.hill,
            r_clean.hill
        );
        assert_eq!(r_dirty.n_observations, r_clean.n_observations);
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
