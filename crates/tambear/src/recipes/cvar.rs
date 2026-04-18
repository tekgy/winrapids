//! Conditional Value-at-Risk (Expected Shortfall) — tail-mean of returns.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! quantile-sketch primitive (Tier 1) plus a Kulisch-backed average of
//! the values below the tail threshold. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! For a return series and tail probability `α ∈ (0, 1)`:
//!
//! ```text
//! VaR_α   =  α-th quantile of returns                    (loss threshold)
//! CVaR_α  =  E[ return  |  return ≤ VaR_α ]              (mean of values below threshold)
//! ```
//!
//! Per the Acerbi-Tasche (2002) formulation, CVaR is a coherent risk
//! measure (subadditive, monotonic, translation-invariant,
//! positive-homogeneous) where VaR alone is not. SIP uses CVaR_1% and
//! CVaR_5% as standard tail-risk header fields.
//!
//! # Composition
//!
//! - **QuantileSketch primitive** (Tier 1) — KLL by default; t-digest
//!   or GK selectable via `using(sketch: ...)`. The sketch lets us
//!   estimate `VaR_α` without sorting the full return series, which
//!   is essential for streaming / mergeable computation.
//! - **Threshold pass + Kulisch-backed mean** — `math::mean` over the
//!   subset of returns at or below the VaR threshold.
//!
//! # Reference
//!
//! Acerbi, C., & Tasche, D. (2002). On the coherence of expected
//! shortfall. *Journal of Banking & Finance* 26(7): 1487–1503.
//!
//! # Default parameters
//!
//! - `tail_pct` — caller-supplied. SIP uses 0.01 (CVaR_1%) and 0.05
//!   (CVaR_5%).
//! - `epsilon` — sketch error in quantile rank. SIP-suitable default
//!   0.005 (half a percent rank error). Tighter is more accurate but
//!   uses more memory.
//! - `sketch` — KLL by default per locked vocabulary. Override per
//!   call: `cvar_with(returns, tail, eps, SketchAlgorithm::Tdigest)`.
//!
//! # NaN/Inf policy
//!
//! Inherited from the sketch and `math::mean`: non-finite values are
//! skipped at both stages. Returns NaN if the threshold pass yields
//! no finite values at or below `VaR_α`.

use crate::primitives::specialist::quantile_sketch::{
    QuantileSketch, SketchAlgorithm,
};

/// CVaR (expected shortfall) at the lower-tail probability `tail_pct`,
/// using the locked-default KLL sketch with the locked-default ε.
///
/// Convenience entry point for the common case. For per-call
/// overrides on sketch algorithm or epsilon, use `cvar_with`.
///
/// # Panics
///
/// Panics if `tail_pct` is non-finite or outside `(0, 1)`.
pub fn cvar(returns: &[f64], tail_pct: f64) -> f64 {
    cvar_with(returns, tail_pct, 0.005, SketchAlgorithm::DEFAULT)
}

/// CVaR with explicit sketch algorithm and epsilon.
///
/// `tail_pct` must be in `(0, 1)`. Typical values: 0.01, 0.05.
///
/// `epsilon` is the sketch's additive error in quantile rank. Smaller
/// is more accurate; uses more memory.
///
/// `algorithm` selects the underlying quantile sketch. Per locked
/// vocabulary, KLL is default; t-digest gives best tail accuracy; GK
/// is intrinsically mergeable.
pub fn cvar_with(
    returns: &[f64],
    tail_pct: f64,
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> f64 {
    assert!(
        tail_pct.is_finite() && tail_pct > 0.0 && tail_pct < 1.0,
        "cvar: tail_pct must be finite and in (0, 1), got {tail_pct}"
    );
    if returns.is_empty() {
        return f64::NAN;
    }

    // Step 1: build the quantile sketch and find the VaR threshold.
    let var_threshold = match algorithm {
        SketchAlgorithm::Kll => {
            let mut sk = crate::primitives::specialist::KllSketch::new(epsilon);
            sk.add_slice(returns);
            sk.quantile(tail_pct)
        }
        SketchAlgorithm::Gk => {
            let mut sk = crate::primitives::specialist::GkSketch::new(epsilon);
            sk.add_slice(returns);
            sk.quantile(tail_pct)
        }
        SketchAlgorithm::Tdigest => {
            let mut sk = crate::primitives::specialist::TdigestSketch::new(epsilon);
            sk.add_slice(returns);
            sk.quantile(tail_pct)
        }
    };

    if !var_threshold.is_finite() {
        return f64::NAN;
    }

    // Step 2: Kulisch-backed mean over returns ≤ threshold.
    let tail: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|r| r.is_finite() && *r <= var_threshold)
        .collect();

    crate::math::mean(&tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = (got - want).abs();
        assert!(
            diff < tol,
            "{label}: got {got}, want {want}, diff {diff} > tol {tol}"
        );
    }

    #[test]
    fn empty_returns_nan() {
        assert!(cvar(&[], 0.05).is_nan());
    }

    #[test]
    fn cvar_5pct_uniform_returns() {
        // Uniform returns -100..0 → VaR_5% ≈ -95, CVaR_5% = mean of bottom 5% ≈ -97.5.
        let returns: Vec<f64> = (0..100).map(|i| -100.0 + i as f64).collect();
        let cv = cvar(&returns, 0.05);
        // Bottom 5 values are -100, -99, -98, -97, -96 → mean = -98.
        // Sketch may add ~ε·N error in threshold; allow generous tol.
        assert_close(cv, -98.0, 5.0, "uniform CVaR_5%");
    }

    #[test]
    fn cvar_more_negative_with_tighter_tail() {
        // Tighter tail → more negative CVaR (further into the loss).
        let returns: Vec<f64> = (0..1000).map(|i| -1000.0 + i as f64).collect();
        let cv_5 = cvar(&returns, 0.05);
        let cv_1 = cvar(&returns, 0.01);
        assert!(cv_1 < cv_5, "CVaR_1% should be more negative: {cv_1} vs {cv_5}");
    }

    #[test]
    fn cvar_with_explicit_sketch_choices() {
        // Same return series, all three sketch algorithms — should
        // give similar (not identical) CVaR estimates.
        let returns: Vec<f64> = (0..2000)
            .map(|i| -100.0 + (i as f64) * 0.1 + ((i as f64) * 0.13).sin() * 5.0)
            .collect();

        let cv_kll = cvar_with(&returns, 0.05, 0.01, SketchAlgorithm::Kll);
        let cv_gk = cvar_with(&returns, 0.05, 0.01, SketchAlgorithm::Gk);
        let cv_td = cvar_with(&returns, 0.05, 0.01, SketchAlgorithm::Tdigest);

        // Allow 5% relative difference between sketches at the same ε.
        let pivot = cv_kll;
        for (label, val) in [("gk", cv_gk), ("tdigest", cv_td)] {
            let rel = ((val - pivot) / pivot.abs()).abs();
            assert!(
                rel < 0.05,
                "{label} CVaR diverges from KLL: {val} vs {pivot} (rel {rel})"
            );
        }
    }

    #[test]
    fn cvar_skips_non_finite_returns() {
        let mut returns: Vec<f64> = (0..100).map(|i| -100.0 + i as f64).collect();
        returns.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let cv = cvar(&returns, 0.05);
        // Should match the no-NaN/Inf case (Kulisch + sketch both skip non-finite).
        assert_close(cv, -98.0, 5.0, "with NaN/Inf");
    }

    #[test]
    #[should_panic(expected = "tail_pct")]
    fn panics_on_tail_zero() {
        let _ = cvar(&[1.0, 2.0, 3.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "tail_pct")]
    fn panics_on_tail_one() {
        let _ = cvar(&[1.0, 2.0, 3.0], 1.0);
    }
}
