//! Kyle's (1985) lambda — linear price-impact coefficient per unit of
//! signed order flow, together with the regression's coefficient of
//! determination.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over five
//! Kulisch-backed accumulations + scalar closed form. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Kyle's (1985) linear price-impact model posits a regression
//!
//! ```text
//! Δp[i]  =  λ · signed_vol[i]  +  ε[i]
//! ```
//!
//! where `Δp[i]` is the price change attributable to trade `i` and
//! `signed_vol[i]` is the signed trade quantity (`+qty` for buys,
//! `−qty` for sells). The OLS slope `λ` is the informed-trader price
//! impact in price units per unit of order flow; `R²` is the fraction
//! of price-change variance explained by order flow, diagnostic for
//! how well the linear-impact model fits the bucket.
//!
//! The closed forms in five Kulisch-backed accumulations:
//!
//! ```text
//! Σx    =  Σ signed_vol[i]
//! Σy    =  Σ Δp[i]
//! Σxy   =  Σ signed_vol[i] · Δp[i]
//! Σx²   =  Σ signed_vol[i]²
//! Σy²   =  Σ Δp[i]²
//!
//! num   =  n·Σxy  −  Σx·Σy
//! denom_x  =  n·Σx²  −  (Σx)²
//! denom_y  =  n·Σy²  −  (Σy)²
//!
//! λ     =  num / denom_x
//! R²    =  num²  /  (denom_x · denom_y)
//! ```
//!
//! All five sums fuse into a single scatter pass over the tick stream
//! under the atom layer.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` (v2,
//! 2026-04-22) the per-bucket `kyle_lambda` field is paired with
//! `kyle_r2` as a multi-output kernel. Both outputs derive from the
//! same five sums — the recipe contributes them together and the SIP
//! writer writes λ to byte offset `[557:561)` and R² to `[655:659)`.
//! The same recipe serves the hourly-aggregate `kyle_lambda_hourly`
//! plus `kyle_r2_hourly` with `kyle_n_observations` via a larger
//! `All`-grouping call.
//!
//! # Reference
//!
//! Kyle, A. S. (1985). Continuous auctions and insider trading.
//! *Econometrica* 53(6): 1315–1335.
//!
//! # Composition
//!
//! - **Five Kulisch-backed reductions** — `Σx`, `Σy`, `Σxy`, `Σx²`,
//!   `Σy²`. Each lowers to
//!   `accumulate(..., Grouping::ByKey{bucket}, Op::Add, ...)` with a
//!   different Expr projection. Fuse into one pass under the atom
//!   layer.
//! - **Scalar closed form** — closed-form λ and R² from the same five
//!   sums (no additional accumulations).
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element.
//!
//! - Empty input or fewer than 2 finite paired observations → both
//!   `lambda` and `r_squared` NaN (`λ` unidentifiable).
//! - Any non-finite entry in either series → that index is skipped in
//!   all five accumulations.
//! - Near-zero signed-volume variance → both `lambda` and `r_squared`
//!   NaN (λ undefined when order flow has no variation).
//! - Near-zero price-change variance → `lambda` is still defined but
//!   `r_squared` is NaN (no variation for OLS to explain).
//!
//! # Default parameters
//!
//! None at this level; the upstream choice of price-change definition
//! (Δp vs log-return, USD vs quote units) is the policy decision.

/// Result of Kyle's lambda regression: slope + coefficient of
/// determination + observation count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KyleLambdaResult {
    /// Kyle's λ in `price_change_unit / signed_volume_unit`.
    pub lambda: f64,
    /// OLS coefficient of determination, `R² ∈ [0, 1]` under a
    /// well-identified regression. NaN when price-change variance is
    /// zero.
    pub r_squared: f64,
    /// Count of finite paired observations that entered the regression.
    pub n_observations: u64,
}

impl KyleLambdaResult {
    /// Result when the regression is unidentifiable (insufficient data,
    /// zero order-flow variance, etc.).
    #[inline]
    pub fn nan() -> Self {
        Self {
            lambda: f64::NAN,
            r_squared: f64::NAN,
            n_observations: 0,
        }
    }
}

/// Kyle's lambda regression on aligned price-change and signed-volume
/// series.
///
/// Returns `KyleLambdaResult` with (λ, R², n). λ is NaN for shorter
/// than 2 finite paired observations, or when signed volume has no
/// variation (λ unidentifiable). If `λ` is defined but price-change
/// variance is zero, `r_squared` is NaN and `lambda` is a valid (if
/// meaningless in the degenerate limit) OLS output.
///
/// # Panics
///
/// Panics if the two input slices have mismatched lengths.
pub fn kyle_lambda(price_changes: &[f64], signed_volumes: &[f64]) -> KyleLambdaResult {
    assert_eq!(
        price_changes.len(),
        signed_volumes.len(),
        "kyle_lambda: price_changes and signed_volumes must match length"
    );
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    if price_changes.len() < 2 {
        return KyleLambdaResult::nan();
    }

    let mut sx = KulischAccumulator::new();
    let mut sy = KulischAccumulator::new();
    let mut sxy = KulischAccumulator::new();
    let mut sx2 = KulischAccumulator::new();
    let mut sy2 = KulischAccumulator::new();
    let mut n: u64 = 0;

    for i in 0..price_changes.len() {
        let y = price_changes[i];
        let x = signed_volumes[i];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        sx.add_f64(x);
        sy.add_f64(y);
        sxy.add_f64(x * y);
        sx2.add_f64(x * x);
        sy2.add_f64(y * y);
        n += 1;
    }

    kyle_lambda_from_sums(
        n,
        sx.to_f64(),
        sy.to_f64(),
        sxy.to_f64(),
        sx2.to_f64(),
        sy2.to_f64(),
    )
}

/// Kyle's lambda closed form from pre-computed sums — the fused-
/// accumulate entry point.
///
/// `n` is the number of finite paired observations. `sum_x`, `sum_y`,
/// `sum_xy`, `sum_x2`, `sum_y2` are the five accumulations over those
/// observations.
///
/// Returns `KyleLambdaResult::nan()` when `n < 2`, any input is non-
/// finite, or the order-flow denominator `n·Σx² − (Σx)²` is non-
/// positive. When the price denominator `n·Σy² − (Σy)²` is non-
/// positive but the order-flow one is fine, `lambda` is still returned
/// and only `r_squared` is NaN.
#[inline]
pub fn kyle_lambda_from_sums(
    n: u64,
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_x2: f64,
    sum_y2: f64,
) -> KyleLambdaResult {
    if n < 2 {
        return KyleLambdaResult::nan();
    }
    if !sum_x.is_finite()
        || !sum_y.is_finite()
        || !sum_xy.is_finite()
        || !sum_x2.is_finite()
        || !sum_y2.is_finite()
    {
        return KyleLambdaResult::nan();
    }
    let nf = n as f64;
    let denom_x = nf * sum_x2 - sum_x * sum_x;
    if !denom_x.is_finite() || denom_x <= 1e-15 {
        return KyleLambdaResult {
            lambda: f64::NAN,
            r_squared: f64::NAN,
            n_observations: n,
        };
    }
    let num = nf * sum_xy - sum_x * sum_y;
    let lambda = num / denom_x;

    let denom_y = nf * sum_y2 - sum_y * sum_y;
    // R² requires both denominators positive; if price-change variance
    // is ≈ 0, R² is not defined (no variance to explain).
    let r_squared = if denom_y.is_finite() && denom_y > 1e-15 {
        // R² = num² / (denom_x · denom_y).  Clamped to [0, 1] to absorb
        // the small negative floating-point noise that can appear when
        // the true R² is exactly 1.
        ((num * num) / (denom_x * denom_y)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };

    KyleLambdaResult {
        lambda,
        r_squared,
        n_observations: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_single_is_nan() {
        let r = kyle_lambda(&[], &[]);
        assert!(r.lambda.is_nan());
        assert!(r.r_squared.is_nan());
        assert_eq!(r.n_observations, 0);

        let r = kyle_lambda(&[0.1], &[1.0]);
        assert!(r.lambda.is_nan());
        assert!(r.r_squared.is_nan());
    }

    #[test]
    fn no_flow_variance_is_nan() {
        // All signed volumes identical → λ unidentifiable.
        let dp = vec![0.1, 0.2, 0.3, -0.1, 0.0];
        let sv = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let r = kyle_lambda(&dp, &sv);
        assert!(r.lambda.is_nan());
        assert!(r.r_squared.is_nan());
        // n still counts how many paired finite obs we saw, even
        // though λ can't be identified.
        assert_eq!(r.n_observations, 5);
    }

    #[test]
    fn all_nan_input_is_nan() {
        let dp = vec![f64::NAN; 5];
        let sv = vec![f64::NAN; 5];
        let r = kyle_lambda(&dp, &sv);
        assert!(r.lambda.is_nan());
        assert_eq!(r.n_observations, 0);
    }

    #[test]
    fn perfect_linear_impact_recovers_slope_and_r2() {
        // Δp = 0.5 · signed_vol exactly → λ = 0.5, R² = 1.
        let sv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let dp: Vec<f64> = sv.iter().map(|v| 0.5 * v).collect();
        let r = kyle_lambda(&dp, &sv);
        assert!((r.lambda - 0.5).abs() < 1e-12, "λ={}", r.lambda);
        assert!((r.r_squared - 1.0).abs() < 1e-10, "R²={}", r.r_squared);
        assert_eq!(r.n_observations, 21);
    }

    #[test]
    fn negative_impact_recovered() {
        let sv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let dp: Vec<f64> = sv.iter().map(|v| -0.3 * v).collect();
        let r = kyle_lambda(&dp, &sv);
        assert!((r.lambda + 0.3).abs() < 1e-12, "λ={}", r.lambda);
        assert!((r.r_squared - 1.0).abs() < 1e-10, "R²={}", r.r_squared);
    }

    #[test]
    fn zero_slope_zero_r2_for_independent_flow() {
        // dp constant, sv varying → λ=0, R²=NaN (no price variance).
        let sv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let dp: Vec<f64> = vec![0.0; sv.len()];
        let r = kyle_lambda(&dp, &sv);
        assert!(r.lambda.abs() < 1e-12, "λ={}", r.lambda);
        // No price variance → R² undefined.
        assert!(r.r_squared.is_nan(), "R²={}", r.r_squared);
    }

    #[test]
    fn noisy_impact_r2_below_one() {
        // Linear impact with added noise → R² < 1 but > 0.
        let sv: Vec<f64> = (-20..=20).map(|x| x as f64).collect();
        let mut dp = Vec::with_capacity(sv.len());
        let mut seed: u64 = 0xfeedface;
        for &v in &sv {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            // 0.5·v plus noise on the order of v's range.
            dp.push(0.5 * v + (u - 0.5) * 10.0);
        }
        let r = kyle_lambda(&dp, &sv);
        // λ should still be near 0.5 (unbiased in expectation) but the
        // finite sample can drift. R² should be in (0, 1).
        assert!(r.lambda.is_finite());
        assert!(r.r_squared > 0.0 && r.r_squared < 1.0, "R²={}", r.r_squared);
    }

    #[test]
    fn intercept_does_not_affect_slope_or_r2() {
        let sv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let dp: Vec<f64> = sv.iter().map(|v| 100.0 + 0.5 * v).collect();
        let r = kyle_lambda(&dp, &sv);
        assert!((r.lambda - 0.5).abs() < 1e-12);
        assert!((r.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn from_sums_matches_slice_form() {
        let sv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let dp: Vec<f64> = sv.iter().map(|v| 0.25 * v + 1.0).collect();

        let n = sv.len() as u64;
        let sx: f64 = sv.iter().sum();
        let sy: f64 = dp.iter().sum();
        let sxy: f64 = sv.iter().zip(dp.iter()).map(|(a, b)| a * b).sum();
        let sx2: f64 = sv.iter().map(|a| a * a).sum();
        let sy2: f64 = dp.iter().map(|a| a * a).sum();

        let r_slice = kyle_lambda(&dp, &sv);
        let r_sums = kyle_lambda_from_sums(n, sx, sy, sxy, sx2, sy2);
        assert!(
            (r_slice.lambda - r_sums.lambda).abs() < 1e-10,
            "λ slice {} vs sums {}",
            r_slice.lambda,
            r_sums.lambda
        );
        assert!((r_slice.r_squared - r_sums.r_squared).abs() < 1e-10);
        assert_eq!(r_slice.n_observations, r_sums.n_observations);
    }

    #[test]
    fn r_squared_in_unit_interval() {
        // For any non-degenerate input, R² ∈ [0, 1].
        let cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 1.0, 3.0, 2.0, 4.0]),
            (vec![-1.0, 0.0, 1.0, 2.0], vec![1.0, 1.0, 1.5, 2.0]),
        ];
        for (dp, sv) in cases {
            let r = kyle_lambda(&dp, &sv);
            if r.r_squared.is_finite() {
                assert!(
                    (0.0..=1.0).contains(&r.r_squared),
                    "R² out of range: {}",
                    r.r_squared
                );
            }
        }
    }

    #[test]
    fn skips_non_finite_observations() {
        let sv_full: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let mut dp_full: Vec<f64> = sv_full.iter().map(|v| 0.5 * v).collect();
        dp_full[5] = f64::NAN;

        let mut sv_clean = Vec::new();
        let mut dp_clean = Vec::new();
        for (s, d) in sv_full.iter().zip(dp_full.iter()) {
            if s.is_finite() && d.is_finite() {
                sv_clean.push(*s);
                dp_clean.push(*d);
            }
        }

        let r_full = kyle_lambda(&dp_full, &sv_full);
        let r_clean = kyle_lambda(&dp_clean, &sv_clean);
        assert!((r_full.lambda - r_clean.lambda).abs() < 1e-12);
        assert!((r_full.r_squared - r_clean.r_squared).abs() < 1e-12);
        assert_eq!(r_full.n_observations, r_clean.n_observations);
    }

    #[test]
    #[should_panic(expected = "match length")]
    fn panics_on_mismatched_lengths() {
        let _ = kyle_lambda(&[0.1, 0.2], &[1.0]);
    }
}
