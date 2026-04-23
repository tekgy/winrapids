//! Jarque–Bera (1987) normality test from the third and fourth
//! standardized moments.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over the
//! `skewness` and `kurtosis_excess` recipes + scalar closed form + a
//! χ² tail-probability gather. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The Jarque–Bera statistic is a joint test of zero skewness and zero
//! excess kurtosis:
//!
//! ```text
//! JB     =  (n / 6) · ( S²  +  K² / 4 )
//! p      =  P(X ≥ JB  |  X ~ χ²(2))
//! ```
//!
//! where `S` is the (biased) sample skewness and `K` is the (biased)
//! sample excess kurtosis. Under the null of normality, `JB` is
//! asymptotically `χ²(2)`, so reject normality at level `α` when
//! `p ≤ α`.
//!
//! Closed-form O(1) once the four raw-moment sums
//! `Σx, Σx², Σx³, Σx⁴` are known: compute `S` and `K` from those, then
//! plug into the formula.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `normality_pvalue` header field is defined as the JB p-value
//! on log returns, computed as a pure function of the hour's already-
//! accumulated skewness and kurtosis. This recipe contributes no new
//! accumulation — it consumes the moment sums the fused-moment pass
//! already produces.
//!
//! # Reference
//!
//! Jarque, C. M., & Bera, A. K. (1987). A test for normality of
//! observations and regression residuals. *International Statistical
//! Review* 55(2): 163–172.
//!
//! # Composition
//!
//! - **`skewness_from_moment_sums`** — the `S` input.
//! - **`kurtosis_excess_from_moment_sums`** — the `K` input.
//! - **Scalar closed form** — `JB = (n/6) · (S² + K²/4)`.
//! - **`chi2_right_tail_p`** — gather from the χ²(2) distribution for
//!   the p-value.
//!
//! # NaN/Inf policy
//!
//! - `n < 3` → returns a result with NaN statistic and NaN p-value (the
//!   test requires at least three observations to estimate skew + kurt).
//! - Any upstream NaN (zero variance, too few observations) propagates
//!   to both the statistic and the p-value.
//!
//! # Default parameters
//!
//! None. JB has no tunable parameters; the upstream choice of
//! biased-vs-unbiased moment estimators is the policy decision (this
//! recipe uses biased, matching the classical Jarque–Bera derivation).

use crate::recipes::{kurtosis_excess, skewness};

/// Jarque–Bera test result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JarqueBeraResult {
    /// The JB statistic = (n/6) · (S² + K²/4).
    pub statistic: f64,
    /// P(X ≥ JB | X ~ χ²(2)). Small p → reject normality.
    pub p_value: f64,
}

/// Jarque–Bera from pre-computed moment sums — the fused-accumulate
/// entry point.
///
/// `n` is the number of finite observations. `sum_x`, `sum_x2`,
/// `sum_x3`, `sum_x4` are the raw-power sums the SIP bucket / hour
/// pass already produces.
pub fn jarque_bera_from_moment_sums(
    n: u64,
    sum_x: f64,
    sum_x2: f64,
    sum_x3: f64,
    sum_x4: f64,
) -> JarqueBeraResult {
    if n < 3 {
        return JarqueBeraResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
        };
    }
    let s = skewness::skewness_from_moment_sums(n, sum_x, sum_x2, sum_x3);
    let k = kurtosis_excess::kurtosis_excess_from_moment_sums(n, sum_x, sum_x2, sum_x3, sum_x4);
    if !s.is_finite() || !k.is_finite() {
        return JarqueBeraResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
        };
    }
    let nf = n as f64;
    let jb = (nf / 6.0) * (s * s + 0.25 * k * k);
    let p = crate::special_functions::chi2_right_tail_p(jb, 2.0);
    JarqueBeraResult {
        statistic: jb,
        p_value: p.clamp(0.0, 1.0),
    }
}

/// Jarque–Bera from a raw slice of observations.
///
/// Accumulates the four moment sums via `KulischAccumulator` over finite
/// observations, then applies the closed form.
pub fn jarque_bera(x: &[f64]) -> JarqueBeraResult {
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    let mut s1 = KulischAccumulator::new();
    let mut s2 = KulischAccumulator::new();
    let mut s3 = KulischAccumulator::new();
    let mut s4 = KulischAccumulator::new();
    let mut n: u64 = 0;
    for &v in x {
        if !v.is_finite() {
            continue;
        }
        let v2 = v * v;
        s1.add_f64(v);
        s2.add_f64(v2);
        s3.add_f64(v2 * v);
        s4.add_f64(v2 * v2);
        n += 1;
    }
    jarque_bera_from_moment_sums(n, s1.to_f64(), s2.to_f64(), s3.to_f64(), s4.to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_too_short_is_nan() {
        let r = jarque_bera(&[]);
        assert!(r.statistic.is_nan());
        assert!(r.p_value.is_nan());
        let r = jarque_bera(&[1.0, 2.0]);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn flat_distribution_is_nan() {
        // Zero variance → skew/kurt undefined → statistic NaN.
        let x = vec![3.14; 100];
        let r = jarque_bera(&x);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn symmetric_mesokurtic_small_statistic() {
        // Perfectly symmetric data around zero with roughly normal-like
        // tails should produce a small JB statistic and a non-tiny p.
        // Use a symmetric 9-point stencil approximating a discretized
        // normal — not a gold oracle, just a sanity check.
        let x = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let r = jarque_bera(&x);
        assert!(r.statistic.is_finite());
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn skewed_data_large_statistic() {
        // One huge positive outlier → big positive skew + heavy tails →
        // JB should spike and p should be very small.
        let mut x: Vec<f64> = (0..99).map(|i| i as f64 * 0.01).collect();
        x.push(1000.0);
        let r = jarque_bera(&x);
        assert!(r.statistic > 10.0, "expected large JB, got {}", r.statistic);
        assert!(r.p_value < 0.01, "expected small p, got {}", r.p_value);
    }

    #[test]
    fn two_point_distribution_kurtosis_term_dominates() {
        // Binary ±1 has skew = 0, excess kurt = -2. JB = n/6 · (0 + 4/4)
        // = n/6.
        let x: Vec<f64> = (0..60)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = jarque_bera(&x);
        assert!((r.statistic - 10.0).abs() < 1e-10, "got {}", r.statistic);
    }

    #[test]
    fn p_value_in_unit_interval() {
        // For any finite statistic the p-value must be in [0, 1].
        let cases: Vec<Vec<f64>> = vec![
            (0..30).map(|i| i as f64).collect(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        ];
        for x in cases {
            let r = jarque_bera(&x);
            if r.p_value.is_finite() {
                assert!(
                    r.p_value >= 0.0 && r.p_value <= 1.0,
                    "p out of [0,1]: {}",
                    r.p_value
                );
            }
        }
    }

    #[test]
    fn from_moment_sums_matches_slice_form() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let r_slice = jarque_bera(&x);
        let n = x.len() as u64;
        let sx: f64 = x.iter().sum();
        let sx2: f64 = x.iter().map(|v| v * v).sum();
        let sx3: f64 = x.iter().map(|v| v * v * v).sum();
        let sx4: f64 = x.iter().map(|v| v * v * v * v).sum();
        let r_sums = jarque_bera_from_moment_sums(n, sx, sx2, sx3, sx4);
        assert!(
            (r_slice.statistic - r_sums.statistic).abs() < 1e-8,
            "{} vs {}",
            r_slice.statistic,
            r_sums.statistic
        );
        assert!(
            (r_slice.p_value - r_sums.p_value).abs() < 1e-10,
            "{} vs {}",
            r_slice.p_value,
            r_sums.p_value
        );
    }

    #[test]
    fn statistic_non_negative() {
        // JB is a sum of squares; cannot be negative by construction.
        let cases: Vec<Vec<f64>> = vec![
            (0..20).map(|i| i as f64).collect(),
            (0..20).map(|i| -(i as f64)).collect(),
            vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
        ];
        for x in cases {
            let r = jarque_bera(&x);
            if r.statistic.is_finite() {
                assert!(r.statistic >= 0.0, "negative JB: {}", r.statistic);
            }
        }
    }
}
