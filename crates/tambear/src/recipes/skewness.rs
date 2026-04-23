//! Skewness — third standardized central moment.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over three
//! Kulisch-backed power sums + scalar closed form. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The sample skewness is the third standardized central moment:
//!
//! ```text
//! μ     =  (1/n) · Σ x[i]
//! m₂    =  (1/n) · Σ (x[i] − μ)²     =  variance_population(x)
//! m₃    =  (1/n) · Σ (x[i] − μ)³
//! skew  =  m₃  /  m₂^(3/2)
//! ```
//!
//! Positive skew indicates a right-heavy tail; negative skew indicates
//! a left-heavy tail; Gaussian skew is 0.
//!
//! The central moments expand in raw power sums via the multinomial:
//!
//! ```text
//! Σ (x − μ)²  =  Σx²  −  n·μ²
//! Σ (x − μ)³  =  Σx³  −  3·μ·Σx²  +  2·n·μ³
//! ```
//!
//! so a single pass that accumulates `Σx`, `Σx²`, `Σx³` suffices — the
//! standard fused-moment pattern used across the SIP per-hour signals.
//!
//! This is the **biased / method-of-moments** skewness (divides by
//! `n`, not by `n−1` or `(n−1)(n−2)/n`). It's the form used by most
//! exploratory tools and the one the Jarque–Bera statistic expects.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `skewness` header field is computed on log returns from the
//! four raw-moment sums already accumulated in the hour's pass:
//! `sum_ret`, `sum_ret2`, `sum_ret3` (plus `n`). This recipe's
//! `skewness_from_moment_sums` entry point consumes those directly.
//!
//! # Composition
//!
//! - **Three Kulisch-backed reductions** — `Σx`, `Σx²`, `Σx³`. Lower to
//!   `accumulate(x, All, Op::Add, expr=Value)`,
//!   `accumulate(x, All, Op::Add, expr=Custom("v·v"))`,
//!   `accumulate(x, All, Op::Add, expr=Custom("v·v·v"))`. Fuse into a
//!   single pass under the atom layer.
//! - **Scalar closed form** — the central-moment expansion plus
//!   `m₃ / m₂^(3/2)`.
//!
//! # NaN/Inf policy
//!
//! - `n < 2` or zero/negative population variance → returns NaN
//!   (skewness undefined).
//! - Any non-finite input in the slice form → that value is skipped
//!   via `math::sum` and `count_finite`.
//!
//! # Default parameters
//!
//! None. This is the biased moment-of-methods form; the less-biased
//! `(n√(n−1))/((n−2))` scaling used by Excel/SAS is available as
//! `skewness_adjusted` in the non-recipe `descriptive` module.

/// Skewness closed form from pre-computed moment sums.
///
/// `n` is the number of finite observations. `sum_x`, `sum_x2`,
/// `sum_x3` are the corresponding raw-power sums.
///
/// Returns NaN when `n < 2` or when the population variance is
/// non-positive (skewness undefined).
#[inline]
pub fn skewness_from_moment_sums(n: u64, sum_x: f64, sum_x2: f64, sum_x3: f64) -> f64 {
    if n < 2 {
        return f64::NAN;
    }
    if !sum_x.is_finite() || !sum_x2.is_finite() || !sum_x3.is_finite() {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = sum_x / nf;
    // Population variance: (Σx² − n·μ²) / n
    let var_pop = (sum_x2 - nf * mu * mu) / nf;
    if !var_pop.is_finite() || var_pop <= 0.0 {
        return f64::NAN;
    }
    // Σ(x − μ)³ = Σx³ − 3·μ·Σx² + 2·n·μ³
    let m3_num = sum_x3 - 3.0 * mu * sum_x2 + 2.0 * nf * mu * mu * mu;
    let m3 = m3_num / nf;
    m3 / var_pop.powf(1.5)
}

/// Skewness from a raw slice of observations.
///
/// Accumulates `Σx`, `Σx²`, `Σx³` over finite observations via
/// `KulischAccumulator` (bit-exact under reordering) and then applies
/// the closed form.
pub fn skewness(x: &[f64]) -> f64 {
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    let mut s1 = KulischAccumulator::new();
    let mut s2 = KulischAccumulator::new();
    let mut s3 = KulischAccumulator::new();
    let mut n: u64 = 0;
    for &v in x {
        if !v.is_finite() {
            continue;
        }
        s1.add_f64(v);
        s2.add_f64(v * v);
        s3.add_f64(v * v * v);
        n += 1;
    }
    skewness_from_moment_sums(n, s1.to_f64(), s2.to_f64(), s3.to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_short_is_nan() {
        assert!(skewness(&[]).is_nan());
        assert!(skewness(&[1.0]).is_nan());
    }

    #[test]
    fn symmetric_distribution_zero_skew() {
        // Perfectly symmetric around the mean → skew = 0.
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let s = skewness(&x);
        assert!(s.abs() < 1e-12, "got {s}");
    }

    #[test]
    fn right_heavy_positive_skew() {
        // One large positive outlier pulls skew positive.
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
        let s = skewness(&x);
        assert!(s > 0.0, "expected positive skew, got {s}");
    }

    #[test]
    fn left_heavy_negative_skew() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -100.0];
        let s = skewness(&x);
        assert!(s < 0.0, "expected negative skew, got {s}");
    }

    #[test]
    fn flat_distribution_nan() {
        // All equal values → zero variance → NaN by convention.
        let x = vec![5.0; 100];
        assert!(skewness(&x).is_nan());
    }

    #[test]
    fn from_moment_sums_matches_slice_form() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let s_slice = skewness(&x);
        let n = x.len() as u64;
        let sx: f64 = x.iter().sum();
        let sx2: f64 = x.iter().map(|v| v * v).sum();
        let sx3: f64 = x.iter().map(|v| v * v * v).sum();
        let s_sums = skewness_from_moment_sums(n, sx, sx2, sx3);
        assert!((s_slice - s_sums).abs() < 1e-10, "{s_slice} vs {s_sums}");
    }

    #[test]
    fn skips_non_finite_values() {
        // Inserting NaN should not change the answer when the surviving
        // subset is the same.
        let clean: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let mut with_nan = clean.clone();
        with_nan.insert(2, f64::NAN);
        with_nan.push(f64::INFINITY);
        let a = skewness(&clean);
        let b = skewness(&with_nan);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn known_value_canonical_example() {
        // Classic textbook: [2, 4, 4, 4, 5, 5, 7, 9] has biased skew
        // close to (15·√8) / (16·(2)^(3/2) · √2) — computed directly:
        //   μ = 5, σ² = 4, m₃ = -2, skew = -2 / 4^1.5 = -2/8 = -0.25?
        // Actually μ = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5
        //   Σ(x-μ)² = 9+1+1+1+0+0+4+16 = 32, /n=8 → σ² = 4
        //   Σ(x-μ)³ = -27 + (-1) + (-1) + (-1) + 0 + 0 + 8 + 64 = 42, /n=8 → m₃ = 5.25
        //   skew = 5.25 / 4^(3/2) = 5.25 / 8 = 0.65625
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = skewness(&x);
        assert!((s - 0.65625).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn invariant_under_translation() {
        // skew(x + c) = skew(x). Using raw-moment expansion, large
        // translations amplify rounding noise in Σxᵏ − k·μ·Σxᵏ⁻¹ + …;
        // shift by a modest amount so the test exercises the
        // mathematical invariance rather than the precision limit of
        // the biased-sums formulation.
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let x_shift: Vec<f64> = x.iter().map(|v| v + 10.0).collect();
        let a = skewness(&x);
        let b = skewness(&x_shift);
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }

    #[test]
    fn sign_flips_under_negation() {
        // skew(-x) = -skew(x).
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let x_neg: Vec<f64> = x.iter().map(|v| -v).collect();
        let a = skewness(&x);
        let b = skewness(&x_neg);
        assert!((a + b).abs() < 1e-10, "{a} vs {b}");
    }
}
