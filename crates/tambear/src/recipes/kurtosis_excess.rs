//! Excess kurtosis — fourth standardized central moment minus 3.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over four
//! Kulisch-backed power sums + scalar closed form. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The sample excess kurtosis is:
//!
//! ```text
//! μ           =  (1/n) · Σ x[i]
//! m₂          =  (1/n) · Σ (x[i] − μ)²     =  variance_population(x)
//! m₄          =  (1/n) · Σ (x[i] − μ)⁴
//! kurt_excess =  m₄ / m₂²  −  3
//! ```
//!
//! The `−3` shifts the Gaussian reference to 0: positive excess
//! kurtosis indicates heavier tails than Gaussian (leptokurtic);
//! negative indicates lighter tails (platykurtic).
//!
//! Expanding the fourth central moment in raw-power sums:
//!
//! ```text
//! Σ (x − μ)⁴  =  Σx⁴  −  4·μ·Σx³  +  6·μ²·Σx²  −  3·n·μ⁴
//! ```
//!
//! so a single pass that accumulates `Σx`, `Σx²`, `Σx³`, `Σx⁴`
//! suffices — the same fused-moment pass that drives `skewness` and
//! `jarque_bera`.
//!
//! This is the **biased / method-of-moments** excess kurtosis
//! (divides by `n`). It's the form Jarque–Bera uses.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `kurtosis` header field is computed on log returns from the
//! four raw-moment sums already accumulated in the hour's pass:
//! `sum_ret`, `sum_ret2`, `sum_ret3`, `sum_ret4` (plus `n`). This
//! recipe's `kurtosis_excess_from_moment_sums` entry point consumes
//! those directly.
//!
//! # Composition
//!
//! - **Four Kulisch-backed reductions** — `Σx`, `Σx²`, `Σx³`, `Σx⁴`.
//!   Lower to `accumulate(x, All, Op::Add, expr=Custom("v^k"))` for
//!   k ∈ {1, 2, 3, 4}. Fuse into a single pass under the atom layer.
//! - **Scalar closed form** — the central-moment expansion plus
//!   `m₄ / m₂² − 3`.
//!
//! # NaN/Inf policy
//!
//! - `n < 2` or zero/negative population variance → returns NaN.
//! - Any non-finite input in the slice form → that value is skipped.
//!
//! # Default parameters
//!
//! None. Biased moment-of-methods form.

/// Excess kurtosis closed form from pre-computed moment sums.
///
/// `n` is the number of finite observations. `sum_x`, `sum_x2`,
/// `sum_x3`, `sum_x4` are the raw-power sums.
///
/// Returns NaN when `n < 2` or when the population variance is
/// non-positive.
#[inline]
pub fn kurtosis_excess_from_moment_sums(
    n: u64,
    sum_x: f64,
    sum_x2: f64,
    sum_x3: f64,
    sum_x4: f64,
) -> f64 {
    if n < 2 {
        return f64::NAN;
    }
    if !sum_x.is_finite()
        || !sum_x2.is_finite()
        || !sum_x3.is_finite()
        || !sum_x4.is_finite()
    {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = sum_x / nf;
    let var_pop = (sum_x2 - nf * mu * mu) / nf;
    if !var_pop.is_finite() || var_pop <= 0.0 {
        return f64::NAN;
    }
    // Σ(x − μ)⁴  =  Σx⁴ − 4μ·Σx³ + 6μ²·Σx² − 3n·μ⁴
    let mu2 = mu * mu;
    let m4_num = sum_x4 - 4.0 * mu * sum_x3 + 6.0 * mu2 * sum_x2 - 3.0 * nf * mu2 * mu2;
    let m4 = m4_num / nf;
    m4 / (var_pop * var_pop) - 3.0
}

/// Excess kurtosis from a raw slice of observations.
///
/// Accumulates the four power sums over finite observations via
/// `KulischAccumulator` and applies the closed form.
pub fn kurtosis_excess(x: &[f64]) -> f64 {
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
    kurtosis_excess_from_moment_sums(n, s1.to_f64(), s2.to_f64(), s3.to_f64(), s4.to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_short_is_nan() {
        assert!(kurtosis_excess(&[]).is_nan());
        assert!(kurtosis_excess(&[1.0]).is_nan());
    }

    #[test]
    fn flat_distribution_nan() {
        let x = vec![7.0; 100];
        assert!(kurtosis_excess(&x).is_nan());
    }

    #[test]
    fn gaussian_like_near_zero() {
        // Uniform-ish samples should have negative excess kurtosis
        // (platykurtic). Exact value for true uniform on [0,1] is -6/5.
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let k = kurtosis_excess(&x);
        // Biased sample estimator on a discrete uniform grid: expect
        // something noticeably below zero; -1.2 ± reasonable tolerance.
        assert!(k < 0.0, "expected negative excess kurtosis, got {k}");
        assert!(k > -2.0, "got {k}");
    }

    #[test]
    fn heavy_tail_positive_excess() {
        // Mostly zeros with one large outlier → heavy-tailed → positive.
        let mut x = vec![0.0; 99];
        x.push(100.0);
        let k = kurtosis_excess(&x);
        assert!(k > 0.0, "expected positive excess, got {k}");
    }

    #[test]
    fn two_point_distribution_minus_two() {
        // Classic: a two-point distribution (half at +1, half at -1)
        // has m₂ = 1, m₄ = 1, excess kurtosis = -2.
        let x: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let k = kurtosis_excess(&x);
        assert!((k - (-2.0)).abs() < 1e-12, "got {k}");
    }

    #[test]
    fn from_moment_sums_matches_slice_form() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let k_slice = kurtosis_excess(&x);
        let n = x.len() as u64;
        let sx: f64 = x.iter().sum();
        let sx2: f64 = x.iter().map(|v| v * v).sum();
        let sx3: f64 = x.iter().map(|v| v * v * v).sum();
        let sx4: f64 = x.iter().map(|v| v * v * v * v).sum();
        let k_sums = kurtosis_excess_from_moment_sums(n, sx, sx2, sx3, sx4);
        assert!((k_slice - k_sums).abs() < 1e-10, "{k_slice} vs {k_sums}");
    }

    #[test]
    fn skips_non_finite_values() {
        let clean: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let mut with_nan = clean.clone();
        with_nan.insert(2, f64::NAN);
        with_nan.push(f64::INFINITY);
        let a = kurtosis_excess(&clean);
        let b = kurtosis_excess(&with_nan);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn invariant_under_translation_and_scaling() {
        // kurt_excess(a·x + b) = kurt_excess(x) for a ≠ 0.
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let y: Vec<f64> = x.iter().map(|v| 3.5 * v + 100.0).collect();
        let a = kurtosis_excess(&x);
        let b = kurtosis_excess(&y);
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }

    #[test]
    fn heavier_tails_have_larger_kurtosis() {
        // Kurtosis is scale-invariant, so "99 zeros + one big outlier"
        // gives the same kurtosis as "99 zeros + one small outlier" —
        // only the SHAPE of the tail matters. Compare two shapes with
        // different tail thickness instead.
        //
        // `thin_tail` — uniform noise in [-1, 1], n=100 (platykurtic).
        // `fat_tail`  — 98 near-zero values + 2 extreme outliers
        //              (heavy-tailed discrete distribution).
        let thin_tail: Vec<f64> = (0..100)
            .map(|i| (i as f64 / 50.0) - 1.0)
            .collect();
        let mut fat_tail: Vec<f64> = vec![0.01; 98];
        fat_tail.push(10.0);
        fat_tail.push(-10.0);
        assert!(
            kurtosis_excess(&fat_tail) > kurtosis_excess(&thin_tail),
            "fat={} thin={}",
            kurtosis_excess(&fat_tail),
            kurtosis_excess(&thin_tail)
        );
    }
}
