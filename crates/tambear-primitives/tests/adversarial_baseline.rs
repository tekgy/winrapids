//! Adversarial baseline tests: pathological inputs that expose silent failures.
//!
//! These tests are DESIGNED to fail or expose bugs in the current implementation.
//! A passing test means the bug is fixed or the behavior is confirmed correct.
//!
//! Every test here corresponds to a pitfall entry:
//!   campsites/expedition/20260411120000-the-bit-exact-trek/pitfalls/
//!
//! Strategy:
//!   - All identical values (zero variance)
//!   - All NaN / all Inf / mixed NaN+real
//!   - Single element / two elements / minimum viable n
//!   - Extreme values (1e300, 1e-300, subnormals)
//!   - Near-singular (catastrophic cancellation)
//!   - Pathological distributions (constant, two-point)
//!   - Order independence (different orderings of same data)
//!   - Min/Max identity value propagation on NaN inputs
//!
//! Silence is the worst failure mode: plausible-but-wrong answers.
//! These tests prefer to fail loudly.

use tambear_primitives::recipes::{
    count, sum, sum_of_squares, product,
    mean_arithmetic, mean_geometric, mean_harmonic, mean_quadratic,
    variance, variance_biased, std_dev, skewness, kurtosis_excess,
    l1_norm, l2_norm, linf_norm,
    min_all, max_all, range_all, midrange,
    dot_product, covariance, pearson_r, sum_squared_diff, rmse, mae,
    Recipe,
};
use tambear_primitives::accumulates::{fuse_passes, execute_pass_cpu};
use tambear_primitives::tbs::eval;

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Harness (mirrors the one in recipes/mod.rs tests)
// ═══════════════════════════════════════════════════════════════════

fn run_recipe(recipe: &Recipe, x: &[f64], y: &[f64]) -> f64 {
    let passes = fuse_passes(&recipe.slots);
    let mut vars: HashMap<String, f64> = HashMap::new();
    for pass in &passes {
        let results = execute_pass_cpu(pass, x, 0.0, y);
        for (name, val) in results {
            vars.insert(name, val);
        }
    }
    let mut final_val = 0.0;
    for g in &recipe.gathers {
        let v = eval(&g.expr, 0.0, 0.0, 0.0, &vars);
        vars.insert(g.output.clone(), v);
        if g.output == recipe.result { final_val = v; }
    }
    final_val
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Variance catastrophic cancellation (one-pass formula)
//
// The formula: (Σx² - (Σx)²/n) / (n-1)
// Fails when mean >> std: Σx² and (Σx)²/n are nearly equal,
// catastrophic cancellation in the subtraction destroys precision.
//
// True variance of {1e9 + k*1e-6 : k=0..999} = (999²/4) * 1e-12
// ≈ 8.3250e-8 (population) / 8.3333e-8 (sample).
//
// The one-pass formula on these values should lose many digits.
// Welford's online algorithm would be correct.
//
// EXPECTED BEHAVIOR: this test should FAIL with the current one-pass
// implementation. When it passes, variance has been fixed to use a
// numerically stable algorithm.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn variance_catastrophic_cancellation_exposed() {
    // Data: 1e9, 1e9+1e-6, 1e9+2e-6, ..., 1e9+999e-6
    // Mean = 1e9 + 499.5e-6
    // True sample variance = (Σ(k - 499.5)² for k=0..999) * 1e-12 / 999
    //   = (999 * 1000 * (2*999+1) / 6 / 1000 - 499.5²*1000 ...
    // Easier: var of arithmetic sequence k=0..999 is (n²-1)/12 = (1000²-1)/12 ≈ 83333.25
    //   scaled by (1e-6)² → true sample var ≈ 8.33325e-8
    let n = 1000usize;
    let data: Vec<f64> = (0..n).map(|i| 1.0e9 + (i as f64) * 1.0e-6).collect();

    // True sample variance: arithmetic sequence 0..n-1 scaled by 1e-6
    // Var of 0..n-1 = n*(n-1)/12 for continuous... actually:
    // E[k] = (n-1)/2, E[k²] = (n-1)(2n-1)/6
    // Var[k] = E[k²] - E[k]² = (n-1)(2n-1)/6 - ((n-1)/2)²
    //        = (n-1)[(2n-1)/6 - (n-1)/4]
    //        = (n-1)[(4n-2 - 3n+3)/12]
    //        = (n-1)(n+1)/12
    // Sample var = n/(n-1) * pop_var = n*(n+1)/12 * (1e-6)²
    let n_f = n as f64;
    let true_sample_var = n_f * (n_f + 1.0) / 12.0 * 1.0e-12;  // ≈ 8.3417e-8

    let computed = run_recipe(&variance(), &data, &[]);

    // Require 1% relative accuracy — a VERY generous tolerance.
    // Even this may fail with the one-pass formula on large mean.
    let rel_err = ((computed - true_sample_var) / true_sample_var).abs();
    assert!(
        rel_err < 0.01,
        "variance catastrophic cancellation: true={:.6e}, computed={:.6e}, rel_err={:.3e}",
        true_sample_var, computed, rel_err
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: NaN propagation
//
// If any element is NaN, the result of most statistics should be NaN.
// Silent failure: a plausible-looking non-NaN result when NaN is present.
//
// Currently sum += NaN gives NaN (IEEE 754 correct).
// But Op::Max and Op::Min use > and < comparisons: NaN > x is false,
// NaN < x is false — so NaN is SILENTLY IGNORED by Min/Max.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sum_with_nan_is_nan() {
    let data = [1.0, f64::NAN, 3.0];
    let result = run_recipe(&sum(), &data, &[]);
    assert!(result.is_nan(), "sum with NaN input should be NaN, got {}", result);
}

#[test]
fn mean_with_nan_is_nan() {
    let data = [1.0, f64::NAN, 3.0];
    let result = run_recipe(&mean_arithmetic(), &data, &[]);
    assert!(result.is_nan(), "mean with NaN input should be NaN, got {}", result);
}

#[test]
fn variance_with_nan_is_nan() {
    let data = [1.0, f64::NAN, 3.0];
    let result = run_recipe(&variance(), &data, &[]);
    assert!(result.is_nan(), "variance with NaN input should be NaN, got {}", result);
}

/// SILENT FAILURE: min with NaN silently ignores the NaN because
/// NaN < current_min is always false. Returns the min of the non-NaN values.
/// This is WRONG — contaminated data should produce a NaN result.
#[test]
fn min_with_nan_is_nan() {
    let data = [3.0, f64::NAN, 1.0];
    let result = run_recipe(&min_all(), &data, &[]);
    assert!(
        result.is_nan(),
        "min with NaN input should be NaN (SILENT FAILURE: NaN ignored by < comparison), got {}",
        result
    );
}

/// SILENT FAILURE: max with NaN silently ignores the NaN because
/// NaN > current_max is always false. Returns the max of non-NaN values.
#[test]
fn max_with_nan_is_nan() {
    let data = [1.0, f64::NAN, 5.0];
    let result = run_recipe(&max_all(), &data, &[]);
    assert!(
        result.is_nan(),
        "max with NaN input should be NaN (SILENT FAILURE: NaN ignored by > comparison), got {}",
        result
    );
}

#[test]
fn pearson_r_with_nan_is_nan() {
    let x = [1.0, f64::NAN, 3.0];
    let y = [2.0, 4.0, 6.0];
    let result = run_recipe(&pearson_r(), &x, &y);
    assert!(result.is_nan(), "pearson_r with NaN in x should be NaN, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Inf arithmetic
//
// sum([+Inf, -Inf, 1.0]) = NaN (correct per IEEE 754: Inf - Inf = NaN)
// sum([+Inf, +Inf, 1.0]) = +Inf
// max([+Inf, -Inf]) = +Inf (correct)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sum_of_pos_and_neg_inf_is_nan() {
    let data = [f64::INFINITY, f64::NEG_INFINITY, 1.0];
    let result = run_recipe(&sum(), &data, &[]);
    assert!(
        result.is_nan(),
        "sum(+Inf, -Inf, 1) should be NaN (Inf - Inf = NaN), got {}",
        result
    );
}

#[test]
fn sum_of_all_pos_inf_is_inf() {
    let data = [f64::INFINITY, f64::INFINITY, 1.0];
    let result = run_recipe(&sum(), &data, &[]);
    assert!(
        result.is_infinite() && result > 0.0,
        "sum(+Inf, +Inf, 1) should be +Inf, got {}",
        result
    );
}

#[test]
fn variance_with_inf_element() {
    // inf in data should make variance NaN (since Σx² = inf, Σx = inf,
    // and (inf)² / n = inf, then inf - inf = NaN)
    let data = [1.0, f64::INFINITY, 3.0];
    let result = run_recipe(&variance(), &data, &[]);
    assert!(
        result.is_nan() || result.is_infinite(),
        "variance with Inf element should be NaN or Inf, got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Empty input
//
// What should happen with empty data?
// - count() → 0.0  (correct: there are 0 elements)
// - sum() → 0.0  (identity for Add: correct)
// - mean() → 0.0 / 0.0 = NaN  (expected)
// - variance() → 0.0 / (-1.0) = -0.0  (WRONG: should be NaN)
// - min_all() → +Inf  (identity for Min: implementation artifact, not correct)
// - max_all() → -Inf  (identity for Max: implementation artifact, not correct)
// - product() → 1.0  (identity for Mul: correct as empty product)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn count_empty_is_zero() {
    let result = run_recipe(&count(), &[], &[]);
    assert_eq!(result, 0.0, "count of empty should be 0");
}

#[test]
fn sum_empty_is_zero() {
    let result = run_recipe(&sum(), &[], &[]);
    assert_eq!(result, 0.0, "sum of empty should be 0 (identity)");
}

#[test]
fn mean_empty_is_nan() {
    let result = run_recipe(&mean_arithmetic(), &[], &[]);
    assert!(
        result.is_nan(),
        "mean of empty should be NaN (0/0), got {}",
        result
    );
}

#[test]
fn variance_empty_is_nan() {
    // (0.0 - 0.0*0.0/0.0) / (0.0 - 1.0) = (0 - NaN) / -1 = NaN
    // BUT: count=0 → (sum_sq - sum*sum/count) / (count-1)
    //   = (0 - 0*0/0) / (0-1) = (0 - NaN) / -1 = NaN
    // The -0.0/(-1.0) path would give 0.0 — WRONG.
    let result = run_recipe(&variance(), &[], &[]);
    assert!(
        result.is_nan(),
        "variance of empty should be NaN, got {}",
        result
    );
}

#[test]
fn min_empty_is_not_silently_inf() {
    // The accumulator initializes to +Inf for Min, which is the identity.
    // Returning +Inf for empty data is an ARTIFACT of the implementation,
    // not a mathematically meaningful answer.
    // The correct behavior is NaN (undefined for empty set).
    let result = run_recipe(&min_all(), &[], &[]);
    assert!(
        result.is_nan(),
        "min of empty set should be NaN (undefined), got {} (the identity value +Inf leaked out)",
        result
    );
}

#[test]
fn max_empty_is_not_silently_neg_inf() {
    let result = run_recipe(&max_all(), &[], &[]);
    assert!(
        result.is_nan(),
        "max of empty set should be NaN (undefined), got {} (the identity value -Inf leaked out)",
        result
    );
}

#[test]
fn product_empty_is_one() {
    // Empty product = 1 (the identity). This is mathematically correct.
    let result = run_recipe(&product(), &[], &[]);
    assert_eq!(result, 1.0, "product of empty should be 1.0 (empty product convention)");
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Single element
//
// Single-element statistics expose divide-by-zero in unbiased formulas.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn variance_single_element_is_nan() {
    // Unbiased sample variance: divide by (n-1) = 0 for n=1.
    // IEEE 754: 0.0 / 0.0 = NaN.
    // BUT: the formula gives (x² - x*x/1) / (1-1) = 0/0 = NaN. Correct!
    // OR if numerator is computed as 0.0 (sum_sq - sum^2 / count = x² - x² = 0):
    //   then it's 0.0 / 0.0 = NaN. ✓
    let result = run_recipe(&variance(), &[5.0], &[]);
    assert!(
        result.is_nan(),
        "variance of single element should be NaN (n-1=0 in denominator), got {}",
        result
    );
}

#[test]
fn mean_single_element_is_that_element() {
    let result = run_recipe(&mean_arithmetic(), &[7.0], &[]);
    assert_eq!(result, 7.0, "mean of single element should be the element itself");
}

#[test]
fn std_dev_single_element_is_nan() {
    let result = run_recipe(&std_dev(), &[5.0], &[]);
    assert!(
        result.is_nan(),
        "std_dev of single element should be NaN (sqrt(nan) = nan), got {}",
        result
    );
}

#[test]
fn skewness_single_element_is_nan() {
    let result = run_recipe(&skewness(), &[5.0], &[]);
    assert!(
        result.is_nan(),
        "skewness of single element should be NaN (zero variance denominator), got {}",
        result
    );
}

#[test]
fn kurtosis_single_element_is_nan() {
    let result = run_recipe(&kurtosis_excess(), &[5.0], &[]);
    assert!(
        result.is_nan(),
        "kurtosis of single element should be NaN (zero variance denominator), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: All-identical values (zero variance)
//
// When all values are the same, variance = 0. Skewness and kurtosis
// divide by σ³ and σ⁴ respectively — they should be NaN (0/0),
// not Inf, not 0.
//
// The one-pass variance formula: sum_sq - sum^2/n = n*c² - (n*c)²/n = 0.
// This is numerically exact for identical values. ✓
//
// But skewness = μ₃ / σ³: numerator = 0, denominator = 0. Result: NaN.
// Kurtosis = μ₄ / σ⁴ - 3: numerator = 0, denominator = 0. Result: NaN.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn variance_of_constant_data_is_zero() {
    let data = vec![5.0; 100];
    let result = run_recipe(&variance(), &data, &[]);
    assert_eq!(result, 0.0, "variance of constant data should be exactly 0, got {}", result);
}

#[test]
fn skewness_of_constant_data_is_nan() {
    // σ = 0 → skewness = 0/0³ = NaN (undefined, not 0)
    let data = vec![5.0; 10];
    let result = run_recipe(&skewness(), &data, &[]);
    assert!(
        result.is_nan(),
        "skewness of constant data should be NaN (0/0), got {}",
        result
    );
}

#[test]
fn kurtosis_of_constant_data_is_nan() {
    // σ = 0 → kurtosis = 0/0⁴ - 3 = NaN - 3 = NaN
    let data = vec![5.0; 10];
    let result = run_recipe(&kurtosis_excess(), &data, &[]);
    assert!(
        result.is_nan(),
        "kurtosis of constant data should be NaN (0/0), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Near-perfect correlation (Pearson r)
//
// When x and y are linearly related with tiny perturbation, r should
// be within ~1e-14 of 1.0. Catastrophic cancellation in the denominator
// could push r outside [-1, 1] — a physically impossible result.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pearson_r_near_perfect_positive_correlation() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    // y = x + tiny perturbation at last bit
    let y: Vec<f64> = x.iter().map(|&xi| xi + xi * f64::EPSILON * 0.5).collect();
    let r = run_recipe(&pearson_r(), &x, &y);
    // r should be in [0.999999, 1.0] — if it's > 1.0 or NaN, catastrophic cancellation
    assert!(
        r >= 0.999 && r <= 1.0,
        "pearson_r near-perfect correlation: got r={}, should be close to 1.0",
        r
    );
}

#[test]
fn pearson_r_of_identical_data_is_nan_or_one() {
    // x == y exactly → r numerically: (Σx² - (Σx)²/n) = variance * (n-1) ≥ 0
    // Should be 1.0 when numerically stable, NaN when not.
    // But the denominator den_x * den_y = (variance*n-1)² ≥ 0.
    // For identical columns, den_y = den_x, so r = den_x / den_x = 1.0. ✓
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let r = run_recipe(&pearson_r(), &x, &x);
    assert!(
        (r - 1.0).abs() < 1e-12 || r.is_nan(),
        "pearson_r of identical columns should be 1.0 or NaN, got {}",
        r
    );
}

#[test]
fn pearson_r_stays_in_bounds() {
    // Pathological: large mean, small variance — same catastrophic cancellation
    // as variance. The denominator sqrt should not go negative (sqrt(negative) = NaN).
    let data_x: Vec<f64> = (0..1000).map(|i| 1.0e9 + (i as f64) * 1.0e-6).collect();
    let data_y: Vec<f64> = (0..1000).map(|i| 2.0e9 + (i as f64) * 2.0e-6).collect();
    let r = run_recipe(&pearson_r(), &data_x, &data_y);
    assert!(
        r.is_nan() || (r >= -1.0 && r <= 1.0),
        "pearson_r must stay in [-1, 1] or be NaN (catastrophic cancellation can push it out), got {}",
        r
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Extreme values — overflow and underflow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sum_of_large_values_no_silent_overflow() {
    // 1000 copies of f64::MAX / 2. Sum = 500 * f64::MAX > f64::MAX → Inf.
    // This is CORRECT behavior (overflow to Inf), not a silent failure.
    // The test documents the behavior.
    let data = vec![f64::MAX / 1000.0; 2000];
    let result = run_recipe(&sum(), &data, &[]);
    assert!(
        result.is_infinite(),
        "sum of 2000 * (MAX/1000) should overflow to Inf, got {}",
        result
    );
}

#[test]
fn l2_norm_of_large_values() {
    // sqrt(Σ (MAX/1000)²) — sum_sq overflows before sqrt
    let data = vec![f64::MAX / 1000.0; 10];
    let result = run_recipe(&l2_norm(), &data, &[]);
    // Should be Inf (overflow before sqrt), not NaN
    assert!(
        result.is_infinite(),
        "l2_norm of large values should overflow to Inf, got {}",
        result
    );
}

#[test]
fn mean_geometric_of_tiny_values() {
    // exp(Σln(1e-300) / n) — Σln(1e-300) = n * (-690.8...) which is finite
    // so geometric_mean should work for subnormals
    let data = vec![1.0e-300; 100];
    let result = run_recipe(&mean_geometric(), &data, &[]);
    let expected = 1.0e-300_f64;
    let rel_err = ((result - expected) / expected).abs();
    assert!(
        rel_err < 1e-10,
        "geometric_mean of subnormals should be ~1e-300, got {:.6e} (rel_err={:.3e})",
        result, rel_err
    );
}

#[test]
fn variance_of_subnormal_data() {
    // Very small values: subnormal arithmetic might lose precision.
    // All values equal → variance should be exactly 0.
    let tiny = f64::MIN_POSITIVE * 2.0;  // smallest normal, not subnormal
    let data = vec![tiny; 100];
    let result = run_recipe(&variance(), &data, &[]);
    assert_eq!(
        result, 0.0,
        "variance of identical subnormal values should be 0, got {:.6e}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Geometric mean with zeros and negatives
//
// ln(0) = -Inf → exp(-Inf) = 0. The result is 0, not NaN.
// ln(-1) = NaN → geometric_mean of negative values = NaN.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn geometric_mean_with_zero_is_zero() {
    // ln(0) = -Inf → log_sum = -Inf → exp(-Inf / n) = exp(-Inf) = 0
    let data = [1.0, 2.0, 0.0, 4.0];
    let result = run_recipe(&mean_geometric(), &data, &[]);
    assert_eq!(
        result, 0.0,
        "geometric_mean with a zero element should be 0 (via ln(0)=-Inf), got {}",
        result
    );
}

#[test]
fn geometric_mean_with_negative_is_nan() {
    // ln(-1) = NaN → log_sum = NaN → result = NaN
    let data = [1.0, -2.0, 3.0];
    let result = run_recipe(&mean_geometric(), &data, &[]);
    assert!(
        result.is_nan(),
        "geometric_mean with negative value should be NaN (ln of negative), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Harmonic mean with zero
//
// 1/0 = +Inf → recip_sum = +Inf → n/Inf = 0.
// This is the convention (Inf added to sum drives n/sum → 0), but
// should be documented. The harmonic mean of {0, ...} is 0.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn harmonic_mean_with_zero() {
    // 1/0 = Inf; n/Inf = 0 → harmonic mean → 0
    // This is mathematically consistent (H → 0 when any element → 0).
    let data = [1.0, 0.0, 2.0];
    let result = run_recipe(&mean_harmonic(), &data, &[]);
    assert_eq!(
        result, 0.0,
        "harmonic_mean with a zero element should be 0 (1/0=Inf, n/Inf=0), got {}",
        result
    );
}

#[test]
fn harmonic_mean_with_negative() {
    // Harmonic mean is undefined for negative values (denominators can cancel).
    // The formula will still produce a number — but is it meaningful?
    // {1, -1}: 1/1 + 1/(-1) = 0 → n/0 = Inf. Dubious.
    let data = [1.0, -1.0];
    let result = run_recipe(&mean_harmonic(), &data, &[]);
    assert!(
        result.is_infinite() || result.is_nan(),
        "harmonic_mean({}, {}) = {} — expected Inf or NaN (undefined for cancelling denominators)",
        1.0, -1.0, result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Two-element edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn variance_two_elements() {
    // Var({a, b}) = (a-b)²/2
    let a = 3.0_f64;
    let b = 7.0_f64;
    let expected = (a - b).powi(2) / 2.0;
    let result = run_recipe(&variance(), &[a, b], &[]);
    assert!(
        (result - expected).abs() < 1e-14,
        "variance of two elements should be (a-b)²/2={}, got {}",
        expected, result
    );
}

#[test]
fn pearson_r_two_elements_is_one_or_neg_one() {
    // With only 2 points, r = ±1 always (two points always fall on a line)
    let x = [1.0, 3.0];
    let y = [2.0, 8.0];
    let r = run_recipe(&pearson_r(), &x, &y);
    assert!(
        (r - 1.0).abs() < 1e-12,
        "pearson_r of 2 points with positive slope should be 1.0, got {}",
        r
    );

    let y2 = [8.0, 2.0];
    let r2 = run_recipe(&pearson_r(), &x, &y2);
    assert!(
        (r2 + 1.0).abs() < 1e-12,
        "pearson_r of 2 points with negative slope should be -1.0, got {}",
        r2
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Order independence
//
// For commutative+associative operations (Add, Mul, Min, Max),
// the result must be the same regardless of element order.
//
// This is TRUE for exact arithmetic. For floating-point, it's not
// guaranteed because (a+b)+c ≠ a+(b+c) in general. However, since
// the CPU path is a simple left-to-right loop, the SAME ordering
// always produces the SAME result. This test verifies that.
//
// NOTE: the GPU path may produce different results due to parallelism.
// That's a Peak 6 problem. Here we test the CPU path.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sum_order_independence_cpu() {
    // For the CPU path (sequential loop), same data same order → same bits.
    let data = [3.1415, 2.7182, 1.4142, 1.6180];
    let r1 = run_recipe(&sum(), &data, &[]);
    let r2 = run_recipe(&sum(), &data, &[]);
    assert_eq!(
        r1.to_bits(), r2.to_bits(),
        "CPU sum must be bit-identical across runs: {} vs {}",
        r1, r2
    );
}

#[test]
fn variance_different_orderings_same_result() {
    // IMPORTANT: this test may FAIL because the one-pass formula
    // (Σx² - (Σx)²/n) / (n-1) is not perfectly order-independent
    // for fp64 — even though the SAME ordering gives the SAME answer,
    // DIFFERENT orderings may give slightly different answers.
    //
    // With small values this is usually fine. With adversarial inputs it's not.
    let data1 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let data2 = [5.0, 4.0, 3.0, 2.0, 1.0];
    let data3 = [3.0, 1.0, 5.0, 2.0, 4.0];
    let v1 = run_recipe(&variance(), &data1, &[]);
    let v2 = run_recipe(&variance(), &data2, &[]);
    let v3 = run_recipe(&variance(), &data3, &[]);
    // These SHOULD all equal 2.5 — if they don't, it's a fp ordering issue.
    assert!(
        (v1 - 2.5).abs() < 1e-13 && (v2 - 2.5).abs() < 1e-13 && (v3 - 2.5).abs() < 1e-13,
        "variance should be 2.5 for all orderings: [1..5]={}, [5..1]={}, [shuffled]={}",
        v1, v2, v3
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Covariance and pearson_r with constant column
//
// If x is constant, var(x) = 0. The Pearson r denominator = 0.
// r = (cov) / 0 = NaN (or Inf if cov ≠ 0, but cov should also be 0).
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pearson_r_constant_x_is_nan() {
    let x = [5.0; 10];
    let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let r = run_recipe(&pearson_r(), &x, &y);
    assert!(
        r.is_nan(),
        "pearson_r with constant x should be NaN (0/0 denominator), got {}",
        r
    );
}

#[test]
fn covariance_constant_x_is_zero() {
    // cov(x, y) = (Σxy - ΣxΣy/n) / (n-1)
    // If x=c: Σxy = c*Σy, Σx = n*c → (c*Σy - n*c*Σy/n)/(n-1) = 0/(n-1) = 0
    let x = [5.0; 10];
    let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let cov = run_recipe(&covariance(), &x, &y);
    assert!(
        cov.abs() < 1e-12,
        "covariance with constant x should be 0, got {}",
        cov
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: RMSE and MAE of empty input
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rmse_empty_is_nan() {
    let result = run_recipe(&rmse(), &[], &[]);
    assert!(
        result.is_nan(),
        "rmse of empty should be NaN (sqrt(0/0)), got {}",
        result
    );
}

#[test]
fn mae_empty_is_nan() {
    let result = run_recipe(&mae(), &[], &[]);
    assert!(
        result.is_nan(),
        "mae of empty should be NaN (0/0), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Product overflow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn product_many_twos_overflows_to_inf() {
    // 2^1075 > f64::MAX → Inf
    let data = vec![2.0_f64; 1075];
    let result = run_recipe(&product(), &data, &[]);
    assert!(
        result.is_infinite(),
        "product(2^1075) should overflow to Inf, got {}",
        result
    );
}

#[test]
fn product_many_halves_underflows_to_zero() {
    // (0.5)^1075 < f64::MIN_POSITIVE → 0.0
    let data = vec![0.5_f64; 1075];
    let result = run_recipe(&product(), &data, &[]);
    assert_eq!(
        result, 0.0,
        "product(0.5^1075) should underflow to 0, got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: L∞ norm with all-NaN (should be NaN, not -Inf)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn linf_norm_all_nan_is_nan() {
    let data = [f64::NAN, f64::NAN, f64::NAN];
    let result = run_recipe(&linf_norm(), &data, &[]);
    // linf = max(|x|). Max initializes to NEG_INFINITY. NaN > NEG_INFINITY is false.
    // So the accumulator stays at NEG_INFINITY — then abs(NaN) = NaN... wait.
    // Actually: the expr is Expr::val().abs() per element.
    // abs(NaN) = NaN. NaN > current_max (NEG_INFINITY) is false.
    // So accumulator stays NEG_INFINITY. WRONG — should be NaN.
    assert!(
        result.is_nan(),
        "linf_norm of all-NaN should be NaN (not -Inf, which leaks the identity value), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Skewness — two-point distribution
//
// For exactly 2 points, the central 3rd moment is 0 (symmetric around mean),
// σ = |a-b|/√2. Skewness = 0/σ³ = 0. This should work.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn skewness_two_points_is_zero() {
    let data = [1.0, 5.0];
    let result = run_recipe(&skewness(), &data, &[]);
    // With 2 points: mean = 3, var = (1-3)² + (5-3)²)/(2-1) = 8 wait...
    // Actually the formula uses biased: m2 = sum_sq/n - mean^2
    // For skewness: the central 3rd moment of a 2-point symmetric dist is 0.
    assert!(
        result.abs() < 1e-12 || result.is_nan(),
        "skewness of 2 points should be 0 (symmetric), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Large N variance stability check
//
// This is the DEFINITIVE catastrophic cancellation test.
// Data: all values = 1000.0001 except one = 1000.0.
// True variance should be computable. The one-pass formula should struggle.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn variance_welford_vs_onepass_stress() {
    // Kahan/Welford check: n=10000 values near 1e8 with std=1.
    // True sample variance ≈ 1.0.
    let n = 10000usize;
    let mean_val = 1.0e8_f64;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            // Values scattered around 1e8 with spacing 1e-4 → std ≈ sqrt(n²-1)/12 * 1e-4
            // Easier: just alternate +1 and -1 around mean
            mean_val + if i % 2 == 0 { 1.0 } else { -1.0 }
        })
        .collect();

    // True sample variance: n/2 values at (mean+1), n/2 at (mean-1).
    // Each deviates by 1 or -1. Variance = E[(x-mean)²] * n/(n-1) = 1.0 * n/(n-1).
    let true_var = 1.0_f64 * (n as f64) / (n as f64 - 1.0);
    let computed = run_recipe(&variance(), &data, &[]);

    let rel_err = ((computed - true_var) / true_var).abs();
    assert!(
        rel_err < 1e-6,
        "variance near large mean: true={:.10e}, computed={:.10e}, rel_err={:.3e}",
        true_var, computed, rel_err
    );
}
