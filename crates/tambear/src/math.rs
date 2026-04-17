//! Composable math building blocks for recipes.
//!
//! Every function in this module is:
//! - **Cross-platform bit-exact deterministic** — Kulisch-backed under the hood.
//!   Same input → same bits regardless of thread count, execution order, backend,
//!   or CPU architecture.
//! - **NaN/Inf-skip by default** — matches the existing Kulisch silent-skip via
//!   `is_finite()`. All-invalid empty input returns the additive identity (0.0)
//!   for sum-family functions; mean/variance returns NaN when count is 0.
//! - **Numerically exact where possible** — products use `two_product_fma` so the
//!   multiplication step is also exact (both hi and lo components added to
//!   Kulisch). Two-pass variance via centering.
//! - **Pure and stateless** — no hidden caches, no session tracking. For
//!   intermediate sharing use `intermediates::TamSession` wrappers separately.
//!
//! # Why this module exists
//!
//! Recipes are pure composition. Recipes should NEVER inline `.iter().sum::<f64>()`,
//! `.fold(0.0, +)`, or `sum +=` patterns directly — those are non-deterministic
//! under parallelism and cross-backend. When a recipe needs a mean, it calls
//! `math::mean(values)`. When it needs a dot product, `math::dot(a, b)`. If a
//! pattern recipes need isn't here, add it here rather than inlining inside the
//! recipe.
//!
//! # When a function belongs here
//!
//! - It's a common numerical operation (sum, mean, variance, dot, correlation)
//! - It has a single canonical definition (no "which kind of variance" ambiguity)
//! - Recipes will want to call it by name
//! - It can be expressed as a small composition of `accumulate(..., Op::Add)` —
//!   i.e., it's Kingdom A
//!
//! # When a function does NOT belong here
//!
//! - It's specific to one recipe or one domain (live in that recipe's module)
//! - It's a primitive arithmetic operation (live in `primitives::`)
//! - It's a literature-named statistical test / transform (live in its
//!   subject module like `hypothesis::` or `time_series::`)

use crate::primitives::compensated::two_product_fma;
use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

// ═══════════════════════════════════════════════════════════════════════════
// Count helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Count of finite values in `values` — helper for mean/variance denominators.
#[inline]
pub fn count_finite(values: &[f64]) -> usize {
    values.iter().filter(|v| v.is_finite()).count()
}

// ═══════════════════════════════════════════════════════════════════════════
// Sums
// ═══════════════════════════════════════════════════════════════════════════

/// Exact sum of `values` via Kulisch. Correctly-rounded final f64.
///
/// Non-finite values (NaN, ±∞) are skipped. Empty / all-non-finite input
/// returns `0.0` — the additive identity, preserves prefix-sum flatline
/// semantics across gaps.
pub fn sum(values: &[f64]) -> f64 {
    let mut acc = KulischAccumulator::new();
    acc.add_slice(values);
    acc.to_f64()
}

/// Exact sum of squares `Σ xᵢ²` via Kulisch over `two_product_fma`.
///
/// Each product is computed exactly as `(hi, lo) = two_product_fma(v, v)`
/// with both components added to the Kulisch register — no rounding
/// accumulates in the squaring step. Non-finite skipped. Empty → 0.0.
pub fn sum_sq(values: &[f64]) -> f64 {
    let mut acc = KulischAccumulator::new();
    for &v in values {
        if v.is_finite() {
            let (hi, lo) = two_product_fma(v, v);
            acc.add_f64(hi);
            acc.add_f64(lo);
        }
    }
    acc.to_f64()
}

/// Exact sum of centered squared deviations `Σ (xᵢ − μ)²` via Kulisch.
///
/// The mean `μ` is supplied by the caller (typically from `mean(values)`
/// as a pre-pass). Each squared deviation is computed via two_product_fma
/// on the centered residual so the multiplication is exact. Non-finite
/// skipped. Returns `0.0` for empty / all-non-finite input.
pub fn centered_sum_sq(values: &[f64], mean: f64) -> f64 {
    let mut acc = KulischAccumulator::new();
    for &v in values {
        if v.is_finite() {
            let d = v - mean;
            let (hi, lo) = two_product_fma(d, d);
            acc.add_f64(hi);
            acc.add_f64(lo);
        }
    }
    acc.to_f64()
}

/// Exact weighted sum `Σ wᵢ · xᵢ` via Kulisch over `two_product_fma`.
///
/// Skips pair `i` if either `w[i]` or `x[i]` is non-finite.
///
/// # Panics
/// If `weights.len() != values.len()`.
pub fn weighted_sum(values: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(
        values.len(),
        weights.len(),
        "weighted_sum: mismatched lengths ({} vs {})",
        values.len(),
        weights.len()
    );
    let mut acc = KulischAccumulator::new();
    for i in 0..values.len() {
        let (w, v) = (weights[i], values[i]);
        if w.is_finite() && v.is_finite() {
            let (hi, lo) = two_product_fma(w, v);
            acc.add_f64(hi);
            acc.add_f64(lo);
        }
    }
    acc.to_f64()
}

// ═══════════════════════════════════════════════════════════════════════════
// Products (dot and related)
// ═══════════════════════════════════════════════════════════════════════════

/// Exact dot product `Σ aᵢ · bᵢ` via Kulisch over `two_product_fma`.
///
/// Each product is computed exactly via FMA-based error-free transform,
/// with both hi and lo components added to Kulisch. Non-finite pair
/// (either element non-finite) skips. Empty → 0.0.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot: mismatched lengths ({} vs {})",
        a.len(),
        b.len()
    );
    let mut acc = KulischAccumulator::new();
    for i in 0..a.len() {
        let (x, y) = (a[i], b[i]);
        if x.is_finite() && y.is_finite() {
            let (hi, lo) = two_product_fma(x, y);
            acc.add_f64(hi);
            acc.add_f64(lo);
        }
    }
    acc.to_f64()
}

/// Exact centered dot product `Σ (aᵢ − ā)(bᵢ − b̄)` via Kulisch.
///
/// Used for covariance computation. Means supplied by caller as pre-pass.
/// Non-finite pair skipped. Empty → 0.0.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn centered_dot(a: &[f64], mean_a: f64, b: &[f64], mean_b: f64) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "centered_dot: mismatched lengths ({} vs {})",
        a.len(),
        b.len()
    );
    let mut acc = KulischAccumulator::new();
    for i in 0..a.len() {
        let (x, y) = (a[i], b[i]);
        if x.is_finite() && y.is_finite() {
            let dx = x - mean_a;
            let dy = y - mean_b;
            let (hi, lo) = two_product_fma(dx, dy);
            acc.add_f64(hi);
            acc.add_f64(lo);
        }
    }
    acc.to_f64()
}

// ═══════════════════════════════════════════════════════════════════════════
// Central tendency and spread
// ═══════════════════════════════════════════════════════════════════════════

/// Arithmetic mean of finite values.
///
/// Returns `NaN` for empty / all-non-finite input. Denominator is
/// `count_finite(values)` (non-finite inputs do not count).
pub fn mean(values: &[f64]) -> f64 {
    let n = count_finite(values);
    if n == 0 {
        return f64::NAN;
    }
    sum(values) / n as f64
}

/// Sample variance (ddof=1) — `Σ(xᵢ − μ)² / (n−1)`.
///
/// Two-pass: exact mean, then exact centered sum of squares. Returns
/// `NaN` for n < 2. Non-finite values skipped in both passes.
pub fn variance_sample(values: &[f64]) -> f64 {
    let n = count_finite(values);
    if n < 2 {
        return f64::NAN;
    }
    let mu = sum(values) / n as f64;
    centered_sum_sq(values, mu) / (n as f64 - 1.0)
}

/// Population variance (ddof=0) — `Σ(xᵢ − μ)² / n`.
///
/// Two-pass. Returns `NaN` for n < 1.
pub fn variance_population(values: &[f64]) -> f64 {
    let n = count_finite(values);
    if n < 1 {
        return f64::NAN;
    }
    let mu = sum(values) / n as f64;
    centered_sum_sq(values, mu) / n as f64
}

/// Sample standard deviation — `sqrt(variance_sample)`.
pub fn std_dev_sample(values: &[f64]) -> f64 {
    variance_sample(values).sqrt()
}

/// Population standard deviation — `sqrt(variance_population)`.
pub fn std_dev_population(values: &[f64]) -> f64 {
    variance_population(values).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Bivariate (covariance, correlation)
// ═══════════════════════════════════════════════════════════════════════════

/// Sample covariance (ddof=1) — `Σ(aᵢ − ā)(bᵢ − b̄) / (n−1)`.
///
/// Two-pass: means then centered dot. Only pairs where both `a[i]` and
/// `b[i]` are finite contribute; denominator is that pair-count.
/// Returns `NaN` for n < 2.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn covariance_sample(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "covariance_sample: mismatched lengths ({} vs {})",
        a.len(),
        b.len()
    );
    let n = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .count();
    if n < 2 {
        return f64::NAN;
    }
    // Means use pairwise-finite count to be consistent with the numerator.
    let mut acc_a = KulischAccumulator::new();
    let mut acc_b = KulischAccumulator::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            acc_a.add_f64(x);
            acc_b.add_f64(y);
        }
    }
    let mean_a = acc_a.to_f64() / n as f64;
    let mean_b = acc_b.to_f64() / n as f64;
    centered_dot(a, mean_a, b, mean_b) / (n as f64 - 1.0)
}

/// Pearson correlation — `cov(a, b) / (σₐ · σ_b)`.
///
/// Returns `NaN` if either series has zero variance or n < 2. Uses
/// pairwise-finite filter (only pairs where both `a[i]` and `b[i]` are
/// finite count in any of the sums).
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "correlation: mismatched lengths ({} vs {})",
        a.len(),
        b.len()
    );
    let n = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .count();
    if n < 2 {
        return f64::NAN;
    }
    let mut acc_a = KulischAccumulator::new();
    let mut acc_b = KulischAccumulator::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            acc_a.add_f64(x);
            acc_b.add_f64(y);
        }
    }
    let mean_a = acc_a.to_f64() / n as f64;
    let mean_b = acc_b.to_f64() / n as f64;

    // Compute Σ (aᵢ-ā)², Σ (bᵢ-b̄)², Σ (aᵢ-ā)(bᵢ-b̄) in one pass.
    let mut css_a = KulischAccumulator::new();
    let mut css_b = KulischAccumulator::new();
    let mut cdot = KulischAccumulator::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            let dx = x - mean_a;
            let dy = y - mean_b;
            let (h1, l1) = two_product_fma(dx, dx);
            css_a.add_f64(h1);
            css_a.add_f64(l1);
            let (h2, l2) = two_product_fma(dy, dy);
            css_b.add_f64(h2);
            css_b.add_f64(l2);
            let (h3, l3) = two_product_fma(dx, dy);
            cdot.add_f64(h3);
            cdot.add_f64(l3);
        }
    }
    let var_a = css_a.to_f64();
    let var_b = css_b.to_f64();
    if var_a == 0.0 || var_b == 0.0 {
        return f64::NAN;
    }
    cdot.to_f64() / (var_a.sqrt() * var_b.sqrt())
}
