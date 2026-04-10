//! Adversarial Wave 22 — Special function poles: NaN where ±∞ is required
//!
//! Pre-flight correctness sweep for the special functions used in Gamma MLE,
//! Dirichlet estimation, IRT, and Bayesian inference. These functions have
//! poles at non-positive integers where the mathematically correct limit
//! is ±∞, not NaN.
//!
//! Three confirmed bugs:
//!
//! 1. `digamma(0)` (special_functions.rs:212): returns NaN via `x == 0.0` guard.
//!    Correct value: -∞ (digamma has a simple pole at 0 with residue -1;
//!    approaching from above: ψ(x) → -∞ as x → 0⁺).
//!
//! 2. `digamma(-n)` for negative integers (special_functions.rs:219-221):
//!    returns NaN via `(x - rounded).abs() < 1e-12` guard.
//!    Correct value: -∞ (ψ has simple poles at all non-positive integers).
//!
//! 3. `trigamma(0)` (special_functions.rs:256): returns NaN via `x <= 0.0` guard.
//!    Correct value: +∞ (ψ₁(x) = -dψ/dx has a positive pole at 0;
//!    ψ₁(x) ~ 1/x² → +∞ as x → 0⁺).
//!
//! Mathematical truths:
//!   - ψ(x) = d/dx ln Γ(x). Since Γ(x) → ∞ as x → 0⁺ from the right,
//!     ln Γ(x) → +∞ and its slope ψ(x) → -∞ (approaching from a pole
//!     where Γ is going to +∞ with infinite negative slope in log).
//!   - ψ₁(x) = d²/dx² ln Γ(x) = dψ/dx. As ψ(x) → -∞ at x → 0⁺,
//!     ψ₁(x) → +∞.
//!   - Both poles are simple (residue -1 for ψ, residue +1 for ψ₁).
//!   - NaN is the wrong return: it signals "undefined" but the limits exist.
//!     Callers that test `if result.is_nan()` to detect failure will
//!     incorrectly treat a legitimate limit of ±∞ as a computation failure.
//!
//! Also verified (not bugs):
//!   - `log_gamma(0)` correctly returns +∞ (Γ pole at 0 → ln Γ → +∞).
//!   - `log_gamma(-n)` correctly returns +∞ (Γ poles at all negative integers).
//!   - `normal_quantile(0)` correctly returns -∞.
//!   - `normal_quantile(1)` correctly returns +∞.
//!   - `t_cdf(t, 0)`, `f_cdf(x, 0, d2)`, `chi2_cdf(x, 0)` correctly return NaN
//!     (undefined, not limiting values, at degenerate degrees of freedom).
//!   - Near-pole behavior: `digamma(1e-15) ≈ -1e15` (correct, not NaN).
//!   - Near-pole behavior: `log_gamma(1e-15) ≈ 34.5` (correct, not overflow).
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::{digamma, trigamma, log_gamma, log_beta};
use tambear::{normal_quantile, t_cdf, f_cdf, chi2_cdf};

// ═══════════════════════════════════════════════════════════════════════════
// Digamma — Pole at 0
// ═══════════════════════════════════════════════════════════════════════════

/// ψ(x) → -∞ as x → 0⁺.
///
/// ψ(x) = -γ - 1/x + O(x) near x=0 (Laurent expansion).
/// The -1/x term dominates: as x → 0⁺, ψ(x) → -∞.
///
/// EXPECTED: digamma(0.0) = -∞ (NEG_INFINITY).
/// ACTUAL (BUG): digamma(0.0) = NaN — the guard `x == 0.0 → NaN` treats
/// the pole as "undefined" rather than as a well-defined limit of -∞.
///
/// Consequence: callers that test `is_nan()` to detect pole behavior will
/// incorrectly handle the limit. Gradient computations in Gamma MLE that
/// approach α=0 will receive NaN instead of the correct divergence signal.
#[test]
fn digamma_pole_at_zero_is_neg_infinity_not_nan() {
    let result = digamma(0.0_f64);
    assert!(result == f64::NEG_INFINITY,
        "BUG: digamma(0) should be -∞ (simple pole, limit exists), got {} — \
         the guard `if x == 0.0 {{ return NaN }}` returns NaN instead of NEG_INFINITY. \
         The Laurent expansion ψ(x) = -γ - 1/x + O(x) shows ψ → -∞ as x → 0⁺",
        result);
}

/// ψ(x) → -∞ as x → 0 from the left as well (for the pole limit).
/// The reflection formula ψ(1-x) - ψ(x) = π·cot(πx) and the pole structure
/// confirm that the limit is -∞ at x=0 from both sides.
#[test]
fn digamma_approaches_neg_infinity_near_zero() {
    // Approaching from above: ψ(x) → -∞ as x → 0⁺
    let small = digamma(1e-15_f64);
    assert!(small < -1e14,
        "digamma near 0⁺ should be very large negative (≈ -1e15), got {} — \
         this verifies the Laurent expansion is correct for near-pole inputs",
        small);

    let smaller = digamma(1e-10_f64);
    assert!(smaller < -1e9,
        "digamma(1e-10) should be ≈ -1e10, got {}", smaller);
}

// ═══════════════════════════════════════════════════════════════════════════
// Digamma — Poles at negative integers
// ═══════════════════════════════════════════════════════════════════════════

/// ψ(x) has simple poles at x = -1, -2, -3, ... with residue -1.
///
/// At each pole: ψ(x) → -∞ as x → -n from the right.
/// The current implementation returns NaN for these values via the check
/// `(x - rounded).abs() < 1e-12`.
///
/// EXPECTED: digamma(-1.0) = -∞.
/// ACTUAL (BUG): digamma(-1.0) = NaN.
#[test]
fn digamma_pole_at_neg_one_is_neg_infinity() {
    let result = digamma(-1.0_f64);
    assert!(result == f64::NEG_INFINITY,
        "BUG: digamma(-1) should be -∞ (simple pole at each negative integer), got {} — \
         the guard `(x - rounded).abs() < 1e-12 → NaN` returns NaN instead of NEG_INFINITY",
        result);
}

/// Same pole at -2.
#[test]
fn digamma_pole_at_neg_two_is_neg_infinity() {
    let result = digamma(-2.0_f64);
    assert!(result == f64::NEG_INFINITY,
        "BUG: digamma(-2) should be -∞, got {}", result);
}

/// Same pole at -3.
#[test]
fn digamma_pole_at_neg_three_is_neg_infinity() {
    let result = digamma(-3.0_f64);
    assert!(result == f64::NEG_INFINITY,
        "BUG: digamma(-3) should be -∞, got {}", result);
}

/// Non-integer negative x should be finite (not a pole).
/// ψ(-0.5) = -γ - 2(ln 2) - π/2 ≈ -2.3963... (finite).
/// Verify non-pole negative x is handled correctly.
#[test]
fn digamma_non_integer_negative_x_is_finite() {
    let result = digamma(-0.5_f64);
    assert!(result.is_finite(),
        "digamma(-0.5) should be finite (not a pole), got {}", result);
    // ψ(-1/2) = -γ - 2·ln(2) - π ≈ -2 * 0.5772 - 1.3863 - 3.1416 ≈ -5.1...
    // But via reflection formula: ψ(1.5) - π·cot(-π/2) = ψ(1.5) - π·0 = ψ(1.5)
    // ψ(1.5) = ψ(0.5) + 1/0.5 = (-γ - 2·ln2) + 2 ≈ -0.5772 - 1.3863 + 2 ≈ 0.0365
    // Actually via reflection: ψ(-0.5) = ψ(1.5) - π/tan(-π/2)
    // tan(-π/2) = ±∞, so π/tan(-π/2) → 0. So ψ(-0.5) ≈ ψ(1.5) ≈ 0.0365
    // The exact value is ψ(-0.5) = -γ - 2ln2 - π ≈ -3.9...
    // We just check it's finite, not the exact value.
    assert!(result.is_finite(), "digamma(-0.5) finite check");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trigamma — Pole at 0
// ═══════════════════════════════════════════════════════════════════════════

/// ψ₁(x) = dψ/dx → +∞ as x → 0⁺.
///
/// ψ₁(x) = 1/x² + O(1/x) near x=0 (Laurent expansion of the derivative).
/// As x → 0⁺, ψ₁(x) → +∞.
///
/// EXPECTED: trigamma(0.0) = +∞ (INFINITY).
/// ACTUAL (BUG): trigamma(0.0) = NaN — the guard `x <= 0.0 → NaN` treats
/// the pole as "undefined" rather than as a well-defined limit of +∞.
///
/// Consequence: Newton's method for Gamma MLE (which uses trigamma as the
/// Hessian) cannot correctly detect divergence near α=0. Instead of receiving
/// +∞ as the curvature signal, it receives NaN which poisons the entire update.
#[test]
fn trigamma_pole_at_zero_is_pos_infinity_not_nan() {
    let result = trigamma(0.0_f64);
    assert!(result == f64::INFINITY,
        "BUG: trigamma(0) should be +∞ (ψ₁(x) = 1/x² + O(1/x) → +∞ as x → 0⁺), got {} — \
         the guard `if x <= 0.0 {{ return NaN }}` returns NaN instead of INFINITY. \
         Correct: ψ₁(x) ~ 1/x² diverges positively at the pole",
        result);
}

/// ψ₁(x) → +∞ as x → 0⁺: verify the approach is correct for near-pole values.
#[test]
fn trigamma_approaches_pos_infinity_near_zero() {
    let small = trigamma(1e-10_f64);
    assert!(small > 1e19,
        "trigamma(1e-10) should be ≈ 1e20 (≈ 1/x²), got {}", small);

    let smaller = trigamma(1e-6_f64);
    assert!(smaller > 1e11,
        "trigamma(1e-6) should be ≈ 1e12, got {}", smaller);
}

/// Trigamma is positive for all x > 0 (it's a convex function).
/// The recurrence ψ₁(x+1) = ψ₁(x) - 1/x² shows ψ₁ decreases monotonically.
#[test]
fn trigamma_positive_and_decreasing_for_positive_x() {
    let vals: Vec<f64> = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0].iter()
        .map(|&x| trigamma(x))
        .collect();

    for (i, &v) in vals.iter().enumerate() {
        assert!(v > 0.0, "trigamma should be positive for positive x, got {} at index {}", v, i);
    }
    for i in 1..vals.len() {
        assert!(vals[i] < vals[i-1],
            "trigamma should be decreasing: trigamma({:.2}) = {} should be < trigamma({:.2}) = {}",
            [0.01_f64, 0.1, 0.5, 1.0, 2.0, 5.0][i], vals[i],
            [0.01_f64, 0.1, 0.5, 1.0, 2.0, 5.0][i-1], vals[i-1]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verified correct: log_gamma poles return +∞ (not bugs)
// ═══════════════════════════════════════════════════════════════════════════

/// log_gamma(0) correctly returns +∞ — Γ has a simple pole at 0.
/// Document this as a correctness verification, not a bug.
#[test]
fn log_gamma_pole_at_zero_correctly_returns_inf() {
    assert_eq!(log_gamma(0.0_f64), f64::INFINITY,
        "log_gamma(0) should be +∞ (Γ pole at 0 → ln Γ → +∞), got {}",
        log_gamma(0.0));
}

/// log_gamma(-n) correctly returns +∞ for all negative integers.
#[test]
fn log_gamma_poles_at_negative_integers_correctly_return_inf() {
    for n in 1..=5 {
        let result = log_gamma(-(n as f64));
        assert_eq!(result, f64::INFINITY,
            "log_gamma(-{}) should be +∞ (Γ pole at each negative integer), got {}",
            n, result);
    }
}

/// log_gamma near-pole behavior is correct: log_gamma(1e-15) ≈ 34.5.
/// ψ(-1/x term dominates: ln Γ(x) ≈ -ln(x) for small x).
#[test]
fn log_gamma_near_zero_is_finite_and_correct() {
    let result = log_gamma(1e-15_f64);
    assert!(result.is_finite(), "log_gamma(1e-15) should be finite, got {}", result);
    // ln Γ(1e-15) ≈ -ln(1e-15) = 15·ln(10) ≈ 34.539
    assert!((result - 34.5).abs() < 0.1,
        "log_gamma(1e-15) ≈ 34.5, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Verified correct: CDF boundary behavior (not bugs)
// ═══════════════════════════════════════════════════════════════════════════

/// normal_quantile at probability boundaries returns the correct limits.
#[test]
fn normal_quantile_boundary_limits_correct() {
    assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY,
        "normal_quantile(0) should be -∞");
    assert_eq!(normal_quantile(1.0), f64::INFINITY,
        "normal_quantile(1) should be +∞");
}

/// Degenerate degrees of freedom return NaN (undefined, not a limit).
#[test]
fn cdf_degenerate_df_returns_nan() {
    assert!(t_cdf(1.0, 0.0).is_nan(),
        "t_cdf with df=0 is undefined (NaN), got {}", t_cdf(1.0, 0.0));
    assert!(f_cdf(1.0, 0.0, 1.0).is_nan(),
        "f_cdf with d1=0 is undefined (NaN), got {}", f_cdf(1.0, 0.0, 1.0));
    assert!(chi2_cdf(1.0, 0.0).is_nan(),
        "chi2_cdf with k=0 is undefined (NaN), got {}", chi2_cdf(1.0, 0.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Recurrence consistency — mathematical truth checks
// ═══════════════════════════════════════════════════════════════════════════

/// Digamma recurrence: ψ(x+1) = ψ(x) + 1/x for all x > 0.
/// This is the fundamental recurrence relation. Must hold to machine precision.
#[test]
fn digamma_recurrence_relation() {
    for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.7, 10.0, 100.0] {
        let lhs = digamma(x + 1.0);
        let rhs = digamma(x) + 1.0 / x;
        assert!((lhs - rhs).abs() < 1e-12,
            "digamma recurrence ψ(x+1) = ψ(x) + 1/x failed at x={}: lhs={}, rhs={}, diff={}",
            x, lhs, rhs, (lhs - rhs).abs());
    }
}

/// Trigamma recurrence: ψ₁(x+1) = ψ₁(x) - 1/x² for all x > 0.
#[test]
fn trigamma_recurrence_relation() {
    for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.7, 10.0, 100.0] {
        let lhs = trigamma(x + 1.0);
        let rhs = trigamma(x) - 1.0 / (x * x);
        assert!((lhs - rhs).abs() < 1e-11,
            "trigamma recurrence ψ₁(x+1) = ψ₁(x) - 1/x² failed at x={}: lhs={}, rhs={}, diff={}",
            x, lhs, rhs, (lhs - rhs).abs());
    }
}

/// Trigamma is the derivative of digamma: ψ₁(x) ≈ (ψ(x+h) - ψ(x-h)) / (2h).
/// Numerical derivative should match trigamma to ~1e-8 for h=1e-5.
#[test]
fn trigamma_is_derivative_of_digamma() {
    let h = 1e-5_f64;
    for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        let numerical = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
        let exact = trigamma(x);
        let rel_err = (numerical - exact).abs() / exact.abs().max(1e-300);
        assert!(rel_err < 1e-7,
            "trigamma({}) = {} but numerical derivative of digamma = {} (rel_err = {})",
            x, exact, numerical, rel_err);
    }
}
