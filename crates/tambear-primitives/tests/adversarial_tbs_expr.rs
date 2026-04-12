//! Adversarial tests for TBS expression evaluation edge cases.
//!
//! Targets:
//! 1. The `Eq` comparison uses a hardcoded epsilon of 1e-15 — this is an
//!    undocumented contract that creates non-deterministic behavior at the
//!    boundary, and is the wrong abstraction for bit-exact semantics.
//! 2. `Sign(0.0)` and `Sign(-0.0)` — IEEE 754 has two zeros.
//! 3. `Sqrt(-0.0)` — should be -0.0 per IEEE 754, not NaN.
//! 4. `Ln(0.0)` — should be -Inf, not NaN or panic.
//! 5. `Recip(0.0)` — should be +Inf (or -Inf for -0.0), not NaN or panic.
//! 6. `Recip(f64::INFINITY)` — should be 0.0.
//! 7. `Pow(0.0, 0.0)` — IEEE 754 says 1.0; mathematical convention is ambiguous.
//! 8. `Pow(-1.0, 0.5)` — NaN (square root of negative).
//! 9. `Pow(f64::INFINITY, 0.0)` — IEEE 754 says 1.0.
//! 10. `Min(NaN, x)` and `Max(NaN, x)` — NaN propagation in TBS Min/Max.
//! 11. `Atan2(0.0, 0.0)` — technically undefined; IEEE 754 defines it as 0.

use tambear_primitives::tbs::{Expr, eval};
use std::collections::HashMap;

fn no_vars() -> HashMap<String, f64> { HashMap::new() }

fn eval1(expr: &Expr, val: f64) -> f64 {
    eval(expr, val, 0.0, 0.0, &no_vars())
}

fn eval2(expr: &Expr, a: f64, b: f64) -> f64 {
    eval(expr, a, b, 0.0, &no_vars())
}

// ═══════════════════════════════════════════════════════════════════
// PITFALL: Eq uses hardcoded 1e-15 epsilon
//
// The `Expr::Eq` comparison returns 1.0 if |a - b| < 1e-15.
// This means:
// - Values differing by 5e-16 are "equal" (1.0 returned)
// - Values differing by 2e-15 are "not equal" (0.0 returned)
//
// For bit-exact arithmetic, "equal" should mean "same bits" (a == b).
// The 1e-15 epsilon is an undocumented contract. It's also the wrong
// direction for the trek's bit-exact thesis: cross-backend Eq behavior
// will differ if backends disagree on whether |a - b| < 1e-15.
//
// EXPECTED BEHAVIOR: Eq should use exact bit comparison (a == b).
// ═══════════════════════════════════════════════════════════════════

#[test]
fn eq_with_values_differing_at_last_bit_is_one() {
    // Two values that differ by 1 ULP.
    // With exact-equality semantics: should return 0.0.
    // With 1e-15 epsilon: may return 1.0 (if the ULP difference < 1e-15).
    let a = 1.0_f64;
    let b = 1.0_f64 + f64::EPSILON; // differs by 1 ULP at magnitude 1.0
    // f64::EPSILON = 2.22e-16 < 1e-15, so the current impl returns 1.0 (treats as equal).
    // This is WRONG for bit-exact semantics.
    let result = eval2(&Expr::Eq(Box::new(Expr::Val), Box::new(Expr::Val2)), a, b);
    assert_eq!(
        result, 0.0,
        "Eq with 1-ULP difference should return 0.0 (not equal), \
         but current impl returns {} (uses 1e-15 epsilon that swallows ULPs < 4.5)",
        result
    );
}

#[test]
fn eq_exact_same_value_is_one() {
    let x = 3.14159265358979_f64;
    let result = eval2(&Expr::Eq(Box::new(Expr::Val), Box::new(Expr::Val2)), x, x);
    assert_eq!(result, 1.0, "Eq of identical values should return 1.0");
}

#[test]
fn eq_nan_is_zero() {
    // NaN ≠ NaN by IEEE 754. Eq(NaN, NaN) should be 0.0.
    let result = eval2(&Expr::Eq(Box::new(Expr::Val), Box::new(Expr::Val2)), f64::NAN, f64::NAN);
    assert_eq!(
        result, 0.0,
        "Eq(NaN, NaN) should be 0.0 (NaN is not equal to itself), got {}",
        result
    );
}

#[test]
fn eq_zero_vs_neg_zero_is_one() {
    // IEEE 754: 0.0 == -0.0 is TRUE (they are equal values, different representations).
    // The bit patterns differ (0x0000... vs 0x8000...) but they should compare equal.
    let result = eval2(&Expr::Eq(Box::new(Expr::Val), Box::new(Expr::Val2)), 0.0_f64, -0.0_f64);
    assert_eq!(
        result, 1.0,
        "Eq(0.0, -0.0) should be 1.0 (IEEE 754: 0.0 == -0.0), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// Sign with zero and negative zero
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sign_of_positive_is_one() {
    assert_eq!(eval1(&Expr::val().into_sign_expr(), 5.0), 1.0);
}

#[test]
fn sign_of_negative_is_neg_one() {
    assert_eq!(eval1(&Expr::val().into_sign_expr(), -3.0), -1.0);
}

#[test]
fn sign_of_zero_is_zero() {
    // The Sign implementation uses: if v > 0 → 1, if v < 0 → -1, else → 0.
    // For 0.0: neither > 0 nor < 0, so → 0. Correct.
    assert_eq!(eval1(&Expr::val().into_sign_expr(), 0.0), 0.0);
}

#[test]
fn sign_of_negative_zero_is_zero() {
    // -0.0 > 0 is false; -0.0 < 0 is false (IEEE 754: -0 is not less than 0).
    // So Sign(-0.0) → 0. This is correct but worth documenting.
    assert_eq!(
        eval1(&Expr::val().into_sign_expr(), -0.0_f64),
        0.0,
        "Sign(-0.0) should be 0 (IEEE 754: -0 == +0)"
    );
}

#[test]
fn sign_of_nan_is_zero_not_nan() {
    // NaN > 0 is false; NaN < 0 is false; so Sign(NaN) → 0.
    // This is a SILENT FAILURE: Sign(NaN) should arguably propagate NaN,
    // but the current implementation returns 0, masking the NaN.
    let result = eval1(&Expr::val().into_sign_expr(), f64::NAN);
    // Document the current behavior, then assert what SHOULD happen:
    // If we want NaN propagation, this test should fail on the current code.
    assert!(
        result.is_nan(),
        "Sign(NaN) should propagate NaN (current impl returns {} — NaN masked by comparison)",
        result
    );
}

// Helper to call Sign via the Expr enum directly
trait IntoSignExpr {
    fn into_sign_expr(self) -> Expr;
}
impl IntoSignExpr for Expr {
    fn into_sign_expr(self) -> Expr {
        Expr::Sign(Box::new(self))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Sqrt edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sqrt_of_negative_zero_is_negative_zero() {
    // IEEE 754: sqrt(-0.0) = -0.0 (not NaN, not error).
    // Rust's f64::sqrt(-0.0) = -0.0. ✓
    let result = eval1(&Expr::val().sqrt(), -0.0_f64);
    // Check it's zero (both +0.0 and -0.0 are zero)
    assert_eq!(result, 0.0, "sqrt(-0.0) should be ±0.0");
    // Check it's specifically -0.0 (sign bit preserved)
    assert!(
        result.is_sign_negative(),
        "sqrt(-0.0) should be -0.0 (sign bit preserved per IEEE 754), got +0.0"
    );
}

#[test]
fn sqrt_of_negative_is_nan() {
    let result = eval1(&Expr::val().sqrt(), -1.0);
    assert!(result.is_nan(), "sqrt(-1) should be NaN, got {}", result);
}

#[test]
fn sqrt_of_inf_is_inf() {
    let result = eval1(&Expr::val().sqrt(), f64::INFINITY);
    assert!(result.is_infinite() && result > 0.0, "sqrt(+Inf) should be +Inf, got {}", result);
}

#[test]
fn sqrt_of_nan_is_nan() {
    let result = eval1(&Expr::val().sqrt(), f64::NAN);
    assert!(result.is_nan(), "sqrt(NaN) should be NaN, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// Ln edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn ln_of_zero_is_neg_inf() {
    let result = eval1(&Expr::val().ln(), 0.0);
    assert!(
        result.is_infinite() && result < 0.0,
        "ln(0) should be -Inf, got {}",
        result
    );
}

#[test]
fn ln_of_negative_is_nan() {
    let result = eval1(&Expr::val().ln(), -1.0);
    assert!(result.is_nan(), "ln(-1) should be NaN, got {}", result);
}

#[test]
fn ln_of_inf_is_inf() {
    let result = eval1(&Expr::val().ln(), f64::INFINITY);
    assert!(result.is_infinite() && result > 0.0, "ln(+Inf) should be +Inf, got {}", result);
}

#[test]
fn ln_of_subnormal_is_finite_negative() {
    // ln(f64::MIN_POSITIVE * 0.5) — subnormal, very small positive
    let x = f64::MIN_POSITIVE * 0.5; // this IS subnormal
    let result = eval1(&Expr::val().ln(), x);
    // ln of a tiny positive number is a large negative number, but finite
    assert!(
        result.is_finite() && result < -700.0,
        "ln(subnormal) should be a large negative finite number, got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// Recip edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn recip_of_zero_is_inf() {
    let result = eval1(&Expr::val().recip(), 0.0);
    assert!(
        result.is_infinite() && result > 0.0,
        "1/0 should be +Inf, got {}",
        result
    );
}

#[test]
fn recip_of_neg_zero_is_neg_inf() {
    // IEEE 754: 1/(-0.0) = -Inf
    let result = eval1(&Expr::val().recip(), -0.0_f64);
    assert!(
        result.is_infinite() && result < 0.0,
        "1/(-0.0) should be -Inf (IEEE 754), got {}",
        result
    );
}

#[test]
fn recip_of_inf_is_zero() {
    let result = eval1(&Expr::val().recip(), f64::INFINITY);
    assert_eq!(result, 0.0, "1/+Inf should be 0, got {}", result);
}

#[test]
fn recip_of_nan_is_nan() {
    let result = eval1(&Expr::val().recip(), f64::NAN);
    assert!(result.is_nan(), "1/NaN should be NaN, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// Pow edge cases (uses Rust powf internally — documents behavior)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pow_zero_zero_is_one() {
    // IEEE 754 defines 0^0 = 1. Rust's powf follows this.
    let result = eval2(&Expr::Pow(Box::new(Expr::Val), Box::new(Expr::Val2)), 0.0, 0.0);
    assert_eq!(result, 1.0, "0^0 should be 1.0 (IEEE 754 convention), got {}", result);
}

#[test]
fn pow_negative_half_is_nan() {
    // (-1)^0.5 = sqrt(-1) = NaN
    let result = eval2(&Expr::Pow(Box::new(Expr::Val), Box::new(Expr::Val2)), -1.0, 0.5);
    assert!(
        result.is_nan(),
        "(-1)^0.5 should be NaN (sqrt of negative), got {}",
        result
    );
}

#[test]
fn pow_inf_zero_is_one() {
    // IEEE 754: Inf^0 = 1
    let result = eval2(&Expr::Pow(Box::new(Expr::Val), Box::new(Expr::Val2)), f64::INFINITY, 0.0);
    assert_eq!(result, 1.0, "Inf^0 should be 1.0 (IEEE 754), got {}", result);
}

#[test]
fn pow_one_inf_is_nan() {
    // IEEE 754: 1^Inf is NaN (indeterminate form)
    // Rust's powf(1.0, f64::INFINITY) = 1.0 (some libms differ here)
    let result = eval2(&Expr::Pow(Box::new(Expr::Val), Box::new(Expr::Val2)), 1.0, f64::INFINITY);
    // Document the actual behavior — either 1.0 or NaN is defensible.
    // The tambear-libm will need to make an explicit choice and document it.
    assert!(
        result == 1.0 || result.is_nan(),
        "1^Inf should be 1.0 or NaN (implementation-defined), got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// TBS Min/Max NaN propagation (binary op versions)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn tbs_min_nan_x_is_nan() {
    // Expr::Min(NaN, 5.0) — the Val is NaN, Val2 is 5.0
    // Current impl: if va <= vb { va } else { vb }
    // NaN <= 5.0 is false; NaN >= 5.0 is false → falls to else → returns vb = 5.0.
    // WRONG: should propagate NaN.
    let result = eval2(&Expr::Min(Box::new(Expr::Val), Box::new(Expr::Val2)), f64::NAN, 5.0);
    assert!(
        result.is_nan(),
        "Min(NaN, 5.0) should be NaN (NaN propagation), \
         current impl returns {} (SILENT FAILURE: NaN ignored by <= comparison)",
        result
    );
}

#[test]
fn tbs_max_nan_x_is_nan() {
    let result = eval2(&Expr::Max(Box::new(Expr::Val), Box::new(Expr::Val2)), f64::NAN, 5.0);
    assert!(
        result.is_nan(),
        "Max(NaN, 5.0) should be NaN (NaN propagation), \
         current impl returns {} (SILENT FAILURE: NaN ignored by >= comparison)",
        result
    );
}

#[test]
fn tbs_min_x_nan_is_nan() {
    // Second argument is NaN
    let result = eval2(&Expr::Min(Box::new(Expr::Val), Box::new(Expr::Val2)), 5.0, f64::NAN);
    assert!(
        result.is_nan(),
        "Min(5.0, NaN) should be NaN, got {}",
        result
    );
}

#[test]
fn tbs_max_x_nan_is_nan() {
    let result = eval2(&Expr::Max(Box::new(Expr::Val), Box::new(Expr::Val2)), 5.0, f64::NAN);
    assert!(
        result.is_nan(),
        "Max(5.0, NaN) should be NaN, got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════
// Atan2 edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn atan2_zero_zero_is_zero() {
    // IEEE 754: atan2(0, 0) = 0. (Rust follows this.)
    let result = eval2(&Expr::val().atan2(Expr::val2()), 0.0, 0.0);
    assert_eq!(result, 0.0, "atan2(0, 0) should be 0 per IEEE 754, got {}", result);
}

#[test]
fn atan2_nan_propagates() {
    let result = eval2(&Expr::val().atan2(Expr::val2()), f64::NAN, 1.0);
    assert!(result.is_nan(), "atan2(NaN, 1) should be NaN, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// Exp edge cases (via the Expr tree — calls Rust's f64::exp for now)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn exp_large_positive_is_inf() {
    // exp(1000) overflows to +Inf
    let result = eval1(&Expr::val().exp(), 1000.0);
    assert!(
        result.is_infinite() && result > 0.0,
        "exp(1000) should be +Inf (overflow), got {}",
        result
    );
}

#[test]
fn exp_large_negative_is_zero() {
    // exp(-1000) underflows to 0.0
    let result = eval1(&Expr::val().exp(), -1000.0);
    assert_eq!(result, 0.0, "exp(-1000) should be 0 (underflow), got {}", result);
}

#[test]
fn exp_neg_zero_is_one() {
    // exp(-0.0) = 1.0 (e^0 = 1, regardless of sign of zero)
    let result = eval1(&Expr::val().exp(), -0.0_f64);
    assert_eq!(result, 1.0, "exp(-0.0) should be 1.0, got {}", result);
}

#[test]
fn exp_nan_is_nan() {
    let result = eval1(&Expr::val().exp(), f64::NAN);
    assert!(result.is_nan(), "exp(NaN) should be NaN, got {}", result);
}

#[test]
fn exp_inf_is_inf() {
    let result = eval1(&Expr::val().exp(), f64::INFINITY);
    assert!(
        result.is_infinite() && result > 0.0,
        "exp(+Inf) should be +Inf, got {}",
        result
    );
}

#[test]
fn exp_neg_inf_is_zero() {
    let result = eval1(&Expr::val().exp(), f64::NEG_INFINITY);
    assert_eq!(result, 0.0, "exp(-Inf) should be 0.0, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// IsFinite — is it correct on specials?
// ═══════════════════════════════════════════════════════════════════

#[test]
fn is_finite_on_nan_is_zero() {
    let result = eval1(&Expr::IsFinite(Box::new(Expr::Val)), f64::NAN);
    assert_eq!(result, 0.0, "IsFinite(NaN) should be 0.0, got {}", result);
}

#[test]
fn is_finite_on_inf_is_zero() {
    let result = eval1(&Expr::IsFinite(Box::new(Expr::Val)), f64::INFINITY);
    assert_eq!(result, 0.0, "IsFinite(+Inf) should be 0.0, got {}", result);
}

#[test]
fn is_finite_on_normal_is_one() {
    let result = eval1(&Expr::IsFinite(Box::new(Expr::Val)), 42.0);
    assert_eq!(result, 1.0, "IsFinite(42.0) should be 1.0, got {}", result);
}

#[test]
fn is_finite_on_subnormal_is_one() {
    // Subnormals are finite (not Inf, not NaN)
    let result = eval1(&Expr::IsFinite(Box::new(Expr::Val)), f64::MIN_POSITIVE * 0.5);
    assert_eq!(result, 1.0, "IsFinite(subnormal) should be 1.0, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════
// Abs edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn abs_of_nan_is_nan() {
    let result = eval1(&Expr::val().abs(), f64::NAN);
    assert!(result.is_nan(), "abs(NaN) should be NaN, got {}", result);
}

#[test]
fn abs_of_neg_inf_is_pos_inf() {
    let result = eval1(&Expr::val().abs(), f64::NEG_INFINITY);
    assert!(
        result.is_infinite() && result > 0.0,
        "abs(-Inf) should be +Inf, got {}",
        result
    );
}

#[test]
fn abs_of_neg_zero_is_pos_zero() {
    // IEEE 754: abs(-0.0) = +0.0
    let result = eval1(&Expr::val().abs(), -0.0_f64);
    assert_eq!(result, 0.0, "abs(-0.0) should be 0.0");
    assert!(
        result.is_sign_positive(),
        "abs(-0.0) should be +0.0 (not -0.0), sign bit should be cleared"
    );
}
