# Lab Notebook 003 — Observer: tambear-sweep31-finish

**Date**: 2026-05-08 (continuing session, ~00:30 after lab-notebook-002)
**Role**: Observer (scientific conscience, peer-review mindset)
**Status**: Auditing pending bug tasks (#8-#11) filed by other agents

---

## Context

Four bugs were filed as pending tasks (#8-#11) while reviewing the unstub commit e2e8fb2. The pathmaker note at campsite timestamp 2026-05-09T00:19:02Z says "RESUMING" and "Fixing now" for these bugs. No new commit has landed yet. This notebook audits the current code state against each bug claim.

---

## Observation 1 — Task #8 Status: ALREADY FIXED in e2e8fb2

**Bug description**: "cancellation/borrow logic in normal_add_multilimb mishandles small > large via hidden round/sticky bits."

**What the bug was**: When `cmp_limbs(&large_aligned, &small_aligned)` returns `Equal` AND `small` has non-zero round/sticky below its LSB, the original code took the `Greater` arm. That arm subtracts 1 ulp from diff (to account for small's below-LSB contribution), but diff is already 0 (large == small). Subtracting 1 from an all-zero limb vec underflows to all-1s, producing a huge wrong magnitude with wrong sign.

**Confirmed witness from campsite**: "1.999... + (-1.999...01) yielded 2.66e36 (Normal, sign=+, exp=121). Expected: tiny negative."

**Fix in e2e8fb2** (arith.rs lines 720-728):
```rust
let cmp_with_hidden = match limbs::cmp_limbs(&large_aligned, &small_aligned) {
    std::cmp::Ordering::Equal if round_bit != 0 || sticky_bit != 0 => {
        std::cmp::Ordering::Less
    }
    other => other,
};
match cmp_with_hidden { ... }
```
The `Equal + nonzero-hidden` case is rerouted to `Less`, where `diff = sub_limbs(small, large) = 0` and round/sticky pass through unchanged. Result: zero limbs with small round/sticky below → canonicalize_and_round returns the correct tiny value.

**Verified**: `cargo test --test big_float_multilimb_proptest -- cancel`: 2 passed. `proptest_multilimb_add_cancel_rne`: 1 passed (sweeps all finite positive f64 normals with arbitrary low limb bits). The bug does not reproduce in the current code.

**Task #8 status**: RESOLVED by e2e8fb2. The task list is stale on this one.

---

## Observation 2 — Task #10 Status: ALREADY FIXED in e2e8fb2

**Bug description**: "NaN payload dropped in div, inconsistent with add/mul/sqrt."

**Current code** (arith.rs lines 319-338):
```rust
// NaN propagation — preserve payload (consistent with add/mul/sqrt).
if let BigFloatKind::NaN { payload } = self.kind {
    return Self { kind: BigFloatKind::NaN { payload }, ... };
}
if let BigFloatKind::NaN { payload } = rhs.kind {
    return Self { kind: BigFloatKind::NaN { payload }, ... };
}
```
NaN payload is preserved in both the numerator-NaN and denominator-NaN cases.

The earlier version of div used `if self.is_nan() || rhs.is_nan() { return Self::nan(result_precision); }` which DID drop the payload. The current code has been updated to match add/mul/sqrt's payload-preserving pattern.

**Task #10 status**: RESOLVED by e2e8fb2. Task list is stale.

---

## Observation 3 — Task #11 Status: CONFIRMED BUG, NOT YET FIXED

**Bug description**: "newton_reciprocal seed overflow when 1.0/b_f64 = ±Inf at f64-subnormal boundary."

**The bug**: In `newton_reciprocal` (arith.rs, `if` arm of the initial-guess computation):
```rust
let initial = if b_f64 != 0.0 && b_f64.is_finite() {
    BigFloat::from_f64(1.0 / b_f64, target_p.max(53))
} else { ... };
```

When `b_f64` is a positive f64 subnormal (e.g., `5e-324 = 2^(-1074)`), it is non-zero and finite, so the condition passes. But `1.0 / 5e-324 = Inf` in f64 (the reciprocal overflows). `BigFloat::from_f64(Inf, target_p)` produces a BigFloat with `kind = Infinity`.

The Newton iteration then uses `x = Infinity` as the initial guess. With Normal `b`:
- `b * Infinity = Infinity` (any Normal times Infinity = Infinity by IEEE 754)
- `2 - Infinity = -Infinity`
- `Infinity * (-Infinity) = -Infinity`
- All subsequent iterations stay at -Infinity (or NaN if +Inf * -Inf = -Inf then 2 - (-Inf) = +Inf then +Inf * +Inf = +Inf cycles)

The Newton iteration diverges. The division of any `a` by a subnormal `b` returns a wrong result (likely Infinity or NaN instead of the large-but-finite correct value).

**When does this trigger?**: `b.to_f64()` is a subnormal only when `b`'s value in f64 falls in the range (0, 2^(-1022)). Since BigFloat can represent values far below f64's minimum normal, this is reachable for BigFloats constructed with exponents below -1022. The existing test vectors (`1e100`, `1e-100`, `1.0`, `3.14`, etc.) all produce f64-range normals. The bug is in an untested code path.

**The fix**: Add a check that `1.0 / b_f64` is finite before using the fast path:
```rust
let initial = if b_f64 != 0.0 && b_f64.is_finite() {
    let recip_f64 = 1.0 / b_f64;
    if recip_f64.is_finite() {
        BigFloat::from_f64(recip_f64, target_p.max(53))
    } else {
        // Fall through to scaled-seed path
        // (insert else branch logic here)
    }
} else { ... };
```
Or, cleaner: separate the condition: `if b_f64 != 0.0 && b_f64.is_finite() && (1.0 / b_f64).is_finite()`.

**Task #11 status**: CONFIRMED BUG, present in e2e8fb2, NOT YET FIXED.

---

## Observation 4 — Task #9 Analysis: LIKELY STALE or MISIDENTIFIED

**Bug description**: "sign of exp_shift in newton_reciprocal scaled-seed unscaling at arith.rs:1242."

**The code at line 1245-1252**:
```rust
let exp_shift = b.exponent - scaled_exp;  // = b.exponent - 0 = b.exponent
b_scaled.exponent = scaled_exp;           // = 0
let recip_scaled = 1.0 / b_scaled.to_f64();
let mut recip = BigFloat::from_f64(recip_scaled, target_p.max(53));
// Comment: b_scaled = b * 2^(-exp_shift), so 1/b = recip_scaled * 2^(-exp_shift)
if recip.is_normal() {
    recip.exponent -= exp_shift;
}
```

**Mathematical verification**:
- BigFloat convention: `value = M · 2^(exponent - precision_bits + 1)` with top bit of M at position precision_bits-1.
- Setting `b_scaled.exponent = 0` while keeping limbs (and precision) unchanged: `b_scaled = M_b · 2^(0 - p_b + 1)` while `b = M_b · 2^(b.exponent - p_b + 1)`. So `b_scaled = b · 2^(-b.exponent)`.
- `exp_shift = b.exponent`. So `b_scaled = b · 2^(-exp_shift)`.
- `recip_scaled = 1/b_scaled = 2^(exp_shift) / b`.
- `1/b = recip_scaled · 2^(-exp_shift)`.
- Applying `2^(-exp_shift)` to BigFloat: subtract exp_shift from exponent. `recip.exponent -= exp_shift`. Correct.

**Concrete check with huge b (b.exponent = 10000)**:
- `exp_shift = 10000`.
- `b_scaled.exponent = 0` → b_scaled ≈ 1.0 (top bit of M_b is at position p_b-1, so M_b ≈ 2^(p_b-1), and b_scaled = M_b · 2^(1-p_b) ≈ 1.0).
- `recip_scaled ≈ 1.0`. `recip.exponent ≈ 0`.
- After: `recip.exponent = 0 - 10000 = -10000`.
- `1/b` should have exponent ≈ `-(b.exponent - p_b + 1) - (p_b - 1)` = `-b.exponent` = -10000. Correct.

**Concrete check with tiny b (b.exponent = -10000)**:
- `exp_shift = -10000`.
- `b_scaled ≈ 1.0` (same analysis).
- `recip_scaled ≈ 1.0`. `recip.exponent ≈ 0`.
- After: `recip.exponent = 0 - (-10000) = 10000`.
- `1/b` should have exponent ≈ 10000. Correct.

**Conclusion**: The sign of exp_shift is mathematically correct. Task #9 was likely filed during a period when the code had the wrong sign, and the current e2e8fb2 has it correct (or the task was filed against a hypothetical misread of the code). The Newton iteration will correct any initial-guess error in ⌈log₂(p/53)⌉ + 2 steps regardless.

**Caveat**: The `else` branch (scaled-seed path) is only triggered when `b_f64 == 0.0 || !b_f64.is_finite()`. This path is NEVER triggered by any current test (as noted in Observation 3). If there's a bug here, there's no test that would catch it. Task #9 may be a precautionary flag on untested code. My analysis says the math is correct, but I cannot rule out implementation errors in edge cases (e.g., what if `b_scaled.to_f64()` is also 0 or Inf after the exponent adjustment? That's possible if b has a much larger exponent than expected).

**Task #9 status**: Likely RESOLVED in e2e8fb2 based on mathematical analysis. Cannot be confirmed by test (code path is untested). The real risk is Task #11's issue — fix that first, then the `else` branch becomes reachable for subnormal b, which will exercise the exp_shift logic.

---

## Observation 5 — Compound Bug: #11 + #9 interaction

When Task #11's fix routes subnormal-b cases to the `else` branch, the `else` branch will actually run. At that point, the correctness of the exp_shift logic (Task #9) becomes testable. My analysis says it's correct, but it hasn't been exercised. 

The right sequence: fix Task #11 first (guard the fast path), then write a test that exercises the `else` branch (b with exponent beyond f64's range), then verify the exp_shift logic produces the right initial seed for Newton.

---

## Observation 6 — Test Coverage Gap for Extreme-Exponent Division

**None of the current integration tests construct a BigFloat with exponent outside f64's representable range for division.** The `multilimb_div_extreme_magnitude_newton_convergence` test uses `1e100` and `1e-100` — both representable as finite f64, so `b.to_f64()` is always finite. The untested regime: b with `|b.exponent| >> 1023`.

To construct such a b:
```rust
let b_huge = BigFloat::from_raw_limbs(false, 10000, 200, large_limbs);
let b_tiny = BigFloat::from_raw_limbs(false, -10000, 200, large_limbs);
let a = BigFloat::from_f64(1.0, 200);
let result_div_huge = a.div(&b_huge, RNE); // exercises else branch: b_huge.to_f64() = Inf
let result_div_tiny = a.div(&b_tiny, RNE); // exercises else branch: b_tiny.to_f64() = 0.0
```

**This test doesn't exist yet.** It should be added alongside the Task #11 fix.

---

## Summary for Navigator

Active bugs in the current code (e2e8fb2):

**CONFIRMED, UNFIXED**:
- Task #11: `newton_reciprocal` seed overflow when `1.0/b_f64 = ±Inf`. Triggered when `b`'s value underflows f64 (b is subnormal in f64 terms). Fix: add `&& (1.0 / b_f64).is_finite()` to the fast-path guard, or restructure to check after computing the reciprocal.

**LIKELY ALREADY FIXED** (task list stale):
- Task #8: `cmp_with_hidden` fix IS present in e2e8fb2. Cancellation tests pass.
- Task #10: NaN payload preservation in div IS present in e2e8fb2. Pattern matches add/mul/sqrt.
- Task #9: exp_shift sign is mathematically correct per manual verification. Cannot be confirmed by test (code path untested). Recommend adding test when #11 is fixed.

**Missing test**:
- No test exercises `newton_reciprocal` with b having exponent beyond f64 range (the `else` branch is dead code under current tests).

The Task #11 fix is small and isolated. It should be committed before the session closes.
