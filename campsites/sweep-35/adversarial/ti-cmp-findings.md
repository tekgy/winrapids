# Adversarial — TI-CMP findings + Phase A/B audit high-severity notes

**Date**: 2026-05-10  
**Status (TI-CMP)**: Fixed and green (1629 tests, 0 regressions)  
**Status (Phase A/B)**: Two new HIGH/MEDIUM findings added below — NOT yet fixed

---

## NEW: Finding A-3 (HIGH) — `normal_sqrt_multilimb` silent Inf for extreme-exponent input

**File**: `R:\tambear\crates\tambear\src\primitives\big_float\arith.rs` ~line 1589

**Bug**: When a Normal BigFloat has `exponent = i64::MIN`, the scaling path computes
`k = exponent.saturating_neg()` which gives `i64::MAX` (since `-i64::MIN` overflows).
Then `a_scaled.exponent = i64::MIN + i64::MAX = -1`, and `a_scaled.to_f64().sqrt()` is finite.
BUT: the guard `g.exponent = g.exponent.saturating_sub(k / 2)` subtracts `i64::MAX / 2`
from the sqrt result's exponent, which saturates to i64::MIN. A Normal with exponent i64::MIN
passed to `to_f64()` returns 0.0 (below f64 range). `0.0.sqrt() = 0.0`. So in this
specific subcase the result might actually be 0 (the correct underflow behavior).

**Revised severity**: On closer analysis, the specific `i64::MIN` input reaches `a_scaled.exponent = a.exponent.saturating_add(k) = i64::MIN + i64::MAX = -1` (not i64::MAX), so `a_scaled.to_f64()` is a normal f64 and Newton converges correctly for values near 2^-1. After `g.exponent -= k/2 = i64::MAX/2 ≈ 4.6e18`, the result exponent saturates at i64::MIN again, so the output is a near-zero Normal. This may actually be correct.

**However** — `i64::MAX` input is the dangerous case: `a.exponent = i64::MAX`, scaling sets `k = -i64::MAX` (= i64::MIN via saturating_neg), then `a.exponent.saturating_add(k) = i64::MAX + i64::MIN = -1` again. Similar shape. Both extreme cases converge to a result with saturated exponent, which `canonicalize_overflow_check` maps to either ±Inf or ±0. Whether this is the correct answer for those inputs is a separate question — but the answer is NOT silently plausible-wrong, it saturates to boundary values.

**Actual verdict**: The saturating chain is likely correct by construction (boundary saturation, not silent wrong-answer). Downgrade from HIGH to MEDIUM-NEEDS-VERIFICATION. Add a specific adversarial test.

---

## NEW: Finding A-4 (MEDIUM) — `add` zero-arm canonical-form break

**File**: `R:\tambear\crates\tambear\src\primitives\big_float\arith.rs` lines ~211-220

**Bug**: When `self.is_zero() && !rhs.is_zero()`, the code does:
```rust
let mut out = rhs.clone();
out.precision_bits = result_precision;
return out;
```
If `result_precision > rhs.precision_bits`, `out` is a Normal BigFloat with
`precision_bits` claiming a higher width than `limbs.len()` can hold.
`ceil(result_precision / 64) > rhs.limbs.len()`.

**Silent failure mode**: Any downstream consumer that trusts `limbs.len() == ceil(precision_bits / 64)`
will read fewer limbs than expected and treat the missing limbs as zero — silently computing with a
truncated mantissa while believing it has full precision. `to_f64()` uses `n_limbs = ceil(p/64)` to
know the top limb index — if it assumes more limbs than are present, it might index out of bounds
(panic in debug, UB in release). Actually: `to_f64()` reads `limbs[top_limb]` where
`top_limb = (precision_bits - 1) / 64` — if `limbs.len() < top_limb + 1`, this panics in debug.

**Trigger**: `BigFloat::from_f64(x, 107).add(&BigFloat::from_f64(y, 200), rounding)` where
`from_f64(x, 107)` happens to be zero. The result is a clone of `y` (normal, p=200) with
`precision_bits = 200` — but wait, `rhs` was already at p=200, so `result_precision = max(107, 200) = 200 = rhs.precision_bits`. No break in this case.

**Trigger (actual)**: `BigFloat::from_f64(0.0, 200).add(&BigFloat::from_f64(y, 107), rounding)`.
Here `result_precision = max(200, 107) = 200`, and `rhs` is a 107-bit BigFloat (2 limbs).
`out.precision_bits = 200` but `out.limbs.len() = 2` (ceil(107/64) = 2). 
`ceil(200/64) = 4`. Gap: 4 expected, 2 actual. `to_f64()` would access `limbs[3]` — panic.

**Status**: OPEN. This is a real panic vector, not just a theoretical tameness issue.

---

# Prior TI-CMP findings (unchanged below this line)
---

## TI-CMP-1 — ieee_eq: precision_bits in Normal arm (FIXED)

**Date**: 2026-05-10
**Status**: Fixed and green (1629 tests, 0 regressions)

---

## TI-CMP-1 — ieee_eq: precision_bits in Normal arm (FIXED)

**File**: `R:\tambear\crates\tambear\src\primitives\big_float\cmp.rs`

**Bug**: `ieee_eq` for two Normal BigFloats included `self.precision_bits == other.precision_bits`
in the equality check. IEEE 754 equality is value-semantic, not representation-semantic:
`BigFloat::from_f64(1.0, 53).ieee_eq(&BigFloat::from_f64(1.0, 106))` returned `false`.

**Silent failure mode**: recipes that compare results across precision tiers (e.g., checking
that a P0F64-precision intermediate equals a P2BigFloat computation rounded down) would
silently get `false` even for identical values. No panic, no error — just wrong.

**Fix**: Replaced the precision-inclusive Normal arm with `normal_values_equal(a, b)`, a
bit-aligned comparison that handles differing limb counts by checking that the higher-precision
value's extra low-order bits are all zero and its upper bits match the lower-precision value.

**Tests added**:
- `ieee_eq_same_value_different_precision_should_be_equal` — 1.0@p=53 vs 1.0@p=106
- `ieee_eq_same_value_different_precision_negative` — -3.14@p=53 vs -3.14@p=200

---

## TI-CMP-2 — total_cmp: negative-NaN payload ordering inverted (FIXED)

**File**: `R:\tambear\crates\tambear\src\primitives\big_float\cmp.rs`

**Bug**: Phase 3 of `total_cmp` reversed the `inner_order` for all kinds when `same_sign=true`
(negative bucket). This reversal is correct for Normal values (larger magnitude = smaller negative
number), but wrong for NaN payload ordering. For negative NaNs, smaller payload = more negative bit
pattern = should sort first — reversing `a.cmp(b)` made larger payload sort first, which is the
opposite of `f64::total_cmp` convention.

**Fix**: Separated the Normal and NaN cases in phase 3. Normals: apply sign-based reversal.
NaNs: never reverse (payload comparison is already correct for both positive and negative NaN buckets).

**Test added**:
- `total_cmp_neg_nan_payload_ordering_matches_f64_convention` — neg-NaN(payload=1) < neg-NaN(payload=2)

---

## TI-CMP-3 — `normal_values_equal` usize underflow on same-limb-count different-precision inputs

**File**: `R:\tambear\crates\tambear\src\primitives\big_float\cmp.rs`
**Found**: idle audit after Task #8 completion (2026-05-10)
**Status**: FIXED

**Bug**: `normal_values_equal` (introduced by TI-CMP-1 fix) assigned `(lo, hi)` by `limbs.len()`.
When two BigFloats have the same limb count but different `precision_bits` (e.g., p=65 and p=127
both use 2 limbs), the tie-breaking was arbitrary. If the higher-precision value ended up as `lo`,
`extra_bits = p_hi - p_lo` underflowed as usize — panic in debug, wrap-to-garbage in release.

**Example**: `ieee_eq(1.0@p=65, 1.0@p=127)` — both have 2 limbs, `a.limbs.len() <= b.limbs.len()`
triggers `lo=a(p=65), hi=b(p=127)` only if `a` is the p=65 one. If `a` is p=127, `lo=a(p=127),
hi=b(p=65)`, `extra_bits = 65 - 127` underflows.

**Fix**: assign by `precision_bits` instead of `limbs.len()`.

**Tests added**:
- `ieee_eq_same_limb_count_different_precision_does_not_panic` — was panicking, now passes.
- `ieee_eq_same_limb_count_different_precision_different_values` — was passing (early exit), still passes.

**Connection to TI-CMP-1**: TI-CMP-1 introduced `normal_values_equal`. TI-CMP-3 was a latent bug
in that new function — the antibody for TI-CMP-1 (cross-precision ieee_eq tests) revealed the attack
surface; the idle audit found the specific crack.

---

## Summary

All three bugs were in `cmp.rs`, which was NOT in the prior session's tameness audit scope.
The audit covered `big_float/ty.rs`, `arith.rs`, `limbs.rs`, `conversions.rs`, and `jit/fingerprint.rs`.
`cmp.rs` was clean-swept this session: 26 tests passing, 0 failing.
