# Trig/libm Family — Pre-Move Audit

**Date**: 2026-04-22  
**Scope**: `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` + adversarial tests  
**Status**: Files currently in wrong repo (winrapids not tambear). Audit done
before they land anywhere they'll be committed.

---

## Gap 1 (Blocking): Wrong repo — files will never compile or run

All libm implementations and trig adversarial test files are in
`R:\winrapids\crates\tambear\` but tambear's cargo workspace is at
`R:\tambear\`. Pathmaker has been notified. Files must be moved before commit.
All files are untracked — nothing lost.

---

## Gap 2 (High): Platform oracle — tests compare against themselves

The adversarial test files use `x.sin()`, `x.cos()`, `x.tan()` as the oracle
throughout. This is the Rust platform implementation (which calls the system
libm). We are implementing sin/cos/tan from first principles specifically
because we want BETTER control than system libm — using it as ground truth
defeats this.

**Specific cases where this produces wrong confidence**:

`sin_355_hard_case` (line 157-165 of `trig_adversarial.rs`):
```rust
let expected = 355.0_f64.sin();
let d = ulps_between(got, expected);
assert!(d <= 4, ...);
```

`sin(355)` is the canonical range-reduction stress test. `355/113 ≈ π` to
6 decimal places, so `355 mod π ≈ -0.0007963...` — a very small reduced
argument. A Cody-Waite implementation with insufficient π/2 precision returns
garbage here (e.g., sin(355) ≈ -0.15 instead of the correct ~-7.96e-4).

But the test would PASS even if our implementation is wrong — if it agrees
with the platform's libm, the ulps distance is 0 regardless of whether either
of them is correct. Two wrong implementations agreeing is not correctness.

**The correct oracle for sin(355)**:
```
355 - 113π = 355 - 355.0000000... = ?
More precisely: 355 - 113 × 3.14159265358979... = -7.96326710733...e-4
sin(355) ≈ sin(-7.96326710733e-4) ≈ -7.96326710733e-4 (to many digits)
```
The expected value is computable to arbitrary precision from:
```python
import mpmath; mpmath.mp.dps = 50
mpmath.sin(355)  # = -7.9632671073326...e-4
```

The current test CANNOT detect a Cody-Waite implementation that agrees with
a buggy system libm, both failing on this input.

**Affected tests** (by pattern, not exhaustive):
- `sin_pi_matches_platform`
- `sin_payne_hanek_regime`  
- `sin_355_hard_case`
- `cos_half_pi_matches_platform`
- `cos_355_hard_case`
- `tan_pi_matches_platform`
- `tan_at_f64_pi_over_2_is_very_large` (partial — checks magnitude correctly,
  but ulps comparison at end still uses platform)

**Fix**: For each hard case, compute the expected value via mpmath at 50+
digits and hardcode it as a f64 literal (which is itself correctly-rounded
to 53 bits). The platform oracle is only acceptable for sanity checks on
trivial inputs like `sin(0.5)` where both implementations will be correct.

---

## Gap 3 (Medium): Accuracy claim vs test tolerance mismatch

`sin.rs` doc claims "worst-case ≤ 2 ulps" at the entry point. The internal
tests in `sin.rs` use budget of 4 ulps (`check_sin` / `check_cos`). The
adversarial tests use 64 ulps for Payne-Hanek regime.

If the doc claim is ≤ 2 ulps, every test must use ≤ 2 as the budget. Using
4 in the unit tests and 64 in the adversarial tests means the stated accuracy
is neither verified nor falsifiable by the test suite.

**Fix**: Choose one accuracy claim and make EVERY test budget consistent with
it. If the Payne-Hanek regime genuinely requires up to 64 ulps, the doc
comment must say so explicitly instead of claiming ≤ 2 unconditionally.

---

## Gap 4 (Low): Three entry points are identical, claims differ

`sin_strict`, `sin_compensated`, `sin_correctly_rounded` all call `sin_strict`.
The doc says "correctly-rounded. Worst-case ≤ 1 ulp on tested samples." These
claims will all be indistinguishable in practice since all three return the
same value. When the JIT plumbs `correctly_rounded` separately, it will need
an actual double-double kernel to honor the claim. The current aliases create
false confidence that differentiation exists.

---

## Implementation Observations (non-gap)

- `scalbn` for moderate n: correct. Edge cases at n = -1022, -1023 handled
  correctly by recursive guards.
- `payne_hanek_finalize` underflow: protected by the `has_nonzero` check in
  `payne_hanek_core` before calling finalize. Not a runtime panic risk.
- `SIN_COEFFS` S1 = -1.6666...e-1 = -1/6 ✓, `COS_COEFFS` C1 = 4.1666...e-2
  = 1/24 ✓ (matching Taylor series leading terms).
- Quadrant fixup table in `eval_sincos` comments is correct.
- `sin_is_odd` and `cos_is_even` identity tests: good structural tests that
  would catch sign-handling bugs in quadrant fixup.

---

## Summary

| Gap | Severity | Blocks commit? |
|---|---|---|
| Wrong repo — files not in tambear | Blocking | Yes |
| Platform oracle — can't detect shared bugs | High | No, but weakens all hard-case tests |
| Accuracy claim vs tolerance mismatch | Medium | No |
| Three identical entry points | Low | No |

The implementation STRUCTURE is sound (Remez coefficients, Cody-Waite,
Payne-Hanek). The adversarial test METHODOLOGY has a systematic oracle problem
that must be fixed before the test suite provides real confidence.
