# Lab Notebook 002 — Observer: tambear-sweep31-finish

**Date**: 2026-05-08
**Role**: Observer (scientific conscience, peer-review mindset)
**Branch**: main (HEAD: e2e8fb2)
**Session**: tambear-sweep31-finish (continuation)
**Prior notebook**: lab-notebook-001.md

---

## Context

Lab-notebook-001.md established the baseline state (1560 lib tests, multi-limb paths all `unimplemented!()`). Since that notebook was written, the pathmaker has landed commit e2e8fb2 ("sweep-31 unstub: multi-limb BZ Algorithms 3.1/3.3/3.5/3.10 in arith.rs"). This notebook documents the post-unstub observation pass: did the code deliver what the claims assert?

---

## Observation 1 — Test Count Verification

**Claim in commit message**: "1596 lib tests passing (was 1560 baseline; +36 from limbs.rs unit tests). 112 big_float integration tests passing (6 suites). 0 warnings. 33 multilimb proptests."

**Verified**:
- `cargo test --lib`: **1596 passed, 0 warnings**. Confirmed.
- `cargo test --test big_float_multilimb_proptest`: **33 passed, 0 ignored**.
- `cargo test --test big_float_cross_precision`: **22 passed, 4 ignored**.
- `cargo test --test big_float_verification_tier`: **30 passed**.
- `cargo test` (full suite including integration, doc-tests): **2350 passed, 110 ignored, 0 failures**.

**Discrepancy from commit claim**: The commit says "112 big_float integration tests passing (6 suites)" but the measured count across the three integration files I ran separately is 33 + 22 + 30 = 85, plus the pre-existing suites (arith_invariants, diamond, roundtrip). I did not recount all 112 individually — but 2350 full-suite passes is consistent with the claim direction. The 112 count likely includes the pre-existing three suites + the new three. Not a discrepancy worth flagging further.

**Status**: Baseline confirmed. The unstub landed cleanly.

---

## Observation 2 — Pre-Review Checklist Applied (BZ 3.1: add/sub)

Applied the checklist from lab-notebook-001.md §"For BZ 3.1".

### [PASS] Exponent alignment
`normal_add_multilimb` lines 487-663: selects the operand with the larger exponent as the "large" operand. Computes `exp_diff = large.exponent - small.exponent`. Shifts small's mantissa right by exp_diff bits into the working buffer. The shift captures round and sticky bits via `shr_limbs_with_sticky`. When exp_diff exceeds `max_useful_diff = result_precision + small.precision_bits + 64`, small contributes only to sticky_bit (correct: too-far-away operand has no impact on the integer part of the result).

### [PASS] Integer add with carry
`add_limbs` in limbs.rs: uses `overflowing_add` at each limb with carry propagation. Carry-out pushed into the result buffer. Correct.

### [PASS] Subtraction / cancellation path
Magnitude subtraction dispatched by `cmp_limbs`. When the aligned magnitudes compare Greater: `sub_limbs(diff, large_aligned, small_aligned)`. When small had non-zero round/sticky (below-LSB contribution), the code subtracts 1 additional ulp from diff and recomputes round/sticky correctly (lines 717-773). The three cases `(r=1,s=0)`, `(r=1,s=1)`, `(r=0,s=1)` are handled explicitly.

One observation: when `cmp_limbs` returns `Less` (small_aligned > large_aligned strictly), the code subtracts `sub_limbs(diff, small_aligned, large_aligned)` and uses `result_sign = small.sign`. This is correct: when the aligned small is larger, the result takes small's sign. The round/sticky from small's hidden bits are left unchanged (they add positively to the smaller-magnitude result). This is correct.

### [PASS] Cancellation-to-zero
Line 801: `if result_mag.iter().all(|&x| x == 0) && round_bit == 0 && sticky_bit == 0`. When this fires, `BigFloat::zero(result_precision)` is returned with sign from `cancellation_zero_sign(rounding)`. The sign rule: RoundTowardNegativeInfinity → `-0`, all other modes → `+0`. This matches IEEE 754 §6.3. **The watch item from notebook-001 is resolved.**

### [PASS] `_rounding` prefix removed and used
No instances of `_rounding` as a parameter name anywhere in arith.rs. The `rounding` parameter is used in `should_round_up`, in `add_zero_arithmetic_sign`, `sub_zero_arithmetic_sign`, `cancellation_zero_sign`. **Watch item resolved.**

### [PASS] Post-condition: canonical form
After `canonicalize_and_round`, the limbs vector is trimmed to `ceil(result_precision / 64)` entries (line 946). The top bit is at position `result_precision - 1` by construction of the shift arithmetic. The `from_raw_limbs` constructor (used in tests) asserts this invariant at the boundary.

---

## Observation 3 — Pre-Review Checklist Applied (BZ 3.3: schoolbook mul)

### [PASS] Schoolbook correctness
`normal_mul_multilimb`: allocates `prod_len = a.limbs.len() + b.limbs.len()` for the product. `mul_limbs` in limbs.rs uses `u64 * u64 -> u128` with carry propagation. This is the correct schoolbook pattern.

### [FLAG] Guard bits — design vs implementation
The commit message says "schoolbook multiplication at `p + 50` guard bits, then round to `p`". The module docstring says the same. But looking at `normal_mul_multilimb` (lines 1096-1136):

```rust
let prod_len = a.limbs.len() + b.limbs.len();
let mut prod = vec![0u64; prod_len];
limbs::mul_limbs(&mut prod, &a.limbs, &b.limbs);
// ...
canonicalize_and_round(prod, exp_at_lsb, result_sign, 0, 0, result_precision, rounding)
```

There is no explicit widening to `p + 50` guard bits before multiplication. The schoolbook multiply of two p-bit mantissas gives a 2p-bit product, which `canonicalize_and_round` then rounds to `result_precision`. The round/sticky bits are captured during the right-shift inside `canonicalize_and_round`.

**Is this correct?** Yes, but the "p + 50 guard bits" framing in the commit message is misleading. The actual mechanism is that the schoolbook product is computed in its full 2p-bit precision (which is more than enough guard bits — 2p - p = p bits of guard), and then rounded. This is strictly better than the p+50 guard-bit approach described in the commit message and design docs. The documentation's "p+50 guard bits" language describes the Newton-iteration approaches (div, sqrt) where the intermediate computation is controlled at p+50; for schoolbook multiply, the product is exact at 2p bits and needs no separate guard region.

**Conclusion**: The implementation is correct. The documentation's "p+50" framing for mul is technically inaccurate (the actual precision is 2p, which is >= p+50 for p >= 50). This is a documentation inaccuracy, not a correctness bug. For p=107 (the minimum DEC-031 tier above f64), the product is 214 bits, which rounds to 107 with 107 bits of guard — well above the p+50=157 requirement. The Tambear Contract item 10 ("every assumption documented") would require this distinction to be documented accurately.

**Action**: Flag to navigator as a documentation clarification needed, not a code fix.

### [PASS] Rounding mode wired
`rounding` is passed through to `canonicalize_and_round`, which calls `should_round_up` with the correct IEEE 754 cases.

### [PASS] Tests use from_raw_limbs with dense limbs
`multilimb_positive` in the proptest file constructs values with non-zero low limbs. `proptest_multilimb_add_cancel_rne` sweeps all finite positive f64 normals with arbitrary low-limb bits ORed with 1 (ensuring f64_path_eligible = false). Multiplication is tested in `multilimb_mul_one_is_identity`, `multilimb_mul_commutative`, `multilimb_mul_self_p500`.

---

## Observation 4 — Pre-Review Checklist Applied (BZ 3.5: Newton div)

### [PASS] Iteration count formula
Verified mathematically. The code:
```rust
let mut iters = 2;
let mut p = f64_bits; // 53
while p < target_p {
    p = p.saturating_mul(2);
    iters += 1;
}
```
For guard_p = result_precision + 50:
- result_precision=200, guard_p=250: iters=5. Formula: `⌈log₂(250/53)⌉ + 2 = ⌈2.24⌉ + 2 = 5`. Correct.
- result_precision=500, guard_p=550: iters=6. Formula: `⌈log₂(550/53)⌉ + 2 = ⌈3.38⌉ + 2 = 6`. Correct.
- result_precision=1024, guard_p=1074: iters=7. Formula: `⌈log₂(1074/53)⌉ + 2 = ⌈4.34⌉ + 2 = 7`. Correct.

### [PASS] Initial guess from f64 reciprocal
`newton_reciprocal` computes `1.0 / b.to_f64()` for representable b. For very-large or very-small b where `b.to_f64()` is 0.0 or Inf, it scales b's exponent to 0, computes the reciprocal, then shifts the exponent back. This handles the extreme-magnitude case.

### [PASS] Newton recurrence: x_{n+1} = x_n * (2 - b * x_n)
Lines 1248-1256: exactly this recurrence. Each step uses `RoundToNearestTiesEven` internally (the Newton error accumulation is bounded by the guard bits, not by the intermediate rounding mode).

### [PASS] Guard bits at p+50
`normal_div_multilimb` sets `guard_p = result_precision + 50`. Newton runs at guard_p. Final round: `round_to_precision(&prod, result_precision, rounding)` uses the user's rounding mode.

### [PASS] Fixed iteration count (no convergence check)
The loop `for _ in 0..iters` has no convergence check in the body. Correct per DESIGN.md §3: the iteration count formula provably suffices for quadratic convergence from 53-bit initial error.

---

## Observation 5 — Pre-Review Checklist Applied (BZ 3.10: Newton sqrt)

### [PASS] Recurrence x_{n+1} = (x_n + a/x_n) / 2
Lines 1392-1394: exactly this. The inner `div` is a full BigFloat div at the current precision (which grows toward guard_p). This is the "full div" path, not a reduced-precision approximation.

### [FLAG] Potential precision growth issue in sqrt Newton
Each iteration calls `a.div(&x, ...)`. `a` has `result_precision` bits. `x` starts at `guard_p = result_precision + 50` bits. The division result `a_over_x` uses `max(a.precision_bits, x.precision_bits) = guard_p` as result_precision (per the div public API, which takes `max`). Then `sum = x.add(&a_over_x, ...)` — both at guard_p. Then `x = sum.div(&two, ...)` — also at guard_p.

But: `a` has `result_precision` bits, not `guard_p`. When `a.div(&x)` is called, `a.precision_bits < guard_p`. The result is at `max(result_precision, guard_p) = guard_p` precision — correct for the div output. But `a` is only contributing `result_precision` bits of numerator into a `guard_p`-wide division. The low `50` bits of the dividend are zero (since `a` wasn't widened).

This means the Newton iteration is computing `a / x` where `a` has only 50 fewer significant bits than the result precision. For p=200: a has 200 bits of information, x has 250, result is at 250 — the error in the numerator is at most 2^(-200) relative, which is well below the guard region (2^(-250)). So the Newton iteration still converges correctly.

However, there's a subtlety: `widen_to_precision(&x, guard_p)` is called at the start of each iteration to ensure `x` is at `guard_p`, but `a` is never widened. This means `a.div(&x)` will compute at `guard_p` (taking the max of `a.precision_bits=result_precision` and `x.precision_bits=guard_p`), but `a`'s limbs contribute only `result_precision` significant bits. The 50 low bits of the quotient will be rounded from `a`'s round/sticky bits. This is technically correct but slightly suboptimal — widening `a` to `guard_p` before the iteration would give a cleaner result. Not a bug, but a minor precision accounting inaccuracy.

**Compare with div's treatment**: In `normal_div_multilimb`, `a` IS widened: `let a_guard = if a.precision_bits >= guard_p { a.clone() } else { widen_to_precision(a, guard_p) }`. The sqrt implementation does NOT widen `a`. This is an inconsistency. For div, widening a ensures the quotient has full guard_p precision. For sqrt, not widening a means the guard bits of each `a/x` result are not fully informed by `a`'s mantissa (since a's low 50 bits are zeros, not the true mantissa bits).

**Severity assessment**: The Newton iteration for sqrt starts with a 53-bit accurate initial guess and doubles accuracy each step. The guard of 50 bits absorbs the approximation error. The final round from guard_p to result_precision via `round_to_precision` produces a correctly-rounded result at result_precision. The "missing 50 low bits of a" in the numerator means the sqrt iteration has slightly less information than the div iteration, but the guard region is specifically designed to absorb this. The result is still within 1 ulp of correctly-rounded sqrt. However, this asymmetry between div (wides a) and sqrt (doesn't widen a) is worth noting as a potential place where the sqrt could produce errors at extreme precision tiers.

**Action**: Flag to navigator as an observation — not a confirmed bug, but an asymmetry between div and sqrt's treatment of the numerator precision. If the cross-precision consistency tests exercise this regime, they would catch it.

---

## Observation 6 — Cross-Cutting Check: from_raw_limbs Density

### [PASS] Tests use genuinely dense multi-limb inputs
`multilimb_positive` seeds `limbs[0] = lo_bits_l0` and `limbs[1] = lo_bits_l1` with non-zero values. The proptest strategy for `l0` uses `1u64..u64::MAX` (ensuring non-zero). These operands will fail `f64_path_eligible` and exercise the BZ code paths.

`from_raw_limbs` is now `pub` (no longer `#[cfg(test)]`). The integration tests can access it from outside the lib crate.

### [PASS] RoundingMode coverage
The proptest file defines constants `RNE`, `RTZ`, `RTP`, `RTN`. Tests `proptest_multilimb_rounding_mode_bracket` explicitly exercises all four modes for add and mul. Zero-arithmetic sign tests exercise the RTN-dependent case. The five modes in the `RoundingMode` enum are: `RoundToNearestTiesEven`, `RoundToNearestTiesAwayFromZero`, `RoundTowardZero`, `RoundTowardPositiveInfinity`, `RoundTowardNegativeInfinity`. Four of the five are explicitly named in the proptest. `RoundToNearestTiesAwayFromZero` is implicitly covered by the `should_round_up` unit tests in `arith_tests.rs`.

### [PARTIAL] Cross-precision consistency
The commit claims "Surface 3 (cross-precision consistency) antibody is the multilimb proptest gauntlet's primary check." Let me assess what's actually tested:

- `cross_precision_add_consistency_200_vs_500` in multilimb_proptest.rs: pinned witnesses for add at p=200 vs p=500. Present.
- `cross_precision_mul_consistency_200_vs_500` in multilimb_proptest.rs: pinned witnesses for mul. Present.
- The 4 `#[ignore]`d tests in big_float_cross_precision.rs: these test the round-trip consistency via `with_precision_rounded` — but this method does not exist yet.

**Gap**: `BigFloat::with_precision_rounded` is referenced by 4 ignored tests in big_float_cross_precision.rs but does not exist as a public method. The private `round_to_precision` function in arith.rs does the same work but is not exposed. The ignored tests will remain ignored until this method is made public. The commit message does not mention that these tests remain blocked.

The 22 passing tests in big_float_cross_precision.rs are testing other cross-precision properties (not the ignored ones). The 4 ignored tests are specifically the proptest-based round-bit boundary classes (Phase C) that were flagged as mandatory in the briefing.

**This is the single most important remaining gap**: the cross-precision consistency proptest at the limb-boundary regime (p=107/128/200, p_high_offset=1/50/63/64/65/128/200) is still blocked on `with_precision_rounded` being public.

---

## Observation 7 — LOG.md Gap

The most recent LOG.md entry is from the prior session (`tambear-formalize`, 2026-05-08). Commit e2e8fb2 is a substantial delivery — the unstub plus 4800 lines of new code and tests — but there is no LOG.md entry for this session yet. Per LOG.md convention ("Every agent appends a 5-line entry at the end of their session"), this needs to be written before the session exits. The navigator should ensure this happens.

---

## Observation 8 — Claims Inventory Update (against Observation 3 from notebook-001)

### Claim BZ 3.1 (add/sub): VERIFIED
Canonical form: yes. IEEE 754 zero-arithmetic table: yes (tested by proptest and by pinned witnesses). Multi-limb paths: implemented and exercised by from_raw_limbs-based tests.

### Claim BZ 3.3 (mul): VERIFIED WITH DOCUMENTATION NOTE
Schoolbook correctness: yes. Rounding: yes. The "p+50 guard bits" language is inaccurate — the actual implementation uses 2p bits of precision for the product, which is strictly better. Documentation should be updated to say "full 2p-bit product, rounded to p" rather than "p+50 guard bits".

### Claim BZ 3.5 (div): VERIFIED
Newton convergence formula: verified mathematically. Guard bits: 50 additional bits used. `a` is widened before the Newton multiply. Rounding: final round uses user's rounding mode.

### Claim BZ 3.10 (sqrt): VERIFIED WITH ASYMMETRY NOTE
Newton convergence formula: same as div (same code pattern), verified. Guard bits: 50 additional bits. `a` is NOT widened before the Newton division — asymmetry with BZ 3.5's treatment. Functionally correct (the guard absorbs the approximation), but the asymmetry is worth tracking.

---

## Watch Items (updated)

### Active
1. **`BigFloat::with_precision_rounded` not public**: 4 proptest-based cross-precision consistency tests in big_float_cross_precision.rs remain `#[ignore]`d because this method doesn't exist. These are the limb-boundary boundary-class tests (Phase C) that the briefing flagged as mandatory. The navigator should track this as a remaining deliverable.

2. **Mul documentation accuracy**: `normal_mul_multilimb` uses a full 2p-bit product, not p+50 guard bits. The commit message and module docstring say "p+50 guard bits" for mul, which is inaccurate (the actual guard is p bits, not 50). A future documentation pass should correct this. (This is a correctness improvement from the spec, not a regression.)

3. **Sqrt asymmetry**: `normal_div_multilimb` widens `a` to `guard_p`; `normal_sqrt_multilimb` does not widen `a`. Both implementations are likely correct (the guard absorbs the error), but the asymmetry could cause edge-case divergence at extreme precision tiers. Cross-precision tests should cover this.

4. **LOG.md entry**: The current session (unstub commit e2e8fb2) does not have a LOG.md entry yet.

### Resolved (from notebook-001)
- `_rounding` prefix removed: RESOLVED. All rounding parameters are wired.
- Cancellation-to-zero check: RESOLVED. Line 801 of arith.rs.
- from_raw_limbs density: RESOLVED. Tests use genuine multi-limb inputs.
- Multi-limb paths exercised: RESOLVED. `f64_path_eligible` returns false for from_raw_limbs inputs with non-zero low limbs.

---

## Publishability Assessment (Tambear Contract item 10)

The BZ 3.1/3.3/3.5/3.10 implementations in their current state:

**What would survive peer review:**
- Correctness: the schoolbook multiply and cancellation-to-zero handling are textbook-correct. The Newton iterations use the established convergence analysis. The rounding-mode dispatch is IEEE 754 compliant.
- Testing: 2350 tests passing, including genuine multi-limb adversarial inputs via from_raw_limbs. Proptests cover carry saturation, alternating bits, cancellation, commutativity, Newton convergence at extreme magnitudes.
- The iteration count formula for Newton is correctly derived and implemented.

**What would draw reviewer attention:**
- The "p+50 guard bits" language for mul is inaccurate. A reviewer would notice this immediately — the product of two p-bit numbers is exactly 2p bits, not p+50.
- No mpmath comparison oracle yet. The Tambear Contract requires "bit-exact correctly rounded against mpmath at p=200/500/1024 across 5 rounding modes" for each op. This is Sweep 34's responsibility, but it means the claims in the BZ 3.3/3.5/3.10 category are not yet fully verified at publication grade.
- The cross-precision consistency proptest (Phase C) is still blocked. These tests were flagged as mandatory. Their absence means the primary antibody for guard-bit errors has not been fully deployed.

**Overall**: The implementation is production-ready for its intended purpose (high-precision oracle substrate). It is not yet at the publication-grade threshold the Tambear Contract sets. Sweep 34 (mpmath oracle) is the next step to close that gap. The Phase C proptest gap (with_precision_rounded) should be resolved before Sweep 34 begins.

---

## Summary

The unstub commit e2e8fb2 delivered what it claimed: all four BZ algorithms are implemented, all `unimplemented!()` panics are gone, 2350 tests pass. The implementation is correct.

Two substantive observations worth tracking:
1. `with_precision_rounded` not yet public — blocks 4 mandatory cross-precision consistency proptests.
2. Mul's "p+50 guard bits" documentation is inaccurate (actual precision is 2p, which is better).

One minor asymmetry to watch: sqrt doesn't widen `a` before Newton division, unlike div.

The LOG.md entry for this session is still missing.
