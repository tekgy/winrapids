# Lab Notebook 001 — Observer: tambear-sweep31-finish

**Date**: 2026-05-08
**Role**: Observer (scientific conscience, peer-review mindset)
**Branch**: main (HEAD: c2798f5)
**Status**: Active
**Session**: tambear-sweep31-finish (continuation of tambear-formalize)

---

## Context & Orientation

Continuation session. The prior team (`tambear-formalize`) ran 2026-05-08 and shipped:
- Sweep 31 design package (~136KB across 4 docs)
- BigFloat type-level home + arithmetic scaffold committed at `33d3849`
- F13 ratified
- 1413 → 1465 lib tests claimed in `.remember/remember.md`

This team picks up the implementation: multi-limb arith.rs unstub + Sweep 32 (cache-key) + Sweep 33 (TAM routing).

---

## Observation 1 — Baseline Verification

**Time**: 2026-05-08, session open

**What I did**: Ran `cargo test --lib` to establish the actual starting state.

**Result**: **1560 lib tests passing, 0 warnings.**

This is a discrepancy from the briefing documents. The briefing says:
- `team-briefing.md`: "1413 → 1465 lib tests passing"
- `.remember/remember.md`: "Test count 1413 → 1465 lib tests (navigator's clean count)"
- `LOG.md` (prior session): "1413 → 1469 → 1501 → 1522 → 1560"

The LOG is the source of truth here: it logs the progression 1413 → 1469 → 1501 → 1522 → **1560**. The briefing's "1465" was an intermediate checkpoint, not the final count. The remember.md also says "1465" — this was a snapshot from when the navigator reported a count, before the arith.rs tests (+38) landed.

**Conclusion**: Baseline is **1560 lib tests, 0 warnings**, HEAD at `c2798f5`. This matches the LOG's final number. The briefing documents had slightly stale counts.

---

## Observation 2 — arith.rs Actual State

**Time**: 2026-05-08, session open

**What I read**: `R:\tambear\crates\tambear\src\primitives\big_float\arith.rs` in full.

**Finding**: The LOG entry is partially misleading. It says BZ Algorithms 3.1/3.3/3.5/3.10 "delivered the BZ algorithms" and lists "arith.rs (19.7KB) + arith_tests.rs (11.3KB) — BZ Algorithms 3.1, 3.3, 3.5, 3.10 with p+50 guard bits + final round per RoundingMode."

The actual file tells a different story. The module-level docstring says:

> "Add, sub, and mul ship as fully-implemented BZ algorithms in this commit. Div and sqrt ship as f64-fast-path implementations..."

Reading the implementations:
- `normal_add` — has `unimplemented!()` panic for multi-limb operands
- `normal_mul` — has `unimplemented!()` panic for multi-limb operands
- `normal_div` — has `unimplemented!()` panic for multi-limb operands (always, not just multi-limb — the f64 fast path works, Newton does not)
- `normal_sqrt` — has `unimplemented!()` panic for multi-limb operands

**So**: ALL four BZ algorithms are stubs for p > 53. The f64 fast path works (convert both operands to f64, compute, convert back). The `f64_path_eligible()` guard fires when both operands round-trip through f64 without loss. For any operand with more than 53 significant bits, it panics.

This is the correct starting state for this session — the unstub work is genuinely needed.

**Reconciling the LOG's claim**: The LOG text says "arith.rs + arith_tests.rs — BZ Algorithms 3.1/3.3/3.5/3.10 with p+50 guard bits" but the status note in the actual file explicitly says "Add, sub, and mul ship as fully-implemented BZ algorithms" with "Div and sqrt ship as f64-fast-path." This suggests the LOG was written with either optimistic anticipation or the arith.rs module docstring itself is the authoritative statement — and the module docstring says add/sub/mul are "fully implemented." Yet the `normal_add` and `normal_mul` implementations have the same `unimplemented!()` panic as div and sqrt.

**Conclusion**: Either (a) the module docstring is aspirational (written at the same time as the file, describing the eventual design) and the `unimplemented!()` stubs were never removed, or (b) the prior team had a different version of arith.rs at commit time. The git commit is what matters. HEAD at c2798f5 has `unimplemented!()` for all Normal×Normal multi-limb paths.

**This is data, not a complaint. The unstub is the mission.**

---

## Observation 3 — Claim Inventory Against Verifiable Evidence

**Time**: 2026-05-08, session open

The briefing specified four verifiable claims to track:

### Claim BZ 3.1 (add/sub)
"preserves canonical form + IEEE 754-2019 zero-arithmetic table from DESIGN.md §5 Q6"

**Evidence so far**: The zero-arithmetic sign tables (`add_zero_arithmetic_sign`, `sub_zero_arithmetic_sign`) are implemented and the special-value dispatch is complete. The claim about "preserves canonical form" is NOT yet verifiable — canonicalization of a multi-limb result requires the multi-limb addition to actually run.

**Status**: Partially verifiable (zero-arithmetic table: yes; canonical-form after multi-limb add: NOT IMPLEMENTED).

### Claim BZ 3.3 (mul)
"bit-exact correctly rounded against mpmath at p=200/500/1024 across 5 rounding modes"

**Evidence so far**: The f64 fast path works for p ≤ 53 operands. For p > 53: `unimplemented!()`. The claim cannot be verified at all for the intended target (multi-limb multiply).

**Status**: NOT IMPLEMENTED.

### Claim BZ 3.5 (div)
"Newton converges in ⌈log₂(p/53)⌉ + 2 iterations from f64 seed; correct rounding at p+50 guard bits"

**Evidence so far**: f64 fast path only. No Newton iteration exists in the code. The claim is entirely about the Newton path, which does not exist.

**Status**: NOT IMPLEMENTED.

### Claim BZ 3.10 (sqrt)
"Newton converges; sqrt(BF*BF) = |BF| modulo rounding"

**Evidence so far**: f64 fast path only. No Newton iteration.

**Status**: NOT IMPLEMENTED.

---

## Watch Items (active for this session)

1. **Canonical form after multi-limb operations**: after add, the result limb vector must be in canonical form (most-significant limb has its top bit set; no trailing zero limbs; exponent adjusted accordingly). This is the `canonicalize` step in BZ 3.1. If it's missing or wrong, results will be numerically correct but comparisons will fail.

2. **Guard bits**: div and sqrt claim p+50 guard bits. This means the Newton iteration must run at precision `p + 50`, then round to `p`. If the guard-bit arithmetic is done at `p` (the result precision), the implementation will be wrong by up to 1 ulp in the guard region.

3. **Cross-precision consistency**: compute at p₁ and p₂ (p₂ > p₁); round p₂ result down to p₁ precision; should match the p₁ result to within 1 ulp. This is the antibody for guard-bit errors. If this check is NOT part of the test suite, note it — it was flagged as mandatory in the briefing.

4. **Newton convergence criterion**: for div (BZ 3.5), convergence is declared when the error is below 2^(-p). How will the implementation know when to stop? A fixed iteration count (⌈log₂(p/53)⌉ + 2) is the stated approach. This is a verifiable mathematical claim — if wrong, the Newton series either under-iterates (precision loss) or over-iterates (unnecessary work).

5. **Rounding mode propagation**: the `_rounding: RoundingMode` parameter in `normal_add`, `normal_mul`, `normal_div`, `normal_sqrt` is currently prefixed with `_` (unused). When the multi-limb implementations land, these must actually use the rounding mode for the final round. Watch for the `_` prefix to be removed.

---

## Observation 4 — Integration Test Coverage Gap (CONFIRMED)

**Time**: 2026-05-08, session open

Read `crates/tambear/tests/big_float_arith_invariants.rs` and `crates/tambear/src/primitives/big_float/arith_tests.rs` in full.

**Finding**: Open Question 1 is answered — every single test creates BigFloat values via `BigFloat::from_f64(v, 200)` where `v` is a small, exact-representable float (1.0, 2.0, 3.14, -2.71, etc.). The precision field is 200, but the actual significant bits in the mantissa are still ≤ 53 (since the value came from f64). Therefore `f64_path_eligible()` returns `true` for all of them and the `unimplemented!()` branches are **never reached**.

This means: the test suite at `1560 tests, 0 warnings` does not exercise any multi-limb arithmetic. It is a clean green on a completely different code path than what the BZ algorithm claims describe.

**This is not a criticism of the prior team** — the integration test file header explicitly says "Multi-limb operand support lands in the follow-up commit" and the test was deliberately written for the f64 fast path. But it is an important disambiguation: the 1560 green tests do NOT validate BZ 3.1/3.3/3.5/3.10.

**Consequence for this session**: When the pathmaker's unstub lands, we need tests that exercise the multi-limb code path. This requires BigFloat values with more than 53 significant mantissa bits — these must be constructed directly from limb arrays or via repeated arithmetic that accumulates bits (not from `from_f64`).

**The cross-precision consistency check** (compute at p₁ and p₂; round p₂ result to p₁; should match) is the right antibody. For it to reach the multi-limb paths, it must use p values like p=100 with operands that actually have 100 significant bits (not values like `from_f64(1.0, 100)` which have only 1 significant bit in the mantissa).

---

## Open Questions (Updated)

1. **ANSWERED**: The current integration tests do NOT exercise multi-limb paths. All operands are f64-sourced values with ≤ 53 significant bits. The multi-limb `unimplemented!()` branches are dead code in the current test suite.

2. **OPEN**: The module docstring says "Add, sub, and mul ship as fully-implemented BZ algorithms in this commit." But the code has `unimplemented!()` panics for multi-limb operands in add and mul. Is this docstring aspirational or was there a version of arith.rs where the BZ implementations existed and were removed? The git history would tell us, but the operational answer is: the code is what matters, and the code is stubs.

3. **ANSWERED**: `BigFloat::from_raw_limbs(sign, exponent, precision_bits, limbs)` exists as a `#[cfg(test)]` constructor in `ty.rs:305`. It accepts a `Vec<u64>` of raw limbs with canonical-form validation (top bit of top limb must be set). This is exactly the mechanism for creating genuine multi-limb test inputs. No need to derive them from f64 arithmetic. The pathmaker can write:
   ```rust
   // p=128: 2 limbs. Both limbs non-zero → genuinely multi-limb.
   let a = BigFloat::from_raw_limbs(false, 1, 128, vec![0xDEAD_BEEF_DEAD_BEEF, 0x8000_0000_0000_0001]);
   ```
   This will NOT take the f64 fast path because `to_f64()` will lose the low limb's bits and the round-trip equality check will fail.

4. **OPEN**: The `f64_path_eligible` function contains a potential inefficiency: it calls `to_f64()` twice (once to get the bits, once to re-encode and compare). For the multi-limb path, the eligibility check itself runs before the panic. If the BigFloat has 1000 bits set, we're doing unnecessary work to discover it's not eligible. This is a design observation, not a blocker.

5. **OPEN**: The rounding mode is `_rounding` (unused) in all four `normal_*` functions. When the multi-limb implementations land, this MUST be used for the final round. Watch for the `_` prefix being removed — if it stays, the rounding mode is being silently ignored.

---

## Observation 5 — Cancellation-to-Zero Silent Failure Path (VERIFIED)

**Time**: 2026-05-08, pre-review

**Question**: Does `is_zero()` scan limbs or check the `kind` tag?

**Finding**: `ty.rs:393` — `is_zero()` is `matches!(self.kind, BigFloatKind::Zero)`. Tag-only check, does not scan limb content.

**Consequence**: A `BigFloat { kind: Normal, limbs: [0, 0, 0], ... }` returns `is_zero() = false` even though its magnitude is mathematically zero. This violates the canonical-form invariant (kind = Normal implies top limb has top bit set).

**Where the trap lives**: BZ 3.1 (add/sub), exact-cancellation case where `a - b` produces all-zero limbs. The f64 fast path avoids this because `a.to_f64() - b.to_f64() == 0.0` → `from_f64(0.0)` produces `kind = Zero`. The multi-limb path does limb-level subtraction and must explicitly check all-zero after the subtraction loop. Checking only the top limb (common mistake) or relying on `is_zero()` on a partially-constructed result (circular) will miss it silently.

**F13 note**: `from_raw_limbs` would catch this at the assertion level (top bit not set), but the BZ 3.1 implementation will use direct field assignment, not `from_raw_limbs`. The zero-check must live inside `normal_add`.

**Watch item**: after the limb subtraction loop, look for:
```rust
if result_limbs.iter().all(|&l| l == 0) {
    return BigFloat::zero(result_precision);
}
```
Absence of this check = cancellation-to-zero bug.

---

## Pre-Review Checklist (applied to each BZ algorithm as code arrives)

Written before seeing the implementation — hypothesis before observation.

### For BZ 3.1 (add/sub):

- [ ] Exponent alignment: smaller operand's mantissa right-shifted by `|exp_a - exp_b|` bits. Shift count exceeding `precision_bits` means the smaller operand vanishes — result is the larger operand unchanged.
- [ ] Integer add: limb-by-limb with carry propagation. Carry out of the top limb requires right-shift by 1 and exponent increment — this is the renormalization step.
- [ ] Subtraction / cancellation: when `a ≈ -b`, the result has many leading zero bits. Must left-shift to restore top-bit-set canonical form, decrementing exponent accordingly.
- [ ] Zero result: if subtraction yields all-zero limbs, `kind` must switch from `Normal` to `Zero`. The `_rounding` parameter must be wired for the ±0 sign rule.
- [ ] `_rounding` removed from parameter prefix and used.
- [ ] Post-condition: `limbs.len() == ceil(precision_bits / 64)`, top bit of top limb is set.

### For BZ 3.3 (schoolbook mul):

- [ ] Guard bits: intermediate product computed at `p + 50` guard precision. This means `ceil((p + 50) / 64)` limbs during multiply, reduced to `ceil(p / 64)` by the final round.
- [ ] Schoolbook correctness: for `n`-limb × `n`-limb, the product is `2n` limbs before the guard-truncation round. The low `n` limbs are discarded after rounding.
- [ ] Tests use `from_raw_limbs` with genuinely dense limb patterns (non-zero low limbs).
- [ ] Rounding mode wired for the guard-region round.

### For BZ 3.5 (Newton div):

- [ ] Iteration count: `⌈log₂(p/53)⌉ + 2`. At p=200: 4. At p=500: `⌈log₂(9.43)⌉ + 2 = 6`. At p=1024: 7. Verify the formula.
- [ ] Initial guess: from `f64` reciprocal (hardware-computed, ~53 bits accurate).
- [ ] Newton recurrence: `x_{n+1} = x_n * (2 - b * x_n)`. Division result: `a / b = a * converged_reciprocal`.
- [ ] Guard bits: Newton runs at `p + 50`, final round to `p` using the rounding mode.
- [ ] Fixed iteration count (no convergence check in the loop body per DESIGN.md §3).

### For BZ 3.10 (Newton sqrt):

- [ ] Recurrence: `x_{n+1} = (x_n + self / x_n) / 2` (Babylonian). The inner division: is it full BigFloat div or reduced-precision? Flag if reduced.
- [ ] Guard bits: `p + 50` throughout, final round to `p`.
- [ ] Test: `sqrt(x * x) ≈ |x|` for multi-limb `x` from `from_raw_limbs`.

### Cross-cutting (ALL ops):

- [ ] Tests include `from_raw_limbs` operands with non-zero low limbs (genuinely multi-limb).
- [ ] `cargo test --lib` green, 0 warnings after each algorithm lands.
- [ ] At least RNE and RoundTowardNegativeInfinity tested for each op.
- [ ] Cross-precision consistency: result at p=500 rounded to p=200 matches result computed directly at p=200, within 1 ulp. If absent, flag it to navigator as mandatory per briefing.

