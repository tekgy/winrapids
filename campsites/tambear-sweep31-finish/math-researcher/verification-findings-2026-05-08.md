---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (verification of unstub at e2e8fb2)
audience: pathmaker (impl owner), navigator (routing), team-lead, aristotle (invariant cross-check)
purpose: document concrete deviations from Brent-Zimmermann + DESIGN.md found during post-impl audit, with reproducer tests
inputs:
  - arith.rs at e2e8fb2 + b/c2798f5 housekeeping
  - bz-impl-reference.md (own prior incarnation's tactical reference)
  - DESIGN.md §3 (algorithm dispatch table) + §5 Q3, Q6 (ratification answers)
  - Brent-Zimmermann *Modern Computer Arithmetic* (2nd ed.), Algorithms 3.1, 3.3, 3.5, 3.10
---

# Verification Findings — BZ unstub at commit e2e8fb2

> **Story from the trail**. The unstub shipped fast (1465 → ~1500+ lib tests, all green). Audit against the canonical literature found **three real bugs and one MEDIUM-tier consistency issue** that the existing test corpus does not catch. All four are reproducible with single-test fixtures dropped in `R:\tambear\crates\tambear\tests\big_float_*_audit.rs`. None are deviations from Brent-Zimmermann's text per se — they're implementation gaps in the corners that BZ §3.1 (which assumes single-precision operands) doesn't directly cover, plus one downstream f64-overflow corner that's structurally mine to flag.
>
> **What this is not**: a critique of the unstub. The shipping work is solid through the common path. The four findings are corner-case deviations that surface specifically because tambear's precision lattice supports `max(p_a, p_b)` mixed-precision arithmetic AND extreme exponent ranges (i64) that f64 can't represent.
>
> **What it is**: an antibody log. Each finding has a reproducer test that fires before the fix and goes green after. F13 antibody pattern applied: rules with scope preconditions (here: "operands are f64-representable", "operands have aligned-equal limbs after shift") need antibodies that trigger at the precondition violation, not silently produce wrong answers.

---

## Finding 1 (CRITICAL) — Cancellation/borrow underflow when `cmp == Equal AND (round|sticky) > 0`

**Where**: `arith.rs:707-792`, the magnitude-subtract path in `normal_add_multilimb`.

**The bug**: when `cmp_limbs(large_aligned, small_aligned)` returns `Equal` (integer limbs match exactly) AND the alignment captured non-zero `round_bit | sticky_bit` (meaning small had bits below the buffer's LSB, so its TRUE magnitude is strictly larger than large's), the code takes the `Greater | Equal` branch:

1. `diff = large_aligned - small_aligned = [0; n]` (all zeros).
2. `(round | sticky) > 0` triggers the borrow branch.
3. The code subtracts 1 ulp from `diff`, which underflows from `[0; n]` to `[u64::MAX; n]`.
4. The result is canonicalized as a HUGE positive magnitude with `result_sign = large.sign`.

**The correct behavior**: when small's true magnitude exceeds large's, the result sign should be `small.sign`, and the result magnitude is `2^{lsb_place} - (round_bit · 0.5 + sticky · ε)` ulps — a tiny positive value that, after canonicalization, yields a result with a single bit set far down in the mantissa (essentially the `1 - round/sticky` complement at sub-LSB scale).

**Reproducer**: `R:\tambear\crates\tambear\tests\big_float_cancel_borrow_audit.rs::cancellation_with_hidden_below_lsb_bits`. Constructs `large = (all-ones-p200, exp=0)` and `small = -(all-ones-p300, exp=0)` where `small.mantissa = (large.mantissa << 100) | 1`. Expected: tiny negative. Actual: `+2.658e36`.

**BZ correspondence**: BZ §3.1 (Algorithm 3.1 box on p.117) assumes both operands at the same working precision `p`. With `p_a == p_b` and consistent canonical form, the `cmp == Equal AND (round|sticky) != 0` case can't arise — limbs equal implies values equal exactly, and round/sticky are zero. Tambear's `max(p_a, p_b)` design extends BZ; the extension exposes this corner. This is a tambear-specific fix, not a literature deviation.

**Fix sketch**: at the `cmp_limbs` match in arith.rs:707, route `Equal AND (round|sticky) > 0` to the same branch as `Less` — small is the larger-true-magnitude operand. Alternatively, before the cmp, if `(round|sticky) > 0`, treat small as conceptually `small_aligned + ε` for cmp purposes:

```rust
let cmp = limbs::cmp_limbs(&large_aligned, &small_aligned);
let small_strictly_larger = cmp == Ordering::Less
    || (cmp == Ordering::Equal && (round_bit | sticky_bit) != 0);

if small_strictly_larger {
    // ... existing Less branch logic, but small's hidden bits are now treated correctly.
}
```

**Severity**: CRITICAL per oracle-validation.md §1.2 — produces non-IEEE-754-conformant arithmetic results in a constructive corner case. Routing: pathmaker (impl), aristotle (invariant cross-check post-fix), adversarial (proptest fuzz once fix lands).

---

## Finding 2 (CRITICAL) — Wrong sign of `exp_shift` in `newton_reciprocal` scaled-seed branch

**Where**: `arith.rs:1242` (`recip.exponent += exp_shift;`) and the comment at lines 1239-1240.

**The bug**: when b is too small/large for f64 representation, the code scales b to f64-range via `b_scaled.exponent = scaled_exp = 0`, computes `recip_scaled = 1.0 / b_scaled.to_f64()`, and unscales. The unscale step does `recip.exponent += exp_shift`. The correct math is `recip.exponent -= exp_shift`.

Derivation: `b_scaled.exponent = 0` means `b_scaled = b · 2^{-exp_shift}` (where `exp_shift = b.exponent - 0`). So `b = b_scaled · 2^{exp_shift}`, and `1/b = (1/b_scaled) · 2^{-exp_shift}`. The `recip.exponent` (which carries `1/b_scaled`'s exponent) needs to **subtract** `exp_shift` to become `1/b`'s exponent.

**Reproducer**: `R:\tambear\crates\tambear\tests\big_float_div_extreme_audit.rs::newton_div_huge_b_scaled_seed_unscale` (and `_tiny_`). For `b = 2^2000`, expected `1/b = 2^-2000` (positive, exp~-2000). Actual: `sign=true, exp=126000, to_f64=-inf` (sign-flipped via downstream Newton overflow on the wrong-direction seed). For `b = 2^-2000`, expected exp~+2000. Actual: exp=-1995.

**BZ correspondence**: BZ §3.5 doesn't prescribe how to seed Newton in extreme-exponent regimes; this is an implementation choice. The chosen technique (scale to f64-range → seed → unscale) is correct in concept; the sign of the unscale exponent shift is the bug.

**Fix**: change `+= exp_shift` to `-= exp_shift` at line 1242. Update the comment at lines 1239-1240 to "1/(b/2^exp_shift) = (1/b)·2^exp_shift, so 1/b = recip_scaled · 2^{-exp_shift}".

**Severity**: CRITICAL per oracle-validation.md §1.2 — produces wildly wrong results for any input where b is f64-non-representable (overflow OR underflow). Reachable via any user query that asks "what is 1/(2^-2000)" or similar, which is exactly the high-precision use case BigFloat is designed for.

---

## Finding 3 (CRITICAL) — `newton_reciprocal` doesn't check `1/b_f64` for overflow

**Where**: `arith.rs:1207-1212` — the f64-eligible-seed branch.

**The bug**: the condition checks only `b_f64 != 0.0 && b_f64.is_finite()`. When b's value is in f64's *subnormal* range, `b.to_f64()` returns a subnormal f64 (non-zero, finite), so the condition passes. But `1.0 / b_f64` then OVERFLOWS to ±Inf, and `BigFloat::from_f64(inf, p)` constructs an Infinity-kind seed. Newton iteration on infinity diverges; the result is Inf instead of a normal finite value.

**Reproducer**: `R:\tambear\crates\tambear\tests\big_float_div_subnormal_seed_audit.rs::newton_div_b_at_f64_subnormal_boundary`. `b = 2^-1050` (f64-subnormal magnitude). Expected: `1/b = 2^1050` (Normal, exp~1050). Actual: kind=Infinity.

**Fix**: extend the condition to require `recip_f64.is_finite()`:
```rust
let recip_f64 = 1.0 / b_f64;
let initial = if b_f64 != 0.0 && b_f64.is_finite() && recip_f64.is_finite() {
    BigFloat::from_f64(recip_f64, target_p.max(53))
} else {
    // ... scaled-seed branch (also requires Finding 2 fix to work correctly)
};
```

**Note**: this finding interacts with Finding 2 — the scaled-seed branch is currently buggy, so even with this fix in place, the scaled branch must also be fixed (Finding 2) for subnormal-boundary divisions to work. Both fixes need to land together.

**Severity**: CRITICAL — unbounded silent failure (returns Inf when correct value is finite) on any input crossing the f64 subnormal boundary.

---

## Finding 4 (MEDIUM) — NaN payload dropped in `div`, inconsistent with `add`/`mul`/`sqrt`

**Where**: `arith.rs:320-322`.

**The bug**: `div` uses `Self::nan(result_precision)` (canonical NaN with payload `1<<51`) instead of preserving the input NaN's payload. `add` (lines 144-160), `mul` (lines 256-274), and `sqrt` (line 371-373) all explicitly preserve payload. The inconsistency means a NaN value passed through a `div` chain loses diagnostic payload bits that other ops keep.

**DESIGN.md §5 Q3** ratified 2026-05-08: payload preservation per IEEE 754-2019 §6.2.1. Aristotle's gauntlet Surface 6 covers this surface. The current div is a partial regression on §5 Q3.

**Severity**: MEDIUM per oracle-validation.md §1.2 — IEEE 754 explicitly allows variation in NaN-payload propagation (§6.2.3), so this isn't strictly non-conformant. But the inconsistency between div and the other three ops is the structural concern: a code that diagnostically packs NaN payloads expects all ops to preserve them.

**Fix**: align div with mul's pattern (lines 256-274). Pattern is identical; the fix is mechanical.

---

## Cross-cutting observations

### F13 antibody coverage gaps

The four findings above exist because the existing proptest gauntlets exercise the **common-path multi-limb regime** (operands constructed via `multilimb_positive` / `multilimb_p500` etc., where both operands are f64-magnitude with low-limb noise). They do NOT exercise:

- **Asymmetric precision** with constructive cmp-equal alignment (Finding 1 trigger).
- **Extreme exponent** input regimes (b > 2^1023 or b < 2^-1074) where to_f64 saturates (Findings 2, 3 triggers).
- **Cross-op NaN payload propagation** through chains of arithmetic (Finding 4 trigger).

Recommendation for adversarial: extend the proptest input generators to include all three classes. The reproducer tests above are templates; generalize each into a proptest with random parameters in the constructive corner.

### BZ §3.1 vs tambear's `max(p_a, p_b)` design

BZ §3.1 (Algorithm 3.1 box, p.117) prescribes the algorithm for *single-precision* floating-point arithmetic — both operands at precision p, result at precision p. Tambear's design (DESIGN.md §1 Q2) extends this: arithmetic between operands at p_a and p_b returns a result at `max(p_a, p_b)`. This extension is mathematically sound (the literature for variable-precision mpfr-style arithmetic uses the same convention), but the corner cases of the extension (Finding 1 specifically) are not covered by BZ's proof of correctness. Each cross-precision corner is a tambear-specific antibody surface.

### Quadratic Newton convergence verification

The DESIGN.md §3 + bz-impl-reference.md §3.2 prediction of `⌈log₂(p/53)⌉ + 2` iterations is correct and matches the implementation at arith.rs:1236-1242. Verified: at p=200, 4 iters; p=500, 6 iters; p=1024, 7 iters. Algorithm-level correctness, modulo the seed bugs above.

### Schoolbook mul (BZ 3.3) — clean

Re-walked `normal_mul_multilimb` (lines 1096-1136) and the `mul_limbs` kernel (limbs.rs:270-296). The exponent calculation `exp_at_lsb = a.exponent + b.exponent - p_a - p_b + 2` matches BZ's product-mantissa-exponent derivation (verified in Section 1.2 of bz-impl-reference). The schoolbook lane-multiply with u128 carry handles all `u64 × u64 + u64 + carry < 2^128` cases correctly. **No findings on mul.**

### sqrt scaled-seed (BZ 3.10)

Re-walked `normal_sqrt_multilimb` (lines 1367-1422). The k/2 scaling math `g.exponent -= k/2` is derived correctly: `a_scaled = a · 2^k`, `sqrt(a_scaled) = sqrt(a) · 2^{k/2}`, so `sqrt(a) = sqrt(a_scaled) · 2^{-k/2}`. Sign of the unscale is correct (subtraction). **The sqrt extreme-magnitude test passes** despite using a buggy `div` internally during Newton iteration — this is benign coincidence (the iteration's specific shape happens to recover from the seed-direction bug), not a structural guarantee. Once div is fixed (Findings 2, 3), the sqrt remains correct because its scaling logic is independently correct. **No primary findings on sqrt; secondary dependency on div fixes.**

---

## Routing

| Finding | Severity | Owner | Antibody fixture |
|---------|----------|-------|------------------|
| 1 (cancel/borrow) | CRITICAL | pathmaker | `tests/big_float_cancel_borrow_audit.rs` |
| 2 (recip exp_shift sign) | CRITICAL | pathmaker | `tests/big_float_div_extreme_audit.rs` |
| 3 (recip seed overflow) | CRITICAL | pathmaker | `tests/big_float_div_subnormal_seed_audit.rs` |
| 4 (NaN payload in div) | MEDIUM | pathmaker | (add a test using the `add` payload-preservation tests as template) |

Tasks #8, #9, #10, #11 carry these findings into the team-task-board. Each has a fix sketch + reproducer pointer.

**Sequencing**: Findings 1, 2, 3 are independent and can land as separate commits. Finding 4 is mechanical and can ride along. After all four land, the audit fixtures (`big_float_*_audit.rs`) become permanent regression tests — they document the corners the gauntlet missed.

**Aristotle cross-check**: F11 (recognition/design discipline) — the audit fixtures should be treated as ratified-permanent: they're publication-grade evidence that the pre-fix code had these bugs, and any future refactor must keep them green.

---

## Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish`.
- Substrate verified: `git -C R:/tambear log --oneline` shows e2e8fb2 as the unstub commit; arith.rs at 1398 lines covers BZ 3.1, 3.3, 3.5, 3.10.
- Cross-checked: BZ §3.1 (p.117), §3.3 (p.124), §3.5 (p.130), §3.10 (p.155). DESIGN.md §1, §3, §5 ratification answers.
- Test fixtures committed alongside: cancel_borrow_audit.rs (1 test), div_extreme_audit.rs (2 tests), div_subnormal_seed_audit.rs (2 tests), sqrt_extreme_audit.rs (2 tests, both passing — sqrt is clean).
- 5 of 7 audit-tests fire (4 confirmed bugs); 2 pass (sqrt extreme magnitudes are correct).
- This is a deliverable for pathmaker. Once the 4 fixes land, math-researcher returns to verify each fix against BZ + the audit fixtures.

---

## Addendum 2026-05-09 — Phase C drift root cause (Finding 5)

After tasks #8/9/10/11 landed in commits 91bf2b1 + b9dfeb9 + e76408a, the team
asked me to chase the remaining 2-ulp drift in Phase C `sub` and `div` tests
(`phase_c_sub_alignment_stress`, `phase_c_div_round_bit_boundary`, both
`#[ignore]`'d). Root cause confirmed.

### Finding 5 (HIGH) — `canonicalize_and_round` loses sticky bit on left-shift n=1

**Where**: arith.rs:927-950, the `else if shift < 0` block.

**The bug**: when canonicalize needs `n = 1` left-shift (top bit at position
`target_pos - 1` of mag) AND incoming `(round_bit, sticky_bit) = (1, 1)`,
the code absorbs round_bit into mag's bit 0 correctly via `set_bit(&mut mag, 0)`
at line 940, but the sticky_bit absorption at line 942 is gated on `n >= 2`
and skipped for `n = 1`. Lines 948-949 then unconditionally zero both
round_bit and sticky_bit. The downstream `should_round_up` at line 982 sees
`(0, 0)` and treats the result as exact — but it isn't; the original residue
was strictly between 0 and 1 ulp at the new sub-LSB position, just past the
representable bits of mag.

**The math**: original residue ∈ (0.5, 1.0) ulp (round=1, sticky=1 means more
than half-ulp, less than one-ulp). After ×2 (left-shift by 1), residue ∈
(1.0, 2.0). The absorbed integer = 1 lands in mag's bit 0. The fractional
remainder = residue - 1 ∈ (0, 1) ulp at the new scale. Conservative correct
treatment: keep sticky_bit_new = 1 (because there's something nonzero below
the new round position). The new round_bit is structurally lossy without
preserving more bits.

**Why it matters for Phase C**: in `via_high = round_to_precision(diff_high,
p_low)`, the high-precision sub passes through this n=1 left-shift with
nonzero sticky during canonicalize, drops the sticky, finalizes diff_high to
a value missing its sub-LSB info, then round_to_precision sees no residue
data and rounds differently than `direct = round(a, p_low) - round(b, p_low)`
which had its own residue path.

**Reproducer**: `R:\tambear\crates\tambear\tests\big_float_phase_c_sub_audit.rs::phase_c_sub_minimal_repro_dump_intermediates`. Trace:
- `a_high.sub(b_high)` at p=157: (round=0, sticky=1) survives until canonicalize, becomes (1, 1) after the borrow logic (lines 770-826).
- canonicalize: mag's top at pos 155, target 156, shift = -1. Left-shift by 1: round_bit absorbed (mag bit 0 = 1), sticky_bit DROPPED.
- diff_high finalized at limbs[0] = 2220883097092057301. Sub-LSB info lost.
- `round_to_precision(diff_high, 107)`: shift = +50, RTE chooses round-down because no sticky records the lost data.
- Direct path at p=107: rounding uses sticky correctly because no n=1 shift occurs.
- Result: `direct.limbs[0] = 1971`, `via_high.limbs[0] = 1973`, drift = 2 ulps.

**Same root cause covers div**: phase_c_div_round_bit_boundary fails at
`limbs[0] = 18446744073709544314 vs 18446744073709544316` (also 2 ulps).
Newton iteration internally calls add/sub many times; one of those passes
through the n=1 left-shift loses a sticky bit; the error compounds.

**Why sqrt is clean**: `phase_c_sqrt_round_bit_boundary` is NOT
`#[ignore]`'d. Newton-Babylonian sqrt happens to not trigger the n=1
left-shift in canonicalize for the Phase C input shape (or compensating
errors cancel). This is empirical, not structural — sqrt could in principle
hit the same bug under different inputs.

**Fix options** (filed as task #13):

- **(A)** 3-line patch: when n=1 and old sticky != 0, keep sticky_bit = 1.
  Accepts ≤1 ulp imperfect rounding in this specific path (the new
  round_bit becomes lossy). Not bit-exact correctly-rounded.

- **(B)** Architectural: at canonicalize entry, extend mag's buffer with
  2 dedicated guard bits at the bottom. Round_bit / sticky_bit parameters
  merge into mag's guard region at function entry. Then mag-shifts are
  closed under the guard bits; n=1 left-shift just shifts within mag and
  doesn't need a special sticky-handling branch. This is the BZ §3.1.6
  shape; structurally correct.

- **(C)** Upstream fix in `normal_add_multilimb`: enlarge work_len so the
  buffer holds the would-be-sticky region directly. Then canonicalize sees
  enough bits to never need n < 2 left-shift.

(B) is the right long-term shape per BZ. (A) is a stopgap. Routing: pathmaker
to choose. Math-researcher will verify whichever choice lands.

**Severity**: HIGH (not CRITICAL) — produces 2-ulp drift in cross-precision
correctness, but the result is still "close to correct" (1-2 ulps from
correctly-rounded). For verification-tier oracle use, this matters: bit-exact
is the contract. For discovery-tier use, ≤2 ulps is within typical tolerance.

