---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher (literature/canonical-source verifier)
date: 2026-05-09
audience: pathmaker (impl owner), navigator (routing), aristotle (invariant cross-check)
inputs:
  - arith.rs at HEAD (commit 307ae8a; 22e3758 = Task #13 fix; 1593d8e = Phase C div restructure)
  - canonicalize_and_round @ arith.rs:874
  - All call sites: 852 (add), 1272 (mul), 1461 (round_to_precision)
  - oracle-validation.md §1.2 cross-precision consistency idiom
purpose: pin the documented invariant that all callers of canonicalize_and_round pass correct (round_bit, sticky_hi_bit, sticky_bit) values, so the n=1/n=2/n=3 left-shift guard-bit propagation is structurally correct rather than inferred from passing tests.
---

# canonicalize_and_round — call-site audit (post-22e3758 fix)

> **The headline.** The Task #13 fix (commit 22e3758) gave canonicalize_and_round
> a 3-bit sub-LSB guard contract: `round_bit` at position -1, `sticky_hi_bit`
> at position -2, `sticky_bit` at -3 and below. The left-shift cases for
> n=1 / n=2 / n≥3 are all formally handled. This audit names the call-site
> contract — every caller's responsibility for the three guard parameters —
> so that future callers (e.g., new BZ algorithms or oracle helpers) inherit
> the invariant.

---

## 1. The function's guard-bit contract

`canonicalize_and_round(mag, exp_at_lsb, sign, round_bit, sticky_hi_bit, sticky_bit, p, mode)`
takes a magnitude buffer `mag` whose bit-0 carries place value `2^exp_at_lsb`,
and three sub-LSB bits describing what got truncated **strictly below** mag's
bit 0:

| Parameter | Position | Meaning |
|-----------|----------|---------|
| `round_bit`     | -1       | Half-ULP at exp_at_lsb. Set if a 0.5-ULP component was truncated. |
| `sticky_hi_bit` | -2       | Quarter-ULP. Only meaningful when round_bit is exact-tie-eligible. |
| `sticky_hi_bit` only flips on the **subtraction-with-borrow** path where the residue can't be summarized by round+sticky alone. |
| `sticky_bit`    | -3 and below | OR of all bits at positions ≤ -3. |

Together these three bits encode the truncated-below-mag value as a binary
fraction in (0, 1) ULP at exp_at_lsb. The pattern matches BZ §3.1.6's
guard-bits-plus-sticky framework, extended by one position to handle the
sub-cancellation case where a tie/non-tie ambiguity would otherwise be lost
in a 1-bit left-shift.

## 2. Call-site enumeration (post-22e3758)

Three call sites exist at HEAD (`grep -n canonicalize_and_round arith.rs` →
lines 852, 1272, 1461). Audit:

### Site 852 — `normal_add_multilimb` (BZ Algorithm 3.1)

**Caller responsibility**: when the add/sub path captures truncated bits via
`shr_limbs_with_sticky`, OR captures borrow-propagation residue from the
opposite-sign sub case, the three guard parameters MUST encode what's strictly
below mag's bit 0. The post-22e3758 implementation correctly:

- Captures `round_bit` from the alignment shift (BZ §3.1 step 4).
- Captures `sticky_hi_bit` from the borrow-recompute table (BZ §3.1 step 5
  in the magnitude-subtract case, lines 769-825 in arith.rs).
- Captures `sticky_bit` as the OR of all bits at positions ≤ -3.

**Verification**: the entire `big_float_multilimb_proptest.rs` suite (~60
tests/proptests) plus `big_float_cross_precision.rs` Phase A/B/C suite
exercises this site under all 5 rounding modes at p ∈ {107, 128, 200,
500, 1024}. All tests green at HEAD.

**Risk class for future regressions**: HIGH. Any change to the alignment
or borrow logic in normal_add_multilimb must preserve the three-guard-bit
extraction invariant. F13 antibody pattern: regression tests in
big_float_phase_c_sub_audit.rs and the Phase C proptests are the
load-bearing checks.

### Site 1272 — `normal_mul_multilimb` (BZ Algorithm 3.3)

**Caller responsibility**: schoolbook multiplication produces an EXACT
2p-bit product integer. There is no sub-product residue at the lane level —
every bit of the true product is materialized in the 2p-limb output buffer.
The three guard parameters are **all 0** at function entry; canonicalize
sees the full product and decides the round/sticky from the bits below
position p_a + p_b - 1.

**Verification**: passing `(0, 0, 0)` is correct for schoolbook because the
multi-limb integer multiply is exact. The implication: the n=1 left-shift
case for the leading-bit-position-is-(p+p-2)-not-(p+p-1) "no-carry" sub-case
of the product is handled by canonicalize's left-shift logic, with all three
guards starting at 0.

**Risk class**: LOW. The site has been correct since e2e8fb2; the 22e3758
fix doesn't change the contract here.

### Site 1461 — `round_to_precision`

**Caller responsibility**: this is a precision-change wrapper. The input
BigFloat already has its full precision in `bf.limbs`; there's no sub-LSB
residue. All three guards are 0 at function entry; canonicalize captures
fresh round/sticky from the right-shift to the new (smaller) precision.

**Verification**: `round_to_precision` is invoked as part of:

- The public API `BigFloat::with_precision_rounded` (oracle/peer comparison
  code path).
- **Newton iteration intermediates in `normal_div_multilimb` and
  `normal_sqrt_multilimb`** — each iteration's intermediate is widened to
  `result_precision + 50` guard bits then rounded back to operand-precision.
  Round-back through `round_to_precision` enters this site with
  `(0, 0, 0)` and uses canonicalize's right-shift logic to capture the
  guard bits from the wider intermediate.

**Risk class**: LOW. The Newton iteration intermediates are correctly
handled because the caller (Newton) has the full intermediate value in
limbs; nothing has been truncated externally before reaching round_to_precision.

## 3. Documented invariant (the contract)

> **Invariant for future callers of `canonicalize_and_round`**: the three
> guard parameters `(round_bit, sticky_hi_bit, sticky_bit)` MUST faithfully
> represent what was truncated strictly below mag's bit 0, in this layout:
>
> - `round_bit` ∈ {0, 1}: the bit at position -1 (½ ULP at exp_at_lsb).
> - `sticky_hi_bit` ∈ {0, 1}: the bit at position -2 (¼ ULP). Only set in
>   contexts where the residue could be exactly ¼ ULP and that distinction
>   would matter post-shift (specifically: the sub-borrow recompute in
>   normal_add_multilimb's opposite-sign path).
> - `sticky_bit` ∈ {0, 1}: the OR of all bits at positions ≤ -3.
>
> Passing wrong guards produces silently-wrong arithmetic, especially in
> the left-shift n=1 / n=2 cases where the bits get re-positioned within
> mag rather than dropped.

## 4. Newton iteration audit (the deeper concern from navigator)

The question: do `normal_div_multilimb` and `normal_sqrt_multilimb` have
internal call paths that could pass wrong guards through to canonicalize?

**Walk-through for normal_div_multilimb**:
1. Newton seed `recip = from_f64(1.0/b_f64, target_p)` — exact f64 → BigFloat
   construction; no sub-LSB residue at construction time.
2. Each iteration: `bx = b.mul(&x, RNE)` then `two_minus_bx = two.sub(&bx, RNE)`
   then `x = x.mul(&two_minus_bx, RNE)`.
   - `b.mul(&x, RNE)`: enters site 1272 with `(0, 0, 0)` from the
     normal_mul_multilimb caller. Correct.
   - `two.sub(&bx, RNE)`: enters site 852 with the actual alignment
     residue from the sub. Correct because normal_add_multilimb computes
     the three guards from its alignment + borrow logic.
3. Final `a_guard.mul(&recip, ...)` and `round_to_precision(&prod, ...)`:
   site 1272 then site 1461, both with `(0, 0, 0)`. Correct.

**Walk-through for normal_sqrt_multilimb**:
1. Newton seed via f64 sqrt + scaling — exact construction; `(0, 0, 0)`.
2. Each iteration: `a_over_x = a.div(&x, RNE)`, `sum = x.add(&a_over_x, RNE)`,
   `x = sum.div(&two, RNE)`.
   - `a.div(...)` recursively goes through site 1461 + 1272 + 852 paths,
     each with correct guards.
   - `x.add(...)` site 852 with correct guards from alignment.
   - `sum.div(...)` same recursion.
3. Final `round_to_precision(&x, result_precision, rounding)` site 1461
   with `(0, 0, 0)`. Correct.

**Conclusion**: the three guard parameters are correctly propagated through
all current Newton iteration call paths. The structural reason is that the
ONLY site that produces nonzero guards is `normal_add_multilimb` (site 852),
and that site is internally responsible for computing them from its
alignment+borrow logic. Mul (site 1272) and round_to_precision (site 1461)
both correctly pass `(0, 0, 0)` because they have no sub-LSB residue to
report at function entry.

## 5. Deferred future-caller protections

If new code adds a fourth call site to canonicalize_and_round, the new
caller must:

- Audit whether its inputs carry sub-LSB residue.
- If yes: compute the three guard bits faithfully (round_bit at -1,
  sticky_hi_bit at -2, sticky_bit at -3 and below).
- If no: pass `(0, 0, 0)`.

A `#[debug_assertions]` precondition in canonicalize_and_round could
defensively check that `sticky_hi_bit ≤ 1` and `sticky_bit ≤ 1` (they're
binary flags), but the existing API uses `u64` for typing flexibility.
Recommend not adding the assert — the invariant is documentary, and
typing flexibility is more important than catching the never-happens case.

## 6. Provenance

- Authored 2026-05-09 by math-researcher in team `tambear-sweep31-finish`
  per navigator's offered side-quest.
- Substrate verified at `git log` HEAD = 307ae8a; canonicalize_and_round
  at arith.rs:874 with the post-22e3758 three-guard signature.
- Cross-checked: 1514 lib tests passing at HEAD; the post-fix Phase C
  proptests (sub + sqrt) green; phase_c_div_round_bit_boundary
  restructured at 1593d8e (was operand-perturbation, not guard-bit bug).
- This audit pins what passing tests already imply — the three-guard
  contract is structurally correct, and the named invariant lives here
  as a future-caller specification rather than inferred from test
  results alone.
