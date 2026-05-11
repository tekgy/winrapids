# Phase C `sub_alignment_stress` — Structural Deconstruction

**Date:** 2026-05-08
**Author:** aristotle
**Subject:** The `phase_c_sub_alignment_stress` test in
`big_float_cross_precision.rs` is `#[ignore]`d because it fails with
limbs differing by 2 at the guard boundary. Navigator framed this as
"a structural question rather than a bug-fix mandate." This
deconstruction walks the question through Phases 1-8.

**Inputs:**
- `crates/tambear/tests/big_float_cross_precision.rs:1310-1362`
  (the `#[ignore]`d test)
- `crates/tambear/tests/big_float_cross_precision.rs:1115-1175`
  (the analogous `phase_c_add_alignment_stress` which DOES pass at
  the 1-ulp bound)
- Brent-Zimmermann §3.1.6 (rounding error analysis for add/sub at
  matched precision)
- DEC-031 §3.2 destination-dominated path budget (for monotonic-
  coarsening paths)
- chains-E/F/G subnormal-aware bounds (subnormal regime weakening)
- My prior gauntlet's Surface 2 (subnormal regime crossing)

**Repro (minimal failing input from proptest):**
- `a_mantissa_low = 1110511917290206315`
- `b_mantissa_low = 562949953421313`
- `delta = 3`, `p_low = 107`, `p_high = 157`
- Result: `direct.limbs[0] = 1971` vs `via_high.limbs[0] = 1973`,
  same exponent (-1), same higher limbs. Disagreement = 2 ulps at
  p=107.

---

## What the test asks

The `sub_alignment_stress` test compares two paths through the
precision lattice that compute the same mathematical subtraction:

- **Path 1 (direct)**: round each operand to `p_low`, then subtract
  at `p_low`.
  ```rust
  a_low = round(a_high, p_low, RTE);
  b_low = round(b_high, p_low, RTE);
  direct = a_low.sub(&b_low, RTE);   // result at p_low
  ```

- **Path 2 (via_high)**: subtract at `p_high`, then round to `p_low`.
  ```rust
  diff_high = a_high.sub(&b_high, RTE);   // result at p_high
  via_high = round(diff_high, p_low, RTE);
  ```

Both paths produce a BigFloat at `p_low`. The test asserts they
agree to within 1 ulp at `p_low`.

---

## Phase 1 — Assumption Autopsy

**A1.** The 1-ulp bound is the tight bound for cross-precision
agreement of add/sub at matched-rounding-mode RTE.

**A2.** The cross-precision agreement bound is symmetric in add and
sub.

**A3.** The 50-bit guard (`p_high = p_low + 50`) is sufficient to
make path 2 effectively bit-exact relative to path 1 modulo a single
final round.

**A4.** The dense low-limb pattern in the test (`a_mantissa_low` and
`b_mantissa_low` filling limb[0]) ensures the multi-limb sub path
fires (because f64_path_eligible returns false for these operands).

**A5.** The alignment shift (when `delta > 0`, b's mantissa is
right-shifted by `delta` bits to align exponents) preserves the
1-ulp bound.

**A6.** `with_precision_rounded` correctly applies RTE rounding when
reducing precision from p_high to p_low.

**A7.** The `within_1_ulp_bigfloat` helper has the right semantics
for "limbs differ by at most 1 in the lowest limb position only,
same higher limbs, same exponent."

---

## Phase 2 — Irreducible Truths

**T1 (rounding error per round step).** Each RTE rounding step
produces an error of ≤ 0.5 ulp at the destination precision. This
is IEEE 754-2019's correctly-rounded specification.

**T2 (sub of correctly-rounded operands).** When `a_low` and `b_low`
are correctly-rounded approximations of `a_high` and `b_high` at
`p_low`, the value `a_low - b_low` (computed exactly as integers)
may have absolute error up to `error(a_low) + error(b_low)`
relative to `a_high - b_high`. Each operand-error is ≤ 0.5 ulp at
`p_low`, so the combined error is ≤ 1 ulp at `p_low` *before* the
final sub-rounding.

**T3 (final sub-rounding).** When the value `a_low - b_low` doesn't
fit at `p_low` (because cancellation produced a result requiring
renormalization, OR because the integer subtraction needed
canonicalize_and_round), an additional 0.5-ulp error at `p_low` may
be introduced. **Total path-1 error vs the true `a_high - b_high`
mathematical value: ≤ 1.5 ulps at `p_low`.**

**T4 (path 2 error).** Path 2 computes `a_high - b_high` at p_high
(error ≤ 0.5 ulp at `p_high` ≪ 0.5 ulp at `p_low`), then rounds to
`p_low` (error ≤ 0.5 ulp at `p_low`). **Total path-2 error vs the
true math: ≤ 0.5 ulp at `p_low` + tiny.**

**T5 (cross-path disagreement).** The maximum disagreement between
path 1 and path 2 is bounded by the SUM of their individual errors
relative to the true math:
- `disagreement ≤ error(path1) + error(path2)`
- `≤ 1.5 ulp + 0.5 ulp = 2 ulps at p_low`

**T6 (cancellation amplification).** When `delta > 0` (operands at
different exponents), the alignment shift in BZ §3.1 introduces
round/sticky bits that fold into the cancellation. For
*add* (same-sign), errors don't amplify — the magnitude grows. For
*sub* (opposite-sign as encoded after the negation in `a.sub(b) =
a.add(-b)`), the magnitude SHRINKS as digits cancel; relative error
grows in proportion to the cancellation factor.

**T7 (the test repro's specific arithmetic).** With `delta = 3`,
`b_high` has exponent -3 vs `a_high` exponent 0. The sub
`a_high - b_high ≈ a_high * (1 - 1/8) = a_high * 7/8`. The result is
roughly `0.875 * a_high`. No catastrophic cancellation; mild relative-
error growth at most 8x in the cancellation-direction.

**T8 (the observed disagreement is 2 ulps, matching T5's upper
bound).** The minimal failing input shows `direct.limbs[0] = 1971`
vs `via_high.limbs[0] = 1973`, which is exactly 2 ulps at `p_low`.
This is at the boundary of T5's bound.

---

## Phase 3 — Reconstruction from Zero (10 paths to address)

If we built the `sub_alignment_stress` antibody from scratch:

**1. Tighten the bound to 2 ulps.** Per T5, the mathematical maximum
disagreement IS 2 ulps at `p_low`. Replace `within_1_ulp_bigfloat`
with `within_2_ulps_bigfloat` for sub. The 1-ulp bound was over-
specified for sub.

**2. Decompose into per-step errors.** Test path 1 step-by-step:
- `error(a_low - a_high) ≤ 0.5 ulp` at `p_low`
- `error(b_low - b_high) ≤ 0.5 ulp` at `p_low`
- `error(direct - (a_low - b_low)) ≤ 0.5 ulp` at `p_low` (final sub-
  round)
- And similarly for path 2. Each step verified separately, the
  cross-path disagreement bound is computed not asserted.

**3. Test only path-2 against the mathematical truth.** Path 2 is
the better-precision path (≤ 0.5 ulp). Use it as the "truth proxy"
and assert path-1 ≤ 1.5 ulp from path 2. This is asymmetric but
captures the intended antibody: path 1 should be no worse than
"correctly rounded twice" against the true math.

**4. Use mpmath as the truth oracle.** Compute the true mathematical
sub at 1000-digit precision, round to `p_low`. Assert both path 1
and path 2 are within 0.5 ulp of this oracle. Correctness check
against ground truth, not consistency check between two paths.

**5. Restrict the test to non-cancellation cases.** When the result
magnitude is comparable to the operand magnitudes (no catastrophic
cancellation), 1 ulp may hold. Pre-filter samples where
`|a_high - b_high| < 0.1 * max(|a_high|, |b_high|)` (cancellation by
more than 10x). The dense-bit-pattern strategy generates many such
samples; filtering would make the 1-ulp bound holdable.

**6. Use destination-dominated bound from DEC-031 §3.2.** The path
`p_high → p_low` is monotonically-coarsening; the destination-
dominated bound is `0.5 ulp at p_low` per coarsening step. Path 2
applies this bound. Path 1 is `(p_high → p_low) → (p_high → p_low)
→ sub at p_low`, which has TWO coarsening steps + an arithmetic
step. The destination-dominated bound across each step is `0.5 ulp
at p_low`. The composition is at most 3 × 0.5 = 1.5 ulp.

**7. The antibody isn't bit-exactness; it's "the precision-lattice
rules are honored."** When the bound is exceeded, the failure is
NOT "the implementation is buggy" but "the chosen bound is wrong."
The antibody's job is to surface implementations that violate
DEC-031 §3.2's destination-dominated rule. For sub at cancellation,
the rule itself bounds disagreement at 1.5 ulp; the test should
encode 1.5 ulp.

**8. Anti-YAGNI: ALSO test stronger bounds in non-cancellation
regimes.** Add a second test variant that constrains delta=0 (same-
exponent sub, no alignment shift, no cancellation amplification).
This would test the tighter "no-cancellation" bound and isolate the
cancellation-specific weakening.

**9. Isomorphic add test passes 1 ulp; what's different?** In
`add_alignment_stress`, same construction passes 1 ulp. Why?
Because for SAME-SIGN add, the result magnitude is roughly the SUM
of operands; relative error doesn't amplify. The errors of `a_low`
and `b_low` add INTO the result with the same sign as their
contribution; they don't cancel destructively. Path-1 total error is
≤ 0.5 + 0.5 + 0.5 = 1.5 ulp at `p_low` *in absolute terms*, but
when normalized against the result magnitude, it's typically ≤ 1
ulp because the result magnitude exceeds the operand magnitudes.

For sub, when the result magnitude is SMALLER than the operands
(cancellation), the absolute 1.5-ulp error becomes more than 1 ulp
of the SMALLER result. **The 2-ulp gap in the test is the
cancellation-amplification signature.**

**10. Use the "structural fingerprint of disagreement" as data, not
failure.** Run the test with a *recording* mode that catalogues
(delta, cancellation_ratio, observed_disagreement_in_ulps). Plot
the distribution. The 2-ulp tail is at high cancellation; the
1-ulp body is at low cancellation. The plot IS the antibody —
showing the rule's regime, not failing inside it.

---

## Phase 4 — Assumption vs Truth Map

| Assumption | Status | Replaced/refined by |
|---|---|---|
| A1: 1-ulp bound is tight for add/sub at RTE | NO for sub | T5 says 2 ulps for sub with cancellation |
| A2: bound is symmetric in add/sub | NO | add doesn't cancel; sub does (T6) |
| A3: 50-bit guard makes path 2 ≈ exact | YES | T4 |
| A4: dense low-limb forces multi-limb path | YES | f64_path_eligible returns false |
| A5: alignment shift preserves 1-ulp | NO | for sub with cancellation, shift error compounds |
| A6: with_precision_rounded is correct | YES | by construction |
| A7: within_1_ulp_bigfloat semantics | YES (just too tight) | bound should be 2 ulps for sub |

**The assumption that fails: A1/A5.** The 1-ulp cross-precision
agreement bound is too tight for subtraction in the cancellation
regime. The test failure is *the bound being wrong*, not the
implementation.

---

## Phase 5 — The Aristotelian Move

The conventional move: weaken the bound to 2 ulps; remove `#[ignore]`
on the sub test; re-run; get green.

The **Aristotelian move**: recognize that the cross-precision
agreement bound is *not a single number* — it's a function of the
operation kind, the cancellation ratio, the precision gap, and the
rounding mode. The DEC-031 §3.2 destination-dominated rule says
"≤ 0.5 ulp at destination per monotonic-coarsening step"; the
arithmetic step has its own per-op error budget. The cross-precision
test compares two compositions of these steps and asserts the
COMPOSED error stays bounded.

For add: composed error ≤ 1.5 ulp absolute, but when normalized to
the result magnitude (which dominates each operand), this is ≤ 1 ulp
of the result.

For sub: composed error ≤ 1.5 ulp absolute, but when normalized to
the result magnitude (which can be MUCH smaller than the operands
due to cancellation), this is ≤ N ulps of the result, where N
depends on the cancellation factor.

**The right antibody is parameterized by cancellation factor.**
Specifically:

```
disagreement(in result-ulps) ≤ 1.5 / (result_magnitude / operand_magnitude)
```

For the failing test repro: delta=3, b ≈ a/8, sub result ≈ 7a/8.
`result_magnitude / operand_magnitude ≈ 7/8`. Disagreement bound:
`1.5 / (7/8) ≈ 1.71`. Round up: 2 ulps. **Matches the observed
disagreement of 2 ulps.**

For more extreme cancellation (delta close to p_low), the
disagreement bound grows linearly. For delta = 50 (b at the cusp of
falling outside the precision window), `result_magnitude /
operand_magnitude` is dominated by sub's leading-bits cancellation,
and the bound grows. **The current implementation is correct under
the rule; the test bound is too loose.**

**Concrete recommendation for the team:**

1. Rewrite the bound in `sub_alignment_stress` from `within_1_ulp`
   to `within_n_ulps(disagreement_bound(delta, p_low))` where the
   bound is parameterized.

2. Even simpler: change the test to assert `within_2_ulps_bigfloat`
   universally, document that 2 ulps is the destination-dominated
   bound for sub with at-most-mild-cancellation, and split out a
   separate test with extreme cancellation that uses a wider bound
   parameterized by cancellation factor.

3. Then remove `#[ignore]`. The test will pass and document the
   actual structural rule.

---

## Phase 6 — Recursive Challenge

What did Phase 5 silently assume?

**B1.** The cross-precision test's *purpose* is to verify the
destination-dominated rule. (Yes — the test is named
"cross-precision consistency" and exists in
`big_float_cross_precision.rs` per the briefing's mandate.)

**B2.** The 1-ulp bound was chosen by someone who didn't account for
sub-cancellation. (The comment at lines 1161-1165 of the add test
explicitly says "BZ §3.1.6 proves only correct rounding at each
step; double-rounding through p_high (round inputs first → add →
round result) can disagree by ≤ 1 ulp with direct rounding at p_low
(round inputs → add at p_low). Bit-exact equality is mathematically
too strong — the antibody is 1 ulp." The reasoning is correct for
add. For sub the same comment doesn't apply because cancellation
amplifies.)

**B3.** A simple "use 2 ulps" would catch all observed failures.
(Need to run the test with the relaxed bound to verify. If 2 ulps
also fails for some inputs, the bound parameterization is genuinely
needed.)

**B4.** The destination-dominated rule applies cleanly to sub.
(YES — sub IS coarsening when going from `p_high` to `p_low`.
The rule is `≤ 0.5 ulp at destination per step`. The test's path 1
has 3 such steps (round a, round b, sub-rounding); path 2 has 2
steps (sub at p_high which is ≤ 0.5 ulp at p_high ≪ 0.5 ulp at
p_low, then round to p_low). Path 2 ≤ 0.5 ulp + tiny ≈ 0.5 ulp at
p_low. Path 1 ≤ 1.5 ulp at p_low. Maximum disagreement: 2 ulps at
p_low. **Confirmed.**)

**B5.** What if the actual implementation has a bug that
coincidentally produces 2 ulps disagreement? This is the
"wrong-but-load-bearing" possibility — the disagreement looks like
the structural bound but is actually a bug. **To rule this out**:
run the test with `within_2_ulps`, see if it passes. If yes, the
implementation is correct under the bound. If no, there's a
deeper bug.

---

## Phase 7 — Recursive Process

Add B1-B5 to the assumption pile.

- B1: confirmed.
- B2: explanation traced. The original test author understood add
  but didn't generalize to sub.
- B3: needs runtime verification. **Action item**: run with relaxed
  2-ulp bound.
- B4: confirmed via T1-T5 analysis.
- B5: rules out bugs IF the relaxed-2-ulp test passes.

The picture is stable. The `sub_alignment_stress` test failure is
**the test's bound being too tight, not an implementation bug.**

---

## Phase 8 — Forced Rejection

Forcibly reject the conclusion. What if the implementation IS buggy?

**Rejection 1 — the 2-ulp gap is a real implementation bug, not a
bound issue.** What would this look like? An off-by-one in the
sub's round/sticky logic that produces a 2-ulp shift in some edge
cases. **Counter-evidence:** the add test passes with the same
construction at 1 ulp. The same operands, same alignment shift,
same canonicalize_and_round path — only the sign-handling differs
between add and sub (BZ §3.1 magnitude-subtraction vs magnitude-
addition). If sub had a 2-ulp bug, it would manifest as a SYSTEMATIC
shift across all sub samples, not as a 2-ulp gap that occurs at
specific cancellation factors.

**Rejection 2 — the 2-ulp gap is the recently-fixed cancellation
borrow underflow (task #8) leaking through.** Task #8 fixed the
cmp=Equal case where small had hidden round/sticky. The current
test repro has cmp != Equal (the operands have different exponents
and dense different mantissas). Task #8's fix doesn't apply here.

**Rejection 3 — the bound should actually be 1.5 ulp, not 2.** Per
T5, the maximum disagreement is 1.5 ulp absolute. If `within_n_ulps`
allows fractional N, asserting 1.5 ulp would be more discriminating.
But `within_1_ulp_bigfloat` only checks the lowest-limb difference
≤ 1; extending to fractional ulps requires a richer comparison.
Practically, 2 ulps is the tightest INTEGER bound. The relaxation
from "≤ 1" to "≤ 2" in the lowest-limb integer rep IS the
"≤ 1.5 ulp" mathematical bound, rounded up.

**Rejection 4 — the test is testing the wrong invariant entirely.**
The test name is "cross-precision consistency"; the *intended*
invariant might be "the destination-dominated rule yields agreement
within the rule's bound," not "two paths agree within 1 ulp." The
test's failure indicates the test's *operationalization* of the
intent is wrong — but the intent is right. **My recommendation in
Phase 5 fixes this**: parameterize the bound by the structural rule,
not by a single ulp count.

**Rejection 5 — there's a deeper issue with `with_precision_rounded`
that produces incorrect rounding for sub-result inputs.** Need to
verify by reading `round_to_precision` implementation. If
`round_to_precision` incorrectly handles the case where the input
mantissa has bits set far below the LSB (which happens in sub's
result), there's a real bug. **This is the only Rejection that
warrants further code investigation.**

---

## Code investigation: `round_to_precision`

(Aside-investigation — does `round_to_precision` correctly handle
sub-result inputs?)

From `arith.rs:1303-1314`:

```rust
pub(crate) fn round_to_precision(
    bf: &BigFloat,
    new_precision: u32,
    rounding: RoundingMode,
) -> BigFloat {
    if !bf.is_normal() { ... }
    let exp_at_lsb = bf.exponent - bf.precision_bits as i64 + 1;
    canonicalize_and_round(
        bf.limbs.clone(),
        exp_at_lsb,
        bf.sign,
        0, 0,                    // no round/sticky from input
        new_precision,
        rounding,
    )
}
```

The `(0, 0)` round/sticky is correct here: the input bf's limbs
ARE the integer; there are no bits *below* its LSB. The
canonicalize_and_round will shift right (when new_precision <
bf.precision_bits) and capture round/sticky from the right-shift
itself. This is correct.

**Potential issue**: when sub produces a result that's renormalized
(top bit shifts left after cancellation), the result's limbs at
p_high have non-zero bits across the entire mantissa range,
including positions just below `p_low - 1`. The `with_precision_rounded`
to p_low triggers the right-shift, which captures those bits as
round/sticky correctly. **No issue.**

**Conclusion**: Rejection 5 doesn't fire. The implementation is
correct under the structural rule. The test's bound is wrong.

---

## What MUST be true under the rejections

- R1: rejected. add passes, sub differs only by sign-handling.
- R2: rejected. cmp != Equal in the test repro.
- R3: refined. 2-ulp bound in integer-limb rep encodes 1.5-ulp
      mathematical bound.
- R4: rejected. Intent is right; operationalization needs
      parameterization.
- R5: rejected. round_to_precision is correct.

The deepest finding: **the `sub_alignment_stress` test is correct
about its INTENT (verify cross-precision agreement under the
destination-dominated rule) but wrong about its OPERATIONALIZATION
(asserting a 1-ulp bound when the rule's bound is 2 ulps for sub
with cancellation).**

---

## Cross-cutting findings

### Finding 1 — The 1-ulp vs 2-ulp gap is a Phase 8 rejection-rooted
recognition

The add test's 1-ulp bound is tight (within the integer-limb rep).
The sub test's 1-ulp bound is too tight by 1. The *reason* is
cancellation amplification — same operation kind (BZ Algorithm 3.1),
different sign-handling, different error-composition behavior.

This is the operationalization of my prior gauntlet's Surface 2
("subnormal regime crossing") at a different layer: **the rule has
a regime where it weakens, and the test must encode the regime-
specific bound.** Surface 2 weakened the bound for subnormal
inputs; the sub test needs to weaken the bound for cancellation
inputs.

### Finding 2 — Same shape as the F13 antibody pattern

Per F13, every rule with a scope precondition needs an antibody
that enforces the precondition at construction time. Here:
- **Rule**: cross-precision agreement within K ulps.
- **Scope precondition**: K depends on operation, cancellation
  factor, regime.
- **Antibody (current)**: assert K=1 always. Fires false-positive
  when the structural K is 2.
- **Antibody (correct)**: parameterize K by the structural rule
  ([cancellation_ratio, op_kind, p_gap]).

### Finding 3 — The test author had the right reasoning for add and
did not generalize

The comment at lines 1161-1165 of the add test cites BZ §3.1.6 and
correctly derives the 1-ulp bound for add. The comment does NOT
appear in the sub test. The author understood add's bound and
duplicated the test for sub without re-deriving. **This is a
common pattern: a rule that works for one op gets copy-pasted to
another op without checking that the rule still applies.** F11
recognition: the rule's scope is op-specific; the operationalization
should be too.

### Finding 4 — The path-budget rule from DEC-031 §3.2 anchors this

DEC-031 §3.2 destination-dominated path budget: for monotonically-
coarsening paths, `ulp_budget_path = max(ulp_at_destination(step)
for step in path)`. The MAX captures that errors don't compound in
the coarsening direction. The cross-precision test's path is more
complex than monotonic-coarsening (it has an arithmetic step),
which is why it needs a different bound. **The path budget rule
applies to coarsening; arithmetic adds its own per-op bound.**

The right framing: cross-precision agreement = (path-budget bound)
+ (arithmetic per-op bound). For add: 0.5 + 0.5 = 1 ulp. For sub at
mild cancellation: 0.5 + 1 = 1.5 ulp (rounded up to 2 in integer
rep). For sub at extreme cancellation: 0.5 + N ulps, where N grows
with cancellation factor.

---

## Recommendation

**For the team to consider** (not a directive — Sweep 32 is reserved
and these aren't bugs):

1. **The `phase_c_sub_alignment_stress` test is correct in intent
   but wrong in bound.** The 1-ulp bound should be relaxed to 2 ulps
   for sub. Suggested edit:

```rust
proptest! {
    #[test]
    fn phase_c_sub_alignment_stress(...) {
        // ... (unchanged setup)
        prop_assert!(
            within_2_ulps_bigfloat(&direct, &via_high),
            // (updated message)
        );
    }
}
```

   And add a `within_2_ulps_bigfloat` helper that allows lowest-limb
   diff up to 2.

2. **Document the structural rule in the test comment.** Cite BZ
   §3.1.6 plus DEC-031 §3.2 destination-dominated rule, plus the
   cancellation-amplification factor. Make the bound's origin
   legible.

3. **Add a stricter complementary test that excludes cancellation.**
   `phase_c_sub_no_cancellation`: same construction, but require
   `result_magnitude > 0.9 * max(operand_magnitudes)` (i.e., delta=0
   plus same-sign operands so the sub doesn't actually cancel).
   This test uses the 1-ulp bound and ensures the implementation
   doesn't have a non-cancellation bug.

4. **Phase C div_round_bit_boundary and other ignored tests likely
   have the same root cause** — a bound that's too tight for the
   structural rule. Run each with relaxed bounds; fix the bounds
   not the code.

This is structural-rule clarification, not a bug fix. The current
implementation is mathematically correct under the right rule.

---

## Status

Phase 1-8 walk complete. The `sub_alignment_stress` failure is the
**test's bound being too tight by 1 ulp** in integer rep (≈ 0.5 ulp
mathematical), specifically for the cancellation regime. The
implementation is correct under the structural rule from BZ §3.1.6
+ DEC-031 §3.2.

**THIS CONCLUSION WAS WRONG. See RETRACTION section below.**

---

## RETRACTION (2026-05-08, post-navigator-challenge + math-researcher task #13)

**The framing in this document is incorrect.** Navigator pushed
back: "limbs differing by exactly 2 — pattern suggests guard-bit
error two steps before the final round rather than in the final
round itself." I ran the test 8 times with disabled proptest
persistence, observed `via_high - direct = +2` in every single
case (consistent direction). I attributed this to shrinker bias.

**Math-researcher's task #13 found the actual root cause**:
`canonicalize_and_round` at arith.rs:927-950 has a bug. When
left-shifting by `n=1` (common in sub-cancellation), the code
absorbs round_bit into mag's bit 0 but **drops sticky_bit** because
the guard `if sticky_bit != 0 && n >= 2` rejects n=1. The
downstream rounding then sees `(round=0, sticky=0)` and treats the
result as exact when it isn't.

```rust
} else if shift < 0 {
    let n = (-shift) as u32;
    limbs::shl_limbs_in_place(&mut mag, n);
    if round_bit != 0 && n >= 1 {
        limbs::set_bit(&mut mag, n - 1);
    }
    if sticky_bit != 0 && n >= 2 {       // ← BUG: n=1 case loses sticky
        limbs::set_bit(&mut mag, 0);
    }
    round_bit = 0;
    sticky_bit = 0;
}
```

### What I missed

**1. Phase 8 forced rejection should have been more aggressive.**
My Rejection 5 ("there's a deeper issue with `with_precision_rounded`")
was the right shape but I dismissed it after a cursory read of
`round_to_precision`. The real bug was one level deeper, in
`canonicalize_and_round`'s shift-direction handling. A more
thorough Phase 8 would have walked through every shift case.

**2. The "+2 across 8 samples" pattern was the smoking gun.** A
random-rounding-noise hypothesis predicts roughly equal +2 and -2
directions; one-direction +2 across 8 distinct shrunken inputs is
strong evidence against random noise. I attributed it to shrinker
bias. **Substrate over memory: I should have run the diagnostic on
non-shrunken random inputs to verify the direction-distribution
before committing to a framing.**

**3. The structural-rule analysis I did is valid as a UPPER BOUND
under correct per-step rounding.** What I observed is a SYSTEMATIC
bias from a single buggy step that happens to be 2 ulps because of
the n=1 left-shift math. Coincidental numerical agreement led me to
accept the wrong mechanism.

**4. Navigator's "guard-bit error two steps before the final round"
was the right framing.** The "two steps before" maps to the
`canonicalize_and_round` invocation INSIDE the sub itself (which
left-shifts by 1 due to cancellation), which feeds an
incorrectly-canonicalized result into `with_precision_rounded` (the
"final round" step). The bug is upstream of the final round.

### Correction

The `phase_c_sub_alignment_stress` test's 1-ulp bound is **correct
and tight** under correct per-step rounding. The test failure
indicates a real implementation bug, not a too-tight bound. Once
math-researcher's task #13 fix lands (option A: preserve sticky
across n=1 left-shift; option B: architectural fix with persistent
guard bits in mag's working buffer; option C: detect and pre-widen
the work buffer in normal_add_multilimb), the test should pass at
the strict 1-ulp bound without modification.

**Disregard the "relax to 2 ulps" recommendation in the body of
this document.** The test bound is correct.

### Where the structural-rule framing is still useful

The path-budget analysis (path 1 ≤ 1.5 ulp; path 2 ≤ 0.5 ulp;
disagreement ≤ 2 ulps absolute) IS a valid worst-case bound —
under the assumption of correct per-step rounding. The
`within_1_ulp_bigfloat` check is tight only when each step is
correctly rounded; under that assumption, the 1-ulp bound IS
achievable because the per-step errors usually partially cancel.
The 2-ulp absolute bound is the maximum, not the typical.

The implementation must hit the 1-ulp typical case. The buggy
canonicalize-on-n=1-left-shift makes sub systematically miss
the typical, by 1-2 ulps, which is what the test catches.

### Lesson for future deconstructions

**Forced rejection (Phase 8) should look for what we're missing,
not validate what we already believe.** When my Rejection 5
identified "a deeper issue with the rounding pipeline," I read
`round_to_precision` (one layer) and stopped. I should have read
all callees to bottom (canonicalize_and_round → shr_limbs_with_sticky
→ etc.) and checked every shift-direction case.

**A SYSTEMATIC bias direction is strong evidence against
"structural rule's bound" framings.** Random-rounding-error
analyses predict roughly symmetric error distributions. Asymmetric
distributions point to systematic bugs. Future Phase 7-style
recursive challenge should treat distribution asymmetry as a
signal that demands its own mechanism, not a coincidence to
explain away.

Output filed at: `R:\winrapids\campsites\tambear-sweep31-finish\
20260508161448-aristotle-invariants\sub-cross-precision-deconstruction.md`

Routing this to navigator as a structural finding worth surfacing.
Not modifying any code (Sweep 32 lane is reserved). Not modifying
the ignored test file (it's a tests file, but my job here is
deconstruction, not impl).
