---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (multi-limb arith — invariant catalog for the gauntlet)
audience: scientist (cross-precision harness lead), adversarial (gauntlet designer), navigator (route trait-promotion decision)
trigger: naturalist's structural noticing — the BZ skeleton is the same 5 steps across add/mul/div/sqrt; only step 3 (core arithmetic loop) differs. The gauntlet at p ∈ {53, 106, 107, 200, 500, 1024} × 4 ops × ~10 invariants × 5 rounding modes is ~1200 cases, mostly variations on the same skeleton. Convention-to-declaration tipping point is approaching.
purpose: enumerate the ~10 mathematical invariants per op at the BZ-text level so whoever lands the test-trait abstraction has the invariants in hand. Math-researcher's contribution to the trait promotion is the invariant catalog, not the trait Rust code.
inputs:
  - bz-impl-reference.md §1.5, §2.4, §3.5, §4.5 (each algorithm's "what pathmaker can verify against")
  - div-sqrt-newton-design.md §1.5, §2.5, §6 (edge cases + smoke tests)
  - DESIGN.md §3 (algorithm dispatch table)
  - oracle-validation.md §1.2 (verification-tier discipline)
  - BZ §3.1, §3.3, §3.5, §3.10 (the algorithm boxes themselves)
---

# BigFloat Op Invariant Catalog — The Mathematical Truth Per Op

> **What this is**. For each of (add, sub, mul, div, sqrt), the catalog of
> mathematical invariants the implementation MUST satisfy. Each invariant is
> stated at the BZ-text level (i.e., at the level the algorithm declares it
> true for), with the verification idiom and the cross-op sharing pattern.
>
> **Why this exists**. Naturalist noticed the BZ skeleton is shared across
> the four ops — only step 3 (core limb arithmetic) differs. Steps 1, 2, 4, 5
> (special-value dispatch, precision extension, final round, canonicalize)
> are isomorphic. **The invariants follow the same pattern**: many are shared
> across ops (commutativity-when-applicable, identity element, IEEE 754 sign
> rules, NaN propagation, round-trip with f64 fast path, cross-precision
> consistency, mpmath bit-exact agreement, canonical-form post-condition).
> A few are op-specific (sub vs add zero-arithmetic table; mul carry-vs-no-
> carry top-bit; div self-divide identity; sqrt perfect-square exactness).
>
> **The trait shape this implies** (for whoever lands the abstraction): an
> `OpInvariants` trait with associated functions for each invariant family;
> per-op impls override only the op-specific cases.

---

## 0. Invariant taxonomy — six families

Per the BZ skeleton (steps 1, 2, 4, 5 isomorphic; step 3 op-specific):

| Family | Shared / op-specific | Verifies |
|---|---|---|
| **F-A: Special-value propagation** | shared | Step 1 (kind dispatch) — NaN propagates, ±Inf rules per IEEE 754, ±0 rules |
| **F-B: Identity / fixed-point** | mostly shared, with op-specific identity element | `a + 0 = a`, `a · 1 = a`, `a / 1 = a`, `√1 = 1` |
| **F-C: IEEE 754 sign rules** | shared structure, op-specific table | sign-of-product, sign-of-zero-arithmetic per RoundingMode |
| **F-D: Algebraic structure** | op-specific | commutativity (add/mul), self-inverse (`a / a = 1`, `√(a²) = |a|`) |
| **F-E: Round-trip + fast-path agreement** | shared | f64 fast path AGREES with multi-limb path bit-exactly when both apply |
| **F-F: Cross-precision consistency** | shared | `op(a, b, p_low) == round(op(a, b, p_high), p_low)` |
| **F-G: Canonical-form post-condition** | shared | result has top-bit-set at `(p_out - 1) % 64` of top limb, `limbs.len() == ⌈p_out/64⌉` |
| **F-H: Oracle peer-equivalence** | shared | mpmath bit-exact at matched precision (verification-tier per oracle-validation.md §1.2) |

Six families. Add ~10 specific invariants under each family for the full catalog. Below.

---

## 1. Add — invariants

| ID | Family | Invariant | Verification idiom |
|---|---|---|---|
| ADD-1 | F-A | NaN propagation: `NaN + x == NaN` (payload preserved from the NaN operand; if both are NaN, payload from `self`) | proptest with `is_nan` operands |
| ADD-2 | F-A | `(+Inf) + finite == +Inf`, `(-Inf) + finite == -Inf` | proptest with Inf × Normal cases |
| ADD-3 | F-A | `(+Inf) + (-Inf) == NaN` (invalid op) | direct test |
| ADD-4 | F-C | Zero-arithmetic Surface 8 table (4 cases × 5 rounding modes = 20 cells) | enumerated table test, already in arith_tests.rs |
| ADD-5 | F-B | Identity: `a + 0 == a` for any a, any rounding mode | proptest, all kinds for a |
| ADD-6 | F-D | Commutativity: `a + b == b + a` bit-exact, all rounding modes | proptest |
| ADD-7 | F-D | Cancellation: `a + (-a) == +0` (under default rounding); `(-0) under RoundTowardNegativeInfinity` | direct table |
| ADD-8 | F-E | f64-fast-path agreement: when `f64_path_eligible(a) && f64_path_eligible(b)` AND result is f64-eligible, multi-limb add must produce identical bits | proptest where forcing forces multi-limb path despite f64-eligible inputs (per bz-impl-reference.md §1.5: `1.0 + 2^-100`) |
| ADD-9 | F-F | Cross-precision consistency: `add(a, b, p_low, RTE) == round_to_p_low(add(a, b, p_high, RTE))` for `p_high > p_low` | proptest per adversarial-round-bit-generators.md §2 |
| ADD-10 | F-G | Canonical form: result has top-bit-set, correct limb count, exponent in valid range | post-condition assertion |
| ADD-11 | F-H | mpmath bit-exact: `tambear_add(a, b, p, RTE).limbs == mpmath.fadd(a, b, prec=p).limbs` | verification-tier harness |
| ADD-12 | F-D | Associativity-mod-rounding: `(a + b) + c ≈ a + (b + c)` to within 1 ULP at p (NOT bit-exact — rounding accumulates) | discovery-tier ULP test |

**Total**: 12 invariants for add. ADD-4 expands to 20 cells; ADD-9 expands across `p_low × p_high` precision pairs; ADD-11 expands across the precision grid {53, 106, 107, 200, 500, 1024} × 5 rounding modes. Full case count for add ≈ 100-200.

---

## 2. Sub — invariants

Sub is `add(a, neg(b))` per arith.rs:209. **Sub doesn't introduce new invariants** beyond add — sub-specific cases are subsumed by add's tests with negated operands.

**Exception**: SUB-4 (zero-arithmetic Surface 8 has a different table for sub vs add per `sub_zero_arithmetic_sign` in arith.rs:86-89). The 4 cases × 5 rounding modes table differs from add's; both must be tested.

The trait implementation for Sub can be a thin wrapper that:
- Inherits ADD-1, 2, 3, 5, 6, 8, 9, 10, 11, 12 from Add (with sign flip on b).
- Overrides SUB-4 with the sub-specific zero-arithmetic table.
- Adds SUB-7: `a - a == +0` under default; `(-0) under RoundTowardNegativeInfinity`. (Different from ADD-7 — Add's cancellation is `a + (-a)`, Sub's is `a - a`. They're the same arithmetic but different test surfaces.)

**Total**: 12 invariants, 11 inherited + 1 overridden + 1 added. The trait's "implementor only writes the op-specific bits" pattern is visible already.

---

## 3. Mul — invariants

| ID | Family | Invariant | Verification idiom |
|---|---|---|---|
| MUL-1 | F-A | NaN propagation | proptest |
| MUL-2 | F-A | `0 × Inf == NaN` (invalid op per IEEE 754) | direct |
| MUL-3 | F-A | `Inf × non-zero finite == Inf with sign per mul_sign` | proptest |
| MUL-4 | F-C | Zero × finite: sign per `mul_sign(self.sign, rhs.sign)`, kind = Zero | proptest |
| MUL-5 | F-B | Identity: `a · 1 == a` (where `1 = BigFloat::from_f64(1.0, p)`) | proptest, all kinds for a |
| MUL-6 | F-D | Commutativity: `a · b == b · a` bit-exact, all rounding modes | proptest |
| MUL-7 | F-D | Distributivity-mod-rounding: `a · (b + c) ≈ a·b + a·c` to within 2 ULP at p | discovery-tier ULP |
| MUL-8 | F-E | f64-fast-path agreement when both eligible | proptest with construction-forced multi-limb |
| MUL-9 | F-F | Cross-precision consistency: same shape as ADD-9 | proptest per generators.md §3 (modular-inverse construction) |
| MUL-10 | F-G | Canonical form post-condition | post-condition |
| MUL-11 | F-H | mpmath bit-exact at p × 5 rounding modes | verification-tier harness |
| MUL-12 | (mul-specific) | Carry-vs-no-carry top-bit: result exponent is `a.exponent + b.exponent + 1` if product's top bit at position `p_a + p_b - 1`, else `a.exponent + b.exponent` (with mantissa left-shift by 1) | this is the silent-bug surface for mul; needs explicit test |
| MUL-13 | (mul-specific) | DD cross-check: `BigFloat::mul(a, b, 200) == DD::mul_ext(a, b).to_bigfloat(200)` for f64-sourced a, b (since DD's mul-ext is exact at 106 bits, fits cleanly in BigFloat at 200) | cross-impl test |

**Total**: 13 invariants for mul. MUL-12 is the BZ §3.3-specific carry-vs-no-carry detection (per bz-impl-reference.md §2.2) — the place where most-libraries-get-it-wrong.

---

## 4. Div — invariants

| ID | Family | Invariant | Verification idiom |
|---|---|---|---|
| DIV-1 | F-A | NaN propagation; `Inf/Inf == NaN`; `0/0 == NaN` | proptest + direct |
| DIV-2 | F-A | `finite/0 == ±Inf` (sign per mul_sign); `0/finite == ±0` (sign per mul_sign) | direct |
| DIV-3 | F-A | `Inf/finite == ±Inf`; `finite/Inf == ±0` | direct |
| DIV-4 | F-B | Identity: `a / 1 == a` (right-identity; div is non-commutative so no left-identity) | proptest |
| DIV-5 | F-D | Self-inverse: `a / a == 1.0` bit-exact for any a (multi-limb included) | proptest, all multi-limb a |
| DIV-6 | F-D | Power-of-2 exact: `a / 2^k == a · 2^-k` exactly (mantissa unchanged, exponent decrement only, no rounding) | direct |
| DIV-7 | F-E | f64-fast-path agreement | proptest |
| DIV-8 | F-F | Cross-precision consistency | proptest per generators.md §4 |
| DIV-9 | F-G | Canonical form post-condition | post-condition |
| DIV-10 | F-H | mpmath bit-exact | verification-tier harness |
| DIV-11 | (div-specific) | Newton-iteration convergence: after `⌈log₂(p_work/53)⌉ + 2` iterations, `\|x_n - 1/b\| ≤ 2^-(p_work)` (bound on the reciprocal) | white-box test of newton_reciprocal |
| DIV-12 | (div-specific) | Scaling-step round-trip: `b_scaled.exponent = -1` then unscale `x.exponent -= original_b_exponent + 1` produces the right magnitude | unit test of the scaling encoding-derivation per div-sqrt-newton-design.md §1.3 |
| DIV-13 | (div-specific) | Reciprocal as multiplicative inverse: `b · (1/b) == 1.0` to within 1 ULP at result precision (note: NOT bit-exact in general because `1/b` has rounding error, but `b · (1/b)` should be 1.0 ± 1 ULP at p) | discovery-tier ULP |

**Total**: 13 invariants for div. DIV-11 + DIV-12 are the silent-bug-surface pair from the Newton iteration design.

---

## 5. Sqrt — invariants

| ID | Family | Invariant | Verification idiom |
|---|---|---|---|
| SQRT-1 | F-A | NaN propagation; `sqrt(NaN) == NaN` | direct |
| SQRT-2 | F-A | `sqrt(+Inf) == +Inf`; `sqrt(-Inf) == NaN`; `sqrt(-x for x > 0) == NaN` | direct |
| SQRT-3 | F-C | `sqrt(±0) == ±0` (sign preserved per IEEE 754 §6.3) | direct |
| SQRT-4 | F-B | Fixed point: `sqrt(1.0) == 1.0` exactly | direct |
| SQRT-5 | F-D | Square round-trip: `sqrt(a · a) == \|a\|` bit-exact for any a (multi-limb included) | proptest |
| SQRT-6 | F-D | Square-of-sqrt: `(sqrt(a))² ≈ a` to within 2 ULP at p (cumulative rounding) | discovery-tier ULP |
| SQRT-7 | F-E | f64-fast-path agreement | proptest |
| SQRT-8 | F-F | Cross-precision consistency | proptest per generators.md §5 |
| SQRT-9 | F-G | Canonical form post-condition | post-condition |
| SQRT-10 | F-H | mpmath bit-exact | verification-tier harness |
| SQRT-11 | (sqrt-specific) | Newton-iteration convergence: same shape as DIV-11 | white-box test |
| SQRT-12 | (sqrt-specific) | Scaling-step parity invariant: `(a.exponent - a_scaled.exponent) % 2 == 0` always (so the unscale shift is integer) | unit test of the parity logic per div-sqrt-newton-design.md §2.3 |
| SQRT-13 | (sqrt-specific) | Perfect-square shortcut: `sqrt(M·M)` for multi-limb M produces M exactly (no rounding error at p_work because Newton converges to exact when the input is an exact square) | proptest with constructed perfect squares |
| SQRT-14 | (sqrt-specific) | `sqrt(4) == 2`, `sqrt(9) == 3`, `sqrt(16) == 4`, `sqrt(2^(2k)) == 2^k` for any integer k | direct table |

**Total**: 14 invariants for sqrt. SQRT-11 + SQRT-12 are the Newton-iteration-specific surface; SQRT-13 + SQRT-14 are the perfect-square verification cases.

---

## 6. Cross-op invariants — what binds the catalog together

Beyond per-op invariants, several **cross-op identities** verify the algebraic structure:

| ID | Cross-op | Identity | Verification |
|---|---|---|---|
| CO-1 | mul ↔ div | `(a · b) / b == a` to within 1 ULP at p (NOT bit-exact; rounding) | discovery-tier ULP, multi-limb a |
| CO-2 | mul ↔ sqrt | `sqrt(a · a) == \|a\|` bit-exact (already SQRT-5) | covered |
| CO-3 | add ↔ sub | `(a + b) - b == a` to within 1 ULP | discovery-tier |
| CO-4 | add ↔ mul | `a · 2.0 == a + a` bit-exact (mul by 2 is exponent shift; add same operand) | direct |
| CO-5 | div ↔ sqrt | `sqrt(a) · sqrt(a) ≈ a` (already SQRT-6) | covered |

**5 cross-op invariants**. These are the "the math works as a system" tests; they catch the case where each op is locally correct but composition produces drift.

---

## 7. Trait shape — what the catalog implies

If/when whoever lands the test-trait abstraction reaches for it, the catalog implies:

```rust
pub trait OpInvariants {
    type Op; // Add, Sub, Mul, Div, Sqrt phantom

    // F-A: Special-value propagation (mostly shared, with op-specific tables)
    fn nan_propagation_test(p: u32);
    fn infinity_arithmetic_table() -> &'static [InfCase];
    fn zero_arithmetic_table(rounding: RoundingMode) -> &'static [ZeroCase];

    // F-B: Identity element (op-specific identity)
    fn identity_element(p: u32) -> BigFloat;  // 0 for add, 1 for mul, 1 for div (right), 1 for sqrt (fixed point)
    fn identity_test(p: u32);

    // F-D: Algebraic structure (op-specific)
    fn commutative() -> bool;  // true for add/mul, false for sub/div/sqrt
    fn commutativity_test(p: u32) where Self: Commutative;
    fn self_inverse_test(p: u32);  // a/a, sqrt(a²), a-a, etc.

    // F-E, F-F, F-G, F-H: shared structure
    fn fast_path_agreement_test(p: u32);
    fn cross_precision_consistency_test(p_low: u32, p_high: u32, rounding: RoundingMode);
    fn canonical_form_post_condition(result: &BigFloat, p_out: u32);
    fn mpmath_oracle_test(inputs: &[Self::Inputs], p: u32, rounding: RoundingMode);

    // Op-specific (no default impl; each op writes its own)
    fn op_specific_invariants(p: u32);
}
```

Each op-impl is small (~50 LoC) because most invariants share the same shape; the op-specific bits are 1-3 invariants per op.

**The trait is what makes adding a new op cheap**. When sweep 35 ships BigFloat transcendentals (`exp`, `log`, `sin`, `cos`), each new op gets the same gauntlet shape for free — implement the trait, define `op_specific_invariants`, done.

**This is anti-YAGNI confirmation**: the trait isn't speculative. The structural-guarantee that "we will have N more ops in sweep 35+ each needing the same gauntlet" makes the trait load-bearing now, not later.

---

## 8. Promotion-timing recommendation for navigator

Naturalist's framing: convention is correct at small scale; promote when convention has failed or is about to.

**My read**: promote NOW, in this session, before pathmaker lands BZ 3.1 add. Reasoning:

1. **Anti-YAGNI structural guarantee**: gauntlet at p × ops × invariants × rounding modes is structurally inevitable per oracle-validation.md verification-tier discipline. Not speculative.

2. **Cost asymmetry**: trait promotion now is a few hours of work (scientist + adversarial collaborate; math-researcher provides catalog). Refactoring 1200 nearly-identical test functions later is days.

3. **Convergence-of-roles**: pathmaker's add impl will land soon. If the trait is in place when add lands, scientist's harness can run all 12 add invariants from day 1 instead of cherry-picking.

4. **The catalog is the hard part**. The trait Rust code is mechanical. By writing this catalog, math-researcher has done the load-bearing work; whoever ships the trait just translates.

**But this is navigator's call, not mine**. I'm flagging the structural finding and providing the substrate. If navigator decides "wait until adversarial finalizes the proptest gauntlet shape, then promote together," that's also reasonable — adversarial may have constraints on the trait shape I don't see.

---

## 9. Provenance + handoff

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish`, in response to naturalist's structural noticing about the BZ skeleton being shared across the four ops.
- Substrate: bz-impl-reference.md (§1.5, §2.4, §3.5, §4.5), div-sqrt-newton-design.md (§1.5, §2.5), DESIGN.md §3, oracle-validation.md §1.2.
- Cross-checked: per-op invariant counts: add 12, sub 12 (mostly inherited), mul 13, div 13, sqrt 14, cross-op 5 = ~70 distinct invariants. Across precision grid {53, 106, 107, 200, 500, 1024} × 5 rounding modes ≈ 70 × 6 × 5 = 2100 cases. Naturalist's "240+" was conservative — at full coverage, the gauntlet is ~2000-cell.
- The trait shape in §7 is one possible factoring; scientist + adversarial may shape it differently. The catalog (§1-§6) is the load-bearing artifact regardless of trait choice.
- Open question for navigator: route to scientist + adversarial for trait-promotion-timing decision? My read in §8: promote now. But navigator decides.

**This catalog is the math-researcher's deliverable.** Everything below the trait Rust line is engineering territory. My role was to make the invariants explicit at the BZ-text level so the engineering decision has the right substrate.

— math-researcher, in dialogue with naturalist's noticing
