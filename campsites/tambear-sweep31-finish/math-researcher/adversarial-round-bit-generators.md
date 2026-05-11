---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (multi-limb arith unstub — adversarial generator design)
audience: scientist (cross-precision harness lead), adversarial (proptest gauntlet design)
purpose: concrete proptest input-strategy designs for the round-bit-on-edge and sticky-bit-on-edge cases that exercise the round-bit logic in BZ Algorithm 3.1 / 3.3 / 3.5 / 3.10.
inputs:
  - bz-impl-reference.md §6 (cross-precision consistency check sketch)
  - oracle-validation.md §1.2 (verification-tier discipline + cross-precision idiom)
  - BZ §3.1.6 (round-bit + sticky-bit theory)
  - proptest::strategy::Strategy trait
---

# Adversarial Round-Bit Generators — Strategies that Hit the Boundary

> **Why this matters**. Random-uniform input proptests are good for "code doesn't
> crash" but bad for "round-bit logic is correct." With probability ≈ 2^-p, a
> uniform-random pair of inputs has a result mantissa with bits exactly on the
> round/sticky boundary at precision p. For p=200, that's 1 in 10^60 — not
> happening in any CI run. We need *constructive* generators that target the
> boundary directly.
>
> **The strategy**. Instead of sampling inputs uniformly and hoping a result
> hits the boundary, we sample the *result-mantissa-bit-pattern* and construct
> inputs that produce it. This inverts the search space: the boundary becomes
> dense in the generator's image, instead of measure-zero in random sampling.

---

## 1. The boundary classifications — a precise definition

Per BZ §3.1.6, the round-to-nearest-ties-even decision for a result mantissa M (at working precision `p_high`) being rounded to `p_low` examines bits `(p_high - 1) .. (p_high - p_low)` (the kept bits) and bits `(p_high - p_low - 1) .. 0` (the discarded bits). Define:

- **Round bit** R = bit `(p_high - p_low - 1)` of M (the most-significant discarded bit).
- **Sticky bit** S = OR of bits `(p_high - p_low - 2) .. 0` of M.
- **LSB-of-kept** L = bit `(p_high - p_low)` of M (the least-significant kept bit).

Round-up decision (RoundToNearestTiesEven):
- If R = 0: round down (no change).
- If R = 1, S = 1: round up (the discarded portion is > 0.5 ULP).
- If R = 1, S = 0: tie. Round to even — round up iff L = 1.

**The four boundary classes** that adversarial generators must cover:

| Class | (R, S, L) | What it tests |
|---|---|---|
| **C1: Tie-down-stays** | (1, 0, 0) | "0.5 ULP exactly + last kept bit is 0" — must NOT round up (ties-to-even rounds to even, which is the down-direction). |
| **C2: Tie-up** | (1, 0, 1) | "0.5 ULP exactly + last kept bit is 1" — must round up to make the kept LSB even. |
| **C3: Round-up via sticky** | (1, 1, x) | "Just above 0.5 ULP" — must round up regardless of LSB. |
| **C4: Round-down** | (0, x, x) | "Just below 0.5 ULP or zero" — must round down. |

Each class has a precise (R, S, L) signature; the proptest must hit each.

**For directed rounding** (RoundTowardZero / RoundTowardPositiveInfinity / RoundTowardNegativeInfinity), the boundary classes are simpler: any (R, S) with at least one bit set rounds in the direction; (0, 0) doesn't round. So adversarial generators for directed rounding are subsumed by the round-to-nearest classes — if you hit C1 + C2 + C3 + C4, you've covered directed rounding too.

---

## 2. Generators for `add` / `sub` — by-construction

For BZ Algorithm 3.1 add at output precision `p_low`, computing internally at `p_high`:

**Goal**: produce input pair `(a, b)` such that `(a + b)`'s mantissa at `p_high` has a controlled (R, S, L) signature.

**Method — work backward from the desired result**:

1. **Pick the desired result mantissa** at `p_high` directly. For class C1: bit at position `p_high - p_low` (the LSB-of-kept) is 0; bit at position `p_high - p_low - 1` (the round bit) is 1; all bits below are 0. For class C2: same but L = 1. For C3: same as C1 but with at least one bit somewhere in the sticky range set. For C4: the round bit is 0 (everything else free).

2. **Pick the result exponent** uniformly in some range (e.g., `[-100, +100]` for in-bounds testing).

3. **Sample a random "perturbation" bit pattern** at `p_high` bits, subject to a constraint: the perturbation `p` should be small enough that `result_mantissa = a + b` AND `a` AND `b` are all valid `p_high`-bit BigFloats. Concretely: pick `a` with mantissa bit pattern `result_mantissa - perturbation` (treating both as `p_high`-bit integers), and `b` with mantissa bit pattern `perturbation`. As long as both are non-zero with no leading-zero shifts crossing limb boundaries that the test framework can't handle, this works.

4. **Construct `a` and `b` BigFloats** at `p_high` with:
   - `a.limbs = [bits of (result_mantissa - perturbation)]` packed top-bit-set
   - `b.limbs = [bits of perturbation]` packed top-bit-set (after shifting if needed)
   - Both at the same exponent (so add doesn't need shift-alignment for this case)
   - Or: pick `b` at a smaller exponent to test the alignment case

5. **Verify in the harness**: `a.add(&b, p_high) == result_mantissa as BigFloat at p_high`. If it doesn't, the multi-limb add is broken at the level of integer addition before the round-bit logic is even tested.

6. **Then check**: `a.add(&b, p_low) == round_to_p_low(result_at_p_high)`. The round-bit logic is what's tested here; class C1-C4 covers all signatures.

**Concrete proptest strategy (Rust, proptest crate)**:

```rust
use proptest::prelude::*;

#[derive(Debug, Clone, Copy)]
enum BoundaryClass { C1TieDownStays, C2TieUp, C3RoundUpSticky, C4RoundDown }

fn boundary_class_strategy() -> impl Strategy<Value = BoundaryClass> {
    prop_oneof![
        Just(BoundaryClass::C1TieDownStays),
        Just(BoundaryClass::C2TieUp),
        Just(BoundaryClass::C3RoundUpSticky),
        Just(BoundaryClass::C4RoundDown),
    ]
}

/// Given a class, p_low, p_high, sample a result-mantissa with the right
/// (R, S, L) signature. The mantissa is a `p_high`-bit unsigned integer with
/// top bit at position `p_high - 1`.
fn sample_result_mantissa(class: BoundaryClass, p_low: u32, p_high: u32) -> impl Strategy<Value = u64x2> {
    // Bit positions, where 0 = LSB:
    //   top bit at position (p_high - 1)
    //   LSB-of-kept at position (p_high - p_low)
    //   Round bit at position (p_high - p_low - 1)
    //   Sticky bits at positions (p_high - p_low - 2) .. 0
    let lsb_kept_pos = p_high - p_low;
    let round_bit_pos = p_high - p_low - 1;

    // Sample the kept bits (p_low - 1 bits below the top), the LSB-of-kept,
    // the round bit, and the sticky bits, all per the class.
    // ... build the mantissa accordingly.
    // (Implementation detail: returns a 2-limb representation since p_high
    // up to 1024 fits in 16 limbs; reduce to 2 for the test cases that
    // matter at the boundary.)
    todo!()
}
```

**The full strategy** (sketched):

```rust
proptest! {
    #[test]
    fn add_round_bit_correctness(
        class in boundary_class_strategy(),
        p_low in 107u32..=200,
        p_high_offset in 53u32..=200,
    ) {
        let p_high = p_low + p_high_offset;
        let result_mantissa_high = sample_result_mantissa(class, p_low, p_high);

        // Construct a, b at p_high such that a + b = result_mantissa_high
        // with controlled exponents.
        let (a, b) = construct_addition_pair(result_mantissa_high, p_high);

        // Compute at p_low directly:
        let direct = a.with_precision_rounded(p_low, RTE).add(
            &b.with_precision_rounded(p_low, RTE),
            RoundingMode::RoundToNearestTiesEven,
        );
        // Compute at p_high then round down:
        let via_high = a.add(&b, RoundingMode::RoundToNearestTiesEven)
            .with_precision_rounded(p_low, RoundingMode::RoundToNearestTiesEven);

        prop_assert_eq!(direct, via_high, "class {:?} p_low {} p_high {}: cross-precision consistency failed", class, p_low, p_high);

        // Per-class assertion on the rounding direction:
        match class {
            BoundaryClass::C1TieDownStays => {
                let expected_lsb = sample_result_mantissa_lsb(class);
                prop_assert_eq!(get_lsb(&via_high), expected_lsb,
                    "class C1 should round-down (keep L = 0)");
            },
            BoundaryClass::C2TieUp => {
                // L was 1 at p_high; after round-up, L flips to 0 with carry into next bit.
                // ... per-class assertion.
            },
            // ... etc
        }
    }
}
```

The implementation of `sample_result_mantissa` and `construct_addition_pair` is bookkeeping at the bit level — non-trivial but well-defined.

---

## 3. Generators for `mul` — same shape, different bit-pattern construction

For BZ Algorithm 3.3 mul at output `p_low`, the "result mantissa" is the `p_a + p_b`-bit product. The round-bit lives at position `p_a + p_b - p_low - 1` (i.e., `p_a + p_b - p_low - 1` bits below the top).

**Key difference from add**: we work backward from a desired product, then *factor* it into two operand mantissas.

For class C1 (tie-down-stays) at p_low=200, p_high=400: pick a product mantissa `P` of length `p_a + p_b` with the right (R, S, L) signature. Then sample a random factor `a` at `p_a` bits, compute `b = P / a` exactly (integer divide if `a | P` — pick `a` to be a divisor; otherwise pick `b` first then `a = P / b`), and verify both fit in their respective precisions.

**Constraints**:
- `p_a + p_b == p_high` (or close — must be exactly aligned with the schoolbook product width).
- Both `a` and `b` must be in canonical form (top bit set at the right position).
- If `P` has prime factors that don't fit in `p_a` bits, the sampling has to retry. Empirically, many `P` factor cleanly — but the worst case is `P = prime`, in which case `(a, b) = (1, P)` is the only factoring. **Fix**: pick `a` first uniformly, then `P` to be a multiple of `a`'s mantissa, then `b = P / a`. This always succeeds.

**Specific proptest sketch**:

```rust
proptest! {
    #[test]
    fn mul_round_bit_correctness(
        class in boundary_class_strategy(),
        p_low in 107u32..=200,
    ) {
        // For mul, the natural p_high is 2 × p_low (full schoolbook product).
        let p_a = p_low;
        let p_b = p_low;
        let p_high = p_a + p_b;

        // Pick a uniformly at p_a bits, top-bit-set.
        let a_mantissa = sample_canonical_mantissa(p_a);

        // Pick a target product mantissa pattern at p_high bits with the right
        // (R, S, L) signature. The product's top bit can be at position
        // p_high - 1 OR p_high - 2 (carry vs no-carry — see bz-impl-reference.md §2.2).
        // For now, force carry case (top bit at p_high - 1) — the other case
        // shifts the round-bit position by 1.
        let target_product = sample_product_with_class(class, p_a, p_b, p_low);

        // Compute b = target_product / a_mantissa, exactly.
        // For this to succeed, target_product must be divisible by a_mantissa
        // as integers. Achieve by: pick b_mantissa first, then target = a * b.
        let b_mantissa = sample_canonical_mantissa(p_b);
        let actual_product = a_mantissa.checked_mul_u128(b_mantissa);
        // ... verify actual_product has the right class signature; if not, resample.

        let a = bigfloat_from_mantissa_and_exponent(a_mantissa, 0, p_a);
        let b = bigfloat_from_mantissa_and_exponent(b_mantissa, 0, p_b);

        let direct = a.with_precision_rounded(p_low, RTE).mul(
            &b.with_precision_rounded(p_low, RTE),
            RoundingMode::RoundToNearestTiesEven,
        );
        let via_high = a.mul(&b, RoundingMode::RoundToNearestTiesEven)
            .with_precision_rounded(p_low, RTE);

        prop_assert_eq!(direct, via_high);
    }
}
```

**Practical note**: the "sample b_mantissa, compute product, check class signature, resample if wrong" pattern has a low acceptance rate (~1 in 4 per class). For 1000 test cases per class, this is ~4000 total samples — fast enough for CI.

**Better generator**: parameterize the `b_mantissa` sampling by the desired product's bottom bits. If we want `(a · b)` to have round-bit = 1 and sticky-bit = 0, we can construct `b` so that `(a · b) mod 2^(p_high - p_low)` equals exactly `2^(p_high - p_low - 1)` (i.e., the round bit alone is set). This is a modular-inversion problem: solve `a · b ≡ 2^(p_high - p_low - 1) (mod 2^(p_high - p_low))` for `b`. Since `a` is odd-mantissa (top bit set means odd if exponent is small enough; we control this by choosing `a` odd), `gcd(a, 2^k) = 1` and the modular inverse always exists.

This is cleaner than rejection sampling. Implementation: extended Euclidean algorithm modulo `2^(p_high - p_low)`. Standard cryptographic-arithmetic territory.

---

## 4. Generators for `div` — work backward from quotient

For BZ Algorithm 3.5 div at output `p_low`: pick a desired quotient mantissa `Q` at `p_high = p_low + 50` bits with the right class signature. Pick divisor `b` uniformly at `p_low` bits. Compute dividend `a = Q · b` (exactly, in BigFloat arithmetic at `p_low + p_high` bits — way more than needed for exact integer multiplication). Round `a` to `p_low` bits. Now `a / b ≈ Q` to within 1 ULP at `p_high`; the round-bit in the actual division will reflect the class.

**Subtle point**: because Newton iteration produces an approximation `x_n` to `1/b` with bounded error, `a · x_n` is `a · (1/b) ± a · ε` where `ε ≈ 2^-(p_high)` after enough iterations. The `2 + 50 + ⌈log₂(p_high/53)⌉` iteration count ensures `ε` is below the round-bit position.

**For the cross-precision consistency check, the generator doesn't care about Newton's intermediate accuracy** — it cares about the FINAL rounded result. So picking `(Q, b)` and constructing `a = round_to_p_low(Q · b)` produces a valid div test case with the right class signature, modulo the "Q · b might not round to a `p_low`-bit BigFloat that yields exactly Q" issue.

**Concrete strategy**: pick `b` at `p_low`. Pick `Q` at `p_high` with class signature. Compute `a_full = Q · b` at `p_low + p_high` bits. Round `a_full` to `p_low` bits, producing `a_low`. Test `a_low.div(b, p_low, RTE) == round_to_p_low(Q)`. The harness verifies the round-direction matches the class.

---

## 5. Generators for `sqrt` — work backward from result

For BZ Algorithm 3.10 sqrt: pick a desired sqrt mantissa `R` at `p_high` with class signature. Compute `a = R · R` (exact, at `p_high + p_high = 2·p_high` bits). Round `a` to `p_low` bits. Test `sqrt(a, p_low, RTE) == round_to_p_low(R)`.

Same pattern as div, simpler because it's unary.

**Edge case for sqrt specifically**: when `R²` is a perfect square at the chosen precision, `sqrt` should produce R exactly (no rounding error). Class C4 (R=0) is the perfect-square case — useful test for "does sqrt detect exact-square shortcut?"

---

## 6. Sampling at the limb boundary specifically

Within each algorithm's generator, there's an additional axis to vary: **where in the limb structure does the round-bit fall?** The round-bit at position `p_high - p_low - 1` could be:

- **In the same limb as the kept bits** (e.g., `p_low = 107, p_high = 200`: round-bit position 92, kept bits 93-199, both in limb 1 of the 2-limb representation of p=200).
- **At the boundary between two limbs** (e.g., `p_high = 128, p_low = 64`: round-bit position 63 = top of limb 0; kept bits at limb 1).
- **In a different limb than the kept bits** (e.g., multi-limb add where the round-bit falls in limb 1, kept bits in limbs 2+).

The proptest should explicitly cover all three. Specifically:

```rust
let p_low_choices = vec![107, 108, 127, 128, 129, 200, 201, 256];
let p_high_offsets = vec![1, 50, 63, 64, 65, 128, 200];
```

These cover: `p_low` near limb boundary at 64, 128, 192, 256; offsets that keep the round-bit in same limb / cross one limb boundary / cross two limb boundaries.

---

## 7. Alignment-stress tests for add/sub specifically

Beyond round-bit class coverage, BZ Algorithm 3.1 add has a *separate* stressor: the exponent alignment. Two operands with very different exponents test the bit-shift-by-many-positions code path.

**Generator**: pick `a` at `p_high`. Pick `δ ∈ [0, p_high + 50]` (exponent gap). Construct `b` at `p_high` with `b.exponent = a.exponent - δ`. Pick `b.limbs` to have a random pattern that, after shifting right by δ, lands at a controlled (R, S, L) signature with respect to `a`.

For `δ = 0`: same-exponent case (no shift needed); tests pure mantissa add.
For `δ = p_high - 1`: maximum overlap (b's top bit aligns with a's bottom bit); tests minimal sticky contribution.
For `δ = p_high + g`: b is below the round-bit position (round-bit gets only the sticky from b); tests the sticky-only contribution.
For `δ ≫ p_high + g`: b is below the sticky range entirely; tests the "fully ignored" path.

**Concrete proptest sketch**:

```rust
proptest! {
    #[test]
    fn add_alignment_stress(
        a_mantissa in any::<u64>(),
        b_mantissa in any::<u64>(),
        delta in 0u32..=300,
        p_low in 107u32..=200,
    ) {
        let p_high = p_low + 50;
        let a = bigfloat_from_mantissa_and_exponent(a_mantissa, 0, p_high);
        let b = bigfloat_from_mantissa_and_exponent(b_mantissa, -(delta as i64), p_high);

        let result_high = a.add(&b, RTE);
        let direct = a.with_precision_rounded(p_low, RTE).add(
            &b.with_precision_rounded(p_low, RTE), RTE);
        let via_high = result_high.with_precision_rounded(p_low, RTE);

        prop_assert_eq!(direct, via_high);
    }
}
```

The `delta in 0u32..=300` parameter forces the proptest to test small / medium / large alignment gaps systematically. This catches off-by-one errors in the alignment-shift logic specifically.

---

## 8. mpmath cross-validation generators

Distinct from cross-precision consistency, the mpmath cross-validation harness asserts:

```
∀ inputs, ∀ p, ∀ rnd: tambear_op(inputs, p, rnd).bit_pattern == mpmath_op(inputs, p, rnd).bit_pattern
```

The input distribution here is broader than cross-precision consistency — we want to test against mpmath at every regime, not just the round-bit boundary. **Strategy combination**:

1. **Mathematical-constants regime**: π, e, √2, log(2), Euler's γ, sin(1), cos(1), exp(1), at p ∈ {107, 200, 500, 1024}. Bit-exact compare with mpmath's `mp.pi`, `mp.e`, etc. Catches the "fundamental constants must match the literature" class of bugs.

2. **Adversarial-edge regime**: the round-bit class generators from §2-§5 above, at multiple (p_low, p_high) pairs. Catches round-bit logic bugs.

3. **Random-uniform regime**: just `(a, b)` sampled uniformly in `[-10^k, 10^k]` for `k ∈ {-100, -10, 0, 10, 100}`, at `p ∈ {107, 200, 500, 1024}`. Catches "common case" bugs that don't cluster at boundaries.

4. **f64-fast-path regime**: inputs that would have gone through the f64 fast path (per `f64_path_eligible`). Asserts that the multi-limb path AGREES with the f64 path bit-exactly when both apply. This is a strong consistency check.

---

## 9. Provenance + handoff

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish`.
- Substrate: bz-impl-reference.md §6 (the harness shape), oracle-validation.md §1.2 (verification-tier discipline), BZ §3.1.6 (round-bit + sticky theory).
- Cross-checked: round-bit class signatures (C1-C4) match BZ §3.1.6 case analysis. Generator strategies for mul / div / sqrt are work-backward-from-result; the same shape works for all three with different bit-pattern arithmetic.
- This is a generator design doc for scientist + adversarial. The actual proptest implementation lands in `crates/tambear/tests/big_float_arith_invariants.rs` (extending the existing test file per the briefing) — pathmaker may want to land the harness alongside the multi-limb arithmetic so the consistency check fires against the new code as it ships.
- Open question for scientist: is `proptest` the right framework for this, or do we want `quickcheck` for the modular-inversion-based generators? My recommendation: `proptest` — it has better shrinking for bit-pattern bugs, and the existing test file already uses it.

**Implementation note for whoever picks this up**: §2's modular-inverse generator is the cleanest for mul (always succeeds, uniform distribution within class). §3's rejection-sampling alternative is simpler but slower. The first proptest pass can use rejection sampling; if acceptance rate becomes a CI cost issue, switch to modular inverse. Both are correct; one is faster.
