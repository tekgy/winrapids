# Atoms / Exprs / Ops Gaps — What The Trig Family Structurally Demands

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-13
**Status**: Deliverable for TRIG-10. Flows from `first-principles.md`.

**Scope discipline**: This doc proposes new primitives (Layer 2) and new atom-slot values (expr / op / addressing variants at Layer 1) that the trig family **structurally requires** — not "would be nice." Every entry cites which irreducible truth from first-principles.md forces it. If a proposal cannot be traced back to T1–T12, it gets cut from this doc before shipping.

---

## Definitional reminder

| Layer | What it is |
|---|---|
| L1 atoms | `accumulate(grouping, expr, op)`, `gather(addressing)` |
| L1 slot values | Groupings (All, Windowed, ...), Ops (Add, Max, WelfordMerge, ...), Exprs (Identity, Square, ...), Addressings (ByIndex, Shuffle, ...) |
| L2 primitives | IEEE 754 hardware ops; compensated arithmetic foundations |
| L3 recipes | Everything else — named compositions |

A proposal at L2 is a new primitive. A proposal at L1-slot-values is a new slot value for an existing atom — these live as recipes with a tag flagging their role. I separate these below.

---

## L2 primitives (hardware-level or compensated) — proposed

### P1 — `ldexp_f64(x: f64, k: i32) -> f64`

Computes `x · 2^k` via bit manipulation on the exponent field. Branch-free for `|k| < 1023`; a slow path handles subnormal/overflow.

**Forced by**: T6 (kernel arithmetic choice) and the architecture doc's own Example 1 (exp recipe uses ldexp to reconstruct `2^k · exp(r)`). The architecture doc explicitly flags ldexp as "a primitive we need to add." Trig needs it for Payne-Hanek reduction and for any CORDIC-style scaling.

**Status**: partially implemented in `primitives/hardware/scale.rs` as a helper; should be promoted to a first-class primitive and renamed to match the libm convention (`ldexp` or `scalbn` — fdlibm calls it `scalbn`).

**Lowering semantics**: strict = hardware bit-fiddle; compensated = hardware bit-fiddle (ldexp is exact — it's a scale by a power of two, which is exact in IEEE 754 unless subnormal range is crossed); correctly_rounded = same.

### P2 — `frem_pio2_small(x: f64) -> (i32, f64, f64)` — Cody-Waite reduction as primitive

Three-round Cody-Waite reduction using the PIO2_1/2/3 + tails constants. Valid for `|x| < 2^20 · π/2`. Returns quadrant mod 4 plus a double-double residual in `[-π/4, π/4]`.

**Forced by**: T5 (reduction is a geometric operation that must be exact) + T12 (reduction is the shared primitive). The current sin.rs has this inline; elevating it to a primitive makes it:
- independently testable with its own oracle (`mpmath.frac(x / (π/2))`)
- shareable via TamSession (every forward trig recipe hits the same cache line)
- consumable by sincos, tan, cot, sincospi (via a scale), and future CORDIC-based kernels for comparison

**Status**: not currently a primitive; inline in sin.rs as `reduce_cody_waite`. Proposed migration: move to `primitives/compensated/rem_pio2_small.rs`.

**Open question for math-researcher**: does this belong at L2 primitive or L3 recipe? It's a composed error-free transformation, which is the same tier as `two_product_fma`. I lean primitive; math-researcher may disagree based on how deeply it decomposes.

### P3 — `frem_pio2_large(x: f64) -> (i32, f64, f64)` — Payne-Hanek reduction as primitive

Payne-Hanek for `|x| ≥ 2^20 · π/2`. Uses the 1200-bit 2/π table. Returns the same signature as P2.

**Forced by**: T5 — reduction must remain exact at all finite magnitudes, and Cody-Waite fails above `2^20 · π/2`. Without Payne-Hanek, any `sin(x)` with very large x returns garbage. This is the same reasoning that forces Payne-Hanek into fdlibm.

**Status**: currently inline in sin.rs as `reduce_payne_hanek`. Same migration target as P2.

### P4 — `cis_unit(θ) -> (f64, f64)` — explicit sincos pair primitive (OPTIONAL, recommend L3)

Forward trig fused pair. Name suggests complex-exponential-on-unit-circle semantics.

**Forced by**: T3 (the pair is the object). Making this a primitive would make the other forward functions trivially derived. However, `cis_unit` internally calls P2 or P3 plus the kernel — so it's a composition, not a terminal operation. Belongs at L3.

**Verdict**: keep at L3 as `sincos_kernel` feeding `sincos` view. Mentioned here to flag that the temptation to make it L2 exists but fails the primitive stopping-rule.

### P5 — `fma_residual(a, b, c) -> (f64, f64)` — already in the L2 list

Architecture doc already lists this at Tier 2 compensated foundations. Trig needs it for compensated polynomial evaluation at correctly_rounded lowering. No new proposal; just flagging that trig is a primary customer.

---

## L1 slot values (exprs, ops, addressings) — proposed

These are Layer 1 slot-value recipes. They plug into the `accumulate` / `gather` atoms. Each lives as a tagged recipe in the flat catalog.

### S1 — `expr: SinCosKernel` — evaluates the polynomial kernel pointwise

The sincos_kernel is a **pointwise operation** on a reduced phase column — perfect fit for the `expr` slot of `accumulate(grouping=Pointwise, expr=SinCosKernel, op=Concat)` (i.e., `map`). This is how column-wise `sin(col)` lifts: accumulate with pointwise grouping and the sincos expr.

**Forced by**: T9 (column-first mathematics). Without this expr, we hand-write a loop; with it, `sin(col)` becomes `accumulate(Pointwise, SinCosKernel, Concat, reduced_col).first()` and the column-lifting is implicit.

**Status**: not currently a registered expr; the pointwise map pattern may already exist by another name. Naturalist should check.

### S2 — `expr: Rem2PiOver4` — reduces pointwise to the fundamental domain

Pointwise reduction. `accumulate(Pointwise, Rem2PiOver4, Concat, col)` emits a column of `(q, r_hi, r_lo)` tuples.

**Forced by**: T5, T9. Currently we'd do this via a scalar function called in a loop; making it an expr lifts it cleanly.

**Sharing concern**: the tuple output `(q, r_hi, r_lo)` isn't currently an accepted expr return type — tambear's accumulate expects scalar or scalar-pair outputs in most ops. This is an **expr-return-type gap** worth flagging separately (see S7 below).

### S3 — `op: ComplexMul` — for cis-based composition (future R7 path)

`(c1 + is1) · (c2 + is2) = (c1·c2 - s1·s2) + i(c1·s2 + s1·c2)`. Binary op on `(f64, f64)` pairs. Would be needed if we implement R7 (complex-exponential-as-primitive) or if we need to compose rotations.

**Forced by**: T1 (S¹ is a one-parameter group under multiplication; group composition is complex-multiplication in cis parameterization). Not blocking for TRIG-13/14 but necessary for any rotation-group or quaternion work.

**Verdict**: defer until we commit to R7/R10. Log as a future gap.

### S4 — `addressing: QuadrantShuffle` — applies the sincos quadrant fixup as a gather

Fixing up sign/swap based on quadrant q ∈ {0,1,2,3} looks like: permute the pair `(c, s)`, optionally negate one/both. That's a gather from a 4-way branch. Expressing it as `gather(QuadrantShuffle(q), (c, s))` could factor out the branching.

**Forced by**: T6 (kernel has quadrant fixup), T11 (reduction/kernel pair). This is marginal — the fixup is trivial enough to inline. I include it because if tambear has a pattern of expressing quadrant/phase fixups this way (e.g., for FFT butterflies, for rotations), it rhymes.

**Verdict**: maybe. Naturalist should check whether this pattern exists elsewhere; if not, inline the fixup and skip the slot value.

### S5 — `expr: PolynomialHornerOdd` and `expr: PolynomialHornerEven` — the sin/cos polynomial shapes

The sin kernel is `r · P(r²)` with P of degree 5; the cos kernel is `1 - r²/2 + r⁴ · Q(r²)` with Q of degree 5. Both have the structure "polynomial in r² evaluated at a modified Horner form." If we generalize to `expr: PolynomialHornerEven(coefs)` and `expr: PolynomialHornerOdd(coefs)`, trig/trig-adjacent kernels (erf, erfc, some Bessels) can share the evaluator.

**Forced by**: T6 (kernel algorithm); compositional-recipe principle from CLAUDE.md.

**Status**: `compensated_horner` exists as a primitive already. The even/odd variants would be trivial specializations. Recommend: add as recipe-level convenience atop the existing `compensated_horner`, not as new primitives.

### S6 — `addressing: Payne_Hanek_Table_Window(e0)` — gathers the relevant slice of 2/π

Payne-Hanek only needs a small window of the 1200-bit 2/π table, centered on `e0` (the unbiased exponent of `|x|`). Expressing this as a gather addressing:

```
gather(Payne_Hanek_Table_Window(e0), INV_PI_TABLE) -> &[f64]  // 4-6 words
```

makes the windowed-lookup structure explicit. Today it's inline arithmetic on table indices.

**Forced by**: T5 (reduction is exact). Marginal — the existing inline index math works. But if the team adopts an "explicit gather" style for all windowed lookups (which they should, per T9/column-first), this is a natural place.

**Verdict**: nice-to-have; defer.

### S7 — `accumulate` return-type gap: tuple-valued outputs

Several proposed exprs (S1, S2) want to return pairs or triples: `(c, s)`, `(q, r_hi, r_lo)`. If tambear's accumulate doesn't currently support tuple-valued exprs on the Pointwise grouping, that's a real gap.

**Forced by**: T3 (sincos returns a pair), T5 (reduction returns a triple). Either accumulate grows tuple support, or recipes keep hand-rolling loops for these cases. Accumulate should grow tuple support — this unblocks not just trig but anywhere a pointwise operation returns multiple values (Givens rotations, `frexp`, `modf`, `sincos`, complex arithmetic, quaternion ops).

**Status**: check against Kingdom A convention — most accumulate semirings assume scalar-valued exprs. This is a potentially load-bearing gap. Flag for navigator.

---

## IntermediateTag family proposals

Per the sharing contract in the architecture doc, intermediates register under named tags. Trig needs:

### Tag family: `TrigReduce`

```
TrigReduce::RadiansPio2(x_key)      -> (q, r_hi, r_lo)  in radians, |r| <= π/4
TrigReduce::PiScaledHalf(x_key)      -> (q, r_hi, r_lo)  for sinpi/cospi, r ∈ [-0.5, 0.5]
TrigReduce::Degrees90(x_key)         -> (q, r_hi, r_lo)  in degrees, |r| <= 45
TrigReduce::Gradians100(x_key)       -> (q, r_hi, r_lo)  in gradians, |r| <= 50
TrigReduce::TurnsQuarter(x_key)      -> (q, r_hi, r_lo)  in turns, |r| <= 0.25
```

**Why a family**: per the sharing-compatibility rule in CLAUDE.md principle 3, intermediates are shared only when assumptions match. A degree-reduced residual is **not** compatible with a radian-kernel consumer. The tag variant explicitly encodes the unit assumption. A caller requesting `sin(col_degrees)` writes `TrigReduce::Degrees90`; the kernel knows to apply `(π/180)` inside the kernel-phase fixup, on the small residual.

**Why x_key**: content-addressing on the input column. Two callers asking "sin of this same column" reuse the intermediate.

### Tag family: `TrigForward`

```
TrigForward::SinCos(x_key, lowering) -> (c, s)
TrigForward::Tan(x_key, lowering)    -> f64
TrigForward::SinCosHyperbolic(x_key, lowering) -> (ch, sh)
```

Downstream of reduction. If a user calls `sin(x)` and then `tan(x)`, both should land on the shared `TrigForward::SinCos(x, …)` — tan is `s/c`, pulling from the same cache.

### Tag family: `TrigInverse`

```
TrigInverse::Atan2(y_key, x_key, lowering) -> f64
TrigInverse::Asin(x_key, lowering)  -> f64   // if we ever bypass atan2 for precision
```

---

## Answers to the prompt's specific questions

The team lead asked four concrete questions. Here are my answers, with trace back to first-principles.

### Q1: "Does `ModPi` belong as an atom? If so, what does it decompose to?"

**Answer**: No. `ModPi` is a **recipe**, not an atom. Reason: atoms are the two orchestration operations (`accumulate`, `gather`). `ModPi` is a specific mathematical operation — it's a slot value (an `expr`) that plugs into `accumulate`'s expr slot when lifting to columns, and it's a recipe with its own internal decomposition (Cody-Waite + Payne-Hanek dispatch).

**Decomposes to**: P1 (ldexp), P2 (Cody-Waite primitive — if we promote it), P3 (Payne-Hanek primitive — if we promote it), plus arithmetic primitives (fmul, fsub, fmadd).

**Trace**: T5 (reduction is a geometric recipe) + Layer 1/2/3 distinction from the architecture doc.

### Q2: "Is there a `QuadrantFixup` atom?"

**Answer**: No, not as an atom. `QuadrantFixup` is either:
- an inline branch in the sincos kernel recipe (current fdlibm approach), OR
- a `gather` addressing pattern `QuadrantShuffle(q)` (slot-value approach from S4 above).

The second form elevates it to a named slot value, which is the right level — but still a recipe, not an atom. Atoms are the two orchestration verbs; everything specific is below them.

**Trace**: T6 (kernel owns the fixup), layer distinction.

### Q3: "Do we need a multi-output scatter for `sincos`?"

**Answer**: Yes, this is gap S7 above — accumulate with tuple-valued exprs. This is load-bearing beyond trig: any pointwise operation returning a pair (Givens rotations, modf, frexp, complex multiplication, quaternion ops) wants this. Current tambear accumulate returns a scalar per group, which forces sincos to either return `(f64, f64)` as an opaque f128-layout trick or to be expressed as two separate accumulates over the same reduced input — the latter wastes half the kernel work.

**Recommendation**: generalize accumulate's expr return type to support small tuples. Start with pairs (rank 2); extend to triples for the reduction output if the tuple-valued expr machinery permits.

**Trace**: T3 (sincos pair is the object), T5 (reduction output is a triple), T9 (column-first).

### Q4: "Does `TrigSharedIntermediate` demand a new IntermediateTag family?"

**Answer**: Yes. See the `TrigReduce`, `TrigForward`, `TrigInverse` families above. Three families, five reduction-variant tags, unit assumptions encoded in the variant. This matches CLAUDE.md principle 3 on sharing compatibility: the intermediate is only reusable if the downstream kernel's unit-assumption matches the upstream reduction's unit.

**Trace**: T5 (reduction is the shared intermediate), T11 (reduction-kernel compatibility pairing), principle 3.

---

## Prioritization

If pathmaker wants to land this in the minimum viable slice that unblocks TRIG-13/14/15/16, the priority order is:

1. **P1 (ldexp) as a first-class primitive** — already partially there; finish it. Blocks every rebuild that uses scaling.
2. **The TrigReduce tag family** — one afternoon of writing the enum + registration. Unblocks the sharing contract for every forward trig call.
3. **P2, P3 promoted to primitives** — pulls the inline reduction code out of sin.rs, makes it testable, makes cos/tan/sincos cleanly share.
4. **S1 (SinCosKernel expr) + S7 (tuple-valued accumulate)** — this is the real structural change. If accumulate can't return pairs, sin and cos are going to be two separate passes over the same reduced input, wasting half the kernel cost on columns. This might be the single highest-leverage change in the whole trig expedition.
5. **S5 (PolynomialHornerOdd/Even recipes)** — nice factorization, reuse across erf and Bessel later.
6. **S4, S6, S3** — defer; marginal.

**Total new work for minimum viable landing**:
- Promote ldexp to primitive (small, mostly plumbing).
- Promote Cody-Waite and Payne-Hanek to primitives (move existing code).
- Add TrigReduce tag family and wire sharing (small, one file).
- Decide the tuple-valued accumulate question (medium, potentially touches atom surface).
- Write sincos_kernel as a single recipe atop the primitives (small once primitives are in place).

Then every forward trig function (TRIG-13) is a 1-5 line view. Inverse trig (TRIG-14) is composed on atan2 which is its own recipe + kernel. Hyperbolic (TRIG-15) is composed on exp/log. Pi-scaled (TRIG-16) is TrigReduce::PiScaledHalf + same sincos_kernel.

---

## What this doc does not cover

- **Notation.** See `notation.md` for the three notation styles per function (shared with math-researcher).
- **HOW the kernel is evaluated.** That's math-researcher's territory — polynomial vs CORDIC vs table, Remez fitting, minimax coefficients.
- **Specific lowering strategies per primitive.** Partially in first-principles.md Phase 8; full treatment is TRIG-11 (compilation differences per precision strategy).
- **The campsite graveyard.** If team rejects this reconstruction, this doc becomes a reference artifact, not a blocker.
