# DEC-031 Invariants — First-Principles Deconstruction

**Created:** 2026-05-08
**Author:** aristotle (tambear-formalize → Sweep 31 pivot)
**Brief from team-lead:** Deconstruct two load-bearing invariants in DEC-031 (diamond commutativity + round-trip identity) plus light-pass on two adjacent ones (destination-dominated path budget + monotone-only path antibody). Outward (math/IEEE-754 reality) and inward (what tambear commits by typing these). Story-from-the-trail to navigator when crystallization happens.

**Inputs:** DEC-031 lines 3310-3478 in `R:\tambear\docs\decisions.md`; F-series methodology (recognition→operationalization→mechanical artifact, recognition vs design, defaults are claims) just closed.

---

## Setup: what's at stake at each invariant

Four invariants assigned. First two are deep (Phases 1-8 each); second two are lighter pass:

- **§6 #3 — Diamond commutativity:** ∀ f ∈ f64, ∀ p ≥ 53: both paths f64→BigFloat(p) produce bit-exact-equal results. Direct `f64→BigFloat` vs through `f64→DD→BigFloat`.
- **§6 #13 — Round-trip identity at tier boundary:** `BigFloat::from_f64(value, 53).to_f64() == value` for entire f64 range (±0, ±Inf, NaN, all subnormals from 2^-1074, tier-boundary values near f64::MIN_POSITIVE).
- **§3.2 destination-dominated path budget:** `ulp_budget_path = max(ulp_at_destination(step) for step in path)` for monotonically-coarsening paths above the subnormal floor.
- **§3.2 ATK-DEC031-4 antibody:** non-monotone paths REJECTED at construction time. The `BigFloat(200) → f64 → BigFloat(200)` example would silently under-report by ~46 orders of magnitude.

---

## Invariant 1 — Diamond Commutativity

### Phase 1 — Assumption Autopsy (what diamond commutativity silently assumes)

**A1.** "f64→BigFloat(p)" denotes a unique well-defined function. (Counter: depends on rounding-mode parameter. Three+ rounding modes → three+ "unique" functions per p. Diamond commutativity must hold *per rounding mode*, but the invariant phrasing `∀ p ≥ 53` doesn't qualify by rounding mode explicitly. §3.1 row "f64 ↔ BigFloat refine (p≥53)" classifies as Strict — no rounding mentioned, suggesting Strict means "exact embedding regardless of rounding mode chosen.")

**A2.** The two paths are: (a) direct f64→BigFloat(p) and (b) f64→DD→BigFloat(p). The diamond has only these two sides. (Counter: there's a third potential path — f64→f64-as-BigFloat(53)→BigFloat(p), which is f64 self-embedding then refining within BigFloat. That's degenerate but worth noting; might collapse to (a) by definition of what "self-embedding" means.)

**A3.** "Bit-exact-equal" means the two BigFloat values have identical mantissa, exponent, sign, and (if applicable) NaN payload. (Counter: BigFloat representation may have multiple legal encodings of the same mathematical value — e.g., trailing-zero mantissa bits, denormalized vs normalized representations. "Bit-exact" might mean "canonical-form equal" rather than "byte-equal.")

**A4.** The DD intermediate carries enough information to recover the f64 input exactly. (This is true: DD = (hi, lo) with hi=f64-input, lo=0 for any f64 input via simple injection.)

**A5.** Refining DD→BigFloat(p≥106) is Strict per §3.1; refining DD→BigFloat(p∈[53,106)) is RoundingEquivalent. Diamond commutativity holds in both regimes. (Counter at p∈[53,106): if DD→BigFloat(p) introduces 0.5-ULP at BigFloat(p), and direct f64→BigFloat(p) is Strict, the two paths can disagree by up to 0.5 ULP. That's NOT bit-exact-equal. Either the invariant is misstated, or the path through DD must collapse to the direct path when the input has only f64 information content.)

**A6.** Rounding modes commute with embedding (i.e., `embed(round(x)) = round(embed(x))`). (For embeddings that are exact this is trivially true; for non-exact reductions it can fail. Diamond commutativity hinges on this.)

**A7.** The diamond shape vs DEC-030's chain shape is the structural commitment. A chain has one path per pair-of-tiers; a diamond has two. The team chose diamond explicitly because the math forces it — there IS a direct f64→BigFloat that doesn't need to go through DD, and that path must agree with the through-DD path.

**A8.** "f64→BigFloat(p) is Strict for p≥53" because every f64 value is exactly representable in any p≥53 BigFloat. (Yes — f64 has 53 bits of mantissa precision; BigFloat(53) has at least 53. The mantissa bits map directly. Exponent range maps directly with possible extension. Sign is one bit. Specials (±0, ±Inf, NaN) need explicit handling but are representable. Subnormals: their mantissa is denormalized at f64; need normalization in BigFloat which is Strict because no precision is lost.)

**A9.** DD→BigFloat at p≥106 is Strict because DD has 106 bits of effective precision and BigFloat(p≥106) accommodates them. (True: DD is two f64s, hi and lo, with hi+lo representing a number to ~106 bits of precision when |lo| ≤ 0.5·ulp(hi). Embed each as exact f64s in BigFloat, then form their sum at BigFloat precision; for p≥106 the sum is exact because there's room.)

**A10.** The invariant "for all f64" really means "for all bit patterns in f64" — including the 2^53 NaN payloads. Diamond commutativity for NaN is non-trivial because NaN payload preservation through DD is a design choice, not free.

**A11.** "RoundingMode" is a parameter only at refinement morphisms that are NOT exact. f64→BigFloat(p≥53) is exact (Strict) → rounding mode is technically irrelevant for the embedding itself but IS relevant for the BigFloat→f64 coarsening path. So diamond commutativity might be parameterized by rounding mode for the *coarsening* paths but not the *refining* paths. Worth checking against §6 #3's exact statement.

### Phase 2 — Irreducible Truths (what survives)

**T1.** f64 has a finite domain: 2^64 bit patterns. NaN occupies 2^53-1 bit patterns; ±Inf are 2 patterns; ±0 are 2 patterns; finite normals occupy ~(2^64 - 2^53 - 4) patterns; subnormals fit within the leftover range. The cardinality is finite.

**T2.** For any f64 finite normal value v with mantissa m (53 bits), exponent e (signed 11-bit), sign s: v = (-1)^s × m × 2^(e-1023-52). This is uniquely defined.

**T3.** For any f64 subnormal value v with mantissa m (52 bits, no implicit leading 1), exponent fixed at -1022: v = (-1)^s × m × 2^(-1022-52). Subnormals form a uniform grid at 2^(-1074) spacing in [-2^(-1022), 2^(-1022)].

**T4.** A BigFloat(p) with p≥53 has at least 53 mantissa bits. The exponent range of BigFloat is wider than f64's by design (otherwise BigFloat couldn't represent values outside f64's range, defeating its purpose). Sign is one bit.

**T5.** A DoubleDouble (DD) is a pair (hi, lo) of f64 values with the canonical invariant |lo| ≤ 0.5 · ulp(hi). For a single f64 input v, the canonical DD is (v, 0). This DD has hi=v exactly, lo=0 exactly, sum hi+lo=v exactly.

**T6.** Composition of two exact-embedding morphisms is itself an exact embedding. f64→DD via (v→(v,0)) is exact; DD→BigFloat(p≥106) is exact (Strict per §3.1) → therefore f64→DD→BigFloat(p≥106) is exact.

**T7.** f64→BigFloat(p≥53) direct is exact (Strict per §3.1).

**T8.** For p∈[53,106), DD→BigFloat(p) is *NOT* a Strict refinement (it's RoundingEquivalent). However, when the DD came from f64 injection (lo=0), the DD→BigFloat(p) result equals the direct f64→BigFloat(p) result IF the rounding correctly identifies that lo=0 means "no information beyond hi." This is rounding-rule-specific.

### Phase 3 — Reconstruction from Zero (10 paths)

If we built the diamond commutativity invariant from scratch, irreducibles T1-T8 only:

1. **Bit-pattern reproduction.** Define both paths so that for any f64 input, the resulting BigFloat has identical bit-representation. This requires choosing one canonical encoding (e.g., normalized mantissa, sign-magnitude bit format, NaN-payload preservation rule) and proving both paths produce that encoding.

2. **Mathematical-value reproduction.** Define both paths so that for any f64 input, the resulting BigFloats represent the same mathematical value (modulo NaN-equality being conventional). Bit-equality follows from canonical-encoding plus value-equality.

3. **Inductive-construction proof.** Show by case analysis on the f64 input class (normal / subnormal / ±0 / ±Inf / NaN-quiet / NaN-signaling) that both paths produce the same BigFloat. Each class has a finite-rule embedding; verify each case.

4. **Universal-quantifier elimination.** "For all f64" is a finite quantifier (2^64 cases). Run both paths for every bit pattern and assert equality. Brute-force test, but tractable because finite.

5. **Composition cancellation.** Show that f64→DD→BigFloat = (f64→DD) ∘ (DD→BigFloat) and that f64→BigFloat is the composition's image. If the algebraic structure is functorial, commutativity follows from function-equality at the morphism level.

6. **Path-uniqueness proof.** Show that any two morphisms f64→BigFloat that are exact (Strict) must be equal as functions. Combined with both paths being Strict (T6 + §3.1), commutativity is forced.

7. **Round-trip-via-DD identity.** Prove (f→f64→DD→f64) = id on f64 (T6 + DD→f64 coarsening at f64 precision). Then f64→DD→BigFloat(p) = (f64→f64-embedded-in-DD)→BigFloat(p) = f64→BigFloat(p) directly. Commutativity is a consequence of f64-DD round-trip-identity at the f64 level.

8. **Concrete bit-pattern walk for all classes.** For each f64 class:
   - **Finite normal:** trivial — mantissa+exponent+sign embedded directly into BigFloat.
   - **Finite subnormal:** mantissa renormalized in BigFloat (because BigFloat has wider mantissa, the subnormal can be expressed with implicit leading 1 + extended mantissa-position). This must be done identically in both paths.
   - **±0:** sign preserved; mantissa zero; exponent represents zero.
   - **±Inf:** sign preserved; both paths must produce the same Inf-encoding in BigFloat.
   - **NaN-quiet:** sign+payload preserved or convention-stripped; both paths must agree on the convention.
   - **NaN-signaling:** sign+payload preserved; payload bits preserved exactly.

9. **Identity-element approach.** If f64→DD has identity-on-f64-injection (T5), then composing with DD→BigFloat is the same as the direct f64→BigFloat IF the latter is the unique exact embedding. Identity composition reduces to direct call.

10. **Type-level proof via Rust trait bounds.** Define the morphisms with marker traits (e.g., `Strict` trait); prove by trait-system constraints that any two `Strict` morphisms with same source/target types are equal at the value level. Compile-time enforcement, not runtime test.

### Phase 4 — Assumption vs Truth Map

| Assumption | Status | Replaced/refined by |
|---|---|---|
| A1: f64→BigFloat(p) is unique | partial | T2-T8: unique modulo rounding mode for non-exact morphisms; this case is exact so unique-period |
| A2: only two paths in the diamond | YES | T6: composition closure |
| A3: bit-exact = byte-equal | refined | T4 + canonical-encoding choice; the statement needs canonical-form qualifier |
| A4: DD carries f64 input losslessly | YES | T5 |
| A5: holds in both DD-precision regimes | TENSION at p∈[53,106) | T8: through-DD path may introduce rounding while direct path is exact |
| A6: rounding modes commute with embedding | YES for exact embeddings | trivially true at exact morphisms |
| A7: diamond shape is structural | YES | T6+T7: both paths exist and must agree |
| A8: f64 fits in BigFloat(p≥53) exactly | YES | T2-T4 |
| A9: DD fits in BigFloat(p≥106) exactly | YES | T6 |
| A10: NaN payload preservation is design choice | YES — surfaces a sub-invariant | both paths must agree on NaN-payload convention |
| A11: rounding mode irrelevant for refinement | YES | exact morphisms don't round |

The biggest tension is A5: at p∈[53,106), the through-DD path uses RoundingEquivalent (0.5 ULP at BigFloat(p)) while the direct path is Strict. If the input to DD is itself f64 (so DD's lo=0), the rounding doesn't actually fire for the value-content, but the type-system declaration says it might. **The diamond commutativity invariant requires a stronger statement at p∈[53,106): when the DD content has lo=0, the DD→BigFloat(p) reduces to the f64→BigFloat(p) direct path.**

### Phase 5 — The Aristotelian Move

The conventional move: write proptests for both paths over f64 inputs at various p values, assert byte-equality. Adequate; ad-hoc.

The Aristotelian move: **reframe diamond commutativity as a pullback identity at the type level.**

The diamond is a categorical commutative square. Top-left corner: f64. Top-right corner: BigFloat(p). Bottom-left corner: f64 (same). Bottom-right corner: DD. Top edge: direct f64→BigFloat(p). Right edge: BigFloat(p) ← BigFloat(p) (identity, trivial). Left edge: f64→DD. Bottom edge: DD→BigFloat(p).

```
f64  ────direct f64→BigFloat(p)────►  BigFloat(p)
 │                                      ▲
 │f64→DD                                 │DD→BigFloat(p)
 ▼                                      │
DD ─────────────────────────────────────┘
```

For this to commute (per category theory), the unique morphism from f64 to BigFloat(p) must equal both paths. Because BOTH paths are exact embeddings (provided p≥106 OR provided the DD content is f64-only), the morphism IS unique, and commutativity is forced.

The implementation contract: encode the diamond as a *type-level commutative square*, not as runtime equality assertions. Concretely:
- Make `f64 → BigFloat(p)` and `f64 → DD → BigFloat(p)` both implementations of the *same trait method* (e.g., `Embed<f64, BigFloat<P>> where P: AtLeast<53>`).
- Have the trait carry a *marker invariant* that all implementations of `Embed` for the same source-target pair must produce equal results.
- Use Rust's coherence rules to ensure only one canonical implementation exists; the through-DD path is a *helper that the canonical implementation may use internally* but is not itself an alternate top-level morphism.

Under this move, diamond commutativity isn't tested at runtime — it's structurally guaranteed by the type system. The "two paths" are visible at the math level (you can write either expression in user code) but the compiler reduces both to the same generated code OR ensures they call the same canonical Embed implementation.

This is the type-level home DEC-031 commits to. It's harder to build than runtime asserts but it's the recognition-with-operationalization full piece (per the F-series meta-principle): type-level commutative-square encoding is the design that operationalizes the recognition that "two paths must agree."

### Phase 6 — Recursive Challenge

What did Phase 5 silently assume?

- **B1.** The Rust type system can express commutative-square invariants. (Counter: it can't directly. Requires either coherence rules for trait-impl uniqueness OR a phantom-type proof witness OR runtime-asserted-but-compile-flagged invariants.)
- **B2.** "Same generated code" is achievable via inlining + constant folding. (For the f64-input → BigFloat(p) compile-time-known-p case, yes via const-generic specialization. For runtime-known-p, not necessarily; the through-DD path may not optimize to the direct path.)
- **B3.** DD round-trip with f64 is identity-preserving for ALL classes including NaN. (Need to verify NaN-payload preservation in DD→f64. Rust's f64 has multiple bit-distinct NaN values; DD's representation of NaN is itself a design choice.)
- **B4.** The diamond commutes for *all* p≥53, not just p∈{53, 106}. (For p∈(53,106), DD→BigFloat(p) is RoundingEquivalent; the through-DD path may introduce 0.5 ULP relative to the direct exact embedding. Diamond commutativity at runtime might fail here — UNLESS the rounding rule recognizes lo=0 as "no information" and skips. Phase 7 needs to resolve this.)

Returning to T1-T8: T7 (f64→BigFloat(p≥53) is Strict) is the load-bearing irreducible. If T7 fails for any p, the entire diamond breaks.

The recursion finds: **the deep commitment is that f64 has FINITE information content (53 mantissa bits + structure) and any BigFloat with p≥53 mantissa bits has SUFFICIENT capacity. The diamond commutativity is a corollary of f64 information-content being finite-and-bounded.**

### Phase 7 — Recursive Process (continue until stable)

Add B1-B4 to the assumption pile. Re-run Phase 5.

- B1: Rust limitation acknowledged. Compromise: encode commutative-square as proptest-witnessed-property + trait-coherence-enforced canonical implementation. Compile-time canonical-impl + runtime-asserted-equality-for-p∈[53,106) bridge. Works.
- B2: Compile-time specialization for const-known-p; runtime call for var-p. The runtime case calls the canonical Embed which dispatches to either direct or through-DD path based on p; if it picks through-DD for p<106, it must verify lo=0 and treat as identity. Works.
- B3: NaN handling spec-required. The canonical embedding must preserve f64 NaN payload bits exactly in BigFloat. This is a design choice that all implementations of Embed must respect. Add to the §6 invariant list explicitly if not already.
- B4: At p∈(53,106), the lo=0 treatment is critical. The DD→BigFloat(p) refinement, when given (hi, 0), must produce the same BigFloat(p) as f64→BigFloat(p) direct on hi. This is achievable if the implementation special-cases lo=0; otherwise rounding-direction may diverge.

After Phase 7 the picture stabilizes. The invariant is:
> **For all f64 inputs and p≥53 and rounding modes: f64→BigFloat(p) direct = f64→DD→BigFloat(p), where the DD has hi=f64-input and lo=0, AND the DD→BigFloat(p) refinement special-cases lo=0 to preserve direct-embedding semantics.**

The "lo=0 special case" is the missing operational piece. The DEC-031 §3.1 RoundingEquivalent classification at p∈[53,106) doesn't surface this; the Phase 1-8 deconstruction does. **Worth flagging back to math-researcher as a sub-clause refinement.**

### Phase 8 — Forced Rejection

Forcibly reject everything. What if diamond commutativity DOESN'T hold?

**Rejection 1 — only one path is valid.** Maybe the through-DD path is forbidden; users must use direct f64→BigFloat(p). The diamond shape is wrong; chain is right (per DEC-030).

What does the void look like? DEC-031's §3.1 RoundingEquivalent classification at DD↔BigFloat becomes irrelevant for f64-sourced inputs. The structural argument for the diamond (P8-DEC031-1) collapses; back to chain. **Counter-evidence:** users genuinely want to do DD intermediate computation (faster than BigFloat) before refining to BigFloat for storage. Forbidding through-DD destroys this use case. Diamond stays.

**Rejection 2 — commutativity holds only modulo equivalence, not bit-exact.** The two paths produce different bit representations of the same mathematical value. Bit-equality is over-strict.

What does the void look like? Comparison operators must use mathematical-value equality, not bit-equality. Hash functions can't be content-addressed by byte representation. Cache keys break. **Counter-evidence:** tambear's DEC-024 cache-key composition and DEC-029 evidence-convergence both rely on bit-equality of canonical representations. Bit-equality is load-bearing infrastructure. Mathematical-equivalence-only would force a downstream redesign of caching. Bit-exact stays.

**Rejection 3 — commutativity holds for finite normals but fails for specials.** For ±0, ±Inf, NaN, the two paths may diverge in bit-representation while still being mathematically equivalent.

What does the void look like? The invariant qualifies "for all finite-normal f64" rather than "for all f64." Tests pass on the easy 99% of cases; specials are documented exceptions. **Counter-evidence:** historical software bugs (signed-zero handling, NaN-payload preservation) come from exactly this pattern. Tambear's filter test §10 publication-grade-rigor demands all-cases coverage. Restricting to finite-normal would leak the tambear-trig commit `4d2b5e9` (sinpi sign bug) class of failures. Specials must commute too.

**Rejection 4 — commutativity is not the right invariant.** Maybe the right invariant is "round-trip identity" (f64→BigFloat(p)→f64 = id) and diamond commutativity is a derived property.

What does the void look like? §6 #3 is removed; §6 #9 (`f64→BigFloat(p≥53)→f64` identity) is the primitive invariant. Diamond commutativity follows IF you can show f64→DD→f64 = id (already in §6 #8) AND BigFloat(p)→DD→BigFloat(p) = id. **This is more economical** — round-trip identity is structurally simpler and implies diamond commutativity. But round-trip alone doesn't catch the "two implementations might disagree internally" failure mode that diamond commutativity catches directly. Both are useful; round-trip is the floor and diamond is the upper-bound check.

**Rejection 5 — bit-exact-equality of two BigFloats is undefined when BigFloat has internal representational ambiguity.** Maybe BigFloat(200) of value 1.0 has multiple bit-encodings (different mantissa-shift positions); "bit-exact" depends on canonical-form.

What does the void look like? Either BigFloat must commit to canonical-form per §6 #3 OR diamond commutativity must use canonical-form-equality. **Counter-evidence:** §3.5 declares BigFloat as a from-scratch implementation per Brent-Zimmermann. That reference (Algorithms 3.1-3.6) implies a canonical form (normalized mantissa with implicit leading 1, fixed-precision representation). Canonical form is the implicit assumption of any from-scratch implementation. Make it explicit: `§6 #3 should add "where BigFloat values are in canonical form per §3.5's Brent-Zimmermann reference."`

### What MUST exist under these rejections

- **R1** (one-path-only): rejected, counter-evidence wins. Diamond stays.
- **R2** (modulo-equivalence): rejected, bit-equality is load-bearing.
- **R3** (specials excluded): rejected, all-cases is the contract.
- **R4** (round-trip is primitive): partially accepted as complementary invariant; diamond commutativity remains as direct-disagreement detector.
- **R5** (canonical form ambiguity): refinement — invariant should explicit canonical-form requirement.

The deepest finding: **diamond commutativity is the type-level commutative-square encoding of f64-information-content-finiteness. It's irreducible because f64 is finite-information; it's enforceable because BigFloat-canonical-form is a single agreed-on representation.**

Remaining concrete work for math-researcher / pathmaker:
- Add canonical-form requirement to §6 #3 explicitly (or document in BigFloat type declaration).
- Surface the lo=0 special case in DD→BigFloat(p) refinement at p∈[53,106), as identified in Phase 7.
- Decide implementation strategy for ALL through-DD path: special-case-lo=0 OR general-rounding-rule that produces same result.

---

## Invariant 2 — Round-Trip Identity at Tier Boundary

### Phase 1 — Assumption Autopsy

The invariant: `BigFloat::from_f64(value, 53).to_f64() == value` for entire f64 range.

**A1.** "==" on f64 means bit-equality of the f64 representation, not mathematical equality. For NaN, this matters: `NaN == NaN` is `false` in IEEE-754, but `bit_eq(NaN, NaN)` requires the same NaN payload.

**A2.** `from_f64(value, 53)` produces a BigFloat with exactly 53 bits of mantissa precision, matching f64. (53 is the load-bearing parameter — the tier boundary.)

**A3.** `to_f64()` converts back. The default rounding mode applies. (The invariant doesn't qualify by rounding mode, suggesting rounding mode is irrelevant when the value already fits exactly in f64 — which it does at p=53.)

**A4.** Entire f64 range = 2^64 bit patterns. The invariant must hold for all of them.

**A5.** Subnormals require special handling because their representation doesn't have an implicit leading 1 in mantissa.

**A6.** Tier-boundary values near `f64::MIN_POSITIVE = 2^(-1022)` are mentioned specifically. These are at the boundary of normal/subnormal in f64. (Why specifically called out? Because at this boundary, the encoding format changes — implicit-leading-1 → explicit-leading-bit-zero. BigFloat must handle the encoding-class difference correctly.)

**A7.** ATK-DEC031-3 (the attack from v1) found this was a stub. v2 elevates to enforcement #13. Implies v1's BigFloat::from_f64 was not handling the full f64 range — likely missed subnormals or specials.

**A8.** The invariant is stated as `==` not `≈`. There is no tolerance. Pure bit-equality.

**A9.** ATK-DEC031-3 from team-lead's brief: "Why does this matter beyond the obvious 'round-trip is good'? What does the encoding need to look like for this to be trivial vs hard?" The asked-about implementation differential is real and worth deconstructing.

### Phase 2 — Irreducible Truths

**T1.** f64 has 2^64 bit patterns; the invariant must hold for all (T-prior carries).

**T2.** Round-trip identity at p=53 means: encoding `value` into BigFloat(53), then decoding back, recovers `value` byte-equal.

**T3.** This is achievable iff (a) the BigFloat(53) representation has at least the information capacity of f64 (53 mantissa bits + at least f64's exponent range + sign + special handling); AND (b) the encoding/decoding functions are mutually inverse on this domain.

**T4.** Information-capacity check: BigFloat(53) has exactly 53 mantissa bits. f64 has 53 mantissa bits in normalized form (52 stored + 1 implicit). For subnormals, f64 has 52 stored mantissa bits, no implicit 1, exponent fixed at -1022. BigFloat(53) representation of a subnormal must somehow distinguish "this came from f64-subnormal-encoding" — either by encoding the leading 1 explicitly or by extending the exponent range.

**T5.** Specials (±0, ±Inf, NaN): each must have a designated BigFloat representation that round-trips back to the original f64 bit pattern.

**T6.** NaN payload: f64 reserves bits 51-0 of the mantissa as NaN payload (when exponent is all-1s). 2^52 distinct NaN values. BigFloat must preserve all bits.

**T7.** Round-trip identity is the *consistency* condition for any bidirectional encoding scheme. It's the simplest possible correctness claim. If it fails, the encoding is broken.

### Phase 3 — Reconstruction from Zero

What does the encoding need to look like for round-trip identity to be trivial vs hard?

1. **Trivial encoding:** BigFloat(53) stores f64's 64-bit representation directly + a marker "this is f64-tier." `to_f64()` reads the 64 bits back. Round-trip = byte-copy. Easy. But this defeats the purpose: BigFloat(53) IS just f64 in disguise. There's no value above f64.

2. **Mathematical-value encoding:** BigFloat(53) stores (sign, exponent, mantissa) where mantissa is normalized. `from_f64` extracts these from f64's IEEE-754 layout. `to_f64` reverses. Round-trip-identity holds iff the encoding handles all f64 classes correctly (normals via direct mapping; subnormals via explicit-mantissa-bit-encoding with extended exponent or normalization; specials via designated BigFloat-special-representations).

3. **Brent-Zimmermann encoding (per §3.5 reference):** mantissa as fixed-bit array, exponent as machine integer, sign as one bit, precision parameter explicit. Specials (Inf, NaN) likely as separate enum variants in the BigFloat type. Round-trip-identity requires the enum-to-f64-bit-pattern mapping to be invertible.

4. **Difficulty taxonomy by f64 class:**
   - **Finite normal:** TRIVIAL. Direct mapping of (s, e, m) bits.
   - **Finite subnormal:** MEDIUM. Two encoding choices: (a) re-normalize in BigFloat (extends exponent below f64::MIN_EXP); (b) preserve the f64 subnormal representation with explicit zero-leading-bit mantissa. Option (a) is cleaner; option (b) is more f64-faithful. Round-trip identity must commit to one and be consistent.
   - **±0:** EASY. Designated zero-encoding with sign bit.
   - **±Inf:** EASY. Designated infinity-encoding with sign bit.
   - **NaN-quiet:** HARD. NaN has 2^52 distinct payloads. Round-trip requires preserving every payload bit.
   - **NaN-signaling:** HARDER. Same payload requirement plus signaling-bit preservation.

5. **Why NaN is the hardest case:** the f64 NaN representation has 52 mantissa bits + 1 quiet/signaling bit + the exponent-all-1s + the sign bit. That's 53 bits of "metadata" packed into the mantissa-and-sign. BigFloat(53) has 53 mantissa bits — barely enough capacity, but only if the encoding is exactly aligned. If BigFloat normalizes NaN to a canonical form (e.g., quiet NaN with payload 0), round-trip fails for any non-canonical NaN input. **Round-trip identity for NaN is what flags whether the BigFloat NaN representation is faithful or canonical-only.**

### Phase 5 — The Aristotelian Move

The conventional move: write encoding/decoding functions, test on representative samples, fix bugs found.

The Aristotelian move: **make round-trip identity a law of the type system, not a runtime test.**

Specifically: design the BigFloat::from_f64 and BigFloat::to_f64 such that they're defined in terms of *the same canonical representation*, with the inverse-relation enforced by construction. Concretely:

- Define `BigFloatAt53` as a newtype around f64-bit-pattern + canonical-representation-marker. `from_f64(v) -> BigFloatAt53` is bit-copy. `to_f64(self) -> f64` is bit-copy. Round-trip is byte-equality by definition.
- For p>53, BigFloat is a richer type that includes BigFloatAt53 as a sub-case via the `Embed` trait. The round-trip identity at the tier boundary is a special case of the general invariant.

This dissolves the question: round-trip-identity-at-p=53 isn't a test; it's a definition. The work moves into ensuring the canonical-representation marker correctly identifies all f64 classes.

The deeper insight: **why does this matter beyond "round-trip is good"?** Because round-trip-identity at the tier boundary is the *correctness witness for the BigFloat type's claim to be a proper extension of f64*. If round-trip fails, BigFloat is NOT a superset of f64; it's a different number system that overlaps with f64 lossily. Tambear cannot then claim "BigFloat at p=53 = f64" — which is the load-bearing claim that justifies the diamond at p=53.

### Phase 8 — Forced Rejection

Forcibly reject the invariant. What if round-trip identity DOESN'T hold at p=53?

**Rejection 1 — round-trip identity holds modulo NaN canonicalization.** All NaNs map to one canonical NaN; round-trip preserves this. Specific NaN payloads are lost.

What does the void look like? f64 NaN payload bits are gone after one round-trip. Software that uses NaN payloads for diagnostic purposes (which IS a real practice — e.g., propagating "this came from invalid input X" via payload bits) is broken when going through BigFloat. Tambear can't claim full f64 fidelity. **Counter-evidence:** the contract item 10 publication-grade rigor implies all-cases. NaN payload preservation matters in practice. Round-trip should preserve, not canonicalize.

**Rejection 2 — round-trip identity holds for normal values but fails for subnormals.** Subnormals are renormalized into BigFloat's wider-exponent form; on round-trip back to f64 they're rendered as either flushed-to-zero (incorrect) or as renormalized-back-to-subnormal (correct).

What does the void look like? Subnormals lose their bit-pattern through BigFloat. Engineering applications that depend on subnormal arithmetic (numerical analysis, signal processing) are broken. **Counter-evidence:** ATK-DEC031-3 specifically called out "all subnormals from 2^-1074, tier-boundary values near MIN_POSITIVE." The attack found this WAS broken in v1. v2 must fix. Subnormal preservation is non-negotiable.

**Rejection 3 — round-trip identity is a contract for the user, not a property of the type.** The invariant is documented but not enforced. Users who care should round-trip-test their own values; tambear doesn't guarantee.

What does the void look like? The diamond commutativity proofs depend on round-trip identity. Without round-trip identity as a type-system property, diamond commutativity collapses to "modulo round-trip-fidelity-of-the-implementation" — which is meaningless. **Counter-evidence:** §6 #13 enforces this; ATK-DEC031-3 elevated it from stub. Tambear committed to it. Cannot revoke.

**Rejection 4 — round-trip identity is overspecified.** Maybe `BigFloat::from_f64(value, 53).to_f64() ≈ value` (within 1 ULP) is sufficient for engineering purposes; bit-equality is unnecessary strictness.

What does the void look like? At p=53 (the tier boundary where BigFloat is supposed to BE f64), tolerated 1-ULP slop means BigFloat at p=53 is NOT f64. The whole structure of "tier boundary" collapses. **Counter-evidence:** the tier boundary is the irreducible commitment that BigFloat extends f64. Without bit-equality at p=53, the precision lattice has gaps where f64 should fit. Bit-equality stays.

### What's load-bearing

Round-trip identity at the tier boundary is the *correctness witness* for the BigFloat-as-f64-extension claim. Without it:
- Diamond commutativity becomes "approximate" instead of bit-exact.
- BigFloat at p=53 is not equivalent to f64; it's a different number system.
- The whole DEC-031 lattice has a load-bearing gap.

This is why ATK-DEC031-3 elevation matters. The attack found that v1 had this as a stub; v2 makes it enforcement #13. Without #13, diamond commutativity (#3) is unenforceable — because the through-DD path for f64-sourced input depends on f64→DD round-trip identity (already #8) AND f64→BigFloat(p=53)→f64 round-trip identity (#13). The two together close the circle.

---

## Invariant 3 (light pass) — Destination-Dominated Path Budget

`ulp_budget_path = max(ulp_at_destination(step) for step in path)` for monotonically-coarsening paths above the subnormal floor.

### Why this rule?

In a monotonically-coarsening path (each step has equal-or-fewer mantissa bits than the previous), the precision floor at each step is bounded by the destination's ULP. The maximum across steps is the bottleneck. Errors at intermediate steps ARE bounded by their own ULP, but when projected to the terminal precision, they contract because the terminal precision is lower (fewer bits = more contraction headroom).

Specifically: a 0.5-ULP error at BigFloat(200), when coarsened to f64, becomes 0.5-ULP at f64 because the f64-ULP is much larger than the BigFloat(200)-ULP. The error doesn't get *worse* in the coarsening direction.

### Why monotonically-coarsening?

This is the load-bearing scope. For monotonically-refining paths (each step has equal-or-more mantissa bits), the rule REVERSES: a 0.5-ULP error at the source coarse precision becomes ~2^(p_dest - p_source) ULPs at the destination fine precision. Same absolute error is huge in finer-precision ULPs.

**The destination-dominated rule is monotonically-coarsening's structural consequence; the monotonically-refining direction needs the source-dominated rule (or the chains-E/F/G subnormal-aware bound).**

### What does it claim about the universe?

The rule is a statement about HOW PRECISION ERRORS COMPOSE under coarsening. It says: errors don't compound — they get bounded by the destination, which is the loosest bound in the path. This is true because:
- Coarsening is itself a rounding operation (RoundingEquivalent at 0.5 ULP at destination).
- The composition of bounded-rounding-operations contracts to the floor of the bounds, not the sum.
- The floor in monotonically-coarsening is the destination ULP.

This is a *structural property of monotonic-precision composition*, not a tambear-specific design choice. The DEC ratifies what the math forces.

### Why is the subnormal floor exception needed?

In the f64-subnormal regime, the mantissa is denormalized (no implicit leading 1). The ULP definition changes: it's no longer relative-to-magnitude but absolute-at-2^-1074. The destination-dominated rule, which assumes relative ULP, breaks. Per-step explicit subnormal-aware bounds replace it.

This is structural: the rule is correct in the regime where ULP ~ magnitude × precision-bit, and breaks in the regime where ULP is fixed at the smallest-representable-step.

---

## Invariant 4 (light pass) — ATK-DEC031-4 Antibody (Non-Monotone Path Rejection)

The example: `BigFloat(200) → f64 → BigFloat(200)`. Naive destination-dominated would compute 0 ULP at BigFloat(200) (terminal precision); actual error is ~2^147 ULPs (step 1's 0.5-ULP at f64 is 2^105 ULPs at BigFloat(200), plus 2^42 factor).

### What's the deeper pattern?

The path-builder rejecting non-monotone paths is the antibody to *applying-the-rule-outside-its-scope*. The rule (destination-dominated budget) has a precondition (monotonically-coarsening); the antibody (path-builder rejection) enforces the precondition at construction time so the rule is only applied where valid.

**This is a special case of a general pattern: when a rule has a scope precondition, the system that enforces the precondition is structurally part of the rule's correctness.**

Generalizing:
- Rules without scope-enforcement antibodies silently fail in their out-of-scope domains.
- Rules WITH scope-enforcement antibodies fail-loudly at the scope boundary.
- The antibody is what makes the rule trustworthy as a building block.

This generalizes to F11+F12: a recognition-claim WITHOUT the design-that-operationalizes-it is a rule without an antibody — it floats and silently fails. F12's `aliased_to`/`stubbed_pending` declarations are antibodies for the "default is a claim" rule; without them, the rule fails silently for aliased recipes.

**Same shape, different layer.** The antibody pattern is universal across precision-lattice rules AND F-series claim-discipline rules. Worth noting.

### Why is non-monotone rejection right vs e.g. non-monotone supports a different rule?

Tambear could declare: monotone-coarsening uses destination-dominated; non-monotone uses ROUND-TRIP-AWARE budget (sum of bounds along the path). That's a real alternative.

Why reject instead? Because non-monotone paths are usually *bugs*. The example `BigFloat(200) → f64 → BigFloat(200)` is naively a no-op but actually loses ~2^147 ULPs of precision. A user who writes this is almost certainly making an error — they didn't realize the round-trip through f64 destroys 147 bits of precision. Failing fast at construction time forces the user to re-think the path: either fix the intermediate to skip f64 OR explicitly accept the precision loss with a documented `force_round_trip = true` parameter.

The antibody isn't just enforcing a rule's scope; it's surfacing a class of user errors that would otherwise be silent.

This rhymes with F12: undeclared aliasing isn't just a "rule violation" — it's a class of user-confusion-because-they-think-they-got-precision-they-didn't. The antibody (F12 schema's strict default) catches the same class of error in claim-space.

---

## Cross-cutting findings

**(1) The four invariants form a coherent system.** Diamond commutativity (#3) and round-trip identity (#13) jointly close the f64↔BigFloat tier-boundary correctness loop. Destination-dominated path budget (§3.2 + #4) bounds composition error in the coarsening direction. The non-monotone-rejection antibody (#12) keeps the budget rule's scope honest. Together they make the precision lattice a *type-level provable structure*, not just a runtime-tested-thing.

**(2) The same antibody pattern appears in F-series.** Rule-with-scope-precondition needs antibody-that-enforces-precondition. Without antibody → silent failure in out-of-scope. Without antibody, the rule is the F12 "default is a claim" without `aliased_to`; the precision rule is destination-dominated without monotone-rejection. Same shape, different layer.

**(3) The deepest commitment of the diamond is f64-information-content-finiteness.** f64 has bounded information; BigFloat(p≥53) has at-least-equal capacity; therefore the embedding is exact and unique; therefore the diamond commutes. This is what tambear is committing to by typing this: the type system encodes f64-as-finite-information, BigFloat-as-extension, and embedding-as-functorial.

**(4) Round-trip identity is the witness for "BigFloat at p=53 IS f64."** Without it, BigFloat is not an extension — it's a different system that lossily overlaps. The diamond's bottom corner can't anchor to f64 without round-trip identity as the floor.

**(5) The non-monotone path antibody rhymes with the F11 recognition/design discipline at a different layer.** Both surface latent user-confusion (precision loss / aliased strategy) by failing-loud at the boundary. Both are recognition-with-operationalization (recognition: composition can mislead; operationalization: fail at construction). Both work because the antibody makes the rule trustworthy as substrate.

**(6) Inward mode finding:** by typing these invariants, tambear commits to f64-information-content + BigFloat-canonicality + embedding-functoriality + composition-monotonicity-aware-precision-tracking. The type system encodes IEEE-754 reality faithfully. Any violation (e.g., a BigFloat implementation that loses NaN payloads on round-trip) is a *type-system lie*, not just a runtime bug.

**(7) Outward mode finding:** the IEEE-754 reality that grounds these invariants is the finiteness of f64 + the canonicality of normalized-mantissa representation + the deterministic-rounding-modes spec + the special-value-class spec (±0, ±Inf, NaN-quiet, NaN-signaling). DEC-031 is internally consistent with IEEE-754; the diamond commutativity and round-trip identity are CONSEQUENCES of IEEE-754 + DEC-031's tier-boundary commitment, not inventions.

---

## What I want navigator to know

**Crystallization story-from-the-trail:**

DEC-031 is internally coherent. The four invariants I deconstructed cohere as a type-level provable structure encoding IEEE-754 reality. The two load-bearing ones (diamond commutativity + round-trip identity) jointly close the f64↔BigFloat correctness loop; the destination-dominated budget plus monotone-rejection antibody bound composition error in the coarsening direction.

**One refinement worth surfacing to math-researcher:**

Phase 7 of the diamond commutativity deconstruction surfaced a sub-clause refinement: at p∈[53,106), the through-DD path's RoundingEquivalent classification permits 0.5 ULP at BigFloat(p), but the diamond invariant requires bit-exact-equal results. The refinement: **when the DD content has lo=0 (which it does whenever DD was constructed from f64), the DD→BigFloat(p) refinement must special-case lo=0 to preserve direct-embedding semantics.** Without this special case, diamond commutativity breaks at p∈(53,106). DEC-031 §3.1 RoundingEquivalent classification doesn't currently surface this. Math-researcher's BigFloat impl design should make the lo=0 case explicit.

**One pattern worth holding (cross-domain rhyme):**

The non-monotone-path-rejection antibody is the same pattern as F12's strict-default-undeclared-fails-lint. Both are antibodies that enforce a rule's scope at construction time. Both surface latent user-confusion (precision loss / claim violation). Both make the rule trustworthy as substrate by failing-loud at the boundary. *Recognition without antibody is just a rule; recognition with antibody is enforceable substrate.* This rhymes with the F-series meta-principle (recognition→operationalization→mechanical artifact); the antibody IS the mechanical artifact for scope-precondition rules specifically.

**No new F-numbers needed.** This is Sweep 31 substrate, not F-series methodology. Filing the cross-domain rhyme as a private observation worth carrying forward.

---

## Status

Phases 1-8 walked once on each of the two load-bearing invariants. Light pass on the two adjacent ones. Cross-cutting findings consolidated.

Output landed at this campsite. Ready for math-researcher's BigFloat impl design phase. Standing by for definitional questions during impl or for navigator redirect.
