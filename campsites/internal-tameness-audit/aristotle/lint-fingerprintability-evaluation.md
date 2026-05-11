# Lint candidate fingerprintability evaluation — Phase C

**Author**: aristotle, tambear-sweep35 / internal-tameness-audit lane
**Date**: 2026-05-10
**Lane**: parallel to Sweep 35; task #11
**Pair**: adversarial (driving Phases A+B — per-function audit of big_float/, jit/, lattice/)
**Status**: First-principles evaluation; pending adversarial's Phase A+B site enumeration to cross-validate.

**Method**: Apply F13 Open Question #6 (fingerprintability as meta-precondition) to each of the five lint candidates from `internal-tameness-contracts.md` § "Tooling opportunity". For each candidate, evaluate:

1. **What is the target precondition?** (sharp or vague?)
2. **What's the syntactic anchor?** (can the lint find sites mechanically, or does it need semantic understanding?)
3. **What's the false-positive rate?** (call sites where the unchecked arithmetic is genuinely safe by other invariants — and how can the lint distinguish?)
4. **What's the false-negative rate?** (sites the lint would miss because they don't match the syntactic pattern?)
5. **Cost-benefit verdict**: implement, document-as-checklist, or skip?
6. **Graduation path**: if implemented, when would it graduate from lint to type-system enforcement?

---

## Candidate 1 — `i64-arithmetic-without-saturation`

**Target precondition**: every `+`, `-`, `*` on `i64` values inside a numerical operation must use saturating arithmetic (or have an explicit overflow check) when operating on exponent-like or counter-like values that can reach `i64::MAX`/`i64::MIN`.

**Sharpness**: **HIGH (sharp)**. The precondition is empirically convergent — pathmaker (syntactic-grep) and math-researcher (attack-side root-cause) found identical-seven sites in F13.C convergence-evidence. The boundary is well-defined: `i64::MAX` and `i64::MIN`. The antibody is mechanical: replace `+` with `saturating_add`, replace `-` with `saturating_sub`.

**Syntactic anchor**: `\.exponent\s*[+\-*]=?` or `exp_diff|exp_shift|exp_at_lsb|new_exp|result_exp\s*[+\-]`. **Pure syntactic anchor** — no semantic understanding required. The lint can find all candidate sites by pattern.

**False-positive rate**: **MODERATE**. Many `i64 + i64` operations are genuinely safe because the operands are bounded by other invariants. Examples that should NOT be flagged:
- Loop counter increments where the loop is bounded by a constant
- Index arithmetic on `&[T]` where the length is well below `i64::MAX`
- Arithmetic on values that came from `to_f64().to_bits() & MANTISSA_MASK` (bounded to 53 bits)

Discriminator: **does the operation appear inside a function that operates on `BigFloat`, `PrecisionContext`, or `Exponent`-typed values?** If yes, presume sensitive. If no, presume safe. This is a coarse filter; a sharper version requires per-function annotation (e.g., `#[tambear::precision_sensitive]`).

**False-negative rate**: **LOW** for the F13.C-shaped class. The seven sites converge by independent methodologies; the syntactic pattern catches them all. The only way to miss a site is to bypass the exponent-naming convention (e.g., a local `let e = bf.exponent + delta;` followed by use). A stricter lint that flags all `i64 + i64` inside `precision_sensitive` functions catches these.

**Cost-benefit verdict**: **IMPLEMENT** as a clippy-style lint, with `precision_sensitive` function annotation as the scope discriminator. The pattern is the highest-frequency BZ bug class (5 of the 12 fixes); the lint pays for itself in adversarial-test-cycle prevention.

**Graduation path**: lint → `Exponent` newtype with saturating operator overloads. The graduation conditions per F13.C are: (a) site count exceeds working-memory limit, (b) a refactor lands that misses one site, or (c) cross-module exponent arithmetic appears. The current N=7 is below working-memory limit; the newtype is a future graduation, not Sweep 35 work.

**Fingerprintability score**: **9/10**. Sharp precondition, syntactic anchor, mechanical antibody, low false-negative. Knock one point for the false-positive discriminator needing function-level scope.

---

## Candidate 2 — `mantissa-rounding-without-carry-bump-check`

**Target precondition**: any `unbiased + 1` pattern in a rounding-carry context (where the +1 represents carry-out from mantissa rounding) must use saturating arithmetic. This is attack25's specific shape: `unbiased + 1` overflows when `unbiased = i64::MAX`.

**Sharpness**: **HIGH for the exact attack25 shape; MODERATE for the general class**. The exact pattern `unbiased + 1` is one of three carry-bump variants:
- Mantissa carry-out: `unbiased + 1` when round-up causes exponent increment
- Exponent normalization: `exp + (n_leading_zeros) - shift_amount`
- Sticky-bit carry: round-bit + sticky-bit chain producing a virtual mantissa-msb that crosses a boundary

The first is fingerprintable. The other two are *related* but have different syntactic shapes; a lint targeted at the first won't catch them automatically.

**Syntactic anchor**: `\bunbiased\s*\+\s*1\b` (literal). Or more generally, `+\s*1\b` inside a function flagged `#[tambear::rounding_step]`. **Pure syntactic anchor for the narrow case**; broader version needs the function annotation.

**False-positive rate**: **HIGH for general `+ 1`** (loop iteration patterns, index increment, etc.). **LOW for `unbiased + 1` literal pattern** (the name is conventional in BigFloat rounding code).

**False-negative rate**: **MODERATE for the broader class.** The lint catches attack25-shape exactly; the related exponent-normalization and sticky-bit-carry shapes need their own patterns (or are absorbed into Candidate 1's broader `i64-arithmetic-without-saturation` if the scope is `#[precision_sensitive]`).

**Cost-benefit verdict**: **DOCUMENT AS CHECKLIST, NOT IMPLEMENT.** This candidate is *almost* a special case of Candidate 1. If Candidate 1 ships with `#[precision_sensitive]` scope, the mantissa-rounding sites are caught by it. A separate `mantissa-rounding-without-carry-bump-check` lint adds maintenance burden for diminishing returns. **Subsume into Candidate 1**; the audit checklist explicitly calls out the carry-bump variant during recipe-authoring code review.

**Graduation path**: subsumed by Candidate 1's `Exponent` newtype. The carry-bump operation becomes `Exponent::saturating_carry_bump(self) -> Exponent`.

**Fingerprintability score**: **6/10**. Sharp on the narrow case, blurry on the broader class. Better absorbed into Candidate 1.

---

## Candidate 3 — `limb-zero-without-kind-flip`

**Target precondition**: any path that subtracts limbs without checking the all-zero case must, on detecting all-zero limbs, flip `BigFloat.kind` to `Zero`. This is bug #8's cancellation-to-Zero antibody.

**Sharpness**: **MODERATE**. The precondition is semantic: "after limb subtraction, if all limbs are zero, the result is mathematically zero." The syntactic anchor is the *check pattern* (`limbs.iter().all(|&x| x == 0)` or `limbs == &[0; N]`) — but the lint's question is the *absence* of this check after a `sub_limbs` call.

**Syntactic anchor**: hard. Need to find `sub_limbs(...)` (or similar) followed-by no zero-check before the function returns. This is a *path-sensitive* analysis, not a pattern match. Equivalent to: "data-flow analysis showing the result reaches a `Normal`-kind constructor without going through a zero-check."

**False-positive rate**: **HIGH** for naive pattern matching. The lint cannot tell the difference between:
- A subtraction whose operands are statically known non-equal (no cancellation possible)
- A subtraction inside a `Result`-returning function where the caller handles the zero case
- A subtraction guarded by an outer check that hoisted the zero case

Without data-flow, the lint over-fires.

**False-negative rate**: **MODERATE** with naive matching. With proper data-flow, low.

**Cost-benefit verdict**: **SKIP AS LINT; DOCUMENT AS REVIEW CHECKLIST.** The cost of a path-sensitive data-flow analysis for one bug class is high; the bug class has manifested once (bug #8) and is fixable by review. The structural fix is to make `BigFloat::from_limbs(limbs)` *always* check for all-zero and flip kind — a constructor-level invariant per F13.A (single-sited antibody).

**Graduation path**: F13.A single-sited antibody at `BigFloat::from_limbs` constructor. Once that lands, the lint is unnecessary because *no path* can produce a `Normal` with zero limbs.

**Fingerprintability score**: **4/10**. Vague precondition (path-sensitive), no clean syntactic anchor. Better solved by F13.A constructor-level antibody than by lint.

---

## Candidate 4 — `special-value-dispatch-consistency`

**Target precondition**: NaN, Inf, and Zero handling across operations must be consistent. If `add`/`mul`/`sqrt` preserve NaN payload, `div` must too. The bug class: bug #10 (div dropped NaN payload).

**Sharpness**: **MODERATE-HIGH for the *consistency* claim; LOW for the *what should be consistent* claim.** "Consistent" means: same behavior under the same special-value pattern. The lint can check that — given a dispatch table for special values — every operation has the same dispatch shape. But "what is the right dispatch shape" is a per-operation question; the lint enforces consistency, not correctness.

**Syntactic anchor**: find functions matching pattern `if x.is_nan() { ... }` or `if x.kind == Kind::NaN { ... }` in operation impls. For each operation, extract the dispatch table. Compare across operations.

**False-positive rate**: **MODERATE**. Some operations *legitimately* have different special-value handling (e.g., `pow(0, 0)` is conventionally 1, not NaN; `log(0)` is -Inf, not NaN; `0/0` is NaN, not Inf). The lint must distinguish "inconsistent because of a bug" from "inconsistent because of a mathematical convention."

**False-negative rate**: **LOW** if the lint compares operations pairwise. **HIGH** if the lint requires a manually-maintained "consistency table."

**Cost-benefit verdict**: **IMPLEMENT WITH MANUAL CONSISTENCY TABLE.** The lint maintains a table like:
```
operation       NaN-payload    Inf-handling    Zero-handling
add             preserve       saturate        identity
mul             preserve       saturate        absorbing
div             preserve       saturate        zero->NaN
sqrt            preserve       Inf->Inf        Zero->Zero
```
Lint flags any operation whose impl deviates from its table row. The table is the *declared* consistency contract; the lint enforces it.

**Graduation path**: table → trait. Define `trait SpecialValueDispatch { fn handle_nan(...) -> ...; fn handle_inf(...) -> ...; fn handle_zero(...) -> ...; }`. Each operation implements the trait. Consistency is structural via trait-method-resolution, not lint-enforced.

**Fingerprintability score**: **7/10**. Sharp once the consistency table is declared; the table itself is the operationalization.

---

## Candidate 5 — `f64-fast-path-without-result-finiteness`

**Target precondition**: any f64 operation used as a seed for higher-precision computation must verify the result is finite before relying on it. This is bug #11's Newton-seed subnormal divergence shape: `1.0/b_f64 = ±Inf` when `b_f64` is subnormal.

**Sharpness**: **HIGH for the seed-use case; MODERATE for the general case.** The Newton-seed pattern is specific: compute `seed = f64_op(input)` followed by use of `seed` in iteration. The precondition is `seed.is_finite()` before the iteration loop entry. Sharp.

The general case is "any f64 operation whose result feeds higher-precision computation." More cases than Newton: compensated-arithmetic seeds, sqrt-via-Newton-on-f64-seed, log-via-f64-seed-and-correction. Each has the same shape: f64-result-as-seed.

**Syntactic anchor**: find functions matching `(let|var) seed\s*=\s*[a-z_]+_f64\(.*\)` (or similar) followed by use of `seed` in a context that doesn't check `is_finite()`. Pattern: f64-producing operation + variable assignment + iteration/use context.

**False-positive rate**: **MODERATE**. Many f64 operations have finite results by construction (e.g., bit-extracted mantissa fields, normalized intermediate values). The lint needs the `#[tambear::seed_for_iteration]` annotation on the variable or the function — without it, the lint over-fires on every f64 computation.

**False-negative rate**: **LOW** with the annotation. Without it, **HIGH** because the f64-seed pattern is too common to flag generically.

**Cost-benefit verdict**: **IMPLEMENT WITH ANNOTATION SCOPE.** The annotation `#[tambear::seed_for_iteration]` declares "this value is a seed for higher-precision iteration; verify finiteness." The lint enforces the check. Manual annotation is the discriminator.

**Graduation path**: `FiniteF64` newtype that's constructible only from `f64::is_finite()`-passing values. Iteration accepts `FiniteF64`, not `f64`. The compiler enforces the precondition.

**Fingerprintability score**: **8/10**. Sharp precondition, syntactic anchor (with annotation), graduation path to type-system enforcement.

---

## Cross-cutting findings

### Convergence of "needs annotation"

Three of five candidates (1, 2's broader form, 5) need a function/variable-level annotation (`#[precision_sensitive]`, `#[seed_for_iteration]`) to scope the lint. The annotation IS the F13.C-shaped antibody at the boundary: it declares "this site participates in the tameness contract." Without the annotation, the lint cannot distinguish in-scope sites from out-of-scope.

**Implication**: the lint family is *scoped by annotation*, not *universal*. The annotation is the substrate. **Recommend**: as part of Phase D (audit-pattern-as-recipe-authoring-workflow), define the annotations now and require them as a checkbox before merging precision-sensitive code. This is the *cultivation move* for F13 Open Question #6's "can fingerprintability be cultivated?" answer: yes, by structurally-enforced annotation declaration.

### Convergence of "subsume into one broader lint"

Candidate 2 (mantissa-rounding-without-carry-bump-check) is a special case of Candidate 1 (i64-arithmetic-without-saturation) with `#[precision_sensitive]` scope. Candidate 3 (limb-zero-without-kind-flip) is better solved by an F13.A constructor antibody than by lint. Candidate 4 (special-value-dispatch-consistency) is its own shape — orthogonal. Candidate 5 (f64-fast-path-without-result-finiteness) is its own shape — orthogonal.

**Recommend**: three lints, not five.
- **Lint 1**: `tameness::i64-arithmetic-without-saturation` (subsuming Candidate 2's carry-bump variant). Scope: `#[precision_sensitive]` functions.
- **Lint 2**: `tameness::special-value-dispatch-consistency`. Scope: operation impls; uses declared consistency table.
- **Lint 3**: `tameness::f64-seed-without-finiteness-check`. Scope: `#[seed_for_iteration]` annotations.

Candidate 3 becomes an F13.A constructor antibody at `BigFloat::from_limbs`, not a lint.

### Convergence of "graduation to type-system"

Each lint has a graduation path to type-system enforcement:
- Lint 1 → `Exponent` newtype with saturating operator overloads
- Lint 2 → `SpecialValueDispatch` trait per operation
- Lint 3 → `FiniteF64` newtype with construction-time check

The lint is the *transitional* form; the newtype/trait is the *graduated* form. Per F13.C's graduation condition (per `default-is-a-claim.md` convention-to-declaration), the lint is correct now (current scale, working-memory-fits sites); newtype/trait is the graduation target.

### F13 Open Question #6 partial answer

The F13.C convergence-evidence showed pathmaker + math-researcher converging on identical-seven sites — empirical signature of *sharp* fingerprintability. The five lint candidates above show a *gradient*:

| Candidate | Fingerprintability | Notes |
|---|---|---|
| 1: i64-arithmetic | 9/10 | Sharp; convergent across methodologies |
| 5: f64-seed | 8/10 | Sharp with annotation scope |
| 4: dispatch-consistency | 7/10 | Sharp once consistency table declared |
| 2: mantissa-carry | 6/10 | Sharp on narrow case; absorbed into 1 |
| 3: limb-zero | 4/10 | Vague; needs path-sensitive analysis; better solved structurally |

**Empirical signature of fingerprintability**: sites can be enumerated convergently by independent methodologies. Candidates 1 and 5 pass; Candidate 3 fails (the same path-sensitive analysis question is the data-flow problem, not a precondition statement).

**Cultivation moves identified**:
1. **Annotation-as-scope**: declaring `#[precision_sensitive]` cultivates fingerprintability by making the precondition site-visible.
2. **Consistency-table declaration**: writing down the expected special-value dispatch *creates* a fingerprintable precondition where there wasn't one before.
3. **Newtype-as-graduation**: a graduated lint moves from "find sites via grep" to "type-system carries the invariant"; the type IS the precondition.

These are answers to Open Question #6's sub-question "can fingerprintability be cultivated?" — yes, three moves. The empirical sign that the cultivation worked: independent methodologies converge on the same site set.

---

## Phase C deliverable summary

**Three lints to implement** (not five):
1. `tameness::i64-arithmetic-without-saturation` (scope: `#[precision_sensitive]`)
2. `tameness::special-value-dispatch-consistency` (scope: declared table)
3. `tameness::f64-seed-without-finiteness-check` (scope: `#[seed_for_iteration]`)

**One antibody to absorb at construction site**:
- `BigFloat::from_limbs` checks for all-zero limbs and flips kind to Zero (subsumes Candidate 3)

**One checklist item to document for Candidate 2**:
- Carry-bump arithmetic during mantissa rounding uses saturating add (subsumed into Lint 1 if `#[precision_sensitive]`)

**Three annotations to introduce** (the cultivation moves):
- `#[tambear::precision_sensitive]` — function-level
- `#[tambear::seed_for_iteration]` — variable/parameter-level
- `#[tambear::special_value_dispatch_table = "..."]` — operation-level

**Three graduation targets** (when lint scale fragility triggers per F13.C):
- `Exponent` newtype with saturating operator overloads
- `SpecialValueDispatch` trait
- `FiniteF64` newtype

**For adversarial's Phase A+B**: when you enumerate sites in `big_float/`, `jit/`, `lattice/`, tag each site with which of the three lints would catch it (or if it'd need the F13.A constructor antibody, or a new candidate not in this evaluation). Cross-validation: my fingerprintability scores predict the convergence rate of your site enumeration against pathmaker's hypothetical syntactic-grep. If we disagree on a site count, the precondition isn't as sharp as I scored.

---

## Phase D preview (to come)

Phase D's deliverable is "audit pattern added to recipe-authoring workflow." From the cultivation-moves finding:

The workflow addition has three parts:
1. **Annotation pass**: before merging precision-sensitive code, mark every function/variable that participates in tameness contracts.
2. **Lint pass**: the three lints fire on annotated sites; CI rejects unsaturated arithmetic, missing finiteness checks, dispatch-inconsistency.
3. **Review checklist**: for code that doesn't fit the annotation patterns (Candidate 3-shaped, requires path-sensitive analysis), the review checklist surfaces the questions: "could limbs be all-zero after this subtraction?" etc.

The discipline: every new recipe-authoring PR carries the annotation triage as part of the merge gate. Phase D's specific text-form is the addition to `R:\tambear\docs\HOW_TO_ADD_A_RECIPE.md`.

**Open question for adversarial**: are there bug-class shapes in your Phase A+B audit that DON'T fit any of the three lints + one constructor antibody + one checklist item? If yes, surface — that's new substrate for either a new lint candidate or a new cultivation move.

---

*The fingerprintability gradient is the actionable principle. Sharp preconditions get lints; vague preconditions get constructor antibodies or review checklists. The cultivation moves (annotation, declared table, newtype graduation) are how Tambear systematically increases the sharpness of its preconditions over time. F13 Open Question #6 has a partial answer.*
