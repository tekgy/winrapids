# Branch-cut conventions for complex-transcendental recipes

**Status**: Roadmap draft, awaiting math-researcher ratification → proposed `DEC-032`.

**Drafted**: 2026-05-08, main-thread Claude (tambear-sweep31-finish team-lead lane), with Tekgy.

**Anchor**: `R:\winrapids\PLEASE_READ_from_gpu_verifier_port.md` finding 4 + naturalist's `R:\winrapids\campsites\tambear-formalize\naturalist\` thread "graph-form keeps appearing as recognition" (2026-05-08).

**Ratification path**: Math-researcher reviews the variant set + identity preservation sub-clauses + open questions, signs off async, lands as DEC-032 in `R:\tambear\docs\decisions.md` with this content as the body. Pathmaker implements at first-complex-transcendental-recipe time. Until then, **no complex-transcendental recipe enters the queue.**

---

## The problem

The mathematical literature has multiple inconsistent conventions for branch cuts of complex-transcendental functions:

- `ln(-1)` is `+iπ` under the standard principal-value convention (counterclockwise, cut along negative real axis).
- `ln(-1)` is `-iπ` under the alternative convention (clockwise).
- `sqrt(-i)` chooses different roots depending on cut placement.
- `arctan(z)` discontinuity placement differs across implementations: along the imaginary axis from `±i` outward (Mathematica), along `(-∞, -1] ∪ [1, ∞)` on the real axis (some MATLAB toolboxes).
- `arccos`, `arcsin`, `arctanh`, `complex_pow` — all have analogous choices.

Without an explicit knob, the **first** complex-transcendental recipe tambear ships will silently bake in **one** convention. Every downstream recipe that composes with it inherits that convention without seeing it. A user computing `ln(-1) + ln(-1) =? ln((-1)·(-1)) = ln(1) = 0` gets `2iπ ≠ 0` (correct under integer-k accounting) or `0` (correct under modular-2π handling), and the choice was made for them by an internal default they never saw.

**This is the worst possible antibody-failure mode**: silent wrong answers under one convention, while the test suite passes because the test author and the implementer both internalized the same default. F13 explicitly exists to prevent exactly this.

---

## Prior art — the GPU verifier port team

`PLEASE_READ_from_gpu_verifier_port.md` finding 4 (paraphrased verbatim):

> Branch-cut conventions are a real distinction, not a "details" matter. `ln(-1)` returning `+iπ` vs `-iπ` flips downstream identities silently. The verifier has explicit `--branch-aware-witness` mode for this. Tambear's complex math primitives need an analog `using(branch: ...)` runtime knob if they touch transcendentals on negative reals.

The verifier-port team — nine agents working on a substrate-search verification problem — independently surfaced the same antibody. **The convergence is the signal.** Tambear inherits both their finding and their proposed shape.

---

## The knob

`using(branch: BranchPolicy)`, where `BranchPolicy` is an enum (final variant set to be ratified):

```rust
/// Branch-cut convention for complex-transcendental recipes.
///
/// Per F13 (every rule with a scope precondition needs an antibody): every
/// complex-transcendental recipe must declare a `BranchPolicy` at
/// construction. Absence is a compile error, not a silent wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BranchPolicy {
    /// Standard principal-value convention. `ln(-1) = +iπ`, cut along the
    /// negative real axis approached from above. Counterclockwise sense.
    /// `arctan(z)` cut along `(-i∞, -i] ∪ [+i, +i∞)`. The most common
    /// convention in modern analysis textbooks (Ahlfors, Conway, Stein).
    Principal,

    /// Alternative principal-value convention. `ln(-1) = -iπ`. Clockwise
    /// sense. Used by parts of the engineering literature and by some
    /// MATLAB toolboxes for backward compatibility with FORTRAN
    /// conventions. Identities under `Principal` may flip sign.
    AntiPrincipal,

    /// Branch picked dynamically per-evaluation to minimize cancellation
    /// error in the floating-point computation. Sacrifices identity-
    /// preservation across calls for numerical stability within a single
    /// call. Required for high-precision numerics where consistency
    /// across calls is less important than per-call accuracy.
    NumericallyStable,

    /// Every branch evaluated; recipe returns `(value, witness)` tuples
    /// where `witness` records the branch chosen. For exploratory
    /// analysis where the user wants to inspect all roots / cuts.
    /// Naturalist's catalog-as-tree pattern: the relationship IS the
    /// answer; the methods are the inputs.
    Discovery,
}
```

`Principal` is the recommended default per the principal-value convention's prevalence in modern literature. **No automatic default is set** — `BranchPolicy` is a non-defaulted parameter on every complex-transcendental recipe, forcing the user (or the pipeline that composes them) to pick consciously.

---

## The F13 antibody

Per F13 (`R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md`):

> Every rule with a scope precondition needs an antibody that enforces the precondition at construction time. Without antibody → silent failure outside scope.

The scope precondition for any complex-transcendental recipe is: *the result is meaningful only relative to a branch convention.* The antibody:

**Every complex-transcendental recipe takes `BranchPolicy` as a non-defaulted parameter.** Absence at the call site = compile error. The recipe body matches on the policy and produces results consistent with it. Outputs are tagged with the branch convention used (in V columns alongside DO columns; the V column carries the `BranchPolicy::tag()` byte).

This means:
- `complex_log(z)` does not compile. `complex_log(z, BranchPolicy::Principal)` does.
- A user who hasn't picked must pick. There is no "cheap default" that lets them skip the decision.
- Cross-call consistency is observable: identical `BranchPolicy` values produce identical results bit-for-bit; differing values produce results that are honestly different.

---

## Sub-clauses

### A. Identity preservation under each policy

Under each convention, the recipe must preserve the identities the convention promises:

- **`Principal`**: `exp(ln(z)) = z` for `z ≠ 0`; `ln(z) + ln(w) = ln(z·w) + 2πi·k` for the integer `k` determined by the principal-branch winding.
- **`AntiPrincipal`**: same identities with sign-flipped `k` selection.
- **`NumericallyStable`**: identity preservation within the active ULP budget per the branch chosen at each evaluation; no integer-`k` discontinuities within a single computation, but identities across calls may differ from `Principal`.
- **`Discovery`**: returns all branches; identity preservation is the *user's* responsibility via the `witness` field.

Identity violations **under a declared policy** are bugs, not user error.

### B. Cross-recipe consistency in pipelines

If recipe A uses `BranchPolicy::Principal` and recipe B uses `BranchPolicy::AntiPrincipal`, composing them in a pipeline raises a policy-resolution ambiguity. Resolution rule:

1. If the pipeline declares `using(branch: ...)`, that wins; A and B are both invoked under the pipeline-level policy.
2. If the pipeline does *not* declare `using(branch: ...)` and A and B disagree, the pipeline fails to compile.

**Mixed-policy pipelines never silently succeed.** Either the pipeline picks (preserving consistency), or compilation refuses.

### C. Cache-key participation (Sweep 32 hook)

Per Sweep 32 (`fa49fec`, `feed_precision_context` at tag `0x1A`): the `BranchPolicy::tag()` byte should participate in the cache key for any kernel that depends on the branch convention. Reserved tag: `0x1B` for `feed_branch_policy`.

When `feed_branch_policy` is added, IR_VERSION bumps `10 → 11`. Cache invalidation is correct — kernels compiled under one convention are not interchangeable with kernels compiled under another.

### D. Tag bytes for `BranchPolicy`

Stable bytes for cache-key serialization (must not change once shipped, per DEC-031 §3.7 enforcement template):

```rust
impl BranchPolicy {
    #[inline]
    pub const fn tag(self) -> u8 {
        match self {
            BranchPolicy::Principal         => 0,
            BranchPolicy::AntiPrincipal     => 1,
            BranchPolicy::NumericallyStable => 2,
            BranchPolicy::Discovery         => 3,
        }
    }
}
```

`#[non_exhaustive]` allows future variants without re-ratifying the existing four.

### E. Discovery-mode output shape

`Discovery` policy returns *all branches* of a multi-valued function. Output type for `complex_log`:

```rust
pub struct BranchedComplex {
    pub primary: Complex<f64>,    // Principal-branch value
    pub witnesses: Vec<(BranchTag, Complex<f64>)>,  // All other branches
}
```

The naturalist's catalog-as-tree pattern surfaces here: `Discovery` is the structural-rhyme analog to `discover()` for sketches/correlations/distances. The relationship IS the answer.

---

## Roadmap

1. **Now**: this doc lands as winrapids-architecture roadmap. Math-researcher reviews open questions, ratifies variant set + sub-clauses A–E, signs off.
2. **Ratification commit (math-researcher async)**: copies ratified content to `R:\tambear\docs\decisions.md` as DEC-032; updates the open questions section with closure rationale; closes any thread loops with the verifier-port team.
3. **First complex-transcendental recipe**: pathmaker implements `BranchPolicy` enum + `feed_branch_policy(0x1B)` + IR_VERSION bump + the first recipe (likely `complex_log`). Adversarial proptests verify that all four policies produce identities the policy promises.
4. **Subsequent complex-transcendental recipes**: every one inherits the antibody — no defaulted `BranchPolicy`, mandatory at call site, tagged in cache key, V-column-recorded in outputs.
5. **Future Tier-2 amendment**: if a fifth convention surfaces (custom cuts, parametric branches, Riemann-surface unfolding), `#[non_exhaustive]` allows extension without re-ratifying.

---

## Open questions for math-researcher

These are the points the spec-author (main-thread Claude) cannot decide alone; math-researcher's literature work is the appropriate authority:

1. **Variant enumeration sufficiency.** Is `Principal | AntiPrincipal | NumericallyStable | Discovery` the right base set, or should the enum be parametric (e.g., `BranchPolicy::Custom(Vec<CutSegment>)`) to handle non-standard cuts (Riemann-surface unfoldings, custom-domain analyses) from day one?

2. **Default choice within `Principal`.** The literature converges on `ln(-1) = +iπ` for principal-branch logarithm, but `arctan` cut placement is more contested. Should `Principal` be one bundled convention (the dominant analyst-facing set), or four sub-policies (`PrincipalLog | PrincipalArctan | PrincipalSqrt | PrincipalAcos`) each independently picked?

3. **NumericallyStable ULP budget interaction with DEC-031.** When `BranchPolicy::NumericallyStable` is active and the recipe's `PrecisionContext` is `P2BigFloat { precision_bits: 1024 }`, what's the right interaction? Does NS pick branches per-call within the 1024-bit precision, or does it require a higher precision tier internally?

4. **Discovery-mode output shape across recipe families.** For `complex_log`, `Discovery` returns one primary + witnesses. For `complex_pow(z, w)`, the output may be infinite (irrational `w`). What's the principled bound — per-call user-specified `max_branches`, or fixed convention (e.g., the n principal roots for `complex_pow(z, 1/n)`)?

5. **Pipeline-level resolution semantics.** Sub-clause B says mixed-policy pipelines fail to compile if no pipeline-level `using(branch: ...)` is set. Is "fail to compile" the right strength, or should it be "warn and pick the alphabetically-first policy as default" with an opt-out? Alphabetical defaults are fragile (rename-sensitive); compile-fail is strict but enforces the antibody. Recommend: keep compile-fail.

---

## Why this matters now (priority rationale)

This doc lands **before** any complex-transcendental recipe enters the queue because the cost-of-being-wrong is asymmetric:

- **Cost of getting this knob right now**: ~one math-researcher review cycle + ~one pathmaker implementation block at first-complex-recipe time. ~2 days total work, all parallelizable.
- **Cost of getting it wrong (silent wrong answers under one convention)**: every downstream recipe that depends on the first complex transcendental inherits a hidden bug. By the time someone notices, the dependency tree is wide and the fix requires re-deriving identities for every consumer. Months of work, plus correctness debt across every shipped recipe.

The verifier-port team paid the second cost on their substrate-cache work; their `--branch-aware-witness` mode was a *retrofit* after the cost was felt. Tambear can pay the first cost instead by ratifying this knob now.

---

## Threads downstream of this spec

If this lands clean, the related unowned thread (A) — TamSession dedupe-strategy choice (bitvector+radix-sort vs concurrent GPU hash map) — is the next architectural F13 antibody to surface. Different scope (data structures vs math semantics), same pattern (silent commitment via implicit choice).

The naturalist's catalog-as-tree pilot for one recipe family (means is the proposed first one) is *deeper* than either thread — it's the meta-pass over the entire recipe catalog asking "what's a parameter assignment vs what's a kernel?" Branch-cut conventions are themselves a parameter assignment in this view; this spec models the practice for one parameter axis.
