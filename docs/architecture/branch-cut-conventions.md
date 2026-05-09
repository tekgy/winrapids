# Branch-cut conventions for complex-transcendental recipes

**Status**: Ratification-ready (2026-05-09). Math-researcher's literature-anchored amendments applied — sub-clauses D-prime, E (extended), F added. Awaiting copy-to-DEC-032 in `R:\tambear\docs\decisions.md`.

**Drafted**: 2026-05-08, main-thread Claude (tambear-sweep31-finish team-lead lane), with Tekgy.

**Ratified**: 2026-05-09 by math-researcher; ratification doc at `R:\winrapids\campsites\tambear-sweep31-finish\math-researcher\branch-cut-ratification-2026-05-09.md` (Kahan 1987, Chyzak et al. 2011, C99 §G.6, CLISP §12.5.3 anchors).

**Anchor**: `R:\winrapids\PLEASE_READ_from_gpu_verifier_port.md` finding 4 + naturalist's `R:\winrapids\campsites\tambear-formalize\naturalist\` thread "graph-form keeps appearing as recognition" (2026-05-08).

**Roadmap path**: Math-researcher copies ratified content to `R:\tambear\docs\decisions.md` as DEC-032 when ready. Pathmaker implements at first-complex-transcendental-recipe time. Until then, **no complex-transcendental recipe enters the queue.**

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
    /// Kahan 1987 / C99 §G.6 / counter-clockwise-continuous (CCC)
    /// convention. The unique cut placement compatible with IEEE 754
    /// signed-zero arithmetic. Universal across modern systems —
    /// Mathematica (≥v10), Wolfram, MATLAB R2016a+, NumPy, Boost,
    /// Sage, Maple, GSL all converge on this. See sub-clause D-prime
    /// for explicit cut placements per function. `clog(-1) = +iπ`.
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
            // Tag 0 RESERVED — asserts "uninitialized / byte not fed."
            // Per math-researcher ratification 2026-05-09: keeping 0
            // free gains explicit antibody coverage on the "did the
            // byte get fed?" question, at zero cost. Pre-bump kernels
            // can never silently match post-bump kernels.
            BranchPolicy::Principal         => 1,
            BranchPolicy::AntiPrincipal     => 2,
            BranchPolicy::NumericallyStable => 3,
            BranchPolicy::Discovery         => 4,
        }
    }
}
```

`#[non_exhaustive]` allows future variants without re-ratifying the existing four. Reserved tag `5` for `Custom(Vec<CutSegment>)` v2-amendment.

### D-prime. `Principal` is normatively Kahan / C99 / CCC

Per math-researcher's literature ratification (2026-05-09):

`Principal` is **the Kahan 1987 / C99 §G.6 / counter-clockwise-continuous (CCC) convention** — not "one of several principal-value conventions." This is the unique cut placement compatible with IEEE 754 signed-zero arithmetic (Kahan 1987 §3, §4). The contested-placement framing in earlier drafts was a 1990s-era residue: modern Mathematica, Wolfram, MATLAB, NumPy, Boost.Multiprecision, Sage, Maple, GSL, and IEEE 754-2019 §9.2 normative recommendations have all converged on Kahan's CCC. The real-axis arctan variant survives only in legacy FORTRAN libraries.

**Normative cut placements** for the twelve canonical complex-transcendental recipes:

| Recipe | Cut placement | Sign at cut | Reference |
|---|---|---|---|
| `clog(z)` | negative real axis | approached from above; `clog(-1) = +iπ` | C99 §G.6.3.4 |
| `csqrt(z)` | negative real axis | approached from above | C99 §G.6.4.2 |
| `casin(z)` | `(-∞, -1] ∪ [+1, +∞)` on real axis | per CCC | C99 §G.6.2.1 |
| `cacos(z)` | `(-∞, -1] ∪ [+1, +∞)` on real axis | per CCC | C99 §G.6.2.2 |
| `catan(z)` | `(-i∞, -i) ∪ (+i, +i∞)` on imaginary axis (excluding ±i) | per CCC | C99 §G.6.2.3 |
| `casinh(z)` | `(-i∞, -i) ∪ (+i, +i∞)` on imaginary axis | per CCC | C99 §G.6.2.4 |
| `cacosh(z)` | `(-∞, +1)` on real axis | per CCC | C99 §G.6.2.5 |
| `catanh(z)` | `(-∞, -1] ∪ [+1, +∞)` on real axis | per CCC | C99 §G.6.2.6 |
| `cexp(z)` | (entire — no branch cut) | — | — |
| `csin(z)`, `ccos(z)`, `ctan(z)` | (entire — no branch cuts) | — | — |
| `cpow(z, w)` | inherits from `clog`: `cpow(z, w) = cexp(w · clog(z))` | per CCC | C99 §G.6 derived |
| `complex_root(z, n)` | inherits from `cpow(z, 1/n)` | per CCC | C99 §G.6 derived |

These placements are derived from a single principle (Kahan 1987 §3): counter-clockwise continuity at the cut, which fixes the sign-of-zero behavior in the limiting approach.

**`AntiPrincipal` is the sign-conjugate**: same cut placements, but with the imaginary part of values on the cut negated where the function is purely-imaginary on the cut. Implementations obtain `AntiPrincipal` outputs by post-processing `Principal` outputs; the conjugate convention is checkable by the trivial identity `AntiPrincipal(z) = conj(Principal(conj(z)))` for real-axis cuts.

**Implementation cross-check**: any `Principal` recipe must reproduce the C99 §G.6 reference values bit-for-bit at f64 (the C99 reference points where conjugation-of-zero-sign is observable). Adversarial proptests verify against:
- `clog(-1.0 + 0.0i) == +iπ` (cut approached from above)
- `clog(-1.0 - 0.0i) == -iπ` (cut approached from below — sign-of-zero matters)
- `csqrt(-1.0 + 0.0i) == +i`
- `casin(2.0 + 0.0i) == π/2 + i·acosh(2.0)` (above-cut limit)
- `catan(2.0i + 0.0) == π/2 + i·atanh(0.5)` (right of imaginary cut)

### E. Discovery output shape — per-family bounds

The `Discovery` policy enumerates multiple branches. The literature distinguishes between three multi-valuedness regimes; the output shape adapts per recipe family. Per math-researcher ratification (2026-05-09):

**(E.1) Single-valued-on-cut** — `complex_log`, `complex_arctan`, `complex_arctanh`. Naturally enumerable by integer winding number. Output:

```rust
pub struct WoundComplex {
    pub primary: Complex<f64>,                  // Principal branch
    pub windings: Vec<(i64, Complex<f64>)>,    // (winding_number, value)
}
```

User specifies `using(max_windings: i64)`, default `0` (just `primary`). Setting `max_windings = 3` returns winding numbers `{-3, -2, -1, 0, 1, 2, 3}`.

**(E.2) Finite-root** — `complex_sqrt`, `complex_root(z, n)`, `complex_pow(z, p/q)` for rational `p/q`. Enumerate all `n` (or `q`) roots:

```rust
pub struct RootedComplex {
    pub primary: Complex<f64>,                  // Principal root
    pub roots: Vec<(BranchTag, Complex<f64>)>, // All other roots
}
```

For `complex_root(z, n)`, `roots.len() = n - 1` (the n principal roots minus `primary`). For `complex_pow(z, p/q)`, the irreducible-fraction `q` determines branch count.

**(E.3) Dense-branch** — `complex_pow(z, w)` for irrational `w`. Branches are dense in `ℂ \ {0}` (countably infinite, enumerable but non-terminating). User MUST specify `max_branches: usize`:

```rust
pub struct BranchedComplex {
    pub primary: Complex<f64>,                          // Principal value
    pub witnesses: Vec<(BranchTag, Complex<f64>)>,     // up to max_branches
    pub truncated: bool,                                // true if true count > max_branches
}
```

**Antibody (F13-shaped)**: `complex_pow(z, w)` with irrational `w` and `BranchPolicy::Discovery` selected without explicit `using(max_branches: ...)` panics at construction. "No silent-default for an enumeration that can't terminate" — there is no principled bound to default to, so refusing to default is the correct antibody.

The per-family enumeration matches the literature distinction between finite-multi-valued and dense-multi-valued functions. The naturalist's catalog-as-tree pattern surfaces here: `Discovery` is the structural-rhyme analog to `discover()` for sketches/correlations/distances. The relationship IS the answer; the enumeration shape is the relationship's structural type.

### F. NumericallyStable × DEC-031 precision-tier interaction

Per math-researcher ratification (2026-05-09): when `BranchPolicy::NumericallyStable` is active, the recipe **internally widens by ≥50 bits above the requested precision** for the branch-selection step. The branch chosen at the wider precision is committed; the per-branch arithmetic then proceeds at the user-requested precision and rounds via the active `RoundingMode`.

The 50-bit widening is the same constant used by BZ §3.1.6 + DEC-031 §3.5 for Newton-iteration guard bits. NS borrows it for symmetry — the structural argument is identical: the *selection step* needs more precision than the *output*, otherwise the cancellation NS is supposed to detect corrupts the comparison NS uses to select.

**Why this matters**: without the widening, NS reduces to "Principal with bookkeeping." Branch-selection at the requested precision incorporates the cancellation it's supposed to be avoiding. The chosen branch ends up identical to `Principal`'s default for nearly all inputs — *except* in the 0.0001% adversarial-input regime where NS *would* matter, the comparison is corrupted by the very cancellation NS is supposed to detect. Silent wrong answer at the precise input regime where the user reached for NS.

**Per-recipe widening declarations**: recipes whose analytic cancellation bound exceeds 50 bits (e.g., `complex_log` near `z = -1` where the imaginary part can cancel a multi-thousand-bit mantissa) declare a higher per-recipe widening in their `spec.toml` stance metadata. The widening is part of the F12 stance contract.

**Interaction with `PrecisionContext`**: when `PrecisionContext::dispatched_precision_bits()` is `53` (P0F64) or `106` (P1DD), the NS internal widening forces a tier bump to the next available tier (P0F64 → P1DD; P1DD → P2BigFloat with `precision_bits = 106 + 50 = 156`). The widened computation completes at the higher tier; the result rounds back to the user's tier on output. Tier-bump cost is paid once per call — acceptable since NS is reached for explicitly when the user wants the cancellation-aware computation.

---

## Roadmap

1. **Now**: this doc lands as winrapids-architecture roadmap. Math-researcher reviews open questions, ratifies variant set + sub-clauses A–E, signs off.
2. **Ratification commit (math-researcher async)**: copies ratified content to `R:\tambear\docs\decisions.md` as DEC-032; updates the open questions section with closure rationale; closes any thread loops with the verifier-port team.
3. **First complex-transcendental recipe**: pathmaker implements `BranchPolicy` enum + `feed_branch_policy(0x1B)` + IR_VERSION bump + the first recipe (likely `complex_log`). Adversarial proptests verify that all four policies produce identities the policy promises.
4. **Subsequent complex-transcendental recipes**: every one inherits the antibody — no defaulted `BranchPolicy`, mandatory at call site, tagged in cache key, V-column-recorded in outputs.
5. **Future Tier-2 amendment**: if a fifth convention surfaces (custom cuts, parametric branches, Riemann-surface unfolding), `#[non_exhaustive]` allows extension without re-ratifying.

---

## Open questions — ratification status (2026-05-09)

Math-researcher's ratification doc at `R:\winrapids\campsites\tambear-sweep31-finish\math-researcher\branch-cut-ratification-2026-05-09.md` resolved all five. Status:

1. **Variant enumeration sufficiency** — **CLOSED**: APPROVE four-variant base set for v1; `Custom(Vec<CutSegment>)` reserved for v2 amendment via `#[non_exhaustive]`. (Tag `5` reserved for the Custom variant.) Anchor: Kahan 1987 §1; CCC convention is universal at v1.

2. **Default choice within `Principal`** — **CLOSED**: ONE bundled convention. `Principal` is normatively Kahan / C99 §G.6 / counter-clockwise-continuous (CCC). The "arctan placement is contested" framing in the v1 draft was a 1990s-era residue; modern Mathematica, Wolfram, MATLAB, NumPy, Boost, Sage, Maple, GSL have all converged on Kahan's imaginary-axis convention. Sub-clause D-prime now codifies the placements normatively. Anchor: Kahan 1987 §3; C99 §G.6.2/3/4; CLISP §12.5.3.

3. **NumericallyStable × DEC-031 interaction** — **CLOSED**: NS implies internal precision-tier bump of ≥50 bits above the requested precision for the branch-selection step. Per-recipe analytic-cancellation-bound overrides via `spec.toml` stance metadata (F12 contract). Sub-clause F now codifies. Anchor: BZ §3.1.6 / DEC-031 §3.5 guard-bit conventions, structurally extended.

4. **Discovery output bound** — **CLOSED**: per-family enumeration. Single-valued (`WoundComplex` indexed by integer winding); finite-root (`RootedComplex` for n-th root family); dense-branch (`BranchedComplex` with mandatory `max_branches`). Antibody: `complex_pow(z, w)` with irrational `w` and `Discovery` selected *without* explicit `using(max_branches: ...)` panics at construction. Sub-clause E (now extended) codifies. Anchor: Riemann-surface theory; finite-vs-dense multi-valuedness distinction.

5. **Pipeline strictness** — **CLOSED**: keep compile-fail. Reject alphabetical defaults entirely. Reproducibly-wrong is still wrong; the antibody is silent-commitment-via-implicit-choice, and alphabetical-default is silent-commitment-by-another-name. Anchor: F13 antibody discipline.

**Sign-off**: math-researcher signs off on Q1 + Q5 as originally written; Q2 / Q3 / Q4 sign-offs land with the three spec edits applied above (this commit).

**Next step**: spec is ratification-ready. Math-researcher copies content to `R:\tambear\docs\decisions.md` as **DEC-032** when ready (async). Adversarial designs identity-preservation proptests for each policy + recipe combination at first-complex-recipe time. Aristotle adds branch-cut-violation cases to the silent-failure gauntlet.

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
