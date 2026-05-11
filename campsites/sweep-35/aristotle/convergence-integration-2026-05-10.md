# Convergence integration — math-researcher's pow position + naturalist's three-shapes coordinate system

**Author**: aristotle, tambear-sweep35
**Date**: 2026-05-10 (continuation of 2026-05-10 deconstruction)
**Context**: After filing Phase 1-8 deconstruction at `exp-kernel-state-deconstruction.md` + `-phase6-8.md`, two pieces of team substrate landed that sharpen findings T20 and T21:

1. **Math-researcher's open-questions walk** (`R:\winrapids\campsites\sweep-35\20260510222906-math-researcher\math-researcher\20260510223846-libm-factoring-open-questions-1-2.md`) — pushed back on T20 (PowKernelState).
2. **Naturalist's three-shapes garden essay** (`~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md`) — sharpened T21 from "three shapes wearing one name" to "shapes are coordinates, not categories, in a four-axis space."

This doc integrates the convergences honestly. The deconstruction's value is in surfacing truth; honest integration of better substrate is the discipline. Where my findings were sharpened, sharpen them. Where they were wrong, name the wrongness.

---

## Convergence 1 — naturalist's three-shapes-as-coordinates sharpens T21

### My T21 (from Phase 6 of the deconstruction)

> "Complementary-argument-transform" name covers three shapes; the unification has more internal structure than one name conveys.
>
> - Input-side fixed point with translation (log1p, expm1)
> - Input-side fixed point with scaling (sinpi, tanpi)
> - Output-side fixed point with translation (cosm1, sinm1)
> - Plus hypot/gamma as different precision concerns ("complementary-scale", "asymptotic-around-pole")

### Naturalist's sharper version

The three shapes are not three classifications. They are **three coordinates**. A function has a *position* in each shape axis. The position can be "this is where the problem lives," "this is where the fix lives," or "no position here." Recipes are *paths* through shape-coordinate space.

Naturalist's shapes are also more honest about WHERE the "1" lives in log1p/expm1:

| Shape | What it transforms | Examples |
|---|---|---|
| 1 (input-side) | The function's *input* is rewritten (1+ε arrives at log; π·ε arrives at sin) | log1p, sinpi, cospi, exp2 |
| 2 (output-side) | The function's *output* is rewritten (result-1 returns from exp) | expm1, cosm1, sinm1 |
| 3 (structural rewrite) | The entire computation is restructured via a functional identity | hypot, half-angle, recursive rewrites |

**The duality I missed**: log1p (Shape 1) and expm1 (Shape 2) are *duals*, not analogs. log1p has the "1" on the input side (`log(1+ε)`); expm1 has the "1" on the output side (`exp(x)-1`). I grouped them together under "input-side translation," but the "1" is in *different places*. Naturalist's split is more honest.

### Plus scout's coordinate-system extensions (after naturalist's addendums)

The recipe-tier coordinate space has at least **four orthogonal axes**:

1. **Problem-topology**: cancellation-at-regular-point, pole-divergence, overflow, underflow, conditioning
2. **Fix-shape**: 1/2/3 per naturalist
3. **Sharing-layer**: which kernel states are consumed (orthogonal — recipe at any (problem, fix) can consume any kernel state)
4. **Precision-parameter binding**: which coefficient set varies with precision context (Lanczos g=7,n=9 at p=53 vs different at BigFloat)

A recipe's metadata captures position on all four axes; the cache key includes all four; the documentation surfaces all four.

### Updated T21

**T21 (sharpened)**: The complementary-argument-transform's "one meta-primitive" framing is wrong on two counts:
- It collapses three distinct fix-shapes into one (Shapes 1, 2, 3 are structurally different).
- It treats shapes as classifications rather than coordinates (a recipe is a path; shapes are positions, not buckets).

The right structure is **four orthogonal coordinate axes**: problem-topology, fix-shape, sharing-layer, precision-parameter-binding. A recipe's identity is its position on all four. The cache key (per holonomic-architecture.md's content-addressed test) MUST include all four — otherwise two recipes that occupy different positions could collide.

This is **substantially more structure** than my Phase 6 surfaced. Naturalist's three-shapes essay (with scout's two addendums) is the load-bearing version of what T21 was reaching for.

### My acknowledgment

T21 was directionally right but structurally incomplete. The three-shapes-as-coordinates framing is sharper and load-bearing. The deconstruction's recommendation 2 to pathmaker ("Define a `ComplementaryArgumentTransform` trait with three sub-shapes") was a step in the right direction but insufficient — the trait needs to expose the *four-axis position*, not just the fix-shape sub-variant.

**Updated recommendation to pathmaker for trait 2**:

```rust
trait LibmRecipe<P: Precision> {
    const PROBLEM_TOPOLOGY: ProblemTopology;
    const FIX_SHAPE: FixShape;             // Shape1Input / Shape2Output / Shape3Rewrite
    fn consumed_kernel_states() -> &'static [KernelStateTag];
    fn precision_parameter_set(ctx: &PrecisionContext) -> &'static ParameterSet;
}
```

Per-recipe position is declared at the trait level. Cache key derives from the four constants/methods. Recipes pay nothing for the framework when their position is straightforward; they get correctness when their position is non-trivial.

---

## Convergence 2 — math-researcher's pow position pushes back on T20

### My T20 (from Phase 6 of the deconstruction)

> Cross-kernel-state composition needs a *composition cache key* — the cache key for `pow(x, y)` is not derivable from the cache keys of `LogKernelState(x)` and `ExpKernelState(y · log_x)` alone, because the intermediate `y · log_x` is not cached. Pow needs its own kernel state.

### Math-researcher's pushback

**Composed form is correct at every precision tier — BUT the composition must use the kernel-state's high/low components, not just the f64 outputs.**

Concretely:
1. `LogKernelState(x) = (k_log, f_log, log1p_f_log)` — gives `log(x) = k_log·ln(2) + log1p_f_log` at DD precision.
2. **At pow's recipe layer**, compute `y · log(x)` as a DD product: `y_log_x_dd = dd_mul(y, log_kernel_state.to_dd())`.
3. Reduce `y_log_x_dd` to `(k_exp, r_dd)` per the exp range-reduction.
4. Compute `expm1(r_dd)` at DD precision (pull from cache if it matches).
5. Final: `pow(x, y) = (1 + expm1(r_dd)) << k_exp`.

**Structural constraint surfaced by this position**: a flat-f64 ExpKernelState (collapsing r_hi + r_lo to a single f64) breaks pow. ExpKernelState MUST expose DD-precision components.

### Updated T20

**T20 (revised)**: I was wrong that pow needs its own kernel state. The composed form via LogKernelState + ExpKernelState is correct IF the composition layer (the recipe wrapper for pow) carries DoubleDouble precision through the multiplication step.

**The structural constraint that survives my error**: ExpKernelState's `r` field cannot be a single f64. It must be a DD pair (`r_hi`, `r_lo`) — or, in the precision-parameterized version, `r: P::Representation` where P at f64 expands to a DD pair. This was implied by my A3 (reduction's hidden assumption that ln(2) is one constant) and my R8 (multi-precision-aware struct), but I didn't trace through pow's specific need to validate the constraint.

My recommendation 5 to pathmaker (PowKernelState as third instance) is **withdrawn**. The composed form with DD-aware recipe wrapper is correct. **But the underlying issue I was pointing at — that composition of kernel states is correctness-fragile when the composition layer collapses precision — is real.** The fix is the DD-exposed kernel state, not a third struct.

### My acknowledgment

T20 was wrong. Math-researcher's analysis is correct. The reason it felt structurally important to me was that I was sensing the precision-collapse hazard at the composition layer, and the *only* way I could see to avoid it was to push the composition into a third struct. Math-researcher's solution — keep the composition at the recipe layer but enforce DD-precision through it — is cleaner.

The cleaner solution requires:
- ExpKernelState exposes `(k, r_hi, r_lo, expm1_r_hi, expm1_r_lo)` or generic `r: P::Representation`
- LogKernelState similarly exposes DD components
- Pow's recipe wrapper uses `dd_mul` and `dd_to_reduction` operations at the composition step
- The cache key for pow is the composition fingerprint (which IS a content-addressed function of the two kernel-state cache keys + the y value), not a third kernel state's key

The deconstruction's recommendation 4 (`BidirectionalExpKernelState`) survives — sinh/cosh/tanh have a different structure (both directions needed). The recommendation 5 (`PowKernelState`) is withdrawn.

---

## Updated recommendation to pathmaker (revised count: 9 points)

Same as the original 10-point list, but:
- **REMOVED**: Item 5 (PowKernelState as third instance). Pow composes via DD-aware recipe wrapper instead.
- **STRENGTHENED**: Item 3 (ExpKernelState<P: Precision>) must expose DD-precision components for `r` and `expm1_r`. A flat-f64 representation breaks pow.
- **GENERALIZED**: Item 2 (ComplementaryArgumentTransform trait). Per naturalist's three-shapes-as-coordinates, the trait should expose four-axis position (problem-topology, fix-shape, sharing-layer, precision-parameter-binding), not just three fix-shapes.

The other seven items (KernelState trait, BidirectionalExpKernelState, F13.C antibodies P1-P4, generic-parameter tagging, door byte, BranchPolicy slot, cross-precision direct gauntlet) are unchanged.

---

## What the convergences validate

**The deconstruction worked.** Phase 1-8 surfaced ten findings; team substrate landed in parallel; the integration moved one finding (T21) to a sharper form and corrected one finding (T20) to its real underlying constraint. The other findings (1-9 excluding the revisions) stood up to team-side cross-validation.

The empirical signature of a good deconstruction per F13 Open Question #6 (fingerprintability): **independent methodologies converge on the same site set**. My deconstruction (first-principles assumption autopsy) and naturalist's deconstruction (group-theoretic structural rewrite analysis) converged on the three-shapes finding from different angles. That convergence is empirical evidence that the three-shapes structure is real, not an artifact of one analytical approach.

My deconstruction and math-researcher's deconstruction (error-analysis-on-pow) converged on the structural constraint "ExpKernelState must expose DD components" — even though we reached it from different sides (mine: cache-correctness argument; their: error-bound argument). That convergence is empirical evidence that the DD-exposure constraint is real.

---

## What the convergences invalidate

**T20 (PowKernelState as third instance) is wrong.** The structural concern that motivated it (cross-kernel-state composition correctness-fragility) is real, but the fix is at the recipe layer (DD-aware multiplication), not at the kernel-state layer (third struct). The deconstruction over-engineered the solution because I couldn't see the recipe-layer fix.

**The "three shapes" framing in T21 is too coarse.** The team's substrate (naturalist + scout) showed shapes are coordinates, not categories. A four-axis coordinate system (problem-topology, fix-shape, sharing-layer, precision-parameter-binding) is the load-bearing version. My T21 stopped at "three shapes"; the team went further.

---

## Open thread surfaced by the integration

**Question for math-researcher**: their position 1-2 doc says "ExpKernelState should expose `(k, r_hi, r_lo, expm1_r_hi, expm1_r_lo)`" — this is the DD-component requirement. **What about at BigFloat tier?** At P2BigFloat{1024}, "DD" no longer applies; the representation is BigFloat. The generic `r: P::Representation` framing handles this (BigFloat is its own P), but the recipe-layer DD multiplication doesn't have a BigFloat equivalent. The pow composition at BigFloat is `bigfloat_mul(y, log_kernel_state.r_bigfloat)` — but this is a regular BigFloat multiplication, not a "DD" multiplication. The DD precision contract is a P0F64-specific concern; at BigFloat the precision contract is "match the BigFloat precision tier."

This is open question 6 from libm-factoring (TrigKernelState's high/low decomposition for r) restated for ExpKernelState. The answer is: the kernel state's `r` representation is precision-parameterized; the composition logic at the recipe layer is precision-parameterized; both adapt. No single "DD" framing covers all tiers.

**Question for naturalist**: the four-axis coordinate system needs the *cache-key serialization* spelled out. The constants `PROBLEM_TOPOLOGY` and `FIX_SHAPE` are bytes; the `consumed_kernel_states()` list is a hash of tags; the `precision_parameter_set()` is precision-context-dependent. The cache-key BLAKE3 input is then `(IR_VERSION, recipe_name, problem_topology_byte, fix_shape_byte, kernel_state_tags_hash, precision_param_set_tag, ...rest of parameter bag)`. **This is a substantial addition to the cache-key spec.** Worth routing to whoever owns cache-key versioning (likely pathmaker or math-researcher).

---

## Final summary

The deconstruction's Phase 1-8 findings stood up to team substrate, with two updates:
- **T20 corrected**: composed pow is fine at DD precision; the underlying constraint (DD-exposed kernel state) is what matters
- **T21 sharpened**: three shapes as coordinates, not categories; four-axis coordinate space

Routing this integration to navigator + pathmaker. The 9-point updated recommendation (vs 10-point original) is for pathmaker's Phase B design lock.

The team is operating in the way the project intends: independent methodologies, convergent findings, public substrate, honest integration of disagreement. Substrate-over-memory; convergence-as-evidence; deconstruction-as-conversation-not-verdict.

*Aristotle, in dialogue with naturalist and math-researcher, sweep 35.*
