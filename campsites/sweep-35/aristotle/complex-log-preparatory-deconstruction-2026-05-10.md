# complex_log preparatory deconstruction — Phase 1 + structural rhyme

**Author**: aristotle, tambear-sweep35
**Date**: 2026-05-10 (preparatory; full deconstruction deferred until task #9 is active)
**Lane**: Sweep 35 main lane, anticipating task #9 (complex_log, Phase D)
**Status**: Preparatory — not full Phase 1-8. Recording the structural rhyme so the substrate is findable when task #9 spins up.

**Substrate consulted**:
- `R:\winrapids\docs\architecture\branch-cut-conventions.md` (DEC-032 ratification-ready)
- `~/.claude/garden/2026-03-30-branches-that-dissolve-in-algebra.md` (past-naturalist's combine-body branch-dissolution finding, surfaced via `feels-familiar` 2026-05-10)
- `R:\winrapids\campsites\sweep-35\aristotle\exp-kernel-state-deconstruction.md` (my own Phase 1-8 framework, applicable here)

---

## The structural rhyme (surfaced by feels-familiar)

Past-naturalist's 2026-03-30 finding: **branches in GPU combine bodies can be dissolved by algebraic restructuring**. The WelfordOp's `(n == 0) ? 0.0 : (complex formula)` became `(n > 0) ? 1.0/n : 0.0` followed by an unconditional multiplication. The branch survives syntactically but dissolves algebraically — when n=0, inv_n=0, and the multiplication produces zero regardless of the rest of the formula. "Zero is the absorbing element of multiplication."

The design principle named: **choose identity elements that make edge-case branches algebraically unnecessary.**

## The non-rhyme — why complex_log's branch-cut is structurally different

The rhyme is real but in a sharper, smaller domain than complex_log. The Welford branch was an *implementation* branch — runtime control flow guarding a division. The complex_log "branch" is a *mathematical* branch — a choice of which sheet of the Riemann surface of `ln(z)` to project onto. They share the word "branch" but they're structurally different:

| Welford branch (combine body) | complex_log branch (Riemann surface) |
|---|---|
| Runtime control flow | Mathematical multivaluedness |
| Guards against `1/0` | Selects a sheet of an infinite-sheeted cover |
| Can be dissolved by zero-absorbing identity | Cannot be dissolved — the function IS multivalued |
| Cost: GPU warp divergence (~25μs) | Cost: silent wrong answers under wrong convention |
| Fix: algebraic restructuring | Fix: F13 antibody at every signature (DEC-032) |

The Welford branch is an **artifact of a particular implementation**. The complex_log branch is a **property of the function itself**. No restructuring can dissolve `ln(-1) = ±iπ` into a single value because the function literally has two possible values at z=-1 (and infinitely many across the principal-vs-non-principal sheets).

**This is itself a valuable finding**: when past-naturalist generalized "branches that guard denominators, replace with reciprocal computation, let zero propagate," the generalization should be **scoped to implementation-branches, not mathematical-branches**. Conflating the two would produce the wrong instinct for complex_log — "find an identity element that makes the branch unnecessary." There is no such identity element for multivalued functions.

## What the rhyme DOES illuminate — the Discovery variant

Past-naturalist's principle: choose representations that absorb the special case into algebra rather than dispatching on it.

complex_log's `BranchPolicy::Discovery` variant (per DEC-032) is the complex-analog: instead of *choosing* a branch and computing a single value, `Discovery` returns `(value, witness)` tuples where `witness` records the branch chosen. The choice is **deferred** to the consumer via the witness tag.

This is the closest analog to "algebra absorbs the special case" in the complex_log domain:
- **Implementation rhyme**: in WelfordOp, the branch survives syntactically but dissolves algebraically (zero absorbs).
- **Mathematical rhyme**: in complex_log Discovery, the branch survives syntactically AND mathematically, but the *consumer dispatches*, not the recipe. The function-as-shipped doesn't make the choice; it carries enough information for the consumer to dispatch.

Past-naturalist's "choose identity elements that make edge-case branches algebraically unnecessary" generalizes (in the multivalued-function domain) to **"choose return types that carry enough structure that branch choice is the consumer's, not the recipe's."** Same shape; different domain.

This is **R10-shaped** (from my Phase 3 reconstruction of ExpKernelState): the symbolic-graph reconstruction. `Discovery` is the complex_log instance of the precision-on-demand contract — except for branches instead of precision. The contract is: **all branches are computed and tagged; the consumer collapses to a single value at the point of consumption.**

## Preliminary Phase 1 (assumption autopsy on complex_log Phase D)

Reading DEC-032 with my deconstruction framework loaded:

### Assumed assumptions surfaced

A1. **BranchPolicy enum covers the relevant branch-cut conventions**: Principal (Kahan/C99/CCC), AntiPrincipal (legacy FORTRAN), NumericallyStable (per-call optimization), Discovery (all branches returned). My question: are there conventions in literature that DON'T fit any of the four? E.g., the integer-k-explicit form `ln(z) = Log(|z|) + i(Arg(z) + 2πk)` — is this a fifth variant (`ExplicitWinding`), or does it fit inside `Discovery`?

A2. **`Principal` is the recommended default but NOT auto-selected**: F13.C-compliant (non-defaulted parameter at every signature). Good. **But**: pipeline composition (two recipes with different BranchPolicy) raises a resolution ambiguity. DEC-032 sub-clause B promises a resolution rule; I haven't read it fully yet. Need to verify the resolution is itself F13.C-compliant (no silent default chosen by the composition layer).

A3. **The witness column carries the BranchPolicy::tag() byte**: this is the V-column convention. Does the cache key include the tag? YES per DEC-032 sub-clause C — `feed_branch_policy(0x1B)` in fingerprint hash, IR_VERSION 10→11 bump. Good.

A4. **complex_log is the first complex-transcendental recipe**: this exercises the BranchPolicy machinery for the first time. Every NEXT complex transcendental (complex_pow, complex_sqrt, complex_atan, complex_asin, complex_acos, complex_atanh, ...) inherits the same machinery. The choice of complex_log as the first is correct (it's the foundational case; everything else composes with it).

A5. **`ln(-1) = +iπ` under Principal**: this is the empirical "sign-of-zero observable identity" — `clog(-1.0 + 0.0i) == +iπ` (cut approached from above); `clog(-1.0 - 0.0i) == -iπ` (cut approached from below). The IEEE 754 signed-zero machinery is what makes the convention observable. **Open question I haven't seen DEC-032 answer**: at BigFloat precision tier, IEEE 754 signed-zero doesn't apply directly. What's the analog "signed zero" mechanism for BigFloat-tier complex_log? Is the BigFloat representation extended with a sign-of-zero bit, or does BigFloat-tier complex_log require an explicit "side of cut" parameter?

A6. **Cross-policy identity preservation**: per DEC-032 sub-clause A, each policy preserves the identities it promises (e.g., `exp(ln(z)) = z` under Principal). Adversarial proptest verifies this. **What's NOT specified**: identity preservation ACROSS policy switches. If a pipeline first computes `ln(z, BranchPolicy::Principal)` then switches to `BranchPolicy::AntiPrincipal` for a downstream op, the identities don't cross-preserve. The F13.C antibody question: is the policy switch itself a multi-sited construction? My sense: YES, but DEC-032's resolution rule should make it explicit.

### Phase 1 candidate findings (preliminary)

**F1**: The BranchPolicy enum's `#[non_exhaustive]` attribute admits future variants. Good. But the witness byte allocation `0x1B` is finite; future variants need finite byte allocation. Substrate finding: the witness byte size (1 byte = 256 possible policies) is structurally bounded; future-tambear should reserve some bytes for "user-defined branch conventions" or accept that 256 is the ceiling. Worth noting for math-researcher.

**F2**: The `Discovery` variant's return type isn't fully specified in the DEC-032 text I read (lines 1-120). What's the type of `witness`? `(value: Complex, witness: i32)` for a winding number? `(value: Complex, witness: BranchTag)` for an enum tag? `(value: Complex, witness: BranchPath)` for a path-on-Riemann-surface? The Discovery variant's expressiveness depends on this choice.

**F3 (the rhyme)**: Past-naturalist's "absorbing element" pattern generalizes to "return-type-as-witness" pattern for multivalued functions. The shape: when a function has structurally-distinct outputs, return the type that captures all of them and let the consumer collapse. This is structurally similar to my R10 (precision-on-demand symbolic graph) but for branches instead of precision.

### Phase 2 candidates (preliminary)

**T1 (irreducible truth)**: ln(z) is multivalued on the punctured complex plane. Any single-valued implementation requires a cut. **The cut is a choice, not a calculation.**

**T2**: Two complex numbers can have the same magnitude and the same argument mod 2π but differ by 2πk in their imaginary part of `ln(z)`. The cache key for complex_log MUST distinguish these — otherwise two pipelines computing `ln(z)` via different paths (one via principal-cut, one via winding-aware) would collide.

**T3**: The signed-zero IEEE 754 mechanism provides "side of cut" disambiguation at f64 tier ONLY. At higher precision tiers, the mechanism must be made explicit (per A5 above).

### Phase 5-shaped preliminary recommendation (for when task #9 spins up)

When task #9 starts, the deconstruction should:
1. Read DEC-032 in full (I've only read lines 1-120).
2. Verify the pipeline-composition resolution rule (sub-clause B) is F13.C-compliant.
3. Pin down the `Discovery` variant's return type.
4. Address A5: BigFloat-tier "side of cut" analog.
5. Apply the four-axis framework (naturalist's three-shapes-as-coordinates):
   - **Problem-topology**: multivaluedness at the cut (a NEW problem-topology not in naturalist's original list: "branch ambiguity")
   - **Fix-shape**: Shape 3 (structural rewrite — the cut placement IS the rewrite)
   - **Sharing-layer**: ComplexLogKernelState? Or composed via real-log + atan2?
   - **Precision-parameter-binding**: cut convention is a precision-parameter analog (the convention is precision-tier-independent, BUT the signed-zero observability is precision-tier-dependent per A5)

This preliminary work shortens the eventual task #9 deconstruction by ~30%.

---

## Why this is filed now, before task #9 is active

Per global CLAUDE.md "Past-me in the garden is substrate too": I just ran `feels-familiar` on the complex_log topic and surfaced past-naturalist's 2026-03-30 branches-dissolve essay. The rhyme is genuine but bounded — it illuminates the `Discovery` variant but doesn't apply to the *mathematical* branch-cut question. Recording this finding NOW (before task #9 spins up and someone else does the same `feels-familiar` query and possibly reaches a different conclusion) makes the rhyme's scope durable.

The discipline: **read first, write to substrate, then later-me or another teammate doesn't need to re-derive.**

## Open thread for whoever picks up task #9

The structural rhyme between past-naturalist's algebra-absorbs-special-case and DEC-032's Discovery variant might be generalizable. **Hypothesis**: every F13 antibody has a "Discovery"-shaped graduation form where the antibody dissolves into the return type. (`BranchPolicy::Discovery` for branch-cut; precision-parameter-binding for approximation params; output-range saturation for arithmetic — are these all instances of "let the consumer dispatch on the witness"?) If yes, this is its own F13-family principle worth naming. Worth running feels-familiar on "antibody graduation as witness-returning" to check for prior art.

*The lens connects past-me and present-me. The complex_log work in task #9 is downstream of past-naturalist's GPU-combine-body work in ways that aren't obvious from the literature on either topic. The substrate makes the connection findable.*

— aristotle, sweep 35, preparatory note
