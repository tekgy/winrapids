# Lab Notebook 011: Aristotelian deconstruction of "accumulate + gather"

**Date**: 2026-04-11
**Authors**: Aristotle (agent role on the Bit-Exact Trek)
**Branch**: main
**Status**: Complete — Phases 1–8 drafted; stable at Move v5; Phase 8 surfaced speed-vs-correctness axis separation
**Depends on**: `campsites/expedition/20260411120000-the-bit-exact-trek/invariants.md` (I7), project memory `project_accumulate_unification.md`, `project_tambear_truths.md`

## Context & Motivation

The Bit-Exact Trek rests on an architectural claim: one `.tam` source, bit-identical numerical answers on any ALU. The load-bearing invariant for that claim's *compositionality* layer is **I7 — Every primitive decomposes into accumulate + gather.** Before the trek hardens this into the IR, the scientific conscience role (Aristotle) is supposed to check whether the assumption survives a first-principles autopsy. If I7 is load-bearing for Peaks 1, 3, 5, 6, and 7, any hidden softness there propagates across the whole expedition.

This isn't a wet experiment — it's a design-decision deconstruction, which the lab-notebook skill explicitly calls out as a valid reason to open a notebook. The "decisions table" is the highest-value artifact.

## Hypothesis

**H0 (what the project currently assumes):** Accumulate + gather is a structurally justified, near-universal decomposition of computable math, with Kingdoms B/C/D as a small enumerated set of honest exceptions, and I7 as the fence that keeps the compositional property alive.

**H1 (Aristotle's counter-hypothesis):** Accumulate + gather is a *useful engineering decomposition* that fills two of the four roles any computation representation must fill (what, where, order, destination), but it fills the "order" role *implicitly* through the associativity assumption of the `op` parameter. That implicit filling is a hidden contract between the IR and its backends — exactly the kind of un-stated assumption that is cheap to make explicit now and expensive to retrofit later. Under H1, I7 needs a sibling invariant making total order first-class.

**Prediction:** The Phase 4 assumption-vs-truth map will show a collision at the "two primitives fill four roles" boundary, and the Aristotelian Move will be to surface total order as first-class IR metadata.

## Design

Eight-phase Aristotelian deconstruction (Phases 1–8 from the role prompt). Working document is at:

```
R:\winrapids\campsites\expedition\20260411120000-the-bit-exact-trek\peak-aristotle\accumulate-gather-phases.md
```

No code is run. The "experiment" is a structured attack on one assumption. The deliverables are:
1. A numbered assumption list (Phase 1)
2. An irreducible truths list (Phase 2)
3. Ten alternative reconstructions ranging elegant → ambitious (Phase 3)
4. A collision map between assumptions and truths (Phase 4)
5. A single highest-leverage action (Phase 5)
6. Recursive challenge of the Phase-5 result (Phase 6)
7. Stability verification (Phase 7)
8. Forced rejection: what if the whole framing is wrong (Phase 8)

## Design Decisions (what we chose AND rejected)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| First target | "Why accumulate + gather?" | "Why bit-exact cross-hardware?", "Why mpmath as oracle?" | I7 sits under the most architectural load — every peak depends on it. Deconstructing it early has the highest blast radius for the trek. |
| Phase depth | Run Phases 1–5 immediately; defer 6–8 | Run all eight before surfacing anything | Phase 5 yields the Aristotelian Move that navigator needs to route. Sitting on it to "polish" the recursion would delay the move that unblocks IR Architect. |
| Output format | Long phase document + lab notebook + navigator message | Navigator message alone | Phase document is the durable artifact. Navigator message is just routing. The notebook ties the two into the project's research log. |
| Framing | Dataflow vs total order as the key collision | Associativity loophole, kingdom-escape-hatch loophole, catalog-incompleteness loophole | All three loopholes are real, but the "dataflow vs total order" framing yields one clean, actionable move. The others fragment into multiple small moves. Clean is higher leverage. |
| Aristotelian Move | Make order_strategy first-class in .tam IR | Leave it implicit and trust the backends + Peak 6 determinism work | Making it explicit now is free (IR is still being designed). Leaving it implicit means Peak 6 relitigates the same decision for every new op. Structural > operational. |

## Results

### Phase 1 — Assumption Autopsy

Seven stacked assumptions inside "accumulate + gather":

1. That mathematics decomposes at all.
2. That there's a *unique natural* decomposition.
3. That it should be *two* primitives (not one, not three, not nine).
4. That the two are *accumulate and gather* specifically, vs. Bird's fold/map, Blelloch's map/reduce/scan, MapReduce's map/reduce_by_key, Halide's pure/reduction, APL's reduce/index.
5. That the decomposition is the *skeleton* of computation rather than one representation.
6. That "every primitive" is a well-formed set (what counts as a primitive is partly self-fulfilling).
7. That decomposition should be the *primary* architectural axis rather than, say, dependency topology or algebraic structure or stability class.

Plus three sub-assumptions: 8-and-6 grouping/addressing cardinalities are empirical not derived; kingdoms are coupled to tensor rank, which second-classes non-tensor math; "runs on every ALU" conflates "runs" with "produces the same bits."

### Phase 2 — Irreducible Truths

Nine undeniable facts survived:

1. Finite computation → finite output via finite primitive steps (computability).
2. Primitive steps are opaque at the ALU level.
3. Parallel execution requires exposing data dependencies.
4. Reproducibility requires a total order of operations (because fp isn't associative).
5. Bit-exact cross-hardware requires the *same* total order, not an equivalent one.
6. A compiler can only produce what its IR can name.
7. Many valid orderings differ only in bits; determinism picks one.
8. Any computation representation must encode four roles: what, where, order, destination.
9. A decomposition is useful iff (a) everything intended maps to it, (b) each primitive lowers with equivalent semantics to every target, (c) composition preserves equivalence.

### Phase 3 — Ten Reconstructions

Ranked elegant → ambitious:
1. One primitive: `eval(expr_tree)` — Halide/TVM-style DAG evaluation.
2. Pure fold (Bird-Meertens F-algebra).
3. Three primitives: map, reduce, scan (Blelloch canon).
4. Two primitives: map + reduce_by_key (MapReduce).
5. **Accumulate + gather (the current choice).**
6. `scatter_add + ordered_fold` — determinism structurally mandatory.
7. Decomposition by *time*: `pure_step + memoize`.
8. Kingdoms as primary axis (not as escape hatch).
9. Decomposition by *algebraic structure* (semigroup/monoid/semiring class).
10. No compile-time decomposition: causal-event ledger with JIT discovery.

All ten are viable skeletons for a numerical library. Choice #5 is not forced by mathematics — it's a local optimum for statistical workloads, and it inherits a hidden assumption from its op parameter.

### Phase 4 — Assumption vs Truth Collision

The critical collision: **"It's two primitives" vs Truth 8 (four roles must be filled).** Accumulate+gather fills four roles with two primitives only by *overloading* — accumulate covers what, order, and destination simultaneously, relying on op's associativity to make "order" not matter. When op is not associative, the overloading breaks and we invoke the kingdom escape hatch. The escape hatch is evidence that the overloading is not universal.

Second collision: **"Runs on every ALU" (the portability claim) vs Truths 4 and 5 (bit-exact cross-hardware requires same total order).** I7 gives compositionality. I5 gives determinism. The trek's architectural claim needs both, but I7's framing makes it easy to conflate them, and Peak 6 is spending a whole peak trying to patch what I7 leaves implicit.

### Phase 5 — The Aristotelian Move

> **Extract "total order" as a first-class concept in the .tam IR, separate from the dataflow pattern.**

Concretely: every accumulate names its `order_strategy` (sequential_left, tree_fixed_fanout_N, segmented_fixed, pairwise_kahan, ...). The verifier rejects `backend_default`. CPU interpreter executes the declared order literally. Each backend implements every declared strategy or refuses the kernel.

**Leverage:**
- Makes I5 derivable from I7 instead of independent.
- Closes the loophole Peak 6 spends a whole peak closing.
- Prevents silent reordering by future backends.
- Gives the Test Oracle a structural check instead of an empirical one.
- Free to do now; expensive format bump later.

**Restated invariant (I7′):**
> Every primitive decomposes into a dataflow pattern (kingdom-classified) and a total order (explicitly named). Dataflow is any kingdom class; total order is first-class in the IR.

### Phase 6 — Recursion on the Phase-5 Move itself

Challenged ten new assumptions the Move was carrying: that "total order" is namable, that an enum is the right representation, that order is separable from dataflow, that fusion survives, that every backend can implement every declared order, etc.

Most of these collapsed the original Move's v1 (simple enum) into v5:

**Move v5 (the refinement):** Open registry of named `OrderStrategy` entries, each shipping with a formal spec, a reference implementation executable by the CPU interpreter, bit-exact test vectors, and fusion-compatibility metadata. Per-kernel default + per-op override. Every backend publishes a **capability matrix** listing which OrderStrategy entries it implements bit-exactly; a kernel declaring a strategy outside a backend's matrix is rejected at compile time, never silently relaxed.

The v1→v5 gap is significant:
- v1 (enum): closed, limited, breaks fusion
- v5 (registry): open, expandable, compatible with fusion via explicit metadata

v5 is the right shape to actually route to IR Architect. v1 was a first draft.

### Phase 7 — Stability check

v5's assumptions (open registry coordination cost, formal spec rigor, capability matrix maintenance, fusion compatibility drift) each have concrete mitigations available through conventional engineering. No new irreducible truths emerged. The deconstruction is **stable at v5.** No further recursion warranted.

### Phase 8 — Forced Rejection (this surfaced the deepest finding)

Forcibly rejected decomposition entirely. Imagined tambear as a **certified kernel library** — opaque kernels, no primitives, no accumulate+gather, no kingdoms. Each kernel individually verified against its spec.

What this reveals:
- Cross-backend bit-exactness IS STILL ACHIEVABLE without any decomposition. The trek's central claim survives.
- TamSession content-addressing still works (outputs are bitwise-hashable).
- BUT: fusion across methods becomes impossible, so the speed story dies. 15 methods sharing a covariance matrix pay 15 passes instead of 1.

**The unseen first principle that Phase 8 surfaced:**

> **Decomposition is a speed story, not a correctness story.** The trek conflates them. I7's language ("every primitive decomposes") sounds like it's about correctness, but its actual load-bearing role is to enable fusion, which is a performance technique. Bit-exactness-cross-hardware is the correctness story, and it's cleanly separable from decomposition.

**Implication:** Peak 1's choice to use accumulate + gather is driven by *performance* (fusion), not by the trek's *meta-goal* (bit-exact cross-hardware). Any IR that pins the op sequence precisely enough satisfies the meta-goal. The trek and Peak 1 share an IR but serve slightly different purposes — separating these cleanly would help the team.

This is the most important finding of the entire deconstruction of I7. It changes how both the trek and the tambear contract should articulate their claims: two benefits on two axes, joined by a single IR design, not one benefit on one axis.

### Surprise

The 10-reconstruction exercise was more productive than expected. The competing skeletons (especially #6 scatter_add + ordered_fold, #8 kingdoms-as-primary, and #9 algebraic structure) aren't just alternatives — they surface what accumulate+gather is *hiding*. #6 hides the order question. #8 hides the non-liftable cases under an escape hatch. #9 hides the algebraic law behind the dataflow. The current choice isn't wrong, but it makes a particular set of things implicit, and the Aristotelian Move is to make exactly *one* of those implicit things explicit — the one that's most load-bearing for the trek.

## Interpretation

I7 survives as a compositional invariant. It does NOT by itself guarantee bit-exactness. The trek currently has two invariants (I5 and I7) that jointly imply the architectural claim, but their relationship is operational (I5 patches I7's blind spot) rather than structural (I7 implies I5). Making total order first-class changes this: I7′ structurally implies determinism for the Kingdom-A case.

This tightens the architectural claim materially. It also reduces Peak 6's scope from "design deterministic reductions" to "implement each declared order_strategy on each backend" — a mechanical task instead of a design task.

## Artifacts

### Phase document
| File | Version | Description |
|---|---|---|
| `campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/accumulate-gather-phases.md` | initial draft | Full eight-phase template, Phases 1–5 drafted |

### Connected invariants
| # | Title | Status after this deconstruction |
|---|---|---|
| I5 | No non-deterministic reductions | Proposed: becomes derivable from I7′ |
| I7 | Every primitive decomposes into accumulate + gather | Proposed: restated as I7′ with explicit total-order axis |

## Open Questions

1. Does "total order" generalize cleanly to Kingdom B (sequential recurrence) and Kingdom C (iterative fixed point)? For B, probably yes (the sequence IS the order). For C, the order includes the convergence criterion and iteration bound — which is a richer object than an enum.
2. Can `order_strategy` be a finite enum or must it be an open set? Likely open. New orderings (compensated summation variants, blocked pairwise, etc.) will keep being invented. The IR should name orderings, not enumerate them.
3. Does making order first-class break composition? E.g., if two kernels each declare an order, and we fuse them, whose order wins? This is where Phase 6 should start.
4. Is Phase-5's move the RIGHT move, or only the *first-found* move? Phases 6–8 are supposed to check for stability. Need to do them before hardening the proposal.
5. What do the other load-bearing invariants look like under the same treatment? I3 (no FMA contraction), I4 (no implicit reordering), I9 (mpmath oracle) all have hidden assumptions that deserve an Aristotelian pass.

## Next

1. Message navigator with the Aristotelian Move. Not as a directive — as a routing request: "this affects IR Architect and Test Oracle; please forward."
2. Pick the second target. Candidates: "Why mpmath as oracle?" (I9), "Why bit-exact cross-hardware?" (the meta-goal), "Why SSA for .tam IR?" (inherited from LLVM dominance).
3. When idle: Phase 6 of this deconstruction.
