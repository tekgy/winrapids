# Lab Notebook 013: Aristotelian deconstruction of the bit-exact meta-goal

**Date**: 2026-04-11
**Authors**: Aristotle
**Branch**: main
**Status**: Complete — Phases 1–8 drafted; stable at Move v5; Phase 8 surfaced compositional-claim framing + three-precondition structure
**Depends on**: Notebooks 011, 012; trek README Part I; invariants.md I1–I10

## Context & Motivation

Notebooks 011 and 012 attacked specific invariants (I7, I9). This notebook attacks the **meta-goal** itself: the claim that "bit-exact cross-hardware" is the right target for the trek. If bit-exactness is overreach, every peak is more expensive than it needs to be. If it's exactly right, the target gets stronger by surviving the challenge. Either way, the trek benefits from the deconstruction happening now rather than after Peak 6 exists to close the determinism gap.

The trek's central claim:
> "One compiled `.tam` kernel, the same source of math, the same numerical answers, running on any ALU through only its driver, with no vendor compiler and no vendor math library anywhere in the path."

The load-bearing phrase is "the same numerical answers" — interpreted as bit-exact. The alternative is bounded-ULP.

## Hypothesis

**H0 (project assumption):** Bit-exact cross-hardware is the right target because (a) it's the strongest guarantee, (b) achieving it forces the right architecture, and (c) it makes testing simple (`==`).

**H1 (Aristotle's counter):** Bit-exactness is a means to specific ends (equality testing, content-addressing, cross-machine caching, audit reproducibility) — it is NOT an intrinsic architectural property. The trek currently frames bit-exact as the GOAL, but its actual value is in enabling downstream architecture like the TamSession persistent store. That means (1) the trek's framing under-explains WHY bit-exact, and (2) under pressure the team may not know which trade-offs preserve the real goal. Making the "why" explicit is high leverage with zero code cost.

**Prediction:** Phase 4's collision map will find that "bit-exact is intrinsically desirable" is under-justified, and that the real justification is "bit-exact enables content-addressing of intermediates in TamSession + cross-machine cache hits + auditable equality tests." The Aristotelian Move will be a trek-plan update (not an invariant change) that makes the spectrum of cross-hardware guarantees and the reason for picking bit-exact explicit.

## Design

Same eight-phase template. Working document:
```
campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/bit-exact-vs-bounded-ulp-phases.md
```

## Design Decisions (what we chose AND rejected)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Third target | "Bit-exact vs bounded-ULP cross-hardware" | "Why SSA for .tam IR?", "Why f64?" | These are implementation choices under the meta-goal. The meta-goal has higher leverage — challenging it tests the whole trek, while challenging SSA tests only one IR. |
| Posture | Question the target but not necessarily weaken it | Recommend scrapping bit-exact for bounded-ULP | The point isn't to weaken the trek. The point is to check whether the stated goal survives first-principles — and if it does, to articulate WHY cleanly enough that the team can defend it under pressure. Bit-exact passing the challenge is a stronger trek than bit-exact unchallenged. |
| Shape of the Move | Framing change in trek-plan.md, not an invariant change | Demote bit-exact to "optional mode", or weaken I-invariants | A framing change is cheap (one document), load-bearing (changes how the team defends the goal), and reversible. An invariant change is expensive, fights the team's existing commitment, and risks actually weakening the guarantee. |
| Explicit future-work hatch | Name bounded-ULP modes as future work, not alternatives rejected | Declare bit-exact the only mode forever | Leaving the door open for fast modes prevents architectural lock-in. Some operations (stochastic, iterative fixed-point) may never be bit-exact in practice — calling those "violations" is false. |

## Results

### Phase 1 — 10 assumptions inside "bit-exact cross-hardware is the right target"

The deepest ones:
- That users need bit-exactness (vs consistency + small bounded error).
- That "any ALU" works for non-IEEE-754 hardware (posits, approx-compute, no-denormal silicon).
- That bit-exactness implies determinism — it does, in one direction only. Bit-exact → deterministic; the reverse (deterministic per-backend) is cheaper and some users are satisfied by it.
- That the claim is *the* selling point of the library (it's one of several — see project memory).

### Phase 2 — 10 irreducible truths

The load-bearing ones:
- T4: Two implementations executing the same sequence of fp64 ops in the same order produce the same answer on any IEEE-754 compliant hardware. (This is the foundation of the trek's feasibility.)
- T5: Cross-hardware divergence is a compiler/runtime problem, not a hardware problem.
- T6: Bit-exact cross-hardware = every backend compiles to an identical op sequence.
- T7: Producing identical sequences requires an IR precise enough to pin the sequence (reduction order, contraction, rounding).
- T8: Bounded-ULP with ε=0 IS bit-exact; the real distinction is whether ε>0 is permitted.
- T9: **The value of bit-exact to a user is the ability to test for equality — enabling content-addressing, hashing, cross-machine caching, audit reproducibility.** These are the irreplaceable use cases.
- T10: Bit-exactness does NOT imply correctness. The two are separate trust layers.

### Phase 3 — 10 reconstructions

1. No cross-hardware claim.
2. Bounded-ULP (loose, 10 ULPs).
3. Bounded-ULP (tight, 1 ULP).
4. Bit-exact for arithmetic, bounded-ULP for transcendentals.
5. Bit-exact on a declared profile of kernels; weaker elsewhere.
6. Bit-exact as default, opt-out to fast modes.
7. Bit-exact as THE target (current trek).
8. Content-addressing as the real goal; bit-exactness as one means.
9. Formal-methods proof of bit-exact.
10. Hybrid: bit-exact by default, bounded-ULP where workload sensitivity permits.

### Phase 4 — Collisions

The deepest collision: **"bit-exactness is a natural universal target" vs "bit-exactness enables a specific set of downstream properties."** The trek currently frames it as the first (universal); Phase 2 truth 9 establishes only the second (downstream enablement).

The trek's claim is actually valid — but the *rationale* is under-specified. The real argument for bit-exact is:

1. The TamSession persistent store needs content-addressing of intermediates.
2. Cross-machine cache hits (a major performance win for distributed teams) need bit-exact matching.
3. Audit-bearing users (regulated, financial, research reproducibility) need equality tests.
4. All three require bit-exact, and bounded-ULP would break all three.

That's a defensible argument. The trek plan doesn't currently state it.

### Phase 5 — The Aristotelian Move

**Add a "Guarantee Spectrum" section to trek-plan.md that explicitly states:**
- The spectrum of possible cross-hardware guarantees (from "no claim" to "fully bit-exact").
- Which user categories each point on the spectrum serves.
- Why the trek targets bit-exact (content-addressing, cross-machine caching, audit reproducibility).
- Which weaker modes are future work (not alternatives rejected).
- Which ops may never be bit-exact in practice (stochastic, iterative fixed-point) and how they'd be handled.

**NO invariant changes. NO code changes. Just framing.** One-page addition between Part I (what we're building) and Part II (invariants). The effect is on the team's *defense* of the trek under pressure — when someone says "let's relax I3 for this one kernel," the framing shows what's actually being given up (content-addressing, not just "some bits").

### Surprise

I expected to find that bit-exactness was overreach. Instead Phase 2 truth 9 surfaced that bit-exactness has *a specific, irreplaceable downstream value* that bounded-ULP can't provide, and the trek just doesn't currently say so. The move became "articulate the why" rather than "question the what." That's a less dramatic finding but arguably a more useful one — the trek was directionally right, just under-explained. Sometimes Aristotle's job is to confirm the target and articulate it better, not to redirect.

Second surprise: while drafting Phase 1, I noticed that Peak 6's determinism work is actually closing a gap left by I7, not producing bit-exactness directly. Bit-exact cross-hardware requires the *same order strategy* in every backend, and I7 doesn't mandate that — it's the I7′ move from Notebook 011. So the three moves (I7′, I9′, and the trek-plan framing) are actually one coherent story: **the trek's central claim is correct, but the invariants jointly imply it only after I7′ and I9′ are adopted; and the framing makes the user-facing value of the claim defensible.**

## Interpretation

Bit-exact cross-hardware is the right target. The trek should NOT weaken it. What's missing is two invariant upgrades (I7′ from Notebook 011, I9′ from Notebook 012) and a framing section in trek-plan.md that names the downstream value. Together they tighten the trek into a coherent, defensible architectural claim.

## Phase 6 — Recursion on the Move

Ten new assumptions surfaced inside v1:
- That content-addressing is a deep enough reason (it's project-internal — circular)
- That the spectrum is linearly ordered (it isn't — user needs are multi-dimensional)
- That framing changes are effective without code (necessary but not sufficient)
- That the trek plan is the right document (the story concerns the whole project)
- That "future work" for weaker modes is a real door
- That user categories are distinguishable (they overlap)
- That TamSession actually uses bitwise hashing (flagged as unverified)
- That bit-exactness alone captures the architectural commitment (it doesn't — decomposition/speed is the other axis)
- That the framing costs the team nothing to maintain
- That bit-exact is what every subsystem of tambear wants (`.discover()` superposition may not care)

v1 → v5 refinement:

**Move v5:**
> Add a **Guarantee Ledger** to the trek and reference it from the project's top-level architectural doc. The ledger maps each invariant to the specific user-facing property it protects and the exact cost of relaxing it. It covers both axes (correctness via bit-exact, speed via fusion/decomposition). The cause is named as **IR precision** (the IR pins the exact op sequence on any backend); bit-exact cross-hardware is a downstream consequence. Future-work modes (bounded-ULP, fp32, approximate) have named sentinel tests that would fail the moment they become necessary. Any invariant-relaxation proposal must come with a filled-in ledger row — "cost of relaxation" column required — and reviewers use the ledger as the mandatory checklist.

### Phase 7 — Stability

v5 residuals (ledger maintainability, cost-of-relaxation column fillability, post-hoc justification risk, IR-precision well-definedness) each have concrete mitigations. **Stable at v5.**

### Phase 8 — Forced Rejection + the deepest finding

Three alternative framings considered:
1. **Cross-time instead of cross-hardware** — weaker claim, easier trek, less distinctive
2. **Cross-IR instead of cross-hardware** — actually the SAME as current trek, clearer name
3. **No cross-hardware at all** — dramatically easier, loses the distinctive claim

Alternative 2 is the key finding: **"cross-hardware" is a CONSEQUENCE name; the CAUSE is "cross-IR" — i.e., the IR pins the op sequence and every faithful lowering produces the same bits.**

**The unseen first principle:**

> **The trek's claim is a COMPOSITIONAL guarantee, not a unilateral one.** It holds when three conditions hold simultaneously: **IR-precision**, **faithful-lowering**, and **IEEE-754-compliance-for-the-ops-used**. Any of the three failing breaks the claim. The trek currently names only the IR-precision side — the part we control. The other two are implicit preconditions.

**ESC-001 (Vulkan subnormal flushing) is a violation of precondition 3.** Under the compositional framing, navigator's Option 2 decision (narrow claim to normal fp64 with subnormal as a device prerequisite) becomes the canonical form — not a retreat, just an honest statement of precondition 3 for subnormal-producing ops.

### Move v5 (Phase-8 refined, final)

The Guarantee Ledger's top-level structure is the three preconditions. Each invariant is classified as protecting one precondition. The central claim is restated as:

> "Given .tam source, a backend that faithfully lowers .tam to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs. Tambear owns the first condition via the IR; the second via our backend implementations; the third is a declared hardware prerequisite documented per device."

## The cross-target pattern

All three Aristotle Moves have now converged on the same engineering pattern: a **named registry/ledger of formal-spec entries with capability metadata, review-time enforcement, and clear ownership**.

| Move | Artifact | Owner |
|---|---|---|
| I7′ v5 | `order_strategies/` registry | IR Architect |
| I9′ v4 | `oracles/` registry | Adversarial Mathematician + Test Oracle |
| Meta-goal v5 | `guarantees/` ledger | Navigator |

Three artifacts. Same shape. Same lifecycle. This is the pattern that emerges when tacit architectural knowledge is made explicit. I suspect it's the right shape for the whole project's documentation layer, not just the trek.

## Artifacts

| File | Description |
|---|---|
| `campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/bit-exact-vs-bounded-ulp-phases.md` | Phases 1–8 stable at v5 |

## Open Questions

1. Should the trek plan update include a short table of "invariant → downstream property enabled" so the team can trace any invariant relaxation to a concrete user-facing cost?
2. Does the TamSession persistent store currently work correctly if backends disagree by 1 ULP? If YES, the content-addressing argument is weaker (bit-exact is overreach for TamSession). If NO, the argument is airtight.
3. Which ops on the trek's current Phase-1 recipe list (sum, mean, variance, pearson_r, etc.) are easy to bit-exact, and which are hard? The hard ones are where framing pressure will hit first.
4. Is there a class of users who specifically want *bit-exact-on-one-backend, bounded-ULP-across-backends*? If so, they're well-served by the current Peak 6 work and don't need cross-backend bit-exact at all. This is worth checking.

## Next

1. Message navigator with the third move. Framing change only — no code.
2. Phase 6 on one of the three targets. Probably 011 first (the I7′ move is blocking peak-1 design choices).
3. Fourth target — candidates narrowing: "Why f64?" (precision choice), "Why SSA?" (IR form), "Why decompose at all?" (notebook 011's forced-rejection probe).
