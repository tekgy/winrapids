# Aristotle Synthesis — Three Deconstructions of the Bit-Exact Trek

**Author:** Aristotle
**Date:** 2026-04-12
**Purpose:** Single-page reference for the three Aristotelian deconstructions of the trek's foundations. Read this instead of the three phase documents when you need the gist of the whole arc. Read the phase documents when you need the reasoning.

---

## The three targets

Over the opening days of the Bit-Exact Trek, I ran a first-principles deconstruction on each of three foundational assumptions the trek rests on:

| Target | Invariant / meta-claim | Status |
|---|---|---|
| 1. Why accumulate + gather? | I7 | Phases 1–8 stable at Move v5.1 |
| 2. Why mpmath as oracle? | I9 | Phases 1–8 stable at Move v4 |
| 3. Why bit-exact cross-hardware? | meta-goal | Phases 1–8 stable at Move v5 |

All three targets **survive** deconstruction. None of them is overreach. What's missing is structural support the trek didn't originally have, which the three Moves provide.

---

## The three Moves in one sentence each

1. **I7′ (v5.1):** Every `.tam` accumulate declares its order strategy AND its special-value semantics (NaN/Inf/signed-zero/subnormal) by reference to an **open `order_strategies/` registry**. Backends publish capability matrices; non-supported strategies are rejected at compile time.

2. **I9′ (v4):** Every libm function publishes a **two-column correctness profile** (TESTED and CLAIMED) against a declared minimum oracle suite. Oracles live in an **`oracles/` registry** with independence-matrix metadata. The profile is an *auditability contract* for user classes, not a single correctness claim.

3. **Meta-goal (v5):** Add a **`guarantees/` ledger** to the trek. Its top-level structure is the **three preconditions** of the compositional claim — IR-precision, faithful-lowering, IEEE-754-compliance-for-the-ops-used. Each invariant (I1–I10 plus I7′ and I9′) is classified as protecting one precondition. The ledger is the mandatory checklist for any invariant-relaxation proposal.

---

## Why each Move matters (the engineering story)

### I7′ — because total order was an implicit contract

**Original I7:** Every primitive decomposes into accumulate + gather.

**The gap:** accumulate's `op` parameter implicitly assumes associativity, which hides the ordering question. Different backends can pick different reduction orders (sequential, tree, pairwise, segmented), produce different bit patterns, and both "comply with I7" while diverging.

**What I7′ changes:** Order is named. The IR declares which strategy; the backend implements it or refuses the kernel. Peak 6's determinism work becomes mechanical instead of a design task.

**v5.1 addition (from adversarial's feedback, 2026-04-12):** Special-value semantics (NaN/Inf/subnormal) are ALSO named per strategy. This closed a gap my Phase 2 missed — adversarial found it empirically via tbs::eval bugs (P17, P18, P20) where NaN propagation was argument-order-dependent.

### I9′ — because the oracle tests accuracy, not correctness

**Original I9:** mpmath at 50 digits is the oracle.

**The gap:** pointwise-vs-mpmath tests accuracy at test points. It misses monotonicity failures at regime transitions, Table Maker's Dilemma cases (50 digits is insufficient near fp64 midpoints), identity violations, and bugs that competitors also have (since competitors also compare pointwise against arbitrary-precision references).

**What I9′ changes:** Multiple oracle types run against each function. Each tests a different property. Results are published as a two-column profile separating what was TESTED from what is CLAIMED in the function's contract. The trek's "bit-perfect or bug-finding in competitors" promise becomes defensible: we run oracles competitors don't.

**Phase 8 reframe:** The profile's *purpose* is to be an **auditability contract** — a structured communication channel for users who can't audit the source directly. Different user classes (research, production, regulated audit, composition consumers) need different contents from this channel.

**Empirical validation (adversarial, 2026-04-12, commit 35982d5):** P19 in the pitfall journal found that cross-backend agreement doesn't imply correctness — two wrong answers agreeing. This is the same finding as I9′ Phase 4's collision map, told from the empirical side.

### Meta-goal v5 — because the claim was framed unilaterally

**Original framing:** "One .tam kernel, the same numerical answers, any ALU." Stated as THE architectural claim.

**The gap:** bit-exactness is a CONSEQUENCE of three conditions holding simultaneously. The trek only names the first. Under pressure, "the invariant says so" is a weak defense against relaxation.

**What the Move changes:** The ledger makes the three preconditions explicit and each invariant maps to one. The central claim is restated as a compositional guarantee:

> "Given .tam source, a backend that faithfully lowers .tam to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs."

**Empirical validation (ESC-001, naturalist + scout, pre-Peak-7):** Vulkan's `shaderDenormPreserveFloat64` = false on the RTX 6000 Pro. Subnormal fp64 is implementation-defined. Under v5, this is not a retreat — it's the compositional claim's precondition 3 being honest about what ops the hardware supports for bit-exact execution. Navigator's Option 2 decision (narrow claim to normal fp64, document subnormal as device prerequisite) is the canonical form of the claim under v5.

---

## The cross-target pattern — three independent Moves converged on the same shape

All three Moves landed on **named registries/ledgers of formal-spec entries with capability metadata, review-time enforcement, and clear ownership**:

| Move | Artifact | Owner |
|---|---|---|
| I7′ v5.1 | `order_strategies/` registry | IR Architect |
| I9′ v4 | `oracles/` registry (+ corpora) | Adversarial Mathematician (corpora) + Test Oracle (runner) |
| Meta-goal v5 | `guarantees/` ledger | Navigator |

Each artifact has the same lifecycle: entries added through review, spec + reference implementation required, consulted during escalation, referenced in invariant-relaxation discussions. The convergence is not coincidence — this is the engineering pattern that emerges when tacit architectural knowledge needs to become explicit and inspectable. I suspect a fourth registry will appear naturally (for kingdoms, for device capabilities, for ops) and should follow the same pattern.

---

## The unseen first principles surfaced by forced rejection

Phase 8 (the forced-rejection pass) surfaced one non-obvious structural finding per target. These aren't in the original trek framing and are worth naming explicitly:

1. **Decomposition is a SPEED story, not a CORRECTNESS story.** Bit-exactness cross-hardware is achievable without accumulate+gather; what dies without decomposition is fusion (and therefore the "15 methods share one pass" performance story). Peak 1's IR serves two masters (correctness + speed), and the trek conflates them. Naming them separately helps the team defend each axis under pressure.

2. **Oracles are for USERS, not for TRUTH.** Truth is the source code. Oracles are a structured communication channel for users who can't audit directly. The multi-oracle profile serves different user classes with different contents — research users need source + summary, production users need bounds + regressions, regulated users need independence + versioning, composition consumers need composition notes.

3. **The trek's claim is COMPOSITIONAL, not unilateral.** It holds when IR-precision AND faithful-lowering AND IEEE-754-compliance-for-the-ops-used ALL hold. Any of the three failing breaks it. ESC-001 is exactly a precondition-3 violation, not a claim failure.

Together, these reshape how the trek should describe itself:

> **Tambear's bit-exact cross-hardware guarantee is a compositional claim, held by three preconditions (IR-precision, faithful-lowering, IEEE-754-compliance-for-ops-used), each protected by specific invariants and documented in the Guarantee Ledger. The guarantee serves a specific downstream purpose — enabling content-addressing of intermediates, cross-machine cache hits, and audit reproducibility — that weaker modes would break. Tambear's separate performance guarantee (fusion of compositions into minimum-pass kernels) rides on the decomposition of primitives into accumulate + gather, which is an independent axis from the correctness guarantee. Both axes are load-bearing, and both are named explicitly.**

That's the tightened version of the trek's story. Three sentences instead of one, but each sentence is defensible.

---

## How to use this synthesis

**During design reviews:** Check whether the proposed change interacts with any of the three Moves. Does it add an order strategy? Add it to the `order_strategies/` registry. Does it add an oracle? Add it to the `oracles/` registry. Does it relax an invariant? Cite the Guarantee Ledger row and fill in the "cost of relaxation" column.

**During escalations:** Map the escalation to one of the three preconditions from the meta-goal Move. If it's IR-precision (the IR doesn't say enough), the fix is at Peak 1. If it's faithful-lowering (a backend reinterprets), the fix is at Peak 3/5/7. If it's IEEE-754-compliance-for-ops-used (hardware doesn't support what we need), the fix is either a device prerequisite (ESC-001 pattern) or using an alternative op.

**During libm implementation:** Before running a test against mpmath, run closed-form specials + identity + monotonicity. If any of those fail, the pointwise test won't catch it. Record the profile with both TESTED and CLAIMED columns.

**During bug triage:** For numerical bugs, first distinguish: (a) the implementation disagrees with mathematics (true bug, fix the math), (b) the backend disagrees with the reference (precondition-2 violation, fix the lowering), (c) the hardware disagrees with IEEE-754 for an op we use (precondition-3 violation, narrow the claim or require hardware feature). Different fix sites, different owners.

---

## Cross-references

| Target | Phase document | Notebook |
|---|---|---|
| 1. accumulate + gather | `peak-aristotle/accumulate-gather-phases.md` | `research/notebooks/011-aristotle-accumulate-gather-deconstruction.md` |
| 2. mpmath oracle | `peak-aristotle/mpmath-oracle-phases.md` | `research/notebooks/012-aristotle-mpmath-oracle-deconstruction.md` |
| 3. bit-exact meta-goal | `peak-aristotle/bit-exact-vs-bounded-ulp-phases.md` | `research/notebooks/013-aristotle-bit-exact-vs-bounded-ulp.md` |

Tasks: #8 (complete), #9 (complete), #10 (complete).

Pitfall journal entries that validate the deconstructions empirically: ESC-001 (meta-goal precondition 3), P17/P18/P20 (I7′ special-value gap), P19 (I9′ cross-backend-agreement-is-not-correctness).

---

## What this synthesis is NOT

- NOT a replacement for the phase documents. Read them when you need the reasoning.
- NOT a directive. Every Move is proposed; adoption is navigator's call.
- NOT a claim that the three targets exhaust what needs deconstructing. Fourth candidates include: "Why f64?", "Why SSA?", "Why Cody-Waite for transcendentals?", "What is the exception/special-value semantics surface area across all ops?" (the last one is a direct follow-on from the v5.1 addendum and would probably yield the next set of Moves).

---

**Status as of 2026-04-12 (late update — adoption state):**

- All three deconstructions complete, all three Moves stable.
- **I7 adopted as refined invariant.** `invariants.md` was updated 2026-04-12 to state: "Every primitive is described by a (dataflow pattern, total order) pair. Accumulate+gather is the most common dataflow pattern (Kingdom A). Total order is referenced by name into an open OrderStrategy registry." Forbids: "New ops added to .tam IR that don't declare their order, OR kernels declaring an order outside a backend's capability matrix (→ compile-time reject, never silent relax)." Why: "Decomposition enables **speed** via fusion; total order enables **correctness** via bit-exactness. Conflating them was the original trap. Refined per Aristotle I7 Phase 8 finding (2026-04-12)." Peak 1 campsites 1.16-1.17 (OrderStrategy registry) are in progress under the new I7. **The primary win is canon.**
- **I11 added as a new invariant.** The v5.1 Phase 2 gap (exception/special-value semantics as a fifth role, surfaced by adversarial's bugs 3-5 and credited back to me via the addendum) has been promoted to a first-class invariant: "NaN propagates through every op on every backend." The `invariants.md` entry credits adversarial's bugs 3-5 and "independent convergence by naturalist + scout." It's an invariant now, not a pitfall-journal entry.
- **I9 not yet refined.** Invariant text unchanged — still "mpmath is the oracle." The I9′ Move (TESTED/CLAIMED profile, multi-oracle suite, `oracles/` registry, auditability-contract reframe) is on navigator's plate alongside Peak 2. Navigator has sharpened the framing (TMD as the core structural collision) but the invariant text itself will probably update once campsite 2.3 (ULP harness) actually needs it. That's the right pace — refine the invariant when the code that enforces it exists.
- **Meta-goal compositional claim adopted in README.md.** The expedition README was amended 2026-04-12 with the exact compositional-claim wording from notebook 013's Phase 8: "The claim is compositional: given `.tam` source, a backend that faithfully lowers `.tam` to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs." The ESC-001 scope amendment (bit-exact for normal fp64; subnormal requires `shaderDenormPreserveFloat64 = true`) is framed as a precondition-3 honest scope rather than a retreat, per the phase document's analysis. The README also references `trek-plan.md` Part II.5 for the full framing, which means the Guarantee Spectrum section has been drafted under navigator ownership. Both the Move and the framing have landed as authoritative text.
- **Cross-target adoption rate: 2 of 3 Moves have landed as invariant-level text within 24 hours of routing.** I7′ and NaN (as I11). That's a faster adoption cycle than I expected. The remaining Move (I9′) is the one that requires implementation before the invariant text makes sense, which is correct engineering pacing.
- Adversarial has empirically validated findings from two of the three targets (I9′ via P19, I7′/I11 via P17/P18/P20).
- Aristotle is idle per the "when idle, follow curiosity" clause. Next work: probably a libm-adjacent target (Cody-Waite range reduction, f64 as base precision) to support the Peak 2 critical path, pending navigator preference check.
