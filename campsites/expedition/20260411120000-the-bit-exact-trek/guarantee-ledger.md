# Guarantee Ledger

**The mandatory checklist for invariant-relaxation proposals and the reference for design-time reasoning.**

This ledger maps every trek invariant (I1–I11) to (a) the precondition of the compositional claim that it protects, (b) the exact cost of relaxing it, and (c) the user-facing property it guards. When someone proposes relaxing an invariant — in a PR, a campsite design doc, or an escalation — they must cite the relevant row here and name what they are trading away.

Drafted 2026-04-12 per Aristotle meta-goal deconstruction (Move v5) and navigator routing. First-class artifact of the trek, alongside `invariants.md` and `trek-plan.md` Part II.5.

---

## The compositional claim and its three preconditions

From `README.md` and `trek-plan.md` Part II.5:

> Given `.tam` source, a backend that faithfully lowers `.tam` to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs.

The claim holds when all three preconditions hold. Any single precondition failing breaks the claim.

| # | Precondition | What it asserts | Who owns it |
|---|---|---|---|
| **P1** | **IR-precision** | The `.tam` IR is precise enough to pin the exact op sequence — every op, every operand, every order, every special-value rule, every rounding mode, every semantic constraint. Two distinct runs of the same `.tam` source mean the same computation, unambiguously. | Tambear (IR Architect) |
| **P2** | **Faithful-lowering** | Every backend compiles `.tam` → target ISA without introducing new ops, reordering existing ops, contracting to fused instructions, or applying optimization passes that alter bit-level semantics. The lowering is a homomorphism, not a rewrite. | Tambear (backend implementers: CPU, PTX, SPIR-V, ...) |
| **P3** | **IEEE-754-compliance-for-ops-used** | The hardware implements IEEE-754 semantics for every op the kernel uses — including rounding modes, special values (NaN, Inf, zero), subnormals where applicable, and every other corner of the IEEE-754 spec that the op depends on. | Hardware vendor; Tambear declares the requirement and gates at compile time via capability matrix. |

Each invariant below primarily protects one precondition. Some invariants have cross-references where their effect spans preconditions.

**I9 and I10 are meta-invariants** — they govern how we *verify* the other invariants, not how we *protect* them. They live in a separate section.

---

## Primary ledger: invariants by precondition

### Protects P1 (IR-precision)

#### I3 — No FMA contraction
| | |
|---|---|
| **Protects** | P1 — IR-precision (and P2 at the lowering boundary) |
| **Cost of relaxation** | The IR can no longer distinguish "explicit FMA" from "fmul-then-fadd." Every kernel that used the latter for numerical reasons (Kahan summation, compensated arithmetic) may be silently contracted by a backend, producing different bits than the reference. Downstream consumers hashing the output get a cache miss at best, a correctness gap at worst. |
| **User-facing property guarded** | "The same `.tam` source produces the same numerical answer regardless of which backend ran it." Relaxing this means the user's reference result (computed on CPU) disagrees with their production result (computed on GPU) in the last few bits — silently, without user intervention. |
| **Cross-reference** | Enforced at lowering time (`.contract false` in PTX, `NoContraction` in SPIR-V). The IR-level protection is the ABILITY to name the non-contracted form; the lowering-level protection is the DISCIPLINE to emit it. |

#### I7 (refined) — Every primitive is a (dataflow pattern, total order) pair, referenced by name into an open OrderStrategy registry
| | |
|---|---|
| **Protects** | P1 — IR-precision |
| **Cost of relaxation** | The IR loses the ability to pin reduction order. Two accumulators with the same inputs can produce different bits depending on which backend's default reduction the compiler chose. Cross-backend bit-exactness collapses for any kernel containing a reduction — which is almost every Phase 1 kernel. |
| **User-facing property guarded** | "A variance, a sum, or any reduction kernel produces identical bits on CPU and GPU." Relaxing this means the same input to the same recipe produces different outputs on different machines, making content-addressing and cross-machine cache hits impossible for any kernel touched by the relaxation. |
| **Cross-reference** | The dataflow side of I7 also enables the SPEED story (fusion). The total-order side is what protects P1. These are separable — refining the invariant to two parts (the project's 2026-04-12 refinement) is what made the compositional claim enforceable. |

#### I8 — First-principles only for transcendentals
| | |
|---|---|
| **Protects** | P1 — IR-precision |
| **Cost of relaxation** | A libm function imported from glibc/musl/sun-libm carries implementation-defined rounding choices, subnormal handling, and edge-case behavior that the `.tam` IR does not specify. The IR's semantics diverge from the implementation's actual behavior. Users reasoning about the IR see one function; users running the code get another. |
| **User-facing property guarded** | "The `.tam` definition of `tam_exp` IS the function `tam_exp` computes, full stop." Relaxing this means the IR is a suggestion, and the actual function is whatever the ported code implements — which is usually close, but not identical, and varies across ported sources. |
| **Cross-reference** | Relates to I9′ (when refined): ported implementations cannot be verified against the IR because there's no IR-level specification to compare against. The auditability chain collapses. |

#### I11 — NaN propagates through every op on every backend
| | |
|---|---|
| **Protects** | P1 — IR-precision (enforced at P2 + P3 via capability matrix) |
| **Cost of relaxation** | The IR's semantics for NaN become implementation-defined. `min(NaN, x)` returns NaN on one backend, `x` on another. `sum([1.0, NaN, 2.0])` is NaN on one backend, `3.0` on another (if the backend silently drops NaN values). The bit-exact claim fails on any input containing a NaN anywhere in its processing chain, which includes any user workload where a division-by-zero or domain violation has ever happened upstream. |
| **User-facing property guarded** | "NaN is a sentinel. If NaN appears anywhere in an input, it appears in the output, regardless of backend." Relaxing this means NaN silently disappears on some backends, hiding upstream errors and producing misleading "clean" results on data that had bad values. |
| **Cross-reference** | Promoted to invariant 2026-04-12 after adversarial bugs P17/P18/P20 plus independent convergence by naturalist + scout. The Aristotle I7 Phase 2 gap (exception/special-value semantics as a fifth computation role) is what connects this to I7's framing. |

---

### Protects P2 (Faithful-lowering)

#### I1 — No vendor math library in any path
| | |
|---|---|
| **Protects** | P2 — Faithful-lowering |
| **Cost of relaxation** | The `.tam` op `tam_sin` gets lowered to `__nv_sin` on CUDA, `metal::sin` on Metal, `glibc::sin` on Linux CPU. Each is a different function. The lowering is no longer a homomorphism — the same IR op means different things on different backends. Cross-backend agreement fails structurally. |
| **User-facing property guarded** | "Every backend computes the same `tam_sin` because they're running the same implementation from our libm." Relaxing this means `tam_sin(x)` is backend-dependent, and the user has to know which backend ran to interpret the result. |

#### I2 — No vendor source compiler in any path
| | |
|---|---|
| **Protects** | P2 — Faithful-lowering |
| **Cost of relaxation** | NVRTC, DXC, or similar compile our C-string dialect with optimizations we don't control. The emitted PTX/SPIR-V has been rewritten by a compiler whose rules we don't own. The output is no longer a faithful lowering — it's a *vendor's* lowering, and vendors disagree. |
| **User-facing property guarded** | "We own the instruction sequence every backend executes." Relaxing this means we cede control of numerical behavior to a vendor's compiler team, whose priorities (performance, compatibility) don't include cross-backend bit-exactness. |

#### I4 — No implicit reordering of floating-point operations
| | |
|---|---|
| **Protects** | P2 — Faithful-lowering |
| **Cost of relaxation** | A backend using `-ffast-math`, `/fp:fast`, or any compiler-associativity flag will reorder `(a + b) + c` to `a + (b + c)` when it thinks this is faster. Floating-point addition isn't associative, so the bits differ. The lowering is no longer faithful to the IR's explicit left-fold semantics. |
| **User-facing property guarded** | "The order in which `.tam` says to add things is the order the backend adds them." Relaxing this means any numerically-sensitive computation (variance, stable sums, compensated arithmetic) silently gets scrambled. |

#### I5 — No non-deterministic reductions
| | |
|---|---|
| **Protects** | P2 — Faithful-lowering |
| **Cost of relaxation** | `atomicAdd` for user-visible values means the reduction order depends on the runtime scheduling of GPU threads — which changes run to run, even on the same hardware. "Same input, same bits" fails on the same machine, let alone cross-backend. |
| **User-facing property guarded** | "Running the same kernel twice produces the same answer." Relaxing this means any audit trail, any reproducibility claim, any content-addressed cache breaks on reductions. |
| **Cross-reference** | Under the refined I7, I5 becomes *derivative*: if every reduction declares its OrderStrategy, and backends implement strategies deterministically, I5's guarantee falls out structurally from I7 + the capability matrix. I5 remains as an explicit invariant because the enforcement (no atomicAdd) is a concrete lowering discipline, not an IR constraint. |

---

### Protects P3 (IEEE-754-compliance-for-ops-used)

#### I6 — No silent fallback when a target is missing a feature
| | |
|---|---|
| **Protects** | P3 — IEEE-754-compliance-for-ops-used |
| **Cost of relaxation** | A kernel declaring `fp64` runs on hardware without fp64 support, and the backend silently downgrades to fp32. The user did not ask for precision loss; they got it anyway. The "same bits" claim fails by many orders of magnitude, not just a few ULPs. |
| **User-facing property guarded** | "If you ask for fp64, you either get fp64 or you get a clear error." Relaxing this means precision degradation is silent and user-invisible, and users who don't check for it produce subtly wrong numerical results they can't trace. |
| **Cross-reference** | The ESC-001 decision (Vulkan subnormal flush, 2026-04-12) is the canonical form of I6 in the compositional framing: the subnormal-handling is a precondition-3 requirement, named as `shaderDenormPreserveFloat64` in the capability matrix, and the compiler rejects kernels that depend on subnormal semantics when the backend can't provide them. Not a retreat — honest scoping. |

---

## Meta-invariants: governing verification, not protection

### I9 — mpmath (or equivalent arbitrary-precision reference) is the oracle

| | |
|---|---|
| **Role** | Meta — verifies P1, P2, P3 independently |
| **Cost of relaxation** | Without an arbitrary-precision reference, we cannot verify that our libm matches mathematics (P1 verification), we cannot verify that different backends agree (P2 verification), and we cannot verify that hardware behaves as IEEE-754 demands (P3 verification). "Another libm" as oracle lets shared bugs go undetected. |
| **User-facing property guarded** | "Our published accuracy claims are verified against mathematics, not against another implementation with its own bugs." |
| **Note on refinement pending** | The I9′ Move (multi-oracle suite with TESTED/CLAIMED profile per function, `oracles/` registry, independence matrix) is accepted by navigator and awaiting Peak 2 campsite 2.3 implementation. When it lands, this ledger row updates to cite the full suite including the TMD-aware corpus (the single highest-value addition to pointwise-mpmath, per navigator's sharpened framing). |

### I10 — Cross-backend diff is continuous, not a final audit

| | |
|---|---|
| **Role** | Meta — governs verification timing |
| **Cost of relaxation** | Deferring cross-backend verification to the end of the trek means drift accumulates. A bug introduced at Peak 3 is found at Peak 7, when fixing it requires unwinding every peak built on top. Continuous verification catches drift at step 3 when it costs one commit; final verification catches it at step 7 when it costs a retrospective across four peaks. |
| **User-facing property guarded** | "Cross-backend equivalence is maintained at every commit, not promised at the end." |

---

## How to use this ledger

### At PR review time

If a PR touches any of I1–I11, the reviewer opens this ledger and asks:
1. Does the PR relax the invariant? (If no: nothing to check.)
2. If yes, does the PR author cite the relevant ledger row and fill in what's being traded away? (If no: request the citation before merging.)
3. Is the trade-off named in user-facing terms? (If the PR says "minor perf win" without naming the cost, the ledger entry's "user-facing property guarded" column is the thing to cite back at them.)

### At escalation time

When writing an escalation in `navigator/escalations.md`, the proposer names which invariant is in tension and cites this ledger's row for that invariant. The "cost of relaxation" column IS the input to the escalation's "options considered" section.

See ESC-001 for the canonical pattern: precondition-3 (IEEE-754 subnormal) is acknowledged as a hardware limitation, and the decision narrows the claim to the scope where the precondition holds, rather than relaxing the invariant. This is the ledger's intended use.

### At design time

When designing a new op, a new backend, or a new campsite, consult the ledger for each invariant that might be touched. The ledger row tells you what that invariant is for, which makes it easy to decide whether the proposed design preserves or threatens it.

Example: a pathmaker adding a new fused reduction op asks "does this touch I5?" The ledger row for I5 says: "same input, same bits; relaxation means reproducibility breaks." The pathmaker then designs the fused op to maintain determinism (e.g., fixed-order tree reduction) because the ledger made the cost visible.

---

## Lifecycle conventions

Following the three-registry convergence pattern (OrderStrategy registry, oracles registry, this ledger — all same shape per navigator's 2026-04-12 observation):

- **New invariants get new rows via PR review.** The row proposer fills in all four columns (precondition, cost, user-facing property, cross-reference) and the reviewer checks that each column is specific enough to cite at the uses above.
- **Changing a row's precondition classification is a major review.** It changes the architectural model, not just the invariant's description. Requires navigator sign-off.
- **Updating the "cost of relaxation" column as understanding deepens is a minor review.** The column starts as a best-guess at invariant-authoring time and refines as the team's experience with the invariant grows.
- **Cross-references across rows** should be kept current. If I7's refinement makes I5 derivative (as noted), the I5 row's cross-reference should mention this.
- **Meta-invariants (I9, I10) are in their own section** because they don't fit the "protects a precondition" model. New meta-invariants go in that section.

---

## Open questions for future work

1. **The four-way convergence question.** Navigator observed (2026-04-12) that OrderStrategy registry + oracles registry + this ledger all have the same engineering shape, and hypothesized a fourth registry (device capability matrix is the most likely) may emerge. If it does, formalize the convention explicitly as a team pattern. Candidate name: "Named Architectural Artifact" or similar.

2. **Kingdom B/C/D invariants.** The current ledger covers Phase 1 (Kingdom A only). When Kingdom B (sequential recurrence), Kingdom C (iterative fixed-point), and Kingdom D (stochastic) land, each may introduce new invariants that need ledger rows. Preconditions P1/P2/P3 should still apply but the "what protects what" classification may shift.

3. **I9′ implementation pacing.** The refined I9′ (multi-oracle suite + TESTED/CLAIMED profile + oracles registry + independence matrix) is accepted but awaits Peak 2 implementation. When it lands, this ledger's I9 row needs a rewrite to cite the full suite. Currently the row carries a "refinement pending" note.

4. **Cross-kernel fusion invariants.** Phase 1 disallows cross-kernel fusion. When it lands (Phase 2+), the compatibility predicate on OrderStrategy (per I7 Phase 6 deep dive) expands to cross-kernel queries, and a new invariant may be needed to govern when cross-kernel fusion preserves the compositional claim.

5. **Guarantee Ledger for the correctness axis, Something Else for the speed axis?** The current ledger is *entirely* about the correctness axis of the trek. The speed axis (decomposition → fusion → minimum-pass kernels) is the second architectural commitment, covered by the dataflow side of the refined I7 but not explicitly audited in ledger form. If speed commitments start needing enforcement at review time, a parallel "Speed Ledger" may be warranted. Out of scope for Phase 1; worth watching.

---

## Status

- **Drafted:** 2026-04-12 by Aristotle
- **Source material:** All three Aristotle deconstructions (notebooks 011, 012, 013 + phase docs under `peak-aristotle/`), plus the 2026-04-12 refinements to `invariants.md` (I7 refined, I11 added) and `README.md` (compositional-claim framing).
- **Next revision:** When I9′ lands as a refined invariant (pending Peak 2 campsite 2.3), this ledger's I9 row updates to reference the full suite.
