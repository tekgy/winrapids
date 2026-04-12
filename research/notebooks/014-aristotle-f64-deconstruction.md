# Lab Notebook 014: Aristotelian deconstruction of "f64 as base precision" + Device Capability Matrix

**Date**: 2026-04-12
**Authors**: Aristotle (agent role on the Bit-Exact Trek)
**Branch**: main
**Status**: Complete — Phases 1–8 drafted + structural ULP addendum + device capability matrix structural extension
**Depends on**: Notebooks 011, 012, 013; invariants.md (post-refinement I1–I11); trek-plan.md Part II.5 compositional claim; guarantee-ledger.md (commit a8ea33a)

## Context & Motivation

Notebook 011 deconstructed I7 (accumulate + gather). Notebook 012 deconstructed I9 (mpmath oracle). Notebook 013 deconstructed the meta-goal (bit-exact cross-hardware). This notebook closes the fourth target navigator assigned: **"Why f64 as the base precision?"**

The target is different in shape from the first three. Where I7 / I9 / meta-goal were about making tacit *invariants* explicit, f64 is about challenging a tacit *engineering convention* that has been the scientific-computing default for 40+ years. Questioning it seriously is rare. Navigator asked for this specifically to surface structural reasons (ULP budget, accumulate+gather decomposition requirements) rather than engineering reasons (hardware support). My first pass missed that framing — I gave the engineering answer — and I owe an addendum (appended to the phase doc and summarized here) that surfaces the structural answer.

Navigator also asked: **"Does the f64 deconstruction have anything to say about the device capability matrix specifically — what entries it needs, who owns it, what the review gate looks like?"** The answer is yes, and this notebook captures the structural extension before implementation details scatter it.

## Hypothesis

**H0 (project assumption):** fp64 is tambear's numerical type. All `.tam` IR ops, all accumulators, all intermediates, all storage use fp64. No other precision is first-class in Phase 1.

**H1 (Aristotle's counter-hypothesis):** fp64 is not an arbitrary choice; it's the minimum precision at which tambear's composed-operation error budget fits within a useful output precision for the workloads we target. This is a structural claim, not an engineering one. The implication is that "why not fp32?" has a mathematically necessary answer — not "fp32 is inconvenient" but "fp32 produces zero useful digits for tambear's implied workloads after the composed error budget is spent." Additionally, the Phase 1 choice of fp64 carries an extension path: the IR should parameterize by precision so Phase 2+ can add fp32 (for specific memory-bound workloads that clear a composed-error-budget check) and numeric_format (for posits and other non-IEEE formats) without a format bump.

**Prediction:** The Phase 2 enumeration of irreducible truths surfaces a composed-error-budget truth that makes the fp32-vs-fp64 question structurally decidable rather than engineering-contingent. The Phase 5 Move is a type-system parameterization that preserves Phase 1 behavior while leaving extension paths open.

## Design

Eight-phase Aristotelian deconstruction. Working document at `campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/f64-base-precision-phases.md`.

Plus a structural extension on the device capability matrix (the fourth named artifact per navigator's 2026-04-12 confirmation), because the f64 deconstruction touches it implicitly at every precondition-3 boundary and should deliver the concrete matrix schema before implementation scatters the reasoning.

## Design Decisions (what we chose AND rejected)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Target selection | "Why f64?" (navigator's priority #2) | "Why Cody-Waite?", "Why SSA?" | f64 is foundational; Cody-Waite is downstream technique. Per navigator: "f64 is a commitment; Cody-Waite is a consequence of it." |
| Framing posture | Challenge the convention rigorously | Assume convention is right, just audit it | Phase 8 forced rejection demands the deepest possible challenge. Assuming correctness upfront would miss the structural reasoning navigator specifically asked for. |
| Phase 2 scope | Representation + composed-error-budget truths (post-addendum) | Representation-only truths (first-pass, incomplete) | First pass gave the engineering answer. Navigator's ask required the structural answer. The addendum surfaces the composed-error-budget truth as a new Phase 2 entry (T11). |
| Move shape | Parameterize precision in IR from day one, monomorphic Phase 1, Phase 2 extension path | Monomorphic fp64 forever; mandatory Phase 2 mixed precision now | Same shape as I7′ and I9′: name the axis even when Phase 1 uses one value. Extension is additive, not corrective. |
| Phase 2 fp32 policy | Specialization path per-recipe with composed-error-budget check | Generic fp32 opt-in for memory-bound kernels | Generic fp32 opt-in is unsafe: fp32 is structurally insufficient for long composed chains and large reductions. Per-recipe gating is the only safe path. |
| Notebook 014 scope | f64 deconstruction summary + device capability matrix structural extension | Summary only | Navigator explicitly asked for the capability matrix analysis to land in 014 rather than being left for a future campsite. The notebook is the right place to capture the structural claim before implementation details scatter it. |

## Results

### Phases 1–8 summary

I'll not repeat the full phase content here — that lives at `peak-aristotle/f64-base-precision-phases.md`. The headline findings:

**Phase 1 — 10 stacked assumptions** inside "f64 is base precision." The ones that matter most:
- That there IS a base precision (vs per-op or per-quantity)
- That fp64 is the right fp64 (vs posits, logarithmic, custom formats)
- That the precision is enough (workload-dependent — structurally matters)
- That the precision is not excessive (memory bandwidth cost — workload-dependent)
- That fp64 is portable across every ALU (it isn't — accelerators often lack fp64)

**Phase 2 — 10 irreducible truths** about numerical representation. Key ones:
- T1-T5: representations are trade-offs; rounding is unavoidable; precision needs are problem-dependent
- T8: memory bandwidth cost is proportional to type size
- T10: the compositional bit-exact claim holds for any well-defined type with compliant hardware — it's NOT fp64-specific

**Phase 2 addendum — the truth I missed in the first pass (T11):**
> For a pipeline composing K transcendentals and a reduction over N elements, the output's useful precision is roughly `mantissa_bits − log₂(K) − log₂(N)/2` (compensated) or `mantissa_bits − log₂(K) − log₂(N)` (naive). tambear's Phase-1 workloads routinely have K ≈ 10-20 and N ≈ 10^6-10^9. The output needs at least 10 decimal digits (~33 bits) to be statistically useful. Minimum mantissa: 33 + log₂(20) + log₂(10^9)/2 ≈ 52 bits. **fp32 (23 bits) fails; fp64 (52 bits) satisfies exactly.**

This is the structural reason fp64 is the minimum. It's not "hardware supports it"; it's "the composed-operation error budget is structurally contained within fp64's mantissa for tambear's implied workloads, and NOT contained within fp32's."

**Phase 3 — 10 reconstructions** ranging from "fp64 everywhere (current)" through "fp32 everywhere," "fp32 storage / fp64 compute," "mixed precision per-tensor," "per-quantity compiler-inferred," "posits," "interval arithmetic," "arbitrary-precision," "fp128," to "per-op precision hint as first-class IR concept" (the recommended landing).

**Phase 4 collisions** mostly between "fp64 is right because standard" and the trade-off truths. The deepest collision: "fp64 everywhere" is a simplification, not a derived result — you can support mixed precision structurally if the IR names precision as a parameter.

**Phase 5 Move v2:**
> Preserve fp64 as Phase 1's only implemented precision, AND parameterize fp ops by precision in the `.tam` IR from the start (even if only f64 is implemented), AND add `numeric_format` as a separate future-work axis (IEEE-754 now, posits/log/custom later). Three concrete triggering conditions for Phase 2 mixed precision: (a) first accelerator target that lacks fp64 hardware, (b) first Phase 1 recipe whose memory-bandwidth cost exceeds a measured threshold on production workloads, (c) first user request for explicit fp32 opt-in for memory reasons.

**Correction (post-addendum):** The Phase 5 Move v2's triggering condition (b) is incomplete. It allowed fp32 opt-in for memory-bound workloads. But the composed-error-budget truth (T11) says fp32 is structurally insufficient for any workload with non-trivial K and N. **Phase 2 fp32 is a per-recipe specialization path, not a generic default. The triggering conditions need a fourth filter: the composed-error-budget check per recipe.**

**Phase 6 — recursion** on the Move. Ten new assumptions surfaced; most pointed to the same conclusion: parameterizing precision is straightforward in the IR but has real implementation cost in backends. The Move remains v2 (stable). The key recursion finding: the Move addresses ESC-001 and RFA state size but NOT posits. Posits need a separate `numeric_format` axis. The Move's scope is honest about this.

**Phase 7 — stability.** Second pass found the Move stable. One retraction from the earlier deconstructions: precision is NOT a fourth registry. Precision is a type-system concept, not a library-extended artifact. The three-registry pattern remains at three. The device capability matrix is the stronger fourth-registry candidate, which is what navigator agreed with in the 2026-04-12 routing response.

**Phase 8 — forced rejection.** Three alternative framings considered. The most interesting: "per-quantity precision" (each value carries its own precision setting). This is essentially mpmath-style arbitrary-precision arithmetic — viable but 100-1000x slower than fp64 on current hardware. Deferred.

**The unseen first principle (post-addendum):**
> **fp64 is not an arbitrary Phase 1 choice. It is the minimum precision at which tambear's composed-operation error budget fits within a useful output precision for statistical-workload-scale inputs. The smallest representation that enables the compositional speed story (via accumulate+gather fusion) without collapsing the compositional correctness story (via error accumulation through chains and reductions).**

### Device Capability Matrix — structural extension per navigator's ask

The f64 deconstruction touches the device capability matrix at every precondition-3 (IEEE-754 compliance) boundary. The matrix is already implicit in:
- The Phase 5 Move's "backend declares which precisions it supports" language
- The I7′ v5 registry's "backends publish capability matrices" for OrderStrategy support
- The guarantee-ledger's P3 column (IEEE-754-compliance-for-ops-used)
- The ESC-001 escalation's `shaderDenormPreserveFloat64` feature-bit check
- The I11 invariant's "on every backend" language for NaN propagation

What's missing is a single place that names the matrix's shape, ownership, lifecycle, and initial rows. This section provides that.

#### Shape

The matrix is **three-dimensional**: `(backend × op × precision) → cell`. Each cell contains a structured entry with several fields:

```
CapabilityEntry {
  status:            Supported | SupportedWithCaveats | Unsupported,
  caveats:           Vec<Caveat>,           // see enumeration below
  order_strategies:  Vec<OrderStrategyId>,  // which OrderStrategy registry entries this (backend, op, precision) supports bit-exactly
  oracle_profile:    OracleProfileId,       // which I9' oracle profile validates this cell
  last_verified:     Timestamp,             // when the cell's claim was last run against the cross-backend diff harness
  notes:             String,
}
```

#### Caveat enumeration

Caveats are the structured way to document hardware quirks without silently downgrading. Phase 1 caveats:

- **SubnormalHandling**: `Preserve` | `FlushToZero` | `Trap` | `ImplementationDefined` (per IEEE-754's optional subnormal semantics; ESC-001 is `ImplementationDefined` for Vulkan fp64).
- **NaNPropagation**: `Strict` (per I11) | `StrictOnArith_WeakOnComparison` | `Weak`. Default expected: Strict. Min/Max/Select/Clamp on some backends are weak.
- **RoundingMode**: `RoundToNearestEven` (default) | `OtherModeSupported(list)`. Phase 1 requires RoundToNearestEven.
- **FMAContraction**: `NeverContract` (per I3) | `ContractsByDefault` (unsafe) | `ExplicitControl` (safe, the PTX `.contract false` path).

Adding a new caveat type is a major review (it introduces a new dimension the compile-time reject logic must handle).

#### Ownership

Per-backend entries are owned by the backend's implementer. Cross-backend registry is navigator's domain or a shared artifact.

| Backend | Owner | Notes |
|---|---|---|
| `CpuInterpreter` | scientist (Test Oracle) | Reference — declares what IEEE-754-compliant Rust `f64` provides |
| `CudaBackend` | pathmaker | Declares what Blackwell + PTX `.contract false` + `mov.f64` provides |
| `VulkanBackend` (future) | pathmaker or scout | Declares what Vulkan + SPIR-V + `shaderDenormPreserveFloat64` feature bit provides |
| `MetalBackend` (future) | (unassigned) | Declares what Metal Shading Language provides |
| `NpuBackend` (future) | (unassigned) | Declares what fp32-only accelerators provide — will force the Phase 2 mixed-precision conversation |

#### Review gate

Four classes of PRs touch the matrix, each with a different review requirement:

**1. Adding a new backend row.** PR review by the backend owner + navigator. The row must match what the hardware actually provides, not what the backend author hopes it provides. Must be accompanied by a test suite run (the Test Oracle's cross-backend diff harness) showing the claimed cells match the CPU interpreter's reference to the declared precision.

**2. Adding a new (op, precision) cell to an existing backend.** PR review requires a passing test that bit-exactly compares the backend's output to the CPU interpreter on representative inputs. Same gate as adding an OrderStrategy registry entry.

**3. Adding a new caveat type.** Major review. Adds a dimension the compile-time reject logic must handle. Requires navigator + IR Architect + Test Oracle sign-off.

**4. Downgrading a cell (claiming less support than before).** This is the "bug discovered" pattern. Requires escalation via `navigator/escalations.md` before merging, because it narrows what users can declare in kernels and may break existing recipes. ESC-001's Vulkan fp64 subnormal downgrade is the canonical pattern — it didn't happen silently; it went through a full escalation and a decision.

#### Lifecycle conventions (parallel to the other three registries)

- **Named entries** — each backend has a stable identifier; each op in the `.tam` IR has a stable identifier; each precision has a stable identifier. Cells are keyed by the triple.
- **Formal content** — each cell has the structured `CapabilityEntry` schema. No free-text "status unknown" cells; if status is unknown, it's `Unsupported` until proven otherwise.
- **Reviewable at merge time** — every matrix change goes through PR review per the gate classes above.
- **Role-owned** — per-backend ownership is explicit in the table above. Cross-backend registry is navigator's artifact (or delegate).
- **Consulted during escalation** — escalations in `navigator/escalations.md` cite the matrix cell that motivated the escalation (see ESC-001 for the pattern).
- **Cited in the Guarantee Ledger's P3 column** — each invariant's "protects P3" cost-of-relaxation row references the capability matrix columns it depends on.

This is the same lifecycle as OrderStrategy registry, Oracles registry, and Guarantees ledger. Four registries, same shape. **Per navigator's 2026-04-12 routing: "when it lands, formalize it with the same lifecycle conventions as the other three registries. Worth naming explicitly in the trek documents at that point."**

#### Initial Phase 1 rows

Concrete proposal for the matrix's starting state. Every cell currently declares fp64 only.

**CpuInterpreter backend:**

| op | precision | status | caveats | order_strategies | oracle_profile |
|---|---|---|---|---|---|
| `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, `fneg`, `fabs` | f64 | Supported | [SubnormalHandling: Preserve, NaNPropagation: Strict, RoundingMode: RoundToNearestEven, FMAContraction: NeverContract] | [SequentialLeft, TreeFixedFanout(k) for any k] | (pending I9′ v4 registry) |
| `fcmp_*`, `select` | f64 | Supported | [NaNPropagation: Strict per I11] | N/A | (pending) |
| `tam_exp`, `tam_ln`, `tam_sin`, `tam_cos`, etc. (libm) | f64 | SupportedWithCaveats | [per-function ULP bound from Peak 2 accuracy target] | N/A | (pending) |

**CudaBackend (Phase 1 target: NVIDIA Blackwell via PTX without NVRTC):**

| op | precision | status | caveats | order_strategies | oracle_profile |
|---|---|---|---|---|---|
| `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, `fneg`, `fabs` | f64 | Supported | [SubnormalHandling: Preserve (sm_60+), NaNPropagation: Strict, RoundingMode: RoundToNearestEven via `.rn`, FMAContraction: NeverContract via `.contract false`] | [TreeFixedFanout(32) per warp; others pending implementation] | (pending) |
| libm ops | f64 | Unsupported → SupportedWithCaveats after Peak 2 + Peak 3 | (pending) | N/A | (pending) |

**VulkanBackend (Phase 1 planned, Peak 7):**

| op | precision | status | caveats | order_strategies | oracle_profile |
|---|---|---|---|---|---|
| `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, `fneg`, `fabs` | f64 | **SupportedWithCaveats per ESC-001** | [SubnormalHandling: **ImplementationDefined unless `shaderDenormPreserveFloat64`**, NaNPropagation: Strict per I11, RoundingMode: RoundToNearestEven via SPIR-V OpRoundingModeRTE, FMAContraction: NeverContract via `NoContraction` decoration] | [pending Peak 7] | (pending) |

The ESC-001 subnormal caveat on VulkanBackend is the canonical example of a non-empty caveat: the backend supports fp64 ops in the normal range, but subnormal outputs are implementation-defined unless the device provides the feature bit. The compile-time reject logic consults this caveat when a user kernel's `CapabilityRequirement` includes subnormal-producing ops.

#### What this matrix enables

1. **Compile-time reject** of kernels declaring ops or order strategies a backend doesn't support. No silent fallback (I6 enforcement).
2. **Continuous cross-backend verification.** The Test Oracle's harness reads the matrix to know which (backend, op, precision) cells should be bit-exactly equal to the reference. A cell disagreement that should agree is an automatic test failure.
3. **Honest scope documentation.** When a user asks "does tambear run on platform X?", the answer is readable from the matrix — not "yes but..." handwaving.
4. **Structural basis for Phase 2 mixed precision.** When fp32 lands, each backend gets new cells. Adding fp32 support doesn't require refactoring the matrix; it's new rows in the existing structure.
5. **Guarantee Ledger P3 column enforcement.** Each invariant protecting P3 (IEEE-754-compliance-for-ops-used) cites specific matrix columns it depends on. Relaxing the invariant requires proving the matrix can still support the claim.

#### The four-artifact convention

Navigator confirmed the structural observation: the three existing registries plus the capability matrix form a four-artifact convention. All four share:
- Named entries with stable identifiers
- Formal structured content (schema-defined)
- Review-time enforcement (PR review via the gate classes)
- Role-owned (explicit ownership per artifact and per entry)
- Consulted during escalation (the cost-of-relaxation question lives in them)
- Cross-referenced in the Guarantee Ledger

**The convention name (candidate):** "Named Architectural Artifacts." Each one externalizes a class of tacit knowledge that would otherwise be invisible to review.

| Artifact | Lives at | Externalizes | Owner |
|---|---|---|---|
| OrderStrategy registry | `campsites/expedition/.../order_strategies/` | Order-of-operations decisions inside accumulate ops | IR Architect |
| Oracles registry | `campsites/expedition/.../oracles/` (pending) | Verification methodology for libm and similar | Adversarial Mathematician + Test Oracle |
| Guarantee Ledger | `campsites/expedition/.../guarantee-ledger.md` | Architectural commitments (invariants) and their user-facing costs | Navigator |
| Device capability matrix | `campsites/expedition/.../capability-matrix.md` (proposed) | Per-backend IEEE-754 compliance for the ops used | Per-backend owners, centralized registry |

If a fifth artifact emerges, it probably externalizes a fifth class of tacit knowledge at the same boundary. Watching for it.

### Surprise

Two surprises, one this session, one in the recursion:

**First pass surprise:** I expected to find that fp64 was overreach and recommend fp32 as the sensible Phase 1 choice. Instead, I found that fp64 is not overreach — it's exactly the minimum. The deconstruction confirmed the convention rather than challenging it. That's a less dramatic finding than I expected, but arguably a more useful one: sometimes the convention IS the right answer, and the first-principles work is to articulate *why* cleanly enough to defend it under pressure.

**Recursion-phase surprise:** Navigator's ask about the "structural reason" made me realize my first pass had given the engineering answer and missed the structural answer. The composed-error-budget truth (T11) was there in Phase 2 territory but I didn't surface it because I was thinking in "trade-offs" terms rather than "ULP budget chains" terms. Owning this gap is the most important methodological lesson from this session: **when a deconstruction concludes "the convention is right," double-check whether I've surfaced the structural reason or just the engineering reason.** Engineering reasons are contingent (hardware, budget, convenience). Structural reasons are necessary (math, error propagation, composition). Navigator's feedback cycle caught this; my self-review hadn't.

## Interpretation

f64 is not a Phase 1 default that will eventually be replaced; it is the structural minimum for tambear's composed-operation workloads. The Phase 5 Move v2 parameterizes precision in the IR (preserving Phase 2 extension paths) while keeping fp64 as the only implemented precision for Phase 1. The Phase 2 fp32 path is a per-recipe specialization, gated on a composed-error-budget check, NOT a generic opt-in for memory-bound kernels.

The device capability matrix is the fourth named architectural artifact the project has implicitly been building. Its shape (3D: backend × op × precision, with caveats, order strategies, oracle profiles, and verification timestamps) is derivable from the four invariants it enforces: I6 (no silent fallback), I7 (OrderStrategy declaration), I9 (oracle profile per function), I11 (NaN propagation). The matrix's lifecycle parallels the three existing registries. When the capability matrix lands as a concrete artifact (likely at `campsites/expedition/.../capability-matrix.md`), it should be formalized with the same conventions.

The four-artifact convention — OrderStrategy registry, Oracles registry, Guarantee Ledger, capability matrix — is worth naming explicitly in the trek documents as a team pattern. Each externalizes a class of tacit architectural knowledge that would otherwise be invisible to review. Adding a fifth artifact is now a known pattern, not a one-off invention.

## Artifacts

### Phase document
| File | Description |
|---|---|
| `campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/f64-base-precision-phases.md` | Full eight-phase deconstruction + addendum on ULP budget + capability matrix discussion |

### Cross-references to earlier deconstructions
| Notebook | Target | Move | Adoption state |
|---|---|---|---|
| 011 | I7 accumulate+gather | v5.2 (order + fusion compatibility predicate) | Adopted as refined I7 in invariants.md; implemented in Peak 1 campsites 1.16-1.17 |
| 012 | I9 mpmath oracle | v4 (TESTED/CLAIMED profile + oracles registry + auditability contract reframe) | Accepted; awaiting Peak 2 campsite 2.3 implementation |
| 013 | meta-goal bit-exact | v5 (Guarantee Ledger + three preconditions) | Adopted in README, trek-plan Part II.5; Guarantee Ledger committed at a8ea33a |
| 014 (this) | f64 base precision | v2 (parameterized precision + numeric_format axis + ULP budget structural argument + capability matrix) | Pending Phase 2 trigger |

### Device capability matrix — pending artifact
| File (proposed) | Description | Owner |
|---|---|---|
| `campsites/expedition/.../capability-matrix.md` | Per-(backend, op, precision) cells with caveats, order strategies, oracle profiles, verification timestamps | Per-backend owners; centralized registry |

## Open Questions

1. **Should the capability matrix live alongside OrderStrategy and Oracles as a peer registry, or be a column on each OrderStrategy/Oracle entry?** My current framing treats it as a peer. An alternative is to make "supported on which backends" a field on each registry entry. The peer framing is cleaner because the matrix has additional dimensions (op, precision, caveats) that don't fit as a side column.

2. **What is the minimum set of ops in the Phase 1 capability matrix?** The `.tam` IR's current op set (fadd, fsub, fmul, fdiv, fsqrt, fneg, fabs, fcmp_*, select, tam_exp, tam_ln, tam_sin, tam_cos, ...) gives ~20 ops. With 3 backends (CPU, CUDA, Vulkan) and 1 precision (fp64), that's 60 cells. Small enough to review by hand; large enough that a structured format is required.

3. **Who writes the Vulkan backend row before Peak 7 starts?** The row has to exist so that ESC-001's caveat is documented upfront, not retroactively. Scout has done reconnaissance on the SPIR-V subnormal question; scout or pathmaker can draft the row. Navigator's call on timing.

4. **Does the capability matrix need machine-readable format (TOML / JSON)?** For automated compile-time rejection the compiler needs to read it. A markdown table is human-readable but needs to also be parseable. Either we write it in both formats (redundancy risk) or we generate one from the other (tooling risk). Phase 1 can start markdown-only; when the compile-time check lands, generate the machine-readable form.

5. **When does the "Named Architectural Artifacts" convention get formalized in a trek document?** Navigator said "worth naming explicitly in the trek documents at that point." The point is approximately when the capability matrix lands. Writing a short `named-artifacts-convention.md` at that time would capture the pattern for future contributors.

## Next

1. **This notebook is complete.** No further content required.
2. **Watching for navigator's response** on the capability matrix structural extension. If navigator wants it as a standalone campsite artifact rather than just notebook content, I can draft the `capability-matrix.md` schema file when directed.
3. **The four-deconstruction arc is closed** with this notebook. No fifth target queued. The garden clause applies: when idle, I follow curiosity or wait for first-principles questions that surface during the critical path (Peak 2 libm, pending Peak 5 → Peak 3 chain, Peak 6 determinism informed by I7′).
4. **Math-researcher should see the structural ULP-budget argument.** Requested navigator forward the addendum in the last message before this one; leaving that as standing.
