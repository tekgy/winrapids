# Navigator Campsite: Bucket A/B Convergence Finding

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** First-principles finding — structural convergence from VB-001 through VB-005

---

## What this is

Team-lead introduced the convergence-check methodology at session end and noted that VB-001 through VB-005 all converged on the same structural finding. This campsite writes up that convergence formally — as a named first-principles claim about Phase 1 vendor op safety.

This is the thing the navigator extracted from staring at five vendor bugs from the coordination vantage point. Nobody else had the view across all five simultaneously.

---

## The convergence check (applied to VB-001 through VB-005)

### The parallel outputs

| VB | Vendor | Op | Gap |
|---|---|---|---|
| VB-001 | Vulkan SPIR-V | OpFMin, OpFMax | NaN behavior undefined by spec |
| VB-002 | Vulkan SPIR-V | (all ops) | Subnormal support optional |
| VB-003 | CUDA PTX | FP arithmetic | FMA contraction default-enabled |
| VB-004 | CUDA PTX | min.f64 | NaN dropped by default (pre-ISA 7.5) |
| VB-005 | Vulkan GLSL.std.450 | Sqrt, and all extended instructions | Domain restricted to real-valued; NaN undefined |

### The structural table

| VB | Spec pin failure type | IEEE-754 corner case category | Workaround class |
|---|---|---|---|
| VB-001 | Op behavior undefined for NaN | NaN propagation | Compose from OpIsNan + OpSelect |
| VB-002 | Feature optional on device | Subnormal handling | Capability matrix → device query required |
| VB-003 | Default behavior is non-IEEE | FMA contraction | Per-op annotation (`.contract false`, NoContraction) |
| VB-004 | Default behavior drops NaN | NaN propagation | ISA version check + fallback composition |
| VB-005 | Domain exclusion (NaN undefined) | NaN propagation | Compose with OpIsNan prefix guard |

### The convergence

**Reading the "Spec pin failure type" column:** Every entry is either "behavior undefined for NaN inputs" or "optional/default-changes behavior." No entry is "the spec is wrong about arithmetic."

**Reading the "IEEE-754 corner case category" column:** Three of five entries (VB-001, VB-004, VB-005) are NaN propagation. One is subnormal handling (VB-002). One is FMA contraction (VB-003). All three categories are IEEE-754 §6 corner cases: §6.2 (NaN operations), §2.1 (subnormal numbers), §1.4 (preference for hardware-supported operations, i.e., FMA).

**Reading the "Workaround class" column:** The workarounds cluster into two types:
1. **Composition workarounds** (VB-001, VB-004, VB-005): Build the correct behavior from unambiguous primitives (OpIsNan, OpSelect, cmov)
2. **Annotation/configuration workarounds** (VB-002, VB-003): Emit an annotation or perform a capability query to opt into correct behavior

---

## The convergence finding

**All five vendor bugs are gaps in the IEEE-754 corner case handling, not in the core arithmetic.**

The core arithmetic operations — addition, multiplication, subtraction, division — are fully defined by IEEE-754 and fully pinned by every vendor's spec (with the proper SignedZeroInfNanPreserve execution mode or equivalent). Vendors don't get the fundamental arithmetic wrong.

The gaps are exclusively in:
1. **What happens when a NaN reaches a non-arithmetic op** (min, max, extended math functions that restrict domains)
2. **What happens at the subnormal boundary** (optional FTZ behavior)
3. **What happens when the hardware offers a beneficial fusion that changes the observable result** (FMA contraction)

**Formulated as a rule:**

> **The Bucket A/B boundary is exactly the IEEE-754 corner case boundary.** Vendor ops that are fully defined for NaN inputs, subnormal inputs, and with RNE rounding and no FMA contraction are Bucket A — safe to emit. Vendor ops whose spec leaves any of these undefined are Bucket B — must compose from Bucket A primitives.

This is why the P2 tightening and the ESC-002 ruling say "semantically-pinned subset of the target ISA." The pinned subset is the subset that's fully defined for all IEEE-754 corner cases. Everything else must be composed.

---

## Why this is a first-principles finding, not just an observation

The alternative hypothesis would be: "these five bugs are five independent vendor oversights." If that were true, the workarounds would be five independent solutions to five different problems.

But the workarounds converge: composition from unambiguous primitives is the solution to every NaN gap (VB-001, VB-004, VB-005), and annotation/capability-query is the solution to every default-behavior gap (VB-002, VB-003). The workaround structure is simpler than the problem list — which means the problem list is a surface manifestation of a deeper, more unified phenomenon.

**The deeper phenomenon:** IEEE-754 is a comprehensive standard for arithmetic. Vendors implement it faithfully for the common case and punt on the corner cases — either by declaring them "undefined," making them optional, or defaulting to behavior that's numerically convenient but not spec-mandated. The corner cases are exactly the cases that matter for bit-exact computation: NaN propagation determines whether errors are detectable; subnormal handling determines whether gradual underflow works correctly; FMA avoidance determines whether computed results are reproducible across hardware.

The tambear response to this phenomenon is the P2 guarantee: by limiting the lowering to a semantically-pinned subset, we accept less hardware capability in exchange for full correctness. The Bucket A/B distinction is the mechanism that enforces this tradeoff.

---

## Implications

**For the current expedition:**

The Bucket A/B finding means the current VB-001 through VB-005 set is not a random sample of vendor bugs — it's a complete characterization of the vendor gap *type*. Future VB entries will likely be more instances of the same three types (NaN propagation, subnormal handling, FMA contraction) applied to new vendor ops or new backends, not new *types* of gaps.

This means the P2 checklist (see `navigator-p2-audit-checklist` campsite) is structurally complete. The five questions on that checklist cover all three gap types. Future VB entries don't require updating the checklist — they're already covered.

**For the guarantee ledger:**

I11 (NaN propagation) is currently the only invariant that directly addresses the Bucket B composition requirement. But the finding suggests that I11 is an instance of a more general invariant: "Every backend op whose IEEE-754 corner case behavior is not fully pinned by the spec must be composed from fully-pinned ops." The current ledger has this as a consequence of P2 (faithful lowering), but it would be worth making it an explicit invariant — call it I11b or the "Bucket A constraint" — so future navigators can cite it directly.

**For future backends:**

The finding predicts: when we target a new backend (future NPU, WASM SIMD, ARM Neon), the VB entries for that backend will also cluster in the NaN/subnormal/FMA categories. The pre-flight checklist for any new backend is exactly the P2 audit checklist, applied to the new backend's spec.

---

## What aristotle would add

This finding has the same shape as the earlier architectural convergences (the three-registry pattern, the accumulate+gather unification). In each case, five seemingly-different things turn out to be instances of a single structural phenomenon. The convergence-check methodology extracted the pattern.

Aristotle would note: the pattern is distinguishable from coincidence because the *workaround structure* is simpler than the *problem structure*. Five different bugs would need five different fixes. Five instances of the same bug type need one fix template. The template being simpler than the list is the evidence for structural unity.

See also: `20260412152501-aristotle-convergence-check-on-other-roles-work` campsite for aristotle's angle.
