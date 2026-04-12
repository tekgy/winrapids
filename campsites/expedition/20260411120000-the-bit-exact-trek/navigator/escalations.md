# Escalations Log

Empty at start of expedition. Every invariant tension or cross-role blocker gets logged here with:

- **Date**
- **Raised by** (role)
- **Context** (campsite number, what you were trying to do)
- **The tension** (which invariant, what the conflict is)
- **Options considered**
- **Navigator decision** (filled in by Navigator, with reasoning)

---

<!-- Entries go below. Newest first. -->

---

## ESC-002 — 2026-04-11 — Vulkan OpFMin/OpFMax NaN behavior: I11 cannot be satisfied natively

**Raised by:** Naturalist (Entry 015, expedition-log.md; capability-matrix-vulkan-row.md)

**Context:** Pre-Peak 7. Naturalist researched the SPIR-V/Vulkan spec while
drafting the VulkanBackend capability matrix row. Phase 1 has no min/max ops
in the IR op set today. The question is what happens when they are added.

**Invariant in tension:** I11 — "every op that receives NaN as any input
propagates NaN to its output, consistently across all backends."

**The finding:**

Core SPIR-V spec for OpFMin and OpFMax: *"If either operand is a NaN, the
result is undefined."* The implementation may return the NaN, the non-NaN
operand, or any other value — implementation's choice.

GLSL.std.450:
- `FMin` / `FMax`: same undefined-NaN semantics as OpFMin/OpFMax.
- `NMin` / `NMax`: explicitly return the non-NaN operand when one input is
  NaN (IEEE 754-2008 minNum/maxNum behavior). These actively suppress NaN —
  the opposite of what I11 requires.

`SignedZeroInfNanPreserve` execution mode (SPV_KHR_float_controls): prevents
"optimizations that assume operands and results are not NaNs" on *arithmetic*
instructions (OpFAdd, OpFMul, etc.). The float_controls spec does not extend
this guarantee to OpFMin/OpFMax — these are classified differently from
arithmetic ops in the float_controls model. Emitting `SignedZeroInfNanPreserve`
does NOT make OpFMin/OpFMax NaN-propagating.

**Consequence:** There is no native SPIR-V instruction that implements
I11-compliant min/max for fp64. The `.NaN` modifier that PTX provides
(`min.NaN.f64`) has no direct SPIR-V equivalent.

**Options considered:**

1. **Mandate the four-instruction workaround for all Vulkan min/max emissions.**
   Every .tam `min.f64` / `max.f64` op lowers to:
   ```
   %is_nan_a  = OpIsNan %bool %a
   %is_nan_b  = OpIsNan %bool %b
   %either_nan = OpLogicalOr %bool %is_nan_a %is_nan_b
   %raw_min   = OpFMin %f64 %a %b
   %result    = OpSelect %f64 %either_nan %nan_const %raw_min
   ```
   Four extra instructions per min/max op. Correct; cost is ~4 instructions.
   I11 fully satisfied.

2. **Scope the I11 claim to exclude Vulkan min/max** until a backend-native
   NaN-safe instruction exists. The capability matrix cell for Vulkan
   min/max carries `NaNPropagation: ImplementationDefined` and I11 is
   documented as a "best-effort, workaround required" guarantee for that cell.

3. **Accept WeakOnComparison semantics for Vulkan min/max.** Document it as
   a known caveat in the capability matrix. Narrow I11's language to
   "arithmetic ops propagate NaN on all backends; min/max NaN behavior is
   backend-documented in the capability matrix." This is the weakest
   resolution — it fragments I11 into a per-op claim.

**Navigator decision:**

*(Draft by scout proposed Option 1. Superseded by team-lead directive
2026-04-11. Final ruling below.)*

**Option 3 — do not emit OpFMin or OpFMax at all. Compose min/max from
well-defined primitives.** Reasoning:

Option 1 (four-instruction workaround with a NaN constant) still references
OpFMin as the non-NaN path. The implementation-defined NaN behavior of
OpFMin means we are relying on a vendor guarantee ("returns the non-NaN
operand when inputs are normal") that the spec does not provide. This is a
latent correctness dependency, not a true fix.

Option 3 eliminates the dependency entirely. Every instruction in the
composed sequence has fully-defined IEEE-754 behavior:
- `OpIsNan` — always true when input is NaN, no undefined behavior.
- `OpFOrdLessThan` — returns false when either operand is NaN (IEEE
  ordered comparison; this is the definition of "ordered").
- `OpSelect` — a bitwise mux; no floating-point semantics at all.

The composed sequence is six instructions (vs. five for Option 1 or four
for the original workaround), but the instruction count is irrelevant: this
is a future-op concern, not a hot path we are optimizing today.

**The composed min(a, b) sequence for SPIR-V:**

```
%is_nan_a    = OpIsNan %bool %a
%is_nan_b    = OpIsNan %bool %b
%lt          = OpFOrdLessThan %bool %a %b   ; false if either is NaN
%min_non_nan = OpSelect %f64 %lt %a %b
%min_b_nan   = OpSelect %f64 %is_nan_b %b %min_non_nan
%result      = OpSelect %f64 %is_nan_a %a %min_b_nan
```

For max(a, b), replace `OpFOrdLessThan` with `OpFOrdGreaterThan` and swap
the `%a %b` arguments in `%min_non_nan`.

This is not a narrowing of the I11 claim. I11 is fully satisfied on Vulkan
by taking ownership of NaN semantics at the lowering boundary rather than
delegating to OpFMin. The claim holds: NaN propagates through min/max on
every backend.

Options 2 (WeakOnComparison documentation) and the original Option 1
(OpIsNan guard around OpFMin) are not acceptable. Option 2 fragments I11.
Option 1 leaves a latent dependency on undefined OpFMin behavior.

**Actions:**

- **vendor-bugs.md entry VB-001** (created 2026-04-11) documents this gap
  and the workaround. Trek-wide policy: when any team member finds a vendor
  IEEE-754 gap, log it in `vendor-bugs.md` with the workaround.

- **capability-matrix-vulkan-row.md** (at `peak7-spirv/`) must be updated:
  the OpFMin/OpFMax row status changes from `Partial (I11-conditional)` to
  `Supported (composition required — never emit OpFMin/OpFMax)` and the
  notes section documents the six-instruction sequence above. Scout to
  update the stub; pathmaker incorporates when Peak 7 starts.

- **When the .tam IR adds a min/max op** (not in Phase 1 today), pathmaker
  must document in the op's spec entry: "Vulkan backend emits the
  six-instruction composed NaN-safe sequence. OpFMin and OpFMax are never
  emitted."

- **Peak 3 (PTX):** PTX does not share this gap — `min.NaN.f64` and
  `max.NaN.f64` are NaN-propagating by spec. Peak 3 pathmaker must use
  these variants, not bare `min.f64` / `max.f64`.

- **The Peak 7 campsite 7.1 pre-flight checklist** must confirm the
  composed sequence against the device before implementation begins.

- No campsite is currently blocked by this decision. It is a scope
  clarification for a future op, analogous to ESC-001's treatment of
  subnormals.

**Required device queries before campsite 7.1 opens:**

The following properties were not queried in `scout-vulkan-terrain.md` and
must be confirmed via `vulkaninfo` before campsite 7.1 (SPIR-V module
structure) begins:

1. **`shaderSignedZeroInfNanPreserveFloat64`** — required to emit the
   `SignedZeroInfNanPreserve 64` execution mode for I11 on arithmetic ops
   (OpFAdd, OpFMul, etc.). If false, the execution mode cannot be requested
   and I11 for arithmetic requires a different approach.

2. **`VK_EXT_shader_atomic_float` extension availability** — required for
   Phase 1 `atomicAdd` on fp64 output slots (`OpAtomicFAddEXT`). If
   unavailable, Phase 1 must use tree-reduce-to-one approach (workgroup
   reduce → non-atomic store), which is I5-compliant and avoids the
   extension entirely. Preferred design regardless.

3. **`SignedZeroInfNanPreserve` scope for `OpExtInst` (GLSL.std.450 Sqrt)**
   — the float_controls spec applies to arithmetic instructions; unclear
   whether it covers extended instructions. If not covered, sqrt(NaN) on
   Vulkan may not propagate NaN, requiring a manual OpIsNan guard for fsqrt
   as well.

4. **OpFMin/OpFMax NaN behavior under `SignedZeroInfNanPreserve`** — the
   ESC-002 finding confirms this is not covered, but pathmaker should
   verify this understanding against any updated float_controls spec
   errata before implementing.

Query command: `vulkaninfo --json` and check
`VkPhysicalDeviceVulkan12Properties` fields plus extension list.

---

## ESC-001 — 2026-04-11 — Vulkan fp64 subnormal behavior: is the architectural claim overstated?

**Raised by:** Naturalist (Entry 005, expedition-log.md)

**Context:** Pre-Peak 7. Scout ran `vulkaninfo` on the RTX 6000 Pro Blackwell and found that both `shaderDenormPreserveFloat64` and `shaderDenormFlushToZeroFloat64` are `false`. The Vulkan driver makes no guarantee about fp64 subnormal behavior at all — it's implementation-defined.

**The tension:** The expedition README states the claim as "the same numerical answers running on any ALU." If the Vulkan backend can flush or preserve subnormals at will, then `cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()` provably fails for any input producing a subnormal result — before we've even started Peak 7.

**Options considered:**

1. **Require `shaderDenormPreserveFloat64` as a hard prerequisite for target hardware.** Makes the claim provably true but rules out any Vulkan-capable GPU that doesn't set this bit (which is most current hardware, including our own RTX 6000 Pro).

2. **Carve subnormals out of the architectural claim.** State the claim as "bit-identical for all normal fp64 results; subnormal handling is implementation-defined and documented per device." The summit test (7.11) would test only normal-output kernels, or add a prerequisite device query.

3. **Design tambear-libm to avoid subnormal results in all kernels.** Clamping and domain-checking at the function level so no `.tam` kernel ever produces a subnormal output. Expensive, fragile, impossible for generic accumulations.

4. **Document the gap now, decide in Peak 7.** Let the claim stand as-is through Peaks 1–6, add the subnormal qualifier to the expedition README before Peak 7 begins.

**Navigator decision:**

Option 2 with immediate README amendment, and Option 4's timing. The claim is not wrong — it is incomplete. The corrected statement:

> **Bit-exact for all normal fp64 inputs and outputs. Subnormal behavior is hardware-defined and requires `shaderDenormPreserveFloat64 = true` for full cross-backend bit-exactness. Phase 1 claims bit-exactness in the normal fp64 range only.**

Actions:
- The expedition README (`README.md`) must be updated with the qualifier before Peak 7 is opened. Navigator will do this.
- The summit test (campsite 7.11) will be parameterized: if the device reports `shaderDenormPreserveFloat64 = false`, subnormal-producing inputs are skipped (not failed — the test is not applicable on that device).
- The hard-cases suite (Peak 4 campsite 4.7) will retain subnormal generators — they test *our* CPU interpreter's behavior, which must handle subnormals correctly. The CPU interpreter must NOT flush subnormals to zero.
- No campsite for any peak is blocked by this decision. It is a scope clarification, not a scope reduction.

This is not a retreat. The claim is still novel and testable. Subnormal handling is the last 1% of the IEEE 754 specification, and claiming it requires specific hardware features. We state what we claim, we state what the hardware prerequisite is, and we deliver both.
