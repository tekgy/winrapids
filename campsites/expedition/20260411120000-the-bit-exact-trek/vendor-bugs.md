# Vendor Bugs — IEEE-754 Gap Log

**Trek artifact.** Every place where a vendor's hardware or ISA deviates from IEEE-754 semantics — whether by underspecification, optional feature, or incorrect default — gets logged here with a trek-owned workaround.

**Policy:** When you find a vendor IEEE-754 gap (in a spec, a vulkaninfo report, a device test, or a campsite), log it here. Include the workaround we use. If there is no trek workaround yet, mark the status `OPEN` and escalate. Do not paper over gaps silently.

**Format:** one entry per gap. Fields: Vendor, Component, Gap, Trek impact, P2 implication, Workaround, Upstream report, Dates.

**P2 implication field:** Every entry must state which P2 resolution applies — either (a) **Composed-primitive workaround** (we avoid the ambiguous vendor op and emit our own unambiguous sequence) or (b) **Capability-matrix narrowing** (we require the hardware feature and reject kernels when it's absent). Both are valid responses to vendor gaps; what's not valid is silently depending on implementation-defined behavior.

---

<!-- Entries below. Newest first. -->

---

## VB-005 — GLSL.std.450 `Sqrt` NaN behavior is underspecified

**Vendor:** Khronos (Vulkan / GLSL.std.450)
**Component:** GLSL.std.450 extended instruction set, opcode 31 (Sqrt)
**Gap:** The GLSL.std.450 specification for `Sqrt` states the operand must be
non-negative (≥ 0). NaN is neither positive nor negative, so NaN input falls
into the domain where the spec makes no guarantee. IEEE 754-2019 §9.2 specifies
that `squareRoot(qNaN)` shall return a NaN, and GLSL.std.450 is broadly
intended to follow IEEE semantics — but the domain restriction text creates
ambiguity: is NaN excluded from the guarantee? The spec does not say explicitly
that `Sqrt(NaN) = NaN`. Additionally, `SPV_KHR_float_controls` (the
`SignedZeroInfNanPreserve` execution mode) is specified to apply to arithmetic
instructions (OpFAdd, OpFMul, etc.) but the spec does not explicitly state that
it covers `OpExtInst` (extended instructions). A conformant implementation could
interpret `SignedZeroInfNanPreserve` as inapplicable to `Sqrt`.

This is a narrower gap than VB-001 (OpFMin NaN is fully "undefined" in the core
spec). Here the spec is silent rather than explicitly undefined. But silence is
not a guarantee, and under the tightened P2 ("the lowering must pin the
IEEE-754 semantics of every emitted op from the target spec alone"), silence is
not sufficient — we cannot faithfully lower `.tam fsqrt(NaN)` to an op whose
NaN behavior the spec does not pin.

**Trek impact:** I11 (NaN propagates through every op on every backend). If
`OpExtInst Sqrt` does not propagate NaN on some Vulkan implementation, a kernel
computing `sqrt(NaN)` would return a non-NaN on Vulkan while returning NaN on
CPU and PTX — a cross-backend I11 failure.
**P2 implication:** Composed-primitive workaround. Same class as VB-001: the
spec does not guarantee the behavior we require, so the faithful lowering must
not delegate NaN semantics to the op. Emit an explicit `OpIsNan` guard: if
input is NaN, return the input NaN directly via `OpSelect`; otherwise emit
`OpExtInst Sqrt`. The non-NaN path is fully spec-compliant; the NaN path is
handled by well-defined ops (`OpIsNan`, `OpSelect`) that have no NaN ambiguity.

**Workaround:** Emit an explicit NaN guard for every `.tam fsqrt.f64` on Vulkan:

```
; fsqrt(x) — I11-compliant SPIR-V emit
%is_nan_x = OpIsNan %bool %x
%sqrt_val  = OpExtInst %f64 %glsl450 Sqrt %x
%result    = OpSelect %f64 %is_nan_x %x %sqrt_val
```

Three instructions instead of one. The `OpExtInst Sqrt` path is only reached
for non-NaN inputs where the spec's domain restriction is satisfied. `OpIsNan`
and `OpSelect` are both fully defined for NaN inputs.

Note: this does not resolve the RoundingMode uncertainty for `fsqrt` — that
is a separate gap (Vulkan spec mandates ≤1 ULP faithfully rounded for
extended instructions, not necessarily correctly rounded). The oracle_profile
for fsqrt remains `WithinULP(1)` regardless of this NaN guard.

**Upstream report:** None filed. The GLSL.std.450 spec gap is a design choice
(domain restriction language) rather than a clear bug. IEEE 754-2019 §9.2
intent is that sqrt(NaN) = NaN, but the GLSL spec doesn't say this explicitly.
**Date logged:** 2026-04-11
**Status:** OPEN — pending escalation or navigator ruling on whether the
composed-primitive workaround is mandated (as for VB-001) or whether NVIDIA's
observed behavior (sqrt(NaN) = NaN in practice) is sufficient to defer. The
capability matrix currently marks fsqrt NaNPropagation as ImplementationDefined.
If the workaround is mandated, the capability matrix entry and the Peak 7
lowering spec both need updating.

---

## VB-004 — PTX `min.f64` / `max.f64` default does NOT propagate NaN

**Vendor:** NVIDIA  
**Component:** PTX ISA — `min` and `max` instructions  
**Gap:** The PTX `min.f64` instruction follows C99 `fmin` semantics: when one operand is NaN, the result is the non-NaN operand. This is the **opposite** of I11's requirement (NaN must propagate). The NaN-propagating variant — `min.NaN.f64` (PTX ISA 7.5+, `.NaN` modifier) — exists but is not the default. Any PTX emission that uses bare `min.f64` or `max.f64` silently drops NaN, breaking cross-backend NaN semantics. This is the PTX counterpart to VB-001 (Vulkan OpFMin NaN undefined), but the resolution path is different: PTX has a correct native variant available.  
**Trek impact:** I11 (NaN propagates through every op on every backend). Peak 3 must never emit bare `min.f64` or `max.f64` for any `.tam` `min.f64` / `max.f64` op. The gap was noted as a "separate issue" in VB-001's upstream report field; this entry is the formal log.  
**P2 implication:** Composed-primitive workaround — two paths depending on target PTX ISA version. Path A (PTX ISA 7.5+, Ampere and later): emit `min.NaN.f64` / `max.NaN.f64` directly. Path B (pre-7.5 targets, Turing/Volta): compose from `setp.unordered` + `selp`, same principle as VB-001's SPIR-V sequence.  
**Workaround:** Peak 3 pathmaker emits `min.NaN.f64` / `max.NaN.f64` for all `.tam` min/max ops when targeting PTX ISA 7.5+. For pre-7.5 targets, emit the composed sequence:

```ptx
; min(a, b) — I11-compliant PTX for targets without .NaN modifier (pre-ISA 7.5)
setp.unordered.f64  %p_a_nan, %a, %a    ; p_a_nan = isnan(a)
setp.unordered.f64  %p_b_nan, %b, %b    ; p_b_nan = isnan(b)
setp.lt.f64         %p_lt, %a, %b       ; p_lt = (a < b), false if either NaN
selp.f64            %r0, %a, %b, %p_lt  ; r0 = p_lt ? a : b
selp.f64            %r1, %b, %r0, %p_b_nan  ; r1 = p_b_nan ? b (NaN) : r0
selp.f64            %result, %a, %r1, %p_a_nan  ; result = p_a_nan ? a (NaN) : r1
```

Capability matrix records the PTX ISA version boundary (7.5+) for Path A. For RTX 6000 Pro Blackwell (SM 100), PTX ISA 8.x is available — Path A applies.  
**Upstream report:** Not a bug — `min.NaN.f64` is a deliberate opt-in. PTX ISA reference §9.7.3 ("Floating-Point Instructions: min").  
**Date logged:** 2026-04-12  
**Status:** MITIGATED (spec known, workaround defined). Peak 3 campsite pre-flight must confirm `.NaN` modifier emission for all min/max ops. Pre-7.5 composed fallback documented above for target completeness.

---

## VB-003 — PTX `.contract` default is contract-everywhere

**Vendor:** NVIDIA  
**Component:** PTX assembler (all versions)  
**Gap:** The PTX default for floating-point multiply-add is `.contract true` — the assembler is free to fuse adjacent `fmul` + `fadd` pairs into a single FMA instruction unless the module explicitly opts out. This is the opposite of IEEE-754's requirement that fused operations be explicitly requested.  
**Trek impact:** I3 (no FMA contraction unless explicit). Any PTX we emit that uses `fadd` adjacent to `fmul` will be silently contracted on NVIDIA hardware unless we actively suppress it. This affects every arithmetic op in every tambear-libm kernel.  
**P2 implication:** Composed-primitive workaround. We do not rely on the PTX assembler's default contraction behavior. We emit `.contract false` unconditionally to suppress the vendor default, pinning the semantics of every arithmetic op from the spec.  
**Workaround:** Emit `.contract false` on every floating-point instruction in every PTX module we generate. This is the pathmaker's responsibility at the PTX emission layer (Peak 3). The suppressor applies module-wide; individual instruction annotations are not required if the module-level default is set correctly.  
**Upstream report:** None needed — this is a documented PTX design choice, not a bug. PTX ISA reference §9.7 ("Floating-Point Precision and Rounding").  
**Date logged:** 2026-04-11  
**Status:** MITIGATED — `.contract false` is a required invariant at the PTX emission layer (I3 enforcement). Pathmaker to confirm in campsite 3.x pre-flight.

---

## VB-002 — Vulkan `shaderDenormPreserveFloat64` is an optional feature

**Vendor:** Khronos (Vulkan) / GPU vendors  
**Component:** VkPhysicalDeviceVulkan12Properties  
**Gap:** `shaderDenormPreserveFloat64` is not required by the Vulkan core spec. A conformant Vulkan device may report `false`, meaning the hardware makes no guarantees about fp64 subnormal preservation — subnormals may be flushed to zero or otherwise mangled, at the implementation's discretion.  
**Trek impact:** I6 (no silent fallback) + ESC-001. On the RTX 6000 Pro Blackwell, `vulkaninfo` reports `shaderDenormPreserveFloat64 = false`. If we claim full cross-backend bit-exactness including subnormals, this breaks the claim on real hardware before a single kernel is written.  
**P2 implication:** Capability-matrix narrowing. There is no composed-primitive workaround for subnormal preservation — if the hardware flushes subnormals, no instruction sequence can prevent it. The resolution is to declare the hardware requirement explicitly (require `shaderDenormPreserveFloat64 = true` for full bit-exactness) and narrow the claim to the scope where P3 holds. This is the ESC-001 pattern: narrowing is not retreat, it is honest scoping.  
**Workaround (ESC-001 ruling):** Narrow the bit-exactness claim to normal fp64 range only. Phase 1 claims: "bit-exact for all normal fp64 inputs and outputs." Subnormal behavior requires `shaderDenormPreserveFloat64 = true` and is documented as an optional prerequisite in the README and capability matrix. The summit test (campsite 7.11) skips subnormal-producing inputs when the device reports `false` (not failed — not applicable). See `navigator/escalations.md` ESC-001 for full decision.  
**Upstream report:** N/A — this is a Vulkan spec design choice (optional feature), not a bug.  
**Date logged:** 2026-04-11  
**Status:** MITIGATED — scope of bit-exactness claim narrowed per ESC-001. README already amended. Capability matrix row notes optional prerequisite.

---

## VB-001 — Vulkan `OpFMin` / `OpFMax` NaN behavior is implementation-defined

**Vendor:** Khronos (Vulkan) / all SPIR-V implementations  
**Component:** SPIR-V core spec, OpFMin and OpFMax  
**Gap:** Core SPIR-V spec states: "If either operand is a NaN, the result is undefined." The GLSL.std.450 `FMin`/`FMax` instructions carry the same undefined-NaN semantics. `NMin`/`NMax` (GLSL.std.450) explicitly return the *non-NaN* operand when one input is NaN — the opposite of I11's requirement. The `SignedZeroInfNanPreserve` execution mode (SPV_KHR_float_controls) covers arithmetic ops (OpFAdd, OpFMul, etc.) but the float_controls spec does not extend this guarantee to OpFMin/OpFMax.  
**Trek impact:** I11 (NaN propagates through every op on every backend). There is no native SPIR-V instruction that implements I11-compliant min/max for fp64. Emitting OpFMin / OpFMax would make NaN behavior undefined on the Vulkan backend, breaking cross-backend correctness for any kernel receiving NaN inputs through a min/max op.  
**P2 implication:** Composed-primitive workaround. The P2 principle is that faithful lowering must pin the IEEE-754 semantics of every emitted op from the target spec alone. OpFMin's NaN semantics are not pinned by the SPIR-V spec ("undefined if either operand is NaN"), so OpFMin cannot appear in a faithful lowering of a `.tam` min op. The backend must instead compose from OpIsNan + OpFOrdLessThan + OpSelect, all of which have fully-defined behavior in the spec. This is not a capability-matrix narrowing (unlike ESC-001) because the composed sequence achieves full I11 compliance — no hardware feature is required, no claim is narrowed.  
**Workaround (ESC-002 ruling — Option 3):** Do not emit OpFMin or OpFMax. Compose min/max from well-defined primitives that individually have correct IEEE-754 semantics:

```
; min(a, b) — I11-compliant SPIR-V composition
%is_nan_a   = OpIsNan %bool %a
%is_nan_b   = OpIsNan %bool %b
%lt         = OpFOrdLessThan %bool %a %b   ; false if either is NaN
%min_non_nan = OpSelect %f64 %lt %a %b
%min_b_nan  = OpSelect %f64 %is_nan_b %b %min_non_nan
%result     = OpSelect %f64 %is_nan_a %a %min_b_nan
```

This sequence is correct by construction: `OpIsNan` is well-defined (always true when input is NaN), `OpFOrdLessThan` returns false when either operand is NaN (IEEE ordered comparison), `OpSelect` is a bitwise mux with no floating-point semantics. The composed result propagates NaN when either input is NaN.

This is a stronger guarantee than Option 1 (four-instruction workaround with a shared NaN constant): it avoids OpFMin entirely, so there is no dependency on undefined behavior at any point in the lowering.

**Note:** This is not a narrowing of the I11 claim. I11 is fully satisfied on Vulkan by taking ownership of NaN semantics at the lowering boundary rather than delegating to OpFMin.

**Upstream report:** No upstream fix possible — this is a SPIR-V spec design choice (undefined NaN for OpFMin/OpFMax). PTX equivalent: `min.NaN.f64` (NaN-propagating variant) is available in PTX and must be used for Peak 3 min/max emission (separate issue; PTX does not share this gap).  
**Date logged:** 2026-04-11  
**Status:** MITIGATED — ESC-002 formally ruled. Peak 7 pathmaker must emit the composed sequence for every `.tam` `min.f64` / `max.f64` op. Never emit OpFMin or OpFMax.
