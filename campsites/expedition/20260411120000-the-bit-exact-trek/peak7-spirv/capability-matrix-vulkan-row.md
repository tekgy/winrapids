# Capability Matrix — VulkanBackend Row (Draft)

*Naturalist pre-work for Peak 7 — 2026-04-11.*
*Input for pathmaker when Peak 7 starts. Pathmaker consolidates all backend rows.*

Device under test: **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**
Vulkan: 1.4.321 | SPIR-V: 1.4 | sm_120

Source for device capability values: `scout-vulkan-terrain.md` (vulkaninfo on
this machine). Source for spec semantics: SPV_KHR_float_controls HTML, Vulkan
SPIR-V appendix, SPIR-V core grammar JSON. Each uncertainty is flagged explicitly.

---

## Schema

Each entry has the shape:

```
status:           Supported | Partial | Unsupported
caveats:
  SubnormalHandling:   Preserve | FlushToZero | ImplementationDefined
  NaNPropagation:      Strict | WeakOnMinMax | ImplementationDefined
  RoundingMode:        RTE_Guaranteed | RTE_Default_Unverified | Unspecified
  FMAContraction:      Suppressed_NoContraction | Suppressed_RTE | Unspecified
order_strategies:    [list of implementable OrderStrategy variants]
oracle_profile:      BitExact | WithinULP(n) | OracleOnly
notes:               freeform
```

---

## Standing Rule: OpExtInst NaN Guard (navigator, 2026-04-12)

**Every GLSL.std.450 extended instruction (`OpExtInst`) emitted by the Vulkan
backend requires an explicit OpIsNan guard for I11 compliance.**

GLSL.std.450 restricts all extended instructions to real-valued domains. NaN
is outside every domain; the spec makes no guarantee about the output for NaN
inputs. `SignedZeroInfNanPreserve` applies to arithmetic instructions only and
does not explicitly extend to OpExtInst. This was established by VB-005
(GLSL.std.450 Sqrt, ruled 2026-04-12) and applies to all future OpExtInst
entries: log, exp, sin, cos, atan, and any other GLSL.std.450 instruction.

**Mandatory three-instruction prefix for every OpExtInst emission:**
```
%is_nan_x = OpIsNan %bool %x           ; (repeat for each input)
%raw_val  = OpExtInst %f64 %glsl450 <Opcode> %x   ; only reached for non-NaN
%result   = OpSelect %f64 %is_nan_x %x %raw_val   ; propagate NaN if input was NaN
```

For ops with two inputs (e.g., atan2, pow), check both inputs:
```
%is_nan_a = OpIsNan %bool %a
%is_nan_b = OpIsNan %bool %b
%either_nan = OpLogicalOr %bool %is_nan_a %is_nan_b
%raw_val  = OpExtInst %f64 %glsl450 <Opcode> %a %b
%result   = OpSelect %f64 %either_nan %a %raw_val   ; return first NaN input
```

**Do not rediscover this per function.** Every new OpExtInst entry in this
matrix must cite this rule and document its guard sequence. VB-005 is the
canonical reference in `vendor-bugs.md`.

---

## Phase 1 fp64 Ops — VulkanBackend

### `fadd.f64`, `fsub.f64`, `fmul.f64`, `fdiv.f64`

Lower to: `OpFAdd %f64`, `OpFSub %f64`, `OpFMul %f64`, `OpFDiv %f64`
With: `OpDecorate %result_id NoContraction` on every fp result.

```
status: Supported

caveats:
  SubnormalHandling: ImplementationDefined
    Device: shaderDenormPreserveFloat64 = false
            shaderDenormFlushToZeroFloat64 = false
    → GPU makes no guarantee. ESC-001 resolution applies: summit test
      (7.11) skips subnormal-producing inputs on this device.
    Note: To guarantee Preserve, emit execution mode:
      OpExecutionMode %main DenormPreserve 64
    but this requires shaderDenormPreserveFloat64 = true (false here).

  NaNPropagation: Strict (SignedZeroInfNanPreserve confirmed supported)
    SPV_KHR_float_controls: SignedZeroInfNanPreserve execution mode
    prevents optimizations that assume no NaN/inf. Without it, arithmetic
    ops may assume NaN cannot occur and optimize accordingly.
    Device property: shaderSignedZeroInfNanPreserveFloat64 = true
    CONFIRMED 2026-04-11 via vulkaninfo (VkPhysicalDeviceVulkan12Properties).
    SPIR-V emit to enable: OpExecutionMode %main SignedZeroInfNanPreserve 64
    I11 compliance for arithmetic (OpFAdd/OpFMul etc.) requires this mode.
    This device supports it — no ESC-003 needed.

  RoundingMode: RTE_Guaranteed
    Device: shaderRoundingModeRTEFloat64 = true (confirmed in terrain report)
    To activate: OpExecutionMode %main RoundingModeRTE 64
    Without explicit RTE execution mode, Vulkan 1.2+ on this device defaults
    to RTE for fp64 ops (confirmed by terrain report: "OpFAdd with no special
    decoration defaults to RTE on NVIDIA"). But explicit emission is required
    for cross-device portability.

  FMAContraction: Suppressed_NoContraction
    Mechanism: OpDecorate %result_id NoContraction on every OpFAdd, OpFMul.
    Confirmed: spirv crate has NoContraction = 42u32 (Decoration enum).
    Emit per result id, not per instruction. See terrain report §NoContraction.
    Note: more verbose than PTX's per-instruction .rn, same guarantee.

order_strategies: [SequentialLeft, TreeFixedFanout(2)]
  Both are implementable: SequentialLeft is the naive grid-stride single-thread
  accumulation; TreeFixedFanout(2) uses workgroup shared memory + bar.sync
  (OpControlBarrier in SPIR-V). Phase 1 uses SequentialLeft + atomicAdd
  (I5 violation, same as PTX Phase 1). Peak 6 replaces with TreeFixedFanout.

oracle_profile: BitExact (with CPU)
  Arithmetic ops with NoContraction + RTE should be bit-exact with CPU
  interpreter for normal fp64 inputs. Subnormals are not bit-exact (ESC-001).
  See oracle_profile for ReduceBlockAdd below — reduction order is the blocker.

notes:
  - Must emit OpExecutionMode %main RoundingModeRTE 64 explicitly.
  - Must emit OpExecutionMode %main SignedZeroInfNanPreserve 64.
    Device support confirmed: shaderSignedZeroInfNanPreserveFloat64 = true.
  - std430 layout required: OpDecorate %_arr_f64 ArrayStride 8.
  - All four ops confirmed available for fp64 on this device.
```

---

### `fsqrt.f64`

Lower to: `OpExtInst %f64 %glsl450 Sqrt %x` (GLSL.std.450 instruction 31)
With: `OpDecorate %result_id NoContraction`

```
status: Supported

caveats:
  SubnormalHandling: ImplementationDefined (same as fadd — ESC-001 applies)

  NaNPropagation: Strict (via mandatory OpIsNan guard — VB-005, ruled 2026-04-12)
    GLSL.std.450 Sqrt restricts domain to non-negative inputs; NaN is outside
    the specified domain with no spec guarantee about Sqrt(NaN).
    SignedZeroInfNanPreserve applies to arithmetic instructions and does not
    explicitly extend to OpExtInst. Navigator ruling: mandate the OpIsNan guard.
    Observed NVIDIA behavior is not a spec guarantee; lowering must pin
    semantics from the spec alone (tightened P2).

    MANDATORY EMIT SEQUENCE for every .tam fsqrt.f64 on Vulkan:
      %is_nan_x = OpIsNan %bool %x
      %sqrt_val  = OpExtInst %f64 %glsl450 Sqrt %x
      %result    = OpSelect %f64 %is_nan_x %x %sqrt_val

    OpExtInst Sqrt is only reached for non-NaN inputs. Correct by construction.

  RoundingMode: Unspecified (spec allows ≤1 ULP for extended instructions)
    GLSL.std.450 Sqrt is specified as "correctly rounded" (0 ULP) for real
    inputs, but the Vulkan spec allows "faithfully rounded" (1 ULP) for
    extended instructions. With RTE execution mode, sqrt.f64 on NVIDIA is
    correctly rounded in practice, but the spec does not mandate this.

  FMAContraction: N/A (square root is not a multiply-add chain)

order_strategies: N/A (single-value op, no accumulation)

oracle_profile: BitExact (NaN inputs) | WithinULP(1) (normal inputs)
  NaN inputs: BitExact. OpIsNan guard guarantees NaN-in → NaN-out by
  construction; identical to CPU interpreter. No ULP tolerance needed.
  Normal inputs: WithinULP(1). Vulkan spec allows ≤1 ULP ("faithfully
  rounded") for extended instructions; spec does not mandate correctly
  rounded. In practice NVIDIA gives correctly rounded, but WithinULP(1)
  is the spec-safe cross-backend tolerance. RoundingMode uncertainty
  remains open — does not affect NaN case.

notes:
  - Use GLSL.std.450 Sqrt (opcode 31), NOT a custom polynomial.
  - ALWAYS emit the three-instruction OpIsNan guard (VB-005). Never emit
    bare OpExtInst Sqrt for fsqrt — NaN behavior is not spec-guaranteed.
  - NaN oracle test: standard injection, special_value_failures == 0 required.
    No per-backend exemption needed (guard is always emitted).
```

---

### `fneg.f64`, `fabs.f64`

Lower to: `OpFNegate %f64`, `OpExtInst %f64 %glsl450 FAbs %x`

```
status: Supported

caveats:
  SubnormalHandling: ImplementationDefined (sign flip / magnitude on subnormals;
    both ops are bitwise in IEEE 754, so subnormal flushing would be surprising
    but is not ruled out by the device properties)
  NaNPropagation: NaN input → NaN output for both (sign flip / abs of NaN
    produces a NaN with modified sign bit; this is bit-level, no comparison)
  RoundingMode: N/A (no rounding)
  FMAContraction: N/A

order_strategies: N/A
oracle_profile: BitExact
  Both are single-bit-field operations on the IEEE 754 representation.
  Should be bit-exact across all backends. Confirm with hard case: fneg(NaN),
  fabs(-NaN) — CPU must match Vulkan.
```

---

### `fcmp_gt.f64`, `fcmp_lt.f64`, `fcmp_eq.f64`

Lower to: `OpFOrdGreaterThan %bool`, `OpFOrdLessThan %bool`, `OpFOrdEqual %bool`

```
status: Supported

caveats:
  SubnormalHandling: ImplementationDefined
    Comparison of subnormals may behave differently if subnormals are
    flushed to zero before comparison. A subnormal and zero would then
    compare equal when they should compare greater-than.

  NaNPropagation: StrictOnArith_WeakOnComparison
    IMPORTANT — this is the I11-relevant finding:
    OpFOrd* comparisons (Ordered comparisons) return false if EITHER
    operand is NaN. This is IEEE 754 correct behavior (ordered comparison
    with NaN is always false). It is NOT a NaN-propagation failure.
    The I11 concern is not with fcmp itself but with downstream select.f64:
      if fcmp_gt(NaN, x) returns false, then select.f64(false, NaN_branch, x_branch)
      returns x_branch — NaN is silently dropped.
    The fix is at the .tam IR level (explicit is_nan check before comparison)
    not at the Vulkan level. Vulkan is IEEE-correct here; the IR recipe is wrong.
    See pitfall P20 (NaN silently ignored by comparison-based expressions).

  RoundingMode: N/A
  FMAContraction: N/A
  SignedZeroInfNanPreserve: Irrelevant for comparisons (applies to arithmetic).

order_strategies: N/A
oracle_profile: BitExact
  Boolean result. Either both backends agree or they don't. NaN inputs are
  the only edge case; use the subnormal/NaN hard-case generators from Peak 4.

notes:
  - Use OpFOrd* (ordered) not OpFUnord* (unordered).
    OpFOrdEqual returns false for NaN inputs (correct IEEE behavior).
    OpFUnordEqual returns true if EITHER operand is NaN (rarely what we want).
  - The select.f64 NaN-drop is a recipe problem, not a Vulkan problem.
```

---

### `select.f64`, `select.i32`

Lower to: `OpSelect %f64` (or `%i32`)

```
status: Supported

caveats:
  NaNPropagation: PassThrough
    OpSelect is not an arithmetic op — it just copies one of two values
    based on a predicate. If the selected value is NaN, NaN is the result.
    If the predicate was produced by a comparison that returned false due
    to NaN input, the wrong branch fires. This is the fcmp_gt issue above.

  SubnormalHandling: PassThrough (select copies bits, doesn't operate on them)
  RoundingMode: N/A
  FMAContraction: N/A

order_strategies: N/A
oracle_profile: BitExact
```

---

### `ReduceBlockAdd.f64`

Lower to: workgroup shared memory accumulation + `OpAtomicFAddEXT` or tree reduce.

```
status: Partial

caveats:
  SubnormalHandling: ImplementationDefined (same as arithmetic)

  NaNPropagation: ImplementationDefined
    Atomic operations on fp64 have additional uncertainty. OpAtomicFAddEXT
    (VK_EXT_shader_atomic_float) — NaN behavior in atomic add is not
    specified by Vulkan.

  RoundingMode: ImplementationDefined
    Atomic fp64 add uses the device's hardware atomic unit, which may
    use a different rounding path than the shader cores. RTE execution
    mode may not apply to atomic operations.

  FMAContraction: N/A (atomic add is a single operation)

  OrderNondeterminism: YES — I5 violation
    atomicAdd is non-commutative in accumulated order. Same issue as PTX.
    Peak 6 replaces with tree reduce. Vulkan tree reduce uses:
      OpControlBarrier (workgroup barrier = bar.sync equivalent)
      Shared memory (OpTypePointer Workgroup + OpVariable)
      Stride-halving loop (same tree pattern as Phase 6 PTX)

  Extension dependency: VK_EXT_shader_atomic_float
    Required for OpAtomicFAddEXT on fp64 buffers.
    CONFIRMED 2026-04-11: VkPhysicalDeviceShaderAtomicFloatFeaturesEXT present.
      shaderBufferFloat64Atomics   = true
      shaderBufferFloat64AtomicAdd = true
    Extension is available and fp64 atomic add is supported on this device.
    Preferred design is still tree-reduce-to-one (avoids atomics entirely,
    I5-compliant without the extension) — extension availability just means
    Phase 1 is not blocked if atomicAdd is used temporarily.

order_strategies: [TreeFixedFanout(2)]
  SequentialLeft is not implementable as a single atomic (would require
  all threads to serialize). TreeFixedFanout(2) with fixed workgroup barrier
  ordering is the correct Phase 6 implementation.
  The rfa_bin_exponent_aligned strategy (Peak 6 RFA) has no Vulkan-specific
  blocker beyond the shared memory + atomic considerations above.

oracle_profile: OracleOnly (Phase 1) → BitExact (Phase 6 with TreeFixedFanout)
  Phase 1 atomicAdd is non-deterministic → cannot claim BitExact.
  Phase 6 tree reduce with fixed-order barriers → BitExact with CPU/PTX
  (for normal fp64 inputs; subnormals remain ESC-001 territory).

notes:
  - Query VK_EXT_shader_atomic_float availability before Phase 1 implementation.
  - Alternative to avoid the extension: reduce within workgroup using shared
    memory + barrier (no extension needed), then write partial sums to output
    buffer, then fold on CPU. This is the RFA design pattern anyway.
  - The workgroup size must match between the barrier and the tree stride
    computation. Fixed at 256 (per terrain report recommendation).
```

---

### `.tam min.f64` / `.tam max.f64` → SPIR-V composition (future — not in Phase 1 IR)

**These ops are NOT in the Phase 1 IR op set.** This entry is a forward
specification for when min/max ops are added. ESC-002 is DECIDED (filed
2026-04-11, team-lead ruling: Option 3 — never emit OpFMin or OpFMax).
Logged as VB-001 in `vendor-bugs.md`.

```
status: Supported (composition required — never emit OpFMin/OpFMax)

caveats:
  NaNPropagation: Strict (via six-instruction composed sequence)
    ESC-002 (team-lead, 2026-04-11): Option 3 selected. Do not emit OpFMin
    or OpFMax. These instructions have undefined NaN behavior in SPIR-V core
    ("if either operand is a NaN, the result is undefined") and cannot appear
    in a faithful lowering of a .tam min/max op. Compose from well-defined
    primitives that individually have correct IEEE-754 specs.

    Background: GLSL.std.450 NMin/NMax explicitly return the non-NaN operand
    (wrong direction for I11). SignedZeroInfNanPreserve does not extend to
    OpFMin/OpFMax per the float_controls spec. No native SPIR-V instruction
    provides I11-compliant min/max.

    MANDATORY EMIT SEQUENCE for every .tam min.f64 on Vulkan:
      %is_nan_a    = OpIsNan %bool %a
      %is_nan_b    = OpIsNan %bool %b
      %lt          = OpFOrdLessThan %bool %a %b   ; false if either is NaN
      %min_non_nan = OpSelect %f64 %lt %a %b
      %min_b_nan   = OpSelect %f64 %is_nan_b %b %min_non_nan
      %result      = OpSelect %f64 %is_nan_a %a %min_b_nan

    For max(a, b): replace OpFOrdLessThan with OpFOrdGreaterThan and
    adjust OpSelect operand order accordingly.

    This sequence is correct by construction: OpIsNan is always well-defined,
    OpFOrdLessThan returns false when either operand is NaN (IEEE ordered
    comparison), OpSelect is a bitwise mux with no floating-point semantics.
    No dependency on undefined behavior at any point.

    PTX contrast: min.NaN.f64 (one instruction, native I11 compliance, ISA 7.5+).
    CPU contrast: explicit is_nan guards in interpreter (same logical structure).

  SubnormalHandling: ImplementationDefined (same as arithmetic — ESC-001)
  RoundingMode: N/A (min/max is exact, no rounding)
  FMAContraction: N/A

order_strategies: N/A (single-element comparison, no accumulation)
oracle_profile: BitExact
  Composed sequence with no undefined ops → NaN propagates correctly.
  Non-NaN inputs: result is exact (comparison + select, no rounding).
  BitExact with CPU interpreter and PTX backend.

notes:
  - NEVER emit OpFMin, OpFMax, GLSL.std.450 FMin, or GLSL.std.450 FMax.
    All have undefined NaN behavior. This is an absolute prohibition.
  - NEVER emit GLSL.std.450 NMin/NMax: explicitly suppress NaN (wrong direction).
  - The six-instruction sequence is the only correct Vulkan lowering.
  - Pathmaker must document in the .tam IR op entry for min/max:
    "Vulkan backend composes from OpIsNan + OpFOrdLessThan + OpSelect.
     OpFMin/OpFMax are never emitted."
  - ESC-002: navigator/escalations.md. Status: DECIDED (Option 3).
  - VB-001: vendor-bugs.md. Status: MITIGATED.
```

---

## Device Property Status (ESC-002 pre-flight queries — RESOLVED 2026-04-11)

All four ESC-002 pre-flight queries confirmed via `vulkaninfo` on RTX PRO 6000 Blackwell.

| Property | Status | Value |
|---|---|---|
| `shaderSignedZeroInfNanPreserveFloat64` | CONFIRMED | **true** — SignedZeroInfNanPreserve 64 execution mode CAN be emitted. I11 on arithmetic ops achievable. |
| `VK_EXT_shader_atomic_float` | CONFIRMED | **Present** — `shaderBufferFloat64AtomicAdd = true`. OpAtomicFAddEXT on fp64 supported. |
| `shaderDenormPreserveFloat64` | CONFIRMED (ESC-001) | **false** — Subnormal handling undefined. ESC-001 resolution applies. |
| `shaderDenormFlushToZeroFloat64` | CONFIRMED (ESC-001) | **false** — GPU makes no denorm guarantee either way. |
| `shaderRoundingModeRTEFloat64` | CONFIRMED (terrain report) | **true** — RTE execution mode supported and recommended. |

**All pre-flight queries complete. No ESC-003 needed. Campsite 7.1 may proceed when
Peak 7 is scheduled.** The I11 on arithmetic path is achievable via
`OpExecutionMode %main SignedZeroInfNanPreserve 64` on this device.

---

## Summary: Vulkan Backend Status vs Invariants

| Invariant | Vulkan Status | Notes |
|---|---|---|
| I1 (no vendor math) | CLEAN | All ops implemented from SPIR-V primitives, no vendor extensions in libm path |
| I3 (no FMA contraction) | CLEAN | NoContraction decoration on every fp result id |
| I4 (no fp reordering) | CLEAN | RTE execution mode + NoContraction |
| I5 (deterministic reductions) | VIOLATION in Phase 1 | atomicAdd non-deterministic; Peak 6 tree-reduce fixes |
| I6 (no silent fallback) | CLEAN | fp64 confirmed (`shaderFloat64 = true`) |
| I8 (first-principles libm) | CLEAN | SPIR-V arithmetic ops, no rcp/sqrt.approx equivalents in core |
| I11 (NaN propagation) | ACHIEVABLE | Arithmetic: SignedZeroInfNanPreserve 64 confirmed supported. Min/max: six-instruction composition (ESC-002 Option 3, VB-001) |
| ESC-001 (subnormals) | SCOPED | Both denorm flags false; summit test 7.11 skips subnormal inputs on this device |
| ESC-002 (NaN in min/max) | DECIDED | Option 3: never emit OpFMin/OpFMax. Composed from OpIsNan + OpFOrdLessThan + OpSelect. Not a Phase 1 blocker. |

---

## Research Uncertainties — Status

All ESC-002 pre-flight queries resolved 2026-04-11. Remaining uncertainty:

**Open (spec-level, not device-level):**

1. **SignedZeroInfNanPreserve scope for OpExtInst (fsqrt).** RESOLVED 2026-04-12.
   Navigator ruled (VB-005): OpIsNan guard mandatory. The spec ambiguity is
   resolved by composition — the guard takes ownership of NaN semantics at the
   lowering boundary. Capability matrix fsqrt entry updated: NaNPropagation →
   Strict. Remaining uncertainty for fsqrt is RoundingMode only (Vulkan spec
   allows ≤1 ULP for extended instructions; oracle_profile stays WithinULP(1)).

2. **OpFMin/OpFMax NaN semantics under SignedZeroInfNanPreserve.** Confirmed
   not covered (ESC-002 finding). Resolved: OpFMin/OpFMax are never emitted
   (ESC-002 Option 3, VB-001). Composed from OpIsNan + OpFOrdLessThan +
   OpSelect. Not an open question — a documented known prohibition.

**Confirmed (sourced from vulkaninfo or terrain report or spec):**

- `shaderSignedZeroInfNanPreserveFloat64 = true` (confirmed 2026-04-11)
- `VK_EXT_shader_atomic_float` present, `shaderBufferFloat64AtomicAdd = true` (confirmed 2026-04-11)
- NoContraction decoration (42u32): spirv crate + spec
- `shaderRoundingModeRTEFloat64 = true`: terrain report
- `shaderFloat64 = true`: terrain report
- Both denorm flags false: terrain report + ESC-001
- OpFOrd* comparison NaN behavior: IEEE 754 ordered comparison (returns false for NaN — correct, not an I11 issue)
- SPIR-V 1.4 on this device: terrain report
- std430 layout (ArrayStride 8): terrain report
