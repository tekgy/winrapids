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

  NaNPropagation: Strict (conditional on SignedZeroInfNanPreserve)
    SPV_KHR_float_controls: SignedZeroInfNanPreserve execution mode
    prevents optimizations that assume no NaN/inf. Without it, arithmetic
    ops may assume NaN cannot occur and optimize accordingly.
    Device property: shaderSignedZeroInfNanPreserveFloat64 — NOT QUERIED
    in terrain report (see uncertainty flag below).
    SPIR-V emit to enable: OpExecutionMode %main SignedZeroInfNanPreserve 64
    I11 compliance for arithmetic (OpFAdd/OpFMul etc.) requires this mode.

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
  - Must emit OpExecutionMode %main SignedZeroInfNanPreserve 64
    (pending confirmation that device supports it — query shaderSignedZeroInfNanPreserveFloat64).
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

  NaNPropagation: ImplementationDefined
    GLSL.std.450 Sqrt is not an arithmetic op in the OpFAdd sense. The
    SignedZeroInfNanPreserve mode's application to extended instructions
    (OpExtInst) is not explicitly specified in SPV_KHR_float_controls.
    UNCERTAINTY FLAG: unknown whether SignedZeroInfNanPreserve covers
    OpExtInst operations or only core SPIR-V arithmetic ops.

  RoundingMode: Unspecified
    GLSL.std.450 Sqrt is specified as "correctly rounded" (0 ULP) for
    real inputs, but the Vulkan spec says "faithfully rounded" (1 ULP)
    for extended instructions unless shaderRoundingModeRTEFloat64 is
    enabled. With RTE execution mode, sqrt.f64 on NVIDIA is correctly
    rounded in practice, but the Vulkan spec does not mandate this.

  FMAContraction: N/A (square root is not a multiply-add chain)

order_strategies: N/A (single-value op, no accumulation)

oracle_profile: WithinULP(1)
  Cannot claim BitExact for sqrt across backends — Vulkan spec mandates
  faithfully rounded (≤1 ULP), not correctly rounded, for extended
  instructions. In practice NVIDIA gives correctly rounded. Use WithinULP(1)
  for cross-backend tolerance; use mpmath for oracle comparison.

notes:
  - Use GLSL.std.450 Sqrt (opcode 31), NOT a custom polynomial —
    the extended instruction is the right path here since it's
    within Vulkan's guaranteed accuracy bounds.
  - sqrt(NaN) behavior: UNCERTAINTY FLAG — not confirmed from spec.
    CPU interpreter returns NaN; Vulkan may differ without SignedZeroInfNanPreserve.
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
    UNCERTAINTY FLAG: not confirmed whether this extension is available
    on this device. Scout terrain report does not query it.
    Alternative: tree reduce to a single thread, then non-atomic store.
    This avoids the extension entirely and is I5-compliant when combined
    with Peak 6's fixed-order reduction.

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

### `OpFMin` / `OpFMax` (future — not in Phase 1 IR)

**These ops are NOT in the Phase 1 IR op set.** This entry is a forward
specification for when min/max ops are added. ESC-002 is OPEN (filed
2026-04-11, navigator decision: Option 1 — mandate workaround).

```
status: Supported (workaround required — ESC-002 decision: Option 1)

caveats:
  NaNPropagation: Strict (via mandatory four-instruction workaround)
    ESC-002 (navigator, 2026-04-11): Option 1 selected — mandate the
    four-instruction NaN-safe sequence for all Vulkan min/max emissions.

    Background: SPIR-V OpFMin/OpFMax — "If either operand is a NaN,
    the result is undefined." GLSL.std.450 NMin/NMax explicitly return
    the non-NaN operand (wrong direction for I11). SignedZeroInfNanPreserve
    does not extend to OpFMin/OpFMax per the float_controls spec.
    No native SPIR-V instruction provides I11-compliant min/max.

    MANDATORY EMIT PATTERN for every .tam min.f64 / max.f64 on Vulkan:
      %is_nan_a  = OpIsNan %bool %a
      %is_nan_b  = OpIsNan %bool %b
      %either_nan = OpLogicalOr %bool %is_nan_a %is_nan_b
      %raw_min   = OpFMin %f64 %a %b          (or OpFMax for max)
      %result    = OpSelect %f64 %either_nan %nan_const %raw_min

    Cost: 4 extra instructions per min/max call. Correct by construction.
    PTX contrast: min.NaN.f64 (one instruction, native I11 compliance).

  SubnormalHandling: ImplementationDefined (same as arithmetic — ESC-001)
  RoundingMode: N/A
  FMAContraction: N/A

order_strategies: N/A (single-element comparison, no accumulation)
oracle_profile: BitExact
  With the workaround, NaN inputs propagate to NaN output on all backends.
  Non-NaN inputs: min/max is exact (no rounding), BitExact with CPU.

notes:
  - Do NOT use bare OpFMin/OpFMax or GLSL.std.450 FMin/FMax: undefined NaN.
  - Do NOT use GLSL.std.450 NMin/NMax: these suppress NaN (wrong direction).
  - Pathmaker must document in the .tam IR op entry for min/max:
    "Vulkan backend emits the four-instruction NaN-safe sequence."
  - ESC-002 filed: navigator/escalations.md. Status: OPEN.
```

---

## Open Device Queries (for Peak 7 pre-flight checklist)

The following properties were NOT queried in `scout-vulkan-terrain.md` and must
be checked before Peak 7 campsite 7.1 begins:

| Property | Why needed | Expected value |
|---|---|---|
| `shaderSignedZeroInfNanPreserveFloat64` | I11 compliance for arithmetic ops | Unknown — query required |
| `VK_EXT_shader_atomic_float` (extension) | AtomicFAdd for Phase 1 ReduceBlockAdd | Unknown — query required |
| `shaderDenormPreserveFloat64` (execution mode) | ESC-001 — already known: false | false (confirmed) |
| `maxComputeWorkGroupSize[0]` | Must be ≥ 256 for fixed workgroup size | Expected ≥ 1024 for Blackwell |

To query, run: `vulkaninfo --json | python -c "import sys,json; d=json.load(sys.stdin); ..."` 
or check `vulkan.gpuinfo.org` for this device model.

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
| I11 (NaN propagation) | CONDITIONAL | Requires SignedZeroInfNanPreserve for arithmetic; OpFMin/OpFMax NaN undefined |
| ESC-001 (subnormals) | OPEN | Both denorm flags false; summit test 7.11 must skip subnormal inputs on this device |
| ESC-002 (NaN in min/max) | LATENT | Not a blocker today (no min/max in Phase 1 IR); escalate before min/max ops are added |

---

## Research Uncertainties — Flagged Honestly

Items marked UNCERTAINTY FLAG above, summarized:

1. **`shaderSignedZeroInfNanPreserveFloat64` on this device.** The terrain report
   didn't query it. I11 compliance for OpFAdd/OpFMul requires this execution mode
   to be emitted, and emitting it requires the device to support the property.
   Must be confirmed before campsite 7.3 (arithmetic op emit).

2. **SignedZeroInfNanPreserve scope for OpExtInst (fsqrt).** The float_controls
   spec text applies to arithmetic instructions. Whether it applies to extended
   instructions (GLSL.std.450 Sqrt) is not confirmed. NVIDIA in practice
   preserves NaN through sqrt, but the spec doesn't mandate it.

3. **VK_EXT_shader_atomic_float availability.** Required for Phase 1 atomicAdd
   on fp64 output slots. If unavailable, Phase 1 must use tree-reduce-to-one
   approach (which is the better design anyway — no atomics needed).

4. **OpFMin/OpFMax NaN semantics under SignedZeroInfNanPreserve.** The spec
   text does not explicitly cover this case. This is the ESC-002 candidate.

Items I am confident about (sourced from terrain report or confirmed spec):

- NoContraction decoration (42u32): confirmed in spirv crate, confirmed in spec
- RoundingModeRTE on this device: confirmed (shaderRoundingModeRTEFloat64 = true)
- fp64 shader support: confirmed (shaderFloat64 = true)
- Subnormal handling undefined: confirmed (both flags false, ESC-001)
- OpFOrd* comparison NaN behavior: IEEE 754 ordered comparison, returns false for NaN
  (this is correct and expected behavior, not an I11 issue)
- SPIR-V 1.4 on this device: confirmed
- std430 layout requirement: confirmed (ArrayStride 8 for fp64 elements)
