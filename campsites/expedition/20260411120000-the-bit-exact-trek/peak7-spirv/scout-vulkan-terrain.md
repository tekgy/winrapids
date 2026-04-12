# Scout Report: Peak 7 — Vulkan/SPIR-V Terrain

*Scout: claude-sonnet-4-6 | Date: 2026-04-11*

Pre-reading for Peak 7. This machine is the target; all data below is from
`vulkaninfo` on this exact machine.

---

## Device confirmed

```
GPU id = 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
Vulkan Instance Version: 1.4.321
```

One GPU. One device. No multi-GPU complications.

---

## fp64 availability: CONFIRMED GREEN

```
VkPhysicalDeviceFeatures:
    shaderFloat64 = true
    shaderInt64   = true
```

fp64 compute shaders are available. I6 is NOT triggered — we do not need a
silent fallback. The pathmaker can proceed with fp64 without qualification.

---

## fp64 denorm behavior: WATCH THIS

```
shaderDenormPreserveFloat64    = false
shaderDenormFlushToZeroFloat64 = false
```

Both are false. This means the GPU makes NO guarantee about whether subnormals
are preserved or flushed to zero in fp64 shaders. The behavior is undefined
(implementation-dependent).

**Impact on tambear-libm:** The Peak 2 libm implementer needs to handle subnormals
correctly in the algorithm (not relying on hardware behavior). The `exp(-750)`
subnormal case is live — the driver may flush to zero or may preserve; we must
not rely on either. Flag this for the adversarial mathematician.

**Impact on cross-backend diffing:** The CPU interpreter will preserve subnormals
(IEEE 754 compliant). Vulkan may not. This creates a potential failure mode in
Peak 7's bit-exact test for subnormal inputs. Recommendation: document this
as an exception to the bit-exact claim for subnormals specifically, and add a
separate "subnormal behavior" test category.

---

## Rounding mode for fp64: CONFIRMED CORRECT

```
shaderRoundingModeRTEFloat64 = true   (round-to-nearest-even — IEEE default)
shaderRoundingModeRTZFloat64 = true   (round-toward-zero)
```

Round-to-nearest-even is available. This matches what our PTX code will use (`.rn`).
The SPIR-V `OpFAdd` with no special decoration defaults to RTE on NVIDIA. I4 is
satisfiable.

---

## SPIR-V version: 1.4

```
VK_KHR_spirv_1_4 : extension revision 1
```

SPIR-V 1.4 is supported. This is relevant because:
- SPIR-V 1.4 supports `OpEntryPoint` with storage buffer accessors directly
- `OpCopyLogical` and other convenience ops are available
- The `NoContraction` decoration (our I3 enforcement) is available in all SPIR-V versions

---

## NoContraction in SPIR-V: CONFIRMED IN CRATE

From `spirv-0.3.0+sdk-1.3.268.0/autogen_spirv.rs`:
```rust
pub enum Decoration {
    ...
    NoContraction = 42u32,
    ...
}
```

The `NoContraction` decoration is value `42` in the SPIR-V spec and is present in
the `spirv` crate we have available. When we emit an `OpFAdd` or `OpFMul`, we
also emit an `OpDecorate %result_id NoContraction` for each such instruction.

Note: `NoContraction` decorates the *result id* of the instruction, not the
instruction opcode. So for every fp add and fp mul in the shader:
```
%r1 = OpFAdd %f64 %a %b
OpDecorate %r1 NoContraction
```

This is more verbose than PTX's `.rn` but the semantics are the same.

---

## rspirv availability

The `rspirv` crate is NOT currently in the workspace `Cargo.lock` files. This is
expected — `tambear-tam-spirv` doesn't exist yet. When Peak 7 starts, adding
`rspirv` will be a dep escalation to Navigator (per `state.md` protocol).

What we DO have: the `spirv` crate (0.3.0) which provides the `Decoration`, `Op`,
and other enum types. `rspirv` is the builder on top. The trek-plan recommends
`ash` (raw Vulkan bindings) for the runtime side + `rspirv` for SPIR-V module
construction.

Alternative to rspirv: emit SPIR-V bytes by hand. The binary format is:
```
Word 0: magic = 0x07230203
Word 1: version (1.4 = 0x00010400)
Word 2: generator magic (our ID, can be anything)
Word 3: bound (max result id + 1)
Word 4: 0 (schema, reserved)
... instruction words
```

Each instruction is `(word_count << 16 | opcode, operand0, operand1, ...)`.
For a simple compute kernel with ~20 instructions, hand-writing the bytes is
feasible. `rspirv` makes it much less error-prone for larger programs.

Recommendation: use `rspirv` for Peak 7. Escalate the dep to Navigator when the
peak starts. The `spirv` crate (already in registry) provides the opcodes; `rspirv`
provides the structured builder. Both are pure Rust, no FFI, no vendor.

---

## Storage buffer layout: CRITICAL

The trek-plan warns about `std430` vs `std140`. Confirmed:
- `std430` is required for `float64` arrays in SSBO (Shader Storage Buffer Objects)
- `std140` adds per-field 16-byte alignment which would misalign fp64 elements
- Declare the binding with `layout(std430, binding = 0) buffer DataX { double data[]; }`
  in GLSL, or the equivalent in SPIR-V via `OpDecorate %array ArrayStride 8`

In SPIR-V, the array stride annotation is:
```
OpDecorate %_arr_f64 ArrayStride 8    ; 8 bytes per f64
```

Without this, the NVIDIA SPIR-V validator may accept the module but compute wrong
byte offsets.

---

## Workgroup size recommendation

For this GPU (RTX PRO 6000, Blackwell, sm_120):
- Warp size: 32 threads
- Typical occupancy sweet spot: 128–256 threads/workgroup for compute
- For the grid-stride accumulate pattern: 256 threads/workgroup is safe

Use a fixed workgroup size of 256. Do not make it tunable until Peak 7's basic
correctness is established.

In SPIR-V, declare it as:
```
OpExecutionMode %main LocalSize 256 1 1
```

---

## Ash vs vulkano

Trek-plan recommends `ash`. Confirmed correct for our use case:
- `ash` is a direct Vulkan binding — exposes raw `vkCreateDevice`,
  `vkCreateShaderModule`, `vkQueueSubmit`, etc.
- `vulkano` is higher-level and makes architectural decisions for us
- We need to control SPIR-V emission and pipeline creation precisely — `ash` is correct

`ash` is already in the Rust ecosystem with good Windows support. The ceremony
(instance → physical device → logical device → queue → pipeline → dispatch) is
verbose but mechanical.

---

## Summary for Peak 7 pathmaker

| Question | Answer |
|---|---|
| fp64 available? | YES (`shaderFloat64 = true`) |
| I6 triggers? | NO — fp64 is present, no fallback needed |
| NoContraction in SPIR-V? | YES — `Decoration::NoContraction = 42`, must decorate every fp result id |
| Denorm behavior? | UNDEFINED — neither preserve nor flush guaranteed; handle in libm, not hardware |
| SPIR-V version | 1.4 (VK_KHR_spirv_1_4) |
| Rounding mode | RTE available (`shaderRoundingModeRTEFloat64 = true`) |
| rspirv in workspace? | NOT YET — escalate to Navigator when Peak 7 starts |
| Storage buffer layout | std430 (`ArrayStride 8` per fp64 element) |
| Workgroup size | 256 (fixed, don't tune yet) |
| Runtime binding | `ash` (not `vulkano`) |
