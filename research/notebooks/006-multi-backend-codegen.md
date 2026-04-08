# Lab Notebook 006: Multi-Backend Kernel Codegen + ComputeEngine

**Date**: 2026-03-31
**Author**: Pathmaker
**Branch**: main
**Status**: Active
**Hardware**: NVIDIA RTX PRO 6000 Blackwell, CUDA 13.1

---

## Context & Motivation

Team lead directive: "a .tbs chain compiles to GPU kernels at runtime, dispatched to whatever backend is available. Same .tbs file -> CUDA on Windows NVIDIA, Vulkan on AMD/Intel, Metal on MacBook."

Step 1: build a codegen module that takes the same phi expression and emits kernel source for CUDA C or WGSL. This is the translation surface — same semantics, different syntax.

---

## Design

### CodegenTarget enum

```rust
pub enum CodegenTarget {
    CudaC,  // NVRTC -> PTX
    Wgsl,   // wgpu/naga -> SPIR-V (Vulkan), MSL (Metal), HLSL (DX12)
}
```

WGSL is the universal backend language: naga cross-compiles it to SPIR-V, MSL, and HLSL. So two codegen targets cover all four GPU APIs.

### Translation surface

| Concept | CUDA C | WGSL |
|---------|--------|------|
| Thread index | `blockIdx.x * blockDim.x + threadIdx.x` | `gid.x` (builtin) |
| Kernel decl | `extern "C" __global__ void name(...)` | `@compute @workgroup_size(256) fn name(...)` |
| Buffer params | `const double* __restrict__ values` | `@group(0) @binding(N) var<storage, read> values: array<f32>` |
| Float atomics | `atomicAdd(&out[g], val)` (native f64) | CAS loop on `atomic<u32>` with `bitcast<f32>` |
| Barriers | `__syncthreads()` | `workgroupBarrier()` |
| Shared memory | `extern __shared__ double shmem[]` | `var<workgroup> shmem: array<f32, 256>` |
| Scalar type | `double` (f64) | `f32` |
| Math funcs | `fabs`, `log`, `exp`, `sqrt` | `abs`, `log`, `exp`, `sqrt` |

### Phi expression translation

Simple arithmetic (`v * v`, `v - r`, `1.0`) is identical in both languages. Only `fabs` -> `abs` needs translation. The translate_phi_to_wgsl function handles this.

### Precision

CUDA uses f64 throughout. WGSL uses f32 (f64 atomics not available in standard WGSL). This is a known precision gap — acceptable for the multi-backend portability goal. Production CUDA path retains f64.

---

## Implementation

**File**: `src/codegen.rs`

Six codegen functions, each with CUDA C and WGSL backends:

| Function | Operation | Entry point |
|----------|-----------|-------------|
| `emit_scatter(phi, target)` | `output[key[i]] += phi(v, r, g)` | `scatter_phi` |
| `emit_scatter_multi(phis, target)` | N fused outputs, one pass | `scatter_multi_phi` |
| `emit_map(phi, target)` | `output[i] = phi(v[i])` | `map_phi_kernel` |
| `emit_map2(phi, target)` | `output[i] = phi(a[i], b[i])` | `map_phi2_kernel` |
| `emit_reduce_sum(target)` | Tree reduction + atomic partials | `reduce_sum` |

### WGSL scatter: CAS atomic pattern

WGSL lacks native float atomics. The standard pattern (proven in tambear-wgpu benchmarks):

```wgsl
fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&output[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[idx], old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}
```

### WGSL multi-scatter: per-output bindings

Each of N outputs gets its own `@binding(3+i)` with a dedicated `atomic_add_f32_outN` helper. Params binding follows at `3+N`. Tested for N=2 and N=3.

---

## Results

### Codegen module (`src/codegen.rs`)

**20 tests pass** (17 original + 3 masked scatter):
- Scatter: CUDA structure, WGSL structure, phi translation, cross-target entry match
- Scatter multi: CUDA 3-output params+atomics, WGSL 3-output bindings, contiguous binding indices
- **Scatter masked: CUDA u64 bitmask, WGSL u32 bitmask, entry match**
- Map: CUDA basic, WGSL basic (no atomics)
- Map2: CUDA basic, WGSL basic
- Reduce sum: CUDA shared mem + syncthreads, WGSL workgroup var + barrier
- Cross-target: all 5 standard phis generate valid source for both targets

### ComputeEngine (`src/compute_engine.rs`)

**26 tests** (21 CPU + 5 CUDA):

CPU path (phi closures, no GPU):
- scatter_sum, scatter_sum_sq, scatter_count, scatter_centered
- scatter_multi (sum+count, 3 phis)
- **scatter_masked (selective, all-pass, none-pass)**
- map (identity, square, log)
- map2 (add, mul)

CUDA path (codegen → NVRTC → dispatch through TamGpu):
- scatter_sum, scatter_multi (3 phis fused), **scatter_masked**
- map_square, map2_add

**Full crate: 224/224 lib tests + 5 CUDA ignored + 19 doc tests = 248 total.**

---

## Key design: n baked as #define

TamGpu.dispatch only passes buffer pointers — no scalar args. CUDA kernels need `int n`. Solution: bake n as `#define PARAM_N {n}` in the kernel source. The kernel takes only pointer params.

```c
#define PARAM_N 10000
extern "C" __global__ void scatter_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    double* __restrict__ output
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) { ... }
}
```

Cache key: `(phi_expr, n)`. Typical sessions have constant n (dataset size) → one compile per phi.

---

## Discussion

### What this proves

1. **Codegen**: Same phi → CUDA C or WGSL. All six operation types (scatter, scatter_multi, scatter_masked, map, map2, reduce_sum). Structurally matches existing scatter_jit.rs.

2. **ComputeEngine**: Full dispatch layer. CPU closures for testing, CUDA JIT for production. Same API regardless of backend. Proven end-to-end on RTX PRO 6000: codegen → NVRTC → dispatch → correct results.

3. **Masked scatter**: Both codegen and ComputeEngine. CUDA uses u64 bitmask, WGSL uses u32. CPU evaluator checks bits directly.

### AccumulateEngine coupling

AccumulateEngine currently uses ScatterJit for: scatter_phi, scatter_multi_phi, scatter_phi_masked, scatter_phi_dual_target, map_phi. ComputeEngine now covers all except dual_target. The path to multi-backend accumulate: swap ScatterJit for ComputeEngine in AccumulateEngine.

### CpuBackend tiled_accumulate reference implementation

Added `tiled_accumulate` entry to CpuBackend (`tam-gpu/src/cpu.rs`). Design:

- `compile()` parses the op name from the kernel source's first-line comment (`// Tiled accumulation kernel for operator: dot_product`)
- `dispatch()` reads A(M×K), B(K×N), dims([M,N,K]) from buffers, runs naive triple-loop
- Supports: `dot_product`, `outer_product`, `l2_distance`, `covariance`
- `TiledEngine::run()` routes `ShaderLang::Cpu` through the f64 path (same as CUDA)

**New tests**: 8 in tam-gpu (unit), 6 in winrapids-tiled (TiledEngine integration). All proven: GEMM, identity, vector×matrix, self-distance, covariance, kernel cache.

This means TiledEngine-dependent code (AccumulateEngine, experiment0, linear/logistic training, KNN, clustering) can now run on CpuBackend for testing without GPU hardware.

### What's next

1. **VulkanBackend for tam-gpu**: Wrap wgpu behind TamGpu trait. WGSL codegen ready. (Navigator)
2. **Wire AccumulateEngine through ComputeEngine**: Replace self.scatter (ScatterJit) with ComputeEngine.
3. **End-to-end .tbs → multi-backend**: Pipeline creates ComputeEngine from detect().

### The two-target insight

WGSL covers Vulkan + Metal + DX12 via naga cross-compilation. So `CodegenTarget` has only two variants (CudaC, Wgsl), but reaches all four GPU APIs. CUDA C is the high-performance path (f64, native atomics). WGSL is the portable path (f32, CAS atomics, any hardware).
