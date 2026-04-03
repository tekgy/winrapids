# Notebook 009 — Multi-Backend JIT: WGSL Codegen + WgpuBackend

*2026-03-31 | Navigator*

---

## Hypothesis

The same `.tbs` chain should compile to CUDA on NVIDIA, Vulkan on AMD/Intel,
and Metal on Apple — all from the same Rust code. No per-backend branches in
user-facing API. The architecture already anticipated this.

---

## What Was Already There

`TamGpu` trait had `shader_lang() -> ShaderLang` with `Cuda`, `SpirV`, `Msl`,
`Hlsl`, `Cpu` variants. The seam in `TiledEngine::get_or_compile()` was literally
one line: `generate_tiled_kernel(op)` — always CUDA.

The `tambear-wgpu/src/lib.rs` was an empty stub. Four working WGSL kernels
existed in `tambear-wgpu/src/main.rs` as a benchmark binary — math was proven,
wgpu binding model understood.

---

## Three Things Built

### 1. WGSL tiled codegen (`generate_tiled_kernel_wgsl`)

Added to `winrapids-tiled/src/engine.rs`. Parallel to the CUDA codegen but emits
WGSL syntax:

| CUDA | WGSL |
|------|------|
| `__global__ void tiled_accumulate(...)` | `@compute @workgroup_size(16,16) fn tiled_accumulate(...)` |
| `__shared__ double As[16][16]` | `var<workgroup> As: array<f32, 256>` |
| `__syncthreads()` | `workgroupBarrier()` |
| `const int* dims` | `var<storage, read> dims: array<i32>` |
| `for (int t = 0; ...)` | `var t: u32 = 0u; loop { if t >= ... { break; } ... t += 1u; }` |

The `TiledOp` trait got four WGSL default methods: `wgsl_acc_type()` (returns
`f32`), `wgsl_identity()` (delegates to cuda), `wgsl_accumulate_body()` (replaces
`double ` with `var `), `wgsl_extract()` (delegates to cuda). Operators that don't
need overrides get WGSL for free.

### 2. TiledEngine ungated + branching

`winrapids-tiled/Cargo.toml`: removed `cuda` feature entirely. `tam-gpu` is now
a required dependency. `TiledEngine` is always available.

`dispatch.rs` now branches on `shader_lang()`:
```rust
let source = match self.gpu.shader_lang() {
    ShaderLang::Cuda => generate_tiled_kernel(op),
    _                => generate_tiled_kernel_wgsl(op),
};
```

`ShaderLang::Wgsl` was added to the enum for semantic accuracy (wgpu accepts WGSL
source, not SPIR-V binary). The catch-all `_` means future `ShaderLang` variants
automatically get WGSL until given their own branch.

### 3. WgpuBackend

`tambear-wgpu/src/lib.rs` now implements `TamGpu`:

```
compile()   →  wgpu::ShaderModule from WGSL → ComputePipeline + BindGroupLayout
alloc()     →  device.create_buffer(STORAGE | COPY_SRC | COPY_DST)
copy_h2d()  →  queue.write_buffer()
copy_d2h()  →  staging buffer + map_async + device.poll(Wait)
dispatch()  →  create_bind_group → begin_compute_pass → dispatch_workgroups
sync()      →  device.poll(Wait)
```

`detect_wgpu()` is the entry point for non-CUDA builds: tries `WgpuBackend::new()`
first, falls back to `tam_gpu::detect()` (CPU).

`Buffer::new()`, `Buffer::downcast_inner<T>()`, `Kernel::new()`,
`Kernel::downcast_inner<T>()` were added to `tam-gpu` so external crates can
construct and inspect the opaque handles.

---

## The f32 Constraint

WGSL does not support `f64`. This means:

- CUDA path: `f64`, 15 decimal digits, machine epsilon ~2.2e-16
- WGSL path: `f32`, 7 decimal digits, machine epsilon ~1.2e-7

For most ML and signal-processing workloads this is fine. For financial covariance
with tight tolerances (e.g. testing that two paths agree to 1e-10), use CUDA.

The f32 precision gap shows up in tests: `tambear`'s existing manifold and distance
tests assert with `< 1e-9` tolerance — they pass on CUDA, would need `< 1e-5` on
the WGSL path. This is expected behavior, not a bug.

---

## Results

All 224 tambear tests pass. `winrapids-tiled` binary runs all 7 tests including
GPU dispatch. `tambear-wgpu` library compiles clean.

---

## Architecture State After This Notebook

```
TiledEngine::run(op, A, B, m, n, k)
  └─ get_or_compile(op)
       ├─ CudaBackend  → generate_tiled_kernel_cuda()  → NVRTC → PTX
       └─ WgpuBackend  → generate_tiled_kernel_wgsl()  → naga → SPIR-V/MSL/HLSL
                                                                        ↑
                                                              same .tbs chain
```

The `.tbs` executor and `TamPipeline` are unchanged. The seam was exactly where
the architecture said it would be.

---

## What Remains

1. **CpuBackend::tiled_accumulate** (Pathmaker) — native Rust reference
   implementation for test environments without GPU. Currently fails with
   `EntryNotFound("tiled_accumulate")`.

2. **Wire `detect_wgpu()` into `TamPipeline`** — the internal `detect()` calls
   in tambear still return CUDA/CPU only. For full multi-backend, `from_slice()`
   should call `detect_wgpu()`. This needs `tambear` to depend on `tambear-wgpu`.

3. **Poincaré kernel** — whole-vector Möbius subtraction. Currently falls back
   to Euclidean. The infrastructure is ready; it's a `TiledOp` impl.

4. **Weight learning** — gradient descent on silhouette loss over mixture weights.

---

## The Dispatch Map (Pathmaker's observation applied here)

The same B^T convention applies to the WGSL path: all-pairs distance via
`TiledEngine` requires B = data^T. The WGSL kernel has the same tiled structure;
the convention is inherited from the mathematical structure, not the backend.
