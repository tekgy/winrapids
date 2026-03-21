# Experiment 009: Dual-Path GPU DataFrame (CUDA C++ vs CuPy)

## Purpose

Measure the abstraction cost of CuPy by implementing the same operations from Experiment 004 in raw CUDA C++. Same operations, same data sizes, CUDA events for sub-microsecond timing.

This is the first experiment under the dual-path directive: for every performance-critical operation, measure both the pragmatic path (CuPy) and the from-scratch path (raw CUDA C++).

## Files

- `cuda_dataframe.cu` — Raw CUDA C++ kernels: warp-shuffle reductions (sum, min, max), filtered sum, scalar FMA, vectorized double2 FMA
- `build.bat` — nvcc build with -arch=sm_120 (Blackwell), MSVC host compiler

## Results (10M float64, ~80 MB per column)

| Operation | CUDA C++ | CuPy/GpuFrame | Abstraction Cost |
|-----------|----------|----------------|-----------------|
| Sum (warp-shuffle) | 0.082 ms (976 GB/s) | 0.099 ms | **1.2x** |
| Min (warp-shuffle) | 0.062 ms (1299 GB/s) | 0.123 ms | **2.0x** |
| Max (warp-shuffle) | 0.062 ms (1299 GB/s) | 0.137 ms | **2.2x** |
| Filtered sum | 0.084 ms (1072 GB/s) | 0.331 ms | **3.9x** |
| FMA (a*b+c) | 0.192 ms (1668 GB/s) | 0.531 ms | **2.8x** |
| Double2 FMA (vectorized) | 0.200 ms | 0.192 ms | no benefit |
| CuPy RawKernel filtered sum | 0.172 ms | — | 2x slower than warp-shuffle |

## Key Findings

**The abstraction cost is not uniform.** It depends on:

1. **Whether CuPy already uses the optimal primitive** — CuPy's sum uses warp-shuffles internally (1.2x gap). CuPy's min/max use atomics (2.0-2.2x gap).

2. **Whether operations compose** — Filtered sum (3.9x): CuPy compacts a temp array via fancy indexing, then sums. The fused kernel does predicated accumulation in one pass with zero intermediates. The gap is architectural, not tuning.

3. **Whether intermediate buffers dominate** — FMA (2.8x): CuPy launches two kernels with an 80 MB intermediate. The fused kernel does one pass.

**Vectorized loads don't help for bandwidth-bound ops.** `double2` loads two doubles simultaneously — same cache lines as scalar load. Already bandwidth-bound; no benefit.

**Kernel fusion and warp-shuffle are the two biggest wins** from going raw CUDA.

## Architecture Decision

CuPy for prototyping. Custom CUDA kernels for production hot paths where operations compose. The backend-as-contract design (`backend="cupy-generic"` vs `backend="cuda-custom"`) exposes both through the same DataFrame API.
