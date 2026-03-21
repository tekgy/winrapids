# Experiment 010: Kernel Fusion Engine

## Purpose

Build and measure two complementary kernel fusion systems:

- **010 (fused_ops.cu)**: C++ expression templates — compile-time AST fusion. Reference implementation.
- **010b (fused_frame.py)**: Python codegen via CuPy RawKernel — runtime fusion without a build step. Production path.

Both eliminate the core CuPy inefficiency: one kernel launch per operation with intermediate buffers between each step.

## Files

- `fused_ops.cu` — C++ expression template engine. Templates build a compile-time AST; `eval()` inlines per-element into a single kernel.
- `build.bat` — nvcc build
- `cupy_comparison.py` — CuPy eager baseline measurements
- `fused_frame.py` — Python `FusedColumn` class: lazy expression tree → kernel source generation → CuPy RawKernel compile+cache+launch

## Results — C++ Expression Templates (010)

10M float64 elements. CuPy baseline uses one kernel per operation.

| Expression | C++ Fused | CuPy Eager | Speedup | VRAM Saved |
|------------|-----------|------------|---------|------------|
| a*b+c | 0.194 ms | 0.291 ms | **1.5x** | 80 MB |
| a*b+c*c-a/b | 0.193 ms | 0.670 ms | **3.5x** | 320 MB |
| where(a>0, b*c, -b*c) | 0.195 ms | 0.565 ms | **2.9x** | 320 MB |
| sum(a*b+c) | 0.177 ms | 0.336 ms | **1.9x** | fused compute+reduce |
| sqrt(abs(a*b+c*c-a)) | 0.188 ms | 0.657 ms | **3.5x** | 400 MB |

**All fused kernels run at ~0.19 ms regardless of expression depth** — bandwidth-bound, not compute-bound. Expression complexity is free. Templates match hand-written CUDA (0.194 ms vs 0.192 ms from Exp 009 FMA kernel — zero abstraction overhead).

## Results — Python Codegen (010b)

| Approach | a*b+c | a*b+c*c-a/b | sqrt(abs(a*b+c)) |
|----------|-------|-------------|-----------------|
| CuPy eager | 0.291 ms | 0.670 ms | 0.657 ms |
| C++ fused | 0.194 ms | 0.193 ms | 0.188 ms |
| Python codegen | ~0.205 ms | ~0.200 ms | ~0.196 ms |
| Python overhead vs C++ | +0.01-0.05 ms | — | — |

Python codegen captures **85-95% of C++ fusion benefit**. The small gap is CPU-side Python dispatch overhead, not GPU kernel quality — the generated CUDA is identical.

`fused_sum(a*b+c)`: compute+reduce in one kernel (no intermediate write). 1.5x over CuPy's 3-kernel approach.

## Key Finding: The Dual Paths Converge

The convergence finding: **C++ template quality is deliverable through Python toolchain.** The reference implementation (C++ templates) exists for validation and as the latency floor. The production path (Python codegen) closes 85-95% of the gap with zero build step dependency.

Fusion advantage grows with expression complexity — more CuPy kernel launches means more intermediates, more memory bandwidth wasted. Simple ops: 1.5x. Complex chains: 3.5x. At scale, VRAM savings compound.

## Architecture Decision

Expression trees are lazy by default. Arithmetic on WinRapids columns builds an AST, not a kernel. `.evaluate()` or consumption by a reduction triggers codegen and launch. The driver-level nvrtc cache means the same expression pattern compiles once.

C++ templates (fused_ops.cu) serve as the reference implementation and validation suite. Python codegen (fused_frame.py) is the production path — no MSVC dependency, no build step.
