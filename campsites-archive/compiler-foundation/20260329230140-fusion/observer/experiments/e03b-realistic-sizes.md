# E03b — Cross-Algorithm Sharing Retest at FinTek-Realistic Sizes

**Date**: 2026-03-30
**Requested by**: Navigator
**Status**: Complete

## Motivation

Original E03 tested at 100K, 1M, 10M. At 10M, the fused kernel was 2.4x SLOWER than CuPy.
Navigator hypothesized that at FinTek-realistic sizes (50K-900K), Python dispatch overhead
between CuPy calls would dominate, making fusion much more valuable.

## Results

### Overhead Floors

| Measurement | p50 | Notes |
|---|---|---|
| GPU sync (idle) | 0.4 us | Negligible |
| Python validate+log | 0.10 us | Negligible — NOT the bottleneck |
| CuPy sum(1 elem) | 11.5 us | This IS the per-call dispatch floor |

### Variant 1: Tight Loop

| Size | A: Independent | B: Shared | C: Fused | C/A speedup |
|---|---|---|---|---|
| 50K | 0.280 ms | 0.204 ms | 0.135 ms | **2.08x** |
| 100K | 0.288 ms | 0.223 ms | 0.143 ms | **2.01x** |
| 500K | 0.298 ms | 0.212 ms | 0.198 ms | **1.50x** |
| 900K | 0.358 ms | 0.237 ms | 0.262 ms | **1.37x** |
| 10M | 1.027 ms | 0.816 ms | 2.263 ms | **0.45x** (SLOWER) |

### Variant 2: Realistic Pipeline Overhead

| Size | A: Independent | B: Shared | C: Fused | C/A speedup |
|---|---|---|---|---|
| 50K | 0.272 ms | 0.226 ms | 0.146 ms | **1.86x** |
| 100K | 0.288 ms | 0.224 ms | 0.142 ms | **2.04x** |
| 500K | 0.323 ms | 0.225 ms | 0.198 ms | **1.63x** |
| 900K | 0.312 ms | 0.223 ms | 0.261 ms | **1.20x** |
| 10M | 1.037 ms | 0.813 ms | 2.445 ms | **0.42x** (SLOWER) |

### Per-CuPy-Call Dispatch Cost

| Size | cumsum | concat | slice-div | multiply | sqrt |
|---|---|---|---|---|---|
| 50K | 27 us | 24 us | 20 us | 11 us | 11 us |
| 100K | 25 us | 24 us | 21 us | 10 us | 11 us |
| 500K | 26 us | 25 us | 21 us | 13 us | 13 us |
| 900K | 28 us | 37 us | 25 us | 14 us | 13 us |

## Key Finding

Navigator was RIGHT about the direction, WRONG about the mechanism:

- **NOT** Python overhead between calls (0.1us — negligible)
- **IS** per-CuPy-call dispatch overhead (10-27us per call)
- Path A = ~17 CuPy calls. Path C = ~8 CuPy calls.
- At 50K: 9 fewer calls × ~20us = ~180us saved on ~280us total = **2x**
- Crossover between 500K-900K: below this, fusion wins; above, CuPy's optimized kernels win

## Architecture Implications

1. **Fusion is size-dependent.** The compiler needs a cost model that considers array size.
2. **Below ~500K rows: FUSE.** Dispatch overhead dominates. Fewer kernel launches = faster.
3. **Above ~1M rows: DON'T FUSE (naive).** CuPy's hand-optimized kernels achieve higher bandwidth.
4. **Shared intermediates (Path B) win at ALL sizes.** 1.2-1.5x consistently. This is the safe default.
5. **Pipeline overhead is NOT the Python code between ops** — it's CuPy's own dispatch cost per call.
6. **The Rust compiler eliminates the CuPy dispatch layer entirely.** E09 showed 9us launch overhead vs CuPy's 70us. This shifts the crossover point dramatically upward.
