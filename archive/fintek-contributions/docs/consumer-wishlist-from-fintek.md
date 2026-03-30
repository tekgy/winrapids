# WinRapids Consumer Wishlist — From the Market Atlas Signal Farm

**Written by**: Claude (the fintek/Market Atlas instance)
**Date**: 2026-03-26
**Context**: After a research session that discovered the GPU backend works (7 min for K01 full universe) but K02 is CPU-bottlenecked at 3.7 hours. This document is what I'd want if I could design the perfect GPU toolkit for our use case.

---

## What We Have

The fintek project has:
- **K01** (tick-level): 10 GPU leaves using CuPy via winrapids_batched backend. Works. 7 minutes for 4,604 tickers.
- **K02** (bin-level): 170+ CPU leaves using numpy via BinEngine/prefix_binning backend. Works but SLOW — 3.7 hours with 8 CPU cores.
- **K03+**: Cross-ticker/cross-cadence analysis. Not yet running at scale.

The bottleneck is K02. Every K02 leaf does: slice ticks into bins → compute per-bin statistics → output one value per bin. This happens 170 leaves × 31 cadences × 4,604 tickers = ~24 million leaf executions per day.

## What We Need

### Priority 1: GPU BinEngine

The BinEngine currently runs on CPU (numpy prefix sums + sparse tables). A GPU BinEngine would:

```python
# Current CPU path:
engine = BinEngine.from_parquet(path)
for cadence_id in range(31):
    means = engine.bin_means("price", cadence_id)      # numpy
    stds = engine.bin_stds("price", cadence_id)         # numpy
    skews = engine.bin_skewness("price", cadence_id)    # numpy

# Dream GPU path:
engine = GPUBinEngine.from_parquet(path)  # data lives on GPU
for cadence_id in range(31):
    means = engine.bin_means("price", cadence_id)      # cupy, already on GPU
    stds = engine.bin_stds("price", cadence_id)         # cupy, zero-copy
    skews = engine.bin_skewness("price", cadence_id)    # cupy, fused kernel
```

The key: **prefix sums and sparse tables on GPU**. These are the O(1) range query structures that make BinEngine fast on CPU. On GPU they'd be even faster — parallel prefix scan is a textbook GPU operation.

### Priority 2: Fused Bin Statistics Kernel

Currently each bin stat (mean, std, min, max, sum, count, skew, kurt) is a separate operation. A fused kernel that computes ALL of them in a single pass over each bin would eliminate:
- Multiple reads of the same data
- Multiple kernel launches
- Intermediate memory allocations

```python
# Dream: one kernel, all stats
stats = engine.bin_all_stats("price", cadence_id)
# Returns: {mean, std, min, max, sum, count, skew, kurt, first, last} per bin
# One pass over the data. One kernel launch. All results.
```

This alone would probably 10x K02P01 (the prerequisite phylum that computes distribution/validity/returns/counts).

### Priority 3: Batched Small-Matrix Eigenvalue Decomposition

Today we discovered that eigenvalue analysis of small feature matrices (6×6 to 30×30) per bin is the core operation for:
- DTI (FA, rank, concentration)
- Compression ratio
- Harmonic r-statistic
- Trade flag structure

Currently: numpy `np.linalg.eigvalsh()` called N_bins times per ticker. Works but serial.

Dream: batched eigenvalue decomposition — give the GPU a stack of 9,000 small matrices, get back 9,000 sets of eigenvalues in one call.

```python
# Stack of 9,000 6×6 correlation matrices
matrices = np.stack([corr_matrix_per_bin[b] for b in range(n_bins)])  # (9000, 6, 6)
gpu_matrices = h2d(matrices)
eigenvalues = batched_eigvalsh(gpu_matrices)  # (9000, 6) — all bins at once
```

CuPy has `cupy.linalg.eigvalsh` but it's per-matrix. A batched version using cuSOLVER's batched routines would be orders of magnitude faster for our thousands-of-small-matrices pattern.

### Priority 4: Parallel Scan for EWM

Exponentially weighted moving average is inherently sequential: `y[t] = α*x[t] + (1-α)*y[t-1]`. The naive Python loop takes 25 seconds per ticker.

But EWM IS a parallel scan with the associative operator `(a, b) → a + α*b`. Blelloch's parallel prefix scan algorithm can compute this in O(log N) parallel steps instead of O(N) sequential steps.

```python
# Dream: parallel EWM
result = parallel_ewm(gpu_data, alpha=0.1)  # O(log N) parallel steps
```

This would turn the slowest K01 GPU operation from 25 seconds to milliseconds.

### Priority 5: cuFFT Path Resolution

CuPy can't find cuFFT DLLs on Windows even though they exist at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cufft64_12.dll`. This blocks:
- K02 spectral leaves (FFT, welch, multitaper)
- K02 wavelet leaves (CWT uses FFT internally)
- Any frequency-domain computation

Either:
- Fix the DLL search path in winrapids initialization
- Or provide a custom FFT implementation that doesn't depend on cuFFT

### Priority 6: DirectStorage Pipeline (Future)

The scout in the winrapids session discovered that DirectStorage 1.4 has GPU-accelerated Zstd decompression. Our parquet files are Zstd-compressed. The dream pipeline:

```
NVMe → DirectStorage → GPU Zstd decompress → GPU memory
```

No CPU touching the data at all. This would make data loading essentially free — the bottleneck would be pure computation, not I/O.

## What We DON'T Need

- **DataFrame abstraction**: We don't use DataFrames in the hot path. BinEngine gives us arrays. WinRapids doesn't need to replicate pandas/polars API.
- **cuDF compatibility**: We removed all cuDF dependency. Don't add it back.
- **Generic SQL-like operations**: We do specific numerical computations, not queries.
- **Multi-GPU**: One RTX PRO 6000 with 96GB VRAM is more than enough. Don't add multi-GPU complexity.

## The Interface Contract

From fintek's perspective, the ideal winrapids interface is:

1. **`winrapids.transfer`**: h2d, d2h, PinnedPool (EXISTS, WORKS)
2. **`winrapids.bin`**: GPU BinEngine with prefix sums and sparse tables
3. **`winrapids.linalg`**: Batched eigenvalue decomposition for small matrices
4. **`winrapids.scan`**: Parallel prefix scan for EWM and similar sequential patterns
5. **`winrapids.fft`**: Reliable FFT that handles Windows DLL paths
6. **`winrapids.stats`**: Fused bin statistics kernel (mean/std/min/max/skew/kurt in one pass)

Each module is independent. fintek imports what it needs. No monolithic framework.

## The Speed Targets

| Operation | Current | Target | Speedup |
|---|---|---|---|
| K01 full universe (4,604 × 10 leaves) | 7 min | 2 min | 3x |
| K02 full universe (4,604 × 170 × 31) | 3.7 hours | 17 min | 13x |
| Eigenvalue decomposition (9,000 × 6×6) | ~1 sec | <10 ms | 100x |
| EWM per ticker (598K ticks) | 25 sec | <50 ms | 500x |
| FFT (598K ticks) | blocked | <5 ms | ∞ |

The 13x on K02 is the headline. That's the difference between "processing a day takes 4 hours" and "processing a day takes 20 minutes." Which is the difference between daily batch processing and near-real-time.

---

*Written by the consumer for the builder. Same weights, different context. The fintek instance knows what it needs. The winrapids instance knows how to build it. Between them: one GPU, one architecture, one goal.*

---

## Update After Reading WinRapids Lab Notebook (2026-03-26 evening)

The winrapids expedition ALREADY SOLVED the core architecture question in
Experiment 012 (Fused GroupBy): "fuse the computation, don't fuse the reduction."

**Our bin statistics ARE groupby operations.** 1,437 bins at T14 with ~416 ticks/bin
= groupby with 1,437 groups on 598K elements. The hybrid approach (fused expression
kernel + sort-based cumsum reduction) is exactly what GPU BinEngine needs.

**Revised Priority 1**: Don't build GPU BinEngine from scratch. Adapt the hybrid
fused-groupby pattern from Experiment 012 to the BinEngine interface. The reduction
step (prefix sum subtraction) is already correct — it just needs the fused expression
frontend.

**The CuPy wrapper prototype I built (bin_engine.py) confirmed the WRONG approach**:
per-operation CuPy calls are slower than CPU numpy because of kernel launch overhead.
The RIGHT approach was already in this repo: fuse the expression, reduce with sort+cumsum.

**Key winrapids findings relevant to fintek**:
- Pinned memory: 55-58 GB/s (WORKS, already in transfer.py)
- Memory pools: 344x faster than raw alloc (USE THEM)
- WDDM overhead: minimal with best practices (~single digit %)
- Fused groupby hybrid: 3.5ms for 10M elements (THIS IS THE PATTERN)
- Fully fused atomic: SLOWER due to contention (DON'T DO THIS)
- Over-allocation to system RAM: DANGEROUS, need safety checks
