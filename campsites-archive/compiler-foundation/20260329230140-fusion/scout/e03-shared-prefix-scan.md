# E03 Experiment Design: rolling_std → PCA Centering via Shared Prefix Scan

*Scout: 2026-03-29*

---

## What CuPy rolling_std Actually Computes

CuPy does NOT have a `rolling()` method like Pandas/Polars. Rolling std in the FinTek codebase is implemented as a prefix-sum computation. The canonical approach:

```python
def rolling_std_cupy(arr, window):
    # Prefix sums for online statistics
    cumsum   = cp.cumsum(arr)              # prefix sum of x
    cumsum_sq = cp.cumsum(arr ** 2)        # prefix sum of x²

    # Rolling window sums (subtract left side from right side)
    win_sum   = cumsum[window:]   - cumsum[:-window]
    win_sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]

    # Rolling mean and variance
    rolling_mean = win_sum / window
    rolling_var  = win_sum_sq / window - rolling_mean ** 2
    return cp.sqrt(rolling_var)
```

**Key observation**: this computes `cumsum` = prefix scan of the data. The rolling mean IS computed internally as `win_sum / window` — it's an intermediate value that gets discarded before returning.

---

## What PCA Centering Needs

PCA centering requires the global column mean:
```python
col_mean = cp.mean(arr)   # = cp.sum(arr) / len(arr) = cumsum[-1] / N
```

This is `cumsum[-1] / N` — the LAST element of the prefix scan, divided by N. Identical computation to the rolling_std prefix scan.

---

## The Shared Intermediate: prefix_scan

Both operations read from the same prefix scan. Today they compute it twice:

**Baseline (two separate computations)**:
```python
# E03 baseline: two prefix scans
rolling_std_result = rolling_std_cupy(data, window)  # computes cp.cumsum internally
pca_center         = cp.mean(data)                    # computes cp.sum internally (separate kernel)
```
VRAM reads: 2× data for two cumsum kernels + further reads for window subtraction

**Shared (one prefix scan feeds both)**:
```python
# E03 shared: one prefix scan
cumsum     = cp.cumsum(data)                        # ONE scan kernel
win_sum    = cumsum[window:] - cumsum[:-window]     # slice arithmetic, no new scan
rolling_std_result = derive_std_from_prefix(cumsum, window)
pca_center = float(cumsum[-1]) / len(data)          # free — just read cumsum[-1]
```
VRAM reads: 1× data for one cumsum kernel + zero additional reads for pca_center

---

## Experiment Structure

### Setup
```python
N = 10_000_000   # 10M floats → 40 MB
W = 50           # rolling window
data = cp.random.randn(N, dtype=cp.float32)
cp.cuda.Stream.null.synchronize()
```

### Baseline Measurement
```python
def baseline(data, W):
    rolling = rolling_std_cupy(data, W)          # kernel A: cumsum, kernel B: std
    center  = cp.mean(data)                       # kernel C: separate sum/mean
    cp.cuda.Stream.null.synchronize()
    return rolling, center
```
Kernel count: 3+ kernels, 2 full data reads (40 MB each = 80 MB total HBM reads)

### Shared Intermediate Measurement
```python
def shared(data, W):
    cumsum    = cp.cumsum(data)                  # ONE kernel: one data read (40 MB)
    win_sum   = cumsum[W:] - cumsum[:-W]         # cheap slice: reads from cumsum (40 MB in L2)
    win_sum_sq = cp.cumsum(data ** 2)             # still needs sq separately (second scan)
    # Note: the sq prefix scan is different — not shareable unless we do both in one pass
    rolling   = derive_std(win_sum, win_sum_sq, W)
    center    = float(cumsum[-1]) / len(data)    # FREE — cumsum[-1] already computed
    cp.cuda.Stream.null.synchronize()
    return rolling, center
```

**Refine the sharing**: the FULL sharing opportunity is if you compute BOTH `cumsum(data)` and `cumsum(data**2)` in one pass:
```python
def fully_shared(data, W):
    # One kernel: two prefix scans simultaneously (multi-output scan)
    cumsum, cumsum_sq = two_output_prefix_scan(data)   # ONE kernel, ONE data read
    win_sum    = cumsum[W:]    - cumsum[:-W]
    win_sum_sq = cumsum_sq[W:] - cumsum_sq[:-W]
    rolling    = cp.sqrt(win_sum_sq/W - (win_sum/W)**2)
    center     = float(cumsum[-1]) / len(data)         # FREE
    cp.cuda.Stream.null.synchronize()
    return rolling, center
```
This is the full fusion: ONE kernel reads the data once, produces both prefix scans. Zero extra reads for PCA centering.

---

## What to Measure

For each approach, measure with `cp.cuda.Event` timing:
1. Number of kernel launches (via `cp.cuda.profiler` or Nsight)
2. Wall time (stream-synchronized)
3. Effective memory bandwidth = (total bytes read from HBM) / time

### Expected Results

| Approach | HBM reads | Kernel launches | Expected time |
|---|---|---|---|
| Baseline | 80 MB (2× data) + 80 MB (2× sq) = 160 MB | 4-6 | ~96 μs |
| Partial shared | 40 MB + 40 MB = 80 MB | 3-4 | ~58 μs |
| Fully shared (two-output scan) | 40 MB (once) | 1-2 | ~31 μs |

Fully shared = **~3× faster** for this shape.

---

## Why This Proves Compiler Value

The E03 experiment doesn't just show a speedup. It proves the following compiler claim:

> "Two separate pipeline stages — feature engineering (rolling_std) and ML preprocessing (PCA centering) — share a common primitive computation. A compiler that crosses algorithm boundaries can detect this sharing and emit a single fused kernel that the programmer could not easily write manually."

The programmer writing `rolling_std(data) + pca_center(data)` separately can't see the shared primitive. The WinRapids compiler decomposing both to `scan` primitive sees the sharing automatically.

---

## Connection to the scan Primitive

The `scan` primitive in WinRapids is exactly this: prefix sum over input data. Rolling std AND PCA centering both decompose to `scan`. The compiler's primitive decomposition registry (E04) will see this sharing and emit one scan kernel.

E03 is the empirical proof that the theoretical sharing is real and measurable.
