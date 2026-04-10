# E03 experiment design

Created: 2026-03-29T23:13:19-05:00
By: navigator

---

## Scout Correction (2026-03-29): The Shared Primitive is cumsum, Not mean

CuPy's `rolling_std` doesn't use a Welford scan — it uses **prefix sums**:
- Window sums via `cumsum(x)`: `window_sum[i] = cumsum[i] - cumsum[i-w]`
- Window squared sums via `cumsum(x²)`

PCA centering uses `cp.mean(x)` = `cumsum(x)[-1] / N`.

**Both specialists decompose to the same underlying primitive: `cumsum(x)`.** The compiler doesn't need to understand statistics — it just needs to detect that two specialists both request `cumsum(A)` and compute it once.

Fused version: single kernel computing `cumsum(x)` and `cumsum(x²)` together → ONE HBM read of A. Satisfies both rolling_std and PCA centering.

This is the cleaner framing. The Welford vs windowed subtlety below is secondary. Keep cumsum as the shared primitive in the experiment.

---

## The Critical Subtlety: Windowed vs Cumulative

The vision.md experiment says "rolling_std results feed PCA centering." There's a subtlety here that matters for experiment design.

**Windowed rolling std** (e.g. `rolling(window=5).std()`) computes LOCAL statistics within each window. The mean at position i is the mean of positions [i-4, i-3, i-2, i-1, i]. The GLOBAL mean is NOT a direct output — it would require an additional pass.

**Cumulative Welford scan** (`scan(data, WelfordOp())`) computes RUNNING statistics: at position i, mean[i] is the mean of all elements [0..i]. The FINAL element mean[-1] IS the global mean.

For E03, **use cumulative Welford**, not windowed rolling. This is cleaner:
- `scan(A, WelfordOp())` → mean[], var[] (cumulative)
- `mean[-1]` = global mean of column A (no extra pass needed)
- PCA centering: `A - mean[-1]`

The compiler can detect: "PCA centering needs global_mean(A)" and "Welford scan of A produces global_mean as mean[-1]" → route the final Welford output directly to PCA centering, skipping the separate `cp.mean(A)` call.

## Experiment Structure

**Baseline (3 HBM passes over A):**
```python
running_mean = cumulative_welford_mean(A)      # pass 1: read A, write mean[]
running_var  = cumulative_welford_var(A)       # pass 2: read A again (or combined)
global_mean  = cp.mean(A)                      # pass 3: read A AGAIN
centered     = A - global_mean                 # pass 4: read A again
```

Actually the naive baseline is worse — separate rolling and centering calls.

**Correct baseline (split primitives, no sharing):**
```python
running_mean, running_var = welford_scan(A)    # 1 pass through A
global_mean  = cp.mean(A)                      # 1 ADDITIONAL pass through A
centered     = A - global_mean                 # 1 more pass
```
HBM cost: 3 reads of A (for 10M float32 = 120 MB), 2 writes (mean[], centered)

**Shared version (the compiler detects the reuse):**
```python
running_mean, running_var = welford_scan(A)    # 1 pass through A
global_mean  = running_mean[-1]                # free — already computed
centered     = A - global_mean                 # 1 more pass
```
HBM cost: 2 reads of A (80 MB), 2 writes. Save: 40 MB = ~24 μs at 1,677 GB/s.

**Fully fused (single kernel):**
```python
running_mean, centered = welford_scan_and_center(A)
```
HBM cost: 1 read of A (40 MB), 1 write (centered). But this requires a two-pass kernel — scan forward to get global_mean, then pass backward to subtract it. This might not be worth the complexity.

## What the Experiment Actually Proves

The benchmark number (24 μs saved on 10M rows) is not the point. The point is:

**Can the compiler DETECT this sharing from the primitive registry?**

The mechanism:
1. `welford_scan` is registered as producing `outputs: ["mean[]", "var[]"]`, where `mean[-1]` is tagged as `alias: "global_mean(A)"`
2. `pca_center` is registered as consuming `inputs: ["global_mean(A)"]`
3. The compiler walks the graph, sees PCA needs `global_mean(A)`, checks if any prior computation satisfies it, finds the Welford scan, routes its output

This is a graph reachability problem. The compiler doesn't need to "understand" statistics — it just needs the registry to declare named inputs and outputs, and a graph analysis pass to find satisfiers.

## The Architectural Claim Being Tested

E03 tests whether the compiler can eliminate redundant computation across algorithm boundaries — not just within a single kernel.

If yes: every pipeline that does (any rolling stat) + (any algorithm that needs the column mean) gets a free optimization. This compounds as the specialist library grows.

If no: we need explicit user annotation ("use the mean from the rolling computation") — which defeats the "user describes WHAT, compiler decides HOW" promise.

The answer should be yes. The mechanism is the named-outputs registry. That's what makes it automatic.

## Suggested Benchmark Sizes

- 1M rows: ~16 MB data, small enough for L2 (128 MB L2). May NOT see sharing benefit (L2 might serve the second read).
- 10M rows: ~40 MB data, clearly exceeds L2. Should see the HBM traffic difference.
- 100M rows: ~400 MB, stress test. The sharing benefit scales linearly.

**Run at 10M and 100M.** At 1M the L2 may hide the difference.

