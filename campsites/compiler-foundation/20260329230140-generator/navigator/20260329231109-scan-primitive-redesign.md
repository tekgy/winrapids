# scan primitive redesign

Created: 2026-03-29T23:11:09-05:00
By: navigator

---

## The Finding

From Tekgy + team-lead research session (2026-03-29): **parallel scan with pluggable associative operators is the single biggest opportunity in GPU computing.**

Everyone reimplements it:
- Mamba (selective SSM) — ML inference
- ARIMA, Kalman filters — time series
- EWM (exponential weighted mean) — data science
- cumsum/cumprod/cummax — base operations

All are the same scan primitive with different associative operators. Nobody packages it generalized.

## The Design

**Current spec (too narrow):**
```
scan: Prefix sum/product/max | contiguous, segmented, windowed
```

**Correct spec:**
```
scan(data, op: AssociativeOp, segments=None) -> State[]
```

Where `AssociativeOp` has three methods:
```python
class AssociativeOp:
    def identity(self) -> State          # neutral element (0 for sum, 1 for product)
    def update(self, s: State, x: T) -> State   # apply one element to state
    def combine(self, l: State, r: State) -> State  # merge two partial states (for parallel prefix)
```

The `combine` method is what enables parallelism — parallel prefix scan requires associativity. Any op where `combine(combine(a,b),c) == combine(a,combine(b,c))` can use the same parallel scan skeleton.

**Examples:**
```python
scan(data, AddOp())             → cumsum
scan(data, MaxOp())             → cummax
scan(data, WelfordOp())         → (mean[], var[]) at each step — rolling stats in one pass
scan(data, KalmanOp(F,H,Q,R))  → filtered state[] — Kalman filter
scan(data, SSMOp(A,B,C,D))     → Mamba selective scan output
scan(data, EWMOp(alpha))        → exponential weighted mean
```

## The NVRTC Path

The CUDA implementation of parallel prefix scan is always the same structure:
1. Thread-level: each thread scans its chunk sequentially
2. Warp-level: warp shuffle to combine across threads
3. Block-level: shared memory to combine across warps
4. Grid-level: multi-block prefix for large arrays

The only thing that changes per-op is the `combine` function. With NVRTC:
1. Hard-code the scan skeleton as a CUDA template with `OP_COMBINE` and `OP_IDENTITY` as placeholders
2. Code-generate the operator definition from the registry
3. Inject + compile with NVRTC
4. Cache (never recompile same op)

This is exactly what FlashInfer does for attention — user-defined transformations injected into kernel templates before NVRTC compilation. FlashInfer is production-proven at scale. We're building the generalized version.

## What This Means for E04

When you build the primitive decomposition registry (E04), the `scan` entry should not be a fixed implementation — it should be a template + operator slot. The registry entry for `ewm` becomes:

```python
"ewm": {
    "primitive": "scan",
    "operator": EWMOp(alpha=params.alpha),
    "variants": ["contiguous", "segmented"]
}
```

And `rolling_std` becomes:
```python
"rolling_std": {
    "primitive": "scan",
    "operator": WelfordOp(window=params.window),
    "outputs": ["mean", "std"],  # named outputs — critical for E03 sharing
    "variants": ["contiguous", "segmented"]
}
```

The **named outputs** field is the key for cross-algorithm sharing (E03). When `rolling_std` declares it produces `mean` as a named output, the compiler can detect that PCA centering's `mean` input can be satisfied by a prior `rolling_std` computation.

## The Market Expansion

Pluggable scan makes WinRapids relevant to:
- **ML inference**: Mamba SSM (competitive with transformers for long sequences) — one scan primitive
- **Time series**: ARIMA, Kalman, state space models — one scan primitive
- **Data science**: all rolling stats, EWM — one scan primitive

This is anti-YAGNI in practice: building the general scan costs almost nothing extra over the narrow scan (same CUDA skeleton, different combine function), but the payoff is enormous scope expansion.

## The Arena IR Note

From the scout: Polars uses an arena-based IR — nodes allocated in an arena, edges as indices. Fast traversal, no GC pressure. Right for Rust. The compiler's primitive dependency graph should use this pattern.

The XLA fusion criterion (also from the session): **fuse if the intermediate would otherwise go to HBM.** This is the precise condition. Not "fuse if possible" — that over-fuses and kills occupancy. Not "never fuse across algorithm boundaries" — that under-fuses. The right question per intermediate: would writing this to HBM and reading it back cost more than the kernel complexity of keeping it in registers/shared memory?

For the scan case: WelfordOp's running (mean, count, sumsq) state lives in registers. Never goes to HBM during the scan. The HBM question only arises when the FINAL mean output needs to be shared with PCA — that's the E03 sharing case, and that final value WOULD go to HBM without sharing.

