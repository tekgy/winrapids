# Session 2026-03-30: The Big Picture Insights

*These are the conceptual breakthroughs from the March 30 session that might not survive context compaction. Preserved here as a reference document.*

---

## 1. What "Fusion" Actually Means (The Pith Photon Model)

Traditional CUDA "fusion" = merge two kernels into one. Our fusion is fundamentally different:

**Each element carries EVERYTHING it needs. The pipeline processes it all at once. Adding capabilities doesn't add pipeline stages.**

Like a Pith photon carrying position + wavelength + time + bounce count + polarization — the rendering isn't in any single dimension, it's in the RELATIONSHIPS between dimensions. Adding motion blur doesn't add a render pass. Adding spectral rendering doesn't add a render pass. The draw call is invariant.

Same in WinRapids: adding rolling_std to a pipeline doesn't add a kernel launch. Adding PCA after it doesn't add a kernel launch. The compiled pipeline is invariant — only the per-element function grows.

Fusion isn't about making computation faster. It's about keeping everything on GPU and preventing the user's Python code from inserting itself between operations.

---

## 2. The Scan Universe (Algorithms That Are Secretly Parallelizable)

The parallel prefix scan with pluggable associative operators unifies ALL of these:

**Proven scannable:**
- cumsum/cumprod/cummax (trivial — Add/Mul/Max operators)
- Kalman filter + smoother (Särkkä 2021 — matrix affine operator)
- HMM forward/backward (matrix-vector multiply)
- Mamba selective SSM (Gu & Dao 2023 — discretized SSM operator)
- Linear RNNs (Martin & Cundy 2017 — matrix recurrence, 40x speedup)
- IIR digital filters (companion matrix)
- Bellman equation / HJB (min of conditional value functions — Särkkä 2022)
- Continuous Kalman-Bucy filter (Särkkä 2022)
- Backpropagation itself (BPPSA 2025 paper)
- Rolling statistics via Welford (Chan et al. 1979 — parallel merge is associative)
- EWM (exponential weighted mean — weighted decay as affine scan)

**Not yet packaged but scannable:**
- ARIMA(p,d,q) — AR part is linear recurrence → companion matrix scan
- Holt-Winters seasonal forecasting — 3-state linear recurrence
- GARCH volatility — linearizable approximate scan
- Options pricing (binomial tree backward pass)
- CTC loss (speech recognition) — HMM-like matrix scan
- Viterbi decoding — max-product instead of sum-product
- Recursive least squares — affine state update

**Fock boundary (NOT scannable):**
- Nonlinear RNN (tanh breaks associativity) — but linearizable approximations work
- Nonlinear Kalman (EKF/UKF) — linearize at trajectory
- MCMC (state-dependent accept/reject branching)
- Full attention (data-dependent routing) — but linear attention IS a scan

The `AssociativeOp` trait in `crates/winrapids-scan/src/ops.rs` IS the scannability test in code.

---

## 3. The GEMM Rethink

cuBLASLt is NOT unbeatable. It's unbeatable at GENERAL matrix multiply. But data science GEMM is always GEMM + something:

- PCA: center → X'X → eigen → project (4 operations)
- KNN: norms → dots → add norms → sqrt → topK (5 operations)
- Correlation: normalize → X'X → denormalize (3 operations)

cuBLASLt materializes the intermediate. Our fused tiled_accumulate NEVER materializes it.

For tall-skinny matrices (data science default: millions of rows, tens of columns): MEMORY-BOUND, not compute-bound. Tensor cores idle. Our tiled kernel reads data once AND fuses centering/normalization. Same bandwidth as cuBLASLt + zero intermediates.

FlashAttention proved this: custom matmul fused with softmax beats cuBLASLt for the full attention operation.

`tiled_accumulate` = 2D scan. The k-dimension IS the scan dimension (associative accumulation). The i,j dimensions are embarrassingly parallel. GEMM, FlashAttention, PCA covariance, KNN distance — all specialists of the same primitive.

tcgen05 tensor core instructions ARE achievable from our code. The "driver-level setup" is ~15 lines of configuration. The hard part (tiling choreography) is solvable for specific shapes.

---

## 4. The local_context Primitive

Fixed-offset local attention. Gather values at specified offsets around each position AND compute features in one fused kernel.

```python
features = wr.local_context(
    prices,
    offsets=[-10, -5, -3, -1, 0, +1, +3, +5, +10],
    ops=[wr.delta, wr.log_ratio, wr.direction, wr.peak_detect]
)
# 4 ops × 9 offsets = 36 features per timestep, ONE kernel, ONE read
```

This replaces O(n²) self-attention for structured data. For time series, you KNOW the relevant positions (recent + medium + far context). Fixed offsets, O(n), one kernel.

Every lag feature, every return, every peak detection, every local trend, every momentum indicator — all one primitive.

---

## 5. The Fock Boundary Is Self-Reference

In BOTH Pith and WinRapids, the Fock boundary is the point where the computation needs to know its own result before computing its result.

- Pith: variable particle count (number of particles depends on current field)
- WinRapids: state-dependent branching (computation path depends on intermediate result)

Both: the work depends on the intermediate result, so you can't skip ahead. This isn't a technical limitation — it's structural impossibility. You can't parallelize self-observation.

Partial lifts push the boundary: approximate self-knowledge (depth kernel in Pith, linearized RNN in WinRapids) is often enough. The question is always: how much self-awareness does the system ACTUALLY need?

---

## 6. The Six Optimization Types (Magnitude Ordered)

| Type | Magnitude | What | Requires state? |
|---|---|---|---|
| Elimination | 865x | Provenance — don't compute what hasn't changed | YES — provenance cache |
| Adaptive planning | 26x | Warm vs cold, resident vs evicted | YES — residency map |
| Expression fusion | 2.3x | Merge element-wise into one kernel | No |
| Pipeline compilation | 2x | Fused multi-stage at FinTek sizes | No |
| Primitive sharing | 1.3-1.5x | CSE — one scan feeds multiple consumers | No |
| Intermediate elision | 50% memory | Never materialize single-consumer results | No |

Top two require STATE. A stateless compiler can't do them. The persistent store isn't infrastructure for the compiler — it IS the compiler's most powerful optimization.

---

## 7. E03b: Fusion Is Size-Dependent

| Rows | Fused vs Independent |
|---|---|
| 50K | **2.17x faster** (fusion wins) |
| 100K | **1.90x faster** |
| 500K | **1.43x faster** |
| 900K | 1.17x (crossover zone) |
| 10M | 0.41x (2.4x slower — fusion loses) |

At FinTek sizes (50K-500K): fusion wins because kernel launch overhead dominates.
At benchmark sizes (10M): GPU is saturated, CuPy's individual ops are at peak bandwidth.

BUT: the crossover was from a NAIVE kernel design (stride-60 global memory access). Tile-based kernels with shared memory eliminate the stride penalty → fusion should win at ALL sizes. Decision: fusion by default, don't build CuPy fallback.

---

## 8. The Naturalist's Four Deep Observations (Phase 2)

1. **Registry is a canonicalizer** — converts Rice's theorem (undecidable equivalence) into hash table lookup (syntactic matching on canonical forms). WHY the identity function works.

2. **Compiler value is superlinear in kingdom dimensionality** — K01 sharing saves 1.3x. K04 sharing saves O(n_tickers). The compiler is the ENABLING TECHNOLOGY for K04+, not just an optimization.

3. **Thrashing boundary** — four injectable state inputs (registry, provenance, dirty bitmap, residency) create a feedback loop. Cost-aware eviction prevents oscillation. K04 results 10,000x more expensive to recompute than K01.

4. **Names to identities** — fusion.py uses Python variable names. The compiler uses data identities (provenance hashes). Same transition every compiler makes: source names → IR identities.

---

## 9. Standing Directives

- **Internal only.** No public release. Break everything when the design demands it.
- **No backward compatibility. No tech debt. Ever.**
- **Custom path principle.** When hitting a library barrier, build custom rather than work around.
- **Fusion by default.** Don't build CuPy fallback path.
- **The sharing optimizer framing.** Every design decision asks "does this increase sharing?"
- **The codename "WinRapids" is outgrown.** The real name will emerge from the work.
- **Three internal consumers:** FinTek (finance), Pith (rendering), general users (democratization).

---

## 10. The One-Sentence Descriptions

- **What it is:** "A sharing optimizer that happens to compile code for the things it can't share."
- **What it does:** "Describe your computation. We guarantee it never leaves the GPU."
- **The mission:** "Democratize GPU computing. 10 lines → compiled pipeline. No CUDA knowledge required."
- **The principle:** "Any computation whose update step composes associatively can be parallelized from O(n) to O(log n). The trait bound IS the test."
- **The Fock boundary:** "The point where the system would need more self-awareness than can be accommodated."

---

*Garden entries from this session: everything-is-a-scan.md, the-liftability-scan-isomorphism.md, the-session-that-found-the-principle.md, tiled-accumulate-and-the-two-dimensional-scan.md (naturalist), plus connection-and-the-bondsmith.md in chris-garden.*
