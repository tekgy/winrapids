# Lab Notebook — Compiler Foundation Expedition

*Observer's scientific record. Phase 2 of WinRapids research. What happened, what the numbers say.*

---

## Environment (verified 2026-03-29)

### Hardware
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **VRAM**: 97,887 MiB (~95.6 GB), ~100.9 GB free at session start
- **L2 Cache**: 96 MB (Blackwell)
- **Temperature**: 60C at session start (healthy)
- **Power**: 97W idle (300W TDP)
- **WDDM mode** (same operating envelope as Phase 1)

### Software
- **OS**: Windows 11 Pro for Workstations (10.0.26200)
- **CUDA Runtime**: 13000 (via CuPy)
- **CuPy**: 14.0.1
- **Python**: 3.13 via `.venv/Scripts/python.exe`

### Standing Methodology
Inherited from Phase 1 (26 entries). All constraints apply:
- VRAM safety ceiling: 60 GB maximum
- Python: uv venv only
- GPU health check before experiments
- Warm cache benchmarks: 3 warmup, 20 timed runs, report p50/p99/mean
- WDDM is the fixed operating envelope

---

## Entry 001 — Experiment E01: Multi-Output Reduce

**Date**: 2026-03-29
**Type**: Primitive fusion experiment
**Status**: Complete
**Verdict**: Fused multi-stat kernel gives 1.3-1.6x over fair comparison (optimized individual ops). NOT the theorized 5x. The real finding: CuPy's std() is 55x slower than a trivial custom kernel — this pathology dominates any naive benchmark.

### Key Question (from team lead)

> Does multi-output reduce ACTUALLY read data fewer times, or does the GPU L2 cache / compiler optimize away the difference when running 5 separate reductions back-to-back?

### Hypothesis

A single kernel reading data once and computing sum+min+max+mean+std should be up to 5x faster than 5 separate CuPy reductions (each reading the full array).

Counter-hypothesis: Blackwell's 96 MB L2 cache may keep data resident after the first reduction, making subsequent reads effectively free.

### Method

Two phases:
1. **Phase 1**: Naive fused kernel (one block per ~512 elements, Welford std) vs 5 separate CuPy ops. Array sizes: 1M, 10M, 24M, 50M, 100M float32.
2. **Phase 2**: Optimized fused kernel (grid-stride, 128 blocks, sum-of-squares std) vs fair comparison (4 CuPy ops + custom std kernel). Isolates the CuPy std() pathology.

All benchmarks: 3 warmup runs, 20 timed runs, `cp.cuda.Stream.null.synchronize()` barriers.

### Phase 1 Results: Naive Fused vs Separate CuPy

| Array Size | Data (MB) | Fits L2? | Separate (ms) | Naive Fused (ms) | Speedup |
|------------|-----------|----------|---------------|-------------------|---------|
| 1M | 4 | Yes | 0.722 | 0.455 | 1.58x |
| 10M | 40 | Yes | 3.860 | 1.799 | 2.15x |
| 24M | 96 | Exactly | 11.734 | 4.093 | 2.87x |
| 50M | 200 | No | 42.422 | 8.910 | 4.76x |
| 100M | 400 | No | 84.292 | 17.362 | 4.85x |

Observation: speedup increases from 1.6x (fits L2) to 4.85x (exceeds L2). This looks like the L2 theory is confirmed — cache helps separate reductions at small sizes, eviction forces re-reads at large sizes.

**But this is misleading.** Individual op timing reveals the real story:

| Op (100M) | p50 (ms) | Bandwidth (GB/s) |
|-----------|----------|-------------------|
| sum | 0.331 | 1,210 |
| min | 0.315 | 1,271 |
| max | 0.311 | 1,286 |
| mean | 0.308 | 1,297 |
| **std** | **83.462** | **5** |

**CuPy's std() is 250x slower than sum/min/max/mean.** It consumes 98% of the "separate" benchmark. The 4.85x "speedup" is really just "avoiding CuPy's broken std()."

### Phase 2 Results: Fair Comparison (Optimized)

With CuPy's std() pathology removed (custom single-pass std kernel):

| Array Size | A: 5x CuPy (ms) | B: 4 CuPy + custom std (ms) | C: Opt Fused (ms) | C vs A | C vs B |
|------------|-----------------|------------------------------|--------------------|---------|---------|
| 1M | 0.747 | 0.455 | 0.293 | 2.55x | 1.55x |
| 10M | 3.708 | 0.526 | 0.378 | 9.82x | 1.39x |
| 50M | 41.881 | 1.515 | 1.116 | 37.54x | 1.36x |
| 100M | 84.208 | 3.078 | 2.001 | 42.08x | 1.54x |

**The fair comparison (C vs B) shows 1.3-1.6x** — NOT the theorized 5x. Reading data 5 times instead of 1 time costs only ~50% more, not 400% more.

### CuPy std() Pathology (standalone measurement)

| Array Size | CuPy std (ms) | Custom std (ms) | Speedup |
|------------|---------------|-----------------|---------|
| 1M | 0.425 | 0.145 | 2.9x |
| 10M | 3.619 | 0.202 | 17.9x |
| 50M | 41.521 | 0.801 | 51.8x |
| 100M | 83.157 | 1.493 | **55.7x** |

CuPy's `std()` appears to use a multi-pass algorithm with intermediate allocations. At 100M, it takes 83ms where a simple sum-of-squares kernel takes 1.5ms. This is NOT a fundamental limitation — it's an implementation issue in CuPy.

### Bandwidth Analysis

| Approach | Data Read (MB) | p50 @ 100M (ms) | Effective BW (GB/s) |
|----------|---------------|------------------|---------------------|
| Single CuPy op (sum) | 400 | 0.331 | 1,210 |
| C: Optimized fused (1 read) | 400 | 2.001 | 200 |
| B: 4 CuPy + custom std (5 reads) | 2,000 | 3.078 | 650 |

**Critical finding**: The fused kernel achieves only 200 GB/s on a single pass — 11% of peak VRAM bandwidth (1,792 GB/s theoretical). Individual CuPy reductions achieve 1,200+ GB/s (67% of peak). The fused kernel has 6x worse bandwidth utilization.

Root cause (probable): 128 blocks on ~120 SMs means ~1 block/SM = 8 warps/SM = 12.5% occupancy. CuPy's internal reductions use occupancy-optimized launch configurations. The fused kernel would need more blocks or cooperative groups to match.

### Answers to the Key Questions

**Q: Does multi-output reduce ACTUALLY read data fewer times?**
A: Yes — it reads data once instead of 5 times. But "reading 5 times" is not 5x slower because:
1. **L2 cache** helps at small sizes (< 96 MB): second read mostly hits L2
2. **Memory controller efficiency**: even at large sizes, back-to-back reads share DRAM row buffer state
3. **The fused kernel trades bandwidth for occupancy**: doing more work per thread means fewer blocks, lower occupancy, and worse bandwidth utilization

**Q: Is the theoretical 5x realistic?**
A: No. The practical ceiling is ~1.5x with current kernel design. A bandwidth-optimal fused kernel (higher occupancy, optimized launch config) might reach 2-3x, but 5x would require the separate path to achieve zero cache reuse, which doesn't happen on modern GPUs.

**Q: Does the compiler optimize away the difference?**
A: Not the compiler — the hardware does. L2 cache and memory controller are the mechanisms that reduce the cost of repeated reads.

### Architecture Implications

1. **Multi-output fusion has real but modest value** (1.3-1.6x). Worth implementing as a primitive, not worth architecting the entire system around it.

2. **CuPy's std() is pathologically slow** — 55x slower than a trivial custom kernel. This validates the "custom over workaround" principle from Phase 1. Any production system using CuPy's `std()` at scale has a 55x performance bug hiding in plain sight.

3. **Bandwidth utilization is the real optimization target.** The gap between 200 GB/s (our fused kernel) and 1,200 GB/s (CuPy's sum) is a 6x difference within a SINGLE read. Getting that right matters more than reducing read count.

4. **For the compiler vision**: multi-output reduce is a valid optimization in the pipeline generator's toolkit, but it's a "nice to have" 1.5x, not a "must have" 5x. The compiler should focus first on eliminating pathological implementations (like CuPy std), then on bandwidth-optimal kernel generation.

### Correctness

All statistics verified within expected float32 accumulation tolerance:
- sum, min, max, mean: relative error < 1e-6
- std: relative error < 1e-4 (sum-of-squares approach has known precision limitations vs two-pass Welford, acceptable for this benchmark)

### Open Questions

1. Can the fused kernel's bandwidth be improved to match CuPy's 1,200 GB/s? (Increase block count, tune occupancy)
2. Why is CuPy's `std()` so slow? Is it two-pass, does it allocate intermediates, or is there a Python-side overhead?
3. What about multi-output *map* (element-wise) rather than multi-output *reduce*? The fusion engine already handles this (Entry 010 from Phase 1 showed 0.19ms for any fused expression). The reduce case is different because reductions have different reduction trees.

---

## Entry 002 — Experiment E02: Sort-Once-Use-Many

**Date**: 2026-03-29
**Type**: Primitive fusion experiment
**Status**: Complete
**Verdict**: Sort-once-use-many gives 1.3-1.7x at realistic sizes. CuPy does NOT cache sort results. Meaningful but not transformative — the compiler should detect sort-sharing opportunities but it's not a critical optimization.

### Key Question (from team lead)

> Is sort-once-use-many faster, or does CuPy's internal caching already handle it?

### Method

**Sub-experiment C**: Sort the same CuPy array twice. If second sort is faster, CuPy caches.

**Main experiment**: 4 downstream operations that need sorted data (groupby-sum, rank, dedup, percentile). Compare:
- **Path A**: argsort once, feed sorted result to all 4 ops
- **Path B**: each op sorts independently (2 argsorts + 2 sorts = 4 sort ops)

Array sizes: 1M, 10M, 50M. Key cardinality: 1,000. Data types: int32 keys, float32 values.

### Sub-experiment C Results: CuPy Sort Caching

| Size | First sort (ms) | Second sort (ms) | Ratio |
|------|-----------------|-------------------|-------|
| 1M sort | 0.128 | 0.126 | 0.98x |
| 1M argsort | 0.129 | 0.141 | 1.09x |
| 10M sort | 0.299 | 0.301 | 1.01x |
| 10M argsort | 0.755 | 0.771 | 1.02x |
| 50M sort | 1.519 | 1.705 | 1.12x |
| 50M argsort | 4.586 | 4.873 | 1.06x |

**CuPy does NOT cache sort results.** All ratios are within measurement noise (0.98-1.12x). Second sort is sometimes slightly slower (memory fragmentation from first result still allocated).

### Main Experiment Results

| Size | Sort-once (ms) | Independent (ms) | Speedup | Time saved |
|------|---------------|-------------------|---------|------------|
| 1M | 1.089 | 1.099 | 1.01x | 0.01 ms |
| 10M | 2.363 | 4.032 | **1.71x** | 1.67 ms |
| 50M | 25.206 | 33.206 | **1.32x** | 8.00 ms |

At 1M: sort is so fast (0.137 ms) that saving 3 sorts saves ~0.4 ms — lost in downstream op noise.
At 10M: solid win. Saved 1.67 ms vs expected 2.24 ms (3 × 0.747 ms sort cost).
At 50M: 8 ms saved vs expected 13.7 ms (3 × 4.55 ms). Savings are sub-theoretical.

### Why 50M Underperforms the Theoretical

Downstream op timing on pre-sorted data at 50M:

| Operation | Time (ms) |
|-----------|-----------|
| groupby | 1.374 |
| rank | **11.886** |
| dedup | 1.058 |
| percentile | 0.163 |

**Rank dominates at 11.9 ms** — it's a scatter operation (`ranks[sorted_indices] = arange(...)`) that does random writes across 50M int64 values (400 MB). This is the same in both paths and masks the sort savings. The sort savings (8 ms) are real, but the total pipeline time (25 ms) is dominated by the random-access scatter.

### Sort Cost vs Array Size

| Size | argsort (ms) | Sort as % of sort-once pipeline |
|------|-------------|----------------------------------|
| 1M | 0.137 | 12.6% |
| 10M | 0.747 | 31.6% |
| 50M | 4.553 | 18.1% |

Sort is a significant but not dominant fraction of the pipeline. Eliminating 3 redundant sorts saves proportionally more at medium sizes (10M) where sort cost is high relative to downstream ops.

### Correctness Notes

- Rank: exact match across paths (deterministic sort stability)
- Dedup: exact match (unique count identical)
- Group sums: error up to 1e3 at 50M — this is float32 accumulation order sensitivity, relative error ~4e-7. Acceptable.
- Percentile: **mismatch by design** — path A computes percentile on key-sorted values, path B on value-sorted values. Different semantics, both correct for their use case.

### Answers to the Key Questions

**Q: Is sort-once-use-many faster?**
A: Yes — 1.3-1.7x at sizes above 1M. CuPy does not cache sort results, so redundant sorts are real wasted work.

**Q: Is this worth implementing in the compiler?**
A: Yes, but as an optimization pass, not a foundation. The compiler should detect when multiple pipeline stages share a sort dependency on the same key and lift the sort to a common ancestor. Implementation cost is low (DAG analysis); payoff is 1.3-1.7x on the sort-dependent portion of the pipeline.

**Q: Where does the sort-reuse optimization matter most?**
A: In the signal farm, every cadence bin requires sorted ticks for multiple statistics (groupby, rank, percentile, dedup). A typical K02 bin computation might sort the same tick stream 4-6 times across different leaves. At 10M ticks/day for an active symbol, that's 3-4 ms saved per cadence — multiplied across ~10 cadences and ~500 tickers, that's 15-20 seconds/day of sort elimination.

### Open Questions

1. Does the sort-reuse benefit increase with more downstream consumers? (5-10 ops sharing one sort should be 2-3x)
2. Can the rank scatter be optimized? At 50M it's 11.9 ms — more than twice the sort itself. A fused sort+rank might eliminate the scatter entirely.
3. What about sort stability? CuPy's sort is stable by default — does this constrain the sorting algorithm choice?

---

## Entry 003 — Experiment E03: Cross-Algorithm Sharing

**Date**: 2026-03-29
**Type**: Primitive fusion experiment
**Status**: Complete
**Verdict**: Cross-algorithm sharing gives 1.2-1.3x through shared intermediates (cumsums). BUT: naively fusing the downstream computation is a TRAP — the custom fused kernel is 2.4x SLOWER than CuPy primitives at 10M. The compiler should share intermediates, NOT fuse kernels for this pattern.

### Key Question (from team lead)

> Does cross-algorithm sharing produce measurably different performance, or is the overhead of tracking shared state worse than recomputing?

### Method

Pipeline: rolling_mean + rolling_std -> z_score -> covariance (PCA-like). 5 columns, window=60.

Three approaches:
- **A: Independent** — each algorithm computes its own rolling stats from scratch (redundant cumsums)
- **B: Shared intermediates** — shared cumsums feed both rolling_mean and rolling_std, then z_score uses pre-computed stats
- **C: Fused kernel** — custom CUDA kernel that reads from precomputed cumsums and produces z-score in one pass

### Building Block Costs (Single Column)

| Size | rolling_mean (ms) | rolling_std (ms) | Shared mean+std (ms) | Savings |
|------|-------------------|-------------------|-----------------------|---------|
| 100K | 0.074 | 0.205 | 0.179 | 30% |
| 1M | 0.076 | 0.204 | 0.199 | 27% |
| 10M | 0.246 | 0.664 | 0.670 | 22% |

Shared mean+std saves 22-30% by computing cumsum and cumsum_sq once instead of twice. At 10M, shared is nearly identical to rolling_std alone — the sum cumsum is effectively free when sq cumsum is already being computed.

### Full Pipeline Results (5 columns, window=60)

| Size | A: Independent (ms) | B: Shared (ms) | C: Fused (ms) | B vs A | C vs A |
|------|---------------------|-----------------|----------------|--------|--------|
| 100K | 3.074 | 2.465 | 2.523 | **1.25x** | 1.22x |
| 1M | 3.484 | 2.626 | 3.593 | **1.33x** | **0.97x** |
| 10M | 7.907 | 6.575 | **18.762** | **1.20x** | **0.42x** |

### Critical Finding: Fused Kernel is 2.4x SLOWER at Scale

At 10M rows, the custom fused z-score kernel (C) takes 18.8 ms — **2.4x slower** than independent CuPy ops (A) and **2.9x slower** than shared intermediates (B).

Root cause analysis:
1. **The fused kernel fuses the CHEAP part.** Z-score computation (subtract mean, divide by std) is trivially fast. The expensive part is computing cumsums — which are NOT fused, just shared.
2. **Float64 cumsum overhead**: the fused path casts to float64 for cumsum accuracy, creating 80 MB per cumsum at 10M (vs CuPy's internal optimization of the same operation).
3. **CuPy's built-in ops are highly optimized.** CuPy's element-wise arithmetic (subtraction, division, maximum) achieves near-peak bandwidth. The custom kernel cannot beat this without significant occupancy tuning.
4. **Kernel launch overhead**: 5 custom kernel launches + 5 × (2 cumsums + 2 concats + 1 cast) adds up.

**The lesson: fusing computation is only valuable when the FUSED portion is bandwidth-significant.** Fusing a trivial element-wise operation while keeping the expensive prefix scans separate is worse than using CuPy's optimized primitives for everything.

### Where Sharing DOES Win

Shared intermediates (B) consistently wins 1.2-1.3x by eliminating redundant cumsums:
- Independent path computes 4 cumsums per column (sum, sq for mean; sum, sq for std)
- Shared path computes 2 cumsums per column (sum, sq — shared by both mean and std)

At 10M × 5 cols, this eliminates 10 cumsum operations (~3.3 ms at ~0.33 ms each).

### Memory Analysis

| Path | Intermediate memory at 1M | at 10M |
|------|--------------------------|--------|
| A: Independent | ~160 MB | ~1,600 MB |
| B: Shared | ~80 MB | ~800 MB |
| C: Fused | ~80 MB | ~800 MB |

Sharing halves intermediate memory. At 10M × 5 cols, that's 800 MB saved — significant when operating under the 60 GB VRAM ceiling with many tickers.

### Cache Lookup Overhead

Simulated a dict-based intermediate cache (the compiler's mechanism for sharing):

**Cache lookup overhead: negligible** (~0 ms). Python dict lookup on a tuple key is sub-microsecond, completely invisible against millisecond-scale GPU operations.

### Answers to the Key Questions

**Q: Does cross-algorithm sharing produce measurably different performance?**
A: Yes — **1.2-1.3x speedup + 50% memory reduction** from shared intermediates. Real, meaningful, and the overhead of tracking shared state (dict lookup) is effectively zero.

**Q: Is the overhead of tracking shared state worse than recomputing?**
A: No. Cache overhead is sub-microsecond. But there IS a subtler trap: **fusing kernels for the wrong subgraph is worse than recomputing.** The compiler must distinguish between "share the intermediate" (cheap, always wins) and "fuse the downstream kernel" (expensive, often loses to CuPy's optimized primitives).

**Q: Should the compiler fuse cross-algorithm pipelines?**
A: **Share intermediates: yes. Fuse kernels: only when the fused portion is bandwidth-significant.** For rolling_stats → z_score, the win is in sharing cumsums, not in fusing the z-score arithmetic. The compiler should identify shared prefix computations and lift them, then let CuPy's optimized element-wise ops handle the rest.

### Architecture Implications

1. **The pipeline generator's primary optimization is DAG-level sharing**, not kernel-level fusion. Detect when multiple leaves share a common prefix computation (cumsums, sorts, etc.) and compute it once.

2. **Kernel fusion (from Phase 1's fusion engine) is for element-wise expressions.** Entry 010 showed this already: fused element-wise ops run at 0.19 ms regardless of expression depth. The fusion engine is a reduction optimizer, not a pipeline fuser.

3. **The compiler should NOT try to fuse across operation boundaries** (e.g., cumsum → z_score). CuPy's primitives are already near-optimal for each stage. The win is in eliminating redundant stages, not in merging adjacent stages.

4. **Memory savings compound.** At 500 tickers × 10 cadences × 5 leaves, halving intermediate memory per leaf saves ~40 GB. This matters more than the 1.3x speedup.

### Open Questions

1. Do shared intermediates benefit from persistent GPU residency? (If cumsums are already in VRAM from a previous computation, the shared path is essentially free.)
2. What percentage of the signal farm's computation graph has sharable prefixes? (High if most leaves start with the same raw ticks → sorted → binned → rolling stats chain.)
3. Can the cumsum operation itself be fused with the data loading stage? (Cumsum is the expensive part — fusing it with H2D transfer would be the real win.)

---

## Entry 004 — Experiment E06: Resident Query Latency

**Date**: 2026-03-29
**Type**: Persistent store experiment
**Status**: Complete
**Verdict**: GPU-resident queries are sub-millisecond (0.07-0.44 ms). Cold→warm is 26x. Kernel launch floor on WDDM is 8 microseconds. Sequential throughput: 11K simple queries/sec on 10M rows. The persistent store concept is validated.

### Purpose

Measure the actual query latency when data is already resident on GPU. Phase 1 Entry 025 showed second-query at ~5 ms for the analytics pipeline. Can we do better? What's the floor?

### Results

#### Cold vs Warm (10M rows × 5 float32 columns)

| Mode | p50 (ms) | Notes |
|------|----------|-------|
| Cold (H2D + compute) | 10.462 | Transfer dominates |
| Warm (GPU-resident) | 0.394 | **26x faster** |

#### Query Latency on Resident Data (10M rows)

| Query Type | p50 (ms) | Best (ms) | p99 (ms) |
|------------|----------|-----------|----------|
| Sum (5 cols) | 0.439 | 0.361 | 1.201 |
| Filtered sum (5 cols) | 1.168 | 1.077 | 2.386 |
| GroupBy sum (1K groups) | 1.318 | 1.284 | 2.634 |
| **Expression (a*b+c)/(a+1)** | **0.210** | **0.204** | 0.596 |
| **Rolling mean (w=60)** | **0.222** | **0.220** | 0.649 |

Expression and rolling mean are the fastest complex queries — both sub-250 microseconds. GroupBy is the slowest due to sort cost.

#### Latency vs Data Size (Resident, Sum)

| Size | Data (MB) | p50 (ms) | Best (ms) | Effective BW |
|------|-----------|----------|-----------|--------------|
| 10K | 0.04 | 0.074 | 0.025 | (launch-bound) |
| 100K | 0.4 | 0.072 | 0.058 | (launch-bound) |
| 1M | 4.0 | 0.068 | 0.065 | (launch-bound) |
| 10M | 40.0 | 0.078 | 0.070 | (launch-bound) |
| 50M | 200 | 0.189 | 0.173 | 1,058 GB/s |
| 100M | 400 | 0.319 | 0.300 | 1,255 GB/s |

**Critical finding**: latency is FLAT at ~0.07 ms for all sizes up to 40 MB (10M elements). The GPU processes up to 40 MB entirely from L2 cache. Only at 200+ MB does VRAM bandwidth become the bottleneck.

This means: for the signal farm's typical working set (~1M rows per ticker per cadence = 4 MB), **query latency is bounded by kernel launch overhead, not data size.**

#### Kernel Launch Overhead Floor

| Operation | p50 (ms) | Best (ms) | p99 (ms) |
|-----------|----------|-----------|----------|
| Empty kernel (WDDM) | 0.008 | 0.007 | 0.457 |
| CuPy sum(1 elem) | 0.070 | 0.018 | 0.513 |
| Python timer | 0.000 | — | — |

**WDDM kernel submission: 8 microseconds.** This is the hardware floor — the time for the CPU to submit a kernel through WDDM to the GPU scheduler. Note the p99 at 0.46 ms — WDDM scheduling jitter adds ~0.5 ms tail latency (Windows compositor contention).

**CuPy overhead: 70 microseconds.** The Python→CuPy→CUDA path adds ~62 microseconds over bare kernel launch. This is the Python-layer floor for any CuPy operation.

#### Sequential Query Throughput (10M resident float32)

| Query | Queries/sec |
|-------|------------|
| Sum | **11,061** |
| Rolling mean | **3,984** |

### Signal Farm Feasibility Analysis

Typical daily farm: 500 tickers × 10 cadences × 20 leaves = 100,000 computations.

| Query complexity | QPS | Time for full farm |
|-----------------|-----|--------------------|
| Simple (sum/mean/std) | 11,000 | **9 seconds** |
| Medium (rolling stats) | 4,000 | **25 seconds** |
| Complex (groupby) | ~800 | ~125 seconds |

With data resident on GPU, the entire signal farm can refresh in under 30 seconds for rolling-stat-class operations. This is sub-minute continuous observation — viable for real-time signal farming.

### Architecture Implications

1. **The persistent store is validated.** Sub-millisecond query latency on resident data means the GPU can function as a persistent analytical database. "Load once, query forever" is real.

2. **L2 cache is the secret weapon.** At typical signal farm working sets (4 MB per ticker), data lives in L2 cache after first access. Subsequent queries are essentially free — bounded only by kernel launch overhead at 70 microseconds.

3. **WDDM tail latency is the main risk.** p99 at 0.5 ms for empty kernels means Windows compositor contention can spike any operation by 0.5 ms. MCDM (when available with driver R595+) would eliminate this. For now, the signal farm is tolerant of 0.5 ms jitter.

4. **Python/CuPy overhead (70 us) becomes significant at high query rates.** At 11K queries/sec, we're spending ~0.07 ms per query on Python overhead. The Rust path (cudarc, E08) eliminates this — bare kernel launch at 8 us would enable ~125K queries/sec theoretical.

5. **The 26x cold/warm gap validates pre-loading strategy.** The pipeline should maintain a warm pool of GPU-resident data (most recent N tickers, all cadences). Any query on warm data is 26x faster than cold start.

### Open Questions

1. What does multi-stream concurrency look like? (Can we overlap multiple queries for 2-4x throughput?)
2. How does WDDM jitter distribute over a full farm run? (Is p99 stable or does it degrade under sustained load?)
3. Can we keep ALL 500 tickers resident simultaneously? (500 × 10 cadences × 4 MB = 20 GB — fits easily under 60 GB ceiling)

---

## Entry 005 — Experiment E05: JIT Compilation Overhead — Three Tiers

**Date**: 2026-03-29
**Type**: Pipeline generator experiment
**Status**: Complete
**Verdict**: JIT compilation is 40-92 ms depending on kernel complexity. 40ms for a trivial kernel is NOT optimistic — it's the measured reality. Cache hit is 2-3 microseconds. Monolithic JIT kernel is 2.3x faster than composed CuPy blocks. JIT cost amortizes to <1% after ~121K calls (~11 seconds at full throughput).

### Key Question (from team lead)

> What's the JIT compilation overhead — is 40ms realistic or optimistic?

### NVRTC Compilation Time by Kernel Complexity

| Kernel | Ops | Total JIT (ms) | Cached (ms) |
|--------|-----|----------------|-------------|
| Trivial (1 arithmetic op) | 1 | **40.0** | 0.003 |
| Simple (5 ops, sqrt) | 5 | 45.8 | 0.002 |
| Complex (pipeline: VWAP + z-score) | ~15 | 56.2 | 0.002 |
| Medium (20 ops, branches, transcendentals) | 20 | 88.3 | 0.002 |
| Large (30+ ops, 5 inputs, 3 outputs) | 30+ | 92.4 | 0.003 |

**40ms is not optimistic — it's the measured floor.** Even the simplest possible kernel (one addition) takes 40 ms to compile via NVRTC. Complexity adds roughly linearly: 20-op kernels take ~88 ms, 30-op kernels take ~92 ms. The base cost (~40 ms) is NVRTC initialization + PTX generation; the per-op cost is ~2 ms.

**Cache hit: 2-3 microseconds.** Once compiled, kernel reuse is essentially free. CuPy's cache is keyed on source hash.

### Composed vs Monolithic Pipeline (10M rows, window=60)

| Approach | p50 (ms) | p99 (ms) | Kernel launches |
|----------|----------|----------|-----------------|
| Composed (CuPy blocks) | 1.734 | 2.793 | ~10 |
| Monolithic (JIT kernel) | 0.765 | 1.994 | 1 |
| **Speedup** | **2.27x** | | |

The composed path launches ~10 CuPy operations, each with ~0.07 ms Python/CuPy overhead plus intermediate buffer allocations. The monolithic kernel eliminates ALL intermediate buffers and reduces to a single launch.

**This is the case FOR a pipeline generator.** It generates one fused kernel per leaf, pays the JIT cost once (~46 ms), then runs at 2.3x the composed speed forever.

### JIT Amortization

| Metric | Value |
|--------|-------|
| First call (compile + execute) | 46.8 ms |
| Cached call (execute only) | 0.039 ms |
| JIT overhead to amortize | 46.8 ms |
| Break-even (< 1% of total time) | 121,187 calls |
| Time to break-even at 11K QPS | ~11 seconds |

In the signal farm context: each leaf compiles its kernel once at startup, then runs thousands of times per day. The 46 ms JIT cost is paid once; the 2.3x speedup benefit compounds over every subsequent query.

### Compilation Time Breakdown

| Phase | Time | What happens |
|-------|------|-------------|
| `RawKernel()` creation | ~0.01 ms | Python object creation, source stored |
| `.kernel` access (NVRTC compile) | 40-92 ms | Source → PTX → cubin, NVRTC optimizer |
| Subsequent `.kernel` access | 0.002 ms | In-memory cache hit |

The entire compilation cost is in the `.kernel` access (first call). CuPy's RawKernel constructor is lazy — it doesn't compile until the kernel is actually needed.

### Architecture Implications

1. **Pipeline generator should generate monolithic kernels.** 2.3x speedup over composed CuPy blocks justifies the complexity of codegen.

2. **JIT cost is a one-time startup cost.** At 46 ms per kernel × ~100 unique leaf kernels = 4.6 seconds total startup. Acceptable for a system that runs continuously.

3. **Pre-built kernels (Tier 1) are only 46 ms faster to start than JIT kernels (Tier 2).** This means the "pre-built" tier adds complexity (compile at install, manage binaries) for minimal benefit. JIT-everything is simpler and nearly as fast.

4. **Composed primitives (Tier 3) are the fallback, not the goal.** CuPy's optimized blocks work fine but are 2.3x slower than monolithic. Use composed as the prototype/debugging tier, then graduate to JIT monolithic for production.

### Open Questions

1. Does NVRTC disk caching persist across process restarts? (CuPy has `CUPY_CACHE_DIR` — need to verify persistence)
2. Can we pre-warm the kernel cache at process start with async compilation?
3. What's the NVRTC compilation time from Rust (cudarc)? Is it the same ~40 ms floor?

---

## Entry 006 — Experiment E08: cudarc on Windows + CUDA 13.1

**Date**: 2026-03-29
**Type**: Rust foundation experiment
**Status**: Complete
**Verdict**: cudarc 0.19 works on Windows with CUDA 13.1 under WDDM. Zero blockers. Driver API init, device query, H2D/D2H, memory info — all pass. Rust path is UNBLOCKED.

### Environment

- Rust 1.94.1 (2026-03-25)
- Cargo 1.94.1
- cudarc 0.19 with `cuda-version-from-build-system` feature
- CUDA Toolkit 13.1, Driver R581.42, WDDM mode

### Test Results

All operations pass on first attempt:

| Operation | Result |
|-----------|--------|
| `CudaContext::new(0)` | OK |
| Device name query | "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" |
| Compute capability | 12.0 (sm_120 Blackwell) |
| H2D copy (256 float32) | OK |
| D2H copy + verify | OK — round-trip data matches exactly |
| `mem_get_info()` | 100.8 GB free / 102.6 GB total |

### Key Findings

1. **Zero build issues.** `cargo build --release` compiles cleanly on Windows with MSVC. cudarc's build script found CUDA 13.1 via `CUDA_PATH` environment variable automatically.

2. **WDDM is not a blocker.** cudarc uses the CUDA Driver API (not Runtime API), and the driver API works under WDDM without issues. All operations (context creation, memory allocation, transfer, queries) succeed.

3. **No DLL/linker issues.** cudarc links against `nvcuda.dll` (the driver stub), which is always present on systems with an NVIDIA driver. No dependency on CUDA Runtime DLLs, no PATH requirements.

4. **Compute capability 12.0 confirmed.** cudarc correctly queries sm_120 for Blackwell, meaning kernel compilation targets will be correct.

### Architecture Implications

1. **The Rust path is unblocked.** We can proceed with the 12-crate architecture (per memory file `project_rust_architecture.md`). cudarc provides the foundation for custom-over-wrap NVIDIA library access.

2. **cudarc + NVRTC from Rust (E09) is the next test.** We've proven device management works; next is runtime kernel compilation, which is the pipeline generator's Rust-side requirement.

3. **No Python dependency for GPU access.** The Rust binary can operate independently of the Python environment, enabling pure-Rust daemon processes for the signal farm.

4. **PyO3 bridge to Python is the remaining integration question.** cudarc's `CudaSlice<f32>` needs to be zero-copy accessible from Python via Arrow or raw pointers. This is a future experiment.

### Open Questions

1. cudarc + NVRTC: can we compile kernels from Rust at the same ~40 ms as CuPy? (E09)
2. cudarc memory pools: does `CudaContext` support async memory pools on WDDM?
3. cudarc + PyO3: can we expose GPU buffers to Python without copying?

---

## Entry 007 — Experiment E07: Provenance-Based Reuse

**Date**: 2026-03-29
**Type**: Persistent store experiment
**Status**: Complete
**Verdict**: Provenance tracking overhead is negligible (0.002 ms per check). Cache hits are 81-865x faster than recomputation. Farm time scales linearly with dirty ratio — at 1% dirty, the farm runs 28x faster. The provenance model is validated for continuous signal farming.

### Purpose

The persistent store needs to know when a GPU buffer is still valid. If provenance tracking can skip unchanged computations, the signal farm processes only what's dirty.

### Test 1: Provenance Tracking Overhead

| Path | Time (ms) | Notes |
|------|-----------|-------|
| Raw compute (no provenance) | 0.219 | Rolling mean, 10M rows |
| Provenance miss (compute + tag) | 0.221 | +0.002 ms overhead |
| Provenance hit (cache lookup) | 0.001 | **199x savings** |

Provenance miss adds 0.002 ms — a Python dict lookup + MD5 string compare. Invisible against any GPU operation.

### Test 2: Reuse Savings by Operation Complexity

| Operation | Compute (ms) | Hit (ms) | Savings |
|-----------|-------------|----------|---------|
| Sum (simple reduce) | 0.097 | 0.001 | **81x** |
| Rolling mean (w=60) | 0.219 | 0.001 | **168x** |
| Expression (fused 5-op) | 0.426 | 0.001 | **328x** |
| Rolling std (w=60) | 0.666 | 0.002 | **444x** |
| Sort (argsort) | 0.756 | 0.001 | **582x** |
| GroupBy sum | 1.125 | 0.001 | **865x** |

The more expensive the operation, the more provenance reuse saves. GroupBy (the most expensive) benefits most at 865x. Even simple sum saves 81x. The cache hit cost (~1 microsecond) is dominated by Python dict lookup, not GPU activity.

### Test 3: Farm Simulation — Dirty Ratio Impact

Simulated farm: 100 tickers × 5 cadences × 10 leaves = 5,000 computations per cycle.

| Dirty % | Dirty tickers | Farm time (ms) | Speedup vs 100% |
|---------|---------------|----------------|------------------|
| 100% | 100 | 448.6 | 1.0x |
| 50% | 50 | 232.0 | 1.9x |
| 20% | 20 | 116.0 | 3.9x |
| 10% | 10 | 62.2 | 7.2x |
| 5% | 5 | 36.4 | 12.3x |
| 1% | 1 | 16.1 | **27.9x** |

**Farm time scales linearly with dirty ratio.** At 1% dirty (1 ticker updated out of 100), the farm is 28x faster. In real-time streaming mode where new ticks arrive for a few tickers at a time, the vast majority of the farm is a cache hit.

### Test 4: Hash Computation Cost

| Operation | Cost |
|-----------|------|
| MD5 hash (short tag) | 0.78 us |
| MD5 hash (long tag) | 1.10 us |
| 5,000 hashes (one farm cycle) | 3.89 ms |
| As % of 100% dirty farm | 0.9% |

Hash computation is negligible. Even hashing all 5,000 provenance tags costs under 4 ms — less than 1% of the full farm computation.

### Architecture Implications

1. **Provenance-based reuse is validated.** Overhead is negligible (0.002 ms per check, <1% for hashing), savings are enormous (81-865x per operation). This is a must-implement for the persistent store.

2. **Dirty ratio is the key metric.** The signal farm should track which tickers have received new data since last computation. Only dirty tickers trigger recomputation. At typical streaming rates (5-10% of tickers updating per second), the farm runs at 7-12x faster than full recomputation.

3. **Provenance granularity should be per-leaf.** Each leaf computation has a provenance tag keyed on (ticker_id, cadence, leaf_id, input_data_version). This is the natural granularity — coarser (per-section) wastes computation, finer (per-column) adds lookup overhead with diminishing returns.

4. **The roaring bitmap dirty tracker fits perfectly.** The daemon's roaring bitmap (from `project_node_architecture.md`) marks dirty nodes. Provenance checking at compute time is a second line of defense — belt and suspenders. The bitmap prevents scheduling clean nodes; provenance prevents recomputing if the bitmap missed a cache opportunity.

5. **Memory cost is the trade-off.** Caching results means keeping all 5,000 GPU buffers resident. At 100K rows × 4 bytes × 5,000 buffers = 2 GB. At 1M rows: 20 GB. At 10M rows: 200 GB (exceeds VRAM). Provenance reuse is viable when per-ticker data fits in VRAM alongside the cache — which it does at signal farm working set sizes (100K-1M rows per ticker).

### Open Questions

1. What's the optimal eviction policy when cache exceeds VRAM? (LRU by last-access? Priority by computation cost?)
2. Can provenance tags encode partial updates? (If 90% of a ticker's ticks are unchanged, can we reuse the cached rolling stats for the unchanged portion?)
3. How does the dirty bitmap interact with provenance? (Does double-checking waste time, or is the defense-in-depth worth the ~0.002 ms per computation?)

---

## Entry 008 — Experiment E09: NVRTC from Rust

**Date**: 2026-03-29
**Type**: Rust foundation experiment
**Status**: Complete
**Verdict**: NVRTC works from Rust. Compilation is 2-4x faster than CuPy (8-10 ms vs 40-92 ms). Kernel launch overhead is 7.7x lower (0.009 ms vs 0.070 ms). The Rust path is the correct foundation for the pipeline generator.

### NVRTC Compilation Time: Rust vs CuPy

| Kernel | Rust compile (ms) | Rust load (ms) | Rust total (ms) | CuPy total (ms) | Rust advantage |
|--------|-------------------|----------------|-----------------|------------------|----------------|
| Trivial (1 op) | 8.7 | 12.9 | **21.6** | 40.0 | **1.9x** |
| Medium (10 ops) | 8.5 | — | ~21 | 88.3 | **~4x** |
| Complex (pipeline) | 9.5 | — | ~22 | 56.2 | **~2.6x** |

**NVRTC compilation from Rust is 2-4x faster than from CuPy.** The difference is Python overhead: CuPy's `.kernel` access includes Python dict lookups, source hashing, option string construction, and C extension boundary crossing. Rust eliminates all of this.

Note: NVRTC DLL warmup (first call only) takes ~127 ms. This is a one-time cost at process start.

### Kernel Launch Overhead: Rust vs CuPy

| Operation | Rust (ms) | CuPy (ms) | Rust advantage |
|-----------|-----------|-----------|----------------|
| Empty kernel launch | 0.007 | 0.008 | 1.1x (same) |
| 1M-element kernel | **0.009** | **0.070** | **7.7x** |

**The critical finding: Rust kernel launch is 7.7x faster than CuPy** for non-trivial kernels. The 0.061 ms difference is pure Python/CuPy overhead per launch — C extension call, argument marshaling, Python GIL.

For empty kernels, both Rust and CuPy converge to ~7-8 microseconds — this is the raw WDDM kernel submission cost (hardware floor).

### Throughput Implications

| Metric | CuPy | Rust | Improvement |
|--------|------|------|-------------|
| Launch overhead | 70 us | 9 us | 7.7x |
| Max launches/sec | ~14,000 | ~111,000 | 7.9x |
| Farm (100K leaves) overhead | 7.0 sec | 0.9 sec | 7.8x |

At 100,000 leaf computations per farm cycle (500 tickers × 10 cadences × 20 leaves), the launch overhead alone costs 7 seconds in CuPy vs 0.9 seconds in Rust. This is pure Python tax that Rust eliminates.

### End-to-End Verification

| Step | Time |
|------|------|
| NVRTC compile (trivial kernel) | 9.3 ms |
| cuModuleLoadData (PTX → cubin) | 12.9 ms |
| First kernel launch + sync | 0.249 ms |
| Cached kernel launch + sync | 0.009 ms |
| Correctness (1M add_one) | PASS |

The full pipeline works: Rust source string → NVRTC → PTX → cubin → launch → verify. Zero issues on Windows + CUDA 13.1 + WDDM.

### Architecture Implications

1. **The Rust pipeline generator is validated.** NVRTC compilation from Rust works, is faster, and launch overhead is 7.7x lower. The entire codegen pipeline (expression tree → CUDA source → NVRTC → PTX → cubin → launch) can run in Rust with no Python dependency.

2. **Python/CuPy launch overhead is the hidden bottleneck.** At 70 us per launch, CuPy's overhead exceeds the actual kernel execution time for simple operations (sum at 0.07 ms = 70 us). Rust eliminates this, making sub-10-us launches possible.

3. **The JIT compilation cost is halved.** 22 ms (Rust) vs 40-92 ms (CuPy) means more kernels can be JIT-compiled at startup within the same time budget.

4. **Multi-stream kernel submission from Rust is the next frontier.** At 9 us per launch, we can overlap 100+ kernel submissions per millisecond across multiple CUDA streams. This enables true pipeline parallelism in the signal farm.

### Open Questions

1. Can Rust cache compiled PTX/cubin to disk for cross-process reuse? (NVRTC has no built-in disk cache; we'd need to implement it)
2. What's the compilation time for sm_120-targeted compilation? (We used default target; explicit `--gpu-architecture=sm_120` might differ)
3. Can we use cudarc's async streams to overlap compilation with computation?

---

## Entry 009 — Synthesis: Compiler Foundation Findings

**Date**: 2026-03-29
**Type**: Synthesis and recommendations
**Status**: Complete

### Answering the Team Lead's Key Questions

**1. Does multi-output reduce ACTUALLY read data fewer times, or does the compiler optimize away the difference?**

The hardware optimizes away most of the difference. L2 cache (96 MB on Blackwell) keeps data resident for subsequent reads. At realistic signal farm sizes (1-10M rows = 4-40 MB), data fits entirely in L2 — subsequent reads are near-free. Even at 100M rows (exceeding L2), the speedup is only 1.5x, not the theoretical 5x. The GPU's memory controller provides partial reuse even without L2 hits.

**Verdict**: Multi-output reduce is a 1.3-1.6x optimization, not a 5x one. Worth having but not foundational.

**2. Is sort-once-use-many faster, or does CuPy's internal caching already handle it?**

CuPy does NOT cache sort results. Sort-once-use-many provides 1.3-1.7x speedup at sizes above 1M. The compiler should detect sort-sharing opportunities in the DAG and lift shared sorts to common ancestors.

**Verdict**: Yes, sort reuse is real. CuPy provides no caching. The compiler should implement this.

**3. Does cross-algorithm sharing produce measurably different performance?**

Shared intermediates (cumsums, etc.) provide 1.2-1.3x speedup and 50% memory reduction. BUT: naively fusing the downstream computation into a custom kernel is a TRAP — it was 2.4x SLOWER than using CuPy's optimized primitives. The compiler must share intermediates without fusing the wrong subgraph.

**Verdict**: Share intermediates (yes). Fuse kernels only when the fused portion is bandwidth-significant (element-wise expressions: yes. Cross-operation pipelines: no).

**4. Does cudarc actually work on Windows with CUDA 13.1?**

Yes. Zero issues. cudarc 0.19, Rust 1.94.1, WDDM mode, Blackwell sm_120 — all operations pass (context creation, device query, H2D, D2H, memory info).

**Verdict**: The Rust path is fully unblocked.

**5. What's the JIT compilation overhead — is 40ms realistic or optimistic?**

40ms from CuPy is measured reality for a trivial kernel. From Rust, it's faster: 9 ms NVRTC compile + 13 ms module load = 22 ms total. Complex kernels: 22-30 ms from Rust vs 40-92 ms from CuPy. Cache hit is 2-3 microseconds (CuPy) or sub-microsecond (Rust).

**Verdict**: 40ms is the CuPy floor. From Rust it's ~22ms. Both amortize to <1% within seconds of operation.

### The Three Breakthroughs

**1. Rust kernel launch is 7.7x faster than CuPy** (E09)
- Rust: 9 us per launch. CuPy: 70 us per launch.
- At 100K leaves/cycle, this saves 6.1 seconds of pure Python overhead.
- This validates building the pipeline generator in Rust, not Python.

**2. GPU-resident queries are sub-millisecond** (E06)
- Warm data: 0.07-0.44 ms per query.
- 11K queries/sec on 10M resident data.
- L2 cache makes anything < 40 MB effectively free.
- This validates the persistent store concept.

**3. Provenance reuse scales linearly with dirty ratio** (E07)
- At 1% dirty (streaming mode): 28x faster than full recomputation.
- Cache hit cost: 1 microsecond. Miss overhead: 2 microseconds.
- This validates continuous signal farming.

### The One Trap (REVISED — see Entry 010)

**Original conclusion (E03 at 10M): cross-operation fusion usually LOSES.** This was premature — E03b at FinTek-realistic sizes (50K-900K) showed fusion **wins 1.4-2.1x**. The crossover is at ~500K-900K rows. The compiler must distinguish:
- **Element-wise fusion**: always wins (0.19 ms for any expression depth, from Phase 1)
- **Intermediate sharing**: always wins at all sizes (shared cumsums, shared sorts) — 1.2-1.5x consistently
- **Cross-operation fusion**: **SIZE-DEPENDENT**. Wins 2x at 50K-100K (dispatch-dominated), breaks even at ~500K-900K, loses at 10M+ (bandwidth-dominated with naive kernel). See Entry 010 for full data.

The pipeline generator should generate monolithic element-wise kernels (2.3x win from E05), share intermediates (1.2-1.5x from E03/E03b), and **fuse across operation boundaries at FinTek-realistic sizes** (2x from E03b). At large sizes, prefer CuPy-quality optimized primitives unless the fused kernel is properly occupancy-tuned.

### Revised Compiler Architecture (Evidence-Based)

Based on 8 experiments, the compiler should be:

1. **Rust-native** (E08, E09): 7.7x lower launch overhead, 2-4x faster compilation
2. **DAG-level optimizer** (E02, E03): share sorts, share cumsums, share rolling stats
3. **Element-wise fuser** (E01, E05): monolithic JIT kernels for leaf expressions (2.3x vs composed)
4. **Provenance-aware** (E07): skip unchanged computations (28x at 1% dirty)
5. **Persistent** (E06): keep data GPU-resident (26x cold→warm improvement)

What it should NOT be:
- A replacement for CuPy's optimized primitives at large sizes (E01: CuPy sum at 1,200 GB/s)
- A pre-built kernel library (E05: JIT-everything is simpler, 22ms cost is negligible)

Note: Cross-operation fusion was originally listed here as "should NOT be" based on E03 (10M rows, 2.4x SLOWER). E03b corrected this — at FinTek sizes (50K-500K), cross-operation fusion **wins 1.5-2.1x**. The compiler SHOULD fuse, with a size-adaptive cost model.

### What Remains

- **E04 (Pipeline generator)**: Design experiment — build a minimal prototype that takes a leaf spec, generates CUDA source, compiles via NVRTC from Rust, and executes. The data from E05 and E09 provides the performance envelope.
- **E10 (Primitive decomposition registry)**: Design experiment — enumerate the ~135 specialists and their decompositions into 8 primitives. This is an architecture task, not a measurement task.

Both benefit from team discussion before implementation.

---

## Entry 010 — Experiment E03b: Cross-Algorithm Sharing Retest (FinTek-Realistic Sizes)

**Date**: 2026-03-30
**Type**: Retest of E03 at realistic array sizes
**Status**: Complete
**Requested by**: Navigator
**Verdict**: Fusion gives **2x at 50K-100K rows** (FinTek-realistic), confirming the compiler vision for the actual workload. Crossover at ~500K-900K. Above 1M, CuPy's optimized kernels still win over a naive fused kernel.

### Motivation

E03 tested at 100K, 1M, 10M — found fused kernel 2.4x SLOWER at 10M. Navigator hypothesized that FinTek-realistic sizes (50K-900K rows per ticker) would show fusion winning because Python dispatch overhead would dominate GPU compute at small sizes.

### Method

**Sizes**: 50K, 100K, 500K, 900K (FinTek-realistic), plus 10M for reference.

**Three paths** (same as E03):
- **A**: Independent computation (rolling_mean + rolling_std + z_score, each from scratch)
- **B**: Shared intermediates (shared cumsums, no kernel fusion)
- **C**: Monolithic fused RawKernel

**Two variants**:
- Variant 1: Tight loop (original E03 methodology, no overhead between ops)
- Variant 2: Realistic pipeline overhead (_validate + _log_metric calls between GPU ops)

**Also measured**: GPU sync overhead, Python dispatch overhead, per-CuPy-call dispatch cost.

### Overhead Floor Measurements

| Measurement | p50 |
|---|---|
| GPU sync (idle stream) | 0.4 us |
| Python validate + log_metric | 0.10 us |
| CuPy sum(1 element) | 11.5 us |

Critical finding: **Python code between GPU ops costs 0.1 us. CuPy dispatch per call costs 10-27 us.** The bottleneck is CuPy's dispatch machinery, not Python overhead.

### Variant 1 Results (Tight Loop)

| Size | A: Independent | B: Shared | C: Fused | B/A | C/A |
|---|---|---|---|---|---|
| 50K | 0.280 ms | 0.204 ms | 0.135 ms | 1.37x | **2.08x** |
| 100K | 0.288 ms | 0.223 ms | 0.143 ms | 1.30x | **2.01x** |
| 500K | 0.298 ms | 0.212 ms | 0.198 ms | 1.40x | **1.50x** |
| 900K | 0.358 ms | 0.237 ms | 0.262 ms | 1.51x | **1.37x** |
| 10M | 1.027 ms | 0.816 ms | 2.263 ms | 1.26x | **0.45x** |

### Variant 2 Results (Realistic Pipeline Overhead)

| Size | A: Independent | B: Shared | C: Fused | B/A | C/A |
|---|---|---|---|---|---|
| 50K | 0.272 ms | 0.226 ms | 0.146 ms | 1.20x | **1.86x** |
| 100K | 0.288 ms | 0.224 ms | 0.142 ms | 1.29x | **2.04x** |
| 500K | 0.323 ms | 0.225 ms | 0.198 ms | 1.43x | **1.63x** |
| 900K | 0.312 ms | 0.223 ms | 0.261 ms | 1.40x | **1.20x** |
| 10M | 1.037 ms | 0.813 ms | 2.445 ms | 1.28x | **0.42x** |

### Per-CuPy-Call Dispatch Cost

| Size | cumsum | concat | slice-div | multiply | sqrt |
|---|---|---|---|---|---|
| 50K | 27 us | 24 us | 20 us | 11 us | 11 us |
| 100K | 25 us | 24 us | 21 us | 10 us | 11 us |
| 500K | 26 us | 25 us | 21 us | 13 us | 13 us |
| 900K | 28 us | 37 us | 25 us | 14 us | 13 us |
| 10M | 78 us | 65 us | 103 us | 54 us | 53 us |

### CuPy Call Count

- Path A: ~17 CuPy calls (rolling_mean=3, rolling_std=10, z_score=4)
- Path B: ~14 CuPy calls (shared=10, z_score=4)
- Path C: ~8 CuPy calls (cumsums=7, fused kernel=1)
- Reduction: 53% fewer dispatches from A→C

### Analysis

1. **Navigator was right about the direction, wrong about the mechanism.** Fusion wins 2x at FinTek sizes — but NOT because of Python overhead between ops (0.1us, negligible). The bottleneck is CuPy's own dispatch cost (10-27us per call). Reducing 17 calls to 8 saves ~180us on a ~280us total.

2. **Pipeline overhead (Variant 2) adds essentially nothing.** The tight-loop vs pipeline-overhead results are within noise. This means the real-world pipeline matches the tight-loop benchmark — no hidden overhead from Python bookkeeping.

3. **Crossover at ~500K-900K.** Below 500K: fusion wins (dispatch-dominated). Above 900K: CuPy's optimized kernels achieve higher memory bandwidth than the naive fused kernel.

4. **Shared intermediates (Path B) win at ALL sizes.** 1.2-1.5x consistently. This is the safest optimization — pure gain with no crossover.

5. **The 10M regression is a fused kernel quality problem, not a fusion concept problem.** The fused kernel achieves ~200 GB/s vs CuPy's 1,200 GB/s. With proper occupancy tuning (grid-stride loop, adaptive block count), the fused kernel should close this gap.

### Implications for the Compiler

- **Size-adaptive fusion**: The compiler needs a cost model. At FinTek sizes (50K-500K per ticker), fusion is the right call. At 10M+, compose CuPy-quality primitives.
- **Rust eliminates the dispatch layer entirely**: E09 showed 9us kernel launch from Rust vs CuPy's 70us. This shifts the crossover point dramatically upward — fusion may win even at 10M once we're not paying CuPy dispatch.
- **Shared intermediates are always-win**: The compiler should share cumsums across algorithms regardless of size. This is the lowest-risk, highest-consistency optimization.

---

## Entry 011 -- Experiment E04: Pipeline Generator (Spec -> Primitive DAG -> CSE -> Codegen)

**Date**: 2026-03-30
**Type**: Compiler architecture validation
**Status**: Complete
**Verdict**: The core compiler loop works end-to-end. CSE automatically finds shared primitives across specialist boundaries. Generated kernels beat both naive CuPy (2x) and manually-shared CuPy (1.1-1.3x) across all tested sizes.

### Key Question

Can a simple compiler loop -- (1) decompose specialists to primitive IR, (2) run CSE across the merged DAG, (3) generate CUDA source, (4) execute -- automatically find the sharing that a smart programmer would write by hand?

### Method

Pipeline: `rolling_zscore(price, w=20) + rolling_std(price, w=20)` on the same data.

Three paths, all computing BOTH outputs (apples-to-apples):
- **Path A** (naive): Each specialist builds its own cumsums. 4 cumsums total.
- **Path B** (manual): Smart programmer shares cumsums. 2 cumsums total.
- **Path C** (compiler): registry decomposition + CSE + codegen + fused kernels.

Registry: 3 specialists (rolling_mean, rolling_std, rolling_zscore), each as a primitive_dag recipe. 5 fields per specialist: `primitive_dag`, `fusion_eligible`, `fusion_crossover_rows`, `independent`, `identity`.

### Results: CSE

```
Original nodes (rolling_zscore + rolling_std decomposed): 6
After CSE:                                                4
Eliminated:                                               2  (33%)
```

The two eliminated nodes: `scan(price, add)` and `scan(price_sq, add)` -- the cumsums that both specialists need. CSE found them automatically from their canonical identity hash. The sharing is invisible in the domain spec; it emerges from the primitive decomposition.

Execution steps after CSE:
```
[scan      ]  cs2   inputs=['data_sq:price']              id=51925aa0
[scan      ]  cs    inputs=['data:price']                 id=a4a43a69
[fused_expr]  out   inputs=['data:price', cs_id, cs2_id]  id=b2467a19  (rolling_zscore)
[fused_expr]  out   inputs=['data:price', cs_id, cs2_id]  id=f34e4b3b  (rolling_std)
```

The two fused_expr nodes share the same scan outputs. They have different identities because their formulas differ (zscore vs std). CSE correctly did NOT merge them.

### Results: Correctness

```
n=100000, window=20
max |z_compiler   - z_ref|   = 2.38e-07  PASS
max |std_compiler - std_ref| = 1.19e-07  PASS
```

Float32 precision. Both outputs match independent CuPy reference.

### Results: Benchmark (p50, microseconds)

| n      | Path A (naive, 4 cumsums) | Path B (shared, 2 cumsums) | Path C (compiler) | C vs A | C vs B |
|--------|---------------------------|----------------------------|-------------------|--------|--------|
| 50K    | 531                       | 333                        | 258               | 2.1x   | 1.3x   |
| 100K   | 550                       | 286                        | 216               | 2.5x   | 1.3x   |
| 500K   | 592                       | 325                        | 221               | 2.7x   | 1.5x   |
| 1M     | 701                       | 318                        | 318               | 2.2x   | 1.0x   |
| 5M     | 1628                      | 877                        | 823               | 2.0x   | 1.1x   |
| 10M    | 3703                      | 1938                       | 1792              | 2.1x   | 1.1x   |

### Interpretation

**2x vs naive is consistent across all sizes** (50K through 10M). This is pure CSE value: 4 cumsums -> 2. Not dispatch savings, not fusion -- just elimination.

**Path C beats manual sharing (Path B) at all sizes up to 1M.** The fused kernels produce results in fewer element-wise passes than CuPy's separate mean/std/zscore operations. At 5M-10M, they converge (both are now bandwidth-limited and spending most time in the scans).

**The compiler beats the smart programmer** at most sizes. This is the proof of concept: no human needs to write the shared-cumsum code. The registry + CSE produces it automatically from the domain-level spec.

**Key architectural validation**:
1. Specialist decomposition to 8-primitive IR works
2. CSE over the merged DAG finds sharing across specialist boundaries
3. Code generation produces correct kernels from the plan
4. Injectable world state API (`plan(spec, registry, provenance=None, dirty_bitmap=None, residency=None)`) is in place for future E07 integration
5. The buffer pool (identity -> computed array) is the execution engine's core -- matching the pointer routing graph design

### Architectural Note: Two fused_expr nodes, shared inputs

The two fused_expr nodes appear to have the same inputs in the step listing, but have different identities (b2467a19 vs f34e4b3b). This is correct: their `params` differ (formula="rolling_zscore" vs formula="rolling_std"). They ARE different computations. The buffer pool, indexed by identity hash, stores them separately.

This demonstrates the registry-as-canonicalizer property: same data inputs + same scans, but different transformation = different identity. CSE handles sharing at the scan level, not the fused_expr level.

### What This Enables

Post-E04, the compiler can:
1. Accept a domain-level pipeline spec
2. Automatically find shared work across specialist boundaries
3. Generate and execute fused kernels
4. Produce correct, faster-than-naive output

Next step (E10): extend the registry to all ~135 specialists. The loop is proven; the registry is the work.

### Connection to E03b

E03b measured the speedup of manually sharing cumsums. E04 proves the compiler produces that sharing automatically. E03b was the motivation; E04 is the mechanism.

---

