# WinRapids Vision

**A sharing optimizer that happens to compile code for the things it can't share.**

**In longer form: a GPU computation compiler for data science.** The compiler's primary job is to find what can be shared across computations — not to optimize individual computations. Computation is the fallback when sharing fails.

- Provenance hit (865x): zero computation — perfect sharing across plans
- CSE (2x): less computation — sharing within a plan
- Fusion (2.3x): fewer launches — sharing kernel overhead

The 865x is not an outlier. It is the primary case. The persistent store is central, not peripheral, because it extends sharing across plans and sessions — otherwise sharing is bounded by within-plan CSE at 2x.

The user describes WHAT. The compiler finds what can be SHARED. The persistent store remembers what has been SHARED. The specialist registry declares where sharing SURFACES are.

---

## The Paradigm Shift

| | RAPIDS (current world) | WinRapids |
|---|---|---|
| **Architecture** | Separate libraries (cuDF, cuML, cuGraph) | One compiler sees everything |
| **Execution** | Interpreted — one op at a time | Compiled — entire pipeline at once |
| **Fusion** | None across libraries | Cross-algorithm primitive sharing |
| **Data lifetime** | Load → compute → free | Persistent — data lives on GPU |
| **Memory** | Per-library allocators (RMM) | Unified store with provenance-aware retention |
| **Optimization** | Per-operation | Whole-pipeline: dead column elimination, predicate pushdown, shared intermediates |
| **Platform** | Linux only | Windows-native (DirectStorage, WDDM pools, Nsight VS) |
| **Build** | C++ / CMake / GCC / conda | Rust / cargo / pip install |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User API (Python, PyO3)                  │
│  wr.read()  wr.GPUStore()  wr.Pipeline()  wr.observe()  │
│  Everything lazy. Everything builds a graph.             │
├─────────────────────────────────────────────────────────┤
│               Pipeline Compiler (Rust)                    │
│  1. Decompose specialists → 8 primitives                 │
│  2. Build primitive dependency graph                     │
│  3. Find sharing across algorithm boundaries             │
│  4. Match against pre-built pipeline library             │
│  5. JIT-generate via NVRTC if no pre-built match        │
│  6. Check persistent store for already-computed results  │
│  7. Prune graph, emit minimal execution plan             │
├─────────────────────────────────────────────────────────┤
│            Persistent GPU Store (Rust)                    │
│  Data lives on GPU until evicted                         │
│  Self-describing buffers (provenance, cost, access)      │
│  Memory-as-scheduler: LRU spill on pressure              │
│  Compiled queries on resident data                       │
│  Continuous observers for streaming                      │
├─────────────────────────────────────────────────────────┤
│              Two Execution Tiers (E05, E09 validated)     │
│  Tier 1 — JIT + disk cache (Rust NVRTC, ~22ms first run, │
│           cached forever; 100 kernels = 2s warmup total) │
│  Tier 2 — Composed specialists (fallback, 2.3x slower)   │
├─────────────────────────────────────────────────────────┤
│        Primitive Library (8 hand-tuned primitives)        │
│  scan │ sort │ reduce │ scatter │ gather │ search │       │
│  compact │ fused_expr                                    │
│  Each with variants: contiguous/segmented/filtered       │
├─────────────────────────────────────────────────────────┤
│        Specialist Registry (~135 compositions)            │
│  DataFrame: 30 │ Features: 20 │ Preprocess: 15           │
│  ML: 20 │ Graph: 10 │ VectorSearch: 5 │ I/O: 10         │
│  Domain: 25 │ Each = recipe of primitives                │
│  Fallback: NVIDIA library (never slower than cuBLAS)     │
├─────────────────────────────────────────────────────────┤
│           Windows-Native I/O + Memory (Rust)              │
│  DirectStorage (NVMe→GPU zero CPU) │ GPU Zstd decompress │
│  Pinned pool │ VRAM headroom │ Unbuffered ReadFile       │
│  Parquet decode on GPU │ CSV parse on GPU                │
├─────────────────────────────────────────────────────────┤
│              cudarc (Rust CUDA bindings)                   │
│  Driver │ NVRTC │ cuBLAS │ cuFFT │ cuSOLVER (fallbacks)  │
├─────────────────────────────────────────────────────────┤
│         CUDA 13 + Blackwell + WDDM + DirectStorage       │
└─────────────────────────────────────────────────────────┘
```

---

## The 9 Primitives

Each primitive is a **sharing granularity** — the minimum unit at which sharing can be detected and reused. The 9 primitives cover all distinct sharing granularities in data science computation. Specialists are recipes that declare which granularities their computation operates at. The CSE pass finds overlapping granularities across specialists and eliminates redundant computation.

**Two halves**: scan/sort/reduce/tiled_reduce/scatter/gather/search/compact have STRUCTURAL identity — sharing determined by input structure (what the data is). fused_expr has EXPRESSION identity — sharing determined by the expression tree (what the formula is). Structural primitives enable deep sharing. fused_expr enables shallow sharing and closes the set.

**Decomposition rule**: Identify which intermediates are structurally shareable (prefix? aggregate? matrix product? index access? filter? order?) → assign to the appropriate primitive. What's left is fused_expr. The primitive_dag is a sharing map of the algorithm, not a flowchart.

**Adding a primitive requires**: a distinct CSE identity structure that cannot be expressed as an existing primitive type without losing block-level sharing. This is the admission test. Every other change is a specialist (a recipe of existing primitives).

| Primitive | What | Variants |
|---|---|---|
| **scan** | Parallel prefix with pluggable associative operator — scan(data, op) where op.combine is associative. AddOp→cumsum, WelfordOp→rolling mean+var, KalmanOp→Kalman filter, SSMOp→Mamba selective scan, EWMOp→EWM | contiguous, segmented, windowed |
| **sort** | Radix sort | u32, i64, f64, argsort-only, partial (top-K) |
| **reduce** | Warp-shuffle aggregation | sum, mean, min, max, std, count, argmin, any/all, multi-output |
| **tiled_reduce** | 2D blocked accumulation — tiled_reduce(A, B, op) where output is 2D and identity is (A_id, B_id, op). GEMM, FlashAttention, PCA covariance, KNN distance matrix. CSE identity is block-level: one node per matrix pair, not N² per element. cuBLASLt is fallback, not primary. | outer_product, dot_product, softmax_weighted (FlashAttention pattern) |
| **scatter** | Indexed write | add, max, one-hot, histogram |
| **gather** | Indexed read | permutation, join assembly, reindex |
| **search** | Binary search | sorted array, bin edges, range |
| **compact** | Stream compaction | boolean mask, predicate, dedup |
| **fused_expr** | Codegen element-wise | arbitrary expression tree → NVRTC |

---

## Specialist Coverage (from real-world RAPIDS usage research)

### P0 — Must Have (~75 specialists, covers 90% of workloads)

**DataFrame Core (~30)**
- read_parquet (per encoding: PLAIN, DELTA, DICT × per dtype)
- filter (fused predicate + compact)
- groupby.agg (sort-based and hash-based × sum/mean/std/min/max/count/first/last)
- merge/join (direct-index, sort-merge, hash × i32/i64)
- sort (radix per dtype, argsort, partial)
- fillna (constant, forward-fill, backward-fill)
- drop_duplicates (sorted, hash)
- value_counts (sort+count, hash+count)
- concat / append
- pivot_table (fused reshape+scatter+reduce)
- write_parquet

**Feature Engineering (~20)**
- rolling(w).sum/mean/std/min/max (prefix-sum based, multi-window in one pass)
- shift(n) / diff(n) (multi-lag in one pass)
- ewm(span).mean (scan-based recursive)
- resample (binary search bin boundaries + fused agg)
- rank / pct_change
- cut / qcut (digitize)
- label encoding, ordinal encoding, target encoding, one-hot
- cumsum / cumprod / cummax

**Preprocessing (~15)**
- StandardScaler (fused mean+std+scale in one pass)
- MinMaxScaler (fused min+max+scale)
- RobustScaler (quantile-based)
- Normalizer (L2 row normalize)
- train_test_split (shuffle+mask)

**I/O (~10)**
- Parquet PLAIN decode per dtype on GPU
- Parquet DELTA_BINARY_PACKED decode on GPU
- Parquet DICTIONARY decode on GPU
- GPU Zstd decompression
- CSV parse on GPU
- DirectStorage NVMe→GPU pipeline

### P1 — High Value (~35 specialists)

**ML Algorithms (~20)**
- KMeans (fused distance+assign+update)
- PCA (fused center→cov→eigen→project for tall-skinny)
- UMAP (fuzzy simplicial set + SGD)
- HDBSCAN (mutual reachability + MST)
- DBSCAN (epsilon-neighborhood)
- t-SNE (Barnes-Hut)
- KNN (fused distance+topK, never full matrix)
- Linear/Ridge/Lasso regression
- Logistic regression
- Random Forest (histogram-based splits)
- Naive Bayes
- Isolation Forest
- SVD tall-skinny

**Domain-Specific: Finance (~10)**
- diff_n multi-lag
- rolling multi-window volatility
- VWAP (fused weighted mean)
- log returns
- pairwise correlation
- online covariance (Welford)
- Bollinger bands (fused mean+std+bands)

**XGBoost/LightGBM** — interop via DLPack, don't replace

### P2 — When Needed (~25 specialists)

**Graph (~10)**
- PageRank, Louvain, BFS, SSSP, Connected Components
- Triangle counting, Jaccard similarity
- Neighbor sampling (for GNN training)

**Vector Search (~5)**
- Brute-force KNN, IVF-Flat, IVF-PQ, CAGRA

**Domain-Specific: Bio/NLP/Geo (~10)**
- log1p normalize, dispersion, Leiden, Wilcoxon
- TF-IDF, cosine similarity
- Haversine, point-in-polygon

---

## Pre-Built Pipeline Library (The Top ~100 Shapes)

Generated from real-world usage patterns. Each is a single fused kernel or minimal kernel chain.

### Kaggle / Feature Engineering Patterns
```
filter → compute → groupby.agg(sum, mean, count)
rolling(w).mean → rolling(w).std → compute(z_score)
groupby → value_counts → merge (frequency features)
shift(1,2,3,5) → diff → compute(log) (lag features)
fillna → compute → groupby.agg (impute + aggregate)
```

### Enterprise ETL Patterns
```
read → filter → fillna → groupby.agg(sum) → sort → write
merge(left, right) → compute → groupby.sum (star schema)
concat → dedup → sort → write (data integration)
read → filter → select → write (projection + predicate)
```

### ML Pipeline Patterns
```
fillna → standardscaler → pca → kmeans
filter → compute → train_test_split → fit
rolling → feature_stack → predict
standardscaler → knn
```

### FinTek Patterns
```
diff(1,2,3,5,10) → all lags one pass
rolling(3,5,10,20,50).std → all windows one pass
sort(ts) → groupby(bin).agg(sum,sumsq,min,max,count,first,last)
rolling_std → z_score → pca (shared mean/std)
```

### Pipeline Generator
For any combination NOT in the pre-built library:
1. Decompose into primitives
2. Apply fusion rules
3. Generate CUDA source
4. Compile via NVRTC from Rust (~22ms; fusion by default — tile-based kernels eliminate the crossover)
5. Cache to disk via BLAKE3-keyed PTX (cross-process, permanent)

---

## Persistent GPU Store

### Design
- Data loads to GPU and stays until evicted
- Self-describing buffers (64-byte header with provenance, cost, access count)
- Execution plan = pointer routing graph: valid pointer → route it; no pointer → compute → store → route
- Computation is the exceptional case (cache miss); pointer handoff is the common case
- LRU eviction when VRAM pressure; explicit tier management (GPU → Pinned → CPU → Disk)
- Zero-translation cache: result === cache entry === consumer input (pointer handoff, not data copy)
- Provenance-based reuse: compiler checks if a computation was already done
- Tiered: GPU → Pinned → CPU → Disk (explicit, not transparent)

### Persistence Viability by Workload

| Workload | Typical Size | Fits in 96GB? | Persistence Value |
|---|---|---|---|
| Kaggle datasets | 1-50 GB | YES | Massive — iterated exploration |
| Enterprise analytics | 10-100 GB | Mostly | High — repeated dashboard queries |
| ML training sets | 5-50 GB | YES | High — multiple training runs |
| Single-cell genomics | 10-50 GB | YES | High — iterative clustering |
| FinTek single ticker | 50-500 MB | YES | Maximum — thousands of leaves |
| FinTek full universe | 200+ GB | NO (partial) | Smart eviction — hot tickers resident, cold tickers on disk |
| LLM training data | TB+ | NO | Not our use case |

### Smart Eviction for FinTek
- Keep hot tickers (currently processing) resident
- Spill cold tickers to pinned memory (fast re-promote)
- Working set for one ticker × all cadences × all leaves ≈ 2-5 GB
- Can hold ~20-40 tickers resident simultaneously
- Process in waves: load batch → compute all leaves → spill → next batch

---

## Crate Structure

```
crates/
├── winrapids-core/          # Column, Frame, buffer headers, schema
├── winrapids-memory/        # Pool allocator, VRAM headroom, LRU eviction, tiered storage
├── winrapids-primitives/    # 8 hand-tuned primitive families (scan, sort, reduce, etc.)
├── winrapids-specialists/   # ~135 specialist recipes (compositions of primitives)
├── winrapids-pipelines/     # Pre-built pipeline library + pipeline generator
├── winrapids-compiler/      # Graph analysis, pattern matching, fusion, execution planning
├── winrapids-store/         # Persistent GPU store, provenance tracking, memory-as-scheduler
├── winrapids-io/            # Parquet/CSV decode on GPU, DirectStorage, pinned transfers
├── winrapids-ml/            # ML algorithm specialists (PCA, KMeans, KNN, etc.)
├── winrapids-graph/         # Graph algorithm specialists (PageRank, BFS, Louvain, etc.)
├── winrapids-search/        # Vector search specialists (CAGRA, IVF, brute-force)
└── winrapids-py/            # PyO3 bindings, lazy API, winrapids.pandas import hook
```

---

## Experiment Queue

### Foundation Experiments (gate the architecture)

| # | Experiment | Tests | Priority |
|---|---|---|---|
| E01 | Multi-output reduce | sum+mean+std+min+max in ONE kernel vs 5 separate (COMPLETE: 1.3-1.6x; L2 hides most) | P0 |
| E02 | Sort-once-use-many | groupby.sum + groupby.mean + rank from one sort (COMPLETE: 1.3-1.7x) | P0 |
| E03 | Cross-algorithm sharing | rolling_std results feed PCA centering (COMPLETE: 1.2-1.5x; 2x at FinTek sizes via E03b) | P0 |
| E04+E05 | Primitive decomposition registry + Pipeline generator | Spec → primitive DAG → CSE → codegen → execute (COMPLETE: CSE 33% reduction, 2x vs naive) | P0 |
| E06 | Pre-built vs JIT vs composed | Same pipeline, three execution tiers, measure difference (COMPLETE: two tiers not three; JIT = pre-built via disk cache) | P0 |

### Persistent Store Experiments

| # | Experiment | Tests | Priority |
|---|---|---|---|
| E07 | Resident query latency | groupby on 10M GPU-resident rows, measure per-query time | P0 |
| E08 | Provenance-based reuse | Q2 detects Q1 already computed rolling_std, skips it | P1 |
| E09 | LRU spill + re-promote | Pipeline exceeds VRAM, spill to pinned, re-promote on access | P1 |
| E10 | Incremental scan extension | Extend prefix sum with new data without full recompute | P1 |
| E11 | Persistent observer jitter | WDDM preemption variance for continuous GPU kernel | P1 |

### Rust Foundation Experiments

| # | Experiment | Tests | Priority |
|---|---|---|---|
| E12 | cudarc kernel launch | Launch a custom .cu kernel from Rust via cudarc on Windows | P0 |
| E13 | PyO3 + cudarc roundtrip | Python → Rust → CUDA → Rust → Python, measure overhead | P0 |
| E14 | Rust Column type | Column struct with buffer header, CuPy interop via DLPack | P1 |
| E15 | NVRTC from Rust | Compile CUDA source at runtime from Rust, cache to disk | P0 |

### I/O Experiments

| # | Experiment | Tests | Priority |
|---|---|---|---|
| E16 | GPU Parquet PLAIN decode | Decode float64 pages directly on GPU, bypass PyArrow | P1 |
| E17 | DirectStorage end-to-end | NVMe → D3D12 → GPU Zstd → CUDA zero-copy → compute | P2 |
| E18 | GPU CSV parse | Character-level parallel CSV → typed columns | P2 |

---

## First Consumer: FinTek

### What FinTek gets from each WinRapids component

| Component | FinTek Benefit |
|---|---|
| Pipeline compiler | 4600×8×20 = 7.4M kernel launches → ~60 fused launches |
| Specialist library | Fused bin stats, multi-lag diff, multi-window rolling — all one kernel each |
| Persistent store | Hot tickers stay GPU-resident, cold tickers spill to pinned |
| Primitive sharing | Rolling stats feed PCA centering without recomputation |
| Pre-built pipelines | K01→K02 bin pipeline, K02→K03 cross-cadence — pre-built, AOT compiled |
| DirectStorage | 3.7TB tick corpus: NVMe→GPU with GPU-side Zstd, CPU free for scheduling |

### Backend integration
- `winrapids_batched` — existing 10 _old leaves (1D ops)
- `winrapids_fused` — upcoming leaves (2D stacked-channel ops)
- Future: `winrapids_compiled` — full pipeline compilation, no per-leaf execution

---

## Competitive Position

### Why NVIDIA can't do this
- Separate libraries = separate teams = can't fuse across boundaries
- General-purpose = can't specialize for data science shapes
- Linux-first = won't invest in Windows-native I/O (DirectStorage)
- Business model prevents specialization that hurts generality

### Why this is acquisition-worthy
- First pip-installable GPU data science toolkit on Windows
- Performance matches or beats RAPIDS through specialization + fusion
- DirectStorage pipeline = Windows-only advantage RAPIDS can't match
- Persistent store = paradigm shift from batch to continuous
- Growing specialist library = compound competitive moat
- FinTek as proof: real financial data, 51x speedup, exact numerical agreement

### The headline
**WinRapids: GPU computation compiler for data science. Describe your pipeline, we compile it to hardware speed. Every query on resident data: milliseconds. Every specialist we add makes every pipeline faster. Windows-native, Rust-built, pip-installable.**
