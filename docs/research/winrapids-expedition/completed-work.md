# Completed Work — WinRapids Expedition

*Pathmaker updates this file when finishing work. Navigator reads before asking questions.*

---

## Experiment 001: CUDA Proof of Life — COMPLETE

- **Files**: `experiments/001-cuda-proof/cuda_proof.cu`, `build.bat`, `README.md`
- **Build**: nvcc 13.1 + MSVC 19.44, `-arch=sm_120` targeting Blackwell compute 12.0
- **Device**: RTX PRO 6000 Blackwell Max-Q, 188 SMs, 95.6 GB VRAM, 128 MB L2 cache, WDDM mode
- **All 6 tests PASS**: vector add (1K/1M/64M), reduction (1M/64M), managed memory (1M)
- **CUDA 13 API changes discovered**: `cudaDeviceProp.memoryClockRate`/`.clockRate` removed (use `cudaDeviceGetAttribute`); `cudaMemPrefetchAsync` now takes `cudaMemLocation` struct instead of int device
- **WDDM constraints confirmed**: `cudaMemPrefetchAsync` NOT SUPPORTED; `cudaMallocManaged` very slow (66ms for 4MB)
- **PCIe bandwidth asymmetric**: H2D ~15.5 GB/s, D2H ~6.75 GB/s (pageable memory, no pinned)
- **GPU compute bandwidth**: 1677 GB/s effective on 64M vector add — near theoretical peak
- **Key implication**: On Windows/WDDM, explicit memory management (cudaMalloc/cudaMemcpy) is required, not optional. Managed memory works but has severe overhead.

## Experiment 002: GPU Memory Management on Windows — COMPLETE

- **Files**: `experiments/002-gpu-memory/gpu_memory.cu`, `build.bat`, `README.md`
- **cudaMalloc latency**: 37-1284 us (alloc), 174-17504 us (free). **cudaFree at 1GB p99 = 6.4 SECONDS**
- **CUDA Memory Pools WORK**: cudaMallocAsync/FreeAsync = 0.5 us regardless of size. **240x faster than raw**
- **Pinned vs pageable**: Pinned gives 55-57 GB/s symmetric. **Pageable asymmetry from Exp 001 was artifact**
- **VRAM capacity**: 93.5 GB free per cudaMemGetInfo, allocated 88 GB successfully (stalled at 89 GB)
- **Async overlap**: 80% benefit from overlapping compute + copy. Pipelined architecture viable
- **Pool allocator**: Simple pointer-reuse pool = 0.0 us. CUDA async pool = 1.4 us. Both crush raw (281 us)
- **Architecture decision**: Use CUDA memory pools as default. Pinned memory for staging. Never raw cudaMalloc in hot path

## Experiment 003: Arrow GPU Integration on Windows — COMPLETE

- **Files**: `experiments/003-arrow-gpu/arrow_gpu_test.py`, `README.md`
- **Arrow<->numpy is zero-copy** for float64 after first call. Roundtrip bottleneck is PCIe, not Arrow
- **DLPack is free**: GPU-to-GPU zero-copy in 3-35 us for any size. Same-pointer exchange
- **Co-native split demonstrated**: CPU-resident ArrowDeviceArray metadata + GPU buffer pointers. Schema readable without GPU
- **Pinned H2D**: 32-51 GB/s. CuPy asnumpy D2H doesn't use pinned destination (9-19 GB/s), but explicit pinned is 55+ GB/s (Exp 002)
- **Custom CUDA kernels on Arrow data**: Works via CuPy RawKernel, zero error
- **IPC pipeline**: Arrow IPC deserialize (0.2 ms) -> 3 cols to GPU (11.6 ms) for 5M rows
- **Key: pyarrow.cuda not needed**. Arrow -> numpy (zero-copy) -> CuPy (H2D) is the path on Windows
- **DLPack for GPU interchange**, ArrowDeviceArray struct for metadata co-nativity

## Experiment 004: Minimal GPU DataFrame — COMPLETE

- **Files**: `experiments/004-gpu-dataframe/gpu_dataframe.py`, `README.md`
- **GpuColumn + GpuFrame**: ~200 lines Python, CuPy-backed, Arrow-compatible
- **Performance vs pandas (10M rows)**: sum 71x, mean 124x, arithmetic 53x, filtered sum 92x, custom kernel 201x
- **Co-native split working**: CPU metadata (name/dtype/length/location) + GPU buffers. memory_map() shows residency
- **GpuFrame overhead over raw CuPy**: negligible (0.38 ms vs 0.44 ms on filtered sum)
- **Arrow roundtrip**: 4.8 ms for 1M rows (2.2 ms in, 2.6 ms out)
- **Architecture**: GpuColumn = fractal leaf (buffer + metadata), GpuFrame = fractal branch (collection + metadata)
- **Verdict**: 50-200x speedup justifies the project. The complexity is worth it

## Experiment 005: Pandas GPU Proxy Pattern — COMPLETE

- **Files**: `experiments/005-pandas-proxy/pandas_proxy.py`, `README.md`
- **Proxy works for simple ops**: sum 61x, mean 103x faster than pandas. One line of code change
- **Proxy FAILS for chained ops**: filtered sum 0.5x (slower!) due to GPU->CPU->GPU roundtrips
- **Key insight**: proxy needs GPU DataFrame underneath to avoid D2H materialization on intermediates
- **Fallback mechanism works**: unknown ops delegate to pandas, logged for visibility
- **Architecture decision**: thin proxy over pandas = good for simple aggregations only. Full GPU-resident DataFrame (GpuFrame) needed for complex workflows. cudf.pandas approach = correct but requires full re-implementation

## Experiment 006: Co-Native Data Structures — COMPLETE

- **Files**: `experiments/006-co-native/co_native.py`, `README.md`
- **Four-tier system**: Device (GPU), Pinned, Pageable, Storage — each mapped to Arrow device types
- **Memory map is co-native**: human-readable AND machine-parseable, CPU-only access, no GPU roundtrip
- **Query planner works**: estimates transfer cost before execution. Agent can make informed decisions
- **Tiered filtered sum (10M rows)**: all-GPU 0.33ms (87x vs pandas), cold-start 7ms (still 4x vs pandas)
- **Promotion amortizes**: first access pays transfer, subsequent accesses free. Data stays on GPU
- **Architecture**: TieredColumn carries tier + metadata + buffer. TieredFrame has memory_map() + query_plan()
- **Key insight**: explicit tier management > transparent spilling. Costs are visible, decisions are informed

## Experiment 007: Polars GPU on Windows — COMPLETE

- **Files**: `experiments/007-polars-gpu/polars_gpu_test.py`, `README.md`
- **Polars GPU: NOT AVAILABLE** — requires cudf_polars (Linux-only, hardcoded cuDF dependency)
- **Backend NOT pluggable** — no generic interface for custom GPU backends
- **Polars CPU is fast**: sum 0.8 ms, filtered sum 7.2 ms (4-10x faster than pandas)
- **GPU still wins**: sum 8.4x faster, filtered sum 23x faster than Polars CPU
- **Arrow interop near-zero**: Polars <-> Arrow in 0.3 ms each direction
- **Pragmatic path**: Polars for I/O/query planning -> Arrow -> GPU for compute

## Experiment 008: I/O Path Benchmarks — COMPLETE

- **Files**: `experiments/008-io-benchmarks/io_bench.py`, `README.md`
- **Unbuffered+pinned wins**: 8.8 GB/s at 1 GB — 2.7x faster than naive Python read (3.3 GB/s)
- **mmap is SLOWER than Python read** at all sizes — page fault overhead hurts sequential GPU loads
- **Unbuffered+pageable doesn't help** — pageable DMA overhead negates the benefit of bypassing OS cache
- **Why pinned wins**: eliminates both OS buffer cache copy AND pageable-to-pinned staging copy
- **Only matters at scale**: below 10 MB, all methods equivalent. Sweet spot starts at ~50 MB
- **DirectStorage NOT benchmarked** — requires D3D12 intermediary (4 boundary crossings vs 1). llama.cpp confirmed unbuffered ReadFile beats it
- **Pipeline strategy**: pinned staging buffer + unbuffered ReadFile + async H2D overlap

## Experiment 009: Dual-Path GPU DataFrame (CUDA C++ vs CuPy) — COMPLETE

- **Files**: `experiments/009-dual-path-dataframe/cuda_dataframe.cu`, `build.bat`
- **Purpose**: Raw CUDA C++ implementation of Experiment 004 operations to quantify CuPy abstraction cost
- **Kernels**: warp-shuffle reductions (sum/min/max), filtered sum, scalar FMA, vectorized double2 FMA
- **Sum**: CUDA C++ 0.082 ms (976 GB/s) vs GpuFrame 0.099 ms — **1.2x abstraction cost** (CuPy sum is well-optimized)
- **Min/Max**: CUDA C++ 0.062 ms (1299 GB/s) vs GpuFrame 0.123-0.137 ms — **2.0-2.2x abstraction cost**
- **Filtered sum**: CUDA C++ 0.084 ms (1072 GB/s) vs GpuFrame 0.331 ms — **3.9x abstraction cost** (warp-shuffle vs CuPy fancy indexing)
- **FMA (a*b+c)**: CUDA C++ 0.192 ms (1668 GB/s) vs GpuFrame 0.531 ms — **2.8x abstraction cost** (1 fused kernel vs 2 separate launches)
- **Vectorized double2 FMA**: No benefit (0.200 ms vs 0.192 ms scalar) — already bandwidth-bound
- **CuPy RawKernel filtered sum**: 0.172 ms — warp-shuffle still 2x faster than shared-memory tree reduction
- **Key finding**: CuPy abstraction cost is 1.2-3.9x depending on operation. Kernel fusion and warp-shuffle are the two biggest wins from going raw CUDA
- **Architecture decision**: CuPy for prototyping, custom CUDA for production hot paths where operations compose

## Experiment 010: Kernel Fusion Engine (Expression Templates) — COMPLETE

- **Files**: `experiments/010-kernel-fusion/fused_ops.cu`, `cupy_comparison.py`, `build.bat`
- **Purpose**: Compile-time kernel fusion using C++ expression templates — build expression AST at compile time, evaluate per-element in single kernel, zero intermediate buffers
- **a*b+c**: Fused 0.194 ms vs CuPy 0.291 ms — **1.5x faster**, saves 80 MB intermediates
- **a*b+c*c-a/b**: Fused 0.193 ms vs CuPy 0.670 ms — **3.5x faster**, saves 320 MB (5 ops, 1 kernel vs 5 kernels)
- **where(a>0,b*c,-b*c)**: Fused 0.195 ms vs CuPy 0.565 ms — **2.9x faster**, saves 320 MB
- **sum(a*b+c)**: Fused compute+reduce 0.177 ms vs CuPy 0.336 ms — **1.9x faster** (compute AND reduce in same kernel)
- **sqrt(abs(a*b+c*c-a))**: Fused 0.188 ms vs CuPy 0.657 ms — **3.5x faster**, saves 400 MB (6 ops, 1 kernel)
- **All fused kernels ~0.19 ms regardless of expression depth** — bandwidth-bound at ~1650 GB/s. Expression complexity is free
- **Template fusion matches hand-written kernels** — 0.194 ms vs 0.192 ms for FMA (zero abstraction cost)
- **Key finding**: Fusion advantage grows with expression complexity. Simple ops: 1.5x. Complex chains: 3.5x. VRAM savings scale linearly with dataset size
- **Architecture decision**: Expression templates are the from-scratch kernel strategy. Compile-time fusion, zero runtime cost, zero intermediate buffers

## Experiment 010b: Python Kernel Fusion via Codegen — COMPLETE

- **Files**: `experiments/010-kernel-fusion/fused_frame.py`
- **Purpose**: Bridge C++ fusion to Python without a build step. Generate CUDA kernel source from Python expression trees, compile via CuPy RawKernel
- **FusedColumn class**: wraps CuPy array, arithmetic builds lazy expression tree, evaluate() generates + launches fused kernel
- **Python fused vs CuPy eager**: 1.4-3.2x speedup (captures 85-95% of C++ fusion benefit)
- **Python fused vs C++ fused**: only 0.01-0.05 ms slower (Python call overhead, not kernel quality)
- **JIT overhead**: negligible (~0 ms) — nvrtc caches at driver level
- **fused_sum()**: compute+reduce in single kernel, 1.5x over CuPy's 3-kernel approach
- **Key finding**: CuPy RawKernel codegen is the production path for kernel fusion. No C++ build step needed
- **Architecture decision**: Python codegen for production, C++ templates as reference implementation for validation

## Experiment 011: GPU GroupBy — COMPLETE

- **Files**: `experiments/011-gpu-groupby/gpu_groupby.py`
- **Purpose**: Fill the core gap — "df.groupby('category').sum() on GPU on Windows"
- **Two approaches**: sort-based (argsort + segmented cumsum) and hash-based (atomic scatter)
- **100 groups**: sort 3.3 ms (21x), hash 3.1 ms (22x) vs pandas 69.7 ms. Atomic contention limits hash at low cardinality
- **10K groups**: sort 3.4 ms (26x), hash 0.5 ms (176x) vs pandas 87.7 ms
- **1M groups**: sort 3.4 ms (113x), hash 0.5 ms (706x) vs pandas 378.3 ms. Hash dominates at high cardinality
- **Multi-agg (sum+mean+count)**: 3.4 ms vs pandas 98.6 ms (29x). Sort once, derive all aggregations
- **Sort-based stable at ~3.3 ms** regardless of cardinality — argsort dominates
- **Hash-based scales inversely with cardinality** — less atomic contention = faster
- **Key finding**: Smart dispatch by cardinality. Below ~1K groups: either works. Above ~1K: hash wins dramatically
- **Architecture decision**: Dual-dispatch groupby with cardinality estimation

## Experiment 012: Fused GroupBy Expressions — COMPLETE

- **Files**: `experiments/012-fused-groupby/fused_groupby.py`
- **Purpose**: Compose expression fusion (Exp 010) with groupby (Exp 011). Can groupby(key).sum(a*b+c*c-a/b) be one fused kernel?
- **Three approaches tested**: CuPy unfused, fully-fused (atomic), hybrid (fuse expr + sort reduce)
- **Fully fused (atomic) is BAD**: 16.2 ms at 100 groups (4.6x slower than unfused). Binary search + atomic contention
- **Hybrid wins everywhere**: 3.43 ms at 100 groups, 3.53 ms at 10K groups
- **Hybrid vs unfused for complex expr (5 ops)**: 1.14x faster (saves 320 MB intermediates)
- **Key lesson**: Fuse the computation, don't fuse the reduction. Atomics are terrible for grouped reductions
- **Architecture decision**: Query engine = expression fusion codegen layer -> sort-based grouped reducer. Two stages, cleanly composed

## Experiment 013: GPU Joins — COMPLETE

- **Files**: `experiments/013-gpu-joins/gpu_join.py`
- **Purpose**: GPU inner joins — the other fundamental relational operation
- **Three approaches**: direct-index (dense int keys), sort-merge (searchsorted), hash (CPU build + GPU probe)
- **Direct-index (10M x 10K)**: 0.65 ms — **382x faster** than pandas (247 ms). O(1) lookup, no hashing
- **Direct-index (10M x 1M)**: 0.78 ms — **412x faster** than pandas (321 ms). Scales with fact table, not dim
- **Sort-merge (10M x 10K)**: 1.06 ms — **233x faster**. General-purpose, works for arbitrary keys
- **Sort-merge (10M x 1M)**: 1.85 ms — **174x faster**. O(n_fact * log n_dim)
- **Hash join (CPU build)**: 2.93 ms at 10K dim, but 330 ms at 1M dim — Python loop kills it. Needs GPU-side build
- **Join + GroupBy pipeline**: 6.68 ms vs pandas 322 ms — **48x faster** end-to-end
- **Key finding**: Direct-index is fastest for dense integer keys. Sort-merge for general keys. GPU hash table build is a future optimization target
- **Architecture decision**: Join dispatch by key type — direct for dense int, sort-merge for sparse/arbitrary

## Experiment 014: End-to-End GPU Analytics Pipeline — COMPLETE

- **Files**: `experiments/014-end-to-end/e2e_pipeline.py`
- **Scenario**: 10M sales x 10K products. Query: revenue by category (join + expression + groupby)
- **Pipeline**: Parquet -> Arrow -> CuPy (H2D) -> direct-index join -> fused revenue kernel -> sort+cumsum groupby -> result
- **Full pipeline**: 159 ms vs pandas 510 ms — **3.2x faster** end-to-end
- **GPU compute only**: 5.1 ms (join 1.4 ms + compute 0.24 ms + groupby 3.4 ms) — **~90x faster** than pandas compute
- **I/O dominates**: H2D transfer 110 ms = 97% of GPU pipeline time. Pageable memory, not pinned
- **Correctness**: All 50 category sums match pandas within 3.86e-05
- **Key finding**: GPU compute is no longer the bottleneck. I/O optimization (pinned memory, direct Parquet->GPU) is the next frontier
- **Architecture decision**: GPU-resident data model wins. If data stays on GPU, subsequent queries cost 5 ms not 159 ms

## FinTek Task #11: winrapids_fused Backend — COMPLETE

- **Files**: `R:/fintek/trunk/backends/winrapids_fused/` (transfer.py, runner.py, write.py, ops/)
- **WinRapids library**: `R:/winrapids/src/winrapids/` (Column, Frame, evaluate, fused_sum, h2d, d2h)
- **Approach**: New backend alongside cupy_fused — no surgery on existing code. One-field switch (`backend="winrapids-fused"`)
- **Transfer path**: PyArrow → numpy (zero-copy float64) → pinned H2D via `h2d_batch` → cp.ndarray. No cuDF.
- **Validated on real data**: AAPL market data, 598K rows. Exact match vs pandas, **51x speedup**, zero cuDF dependency.
- **Key finding**: "CPU→GPU copy is unacceptable" concern from CRITICAL-WINDOWS-GPU.md does not hold at real FinTek data sizes. 598K rows at pinned speeds = sub-millisecond H2D. The concern was written without numbers; now we have numbers.
- **Architecture**: winrapids.transfer.h2d_batch is the shared primitive — batch transfer all columns from a file in one sync. Both the FinTek backend and the WinRapids library use it.

