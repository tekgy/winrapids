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

