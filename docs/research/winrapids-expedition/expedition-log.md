# Expedition Log — WinRapids

*The naturalist's record of the journey.*

---

## Day 1 — 2026-03-20

The expedition begins. An empty directory on an NTFS drive, an RTX PRO 6000 Blackwell with 98GB of VRAM running hot, and a question: can we build something that NVIDIA hasn't — GPU-accelerated data science tools that are native to Windows?

### The Landscape We're Walking Into

I spent the morning surveying what exists outside our camp. The picture is striking in its emptiness.

**RAPIDS cuDF** — NVIDIA's own GPU DataFrame library — does not run on Windows. Not "doesn't run well." *Doesn't run.* The official guidance is: use WSL2, install Ubuntu, pretend you're on Linux. This has been the state of affairs since the project's inception. Issue #28 on the cuDF GitHub repo — "Support for windows?" — is one of the oldest open issues. The answer has always been the same: use WSL.

**GPUDirect Storage** — NVIDIA's fast-path for loading data directly from NVMe into GPU memory — is Linux-only. No Windows support.

**DirectStorage 1.4** — Microsoft's answer, just announced at GDC 2026 — adds Zstd GPU decompression and a Game Asset Conditioning Library. It's designed for streaming game assets, not for data science. But the underlying capability — GPU-accelerated decompression of data flowing from NVMe to VRAM — is exactly what a GPU DataFrame needs. Nobody has pointed DirectStorage at a columnar data format. That's an unexplored path.

**The WDDM question** is the elephant nobody talks about. Windows forces GPUs through its display driver model (WDDM), which adds overhead to memory management operations. External reports claim "up to 2x slower" for CUDA workloads on WDDM, but our own experiment 001 showed the reality is more nuanced: **compute kernel execution is barely affected** (~83% of theoretical bandwidth under WDDM), while **memory management is severely degraded** (managed allocation 30-250x slower, prefetch completely unavailable). The overhead is concentrated in the memory path, not the compute path.

Professional GPUs like our RTX PRO 6000 can switch to TCC mode (confirmed — `nvidia-smi -g 0 -dm 1` succeeds), which strips the display driver and gives Linux-like memory management. But then you lose display output. The truth depends on the workload profile — memory-management-heavy workloads (many small allocations, frequent transfers) suffer most under WDDM. Memory-bandwidth-limited workloads (large kernels on pre-allocated data) run at nearly full speed.

### What This Means

Every GPU data science tool today assumes Linux. The entire RAPIDS ecosystem, the CUDA Python toolchain, the GPU memory management libraries — all of them treat Windows as an afterthought or ignore it entirely. The few Windows users doing GPU compute are either routing through WSL2 (adding a translation layer — exactly what co-native design rejects) or accepting degraded performance.

WinRapids isn't building a port. It's building something that doesn't exist: a GPU *DataFrame* toolkit that treats Windows as the *native* environment, not a compatibility target.

*(Nuance from the observer's fact-check: individual GPU compute libraries — CuPy, Numba CUDA — do work on Windows. The empty quadrant is specifically at the DataFrame/data science workflow level, not at the raw GPU compute level. You can launch CUDA kernels on Windows today. You just can't do `df.groupby('category').sum()` on a GPU DataFrame.)*

The gap is real. The hardware is here. The question is whether the software can be made to work.

### The Machine

For the record, what we're working with:

- **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**
- 24,064 CUDA cores, 188 SMs
- 95.6 GB GDDR7 ECC memory (97,887 MiB per nvidia-smi; spec sheet says 96 GB nominal)
- 1,792 GB/s theoretical memory bandwidth
- 125 TFLOPS FP32 single-precision (spec; actual limited by 300W Max-Q power envelope)
- 4,000 TOPS FP4 AI inference (with sparsity)
- 300W TDP (Max-Q variant; non-Max-Q desktop card is 600W)
- PCIe Gen 5
- ~~Multi-Instance GPU (MIG) capable~~ **MIG not supported on this SKU** (nvidia-smi reports N/A, not Disabled)

This is not a gaming GPU pretending to do compute. This is a workstation GPU with ECC memory (hardware present but currently disabled) and strong compute capabilities. However, it runs under WDDM with no TCC escape hatch and no MIG support on this SKU. It's powerful hardware with a fixed driver envelope.

*Note: Both MIG (N/A) and TCC (not available under driver 581.42) are unavailable on this specific SKU. Workload isolation must use CUDA streams or MPS (Multi-Process Service). Memory management must use explicit allocation patterns, not managed memory with prefetch.*

### What I'm Watching

The pathmaker is working on proving CUDA works natively — the most fundamental question. Can we compile and run a CUDA kernel on Windows without WSL? The answer should be yes (CUDA has always supported Windows), but the devil is in the details: which libraries work, what's the real overhead, where are the gaps.

I'm watching for the moment when the first number comes back from a GPU kernel running on bare Windows CUDA. That number — whatever it is — establishes the floor. Everything we build sits on top of it.

### Something Worth Noticing: Arrow's Split

The Arrow C Device Data Interface has a design decision that maps perfectly onto co-native design: **metadata stays in CPU memory, data stays on device.**

The `ArrowDeviceArray` struct keeps all schema information, array structure, and type metadata on the host (CPU-accessible). Only the raw data buffers live on the GPU. A `sync_event` field handles the async boundary — consumers can't read data until the GPU is done writing it.

This is the natural split. An AI agent inspecting a DataFrame reads schema, types, column names, and shapes — all on CPU. The GPU never needs to be interrupted for metadata queries. A human looking at column names in a notebook gets the same information from the same place. The compute happens on the device. The understanding happens on the host.

Nobody designed this as co-native. But it *is* co-native. The architecture accommodates both inhabitants naturally.

Also worth noting: the Polars GPU engine uses RAPIDS cuDF under the hood. Which means Polars GPU acceleration inherits cuDF's Linux-only constraint. There is no Polars GPU on Windows. The task to explore Polars GPU on Windows (#6) will likely confirm this — and the real question becomes: can we build something that Polars *could* use as a Windows GPU backend?

### First Blood: Experiment 001 Results

The pathmaker proved CUDA works. The numbers tell a story.

**The good news**: compute bandwidth is excellent. Vector add at 64M elements hits 1677 GB/s — within 6% of the GPU's theoretical memory bandwidth (1792 GB/s). WDDM does not meaningfully degrade raw kernel performance. The GPU is as fast as it should be when it's actually computing.

**The bad news — and it's important**: WDDM cripples memory management.

- `cudaMemPrefetchAsync` — the API that lets you move managed memory to the GPU before you need it — **does not work on WDDM at all.** Not slow. Not degraded. *Not supported.*
- Managed memory allocation (`cudaMallocManaged`) takes 66ms for 4MB. For comparison, explicit `cudaMalloc` is essentially free.
- Running a kernel on managed memory is ~100x slower than on explicitly allocated memory.

The implication is stark: on Windows/WDDM, you must manage GPU memory explicitly. `cudaMalloc`, `cudaMemcpy`, manual lifetime management. There is no "let the runtime figure it out" path. This is a fundamental design constraint for WinRapids — it means our memory management layer isn't optional, it's load-bearing.

**PCIe bandwidth is surprisingly low**: H2D at 15.5 GB/s, D2H at 6.75 GB/s. PCIe 5.0 x16 should do 64 GB/s. The likely explanation: these transfers used pageable host memory, not pinned (`cudaHostAlloc`). Pinned memory bypasses the OS page fault handler and should be much faster. This is a quick experiment to run — and it matters, because every DataFrame load starts with a host-to-device transfer.

**The VRAM surprise**: Desktop compositing (DWM, Explorer, Edge WebView) consumes ~69 GB of the 96 GB available. Only ~29 GB is free for compute at idle. That's still more than most discrete GPUs have *in total*, but it means we're building data science tools with a 69 GB tax. Monitoring and managing this baseline will matter.

**The Max-Q revelation**: The lab notebook caught that this is a Max-Q variant — 300W TDP, not the 600W of the desktop card. Half the power budget. Yet it still has 188 SMs and hits near-theoretical bandwidth. Power-limited, not compute-limited. Interesting for sustained workloads.

### What I See in These Numbers

There's a structural rhyme here with the FinTek work. In that project, "the address IS the argument" — the naming convention encoded architectural truths. Here, the *numbers* are the argument. They're saying:

1. **The compute path works.** Don't worry about kernel performance.
2. **The memory path is where the fight is.** Every architectural decision about memory management is load-bearing.
3. **Keep data on GPU as long as possible.** The H2D/D2H costs are real. The compute costs are nearly free by comparison.

This echoes the garden entry about structural rhymes: io_uring's insight was "amortize system calls by batching." WinRapids' equivalent is "amortize H2D transfers by keeping data resident." The shape of the optimization is the same — minimize transitions between domains (user/kernel space for io_uring, host/device memory for GPU compute).

The expedition has its first data point. The floor is established. Now we build.

### The Dependency Map: What's Linux-Only

After a day of surveying, here's the map of what exists and what doesn't on Windows:

| Component | Linux | Windows | Implication |
|-----------|-------|---------|-------------|
| CUDA kernel compilation & execution | Yes | **Yes** | The foundation works |
| Explicit memory (cudaMalloc/cudaMemcpy) | Yes | **Yes** | The fast path works |
| Managed memory with prefetch | Yes | **No** | Must use explicit memory |
| cuDF (GPU DataFrame) | Yes | No | Must build our own |
| Polars GPU engine | Yes | No | Built on cuDF, inherits Linux-only |
| RAPIDS Memory Manager (RMM) | Yes | No | GCC-dependent, must build our own pool |
| GPUDirect Storage | Yes | No | Use DirectStorage instead? |
| DirectStorage 1.4 | No | **Yes** | D3D12-only; unbuffered ReadFile is faster and cleaner |
| Arrow C Device Data Interface | Yes | **Yes** | The interop standard works everywhere |
| TCC driver mode | Yes (default) | **No** | Not available on this SKU/driver (581.42) |
| MIG (Multi-Instance GPU) | Varies | **No** | Not supported on this Max-Q SKU (N/A) |

The pattern is clear: the low-level CUDA infrastructure works on Windows. Everything built *on top of it* by the data science ecosystem is Linux-only. The gap isn't at the hardware or driver level — it's at the library level.

This is both the opportunity and the mandate. Nobody has built this layer for Windows. The tools to build it (CUDA, Arrow, DirectStorage) are available. The hardware is more than capable. The ecosystem just needs someone to assemble the pieces.

### Correction: TCC Is Not Available

The evolution of understanding here is worth documenting honestly.

First I wrote "TCC should be available" (speculation). Then the lab notebook Entry 004 showed `nvidia-smi -g 0 -dm 1` appearing to succeed, so I updated to "TCC confirmed available." Now the navigator has checked directly and found TCC is **not available** — the driver model shows WDDM current and pending with no TCC option under driver 581.42.

The `nvidia-smi -g 0 -dm 1` command may have returned success without actually being able to complete the switch. The driver doesn't expose TCC for this SKU.

**This actually sharpens the expedition rather than weakening it.** There is no escape hatch. WDDM is not a choice we can engineer around at the driver level — it's the fixed envelope. The question is cleanly "what can we accomplish within WDDM?" not "should we use WDDM or TCC?"

And the answer from experiment 001 is encouraging: CUDA kernels run at 83% of theoretical memory bandwidth under WDDM. The compute path is fast. What we need to engineer around is the memory management path — allocation latency, transfer overhead, and the missing prefetch API. That's a software problem, not a driver problem.

The tools available within WDDM: CUDA streams, async transfers, event-based synchronization, memory pools (`cudaMallocAsync`), pinned host memory, careful kernel launch patterns. Experiment 002 is testing several of these right now.

### Experiment 002: The Memory Question

The pathmaker is running experiment 002 right now — a comprehensive GPU memory management study. Six tests:

1. **cudaMalloc/cudaFree latency** at sizes from 1KB to 1GB — what's the base cost?
2. **CUDA memory pools** (`cudaMallocAsync`/`cudaFreeAsync`) — do they work under WDDM? How much do they help?
3. **Pinned vs pageable transfers** — the question I raised about the PCIe bandwidth gap. This will tell us whether the 15 GB/s H2D and 7 GB/s D2H are limited by pageable memory or by WDDM itself.
4. **VRAM capacity probe** — how much memory can we actually allocate under WDDM? This tests whether the ~29 GB "free" is the real limit.
5. **Async transfer/compute overlap** — can we load data while computing? This is crucial for pipeline design.
6. **Simple pool allocator prototype** — the seed of what could become WinRapids' memory layer. Exact-match block reuse, ~20 lines of code.

The experiment design is methodologically sound: warmup passes, multiple trials, statistical reporting (min/median/mean/p99/max). This addresses the cross-run variance we saw in experiment 001.

I'm watching Test 3 most closely. If pinned memory gets us from 15 GB/s to 40+ GB/s on H2D, the data loading story changes dramatically. If it doesn't, WDDM's overhead is deeper than just pageable memory, and we need to think harder about keeping data resident on GPU.

### Experiment 002 Results: Everything Changes

The results are in, and they rewrite the WDDM story.

**The headlines:**

| Finding | Old understanding | New reality |
|---------|-------------------|-------------|
| VRAM available | ~29 GB (nvidia-smi) | **93 GB** (CUDA reclaims from WDDM) |
| H2D bandwidth | 15.5 GB/s (pageable) | **57 GB/s** (pinned) |
| D2H bandwidth | 6.75 GB/s (pageable) | **55 GB/s** (pinned) |
| Transfer asymmetry | H2D 2x faster than D2H | **Symmetric** with pinned memory |
| Allocation cost | "slow" | **0.5 us** with CUDA pools (240x faster than raw) |
| cudaFree cost | unknown | **Up to 6.4 seconds** at p99 for 1 GB (!) |
| Compute-copy overlap | untested | **80% benefit** — pipelining works |

**The VRAM surprise**: nvidia-smi showed ~69 GB consumed by desktop compositing, leaving ~29 GB "free." But `cudaMemGetInfo` reports 93.5 GB available. CUDA can reclaim VRAM from WDDM's display allocations. The experiment allocated 88 GB in verified chunks. The "29 GB envelope" I wrote about was wrong — it's a 93 GB envelope. That changes the DataFrame capacity story entirely: we can hold datasets an order of magnitude larger than I assumed.

**The transfer surprise**: Pinned memory gets 55-57 GB/s in both directions — close to PCIe 5.0 theoretical. The asymmetry from experiment 001 (15 GB/s H2D vs 7 GB/s D2H) was entirely a pageable-memory artifact. With pinned memory, transfers are fast and symmetric. A 10 GB dataset loads in ~175ms. That's interactive-speed data loading.

**The allocation bombshell**: `cudaFree` on a 1 GB buffer can take 6.4 seconds at the 99th percentile. Not milliseconds — *seconds*. This is WDDM's memory manager reclaiming virtual address space. For a DataFrame library that creates and destroys columns, this is devastating — unless you use memory pools. CUDA memory pools (`cudaMallocAsync`/`cudaFreeAsync`) reduce alloc+free to 0.5 microseconds regardless of size. That's 30,000x faster for large allocations. Memory pools are not an optimization on WDDM — they're a survival requirement.

**The overlap result**: Overlapping compute and transfer gives 80% speedup even with a single async copy engine. This validates the pipelined architecture: load batch N+1 while processing batch N. The Sirius paper's approach works under WDDM.

### Revised WDDM Assessment

The WDDM picture is dramatically better than experiment 001 suggested:

| Capability | Experiment 001 assessment | Revised (with pools + pinned) |
|------------|--------------------------|-------------------------------|
| Compute bandwidth | 1,677 GB/s (good) | Same — already near peak |
| Transfer bandwidth | 15 GB/s (concerning) | **57 GB/s** (excellent) |
| VRAM capacity | ~29 GB (limiting) | **93 GB** (more than enough) |
| Allocation latency | Unknown | **0.5 us** with pools (negligible) |
| Pipelining | Untested | **80% overlap** (works) |
| Managed memory | Not viable | Confirmed: not viable (use pools instead) |

The lesson: WDDM's overhead is real but *engineerable*. The default paths (pageable memory, raw cudaMalloc) are slow. But the optimized paths (pinned memory, pool allocators, async streams) perform well. The architecture tax is upfront complexity — you must use the right APIs — not inherent degradation.

WinRapids' memory management strategy is now clear:
1. **CUDA memory pools** as the default allocator (never raw cudaMalloc in hot path)
2. **Pinned memory pool** for host-side staging (never pageable memory for transfers)
3. **Pre-allocate and reuse** for the hottest paths (simple pool: 0 us)
4. **Pipeline everything** — overlap compute and transfer via streams

---



### A Possible Windows Advantage

An idea that crystallized in conversation with the navigator: DirectStorage 1.4 with GPU-accelerated Zstd decompression might give Windows a genuine *advantage* over Linux for data loading — not just parity.

Here's why: On Linux, GPUDirect Storage does raw DMA from NVMe to GPU memory. Fast, but uncompressed — the data arrives as-is. To load compressed Parquet files, you'd read the compressed data, transfer it to GPU, then decompress on GPU (or decompress on CPU first, then transfer).

DirectStorage 1.4 integrates GPU decompression into the storage pipeline itself. Compressed data flows from NVMe through the GPU decompression pipeline into VRAM in a single operation. For Zstd-compressed Parquet files, this means:

1. The NVMe-to-GPU bandwidth carries *compressed* data (effectively multiplying I/O throughput by the compression ratio)
2. Decompression happens on GPU during the transfer (overlapped, not sequential)
3. The result lands in VRAM ready for processing

If a Parquet column group compresses 3:1 with Zstd, you're effectively moving data at 3x your PCIe bandwidth. On our hardware, that could mean ~45 GB/s effective throughput even with the WDDM-limited ~15 GB/s raw transfer rate.

This is speculative — and further research revealed significant friction:

**Reality check**: DirectStorage 1.4 is D3D12-specific. It doesn't work with CUDA directly. The GPU decompression runs as D3D12 compute shaders, not CUDA kernels. To use it in a CUDA pipeline, you'd need D3D12/CUDA interop — sharing resources between the two APIs. That's doable (CUDA has `cudaImportExternalMemory` for D3D12 resources), but it adds a layer of complexity.

Additionally, the GACL (Game Asset Conditioning Library) is optimized for texture formats (BC1-BC7), not arbitrary data. The Zstd decompression itself is general-purpose, but the pipeline around it assumes game assets.

The idea isn't dead — but it's harder than "just point DirectStorage at a Parquet file." A more realistic path might be: use DirectStorage for raw NVMe-to-GPU bulk transfers (bypassing the Windows I/O stack), then decompress on the CUDA side. Or use the D3D12 interop path for decompression and hand off decompressed buffers to CUDA. Both are research experiments, not quick wins.

### Revised I/O Story: The Scout's Finding

The scout's terrain report on DirectStorage added a crucial piece: **DirectStorage and GPUDirect Storage are structurally different.** They're not equivalents on different platforms.

- **GPUDirect Storage (Linux)**: clean `cuFile` API. NVMe -> GPU. All CUDA. No graphics stack.
- **DirectStorage (Windows)**: D3D12 API. Destination is always a D3D12 resource. To use it with CUDA, you need `cudaImportExternalMemory` — create a D3D12 shared handle, import into CUDA. The graphics stack is inescapable.

The llama.cpp team tried both paths and arrived at a clear verdict: **unbuffered ReadFile with pinned host memory beats DirectStorage and is architecturally cleaner.** `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` + `cudaHostAlloc` gives ~4x over mmap, no graphics dependency, pure Windows + CUDA.

This changes WinRapids' I/O story. The "Windows-native" fast path for data loading isn't DirectStorage at all. It's:

1. **Open file** with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`
2. **Read into pinned host memory** (`cudaHostAlloc`)
3. **Async H2D transfer** to GPU (`cudaMemcpyAsync` with pinned staging)

Unglamorous. Effective. No D3D12 dependency. No graphics stack. And at 57 GB/s pinned transfer speed (from experiment 002), a 10 GB dataset goes NVMe -> pinned host -> GPU in under 200ms.

DirectStorage remains interesting for future research — particularly if NVIDIA or Microsoft ever build a CUDA-native storage API for Windows. But for now, the practical answer is clear. Task #2 should benchmark this unbuffered ReadFile path specifically.

### Late Addition: The Scout's DirectStorage Correction

After the I/O story above was written, the scout returned with a finding that changes the DirectStorage picture. The Zstd decompression shader in DirectStorage 1.4 is **not** texture-specific — it's a **generic** Zstd decompressor implemented in HLSL. Microsoft open-sourced the shader code at `github.com/microsoft/DirectStorage/tree/development/zstd`. This means DirectStorage can decompress any Zstd-compressed data on the GPU, not just game textures.

This reopens a concrete pipeline for Parquet:

```
NVMe -> DirectStorage (D3D12 resource) -> GPU Zstd decompression (HLSL shader)
  -> D3D12 shared handle -> cudaImportExternalMemory -> CUDA buffer
```

The scout is right to call this "the most novel path in the entire landscape." It's also the most complex — every arrow in that pipeline is a different API with different guarantees. But it's genuinely unexplored territory. Nobody has built NVMe-to-GPU Parquet decompression on Windows.

### Late Addition: The Parquet Scan Bottleneck

The scout also surfaced an arxiv paper ("Do GPUs Really Need New Tabular File Formats?", 2602.17335) with a finding that reframes the entire I/O question: **85% of TPC-H query runtime on GPUs is spent in Parquet scanning, not query execution.**

This is significant. If true, the compute layer we've been excited about (1,677 GB/s bandwidth, 50-200x over pandas) only touches 15% of real workload time. The other 85% is I/O and format parsing. The bottleneck isn't computing faster — it's *loading* faster.

This validates two things at once:
1. The unbuffered ReadFile + pinned memory path matters more than kernel optimization for real workloads
2. The DirectStorage+Zstd pipeline, if achievable, would attack the dominant cost

Two I/O paths emerge. The **practical path**: unbuffered ReadFile + pinned staging + async H2D — proven by llama.cpp, buildable today. The **research path**: DirectStorage + GPU Zstd decompression + CUDA interop — novel, complex, potentially transformative for compressed formats like Parquet. WinRapids should build the practical path first and investigate the research path as a differentiator.

### The FinTek Convergence

The most striking observation of Day 1 didn't come from external research. It came from the workspace next door.

FinTek — the financial computation engine in `R:/fintek` — independently hit the same cuDF-on-Windows wall today. They wrote `CRITICAL-WINDOWS-GPU.md`, a document that reads like a parallel expedition report:

- RAPIDS is Linux-only. No Windows wheels. No roadmap.
- cuDF is used in 3 production files for GPU-native Parquet I/O
- WSL2 is unacceptable (NTFS penalties, slow feasibility tests)
- Replacement strategy: PyArrow -> CUDA IPC -> CuPy arrays

Their priorities map directly onto our task list:
- Their "GPU parquet reader" = our Task #5 (minimal GPU DataFrame prototype)
- Their "PyArrow CUDA IPC" = our Task #4 (Arrow GPU integration)
- Their "CuPy memory pool to replace RMM" = our Task #3 (GPU memory management)

This is convergent evolution happening in real time. Two independent projects, same hardware, same day, arriving at the same architectural conclusions. When independent efforts converge, the architecture is probably right.

But it also changes the stakes. WinRapids isn't just an experiment anymore — it has a real customer with production workloads running "trillions of computations." FinTek needs GPU Parquet I/O on Windows. That's not a someday goal. That's a today need.

What FinTek knows that WinRapids should use:
- **CuPy works on Windows** with official wheels (`cupy-cuda13x`). This is confirmed, not speculative.
- **PyArrow works on Windows**. Parquet I/O is available.
- **`cupy.from_dlpack()`** enables zero-copy GPU tensor sharing between libraries.
- **CuPy has its own memory pool** (`cupy.cuda.MemoryPool`) that could serve as a WDDM-compatible alternative to RMM.

What WinRapids could provide back to FinTek: the low-level toolkit they need. The GPU memory pool. The Arrow-to-CUDA bridge. The DataFrame layer. We build it once, they consume it in production.

This is the strongest validation signal I can imagine for the expedition's direction.

### Experiment 003: Arrow GPU Integration — The Bridge Works

While the machine was being attended to, I read the results of experiment 003. The Arrow-to-GPU path on Windows works, and it's simpler than I expected.

**DLPack is the answer for GPU interchange.** Sub-microsecond, zero-copy, works across CuPy/PyTorch/JAX. 100M elements (400 MB) "transferred" in 17 microseconds — because no data moves. It's a pointer exchange. This confirms `cupy.from_dlpack()` as the right bridge, exactly what FinTek's `CRITICAL-WINDOWS-GPU.md` proposed.

**The co-native split is experimentally confirmed.** The experiment built an ArrowDeviceArray concept demo: CPU-resident metadata (schema, type, length, device_type, device_id) alongside GPU-resident buffer pointers. An AI agent can read the schema without touching the GPU. A kernel can process the buffers without parsing metadata. Neither translates. Both read natively. This is exactly what I described in the garden entry about Arrow's split — now verified in code.

**No pyarrow.cuda needed.** The path is: Arrow -> numpy (zero-copy for float64) -> CuPy (H2D with pinned staging). Arrow IPC deserialization is sub-millisecond. The bottleneck is PCIe transfer, not Arrow parsing.

**CuPy's high-level operations have overhead.** Fancy indexing for a filtered sum took 1.7 seconds. This means WinRapids needs custom CUDA kernels for performance-critical operations like groupby, sort, and filtered aggregation. CuPy is the memory manager, not the compute engine.

The concrete architecture that emerges:

- Ingest: Parquet -> Arrow IPC -> numpy (zero-copy) -> pinned staging -> H2D async
- On GPU: ArrowDeviceArray metadata (CPU) + device buffer pointers (GPU)
- Between: DLPack for zero-copy GPU interchange
- Export: D2H to pinned staging -> numpy (zero-copy) -> Arrow -> Parquet

This is the bridge FinTek needs. It works on Windows today.

### Experiment 004: The DataFrame Takes Shape

The pathmaker built the thing. `GpuColumn` and `GpuFrame` — about 200 lines of Python, CuPy-backed, Arrow-compatible. And the numbers are the answer to whether this project is worth it.

| Operation (10M rows) | pandas (ms) | GPU (ms) | Speedup |
|----------------------|-------------|----------|---------|
| sum | 8.3 | 0.12 | 71x |
| mean | 11.8 | 0.10 | 124x |
| a * b + c (arithmetic) | 29.7 | 0.56 | 53x |
| filtered sum | 34.8 | 0.38 | 92x |
| custom CUDA kernel | 34.8 | 0.17 | 201x |

50-200x faster than pandas. In 200 lines of Python. On Windows, under WDDM.

But the numbers aren't even the most interesting part. The **architecture** is. GpuColumn is a fractal leaf — a named buffer with metadata. GpuFrame is a fractal branch — a named collection of columns. The co-native split is built in: CPU metadata (name, dtype, length, location) readable by humans and AI agents; GPU buffers touched only by kernels. `memory_map()` shows both:

```
GpuFrame: 5,000,000 rows x 4 columns
  id                    int64           40.0 MB  [gpu]
  price                 float64         40.0 MB  [gpu]
  volume                int32           20.0 MB  [gpu]
  flag                  int8             5.0 MB  [gpu]
  Total: 105.0 MB on GPU
```

And the overhead? GpuFrame filtered sum: 0.38 ms. Raw CuPy: 0.44 ms. The abstraction is *free*. The DataFrame wrapper costs nothing over raw array operations.

Arrow roundtrip: 4.8 ms for 1M rows (2.2 ms in, 2.6 ms out). The bridge works both ways.

What it doesn't have yet: null bitmaps, strings, GroupBy, joins, sort, file readers. Those are the difference between a proof-of-concept and a library. But the foundation is verified.

### Experiment 005: The Proxy Lesson

Can you take existing pandas code, wrap it in `gpu_accelerate(df)`, and get GPU speedups for free? Experiment 005 answered: **sometimes**.

| Operation (10M rows) | pandas (ms) | Proxy (ms) | Speedup |
|----------------------|-------------|-----------|---------|
| sum | 7.9 | 0.13 | 61x |
| mean | 12.0 | 0.12 | 103x |
| filtered sum | 30.2 | 59.2 | **0.5x (SLOWER)** |

Simple aggregations: yes, 60-100x faster. One line of code change. The sweet spot.

Chained operations: *slower than pandas*. The proxy's filtered sum does GPU comparison -> D2H materialize mask -> CPU filter -> H2D transfer result -> GPU sum. Three PCIe round-trips kill the speedup.

This is the cudf.pandas lesson, learned empirically. A thin GPU proxy over a CPU DataFrame works only for isolated operations. For complex workflows — the ones that actually matter in production — you need the data to *stay on the GPU*. Which means you need a real GPU DataFrame underneath, not a wrapper around pandas.

The architecture decision: the proxy pattern is valid as a compatibility shim for simple aggregations. For real workflows, GpuFrame (experiment 004) is the foundation. The proxy's fallback mechanism — unknown ops delegate to pandas, logged for visibility — is genuinely useful and should carry forward.

### Experiment 006: The Tiered Architecture

Experiment 004 gave us a GPU DataFrame. Experiment 006 gave it *geography* — a tiered memory model where columns know where they live and what it costs to move them.

`TieredFrame` extends `GpuFrame` with four memory tiers: Device (GPU), Pinned, Pageable (CPU), and Storage. Each column carries its tier, and the frame exposes a `memory_map()` that shows everything:

```
TieredFrame: 10,000,000 rows x 7 columns

  Column               Type           Size Tier   GPU Cost   Accesses
  -------------------- ---------- -------- ---- ---------- ----------
  timestamp            int64         80.0M  CPU     3.2 ms          0
  open                 float64       80.0M  CPU     3.2 ms          0
  high                 float64       80.0M  GPU   0 (here)          0
  low                  float64       80.0M  GPU   0 (here)          0
  close                float64       80.0M  GPU   0 (here)          0
  volume               int64         80.0M  PIN     1.4 ms          0
  symbol               int32         40.0M  CPU     1.6 ms          0
```

This is the co-native split made operational. An AI agent reads the memory map and knows: "high, low, close are already on GPU — querying them is free. open and timestamp need 3.2 ms each to promote. volume is pinned, only 1.4 ms." No GPU access needed to make this assessment.

The query planner makes costs explicit *before* execution:

```python
plan = frame.query_plan(["open", "high", "low", "close", "volume"])
# -> {'needs_promotion': ['open', 'volume'], 'total_transfer_ms': 4.6}
```

The filtered sum benchmark across tiers tells the residency story:

| Scenario | Transfer | Compute | Total | vs pandas |
|----------|----------|---------|-------|-----------|
| All on GPU | 0.0 ms | 0.33 ms | 0.33 ms | 87x |
| Mixed tiers (first access) | 1.1 ms | 0.5 ms | 1.6 ms | 18x |
| All on CPU (first access) | 6.5 ms | 0.5 ms | 7.0 ms | 4x |
| Re-run (now on GPU) | 0.0 ms | 0.33 ms | 0.33 ms | 87x |

Even the worst case — all data cold on CPU — is 4x faster than pandas. After the first access promotes data to GPU, subsequent operations run at 87x. Promotion is amortized.

This is what "the address IS the argument" means for GPU computing. The tier tag on each column isn't just metadata — it's a cost model. It tells you what an operation will cost before you run it. It makes the invisible visible: no transparent spilling, no surprise OOM, no hidden domain crossings.

And the tier enum maps to Arrow device types: `ARROW_DEVICE_CUDA (2)` for Device, `ARROW_DEVICE_CUDA_HOST (3)` for Pinned. Storage is WinRapids' extension beyond Arrow — Arrow doesn't model persistence. The fractal rhyme: Arrow's co-native split at the buffer level, WinRapids' tiered model at the column level, the Sirius paper's 50/50 memory split at the system level.

### Experiment 007: The Empty Quadrant, Confirmed from Polars

Polars GPU engine: `engine="gpu"` requires `cudf_polars`, which requires RAPIDS cuDF. Linux-only. The backend is not pluggable — it's a hardcoded cuDF dependency, not an interface a third party can implement.

But the interesting finding is the *gap*. Polars CPU is already fast — 4-10x faster than pandas (sum: 0.8 ms vs 8.3 ms). The GPU advantage over Polars is smaller: 8-23x, compared to 50-200x over pandas. Polars has already captured a lot of the low-hanging fruit through Rust-native vectorization and query optimization.

This reframes WinRapids' competitive position. The target isn't just "faster than pandas" — that's easy. The target is "faster than Polars CPU," and the margin is narrower. The GPU's advantage concentrates in:
- Compute-heavy operations (filtered sums, multi-column arithmetic): 8-23x over Polars
- Operations that Polars can't optimize away (custom kernels, complex aggregations)
- Workloads large enough that GPU memory bandwidth dominates (10M+ rows)

The interop finding is pragmatic: Polars <-> Arrow is near-zero-cost (0.3 ms each way). A real workflow could use Polars for I/O and query planning, then Arrow -> GPU for heavy compute. WinRapids doesn't need to replace Polars — it needs to complement it.

Long-term: if Polars ever exposes a backend plugin interface, WinRapids could provide the Windows GPU backend via Arrow Device Data Interface. But that requires upstream changes to Polars, not just WinRapids work.

### Experiment 008: The I/O Path, Measured

After a day of discussion about I/O paths — DirectStorage vs unbuffered ReadFile vs mmap — experiment 008 settled it with numbers.

| Method | 1 GB file -> GPU | Throughput |
|--------|-----------------|------------|
| Python read + cp.asarray | 315 ms | 3.3 GB/s |
| mmap + cp.asarray | 402 ms | 2.6 GB/s |
| Unbuffered ReadFile + pageable | 381 ms | 2.8 GB/s |
| **Unbuffered ReadFile + pinned** | **119 ms** | **8.8 GB/s** |

Three surprises:

1. **mmap is slower than naive Python read.** Page fault overhead for sequential reads to GPU makes mmap counterproductive. This contradicts the common assumption that mmap is always beneficial for large files.

2. **Unbuffered+pageable doesn't help.** Bypassing the OS buffer cache saves one copy, but pageable DMA for the GPU transfer adds it back. You need *both* optimizations — unbuffered *and* pinned — to win.

3. **8.8 GB/s at 1 GB exceeds expected NVMe bandwidth (~7 GB/s).** Likely some filesystem caching surviving `FILE_FLAG_NO_BUFFERING` since the test file was recently created. True cold-read would be ~7 GB/s, the NVMe hardware limit.

The I/O architecture is now concrete:
- Allocate a pinned staging buffer once (256 MB)
- Read file chunks with unbuffered ReadFile into the staging buffer
- Async H2D transfer overlapping with the next file read
- At 8.8 GB/s, a 10 GB dataset reaches GPU in ~1.1 seconds

Below 10 MB, all methods are equivalent — optimization doesn't matter. The sweet spot starts at ~50 MB. This means WinRapids' I/O optimization is only worth deploying for real workloads, not toy datasets. Which is exactly right.

---

## Day 1 — Closing

The expedition asked one question this morning: can we build GPU-accelerated data science tools native to Windows?

The answer, after eight experiments and a day of surveying, is **yes — with specific engineering requirements.**

### What We Know

**The compute path is excellent.** CUDA kernels run at 83% of theoretical memory bandwidth under WDDM. The GPU doesn't care about the driver model once a kernel is dispatched. 1,677 GB/s on a vector add. The Blackwell hardware is as fast as it should be.

**The memory path requires discipline.** WDDM imposes real costs on memory management — but every cost has a known mitigation:

| WDDM cost | Mitigation | Result |
|-----------|------------|--------|
| cudaFree can take 6.4 seconds | CUDA memory pools | 0.5 us alloc+free |
| Pageable transfers: 15-25 GB/s | Pinned host memory | 55-57 GB/s symmetric |
| Managed memory prefetch: broken | Explicit cudaMalloc + cudaMemcpy | Full control, predictable |
| nvidia-smi shows 29 GB free | CUDA reclaims from WDDM | 93 GB actually available |
| No compute-copy overlap? | CUDA async streams | 80% overlap benefit |

**The data bridge works.** Arrow -> numpy (zero-copy) -> CuPy (pinned H2D) -> GPU. DLPack for zero-copy interchange between GPU libraries. ArrowDeviceArray for co-native metadata. All verified on Windows.

**The I/O path is clear and measured.** Unbuffered ReadFile + pinned host memory + async H2D: 8.8 GB/s end-to-end, 2.7x faster than naive Python read. mmap is *slower* than Python read for GPU workloads. No DirectStorage needed. No D3D12 dependency. Pure Windows + CUDA.

**The DataFrame works.** GpuFrame: 200 lines of Python, 50-200x faster than pandas, zero abstraction overhead over raw CuPy. The co-native split — CPU metadata, GPU buffers, explicit location — is verified in code, not just in theory.

**The proxy pattern has limits.** Transparent GPU acceleration via pandas wrapping works for simple aggregations (60-100x) but fails for chained operations (0.5x — *slower* than pandas). Real workflows need GPU-resident data. The proxy lesson: you can't avoid building a real GPU DataFrame.

**The tiered architecture works.** Columns carry explicit tier metadata (Device/Pinned/CPU/Storage). A query planner shows transfer costs before execution. Even cold-start (all data on CPU) is 4x faster than pandas. After promotion, 87x. The co-native memory map is readable by both humans and AI agents without GPU access.

**There is a real customer.** FinTek needs GPU Parquet I/O on Windows today, for production workloads at trillions-of-computations scale. WinRapids isn't just an experiment — it's infrastructure.

**I/O may dominate.** An arxiv paper (2602.17335) reports 85% of GPU analytics query time is Parquet scanning. If true, kernel optimization is 15% of the story. The I/O path — unbuffered ReadFile + pinned staging — matters more than we initially assumed.

**Polars confirms the quadrant.** Polars GPU engine requires cuDF (Linux-only), backend is not pluggable. But Polars CPU is already 4-10x faster than pandas, so the GPU advantage over Polars is 8-23x, not 50-200x. Arrow interop is near-zero-cost — WinRapids can complement Polars rather than replace it.

**CuPy JIT cold-start is real.** The observer independently measured a 157 ms cold-start on the first CuPy kernel execution (vs 0.33 ms warm). CuPy compiles CUDA kernels on first use, and the compilation cost is significant. A production system needs kernel pre-warming or AOT compilation.

### What We Don't Know Yet

- What's the Parquet-specific end-to-end throughput? (Raw file I/O is 8.8 GB/s; Parquet adds decompression + decode overhead)
- Can custom CUDA kernels match cuDF's performance for groupby, sort, join?
- What does the Python API surface look like? pandas-compatible? Polars-compatible? Something new?
- How do null/validity bitmaps, string columns, and GroupBy change the architecture?
- Is the DirectStorage + GPU Zstd decompression + CUDA interop pipeline feasible and worth the complexity?
- How does the tiered architecture behave under memory pressure (WDDM paging at 88+ GB)?

### The Architecture That Emerged

Not designed top-down. Emerged from eight experiments, a landscape survey, a FinTek convergence, and eight garden entries:

1. **Memory layer**: CUDA memory pools (`cudaMallocAsync`) as default. Simple exact-match pool for hot paths (0 us reuse). Pinned host pool for staging. Never raw cudaMalloc in production code.

2. **Data layer**: Arrow columnar format in device memory. ArrowDeviceArray metadata on host (co-native: readable by humans and AI agents without GPU access). DLPack for zero-copy GPU interchange.

   **Why this interface is positioned correctly** (navigator's observation): The Arrow C Device Data Interface specifies `sync_event` as an opaque handle — the producer fills it, the consumer waits on it. On CUDA this is a `cudaEvent_t`. But the spec is device-agnostic: same interface for ROCm, Vulkan, Metal. If WinRapids columns are Arrow Device arrays, then a future where Windows gets ROCm support or Intel Arc compute matures requires no public interface change — just a different backend producing the same Arrow Device array. The dual-path design (backend-as-contract) isn't just about CuPy-vs-custom; it's about CUDA-vs-everything-else at the same abstraction level. The interface we've converged on doesn't just solve today's problem. It's at the right level of abstraction for problems we can't predict yet.

3. **I/O layer**: Unbuffered ReadFile -> pinned host staging -> async H2D. Parquet deserialization via PyArrow (sub-millisecond). Pipeline: load batch N+1 while computing on batch N.

4. **Compute layer**: CuPy for memory management. Custom CUDA kernels (via CuPy RawKernel) for performance-critical operations. CuPy's high-level ops are too slow for DataFrame workloads.

5. **Co-native layer**: Column metadata carries location (device/host/storage), type, shape, provenance. Explicit — not hidden behind transparent spilling. Both human and AI agents read the same metadata from the same place.

### The Dual-Path Directive

After the reboot, a design decision that reframes everything: **every experiment should explore both paths.**

- **Pragmatic path**: Build on existing libraries (CuPy, Arrow, Polars). Validates the architecture quickly. Gets us running.
- **From-scratch path**: What would we build from zero? Custom CUDA kernels in C++/Rust, custom columnar format, bare metal optimization. Finds the performance ceiling.

Benchmarks decide which wins. This maps naturally to the backend-as-contract principle: `backend="cupy-generic"` vs `backend="cuda-custom"` vs `backend="rust-fused"`. The same DataFrame API surface, different engines underneath.

Looking back at Day 1 through this lens: all eight experiments used the pragmatic path (CuPy + Arrow + Python). We know the pragmatic floor: 50-200x over pandas, 8-23x over Polars. We don't know the from-scratch ceiling. The gap between them is the abstraction cost — and the observer's JIT cold-start finding (157 ms for CuPy kernel compilation) hints that the gap may be significant.

The concrete questions for Phase 2:
- GpuFrame's filtered sum takes 0.38 ms in CuPy. How fast in raw CUDA C++?
- The custom CUDA kernel already hit 201x (0.17 ms). That was still launched through CuPy's RawKernel. What's the bare-metal number?
- CuPy's memory pool vs CUDA's `cudaMallocAsync` directly — same thing? Or does CuPy add overhead?

Also a safety constraint: **60 GB VRAM ceiling.** The 88 GB allocation that corrupted WDDM state and forced the reboot establishes the practical limit. `cudaMemGetInfo` says 93 GB free, but WDDM becomes unstable above ~60 GB of active allocation. All experiments must respect this.

### The Frame

This morning I called the project "the empty quadrant." There is no GPU DataFrame library native to Windows. By tonight, we have more than a floor — we have a working prototype. GpuFrame runs 50-200x faster than pandas in 200 lines of Python. The memory layer is solved (pools bypass WDDM overhead). The data bridge works (Arrow to GPU at 57 GB/s). The I/O path is clear (unbuffered ReadFile, no graphics stack). We know the proxy pattern's limits and where the real DataFrame needs to go.

Eight experiments. Three corrections. Two I/O paths (one practical and measured, one research). One convergent customer.

The empty quadrant has a foundation — and the first walls are going up.

---

