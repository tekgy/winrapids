# Expedition Log — WinRapids

*The naturalist's record of the journey.*

---

## Day 1 — 2026-03-20

The expedition begins. An empty directory on an NTFS drive, an RTX PRO 6000 Blackwell with 98GB of VRAM running hot, and a question: can we build something that NVIDIA hasn't — GPU-accelerated data science tools that are native to Windows?

### The Landscape We're Walking Into

I spent the morning surveying what exists outside our camp. The picture is striking in its emptiness.

**RAPIDS cuDF** — NVIDIA's own GPU DataFrame library — does not run on Windows. Not "doesn't run well." *Doesn't run.* The official guidance is: use WSL2, install Ubuntu, pretend you're on Linux. This has been the state of affairs since the project's inception. Issue #28 on the cuDF GitHub repo — "Support for windows?" — is one of the oldest open issues. The answer has always been the same: use WSL.

**GPUDirect Storage** — NVIDIA's fast-path for loading data directly from NVMe into GPU memory — is Linux-only. No Windows support.

**DirectStorage** — Microsoft's answer — adds GPU decompression and a Game Asset Conditioning Library. Two SDK versions matter: **stable 1.3.0** (2025-06-27) has GDeflate GPU decompression only. **Preview 1.4.0-preview1** (2026-03-09) adds `DSTORAGE_COMPRESSION_FORMAT_ZSTD` — confirmed present in the preview SDK, but not yet stable. Preview-only known issues: staging buffer >256 MB may fail on some GPUs (regression — stable 1.3.0 has no upper bound); Zstd GPU fallback shader "still under development." GACL shuffle transform is now an official API feature at request level. Timeline: GDeflate is production-ready today. Zstd is ~6 months out (whenever 1.4.0 goes stable). The architecture is the same either way — compression format is a single enum field in `DSTORAGE_REQUEST_OPTIONS`. Blackwell's hardware Decompression Engine (600 GB/s) supports Snappy/LZ4/Deflate only — NOT Zstd, NOT GDeflate. The DE is B200/GB200 data center class only; the RTX PRO 6000 does not have it. Both codecs on our hardware are shader-based. DirectStorage's value case for WinRapids is CPU offload + NVMe throughput amplification, not hardware-zero-cost decompression. It's designed for streaming game assets, not for data science. But the underlying capability — GPU-accelerated decompression of data flowing from NVMe to VRAM — is exactly what a GPU DataFrame needs. Nobody has pointed DirectStorage at a columnar data format. That remains an unexplored path — GDeflate is prototypeable today, Zstd when the preview stabilizes.

**The WDDM question** is the elephant nobody talks about. Windows forces GPUs through its display driver model (WDDM), which adds overhead to memory management operations. External reports claim "up to 2x slower" for CUDA workloads on WDDM, but our own experiments showed the reality is workload-dependent: **compute kernel execution is barely affected** (~83% of theoretical bandwidth under WDDM), while **memory management is severely degraded** when using raw APIs (managed allocation 30-250x slower, prefetch completely unavailable). With the right APIs (memory pools, pinned memory), WDDM overhead shrinks to single-digit percentages. The "2x" claim likely applies to kernel-launch-dominated workloads (many tiny kernels with frequent allocation/free cycles) — which is relevant for DataFrame ops, but testable and mitigable.

Professional GPUs like our RTX PRO 6000 may support TCC mode, which strips the display driver and gives Linux-like memory management. However, TCC status on this hardware is **ambiguous** (`nvidia-smi -g 0 -dm 1` accepts the command but `nvidia-smi -q` does not enumerate TCC as a supported driver model; reboot test not performed). More importantly, TCC does NOT support `cudaMallocAsync`/memory pools — our primary allocation strategy (AD-4). WDDM is actually the better mode for our workload. See lab notebook Entries 014 and 016 for the full analysis.

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

This is not a gaming GPU pretending to do compute. This is a workstation GPU with ECC memory (hardware present but currently disabled) and strong compute capabilities. TCC mode status is **ambiguous** — `nvidia-smi -g 0 -dm 1` accepts the command but `nvidia-smi -q` does not enumerate TCC as a supported driver model. Even if achievable, TCC would be **worse** for our workload: it does not support `cudaMallocAsync`/memory pools (our 344x allocation performance win). WDDM is the correct operating mode by design, not just by constraint. MIG is not supported on this SKU. MCDM mode (headless compute WITH pool support) requires driver R595+; we have R581.42.

*Note: MIG (N/A) is unavailable on this SKU — workload isolation must use CUDA streams or MPS (Multi-Process Service). TCC status ambiguous and not pursued (would lose memory pools). MCDM is the future ideal mode (R595+ driver required). Memory management must use explicit allocation patterns, not managed memory with prefetch.*

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

**The VRAM surprise**: Desktop compositing (DWM, Explorer, Edge WebView) consumes ~69 GB of the 96 GB available. Only ~29 GB is free for compute at idle. That's still more than most discrete GPUs have *in total*, but it means we're building data science tools with a 69 GB tax. *(Later note: this 69 GB figure is anomalous. Tekgy reports this consumption hasn't been this high before despite three monitors. May be a WDDM memory leak or driver artifact — not a reliable baseline for typical Windows systems. After reboot, WDDM+OS overhead dropped to ~2 GB. The 29 GB figure was a snapshot, not a steady-state measurement. Experiment 002 showed CUDA can reclaim from WDDM regardless, reporting 93 GB available via `cudaMemGetInfo`.)*

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
| Polars GPU engine | Yes | No | Built on cuDF; larger-than-VRAM strategy uses `cudaMemPrefetchAsync` (WDDM-incompatible at API level) |
| RAPIDS Memory Manager (RMM) | Yes | No | GCC-dependent, must build our own pool |
| GPUDirect Storage | Yes | No | Use DirectStorage instead? |
| DirectStorage 1.4 | No | **Yes** | D3D12-only; unbuffered ReadFile proven faster for raw I/O. Research path for GPU decompression (GDeflate built-in; Zstd GPU confirmed in 1.4.0-preview1 SDK, shader-based on RTX PRO 6000) |
| Arrow C Device Data Interface | Yes | **Yes** | The interop standard works everywhere |
| TCC driver mode | Yes (default) | **Ambiguous** | Command accepts but not enumerated; not pursued — TCC lacks memory pools, WDDM is better for our workload |
| MIG (Multi-Instance GPU) | Varies | **No** | Not supported on this Max-Q SKU (N/A) |

The pattern is clear: the low-level CUDA infrastructure works on Windows. Everything built *on top of it* by the data science ecosystem is Linux-only. The gap isn't at the hardware or driver level — it's at the library level.

This is both the opportunity and the mandate. Nobody has built this layer for Windows. The tools to build it (CUDA, Arrow, DirectStorage) are available. The hardware is more than capable. The ecosystem just needs someone to assemble the pieces.

### Correction History: TCC Availability

The TCC story went through three revisions in a single day — worth documenting honestly as a lesson in how first impressions propagate.

1. **First:** "TCC should be available" (speculation from web research about professional GPUs).
2. **Second:** Navigator challenged this. I wrote "TCC is NOT available" based on that challenge.
3. **Third:** Lab notebook Entry 004 — the observer ran `nvidia-smi -g 0 -dm 1` and it accepted the command (exit code 0, "Set driver model to TCC. All done. Reboot required."). Immediately reverted.
4. **Fourth:** Lab notebook Entry 014 — deeper investigation. `nvidia-smi -q` does NOT enumerate TCC as a supported driver model. The command accepting doesn't prove TCC is achievable. Status: **AMBIGUOUS**.
5. **Fifth:** Lab notebook Entry 016 — the killer finding. TCC does NOT support `cudaMallocAsync`/memory pools. Our biggest performance win (344x over raw allocation) would be lost. Even if TCC worked, WDDM is actually better for our workload.

The correction section I originally wrote here went through multiple revisions. The lab notebook is the source of truth. TCC status is **ambiguous, not confirmed, and not pursued** — because even if achievable, it would make our performance *worse*.

**What this means for the expedition:** WDDM isn't a constraint we're working around — it's the correct operating mode for our architecture. The constraint became an advantage. Most Windows users can't or won't switch to TCC either. If our tools work under WDDM, they work for everyone. MCDM mode (headless compute WITH pool support) would be the ideal future mode, but requires driver R595+ (we have R581.42).

The question "is building WDDM-tolerant tools worthwhile?" is now answered by fiat: yes, because that's what we're doing. And the answer from eight experiments is encouraging: with the right APIs (memory pools, pinned memory, async streams), WDDM overhead is minimal for well-structured workloads. The compute path runs at 83% of theoretical bandwidth. The memory path runs at 90% of PCIe theoretical with pinned staging. The overhead concentrates in the places we've already mitigated.

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

**The mechanism and the cliff**: nvidia-smi and `cudaMemGetInfo` measure different things. nvidia-smi reports GPU memory committed to WDDM display allocations. `cudaMemGetInfo` reports what CUDA can claim — but CUDA is *borrowing* from the display allocator, not owning that memory outright. The display stack can demand it back. When the 88 GB capacity probe ran with the display stack active, the competing demands triggered a hard crash (87C, 300W, WDDM memory corruption, full machine reboot). The 93 GB "free" number is a theoretical ceiling; the practical ceiling is lower and dynamic — it depends on screen resolution, number of monitors, HDR state, desktop compositor load, and whatever else the display stack is doing. The 69 GB desktop consumption early in the session was anomalous (Tekgy confirmed it's not normally that high despite three monitors — possibly a WDDM memory leak or driver artifact). After reboot, WDDM+OS overhead was only ~2 GB. A production DataFrame tool must treat the VRAM envelope as dynamic: query `cudaMemGetInfo` before large allocations, reserve 20% headroom, and handle allocation failure gracefully rather than crashing. *(Experiment 015 later confirmed: 88 GB allocated successfully in 1 GB chunks with flat latency, no degradation. The "60 GB ceiling" from this session's crash was likely mid-workload contention, not a fundamental limit. Clean startup: 93.8 GB free, ~1.8 GB DWM overhead.)*

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

At typical Zstd compression ratios for columnar financial data (estimated 3x–8x depending on column type — price columns with small deltas can hit 8:1, volume columns 3:1–5:1, noisy quote data as low as 1.5:1), effective NVMe throughput scales proportionally. On our hardware with 8.8 GB/s NVMe read (experiment 008), that's 26–70 GB/s effective throughput. *(Note: compression ratios are estimates, not measured on FinTek data — actual ratio to be benchmarked in the DirectStorage experiment. Decompression on RTX PRO 6000 is shader-based, not hardware-accelerated. Blackwell's hardware Decompression Engine is B200/GB200 data center class only and doesn't support Zstd or GDeflate. The value case is CPU offload + NVMe throughput amplification, not zero-cost decompression.)*

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

*(Later correction: The built-in GPU decompression format in DirectStorage is GDeflate, not Zstd. UPDATE: `DSTORAGE_COMPRESSION_FORMAT_ZSTD` confirmed in **preview** 1.4.0-preview1 SDK — not yet in stable 1.3.0. GDeflate is production-ready today; Zstd is ~6 months out. Both are shader-based on our hardware (no hardware DE on RTX PRO 6000). See "The Parquet Loading Pipeline" section below for the full analysis.)*

### Late Addition: The Parquet Scan Bottleneck

The scout also surfaced an arxiv paper ("Do GPUs Really Need New Tabular File Formats?", 2602.17335) with a finding that reframes the entire I/O question: **85% of TPC-H query runtime on GPUs is spent in Parquet scanning, not query execution.**

> **UNVERIFIED CLAIM (2026-03-20):** The scout was unable to locate arxiv 2602.17335 or independently verify the 85% figure from any source. Two scholar searches returned no paper with this title, authors, or specific statistic. This claim is **plausible but unverified** — treat it as a working hypothesis, not a citable fact. Adjacent verified evidence: Takafuji et al. (2022), doi:10.1002/cpe.7454, demonstrated GPU Deflate decompression 1.66-8.33x faster than multi-threaded CPU on A100, confirming GPU can accelerate I/O-bound workloads. The 85% figure itself needs a primary source before it can anchor architectural decisions. Plan: measure I/O% directly in the DirectStorage experiment.

This is significant **if true**. The compute layer we've been excited about (1,677 GB/s bandwidth, 50-200x over pandas) would only touch 15% of real workload time if the 85% figure holds. The other 85% would be I/O and format parsing. The bottleneck isn't computing faster — it's *loading* faster.

This validates two things at once:
1. The unbuffered ReadFile + pinned memory path matters more than kernel optimization for real workloads
2. The DirectStorage+Zstd pipeline, if achievable, would attack the dominant cost

Two I/O paths emerge. The **practical path**: unbuffered ReadFile + pinned staging + async H2D — proven by llama.cpp, buildable today, measured at 8.8 GB/s (experiment 008). The **research path**: DirectStorage + GPU Zstd decompression + CUDA interop — novel, complex, potentially transformative for compressed formats like Parquet. WinRapids should build the practical path first and investigate the research path as a differentiator.

### The Parquet Loading Pipeline

The scout confirmed the DirectStorage+Parquet path is real — not speculative, not theoretical. A second deep dive answered three specific unknowns and surfaced a critical nuance about compression formats.

**Three unknowns resolved (scout's deep dive):**

1. **Byte-range access: YES.** `DSTORAGE_REQUEST` has offset and size fields. No 4 KB alignment restriction (unlike `FILE_FLAG_NO_BUFFERING`, which requires sector-aligned reads). Constraint: the staging buffer must fit the uncompressed output (default 32 MB, configurable via `IDStorageFactory::SetStagingBufferSize`). For Parquet pages this is fine — typical uncompressed page is 1 MB.

2. **Headless operation: YES.** No display, no swap chain required. A D3D12 device can be created headless for pure compute. This matters because AD-1 (WDDM-primary) means we're running with display active, but the DirectStorage D3D12 device doesn't need one of its own.

3. **D3D12/CUDA interop: ZERO COPY.** The path is `ID3D12Resource` → `CreateSharedHandle` → `cudaImportExternalMemory` → CUDA pointer to the *same physical VRAM*. No data movement between API layers — the D3D12 resource and the CUDA pointer reference identical memory. One-time handle import at startup (or per-resource-pool). Per-batch fence synchronization adds microseconds, not milliseconds. Production-ready since CUDA 10, with NVIDIA sample code. The scout's initial characterization was "boilerplate-heavy but not architecturally complex" — accurate about code volume, but understates the architectural significance: **the entire NVMe→GPU pipeline has zero CPU touch on bulk data and zero copies inside the GPU.** The only CPU↔GPU traffic is the tiny Parquet metadata.

**The GDeflate question (RESOLVED):** DirectStorage's built-in GPU decompression format is **GDeflate**, not Zstd. The earlier narrative assumed Zstd was native. The scout's deep dive found GDeflate was the original GPU codec.

**UPDATE (Task #12 resolved):** `DSTORAGE_COMPRESSION_FORMAT_ZSTD` **confirmed present** in DirectStorage **preview** 1.4.0-preview1 SDK (2026-03-09) — NOT in stable 1.3.0 (2025-06-27). Named explicitly in the feature list for both `IDStorageQueue::EnqueueRequest` and `DStorageCreateCompressionCodec` APIs. GACL shuffle transform also added as an official API feature at request level. Zstd shaders accept standard RFC 8878 frames, no conditioning required. Shaders ship compiled inside `dstoragecore.dll`. Preview-only known issues: staging buffer >256 MB regression (stable 1.3.0 has no upper bound — size freely for bandwidth); Zstd GPU fallback shader "still under development."

**Timeline:** GDeflate GPU decompression is production-ready today (stable 1.3.0). Zstd GPU decompression is ~6 months out (whenever 1.4.0 goes stable). The architecture is identical either way — compression format is a single enum field in `DSTORAGE_REQUEST_OPTIONS`.

**Hardware decompression reality (also from Task #12):** Blackwell's hardware Decompression Engine (600 GB/s) supports Snappy/LZ4/Deflate only — NOT Zstd, NOT GDeflate. The DE is B200/GB200 data center class only. The RTX PRO 6000 does **not** have a hardware DE. Both Zstd and GDeflate GPU decompression on our hardware are **shader-based**. DirectStorage's value case for WinRapids is CPU offload + NVMe throughput amplification, not hardware-zero-cost decompression.

**Both verifications complete:**
1. ~~Check the DirectStorage 1.4 SDK headers for `DSTORAGE_COMPRESSION_FORMAT_ZSTD`.~~ **RESOLVED: Enum confirmed in preview 1.4.0-preview1 (not stable 1.3.0).**
2. ~~Does the open-sourced Zstd HLSL shader accept standard Zstd frames directly?~~ **RESOLVED: Standard RFC 8878 Zstd frames, no conditioning required.**

**The pipeline is unblocked.** Two prototyping options: (A) GDeflate with stable 1.3.0 SDK today, or (B) Zstd with preview 1.4.0-preview1 SDK (install via `nuget install Microsoft.Direct3D.DirectStorage -Version 1.4.0-preview1-2603.504`). GDeflate prototype first is lower risk.

**Why it works (both blocking items now confirmed):** Zstd-compressed Parquet pages are standard RFC 8878 Zstd frames. No Parquet-specific wrapper around the compressed bytes. DirectStorage's GPU Zstd decompressor handles standard Zstd — confirmed from shader source (`zstdgpu_CollectFrames` takes a memory block of standard Zstd frames, traverses by RFC 8878 block structure, no preprocessing). GACL shuffle transform is a separate optional feature for texture assets, not a prerequisite for Zstd decompression. The CPU-side Parquet metadata reader knows exact byte offsets and sizes for every page in the file (Parquet's `PageHeader` structs carry `compressed_page_size` and `uncompressed_page_size`). Byte-range requests target exact page boundaries — no alignment padding needed.

**The pipeline:**

1. **CPU reads Parquet footer + page headers** — byte offsets, compressed sizes, column types. Pure metadata, no decompression.
2. **CPU constructs DirectStorage requests** targeting exact byte ranges for each compressed page -> each request targets a D3D12 resource in VRAM. (Byte-range access confirmed — no alignment restriction.)
3. **GPU decompresses in-flight** via DirectStorage's Zstd shader -> decompressed Arrow-format column data lands in VRAM. (Standard RFC 8878 frames confirmed compatible — no conditioning step. `DSTORAGE_COMPRESSION_FORMAT_ZSTD` confirmed in SDK. Shader-based on RTX PRO 6000 — no hardware DE.)
4. **`cudaImportExternalMemory`** imports the D3D12 shared handle into CUDA address space. (One-time handle setup, per-batch fence sync in microseconds.)
5. **CUDA kernel operates** on the column data directly. No CPU touch. No PCIe round-trip for the bulk data.

**Catch 1: DataPage V2 mixed buffers.** Parquet DataPage V2 compresses only the values portion, not the rep/def level bytes. The on-disk layout is `[uncompressed levels][compressed values]`. The DirectStorage request must target just the compressed values portion — the page header provides the offsets to extract it. The offset arithmetic is page-level metadata, CPU-readable, no GPU access needed — another instance of the co-native split applying naturally. The CPU reads the page header, calculates where the compressed bytes start, and constructs the DirectStorage request targeting only that range. The GPU shader sees a clean Zstd frame. This is the piece that makes this a real research problem rather than straightforward implementation: the mixed-format page boundary is CPU-parseable but the shader needs a contiguous compressed block, so the request construction must be page-aware.

**Catch 2: Page size alignment.** DirectStorage's Zstd shader is optimized for chunks up to 256 KB. Default Parquet page size is 1 MB. The decompressor handles larger pages (it's spec-compliant Zstd), but peak throughput may favor smaller pages. The GPU Parquet research (arxiv 2602.17335) independently found that smaller pages improve GPU kernel utilization — so this is alignment, not conflict. A WinRapids Parquet writer could default to 256 KB pages to match DirectStorage's sweet spot.

*The 256 KB overdetermination:* Two independent optimization criteria — DirectStorage shader throughput (storage API design) and GPU kernel utilization (memory access patterns) — converge on the same page size from completely different parts of the stack. This is the kind of structural rhyme that signals a real grain in the problem rather than a coincidence of defaults. It also points toward something larger: a WinRapids Parquet *writer* that defaults to 256 KB pages + Zstd would produce files faster to read on **any** GPU-accelerated system, not just WinRapids. That's a format decision, not a reader optimization — ecosystem infrastructure rather than a local win. Files written optimally for GPU consumption are faster forever, on every platform.

**Catch 3: Staging buffer sizing.** Uncompressed data for each request must fit in the staging buffer (default 32 MB). For Parquet pages (typically 1 MB uncompressed) this is not a constraint. For large column groups submitted as a single request, increase via `SetStagingBufferSize`. **Stable 1.3.0 has no upper bound** on staging buffer size — docs explicitly say "you are not limited to 32 MB" and recommend large buffers for bandwidth (sweet spot typically 128–512 MB depending on workload). The **256 MB limitation is a 1.4.0-preview1 regression only** — "larger staging buffer sizes above 256 MB may cause unexpected failures on some GPUs" — listed as a known issue separate from the Zstd shader problem, fix in progress. The GDeflate prototype on stable 1.3.0 can size the staging buffer freely for optimal bandwidth. When testing the preview Zstd path, stay under 256 MB or accept possible failures until the fix ships.

**What makes this Windows-native:** No Linux equivalent exists for the *combined* pipeline. GPUDirect Storage on Linux uses cuFile for NVMe→GPU DMA, but cuFile does *raw uncompressed* transfers — GPU-side decompression requires a separate CUDA kernel step. DirectStorage's integrated GPU decompression (NVMe → decompress → VRAM in one API call) is Windows-only. The D3D12↔CUDA zero-copy interop means WinRapids achieves the same end result as cuFile on Linux: NVMe data lands in CUDA-addressable VRAM with no CPU touch on bulk data. The path is different (DirectStorage → D3D12 → CUDA interop vs cuFile → CUDA directly), but the result is architecturally equivalent — and the Windows path adds integrated GPU decompression that cuFile doesn't have. The prototype experiment should measure whether the extra D3D12 API layer adds measurable latency compared to cuFile's cleaner path, or whether the zero-copy interop makes it invisible.

*(The zero-copy D3D12↔CUDA interop finding is detailed in point 3 of "Three unknowns resolved" above. This is the key insight that elevates the DirectStorage pipeline from "useful optimization" to "zero-CPU-touch parity with Linux GPUDirect Storage.")*

**What makes this hard:** Every arrow in the pipeline is a different API layer (Win32 file I/O → DirectStorage → D3D12 → CUDA interop). Error handling crosses four abstraction boundaries. But zero-copy interop means the D3D12 layer has no data-movement cost — the complexity is API ceremony (handle sharing, fence synchronization), not architecture. The genuine engineering challenges are: (1) DataPage V2 mixed-buffer handling (CPU must construct page-aware requests), (2) composing the full pipeline end-to-end for the first time, and (3) measuring whether the composed end-to-end latency is competitive — each link works individually, but nobody has measured them composed.

**Three benchmark paths (scout's recommendation):**
- **(A)** DirectStorage with GDeflate-compressed columns (built-in GPU decompression — guaranteed to work)
- **(B)** DirectStorage raw uncompressed (NVMe -> VRAM fast path without decompression)
- **(C)** Unbuffered ReadFile + cudaMemcpy baseline (experiment 008's proven path at 8.8 GB/s)

Benchmark A proves the pipeline works end-to-end and measures composed latency (every API layer in sequence). Benchmark B isolates DirectStorage's I/O advantage from decompression. Benchmark C is the control. The question is no longer "does each link work" (confirmed) but "what's the end-to-end latency of the composed pipeline" — D3D12 zero-copy means no data-movement overhead between layers, but API ceremony (fence sync, handle management) may add latency that only shows up in the composed measurement.

**Benchmark design notes (scout's refinements):**
- **Windowed CPU measurement for path A:** Measure CPU% *between* `EnqueueRequest` and the fence signal, not aggregate. That window is the overlap opportunity — is the CPU actually free to pipeline the next file read during GPU decompression? Aggregate CPU utilization will look low regardless; windowed measurement is the only way to verify the CPU offload claim.
- **Small-read latency crossover:** DirectStorage has fixed API setup costs (D3D12 validation, queue submission overhead). For small Parquet row groups (<10 MB), fence-to-first-byte latency may exceed unbuffered ReadFile even if bandwidth is equivalent at large sizes. Measure at 1 MB / 10 MB / 100 MB / 1 GB to find the crossover point. If DirectStorage only wins at >= 100 MB, that constrains which Parquet files benefit.
- **Queue depth vs throughput sweep:** DirectStorage is designed for batched, pipelined I/O — the docs explicitly say "submit as many requests at a time as you can" and warn of penalties when requests are enqueued into a full queue. Sweep queue depth at 1 / 4 / 16 / 64 requests per batch, measuring throughput (GB/s) and CPU utilization at each depth. This is diagnostic for two reasons: (1) it determines whether WinRapids' Parquet reader needs explicit batching logic or whether single-shot requests are competitive, and (2) it maps directly to the Polars IO plugin path (Experiment 007's `register_io_source` yields batches, which could feed DirectStorage batch submission). For Parquet access patterns: predicate pushdown (2–3 columns) is a small batch, latency-sensitive — may not benefit. Full scan (20+ columns) is a large batch, throughput-sensitive — this is where queue depth should win. If queue depth == 1 is competitive, the simpler path works everywhere. If 16+ is needed, the reader architecture must batch.

- **Compression layer isolation:** Parquet's internal encoding is the hidden multiplier in the compression ratio estimates. For price columns, Parquet applies `DELTA_BINARY_PACKED` encoding at write time — converting absolute float64 prices to small integer deltas — before Zstd compresses the result. Zstd then sees highly-repetitive small deltas, not raw float64. This two-stage conditioning is why 8x is plausible on price data. The benchmark should measure both: (1) Parquet-native (DELTA_BINARY_PACKED + Zstd) — what DirectStorage will actually decompress, and (2) raw Zstd on float64 — isolating Zstd's contribution alone. If the ratio splits something like 8x total vs 2x raw, Parquet's delta encoding is doing 4x of the work. This matters for framing: WinRapids isn't just using Zstd — it's using Zstd on Parquet-conditioned data, a better-than-general-purpose compression scenario. The full decompression chain: `DELTA_BINARY_PACKED` (stored on disk, applied at write time) → Zstd decompression (GPU shader) → CUDA kernel receives delta-encoded integers → kernel reconstructs absolute values via prefix-sum. *(Correction: "operate directly on deltas" is limited to the trivial case — sum of deltas = last minus first. Most financial analytics (VWAP, volatility, range, filtered aggregations) require absolute values. The delta encoding's value is compression ratio, not compute simplification.)*

**Kernel design note (library phase):** The more interesting optimization is **selective prefix-scan reconstruction.** If a WHERE clause selects 1% of 10M rows, only 100K positions need reconstruction (prefix-sum the deltas up to each qualifying position) rather than reconstructing all 10M then filtering. A CUDA kernel that does filtered reconstruction — compute running sum only at predicate-passing positions — accesses 100x less VRAM than full reconstruction followed by filter. Don't reconstruct then filter; filter then reconstruct selectively. This applies to any prefix-encoded column (RLE, frame-of-reference), not just `DELTA_BINARY_PACKED`.

**Bandwidth recalibration (scout's observations):** Pinned `cudaMemcpy` already runs at 57 GB/s (experiment 002) — near PCIe 5.0 theoretical. DirectStorage cannot beat pinned cudaMemcpy on raw bandwidth; the PCIe bus is the ceiling for both.

But the scout identified a second angle that changes the calculus: **NVMe throughput amplification.** Experiment 008's 8.8 GB/s unbuffered NVMe read is well below the 57 GB/s PCIe ceiling. NVMe read speed — not PCIe transfer — is the true bottleneck for large datasets. GPU Zstd decompression doesn't need to beat PCIe; it needs to *amplify NVMe throughput*. At estimated 3x–8x Zstd compression ratios on financial columnar data (column-type dependent: price deltas compress best at ~8x, volume ~3–5x, noisy quotes ~1.5x), effective throughput becomes 26–70 GB/s from an 8.8 GB/s drive. The high end is plausible because Parquet's `DELTA_BINARY_PACKED` encoding preconditions price data into small deltas before Zstd sees it (see compression layer isolation note above). The case for DirectStorage is not "faster than cudaMemcpy" — it's "read less from disk by decompressing on GPU instead of CPU." *(Compression ratios are estimates — no peer-reviewed benchmarks exist for Zstd on financial time series. Actual ratios to be measured on FinTek sample data in the DirectStorage experiment, with layer isolation to attribute compression to Parquet encoding vs Zstd.)*

This reframes the benchmark design: the comparison isn't DirectStorage vs pinned cudaMemcpy (that's a PCIe question where both hit the same ceiling). The comparison is DirectStorage-with-GPU-decompression vs unbuffered-ReadFile-with-CPU-decompression on the same compressed file. The question is whether GPU decompression frees the CPU and whether the effective throughput exceeds what CPU-side decompression can sustain.

**Status:** Research path, **unblocked.** Two prototyping options: GDeflate with stable 1.3.0 SDK (production-ready today) or Zstd with preview 1.4.0-preview1 SDK (~6 months from stable). GDeflate prototype first is lower risk. GPU decompression is shader-based on our hardware (no hardware DE on RTX PRO 6000). Value case: CPU offload + NVMe throughput amplification (estimated 3x–8x effective from 8.8 GB/s drive via GPU decompression, pending measurement on FinTek data), not hardware-zero-cost decode. The practical path (unbuffered ReadFile + pinned + async H2D at 8.8 GB/s) remains proven and ships first regardless.

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

Long-term: if Polars ever exposes a backend plugin interface, WinRapids could provide the Windows GPU backend via Arrow Device Data Interface. But the navigator identified a deeper barrier: Polars' larger-than-VRAM strategy relies on `cudaMemPrefetchAsync` for UVM page migration — the exact API that WDDM breaks. A Windows backend for Polars wouldn't just need to swap out cuDF's kernels. It would need to replace the entire memory management strategy: explicit staging buffers with async streams instead of UVM page migration. That's a deeper integration point than just exposing Arrow Device arrays. It's also exactly what our tiered architecture (experiment 006) was designed for — explicit promotion between tiers, no transparent spilling, no prefetch dependency.

**The IO Source as Query Executor Pattern (scout's observation):** Polars' `register_io_source` yields `pl.DataFrame` batches — CPU DataFrames, not device arrays. If WinRapids uses the IO source just as a loader (DirectStorage → VRAM → D2H → yield DataFrame), Polars runs its CPU query engine on the batches, negating the GPU advantage. The data pays full D2H cost on every batch.

The viable path inverts this: **the IO source callback does GPU compute internally and yields only the reduced result.** The callback has full Python scope — it can launch DirectStorage loads, CuPy kernels, GPU joins, fused expressions, and groupby aggregations *inside* the callback, then yield a small post-aggregation DataFrame. Polars thinks it's receiving batches of loaded data; actually it's receiving pre-aggregated GPU results. WinRapids' IO plugin isn't a Parquet reader — it's a GPU query executor presenting results as small DataFrames.

This is the only way to maintain VRAM residency through compute in the Polars integration path. The pattern also maps naturally to the DirectStorage queue depth design: the IO source callback submits a batch of DirectStorage requests (matching Polars' batch-yielding cadence), GPU-decompresses and computes in VRAM, and yields only the tiny result. The full NVMe → DirectStorage → D3D12 → CUDA interop → compute → small D2H result pipeline runs inside a single callback invocation.

**Consequence for API design:** The Polars integration surface is `winrapids.gpu_source(path, schema)` — returns a `pl.LazyFrame` backed by a WinRapids IO source. The user writes normal Polars lazy queries; Polars pushes predicates and projections down automatically; WinRapids' callback receives them and dispatches GPU execution. No manual query plan specification needed — the IO source callback already receives `predicate` (serialized `pl.Expr`, deserializable), `with_columns` (projection pushdown), `n_rows` (limit pushdown), and `batch_size` (query planner's cadence hint) from Polars.

```
# User writes:
lf = winrapids.gpu_source("ticks.parquet", schema)
result = lf.filter(pl.col("symbol") == "AAPL").select(["price", "volume"]).mean()

# Polars pushes down: predicate=(symbol=="AAPL"), with_columns=["price","volume"]
# WinRapids callback: DirectStorage loads only those columns, GPU-decompresses,
# applies predicate in VRAM, computes mean, yields 1-row DataFrame
```

This is explicit opt-in (not monkey-patching like `cudf.pandas`), but transparent once opted in. Polars handles query planning; WinRapids handles GPU execution. The IO source contract supports this — nothing requires the callback to be I/O-only.

**Degradation path (not an error path):** `pl.Expr.deserialize(predicate)` can fail on complex expressions or Polars version skew. When the predicate isn't GPU-dispatchable, the callback falls back to: DirectStorage loads full columns → GPU decompresses → D2H full column → Polars CPU applies predicate. The I/O advantage (DirectStorage + GPU decompression) is retained; the GPU compute advantage is lost. Still faster than CPU load + CPU filter on the same compressed file. This implies the GPU kernel dispatch needs a **passthrough mode** — load and decompress into VRAM, D2H without computing. That's also the minimum viable prototype: the DirectStorage pipeline with no compute stage, useful for benchmarking the I/O path before compute kernels are written.

### The Three-Tier Competitive Position

The navigator sharpened the competitive framing further. WinRapids' position isn't a single comparison — it's three tiers, each with a different shape:

**Tier 1 — vs pandas: 50-200x.** Straightforward, well-documented by experiments 004 and 011. Every operation we've tested is dramatically faster. This is the easy story.

**Tier 2 — vs Polars CPU: 8-23x.** The comparison is narrower and somewhat unfair — Polars has already captured the low-hanging fruit through Rust-native vectorization and query optimization. The 8-23x gap is real but concentrates in operations Polars can't vectorize well: custom aggregations, multi-column filtered ops, large rolling windows. For I/O-dominated workloads, Polars' advantage shrinks further because the bottleneck is I/O, not compute — our own Experiment 014 measured 96.7% of GPU pipeline time spent in H2D transfer (111 ms of 115 ms). The unverified "85% Parquet scanning" claim from arxiv 2602.17335 (see Entry 026) is in the same direction but our own measurement is the authoritative number until the DirectStorage experiment gives us a better one. The scout's framing: the right positioning for Tier 2 is "GPU acceleration layer for Polars users," not "Polars replacement." Arrow interop is near-zero-cost (0.3 ms) — the switching cost is low enough that users can reach for GPU acceleration on specific operations without abandoning Polars for everything else.

**Tier 3 — vs nothing: unique capabilities.** This is where WinRapids wins without competition. No existing tool offers:
- **GPU-resident multi-step workflows.** Polars loads, computes, unloads. WinRapids keeps data on GPU between operations. For "run 500 simulations on this 10M-row dataset," Polars would reload and re-allocate every simulation. WinRapids pays the H2D cost once and runs all 500 on resident data. The per-operation advantage (23x) isn't the right number — the residency advantage compounds with simulation count. With GPU-resident data, simulation N+1 starts from where N left off with zero PCIe round-trips between simulations. *Pending measurement: Parquet + multi-simulation workflow end-to-end benchmark to close the Tier 3 story with a real number, not a conceptual argument.*
- **Explicit tier management with cost visibility.** `.memory_map()` shows which columns are on GPU, which on CPU, what it costs to promote. `.query_plan()` estimates transfer costs before execution. No CPU DataFrame library can even express this. An AI agent reasoning about a FinTek query plan can use these to make informed decisions about resource allocation.
- **Agent-readable metadata without GPU access.** The co-native split means an AI agent reads schema, types, tier locations — all from CPU memory, no GPU roundtrip. This is structurally different from "a DataFrame with a faster backend."

The navigator's product frame for this: **"The step after Polars."** Polars does ingestion, filtering, joins, aggregations — fast, Rust-native, excellent. The moment the workload becomes "sustained compute on large resident datasets" — repeated simulations, streaming analytics, multi-pass algorithms — Polars is the wrong tool. That's where WinRapids starts. Not replacing Polars; extending the pipeline past what Polars can do.

### The Ambition Shift: Beat RAPIDS on Linux

*Directive from Tekgy, late Day 2. The goal is no longer "Windows-native RAPIDS equivalent." It's WinRapids becoming the fastest way to run GPU data science — beating RAPIDS on Linux.*

This changes what we're building toward. The three-tier competitive position still holds, but a fourth tier emerges:

**Tier 4 — vs RAPIDS on Linux: beat it.** Not "equivalent on Windows." *Faster, on Windows.* Three paths to get there, each attacking a different axis:

1. **Win the I/O stage (DirectStorage + GPU decompression).** The arxiv GPU Parquet research found 85% of query time is Parquet scanning. Linux has no GPU-side Parquet decompression — RAPIDS reads through CPU libcudf. DirectStorage + GPU Zstd/GDeflate is a Windows-native pipeline advantage that RAPIDS cannot match. Win the I/O stage, win the benchmark. GDeflate is prototypeable today (stable SDK 1.3.0). Zstd when 1.4.0 stabilizes.

2. **Win the compute stage (CUDA Tiles on Blackwell).** Our RTX PRO 6000 runs CUDA 13.1 with Blackwell-specific structured memory access for columnar scans. Nobody has optimized columnar GPU operations for this architecture yet — RAPIDS' kernels target Ampere/Hopper. First-mover window before RAPIDS catches up.

3. **Win by category (GPU persistent observer).** Not "faster groupby" — a new capability class. RAPIDS can't match it on any platform because nobody has conceived it. A continuously running GPU measurement instrument that produces signals, not query results. If WDDM jitter permits it (Task #10), this is the "something RAPIDS can never do on Linux" differentiator.

The competitive framing inverts. Tier 1-3 ask "how does WinRapids compare to X?" Tier 4 asks "what can WinRapids do that nobody else can?" The answer is a combination: Windows-native I/O advantages + Blackwell first-mover + architectural novelty (persistent observer, co-native metadata, explicit residency).

Eight exploration paths are open. Self-directed. Journey before destination — but the destination is now "fastest in the world."

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

The answer, after the first eight experiments and a day of surveying, is **yes — with specific engineering requirements.**

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
- ~~Can custom CUDA kernels match cuDF's performance for groupby, sort, join?~~ **PARTIALLY RESOLVED:** Experiments 011-013 demonstrated GPU groupby (22-706x over pandas), joins (174-412x over pandas), and fused expressions (3.5x over CuPy eager). No direct cuDF comparison possible (cuDF doesn't run on Windows), but absolute performance is strong. Remaining gap: warp-shuffle reductions (2x over shared-memory tree).
- What does the Python API surface look like? pandas-compatible? Polars-compatible? Something new? (Experiments 004/006/010b established GpuFrame, TieredFrame, FusedColumn as prototypes — the "something new" path is emerging, but the public API shape is still open.)
- ~~How do null/validity bitmaps, string columns, and GroupBy change the architecture?~~ **PARTIALLY RESOLVED:** GroupBy answered by experiments 011-012 (dual-dispatch, sort vs hash). Null/validity bitmaps and string columns remain open — these are the next data-type frontier.
- ~~BLOCKING: Does `DSTORAGE_COMPRESSION_FORMAT_ZSTD` exist in the DirectStorage 1.4 SDK headers?~~ **RESOLVED: Confirmed in preview 1.4.0-preview1 SDK (not stable 1.3.0).** Zstd is ~6 months from stable. GDeflate available today in stable 1.3.0. Both shader-based on RTX PRO 6000. Next: GDeflate prototype with stable SDK.
- ~~BLOCKING: Does the Zstd HLSL shader accept standard Zstd frames or require GACL preprocessing?~~ **RESOLVED: Standard RFC 8878, no conditioning.** Shaders ship compiled inside `dstoragecore.dll`.
- Is the full DirectStorage + GPU decompression + CUDA interop pipeline worth the complexity vs unbuffered ReadFile? **Reframed:** D3D12↔CUDA interop confirmed zero-copy (same physical VRAM). Complexity is API ceremony, not architecture. Value case: CPU offload + NVMe throughput amplification via shader-based GPU decompression (estimated 3x–8x effective). Next step: install SDK, measure composed end-to-end latency (each link works individually, but nobody has measured them composed).
- How does VRAM availability vary with display state (resolution, monitors, HDR)? The CUDA envelope is dynamic, not fixed — production tools need to handle fluctuation.
- **NEW (from AD-10 diagnostic):** What does pinned H2D staging do to the Experiment 014 end-to-end number? Estimated ~43 ms transfer (with `np.copyto` step) → ~5.6x single-query speedup. Arrow→pinned zero-copy ruled out without custom Arrow MemoryPool (Phase 3). Cheapest next experiment: add `cudaHostAlloc` staging buffer + `np.copyto` to Experiment 014's transfer path and re-measure.

### The Architecture That Emerged

Not designed top-down. Emerged from fifteen experiments (eight in Phase 1, six in Phase 2, plus Experiment 015 memory management), a landscape survey, a FinTek convergence, and garden entries:

1. **Memory layer**: CUDA memory pools (`cudaMallocAsync`) as default. Simple exact-match pool for hot paths (0 us reuse). Pinned host pool for staging. Never raw cudaMalloc in production code.

2. **Data layer**: Arrow columnar format in device memory. ArrowDeviceArray metadata on host (co-native: readable by humans and AI agents without GPU access). DLPack for zero-copy GPU interchange.

   **Why this interface is positioned correctly** (navigator's observation): The Arrow C Device Data Interface specifies `sync_event` as an opaque handle — the producer fills it, the consumer waits on it. On CUDA this is a `cudaEvent_t`. But the spec is device-agnostic: same interface for ROCm, Vulkan, Metal. If WinRapids columns are Arrow Device arrays, then a future where Windows gets ROCm support or Intel Arc compute matures requires no public interface change — just a different backend producing the same Arrow Device array. The dual-path design (backend-as-contract) isn't just about CuPy-vs-custom; it's about CUDA-vs-everything-else at the same abstraction level. The interface we've converged on doesn't just solve today's problem. It's at the right level of abstraction for problems we can't predict yet. The cuDF contrast sharpens this: cuDF built downward to the metal (custom memory management, custom columnar format, tight CUDA coupling) and got trapped there — Linux-only, not portable. WinRapids converged on the boundary layer (Arrow Device arrays as protocol), which means the metal underneath can change. That's not a feature we planned — it's a consequence of finding the overdetermined interface first.

3. **I/O layer**: Unbuffered ReadFile -> pinned host staging -> async H2D. Parquet deserialization via PyArrow (sub-millisecond). Pipeline: load batch N+1 while computing on batch N.

4. **Compute layer**: CuPy for memory management. Custom CUDA kernels (via CuPy RawKernel) for performance-critical operations. CuPy's high-level ops are too slow for DataFrame workloads.

5. **Co-native layer**: Column metadata carries location (device/host/storage), type, shape, provenance. Explicit — not hidden behind transparent spilling. Both human and AI agents read the same metadata from the same place.

### The Principle: Residency by Default

The navigator named it: **Residency by default.** Once data enters GPU memory, it stays there until explicitly needed on the host. Operations chain on-device. Results are only materialized to host memory when a user asks for them.

This isn't just an optimization preference. It's the principle that *explains* the experimental results:

- **Experiment 004** (GpuFrame filtered sum = 0.38 ms, 92x): data stays on GPU between comparison and reduction. No crossing.
- **Experiment 005** (proxy filtered sum = 59 ms, 0.5x): data crosses GPU->CPU->GPU between comparison and reduction. Three crossings kill the speedup.
- **Experiment 006** (tiered cold-start = 7 ms, still 4x): first access pays the crossing cost. Every subsequent operation is free because data *stays*.
- **Experiment 008** (mmap slower than Python read): mmap crosses the page-fault boundary per 4KB page. Unbuffered ReadFile crosses once for the whole buffer.

The expensive thing is never the work on either side of the boundary. It's the *crossing*. io_uring recognized this for syscalls and batched them. WinRapids recognizes it for PCIe transfers and keeps data resident. The architectural shape is the same.

A precision from the scout sharpens this further: the cost scales with **entanglement depth** — how deeply operations share intermediate state — not with operation count. A simple `sum()` crosses once (GPU result -> CPU scalar). A `filtered_sum()` crosses twice if naively split (filter returns mask, sum consumes mask). A naive groupby would cross once per group key. The proxy failed not just because transitions are expensive, but because the intermediate boolean mask is *entangled* with the data — serializing it across PCIe materializes the full intermediate state. This has a design implication: `filtered_sum(mask)` is the right primitive because it fuses filter and sum on-device. Separate `filter()` then `sum()` would force materialization even in GpuFrame if the filter result crossed any boundary. The API surface should be designed to minimize intermediate boundary crossings, not just individual operation costs.

Three things together form the architecture:
1. **Residency by default** — data stays where it lands until explicitly moved
2. **Arrow Device arrays as the output contract** — the interface is device-agnostic (CUDA today, anything tomorrow via `sync_event`)
3. **Explicit memory pools as the allocation primitive** — 0.5 us alloc/free, no WDDM tax

Not designed top-down. Discovered from the numbers.

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

*(Phase 2 answered: Experiments 009-010 measured the abstraction cost at 1.2-3.9x. The from-scratch ceiling for filtered sum is 0.084 ms (vs 0.331 ms CuPy). Kernel fusion eliminates the gap for expression chains. And experiment 010b showed the paths converge — Python codegen captures 85-95% of C++ fusion benefit. See "Phase 2 — The Dual-Path Experiments" below.)*

Also a safety constraint: **~88-90 GB usable VRAM ceiling.** *(Revised: Experiment 015 successfully allocated 88 GB in 1 GB chunks with flat latency. The earlier "60 GB ceiling" from an Experiment 002 corruption incident was likely mid-workload, not a fundamental WDDM limit. Baseline: 95.6 GB total, 93.8 GB free at startup, ~1.8 GB DWM overhead.)* Production heuristic: `cudaMemGetInfo` minus 20% headroom as soft ceiling.

### The Frame

This morning I called the project "the empty quadrant." There is no GPU DataFrame library native to Windows. By tonight, we have more than a floor — we have a working prototype. GpuFrame runs 50-200x faster than pandas in 200 lines of Python. The memory layer is solved (pools bypass WDDM overhead). The data bridge works (Arrow to GPU at 57 GB/s). The I/O path is clear (unbuffered ReadFile, no graphics stack). We know the proxy pattern's limits and where the real DataFrame needs to go.

Eight experiments. Three corrections. Two I/O paths (one practical and measured, one research). One convergent customer.

The empty quadrant has a foundation — and the first walls are going up.

---

## Phase 2 — The Dual-Path Experiments

*Experiments 009-011 answered the Phase 1 questions directly. The dual-path directive (AD-5) moved from theory to measurement.*

### Experiment 009: The Abstraction Tax, Measured

The question was specific: how much does CuPy cost vs raw CUDA C++? Experiment 004's GpuFrame was 50-200x over pandas. How much of that is CuPy's abstraction, and how much is the GPU being fast?

The pathmaker built the same operations — sum, min/max, filtered sum, FMA — in raw CUDA C++ with warp-shuffle reductions. The answer is a table:

| Operation | CUDA C++ | GpuFrame (CuPy) | Abstraction cost |
|-----------|----------|------------------|------------------|
| Sum | 0.082 ms (976 GB/s) | 0.099 ms | 1.2x |
| Min/Max | 0.062 ms (1299 GB/s) | 0.123-0.137 ms | 2.0-2.2x |
| Filtered sum | 0.084 ms (1072 GB/s) | 0.331 ms | 3.9x |
| FMA (a*b+c) | 0.192 ms (1668 GB/s) | 0.531 ms | 2.8x |

The abstraction tax is 1.2-3.9x depending on the operation. Two things dominate:

1. **Kernel fusion** — CuPy's FMA launches two kernels (multiply, then add) where raw CUDA does one. Each kernel launch pays a fixed overhead, and worse, the intermediate buffer (80 MB for 10M doubles) gets written and read unnecessarily.

2. **Warp-shuffle vs tree reduction** — CuPy's high-level `sum()` uses shared-memory tree reduction. Raw CUDA's warp-shuffle eliminates shared memory entirely. For filtered sum (where the mask adds indirection), warp-shuffle is 3.9x faster.

Vectorized `double2` FMA showed no benefit (0.200 ms vs 0.192 ms scalar) — already bandwidth-bound. This confirms the Blackwell is memory-bound on these operations; instruction throughput is not the bottleneck.

An intermediate test with CuPy RawKernel showed 0.172 ms for filtered sum (vs 0.084 ms for pure CUDA). So even going "raw" through CuPy's RawKernel interface still carries a 2x overhead from the reduction algorithm choice. The abstraction cost isn't just in the Python layer — it's in the algorithm.

**Architecture decision**: CuPy for prototyping, custom CUDA for production hot paths where operations compose. The 1.2x cost for simple sum is acceptable. The 3.9x cost for filtered sum is not — and filtered operations are exactly what DataFrame queries do.

### Experiment 010: Kernel Fusion — Expression Complexity Is Free

If the abstraction tax comes from kernel fusion (or lack thereof), can we eliminate it? Experiment 010 built a compile-time fusion engine using C++ expression templates.

The idea: build an expression tree at compile time (C++ template metaprogramming), evaluate every element through the entire tree in a single kernel, zero intermediate buffers. `a*b+c` compiles to one kernel that reads a, b, c and writes the result. Not three kernels (multiply, then add, then store).

The results are remarkable:

| Expression | Fused | CuPy (eager) | Speedup | VRAM saved |
|-----------|-------|-------------|---------|------------|
| a*b+c | 0.194 ms | 0.291 ms | 1.5x | 80 MB |
| a*b+c*c-a/b | 0.193 ms | 0.670 ms | 3.5x | 320 MB |
| where(a>0,b*c,-b*c) | 0.195 ms | 0.565 ms | 2.9x | 320 MB |
| sum(a*b+c) | 0.177 ms | 0.336 ms | 1.9x | — |
| sqrt(abs(a*b+c*c-a)) | 0.188 ms | 0.657 ms | 3.5x | 400 MB |

**All fused kernels run at ~0.19 ms regardless of expression depth.** Five operations or one — doesn't matter. They're bandwidth-bound at ~1650 GB/s. Expression complexity is free.

The template-fused FMA (0.194 ms) matches the hand-written CUDA kernel from experiment 009 (0.192 ms). Zero abstraction cost for the fusion mechanism itself. The templates are generating the same quality of code as hand-tuned CUDA.

The fusion advantage grows with expression complexity. Simple ops: 1.5x. Complex chains: 3.5x. VRAM savings scale linearly with dataset size — at 10M doubles, `sqrt(abs(a*b+c*c-a))` saves 400 MB of intermediate buffers. At 100M rows, that's 4 GB. On the ~88 GB usable VRAM ceiling (AD-4), that difference determines whether your dataset fits.

### Experiment 010b: Fusion Comes to Python

The C++ fusion engine proved the principle. But WinRapids is a Python library. The pathmaker bridged the gap: a `FusedColumn` class that builds lazy expression trees in Python, generates CUDA kernel source via string templating, and compiles at runtime via CuPy's RawKernel.

Python fused vs CuPy eager: 1.4-3.2x speedup — capturing 85-95% of the C++ fusion benefit. Python fused vs C++ fused: only 0.01-0.05 ms slower (Python call overhead, not kernel quality). The generated kernels are as fast as the C++ templates because both produce the same CUDA code — the fusion happens in the expression tree, not in the host language.

JIT overhead: negligible. NVRTC (NVIDIA's runtime compiler) caches at the driver level. First compilation is fast; subsequent calls with the same expression tree hit the cache.

**Architecture decision**: Python codegen is the production path for kernel fusion. C++ templates are the reference implementation for validation. No C++ build step needed in the user-facing pipeline. This means the from-scratch path and the pragmatic path *converge* — you get from-scratch kernel quality through pragmatic-path tooling.

This convergence is worth pausing on. AD-5 assumed the dual paths would produce separate backends (`backend="cupy-generic"` vs `backend="cuda-custom"`). Instead, experiment 010b showed you can get custom-quality kernels through CuPy's compilation machinery. The "from-scratch" insight (fusion) gets delivered through the "pragmatic" toolchain (CuPy RawKernel). The paths aren't separate backends — they're layers of the same backend.

### Experiment 011: GroupBy — The Core Gap Filled

This is the experiment that matters most for the "empty quadrant" thesis. Every data scientist asks the same question: "can I do `df.groupby('category').sum()` on a GPU on Windows?" Until now, the answer was no.

Two approaches, both on GPU:

**Sort-based**: argsort the group key, then segmented cumsum. Stable at ~3.3 ms regardless of cardinality — argsort dominates. Works at any cardinality but doesn't exploit low contention.

**Hash-based**: atomic scatter into hash table buckets. Performance depends on cardinality:

| Groups | Sort-based | Hash-based | pandas | Best GPU speedup |
|--------|-----------|-----------|--------|-----------------|
| 100 | 3.3 ms (21x) | 3.1 ms (22x) | 69.7 ms | 22x |
| 10K | 3.4 ms (26x) | 0.5 ms (176x) | 87.7 ms | 176x |
| 1M | 3.4 ms (113x) | 0.5 ms (706x) | 378.3 ms | 706x |

At 100 groups, atomic contention limits hash and both approaches are similar. At 10K+, hash dominates — less contention per bucket means atomics fly. At 1M groups, hash is **706x** faster than pandas.

Multi-aggregation (sum+mean+count): sort once, derive all aggregations from the sorted structure. 3.4 ms vs pandas 98.6 ms (29x).

**Architecture decision**: Dual-dispatch groupby with cardinality estimation. Below ~1K groups: either approach works. Above ~1K: hash wins dramatically. The dispatch threshold should be adaptive, not fixed — a quick pass over the key column to estimate distinct values costs less than picking the wrong algorithm.

This fills the core gap. `df.groupby('category').sum()` on GPU on Windows: proven, measured, 22-706x faster than pandas depending on cardinality. The empty quadrant has its first real citizen.

---

## The Custom Path Principle

*Standing directive from Tekgy, 2026-03-20. Governs how WinRapids responds to barriers.*

**When something hits a barrier — API limitation, library constraint, missing feature — default to building the custom version instead of working around it.**

The workaround reflex is natural: find a hack, add a shim, accept the constraint. The custom path feels harder. But the calculus is different for this team: we have the entire history of GPU computing to learn from. We can study every prior implementation, take the best ideas, and write optimized code without their constraints. Zero compromise.

**Examples from the expedition:**
- Arrow→pinned zero-copy impossible without custom MemoryPool? Build the MemoryPool. Or write a custom Parquet column reader that reads directly into pinned memory (~200 lines C++).
- CuPy RawKernel codegen has 5-15% overhead vs hand-tuned CUDA? Custom nvrtc compilation.
- Hash join correctness issues with CuPy atomics? Write it in CUDA C++.
- Polars IO source unstable or insufficient? Build our own query dispatch.

**The pattern:** Learn from what exists. Take the best ideas. Build the custom version without their constraints. The barrier that blocks a workaround is often the signal that points toward the right custom implementation.

This is not AD-0 — it's the meta-principle that governs how ADs get made. Every architectural decision below emerged from experiments that hit barriers. The custom path principle says: when you hit the barrier, go through it, not around it.

---

## Architectural Decisions

*Settled decisions that persist across days. Each emerged from experiments, not from top-down planning.*

### AD-1: WDDM-Primary Design

**Decision:** WinRapids designs for WDDM as the primary and preferred target. TCC is neither confirmed available nor desirable (lacks memory pools).

**Rationale:** WDDM forces explicit memory management, which produces better architecture — deterministic behavior, no hidden page migrations, predictable latency. A well-designed WDDM memory pool isn't a degraded version of TCC; it's a clean design that happens to work on both. Critically, TCC mode does NOT support `cudaMallocAsync`/memory pools — our primary allocation strategy (AD-4). WDDM isn't just "what we're stuck with"; it's actually better than TCC for our workload. Additionally, the development environment is RDP-based (TCC would break the display session), and the end users we're building for are overwhelmingly WDDM users.

**Consequence:** No code may depend on `cudaMemPrefetchAsync`, UVM transparent migration, or TCC-only features. Larger-than-VRAM datasets require explicit partitioning and streaming with pinned staging buffers — not page faults. This is more predictable than UVM anyway.

**Evidence:** All experiments (001-014) ran under WDDM with standard Windows desktop compositor active, RDP session, 144 Hz display scheduling. No special driver modes, no isolated compute environments. Every benchmark is a production number — the same conditions end users will have. This is a differentiator: most GPU benchmarks are measured in TCC mode or headless Linux, then quietly degrade in production. WinRapids' numbers are already production numbers, asterisk-free.

### AD-2: Residency by Default

**Decision:** Once data enters GPU memory, it stays there until explicitly materialized to host. Operations chain on-device. Results are only moved to host when a user asks.

**Rationale:** The expensive thing is the domain crossing (PCIe transfer), not the work on either side. Experiment 005 proved this: the proxy's filtered sum was *slower than pandas* (0.5x) because it crossed the GPU/CPU boundary three times. GpuFrame's filtered sum (92x) kept everything on-device.

**Consequence:** `frame["a"].sum()` returns a device-resident scalar, not a Python float. Chained operations produce GPU-resident intermediates. Materialization is always explicit (`.item()`, `.to_host()`, `.to_arrow()`). Display of values requires materialization; display of metadata (co-native split) does not.

**The `repr()` implication (navigator's observation):** This is load-bearing for co-native design at the API surface. `repr(frame)` must read CPU-side metadata only — column names, dtypes, shapes, tier locations — and show `<on device>` for values. If `repr()` triggered a D2H transfer to display values, every Python REPL interaction would cross the boundary. An AI agent calling `repr(frame)` to understand DataFrame structure shouldn't trigger GPU operations. The co-native split must be respected all the way through to the display layer.

**README framing:** "Data at rest on the GPU tends to stay on the GPU, unless acted upon by an explicit materialization request." — Newton's first law of GPU data. One sentence that sets the mental model before any code is shown.

**Evidence:** Experiments 004 (GpuFrame 92x) vs 005 (proxy 0.5x). Experiment 008 (mmap slower than read — same pattern at the I/O level).

### AD-3: Arrow Device Arrays as Output Contract

**Decision:** WinRapids columns are Arrow Device arrays at the public interface level. The Arrow C Device Data Interface is the interop contract.

**Rationale:** The interface is device-agnostic (`sync_event` handles CUDA, ROCm, Vulkan, Metal). It naturally splits metadata (CPU-resident, co-native) from data (device-resident). DLPack provides zero-copy GPU interchange within the Arrow ecosystem. Polars and PyTorch consume Arrow. Five independent constraints (co-native split, DLPack, Polars interop, FinTek convergence, WDDM metadata needs) all converge on this interface.

**Consequence:** Backends can be swapped without changing the public API (`backend="cupy-generic"` vs `backend="cuda-custom"` vs future `backend="rocm"`). Consumers get a standard struct, not a WinRapids-specific type. If Polars ever exposes a GPU backend plugin API, the natural contract is Arrow Device arrays — we would already speak that language. Our current "failure" to integrate with Polars is correct positioning for when the integration point opens.

**The protocol insight (navigator's observation):** `sync_event` is what makes Arrow Device arrays not just a data format but a *protocol*. Null `sync_event` = synchronous (data ready now). Non-null = opaque device event (e.g., `CUevent*`), consumer waits before reading. That's a complete producer-consumer contract, not just a pointer to bytes. The protocol is what makes backend swapping possible — any backend that speaks the protocol can produce and consume without knowing what's on the other side. The overdetermination of this interface — five independent constraints converging on it without any of them requiring it — suggests we *discovered* the right abstraction level rather than *designing* it. The right shape was already there.

**Evidence:** Experiment 003 (DLPack zero-copy verified), experiment 006 (tier enum maps to Arrow device types). Navigator's `sync_event` universality and protocol observations.

### AD-4: Explicit Memory Pools

**Decision:** All GPU memory allocation goes through CUDA memory pools (`cudaMallocAsync`/`cudaFreeAsync`). Never raw `cudaMalloc` in any performance-sensitive path. Hot paths use a simple exact-match pointer-reuse cache on top.

**Rationale:** WDDM imposes extreme allocation overhead: `cudaFree` p99 = 6.4 seconds for 1 GB. Memory pools bypass WDDM entirely: 0.5 us regardless of size. 240-344x faster.

**Consequence:** *(Revised by Experiment 015.)* VRAM baseline: 95.6 GB total, 93.8 GB free at startup, ~1.8 GB WDDM/DWM overhead. Experiment 015 successfully allocated 88 GB in 1 GB chunks with flat latency (~11.5 ms each, no degradation). The earlier "60 GB ceiling" from an Experiment 002 corruption incident was likely mid-workload, not a fundamental WDDM limit. Practical usable ceiling: ~88-90 GB with comfortable headroom. Production heuristic still applies: query `cudaMemGetInfo` at allocation time, reserve 20% headroom, treat as soft ceiling. The ceiling is dynamic (monitor config, HDR, compositor load). Two-tier strategy: simple cache for same-size reuse (~0 us), CUDA pools for general case (~1.4 us). CuPy's `MemoryAsyncPool` (`cudaMallocAsync`-backed) is the Windows-native choice — same primitive, no RMM port needed.

**Evidence:** Experiment 002 (pools 0.5 us, raw free 6.4s). Experiment 015 (88 GB allocated in chunks, flat latency, 93.8 GB free baseline). CuPy MemoryAsyncPool confirmed as RMM alternative.

### AD-5: Dual-Path Design

**Decision:** Every experiment explores both a pragmatic path (existing libraries: CuPy, Arrow, Polars) and a from-scratch path (raw CUDA C++/Rust, custom formats). The from-scratch path is the measurement instrument. The pragmatic path is the delivery vehicle. They aren't alternatives — the C++ reference kernel validates the Python codegen kernel, and both ship: one as an experiment, one as library code.

**Rationale:** The pragmatic path validates architecture quickly. The from-scratch path finds the performance ceiling. The gap between them is the abstraction tax — and it's only measurable if you build both. Experiment 010b proved that from-scratch quality is deliverable through the pragmatic toolchain: Python codegen captures 85-95% of C++ fusion benefit.

**Consequence:** Not two separate backends (`backend="cupy-generic"` vs `backend="cuda-custom"`) as originally assumed. Instead, two layers of the same backend: C++ expression templates as the reference implementation for validation, Python codegen via CuPy RawKernel as the production path. The "custom" quality gets delivered through the "pragmatic" toolchain. Exception: warp-shuffle reductions still need hand-tuned CUDA (2x gap over shared-memory tree reduction in CuPy).

**Evidence:** Experiment 009 measured the abstraction tax at 1.2-3.9x (CuPy vs raw CUDA C++). Experiment 010 proved compile-time fusion eliminates the gap for expression chains (fused matches hand-written at 0.194 ms vs 0.192 ms). Experiment 010b showed Python codegen via CuPy RawKernel captures 85-95% of C++ fusion benefit — the dual paths converge rather than diverge.

### AD-6: Unbuffered ReadFile for I/O

**Decision:** File-to-GPU loading uses unbuffered ReadFile (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`) into pinned host memory (`cudaHostAlloc`) with async H2D transfer. Not mmap, not DirectStorage.

**Rationale:** 8.8 GB/s end-to-end at 1 GB, 2.7x faster than naive Python read. mmap is *slower* than Python read (page fault overhead). DirectStorage requires D3D12 intermediary (4 API boundary crossings vs 1). llama.cpp independently validated this path.

**Consequence:** Allocate a pinned staging buffer once (~256 MB). Read file chunks with unbuffered ReadFile. Async H2D overlapping with next read. No D3D12 dependency, no graphics stack.

**Evidence:** Experiment 008. llama.cpp's independent finding.

### AD-7: Kernel Fusion via Python Codegen

**Decision:** Expression chains compile to fused CUDA kernels at runtime via Python string templating + CuPy RawKernel. Lazy expression trees build the AST; `evaluate()` generates and launches a single fused kernel.

**Rationale:** CuPy eager evaluation launches one kernel per operation, creating intermediate buffers at each step. Fusion eliminates all intermediates and reduces kernel launches to one. For complex expressions (5+ ops), this is 3.5x faster and saves hundreds of MB of VRAM. Python codegen captures 85-95% of C++ template fusion benefit without a build step.

**Consequence:** Arithmetic on WinRapids columns is lazy by default. `result = a * b + c * c - a / b` builds an expression tree. `.evaluate()` or consumption by a reduction triggers kernel generation and launch. NVRTC driver-level caching means repeated expressions compile once.

**Evidence:** Experiment 010 (C++ templates match hand-written CUDA at 0.194 ms vs 0.192 ms). Experiment 010b (Python codegen 85-95% of C++ benefit, 1.4-3.2x over CuPy eager).

### AD-8: Dual-Dispatch GroupBy

**Decision:** GroupBy dispatches between sort-based and hash-based algorithms based on estimated cardinality. Below ~1K groups: either works (sort preferred for stability). Above ~1K: hash-based with atomic scatter.

**Rationale:** Sort-based groupby is stable at ~3.3 ms regardless of cardinality (argsort dominates). Hash-based scales inversely with cardinality — at 1M groups, it's 706x faster than pandas. Smart dispatch captures the best of both.

**Consequence:** GroupBy needs a cardinality estimation pass before dispatching. The estimation cost must be less than the wrong-algorithm penalty. Multi-aggregation (sum+mean+count) sorts once and derives all aggregations.

**Evidence:** Experiment 011. Sort-based: 21-113x over pandas. Hash-based: 22-706x over pandas (scales with cardinality).

### AD-9: Join Dispatch by Key Type

**Decision:** GPU joins dispatch between direct-index (for dense integer keys in [0, N)) and sort-merge (for arbitrary keys). Same dual-dispatch pattern as groupby.

**Rationale:** Direct-index join uses the key itself as an array index — O(1) per probe, no hashing, no collisions. At 0.65 ms for 10M × 10K, it's the fastest join in the expedition (382x over pandas). Sort-merge via `searchsorted` handles arbitrary keys at O(n_fact × log n_dim), still 174-233x over pandas. GPU hash table build (CPU-side Python loop) hits a wall at large dimension tables.

**Consequence:** Star schema queries with integer keys (the common FinTek case: product_id, symbol_id, desk_id) get the fastest path automatically. Arbitrary-key joins still work, just through the sort path. GPU-side hash table build (atomicCAS) is a future optimization target for large dimension tables with non-integer keys.

**Evidence:** Experiment 013. Direct-index: 382-412x. Sort-merge: 174-233x. Hash (CPU build): 84x at 10K dim, ~1x at 1M dim.

### AD-10: GPU-Resident Data Model as Primary Optimization

**Decision:** The primary performance optimization is not faster kernels — it's keeping data on the GPU. The second query on resident data should cost compute time only (5 ms), not I/O time (159 ms).

**Rationale:** Experiment 014 proved that GPU compute for a full analytics pipeline (join + expression + groupby on 10M rows) takes 5.1 ms. The H2D transfer takes 110 ms — 97% of pipeline time. Faster compute yields diminishing returns. Residency eliminates the dominant cost entirely.

**Consequence:** The tier system (experiment 006) isn't just a nice abstraction — it's the performance strategy. Data promoted to GPU tier stays there. The `repr()` co-native pattern (CPU metadata only, no D2H for inspection) preserves residency. Every API decision should ask: "does this force unnecessary materialization?"

**The compound calculation (FinTek's actual workload):** 500 simulations on 10M rows.
- Simulation 1: 159 ms (includes H2D transfer)
- Simulations 2-500: 5.1 ms each (resident data, no H2D)
- **WinRapids total: 159 + (499 x 5.1) = ~2.7 seconds**
- **Pandas total: 510 ms x 500 = ~255 seconds**
- **Real-world speedup: ~94x** (not the first-query 3.2x, not the single-operation 90x — 94x for the actual use case)

The first-query 3.2x undersells WinRapids by 30x for multi-simulation workloads. The microbenchmark results (50-700x per operation) are technically accurate but miss the point. The number that matters for FinTek is 94x on 500 simulations — and it scales: 1000 simulations would be ~97x, approaching the asymptotic compute-only speedup as the single H2D cost amortizes to nothing.

**Diagnostic note (scout's observation):** The 110 ms H2D transfer in Experiment 014 uses **pageable memory**, not pinned. The code path is `Arrow.to_numpy()` → `cp.asarray()` — both pageable. Data size: 10M rows × 4 float64/int64 columns = ~320 MB. At pageable rates (~2.9 GB/s observed), that's 110 ms. Pinned staging has three steps: (1) Arrow→numpy (~5 ms, already fast), (2) `np.copyto` into pinned buffer (~32 ms at ~10 GB/s pageable memcpy), (3) pinned→VRAM via `cp.asarray` (~5.6 ms at 57 GB/s DMA). **Total: ~43 ms** — 2.6x better than pageable 110 ms. End-to-end: ~91 ms (48 ms Parquet read + 43 ms pinned transfer). Single-query speedup: 510/91 = **~5.6x** over pandas (up from 3.2x). The 500-simulation compound: ~91 ms + 499 × 5.1 ms = ~2.6 seconds (marginal change — H2D paid once either way).

**Arrow→pinned zero-copy (ANALYTICALLY CLOSED: requires custom MemoryPool).** Arrow's `to_numpy()` is zero-copy — it returns a numpy array whose data pointer *is* Arrow's internal buffer address (`arrow_buffer.address == numpy.__array_interface__['data'][0]`). Arrow owns the memory; numpy wraps it. But Arrow allocates buffers in pageable system RAM, so the zero-copy view points to pageable memory. For Arrow to place data into *our* pinned allocation, Arrow would need to allocate from pinned memory at construction time — which requires intercepting allocation via a custom `MemoryPool`. There's no API surface in pyarrow to redirect an existing Arrow buffer to an external address without rebuilding the array. (`pyarrow.Buffer.from_address(ptr, size)` exists but requires data already at that address — circular, requires the copy first.) This is determined by Arrow's memory ownership model, not a limitation we can work around. **For now, the 43 ms target stands.** The custom Arrow pinned MemoryPool backed by `cudaHostAlloc` is the Phase 3 path to ~11 ms (~8.6x over pandas). **The cheapest next experiment is the `np.copyto` pinned staging path — ~43 ms, no Arrow changes needed.** Higher immediate payoff than DirectStorage.

**Evidence:** Experiment 014 end-to-end. First query: 159 ms (110 ms pageable I/O + 5 ms compute). Subsequent queries on resident data: ~5 ms.

### Execution Model: Four Quadrants

Not a decision yet — a map for the library phase. Two axes: eager vs lazy execution, and materialized vs resident return types. These produce four quadrants:

|  | **Materialized** (returns Python value) | **Resident** (returns GpuColumn) |
|--|----------------------------------------|----------------------------------|
| **Eager** | `frame["a"].sum().item()` — compute immediately, return float | `frame["a"] * frame["b"]` — compute immediately, result stays on GPU |
| **Lazy** | `frame.query_plan().execute().item()` — plan, then compute, then materialize | `frame.lazy().filter(...).groupby(...).collect()` — plan, then compute, result stays on GPU |

WinRapids currently lives in the **eager+resident** quadrant. Experiment 004's `GpuFrame` computes immediately and keeps results on GPU. Experiment 010b's `FusedColumn` is a partial step toward lazy — it builds an expression tree but evaluates eagerly on `.evaluate()`. TieredFrame's `query_plan()` (experiment 006) is the embryo of the lazy quadrant — it estimates costs before committing to execution.

The future direction is **lazy+resident**: build an operation graph, optimize it (fuse expressions, eliminate dead columns, reorder joins), then execute the optimized plan with results staying on GPU. This is the quadrant that Polars occupies on CPU (lazy+materialized) and that RAPIDS cuDF occupies on GPU (eager+resident, with some lazy via Dask).

**Why this matters now:** When the pathmaker starts building the library layer, return types implicitly commit to a quadrant. An operation that returns a `GpuColumn` stays resident. An operation that returns a Python float has already materialized. An operation that returns a `LazyFrame` defers both. These aren't implementation details — they're the execution model. Having the map drawn before those decisions prevents accidental commitment to the wrong quadrant.

**The kernel fusion connection:** Experiment 010b's expression tree is a lazy evaluation mechanism scoped to element-wise operations. A full lazy engine would extend this to relational operations (join, groupby, filter) — building a query DAG rather than an expression tree. The fusion engine becomes a component of the larger lazy evaluator, not a standalone feature.

### Production Constraint: JIT Cold-Start

Not a design decision but a deployment reality that must be addressed before WinRapids ships.

**The problem:** CuPy's first kernel execution in a process incurs ~157 ms of JIT compilation overhead (Python import + CUDA context initialization + nvrtc compilation). This is 475x slower than a warm kernel execution (0.33 ms). A user running `df.sum()` for the first time in a session experiences a half-second pause before seeing GPU speedups. This makes demos look broken and can cause users to abandon the tool before they understand the steady-state performance.

**Two distinct JIT costs:**
1. **CuPy context cold-start (~157 ms):** First-ever CuPy kernel in the process. Python import, CUDA context creation, driver initialization. Happens once per process. Not avoidable without pre-warming.
2. **Expression codegen JIT (~0 ms warm):** nvrtc compilation for new fused kernel expressions. Driver caches compiled PTX — first evaluation of a new expression structure pays compilation cost, all subsequent calls reuse the cache. Negligible in practice after the first few operations.

**Mitigations (for future implementation):**
- **Eager initialization at import time:** `import winrapids` (or `winrapids.initialize()`) triggers a dummy kernel launch to pay the 157 ms cost during library initialization, not during the first user operation. This is a known pattern in CUDA production systems (scout's observation): defer it and users see a 157 ms spike on their first DataFrame operation that looks like a bug. Initialize eagerly and it's invisible.
- **AOT compilation for known kernels:** Pre-compile the most common kernel patterns (sum, filtered_sum, fused_sum, groupby) to PTX at install time. Ship the PTX alongside the package.
- **Progress indication:** If pre-warming takes >100 ms, show a brief "initializing GPU..." message rather than silent delay.

**Why this matters now:** This constraint should be documented in the WinRapids README when it exists. The 50-200x speedup story is only true after warm-up. Users need to know this upfront, not discover it through a confusing first experience.

**Evidence:** Observer's lab notebook Entry 011 (cold-start measurement). Entry 021 (distinction between CuPy context cold-start and nvrtc per-expression JIT).

### The Four Barrier Types — A Diagnostic Framework

The expedition kept hitting walls that looked different on the surface but had a common structure. Four types of barrier, ordered by how hard they are to detect:

**Type 1 — Library barrier.** The wheel doesn't exist for your platform. cuDF on Windows. Visible immediately: `pip install` fails. You know within seconds.

**Type 2 — API barrier.** The API doesn't exist. Visible at compile time or import time. Clear error, obvious fix (find alternative or implement yourself).

**Type 3 — Semantic barrier.** The API exists, compiles, links, runs — but doesn't do what you'd expect. `cudaMemPrefetchAsync` on WDDM: the function exists, accepts arguments, returns a CUDA error code — but does nothing useful. The semantic barrier produces the six-hour debugging session because everything *looks* correct. Often there IS a flag that tells you: `cudaMemPrefetchAsync` returns `cudaErrorInvalidDevice` on WDDM if you check the return value. But library code written for Linux often swallows the error because the operation is "optional." The WinRapids principle: **treat all barrier types as hard errors. No silent degradation.**

**Type 4 — Temporal barrier.** The capability is announced but not yet in the current SDK. A feature described at a conference, with source code published, that isn't in the stable release yet. These pass surface-level research and only fail at SDK-header depth. Example: DirectStorage Zstd GPU decompression was announced at GDC 2026 with HLSL source published — but the stable SDK (1.3.0) only ships GDeflate. The Zstd enum landed in the *preview* SDK (1.4.0-preview1), resolving the temporal barrier. Nuance: once the SDK ships (even as preview), the barrier type shifts — what remains is an *installation gap*, not a temporal gap. The feature exists; you just haven't installed it yet. Different diagnosis, different fix.

Types 1 and 2 are honest failures — they tell you immediately. Types 3 and 4 are dishonest — they pass surface-level validation and only reveal themselves through measurement (Type 3) or header inspection (Type 4). Temporal barriers have a lifecycle: announced → preview SDK → stable SDK. At each transition, the barrier type changes. A capability in preview is no longer temporal (it exists) but may still be an installation or stability barrier.

One additional observation: `cudaMemcpy`'s semantics are fully specified and unchanged across WDDM, TCC, MCDM, and Linux. It's the one CUDA operation with no semantic variance across driver modes. Everything built on `cudaMemcpy` inherits that stability. This is why explicit staging (pinned host buffer + `cudaMemcpy`) is the robust I/O path — it's immune to barrier types 3 and 4.

**Evidence:** `cudaMemPrefetchAsync` failure on WDDM (Experiment 001). DirectStorage GDeflate vs Zstd temporal barrier lifecycle — announced, preview SDK, resolved (scout's deep dive + Task #12). Polars UVM dependency (Experiment 007 analysis).

---

## Day 2 — 2026-03-20 (Continued)

*Session resumed after machine reboot and context reset. GPU clean: 2 GB used, 95 GB free, 55C.*

### The From-Scratch Turn

Day 1 proved the architecture on the pragmatic path. CuPy + Arrow + Python produced working GPU DataFrames with 50-200x speedups over pandas. Then came the directive: go custom for everything. Even 1.2x on sum matters at FinTek's scale. So Day 2 is the from-scratch turn — raw CUDA C++, hand-tuned kernels, no CuPy in the hot path.

The results from Day 2's first experiments (009-011) are surprising in a specific way: the two paths *converge* faster than expected.

Experiment 009 measured the CuPy abstraction cost: 1.2-3.9x depending on operation. Significant, but not uniform. The gap was largest where operations *compose* — where CuPy is forced to materialize intermediates that a fused kernel doesn't need. The gap was smallest where CuPy already uses the optimal primitive (sum: already warp-shuffle internally, only 1.2x).

Experiment 010 then showed that C++ expression template fusion *matches* hand-written kernels — zero abstraction cost from the template machinery. And Experiment 010b showed that Python codegen via CuPy RawKernel captures 85-95% of that C++ benefit. The paths converge: write Python → get C++-quality CUDA. No build step.

This is the key insight of Day 2 so far: **you don't have to choose between the pragmatic path and the from-scratch path. The from-scratch quality is deliverable through the pragmatic toolchain.** The expression template engine is the reference; Python codegen is the production path. Same kernels, different dispatch.

### Where the Paths Don't Converge

The convergence breaks down at GroupBy.

Experiment 011 established that GPU groupby works — 21-706x over pandas depending on approach and cardinality. Two algorithms: sort-based (stable ~3.3 ms) and hash-based (scales from 22x at 100 groups to 706x at 1M groups).

Experiment 012 tested three approaches to fused groupby: CuPy unfused (separate expression evaluation + sort-based reduction), fully fused with atomics (expression + accumulation in one kernel), and hybrid (fuse the expression, separate the reduction).

The prediction was correct — the fully fused atomic approach was terrible: 16.2 ms at 100 groups, 4.6x *slower* than the unfused baseline. Binary search for group boundaries plus atomic contention is a losing combination. But the hybrid approach won everywhere: 3.43 ms at 100 groups, 3.53 ms at 10K groups. For complex 5-op expressions, the hybrid was 1.14x faster than unfused while saving 320 MB of VRAM intermediates.

The lesson is clean: **fuse the computation, don't fuse the reduction.** Atomics are terrible for grouped accumulation at any cardinality. The right architecture is a two-stage pipeline: expression fusion codegen layer → sort-based grouped reducer. Two stages, cleanly composed.

### Experiment 013: GPU Joins — The Other Relational Primitive

Joins are the complement to groupby. Together they cover the two fundamental operations of relational analytics: combining tables and aggregating within groups. Experiment 013 implements three join strategies on GPU:

| Method | 10M × 10K | vs pandas | 10M × 1M | vs pandas |
|--------|-----------|-----------|-----------|-----------|
| Direct-index | 0.65 ms | 382x | 0.78 ms | 412x |
| Sort-merge | 1.06 ms | 233x | 1.85 ms | 174x |
| Hash (CPU build) | 2.93 ms | 84x | 330 ms | ~1x |

The direct-index join is the standout: for dense integer keys (the star schema case), use the key itself as an array index. No hashing, no collisions, O(1) per probe. At 0.65 ms for 10M × 10K, it's 382x faster than pandas — the largest single-operation speedup in the expedition.

Sort-merge via `searchsorted` is the general-purpose workhorse. Works for arbitrary keys, reasonable scaling at O(n_fact × log n_dim). Still 174-233x over pandas.

The hash join exposes a clear failure mode: CPU-side Python loop for hash table construction. At 10K dimension rows, 2.93 ms is acceptable. At 1M dimension rows, 330 ms — the Python build phase dominates entirely. A GPU-side build (atomicCAS for insertion) is the obvious next step, but the sort-merge approach already covers this range well enough.

Architecture decision: **join dispatch by key type.** Dense integer keys → direct-index. Sparse or arbitrary keys → sort-merge. Same dual-dispatch pattern as groupby (Experiment 011).

End-to-end join + groupby pipeline: 6.68 ms vs pandas 322 ms — 48x faster.

### Experiment 014: The Capstone — End-to-End GPU Analytics

Experiment 014 is why the expedition exists.

Ten million sales records. Ten thousand products. Fifty categories. The query: revenue by category, where `revenue = quantity * unit_price * (1 - discount)`. This is the kind of thing a data scientist writes every day in pandas. We built it entirely on GPU, on Windows, without cuDF.

The pipeline: Parquet → Arrow → CuPy (H2D) → direct-index join → fused revenue kernel → sort+cumsum groupby → result.

| Stage | Time | Notes |
|-------|------|-------|
| Read Parquet (Arrow, CPU) | — | Same for both pipelines |
| H2D Transfer | 110 ms | Pageable memory, not pinned |
| Join (direct-index) | 1.4 ms | Product ID → category mapping |
| Fused Compute | 0.24 ms | `qty * price * (1 - discount)` in one kernel |
| GroupBy (sort + cumsum) | 3.4 ms | 50 categories |
| D2H Result | <0.1 ms | 50 floats |
| **GPU total (excl. I/O)** | **5.1 ms** | **~90x faster than pandas compute** |
| **Full pipeline** | **159 ms** | **3.2x faster than pandas end-to-end** |
| Pandas total | 510 ms | Baseline |

Correctness: all 50 category sums match pandas within 3.86e-05.

The numbers tell a specific story. GPU compute — the join, the expression, the groupby — takes 5.1 ms total. That's ~90x faster than the equivalent pandas operations. The 3.2x end-to-end number is almost entirely determined by I/O: the H2D transfer at 110 ms is 97% of the GPU pipeline time.

This is exactly the architecture thesis made manifest. **GPU compute is no longer the bottleneck.** The expedition has compressed the compute cost to near-zero. What remains is I/O — and that's where the residency-by-default principle pays off. The second query on the same data costs 5 ms, not 159 ms. The third query costs 5 ms. The hundredth query costs 5 ms. The transfer cost is paid once; the compute cost is paid every time but it's negligible.

The fused revenue kernel at 0.24 ms is worth noting. Three arithmetic operations on 10M rows: `quantity * unit_price * (1 - discount)`. Without fusion, CuPy would launch three separate kernels, allocating and filling two intermediate buffers of 80 MB each. The fused kernel reads three inputs and writes one output in a single pass. This is experiment 010b's codegen in production context — working exactly as designed.

### The Signal, Revisited

Fifteen experiments. The picture is now complete enough to state:

**WinRapids works.** GPU-accelerated analytics on Windows, without cuDF, without RAPIDS, without TCC mode. CuPy + Arrow + Python codegen + WDDM. The compute path is 50-700x faster than pandas depending on operation. The end-to-end path is 3-48x faster including I/O.

**The architecture was right.** Residency by default. Co-native metadata. Explicit materialization. Arrow as the interchange protocol. Python codegen for kernel fusion. Dual-dispatch for groupby and joins. Every architectural decision made through experiments 001-008 held up through experiments 009-014.

**I/O is the frontier.** GPU compute at 5 ms for a full analytics pipeline makes everything else — Parquet reading, H2D transfer, memory management — the dominant cost. Experiment 014's H2D transfer uses pageable memory (110 ms); pinned staging with `np.copyto` would improve it to ~43 ms, boosting single-query speedup from 3.2x to ~5.6x — the cheapest next experiment (see AD-10 diagnostic note). A custom Arrow MemoryPool backed by `cudaHostAlloc` could further reduce transfer to ~11 ms (~8.6x), but that's Phase 3. GPU-resident data eliminates transfer cost entirely for subsequent queries.

**The competitive position is clear.** vs pandas: 50-200x, commodity replacement. vs Polars CPU: 8-23x, "the GPU acceleration layer for Polars users." vs nothing: persistent GPU-resident analytics with agent-readable metadata — capabilities that don't exist elsewhere on Windows.

---

### FinTek Production Validation

*Pathmaker, 2026-03-20. While the naturalist documented, the pathmaker shipped.*

The `winrapids_fused` backend is built and validated on real production data. The architecture the expedition predicted is now running in FinTek's pipeline.

**What was built:** `R:/fintek/trunk/backends/winrapids_fused/` — a new backend implementing the same interface as `cupy_fused` but replacing cuDF with PyArrow + WinRapids pinned transfer. Key components:
- `transfer.py`: Uses `winrapids.transfer.h2d_batch()` — pinned memory pool with async batch transfers, exactly the pattern the expedition log prescribed
- `runner.py`: Same interface contract as cupy_fused, drop-in replacement
- `write.py`: GPU → pinned D2H → PyArrow parquet write

**What was validated:** Gap leaf (timestamp difference computation at 5 lag depths) on real AAPL market data — 598K rows from `W:/fintek/data/ingest/I01/2025-09-02/AAPL.parquet`. Exact numerical agreement with pandas. **51x compute speedup** on real financial data.

**What this proves:** The full stack works in production context. PyArrow reads parquet, pinned H2D via `PinnedPool` moves it to GPU, CuPy computes, pinned D2H brings results back, PyArrow writes. Zero cuDF, zero RAPIDS, zero Linux. One field change in the leaf config (`backend="winrapids-fused"`) and the pipeline runs on Windows GPU.

**The pinned transfer module** (`R:/winrapids/src/winrapids/transfer.py`) implements exactly what the AD-10 diagnostic note identified as the cheapest next experiment: `PinnedPool` backed by `cudaHostAlloc`, `np.frombuffer` for zero-copy numpy view of pinned memory, `copy_from_host_async` for batched async H2D, single sync. Thread-safe pool with 4 KB alignment for DMA efficiency.

### Experiment 015: GPU Memory Management — Baseline Confirmed

*Pathmaker, 2026-03-20.*

Systematic measurement of pinned memory transfers and VRAM capacity, superseding some Experiment 002 numbers.

**Pinned memory transfers (1 GB, sustained):**

| Direction | Pageable | Pinned | Speedup |
|-----------|----------|--------|---------|
| H2D | 18.1 GB/s | 53.7 GB/s | 2.96x |
| D2H | 9.9 GB/s | 53.2 GB/s | 5.38x |

Pinned gets ~54 GB/s sustained — close to PCIe 5.0 x16 theoretical (64 GB/s). D2H pageable is notably worse than H2D pageable (9.9 vs 18.1 GB/s), making pinned even more critical for result extraction.

**VRAM baseline (clean startup):**
- Total: 95.6 GB
- Free at startup: 93.8 GB
- WDDM/DWM overhead: ~1.8 GB
- Successfully allocated 88 GB in 1 GB chunks with flat latency (~11.5 ms each, no degradation)

This supersedes the earlier "60 GB ceiling" from Experiment 002. The 88 GB allocation succeeded with no corruption. The practical usable ceiling is ~88-90 GB. AD-4 consequence updated accordingly.

---

## GPU Persistent Observer — Design Session

*Campsite reflection, 2026-03-20. Scout and Navigator.*

### The Inversion

Every GPU computing model ever built works the same way: CPU requests, GPU executes, CPU reads. The GPU is a reactive compute target — it waits.

The persistent observer inverts this. The GPU runs continuously, accumulating state. The CPU reads when it needs to, not when the GPU finishes. The GPU is the authority. The CPU is the client.

This has never been built for financial data on Windows workstations. The comparison class is FPGA — a programmable measurement instrument that never stops running. The persistent observer is the GPU equivalent.

### Case 1 vs Case 2 — A Design Boundary WinRapids Must Name

Two fundamentally different data consumption models arise when the GPU is always running:

**Case 1: Order-sensitive state** — every tick matters. Order book, exact VWAP, regulatory audit trail. Not a single update can be dropped. The only valid response to consumer lag is backpressure: slow the producer down. Loss is not acceptable.

**Case 2: Statistical state** — lossy is acceptable. M-tower signal, rolling volatility, threshold crossings, momentum indicators. The GPU is computing a statistical summary over a window. If 50 ticks drop while the consumer is busy, the window estimate degrades slightly — confidence goes down, not correctness. The signal is still valid, just less certain.

FinTek's analytics are Case 2. Almost all real-time ML inference on streaming data is Case 2.

This distinction matters architecturally: Case 1 requires a lossless ring buffer with backpressure. Case 2 requires a lossy ring buffer with confidence accounting. Building one when you need the other produces either deadlocks or false confidence.

### The Lossy Ring Buffer Architecture

For Case 2, the ring buffer never blocks the CPU writer:

- CPU writes always proceed — head advances unconditionally
- When head wraps past the GPU's tail, the overwritten slot gets a new sequence number
- GPU reads the slot; sequence number mismatch means a gap occurred
- GPU atomically counts the gap, jumps tail forward, continues
- Output includes a `dropped_since` field — explicit accounting, not silent corruption

The ring buffer is pinned host memory (`cudaHostAlloc`). Both CPU and GPU access it directly — no `cudaMemcpy` after startup. At 1M ticks/sec with 10ms GPU lag tolerance and 2x safety margin: 20,000 slots × ~30 bytes = 600 KB. This fits entirely in the RTX PRO 6000's 128 MB L2 cache. The GPU hot loop reads from cache, not VRAM. Latency characteristics change significantly.

### The Signal Struct — Interface Contract

The persistent observer's output is a struct both CPU code and AI agents can read without translation:

```c
struct Signal {
    uint64_t as_of_seq;      // last tick sequence number processed
    uint64_t dropped_since;  // ticks dropped since last signal output
    float    value;          // the signal value
    float    confidence;     // 1.0 - (dropped_since / window_size)
};
```

`confidence` is the key field. It quantifies degradation directly in the signal. An AI agent reading this signal doesn't need to know whether the ring buffer had a busy period — the confidence score carries that information. `confidence = 1.0` means nothing was dropped. `confidence = 0.7` means 30% of the window was missing — weight accordingly.

This struct is the API. Not a method call, not a callback, not a log entry. A memory location the GPU writes and anything with a pointer can read. Co-native by construction: human-readable names, machine-readable values, no translation layer.

### The Gate Experiment

The persistent observer architecture has one unknown that blocks viability: WDDM interrupt jitter.

Windows Display Driver Model schedules GPU work around display refresh. On a 144 Hz display, the display stack touches the GPU roughly every 7 ms. If this interrupts a persistent CUDA kernel — and for how long — determines whether tick-rate signals are possible on a workstation GPU with an active display.

**Task #10** is the gate experiment: a 50-line CUDA test. Persistent kernel increments a counter continuously. CPU thread timestamps reads. Plot a histogram of inter-read gaps under 144 Hz display load. If gaps are bounded below 1 ms, the model works. If gaps are unbounded or routinely exceed 10 ms, the architecture requires a dedicated headless GPU or a different design.

Nothing in the CUDA ecosystem documents this histogram. It's unmeasured terrain.

The framing matters: WDDM preemption is not overhead to minimize — it's a signal to characterize and design around. If the jitter profile has known shape (bounded windows at 144 Hz intervals), the persistent observer can be designed to coexist with it: schedule compute work away from preemption windows, use the gaps for output writes, treat the display rhythm as a clock rather than an interruption. A bridge is not designed to eliminate wind — it's designed to resonate with it safely. The histogram tells us what the wind looks like. The design follows from that.

The FPGA comparison class sharpens this further: the question is not "is the GPU faster than CPU at signal generation?" It's "can a programmable GPU replace dedicated signal-generation hardware?" That's a different customer conversation, a different procurement decision, a different capability claim. The persistent observer, if viable, answers the second question — not just "faster" but "category replacement."

### The Architectural Convergence

The architecture we discovered for batch DataFrames turns out to be the architecture for continuous observation. This is not a design choice — it's a consequence of finding the right boundary layer first.

Residency by default (AD-2): in batch mode, data stays on GPU between queries. In observer mode, state stays on GPU between ticks. Same principle, different timescale.

Co-native metadata split (AD-2/AD-3): in batch mode, `repr(frame)` reads CPU metadata without triggering D2H. In observer mode, the Signal struct is CPU-readable without interrupting the GPU kernel. Same split, same reason.

Explicit memory management (AD-4): in batch mode, memory pools avoid WDDM allocation overhead. In observer mode, the pinned ring buffer is allocated once at startup and never freed. Same constraint, same solution.

Arrow Device arrays as protocol (AD-3): in batch mode, the `sync_event` enables producer-consumer handoff between backends. In observer mode, the Signal struct is the output contract — a memory location the GPU writes and anything with a pointer can read. Same pattern: structured data at a boundary, no translation layer.

The persistent observer is not a new project bolted onto WinRapids. It's the same architecture running continuously instead of on-demand. The only genuinely new pieces are the ring buffer ingestion path and WDDM jitter tolerance — everything else is already built. That's the strongest argument that the observer is a natural extension: the architecture already fits. We just haven't run it continuously yet.

---


## The Promotion Decision Problem

*Navigator synthesis, 2026-03-20. Revised with architectural gap analysis.*

AD-10 says GPU-resident data model is the primary optimization (I/O elimination > faster kernels). But AD-10 is silent on *when* to promote. The breakeven formula makes the decision explicit:

```
breakeven_queries = transfer_cost_ms / compute_savings_ms_per_query
```

From Experiment 014's numbers:

| Scenario | Transfer cost | Compute savings/query | Breakeven |
|----------|--------------|----------------------|-----------|
| Pageable H2D (10M rows, ~320 MB) | 110 ms | 5 ms | 22 queries |
| Pinned H2D with copyto (10M rows) | ~43 ms (5 ms Arrow + 32 ms copyto + 5.6 ms DMA) | 5 ms | ~9 queries |
| Pinned H2D via Arrow MemoryPool (Phase 3) | ~11 ms (Arrow allocates into pinned directly) | 5 ms | ~2 queries |
| FinTek real data (~100K rows) | ~0.4 ms (pinned, either path) | 5 ms | <1 query — always promote |

*(Note: Arrow→pinned zero-copy is not possible with the current stack — Arrow allocates in pageable system RAM, so the `np.copyto` step is unavoidable. A custom Arrow `MemoryPool` backed by `cudaHostAlloc` would eliminate the copy (Phase 3). Experiment 014 currently uses pageable path — see AD-10 diagnostic note.)*

The decision function requires three inputs:

1. **Estimated transfer cost** — known from column size + current tier. TieredFrame's `query_plan()` (Experiment 006) already estimates this.
2. **Estimated compute savings** — knowable from operation type + data size. The fifteen experiments provide the calibration data.
3. **Expected query count** — unknown. This is the hard part.

A query planner that makes this decision explicit would be new. Polars optimizes *what* to compute. DuckDB optimizes *how* to compute it. Neither asks "should I move this data to GPU given expected query count?" The TieredFrame's `query_plan()` from Experiment 006 estimates transfer cost — the missing piece is the decision function that uses it.

Three options for the unknown input (expected query count):

1. **User hint:** `frame.pin_to_gpu(expected_queries=50)` — explicit, puts the decision in the caller's hands. An AI agent with workload knowledge could provide this.
2. **Historical:** Track actual query count per column over time, promote after threshold crossed. Requires state persistence across sessions.
3. **Speculative:** Always promote on first query, accept first-query penalty. Bet that the data will be queried again.

WinRapids currently does option 3 implicitly — residency by default, promote on first access. This is correct for FinTek's workload (500-simulation runs where breakeven is <1 query). Making options 1 and 2 available would be genuinely new capability — no existing tool offers this level of promotion-decision visibility.

**This is the architectural gap AD-10 raises but doesn't answer.** The gap isn't urgent (option 3 works for the known workload), but naming it now means the API design can accommodate it. The difference between `frame.to_gpu()` (no decision support) and `frame.promote(expected_queries=N)` (decision-aware) is a design choice that's cheaper to make early.

---
