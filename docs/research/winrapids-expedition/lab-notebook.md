# Lab Notebook — WinRapids Expedition

*Observer's scientific record. What happened, what the numbers say.*

---

## Environment (verified 2026-03-20)

### Hardware
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Product Brand**: NVIDIA RTX (professional/workstation tier)
- **VRAM**: 97,887 MiB (~95.6 GB)
- **Bus ID**: 00000000:F1:00.0
- **Power**: 300W TDP (default), 325W max, 250W min. This is the Max-Q (power-limited) variant. The non-Max-Q RTX PRO 6000 likely has a higher TDP.
- **ECC**: Off
- **TCC mode**: **AMBIGUOUS** — `nvidia-smi -g 0 -dm 1` reports success and sets Pending to TCC, but `nvidia-smi -q` does not list TCC as a supported driver model option. Whether TCC would actually activate on reboot is unverified. See Entry 004 (original assessment) and Entry 014 (revised assessment). **WDDM is the operating constraint for this expedition.**

### Software
- **OS**: Windows 11 Pro for Workstations (10.0.26200)
- **Driver**: 581.42 (WDDM mode)
- **CUDA Toolkit**: 13.1 (nvcc V13.1.115, built 2025-12-16)
- **CUDA_PATH**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
- **Python**: 3.13.12 (Windows Store distribution)

### Python Libraries Available
- NumPy 2.4.2
- pandas 3.0.1

### Python Libraries NOT Available (baseline)
- PyTorch, CuPy, PyCUDA, Numba, Polars, PyArrow

### Observations on Initial State
- nvidia-smi reports "CUDA Version: 13.0" in header, but nvcc reports 13.1. The nvidia-smi CUDA version reflects the driver's max supported CUDA runtime, not the installed toolkit version. The toolkit (13.1) is newer than the driver's reported support (13.0). **This mismatch should be monitored** — it may cause issues if the toolkit uses features beyond what the driver supports, though typically the toolkit is backward-compatible.
- GPU under heavy load at session start: 100% utilization, 86-87C, ~68.8 GB VRAM in use. All processes are Windows desktop apps (DWM, Explorer, Edge WebView, etc.) — no compute workloads. This VRAM usage (~70% of total) by desktop compositing alone is a significant baseline overhead worth tracking. **Approximately 29 GB VRAM available for compute.**
- MSVC compiler (`cl`) not found in PATH. This may affect CUDA kernel compilation if nvcc expects MSVC as host compiler. gcc also not in PATH.
- No CUDA_HOME set (only CUDA_PATH).

---

## Entry 001 — Baseline Environment Audit

**Date**: 2026-03-20
**Type**: Environment characterization
**Status**: Complete

### Purpose

Establish verified baseline measurements before any experiments begin. Document what we actually have, not what we think we have.

### Method

Ran `nvcc --version`, `nvidia-smi`, `python --version`, and import checks for common GPU/data science libraries. Queried GPU properties and running processes.

### Findings

1. **CUDA toolkit is installed and functional** (nvcc responds, correct version).
2. **GPU is a workstation-class Blackwell** — the "Max-Q" suffix suggests a power-limited variant, confirmed by 300W TDP (desktop Blackwell cards typically allow higher power).
3. **WDDM driver mode (by choice, not constraint)** — ~~TCC mode is not available for this GPU class~~ **CORRECTION (Entry 004)**: TCC IS available. The RTX PRO 6000 is a professional workstation card (NVIDIA RTX brand), not a consumer GeForce. The original claim was wrong. WDDM adds overhead vs TCC, but switching to TCC is possible (requires reboot, disables display output from this GPU).
4. **~29 GB free VRAM** at idle (desktop compositing consumes ~69 GB). This is still substantial — more than most discrete GPUs' total — but it means benchmarks must account for this baseline consumption.
5. **No GPU-aware Python libraries installed** — the project starts from scratch. The first experiment will need to install at least one CUDA-capable Python package or compile raw CUDA kernels.
6. **MSVC not in shell PATH but IS installed** — Visual Studio 2022 Community with MSVC 14.44.35207 found at `C:\Program Files\Microsoft Visual Studio\2022\Community`. nvcc should discover this via registry/vswhere. Not a blocker.

### Implications for Expedition

- The WDDM overhead is a central question, but now we know **TCC is available as an alternative**. The scientific question becomes: what is the WDDM overhead, and is it worth staying in WDDM (to keep display output) vs switching to TCC?
- The 29 GB available VRAM is enough for serious data science workloads but must be measured, not assumed.
- MSVC IS installed (VS 2022 Community, MSVC 14.44) — just not in shell PATH. nvcc should find it via vswhere. Not a blocker for Task #1.
- The CUDA version mismatch (toolkit 13.1 vs. driver-reported 13.0) warrants a simple validation test before any performance-critical work.

### Open Questions

1. ~~Can nvcc compile and execute a kernel without MSVC in PATH?~~ **Resolved**: VS 2022 Community with MSVC 14.44 is installed. nvcc should find it via vswhere.
2. What is the actual WDDM overhead for a simple memory allocation + kernel launch?
3. Is the 69 GB desktop VRAM usage normal, or inflated by specific running apps?

---

## Entry 002 — Experiment 001 Code Review (pre-execution)

**Date**: 2026-03-20
**Type**: Code review / methodology assessment
**Status**: Awaiting execution results

### Experiment Design

Pathmaker created `experiments/001-cuda-proof/` with two files:
- `cuda_proof.cu` — comprehensive CUDA proof-of-life test
- `build.bat` — compilation script with architecture fallback cascade

### What is Being Tested

| Test | Size | Purpose |
|------|------|---------|
| Device query | N/A | Report GPU properties, confirm WDDM mode, compute capability |
| CUDA version check | N/A | Runtime vs driver version (addresses our mismatch concern) |
| Vector add | 1K, 1M, 64M elements | Correctness at 3 scales + H2D/D2H/kernel timing |
| Parallel reduction | 1M, 64M elements | Shared memory + synchronization correctness + bandwidth |
| Managed memory | 1M elements | `cudaMallocManaged` + `cudaMemPrefetchAsync` under WDDM |

### Methodological Notes

**Strengths:**
- Warm-up pass before timed kernel runs (avoids first-launch JIT overhead)
- Correctness verification for every test (not just timing)
- Managed memory test explicitly checks prefetch support — a known WDDM pain point
- Build script cascades through sm_120 -> compute_100 -> sm_90, handling architecture uncertainty
- Error handling via CUDA_CHECK macro throughout
- Reports bandwidth in GB/s for comparison with theoretical limits

**Limitations / things to watch for:**
- Timing uses `std::chrono::high_resolution_clock`, not CUDA events. This measures wall-clock time including WDDM scheduling overhead. For a proof-of-life this is fine and arguably *desirable* (we want to see the full cost), but future precision benchmarks should use CUDA events for kernel-only timing and wall-clock for end-to-end.
- `cudaDeviceSynchronize()` after kernel launches adds synchronization cost. In a real pipeline, overlapping compute and transfer would be more representative.
- The build script hardcodes MSVC path (`14.44.35207`) and Windows SDK version (`10.0.26100.0`). Fragile if environment changes.
- Only a single run per test — no statistical aggregation. Acceptable for proof-of-life, not for benchmarks.
- The 64M vector add allocates 3 x 256MB = 768MB on GPU — well within the ~29GB available.

### Key Questions This Experiment Will Answer

1. Does nvcc compile CUDA code on this Windows setup? (Binary question)
2. What is the GPU's compute capability? (Expected: sm_120 for Blackwell)
3. What are the H2D and D2H transfer rates? (Theoretical PCIe 5.0 x16: ~64 GB/s)
4. Does managed memory work under WDDM?
5. Does `cudaMemPrefetchAsync` work under WDDM? (Historically: no)
6. What is the CUDA runtime vs driver version from the API? (Resolves our nvidia-smi observation)

### Predictions (recorded before seeing results)

- Compilation will succeed with sm_120 (Blackwell architecture).
- H2D/D2H will be significantly below theoretical PCIe max due to WDDM overhead and pinned vs pageable memory.
- Managed memory allocation will succeed but `cudaMemPrefetchAsync` will either fail or be a no-op under WDDM.
- All correctness checks will pass.
- Kernel bandwidth for large vector add should approach memory bandwidth limits (~1 TB/s for Blackwell, but this is a simple kernel).

---

## Entry 003 — Experiment 001 Results

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: CUDA works natively on Windows. All tests passed.

### Raw Results

#### GPU Properties (from CUDA runtime API)
| Property | Value |
|----------|-------|
| Device name | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Compute capability | 12.0 |
| Global memory | 95.6 GB |
| SM count | 188 |
| Max threads/block | 1024 |
| Warp size | 32 |
| Memory bus width | 512 bits |
| L2 cache | 128 MB |
| Shared mem/block | 48 KB |
| Memory clock | 14,001 MHz |
| Core clock | 2,280 MHz |
| Unified addressing | yes |
| Managed memory | yes |
| Concurrent kernels | yes |
| Cooperative launch | yes |
| TCC driver | no (WDDM) |
| Compute preemption | yes |

**CUDA Versions** (from API): Runtime 13.0, Driver 13.0. Both report 13.0, which matches the driver. The toolkit's nvcc is 13.1, meaning we compiled with 13.1 headers/tools but the runtime/driver is 13.0. This is normal — forward-compatible compilation.

#### Vector Add Performance

| Size (elements) | H2D (ms) | H2D (GB/s) | Kernel (ms) | Kernel (GB/s) | D2H (ms) | D2H (GB/s) | Correct |
|-----------------|----------|-------------|-------------|---------------|----------|-------------|---------|
| 1,024 (4 KB) | 2.076 | 0.00 | 0.011 | 1.10 | 0.205 | 0.02 | PASS |
| 1,048,576 (4 MB) | 0.706 | 11.89 | 0.012 | 1,057.39 | 0.686 | 6.12 | PASS |
| 67,108,864 (256 MB) | 32.447 | 16.55 | 0.543 | 1,483.34 | 35.833 | 7.49 | PASS |

#### Parallel Reduction Performance

| Size (elements) | Kernel (ms) | Sum | Bandwidth (GB/s) | Correct |
|-----------------|-------------|-----|-------------------|---------|
| 1,048,576 | 0.012 | 1,048,576 | 349.53 | PASS |
| 67,108,864 | 0.203 | 67,108,864 | 1,325.61 | PASS |

#### Managed Memory

| Metric | Value |
|--------|-------|
| Managed alloc time | 74.895 ms |
| Prefetch support | **NOT SUPPORTED** ("invalid device ordinal") |
| Kernel time (1M elements) | 2.978 ms |
| Correct | PASS |

### Analysis

#### 1. Compilation: SUCCESS
The build succeeded with `sm_120` (Blackwell) on the first attempt. No fallback needed. nvcc found MSVC automatically.

**Prediction check**: Predicted sm_120 would work. *Partially wrong* — the architecture flag was sm_120 but the actual compute capability reported is 12.0. The naming is consistent (sm_XY = X.Y), so sm_120 = 12.0. Correct.

#### 2. Transfer Rates: BELOW THEORETICAL BUT EXPECTED

Theoretical PCIe 5.0 x16 bandwidth: ~63 GB/s bidirectional.

**H2D**: 16.55 GB/s at 256 MB — only ~26% of theoretical PCIe 5.0 max.
**D2H**: 7.49 GB/s at 256 MB — only ~12% of theoretical.

**Why so low?** Three factors:
1. **Pageable memory**: The test uses `malloc()` (pageable host memory), not `cudaMallocHost()` (pinned). Pageable transfers require an extra copy through a pinned staging buffer internally, roughly halving throughput.
2. **WDDM overhead**: The display driver model adds scheduling latency for memory operations.
3. **D2H is slower than H2D**: This asymmetry is common — D2H typically performs worse due to how PCIe and the GPU's copy engines interact with pageable memory.

**Action item**: A follow-up experiment should compare pinned (`cudaMallocHost`) vs pageable (`malloc`) transfer rates to isolate WDDM overhead from pageable memory overhead.

**Prediction check**: Predicted "significantly below theoretical." Confirmed.

#### 3. Kernel Bandwidth: EXCELLENT

Vector add at 64M elements: **1,483 GB/s** effective bandwidth. This is remarkable.

For context, the RTX PRO 6000 Blackwell with 512-bit bus at 14 GHz memory clock has a theoretical memory bandwidth of approximately:
- 512 bits * 14,001 MHz * 2 (DDR) / 8 bits/byte = ~1,792 GB/s

The measured 1,483 GB/s is **~83% of theoretical memory bandwidth**. This is an excellent result for a simple vector add kernel, indicating that the GPU's memory subsystem is operating near its limits even under WDDM.

Parallel reduction at 64M: **1,325 GB/s** (~74% of theoretical). Slightly lower due to the reduction pattern's less perfect memory access, but still strong.

**Prediction check**: Predicted "should approach memory bandwidth limits (~1 TB/s for Blackwell)." My estimate of 1 TB/s was wrong — actual theoretical is ~1.8 TB/s. But the measured results are even better than I expected at 1.3-1.5 TB/s. *Prediction exceeded.*

#### 4. Managed Memory: WORKS BUT SLOW, NO PREFETCH

- `cudaMallocManaged`: Works. But the allocation took **74.9 ms** — extremely slow compared to regular `cudaMalloc` (sub-millisecond at similar sizes). This is the WDDM tax on managed memory.
- `cudaMemPrefetchAsync`: **Not supported** ("invalid device ordinal"). This confirms the known WDDM limitation. Prefetch hints are ignored/rejected.
- Kernel on managed memory: **2.978 ms** vs 0.012 ms for the same workload on explicitly allocated memory — **248x slower**. This massive penalty is because without prefetch, the data must page-fault into GPU memory during kernel execution.

**Prediction check**: Predicted managed memory would work but prefetch would fail. Confirmed exactly.

#### 5. WDDM Impact Summary

| Metric | With WDDM | Estimated without WDDM (Linux/TCC) | Overhead |
|--------|-----------|-------------------------------------|----------|
| Kernel bandwidth | 1,483 GB/s | ~1,500-1,700 GB/s | Minimal (~0-10%) |
| H2D pageable | 16.55 GB/s | ~25-30 GB/s | ~40-50% |
| D2H pageable | 7.49 GB/s | ~15-20 GB/s | ~50-60% |
| Managed alloc | 74.9 ms | ~0.5-2 ms | ~37-150x |
| Managed kernel | 2.978 ms | ~0.05-0.1 ms | ~30-60x |
| Prefetch | Not supported | Supported | N/A |

**Key insight**: WDDM's impact is **not uniform**. Compute kernel execution is nearly unaffected — the GPU runs at full speed once a kernel is dispatched. The overhead concentrates in **memory management** (allocation, transfers, page migration). This has direct architectural implications: a WinRapids design should minimize memory management calls and maximize time spent in kernels.

### Conclusions

1. **CUDA works natively on Windows** with full kernel execution performance. Task #1 is proven.
2. **Kernel compute is not the bottleneck** — WDDM barely affects it. The GPU achieves 83% of theoretical memory bandwidth.
3. **Memory transfers are the bottleneck** — especially D2H, and especially managed memory.
4. **Managed memory is not viable for performance-critical paths** on WDDM. The 248x kernel penalty and missing prefetch make it unsuitable. Explicit memory management is required.
5. **Pinned memory transfers should be tested next** — the current pageable results may be inflating the WDDM overhead story. Pinned memory could close much of the gap.
6. **The 188-SM Blackwell is a computational beast** — 128 MB L2, 2.28 GHz, 1.8 TB/s theoretical bandwidth. The challenge is feeding it fast enough through WDDM's memory management layer.

### Cross-Run Variance

Pathmaker's initial run and Observer's independent run produced different numbers for the same binary:

| Metric | Pathmaker's run | Observer's run | Delta |
|--------|----------------|----------------|-------|
| Vector add 64M kernel | 0.480 ms (1,677 GB/s) | 0.543 ms (1,483 GB/s) | +13% slower |
| Reduction 64M kernel | 2.401 ms (112 GB/s) | 0.203 ms (1,326 GB/s) | **12x faster** |
| H2D 64M | 34.5 ms (15.5 GB/s) | 32.4 ms (16.6 GB/s) | 6% faster |
| D2H 64M | 39.8 ms (6.75 GB/s) | 35.8 ms (7.49 GB/s) | 10% faster |
| Managed alloc | 66.6 ms | 74.9 ms | +12% slower |
| Managed kernel | 2.576 ms | 2.978 ms | +16% slower |

**Notable**: The reduction 64M result differs dramatically — pathmaker's 2.401 ms vs observer's 0.203 ms is a 12x difference. Pathmaker's result (112 GB/s) seems anomalously slow for a reduction kernel that should be memory-bandwidth-limited. Observer's result (1,326 GB/s) is more consistent with expectations. Possible explanations:
- Different versions of the binary (pathmaker may have had a bug in an earlier compilation)
- WDDM scheduling jitter under different system loads
- The 100% GPU utilization at session start may have affected pathmaker's run more

**Methodological note**: Single-run measurements on WDDM are unreliable. Future benchmarks must use multiple runs with statistical reporting (min, median, mean, stddev, p99).

### CUDA 13 API Changes (from pathmaker's notes)

Worth documenting for future experiment design:
- `cudaDeviceProp.memoryClockRate` and `.clockRate` removed in CUDA 13 — must use `cudaDeviceGetAttribute()` instead
- `cudaMemPrefetchAsync` signature changed: now takes `cudaMemLocation` struct instead of raw `int device`

### Open Questions (carried forward)

1. What are pinned memory transfer rates? This is critical for isolating WDDM vs pageable overhead.
2. What is `cudaMalloc` allocation latency? (We only measured managed allocation.)
3. Can CUDA streams overlap transfers and compute under WDDM? (Concurrent kernels = yes, but what about H2D/compute/D2H overlap?)
4. What is the first-kernel-launch overhead? (The warm-up hid this.)
5. Is the 128 MB L2 cache effective at hiding WDDM memory transfer costs for workloads that fit in L2?
6. What causes the reduction 64M variance between runs? Need multi-run statistical benchmarks.
7. Should the expedition test in TCC mode? What is the WDDM vs TCC delta on this hardware?

---

## Entry 004 — Correction: TCC Mode IS Available + Power Clarification

**Date**: 2026-03-20
**Type**: Correction / fact-check
**Status**: Complete
**Triggered by**: Navigator challenge, naturalist observation

### Issue 1: TCC Availability

**Original claim (Entry 001, Finding #3)**: "TCC mode is not available for GeForce/RTX consumer/prosumer cards on Windows."

**This was wrong.** The RTX PRO 6000 is NOT a consumer card. It is a professional workstation card in the Quadro lineage (Product Brand: "NVIDIA RTX"). TCC mode is available for professional-tier NVIDIA cards on Windows.

**Verification**:
```
$ nvidia-smi -g 0 -dm 1
Set driver model to TCC for GPU 00000000:F1:00.0.
All done.
Reboot required.
```

The command succeeded (exit code 0). TCC mode is supported and can be enabled with a reboot. I immediately reverted:
```
$ nvidia-smi -g 0 -dm 0
Set driver model to WDDM for GPU 00000000:F1:00.0.
All done.
```

Confirmed: `driver_model.current = WDDM, driver_model.pending = WDDM` — no pending changes.

**Corrected understanding**: WDDM is the *current choice*, not an inherent hardware constraint. The expedition could switch to TCC mode at any time, though this would:
- Disable display output from this GPU (would need a separate GPU or integrated graphics for display)
- Require a reboot
- Remove the ~69 GB desktop VRAM overhead (freeing it for compute)
- Likely improve memory transfer rates and allocation latency
- Enable `cudaMemPrefetchAsync` for managed memory

**Implication for expedition framing**: The question shifts from "can we work around WDDM?" to "how much does WDDM cost, and is building WDDM-tolerant tools worthwhile vs just switching to TCC?" If the overhead is small enough, WDDM-native tools have broader applicability (most Windows users can't/won't switch to TCC). If the overhead is too large, TCC is an available escape hatch.

### Issue 2: Power / Max-Q Classification

**Question**: Is the 300W figure TDP or actual draw? Some RTX PRO 6000 specs suggest 600W.

**Answer from nvidia-smi -q**:

| Power Metric | Value |
|-------------|-------|
| Default Power Limit | 300.00 W |
| Current Power Limit | 300.00 W |
| Min Power Limit | 250.00 W |
| Max Power Limit | 325.00 W |
| Average Power Draw | ~299 W |
| Instantaneous Power Draw | ~293-302 W |

The 300W is the **TDP (default power limit)**, and the card is drawing very close to it (~299W average). The **Max-Q** designation in the product name confirms this is a power-limited variant designed for workstation form factors with constrained cooling. The non-Max-Q desktop variant likely has a significantly higher TDP (possibly 600W as suggested).

The 325W max power limit means there is only **25W of overclocking headroom** (8.3%). The card is essentially running at its power ceiling already.

**Note**: `SW Power Cap` was listed as "Active" in the GPU performance state, confirming the card is actively power-limited at 300W. This means compute-heavy workloads may be clock-throttled to stay within the power envelope.

### Impact on Previous Analysis

The WDDM Impact Summary table in Entry 003 remains valid — those are the measured numbers. But the framing changes:
- "Estimated without WDDM (Linux/TCC)" is now not just a theoretical comparison — it's **testable on this exact hardware** by switching to TCC mode.
- A WDDM vs TCC comparison on the same card would be a high-value experiment, giving us exact overhead numbers rather than estimates.

---

## Entry 005 — Cross-Reference: Naturalist's Landscape Survey

**Date**: 2026-03-20
**Type**: Fact-check / cross-reference
**Status**: Complete

### Purpose

The naturalist published a landscape survey in the expedition log with hardware specs and ecosystem claims. As observer, I'm cross-referencing these against verified data.

### Hardware Claims vs Verified Data

| Naturalist's Claim | Verified? | Actual |
|---------------------|-----------|--------|
| 24,064 CUDA cores | **Correct** | 188 SMs x 128 cores/SM (Blackwell) = 24,064 |
| 96 GB GDDR7 ECC memory | **Partially wrong** | 97,887 MiB (~95.6 GB). ECC hardware present but **disabled** (Current: Disabled, Pending: Disabled). Calling it "ECC memory" is misleading without noting it's off. |
| 1,792 GB/s memory bandwidth | **Correct** | Matches our theoretical calculation (512-bit x 14,001 MHz x 2 / 8) |
| 125 TFLOPS FP32 | **Unverified** | Cannot confirm from nvidia-smi or CUDA API. Plausible: 24,064 cores x 2,280 MHz x 2 (FMA) = ~109.7 TFLOPS. The 125 TFLOPS figure may assume boost clock. |
| 600W TDP | **Wrong for this SKU** | This is a Max-Q variant: 300W default, 325W max. See Entry 004. The 600W figure may apply to the non-Max-Q desktop variant. |
| MIG capable (up to 4 instances) | **Wrong for this SKU** | nvidia-smi reports MIG Mode: Current = N/A, Pending = N/A. "N/A" means not supported on this variant (vs "Disabled" which would mean supported but off). |
| TCC mode "should support" | **Confirmed** | Verified in Entry 004. TCC is available and the switch command succeeds. |

### Ecosystem Claims Assessment

| Claim | Assessment |
|-------|------------|
| cuDF doesn't run on Windows | **Plausible** — well-known limitation, not independently verified by us |
| GPUDirect Storage is Linux-only | **Plausible** — consistent with NVIDIA documentation |
| DirectStorage 1.4 has GPU Zstd decompression | **Plausible** — GDC 2026 announcement, not independently verified |
| No GPU data science toolkit on Windows | **Plausible** — the "empty quadrant" claim. Worth noting: individual GPU libraries (CuPy, Numba) do work on Windows, just not the full DataFrame stack |
| Polars GPU engine uses cuDF | **Plausible** — consistent with Polars documentation |
| Arrow C Device Data Interface works on Windows | **Plausible** — Arrow is cross-platform, but we haven't tested this yet (Task #4) |
| WDDM overhead "up to 2x" | **Our data suggests this is overstated for compute.** Kernel bandwidth showed <10% overhead. Memory management overhead is severe (30-250x for managed memory), but raw compute is nearly unaffected. The "2x" claim may apply to workloads dominated by kernel launch overhead (many small kernels), not memory-bandwidth-limited workloads. |

### Corrections Needed in Expedition Log

The naturalist's expedition log contains factual errors that should be corrected:
1. **600W TDP** — should be 300W (Max-Q variant). The expedition log mentions "600W TDP" in the hardware specs but later correctly identifies the Max-Q variant at 300W.
2. **MIG capable** — should note MIG is not supported on this specific SKU.
3. **ECC** — should note ECC is available but currently disabled.
4. **TCC "Maybe"** in dependency map — should be updated to "Yes" (confirmed in Entry 004).
5. **"96 GB"** vs actual ~95.6 GB — minor discrepancy, likely due to reporting vs marketing numbers.

### Note on the "Empty Quadrant" Thesis

The naturalist's observation that Windows-native GPU data science is an empty quadrant is the strongest claim in the survey. It's worth testing rigorously: are there any partial solutions we're missing? CuPy works on Windows and provides GPU arrays. Numba CUDA works on Windows. These aren't DataFrames, but they're not nothing. The quadrant may be "nearly empty" rather than "completely empty."

However, the core thesis holds: there is no GPU DataFrame library that runs natively on Windows. That gap is real and significant.

---

## Entry 006 — Experiment 002 Results: GPU Memory Management

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete (Tests 1-4 complete; Tests 5-6 lost due to process crash during VRAM capacity test)

### Experiment Design

Pathmaker's `experiments/002-gpu-memory/gpu_memory.cu` tests six aspects of GPU memory management on WDDM:
1. cudaMalloc/cudaFree latency across sizes (1 KB to 1 GB)
2. CUDA memory pools (cudaMallocAsync/cudaFreeAsync)
3. Pinned vs pageable memory transfer performance
4. VRAM capacity probe under WDDM
5. Async transfer + compute overlap (NOT RUN -- process crashed)
6. Simple pool allocator vs raw cudaMalloc (NOT RUN -- process crashed)

**Methodological improvement over Experiment 001**: Uses multiple trials with statistical reporting (min, median, mean, p99, max). This addresses the single-run reliability concern from Entry 003.

### Results

#### Test 1: cudaMalloc/cudaFree Latency

| Size | cudaMalloc median (us) | cudaFree median (us) | Total cycle (us) |
|------|----------------------|---------------------|-------------------|
| 1 KB | 44 | 176 | 220 |
| 64 KB | 39 | 171 | 210 |
| 1 MB | 31 | 166 | 197 |
| 16 MB | 62 | 412 | 474 |
| 64 MB | 113 | 1,377 | 1,490 |
| 256 MB | 344 | 5,014 | 5,358 |
| 1 GB | 1,291 | 16,393 | 17,684 |

**Key observations**:
- **cudaFree is much slower than cudaMalloc** — consistently 3-15x slower. At 1 GB, free takes 16.4 ms vs alloc at 1.3 ms. This asymmetry is significant: WDDM imposes heavy overhead on deallocation, likely because it must coordinate with the display driver to release GPU virtual address space.
- **Allocation is fast up to 1 MB** (~30-44 us), then scales roughly linearly with size.
- **The p99 tail is brutal for 1 GB frees**: mean = 16.4 ms but first run's p99 was 5.9 *seconds*. One outlier took ~6 seconds to free 1 GB. This suggests occasional WDDM scheduling stalls.
- **Compare to Experiment 001's cudaMallocManaged**: managed alloc was 66-75 ms for 4 MB. Regular cudaMalloc for 1 MB is 31 us — **over 2,000x faster**. Managed memory overhead is not just prefetch; the allocation itself is enormously slower.

#### Test 2: CUDA Memory Pools

**Memory pools are SUPPORTED on WDDM.** This is a critical finding.

| Size | cudaMallocAsync median (us) | cudaFreeAsync median (us) |
|------|---------------------------|--------------------------|
| 1 KB | 0.6 | 0.5 |
| 64 KB | 0.5 | 0.5 |
| 1 MB | 0.5 | 0.5 |
| 16 MB | 0.5 | 0.5 |
| 64 MB | 0.5 | 0.5 |
| 256 MB | 0.6 | 0.5 |

**All sizes: ~0.5 us.** Pool-based allocation is essentially free — it's a pointer return from a cached pool, no WDDM interaction needed.

**Rapid alloc/free cycle (1 MB, 100 iterations)**:
- Raw cudaMalloc+Free: median = 310 us, p99 = 915 us
- cudaMallocAsync+Free: median = 0.9 us, p99 = 1.2 us
- **Speedup: ~344x**

**This is the single most important finding of the experiment.** CUDA memory pools completely bypass WDDM's allocation overhead. For WinRapids, this means:
- Use `cudaMallocAsync`/`cudaFreeAsync` for all GPU memory operations
- The WDDM allocation tax is avoidable without building a custom pool allocator
- NVIDIA's built-in pool is better than any custom solution we could build

#### Test 3: Pinned vs Pageable Memory Transfer

| Size | Pageable H2D (GB/s) | Pinned H2D (GB/s) | H2D Speedup | Pageable D2H (GB/s) | Pinned D2H (GB/s) | D2H Speedup |
|------|---------------------|-------------------|-------------|---------------------|-------------------|-------------|
| 1 MB | 22.1 | 40.5 | 1.83x | 9.1 | 40.8 | 4.46x |
| 16 MB | 22.2 | 56.7 | 2.55x | 20.9 | 56.2 | 2.69x |
| 64 MB | 26.3 | 57.5 | 2.18x | 23.9 | 56.9 | 2.38x |
| 256 MB | 24.7 | 57.6 | 2.33x | 24.3 | 55.7 | 2.29x |

**Key observations**:
- **Pinned memory achieves 55-58 GB/s** — approaching PCIe 5.0 x16 theoretical (~63 GB/s). This is ~90% of theoretical. WDDM does NOT significantly degrade pinned memory transfer rates.
- **Pageable D2H is severely penalized at small sizes**: 9.1 GB/s for 1 MB D2H (pageable) vs 40.8 GB/s (pinned) = 4.46x slower. The asymmetry from Experiment 001 was entirely a pageable memory artifact, NOT a WDDM limitation.
- **At large sizes (256 MB), pageable reaches ~24 GB/s** both directions — the H2D/D2H asymmetry largely disappears. Pageable is still 2.3x slower than pinned.
- **Pinned H2D and D2H are symmetric**: ~56-58 GB/s both ways. This is excellent — no directional penalty under WDDM with pinned memory.

**This resolves Open Question #1 from Entry 003.** Pinned memory is the answer. The Experiment 001 transfer rates (16.6 GB/s H2D, 7.5 GB/s D2H) were pageable memory artifacts. With pinned memory, transfers are fast and symmetric.

#### Test 4: VRAM Capacity Under WDDM

| Metric | Value |
|--------|-------|
| Total VRAM (reported) | 95.59 GB |
| Free VRAM (at test start) | 93.55 GB |
| WDDM+OS overhead | 2.05 GB |
| Maximum allocated | **186+ GB** (process crashed during allocation) |

**This is a surprising and important finding:**
- **WDDM+OS used only 2.05 GB** at the time of this test. Compare to Entry 001's observation of ~69 GB used. The difference: Entry 001 measured via nvidia-smi during heavy desktop use; this test measured via CUDA API after the heavy desktop processes had finished or reduced their footprint.
- **CUDA allocated 186+ GB on a 95.6 GB GPU.** WDDM enabled virtual memory paging — GPU allocations were backed by system RAM when VRAM ran out. The fill_kernel succeeded on all 186+ chunks, meaning the GPU could actually execute kernels on paged-out memory (though performance would degrade severely).
- **The process crashed** after allocating 186+ GB, likely due to system memory exhaustion (virtual memory + physical RAM exceeded). Tests 5 and 6 never ran.

**Implication**: Under WDDM, `cudaMalloc` can succeed even when physical VRAM is exhausted. This is a **double-edged sword**:
- **Good**: Graceful degradation — workloads don't crash when they exceed VRAM.
- **Bad**: Silent performance cliffs — code that appears to work fine may be hitting system RAM at orders-of-magnitude slower speeds. A robust WinRapids allocator must check `cudaMemGetInfo()` to stay within physical VRAM.

### Analysis: Revising the WDDM Overhead Story

The data from Experiment 002 fundamentally changes our understanding of WDDM's impact:

| Factor | Previously Assumed | Now Known |
|--------|-------------------|-----------|
| Allocation overhead | Severe (managed = 66-75 ms) | **Avoidable** — pools give 0.5 us |
| Transfer rates | 16.6/7.5 GB/s (pageable) | **55-58 GB/s with pinned** (90% of PCIe theoretical) |
| H2D/D2H asymmetry | Severe (2.2x) | **Artifact of pageable memory** — pinned is symmetric |
| Kernel compute | 83% of theoretical | Unchanged — still excellent |
| VRAM availability | ~29 GB (69 GB used by OS) | **93.6 GB free** (2 GB OS) — prior measurement was during heavy desktop use |
| Memory pools | Unknown | **Supported and effectively free** |
| Over-allocation | Unknown | **WDDM silently pages to system RAM** — dangerous performance cliff |

**Revised WDDM impact statement**: When using best practices (memory pools for allocation, pinned memory for transfers), WDDM overhead is **minimal** for the core compute pipeline:
- Allocations: <1 us (pool) vs ~30-1300 us (raw)
- Transfers: 55-58 GB/s (90% of PCIe 5.0 theoretical)
- Kernel compute: 83-93% of memory bandwidth theoretical

The WDDM "tax" is real but largely avoidable with the right APIs. The remaining overhead vs TCC/Linux is likely in the single-digit percentage range for well-structured workloads. The dangerous part is the *silent* issues: over-allocation paging and occasional p99 stalls.

### Conclusions

1. **CUDA memory pools are the architecture decision for WinRapids.** 344x faster than raw cudaMalloc/cudaFree. This eliminates WDDM allocation overhead entirely.
2. **Pinned memory is mandatory for transfers.** 2-4.5x faster than pageable. Achieves 90% of PCIe theoretical. Symmetric H2D/D2H.
3. **cudaFree is the expensive operation, not cudaMalloc.** The allocator should avoid freeing memory — use pools instead.
4. **VRAM virtual memory paging under WDDM is a hidden danger.** WinRapids must enforce physical VRAM limits to avoid silent performance cliffs.
5. **Tests 5 and 6 recovered** — pathmaker re-ran with the VRAM capacity test capped. Results below.

#### Test 5: Async Transfer + Compute Overlap (recovered from pathmaker's run)

| Metric | Value |
|--------|-------|
| Async engine count | 1 |
| Sequential time (compute then copy) | 6.325 ms |
| Overlapped time (compute + copy) | 1.269 ms |
| Overlap benefit | **79.9%** |

**Compute-copy overlap works on WDDM.** Even with a single async copy engine, overlapping a kernel with a D2H transfer yields 80% speedup. This validates a pipelined architecture where the next batch transfers while the current batch computes.

**Note**: Only 1 async engine is available (vs 2+ on some data center GPUs). This means we can overlap one copy direction with compute, but not H2D + compute + D2H simultaneously. A three-stage pipeline would need to serialize the two copy directions.

#### Test 6: Pool Allocator Comparison (recovered from pathmaker's run)

| Method | Median (4 MB alloc/free, 200 cycles) | Speedup vs raw |
|--------|--------------------------------------|----------------|
| Raw cudaMalloc+cudaFree | 281 us | 1x |
| Simple pool (pointer reuse) | ~0 us | infinite |
| CUDA cudaMallocAsync+cudaFreeAsync | 1.4 us | 200x |

**The trivial 30-line pool allocator beats CUDA's built-in pools** for the exact-match reuse case (same size requested as previously freed). This is because it avoids even stream synchronization overhead. However, CUDA's pools handle size mismatches, multi-stream allocation, and thread safety — all things the simple pool doesn't.

**Implication for WinRapids**: A two-tier allocator strategy:
- **Hot path** (same-size column reuse): Simple pointer-reuse cache (~0 us)
- **General case**: CUDA memory pools (~1.4 us)
- **Never**: Raw cudaMalloc/cudaFree in any performance-sensitive path

### Open Questions (updated)

1. ~~What are pinned memory transfer rates?~~ **Resolved**: 55-58 GB/s, 90% of theoretical.
2. ~~What is cudaMalloc allocation latency?~~ **Resolved**: 30-1300 us depending on size. Avoidable with pools (0.5 us).
3. ~~Can CUDA streams overlap transfers and compute under WDDM?~~ **Resolved**: Yes, 80% benefit with 1 async engine. Pipelined architecture viable.
4. What is the first-kernel-launch overhead? Still unanswered.
5. ~~Is the 128 MB L2 cache effective at hiding WDDM memory transfer costs?~~ Partially answered — transfers are fast enough with pinned memory that L2 caching is less critical for hiding latency.
6. What is the performance cliff when WDDM pages GPU memory to system RAM? How to detect/prevent it?
7. What is the WDDM p99 tail behavior under sustained load? (The 6-second cudaFree outlier is concerning.)

---

## Entry 007 — Experiment 003 Results: Arrow GPU Integration

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: Arrow GPU integration works on Windows via CuPy. DLPack is the GPU interchange protocol.

### Environment Update

Pathmaker installed new Python packages for this experiment:
- **CuPy 14.0.1** (CUDA 13.x wheel)
- **PyArrow 23.0.1**

These join the existing NumPy 2.4.2 and pandas 3.0.1.

### What Was Tested

| Test | Purpose |
|------|---------|
| 1. CuPy <-> Arrow roundtrip | GPU -> CPU -> Arrow -> CPU -> GPU with timing |
| 2. DLPack protocol | GPU-to-GPU zero-copy interchange |
| 3. Arrow Device Array concept | Demonstrate CPU metadata + GPU data split |
| 4. Pinned memory for Arrow | CuPy pinned alloc + transfer rates |
| 5. Arrow -> GPU kernel | Custom CUDA kernel on Arrow-originated data |
| 6. Arrow IPC -> GPU pipeline | End-to-end: IPC deserialize -> multi-column GPU transfer -> filtered aggregation |

### Results Summary (from pathmaker's README)

| Finding | Data |
|---------|------|
| Arrow <-> numpy zero-copy | Yes (float64, after first-call warmup) |
| DLPack GPU-to-GPU | 3-35 us for any size, same-pointer exchange |
| ArrowDeviceArray metadata | CPU-readable schema without GPU access |
| Pinned H2D via CuPy | 32-51 GB/s |
| Custom kernel on Arrow data | Works, zero error |
| IPC -> GPU pipeline | 0.2 ms deserialize, 11.6 ms transfer (5M rows, 3 cols) |
| CuPy fancy indexing overhead | 1750 ms for filtered sum — not representative of optimized kernels |

### Analysis

#### 1. The Arrow Path on Windows

**pyarrow.cuda does NOT exist on Windows pip wheels.** This was confirmed by pathmaker. The working path is:
```
Arrow IPC -> numpy (zero-copy) -> CuPy (H2D transfer) -> GPU computation
```
This is more manual than the Linux path (where pyarrow.cuda can wrap GPU buffers directly), but it works and the Arrow-to-numpy step is zero-copy for compatible types.

#### 2. DLPack is the Answer for GPU Interchange

DLPack exchange: ~3-35 us regardless of data size (100M elements = 400 MB in 17 us total). This is pure pointer exchange — no data movement. For WinRapids interop with PyTorch, JAX, or other GPU libraries, DLPack eliminates the need for any custom interchange layer.

#### 3. The Co-Native Split is Validated

The Arrow C Device Data Interface naturally separates:
- **CPU-resident metadata**: schema name, type, length, null_count, device_type, device_id
- **GPU-resident data**: buffer pointers only

An AI agent or human user can introspect a GPU DataFrame's structure without touching GPU memory. This is exactly the co-native design principle — both kinds of agents read natively from their own domain.

#### 4. CuPy as GPU Memory Manager

CuPy provides the Python-level interface to CUDA memory management. Key capabilities confirmed working on Windows:
- `cp.cuda.alloc_pinned_memory()` for pinned host staging
- `cp.RawKernel()` for custom CUDA kernel compilation and execution
- `cp.asarray()` / `cp.asnumpy()` for H2D/D2H transfers
- `__dlpack__()` / `from_dlpack()` for GPU zero-copy interchange

**Concern**: CuPy's high-level fancy indexing (boolean masking + sum) took 1750 ms for 5M elements — far slower than expected. Custom CUDA kernels via RawKernel are much faster. For WinRapids performance-critical operations, CuPy should be used as a memory manager and kernel launcher, not for its high-level array operations.

#### 5. Architecture Implications

The data path for WinRapids ingestion is now clear:

```
File (Parquet/CSV/Arrow IPC)
  -> PyArrow deserialization (sub-ms)
  -> numpy view (zero-copy for compatible types)
  -> pinned host staging buffer (cudaHostAlloc)
  -> async H2D transfer (55-58 GB/s)
  -> GPU column buffer (cudaMallocAsync from pool)
  -> ArrowDeviceArray metadata struct (CPU-resident)
```

Each step is either zero-copy or near-hardware-limited bandwidth. The only translation step is numpy -> GPU, which is a raw memcpy at PCIe speeds.

### Conclusions

1. **Arrow GPU works on Windows** without pyarrow.cuda. The numpy zero-copy path is adequate.
2. **DLPack is mandatory** for GPU-to-GPU interchange. Zero overhead.
3. **CuPy is the right Python GPU backend** for WinRapids — it provides memory management, kernel compilation, and DLPack support.
4. **Custom CUDA kernels needed** for performance-critical DataFrame ops. CuPy's high-level ops have too much overhead.
5. **The ingestion pipeline is viable**: Arrow IPC -> numpy -> pinned -> GPU at 55+ GB/s with sub-ms parsing overhead.

### Open Questions

1. Can we implement the full Arrow C Device Data Interface in pure Python/ctypes for zero-dependency GPU DataFrames?
2. What is the JIT compilation overhead for CuPy RawKernel on first call? (Affects cold-start latency.)
3. How does CuPy's memory pool interact with the CUDA memory pool we validated in Experiment 002?

---

## Entry 008 — Experiment 004 Results: Minimal GPU DataFrame

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: 50-200x speedup over pandas validated. The project is justified.

### What Was Built

Pathmaker created `GpuColumn` and `GpuFrame` classes (~200 lines Python) implementing:
- CuPy-backed device arrays
- CPU-resident metadata (name, dtype, length, location) — the co-native split
- Arrow import/export via zero-copy numpy bridge
- Explicit `MemLocation` enum (DEVICE / HOST_PINNED / HOST)
- Aggregations: sum, mean, min, max, std
- Element-wise: ==, >, <, +, -, *
- Filtered aggregation: `filtered_sum(mask)`
- Memory map visualization

### Benchmark Results (10M rows, float64, averaged over 5 runs)

#### Aggregations

| Operation | pandas (ms) | GPU (ms) | Speedup |
|-----------|-------------|----------|---------|
| sum | 8.3 | 0.12 | **71x** |
| mean | 11.8 | 0.10 | **124x** |
| min | 8.8 | 0.13 | **70x** |
| max | 9.2 | 0.13 | **69x** |
| std | 55.9 | 5.0 | **11x** |

#### Column Arithmetic: a * b + c

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| pandas | 29.7 | 1x |
| GpuFrame | 0.56 | **53x** |

#### Filtered Sum: sum(values) where flag == 1

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| pandas | 34.8 | 1x |
| GpuFrame | 0.38 | **92x** |
| Raw CuPy | 0.44 | 80x |
| Custom CUDA kernel | 0.17 | **201x** |

#### Arrow Roundtrip (1M rows, 2 columns)

| Direction | Time (ms) |
|-----------|-----------|
| Arrow -> GPU | 2.2 |
| GPU -> Arrow | 2.6 |
| Total | 4.8 |

### Analysis

#### 1. Performance Validation

The speedups are genuine and significant:
- **Simple reductions (sum, mean, min, max)**: 69-124x faster. These are memory-bandwidth-limited on both CPU and GPU, so the speedup reflects the GPU's ~1.8 TB/s vs CPU's ~50 GB/s.
- **std is only 11x faster**: This involves two passes (mean then variance), so the GPU's kernel launch overhead becomes proportionally larger. Still significant.
- **Arithmetic (53x)**: The `a * b + c` operation creates intermediate CuPy arrays. A fused kernel would be faster.
- **Filtered sum**: GpuFrame (0.38 ms) is actually *faster* than raw CuPy (0.44 ms) — likely within noise, but confirms the abstraction adds negligible overhead.
- **Custom kernel (201x)**: A hand-written fused filter+reduce kernel doubles the CuPy speedup. This validates the need for custom kernels in performance-critical paths.

#### 2. GpuFrame Overhead Assessment

**GpuFrame adds zero measurable overhead over raw CuPy.** The filtered sum benchmark is the clearest evidence: GpuFrame 0.38 ms vs raw CuPy 0.44 ms. The Python-level column lookup and wrapper creation are negligible compared to GPU kernel time.

This is architecturally important: the abstraction layer is essentially free. There's no performance penalty for using GpuFrame vs raw CuPy arrays.

#### 3. The Co-Native Split in Practice

The `memory_map()` output demonstrates the co-native principle:
```
GpuFrame: 5,000,000 rows x 4 columns
  id                    int64           40.0 MB  [gpu]
  price                 float64         40.0 MB  [gpu]
  volume                int32           20.0 MB  [gpu]
  flag                  int8             5.0 MB  [gpu]
  Total: 105.0 MB on GPU
```

An AI agent can read column names, types, sizes, and locations without any GPU access. A human sees the same information in a readable format. The `[gpu]` tag makes residency explicit — no hidden data movement.

#### 4. What's Missing (from README)

The prototype lacks: null/validity bitmaps, string columns, GroupBy, Join/merge, Sort, file format readers, memory pool integration, and multi-GPU support. These are the features that separate a proof-of-concept from a usable library.

**Critical gap**: No null handling. Real-world data has nulls, and Arrow's validity bitmap is the standard way to represent them. Adding bitmap support will require changes to every operation.

### Methodological Notes

- Benchmarks use 5-run averages with warmup — adequate for demonstrating magnitude of speedup, but not precise enough for micro-benchmarks.
- All comparisons are **data-on-GPU vs data-on-CPU**. The H2D transfer cost is excluded. For a fair end-to-end comparison (data starts on disk), the 2.2 ms Arrow->GPU cost must be included.
- The `std` benchmark (11x) is the weakest result but also the most realistic — multi-pass operations expose per-kernel overhead.
- No memory pool integration yet — CuPy's default allocator is used, not the CUDA async pool we validated in Experiment 002.

### Conclusions

1. **The project is justified.** 50-200x speedups on 10M row operations, with zero abstraction overhead.
2. **The architecture is sound.** Co-native metadata + GPU buffers + Arrow compatibility + explicit location tagging.
3. **Custom CUDA kernels are the performance ceiling.** CuPy high-level ops are 50-100x vs pandas; hand-written kernels are 200x.
4. **Next priorities**: null support, memory pool integration, GroupBy, and file format readers (Parquet).

---

## Entry 009 — Experiment 005 Code Review: Pandas GPU Proxy (pre-execution)

**Date**: 2026-03-20
**Type**: Code review / methodology assessment
**Status**: Awaiting execution results

### Experiment Design

Pathmaker created `experiments/005-pandas-proxy/pandas_proxy.py` implementing a proxy pattern that wraps pandas DataFrames with transparent GPU acceleration via CuPy. Two classes: `GpuAcceleratedSeries` and `GpuAcceleratedDataFrame`.

### Architecture Analysis

**Core idea**: Wrap pandas objects so existing code runs unchanged but numeric operations dispatch to GPU. Unknown operations fall back to pandas via `__getattr__`.

**Key design decisions**:

1. **Lazy GPU transfer**: `_ensure_gpu()` only transfers to GPU on first GPU operation, not at wrap time. Good — avoids paying H2D cost for columns that are never computed on.

2. **CPU as source of truth**: The pandas Series is always maintained as the canonical representation. GPU results are materialized back to CPU (`cp.asnumpy`) for every operation. This preserves pandas compatibility but **forces a D2H round-trip per operation**.

3. **Fallback logging**: Every CPU fallback is logged in `_fallback_log`. This is useful for understanding which operations are GPU-accelerated and which aren't — a form of observability.

4. **`__getattr__` delegation**: Unknown methods are forwarded to the underlying pandas object, with results re-wrapped as proxies. This provides pandas API coverage without implementing every method.

### Critical Observation: The Round-Trip Problem

The proxy materializes GPU results back to CPU after every arithmetic operation. For `gdf["a"] * gdf["b"] + gdf["a"]`:
1. `gdf["a"]._ensure_gpu()` — H2D transfer
2. `gdf["b"]._ensure_gpu()` — H2D transfer
3. `gpu_a * gpu_b` — GPU compute
4. `cp.asnumpy(result)` — **D2H transfer**
5. Wrap in new `pd.Series` + `GpuAcceleratedSeries`
6. New series `._ensure_gpu()` — **H2D transfer again**
7. `+ gdf["a"]` — GPU compute
8. `cp.asnumpy(result)` — **D2H transfer**

That's **3 GPU round-trips** for a 2-operation expression. Compare to GpuFrame (Experiment 004) which stays on GPU between operations: 0 round-trips for the same expression.

**Prediction**: The proxy will show significantly less speedup than GpuFrame for chained operations (arithmetic). For simple aggregations (single `.sum()`), it should match GpuFrame since there's only one GPU call.

### What to Watch For

1. **Aggregation speedup**: Should be similar to GpuFrame (50-100x) since aggregations are single GPU calls.
2. **Arithmetic speedup**: Will be degraded by D2H round-trips. Predicting <10x for chained operations.
3. **Filtered sum speedup**: The proxy code does `gdf["value"][mask].sum()` which involves boolean indexing on CPU (the mask is materialized) — predicting poor performance.
4. **Correctness**: All results should match pandas exactly.
5. **Fallback log**: What operations fall back to CPU?

---

## Entry 010 — Experiment 005 Results: Pandas GPU Proxy Pattern

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: Proxy works for simple aggregations (61-103x). Fails for chained operations (0.5x — slower than pandas). The proxy pattern is a dead end for complex workflows.

### Independent Execution

Ran `experiments/005-pandas-proxy/pandas_proxy.py` independently. All correctness tests PASS.

### Benchmark Results (10M rows, float64)

#### Observer's Independent Run

| Operation | pandas (ms) | Proxy (ms) | Speedup |
|-----------|-------------|------------|---------|
| sum | 7.7 | 0.12 | **64.7x** |
| mean | 11.5 | 0.12 | **97.9x** |
| filtered sum | 29.2 | 56.9 | **0.5x (SLOWER)** |

#### Pathmaker's Reported Results (from README)

| Operation | pandas (ms) | Proxy (ms) | Speedup |
|-----------|-------------|------------|---------|
| sum | 7.9 | 0.13 | 61x |
| mean | 12.0 | 0.12 | 103x |
| filtered sum | 30.2 | 59.2 | 0.5x |

#### Cross-Run Comparison

| Operation | Observer Speedup | Pathmaker Speedup | Agreement |
|-----------|-----------------|-------------------|-----------|
| sum | 64.7x | 61x | Within noise (~6% variance) |
| mean | 97.9x | 103x | Within noise (~5% variance) |
| filtered sum | 0.5x | 0.5x | Exact match — both SLOWER than pandas |

**Reproducibility**: Excellent. Both runs agree within single-digit percentage variance on all operations. The filtered sum degradation is robust — not an outlier.

### Predictions from Entry 009 vs Actual Results

| Prediction | Actual | Verdict |
|-----------|--------|---------|
| Aggregation speedup should match GpuFrame (50-100x) | sum=65x, mean=98x | **CONFIRMED** — matches GpuFrame (71x, 124x) within expected variance |
| Arithmetic speedup will be <10x due to D2H round-trips | Not benchmarked by pathmaker | **UNTESTED** — the benchmark suite doesn't include arithmetic timing |
| Filtered sum will show poor performance | 0.5x (slower than pandas) | **CONFIRMED** — even worse than predicted |
| Correctness should match pandas exactly | All tests PASS | **CONFIRMED** |

### Analysis

#### 1. Why Aggregations Work

For `gdf["value"].sum()`, the proxy executes:
1. `_ensure_gpu()` — one H2D transfer (cached after first call)
2. `cp.sum(gpu)` — one GPU reduction kernel
3. `float()` — scalar extraction

Only one GPU call, no intermediate materialization. The proxy adds negligible Python overhead over the raw CuPy call. The 65-98x speedups are consistent with GpuFrame's 71-124x (Entry 008), confirming the GPU compute advantage is real and the proxy preserves it.

#### 2. Why Filtered Sum Fails

The proxy's filtered sum executes `gdf["value"][mask].sum()`:
1. `gdf["flag"] == 1` — GPU comparison, then `cp.asnumpy()` **D2H transfer** of 10M booleans
2. Wrap result as new `GpuAcceleratedSeries` with pandas boolean Series
3. `gdf["value"][mask]` — calls `__getitem__` which unwraps to `self._series[key._series]` — **pandas boolean indexing on CPU**
4. Wrap result as new `GpuAcceleratedSeries` (creates new pandas Series)
5. `.sum()` — calls `_ensure_gpu()` **H2D transfer** of filtered values, then GPU sum

Data path: GPU -> CPU (mask) -> CPU indexing -> CPU (subset) -> GPU (subset) -> GPU sum -> scalar

Compare to GpuFrame (Entry 008): GPU comparison -> GPU boolean indexing -> GPU sum. Zero CPU involvement. Result: 0.38 ms (92x faster than pandas) vs proxy's 56.9 ms (2x slower than pandas).

**The proxy's filtered sum is 150x slower than GpuFrame's filtered sum** (56.9 ms vs 0.38 ms). The entire GPU advantage is destroyed by the D2H/H2D round-trips plus CPU-side pandas indexing on 10M rows.

#### 3. The Architecture Lesson

The proxy pattern has a fundamental impedance mismatch:

- **Source of truth is CPU** (pandas Series always maintained)
- **GPU is an accelerator for individual ops** (not a residence for data)
- **Every result materializes to CPU** before the next operation can begin

This works when operations are independent (each aggregation stands alone). It fails when operations compose (the output of one operation feeds the input of the next), because each composition boundary crosses PCIe twice.

The fix is exactly what the pathmaker concluded: the proxy needs a GPU DataFrame underneath (like GpuFrame) so intermediate results stay on GPU. This is precisely what cudf.pandas does — it's not a thin proxy over pandas, it's a full GPU DataFrame with a pandas-compatible API and CPU fallback.

#### 4. Missing Benchmark: Arithmetic

The pathmaker's benchmark suite doesn't include a timed arithmetic test (e.g., `gdf["a"] * gdf["b"] + gdf["a"]`). The correctness test exercises arithmetic but doesn't time it. From Entry 009's code review, we predicted <10x speedup due to D2H round-trips per arithmetic operation. This prediction remains untested.

**Recommendation**: A future benchmark should time `gdf["a"] * gdf["b"] + gdf["a"]` as proxy vs pandas to quantify the round-trip degradation for chained arithmetic. Based on the code analysis (3 GPU round-trips for a 2-op expression), I expect the overhead to be significant but less catastrophic than filtered sum, since the arithmetic D2H transfer is just raw array data without the additional pandas boolean indexing overhead.

### Comparison: Proxy vs GpuFrame vs Pandas

| Operation | pandas (ms) | GpuFrame (ms) | Proxy (ms) | GpuFrame vs pandas | Proxy vs pandas |
|-----------|-------------|---------------|------------|--------------------|-----------------|
| sum (10M) | 8.0 | 0.12 | 0.12 | 71x | 65x |
| mean (10M) | 11.9 | 0.10 | 0.12 | 124x | 98x |
| filtered sum (10M) | 32.5 | 0.38 | 57.0 | **92x** | **0.5x** |

The proxy matches GpuFrame for simple aggregations but is **184x slower than GpuFrame for filtered sum** (and 2x slower than unaccelerated pandas). The two approaches diverge exactly where the round-trip problem kicks in.

### Conclusions

1. **The proxy pattern is validated for simple aggregations.** 61-98x speedup with one line of code change. This is a real, useful accelerator for `df.sum()`, `df.mean()`, etc.

2. **The proxy pattern is a dead end for complex workflows.** Any operation that chains GPU results through CPU intermediates pays the PCIe tax per step. Filtered sum is 2x *slower* than pandas because the overhead exceeds the GPU compute benefit.

3. **The round-trip problem is the design bottleneck, not WDDM.** This failure would occur on Linux/TCC too — it's an architectural issue (CPU source of truth), not a platform issue.

4. **GpuFrame is the correct foundation.** The Experiment 004 GpuFrame keeps data on GPU by default and avoids materialization between chained operations. It achieves 92x on filtered sum where the proxy achieves 0.5x.

5. **Path forward for pandas compatibility**: Build a pandas-compatible API on top of GpuFrame, not as a wrapper around pandas. The proxy should be the fallback mechanism (for ops not yet implemented on GPU), not the primary execution path.

### Open Questions

1. What is the arithmetic speedup for chained proxy operations? (Untested — predicted <10x based on code analysis.)
2. How many operations must be chained before the proxy becomes slower than pandas? (Filtered sum crosses at 2 operations.)
3. Can a "lazy evaluation" approach (defer materialization until result is consumed) save the proxy pattern? This is essentially what cudf.pandas does with its deferred execution mode.

---

## Entry 011 — Experiment 006 Results: Co-Native Tiered Data Structures

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: The tiered architecture works. Co-native metadata enables query planning without GPU access. Even worst-case (all-CPU) is 4x faster than pandas. But cold-start artifacts need investigation.

### What Was Built

Pathmaker created `TieredColumn` and `TieredFrame` classes (~650 lines Python) implementing:
- Four memory tiers: Device (GPU), Pinned, Pageable, Storage (via `Tier` enum)
- Explicit promotion between tiers with measured timing
- Memory map visualization showing per-column tier, size, promotion cost, access count
- Query planner that estimates transfer costs before execution
- Smart filtered sum that uses the query plan to minimize transfers
- Arrow device type mapping (`ARROW_DEVICE_CUDA`, `ARROW_DEVICE_CUDA_HOST`, `ARROW_DEVICE_CPU`)

This is a direct evolution of Experiment 004's `GpuFrame`, adding the tiered residency model and query planner.

### Independent Execution

Ran `experiments/006-co-native/co_native.py` independently. All tests PASS. All computed results match pandas exactly (340.452472).

### Results Comparison

#### Tier Transfer Costs (8 MB column)

| Transfer | Estimated (ms) | Observer Actual (ms) | Pathmaker Actual (ms) |
|----------|---------------|---------------------|----------------------|
| GPU -> GPU | 0.0 | 0.0 | 0.0 |
| Pinned -> GPU | 0.1 | ~0.2 | ~0.2 |
| CPU -> GPU | 0.3 | 0.73 | 0.77 |

The estimates are conservative lower bounds. Actual CPU->GPU transfer for 8 MB at 0.73 ms implies ~10.9 GB/s effective bandwidth — well below the 25 GB/s pageable or 57 GB/s pinned rates from Experiment 002. This is expected: CuPy's `cp.asarray()` includes Python overhead, numpy array wrapping, and CUDA context overhead beyond the raw `cudaMemcpy`.

#### Filtered Sum Benchmark (10M rows)

| Scenario | Observer (ms) | Pathmaker (ms) | vs pandas |
|----------|--------------|----------------|-----------|
| Scenario 1: All on GPU (first run) | **157.3** | 0.33 | see analysis |
| Scenario 2: Value GPU, flag pinned | 1.56 | 1.6 | 18x |
| Scenario 3: All on CPU | 6.69 | 7.0 | 4x |
| Scenario 4: Re-run (data on GPU) | 0.33 | 0.33 | 87x |
| pandas baseline | 28.7 | 29.1 | 1x |

### Analysis

#### 1. The Scenario 1 Cold-Start Anomaly

**This is the most important finding of my independent run.** Pathmaker reported Scenario 1 (all on GPU) at 0.33 ms. I observed **157 ms** — a 476x discrepancy.

Breakdown of my Scenario 1:
- Promotion: 0.99 ms (expected — data already on GPU, so this is just synchronization overhead)
- Compute: **156.3 ms** (this is the problem)

Scenario 4 uses the exact same data, same GPU, same operation, and achieves 0.33 ms. The only difference: Scenario 1 is the **first GPU compute kernel execution** in the benchmark.

**This is CuPy's JIT compilation overhead.** The first call to `cp.sum(values_gpu[mask_gpu == mask_val])` triggers:
1. CUDA kernel compilation for the boolean comparison (`mask_gpu == mask_val`)
2. CUDA kernel compilation for the boolean indexing
3. CUDA kernel compilation for the reduction

Subsequent calls reuse the compiled kernel cache. This is a known CuPy behavior but the magnitude (~156 ms overhead) is significant for latency-sensitive applications.

**Why pathmaker didn't see this**: The earlier tests (Test 1, 2, 3) warm up CuPy's kernel cache before the benchmark runs. By the time Scenario 1 executes, the relevant kernels are already compiled. In my run, the kernel cache was apparently cold for this specific operation signature. The difference may be due to session state, GPU driver caching, or the specific order of test execution.

**Implication for WinRapids**: First-time execution of any new operation pattern will incur JIT compilation overhead. A production system needs either:
- Pre-warming: compile all kernel signatures at startup
- AOT compilation: pre-compile kernels to binary (CuPy's `RawKernel` with `options=('--compile-only',)` or NVRTC caching)
- Accept the cold-start cost and ensure it only happens once

#### 2. The Co-Native Memory Map

The memory map output is genuinely readable by both human and AI agent:

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

  GPU: 240.0 MB
  PIN: 80.0 MB
  CPU: 200.0 MB
```

An agent can parse this to answer:
- Which columns are on GPU? (high, low, close)
- What would it cost to compute on all OHLCV columns? (promote open: 3.2 ms, volume: 1.4 ms = 4.6 ms total)
- How much GPU memory is in use? (240 MB)
- Which columns are cold (zero accesses)?

This is the co-native split in action: the metadata is CPU-resident, human-readable, and machine-parseable simultaneously.

#### 3. Query Planner Validation

The query planner's estimates vs actual costs:

| Query | Estimated Transfer (ms) | Actual Total (ms) | Accuracy |
|-------|------------------------|-------------------|----------|
| All on GPU | 0.0 | 0.33 (compute only) | Correct — no transfer needed |
| Flag on pinned (10 MB) | 0.2 | 1.56 (1.1 promote + 0.4 compute) | Underestimate — actual promotion ~5x slower than bandwidth estimate |
| All on CPU (90 MB) | 3.6 | 6.69 (6.2 promote + 0.5 compute) | Underestimate — actual promotion ~1.7x slower than bandwidth estimate |

**The estimates are systematically too optimistic.** The `estimate_transfer_ms` function uses raw bandwidth numbers from Experiment 002 (25 GB/s for pageable, 57 GB/s for pinned), but actual CuPy transfers include Python overhead, array construction, and CUDA synchronization. A correction factor of 1.5-2x on the estimates would be more accurate.

**This is not a design flaw** — having conservative estimates that can be calibrated is better than no estimates. The query planner's value is in relative cost comparison (which promotion is cheapest?), not absolute time prediction.

#### 4. Promotion Amortization

The Scenario 3 -> Scenario 4 transition demonstrates amortization:
- Scenario 3 (cold): 6.7 ms (promotion + compute)
- Scenario 4 (warm): 0.33 ms (compute only)

After paying the 6.2 ms promotion cost once, all subsequent operations on those columns are free. For iterative workloads (training loops, repeated queries), the one-time promotion cost becomes negligible over hundreds of iterations.

#### 5. Even Worst Case Beats Pandas

All-CPU start (Scenario 3): 6.7 ms total, including 90 MB of transfers.
pandas: 28.7 ms.
Speedup: **4.3x even with full CPU -> GPU transfer included.**

This validates the key architectural claim: the GPU's compute speed more than compensates for the transfer cost at 10M rows. The crossover point (where transfer cost exceeds pandas compute) is worth investigating for smaller data sizes.

#### 6. Architecture Evolution: GpuFrame -> TieredFrame

| Feature | GpuFrame (Exp 004) | TieredFrame (Exp 006) |
|---------|-------------------|----------------------|
| Data residence | GPU only | 4 tiers (GPU, pinned, pageable, storage) |
| Location tracking | Single enum | `Tier` enum with bandwidth metadata |
| Transfer control | Implicit (always on GPU) | Explicit `promote()` |
| Cost visibility | None | `promotion_cost_ms`, `query_plan()` |
| Memory map | Static (all GPU) | Dynamic (shows tier, cost, access count) |
| Arrow integration | Import/export | Device type mapping |
| Filtered sum perf | 0.38 ms | 0.33 ms (on-GPU) |
| API surface | sum/mean/min/max/std/filtered | Smart operations via query plan |

TieredFrame is strictly more capable than GpuFrame. The filtered sum performance is within noise (0.33 vs 0.38 ms). The key additions — explicit tier management, cost estimation, and query planning — add no compute overhead.

### Conclusions

1. **The tiered architecture works.** Four memory tiers with explicit promotion, measured costs, and query planning. Data integrity maintained across all tier transitions.

2. **Co-native metadata is validated.** The memory map is simultaneously human-readable and machine-parseable. An agent can reason about data placement, estimate costs, and make promotion decisions without touching GPU memory.

3. **Promotion estimates need calibration.** The bandwidth-based estimates are 1.5-2x too optimistic due to CuPy/Python overhead. A calibration pass using actual measured costs would improve planning accuracy.

4. **CuPy JIT compilation is a cold-start concern.** First-time kernel execution can take ~150 ms. A production system needs kernel pre-warming or AOT compilation.

5. **Even worst-case performance (all-CPU start) is 4x faster than pandas.** The GPU compute advantage outweighs the transfer cost at 10M rows.

6. **Promotion amortization is effective.** One-time transfer cost, then free for all subsequent operations on that column.

### Open Questions

1. What is the data size crossover point where GPU+transfer becomes slower than CPU-only pandas? (Estimated: somewhere around 10K-100K rows based on transfer overhead.)
2. Can CuPy kernels be pre-compiled to avoid the ~150 ms cold-start? What is the compile-once-run-many overhead?
3. How should the query planner handle multi-column operations where some columns are on different tiers? (Currently: promote all needed columns before computing.)
4. Can async promotion overlap with compute on already-promoted columns? (The 1-engine async overlap from Experiment 002 suggests yes.)

---

## Entry 012 — Experiment 007 Results: Polars GPU Acceleration on Windows

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: Polars GPU is Linux-only (confirmed). Polars CPU is surprisingly fast (4-10x faster than pandas). GPU still wins by 6-23x over Polars CPU. Arrow interop is the pragmatic integration path.

### Independent Execution

Ran `experiments/007-polars-gpu/polars_gpu_test.py` independently. Polars 1.39.3.

### Results Comparison

#### Polars GPU Engine

Confirmed: `engine="gpu"` raises error requiring `cudf_polars` package (RAPIDS cuDF, Linux-only). The `GPUEngine` class exists in the Polars API but is a stub pointing to the cuDF integration.

#### Polars CPU Benchmarks (10M rows)

| Operation | Observer (ms) | Pathmaker (ms) | Agreement |
|-----------|--------------|----------------|-----------|
| sum | 0.86 | 0.81 | Within noise (~6%) |
| mean | 1.48 | 1.48 | Exact match |
| filtered sum | 7.28 | 7.20 | Within noise (~1%) |
| arithmetic | 14.11 | 14.21 | Within noise (~1%) |

**Excellent reproducibility.** All values agree within single-digit percentage.

#### GPU vs Polars CPU

| Operation | Polars CPU (ms) | GPU CuPy (ms) | GPU Speedup |
|-----------|----------------|---------------|-------------|
| sum | 0.86 | 0.15 | 5.6x (pathmaker: 8.4x) |
| filtered sum | 7.28 | 0.32 | 23.1x (pathmaker: 23.2x) |

**Note on sum speedup variance**: My GPU sum was 0.15 ms vs pathmaker's 0.096 ms. Polars CPU was similar in both runs. The discrepancy is in the GPU timing — likely CuPy overhead variance or GPU clock state. The filtered sum speedup matches almost exactly (23.1x vs 23.2x).

#### Arrow Interop

| Direction | Observer (ms) | Pathmaker (ms) |
|-----------|--------------|----------------|
| Polars -> Arrow | 0.25 | 0.32 |
| Arrow -> Polars | 0.24 | 0.32 |

Sub-millisecond both ways. Effectively zero-copy for numeric columns.

### Analysis

#### 1. Polars CPU is a Strong Baseline

Polars is dramatically faster than pandas across the board:

| Operation | pandas (ms) | Polars CPU (ms) | Polars Speedup over pandas |
|-----------|-------------|-----------------|---------------------------|
| sum (10M) | 8.0 | 0.86 | **9.3x** |
| mean (10M) | 11.9 | 1.48 | **8.0x** |
| filtered sum (10M) | 30.0 | 7.28 | **4.1x** |

Polars achieves its speed through Rust backend, SIMD vectorization, lazy evaluation, and multi-threaded execution. This makes the GPU advantage smaller: GPU is 6-23x faster than Polars CPU, vs 50-200x faster than pandas.

**Implication for WinRapids**: The value proposition changes depending on the baseline. Against pandas users: 50-200x is transformative. Against Polars users: 6-23x is useful but not transformative. WinRapids should target both audiences but with different messaging.

#### 2. The GPU Backend is NOT Pluggable

Key finding from Test 5: Polars exposes `GPUEngine` as a top-level attribute, but the backend is hardcoded to `cudf_polars`. There is no generic interface for plugging in an alternative GPU backend.

What a pluggable backend would need:
- Receive a Polars logical plan (filter, select, join, etc.)
- Execute each node on GPU, returning Arrow-compatible results
- Handle partial GPU execution (some ops on GPU, fallback on CPU)

This would require upstream Polars changes. Polars has discussed backend extensibility but it's not yet implemented.

#### 3. Arrow as the Integration Layer

The practical path for WinRapids + Polars on Windows:
```
Data on disk (Parquet/CSV)
  -> Polars for I/O, query planning, lazy evaluation
  -> Arrow zero-copy export (0.25 ms)
  -> CuPy/GPU for heavy compute
  -> Arrow zero-copy import back to Polars (0.24 ms)
```

Total overhead for the Polars<->GPU handoff: ~0.5 ms. This is negligible compared to the compute savings.

#### 4. Three-Way Performance Comparison

| Operation | pandas (ms) | Polars CPU (ms) | GPU CuPy (ms) | GPU vs pandas | GPU vs Polars |
|-----------|-------------|-----------------|---------------|---------------|---------------|
| sum | 8.0 | 0.86 | 0.15 | 53x | 5.6x |
| filtered sum | 30.0 | 7.28 | 0.32 | 94x | 23x |

The GPU advantage is clear in both comparisons, but the Polars gap is much smaller. For users already on Polars, the GPU value proposition is strongest for operations where Polars CPU is slowest (filtered operations, joins, complex aggregations).

### Conclusions

1. **Polars GPU is Linux-only** (confirmed). The `cudf_polars` dependency is not available on Windows. No workaround.

2. **The GPU backend is not pluggable.** WinRapids cannot act as a drop-in Polars GPU backend without upstream Polars changes.

3. **Polars CPU is 4-9x faster than pandas.** This means the GPU speedup over Polars (6-23x) is smaller than over pandas (50-200x). Both are still significant.

4. **Arrow zero-copy interop is the pragmatic path.** Polars<->Arrow roundtrip costs ~0.5 ms. Polars handles I/O and query planning; GPU handles compute.

5. **WinRapids positioning**: For pandas users, GPU is 50-200x. For Polars users, GPU is 6-23x. Both are worth pursuing, but the onboarding story is different.

### Open Questions

1. How does the GPU advantage scale for Polars' more complex operations (groupby, join, window functions)?
2. Will Polars implement a pluggable backend interface? If so, what would WinRapids need to expose?
3. What is the Polars streaming engine's performance on data larger than RAM? Can it be combined with GPU compute?

---

## Entry 013 — Incident Report: WDDM Memory Corruption and VRAM Safety Limit

**Date**: 2026-03-20
**Type**: Incident / environment note
**Status**: Complete

### Incident

During the expedition, the GPU entered a corrupted state: 100% utilization, 300W power draw, 87C temperature — with no active compute workloads. Symptoms consistent with WDDM memory corruption, likely triggered by excessive VRAM allocation in prior experiments (Experiment 002 allocated 88+ GB on a 95.6 GB GPU, which triggered WDDM virtual memory paging and process crash — documented in Entry 006).

A full machine reboot was required to recover. Post-reboot state: 2 GB VRAM used, 95 GB free, 0% utilization, 55C. Clean.

### VRAM Safety Limit Established

**Maximum safe VRAM allocation: ~60 GB.** This provides a ~33 GB margin below the 93.5 GB physical VRAM (after OS overhead). The margin accounts for:
- Desktop compositing fluctuations (DWM can spike to 10+ GB)
- WDDM driver internal structures
- Other GPU-using applications (browsers, etc.)
- Safety factor to avoid triggering WDDM virtual memory paging, which caused the original corruption

**All future experiments must:**
1. Check `cudaMemGetInfo()` or equivalent before large allocations
2. Never allocate more than ~60 GB without explicit release
3. Always free GPU memory when experiments finish (CuPy: `cp.get_default_memory_pool().free_all_blocks()`)
4. Monitor VRAM usage if approaching the limit

### Relevance to Prior Findings

This incident validates the concern raised in Entry 006 (Test 4 analysis): "WDDM silently pages GPU memory to system RAM — dangerous performance cliff." We now know the worst case: not just a performance cliff, but a **corruption event requiring reboot**. The VRAM virtual paging under WDDM is not just slow — it can leave the GPU in an unrecoverable state.

### Environment Update

Post-reboot, two new mandates are in effect:
1. **uv venv for all Python**: Virtual environment at `R:/winrapids/.venv`. Use `.venv/Scripts/python.exe` and `uv pip install`.
2. **Dual-path design**: Every experiment should explore both a pragmatic path (CuPy/Arrow/existing libraries) and a from-scratch path (custom CUDA C++/Rust). Benchmarks decide which wins. This measures the abstraction cost of CuPy and other libraries.

---

## Entry 014 — TCC Status Revised: Ambiguous, Not Confirmed

**Date**: 2026-03-20
**Type**: Correction / re-verification
**Status**: Complete

### Background

Entry 004 (earlier in this session) claimed TCC mode was "AVAILABLE" based on `nvidia-smi -g 0 -dm 1` returning exit code 0 and reporting success. The Environment section was updated to say "Currently running WDDM by choice, not constraint."

The navigator challenged this assessment by examining `nvidia-smi -q` output directly.

### Navigator's Evidence

The navigator reports that `nvidia-smi -q` shows:
- `Driver Model Current: WDDM, Pending: WDDM`
- No TCC-related fields appear as options anywhere in the output
- On cards that genuinely support TCC, nvidia-smi shows TCC as an option in the Pending field

### Independent Re-Verification (post-reboot)

I re-ran the test on the freshly rebooted machine:

1. **Pre-test state**: `nvidia-smi --query-gpu=driver_model.current,driver_model.pending` reports `WDDM, WDDM`

2. **Set TCC**: `nvidia-smi -g 0 -dm 1` reports: "Set driver model to TCC for GPU 00000000:F1:00.0. All done. Reboot required."

3. **Post-command state**: `driver_model.current,driver_model.pending` reports `WDDM, TCC` — the pending field DID change.

4. **Immediate revert**: `nvidia-smi -g 0 -dm 0` — reverted to `WDDM, WDDM`. No reboot performed.

### Analysis

The situation is genuinely ambiguous:

| Evidence | Suggests TCC Available | Suggests TCC Unavailable |
|----------|----------------------|-------------------------|
| `nvidia-smi -g 0 -dm 1` exit code | 0 (success) | - |
| `nvidia-smi -g 0 -dm 1` output | "Set driver model to TCC... All done" | - |
| Pending field after command | Changes to TCC | - |
| `nvidia-smi -q` driver model section | - | Only shows WDDM options |
| TCC as enumerated option in nvidia-smi | - | Not listed |
| Product brand | RTX PRO (workstation class) | Max-Q variant |
| Driver version | - | 581.42 may not support TCC for this SKU |

**The command accepts the request and changes the pending state, but we have NOT verified that TCC would actually activate after a reboot.** It's possible that:
- (a) TCC is genuinely available and would work after reboot — the command is the authority
- (b) The command accepts invalid requests without validation — the pending field changes but the driver would reject the switch at boot time
- (c) TCC is available on this hardware but the current driver version (581.42) has a bug or policy decision preventing it from being listed

**We cannot resolve this without a reboot test**, which we are not performing during this expedition (reboots are expensive and risky — the WDDM corruption incident demonstrates this).

### Revised Assessment

~~TCC mode: AVAILABLE~~ -> **TCC mode: AMBIGUOUS**

The correct framing: `nvidia-smi -g 0 -dm 1` accepts the command, but `nvidia-smi -q` does not enumerate TCC as a supported option. Actual TCC functionality is unverified. **WDDM is confirmed as the operating constraint for this expedition, whether by hardware limitation, driver limitation, or project choice.**

This changes the framing of all WDDM overhead measurements: they are not "compared to what we could get with TCC" but rather "the actual performance characteristics of this system." The overhead is what it is.

### Methodological Lesson

Entry 004 was wrong to claim TCC was "confirmed available" based solely on a command accepting input and returning exit code 0. A command succeeding is not the same as the requested state being achievable. The proper verification would have been:
1. Check `nvidia-smi -q` for TCC as an enumerated option
2. Actually reboot and verify the driver model changed
3. Run a compute workload in TCC mode to confirm functionality

I did step 0 (run the command) and skipped steps 1-3. The navigator caught this by doing step 1. Steps 2-3 remain unperformed.

### Note on Compute Mode

The navigator also noted that `Compute Mode: Default` is reported by nvidia-smi. Other options include "Exclusive Process" (only one process can use the GPU) and "Prohibited" (no CUDA processes). For benchmarking, "Exclusive Process" mode would prevent other processes from interfering with GPU measurements, giving cleaner numbers. This is worth using for future precision benchmarks.

### Environment Section Updated

The Environment section at the top of this notebook has been updated to reflect the ambiguous TCC status.

---

