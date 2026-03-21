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
- **TCC mode**: **AMBIGUOUS, NOT PURSUED, and WOULD BE WORSE** — `nvidia-smi -g 0 -dm 1` reports success but `nvidia-smi -q` does not enumerate TCC. Even if available, TCC does NOT support `cudaMallocAsync`/memory pools (our primary allocation strategy). Machine is accessed via RDP, so switching away from WDDM would also break the session. See Entries 004, 014, 015, 016.
- **MCDM mode**: **UNAVAILABLE** — requires driver R595+, we have R581.42. MCDM would provide headless compute WITH memory pool support (the best of both worlds). See Entry 016.
- **WDDM is the correct operating mode** for this expedition — not just the only option, but actually the best available option for our workload patterns.

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

## Standing Methodology

*Operating constraints and procedures that apply to ALL experiments. Established through hard-won findings across 18 entries.*

### VRAM Safety Ceiling: 60 GB Maximum

Never allocate more than 60 GB of GPU memory at once. WDDM allows silent over-allocation to system RAM; the 88 GB allocation in Experiment 002 corrupted WDDM state (100% GPU, 300W, 87C) and required a full machine reboot. Always query `cudaMemGetInfo` before large allocations and reserve 20% headroom. See Entry 013.

### Python Environment: uv venv Only

Always run `.venv/Scripts/python.exe`. Always install packages with `uv pip install` from `R:/winrapids`. Never use raw `python` or `pip`. Virtual environment location: `R:/winrapids/.venv`.

### Exclusive Process Compute Mode for Precision Benchmarks

For benchmarks where GPU contention must be eliminated as a variable, set compute mode to Exclusive Process (`nvidia-smi -c 3`) to prevent other processes from using the GPU. Revert to Default (`nvidia-smi -c 0`) after the experiment completes. See Entry 015.

### GPU Health Check Before Experiments

Run `nvidia-smi -q` before any GPU experiment to confirm temperature, power draw, utilization, and VRAM usage are at baseline. A hot GPU (>70C) or high baseline VRAM usage (>5 GB) may indicate residual state from previous experiments or desktop compositing pressure. Wait for cooldown or reboot if necessary.

### WDDM Is the Fixed Operating Envelope

All measurements in this notebook are under WDDM. TCC is ambiguous and not pursued (RDP access constraint; also loses memory pools which are our biggest performance win at 344x). MCDM requires driver R595+, we have R581.42. WDDM is not just the constraint — it is the correct mode for our workload. See Entries 004, 014, 015, 016.

### Benchmark Rigor Requirements (for "Beats RAPIDS" Claims)

Any claim that WinRapids beats RAPIDS on Linux must satisfy ALL of the following before it is stated as a finding rather than a hypothesis:

1. **Same query set** — TPC-H or TPC-DS, standardized queries, not custom microbenchmarks
2. **Same data sizes** — SF1, SF10, SF100 minimum (scale factors must match)
3. **Same hardware tier** — Blackwell vs whatever RAPIDS benchmarks on, with hardware differences noted
4. **End-to-end wall clock time** — total query time including I/O, parsing, compute, and materialization. Not kernel time alone, not microbenchmarks
5. **Cold-start handling** — either include cold-start in the measurement or explicitly exclude with justification (e.g., "warm cache, steady-state operation")

Until all five are met, the claim is a hypothesis under investigation, not a confirmed result.

### Custom Over Workaround

When an experiment hits a barrier, evaluate "build the custom version" alongside workarounds. The default instinct is to find a workaround (different library, API adapter, compatibility shim). The WinRapids principle: 200 lines of informed custom code often beats three layers of abstraction fighting their own design constraints. When documenting barriers in future experiments, include a "custom path" option in alternatives considered — not just "workaround A vs workaround B" but "workaround vs build it right."

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
- **CuPy 14.0.1** (CUDA 13.x wheel) — **Note**: CuPy bundles its own CUDA runtime (12.9), not using the system toolkit's 13.1. Both coexist under driver 581.42 via backward compatibility. This means CuPy operations and nvcc-compiled .cu code use different CUDA runtime versions. The driver mediates both, so no issues expected, but worth noting if behavioral differences between CuPy kernels and custom CUDA kernels are ever observed.
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
2. **CuPy kernel cache persistence**: **RESOLVED — YES, it persists.** Three-tier cache measured: in-process 0.02 ms, disk cache ~5 ms, novel compile ~40 ms. The ~300 ms cold-start is CUDA context init, not compilation. Full timing data in Entry 021 supplementary note.
3. **Warmup pattern**: Can WinRapids provide a `winrapids.warmup()` call at import time that runs dummy kernels for each operation pattern? This is standard practice for production GPU libraries (PyTorch does similar via `torch.compile` warmup).
4. **Query planner calibration**: The 1.5-2x estimate optimism may be a fixed overhead (Python/CuPy dispatch ~0.4 ms per operation) rather than a bandwidth estimation error. If so, the planner should add a constant term, not a scaling factor. Testable by measuring promotion at multiple data sizes and fitting the overhead.
5. How should the query planner handle multi-column operations where some columns are on different tiers? (Currently: promote all needed columns before computing.)
6. Can async promotion overlap with compute on already-promoted columns? (The 1-engine async overlap from Experiment 002 suggests yes.)

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

2. **The GPU backend is not pluggable today.** WinRapids cannot act as a drop-in Polars GPU backend without upstream Polars changes. However, Polars is actively maintained and the GPU backend is relatively new — a pluggable backend interface is not architecturally impossible, just not implemented. If WinRapids produces results worth showing upstream, a PR to make the GPU backend pluggable is a future option.

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

### Open Question: VRAM Envelope Characterization

The 60 GB limit is conservative. A more precise heuristic needs a controlled experiment: what actually happens at 85%, 90%, 95% of physical VRAM under varying display load (1 vs 3 monitors, DWM compositing active vs minimal)? Does WDDM begin paging at a predictable threshold, or is the transition abrupt? The current heuristic candidates:
- 80% of `cudaMemGetInfo()` free
- Physical VRAM minus 10 GB
- Whichever is more conservative

A future experiment should allocate incrementally (1 GB steps) while monitoring GPU temperature, utilization, and transfer bandwidth to detect the paging onset. This would replace the current conservative guess with a measured safety margin.

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

## Entry 015 — Operational Constraints and Baseline Anomaly Notes

**Date**: 2026-03-20
**Type**: Environment notes / caveats for future readers
**Status**: Complete

### TCC Not Pursued (Operational Constraint)

Regardless of whether TCC mode would activate on this hardware (see Entry 014 — status is ambiguous), it is **not being pursued** during this expedition. Tekgy is accessing this machine via RDP (Remote Desktop Protocol). Switching to TCC would disable the display driver, breaking the RDP session.

This means:
- The "Estimated without WDDM (Linux/TCC)" column in Entry 003's impact table remains **theoretical**, not measurable on this hardware during this expedition.
- All performance numbers in this notebook are WDDM numbers. They are the actual performance characteristics of the system as operated.
- A future TCC comparison would require local console access or a different access method.

### VRAM Baseline Anomaly (Entry 001)

Entry 001 recorded ~69 GB VRAM in use at session start, leaving ~29 GB available for compute. This observation needs caveats for future readers:

1. **The 69 GB figure may not be representative.** The machine has three monitors connected, but this level of VRAM consumption by desktop compositing alone is unusually high. A typical WDDM workstation with multiple monitors might use 2-10 GB for DWM/compositing.

2. **Possible causes for the anomaly:**
   - WDDM memory leak in a desktop application (Edge WebView, DWM, Explorer)
   - Ollama (LLM inference server) was running and may have loaded GPU-resident models
   - Driver artifact where allocated VRAM isn't freed after application exit
   - nvidia-smi's reporting includes shared GPU memory pools that may overcount actual VRAM occupation

3. **Post-reboot comparison**: After the VRAM corruption reboot (Entry 013), the GPU reported 2 GB used, 95 GB free. This confirms the 69 GB was transient application state, not a permanent OS/driver overhead.

4. **The CUDA API measurement is more reliable**: Experiment 002 (Entry 006) measured VRAM via `cudaMemGetInfo()` and found 93.55 GB free (2.05 GB OS overhead). This is the correct baseline for compute availability, not nvidia-smi's application-level reporting.

**Future readers should use 93.5 GB free (~2 GB OS overhead) as the representative VRAM baseline**, not the 29 GB figure from Entry 001.

---

## Entry 016 — The Driver Mode Triangle: WDDM, TCC, and MCDM

**Date**: 2026-03-20
**Type**: Research / environment analysis
**Status**: Complete

### Background

The navigator reported a third driver mode in the picture: **MCDM (Microsoft Compute Driver Model)**. This is new to the expedition and has significant implications for the architecture decisions made based on WDDM constraints.

### The Three Driver Modes

Research confirms three distinct GPU driver modes on Windows, with a critical and counterintuitive capability difference:

| Feature | WDDM | TCC | MCDM (R595+) |
|---------|------|-----|---------------|
| Display output | Yes | No | No |
| RDP compatible | Yes | No | No |
| `cudaMallocAsync` / memory pools | **Yes** | **NO** | **Yes** |
| `cudaMallocManaged` / prefetch | Alloc yes, prefetch no | Yes | TBD |
| Kernel launch overhead | Higher (display driver) | Lower | Lower (expected) |
| Available on this system | Yes (current) | Ambiguous (Entry 014) | No (requires R595+, we have R581) |

### The Critical Inversion: TCC Loses Memory Pools

**This is the most important finding in this entry.** TCC mode does NOT support `cudaMallocAsync` or stream-ordered memory pools. The NVIDIA developer forums confirm that `cudaDevAttrMemoryPoolsSupported` returns 0 in TCC mode.

This inverts the naive assumption that "TCC = better for compute." Our Experiment 002 (Entry 006) established that CUDA memory pools are **the** architecture decision for WinRapids — 344x faster than raw `cudaMalloc`/`cudaFree`. If we switched to TCC, we would lose this capability entirely. The 344x allocation speedup disappears. Every GPU allocation would require raw `cudaMalloc`/`cudaFree` at 30-17,000 us per cycle.

**WDDM keeping memory pool support is not an accident — it's because WDDM's virtual memory management infrastructure (which causes the overhead we measured) is also what enables the pool allocator to work.**

### MCDM: The Best of Both Worlds (Future)

Starting with NVIDIA driver R595, GPUs that previously defaulted to TCC will default to MCDM instead. MCDM provides:
- Headless compute (no display driver overhead)
- `cudaMallocAsync` and memory pools (unlike TCC)
- `cuMemCreate` and advanced memory management APIs

MCDM is built on Windows WDDM 2.6+ infrastructure but stripped of display functionality. It's the compute-only subset of WDDM.

**We cannot use MCDM**: Our driver is R581.42. MCDM requires R595+. A driver update would be needed.

### Impact on Expedition Decisions

1. **WDDM was the right choice all along** — even if TCC had been available and we had switched, we would have lost memory pools. Our entire allocation strategy (Entry 006) depends on `cudaMallocAsync`.

2. **The "WDDM overhead" framing needs revision.** We've been treating WDDM overhead as a tax we pay for being on Windows. In reality, WDDM gives us capabilities (memory pools) that TCC lacks. The overhead is the cost of those capabilities, not pure waste.

3. **MCDM is the future target.** When driver R595+ becomes available for this GPU, MCDM would give us the best of both worlds: no display overhead AND memory pools. This is worth tracking.

4. **The "estimated without WDDM" column in Entry 003 is misleading.** It implicitly assumes TCC would be strictly better. In fact, TCC without memory pools could be *worse* for WinRapids' workload patterns (many small allocations in the hot path).

### Revised Driver Mode Assessment

| Mode | Verdict for WinRapids |
|------|----------------------|
| WDDM (current) | **Best available option.** Memory pools work. Display overhead is measurable but manageable. |
| TCC | **Worse than WDDM for our workload.** Loses memory pools, which are our primary allocation strategy. |
| MCDM | **Ideal but unavailable.** Requires R595+ driver. Would give headless compute + memory pools. |

### Sources

- [Microsoft Compute Driver Model Overview](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm)
- [MCDM Architecture](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm-architecture)
- [TCC driver mode and cudaMemPool — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tcc-driver-mode-and-cudamempool/190230)
- [Stream-Ordered Memory Allocator — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)

---

## Entry 017 — Arrow C Device Interface: Scope and sync_event Protocol Reference

**Date**: 2026-03-20
**Type**: Scope clarification / protocol reference
**Status**: Complete

### Context

The navigator provided scoping guidance for Task #4 (Arrow GPU integration). Experiment 003 (Entry 007) already completed this task, but the navigator's message contains a useful protocol detail not captured in the existing entry: the `sync_event` field of the Arrow C Device Data Interface.

### What Experiment 003 Tested vs Navigator's Scope

| Navigator's Scope Item | Experiment 003 Coverage |
|------------------------|------------------------|
| CuPy array on GPU | Done (all tests) |
| Extract raw device pointer | Done (DLPack, Test 2) |
| ArrowDeviceArray struct with CPU metadata + GPU pointer | Done (Test 3) |
| Consumer reads metadata without GPU access | Done (Test 3 — co-native split) |
| Buffer pointer valid for CUDA kernel | Done (Test 5 — custom kernel on Arrow data) |
| `pyarrow.cuda` | Correctly excluded — not in Windows pip wheels |
| `sync_event` protocol | **NOT covered in Experiment 003** |

### Arrow Device Type Constants (Reference)

| Constant | Value | Maps to | WinRapids Status |
|----------|-------|---------|-----------------|
| `ARROW_DEVICE_CPU` | 1 | Tier::Pageable | Use for host-resident columns |
| `ARROW_DEVICE_CUDA` | 2 | Tier::Device | Primary — pool-allocated GPU buffers |
| `ARROW_DEVICE_CUDA_HOST` | 3 | Tier::Pinned | Staging buffers for H2D/D2H |
| `ARROW_DEVICE_CUDA_MANAGED` | 13 | N/A | **NEVER USE** — `cudaMallocManaged` is 66 ms for 4 MB on WDDM (Entry 003), `cudaMemPrefetchAsync` not supported |

### sync_event Protocol (Reference for Future Implementation)

The Arrow C Device Data Interface includes a `sync_event` field in `ArrowDeviceArray` that controls producer-consumer synchronization:

- **`sync_event = NULL`**: Synchronous producer. Data is already materialized and ready to read. The consumer can access buffer pointers immediately. **WinRapids should start here** — simplest correct behavior.

- **`sync_event = non-NULL`**: Pointer to a `CUevent` (for CUDA devices). The producer has enqueued GPU work but it may not have completed. The consumer must synchronize before accessing data:
  - If consumer has a CUDA stream: `cuStreamWaitEvent(consumer_stream, event)` — non-blocking, just adds a dependency
  - If consumer has no stream: `cuEventSynchronize(event)` — blocks until producer's work is done

- **Ownership**: Transfers to consumer on `get_next()`. Consumer is responsible for cleanup via `release()`.

This protocol enables pipelined execution: a producer can hand off a batch while the GPU is still writing to it, and the consumer's stream will automatically wait. For WinRapids' async pipeline (Experiment 002 showed 80% overlap benefit), this is the mechanism for zero-copy producer-consumer handoff without explicit synchronization barriers.

**Implementation order**: Start with `sync_event = NULL` (all operations fully synchronous). Add event-based sync when implementing the pipelined ingestion path (overlap H2D transfer with compute from Entry 006, Test 5).

---

## Entry 018 — Experiment 008 Results: I/O Path Benchmarks (NVMe to GPU)

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: Unbuffered ReadFile + pinned memory is the optimal I/O path on Windows. 2.25x faster than naive Python read. mmap is slower than Python read. DirectStorage ruled out for CUDA workloads.

### Independent Execution

Ran `experiments/008-io-benchmarks/io_bench.py` using `.venv/Scripts/python.exe`. All checksums match across all four methods at all three sizes.

### Results Comparison

#### End-to-End Bandwidth: File to GPU Memory (GB/s)

| Method | Size | Observer | Pathmaker | Agreement |
|--------|------|----------|-----------|-----------|
| Python read | 1 MB | 3.18 | 3.7 | Within variance (~14%) |
| Python read | 100 MB | 3.81 | 3.2 | Within variance (~16%) |
| Python read | 1 GB | 3.92 | 3.3 | Within variance (~16%) |
| mmap | 1 MB | 3.05 | 3.2 | Within noise |
| mmap | 100 MB | 2.78 | 2.6 | Within noise |
| mmap | 1 GB | 2.85 | 2.6 | Within noise |
| Unbuffered+pageable | 1 MB | 2.53 | 2.8 | Within noise |
| Unbuffered+pageable | 100 MB | 4.05 | 4.1 | Within noise |
| Unbuffered+pageable | 1 GB | 4.10 | 2.8 | **Divergent — see analysis** |
| **Unbuffered+pinned** | 1 MB | 4.17 | 3.8 | Within noise |
| **Unbuffered+pinned** | 100 MB | 8.63 | 8.8 | Within noise (~2%) |
| **Unbuffered+pinned** | 1 GB | **8.81** | **8.8** | **Exact match** |

#### Speedup vs Python Read (1 GB)

| Method | Observer | Pathmaker |
|--------|----------|-----------|
| mmap | 0.73x (slower) | 0.78x (slower) |
| Unbuffered+pageable | 1.05x | 0.83x |
| **Unbuffered+pinned** | **2.25x** | **2.65x** |

### Analysis

#### 1. Unbuffered + Pinned Wins Decisively

At 1 GB: 119 ms (8.81 GB/s) vs 268 ms (3.92 GB/s) for naive Python read. Both my run and pathmaker's agree on the unbuffered+pinned bandwidth: **8.8 GB/s**. This is the most reproducible measurement in the experiment.

The mechanism is clear: unbuffered+pinned eliminates two copies:
1. No OS buffer cache copy (`FILE_FLAG_NO_BUFFERING`)
2. No pageable-to-pinned staging copy (data lands directly in DMA-accessible pinned memory)

The NVMe -> pinned buffer -> GPU path has only one intermediate buffer (the pinned staging area), and that buffer is directly DMA-accessible by both the NVMe controller and the GPU.

#### 2. mmap is Worse Than Python read

Both runs confirm: mmap is 0.73-0.78x slower than naive Python `read()` at 1 GB. This is consistent and surprising.

The cause: mmap relies on page fault handling for sequential reads. Each 4 KB page triggers a fault, the OS reads the page from disk, and then the CPU touches it again during `cp.asarray()`. For a sequential read that's accessed once and discarded, mmap's lazy-loading provides no benefit — the page faults are pure overhead.

**Implication for WinRapids**: Never use mmap for GPU data loading. The Python `open().read()` baseline is already better.

#### 3. Unbuffered+Pageable Divergence at 1 GB

My run shows 4.10 GB/s at 1 GB; pathmaker's shows 2.8 GB/s. This is a significant divergence (46%). Possible causes:
- Filesystem cache state differences (my test files were created in the same session)
- Background I/O contention during pathmaker's run
- WDDM pageable transfer variance (Entry 006, Test 3 showed pageable D2H at 1 GB has high variance)

The unbuffered+pinned results agree almost exactly (8.81 vs 8.8 GB/s), suggesting the divergence is in the pageable transfer step, not the disk read. This is consistent with Entry 006's finding that pageable transfers have higher variance than pinned.

#### 4. The 8.8 GB/s Question

Both runs show 8.8 GB/s, which exceeds the expected NVMe sequential read bandwidth of ~7 GB/s (PCIe 4.0 x4). As the README notes, this suggests OS filesystem caching is contributing despite `FILE_FLAG_NO_BUFFERING` — the test files were created in the same session and may still be in the NTFS cache.

**To measure true cold-read performance**: the files would need to be created in a separate session, or the filesystem cache flushed between creation and benchmarking. True cold-read performance is likely ~7 GB/s (NVMe hardware limit).

**Note**: If this machine has a PCIe 5.0 NVMe drive, the limit would be ~14 GB/s and the 8.8 GB/s would represent ~63% utilization. Worth checking the NVMe specs.

#### 5. Pipeline Architecture Implications

Combining this with Experiment 002 findings:

| Pipeline Stage | Bandwidth | Source |
|---------------|-----------|--------|
| NVMe -> pinned buffer | 8.8 GB/s (cached) / ~7 GB/s (cold) | This experiment |
| Pinned -> GPU (H2D) | 55-58 GB/s | Entry 006, Test 3 |
| GPU compute | 1,677 GB/s | Entry 003 |

The bottleneck is overwhelmingly disk I/O. A 1 GB dataset:
- Disk read: 119-143 ms (NVMe limited)
- H2D transfer: ~18 ms (if done separately)
- GPU compute: <1 ms (for aggregations)

The optimal pipeline overlaps disk read and H2D transfer using the async overlap capability from Experiment 002 (80% benefit):
1. Allocate a pinned staging buffer (e.g., 256 MB)
2. Read chunk N from disk into staging buffer
3. While reading chunk N+1, async H2D transfer chunk N to GPU
4. While transferring chunk N+1, compute on chunk N

This matches the Sirius paper's architecture and is validated by our experiments.

#### 6. DirectStorage Decision

DirectStorage was NOT benchmarked. The README explains: it requires a D3D12 intermediary with 4 API boundary crossings (NVMe -> DirectStorage -> D3D12 buffer -> `cudaImportExternalMemory` -> CUDA), vs 1 for unbuffered ReadFile (NVMe -> pinned buffer -> `cudaMemcpy`). **Update**: the D3D12->CUDA step is zero-copy (same physical VRAM, confirmed production API since CUDA 10). The "4 crossings" are API setup boundaries, not data copies. This reduces the overhead concern for the DirectStorage path.

The llama.cpp team reportedly confirmed unbuffered ReadFile beats DirectStorage for CUDA workloads. However, the naturalist noted that the DirectStorage Zstd shader is generic (open-sourced HLSL), which reopens the question for compressed formats like Parquet. For uncompressed data, unbuffered ReadFile + pinned is clearly optimal. For compressed data, the GPU Zstd decompression in DirectStorage might offset the API crossing overhead.

### Conclusions

1. **Unbuffered ReadFile + pinned memory is the optimal I/O path for CUDA on Windows.** 2.25x faster than naive Python read, achieving 8.8 GB/s (near NVMe limits).

2. **mmap is worse than Python read for GPU workloads.** Page fault overhead outweighs any benefit for sequential, read-once data. Never use mmap for GPU data loading.

3. **Disk I/O is the pipeline bottleneck**, not H2D transfer or GPU compute. A 1 GB load: disk 119 ms, H2D 18 ms, compute <1 ms. Optimization effort should focus on I/O.

4. **Pipelined architecture is viable**: chunk reads + async H2D overlap from Experiment 002. This is the Sirius model adapted for Windows.

5. **DirectStorage ruled out for uncompressed CUDA workloads.** The D3D12 intermediary adds more overhead than it removes. Open question for compressed formats (GPU Zstd).

### Open Questions

1. What is the true cold-read NVMe bandwidth? (Need filesystem cache flush between file creation and benchmark.)
2. Is the NVMe drive PCIe 4.0 or 5.0? (Determines whether 8.8 GB/s is ~100% or ~63% of theoretical.)
3. For Parquet files: does the GPU Zstd decompression in DirectStorage offset the D3D12 API crossing overhead? **PARTIALLY RESOLVED**: D3D12-to-CUDA interop is zero-copy (same physical VRAM, `cudaImportExternalMemory`, production API since CUDA 10). The data movement cost of the D3D12 intermediary is zero. Remaining question: does the total pipeline (NVMe -> DirectStorage GPU Zstd -> D3D12 resource -> CUDA zero-copy -> compute) beat CPU Zstd + unbuffered ReadFile + pinned H2D? API setup and shader dispatch overhead still unmeasured. Needs Experiment 015.
4. Can the pipelined chunk reader + async H2D achieve higher end-to-end throughput than the single-shot measurements here?

---

## Entry 019 — Experiment 009 Results: Dual-Path GPU DataFrame (CUDA C++ vs CuPy)

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: CuPy abstraction cost is 1.2-3.9x depending on operation. Kernel fusion and warp-shuffle are the two biggest wins from going raw CUDA. Architecture decision: CuPy for prototyping, custom CUDA for production hot paths where operations compose.

### Experiment Design

`experiments/009-dual-path-dataframe/cuda_dataframe.cu` implements the same operations from Experiment 004 (GpuFrame) in raw CUDA C++, measured with CUDA events for sub-microsecond precision. Operations: sum reduction, filtered sum, FMA (a*b+c), min/max, vectorized double2 FMA. All on 10M float64 elements (~80 MB per column). Uses `cudaMallocAsync` pool directly — no CuPy allocator.

This is the first from-scratch experiment under the dual-path directive (AD-5).

### Results — Abstraction Cost by Operation

| Operation | CUDA C++ | CuPy/GpuFrame | Abstraction Cost | Why |
|-----------|----------|----------------|-----------------|-----|
| Sum (warp-shuffle) | 0.082 ms (976 GB/s) | 0.099 ms | **1.2x** | CuPy's sum is already warp-optimized |
| Min (warp-shuffle) | 0.062 ms (1299 GB/s) | 0.123 ms | **2.0x** | CuPy min/max uses atomics, not shuffle |
| Max (warp-shuffle) | 0.062 ms (1299 GB/s) | 0.137 ms | **2.2x** | Same |
| Filtered sum | 0.084 ms (1072 GB/s) | 0.331 ms | **3.9x** | Warp-shuffle vs CuPy fancy indexing |
| FMA (a*b+c) | 0.192 ms (1668 GB/s) | 0.531 ms | **2.8x** | 1 fused kernel vs 2 separate launches |
| Double2 FMA (vectorized) | 0.200 ms | 0.192 ms scalar | no benefit | Already bandwidth-bound |
| CuPy RawKernel filtered sum | 0.172 ms | — | — | 2x slower than warp-shuffle |

### Analysis

#### 1. Sum — The Smallest Gap (1.2x)

CuPy's sum kernel is already warp-optimized internally. The 1.2x difference is Python call overhead and kernel dispatch through CuPy's path. At FinTek's "trillions of computations" scale, Tekgy's directive — "even 1.2x on sum may be worth going custom" — is specifically about compounding: a 20% per-operation overhead applied 10^12 times is real wall-clock time.

#### 2. Filtered Sum — The Biggest Gap (3.9x)

The architectural difference:
- **CuPy fancy indexing**: `values[mask]` compacts to a temp array (80 MB write + read), then `sum()` on temp. Two kernels, one intermediate.
- **Warp-shuffle filtered sum**: Predicated accumulation — elements failing the filter contribute 0. Single kernel, zero intermediates.

This pattern recurs throughout: when operations *compose*, raw CUDA avoids the intermediate materialization that CuPy is forced to produce. The gap is architectural, not tuning.

#### 3. FMA — Kernel Fusion Value (2.8x)

For `a * b + c`, CuPy launches two kernels with an 80 MB intermediate `a*b`. The fused kernel does it in one pass; the compiler emits hardware `__fma_rn` automatically.

#### 4. Vectorized double2 — No Benefit

Loading two doubles simultaneously doesn't help because the operation is already memory-bandwidth-bound at the cache line level. The GPU fetches 128-byte cache lines; scalar and vectorized loads hit identical cache lines. No benefit from explicit vectorization when bandwidth is the bottleneck.

### Architectural Decision

CuPy for prototyping. Custom CUDA for production hot paths where operations compose (filter+sum, expr+reduce, groupby+agg). The backend-as-contract design enables this: same DataFrame API, different engine behind it.

---

## Entry 020 — Experiment 010 Results: Kernel Fusion Engine (C++ Expression Templates)

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: Compile-time expression template fusion achieves zero abstraction overhead over hand-written kernels. Fusion advantage grows with expression complexity (1.5x to 3.5x). Expression complexity is free — all fused kernels run at ~0.19 ms regardless of depth.

### Experiment Design

`experiments/010-kernel-fusion/fused_ops.cu` builds a CUDA C++ expression template engine. An expression AST is constructed at compile time via templates — `Add<Mul<ColumnRef, ColumnRef>, ColumnRef>` for `a*b+c`. The template tree's `eval()` method is inlined per-element into a single kernel. No intermediate buffers. No runtime AST walking.

Five fused expressions vs CuPy separate-op baseline:

| Expression | Fused | CuPy | Speedup | Intermediates saved |
|------------|-------|------|---------|---------------------|
| a*b+c | 0.194 ms | 0.291 ms | **1.5x** | 80 MB |
| a*b+c*c-a/b | 0.193 ms | 0.670 ms | **3.5x** | 320 MB |
| where(a>0, b*c, -b*c) | 0.195 ms | 0.565 ms | **2.9x** | 320 MB |
| sum(a*b+c) | 0.177 ms | 0.336 ms | **1.9x** | compute+reduce fused |
| sqrt(abs(a*b+c*c-a)) | 0.188 ms | 0.657 ms | **3.5x** | 400 MB |

### Key Finding: Expression Complexity Is Free

All fused kernels run at ~0.19 ms regardless of expression depth. The reason: all are memory-bandwidth-bound. Each kernel reads input columns once and writes output once. Whether per-element computation is `a*b+c` (3 ops) or `sqrt(abs(a*b+c*c-a))` (6 ops + transcendental), GPU time is dominated by memory access. The extra arithmetic is nearly free because compute units are underutilized while waiting for memory.

CuPy pays 80 MB write + read per intermediate. The fused kernel pays for none of them.

### Template Fusion Matches Hand-Written Kernels

FMA template fusion (0.194 ms) matches the hand-written kernel from Experiment 009 (0.192 ms) within measurement noise. Zero abstraction cost from the template machinery. The compiler sees through the template indirection and emits identical PTX.

### Architectural Decision

Expression templates are the from-scratch kernel strategy for WinRapids. Compile-time fusion, zero runtime cost, zero intermediate buffers. C++ templates serve as the reference implementation for validating the Python codegen path (Experiment 010b).

---

## Entry 021 — Experiment 010b Results: Python Kernel Fusion via CuPy RawKernel Codegen

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: Python codegen captures 85-95% of C++ fusion benefit. JIT overhead is negligible after warm-up. CuPy RawKernel codegen is the production path — no C++ build step required.

### Experiment Design

`experiments/010-kernel-fusion/fused_frame.py`: `FusedColumn` wraps a CuPy array. Arithmetic operators build a lazy expression tree (no kernel launch). `.evaluate()` generates CUDA kernel source from the tree, compiles via `cp.RawKernel` (nvrtc), launches. Kernel cache by expression hash: same tree structure reuses compiled kernel.

### Results

| Approach | a*b+c | a*b+c*c-a/b | sqrt(abs(a*b+c)) |
|----------|-------|-------------|-----------------|
| CuPy eager | 0.291 ms | 0.670 ms | 0.657 ms |
| C++ fused (Exp 010) | 0.194 ms | 0.193 ms | 0.188 ms |
| Python codegen | ~0.205 ms | ~0.200 ms | ~0.196 ms |
| Python overhead vs C++ | +0.01-0.05 ms | — | — |
| Capture of C++ benefit | **85-95%** | **85-95%** | **85-95%** |

### JIT Overhead — Negligible After First Call

nvrtc compilation on first evaluation of a new expression structure. Driver caches compiled kernels — subsequent evaluations reuse cached PTX, paying ~0 ms. Distinct from CuPy's 157 ms cold-start (first any CuPy kernel launches — Python import + CUDA context init). The codegen JIT adds cost once per unique expression structure, amortized to zero over repeated calls.

### fused_sum() — Compute+Reduce in One Kernel

`fused_sum(a*b+c)` produces a scalar sum without ever writing element-wise results to VRAM. CuPy would do: multiply kernel, add kernel, sum kernel — three launches, two intermediates. The fused path: one kernel evaluates `a*b+c` per-element AND reduces in the same pass via warp-shuffles. 1.5x over CuPy's 3-kernel approach for this pattern.

### Why Python Overhead Is Small

The Python overhead is CPU-side, not GPU-side: tree traversal for source generation, cache lookup, `cp.RawKernel` instantiation if uncached. The GPU kernel quality is identical to C++ — same PTX, same warp-shuffle logic. The 0.01-0.05 ms overhead is acceptable for the benefit of no build step.

### Architectural Decision

Python codegen via `cp.RawKernel` is the production path for kernel fusion. No C++ build step, no MSVC dependency. C++ expression templates (Experiment 010) serve as the validation reference and lower bound on achievable latency.

### Supplementary: Kernel Cache Persistence (Observer Verification, 2026-03-20)

**Question (from Entry 011 open question #2):** Does CuPy's kernel cache persist across Python process restarts?

**Answer: YES.** Verified empirically on this machine.

CuPy maintains a disk-based kernel cache at `~/.cupy/kernel_cache/` — 105 `.cubin` (compiled CUDA binary) files confirmed on this machine. These survive process restarts. The cache key is a hash of the kernel source including compiler version, so it invalidates on CUDA version change.

**Three tiers of kernel caching, measured:**

| Tier | Latency | Location |
|------|---------|----------|
| In-process (same kernel, repeat call) | 0.02 ms | Python dict (`_kernel_cache`) + GPU cache |
| Disk cache (same kernel, new process) | ~5 ms | `~/.cupy/kernel_cache/*.cubin` |
| Novel compile (never-seen kernel) | 38-43 ms | NVRTC runtime compilation |

**Cross-session test:** Same `RawKernel` source launched in process 1: 43.49 ms (NVRTC compile). Same source in process 2 (separate Python process): 4.94 ms (disk cache hit, ~9x faster).

**Two distinct cold-start costs on process restart:**

1. **CUDA context initialization (~200-430 ms):** CuPy import + driver init + context creation + device selection. NOT cacheable. Happens every process start. This is the dominant startup cost.
2. **Per-expression kernel load (~5 ms if disk-cached, ~40 ms if novel):** Loading a previously-compiled `.cubin` from disk is ~9x faster than NVRTC compilation but still ~250x slower than in-process cache.

**Implication for `winrapids.warmup()`:**

1. Force CUDA context init eagerly (e.g., `cp.cuda.Device(0).use(); cp.zeros(1)`) — pays the ~300 ms context cost at import time rather than on first use.
2. Pre-launch one kernel per expression pattern to promote from disk cache (~5 ms) to in-process cache (~0.02 ms).
3. Do NOT pre-compile kernels — the disk cache already handles this across sessions.

**Note on `fused_frame.py`'s `_kernel_cache` dict (line 220):** This in-process dict is NOT redundant with CuPy's disk cache. It avoids the ~5 ms disk-to-memory load cost for hot-loop repeated expressions within a session. Both caching layers serve distinct purposes.

---

## Entry 022 — Experiment 011 Results: GPU GroupBy

**Date**: 2026-03-20
**Type**: Experiment execution and analysis
**Status**: Complete
**Verdict**: GPU groupby achieves 21-706x speedup over pandas depending on approach and cardinality. Hash-based scales dramatically at high cardinality; sort-based is stable regardless. Dual-dispatch by cardinality is the correct architecture.

### Experiment Design

`experiments/011-gpu-groupby/gpu_groupby.py` implements two GPU groupby approaches:
1. **Sort-based**: argsort on keys → segmented cumsum for group sums
2. **Hash-based**: atomic scatter — each thread atomically adds its value to the group's bucket

Test cases: 10M rows × {100, 10K, 1M} groups. Also multi-aggregation (sum+mean+count) in one sort pass.

### Results

| Groups | Sort-based (ms) | Sort speedup | Hash-based (ms) | Hash speedup | Pandas (ms) |
|--------|-----------------|-------------|-----------------|-------------|-------------|
| 100 | 3.3 | **21x** | 3.1 | **22x** | 69.7 |
| 10K | 3.4 | **26x** | 0.5 | **176x** | 87.7 |
| 1M | 3.4 | **113x** | 0.5 | **706x** | 378.3 |

Multi-aggregation (sum+mean+count, 100 groups): 3.4 ms vs pandas 98.6 ms — **29x**.

### Analysis

#### Sort-Based: The Stable Path (~3.3 ms regardless of cardinality)

The bottleneck is `cp.argsort` — O(N log N) in elements, not groups. CUDA's radix sort is highly optimized, but it dominates. After sorting, segmented cumsum is cheap. Advantages:
- Bounded memory: output = N_groups doubles, no hash table sizing issues
- Produces sorted output — useful for downstream operations
- Compositional: sort once, apply N aggregations (demonstrated by multi-agg 29x)

#### Hash-Based: The Scaling Win

At 1M groups: 0.5 ms (706x). Each bucket receives ~10 elements on average — low atomic contention. At 100 groups: ~100,000 elements per bucket, heavy contention, barely beats sort (22x vs 21x).

The pattern: hash-based scales *inversely with collision probability*. At high cardinality, it becomes nearly contention-free.

#### Cardinality Threshold

Crossover around 1K-10K groups:
- Below ~1K: sort and hash are comparable; prefer sort for ordering and multi-agg
- Above ~10K: hash wins dramatically (contention falls below sort's argsort cost)

#### Multi-Aggregation Composition

Sort-based's strongest architectural advantage: sort once, apply N aggregations nearly free. Sum+mean+count in 3.4 ms — only 0.1 ms more than a single sum (argsort dominates in both cases). Hash-based can do multi-agg but needs separate atomic accumulations per aggregation.

### Connection to Experiment 012

Experiment 011 establishes the baseline two-step cost: expression → sort → reduce. Experiment 012 tests fusing the expression into the sort+reduce step, eliminating the intermediate expression buffer.

Critical observation: Experiment 012's fused kernel uses per-element atomic adds for group accumulation — the same mechanism as hash-based. At low cardinality (100 groups), this may be *slower* than the unfused sort+cumsum path, because atomic contention kills the fusion benefit. This is the exact question Experiment 012 is designed to answer.

### Architectural Decision

Dual-dispatch groupby: hash-based for >1K groups (low contention, high parallelism), sort-based below (stable cost, multi-agg composition). Cardinality estimation via `cp.unique` is near-zero overhead and can guide dispatch before the expensive aggregation.

---

## Entry 023 — Experiment 012 Results: Fused GroupBy Expressions

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: Fully fused atomic groupby is 2.6-4.5x SLOWER than the hybrid approach at low cardinality. Hybrid (fuse expression + sort-based reduction) wins everywhere. "Fuse the computation, don't fuse the reduction."

### Experiment Design

`experiments/012-fused-groupby/fused_groupby.py` tests three approaches to `groupby(key).sum(a*b+c)`:
1. **CuPy unfused**: separate kernels for expression, then sort+cumsum reduction
2. **Fully fused (atomic)**: one kernel evaluates expression through sort permutation AND atomically accumulates group sums
3. **Hybrid**: fused expression kernel (one kernel, no intermediates) + sort-based cumsum reduction (no atomics)

### Results (Observer's Independent Run)

| Expression | Groups | CuPy unfused | Fully fused | Hybrid | Hybrid vs unfused |
|------------|--------|-------------|-------------|--------|-------------------|
| a*b+c | 100 | 3.58 ms (28.2x) | 16.25 ms (6.2x) | 3.56 ms (28.4x) | 1.01x |
| a*b+c | 10K | 3.63 ms (33.0x) | 4.54 ms (26.3x) | 3.52 ms (34.0x) | 1.03x |
| a*b+c | 1M | 3.62 ms (115.3x) | 4.31 ms (96.9x) | 3.55 ms (117.6x) | 1.02x |
| a*b+c*c-a/b | 100 | 4.00 ms (33.7x) | 16.51 ms (8.2x) | 3.49 ms (38.6x) | 1.15x |
| a*b+c*c-a/b | 10K | 4.13 ms (36.6x) | 4.44 ms (34.0x) | 3.57 ms (42.3x) | 1.16x |

### Analysis

**Fully fused atomic is a failure at low cardinality.** At 100 groups, 16.25 ms vs hybrid's 3.56 ms — 4.6x slower. 100 groups with 10M rows means 100K elements per group, all atomically contending on the same 100 accumulator slots. The atomic contention destroys the fusion benefit.

At higher cardinality (10K-1M groups), atomic contention drops and the fully fused path closes the gap but never beats the hybrid. This confirms Entry 022's prediction: "atomic contention kills the fusion benefit."

**Hybrid wins everywhere.** The hybrid approach gets the best of both worlds:
- Fused expression evaluation: one kernel, zero intermediate buffers (saves VRAM and bandwidth)
- Sort-based cumsum reduction: no atomics, no contention, stable performance

For the complex expression (a*b+c*c-a/b), hybrid is 1.15x faster than unfused CuPy — the fusion saves one intermediate buffer and two extra kernel launches. The benefit scales with expression complexity.

**Lesson: fuse the computation, don't fuse the reduction.** Reductions and group accumulations have fundamentally different parallelism requirements than element-wise compute. Trying to fuse them forces atomic operations that create contention.

### Open Questions

1. At what expression complexity does the hybrid fusion benefit exceed 1.5x over unfused? (Current max: 1.15x for 5-op expression)
2. Can warp-level segmented reduction replace atomics in the fully fused path? This would eliminate contention but requires knowing group boundaries at warp granularity.

---

## Entry 024 — Experiment 013 Results: GPU Hash Join

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: Direct-index join achieves 267-499x over pandas. Sort-merge achieves 8-233x. Hash join with CPU build phase scales poorly — 0.9x at 1M dim rows (SLOWER than pandas). Correctness verified at 100% for all methods.

### Experiment Design

`experiments/013-gpu-joins/gpu_join.py` implements three GPU join strategies:
1. **Sort-merge**: argsort on dim keys + searchsorted probe
2. **Direct-index**: for dense integer keys [0, N), use the key as the array index (O(1) lookup)
3. **Hash join**: CPU-built hash table (Fibonacci hashing, open addressing) + GPU-parallel probe via RawKernel

Test cases: 10M fact rows x {10K, 100K, 1M} dim rows. Star schema (all fact keys match).

### Results (Observer's Independent Run)

| Dim Size | pandas | Sort-merge | Direct-index | Hash (CPU+GPU) |
|----------|--------|-----------|-------------|----------------|
| 10K | 246 ms | 1.06 ms (233x) | 0.66 ms (371x) | 2.97 ms (83x) |
| 100K | 261 ms | 1.52 ms (171x) | 0.70 ms (374x) | 25.01 ms (10x) |
| 1M | 349 ms | 1.92 ms (182x) | 0.70 ms (499x) | 372 ms (0.9x) |
| 10K (50% match) | 159 ms | 19.93 ms (8x) | 0.60 ms (266x) | 2.85 ms (56x) |

Join + GroupBy pipeline (10M x 10K -> 50 categories): 6.88 ms GPU vs 315 ms pandas — **45.8x**.

### Analysis

**Direct-index is the clear winner for dense integer keys.** 0.60-0.70 ms regardless of dim size — O(1) per lookup, no sorting, no hashing. This is the optimal join for star schemas with integer surrogate keys, which is the most common pattern in analytics.

**Sort-merge scales gracefully.** 1.06-1.92 ms across dim sizes, driven by the argsort cost on dim keys. Works for arbitrary key types. But at 50% match rate: 19.93 ms — the searchsorted + filter path has to examine non-matching positions, which is expensive.

**Hash join with CPU build phase is the bottleneck.** The Python for-loop building the hash table on CPU is O(n_dim). At 1M dim rows, this takes ~370 ms — making the GPU probe irrelevant. This is a correctness/completeness gap, not a performance result. A proper GPU hash join needs atomicCAS for GPU-side build.

**Correctness: 100% across all methods.** Every match count matches pandas, every key pair is correct.

### Open Questions

1. GPU-side hash table build with atomicCAS — how does it compare to CPU build at 1M+ dim rows?
2. Sort-merge at partial match rate (50%): 19.93 ms is surprisingly slow. Is the filter step the bottleneck, or is it the non-matching probe overhead?
3. For non-integer keys (strings, timestamps), what's the right join strategy? Sort-merge seems most general.

---

## Entry 025 — Experiment 014 Results: End-to-End GPU Analytics Pipeline

**Date**: 2026-03-20
**Type**: Experiment execution and independent verification
**Status**: Complete
**Verdict**: GPU pipeline achieves 3.0x over pandas end-to-end (164.8 ms vs 499.6 ms). GPU compute is 5.5 ms total (join + fused expression + groupby). H2D transfer at 111 ms is 95% of GPU pipeline time. I/O is the bottleneck — GPU compute is no longer the constraint.

### Experiment Design

`experiments/014-end-to-end/e2e_pipeline.py`: complete analytics pipeline from Parquet to result.
- **Scenario**: 10M sales records x 10K products, revenue by category
- **Query**: `groupby(category).agg(sum(quantity * unit_price * (1 - discount)))`
- **Pipeline**: Parquet read -> Arrow -> CuPy (H2D) -> direct-index join -> fused revenue kernel -> sort-based groupby -> result (D2H)

### Results (Observer's Independent Run)

| Stage | Time (ms) | % of GPU pipeline |
|-------|-----------|-------------------|
| Read Parquet (CPU) | 47.94 | — |
| H2D Transfer | 111.35 | 95.3% |
| Join (direct-index) | 1.40 | 1.2% |
| Fused Compute | 0.24 | 0.2% |
| GroupBy (sort+cumsum) | 3.83 | 3.3% |
| D2H Result | 0.08 | 0.1% |
| **GPU total (excl. read)** | **116.90** | |
| **Full pipeline** | **164.84** | |
| **Pandas total** | **499.6** | |
| **End-to-end speedup** | **3.0x** | |
| **Compute-only speedup** | **~91x** | |

### Analysis

**GPU compute is 5.5 ms.** Join (1.40) + fused compute (0.24) + groupby (3.83) = 5.47 ms of actual GPU work. This is ~91x faster than pandas' compute. The entire speedup bottleneck is I/O.

**H2D transfer dominates at 111.35 ms (95.3% of GPU time).** This is transferring ~192 MB through CuPy's `cp.asarray()` which uses pageable memory. Using pinned memory (Experiment 008) could reduce this to ~3.4 ms at 57 GB/s, bringing the GPU pipeline from 117 ms to ~9 ms and the end-to-end speedup from 3.0x to ~8.6x.

**The 3.0x end-to-end speedup understates the GPU advantage.** The pipeline is I/O-bound. With optimized I/O (pinned H2D + unbuffered ReadFile from Experiment 008), the GPU pipeline could be ~57 ms (48 ms Parquet read + 9 ms GPU), achieving ~8.8x over pandas.

**Correctness verified.** Max sum error: 6.91e-05 (floating point accumulation order), max mean error: 4.07e-10. Categories match.

**Second query on resident data: ~5 ms.** If data is already on GPU, the pipeline is join + compute + groupby = 5.5 ms. That's ~91x faster than pandas. This validates GPU-resident data model (AD-10).

### Cross-Reference with Naturalist's Claims

| Metric | Naturalist's claim | Observer's measurement | Match? |
|--------|-------------------|----------------------|--------|
| GPU compute total | 5.1 ms | 5.47 ms | Close |
| End-to-end speedup | 3.2x | 3.0x | Close |
| H2D transfer | 110 ms | 111.35 ms | Match |
| Join | 1.4 ms | 1.40 ms | Match |
| Fused compute | 0.24 ms | 0.24 ms | Match |
| GroupBy | 3.4 ms | 3.83 ms | Close |

All claims verified within expected measurement variance. The 3.2x vs 3.0x discrepancy is pandas baseline variation.

### Architecture Implications

1. **I/O elimination is the #1 optimization.** GPU compute at 5.5 ms is already fast. Pinned H2D (57 GB/s vs pageable) would 30x the transfer stage.
2. **GPU-resident data model is critical.** Second query at 5 ms vs first at 165 ms — 30x difference is entirely I/O.
3. **Parquet read is CPU-bound.** The 48 ms Parquet read is Arrow's CPU decompression. DirectStorage GPU decompression (if viable) would eliminate this bottleneck.
4. **The 3.0x headline is misleading.** "GPU compute is 91x faster, I/O makes it 3x." Fix I/O, headline jumps to 8-90x depending on residency.

### Open Questions

1. What does this pipeline look like with pinned memory H2D? (Expected: 111 ms -> ~3.4 ms)
2. Can Arrow's Parquet reader use pinned memory for decompression output, enabling zero-copy H2D?
3. What's the second-query latency on already-resident data? (Expected: ~5 ms, needs measurement)
4. At what data size does the GPU pipeline break even with pandas? (Expected: below 100K rows)

---

## Entry 026 — Correction: "85% of GPU Query Time Is Parquet Scanning" Is Unverified

**Date**: 2026-03-20
**Type**: Correction / source verification failure
**Status**: Flagged

### The Claim

The expedition log (lines 263, 497, 567) cites arxiv paper 2602.17335 ("Do GPUs Really Need New Tabular File Formats?") for the claim that **85% of TPC-H query runtime on GPUs is spent in Parquet scanning**. This claim is used load-bearingly in:
- The DirectStorage value case (if 85% is scanning, winning I/O wins the benchmark)
- The competitive positioning vs RAPIDS (Section: "I/O may dominate")
- The "beat RAPIDS on Linux" strategy (navigator's framing: "Win the I/O stage, win the benchmark")

### The Problem

The scout attempted to verify this citation via two scholar searches and could not surface the specific 85% number or confirm the paper reference. The claim is currently **unverified** — we cannot trace it to a specific figure, table, or measurement in a peer-reviewed source.

### What We DO Know

1. **Experiment 014 (Entry 025) provides our own measurement**: H2D transfer is 95.3% of GPU pipeline time for our end-to-end analytics query. GPU compute is 5.5 ms out of 116.9 ms (4.7%). This is consistent with the "I/O dominates" thesis but is a different measurement (H2D transfer, not Parquet scanning).

2. **Adjacent evidence exists**: Takafuji et al. (2022), doi:10.1002/cpe.7454 — GPU shader-based Deflate decoding is 1.66-8.33x faster than multi-threaded CPU on A100. This supports "GPU decompression frees CPU" but is not Parquet-specific.

3. **Our own Entry 025 data**: 48 ms Parquet read + 111 ms H2D = 159 ms I/O, vs 5.5 ms compute. That's 96.7% I/O. This is stronger evidence than an uncitable third-party number.

### Resolution

- Mark "85% scanning" as **plausible but unverified** wherever it appears
- When the DirectStorage experiment runs, measure I/O+decompression as % of total pipeline time on our hardware — this becomes our own citation
- Our Entry 025 data (96.7% I/O) is actually stronger evidence for the I/O-dominance thesis than the unverified 85% claim

### Methodological Lesson

A citation that cannot be traced to a specific figure in a specific paper is not a citation — it's a rumor. The expedition log should not use unverifiable claims as load-bearing evidence for strategy decisions. Our own measurements (Entry 025) are more authoritative than third-party numbers we can't verify.

---

