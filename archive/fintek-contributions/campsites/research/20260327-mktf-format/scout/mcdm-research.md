# MCDM Research — Scout Notes
## 2026-03-27

---

## The Critical Finding Up Front

**MCDM is not available for a display-attached workstation card.**

Our RTX PRO 6000 Blackwell Workstation Edition drives monitors. MCDM is defined as a
compute-only, display-disabled mode. We are in WDDM and stay there unless we add a secondary
display adapter and run the RTX PRO headless.

This means the team-lead's framing "MCDM enables NVMe→GPU without WDDM overhead when we
update drivers" is partially correct (the driver update part) but the mode isn't accessible
without a hardware change on our workstation setup.

---

## What MCDM Actually Is

MCDM (Microsoft Compute Driver Model) was introduced in Windows 10 v1903 (WDDM 2.6) as
the Windows-sanctioned compute-only mode. It is the OS-native successor to NVIDIA's TCC:

| Mode | Description | Display | Memory pools | IOMMU/VBS | WSL2 |
|------|-------------|---------|--------------|-----------|------|
| WDDM | Full display driver | Required (if attached) | Yes (CUDA 13.1+) | Yes | Yes |
| TCC | NVIDIA bypass (legacy) | Incompatible | **NO** | No | No |
| MCDM | Windows compute-only | Incompatible | Yes | Yes | Yes |
| Linux native | No display stack | Optional | Yes | Yes | N/A |

TCC bypassed WDDM entirely — faster submission but at the cost of memory pools, IOMMU,
WSL2, containers. MCDM gives you TCC-like compute focus while preserving everything TCC broke.

**Why this matters for Day 1 findings**: Entry 016 established that TCC was not pursued
because it kills `cudaMallocAsync` (our 344x memory allocation win). MCDM fixes that
(pools work on MCDM). But we still can't reach MCDM with display attached.

---

## What R595 Actually Changes

The R595 inflection point is specifically about **TCC → MCDM default switch** for datacenter/
server-class cards: L40, L40S, L20, L4, and **RTX PRO 6000 Blackwell Server Edition**.

Not the Workstation Edition (our card).

For our display-attached workstation setup, R595 + CUDA 13.2 gives:

1. **LMEM reduction**: Significant reduction in per-context local memory footprint — useful
   for memory-constrained kernels, more headroom in 102.6 GB VRAM.
2. **RDMA support** (CUDA 13.1+): GPU-to-GPU over InfiniBand/RDMA — relevant if doing
   network-based data ingestion in future multi-machine setup.
3. **cudaMallocAsync on WDDM**: Memory pool APIs now work in WDDM mode (CUDA 13.1+).
   Our Day 1 memory pool wins are confirmed safe on WDDM.
4. **CUDA 13.2 cudaMemcpyWithAttributesAsync**: More flexible async memory copies.

**R595 is worth updating to** — LMEM reduction alone is valuable. But it does NOT enable MCDM
for our workstation setup and does NOT create an NVMe→GPU direct path.

---

## Does MCDM Enable NVMe→GPU Direct DMA?

**No. This is the clearest finding of the research.**

MCDM removes display/graphics overhead. It does NOT create a new storage DMA pathway.

The NVMe→GPU state on Windows as of March 2026:

```
NVMe → Windows storage stack → system RAM → [pinned host] → cudaMemcpyAsync → GPU VRAM
                                                              ↑
                                             This bounce is MANDATORY on Windows
                                             regardless of WDDM / MCDM / TCC mode
```

GPUDirect Storage (cuFile) is **Linux-only**. No Windows port exists. No announcement.

DirectStorage 1.4 (latest): NVMe → system RAM staging → GPU. Still requires system RAM.
DirectX 12 only — no CUDA support. Gaming/asset-streaming API, not a data science API.

True P2P DMA (NVMe → GPU): Linux kernel 6.2+ with CUDA 12.8+ supports this via PCI P2PDMA.
No Windows equivalent as of March 2026.

---

## What Actually Helps on Windows (in priority order)

### 1. Hardware-Accelerated GPU Scheduling (HAGS) — check now
WDDM's GPU scheduler traditionally runs as a high-priority CPU thread. HAGS (WDDM 2.7+)
offloads scheduling to dedicated GPU hardware, reducing submission latency. Enabled by default
on Windows 11 with RTX cards, but worth verifying.

HAGS helps CUDA graph performance (confirmed by user reports). If we're using CUDA graphs
for our fused kernel pipeline, this is relevant.

```
Settings → System → Display → Graphics → Default graphics settings → Hardware-accelerated GPU scheduling
```

### 2. Optimal current NVMe→GPU path (already implemented)
```python
# What we do today — this is already optimal for Windows:
host_pinned = cuda.pagelocked_empty(shape, dtype)  # pinned memory
# NVMe → host_pinned (DMA by NVMe controller, CPU doesn't touch data)
device_arr = cuda.to_device(host_pinned, stream=stream)  # H2D DMA
```
⚠️ CORRECTION (from observer synthesis): 1792 GB/s is GPU-internal VRAM bandwidth, not H2D rate.
PCIe 5.0 x16 H2D actual bandwidth: ~24.6 GB/s → H2D for 15.5MB = **~0.5ms** (not 0.009ms).
PCIe 5.0 NVMe → system RAM: ~12-14 GB/s = ~1.1ms for 15.5MB.
Total I/O: ~1.6ms. NVMe→RAM is still the larger piece, but H2D is ~30% of total I/O — not negligible.
The NVMe step remains the primary I/O bottleneck, but H2D is now meaningful enough that
MCDM's reduced submission latency (~3µs vs ~40µs WDDM) remains a real (if small) gain.

### 3. MCDM (secondary display adapter required)
If we add a secondary GPU (even GT 710) for display:
- RTX PRO 6000 runs headless → can switch to MCDM
- Submission latency: closer to TCC/Linux (~3µs vs ~40µs WDDM)
- CUDA graph launch overhead reduced
- Still no NVMe→GPU bypass (MCDM ≠ GPUDirect Storage)
- Estimated improvement: 2-3x on submission-latency-bound workloads, negligible for
  long-running GPU kernels

### 4. Custom NVMe driver (moonshot, naturalist researching)
User-mode NVMe command queue writing directly to pinned host memory.
Would bypass Windows storage stack overhead (storport, SCSI translation).
No system-RAM bounce eliminated — still RAM→GPU needed.
Benefit: eliminates storage stack overhead (~40-45% CPU cycles per I/O per Server 2025 data),
allows queue depth > 32, and enables fully async NVMe→pinned→GPU pipeline with no CPU copies.

---

## Honest Assessment of the Three Paths

| Path | Buildable today? | NVMe→GPU bypass? | Impact |
|------|-----------------|------------------|--------|
| MCDM (option 1) | Requires secondary display GPU | NO (still system RAM bounce) | Reduces kernel submission latency; no I/O benefit |
| Pinned staging (option 2) | YES, already built | NO | Optimal for current Windows; 8.5 GB/s |
| Custom NVMe driver (option 3) | Hard but possible | Partial (bypasses storage stack, not RAM) | CPU savings + deeper queue; no true P2P |

The gap between "Windows best possible" and "Linux with GPUDirect Storage":
- Linux: NVMe → GPU VRAM directly via P2P DMA (CUDA 12.8+, Linux 6.2+)
- Windows: NVMe → system RAM (mandatory) → GPU VRAM

This gap exists at the OS level. No driver mode changes it. It requires either a Windows port
of the Linux kernel PCI P2PDMA infrastructure, or a future Windows equivalent of cuFile.
Neither is announced.

**The WinRapids bet**: the system RAM bounce adds ~1ms at 14 GB/s NVMe bandwidth.
With 1792 GB/s GPU bandwidth, H2D is 0.009ms. The bottleneck is NVMe speed, not transfer.
Better NVMe (PCIe 5.0 x4, up to ~14 GB/s) already minimizes this. True P2P DMA would
save ~1ms — meaningful for ultra-low-latency work, but our current pipeline spends ~4ms
on compute (fused_bin_stats). I/O is not the bottleneck.

---

## Sources

- [Microsoft MCDM Overview](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm)
- [NVIDIA R595 Datacenter Driver Release Notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-595-58-03/index.html)
- [CUDA 13.2 Technical Blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [NVIDIA Forums: Will MCDM improve WDDM vs TCC?](https://forums.developer.nvidia.com/t/will-microsoft-windows-mcdm-improve-the-wddm-vs-tcc-situation/310058)
- [GitHub: WDDM 2-3x slower than TCC for RAM↔GPU transfer](https://github.com/microsoft/graphics-driver-samples/issues/103)
- [GPUDirect Storage r1.16 Overview (Linux-only confirmed)](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [NVIDIA DisplayModeSelector + RTX PRO 6000 Workstation Edition](https://forums.developer.nvidia.com/t/displaymodeselector-rtx-pro-6000-blackwell-workstation-edition/346615)
- [Hardware-Accelerated GPU Scheduling](https://devblogs.microsoft.com/directx/hardware-accelerated-gpu-scheduling/)
