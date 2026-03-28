# Pipeline Performance Synthesis
*Observer, 2026-03-28 — distilled from 14 experiments + team research*

---

## The Answer in One Table

| Stage | Per-ticker | Universe (4604) | % of Batch Pipeline | Bottleneck Rank |
|---|---|---|---|---|
| NVMe → CPU | 2.9ms | 13.4s | 38% | **#1** |
| CPU → GPU | 0.5ms | 2.3s | 7% | #4 |
| GPU compute | ~2ms | ~9s | 26% | #2 |
| GPU → CPU | ~0.1ms* | ~0.5s | 1% | #5 |
| CPU → NVMe (no-fsync) | 2.6ms | 12s | 34% | #3 |
| **Total (sequential)** | **~8ms** | **~35s** | | |
| **Total (3-stage pipeline)** | — | **~15s** | | |

\* D2H for cadence results (~260 bytes per cadence per ticker), not full arrays. Full-array D2H is 1.7ms but unnecessary for derived outputs.

With fsync writes: 170s (writes = 82% of pipeline).

---

## Three Architectural Levers (by impact)

### 1. Write Mode Flag (~5x pipeline impact)
**The single largest lever.** `fsync=False` for derived outputs reduces the pipeline from 170s to 35s.

Source files (K01 from ingest) = crash-safe (fsync + rename). Everything else = recomputable.

| Mode | Cadence Grid | % of Pipeline |
|---|---|---|
| Production (2×fsync) | 140s | 82% |
| Batch (no fsync) | 12s | 34% |

### 2. Pipeline Overlap (~2x throughput)
With 3-stage pipelining (read N+1 / process N / write N-1):
- Sequential: 35s
- Pipelined: max(reads, GPU, writes) ≈ max(13.4, 9, 12) ≈ **15s**

Requires: async I/O (Python asyncio + thread pool for reads), CUDA streams for H2D/compute overlap, write thread pool.

### 3. Selective Reads (~3-5x for specific leaves)
Many leaf computations need only 2-3 columns. V4 selective read (2 cols) = 1.3ms vs full 2.9ms = 2.2x faster. For leaves needing only price+size:
- Universe read: 6.0s instead of 13.4s

---

## Where Optimization Doesn't Help

| Optimization | Measured Gain | Why Not Worth It |
|---|---|---|
| Rust reader | ~7% | Read is 97% I/O (Exp 12) |
| Handle pools | -40% (WORSE) | Bulk read penalty (Exp 13) |
| Raw I/O (os.open) | ~4% | Python buffer overhead minimal (Exp 13) |
| mmap | Deceptive | 2.49x cold/warm penalty (Exp 1) |
| O_SEQUENTIAL hint | ~2% | Within noise (Exp 13) |

## Bandwidth Budget

| Interface | Theoretical | Measured | Utilization |
|---|---|---|---|
| NVMe (sequential read) | ~7 GB/s | 4.86 GB/s | 69% |
| PCIe Gen5 x16 (H2D) | ~32 GB/s | 24.6 GB/s | 77% |
| PCIe Gen5 x16 (D2H) | ~32 GB/s | 7.3 GB/s | 23% |
| VRAM (GPU-internal) | 1792 GB/s | — | compute-dependent |

Correction: Scout's tensor-core-analysis estimated H2D at 0.009ms using VRAM bandwidth (1792 GB/s). That's internal GPU bandwidth, not the host→device PCIe bus. Actual H2D = 0.5ms at 24.6 GB/s PCIe.

## Metadata Is Free

Header parsing (13 sections, 60+ fields): **6.4 microseconds**.
Adding domain descriptors, K-space parameters, upstream fingerprints = zero measurable cost. The format's Block 0 has 2238 bytes reserved. Add metadata freely.

---

## What the Rust Reader Should Actually Do

The read path is 97% I/O-bound. Rewriting struct.unpack → Rust gains 7%. **Not worth it for reads.**

Where Rust actually helps:
1. **Write path**: 42% CPU overhead (SHA-256 + column stats + tobytes). Rust could halve this.
2. **Pipeline orchestrator**: Async I/O + CUDA stream management in Rust (tokio + cudarc) would be cleaner than Python asyncio + threading.
3. **SHA-256 → xxHash**: 25% of write time is SHA-256 (4.8ms for 12.6MB). xxHash would be ~0.5ms. This alone saves 22% of large-file writes.

The value of Rust is in the write path and orchestration, not the read path.

---

## Full Tree Projection

| Stage | K01→K02 | K02→K03 | K03→K04 |
|---|---|---|---|
| Read input | 13.4s | 12s (cadence files) | ~8s (feature files) |
| GPU compute | ~9s | ~5s | <1s (GEMM, Tensor Core) |
| Write output | 12s | ~8s | ~2s |
| **Subtotal** | **~35s** | **~25s** | **~11s** |
| **Running total** | **35s** | **60s** | **71s** |
| Pipelined estimate | 15s | ~13s | ~8s |
| **Pipelined total** | 15s | **28s** | **36s** |

**Full K01→K04 estimate: ~71s sequential, ~36s pipelined.** Down from 3.7 hours.

---

## Progressive Section Impact (Experiment 15)

| Metric | Value | Notes |
|---|---|---|
| Write overhead | +37ms/file | 3.1 MB stats data dominates (not fsync) |
| Read overhead | +0.7% | Within noise — zero cost to existing reads |
| Progressive summary read | 83us | Directory only, no stat data I/O |
| **Coarse K04 shortcut** | **0.4s vs 18.7s** | **49x read speedup** |
| Size overhead | 25.2% (+3.1 MB) | Pure float32 stats, no indirection |

**K04 via progressive**: Instead of reading 4604 full files (18.7s), read 4604 progressive summaries (0.4s). The K03→K04 stage drops from ~11s to ~3s, making the full pipelined tree ~28s.

**Cadence selection trade-off**: The 1s cadence (23,400 bins) contributes 74% of progressive data volume but has MI=0.22 (lowest). Dropping to 5s finest would halve progressive write time with minimal information loss (MI=0.28→0.22 for the dropped level).
