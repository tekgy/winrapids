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

### 1. Write Mode Flag (context-dependent impact)
**The single largest lever for cadence grid writes without progressive.** `fsync=False` reduces 140s to 12s (91% savings).

Source files (K01 from ingest) = crash-safe (fsync + rename). Everything else = recomputable.

| Mode | Cadence Grid (no progressive) | With 7-cadence progressive |
|---|---|---|
| Production (2×fsync) | 140s (91% of this is fsync) | ~25s (fsync is ~10% — data volume dominates) |
| Batch (no fsync) | 12s | ~23s |

**Important framing**: The 91% savings applies to small files WITHOUT progressive. With progressive stats, data volume (not fsync) dominates write time, compressing the safe/batch gap to ~10%. Both modes should still use batch for K02+, but don't misquote the 91% figure for progressive-enabled files.

### 2. Pipeline Overlap (~2x throughput)
With 3-stage pipelining (read N+1 / process N / write N-1):
- Sequential (no progressive): 35s → Pipelined: max(13.4, 9, 12) ≈ **15s**
- Sequential (7-cad progressive): ~46s → Pipelined: max(13.4, 9, 23) ≈ **23s**

Progressive writes become the pipeline bottleneck. Selective reads (2-3x from reading only needed columns) would cut reads from 13.4s to ~6s, but writes at 23s still dominate. Pipeline overlap's main value with progressive: overlapping writes with next ticker's reads.

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
| Read input | 13.4s | 12s (cadence files) | ~8s (feature) / 0.4s (progressive) |
| GPU compute | ~9s | ~5s | <1s (GEMM, Tensor Core) |
| Write output (no prog) | 12s | ~8s | ~2s |
| Write output (7-cad prog) | 23s | ~10s | ~2s |
| **Subtotal (no prog)** | **~35s** | **~25s** | **~11s** |
| **Subtotal (7-cad prog)** | **~46s** | **~27s** | **~3s** (via progressive) |
| Pipelined (no prog) | 15s | ~13s | ~8s |
| Pipelined (7-cad prog) | 23s | ~13s | ~3s |
| **Pipelined total (no prog)** | 15s | **28s** | **36s** |
| **Pipelined total (7-cad prog)** | 23s | **36s** | **39s** |

**Without progressive: ~71s sequential, ~36s pipelined.** Down from 3.7 hours.
**With 7-cadence progressive: ~76s sequential, ~39s pipelined.** +3s for 49x K04 read speedup — progressive pays for itself if K04 runs more than once.

---

## Progressive Section Impact (Experiments 15-17)

| Metric | Value | Notes |
|---|---|---|
| Write overhead (10 cadences) | +37ms/file | 3.1 MB stats, dominated by 1s cadence |
| Write overhead (7 cadences) | +2ms/file | ~80 KB stats, 30min→30s only |
| Read overhead on full columns | +0.7% | Within noise — zero cost to existing reads |
| Progressive summary read | 83us | Directory only, no stat data I/O |
| **Coarse K04 shortcut** | **0.4s vs 18.7s** | **49x read speedup** |

**~~Read crossover at 1,548 bins~~ CORRECTED (Exp 17)**: The crossover was an artifact of a nested Python loop in `unpack_progressive_level()` (117,000 `np.frombuffer` calls for 1s cadence). Replacing with single `np.frombuffer` + `reshape` eliminates the crossover entirely — progressive reads are faster than full column reads at ALL cadences, including 1s (1,398us vs 2,708us = 0.52x).

**With optimized reader, decision table becomes**:
- ALL cadences: **PROGRESSIVE** — 0.03x to 0.52x of full read cost
- The cadence selection question shifts from reader performance to: MI value (1s adds MI=0.22) and write cost (+37ms for 10 vs +2ms for 7)

**Cadence set**: If the reader fix lands, 10 cadences are viable. The remaining trade-off is write cost only. 7 cadences (+2ms write) if write pipeline is bottleneck; 10 cadences (+37ms) if K04 query speed matters more.
