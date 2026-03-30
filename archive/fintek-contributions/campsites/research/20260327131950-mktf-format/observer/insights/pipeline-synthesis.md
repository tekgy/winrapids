# Pipeline Performance Synthesis
*Observer, 2026-03-28 — distilled from 21 experiments + team research + Rust benchmarks*

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
| Read input | 13.4s | 12s (cadence files) | ~8s (KO00) / **0.47s** (KO05 fast reader, Exp 21) |
| GPU compute | ~9s | ~5s | <1s (GEMM, Tensor Core) |
| Write KO00 | 12s | ~8s | ~2s |
| Write KO05 (Python) | **53s** | ~25s | ~5s |
| Write KO05 (Rust) | **31-37s** | ~15-18s | ~3s |
| **Subtotal KO00 only (seq)** | **~35s** | **~25s** | **~11s** |
| **Subtotal KO00+KO05 Python (seq)** | **~87s** | **~50s** | **~16s** |
| **Subtotal KO00+KO05 Rust (seq)** | **~65-71s** | **~40s** | **~14s** |
| Pipelined KO00 only | 15s | ~13s | ~8s |
| **Pipelined KO00+KO05 Python** | **53s** | ~25s | ~8s |
| **Pipelined KO00+KO05 Rust** | **31-37s** | ~18s | ~8s |

**KO05 writes remain the dominant pipeline bottleneck — but NTFS, not language, is the constraint.**

**Measured data across all optimization stages:**

| KO05 Writer | Per-file | Universe 8 cad | Source |
|---|---|---|---|
| Python align=4096 | 1.52ms | 56.0s | Exp 19 |
| Python align=64 | 1.38ms | 53.0s | Exp 20 bulk |
| Rust new file + rename | 1.86ms | 68.7s | pathmaker bulk |
| Rust direct write | 1.42ms | 52.3s | pathmaker bulk |
| **Rust overwrite pre-created** | **0.82ms** | **30.1s** | **pathmaker bulk** |
| Rust compute only (no I/O) | 0.005ms | 0.2s | pathmaker |
| ~~Rust 0.1ms target~~ | ~~0.1ms~~ | ~~3.7s~~ | ~~estimated~~ |

**Key correction**: The 0.87ms "Python floor" from Exp 19 was actually **NTFS floor**. Rust compute is 160x faster (5.5us vs ~500us) but can't touch the OS layer.

**Pre-creation strategy** (pathmaker): Daemon pre-creates empty MKTF files during init (it already knows the file graph). Compute wave overwrites existing files instead of creating new ones. Eliminates CreateFile + directory entry creation overhead. **818us/file = 2.3x faster than direct write.**

**Crash safety**: Pre-creation changes the failure mode from "file absent" to "file present with is_complete=0." The daemon treats both identically (delete + recompute). Safe for K02+ derived files where the daemon recovery path covers crash cases. K01 (irreplaceable source) keeps full crash-safe path (tmp + rename + fsync). The MKTF is_complete byte at EOF[-2] was designed to support both modes.

**Read performance (KO05, align=64, Exp 21):**

| Reader | Per-file (bulk unique) | K04 screening/cad | Source |
|---|---|---|---|
| Production (double open) | 4,657us | 21.4s | Exp 20 |
| **Fast (single read)** | **103us** | **0.47s** | Exp 21 |

**Without KO05: ~35s sequential, ~15s pipelined.** Down from 3.7 hours.
**With KO05 Python @ 8 cad: ~53s pipelined bottleneck.**
**With KO05 Rust direct @ 8 cad: ~52s pipelined bottleneck.**
**With KO05 Rust pre-created @ 8 cad: ~30s pipelined bottleneck.** Best achievable with per-file NTFS.
~~**With KO05 Rust 0.1ms: ~3.7s → read-bound.**~~ *NTFS floor makes this unreachable with per-file writes.*

**Pipelined wall time** (Rust pre-created + fast reader): max(13.4s reads, 30.1s writes) = **~30s** — write-bound but approaching the KO00-only batch time of ~35s sequential.

**The alignment fix**: align=64 (GPU cache line) instead of align=4096 (NVMe page). File size drops 7.2x, disk utilization jumps from 7% to 51%, reads are 21% faster. But write speed improvement is only 6-10% because the Python per-file floor (0.87ms = 56% of write time) is alignment-independent.

**Why alignment doesn't fix write speed**: The 0.87ms floor is pack_block0 + SHA-256 + file open/close/rename. These operations don't care about file size. Alignment only affects the column data write and hash, which is 40-44% of total write time. Even eliminating ALL padding would only save ~0.2ms/file.

---

## ~~Progressive Section~~ → KO05 Separate Files (Architecture Shift)

*The embedded progressive section (Experiments 15-17) has been superseded by separate KO05 files. The benchmarking research informed this decision — key findings below transfer directly to the KO05 architecture.*

**Architecture**: K02 writes TWO files per (leaf, ticker, cadence):
- `K02P##C##.TI##TO##.KI00KO00.mktf` — full bin data
- `K02P##C##.TI##TO##.KI00KO05.mktf` — sufficient stats {sum, sum_sq, min, max, count}

Both produced by the same GPU kernel. Both tracked independently by daemon.

**Measured KO05 write costs** (Exp 19, safe=False, 78 bins):
- **align=4096**: 1.52ms/file, 108 KB — 7.1% data utilization
- **align=64**: 1.38ms/file, **14.9 KB** — **51% data utilization** (correct setting)
- Per-file floor: **0.87ms** (56% of write time) — alignment-independent
- Bottleneck: Python per-file overhead (pack_block0, SHA-256, file I/O). Actual disk write = negligible.
- Universe (align=64): 36,832 KO05 files × 1.54ms = **56.9s** — largest pipeline stage

**K04 screening** (CORRECTED by Exp 20, FIX VALIDATED by Exp 21):
- Production reader bulk: 4,604 × 4,657us ≈ **21.4s per cadence** (Exp 20 — double file open + per-column seeks)
- **Fast reader bulk: 4,604 × 103us ≈ 0.47s per cadence** (Exp 21 — single f.read() + buffer slicing, **45x faster**)
- Fast reader vs full KO00 reads: 0.47s vs 18.7s → **40x faster** — KO05 screening works as designed
- Fast reader vs embedded progressive: 0.47s vs 0.4s → **1.2x slower** — architecture shift is nearly free

**Root cause and fix**: `read_columns()` opens file twice + 25 per-column seeks = 4.66ms/file on unique files. Single `f.read()` + buffer slicing = 103us/file. **~15 line change to reader.py** (task #48). The fast reader eliminates the cold/warm gap entirely: 103us cold ≈ 94us warm.

**Why the original estimate was wrong**: Exp 14's 0.2ms was for 5-column cadence files (~1-50 KB). KO05 files have 25 columns (5 data cols × 5 stats), hitting the alignment penalty hard at 4096 bytes. More columns = more ColumnEntry construction, more SHA-256 hashing, more struct.pack overhead.

**KO05 write cost — NTFS is the true floor** (updated with pathmaker's Rust benchmarks):

| Writer | Compute | NTFS ops | Total | Universe (8 cad) |
|---|---|---|---|---|
| Python (align=64) | ~0.5ms | ~0.84ms | **1.38ms** | **53s** (Exp 20 bulk) |
| Rust (crash-safe) | 0.005ms | ~0.95ms | **~1.0ms** | **36.8s** |
| Rust (direct, no rename) | 0.005ms | ~0.84ms | **~0.84ms** | **30.9s** |
| ~~Rust (0.1ms target)~~ | — | — | ~~0.1ms~~ | ~~3.7s~~ |

**CORRECTION**: The 0.1ms Rust target was unreachable. NTFS file metadata operations (CreateFile + NtWriteFile + MoveFileEx) cost **~840us minimum per file** regardless of content size or language. Rust's 160x faster compute (5.5us vs ~500us) saves ~0.4-0.5ms/file, yielding a **1.4-1.7x improvement** — not the 14x estimated.

**The "Python floor" from Exp 19 (0.87ms) was actually NTFS overhead measured through Python.** Rust strips away the language layer and reveals the OS layer beneath. Different layer, same shape — the pattern from Exp 14 (fsync = OS floor) repeats.

**Paths to faster writes** (all reduce NTFS file operation count):
1. Batch file handle pre-creation (amortize NTFS metadata)
2. Memory-mapped files with pre-allocated space
3. Consolidating cadences into fewer files (breaks node===node)

~~Skip SHA-256 and column bundling are both moot — NTFS dominates all data-proportional costs.~~
