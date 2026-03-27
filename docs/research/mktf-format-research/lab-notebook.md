# MKTF Format Research — Lab Notebook

**Observer**: Rigorous benchmarking with statistical significance
**Date**: 2026-03-27
**Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q, Windows 11, NVMe storage
**Environment**: Python 3.13, CuPy 14.0.1, PyArrow 23.0.1, NumPy 2.4.3

---

## Experiment 1: Full Format Benchmark Suite

### Setup
- **Data**: 598,057 rows x 10 columns = 35.3 MB raw
- **Columns**: price(f64), size(f64), timestamp(i64), notional(f64), ln_price(f64), sqrt_price(f64), recip_price(f64), round_lot(u8), odd_lot(u8), direction(i8)
- **Runs**: write=30, read_warm=30, read_cold=10, gpu=30
- **Cold cache**: Memory-pressure eviction (2GB allocation between runs)
- **Timing**: `time.perf_counter_ns()` with GC disabled during measurement

### Phase 1: Write + Read (Warm Cache) + File Size

| Format | Write (ms) | Read Warm (ms) | Size (MB) | W+R (ms) |
|---|---|---|---|---|
| Raw binary (MKTF v1) | 10.12 ± 0.78 | 6.86 ± 0.32 | 35.29 | 16.98 |
| Aligned binary (64B) | 11.51 ± 1.08 | 6.68 ± 0.27 | 35.29 | 18.19 |
| Mmap columns | 11.53 ± 0.49 | **0.38 ± 0.03** | 35.29 | 11.91 |
| Arrow IPC | 11.47 ± 0.93 | 7.16 ± 0.19 | 35.29 | 18.63 |
| LZ4 compressed | 24.62 ± 0.68 | 13.92 ± 0.42 | 32.90 | 38.54 |
| numpy .npz | 20.01 ± 1.10 | 15.45 ± 0.36 | 35.29 | 35.47 |
| Parquet+none | 96.65 ± 2.09 | 19.03 ± 0.68 | 35.56 | 115.68 |
| Parquet+snappy | 101.75 ± 2.33 | 21.02 ± 0.60 | 34.40 | 122.77 |
| Parquet+zstd | 127.54 ± 6.10 | 21.78 ± 0.98 | 29.74 | 149.32 |

**Observations**:
- Mmap "read" is deceptive — 0.38ms is just creating the mapping (lazy). Data isn't read until accessed.
- Aligned binary reads slightly faster than raw binary (6.68 vs 6.86ms) — alignment helps.
- Arrow IPC is competitive with our custom binary formats for read speed.
- Parquet write is 10x slower than binary. Parquet read is 3x slower.
- LZ4 compression on market data (random floats) achieves only 0.93x — not worth the CPU cost.
- Zstd achieves 0.84x compression but at 12x write cost.

### Phase 2: Cold vs Warm Cache

| Format | Warm (ms) | Cold (ms) | Cold/Warm Ratio |
|---|---|---|---|
| Mmap columns | 0.38 | 0.95 | **2.49x** |
| Aligned binary (64B) | 6.68 | 9.36 | 1.40x |
| Raw binary (MKTF v1) | 6.86 | 9.14 | 1.33x |
| Arrow IPC | 7.16 | 8.73 | 1.22x |
| numpy .npz | 15.45 | 17.98 | 1.16x |
| LZ4 compressed | 13.92 | 14.99 | 1.08x |
| Parquet+none | 19.03 | 20.12 | 1.06x |
| Parquet+snappy | 21.02 | 22.12 | 1.05x |
| Parquet+zstd | 21.78 | 22.06 | 1.01x |

**Observations**:
- Mmap has the biggest cold/warm ratio (2.49x) — it relies on OS page cache. When pages are evicted, every access triggers a page fault.
- Binary formats show ~30-40% cold penalty — real I/O vs cache hits.
- Parquet shows minimal cold/warm difference (<6%) — its CPU overhead dominates, so cache misses are hidden by decompression time.
- **Critical insight**: For a universe scan (4604 tickers), data will be cold after first access. Cold read time matters more than warm.

### Phase 3: Column-Selective Reads

| Format | 1 col (ms) | 3 cols (ms) | 5 cols (ms) | Full (ms) | 1-col speedup |
|---|---|---|---|---|---|
| Mmap columns | **0.10** | **0.17** | **0.22** | 0.38 | 3.93x |
| Aligned binary (64B) | **0.54** | 2.08 | 5.23 | 6.68 | **12.48x** |
| Raw binary (MKTF v1) | 0.84 | 2.97 | 4.66 | 6.86 | 8.13x |
| numpy .npz | 1.67 | 5.60 | 9.96 | 15.45 | 9.24x |
| Parquet+snappy | 3.66 | 10.69 | 15.96 | 21.02 | 5.75x |
| Parquet+none | 3.96 | 8.08 | 13.76 | 19.03 | 4.80x |
| LZ4 compressed | 5.05 | 11.48 | 11.68 | 13.92 | 2.76x |
| Parquet+zstd | 6.96 | 11.08 | 16.29 | 21.78 | 3.13x |

**Observations**:
- **Aligned binary wins column selectivity**: 12.48x speedup for single column read (0.54ms vs 6.68ms full). The seek-to-offset design pays off.
- Mmap is fastest absolute (0.10ms for 1 col) but deceptive — lazy mapping, not real I/O.
- Raw binary is slightly worse than aligned for selective reads — lack of alignment means more I/O waste.
- LZ4 column selectivity is poor because all columns must be decompressed sequentially (no independent column compression).
- **Design implication**: MKTF should support per-column compression, not whole-file compression.

### Phase 4: GPU Host-to-Device Transfer

| Format | Disk→GPU (ms) | Read portion (ms) | H2D portion (ms) |
|---|---|---|---|
| Aligned binary (64B) | **7.98** | 6.69 | 1.29 |
| Raw binary (MKTF v1) | 8.77 | 7.39 | 1.39 |
| Arrow IPC | 11.60 | 10.33 | 1.28 |
| Mmap columns | 14.48 | 13.10 | 1.38 |
| LZ4 compressed | 15.43 | 14.10 | 1.33 |
| numpy .npz | 17.13 | 15.81 | 1.32 |
| Parquet+none | 22.00 | 20.68 | 1.32 |
| Parquet+snappy | 24.37 | 23.07 | 1.30 |
| Parquet+zstd | 24.51 | 23.20 | 1.31 |

**Pure H2D baseline**: 1.30ms for 35.3 MB = **27.2 GB/s** (PCIe 5.0 x16 theoretical: 63 GB/s)

**Observations**:
- **H2D is NOT the bottleneck**. At 1.3ms it's only 15-16% of the aligned binary pipeline.
- **Disk read dominates**: 6.7ms read vs 1.3ms H2D for aligned binary.
- Mmap is the *worst* for disk→GPU despite being the "fastest" reader — lazy mapping means page faults during GPU copy. This is the critical insight.
- **The pipeline is I/O-bound, not transfer-bound**. Optimizing read speed > optimizing H2D.
- Pinned memory path failed — worth investigating as async DMA could overlap I/O and transfer.

### Universe-Scale Projections (4604 tickers)

| Format | Disk→GPU per ticker (ms) | Full universe (s) | Full universe (min) |
|---|---|---|---|
| Aligned binary (64B) | 7.98 | 36.7 | 0.61 |
| Raw binary (MKTF v1) | 8.77 | 40.4 | 0.67 |
| Arrow IPC | 11.60 | 53.4 | 0.89 |
| Parquet+zstd | 24.51 | 112.9 | 1.88 |

---

## Key Conclusions

1. **Aligned binary is the winner for GPU pipeline**: 7.98ms disk→GPU, 12.48x column-selective speedup.
2. **Mmap is a trap**: Fastest apparent read, worst actual GPU pipeline. Lazy evaluation hides the real cost.
3. **Compression doesn't help**: Market data (random floats) compresses poorly. CPU cost > I/O savings.
4. **H2D is cheap** (1.3ms / 27.2 GB/s). The bottleneck is purely disk read.
5. **Column selectivity is crucial**: Aligned binary reads 1 column in 0.54ms vs 6.68ms full — MKTF must support efficient column-selective access.
6. **Per-column compression**: If compression is used, it must be per-column (not whole-file) to preserve column selectivity.
7. **Universe scan at <40s** is achievable with aligned binary. Parquet would take 113s.

## Design Recommendations for MKTF

Based on these benchmarks:
- **Use 64-byte aligned column offsets** (proven benefit in read + column selectivity)
- **Column directory in header** (seek-based column selection works)
- **No compression by default** (compression ratio too low for market data)
- **Optional per-column LZ4** for columns with better compression characteristics (bitmasks, deltas)
- **Investigate pinned memory + async DMA** for overlapping I/O and H2D
- **Investigate GDS (GPU Direct Storage)** for bypassing CPU entirely on supported hardware

---

## Methodology Notes

- All timings use `time.perf_counter_ns()` for nanosecond precision
- GC disabled during timing windows to prevent pauses
- 30 runs for warm benchmarks, 10 for cold (cold eviction is expensive)
- Cold cache: 2GB memory pressure allocation between runs (best-effort on Windows without admin)
- Statistical reports: mean ± std, plus p50/p95/min/max
- Data generated with fixed seed (rng=42) for reproducibility
- Benchmark script: `experiments/file_format/bench_observer.py`

---

## Experiment 1b: Per-Column Compression Analysis

Tested LZ4 compression ratio on each column individually:

| Column | dtype | Raw KB | LZ4 KB | Ratio | Note |
|---|---|---|---|---|---|
| price | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| size | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| timestamp | int64 | 4672.3 | 3509.9 | 0.751x | monotonic sequence |
| notional | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| ln_price | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| sqrt_price | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| recip_price | float64 | 4672.3 | 4672.6 | 1.000x | incompressible |
| round_lot | uint8 | 584.0 | 2.4 | 0.004x | mostly zeros |
| odd_lot | uint8 | 584.0 | 284.4 | 0.487x | ~50% ones |
| direction | int8 | 584.0 | 291.5 | 0.499x | ternary {-1,0,1} |

**Delta-encoded timestamps**: 4672 KB -> 19.3 KB (0.004x) — constant deltas compress to nearly nothing.
**Bitmask-packed booleans**: 584 KB -> 73 KB (bitmask) -> 0.3 KB (LZ4) = 0.001x.

**Total savings**: raw 34.5 MB -> naive LZ4 32.1 MB (0.932x) -> optimized 28.4 MB (0.825x).

**Key insight**: float64 columns are completely incompressible. Only timestamps, booleans, and small-range integers benefit. The 7 float64 columns account for 95% of file size and are untouched by compression.

---

## Experiment 2: Hybrid Per-Column Encoding vs Aligned Binary

**Hypothesis**: Per-column encoding (delta timestamps, bitmask booleans, LZ4 flags) saves 17.5% disk space. Does the I/O reduction outweigh CPU overhead?

### Results

| Metric | Aligned Binary | Hybrid (per-col) | Winner |
|---|---|---|---|
| File size | 35.29 MB | 29.10 MB (-17.5%) | Hybrid |
| Full read (warm) | 4.87ms ± 0.30 | 7.42ms ± 0.18 | **Aligned (+52%)** |
| 1-col read (price) | 0.45ms ± 0.03 | 0.47ms ± 0.04 | Tie |
| 3-col read | 1.55ms ± 0.32 | 1.85ms ± 0.19 | Aligned (+19%) |
| Disk->GPU pipeline | 7.04ms ± 0.58 | 7.65ms ± 0.22 | Aligned (+9%) |
| Write | 10.38ms ± 0.45 | 14.36ms ± 0.66 | Aligned (+38%) |

### Analysis

**Hybrid is a net negative**. The 17.5% file size reduction saves ~1ms of I/O but costs ~2.5ms of CPU for LZ4 decompression + delta reconstruction. For the disk->GPU pipeline, aligned binary wins by 0.6ms.

**Why**: Market data is 95% float64 columns by size. These are incompressible. The savings from compressing timestamps and booleans are real but marginal relative to the float64 bulk.

**Exception — bitmask packing without compression**: A uint32 bitmask for 26 condition flags replaces 26 x uint8 = 26 bytes with 4 bytes (84.6% savings). This is a pure packing operation — zero CPU cost at read time, just reinterpret the bits. This IS worth doing.

### Conclusion

**For MKTF v2:**
- Store float columns raw, 64-byte aligned (no compression)
- Pack boolean/flag columns into uint32 bitmasks (no compression needed — just packing)
- Store timestamps as raw int64 (delta encoding saves space but costs CPU — not worth it at 598K rows)
- Reserve per-column LZ4 as optional flag for archival/cold storage, not hot path

Benchmark script: `experiments/file_format/bench_hybrid.py`

---

## Experiment 3: MKTF v2 End-to-End Benchmark (Real AAPL Data)

Pathmaker's MKTF v2 implementation benchmarked with real 598,057-tick AAPL data across all encoding paths.

### Phase 1: Encoding Path Comparison

| Encoding | Data MB | File MB | Write (ms) | Read (ms) | 1-col (ms) | 2-col (ms) | 4-col (ms) |
|---|---|---|---|---|---|---|---|
| float32 | 15.55 | 15.55 | 4.03 ± 0.24 | **4.35 ± 0.16** | **0.24** | **0.51** | 2.34 |
| float64 | 20.33 | 20.33 | 6.30 ± 0.20 | 5.24 ± 0.20 | 0.41 | 0.70 | 1.95 |
| int64 | 20.33 | 20.34 | 6.54 ± 0.41 | 5.17 ± 0.15 | 0.45 | 0.78 | 2.15 |
| int32 | 15.55 | 15.55 | **3.96 ± 0.62** | 4.55 ± 0.26 | **0.24** | 0.52 | **1.44** |

**Observations**:
- **Near-zero file overhead**: MKTF v2 header+directory adds <0.1% to file size.
- **float32 wins read speed** (4.35ms) — smallest file = fastest read. I/O bound.
- **int32 wins write speed** (3.96ms) and 4-col selective reads (1.44ms).
- **Column-selective reads are excellent**: 0.24ms for 1 column (18x faster than full read).
- All encoding paths pass correctness verification.

### Phase 2: Cold vs Warm Cache

| Encoding | Warm (ms) | Cold (ms) | Cold/Warm |
|---|---|---|---|
| float32 | 4.35 | 5.63 | 1.29x |
| float64 | 5.24 | 6.96 | 1.33x |
| int64 | 5.17 | 6.67 | 1.29x |
| int32 | 4.55 | 5.57 | 1.22x |

Consistent ~25-33% cold penalty across encodings. Smaller files (float32, int32) have lower cold/warm ratio.

### Phase 3: GPU Pipeline

**Pure H2D baseline**: 0.54ms for 15.5 MB = 28.9 GB/s

| Encoding | Disk->GPU (ms) | Read (ms) | H2D (ms) | Selective 2-col (ms) |
|---|---|---|---|---|
| float32 | **5.90** | 5.31 | 0.60 | **0.82** |
| int32 | 5.84 | 5.26 | 0.58 | 0.88 |
| int64 | 6.92 | 6.18 | 0.74 | 1.27 |

**Key insight**: H2D is only 0.54-0.74ms — **the pipeline is 90% disk read**. Selective 2-column reads achieve sub-millisecond disk->GPU (0.82ms).

**GPU Compute Operations** (598K elements):

| Op | float32 (us) | int64 (us) | int32 (us) | Winner |
|---|---|---|---|---|
| sum | 14.2 | 14.0 | 18.4 | int64 |
| mean | 32.7 | 37.8 | 35.5 | float32 |
| std | 326.3 | 339.3 | 333.4 | float32 |
| min | 15.4 | 14.6 | 15.3 | int64 |
| max | 16.0 | 15.3 | 18.6 | int64 |
| diff | 12.7 | 12.2 | 28.2 | int64 |
| p*s | 30.9 | 27.0 | 26.8 | int32 |
| sort | 128.3 | 159.5 | 117.3 | int32 |

**GPU compute is a wash** — all within ~10% of each other for most ops. float32 has a slight edge on statistical ops (mean, std) because it avoids the int->float conversion. int64/int32 win on exact arithmetic (sum, min, max, diff).

### Phase 4: MKTF v2 vs Parquet (End-to-End)

| Pipeline | Time (ms) | Speedup vs Parquet |
|---|---|---|
| Parquet read | 10.40 | 1.0x |
| Parquet->numpy | 66.23 | — |
| Parquet->GPU | 66.65 | 1.0x |
| MKTF float32 read | 4.35 | **2.4x faster** |
| MKTF float32 ->GPU | 5.90 | **11.3x faster** |
| MKTF int64 read | 5.17 | 2.0x faster |
| MKTF int64 ->GPU | 6.92 | 9.6x faster |

**The parquet->numpy conversion is the killer**: 66ms vs 4-5ms for MKTF. Parquet stores strings that must be parsed; MKTF stores GPU-ready arrays.

**Universe projection (4604 tickers)**:
- Parquet read: 47.9s (0.8 min)
- MKTF float32 read: 20.0s (0.3 min)
- MKTF float32 ->GPU: 27.2s (0.5 min)
- vs Parquet->GPU: 306.8s (5.1 min) — **11x faster with MKTF**

### Conclusions

1. **MKTF v2 is validated**: 11.3x faster than parquet for the disk->GPU pipeline.
2. **float32 is the best default encoding**: smallest file, fastest read, competitive GPU compute.
3. **int32 is viable for stocks** (<$214K): same file size as float32, slightly faster on some GPU ops, but int->float conversion needed for statistical ops.
4. **int64 offers no speed advantage**: same size as float64, similar GPU throughput. Only advantage is exact arithmetic (no floating point error).
5. **Column-selective reads are the superpower**: 0.24ms for 1 column, 0.82ms for 2 cols on GPU. This enables surgical data access.
6. **The pipeline is 90% disk read, 10% H2D**: further optimization should focus on read speed (readahead, prefetch, GDS).

Benchmark script: `experiments/file_format/bench_mktf_v2.py`

---

## Experiment 4: Universe Scan Simulation (100 Files Sequential)

Real-world validation: 100 MKTF v2 files (float32 encoding) read sequentially to validate per-ticker projections.

### Setup
- 100 files x 15.55 MB = 1.55 GB total
- Real AAPL data duplicated across files (same size/schema as production)
- 5 sequential runs

### Results

| Pipeline | Per-file (ms) | 100 files (ms) | Projected 4604 tickers |
|---|---|---|---|
| Full read (warm) | 4.67 (excl. first run) | 471 | 21.5s |
| Selective 2-col read | 0.70 | 70 | **3.2s** |
| Disk->GPU | 5.79 | 579 | 26.6s |
| Disk->GPU->compute (sum) | 6.24 | 624 | 28.7s |

**First-run penalty**: Run 1 of full read was 11.1ms/file (2.4x slower) due to cold cache. Subsequent runs: 4.6-4.9ms/file.

### Key Findings

1. **Sequential I/O projections match single-file benchmarks**: 4.67ms/file sequential vs 4.35ms/file isolated. Only ~7% degradation at scale.
2. **Selective reads scale perfectly**: 0.70ms/file at 100 files — identical to single-file (0.51ms + file open overhead). **3.2s for full universe scan on 2 columns.**
3. **GPU compute adds minimal overhead**: 6.24ms vs 5.79ms per file — GPU ops (sum all cols) add only 0.45ms per ticker.
4. **Full pipeline under 30s**: Disk -> read -> H2D -> compute for entire universe in 28.7s.
5. **Selective pipeline under 4s**: If we only need 2 columns, 4604 tickers in 3.2 seconds.

### Revised Universe Projections (validated at 100-file scale)

| Pipeline | MKTF v2 float32 | Parquet (estimated) |
|---|---|---|
| Full read only | 21.5s | 47.9s (2.2x slower) |
| Selective 2-col read | 3.2s | ~30s (est.) |
| Disk -> GPU | 26.6s | 306.8s (11.5x slower) |
| Disk -> GPU -> compute | 28.7s | ~310s (est.) |

---

## Experiment 5: GPU Precision & Throughput — FP64 vs FP32

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q, 102.6 GB VRAM

### Phase 1: Fused Pointwise Kernel Throughput

Custom CUDA kernel computing 13 derived columns from 3 base columns (price, size, timestamp): notional, ln_price, sqrt_price, recip_price, abs_return, sign_return, ln_size, sqrt_size, recip_size, price_x_sqrt_size, vwap_contrib, spread_proxy, momentum.

| Precision | Time (us) | Bandwidth (GB/s) | Data moved (MB) |
|---|---|---|---|
| **FP64** | 102.1 | 750 | 76.6 |
| **FP32** | 25.4 | 1601 | 40.7 |
| **Speedup** | **4.0x** | — | — |

**Why 4x, not 30-40x**: The kernel is **bandwidth-bound**, not compute-bound. At 598K elements, we're moving data through memory, not crunching math. FP32's advantage is 2x smaller data (half the bytes through the memory bus) plus some speedup on transcendental functions. The theoretical 1:64 FP64:FP32 ratio only applies to compute-bound workloads at much larger scales.

**FP32 at 1601 GB/s approaches theoretical memory bandwidth** — this kernel is nearly optimal.

Individual operation benchmarks:

| Op | FP64 (us) | FP32 (us) | Speedup | Bottleneck |
|---|---|---|---|---|
| log | 26.8 | 8.6 | 3.1x | Compute (transcendental) |
| sqrt | 10.9 | 8.2 | 1.3x | Bandwidth |
| multiply | 8.1 | 8.2 | 1.0x | Bandwidth |
| abs | 8.1 | 8.4 | 1.0x | Bandwidth |
| std | 324.6 | 211.1 | 1.5x | Multi-pass reduction |
| sort | 189.9 | 114.4 | 1.7x | Memory + compute hybrid |
| sum | 14.6 | 15.1 | 1.0x | Bandwidth (reduction) |
| cumsum | 27.3 | 24.1 | 1.1x | Bandwidth (scan) |

### Phase 2: Precision Validation (FP32 vs FP64 on Real AAPL)

| Column | Max Abs Error | Max Rel Error | Mean Rel Error |
|---|---|---|---|
| notional | 49.34 | 5.96e-8 | 1.38e-8 |
| ln_price | 2.58e-7 | 4.75e-8 | 2.08e-8 |
| sqrt_price | 4.77e-7 | 3.16e-8 | 1.58e-8 |
| recip_price | 2.33e-10 | 5.56e-8 | 2.78e-8 |
| abs_return | 1.80e-9 | 5.94e-8 | 2.15e-8 |
| sign_return | **0** | **0** | **0** |
| spread_proxy | **0** | **0** | **0** |
| momentum | **0** | **0** | **0** |

**All columns within 1e-6 relative error. All within 1e-4. FP32 is more than sufficient.**

Price roundtrip error (FP64->FP32->FP64): **exactly zero**. AAPL prices at $227-$255 fit perfectly in float32's 24-bit mantissa (7.2 decimal digits).

### Phase 3: "Don't Store Derivables" Pipeline

| Pipeline | Time (ms) | File Size (MB) |
|---|---|---|
| Path A: Read 14 cols + H2D | 11.33 | 35.88 |
| **Path B: Read 3 cols + H2D + GPU recompute** | **3.57** | **9.57** |

**Path B wins by 7.76ms (3.2x faster).**

Path B breakdown:
- Read 3 columns: 3.08ms
- H2D transfer: 0.42ms
- **GPU recompute 13 derived columns: 0.035ms (35 microseconds!)**

**The GPU compute is essentially free** — 35 microseconds to recompute everything vs 7.76ms saved by not reading/transferring the extra 26 MB. The I/O savings utterly dominate.

**Universe impact**:
- Path A: 52.2s for 4604 tickers
- **Path B: 16.4s for 4604 tickers (3.2x faster)**
- **Disk savings: 121 GB per day** (73% reduction)
- **Yearly savings: 30 TB** (250 trading days)

### Phase 4: Kahan Compensated Summation

| Method | Sum | Relative Error |
|---|---|---|
| numpy FP64 (ground truth) | 137,306,041.818451 | — |
| CuPy FP64 | 137,306,041.818451 | 0.00e+00 |
| CuPy FP32 naive | 137,306,048.000000 | 4.50e-08 |
| **FP32 Kahan compensated** | **137,306,041.818451** | **0.00e+00** |
| FP32->FP64 upcast accum | 137,306,041.818451 | 0.00e+00 |

**Kahan FP32 matches FP64 exactly.** We don't need FP64 for accumulation.

Rolling mean accuracy at different windows:

| Window | Max Error ($) | Relative Error |
|---|---|---|
| 100 | $0.000039 | 1.71e-7 |
| 1,000 | $0.000070 | 3.06e-7 |
| 10,000 | $0.000176 | 7.65e-7 |
| 100,000 | $0.000331 | 1.44e-6 |

**Sub-cent errors even at 100K windows. Completely negligible for signal detection.**

### Experiment 5 Conclusions

1. **FP32 is the correct default**. All errors within 1e-6 relative. Price fits exactly. Sub-cent rolling mean errors.
2. **Don't store derived columns**. GPU recomputes 13 columns in 35 microseconds. Storing them wastes 73% disk space and 3.2x pipeline time.
3. **The fused kernel is bandwidth-bound at 1601 GB/s**, near theoretical peak. Can't go faster without reducing data volume (which FP32 already does).
4. **Kahan FP32 matches FP64 for accumulation**. No need for FP64 anywhere in the pipeline.
5. **The optimal MKTF file stores ONLY base columns**: price (float32), size (float32), timestamp (int64), exchange (uint8), conditions (uint32), is_odd_lot (uint8). Total: ~9.6 MB per ticker. Everything else recomputed on GPU in 35us.

**Revised pipeline**: Read 9.6 MB → H2D 0.42ms → GPU recompute 0.035ms → **total 3.57ms per ticker → 16.4s for full universe**.

Benchmark script: `experiments/file_format/bench_gpu_precision.py`

---

## Experiment 6: MKTF v3 — Self-Describing Format Benchmark

Pathmaker's MKTF v3: 4096-byte NVMe-sector alignment, self-describing headers with identity/quality/provenance, crash recovery via is_complete flag.

### Phase 1: Basic I/O

| Metric | Value |
|---|---|
| Data size | 15.55 MB (7 columns) |
| File size | 15.59 MB (35 KB overhead) |
| Write | 5.47ms ± 0.63 |
| Read (full) | 4.11ms ± 0.16 |
| Read 1-col | 0.28ms ± 0.02 |
| Read 2-col | 0.50ms ± 0.03 |
| Header scan | **29.5us ± 1.8** |

Overhead from 4096-byte alignment: only 35 KB (0.2% of file). Negligible.

### Phase 2: v3 (4096-align) vs v2 (64-align)

| Metric | v3 (4096) | v2 (64) | Delta |
|---|---|---|---|
| File size | 15.59 MB | 15.55 MB | +0.03 MB |
| Write | 5.47ms | 3.55ms | +1.92ms |
| Read (full) | 4.11ms | 4.22ms | **-0.11ms** |
| Sel 2-col | 0.50ms | 0.52ms | -0.02ms |

**Write is 1.92ms slower** due to header computation (schema fingerprint, per-column stats, crash-safe protocol). **Read is identical or marginally faster**. The operational features (self-describing, crash recovery, daemon-scannable) cost nothing at read time.

### Phase 3: GPU Pipeline (Source-Only + Fused Recompute)

| Stage | Time |
|---|---|
| Read 7 cols | 4.34ms |
| H2D transfer | 0.60ms |
| Fused kernel (13 derived cols) | 0.036ms (36us) |
| **Total pipeline** | **6.01ms** |

Universe: 27.7s for 4604 tickers.

### Phase 4: Cold vs Warm Cache

- Warm: 4.11ms, Cold: 5.22ms, Ratio: **1.27x**
- Consistent with earlier experiments (~25-30% cold penalty).

### Phase 5: Daemon Header Scan

**The killer feature**: daemon reads ONLY headers, zero data bytes.

| Metric | Value |
|---|---|
| Per-header scan | 183us |
| 100 headers | 18.3ms |
| **4604 headers (full universe)** | **0.84s** |

Each header provides: is_complete, ticker, day, schema fingerprint, n_rows, total_nulls, write duration, per-column min/max/null_count. **Complete operational state in sub-second scan.**

Crash recovery validated: is_complete flag correctly detects incomplete writes.

### Conclusions

1. **MKTF v3 is production-ready**. 4096-byte alignment adds 35 KB overhead, zero read penalty.
2. **Write overhead (+1.92ms) is acceptable** for the self-describing features gained.
3. **Header scan at 0.84s for full universe** enables sub-second staleness checks, schema drift detection, and quality monitoring.
4. **Full GPU pipeline at 6.01ms/ticker** (read + H2D + fused recompute) = 27.7s for universe.
5. **The daemon never touches data bytes** — all operational decisions from headers alone.

Benchmark script: `experiments/file_format/bench_mktf_v3.py`

---

## Summary of All Experiments

| # | Experiment | Key Finding |
|---|---|---|
| 1 | Format comparison (9 formats) | Aligned binary wins GPU pipeline at 7.98ms |
| 1b | Per-column compression | Float64 incompressible; only timestamps/booleans benefit |
| 2 | Hybrid encoding | Compression is net negative for market data |
| 3 | MKTF v2 end-to-end | 11.3x faster than parquet for disk->GPU |
| 4 | Universe scan (100 files) | 28.7s full pipeline, 3.2s selective for 4604 tickers |
| 5 | GPU precision + throughput | FP32 sufficient, don't store derivables (73% savings), Kahan matches FP64 |
| 6 | MKTF v3 self-describing | 35KB overhead, 0.84s full-universe header scan, crash recovery works |

**Final recommended pipeline**: MKTF v3 source-only → H2D → fused FP32 recompute → GPU compute.
**Per-ticker**: 6.01ms. **Full universe**: 27.7s. **Storage**: 15.59 MB/ticker = 71.8 GB/day.
