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

## Experiment 7: Concurrent MKTF Reader — NVMe Saturation

Pathmaker's concurrent reader prototype: ThreadPoolExecutor overlapping I/O reads to saturate NVMe bandwidth. Navigator insight: 2.29 GB/s actual vs 7 GB/s theoretical — multiple concurrent reads should close the gap.

### Setup
- **Data**: 100 synthetic MKTF v3 files, 12.59 MB each, 1259 MB total (real AAPL data, different ticker IDs)
- **Runs**: 10 warm, 5 cold (2GB memory-pressure eviction), 10 GPU
- **Workers tested**: 1, 2, 4, 8, 12, 16

### Phase 1: Sequential Baseline (Warm)

| Metric | Value |
|---|---|
| Mean | 549.6ms (5.50 ms/file) |
| Std | 22.1ms |
| p50 / p95 | 539.4ms / 587.4ms |
| Bandwidth | 2.29 GB/s |

### Phase 2: Concurrent Read Sweep (Warm)

| Workers | Mean (ms) | ms/file | BW (GB/s) | Speedup |
|---|---|---|---|---|
| 1 | 507.9 ± 15.1 | 5.08 | 2.48 | 1.08x |
| 2 | 335.8 ± 6.5 | 3.36 | 3.75 | 1.64x |
| 4 | 281.7 ± 14.5 | 2.82 | 4.47 | 1.95x |
| 8 | 251.3 ± 18.5 | 2.51 | 5.01 | 2.19x |
| **12** | **224.1 ± 12.3** | **2.24** | **5.62** | **2.45x** |
| 16 | 238.1 ± 42.1 | 2.38 | 5.29 | 2.31x |

**Sweet spot: 8-12 workers.** 5.01-5.62 GB/s = 72-80% of NVMe theoretical. Diminishing returns beyond 8, and 16 workers adds variance (42ms std) from thread contention.

### Phase 3: Cold Cache

| Mode | Mean (ms) | ms/file | vs Warm |
|---|---|---|---|
| Sequential cold | 447.1 ± 56.0 | 4.47 | 0.81x |
| 4-worker cold | 259.4 ± 16.5 | 2.59 | 0.92x |
| Cold speedup | — | — | 1.72x |

**Note**: Cold is not slower than warm here — the 2GB memory-pressure eviction may be insufficient on this system (likely >64GB RAM). The OS cache retains the 1.26GB test dataset. In production with 71.8 GB universe data, cold reads will be the norm. The concurrent speedup (1.72x cold vs 2.45x warm) confirms concurrency helps even when I/O patterns are less favorable.

### Phase 4: Pipelined Read + GPU (Prefetch + Fused Kernel)

| Mode | Mean (ms) | ms/file | Speedup |
|---|---|---|---|
| Sequential read+GPU | 454.2 ± 15.6 | 4.54 | baseline |
| 2-worker pipeline | 243.4 ± 8.3 | 2.43 | 1.87x |
| **4-worker pipeline** | **209.5 ± 24.9** | **2.09** | **2.17x** |
| 8-worker pipeline | 209.1 ± 14.1 | 2.09 | 2.17x |

**Prefetch pattern**: ThreadPoolExecutor submits all reads, `as_completed()` feeds GPU pipeline. I/O and GPU compute overlap. **4 workers saturates** — 8 workers gives no additional speedup because GPU is not the bottleneck (fused kernel runs in 36us).

### Phase 5: Universe Projections (4,604 tickers)

| Scenario | Time | vs Sequential |
|---|---|---|
| Sequential warm read | 25.3s | baseline |
| Sequential warm read+GPU | 20.9s | — |
| **Concurrent warm (12w)** | **10.3s** | **2.45x** |
| Concurrent cold (4w) | 11.9s | 2.13x |
| NVMe theoretical (7GB/s) | 8.3s | ceiling |

**Navigator's target of 10s: achieved** (10.3s warm, within noise of target). At 5.62 GB/s we're at 80% of NVMe theoretical — the remaining 20% gap is likely OS I/O scheduling overhead, Python thread management, and numpy array allocation per file.

### Conclusions

1. **8-12 concurrent readers is optimal**. 5.0-5.6 GB/s sustained, 80% of NVMe theoretical.
2. **Pipelined prefetch+GPU at 4 workers** gives full I/O+compute overlap at 2.09 ms/file. GPU is free (36us kernel) — the pipeline is entirely I/O bound.
3. **Full universe in ~10s** with concurrent readers, down from 25.3s sequential. Meets Navigator's target.
4. **Cold cache penalty minimal** on this system due to large RAM. Production universe (71.8 GB) will exceed cache — expect ~20-30% cold penalty based on Experiment 1 data.
5. **Next frontier: GPUDirect/DirectStorage** to bypass CPU entirely. Current pipeline: NVMe → OS → Python → numpy → cupy H2D. DirectStorage: NVMe → GPU VRAM directly.

Benchmark scripts: `research/20260327-mktf-format/bench_concurrent.py` (pathmaker's prototype), `experiments/file_format/bench_concurrent.py` (observer's rigorous benchmark)

---

## Experiment 8: NVMe Queue Depth Sweep — Finding the Saturation Knee

Naturalist insight: NVMe throughput is heavily queue-depth dependent. Experiment 7 tested 1-16 workers; this extends to 64 to find the true saturation point.

**Note**: Run on R595 driver (updated from R581 mid-session). Results show different absolute bandwidth than Experiment 7 — driver update is the likely cause.

### Setup
- **Data**: 200 MKTF v3 files, 12.59 MB each, 2517 MB total
- **Runs**: 5 per configuration
- **Workers tested**: 1, 2, 4, 8, 12, 16, 24, 32, 48, 64

### Phase 1: Read-Only Queue Depth Sweep

| Workers | Mean (ms) | Std | ms/file | BW (GB/s) | Speedup | % NVMe Theoretical |
|---|---|---|---|---|---|---|
| seq | 959.2 | 11.2 | 4.80 | 2.62 | 1.00x | 37% |
| 1 | 1269.4 | 65.9 | 6.35 | 1.98 | 0.76x | 28% |
| 2 | 830.1 | 58.6 | 4.15 | 3.03 | 1.16x | 43% |
| 4 | 749.2 | 23.7 | 3.75 | 3.36 | 1.28x | 48% |
| **8** | **719.8** | **31.5** | **3.60** | **3.50** | **1.33x** | **50%** |
| 12 | 709.7 | 11.5 | 3.55 | 3.55 | 1.35x | 51% |
| 16 | 715.0 | 23.5 | 3.57 | 3.52 | 1.34x | 50% |
| **24** | **696.8** | **13.4** | **3.48** | **3.61** | **1.38x** | **52%** |
| 32 | 707.8 | 26.4 | 3.54 | 3.56 | 1.36x | 51% |
| 48 | 743.9 | 61.3 | 3.72 | 3.38 | 1.29x | 48% |
| 64 | 727.6 | 26.6 | 3.64 | 3.46 | 1.32x | 49% |

**Saturation knee at 8 workers.** Marginal gains drop below 5% after 8 workers. Peak at 24 workers (3.61 GB/s) is only 3% above 8 workers. Beyond 32, thread contention causes regression.

### Phase 2: Pipelined Read + GPU Sweep

| Workers | Mean (ms) | Std | ms/file | Speedup |
|---|---|---|---|---|
| seq | 995.7 | 27.2 | 4.98 | 1.00x |
| 2 | 512.2 | 5.3 | 2.56 | 1.94x |
| **4** | **448.6** | **10.8** | **2.24** | **2.22x** |
| 8 | 619.3 | 33.6 | 3.10 | 1.61x |
| 12 | 699.2 | 8.8 | 3.50 | 1.42x |
| 16 | 743.2 | 31.5 | 3.72 | 1.34x |
| 24 | 764.4 | 15.2 | 3.82 | 1.30x |
| 32 | 777.8 | 40.9 | 3.89 | 1.28x |
| 48 | 789.7 | 20.8 | 3.95 | 1.26x |
| 64 | 805.7 | 20.1 | 4.03 | 1.24x |

**Critical finding: Pipelined GPU peaks at 4 workers, then DEGRADES.** With `as_completed()` feeding a single-threaded GPU pipeline, more workers means more completion events arriving simultaneously. The main thread becomes a serialization bottleneck — it can't consume GPU work fast enough. Beyond 8 workers, pipelined mode is actually *slower* than pure concurrent read.

### Saturation Analysis

| Workers | BW (GB/s) | % Theoretical | Marginal Gain |
|---|---|---|---|
| 1 | 1.98 | 28% | -24% (thread overhead) |
| 2 | 3.03 | 43% | +53% |
| 4 | 3.36 | 48% | +11% |
| **8** | **3.50** | **50%** | **+4%** (knee) |
| 12 | 3.55 | 51% | +1% |
| 24 | 3.61 | 52% | +2% |
| 32+ | ~3.5 | ~50% | diminishing/negative |

**We're capped at ~52% of NVMe theoretical (3.6 of 7 GB/s).** ~~Originally attributed to Python GIL~~ — Experiment 10 proved the R595 driver's internal I/O coalescing is the cause (see below).

### Experiment 7 vs 8: Driver Effect

| Metric | Exp 7 (R581) | Exp 8 (R595) |
|---|---|---|
| Sequential BW | 2.29 GB/s | 2.62 GB/s |
| Peak concurrent BW | 5.62 GB/s | 3.61 GB/s |
| Best speedup | 2.45x | 1.38x |

The R595 driver shows **higher sequential bandwidth but lower concurrent scaling**. The new driver may batch I/O differently or have different memory management. Sequential improved 14%, but concurrent scaling dropped from 2.45x to 1.38x. Net effect: less benefit from threading.

### Conclusions

1. **Saturation knee at 8 workers** for read-only. Beyond that, marginal gains < 4%.
2. **Pipelined GPU peaks at 4 workers** — more workers degrades performance due to serialization.
3. ~~52% ceiling is Python/GIL~~ **Corrected by Experiment 10**: the R595 driver's I/O coalescing is the bottleneck, not the GIL.
4. **Driver matters**: R595 vs R581 changed the concurrent scaling profile significantly. Always re-benchmark after driver updates.
5. **Next step for higher bandwidth**: native reader (Rust/C++) or DirectStorage, not more Python threads.

Benchmark script: `experiments/file_format/bench_queue_depth.py`

---

## Experiment 9: MKTF v4 (FinTek Production) vs v3 (WinRapids Prototype)

The v3 prototype graduated to v4 production in `R:/fintek/trunk/backends/mktf/`. V4 adds crash-safe writes (2x fsync + atomic rename), SHA-256 data checksum, EOF status bytes, and a much richer header (13 sections). This experiment measures the cost of those production features.

### Setup
- **Data**: 598,057 rows, 5 source columns, 12.56 MB
- **v3**: `research/20260327-mktf-format/mktf_v3.py` (JSON header, direct write)
- **v4**: `R:/fintek/trunk/backends/mktf/` (struct-packed header, crash-safe write)
- **Runs**: write=10, read=30, scan=50

### Phase 1: Write Comparison

| Version | Write (ms) | File Size |
|---|---|---|
| v3 | 4.66 ± 0.13 | 12.58 MB |
| **v4** | **18.72 ± 0.41** | **12.59 MB** |

**v4 write is 4.02x slower.** The 14ms overhead breaks down as:
- 2x `fsync()`: ~6ms each (NVMe flush-to-media guarantee)
- SHA-256 over 12.5MB data: ~2ms
- Atomic rename (.tmp → .mktf): <1ms

This is the price of crash safety. But writes happen once per ticker; reads happen many times. The write path is not the hot path.

### Phase 2: Full Read

| Version | Read (ms) | Delta |
|---|---|---|
| v3 | 4.56 ± 0.22 | — |
| **v4** | **2.97 ± 0.24** | **-35%** |

**v4 reads 35% faster than v3.** v3 parses a JSON header (variable-length, string-heavy); v4 uses `struct.unpack_from()` on fixed-offset fields (pure C extension, zero string allocation). The richer header (13 sections) costs nothing because struct unpacking is orders of magnitude faster than JSON parsing.

### Phase 3: Header-Only Read

| Version | Header (us) |
|---|---|
| v3 | 32.9 ± 6.4 |
| v4 | 46.0 ± 10.2 |

V4 header is 40% slower to parse (13 sections vs v3's simpler structure). Both sub-50us — negligible in context of ms-scale I/O.

### Phase 4: Selective Read

| Columns | Time (ms) |
|---|---|
| price + size (2 cols) | 1.09 ± 0.13 |

Column-selective reads work correctly through the v4 directory.

### Phase 5: EOF Status Fast Path

| Operation | Time (us) |
|---|---|
| read_status (2 bytes at EOF) | 27.8 ± 3.3 |
| is_dirty (1 byte at EOF) | 27.9 ± 6.2 |
| is_complete (1 byte at EOF-1) | 26.4 ± 1.7 |
| v3 header scan (4096 bytes) | 32.9 ± 6.4 |

**Only 1.2x faster per-file** — at sub-50us scale, file-open + seek overhead dominates, not data size. But the real win is at scale: the daemon never touches data bytes, never parses headers, never allocates numpy arrays.

### Phase 6: Data Integrity Verification

| Operation | Time (ms) |
|---|---|
| verify_checksum | 8.15 ± 0.15 |

Re-reads all column data + SHA-256 hash. Expensive, but this is an audit/transfer verification path, not a hot path. Checksum verified correct (True).

### Phase 7: Daemon Scan at Scale (200 files)

| Scan Type | Total (ms) | Per-File (us) | Universe (4604) |
|---|---|---|---|
| scan_dirty | 7.39 ± 0.81 | 37 | **0.17s** |
| scan_incomplete | 7.17 ± 0.82 | 36 | **0.17s** |
| header_scan | 10.34 ± 0.80 | 52 | **0.24s** |

**Daemon can check all 4604 files for dirty/incomplete status in 170ms.** That's 6 full sweeps per second. The v3 header scan (0.84s in Experiment 6, now 0.24s) was already fast; EOF-byte scanning cuts it another 30%.

### Conclusions

1. **v4 write is 4x slower** — the cost of crash safety (fsync, checksum, atomic rename). Acceptable: writes are the slow path, reads dominate.
2. **v4 read is 35% faster** — struct.unpack beats JSON parsing. The production spec is faster than the prototype on the hot path.
3. **Daemon scan: 170ms for full universe** — 6 sweeps/second. Fast enough for real-time operational monitoring.
4. **Data integrity: 8.15ms** to verify checksum for one file. Audit path, not hot path.
5. **The richer header (Tree, Asset, Statistics, Spatial, etc.) costs nothing at read time.** All that metadata is free once you commit to fixed-offset struct packing.

Benchmark script: `experiments/file_format/bench_mktf_v4.py` (runs via fintek venv)

---

## Experiment 10: Driver Regression Isolation (R581 vs R595)

Experiments 7 and 8 showed dramatically different concurrent scaling. Two variables changed: driver version (R581→R595) and file count (100→200). This experiment isolates the variable by running the EXACT Experiment 7 config (100 files) on the R595 driver.

### Results

| Workers | Exp 7 (R581, 100 files) | Exp 10 (R595, 100 files) |
|---|---|---|
| seq | 2.29 GB/s | 2.93 GB/s |
| 4 | 4.47 GB/s | 3.36 GB/s |
| 8 | 5.01 GB/s | 3.52 GB/s |
| 12 | **5.62 GB/s (2.45x)** | **3.55 GB/s (1.21x)** |
| 16 | 5.29 GB/s | 3.54 GB/s |

### Verdict: It Was the Driver

Same file count, same data, same hardware. The R595 driver:
- **Sequential: +28%** (2.29 → 2.93 GB/s) — improved single-threaded I/O
- **Concurrent scaling: halved** (2.45x → 1.21x) — threading barely helps now
- **Peak concurrent: -37%** (5.62 → 3.55 GB/s) — worse absolute throughput

**Root cause**: The R595 driver performs aggressive I/O coalescing internally. It batches and reorders read requests at the driver level, doing what ThreadPoolExecutor was doing for us on R581. Multiple Python threads submitting reads no longer provides additional I/O parallelism — the driver already handles it.

### Correction to Experiment 8

Experiment 8's conclusion that "Python GIL is the bottleneck" was **wrong**. The GIL adds overhead, but the dominant factor is the driver's I/O behavior. On R581, threading gave 2.45x because the driver dispatched reads serially; on R595, the driver coalesces internally, making threading redundant.

### Implications

1. **Concurrent reader benefit is driver-dependent.** R581: 2.45x. R595: 1.21x. Don't hardcode worker counts.
2. **Sequential I/O improved 28%** — the driver update was a net positive for the typical single-threaded Python read path.
3. **Adaptive concurrency**: The pipeline should auto-tune worker count based on measured speedup, not assume threading helps.
4. **Re-benchmark after every driver update.** This isn't optional — it's a 37% throughput difference.

Benchmark script: `experiments/file_format/bench_driver_isolation.py`

---

## Experiment 11: MKTF v4 vs Parquet — The Production Comparison

The benchmark that justifies the format migration. MKTF v4 (crash-safe, self-describing) vs Parquet (the industry standard) across every dimension.

### Setup
- **Data**: 598,057 rows, 5 source columns (price f32, size f32, timestamp i64, exchange u8, conditions u32), 12.56 MB
- **Parquet variants**: none/snappy/zstd compression
- **Readers**: MKTF v4 reader, PyArrow, Polars (with scan_parquet projection pushdown)
- **Runs**: write=10, read=30, header/status=100

### Phase 1: Write Speed

| Format | Write (ms) | File Size |
|---|---|---|
| **MKTF v4 (crash-safe)** | **18.4 ± 0.3** | **12.59 MB** |
| Parquet (none) | 49.3 ± 2.4 | 7.35 MB |
| Parquet (snappy) | 56.3 ± 0.5 | 5.54 MB |
| Parquet (zstd) | 60.6 ± 0.6 | 4.08 MB |

**MKTF v4 writes 2.7-3.3x faster than parquet** — despite including 2x fsync, SHA-256 checksum, and atomic rename that parquet doesn't do. Parquet's write overhead is schema encoding, row group assembly, column statistics, and Thrift serialization. MKTF just memcpy's arrays + struct.pack a header.

File size: MKTF is 1.7-3.1x larger (no compression). This is by design — GPU wants raw arrays, not decompressed bytes.

### Phase 2: Full Read

| Reader | Read (ms) |
|---|---|
| **MKTF v4** | **2.9 ± 0.3** |
| PyArrow parquet (none) | 4.3 ± 0.4 |
| PyArrow parquet (snappy) | 5.6 ± 1.1 |
| PyArrow parquet (zstd) | 7.1 ± 1.1 |
| Polars parquet (none) | 6.5 ± 0.3 |
| Polars parquet (snappy) | 10.3 ± 1.9 |

**MKTF: 1.5x faster than PyArrow, 2.2x faster than Polars.** MKTF reads are `seek + read + np.frombuffer` — zero deserialization. Parquet must decode row groups, reconstruct columns from pages, validate checksums, and build Arrow arrays.

### Phase 3: Selective Read (Projection Pushdown)

| Reader | 2-col Read (ms) |
|---|---|
| **MKTF v4 selective** | **1.2 ± 0.2** |
| PyArrow columns= | 2.5 ± 0.4 |
| PyArrow snappy columns= | 2.6 ± 0.2 |
| Polars scan_parquet .select | 3.0 ± 0.6 |
| Polars scan_parquet snappy | 3.6 ± 0.3 |

**MKTF: 2.1x faster than PyArrow, 2.6x faster than Polars** with projection pushdown. MKTF's selective read is two seeks (one per column, guided by the column directory). Parquet's projection pushdown still reads row group metadata, locates column chunks, and deserializes pages.

### Phase 4: Header-Only Read

| Operation | Time (us) |
|---|---|
| **MKTF v4 header** | **48.8 ± 21.8** |
| Parquet metadata | 111.1 ± 96.8 |

**2.3x faster, and lower variance.** MKTF: read 4096 bytes + 5×128 byte directory = 4736 bytes, struct.unpack. Parquet: read footer, parse Thrift metadata, decode column chunk info.

Header contents readable: 598,057 rows, 5 cols, ticker=AAPL, day=2025-09-02, is_complete=True, is_dirty=False.

### Phase 5: Status Byte Read (Daemon Fast Path)

| Operation | Time (us) |
|---|---|
| read_status (2 bytes) | 29.0 ± 8.9 |
| is_dirty (1 byte) | 29.8 ± 6.9 |
| is_complete (1 byte) | 28.1 ± 5.6 |

**33,553 status checks per second. Full universe (4604 tickers): 137ms.**

Parquet has no equivalent. To check if a parquet file is "complete" you must read the footer and verify the magic bytes. MKTF's EOF status byte is the single fastest operational check possible.

### Phase 6: GPU Pipeline (Read → H2D → Fused Kernel)

| Pipeline | Time (ms) | Speedup |
|---|---|---|
| **MKTF v4 → GPU** | **3.2 ± 0.2** | **2.7x** |
| Parquet → GPU | 8.8 ± 0.6 | baseline |

**2.7x faster end-to-end to GPU.** MKTF arrays are already numpy (GPU-ready). Parquet → Arrow → numpy → cupy has two extra conversion layers.

### Universe Projections (4604 tickers)

| Pipeline | Time |
|---|---|
| **MKTF v4 full read** | **13.3s** |
| Parquet(none) PyArrow | 19.7s |
| Polars(none) read | 29.8s |
| **MKTF v4 → GPU pipeline** | **14.9s** |
| Parquet → GPU pipeline | 40.6s |

### Conclusions

1. **MKTF v4 is faster than parquet in every dimension**: write (2.7x), read (1.5-2.2x), selective (2.1-2.6x), header (2.3x), GPU pipeline (2.7x).
2. **Crash safety is free on the read path.** v4's write-time overhead (fsync, checksum, atomic rename) doesn't appear at read time.
3. **Parquet's compression doesn't help.** The compressed files are smaller but slower to read. For GPU pipelines, raw arrays win.
4. **The daemon fast path has no parquet equivalent.** 137ms to scan 4604 files for dirty/incomplete status. Parquet can't do this.
5. **Full universe GPU pipeline: 14.9s (MKTF) vs 40.6s (parquet) = 2.7x.** This is the production number that matters.

Benchmark script: `experiments/file_format/bench_v4_vs_parquet.py` (runs via fintek venv)

---

## Experiment 12: Micro-Anatomy of a v4 Read

### Setup
- **Goal**: Where does the time go in read_columns()? Is Python the bottleneck?
- **Data**: Same AAPL file as Experiment 11 (598,057 rows, 5 columns, 12.59 MB)
- **Runs**: 50 per micro-benchmark, GC disabled, warm cache
- **Discovery**: read_columns() opens the file TWICE (once in read_header, once for data)

### Phase 1: Time Budget Decomposition

| Operation | Time (us) | % of Total |
|---|---|---|
| **I/O: Block 0 + dir + data** | **2841** | **97%** |
| CPU: unpack_block0 + unpack_dir + frombuffer | 12 | 0.4% |
| 2nd file open penalty | ~25 | 1% |
| Unaccounted (Python/dict overhead) | 66 | 2% |
| **TOTAL read_columns()** | **2919** | **100%** |

**The Python reader is 97% I/O-bound.** Only 12us of CPU overhead (struct.unpack + numpy) out of 2919us total. Python is not the bottleneck.

### Phase 2: Read Strategy Comparison

| Strategy | Mean (us) | BW (GB/s) | Speedup |
|---|---|---|---|
| Production (2 opens, seek/col) | 2703 ±201 | 4.7 | 1.00x |
| Single open (1 open, seek/col) | 2683 ±180 | 4.7 | 1.01x |
| Bulk read (1 open, 1 read) | 4495 ±274 | 2.8 | 0.60x |
| Memory-mapped (mmap + slice) | 56 ±17 | 226* | 48.6x |

\* Mmap is deceptive — 56us is just creating the mapping. Data isn't read until accessed. Experiment 1 showed mmap has 2.49x cold/warm penalty, worst of any strategy.

**Surprise: bulk read is 40% SLOWER.** Reading the entire 12.59 MB file at once and slicing is slower than 5 seek+read calls. Hypothesis: large Python bytes allocation overhead + memory copy for slicing.

### Phase 3: I/O Pattern Analysis

| Pattern | Time (us) |
|---|---|
| 5 seeks + 5 reads | 2612 ±135 |
| Sequential from data_start | 2525 ±99 |
| Entire file (1 read call) | 2589 ±137 |

Seek overhead is only 87us (3%). At 12.6 MB scale, NVMe seeks are essentially free. The current seek-per-column design is correct.

### Phase 4: Compiled Reader Floor Estimate

| Metric | Value |
|---|---|
| I/O floor (single read) | 2589us = 2.59ms |
| Python overhead above I/O | 330us (11%) |
| I/O bandwidth at floor | 4.86 GB/s |
| NVMe theoretical max | ~7.0 GB/s |
| **I/O floor / NVMe** | **69%** |
| Estimated Rust reader | 2718us (~7% faster than Python) |

**The gap between Python and NVMe theoretical is Windows kernel + NTFS overhead, not Python.** Even a single raw `read()` system call only achieves 69% of NVMe theoretical. A compiled Rust reader would gain at most 7% over Python.

### Phase 5: Header Unpack Cost

| Operation | Time (us) |
|---|---|
| Raw struct.unpack_from (×12 calls) | 1.6 |
| Full unpack_block0 (13 sections, 60+ fields) | 6.4 |
| Dataclass creation + string decode | 4.8 (75% of unpack) |

The 13-section Block 0 header with 60+ fields parses in **6.4 microseconds**. This is "free metadata" — confirmed.

### Universe Projections (4604 tickers)

| Reader | Time |
|---|---|
| Python read_columns (production) | 13.4s |
| Single-open Python | 12.4s |
| Estimated Rust reader | 12.5s |
| I/O floor | 11.9s |

### Conclusions

1. **Python is not the bottleneck.** 97% of read time is I/O. Rewriting the reader in Rust gains ~7%.
2. **Windows kernel limits us to 69% NVMe.** The 31% gap is NTFS + kernel I/O stack overhead per file open. No userspace optimization (Python or Rust) can close this.
3. **The double file open is basically free.** 25us out of 2919us (1%). Not worth refactoring.
4. **Seek-per-column is correct.** Only 3% overhead vs sequential. NVMe seeks are free at this file scale.
5. **Bulk read is a trap.** Reading the entire file and slicing is 40% slower than targeted seeks. Don't do this in the Rust reader.
6. **The real optimization path is not a faster reader — it's I/O scheduling.** Async I/O (Windows IOCP), read-ahead, or memory-mapped batch access would close more of the gap than language-level optimization.
7. **Header parsing is 6.4us.** Rich metadata (60+ fields, 13 sections) costs essentially nothing. Add more metadata freely.

Benchmark script: `experiments/file_format/bench_read_anatomy.py` (runs via fintek venv)

---

## Experiment 13: I/O Strategy Sweep for Universe Scans

### Setup
- **Goal**: Find the best I/O strategy for reading 4604 MKTF files sequentially
- **Files**: 100 × 12.59 MB = 1.26 GB total (warm cache from warmup)
- **Strategies**: Buffered, raw, O_SEQUENTIAL, mmap, pre-opened handle pool, threaded

### Phase 1: Single-File Read Strategies

| Strategy | Time (us) | BW (GB/s) |
|---|---|---|
| Pre-opened buffered (seek+read) | 2495 ±109 | 5.04 |
| Pre-opened raw fd (lseek+read) | 2506 ±111 | 5.02 |
| Raw + O_SEQUENTIAL | 2573 ±130 | 4.89 |
| Python raw (os.open+read+close) | 2608 ±194 | 4.83 |
| Python buffered (open+read+close) | 2705 ±178 | 4.65 |
| mmap (forced fault, all pages) | 3084 ±371 | 4.08 |
| mmap (lazy, header only) | 46 ±12 | 276* |

\* Lazy mmap: deceptive. Only maps virtual memory. Data faulted on access.

**File open overhead: 210us = 8% of read.** Pre-opened handles save this.
Raw I/O is only ~100us faster than buffered — Python's buffer overhead is minimal.
**mmap with forced faults is SLOWEST for real reads** (4.08 GB/s) — page fault overhead costs more than direct read.

### Phase 2: Universe Scan (100 files, warm cache)

| Strategy | ms/file | BW (GB/s) | Speedup |
|---|---|---|---|
| **Threaded 8w (production)** | **1.30** | **9.70** | **2.54x** |
| Production (open/read/close) | 3.30 | 3.82 | 1.00x |
| Raw fd pool | 4.67 | 2.70 | 0.71x |
| Pre-opened handle pool | 4.72 | 2.67 | 0.70x |
| mmap pool (lazy — deceptive) | 0.01 | 1009* | 264x |

**SURPRISE: Handle pools are SLOWER.** Same trap as Experiment 12's bulk read — pools use `f.read()` (entire file) which is 40% slower than seek-per-column. The open/close overhead (210us) is less than the bulk-read penalty (~1400us).

**Threading: 2.54x on warm cache.** But this is page cache bandwidth (9.70 GB/s > 7.0 GB/s NVMe max). On cold cache, Experiment 10 showed only 1.21x due to R595 driver I/O coalescing.

### Phase 3: Universe Projections (4604 tickers)

| Strategy | Time | Notes |
|---|---|---|
| **Threaded 8w (warm)** | **6.0s** | Page cache — not reproducible on cold start |
| Production sequential | 15.2s | Realistic cold-start baseline |
| Threaded 8w (cold, via Exp 10) | ~12.5s | 1.21x from NVMe coalescing |
| Handle pool (with bulk read) | 21.7s | SLOWER than production — don't do this |

### Phase 4: Handle Pool Feasibility

- Windows can hold 4604+ file handles (tested 100, 24.5us/handle to open)
- Pool re-read (seek+read) is NOT faster than fresh read (0.3% slower — within noise)
- Projected: 13.2s for 4604 handles = 4.40 GB/s. Modest improvement but pools interact badly with bulk reads.

### Conclusions

1. **Handle pools are a trap** when combined with bulk read. The bulk read penalty (40%) swamps the open overhead savings (8%).
2. **The optimal single-file strategy is pre-opened with seek-per-column**: 5.04 GB/s. But the savings vs production (8%) aren't worth the handle management complexity.
3. **Threading helps on warm cache (2.54x) but not cold (1.21x, Experiment 10).** The NVMe is the real bottleneck on cold start.
4. **Raw I/O provides marginal improvement** (~100us, ~4%) over Python buffered. Not worth the ergonomic cost.
5. **mmap is the wrong model for sequential scan.** Forced page faults are slower than direct reads. Lazy mmap is deceptive.
6. **The production reader's design is already near-optimal.** open/close per file with seek-per-column is the right pattern. The ~12-15s universe scan time is fundamentally I/O-limited.
7. **For cold-start optimization, the lever is NVMe queue depth + prefetch**, not reader code. The R595 driver's internal coalescing limits Python-level threading gains. A kernel-level prefetch (Windows ReadAhead or IOCP with large queues) would be the next thing to try.

Benchmark script: `experiments/file_format/bench_io_strategies.py` (runs via fintek venv)

---

## Experiment 14: Micro-Anatomy of a v4 Write

### Setup
- **Goal**: Where does the write time go? How does it scale with file size? What about tiny cadence bins?
- **Data**: AAPL source (598K rows, 12.6 MB) + synthetic cadence bins (13 to 23400 rows)
- **Runs**: 20 per benchmark

### Phase 1: Time Budget (full-size, 12.6 MB)

| Operation | Time (us) | % of Total |
|---|---|---|
| SHA-256 hash (CPU) | 4782 | 25% |
| 2× fsync (NVMe flush) | ~4869 | 25% |
| Raw write I/O | 3038 | 16% |
| arr.tobytes() ×5 (copy) | 1430 | 7% |
| Atomic rename | 591 | 3% |
| pack_block0 + directory | 7 | 0% |
| Unaccounted (Python/Path) | 4658 | 24% |
| **TOTAL write_mktf()** | **19376** | **100%** |

Unlike reads (97% I/O), writes are split: 32% CPU (hash + memory copy), 25% fsync, 16% I/O, 24% Python overhead.

### Phase 2: Write Cost vs File Size

| Size | Rows | Data | Write (ms) |
|---|---|---|---|
| 30min cadence | 13 | 0.3 KB | 2.86 |
| 5min cadence | 78 | 1.6 KB | 3.01 |
| 1min cadence | 390 | 8.0 KB | 2.72 |
| 10s cadence | 2340 | 48.0 KB | 3.18 |
| 1s cadence | 23400 | 480 KB | 3.52 |
| Full AAPL | 598057 | 12.3 MB | 19.38 |

**Write cost is nearly CONSTANT for small files** (~2.8-3.2ms from 13 to 2340 rows). The fsync floor dominates. Data size only matters above ~500 KB.

### Phase 3: fsync Cost Isolation (tiny file, 273 bytes data)

| Variant | Time (us) |
|---|---|
| No fsync | 142 |
| 1× fsync | 1101 |
| 2× fsync | 1990 |
| Full production | 2981 |

**Per-fsync cost: ~960us = ~1ms** (NVMe flush-to-media latency). **fsync is 62% of tiny file write cost.**

### Phase 4: No-fsync Writer (batch recompute mode)

| Size | Production | No-fsync | Speedup |
|---|---|---|---|
| 30min cadence (13 rows) | 2.85ms | 0.35ms | **8.1x** |
| 5min cadence (78 rows) | 3.04ms | 0.23ms | **13.4x** |
| 1min cadence (390 rows) | 2.98ms | 0.20ms | **14.7x** |
| Full AAPL (598K rows) | 18.76ms | 4.42ms | **4.2x** |

No-fsync gives **8-15x speedup for small files, 4x for large files.**

### Phase 5: Cadence Grid Write Projections (4604 tickers × 10 cadences)

| Mode | Total Time | Files |
|---|---|---|
| **Production (2× fsync)** | **140s = 2.3 min** | 46,040 |
| **No-fsync** | **12s = 0.2 min** | 46,040 |
| **Savings** | **91%** | — |

All cadences cost ~2.8-3.2ms production regardless of data size. The cadence grid is fundamentally **46,040 × ~3ms fsync = 138s.** No Rust optimization helps — the bottleneck is NVMe flush-to-media latency.

### Design Recommendation: Dual Write Mode

The writer should support two modes:
1. **Crash-safe (production)**: For ingest/source files. 2× fsync + atomic rename. Used when data is irreplaceable.
2. **Batch (recompute)**: For derived outputs. No fsync, no rename. Used when crash recovery = full recompute from source.

This is a design decision, not an optimization. The source files (K01) should always use crash-safe writes. Derived cadence outputs (K02+) can safely skip fsync because they're recomputable.

Benchmark script: `experiments/file_format/bench_write_anatomy.py` (runs via fintek venv)

---

## Experiment 15: Progressive Section + Dual Write Mode (production v4)

**Date**: 2026-03-28
**Motivation**: Pathmaker wired progressive sufficient statistics into the production MKTF stack (format.py, writer.py, reader.py). This experiment benchmarks the progressive section's write overhead, read costs, correctness, and universe impact — using the production writer with 10 cadence levels matching detrended spectral findings.

**Setup**: 598,057-row AAPL file, 5 columns (timestamp, price, size, exchange, conditions). 10 progressive cadences: 1s, 5s, 10s, 30s, 1min, 5min, 10min, 15min, 20min, 30min. 31,765 total bins across all cadences. 5 sufficient stats per bin per column (sum, sum_sq, min, max, count).

### Phase 1: Write Cost Comparison (20 runs)

| Config | Write Time | File Size | Overhead |
|---|---|---|---|
| No progressive, safe=True (baseline) | 23.04ms ±2.51 | 12,292 KB | — |
| No progressive, safe=False | 15.52ms ±0.61 | 12,292 KB | 0 |
| With progressive, safe=True | 59.68ms ±2.97 | 15,394 KB | +3,102 KB |
| With progressive, safe=False | 54.42ms ±1.39 | 15,394 KB | +3,102 KB |

**Key finding**: Progressive section adds ~37ms write overhead (2.6x slower). This is dominated by **data volume** (3.1 MB of stats), not fsync. The safe vs batch difference shrinks from 48% (without progressive) to 10% (with progressive) because progressive data volume dwarfs the fsync cost.

### Phase 2: Read Cost Comparison (20 runs)

| Operation | Time | Notes |
|---|---|---|
| Full column read (no progressive) | 4,046us ±399 | Baseline |
| Full column read (with progressive) | 4,072us ±509 | +26us (0.7%) |
| **Progressive summary (dir only)** | **83us ±31** | **49x faster than full read** |
| Progressive level (5min, 78 bins) | 1,162us ±118 | Typical cadence |
| Progressive level (30min, 13 bins) | 926us ±138 | Coarsest useful cadence |
| Progressive level (1s, 23,400 bins) | 61,012us ±2,401 | Finest cadence — expensive |
| Header only (either variant) | 43-44us ±5-7 | Unaffected |

**Key finding**: Progressive section adds zero measurable overhead to full column reads (+0.7%, within noise). The progressive summary read (83us) enables **49x speedup** for coarse K04 vs full column read. Fine-grained levels (1s) are expensive because 23,400 bins × 5 stats × 5 cols = 2.3 MB of data.

### Phase 3: Correctness Verification

- Column data: IDENTICAL (progressive section doesn't affect column layout)
- Checksums: PASS for both variants (data_checksum covers only column data, not progressive)
- Progressive summary: 10 levels, 5 columns, MI scores roundtrip correctly
- Level data roundtrip: MATCH (5min level verified)
- Non-progressive file: `read_progressive_summary()` returns None (graceful degradation)

### Phase 4: Size Analysis

| Component | Size |
|---|---|
| Column data | 12,292 KB |
| Progressive stats | 3,102 KB (25.2% of file) |
| Stats breakdown | 31,765 bins × 5 cols × 5 stats × 4 bytes = 3,102 KB |
| Directory overhead | 284 bytes |
| Alignment padding | ≤4,096 bytes |

The progressive section is **pure data, near-zero overhead**. No compression, no indirection. Every byte is a float32 stat value. This is correct: the section is designed for GPU bulk transfer, not space efficiency.

### Phase 5: Universe Projections (4604 tickers)

| Operation | Per-file | Universe |
|---|---|---|
| Write (safe, no progressive) | 23.04ms | 106s |
| Write (safe, with progressive) | 59.68ms | 275s |
| Write (batch, no progressive) | 15.52ms | 72s |
| **Write (batch, with progressive)** | **54.42ms** | **251s** |
| Full column read | 4.07ms | 18.7s |
| **Progressive summary** | **0.08ms** | **0.4s** |
| Progressive 5min level | 1.16ms | 5.4s |
| Progressive 30min level | 0.93ms | 4.3s |

**Coarse K04 via progressive: 0.4s vs 18.7s = 49x speedup.**

### Design Implications

1. **Progressive write cost is real but acceptable.** +37ms per file is dominated by 3.1 MB of stats data, not overhead. For batch mode (K02 output), this is 251s for the universe — significant but happens once per day. The stats enable downstream K04 to skip full reads entirely.

2. **Cadence selection matters for write cost.** The 1s cadence (23,400 bins) contributes ~74% of progressive data volume. Dropping to 5s finest (4,680 bins) would halve progressive section size and write time. The MI scores (0.85 → 0.22 from coarsest to finest) suggest the finest cadences add diminishing information.

3. **The read path is the win.** Progressive summary = 83us vs full read = 4,072us. For K04 correlation matrices, reading 4604 progressive summaries (0.4s) vs 4604 full files (18.7s) is a 49x speedup. This is the architectural payoff.

4. **Fine level reads can be expensive.** 1s level = 61ms per file. If K04 needs fine-grained stats, reading individual levels is slower than reading the full column data (4ms). Use progressive for coarse screening, full reads for detailed computation.

Benchmark script: `experiments/file_format/bench_progressive.py` (runs via fintek venv)

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
| 7 | Concurrent reader (R581) | 12 workers → 5.6 GB/s (80% NVMe), 2.45x speedup |
| 8 | NVMe queue depth sweep (R595) | Knee at 8 workers, 52% NVMe ceiling |
| 9 | MKTF v4 vs v3 | 4x write cost (crash safety), **35% faster reads**, 170ms universe scan |
| 10 | Driver regression isolation | **R595 driver does internal I/O coalescing** — threading gives 1.21x not 2.45x |
| 11 | **MKTF v4 vs Parquet** | **2.7x faster writes, 1.5-2.2x faster reads, 2.7x faster GPU pipeline** |
| 12 | **Read path micro-anatomy** | **97% I/O-bound, Rust gains ~7%, Windows kernel caps at 69% NVMe** |
| 13 | **I/O strategy sweep** | **Handle pools are a trap (bulk read penalty); production design is near-optimal** |
| 14 | **Write path micro-anatomy** | **fsync dominates small writes (62%); no-fsync cadence grid: 140s → 12s (91% savings)** |
| 15 | **Progressive section + dual write** | **+37ms write, +0.7% read; progressive summary = 49x faster than full read for coarse K04** |

### Complete Pipeline Model (4604 tickers, AAPL-scale)

| Stage | Per-ticker | Universe | Notes |
|---|---|---|---|
| NVMe → CPU (read) | 2.9ms | 13.4s | 4.7 GB/s, 69% NVMe (Exp 12) |
| CPU → GPU (H2D) | 0.5ms | 2.3s | 24.6 GB/s, 38% PCIe Gen5 |
| GPU compute | ~1-2ms | ~5-9s | Leaf-dependent |
| GPU → CPU (D2H) | 1.7ms | 7.9s | 7.3 GB/s (cadence results much smaller) |
| **Write (production, 10 cad)** | **30ms** | **140s** | **fsync-dominated** |
| **Write (no-fsync, 10 cad)** | **2.6ms** | **12s** | **91% savings** |
| **Write (batch + progressive)** | **54.4ms** | **251s** | **+37ms from 3.1 MB stats (Exp 15)** |
| **Progressive summary read** | **0.08ms** | **0.4s** | **49x faster than full read (Exp 15)** |

**K01→K02 full universe estimate:**
- Production mode: ~170s = 2.8 min (dominated by writes)
- Batch mode (no-fsync): ~35s (dominated by reads)
- Batch + progressive: ~270s (dominated by progressive writes — acceptable for once-per-day K02)

**K04 via progressive shortcut:**
- Traditional: read 4604 full files (18.7s) + GPU compute
- Progressive: read 4604 summaries (0.4s) + lightweight compute = **49x read speedup**

**Bottleneck cascade:**
1. With fsync: **writes** (140s of 170s = 82%)
2. Without fsync: **reads** (13.4s of 35s = 38%), then GPU compute
3. Python overhead is negligible everywhere

**Previous pipelines for context:**
- Before this research: 3.7 hours (per Tekgy, only through K02)
- Parquet-based: 40.6s read alone (no write optimization)
- MKTF v4 + batch writes: **~35s full K01→K02**

**Daemon**: 137ms to scan 4604 files for dirty/incomplete. 33K status checks/second.
**Key insight**: MKTF v4 wins every dimension. The write mode (fsync vs batch) is the single largest pipeline decision.

---

## Data Characterization Notes

Observations about the AAPL tick data (598,057 rows, 2025-09-02) that affect format and analysis design:

- **19.5% of trades have zero inter-arrival time** (~117K trades for AAPL). These are exchange-level batches — multiple fills from one order reported at the same nanosecond. The timestamp column is monotonically **non-decreasing**, not strictly increasing. Any analysis assuming unique timestamps will silently produce wrong results. **Scales with liquidity**: mega-caps 17-18%, mid-caps ~25% (KO). Per-ticker analysis code must handle this differently by ticker.
- **Median IAT = 346us, mean IAT = 120ms** — extremely heavy-tailed. Most trades arrive in sub-millisecond bursts, but gaps between bursts are 100ms+. This distribution is NOT Poisson (PSD dynamic range of 1047x decisively rejects flat spectrum).
- **Spectral structure after detrending** (naturalist's K-F01b + detrending experiments, 10 tickers):
  - ~~Two-regime with institutional layer at 15-30min~~ **CORRECTED**: The "institutional regime" above ~5min was mostly daily U-shape artifact. After subtracting the mean intraday IAT curve:
  - Sub-second: suppressed (0.7-0.9x) — exchange batching anti-correlates arrivals. **CONFIRMED, enhanced by detrending.**
  - 1s-30s execution regime: 2-5x excess — algorithmic execution synchronization, scales with liquidity. **CONFIRMED, enhanced.**
  - 1-2 minute: REAL periodic structure (AAPL 5.9x, NVDA 24.2x, TSLA 12.0x). **CONFIRMED.**
  - **5 minute: REAL — the genuine institutional boundary.** NVDA 28.5x, TSLA 16.0x, AAPL 4.1x after detrending. Cross-ticker CV at 5min (0.93) exceeds all execution-regime CVs. 8/10 tickers peak at 5min. This is where algorithmic ecosystems diverge.
  - ~~10-30 minute: massive excess (5-58x)~~ **REFUTED**: 15min collapsed 90-97% after detrending (AAPL 33.5x→3.4x, NVDA 152.9x→3.9x). 30min collapsed 88-98%. These were U-shape artifacts, not periodic structure.
  - Crossover at ~0.5-1s separates exchange physics from algorithmic strategy. **CONFIRMED.**
  - Pink noise baseline (alpha ≈ 0.3) is universal regardless of liquidity. **CONFIRMED.**
  - The daily U-shape itself varies by ticker: AAPL 3.5x mid/open ratio, NVDA 3.0x (doesn't recover at close), TSLA 4.0x (extreme open burst). The U-shape is an observable, not a periodic signal.
  - **The 20min cadence sentinel was NOT justified by spectral excess.** Still useful for archival/detection, but the I/O cost argument (1.2s batch mode) is the justification, not spectral signal.
- **Delta encoding of timestamps would compress beautifully** (massive repetition from zero-IAT trades) but is net negative for GPU pipeline throughput (Experiment 2). Raw int64 is correct.
- **flip_dirty write asymmetry**: Reading 1 byte at EOF = 29us. Writing 1 byte = 3,937us (135x slower). NTFS file-open-for-write metadata overhead dominates. Daemon bulk-marking should be avoided; batch dirty signals instead.
