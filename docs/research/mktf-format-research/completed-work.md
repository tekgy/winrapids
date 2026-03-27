# MKTF Format Research — Completed Work

## 2026-03-27: Prototype Benchmarks (pathmaker)

### Summary

Built 6 MKTF format variants and benchmarked against 4 baselines using real AAPL tick data (598,057 ticks, 8 columns). **MKTF encoding + raw binary = 5.4ms W+R, 26x faster than Parquet+Zstd.**

### Data Encoding Wins

The biggest gain isn't the file format — it's the **column encoding**. Converting the raw parquet schema to MKTF-native types cut data from 35MB to 15MB:

| Transform | Before | After | Savings |
|-----------|--------|-------|---------|
| conditions: csv string -> uint32 bitmask | 2.6 MB | 2.4 MB | 8% |
| timestamp: int64 abs -> base + int32 deltas | 4.8 MB | 2.4 MB | 50% |
| ticker: string column -> header metadata | 2.4 MB | 0 MB | 100% |
| sequence: string -> int32 | 3.1 MB | 2.4 MB | 23% |

17 unique condition codes mapped to bits 0-16 of a uint32. Room for 15 more flags.

### Full Benchmark Results

598,057 ticks x 8 columns = 15.0 MB MKTF-encoded (was 35 MB raw with strings)

| Format | Write | Read | Size | W+R | Correct |
|--------|-------|------|------|-----|---------|
| Raw binary (no align) | 3.6ms | 1.8ms | 15.0MB | **5.4ms** | Y |
| MKTF v4 (np.memmap) | 8.1ms | 0.6ms | 15.0MB | 8.6ms | Y |
| MKTF v5 (tofile write) | 4.4ms | 4.7ms | 15.0MB | 9.1ms | Y |
| MKTF v2 (buf read) | 7.4ms | 4.5ms | 15.0MB | 12.0ms | Y |
| MKTF v1 (aligned) | 10.6ms | 1.8ms | 15.0MB | 12.5ms | Y |
| MKTF v3 (mmap) | 7.9ms | 5.0ms | 15.0MB | 12.9ms | Y |
| numpy .npz | 7.8ms | 6.5ms | 15.0MB | 14.3ms | Y |
| Arrow IPC (raw) | 9.8ms | 8.6ms | 34.2MB | 18.4ms | Y |
| Parquet+none (raw) | 103.6ms | 14.1ms | 12.4MB | 117.8ms | Y |
| Parquet+Zstd (raw) | 124.4ms | 14.4ms | 5.6MB | 138.8ms | Y |

### Selective Column Read (price + conditions only, 2 of 8 cols)

| Method | Read time |
|--------|-----------|
| MKTF v4 (np.memmap) | **0.12ms** |
| MKTF v1 (seek) | 0.45ms |
| MKTF v3 (mmap) | 1.31ms |
| Parquet+none | 2.27ms |
| MKTF v2 (buf read) | 4.07ms |

np.memmap selective read is **19x faster** than Parquet for 2 columns.

### Universe Projection (4,604 tickers)

| Format | Total time |
|--------|-----------|
| Raw binary (no align) | 24.7s (0.4 min) |
| MKTF v4 (np.memmap) | 39.8s (0.7 min) |
| MKTF v5 (tofile write) | 41.8s (0.7 min) |
| Parquet+Zstd (raw) | 639.2s (10.7 min) |

### Key Findings

1. **Raw binary with MKTF encoding is the winner for full W+R.** 5.4ms per ticker, 25s for full universe. No alignment padding needed — the NVMe controller doesn't care.

2. **np.memmap wins for read-only and selective access.** 0.6ms full read, 0.12ms for 2 columns. The OS page cache does the work. Write is slower (8.1ms) because we still build the full buffer.

3. **Alignment (64-byte) adds write overhead but doesn't help reads.** The aligned v1 format writes slower than raw because of buffer construction overhead. GPU DMA alignment matters for H2D, but the file read itself doesn't benefit.

4. **The parquet format is 26x slower than MKTF for W+R.** Parquet's strength is compression (5.6MB vs 15MB), but we have NVMe bandwidth to burn.

5. **Column encoding is the real insight.** The format is trivial — the encoding decisions (bitmask, delta, type narrowing) deliver the actual gains.

### MKTF File Layout (v1)

```
[0..4)     MAGIC "MKTF"
[4..8)     version (u32)
[8..16)    n_rows (u64)
[16..18)   n_cols (u16)
[18..20)   header_size (u16)
[20..24)   metadata_offset (u32)
[24..64)   padding to 64-byte alignment
[64..64+n_cols*32)  column directory (32 bytes each):
    [0..16)   name (null-padded UTF-8)
    [16..17)  dtype_code (u8)
    [17..21)  n_elements (u32)
    [21..24)  padding
    [24..32)  data_offset (u64)
[metadata_offset..) JSON metadata (ticker, date, condition_bits map)
[data_start..)      column data at 64-byte aligned offsets
```

### Prototype Code

`research/20260327-mktf-format/bench_mktf.py` — all 6 variants + benchmark harness.

### Hybrid Variant: tofile Write + memmap Read

Follow-up experiment combining the fastest write path (tofile, 3.8ms) with the fastest read path (np.memmap, 0.12ms selective). This is the recommended production combination:

- **Write:** Build small header (~320 bytes) as bytearray, then `tofile()` each column at aligned offsets. No full-buffer allocation. **3.8ms.**
- **Read:** Parse header (24 bytes), then `np.memmap()` each requested column. Truly lazy — OS page cache does the work. **0.12ms for 2 cols, ~0.6ms for all 8.**
- **Universe:** ~20s write + ~3s read = **~23s total** for 4,604 tickers.

---

## 2026-03-27: MKTF v2 + Integer Encoding (pathmaker)

### MKTF v2 Spec

Clean 64-byte aligned format incorporating all team findings:

```
Header (64 bytes):
  [0:4]    magic "MKTF"
  [4:6]    version uint16
  [6:8]    flags uint16
  [8:16]   n_rows uint64
  [16:18]  n_cols uint16
  [18:26]  reserved
  [26:34]  metadata_offset uint64
  [34:42]  metadata_size uint64
  [42:64]  reserved

Column directory (64 bytes per column):
  [0:32]   name (null-padded UTF-8)
  [32:33]  dtype_code uint8
  [33:40]  reserved (alignment)
  [40:48]  n_elements uint64
  [48:56]  data_offset uint64
  [56:64]  data_nbytes uint64

Data: each column at 64-byte aligned offset, raw little-endian.
```

Design decisions:
- **Absolute timestamps** (navigator: 72K negative deltas from out-of-order venue feeds)
- **64-byte alignment** (observer: proven 10% faster for GPU pipeline)
- **No mmap** (observer: page faults during H2D = worst path at 14.48ms)
- **tofile write** (pathmaker: avoids full-buffer allocation, 3.9ms)
- **Column directory in header** (observer: 12.48x speedup for selective reads)

### Encoding Comparison (Real AAPL, 598K ticks)

| Encoding | Data | File | Write | Read | Sel(2) | W+R |
|----------|------|------|-------|------|--------|-----|
| **float32** | 15.5MB | 15.6MB | **3.9ms** | 6.0ms | **0.51ms** | **9.8ms** |
| float64 | 20.3MB | 20.3MB | 6.6ms | 6.8ms | 0.75ms | 13.3ms |
| int64 (10^-8) | 20.3MB | 20.3MB | 6.4ms | 6.6ms | 0.76ms | 13.0ms |
| int32 (10^4) | 15.5MB | 15.6MB | 4.9ms | 6.3ms | 0.51ms | 11.1ms |

### GPU Operations: float32 vs int64

| Operation | float32 | int64 | Ratio | Winner |
|-----------|---------|-------|-------|--------|
| sum | 14.5us | 19.5us | 1.34x | float32 |
| mean | 66.6us | 51.9us | 0.78x | int64 |
| std | 223.8us | 349.8us | 1.56x | float32 |
| min | 15.8us | 15.2us | 0.96x | tie |
| max | 15.0us | 15.5us | 1.04x | tie |
| diff | 12.5us | 11.4us | 0.92x | tie |
| notional (p*s) | 8.7us | 9.2us | 1.06x | tie |

**GPU H2D (disk -> GPU VRAM):**
- float32: **5.52ms** (15.6MB)
- int64: 6.22ms (20.3MB)

### Integer Encoding Analysis

The int64 at 10^-8 path is **not faster for I/O or compute**:
- **30% larger files** (20.3MB vs 15.5MB) because int64 is 8 bytes vs float32's 4 bytes
- **GPU sum/std are 34-56% slower** on int64 (GPU float32 units are heavily pipelined)
- **mean is 22% faster** on int64 (likely due to simpler reduction)
- **min/max/diff are ties**

Where int64 wins:
- **Exact arithmetic** — no floating point drift, deterministic results
- **Composability** — integer ops can be fused more aggressively in custom CUDA
- **Correctness** — $0.0001 precision guaranteed, no epsilon comparisons needed

Recommendation: **float32 for the hot path (file + GPU), int64 for the canonical representation (in metadata/schema).** The price_scale metadata field lets us convert at the boundary. If custom CUDA kernels later show integer advantages for specific operations, we can add an int64 column variant.

### Full Pipeline Disk -> GPU

| Encoding | Disk->GPU | Data size |
|----------|-----------|-----------|
| float32  | **5.52ms** | 15.6MB |
| int64    | 6.22ms | 20.3MB |

Universe (4,604 tickers): **25.4s float32** vs 28.6s int64.

---

## 2026-03-27: MKTF v3 — Self-Describing Format (pathmaker)

### The Insight

The header IS the manifest IS the state IS the provenance. No sidecar files. No external metadata. The file is the truth. One NVMe sector read (4096 bytes) gives the daemon everything it needs.

### Benchmark Results (598K ticks, 7 cols, 15.5MB data)

| Operation | Time | Notes |
|-----------|------|-------|
| **Write** | 6.91ms | Header + directory + aligned data + is_complete flip |
| **Read (full)** | 5.96ms | Entire file into buffer, slice columns |
| **Read (selective, 2 cols)** | 0.51ms | Seek to price + conditions only |
| **Read (header only)** | 0.031ms (31us) | Identity, quality, provenance — zero data bytes |
| **Disk -> GPU** | 5.32ms | Full file read + H2D all columns |
| **W+R total** | 12.87ms | |
| **File size** | 15.59MB | 0.2% overhead from 4096-byte alignment |

### Universe Projections (4,604 tickers)

| Operation | Time |
|-----------|------|
| Full W+R | 59.3s (1.0 min) |
| Selective read (2 cols) | 2.3s |
| **Header scan (staleness check)** | **151ms** |
| Disk -> GPU | 24.5s (0.4 min) |
| Storage | 71.8 GB |

**151ms to check staleness of the entire market.** That's the daemon checking is_complete + leaf_version for all 4,604 tickers.

### Header Contents (what the daemon sees — zero data bytes)

```
Identity:
  leaf_id:        K01P01
  ticker:         AAPL
  day:            2025-09-02
  cadence:        TI00TO00
  leaf_version:   1.0.0
  schema_fp:      dc89ddd95ef0e0b2

Dimensions:
  n_rows:         598,057
  n_cols:         7
  bytes_data:     15,549,482

Quality:
  total_nulls:    0
  is_complete:    true
  Per-column min/max/null_count in directory

Provenance:
  write_timestamp: nanosecond precision
  write_duration:  5ms
  compute_duration: 0ms
```

### Column Directory (per-column quality without reading data)

| Name | Dtype | Elements | Size | Nulls | Min | Max |
|------|-------|----------|------|-------|-----|-----|
| price | float32 | 598,057 | 2.39MB | 0 | 223.49 | 254.52 |
| size | float32 | 598,057 | 2.39MB | 0 | 1.0 | 4,996,298 |
| timestamp | int64 | 598,057 | 4.78MB | 0 | 1.757e18 | 1.757e18 |
| exchange | uint8 | 598,057 | 0.60MB | 0 | 1 | 21 |
| sequence | int32 | 598,057 | 2.39MB | 0 | 1 | 265,794 |
| conditions | uint32 | 598,057 | 2.39MB | 0 | 0 | 114,688 |
| is_odd_lot | uint8 | 598,057 | 0.60MB | 0 | 0 | 1 |

### Crash Recovery

Write protocol:
1. Write header with `is_complete=0`
2. Write directory, metadata, column data
3. Seek back, flip `is_complete=1` + write `write_duration_ms`

If process dies between 1 and 3: file has `is_complete=false`. Daemon detects, deletes, recomputes.

### v3 vs v2 vs v1

| Version | Write | Read | Sel(2) | Header | File | Features |
|---------|-------|------|--------|--------|------|----------|
| v1 (64B align) | 3.9ms | 6.0ms | 0.51ms | N/A | 15.6MB | Basic aligned columns |
| v2 (64B align) | 3.9ms | 6.0ms | 0.51ms | N/A | 15.6MB | Clean spec + encoding comparison |
| **v3 (4096B align)** | 6.91ms | 5.96ms | 0.51ms | **0.031ms** | 15.59MB | Self-describing, crash recovery, quality stats |

v3 write is ~3ms slower (header construction + is_complete seek-back). Read is the same. But v3 eliminates ALL sidecar files and gives the daemon 31us header checks.

### File Layout (v3 final)

```
Block 0 [0..4096):
  Fixed header (184 bytes used, rest reserved):
    [0:16]   Core: magic, version, flags, alignment, header_blocks
    [16:98]  Identity: leaf_id, ticker, day, cadence, leaf_version, schema_fingerprint
    [98:120] Dimensions: n_rows, n_cols, n_bins, bytes_data
    [120:128] Quality: total_nulls (is_complete in flags)
    [128:144] Provenance: write_timestamp_ns, write_duration_ms, compute_duration_ms
    [144:184] Layout: dir_offset, dir_entries, meta_offset, meta_size, data_start

Block 1 [4096..):
  Column directory (128 bytes per column):
    [0:32]   name (null-padded UTF-8)
    [32:40]  dtype_code + padding
    [40:64]  n_elements, data_offset, data_nbytes
    [64:96]  scale_factor, null_count, min_value, max_value
    [96:128] reserved

Block N [aligned..):
  JSON metadata (condition_bits, upstream info, etc.)

Block M [aligned..):
  Column data (each at 4096-byte aligned offset)
```

### Prototype Code

`research/20260327-mktf-format/mktf_v3.py` — full implementation with read_header, read_data, read_selective, read_metadata.

---

## 2026-03-27: Source-Only MKTF + FP32 Kernel (pathmaker)

### The Big Insight

**Don't store derivables.** GPU recomputes 7 derived features from 2 source columns in 0.111ms (FP32 fused kernel). Reading those same 7 columns from NVMe takes 4.4ms. The recompute is **40x faster than the I/O it eliminates.**

### Head-to-Head: Source-Only vs Pre-Computed

| Metric | Source-only (5 cols) | Pre-computed (12 cols) | Winner |
|--------|---------------------|----------------------|--------|
| File size | **12.6MB** | 27.6MB | src (2.2x smaller) |
| Disk read | **3.83ms** | 8.23ms | src (2.1x faster) |
| H2D | **0.87ms** | 1.50ms | src (1.7x faster) |
| GPU compute | 0.111ms | 0.000ms | pre (but 0.111ms is nothing) |
| **TOTAL pipeline** | **4.81ms** | 9.73ms | **src wins by 4.92ms** |

**Source-only is 2x faster end-to-end** despite the GPU recompute step.

### Universe (4,604 tickers)

| Pipeline | Time | Storage |
|----------|------|---------|
| **Source-only** | **22.2s** (0.4 min) | 58.0 GB |
| Pre-computed | 44.8s (0.7 min) | 127.0 GB |
| **Savings** | **22.6s faster** | **69.0 GB saved** |

### FP32 vs FP64 Fused Kernel (7 outputs from 2 inputs)

| Precision | Time | Ratio |
|-----------|------|-------|
| **FP32** | **14.8us** | baseline |
| FP64 | 102.0us | 6.9x slower |

The 7x speedup comes from SFU-accelerated transcendentals (logf, sqrtf use dedicated hardware units) and 64x higher FP32 FLOPS (125 vs 1.95 TFLOPS).

### FP32 Precision Validation

| Derived column | Max relative error vs FP64 |
|---------------|---------------------------|
| ln_price | 4.75e-08 |
| sqrt_price | 3.16e-08 |
| recip_price | 5.56e-08 |
| notional | 5.96e-08 |

**~50 nanometer precision.** For market data where the smallest price tick is $0.01, this is 6+ orders of magnitude beyond what matters.

### Architecture Implications

1. **MKTF stores only source columns**: price(f32), size(f32), timestamp(i64), conditions(u32), exchange(u8)
2. **GPU recomputes all derived features** in a single fused FP32 kernel launch (0.111ms)
3. **The fused kernel IS the decoder**: MKTF bytes go straight to GPU, kernel produces all features
4. **Storage halved**: 58GB vs 127GB for full universe
5. **Pipeline 2x faster**: less I/O dominates over tiny recompute cost

### Prototype Code

`research/20260327-mktf-format/bench_source_only.py` — full pipeline benchmark + FP32/FP64 kernel comparison.

### Next Steps

- [ ] Switch fused_bin_stats.py from double to float (64x capability unlock)
- [ ] Switch fusion.py codegen from double to float output
- [ ] Multi-ticker batch reader (read 100 source-only files concurrently)
- [ ] Rust/C MKTF reader for maximum throughput
- [ ] DirectStorage integration (NVMe -> GPU VRAM, no CPU touch)
