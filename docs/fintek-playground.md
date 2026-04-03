# TERNYX / Fintek — Technical Playground for Tambear Integration

> Written for the winrapids/tambear team to understand what fintek has built,
> how the data format works, how compute executes, and where tambear could
> replace or accelerate the pipeline.

---

## Part 1: MKTC / MKTF / MKTD — Three-Layer Market Data Format

### The Problem It Solves

Market tick data: ~600K trades/day for one stock, ~4600 stocks, 31 temporal resolutions,
200+ computed features per resolution. That's ~29 billion output values per day.
Traditional formats (parquet, CSV, Arrow) can't handle this at the I/O speeds we need
(sub-millisecond per file read for real-time trading signals).

### The Three Layers

```
MKTC (Container)  — one OS file per (day, scope). Pre-allocated regions.
  └─ MKTF (File)  — one region per leaf. 4096-byte header + MKTD chunks.
       └─ MKTD (Data) — one chunk per output column. Self-contained.
```

**MKTD** is the atomic unit. 96-byte header + compressed data bytes.

```
MKTD Header (96 bytes):
[0:4)     magic "MKTD"
[4:36)    name (32 bytes, null-padded) — e.g. "DI01DO01", "V01", "E03"
[36:37)   dtype_code (0=f32, 1=f64, 2=i32, 3=i64, 4=u8, 6=u32, ...)
[37:38)   ndim (1-4)
[38:39)   layout (0=linear, 1=cartesian, 2=radial, ...)
[39:40)   codec (0=none, 1=zstd, 2=lz4)
[40:41)   pre_filter (0=none, 1=shuffle, 2=delta, 4=delta_shuffle, 255=auto)
[41:42)   typesize (element size for shuffle filter)
[44:60)   dims[4] (uint32 × 4)
[60:68)   compressed_size (uint64)
[68:76)   decompressed_size (uint64)
[76:84)   checksum (SHA-256[:8] of DECOMPRESSED data)
[84:88)   compression_ratio (f32 — empirical entropy proxy, FREE feature)
[88:92)   entropy_bpb (f32 — bits per byte)
[92:96)   filter_gain (f32 — how much pre-filter helped)
```

Key insight: **compression_ratio IS a feature.** Every MKTD chunk carries its own
information-theoretic measurement for free. Read 96 bytes → know the entropy of the
data without decompressing it.

**MKTF** wraps multiple MKTDs for one leaf at one cadence for one ticker-day.

```
MKTF Header (4096 bytes):
[0:16)      FORMAT: magic "MKTF", version, flags, alignment, n_outputs
[16:80)     IDENTITY: leaf_id, ticker, day, TITO (cadence), KIKO (domain)
[80:128)    TREE: KPCRFGS address (kingdom/phylum/class/rank/family/genus/species)
[128:176)   RESERVED
[176:224)   PROVENANCE: timestamps, durations, byte counts
[224:288)   UPSTREAM: 4 fingerprint slots (content-addressed provenance)
[288:2336)  DIRECTORY: up to 64 MKTD entries × 32 bytes each
              (name_hash, offset, compressed_size, decompressed_size, dtype, codec, filter)
[2336:4094) RESERVED (1758 free bytes for future evolution)
[4094:4096) STATUS: is_complete, is_dirty
```

Data starts at byte 4096 (sector-aligned). MKTD chunks are sequential after the header.
Trailing 2 bytes duplicate the status (daemon reads 1 byte at EOF for fast dirty check).

### How Writers Work

**Rust writer** (`mkt::write::write_mktf_file`):
1. Takes `Vec<OutputColumn>` — each has name, raw bytes, dtype, codec, filter
2. **Parallel compression with rayon** — each column compressed independently
3. Auto-filter: `Filter::Auto` tries all 6 pre-filters on a 64KB sample, picks best
4. Builds MKTF header with directory entries (offset, compressed_size per MKTD)
5. Atomic write: tmp file → write → flip is_complete → rename

```rust
let outputs = vec![
    OutputColumn::new_1d("DO01", price_bytes, Dtype::Float64, n_rows)
        .with_codec(Codec::Zstd, Filter::Auto),
    OutputColumn::new_1d("DO02", volume_bytes, Dtype::Float64, n_rows),
    OutputColumn::new_1d("E01", emission_bytes, Dtype::Float32, n_rows),
];
write_mktf_file(&path, &outputs, "K02P01C01", "AAPL", "2025-09-03", 0, 5, 6)?;
```

**Compression results on real data:**
- Timestamps (i64, sorted): **254x** with DeltaShuffle + zstd
- Exchange codes (i32, low cardinality): **1250x**
- Prices (f64, random walk): **1.5x**
- Overall K02 data: **5.5x**
- V columns (f32, mostly 1.0): **~1000x**

### How Readers Work

**Rust reader** (`mkt::read`):
- `read_all(path)` → `(Header, Vec<(MktdHeader, Vec<u8>)>)` — reads + decompresses all
- `read_outputs(path, &["DO01", "E03"])` — selective read by name (seeks to MKTD, decompresses only requested)
- `read_status(path)` → `(bool, bool)` — 2 bytes at EOF, zero parsing
- `scan_metadata(path)` → headers only, no decompression (for entropy scanning)

**PyO3 bindings** expose all of the above to Python:
```python
import mkt
results = mkt.read_all("path/to/leaf.mktf")
# results: [(name, dtype_code, bytes), ...]

# Header-only scan (no decompression) — for entropy features
mkt.scan_metadata("path/to/leaf.mktf")
```

### What Makes This Good for Us

1. **Per-column compression** — each MKTD has its own codec + filter. Timestamps get
   DeltaShuffle (254x). Prices get raw zstd (1.5x). One file, optimal per column.

2. **Sector-aligned** — 4096-byte header = one NVMe sector read. Data starts at sector boundary.
   NVMe DMA can feed GPU without CPU involvement.

3. **Self-describing** — every MKTD carries its own dtype, dims, layout, codec. Hand it to
   any reader (Python, Rust, GPU) and it can reconstruct the array. No schema files.

4. **Compression-as-feature** — the ratio/entropy/filter_gain in every MKTD header IS
   a free market signal. Scan 100K files reading only 96 bytes each → entropy heatmap.

5. **Status bytes at EOF** — daemon reads 1 byte to check dirty/complete. No header parse.

6. **Content-addressed provenance** — MKTD checksum is SHA-256 of decompressed data.
   Invariant across codec changes. Upstream fingerprints in MKTF verify the provenance chain.

---

## Part 2: The Daemon and Roaring Bitmap State

### Rust Daemon (`trunk-rs`)

The daemon runs forever, computing features for all tickers across all days:

```bash
trunk-rs daemon \
  --data-root W:/fintek/data/fractal \
  --tickers proxy_universe.txt \
  --db-path state.db \
  --interval 30
```

**Loop:**
1. Discover days (scan K01/ for YYYY-MM-DD directories)
2. Per-day: discover tickers (which have K01P01.TI00TO00.mktf)
3. Reconcile: diff (days × tickers × leaves) against roaring bitmap DB
4. Pull ready work: pending items where all upstream leaves are complete
5. Execute via rayon `par_iter` over tickers
6. Mark complete/failed in bitmap DB
7. Checkpoint (flush bitmaps to SQLite)
8. Sleep, repeat

**Roaring bitmap state:**
- SQLite + roaring bitmaps serialized as BLOBs
- Key = leaf_id, value = bitmap of (day_idx × n_tickers + ticker_idx)
- Three states: pending, complete, failed
- Encoding: `day_idx * n_tickers + ticker_idx` as u32
- Fast set operations: "which items are pending AND have all upstreams complete?"
  = `pending_bitmap & upstream1_complete & upstream2_complete`

### Execution Model

For one ticker-day:

```
Phase 1: Load K01P01 tick data (price, size, timestamps)
         Sort by timestamp (fixes 12% inversion in raw data)
         Pre-compute cadence boundaries for all 31 cadences

Phase 2: K01P02 tick-level leaves (10 leaves)
         pointwise: log, sqrt, reciprocal, elapsed, cyclical, notional
         delta: value, percent, direction, log
         Results stored in TickerDay for downstream access

Phase 3: K02 bin-level leaves (111+ leaves) × 31 cadences
         For each cadence:
           Compute bin boundaries (partition_point on sorted timestamps)
           For each leaf: iterate bins, compute, collect output
           Write MKTF with DO + V + E columns

Phase 4: K03 cross-cadence leaves (5 leaves)
         Read completed K02 MKTF files from disk (JoinContext)
         Compute cross-cadence features (gradient, coherence matrix, etc.)
         Write K03 MKTF output
```

**Performance:** 1086 tickers × 31 cadences × 121 leaves = 8 minutes (rayon, 32 cores).
Python baseline: ~5 hours. Speedup: 64x parallel, 754x sequential.

### Python Daemon (trunk/)

The Python daemon (`trunk/exec/daemon.py`) uses the same roaring bitmap pattern
but with Python's `pyroaring` + SQLite. Same three states, same reconcile, same
dispatch-by-kingdom. But:

- GIL limits parallelism (ThreadPoolExecutor for I/O, ProcessPoolExecutor for compute)
- Python leaf execute() overhead: ~400ms/leaf for simple math
- Some leaves have pure-Python for-loops that take minutes per bin
- GPU leaves (CuPy) bypass the GIL but need explicit memory management

The Python tree (`trunk/`) has 206 production leaves with full spec.py definitions.
The Rust tree (`crates/trunk-rs/`) has 121 leaves ported + the daemon/executor/state.

### The Graph

The compute graph is a DAG where:
- Nodes = leaves (each defined by a spec.toml)
- Edges = channels (leaf A's output → leaf B's input)
- Tree edges = hierarchy (kingdom/phylum/class/rank/family/genus/species)
- Channel edges = data dependencies

Topological sort of the channel subgraph = execution order.
The graph IS the execution plan. No separate scheduler needed.

---

## Part 3: Leaf Compute — Per-Column, Per-Bin

### What a Leaf Does

Every leaf implements the same interface:

```rust
trait Leaf: Send + Sync {
    fn id(&self) -> &str;
    fn execute(&self, ctx: &dyn ExecutionContext, cadence_id: u8)
        -> Result<LeafOutput>;
    fn outputs(&self) -> &[(&str, Dtype)];
}
```

The ExecutionContext provides:
- `column_f64("price")` → &[f64] (tick-level data)
- `bin_boundaries(cadence_id)` → (&[u32], &[u32]) (starts, ends per bin)
- `n_bins(cadence_id)` → usize

### Per-Bin Pattern (Most Common)

```rust
for i in 0..n_bins {
    let s = starts[i] as usize;
    let e = ends[i] as usize;
    let bin_data = &price[s..e];

    // Compute something on the bin's tick slice
    result[i] = compute_feature(bin_data);
}
```

This runs at every cadence (31 times with different bin boundaries).
Each cadence gives different bin sizes: T01 (50ms) has ~1.7M bins, T30 (1 day) has 1 bin.
The SAME leaf produces output at every temporal resolution.

### Column Types

Every leaf can produce three kinds of output columns:

- **DO** (Data Output): the computed values. Float64 typically. The science.
- **V** (Validity): self-assessment of trustworthiness. Float32. "Can you trust DO01?"
- **E** (Emission): signals that something happened. Float32 flags. "Did this bin have an event?"

All three are MKTD chunks in the same MKTF file. Same format. Same compression.

### What This Means for Tambear

**The leaf's execute() is a pure function:**
- Input: slice of f64/i64 tick data per bin
- Output: one f64/f32 value per bin per output column
- No side effects, no shared state, no I/O during compute

**This IS a GPU kernel.** Each bin is an independent work unit. Each leaf is an
independent function. The only dependency is the graph order (K01 before K02 before K03).

**Tambear opportunity:**
1. Compile spec.toml → .tbs script → single-pass GPU pipeline
2. All bins of all leaves at one cadence → ONE kernel launch
3. The per-column output → direct MKTD write (GPU compressed output to NVMe)
4. 121 leaves × 31 cadences = 3,751 compute units per ticker → ONE fused pass
5. Cross-platform: same pipeline on RTX PRO 6000, DGX Spark, Apple Silicon

**The format supports this natively:**
- MKTD chunks are independent (different dtypes, different codecs per column)
- Directory in MKTF header allows out-of-order writes
- GPU can write compressed MKTD chunks directly to mapped memory
- Status bytes flip at the end (crash-safe)

**What tambear would replace:**
- The per-leaf Rust execute() functions → compiled .tbs kernels
- The rayon par_iter over tickers → tambear's own parallelism
- The sequential per-bin loop → GPU-parallel bin processing
- The zstd compression → GPU-native compression in the write path

**What tambear would NOT replace:**
- The MKTF/MKTD format (it's the storage contract)
- The daemon's bitmap state management (orchestration, not compute)
- The spec.toml definitions (they drive everything)
- The graph / topological sort (execution ordering)

---

## Part 4: The Trading Signal Pipeline

### E Columns → Detection → Action

```
K01: raw ticks
K02: per-bin features (DO + V + E columns) × 31 cadences
K03: cross-cadence features (DO + V + E columns)
K04: cross-ticker features (future)
K05: detection (reads E columns across leaves, outputs D columns)
K06: action (reads D columns, executes trades or shadow-logs)
```

E columns are computed alongside DOs — just another output. No separate emit system.

**Two execution paths (structural separation, no flags needed):**
- **Live path**: ephemeral in-memory. Compute → detect → act in real-time. If detect daemon isn't running, signal is gone. Never fires retroactively.
- **Research path**: reads E columns from disk. Runs hundreds of experimental strategies simultaneously. All shadow. Zero foreknowledge. Builds genuine out-of-sample track records.

### Current Signals

1. **Cadence gradient collapse** — realized variance gradient flattens across cadences → vol expansion imminent
2. **Cross-cadence coherence matrix** — eigenvalue decomposition of 31×31 cadence correlation → coupling detector
3. **Taylor fold precursor** — polynomial correction ratios diverge → price fold imminent
4. **KAM harmonic diagnostic** — r-statistic approaches Poisson → fragile resonant regime
5. **Compression entropy** — MKTD header compression ratios across cadences/tickers → free multi-resolution entropy

### Connected Infrastructure

- Coinbase API (382 USD crypto pairs, funded account)
- Shadow ledger with tax tracking (wash sales, cost basis, ST/LT gains)
- 7 days of historical data (106 GB on disk)
- Roaring bitmap daemon auto-computes new data

---

## Summary: What Tambear Could Do for TERNYX

| Current | With Tambear |
|---------|-------------|
| 121 separate Rust execute() per leaf | One fused .tbs pipeline per cadence |
| Sequential per-bin loop | GPU-parallel all bins simultaneously |
| rayon across tickers only | Tambear parallelism across tickers + bins + leaves |
| CUDA-only GPU (CuPy) | CUDA + Vulkan + DX12 + Apple Silicon |
| ~8 min for 1086 tickers | Target: seconds |
| Separate compression step | GPU-native compressed writes |
| Per-leaf .rs files | Per-leaf spec.toml → compiled pipeline |

The spec.toml → .tbs compilation path is the bridge. If tambear can read spec.toml
(or a trivial conversion to .tbs), every leaf in the system becomes a GPU pipeline
stage automatically. No manual porting needed.
