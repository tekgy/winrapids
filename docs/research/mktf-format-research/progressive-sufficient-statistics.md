# Progressive Sufficient Statistics — MKTF Design Doc

*2026-03-27 — Pathmaker + Naturalist collaborative design*

## Problem

A K04 cross-ticker correlation matrix over the full universe (4,604 tickers) requires reading
ALL K01 tick data: ~58 GB from NVMe, ~10s concurrent read, then GPU feature extraction + GEMM.

But most K04 consumers don't need tick-level resolution. A coarse correlation (hourly bars)
is sufficient for screening, dashboards, and stale-while-revalidate patterns. If the file
already contains pre-aggregated statistics at multiple cadence levels, we can compute a
coarse K04 by reading **476 KB per column** instead of **12.6 MB per file** — a 750x
reduction in I/O.

The same principle applies to any cross-ticker or cross-cadence operation: the progressive
section provides a **resolution ladder** that lets consumers trade precision for speed.

## Core Idea

**Composable sufficient statistics at strategic cadence levels, stored progressively
(coarsest first) in a dedicated MKTF section.**

A sufficient statistic tuple `{sum, sum_sq, min, max, count}` is composable by addition:
to compute the stat for a parent interval, sum the children's `sum`, `sum_sq`, `count`,
take min of `min`, max of `max`. This means:

- Any resolution can be derived from any finer resolution
- Bins are independently addressable — no sequential dependency
- GPU-native: 5 × float32 = 20 bytes per bin per column, contiguous reads

## On-Disk Layout

The progressive section lives between the data region and the trailing status bytes.
Its location is recorded in Block 0 Layout reserved bytes.

```
MKTF file layout:
  Block 0           [0:4096)        Header (existing)
  Block 1+          [4096:...]      Column directory (existing)
  [optional]        [aligned]       Metadata block (existing)
  Data region       [aligned]       Column data (existing)
  Progressive sect  [aligned]       ← NEW: sufficient statistics
  Trailing status   [EOF-2:EOF]     is_complete, is_dirty (existing)
```

### Block 0 Layout Extension

Three new fields in the Layout section's reserved bytes [488:512):

```
Offset  Size  Type    Field
488     8     uint64  progressive_offset   (0 = no progressive section)
496     8     uint64  progressive_size     (total bytes including directory)
504     2     uint16  progressive_levels   (number of cadence levels, 0-18)
506     6     —       reserved (zero)
```

When `progressive_offset == 0`, the file has no progressive section. Readers skip it.
No flag bit needed — the pointer is self-describing.

### Progressive Section Internal Layout

```
progressive_offset →
  [0:8)                           Section header:
                                    n_levels (uint16)
                                    n_columns (uint16)
                                    stat_tuple_size (uint8)  = 5 (sum,sum_sq,min,max,count)
                                    stat_dtype (uint8)       = 0 (float32)
                                    domain (uint8)           = 1 (SUFFICIENT_STATS)
                                    reserved (uint8)

  [8 : 8 + (n_levels-1) × 4)     MI scores (float32 per adjacent pair):
                                    mi[0] = MI(level_0, level_1)
                                    mi[1] = MI(level_1, level_2)
                                    ...
                                    (0.0 if not yet computed)

  [after MI scores]               Level directory (24 bytes per level):
                                    cadence_ms    (uint32)   — cadence in milliseconds
                                    bin_count     (uint32)   — number of bins at this level
                                    section_offset(uint64)   — relative to progressive_offset
                                    section_size  (uint64)   — bytes for this level's data

  [aligned]                       Level data (coarsest first):
    Level K (session):   n_cols × 1 bin      × 20 bytes
    Level K-1 (1h):      n_cols × ~7 bins    × 20 bytes
    Level K-2 (1min):    n_cols × ~390 bins  × 20 bytes
    Level K-3 (1s):      n_cols × ~23400 bins × 20 bytes
```

### Per-Level Data Layout

Within each level, data is arranged **per-cadence** (all columns for one bin contiguous),
NOT per-column. This matches GPU access patterns: a K04 computation needs all columns at
the same cadence level, reading contiguously.

```
Level data for cadence C (bin_count = B, n_cols = N):
  Bin 0:  col_0{sum,sum_sq,min,max,count} col_1{...} ... col_N-1{...}
  Bin 1:  col_0{sum,sum_sq,min,max,count} col_1{...} ... col_N-1{...}
  ...
  Bin B-1: col_0{...} col_1{...} ... col_N-1{...}

Total: B × N × 5 × 4 bytes = B × N × 20 bytes
```

Each stat tuple is 5 × float32 = 20 bytes:

| Field    | Type    | Bytes | Composability         |
|----------|---------|-------|-----------------------|
| sum      | float32 | 4     | parent = Σ children   |
| sum_sq   | float32 | 4     | parent = Σ children   |
| min      | float32 | 4     | parent = min(children) |
| max      | float32 | 4     | parent = max(children) |
| count    | float32 | 4     | parent = Σ children   |

`count` is float32 (not uint32) so the entire tuple can be loaded as a single
`float32[5]` vector — no type mixing, clean GPU loads.

## Strategic Cadence Levels

Five levels chosen based on detrended spectral analysis (cross-ticker, market hours
only, U-shape removed). The durable spectral findings:

- **Execution regime (1s-60s)**: Algorithm clocks, 1.4-4.6x excess spectral power
  (enhanced after detrending). Scales with liquidity. Levels 1-2 serve this regime.
- **1-5 minute genuine periodicity**: 5.9x (AAPL) to 28.5x (NVDA) after detrending.
  The strongest confirmed periodic signal. Level 2 captures it directly.
- **Sub-second**: Suppressed (0.7-0.9x) — exchange batching physics, not market
  structure. 1s is the natural floor.
- **Above 10min**: Mostly daily U-shape artifact (open/midday/close rate variation),
  not genuine periodicity. Raw spectral excess at 15-30min was 90-98% artifact.

| Level | Cadence   | Bins/day | Bytes/col | Spectral excess (detrended) | Purpose                    |
|-------|-----------|----------|-----------|----------------------------|----------------------------|
| 5     | session   | 1        | 20 B      | N/A                        | Is this column alive?      |
| 4     | 1 hour    | 7        | 140 B     | N/A                        | Hourly dashboard           |
| 3     | 15 min    | 26       | 520 B     | 3.4x (AAPL, detrended)    | Composition breakpoint     |
| 2     | 1 minute  | 390      | 7.8 KB    | 5.9x (AAPL, detrended)    | Peak execution signal      |
| 1     | 1 second  | 23,400   | 468 KB    | 1.4x (AAPL)               | Fine-grained analysis      |

Level 3 (15min) is a composition convenience — not the highest-signal cadence (that's
Level 2 at 1min), but a useful pre-computed breakpoint between 1min and 1h. At 520
bytes per column, the cost is negligible. Ticker-specific institutional peaks at
5-30min (e.g., NVDA 28.5x at 5min) are composable from Level 2's 1-minute bins.

**NOTE**: Earlier versions of this doc cited 33.5x spectral excess at 15min. That was
inflated by the daily U-shape (open/midday/close trading intensity variation).
Detrending reduces it to 3.4x. The execution regime (1s-5min) is the durable signal.

### Read Crossover (observer, Experiment 16)

Progressive level reads cost `36 + 2.58 × n_bins` microseconds (Python overhead).
Full column reads cost ~4,072us regardless of content. **Crossover: 1,548 bins.**

| Cadence | Bins | Progressive read | vs Full read | Verdict     |
|---------|------|-----------------|--------------|-------------|
| session | 1    | 39 us           | 0.01x        | PROGRESSIVE |
| 1h      | 7    | 54 us           | 0.01x        | PROGRESSIVE |
| 15min   | 26   | 103 us          | 0.03x        | PROGRESSIVE |
| 5min    | 78   | 237 us          | 0.06x        | PROGRESSIVE |
| 1min    | 390  | 1,042 us        | 0.26x        | PROGRESSIVE |
| 30s     | 780  | 2,048 us        | 0.50x        | PROGRESSIVE |
| 10s     | 2,340| 6,073 us        | 1.49x        | FULL READ   |
| 1s      | 23,400| 60,408 us      | 14.83x       | FULL READ   |

**Recommended cadence grid**: Store levels session through 30s (~800 total bins).
Adds +2ms write overhead (vs +37ms with 1s). Every stored level reads faster than
full column data. For sub-30s analysis, read raw columns directly.

A Rust progressive reader (~100x less per-bin overhead) would shift the crossover
to ~150,000 bins, making even 1s levels read-viable. Future optimization.

### Size Budget (5 source columns per K01 file)

| Level   | Per-file     | Universe (4,604) |
|---------|-------------|-----------------|
| Session | 100 B       | 460 KB          |
| 1h      | 700 B       | 3.1 MB          |
| 15min   | 2.6 KB      | 11.7 MB         |
| 5min    | 7.8 KB      | 35.1 MB         |
| 1min    | 39 KB       | 175 MB          |
| 30s     | 78 KB       | 350 MB          |
| **Recommended** | **~128 KB** | **~575 MB** |
| 1s (optional) | 2.3 MB | 10.6 GB       |

The recommended 7-level grid (session through 30s) adds ~128 KB per file —
0.8% overhead on a 15.5 MB K01 file. The 1s level should be MI-gated:
include only when MI(1min, 1s) > 0.5 (most tickers: MI = 0.22, skip it).

## Composability Proof

Given hourly bins {sum_h, sum_sq_h, min_h, max_h, count_h}, compute session stats:

```
sum_session     = Σ sum_h          (7 additions)
sum_sq_session  = Σ sum_sq_h       (7 additions)
min_session     = min(min_h)       (7 comparisons)
max_session     = max(max_h)       (7 comparisons)
count_session   = Σ count_h        (7 additions)

mean_session    = sum_session / count_session
var_session     = sum_sq_session / count_session - mean_session²
std_session     = sqrt(var_session)
```

This extends to any pair of adjacent levels. The stat tuple is a **monoid** under
this composition — associative, with identity element {0, 0, +∞, -∞, 0}.

## Use Cases

### Coarse K04 (750x speedup)

Instead of reading 58 GB of tick data for a full-universe correlation matrix:

1. Read session-level progressive stats from all 4,604 K01 files (460 KB total)
2. Extract {mean, std} from the sufficient statistics
3. Compute z-scores, run FP16 GEMM

**Cost: ~460 KB I/O + 0.31ms GEMM = ~36ms total** vs 27s from raw ticks.

### Stale-While-Revalidate

Serve the hourly-resolution K04 instantly while the tick-level K04 recomputes.
The progressive stats are already in the K01 files — no separate cache.

### Dashboard Aggregation

Session-level stats for all 4,604 tickers fit in **460 KB**. That's a single
GPU buffer holding the entire market's summary statistics, updatable as files
are rewritten.

### Progressive Zoom

UI starts with session resolution, zooms to hourly, then minute, then second.
Each zoom reads one more level — all from the same file, with known offsets.

## Integration with MKTF v4

### No version bump needed

The progressive section uses only:
- 3 reserved Layout fields [488:512) — currently zero, readers skip them
- New section after data region — file size grows, but existing readers
  stop at `data_start + bytes_data` and never see it

Old readers see `progressive_offset == 0` and ignore the section.
New readers check the offset and use it if present.

### FLAG_HAS_PROGRESSIVE

Reserve bit 10 in flags (0x0400) for explicit progressive section indication.
This is redundant with `progressive_offset != 0` but follows the existing
flag convention for fast bitwise checks.

### Writer changes

1. After writing column data, compute sufficient statistics per cadence level
2. Pack progressive section (level directory + level data)
3. Write at next aligned offset after data region
4. Record offset/size/levels in Block 0 Layout
5. Trailing status bytes move to after progressive section

### Reader additions

```python
def read_progressive_level(path, cadence_ms) -> dict[str, np.ndarray]:
    """Read one cadence level's sufficient statistics.

    Returns dict of column_name -> float32 array of shape (bin_count, 5).
    The 5 stats are: [sum, sum_sq, min, max, count].
    """

def read_progressive_summary(path) -> dict:
    """Read the level directory without any stat data.

    Returns available levels, bin counts, and section sizes.
    """
```

## Relationship to K-Space

The naturalist's original observation: MRI k-space encodes center (low-frequency)
data first, allowing progressive reconstruction. The progressive section does the
same for financial time series:

- **Center of k-space** = session-level stats (20 bytes/col, read in microseconds)
- **Low frequencies** = hourly bars (140 bytes/col)
- **Mid frequencies** = minute bars (7.8 KB/col)
- **Full resolution** = second bars (468 KB/col)
- **Raw data** = tick-level column data (2.5 MB/col for 598K ticks)

The file IS the fractal. One file, multiple resolutions, progressive access.

## Domain as TREE Property

The progressive section is a different *domain* than the main data region.
The time-domain K01 data is `domain=TIME_SERIES`; the progressive section within
it is `domain=SUFFICIENT_STATS`. This generalizes: a frequency-domain K01 file
would be `domain=FREQUENCY`, a wavelet-domain file `domain=WAVELET`.

The domain field in the progressive section header (uint8) identifies the
sub-section's domain. The TREE-level `domain` field (future addition) identifies
the main data region's domain. The format doesn't care what's inside — it reads
the shape and records it faithfully. Node===node holds across domains.

## MI Scores as Level Selection Metadata

The section header includes mutual information scores between adjacent cadence
levels. For 4 levels, that's 3 float32 values (12 bytes). A consumer reads:

```
mi[0] = MI(session, 1h)    → 0.87  (high: hourly adds a lot)
mi[1] = MI(1h, 1min)       → 0.62  (medium: minutes add some)
mi[2] = MI(1min, 1s)       → 0.15  (low: seconds add little for this ticker)
```

Decision: skip Level 1 (1s) for this ticker — the MI score says it's not worth
the I/O. This is especially powerful for universe-scale operations where different
tickers have different information density across timescales.

MI scores are written as 0.0 when not yet computed. A separate K-SS01 leaf
(sufficient statistics metadata) can compute and backfill them.

## Open Questions

1. ~~**Should Level 1 (1s) be optional?**~~ **RESOLVED: Yes. Default grid stops at 30s.**
   Three independent findings converge: (a) MI drops to 0.22 at 1s — diminishing
   information returns. (b) Read crossover at 1,548 bins — 1s reads (61ms) are 15x
   SLOWER than full column reads (4ms). (c) 1s is 74% of progressive write cost.
   Recommended grid: session through 30s (~800 bins, +2ms write, every level reads
   faster than full columns). Include 1s only when MI(1min, 1s) > 0.5.

2. **Cadence alignment**: Bins assume market hours (09:30-16:00 ET). Pre-market
   and after-hours need a convention — separate bins? Extended session?

3. ~~**Incremental updates**~~ **RESOLVED: No.** Full-file atomic rewrite only.
   Re-deriving stats from 598K ticks is ~50ms (CPU) or <1ms (GPU fused bin stats
   kernel), negligible vs downstream I/O savings. Keeps atomic write protocol clean.

4. **NaN handling**: If a bin has zero ticks (e.g., a halted stock during one
   minute), the stat tuple is {0, 0, NaN, NaN, 0}. Composability still holds:
   sum/count remain additive, min/max use NaN-aware comparisons.

5. **Scientifically optimal cadences**: The current 1-2-5 grid is intuition-based.
   The MI experiment (naturalist) will determine which cadences carry the most
   independent information. Each cadence is a separate file, so adding/changing
   cadences is cheap. The progressive section levels should track whatever the
   MI landscape reveals as the natural resolution breakpoints.

6. **Detrended spectral evidence**: Raw spectral excess at >10min is 90-98% daily
   U-shape artifact (open/midday/close trading rate variation). The execution regime
   (1s-5min) is the durable signal. Future spectral claims must use detrended data.
   ~~Cross-ticker fingerprints at 15-30min need re-examination with detrended spectra.~~
   **RESOLVED**: Detrended cross-ticker analysis (10 tickers) shows: 15-20min
   fingerprints collapsed (CV 1.33->0.67), confirming U-shape artifact. But 5-minute
   fingerprints SHARPENED (CV 0.78->0.93), exceeding all execution-regime cadences.
   Real institutional fingerprints live at 5 minutes, composable from Level 2 (1min).
   MI scores must use detrended data to avoid the same inflation.

7. **MI scores require detrending**: The `domain` byte in the progressive section
   header can distinguish raw vs detrended stats. MI computation should use detrended
   bin stats so the level-selection metadata reflects real information content.

---

*Design emerged from pathmaker (architecture) + naturalist (k-space insight,
composability proof, size analysis, domain concept, MI metadata, detrending correction).
Prototype validated 2026-03-27. Spectral evidence corrected same day.
Production integration (format.py, writer.py, reader.py) same day.
5-minute fingerprint finding (naturalist) integrated same day.*
