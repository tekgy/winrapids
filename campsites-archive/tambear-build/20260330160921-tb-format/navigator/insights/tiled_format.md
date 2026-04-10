# Tiled .tb Format: Navigator Notes

## The headline result

"Read 5KB of tile headers → global sum/mean/min/max of 40MB data."
99.99% of data skipped. This is not an optimization — it's a different
execution model. The file speaks for itself before any data is read.

"Tam doesn't read. Tam knows the summary."

## What tile headers enable

**Header-only aggregation** (the winner):
- Load the tile header section (~5KB) once at file-open
- Aggregate tile stats: sum tile sums, union tile min/max, sum tile counts
- Result: global stats with zero data reads
- Use case: dashboard queries, outlier detection, sanity checks

**Predicate pushdown** (the second winner):
- Filter "close > 110": for each tile, check `tile_header.close.max < 110`
- If max < threshold: entire tile can't satisfy predicate → skip I/O entirely
- Result: 56% of tiles skipped in experiments
- Combined: load only relevant tiles + mask-not-filter within each tile

**Streaming (capability, not optimization)**:
- For data that doesn't fit in VRAM: process tile-by-tile
- Async double-buffered: load tile K+1 while GPU processes tile K
- Python streaming is 20x slower (Python overhead per tile) — Rust is fine

## The two-level filter architecture

```
df.filter("close > 110").groupby("ticker_id").sum("volume")

Level 1: tile header scan (5KB read, O(n_tiles))
  → tile 0: max_close = 95.3 → SKIP
  → tile 1: max_close = 125.7 → LOAD
  → tile 2: max_close = 108.2 → SKIP
  → tile 3: max_close = 155.0 → LOAD
  [56% skip rate]

Level 2: GPU mask-not-filter (on loaded tiles only)
  → bitmask: row 0 close=112.3 → 1 (passes)
  → bitmask: row 1 close=98.5  → 0 (fails)
  [per-row precision within tile]

Level 3: hash scatter (mask-aware, on unmasked rows only)
  → atomicAdd to group accumulators for rows where mask bit = 1
```

No compaction between levels. Each level passes metadata to the next.

## The .tb layout that enables this

Key insight: ALL tile headers must be contiguous at the start of the file
(or in a small early section), NOT interleaved with tile data.
This enables "read 5KB → know everything" with a single small read.

```
[File Header: 4096 bytes, fixed]
  magic, version, n_rows, n_columns, pipeline_dtype
  tile_size (rows per tile), n_tiles
  tile_header_section_offset, tile_header_section_size  ← points to all tile stats
  data_section_offset  ← where tile data starts
  dictionary_sections, group_index_metadata, scratch_regions, provenance_offset

[Tile Header Section: n_tiles × n_columns × 40 bytes]
  Per tile, per column: { min: f64, max: f64, sum: f64, count: u32, n_valid: u32 }
  Total for 1000 tiles × 5 columns × 40B = 200KB — one read for all stats

[Data Section: column-major tiled]
  Column 0: [tile_0_data | tile_1_data | ... | tile_{n_tiles-1}_data]
  Column 1: [tile_0_data | tile_1_data | ... | tile_{n_tiles-1}_data]
  ...
  Tile k, column c data offset = data_section_offset
                                + c * n_tiles * tile_size * dtype_bytes
                                + k * tile_size * dtype_bytes

[Mask Section: optional, same tiled layout as data]
[Scratch Regions: pre-allocated at file creation]
[Provenance Tail: appended by each execution]
```

Column-major layout: loading column C, tiles J through K = one contiguous
disk read. Predicate pushdown = load a subset of that range.

## Tile size

Default: 65,536 rows (2^16). Rationale:
- 65,536 × 8 bytes = 512KB per column per tile
- Fits L2 cache on modern GPUs (useful for GPU-side tile processing)
- Enough rows per tile that header overhead is negligible
- Fine-grained enough for predicate pushdown (56% skip = useful granularity)

Store tile_size in the header — anti-YAGNI, different datasets may want
different tile sizes. Don't hardcode.

## Per-tile stats struct (40 bytes per column per tile)

```rust
#[repr(C)]
pub struct TileColumnStats {
    pub min: f64,       // 8 bytes
    pub max: f64,       // 8 bytes
    pub sum: f64,       // 8 bytes — use KahanSum during write for accuracy
    pub count: u32,     // 4 bytes — rows in this tile (may be < tile_size for last tile)
    pub n_valid: u32,   // 4 bytes — non-null rows
    pub _reserved: u64, // 8 bytes — future use (e.g., sum_sq for variance pushdown)
}
// Total: 40 bytes. Pad to 64 bytes for cache alignment? 24 bytes wasted.
// Keep at 40 for now — can change before 1.0 (no backward compat).
```

Note: `_reserved` is the anti-YAGNI slot. sum_sq in the tile header would
enable variance pushdown ("skip tile if tile_var < threshold"). Build it now.

## The GroupIndex connection

GroupIndex::build() becomes tile-aware: only scatter rows from non-skipped tiles.
The GroupIndex can carry a tile_skip_mask (one bit per tile) from predicate
pushdown. This eliminates even the row_to_group scatter for skipped tiles.

Phase 4 builds the tile format. GroupIndex tile-awareness is Phase 4b.
For now: note the extension point in GroupIndex::build().

## What observer should build first

Priority order (matching team lead's directive):
1. TileColumnStats struct (40 bytes, compile-time stable)
2. Tile header section serialization (write path: scan tile data → fill stats)
3. Tile header section deserialization (read path: mmap header → stats in memory)
4. Header-only aggregation: global_sum(), global_min(), global_max() from tile headers
5. Predicate pushdown: tile_skip_mask(col, predicate) → Vec<bool>
6. Tiled data read (using skip mask to skip I/O)
7. Streaming I/O (async double-buffer, Rust async) — Phase 4b
