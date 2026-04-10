# format.rs Scaffold: Navigator Notes

## What was built

`crates/tambear/src/format.rs` — the shared type layer for the .tb format.

Observer builds the write/read paths on top of these types.
Scout and naturalist use TileColumnStats if they need tile-aware groupby.

## The three types

**TileColumnStats** (40 bytes, const-asserted):
- min, max, sum (f64 each), count, n_valid (u32 each), _reserved (u64)
- `from_slice(&[f64])` — Kahan summation, scan once, produce all stats
- `can_skip_gt/lt/gte/lte(threshold)` — predicate pushdown per tile
- `_reserved` slot pre-built for sum_sq (variance pushdown, anti-YAGNI)

**TbColumnDescriptor** (56 bytes, const-asserted):
- name [u8;32], dtype u8, encoding u8, max_key u32, _reserved [u8;16]
- `new(name, dtype, encoding, max_key)` — builds from str
- `name_str()` — decodes null-padded name back to &str

**TbFileHeader** (4096 bytes, const-asserted):
- All scalar fields laid out with explicit padding — no implicit repr(C) surprises
- `columns: [TbColumnDescriptor; 64]` — up to 64 columns in the header itself
- `new(n_rows, n_columns, pipeline_dtype, tile_size, columns)` — computes offsets
- `tile_stats_offset(tile, col)` — absolute file offset of a TileColumnStats
- `tile_data_offset(col, tile)` — absolute file offset of column-major tile data
- `tile_byte_size(tile)` — handles last-tile partial size
- `as_bytes()` / `from_bytes()` — byte casting (safe: explicit zeroed padding)

## The size math (critical — verified by const assertions)

Scalars section:
```
magic [u8;8]       = 8
version u32        = 4
_pad0 [u8;4]       = 4   ← explicit, aligns n_rows to 8
n_rows u64         = 8
n_columns u32      = 4
pipeline_dtype u8  = 1
_pad1 [u8;3]       = 3   ← explicit, aligns tile_size to 4
tile_size u32      = 4
n_tiles u32        = 4
tile_hdr_offset u64 = 8
tile_hdr_size u64  = 8
data_offset u64    = 8
provenance_offset  = 8
_reserved_scalars  = 24
                    ────
                     96 bytes
```

Column array: 64 × 56 = 3584 bytes.
Reserved tail: 416 bytes.
Total: 96 + 3584 + 416 = **4096 bytes** ✓

## Navigation formulas

Tile header section starts at offset 4096 (immediately after file header).

TileColumnStats offset for (tile t, column c):
```
4096 + (t * n_columns + c) * 40
```

Column data offset for (column c, tile t) — column-major:
```
data_section_offset + c * n_tiles * tile_size * dtype_bytes + t * tile_size * dtype_bytes
```

Loading all tiles J..K for column C = byte range:
```
[tile_data_offset(C, J) .. tile_data_offset(C, K+1))
```

One contiguous read. Predicate pushdown = load only the non-skipped subset.

## What observer builds next

1. `TbWriter` struct — writes file header, tile stats, column data in one pass
2. `TbReader` struct — mmaps file header, reads tile header section, implements predicate pushdown
3. `TbReader::header_only_agg()` — uses `global_min/max/sum/mean()` from format.rs
4. `TbReader::tile_skip_mask()` — delegates to `tile_skip_mask_gt/lt()` from format.rs

The global aggregation helpers are already implemented in format.rs.
Observer does NOT need to reimplement them.

## The connection to GroupIndex

`GroupIndex::build()` takes `max_key` from the `.tb` header.
`TbColumnDescriptor.max_key` carries this value.
No counting pass — the header already knows.
