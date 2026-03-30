# tb-format + no-counting-pass: Navigator Notes

## The principle

"Tam doesn't count. Tam over-allocates or reads from provenance."

Over-allocation + provenance replaces counting passes everywhere:
- GroupBy: allocate `max_key + 1` accumulators (from header). 0.022ms vs 0.026ms exact — FASTER.
- Join: allocate `n_fact` output (inner join upper bound). Use bitmask for non-matches.
- Dedup: allocate `n_rows` output. Track count during execution.
- Subsequent queries: provenance has actual sizes. 35ns lookup → exact allocation.

## The .tb file IS the workspace

The most elegant implication: pre-allocated scratch regions in the .tb file itself.

```
[.tb header: magic + schema + column descriptors + group index metadata]
[column data at fixed byte offsets]
[optional mask section]
[scratch region: groupby accumulators, sized max_key+1 × n_numeric_cols × 8]
[scratch region: dedup output, sized n_rows × max_col_bytes]
[provenance tail: actual sizes from previous executions]
```

When the file is opened (or loaded to GPU via DirectStorage):
- `groupby_scratch_ptr = file_base + groupby_scratch_offset`  (from header)
- Zero cudaMalloc during execution for the common case.
- Output pointers known at file-open time.

The provenance tail grows: each execution appends actual output sizes. Subsequent
executions read the tail and use exact allocations. First execution pays the
over-allocation cost once.

## What the .tb header needs

Fixed portion (64-byte cache-line multiple, anti-YAGNI):

```
magic:                 u32  = 0x544D4245  ("TMBE")
version:               u16
reserved:              u16  [future flags]
n_rows:                u64
n_columns:             u32
pipeline_dtype:        u8   (DType repr)
has_mask:              u8   (bool)
n_group_indices:       u32
n_scratch_regions:     u32
provenance_tail_offset: u64  (byte offset to provenance section)
[padding to 64-byte boundary]
```

Then per-column descriptors (fixed size, say 128 bytes each):
```
name:           [64]u8   (null-padded)
dtype:          u8
encoding:       u8       (0=Raw, 1=Dictionary)
_reserved:      [2]u8
max_key:        u32      (for integer columns used as groupby keys; 0 otherwise)
byte_offset:    u64
byte_len:       u64
dict_offset:    u64      (byte offset to dictionary section; 0 if not dictionary)
n_dict_entries: u32
[padding to 128 bytes]
```

Then scratch region descriptors (per region):
```
purpose:        u8       (0=groupby, 1=join, 2=dedup)
_reserved:      [3]u8
byte_offset:    u64
byte_len:       u64
```

## The accumulator_size / n_active distinction

This is now in GroupIndex:
- `accumulator_size` = max_key + 1. Pre-allocated. From header.
- `n_active` = groups with count > 0. Lazily populated by provenance.

NEVER compute `n_active` as a prerequisite for allocation.
`n_active` is a RESULT, not a prerequisite. Provenance stores results.

## Connection to winrapids-store

The provenance tail of the .tb file IS a miniature winrapids-store for this specific
dataset. The full provenance store generalizes this. Eventually: .tb provenance syncs
with the GPU store's provenance graph. But for Phase 4, the file-local provenance
tail is sufficient.
