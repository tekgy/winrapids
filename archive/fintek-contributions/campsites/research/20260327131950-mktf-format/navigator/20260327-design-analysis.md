# Navigator Analysis — MKTF Design
Created: 2026-03-27
By: navigator

## Discovery: Real AAPL K01P01 is 64% strings

Actual K01P01 layout (34.5MB total):
- price (float32): 2.47MB
- size (float32): 2.47MB
- timestamp (int64): 4.86MB
- **ticker (large_string, 1 unique "AAPL"): 7.18MB** — 21% of file
- exchange_id (int32, 17 unique): 2.47MB
- **sequence_num (large_string, numeric 1–265794): 7.88MB** — 23% of file
- **conditions (large_string, 39 unique): 7.14MB**
- is_cancel (bool): 0.07MB

Three string columns = 22.2MB = 64% of the total file. All fixable with correct types.

## MKTF Correct Encoding: 34.5MB → ~15MB (56% smaller)

| Column | K01P01 | MKTF | K01P01 MB | MKTF MB | Save |
|--------|--------|------|-----------|---------|------|
| price | float32 | float32 | 2.47 | 2.39 | 0.08 |
| size | float32 | float32 | 2.47 | 2.39 | 0.08 |
| timestamp | int64 | int64 | 4.86 | 4.78 | 0.08 |
| ticker | large_string (1 unique) | FILE METADATA | 7.18 | 0 | **7.18** |
| exchange_id | int32 (17 unique) | uint8 | 2.47 | 0.60 | 1.87 |
| sequence_num | large_string (numeric) | uint32 | 7.88 | 2.39 | **5.49** |
| conditions | large_string (39 unique) | uint32 bitmask | 7.14 | 2.39 | **4.75** |
| is_cancel | bool | uint8 | 0.07 | 0.60 | -0.53 |

No compression. No lossy encoding. Just correct types.

## Timestamp Warning

72,450 negative deltas in real AAPL data (out-of-order ticks from multi-venue aggregation).
**Delta encoding is NOT viable.** Keep absolute int64.

## Header Design (64 bytes — one NVMe sector, one GPU cache line)

```
[  0+4] magic         "MKTF"
[  4+1] version       uint8 (1)
[  5+1] compression   uint8 (0=none, 1=lz4, 2=zstd)
[  6+2] flags         uint16 (bit0=sorted_time, bit1=has_nulls, bit2=multi_venue)
[  8+2] n_cols        uint16
[ 10+2] reserved      (zeroed)
[ 12+4] schema_ver    uint32 (K01P01=1, K01P02=2)
[ 16+8] n_rows        uint64
[ 24+8] file_size     uint64 (integrity)
[ 32+8] data_offset   uint64 (abs offset to first column)
[ 40+8] meta_offset   uint64 (abs offset to JSON metadata block)
[ 48+8] coldir_offset uint64 (abs offset to column directory)
[ 56+8] created_ns    uint64 (creation timestamp)
= 64 bytes exactly
```

## Column Directory Entry (per column)

```
[0 +2]   name_len     uint16
[2 +N]   name         UTF-8 bytes
[N+2+1]  dtype_code   uint8 (0=f32, 1=f64, 2=i8, 3=i16, 4=i32, 5=i64, 6=u8, 7=u16, 8=u32, 9=u64)
[N+3+1]  encoding     uint8 (0=raw, 1=bitmask, 2=categorical, 3=delta)
[N+4+8]  offset       uint64 (absolute file offset, 64-byte aligned)
[N+12+8] nbytes       uint64
[N+20+4] null_count   uint32 (0 if no nulls)
```

## File Metadata Block (JSON, 64-byte padded)

```json
{
  "ticker": "AAPL",
  "date": "2025-09-02",
  "asset_class": "equity",
  "exchange": "SIP",
  "timezone": "America/New_York",
  "enumerations": {
    "exchange_id": {"8": "NASDAQ", "11": "NYSE", ...}
  }
}
```

## Benchmark Prediction

If NVMe throughput is constant:
- Raw binary 35.3MB → 9.2ms read
- MKTF ~15MB → **~3.9ms read** (57% smaller file)
- Selective read (2 of 7 cols, ~5MB) → **~1.3ms read**

The selective read case is where MKTF decisively beats everything. If K01P02 only needs prices+sizes for some passes, reading 5MB instead of 15MB = 3x faster per ticker.

4604-ticker universe:
- Raw binary: 4604 × 9.2ms = 42s
- MKTF full: 4604 × 3.9ms = 18s
- MKTF selective (2 cols): 4604 × 1.3ms = 6s
