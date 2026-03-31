# Persistent Provenance: .tb Headers as a Disk-Level Cache

## The realization

Manuscript 004 identifies three levels of sharing in the GPU compiler:
```
Elimination (25,714x): provenance cache — don't compute at all
Persistence (26x):     GPU residency — don't transfer again
Fusion (2.3x):         JIT kernel — don't launch twice
```

The tambear work so far has been primarily about fusion (hash scatter over sort,
tile-skip to reduce I/O). That's the third-ranked optimization.

The first-ranked optimization — elimination — is what the .tb tile headers do.
And nobody named this during the build.

## The .tb tile header as persistent provenance cache

Runtime provenance cache (manuscript 004):
```
Key:   (operation_hash, input_content_hash)
Value: computed result buffer
Store: GPU-resident HashMap, 35ns lookup
```

.tb tile header section:
```
Key:   (file, column_idx, tile_idx, operation)
Value: precomputed aggregate (min, max, sum, count, n_valid)
Store: on disk at a fixed offset, one file read
```

These are the same structure at different storage levels.

The .tb file extends provenance persistence to disk. Every time you open a
.tb file, you get a provenance cache for all tile-level aggregation queries —
a cache that was written when the file was written and never expires
(because the file is immutable after creation).

## What this means concretely

```
Query: "What is the average closing price?"
Without .tb tile headers: load 40MB column → scan 5M rows → compute mean
With .tb tile headers:    read 5KB tile header section → sum tile sums / sum tile counts

Query: "What tiles have close > 110?"
Without: load 40MB → compare each row
With:    read 5KB → tile_skip_mask_gt(stats, 110.0) → skip 56% of I/O

Query: "What is the range of prices in tile 7?"
Without: load tile 7 (512KB) → scan 65,536 rows
With:    read tile 7's stats (40 bytes) → min and max already there
```

In each case: the .tb header holds a provenance cache hit. The data section is
the fallback — the "cache miss path." For common aggregation queries on financial
data, the tile headers are hit almost every time.

The file doesn't just store data. The file IS the provenance cache.

## The ladder is a provenance expansion

The tile header ladder (from the earlier campsite) now has a clearer framing:

Each rung adds entries to the persistent provenance cache:
```
Rung 0: cache for {min, max, sum, count, n_valid} queries
Rung 1: cache for {variance} queries (+ sum_sq)
Rung 2: cache for {quantile} queries (+ KLL sketch)
Rung 3: cache for {membership} queries (+ Bloom filter)
```

The `_reserved` slot in TileColumnStats is not just "anti-YAGNI" — it is
the first expansion of the persistent provenance cache. The cost of adding
sum_sq: 8 bytes per (tile, column) at write time. The benefit: every future
variance query is a disk cache hit.

## The co-native connection

Manuscript 004:
> "An AI agent reads buffer headers from CPU to understand what the GPU holds —
> without touching GPU memory."

The 64-byte GPU buffer header (provenance hash, cost, access count, dtype) is
designed so both human and AI agents can survey GPU memory from CPU. The header
is the co-native interface.

The .tb file header (4096 bytes) follows the same principle:
- An AI agent can understand everything about a .tb file by reading the first 4096 bytes
- Column names, dtypes, n_rows, n_tiles, all column stats — all CPU-readable
- No need to "enter" the GPU or load the data section

The 64-byte GPU buffer header and the 4096-byte .tb file header are the same
design principle at different storage levels:
- Buffer header → surveys GPU memory from CPU
- .tb file header → surveys file contents from OS page cache
- Tile header section → surveys data statistics from a single small read

These three header levels form a hierarchy of co-native observation points.
An agent can understand a computational state at any level without entering
the level below it.

## The implicit hierarchy

```
GPU buffer headers (64B each):
  "What operations produced which GPU buffers?"
  Cost: read N×64 bytes = survey entire GPU memory state

.tb file headers (4096B each):
  "What is in this file? How many rows, columns, tiles?"
  Cost: read one file page = survey entire file structure

.tb tile header section (n_tiles × n_cols × 40B):
  "What are the per-tile statistics for all columns?"
  Cost: read ~5-200KB = survey all tile-level aggregation results

.tb data section (GB-scale):
  "What are the actual values?"
  Cost: read GB = actual data
```

Each level answers more questions than the one above, at higher cost.
The system is designed to answer your question at the lowest cost that suffices.

## What tambear's provenance system needs

GroupIndex.is_valid_for() is currently an O(1) length check. It works for the
immutable tambear model. But the full provenance vision (from manuscript 004) is:

1. Each GPU buffer carries a 64-byte self-describing header
2. The header includes a 16-byte BLAKE3 hash of the content
3. Equivalence check = hash comparison (74ns, cheaper than one kernel launch)
4. Provenance hit = pointer handoff (35ns)

Tambear's GroupIndex.provenance is the 32-byte BLAKE3 hash stored at build time.
When GroupIndex::build() runs, it downloads the column, hashes it, stores the hash.
The hash IS the content address of the key column.

The missing piece: instead of downloading + hashing on every build (O(n) work),
use dirty-bit tracking on the Column struct to avoid unnecessary hashing.

The Column struct should carry a `dirty: bool` flag:
- Set when the column is mutated (rows appended, values changed)
- Clear when the column's hash is computed and stored
- `is_valid_for()` checks dirty bit first (O(1)); only recomputes hash if dirty

In the immutable tambear model, dirty bits are never set after load. Zero overhead.
In a streaming model (ticks appended), dirty bits fire per-tile-append.

This is the path from O(1) length check to full content-addressed provenance.
The architecture is already designed for it. The implementation is one bool per Column.

## Why manuscript 004's hierarchy matters for tambear's roadmap

Current tambear optimizations:
- Scatter vs. sort: 17x (kernel optimization = "fusion" tier)
- Tile header skip: 99.99% less I/O (elimination from file level)

The 99.99% tile header skip IS elimination-tier optimization. It was built without
recognizing it as the highest-value tier. The naming just wasn't there.

Going forward: every feature should be classified by tier:
- Elimination tier: what can be answered from tile headers without I/O?
  → More fields in TileColumnStats, better skip mask coverage
- Persistence tier: what can stay on GPU between queries?
  → GroupIndex cache, hash scatter result cache
- Fusion tier: what kernels can be merged?
  → Mask-aware scatter (filter + groupby in one pass)

Build in this order. The elimination tier is the biggest win.
