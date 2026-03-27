# MKTF Header as Single Source of Truth
## Scout Design Notes — 2026-03-27

---

## State of v3

Pathmaker's MKTF v3 already has the core pattern right. Block 0 (first 4096-byte sector, 31us read) contains:

```
[0:16]   Core: magic, version, flags (is_complete in bit 0), alignment, header_blocks
[16:98]  Identity: leaf_id, ticker, day, cadence, leaf_version, schema_fingerprint
[98:120] Dimensions: n_rows, n_cols, n_bins, bytes_data
[120:128] Quality: total_nulls
[128:144] Provenance: write_timestamp_ns, write_duration_ms, compute_duration_ms
[144:184] Layout: dir_offset, dir_entries, meta_offset, meta_size, data_start
[184..4096) RESERVED (3912 bytes free)
```

**Header scan result:** 31us per file, 151ms for 4,604 tickers. The daemon already works.

---

## The Single Gap: Upstream Fingerprints in Block 0

Currently, upstream provenance (which K01 produced this K02) goes into JSON metadata at Block N.
For the daemon's staleness check, this means two sector reads instead of one.

**What the daemon needs for K02 staleness (from Block 0 only):**
1. K02's `upstream_write_ts` — "what was K01's write_timestamp_ns when I was computed?"
2. K02's `upstream_leaf_id` — "which upstream node?" (to find the K01 path)

Compare against K01's current `write_timestamp_ns`. If K01 is newer → K02 is stale.

**Proposed addition to Block 0 reserved space (offset 184):**

```
Upstream fingerprint table (up to 4 entries, 40 bytes each = 160 bytes total)
Each entry:
  [0:16]  upstream_leaf_id   bytes[16]   null-padded, e.g. "K01P01"
  [16:24] upstream_write_ts  int64       copy of upstream's write_timestamp_ns at compute time
  [24:32] upstream_data_hash uint64      xxHash64 of upstream's data bytes (0 = not computed)
  [32:40] reserved           bytes[8]

Location in file:
  UPSTREAM_TABLE_OFFSET = 184
  UPSTREAM_ENTRY_SIZE = 40
  UPSTREAM_MAX_ENTRIES = 4

Total: 160 bytes at [184..344). Remaining reserved: [344..4096) = 3752 bytes.
```

**For K01 (root node):** all 4 entries are zeroed. upstream_leaf_id = b"\x00" * 16.
**For K02:** entry 0 has upstream_leaf_id = "K01P01", upstream_write_ts = K01's timestamp.

This is the only addition v3 needs to eliminate ALL sidecar files.

---

## The "Files Win" Invariant

```
Priority order (highest to lowest):
  1. File with is_complete=true    → canonical truth
  2. File with is_complete=false   → corrupt, DELETE and recompute
  3. File absent                   → not computed
  4. BitmapStateDB entry           → READ CACHE only, NEVER source of truth

If BitmapStateDB disagrees with file state:
  - DB says DONE but file has is_complete=false → files win, delete+recompute
  - DB says DONE but file absent → files win, recompute
  - DB says NOT_DONE but file has is_complete=true → DB is stale, update DB

BitmapStateDB is rebuilt by scanning headers. It cannot contradict files.
It can be deleted and rebuilt at any time with no data loss.
```

---

## Reconcile Algorithm (full specification)

```python
def reconcile(root_dir: Path, state_db: BitmapStateDB) -> ReconcileReport:
    """
    Scan all MKTF files. Files win over state_db. Returns what changed.

    For each file, reads ONLY Block 0 (31us). Zero data bytes touched.
    Total for 4,604 tickers: ~151ms.
    """
    state_db.clear()
    deleted = []
    stale = []

    for path in walk_mktf_files(root_dir):
        h = read_header_block0_only(path)  # reads exactly 4096 bytes

        # 1. Completeness check
        if not h.is_complete:
            path.unlink()  # incomplete write, atomic delete
            deleted.append((path, "incomplete"))
            continue

        # 2. Staleness check (for derived files with upstreams)
        for entry in h.upstream_fingerprints:
            if entry.upstream_leaf_id == "":
                break  # no more upstreams

            upstream_path = resolve_path(
                root_dir, h.ticker, h.day, entry.upstream_leaf_id
            )

            if not upstream_path.exists():
                path.unlink()
                deleted.append((path, "upstream_gone"))
                break

            upstream_h = read_header_block0_only(upstream_path)
            if not upstream_h.is_complete:
                path.unlink()
                deleted.append((path, "upstream_incomplete"))
                break

            if upstream_h.write_timestamp_ns > entry.upstream_write_ts:
                path.unlink()
                stale.append((path, "upstream_changed"))
                break
        else:
            # All upstreams valid (or no upstreams)
            state_db.mark_done(h.ticker, h.day, h.leaf_id, h.leaf_version)

    return ReconcileReport(deleted=deleted, stale=stale,
                           done_count=state_db.count())
```

---

## Daemon Staleness Check (hot path, per-ticker)

```python
def is_stale(k02_path: Path, k01_path: Path) -> bool:
    """
    Check if K02 needs recomputation. Two header reads, ~62us total.
    Called before each pipeline run.
    """
    if not k02_path.exists():
        return True  # not computed yet

    k02_h = read_header_block0_only(k02_path)

    if not k02_h.is_complete:
        k02_path.unlink()  # corrupt, clean up
        return True

    if not k01_path.exists():
        k02_path.unlink()  # upstream gone, clean up
        return True

    # Read upstream fingerprint from K02's Block 0 (already in k02_h)
    upstream_entry = k02_h.upstream_fingerprints[0]

    k01_h = read_header_block0_only(k01_path)

    if not k01_h.is_complete:
        return True  # K01 itself is broken

    # The critical comparison: is K01 newer than when K02 was computed?
    return k01_h.write_timestamp_ns > upstream_entry.upstream_write_ts
```

---

## BitmapStateDB Rebuild Protocol

```python
def rebuild_state_db(root_dir: Path) -> BitmapStateDB:
    """
    Rebuild from scratch. Called on startup or after corruption.
    ~151ms for full universe.
    """
    db = BitmapStateDB()

    for path in walk_mktf_files(root_dir):
        h = read_header_block0_only(path)
        if h.is_complete:
            db.mark_done(h.ticker, h.day, h.leaf_id, h.leaf_version)
            # Note: no staleness check here — that's reconcile's job

    return db
```

The DB does NOT check upstream freshness on rebuild. That's intentional — rebuild is fast
(just scan is_complete). A separate reconcile pass does the deep check with upstream comparison.

---

## Why This Pattern is Correct

The NIfTI insight applies exactly: a valid NIfTI file with `sizeof_hdr=348` is self-sufficient.
You don't need a database to know if the file is valid — the file tells you.

For MKTF:
- `is_complete=true` + correct `schema_fingerprint` = file is trustworthy
- `upstream_write_ts` = auditable provenance without external logs
- Column `min/max/null_count` = data quality without reading data

The daemon's entire operation becomes: read Block 0, make a decision, take an action.
Data bytes are touched only when computation actually happens. Everything else is header reads.

---

## What This Doesn't Replace

The JSON metadata block (Block N) still has value for:
- Full condition_bits mapping (which bit = which exchange code)
- Extended string metadata (analyst notes, pipeline config snapshot)
- Future: compression parameters, custom encoder state

But nothing in the JSON block is load-bearing for operational decisions.
The daemon never reads it. Reconcile never reads it. Only analytical tools do.

---

## Implementation Delta from v3

What needs to change in `mktf_v3.py`:

1. Add `UPSTREAM_TABLE_OFFSET = 184`, `UPSTREAM_ENTRY_SIZE = 40`, `UPSTREAM_MAX_ENTRIES = 4`
2. Add `upstream_fingerprints` parameter to `write_mktf()`
3. Pack upstream fingerprint table into Block 0 at offset 184
4. Add `upstream_fingerprints` to `MKTFHeader.__slots__` and parse in `read_header()`

That's it. Everything else v3 already has.
