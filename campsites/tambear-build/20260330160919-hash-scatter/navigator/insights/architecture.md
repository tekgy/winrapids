# hash-scatter: Navigator Notes

## What I scaffolded

`crates/tambear/` is now live with:
- `frame.rs` — Frame, Column, DType core types
- `group_index.rs` — GroupIndex with `build()` stub
- `hash_scatter.rs` — HashScatterEngine stub
- `stats.rs` — RefCenteredStatsEngine stub (naturalist's territory)
- `main.rs` — integration checklist

## The architectural decision that matters

**GroupIndex is the killer feature, not just hash scatter.**

Hash scatter alone is 17x. But the persistent GroupIndex makes every *subsequent* groupby O(n_groups) metadata check + O(n) scatter — no group discovery, no hash table rebuild. The first groupby pays O(n). Every subsequent groupby on the same key is free for discovery.

The CSR question: `row_to_group[i]` (n_rows ints) is sufficient for scatter-based groupby. CSR (group_offsets + group_rows) is needed for group-wise iteration. I kept it minimal — `row_to_group` + `group_counts`. Scout can add CSR as a lazy field when group-wise operations are needed.

## The provenance stub

`is_valid_for()` currently returns `false` always (always rebuild). The real impl hashes the key column's GPU buffer via BLAKE3, comparing against stored provenance. A dirty-bit approach would be even cheaper: if the column hasn't been written since the index was built, skip the hash. The dirty bit is O(1) to check. This is a Phase 4b concern — get the index building first.

## The direct-index assumption

The `GroupIndex::build()` stub assumes keys in [0, n_groups) — direct index (key IS group id). This works for integer ticker IDs, bin IDs, etc. For string keys or non-contiguous integers, we need a hash table mapping key → group_id. Start with direct-index; generalize when needed (anti-YAGNI: we DO know we'll need string keys eventually).

## Sort-free contract enforced here

The doc comment in lib.rs states the contract explicitly. When the compiler is built on top of tambear, it will reference this contract. No path through tambear emits sort for groupby — the compiler can't accidentally route around it because there IS no sort-based groupby implementation.
