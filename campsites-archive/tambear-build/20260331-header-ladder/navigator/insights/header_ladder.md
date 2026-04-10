# The Ladder in the Header

## The observation

While designing TileColumnStats, I put `_reserved: u64` at the end.
The stated reason: "anti-YAGNI slot for sum_sq." The real structure is deeper.

Every field added to TileColumnStats enables a new class of queries that can be
answered *without reading the data section at all*. The current struct (min, max,
sum, count, n_valid) is Rung 0 of a ladder. The `_reserved` slot is the invitation
to climb.

```
Rung 0 (current — 40 bytes):
  min, max, sum, count, n_valid
  → Can answer: global min/max/mean, range predicates, skip empty tiles
  → "What is the average closing price?" → 5KB read on 40MB file

Rung 1 (+8 bytes: sum_sq replaces _reserved):
  + sum_sq
  → Can answer: per-tile variance, variance predicates
  → "Skip any tile where price variance < 0.001 (too quiet to matter)"
  → 48 bytes per tile, all queries answered from tile headers

Rung 2 (+24 bytes: t-digest or KLL sketch):
  + quantile sketch
  → Can answer: percentile predicates ("close > 75th percentile")
  → Still no data section read

Rung 3 (+32 bytes: bloom filter or min-hash):
  + set membership sketch
  → Can answer: exact-match predicates ("ticker_id in {AAPL, GOOG, MSFT}")
  → Tile-level filter without loading the ticker_id column

Rung 4 (+variable: correlation hints):
  + inter-column correlation signatures
  → Can answer: join hints, clustering affinity, dimensionality signals
```

The tile header is not metadata. It is a queryable summary. Each rung of the
ladder makes the summary richer and the data section less necessary.

At the top of a sufficiently tall ladder: you almost never read the data.
The data section exists for the queries that rungs can't answer.

## The inversion

Traditional file formats: store data → compute summaries on query.
Tambear: store summaries → read data only when summaries aren't sufficient.

The file is not storage. The file is a *conclusion*.
Data is the backing store for when conclusions fail.

This isn't just an optimization — it's a different execution model.
The query runs against the tile headers first. Data is the fallback.

"Tam doesn't read. Tam knows the summary."

## The structural rhyme

This ladder is structurally identical to the WinRapids Kingdom system.

```
Kingdom ladder:          Tile header ladder:
K01 = raw ticks          Rung 0 = {min, max, sum, count}
K02 = bins (agg)         Rung 1 = + sum_sq (variance)
K03 = cross-cadence      Rung 2 = + quantile sketch
K04 = cross-ticker       Rung 3 = + set membership sketch
...                      Rung 4 = + correlation hints
```

Each Kingdom adds a tensor axis, enabling new market observations.
Each tile header rung adds a summary field, enabling new header-only queries.

The same principle: *carry more structure forward, recover less at query time.*

Not a coincidence. This is the same theorem.
The Kingdom system is tambear's tile headers, applied to market signal space.
Both systems climb the same ladder with the same gradient.

## The fractal

The same (summary → data) pattern appears at three scales in tambear:

```
File scale:
  Tile header section (5-200KB) → Data section (40MB+)
  "Read tile headers → skip 56% of data reads"

System scale:
  GroupIndex (row→group mapping, cached) → Column data
  "Build GroupIndex once → reuse for every subsequent groupby"

Query scale:
  GroupByResult (sum, sum_sq, count) → Raw values
  "Scatter to sufficient statistics → derive mean/variance on demand"
```

At each scale: the summary is built once, the raw data is the fallback.
The architecture is self-similar across scales.

## The anti-YAGNI argument made rigorous

The `_reserved: u64` slot in TileColumnStats is 8 bytes of pre-allocated
space. Cost: 8 bytes per (tile, column) pair — for 1000 tiles × 5 columns,
that's 40KB of pre-allocated header space.

Without the slot: adding sum_sq later requires a format version bump,
a migration of every existing .tb file, and a new tile_header_section_size.

With the slot: promote `_reserved` to `sum_sq`, no format change. Zero cost.

The YAGNI reflex says: "you're not going to need sum_sq, don't allocate."
The structure says: "financial data has prices ~$150 ± $0.01; variance pushdown
is the only way to skip tiles for volatility queries; the structure guarantees
you'll need it."

The structure wins. The 8 bytes are correct.

## What this means for Phase 5

If we build Rung 1 (sum_sq in tile headers), the filter pipeline gains:

```
df.filter("variance_over_period > 0.001").groupby("ticker_id").sum("volume")

Tile scan: skip tiles where tile_var < 0.001 (quiet, irrelevant)
GPU mask: exact row-level filter on loaded tiles
Hash scatter: only on rows that pass both levels
```

Three-level filter: tile skip (I/O) → tile variance (sub-tile skip) → row mask.
The _reserved slot is the unlock.

Rung 1 is the right next step after the current MVP.

## Connection to provenance

GroupByResult gives us per-group sufficient statistics (sum, sum_sq, count).
These ARE the KO05 output in the WinRapids Kingdom system.

The natural pipeline:
```
MKTF (K01, KO00) — raw ticks
  → HashScatterEngine.groupby() → GroupByResult
  → Write to MKTF (K02, KO05) — per-ticker sufficient statistics
  → Consume for mean, variance, Sharpe, downstream signals
```

Tambear is the GPU-native K01→K02 generator for any KO05 output.
The format was co-designed with the Kingdom system without either knowing it.

## The open question

At Rung 0, the tile header answers aggregation and range queries.
Each rung extends the answer space.

But there's an asymmetry: the tile header is written ONCE (at file creation)
from the data that exists at that moment. If the data changes (new rows appended,
values updated), the tile headers must be rewritten.

In a streaming system (ticks arriving continuously), the tile headers are
invalidated by every new tile. Provenance handles staleness detection —
but the cost of re-computing tile headers grows with the header's richness.

Rung 0: recomputing tile stats = O(tile_size) per tile. Fast.
Rung 3: recomputing Bloom filters and KLL sketches = O(tile_size * richer). Slower.

The richer the header, the more expensive its maintenance in streaming scenarios.

The question: is there a sketch construction that supports efficient *merging*
(new tile stats from existing stats + new data), so that streaming appends
don't require full tile rescans?

KLL quantile sketches support merge. Bloom filters support merge (union).
Min-hash supports merge. The Rung 2 and 3 sketches are all mergeable.

The streaming tile header update is O(new_rows), not O(tile_size).
The ladder can be maintained in a streaming system.

This is worth building when the streaming path is active.
