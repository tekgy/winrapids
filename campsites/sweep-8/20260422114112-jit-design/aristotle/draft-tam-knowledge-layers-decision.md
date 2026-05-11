# Decision on `docs/tam-knowledge-layers.md`

**Date:** 2026-04-22 · **Author:** aristotle

Team-lead asked whether my earlier draft (`draft-tam-knowledge-layers.md`)
should land as-is, or whether it's now redundant given that
`docs/LIVE_COMPILER.md` has absorbed the three-layer + three-cache +
future-dependence framing via DEC-020 + DEC-021.

## Audit

Reading `docs/LIVE_COMPILER.md` lines 100-169, it already covers:

- The three-layer table (identical to my draft's)
- The lift/sequential boundary at layer 2→3 with worked examples
- The negative formulation ("TAM can't tell the future")
- The three-cache table with invalidation policies
- Content-addressed-reconstructible-from-source-of-truth discipline

Plus the earlier LIVE_COMPILER.md sections cover state conservation
+ provenance comprehensively.

## Decision

**Withdraw my full draft of `tam-knowledge-layers.md`.** Replace with
a tiny pointer doc that cross-links `LIVE_COMPILER.md` + `DEC-021`
for Sweep 8 code comments to reference. This avoids
duplicate-source-of-truth (which would itself violate DEC-020 at
the documentation layer).

## Proposed `docs/tam-knowledge-layers.md` (short pointer version)

```markdown
# TAM Knowledge Layers — Pointer

This short pointer exists because `crates/tambear/src/jit/strategy.rs`
(and other substrate code) references "layer N" in doc-comments. The
canonical definition of the layers lives in two authoritative places:

- `docs/LIVE_COMPILER.md` § "The three layers of TAM knowledge"
  and § "The three caches at three layers"
- `docs/decisions.md` [DEC-021] — architectural lock

Read either first. This pointer is intentionally short; maintaining
it as a redundant deep-dive would violate state conservation at the
documentation layer (two sources of truth for the same concept).

## Summary for quick reference

Three layers:

- **Layer 1 — eternal truths** (proof engine). Algebraic facts about
  Ops / Groupings / atoms. No pipeline, no data needed.
- **Layer 2 — composition truths** (pipeline compiler + data-quality
  analyzer). Everything TAM can know BEFORE dispatch: pipeline
  structure × data profile × hardware capability × Op-composition
  facts.
- **Layer 3 — numerical truths** (JIT dispatch). The actual output
  values. The one thing TAM cannot predict.

Three caches with matching invalidation policies:

| Cache | Layer | Invalidates when |
|---|---|---|
| Proof-engine StructuralFact | 1 | Op enum changes (almost never) |
| Pipeline schedule | 2 | Recipe IR / assumptions / data profile / hardware change |
| Kernel binary | 3-materialization | Tambear version / kernel-IR / Shape / door change |

The **lift/sequential boundary** is exactly layer 2 → 3: a step is
sequential iff its control flow depends on a yet-uncomputed value.
Otherwise, liftable.

See `LIVE_COMPILER.md` for worked examples, rationale, and the
conservative-profile invariant.
```

Total ~40 lines. Lives as a per-layer reference jump-table, not a
redundant explainer.

## If team-lead prefers no pointer doc at all

Even smaller option: don't create `tam-knowledge-layers.md` at all.
The Sweep 8 code comments reference `LIVE_COMPILER.md` + `DEC-021`
directly. Easier to maintain; one fewer file to keep in sync.

My lean: **skip the pointer doc.** Code comments can cross-link to
canonical sources; there's no meaningful jump-table value in a
40-line intermediary. Pathmaker's call.

## What I'm doing instead

I withdraw the full draft file
(`draft-tam-knowledge-layers.md` in this campsite) from the "to
land in `R:\tambear\docs\`" queue. It stays in the campsite as a
historical reference for future aristotle; `LIVE_COMPILER.md` is
the canonical location.
