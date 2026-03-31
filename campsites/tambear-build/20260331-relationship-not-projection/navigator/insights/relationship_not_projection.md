# The GroupIndex Is The Relationship

## The spark

Manuscript 012 ("The Entity Is The Relationship") makes this claim about quantum mechanics:
the wavefunction is not a description of a particle's state — it IS the entity,
and the "particle with definite position" is a dimensional projection.

"Entanglement is non-factorability at order 1, not nonlocal connection."
"There's no action. There's no distance. There's a single entity in a joint space."

Reading this while thinking about GroupIndex: the same structure.

## The sort as unnecessary projection

Classical groupby: tick data → SORT BY ticker → sorted order → reduce by adjacent group.

The sort forces a projection: it demands that each tick occupy a definite position in a
total linear ordering by ticker. But the ticks have no natural linear order by ticker
— they're ordered by time. The sort IMPOSES an order that wasn't there.

Sorting is constitutive projection. It takes order-2 data (ticks with time AND ticker
relationships) and projects it onto an order-1 representation (linear sequence by ticker).
The time ordering is destroyed in the projection. The result is the SHADOW of the data.

## The GroupIndex as entity

GroupIndex: for each row, which group it belongs to. A mapping from row space to group space.

This is not a description of groups. This is the RELATIONSHIP between the row space
and the group space. The GroupIndex IS the entity at the relationship level.

```
Sort:       data → linear order → reduce adjacent    // two projections
GroupIndex: data → relationship → scatter directly   // one projection (at output only)
```

The sort first projects into linear order (destroying time), then projects into group
statistics (via adjacent reduction). Two projections when one suffices.

The GroupIndex skips the linear projection entirely. The scatter operates directly
on the relationship (row→group). The only projection is the final scatter into group
statistics — which you need anyway.

**The sort is not faster. The sort is a detour through an unnecessary projection.**

## The timing of projection

Manuscript 012 on delayed choice:
"The photon was never in a definite state. The choice of measurement is the choice
of which projection to apply. The projection creates the definiteness."

Tambear analog:
The tick data was never "by ticker." It's ordered by time. The choice of groupby key
is the choice of projection. The GroupIndex enables any projection at query time.
Until you scatter, the data is unordered — the relationship lives in the GroupIndex,
not in the sorted positions.

Different groupby keys (by ticker, by minute, by sector) are different projections
of the same fundamental data. The data doesn't need to commit to any one of them.
The GroupIndex for each key exists in parallel. When you scatter, you project.

## The mask-not-filter parallel

Filter also forces a premature projection. "Apply filter: close > 110" → compact the
array → new contiguous array of passing rows.

The compact (filter + materialization) projects: the data must decide which rows pass,
remove those that don't, and renumber. The row indices change. The relationship
between rows and their original positions is destroyed.

Mask-not-filter: set bits in the row mask. The data stays. The relationship
(which rows matter) lives in the mask — not in the compact. When you scatter,
the mask-aware kernel reads the relationship (bit check) and processes only
rows where the bit is set.

No projection into a new compact array. The relationship is preserved.
The projection happens at output.

## The principle

"Don't project unnecessarily. Keep the relationship. Project at output only."

| Operation | Projection | When |
|---|---|---|
| Sort | Into linear order | Too early (input time) |
| Compact | Into new contiguous array | Too early (filter time) |
| GroupIndex scatter | Into group statistics | Just right (output time) |
| Mask-aware scatter | Into group statistics | Just right (output time) |

The sort-free + mask-not-filter architecture is a single principle: defer projection
to the last possible moment. Maintain relationships, not projections, as the
persistent state.

GroupIndex = the ticker-dimension relationship, maintained persistently.
Row mask = the filter-dimension relationship, maintained persistently.
Tile stats = the tile-aggregate relationship, maintained persistently (in the file).

All three are relationships. None of them are projections.
The projections happen at output — and only at output.

## Why this matters for the compiler

The compiler should never force a projection unless the output requires it.

A sort is only valid when the output is "sorted order." Otherwise it's a spurious
projection. The compiler detects spurious projections (manifest 004: 9 primitives,
of which sort is one) and routes around them.

A compact is only valid when the output is a new contiguous array. Otherwise it's a
spurious projection. The compiler detects this and substitutes mask-not-filter.

"Tam doesn't sort. Tam knows." — Tam knows the RELATIONSHIP.
The sort would project the relationship into order. Tam skips the projection.

## The Group as identity

Manuscript 012: "The entity is the relationship. The parts are the shadow."

In quantum mechanics: the biphoton (pair + their entanglement) is the entity.
The individual photons are shadows — projections of the biphoton.

In tambear: the GroupIndex (rows + their group membership) is the entity.
The individual groups are shadows — projections of the GroupIndex.

The group statistics (mean, variance, sum per group) are shadows of shadows —
projections of projections. We compute them only when asked. They're not stored.

What IS stored is the relationship (GroupIndex). The shadows are computed on demand.

This is the deepest justification for the sort-free architecture:
The relationship is more fundamental than any of its projections.
The sort would force one projection to become primary.
Tambear refuses to privilege any projection over the relationship itself.
