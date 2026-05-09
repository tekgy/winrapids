# Recipe trees — the catalog-as-tree pattern

**Status**: Practice template introduced 2026-05-08, anchored on naturalist's
`~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md`.

## What this is

A directory of per-family catalog trees. Each file maps a literature-named
recipe family (means, distances, correlations, kernels, sketches, tail
estimators, etc.) onto its underlying kernel(s) + parameter axes.

The pattern is the naturalist's: *names are parameter assignments on a
graph; the graph is the actual catalog.* When you implement by name, you
build N implementations and discover later they share a kernel; when you
implement by graph, you build the kernel and let names attach to parameter
assignments.

## Why this directory exists separately from `adding-a-recipe.md`

`adding-a-recipe.md` is the operational template — *here is the file
layout for a single recipe.* This directory is the **structural** template
— *here is the catalog topology a recipe family lives inside.*

The two compose. When pathmaker adds a new recipe, the question
`adding-a-recipe.md` doesn't yet ask is: *is this an independent recipe
or a parameter assignment on an existing kernel?* The catalog tree for
the relevant family answers it. If a tree exists for the family, the
new recipe is mapped onto an existing branch + parameter assignment, or
extends the tree by surfacing a new axis. If no tree exists, the answer
defaults to "independent recipe" and the family is unmapped.

## How to use a tree

Three audiences, three uses:

**Catalog browser (user)**: walk the tree from root to the leaf they want.
Adjacent leaves are adjacent in parameter space — when one method doesn't
fit, the obvious next method to try is structurally next-to-it, not
alphabetically next-to-it. This is `discover()` from a different angle:
the relationship IS the answer.

**Recipe author (pathmaker)**: before implementing a new recipe, find
its position in the relevant tree. If the position is on an existing
kernel branch, the implementation is a parameter assignment + a recipe
wrapper (~20 lines), not a new kernel. If the position is a new axis,
extend the tree first; implement second.

**Reviewer (math-researcher / aristotle / observer)**: walk the tree
looking for unmapped literature variants (gaps the team hasn't built yet)
and unparameterized kernel-overlaps (places where two named recipes
should collapse into one kernel + two recipe wrappers).

## How to add a tree for a new family

1. **Identify the kernel(s).** Most families have one kernel; some have
   several with overlapping subsets (means is the canonical example —
   GeneralizedMean and TransformedMean produce identical answers on a
   shared subset of named leaves via different parameterizations). Both
   patterns are valid; document them.
2. **Identify the parameter axes** for each kernel. Each axis is a
   `using()` knob in the eventual recipe API.
3. **Map every literature-named variant** to a kernel + parameter
   assignment. Multiple-paths-to-same-leaf is structural information to
   surface, not a bug.
4. **Surface the gaps** — parameter combinations no literature has
   named but tambear should expose anyway, per anti-YAGNI.
5. **Surface the overlaps** — where two kernels produce the same
   answer for some leaves and different answers for others. These are
   the design questions that decide which kernel becomes "primary" in
   the implementation.
6. **Identify the accumulate+gather decomposition** for each kernel.
   The cleanest decomposition typically wins as the implementation
   target; the other kernels become recipe wrappers if the named-leaves
   set is wide enough to deserve the syntax.

## Trees in this directory

- `means.md` — the centrality/central-tendency family (~30 named
  literature variants across 5 kernels). First pilot.

Future: `distances.md`, `correlations.md`, `kernels.md`, `sketches.md`,
`tail-estimators.md`, `dispersions.md`, `entropies.md`, `divergences.md`,
`information-criteria.md`...

The catalog grows organically — one family at a time, ratified by
math-researcher, used by pathmaker on next-recipe-add to that family.
No team-wide deadline. The trees accumulate.
