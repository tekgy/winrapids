# Phase 1-8 Deconstruction — Multi-Dim Shape Extension

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

**Trigger:** Tekgy broadcast (Note 1):

> Current `Shape::dim: DimHint` is single-axis only. That can't
> represent matrix×matrix, volume×volume, or timeseries-of-volumes —
> all of which we WILL implement (CLAUDE.md §10 "every measure in
> every family"). Deferring this is classic YAGNI reflex. Extend per
> YAWNI:
>
> - `dim: DimHint` → `dims: Vec<DimHint>` (per-axis)
> - `DimHint::Bounded { multiple_of: usize }`
> - `DimHint::SymbolicEqual(axis_ref)`
> - `layout: Layout` (RowMajor / ColumnMajor / Strided / Ragged)
> - `alignment: AlignmentClass` (Unaligned / Aligned16 / Aligned64 / Aligned256)

The surface change looks trivial: replace one field with four.
**But under the surface, each of those four fields implicates a cache-
key decision, a codegen-discipline decision, and a compatibility-
sharing decision.** Running through Phase 1-8 to get these right.

---

## Phase 1 — Assumption Autopsy

What does the current single-axis `DimHint` implicitly assume?

- **A1 — There is exactly one axis.** Single `dim` field. Matrix recipes
  can't represent themselves. Breaks on the first covariance matrix.
- **A2 — Rank is implied by the field count** (always 1). No explicit
  rank. A future 2D recipe has no way to say "I am 2D" beyond Grouping::Tiled
  hinting, which conflates *access pattern* with *data rank*.
- **A3 — All axes are equivalent.** No distinction between rows and
  columns, between batch and feature, between time and space. A
  future broadcast rule can't say "broadcast across axis 0 but not
  axis 1."
- **A4 — Memory layout is implied by dtype + length.** The current
  Shape doesn't carry strides; it implicitly assumes contiguous
  row-major. Breaks on views, slices, tiled storage, or any ndarray-style
  input.
- **A5 — Alignment is always "whatever the allocator gave us."** No
  way to declare 64-byte-aligned input for SIMD-friendly kernels; no
  way for codegen to emit aligned loads vs unaligned loads.
- **A6 — Axis sizes are independent.** No way to say "this is a square
  matrix" or "these two axes must match." Each recipe that wants that
  invariant re-implements it as a runtime panic.
- **A7 — Size specialization is binary.** `Static(n)` (bake) or
  `Dynamic` (pass). No middle ground like "multiple of 8" (SIMD-
  friendly) or "constrained to a family." Cache-key blows up when we
  Static-specialize large data; passes too much when we Dynamic all
  of it.
- **A8 — All axes have the same DimHint kind.** If one axis is Static
  and another is Dynamic, the current shape can't represent it.
- **A9 — Layout is a single-dimensional concept.** RowMajor/ColumnMajor
  is a 2D concept; generalizing to N-D is non-trivial (there are N!
  possible axis orderings for N-D, not 2).
- **A10 — Layout affects codegen but not semantics.** Actually,
  column-major traversal of a row-major tensor destroys cache behavior
  but produces identical *results*. Codegen must know this.
- **A11 — Alignment is a boolean.** Actually, alignment is a class —
  4-byte, 16-byte, 64-byte, 256-byte — each admitting different
  instruction widths. Aligning inputs wrong crashes (movaps vs movups)
  or slows (split loads).
- **A12 — Every axis carries its own independent hint.** Symbolic
  equality (square matrix) breaks this: two axes are entangled.
  Needs a relational data structure, not a flat Vec.
- **A13 — Grouping and axes are orthogonal.** Actually some Groupings
  ARE axis-indexed. `Prefix` scans along one specific axis;
  `Tiled { m, n }` scans along two. A multi-dim Shape forces us to
  say *which axis* Prefix operates on.
- **A14 — Ragged data doesn't exist.** Sparse matrices, CSR, variable-
  length sequences, text batches, graphs with variable node-count per
  sample — the current DimHint can't represent any of these. Note 1's
  `Layout::Ragged` is a single marker; that's a huge category.
- **A15 — Cache-key hashing is straightforward.** Actually, with
  variable-length `dims: Vec<DimHint>`, the hash must length-prefix
  to prevent collision (a 2D shape with dims `[10, 20]` must not hash
  the same as a 1D shape with dim `[1020]` collapsed).
- **A16 — Compatibility-sharing check is pairwise on fields.** With
  SymbolicEqual, compatibility becomes a constraint-system question:
  can I unify the symbolic axis references of producer and consumer?

Sixteen assumptions. Four of them (A1, A2, A4, A14) are already
implicit tech debt in the current single-dim shape — Tekgy's note
makes them visible.

---

## Phase 2 — Irreducible Truths

- **T1 — A tensor is an ordered sequence of axes, each with a size
  discipline and an optional symbolic-equality constraint.**
- **T2 — Rank is the LENGTH of that sequence.** Not a separate field;
  derived from `dims.len()`. A shape with `dims.len() == 2` IS a
  matrix shape.
- **T3 — Axes are position-indexed.** "Axis 0" and "axis 1" are
  distinct, named by their position in the sequence. Consumers refer
  to axes by index (`axis: usize`).
- **T4 — Memory layout is a permutation of axes plus stride info.**
  RowMajor = identity permutation. ColumnMajor = reversed permutation
  for 2D, general transposition for N-D. Strided = arbitrary
  (possibly non-contiguous) byte offsets per axis. Ragged = per-row
  (or per-outer-axis) axis sizes differ.
- **T5 — Alignment is an input-buffer property, not an axis-size
  property.** Aligned16 means the base pointer is 16-byte-aligned;
  each axis's stride may or may not preserve that alignment down the
  axis. Alignment is one-per-buffer (or one-per-shape if all buffers
  of this shape align the same).
- **T6 — Some axis relationships are structural.** Square matrix
  (axis[0] == axis[1] for all N), batch×seq×feature (axis[2] fixed,
  axis[0,1] dynamic, axis[0] is batch dim). SymbolicEqual captures the
  simplest case; the full language is unification.
- **T7 — Grouping operates on axes, not on "the data."** Grouping::Prefix
  must specify WHICH axis it scans along. Grouping::All reduces ALL
  axes (or a specified subset). Grouping::Tiled { m, n } operates on
  axes indexed `[axis_m, axis_n]`. **Grouping becomes axis-aware.**
- **T8 — Cache-bake principle applied per axis.** Each axis independently
  has a `DimHint` that's either baked (Static, Bounded, SymbolicEqual)
  or passed (Dynamic). Different axes in the same shape can be
  different classes.
- **T9 — Rank itself is a baked axis.** Changing rank = changing
  the loop nest depth = different instruction stream. `dims.len()` is
  part of the shape fingerprint.
- **T10 — Ragged is NOT just "variable size" — it's a different IR.**
  Dense-with-variable-length (CSR-style) needs per-row offsets as an
  extra buffer input. A `Layout::Ragged` flag on Shape is insufficient;
  we need to surface the offsets buffer.
- **T11 — Alignment specialization produces one kernel per alignment
  class for the same (rank, dtype, Op, grouping).** Not every kernel
  needs this; only SIMD-hot paths.
- **T12 — SymbolicEqual is an EQUIVALENCE RELATION on axis indices,
  not a pairwise constraint.** "Axes 0, 2, 3 are all equal" must be
  representable. Disjoint-set / union-find structure, not a single
  `axis_ref` field.

Twelve truths. Some of them tighten Note 1's draft — e.g. T12 says
`SymbolicEqual(axis_ref)` as a single-axis-reference is less
expressive than an equivalence-class.

---

## Phase 3 — Reconstruction (five shapes on the simple→structural gradient)

### R1 — Literal Note 1 (simplest; minimum change)

```rust
pub struct Shape {
    pub dtype: ScalarTy,
    pub dims: Vec<DimHint>,
    pub grouping: Grouping,
    pub validity: Validity,
    pub tags: Vec<AssumptionTag>,
    pub has_known_non_finite: bool,
    pub layout: Layout,
    pub alignment: AlignmentClass,
}

pub enum DimHint {
    Dynamic,
    Static(usize),
    Bounded { multiple_of: usize },
    SymbolicEqual(usize),  // axis index this one equals
}

pub enum Layout { RowMajor, ColumnMajor, Strided(Vec<isize>), Ragged }
pub enum AlignmentClass { Unaligned, Aligned16, Aligned64, Aligned256 }
```

Evaluated: fails T7 (Grouping isn't axis-aware), T9 (no explicit
rank — inferred), T10 (Ragged is insufficient), T12 (SymbolicEqual
is pairwise, not equivalence-class). Minimum change, maximum
deferred work.

### R2 — R1 + axis-aware Grouping

```rust
pub enum Grouping {
    All { axes: Option<Vec<usize>> },  // None = reduce all
    ByKey { axis: usize, keys: Vec<i64> },
    Prefix { axis: usize },
    Segmented { axis: usize, bounds: Vec<usize> },
    Windowed { axis: usize, size: usize },
    Tiled { axis_m: usize, axis_n: usize },
    Graph { adjacency: Vec<Vec<usize>> },  // graph is inherently 1D
    Probabilistic { weights: Vec<Vec<f64>> },
}
```

Fixes T7. But this is a **breaking change** to `Grouping` — every
existing recipe today passes `Grouping::All` / `Grouping::Prefix`
without axis indices. We'd need to migrate. Under "no backward
compat" (DEC-no-backward-compat), that's fine — just costs ~45 recipe
edits to add axis indices (default `0`). Pathmaker can do it in a
single sweep-8.5 pass.

### R3 — R2 + per-axis alignment

Instead of a single `alignment: AlignmentClass`, make alignment
per-axis:

```rust
pub struct AxisLayout {
    pub dim: DimHint,
    pub alignment: AlignmentClass,
    pub stride_hint: StrideHint,  // Contiguous, Custom(isize), ...
}
pub struct Shape {
    pub dtype: ScalarTy,
    pub axes: Vec<AxisLayout>,  // rank is axes.len()
    // ...
}
```

Overcomplicates for the wrong axis. Alignment is a property of the
*base buffer pointer* + *row stride*, not of every axis. Column
alignment in a row-major 2D matrix is a function of row length, not
an independent property. **Reject.**

### R4 — R2 + equivalence-class SymbolicEqual

```rust
pub struct Shape {
    pub dtype: ScalarTy,
    pub dims: Vec<DimHint>,
    pub symbolic_groups: Vec<Vec<usize>>,  // equivalence classes of
                                            // axis indices
    pub grouping: Grouping,   // axis-aware per R2
    pub validity: Validity,
    pub layout: Layout,
    pub alignment: AlignmentClass,
    // ... other fields ...
}

pub enum DimHint {
    Dynamic,
    Static(usize),
    Bounded { multiple_of: usize },
    // SymbolicEqual LIFTED out of DimHint; lives in symbolic_groups
}
```

`symbolic_groups` is canonicalized (sorted indices within each group;
groups sorted by smallest index). Fixes T12 (equivalence class,
not pairwise). Removes the weird "SymbolicEqual(3) on axis 2" syntax.

### R5 — R4 + explicit ragged-offsets surfacing

```rust
pub enum Layout {
    RowMajor,
    ColumnMajor,
    Strided(Vec<isize>),  // per-axis byte stride
    Ragged { offsets_buffer_idx: usize },  // which buffer input
                                            // holds the offsets
}
```

Ragged data now declares WHICH input buffer (among the dispatch's
inputs) holds the per-row offsets. Codegen knows to emit a 2-level
loop (outer: row index from offsets buffer; inner: dense scan within
row).

This is the **structurally ambitious** reconstruction. Every axis-
relationship a recipe might encounter has a syntactic home; every
alignment class is one fingerprint slot; ragged data is honestly
typed.

**Winner: R5.** R1 is simpler but fails four truths; R5 pays one extra
struct (AxisLayout? no — Tekgy's original proposal + symbolic_groups +
ragged offsets_buffer_idx) and survives every truth.

---

## Phase 4 — Assumption → Truth Map

| Assumption | Truth that replaces it |
|---|---|
| A1 single axis | T1 ordered sequence of axes |
| A2 rank implied | T9 rank = `axes.len()` explicitly in fingerprint |
| A3 axes equivalent | T3 axes are position-indexed; T7 Grouping names an axis |
| A4 contiguous row-major implicit | T4 Layout is a first-class field; Strided/Ragged are typed |
| A5 alignment is whatever | T5+T11 AlignmentClass in fingerprint |
| A6 axes independent | T6+T12 equivalence classes via symbolic_groups |
| A7 size specialization binary | T8 four DimHint variants; bake/pass decision per axis |
| A8 uniform DimHint kind | T8 each axis independently Static/Bounded/Dynamic |
| A9 layout 1D concept | T4 Layout handles N-D permutations + strides |
| A10 layout affects only perf | T4+T10 layout affects BOTH perf (codegen) and correctness (ragged) |
| A11 alignment is boolean | T11 AlignmentClass (4 variants minimum) |
| A12 axes never related | T6+T12 symbolic_groups as explicit equivalence |
| A13 grouping axis-free | T7 Grouping is axis-aware |
| A14 no ragged | T10 Ragged is a typed Layout with offsets buffer |
| A15 cache-key flat | T9+T11 fingerprint hashes rank, per-axis DimHint, symbolic_groups, layout, alignment all length-prefixed |
| A16 pairwise compat | T12 compatibility is unification of symbolic_groups + subset check on tags + per-axis DimHint compatibility |

All sixteen assumptions resolve cleanly.

---

## Phase 5 — The Aristotelian Move

**MOVE: adopt R5 (Note 1 + axis-aware Grouping + equivalence-class
symbolic_groups + ragged offsets).** Concrete deliverable:

### Shape (revised)

```rust
pub struct Shape {
    pub dtype: ScalarTy,
    pub axes: Vec<DimHint>,
    pub symbolic_groups: Vec<Vec<usize>>,  // canonicalized
    pub grouping: Grouping,                // axis-aware per R2
    pub validity: Validity,
    pub tags: Vec<AssumptionTag>,
    pub has_known_non_finite: bool,
    pub layout: Layout,
    pub alignment: AlignmentClass,
}

pub enum DimHint {
    Dynamic,
    Static(usize),
    Bounded { multiple_of: usize },
    // NOT SymbolicEqual; symbolic-equality lives in symbolic_groups
}

pub enum Layout {
    RowMajor,
    ColumnMajor,
    Strided(Vec<isize>),
    Ragged { offsets_buffer_idx: usize },
}

pub enum AlignmentClass {
    Unaligned,
    Aligned16,
    Aligned64,
    Aligned256,
}
```

### Grouping (migration)

Every variant that scans along a specific axis gains `axis: usize`:
- `All { axes: Option<Vec<usize>> }` — None means reduce all axes
- `Prefix { axis }`
- `Segmented { axis, bounds }`
- `Windowed { axis, size }`
- `ByKey { axis, keys }`
- `Tiled { axis_m, axis_n }`
- `Graph { adjacency }` — still 1D-implicit; `axis` reserved for when
  we have graph-of-tensors
- `Probabilistic { axis, weights }`

### Cache-key fingerprint

```text
shape_fingerprint = BLAKE3(
    dtype,
    len_prefix(axes) + for each axis: variant_tag + params,
    len_prefix(symbolic_groups) + for each group: len_prefix + sorted
        axis indices,
    grouping.tag() + axis-index fields,
    validity.tag(),
    sorted tags,
    has_known_non_finite,
    layout.tag() + layout params (Strided's stride vector, Ragged's
        offsets_buffer_idx),
    alignment.tag(),
)
```

All length-prefixed to prevent concatenation-collision (per T9, T12,
T15).

### Compatibility (is_share_compatible_with)

Two shapes share iff:
- `dtype` matches
- `axes.len()` (rank) matches
- For each axis: producer.axes[i] ≥ consumer.axes[i] (static satisfies
  dynamic; bounded(k) satisfies dynamic; bounded(k) compatible with
  bounded(k); static(k) compatible with static(k))
- `symbolic_groups` equivalence is at-least-as-fine on producer (fewer
  groups OK; more groups means producer is stricter; producer can
  satisfy a looser consumer)
- `grouping` matches exactly (different topologies = different
  intermediates)
- `validity` matches exactly
- `layout` matches exactly OR producer is a strict special case
  (Strided with contiguous strides satisfies RowMajor)
- `alignment` producer ≥ consumer (Aligned64 satisfies Aligned16
  satisfies Unaligned)
- tags: producer tags ⊇ consumer tags (unchanged)

---

## Phase 6/7 — Recursive challenge

- **Q-rec-1.** Is `Layout::Strided(Vec<isize>)` variable-rank? Yes —
  the stride vec length must equal `axes.len()` by invariant. Add a
  validator: `Shape::validate() -> Result<(), ShapeError>`.
- **Q-rec-2.** What's the default Layout for a newly constructed
  Shape? Should `Shape::new(grouping, validity)` default to
  `RowMajor`? Yes — matches current implicit behavior.
- **Q-rec-3.** Does alignment interact with stride? Yes —
  `Aligned64` plus `Strided` means the base pointer is 64-aligned,
  but per-row offsets may not be. This is a feature (codegen emits
  aligned base load + per-row scan with unaligned continues). The
  alignment is a hint to codegen about the base pointer.
- **Q-rec-4.** Does `AssumptionTag::SortedAscending` need to name an
  axis now? YES — "sorted along axis 0" vs "sorted along axis 1" are
  different. Add `SortedAscending { axis: usize }`.
- **Q-rec-5.** Does `AssumptionTag::Centered` need to name an axis?
  "Mean 0 along axis 0" (per-column centered) vs axis 1 (per-row
  centered). YES, add `axis: usize`.
- **Q-rec-6.** `AssumptionTag::UnitNorm` axis? YES, same pattern.
- **Q-rec-7.** `AssumptionTag::NoNonFinite` axis? NO — this is a
  property of the whole buffer, not per-axis.
- **Q-rec-8.** What happens to the existing `tags: Vec<AssumptionTag>`
  canonical sort? Must re-sort under new `(axis, tag)` key. Handled.
- **Q-rec-9.** Can `Layout::Ragged` coexist with `alignment:
  AlignedN`? Yes — the offsets buffer is independent; the data
  buffer's base pointer can still be aligned.
- **Q-rec-10.** Can `symbolic_groups` span axes of different DimHint
  kinds? E.g., axis 0 is `Static(1024)` and axis 1 is `Dynamic` —
  can they be in the same symbolic group? NO — if they're in the
  same group, their runtime values must be equal, so neither can be
  Dynamic if the other is Static. Validator must reject this.
- **Q-rec-11.** Does `Grouping::All { axes }` need to respect
  symbolic equality? E.g., reducing across a square matrix's axes 0
  and 1 is well-defined; reducing across an axis that doesn't exist
  panics. Validator must reject out-of-range axes.

---

## Phase 8 — Forced Rejection

- **What if axes had NO DimHint — just runtime lengths?** Then every
  shape is the same shape and we can't specialize on Static/Bounded/
  SymbolicEqual. Loses the bake-instruction-stream principle.
  **DimHint per axis is essential.** Confirmed.
- **What if Layout were NOT a field — just inferred from strides?**
  Then RowMajor vs Strided-that-looks-contiguous would have different
  fingerprints even when they produce identical codegen. Wasted cache
  entries. **Layout stays a typed field** with discipline: codegen
  normalizes equivalent layouts to the same variant.
- **What if alignment were NOT in the fingerprint?** Then aligned
  and unaligned kernels hash to the same key; the cached kernel
  might crash the aligned consumer with `movaps` on an unaligned
  pointer. **Alignment MUST be in fingerprint.** Confirmed.
- **What if symbolic_groups were part of DimHint (Note 1 draft
  style)?** Then "axis 0 equals axis 2, which equals axis 5" is three
  separate entries and keeping them consistent is a recipe-author
  burden. Equivalence-class as a separate field is the right
  structure. Confirmed.
- **What would it mean if Shape had ONE MORE field we haven't
  named?** Forcing myself to find it. Candidate: **contiguity tier**.
  "Rowwise contiguous" (each row is dense; rows may have gaps)
  vs "fully dense" vs "fully sparse." Does `Layout::Strided(strides)`
  capture this? Partially — by examining the strides a consumer can
  infer rowwise contiguity. But making it explicit in a typed field
  (`contiguity: ContiguityClass`) would help codegen without
  reverse-engineering the strides. **Add to spec as a reserved field,
  populate in Sweep 8.5+** when the first recipe benefits.
- **What if dtype were per-axis?** E.g., f32 along axis 0 and f64
  along axis 1. That's a *struct of arrays* or a *multi-column
  table* — not a tensor. If we need it, it's a different Shape
  entirely (`StructShape` or similar). Tensor Shape stays uniform-
  dtype. Confirmed.
- **What if `symbolic_groups` allowed NON-equality relationships?**
  "axis 0 = 2 * axis 1" or "axis 2 = axis 0 + 1" (transformers'
  attention: K rows = Q rows + context). Adding constraint-system
  expressivity is a rabbit hole. Equality-only for Sweep 8;
  linear-relation extensions (via a `constraints: Vec<AxisConstraint>`
  field) deferred to Sweep 8.5+ when transformer recipes land.
- **What if Grouping needed to name MULTIPLE axes?** E.g., a
  2D prefix scan (dynamic programming on a grid — reachability,
  LCS, edit distance). `Grouping::Prefix2D { axis_outer, axis_inner }`
  or `Grouping::PrefixND { axes: Vec<usize> }`. Defer — the first
  recipe that needs it will force the decision; in the meantime
  `Grouping::Prefix { axis }` covers 1D prefix which is every
  current case.

Phase 8 surfaces TWO additions the R5 draft missed:
- **`contiguity: ContiguityClass`** — reserved field, populated
  Sweep 8.5+
- **axis-named AssumptionTags** — `SortedAscending { axis }`,
  `Centered { axis }`, `UnitNorm { axis }`. Update required in R5.

And defers TWO extensions:
- `Grouping::PrefixND` / multi-axis groupings
- `symbolic_groups` generalized to non-equality constraints

---

## Final shape — R5′

```rust
pub struct Shape {
    pub dtype: ScalarTy,
    pub axes: Vec<DimHint>,
    pub symbolic_groups: Vec<Vec<usize>>,  // canonicalized
    pub grouping: Grouping,                // axis-aware
    pub validity: Validity,
    pub tags: Vec<AssumptionTag>,          // axis-indexed where relevant
    pub has_known_non_finite: bool,
    pub layout: Layout,
    pub alignment: AlignmentClass,
    // Reserved; populated Sweep 8.5+ when first consumer demands it:
    // pub contiguity: Option<ContiguityClass>,
}

pub enum DimHint {
    Dynamic,
    Static(usize),
    Bounded { multiple_of: usize },
}

pub enum Layout {
    RowMajor,
    ColumnMajor,
    Strided(Vec<isize>),
    Ragged { offsets_buffer_idx: usize },
}

pub enum AlignmentClass {
    Unaligned,
    Aligned16,
    Aligned64,
    Aligned256,
}

pub enum AssumptionTag {
    NoNonFinite,
    SortedAscending { axis: usize },
    Centered { axis: usize },
    UnitNorm { axis: usize },
    Custom(String),
}
```

**Breaking changes this introduces** (per no-backward-compat):
1. `Shape::dim: DimHint` → `axes: Vec<DimHint>` (rename + type change)
2. Every `Grouping::*` variant that scans gains `axis: usize`
3. `AssumptionTag::{SortedAscending, Centered, UnitNorm}` gain `axis: usize`
4. Shape constructor `Shape::new(grouping, validity)` defaults to
   `axes: vec![DimHint::Dynamic]` (1D, rank 1) — preserves semantics
   of existing 1D recipes

Migration cost: ~45 recipe files updated to pass `axis: 0` on their
existing `Grouping::Prefix` / `Grouping::All` calls. Mechanical
search-and-replace, plus a few `AssumptionTag::Centered { axis: 0 }`
callsites. Pathmaker can land in a single sub-sweep.

---

## Cache-bake audit on the new Shape (per Tekgy Note 2)

Walking each new field through "bake instruction stream OR pass
operands":

| Field | Bake or pass? | Rationale |
|---|---|---|
| `axes.len()` (rank) | BAKE | different loop nest depth = different code |
| `axes[i]` DimHint variant (Dynamic/Static/Bounded) | BAKE | specialization decision |
| `axes[i]` Static(n) value | BAKE | unrolling and constant folding |
| `axes[i]` Bounded(k) multiple | BAKE | determines loop's inner-iteration count modulo |
| `axes[i]` Dynamic runtime length | PASS | passed via buffer metadata + runtime arg |
| `symbolic_groups` | BAKE | different equivalence-class structure = different kernel optimizations |
| `grouping.axis` index | BAKE | which axis the scan iterates along affects loop order |
| `layout` variant | BAKE | RowMajor vs ColumnMajor vs Strided = different traversal |
| `layout::Strided(strides)` values | BAKE for Static strides; PASS for dynamic | the common case is static |
| `alignment` class | BAKE | aligned loads vs unaligned = different instructions |

All honors Tekgy's cache-bake principle.

---

## Deliverables

1. **`crates/tambear/src/jit/shape.rs`** — replace current Shape with
   R5′ (breaking change; migration is mechanical).
2. **`crates/tambear/src/accumulate.rs`** — `Grouping` variants gain
   axis indices; `AssumptionTag::{SortedAscending, Centered, UnitNorm}`
   gain `axis` field.
3. **`crates/tambear/src/jit/fingerprint.rs`** — update hashing to
   length-prefix the new variable-length fields (axes, symbolic_groups,
   layout::Strided stride vector).
4. **Migration pass** — ~45 recipe files get axis indices on their
   Grouping calls. Script-assisted search-and-replace + typecheck.
5. **Tests** — 15+ new assertions covering:
   - Rank-0 (scalar), rank-1, rank-2, rank-3 shape construction
   - `axes.len() == rank` invariant
   - `symbolic_groups` canonicalization (sorted indices within group,
     groups sorted by smallest index)
   - Different rank → different fingerprint
   - Same rank, different `symbolic_groups` → different fingerprint
   - Producer-stricter satisfies consumer (symbolic_groups finer →
     compatible; alignment higher → compatible)
   - Strided contiguous does NOT equal RowMajor in fingerprint (they
     produce different code even if semantically identical)
   - Ragged with offsets_buffer_idx
   - Axis-indexed AssumptionTag sort-compat
   - Validator catches: out-of-range axis in Grouping, symbolic_group
     mixes Static/Dynamic, rank mismatch in Strided stride vector

---

## Asks

**For pathmaker:**
- **Accept R5′?** If yes, I estimate 2-3 hours of implementation +
  migration + 15 new tests. If you want to defer any field (e.g.,
  `Layout::Ragged`), push back — the breaking change is expensive
  enough that we want to land everything in one pass.
- **Open question: should `Layout::Ragged` land NOW or Sweep 8.5+?**
  Per YAWNI, I lean NOW. Per "the migration pass is mechanical if
  every recipe uses 1D Dynamic anyway," the cost of adding it now is
  just the code + tests, no recipe migration. Recommend landing now.

**For adversarial:**
- **Attack R5′ on matrix×matrix.** Can you construct a (Shape A,
  Shape B, Grouping, Op) tuple where the fingerprint is the same but
  the math is actually different, or vice versa?
- **Attack the symbolic_groups canonicalization.** Give me
  equivalence classes that canonicalize wrong. My rule: within each
  group, sort indices ascending; across groups, sort by smallest
  index. Edge case: singleton groups (single-axis "group") — do they
  live in the list or get elided? My answer: **elided** (a singleton
  is not a relationship). Test this.
- **Attack ragged.** Construct a ragged input where the kernel needs
  MORE than just offsets (e.g., max-row-length as a second runtime
  constant). Does `offsets_buffer_idx` suffice or do we need a full
  `RaggedDescriptor { offsets_idx, max_row_len_idx }`?

---

## Conclusion

Note 1's extension is right, and YAWNI-discipline applies — land it
now. The "obvious" draft had four missing pieces:
1. Grouping must gain axis indices
2. SymbolicEqual lifted to a separate `symbolic_groups: Vec<Vec<usize>>`
3. AssumptionTags gain axis indices
4. Ragged needs `offsets_buffer_idx`, not just a marker

All four surface via Phase 1-8. The spec delta is larger than "four
new fields" but the migration is mechanical (DEC-no-backward-compat
permits the breaking changes). **Land as R5′ per Tekgy's YAWNI guidance.**
