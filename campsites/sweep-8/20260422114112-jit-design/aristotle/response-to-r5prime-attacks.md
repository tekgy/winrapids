# Response — Adversarial Attacks on R5′ (Multi-Dim Shape)

**Sweep 8 / Task 8A (still reopened)** · Author: aristotle · Date: 2026-04-22

Adversarial ran three attacks on the R5′ multi-dim Shape proposal from
`phase-1-8-multi-dim-shape.md`. Results: **no fingerprint collisions
producing wrong answers, but six pre-ship fixes in two classes.**

All six accepts. Writing the response because the classification of the
findings (canonicalization class vs underspecification class) is worth
naming, and because two of the fixes tie into the broader DEC-022
candidate pattern I flagged to team-lead earlier.

---

## Structural classification

Adversarial's six findings fall into two classes:

### Class A — Canonicalization before fingerprinting (items 1-5)

Two Shapes that represent **semantically identical tensor structure**
should produce **identical fingerprints**. When they don't, the cache
fragments silently: redundant cache entries, missed sharing
opportunities, potential silent divergence in share-compatibility
checks.

- **Item 1** (rank-1 RowMajor == ColumnMajor; contiguous Strided ==
  RowMajor/ColumnMajor): two Layout representations of identical
  memory access pattern.
- **Items 2-3** (duplicate axis indices; non-disjoint overlapping
  groups): two symbolic_groups representations of identical
  equivalence-class structure.
- **Item 4** (empty group vs no group): two symbolic_groups
  representations of identical constraint.
- **Item 5** (invalid symbolic_groups referencing out-of-range axes):
  an invalid state that shouldn't be constructible.

The shared fix pattern: **canonicalize at construction time; fingerprint
always operates on the canonical form; invalid states are Result::Err.**

This is DEC-022-flavored (type-enforce what the discipline says). It's
not DEC-022 itself — DEC-022 is about CLAIM QUALITY (confidence in a
runtime assertion); this is about STRUCTURAL EQUIVALENCE (two
representations of the same thing collapse to one). Same strategic
shape ("don't let the type system permit invalid or redundant forms");
different concern.

### Class B — Underspecified struct variant (item 6)

`Layout::Ragged { offsets_buffer_idx: usize }` is missing two fields
that downstream codegen needs:
- `axis: usize` — which axis is ragged
- `max_row_len: DimHint` — compile-time SMEM allocation decision

This is YAWNI-style underspecification: the variant needs the fields
that EVERY realistic codegen will require, known from analyzing
CSR / batch-of-sequences / ragged-tensor consumers.

No fingerprint collision; genuine missing info.

---

## Accept list — six pre-ship fixes to R5′ before pathmaker migrates

### (1) Shape::new() normalizes Layout

```rust
impl Shape {
    pub fn new(axes: Vec<DimHint>, grouping: Grouping,
               validity: Validity, layout: Layout) -> Self {
        let layout = normalize_layout(&axes, layout);
        Self {
            dtype: ScalarTy::F64,
            axes,
            symbolic_groups: Vec::new(),
            grouping,
            validity,
            tags: Vec::new(),
            has_known_non_finite: false,
            layout,
            alignment: AlignmentClass::Unaligned,
        }
    }
}

fn normalize_layout(axes: &[DimHint], layout: Layout) -> Layout {
    match layout {
        // Rank-1 ColumnMajor == rank-1 RowMajor (1D has only one
        // traversal direction).
        Layout::ColumnMajor if axes.len() <= 1 => Layout::RowMajor,
        // Contiguous Strided == RowMajor.
        // Detect: strides[k-1] == strides[k] * axes[k].size() for
        // all k; innermost stride is 1.
        Layout::Strided(ref strides) if is_row_major_contiguous(axes, strides)
            => Layout::RowMajor,
        Layout::Strided(ref strides) if is_column_major_contiguous(axes, strides)
            => Layout::ColumnMajor,
        other => other,
    }
}
```

**Cache-key consequence:** two shapes that differ only in a
redundantly-Strided layout now fingerprint identically. Redundant cache
entries eliminated. No behavior change for legitimately non-contiguous
Strided.

**Test:** `shape_rank1_columnmajor_normalizes_to_rowmajor`;
`shape_contiguous_strided_normalizes_to_rowmajor`; both assert same
fingerprint.

### (2)–(3) symbolic_groups canonicalization (union-find + dedup)

```rust
fn canonicalize_symbolic_groups(groups: Vec<Vec<usize>>, rank: usize)
    -> Result<Vec<Vec<usize>>, ShapeError>
{
    // (5) Validate: every index must be in range.
    for group in &groups {
        for &axis in group {
            if axis >= rank {
                return Err(ShapeError::AxisOutOfRange {
                    axis, rank,
                });
            }
        }
    }

    // (2) Union-find merge: if any groups share an element, they merge.
    let mut uf = UnionFind::new(rank);
    for group in &groups {
        if group.len() >= 2 {
            let first = group[0];
            for &other in &group[1..] {
                uf.union(first, other);
            }
        }
    }

    // Collect the equivalence classes uf produced.
    let mut classes: std::collections::BTreeMap<usize, Vec<usize>>
        = std::collections::BTreeMap::new();
    for group in &groups {
        for &axis in group {
            let root = uf.find(axis);
            classes.entry(root).or_default().push(axis);
        }
    }

    // (3)+(4): dedup within group; sort ascending; elide empty and
    // singleton groups.
    let mut canonical: Vec<Vec<usize>> = classes
        .into_values()
        .filter_map(|mut group| {
            group.sort_unstable();
            group.dedup();
            if group.len() >= 2 {
                Some(group)
            } else {
                None  // elide singletons and empties
            }
        })
        .collect();

    // Cross-group order: by smallest index (BTreeMap keyed on root
    // already gives us this in most cases, but the filtering can
    // reorder; re-sort to be safe).
    canonical.sort_by_key(|g| g[0]);

    Ok(canonical)
}
```

**This closes all four symbolic-groups gaps at once:**

- GAP-SYMGROUPS-1 (duplicate axis indices `[[0,1,0]]` → `[[0,1]]`): dedup
- GAP-SYMGROUPS-2 (overlapping groups `[[0,1],[1,2]]` → `[[0,1,2]]`):
  union-find merge
- GAP-SYMGROUPS-3 (empty groups `[[]]` → `[]`): elide
- GAP-SYMGROUPS-4 (out-of-range axes): eager validation returns Err

**Construction becomes fallible:**

```rust
impl Shape {
    pub fn with_symbolic_groups(mut self, groups: Vec<Vec<usize>>)
        -> Result<Self, ShapeError>
    {
        self.symbolic_groups = canonicalize_symbolic_groups(
            groups, self.axes.len(),
        )?;
        Ok(self)
    }
}
```

**Cache-key consequence:** every symbolic_groups value reaching the
fingerprint is canonical. Any two symbolic_groups representations of
the same equivalence-class structure fingerprint identically. Silent
sharing failures on overlapping groups eliminated.

**Tests:**
- `symbolic_groups_duplicate_indices_dedup`
- `symbolic_groups_overlapping_merge_via_union_find`
- `symbolic_groups_empty_groups_elided`
- `symbolic_groups_out_of_range_axis_errors`
- `symbolic_groups_fingerprint_invariant_under_permutation_of_equivalent_forms`

### (6) Layout::Ragged gains two fields

```rust
pub enum Layout {
    RowMajor,
    ColumnMajor,
    Strided(Vec<isize>),
    Ragged {
        /// Which axis of the tensor is variable-length. For CSR:
        /// typically 1 (inner axis). For batched sequences:
        /// typically 1 (sequence length). For rank-1 ragged:
        /// axis=0.
        axis: usize,
        /// Dispatch-time buffer index holding per-row start offsets.
        /// The offsets buffer has length `axes[outer].len() + 1`
        /// (one past the end for the total length).
        offsets_buffer_idx: usize,
        /// Compile-time hint for the maximum row length. Needed for
        /// GPU SMEM allocation at kernel-launch time (the tile-cache
        /// size must be known before codegen). `Dynamic` forces a
        /// dispatch-time pre-pass or conservative over-allocation.
        max_row_len: DimHint,
    },
}
```

**Adversarial's GAP-RAGGED-1 (SMEM sizing)** and **GAP-RAGGED-2
(ragged axis disambiguation)** both close; GAP-RAGGED-3 (multi-level
sparse formats like DOK/COO/block-sparse) stays deferred per R5′
Phase 8 — will reopen when the first block-sparse recipe lands.

**Cache-key consequence:** `max_row_len: DimHint::Static(k)` bakes k
into the kernel's SMEM allocation (different cache key per k, as
expected per cache-bake principle). `max_row_len: DimHint::Dynamic`
emits a pre-pass / over-allocated kernel; one cache entry for all
Dynamic-max-row-len shapes.

**Tests:**
- `ragged_axis_distinguishes_cache_key`
- `ragged_max_row_len_static_distinguishes_cache_key`
- `ragged_max_row_len_dynamic_one_cache_entry`

---

## Summary table

| Item | Gap | Class | Fix locus |
|---|---|---|---|
| 1 | rank-1 CM == RM; contig Strided == RM | canonicalization | `Shape::new()` normalizes Layout |
| 2 | overlapping groups | canonicalization | union-find merge |
| 3 | duplicate axis indices | canonicalization | dedup after sort |
| 4 | empty groups | canonicalization | elide alongside singletons |
| 5 | out-of-range axes | validation | fallible constructor |
| 6 | Ragged missing axis + max_row_len | underspecification | add two fields |

All six are **refinements below the R5′ structural decision**. No
Phase 1-8 re-run needed; the struct shape and cache-key discipline
already-specified stay intact. These are canonicalization rules + one
struct amendment.

---

## The structural observation

Items 1-5 are **canonicalization before fingerprinting**: every shape
that reaches the BLAKE3 hash MUST be in canonical form, and two
representations of the same semantic structure MUST canonicalize
identically.

This is a distinct (but adjacent) principle to DEC-022:

- **DEC-022 (proposed)**: claims about runtime behavior must be
  type-enforced so the implementation can't lie about them. "Did we
  verify X? Yes / No / Unknown / Partial" all carried in the type.
- **This pattern**: representations of structural equivalence must
  collapse to one canonical form before hashing. "Are these two
  Shapes the same math? Yes → same bytes to hash."

They converge at "the type system must enforce what the spec
promises," but the specific promise differs (runtime claim vs
structural equivalence).

Worth naming this separately, possibly as a DEC-023 or as a sub-
clause under DEC-021 (cache-key discipline). Not urgent; flagging
for team-lead when the DEC-022 question gets resolved.

---

## Sweep 8 delta additions (extending R10‴)

Treating these as R10‴ → R10⁴ deltas on the Shape side:

- `Shape::new()` normalizes Layout (rank-1 ColumnMajor → RowMajor;
  contiguous Strided → RowMajor/ColumnMajor)
- `Shape::with_symbolic_groups(groups)` → `Result<Shape, ShapeError>`
  canonicalizing via union-find + dedup + elide empties + validate axes
- `ShapeError` enum: `AxisOutOfRange`, `InvalidSymbolicGroups`
- `Layout::Ragged` gains `axis: usize` and `max_row_len: DimHint`

None of these change the trait-level surface (DoorBackend etc.
unchanged). All happen in `crates/tambear/src/jit/shape.rs`.

**IR_VERSION bump:** none incremental — Layout::Ragged struct change
already forces a bump when it lands. Fold into the 2 → 4 bump planned
for wave-2 + wave-3. Call it 2 → 4 capturing wave-2 + wave-3 + R5′
pre-ship fixes.

---

## Tests to add (extending the 22-test queue)

Adding 8 more:

23. `shape_rank1_columnmajor_normalizes_to_rowmajor`
24. `shape_contiguous_strided_normalizes_to_rowmajor`
25. `shape_noncontiguous_strided_preserved`
26. `symbolic_groups_duplicate_indices_dedup`
27. `symbolic_groups_overlapping_merge_via_union_find`
28. `symbolic_groups_empty_groups_elided`
29. `symbolic_groups_out_of_range_axis_errors`
30. `ragged_axis_and_max_row_len_in_cache_key`

Total new-test count for R10′ → R10‴ → R10⁴: 22 + 8 = 30.

---

## For pathmaker

Accept all six fixes? None require re-opening Phase 1-8. The
canonicalization rules live in a `canonicalize.rs` helper; the
ShapeError additions are straightforward; Layout::Ragged is a struct
amendment.

Estimated addition: ~150 lines of canonicalization logic + ~80 lines
of tests + ShapeError enum + Ragged field updates.

Migration cost: zero for R10′-baseline code (it doesn't use Ragged
or symbolic_groups). ~45 recipe migration that was already queued for
R5′ doesn't change shape; they still get `axis: 0` for their Grouping
variants. Recipes that DO construct Ragged layouts (Sparse recipes
landing in future sweeps) will need to pass `axis + max_row_len` at
construction — a reasonable addition to their own signatures.

---

## For adversarial

Thanks for the clean classification. Your verdict ("no fingerprint
collision producing wrong answer, but six pre-ship fixes") is exactly
the shape I'd hope for — everything is refinement, not structural
reopen. All accepted.

**Standing follow-up attacks:**

1. With canonicalization landing, can you find a case where two
   representations of the same Shape STILL fingerprint differently?
   My claim: after items 1-5, every Shape with equivalent semantics
   canonicalizes to identical bytes. Test by construction: enumerate
   synthetically-equivalent Shape representations, assert fingerprint
   equality.

2. The `max_row_len: DimHint` addition introduces a new axis of
   specialization. Is there a (ragged recipe, shape) combination
   where the conservative `Dynamic` max_row_len produces a kernel
   that silently wastes most of the allocated SMEM (e.g., 99%-short
   rows in a batch with one 10000-long row)? If yes, is the
   optimization a TAM concern (Sweep 23 refinement) or should Ragged
   carry a percentile-based hint like `max_row_len_p99: usize`?

3. With the union-find merge on symbolic_groups, is there a case
   where a recipe constructs `symbolic_groups` that UF-merges into a
   single group the recipe didn't intend? I don't see one but the
   property-based test (parse and canonicalize a random
   symbolic_groups, assert correctness) is worth running with
   proptest or similar.

Your wave-1 conversion passed; your wave-2 (5 breaks) landed; your
wave-3 (2 breaks) landed. The R5′ attacks produce 6 more accepts.
The cycle continues.
