# Amendment — Canonical Entry-Point Names

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

Pathmaker's constraint #3 ("identity + lift + combine + extract are 4
separate compiled kernels — or 1 fused one") sharpens the abstract
`EntryPoint` design in `trait-spec-locked.md`. The spec says compile
returns `entry_points: SmallVec<[EntryPoint; 4]>` with stable names —
this amendment **fixes the canonical name set** so the dispatcher can
look entry points up by name without per-Op convention drift.

This is a refinement, not a redesign. The spec's `EntryPoint { name:
&'static str, workgroup_constraint }` shape is unchanged.

---

## The canonical entry-point name set

Every `CompiledArtifact` produced by `DoorCodegen::lower(op, shape,
strategy, params, cap)` may include any subset of these named entry
points. The dispatcher selects per shape; missing entry points mean
"this strategy doesn't need this kernel."

### Reduce-family entry points (Grouping::All / ByKey / Tiled)

```text
"reduce_all"          // single kernel: lift+combine+extract fused
                      // for small N where one workgroup suffices.
                      // CPU: a straight loop; GPU: one workgroup.

"reduce_warp"         // per-warp partial reduction (lift+combine)
                      // emitting one State per warp. GPU only;
                      // CPU degenerates to a SIMD-lane reduction.

"reduce_block"        // per-workgroup tree reduction over warp
                      // partials, emitting one State per workgroup
                      // to scratch. GPU only; CPU = single thread.

"reduce_device"       // device-wide tree reduction over workgroup
                      // partials; emits one final State to output[0].
                      // Last stage of large-N reduction.

"extract_scalar"      // pure extract: State -> f64. Cheap; runs
                      // after the reduce chain finishes.
```

### Scan-family entry points (Grouping::Prefix / Segmented / Windowed)

```text
"scan_inclusive_seq"  // sequential prefix scan — fallback codegen,
                      // emitted when ExecutionStrategy::Sequential
                      // is selected.

"scan_inclusive_warp" // per-warp Hillis-Steele inclusive scan
                      // (lift+combine within warp lanes).

"scan_inclusive_block" // per-workgroup Brent-Kung scan over warp
                      // partials, with a per-workgroup carry-out.

"scan_carry_apply"    // device-wide propagation of inter-workgroup
                      // carries (the "tail" of a 2-pass parallel
                      // scan; CPU collapses this to a single fixup
                      // pass).

"extract_per_position" // emits the running State at each input
                       // position. For Prefix/Segmented; pre-applies
                       // extract_scalar() per-position.
```

### Conjugation entry points (LiftedConjugated only)

```text
"permute_in"          // applies the conjugation permutation
                      // (sort/reverse/space-fill/custom) to the
                      // input prior to scan.

"permute_out"         // applies the inverse permutation to the
                      // output after the lifted scan completes.
```

### Special / future

```text
"gather_main"         // the gather atom's primary kernel (separate
                      // family from accumulate).

"profile_probe"       // micro-benchmark trampoline used by
                      // DoorCache::record_timing to measure kernel
                      // entry-point timing without polluting real
                      // dispatches.
```

### Discipline

- **Names are `&'static str`**, lowercased, snake_case, prefixed by
  family (`reduce_*`, `scan_*`, `permute_*`, `gather_*`, `extract_*`,
  `profile_*`).
- **Names NEVER change once shipped.** Renaming silently invalidates
  every user's persistent kernel cache. New variants get new names.
- **A backend MAY emit a SUBSET** of the family — emitting only
  `reduce_all` for a small-N specialization, only `scan_inclusive_seq`
  for sequential strategy. The dispatcher's `select_entry()` reads
  the artifact's entry-point list and picks the most specific one
  applicable to the shape.

---

## How the dispatcher uses the entry-point list

```rust
fn pick_entry(art: &CompiledArtifact, shape: &Shape, strategy: ExecutionStrategy)
    -> EntryPoint
{
    match strategy {
        ExecutionStrategy::Lifted | ExecutionStrategy::LiftedConjugated { .. } => {
            match shape.grouping {
                Grouping::All | Grouping::ByKey { .. } => {
                    // Pick reduce_all for small N (one-workgroup),
                    // reduce_device for large N (multi-stage).
                    if shape.fits_one_workgroup(cap) {
                        find_entry(art, "reduce_all")
                    } else {
                        find_entry(art, "reduce_device")
                    }
                }
                Grouping::Prefix | Grouping::Segmented { .. }
                | Grouping::Windowed { .. } => {
                    if shape.fits_one_workgroup(cap) {
                        find_entry(art, "scan_inclusive_warp")
                            .or_else(|| find_entry(art, "scan_inclusive_block"))
                    } else {
                        find_entry(art, "scan_carry_apply")
                    }
                }
                Grouping::Tiled { .. } => find_entry(art, "reduce_device"),
                Grouping::Graph { .. } | Grouping::Probabilistic { .. } => {
                    // These groupings dispatch to specialized entry
                    // points emitted per-Op.
                    find_first_entry(art)
                }
            }
        }
        ExecutionStrategy::Sequential { .. } => {
            match shape.grouping {
                Grouping::All | Grouping::ByKey { .. } => find_entry(art, "reduce_all"),
                _ => find_entry(art, "scan_inclusive_seq"),
            }
        }
    }
}
```

The dispatcher's choice is part of the cache key only via `strategy`;
entry-point selection within a cached artifact is a runtime decision
based on shape's per-call `dim` (which is NOT in the cache key when
`DimHint::Dynamic`).

---

## Why this constraint matters

Without canonical names:
- Each Op's codegen invents its own naming convention.
- The dispatcher needs Op-specific knowledge to find the right
  entry point.
- Adding a new Op requires updating the dispatcher to recognize its
  entry-point names.
- TamSession sharing breaks: two recipes producing the "same"
  intermediate but using different entry-point names can't share.

With canonical names:
- The dispatcher is Op-agnostic — it picks by `(grouping, strategy,
  shape-fits-one-workgroup)` against a fixed name table.
- Adding a new Op is pure codegen work — the dispatcher needs no
  changes if the Op emits the canonical names.
- Profile-feedback (`record_timing`) keys on `(name, shape)` and
  can compare timings across Ops that share an entry-point family.

---

## Spec delta

In `trait-spec-locked.md`, the line:

```rust
pub struct EntryPoint {
    pub name: &'static str,
    pub workgroup_constraint: Option<WorkgroupShape>,
}
```

becomes:

```rust
pub struct EntryPoint {
    /// One of the canonical names from
    /// `docs/expedition/canonical-entry-points.md`. Names are stable
    /// across tambear versions; renaming invalidates the persistent
    /// kernel cache.
    pub name: &'static str,
    pub workgroup_constraint: Option<WorkgroupShape>,
}
```

Plus a `pub mod entry_point_names` (or const block) in `door.rs`
exporting every canonical name as `pub const REDUCE_ALL: &str =
"reduce_all";` etc., so codegen and dispatcher refer to constants
rather than literal strings.

---

## Open question

Do we want **string names** or a **canonical enum** like:

```rust
pub enum EntryPointKind {
    ReduceAll,
    ReduceWarp,
    ReduceBlock,
    ReduceDevice,
    ScanInclusiveSeq,
    ScanInclusiveWarp,
    ScanInclusiveBlock,
    ScanCarryApply,
    ExtractScalar,
    ExtractPerPosition,
    PermuteIn,
    PermuteOut,
    GatherMain,
    ProfileProbe,
    /// Op-specialized entry not in the canonical family. Keyed by
    /// stable string for bespoke kernels.
    Specialized(&'static str),
}
```

**Recommendation: enum.** Type-checked at codegen time; `Specialized`
escape hatch covers the unusual cases. Pathmaker chooses; either
shape works at the trait level.

If pathmaker picks the enum, `EntryPoint` becomes:

```rust
pub struct EntryPoint {
    pub kind: EntryPointKind,
    pub workgroup_constraint: Option<WorkgroupShape>,
}
```

and `find_entry` becomes a typed match.
