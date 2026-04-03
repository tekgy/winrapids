# Notebook 005 — Manifold Type Scaffolding

*2026-03-31 | Navigator*

---

## Background

From garden note `manifold-as-jit-parameter-2026-03-31.md`:

> `Metric` is a limited special case of `Manifold`. Metric gives you distance. Manifold gives you four things:
> - Distance (for neighborhood)
> - Mean/centroid (for cluster centers)
> - Gradient scaling (for optimization)
> - Projection (for keeping points on the manifold)

The current `Metric` enum in `intermediates.rs` handles the first. `Manifold` would handle all four — each expressible as a JIT phi_expr string that TiledEngine can compile.

## Hypothesis

A `Manifold` enum can be defined now, as type-level scaffolding, without touching the JIT wiring. Each variant carries its geometric parameters. Each implements `tiled_dist_expr()`, `centroid_update_expr()`, and `gradient_scale_expr()` — CUDA expression strings that future TiledEngine JIT will compile.

The critical constraint: `Manifold` must be `Hash + Eq` so it can serve as an `IntermediateTag` key. `f64` curvature parameters are hashed by bit pattern (reproducible for compile-time constants).

`Metric` is preserved for now — the garden note recommends keeping `Metric` until Poincaré/Sphere/Learned are actually needed. `Manifold` is additive, not a replacement.

---

## Design

```rust
pub enum Manifold {
    Euclidean,
    Poincare { curvature_bits: u64 },  // f64 stored as bits for Hash
    Sphere { radius_bits: u64 },
    Learned { params_id: DataId },      // params are a Buffer identified by content
    Bayesian { prior: Box<Manifold>, posterior_id: DataId },
}
```

The `f64` → `u64` bit-storage trick makes `Manifold: Hash` without floating-point comparison issues. The user-facing API takes `f64` and converts internally.

Each variant implements:
- `tiled_dist_expr(&str a, &str b, usize d)` → CUDA expression for pairwise distance
- `centroid_update_expr()` → CUDA expression for weighted mean
- `gradient_scale_expr(pt_var)` → conformal factor for Riemannian gradient

---

## Results

**12 new manifold tests pass. 166 total (148 unit + 18 doctest), all green.**

| Test | Verifies |
|------|---------|
| `euclidean_hash_eq` | Euclidean is hashable and self-equal |
| `poincare_same_curvature_eq` | Same f64 curvature → same hash |
| `poincare_different_curvature_ne` | Different curvature → different variant |
| `poincare_curvature_roundtrip` | bits→f64 round-trip is exact |
| `sphere_radius_roundtrip` | Same for sphere |
| `euclidean_dist_expr_contains_l2sq` | `tiled_dist_expr()` returns L2Sq form |
| `poincare_gradient_scale_contains_curvature` | Conformal factor expression has `x_norm_sq` |
| `projection_needed` | Euclidean=false, Poincaré=true, Sphere=true |
| `bayesian_hashable` | Recursive `Box<Manifold>` variant hashes correctly |
| `learned_hashable` | DataId-keyed variant works |
| `display_names` | All variants have readable names |
| `euclidean_in_hashmap` | `Manifold` works as `HashMap` key |

---

## Discussion

The type scaffolding is solid. `Manifold` is `Hash + Eq` and can serve as an `IntermediateTag` key when manifold-aware algorithms need their distance matrices cached separately by geometry.

**The f64 bit-storage trick** — storing `curvature_bits: u64` instead of `curvature: f64` — deserves a note: it makes `Manifold: Hash` without the `Eq` problems that floating-point comparison creates. Two Poincaré manifolds created with the same literal `f64::to_bits(-1.0)` will be considered equal, which is correct — curvature is a compile-time constant, not a computed value. If somehow floating-point arithmetic produces `Poincare { curvature_bits: X }` and `Poincare { curvature_bits: X+1 }` for what should be the same geometry, that's a bug in the caller, not a design flaw.

**JIT expressions are ready**: `tiled_dist_expr()`, `gradient_scale_expr()`, `centroid_update_expr()` return strings that will plug directly into TiledEngine kernel generation. The `Euclidean` case matches what TiledEngine currently hardcodes — confirming the design is consistent. `Poincaré` and `Sphere` cases are either placeholders or working expressions (gradient scale for Poincaré is the exact conformal factor formula).

**What's missing before the manifold dimension is live:**
1. TiledEngine needs to accept a `phi_expr` for its inner loop (currently hardcoded). The JIT infrastructure from `ScatterJit` is the pattern.
2. `KMeans::fit()` needs to accept a `Manifold` parameter and use `centroid_update_expr()` for the centroid step.
3. `IntermediateTag::DistanceMatrix` should accept `Manifold` instead of just `Metric` — or alongside it.

These are one-layer extensions. The type is ready; the wiring comes when needed.

