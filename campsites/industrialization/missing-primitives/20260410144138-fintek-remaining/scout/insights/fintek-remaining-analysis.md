# Fintek Remaining — Scout Analysis
Written: 2026-04-10

## The Core Finding: These Primitives Already Exist

The campsite note said: "6 Tier 2 leaves need new tambear primitives: ccm, mfdfa,
phase_transition, harmonic r-stat."

After the phantom scan: ALL FIVE exist in `tambear::complexity`. They are not missing.
They are unreachable — missing from `lib.rs` pub use AND have API shape mismatches with
the fintek bridges.

| Primitive | Tambear location | Status |
|-----------|------------------|--------|
| `ccm` | `complexity.rs:1582` | Exists, NOT in lib.rs pub use |
| `mfdfa` | `complexity.rs:1427` | Exists, NOT in lib.rs pub use |
| `phase_transition` | `complexity.rs:1673` | Exists, NOT in lib.rs pub use |
| `harmonic_r_stat` | `complexity.rs:1752` | Exists, NOT in lib.rs pub use |
| `hankel_r_stat` | `complexity.rs:1778` | Exists, NOT in lib.rs pub use |
| `CcmResult` | `complexity.rs` | Exists, NOT in lib.rs pub use |
| `MfdfaResult` | `complexity.rs` | Exists, NOT in lib.rs pub use |
| `PhaseTransitionResult` | `complexity.rs` | Exists, NOT in lib.rs pub use |

## Why Fintek Reimplemented Them Anyway

Two problems blocked the fintek bridges from calling the tambear primitives:

**Problem 1: API shape mismatch.**

`tambear::complexity::ccm` signature:
```rust
pub fn ccm(x: &[f64], y: &[f64], embed_dim: usize, tau: usize, k: usize) -> CcmResult
```
`tambear::complexity::CcmResult` fields: `rho_xy, rho_yx, rho_xy_half, rho_yx_half, convergence`

`family24_manifold::ccm` signature:
```rust
pub fn ccm(x: &[f64], y: &[f64]) -> CcmResult  // uses hardcoded E=3, τ=1, k=E+1
```
`family24_manifold::CcmResult` fields: `ccm_xy_full, ccm_yx_full, ccm_xy_half, ccm_yx_half, convergence_ratio`

Same math, same output meaning — different field names, different parameter defaults.

`tambear::complexity::mfdfa` signature:
```rust
pub fn mfdfa(data: &[f64], q_values: &[f64], min_seg: usize, max_seg: usize) -> MfdfaResult
```
`family22_criticality::mfdfa` signature:
```rust
pub fn mfdfa(returns: &[f64]) -> MfdfaResult  // hardcoded q=[-5,-2,-1,0,1,2,5], segments
```

**Problem 2: The missing pub use.**

Even if the shapes matched, `use tambear::ccm` would fail because `lib.rs` doesn't
re-export these symbols. The fintek crate imports directly from `tambear::complexity`
only for what it knows is there.

## The Fix Architecture (Two Paths)

### Path A: Thin bridge adapters (lower risk)

Keep family22/24 as-is. Add the missing `pub use` to lib.rs so callers CAN use them
if they want. Document the API difference between the tambear and fintek versions.

```rust
// lib.rs additions:
pub use complexity::{ccm, CcmResult, mfdfa, MfdfaResult, phase_transition,
    PhaseTransitionResult, harmonic_r_stat, hankel_r_stat, rqa, RqaResult};
```

Result: tambear exposes the full-parameter versions. Fintek keeps the no-parameter
convenience wrappers. Both exist, neither is wrong.

### Path B: Delegate bridges to primitives (cleaner, preferred)

Make family22/24 call tambear::complexity and just extract what they need:

```rust
// family24_manifold::ccm — after pub use is fixed:
pub fn ccm(x: &[f64], y: &[f64]) -> CcmResult {
    let result = tambear::complexity::ccm(x, y, 3, 1, 4); // E=3, τ=1, k=E+1
    CcmResult {
        ccm_xy_full: result.rho_xy,
        ccm_yx_full: result.rho_yx,
        ccm_xy_half: result.rho_xy_half,
        ccm_yx_half: result.rho_yx_half,
        convergence_ratio: result.convergence,
    }
}
```

This makes fintek a thin wrapper: "call tambear, extract fintek columns."
The math lives in one place (tambear). The fintek API stays unchanged.

For mfdfa, same pattern — call tambear with the fintek hardcoded defaults, extract output.
For phase_transition — same.

## What Actually Needs to Happen (Ordered)

**Step 1 (5 minutes, pathmaker):** Add to `lib.rs` pub use block:
```rust
pub use complexity::{
    ccm, CcmResult,
    mfdfa, MfdfaResult,
    phase_transition, PhaseTransitionResult,
    harmonic_r_stat, hankel_r_stat,
    rqa, RqaResult,
};
```

**Step 2 (optional, pathmaker):** Refactor family22/24 to delegate to tambear::complexity
instead of reimplementing. These files currently have:
- `family22_criticality.rs`: full `mfdfa` + `phase_transition` reimplementations (~200 lines)
- `family24_manifold.rs`: full `ccm` + `harmonic` implementations using `tambear::linear_algebra::svd`

The `harmonic` function in family24 is genuinely a fintek leaf (tick analysis → SVD for harmonic
structure), not a pure tambear primitive. Its SVD usage IS correct delegation. So it's
a partial bridge — it uses tambear math but contains fintek-specific framing.

## The Tick Leaves

The creation note also mentioned "~15 tick_* custom variants." Looking at the current
family structure: families 1-24 are on raw returns/prices. The "tick_*" naming refers to
a proposed future extension where bins contain raw tick sequences (not just returns).

These don't exist yet as code. They're a future scope item, not a missing primitive.
Nothing to implement — just confirm the scope is tracked for when tick-level data
feeds the fintek pipeline.

## Summary for Navigator

The "fintek remaining" problem is NOT about missing math. The math is implemented.
The problems are:
1. Missing pub use in lib.rs (5-minute fix, unblocks any external caller)
2. API shape mismatch between tambear and fintek versions (architectural, lower urgency)
3. The tick_* leaves are future scope, not current debt

Recommend: treat this as a pub-use gap fix (Step 1), and delegate the bridge refactor
to the pathmaker as a separate task with clear API matching spec.
