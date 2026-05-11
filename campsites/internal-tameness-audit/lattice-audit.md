# Lattice surface — internal-tameness audit (Phase B, part 2)

Auditor: adversarial  
Date: 2026-05-10  
Source: `R:\tambear\crates\tambear\src\lattice\precision.rs`

The lattice is a typing and routing module. It does not perform floating-point arithmetic. The tameness predicates here are:
- **Dispatch correctness**: `PrecisionContext::dispatch_level()` always routes to the correct tier.
- **Monotone-path enforcement**: `is_monotonically_coarsening` / `try_from_steps` rejects non-monotone paths.
- **Tag stability**: `PrecisionLevel::tag()`, `RoundingMode::tag()` — tags must not change after shipping.

---

## `PrecisionContext::dispatch_level` and `dispatched_precision_bits`

**Finding L-1 (LINT-3 — dispatch table duplication):** `dispatch_level()` and `dispatched_precision_bits()` contain the same `match self.requested_precision_bits` table duplicated. The comment at line ~233 explicitly acknowledges the duplication: "Equivalent to `self.dispatch_level().native_precision_bits()` but `match` on a non-Copy method-result isn't const."

The tameness predicate: both tables must agree. The proptest at "§3.4 gauntlet-Surface-4" is supposed to pin the equivalence, but that test is in the proptest gauntlet (a separate file), not in this module. If someone edits one table without the other, the tests may still pass (because the gauntlet covers a sampling, not exhaustive coverage of all u32 inputs).

**Finding L-1 (MEDIUM):** The two dispatch tables are logically identical but physically separate. The invariant "they agree for all inputs" is tested by sampling, not by construction. A correctness-by-construction fix: make `dispatched_precision_bits` call `self.dispatch_level().native_precision_bits()` and make the function non-const (accepting the const limitation). Alternatively: add a `const fn` that computes the tier boundary table once and has both methods delegate to it.

**Tag:** LINT-3 (special-value dispatch consistency — duplicate dispatch tables that must stay in sync).

---

## `is_monotonically_coarsening` and `first_non_monotone_index`

**Finding L-2 (INFO — correctness):** `is_monotonically_coarsening` uses `windows(2)` which short-circuits on a length-0 or length-1 slice (returns `true`). The docstring says "A length-0 or length-1 path is trivially monotone." This matches. But there's a subtle tameness predicate: a zero-length path is trivially monotone AND trivially not monotone. The function returns `true` for empty paths; `is_monotonically_refining` also returns `true` for empty paths. Both can be simultaneously true. Any consumer that does `if is_monotone && !is_refining` to detect "purely coarsening" would silently misclassify an empty path. The empty path case should be handled by `PathError::Empty` at the higher level — but if that check is absent in a consumer, the silent misclassification can occur.

**Tag:** F13.A antibody gap — the empty-path invariant is documented but not structurally enforced at the `is_monotonically_coarsening` level. Rely on callers to check for empty first.

---

## `UlpBudget`

**`max_ulps: u64::MAX` sentinel for "no proven bound":** This is a convention documented in the docstring but not structurally enforced. A consumer that does `budget.max_ulps < 10` would accept the "no proven bound" budget as satisfying a 10-ULP requirement. 

**Finding L-3 (F13.A):** The `u64::MAX` sentinel is not a type-enforced "unbounded" marker — it's a convention. A consumer comparing `max_ulps` numerically can accidentally treat "unbounded" as "within budget." A `UlpBoundedness` enum (`Bounded(u64)` / `Unbounded`) would make the distinction structural. Alternatively, a helper `fn is_bounded(&self) -> bool` and `fn bounded_max_ulps(&self) -> Option<u64>` would surface the distinction at consumer callsites.

**Tag:** F13.A (constructor antibody). `UlpBudget::unbounded()` accepts an `at_destination` but the `max_ulps = u64::MAX` convention is invisible at construction.

---

## `RoundingMode::Default` implementation

`impl Default for RoundingMode` returns `RoundToNearestTiesEven`. This is a potentially hazardous `Default` implementation in the JBD context. The F13.C discipline (BranchPolicy must NOT implement Default) was established because a defaulted policy can silently degrade to a wrong behavior when code forgets to specify it.

**Finding L-4 (LINT-3 / F13.C analogue):** `RoundingMode` implements `Default`. Any code that derives or relies on `Default` for a struct containing `RoundingMode` will silently use RNE. This is usually the correct choice, but in one specific context it's dangerous: internal Newton iterations in `normal_div_multilimb` and `normal_sqrt_multilimb` hardcode `RoundingMode::RoundToNearestTiesEven` explicitly (good). But a future engineer who adds a new internal iteration and forgets to pass the rounding mode, using `..Default::default()` instead, would silently run RNE instead of the user-requested mode.

**Severity:** LOW (defensive concern; currently the hardcoded RNE is correct for internal Newton steps). Not a current bug, but the `Default` impl creates the surface.

---

## `PrecisionLevel::tag()` and `RoundingMode::tag()`

**Finding L-5 (tag-stability invariant):** Tags are const fns returning u8. The docstrings say "must not change once shipped." There is no `const_assert!` or test pinning the specific tag values. If a future enum variant is added before an existing variant (changing variant ordinals), and a developer naively uses the ordinal for the tag, they'd silently reassign existing tags. The current implementation uses explicit match arms (not `as u8`) which is safe — but there's no failing test that would catch "someone accidentally changed a tag value."

**Tag:** Same pattern as J-2 (no `assert_all_tags_distinct`/`assert_tag_values_pinned` test). Add tests:

```rust
#[test]
fn precision_level_tags_pinned() {
    assert_eq!(PrecisionLevel::P0F64.tag(), 0);
    assert_eq!(PrecisionLevel::P1DoubleDouble.tag(), 1);
    assert_eq!(PrecisionLevel::P2BigFloat { precision_bits: 200 }.tag(), 2);
}
#[test]
fn rounding_mode_tags_pinned() {
    assert_eq!(RoundingMode::RoundToNearestTiesEven.tag(), 0);
    assert_eq!(RoundingMode::RoundToNearestTiesAwayFromZero.tag(), 1);
    assert_eq!(RoundingMode::RoundTowardZero.tag(), 2);
    assert_eq!(RoundingMode::RoundTowardPositiveInfinity.tag(), 3);
    assert_eq!(RoundingMode::RoundTowardNegativeInfinity.tag(), 4);
}
```

**Severity:** MEDIUM (silent cache-key invalidation if tags change without IR_VERSION bump).

---

## Summary: lattice tameness sites

| ID | Location | Lint | Severity | Status |
|---|---|---|---|---|
| **L-1** | `PrecisionContext::dispatched_precision_bits` | LINT-3 | MEDIUM | Duplicate dispatch table, sync tested by sampling not construction |
| L-2 | `is_monotonically_coarsening` empty-path | F13.A | LOW | Empty-path both monotone and refining; consumer must guard |
| L-3 | `UlpBudget::unbounded` | F13.A | LOW | u64::MAX sentinel not type-enforced |
| L-4 | `RoundingMode::Default` | LINT-3/F13.C analogue | LOW | Default impl creates future silent-degradation surface |
| **L-5** | `PrecisionLevel::tag`, `RoundingMode::tag` | LINT-3 | MEDIUM | Tag values not pinned by failing tests |

**Post-audit correction (verified after writing):**

L-5 is already covered: `lattice/tests.rs` has `precision_level_tag_is_stable` (lines 28-39) and `rounding_mode_tag_is_stable` (lines 43-50) that pin exact tag values.

L-1 is already covered: `lattice/tests.rs` has seven `dispatch_at_*` tests that assert both `dispatch_level()` and `dispatched_precision_bits()` for the same context at every boundary value (0, 53, 54, 106, 107, 200, 1024). The two-table sync is tested per-boundary, not by construction, but the coverage is complete for the boundary cells that matter.

The only genuinely open items from the lattice audit are L-3 (UlpBudget::unbounded u64::MAX sentinel not type-enforced) and L-4 (RoundingMode implements Default, creating future silent-degradation surface). Both are advisory, not active bugs.

