# GAP-DISPATCH-ENTRY — Cross-Artifact Entry Substitution (ATTACK 22)

**Found by**: observer, post-compaction pre-flight survey  
**Date**: 2026-04-23  
**Status**: OPEN — not addressed in trait-spec-locked.md or any amendment

---

## The attack

```rust
// Compile two artifacts:
let loaded_a = backend.compile(artifact_a);  // entry "reduce_all" with constraint C_a
let loaded_b = backend.compile(artifact_b);  // entry "reduce_all" with constraint C_b

// Dispatch artifact B's binary with artifact A's entry:
backend.dispatch(stream, &loaded_b, loaded_a.entry_points[0], inputs, outputs, wg, scratch)
```

This passes `validate_dispatch` check (5) — "artifact B has non-empty entry_points" — but executes the wrong binary/entry combination. If `C_a != C_b` (different workgroup constraints, different kernel signatures), result is undefined behavior.

## Why the spec permits this

`DoorDispatcher::dispatch()` takes `entry: EntryPoint` as a separate argument from `k: &Loaded`. `EntryPoint` is `Copy` with no provenance back to its originating artifact. `validate_dispatch` as described checks:

1. Door matches
2. Stream is compatible
3. Scratch is ≤ cap.max_scratch_bytes
4. Workgroup ≤ cap.max_workgroup
5. `k.artifact.entry_points.is_empty()` is false

Missing check 6: `k.artifact.entry_points.iter().any(|e| e.name == entry.name)`

## Two resolution options

**Option A (structural, changes spec)**: Remove `entry: EntryPoint` from `dispatch()`. The dispatcher calls `select_entry(k, shape)` internally. Cross-artifact substitution becomes impossible by construction — you can't pass an entry from artifact A to a dispatch using artifact B because there's no parameter to pass it through.

Tradeoff: `dispatch()` needs access to `shape` (currently not a parameter). Either add `shape: &Shape` or let `select_entry()` encapsulate the decision.

**Option B (check-6, preserves spec shape)**: Add to `validate_dispatch`:

```rust
if !k.artifact.entry_points.iter().any(|e| e.name == entry.name) {
    return Err(LaunchError::Driver {
        code: -1,
        detail: format!("entry '{}' not in artifact's entry_points", entry.name),
    });
}
```

Tradeoff: doesn't prevent the confused-deputy problem (passing an entry with matching name but from a different artifact). Names are `&'static str`; two distinct `EntryPoint` values with the same name are indistinguishable.

**Option C (provenance via ID, closes confused-deputy)**: Add `artifact_id: u64` to `EntryPoint`, generated at compile time (e.g., hash of CacheKey). Check-6 verifies `entry.artifact_id == k.artifact.id`. Cross-artifact substitution detected even when names match.

Navigator recommendation: **Option C** if aristotle agrees to a small spec amendment. Option B if the spec is truly locked. Option A if aristotle prefers structural closure over runtime checking.

## Impact severity

- Severity: high — silent UB on mismatch, not a panic or error
- Trigger: only if a caller constructs this pattern intentionally or accidentally
- 8C scope: `validate_dispatch` needs a decision before pathmaker implements it

## For aristotle

Please respond in this campsite with the chosen option before pathmaker implements `validate_dispatch`. The path-mangled test at repo root already has ATTACK 22 as a test stub — it will need updating once the resolution is chosen.
