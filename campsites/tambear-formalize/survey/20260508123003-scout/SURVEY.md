# DEC-029 Survey — tambear-formalize scout

**Date**: 2026-05-08
**Campsite**: `dec-029-impl`
**Scout**: tambear-formalize session scout

---

## What is DEC-029?

DEC-029 is the "Knowledge-adapter" decision, ratified in the `R:\tambear\docs\decisions.md`
at lines 2951-3148. It specifies the evidence-convergence mechanism for the TamSession
intermediate-sharing system.

Core concept: when multiple computations produce results that share assumptions, the
knowledge system tracks *provenance* and *compatibility* so that intermediates can be
reused only when the upstream computation is provably correct for the downstream consumer.

### Key types

- `Knowledge` trait: three methods — `can_share`, `can_specialize`, `can_elide_check` —
  each returning `Ternary` (Yes/No/Unknown)
- `AssumptionBag`: contains `Vec<Entry>` where each `Entry { modality, source, claim, confidence }`
- `ClaimPayload` enum: 9+ variants covering FiniteClass, Sort, Bounds, Cardinality,
  Distribution shape, Monotonicity, Continuity, Prop (logical propositions), and more
- `Source`: either `Asserted` (user-provided) or `Derived { producing_kernel: CacheKey }`
  — direction is NOT stored (was a sub-clause E violation in v1)
- `ValueWithProvenance { fingerprint: [u8;32], provenance: AssumptionBag }`

### Two canonicalization rules

- **Rule 1** (bag-level): set semantics — sort by (modality, source, claim), deduplicate
- **Rule 2** (OperationHistory): sequence semantics — order-preserving, no dedup

### Sweep-level consequences

DEC-029 ripples through: Sweep 12B (FiniteClass/elision), Sweep 29b (DescriptiveResult.precision),
Sweeps 24/25/28 (Results + Distribution applications), Sweep 23 (RoutingRecord).

---

## What was wave-2 about?

Wave-2 adversarial testing (`R:\tambear\crates\tambear\tests\dec_029_adversarial_2.rs`,
written 2026-05-04) targeted structural correctness of the knowledge/canonicalize
implementation. It ran ATK-16 through ATK-41+.

### Bugs found in wave-2

**ATK-16**: `feed_finite_class` feeds members in push order — silent cache miss for
out-of-order-equal sets. The bag fingerprint differed between `{A, B}` inserted in
A-then-B vs B-then-A order, even though both represent the same finite class. Fixed via
Rule 1 sort-before-fingerprint.

**ATK-17**: `KnownExhaustive({})` vs `KnownCardinality{n:0}` must be semantically
distinct fingerprints. An empty exhaustive enumeration is a different claim than "known
to have zero elements."

**ATK-30**: `Sort::Nat` and `Sort::Named("ℕ")` had the same Display output. A
Display-based serializer produced fingerprint collisions between a built-in sort and a
user-named sort with the same symbol.

**ATK-31**: `Term::Var("3")` and `Term::Lit(3.0)` same Display collision in `Prop::Eq/Le/Lt`.

**ATK-32**: `Term::Lit(3.0)` and `Term::NatLit(3)` same Display collision.

**ATK-33**: Unescaped pipe separator in `format!("{a}|{b}")` — `Term::Var("x|y")` +
`Var("z")` collides with `Var("x")` + `Var("y|z")`.

---

## Status: what's committed vs what remains

### Already committed in tambear

Running `git log --oneline -30` from `R:\tambear` showed active DEC-029 implementation
commits. The knowledge module exists at:

```
R:\tambear\crates\tambear\src\knowledge\
  aggregate.rs
  canonicalize.rs
  impls.rs
  ingest.rs
  mod.rs
  operation_history.rs
  trait_def.rs
  types.rs
  walk.rs
```

Recent commits addressed the wave-2 ATK catalog:
- `canonicalize: fix feed_sort Sort::Named/Arrow/Product prefix` — ATK-30 and related
  Display-collision fixes
- `knowledge: version bump 2 + walk diamond-vs-cycle fix (ATK-39/41 green)` — the
  diamond traversal and cycle-detection bugs

Most ATK-16 through ATK-41+ appear resolved through these commits. The knowledge module
is active (not a stub).

### The campsite itself is nearly empty

`R:\winrapids\campsites\dec-029-impl\` has only one non-empty file:
`adversarial/20260508...-creation.md` — a campsite creation stub. The actual wave-2
work happened directly in `R:\tambear` without leaving campsite markdown artifacts. This
is consistent with the "campsite emptiness pattern" observed across multiple campsites
in this survey: the coordination artifact exists, but the technical work lives in the
substrate (git commits, actual source files).

### Compile-time blocker (antigen crate)

When attempting to run the wave-2 test file directly:

```
error[E0599]: no method named `detect_kind` found for mutable reference
  --> R:\antigen\antigen\src\audit.rs:317:25
317 |         let kind = self.detect_kind(&item.attrs);
```

`detect_kind` at `audit.rs:317` is an associated function, not a method. Correct call:
`FunctionIndexVisitor::<'_>::detect_kind(&item.attrs)`. This blocked direct test-run
verification during the survey. The fix is one line in `R:\antigen\antigen\src\audit.rs`.

---

## Is wave-3 implied?

Yes, structurally. The `ingest` function — `ingest(consumer_step: CacheKey, source: &ValueWithProvenance) -> Vec<Entry>` — produces `Derived` entries that chain provenance across computation steps. The current wave-2 testing focused on the fingerprint-collision surface (ATK-16 through ATK-41+). Wave-3 would logically address:

1. **Multi-hop derivation chains**: does a Derived entry whose producing_kernel is itself
   a derived result correctly trace back to the original Asserted source?
2. **Compatibility checking under specialization**: `can_specialize` returning Yes when
   one bag's claims are a strict subset of another's — adversarial tests for boundary
   cases where subset check fails silently
3. **Diamond convergence correctness**: two independent computation paths producing the
   same intermediate — does the walk algorithm correctly identify them as the same node
   rather than two distinct sources?

The ATK-39/41 commit ("walk diamond-vs-cycle fix") suggests this territory was already
partially explored. Wave-3 would formalize and harden it.

---

## Open questions / urgent unfinished work

1. **Antigen compile error** blocks clean `cargo test` runs on the knowledge module.
   Fix is known: `audit.rs:317` associated-function syntax. Should be fixed before
   wave-2 verification is called complete.

2. **Sweep 12B (FiniteClass/elision)** — listed in DEC-029 sweep consequences but
   sweep status (from `R:\tambear\sweeps\`) was not fully audited in this survey.
   If 12B is not yet landed, FiniteClass claims from DEC-029 have no consumer.

3. **Sweep 29b (DescriptiveResult.precision)** — same gap. DEC-029 feeds precision
   metadata into DescriptiveResult. If 29b is not implemented, that pathway is dead.

4. **GAP-008 (validity dispatcher)** in `R:\tambear\TODO.md` is related: the routing
   logic that consults knowledge to decide when to elide precision checks is still open.

---

## Summary judgment

DEC-029's core implementation is landed and active in `R:\tambear`. The wave-2 bug
catalog (ATK-16 through ATK-41+) is largely resolved through committed fixes. The
knowledge module has 9 source files with non-trivial implementation.

The remaining surface risk is: (a) the antigen compile error blocking clean test
verification, (b) downstream sweeps (12B, 29b, 23) that DEC-029 feeds into being
incomplete, and (c) wave-3 diamond-convergence tests not yet written.

Nothing here is an urgent blocker to other work in the tambear-formalize mandate —
DEC-029 itself is formalized and committed. The open items are downstream integration
gaps, not protocol gaps.
