# Challenge 26 — Kingdom Taxonomy Should Be Derived, Not Declared

**Date**: 2026-04-06  
**Type A: Representation Challenge — fix the data structure**

---

## The Traditional Assumption

The kingdom of an operation (A = parallel accumulate, B = sequential, C = iterative convergence) must be manually declared in a lookup table. The lookup table `kingdom_of()` in `tbs_lint.rs` maps string names to kingdoms.

## Why It Dissolves

The kingdom of an operation IS its algebraic structure. If operations were typed with their algebraic properties (challenge 12), kingdoms fall out automatically from the type.

---

## The Concrete Problem

`tbs_lint.rs::kingdom_of()` is a string→Kingdom lookup table:
```rust
("garch", None) => (Kingdom::C, Some(SharedSubproblem::Covariance)),
```

This has two bugs:

**Bug 1**: GARCH is classified as Kingdom C (iterative convergence), but it's actually Kingdom A (parallel prefix scan). The sequential variance recursion `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}` is a 2×2 matrix product, and matrix products are associative — so it's a single-pass parallel prefix scan. Kingdom A.

**Bug 2**: The SharedSubproblem is Covariance, but GARCH doesn't compute a covariance matrix — it computes a sequential variance. The misclassification is self-consistent but wrong.

Both bugs exist because kingdoms are DECLARED, not DERIVED. When challenge 13 (GARCH as matrix prefix scan) is implemented, someone has to remember to update the lookup table. They won't.

---

## The Derived Alternative

If `accumulate(grouping, expr, op)` takes typed algebraic parameters:
- `Monoid<f64, Add, 0.0>` → Any operation using this is Kingdom A
- `Sequential<State, Transition>` where Transition is not associative → Kingdom B
- `Monoid<f64, Add, 0.0>` in an iterative solver → Kingdom A subproblem inside Kingdom C

The kingdom is a TYPE-LEVEL property:
```
Kingdom::A  ⟺  op satisfies Monoid (associative + identity)
Kingdom::B  ⟺  op is state-dependent, non-associative
Kingdom::C  ⟺  op has a convergence criterion (fixed-point seeking)
```

The lint system could then derive the kingdom from the type signature, not from the string name.

---

## The Impact of the GARCH Bug

L201 currently says: "GARCH and [subsequent op] both compute covariance matrix — shared automatically." This is wrong. GARCH doesn't compute a covariance matrix and isn't Kingdom C. If a user chains `garch().pca()`, L201 will fire (incorrectly suggesting kernel fusion of GARCH with PCA's covariance computation).

After challenge 13 is implemented:
- GARCH kingdom should update to: `(Kingdom::A, Some(SharedSubproblem::PrefixScan))` where PrefixScan is a new subproblem type

This won't happen automatically. The lookup table will remain wrong.

---

## The Insight About SharedSubproblem

The `SharedSubproblem` enum is already the beginning of a **fusion register** — a compile-time index of which kernel computations can be shared. It currently has 8 types (DistanceMatrix, Covariance, GramMatrix, WeightedCovariance, Henderson, CrossProduct, RiskSet, Autocorrelation).

If this were derived from algebraic types:
1. Operations with identical `(grouping, op)` signatures share the subproblem automatically
2. No lookup table needed — the type IS the subproblem identity
3. New operations get correct fusion behavior for free

The SharedSubproblem enum = the quotient of the operation space by shared accumulate structure. The type system should generate this quotient, not a human.

---

## Most Actionable

1. Add `SharedSubproblem::PrefixScan` to the enum (needed for GARCH/ARMA/AR after challenge 13)
2. Fix GARCH classification: `(Kingdom::A, Some(SharedSubproblem::PrefixScan))` (or Kingdom B until the matrix scan primitive exists — but mark it transitional)
3. Add comment in `kingdom_of()` noting this is a transitional lookup table, not the final architecture

The long-arc fix (challenge 12 + 26 together): accumulate type parameters → kingdoms + subproblems derived automatically.
