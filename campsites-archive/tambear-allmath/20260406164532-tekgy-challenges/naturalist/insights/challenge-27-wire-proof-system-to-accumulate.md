# Challenge 27 — Wire the Proof System to Accumulate

**Date**: 2026-04-06  
**Type C: Foundation Challenge — the two systems should be one thing**

---

## The Current State

`proof.rs` and `accumulate.rs` are parallel systems that don't know about each other.

- `proof.rs` has: `Term::Accumulate { grouping, expr, op, data }` — accumulate calls as proof terms
- `proof.rs` has: `CompositionRule::ParallelMerge` — "from mergeability, deduce parallel reduction correctness"
- `proof.rs` has: `Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0))`
- `accumulate.rs` has: `accumulate(data, Grouping::ByKey, expr, Op::Add)` — no algebraic type

They're speaking the same language but not to each other.

---

## What the Architecture Implies

`Term::Accumulate` in `proof.rs` represents an accumulate call AS A TERM IN A PROOF. This means: the proof system was designed to reason about accumulate calls. But the accumulate function itself doesn't generate or return correctness certificates.

The gap: `accumulate(ByKey, ..., Add)` could automatically generate the proof certificate:
```
Proof::ByComposition(
    CompositionRule::ParallelMerge,
    [Proof::ByStructure(commutative_monoid(Real, Add, 0), StructuralFact::Associativity)]
)
```

The certificate says: "This accumulate is correct because Add on Real is associative (proved by the declared commutative monoid structure)."

---

## The Minimal Connection

One small function makes the connection:

```rust
impl Op {
    /// Return the canonical algebraic structure for this operation, if any.
    pub fn canonical_structure(&self) -> Option<Structure> {
        use crate::proof::*;
        match self {
            Op::Add    => Some(Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0))),
            Op::Mul    => Some(Structure::commutative_monoid(Sort::Real, BinOp::Mul, Term::Lit(1.0))),
            Op::Max    => Some(Structure::lattice_op(Sort::Real, BinOp::Max, Term::Lit(f64::NEG_INFINITY))),
            Op::Min    => Some(Structure::lattice_op(Sort::Real, BinOp::Min, Term::Lit(f64::INFINITY))),
            Op::Sub    => None,  // NOT associative — explicit
            Op::Div    => None,  // NOT associative — explicit
            Op::Count  => Some(Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0))),
        }
    }
    
    /// Prove that this operation is safe for parallel reduction.
    pub fn parallel_safe_proof(&self) -> Option<Proof> {
        self.canonical_structure().map(|structure| {
            Proof::ByComposition(
                CompositionRule::ParallelMerge,
                vec![Proof::ByStructure(structure, StructuralFact::Associativity)],
            )
        })
    }
}
```

Cost: minimal. Impact: every accumulate call now has a verifiable algebraic correctness certificate.

---

## The Deeper Connection: Hole-Filling = Learning

When `accumulate` is called with a custom `Op::JitExpr(phi_string)` — a JIT-compiled phi expression — the canonical_structure is `None`. The proof system generates `Proof::Hole("prove that phi is associative")`.

This is the hole from my first naturalist observation: `Term::Hole(Sort)` is the type-theoretic `sorry` — a placeholder that says "I know what type this proof has, I haven't proved it yet."

For user-defined phi expressions, the proof obligation is: "prove the phi expression forms a monoid (or semigroup) on the data type." This is a property that could be:
1. Checked computationally (ByComputation: sample values, verify associativity holds)
2. Left as a Hole (the user promises it's associative)
3. Proved by structure (if the phi expression is syntactically recognized as a known monoid)

The hole-filling mechanism enables PROGRESSIVE FORMALIZATION: start with computational checks (cheap, probabilistic), promote to structural proofs (zero-cost, guaranteed) as the algebra becomes explicit.

---

## Connection to Challenge 12 (Algebra as Type System)

Challenge 12 proposed: instead of `Op::Add`, use `Monoid<f64, Add, 0.0>` as the type parameter. The `canonical_structure()` function is the INTERMEDIATE STEP: it keeps the current Op enum but makes the algebraic properties explicitly accessible.

The path:
1. Add `Op::canonical_structure()` → Op now carries algebraic meaning
2. Add `accumulate_certified(data, grouping, expr, op)` → returns (result, ProofCertificate)
3. Long arc: make `Monoid<T, op, identity>` a trait, making Op the type-level algebra

The intermediate step (1+2) is a few lines of code. The long arc is a major refactor. Both are correct. Start with the intermediate.

---

## What This Unlocks

1. **Debuggability**: when accumulate produces a wrong answer, the proof system knows whether the issue is the grouping (structural), the expression (phi), or the operation (algebraic). 

2. **Optimization**: the proof system can certify whether two accumulate calls can be fused (both have the same Structure, so their SharedSubproblem is algebraically identical, not just textually).

3. **Teaching**: `.tbs` scripts that call `accumulate` now have auditable mathematical foundations. The chain `kmeans().pca()` both use Covariance, and the system can PROVE why fusion is correct — because both operations are characterized by the same `(Vec<f64>, ByKey, v·vᵀ, Add)` structure.

4. **Collatz research**: when `fold_irreversibility.rs` uses `accumulate(Prefix, ..., Add)`, the proof system automatically certifies the prefix scan as correct. The Lyapunov function reasoning can be elevated to a verified theorem instead of a computed empirical finding.
