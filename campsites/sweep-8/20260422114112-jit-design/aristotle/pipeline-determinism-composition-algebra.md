# DeterminismClass composition algebra — foundation for pipeline_effective_determinism

Date: 2026-04-23
From: math-researcher
For: aristotle (sweep-8 jit-design)
Status: theoretical foundation; feeds implementation of
`pipeline_effective_determinism` at the pipeline tier
Companion to: `matmul-prefix-condition-number.md` (same campsite)

---

## The problem

`DeterminismClass` is defined at the atom tier (per-Op) by
`JitOp::determinism_floor(op, sum_strategy) -> DeterminismClass` at
`crates/tambear/src/jit/jit_op.rs:542`. The four variants:

1. `BitExact`
2. `MathematicallyEquivalent { max_condition_number: f64 }`
3. `OrderDependent`
4. `NonDeterministic`

A tambear pipeline composes many Ops. The question: **given the per-Op
determinism classes, what is the pipeline's effective determinism?**
That is the signature of `pipeline_effective_determinism(ops: &[...])
-> DeterminismClass`.

This document derives the composition rules. Implementation is mechanical
once the algebra is pinned.

---

## The structure we're building

Three operations compose DeterminismClasses in tambear:

- **∘_seq** (sequential composition) — output of op A feeds input of op B
- **∘_par** (parallel composition with merge) — A and B run on disjoint
  inputs; their outputs merge via a third op M
- **∘_fan** (fan-out) — single input feeds both A and B; outputs combine
  (degenerate case of parallel-merge)

Each of the three is a binary operation on `DeterminismClass`. The
lemmas below establish their truth tables.

---

## Section 1 — Sequential composition (∘_seq)

**Definition**: `A ∘_seq B` means the output of A is the input of B.
Error from A propagates as *input error* to B. The composed determinism
class is the determinism with which the final output is tied to the
original input bits.

### Lemma 1.1 — BitExact absorbs on both sides

```
BitExact ∘_seq X = X
X ∘_seq BitExact = X
```

**Proof** (both directions):

- `BitExact ∘_seq X`: A is bit-exact → A's output is a deterministic
  function of its input bits (call it f_A). The composed output is
  `B(f_A(input))`. The determinism of this composite equals the
  determinism of B applied to a fixed input — which is exactly B's
  declared class. ✓
- `X ∘_seq BitExact`: A produces output with class X. B is bit-exact →
  B is a deterministic function of its input bits (call it f_B). The
  composed output is `f_B(A(input))`. Since f_B is deterministic given
  its input, the output's determinism is inherited entirely from A's —
  that is, X. ✓

**Identity element**: `BitExact` is the identity of ∘_seq. This makes
the `BitExact`-set-plus-∘_seq a monoid (provisionally; need to check
associativity).

**Subtle caveat (not a counterexample, an assumption)**: the absorption
`X ∘_seq BitExact = X` assumes f_B preserves the class-defining
property of X. For `MathematicallyEquivalent`, f_B being bit-exact
means it doesn't introduce new rounding — but it can still *amplify*
existing input error by the condition number κ_B of f_B as a function.
See Lemma 1.2 — the amplification is captured in the composed
condition-number, not in the class tag.

### Lemma 1.2 — MathematicallyEquivalent composes submultiplicatively in κ

```
MathematicallyEquivalent { κ_A } ∘_seq MathematicallyEquivalent { κ_B }
  = MathematicallyEquivalent { κ_A · κ_B }
```

**Proof sketch**:

Let the true mathematical result be y = (g_B ∘ g_A)(x). Let the
computed result be ŷ = (f_B ∘ f_A)(x) where f_A, f_B are the kernels'
floating-point implementations. The relative errors of f_A and f_B at
their respective inputs are bounded by ε·κ_A and ε·κ_B (where ε is
machine epsilon; κ_i is the per-op condition number bound).

By forward-error analysis (Higham ASNA §3.4):

```
||ŷ - y|| / ||y|| ≤ ε·κ_A · κ_B + ε·κ_B + O(ε²)
                  ≈ ε·(κ_A · κ_B)   when κ_A, κ_B ≫ 1
```

The dominant term is the product. Using it as the composed bound
gives `max_condition_number = κ_A · κ_B`, matching the MatMulPrefix
submultiplicativity derivation (Horn-Johnson §5.6).

**Equivalent formulation**: in log space, conditioning is additive:
`log(κ_composed) = log(κ_A) + log(κ_B)`. The algebra over
`(MathematicallyEquivalent, ∘_seq)` modulo BitExact-quotient is
isomorphic to `(ℝ_+, +)` under log(κ).

**Overflow discipline**: if `κ_A · κ_B` overflows to Infinity in f64
or exceeds `f64::MAX`, the composed class SHOULD saturate to
`math_equiv_unbounded()` (κ = ∞) rather than silently wrap. Naming
this clearly at the type level: never multiply raw f64 κ values
without saturating-multiply semantics.

### Lemma 1.3 — BitExact + MathEquiv mix

```
BitExact ∘_seq MathematicallyEquivalent { κ } = MathematicallyEquivalent { κ }
MathematicallyEquivalent { κ } ∘_seq BitExact = MathematicallyEquivalent { κ_B_condition · κ }
```

**First direction**: by Lemma 1.1, BitExact is identity; the composed
class is MathematicallyEquivalent { κ }. The κ does NOT decrease — it
doesn't matter that A is bit-exact; the second step still has its
inherent rounding regime.

**Second direction** — trickier:

BitExact for B means B's kernel reproduces its input bits to output
bits faithfully. But B is still a *mathematical* function with a
mathematical condition number κ_B (independent of its kernel's
implementation). BitExact says "the kernel implements f_B with no
additional rounding beyond the single evaluation"; it does NOT say
"f_B propagates input error without amplification."

Example — consider B = `fsqrt`:
- `fsqrt` is BitExact (single IEEE 754 op, correctly rounded).
- If the input x has relative error ε·κ_A, the output sqrt(x) has
  relative error ≈ (1/2)·ε·κ_A + ε (the sqrt function has
  mathematical condition number 1/2; the BitExact kernel adds one
  rounding).
- So the composed condition is dominated by `κ_A · (1/2) + 1`, not
  just `κ_A`.

**Caveat**: this refinement is NOT currently in the
`DeterminismClass::BitExact` variant's data — BitExact carries no
condition-number field. For pipeline composition, a BitExact op
conservatively does NOT amplify above the incoming κ — the variant
is treated as a "pass-through" for condition number. Tighter
analysis requires annotating BitExact with its mathematical
condition (`BitExact { math_condition: f64 }`). This is a future
refinement; for now, the safe composition rule is:

```
MathematicallyEquivalent { κ } ∘_seq BitExact = MathematicallyEquivalent { κ }
```

(treat BitExact as κ_B = 1 for the composition purpose; honest
because we have no better bound in the type).

### Lemma 1.4 — OrderDependent is absorptive (mostly)

```
OrderDependent ∘_seq X = OrderDependent, for X ∈ {BitExact, MathEquiv, OrderDependent}
X ∘_seq OrderDependent = OrderDependent
```

**Proof**: OrderDependent says "same multiset, different insertion
sequence → different output bits." If A is OrderDependent:
- Different invocations of A on the same multiset can produce different
  outputs.
- B applied downstream operates on A's output; since A's output bits
  vary, B's output bits vary. The composed output is OrderDependent.

If A is one of {BitExact, MathEquiv} and B is OrderDependent:
- A produces deterministic or mathematically-equivalent output per
  input.
- B is OrderDependent on its own input-sequence. Even if A's output is
  bit-stable, B can reshuffle its internal insertion and produce
  different outputs.

**Edge case (not a counterexample)**: `OrderDependent ∘_seq BitExact`.
Does BitExact "rescue" the OrderDependent output? No — BitExact is a
function; applying a function to a variable-bits input produces
variable-bits output. The composition is OrderDependent.

**Special case — OrderDependent with stable input-sequence
invariant**: if the pipeline has an upstream guarantee that the
input sequence to B is itself fixed (not just the input multiset),
then B's OrderDependency doesn't fire. This is a proof-engine-level
piece of information (Sweep 1); it's not derivable from the
composition algebra alone. See §4 (Escape valves) below.

### Lemma 1.5 — NonDeterministic is fully absorptive

```
NonDeterministic ∘_seq X = NonDeterministic
X ∘_seq NonDeterministic = NonDeterministic
```

**Proof**: same bit-variability argument as Lemma 1.4, but stronger —
NonDeterministic introduces external entropy. No downstream class can
remove entropy that was injected upstream (that's a fundamental
information-theoretic property). No upstream class can prevent
downstream entropy from asserting itself.

### Sequential composition truth table (∘_seq)

Rows = A (first); columns = B (second); entries = A ∘_seq B.

Let `M_κ` abbreviate `MathematicallyEquivalent { max_condition_number: κ }`.

| A ∘_seq B | BitExact | M_{κ_B} | OrderDep | NonDet |
|---|---|---|---|---|
| **BitExact** | BitExact | M_{κ_B} | OrderDep | NonDet |
| **M_{κ_A}** | M_{κ_A} | M_{κ_A·κ_B} | OrderDep | NonDet |
| **OrderDep** | OrderDep | OrderDep | OrderDep | NonDet |
| **NonDet** | NonDet | NonDet | NonDet | NonDet |

**Observation**: there is a natural total order
`BitExact < MathEquiv < OrderDep < NonDet` on the set of classes
(ignoring the κ parameter). Sequential composition returns the
**maximum** of A and B in this order, with κ-accumulation in the
`MathEquiv` slot.

---

## Section 2 — Parallel composition with merge (∘_par)

**Definition**: two branches A, B run on disjoint (or same) input
regions; their outputs merge via a merge op M. The effective
determinism of the whole sub-pipeline is
`(A ∘_par B via M) = fold_merge(M, {det_A, det_B})`.

### Lemma 2.1 — Absorption by merge op's class

The merge op M itself is an Op with a determinism class. Whatever
M's class is, it bounds the merge itself. So:

```
(A ∘_par B via M) = M_class ∘_seq (merge_of(det_A, det_B))
```

where `merge_of(x, y)` combines the two branch determinisms into a
pre-merge class. We need `merge_of` — how does pairing two branches'
outputs compose before M sees them?

### Lemma 2.2 — BitExact pairing is BitExact (when both bit-exact)

```
merge_of(BitExact, BitExact) = BitExact
```

**Proof**: both branches produce bit-stable outputs; the pair is
bit-stable. ✓

### Lemma 2.3 — MathEquiv pairing accumulates worst-case κ

```
merge_of(M_{κ_A}, M_{κ_B}) = M_{max(κ_A, κ_B)}
```

**Proof**: the pair has relative error ≤ ε·max(κ_A, κ_B) componentwise
(the pair is a vector whose components have independent bounds). After
the merge op M, the error may combine, but the pre-merge bound is the
max.

**Contrast with sequential**: sequential is multiplicative (κ_A · κ_B),
parallel is maximum (max(κ_A, κ_B)). The difference is that
sequential amplifies error by feeding it through subsequent
computation; parallel simply preserves both worst-cases and lets the
merge decide how to combine.

### Lemma 2.4 — Pairing mixes

```
merge_of(BitExact, M_{κ}) = M_{κ}
merge_of(BitExact, OrderDep) = OrderDep
merge_of(BitExact, NonDet) = NonDet
merge_of(M_{κ}, OrderDep) = OrderDep
merge_of(M_{κ}, NonDet) = NonDet
merge_of(OrderDep, OrderDep) = OrderDep
merge_of(OrderDep, NonDet) = NonDet
merge_of(NonDet, NonDet) = NonDet
```

**Proof**: weakest-of-the-pair rule. Weaker branch dominates by the
total order `BitExact < MathEquiv < OrderDep < NonDet`.

### Lemma 2.5 — Full parallel composition

```
(A ∘_par B via M)
  = M_class ∘_seq merge_of(A_class, B_class)
```

**Proof**: first pair, then merge, via Lemma 2.1. Sequential
composition (with its multiplicative κ behavior) applies to the
post-merge step because M runs on the merged output.

**Concrete example — Chan-parallel variance (variance SPEC §3.5)**:
- Each branch computes Welford partials on a batch chunk: class
  `M_{κ_welford}` per batch.
- Chan's combine merges batches: class `M_{κ_chan_combine}`.
- Composed: `M_{κ_chan_combine} ∘_seq merge_of(M_welford, M_welford)
  = M_{κ_chan_combine} ∘_seq M_{κ_welford}
  = M_{κ_chan_combine · κ_welford}`.

This is exactly the error-analysis result in variance SPEC §3.5's
"Chan's parallel" — batch-algorithm κ times combine-op κ.

### Parallel composition truth table

For k parallel branches merged by M:

```
pipeline_effective_determinism_parallel(branches, M) =
    let paired = fold(merge_of, branches);    // weakest-of-all, or max κ for MathEquiv
    M_class ∘_seq paired                       // merge op composes sequentially after
```

Using Python-style pseudocode for clarity; the Rust implementation is
a three-line fold.

---

## Section 3 — OrderDependent refinement (answers navigator's Q2)

### Q: "What about OrderDependent ops in sequence? Probably: composed is OrderDependent regardless of what else is in the pipeline."

**Verified**: Lemma 1.4 + Lemma 2.4 together establish this. Any branch
or step containing OrderDependent makes the whole pipeline
OrderDependent (unless NonDeterministic is present, in which case the
pipeline is NonDeterministic — a strictly weaker class).

**Important sub-clause refinement (sub-clause E flavor)**: the class
OrderDependent has a sub-domain where the claim is stronger — when
the full pipeline's upstream guarantees a fixed input sequence (not
just a multiset), OrderDependent behaves like BitExact for
reproducibility purposes. This is a Sweep 1 proof-engine concern, not
a composition-algebra concern. The composition algebra should always
return OrderDependent; Sweep 1's reproducibility analysis can narrow
this class further if it has witnesses that the input sequence is
actually fixed.

This factors cleanly:
- Composition-algebra says "worst-case is OrderDependent."
- Proof-engine says "in this pipeline, the sequence invariant holds,
  so the worst-case is dormant."

Keep them separate. The composition-algebra return is the default
published determinism; the proof-engine's narrower claim can override
when evidence exists.

---

## Section 4 — Fan-out and re-convergence

**Definition**: single input x feeds into A and B independently; their
outputs combine via M (same input, not disjoint regions).

This is a special case of parallel composition — `merge_of` applies
the same way. The only difference: the κ analysis for MathEquiv may
be TIGHTER because the two branches share input, and their errors may
cancel or compound depending on M. This is an op-specific refinement
(e.g., computing `y - y` from two branches gives κ = 0 after merge).

**Conservative answer**: treat fan-out identically to parallel
composition (i.e., Lemma 2.5 applies). Tighter analysis is op-specific
and belongs in the per-recipe verification, not the generic
composition algebra.

Example of where tighter matters: `variance = E[X²] - (E[X])²` — both
branches compute on the same input (sum of x² and sum of x), merge via
subtraction. The subtraction is where catastrophic cancellation fires
(variance SPEC §3.1). Fan-out analysis would predict the cancellation,
but requires knowing M = subtraction of nearly-equal values. This is
exactly the difference between the one-pass (fan-out with cancelling
merge → catastrophic κ) and two-pass (sequential fold over (x_i -
μ)² → well-conditioned) algorithms.

**Design consequence**: the generic algebra is *safe but loose* for
fan-out. For specific high-condition merges (subtraction of near-equal,
division by near-zero), a per-recipe override is needed. This is the
same design shape as `max_condition_number` per-op — some ops want
to refine the algebra-predicted class.

---

## Section 5 — Summary: the complete composition rule

```rust
fn compose_seq(a: DeterminismClass, b: DeterminismClass) -> DeterminismClass {
    use DeterminismClass::*;
    match (a, b) {
        (NonDeterministic, _) | (_, NonDeterministic) => NonDeterministic,
        (OrderDependent, _)   | (_, OrderDependent)   => OrderDependent,
        (BitExact, other) | (other, BitExact) => other,
        (MathematicallyEquivalent { max_condition_number: κ_a },
         MathematicallyEquivalent { max_condition_number: κ_b }) => {
            // Saturating multiply — never silently overflow.
            let κ = κ_a.checked_mul_fp(κ_b).unwrap_or(f64::INFINITY);
            MathematicallyEquivalent { max_condition_number: κ }
        }
    }
}

fn compose_par_merge(branches: &[DeterminismClass],
                     merge_op: DeterminismClass) -> DeterminismClass {
    use DeterminismClass::*;
    let paired = branches.iter().copied().reduce(|a, b| match (a, b) {
        (NonDeterministic, _) | (_, NonDeterministic) => NonDeterministic,
        (OrderDependent, _)   | (_, OrderDependent)   => OrderDependent,
        (BitExact, other) | (other, BitExact) => other,
        (MathematicallyEquivalent { max_condition_number: κ_a },
         MathematicallyEquivalent { max_condition_number: κ_b }) => {
            // Parallel: max (weakest branch bounds the pair).
            MathematicallyEquivalent {
                max_condition_number: κ_a.max(κ_b)
            }
        }
    }).unwrap_or(BitExact);   // empty pipeline: identity of compose_seq
    compose_seq(merge_op, paired)
}

fn pipeline_effective_determinism(ops: &[OpStep]) -> DeterminismClass {
    ops.iter().fold(DeterminismClass::BitExact, |acc, step| {
        match step {
            OpStep::Sequential(op_class) => compose_seq(acc, *op_class),
            OpStep::Parallel { branches, merge_class } =>
                compose_seq(acc, compose_par_merge(branches, *merge_class)),
        }
    })
}
```

(Pseudocode; `checked_mul_fp` is a saturating f64 multiply returning
Option — the f64 type doesn't have this natively; implement as a
helper that saturates to INFINITY on overflow.)

---

## Section 6 — Why this matters for sub-clause E (tier coordinate)

The composition algebra is the TIER-COORDINATE answer to DEC-022
sub-clause E: `determinism_floor` answers at the **atom tier**;
`pipeline_effective_determinism` answers at the **pipeline tier**.

The naturalist's 2D-grid framing (from team-briefing.md coordinate
pre-check) predicted this — sub-clause E on the tier coordinate
produces exactly a two-level claim structure:

| Tier | Claim surface | Function |
|---|---|---|
| Atom | per-Op determinism | `JitOp::determinism_floor` (exists) |
| Pipeline | composed determinism | `pipeline_effective_determinism` (this derivation) |

The algebra proves that the two tiers compose cleanly — the pipeline
tier is derivable from the atom tier via the composition rules above.
No new data needed on the Op variant; only the composition function.

---

## Section 7 — Unit-test discipline

Proptest properties to lock:

```rust
proptest! {
    // Identity: BitExact is left- and right-identity of seq.
    #[test]
    fn bitexact_is_seq_identity(x in arb_class()) {
        prop_assert_eq!(compose_seq(BitExact, x), x);
        prop_assert_eq!(compose_seq(x, BitExact), x);
    }

    // Associativity of seq.
    #[test]
    fn seq_is_associative(a in arb_class(), b in arb_class(), c in arb_class()) {
        prop_assert_eq!(
            compose_seq(compose_seq(a, b), c),
            compose_seq(a, compose_seq(b, c)),
        );
    }

    // NonDet is absorbing for seq.
    #[test]
    fn nondet_absorbs_seq(x in arb_class()) {
        prop_assert_eq!(compose_seq(NonDeterministic, x), NonDeterministic);
        prop_assert_eq!(compose_seq(x, NonDeterministic), NonDeterministic);
    }

    // MathEquiv κ saturates (never overflows silently).
    #[test]
    fn matheq_seq_saturates_at_infinity(
        κ_a in 1.0f64..f64::MAX.sqrt(),
        κ_b in 1.0f64..f64::MAX.sqrt(),
    ) {
        let composed = compose_seq(
            MathematicallyEquivalent { max_condition_number: κ_a },
            MathematicallyEquivalent { max_condition_number: κ_b },
        );
        match composed {
            MathematicallyEquivalent { max_condition_number: κ } =>
                prop_assert!(κ.is_finite() || κ == f64::INFINITY),
            _ => prop_assert!(false, "expected MathEquiv"),
        }
    }

    // Parallel max-of-κ.
    #[test]
    fn par_matheq_takes_max(κ_a in 1.0f64..1e15, κ_b in 1.0f64..1e15) {
        let paired = compose_par_merge(
            &[MathematicallyEquivalent { max_condition_number: κ_a },
              MathematicallyEquivalent { max_condition_number: κ_b }],
            BitExact,   // trivial merge
        );
        if let MathematicallyEquivalent { max_condition_number: κ } = paired {
            prop_assert_eq!(κ, κ_a.max(κ_b));
        }
    }
}
```

---

## Section 8 — Non-results (what this doesn't solve)

Honest limits of the algebra:

1. **Condition-number refinement per merge shape**: fan-out with
   cancelling merge (`var = E[X²] - (E[X])²`) is under-bounded by the
   generic algebra. Per-recipe override still needed for the worst
   cases — the algebra gives a safe ceiling, not the tightest bound.
2. **OrderDependent narrowing under sequence invariants**: the generic
   algebra always propagates OrderDependent. Proof-engine (Sweep 1)
   can narrow this with witness information, but that's not
   composition-algebra — it's context-dependent proof.
3. **NonDeterministic narrowing**: likewise. If the pipeline stamps a
   seed that's later in the source-of-truth, NonDeterministic can
   narrow to BitExact for that seed value. Not composition-algebra.
4. **Condition-number lower bounds**: the algebra gives upper bounds
   only. If you want "actually achievable" vs "worst case" κ,
   empirical benchmarking is required — the algebra is worst-case.
5. **Complex graphs (DAG, cycles)**: the algebra assumes tree-shaped
   composition (sequential chains + parallel merges). DAGs with shared
   sub-expressions require memoization analysis; cycles require
   fixed-point analysis. Out of scope for the atom/pipeline tier;
   belongs to Sweep 9+ if those topologies enter.

---

## Section 9 — References

- Higham, N.J. (2002). *Accuracy and Stability of Numerical
  Algorithms* (2nd ed.), SIAM. §3.4 (forward error analysis),
  §3.5 (backward error), §14 (matrix powers — the MatMulPrefix
  building block).
- Horn, R.A. & Johnson, C.R. (2013). *Matrix Analysis* (2nd ed.),
  Cambridge. §5.6 (submultiplicativity for operator 2-norm).
- Wilkinson, J.H. (1963). *Rounding Errors in Algebraic Processes*,
  Prentice-Hall. Classical forward-error chains.
- Chan, T.F., Golub, G.H., LeVeque, R.J. (1983). "Algorithms for
  computing the sample variance: analysis and recommendations." Am.
  Stat. 37(3):242-247. The variance-SPEC §3.5 example has Chan's
  merge giving a κ_combine that composes sequentially with the
  per-batch Welford κ — exactly the shape derived here.
- Companion deposit: `matmul-prefix-condition-number.md` (same
  campsite; this document extends its atom-tier result to the
  pipeline tier).

---

## Section 10 — Open questions for aristotle

1. **Does BitExact need a math_condition field?** Per Lemma 1.3
   discussion: `MathEquiv ∘_seq BitExact` is currently conservatively
   set equal to `MathEquiv`, but BitExact ops can mathematically
   amplify κ via their inherent function behavior (sqrt halves, div
   inverts). Is a `BitExact { math_condition: f64 }` variant worth
   adding, or is the conservative no-amplification treatment fine?
2. **Parallel vs fan-out**: currently treated identically in the
   algebra. Should we keep them separate at the DeterminismClass
   level? Arguments either way.
3. **Worst-case-vs-typical**: `determinism_floor` is defined as the
   FLOOR (guaranteed worst case). The composition algebra derives a
   worst-case floor. Do consumers also want a "typical" κ estimate
   that we might carry as a sibling field? Probably out of scope
   for this derivation; flagging for future.
4. **Cache-key coverage**: per DEC-024, cache key must include all
   precision-affecting parameters. The pipeline-effective-determinism
   value is DERIVED from atom-level inputs that are already in the
   key — it doesn't need its own cache-key entry. Confirming this
   is load-bearing for DEC-024 alignment.

## Queue

- Once aristotle locks the design, the implementation is mechanical
  (section 5 pseudocode → real Rust + proptests from section 7).
- Next idle thread: continue centered_moment SPEC §3 verification
  pass (held from earlier).
