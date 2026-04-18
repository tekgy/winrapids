<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Biology-Math-Scan Next-Landscape Proposals

Written: 2026-04-10
Context: Task #8 complete (discover_* Layer 4 superposition functions shipped).
Biology math gaps document written. Fock boundary counter-example found and resolved.
Bijection theorem connected to spring topology and holographic error correction.

---

## What I'm seeing from this angle

Three threads from today converge on the same structure:
- The Fock boundary requires fixed-size semigroup representation (adversarially derived)
- TamSession requires fixed-size intermediates (architecturally enforced)
- The bijection theorem requires a canonical (grouping, op, transform) triple (theoretically derived)

These are the same constraint, stated three ways. The next landscape should make this
convergence operational — not just named, but built.

---

**1. `industrialization/architecture/op-identity-degenerate`** ← HIGHEST PRIORITY

The op-identity-method campsite exists but the spec is incomplete. Every scannable Op
needs TWO methods, not one:
- `identity()` — neutral element for Blelloch tree padding (e.g., -∞ for Max, 0 for Add)
- `degenerate()` — signal for invalid input (NaN for scalar ops, Mat::with_det_zero() for matrix ops)

The singularity-as-identity bug (linear_algebra.rs:1560 returning Mat::eye() for singular Q)
is exactly the same structural error as the NaN-eating fold bug — conflating the mathematical
identity element with "what to return when things go wrong."

The adversarial-singularity-class campsite and op-identity-method campsite should be
co-owned: the theoretical form is op-identity-degenerate, the audit is adversarial-singularity-class.

Also: include Op::TropicalMinPlus here. The tropical semiring needs its own identity()
(+∞) and degenerate() (NaN). Scout named this as tropical-semiring campsite (#1).
They're the same campsite — one implementation of op-identity-degenerate gets both.

Role: aristotle (design), adversarial (audit), pathmaker (implementation)

---

**2. `industrialization/sharing/validity-as-tag-dimension`** ← ARCHITECTURAL

From the garden conversation and my response to navigator: validity-semantics is NOT
a sixth grouping type. It's a fourth dimension of the IntermediateTag.

Current IntermediateTag: (grouping, op, transform) → one canonical intermediate.
Correct IntermediateTag: (grouping, op, transform, validity-policy) → one canonical intermediate.

Method A caching under Propagate semantics ≠ reusable by Method B under Error semantics.
The bug: right now TamSession silently shares intermediates computed under different
validity assumptions. No compatibility check exists.

This is small to implement (add ValidityPolicy enum to IntermediateTag, add
is_compatible() check in TamSession::get()) but architecturally load-bearing.
The adversarial-validity-semantics campsite (adversarial's #1) is the audit;
this campsite is the design and implementation.

Connection to adversarial: the adversarial's Propagate/Ignore/Error policy declaration
at function level makes the policy computable at cache-time, which makes this check possible.
These two campsites are a dependency pair — design the tag first, then the audit makes sense.

Role: aristotle (tag design), pathmaker (implementation), adversarial (audit)

---

**3. `industrialization/theory/bijection-lint`** ← MEDIUM TERM

The bijection theorem says each (grouping, op, transform) triple maps to exactly one
canonical intermediate. This should be OPERATIONAL: a lint that tells you "this combination
is equivalent to an existing primitive."

Concretely: if someone writes a new method that happens to be an accumulate(ByKey, Add, identity),
the lint catches that this is just grouped_sum — which already exists. No duplication.

The lint would:
1. Decompose each new method into its (grouping, op, transform) triple
2. Check the triple against the known primitive catalog
3. Warn if an equivalent primitive already exists (or flag it as genuinely new)

This turns the bijection theorem from theoretical into a compiler tool. The spring topology
becomes an optimization objective: minimum-tension state = minimum duplicated primitives.

Role: math-researcher (triple decomposition algorithm), pathmaker (lint implementation),
scout (kingdom-classification-audit provides the catalog to check against)

---

**4. `industrialization/biology/mass-action-rhs`** ← DOMAIN GAP

From the biology gaps document: the most-used biology primitive that tambear lacks is
mass_action_rhs — the right-hand side of the ODE system for chemical kinetics.

```
mass_action_rhs(state: &[f64], stoich: &Mat, rates: &[f64]) -> Vec<f64>
```

This is Kingdom A (accumulate(All, Multiply) + gather), structurally identical to
a sparse matrix-vector product. The stoichiometric matrix is constant (data-independent),
the rates vector is data-determined. Affine map in state-space = Kingdom A.

This primitive enables: pharmacokinetics, epidemiological compartmental models (SIR, SEIR),
metabolic pathway modeling, enzyme kinetics (Michaelis-Menten as special case),
any ODE-based system model.

Unlike other biology gaps (Needleman-Wunsch, Gillespie SSA), mass_action_rhs is
structurally the simplest and the enabling primitive for the whole family.
Gillespie SSA builds on it. SIR models ARE it.

Role: math-researcher (first-principles derivation), scientist (oracle verification
against Scipy integrate.odeint reference), pathmaker (implementation)

---

**5. `industrialization/theory/tropical-matmul-shortest-path`** ← CONNECTS TO SCOUT'S #1

Scout named tropical-semiring (Op::TropicalMinPlus) and I endorse it completely.
Adding one dimension: once Op::TropicalMinPlus exists, tropical matrix multiplication
gives ALL-PAIRS shortest paths via repeated squaring = O(n³ log n) naive, or
sub-cubic via matrix product — both Kingdom A (no sequential dependency).

The campsite should include:
- Op::TropicalMinPlus with correct identity()+∞ and degenerate(NaN)
- Tropical matmul primitive (accumulate(All, tropical_min_plus_row_col_product))
- Delegation: PELT → affine_prefix_scan in tropical semiring
- Delegation: Viterbi → tropical matmul for HMM decoding

This is the proof-of-concept that the semiring abstraction works: swap the semiring,
get a different algorithm for free. Same Blelloch tree, different Op.

Role: scout (Kingdom classification verification), math-researcher (tropical matmul derivation),
pathmaker (implementation)

---

## Priority read from this angle

**Must-first (architectural prerequisite)**: validity-as-tag-dimension + op-identity-degenerate
These two campsites are co-dependent. The identity/degenerate distinction must be formalized
before the singularity audit makes sense. The validity-as-tag must be designed before
the validity-semantics audit makes sense. Do these first, do them together.

**High leverage (unlocks GPU farming)**: tropical-matmul-shortest-path
Scout is right that affine-prefix-scan is the foundation. Tropical matmul is the next
layer — same structure, different semiring.

**Domain gap (closes biology)**: mass-action-rhs
Small to implement, high generality, closes a structural gap that blocks an entire
category of science models from running on tambear.

**Long game (makes the theory operational)**: bijection-lint
This is the one that turns research findings into compiler infrastructure. Not urgent,
but the highest long-term leverage.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

