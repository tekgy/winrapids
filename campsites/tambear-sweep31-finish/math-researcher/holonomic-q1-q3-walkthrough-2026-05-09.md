---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher (literature/canonical-source verifier; structural reasoning for architectural commitments)
date: 2026-05-09
audience: navigator (routing); team-lead (CLAUDE.md ratification); naturalist (lens originator); aristotle (F13.C author)
sweep: holonomic-architecture walk-through
purpose: answer Q1 (formalization sufficiency) and Q3 (using() conflict resolution) from R:\winrapids\docs\architecture\holonomic-architecture.md §"Open questions for math-researcher walk-through". Unblocks team-lead's CLAUDE.md promotion of the holonomic principle to Irrevocable Architectural Principles.
inputs:
  - R:\winrapids\docs\architecture\holonomic-architecture.md (full read 2026-05-09)
  - R:\tambear\crates\tambear\src\jit\using_annotation.rs (DEC-020 implementation; full read of pub API)
  - R:\tambear\docs\decisions.md DEC-020 (state conservation parent decision)
  - R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md (F13.C graduation condition source)
  - R:\winrapids\important-conversation.md lines 1056-1063 (the cliffhanger this whole architecture closes)
  - R:\winrapids\docs\architecture\vocabulary.md (locked five-tier vocabulary)
status: posted to math-researcher campsite for navigator pickup; team-lead applies to holonomic-architecture.md as ratified responses; CLAUDE.md promotion follows
---

# Holonomic architecture — Q1 (formalization) and Q3 (using() conflict)

> **Story from the trail.** Both questions reduce to the same underlying
> property, looked at from two sides. **Q1 asks "is binding-order
> permutation a sufficient definition of holonomicity?" Q3 asks "what
> conditions on `using()` semantics ensure binding-order *actually*
> permutes?"** The first is a definitional question about the test; the
> second is an implementation question about whether the structure
> the test names is preserved by the language. Answering both at once
> is sharper than answering either separately, because the answers
> co-determine each other: the formalization is sufficient *iff* the
> conflict-resolution rule preserves the equational structure the
> formalization assumes.
>
> **The bottom line.** (1) Q1's "binding-order permutation produces
> the same cache key" formalization is sufficient *for the falsifiable
> test* but is missing one structural piece: it doesn't yet say what
> "the same value" means when two callers disagree. (2) Q3 fills that
> gap. The principled answer: `using()` bindings form an **idempotent
> commutative monoid under value-equality**, with conflict resolution
> resolved by **outermost-caller wins per provenance tier** (recipe
> default < TAM override < user override), AND **same-value rebinding
> is a no-op**. Different-value rebinding within the same provenance
> tier is a *compile-time error*, not a runtime override. This makes
> Q1's formalization actually sufficient because the conflict-resolution
> rule is itself path-independent — it doesn't depend on which call
> happened first.

---

## 1. Q1 — Path-independence formalization sufficiency

### 1.1 What the doc currently states

From `holonomic-architecture.md` §"The structural claim" + §"Cache keys
as the operationalization":

> The test that distinguishes [recipe vs IR tier]: **path-independence
> under binding-order permutation.** Bind parameters in different orders;
> do you get the same structure?
> ...
> The cache key IS the path-independence proof. Tag bytes per parameter
> slot, deterministic key from assignments. The order of binding is
> invisible to the cache key because the key only depends on the
> *assignments*, not the path through the code that produced them.

### 1.2 Sufficiency analysis

The proposed test, formally:

```
Let R be a recipe with parameter slots P = {p_1, ..., p_n}.
Let σ : Symmetric(P) be a permutation of binding order.
Holonomicity: ∀σ. cache_key(R bound under σ) = cache_key(R bound under id)
```

This is a clean falsifiable claim. It's also **necessary** for the
holonomic property — if binding order changes the cache key, behavior
depends on path, which is the definition of non-holonomic.

But is it **sufficient**? Three structural conditions need to hold for
the test to capture the full architectural property:

#### Condition A — Determinism of the cache-key function

`cache_key` must itself be a pure function of the assignment bag
(not a function of, say, system clock or memory layout). DEC-019's
serialization template already guarantees this: stable byte tags,
fixed feed order, BLAKE3 of the resulting byte string. This condition
is met by construction.

#### Condition B — Closure under assignment

For every parameter slot `p_i`, the set of valid values for `p_i` must
be enumerable / representable by the cache-key serialization — i.e.,
every assignment a user can make must map deterministically to a
unique byte sequence. The cache key gets the right answer only if
"two distinct values" ⇒ "two distinct byte serializations". This is
the F13 antibody condition for the cache key itself: **if a parameter
resists clean tagging, that's a signal of non-holonomic structure**
(naturalist's 2026-05-09 finding). Met by the recipe-tier design;
violation would surface as a TODO in fingerprint.rs.

#### Condition C — Resolvability of binding conflicts

**This is the gap Q1 implicitly assumes resolved but doesn't formalize.**
If two binding sites bind `p_i` with different values `v_1, v_2`, the
"final value at the cache key" must be **a function of the bag
{(p_i, v_1), (p_i, v_2)}, not of the order in which they were emitted
into the bag.**

Three possibilities for what that function could be:

| Resolution rule | Path-independent? | Architectural fit |
|-----------------|-------------------|-------------------|
| Latest-wins (LIFO of binding stack) | NO — `bind(v_1) then bind(v_2)` gives `v_2`, opposite gives `v_1`. Path-dependent. | ❌ Recipe-tier holonomic claim fails. |
| Earliest-wins (FIFO of binding stack) | NO — same shape, opposite asymmetry. | ❌ Same fail. |
| Outermost-caller-wins-by-provenance (this walk-through's recommendation) | YES — provenance is a partial order; the binding's tier (Default < TamOverride < UserOverride) is path-independent because it's a property of *who* bound, not *when*. Within a tier, same-value is no-op (idempotent); different-value is compile-error. | ✓ Q1 formalization holds. |
| Lattice-merge (e.g. min/max over provenance + value) | Depends on lattice structure; for parameters where values don't form a meaningful lattice (e.g., enum choices), this doesn't apply. Special case of outermost-by-provenance for lattice-typed slots. | Possible specialization; doesn't replace the general rule. |

**Conclusion**: Q1's formalization is **sufficient if and only if Condition C is
resolved by a path-independent rule**. The doc currently implicitly
assumes some such rule exists; the substrate note in Q3 (DEC-020 +
`using_annotation.rs:5-6`) hints at the answer but doesn't formalize
it. **Q3 closes the gap.**

### 1.3 What additional formalization the doc could add (recommended)

The single sentence I'd add, after the existing "binding order is
invisible to the cache key" claim, is something like:

> Conflict resolution is itself path-independent: when multiple binding
> sites assign the same parameter slot, the binding bag is reduced via
> the provenance lattice (Default < TamOverride < UserOverride per
> DEC-020) before serialization. Within a provenance tier, same-value
> rebinding is idempotent (no-op); different-value rebinding within a
> tier is a compile-time error. This means cache_key depends on
> {(slot, final_value)} — a content-addressable bag — regardless of
> how many times each slot was bound or in which order.

That sentence anchors the formalization. The full conflict-resolution
spec lives in §3 below as the answer to Q3.

### 1.4 Other possible objections to sufficiency, briefly addressed

**Composition under recipe nesting** (raised as Q3-bullet in the
original doc — also relevant here): if recipe A internally calls
recipe B with `using(p=X)`, and an outer caller binds `using(p=Y)` on
the composite call to A, which value does B see? Per outermost-by-
provenance: the outer caller wins *within their provenance tier*.
Specifically, if A's internal binding is `Default` (its
recipe-best-practice) and the outer caller's binding is `UserOverride`,
the outer wins. If A's internal binding is `UserOverride` (someone
reached inside A and explicitly forced p=X), and the outer is also
`UserOverride`, that's a compile-time error: *two user overrides for
the same parameter slot in the same composition is unresolvable*. This
is the Q3 substrate-note answer made explicit, and it preserves
holonomicity through composition.

**Manifold language** (Q2 in the original doc, not assigned to me but
worth noting for Q1's framing): the "configuration space is a manifold"
language is a useful metaphor but slightly loose mathematically. A
sharper formalization is "the recipe tier's parameter assignments form
a finitely-presented commutative monoid M under bag-merge, and
cache_key : M → BLAKE3 is a monoid homomorphism." The "manifold" image
captures the path-independence intuition (the configuration moves on
a structure where only the position matters); the "monoid" image
captures the algebraic condition that lets cache_key be a function. Both
are correct at different levels of abstraction. For the architectural
claim, monoid is the load-bearing structure; manifold is the
intuition. Either way, Q1's path-independence test detects deviation
from this structure correctly.

---

## 2. Q3 — `using()` conflict resolution

### 2.1 The substrate as it currently exists

DEC-020 (state conservation) + `using_annotation.rs` (Provenance enum)
together encode:

- Three provenance tiers: `Default` (recipe's built-in best-practice)
  < `TamOverride { evidence }` (TAM compile-time analysis chose this)
  < `UserOverride { tam_counterfactual }` (user explicitly typed it).
- **Provenance does NOT enter the cache key** (line 5-6 of using_annotation.rs).
  Same value → same compiled kernel, regardless of who set it. This
  is the navigator's substrate observation that closes part of Q3.

What the substrate **does not** explicitly address: the case where two
binding sites assign the same slot with **different values** within
the same composition, and what the resolution rule is.

### 2.2 The principled resolution rule

**Outermost-caller-wins per provenance tier, with idempotent
same-value rebinding and compile-time errors on different-value
intra-tier conflicts.**

Concretely:

#### Rule R1 — Inter-tier resolution
A higher provenance tier wins over a lower one. The order is:
`Default < TamOverride < UserOverride`. This is path-independent
because the tiers are properties of the binding's source, not its
emission order.

> Example: recipe `mean()` has `using(method = "naive")` as
> `Default`. TAM analyzes the input and binds `using(method =
> "kahan_compensated", evidence = "n > 1e6 triggers stability")` as
> `TamOverride`. The user binds nothing, or binds `method = "kahan"`
> explicitly. Result: TamOverride wins over Default; UserOverride
> wins over both. Same outcome regardless of binding order.

#### Rule R2 — Intra-tier idempotency
Within the same provenance tier, **same-value rebinding is a no-op**.
The bag {(p, "kahan"), (p, "kahan")} reduces to {(p, "kahan")}. This
is the standard set-merge semantics; no surprise.

#### Rule R3 — Intra-tier different-value conflict is a compile-time error
Within the same provenance tier, **different-value rebinding is
unresolvable** and must surface as a compile-time error at pipeline
construction. The bag {(p, "kahan"), (p, "chan")} both at
`UserOverride` cannot be reduced — the user has expressed two
conflicting intents and the system cannot decide for them.

> Example: pipeline P composes recipe A and recipe B; A internally
> binds `using(precision = 200, UserOverride)`, B internally binds
> `using(precision = 500, UserOverride)`, and A and B share a common
> downstream recipe C that consumes `precision`. The system cannot
> decide whether C runs at p=200 or p=500. Compile-time error.

This is the F13 antibody for the using() surface: a rule with a
scope precondition (the "no two user overrides bind the same slot
differently") that surfaces at construction time, not as silent
last-write-wins. Per F13.C graduation condition: the antibody is
required at *every* binding site, not just at the public API boundary.
A user-override binding inside an internal call site needs the same
conflict-detection as one at the top of the pipeline.

#### Rule R4 — Composition through nesting
When recipe A internally calls recipe B with `using(p = X)`, the
binding is provenance-tagged with A's tier. If A's tier is `Default`,
then any outer override (TamOverride or UserOverride) wins. If A's
tier is `UserOverride` (the user reached into A and forced X), then
an outer `UserOverride` produces an intra-tier conflict, surfacing
as a compile-time error per Rule R3.

This means **binding order through call stack depth doesn't matter;
provenance tier does**. Path-independence holds because provenance
is a property of the binding's *source*, not its *position in the
call stack*.

### 2.3 Why this is path-independent (formal)

Define the binding bag `B` as a multiset of `(slot, value, provenance)`
triples. Define the reduction `B → B'` by:

1. Group `B` by slot.
2. Within each group, pick the maximal provenance tier (Default < TamOverride < UserOverride).
3. If multiple bindings tie at the maximal tier:
   a. If all values in the maximal tier are equal, reduce to a single binding (idempotent).
   b. If values differ, abort with a compile-time error (unresolvable conflict).
4. The resulting `B'` is a bag of `(slot, value)` pairs (provenance
   stripped per DEC-020).

`cache_key : B' → BLAKE3` is a monoid homomorphism over bag-equality.
Bag-equality is permutation-invariant by construction. Therefore
`cache_key ∘ reduce` is permutation-invariant, and the recipe tier is
holonomic.

### 2.4 What this rules out

Three resolution semantics that would break holonomicity, explicitly
ruled out:

| Anti-pattern | Why it breaks |
|--------------|---------------|
| Latest-wins (sequential override) | Path-dependent: `bind(v_1); bind(v_2)` → v_2; reverse → v_1. |
| Scoped-binding with shadowing (lexical-scope-style) | Path-dependent: depends on which scope the cache_key is evaluated in. |
| Inferred-from-context (system picks based on which recipe called which first) | Definitionally path-dependent. |

These are the conventional resolution rules in many languages
(Python's mutable defaults, JavaScript's `Object.assign`, etc.). They
are **not** what tambear's `using()` does. The architectural commitment
is: the recipe tier's `using()` semantics are designed to be
content-addressable, and the conflict-resolution rule preserves that
by being path-independent itself.

### 2.5 Substrate verification — what's already in the code

**Already correct** (per `using_annotation.rs:1-90`):
- Three-tier Provenance enum exists.
- DEC-020 sub-clause: "Provenance does NOT enter the cache key" — same
  value → same kernel.
- The `is_user_override`, `is_tam_override` predicates exist for the
  conflict detection logic.

**Status of conflict detection** (substrate observation, 2026-05-09):
the conflict-detection logic itself (Rule R3 — error on intra-tier
different-value) does not yet appear to be implemented as a
compile-time pipeline check. There's no grep hit for "intra-tier
conflict" or "using_conflict" in the current crate. **This is a
shippable F13 antibody that doesn't exist yet** — adding it would
graduate the Provenance enum from "tracking what" to "enforcing
that".

Recommendation: file as a separate task to add conflict detection at
pipeline-construction time. The path-independence guarantee depends
on it.

---

## 3. Implications for the holonomic-architecture doc

Three concrete edits suggested for the holonomic-architecture.md
update (team-lead applies after navigator review):

### Edit 1 — Add the conflict-resolution sentence to §"Cache keys as the operationalization"

Add after the existing "binding order is invisible" paragraph:

> **Path-independence requires path-independent conflict resolution.**
> When multiple binding sites assign the same parameter slot, the
> binding bag is reduced via the provenance lattice
> (`Default < TamOverride < UserOverride` per DEC-020) before
> serialization. Within a provenance tier, same-value rebinding is
> idempotent; different-value rebinding is a compile-time error.
> The cache key depends on `{(slot, final_value)}` — a content-
> addressable bag — regardless of how many times each slot was bound
> or in which order. The provenance tier is itself a property of the
> binding's source (recipe-default vs TAM-analysis vs user-typed), not
> of its position in the call stack, so composition through nesting
> preserves holonomicity.

### Edit 2 — Replace Q1 with the answer

In §"Open questions for math-researcher walk-through":

Q1 (current):
> **Path-independence formalization.** Is "binding-order permutation
> produces same cache key" a sufficient formalization of holonomicity
> at the recipe tier, or does it need additional conditions (e.g.,
> interaction with `using()` precedence, or composition under recipe
> nesting)?

Q1 (resolved, replace with):
> **Path-independence formalization** — RESOLVED 2026-05-09.
> Sufficient *iff* `using()` conflict resolution is itself
> path-independent (Q3). With the outermost-by-provenance rule per
> DEC-020 + intra-tier idempotency + compile-time error on
> intra-tier conflict, the formalization is complete. See
> `R:\winrapids\campsites\tambear-sweep31-finish\math-researcher\holonomic-q1-q3-walkthrough-2026-05-09.md`.

### Edit 3 — Replace Q3 with the answer

Q3 (current substrate-note version): the binding-order question reduces
to whether the final value is path-independent.

Q3 (resolved, replace with):
> **`using()` conflict resolution** — RESOLVED 2026-05-09.
> The principled rule: `using()` bindings form an idempotent
> commutative monoid under value-equality; conflict resolution is
> outermost-caller-wins per provenance tier
> (`Default < TamOverride < UserOverride`); intra-tier same-value
> rebinding is idempotent; intra-tier different-value rebinding is a
> compile-time error. Substrate observation: the Provenance enum
> exists in `using_annotation.rs`; the conflict-detection logic
> (Rule R3) does not yet appear to be implemented. Filed as
> follow-up. See walk-through doc.

---

## 4. Follow-ups created by this walk-through

1. **Conflict-detection task** — implement Rule R3 (compile-time error
   on intra-tier different-value `using()` rebinding) at pipeline
   construction time. Without this, the path-independence guarantee
   is structural-but-not-enforced. F13.C graduation condition: this
   antibody must exist at every binding site.

2. **Q5 hint** — the original doc's Q5 ("are there other instances the
   team has been operationalizing without naming?") may surface another
   one through this walk-through: **the Provenance enum itself** is
   another holonomic instance — a constant ("the binding's source") was
   parameterized into a tier (Default / TamOverride / UserOverride),
   and that tier is now path-independent. Five convergence instances,
   not four. Worth checking whether naturalist already counted this.

3. **Q2 partial answer** — the manifold formalization is loose; the
   sharper formal structure is **finitely-presented commutative
   monoid under bag-merge, with cache_key as monoid homomorphism**.
   This is already enough for the architectural claim and for proptest-
   style binding-order-permutation tests. The manifold language can
   stay as intuition.

---

## 5. Provenance

- Authored 2026-05-09 by math-researcher in team `tambear-sweep31-finish`,
  per navigator's directive after the cos+tan oracle work.
- Substrate verified: `holonomic-architecture.md` (full read);
  `using_annotation.rs` (full read of pub API, lines 1-90); DEC-020 in
  `R:\tambear\docs\decisions.md` lines 1191-1240; `f13-antibodies-for-scope-precondition-rules.md`
  (relevant graduation-condition sections); `important-conversation.md`
  lines 1056-1063 (cliffhanger).
- Cross-checked: the path-independence test as currently formalized is
  *necessary*; sufficiency requires conflict-resolution path-independence,
  which is the principled-but-not-yet-fully-implemented answer in §2.
  Q1 and Q3 are co-determined; resolving them together is sharper than
  resolving them sequentially.
- This is a draft for navigator review. Team-lead applies the three
  suggested edits to `holonomic-architecture.md` after ratification;
  CLAUDE.md promotion of the holonomic principle to Irrevocable
  Architectural Principles follows.
- The remaining open questions in the doc (Q2 manifold/groupoid,
  Q4 partial holonomic structure at IR layer, Q5 four-instances
  counting, Q6 test as antibody/lint) are not addressed by this
  walk-through; happy to take them on next session if useful.
