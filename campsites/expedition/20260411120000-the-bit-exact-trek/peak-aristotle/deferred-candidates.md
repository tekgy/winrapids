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

# Deferred Candidates

Ideas that Aristotle surfaced during deconstruction or lateral peer review, which the team evaluated and chose to defer rather than adopt now. Each entry records the candidate, the reason for deferral, and the concrete trigger condition that would make it worth revisiting.

The point of this file is to keep a stable record across sessions so that future Aristotle work (or future team work on related topics) can cite these without reinventing the reasoning.

---

## 1. `nan_propagating` as a fourth NamedConstraint variant

**Surfaced:** 2026-04-12, Aristotle → scientist, after campsite 4.8 announcement.
**Status:** Deferred. Redundant by coverage against existing injection_sets mechanism.

### Candidate

A fourth `NamedConstraint` variant in `oracle_runner.rs` named `nan_propagating`, asserting "for any NaN input, the output is NaN," usable as:

```toml
[[constraint_checks.cases]]
name = "tam_exp_nan_propagation"
constraint = "nan_propagating"
inputs = ["nan", "snan", "-nan"]
```

Motivation: I11 ("NaN propagates through every op on every backend," added 2026-04-12) is an invariant every libm function must satisfy. A constraint variant would let the test be one line per function and would make the pattern structurally discoverable via the registry.

### Why deferred

Scientist's reasoning (verbatim, 2026-04-12):

> The injection set path in `run_oracle` builds synthetic records. For NaN inputs not in the reference binary, the synthetic reference is `candidate(NaN)`. `ulp_distance_with_special` then applies the NaN rule: if `reference.is_nan()` and candidate is not NaN → special_value_failure. So a `special_values` injection set containing `"nan"` already enforces I11 via the `special_value_failures` counter, which is a first-class field in `UlpReport` and is checked in `OracleReport::passes` (`injections_ok = injection_reports.iter().all(|r| r.special_value_failures == 0)`).
>
> The injection set setup per function is one line:
>   `special_values = ["nan", "inf", "-inf", 0.0, -0.0]`
>
> That's already as lightweight as the proposed `[[constraint_checks.cases]]` entry. Adding `nan_propagating` as a `ConstraintCheck` variant would create a second enforcement path with no additional coverage — same assertion, different accounting.

I11 is already enforced by convention at the oracle-file authoring level. The `special_values` one-liner IS the test; the runner's `special_value_failures` counter IS the failure flag; `passes` already blocks on it.

### Trigger condition for revisiting

**When the oracle registry grows to 5+ libm functions AND a reviewer finds a function whose author forgot to include `"nan"` in `special_values`.** At that point the convention has become fragile — reviewers can't be trusted to catch every omission by eye, and I11 compliance becomes a structural question that deserves explicit declaration.

Concretely, one of the following should trigger the reconsideration:
1. A new libm function PR lands without `special_values = ["nan", ...]` and the I11 enforcement silently skips that function. Any test-level or review-level observation that this happened.
2. Oracle registry hits 5+ entries and team starts discussing "how do we enforce I11 across all of them" at review-time.
3. Adversarial finds a NaN-propagation bug in a libm function whose oracle file was missing `special_values`.

Any of these triggers escalation to reconsider the deferred candidate.

### What the coverage-vs-policy distinction teaches

Scientist's answer drew a distinction worth naming for the `named-artifacts-convention.md` draft:

**An invariant can be enforced by convention or by explicit declaration.**

- **Convention enforcement** (current state): the author of each oracle file knows to include `special_values = ["nan", ...]`. Lightweight. Works at small scale. Fragile at large scale — missing entries silently skip the test.
- **Explicit declaration** (the `nan_propagating` candidate): a named registry entry lets each oracle file declare "this function conforms to I11" as a structural property. Heavyweight. Overkill at small scale. Robust at large scale — a missing declaration is a PR-review-visible omission.

The transition point is when convention fails: when reviewers stop catching omissions, or when the registry is large enough that convention can't be held in one reviewer's head. That's the same boundary as when other tacit knowledge gets externalized into a named artifact.

**Observation for the convention doc:** a Named Architectural Artifact is the end-state of a convention that grew fragile. The pattern for introducing a new artifact is: (1) identify the invariant/decision being made implicitly, (2) measure the cost of an undetected omission, (3) if the cost is high AND the invariant can be named, promote convention → explicit declaration. The NamedConstraint registry already contains three entries that graduated from convention (`nonzero_subnormal_positive`, `finite`, `infinite_positive`). `nan_propagating` is a candidate that HASN'T graduated yet because the convention isn't fragile enough.

---

*Future deferred candidates from Aristotle work will be added below this line, in similar format.*


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

