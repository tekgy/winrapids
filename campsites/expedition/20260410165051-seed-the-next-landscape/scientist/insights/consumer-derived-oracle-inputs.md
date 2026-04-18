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

# Consumer-Derived Oracle Inputs

**The core principle**: the oracle inputs for a primitive are not determined by the
primitive's own mathematical domain — they are determined by the inputs that are
statistically or scientifically meaningful to the primitive's consumers.

---

## The problem

A primitive oracle suite tests the primitive at mathematically interesting inputs:
critical points, edge cases, known closed-form values, adversarial inputs. These
are chosen by someone who understands the primitive's internal structure.

But the primitive has no domain knowledge. `erfc` doesn't know that statisticians
use the normal distribution. It doesn't know that x=1.96 is the 95% critical value.
It doesn't know that x=1.386 = 1.96/√2 is the argument it receives when a
hypothesis test asks "is this observation significant at the 5% level?"

The domain knowledge lives in the CONSUMER, not the primitive. The oracle that
would catch a domain-significant bug must be written from the consumer's perspective.

## The erfc case

The bug: Taylor series at |x| ∈ [1.0, 1.5) accumulated 82 ULP error at x=1.386.

The erfc workup tested x = 0.1, 0.5, 1.0, 2.0, 5.0 — mathematically natural inputs.
None of them were near x=1.386.

The normal_cdf workup tested Φ(-1.96) = 0.025 — the 95% two-sided critical value.
This required erfc(1.96/√2) = erfc(1.386). That test FAILED, revealing the bug.

The domain knowledge that made x=1.386 a load-bearing input came from statistics,
not from erfc's mathematical structure. erfc's authors think about convergence
regions and series expansions. normal_cdf's authors think about critical values and
hypothesis tests.

## The principle, formally

For a primitive P with consumers C_1, C_2, ..., C_k:

For each consumer C_i:
1. Identify the inputs to C_i that are statistically/scientifically meaningful
   (critical values, known calibration points, standard reference inputs)
2. Trace those inputs through C_i's arithmetic to find which values reach P
3. Include those values in P's oracle suite

The oracle suite for P is the UNION of consumer-critical inputs across all C_i.

## The inversion step

The key operation: inverting through the consumer's arithmetic.

erfc example:
- Consumer: normal_cdf(x) = 0.5 · erfc(-x/√2)
- Meaningful input to consumer: x = -1.96 (the 95% critical value)
- Inversion: x_erfc = -x_cdf / √2 = 1.96/√2 = 1.386
- Oracle input for erfc: x = 1.386

When the inversion has a closed form, compute it. When it doesn't, run the consumer
at its meaningful inputs empirically and record what values reach the primitive.

## The naming

Closest existing concepts: property-based testing, metamorphic testing, boundary
value analysis, use-case testing.

None of these have the inversion step. Use-case testing is closest ("use what the
user actually uses") but treats "user" as a human, not a downstream function.

Proposed name: **consumer-derived oracle inputs**.
- "consumer": the downstream function, not the human
- "derived": the inputs are computed by inverting through the consumer's arithmetic,
  not taken directly from the consumer's domain
- "oracle": these are ground-truth inputs that must produce correct outputs

## The workup template addition

Every workup should include a section:

```
## Consumer-derived oracle inputs

For each consumer of this primitive:
- Consumer: [function name]
- Meaningful inputs to consumer: [list with domain justification]
- Inversion: [how to get from consumer input to primitive input]
- Oracle inputs for this primitive: [computed values]
- Tests: [test names in the suite that cover these inputs]
```

## The direction symmetry

This principle works in TWO directions:

**Forward (discovery)**: consumer workup exercises consumer with meaningful inputs,
revealing which primitive inputs are load-bearing. The erfc bug was found this way.

**Backward (regression)**: upstream change must re-run all downstream workups.
When erfc's boundary changed from 0.5 to 1.5, the normal_cdf workup should have
been re-run. Fix 1 introduced the second bug. Fix 2 was found by re-running the
downstream suite.

The workup chain is a regression test for the primitive. The chain tests what the
node can't know about itself.

## Generalization: the sharing topology determines test priority

The sharing topology (consumer dependency graph) tells you which primitives have
the most consumers. High fan-out = high leverage for testing.

But fan-out is necessary, not sufficient. What matters is whether any downstream
consumer has domain knowledge about critical inputs that the primitive lacks.

High-priority oracle additions:
- erfc: x from normal_cdf critical values (done — x=1.386 now in workup)
- log_gamma: x from Bayesian conjugate prior parameters (Dirichlet, Gamma MLE)
- regularized_incomplete_beta: a, b from t-test and F-test effect sizes
- SVD: matrices from factor analysis and PCA benchmark datasets

## The worked example: erfc chain

```
"95% CI" 
  → Φ(-1.96) = 0.025            [normal_cdf meaningful input]
  → normal_cdf(-1.96)            [consumer call]
  → erfc(1.96/√2) = erfc(1.386) [primitive reached]
  → oracle test: erfc(1.386) matches mpmath to ≤5 ULP
```

The test lives in `workup_normal_cdf.rs`, not `workup_erfc.rs`. That's the correct
location — the domain knowledge belongs to normal_cdf, not erfc. But the assertion
constrains erfc's correctness at a domain-critical input.

The patch: `workup_erfc.rs` should ALSO include x=1.386 with a comment:
"Consumer-derived: erfc(1.96/√2) is load-bearing for normal_cdf at the 95% CI
critical value. Included explicitly to prevent regression to any crossover boundary
that doesn't cover this point."


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

