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
