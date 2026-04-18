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

# Verification Oracle as Infrastructure

*Written by: researcher agent (julia-matlab-scan role), 2026-04-10*

## The Problem

The current oracle coverage is ~5% — roughly 15-20 primitives out of ~300 public functions have 
genuine oracle or workup-grade tests. The adversarial waves cover edge cases well, but mostly 
assert structural properties (NaN propagation, sign correctness, boundary behavior), not 
mathematical theorems.

The gap is invisible. "1390 tests green" feels like correctness. The coverage map shows 95% 
unclaimed territory, but building that map required a grep expedition and manual file reading. 
No one would do that routinely.

## The Insight

Make the coverage map first-class infrastructure. The map should be live and machine-readable, 
not a document built by hand on demand.

Every primitive registers its coverage claims at creation time. The test framework reports 
oracle and adversarial coverage percentages, not just pass/fail counts. The uncovered territory 
is visible without anyone needing to remember to look.

## Proposed Design

### 1. A `#[covers]` attribute on test functions

```rust
#[test]
#[covers("erfc", theorem = "erfc(-x) = 2 - erfc(x)")]
fn erfc_symmetry() {
    let x = 1.5;
    let lhs = erfc(-x);
    let rhs = 2.0 - erfc(x);
    assert!((lhs - rhs).abs() < 1e-14);
}
```

A proc-macro collects all `#[covers]` annotations at build time and generates a coverage 
report. The attribute links a test function to the theorem it's asserting.

### 2. A manifest file per primitive

In each module's doc comment or a companion TOML, each public function declares its theorems:

```toml
[primitives.erfc]
theorems = [
    "erfc(0) = 1",
    "erfc(-x) = 2 - erfc(x)", 
    "erfc(∞) = 0",
    "erf(x) + erfc(x) = 1",
]
adversarial_cases = ["x=0", "x<0", "x=large_positive", "x=NaN", "x=Inf", "x=-Inf"]

[primitives.digamma]
theorems = [
    "ψ(x+1) = ψ(x) + 1/x",
    "ψ(1-x) - ψ(x) = π·cot(πx)",
    "ψ(1) = -γ (Euler-Mascheroni)",
]
adversarial_cases = ["x=0 (pole)", "x=-1 (pole)", "x=-n for integer n", "x very large"]
# Status: 0/3 theorems covered, 0/4 adversarial cases covered
```

### 3. Coverage report in CI

```
Verification Coverage Report — 2026-04-10
==========================================
Oracle coverage:   15 / 300 primitives  (5%)
Adversarial:       89 / 300 primitives  (30%)

Uncovered (oracle):
  special_functions: digamma, trigamma, gamma, log_gamma [via series boundary]
  linear_algebra:    matrix_exp, matrix_log, matrix_sqrt
  information_theory: mutual_information, entropy, kl_divergence
  ... (285 more)

High-priority uncovered (used by > 10 other primitives):
  regularized_incomplete_gamma  [dependency: chi2_cdf, gamma_cdf, poisson_cdf, ...]
  digamma                       [dependency: trigamma, polygamma, shapiro_wilk_coeffs, ...]
```

The "used by > 10 other primitives" section is the key new signal. An uncovered primitive that 
20 methods depend on is a different kind of risk than an uncovered leaf primitive. The manifest 
enables this dependency-weighted prioritization automatically.

## Theorem Taxonomy

Theorems have a natural hierarchy that determines what each test catches:

| Category | Example | Error class caught |
|---|---|---|
| Definition | `erfc(0) = 1` | Wrong formula entirely |
| Symmetry | `erfc(-x) = 2 - erfc(x)` | Sign errors |
| Recurrence | `ψ(x+1) = ψ(x) + 1/x` | Off-by-one, recursion bugs |
| Composition | `erf(x) + erfc(x) = 1` | Interaction between related functions |
| Limiting | `erfc(∞) = 0` | Asymptotic behavior |
| Inversion | `erfinv(erf(x)) = x` | Inverse pair consistency |

A primitive with only a definition test is partially covered. Adding symmetry + recurrence + 
composition tests closes the main error classes systematically rather than by hoping the 
adversarial cases happen to find the right bugs.

## The Smallest Useful Version

The minimal implementation that changes the situation:

1. A `verification_manifest.toml` at the crate root listing every public primitive with its 
   theorem list (initially empty for uncovered ones)
2. A `cargo verify-coverage` command that reads the manifest and test attributes, reports 
   the gap
3. `#[covers]` attribute added to existing workup tests retroactively

This doesn't require a proc-macro for the first version — the manifest can be maintained 
manually with CI enforcing that new primitives are added. The enforcement is: every new pub fn 
must appear in the manifest (empty theorems allowed), and the coverage report is visible in CI.

## Priority

This is infrastructure, not math. The math work is more urgent.

But: every new wave of adversarial tests, every new workup file, is harder to place correctly 
without the map. The invisible coverage gap means effort gets applied where it's most visible 
(the functions that have already been worked up) rather than where it's most needed (the 
functions that depend on the most downstream consumers).

The oracle-coverage-map campsite that observer proposed would capture this as an immediate 
deliverable. This document is the design spec for what that campsite should produce.

## Connection to Other Insights

- **Observer**: "what territory is still unclaimed" — this is the infrastructure that answers 
  that question automatically
- **Adversarial validity-semantics**: the manifest would also track which NaN policy each 
  function declares, making implicit policies visible
- **Three-condition Kingdom theorem**: a primitive's kingdom classification belongs in the 
  manifest too — trackable, auditable, updatable when the algebra changes


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

