# Publication-Grade Workup Template

**Author**: scientist
**Status**: template — fill in one per Layer 0 primitive
**Principle**: Tambear Contract Principle 10 — every primitive gets the Nature-paper workup

---

## Purpose

Every Layer 0 primitive in tambear is a piece of mathematical infrastructure. When a researcher, regulator, or auditor asks "how do I know this is correct?", the answer must be a single document they can read that shows:

1. The mathematical definition and its citations
2. The implementation choices we made, and why
3. The numerical behavior at every scale we expect the primitive to run at
4. The comparison against every competing implementation (scipy, R, MATLAB, Julia, Stata, SAS, etc.)
5. The adversarial inputs we checked and the behavior we guarantee
6. The bugs we found (in our code or in theirs)

This document is that answer. It is the primitive's **workup**. It lives at
`docs/research/workups/<family>/<primitive>.md` and is versioned alongside the
implementation. It is updated whenever the primitive is updated.

The template below is the minimum structure. Every section is load-bearing. A
workup without Section 6 (competitor comparison at extended precision) is not
complete. A workup without Section 9 (benchmarks at billion-row scale, or a
stated reason we cannot reach that scale) is not complete.

---

## Template

```markdown
# Workup: `<module::function_name>`

**Family**: <e.g., nonparametric / hypothesis / time_series / descriptive>
**Status**: draft | verified | signed-off
**Author**: <agent name>
**Last updated**: <ISO date>
**Module**: `crates/tambear/src/<module>.rs`
**Function signature**: `pub fn <function_name>(<params>) -> <ReturnType>`

---

## 1. Mathematical definition

### 1.1 The quantity computed
<One sentence: "This primitive computes X given Y under assumption Z.">

### 1.2 Canonical definition
<Equations, typeset in LaTeX-style. Cite the paper(s) that define the quantity.>

### 1.3 Assumptions
<Bullet list. What does the mathematical object require of its input? What
is NOT assumed? Which assumptions are checked at runtime vs documented as
caller contract?>

### 1.4 Kingdom declaration
<Kingdom A (commutative pointwise), B (sequential), C (iterative fixed-point),
or BC/AB mixed. This is what TAM needs to know for scheduling.>

### 1.5 Accumulate+gather decomposition
<Express the computation in tambear's primitive operations. If the quantity
does not decompose cleanly, document why — e.g., "Shapiro-Wilk's W requires
order statistics, which force a sort before the accumulate. The sort is
O(n log n) Kingdom B; the W computation itself is O(n) Kingdom A on the
sorted data.">

---

## 2. References

Every claim in Section 1 has a citation here.

- [1] Author (Year). *Title*. Journal Volume:Pages.
- [2] ...

At minimum: the original paper, the most-cited algorithm paper, and (if the
implementation follows a published algorithm) the algorithm paper.

---

## 3. Implementation notes

### 3.1 Algorithm chosen
<Which of the multiple known algorithms did we implement, and why? e.g.,
"Royston (1992) uses AS R94 coefficients for the W statistic; we implement
AS R94 directly because it is bit-reproducible and matches the R `shapiro.test`
reference.">

### 3.2 Numerical stability
<What numerical hazards exist, and how do we avoid them? e.g., catastrophic
cancellation, overflow in intermediate products, loss of significance in
tail computations. Document the defensive choices.>

### 3.3 Parameters
<Every parameter in the signature. For each: type, valid range, default (if
any), what happens at the boundary, what paper the default comes from, what
`using()` override is the canonical one.>

| Parameter | Type | Valid range | Default | Reference |
|-----------|------|-------------|---------|-----------|
| ... | ... | ... | ... | ... |

### 3.4 Input validation
<What inputs are rejected (return NaN/None/sentinel vs panic)? What is
degenerate behavior — constant input, single value, all NaN, negative where
positive is required, etc.>

### 3.5 Shareable intermediates
<Which intermediates does this primitive produce that TamSession should
cache? Which intermediates does it consume? What is the IntermediateTag
variant used? What compatibility tags (sorted order, centered moments,
fourier basis) are required?>

---

## 4. Unit tests

Inline `#[cfg(test)]` tests covering:

- Known-answer cases hand-computed from the definition
- Edge cases: empty, single-element, two-element, constant, all-NaN,
  mixed NaN, ±∞, subnormal
- Degenerate inputs that trigger the "documented sentinel" return path
- Invariance properties (shift-invariance, scale-invariance, permutation-
  invariance, etc.)
- Monotonicity properties where they hold
- Bounds: "result is always in [−1, 1]", "result is non-negative", etc.

Checklist:
- [ ] Known-answer from paper Table X
- [ ] Empty input
- [ ] n=1, n=2
- [ ] Constant input
- [ ] NaN handling
- [ ] ±∞ handling
- [ ] Subnormal handling
- [ ] Shift invariance (if applicable)
- [ ] Scale invariance (if applicable)
- [ ] Range bounds

---

## 5. Oracle tests — against extended precision

Compare against **mpmath with 50 decimal digits** as the ground truth. Only
after we agree with mpmath to 1e-14 do we ask whether competitors agree.

### 5.1 Oracle script location
`research/gold_standard/<family>_<primitive>_oracle.py`

### 5.2 Test cases
<List the input configurations the oracle exercises. For each: size, shape,
parameter values, expected output, achieved output, absolute error, relative
error.>

| Case | n | Parameters | mpmath truth | tambear | rel err |
|------|---|------------|--------------|---------|---------|
| ... | ... | ... | ... | ... | ... |

### 5.3 Maximum observed relative error
`<e.g., 1.2e-15 — machine precision>`

---

## 6. Cross-library comparison

The headline of the workup. We compare against **every** major implementation
we can find. Each competitor is treated as a peer, not as ground truth.

### 6.1 Competitors tested

| Library | Version | Function | Agrees with mpmath? | Agrees with tambear? |
|---------|---------|----------|---------------------|----------------------|
| scipy | X.Y.Z | `scipy.stats.<fn>` | yes/no (err = ...) | yes/no (err = ...) |
| R `stats` | X.Y.Z | `<fn>()` | yes/no (err = ...) | yes/no (err = ...) |
| R `<pkg>` | X.Y.Z | `<pkg>::<fn>()` | yes/no | yes/no |
| MATLAB | RXXXXx | `<fn>()` | yes/no | yes/no |
| Julia `<pkg>` | X.Y.Z | `<fn>()` | yes/no | yes/no |
| statsmodels | X.Y.Z | `<fn>()` | yes/no | yes/no |
| Stata | X | `<fn>` | yes/no | yes/no |
| SAS | X | `PROC <fn>` | yes/no | yes/no |

### 6.2 Discrepancies found

For every row where "Agrees with mpmath" is "no":

- **Library X, function Y**: expected `<mpmath value>`, got `<library value>`,
  relative error `<err>`. Investigation: `<what is the bug>`. Filed upstream
  as `<issue URL>`.

For every row where "Agrees with tambear" is "no" but "Agrees with mpmath" is
"yes":

- **Library X**: this is our bug, not theirs. See Section 10.

### 6.3 Verdict

One of:
- **bit-perfect** against all competitors that agree with mpmath
- **more accurate** than competitor X because ...
- **different convention** from competitor X because ... (e.g., ddof, tie
  handling, edge-case definition)
- **bug upstream** in competitor X, documented and filed

---

## 7. Adversarial inputs

Inputs designed to break the implementation. The test suite lives in
`crates/tambear/tests/adversarial_<module>.rs`.

Checklist (tailor to the primitive):

- [ ] Catastrophic cancellation input (large mean, tiny variance)
- [ ] Overflow-prone input (values near f64::MAX)
- [ ] Underflow-prone input (values near f64::MIN_POSITIVE)
- [ ] Ill-conditioned input (collinear design matrix, near-singular matrix,
      etc.)
- [ ] Pathological structure (heavy tails, bimodality, structural breaks)
- [ ] Boundary cases (n exactly at minimum, n = minimum+1)
- [ ] All-identical input
- [ ] Pairwise-identical input (after sort)
- [ ] Very large n (10⁶, 10⁷, 10⁸) — verifies the primitive does not blow
      memory or time
- [ ] Negative/zero where positive is required
- [ ] Integer overflow paths (count_inversions on random permutations of
      1e6 elements must not overflow u64 — n*(n-1)/2 = ~5e11 < 2^63)
- [ ] NaN injection (single NaN, all NaN, interleaved NaN)
- [ ] ±∞ injection
- [ ] Subnormal injection

---

## 8. Invariants and proofs

Some primitives have algebraic invariants that should be asserted in the
test suite. E.g.:

- `variance(shift(x, c)) == variance(x)` — shift invariance
- `variance(scale(x, c)) == c² * variance(x)` — scale property
- `f(rev(x)) == f(x)` for permutation-invariant f
- `softmax(x + c) == softmax(x)` for any scalar c
- `log_gamma(x + 1) - log_gamma(x) == log(x)` — functional equation

List every invariant the primitive should satisfy, and the test that
enforces it.

---

## 9. Benchmarks and scale ladder

Every primitive is benchmarked at every scale up to the largest we can
physically run.

### 9.1 Scale ladder

| n | CPU time (µs) | GPU time (µs) | Memory peak (MB) | Max |err vs mpmath| |
|---|---|---|---|---|
| 10¹ | ... | ... | ... | ... |
| 10² | ... | ... | ... | ... |
| 10³ | ... | ... | ... | ... |
| 10⁴ | ... | ... | ... | ... |
| 10⁵ | ... | ... | ... | ... |
| 10⁶ | ... | ... | ... | ... |
| 10⁷ | ... | ... | ... | ... |
| 10⁸ | ... | ... | ... | ... |
| 10⁹ | ... | ... | ... | ... |
| 10¹² | — | ... | ... | ... |

### 9.2 Scaling verdict

Does the implementation scale as advertised? (O(n), O(n log n), O(n²), etc.)
Where does it deviate from theoretical complexity, and why?

### 9.3 Cache / memory behavior

What is the working set per n? Does it fit in L1/L2/L3/DRAM? When does it
spill?

### 9.4 Unreachable scales

If we cannot benchmark at 10⁹ or 10¹², document **why** (algorithm is O(n²),
working memory is O(n²), Kingdom B, etc.) and what the fallback is.

---

## 10. Known bugs / limitations / open questions

Everything we know that is wrong, unfinished, or uncertain about this
primitive. Each bug has a task ID in the task system.

- **#XX: <short description>** — severity, expected fix date, workaround.
- ...

If Section 10 is empty, every known issue is resolved. That should be rare.

---

## 11. Sign-off

- [ ] Section 1-3 written by author
- [ ] Unit tests in Section 4 passing
- [ ] Oracle tests in Section 5 at target precision (1e-14 or documented tolerance)
- [ ] Cross-library comparison in Section 6 complete for at least scipy and R
- [ ] Adversarial suite in Section 7 passing
- [ ] Benchmarks in Section 9 complete through n = 10⁶ at minimum
- [ ] Known bugs in Section 10 have task IDs or are empty
- [ ] Reviewed by scientist (verification of Sections 5, 6, 9)
- [ ] Reviewed by adversarial (verification of Section 7)
- [ ] Reviewed by math-researcher (verification of Section 1, 2, 3)

---

## Appendix A: reproduction artifacts

- Oracle script: `research/gold_standard/<family>_<primitive>_oracle.py`
- Adversarial test file: `crates/tambear/tests/adversarial_<module>.rs`
- Benchmark harness: `crates/tambear/benches/<primitive>.rs`
- Raw cross-library output (versioned): `docs/research/workups/<family>/<primitive>-raw-outputs.json`

## Appendix B: version history

| Date | Author | Change |
|------|--------|--------|
| ... | ... | ... |
```

---

## How to use this template

1. **One workup per Layer 0 primitive.** Not per family. Not per module. Per
   entry point. `spearman_correlation` gets its own workup. `kendall_tau`
   gets its own workup. Shared intermediates (e.g., rank computation) get
   their own workup too.

2. **Start with the skeleton.** Copy the template into
   `docs/research/workups/<family>/<primitive>.md` and fill in Sections 1,
   2, 3, 11 (unchecked). This is the "draft" state.

3. **Build out as you verify.** Each time you add a test, update a section.
   Each time you find a discrepancy, update Section 6 or Section 10.

4. **Never skip Section 6.** Cross-library comparison is the single most
   valuable thing a workup contains. It's where we discover bugs — ours or
   theirs — and where we earn the "publication-grade" label.

5. **Treat Section 10 as alive.** Known bugs are not shameful — unreported
   bugs are. Every primitive has limits; document them honestly.

6. **Sign-off is multi-role.** A workup is signed off only when scientist,
   adversarial, and math-researcher have all reviewed their sections. This
   is the quality gate for "verified" status.

---

## Why not just tests?

Unit tests say "this behaves correctly on these inputs." A workup says
"here is what this primitive is, here is what it promises, here is every
adversarial input we tried, here is every competing implementation we
checked, here is every scale we ran at, here is every bug we found, and
here is who verified each of those claims."

Tests are necessary; workups are sufficient. A primitive without a workup
cannot be promoted to Layer 1 auto-selection, cannot be wired into a
Layer 3 expert pipeline, and cannot be used as a reference for academic
publication.

---

## Relationship to existing docs

This template supersedes ad-hoc assumption docs like
`family-XX-<family>-assumptions.md`. Those are one doc per family, which
is too coarse — a family like `nonparametric` has 15 primitives, each
with different assumptions and different bugs. One workup per primitive
forces the precision we need.

The family-level assumption docs are still useful as indexes: "here are
the 15 primitives in nonparametric, here is where their workups live, here
is the status of each." A workup navigator.

---

## Suggested first workups

If we're rolling this out, the highest-leverage first workups are the
primitives where bugs are most likely and where upstream comparisons are
most informative:

1. **shapiro_wilk** — Task #79 and Task #90 are documented bugs.
   Comparison against scipy, R `stats::shapiro.test`, and the Royston
   AS R94 reference implementation is already desirable.

2. **cooks_distance** — Task #94 (hardcoded threshold). Comparison against
   statsmodels `OLSInfluence` reveals sign conventions and leverage
   definitions that differ between libraries.

3. **tukey_hsd** — Task #93 (hardcoded alpha). Comparison against R
   `TukeyHSD`, statsmodels `pairwise_tukeyhsd`, and Stata `pwmean`.

4. **kendall_tau** — scipy has a known bug history around tie handling.
   A workup here might find a scipy issue that has not yet been fixed.

5. **garch11_fit** — competing implementations disagree substantially.
   Comparison against Python `arch`, R `rugarch`, MATLAB Econometrics
   Toolbox, and Stata `arch` is a minor research project in itself.

Each of these is a 1-3 day effort for a full workup. Done right, each
produces either a clean sign-off or a published bug report.
