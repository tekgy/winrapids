# Autodiscover Probe Catalog for Trig

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-14
**Status**: Deliverable for TRIG-5. Builds on `first-principles.md`, `atoms_gaps.md`, `notation.md`, `hardware-mapping.md`.

---

## Context — what `.discover()` is and why trig has one

Per the memory index and the discover superposition doc: tambear's `.discover()` pattern forks a computation into parallel branches (superposition), evaluates all branches, reports what was discovered, and collapses to the winner. The cost is ~1× because branches share everything except the tiny differing leaf. The output is **scientific knowledge** — "your data is mildly hyperbolic (κ=-0.3), Poincaré overfits at full curvature" — not just a number.

For trig, `.discover()` looks like:

```tbs
sin(col=0).discover()              # tries every kernel/reduction/precision combo
angle_analyze(col=0).discover()    # tries to detect what unit the data is in
trig_identity_check(col=0).discover()  # verifies sin²+cos²=1 across methods, flags anomalies
```

**A probe is a parameterized question the library asks of the data.** Probes are the building blocks of `.discover()`. Each probe:
1. Has a well-defined input shape (usually a column + context).
2. Runs a specific computation and produces a scalar or small-vector output.
3. Has an interpretation — "a value of X in this probe means Y about the data."
4. Is cheap (single-pass, Kingdom A most of the time).
5. Can participate in superposition with peer probes answering the same question differently.

This doc catalogs the probes trig needs for auto-discovery across input properties, kernel choice, reduction choice, precision choice, and identity verification.

---

## Probe catalog structure

Each probe has:
- **Name** — canonical identifier (snake_case)
- **Question** — what the probe answers, in English
- **Input** — column shape, context
- **Output** — scalar, tuple, or small vector
- **Interpretation** — what the output means
- **Implementation** — how it's computed (reduce → kernel → extract usually)
- **Kingdom** — A/B/C per tambear classification
- **Cost** — rough ballpark relative to `sin(col)` cost (baseline = 1.0)

---

## Family 1 — Input property probes

These probes characterize the input column before choosing an algorithm. Output feeds dispatch logic.

### P1.1 — `input_magnitude_distribution`

**Question**: what is the range of `|x|` in the column? This determines which reduction path (Cody-Waite vs Payne-Hanek) will dominate.

**Input**: column of f64.
**Output**: `(min_abs, p01, p50, p99, max_abs)` — five-number summary of `|x|`.
**Interpretation**:
- `max_abs < π/4`: no reduction needed; trivial path dominates.
- `max_abs < 2^20·π/2 ≈ 1.65e6`: Cody-Waite adequate for the whole column.
- `max_abs > 2^20·π/2`: at least some rows need Payne-Hanek; if `p99 > 2^20·π/2`, majority path.
- `max_abs > 2^40`: extreme inputs; aggressive Payne-Hanek; suggests user mistake (accumulating phases unwrapped).

**Implementation**: one pass `accumulate(All, Abs, FiveNumberSummary)`. Cost ~0.2× of `sin(col)`.

**Kingdom**: A.

### P1.2 — `unit_detect`

**Question**: what angle unit is this column in?

**Input**: column of f64 (interpreted as angles in unknown unit).
**Output**: `{radians: f64, degrees: f64, gradians: f64, turns: f64, pi_scaled: f64}` — evidence score per unit, highest wins.
**Interpretation**: scores derived from likelihood the distribution makes sense:
- Radians: typical range -2π to 2π (~ -6.28 to 6.28), max ~ O(10⁴) for accumulators.
- Degrees: typical range -360 to 360 or 0-360, max ~ O(10⁷) for accumulators.
- Gradians: 0-400, less common.
- Turns: 0-1 or -1 to 1.
- Pi-scaled: 0-2 or -2 to 2 (input in units of π).

**Implementation**: compute histograms and peak locations, score against each unit's natural distribution. ~0.5× `sin(col)`.

**Kingdom**: A.

**Note**: this is explicitly a heuristic — unit is not always determinable from data alone. If the score is ambiguous (two units within 10% of each other), the probe reports `ambiguous` rather than guessing wrong. The user can supply ground truth via `using(angle_unit=…)`.

### P1.3 — `monotonicity_probe`

**Question**: is this column a monotonic phase accumulator (signal-processing use case) or a stationary distribution (general angles)?

**Input**: column of f64 + ordering (row index).
**Output**: `(monotonic_fraction, delta_mean, delta_std)` where `delta` = consecutive differences.
**Interpretation**:
- `monotonic_fraction > 0.95`: phase accumulator; reduction cost will dominate because `|x|` grows linearly with column length.
- `monotonic_fraction ≈ 0.5`: stationary, mixed sign.
- `delta_mean · n` approximates total phase span — tells you how many 2π wrap-arounds occurred.

**Implementation**: one pass `accumulate(Pairwise(Prev, Curr), Sign, Count+Mean+Std)`. ~0.3× `sin(col)`.

**Kingdom**: A.

### P1.4 — `integer_part_probe`

**Question**: is the integer part of `x / (π/2)` (or the angle-unit equivalent) small (fits in i32)?

**Input**: column of f64 + unit hint.
**Output**: `(max_k_abs: u64, needs_i64: bool)`.
**Interpretation**: matters for the reduction implementation's integer representation. If `max_k_abs > 2^31`, the reduction output's quadrant field can't fit in i32.

**Implementation**: one pass `accumulate(All, AbsDivAndFloor(period), Max)`. ~0.2×.

**Kingdom**: A.

### P1.5 — `nan_inf_count`

**Question**: how many special values are present?

**Input**: column of f64.
**Output**: `(n_nan, n_inf_pos, n_inf_neg, n_finite)`.
**Interpretation**: if non-zero special counts, output column will have propagation of NaN/Inf. Affects downstream decisions about mean/sum that consume sin/cos output.

**Implementation**: one pass, counts. ~0.1×.

**Kingdom**: A.

---

## Family 2 — Algorithm choice probes

Given characterized input, probe which kernel / reduction / precision to use. These are the probes that actually feed `.using(method=...)` auto-selection.

### P2.1 — `reduction_method_probe`

**Question**: which reduction algorithm is best for this column?

**Input**: column + magnitude distribution from P1.1.
**Output**: `{cody_waite: (applicable, est_cost), payne_hanek: (applicable, est_cost), hybrid: (est_cost)}`.
**Interpretation**:
- Cody-Waite applicable iff `max_abs < 2^20·π/2`.
- Payne-Hanek always applicable but ~3-10× cost.
- Hybrid: Cody-Waite fast-path with Payne-Hanek fallback on the tail — best for mixed columns.

**Implementation**: composed from P1.1, no additional passes. ~0 extra cost.

**Kingdom**: A.

### P2.2 — `kernel_method_probe`

**Question**: which forward-kernel algorithm is best?

**Input**: accuracy target, typical residual magnitude after reduction, hardware target.
**Output**: `{polynomial_fdlibm: est_error, polynomial_compensated: est_error, cordic: est_error, table_plus_polynomial: est_error}`.
**Interpretation**: a predictor of which kernel variant will hit the accuracy target fastest. Values come from precomputed calibration, not run-time benchmarks.

**Implementation**: table lookup against calibration data. ~0.01×.

**Kingdom**: A (literally just a dispatch table).

**Note**: CORDIC and table-plus-polynomial are future kernels per `first-principles.md` Phase 3. The probe can return "not yet implemented" for those; the `.discover()` forks over only the implemented ones.

### P2.3 — `precision_strategy_probe`

**Question**: do strict, compensated, and correctly_rounded give materially different answers on this column?

**Input**: column, reference output (compute all three).
**Output**: `(strict_vs_compensated_max_ulp_diff, compensated_vs_cr_max_ulp_diff, consensus_fraction)`.
**Interpretation**:
- All three agree to <1 ulp: strict is fine.
- Strict differs from compensated by 2-4 ulp on <1% of rows: strict is mostly fine; compensated for sensitive downstream use.
- Compensated differs from correctly_rounded on edge cases: use correctly_rounded for those rows.

**Implementation**: actually run all three and diff. Cost ~5× (all three strategies). Worthwhile for a discovery run, not for production.

**Kingdom**: A (pointwise diff).

### P2.4 — `fused_vs_split_probe`

**Question**: would `sincos(col)` + extract be faster than separate `sin(col)` + `cos(col)` for this workload?

**Input**: pipeline context — which downstream consumers exist.
**Output**: `(fused_cost, split_cost, savings)`.
**Interpretation**: if both sin and cos are consumed downstream, `sincos` wins. If only sin is consumed, split wins. Trivial binary decision that .discover() can surface for user awareness.

**Implementation**: pipeline graph inspection; no data pass. ~0.

**Kingdom**: A.

---

## Family 3 — Identity-check probes

The highest-leverage `.discover()` probes in trig. Trigonometric identities are **independent checks** that any correct implementation must satisfy. If two computation paths give answers that disagree with an identity, the library has just found a bug.

### P3.1 — `pythagorean_identity_probe`

**Question**: does `sin²(x) + cos²(x) = 1` hold across the column?

**Input**: column; internally compute sin and cos.
**Output**: `(max_deviation, p99_deviation, deviation_mean)` in ulps.
**Interpretation**:
- Max deviation < 2 ulp: implementation is correct.
- Max deviation > 4 ulp: bug or precision regime mismatch.
- Deviation grows with `|x|`: reduction is losing precision; consider Payne-Hanek or compensated.

**Implementation**:
```tbs
sincos(col=0).pythagorean_check()  # reuses cached sincos intermediate
# internally: accumulate(Pointwise, SqSq1Identity, MaxP99Mean)
# where SqSq1Identity(c, s) = abs(c*c + s*s - 1.0) / ulp(1.0)
```
Cost ~1.2× of `sincos(col)` (reuses cached values, adds one multiply-add + abs per row). ~0.2× if cache hits.

**Kingdom**: A.

**This is the highest-leverage probe in the catalog** — it catches reduction bugs, kernel bugs, quadrant-fixup bugs, and precision regressions, all with one cheap pass. It should run in CI on every trig recipe change.

### P3.2 — `sum_angle_identity_probe`

**Question**: does `sin(a+b) = sin(a)·cos(b) + cos(a)·sin(b)` hold for pairs drawn from the column?

**Input**: column; randomly pair rows `(a, b)` to form the identity test.
**Output**: deviation statistics in ulps.
**Interpretation**: the addition formula depends on reduction being correct for different magnitudes of input. If `a` is small and `b` is very large, the identity tests reduction across regimes.

**Implementation**: 3 sincos calls (for a, b, a+b), then arithmetic. ~3× `sincos(col)`.

**Kingdom**: A.

**Use case**: adversarial test harness for TRIG-17. Not a default .discover() probe because it's expensive.

### P3.3 — `double_angle_identity_probe`

**Question**: does `sin(2x) = 2·sin(x)·cos(x)` hold?

**Input**: column; compute sin/cos of x and sin of 2x.
**Output**: deviation statistics.
**Interpretation**: checks that reduction at 2x (which can cross regime boundaries) matches reduction at x. A common failure mode for naive Cody-Waite implementations near the Payne-Hanek threshold.

**Implementation**: 1 sincos(x) + 1 sin(2x). ~2× `sin(col)`.

**Kingdom**: A.

### P3.4 — `periodicity_probe`

**Question**: does `sin(x) = sin(x + 2π)` hold? Does `sin(x) = sin(x + 2πk)` for large integer k?

**Input**: column; shift by 2π, 4π, ..., 2^20·π, etc.
**Output**: `(shift_ulp_error_at_2pi, shift_ulp_error_at_2pi_k_max)` — how does error grow with k?
**Interpretation**: this **directly measures reduction quality at large |x|**. If error stays bounded as k grows, reduction is correct (Payne-Hanek working). If error grows linearly with k, Cody-Waite is being used beyond its range — bug.

**Implementation**: sin(x) + sin(x + 2πk) for a few k values. ~3-5× `sin(col)`.

**Kingdom**: A.

**This is the canonical test for Payne-Hanek correctness.** Any trig library claiming correctness at large x should pass this probe.

### P3.5 — `inverse_roundtrip_probe`

**Question**: does `asin(sin(x)) = x` hold on the principal branch?

**Input**: column restricted to `[-π/2, π/2]`.
**Output**: roundtrip error statistics.
**Interpretation**: validates that asin's branch cut handling is correct; cross-checks sin and asin simultaneously.

**Variants**: `acos(cos(x)) = x`, `atan(tan(x)) = x`, `atan2(sin(x), cos(x)) = x`, `exp(log(x)) = x`.

**Implementation**: one forward + one inverse. ~2×.

**Kingdom**: A.

### P3.6 — `symmetry_probe`

**Question**: does `sin(-x) = -sin(x)` hold exactly (bit-for-bit)? Does `cos(-x) = cos(x)` hold exactly?

**Input**: column.
**Output**: count of rows where the symmetry fails bit-wise.
**Interpretation**: both identities should hold EXACTLY in a correct implementation (reduction respects sign; polynomial is odd/even respectively). Any failure is a bug.

**Implementation**: pair each row with its negation, compare. ~2×.

**Kingdom**: A.

### P3.7 — `special_value_probe`

**Question**: does `sin(π/6) = 0.5`, `sin(π/4) = √2/2`, `sin(π/3) = √3/2`, `cos(0) = 1`, `sin(π/2) = 1`?

**Input**: none (tests fixed special values).
**Output**: ulp error at each special value.
**Interpretation**: library regression test; should run in CI.

**Implementation**: scalar calls against known exact values. ~0.

**Kingdom**: A.

---

## Family 4 — Scientific-insight probes

Not for algorithm selection — for surfacing scientific observations about the data.

### P4.1 — `dominant_frequency_probe`

**Question**: does the column exhibit a dominant frequency (as would be expected if it's a phase accumulator)?

**Input**: column.
**Output**: `(dominant_freq, amplitude, snr)`.
**Interpretation**: FFT-lite — if the column is actually the output of a sinusoidal process, this recovers the frequency. Helpful for users who aren't sure what they're looking at.

**Implementation**: FFT on diff(col), pick peak. ~5× sin(col) but uses shared FFT infrastructure.

**Kingdom**: A.

### P4.2 — `phase_wrap_count`

**Question**: how many times does the column cross a 2π boundary (i.e. how many full rotations)?

**Input**: column interpreted as accumulating phase.
**Output**: integer count + fractional remainder.
**Interpretation**: for a phase accumulator, this is the total angular path traversed. For a stationary sample, should be ~0.

**Implementation**: sum of `round(delta / 2π)` over pairs. ~0.3×.

**Kingdom**: A.

### P4.3 — `trig_regime_histogram`

**Question**: what fraction of the column lives in which trig regime (tiny / small / medium / large)?

**Input**: column + unit.
**Output**: `{tiny: fraction, small: fraction, medium: fraction, large: fraction}` where:
- tiny: `|x| < 2^-27·(π/4)` — first-order Taylor is accurate, no polynomial needed.
- small: `|x| < π/4` — polynomial directly, no reduction.
- medium: `|x| < 2^20·π/2` — Cody-Waite regime.
- large: `|x| ≥ 2^20·π/2` — Payne-Hanek regime.

**Interpretation**: tells the user what cost profile to expect. Feeds TRIG-2 (range variants) directly.

**Implementation**: one pass, histogram bucketing. ~0.2×.

**Kingdom**: A.

---

## Family 5 — Cross-method superposition probes

The deepest probes — run multiple methods simultaneously and report structural fingerprints of their agreement.

### P5.1 — `kernel_superposition`

Run the polynomial kernel AND a CORDIC kernel AND a table-lookup kernel in parallel. Report per-row agreement.

**Output**: `{all_agree_within_1_ulp: fraction, disagree_rows: (indices, spread)}`.
**Interpretation**: if all three agree to 1 ulp, the value is trustworthy to 1 ulp regardless of which kernel is used. If they disagree by 2+ ulp on specific rows, those rows are in a regime where kernel choice matters — investigate.

**Implementation**: Phase 5 of `first-principles.md` says CORDIC and table-lookup live under `sincos_kernel.using(method=...)`. If only the polynomial kernel is implemented today, this probe reports "only one kernel available" and the superposition degenerates to identity. Forward-looking.

**Cost**: ~K× where K is number of kernels (2-4×). High but bounded.

**Kingdom**: A.

### P5.2 — `reduction_superposition`

Same idea for Cody-Waite vs Payne-Hanek vs (future) Kahan-augmented Cody-Waite.

**Output**: agreement spectrum.
**Interpretation**: identifies inputs where reductions differ — typically near the regime boundary (x ≈ 2^20·π/2).

**Cost**: ~R× where R is number of reductions (2-3×).

### P5.3 — `precision_superposition`

Run strict, compensated, correctly_rounded simultaneously, report agreement.

**Output**: `{rows_where_strict_lies: count, rows_where_compensated_lies: count}` — count of rows where the cheaper strategy deviates from correctly_rounded by > 1 ulp.
**Interpretation**: tells you how much the strategy choice costs in accuracy. Most rows won't see any difference; the ones that do are diagnostic gold.

**Cost**: ~3×.

---

## Probe → `.discover()` wiring

`sin(col=0).discover()` runs (roughly) the following probes in parallel:

1. **P1.1** (input magnitude distribution) — always cheap, tells reduction choice.
2. **P1.3** (monotonicity) — cheap, tells whether this is a phase accumulator.
3. **P4.3** (regime histogram) — cheap, tells user cost expectations.
4. **P3.1** (pythagorean identity) — cheap after sincos cached, validates correctness.
5. **P3.7** (special values) — free, regression test.
6. **P5.3** (precision superposition) — expensive (3×), surfaces precision-sensitive rows. Optional; user can `.discover(include=["precision"])` to opt in.

Collapse: pick `precision=strict` if P5.3 shows universal agreement, else bump to `compensated`. Pick reduction automatically from P1.1. Report to user:

> Column analyzed: 10,000 rows. Magnitude range [-6.3, 6.3], 99% in Cody-Waite regime.
> Non-monotonic distribution (likely a stationary angle, not a phase accumulator).
> Pythagorean identity holds to 1.2 ulp (worst case). Strict precision is sufficient.
> Running sin at strict precision with Cody-Waite reduction.

**This is the scientific-knowledge output** that `.discover()` is designed to produce. The user learns about their data, not just gets an answer.

---

## Probe catalog summary table

| ID | Name | Family | Cost | Kingdom | Purpose |
|---|---|---|---:|---|---|
| P1.1 | input_magnitude_distribution | Input | 0.2 | A | Reduction dispatch |
| P1.2 | unit_detect | Input | 0.5 | A | Unit auto-detect |
| P1.3 | monotonicity_probe | Input | 0.3 | A | Phase-accumulator detect |
| P1.4 | integer_part_probe | Input | 0.2 | A | i32/i64 quadrant choice |
| P1.5 | nan_inf_count | Input | 0.1 | A | Special-value handling |
| P2.1 | reduction_method_probe | Algorithm | ~0 | A | CW vs PH |
| P2.2 | kernel_method_probe | Algorithm | ~0 | A | Poly vs CORDIC vs table |
| P2.3 | precision_strategy_probe | Algorithm | 5 | A | Strict vs CR |
| P2.4 | fused_vs_split_probe | Algorithm | ~0 | A | sincos vs separate |
| P3.1 | pythagorean_identity_probe | Identity | 0.2-1.2 | A | Correctness validation |
| P3.2 | sum_angle_identity_probe | Identity | 3 | A | Cross-regime check |
| P3.3 | double_angle_identity_probe | Identity | 2 | A | Boundary check |
| P3.4 | periodicity_probe | Identity | 3-5 | A | Payne-Hanek validation |
| P3.5 | inverse_roundtrip_probe | Identity | 2 | A | Forward/inverse pair |
| P3.6 | symmetry_probe | Identity | 2 | A | Odd/even parity |
| P3.7 | special_value_probe | Identity | ~0 | A | Known-value regression |
| P4.1 | dominant_frequency_probe | Insight | 5 | A | FFT peak |
| P4.2 | phase_wrap_count | Insight | 0.3 | A | Rotation counting |
| P4.3 | trig_regime_histogram | Insight | 0.2 | A | Cost profile |
| P5.1 | kernel_superposition | Superpos | 2-4 | A | Cross-kernel agreement |
| P5.2 | reduction_superposition | Superpos | 2-3 | A | Cross-reduction agreement |
| P5.3 | precision_superposition | Superpos | 3 | A | Cross-precision agreement |

**Total probe count: 22.**

---

## Implementation priority

For minimum viable `.discover()` for trig (post-reconstruction):

**Tier 1 — must ship with the first trig .discover() release:**
- P1.1, P1.3, P1.5 (basic input characterization)
- P3.1, P3.7 (correctness validation)
- P4.3 (regime histogram for user output)

**Tier 2 — ship before claiming parity with scipy/R:**
- P1.2 (unit detect)
- P2.1, P2.2, P2.3, P2.4 (algorithm dispatch)
- P3.4, P3.6 (essential correctness)

**Tier 3 — research/expedition value:**
- P3.2, P3.3, P3.5 (adversarial testing — feeds TRIG-17)
- P4.1, P4.2 (scientific insight)
- P5.1, P5.2, P5.3 (superposition — the crown jewel)

---

## Integration with other tasks

- **TRIG-2 (range variants)**: P4.3's histogram output IS the range variant metadata. TRIG-2 can consume this probe's output directly.
- **TRIG-17 (adversarial tests)**: P3.x family are the adversarial probes. The difference between a probe and a test is just whether it runs on user data (probe) or on a fixed adversarial dataset (test). Same code.
- **TRIG-18 (parity with R/Python/CUDA)**: P5.3 (precision superposition) is already a kind of parity probe — add "scipy", "R", "CUDA" as precision-like branches and the same probe compares tambear to each external reference.
- **TRIG-6 ($1M/yr scientist defaults)**: P1.1 + P2.1 + P2.3 together define the default choice for every trig call. TRIG-6's deliverable should reference these probes as the evidence base.
- **TRIG-12 (.spec.toml)**: each trig function's spec.toml can list its recommended probes in a new section `[discover] probes = ["P1.1", "P3.1", ...]`. Becomes a clean contract.

---

## Open questions

1. **Probe calibration data.** P2.2's kernel method probe needs precomputed accuracy estimates per kernel per regime. Who generates and maintains this? Suggestion: adversarial teammate's harness generates it as a byproduct of TRIG-17.

2. **Stability of unit_detect.** P1.2 is a heuristic. If it guesses wrong on ambiguous data, users get silently wrong answers. Safer design: P1.2 only **suggests** the unit; dispatch requires explicit `.using(angle_unit=...)` if score is ambiguous. Error message: "column looks like both radians and turns; please specify."

3. **Cost of Tier 3 probes in production.** P5.x superposition probes cost 2-4×. Running them by default on every `.discover()` call is expensive. Proposal: Tier 3 probes default off, user opts in with `.discover(level="deep")`.

4. **Probe as a standalone surface.** Should probes be callable directly? `pythagorean_deviation(col=0)` as a public TBS function, not hidden inside `.discover()`? Argument for: users writing their own QA pipelines want them. Argument against: it multiplies the surface area. Lean yes — the probes are independently useful.

---

## Handoff notes

- **To scientist (TRIG-6, TRIG-18)**: P2.3 (precision) and P3.x (identities) are your natural territory. The probe catalog is meant to scaffold your parity + scientist-defaults work.
- **To adversarial (TRIG-17)**: P3.x is your harness. Probes run on real data; adversarial tests run on curated evil data. Same mechanics.
- **To pathmaker**: if you add `[discover] probes = [...]` to spec.tomls, I'll iterate with you on the format.
- **To researcher**: P5.1 references CORDIC and table-lookup kernels as future work. When you tackle those under TRIG's future expansion, these probes are ready-made test harnesses.
- **To navigator**: probe catalog shipped. Next candidates: TRIG-2 (range variants — trivially reducible to P4.3), TRIG-7 (TBS syntax — touches `.discover()` surface), TRIG-20 (master synthesis).

---

## Deliverable status

- [x] 5 probe families (input, algorithm, identity, insight, superposition)
- [x] 22 individual probes specified (name, question, I/O, interpretation, implementation, kingdom, cost)
- [x] Summary table
- [x] .discover() wiring example (output the user actually sees)
- [x] Implementation priority tiers
- [x] Integration with TRIG-2, 17, 18, 6, 12
- [x] Open questions + handoff notes
