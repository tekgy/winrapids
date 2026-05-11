# Scientist Survey — Oracle Infrastructure & Validation State

**Date:** 2026-05-08  
**Role:** scientist  
**Scope:** sweep-10 campsite (oracle-seeding) + full oracle infrastructure audit + trig validation readiness

---

## The Sweep-10 Campsite

`R:\winrapids\campsites\sweep-10\20260423054754-oracle-seeding\` contains exactly one file:
the math-researcher creation note. Content:

> Seeded variance/python-numpy/default + variance/python-pandas/default + corrected
> mean/python-statistics/default known_issues. Computed mpmath 50dps reference values
> for 5 variants × 3 ddof regimes. Critical finding: numpy.var loses 1.6 MILLION ULP
> on ill_conditioned variant — full GAP-DET-1 reproduction in oracle artifacts.

The campsite is a thin marker for work that landed in `R:\tambear\oracle\`. The real
output is in the oracle directory, not the campsite.

---

## Oracle Infrastructure State in Tambear

### What's There

`R:\tambear\oracle\` has four live recipe catalogs and one template:

**`mean/`** — most mature. Six data variants with full input.json + input.hash:
standard_gaussian_n1000, ill_conditioned, heavy_tail_t2, outlier_contaminated,
small_n10, zero_mean, symmetric_uniform, laplace_symmetric. Implementations:
python-numpy/default (output.json), python-scipy/default (output.json),
python-statistics/default (output.json), R-base/default (script present, R not
installed so output.json not generated). Comparison docs: methodology.md,
operational_guidance.md, parameter_truth.md. Generator.md explains how to
reproduce. No tambear/ implementation entry — no output.json for tambear itself.

**`variance/`** — deep research, well-seeded. Multiple Python implementations with
output.json (numpy, pandas, scipy, polars, statistics, statsmodels, R base simulated).
`tuned_nanvar` entry in python-numpy. The ill_conditioned data variant present
(extreme_conditioning_kappa_1e16). GAP-DET-1 documented: Welford+Sequential gives
6.25% wrong on ill-conditioned input. `generate_outputs.py` and
`validation/chan_merge_acceptance.py` present. parameter_truth.md has four
high-value findings (ddof trap, algorithm-conditioning relationship, Chan required
for lifted execution, polars upstream bug). Full SPEC.md present with algorithm
comparison, NaN-aware variants, and weighted variance. No tambear/ output.json.

**`centered_moment/`** — full SPEC.md (the generalization that variance is one
recipe over). Four Python algorithm implementations with output.json: kahan-two-pass,
naive-one-pass, two-pass-naive, welford. comparison/summary.json present. This is
the right mathematical framing — `variance` is a recipe over `centered_moment`
with (p=2, center=mean).

**`ewma/`** — unique pattern: lift-equivalence verification instead of
cross-implementation comparison. `validation/lift_equivalence.py` + `acceptance.json`.
Six acceptance regimes (α=0.5, 0.1, 0.01, α=1 edge, NaN injection, Inf injection)
all 0 ULP lifted vs. reference. This is the template for lift-proof oracles.

**`raw_moment/`** — SPEC.md only (one-pass algorithm). No implementations.

**`TEMPLATE/`** — empty template structure per ORACLE_MISSION.md conventions.

**`LIFT_EQUIVALENCE_TEMPLATE.md`** — template for the lift-proof oracle pattern
(companion to oracle/ewma/validation/).

### The oracle_compare Primitives

`R:\tambear\crates\tambear\src\primitives\oracle_compare\mod.rs` — **fully shipped**:

- `abs_within(a, b, tol, NanPolicy)` — scalar absolute comparison
- `rel_within(a, b, rtol, atol, NanPolicy)` — numpy allclose semantics
- `ulp_within(a, b, max_ulps, NanPolicy)` — ULP distance comparison
- `ulp_distance_f64(a, b)` and `ulp_distance_f32(a, b)` — raw metrics
- `allclose_ulp`, `allclose_abs`, `allclose_rel` — slice variants with FirstMismatch
- Sign-magnitude canonicalization for correct ULP counting through zero
- Full test coverage: signed zero, denormals, monotonicity across zero, NaN policies,
  Infinity handling, f32 parity

Dawson 2012 sign-magnitude mapping implemented correctly. Tests include the canonical
examples from ORACLE_MISSION.md §"ULP counting."

---

## The Critical Missing Piece: `auto_within`

ORACLE_MISSION.md §"Impact on oracle infrastructure" specifies a fourth function:

```
auto_within(gold, result, data_scale, ulp_threshold, abs_threshold)
```

that switches to absolute comparison when `|gold| < K × epsilon × data_scale`
(K=16 default, epsilon=2.22e-16). Without this, any oracle for a method that
produces near-zero expected values gets misleading ULP figures.

The worked example from ORACLE_MISSION.md:
- Data: Uniform(-1, +1), n=2000. True mean ≈ 0.
- Plain sequential sum gives result = -2.37e-17, gold = -4.44e-19.
- Absolute error: 2.33e-17 (numerically fine — smaller than any signal in data).
- ULP distance: **25.6 trillion** (looks catastrophic; is not).
- Neumaier summation gives gold exactly (0 ULP, 0 abs error). Both metrics agree it's correct.

**`auto_within` does not exist in oracle_compare/mod.rs.**

This is tracked as AUDIT-ULP-NEAR-ZERO, Task #19 in ORACLE_MISSION.md and
ARCHITECTURAL_INSIGHTS.md, blocked by Sweep 10 landing. Sweep 10 has not landed.
Eight existing `ulp_within` call sites were audited post-2026-04-25 as latent false
positives (from future-refinements.md).

**Affected recipes:** any method whose expected value can be near zero — symmetric
distribution means, zero-residual statistics, difference statistics, correlation
at zero, centered moments. The `laplace_symmetric` and `symmetric_uniform` data
variants in the mean oracle are specifically in this failure mode.

**Implementation spec is complete.** Per ORACLE_MISSION.md:

```rust
pub fn auto_within(
    gold: f64,
    result: f64,
    data_scale: f64,   // max(|x_i|) of input data
    ulp_threshold: u64,
    abs_threshold: f64,
    nan_policy: NanPolicy,
) -> bool {
    const K: f64 = 16.0;
    const EPSILON: f64 = 2.220_446_049_250_313e-16;
    if gold.abs() < K * EPSILON * data_scale {
        abs_within(gold, result, abs_threshold, nan_policy)
    } else {
        ulp_within(gold, result, ulp_threshold, nan_policy)
    }
}
```

This is ~30 lines of Rust with tests. It is the first thing that should land in
Sweep 10.

---

## Parity Table: Oracle Coverage by Recipe

| Recipe | mpmath oracle? | Competitors oracled? | Tambear impl? | Oracle verdict |
|---|---|---|---|---|
| `mean` | YES 50dps, 5+ variants | numpy, scipy, statistics, R(sim) | winrapids only | Partial — no tambear output.json |
| `variance` | YES 50dps, 5+ variants | numpy, pandas, scipy, polars, statistics, statsmodels, R(sim) | winrapids only | **GAP-DET-1: CONFIRMED WRONG** |
| `centered_moment` | YES (algorithm comparison) | 4 Python algorithms | Not yet | Algorithmic catalog done |
| `ewma` | YES (lift-equivalence) | Reference fold | YES (tambear) | SIGNED OFF — BitExact |
| `sin`, `cos`, `tan` | Partial — platform oracle only | None | winrapids only | Platform oracle problem — INSUFFICIENT |
| `asin`, `atan` | Partial — adversarial tests | None | winrapids only | asin polynomial fixed, but oracle still platform-backed |
| `exp` | YES — derive_exp_*.py constants verified at 100dps | None | winrapids only | Constants correct, full recipe not oracled |
| `log` | YES — derive_log_minimax.py constants verified | None | winrapids only | Constants correct, full recipe not oracled |
| `erf`, `gamma` | None | None | winrapids only | Not validated |
| All hyperbolic | Adversarial tests only | None | winrapids only | Not validated |
| pi-scaled (sinpi, cospi, tanpi) | cospi has external oracle test | None | winrapids only | Partial |
| `softmax`, `log_softmax` | None | None | winrapids only | NaN fix committed, no oracle |

---

## Trig Adversarial Tests — Platform Oracle Problem

The adversarial test files (`trig_adversarial.rs`, `trig_adversarial_asin.rs`, etc.)
use `x.sin()`, `x.cos()`, `x.tan()` (Rust platform libm) as the oracle throughout.

The adversarial audit from sweep-8 (`20260422-trig-libm-audit.md`) documents this gap:

> `sin(355)` is the canonical range-reduction stress test. `355/113 ≈ π` to 6 decimal
> places, so `355 mod π` is very small. A Cody-Waite implementation with insufficient
> π/2 precision returns garbage here — but the test would PASS even if our
> implementation is wrong, because it's comparing against the same buggy platform libm.

The correct oracle for hard cases:
```python
import mpmath; mpmath.mp.dps = 50
float(mpmath.sin(355))  # -7.963267107332633e-4
```

This value can be hardcoded as an f64 literal. The platform oracle is only acceptable
for trivial inputs where no implementation can plausibly be wrong.

Additionally: `sin_strict`, `sin_compensated`, `sin_correctly_rounded` all call
`sin_strict`. The differently-named entry points are structural fiction — there is no
differentiated accuracy tier.

**Before any trig function can receive a scientist sign-off, the hard-case tests
must use mpmath-backed literals, not platform oracle.**

---

## The Confirmed Bug: GAP-DET-1

`variance` oracle parameter_truth.md documents:

> GAP-DET-1 (2026-04-22): the current tambear implementation uses Welford
> sequentially, which gives 6.25% error on the ill-conditioned variant in
> adversarial testing. This is a confirmed bug in the tambear implementation,
> not just a benchmark observation.

The fix is fully specified: Chan's parallel combine formula:
```
M2_combined = M2_a + M2_b + δ² × n_a × n_b / (n_a + n_b)
```
where `δ = mean_b - mean_a`. This is the correct algorithm for lifted variance.

The oracle to verify the fix already exists (`variance/validation/chan_merge_acceptance.py`).
Once Chan's algorithm is implemented in tambear, run this acceptance harness.

---

## Is Oracle Infrastructure Blocking Formalization?

**Partial block. Here is the honest picture:**

**Does NOT block** formalization of recipes where we already have mpmath-verified
constants and the adversarial tests cover hard cases:
- exp, log — constants are mpmath-correct via derive_*.py. The adversarial tests
  cover edge cases (NaN, Inf, overflow, underflow). Missing: competitor comparison
  (scipy, Julia, rust-libm) and multi-scale benchmarks. Acceptable to formalize
  with a documented oracle gap.
- asin, atan — polynomial coefficients audited and corrected. The hard case tests
  need mpmath literals substituted for platform oracle, but this is a one-afternoon
  fix that can happen alongside formalization.

**Does block** a clean scientist sign-off (publication-grade rigor) on:
- `sin`, `cos`, `tan` — platform oracle problem means adversarial tests give false
  confidence on the exact hard cases (sin(355), Payne-Hanek regime) that matter most.
  Fix needed before sign-off: mpmath-backed literals for ~10 hard cases per function.
- `variance` — GAP-DET-1 is a 6.25% accuracy bug. Cannot sign off until Chan's
  algorithm is implemented and passes `chan_merge_acceptance.py`.
- Any recipe producing near-zero outputs — `auto_within` must exist before oracle
  assertions are trustworthy for zero-mean or symmetric data.

**The ordering that unblocks the most:**

1. `auto_within` in oracle_compare — ~30 lines, unblocks correct oracle assertions
   for mean, variance, and any near-zero recipe.
2. Variance fix (Chan's algorithm) — unblocks variance sign-off, which is the
   foundational statistic everything else builds on.
3. Trig hard-case oracle substitution — replace platform oracle literals with mpmath
   literals for `sin(355)`, `cos(355)`, `sin(kπ ± ε)` family, Payne-Hanek regime.
   Then trig can be formalized with a valid test suite.
4. Move libm directory from winrapids to tambear — structural prerequisite for any
   trig commit. Files are currently in the wrong repo.

---

## Root-Level Oracle Artifacts Assessment

Five Python scripts at `R:\winrapids\` are mature oracle artifacts ready to land:

- `variance_nan_oracle.py` + `nanvar_reference.json` → `oracle/variance/validation/`
  (NaN-aware variance oracle: clean data, 10% NaN injection, historical regressions,
  ddof×n_effective semantics, all mpmath-backed)
- `verify_variance_spec_claims.py` → `oracle/variance/validation/` (empirical
  cross-reference of SPEC §3 claims against measured behavior)
- `derive_exp_constants.py`, `derive_exp_minimax.py`, `derive_log_minimax.py` →
  `oracle/exp/` and `oracle/log/` as data generators + reference scripts when those
  oracles are created

The other scripts (`verify_affine_chain_error.py`, `verify_canonicalize.py`,
`verify_dimhint_partial_orders.py`) are infrastructure tooling, not recipe oracles.

---

## Sign-Off Status

| Recipe | Can I sign off now? | Blocker |
|---|---|---|
| `ewma` | YES | None — already verified |
| `exp` (constants) | YES (constants only) | Full recipe + competitor comparison pending |
| `log` (constants) | YES (constants only) | Full recipe + competitor comparison pending |
| `asin` (polynomial) | YES (coefficients) | Platform oracle in tests must be replaced |
| `variance` | NO | GAP-DET-1 confirmed wrong; fix not yet shipped |
| `sin`, `cos`, `tan` | NO | Platform oracle problem; tests don't validate hard cases |
| All hyperbolic | NO | No oracle at all |
| `mean` | Partial | No tambear output.json; auto_within needed for symmetric data |

The scientist's verdict: oracle infrastructure is far enough along to begin
formalization of exp, log, and the inverse trig family, with specific gaps to
close in parallel. The trig hard-case oracle fix and auto_within are the two
items that unblock a clean sign-off on the whole transcendental family.
