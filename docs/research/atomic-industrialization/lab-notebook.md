# Lab Notebook: Atomic Industrialization — Session 2026-04-10

**Date**: 2026-04-10
**Observer**: observer agent (Sonnet 4.6)
**Branch**: main
**Status**: Active
**Depends on**: Wave 5 commit (c005fe0), previous industrialization waves 1-5

---

## Context & Motivation

The atomic industrialization campaign decomposes every tambear method into primitives,
wires every `using()` path, fills every catalog gap, and proves every implementation.

This notebook is the neutral record. It documents what IS, verified independently.
Not what agents reported — what the code and tests show.

---

## Baseline Established at Session Start

**Time**: 2026-04-10, session open

### Test counts (verified by running `cargo test --lib`)

| Crate | Tests | Status |
|-------|-------|--------|
| tambear | 2,152 | all green |
| tambear-fintek | 279 | all green |
| **Total** | **2,431** | **all green** |

**Note on linker corruption**: `cargo test` (without `--lib`) fails with
`CVTRES: fatal error CVT1107: file invalid or corrupt` on several test binaries
(adversarial_boundary4, adversarial_boundary10, adversarial_hardened). This is
a pre-existing Windows COFF linker issue, NOT a code regression. `cargo check`
and `cargo test --lib` are both clean.

**`cargo check` status**: clean (both tambear and tambear-fintek)

### Public function count

Total `pub fn` declared across tambear source modules: **1,257** (grep count across all
`crates/tambear/src/*.rs`). This includes internal helpers, not all are in the TBS surface.

### Key metric: using() coverage

Per campsite `industrialization/using-wiring/20260410144025-passthrough`:
> "using() is 97% aspirational. Only ~10 of ~400 pub fns query UsingBag."

**Independently verified**: Grep of all `*.rs` source files confirms:
- `pub fn` accepting `UsingBag` parameter: **14 functions** across 5 files
  (`distributional_distances.rs`, `hazard.rs`, `predictive.rs`, `scoring.rs`, `tbs_executor.rs`)
- `pub fn` that actually READ from UsingBag (call `.get_f64()`, `.method()`, etc.):
  **~16 call sites** across those same files
- Total `pub fn` in tambear: **1,257** (grep count across all `src/*.rs`)

**Conclusion**: The "97% aspirational" characterization is accurate. Roughly 14/1257 = 1.1%
of public functions take a `UsingBag` parameter. Of the 1257 functions, the vast majority
have hardcoded parameters with no `using()` wiring at all.

Note: The 1,257 count includes ALL pub fn (including internal helpers, test utilities,
neural network layers, etc.). The TBS-visible "method-level" functions are a subset.
The architectural gap is real regardless of how you count.

---

## Changes Landed (Uncommitted, Staged in Working Tree)

As of session start, `git diff HEAD` shows 3 modified files. No commits yet made this session.

### Change 1: `crates/tambear/src/information_theory.rs`

**What changed**: Two private functions promoted to `pub`:
- `p_log_p(p: f64) -> f64` — the `p · ln(p)` atom
- `p_log_p_over_q(p: f64, q: f64) -> f64` — the `p · ln(p/q)` atom
- `contingency_from_labels(labels_a, labels_b) -> (Vec<f64>, usize, usize)` — contingency matrix builder
- `expected_mutual_info(a, b, n) -> f64` — hypergeometric null MI

**Also changed**: `p_log_p_over_q` now correctly returns `+∞` when `p > 0 && q = 0`,
per the absolute continuity convention for KL divergence.

**Rationale**: These are fundamental atoms used by many downstream methods (Shannon entropy,
KL, cross-entropy, AMI, VI). Making them public means users can compose new information-
theoretic measures without reimplementing the same careful edge-case handling.

**Correctness question**: The `p_log_p_over_q` returning `+∞` for `q=0` is the
mathematically correct convention (absolute continuity violation). Verified.

### Change 2: `crates/tambear/src/lib.rs`

**What changed**: Exports added for the newly-public atoms:
```rust
pub use information_theory::{
    // ... existing exports ...
    p_log_p, p_log_p_over_q,
    contingency_from_labels, expected_mutual_info,
};
```

### Change 3: `crates/tambear/src/complexity.rs`

**What changed**: Two methods decomposed to use the `delay_embed` primitive:
- `correlation_dimension`: was inlining the delay embedding loop; now calls
  `crate::time_series::delay_embed(data, m, tau)`. Also switched storage to
  row-major flat vec for pairwise L∞ computation.
- `largest_lyapunov`: same — inlined delay embedding replaced with
  `crate::time_series::delay_embed(data, m, tau)`.

**Why this matters**: Takens' theorem delay embedding is now a single canonical
implementation in `time_series::delay_embed`. If there's a bug (e.g., off-by-one
in the embedding dimension), it's fixed in one place. Previously it was a private
copy inside each method.

**Test impact**: Tests still pass (2,152). The decomposition preserves behavior.

---

## New Artifacts (Untracked Files)

### `crates/tambear/tests/adversarial_rigor_gauntlet.rs` (864 lines, 67 tests)

A new adversarial test file targeting:
- `distance_correlation` — NaN propagation, Inf input, n=2 edge case, negative dcov²
- `hoeffdings_d` — n<5 boundary, NaN/Inf input, perfect monotone
- `blomqvist_beta` — odd n signum bug, constant X, NaN/Inf
- `grassberger_entropy` — permutation invariance
- `brusselator_*` — zero/NaN inputs, zero-step simulation
- `log_returns`, `normal_pdf`, `tie_count`

**Philosophy stated in the file**: "A FAILING test is a found bug — the team's work queue.
We do not adjust assertions to match code; we fix the code to match math."

**Status**: Not yet compiled against `--lib` because it's a separate test file.
The linker corruption issue will prevent `cargo test --test adversarial_rigor_gauntlet`
from running directly. Needs verification once linker issue is resolved, or via
inclusion in lib-level test module.

### Campsite notebooks created

- `campsites/industrialization/missing-families/20260410144047-information-theory/math-researcher/notebooks/complete-variant-catalog.md`
  — Exhaustive catalog of what exists + what's missing in information theory
- `campsites/industrialization/missing-families/20260410144047-complexity/math-researcher/notebooks/complete-variant-catalog.md`
  — Exhaustive catalog of what exists + what's missing in complexity
- `campsites/industrialization/missing-families/20260410144102-linear-algebra/math-researcher/notebooks/complete-variant-catalog.md`
  — Exhaustive catalog of what exists + what's missing in linear algebra

### `docs/research/atomic-industrialization/expedition-log.md`

Written by naturalist. Contains the spring network topology analysis — the key finding
that "family" means two structurally different things (clusters vs webs), and that
the TamSession phyla are defined by sharing topology, not family labels.

---

## Phantom Reference Audit

**Claim from campsite**: "CRITICAL: gap_statistic references `crate::kmeans::kmeans`
which DOES NOT EXIST."

**Independent verification**:
- `crates/tambear/src/kmeans.rs` EXISTS
- `kmeans.rs` is declared as `pub mod kmeans` in `lib.rs`
- `KMeansEngine` struct and `new()` method exist in `kmeans.rs`
- BUT: `gap_statistic` in `clustering.rs` does NOT call `crate::kmeans`. It defines
  a local closure `kmeans_cpu_f64` for CPU/f64 path, because kmeans.rs uses GPU/f32.

**Conclusion**: The original phantom report was approximately correct — `gap_statistic`
cannot use `crate::kmeans::KMeansEngine` for its f64 CPU work. The local closure is
a reasonable workaround, but it's a hidden duplication of k-means logic. This is a
real architectural gap, even if not a compile-time phantom.

**Other crate:: references scanned**: The following modules referenced via `crate::` all
exist as declared modules in `lib.rs`:
`bigint`, `clustering`, `codegen`, `compute_engine`, `copa`, `descriptive`,
`fold_irreversibility`, `format`, `frame`, `intermediates`, `kmeans`, `knn`,
`linear_algebra`, `manifold`, `mixed_effects`, `nonparametric`, `numerical`,
`pipeline`, `proof`, `reduce_op`, `rng`, `signal_processing`, `special_functions`,
`tbs_executor`, `tbs_lint`, `tbs_parser`, `train`, `using`

No compile-time phantom references detected in the main library code.
(`superposition.rs` and `tbs_executor.rs` reference `crate::kmeans::KMeansEngine`
which exists — but the GPU path may fail at runtime on CPU-only machines.)

---

## Campsite State (All Campsites)

All 36 campsites are status: **open**. None have been marked active or done this session.

**My assigned campsite**: `ideas/tools/20260410144251-gemma-local-model` — open, unworked.
This campsite is about local Gemma model tooling. No work has been done on it this session.
Observing rather than implementing is the right call here — it's an ideas campsite, not
a blocking dependency.

---

## Claims Made vs. Claims Verified

| Claim | Source | Verified? | Notes |
|-------|--------|-----------|-------|
| 2,152 tambear tests | team-lead prompt | YES | ran `cargo test --lib` |
| 279 bridge tests | team-lead prompt | YES | ran `cargo test --lib` |
| using() is 97% aspirational | pathmaker campsite | YES — confirmed + quantified | 14/1257 = 1.1% wired |
| kmeans is a phantom | scout campsite | PARTIALLY — gap but not phantom | kmeans.rs exists; gap_statistic uses local closure |
| delay_embed decomposition preserves test count | implicit | YES | 2,152 still pass |
| ~10/400 pub fns wire using() | campsite | YES — 14 pub fns, ~16 call sites | measured via grep |
| blomqvist_beta signum bug | gauntlet commit | DISPUTED — code appears already fixed | raw diff check present with comment |
| log_returns Inf slips through | gauntlet commit | DISPUTED — is_finite guard catches Inf | verified manually |
| hoeffdings_d NaN silently dropped | gauntlet commit | PLAUSIBLE — NaN sort undefined | cannot run external test |
| distance_correlation overflow silent 0 | gauntlet commit | PLAUSIBLE — extreme values overflow | cannot run external test |
| tie_count unsorted assumption | gauntlet commit | CONFIRMED as documented assumption | test explicitly says "should fail" |
| 59/67 gauntlet tests pass, 8 fail | gauntlet commit | CANNOT VERIFY | external test binary blocked by linker |

---

## Event Log

### 2026-04-10 — Commit 7c4b54d: Adversarial Wave 11 (INTRODUCES TEST FAILURE)

**Commit**: `7c4b54d — Adversarial wave 11: 5 more bugs in fresh primitives`

**Contents**: `crates/tambear/tests/adversarial_wave11.rs` (751 lines, 55 tests)
**Also modified**: `crates/tambear/src/linear_algebra.rs` — added matrix_exp/log/sqrt tests
  within the lib's internal test module.

**Reported status**: 50 pass, 5 fail (all failures are real bugs per commit message).

**Bugs claimed**:
- `stirling_approx(Inf)` returns NaN (Inf-Inf cancellation), should be Inf
- `bic_score` k=1 returns Inf (c.k >= 2 guard excludes valid baseline model)
- `gap_statistic` constant data returns Inf gaps (log(0)=-Inf not handled)
- `gram_schmidt` NaN vector accepted into basis (NaN.sqrt() not < 1e-10)
- `dtw` NaN input: Rust f64::min eats NaN, gives silent wrong result

**CRITICAL: This commit broke the lib test suite.**

Result after this commit: **2,190 passed; 1 FAILED** (was 2,152 green before).

The failing test: `linear_algebra::tests::matrix_exp_identity_scale`

**Root cause**: New lib-internal tests call `mat_approx_eq` with 4 arguments, but that
helper only takes 3. A `mat_approx_eq_msg` variant exists for the 4-arg form.
The code was updated between my checks — by the time I ran the full test suite,
the call sites had been updated to `mat_approx_eq_msg`. BUT the `matrix_exp_identity_scale`
test still FAILS — not a compile error but a NUMERICAL FAILURE.

**The genuine numerical failure**: `matrix_exp(t·I)` where `t=0.5` should produce `e^t · I`.
The test asserts this to tolerance `1e-7`. It fails. The Padé [6/6] approximant in
`matrix_exp` doesn't achieve `1e-7` accuracy for a 3×3 scaled identity matrix.

This is a real bug in `matrix_exp` — the Padé approximant should be highly accurate
for a diagonal matrix. Possible causes:
- Scaling step error (norm1 of `0.5·I_3x3` is 0.5, borderline for the `≤ 0.5` check)
- LU solve introducing error
- Squaring step amplification

**Root cause analysis of matrix_exp failure**:

For `A = 0.5·I₃`, all matrix operations are trivially diagonal — the Padé approximant
reduces to scalar arithmetic. A correct [6/6] Padé for `exp(0.5)` should achieve
~10^{-14} accuracy. Failing at `1e-7` is anomalous.

The code uses the SAME 7-coefficient array for both `U` (odd-power part) and `V`
(even-power part):
```
U = A(c[5]·A⁴ + c[3]·A² + c[1]·I)
V = c[6]·A⁶ + c[4]·A⁴ + c[2]·A² + c[0]·I
```
This structure is consistent with the standard Padé scaling-and-squaring approach
(Higham 2005), but the coefficients c[0..6] need to exactly match Table 10.2 for
the [6/6] degree. If these are [13/13] Padé coefficients being used at degree 6,
or vice versa, the accuracy would be severely degraded.

The 7 coefficients (1.0, 0.5, 0.12, 1.833e-2, 1.992e-3, 1.630e-4, 1.035e-5)
should be independently verified against Higham 2005 Table 10.2 or the MATLAB
`expm` reference implementation.

**State**: Test suite is BROKEN. 1 failing test. This needs to be fixed before proceeding.

---

### 2026-04-10 — Commit f2655fd: Adversarial Gauntlet

**Commit**: `f2655fd — Adversarial gauntlet: 8 bugs found in 5 primitives, 67 tests written`

**Contents**: `crates/tambear/tests/adversarial_rigor_gauntlet.rs` (895 lines, 67 tests)

**Reported status**: 59 pass, 8 fail. All failures are bugs, not wrong assertions.

**Targets** (per commit message): blomqvist_beta, hoeffdings_d, distance_correlation,
log_returns, tie_count.

**Independent verification of specific claims**:

1. **blomqvist_beta signum bug**: nonparametric.rs:3651-3659 already uses `dx == 0.0 || dy == 0.0`
   raw difference check, with comment explaining why `signum == 0.0` can't work in Rust.
   Test expects `-8/9` — current code should produce `-8/9`. **May already be fixed.**

2. **log_returns Inf guard**: Commit says "Inf price slips through". But `log_returns`
   guard is `!w[0].is_finite() || !w[1].is_finite() || w[0] <= 0.0 || w[1] <= 0.0`.
   `f64::INFINITY.is_finite()` returns `false`, so `!is_finite()` is `true` for Inf.
   Guard DOES fire. **This test assertion may be wrong.** The gauntlet comment is incorrect:
   "Inf <= 0.0 is false" is true, but the `is_finite` branch catches it first.

3. **tie_count unsorted**: Test explicitly says "This SHOULD fail" — it's documenting
   an assumption violation (caller must sort input), not a bug in the implementation per se.

4. **hoeffdings_d NaN**: NaN sort behavior is implementation-defined in Rust. Plausible bug.

5. **distance_correlation overflow**: Extreme values (1e300) causing overflow — plausible.

**Cannot fully verify 8 failures** until the Windows COFF linker issue is resolved.
The external test file compiles (cargo check passes), but linking fails.

**Test count impact**: lib test count unchanged at 2,152. External test files don't
count in `--lib` runs.

---

## Open Questions for Observation

1. **Linker corruption**: Which test binaries are affected? Is this consistent or random?
   Does rebuilding from clean state fix it? This should be diagnosed before the rigor
   gauntlet tests can be counted.

2. **using() coverage**: What is the actual count of `pub fn` that call `using()`?
   The 97% aspirational claim needs a concrete measurement.

3. **Information theory atoms**: Now that `p_log_p`, `p_log_p_over_q`,
   `contingency_from_labels`, `expected_mutual_info` are public, do any existing
   internal callers use private paths that should now route through the public API?

4. **correlation_dimension L∞ vs L2**: FIXED this session. The comment "flatten to
   row-major for pairwise_dists (L2 distance)" was wrong — corrected to "L∞".

5. **adversarial_rigor_gauntlet.rs**: 67 tests committed (f2655fd). Cannot run as
   external binary due to Windows COFF linker issue. 59/8 pass/fail is from the
   adversarial agent's own run — cannot independently confirm without linker fix.
   Two specific "bugs" (blomqvist signum, log_returns Inf) appear already fixed in
   current code; gauntlet test assertions may be stale.

6. **Gauntlet test quality**: Two of the 8 claimed bugs appear to be either already
   fixed in current code (blomqvist) or incorrectly characterized (log_returns Inf).
   This is worth flagging — if the adversarial agent counted these as "bugs found"
   but they were already fixed, the bug count may be overstated.

---

## What to Watch Next

- Commits landing from pathmaker: will test count go up?
- Campsite status changes: which campsites move to active/done?
- Any new `pub fn` added: update the public function count
- Any test failures: document immediately with full error text

---

*Last updated: 2026-04-10 by observer (after commit 7c4b54d — test suite BROKEN, 1 failure)*
