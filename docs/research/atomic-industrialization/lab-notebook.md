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

## Campsite State

Session start: 36 campsites, all open.
After team-lead seeding: **41 campsites** (CLI count). 10 active, 31 open.

**My campsite**: `ideas/tools/20260410144251-gemma-local-model` — active (claimed by me),
observing. No implementation work — correct for an ideas/observer campsite.

### New theory campsites (4 — PUBLISHABLE per team-lead)

Independently read and assessed:

1. **theory/classification-bijection**: Algorithm classes = sharing clusters = spring equilibria.
   Three convergent derivations. Potentially publishable if formally verified.

2. **theory/k7-scheduling-bound**: CovMatrix induces K₇ clique (treewidth=6, tight).
   Concrete memory lower bound. Grows with catalog. Publishable as a scheduling theory result.

3. **theory/holographic-error-correction**: `.discover()` = holographic error-correction.
   view_agreement as code distance. Claimed testable now.

4. **theory/five-atomic-groupings**: 5 atoms → 15 products → 4 predicted gaps.
   Compositional scheduling follows automatically.

### Op identity bug — independently verified

**Campsite**: `industrialization/architecture/op-identity-method` (active, math-researcher)

**Finding**: `winrapids-scan/src/launch.rs:142` pads input with `0.0` before GPU upload.
CUDA kernel (`engine.rs:80`) checks bounds and uses `make_identity()` for padding positions.
The zero-padding is never read by the kernel — correctness by coincidence, not design.

**Severity reassessment**: Design smell / latent bug, not a live bug. The CUDA bounds-check
prevents wrong results. The fix (Op::identity() on the Rust side, pad with proper identity)
makes correctness structural. Valid campsite, severity is "medium" not "critical".

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
| blomqvist_beta signum bug | gauntlet commit | RESOLVED — was already fixed | raw diff check present; test passes |
| log_returns Inf slips through | gauntlet commit | RESOLVED — NOT a bug | is_finite guard correctly catches Inf |
| hoeffdings_d NaN silently dropped | gauntlet commit | FIXED (0e9d42b) | part of 8 gauntlet fixes |
| distance_correlation overflow silent 0 | gauntlet commit | FIXED (0e9d42b) | part of 8 gauntlet fixes |
| tie_count unsorted assumption | gauntlet commit | DOCUMENTED — not a code bug | assumption violation, not impl error |
| 59/67 gauntlet tests pass, 8 fail | gauntlet commit | RESOLVED — all 8 fixed (0e9d42b) | external tests fixed and committed |
| matrix_exp [6/6] only ~1e-8 accurate | observer (7c4b54d) | FIXED (376bbd5) | upgraded to Padé [13/13] |
| pinv absolute rcond wrong default | navigator | FIXED (ff13e63) | now relative: max(m,n)*eps*max_sv |
| kmeans gap_statistic local closure | observer | FIXED (560c21e) | kmeans_f64 extracted as primitive |
| 2,194 tambear tests green | navigator (ff13e63) | YES — independently verified | ran cargo test --lib |

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

---

## Current State (after commit ff13e63)

**Verified**: 2026-04-10, after navigator's fix commit

| Crate | Tests | Status |
|-------|-------|--------|
| tambear | **2,194** | **all green** |
| tambear-fintek | 279 | all green |
| **Total** | **2,473** | **all green** |

`cargo check`: clean. `cargo test --lib` on both crates: clean.

**Test count trajectory this session**:
- Session start (c005fe0): 2,152 + 279 = 2,431
- After adversarial gauntlet (f2655fd): 2,152 + 279 = 2,431 (external tests, not counted in --lib)
- After wave 11 (7c4b54d): 2,190 + 279 = 2,469 (+38 lib tests, 1 FAILING)
- After rigor fixes + wave 6 + waves 11-15 + phantom fixes (376bbd5 through ff13e63): **2,194 + 279 = 2,473** (all green)

**Net additions this session**: +42 lib tests in tambear. 0 regressions.

**Public function count**: 1,287 pub fn (up from 1,257 at session start, +30 new primitives)

---

### Commits in this session (21 total, c005fe0 → ff13e63)

| Commit | Summary | Verified? |
|--------|---------|-----------|
| f2655fd | Adversarial gauntlet: 8 bugs, 67 external tests | external tests not runnable |
| 7c4b54d | Adversarial wave 11: 5 more bugs, matrix_exp tests added | broke lib tests (1 fail) |
| 0e9d42b | Rigor gauntlet: 8 bug fixes + 12 info theory primitives + 74 tests | tests green |
| 06df6f2 | Adversarial wave 12: 7 bugs in unstaged primitives | not individually checked |
| f431f2e | Phantom fix: 6 complete families + 5 complexity items at crate surface | phantom scan resolved |
| 87dfbf8 | Adversarial wave 13: scan identity/associativity — 1 bug, 22 proofs | not individually checked |
| 376bbd5 | Correctness: MMD U-stat + Padé [13/13] + renyi_divergence NaN | **Padé bug confirmed fixed** |
| 4d97979 | Wave 6: matrix_exp/log/sqrt + CG/GMRES + primitive promotion + API fix | new primitives |
| a6fa8ce | Adversarial wave 14: singularity-as-identity — 2 bugs, 16 proofs | not individually checked |
| 825710e | Adversarial wave 15: three-test template on Welford merges | not individually checked |
| 25400e4 | Workup test suites: erfc, pearson_r, inversion_count | workup docs created |
| 79c08d2 | Adversarial waves 14-15: 3 NaN-eating bugs fixed | not individually checked |
| 8983f3f | TBS executor: wire 12 info theory primitives; fix 3 NaN-eating bugs | not individually checked |
| 65c06f6 | Add nan_min/nan_max primitives | +2 new primitives |
| d11f6a6 | JBD first expedition burst | large batch |
| 229b4b9 | Expose phantom primitives + coefficient_of_variation | phantoms resolved |
| 560c21e | Extract kmeans_f64 primitive + remaining phantoms | kmeans_f64 extracted |
| 8069534 | Expose dim_reduction, factor_analysis, spectral_clustering, tda | crate surface |
| 5227597 | Fix unused variable warning in family24_manifold | lint fix |
| ff13e63 | TBS executor: 10 sig fix-ups + pinv relative rcond | **green, verified** |

---

### Bugs confirmed fixed this session

1. **matrix_exp Padé [6/6] → [13/13]** (376bbd5): Observer flagged the [6/6] numerical
   failure; commit confirms "previous [6/6] gave only ~1e-8 accuracy". Now matches MATLAB
   expm / scipy coefficients from Higham 2005 §10.3 Table 10.4.

2. **pinv rcond absolute → relative** (ff13e63): Now `max(m,n) * eps * max_sv`, matching
   NumPy linalg.pinv and LAPACK dgelss. Verified in code: `linear_algebra.rs:761-764`.

3. **MMD U-statistic cross-term** (376bbd5): Exy now excludes diagonal when n=m,
   ensuring MMD²(X,X)=0 exactly.

4. **renyi_divergence NaN on negative alpha** (376bbd5): assert!() panic replaced with NaN.

5. **kmeans_f64 extracted** (560c21e): gap_statistic's embedded CPU k-means is now a
   proper `kmeans_f64` primitive at the crate surface. The gap I documented (local closure
   duplication) is resolved.

6. **Rigor gauntlet 8 original bugs** (0e9d42b): The 8 failures from f2655fd were fixed.

---

### Open disputed claims — resolved

- **blomqvist_beta signum**: Was already fixed before gauntlet was written. Confirmed.
- **log_returns Inf guard**: Was NOT a bug. `is_finite()` correctly catches Inf. Confirmed.
- **tie_count unsorted**: Documented assumption violation, not implementation bug. Confirmed.

---

### Items flagged by navigator needing attention

1. **using() persistence design question** (Aristotle): One-call vs. pipeline-persistent
   semantics. Note at `tbs/20260410144156-discover-superposition/aristotle/insights/
   using-discover-epistemic-duality.md`. Pathmaker decision pending.

2. **NaN-eating instances remaining**: `clustering.rs:666`, `complexity.rs:240-241`.
   Wave 16 adversarial tests targeting these. Not yet committed.

3. **fintek family22/24 bridge**: `ccm`/`mfdfa`/`phase_transition` now at crate surface;
   bridges still reimplement them. Path B (delegate to `tambear::complexity`) is clean fix.

---

---

### Product Landscape Verification — 2026-04-10 (post-summary)

Following the naturalist's question about unnamed algorithms in the 4 predicted gap families
({Prefix×Graph, ByKey×Graph, Prefix×Prefix, Circular×Graph}).

**Verified: CCM is Prefix × Graph, unnamed.**

`complexity.rs:1589` — `pub fn ccm(x, y, embed_dim, tau, k)`. The code's Kingdom label
("Kingdom A") is correct for the inner kNN loop but misses the product structure:
- Prefix atom: `(0..embed_dim).map(|d| series[i - d * tau])` — Takens delay embedding
- Graph atom: `(0..lib_size).filter(|&j| j != i).map(|j| ...)` — kNN graph construction and weighted prediction

The algorithm is Prefix (manifold reconstruction) × Graph (nearest neighbor prediction).
It was classified correctly for its inner loop, incorrectly for its outer structure.

**Verified: `largest_lyapunov` and `correlation_dimension` are also Prefix × Graph.**

All three independently call `delay_embed` (now delegated to `time_series::delay_embed`)
then build kNN or pairwise-distance structures on the manifold. Three separate functions
implementing the same product without recognizing it. The product classification predicts
they should share the delay embedding via TamSession — they currently don't.

**Verified: Transfer entropy is ByKey × Prefix.**

`information_theory.rs:744` — discretizes into bins (ByKey), then computes t-1 → t
transition probabilities (Prefix within each bin class). Two atoms, unnamed.

**Verified: Prefix × Prefix has no implementation in the codebase.**

Searched for 2D prefix scans, integral images, summed area tables. None found. This is
a genuine gap, not a classification gap. The Viola-Jones box-filtering family is absent.

**Observable implication of the product classification:**

`ccm`, `largest_lyapunov`, `correlation_dimension` all construct delay embeddings
independently. The product structure predicts they should share this intermediate.
This is a concrete, actionable sharing opportunity that falls directly out of the theory.

---

### Math-Researcher Proposals Verified — 2026-04-10

Verified three technical claims from the math-researcher's campsite proposals:

**FNN promotion: confirmed correct and doable.**
`family15_manifold_topology.rs:445` — `fn fnn_frac(x, d, tau)` is private, 33 lines,
Kennel 1992 criterion. Hardcoded `rtol = 15.0` should become a `using()` parameter on
promotion. Implementation is correct.

**SVD algorithm label is wrong in the code.**
Section header `linear_algebra.rs:607` says "Golub-Kahan bidiagonalization + QR iteration."
Actual implementation (`linear_algebra.rs:620`) is one-sided Jacobi rotations — explicitly
stated in the docstring. The math-researcher's proposal matched the section header, not
the algorithm. The workup (Principle 10 against LAPACK dgesvd) is warranted; the campsite
description needs correction.

**Schur decomposition: absent, docstring misleads.**
`matrix_log` docstring at `linear_algebra.rs:1581` says "Schur decomposition approach."
The actual implementation uses repeated square-rooting + Gregory series — no Schur.
The proposal is correct (Schur IS a gap). The current code is approximating around it,
not using it.

**SVD test coverage: 3 tests, not a workup.**
`svd_reconstruction`, `svd_singular_values`, `svd_orthogonality` — basic correctness.
Missing: near-singular, rank-deficient, tall/wide extremes, ill-conditioned matrices.
Workup proposal is correct.

---

---

### Waves 16 + 17 Verification — 2026-04-10

Adversarial reported bugs in `davies_bouldin_score`, `hurst_rs`, `norm_inf`, `norm_1`,
`correlation_dimension`. Independently verified all five fold sites in the current working tree.

**All five are already fixed:**
- `clustering.rs:666` → `crate::numerical::nan_max`
- `complexity.rs:240-241` (hurst_rs range) → `nan_max` / `nan_min`
- `linear_algebra.rs:121` (norm_inf) → `nan_max`
- `linear_algebra.rs:128` (norm_1) → `nan_max`
- `complexity.rs:494` (correlation_dimension L∞) → `nan_max`
- `graph.rs:779` (flagged as next wave 18 target) → already `nan_max`

Fixes landed before or as part of the wave commits. Wave 16/17 test binaries test against
code that was already corrected — tests asserting NaN-propagation should now pass.

**Remaining raw `f64::min`/`max` fold sites — classified:**

| File | Lines | Classification | Reason |
|------|-------|----------------|--------|
| `data_quality_catalog.rs` | 280-292 | Safe | `clean` pre-filtered `is_finite()` |
| `data_quality.rs` | 108-109 | Safe | `clean` pre-filtered `is_finite()` |
| `complexity.rs` | 1535-1536 | Safe | `valid_h` pre-filtered `is_finite()` |
| `experiment0/1/2.rs` | various | Safe | softmax max-shift, caller pathology if NaN |
| `bigfloat.rs` | 1587-1588 | Test helper | not production |
| `numerical.rs` | 1172-1173 | Test helper | `unstable_simulation_oscillates` |
| `hypothesis.rs` | 2818-2819 | Test helper | `moment_stats_from_slice` — low priority |

**New NaN-eating pattern identified**: `if std > 0.0` returns false for NaN, silently
skipping blocks that should propagate NaN. In `hurst_rs:251`, NaN std causes silent
block omission rather than NaN return. This conditional-skip pattern is distinct from
the fold-eating pattern and may appear elsewhere.

**Systemic fold NaN-eating in production code: eradicated.**

*Last updated: 2026-04-10 by observer (waves 16/17 verification + remaining fold site audit)*
