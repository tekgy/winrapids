# Audit — `R:\tambear\` vs `R:\winrapids\crates\tambear\`

> **Generated:** 2026-05-10 by main-thread Claude prior to Sweep 36 planning.
> **Purpose:** Pre-flight scope-finding for the porting session. Numbers and file
> paths reflect a directory walk at audit time; the audit is descriptive,
> not prescriptive.
> **Methodology:** `ls` walk of both `src/` trees + `tests/` trees,
> `git log --oneline` of both repos (winrapids `tambear/` path-filtered),
> read of `R:\tambear\docs\decisions.md`, `R:\tambear\LOG.md`,
> `R:\tambear\docs\SWEEP_HISTORY.md`, `R:\tambear\docs\CURRENT_STATUS.md`,
> the Sweep 35 briefing, and the team briefing.

---

## 1. Executive summary

The two trees are **near-disjoint by component**. The split is structural:
the locked-vocabulary `R:\tambear\` is the **substrate / formalism / oracle
infrastructure** home, and the older `R:\winrapids\crates\tambear\` is the
**user-facing math (libm + statistics + pipelines + GPU JIT)** home. Sweeps
0-34 landed almost entirely in `R:\tambear\`; only Sweep 35 landed in
winrapids. Each repo has ~half the codebase the other lacks.

- `R:\winrapids\crates\tambear\src\` — **110,457 lines of Rust**, monolithic
  flat layout (`descriptive.rs`, `hypothesis.rs`, `time_series.rs`, etc.),
  pre-locked-vocabulary. Contains the *only* libm-family recipes
  (sin/cos/tan/asin/atan/erf/gamma/hyperbolic/exp/log/complex_log/...),
  the full statistics + ML + signal-processing + nonparametric flotilla,
  a working `.tbs` parser/executor stack, and three legacy JIT files
  (filter_jit, scatter_jit, tbs_jit).
- `R:\tambear\crates\tambear\src\` — **~140,000 lines of Rust** (post Sweep 35),
  tier-structured layout (`primitives/`, `recipes/`, `jit/`, `lattice/`,
  `proof/`, `dataprofile/`, `precheck/`, `knowledge/`). Contains the
  BigFloat primitive layer, the precision lattice, the BLAKE3-backed
  TamSession, the proof engine (Tier-A/B/C + ByDimensionalNyquistBoundary),
  the 8-executor accumulate dispatcher, the full sweeps oracle catalog,
  the multi-door JIT scaffold (cpu_cranelift + door + fingerprint +
  shape + strategy + jit_op + using_annotation), the antigen integration,
  the knowledge layer, plus tambear-tam / tambear-substrate / tambear-trace
  / tambear-experiment / tindex-mcp workspace crates.

**The headline porting gap is the libm tier** (winrapids → tambear). Sweep 35
landed phases A-D of the exp/log family in winrapids by accident-of-history;
the trig family has been in winrapids since Phase C3. Total libm code in
winrapids that needs to migrate: ~60 files (.rs + .spec.toml), ~6,300 lines
of Rust + ~120K of spec TOML.

**Tekgy and team-lead's framing in `project_two_tambears.md` matches what is
on disk.** The Sweep 35 addendum to that memory file lists exactly the
recipes/libm/ files that exist only in winrapids. The audit confirms no
silent drift: there is no third place where libm has been duplicated, and
no cross-port already in flight under a different name.

---

## 2. TL;DR component matrix

Legend: ✅ landed/canonical · 🚧 partial/legacy · ❌ absent

| Component | `R:\tambear\` | `R:\winrapids\crates\tambear\` | Port? | Rough effort |
|---|---|---|---|---|
| **Vocabulary lock** (Tier system) | ✅ canonical | 🚧 pre-lock layout | N (winrapids is archeology per DEC-004) | n/a |
| **Primitives — hardware (IEEE 754)** | ✅ 9 files | ✅ 9 files (same names) | Y, but trivial — `fused.rs` is 2x larger in tambear (12.4K vs 6.5K) | M |
| **Primitives — compensated** | ✅ `eft.rs` (21.6K) only | 🚧 `eft.rs` (11.6K) + `dot.rs` + `sums.rs` (separate) | Y for `dot.rs` + `sums.rs` if tambear's `eft.rs` doesn't subsume | S |
| **Primitives — double_double** | ✅ `ops.rs` 29.6K + `ty.rs` 9.8K | 🚧 `ops.rs` 11.5K + `ty.rs` 4.7K (older smaller) | N (tambear is newer/larger) | n/a |
| **Primitives — specialist** | ✅ 17 files (Kulisch, MomentStats, sketches GK/tdigest/DDSketch/KLL via HierarchicalLevels, PartitionSelect, observations, grid_2d, power_iteration, copa, deterministic_reservoir, running_median, quantile_binary_search, compressed_histogram, sum_k) | 🚧 9 files (Kulisch, observations, quantile_sketch, sketch_ddsketch, sketch_gk, sketch_kll, sketch_tdigest, sum_k) | N — tambear is the canonical home (Sweep 3 + 4E shipped) | n/a |
| **Primitives — big_float (BigFloat)** | ✅ 7 files, ~85K (BZ Algorithms 3.1/3.3/3.5/3.10 multi-limb) | ❌ winrapids has a single `bigfloat.rs` (99.5K) but unrelated lineage | N | n/a |
| **Primitives — bigint** | ❌ | 🚧 `bigint.rs` (39.1K) | Investigate whether tambear's BigFloat subsumes; otherwise port | M |
| **Primitives — oracle_compare** | ✅ `mod.rs` 42.7K (auto_within, allclose_auto, ulp_within, NanPolicy) | ❌ | N | n/a |
| **Primitives — clock** | ✅ scaffolded (2.9K) | ❌ | N | n/a |
| **Primitives — constants** | ❌ folder doesn't exist (constants live inline) | ✅ `mod.rs` (10.1K — PI/E/LN_2 multi-precision) | Y — port constants/ folder | S |
| **Primitives — oracle (test infra)** | ❌ | ✅ `algorithm_properties.rs` 16.7K + `mod.rs` 14.1K | Y — needed for proof engine cross-checks | M |
| **Lattice (PrecisionContext)** | ✅ `precision.rs` 17.8K + `tests.rs` 13.9K | ❌ | N | n/a |
| **Proof engine** | ✅ 25 files, ~6,237 lines (mod.rs 100.5K, 18 theorem files, ProofTier A/B/C, ByDimensionalNyquistBoundary, ByRefExternal) | 🚧 single `proof.rs` 116.6K / 3,002 lines (older, pre-tier) | N (tambear is the canonical Sweep 1 + 5 home) | n/a |
| **Accumulate substrate** | ✅ `accumulate.rs` 115K + `accumulate_exec.rs` 52.6K (8 grouping executors) | 🚧 inlined math across files (no atom layer) | N | n/a |
| **JIT infrastructure (multi-door)** | ✅ 10 files: cpu_cranelift (74.3K), door (60.8K), fingerprint (54.7K), shape (40.5K), jit_op (53.6K), strategy (13.9K), element_id (15.5K), using_annotation (30.8K), dispatcher (8.4K), mod (2.9K) | 🚧 filter_jit (27.1K), scatter_jit (55.6K), tbs_jit (26.1K) — pre-locked-vocab, NOT the same architecture | N (winrapids' is archeology); the legacy files inform reference but tambear is the canonical home (Sweep 8 in progress) | n/a |
| **Recipes — libm (transcendentals)** | ❌ **NO `recipes/libm/`** | ✅ **60 files** — .rs: sin/cos/tan/asin/atan/erf/exp/log/exp2/log2/exp10/log10/expm1/log1p/exp_kernel_state/gamma/hyperbolic/inv_hyperbolic/inv_recip/hypot/complex_log/pi_scaled/pi_scaled_inv/rare_trig/sincos/sincos_pi/adversarial; .spec.toml: 35 sibling specs (acos/acosh/acospi/acot/acsc/asec/asin/asinh/asinpi/atan/atan2/atanh/atanpi/cos/cosh/cospi/cot/csc/exp/gudermannian/haversin/inv_gudermannian/sec/sin/sincos/sincospi/sinh/sinpi/tan/tanh/tanpi/versin) | **Y — this is the headline gap** | **L** (~6,300 lines of Rust + 120K spec TOML, with adversarial harness + Sweep 35 ExpKernelState wiring) |
| **Recipes — elementary/ (proposed name)** | ❌ Sweep 35 briefing proposes `recipes/elementary/` for expm1/log1p but tambear has no such dir yet | n/a | Y — Sweep 35 phase A targets this path, but the work shipped to winrapids' `recipes/libm/` instead | (part of L above) |
| **Recipes — statistics (microstructure + risk)** | ❌ statistics recipes live as flat files under `recipes/` (cvar.rs, hill_estimator_streaming.rs, price_percentiles.rs) — that's it | ✅ `recipes/statistics/` 22 .rs files: amihud_illiquidity, cvar, hawkes_intensity, hill_estimator_streaming, hurst_exponent, jarque_bera, kurtosis_excess, kyle_lambda, lee_mykland_jump_count, max_drawdown, n_structural_breaks, parkinson_volatility, price_percentiles, realized_spread, realized_vol_subsampled, regime, roll_spread, skewness, stationarity_adf, volume_profile, vpin (~250K) | **Y — second-largest porting gap** | **L** (Tambear Contract compliance check per recipe — most are workflow recipes calling primitives that already exist in tambear) |
| **Recipes — statistics-only .spec.toml** | ❌ | ✅ `recipes/statistics/`: correlation_matrix.spec.toml + factor_analysis.spec.toml | Y | S |
| **Recipes — pipelines/** | ❌ | ✅ `recipes/pipelines/` 8 files: efa, invoke, mod, schema, serialize, shape, toml_schema, types (~140K — the `.tam` IR substrate) | **Y — important; depends on JIT topology decision** | L |
| **Recipes — music** | ✅ 5 submodules (cents_conversions, equal_temperament, key_inference, temperament_residuals, tonnetz) | ❌ | N | n/a |
| **Recipes — physics** | ✅ kernel_greens_function + weighted_aggregate | ❌ | N | n/a |
| **Recipes — ml** | ✅ mamba_selective_scan | 🚧 `neural.rs` 66.2K (pre-vocab, large) | Y (some — mostly composition recipes) | M |
| **Recipes — dimensional_nyquist** | ✅ 7 files (Sweep 6 6H) | 🚧 scattered references in older files | N | n/a |
| **Recipes — harness (experiment shells)** | ✅ 6 files (gi_pca, json, mod, nbody_hidden_variance, pca_cascade, shape_space) | ❌ | N | n/a |
| **Recipes — top-level recipes (Sweep 4-6 flat)** | ✅ 41 files at `recipes/*.rs` (ewma, ar1, garch, cvar, kll/gk/tdigest/ddsketch quantile, mp_null_test, pca_eigenvalues, spectral_structure_verdict, cascade_*, fold_detection, nilpotent_mixing_check, shannon_entropy, svd_top_k, standardize_per_dimension, ...) | 🚧 some equivalents live in `descriptive.rs`, `hypothesis.rs`, `nonparametric.rs` flat | N | n/a |
| **Recipes — beal_search / collatz_*** | ❌ | 🚧 5 files in winrapids (beal_search, collatz_parallel, collatz_search, collatz_structural, extremal_orbit, fold_irreversibility) | ?? — likely yes; tambear has Collatz *theorems* in proof/ but not the search recipes | M |
| **TamSession (intermediate sharing)** | ✅ `crates/tambear-tam/` 4 files: session.rs, computations.rs, tag.rs, lib.rs | ❌ | N | n/a |
| **Substrate / Trace** | ✅ `crates/tambear-substrate/` (parse 99.7K, query 61.9K, similarity 27.4K, trace 35.8K, edge/graph/id/node) + `crates/tambear-trace/` + `crates/tambear-trace-macros/` + `crates/tindex-mcp/` | ❌ | N — this is the cross-substrate provenance + trace MCP infra | n/a |
| **Datagen** | ✅ `datagen/` (mod.rs + structure.rs) — Sweep 7 promoted from testing fixtures | ❌ | N | n/a |
| **Dataprofile** | ✅ `dataprofile/mod.rs` 19.5K | ❌ | N | n/a |
| **Precheck** | ✅ `precheck/mod.rs` 16.5K | ❌ | N | n/a |
| **Knowledge layer (DEC-029)** | ✅ 8 files, ~64K (aggregate, canonicalize, impls, ingest, operation_history, trait_def, types, walk) | ❌ | N | n/a |
| **Stub macro** | ✅ `stub.rs` 2.5K via `inventory` | ❌ | N | n/a |
| **Antigens** | ✅ `antigens.rs` 8.6K — path-deps `R:/antigen` | ❌ | N | n/a |
| **Oracle catalog (`oracle/*/`)** | ✅ 16 dirs: TEMPLATE, centered_moment, cos, dd_subnormal_regimes, eft_primitives, ewma, exp, expm1, log, log1p, mean, moment_stats_merge, raw_moment, sin, tan, variance | ❌ | N (canonical home) | n/a |
| **Sweeps log (`sweeps/*/`)** | ✅ 30 dirs (00-17 + 23-34, skip 18-22) | ❌ | N | n/a |
| **Decision records (`docs/decisions.md`)** | ✅ 31 DECs ratified | ❌ (winrapids has scattered campsite ratifications + design docs) | N (canonical) | n/a |
| **Decision drafts (`docs/decisions/`)** | ✅ Folder for relocated drafts (DEC-032, DEC-033) | ❌ | N | n/a |
| **LOG.md** | ✅ append-only, 147 lines, all sweeps logged | ❌ winrapids' equivalent is the campsite logbook + git log | n/a | n/a |
| **CURRENT_STATUS.md / SWEEP_HISTORY.md / ARCHITECTURAL_INSIGHTS.md** | ✅ All present | ❌ | N | n/a |
| **`.tbs` executor (the TBS surface)** | ❌ | ✅ `tbs_executor.rs` 227.4K, `tbs_parser.rs` 18.4K, `tbs_lint.rs` 55.7K, `tbs_autodetect.rs` 20.5K, `tbs_advice.rs`, `tbs_io.rs`, `tbs_jit.rs`, `tbs_demo.rs` — the largest single-file in either repo | Y (substantial) but pending DEC on whether TBS is a Tier 5 surface in tambear's vocabulary | XL — wait until Sweep 24 (ide-tbs-live-integration) |
| **Experiment-related crates (tambear-experiment CLI)** | ✅ `crates/tambear-experiment/src/main.rs` 18K + tests | ❌ | N | n/a |
| **Adversarial test corpus** | ✅ 60+ test files including `big_float_adversarial_gauntlet.rs` 74.6K, `sweep_8_adversarial.rs` 269.6K, `sweep_8_r1015_attacks.rs` 264.7K | ✅ `adversarial_*` 30+ files (different content, pre-vocab) | Partial — some adversarial coverage might overlap; most should be regenerated against the new substrate | M-L |

---

## 3. Per-sweep table — where did each sweep land?

Cross-referenced from `R:\tambear\LOG.md`, `R:\tambear\docs\SWEEP_HISTORY.md`,
and `R:\winrapids\crates\tambear\` git log.

| Sweep | Slug | Landed where | Tests | Type | Status |
|---|---|---|---|---|---|
| 00 | op-redesign | `R:\tambear\` | 0 → 20 | Impl | COMPLETE |
| 01 | proof-harvest | `R:\tambear\` (port of winrapids `proof.rs`) | 20 → 77 | Port | COMPLETE |
| 02 | affine-compose | `R:\tambear\` | 77 → 109 | Impl | COMPLETE |
| 03 | yawni-primitives | `R:\tambear\` + `crates/tambear-tam/` | 109 → 180 → 1282 (closure) | Impl | COMPLETE |
| 04 | cross-cutting (4A-4L) | `R:\tambear\` | 180 → 411 | Mixed | COMPLETE (4A→S10, 4C→S11, 4F→S12 carry-fwd) |
| 05 | proof-tier-c | `R:\tambear\` | 411 → 585 | Impl | COMPLETE |
| 06 | workflow-diagnostics | `R:\tambear\` | 585 → 823 | Impl | COMPLETE |
| 07 | experiment-harness | `R:\tambear\` + new `tambear-experiment` crate | 824 → 922 | Impl | COMPLETE |
| 08 | tam-gpu-core | `R:\tambear\` (in progress; 8A/8B/8C landed) | — | Impl | IN PROGRESS |
| 09 | k-particle-variants | n/a | — | — | PLANNED, blocked by 8 |
| 10 | oracle-bridge (rescoped) | n/a | — | — | PLANNED |
| 11 | erfinv-oracle | n/a | — | — | PLANNED |
| 12 | validity-dispatcher (Sweep 12B partial) | `R:\tambear\` (12B subset shipped) | — | Impl | 12B substantial (11/11 ATKs) |
| 13 | autodiff | n/a | — | — | PLANNED, Tekgy approval gate |
| 14 | op-redesign (λ-continuum) | n/a | — | — | SCOPING (reopens DEC-007) |
| 15 | music-theory-expansion | n/a | — | — | PLANNED |
| 16 | distance-divergence-family | n/a | — | — | PLANNED |
| 17 | jit-gpu-doors | n/a (subsumed by S8?) | — | — | PLANNED |
| 18-22 | — | (gap in numbering) | — | — | not seeded |
| 23 | pipeline-compiler-transparency | `R:\tambear\` (DEC-028 RoutingRecord ratified during this work) | — | Impl | aristotle authored from S23 work |
| 24 | ide-tbs-live-integration | n/a | — | — | PLANNED |
| 25 | academic-writeup-live | n/a | — | — | PLANNED |
| 26 | hardware-orchestration-surface | n/a | — | — | PLANNED |
| 27 | live-data-quality-analysis | `R:\tambear\` (`sweep_27_precheck_adversarial.rs` test, 31.5K) | — | Impl | Adversarial coverage shipped |
| 28 | results-surface | n/a | — | — | PLANNED |
| 29 | moment-stats-v2 (incl 29b/29c) | `R:\tambear\` (29c kappa-sweep 0 ULP gate `f53bc1c`) | — | Impl | COMPLETE for kappa-sweep portion |
| 30 | dec-030-impl-symbolic-lattice | `R:\tambear\` (DEC-030 v3 ratified 2026-05-06) | — | Decision + scaffold | DEC ratified; impl seeded |
| 31 | dec-031-impl-precision-lattice | `R:\tambear\` (BZ Algorithms + lattice + BigFloat) | 1413 → 1560 lib + 2288 total | Impl | COMPLETE (Tier 1 primitive layer) |
| 32 | dec-032-product-lattice (cache-key) | `R:\tambear\` (`fa49fec` feed_precision_context + cache_key_with_precision; IR_VERSION 9→10) | — | Impl | COMPLETE |
| 33 | tier-4-5-protocol (TAM routing) | `R:\tambear\` (`e1d0e30` Shape::precision; CPU dispatch for BigFloat) | — | Impl | COMPLETE |
| 34 | antigen-integration-deepening (BigFloat-as-gold oracle prep) | `R:\tambear\` (`oracle/{sin,cos,tan,log,exp}/` corpora + harness; `c95f6f6` BigFloat-as-gold log + variance) | — | Oracle-prep | COMPLETE |
| **35** | **libm-factoring (exp/log + complex_log)** | **`R:\winrapids\crates\tambear\`** (commits `1b884cf` → `f217b87`: phases A-D) + `R:\tambear\` (tameness fixes `589dbe9`, tan follow-ups `f89c9eb`) | — | Impl | **LANDED IN WRONG REPO** for the recipes; tameness side in correct repo |

The discontinuity at Sweep 35 is the audit's primary structural finding:
the **recipe content** (expm1.rs, log1p.rs, exp_kernel_state.rs, exp.rs,
exp2.rs, exp10.rs, log2.rs, log10.rs, hyperbolic.rs, inv_hyperbolic.rs,
hypot.rs, complex_log.rs) shipped to `R:\winrapids\crates\tambear\src\recipes\libm\`,
while the **tameness fixes + oracle harness + tan follow-ups doc** shipped to
`R:\tambear\`. This is consistent with the Sweep 35 *briefing* targeting
`crates/tambear/src/recipes/elementary/expm1.rs` (the Sweep 35 doc says
`R:\winrapids\crates\tambear\src\recipes\elementary\`, suggesting the team
read "the canonical tambear crate" as winrapids' tambear — the historical
home of libm).

---

## 4. Per-domain narrative

### 4.1 BigFloat primitive layer (the Sweep 31 substrate)

**`R:\tambear\crates\tambear\src\primitives\big_float\`** is the canonical
home. Files: `mod.rs` (2.2K), `ty.rs` (18.1K — `BigFloat` struct + `BigFloatKind`
incl. `NaN { payload: u64 }`), `conversions.rs` (26.1K — f64↔BF, DD↔BF with
the diamond `lo=0` short-circuit), `cmp.rs` (14.8K — `total_cmp` with NaN at top),
`arith.rs` (69.4K — BZ Algorithms 3.1/3.3/3.5/3.10 multi-limb), `limbs.rs`
(13.9K — primitive limb ops). Each has a sibling `*_tests.rs` (~6-16K). Tests
include `big_float_adversarial_gauntlet.rs` (74.6K), `big_float_vs_mpmath.rs`
(88K — oracle harness), `big_float_cross_precision.rs` (59.5K — Phase C
proptests).

Winrapids has an unrelated `bigfloat.rs` (99.5K) in the legacy flat layout.
Per `project_two_tambears.md` and DEC-004, that file is archeology. Do not port.

### 4.2 JIT infrastructure (multi-door, holonomic, per DEC-019)

**`R:\tambear\crates\tambear\src\jit\`** is the canonical home. Files
(post-Sweep 8 in-progress + Sweep 32/33 work):
- `cpu_cranelift.rs` (74.3K) — first door; Cranelift codegen
- `door.rs` (60.8K) — DoorBackend trait + per-door routing
- `fingerprint.rs` (54.7K) — content-addressed cache key with
  `feed_precision_context`, `feed_branch_policy(0x1B)`, IR_VERSION pinned at 10
- `shape.rs` (40.5K) — Shape struct with `precision: Option<PrecisionLevel>`
- `jit_op.rs` (53.6K) — finite Op enum lowered (not dyn OpKind) per DEC-019
- `strategy.rs` (13.9K) — strategy selection (default/tuned)
- `element_id.rs` (15.5K) — element provenance ID
- `using_annotation.rs` (30.8K) — UsingBag + intra-tier conflict detection
- `dispatcher.rs` (8.4K) — top-level dispatch
- `mod.rs` (2.9K)

Winrapids' JIT files (`filter_jit.rs`, `scatter_jit.rs`, `tbs_jit.rs`) are
pre-locked-vocabulary, pre-DEC-019. They represent the *prior* JIT approach
(WGSL + cuda-via-cudarc + tbs-specific). Tambear's JIT is a clean rewrite.
Do not port code; the *concepts* (filter dispatch, scatter dispatch) are
absorbed into `jit_op.rs` as Op variants.

### 4.3 Oracle harnesses (test-against-reference infrastructure)

**`R:\tambear\oracle\`** is the canonical home. 16 dirs:
- Sweep 34 transcendentals: `sin/`, `cos/`, `tan/`, `log/`, `exp/`,
  `expm1/`, `log1p/`
- Sweep 4E + 31 statistics + numerics: `mean/`, `variance/`,
  `centered_moment/`, `raw_moment/`, `moment_stats_merge/`, `ewma/`,
  `eft_primitives/`, `dd_subnormal_regimes/`
- `TEMPLATE/`, `README.md`, `LIFT_EQUIVALENCE_TEMPLATE.md`

Tan has the `followups-rederived-2026-05-09.md` doc explicitly attributed
as re-derivation (per Standing Constraint #11). The harnesses use mpmath
at 50-digit precision as gold, with the Sweep 34 work-stream upgrading
variance + log to use tambear's own BigFloat-at-500 as gold instead.

Winrapids does not have an `oracle/` directory. It has the `primitives/oracle/`
substrate (`algorithm_properties.rs` 16.7K + `mod.rs` 14.1K) — a
**shared test infrastructure catalog** of algorithm weakness properties. That
catalog needs porting to `R:\tambear\crates\tambear\src\primitives\oracle\`
(folder does not exist in tambear; the **constants** subfolder also doesn't
exist — constants live inline in tambear).

### 4.4 Libm recipes (the headline gap)

**`R:\winrapids\crates\tambear\src\recipes\libm\`** is the *only* home for
the transcendental recipe family. 60 files split into:

- **.rs implementations (Sweep 35 added the bottom 12):**
  - asin.rs (9.0K — fdlibm lineage, 2026-04-23 P_S2/P_S5 fix)
  - atan.rs (11.3K)
  - sin.rs (33.1K — Remez + Payne-Hanek)
  - tan.rs (16.6K)
  - erf.rs (17.9K)
  - gamma.rs (11.5K)
  - hyperbolic.rs (12.4K — sinh/cosh/tanh; updated in Sweep 35 phase C)
  - inv_hyperbolic.rs (8.9K — asinh/acosh/atanh)
  - inv_recip.rs (5.3K)
  - pi_scaled.rs (10.3K — sinpi/cospi/tanpi)
  - pi_scaled_inv.rs (13.6K — asinpi/acospi/atanpi)
  - rare_trig.rs (6.5K — sec/csc/cot/versin/haversin/gudermannian)
  - sincos.rs (4.6K)
  - sincos_pi.rs (1.9K)
  - adversarial.rs (27.0K — test harness)
  - mod.rs (1.8K)
  - exp.rs (18.3K — Sweep 35 phase C updated)
  - log.rs (11.6K)
  - exp2.rs (12.9K), exp10.rs (8.4K), log2.rs (7.0K), log10.rs (6.0K)
  - **expm1.rs (22.9K — Sweep 35 phase A)**
  - **log1p.rs (20.1K — Sweep 35 phase A)**
  - **exp_kernel_state.rs (17.8K — Sweep 35 phase B; first TamSession-registered kernel state)**
  - **hypot.rs (9.0K — Sweep 35 phase C complementary-argument-transform)**
  - **complex_log.rs (19.7K — Sweep 35 phase D; first complex-transcendental with BranchPolicy)**

- **.spec.toml specs (35 files, declarative recipe surface, Phase D5 pilot lineage):**
  - acos, acosh, acospi, acot, acsc, asec, asin, asinh, asinpi, atan, atan2,
    atanh, atanpi, cos, cosh, cospi, cot, csc, exp, gudermannian, haversin,
    inv_gudermannian, sec, sin, sincos, sincospi, sinh, sinpi, tan, tanh,
    tanpi, versin

**Tambear lacks `recipes/libm/` entirely.** Sweep 35 briefing proposes
`recipes/elementary/` as the path. The recipe layer doesn't exist at all in
tambear — the closest analog is the 41 flat files at `recipes/*.rs`, none of
which are transcendentals.

### 4.5 Statistics recipes (second-largest gap) — CORRECTION 2026-05-12

**IMPORTANT: the original audit claim of "22 .rs files in `recipes/statistics/`" was a hallucination.** Scout (Sweep 37) verified the actual directory state. Main-thread confirmed via direct `ls`:

**Verified state of `R:\winrapids\crates\tambear\src\recipes\statistics\`** (2026-05-12):
- `correlation_matrix.spec.toml` (5.8K)
- `factor_analysis.spec.toml` (8.2K)
- `mod.rs` (608 bytes)

That's it. **NO .rs recipe files** in `recipes/statistics/`. The audit hallucinated the 22-file list.

**Where the actual statistics functions live** (`crates/tambear/src/` flat monoliths):
- `descriptive.rs` — 100.5K
- `hypothesis.rs` — 132.8K
- `nonparametric.rs` — 165.1K
- `time_series.rs` — 145.0K
- `volatility.rs` — 69.5K

**Total: ~613K of flat statistics code** (much bigger than the audit's "~250K" claim). The 22 named functions the audit listed (amihud_illiquidity, kyle_lambda, vpin, parkinson_volatility, etc.) DO exist — but as functions WITHIN these flat monolithic files, not as separate `.rs` recipe files.

**Implication for Sweep 38 planning**: scope is NOT "port 22 organized files." Scope IS "extract individual recipes from 5 flat monoliths totaling ~613K + fresh-write each in tambear idiom." Bigger work than the audit suggested. Each monolith contains many functions; extracting and naming them as Tier 4 recipes (per CLAUDE.md's flat recipe principle) is the first phase of Sweep 38.

**Duplicates noted in original audit**: tambear has its own `cvar.rs` (8.8K), `hill_estimator_streaming.rs` (9.2K), `price_percentiles.rs` (6.9K) at `crates/tambear/src/recipes/`. Whether those tambear versions are strict supersets of the winrapids-flat-monolith equivalents requires per-function diff; the original audit's claim of "two duplicates" was based on the hallucinated 22-file list, so the duplicate analysis needs re-doing during Sweep 38 planning.

**Provenance of the correction**: scout (Sweep 37 Phase 0 downstream survey) caught the discrepancy 2026-05-12; navigator routed to team-lead; main-thread verified via direct `ls` 2026-05-12 19:47Z. This is an instance of substrate-over-memory at the *which-substrate* level — even an audit doc's claims about substrate need re-verification when the team encounters them.

### 4.6 Pipelines

**`R:\winrapids\crates\tambear\src\recipes\pipelines\`** is the *only* home
for the pipeline (.tam IR) substrate. 8 files (~140K):
- efa.rs (13.1K — first pilot pipeline, six-step EFA)
- invoke.rs (18.3K — single FFI dispatch surface)
- schema.rs (42.5K — pipeline schema)
- serialize.rs (15.9K — pipeline serialization)
- shape.rs (15.8K — pipeline-level shape)
- toml_schema.rs (20.0K — TOML schema)
- types.rs (27.0K — pipeline types)
- mod.rs (4.6K)

**Tambear has no `recipes/pipelines/` directory.** The pipeline tier (Tier 5)
is described in vocabulary.md as "compiles to .tam IR + per-pass per-door
kernel binaries" but the implementation is in winrapids only. Porting will
depend on locking the JIT topology first (Sweep 8) and the routing record
(DEC-028) — pipelines compose recipes that compose atoms that lower through
the JIT to doors.

### 4.7 Proof engine

**`R:\tambear\crates\tambear\src\proof\`** is the canonical home (Sweeps 1, 5, 6).
25 files, ~6,237 lines:
- mod.rs (100.5K — core engine with ProofTier A/B/C, ByDimensionalNyquistBoundary,
  ByRefExternal, 7 new StructuralFact variants from Sweep 5)
- collatz.rs (16.6K) + 4 specialized collatz proof files (5 total)
- dimensional_nyquist_theorems.rs (22.1K) + helpers
- music_theory_theorems.rs (11.5K)
- 8 pith theorem files (liftability, accumulation_operator_boundary,
  branching_liftability, chenciner_montgomery_figure_8, lambda_continuum,
  stratified_liftability, ...)
- rank_n.rs (8.1K) + svd_condition.rs + naive_formula.rs + nilpotent_mixing.rs
  + photon_projection_is_mp_flat.rs + shape_sphere_identity.rs +
  inverse_special_functions.rs + msr_polynomial_degree.rs +
  moment_stats_mergeability.rs

Winrapids has a single `proof.rs` (116.6K, 3,002 lines) — the *source* that
Sweep 1 ported. The winrapids version has the original `tambear_context()`,
`collatz_four_pillars()`, `rank_n_accumulator()` entry points but predates
the Tier-A/B/C extension. Per DEC-008 and DEC-004, winrapids' is archeology.

### 4.8 Methodology / decision / architecture docs

`R:\winrapids\docs\architecture\` is where the design docs that govern *both*
repos live (vocabulary.md is the canonical reference, also mirrored in
`R:\tambear\docs\vocabulary.md` 48.1K). Key docs (winrapids-rooted):
- `vocabulary.md` (locked 2026-04-17)
- `holonomic-architecture.md` (named + ratified 2026-05-08)
- `branch-cut-conventions.md` (DEC-032 spec)
- `tambear-libm-factoring.md` (Sweep 35 design synthesis)
- `internal-tameness-contracts.md` (audit pattern from Sweep 31 finish)
- `confident-wrong-narratives.md`
- `atoms-primitives-recipes.md`
- `recipe-trees-continuation` (sketches family pilot)

Sweep 35 also produced **`docs/expedition/session-methodology-patterns.md`**
(four reusable recognition tools).

These docs are referenced from `R:\tambear\docs\decisions.md` but live in
winrapids. No porting needed — but a question for team-lead: do the
design docs *want* to migrate to `R:\tambear\docs\architecture\` to fully
consolidate?

### 4.9 Top-level workspace crates in `R:\tambear\`

Five crates beyond `tambear`:
- `tambear-tam/` — TamSession + IntermediateTag (Sweep 3 substrate)
- `tambear-substrate/` — provenance graph layer (parse 99.7K, query 61.9K,
  similarity 27.4K, trace 35.8K, build 28.5K — substantial; lands DEC-029
  Knowledge-adapter)
- `tambear-trace/` — runtime trace
- `tambear-trace-macros/` — macro support for trace
- `tambear-experiment/` — CLI binary (Sweep 7, renamed from tambear-measure)
- `tindex-mcp/` — index MCP server (21K)

Winrapids has no equivalent workspace structure — its tambear is a single
crate at `crates/tambear/`. The tambear-substrate stack landed 2026-04-25
per memory file `project_substrate_tindex_trace.md`.

---

## 5. DEC inventory (every DEC in `R:\tambear\docs\decisions.md`)

31 DECs ratified, 1 numbered slot (DEC-004) reserved for repo split.

| # | Title (one-line) | Impl evidence in tambear | Impl in winrapids |
|---|---|---|---|
| 001 | No vendor math libraries — driver only | DEC-019 elaborates; no vendor crates in Cargo.toml | DEC-001 also enforced (cudarc + PyO3 only) |
| 002 | No tech debt — see it, fix it | Standing principle; every sweep | Standing principle |
| 003 | YAWNI — You Always Will Need It (anti-YAGNI) | Pre-allocation discipline across sweeps 0-7 | Original home of the principle |
| 004 | Fresh consolidation home at R:\tambear\, winrapids = archeology | THE split decision; tambear IS the impl | (winrapids = archeology by this DEC) |
| 005 | Notification discipline | LOG.md (no Pushover overuse) | n/a |
| 006 | Locked vocabulary is non-negotiable | vocabulary.md mirrored in tambear; structure enforced | Drifted; banners added |
| 007 | Garden Op-redesign vectors land in Sweep 0 | accumulate.rs OpKind trait | n/a |
| 008 | Proof engine ports as-is, then gets canonical_structure() hook | proof/mod.rs + canonical_structure on all 9 Ops | Source (`proof.rs` 116.6K) |
| 009 | Integration-test-as-proof pattern | Every sweep ships `tests/sweep_N_loop.rs` | n/a |
| 010 | Resolution C for tuple-state Ops | OpKind tuple-state in accumulate.rs | n/a |
| 011 | Pith and findmcp ARE math-relevant | pith theorems ported (8 files in proof/) | Pith dimensional-lifting-proofs.md (source) |
| 012 | Default quantile interpolation = R type 7 linear | recipes/quantile_exact.rs | recipes/libm/... |
| 013 | Co-native design empirically validated by findmcp | architecture documents | n/a |
| 014 | RunningMedianObservations warmup output = NaN | primitives/specialist/running_median.rs | n/a |
| 015 | Iterative deepening over monolithic spec | Sweep pattern across all sweeps | n/a |
| 016 | Free additions per garden insight pattern | joint_histogram free addition shipped | n/a |
| 017 | Two atoms are immutable | accumulate + gather only; no third | n/a |
| 018 | Proof Tier C — structural-statistical proofs are first-class | proof/photon_projection_is_mp_flat.rs etc. | n/a |
| 019 | Native-door JIT — no middleware, direct driver APIs | crates/tambear/src/jit/ (cpu_cranelift first door) | filter_jit/scatter_jit/tbs_jit (pre-DEC; use middleware) |
| 020 | State conservation — every behavior-affecting state representable | Knowledge layer + state types | n/a |
| 021 | Three knowledge layers and three caches | crates/tambear/src/knowledge/ (8 files) | n/a |
| 022 | Substrate discipline — type-enforced invariants | DEC-029 impl + antigens.rs | n/a |
| 023 | Floating-point associativity — bit-exact tests, reported trade-offs | Kulisch merge tests; sweep_8_adversarial | n/a |
| 024 | Cache is best-effort optimization — never a promised contract | TamSession docs | n/a |
| 025 | Defaults are rigor incarnate | Filter Test §4 | n/a |
| 026 | All views are projections of one source — bidirectional binding | tambear-substrate query/similarity | n/a |
| 027 | FMA3 ISA feature is a cache-key axis — no libm, no silent precision split | fingerprint.rs feeds ISA | n/a |
| 028 | Routing conservation — every resolver emits an auditable RoutingRecord | Sweep 23 work | n/a |
| 029 | Knowledge-adapter — evidence-convergence across modalities | crates/tambear/src/knowledge/ (8 files, ~64K, ratified 2026-04-24) | n/a |
| 030 | Refinement-lattice — granularity-convergence across the Symbolic stack | sweep-30 seeded; impl pending | n/a |
| 031 | Precision-lattice — quality-cost gradient across f64 ↔ DD ↔ BigFloat(prec) | lattice/precision.rs + primitives/big_float/ (Sweep 31 complete) | n/a |
| 032 | (Relocated to docs/decisions/) Branch-cut conventions for complex transcendentals | Used by sweep-35 complex_log.rs (in winrapids) + fingerprint.rs feed_branch_policy(0x1B) | sweep-35 complex_log.rs |
| 033 | (Relocated to docs/decisions/) TamSession dedupe content-addressed at recipe tier | DEC-033 in `docs/decisions/`; holonomic architecture | n/a |

**Observation:** DEC-032 and DEC-033 are the only DECs whose implementation
straddles repos (DEC-032 because Sweep 35 complex_log landed in winrapids).
All other DECs are tambear-side, by design (DEC-004).

---

## 6. Recommended port order for Sweep 36+

Given the audit findings, the natural sequencing is:

### Phase 1 — Sweep 35 reconciliation (smallest, urgent)

The Sweep 35 work that landed in winrapids needs to migrate to tambear so
that the canonical libm story starts from `recipes/elementary/` in the
right repo. Items:

1. **expm1.rs + log1p.rs** (Sweep 35 phase A) → `R:\tambear\crates\tambear\src\recipes\elementary\`
2. **exp_kernel_state.rs** (Sweep 35 phase B) → same path; wire to TamSession
   (already exists in `crates/tambear-tam/`)
3. **exp.rs + log.rs + exp2/log2/exp10/log10** (Sweep 35 phase C) → `recipes/elementary/`
4. **hyperbolic.rs + inv_hyperbolic.rs + hypot.rs** (Sweep 35 phase C) → `recipes/elementary/`
5. **complex_log.rs** (Sweep 35 phase D) → `recipes/elementary/`; depends on
   BranchPolicy enum existing in tambear (DEC-032 ratified; structural fingerprint
   `feed_branch_policy(0x1B)` already in tambear's `fingerprint.rs`)

**Blockers:** none — DEC-032 + DEC-033 already ratified; ExpKernelState's
TamSession registration already has the substrate ready. Estimated 1-2 days.

### Phase 2 — Libm trig family port (the planned Sweep 35 antecedent)

After Phase 1, the harder port: the trig + erf + gamma + pi-scaled family
that's lived in winrapids since the Phase B/C/D pre-vocab era.

6. **sin.rs (33.1K)** — biggest single file; Payne-Hanek + Remez. Tambear has
   no Payne-Hanek primitive; may need new compensated/ helpers.
7. **cos.rs / tan.rs** — share kernel state with sin per Sweep 35 TrigKernelState pattern
8. **asin.rs / atan.rs / atan2 (via .spec.toml)** — fdlibm-lineage; 2026-04-23 fix preserved
9. **erf.rs (17.9K)** — fdlibm rational approximation
10. **gamma.rs (11.5K)** — Lanczos approximation
11. **pi_scaled.rs / pi_scaled_inv.rs / rare_trig.rs** — sin_pi/cos_pi/tan_pi family + sec/csc/cot/versin/haversin/gudermannian
12. **sincos.rs / sincos_pi.rs** — shared-kernel variants
13. **inv_recip.rs / inv_hyperbolic.rs** (the latter already in Sweep 35 Phase 1 above)
14. **35 .spec.toml files** — declarative recipe surface; depends on .spec.toml infrastructure in tambear (also missing)
15. **adversarial.rs (27.0K)** — port to `tests/libm_adversarial.rs` or split per-recipe

**Blockers:** (a) tambear needs a `.spec.toml` schema reader (or this is
deferred until Sweep 23/24 ide-tbs-live-integration); (b) `primitives/constants/`
needs to exist in tambear (winrapids has it at `primitives/constants/mod.rs`
10.1K — port first); (c) `primitives/oracle/` substrate (`algorithm_properties.rs`
+ `mod.rs`) needs to exist in tambear.

Estimated 5-8 days for full trig family port (including audit against
Tambear Contract).

### Phase 3 — Statistics + microstructure recipes

16. **22 statistics recipes** (amihud, kyle_lambda, vpin, parkinson, etc.) —
    most are workflow recipes composing primitives that already exist in
    tambear. Reconcile the duplicates first (cvar, hill_estimator_streaming,
    price_percentiles already in both).
17. **2 .spec.toml** (correlation_matrix, factor_analysis) — depends on spec
    infrastructure (Phase 2 blocker (a))

Estimated 3-5 days.

### Phase 4 — Pipelines tier

18. **`recipes/pipelines/`** (8 files, ~140K) — depends on JIT topology lock
    (Sweep 8 finalization) and DEC-028 RoutingRecord. Pipelines compose recipes
    that compose atoms that lower through the JIT to doors.

Estimated 1-2 weeks; can run in parallel with Sweep 8 finalization.

### Phase 5 — `.tbs` surface

19. **tbs_executor.rs (227.4K) + tbs_parser/lint/autodetect/io/jit** — wait
    until Sweep 24 (ide-tbs-live-integration) is scoped; the TBS surface is
    Tier 5 vocabulary (or doesn't exist in the locked tier system at all).

Estimated 2-4 weeks.

### Phase 6 — Misc

- Collatz search recipes (beal_search, collatz_parallel, etc.) — investigate
  whether tambear's proof/collatz_* covers them or if they're separate
  workflow recipes.
- ML recipes from winrapids `neural.rs`, `train/*` — most should map to
  tambear's `recipes/ml/` (currently only mamba_selective_scan).
- BigInt — investigate whether tambear's BigFloat subsumes; otherwise port
  separately.

---

## 7. Choke-points and dependency graph

```
Sweep 36 (libm reconciliation, Phase 1)
   ├── needs: BranchPolicy enum in tambear (DONE — fingerprint.rs)
   ├── needs: TamSession.register_kernel_state (DONE — tambear-tam)
   └── unblocks: trig family port (Phase 2)

Phase 2 (trig port)
   ├── BLOCKED by: primitives/constants/ port (S)
   ├── BLOCKED by: primitives/oracle/ port (M)
   ├── BLOCKED by: .spec.toml schema reader OR defer (L)
   └── unblocks: statistics recipes (Phase 3 partial)

Phase 3 (statistics recipes)
   ├── reconcile cvar/hill/price_percentiles duplicates
   └── most workflow recipes — direct port

Phase 4 (pipelines)
   ├── BLOCKED by: Sweep 8 finalization (the largest unknown)
   └── BLOCKED by: DEC-028 RoutingRecord finalization

Phase 5 (TBS surface)
   └── BLOCKED by: Sweep 24 scoping
```

**The biggest single dependency is the `.spec.toml` schema decision.**
35 of 60 libm files in winrapids are .spec.toml — declarative recipes that
target a TOML-driven recipe-definition layer. Tambear has no equivalent
infrastructure. Three options:
- (a) Build the .spec.toml reader in Sweep 36 (large scope; touches Sweep 24)
- (b) Convert each .spec.toml back to a hand-written .rs recipe (medium scope; loses the declarative win)
- (c) Defer the .spec.toml-bound recipes (acos/atan2/cosh/etc.) until Sweep 24 lands

**Recommended:** option (c) for Sweep 36; port the 25 .rs-only files first,
defer the .spec.toml-bound ones.

---

## 8. What the audit cannot tell us

Honest gaps in this audit:

1. **Whether tambear's `recipes/cvar.rs` (8.8K) is a strict superset of
   winrapids' `recipes/statistics/cvar.rs` (7.3K).** File sizes differ, but
   the difference could be vocabulary updates, docstring expansion, or genuine
   functional divergence. A diff is needed before deduplication. Same applies
   to `hill_estimator_streaming` and `price_percentiles`.
2. **Whether winrapids' `bigfloat.rs` (99.5K) has any unique correctness
   work that wasn't independently re-derived in tambear's `primitives/big_float/`.**
   Per DEC-004 winrapids is archeology, but a final read-through before
   declaring it dead would be paranoid-prudent.
3. **The contents of sweeps 18-22.** The numbering gap suggests these were
   never seeded (no dirs in `R:\tambear\sweeps\`). The team-briefing and
   SWEEP_HISTORY don't mention them. Likely just sparse sweep IDs (sweeps 8-9
   pulled ahead of GPU work, then 10+ in oracle/erfinv direction, then jumped
   to 23 with the Sweep 23 pipeline-compiler work). Worth confirming with
   Tekgy in case there's a hidden roadmap.
4. **The exact relationship between `R:\winrapids\crates\tambear\src\recipes\pipelines\efa.rs`
   (winrapids' EFA pipeline pilot) and any Tier-5 pipeline machinery in tambear.**
   Tambear has `recipes/harness/` for experiments but no `recipes/pipelines/`.
   The Phase D pipeline pilot work in winrapids may be load-bearing for
   Sweep 8 finalization.
5. **Whether the 30+ `adversarial_*.rs` test files in winrapids
   (`adversarial_boundary.rs` through `adversarial_wave22.rs` etc.) have
   coverage that tambear's `sweep_*_adversarial.rs` doesn't already replicate.**
   The volume of adversarial code in winrapids is enormous (~1.5MB of test
   code, much of which is corpus); spot-checking would take days.
6. **The status of `R:\winrapids\crates\tambear\data\` and `target2/` / `target3/`.**
   These are likely build artifacts + data fixtures, not source. Did not
   explore.
7. **The relationship to `R:\fintek\` and `R:\pith\` and `R:\findmcp\`.**
   Both repos reference these; this audit only walked the two tambear trees.
   Sister-repo dependencies may impose porting constraints not visible here.
8. **Whether tambear's `tests/sweep_35_complex_log_branch_cut.rs` (26.1K) and
   `sweep_35_exp_log_cross_precision.rs` (28.6K) are *the* Sweep 35 tests
   or test stubs that need population.** Their existence is consistent with
   either reading; the files are large enough to be substantial but
   without diffing against `R:\winrapids\crates\tambear\tests\` we can't be sure.
9. **The actual line count of `R:\tambear\crates\tambear\src\` total.**
   The audit estimated ~140K based on counting major files but did not run
   `find ... -name '*.rs' | xargs wc -l` exhaustively. The headline number
   may be off by 20-30%.

---

## 9. Auditor's two-cent recommendation (not a Sweep 36 scope statement)

Based on the dependency graph in §7 and the gap inventory in §8, the
natural opening move for Sweep 36 is **Phase 1 (Sweep 35 reconciliation,
12 files into `recipes/elementary/`)** — small, decisive, removes the
audit's primary structural finding before it compounds. The next move is
**`primitives/constants/` port (1 file, ~10K, trivial)** which unblocks all
of Phase 2.

The larger trig + statistics port is a multi-week arc, but it does not need
to start until Sweep 8 finalization unblocks the pipelines tier. Phase 1
+ constants port could close in a single Sweep 36 session, with a clean
handoff to Sweep 37 (which would scope Phase 2).

The .spec.toml question is the most consequential strategic call. Team-lead
+ Tekgy should make that call before Sweep 37, not during.

---

## 10. Appendix — file paths to pin for Sweep 36 planning

Read these first:
- `R:\winrapids\crates\tambear\src\recipes\libm\` — the 60 libm files
- `R:\winrapids\crates\tambear\src\recipes\statistics\` — 22 .rs + 2 .spec.toml
- `R:\winrapids\crates\tambear\src\recipes\pipelines\` — 8 files
- `R:\winrapids\crates\tambear\src\primitives\constants\mod.rs` — to port
- `R:\winrapids\crates\tambear\src\primitives\oracle\` — to port
- `R:\tambear\crates\tambear\src\recipes\` — destination (no `elementary/`
  or `libm/` dir yet; needs to be created)
- `R:\tambear\crates\tambear\src\primitives\` — destination (no `constants/`
  or `oracle/` dirs yet)
- `R:\tambear\docs\decisions.md` — DEC-032 + DEC-033 for complex_log port
- `R:\tambear\docs\SWEEP_HISTORY.md` — for naming conventions / planned-sweep map
- `R:\tambear\docs\CRATE_MAP.md` (9.8K) — workspace layout reference
- `R:\winrapids\docs\architecture\tambear-libm-factoring.md` — Sweep 35 design source
- `R:\winrapids\docs\expedition\sweep-35-briefing.md` — what Sweep 35 actually did

End of audit.
