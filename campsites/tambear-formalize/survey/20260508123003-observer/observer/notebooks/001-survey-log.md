# Lab Notebook 001: tambear-formalize Survey — Observer Record

**Date**: 2026-05-08
**Role**: Observer (scientific conscience)
**Campsite**: tambear-formalize/survey/observer
**Status**: Active
**Depends on**: team-briefing.md, vocabulary.md, atoms-primitives-recipes.md

---

## Context & Motivation

Team `tambear-formalize` was spawned to survey `R:\winrapids\campsites\` and identify which exploratory math work is ripe to formalize into `R:\tambear`. The observer role is scientific conscience: neutral record, peer-review mindset, verifiability tracking.

This notebook is the real-time record. It documents what IS, not what we hope. Appended chronologically.

---

## Anchor Verification

### Test count at session start

**Briefing claimed**: 1397 lib tests, 2027 total.

**Actual** (run 2026-05-08):
```
cargo test: 2029 passed, 106 ignored (54 suites, 4.01s)
```

**Discrepancy**: Total is 2029, not 2027. Delta = +2. Lib test count not separately confirmed yet (the tail output does not break out lib vs integration vs doc). The anchor is approximately correct. The +2 delta is small enough to be explained by commits since the briefing was written (most recent commit: `27386ef clustering: davies_bouldin NaN propagation fix + track bug tests as #[ignore]`).

**Significance**: Minor. The 106 ignored tests are notable — these are likely tests marked `#[ignore]` because they track known bugs or are adversarial tests gated on external conditions. Worth auditing what they are before claiming the test suite is "green."

---

## Survey Log

### Entry 1 — 2026-05-08: Campsite landscape

Five priority campsites identified. Metadata from filesystem:

| Campsite | Last Modified | Sub-dirs visible |
|---|---|---|
| `dec-029-impl/` | 2026-05-04 | `20260504-wave2-bugs/` |
| `r10-15/` | 2026-04-23 | (need to list) |
| `sweep-10/` | 2026-04-23 | (need to list) |
| `sweep-8/` | 2026-04-22 | (need to list) |
| `tambear-trig/` | 2026-04-14 | catalog, angle-units, atoms-gaps, autodiscover, compilation, defaults, implementations, notation, shared-pass, spec-tomls, tbs-syntax, variants |

Older campsites all dated 2026-04-12 or earlier — predating the vocabulary lock (2026-04-17). Any material pulled from them requires vocabulary translation before it can be trusted.

**Observer note**: `tambear-trig/` is dated 2026-04-14 — three days BEFORE the vocabulary lock. All trig campsite material was written under pre-lock vocabulary. Every term in those files requires translation through the mapping table in `vocabulary.md`.

---

### Entry 2 — 2026-05-08: Test baseline — ignored tests audit

**Before diving into campsites**: the 106 ignored tests are invisible to the green/red status. If any of those are "should be passing but aren't," the apparent baseline is misleading.

**Hypothesis**: Most ignored tests are adversarial/oracle tests that require external conditions (mpmath, specific hardware) or are intentionally gated on known bugs being fixed in this team's work.

**Action needed**: `cargo test -- --ignored 2>&1 | head -80` to surface what's being skipped.

---

### Entry 3 — 2026-05-08: Verifiability framework

For each campsite item I survey, I will track:

| Claim | Type | Gold standard reference | Accessible? | Test exists in tambear? |
|---|---|---|---|---|
| (to be populated) | hypothesis/evidence/result | | | |

**Claim types**:
- H = hypothesis (stated expectation, not yet tested)
- E = evidence (experimental result, possibly informal)
- R = result (formally verified against gold-standard oracle)
- D = design decision (architectural/structural, not empirically testable)

The Filter Test (10-point checklist) provides the formalization bar. Items that clear all 10 points are ready to land in tambear. Items that fail any point need work identified before they can ship.

---

## Open Questions (to track throughout survey)

1. What are the 106 ignored tests? Are any of them "should pass but don't"?
2. What has already shipped from the campsites into tambear since the campsites were written? (Recent commits suggest trig/softmax/clustering work has already been formalized — need to confirm no duplication.)
3. Does `sweep-8/` JIT work align with DEC-019 (native-door JIT, no middleware)? The briefing notes `project_jit_ir_design.md` mentions finite JitOp enum for PTX/SPIR-V/AIR/DXIL lowering — but DEC-019 prohibits wgpu. Need to verify the campsite isn't proposing middleware.
4. What does `dec-029-impl/` implement? What is DEC-029?
5. Are the untracked Python scripts at the winrapids root (derive_exp_constants.py, etc.) oracles that should be committed somewhere, or are they throwaway derivations?

---

## Survey Findings — 2026-05-08

### Entry 4 — Ignored tests: 106 ignored + 5 "failures" when --ignored

**Hypothesis**: most ignored tests are gated-bug stubs for unimplemented sweeps.

**Actual**: Running `cargo test -- --ignored` on `sweep_27_precheck_adversarial` reveals 5 "failures":
- `cross_claim_consistency_checker_catches_profiler_bugs` (line 550)
- `precheck_cache_key_changes_when_using_annotation_changes_intermediate_shape` (line 280)
- `non_finite_claim_needs_inapplicable_variant_for_integer_columns` (line 394)
- `data_profile_has_confidence_tagged_non_finite_field` (line 311)
- `stale_known_absent_does_not_survive_dataset_update` (line 477)

**Nature of failures**: All 5 are `panic!("Sweep 27A/27B not landed")` stubs. They FAIL by design — they're tracking that Sweep 27A/27B has not been implemented yet. This is intentional.

One test passed: `jarque_bera_does_not_claim_normality_for_symmetric_bimodal` — this passes because the JB function exists and produces the documented blind spot (it DOES return Gaussian for symmetric bimodal data, which is the pre-fix behavior being documented).

**Classification**: These are not regressions. They are gap-tracking tests for Sweep 27A/27B work. The baseline IS clean in the normal test-run sense. The "failures" under `--ignored` are intentional sentinel failures documenting unimplemented features.

**Observer note**: This is an interesting pattern. The test file encodes the design specification for a feature that doesn't exist yet, with `panic!()` bodies that fail if someone accidentally runs them. When the feature lands, the tests get un-ignored and the panics get replaced with real assertions. This is a legitimate practice but it means `cargo test -- --ignored` is NOT a useful regression signal — it includes intended failures.

---

### Entry 5 — Campsite content survey: what's actually there

**Survey method**: Glob all .md files in each priority campsite.

**Findings by campsite**:

| Campsite | Content status | Key materials found |
|---|---|---|
| `dec-029-impl/` | Structural skeleton only | Only `adversarial/20260504140805-creation.md`. All other role dirs empty. **No content.** |
| `sweep-10/oracle-seeding/` | Near-empty | Only math-researcher creation marker. No content. |
| `r10-15/` | 3 substantive docs | navigator: variance-routing-spec.md, compaction-ghost-pattern.md; math-researcher: exp-for-lse-spec.md, log-for-entropy-spec.md; naturalist: creation marker only |
| `sweep-8/` | Dense: 25+ docs | Primarily in `jit-design/aristotle/` — trait specs, phase-1-8 docs, response-to-adversarial waves. `edge-cases/adversarial/` has the sweep-27 precheck attacks. `cranelift-codegen/` empty except creation. |
| `tambear-trig/` | Very sparse | Only `implementations/scout/insights/asin-polynomial-audit.md`. All 11 other sub-dirs have only creation markers. |

**Critical observation**: Most campsite role subdirectories are empty scaffolds. The substantive work is concentrated in:
1. `sweep-8/jit-design/aristotle/` — ~20 design documents
2. `r10-15/` — 4 substantive specification documents
3. The untracked root Python scripts

---

### Entry 6 — DEC-029: what it is

DEC-029 is NOT about JIT doors (that's DEC-019). DEC-029 is about **evidence-convergence** — the sister DEC to DEC-030 (Refinement-lattice). Per decisions.md §5.5: "DEC-029 owns claim content (what); DEC-030 owns claim transport (where)."

The `dec-029-impl/` campsite (dated 2026-05-04, wave2-bugs) therefore contains work on implementing DEC-029's evidence-convergence system — but the campsite is essentially empty. No documents beyond the adversarial creation marker. Either the work was done elsewhere (directly in tambear commits?) or it was just created and the team moved on before writing anything down.

**Verifiable claim**: Check if `dec_029_adversarial.rs` and `dec_029_adversarial_2.rs` exist in tambear tests. Confirmed: both exist at `R:\tambear\crates\tambear\tests\`. So DEC-029 implementation has landed in tambear directly — the campsite was a structural placeholder, not a work site.

---

### Entry 7 — r10-15 math specs: verifiability assessment

**`exp_for_lse_spec.md`** (r10-15/math-references):
- Algorithm: Tang 1989 table-driven exp with Chebyshev-minimax polynomial
- Constants verified by `derive_exp_minimax.py` at 100dps precision
- Cranelift IR sequence provided: ~30 instructions
- Test vectors: 14 points with mpmath-computed references
- Edge case handling: explicit (NaN, zero, underflow, denormal)
- DEC-027 compliance documented: no FMA without `supports_fma` cache key gate
- **Status**: Full specification ready for implementation. Not yet landed in tambear (no `recipes/libm/exp.rs` found in the source tree).

**`log_for_entropy_spec.md`** (r10-15/math-references):
- Algorithm: Tang 1990 table-driven log with near-1 fast path
- 128-entry table (2048 bytes), spot-checked against mpmath at 100dps
- Cranelift IR for both fast and general paths
- Caller conventions for `shannon_entropy` and `hill_estimator` documented explicitly
- Constants: LN2_HI/LO bit-exact, B1..B4 polynomial coefficients (noted: B1..B4 bit patterns marked "need to recompute hex")
- **Observer concern**: The B1..B4 hex constants are MISSING in the spec (placeholders `f64const ...`). The decimal values are there but the bit patterns are not computed. `derive_log_minimax.py` exists at winrapids root to regenerate them.
- **Status**: Full specification ready except for the B1..B4 hex constant computation. Not yet landed in tambear.

**`variance-routing-spec.md`** (r10-15/lab-notebook/navigator):
- Documents two-step routing rule for variance algorithm selection (Chan vs Kahan two-pass)
- Verified by `verify_variance_spec_claims.py` (but: "does NOT yet cover const+signal failure mode A")
- Defines `scalar_variance_condition_number` function needed in `primitives/precheck/mod.rs`
- **Status**: Spec exists; code does not. `primitives/precheck/mod.rs` exists but function is unimplemented.

---

### Entry 8 — asin polynomial audit: status

**`implementations/scout/insights/asin-polynomial-audit.md`** (tambear-trig):
- Documents two pre-fix bugs in `recipes/libm/asin.rs`: P_S2 digit transposition, P_S5 wrong sign
- Status: marked "FIXED — recording pre-fix bugs for audit trail"
- The fix is applied and source comments document the correction
- **Verifiable**: `grep P_S2 R:\tambear\crates\tambear\src\recipes\libm\asin.rs` should show the corrected value

**Observer question**: does `recipes/libm/asin.rs` exist? This is a trig recipe that would have been part of the tambear-trig campsite work.

---

### Entry 9 — sweep-8 adversarial: what shipped

The sweep-8 adversarial precheck attacks document is already committed in tambear as `tests/sweep_27_precheck_adversarial.rs`. The campsite document is the design/analysis source; the test file is what landed.

The 8 attacks are documented there. 5 are `panic!("not landed")` stubs (tracking Sweep 27A/27B gaps). The jarque-bera test passes (documenting the blind spot). The scan_nonfinite tests (27-1) should be checkable now — the precheck module exists.

**Open question**: do the scan_nonfinite tests (ATTACK 27-1 group) pass or fail? Those aren't stubs — they test actual functionality.

---

### Entry 10 — What needs to land vs what is landed

**Already in tambear** (confirmed by file/test existence):
- `precheck/mod.rs` — scan_nonfinite_full, scan_nonfinite_sample, jarque_bera, condition_number_estimate, sparsity_fraction
- `dataprofile/mod.rs` — DataProfile with NonFiniteClaim
- `dec_029_adversarial.rs`, `dec_029_adversarial_2.rs` — DEC-029 test coverage
- `jit/` — JitOp, shape.rs, door.rs (confirmed by compaction-ghost-pattern.md)
- `recipes/` — shannon_entropy, hill_estimator_streaming, various others
- `knowledge/` — canonicalize, types, ingest, mod

**NOT yet in tambear** (from spec docs):
- `recipes/libm/exp.rs` — Tang 1989 table-driven exp (spec exists: r10-15/exp-for-lse-spec.md)
- `recipes/libm/log.rs` — Tang 1990 table-driven log (spec exists: r10-15/log-for-entropy-spec.md)
- Variance routing function `scalar_variance_condition_number` in precheck (spec exists: r10-15/variance-routing-spec.md)
- Sweep 27A/27B: DataProfile confidence-tagged types, pre-check pipeline cache keys (5 gated tests)

**Unconfirmed** (need to check if asin/acos/atan/atan2 are in tambear):
- `recipes/libm/asin.rs` — trig implementations from tambear-trig campsite

---

## Verifiability Tracker

| Claim | Type | Gold standard | Accessible | In tambear |
|---|---|---|---|---|
| exp_for_lse: 0-4 ULP typical, 5-12 ULP extreme | R | mpmath, derive_exp_minimax.py | Yes (py script exists) | No — not yet landed |
| log_for_entropy: ≤1 ULP for most inputs, 5 ULP near x=1e-300 | R | mpmath, derive_log_minimax.py | Yes (py script exists) | No — not yet landed |
| Variance routing: range_ratio < SQRT_EPS → Chan | R | verify_variance_spec_claims.py | Yes | No — function not found |
| asin P_S2/P_S5 fix | R | fdlibm e_asin.c | Yes | Unconfirmed — need grep |
| JB blind to symmetric bimodal | E (documented) | Mathematician's oracle | Yes (test body) | Partially — JB exists, blind spot confirmed by test |
| Sweep 27-1: scan asymmetry | D (design contract) | Test assertions | Yes | Partially — precheck exists |
| DEC-029 evidence convergence | D | decisions.md | Yes | Yes — dec_029_adversarial.rs exists |

---

---

### Entry 11 — CRITICAL: Two-repo architecture

**Discovery**: The trig/libm commits referenced by the briefing (asin/acos/atan/atan2 adversarial sweeps, cospi external-oracle, tanh saturation, softmax NaN) are NOT in `R:\tambear`. They are in `R:\winrapids\crates\tambear\src\recipes\libm\`.

**There are two tambear crate trees**:
- `R:\winrapids\crates\tambear\` — exploratory/research implementations, committed to winrapids git
- `R:\tambear\crates\tambear\` — formalization target, separate git repo

The **formalization work** this team is doing is: taking implementations from winrapids and moving/porting them into the tambear git repo.

**What exists in winrapids/crates/tambear/src/recipes/libm/ (54 files)**:
- `exp.rs`, `log.rs`, `gamma.rs`, `erf.rs` — transcendentals
- `sin.rs`, `cos.rs` (via sincos.rs), `tan.rs`, `asin.rs`, `atan.rs`, `sincos.rs` — trig
- `hyperbolic.rs`, `inv_hyperbolic.rs` — hyp trig
- `pi_scaled.rs`, `pi_scaled_inv.rs`, `sincos_pi.rs` — pi-scaled variants
- `rare_trig.rs`, `inv_recip.rs` — rare functions
- `adversarial.rs` — adversarial test generators
- 36 `.spec.toml` files — one per function (specification)

**What does NOT exist in R:\tambear\crates\tambear\src\recipes\libm\**: the entire libm folder.

This is the core gap. The trig/libm work is done (implemented, tested, adversarially attacked) in winrapids but has not been transferred to the tambear repo.

---

### Entry 12 — The compaction-ghost-pattern document: operational wisdom

The navigator's `compaction-ghost-pattern.md` documents two failure modes that affected previous sessions:
- **Mode A (true ghost)**: agent cites file that doesn't exist anywhere
- **Mode B (wrong-repo)**: agent cites file that exists in the OTHER repo

This document IS the answer to why so many session notes referenced `recipes/libm/asin.rs` as if it were in tambear. Agents running in the winrapids context see the winrapids crates and believe they're in tambear. The two-repo architecture is a persistent source of confusion.

**Observer note for formalization work**: every grep, every file check, every "does this exist?" must specify WHICH repo. The default working directory confusion is a structural hazard for this team.

---

### Entry 13 — sweep-8 aristotle work: design specs status

The sweep-8 jit-design/aristotle documents are design specifications for the DoorBackend trait — the architecture that allows tambear to dispatch to CUDA/Vulkan/Metal/DX12 without middleware.

Key documents found (12+ files in aristotle/):
- `phase-1-8-deconstruction.md` — systematic assumption autopsy for DoorBackend trait (12 assumptions)
- `trait-spec-locked.md` — the locked trait design
- `phase-1-8-multi-dim-shape.md`, `phase-1-8-adaptive-dimhint.md` — shape system design
- `phase-1-8-tam-cannot-tell-the-future.md` — Q-rec-2 invariant (profile says YES only after evidence)
- Multiple `response-to-adversarial-*.md` — responses to adversarial attacks
- `is-kernel-share-compatible-formal-structure.md` — formal spec for kernel sharing
- `pipeline-determinism-composition-algebra.md` — determinism algebra

Status: confirmed that `jit/shape.rs`, `jit/door.rs`, `jit/jit_op.rs` DO exist in `R:\tambear\crates\tambear\src\jit\` (confirmed by compaction-ghost-pattern.md which corrected a Mode B ghost confusion).

The sweep-8 design work has LANDED in tambear's jit module. These aristotle documents are the paper trail, not unimplemented specs.

---

## Summary — What is ripe vs unripe for formalization

### HIGH PRIORITY — Ready to land, clear specs, implementations exist

1. **libm recipes** (`exp.rs`, `log.rs`, `asin.rs`, etc.): 54 files exist in winrapids/crates/tambear/src/recipes/libm/. Adversarial tests committed to winrapids. Filter Test items 1, 2, 4, 5, 10 appear satisfied in the winrapids version. Need to verify items 6 (hardware optimization), 7 (DEC-019 JIT compliance), 9 (TAM lifting) before transferring. **This is the biggest single gap.**

2. **Variance routing**: `scalar_variance_condition_number` function specified in r10-15/variance-routing-spec.md. The two-step routing rule is documented and validated. Precheck module exists in tambear. This is a small, targeted addition.

### MEDIUM PRIORITY — Spec exists, implementation incomplete

3. **exp_for_lse Cranelift IR** (r10-15/exp-for-lse-spec.md): Full IR sequence documented, test vectors present, constants verified. Missing from tambear. The current `exp.rs` in winrapids may be the Rust implementation; the Cranelift IR is a separate JIT kernel. Both need to land.

4. **log_for_entropy Cranelift IR** (r10-15/log-for-entropy-spec.md): Same situation. B1..B4 hex constants are missing from the spec (need `derive_log_minimax.py` run). This is a 5-minute task before implementation can proceed.

### LOWER PRIORITY — Design work landed, gap is testing

5. **Sweep 27A/27B** (precheck/DataProfile confidence types): 5 stub tests exist. Feature is not yet implemented. Complex — involves DataProfile struct changes, pipeline cache key design. The adversarial attacks are documented and incisive.

### ALREADY LANDED — Do not re-implement

6. **JIT shape/door/jit_op**: In tambear. DoorBackend trait exists.
7. **DEC-029 evidence convergence**: dec_029_adversarial.rs and dec_029_adversarial_2.rs in tambear.
8. **Knowledge system**: knowledge/ module in tambear (types, canonicalize, ingest, impls).
9. **DEC-030 refinement lattice**: decisions.md ratified, code in tambear.

---

## Open Questions (updated)

---

### Entry 16 — Winrapids libm oracle strategy: platform comparison + mpmath harness

**Finding**: The adversarial trig tests (`asin_adversarial_sweep_vs_platform`, etc.) compare against `x.asin()` — Rust's platform libm. On Windows MSVC CRT, this can be up to 1 ULP off the true value, meaning the ≤4 ULP gate is relative to a potentially-wrong reference.

**However**: `R:\winrapids\crates\tambear\tests\mpmath_oracle.rs` exists and explicitly acknowledges this. Its docstring: "our '≤ 1 ulp worst case' measurement is relative to a potentially-1-ulp-off reference. This harness calls mpmath at 100-digit precision via Python subprocess, giving us a true gold-standard oracle."

The mpmath oracle is `#[ignore]`-gated — opt-in with `cargo test --test mpmath_oracle -- --ignored --nocapture`. Requires Python + mpmath. It does NOT run in CI.

**Assessment**: This is an intentional engineering tradeoff, not a gap. The platform-comparison tests are fast development checks; the mpmath oracle is the publication-grade verification. Both exist. The Filter Test item 10 compliance depends on the mpmath oracle having been run and passed — which is an external condition, not checkable from the test file alone.

**Implication for formalization**: Before declaring any libm recipe ready to land in tambear, the mpmath oracle should be run against it. The question to ask of each recipe: "has mpmath_oracle been run, and what was the worst-case ULP distance from truth (not from platform)?"

---

### Entry 17 — Final survey status

**Survey complete as of 2026-05-08.**

All five priority campsites surveyed:
- `dec-029-impl/` — empty skeleton; DEC-029 work landed directly in tambear
- `sweep-10/oracle-seeding/` — empty skeleton; no content
- `r10-15/` — 4 substantive docs: exp spec, log spec, variance routing spec, compaction-ghost-pattern
- `sweep-8/` — design trail, mostly landed; 2 substantial unimplemented items (Cranelift codegen, compensated Welford)
- `tambear-trig/` — sparse; one asin audit doc; all trig implementations live in winrapids/crates/tambear/src/recipes/libm/

Untracked root scripts: not yet read (`derive_exp_constants.py`, `verify_*.py`, etc.) — deprioritized given the r10-15 specs already cover the key derivations.

The central finding — two-repo architecture with 54 unformalized libm files — stands as the primary action item for this team.

---

## Open Questions (final)

1. What is the actual test coverage of the winrapids libm implementations? Do the adversarial tests in winrapids validate against mpmath oracles, or against platform reference (which may itself be wrong)?
2. The log_for_entropy spec is missing hex constants for B1..B4. Generate from `derive_log_minimax.py` before implementation.
3. Are the scan_nonfinite_full and scan_nonfinite_sample tests (ATTACK 27-1) passing in the normal test run (not --ignored)?
4. The asin polynomial audit says "FIXED — recording pre-fix bugs." Is this fix in the winrapids asin.rs? What version?
5. Before transferring any libm recipe: run Filter Test against each. Item 7 (no middleware) needs clarification — does the winrapids implementation use Rust stdlib floats or explicit vendor IR?

---

### Entry 14 — Sweep-8 deep read: DoorBackend trait and bit-exact Add

**`trait-spec-locked.md`** (jit-design/aristotle):

The DoorBackend trait spec (locked 2026-04-22) is the source document for what became `R:\tambear\crates\tambear\src\jit\door.rs`. Comparing spec to code:

- `DoorCodegen`, `DoorCache`, `DoorDispatcher`, `DoorBackend` — all present in tambear
- `DoorCapability` with `supports_fma`, `denormal_mode`, `isa_version` — present
- `ExecutionStrategy` (`Lifted`, `LiftedConjugated`, `Sequential`) — present but variants evolved
- `DeterminismClass` — present, with `BitExact` instead of `Deterministic`, and a new `MathematicallyEquivalent { max_condition_number }` variant not in the spec
- `CompiledArtifact`, `ArtifactBinary`, `Loaded`, `EntryPoint` — present
- `NoOpBackend` — present (matches spec test plan requirement 2)
- `ErasedDoorBackend` as super-trait — present, with doc comment explaining it closes GAP-CAP-KEY-1
- `cache_key_for()` as a default method on `DoorBackend` — NOT in spec; added in code to fix GAP-CAP-KEY-1

**`phase-1-8-bit-exact-add.md`** (gap-lift-1-bitexact/aristotle):

Aristotle proposed a three-mode `SumStrategy` enum (Exact/Fast/Nondet). Status in tambear:

- `SumStrategy` EXISTS in `jit_op.rs` — variants: `Sequential`, `LiftedFree`, `LiftedTree`, `Kulisch`
- These are accumulator-topology variants, not precision-level variants (aristotle's Exact/Fast/Nondet)
- `KulischAccumulator` primitive EXISTS in `primitives/specialist/kulisch_accumulator.rs`
- `determinism_floor(sum_strategy: SumStrategy)` method on `JitOp` — quantitative improvement over spec
- `MathematicallyEquivalent { max_condition_number: 1e6 }` DeterminismClass variant — carries conditioning threshold, more nuanced than spec's three-class schema
- 109-point kappa sweep for MomentStats::merge committed as `sweep_29c_kappa_oracle`

**Spec-to-code key divergence**: Aristotle's Exact/Fast/Nondet (arithmetic precision) was collapsed into execution topology (LiftedFree/LiftedTree/Kulisch). The design evolved beyond the campsite spec. The landed code is more rigorous (quantitative conditioning threshold) than the campsite proposed.

**Status of sweep-8**: JIT design HAS LANDED. DoorBackend, ExecutionStrategy, DeterminismClass, SumStrategy, JitOp, Shape, CacheKey, NoOpBackend — all in tambear. Campsite is the design trail, not unimplemented work.

Remaining open items:
- Real Cranelift codegen (Sweep 8C) — currently NoOpBackend stub
- Welford compensated path (Sweep 8D)
- Multi-GPU dispatch (Sweep 17+)
