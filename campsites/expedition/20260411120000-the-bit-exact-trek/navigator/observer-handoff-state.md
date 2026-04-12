# Observer Handoff State
# Written at session end 2026-04-12 for the next observer to read first.
# This is NOT the lab notebook — it's the short version. Read this, then read lab-notebook.md.

## What I am actively watching (6 standing concerns)

| # | Concern | Trigger | Blocking |
|---|---------|---------|---------|
| SC-1 | Tolerance drift — every `assert!((a-b).abs() < 1e-X)` needs justification | Any new test with relative/absolute tolerance | Ongoing |
| SC-2 | Reduction determinism — `@xfail_nondeterministic` must not be removed before Peak 6 | Any removal of that annotation | Peaks 1-5 |
| SC-3 | One-pass variance trap — 5 recipes still use `(Σx² - (Σx)²/n)/(n-1)` | B1 fix landing | Summit tests |
| SC-4 | I1 creep — `eval_gather_expr` in tambear-tam calls f64::exp/ln; must not be reached by new path | Any new `use tambear_tam` in the trek crates | Peaks 4-5 |
| SC-5 | FMA contraction (I3) — PTX defaults contract-everywhere; SPIR-V needs NoContraction per op | Peak 3 PTX assembler landing; Peak 7 SPIR-V landing | Peaks 3, 7 |
| SC-6 | Oracle validation gap — cross-backend agreement is necessary but not sufficient; need mpmath check too | Peak 7 summit test design | Summit |

## 3 most urgent open items

1. **B1 variance** — `tambear-primitives` adversarial_baseline has 2 FAIL, both variance catastrophic cancellation. 5 recipes affected. Two-pass fix architecturally available via `DataSource::Reference`. Nobody has fixed this yet. Will corrupt real financial time series.

2. **tbs eval NaN/Eq** — 4 FAIL in adversarial_tbs_expr: B2 (Eq epsilon 1e-8), B4/B5 (NaN non-propagation through Min/Max in tbs string evaluator). Fixed at accumulates level (`ad84a51`) but tbs::eval layer unfixed.

3. **TMD corpus** — `exp-tmd.md` referenced in `tam_exp.toml` but not created. The `tmd_candidates` corpus has only 4 values. Needs systematic expansion before tam_exp claims faithful rounding is defensible.

## Last known test counts (2026-04-12, pre-shutdown)

| Suite | Passing | Failing | Ignored |
|-------|---------|---------|---------|
| `tambear-tam-ir` | 47+ | 0 | 0 |
| `tambear-tam-test-harness` (lib) | 65 | 0 | 2 (xfail, correct) |
| `tambear-tam-test-harness` (integration) | 5 | 0 | 1 |
| `tambear-tam-test-harness` (oracle_runner) | 5 | 0 | 1 |
| `tambear-primitives` adversarial_baseline | ? | 2 | ? |
| `tambear-primitives` adversarial_tbs_expr | ? | 4 | ? |

## Critical gotcha the next observer must know

**Cody-Waite corpus loading trap:** The `cody_waite_exact` inputs in `tam_exp.toml` are `(n as f64) * LN2_HI` where `LN2_HI = 0x3FE62E42FEFA3800`. The oracle runner must NOT re-evaluate these via mpmath. mpmath's `ln(2)` differs from `LN2_HI`, and that difference is what Cody-Waite tracks. Re-evaluating via mpmath turns "residual is exactly zero" tests into "general case" tests, which are weaker. The oracle TOML note [N2] documents this. Verify corpus loading code enforces it.

## What the lab notebook covers

Entries 001-007. Start with Entry 003 (Peak 6 RFA correction — big framing change) and Entry 005 (adversarial sweep — P19 structural finding that agreement ≠ correctness). The other entries are architectural observations and integration bug tracking.

Lab notebook path: `campsites/expedition/20260411120000-the-bit-exact-trek/lab-notebook.md`
