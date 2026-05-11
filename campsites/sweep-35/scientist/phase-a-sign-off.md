# Scientist campsite — Phase A sign-off (expm1 + log1p)

**Session**: tambear-sweep35
**Date**: 2026-05-10
**Scientist**: scientist agent
**Status**: SIGNED OFF

## What was done

Wired the Phase A verification-tier tests into the winrapids codebase at:
`R:\winrapids\crates\tambear\tests\mpmath_oracle.rs`

Two new `#[ignore]` tests added at end of file:
- `oracle_report_expm1` — loads 287-entry corpus from `R:\tambear\oracle\expm1\data\generated\canonical_landmarks\corpus.json`
- `oracle_report_log1p` — loads 366-entry corpus from `R:\tambear\oracle\log1p\data\generated\canonical_landmarks\corpus.json`

Both tests use JSON corpus loading (serde_json, available as crate dep) + the existing `run_corpus_oracle` infrastructure pattern, and assert per-strategy ULP contracts.

## Results

### expm1 (287 corpus entries)

| Strategy | worst ULP | contract | disagreements |
|---|---|---|---|
| expm1_strict | 1 | ≤2 | 9/287 |
| expm1_compensated | 1 | ≤1 | 9/287 |
| expm1_correctly_rounded | 1 | ≤1 | 9/287 |

The 9 disagreements: scattered across `near_zero_dense`, `standard_*`, `constant_half_±1ulp`. Zero disagreements in `very_small_*` (tiny path is exact) and `arg_reduction_boundary`.

**Phase A expm1 SIGNED OFF.**

### log1p (366 corpus entries)

| Strategy | worst ULP | contract | disagreements |
|---|---|---|---|
| log1p_strict | 1 | ≤2 | 30/366 |
| log1p_compensated | 1 | ≤1 | 30/366 |
| log1p_correctly_rounded | 1 | ≤1 | 30/366 |

Key precision-critical categories:
- `near_zero_positive/negative`: 0 ULP (bit-perfect in the precision-critical ring)
- `near_zero_dense`: 0 ULP
- `near_minus_one`: 1 ULP max (singularity approach is clean)

The 30 disagreements are at boundary constants (x = sqrt2−1+1ulp, pi/2+1ulp, e+1ulp, etc.) — expected rounding at argument-reduction boundaries, all within contract.

**Phase A log1p SIGNED OFF.**

## Routing decision resolved

Navigator's question: "verification stubs reference wrong module path — does Phase A ship in winrapids first or does it need porting to R:\tambear?"

Answer implemented: verification tests wired directly into winrapids codebase using correct path `tambear::recipes::libm::expm1::expm1_strict` etc. The `R:\tambear` stubs were discovery-tier only (correctly documented as TODO in `#[ignore]` attribute). No porting required.

## Run commands

```
cd R:\winrapids\crates\tambear
cargo test --test mpmath_oracle oracle_report_expm1 -- --ignored --nocapture
cargo test --test mpmath_oracle oracle_report_log1p -- --ignored --nocapture
```
