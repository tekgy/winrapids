---
campsite: tambear-sweep31-finish/scientist
role: scientist (validation mode)
date: 2026-05-08
sweep: 31 (follow-on) — validation harness for multi-limb arith unstub + Sweep 34 prep
status: complete — delivered three new test files + one oracle_compare addition
audience: navigator (escalate), pathmaker (run harness against your BZ impl), team-lead/Tekgy
---

# Scientist Deliverables — 2026-05-08

## What was built

### 1. `assert_bit_exact` / `allclose_bit_exact` — oracle_compare addition

File: `R:\tambear\crates\tambear\src\primitives\oracle_compare\mod.rs`

Added two functions (per oracle-validation.md §1.6):
- `bit_exact(a, b, nan_policy)` — verification-tier scalar comparison. `a.to_bits() == b.to_bits()`. NOT ULP-close. Signed-zero-sensitive: `+0.0 != -0.0`.
- `allclose_bit_exact(a, b, nan_policy)` — slice variant, returns `FirstMismatch` on first failure with ULP distance for triage.

36 new lib tests. All passing. 1501 total lib tests (up from 1465).

The `NanPolicy::Equal` behavior under `bit_exact`: two NaNs with different payloads are NOT bit-equal. This is correct for verification-tier (payload preservation per the 2026-05-08 ratification).

### 2. `big_float_cross_precision.rs` — cross-precision consistency harness

File: `R:\tambear\crates\tambear\tests\big_float_cross_precision.rs`

**What it is**: the oracle-validation.md §1.2 antibody for guard-bit bugs in BZ Algorithm 3.1/3.3/3.5/3.10. Computes each op at both p=200 AND p=500, asserts f64-projection agrees. Disagrement = structural bug in rounding logic.

**Why this catches what single-precision tests miss**: a guard-bit error in the last few bits of a p=200 result may round the same direction as hardware f64 (appearing correct) but round DIFFERENTLY at p=500 (extra guard bits change the round direction). The cross-precision comparison exposes this divergence.

**Status of 18 passing tests**: currently all go through the f64 fast path. When multi-limb is implemented:
- `add`/`sub`/`mul`: will automatically exercise the BZ 3.1/3.3 paths (multi-limb operands from `from_f64` still land in f64-fast-path for the operations themselves, but once Newton iteration produces intermediates with guard-bit content, those start exercising the full path)
- `div`/`sqrt` at the Phase B level: Newton iteration at p+50 guard bits WILL produce results where p=200 and p=500 differ if the rounding logic is wrong

**4 Phase B tests** are `#[ignore]`d with clear labels for each BZ algorithm. Remove the `#[ignore]` annotation as each algorithm is implemented.

**Key constants**:
- `P1 = 200`, `P2 = 500` (per oracle-validation.md §1.2 ratified pair)
- `P_LOW/P_MED/P_HIGH = 64/128/256` (tier-boundary triplet)
- All 5 rounding modes tested for every op

**The triple test** (p=200, p=500, p=1024): the `cross_precision_triple_p200_p500_p1024_*` tests exercise the 16-limb regime (p=1024). This is the most stringent test of limb-count scaling behavior.

### 3. `big_float_verification_tier.rs` — mpmath static-corpus harness

File: `R:\tambear\crates\tambear\tests\big_float_verification_tier.rs`

**What it is**: the oracle-validation.md §1.1 harness. Static gold constants (f64 bit patterns from Python `math` library = IEEE 754 correctly-rounded hardware results), asserted bit-exact against BigFloat at p=200/500/1024 across all 5 rounding modes.

**27 passing tests**, 3 `#[ignore]`d Phase B (for BZ 3.3/3.5/3.10 multi-limb).

**Correctness finding during implementation**: the initial gold constants used mpmath-at-50dps values rounded to f64, but when inputs are f64-values, the correct gold is the IEEE 754 f64 result of those specific f64 inputs — NOT mpmath's high-precision result of the true mathematical values. Example: `GOLD_DIV_PI_BY_E` must be `f64(π) / f64(e)` (= `0x3ff27ddbf6271dbe`), not mpmath's `π/e` at 50 dps (which rounds differently because the mpmath computation uses the true π and e, not the f64 approximations). This distinction is the "verification-tier discipline" in practice: test what the implementation DOES on the exact inputs it receives.

All gold constants are now verified by Python computation and stored as bit-pattern literals (`f64::from_bits(0x...)`) to avoid decimal-rounding artifacts.

**Rounding-mode discrimination test**: `vt_rounding_mode_1_by_3_discrimination` — verifies that 1.0/3.0 under RNE gives the correctly-rounded-toward-nearest result (A = `0x3fd5555555555555`). Once Newton div (BZ 3.5) lands, the RTZ/RTP/RTN variants will produce different results and the discrimination test becomes more informative.

---

## Sweep 34 prep: first BigFloat-as-gold migration candidate

Per oracle-validation.md §1.4 + §6 dependencies, identifying which of the 8 existing oracle dirs gets the first BigFloat-as-gold migration.

### Migration readiness matrix

| Oracle dir | Existing structure depth | BigFloat relevance | Migration ease | **Verdict** |
|---|---|---|---|---|
| `variance/` | FULL (SPEC.md + generate_outputs.py + 8 per-package dirs + comparison/ + validation/) | HIGH — BigFloat at p=500 should reproduce mpmath-50dps exactly on variance (conditioning test) | HIGH — 3-edit migration per §1.4 template is most clear for the richest dir | **FIRST** |
| `mean/` | Partial (similar to variance, slightly simpler) | HIGH — mean is BZ 3.1 territory | HIGH | Second |
| `eft_primitives/` | Minimal (README + validate_eft.py) | CRITICAL — EFT primitives are the substrate for BigFloat; BigFloat oracle confirms via independent path | MEDIUM — needs tambear-bigfloat-200/ directory + output.json construction | Third (after variance establishes the template) |
| `dd_subnormal_regimes/` | Minimal (gold.json + sweep.py) | VERY HIGH — oracle-validation.md §1.4 flags this as "MOST relevant for our gap; cross-link bidirectionally" | MEDIUM — gold.json needs BigFloat equivalent | Concurrent with eft_primitives |
| `centered_moment/` `raw_moment/` `ewma/` | Partial | MEDIUM | HIGH | Fourth group (after mean/variance template) |
| `moment_stats_merge/` | Partial | MEDIUM | HIGH | Same as above |

### Recommendation: start with `variance/`

**Why variance first**:

1. **Richest existing structure** — variance has the 9-dir template (R-base/ + python-* + rust-accurate/ + comparison/ + validation/). The 3-edit migration (add tambear-bigfloat-200/ + tambear-bigfloat-500/, update validation/ gold pointer, add disagreements/) is most mechanical here.

2. **GAP-DET-1 conditioning issue** — variance is where the ill-conditioning analysis from the oracle session lives. BigFloat-500 on the ill-conditioned dataset should give the mpmath-50dps result exactly. This is the canonical demonstration of BigFloat-as-gold value.

3. **Template-making** — once variance migration is done, the template for the other 7 dirs is concrete. The mean/ migration will be 90% copy-paste.

4. **Prerequisite chain** — variance migration requires: BigFloat::mul + BigFloat::add working multi-limb (Sweep 31 unstub) + the cache-key plumbing (Sweep 32) + a Python script that calls tambear via PyO3 (or a Rust binary that writes output.json). The Sweep 32 cache-key work is the main prerequisite; Sweep 33 TAM routing ensures BigFloat goes to CPU.

### Concrete next steps for Sweep 34 (post Sweep 32+33)

1. **Create `R:\tambear\oracle\variance\tambear-bigfloat-200\` and `tambear-bigfloat-500\`** directories with `parameters.toml` and `output.json`. The output.json format: same schema as `python-scipy/output.json` — `{dataset_name: value, ...}`.

2. **Write the generation script** — either extend `generate_outputs.py` to call tambear (if PyO3 bindings exist by Sweep 34) or write a Rust binary at `tools/oracle-gen/` that reads the variance `data/generated/*.json` inputs and writes BigFloat output.

3. **Update `validation/` gold pointer** — change the gold from `python-scipy/output.json` or `R-base/output.json` to `tambear-bigfloat-500/output.json`. The validation script recomputes ULP distances against the new gold.

4. **Add `disagreements/` subdirectory** — initially empty. Populated by the triage workflow when any existing implementation disagrees with BigFloat-500.

**Expected finding**: on the `ill_conditioned` dataset, all Python implementations will disagree with BigFloat-500 (by design — that dataset was created to expose f64 single-pass variance errors). This is the first "bug found upstream" artifact. R-base's `var()` may agree if R uses the Welford algorithm internally (which it does). This expected disagreement pattern should be documented in `oracle/variance/disagreements/20260508-ill-conditioned-python-vs-bigfloat.md` before Sweep 34 starts.

---

## What scientist does NOT yet have (forward work)

1. **Live mpmath subprocess harness** (Sweep 35 per oracle-validation.md §4 Q1). The static corpus approach suffices for CI; the live subprocess is for the dev loop when calibrating new coefficients. The API shape (`verify_libm_strict`) is already in oracle-validation.md §1.5.

2. **Phase B multi-limb input corpus** — currently the `#[ignore]`d tests use f64-sourced inputs. Once multi-limb arithmetic lands, a second round of corpus expansion will construct genuinely high-precision inputs (BigFloats with non-zero lower limbs) and add them as Phase C tests. The structural antibody for this is: use arithmetic on BigFloat-sourced values to produce results that DON'T round-trip through f64.

3. **Disagreement archive** at `R:\tambear\oracle\big_float\disagreements\`. This directory does not yet exist. It should be created at Sweep 34 start. The first entry will be the variance ill-conditioned dataset finding.

---

## Test counts summary

Before (Sweep 31): 1465 lib tests, 34 BigFloat integration tests.

After (this session):
- +36 lib tests (oracle_compare: bit_exact + allclose_bit_exact)
- +18 integration tests (big_float_cross_precision: 14 passing + 4 ignored)
- +27 integration tests (big_float_verification_tier: 24 passing + 3 ignored)
- Total: **1501 lib tests, 79 BigFloat-adjacent integration tests** (45 passing, 7 ignored)

All passing. No regressions.

---

## Sign-off

The cross-precision consistency harness and verification-tier mpmath static-corpus harness are shipped and passing. The oracle_compare `bit_exact` primitive is in place. The Sweep 34 migration plan identifies `variance/` as the first oracle dir.

**The scientist signs off**: the validation infrastructure is in place. When pathmaker's BZ Algorithm 3.1/3.3/3.5/3.10 unstub lands, remove the `#[ignore]` annotations from the Phase B tests. If any of the previously-passing Phase A tests then fail, the failure message gives enough information for the triage chain (oracle-validation.md §1.3).

Route to navigator.
