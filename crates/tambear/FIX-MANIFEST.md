# Tambear Fix Manifest — Adversarial Findings

**Source**: adversarial test suite (88 tests across 30+ domains, session 2026-04-02)
**Boundary taxonomy**: 5 types (Denominator, Convergence, Cancellation, Equipartition, Structural/Fock)

---

## Status Summary

| # | Bug | File | Boundary Type | Status |
|---|-----|------|---------------|--------|
| 1 | Panel FE single-cluster SE | panel.rs:165-173 | Type 1 (Denominator) | **FIXED** |
| 2 | R̂ NaN for constant chains | bayesian.rs:237-238 | Type 1 (Denominator) | **FIXED** |
| 3 | PAF Heywood detection | factor_analysis.rs:118-123,145 | Type 2 (Convergence) | **FIXED** |
| 4 | GARCH IGARCH boundary | volatility.rs:85-107 | Type 2 (Convergence) | **FIXED** |
| 5 | McDonald's ω bipolar | factor_analysis.rs:261-283 | Type 3 (Cancellation) | **FIXED** |
| 6 | ESS overestimation | bayesian.rs:179-217 | Type 3 (Cancellation) | **NOT A BUG** — Geyer IPS is correct method |
| 7 | Digamma poles | special_functions.rs:124-136 | Type 1 (Denominator) | **FIXED** |

---

## Detailed Fix Records

### 1. Panel FE single-cluster SE — FIXED

**File**: `src/panel.rs:165-173`
**Bug**: `N/(N-1) = 1/0 → Inf` when only 1 cluster
**Fix**: Guard `if n_units <= 1 { correction = 1.0 }` — falls back to uncorrected sandwich
**Boundary type**: Type 1 (Denominator)
**Test**: `adversarial_disputed::panel_fe_single_unit`
**Behavior before**: `se_clustered = [Inf, ...]`
**Behavior after**: `se_clustered = [finite]` (uncorrected sandwich, still correct β)

### 2. R̂ NaN for constant chains — FIXED

**File**: `src/bayesian.rs:237-238`
**Bug**: Within-chain variance `w = 0` → `var_hat / w = 0/0 → NaN`
**Fix**: Guard `if w == 0.0 { return if var_hat == 0.0 { 1.0 } else { f64::INFINITY } }`
- `w=0, var_hat=0`: all chains constant at same value → perfectly converged → R̂ = 1.0
- `w=0, var_hat>0`: chains constant at different values → unmixed → R̂ = Inf
**Boundary type**: Type 1 (Denominator)
**Test**: `adversarial_disputed::r_hat_constant_chains`
**Behavior before**: NaN
**Behavior after**: 1.0 (converged) or Inf (unmixed)

### 3. PAF Heywood detection — FIXED

**File**: `src/factor_analysis.rs:118-123, 145`
**Bug**: Communality > 1.0 for rank-deficient input (adversarial measured 3.63)
**Fix**:
- Communality clamping each iteration (lines 118-123): `c.clamp(0.0, 1.0)` prevents divergence
- `heywood: bool` flag in result (line 145): set when any raw communality > 1.0
**Boundary type**: Type 2 (Convergence) — arguably Type 5 (Structural) since rank deficiency is a property of the input class
**Test**: `adversarial_disputed::paf_rank_deficient_correlation`
**Behavior before**: communality = 3.63, loading = 1.91
**Behavior after**: communalities clamped to [0,1], `heywood = true` flag set
**Future**: dedicated IGARCH model for long-term structural fix; constrained PAF or condition-number pre-check

### 4. GARCH IGARCH boundary — FIXED

**File**: `src/volatility.rs:85-107`
**Bug**: ω explodes to 10¹³ when α+β ≈ 1 (IGARCH boundary)
**Fix**: 
- Stationarity enforcement: `α + β < 1` (line 85)
- `near_igarch` flag: `α + β > 0.99` (line 102)
- Omega clamping: `ω.min(100.0 * uncond_var)` when near IGARCH (line 105)
**Boundary type**: Type 2 (Convergence)
**Test**: `adversarial_disputed::garch_igarch_boundary`
**Behavior before**: `ω = 10¹³` (vs true 10⁻⁴)
**Behavior after**: `ω` clamped, `near_igarch = true` flag set

### 5. McDonald's ω bipolar — FIXED

**File**: `src/factor_analysis.rs:253-282`
**Bug**: ω = 0.0 for bipolar factors (positive + negative loadings cancel in numerator)
**Fix**: 
- Detect bipolar: `has_pos && has_neg` (line 267)
- Use absolute loadings when bipolar (line 272) — equivalent to reverse-scoring
- Return `OmegaResult { omega, bipolar }` — `bipolar: bool` is the V column
**Boundary type**: Type 3 (Cancellation)
**Test**: `adversarial_disputed::mcdonalds_omega_bipolar`
**Behavior before**: ω = 0.0 (meaningless)
**Behavior after**: ω = correct value with `bipolar = true` flag

### 6. ESS overestimation — NOT A BUG

**File**: `src/bayesian.rs:179-217`
**Observation**: ESS = 18.3 for AR(1) ρ=0.99 (theoretical ~5), 3.7× overestimate
**Assessment (math-researcher)**: Geyer's IPS estimator IS the correct method. The 3.7× factor is within statistical tolerance for n=1000 with ρ=0.99 — effective sample is only ~5, so any estimator has high relative variance. This is a fundamental limitation of ESS estimation at extreme autocorrelation, not an implementation bug.
**Boundary type**: Type 4 (Equipartition) — correct answer to a poorly-conditioned question
**Test**: `adversarial_disputed::ess_white_noise_vs_autocorrelated`
**No fix needed.**

### 7. Digamma poles — FIXED

**File**: `src/special_functions.rs:124-136`
**Bug**: ψ(x) has poles at 0, -1, -2, ... which can cause blowup in reflection formula
**Fix**: 
- NaN guard for x = 0 (line 125-126)
- Non-positive integer detection with tolerance (line 131-135): returns NaN
- Tolerance `1e-12` prevents near-pole catastrophic cancellation in `tan(πx)` (line 130)
**Boundary type**: Type 1 (Denominator)
**Test**: `special_functions::tests::digamma_known_values`

---

## Not-A-Bug Cases (Equipartition — Type 4)

These are correct answers to ill-posed questions. No fix needed; document behavior.

| Finding | File | What happens | Why it's correct |
|---------|------|-------------|-----------------|
| LME ICC=0.5 with singletons | mixed_effects.rs | Variance split equally | Maximum entropy — data non-informative |
| Kriging extrapolation → GLS mean | spatial.rs | Far-field pred = constant | Beyond variogram range, kriging regresses to mean |
| Panel FE singleton units | panel.rs | β=0, R²=0 | Demeaning zeros everything — no within-unit variation |

---

## Structural Findings (Type 5 — requires method change, not fix)

| Finding | What failed | Why | Right method |
|---------|-----------|-----|-------------|
| Cesàro∘Wynn 10¹³× error | Wynn after Cesàro | Cesàro destroys convergence pattern Wynn needs | Don't compose; or use Abel→Richardson instead |
| Chebyshev filter diverges on Collatz | Polynomial acceleration | Operator is non-normal (sub-stochastic truncation) | GMRES-family (Krylov, no normality assumption) |
| PAF on rank-deficient input | Communality escapes [0,1] | PAF assumes full-rank correlation | Check condition number first; use constrained PAF |

These are compile-time rejections in the convergence type system. V columns cannot fix them.

---

## Test Counts

- **Lib tests**: 984
- **Adversarial tests**: 92 (30+ domains)
- **Series accel tests**: 48
- **Total boundary guards added this session**: 6 fixed + 1 not-a-bug (ESS)

---

*Observer lab notebook — session 2026-04-02*
*Updated as fixes land. Check off completed items.*
