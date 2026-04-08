# F13 Sharing Surface: Survival Analysis as Sorted Prefix + Cox IRLS

Created: 2026-04-01T06:22:04-05:00
By: navigator

Prerequisites: F07 complete (chi-square), F10 complete (GramMatrix + Cholesky).
F08 SortedPermutation strongly recommended (Kaplan-Meier needs time-sorted data).

---

## Core Insight: Three Algorithms, Three Tambear Patterns

Survival analysis has three canonical algorithms, each mapping to a different tambear primitive:

| Algorithm | Tambear Pattern | New? |
|-----------|----------------|------|
| Kaplan-Meier | Prefix(Multiply) on sorted event times | SortedPermutation required |
| Log-rank test | chi-square on risk-set contingency table | F07 reuse |
| Cox Proportional Hazards | IRLS on risk sets (scatter_multi_phi_weighted) | IRLS master template |

---

## Kaplan-Meier Estimator

**What it computes**: S(t) = P(T > t) — survival probability as a function of time.

```
S(t) = ∏_{t_i ≤ t} (1 - d_i / n_i)
where t_i = event times
      d_i = number of deaths at t_i
      n_i = number at risk just before t_i
```

### Tambear Decomposition

Step 1: Sort by event time → `SortedPermutation(ByTime)` (F08 infrastructure)
Step 2: At each event time, count `{d_i, n_i}` — event count and risk set size
        = `scatter_phi("count", ByTime, where event_indicator==1)` + risk set prefix
Step 3: Cumulative product `∏ (1 - d_i/n_i)` = `Prefix(Multiply)` over sorted event times

**New primitive**: `Prefix(Multiply)` — a prefix scan with Multiply as the combine op.
This is analogous to the cumulative sum prefix scan (Affine with A=1, b=x_t), but with
multiplication. In Kingdom B terms: state_t = state_{t-1} · (1 - d_t/n_t).

Prefix(Multiply) = one-pass scan on the sorted event times. O(N). Very fast.

**Greenwood's formula** (variance of S(t)):
```
Var(S(t)) = S(t)² · Σ_{t_i ≤ t} d_i / (n_i(n_i - d_i))
```
Second prefix scan: `Prefix(Add)` on `d_i / (n_i(n_i-d_i))` terms. Multiply by S(t)² at the end.
Both prefix scans use the same sorted time array. One shared SortedPermutation.

### KM MSR Type

```rust
pub struct KaplanMeierEstimate {
    /// Unique event times. Shape: (m,) where m = number of distinct event times.
    pub times: Vec<f64>,
    /// Survival probability S(t) at each event time. Shape: (m,).
    pub survival: Vec<f64>,
    /// 95% CI bounds (Greenwood). Shape: (m,) each.
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    /// Number at risk at each time. Shape: (m,).
    pub n_at_risk: Vec<usize>,
    /// Number of events at each time. Shape: (m,).
    pub n_events: Vec<usize>,
    /// Median survival time (t where S(t) < 0.5 first).
    pub median_survival: Option<f64>,
}
```

---

## Log-Rank Test

**What it computes**: test whether K survival curves are equal (H₀: S₁(t) = S₂(t) = ... = S_K(t)).

```
At each event time t_j, for each group g:
  E_{gj} = n_{gj}/n_j · d_j   (expected deaths in group g at time j)
  O_{gj} = d_{gj}              (observed deaths)

χ² = Σ_g (Σ_j O_{gj} - E_{gj})² / Var(Σ_j O_{gj} - E_{gj})
```

This is a weighted contingency table analysis over event times. The "contingency table" is
(groups × event_times) with observed and expected counts.

**Tambear decomposition**:
```
1. At each event time: scatter(ByTime, {n_g, d_g}) → risk set statistics per group-time
2. E_{gj} = n_{gj}/n_j · d_j — scalar arithmetic from step 1
3. O - E per group: Σ_j (O_{gj} - E_{gj}) — scatter sum over time within group
4. Variance: Σ_j n_{gj}·n_{control,j}·d_j·(n_j-d_j) / (n_j²(n_j-1)) — same structure
5. Chi-square statistic and p-value — F07 chi-square
```

**This IS F07 chi-square** applied to grouped event-time contingency data. No new primitives.
The only new code: constructing the event-time contingency table from sorted survival data.

---

## Cox Proportional Hazards

**The model**: hazard h(t|x) = h₀(t) · exp(x'β) where h₀(t) = baseline hazard (unspecified).

**Key insight**: the partial likelihood eliminates h₀(t) entirely:
```
L(β) = ∏_{i: event} exp(x_i'β) / Σ_{j ∈ R(t_i)} exp(x_j'β)
     = ∏_{i: event} exp(x_i'β) / Σ_{j ∈ R(t_i)} exp(x_j'β)
where R(t_i) = {j : t_j ≥ t_i} = risk set at time t_i
```

The score and Hessian of log L(β):
```
U(β) = Σ_{events} [x_i - weighted_mean(x, risk set, softmax weights)]
H(β) = Σ_{events} [weighted_covariance(x, risk set, softmax weights)]
```

Where softmax weights = `exp(x_j'β) / Σ_{R(t_i)} exp(x_k'β)` for j in risk set.

### IRLS Master Template Instantiation

This IS the IRLS master template:
- **weights** = softmax weights over risk set = LogSumExp computation over R(t_i)
- **weighted GramMatrix** = `Σ_{j ∈ R(t_i)} w_j · x_j x_j'` = scatter_multi_phi_weighted on risk set
- **Newton step** = Cholesky solve of weighted GramMatrix against score vector

**New infrastructure need**: risk set management.

Risk sets are nested: `R(t_1) ⊃ R(t_2) ⊃ ... ⊃ R(t_m)` (people leave the risk set as they die/are censored). They're computed via a BACKWARD scan over sorted event times:

```
Sort all observations by time (SortedPermutation)
Risk set R(t_i) = all j with t_j ≥ t_i
= suffix starting at position i in sorted array
```

The Breslow estimator of H(β) accumulates over suffixes:
```
H(β) = Σ_{events i} scatter_multi_phi_weighted(risk_set_i, softmax_weights_i, x_matrix)
```

For each event time, this is one `scatter_multi_phi_weighted` call on the current risk set.
Repeated O(n_events) times = O(n²p) total work for naive implementation.

**GPU optimization for Phase 2**: compute the cumulative suffix GramMatrix via a reverse prefix
scan — one O(N) pass instead of O(N²). Each event time's GramMatrix = previous + subtract
departing observations. This is a Kingdom B scan (reverse order, subtract instead of add).

### Phase 1 (CPU-side): O(N²)

For moderate N (< 10K observations), the naive loop is fine:
```
for each event time t_i (sorted):
    risk_set = [j : t_j >= t_i]
    softmax_w = softmax(x[risk_set] @ beta)
    weighted_mean_x = risk_set' @ (softmax_w * x[risk_set])
    weighted_gram = x[risk_set]' @ diag(softmax_w) @ x[risk_set]
    score += x[event_i] - weighted_mean_x
    hessian += weighted_gram - outer(weighted_mean_x, weighted_mean_x)
beta += cholesky_solve(hessian, score)  // Newton step
```

Each Newton step = K `scatter_multi_phi_weighted` calls (K = number of covariates).
N_events Newton steps to convergence (typically 10-20).

### CoxModel MSR Type

```rust
pub struct CoxModel {
    pub n_obs: usize,
    pub n_events: usize,       // number of non-censored observations
    pub n_params: usize,       // K covariates

    pub beta: Vec<f64>,        // hazard ratios on log scale
    pub beta_se: Vec<f64>,     // SE from inverse Hessian at convergence
    pub hazard_ratio: Vec<f64>, // exp(beta) — multiplicative hazard factor

    pub log_likelihood: f64,
    pub concordance: f64,       // C-index: proportion of pairs correctly ordered

    /// Breslow baseline hazard estimate H₀(t) at event times.
    pub baseline_times: Vec<f64>,
    pub baseline_hazard: Vec<f64>,
}
```

---

## Build Order

**Phase 1 (KM + log-rank)**:
1. `SortedPermutation` — must exist (from F08 if built, else implement here)
2. `fn kaplan_meier(times: &[f64], events: &[bool]) -> KaplanMeierEstimate` — Prefix(Multiply) over sorted event times (~50 lines)
3. `fn greenwood_variance(km: &KaplanMeierEstimate) -> Vec<f64>` — second Prefix(Add) (~20 lines)
4. `fn log_rank_test(groups: &[Vec<f64>], events_per_group: &[Vec<bool>]) -> TestResult` — contingency over event times, F07 chi-square (~60 lines)
5. Tests: `survival::survfit()` in R, `lifelines.KaplanMeierFitter` in Python

**Phase 2 (Cox PH)**:
1. Risk set boundaries via backward scan on sorted time array (~20 lines)
2. Newton-Raphson loop: risk-set softmax weights, weighted GramMatrix per event time (~100 lines)
3. Convergence check: ||score||² / ||initial_score||² < tol
4. C-index computation: O(N²) pairwise comparison (~30 lines)
5. Breslow baseline hazard: `Ĥ₀(t) = Σ_{t_i ≤ t} 1 / Σ_{R(t_i)} exp(x_j'β̂)` (~20 lines)
6. Tests: `survival::coxph()` in R, `lifelines.CoxPHFitter` in Python

**Gold standards**:
- R: `survival::survfit()`, `survival::coxph()`, `survival::survdiff()` (log-rank)
- Python: `lifelines.KaplanMeierFitter`, `lifelines.CoxPHFitter`
- Dataset: WHAS500 (500 patients, AMI study) — C-index ≈ 0.73 for age+BMI

---

## The Lab Notebook Claim

> Kaplan-Meier = Prefix(Multiply) over sorted event times — a one-pass scan after SortedPermutation. Log-rank = chi-square (F07) applied to grouped risk-set contingency data. Cox PH = the IRLS master template instantiated with softmax risk-set weights — the same `scatter_multi_phi_weighted` that serves F09/F10/F16. F13 confirms: every estimator in the library that iteratively reweights observations IS the IRLS template. The only structural novelty is risk set management (backward suffix scan), which is ~20 lines of wiring.
