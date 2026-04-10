# GARCH Variants — Complete Variant Catalog

## What Exists (tambear::volatility)

- `garch11_fit(returns, max_iter)` — GARCH(1,1): σ² = ω + α ε² + β σ²
- `garch11_forecast(res, last_return, horizon)` — multi-step forecast
- `egarch11_fit(returns, max_iter)` — EGARCH: log σ² = ω + α(|z|-E|z|) + γz + β log σ²
- `gjr_garch11_fit(returns, max_iter)` — GJR-GARCH: adds γ ε² I(ε<0)
- `tgarch11_fit(returns, max_iter)` — TGARCH: σ = ω + α|ε| + γ ε⁻ + β σ
- `ewma_variance(returns, lambda)` — RiskMetrics EWMA

### Related
- `realized_variance`, `realized_volatility`, `bipower_variation`
- `jump_test_bns` — Barndorff-Nielsen & Shephard
- `parkinson_variance`, `garman_klass_variance`, `rogers_satchell_variance`, `yang_zhang_variance`
- `hill_estimator`, `hill_tail_alpha` — tail index
- `tripower_quarticity`
- `arch_lm_test` — ARCH effects test

---

## What's MISSING — Complete Catalog

### A. GARCH(p,q) — Higher Orders

1. **GARCH(p,q)** — general order
   - σ²_t = ω + Σᵢ₌₁ᵖ αᵢ ε²_{t-i} + Σⱼ₌₁ᵍ βⱼ σ²_{t-j}
   - Parameters: `returns`, `p`, `q`, `max_iter`
   - Currently only (1,1); need general (p,q)
   - In practice (1,1) dominates, but (2,1), (1,2), (2,2) are used

2. **IGARCH** — Integrated GARCH (Engle & Bollerslev 1986)
   - GARCH with α + β = 1 (persistence = 1)
   - No finite unconditional variance
   - Parameters: `returns`, `max_iter`
   - Constraint: optimize ω, α; set β = 1 - α

### B. Asymmetric / Leverage Models

3. **APARCH** (Asymmetric Power ARCH) — Ding et al. 1993
   - σᵟ_t = ω + Σ αᵢ(|ε_{t-i}| - γᵢ ε_{t-i})ᵟ + Σ βⱼ σᵟ_{t-j}
   - Parameters: `returns`, `p`, `q`, `delta` (or estimate delta), `max_iter`
   - Nests: GARCH (δ=2, γ=0), GJR (δ=2), TGARCH (δ=1), NARCH
   - Most general single-equation volatility model

4. **NGARCH** (Nonlinear GARCH) — Engle & Ng 1993
   - σ²_t = ω + α(ε_{t-1} - θσ_{t-1})² + β σ²_{t-1}
   - θ captures leverage; θ>0 means negative shocks increase vol more
   - Parameters: `returns`, `theta`, `max_iter`

5. **QGARCH** (Quadratic GARCH) — Sentana 1995
   - σ²_t = ω + α ε²_{t-1} + γ ε_{t-1} + β σ²_{t-1}
   - Linear term γε allows asymmetry without indicator function
   - Parameters: `returns`, `max_iter`

6. **NAGARCH** — Engle & Ng 1993
   - σ²_t = ω + α(ε_{t-1}/σ_{t-1} + γ)² σ²_{t-1} + β σ²_{t-1}
   - News impact curve is a shifted parabola

7. **AVGARCH** (Absolute Value GARCH)
   - σ_t = ω + α|ε_{t-1}| + γ ε_{t-1} + β σ_{t-1}
   - Similar to TGARCH but with signed term

### C. Component / Long-Memory Models

8. **Component GARCH** (CGARCH) — Engle & Lee 1999
   - σ²_t = q_t + α(ε²_{t-1} - q_{t-1}) + β(σ²_{t-1} - q_{t-1})
   - q_t = ω + ρ q_{t-1} + φ(ε²_{t-1} - σ²_{t-1})
   - Decomposes into permanent (q_t) and transitory components
   - Parameters: `returns`, `max_iter`

9. **FIGARCH** (Fractionally Integrated GARCH) — Baillie et al. 1996
   - σ²_t = ω + [1 - β(L) - (1-L)^d φ(L)] ε²_t + β(L) σ²_t
   - d ∈ (0, 1) is the fractional integration parameter
   - Parameters: `returns`, `d_init`, `max_iter`
   - Long memory in volatility

10. **HYGARCH** (Hyperbolic GARCH) — Davidson 2004
    - Generalization of FIGARCH
    - σ²_t = ω + {1 - [1-α(L)][(1-L)^d - 1 + β(L)·α]} ε²_t
    - Nests IGARCH (d=1) and FIGARCH

11. **FIEGARCH** — Fractionally integrated EGARCH
    - Long memory in log-volatility
    - Parameters: `returns`, `d_init`, `max_iter`

### D. Distribution Variants

All GARCH models above can use different innovation distributions:

12. **GARCH-t** — Student-t innovations
    - Fat tails, df parameter estimated jointly
    - Parameters: all GARCH params + `df_init`
    - Most common in financial applications

13. **GARCH-GED** — Generalized Error Distribution innovations
    - P(z) ∝ exp(-½|z/λ|^v)
    - v=2: Gaussian, v=1: double exponential
    - Parameters: all GARCH params + `shape_init`

14. **GARCH-skew-t** — Hansen's skewed t-distribution
    - Asymmetric fat tails
    - Parameters: `df`, `skew` estimated jointly
    - Most realistic for financial returns

15. **GARCH-NIG** — Normal Inverse Gaussian innovations
    - Four-parameter distribution (α, β, δ, μ)
    - Very flexible tail behavior

### E. Multivariate GARCH

16. **DCC-GARCH** (Dynamic Conditional Correlation) — Engle 2002
    - Each series has univariate GARCH; correlations evolve
    - Q_t = (1-a-b)Q̄ + a(ε_{t-1} ε'_{t-1}) + b Q_{t-1}
    - R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    - Parameters: `returns_matrix`, `max_iter`
    - Most widely used multivariate GARCH

17. **CCC-GARCH** (Constant Conditional Correlation) — Bollerslev 1990
    - Simpler: each series has univariate GARCH, correlation is constant
    - Parameters: `returns_matrix`, `max_iter`

18. **BEKK-GARCH** — Engle & Kroner 1995
    - H_t = C'C + A'ε_{t-1}ε'_{t-1}A + B'H_{t-1}B
    - Guarantees positive definiteness
    - Parameters: `returns_matrix`, `max_iter`
    - Curse of dimensionality: O(p⁴) parameters for p assets

19. **GO-GARCH** (Generalized Orthogonal) — van der Weide 2002
    - H_t = Λ diag(h_t) Λ'
    - Orthogonal rotation + univariate GARCH
    - More parsimonious than BEKK

20. **ADCC** (Asymmetric DCC) — Cappiello et al. 2006
    - DCC with asymmetric dynamics (leverage in correlations)
    - Parameters: `returns_matrix`, `max_iter`

### F. Stochastic Volatility

21. **SV (basic)** — log σ²_t = μ + φ(log σ²_{t-1} - μ) + η_t
    - State space model, estimated via: particle filter, MCMC, or quasi-MLE
    - Parameters: `returns`, `method` ("qmle" | "particle" | "mcmc")
    - Already have Kalman filter; can use for quasi-MLE

22. **SV-t** — SV with Student-t observation noise
    - Parameters: `returns`, `df_init`

23. **SV with leverage** — corr(ε_t, η_t) = ρ
    - Contemporaneous correlation between return and vol shocks
    - Parameters: `returns`, `method`

24. **Realized SV** — Hansen & Huang 2016
    - Uses realized volatility as additional observation
    - Parameters: `returns`, `realized_vol`, `method`

### G. Regime-Switching

25. **Markov-Switching GARCH** (MS-GARCH) — Haas et al. 2004
    - GARCH parameters switch between regimes
    - Parameters: `returns`, `n_regimes`, `max_iter`
    - Combines: HMM + GARCH estimation

26. **SWARCH** (Switching ARCH) — Hamilton & Susmel 1994
    - Variance level switches, ARCH dynamics within regime
    - Parameters: `returns`, `n_regimes`, `arch_order`, `max_iter`

---

## Decomposition into Primitives

```
log_likelihood(params, returns, model) ──── ALL GARCH fitting

optimizer (L-BFGS / Nelder-Mead) ──────── parameter estimation for all models

variance_recursion:
  garch:    σ²_t = ω + α ε² + β σ²
  egarch:   log σ²_t = ω + α f(z) + β log σ²
  gjr:      σ²_t = ω + α ε² + γ ε² I(ε<0) + β σ²
  tgarch:   σ_t = ω + α|ε| + γ ε⁻ + β σ
  aparch:   σ^δ_t = ω + α(|ε|-γε)^δ + β σ^δ

distribution_log_pdf:
  normal:   -½(z² + log 2π)
  t:        log Γ((ν+1)/2) - log Γ(ν/2) - ... (uses log_gamma, digamma)
  ged:      ...
  skew_t:   ...

constraint_check:
  stationarity:  α + β < 1 (GARCH), α + β = 1 (IGARCH)
  positivity:    ω > 0, α ≥ 0, β ≥ 0
  leverage:      -1 < γ < 1 (EGARCH)
```

## Intermediate Sharing

| Intermediate | Consumers |
|---|---|
| Squared returns (ε²) | All GARCH variants, ARCH-LM test |
| Conditional variance series (σ²_t) | Forecasting, VaR, standardized residuals |
| Standardized residuals (z_t = ε_t/σ_t) | Distribution fit, diagnostic tests |
| Log-likelihood at MLE | AIC/BIC model selection |
| News impact curve | Asymmetry analysis |

## Priority

**Tier 1** — Most requested variants:
1. `garch_pq(returns, p, q)` — general order
2. `garch_t_fit(returns)` — Student-t innovations (by far most common in practice)
3. `aparch_fit(returns)` — nests all asymmetric models
4. `dcc_garch(returns_matrix)` — multivariate, most widely used

**Tier 2**:
5. `igarch_fit` — integrated (RiskMetrics generalization)
6. `cgarch_fit` — component model (permanent + transitory)
7. `ccc_garch` — simpler multivariate
8. `garch_skew_t_fit` — most realistic for financial returns
9. `sv_basic` — stochastic volatility via Kalman

**Tier 3**:
10-26: FIGARCH, HYGARCH, BEKK, GO-GARCH, MS-GARCH, etc.
