# Hardcoded Constants — Mathematical Justifications

For each violation in the hardcoded-constants-audit, this document provides:
- The mathematical/statistical origin of the default value
- Whether the value is a convention, an optimization, or mathematically determined
- Literature references for the default
- Guidance on when users should override

---

## V-01: Vol clustering ACF threshold = 0.05

**Origin**: This is simply "statistically significant at α=0.05 for a correlation coefficient."
For n observations, the 95% CI for ACF under null is ±1.96/√n. For n=100, that's ±0.196.
The 0.05 threshold is much looser than this — it catches very weak clustering.

**Convention vs math**: Convention. Practitioners use 0.05-0.15 depending on context.

**When to override**: Financial tick data with known microstructure noise → higher threshold (0.10-0.15). Long-memory processes → lower threshold (0.02-0.03).

## V-02: Jump ratio k=3.0, Vol ACF=0.1, Trend R²=0.5

**Jump k=3.0**: Based on the "3-sigma rule" — events >3σ from mean have P < 0.003 under normality. For fat-tailed financial data, 3σ events are more common, so this threshold is conservative.

**Vol ACF=0.1**: Different from V-01's 0.05. This is in the summary constructor where a higher bar makes sense (summary should flag obvious clustering, not borderline cases).

**Trend R²=0.5**: "The trend explains at least 50% of variance." This is a fairly strong threshold. For drift detection in financial bins, 0.3 might be more appropriate. For long time series with clear seasonality, 0.7+ is common.

## V-03: Normality test n-threshold = 5000

**Origin**: Shapiro-Wilk has O(n²) complexity and numerical instability for large n. D'Agostino-Pearson is O(n) and well-behaved at any n. The crossover at 5000 is a practical choice.

**Literature**: Shapiro-Wilk is more powerful for n < 2000 (Razali & Wah 2011). D'Agostino becomes competitive around n=1000-2000 and is preferred above n=5000 for computational reasons.

**When to override**: If using pre-sorted data (Shapiro-Wilk's sort is free), raise to 50000. If speed matters, lower to 1000. If power matters and you can afford the computation, always use Shapiro-Wilk.

## V-04: Normality alpha = 0.05

**Origin**: Fisher (1925). The 0.05 convention is the most widely used significance level.

**When to override**: 
- Exploratory analysis: α=0.10 (more permissive, catches marginal non-normality)
- Confirmatory: α=0.01 (stricter, fewer false positives)
- Bayesian alternative: don't use a threshold at all — use Bayes factors

**Important**: In the auto-detection pipeline, this alpha cascades: it determines whether the pipeline selects parametric vs nonparametric methods. Making it stricter (0.01) means MORE parametric tests are used (because fewer datasets are flagged as non-normal). Making it looser (0.10) means MORE nonparametric tests.

## V-05: VIF threshold = 10.0

**Origin**: Marquardt (1970) and Belsley et al. (1980). VIF=10 corresponds to R²=0.90 between a predictor and all others. Common textbook rule of thumb.

**Alternatives**: 
- VIF > 5: "moderate" multicollinearity (Hair et al. 2010)
- VIF > 10: "severe" (Marquardt 1970)
- 1/(1-R²) > 10: equivalent formulation
- Some economists use VIF > 30 for macroeconomic data where correlated regressors are expected

## V-06: KMO threshold = 0.5

**Origin**: Kaiser (1974). KMO classification:
- < 0.5: "unacceptable" 
- 0.5-0.6: "miserable"
- 0.6-0.7: "mediocre"
- 0.7-0.8: "middling"
- 0.8-0.9: "meritorious"
- > 0.9: "marvelous"

The 0.5 threshold = "acceptable for factor analysis to proceed." Strictly, this should be paired with Bartlett's test (already is).

## V-07: Hopkins threshold = 0.5

**Origin**: Hopkins (1954). Hopkins statistic H ∈ [0,1]:
- H ≈ 0.5: data resembles uniform random (no cluster structure)
- H → 1.0: strong cluster structure
- H → 0.0: anti-clustering (uniform lattice)

The threshold H > 0.5 means "more clustered than random." This is the minimal requirement. In practice, H > 0.7 is more convincing evidence of clusters.

## V-11: EWMA lambda = 0.94

**Origin**: RiskMetrics Technical Document (1996), J.P. Morgan. The "RiskMetrics daily" lambda = 0.94 is derived from minimizing the MSE of 1-day variance forecasts on a large portfolio of financial instruments.

**Convention**: This is THE canonical value for daily financial returns. For monthly: λ = 0.97. For intraday: λ = 0.90-0.93. For non-financial data: no canonical default.

## V-16: IRT Newton-Raphson inner steps = 10

**Origin**: Empirical. The 2PL joint MLE (EM-like alternation between person and item parameters) converges in 3-5 inner Newton steps for well-identified items. 10 is generous.

**When to override**: For very sparse response matrices (many missing), increase to 20-30. For fully observed matrices, 5 is sufficient.

## V-17: IRT parameter bounds

**Discrimination a ∈ [0.1, 5.0]**: 
- a < 0.1: Item doesn't discriminate at all (flat ICC). Practically useless.
- a > 5.0: Item is so sharp it acts like a step function. Rare but legitimate in CAT.
- Baker & Kim (2004) suggest a ∈ [0.25, 3.5] for typical educational testing.

**Difficulty b ∈ [-5.0, 5.0]**: 
- On the standard normal θ scale, ±5 covers 99.99997% of the ability distribution.
- In operational CAT: b ∈ [-3, 3] covers the usable range. Items outside this are too easy/hard.

**Ability θ ∈ [-6.0, 6.0]**: 
- Wider than b because ability can be estimated beyond the item range.
- EAP estimates are bounded by the prior, but MLE can drift to ±∞ for perfect response patterns.

## V-19: pinv SVD threshold = 1e-12

**Origin**: This is an absolute threshold, which is generally wrong. The standard approach (MATLAB, numpy) uses a RELATIVE threshold:

rcond = max(m,n) × max(σ) × ε_machine

where ε_machine ≈ 2.2e-16 for f64. For a 100×100 matrix with max σ = 1.0, this gives rcond ≈ 2.2e-14.

The 1e-12 absolute threshold is fine for well-scaled matrices (max σ ≈ 1) but wrong for matrices with large singular values (e.g., if σ_max = 1e6, then the correct threshold ≈ 2.2e-8, not 1e-12).

**Recommendation**: Change to relative threshold. This is not just a using() issue — it's a correctness issue for non-unit-scale matrices.

---

## Meta-pattern

Most VIOLATION constants fall into three categories:

1. **Statistical significance thresholds (α=0.05)**: These should ALL flow through `using(alpha=...)`. The default 0.05 is fine but researchers must be able to override.

2. **Method-selection boundaries (n=5000, VIF=10, KMO=0.5, Hopkins=0.5)**: These are convention-dependent heuristics. Each has a published justification but no mathematical certainty. They belong in `using()`.

3. **Numerical tolerances (1e-12, max_iter=10)**: These affect convergence/precision. Power users need them; default users don't. Optional parameters with documented defaults.

The using() wiring for category 1 (alpha) is the highest priority — it affects method selection and p-value interpretation across the entire pipeline.
