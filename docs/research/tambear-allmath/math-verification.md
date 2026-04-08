# Tambear Math Verification Report
**Author**: math-researcher  
**Date**: 2026-04-06  
**Scope**: Verification of implemented algorithms against published mathematical literature

---

## Summary

Verified ~145 algorithms across 26 modules. **All core formulas are mathematically correct with six issues.**  
Issues: 2 bugs (LME σ² M-step, KS test non-standardization), 1 concerning (RNG quality in MCMC), 3 advisory (GARCH optimizer, ADF critical values, Pillai F df2).  
One documentation mismatch: EAP docstring says "Gauss-Hermite" but uses uniform quadrature (math is correct).  
One unimplemented feature: `rng.rs` docstring claims Sobol quasi-random sequences but no implementation exists.  

**Note from observer (2026-04-06)**: 77 additional bugs are documented in test comments in `tests/adversarial_boundary*.rs` but the tests pass (they use `eprintln!` to log the bug then assert a relaxed property). The green CI is not full correctness — it's a snapshot. Panic risks identified in kaplan_meier/cox_ph/log_rank_test via `partial_cmp().unwrap()` on NaN input.

**LME fix status**: An incomplete fix was applied (missing n_g multiplier). The correct fix is specified in Issue #4 below. Tests pass either way due to loose assertions on σ² convergence.

---

## VERIFIED CORRECT

### 1. Nonparametric Statistics (`nonparametric.rs`)

**Kendall's tau-b** (Kendall 1938, Agresti 2010 §10.3):
- Computes concordant C, discordant D, x-ties T_x, y-ties T_y
- Formula: τ_b = (C-D) / √((C+D+T_x)(C+D+T_y))
- ✅ Matches Kendall's tau-b definition exactly (ties in x and y handled correctly; joint ties excluded from all counts)

**KDE** (Parzen 1962, Silverman 1986):
- Gaussian kernel: K(u) = exp(-u²/2) / √(2π) ✅
- Epanechnikov kernel: K(u) = 0.75(1-u²) for |u|≤1 ✅
- Silverman bandwidth: h = 0.9·min(σ̂, IQR/1.34)·n^(-1/5) ✅ (Rule of Thumb B from Silverman 1986 §3.4.2)

### 2. Volatility (`volatility.rs`)

**GARCH(1,1) log-likelihood** (Bollerslev 1986):
- Recursion: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1} ✅
- Log-likelihood: ℓ = Σₜ [-½(ln(2π) + ln(σ²_t) + r²_t/σ²_t)] ✅
- Code: `-0.5 * (TAU.ln() + sigma2[t].ln() + r²/sigma2)` — TAU = 2π, TAU.ln() = ln(2π) ✅
- Backcast: σ²_0 = sample variance ✅ (standard initialization)

**GARCH(1,1) forecasting** (Bollerslev 1986 §5):
- h=1 step: σ²_{t+1} = ω + α·r²_t + β·σ²_t ✅
- h>1: E[σ²_{t+h}] = ω + (α+β)·E[σ²_{t+h-1}] ✅ (since E[r²_{t+h-1}] = E[σ²_{t+h-1}] under normality)
- Unconditional variance: σ²_∞ = ω/(1-α-β) ✅

**Bipower variation** (Barndorff-Nielsen & Shephard 2004, Eq. 2.2):
- BV = μ₁⁻² Σ|r_t||r_{t-1}| where μ₁ = √(2/π) = E[|Z|], Z~N(0,1)
- Code: `mu1 = sqrt(2/π); sum / (mu1 * mu1)` ✅ (equivalent to π/2 × Σ|r_t||r_{t-1}|)

**Roll spread** (Roll 1984, JF):
- S = 2√(-Cov(ΔP_t, ΔP_{t-1})) ✅
- Returns 0 when Cov ≥ 0 (no adverse selection signal) ✅

### 3. Hypothesis Testing (`hypothesis.rs`)

**One-sample t-test**: t = (x̄ - μ₀) / (s/√n), df = n-1 ✅  
**Two-sample t-test (pooled)**:  
- pooled_var = (SS₁ + SS₂) / (n₁+n₂-2) ✅
- SE = √(pooled_var · (1/n₁ + 1/n₂)) ✅

**Welch's t-test** (Welch 1947, Satterthwaite 1946):
- SE = √(s²₁/n₁ + s²₂/n₂) ✅
- df = (s²₁/n₁ + s²₂/n₂)² / [(s²₁/n₁)²/(n₁-1) + (s²₂/n₂)²/(n₂-1)] ✅ (Welch-Satterthwaite exactly)

**One-way ANOVA**:
- SS_between = Σ nᵢ(x̄ᵢ - x̄)² ✅
- SS_within = Σ m2ᵢ (sum of squared deviations within groups) ✅
- F = (SS_between/(k-1)) / (SS_within/(N-k)) ✅

### 4. Bayesian Methods (`bayesian.rs`)

**Metropolis-Hastings** (Metropolis et al. 1953, Hastings 1970):
- Box-Muller transform for proposals ✅
- log α = log π(x*) - log π(x) ✅ (symmetric Gaussian proposal cancels)
- Accept if ln(U) < log_α ✅

**Bayesian linear regression** (Normal-InverseGamma conjugate, Murphy 2007):
- Λ_n = Λ₀ + X'X ✅
- β_n = Λ_n⁻¹(Λ₀β₀ + X'y) ✅
- α_n = α₀ + n/2 ✅
- b_n = b₀ + ½(y'y + β₀'Λ₀β₀ - β_n'Λ_nβ_n) ✅
- E[σ²] = b_n/(α_n - 1) ✅ (mean of Inverse-Gamma)
- Cov(β|σ²̂) = σ²̂ · Λ_n⁻¹ ✅

**ESS via Geyer IMSE** (Geyer 1992, Gelman et al. BDA3 §11.5):
- Pairs ρ_{2k-1} + ρ_{2k} > 0 with monotone enforcement ✅
- ESS = n / (1 + 2Σ ρ_k) ✅

**R-hat** (Gelman & Rubin 1992):
- B = n/(m-1) Σ(θ̄ⱼ - θ̄)² ✅
- W = (1/m) Σ sⱼ² ✅
- V̂ar = (n-1)/n · W + B/n ✅
- R̂ = √(V̂ar/W) ✅

### 5. Survival Analysis (`survival.rs`)

**Kaplan-Meier** (Kaplan & Meier 1958):
- Ŝ(t) = ∏_{tᵢ≤t} (1 - dᵢ/nᵢ) ✅
- Greenwood SE: Ŝ(t) · √(Σ dᵢ/(nᵢ(nᵢ-dᵢ))) ✅

**Cox PH partial likelihood** (Cox 1972, Breslow 1974 for ties):
- Gradient: ∂ℓ/∂β = Σ_{events} [xᵢ - S₁(t)/S₀(t)] ✅
- Hessian: -Σ_{events} [S₂/S₀ - S₁S₁'/S₀²] ✅
- Risk set processing: forward scan with removal ✅ (equivalent to backward accumulation)
- Breslow tie handling: tied events share same risk set ✅
- Log-sum-exp trick for numerical stability ✅

### 6. Time Series (`time_series.rs`)

**ADF test** (Dickey & Fuller 1979, augmented):
- Regression model: Δy_t = α + γ·y_{t-1} + Σⱼ δⱼ·Δy_{t-j} + ε_t ✅
- t-statistic on γ coefficient ✅
- Lagged differences dy[t-1-j] for j=0,...,p-1 ✅
- Critical values: MacKinnon asymptotic for "constant" model ✅

### 7. Dimension Reduction (`dim_reduction.rs`)

**PCA via SVD** (Hotelling 1933, modern form):
- Center → SVD(X_c) = U·Σ·Vᵀ ✅
- Components = V columns ✅
- Explained variance ratio = σᵢ²/Σσⱼ² ✅
- Transformed = X_c · V ✅

**Classical MDS** (Torgerson 1952):
- B = -½ · J · D² · J (double centering) ✅
- B_{ij} = -½(d²_{ij} - d̄²_{i.} - d̄²_{.j} + d̄²_{..}) ✅

### 8. Information Theory (`information_theory.rs`)

**Shannon entropy**: H = -Σ pᵢ ln(pᵢ), with 0·ln(0) = 0 ✅  
**Rényi entropy**: H_α = (1/(1-α))·ln(Σpᵢ^α), with correct α→1 and α→∞ limits ✅  
**Tsallis entropy**: S_q = (1/(q-1))(1 - Σpᵢ^q), with q→1 limit ✅  
**KL divergence**: D(P||Q) = Σ pᵢ·ln(pᵢ/qᵢ), +∞ when q=0,p>0 ✅  
**JS divergence**: ½KL(P||M) + ½KL(Q||M), M=(P+Q)/2 ✅  
**Cross entropy**: H(P,Q) = -Σ pᵢ·ln(qᵢ) ✅

### 9. Series Acceleration (`series_accel.rs`)

**Aitken Δ²**: S'_n = S_n - (ΔS_n)²/Δ²S_n ✅ (Aitken 1926)  
**Wynn epsilon**: ε_{k+1}(n) = ε_{k-1}(n+1) + 1/(ε_k(n+1) - ε_k(n)) ✅ (Wynn 1956)
- Early stopping on instability ✅ (prevents catastrophic numerical blow-up)
- Even columns = Shanks transforms ✅

### 10. Complexity Measures (`complexity.rs`)

**Sample entropy** (Richman & Moorman 2000):
- SampEn = -ln(A/B) where A = m+1 template matches, B = m template matches ✅
- L∞ (Chebyshev) distance, no self-matches ✅
- A/B ratio unaffected by counting i<j vs all ordered pairs ✅

**Approximate entropy** (Pincus 1991):
- ApEn = φ^m(r) - φ^(m+1)(r), φ^m = (1/N') Σ ln(Cᵢ^m(r)/N') ✅
- Includes self-matches ✅

**Permutation entropy** (Bandt & Pompe 2002):
- Lehmer code via Horner's method: `index = index*(m-i) + count` ✅
- count = #{j > i : pattern[j] < pattern[i]} — correct factorial number system ✅
- PE = -Σ p(π) ln(p(π)) normalized by ln(m!) ✅

**Hurst R/S** (Hurst 1951):
- Cumulative deviations from mean ✅
- R/S = (max-min cum. dev.) / sample std ✅ (note: uses sample std, ddof=1 — standard convention)
- H = OLS slope log(R/S) vs log(n) ✅

**DFA** (Peng et al. 1994):
- Profile = mean-subtracted cumulative sum ✅
- Linear detrending within windows (correct OLS using centered formulation) ✅
- **NOTE**: Code uses AM of per-window F (not RMS). Both AM and RMS give the same slope α in log-log, so the Hurst-analog exponent is unaffected. Only the prefactor differs. This is acceptable and noted in literature.

**Higuchi FD** (Higuchi 1988):
- L(k) = (1/k) Σₘ Lₘ(k) with normalization factor (N-1)/(n_seg × k) ✅
- FD = -slope(log L vs log k) → code: `-ols_slope(...)` ✅

**Correlation dimension** (Grassberger-Procaccia 1983):
- L∞ norm for pairwise distances ✅ (standard for GP algorithm)
- C(r) = count / (N(N-1)/2) = proportion of pairs within distance r ✅
- r range: 10th to 90th percentile of distances ✅ (avoids edge effects, standard practice)
- D₂ = slope log(C(r)) vs log(r) ✅

**Largest Lyapunov (Rosenstein 1993)**:
- Takens delay embedding ✅
- Nearest neighbor excluding temporal neighbors within mean_period ✅
- Average log divergence across pairs ✅
- λ₁ = slope of mean log-divergence vs time using first ~1/3 of trajectory (linear scaling region) ✅

### 11. Mixed Effects (`mixed_effects.rs`)

**Henderson's mixed model equations** (Henderson 1950, Laird & Ware 1982):
- Henderson system built correctly: [X'X, X'Z; Z'X, Z'Z+λI] ✅
- λ = σ²/σ²_u regularization on Z'Z diagonal ✅
- Augmented design [X̃|Z] where X̃ includes intercept column ✅
- Cholesky solve gives BLUE (β) and BLUP (u) jointly ✅

**EM M-step for σ²_u**:
- σ²_u_new = (||u||² + Σ_g τ_g²) / k ✅
- τ_g² = σ²σ²_u / (n_g σ²_u + σ²) (posterior variance of u_g|y) ✅
- See Issue 4 for σ² update (has a bug)

**ICC** (Laird & Ware 1982):
- ICC = σ²_u / (σ²_u + σ²) ✅ (standard definition)

**ICC one-way ANOVA** (Shrout & Fleiss 1979, ICC(1,1)):
- MS_between = Σ_g n_g(ȳ_g - ȳ)² / (k-1) ✅
- MS_within = Σ_i (y_i - ȳ_{g(i)})² / (n-k) ✅
- n₀ = (N - Σn_g²/N) / (k-1) — harmonic effective group size ✅
- ICC = max(0, (MS_B - MS_W) / (MS_B + (n₀-1)MS_W)) ✅

**Design effect** (Kish 1965):
- DEFF = 1 + (m-1)·ICC ✅

### 12. Special Functions (`special_functions.rs`)

**erf / erfc** (Abramowitz & Stegun 7.1.26):
- Rational approximation: p=0.3275911, (a1,...,a5) = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429) ✅
- erfc uses direct formula for x>0 to avoid 1-erf cancellation ✅
- erfc(negative x) = 2 - erfc(-x) ✅

**log_gamma** (Lanczos, g=7, Godfrey coefficients):
- Reflection: log Γ(x) = ln(π) - ln|sin(πx)| - log Γ(1-x) for x<0.5 ✅
- Lanczos: A_g(z) = C[0] + Σ C[k]/(z+k), t = z + 7.5, result = ½ln(2π) + (z+½)ln(t) - t + ln(A_g) ✅

**digamma** (A&S 6.3; asymptotic expansion):
- Shift to x≥8 via recurrence ψ(x+1)=ψ(x)+1/x ✅
- Asymptotic: ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + 1/(240x⁸) - 1/(132x¹⁰)
  (= Bernoulli series: -B₂/(2x²) - B₄/(4x⁴) - ..., B₂=1/6, B₄=-1/30, B₆=1/42, B₈=-1/30, B₁₀=5/66) ✅
- Reflection: ψ(x) = ψ(1-x) - π·cot(πx) ✅

**trigamma**:
- Shift to x≥8 via recurrence ψ₁(x+1)=ψ₁(x)-1/x² ✅
- Asymptotic: 1/x + 1/(2x²) + Σ B_{2n}/x^{2n+1} = 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - 1/(30x⁹) + 5/(66x¹¹) ✅

**regularized_incomplete_beta I_x(a,b)** (Lentz's CF algorithm):
- Symmetry swap x↔(1-x) at crossover (a+1)/(a+b+2) for CF convergence ✅
- Front factor x^a(1-x)^b/(a·B(a,b)) ✅
- Lentz CF: alternating even/odd steps per Press et al. §6.4 ✅

**regularized_gamma P(a,x) / Q(a,x)**:
- Series for x < a+1; CF for x ≥ a+1 ✅ (standard A&S crossover)
- Series: e^(-x) x^a Σ x^n/Γ(a+n+1) via recurrence ✅
- CF: modified Lentz for Q(a,x) ✅

**CDFs from special functions**:
- normal_cdf(x) = ½·erfc(-x/√2) ✅ (exact)
- normal_sf(x) = ½·erfc(x/√2) ✅ (avoids cancellation for large x)
- t_cdf(t, ν): P(T≤t) = 1 - ½·I_{ν/(ν+t²)}(ν/2, 1/2) for t≥0 ✅
- f_cdf(x, d₁, d₂): I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2) ✅
- chi2_cdf(x, k): P(k/2, x/2) ✅ (Gamma(k/2,2) connection)

All special functions verified against A&S, Press et al. "Numerical Recipes", and known analytic values.

### 13. Multivariate Analysis (`multivariate.rs`)

**Hotelling's T² — one-sample** (Hotelling 1931):
- T² = n·d'·S⁻¹·d ✅
- F = (n-p)/((n-1)p) · T², df=(p, n-p) ✅

**Hotelling's T² — two-sample** (Hotelling 1931):
- Pooled cov: S_p = ((n₁-1)S₁ + (n₂-1)S₂)/(n₁+n₂-2) ✅
- T² = n₁·n₂/(n₁+n₂) · d'·S_p⁻¹·d ✅
- F = (n-p-1)/((n-2)p) · T², df=(p, n-p-1) ✅ (Rencher 2002, Eq. 5.9)

**MANOVA** (Wilks 1932; Pillai 1955; Hotelling 1951; Roy 1953):
- M = L⁻¹·H·L⁻ᵀ where W = LL' (Cholesky), eigenvalues(M) = eigenvalues(HW⁻¹) ✅
- Wilks' Λ = Π 1/(1+θᵢ) ✅
- Pillai's trace = Σ θᵢ/(1+θᵢ) ✅ (trace of H(H+W)⁻¹)
- Hotelling-Lawley = Σ θᵢ ✅ (trace of HW⁻¹)
- Roy's largest root = max(θᵢ) ✅
- Pillai F approximation: df1=s·p, df2=s·(N-k) — see Issue 5 below

**LDA** (Fisher 1936):
- Generalized eigenvalue H·v = λ·W·v via Cholesky transform ✅
- Back-transform: axes[:,k] = L⁻ᵀ·u_k where u_k = eigenvector of M ✅
- Nearest-centroid classification in discriminant space ✅

**CCA** (Hotelling 1936; Anderson 1984 §12):
- Canonical correlations = singular values of L_X⁻¹·Σ_XY·L_Y⁻ᵀ ✅
- X weights a_k = L_X⁻ᵀ·u_k, Y weights b_k = L_Y⁻ᵀ·v_k ✅
- Wilks' Λ = Π(1-ρ²ᵢ) ✅
- Bartlett χ² = -(n-1-(p+q+1)/2)·ln(Λ), df = p·q ✅ (Bartlett 1938)

**Mardia's multivariate normality** (Mardia 1970):
- b₁,p = (1/n²)Σᵢ Σⱼ dᵢⱼ³ where dᵢⱼ = (xᵢ-x̄)'S⁻¹(xⱼ-x̄) ✅
- b₂,p = (1/n)Σᵢ dᵢᵢ² ✅ (squared Mahalanobis distances)
- Skewness test: n·b₁/6 ~ χ²(p(p+1)(p+2)/6) ✅
- Kurtosis test: z = (b₂-p(p+2)) / √(8p(p+2)/n) ~ N(0,1) ✅

### 14. Item Response Theory (`irt.rs`)

**IRT probability models** (Lord & Novick 1968; Birnbaum 1968):
- 1PL (Rasch): P(θ,b) = σ(θ-b) ✅
- 2PL: P(θ,a,b) = σ(a(θ-b)) ✅
- 3PL: P = c + (1-c)·σ(a(θ-b)) ✅
- logistic(x) = 1/(1+exp(-x)) ✅

**Joint MLE for 2PL** (Birnbaum 1968 §17):
- Initialization: a=1, b=-logit(p̂ⱼ) where p̂ⱼ = proportion correct ✅
- E-step (ability NR): grad=Σⱼaⱼ(rᵢⱼ-pᵢⱼ), hess=-Σⱼaⱼ²pᵢⱼ(1-pᵢⱼ), prior θ~N(0,1) adds -θ and -1 ✅
- M-step (item NR): grad_a=Σᵢ(θᵢ-b)(rᵢⱼ-pᵢⱼ), grad_b=Σᵢ(-a)(rᵢⱼ-pᵢⱼ), diagonal hessian ✅
- Note: uses diagonal Newton (ignores cross-derivative ∂²ℓ/(∂a∂b)) — standard approximation ✅

**EAP ability** (Bock & Mislevy 1982):
- Uniform quadrature over [-4,4] with N(0,1) prior ✅
- Log-sum-exp for numerical stability ✅
- **Advisory**: docstring says "Gauss-Hermite quadrature" but implementation uses uniform spacing (Riemann sum). Mathematically valid; GH would be more efficient but gives same result as n_quad→∞.

**Item information** (Lord 1980, §12):
- I(θ) = a²·P(θ)·(1-P(θ)) ✅ (Fisher information for 2PL)
- SEM = 1/√I(θ) ✅

**Mantel-Haenszel DIF** (Holland & Thayer 1988):
- log(OR_MH) = ln(Σ_k A_k·D_k/N_k) - ln(Σ_k B_k·C_k/N_k) ✅

### 15. Neural Network Operations (`neural.rs`)

**Activations** (all verified against primary sources):
- ReLU, Leaky ReLU, ELU: standard definitions ✅
- SELU (Klambauer 2017): constants α=1.6732632423543772, λ=1.0507009873554805 ✅ (exact paper values)
- GELU tanh approx (Hendrycks & Gimpel 2016): 0.5x(1+tanh(√(2/π)(x+0.044715x³))) ✅
- GELU backward: 0.5(1+tanh) + 0.5x·sech²·d_inner, d_inner=√(2/π)(1+3·0.044715·x²) ✅
- Swish/SiLU (Ramachandran 2017): x·σ(x) ✅, backward: σ+xσ(1-σ) ✅
- Sigmoid: numerically stable (both branches for x≥0 and x<0) ✅
- Softplus: stable with x>20 and x<-20 cutoffs ✅
- Hard sigmoid: clamp((x+3)/6, 0, 1) ✅ (standard PyTorch convention)
- Softmax: max-stable subtract ✅
- Log-softmax: log-sum-exp stable ✅

**Normalization**:
- Layer norm (Ba 2016): biased variance (÷d, not d-1) ✅ (standard convention)
- RMS norm (Zhang 2019): scale by 1/RMS(x) without centering ✅
- Group norm: within-group statistics ✅
- Batch norm: channel-wise mean/variance ✅

**Attention** (Vaswani 2017):
- scores = Q·Kᵀ/√d_k ✅
- Causal mask: upper triangle → -∞ ✅ (positions j>i masked)
- Row-wise softmax ✅
- output = weights·V ✅
- Multi-head: project Q,K,V → split by head dimension → concat → output projection ✅

**Loss functions**:
- MSE = Σ(pred-target)²/n ✅; backward = 2(pred-target)/n ✅
- BCE = -[y·ln(p) + (1-y)·ln(1-p)]/n with clamping ✅

### 16. Optimization (`optimization.rs`)

**Backtracking line search** (Armijo 1966):
- f(x+αd) ≤ f(x) + c₁·α·∇f(x)'d ✅ (Armijo sufficient decrease)
- Contract by ρ until satisfied ✅

**Golden section search** (Kiefer 1953):
- φ = (√5-1)/2 ≈ 0.618 ✅
- x₁ = b - φ(b-a), x₂ = a + φ(b-a) ✅
- Standard bracketing update ✅

**Gradient descent with momentum** (Polyak 1964):
- v_t = γv_{t-1} + ∇f, x_t = x_{t-1} - lr·v_t ✅

**Adam** (Kingma & Ba 2014):
- m_t = β₁m_{t-1} + (1-β₁)g_t ✅
- v_t = β₂v_{t-1} + (1-β₂)g²_t ✅
- m̂_t = m_t/(1-β₁ᵗ), v̂_t = v_t/(1-β₂ᵗ) (bias correction) ✅
- x_t = x_{t-1} - lr·m̂_t/(√v̂_t + ε) ✅

**AdaGrad** (Duchi et al. 2011):
- G_i += g²_i; x_i -= lr·g_i/(√G_i + ε) ✅

**RMSProp** (Hinton 2012):
- v_i = decay·v_i + (1-decay)·g²_i; x_i -= lr·g_i/(√v_i + ε) ✅

**L-BFGS** (Liu & Nocedal 1989):
- Two-loop recursion (Algorithm 7.4, Nocedal & Wright): ✅
  - First loop (backward): α_i = ρ_i·s_i'q; q -= α_i·y_i
  - Scaling: γ = s_{k-1}'y_{k-1}/y_{k-1}'y_{k-1}; q *= γ
  - Second loop (forward): β = ρ_i·y_i'q; q += (α_i-β)s_i
  - direction = -q ✅
- ρ_i = 1/(s_i'y_i) — only stored if s_i'y_i > 1e-14 (curvature condition) ✅
- Ring buffer of m (s,y) pairs; removes oldest when full ✅
- Armijo line search ✅

**Nelder-Mead** (Nelder & Mead 1965): standard simplex operations (reflect, expand, contract, shrink). ✅

All optimization algorithms verified against Nocedal & Wright "Numerical Optimization" (2006).

### 17. Signal Processing (`signal_processing.rs`)

**Cooley-Tukey radix-2 DIT FFT** (Cooley & Tukey 1965):
- Bit-reversal permutation ✅
- Sign convention: forward exp(-2πi/N), inverse exp(+2πi/N) ✅ (numpy convention)
- Butterfly: u=data[i+j], t=w·data[i+j+half], data[i+j]=u+t, data[i+j+half]=u-t ✅
- IFFT normalization: divide by N ✅
- Standard DFT convention: X[k] = Σⱼ x[j]·exp(-2πijk/N); x[j] = (1/N)Σₖ X[k]·exp(+2πijk/N) ✅

**Real FFT / IRFFT**:
- rfft: pack as complex, FFT, truncate to N/2+1 ✅ (Hermitian symmetry exploitation)
- irfft: mirror conjugate, IFFT, take real part ✅

**Window functions** (Harris 1978):
- Hann: 0.5(1-cos(2πn/(N-1))) ✅
- Hamming: 0.54-0.46cos(2πn/(N-1)) ✅
- Blackman: a0=0.42, a1=0.5, a2=0.08 ✅

**Daubechies-4 wavelet** (Daubechies 1988):
- Scaling filter: h = [(1+√3), (3+√3), (3-√3), (1-√3)] / (4√2) ✅
- QMF wavelet filter: g[k] = (-1)^k h[3-k] = [h3, -h2, h1, -h0] ✅
- Analysis (DWT): periodic convolution-downsampling with h and g ✅
- Synthesis (IDWT): transpose convolution via polyphase ✅
- Roundtrip tested ✅

**Goertzel algorithm**:
- k = freq·N/fs; w = 2πk/N; coeff = 2cos(w) ✅ (standard form for single-bin DFT)

All signal processing formulas verified against primary sources and numpy conventions.

### 18. Spectral Gap (`spectral_gap.rs`)

**Arnoldi iteration** (Arnoldi 1951, modern Krylov form):
- Modified Gram-Schmidt orthogonalization ✅
- DGKS re-orthogonalization (one round) ✅ (Parlett & Kahan, stability improvement)
- Krylov breakdown detection (h_{k+1,k} < tol) ✅
- Upper Hessenberg matrix formed correctly ✅

**QR iteration for Hessenberg eigenvalues**:
- Implicit single-shift QR ✅
- Wilkinson shift for quadratic convergence ✅

### 19. Hypothesis Testing (`hypothesis.rs`)

**t-tests** (Student 1908; Welch 1947):
- One-sample: t = (x̄ - μ₀)/(s/√n), df = n-1 ✅
- Two-sample (Student): pooled s²_p = (SS₁+SS₂)/(n₁+n₂-2); t with df=n₁+n₂-2 ✅
- Welch's: t = (x̄₁-x̄₂)/√(s²₁/n₁+s²₂/n₂); Satterthwaite df ✅
- Paired: reduce to one-sample on differences ✅

**One-way ANOVA** (Fisher 1925):
- SS_B = Σ nᵢ(x̄ᵢ-x̄)²; SS_W = Σᵢ Σⱼ(yᵢⱼ-x̄ᵢ)² = sum of M2 ✅
- F = (SS_B/(k-1)) / (SS_W/(N-k)) ✅

**Chi-square tests** (Pearson 1900):
- Goodness of fit: χ² = Σ(O-E)²/E, df=k-1 ✅
- Independence: expected = row_total·col_total/grand_total, same formula ✅

**Multiple comparisons** (Bonferroni 1935; Holm 1979; Benjamini & Hochberg 1995):
- Bonferroni: p_adj = min(m·p, 1) ✅
- Holm step-down: sorted ascending, p_adj[rank] = max(adj[0..rank], each ≥ p[rank]·(m-rank)) ✅
- BH step-up (FDR): sorted ascending, p_adj[rank] = min(p[rank]·m/rank, running_min from top) ✅

**Effect sizes** (Cohen 1988; Glass 1976; Hedges 1981):
- Cohen's d = (x̄₁-x̄₂) / s_pooled ✅ (Bessel-corrected pooled SD)
- Glass's Δ = (x̄_treatment - x̄_control) / s_control ✅
- Hedges' g = d · Γ((n-2)/2)·√(n-2/2) / Γ((n-3)/2)·√(n/2) ≈ d·(1 - 3/(4(n-2)-1)) ✅
- Point-biserial r ≡ Pearson r between binary group indicator and continuous var ✅

### 20. Nonparametric Tests (`nonparametric.rs`)

**Mann-Whitney U** (Wilcoxon 1945; Mann & Whitney 1947):
- U = Σᵢ∈G₁ Σⱼ∈G₂ 1[xᵢ > yⱼ]; normal approx for large n ✅
- Mean: n₁n₂/2; Var: n₁n₂(n₁+n₂+1)/12 (tie-corrected) ✅

**Wilcoxon signed-rank** (Wilcoxon 1945):
- W+ = sum of positive-difference ranks; normal approx ✅
- Mean: n(n+1)/4; Var: n(n+1)(2n+1)/24 ✅

**Kruskal-Wallis** (Kruskal & Wallis 1952):
- H = [12/(N(N+1)) Σₖ Rₖ²/nₖ] - 3(N+1) where Rₖ = sum of ranks in group k ✅
- H ~ χ²(k-1) under H₀ ✅

**KS tests** (Kolmogorov 1933; Smirnov 1948):
- One-sample: D = max over i of max((i+1)/n - F(x_{(i)}), F(x_{(i)}) - i/n) ✅
- Two-sample: merge-and-max over ECDF differences ✅
- Asymptotic p: P(D > d) = 2Σ_{k=1}^∞ (-1)^{k-1} exp(-2k²z²), z=√n·d ✅
- **See Issue 6**: one-sample version tests against N(0,1) without standardizing data

### 21. Dimensionality Reduction (`dim_reduction.rs`)

**PCA** (Hotelling 1933; Golub & Reinsch 1970 for SVD):
- Center data (column means) ✅
- SVD of centered matrix X = UΣV' ✅
- Components = columns of V (rows of V') ✅
- Explained variance ratio = σᵢ² / Σσⱼ² ✅
- Scores = X_centered · V = UΣ ✅

**Multi-adic analysis** (`multi_adic.rs`):
- v_p(n) = largest k such that p^k | n; v_p(0) = ∞ ✅
- d_p(a,b) = p^{-v_p(a-b)} (non-Archimedean ultrametric) ✅
- p-adic expansion: base-p digits low-order first ✅

### 22. Factor Analysis (`factor_analysis.rs`)

**Principal Axis Factoring** (Thurstone 1947):
- Initial communalities: max |r_jk| for k≠j ✅ (standard approximation)
- Iteration: set diagonal to h², eigendecompose reduced matrix, update h² = Σ_f L²_jf ✅
- Loadings: L_jf = √λ_f · v_jf ✅
- Heywood detection and clamping ✅ (prevents divergence for near-singular input)

**Varimax rotation** (Kaiser 1958):
- Pairwise rotation algorithm: for each (i,j) factor pair ✅
- a=Σ(l_i²-l_j²), b=Σ(2l_il_j), c=Σ((l_i²-l_j²)²-(2l_il_j)²), d=2Σ(l_i²-l_j²)(2l_il_j) ✅
- Angle: θ = ¼ atan2(d-2ab/p, c-(a²-b²)/p) ✅ (Kaiser 1959 formula)
- Rotation: L[:,i]·cos + L[:,j]·sin; -L[:,i]·sin + L[:,j]·cos ✅
- Note: "raw" varimax (no Kaiser normalization); R uses Kaiser normalization by default

**Cronbach's alpha** (Cronbach 1951):
- α = p/(p-1) · (1 - Σσ²_j/σ²_total) ✅
- Both item and total variances use Bessel correction (n-1) ✅

**McDonald's omega** (McDonald 1999):
- ω_h = (Σλ_j)² / ((Σλ_j)² + Σδ_j) ✅ (first-factor loading approximation)
- Bipolar factor detection + absolute value fix ✅ (adversarial-hardened)

**Scree test / Kaiser criterion**: standard implementations ✅

### 24. Random Number Generation (`rng.rs`)

**SplitMix64** (Stafford 2011 — public domain):
- Constant: 0x9e3779b97f4a7c15 (golden ratio) ✅
- Finalizer: Stafford mix13 with constants 0xbf58476d1ce4e5b9, 0x94d049bb133111eb ✅
- Shifts: 30, 27, 31 ✅

**xoshiro256\*\*** (Blackman & Vigna 2019):
- Output: rotl(s[1]*5, 7)*9 ✅ (the double-starstar scrambler, not xoroshiro+)
- State update: matches Blackman-Vigna reference implementation exactly ✅
- Seeding via SplitMix64 to expand 64-bit seed to 256-bit state ✅
- Jump polynomial constants for 2^128 steps match reference ✅
- next_f64: top 53 bits / 2^53 (standard; avoids representability issues) ✅

**LCG (Knuth MMIX)**:
- mul = 6364136223846793005, add = 1442695040888963407 ✅
- Note: documented as "fast, low quality" — not used in any statistical algorithm ✅

**Box-Muller normal sampling** (Box & Muller 1958):
- r = sqrt(-2 ln u₁), θ = 2π u₂, return (r cos θ, r sin θ) ✅
- Guard against u₁ = 0 (returns pair when u₁ > 1e-300) ✅

**Marsaglia-Tsang gamma sampler** (Marsaglia & Tsang 2000):
- d = α - 1/3, c = 1/√(9d) ✅
- Squeeze: u < 1 - 0.0331·x⁴ (0.0331 is from the paper) ✅
- Log acceptance: ln(u) < ½x² + d(1 - v³ + ln(v³)) ✅
- α < 1 case: Gamma(α) = Gamma(α+1) · U^{1/α} ✅

**Derived samplers**: Beta (X/(X+Y)), χ² (Gamma(k/2, 0.5)), t (z/sqrt(χ²/ν)), F — all correct ✅

**Inverse-CDF samplers**:
- Cauchy: x₀ + γ·tan(π(u-½)) ✅
- Geometric (failures before first success): floor(ln(u)/ln(1-p)) ✅
- Exponential: -ln(u)/λ ✅

**Poisson** (Knuth for λ<30, normal approx for λ≥30): ✅ (normal approx acceptable for large λ)

**Fisher-Yates shuffle** (Knuth 1969 §3.4.2): iterate i from n-1 down to 1, swap [i] with uniform [0,i] ✅

**Floyd's sample without replacement** (Floyd 1978): O(k) set-based algorithm ✅

**Weighted sampling**: CDF + binary_search; binary_search_by using total_cmp ✅

**Not implemented (docstring gap)**: Module claims "Sobol quasi-random (low discrepancy)" but no Sobol implementation exists. Low priority gap.

---

### 23. Clustering (`clustering.rs`)

**DBSCAN** (Ester et al. 1996):
- Density: count all j with dist(i,j) ≤ ε (includes self) ✅
- Core points: density ≥ min_samples ✅ (Ester definition includes point itself)
- Union-find over core-core edges ≤ ε ✅
- Border assignment: first reachable core ✅ (valid; standard doesn't specify nearest)
- Noise: unassigned points → label -1 ✅

---

## ISSUES FOUND

### Issue 1: LCG RNG in Metropolis-Hastings (CONCERNING)

**Location**: `bayesian.rs:80` — `lcg_next()` function  
**Problem**: The Metropolis-Hastings sampler uses a hardcoded LCG (Linear Congruential Generator) with seed 12345 as its RNG. LCGs have:
- Poor statistical quality (failed by BigCrush, DIEHARD)
- Only 64-bit period — inadequate for long chains
- Known correlations between consecutive pairs (lattice structure in k dimensions for k-tuples)
- Cannot be seeded from outside the function

**Mathematical consequence**: For high-dimensional targets (large d), the LCG-based proposals will sample from a lattice in d-space, not a continuous Gaussian. This systematically biases the acceptance probabilities and breaks ergodicity in practice.

**Fix**: Replace `lcg_next()` with `Xoshiro256` from `rng.rs`, which passes all BigCrush tests and has 2^256-1 period.

```rust
// In metropolis_hastings signature, add: rng: &mut dyn TamRng
// Or: let mut rng = Xoshiro256::new(seed);
```

**Priority**: High. MCMC correctness depends critically on RNG quality.

---

### Issue 2: GARCH(1,1) Optimizer Quality (ADVISORY)

**Location**: `volatility.rs:70-99`  
**Problem**: The optimizer uses finite-difference gradient ascent with a fixed step size of 1e-5. This is:
- Slow to converge for the GARCH parameter manifold (ω, α, β have very different scales)
- No adaptive step size
- May converge to a local maximum rather than global
- `alpha + beta ≈ 1` (near IGARCH) causes the optimizer to dance along the constraint boundary

**Mathematical consequence**: Parameter estimates may be inaccurate for difficult cases (near-integrated volatility, very short series). The likelihood function value is correct even when the optimizer underperforms.

**Fix**: Replace with L-BFGS (already in `optimization.rs`) using analytical gradient or better finite differences with parameter scaling.

**Priority**: Medium. Results are usable; edge cases may give suboptimal estimates.

---

### Issue 6: `ks_test_normal` Tests Against N(0,1), Not Generic Normality (BUG/DOCUMENTATION)

**Location**: `nonparametric.rs:344-369`  
**Problem**: `ks_test_normal(data)` tests data against the standard normal N(0,1) without standardizing:
```rust
let cdf = normal_cdf(x);   // always N(0,1)
```
A user calling this on data with mean ≠ 0 or σ ≠ 1 will get a test of "does this data come from N(0,1)?" — almost certainly wrong for normality testing.

**Mathematical consequence**: 
- `ks_test_normal` is a valid KS test but against the wrong null hypothesis for most use cases.
- Correct normality test via KS should either: (a) standardize to z=(x-x̄)/s first, then test against N(0,1); or (b) use Lilliefors critical values (Lilliefors 1967) because estimating μ and σ from data inflates Type I error.

**Fix**:
Option A (simple, with caveat): standardize data before testing:
```rust
let mean = data.iter().sum::<f64>() / n as f64;
let std = ...;
let cdf = normal_cdf((x - mean) / std);
```
But then the KS asymptotic p-value is wrong (Lilliefors correction needed).

Option B (correct): use the Lilliefors test with simulated critical values. Lilliefors critical values differ substantially from Kolmogorov's at small n.

**Priority**: Medium-High. The current implementation gives wrong results when mean ≠ 0 or σ ≠ 1. The docstring doesn't warn users about this.

---

### Issue 5: MANOVA Pillai F-statistic df2 (ADVISORY)

**Location**: `multivariate.rs:282-284`  
**Problem**: The Pillai F-approximation always uses `df2 = s*(n-k)`, but the correct denominator depends on whether `s = p` or `s = k-1`:
- When p ≤ k-1 (s=p): df2 = s*(n-k) ✅ (correct)
- When k-1 < p (s=k-1): df2 should be s*(n-p-1), not s*(n-k)

When k-1 < p (more variables than groups minus one), the code uses a slightly too-large df2, making the test slightly anti-conservative (more rejections than nominal).

**Mathematical consequence**: Type I error slightly above nominal α when p > k-1. Most practical MANOVA cases have k > p, so this is a rare edge case. The test statistics themselves (Wilks, Pillai, HL, Roy) are unaffected.

**Fix**: Add a conditional:
```rust
let df2 = if p <= k - 1 { sf * (nf - kf) } else { sf * (nf - pf - 1.0) };
```

**Priority**: Low. Rare edge case; test statistics are correct; F approximation is inherently approximate.

---

### Issue 4: LME Random Intercept σ² M-step Error (BUG)

**Location**: `mixed_effects.rs:146-151`  
**Problem**: The EM M-step for σ² (residual variance) uses σ² where σ²_u should appear in the trace correction numerator.

The standard EM derivation (Dempster, Laird & Rubin 1977; Laird & Ware 1982) for the random intercept model gives:
```
σ²_new = [||y - Xβ - Zu||² + Σ_g n_g · τ_g²] / n
```
where the posterior variance of u_g|y is `τ_g² = σ² σ²_u / (n_g σ²_u + σ²)`.

So the trace correction should be `Σ_g n_g · σ² σ²_u / (n_g σ²_u + σ²)`.

The code computes:
```rust
let trace_correction = Σ_g (1 - ng * sigma2_u / (ng * sigma2_u + sigma2))
                     = Σ_g sigma2 / (ng * sigma2_u + sigma2)

sigma2_new = ss_resid / n + sigma2 * trace_correction / n
           = ss_resid / n + Σ_g sigma2² / (n * (ng * sigma2_u + sigma2))
```

But the correct formula gives:
```
sigma2_new = ss_resid / n + Σ_g ng * sigma2 * sigma2_u / (n * (ng * sigma2_u + sigma2))
```

The code has `sigma2` in the numerator where it should be `ng * sigma2_u`.

**Mathematical consequence**: The σ² estimate will be biased. The σ²_u update (lines 154-159) is correct. Because σ² and σ²_u are jointly updated, both estimates will be off, but the ICC (σ²_u/(σ²_u+σ²)) may still land in a reasonable range qualitatively. The bias is most severe when σ²_u << σ² (weak group effect), which is precisely the case where accurate variance component estimation matters most.

**Fix**: Change the trace correction to compute the correct quantity:
```rust
// Correct trace correction for σ² update
let trace_correction_sigma2: f64 = (0..k).map(|g| {
    let ng = n_g[g] as f64;
    ng * sigma2_u / (ng * sigma2_u + sigma2)  // = ng * tau_g² / sigma2
}).sum::<f64>() * sigma2_u;  // * sigma2_u gives ng * sigma2 * sigma2_u / (ng*sigma2_u + sigma2) when divided by sigma2

// Then:
let sigma2_new = (ss_resid + sigma2 * trace_correction_sigma2) / n as f64;
```

Or more directly:
```rust
let trace_correction: f64 = (0..k).map(|g| {
    let ng = n_g[g] as f64;
    let tau2_g = sigma2 * sigma2_u / (ng * sigma2_u + sigma2);
    ng * tau2_g   // CRITICAL: ng * tau2_g, not just tau2_g
}).sum::<f64>();
let sigma2_new = (ss_resid + trace_correction) / n as f64;
```

**Derivation of n_g factor**: The E-step quantity is trace(Z·Var(u|y)·Z'). For the random intercept model, Z is block-diagonal with blocks 1_{n_g} (column of n_g ones). So trace(Z·Var(u|y)·Z') = Σ_g τ_g² · trace(1_{n_g}·1_{n_g}') = Σ_g τ_g² · n_g. The n_g comes from the ZZ' structure, not the posterior variance.

**WARNING — Incomplete fix applied**: An observer fix (2026-04-06) computed Σ_g τ_g² without the n_g multiplier. The tests still pass because no test asserts σ² convergence directly. The correct fix requires `ng * tau2_g`.

**Priority**: Medium-High. Current estimates are biased; the σ²_u update is correct, so ICC will converge qualitatively but not quantitatively.

---

### Issue 3: ADF Critical Values (ADVISORY)

**Location**: `time_series.rs:264-270`  
**Problem**: Critical values are hardcoded as MacKinnon asymptotic approximations for the "constant-only" model:
- {-3.43, -2.86, -2.57} at {1%, 5%, 10%}
- These are correct for n→∞ and a constant in the regression
- Finite-sample critical values are more negative (harder to reject H₀)
- Critical values differ if you include a trend term (which this code does not)

**Mathematical consequence**: For small n (< 100), the test is slightly undersized (rejects too rarely). This is standard practice and widely accepted.

**Fix**: Implement MacKinnon (2010) response surface approximation for finite-sample critical values. Or table the commonly used critical values at n={25, 50, 100, 250, 500, ∞}.

**Priority**: Low. Acceptable for practical use; document the assumption.

---

## MATHEMATICAL GENEALOGY NOTES

### The Kendall tau family
τ_a = (C-D)/C(n,2) — no ties  
τ_b = (C-D)/√((C+D+T_x)(C+D+T_y)) — ties  
τ_c = 2(C-D)/(n²(m-1)/m) — contingency table  
Goodman-Kruskal γ = (C-D)/(C+D) — ignore ties  
Somers' d = (C-D)/(C+D+T_y) — asymmetric

Tambear implements τ_b. Should implement τ_c, γ, Somers' d for completeness.

### The Rényi entropy family
- q=0: Hartley entropy / diversity index
- q→1: Shannon entropy
- q=2: collision entropy (related to Gini impurity)
- q→∞: min-entropy (cryptographic)
- Tsallis entropy = non-additive generalization with different normalization
- All of these are special cases of the (α,β)-entropy family of Sharma & Mittal (1975)

### The KL divergence family
KL(P||Q) and KL(Q||P) are not equal but both valid  
JS(P,Q) = ½KL(P||M) + ½KL(Q||M) — symmetric, bounded  
Hellinger distance² = 1 - Σ√(pᵢqᵢ) — related to Bhattacharyya  
Total variation = ½Σ|pᵢ - qᵢ|  
Wasserstein-1 (Earth Mover's Distance) — optimal transport; not in tambear yet  
f-divergences generalize all: D_f(P||Q) = Σ qᵢ f(pᵢ/qᵢ)

### The Gaussian process family
Standard GP regression → sparse GP (Nyström/inducing points) → kernel ridge regression (same solution, different path) → SVR (structural risk minimization viewpoint). The sufficient stats for GP posterior (alpha vector + Cholesky) unifies all three. Not yet implemented in tambear.

---

## NEXT PRIORITY FOR VERIFICATION

When implementations are added, verify in this order (highest mathematical risk first):

1. **Distribution PPF/inverse CDF** — numerical inversion is numerically subtle; Brent's method needed
2. **Lasso/LARS path** — active set transitions need careful verification
3. **ARIMA** — integration order identification + KPSS/ADF must agree
4. **Kalman smoother** — backward pass sign conventions are error-prone
5. **GMM EM** — numerical covariance regularization during M-step
6. **NUTS sampler** — U-turn criterion, dual averaging step size
7. **Elliptic curve arithmetic** — field overflow in modular arithmetic
8. **LDPC belief propagation** — log-domain vs probability domain stability

---

_All citations above reference original papers. Implementations have been verified against those papers directly._
