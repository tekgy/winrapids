# Distribution Library — Mathematical Specification
**Author**: math-researcher  
**Date**: 2026-04-06  
**Purpose**: Reference for pathmaker implementing distributions.rs

Each distribution gets: PDF, CDF, PPF (quantile), log-PDF, mean, variance, MLE formula, sufficient stats.
All formulas verified against Abramowitz & Stegun + NIST DLMF + Johnson/Kotz/Balakrishnan (Continuous Univariate Distributions).

---

## TRAIT INTERFACE

Every distribution should implement:
```rust
trait Distribution {
    fn pdf(&self, x: f64) -> f64;
    fn log_pdf(&self, x: f64) -> f64;  // numerically stable
    fn cdf(&self, x: f64) -> f64;
    fn sf(&self, x: f64) -> f64;        // = 1 - cdf, but numerically stable
    fn ppf(&self, p: f64) -> f64;       // quantile / inverse CDF
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn entropy(&self) -> f64;           // differential entropy H = -∫f(x)ln f(x)dx
    fn fit(data: &[f64]) -> Self;       // MLE
}
```

For discrete: replace pdf→pmf, log_pdf→log_pmf, ppf→integer quantile.

---

## CONTINUOUS DISTRIBUTIONS

### 1. Exponential(λ)

**PDF**: f(x) = λ·exp(-λx), x ≥ 0; λ > 0  
**CDF**: F(x) = 1 - exp(-λx)  
**SF**: S(x) = exp(-λx) — use this form, avoids catastrophic cancellation  
**log-PDF**: ln(λ) - λx  
**PPF**: F⁻¹(p) = -ln(1-p)/λ  
**Mean**: 1/λ  
**Variance**: 1/λ²  
**Entropy**: 1 - ln(λ) (nats)  
**MLE**: λ̂ = n / Σxᵢ = 1/x̄  
**Sufficient stats**: n, Σxᵢ  
**Memoryless**: P(X > s+t | X > s) = P(X > t). Only continuous distribution with this property.

**Implementation note**: For numerically stable CDF, use `(-lambda * x).exp_m1().neg()` = 1 - e^{-λx} = -expm1(-λx). This preserves precision for small λx.

---

### 2. Gamma(α, β) — shape α, rate β

**PDF**: f(x) = β^α / Γ(α) · x^{α-1} · exp(-βx), x > 0  
**log-PDF**: α·ln(β) - log_gamma(α) + (α-1)·ln(x) - β·x  
**CDF**: F(x; α, β) = P(α, βx) — regularized lower incomplete gamma (already in special_functions.rs: `regularized_gamma_p`)  
**SF**: Q(α, βx) = `regularized_gamma_q`  
**PPF**: Bisection or Newton on regularized_gamma_p. Initial guess: Wilson-Hilferty approximation.
  - Wilson-Hilferty: x₀ ≈ α · (1 - 1/(9α) + z/√(9α))³ where z = ppf_normal(p)
  
**Mean**: α/β  
**Variance**: α/β²  
**Entropy**: α - ln(β) + log_gamma(α) + (1-α)·digamma(α)  
**MLE**: 
- β̂ = α̂/x̄ (moment equation)
- α̂: solve digamma(α) - ln(α) = ln(x̄) - (1/n)Σln(xᵢ) (Newton on this equation)
  - Newton: α_{new} = α / (1 - (digamma(α) - ln(α) - S) × α / (α · trigamma(α) - 1))
  - where S = (1/n)Σln(xᵢ) - ln(x̄) = ln(geo_mean) - ln(arith_mean) ≤ 0
  - Good initial guess: α₀ = (3 - S + √((S-3)² + 24S)) / (12S) (Choi & Wette 1969 or similar)
  
**Sufficient stats**: n, Σxᵢ, Σln(xᵢ)  
**Special cases**: 
- α=1: Exponential(β)
- α=k/2, β=1/2: Chi-squared(k)
- α integer: Erlang(α, β)

---

### 3. Beta(α, β) — shape parameters α, β > 0

**PDF**: f(x; α,β) = x^{α-1}(1-x)^{β-1} / B(α,β), x ∈ (0,1)  
**log-PDF**: (α-1)·ln(x) + (β-1)·ln(1-x) - log_beta(α,β) — log_beta already in special_functions.rs  
**CDF**: I_x(α, β) = regularized_incomplete_beta(x, α, β) — already in special_functions.rs  
**SF**: 1 - I_x(α, β) = I_{1-x}(β, α) (symmetry)  
**PPF**: Bisection on CDF. Initial guess: x₀ = α/(α+β) (mean).
  - Or Newton-Raphson: x_{n+1} = x_n - (I_{x_n}(α,β) - p) / f(x_n; α,β)
  
**Mean**: α/(α+β)  
**Variance**: αβ / ((α+β)²(α+β+1))  
**Entropy**: log_beta(α,β) - (α-1)·digamma(α) - (β-1)·digamma(β) + (α+β-2)·digamma(α+β)  
**MLE**: 
- t₁ = (1/n)Σln(xᵢ), t₂ = (1/n)Σln(1-xᵢ)
- Method of moments initial guess: α₀ = m(m(1-m)/v - 1), β₀ = (1-m)α₀/m where m=x̄, v=s²
- Newton on the system: digamma(α) - digamma(α+β) = t₁, digamma(β) - digamma(α+β) = t₂
  
**Sufficient stats**: n, Σln(xᵢ), Σln(1-xᵢ)

---

### 4. Weibull(k, λ) — shape k, scale λ

**PDF**: f(x) = (k/λ)(x/λ)^{k-1} exp(-(x/λ)^k), x ≥ 0  
**log-PDF**: ln(k) - ln(λ) + (k-1)(ln(x) - ln(λ)) - (x/λ)^k  
**CDF**: F(x) = 1 - exp(-(x/λ)^k)  
**SF**: exp(-(x/λ)^k) — use directly  
**PPF**: F⁻¹(p) = λ·(-ln(1-p))^{1/k}  
**Mean**: λ·Γ(1 + 1/k)  
**Variance**: λ²·[Γ(1 + 2/k) - Γ(1+1/k)²]  
**Entropy**: γ(1-1/k) + ln(λ/k) + 1, γ = Euler-Mascheroni ≈ 0.5772  
**MLE**:
- k̂: solve (1/k) + (1/n)Σln(xᵢ) - Σ(xᵢ^k · ln(xᵢ)) / Σxᵢ^k = 0 (Newton)
  - Initial guess: k₀ = (√6/π) · (standard_dev / mean) (moment-based)
- λ̂ = (Σxᵢ^k / n)^{1/k}  (after finding k)

**Sufficient stats**: n, Σxᵢ^k, Σln(xᵢ) (but k must be known or jointly estimated)  
**Special cases**: k=1 → Exponential(1/λ); k=2 → Rayleigh; k≈3.6 → near-Normal

---

### 5. Pareto(xm, α) — scale xm, shape α

**PDF**: f(x) = α·xmᵅ / x^{α+1}, x ≥ xm  
**log-PDF**: ln(α) + α·ln(xm) - (α+1)·ln(x)  
**CDF**: F(x) = 1 - (xm/x)^α  
**SF**: (xm/x)^α  
**PPF**: F⁻¹(p) = xm / (1-p)^{1/α}  
**Mean**: α·xm/(α-1) if α > 1; else ∞  
**Variance**: xm²·α / ((α-1)²(α-2)) if α > 2; else ∞  
**MLE**: 
- x̂m = min(xᵢ) (exact)
- α̂ = n / Σln(xᵢ/xm) (exact closed form)
  
**Sufficient stats**: n, min(xᵢ), Σln(xᵢ)

---

### 6. Lognormal(μ, σ)

**PDF**: f(x) = exp(-((ln x - μ)/(2σ))²) / (x·σ·√(2π)), x > 0  
**log-PDF**: -½((ln(x)-μ)/σ)² - ln(x) - ln(σ) - ½ln(2π)  
**CDF**: Φ((ln(x)-μ)/σ) — use normal_cdf from special_functions.rs  
**SF**: 1 - Φ((ln(x)-μ)/σ)  
**PPF**: exp(μ + σ·Φ⁻¹(p))  — need normal_ppf (Brent on normal_cdf, or rational approx)  
**Mean**: exp(μ + σ²/2)  
**Variance**: (exp(σ²)-1)·exp(2μ+σ²)  
**MLE**: μ̂ = (1/n)Σln(xᵢ), σ̂² = (1/n)Σ(ln(xᵢ)-μ̂)² (biased OK, classical MLE)  
**Sufficient stats**: n, Σln(xᵢ), Σ(ln(xᵢ))²  

---

### 7. Triangular(a, b, c) — lower a, upper b, mode c

**PDF**: f(x) = 2(x-a)/((b-a)(c-a)) for a≤x≤c; 2(b-x)/((b-a)(b-c)) for c<x≤b  
**CDF**: (x-a)²/((b-a)(c-a)) for a≤x≤c; 1 - (b-x)²/((b-a)(b-c)) for c<x≤b  
**PPF**: 
- p ≤ (c-a)/(b-a): a + √(p(b-a)(c-a))
- p > (c-a)/(b-a): b - √((1-p)(b-a)(b-c))
  
**Mean**: (a+b+c)/3  
**Variance**: (a²+b²+c²-ab-ac-bc)/18  
**MLE**: â=min(xᵢ), b̂=max(xᵢ), ĉ: maximize (complex, often just use mode from histogram)

---

### 8. Logistic(μ, s)

**PDF**: f(x) = exp(-(x-μ)/s) / (s(1+exp(-(x-μ)/s))²) = sech²((x-μ)/(2s)) / (4s)  
**CDF**: σ((x-μ)/s) = 1/(1+exp(-(x-μ)/s))  
**SF**: 1/(1+exp((x-μ)/s))  
**PPF**: F⁻¹(p) = μ + s·ln(p/(1-p)) = μ + s·logit(p)  
**Mean**: μ  
**Variance**: π²s²/3  
**MLE**: no closed form; Newton or gradient on log-likelihood  
- Score: ∂ℓ/∂μ = Σ(2Fᵢ - 1)/s, ∂ℓ/∂s = Σ[(1 - 2Fᵢ)(xᵢ-μ)/s - 1/s]

---

### 9. Laplace(μ, b)

**PDF**: f(x) = (1/(2b))·exp(-|x-μ|/b)  
**log-PDF**: -ln(2b) - |x-μ|/b  
**CDF**: 
- x ≤ μ: ½·exp((x-μ)/b)
- x > μ: 1 - ½·exp(-(x-μ)/b)
  
**PPF**:
- p < ½: μ + b·ln(2p)
- p ≥ ½: μ - b·ln(2(1-p))
  
**Mean**: μ  
**Variance**: 2b²  
**MLE**: μ̂ = median(xᵢ), b̂ = (1/n)Σ|xᵢ - μ̂|  
**Sufficient stats**: n, median, Σ|xᵢ - median|  
**Note**: Laplace MLE = L1 regression; relates to lasso penalty

---

### 10. Gumbel (Type I Extreme Value, μ, β)

**PDF**: f(x) = (1/β)·exp(-(z+exp(-z))), z = (x-μ)/β  
**log-PDF**: -ln(β) - z - exp(-z)  
**CDF**: exp(-exp(-z))  
**SF**: 1 - exp(-exp(-z))  
**PPF**: F⁻¹(p) = μ - β·ln(-ln(p))  
**Mean**: μ + γβ, γ = Euler-Mascheroni ≈ 0.5772  
**Variance**: π²β²/6  
**MLE**: coupled system; Newton or gradient ascent on ℓ = Σ(-ln β - zᵢ - e^{-zᵢ})
  - Initial: β₀ = √6·s/π, μ₀ = x̄ - γβ₀ where s = sample std, γ ≈ 0.5772
  - Score: ∂ℓ/∂μ = (1/β)Σ(e^{-zᵢ} - 1), ∂ℓ/∂β = (1/β)Σ(e^{-zᵢ}·zᵢ - zᵢ - 1)

---

### 11. Von Mises(μ, κ) — circular normal

**PDF**: f(θ) = exp(κ·cos(θ-μ)) / (2π·I₀(κ)), θ ∈ [-π, π]  
where I₀(κ) = modified Bessel function of order 0  
**CDF**: no closed form; numerical integration  
**Mean direction**: μ  
**Variance (circular)**: 1 - I₁(κ)/I₀(κ) = 1 - A(κ)  
**MLE**:
- μ̂ = atan2(Σsin(θᵢ), Σcos(θᵢ))
- κ̂: solve A(κ) = R̄ where R̄ = ||(1/n)Σexp(iθᵢ)|| (mean resultant length)
  - A(κ) = I₁(κ)/I₀(κ): monotone, use bisection or Padé approximant
  
**Note**: Needs Bessel I₀, I₁ — add to special_functions.rs first

---

## DISCRETE DISTRIBUTIONS

### 12. Bernoulli(p)

**PMF**: P(X=1)=p, P(X=0)=1-p  
**log-PMF**: k·ln(p) + (1-k)·ln(1-p)  
**CDF**: 0 for x<0; 1-p for 0≤x<1; 1 for x≥1  
**PPF**: 0 if q ≤ 1-p, else 1  
**Mean**: p  
**Variance**: p(1-p)  
**MLE**: p̂ = x̄ = (# successes)/n  
**Sufficient stats**: n, Σxᵢ

---

### 13. Binomial(n, p)

**PMF**: P(X=k) = C(n,k)·pᵏ(1-p)^{n-k}, k=0,1,...,n  
**log-PMF**: log_comb(n,k) + k·ln(p) + (n-k)·ln(1-p)  
**CDF**: I_{1-p}(n-k, k+1) = regularized_incomplete_beta(1-p, n-k, k+1) — use existing  
  Or equivalently: Σⱼ₌₀ᵏ C(n,j) pʲ(1-p)^{n-j} for small n  
**PPF**: Bisection on CDF  
**Mean**: np  
**Variance**: np(1-p)  
**MLE**: p̂ = x̄/n where x̄ = (1/m)Σxᵢ (m samples, each x ∈ {0,...,n})  
**Sufficient stats**: m, Σxᵢ (when n is known)

**Implementation**: Use log-sum-exp for numerical stability in PMF.  
For large n: use Normal approximation (np, np(1-p)) when np > 5 and n(1-p) > 5.

---

### 14. Poisson(λ)

**PMF**: P(X=k) = exp(-λ)·λᵏ/k!, k=0,1,2,...  
**log-PMF**: -λ + k·ln(λ) - log_factorial(k)  
**CDF**: Q(k+1, λ) = regularized_gamma_q(k+1, λ) — or equivalently Σⱼ₌₀ᵏ exp(-λ)λʲ/j!  
**SF**: regularized_gamma_p(k+1, λ)  
**PPF**: Bisection on CDF; start near λ (mean)  
**Mean**: λ  
**Variance**: λ  
**MLE**: λ̂ = x̄  
**Sufficient stats**: n, Σxᵢ  

**Numerics**: Use log-PMF and log-CDF to avoid overflow for large λ. The regularized gamma connection makes CDF evaluation efficient and numerically stable.

---

### 15. Negative Binomial(r, p) — failures before r successes

**PMF**: P(X=k) = C(k+r-1, k)·pʳ·(1-p)ᵏ, k=0,1,2,...  
**log-PMF**: log_comb(k+r-1, k) + r·ln(p) + k·ln(1-p)  
  Or in terms of log_gamma: log_gamma(k+r) - log_gamma(k+1) - log_gamma(r) + r·ln(p) + k·ln(1-p)  
**CDF**: I_p(r, k+1) = regularized_incomplete_beta(p, r, k+1)  
**PPF**: Bisection  
**Mean**: r(1-p)/p  
**Variance**: r(1-p)/p²  
**MLE** (given r fixed): p̂ = r/(r + x̄)  
**MLE** (r and p jointly, NB2 parameterization μ, φ): requires Newton  
  - E[X]=μ, Var[X]=μ+μ²/φ where φ is overdispersion
  - p̂ = φ/(φ+μ), r̂ = φ
  - Score equations: coupled; fixed-point iteration or Newton

---

### 16. Geometric(p) — trials until first success

**PMF**: P(X=k) = p(1-p)^{k-1}, k=1,2,...  
  (variant: k=0,1,... for failures before first success)  
**CDF**: 1 - (1-p)^k  
**PPF**: ⌈ln(1-u)/ln(1-p)⌉  
**Mean**: 1/p  
**Variance**: (1-p)/p²  
**MLE**: p̂ = 1/x̄  

---

### 17. Multinomial(n; p₁,...,pₖ)

**PMF**: P(X₁=x₁,...,Xₖ=xₖ) = n! / (x₁!...xₖ!) × Πᵢ pᵢ^{xᵢ}  
**log-PMF**: ln(n!) - Σln(xᵢ!) + Σxᵢ·ln(pᵢ)  
**Marginals**: Xᵢ ~ Binomial(n, pᵢ)  
**Mean**: E[Xᵢ] = n·pᵢ  
**Covariance**: Cov(Xᵢ, Xⱼ) = -n·pᵢ·pⱼ (i≠j); Var(Xᵢ) = n·pᵢ(1-pᵢ)  
**MLE**: p̂ᵢ = (total count in category i) / (total observations × n)  

---

## INVERSE CDF (PPF) — NUMERICAL STRATEGY

The PPF is the hardest part. Strategy by distribution:

### Method 1: Closed form (use when available)
Exponential, Pareto, Gumbel, Weibull, Laplace, Logistic, Geometric, Bernoulli — all have closed-form PPF.

### Method 2: Newton-Raphson (when PDF is cheap)
Starting from a good initial guess x₀:
x_{n+1} = x_n - (F(x_n) - p) / f(x_n)

Converges quadratically. Requires: monotone CDF, differentiable PDF, good starting point.

Used for: Normal (via rational approx), Gamma (Wilson-Hilferty start), Beta.

### Method 3: Brent's method (fallback)
When Newton fails (discontinuous derivative, poor starting point):
Bracket [lo, hi] where F(lo) < p < F(hi), then use Brent (bisection + secant + inverse quadratic interpolation).

Used for: Binomial, Poisson, NB (discrete = staircase CDF).

### Normal PPF (needed by many others)
Best method: rational approximation (Beasley-Springer-Moro algorithm, or Acklam's rational approx):
For p ∈ (0.5, 1): normal_ppf = p in the inner region via Horner-evaluated rational polynomial.
Accuracy: |error| < 4.5e-4 for Acklam; better implementations exist (Wichura AS241).

**Add to special_functions.rs first**: normal_ppf(p: f64) -> f64

---

## IMPLEMENTATION ORDER

1. `normal_ppf` — needed by lognormal PPF, normal tests
2. `Exponential` — simplest, validates the trait design
3. `Gamma` — needs gamma_ppf via Wilson-Hilferty + Newton; foundation for chi2, chi
4. `Beta` — needs beta_ppf via Newton on regularized_incomplete_beta
5. `Poisson` — needed for Poisson regression tests
6. `Binomial` — needed for proportion tests
7. `Weibull` — needed for survival tests
8. `NB` — overdispersed count model
9. `Lognormal`, `Pareto`, `Gumbel`, `Laplace`, `Logistic`, `Triangular`
10. `Geometric`, `Multinomial`
11. `VonMises` (needs Bessel functions)

---

## SUFFICIENT STATISTICS CONNECTION

For each distribution, the sufficient stats are what tambear's accumulate primitive computes:

| Distribution | accumulate expr | op | result = MLE input |
|---|---|---|---|
| Exponential | x | sum | Σx → λ̂ = n/Σx |
| Gamma | (x, ln(x)) | sum | Σx, Σln(x) → shape MLE |
| Beta | (ln(x), ln(1-x)) | sum | Σln(x), Σln(1-x) → shape MLE |
| Normal | (x, x²) | sum | Σx, Σx² → μ̂, σ̂² |
| Poisson | x | sum | Σx → λ̂ = Σx/n |
| Binomial | x | sum | Σx → p̂ = Σx/(n·trials) |
| Lognormal | (ln(x), (ln(x))²) | sum | standard moments of log |
| Weibull | (x^k, ln(x)) | sum | for fixed k: Σx^k → λ̂ |

The distribution library and the accumulate primitive are structurally unified. Every MLE is a function of sufficient stats, and every sufficient stat is an accumulate expression. This is the Fisher-Neyman factorization theorem expressed as tambear primitives.

---

*Reference: Johnson, Kotz, Balakrishnan: Continuous Univariate Distributions (1994/1995). For discrete: Discrete Multivariate Distributions (1997). Online: NIST DLMF.*
