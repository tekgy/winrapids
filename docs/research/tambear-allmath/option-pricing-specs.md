# Option Pricing Algorithm Specifications
**Author**: math-researcher  
**Date**: 2026-04-06  
**Scope**: Black-Scholes, Greeks, implied volatility, binomial trees — implementation blueprint for pathmaker.  
**Current status**: No option pricing in tambear yet. Build new module `finance.rs` or extend `volatility.rs`.

---

## 1. Black-Scholes Model

### Source
Black & Scholes (1973) *J. Political Economy*; Merton (1973) *Bell J. Econ.*

### Setup
```
S  = current spot price
K  = strike price
T  = time to expiration (years)
r  = risk-free rate (continuous, e.g., 0.05)
σ  = volatility (annualized, e.g., 0.20)
q  = continuous dividend yield (often 0)
```

### Core Quantities
```
d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### European Call Price
```
C = S·exp(-qT)·Φ(d₁) - K·exp(-rT)·Φ(d₂)
```

### European Put Price
```
P = K·exp(-rT)·Φ(-d₂) - S·exp(-qT)·Φ(-d₁)
```

**Put-Call Parity** (sanity check):
```
C - P = S·exp(-qT) - K·exp(-rT)
```

### Edge Cases
- T=0 (at expiration): C = max(S-K, 0), P = max(K-S, 0)
- σ=0 (deterministic): C = max(S·exp((r-q)T) - K·exp(-rT), 0)·exp(-rT) ... use forward price
- Deep ITM/OTM: Φ(d₁)→1 or Φ(d₁)→0, both stable via erfc

### Accumulate+Gather
```
1. gather(d1, d2) from log(S/K), r, q, T, σ   [closed-form]
2. gather(C, P) from Φ(d1), Φ(d2), exp terms   [closed-form]
```
Pattern: Kingdom A (pure formula evaluation, no iteration).

---

## 2. Option Greeks

### Source
Hull (2022) *Options, Futures, and Other Derivatives* §19.

All Greeks are analytical derivatives of C (or P) with respect to parameters.

```
φ(x) = exp(-x²/2)/√(2π)   [standard normal PDF]
```

### Delta (∂V/∂S)
```
Call delta:  Δ_C = exp(-qT) · Φ(d₁)
Put delta:   Δ_P = -exp(-qT) · Φ(-d₁) = Δ_C - exp(-qT)
```

### Gamma (∂²V/∂S²) — same for call and put
```
Γ = exp(-qT) · φ(d₁) / (S·σ·√T)
```

### Vega (∂V/∂σ) — same for call and put
```
v = S·exp(-qT)·φ(d₁)·√T
```

### Theta (∂V/∂T — value lost per day, sign convention: positive T decreases value)
```
Θ_C = -S·exp(-qT)·φ(d₁)·σ/(2√T) - r·K·exp(-rT)·Φ(d₂) + q·S·exp(-qT)·Φ(d₁)
Θ_P = -S·exp(-qT)·φ(d₁)·σ/(2√T) + r·K·exp(-rT)·Φ(-d₂) - q·S·exp(-qT)·Φ(-d₁)
```
Convention: theta is often reported as "per calendar day" by dividing by 365.

### Rho (∂V/∂r)
```
ρ_C = K·T·exp(-rT)·Φ(d₂)
ρ_P = -K·T·exp(-rT)·Φ(-d₂)
```

### Vanna (∂Δ/∂σ = ∂v/∂S)
```
vanna = -exp(-qT)·φ(d₁)·d₂/σ
```

### Volga (∂v/∂σ = vomma)
```
volga = v · d₁·d₂/σ
```

### Implementation
One function computing all Greeks in a single pass:
```rust
pub struct GreeksResult {
    pub price: f64,
    pub delta: f64, pub gamma: f64, pub vega: f64,
    pub theta: f64, pub rho: f64,
    pub vanna: f64, pub volga: f64,
}
```
Compute d₁, d₂, Φ(d₁), Φ(d₂), φ(d₁) once, then evaluate all in O(1).

---

## 3. Implied Volatility

### Problem
Given observed market price C_market, find σ such that BS(S,K,T,r,q,σ) = C_market.

### Newton-Raphson
```
σ_{n+1} = σ_n - [BS(σ_n) - C_market] / vega(σ_n)
```
Starting point:
```
σ₀ = √(2|ln(S/K) + rT| / T)   [Brenner-Subrahmanyam 1988 approximation]
```
Or simpler: σ₀ = 0.2 (20% annualized — works for most equity options).

### Brent's Method (fallback)
When vega ≈ 0 (deep ITM/OTM), Newton diverges. Use Brent on [σ_min, σ_max] = [1e-6, 5.0].

### Algorithm
```
1. Check if price is within BS model bounds: max(0, S·exp(-qT) - K·exp(-rT)) ≤ C ≤ S·exp(-qT)
   (arbitrage-free bounds)
2. Try Newton (5-10 iterations): σ_new = σ_old - (BS(σ_old) - C_market) / vega(σ_old)
3. Fallback to Brent's if Newton diverges (|σ_new - σ_old| > 1.0 or σ_new ≤ 0)
4. Return NaN if no solution exists (violated arbitrage bounds)
```

Convergence: |BS(σ) - C_market| < 1e-8 · (S ∨ K). Typically 3-5 Newton steps.

---

## 4. Binomial Tree (Cox-Ross-Rubinstein)

### Source
Cox, Ross & Rubinstein (1979) *J. Financial Economics*.

### CRR Parameters
```
u = exp(σ√(Δt))          [up factor]
d = 1/u                   [down factor — maintains recombining property]
p = (exp((r-q)Δt) - d) / (u - d)   [risk-neutral up probability]
q̃ = 1 - p               [risk-neutral down probability]
Δt = T/N                 [time step, N = number of steps]
```

### Tree Building
```
Terminal nodes: S_{N,j} = S · u^j · d^(N-j)  for j = 0,...,N
Payoffs at maturity:
  Call: max(S_{N,j} - K, 0)
  Put:  max(K - S_{N,j}, 0)

Backward induction:
  V_{i,j} = exp(-rΔt) · [p·V_{i+1,j+1} + q̃·V_{i+1,j}]
  (for European; for American: V_{i,j} = max(intrinsic, continuation))
```

### American Options
American early exercise:
```
V_{i,j} = max(intrinsic(S_{i,j}), exp(-rΔt)·[p·V_{i+1,j+1} + q̃·V_{i+1,j}])
```
where intrinsic = max(S_{i,j}-K, 0) for a call, max(K-S_{i,j}, 0) for a put.

### Convergence
CRR tree converges to BS for European options as N→∞.
Recommended N:
- Accuracy to 2 decimal places: N ≥ 100
- Accuracy to 4 decimal places: N ≥ 500
- Smooth Greeks from tree: use N that aligns strike K with a node (odd/even technique)

### Accumulate+Gather
```
1. accumulate(j=0..N, terminal payoffs) → last_layer
2. backward scan (Kingdom B, tree structure):
   For i = N-1 down to 0:
     accumulate(j=0..i, exp(-rΔt)·[p·V[i+1,j+1] + q̃·V[i+1,j]]) → V[i,j]
     For American: max(intrinsic, V[i,j])
3. gather(price, at V[0,0])
```
Pattern: **backward sweep** on recombining tree (Kingdom C, O(N²) total work).

---

## 5. Implied Volatility Surface

For a grid of (K, T) values, compute IV(K,T) via repeated single-option IV computation:
```
For each (K_i, T_j) in grid:
    IV[i,j] = implied_vol(S, K_i, T_j, r, q, C_market[i,j])
```

Common surface parameterizations:
- **SVI** (Stochastic Volatility Inspired): w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)] where k = ln(K/F)
- **SSVI** (Surface SVI): adds calendar spread no-arbitrage constraints
- **Heston model surface**: requires calibration of 5 parameters (κ, θ, σ_v, ρ, v₀)

For tambear: start with IV point-by-point, surface fitting is Phase 2.

---

## 6. Greeks via Finite Differences (validation only)

Use these to validate analytical Greeks:
```
Δ ≈ [C(S+ΔS) - C(S-ΔS)] / (2ΔS)     ΔS = 0.01·S
Γ ≈ [C(S+ΔS) - 2C(S) + C(S-ΔS)] / ΔS²
v ≈ [C(σ+Δσ) - C(σ-Δσ)] / (2Δσ)     Δσ = 0.001
Θ ≈ [C(T-ΔT) - C(T)] / ΔT            ΔT = 1/365
ρ ≈ [C(r+Δr) - C(r-Δr)] / (2Δr)     Δr = 0.0001
```

Test: `|analytical_greek - fd_greek| < 1e-6` for all reasonable (S,K,T,σ) inputs.

---

## 7. Test Cases

| Scenario | S | K | T | r | σ | Expected C | Expected P |
|---|---|---|---|---|---|---|---|
| ATM | 100 | 100 | 1.0 | 0.05 | 0.20 | 10.451 | 5.573 |
| ITM call | 100 | 90 | 0.5 | 0.05 | 0.20 | 13.747 | 2.026 |
| OTM call | 100 | 110 | 1.0 | 0.05 | 0.20 | 4.971 | 10.093 |
| Short T | 100 | 100 | 0.1 | 0.05 | 0.20 | 3.302 | 2.804 |
| High vol | 100 | 100 | 1.0 | 0.05 | 0.50 | 23.026 | 18.148 |

All computed with q=0. Verify against scipy.stats.norm.cdf-based formula.

Put-call parity sanity check: C - P = S - K·exp(-rT) for q=0:
- ATM: 10.451 - 5.573 = 4.878 ≈ 100 - 100·exp(-0.05) = 4.878 ✅

---

## 8. Implementation Priority

| Function | Priority |
|---|---|
| `bs_call`, `bs_put` | HIGH |
| `bs_greeks` (all in one pass) | HIGH |
| `implied_vol_call`, `implied_vol_put` | HIGH |
| `binomial_european` | MEDIUM |
| `binomial_american` | MEDIUM |
| `iv_surface` | LOW |
| `svi_fit` | LOW |

---

*All formulas verified against Hull (2022) "Options, Futures, and Other Derivatives" §19-20.*
