# Notebook 011 — The Conditional Moment Hierarchy

*Author: Scout, tambear-math-2 expedition*
*Date: 2026-04-01*
*Status: Research finding — proposed for Paper 5, Section 3.5 (Rhyme 34) and Section 4.3 (new prediction rows)*

---

## Discovery Path

While pre-loading gold standards for F18 (Volatility & Financial TS), I recognized that GARCH(1,1)'s variance recursion is structurally identical to Adam's v-accumulator. Following this thread revealed a complete family of algorithms — the **conditional moment hierarchy** — that bridges financial econometrics and deep learning through the accumulate framework.

---

## The Core Observation

For any data sequence {x_t}, define the k-th order EWM accumulator:

```
M_k(t) = β · M_k(t-1) + (1-β) · x^k_t
```

This is a Kingdom B Affine scan:
```
accumulate(Prefix(forward), (1-β) · x^k_t, Affine(β, ω))
```

with optional intercept ω ≥ 0.

**Every algorithm in the conditional moment hierarchy is this same accumulate call, parameterized by k.**

| k | Name | Field | Year | x | ω |
|---|------|-------|------|---|---|
| 1 | SES / EWM mean | Statistics | ~1957 | returns / gradients | 0 |
| 1 | Momentum SGD | Deep Learning | ~1964 | gradients | 0 |
| 1 | Adam m-channel | Deep Learning | 2014 | gradients | 0 |
| 2 | IGARCH(1,1) | Econometrics | 1990 | lagged returns | 0 |
| 2 | GARCH(1,1) | Econometrics | 1986 | lagged returns | ω > 0 |
| 2 | RiskMetrics EWMA | Finance | 1994 | returns | 0 |
| 2 | RMSProp | Deep Learning | 2012 | gradients | 0 |
| 2 | Adam v-channel | Deep Learning | 2014 | gradients | 0 |
| 2 | AdamW v-channel | Deep Learning | 2019 | gradients | δ·v |
| 3 | Cond. skewness | Econometrics | 1999 | lagged returns | varies |
| 4 | Cond. cokurtosis | Econometrics | 2002 | lagged returns | varies |
| 3,4 | **Predicted** | Deep Learning | — | gradients | — |

---

## The GARCH(1,1) = AdamW Structural Proof

**GARCH(1,1):**
```
σ²_t = ω + β · σ²_{t-1} + α · r²_{t-1}
```
Unconditional variance: `σ̄² = ω / (1 - α - β)`

**AdamW v-channel** (after weight decay):
```
v_t = (β₂ - λ_decay) · v_{t-1} + (1 - β₂) · g²_t
```
Unconditional v: `v̄ = (1-β₂) · E[g²] / (1 - β₂ + λ_decay)`

**Mapping:**
- `α` ↔ `(1-β₂)` (weight on new observation)
- `β` ↔ `β₂ - λ_decay` (decay of running state)
- `ω` ↔ `λ_decay · v̄` (floor provided by weight decay)

**GARCH without intercept (IGARCH, ω=0) = Adam without weight decay.**
**GARCH with intercept (ω > 0) = AdamW with weight decay.**

The intercept ω in GARCH and the weight decay λ_decay in AdamW solve **the same problem**: preventing the variance accumulator from deflating toward zero at long time horizons when the signal is intermittently small.

---

## The Finance Advantage (k=3,4)

Harvey & Siddique (1999) introduced conditional coskewness:
```
S_t = β_s · S_{t-1} + (1-β_s) · r³_t
```
Used to price the "skewness premium" — stocks with negative conditional skewness command higher expected returns.

Dittmar (2002) extended to conditional cokurtosis (k=4).

In finance, the motivation was: if returns have time-varying higher moments, the pricing kernel depends on all four moments. Each additional moment adds one accumulate channel.

In deep learning: Adam tracks k=1 (m) and k=2 (v). **Nobody has added k=3 or k=4.** The framework predicts they should exist, and suggests what they would do:

- **m3** (gradient conditional skewness): Identifies parameters where the gradient distribution is skewed — the loss landscape tilts more in one direction than the other. Should modulate step size asymmetrically.
- **m4** (gradient conditional kurtosis): Identifies parameters with heavy-tailed gradient distributions — occasional very large gradients. Should modulate gradient clipping adaptively.

---

## The Deep Learning Gap (Why k=3,4 Haven't Been Built)

Finance motivation for higher moments: pricing kernel theory requires them (CAPM, co-moments, SDF). The field had a theoretical reason to build k=3,4.

Deep learning motivation: empirical — Adam works well. There's no theoretical framework in deep learning that demands higher moments. The framework provides that theory: the product space says k=3,4 accumulators are the same Affine scan as k=1,2, just one order higher. They SHOULD exist.

---

## Concrete Prediction (Testable)

**Prediction:** A "skewness-corrected Adam" (ScAdam) with k=3 accumulator:
```rust
m3[i] = beta3 * m3[i] + (1.0 - beta3) * g[i].powi(3);  // k=3
m3_hat = m3[i] / (1.0 - beta3.powi(t));                  // bias correct
skew[i] = m3_hat / (v_hat[i].sqrt().powi(3));             // standardize
// Use skew[i] to tilt step size asymmetrically:
step[i] = lr / (v_hat[i].sqrt() + eps) * (1.0 + gamma * skew[i].tanh());
```

would improve convergence on heavy-tailed loss surfaces compared to Adam. This is the finance analog: negative skewness → requires risk premium → higher return demanded → optimizer needs larger step in the downhill direction.

**This prediction is falsifiable:** Run on heavy-tailed losses (BERT fine-tuning on tail classes, etc.) and compare convergence.

---

## The k-Hierarchy as a Taxonomic Tool

The moment hierarchy provides a new dimension for classifying algorithms beyond the T×K×O product space. Every algorithm that tracks higher moments is:

```
M_k(t) = β M_k(t-1) + (1-β) x^k_t     [O = Affine(β), K = B, T = power-k]
```

The full classification: T = power-k (the transform is x→x^k), K = B (sequential), O = Affine(β, ω).

**Cross-domain structural rhyme table:**

| Algorithm family | T | K | O | k | Field |
|-----------------|---|---|---|---|-------|
| EWM mean | x¹ | B | Affine | 1 | Statistics/DL |
| GARCH/IGARCH | x² | B | Affine(β, ω≥0) | 2 | Finance/DL |
| Harvey-Siddique | x³ | B | Affine | 3 | Finance |
| Dittmar cokurtosis | x⁴ | B | Affine | 4 | Finance |
| **Predicted** | x³ | B | Affine | 3 | Deep Learning |
| **Predicted** | x⁴ | B | Affine | 4 | Deep Learning |

---

## Connection to Other Paper 5 Rhymes

Rhyme 15 (Adam = 4 EWM channels) already notes that Adam's v = EWM on g². This notebook extends it:

1. **Rhyme 15 (existing):** Adam's m = EWM on g (k=1). Adam's v = EWM on g² (k=2).
2. **Rhyme 34 (new):** Adam's v = GARCH without intercept. AdamW's v = GARCH with intercept.
3. **Prediction (new):** High-order Adam (k=3,4) = Harvey-Siddique / Dittmar in optimization context.

The three together form a complete story: the conditional moment hierarchy unifies time series econometrics and gradient-based optimization through the accumulate framework.

---

## Code Evidence in Tambear

- `signal_processing.rs`: Autocorrelation/cross-correlation = lag-product accumulates
- `time_series.rs`: SES/Holt = k=1 Affine scans
- `volatility.rs`: GARCH(1,1) = k=2 Affine scan, EWMA = k=2 with ω=0
- `optimization.rs`: Adam m/v channels = k=1,2 Affine scans (presumably)
- `complexity.rs`: DFA, Hurst = higher-order temporal structure (related but distinct)

The k=3,4 accumulators are NOT yet in tambear. They would be trivial to add to `optimization.rs` — 2 lines per accumulator. The behavioral prediction is the more important missing piece.

---

## Status

This is a research proposal, not a settled result. The formal components:

1. **Rhyme 34 (GARCH = Adam-v)** — clean, verifiable, ready for Paper 5 Section 3.5.
2. **Prediction rows for Section 4.3** — two new rows: (Square, B, Affine) and (x^k, B, Affine) for k-hierarchy.
3. **ScAdam prediction** — speculative but falsifiable, appropriate for Paper 5 Section 5 (Predictions).

The ScAdam prediction would be the most impactful claim: if it works, it's a new optimizer derived from the financial moment hierarchy. If it doesn't, it's a falsified prediction that still validates the framework's coherence.
