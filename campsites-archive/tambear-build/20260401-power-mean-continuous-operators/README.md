# Campsite: Power Mean as Universal Accumulate Operator

**Opened:** 2026-04-01  
**Scout:** scout (from naturalist's Rényi finding + garden exploration)  
**Thread:** "The continuous operator space (naturalist's Rényi finding) — power mean M_α as universal operator family"  
**Status:** Conceptual complete — needs architect review for API implications

---

## The Core Claim

Tambear's discrete accumulate operators (Add, Max, Min, GeoMean, HarmonicMean) are all **specific samples of the continuous power mean family M_α**.

```
M_α(x₁,...,xₙ) = ((Σ xᵢ^α) / n)^(1/α)

α → -∞ :   min              ← Min accumulate
α = -1 :   harmonic mean    ← HarmonicMean (descriptive.rs:710)
α → 0  :   geometric mean   ← GeoMean (descriptive.rs:693)
α = 1  :   arithmetic mean  ← Add then divide
α = 2  :   quadratic mean   ← sum of squares, then root
α → +∞ :   max              ← Max accumulate
```

Every operator in this list is already in the codebase. They're scattered across `descriptive.rs`, `accumulate.rs`, and special cases. They're not recognized as a family.

---

## The Accumulate Decomposition

Every M_α accumulate has the form:

```
accumulate(data, grouping, phi = x^α, op = Add) → sum_alpha
result = (sum_alpha / n)^(1/α)
```

The only thing that varies is α. The accumulate expression `phi = x^α` is the same for the entire family. No special operator is needed for any α.

**Current codebase state**: each mean type is implemented separately with its own function. Unification would be:

```rust
pub fn power_mean(data: &[f64], alpha: f64) -> f64 {
    // Special cases at the singular points
    if alpha == f64::NEG_INFINITY { return data.iter().cloned().fold(f64::INFINITY, f64::min); }
    if alpha == f64::INFINITY     { return data.iter().cloned().fold(f64::NEG_INFINITY, f64::max); }
    if alpha.abs() < 1e-12 {
        // Geometric mean via log-mean
        let n = data.len() as f64;
        return data.iter().map(|&x| x.ln()).sum::<f64>().exp() / n.exp();
    }
    // General case
    let n = data.len() as f64;
    let sum_alpha: f64 = data.iter().map(|&x| x.powf(alpha)).sum();
    (sum_alpha / n).powf(1.0 / alpha)
}
```

---

## The Rényi Entropy Connection

The Rényi entropy of order α is:
```
H_α(p) = (1/(1-α)) × log(Σ pᵢ^α)
```

This is `accumulate(p^α, All, Add)` followed by log-rescaling. In other words:

**Rényi entropy of order α = log-transform of M_α^α(p) × n**

The entropy order IS the power mean parameter. Every Rényi entropy corresponds to exactly one M_α accumulate. The information-theoretic family and the mean family are the same family, viewed through different transforms.

### The three singular points

At α ∈ {0, 1, ∞}, the power mean / accumulate expression changes character:

| α | Accumulate phi | What it computes |
|---|---------------|-----------------|
| 0 | count(x > 0) | Hartley entropy H₀ = log(|support|) |
| 1 | x·log(x) | Shannon entropy (derivative of x^α at α=1) |
| ∞ | max(x) | Min-entropy H_∞ = -log(max p) |
| else | x^α | Power sum → Rényi H_α |

Shannon is the tangent to the Rényi family at α=1, not a power mean. It's structurally different: the accumulate expression is `x·log(x)`, the derivative of `x^α` with respect to α evaluated at α=1.

The three singular points are the three historically "important" entropies. Their importance IS their structural singularity.

---

## Tsallis: Linearization of Rényi

Tsallis entropy S_q uses the same `x^α` accumulate but takes `1 - sum` instead of `log(sum)`:
```
S_q(p) = (1/(q-1)) × (1 - Σ pᵢ^q)
```

Near q=1: Rényi ≈ Tsallis ≈ Shannon (all three agree to first order). The choice between them is a choice of geometry near the Shannon point:
- **Rényi**: log geometry (additive for independent systems)
- **Tsallis**: linear geometry (non-additive, for correlated/complex systems)
- **Shannon**: the limit, where geometry is tangent to both

The accumulate is identical (x^q). The post-processing chooses the geometry. This is a post-operator decision, not a phi decision.

---

## Architecture Implication: `.accumulate(phi=|x, α| x.powf(α))`

Currently, adding a new mean type requires a new function in `descriptive.rs`. With the M_α framing, you add a single parameter:

```rust
// Current (5 separate functions):
geometric_mean(data)
harmonic_mean(data)
arithmetic_mean(data)   // implicit
rms(data)
max(data)

// Proposed (one function):
power_mean(data, alpha)    // alpha = 0, -1, 1, 2, ∞ recover the above
```

This is a pure unification — no new capability, same code paths. The benefit is that `alpha` becomes a tunable parameter in optimization contexts. When you want "a mean that's robust to outliers," you can tune α rather than choosing between named alternatives.

In the `.tbs` scripting language:
```
data.accumulate(expr=x^α, op=Sum)  // parameterized by α
```

---

## The Softmax Connection

Neural network softmax is:
```
softmax_i(x/T) = exp(xᵢ/T) / Σ_j exp(xⱼ/T)
```

As T→0: approaches hard argmax (Max operator)
As T→∞: approaches uniform distribution

This is the log-domain power mean with α=1/T in the exponent, via the identity:
```
log M_α(exp(x)) = (1/α) log Σ exp(α·xᵢ) - log n
                = LogSumExp(α·x) / α - log n
```

Softmax is M_α in log-domain, with α=1/T. Temperature annealing is moving along the power mean axis from uniform (α→0) toward argmax (α→∞).

This means the attention mechanism in transformers (softmax over attention logits) is literally a power mean accumulate on the logit space, temperature-parameterized. The "temperature" hyperparameter is α.

---

## Open Questions

1. **Should tambear expose M_α directly in the .tbs API?** The parameterization is natural and the implementation is trivial. But does it add complexity to the scripting surface without adding understanding?

2. **Weighted power mean**: M_α with weights wᵢ: `(Σ wᵢ xᵢ^α)^(1/α)`. This is what attention actually computes (weighted softmax → weighted arithmetic mean in value space). Worth exposing?

3. **Power mean over groups**: `power_mean(grouping=unit, alpha=α)` — parameterized group mean. This is what the RE panel estimator's quasi-demeaning needs when α≠1.

4. **Anomalous diffusion**: Replace M_1 (arithmetic mean of neighbors, α=1) with M_α (power mean of neighbors, α≠1) in the diffusion operator. Produces fractional diffusion, superdiffusion/subdiffusion depending on α. Relevant to market microstructure modeling.

---

## Evidence Already in Codebase

- `descriptive.rs:693` — `geometric_mean` (M_0)
- `descriptive.rs:710` — `harmonic_mean` (M_{-1})
- `accumulate.rs` — `Add`, `Max`, `Min` operators (α=1, +∞, -∞)
- `information_theory.rs:84` — `renyi_entropy` (Σ p^α, parameterized)
- `information_theory.rs:111` — `tsallis_entropy` (same accumulate, different post-process)
- `nonparametric.rs:560` — `silverman_bandwidth` uses variance (M_2)
- Every test that computes RMS implicitly uses M_2

The family is fully present. The unification is recognition, not construction.

---

## Related Campsites
- [series-acceleration-prefix-scan](../20260401-series-acceleration-prefix-scan/) — "no new primitive, it's composition"
- [scatter-attract-duality](../20260401-scatter-attract-duality/) — the `attract()` primitive, Kingdom C
- Garden: `20260401-renyi-power-mean-and-the-continuous-operator.md`
- Garden: `20260401-navier-stokes-and-the-self-referential-scan.md`
