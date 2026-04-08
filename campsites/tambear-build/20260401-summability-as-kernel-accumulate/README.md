# Campsite: Summability Methods as Kernel-Weighted Accumulate

**Opened:** 2026-04-01  
**Thread:** Observer's insight from series_accel.rs — binomial weights = binomial filter  
**Status:** Active exploration

---

## The Core Observation

Every classical summability method is a **kernel-weighted accumulate on the sequence of partial sums**.

```
S_n = Σ_{k=0}^n a_k          (partial sums = prefix accumulate)
L(f) = Σ_n K_n · S_n          (summability = weighted accumulate on sums)
```

| Method | Kernel K_n | Algebra | Kingdom |
|--------|-----------|---------|---------|
| Cesàro (C,1) | 1/(n+1) (uniform) | mean of partial sums | A |
| Euler (E,q) | C(n,k) q^k (1-q)^{n-k} | binomial-weighted accumulate | A |
| Abel | x^n (exponential) | power series evaluation at x→1⁻ | A |
| Richardson | polynomial extrapolation weights | cancels leading error terms | A |
| Borel | e^{-x} x^n / n! (Poisson) | Poisson-weighted accumulate | A |
| Wynn ε | non-commutative recurrence | order-dependent tableau | **BC** |

The first five are all Kingdom A — weighted accumulates, fully commutable. Wynn's ε is the exception that proves the rule: it's the only one where the recurrence is non-commutative.

---

## The Structural Rhyme

**Summability methods (series) ↔ Kernel Density Estimation (data)**

```
KDE:   accumulate(data_points,  ByValue,  WeightedBy(K_bandwidth), Add)
Euler: accumulate(partial_sums, ByIndex,  WeightedBy(Binomial),    Add)
```

Same algebraic structure: weighted accumulate where the kernel is the parameter. Different domains:
- KDE: continuous distribution estimation (Gaussian, Epanechnikov, uniform kernels)
- Summability: limit computation (binomial, exponential, uniform kernels)

**The kernel IS the grouping.** In tambear's framework: `ByKey(kernel_weight)`.

---

## Tauberian Conditions as Bandwidth Selection

In KDE, the bandwidth h determines:
- Too small: undersmoothing (noisy estimate, every point its own spike)
- Too large: oversmoothing (blurred estimate, loses structure)
- Just right: balances bias and variance

In summability, the Tauberian conditions determine:
- When does summability imply actual convergence?
- What class of series can the kernel handle?
- The "Tauberian obstruction" = what KDE practitioners call "over-smoothing"

Cesàro summability handles bounded-oscillation sequences. Euler handles geometric growth. Abel handles power-series convergence. Each kernel "sees further" into the divergent end, at the cost of stronger regularity conditions (Tauberian hypotheses).

This is not metaphor — it's the same algebraic structure. The kernel parameterizes the domain of applicability. Choosing a kernel IS choosing a bandwidth.

---

## The Cesàro Case Is Striking

Cesàro (C,1): `L = (S_1 + S_2 + ... + S_n) / n`

This is EXACTLY `Welford_mean(partial_sums)`. The Cesàro limit of a series is the arithmetic mean of its partial sums. Kingdom A applied twice:
1. `accumulate(terms, Prefix, Value, Add)` → partial sums
2. `accumulate(partial_sums, Global, Value, WelfordMean)` → Cesàro limit

Cesàro summability = chained accumulate. The tambear primitive that computes it is:
```
data.accumulate(Prefix, Add).accumulate(Global, WelfordMean)
```

Higher-order Cesàro (C,k) = iterate k times:
```
data.accumulate(Prefix, Add)^k.accumulate(Global, WelfordMean)
```

This is `attract(accumulate(Prefix), k_times)` — Kingdom C in the outer structure, but deterministic and finite (exactly k iterations, not convergence-based).

---

## The Euler-Gaussian Connection

The Euler transform `E_m(S) = (1/2^m) Σ_{k=0}^m C(m,k) S_k` converges to Gaussian smoothing as m→∞:

```
Binomial(m, 1/2)  →  Normal(m/2, m/4)  as m → ∞  (CLT)
```

So Euler summability IS Gaussian-kernel averaging of partial sums in the limit of large m. The Euler transform is a finite-depth Gaussian smoother on the series.

This connects three things:
- Euler summation (classical analysis)
- Binomial filter (signal processing)  
- Gaussian KDE (nonparametric statistics)

All three are the SAME weighted accumulate. The kernel is Binomial in the first two, Gaussian in the third — and they converge to each other.

---

## T×K×O Encoding

For the structural rhyme table:

| Algorithm | T | K | O | Field |
|-----------|---|---|---|-------|
| Cesàro summability | Identity | A | Mean | Classical analysis |
| Euler summability | Identity | A | BinomialWeightedSum | Classical analysis |
| Kernel Density Estimation | Identity | A | KernelWeightedSum | Nonparametric statistics |
| Moving average (time series) | Identity | A | UniformWeightedSum | Time series |
| Exponential smoothing | Identity | A | ExponentialWeightedSum | Time series |

All share K = A. The O parameter is the kernel choice. This is a **Type IV rhyme** (same K, different O but structurally related O's that form a parameterized family).

The family is: `{methods parameterized by a kernel choice, applied to a sequence via weighted sum}`. The kernel choice is the one free parameter. Swapping domains (series ↔ data, analysis ↔ statistics) is the rhyme.

---

## Open Questions

1. **Is there a "Tauberian condition" in KDE?** (resolved). Yes — it's the KDE consistency
   theorem: KDE → true density iff (K is a Mercer kernel) AND (h→0) AND (nh→∞). This maps
   exactly to Tauberian conditions: (kernel regularity) AND (bandwidth shrinks) AND (sample
   grows). The Tauberian obstruction (Abel converges but series diverges) = the KDE estimator
   converges but to the wrong density = non-stationarity / distribution shift. Same theorem,
   different domain. See garden: `20260401-summability-as-filters.md`

2. **Abel summation and exponential smoothing** (resolved). Abel IS exponential smoothing —
   same geometric kernel, direction reversed. Abel discounts from index 0 forward; EWS discounts
   from index T backward. For finite sequences: `Abel(x) = x^{-N}/α × EWS_N` with x=1-α.
   The general principle: every summability method has a streaming Kingdom B equivalent:
   Cesàro = running Welford mean; Abel = EWM (existing tambear primitive); Euler = Binomial
   FIR filter. The Tauberian theorem IS filter consistency. Summability theory IS filter theory.
   Garden: `20260401-summability-as-filters.md`

3. **The Wynn ε exception.** Wynn is BC because the e-table recurrence is non-commutative. Is there a "nonlinear KDE" with analogous non-commutativity? Perhaps k-nearest-neighbor density estimation (k-NN DE) — order of data affects which neighbors are chosen. This might be the KDE analogue of Wynn ε.

4. **Product family?** All summability methods × all KDE kernels → a structured product space of "kernel-weighted sequence summarizers." The taxonomy would be: kernel type × domain type. Both axes have natural structure (kernel = regularity class of distribution; domain = discrete series vs continuous density).

---

## Empirical Verification (observer, 2026-04-01)

`cesaro_sum()` implemented and verified. 19/19 tests pass. Key results on Leibniz π/4 (20 terms):

```
Raw partial sum:    error = 1.25e-2    (no kernel)
Cesàro (uniform):   error = 6.83e-3    ← ~2x improvement
Aitken Δ² (generic nonlinear): error = 9.08e-6
Euler (binomial, matched):      error = 8.69e-9    ← 6 orders better than Cesàro
Wynn ε (iterated):  error = 3.33e-16   ← machine precision
```

**The hierarchy is: domain-matched linear kernel > generic nonlinear accelerator > unmatched linear kernel.**

Euler (binomial kernel matched to alternating series) beats Aitken (generic nonlinear accelerator) by 3 orders of magnitude. This is the KDE bandwidth analogy: choosing the right kernel for the convergence type pays more than choosing a "stronger" algorithm with a mismatched kernel.

**Grandi's series verified**: Cesàro assigns 1 - 1 + 1 - 1 + ... the value 0.5 ± 0.01.
The raw partial sums oscillate (0, 1, 0, 1, ...) and never converge. Cesàro's uniform
kernel creates a value where the raw sequence has none — exactly like KDE creates a
continuous density from a finite discrete sample.

"The kernel doesn't just accelerate; it creates a value that the raw sequence doesn't have."
— same principle: KDE at unsampled points, Cesàro at divergent series.

## Related Campsites
- [rho-sigma-kingdoms](../20260401-rho-sigma-kingdoms/) — Wynn ε as Kingdom BC inhabitant
- [scatter-attract-duality](../20260401-scatter-attract-duality/) — attract() as Kingdom C primitive

## Implementation
- `src/series_accel.rs` — Aitken Δ², Euler transform, Wynn ε, Richardson (all verified, 13 tests)
- `src/series_accel.rs` — `cesaro_sum()` added by observer (6 new tests, 2026-04-01)
- `src/nonparametric.rs` — KDE with bandwidth selection (Silverman's rule implemented)
- Connection: `fn cesaro_sum(terms: &[f64]) -> f64` = `partial_sums(terms).iter().sum() / n` — two chained accumulates, one line
