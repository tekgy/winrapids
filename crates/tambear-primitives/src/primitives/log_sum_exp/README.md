# log_sum_exp

Numerically stable computation of `log(Σ exp(xᵢ))`.

## When to use

Any time you're working in log-probability space and need to combine probabilities. The naive `(values.iter().map(|v| v.exp()).sum::<f64>()).ln()` overflows for large values and underflows for small values. This primitive handles both.

## Formula

```
lse(x₁, ..., xₙ) = max(x) + ln(Σᵢ exp(xᵢ - max(x)))
```

## Composes with

- `hmm_forward`: the inner combine step IS log_sum_exp
- `softmax`: `softmax(x)ᵢ = exp(xᵢ - lse(x))`
- `mixture_log_likelihood`: `log p(x) = lse(log_weights + log_component_densities)`
- `bayes_model_evidence`: `log p(data) = lse(log_priors + log_likelihoods)`
- `attention`: `attn_weights = softmax(QK^T/√d) = exp(scores - lse(scores))`

## Semiring

This is the `add` operation of the LogSumExp semiring `(ℝ, lse, +)`:
- add = log_sum_exp (associative, commutative)
- mul = real addition (in log space, multiplication becomes addition)
- zero = -∞ (identity for lse)
- one = 0 (identity for addition)

HMM forward and Viterbi are the SAME prefix scan under different semirings:
- Forward: LogSumExp semiring (sum over paths)
- Viterbi: TropicalMaxPlus semiring (max over paths)

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| values | &[f64] | yes | — | Log-space values to combine |

No tunable parameters. This is a pure mathematical operation.
