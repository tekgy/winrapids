# Information Theory Family — Tambear Decomposition Notes

Brief notes on Family 25 (Information Theory) from the landscape, informed by the
fintek execute.py implementations.

---

## Shannon Entropy (fintek: entropy/execute.py)

Current Python:
```python
counts, _ = np.histogram(returns, bins=n_bins)
probs = counts / n
h = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
```

**Tambear decomposition**:

```
Step 1: Compute bin index per element
  bin_idx[i] = clamp(floor((x[i] - x_min) / bin_width), 0, n_bins-1)
  → ScatterJit phi: "(int)((v - min) / bin_width)"  — custom JIT
  OR: FilterJit threshold sweep to find bin membership

Step 2: Count per bin
  counts[b] = accumulate(data, ByKey{bin_idx, n_bins}, One, Add)
  → scatter_phi("1.0", bin_idx, n_bins)  — pure count scatter

Step 3: Normalize
  probs = counts / n   — elementwise division (scalar)

Step 4: Entropy formula
  h = -Σ p * log2(p)  = -accumulate(probs, All, Custom("-v * log2(v)"), Add)
                          where we skip p=0 via a filter first

Steps 2 and 4 are existing accumulate operations.
Step 1 (bin quantization) needs a custom phi expression or a new quantization pass.
```

**Key**: histogram entropy is `scatter(ByKey{bin_idx}, Count)` + `reduce(All, -p*log2(p))`.
Two accumulate calls. No distance matrix. No new GPU primitives beyond quantization.

---

## Entropy Rate — Joint Histogram (fintek)

```python
joint_counts, _, _ = np.histogram2d(x_prev, x_curr, bins=n_bins)
```

**Tambear decomposition**: 2D histogram = 1D scatter with a composite key:
```
key = bin_idx_prev * n_bins + bin_idx_curr   — linear index for 2D grid
counts[key] = accumulate(data[1:], ByKey{key, n_bins²}, One, Add)
```
Where `bin_idx_prev` and `bin_idx_curr` are the bin indices of consecutive returns.
This is one ScatterJit call with a 2D key linearized to a 1D index.

---

## Sample Entropy (fintek: sample_entropy/execute.py)

The core inner loop:
```python
if np.max(np.abs(x[i:i+m] - x[j:j+m])) <= r:
    b_count += 1
```

This is an L∞ (Chebyshev) distance threshold count on template pairs.
Tambear decomposition:
1. Build template matrix: M[i] = [x[i], x[i+1], ..., x[i+m-1]] — delay embedding (CPU)
2. L∞ distance matrix: requires `TiledOp` with max-combine instead of sum-combine
3. Threshold count B = |{(i,j): L∞(M[i], M[j]) ≤ r}|
4. Length-(m+1) count A = |{(i,j): L∞(M[i], M[j]) ≤ r AND |x[i+m] - x[j+m]| ≤ r}|
5. SampEn = -log(A/B)

The L∞ step requires a max-combine TiledOp (not yet in TiledEngine).
Alternative: compute L∞ as `max(|x_1-y_1|, |x_2-y_2|, ..., |x_m-y_m|)` via GPU kernel.
This is the one algorithm in the fintek chaos family that needs a new TiledEngine op.

---

## Mutual Information (fintek: mutual_info/execute.py — not yet read)

Standard MI: `I(X;Y) = H(X) + H(Y) - H(X,Y)`
All three terms are histogram entropies → all computable as scatter(ByKey{bin_idx}, Count).
`I(X;Y)` = three histogram entropy computations, sharing the quantization pass.

For continuous MI (KSG estimator): requires KNN distances → TamSession distance matrix.
KSG MI is standard in information theory for continuous variables.

---

## Transfer Entropy (fintek: transfer_entropy/execute.py — not yet read)

`TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})`
= 3D histogram entropy (joint of Y_t, Y_{t-1}, X_{t-1}) minus 2D
→ Three histogram computations with linearized 2D/3D bin keys.
Same scatter pattern, higher-dimensional key.

The k-th bin key for a d-dimensional histogram: `key = Σ_j bin_j * n_bins^j`
For n_bins=16 and d=3: key ∈ [0, 4095] — manageable range.

---

## Key Insight: Entropy = Histogram = ByKey Count

All entropy-based metrics (Shannon, entropy rate, mutual information, transfer entropy)
reduce to:
1. **Quantize** each dimension to bin indices (custom phi)
2. **Count per bin** via `accumulate(ByKey{linearized_key}, One, Add)`
3. **Apply entropy formula** `-Σ p log p` via `accumulate(All, Custom("-v*log(v)"), Add)`

The quantization step (1) requires a floor/int-cast JIT expression — currently missing
but trivial to add to ScatterJit's phi vocabulary.

Once bin quantization is available as a JIT phi, ALL histogram-based entropy computations
fall out of existing accumulate infrastructure. No new GPU primitives beyond that.
