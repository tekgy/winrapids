# Dimensional Entropy — Spec for Tambear

**Status:** idea spec. Not prescribing implementation — tambear should
figure out the right way to build this using its own primitives.
**Source:** TERNYX-SIP signal farm design session, 2026-04-17.

---

## The insight

Standard compression-based entropy (zstd ratio, Shannon entropy of
byte streams) measures how unpredictable individual VALUES are. It
misses structure that lives in the RELATIONSHIPS between dimensions.

A matrix of 22 correlated time series signals, all ramping together
in lockstep, has HIGH byte entropy (every value is different) but
LOW dimensional entropy (one principal component explains 99% of
variance). zstd says "chaotic." PCA says "trivially simple — one
hidden factor drives everything."

The DIVERGENCE between byte-level entropy and dimensional entropy
is itself a signal: it detects coordinated regime shifts, single-
factor crises, and structural decorrelation events that no single-
column measure can see.

---

## What to compute

Given a matrix `X` of shape `(T, D)` — T time steps × D signals:

### 1. Effective dimensionality

```
eigenvalues = PCA(X).explained_variance_ratio_  // D values, sum to 1.0
effective_dim = exp(-Σ λ_i × ln(λ_i))          // Shannon entropy of eigenspectrum
                                                 // Range: 1.0 (one factor) to D (all equal)
```

This is the "participation ratio" or "effective rank" — how many
independent factors are driving the data. Pure accumulate-style:
compute the covariance matrix (outer products → sum), eigendecompose,
apply the entropy formula to the eigenspectrum.

### 2. Factor concentration

```
factor_concentration = eigenvalues[0]  // fraction of variance in top component
                                       // Range: 1/D (uniform) to 1.0 (single factor)
```

When this spikes toward 1.0, everything is moving together. Crisis
signal.

### 3. Byte entropy (baseline comparison)

```
byte_entropy = compressed_size(X, zstd) / raw_size(X)
```

Standard compression ratio. The reference that dimensional entropy
diverges FROM.

### 4. Entropy divergence

```
divergence = byte_entropy / (effective_dim / D)
// Normalized so both terms are in [0, 1].
// High divergence = bytes look chaotic but structure is simple
//                  = coordinated regime shift
// Low divergence  = bytes and structure agree
```

---

## Where this fits in tambear's framework

The covariance matrix computation is Kingdom A:

```
Cov(X) = (1/T) × Xᵀ X - μμᵀ

Xᵀ X = accumulate(X, Tiled{b=X, m=D, n=D, k=T}, Value, DotProduct)
μ    = accumulate(X, All, Value, Add) / T
```

That's a tiled dot product (already in tambear) plus a rank-1 outer
product subtraction. The eigendecomposition of a D×D matrix (D=22)
is tiny — Jacobi or QR iteration on a 22×22 matrix is microseconds
on CPU. Not worth a GPU kernel.

The interesting tambear question: can the covariance + eigendecomp
be FUSED with the other signal accumulations from the SIP spec? The
covariance needs the same `(T, D)` signal matrix as input. If
Launch 1 (per-bucket accumulations) produces the signal matrix, and
Launch 3 (per-hour reductions) includes the covariance as one of
its outputs, the dimensional entropy falls out of the same pipeline.

### Rotation-based compression (the TurboQuant-inspired path)

An alternative to PCA for measuring dimensional entropy: actually
COMPRESS the matrix with a rotation-based quantizer and measure how
well it compresses at various bit depths.

```
For each bit_depth in [2, 3, 4, 8, 16]:
    rotated = random_orthogonal_matrix(D) @ X.T   // or PCA rotation
    quantized = lloyd_max_quantize(rotated, bits=bit_depth)
    reconstruction_error = mse(X, inverse_rotate(dequantize(quantized)))
    
    entropy_at_depth[bit_depth] = reconstruction_error
```

The SHAPE of the `entropy_at_depth` curve tells you about the
information structure:
- Steep drop (2-bit is almost as good as 16-bit) → low-dimensional,
  highly compressible, one-factor regime
- Gradual decline → information is spread across many dimensions,
  high effective dimensionality

This is an empirical rate-distortion curve. Its shape is the
dimensional entropy measure. More robust than eigenvalue-based
measures because it accounts for non-linear structure that PCA
misses.

### The NaN-skip contract

Same as all SIP accumulations: skip non-finite values, substitute
identity elements. The covariance matrix is computed from valid
observations only. `n_valid` determines the normalization factor.

---

## What NOT to prescribe

- The specific rotation method (random vs PCA vs Householder).
  Tambear should experiment.
- The eigendecomposition algorithm. D=22 is tiny; anything works.
- Whether to fuse with other SIP accumulations or run separately.
  Tambear knows its kernel fusion architecture better than we do.
- The exact bit depths for the rate-distortion curve. 
- Whether GPU or CPU is right for the 22×22 eigendecomp.

## What IS prescribed

- Input: `(T, D)` f64 matrix where T = time steps, D = signal count
- Output: `effective_dim: f64`, `factor_concentration: f64`,
  `entropy_divergence: f64`
- NaN-skip semantics on input
- Must be deterministic (same input → same output)
