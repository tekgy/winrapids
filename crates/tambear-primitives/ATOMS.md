# Atom Inventory

## Current: 40 atoms (27 Expr + 7 Grouping + 6 Op)

### Expr Nodes (27)
Leaves: Val, Val2, Ref, Lit(f64), Var(String)
Unary: Neg, Abs, Recip, Sq, Sqrt, Ln, Exp, Floor, Ceil, Sign, IsFinite
Binary: Add, Sub, Mul, Div, Pow, Min, Max
Ternary: Clamp
Comparison: Gt, Lt, Eq

### Grouping (7)
All, ByKey, Prefix, Segmented, Windowed, Tiled, Graph

### Op (6)
Add, Max, Min, Mul, And, Or

## Missing Expr Nodes

### Trigonometric
- Sin, Cos, Tan — needed for: Fourier, signal processing, angular stats, physics
- Asin, Acos, Atan — inverse trig
- Atan2(y, x) — two-argument arctangent, needed for: phase computation, polar coords
- Sinh, Cosh, Tanh — hyperbolic, needed for: neural networks (tanh activation), physics

### Rounding / Modular
- Round — nearest integer
- Trunc — toward zero
- Mod(a, b) — remainder, needed for: modular arithmetic, circular stats, hash functions
- Fract — fractional part

### Bitwise / Integer (if we support integer types)
- Not clear we need these in the Expr AST — maybe in the IR

### Conditional
- If(cond, then, else) — ternary conditional expression
  Currently approximated by Gt/Lt + Mul, but a proper conditional is cleaner
  Needed for: piecewise functions, ReLU, Huber loss, censored data

### Multi-input
- Val3, Val4, ... — additional column references beyond Val2
  Needed for: multivariate operations, MANOVA, multi-column transforms
  OR: ValN(usize) — indexed column reference

### Accumulate-specific
- Lag(n) — value from n steps ago in a scan context
  Needed for: AR models, difference operations, autocorrelation
  This might be a Grouping feature rather than an Expr node

## Missing Grouping Variants

### Likely needed soon
- Strided — every k-th element (for downsampling, interleaving)
- Blocked — fixed-size blocks (for batch processing)
  May be expressible as ByKey with block-index keys

### Longer term
- Tree — tree-structured accumulation (for hierarchical, phylogenetic)
- Circular — modular wrap-around (for Ising, lattice, periodic boundary)
- Adaptive — data-dependent partitioning

## Missing Op Variants

### Maybe needed
- BitwiseAnd, BitwiseOr — for bitmask operations
- Xor — for parity, hash
- Concat — for building lists/vectors (not a scalar op)

### Semiring integration
- The existing Semiring trait (Additive, TropicalMinPlus, etc.)
  should be accessible as Op variants or Op parameters.
  Currently Semiring is separate from Op. They should unify.

## Assessment

~27 Expr + 7 Grouping + 6 Op = 40 atoms handles:
- All scalar statistics (mean, variance, skewness, kurtosis, all flavors)
- All correlations (Pearson, covariance, cross-products)
- All counting/filtering (count, count_positive, count_finite)
- All norms (L1, L2, Linf)
- All basic transforms (log, exp, power, reciprocal, abs)
- Geometric and harmonic means
- Centered deviations (for two-pass methods)
- Matrix operations (via Tiled+DotProduct)
- Prefix scans (cumsum, running max/min)

Adding ~15 trig/hyperbolic + If + Mod + Atan2 + ValN gets us to ~55 atoms,
which likely covers 95%+ of all mathematical recipes.

The remaining 5% are domain-specific operations (FFT butterfly, sort,
rank, string operations) that may need special treatment outside the
Expr → Accumulate → Gather pattern.
