# Family 36: Symbolic Computation — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (polynomial coefficient arithmetic), C (term rewriting / simplification)

---

## Core Insight: Two Distinct Layers

Symbolic computation splits into two fundamentally different layers:

1. **Coefficient arithmetic**: polynomial add/mul/div, GCD, resultant — these are accumulate over coefficient arrays. Kingdom A. GPU-friendly.
2. **Term rewriting**: simplification, pattern matching, canonical forms — these are tree-recursive transformations. Kingdom C (iterative tree walks). Inherently sequential at the tree level.

The honest assessment: tambear can accelerate the ARITHMETIC layer (coefficients are arrays, operations are accumulate). The REWRITING layer is a tree walker — it doesn't parallelize in the same way. The right architecture: rewriting on CPU orchestrates coefficient arithmetic on GPU.

This is NOT a drop-in Wolfram Alpha replacement. It's a *computational engine for symbolic expressions* — the part that involves data-parallel number crunching. Pattern matching and simplification remain CPU-bound tree operations.

---

## 1. Polynomial Representation

### Dense Representation
Polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxₙ stored as coefficient vector [a₀, a₁, ..., aₙ].

**This IS an accumulate problem.** Coefficient arrays are the data. Operations are element-wise or convolution-like.

### Sparse Representation
For multivariate or high-degree polynomials: list of (coefficient, exponent_vector) pairs, sorted by monomial order.

### Monomial Orderings (CRITICAL for multivariate)

| Order | Definition | Used by |
|-------|-----------|---------|
| Lex | Compare exponents left-to-right | Human-readable output |
| GrLex | Total degree first, then lex | Gröbner basis (standard) |
| GRevLex | Total degree first, then reverse lex reversed | Buchberger (fastest in practice) |

**Decision**: Store in GRevLex (fastest for computation), convert to Lex for display.

---

## 2. Polynomial Arithmetic

### Addition
Element-wise: `accumulate(Contiguous, a_i + b_i, Identity)` — trivially parallel.

For sparse: merge sorted lists by monomial order, combine like terms. This is a merge-join.

### Multiplication (Convolution)
Dense: coefficient of degree k = Σ_{i+j=k} aᵢ·bⱼ.

```
c_k = accumulate(All, a_i · b_{k-i}, Sum)    for each k
```

**Three algorithms by degree**:

| Method | Complexity | Use when |
|--------|-----------|----------|
| Schoolbook | O(n²) | deg < 64 |
| Karatsuba | O(n^1.585) | 64 ≤ deg < 1024 |
| FFT-based | O(n log n) | deg ≥ 1024 |

FFT-based: pad to 2N, FFT both, pointwise multiply, IFFT. Uses F03 infrastructure directly.

### Division (with remainder)
Polynomial long division: sequential — each step depends on the previous quotient term.

**Kingdom B**: this is a sequential scan with state = (quotient_so_far, remainder_so_far).

### GCD (Greatest Common Divisor)
**Euclidean algorithm**: gcd(a,b) = gcd(b, a mod b). Inherently sequential (Kingdom C).

**Subresultant PRS** (pseudo-remainder sequence): avoids coefficient explosion.

**Modular GCD** (for large polynomials): evaluate at many points → integer GCD → reconstruct. The evaluation step IS parallel (accumulate over evaluation points).

### Resultant
```
Res(f,g) = det(Sylvester_matrix(f,g))
```
Uses F02 determinant. Useful for elimination and implicitization.

---

## 3. Symbolic Differentiation

### Rules (tree-recursive)
```
d/dx [c] = 0
d/dx [x] = 1
d/dx [f+g] = f' + g'
d/dx [f·g] = f'·g + f·g'      (product rule)
d/dx [f(g)] = f'(g)·g'         (chain rule)
d/dx [xⁿ] = n·xⁿ⁻¹            (power rule)
```

This is pure tree transformation — NOT an accumulate problem. CPU-bound.

### For polynomials specifically
Differentiation on coefficient array: `a'_k = (k+1)·a_{k+1}`. This IS parallel:
```
accumulate(Contiguous, (i+1) * a_{i+1}, Identity)
```

### Partial derivatives
For multivariate: differentiate w.r.t. one variable, treating others as constants. Same rules, different variable.

---

## 4. Symbolic Integration

### Polynomial Integration
Antiderivative on coefficient array: `A_k = a_{k-1}/k`. Parallel (same as differentiation, reversed).

### Risch Algorithm (general symbolic integration)
Decidable but EXTREMELY complex. Full implementation is ~10,000 lines in Mathematica's codebase.

**Pragmatic approach**: handle the cases that arise in practice:
1. Polynomial: trivial (coefficient manipulation)
2. Rational functions: partial fraction decomposition → integrate each term
3. Elementary functions: table of known integrals + pattern matching
4. Everything else: return unevaluated (honest answer)

### Partial Fraction Decomposition
Given p(x)/q(x) where deg(p) < deg(q):
1. Factor q(x) (requires root-finding or factorization)
2. Solve for coefficients via system of equations (F02 linear solve)

---

## 5. Equation Solving

### Linear Systems
Already F02. `Ax = b` → factorize and solve.

### Polynomial Root Finding

| Degree | Method | Exact? |
|--------|--------|--------|
| 1 | x = -b/a | Yes |
| 2 | Quadratic formula | Yes |
| 3 | Cardano's formula | Yes (but use depressed cubic) |
| 4 | Ferrari's formula | Yes (but use resolvent cubic) |
| ≥5 | Abel-Ruffini: NO general radical formula | Numerical only |

**CRITICAL**: For degree ≥ 5, no closed-form solution exists (Abel-Ruffini theorem). Must use numerical methods (F05 Newton's method, companion matrix eigenvalues from F02).

### Companion Matrix Method
Roots of p(x) = eigenvalues of the companion matrix. This is an F02 eigendecomposition.

```
C = [[0, 0, ..., -a₀/aₙ],
     [1, 0, ..., -a₁/aₙ],
     [0, 1, ..., -a₂/aₙ],
     ...
     [0, 0, ..., -aₙ₋₁/aₙ]]
```

### Systems of Polynomial Equations
**Gröbner basis**: the "Gaussian elimination for polynomials." Buchberger's algorithm.

**Complexity**: doubly exponential in worst case (EXPSPACE-complete). In practice, much better for low-degree systems.

**Implementation**: Buchberger with Gebauer-Möller criteria for pair selection. F₄/F₅ algorithms for performance.

---

## 6. Expression Simplification

### Canonical Forms
The fundamental problem: when are two expressions equal? Undecidable in general (Richardson 1968).

**Practical approach**: normalize to canonical form, then compare.

| Expression type | Canonical form |
|----------------|---------------|
| Polynomial | Sorted by monomial order, like terms combined |
| Rational function | p(x)/q(x) with gcd(p,q) = 1 |
| Trigonometric | All in terms of sin and cos |
| Logarithmic | Expanded (log product → sum of logs) |

### Rewriting Rules (Kingdom C)
- Commutativity: a+b → a+b (sorted)
- Associativity: (a+b)+c → a+b+c (flattened)
- Distribution: a(b+c) → ab+ac (when beneficial — heuristic decision)
- Evaluation: 2+3 → 5 (constant folding)
- Identity: a+0 → a, a·1 → a, a·0 → 0
- Power: xⁿ·xᵐ → x^(n+m)

### Convergence
Simplification must terminate. Use a well-founded ordering on expressions (e.g., total degree decreases, then size decreases). Without a termination guarantee, simplification loops.

---

## 7. Special Functions (Symbolic Knowledge)

### Table of Derivatives
```
d/dx sin(x) = cos(x)
d/dx cos(x) = -sin(x)
d/dx eˣ = eˣ
d/dx ln(x) = 1/x
d/dx tan(x) = sec²(x)
d/dx arcsin(x) = 1/√(1-x²)
d/dx arctan(x) = 1/(1+x²)
```

### Table of Integrals
```
∫ xⁿ dx = x^(n+1)/(n+1) + C    (n ≠ -1)
∫ 1/x dx = ln|x| + C
∫ eˣ dx = eˣ + C
∫ sin(x) dx = -cos(x) + C
∫ cos(x) dx = sin(x) + C
∫ 1/(1+x²) dx = arctan(x) + C
∫ 1/√(1-x²) dx = arcsin(x) + C
```

These are lookup tables (gather), not computation.

---

## 8. Numerical Evaluation

### Horner's Method
Evaluate polynomial p(x) = a₀ + a₁x + ... + aₙxⁿ:
```
result = aₙ
for k in (n-1)..0:
    result = result * x + aₖ
```

**This IS an Affine scan**: state = result, update = Affine(x, aₖ). Kingdom B.

**Multi-point evaluation**: evaluate at many points simultaneously. Each point is independent → embarrassingly parallel.
```
accumulate(Tiled{points, coeffs}, horner_step, Affine)
```

### Interval Arithmetic
For certified evaluation: [a,b] + [c,d] = [a+c, b+d], [a,b]·[c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)].

Useful for root isolation (Sturm's theorem) and certified plotting.

---

## 9. Edge Cases

| Operation | Edge Case | Expected |
|-----------|----------|----------|
| Poly mul | One operand is zero polynomial | Zero polynomial |
| Poly div | Division by zero polynomial | Error (not NaN) |
| Poly GCD | Both zero | Zero (convention) |
| Root finding | Repeated roots | Multiplicities correct |
| Root finding | Complex roots of real poly | Come in conjugate pairs |
| Differentiation | Constant expression | Exact 0, not 0.0 |
| Integration | 1/x | ln|x|, not error |
| Simplification | 0/0 | Undefined, not 0 |
| Degree ≥ 5 | Exact roots requested | Return "no closed form" |

---

## Sharing Surface

### Reuses from Other Families
- **F02 (Linear Algebra)**: eigendecomposition for companion matrix roots, linear solve for interpolation
- **F03 (Signal Processing)**: FFT for fast polynomial multiplication (deg ≥ 1024)
- **F05 (Optimization)**: Newton's method for numerical root polishing
- **F31 (Interpolation)**: polynomial interpolation, Chebyshev approximation
- **F38 (Arbitrary Precision)**: exact integer arithmetic for GCD, resultant without coefficient blowup

### Provides to Other Families
- **F31**: polynomial evaluation, derivative computation
- **F05**: symbolic gradient (exact, not finite difference)
- **F32 (Numerical)**: symbolic differentiation for automatic Jacobian generation
- **All families**: symbolic expression type for formula representation in .tbs

---

## Implementation Priority

**Phase 1** — Polynomial coefficient arithmetic (~150 lines):
1. Dense polynomial representation
2. Add, subtract, scalar multiply (element-wise)
3. Multiply (schoolbook for small, FFT for large)
4. Evaluate (Horner's method, multi-point parallel)
5. Differentiate, integrate (coefficient manipulation)

**Phase 2** — Polynomial algebra (~200 lines):
6. Division with remainder
7. GCD (Euclidean, subresultant)
8. Root finding (quadratic/cubic/quartic formulas + companion matrix)
9. Partial fraction decomposition

**Phase 3** — Expression trees (~300 lines):
10. Expression AST (tree representation)
11. Symbolic differentiation (tree rules)
12. Simplification (canonical forms, rewriting)
13. Pattern matching for integration table

**Phase 4** — Advanced (~200 lines):
14. Sparse multivariate representation
15. Gröbner basis (Buchberger)
16. Interval arithmetic for certified evaluation

---

## Composability Contract

```toml
[family_36]
name = "Symbolic Computation"
kingdom = "Mixed — A (coefficient arithmetic), C (term rewriting)"

[family_36.shared_primitives]
poly_arithmetic = "Dense polynomial add/mul/div via coefficient arrays"
poly_eval = "Horner evaluation (Affine scan) + multi-point parallel"
differentiation = "Coefficient shift (polynomial) or tree recursion (general)"
root_finding = "Companion matrix eigenvalues (F02) or closed-form (deg ≤ 4)"

[family_36.reuses]
f02_linear_algebra = "Eigendecomposition for roots, linear solve for coefficients"
f03_signal_processing = "FFT for fast polynomial multiplication"
f05_optimization = "Newton root polishing, numerical equation solving"
f31_interpolation = "Polynomial interpolation"
f38_arbitrary_precision = "Exact integer arithmetic for GCD/resultant"

[family_36.provides]
poly_type = "Dense/sparse polynomial representation"
symbolic_gradient = "Exact derivative (no finite differences)"
expression_ast = "Symbolic expression tree for .tbs formula display"
root_certificates = "Certified root isolation via interval arithmetic"

[family_36.consumers]
f05_optimization = "Symbolic gradients for exact Newton"
f31_interpolation = "Polynomial evaluation and manipulation"
f32_numerical = "Symbolic Jacobians for ODE solvers"
tbs_language = "Expression display and manipulation"
```
