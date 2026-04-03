# Family 31: Interpolation & Approximation -- Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.

---

## Overview

Family 31 covers: interpolation (exact passage through data points) and approximation (best fit without exact passage). These are foundational for everything from data resampling and missing-value fill to Gaussian process regression (F31 feeds F20, F10, F28, F03).

The core distinction: **interpolation** satisfies f(xi) = yi exactly; **approximation** minimizes some error norm without that constraint.

### Accumulate Connection

Most interpolation methods decompose into:
1. A **setup phase** (solve a linear system for coefficients) -- often sequential or requiring F02 linear algebra
2. An **evaluation phase** (evaluate the interpolant at query points) -- embarrassingly parallel

The evaluation phase is almost always `gather(query_points, coefficient_table) + fused_expr(polynomial_eval)`. The setup phase varies: some are purely local (Akima, PCHIP -- parallelizable), some require global solves (cubic spline tridiagonal system, RBF dense system).

### Key Principle

Every method in this family has a **locality spectrum**:
- **Fully local**: linear interp, Akima, PCHIP (changing one data point affects only neighbors)
- **Global but structured**: cubic spline (tridiagonal -- O(n) solve, but sequential)
- **Fully global**: polynomial interp, RBF, Chebyshev, Pade (every coefficient depends on all data)

Locality determines GPU strategy. Local methods parallelize trivially. Global methods need either parallel prefix (tridiagonal) or dense linear algebra (RBF).

---

## 1. Linear Interpolation

### 1.1 Definition

Given sorted pairs (x0,y0), (x1,y1), ..., (xn,yn) with x0 < x1 < ... < xn, the piecewise linear interpolant on interval [xi, xi+1] is:

```
L(x) = yi + (yi+1 - yi) / (xi+1 - xi) * (x - xi)
```

Equivalently, using the parameter t = (x - xi) / (xi+1 - xi) in [0,1]:

```
L(x) = (1 - t) * yi + t * yi+1
```

### 1.2 Inputs / Outputs

- **Inputs**: Sorted knot array x[0..n], value array y[0..n], query points q[0..m]
- **Outputs**: Interpolated values f[0..m]
- **Assumptions**: x strictly increasing (no duplicates). n >= 1 (at least 2 points).

### 1.3 Extrapolation Handling

For query point q outside [x0, xn], choices:
1. **Linear extrapolation**: extend the first/last segment (default in numpy.interp with appropriate args)
2. **Constant extrapolation**: f(q) = y0 for q < x0, f(q) = yn for q > xn (numpy.interp default)
3. **NaN**: return NaN outside bounds (scipy.interp1d default with bounds_error=False, fill_value=nan)
4. **Error**: raise/signal out-of-bounds

**Cross-package**: numpy.interp uses constant extrapolation by default. scipy.interp1d raises an error by default. MATLAB interp1 returns NaN by default. We must support all modes.

### 1.4 Edge Cases

- **Query exactly at knot**: return yi exactly (no floating point issues since t=0)
- **Two points only (n=1)**: single line segment. Extrapolation behavior matters.
- **Repeated x values**: UNDEFINED. x must be strictly increasing. If xi = xi+1 with yi != yi+1, the slope is infinite. Detect and error.
- **Single point (n=0)**: constant function f(q) = y0 for all q (debatable; some packages error)
- **NaN in data**: NaN in y propagates to any interval touching that point. NaN in x is invalid (sorting undefined).

### 1.5 Numerical Stability

Excellent. Only one multiply and one add per evaluation. The division (yi+1 - yi)/(xi+1 - xi) can lose precision when xi+1 - xi is very small (closely spaced points with very different y values), but this is inherent to the data, not the method.

**Catastrophic cancellation**: When x is very close to xi and yi is very large, computing yi + slope*(x-xi) can lose trailing bits. Use the t-form: (1-t)*yi + t*yi+1 which is more stable (convex combination, both terms same sign if yi,yi+1 same sign).

### 1.6 GPU Decomposition

**Embarrassingly parallel.** Each query point is independent.

1. **Locate interval**: binary search to find i such that xi <= q < xi+1. This is `gather(q, x_knots)` with binary search addressing.
2. **Evaluate**: fused_expr computing the linear combination.

```
accumulate decomposition:
  - Pre-step: sort knots (if not pre-sorted)
  - Per query: gather(binary_search(q, x), {x, y}) then fused_expr(lerp)
  - Grouping: None (each query independent)
  - Op: None (no reduction)
```

For m query points on n knots: O(m * log n) with binary search, O(m) if queries are also sorted (can use merge-path for sorted-sorted intersection).

### 1.7 Failure Modes

- Non-sorted x: silently wrong results (binary search assumes sorted)
- Extrapolation without explicit mode: users get unexpected behavior
- Very non-uniform spacing: some intervals may have extreme slopes while others are flat; linear interp is not smooth (C0 continuous only, discontinuous first derivative at knots)

---

## 2. Polynomial Interpolation

### 2.1 The Fundamental Problem

Given n+1 distinct points (x0,y0),...,(xn,yn), there exists a UNIQUE polynomial P of degree <= n such that P(xi) = yi for all i.

**Existence and uniqueness**: guaranteed by the Vandermonde determinant being nonzero when all xi are distinct. The Vandermonde matrix V with Vij = xi^j has det(V) = product_{i<j} (xj - xi).

### 2.2 Lagrange Form

```
L(x) = sum_{i=0}^{n} yi * li(x)

where li(x) = product_{j=0, j!=i}^{n} (x - xj) / (xi - xj)
```

Each li(x) is the unique degree-n polynomial that equals 1 at xi and 0 at all other xj.

**Properties**:
- Evaluating L(x) at one point: O(n^2) operations
- No setup phase required (weights can be precomputed in O(n^2))
- Not suitable for adding new points (must recompute everything)

**Barycentric form** (numerically superior):

Define weights:
```
wi = 1 / product_{j!=i} (xi - xj)
```

Then:
```
L(x) = [sum_{i=0}^{n} wi*yi/(x-xi)] / [sum_{i=0}^{n} wi/(x-xi)]
```

**Advantages of barycentric form**:
- O(n) evaluation after O(n^2) weight precomputation
- Numerically stable (Higham 2004 proved backward stability)
- Adding a point: O(n) to update all weights
- If x = xi exactly: just return yi (avoid 0/0; special-case this)

### 2.3 Newton's Divided Differences

Define the divided differences recursively:
```
f[xi] = yi
f[xi, xi+1] = (f[xi+1] - f[xi]) / (xi+1 - xi)
f[xi, xi+1, ..., xi+k] = (f[xi+1,...,xi+k] - f[xi,...,xi+k-1]) / (xi+k - xi)
```

The interpolating polynomial in Newton form:
```
P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ... + f[x0,...,xn](x-x0)...(x-xn-1)
```

**Evaluation via Horner-like nesting** (O(n) per point):
```
P(x) = f[x0] + (x-x0)(f[x0,x1] + (x-x1)(f[x0,x1,x2] + ... ))
```

**Advantages over Lagrange**:
- Adding a new point (xn+1, yn+1): only need ONE new column of the divided difference table -- O(n) work
- Evaluation is O(n) per query via nested form
- Divided difference table construction: O(n^2) total, can be done incrementally

**Divided difference table construction** (triangular array):

```
Stage 0: f[x0], f[x1], ..., f[xn]              (n+1 values)
Stage 1: f[x0,x1], f[x1,x2], ..., f[xn-1,xn]   (n values)
Stage 2: f[x0,x1,x2], ..., f[xn-2,xn-1,xn]      (n-1 values)
...
Stage n: f[x0,...,xn]                              (1 value)
```

Each stage k has n-k+1 entries. Total: (n+1)(n+2)/2 entries, O(n^2) operations.

### 2.4 Runge's Phenomenon

**The problem**: For uniformly spaced nodes, high-degree polynomial interpolation produces wild oscillations near the interval endpoints.

**Classic example**: f(x) = 1/(1 + 25x^2) on [-1,1]. With n equally spaced nodes, the interpolation error GROWS exponentially near the endpoints as n increases:

```
max|f(x) - Pn(x)| -> infinity as n -> infinity
```

This is not a numerical artifact -- it is a theorem (Runge 1901). The error bound for degree-n polynomial interpolation on [-1,1] with equally spaced nodes includes the factor:

```
max |product_{i=0}^{n} (x - xi)| = h^{n+1} * n! / 4   (near endpoints)
```

where h = 2/n. This grows exponentially despite h -> 0.

**Consequence for GPU**: high-degree polynomial interpolation on uniform grids is MATHEMATICALLY UNSUITABLE. This is not fixable by better numerics. It is a property of uniform nodes + high degree.

### 2.5 Chebyshev Nodes (The Fix for Runge)

**Chebyshev nodes of the first kind** on [-1,1]:
```
xi = cos((2i + 1) * pi / (2(n+1)))   for i = 0, 1, ..., n
```

**Chebyshev nodes of the second kind** (extrema, including endpoints):
```
xi = cos(i * pi / n)   for i = 0, 1, ..., n
```

**Why they work**: Chebyshev nodes minimize max|product(x - xi)| over [-1,1]. The optimal node distribution clusters near endpoints, exactly compensating for the oscillation tendency.

**Error bound with Chebyshev nodes**:
```
max|f(x) - Pn(x)| <= (1/(2^n * n!)) * max|f^(n+1)(x)|
```

This DECREASES as n increases (for sufficiently smooth f).

**Mapping to [a,b]**: transform xi_ab = (a+b)/2 + (b-a)/2 * xi.

**Practical implication**: If you must do high-degree polynomial interpolation, ALWAYS use Chebyshev nodes. If you cannot choose your nodes (data is given), use piecewise methods (splines) instead.

### 2.6 Edge Cases and Failure Modes

- **Degree n polynomial through n+1 points**: exact but potentially oscillatory
- **Confluent nodes (xi = xj)**: Vandermonde is singular. Divided differences have 0/0. Detect and error.
- **Nearly confluent nodes**: catastrophic cancellation in divided differences. Condition number of Vandermonde ~ product of 1/min|xi-xj|.
- **Very high degree** (n > ~20 for uniform nodes): Runge phenomenon. Even with Chebyshev nodes, n > ~50 can have issues for non-analytic functions.
- **Equispaced nodes with n > 10**: suspect. Cross-check with spline.

### 2.7 GPU Decomposition

**Setup (divided differences)**: Sequential dependency between stages but within each stage, entries are independent.
```
accumulate decomposition:
  Stage k: accumulate(prev_stage, Windowed(2), (f[right]-f[left])/(x[i+k]-x[i]), identity)
  This is a sequential pipeline of n stages, each embarrassingly parallel within.
  Total: O(n) stages of O(n) parallel work = O(n) depth, O(n^2) work.
```

**Evaluation**: Embarrassingly parallel over query points. Each uses Horner's method (sequential over degree, parallel over queries).

```
Per query: scan-like (Prefix) but short (n steps). Better: just sequential loop per thread.
Grouping: None
Parallelism: over query points, not within one evaluation.
```

---

## 3. Cubic Spline Interpolation

### 3.1 Definition

Given n+1 points (x0,y0),...,(xn,yn) with x0 < x1 < ... < xn, a cubic spline S(x) is a piecewise cubic polynomial:

```
Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3    for x in [xi, xi+1]
```

satisfying:
1. **Interpolation**: Si(xi) = yi for all i
2. **C2 continuity**: S, S', S'' are all continuous at interior knots
3. **Boundary conditions** (two additional equations to close the system)

This gives n intervals, 4n unknowns, and 4n equations (n interpolation at left endpoints + n interpolation at right endpoints + (n-1) first derivative continuity + (n-1) second derivative continuity + 2 boundary conditions).

### 3.2 Coefficient Derivation

From the interpolation and continuity conditions, the coefficients can be expressed in terms of the second derivatives Mi = S''(xi):

Let hi = xi+1 - xi (interval widths).

```
ai = yi
bi = (yi+1 - yi)/hi - hi*(2*Mi + Mi+1)/6
ci = Mi/2
di = (Mi+1 - Mi)/(6*hi)
```

The continuity of S' at interior knots xi (i = 1,...,n-1) gives the **tridiagonal system**:

```
hi-1 * Mi-1 + 2*(hi-1 + hi) * Mi + hi * Mi+1 = 6*((yi+1-yi)/hi - (yi-yi-1)/hi-1)
```

This is n-1 equations in n+1 unknowns {M0,...,Mn}. The two boundary conditions determine M0 and Mn.

### 3.3 Natural Spline

**Boundary conditions**: S''(x0) = 0, S''(xn) = 0. That is: M0 = 0, Mn = 0.

The system reduces to (n-1) x (n-1) tridiagonal:

```
| 2(h0+h1)    h1          0     ...  0        | |M1  |   |d1|
| h1      2(h1+h2)    h2        ...  0        | |M2  | = |d2|
| 0         h2      2(h2+h3)   ...  0        | |M3  |   |d3|
| ...                                         | |... |   |..|
| 0         0         0     ... 2(hn-2+hn-1)  | |Mn-1|   |dn-1|
```

where di = 6*((yi+1-yi)/hi - (yi-yi-1)/hi-1).

**Properties**:
- Minimizes integral of (S''(x))^2 over the interval (among all C2 interpolants) -- the "smoothest" interpolant in this sense
- S''(x) is linear on each subinterval (since S is cubic)
- Second derivative is zero at endpoints: the curve "straightens out" at boundaries
- This can cause the spline to "flatten" near endpoints, undershooting/overshooting

### 3.4 Clamped Spline (Complete Spline)

**Boundary conditions**: S'(x0) = f'(x0), S'(xn) = f'(xn) (given endpoint slopes).

This modifies the first and last rows of the tridiagonal system:

```
Row 0: 2*h0*M0 + h0*M1 = 6*((y1-y0)/h0 - f'(x0))
Row n: hn-1*Mn-1 + 2*hn-1*Mn = 6*(f'(xn) - (yn-yn-1)/hn-1)
```

The full system is now (n+1) x (n+1) tridiagonal.

**Properties**:
- Requires knowledge of the derivative at endpoints
- Better approximation than natural spline when endpoint derivatives are known (O(h^4) vs O(h^2) error)
- Often the best choice when derivative information is available

### 3.5 Not-a-Knot Spline

**Boundary conditions**: S'''(x) is continuous at x1 and xn-1. Equivalently, the first two cubic segments form a single cubic, and the last two cubic segments form a single cubic.

This means:
```
d0 = d1   (first two intervals have the same third derivative)
dn-2 = dn-1  (last two intervals have the same third derivative)
```

Which gives:
```
h1*M0 - (h0+h1)*M1 + h0*M2 = 0
hn-1*Mn-2 - (hn-2+hn-1)*Mn-1 + hn-2*Mn = 0
```

**Properties**:
- No user-specified boundary data needed
- MATLAB's default (spline function)
- Uniquely determined, interpolatory, C2
- Generally better endpoint behavior than natural spline

### 3.6 Tridiagonal System Solution

All three spline types produce a tridiagonal system Ax = b where A is symmetric positive definite (for natural and clamped) or at least diagonally dominant.

**Thomas algorithm** (sequential O(n)):

Forward sweep:
```
c'_0 = c_0 / b_0
d'_0 = d_0 / b_0
for i = 1 to n-1:
    m = a_i / (b_i - a_i * c'_{i-1})
    c'_i = c_i / (b_i - a_i * c'_{i-1})
    d'_i = (d_i - a_i * d'_{i-1}) / (b_i - a_i * c'_{i-1})
```

Back substitution:
```
x_{n-1} = d'_{n-1}
for i = n-2 downto 0:
    x_i = d'_i - c'_i * x_{i+1}
```

Here a_i = lower diagonal, b_i = main diagonal, c_i = upper diagonal.

**Numerical stability**: Thomas algorithm is stable for diagonally dominant systems (which the spline tridiagonal always is when hi > 0). No pivoting needed.

**CRITICAL for GPU**: Thomas algorithm is inherently sequential (O(n) depth). This is the bottleneck for cubic spline on GPU.

### 3.7 GPU Strategy for Tridiagonal Systems

**Option 1: Cyclic Reduction (CR)**
- Eliminate even-indexed unknowns, reduce to half-size system, recurse
- O(n) work, O(log^2 n) depth
- Each reduction step is parallel across equations
- Accumulate decomposition: Prefix scan with 2x2 matrix combine (Affine operator variant)

**Option 2: Parallel Cyclic Reduction (PCR)**
- All equations updated simultaneously each step
- O(n log n) work, O(log n) depth
- More work than CR but lower depth (better for massively parallel GPU)
- Each step: accumulate(equations, All, matrix_combine, Affine)

**Option 3: CR-PCR Hybrid**
- CR to reduce size to ~warp_size, then PCR for the small system
- Best practical performance on GPU (see Zhang et al. 2010)
- cuSPARSE's gtsv uses this approach

**Option 4: Parallel prefix with 2x2 matrices**
- Reformulate tridiagonal solve as prefix scan with 2x2 matrix multiplication
- The recurrence xi = (di - ai*xi-1) / bi can be written as:
  ```
  [xi]     [ci  di] [xi-1]
  [1 ] =   [0   1 ] [1   ]
  ```
  where ci = -ai/bi, di = di/bi (after normalization)
- This is exactly accumulate(equations, Prefix(forward), identity, Affine(2x2))
- O(n) work, O(log n) depth
- **This is the tambear-native approach**: tridiagonal solve IS a prefix scan with Affine(2x2) operator

**Recommended**: Option 4 (prefix scan with 2x2 Affine). It maps directly to the existing accumulate framework. The 2x2 matrix multiplication is associative, fitting the Affine operator family.

### 3.8 Complete Coefficient Formulas

Once {Mi} are solved from the tridiagonal system, for each interval [xi, xi+1]:

```
ai = yi
bi = (yi+1 - yi)/hi - hi*(2*Mi + Mi+1)/6
ci = Mi/2
di = (Mi+1 - Mi)/(6*hi)
```

Evaluation at query point x in [xi, xi+1]:
```
t = x - xi
S(x) = ai + bi*t + ci*t^2 + di*t^3
```

Use Horner form for evaluation:
```
S(x) = ai + t*(bi + t*(ci + t*di))
```

Derivatives:
```
S'(x) = bi + t*(2*ci + 3*di*t)
S''(x) = 2*ci + 6*di*t
```

### 3.9 Edge Cases and Failure Modes

- **n = 1** (2 points): cubic spline degenerates. Natural spline = linear interpolation. Clamped spline = Hermite cubic with given slopes.
- **n = 2** (3 points): not-a-knot has only one interior point; the constraint forces a single cubic through all three points (= polynomial interpolation). Natural spline gives a proper 2-piece cubic.
- **Non-uniform spacing**: the tridiagonal system handles this naturally (hi appear in coefficients). Very non-uniform spacing can cause the spline to behave poorly in wide intervals.
- **Duplicate x values**: undefined (division by zero in hi). Detect and error.
- **Very large n**: tridiagonal solve is O(n) either way. The O(log n) depth GPU method makes this practical for n in the millions.
- **Overshoot**: cubic splines can overshoot between data points, even creating spurious oscillations for monotone data. This motivates monotone interpolation methods (Section 5, 6, 7).

### 3.10 Error Analysis

For a function f in C^4[a,b]:

**Natural spline**:
```
|f(x) - S(x)| <= (5/384) * h^4 * max|f''''(x)|
```
where h = max(hi).

**Clamped spline** (exact endpoint derivatives):
```
|f(x) - S(x)| <= (5/384) * h^4 * max|f''''(x)|
```

This is O(h^4) -- MUCH better than linear interpolation (O(h^2)) or high-degree polynomial (Runge problems). Cubic spline is fourth-order accurate, matching its degree+1.

---

## 4. B-Spline Basis

### 4.1 Knot Vectors

A **knot vector** is a non-decreasing sequence:
```
T = {t0, t1, ..., tm}    with t0 <= t1 <= ... <= tm
```

For a B-spline of degree p (order k = p+1), with n+1 control points, we need m = n + p + 1 knots.

**Types**:
- **Uniform**: equally spaced knots
- **Open/clamped**: first and last knots repeated p+1 times (curve passes through first/last control points)
- **Non-uniform**: arbitrary spacing

### 4.2 Cox-de Boor Recursion

The i-th B-spline basis function of degree p is defined recursively:

**Degree 0** (piecewise constant):
```
Ni,0(x) = 1   if ti <= x < ti+1
           0   otherwise
```

**Degree p** (recursive):
```
Ni,p(x) = ((x - ti) / (ti+p - ti)) * Ni,p-1(x) + ((ti+p+1 - x) / (ti+p+1 - ti+1)) * Ni+1,p-1(x)
```

**Convention for 0/0**: When ti+p = ti (repeated knots), define the fraction (x-ti)/(ti+p-ti) = 0. This handles the degenerate case where the knot span is zero.

### 4.3 Properties of B-Spline Basis Functions

1. **Non-negativity**: Ni,p(x) >= 0 for all x
2. **Partition of unity**: sum_i Ni,p(x) = 1 for x in [tp, tm-p]
3. **Local support**: Ni,p(x) = 0 outside [ti, ti+p+1] (support spans p+2 knots)
4. **Continuity**: At a knot of multiplicity r, Ni,p is C^(p-r) continuous. Simple knot (r=1): C^(p-1). Repeated p times: C^0.
5. **Linear independence**: The {Ni,p} are linearly independent over [tp, tm-p]

### 4.4 B-Spline Curve Evaluation

A B-spline curve:
```
S(x) = sum_{i=0}^{n} ci * Ni,p(x)
```

where ci are control points (coefficients).

**Evaluation via de Boor's algorithm** (stable, O(p^2) per point):

To evaluate S(x) where x is in knot span [tr, tr+1]:
```
d_i^0 = c_i     for i = r-p, ..., r
for k = 1 to p:
    for i = r downto r-p+k:
        alpha = (x - ti) / (ti+p-k+1 - ti)
        d_i^k = (1 - alpha) * d_{i-1}^{k-1} + alpha * d_i^{k-1}
S(x) = d_r^p
```

This is the B-spline analog of de Casteljau's algorithm for Bezier curves.

### 4.5 Relationship to Cubic Splines

A cubic spline (Section 3) IS a B-spline curve of degree 3 with specific control points. Given a cubic spline in coefficient form {ai, bi, ci, di}, the equivalent B-spline representation uses:
- Degree p = 3
- Open knot vector with knots at the data x-values (repeated 4 times at endpoints for clamped)
- Control points ci computed from the spline coefficients

The conversion is a linear transformation. The B-spline form is more general (allows non-interpolatory curves, NURBS, etc.) while the explicit coefficient form (Section 3) is more direct for interpolation.

### 4.6 Knot Insertion and Refinement

**Boehm's algorithm**: insert a new knot without changing the curve. This is a local operation affecting only p+1 control points.

**Oslo algorithm**: insert multiple knots simultaneously.

These are relevant for adaptive refinement on GPU: identify regions needing more resolution, insert knots there.

### 4.7 GPU Decomposition

**Basis function evaluation**: For each query point, find the knot span (binary search), then evaluate p+1 nonzero basis functions via Cox-de Boor. Each query is independent.

```
accumulate decomposition:
  Per query: gather(binary_search(x, T), control_points) + de_boor evaluation
  Grouping: None (each query independent)
  Parallelism: over query points
```

**Knot span search**: Same as linear interp -- binary search over knot vector.

**De Boor evaluation**: p+1 iterations, each with p+1-k operations. For cubic (p=3): 3 iterations, ~6 lerps. Fits in registers.

**Fitting B-splines to data** (least squares): requires solving the normal equations N'N c = N'y where N is the collocation matrix (Ni,p(xj)). This is a banded system (bandwidth ~ p+1). Can use banded Cholesky (F02 linear algebra) or the accumulate framework with Tiled grouping for the N'N product.

### 4.8 Edge Cases

- **Coincident knots**: multiplicity reduces continuity. Maximum multiplicity p+1 at a knot = C^(-1) = discontinuity (used for Bezier segments).
- **Extrapolation**: B-spline basis is zero outside [tp, tm-p]. Evaluation outside this range requires extending the knot vector.
- **Very high degree**: basis function evaluation is O(p^2). For p > ~10, consider using explicit matrix form.
- **Empty knot spans** (ti = ti+1): valid, just means Ni,0 has zero support. The convention 0/0 = 0 handles this.

---

## 5. Monotone Interpolation (Fritsch-Carlson 1980)

### 5.1 The Problem

Cubic spline interpolation can produce oscillations even when data is monotone. Example: data (0,0), (1,0), (2,0), (3,1), (4,1), (5,1) -- a step function. Cubic spline will overshoot above 1 and undershoot below 0 near the step.

Monotone interpolation GUARANTEES: if yi <= yi+1 for all i, then S(x) <= S(x') for all x <= x'.

### 5.2 Hermite Basis Functions

The cubic Hermite interpolant on [xi, xi+1] with values yi, yi+1 and slopes fi', fi+1' is:

```
p(x) = h00(t)*yi + h10(t)*hi*fi' + h01(t)*yi+1 + h11(t)*hi*fi+1'
```

where t = (x - xi)/hi, hi = xi+1 - xi, and:
```
h00(t) = 2t^3 - 3t^2 + 1       (value at left)
h10(t) = t^3 - 2t^2 + t         (slope at left)
h01(t) = -2t^3 + 3t^2           (value at right)
h11(t) = t^3 - t^2              (slope at right)
```

These are the standard Hermite basis functions satisfying:
- h00(0)=1, h00(1)=0, h00'(0)=0, h00'(1)=0
- h10(0)=0, h10(1)=0, h10'(0)=1, h10'(1)=0
- h01(0)=0, h01(1)=1, h01'(0)=0, h01'(1)=0
- h11(0)=0, h11(1)=0, h11'(0)=0, h11'(1)=1

The interpolant is C1 (continuous first derivative) but NOT necessarily C2.

### 5.3 Fritsch-Carlson Slope Modification Algorithm

**Step 1**: Compute secant slopes:
```
delta_i = (yi+1 - yi) / hi
```

**Step 2**: Initialize slopes at data points:
```
fi' = (delta_{i-1} + delta_i) / 2    for interior points
f0' = delta_0                         for first point
fn' = delta_{n-1}                     for last point
```

(Alternative: use three-point difference formula for endpoints.)

**Step 3**: Check monotonicity conditions. For each interval [xi, xi+1]:

If delta_i = 0 (flat interval): set fi' = fi+1' = 0 (FORCED -- otherwise the cubic overshoots).

If delta_i != 0, let:
```
alpha_i = fi' / delta_i
beta_i = fi+1' / delta_i
```

**Necessary condition for monotonicity** (Fritsch-Carlson):
```
alpha_i^2 + beta_i^2 <= 9
```

If this is violated, scale down the slopes:
```
tau = 3 / sqrt(alpha_i^2 + beta_i^2)
fi' = tau * alpha_i * delta_i
fi+1' = tau * beta_i * delta_i
```

**Step 4**: The interpolant is now guaranteed monotone on each interval.

### 5.4 The Full Fritsch-Carlson Sufficient Condition

The region of monotone-preserving (alpha, beta) pairs is:
```
phi(alpha, beta) = alpha - (2*alpha + beta - 3)^2 / (3*(alpha + beta - 2)) >= 0
```

when alpha + beta - 2 > 0, and always monotone when alpha + beta - 2 <= 0. The circle alpha^2 + beta^2 <= 9 is a conservative (but simple) approximation inscribed in this region.

### 5.5 GPU Decomposition

**Fully local** -- each interval's monotonicity check depends only on that interval and its neighbors.

```
accumulate decomposition:
  Step 1 (secants): fused_expr on consecutive pairs. Parallelism: over intervals.
  Step 2 (initial slopes): Windowed(3) averaging. Parallelism: over points.
  Step 3 (correction): fused_expr on (fi', fi+1', delta_i). Parallelism: over intervals.
  Step 4 (evaluation): per query, same as Hermite (Section 5.2). Parallelism: over queries.
```

The entire setup is embarrassingly parallel (no tridiagonal solve!). This is why Fritsch-Carlson is GPU-friendly.

### 5.6 Edge Cases

- **Non-monotone data**: method still works but does NOT make the interpolant monotone -- it only preserves existing monotonicity. For non-monotone data, it preserves local monotonicity within each monotone piece.
- **Flat regions (delta_i = 0)**: slopes forced to zero. Adjacent non-flat intervals may have their slopes reduced.
- **Single flat point amid monotone data**: creates a "flat spot" in the interpolant (C1 but with zero derivative).
- **Very steep transitions**: slopes can be large; no overflow protection beyond what the data dictates.

---

## 6. Akima Interpolation (1970)

### 6.1 Philosophy

Akima's method is a LOCAL piecewise cubic method that avoids the wiggles of global cubic splines. The key idea: the slope at each data point is determined by a WEIGHTED AVERAGE of neighboring secants, where the weights are chosen to make the interpolant more "visually pleasing" (Akima's term).

### 6.2 Slope Computation

**Step 1**: Compute secant slopes for all intervals:
```
mi = (yi+1 - yi) / (xi+1 - xi)   for i = 0, ..., n-1
```

**Step 2**: Extend the secant sequence. For the first and last two points, we need secants outside the data range. Akima's original extension:
```
m_{-2} = 3*m_0 - 2*m_1      (or: 2*m_0 - m_1)
m_{-1} = 2*m_0 - m_1
m_n    = 2*m_{n-1} - m_{n-2}
m_{n+1} = 3*m_{n-1} - 2*m_{n-2}   (or: 2*m_{n-1} - m_{n-2})
```

**CROSS-PACKAGE WARNING**: Different implementations use different endpoint extensions. scipy uses one formula; MATLAB's `makima` uses a modified version. The choice affects the first and last two intervals.

**Step 3**: Compute weights at each data point i:
```
w1 = |m_{i+1} - m_i|
w2 = |m_{i-1} - m_{i-2}|
```

**Step 4**: Compute slope at xi:
```
if w1 + w2 != 0:
    ti = (w1 * m_{i-1} + w2 * m_i) / (w1 + w2)
else:
    ti = (m_{i-1} + m_i) / 2
```

The intuition: the slope is a weighted average of the two adjacent secants, where the weight given to each secant is determined by how much the secants on the OTHER side change. If the secants are changing rapidly on the right (|m_{i+1} - m_i| is large), more weight is given to the left secant (m_{i-1}), and vice versa.

**Step 5**: Use the Hermite basis (Section 5.2) with these slopes.

### 6.3 Modified Akima (makima)

MATLAB's `makima` (introduced ~R2019b) modifies the weight formula to prevent overshoot:
```
w1 = |m_{i+1} - m_i| + |m_{i+1} + m_i| / 2
w2 = |m_{i-1} - m_{i-2}| + |m_{i-1} + m_{i-2}| / 2
```

The extra term biases the weights to prevent the interpolant from overshooting when the function is locally flat.

### 6.4 Properties

- **Locality**: changing one data point affects at most 6 intervals (3 on each side)
- **C1 continuous**: first derivative continuous, second derivative generally not
- **No system solve**: purely local computation
- **No overshoot for linear data**: if the data lies on a line, the interpolant reproduces it exactly
- **Handles outliers well**: the weight formula naturally reduces the influence of outlier points

### 6.5 GPU Decomposition

**Fully local** -- even more parallel-friendly than cubic spline.

```
accumulate decomposition:
  Step 1 (secants): fused_expr. Parallel over intervals.
  Step 2 (endpoint extension): fused_expr on boundary secants. 4 values.
  Step 3-4 (weights and slopes): Windowed(5) stencil over secant array.
    Each slope depends on m_{i-2}, m_{i-1}, m_i, m_{i+1} -- a 4-element stencil.
    Parallelism: over data points.
  Step 5 (evaluation): per query, Hermite. Parallel over queries.
```

No global solve. No tridiagonal system. This is the most GPU-friendly cubic-quality method.

### 6.6 Edge Cases

- **All weights zero** (w1 = w2 = 0): happens when m_{i-2} = m_{i-1} = m_i = m_{i+1} (all four secants equal). The data is locally linear, and the arithmetic mean is correct.
- **n < 4** (fewer than 5 points): the stencil doesn't fit. Fall back to simpler methods (cubic polynomial through 3-4 points, or linear).
- **Non-equispaced data**: works fine (secants account for variable spacing). No assumption of uniform spacing.
- **Monotone data**: Akima does NOT guarantee monotonicity (unlike Fritsch-Carlson). The modified Akima (makima) reduces but does not eliminate overshoot.

---

## 7. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)

### 7.1 Definition

PCHIP is MATLAB's default "shape-preserving" interpolant (`pchip` function). It produces a C1 piecewise cubic that preserves the monotonicity of the data within each interval.

PCHIP is essentially a specific implementation of the Fritsch-Carlson framework (Section 5) with a particular choice of initial slopes.

### 7.2 Slope Computation (MATLAB's Algorithm)

**Step 1**: Compute secant slopes:
```
delta_i = (yi+1 - yi) / hi    where hi = xi+1 - xi
```

**Step 2**: For interior points (i = 1, ..., n-1):

If delta_{i-1} and delta_i have the **same sign** (or either is zero):
```
w1 = 2*hi + hi-1        (weight for delta_{i-1})
w2 = hi + 2*hi-1        (weight for delta_i)
fi' = (w1 + w2) / (w1/delta_{i-1} + w2/delta_i)
```

This is a weighted **harmonic mean** of the two adjacent secants.

If delta_{i-1} and delta_i have **different signs** (local extremum):
```
fi' = 0
```

**Step 3**: Endpoint slopes. MATLAB uses a one-sided three-point formula:
```
f0' = ((2*h0 + h1)*delta_0 - h0*delta_1) / (h0 + h1)
```
Then if sign(f0') != sign(delta_0): f0' = 0.
Then if sign(delta_0) != sign(delta_1) and |f0'| > |3*delta_0|: f0' = 3*delta_0.

(Same logic mirrored for fn'.)

**Step 4**: Use Hermite basis (Section 5.2) with these slopes.

### 7.3 Why Harmonic Mean?

The harmonic mean of two positive numbers is always <= their arithmetic mean, and is zero if either input is zero. This naturally provides:
- Smaller slopes than arithmetic averaging (less overshoot)
- Automatic zero slope when either secant is zero (preserves flat segments)
- Proper handling of non-uniform spacing (the weights involve hi)

### 7.4 Properties

- **Shape-preserving**: monotonicity in data is preserved in interpolant
- **C1 continuous**: continuous first derivative, discontinuous second derivative
- **Local**: each slope depends on at most 2 neighboring intervals
- **No linear system**: purely local computation, like Akima
- **Undershoots rather than overshoots**: the conservative slope choice means PCHIP may "flatten" where cubic spline would give a tighter fit

### 7.5 Comparison with Other Methods

| Property | Cubic Spline | Akima | PCHIP/Fritsch-Carlson |
|----------|-------------|-------|----------------------|
| Continuity | C2 | C1 | C1 |
| Monotone-preserving | NO | NO (mostly) | YES |
| Local | NO (tridiagonal) | YES | YES |
| Oscillation | Can oscillate | Rarely oscillates | Never oscillates (monotone data) |
| Accuracy on smooth data | Best (O(h^4)) | Good (O(h^3)) | Good (O(h^3)) |
| GPU parallel setup | Needs prefix scan | Embarrassingly parallel | Embarrassingly parallel |

### 7.6 GPU Decomposition

Identical to Fritsch-Carlson (Section 5.5). The only difference is the slope formula in Step 2.

```
accumulate decomposition:
  Secants: parallel over intervals (fused_expr)
  Slopes: parallel over data points (Windowed(3) -- harmonic mean of neighbors)
  Evaluation: parallel over query points (Hermite)
  No global solve. No tridiagonal system.
```

### 7.7 Edge Cases

- **n = 1** (2 points): single interval, slopes set to the secant. Reduces to linear interpolation.
- **n = 2** (3 points): one interior point. Slope computed from harmonic mean of two secants. Endpoint slopes from one-sided formula.
- **Constant data**: all secants zero, all slopes zero, interpolant is constant. Correct.
- **Very steep adjacent intervals**: slopes may be much smaller than expected (harmonic mean pulls toward zero). This is by design -- shape preservation.
- **Non-uniform spacing**: fully supported (weights depend on hi).

---

## 8. Radial Basis Function (RBF) Interpolation

### 8.1 Definition

Given scattered data points {xi, yi} where xi may be in R^d (multivariate!), the RBF interpolant is:

```
f(x) = sum_{i=1}^{N} lambda_i * phi(||x - xi||) + p(x)
```

where:
- phi(r) is a radial basis function (depends only on distance)
- lambda_i are coefficients to determine
- p(x) is a polynomial of degree <= m (for conditionally positive definite RBFs)
- || . || is typically the Euclidean norm

### 8.2 Common RBFs

| Name | phi(r) | Parameter | Conditionally PD order |
|------|--------|-----------|----------------------|
| Gaussian | exp(-epsilon^2 * r^2) | epsilon (shape) | 0 (strictly PD) |
| Multiquadric | sqrt(1 + epsilon^2 * r^2) | epsilon | 1 |
| Inverse multiquadric | 1/sqrt(1 + epsilon^2 * r^2) | epsilon | 0 (strictly PD) |
| Thin plate spline | r^2 * ln(r) | none | 2 |
| Polyharmonic r^k (k odd) | r^k | k | ceil(k/2) |
| Polyharmonic r^k*ln(r) (k even) | r^(2k) * ln(r) | k | k+1 |
| Wendland (compactly supported) | (1-r)_+^l * p(r) | support radius | 0 |

**Shape parameter epsilon**: controls the "flatness" of the basis function. Small epsilon = flat (global influence), large epsilon = peaked (local influence).

### 8.3 Linear System

The interpolation conditions f(xi) = yi plus orthogonality conditions give:

```
[A  P] [lambda]   [y]
[P' 0] [c     ] = [0]
```

where:
- A is N x N with Aij = phi(||xi - xj||) (the kernel matrix)
- P is N x M with Pij = pj(xi) (polynomial basis evaluated at data points)
- M = dim(polynomial space of degree <= m)
- lambda is N x 1 (RBF coefficients)
- c is M x 1 (polynomial coefficients)

For strictly positive definite RBFs (Gaussian, inverse multiquadric): A is positive definite, P = 0 is valid (no polynomial term needed), and the system is just A * lambda = y.

For conditionally positive definite RBFs of order m (thin plate spline, multiquadric): the polynomial term of degree <= m-1 is REQUIRED for the system to be nonsingular.

### 8.4 Conditioning Issues

**The epsilon dilemma**: Small epsilon (flat RBFs) give more accurate interpolation but TERRIBLE conditioning. This is the "trade-off principle" (Schaback 1995):

```
interpolation error ~ O(exp(-c/epsilon))    (decreases as epsilon -> 0)
condition number ~ O(exp(c'/epsilon^2))      (increases as epsilon -> 0)
```

**Practical consequence**: Direct solution of the linear system breaks down for small epsilon in f32 and even f64 for modest N.

**Mitigations**:
1. **RBF-QR** (Fornberg et al. 2011): stable computation in a different basis. O(N^2) storage.
2. **RBF-GA** (Fornberg et al. 2013): stable Gaussian elimination variant.
3. **Contour-Pade**: analytic continuation method.
4. **Just use larger epsilon**: sacrifice accuracy for stability.
5. **Regularization**: Tikhonov (add lambda*I to A). Changes interpolation to approximation.

### 8.5 Evaluation

Once lambda (and c) are determined:

```
f(x) = sum_{i=1}^{N} lambda_i * phi(||x - xi||) + sum_{j=1}^{M} cj * pj(x)
```

For each query point: compute N distances, N RBF evaluations, one dot product. Cost: O(N*d) per query.

### 8.6 GPU Decomposition

**System construction** (A matrix):
```
accumulate decomposition:
  Tiled(N, N) with expr = phi(||xi - xj||)
  This is exactly the distance matrix / kernel matrix pattern from F01/F20.
  Same as KMeans distance computation.
```

**System solve**:
```
Dense N x N system. Requires F02 linear algebra (Cholesky for PD, LU for general).
Sequential depth O(N) for Cholesky, O(N^2) work.
For large N: use preconditioned iterative solver (CG for PD systems).
```

**Evaluation**:
```
Per query: accumulate(data_points, All, lambda_i * phi(||x - xi||), Add)
This is a reduce (dot product of lambda with phi-vector).
Parallelism: over query points (embarrassingly parallel).
Within one query: the sum over N is a standard reduce.
```

**Scaling concern**: The dense N x N system makes RBF interpolation O(N^3) for setup and O(N*M) for M query evaluations. For large N (>10,000), use:
- Compactly supported RBFs (sparse A matrix)
- Fast multipole method (FMM) -- O(N) evaluation
- Partition of unity methods (local RBF patches)
- Hierarchical/treecode methods

### 8.7 Multivariate Generalization

RBF interpolation works in ANY dimension -- the formula is identical. This is its primary advantage over tensor-product methods (which scale exponentially with dimension).

For d-dimensional data: ||x - xi|| is the d-dimensional Euclidean distance. The kernel matrix A is the same structure. The polynomial space has M = C(d+m, d) terms.

### 8.8 Edge Cases and Failure Modes

- **Duplicate points**: A becomes singular (two identical rows). Detect and merge.
- **Nearly duplicate points**: A is nearly singular. Condition number explodes.
- **Too few points for polynomial term**: Need N >= M for the system to be solvable. For thin plate spline in 2D (m=2): M = 3 (constant + linear), so need N >= 3.
- **Poorly distributed points** (all in a line in 2D, for example): polynomial system P'*lambda = 0 may not have full rank.
- **Very large N**: O(N^3) setup is prohibitive. Must use fast methods.
- **Negative interpolation**: RBF interpolation can produce negative values even when all yi >= 0. No shape-preservation guarantee.

---

## 9. Least Squares Approximation

### 9.1 Distinction from Interpolation

Interpolation: f(xi) = yi exactly. Requires N equations for N unknowns.
Approximation: minimize ||f(xi) - yi||^2 over all i. Uses M << N parameters.

The data is OVER-determined: more equations than unknowns.

### 9.2 Polynomial Least Squares

Fit a polynomial of degree p to N data points (N > p+1):

```
f(x) = a0 + a1*x + a2*x^2 + ... + ap*x^p
```

The normal equations (V'V)a = V'y where V is the Vandermonde matrix:
```
Vij = xi^j    (N x (p+1) matrix)
```

**NEVER form V'V directly.** The condition number of V'V is the SQUARE of V's condition number. For the Vandermonde matrix with typical data ranges, cond(V) ~ 10^p, so cond(V'V) ~ 10^(2p). This is catastrophic for p > 5 in f64, p > 3 in f32.

### 9.3 Solution Methods (Ranked by Stability)

**Method 1: Normal equations** (WORST stability, fastest)
```
(V'V)a = V'y
Solve via Cholesky: O(Np^2 + p^3)
```
Condition: cond(V'V) = cond(V)^2. Fails for ill-conditioned V.

**Method 2: QR factorization** (GOOD stability, moderate speed)
```
V = QR  (thin QR: Q is N x (p+1), R is (p+1) x (p+1))
Ra = Q'y
Back-substitution: O(Np^2) for QR, O(p^2) for solve.
```
Condition: cond(R) = cond(V). Much better.

**Method 3: SVD** (BEST stability, slowest)
```
V = U * Sigma * V'  (thin SVD)
a = V * Sigma^{-1} * U' * y
```
Cost: O(Np^2) for SVD. Can handle rank-deficient V (set small singular values to zero).

**Recommendation**: QR for routine use. SVD when the system is suspected ill-conditioned or rank-deficient.

### 9.4 Orthogonal Polynomial Basis

Instead of monomials {1, x, x^2, ...}, use orthogonal polynomials {P0(x), P1(x), ...} with respect to the data points. Then V'V is diagonal (or nearly so), and the normal equations decouple.

**Practical options**:
- Chebyshev polynomials (Section 10) on [-1,1]
- Legendre polynomials
- Discrete orthogonal polynomials (Gram polynomials, constructed from the data)

Map data to [-1,1] via affine transform, then use Chebyshev basis. Condition number drops from O(10^p) to O(1).

### 9.5 Regularized Least Squares

**Tikhonov / Ridge Regression (L2 penalty)**:
```
minimize ||Va - y||^2 + lambda * ||a||^2
Solution: (V'V + lambda*I)a = V'y
```

The regularization parameter lambda > 0 improves conditioning: cond(V'V + lambda*I) <= cond(V'V) * max_sigma^2 / (min_sigma^2 + lambda). For lambda >> sigma_min(V)^2, the solution is biased toward zero but stable.

**LASSO (L1 penalty)**:
```
minimize ||Va - y||^2 + lambda * ||a||_1
```
No closed-form solution. Requires iterative methods:
- Coordinate descent (most common)
- ISTA/FISTA (proximal gradient)
- ADMM

L1 promotes sparsity: many coefficients become exactly zero. Useful for feature selection.

**Elastic Net (L1 + L2)**:
```
minimize ||Va - y||^2 + lambda_1 * ||a||_1 + lambda_2 * ||a||^2
```
Combines sparsity of LASSO with grouping effect of ridge.

### 9.6 Connection to F10 Regression

Family 10 (Regression) uses exactly these building blocks:
- OLS = least squares with monomial/feature basis (QR or SVD)
- Ridge = Tikhonov with L2
- LASSO = L1 penalty with coordinate descent
- Elastic net = mixed penalty

F31 provides the mathematical core; F10 provides the statistical framework (diagnostics, inference, etc.).

### 9.7 Weighted Least Squares

```
minimize sum wi * (f(xi) - yi)^2
```

Equivalent to: minimize ||W^{1/2}(Va - y)||^2 where W = diag(wi).

Normal equations: (V'WV)a = V'Wy. Or: transform V -> W^{1/2}V, y -> W^{1/2}y, then use standard methods.

### 9.8 GPU Decomposition

**V'V computation (normal equations path)**:
```
accumulate(data, Tiled(p+1, p+1), xi^j * xi^k, Add)
```
This is a GEMM: V'V where V is N x (p+1). Standard Tiled accumulate.

**V'y computation**:
```
accumulate(data, All, xi^j * yi, Add)   for each j
```
Or: a single matrix-vector product (GEMV).

**QR factorization**: Householder reflections are sequential in the column dimension but parallel within each reflection. O(p) stages of O(N) parallel work.

**SVD**: Golub-Kahan bidiagonalization followed by QR iteration. Complex GPU kernel, but well-studied (MAGMA, cuSOLVER).

**Coordinate descent (LASSO)**:
```
Each coordinate update: accumulate(residuals, All, xi^j * residual, Add) + soft threshold
Sequential over coordinates, parallel within each coordinate update.
O(p) stages of O(N) parallel work per iteration.
```

**Overall**: Setup is Tiled (GEMM) or column-sequential (QR). Evaluation is embarrassingly parallel.

### 9.9 Edge Cases

- **Rank deficiency**: When p >= N-1, the system is underdetermined. SVD gives the minimum-norm solution.
- **Collinear features**: V'V singular. Ridge regression or SVD needed.
- **Outliers**: Standard least squares is not robust. Use robust regression (F09) or iteratively reweighted least squares (IRLS).
- **lambda selection**: Cross-validation (leave-one-out has O(N) closed form for ridge), GCV, AIC, BIC.

---

## 10. Chebyshev Approximation

### 10.1 Chebyshev Polynomials

**Definition** (first kind):
```
Tn(x) = cos(n * arccos(x))    for x in [-1,1]
```

**Equivalent recursive definition**:
```
T0(x) = 1
T1(x) = x
Tn+1(x) = 2x * Tn(x) - Tn-1(x)
```

First few:
```
T0 = 1
T1 = x
T2 = 2x^2 - 1
T3 = 4x^3 - 3x
T4 = 8x^4 - 8x^2 + 1
T5 = 16x^5 - 20x^3 + 5x
```

**Key properties**:
1. **Orthogonality**: integral_{-1}^{1} Tm(x)*Tn(x)/sqrt(1-x^2) dx = 0 for m != n
2. **Bounded**: |Tn(x)| <= 1 for x in [-1, 1]
3. **Extrema**: Tn(cos(k*pi/n)) = (-1)^k for k = 0,...,n
4. **Roots (Chebyshev nodes)**: Tn(cos((2k+1)*pi/(2n))) = 0 for k = 0,...,n-1
5. **Leading coefficient**: 2^{n-1} for n >= 1

### 10.2 Minimax Property

Among all monic polynomials of degree n, Tn(x)/2^{n-1} has the smallest maximum deviation from zero on [-1,1]:

```
min_{p monic, deg n} max_{x in [-1,1]} |p(x)| = 1/2^{n-1}
```

achieved uniquely by Tn(x)/2^{n-1}.

**Consequence**: Chebyshev interpolation minimizes the maximum interpolation error (in the polynomial approximation sense). This is the **equioscillation theorem** (Chebyshev 1854).

### 10.3 Chebyshev Expansion

Any continuous function f on [-1,1] can be expanded:
```
f(x) = (c0/2) + sum_{k=1}^{infinity} ck * Tk(x)
```

where:
```
ck = (2/pi) * integral_{-1}^{1} f(x) * Tk(x) / sqrt(1-x^2) dx
```

**Truncation to degree n**: f(x) ~ (c0/2) + sum_{k=1}^{n} ck * Tk(x).

The error of truncation depends on how fast ck -> 0:
- Analytic f: |ck| ~ O(rho^{-k}) for some rho > 1 (exponential convergence)
- f in C^m: |ck| ~ O(k^{-m}) (algebraic convergence)

### 10.4 Discrete Chebyshev Transform (DCT Connection)

**Computing Chebyshev coefficients from samples**:

Sample f at the Chebyshev nodes xj = cos(j*pi/n) for j = 0,...,n:
```
ck = (2/n) * sum_{j=0}^{n} (double-prime) f(xj) * cos(k*j*pi/n)
```

where double-prime means the first and last terms are halved.

This is a **Type I DCT** (DCT-I). Can be computed via FFT in O(n log n).

**Conversely**: Given Chebyshev coefficients, evaluate at Chebyshev nodes via inverse DCT (also O(n log n)).

**Practical algorithm**:
1. Sample f at n+1 Chebyshev nodes: O(n) evaluations of f
2. Apply DCT-I to get coefficients: O(n log n)
3. Truncate small coefficients (adaptive degree selection)

### 10.5 Clenshaw's Recurrence (Evaluation)

Given coefficients c0,...,cn, evaluate S(x) = (c0/2) + sum_{k=1}^{n} ck * Tk(x) using the three-term recurrence backward:

```
d_{n+2} = 0
d_{n+1} = 0
for k = n downto 1:
    dk = 2*x*d_{k+1} - d_{k+2} + ck
S(x) = x*d1 - d2 + c0/2
```

This is **numerically stable** and costs O(n) per evaluation (2 multiplies, 2 adds per step).

**Why not Horner?** Converting Chebyshev to monomial form loses the conditioning advantages. Clenshaw evaluates DIRECTLY in the Chebyshev basis without conversion.

### 10.6 Adaptive Approximation

The power of Chebyshev approximation for implementation:
1. Start with moderate n (say 16)
2. Compute coefficients
3. If |cn| > tolerance, double n and recompute
4. Stop when trailing coefficients are below tolerance
5. Truncate: the degree is AUTOMATICALLY chosen

This gives near-minimax approximation with automatic degree selection. Libraries like Chebfun (MATLAB) use this to represent functions to machine precision.

### 10.7 GPU Decomposition

**Coefficient computation** (via DCT):
```
Step 1: Evaluate f at Chebyshev nodes. Parallel over nodes.
Step 2: DCT-I. This is an FFT variant.
  accumulate decomposition: F03 (Signal Processing) FFT primitive.
  O(n log n) work, O(log n) depth.
```

**Evaluation via Clenshaw**:
```
Per query point: sequential scan of length n (backward through coefficients).
  accumulate(coefficients, Prefix(reverse), clenshaw_step, Affine(2x2))

The recurrence dk = 2*x*d_{k+1} - d_{k+2} + ck can be written as:
  [dk  ]   [2x  -1] [d_{k+1}]   [ck]
  [dk+1] = [1    0] [d_{k+2}] + [0 ]

This is an Affine(2x2) scan! The state is (dk, dk+1) and the combine involves
a 2x2 matrix multiply plus a shift.

But: this runs BACKWARDS through the coefficients, so it's Prefix(reverse).
For a single query point: n sequential steps.
For M query points: parallelize over queries, each runs the scan internally.
```

**Alternative**: For many queries evaluated at different x but same coefficients, the Clenshaw recurrence for each x is independent. Parallelize over queries, sequential within each (n steps per query). For n <= ~100, this fits in registers.

### 10.8 Mapping to Arbitrary Intervals

To approximate f on [a,b], use the affine map:
```
x_cheb = (2*x - (a+b)) / (b-a)    maps [a,b] -> [-1,1]
```

Coefficients are computed on the mapped domain. Evaluation maps the query point first.

### 10.9 Edge Cases and Failure Modes

- **Non-smooth functions**: Chebyshev convergence is algebraic (slow) for functions with discontinuities or kinks. For discontinuous f, Gibbs phenomenon appears (same as Fourier).
- **Functions with singularities near [-1,1] in the complex plane**: convergence rate determined by the largest Bernstein ellipse in which f is analytic.
- **Very high degree**: roundoff error accumulates in DCT. For n > ~10^6, use compensated summation.
- **Outside [-1,1]**: Chebyshev polynomials grow exponentially. Extrapolation is UNSTABLE (faster divergence than monomial basis).

---

## 11. Pade Approximation

### 11.1 Definition

The [L/M] Pade approximant of a function f(x) is the rational function:

```
R_{L,M}(x) = P_L(x) / Q_M(x)
```

where P_L is a polynomial of degree <= L, Q_M is a polynomial of degree <= M, Q_M(0) = 1, and:

```
f(x) - P_L(x)/Q_M(x) = O(x^{L+M+1})
```

That is, the Taylor expansion of R_{L,M} matches f through order L+M.

### 11.2 Construction from Taylor Coefficients

Given f(x) = sum_{k=0}^{infinity} fk * x^k, write:

```
P_L(x) = p0 + p1*x + ... + pL*x^L
Q_M(x) = 1 + q1*x + ... + qM*x^M
```

The condition f(x)*Q_M(x) - P_L(x) = O(x^{L+M+1}) gives a system of equations.

**For the denominator coefficients** (solve this first):
```
f_{L+1}   + f_L*q1     + ... + f_{L-M+1}*qM = 0
f_{L+2}   + f_{L+1}*q1 + ... + f_{L-M+2}*qM = 0
...
f_{L+M}   + f_{L+M-1}*q1 + ... + f_L*qM     = 0
```

This is an M x M linear system (Toeplitz-like) for {q1,...,qM}.

**For the numerator coefficients** (direct once q's are known):
```
pk = fk + f_{k-1}*q1 + ... + f_{k-M}*qM    (with fi = 0 for i < 0)
for k = 0, 1, ..., L
```

### 11.3 Advantages Over Taylor Series

**Near poles**: Taylor series diverges. Pade approximants can represent poles exactly (as zeros of Q_M). Example:

```
f(x) = 1/(1-x):    Taylor truncated at n diverges for |x| > 1
                    [0/1] Pade = 1/(1-x) -- EXACT for all x != 1
```

**Analytic continuation**: Pade approximants often converge in regions where Taylor series diverges. The Pade approximant can "see around corners" in the complex plane.

**Exponential function**: [n/n] Pade for exp(x) converges uniformly on compact sets, with error O(x^{2n+1}/(2n+1)!). Much better than degree-2n Taylor for large |x|.

### 11.4 Common Pade Approximants

**exp(x)**: [n/n] is optimal for stability (A-stable in ODE context).
```
[1/1]: (1 + x/2) / (1 - x/2)     (Cayley transform / trapezoidal rule)
[2/2]: (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12)
```

**ln(1+x)**: Pade is excellent (Taylor converges only for |x| < 1).

**tan(x)**: [n/n+1] or [n+1/n] depending on desired pole structure.

### 11.5 The Pade Table

All [L/M] approximants for L,M = 0,1,2,... form the Pade table. Movement along the diagonal (L=M) is called "diagonal Pade" and often gives the best convergence.

**Block structure**: Some entries in the Pade table may be degenerate (the linear system for q is singular). This happens when f has special structure (e.g., f is a rational function of degree < L/M).

**Convergence theorems**:
- **de Montessus de Ballore** (1902): If f is meromorphic with exactly M poles in a disk, the [L/M] Pade approximants (L -> infinity, M fixed) converge uniformly on compact subsets avoiding the poles.
- **Nuttall-Pommerenke**: Almost all rows/diagonals of the Pade table converge in capacity.

### 11.6 Numerical Construction

**Direct method**: Solve the M x M system for {qk}. Cost: O(M^3) for general solver, O(M^2) for Toeplitz structure.

**Epsilon algorithm** (Wynn 1956): Computes diagonal Pade approximants from partial sums of the Taylor series. No explicit system solve.

```
e_{-1}^{(k)} = 0,  e_0^{(k)} = S_k (partial sums)
e_{n+1}^{(k)} = e_{n-1}^{(k+1)} + 1/(e_n^{(k+1)} - e_n^{(k)})
```

The diagonal Pade [n/n] equals e_{2n}^{(0)}.

**qd algorithm** (Rutishauser): Another O(n^2) method based on quotient-difference scheme.

### 11.7 GPU Decomposition

**Taylor coefficient computation**: Depends on how f is defined. If from a recurrence or differential equation, may be sequential.

**Denominator system solve**: M x M dense system. For small M (typical: M <= 20), this fits in shared memory or even registers on GPU. Parallelize over different [L/M] choices if computing multiple approximants.

**Numerator coefficients**: Once q's are known, pk are a convolution/cumulative sum. This is:
```
accumulate(taylor_coeffs, Prefix(forward), convolution_with_q, Add)
```

**Evaluation of rational function**: For each query point x:
```
P_L(x): Horner evaluation (O(L) sequential)
Q_M(x): Horner evaluation (O(M) sequential)
R = P/Q: one division
```
Embarrassingly parallel over query points.

**Epsilon algorithm**: The table has inherent sequential dependencies along the diagonals, but anti-diagonals can be computed in parallel. O(n) stages of O(n) parallel work.

### 11.8 Edge Cases and Failure Modes

- **Deficiency**: The M x M system may be singular -- the [L/M] Pade approximant does not exist. Move to neighboring entry in the Pade table.
- **Spurious poles** (Froissart doublets): Pade approximants can develop pole-zero pairs near each other that "cancel" but cause numerical grief. These appear as spikes in the approximant.
- **Coefficients computed from noisy data**: Pade is EXTREMELY sensitive to coefficient perturbation. Small noise in Taylor coefficients produces completely wrong poles.
- **Division by zero in evaluation**: Q_M(x) = 0 at a pole of the approximant. Return Inf or NaN.
- **Very high order**: [L/M] with L+M > ~30 in f64 often has conditioning issues in the coefficient system.

---

## 12. Multivariate Interpolation

### 12.1 Bilinear Interpolation (2D, Regular Grid)

Given values on a regular grid {(xi, yj, zij)} for i=0,...,nx, j=0,...,ny, with uniform or non-uniform spacing:

For query point (x,y) in the cell [xi, xi+1] x [yj, yj+1]:

```
t = (x - xi) / (xi+1 - xi)
u = (y - yj) / (yj+1 - yj)

f(x,y) = (1-t)*(1-u)*z_{i,j} + t*(1-u)*z_{i+1,j} + (1-t)*u*z_{i,j+1} + t*u*z_{i+1,j+1}
```

This is the **tensor product** of two linear interpolations: first interpolate in x, then in y (or vice versa -- the result is the same).

**Properties**:
- Exact for bilinear functions f(x,y) = a + bx + cy + dxy
- C0 across cell boundaries (continuous but not smooth)
- Isoparametric: maps the unit square to a quadrilateral
- Does NOT reproduce linear functions in general (the xy cross-term introduces distortion for non-rectangular cells)

### 12.2 Bicubic Interpolation (2D, Regular Grid)

The tensor product of two cubic interpolations. For each cell, uses a 4x4 neighborhood of grid points.

**Hermite bicubic**: requires values, partial derivatives df/dx, df/dy, and cross-derivative d^2f/dxdy at the four corners of each cell (16 pieces of information per cell for 16 coefficients).

```
f(x,y) = sum_{i=0}^{3} sum_{j=0}^{3} aij * t^i * u^j
```

The 16 coefficients {aij} are determined by the 16 constraints (4 values + 4 fx + 4 fy + 4 fxy at the corners).

**In matrix form**:
```
[a] = B' * [values and derivatives] * B
```

where B is the 4x4 Hermite basis matrix:
```
B = [ 1  0  0  0]
    [ 0  0  1  0]
    [-3  3 -2 -1]
    [ 2 -2  1  1]
```

**When derivatives are not given**: estimate them from the grid data using finite differences:
```
fx_{i,j} ~ (z_{i+1,j} - z_{i-1,j}) / (2*hx)
fy_{i,j} ~ (z_{i,j+1} - z_{i,j-1}) / (2*hy)
fxy_{i,j} ~ (z_{i+1,j+1} - z_{i+1,j-1} - z_{i-1,j+1} + z_{i-1,j-1}) / (4*hx*hy)
```

**B-spline bicubic**: use B-spline basis functions in each direction. Equivalent to Hermite bicubic with specific derivative approximations.

**Properties**:
- C1 continuous across cell boundaries (C2 if using B-spline variant)
- 4th-order accurate on smooth functions
- Requires 4x4 stencil per evaluation (vs 2x2 for bilinear)

### 12.3 Tensor Product Extension to Higher Dimensions

For d dimensions on a regular grid: tensor product of d 1D interpolations.

Cost per evaluation:
- d-linear: 2^d grid point lookups, 2^d multiplies
- d-cubic: 4^d grid point lookups, 4^d multiplies

**Curse of dimensionality**: For d > ~5, tensor product methods become impractical (4^10 ~ 10^6 lookups per evaluation for cubic in 10D). Use sparse grids (Smolyak) or RBF/mesh-free methods instead.

### 12.4 Scattered Data: Delaunay Triangulation + Barycentric Interpolation

For irregularly spaced 2D data points {(xi, yi, zi)}:

**Step 1**: Compute the Delaunay triangulation of {(xi, yi)}. This partitions the convex hull into triangles with the property that no data point lies inside the circumcircle of any triangle.

**Step 2**: For each query point (x,y), find the containing triangle.

**Step 3**: Compute barycentric coordinates:
```
Given triangle with vertices (x1,y1), (x2,y2), (x3,y3):
det = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / det
lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / det
lambda3 = 1 - lambda1 - lambda2
```

**Step 4**: Interpolate:
```
f(x,y) = lambda1*z1 + lambda2*z2 + lambda3*z3
```

**Properties**:
- Linear interpolation within each triangle (C0 overall)
- No overshoots within each triangle (barycentric coordinates are in [0,1])
- The Delaunay triangulation maximizes the minimum angle, avoiding thin triangles

**Higher-order on triangulations**: Use Clough-Tocher (cubic, C1) or Powell-Sabin (quadratic, C1) elements. These require derivative estimation at vertices.

### 12.5 Shepard's Method (Inverse Distance Weighting)

```
f(x) = sum_{i=1}^{N} wi(x) * zi / sum_{i=1}^{N} wi(x)

where wi(x) = 1 / ||x - xi||^p     (p > 0, typically p = 2)
```

For ||x - xi|| = 0: f(x) = zi (exact interpolation at data points).

**Properties**:
- Works in ANY dimension
- No grid structure required
- Global method (all data points contribute)
- Produces a "flat spot" at each data point (gradient is zero): the interpolant looks like a plateau at each zi

**Modified Shepard's method** (Franke-Nielson):
- Use only the K nearest neighbors (locality)
- Use a modified weight: wi(x) = ((R - ||x-xi||)_+ / (R * ||x-xi||))^p where R is a radius parameter
- Reduces flat-spot effect

### 12.6 GPU Decomposition for Multivariate Methods

**Bilinear/Bicubic on regular grid**:
```
Per query: gather(binary_search_2d(x,y), grid_values) + fused_expr(interpolation_formula)
Grouping: None. Embarrassingly parallel over queries.
The 2D binary search is two independent 1D binary searches.
```

**Delaunay triangulation** (setup):
```
Incremental algorithms (Bowyer-Watson) are inherently sequential.
GPU Delaunay: divide-and-conquer (split points spatially, triangulate halves, merge).
  The merge step is the bottleneck -- O(n) sequential in worst case.
  Practical GPU implementations (gDel2D, gDel3D) achieve good speedups
  by handling the common case in parallel and serializing only conflicts.
Not naturally expressible as accumulate -- this is a geometric algorithm.
```

**Point location in triangulation** (per query):
```
Walking algorithm: start from a known triangle, "walk" toward the query point.
  O(sqrt(N)) expected steps. NOT easily parallelized per query step,
  but queries are independent.
Alternative: build a spatial index (grid, quadtree) on top of the triangulation.
  gather(spatial_hash(x,y), triangle_list) + walking from nearby triangle.
```

**Shepard's method**:
```
Naive: accumulate(all_data_points, All, wi(x)*zi, Add) / accumulate(..., All, wi(x), Add)
  Per query: O(N) work. Parallelize over queries.
  This is two reduces (weighted sum and weight sum).

With K nearest neighbors: first gather(KNN(x, data), data) then reduce over K points.
  The KNN step is the bottleneck -- requires a spatial index.
```

**Barycentric interpolation on triangulation**:
```
Per query: locate_triangle(x,y) + fused_expr(barycentric_coords * z_values)
  locate_triangle: gather with spatial addressing
  barycentric eval: 3 multiplies + 2 adds
  Embarrassingly parallel over queries once triangulation is built.
```

### 12.7 Edge Cases (Multivariate)

- **Collinear points in 2D**: Delaunay triangulation degenerates (all triangles are degenerate/thin). Detect and handle (1D interpolation along the line).
- **Query outside convex hull**: bilinear/bicubic on grid handles via extrapolation modes. Delaunay: query is outside all triangles. Shepard: still computes but extrapolates poorly.
- **Very non-uniform point distribution**: Delaunay triangles can be extremely thin in sparse regions, causing poor interpolation quality. RBF may be better.
- **High dimension**: tensor product is exponential. Use RBF (Section 8), sparse grids, or dimension-reduction first.
- **Coincident points**: Delaunay is undefined. Shepard has 0/0. Detect and merge.

---

## Cross-Cutting Concerns

### Numerical Precision: f32 vs f64

| Method | f32 viable? | Notes |
|--------|-------------|-------|
| Linear interp | YES | Only lerp; minimal precision loss |
| Polynomial (degree <= 5) | Marginal | Barycentric form essential; monomial form fails |
| Polynomial (degree > 10) | NO | Even Chebyshev nodes can't save f32 for high degree |
| Cubic spline | YES (carefully) | Tridiagonal system is well-conditioned; coefficient computation ok |
| B-spline (degree <= 5) | YES | De Boor algorithm is stable |
| Monotone/Akima/PCHIP | YES | Purely local; same precision as cubic Hermite |
| RBF (large epsilon) | Marginal | Kernel matrix conditioning depends on epsilon |
| RBF (small epsilon) | NO | Condition number makes f32 useless |
| Least squares (QR) | YES for p <= 3 | QR on Vandermonde with p=3 is ok in f32 |
| Least squares (normal eq) | NO for p > 2 | cond^2 effect kills f32 |
| Chebyshev approximation | YES for n <= ~30 | DCT is stable; Clenshaw is stable |
| Pade | NO for L+M > ~10 | Coefficient sensitivity too high |
| Bilinear/Bicubic | YES | Same as 1D linear/cubic |
| Shepard | YES | Only weighted averaging |

### Complete Accumulate Decomposition Summary

| Method | Setup | Setup Accumulate | Eval | Eval Accumulate |
|--------|-------|-----------------|------|-----------------|
| Linear | Sort knots | gather(sorted) | Binary search + lerp | gather + fused_expr |
| Polynomial (Newton) | Divided diff table | n stages of Windowed(2) | Horner | fused_expr (sequential per query) |
| Cubic spline | Tridiagonal solve | Prefix(forward) with Affine(2x2) | Binary search + Horner | gather + fused_expr |
| B-spline | Knot setup + possibly least squares | Tiled (if fitting) | De Boor | gather + fused_expr |
| Fritsch-Carlson | Secants + slope correction | fused_expr (parallel) | Hermite | gather + fused_expr |
| Akima | Secants + weighted slopes | Windowed(5) stencil | Hermite | gather + fused_expr |
| PCHIP | Secants + harmonic mean slopes | Windowed(3) stencil | Hermite | gather + fused_expr |
| RBF | Kernel matrix + dense solve | Tiled (distance matrix) + F02 solve | Sum of RBF evals | All reduce (dot product) |
| Least squares | V'V and V'y (or QR) | Tiled (GEMM) + F02 solve | Polynomial eval | fused_expr |
| Chebyshev | Sample + DCT | F03 FFT | Clenshaw | Prefix(reverse) with Affine(2x2) |
| Pade | Taylor coeffs + M x M solve | F02 solve | Horner (num/denom) | fused_expr |
| Bilinear | Grid setup | None | 2D binary search + lerp | gather + fused_expr |
| Bicubic | Grid + derivative estimation | Windowed stencil (finite diff) | 4x4 stencil eval | gather + fused_expr |
| Delaunay+bary | Triangulation | Geometric (not accumulate) | Point location + bary | gather + fused_expr |
| Shepard | None | None | IDW sum | All reduce (weighted) |

### GPU Parallelism Classification

**Embarrassingly parallel (setup AND eval)**:
- Linear interpolation
- Akima
- PCHIP / Fritsch-Carlson
- Shepard's method
- Bilinear interpolation

**Parallel eval, structured setup (tridiagonal/banded)**:
- Cubic spline (tridiagonal -> prefix scan with Affine(2x2))
- B-spline fitting (banded system)

**Parallel eval, dense setup (requires F02 linear algebra)**:
- RBF interpolation (N x N dense system)
- Pade approximation (M x M dense system)
- Least squares via QR/SVD

**Sequential setup, parallel eval**:
- Polynomial (divided differences: n stages)
- Chebyshev (DCT: O(log n) depth via FFT)
- Bicubic (derivative estimation: stencil, parallel; coefficient computation: per-cell parallel)

**Fundamentally hard to parallelize**:
- Delaunay triangulation (geometric, conflict resolution)
- Adaptive methods (degree/resolution selection depends on error estimates)

### Dependencies on Other Families

| F31 Method | Depends On |
|------------|-----------|
| Cubic spline tridiagonal | F02 (tridiagonal solve) OR accumulate Prefix(forward) with Affine(2x2) |
| RBF system solve | F02 (Cholesky, LU, or iterative) |
| RBF distance matrix | F01 (distance computation) |
| Least squares QR/SVD | F02 (QR factorization, SVD) |
| Chebyshev transform | F03 (DCT/FFT) |
| LASSO coordinate descent | F05 (optimization, proximal operators) |
| Delaunay triangulation | F29 (graph/geometric algorithms) |
| KNN for modified Shepard | F20 (KNN) via F01 (distance) |

---

## Summary of Method Selection Guide

**For 1D interpolation of smooth data on regular/irregular grids**:
- First choice: Cubic spline (natural or not-a-knot). Best accuracy for smooth data.
- If monotonicity matters: PCHIP or Fritsch-Carlson.
- If locality matters and no system solve desired: Akima.
- If data is cheap to evaluate and you want automatic precision: Chebyshev approximation.

**For 1D approximation (noisy data, more points than parameters)**:
- Low degree: Polynomial least squares via QR.
- With regularization: Ridge (L2) or LASSO (L1).
- Near-minimax: Chebyshev with discrete transform.
- Functions with poles/singularities: Pade approximation.

**For 2D interpolation on regular grids**:
- Bilinear (fast, C0) or bicubic (accurate, C1/C2).

**For scattered data in 2D**:
- Few points (<1000): Delaunay + linear barycentric (simple, robust).
- Many points, smooth function: RBF (global accuracy).
- Many points, local queries: Modified Shepard or local RBF patches.

**For high-dimensional scattered data**:
- RBF is the only general option (tensor products scale exponentially).
- Sparse grids (Smolyak construction) for smooth functions on hypercubes.

**For GPU implementation priority**:
1. Linear interpolation (trivial, needed everywhere)
2. PCHIP / Fritsch-Carlson (shape-preserving, no system solve, embarrassingly parallel)
3. Akima (local, no system solve, embarrassingly parallel)
4. Cubic spline (needs tridiagonal solve as prefix scan -- tests the Affine(2x2) operator)
5. Chebyshev approximation (needs DCT -- tests the F03 connection)
6. Bilinear/bicubic (regular grid, needed for image processing and 2D data)
7. Least squares polynomial (needs GEMM and QR -- tests F02 connection)
8. RBF (needs dense linear algebra -- large-scale F02)
9. B-spline (CAD/NURBS connection, less urgent for data science)
10. Pade (specialized, small systems only)
11. Delaunay (geometric, hardest to parallelize)
