# Family 38: Arbitrary Precision Arithmetic — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (limb-parallel operations), B (carry propagation scan)

---

## Core Insight: Two Kingdoms in One Number

A multi-precision integer is an array of machine-word "limbs." Arithmetic on limbs is parallel (Kingdom A). Carry propagation between limbs is sequential (Kingdom B — a prefix scan). Every operation has both layers, and the architecture must respect this.

The GPU opportunity: for LARGE numbers (thousands of limbs — RSA, elliptic curve, polynomial GCD over Z), the limb-parallel layer dominates. For small bignums (tens of limbs — exact financial arithmetic), CPU is faster due to launch overhead. The crossover depends on limb count.

**Honest assessment**: arbitrary precision is useful as a SUPPORT family — enabling exact GCD in F36 (Symbolic), exact accumulation for catastrophic cancellation in F06 (Descriptive), and certified computation. It is not the primary compute bottleneck for any algorithm family.

---

## 1. Representation

### Multi-Precision Integer
A number N = Σ_{i=0}^{n-1} dᵢ · B^i where B = 2^64 (limb base) and dᵢ ∈ [0, B-1].

```
struct BigInt {
    sign: bool,         // true = negative
    limbs: Vec<u64>,    // little-endian (limbs[0] = least significant)
}
```

**Little-endian limb order**: limbs[0] is the least significant. This matches carry propagation direction (low to high) and makes addition natural.

### Multi-Precision Rational
```
struct BigRational {
    numerator: BigInt,
    denominator: BigInt,   // always positive, gcd(num, den) = 1
}
```

**Invariant**: always in lowest terms (divide by GCD after every operation). The denominator is always positive (sign carried by numerator).

### Multi-Precision Float
```
struct BigFloat {
    sign: bool,
    significand: BigInt,   // normalized: MSB is 1
    exponent: i64,         // binary exponent
    precision: u32,        // bits of precision (user-specified)
}
```

**MPFR-style**: user specifies precision at creation time. All operations round to that precision.

---

## 2. Addition and Subtraction

### Algorithm (schoolbook with carry)
```
carry = 0
for i in 0..max(n, m):
    sum = a_i + b_i + carry
    result_i = sum mod B
    carry = sum / B        // 0 or 1
result_n = carry           // possible extra limb
```

### GPU Decomposition
- **Limb addition**: `accumulate(Contiguous, a_i + b_i, Identity)` — parallel per limb. Kingdom A.
- **Carry propagation**: `accumulate(Prefix(forward), carry_generate_i, CarryOp)` — sequential scan. Kingdom B.

The carry operation: generate = (sum ≥ B), propagate = (sum = B-1). This is a classic parallel prefix problem (Brent-Kung, Kogge-Stone).

### Parallel Carry (Brent-Kung)
```
Generate_i = (a_i + b_i ≥ B)
Propagate_i = (a_i + b_i = B-1)
```
Combine: (G,P) ⊕ (G',P') = (G | (P & G'), P & P'). This IS associative → prefix scan works.

**Complexity**: O(n/p + log p) where n = limbs, p = processors.

### When to Parallelize
- n < 64 limbs (~4096 bits): sequential is faster (no launch overhead)
- n ≥ 64 limbs: Brent-Kung prefix scan
- n ≥ 1024 limbs: full GPU launch worthwhile

---

## 3. Multiplication

### Schoolbook O(n²)
```
for i in 0..n:
    carry = 0
    for j in 0..m:
        product = a_i * b_j + result_{i+j} + carry
        result_{i+j} = product mod B
        carry = product / B
    result_{i+m} = carry
```

### Karatsuba O(n^1.585)
Split a = a₁·B^(n/2) + a₀, b = b₁·B^(n/2) + b₀.
```
z₀ = a₀ · b₀
z₂ = a₁ · b₁
z₁ = (a₀ + a₁)(b₀ + b₁) - z₀ - z₂
result = z₂ · B^n + z₁ · B^(n/2) + z₀
```
3 multiplications instead of 4. Recursive.

### Toom-Cook-3 O(n^1.465)
Split into 3 parts. 5 multiplications instead of 9. Used by GMP for medium sizes.

### Schönhage-Strassen / Harvey-van der Hoeven O(n log n)
FFT-based multiplication. For very large numbers (>10,000 limbs).
Uses F03 FFT infrastructure (NTT — Number Theoretic Transform over Z/pZ).

### Algorithm Selection

| Limb count | Method | Why |
|-----------|--------|-----|
| < 32 | Schoolbook | Overhead of recursion exceeds savings |
| 32-256 | Karatsuba | 1.585 exponent vs 2.0 |
| 256-2048 | Toom-Cook-3 | 1.465 exponent |
| > 2048 | NTT-based | O(n log n) |

### GPU Decomposition
Schoolbook: outer products → `accumulate(Tiled{a_limbs, b_limbs}, a_i * b_j, Sum)` per result limb. Then carry propagation scan.

NTT: use F03's existing FFT but over finite field Z/pZ instead of complex numbers. Same algorithm, different ring.

---

## 4. Division

### Newton's Method for Reciprocal
To compute a/b: first compute 1/b, then multiply.

1/b via Newton: xₖ₊₁ = xₖ(2 - b·xₖ). Quadratic convergence. Kingdom C (iterative).

Each iteration: 2 multiplications + 1 subtraction. Multiplications are the bottleneck → use fast multiply.

### Barrett Reduction
For modular arithmetic (a mod m): precompute μ = ⌊B²ⁿ/m⌋, then:
```
q̂ = ⌊(a · μ) / B²ⁿ⌋
r = a - q̂ · m
if r ≥ m: r -= m    // at most 2 corrections
```

One multiplication by precomputed μ. Useful when dividing many numbers by the same m (modular exponentiation).

### Montgomery Multiplication
For modular multiplication in a loop (RSA, EC): work in Montgomery form where multiply-then-reduce is replaced by multiply-and-shift.

```
REDC(T) = (T + (T · m' mod R) · m) / R
```

Avoids expensive division entirely. All operations are multiply + shift.

---

## 5. GCD and Modular Arithmetic

### Binary GCD (Stein's algorithm)
Avoids division entirely — uses only shifts and subtraction:
```
while a ≠ b:
    if a is even: a >>= 1
    elif b is even: b >>= 1
    elif a > b: a = (a - b) >> 1
    else: b = (b - a) >> 1
```

**Complexity**: O(n²) for n-bit inputs. Sequential (Kingdom C).

### Lehmer's GCD
For large inputs: use leading limbs to predict several quotient steps, batch them. Reduces to O(n · M(n) / n) = O(M(n) log n) where M(n) is multiplication cost.

### Extended GCD
Compute gcd(a,b) AND Bezout coefficients (s,t) where as + bt = gcd(a,b). Needed for modular inverse.

### Modular Exponentiation
```
a^e mod m
```

**Square-and-multiply** (binary method):
```
result = 1
for bit in e (MSB to LSB):
    result = result² mod m
    if bit == 1: result = result · a mod m
```

O(log e) multiplications. Each multiplication uses Barrett or Montgomery reduction.

### Chinese Remainder Theorem
Given x ≡ rᵢ (mod mᵢ) for coprime mᵢ, reconstruct unique x mod (m₁·m₂·...·mₖ).

Uses extended GCD for modular inverses. Enables splitting large modular arithmetic into independent smaller computations (parallelizable!).

---

## 6. Exact Rational Arithmetic

### Operations
```
a/b + c/d = (a·d + b·c) / (b·d)     then reduce by GCD
a/b · c/d = (a·c) / (b·d)            then reduce by GCD
a/b ÷ c/d = (a·d) / (b·c)            then reduce by GCD
```

### CRITICAL: Coefficient Blowup
Rational arithmetic without reduction: denominators grow exponentially. MUST reduce by GCD after every operation.

Even with reduction: Gaussian elimination on rational matrices → denominators grow as O(n! · max(entry)^n). This is why numerical (float) linear algebra is preferred for large systems.

### When to Use Rational Arithmetic
- **Yes**: F36 symbolic computation (exact polynomial GCD), financial calculations requiring exact decimal answers, verification of numerical algorithms
- **No**: Large matrix operations (coefficient blowup), iterative algorithms (convergence requires floating point), GPU-bound computation (rational reduces throughput 100-1000x)

---

## 7. Decimal Arithmetic (Financial)

### IEEE 754-2008 Decimal
Exact representation of decimal fractions: 0.1 IS exactly 0.1 (unlike binary float where 0.1 ≈ 0.1000000000000000055511151231257827021181583404541015625).

### Representation
```
struct Decimal {
    coefficient: i128,    // or BigInt for arbitrary precision
    exponent: i32,        // value = coefficient × 10^exponent
}
```

### Rounding Modes (IEEE 754)

| Mode | Description | Example (round 2.5) |
|------|------------|---------------------|
| RoundHalfEven | Banker's rounding | 2 |
| RoundHalfUp | Traditional | 3 |
| RoundDown | Toward zero | 2 |
| RoundUp | Away from zero | 3 |
| RoundCeiling | Toward +∞ | 3 |
| RoundFloor | Toward -∞ | 2 |

**Decision**: Default to RoundHalfEven (banker's rounding). Matches financial industry standard and IEEE 754 default.

### Fixed-Point vs Floating-Point Decimal
- **Fixed-point**: exponent is fixed (e.g., always 10⁻²  for cents). Faster, simpler.
- **Floating-point**: exponent varies. More flexible, more complex.
- **For financial**: fixed-point usually sufficient (known currency precision).

---

## 8. Numerical Stability Bridge

### Compensated Summation via Exact Arithmetic
When f64 summation produces unacceptable error, use exact accumulation:
```
// Priest's doubly-compensated summation
// or: accumulate in exact rational, extract f64 at end
```

### Error-Free Transformations
Given a + b = s (floating point), the error e = (a + b) - s can be computed exactly:
```
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)     // s + e = a + b exactly
}
```

This enables **compensated algorithms** that track rounding error without arbitrary precision overhead. Useful for:
- Exact dot product (for condition number estimation)
- Verified summation (bound on accumulated error)
- Residual computation in iterative refinement

### When Exact Arithmetic Helps F06
The naive formula E[x²]-E[x]² can be computed exactly via compensated summation. But RefCentered (Welford/Pebay) is a better solution — it avoids the precision issue structurally rather than computationally.

**Rule**: prefer structural solutions (centered basis) over brute-force precision. Use exact arithmetic only when structural solutions are unavailable.

---

## 9. Edge Cases

| Operation | Edge Case | Expected |
|-----------|----------|----------|
| BigInt add | Opposite signs, equal magnitude | Exact zero |
| BigInt mul | One operand is zero | Zero (no allocation) |
| BigInt div | Division by zero | Error (not NaN) |
| BigInt div | Exact division | No remainder |
| BigRational | 0/0 | Undefined — error |
| BigRational | Denominator overflow | GCD reduction should prevent |
| GCD | gcd(0, 0) | 0 (convention) |
| GCD | gcd(0, n) | |n| |
| Mod exp | 0^0 mod m | 1 (convention in number theory) |
| Mod exp | a^e mod 1 | 0 |
| Decimal | 0.1 + 0.2 | Exactly 0.3 (unlike f64) |
| Carry prop | All limbs = B-1 (max carry chain) | Correct carry to new limb |
| Two_sum | Inf + (-Inf) | NaN (IEEE 754) |

---

## Sharing Surface

### Reuses from Other Families
- **F03 (Signal Processing)**: NTT (Number Theoretic Transform) for fast bignum multiplication
- **F02 (Linear Algebra)**: matrix operations over exact fields (rational verification)

### Provides to Other Families
- **F36 (Symbolic)**: exact integer/rational arithmetic for polynomial GCD, resultant, factorization
- **F06 (Descriptive)**: compensated summation for catastrophic cancellation prevention
- **F32 (Numerical)**: interval arithmetic endpoints, error bounds
- **Financial pipeline**: exact decimal arithmetic for currency calculations

### Structural Rhymes
- **Carry propagation = prefix scan**: same as F17 cumulative operations (same Prefix(forward) grouping)
- **Limb multiplication = tiled accumulate**: same as F02 GEMM (tile of limbs × tile of limbs)
- **NTT = FFT over finite field**: same algorithm as F03, different algebraic structure
- **GCD iteration = Kingdom C outer loop**: same pattern as F05 optimization convergence

---

## Implementation Priority

**Phase 1** — Core bignum (~200 lines):
1. BigInt representation (sign + limbs)
2. Addition, subtraction (with Brent-Kung carry for large)
3. Schoolbook multiplication
4. Comparison, shift, bitwise operations
5. Division (Newton reciprocal)

**Phase 2** — Fast multiplication (~150 lines):
6. Karatsuba multiplication
7. Toom-Cook-3
8. NTT-based multiplication (via F03)
9. Algorithm selection by size

**Phase 3** — Modular arithmetic (~150 lines):
10. Barrett reduction
11. Montgomery multiplication
12. Modular exponentiation (square-and-multiply)
13. GCD (binary), extended GCD
14. Chinese Remainder Theorem

**Phase 4** — Exact rational and decimal (~100 lines):
15. BigRational (with auto-reduction)
16. Decimal (fixed-point, IEEE 754 rounding)
17. Error-free transformations (two_sum, two_product)
18. Compensated summation

---

## Composability Contract

```toml
[family_38]
name = "Arbitrary Precision Arithmetic"
kingdom = "Mixed — A (limb-parallel), B (carry scan)"

[family_38.shared_primitives]
bigint_arithmetic = "Add/sub/mul/div on multi-precision integers"
modular_arithmetic = "Barrett/Montgomery mod, modexp, GCD"
rational_arithmetic = "Exact rational with auto-reduction"
decimal_arithmetic = "IEEE 754 decimal with rounding modes"

[family_38.reuses]
f03_signal_processing = "NTT for fast bignum multiplication"
f02_linear_algebra = "Matrix operations for verification over exact fields"

[family_38.provides]
exact_integer = "BigInt type for polynomial GCD, factorization"
exact_rational = "BigRational type for symbolic computation"
exact_decimal = "Decimal type for financial arithmetic"
compensated_ops = "Error-free transformations for numerical stability"

[family_38.consumers]
f36_symbolic = "Exact polynomial arithmetic (GCD, resultant)"
f06_descriptive = "Compensated summation when centered basis insufficient"
f32_numerical = "Interval arithmetic, error bounds"
financial = "Exact currency calculations"
```
