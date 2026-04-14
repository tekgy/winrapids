# Validation Report: sin / cos

Scientist: tambear-trig team (TRIG-18)  
Date: 2026-04-13  
Implementation: `crates/tambear/src/recipes/libm/sin.rs` (commit 0bbae82)

---

## Method

**Gold standard**: mpmath at 100-digit precision, rounded to f64 for comparison  
**Reference**: numpy (Windows platform libm / SVML)  
**Adversarial inputs**: 2521 points (see parity_runner.py)  
  - Multiples of pi/4 in [-50pi, 50pi]  
  - Near-zero: powers of 2 from 2^-52 to 2^0  
  - Large argument: 1e4, 1e5, 1e6, 1e7, 1e10, 1e15, 1e17 (Payne-Hanek territory)  
  - Kahan hard cases: float(pi), float(2pi), float(pi/2), 355.0, 1e5  
  - Dense sweep: 2000 points in [0, 2*pi]  

---

## Synthetic Ground Truth

sin(n*pi/6) for n in {0..12} — known exact rational or sqrt values.

Key observations:

1. **n=0, 3, 9**: sin(0)=0, sin(pi/2)=1, sin(3*pi/2)=-1 are exact in f64.
   numpy returns 0, 1, -1 exactly. 0 ulp.

2. **n=6, 12**: sin(pi)=0 and sin(2*pi)=0 mathematically, but float(pi) != pi.
   numpy correctly returns ~1.22e-16 and ~-2.45e-16 respectively.
   ULP distance from 0.0 is enormous — but this is NOT a bug.
   The correct test is: numpy_sin(float_pi) vs mpmath_sin(float_pi) = 0 ulp.
   Both agree: they both compute sin of the same irrational-approximated argument.

3. **n=1, 4, 5, 11**: sin(pi/6)=0.5 is NOT exact in f64 because pi/6 is not
   rational and float(pi)/6 rounds. numpy shows 1 ulp vs exact 0.5.
   This is the expected residual from the argument irrreducibility.

4. **n=8**: 1 ulp gap between numpy and mpmath at n=8 (x=4*pi/3).
   This is genuine computation difference, within budget.

---

## Full Adversarial Sweep Results

### sin

| Metric | Value |
|--------|-------|
| Total inputs | 2521 |
| Bit-perfect vs mpmath | 2421 (96.0%) |
| <= 1 ulp vs mpmath | 2521 (100.0%) |
| Worst ulp | 1 |
| Worst input | x = -153.15264186250241 |

### cos

| Metric | Value |
|--------|-------|
| Total inputs | 2521 |
| Bit-perfect vs mpmath | 2429 (96.4%) |
| <= 1 ulp vs mpmath | 2521 (100.0%) |
| Worst ulp | 1 |
| Worst input | x = -1.0e15 |

---

## Large-Argument Analysis

For |x| > 2^20 * pi/2 ~= 1,647,100 (Payne-Hanek reduction territory):

- sin: 8 inputs tested, worst = 0 ulp. Platform libm and mpmath agree exactly on all.
- cos: 8 inputs tested, worst = 1 ulp at x=-1e15.

The 1-ulp gap at x=-1e15 for cos is within the ≤2 ulp budget. At this magnitude
the range reduction dominates — libm uses ~85-bit Cody-Waite, which is exhausted
at this scale (2^20*pi/2 threshold). The 1 ulp is rounding boundary behavior.
Tambear's Payne-Hanek for |x| >= 2^20*pi/2 would match mpmath here to ~120 bits.

---

## Special Cases

Verified in tambear Rust tests:
- sin(+0.0) = +0.0 (IEEE 754, sign preserved)
- sin(-0.0) = -0.0 (IEEE 754, sign preserved)
- sin(NaN) = NaN
- sin(+inf) = NaN
- sin(-inf) = NaN
- cos(+0.0) = 1.0
- cos(-0.0) = 1.0 (cos is even, -0 treated as +0 for output)
- cos(NaN) = NaN
- cos(+inf) = NaN
- cos(-inf) = NaN

---

## tambear Implementation Notes

The sin.rs implementation (commit 0bbae82) uses:
- Remez-optimized polynomial coefficients fit in mpmath at 80-digit precision
- Three-strategy entry points: _strict, _compensated, _correctly_rounded
  (currently all three delegate to the same path — differentiation is planned)
- Cody-Waite three-part pi/2 reduction for |x| < 2^20 * pi/2
- Payne-Hanek with 1200-bit 2/pi table for |x| >= 2^20 * pi/2

The existing tests compare against x.sin() (platform libm). The adversarial test
battery in adversarial.rs bounds |x| at 1e6 for sin_cos, which is BELOW the
Payne-Hanek threshold. This means the adversarial tests currently don't exercise
Payne-Hanek for sin/cos. This is a gap to flag.

**Gap**: sin_cos_adversarial() should extend to 1e17 once Payne-Hanek is verified.

---

## Convergence Check (Policy Gaps)

Running the policy-gap pattern search:

| Implementation | sin(+0) | sin(-0) | cos(-0) | sin(inf) | cos(inf) |
|----------------|---------|---------|---------|----------|----------|
| numpy          | +0.0    | -0.0    | 1.0     | NaN      | NaN      |
| mpmath         | 0       | 0       | 1       | NaN      | NaN      |
| tambear        | +0.0    | -0.0    | 1.0     | NaN      | NaN      |

No policy gaps found. All implementations agree on IEEE 754-2019 special cases.

**Structural observation**: numpy and mpmath agree on every tested special case
for sin and cos. There are no "policy choice" inputs here — these functions are
fully specified by IEEE 754-2019. This contrasts with atan2 (20+ edge cases that
implementations handle differently) which will need explicit policy decisions.

---

## Sign-off

numpy reference validation (sin, cos): **VERIFIED**  
- ≤ 1 ulp vs mpmath on 2521 adversarial inputs  
- 100% of inputs within 1 ulp of gold standard  
- No policy gaps vs IEEE 754-2019  

tambear Rust implementation: **pending Rust bindings for Python comparison**  
- Rust tests compare against x.sin() (platform libm, ≤ 1 ulp vs mpmath)  
- When tambear Python bindings ship, add tambear column to parity table  
- Expected: tambear sin/cos should match or beat numpy (same algorithm class)  

Scientist sign-off (preliminary): baseline established. Full sign-off pending
tambear column.
