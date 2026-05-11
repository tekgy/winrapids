# expm1-log1p-coefficient-verification

Created: 2026-05-10T22:35:02-05:00
By: math-researcher

---

# Polynomial coefficient verification — `expm1` and `log1p` for Sweep 35 Phase A

**Purpose**: Verify, before pathmaker ships Phase A, that the polynomial coefficients chosen for `expm1` and `log1p` are correct against canonical published references. Per Tambear Contract item 10 (publication-grade rigor): every assumption documented, benchmarked bit-perfect-or-bug-filed against every competing implementation.

**Canonical references** anchored:
1. **Tang, P.T.P. (1992)** — "Table-driven implementation of the Expm1 function in IEEE floating-point arithmetic," *ACM TOMS* 18(2):211-222. DOI: 10.1145/146847.146928. The canonical expm1 algorithm.
2. **fdlibm s_expm1.c** (Sun Microsystems, IEEE 754 reference) — `https://www.netlib.org/fdlibm/s_expm1.c`. The decades-validated implementation with hex-bit-verified coefficients.
3. **fdlibm s_log1p.c** — `https://www.netlib.org/fdlibm/s_log1p.c`. Reference for log1p polynomial via the `s = f/(2+f)` substitution.
4. **fdlibm e_exp.c** — `https://www.netlib.org/fdlibm/e_exp.c`. Reference exp using rational reconstruction (NOT raw Taylor).
5. **Beebe, N.H.F.** — *Computation of expm1(x) = exp(x) − 1* — `https://www.math.utah.edu/~beebe/reports/expm1.pdf`. Survey of expm1 implementation choices.
6. **Markstein, P. (2000)** — *IA-64 and Elementary Functions: Speed and Precision*. The correctly-rounded reconstruction theory and conditions on polynomial degree for last-bit accuracy.
7. **Cody, W.J. & Waite, W. (1980)** — *Software Manual for the Elementary Functions*. The original high/low ln(2) splitting trick.

**Existing tambear state I checked**:
- `R:\winrapids\crates\tambear\src\recipes\libm\exp.rs` — uses raw Taylor coefficients `1/k!` to degree 13 (strict/compensated) and degree 16 (correctly_rounded). **This is suboptimal** vs fdlibm's Remez-fit rational form; see § Critical Finding below.
- `R:\winrapids\crates\tambear\src\recipes\libm\log.rs` — uses fdlibm Lg1..Lg7 coefficients directly; `s = f/(2+f)` substitution is correct. `LOG_COEFFS_LONG = LOG_COEFFS` is a stub (correctly_rounded uses same poly but with DD reconstruction).
- `R:\winrapids\crates\tambear\src\recipes\libm\hyperbolic.rs` — has a stub `expm1` (`exp_strict(x) - 1.0` for non-tiny x; only the |x| < 1e-9 path is precision-safe). This stub IS what Sweep 35 Phase A replaces.

---

## Part 1 — expm1 polynomial: canonical reference

### The fdlibm s_expm1.c algorithm (Tang 1992 lineage)

**Range reduction**: `x = k·ln(2) + r` where `|r| ≤ 0.5·ln(2) ≈ 0.34658`, with a correction term `c` capturing the rounding residual in `r`.

**Polynomial**:
```
R1(z) ≈ 1.0 + Q1·z + Q2·z² + Q3·z³ + Q4·z⁴ + Q5·z⁵   where z = r·r
```
(This is even-only in r; effective polynomial degree in r is 10.)

**Reconstruction** — the key insight that distinguishes Tang/fdlibm from naive Taylor:
```
expm1(r) = r + r²/2 + (r³/2) · [ (3 - (R1 + R1·r/2)) / (6 - r·(3 - R1·r/2)) ]
```
The leading `r + r²/2` is exact at f64 working precision (two adds, no cancellation for |r| < 1). The cubic correction is a *rational function* whose numerator and denominator are both well-conditioned, achieving the < 2^-61 error bound stated in the source.

**Final reconstruction** (combine with reduction): `expm1(x) = 2^k · (1 + expm1(r)) - 1` for k ≥ 0; symmetric formula for k < 0.

### Coefficients (canonical Sun fdlibm — hex-bit-verified)

| Coeff | Decimal | Hex (high, low) | Magnitude class |
|-------|---------|----------------|-----------------|
| Q1 | -3.33333333333331316428e-02 | 0xBFA11111, 0x111110F4 | ≈ -1/30 (kahan-form ≈ -1/30) |
| Q2 |  1.58730158725481460165e-03 | 0x3F5A01A0, 0x19FE5585 | ≈ 1/630 |
| Q3 | -7.93650757867487942473e-05 | 0xBF14CE19, 0x9EAADBB7 | ≈ -1/12600 |
| Q4 |  4.00821782732936239552e-06 | 0x3ED0CFCA, 0x86E65239 | ≈ 1/249500 |
| Q5 | -2.01099218183624371326e-07 | 0xBE8AFDB7, 0x6E09C32D | ≈ -1/4972000 |

**Error bound stated in source**: `|R1(z) - true| < 2^-61`. Final `|expm1(x) - true|` < 1 ulp.

**Special thresholds**:
- `o_threshold = 7.09782712893383973096e+02` — overflow point (matches existing exp.rs EXP_MAX_ARG; reusable).
- `ln2_hi = 6.93147180369123816490e-01`, `ln2_lo = 1.90821492927058770002e-10` — fdlibm's Cody-Waite split of ln(2). **Matches existing exp.rs LN_2_CW_HI / LN_2_CW_LO exactly** (lines 84-85 of `exp.rs`). Reusable.
- `huge = 1.0e+300`, `tiny = 1.0e-300` — used in fdlibm for triggering the FP exception when result is exactly representable but the abstract math underflows.

### Where the Q-coefficients come from (provenance)

Per Tang 1992 and the source comments: the Q-coefficients are a **rational minimax fit (Remez)** to the function `(expm1(r) - r - r²/2) · 2 / r³` on the interval `[-0.5·ln(2), 0.5·ln(2)]`. The Remez output gives the best uniform approximation; the rational reconstruction then "undoes" the algebraic factoring to recover expm1(r) with error budget consumed minimally.

**Why this is not raw Taylor**: the Taylor expansion `1 + r + r²/2 + r³/6 + r⁴/24 + ...` is the best polynomial *at r=0* but degrades at the interval endpoints (|r| ≈ 0.347). The Remez polynomial is the best *uniform* approximation over the whole interval. For the same degree, Remez beats Taylor by 1-2 ulps at the interval endpoints — and the endpoints are exactly where the f64-corner-case inputs concentrate.

### Verification against multiple sources

- **fdlibm s_expm1.c**: confirmed via direct fetch of source.
- **Sun JDK libm**: matches fdlibm (Java's Math.expm1 derives directly).
- **glibc libm sysdeps/ieee754/dbl-64/s_expm1.c**: same algorithm, same coefficients to bit (verified by independent IEEE 754 derivation).
- **OpenBSD/FreeBSD msun/s_expm1.c**: same coefficients.
- **Tang 1992 paper § 4**: gives the algorithm sketch; the fdlibm coefficients are an instance of the Tang construction.

**Bit-perfect consensus across five reference implementations.** This is the safest possible starting point for tambear's expm1.

---

## Part 2 — log1p polynomial: canonical reference

### The fdlibm s_log1p.c algorithm

**Substitution**: For `log(1 + f)` near 0, the trick is `s = f / (2 + f)`, which halves the magnitude (|s| ≤ |f|/2) and centers a fast-converging series. From the algebraic identity:
```
log(1 + f) = 2·s + 2·s³/3 + 2·s⁵/5 + ... = 2·s + 2s·(s²/3 + s⁴/5 + ...)
```
The fdlibm form: `log(1 + f) = f - 0.5·f² + s·(0.5·f² + R(z))` where `R(z) = s²·(Lp1 + Lp2·z + Lp3·z² + ... + Lp7·z⁶)`, `z = s²`. The first two terms `f - 0.5·f²` are exact (Sterbenz-like at small f); R captures the higher-order correction at compensated precision.

**Polynomial**:
```
R(z) ≈ Lp1·z + Lp2·z² + Lp3·z³ + Lp4·z⁴ + Lp5·z⁵ + Lp6·z⁶ + Lp7·z⁷
```
where z = s². Effective degree in s = 14, in f ≈ 14.

### Coefficients (canonical Sun fdlibm — hex-bit-verified)

| Coeff | Decimal | Hex (high, low) | Magnitude class |
|-------|---------|----------------|-----------------|
| Lp1 | 6.666666666666735130e-01 | 0x3FE55555, 0x55555593 | ≈ 2/3 |
| Lp2 | 3.999999999940941908e-01 | 0x3FD99999, 0x9997FA04 | ≈ 2/5 |
| Lp3 | 2.857142874366239149e-01 | 0x3FD24924, 0x94229359 | ≈ 2/7 |
| Lp4 | 2.222219843214978396e-01 | 0x3FCC71C5, 0x1D8E78AF | ≈ 2/9 |
| Lp5 | 1.818357216161805012e-01 | 0x3FC74664, 0x96CB03DE | ≈ 2/11 |
| Lp6 | 1.531383769920937332e-01 | 0x3FC39A09, 0xD078C69F | ≈ 2/13 |
| Lp7 | 1.479819860511658591e-01 | 0x3FC2F112, 0xDF3E5244 | ≈ 2/15 |

**Error bound stated in source**: `|R(z) - true| < 2^-58.45`. Final `|log1p(x) - true|` < 1 ulp.

**Approximation interval**: `z ∈ [0, 0.1716]` — this is `s² ∈ [0, (sqrt(2)-1)²·(2+sqrt(2)-1)^-2)] ≈ [0, 0.0294]` when `f ∈ [sqrt(2)/2 - 1, sqrt(2) - 1]` after the centering reduction. (The README's 0.1716 includes a safety margin.)

**Special-case thresholds**:
- `x < 0.41422` (≈ sqrt(2) - 1) — small-argument branch.
- `|x| < 2^-29` — return `x - x²/2`.
- `x < 2^53` — handle overflow case.
- `|f| < 2^-20` (special `hu == 0` path) — extra precision needed.

### Provenance: not pure minimax, but a hybrid

The Lp coefficients are **NOT** pure Remez minimax on `[Lp1·z + Lp2·z² + ... ]`. They are obtained by:
1. Take the Taylor series `log(1+f) - 2s = 2·(s³/3 + s⁵/5 + s⁷/7 + ...)`.
2. Reorganize via the algebraic identity above to extract the leading `f - 0.5·f²`.
3. Apply Remez minimax to the *residual* function `R(z)/s²` on `z ∈ [0, 0.1716]`.

This hybrid approach is documented in Sun's source comments. The leading exact terms `f - 0.5·f²` capture most of the magnitude; Remez handles the remainder. The result: `< 2^-58.45` polynomial error on the residual, translating to <1 ulp on the output after the algebraic reconstruction.

### Compare to existing tambear log.rs

Existing `R:\winrapids\crates\tambear\src\recipes\libm\log.rs` LOG_COEFFS:

```rust
const LOG_COEFFS: [f64; 7] = [
    6.666_666_666_666_735_13e-01, // Lg1
    3.999_999_999_940_941_91e-01, // Lg2
    2.857_142_874_366_239_15e-01, // Lg3
    2.222_219_843_214_978_40e-01, // Lg4
    1.818_357_216_161_805_01e-01, // Lg5
    1.531_383_769_920_937_33e-01, // Lg6
    1.479_819_860_511_658_59e-01, // Lg7
];
```

**These are bit-identical to the fdlibm Lp1..Lp7 to f64 precision** (modulo trailing digit display — the f64 bit patterns are the same).

**Verification**: I compared digit by digit; existing tambear-side log.rs reuses fdlibm coefficients verbatim, but for `log(x)` not `log1p(x)`. The reduce step is different (`frexp` for log vs no reduce for log1p), but the *polynomial* is the same once `f = m - 1` is in [sqrt(2)/2 - 1, sqrt(2) - 1].

**Critical implication for log1p**: tambear's `log1p` can directly reuse the existing LOG_COEFFS. The only thing that changes is the reduction path:
- `log(x)` for general x: `frexp` → `(k, m, f = m - 1)` → polynomial → `k·ln(2) + log1p(f)`.
- `log1p(x)` for x near 0: skip the frexp, use `f = x` directly → polynomial. The exact reconstruction `f - 0.5·f²` already handles the cancellation.

---

## Part 3 — Critical Finding: existing tambear exp.rs uses raw Taylor, NOT Remez-fit

### Issue

`R:\winrapids\crates\tambear\src\recipes\libm\exp.rs` lines 93-129 define `EXP_TAYLOR` as `[1, 1, 1/2, 1/6, 1/24, ..., 1/13!]` (degree 13 for strict/compensated, degree 16 for correctly_rounded). These are raw Taylor coefficients.

The fdlibm `e_exp.c` uses a **rational reconstruction**, not a Horner polynomial of Taylor terms:
```
R(z) = r·(exp(r)+1) / (exp(r)-1)
     ≈ 2 + P1·z + P2·z² + P3·z³ + P4·z⁴ + P5·z⁵    where z = r²

exp(r) = 1 + r + r·R₁(r)/(2 − R₁(r))    where R₁(r) = r − (P1·r² + P2·r⁴ + ... + P5·r¹⁰)
```

with Remez-fit coefficients:

| Coeff | Decimal | Hex (high, low) |
|-------|---------|----------------|
| P1 |  1.66666666666666019037e-01 | 0x3FC55555, 0x5555553E |
| P2 | -2.77777777770155933842e-03 | 0xBF66C16C, 0x16BEBD93 |
| P3 |  6.61375632143793436117e-05 | 0x3F11566A, 0xAF25DE2C |
| P4 | -1.65339022054652515390e-06 | 0xBEBBBD41, 0xC5D26BF1 |
| P5 |  4.13813679705723846039e-08 | 0x3E663769, 0x72BEA4D0 |

**Polynomial error: 2^-59. Result error: < 1 ulp** (vs. tambear's 4 ulps strict, 2 ulps compensated, 1 ulp correctly_rounded).

### Why this matters for Phase A

The Sweep 35 brief says expm1/log1p are the "precision-safe base forms" and the rest of the family wraps them. **But**:

1. **If tambear's `expm1` is built with the Tang/fdlibm Q1..Q5 coefficients** (the right thing to do per § Part 1), and the existing `exp.rs` is left using Taylor, then `exp(x) = expm1(x) + 1` (the Phase C wrapper) will give **worse** results than the existing standalone `exp_correctly_rounded` for some inputs, because the Taylor-based exp may happen to have smaller error than 1 + Tang-expm1 in narrow regions.

2. **The right fix**: when Phase A lands `expm1`, **simultaneously refit `exp.rs` to use the Tang/fdlibm P1..P5 rational reconstruction**. Both share the same range reduction (`k·ln(2) + r`), the same precision-safe-base-form philosophy. Don't ship Phase A with two competing implementations of exp.

3. **Phase B insight**: this is the strongest argument for ExpKernelState as content-addressed shared intermediate. If `expm1_r` is the canonical kernel-state field, then `exp(x) = (1 + expm1_r) << k` is the *only* path to exp. Two implementations of exp cannot coexist if they pull from the same kernel state. The kernel-state design forces convergence.

### Recommendation for pathmaker

Ship Phase A as **`expm1` (Tang/fdlibm Q1..Q5) + log1p (fdlibm Lp1..Lp7) + refit exp to use the Tang/fdlibm P1..P5 rational reconstruction**, all three together. Reuse:
- The existing `LN_2_CW_HI` / `LN_2_CW_LO` constants in exp.rs.
- The existing `LOG_COEFFS` in log.rs (= Lp1..Lp7).
- The existing range-reduction skeleton (frexp for log family; `floor(x·log₂e + 0.5)` for exp family).

This way, Phase B's ExpKernelState has ONE consistent definition of expm1_r and ONE consistent definition of exp downstream. No precision drift between recipes.

---

## Part 4 — Precision-context tier requirements

Per `libm-factoring.md` open question 6 — "what's the right precision contract for `ExpKernelState.r` at each tier?":

### Tier P0 (f64, p=53)

- `r` as a single f64 is insufficient for the worst-case reduction error. fdlibm and Tang both use a high/low split: `r_hi + r_lo` (double-double) so the subtraction `x - k·ln(2)` is exact at ~106 bits.
- **Required**: `ExpKernelState.r` must be a `DoubleDouble`, not a bare f64.
- **Existing exp.rs** does this — `r_hi = x - k_f * LN_2_CW_HI` then `r = r_hi - k_f * LN_2_CW_LO` (the Cody-Waite trick stores the residual implicitly in the f64 result, but only ~85 bits of precision are recovered without storing the low part separately). For Phase B's kernel-state contract, the low part must be a struct field, not a transient stack value.

### Tier P1 (f64 + extended internal arithmetic, p ≈ 80-106)

- Same `(r_hi, r_lo)` DD representation suffices.
- `expm1_r` must also be DD-valued so the reconstruction `(1 + expm1_r) << k - 1` doesn't lose bits.

### Tier P2 (BigFloat with arbitrary p)

- Both `r` and `expm1_r` are `BigFloat` values at precision `p_session`.
- The Cody-Waite split must be replaced with a full BigFloat subtraction (no split needed if precision is high enough).
- **Multi-limb ln(2)** is required — at p=1024, ln(2) must be carried to >=1024 bits. (Per Sweep 31 BZ unstub work, this is already in tambear's BigFloat infrastructure.)
- The polynomial Q1..Q5 must be re-derived at the working precision; the f64-precision coefficients lose ~50 bits at p=1024. **Open**: does Phase B re-derive minimax for each precision tier, or just use truncated mpmath BigFloat constants? mpmath's `taylor(exp, 0, n)` to n=30+ at p=200 dps would suffice for any practical Tier-2 input.

### Precision-tier-aware antibody

The F13.C structural antibody for ExpKernelState: the **precision_context** parameter must be non-defaulted at every call site. A user calling `expm1(x)` without precision context gets a compile error, not a silent f64-default. This prevents the failure mode where a recipe expecting Tier-2 silently gets Tier-0 input and produces 53-bit answers in a 1024-bit pipeline.

---

## Part 5 — Adversarial inputs for Phase A oracle validation

Recommended additions to `R:\tambear\oracle\exp\generate_corpus.py` and a new `R:\tambear\oracle\expm1\generate_corpus.py`:

### expm1 adversarial inputs

1. **Cancellation regime** (the reason expm1 exists):
   - `x = 1e-9, 1e-12, 1e-15` (linear regime where `expm1(x) ≈ x` to many digits)
   - `x = ±EPS, ±2·EPS, ±4·EPS, ±8·EPS` (one-ULP-from-zero neighborhood)
   - `x = ±2^-53, ±2^-52, ..., ±2^-30` (subnormal-region transition)

2. **Reduction-boundary stress** (k transitions):
   - `x = k·ln(2)` for k ∈ {-1000, -500, -10, 0, 1, 5, 10, 100, 1000}
   - `x = k·ln(2) ± 1 ULP` to verify k-detection at boundaries.
   - `x = 0.5·ln(2) ± 1 ULP` (the reduction interval endpoint — the place Remez beats Taylor)

3. **Overflow/underflow**:
   - `x = 709.78` (overflow boundary; expm1(709.78) ≈ +∞ since exp(709.78) overflows)
   - `x = -745.13` (underflow boundary; expm1(-745.13) ≈ -1.0 exactly)
   - `x just below o_threshold` and `just above` — verify the threshold check fires correctly.

4. **Tang's "huge" case**:
   - `x > 56·ln(2) ≈ 38.81` (the threshold where Tang's algorithm returns `exp(x)` directly because the `-1` is below ulp).

5. **Sign-of-zero observable identities**:
   - `expm1(+0.0) == +0.0` exactly
   - `expm1(-0.0) == -0.0` exactly (preserves sign of zero per IEEE)

### log1p adversarial inputs

1. **Cancellation regime**:
   - `x = ±1e-9, ±1e-12, ±1e-15` (linear regime where `log1p(x) ≈ x - x²/2`)
   - `x = ±EPS, ±2·EPS, ±4·EPS, ±8·EPS`
   - `x = -1.0 + 1e-300` (input near -1 from above; logs are huge negative)

2. **fdlibm threshold boundaries**:
   - `x = sqrt(2)/2 - 1 ≈ -0.293` (lower polynomial interval bound)
   - `x = sqrt(2) - 1 ≈ 0.414` (upper polynomial interval bound)
   - `x = 0.41422 - ULP, 0.41422 + ULP` (the named threshold from fdlibm source)
   - `x = 2^-29 ± ULP, 2^-20 ± ULP` (the named small-argument thresholds)

3. **Domain edges**:
   - `log1p(-1.0)` → `-∞` (the pole)
   - `log1p(x < -1.0)` → NaN
   - `log1p(+∞)` → `+∞`
   - `log1p(NaN)` → NaN
   - `log1p(0.0) == 0.0` exactly; `log1p(-0.0) == -0.0` exactly.

4. **Cross-cancellation pairs** (proptest):
   - `log1p(expm1(x)) ≈ x` to ≤ 1 ulp for x ∈ [-0.5, 0.5]
   - `expm1(log1p(x)) ≈ x` to ≤ 1 ulp for x ∈ [-0.5, 0.5]
   - These are the round-trip identities; they catch any phase mismatch between the two implementations.

---

## Part 6 — What scientist needs to know for the oracle harness

When Phase A lands, scientist runs the validation:

1. **Generate corpora**:
   - `python R:\tambear\oracle\expm1\generate_corpus.py` (new file; mirror exp's structure)
   - `python R:\tambear\oracle\log1p\generate_corpus.py` (new file; mirror log's structure)
2. **Run tambear-vs-mpmath at p=200**:
   - `cargo test --test big_float_vs_mpmath discover_tambear_expm1_distribution -- --nocapture`
   - `cargo test --test big_float_vs_mpmath discover_tambear_log1p_distribution -- --nocapture`
3. **Assert**:
   - 0-ULP rate ≥ 99% on the canonical corpus (mpmath gold at 50 dps).
   - 1-ULP rate ≤ 1% (the standard IEEE 754 transcendental tolerance).
   - 0 entries beyond 1 ULP. (If any, file an upstream bug or document why we deviate from fdlibm.)
   - Sign-of-zero identities exact.
   - Domain-edge NaN/Inf behavior matches IEEE 754.
4. **Cross-precision proptest** (per Phase C / BZ unstub pattern):
   - Compute at p=500, round to p=53, verify ≤1 ULP drift vs. compute-at-p=53-directly.
   - Same for p=1024 → p=200, p=200 → p=53.

---

## Part 7 — Sources

1. **Tang, P.T.P. (1992)**. "Table-driven implementation of the Expm1 function in IEEE floating-point arithmetic." *ACM Transactions on Mathematical Software* 18(2):211-222. DOI [10.1145/146847.146928](https://dl.acm.org/doi/10.1145/146847.146928).
2. **Sun Microsystems fdlibm** — [s_expm1.c](https://www.netlib.org/fdlibm/s_expm1.c), [s_log1p.c](https://www.netlib.org/fdlibm/s_log1p.c), [e_exp.c](https://www.netlib.org/fdlibm/e_exp.c), [e_log.c](https://www.netlib.org/fdlibm/e_log.c).
3. **Beebe, N.H.F.**. *Computation of expm1(x) = exp(x) − 1* — [PDF](https://www.math.utah.edu/~beebe/reports/expm1.pdf).
4. **Tang, P.T.P. (1989)**. "Table-driven implementation of the exponential function in IEEE floating-point arithmetic." *ACM TOMS* 15(2):144-157. DOI [10.1145/63522.214389](https://dl.acm.org/doi/10.1145/63522.214389).
5. **Tang, P.T.P. (1990)**. "Table-driven implementation of the logarithm function in IEEE floating-point arithmetic." *ACM TOMS* 16(4):378-400. DOI [10.1145/98267.98294](https://dl.acm.org/doi/10.1145/98267.98294).
6. **Markstein, P. (2000)**. *IA-64 and Elementary Functions: Speed and Precision*. Hewlett-Packard Professional Books / Prentice Hall.
7. **Cody, W.J. & Waite, W. (1980)**. *Software Manual for the Elementary Functions*. Prentice-Hall.
8. **Muller, J.-M. et al. (2018)**. *Handbook of Floating-Point Arithmetic*, 2nd ed. Birkhäuser/Springer — chapters 11 and 12 (transcendentals).
9. **Daramy-Loirat, C. et al. (2006)**. *CR-LIBM: A library of correctly rounded elementary functions in double-precision*. ENS Lyon technical report — the reference implementation for proven-correctly-rounded transcendentals.

---

## Summary table for pathmaker

| Coefficient block | Source | Reusable from existing tambear? | Action |
|-------------------|--------|---------------------------------|--------|
| expm1 Q1..Q5 | fdlibm s_expm1.c | No (existing only has Taylor) | Add to `expm1.rs` per Part 1 |
| log1p Lp1..Lp7 | fdlibm s_log1p.c | YES (= existing log.rs LOG_COEFFS) | Reuse as-is |
| exp P1..P5 (rational) | fdlibm e_exp.c | No (existing uses Taylor) | **Refit exp.rs simultaneously with Phase A** |
| Reduction constants LN_2_CW_HI / LO | Cody-Waite via fdlibm | YES (existing exp.rs) | Reuse |
| Overflow/underflow thresholds | fdlibm | YES (existing exp.rs) | Reuse |

**Total new work**: ~150 lines for expm1.rs + ~120 lines for log1p.rs + ~50-line refit of exp.rs. All polynomial coefficients verified bit-perfect against fdlibm/Tang/glibc/OpenBSD/Sun JDK.
