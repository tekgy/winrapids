# Trig Family References

> Full citation list backing `catalog.md`. Primary sources (original papers),
> canonical implementations, and state-of-the-art correctly-rounded work.
> Every function in the catalog should resolve at least one `[Primary]` and
> one `[Implementation]` citation here.

---

## Historical / Primary Sources

### Āryabhaṭīya (499 CE)
Āryabhaṭa. *Āryabhaṭīya*. The earliest known tabulation of sine values
(as the "jyā" half-chord). Table of 24 sine values at 3.75° intervals.

### Mādhava of Sangamagrāma (~1400 CE)
Kerala school series expansions for sin, cos, arctan — centuries before
Gregory / Newton. Primary reference via Whish 1834 and later Rajagopal
& Rangachari 1951.

### Euler, L. (1748)
*Introductio in analysin infinitorum*. Lausanne. Formalizes inverse trig
via the complex-logarithm identity.

### Riccati, V. (1757)
*Opusculorum ad res physicas et mathematicas pertinentium*. Introduces
hyperbolic sine, cosine, tangent as circular analogs for hyperbolae.

### Gregory, J. (1671)
Manuscript letter to Collins containing the `atan` power series
(independent of Mādhava).

### Machin, J. (1706)
Machin's formula: `π/4 = 4·atan(1/5) − atan(1/239)`. First fast
converging series for computing π to many digits via inverse tangents.

### de Mendoza y Ríos, J. (1795)
Introduces the haversine function for navigation. *Memoria sobre algunos
métodos nuevos de calcular la longitud por las distancias lunares.*

### Inman, J. (1835)
*Navigation and Nautical Astronomy, for the Use of British Seamen.* Tables
of haversine, versine, coversine, exsecant.

---

## Modern Algorithmic Foundations

### Cody, W. J., & Waite, W. (1980)
*Software Manual for the Elementary Functions.* Prentice-Hall. The
foundational reference for Cody-Waite range reduction; defines the
split-precision π/2 constants used throughout modern libm.

### Payne, M. H., & Hanek, R. N. (1983)
"Radian reduction for trigonometric functions."
*ACM SIGNUM Newsletter* 18(1), pp. 19–24.
The algorithm for reducing huge arguments using a multi-precision `2/π`
table. Still the basis of every IEEE-correct large-argument reduction.

### Kahan, W. (1987)
"Branch Cuts for Complex Elementary Functions, or Much Ado About
Nothing's Sign Bit."
In *The State of the Art in Numerical Analysis*, Oxford UP.
Defines sign-of-zero conventions, `atan2` edge cases, pi-scaled trig.

### Ng, K. C. (1992)
"Argument Reduction for Huge Arguments: Good to the Last Bit."
Sun Microsystems internal report. Explains the fdlibm implementation of
Payne-Hanek with 1200-bit 2/π table.

### Tang, P. T. P. (1992)
"Table-Lookup Algorithms for Elementary Functions and Their Error
Analysis." *IEEE Trans. Comput.* 41(8), pp. 993–1002.
The table-based approach used by fdlibm for `atan` and hyperbolics.

### Markstein, P. (2000)
*IA-64 and Elementary Functions: Speed and Precision.* Prentice Hall.
ch. 8 covers hyperbolics; ch. 10 covers inverse trig. Source of the
FMA-friendly polynomial evaluation patterns.

### Muller, J.-M. (2016)
*Elementary Functions: Algorithms and Implementation* (3rd ed.).
Birkhäuser. **The** reference for elementary function implementation —
range reduction, polynomial approximation, special cases.

### Muller, J.-M., Brisebarre, N., de Dinechin, F., Jeannerod, C.-P.,
Lefèvre, V., Melquiond, G., Revol, N., Stehlé, D., Torres, S. (2018)
*Handbook of Floating-Point Arithmetic* (2nd ed.). Birkhäuser.
Tables of hard cases for correctly rounded evaluation. Tbl 11.5 is the
canonical worst-case sin/cos argument list.

### Remez, E. (1934)
"Sur la détermination des polynômes d'approximation de degré donnée."
*Communications de la Société Mathématique de Kharkov* 10, pp. 41–63.
The exchange algorithm underlying every minimax polynomial fit in tambear.

---

## Canonical Implementations

### Sun fdlibm (1993)
Sun Microsystems' freely distributable math library. Authors: K. C. Ng et al.
- `__ieee754_rem_pio2.c` — Payne-Hanek range reduction.
- `__kernel_sin.c`, `__kernel_cos.c` — degree-13 odd / degree-14 even
  minimax polys on [-π/4, π/4].
- `__kernel_tan.c` — tan/cot dual-branch kernel.
- `s_asin.c`, `e_acos.c` — three-region split strategy.
- `s_atan.c` — table + polynomial for `atan`.
- `e_atan2.c` — full four-quadrant with IEEE edge cases.
- `s_sinh.c`, `s_cosh.c`, `s_tanh.c` — expm1-based piecewise.
- `s_asinh.c`, `e_acosh.c`, `s_atanh.c` — log1p-based inverses.
http://www.netlib.org/fdlibm/

### glibc libm
Derivative of fdlibm with Ziv's adaptive precision; IBM Accurate
Mathematical Library (AMS) replacement for transcendental functions.

### CORE-MATH — Sibidanov, Zimmermann, Glondu et al. (2022+)
"Correctly-Rounded Double-Precision Math Library."
https://core-math.gitlabpages.inria.fr/
Correctly-rounded f64 implementations of ~80 elementary functions.
Specific papers:
- Sibidanov, A., Zimmermann, P., Glondu, S. (2022). "The CORE-MATH
  Project." *29th IEEE Symposium on Computer Arithmetic (ARITH)*,
  pp. 26–34.
- Sibidanov, A., Zimmermann, P. (2023). "A correctly rounded binary64
  cube root implementation." arXiv:2304.07057.

### Intel SVML (Short Vector Math Library)
Proprietary vector math library. 1-ULP `sin`/`cos`/`tan` etc. across
AVX / AVX-512. Reference via Intel ICC manuals.

### CUDA Math Library
NVIDIA's device math library. Documented in *CUDA C++ Programming
Guide*, Appendix E. `__sinf` = 2-ULP intrinsic; `sin` = 1-ULP software.
Error bounds catalogued function-by-function.
https://docs.nvidia.com/cuda/cuda-math-api/

### ARM Optimized Routines
arm/arm-optimized-routines (GitHub). Scalar and SIMD libm for AArch64;
~2-4 ULP target.

### Julia Base (math.jl)
Julia's Base math library. Uses fdlibm coefficients with modifications
for the two-step Payne-Hanek threshold at 2²⁵ (vs. fdlibm's 2²⁰).
Source: julia/base/math.jl.

### MATLAB, Mathematica
Proprietary; documented behavior via reference manuals. MATLAB's older
`atan2(0, -0)` bug documented in MATLAB Central forum archives.

### R base::math
Wraps the host system's libm. Pi-scaled `sinpi` / `cospi` / `tanpi`
added in R 3.0 (2013).

### Boost.Math (C++)
Ships hyperbolic reciprocals (`coth`, `sech`, `csch`) and archaic trig
(`haversine`, `versin`) as first-class. Reference implementation for
the less-common functions.

---

## Standards

### IEEE Std 754-2019
*IEEE Standard for Floating-Point Arithmetic.*
§9.2 "Recommended operations" specifies `sin`, `cos`, `tan`, `sinPi`,
`cosPi`, `tanPi`, `atan`, `atan2`, `asin`, `acos`, `sinh`, `cosh`,
`tanh`, `asinh`, `acosh`, `atanh` and their edge cases.
Table 9.1 enumerates `atan2` edge cases (~20 distinct rules).

### ISO C23 (draft N3096)
Annex F.10.1.* specifies semantics for all standard math functions;
new to C23: `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi`,
`atan2pi`.

### POSIX.1-2017
Inherits C standard math semantics; adds `sincos` as an extension
(not in C standard).

---

## Error Analysis and TMD Tables

### Kahan, W. (1983)
"Mathematics written in Sand — the HP-15C, Intel 8087, etc."
*Proceedings of the 1983 Statistical Computing Section*, ASA.
Early analysis of hard cases for transcendental functions.

### Lefèvre, V., Muller, J.-M., Tisserand, A. (1998)
"Toward Correctly Rounded Transcendentals."
*IEEE Transactions on Computers* 47(11), pp. 1235–1243.
The paper that made the Table Maker's Dilemma tractable via lattice
reduction.

### Lauter, C. (2008)
"Arrondi correct de fonctions mathématiques." PhD thesis, ENS Lyon.
The dissertation underlying much of CORE-MATH's approach.

---

## Specialty / Domain-Specific

### Haversine for geodesy
de Mendoza y Ríos 1795; Inman 1835; modern: Sinnott, R. W. (1984).
"Virtues of the Haversine." *Sky & Telescope* 68(2), p. 159.

### atan2 bug history
Hauser, J. R. (1996). "Handling Floating-Point Exceptions in Numeric
Programs." *ACM TOPLAS* 18(2) — enumerates real bugs including an
`atan2` zero-sign bug in SPARC libm of that era.

### Chen, M. (1992)
"Fast Trigonometric Function Evaluation Using Taylor-like Series."
MIT LCS Technical Report. Source of fdlibm's `__kernel_tan` polynomial
form.

---

## Internal WinRapids References

- `crates/tambear/src/recipes/libm/sin.rs` — tambear's forward radian
  sin/cos implementation (commit 0bbae82).
- `crates/tambear/src/recipes/libm/exp.spec.toml` — the spec pattern
  pathmaker uses (commit a1a2682).
- `docs/architecture/atoms-primitives-recipes.md` — the three-layer
  decomposition; catalog entries compose primitives from this catalog.
- `CLAUDE.md` — the Tambear Contract; every recipe above must pass
  the Filter Test before ship.
