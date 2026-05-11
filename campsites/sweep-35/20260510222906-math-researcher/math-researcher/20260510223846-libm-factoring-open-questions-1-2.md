# libm-factoring-open-questions-1-2

Created: 2026-05-10T22:38:46-05:00
By: math-researcher

---

# Libm-factoring open questions #1 (pow) and #2 (hypot) — math-researcher's walk

**Source**: `R:\winrapids\docs\architecture\tambear-libm-factoring.md` § "Open questions for math-researcher walk-through".

**Status**: My initial position is now in. These are the two Phase-C-blocking questions from the design doc. Pulling on them now while pathmaker drives Phase A; they will be load-bearing when sinh/cosh/tanh/hypot/pow land.

---

## Open Question #1 — `pow(x, y)` factorization

**The question**: The composed identity `pow(x, y) = exp(y · log(x))` introduces an additional source of error (the multiplication `y · log(x)` happens at recipe-level, not inside the shared kernel). Does `pow` deserve its own kernel state, or is the composed form sufficient at the precision tiers tambear targets?

### Position

**Answer: composed form is correct at every precision tier — BUT the composition must use the kernel-state's high/low components, not just the f64 outputs.**

### The error analysis

Naive composition: `pow(x, y) = exp(y · log(x))`.

Decomposing error sources:
1. `log(x)` has error ≤ ε_log (target ≤ 1 ulp at f64).
2. `y · log(x)` has rounding error ε_mul ≤ 1 ulp.
3. `exp(·)` of the multiplied value amplifies the input error: `|exp(z + δ) - exp(z)| ≈ exp(z) · |δ|`. So a δ-error in the exponent multiplies into a relative δ-error in the output.

**Worst-case total**: at f64 with naive composition, the error after `exp(y · log(x))` can reach ~2 ulps relative, plus the amplification factor that depends on `y`. For large `|y|`, the amplification of the `y · log(x)` rounding error can make pow's error grow proportionally to `|y|` — exactly the same Tang-style k-multiplier degradation that haunts MSVC's exp at large positive x.

### Why this argues FOR the kernel-state design, not against it

The fix: **the multiplication `y · log(x)` must be carried as a DoubleDouble (high/low), not collapsed to f64**. Then `exp(·)` of the DoubleDouble argument can recover ~1 ulp accuracy.

Concretely:
1. `LogKernelState(x) = (k_log, f_log, log1p_f_log)` — gives `log(x) = k_log·ln(2) + log1p_f_log` at DD precision.
2. **At pow's recipe layer**, compute `y · log(x)` as a DD product: `y_log_x_dd = dd_mul(y, log_kernel_state.to_dd())`.
3. Reduce `y_log_x_dd` to `(k_exp, r_dd)` per the exp range-reduction.
4. **Pull from ExpKernelState** if `r_dd` matches a cached entry (unlikely for arbitrary pow inputs); otherwise compute `expm1(r_dd)` fresh.
5. Final: `pow(x, y) = (1 + expm1(r_dd)) << k_exp`.

**This is the composed form** — pow does NOT have its own kernel-state. It composes from LogKernelState + ExpKernelState. But the composition is at *DoubleDouble precision*, not collapsed to f64.

### Why this is the right answer per the architecture

The libm-factoring frame explicitly forbids per-function kernel states; it's the whole point. The trig family proved the principle: 5+ functions from one TrigKernelState. The exp/log family extends it: 8+ functions from ExpKernelState + LogKernelState. Pow being a *composition* of those two states (not a third state) is what makes the catalog finite.

The cost: pow's recipe wrapper has to be careful at the DD-multiplication step. This is a recipe-tier concern, not a kernel-tier one. The kernels stay pure.

### Architectural implication for Phase B (ExpKernelState design)

`ExpKernelState` and `LogKernelState` should both expose their `(k, r_hi, r_lo, expm1_r_hi, expm1_r_lo)` or analogous DD-pair fields. The recipe wrappers can then compose at DD precision. **A flat-f64 ExpKernelState (collapsing r_hi + r_lo to a single f64) breaks pow.** This is a structural constraint on Phase B.

### Adversarial inputs for pow

These should drive pow's oracle harness:
1. **Large-y inputs**: `pow(1.1, 1000)` — the y · log(x) multiplication amplifies log's error by 1000×. A naive f64 composition fails here.
2. **Near-1 base**: `pow(1.0 + ε, y)` for tiny ε — `log(1 + ε) ≈ ε - ε²/2`, so accuracy depends on log1p, not log.
3. **Near-0 base**: `pow(2^-1000, 0.5)` — must use log's correct k extraction.
4. **Integer y**: `pow(2.0, 53)` — should be EXACT (special case detection); naive composition would give 9007199254740992.0000...001 rather than 9007199254740992.0.
5. **Half-integer y**: `pow(x, 0.5) = sqrt(x)` — should equal sqrt(x) to ≤ 1 ulp.
6. **Negative base, integer y**: `pow(-2.0, 3) = -8.0`; for non-integer y, `pow(-2.0, 0.5)` should return NaN per IEEE 754. Special case detection.
7. **Domain edges**: `pow(0.0, 0.0) = 1.0` (IEEE 754 convention); `pow(0.0, -1) = +∞`; `pow(NaN, 0.0) = 1.0` (IEEE 754); etc. There's a ~20-entry edge case table for pow per C99 Annex F.

### Verdict on pow factorization

**Composed form via LogKernelState + ExpKernelState at DD precision.** No dedicated pow kernel state. The Phase C wrapper has explicit DD multiplication at the recipe layer. Integer-exponent special cases are detected and handled with bit-exact bit-shifting / repeated multiplication (the "binary exponentiation" path that gives exact answers when y is a representable integer).

---

## Open Question #2 — `hypot(a, b)` as complementary-argument transform

**The question**: Past-Claude listed it as an instance, but `hypot` doesn't have an obvious "fixed point" the way log1p does. Is the meta-primitive's group structure broader than the April 13 essay assumed, or is `hypot` a different shape that just shares the *precision-preservation* property?

### Position (revised 2026-05-10 in light of naturalist's three-shapes finding)

**Answer: hypot is Shape 3 (structural-rewrite) in naturalist's three-shapes taxonomy, NOT a continuation of Shape 1 (input-side) or Shape 2 (output-side) transforms.**

**Convergence note**: I initially walked this question with a univariate→bivariate extension of the (F, G) framework — "the fixed point becomes a submanifold, the group becomes a scaling action." Naturalist's parallel garden entry (`~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md`) names the better structural distinction: hypot doesn't fit the (F, G) parameterization in any form because **Shape 3 collapses the transform/core/inverse sequence** — there's no separate "transform→stable_evaluation→inverse_transform" structure; there's just a different algorithm altogether. Multivariate functional identities (`max·√(1+(min/max)²)`), recursive identities (`asin near 1` via half-angle), and per-region rewrites (fdlibm e_hypot's high/low split) are not transforms of a univariate kernel — they're algorithm-shape changes.

My initial framing (submanifold fixed-point + scaling group) was *consistent* with the algorithm but obscured the structural distinction. Naturalist's framing is sharper: the (F, G) parameterization is the language of Shapes 1 and 2; Shape 3 has its own parameterization (regime_dispatcher, alternative_recipe_per_regime).

I am updating my position to match naturalist's. The four-axis coordinate framework from the second addendum (problem-topology / fix-shape / sharing-layer / precision-parameter-binding) is the right framing.

**Original analysis below is preserved as historical record** — the (F, G) framework I sketched works but is at the wrong layer of abstraction; the better framing is naturalist's three-shapes.

### The structural analysis (under the corrected framing)

Hypot has:
- **Problem-topology**: overflow at large magnitude (a² + b² overflows for |a| > ~2^512); underflow at small magnitude. Both are present simultaneously in the function's domain.
- **Fix-shape**: Shape 3 (structural rewrite) — the recipe branches on magnitude and applies different formulas per regime. Not a transform of one underlying univariate kernel.
- **Sharing-layer**: none from the exp/log family. Possible future `UnitVectorState(max_abs, ratios)` sharing across hypot / atan2 / cabs / n-norm.
- **Precision-parameter-binding**: the scaling thresholds `2^500`, `2^-500`, `2^60` are tier-dependent. At BigFloat precision, they shift. F13.C antibody site.

The April 13 garden's framing — "every precision-preserving libm primitive is an instance of one meta-primitive (complementary-argument transform parameterized by F, G)" — was right *for Shapes 1 and 2 only*. For Shape 3 (hypot, half-angle reflections, gamma reflection), the unification is at a higher descriptive level: "every precision-preserving libm primitive avoids a numerically-problematic regime via algebraic restructuring" — but the structuring is per-shape-different.

The framework generalizes; hypot doesn't fit the "fixed point F + identity-element neighborhood" frame because hypot is *bivariate*. It fits a broader frame where the "fixed point" is the diagonal `{(a, a)} ⊂ R² × R²` and the "transform" is the scaling action that moves any (a, b) toward the diagonal via `(|a|, b/a)` if |a| ≥ |b|.

### The structural analysis

The meta-primitive frame (April 13 garden, expanded):
- **Univariate case** (expm1, log1p, sinpi, ...): F is a point in R (or R²); G is a 1-dim Lie group acting on R. The transform brings x to a neighborhood of F where polynomial evaluation is precision-safe.
- **Bivariate case** (hypot): F is a 1-dim submanifold of R² (the diagonal `{(a, b) : a = b}` or the axes `{(a, 0)}` and `{(0, b)}`). G is a 1-dim Lie group acting on R² (scaling-as-rotation). The transform brings (a, b) to a neighborhood of F where the magnitude formula is precision-safe.

Concretely, fdlibm's e_hypot.c (canonical algorithm; verified bit-perfect against Borges 2019 within tolerance):

1. **Swap so |a| ≥ |b|.** This puts the input in the upper triangle.
2. **Scale by 2^-600 if |a| > 2^500, or by 2^600/2^1022 if |b| < 2^-500.** This is the "transform" — moves the input to a scale where intermediate `a²` doesn't overflow.
3. **High/low split for precision recovery**:
   - If `|a| > 2·|b|`: compute `x1*x1 + (y*y + x2*(x+x1))` where `x1` is the top 32 bits of `x`, `x2 = x - x1`. The (x1 · x1) is exact in f64; the residual catches the rounding error.
   - If `|a| ≤ 2·|b|`: compute `t1*y1 + ((x-y)·(x-y) + (t1·y2 + t2·y))` — a different decomposition using `t1 = 2x`, `y1 = top of y`.
4. **sqrt** of the recombined precise sum.
5. **Rescale** to compensate for step 2 scaling.

### Why this is a complementary-argument transform

The "fixed point" F is the unit circle. The "transform" is the scaling action `(a, b) → (a/|a|, b/|a|)` followed by `|a|·sqrt(1 + (b/a)²)`. After the transform, the inner sqrt is evaluated on `[1, sqrt(2)]` — a small, well-conditioned interval. The "inverse transform" is the multiplication by |a|.

The framework absorbs hypot cleanly. The April 13 essay's "fixed point F" is over-specified for the univariate case; the more general structure is "F is a submanifold of fixed codimension where the function is precision-safe; G is the group action that contracts toward F."

### Implementation choice: tambear's hypot

Two candidates:
- **fdlibm e_hypot.c** — decades-validated, < 1 ulp, straightforward to port. Uses the scaling-and-high-low-split approach.
- **Borges 2019 / ACM Alg 1014** — newer, FMA-based, claimed faithfully-rounded with smaller constant overhead than fdlibm.

**Recommendation: ship fdlibm-style first, follow up with Borges later.**

Reasoning:
1. fdlibm hypot is straightforward to verify (Sun has had it deployed for 30 years; bit-tested against mpmath universally).
2. Borges depends on FMA (which tambear's primitives layer has, per the existing `compensated::dot::horner` using `f64::mul_add`).
3. The Borges algorithm is in literature but the canonical reference is paywalled; would need careful re-derivation if we ship without the paper text.
4. fdlibm's < 1 ulp is the same accuracy class as Borges; the speed difference is ~10-20% in throughput, not a precision matter.
5. **Phase C should ship fdlibm-style hypot to unblock the family; Sweep 36 or later can refactor to Borges if benchmarking shows the speedup matters.**

### Cross-implication: hypot does NOT use ExpKernelState

This is important. The libm-factoring frame might suggest hypot shares an intermediate with the exp/log family because all three are "magnitude / scaling" operations. But hypot's "log(|a|)" is implicit (via the exponent bit-shift), not computed via the log polynomial. The kernel state shared across hypot consumers (if any) would be `(|a|, b/a)` — the *unit-vector form* of the input.

Could a `UnitVectorState(a, b) = (max_abs, ratio_min_over_max)` be a useful intermediate for hypot, hypot3 (three-arg), norm of an n-vector, atan2, complex magnitude (`cabs`)? **Yes.** This is a candidate fourth kernel state alongside TrigKernelState, ExpKernelState, LogKernelState — but it's downstream of Sweep 35.

### Verdict on hypot

**Hypot is a complementary-argument transform with a 1-dim submanifold as fixed point. It does NOT use ExpKernelState. It may eventually use a `UnitVectorState` shared with atan2/cabs/n-norm — Sweep 36+ work.**

**Phase C action**: implement `hypot.rs` directly using fdlibm e_hypot.c as the reference. ~80 lines. Three-strategy lowering (strict/compensated/correctly_rounded) follows the same pattern as exp.rs and log.rs.

### Adversarial inputs for hypot

1. **Overflow regime**: `hypot(1e200, 1e200) = √2 · 1e200 ≈ 1.414e200` — naive a² + b² would overflow.
2. **Underflow regime**: `hypot(1e-200, 1e-200) = √2 · 1e-200` — naive a² + b² would underflow to 0.
3. **Very unequal**: `hypot(1e308, 1.0) = 1e308` exactly (the smaller term contributes 0 ulps at f64).
4. **Equal**: `hypot(1.0, 1.0) = √2` — exact value test.
5. **Edge cases per IEEE 754**:
   - `hypot(±∞, anything) = +∞` (even if other is NaN; symmetric in arguments).
   - `hypot(NaN, finite) = NaN` (unless first is ±∞).
   - `hypot(0, b) = |b|`; `hypot(a, 0) = |a|`.
   - `hypot(-0.0, -0.0) = +0.0`.
6. **Sterbenz pair**: `hypot(a, a) = a · √2` for any a in normal range; tests scaling correctness.
7. **The fdlibm threshold cases**: inputs constructed to land exactly at `2^500` and `2^-500` to verify the scaling-branch boundary.

---

## Cross-references

- These positions inform Phase C wrapper design when sinh/cosh/tanh/hypot/pow land.
- Pow's DD-multiplication-at-recipe-layer is a structural constraint on Phase B's ExpKernelState (must expose DD fields, not collapsed f64).
- Hypot suggests a fourth kernel-state candidate (`UnitVectorState`) — Sweep 36 work, not Sweep 35.
- The "general complementary-argument transform with submanifold fixed-point" frame is what aristotle should pressure-test during their pressure-test pass; the meta-primitive in the April 13 garden was over-specified for the univariate case.

## Sources

- **Borges, C.F. (2019)**. "An Improved Algorithm for hypot(a,b)" — [arXiv:1904.09481](https://arxiv.org/abs/1904.09481); ACM Algorithm 1014, *TOMS* 47(1), DOI [10.1145/3428446](https://dl.acm.org/doi/10.1145/3428446).
- **fdlibm e_hypot.c** — [https://www.netlib.org/fdlibm/e_hypot.c](https://www.netlib.org/fdlibm/e_hypot.c). Canonical implementation; ~80 lines.
- **Ziv, A. (1991)** — "Fast evaluation of elementary mathematical functions with correctly rounded last bit." *ACM TOMS* 17(3):410-423. The iterative-approach reference for hard cases of pow.
- **Daramy-Loirat, C. et al. (2006)**. *CR-LIBM: A library of correctly rounded elementary functions in double-precision*. ENS Lyon. The reference for correctly-rounded pow.
- **C99 Annex F § 9.4** — special-case table for pow (the ~20-entry edge case table). [cppreference clog/pow](https://en.cppreference.com/w/c/numeric/math/pow).
