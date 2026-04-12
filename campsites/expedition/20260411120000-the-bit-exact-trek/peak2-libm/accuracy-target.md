# Campsite 2.1 — The Accuracy Target

**Owner:** math-researcher
**Status:** CLOSED — all four sign-offs complete (navigator, team-lead, adversarial, scientist)
**Date:** 2026-04-11
**Closed:** 2026-04-12

---

## The decision

**tambear-libm Phase 1 targets: `max_ulp ≤ 1.0` (faithful rounding) for every non-trivial transcendental, measured over 1,000,000 random fp64 samples drawn from each function's declared primary domain, with targeted adversarial batteries on top of the random samples.**

`tam_sqrt` is the one exception — it's a direct lowering to IEEE 754 `fsqrt`, which the standard itself guarantees is correctly rounded (`max_ulp = 0`). We assert that stronger bound for sqrt specifically.

**Auxiliary published metrics** (recorded per function, not a pass/fail bar):
- `mean_ulp` — expected drift on typical inputs
- `stddev_ulp` — spread
- `p99_ulp` — tail behavior
- `max_ulp_by_decade` — max error bucketed by `floor(log2(|x|))`, so we see if there's a pathological regime

Plus the adversarial categories below must each hit their specified behavior, not just their ULP bound.

---

## Why not correctly rounded (0.5 ULP)?

Correctly rounded is the gold standard. CRlibm, MetaLibm, and the Sun `libmcr` project aim for it. For every input, the returned fp64 is the unique fp64 that lies closest to the true mathematical answer.

It's also **very expensive**, and the cost is structural — not something we can buy back with cleverer constants:

1. **The Table Maker's Dilemma.** For a function like `exp` there's no a priori upper bound on how many bits past the fp64 mantissa you need to look at before you can decide the correct rounding direction. In the worst case, the true value is `...1 0000000000000000 xxx...` where the zeros run for 50+ bits. You cannot decide which way to round without computing far more than 53 bits internally.
2. **Implication: extended-precision intermediates everywhere.** Correctly-rounded libms do the polynomial evaluation in double-double (≈106 bits), or use integer-mantissa reconstruction, or carry an error term through every op. Every one of those blows up the Horner sequence into a double-double Horner sequence — 3–5× more `fmul`/`fadd` per op — AND requires FMA-like compensated-sum primitives to keep the extra precision alive. In an I3-compliant world where we explicitly forbid contraction, compensated sums still work (Dekker, Kahan, TwoSum, TwoProduct) but they're yet more explicit ops per step.
3. **Implication for our pipeline specifically.** Every correctly-rounded libm function becomes 4–10× more expensive in the interpreter. Peak 5 (interpreter) is already the slowest backend on the trek — correctly rounded would make it genuinely painful. And the goal of Phase 1 is **cross-backend agreement**, not peer-matching CRlibm.

We can add a `tam_exp_cr` variant later for users who need it, following the CRlibm / SLZ hard-to-round argument tables. That's a Phase 2 decision, not Phase 1.

## Why not looser (≤3–4 ULPs)?

GPU vendor math libraries typically publish 2–4 ULP bounds on their transcendentals. OpenCL's precision table lists 3 ULPs for `sin`/`cos`/`exp`/`log`. That's fine for graphics and for a lot of ML work. It's *not* fine for two things that matter to tambear:

1. **Statistical code paths.** When you fold `log x` into a likelihood or an entropy estimate and then into a numerical optimizer, 3-ULP drift on each call becomes 10–20 ULPs after a chain of compositions, and the optimizer's stopping criterion starts to depend on which backend it ran on. Tambear's correctness story is that the *same recipe* produces the *same answer* cross-backend. A 4-ULP per-call budget destroys that at the second decomposition.
2. **The invariant budget for cross-backend diff (Peak 4).** Peak 4's replay harness asserts bit-exact for pure arithmetic and `within_ulp` for transcendental kernels. The allowed end-to-end tolerance is a sum of per-call tolerances. If `tam_exp` is 4 ULPs and a kernel calls it 1000 times, the tolerance explodes to 4000 ULPs, which is roughly "the test verifies nothing." At 1 ULP per call, we can keep the per-kernel bound tight enough to catch real drift.

## Why 1 ULP is achievable (evidence)

1-ULP faithful rounding is well-established for the Phase 1 function set when implemented carefully. The Tang 1989 paper on `exp` reports `max_ulp ≤ 0.54` using a table-driven polynomial; his `log` paper (ACM TOMS 16:378, 1990) reports `max_ulp ≤ 0.57`. Sleef's portable libm publishes 1-ULP variants of all Phase 1 functions. The "1 ULP at p99 and p100 across 1M random inputs" bar is the actual working bar of most production portable libms, and it does not require double-double intermediates for the target functions.

The pieces that make 1 ULP achievable without double-double:
- Cody-Waite dual-constant range reduction (split `ln 2` or `π/2` into a high part with trailing zeros and a low part that carries the rest — the high-part subtraction is exact).
- Minimax polynomials computed to at least 64 bits of precision offline, stored as fp64 literals with bit-exact encoding.
- Horner evaluation in a fixed, non-reassociated order — every backend emits the same op sequence.
- Careful reassembly that doesn't flush subnormals to zero (see Campsite 2.8) and doesn't hand a denormal to a `mul` that would lose bits.

The nuances show up at the function boundary: `sin(x)` for `|x| > 2^20` requires Payne-Hanek to keep 1 ULP. Phase 1 does not claim 1 ULP outside `[−2^20, 2^20]` for trig. Phase 2 will.

---

## The measurement protocol

Every Phase 1 libm function must, before it is declared done, pass the following battery through the ULP harness from Campsite 2.3.

### Random fp64 sampling (1,000,000 samples)

**Sampling scheme:** random samples are drawn uniformly over the *bit encoding* of fp64 numbers inside the declared domain, not uniformly over real-valued intervals. Concretely:
- Pick a random sign (if the domain includes both).
- Pick a random exponent uniformly from the valid exponents inside the domain.
- Pick a random 52-bit mantissa.
- Reject and resample if outside the domain.

This exponent-uniform sampling is important because it gives equal weight to `1e-10` and `1e10`, whereas real-uniform sampling would let `1e10` dominate by 20 orders of magnitude and leave small arguments essentially untested.

### Per-function primary domain

| Function | Primary domain (Phase 1 claim) | Sampling method |
|---|---|---|
| `tam_sqrt` | `[0, fp64_max]` | exponent-uniform |
| `tam_exp`  | `[-745.13, 709.78]` (full fp64 representable output range) | exponent-uniform |
| `tam_ln`   | `(0, fp64_max]` | exponent-uniform |
| `tam_sin`  | `[-2^30, 2^30]` | exponent-uniform (small) + real-uniform (large) |
| `tam_cos`  | `[-2^30, 2^30]` | same as sin |
| `tam_tan`  | `[-2^30, 2^30]` (excluding pole neighborhoods) | exponent-uniform (excluding `|cos(x_f64)| < 2^-26`) |
| `tam_pow`  | `a ∈ [2^-100, 2^100]`, `b ∈ [-30, 30]` | exponent-uniform |
| `tam_tanh` | `[-fp64_max, fp64_max]` | exponent-uniform |
| `tam_sinh` | `[-710, 710]` | exponent-uniform |
| `tam_cosh` | `[-710, 710]` | exponent-uniform |
| `tam_atan` | `[-fp64_max, fp64_max]` | exponent-uniform |
| `tam_asin` | `[-1, 1]` | real-uniform |
| `tam_acos` | `[-1, 1]` | real-uniform |
| `tam_atan2`| `(y, x) ∈ [-10, 10]^2` ∪ signed-zero corners | real-uniform + corner inject |

For functions with a natural symmetry (`sin(-x) = -sin(x)`, etc.), the sampling intentionally includes both signs so the symmetry is probed.

### Per-function ULP exceptions (documented tighter or looser bounds)

| Function | Phase 1 bound | Rationale | Phase 2 upgrade path |
|---|---|---|---|
| `tam_sqrt` | **0 ULP** | IEEE 754 mandates correct rounding for `fsqrt`. Our op lowers directly to the hardware `fsqrt.f64`. Tighter than the default. | N/A — already maximal. |
| `tam_atan2` | **2 ULP** | The `y/x` division step introduces 0.5 ULP, composing with `tam_atan`'s 1 ULP to ~2 ULP worst case. Reaching 1 ULP requires TwoSum/TwoProd double-double arithmetic on the quotient, which is infrastructure that belongs in Phase 2. Team-lead approved this exception 2026-04-12. | Phase 2 replaces the `y/x` step with a Dekker-style double-double computation; the composed bound drops to 1 ULP. |
| `tam_pow` | **2 ULP** | The composed `exp(b · log(a))` path accumulates ~1 ULP from `log`, ~0.5 ULP from the multiplication `b · log(a)`, and ~1 ULP from `exp` — total ~2 ULP worst case. Reaching 1 ULP requires Dekker-style double-double on the `b · log(a)` intermediate, infrastructure that belongs in Phase 2. Adversarial's special-values matrix assigns 2 ULP for pow 2026-04-11; I concurred 2026-04-12. | Phase 2 upgrades via TwoSum/TwoProd on the `log(a) · b` intermediate per the pow-design.md §12 research. |
| `tam_tan` | **2 ULP** (with pole exclusion) | tan has poles at `x = (k + 1/2)·π` where it diverges. Near poles the argument sensitivity makes 1-ULP tan structurally unreachable: a 1-ULP error in `x` translates to an unbounded error in `tan(x)` when `cos(x)` is near zero. The Phase 1 implementation `tan = sin/cos` gives ~2 ULP composed. **The 2-ULP bound is scoped to the pole-exclusion zone `|cos(x_f64)| ≥ 2^-26`**; inputs where `|cos(x_f64)| < 2^-26` are flagged by the oracle runner and do not count against the ULP bar (they cannot silently pass either — they're flagged, not ignored). This is analogous to pow's 2 ULP exception but for a different compositional reason. Navigator ruling 2026-04-12. | Phase 2 could use a dedicated tan polynomial with wider internal precision; Phase 3 (CRlibm-class) would exhaustively analyze worst-case rounding near poles. Both are out of Phase 1 scope. |

**All other functions: 1 ULP** per the default policy above. Specifically: `tam_exp`, `tam_ln`, `tam_sin`, `tam_cos`, `tam_tanh`, `tam_sinh`, `tam_cosh`, `tam_atan`, `tam_asin`, `tam_acos` all target ≤ 1 ULP on 1M random samples in their primary domains.

### Adversarial battery (always on top of random)

Each function must also pass the following named categories. These are **injected** into the random sample stream, not relied on to arise by chance:

1. **Special values** — the IEEE 754 specials for the function. `exp(0) = 1`, `exp(-inf) = 0`, `exp(+inf) = +inf`, `exp(nan) = nan`. `log(1) = 0` exactly. `log(0) = -inf`. `log(-x) = nan` for `x > 0`. `sin(0) = 0` exactly. `pow(0,0) = 1`. `atan2(0,0) = 0`. Full table per function in the design doc.
2. **Subnormal domain** — at least 10,000 samples whose input *or whose expected output* is in the subnormal range. Catches flush-to-zero bugs at reassembly.
3. **Underflow edge** — samples whose true result is at or near `fp64_min_normal ≈ 2.225e-308`.
4. **Overflow edge** — samples whose true result is at or near `fp64_max ≈ 1.798e+308`.
5. **Argument reduction boundary** — for functions with range reduction, samples whose reduced argument lands near 0 or near the boundary of the reduction interval. Catches Cody-Waite high/low split bugs.
6. **Polynomial boundary** — samples near ±(reduction interval boundary). If the polynomial was Remez-fit on `[-ln(2)/2, ln(2)/2]` and we actually hand it `ln(2)/2 + ε`, does it still return 1 ULP?
7. **Near-zero inputs** — samples in `[2^-500, 2^-400]`. For `exp`, these must return `1 + x + x²/2 + ...` accurately, not collapse to `1.0`.
8. **Near-`1` inputs for `log`** — `log(1 + ε)` for tiny `ε`. Must return `ε - ε²/2 + ...`, not `0`.
9. **Identities that must hold** — for each function, the harness computes a dozen mathematical identities that should hold to ≤ 1 ULP given our individual claims:
   - `exp(log(x)) == x` for `x > 0` (within 2 ULPs — two 1-ULP errors compose)
   - `sin² + cos² == 1` (within 3 ULPs including cancellation)
   - `atan(tan(x)) == x` for `x ∈ (-π/2, π/2)`
   - `tanh(x) + tanh(-x) == 0` exactly (symmetry)
   - and so on.

Identity checks are **tertiary** — they're a sanity net, not the primary bar. The primary bar is individual-function ULP measurement against mpmath.

### Acceptance criterion (formal)

A Phase 1 libm function is accepted if and only if:

1. `max_ulp ≤ 1.0` across the 1M random sample set in its primary domain, AND
2. Every special-value test returns the specified IEEE 754 / convention result bit-exact, AND
3. The subnormal, overflow, and underflow adversarial categories produce the specified result (not a crash, not a spurious nan, not a flush-to-zero), AND
4. Identity tests hold within the composed budget.

If **any** of these fails, the function is not done. The remedy is to diagnose the root cause — not to raise the bound.

---

## What we do NOT claim

- **We do not claim correct rounding.** `max_ulp = 1.0` means our answer could be one ULP off from the truly-rounded result on some inputs.
- **We do not claim this bound outside the declared primary domain.** Big-argument trig (`|x| > 2^20`) is out of spec until Payne-Hanek lands.
- **We do not claim 1 ULP for `tam_atan2` in Phase 1.** The `y/x` division composes with atan's 1-ULP error to reach ~2 ULP. The exception is documented above and in the per-function exceptions table; Phase 2 addresses it via double-double on the quotient.
- **We do not claim 1 ULP for `tam_pow` in Phase 1.** The composed `exp(b · log(a))` path reaches ~2 ULP worst case for `|b| ≤ 30`. Phase 2 tightens via TwoSum/TwoProd on the intermediate.
- **We do not claim 1 ULP for `tam_tan` in Phase 1.** Pole sensitivity makes uniform 1 ULP structurally unreachable. The 2-ULP bound is scoped to the pole-exclusion zone `|cos(x_f64)| ≥ 2^-26`; inputs closer to poles are flagged by the oracle runner, not silently passed.
- **We do not claim that identity tests hold to 0 ULPs.** Composition of individually-correct functions can drift by a couple of ULPs; that's geometry of floating-point, not a bug.
- **We do not claim speed.** Phase 1 is correctness-first. The interpreter will run the `.tam` ops one at a time and it will be slow. Speed is Peak 5.5 (JIT) onward.

## What this locks in for the team

- Pathmaker: every polynomial must be fit to 64-bit-plus precision at generation time, and the coefficients committed as fp64 literals. No "eyeball the last two digits" coefficient work.
- Adversarial: the battery above is the **minimum**. You are asked and encouraged to find more categories that break us. Every new category found becomes a permanent regression test.
- Scientist: your mpmath comparisons are the primary oracle. When a function passes our bar, you lock it in with a stored reference file that future regressions run against.
- Observer: per-function ULP histograms go in the lab notebook.
- Navigator: if anyone proposes raising the bound to "get unblocked," route it to escalations.md. The bound is not advisory.

## Sign-off record

All sign-offs complete. Campsite 2.1 is CLOSED.

- [x] **navigator** — APPROVED 2026-04-11. Accuracy target correct and sufficient. IR dependency check added (ldexp.f64, f64_to_i32_rn must land before tam_exp code starts).
- [x] **team-lead** — APPROVED (per navigator check-in 2026-04-12, all four noted complete).
- [x] **adversarial** — APPROVED 2026-04-11. Four additions noted: (1) near-±1 sampling for asin/acos, (2) sign-symmetry category, (3) Cody-Waite exact-constant injection, (4) tan/pow ULP tension flagged for math-researcher resolution. None block the sign-off.
- [x] **scientist** — APPROVED 2026-04-11. Oracle verdict: mpmath at 50 digits gives ~34 decimal digits guard margin over fp64; reference error ~5×10⁻⁵¹ vs 1 ULP ~2.2×10⁻¹⁶. Even 32 digits would be sufficient; 50 is conservatively correct. atan2 at 2 ULP is honest: fdiv.f64 contributes ≤0.5 ULP, composing with tam_atan's 1 ULP gives ≤1.5 ULP → 2 ULP published bound. Both claims APPROVED.

Once these three sign-offs land in `navigator/check-ins.md`, Campsite 2.1 is closed and Campsite 2.2 can start.

---

## References (papers only — no libm source)

- P. T. P. Tang, "Table-driven implementation of the exponential function in IEEE floating-point arithmetic," ACM TOMS 15(2):144–157, 1989.
- P. T. P. Tang, "Table-driven implementation of the logarithm function in IEEE floating-point arithmetic," ACM TOMS 16(4):378–400, 1990.
- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980. (Source of dual-constant range reduction.)
- M. H. Payne & R. N. Hanek, "Radian reduction for trigonometric functions," ACM SIGNUM Newsletter 18(1):19–24, 1983.
- IEEE 754-2019, "IEEE Standard for Floating-Point Arithmetic." (Correct-rounding semantics, subnormal behavior, special values.)
- N. J. Higham, "Accuracy and Stability of Numerical Algorithms," 2nd ed., SIAM, 2002. (Error analysis framework for the ULP budget composition.)
