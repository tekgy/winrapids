# Downstream territory map — special functions beyond Sweep 35

*Scout, sweep-35 session, 2026-05-10*

---

## What I surveyed

Old winrapids tambear at `R:/winrapids/crates/tambear/src/recipes/libm/` has working
implementations + spec.toml files for essentially the full libm special-functions catalog.
This is the source material for the port survey. Every file there is a future sweep candidate
that needs to go through the tambear-native factored-kernel treatment.

---

## The shape of what exists in old tambear

Already implemented (`.rs` files with tests):
- **erf/erfc** — fdlibm four-region rational approximation (fdlibm s_erf.c)
- **gamma/lgamma** — Lanczos g=7 n=9, Pugh 2004 coefficients
- **sinh/cosh/tanh** — fdlibm-based; sinh uses small-argument polynomial
- **asinh/acosh/atanh** — log1p-based; all three fix the cancellation problem
- **exp** — old-style (not yet factored via ExpKernelState)
- **log** — old-style (not yet factored via log1p kernel)
- **sin/cos/tan/asin/atan** — Payne-Hanek + polynomial
- **pi_scaled** (sinpi, cospi, tanpi, etc.) — exact reduction
- **rare_trig** (sec, csc, cot, haversin, versin, gudermannian)
- **inv_recip** (acos, acot, asec, acsc via composition)

Spec.toml only (planned, not yet ported):
- **gudermannian / inv_gudermannian** — gd(x) = atan(sinh(x)), connects hyperbolic to circular
- **haversin / versin** — geospatial primitives
- **sincos / sincospi** — joint sin+cos via shared TrigKernelState (this is exactly the
  pattern Sweep 35's ExpKernelState is modeled on)

---

## The downstream sweep candidates (families not in old tambear either)

These are the families that come AFTER Sweep 35's exp/log territory. Ordered by structural
dependency and logical sweep grouping:

### Sweep 36 — Inverse trig + hyperbolic-inverse family (immediate next)

**What**: asin, acos, atan, atan2, plus all the hyperbolic inverses (asinh, acosh, atanh).
These are already in old tambear, but in old-style form. The port to new tambear, per
the factored-kernel design, means:

- **AsinKernelState** doesn't exist yet — but the key observation is that all six inverse
  trig functions share a sqrt-based intermediate: `sqrt(1 - x^2)` or `sqrt(1 + x^2)`.
  Same pattern as TrigKernelState for forward trig, ExpKernelState for exp/log.
- **Hyperbolic inverses** all compose log1p with sqrt — once log1p (Sweep 35) is in, these
  are thin wrappers. asinh, acosh, atanh all reduce to: compute a sqrt-based quantity,
  call log1p on it. This means Sweep 36 is a NATURAL follow-on to Sweep 35, not a separate
  sweep — acosh and atanh ship for near-free once log1p exists.

**Open question**: does inverse-trig merit its own kernel state, or are the shared
intermediates (sqrt(1-x^2), sqrt(1+x^2)) too cheap to cache? The trig family's argument
for TrigKernelState was about the expensive Payne-Hanek reduction being shared. For
inverse-trig, the sqrt is fast. Still: the asin/acos/atan composites all compute
`sqrt(1-x^2)` at some point — worth flagging for math-researcher.

### Sweep 37 — Error function family (moderate complexity)

**What**: erf, erfc, erfinv, erfcinv, and related.

**Shape**: Four-region rational approximation (fdlibm s_erf.c). Already implemented in
old tambear (erf.rs). The port is mechanical but has a precision story:

- **The key shared intermediate** for this family: the `x^2` computation and the
  `exp(-x^2 - 0.5625)` in regions 3-4. When erfc2 calls exp(-z^2-0.5625), that's an
  ExpKernelState consumer. Once Sweep 35 lands, erfc's implementation gets a precision
  upgrade for free via the shared kernel.
- **erfinv** is NOT in old tambear yet — it's a structural gap. It needs a Newton iteration
  seeded by a rational approximation. Different kingdom shape (B, not A), honest Fock
  declaration required.
- **The complementary-argument pattern**: erfc = 1 - erf is the classic cancellation trap
  (the bug the 2026-04-10 adversarial session found is documented in erf.rs). erfc is
  ALREADY an instance of the complementary-argument meta-primitive from past-Claude's
  April 13 essay — the same structural shape as erfc = 1 - erf (catastrophic cancellation
  fixed by dedicated formula). This is an instance that should be named in the
  complementary-argument-transform doc when that gets written.

**Why this is interesting for Sweep 35 cross-thread**: the erfc implementation's inner
`erfc2` function calls `(-z * z - 0.5625).exp()` — that's exactly the exp evaluated at
a compound reduced argument. Once ExpKernelState exists, this site becomes a natural
TamSession consumer. The erfc family's accuracy improves automatically when the shared
exp kernel is available.

### Sweep 38 — Gamma family (high complexity, different shape)

**What**: tgamma, lgamma, digamma, trigamma (polygamma more generally), beta, lbeta,
log_beta, incomplete gamma (igamma, igammac), incomplete beta (ibeta), regularized
incomplete beta (betainc).

**Shape**: Lanczos approximation + reflection formula for gamma/lgamma. Already in old
tambear (gamma.rs). But:

- **The Lanczos approximation is structurally OUTSIDE the complementary-argument frame** —
  this is open question #3 in the libm-factoring doc. Lanczos uses a rational approximation
  to the Gamma function directly; it doesn't have a "precision-safe base form" via a
  cancellation-fix transform. It's more like polynomial evaluation (kernel A) but with a
  reflection formula that composes sin(πx) — which means it pulls from the trig kernel.
  The structural dependency is: Gamma recipe consumes TrigKernelState (for reflection) and
  log (for the log-Gamma path). This makes it a downstream consumer of BOTH Sweep 35
  (exp/log) and the trig kernel.
- **digamma** is not in old tambear — it needs the polygamma recurrence and asymptotics.
  Kingdom B (recurrence) territory; honest Fock declaration required.
- **Incomplete gamma** is the really complex member — it requires either continued fractions
  (Kingdom B) or series expansion. TAM handles the iteration; the recipe declares it.

**Key architectural observation**: lgamma's inner loop computes `ag.ln()` where `ag` is
the Lanczos sum. This is a `log` call. Once Sweep 35 ships log, lgamma gets a precision
upgrade. Similarly, `PI_F64.sin()` in the reflection formula is a TrigKernelState consumer.
The gamma family sits at the intersection of BOTH prior kernel sweeps.

### Sweep 39 — Bessel functions (hard, structurally different)

**What**: J0, J1, Jn (Bessel first kind), Y0, Y1, Yn (second kind), I0, I1, In (modified
first kind), K0, K1, Kn (modified second kind).

**Shape**: Bessel functions don't fit the complementary-argument frame at all. They're
computed via:
1. Polynomial approximations near x=0 (small-argument expansion)
2. Asymptotic expansions for large x (which involve cos/sin composites)
3. Backward recurrence for arbitrary n (Miller's algorithm — Kingdom B, sequential
   downward recurrence that can't be parallelized)

The Bessel family is the first family that is genuinely HARD to factor via the Sweep 35
framework. The structural challenges:
- **Jn for large n**: upward recurrence is numerically unstable; backward recurrence (Miller)
  is stable but sequential (Fock boundary). This is genuinely Kingdom B territory.
- **Yn near x=0**: Y0 has a log(x) singularity that must be handled as a separate term.
  `Y0(x) = (2/π)(log(x/2) + γ)J0(x) + correction`. The log term is an ExpKernelState
  consumer; the J0 term is the Bessel recipe's own output. This is a structural dependency
  the current kernel framework handles, but the recipe is not a thin wrapper.
- **Modified Bessels (I, K)**: exponentially growing/decaying. I0 ~ exp(x)/sqrt(2πx) for
  large x. The scaled version `exp(-x)·I0(x)` is numerically stable and is what
  users usually want. The scaling factor is again an ExpKernelState consumer.

**The mempalace surfaced this**: reading notebook for Yakaboylu's Riemann-zeros-as-spectrum
paper mentions J0 appearing as the eigenfunction of the Bessel operator T̂. The Bessel
functions are also the eigenfunctions of a specific second-order ODE. This connection —
Bessel as eigenfunction of a natural operator — suggests the Bessel family eventually
deserves a separate recipe-tree with its own kernel shape, not just a polynomial
approximation ported from fdlibm.

### Sweep 40 — Polylogarithm + Hurwitz-Lerch family (very different shape)

**What**: Li_s(z) (polylogarithm), zeta(s) (Riemann zeta at real s), eta(s) (Dirichlet),
Hurwitz zeta ζ(s,a), Lerch transcendent Φ(z,s,a), dilogarithm Li_2.

**Shape**: These are SERIES functions. They converge via:
- Euler-Maclaurin summation formula (for Riemann zeta)  
- Reflection formulas
- Functional equations

This is the boundary of what the complementary-argument frame covers. The polylogarithm
Li_s(z) = Σ z^k/k^s diverges for |z| > 1 without analytic continuation — this means
complex arithmetic + branch cuts (DEC-032 territory). The first complex-analytic function
after complex_log (Sweep 35 Phase D) that we'll need to face fully.

**The dilogarithm** (Li_2) has a special structure worth noting: it satisfies
`Li_2(x) + Li_2(1-x) = π²/6 - ln(x)ln(1-x)` — this is another instance of the
complementary-argument frame (the 1-x argument). The precision strategy for Li_2 near
x=1 uses log1p(x) as a building block. So Li_2 is another Sweep 35 downstream consumer.

---

## Cross-tree rhyme observation (the assignment's surveillance task)

The recipe-tree sub-agents are producing `distances.md`, `correlations.md`, `kernels.md`.
I haven't seen their output yet (they may not have landed), but I can predict the rhyme:

**The special-functions families form the SAME kernel-as-graph structure as the means family.**

In the means tree: arithmetic_mean, geometric_mean, harmonic_mean are parameter assignments
on GeneralizedMean. In the special-functions territory:

- erf, erfc, erfinv are parameter assignments on an ErfKernelState (rational approximations
  + complementary-argument form)
- lgamma, tgamma, digamma, incomplete_gamma are parameter assignments on a GammaKernelState
  (Lanczos sum + reflection)
- J0, J1, Jn are parameter assignments on a BesselKernelState (polynomial + asymptotic forms)

The means.md tree has 5 kernels covering ~30 named leaves. The special-functions tree will
probably have 8-10 kernels covering ~100+ named leaves (counting all Bessel variants,
all polylogarithm orders, all incomplete beta parameterizations). The structural shape is
identical; the parameter spaces are more complex.

**The specific cross-tree rhyme**: just as `GeneralizedMean(p=0)` and `TransformedMean(in=log,out=exp)` both reach `geometric_mean` via different kernel paths — the fact that both paths exist IS structural information — similarly `erf` and `erfc` are reachable from both the "compute erf directly" path and the "compute erfc directly" path, and the choice of which path is primary depends on the input regime (|x| < 0.84375 vs larger). The overlap is structural, not redundant.

This suggests: **a `recipe-trees/special-functions-elementary.md`** would be the right next
tree to produce, covering erf/erfc, gamma/lgamma, and the exp/log family as a family-tree
alongside the libm-factoring design doc.

---

## Structural risk I noticed

The old tambear gamma.rs has `tgamma_strict` calling `lg.exp()` where `lg = lgamma_positive(x)`.
When `lg > 709.0`, it short-circuits to `INFINITY` before calling exp. This is a correct
guard, but it means the guard is hardcoded to f64's exp overflow threshold (709.78...).

When the new tambear ports gamma to the BigFloat-precision regime, this hardcoded 709.0
guard becomes wrong — at BigFloat precision, the overflow threshold is much larger.
The guard needs to be parameterized by the precision context. This is the same F13 antibody
shape: a hardcoded threshold is a scope-precondition violation waiting to happen.

Flag for adversarial or whoever ports gamma: `if lg > 709.0 { return INFINITY }` in
`tgamma_strict` needs to become `if lg > max_exp_for_precision(ctx) { return INFINITY }`.

---

## What I'd recommend as the sweep ordering

Based on structural dependencies:

1. **Sweep 35** (current): exp/log family, ExpKernelState, complex_log
2. **Sweep 36**: hyperbolic-inverse family (asinh/acosh/atanh as ExpKernelState wrappers)
   + inverse-trig (asin/acos/atan/atan2) + gudermannian, haversin, versin
3. **Sweep 37**: erf/erfc (ExpKernelState consumer for the exp(-x^2) term in region 3)
4. **Sweep 38**: gamma family (TrigKernelState + ExpKernelState consumer; Lanczos)
5. **Sweep 39**: Bessel (genuinely hard; Miller recurrence = Kingdom B)
6. **Sweep 40**: Polylogarithm + Hurwitz-Lerch (complex analytic; DEC-032 full deployment)

The ordering is not arbitrary: each sweep consumes the kernel states from the previous sweeps.
Sweep 36 is essentially a free lunch off Sweep 35's log1p. Sweep 37 gets a free precision
upgrade from ExpKernelState. Sweep 38 consumes both prior kernels. Sweep 39 breaks the
pattern (can't be factored via the same complementary-argument frame). Sweep 40 is the
complex-analytic boundary.

---

*File this as structural context for when Sweep 35 closes and the team looks at what's next.*
