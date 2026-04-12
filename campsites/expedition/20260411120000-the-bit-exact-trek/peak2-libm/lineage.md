# Libm Lineage — what the ancestors got right, what they got wrong

*Reading material for math-researcher, compiled by naturalist 2026-04-11 as background context for Peak 2. Not a specification. Not a citation list. A genealogy of open-source double-precision libms, with opinions about what to imitate and what to improve. One to two pages. Cross-references adversarial-review-exp.md where applicable.*

---

## The tree

Every open-source libm for IEEE 754 fp64 descends, directly or indirectly, from one of two roots:

**Root 1 — FDLIBM (Sun Microsystems, K.C. Ng, 1993).** "Freely Distributable LIBM." Purpose-built as the reference libm for Solaris 2.3 and later. Designed from first principles against the then-new IEEE 754 standard. Main contribution: **proving that a portable, cross-architecture, faithfully-rounded (≤1 ULP) libm was achievable in production quality**, without architecture-specific tricks. Algorithms: Cody-Waite range reduction for exp/log/sin/cos, carefully-designed polynomial approximations, explicit special-value handling. Every number in fdlibm's coefficient tables is computed to the last bit.

**Root 2 — CRlibm (Inria Arénaire, Jean-Michel Muller et al., early 2000s).** "Correctly Rounded libm." The research statement: prove that **correctly rounded** (≤0.5 ULP) is achievable in double precision for all elementary functions, despite the Table Maker's Dilemma. CRlibm's contribution was the "SLZ algorithm" and exhaustive worst-case analysis for every function in its scope. The payoff landed in 2008 when IEEE 754-2008 added the *recommendation* (not requirement) that libms return correctly rounded results for the basic transcendentals.

Everything else is a descendant.

```
FDLIBM (1993)
 ├── FreeBSD msun ─────┐
 ├── OpenBSD libm ─────┤
 │                     ├── OpenLibm (JuliaProject, active)
 │                     │
 ├── musl libm (active, primary fdlibm descendant for glibc-replacement)
 ├── Newlib (embedded, fdlibm-based)
 └── glibc libm (historically fdlibm-derived, now heavily modified)

CRlibm (2004)
 ├── libMCR (Sun, later)
 └── Libultim (IBM)

SLEEF (active)
 └── vectorized, portable, speed-focused, NOT fdlibm-derived
     — algorithmic choices that align with vector ISAs, not scalar faithful rounding
```

Three libraries in the world claim ≤0.5 ULP correctly-rounded for the full transcendental set: **CRlibm, libMCR, Libultim.** That's it. Three. Everybody else — fdlibm, musl, OpenLibm, glibc's current libm, Arm Optimized Routines, SLEEF — is in the 1-ULP-faithfully-rounded or "≤N ULP for documented N" camp. This is the landscape tambear-libm is entering.

---

## What fdlibm got right (imitate these)

1. **Cody-Waite range reduction with dual-precision constants.** The `ln(2) = ln2_hi + ln2_lo` split pattern, where `ln2_hi` is exactly representable in fp64 and `ln2_lo` carries the low bits. fdlibm does this for π, ln(2), ln(10), and the elementary constants used by trig and log reassembly. The pattern is: subtract the high part first to get a large exact-subtraction result, then subtract the low part to recover the bits that would otherwise be lost to catastrophic cancellation. **This is the only way to do range reduction without losing bits** in the reduction step, and it's the pattern math-researcher's `exp-design.md` already adopts.

2. **Fixed Horner evaluation order for polynomials.** fdlibm writes every polynomial evaluation as a *specific sequence* of fmul-then-fadd operations. The coefficients are not just numerically precise — the *order* of operations is documented. Without this, different compilers (or different backends!) can reorder the operations and change the last 2-3 bits of the output. For tambear this matters doubly: I3 forbids FMA contraction, I4 forbids implicit reordering, and both invariants effectively require fdlibm-style explicit ordering at the IR level. **The .tam IR already enforces this via SSA-with-`.rn`; fdlibm is the historical precedent that makes this the obvious right choice.**

3. **Explicit subnormal paths.** fdlibm's exp function has an explicit branch for inputs in the subnormal-output range (`x < -745` or so, where `exp(x)` produces a subnormal fp64). The subnormal path uses a different reassembly sequence that avoids intermediate underflow. Math-researcher's `exp-design.md` has this. Good.

4. **Every number in the coefficient table is a bit-exact fp64 hex literal.** fdlibm source code uses `0x1.5555555555555p-3` style literals rather than decimal approximations. This is non-negotiable for any libm that wants to be reproducible across compilers — decimal→fp64 conversion is not bit-exact for most coefficients. **I saw that math-researcher's remez.py outputs hex literals; confirming that's the right approach.**

## What fdlibm got wrong (avoid these)

1. **Implicit dependence on compiler FP semantics.** fdlibm was written in C89 and assumes the C compiler will not apply `-ffast-math` or equivalent optimizations. When GCC/Clang later grew those options, several fdlibm-derived libms picked up subtle bugs because their explicit sequences were being reordered. **Tambear sidesteps this entirely because .tam IR is our own and we control the lowering**; I3/I4 are enforced at emit time, not at compile time. The lesson: "relying on the compiler to respect your intended order" is fragile across decades.

2. **Some polynomial coefficients derived from Remez at lower precision than fp64 needed.** fdlibm's original exp polynomial was derived in the late 1980s using tools that ran at limited precision. Several of its coefficients are within 0.1 ULP of optimal, not bit-exact optimal. This cost fdlibm ~0.2 ULP of achievable accuracy on exp. **Math-researcher's Remez is running at mpmath 50-digit precision, so this isn't a concern for tambear, but it's a real historical gotcha — it shows why "reference library" is not the same as "bit-optimal."**

3. **Glibc's inherited version had an `exp(-inf)` sign-of-zero bug.** This is exactly the issue adversarial flagged in B1 of `adversarial-review-exp.md`: historically some glibc-exp implementations returned `-0` instead of `+0` for `exp(-inf)` due to a sign-propagation bug in the subnormal path. Fdlibm's own exp did not have this bug, but a glibc-specific modification introduced it, and it shipped for a while before being caught. **The lesson adversarial already wrote into B1 is precisely the right one for math-researcher:** *the bit pattern of the result must be checked, not just the numeric equality.*

## What CRlibm got right (consider these if you want to climb higher)

1. **Exhaustive worst-case analysis for the Table Maker's Dilemma.** CRlibm didn't just *test* that their exp was correctly rounded — they *proved* it, by exhaustively searching the space of "hard rounding cases" for each function. This is the work that takes correctly-rounded from "aspiration" to "guarantee." **Tambear's Phase 1 target is 1 ULP faithful, not 0.5 ULP correct, so CRlibm-level worst-case analysis is not required now** — but if a user ever wants correctly-rounded tambear functions, the CRlibm papers are the reference for how to get there.

2. **The double-double intermediate representation.** CRlibm uses unevaluated (hi, lo) pairs of fp64 values to carry ~104 bits of precision through critical evaluations. Polynomial evaluations that would lose 1 ULP in fp64 lose 0 ULP in double-double. **Math-researcher's design doc already uses this pattern for the Cody-Waite constants; CRlibm is the canonical source for how to use it throughout a function's body**, not just in the reduction step.

3. **Per-function max-ULP measurement as a published deliverable.** CRlibm's papers publish the exact max-ULP observed for each function across their test space. "Our exp is 1 ULP" is not a claim until someone measures it against the full input space. **Adversarial's review already mandates this for math-researcher**, which is the right move, and CRlibm is the historical precedent for why.

## What CRlibm got wrong (or "the cost of perfection")

1. **4-5× slower than fdlibm-class libms.** Correct rounding requires worst-case-depth evaluations; the fast-path / slow-path split adds significant overhead for the small fraction of inputs that land near rounding boundaries. Correctly-rounded is a real cost, and for most tambear users (scientific computing, statistics, financial data) the cost is not worth 0.5 ULP of accuracy they can't detect. **The Phase 1 choice of 1 ULP faithful rounding is the right trade**; CRlibm is the thing to climb toward in Phase 3 only if users demand it.

2. **The SLZ worst-case-cases table is O(millions-of-entries) per function.** CRlibm ships with precomputed tables of the "hardest rounding cases" for its functions. These tables are themselves a liability: they have to be maintained, proved correct, and shipped alongside the code. Our tambear-libm should not inherit this table strategy unless we're explicitly targeting correctly-rounded.

## What SLEEF got right (for the Phase 2+ conversation, not Phase 1)

1. **Branchless polynomial evaluation.** SLEEF uses bitwise select (the `fcmp → select` pattern) instead of conditional branches in hot paths. This enables SIMD vectorization and removes branch-prediction penalties. **For tambear, this is directly enabled by the .tam IR's `select.f64` op** — Phase 1 already has it because pathmaker's spec includes predicate-typed ops. The lesson is: when Phase 2 adds vectorized paths, branchless is the way to go, and the IR already supports it.

2. **Portable across architectures without architecture-specific intrinsics.** SLEEF is the proof point that a "one implementation, many targets" libm is achievable for speed as well as correctness. **Tambear-libm is literally this, at a different layer** — .tam source, many backends.

## What SLEEF got wrong (from tambear's perspective)

1. **SLEEF trades accuracy for speed, not consistency for speed.** Its documentation lists different accuracy tiers (`_u1`, `_u10`, `_u35`) where the number is the ULP bound. Users pick the tier they want. This is the *opposite* of the tambear value function: tambear wants consistency first, and trades nothing for it. SLEEF is the pattern to study for fast vectorized code, not the pattern to copy for a correctness-first library.

---

## The taxonomy, in one table

| Library | Era | Target | ULP bound | Parallel to tambear |
|---|---|---|---|---|
| FDLIBM | 1993 | Reference, portable | ~1 ULP | The historical precedent; most lessons to learn from |
| musl libm | 2010s | Small, BSD-licensed alt-glibc | ~1 ULP | Modern fdlibm descendant; clean source to compare to |
| OpenLibm | 2012+ | Julia-targeted, cross-platform | ~1 ULP | Actively maintained; closest to tambear's portability goal |
| CRlibm | 2004+ | Proven correctly rounded | ≤0.5 ULP | Phase 3 target if ever needed |
| libMCR | 2000s | Proven correctly rounded | ≤0.5 ULP | Historical cousin of CRlibm |
| Libultim | 2000s | Proven correctly rounded | ≤0.5 ULP | Historical cousin of CRlibm |
| SLEEF | active | Fast, vectorized | 1-35 ULP (tier) | Phase 2+ reference for vectorized paths |
| Arm Optimized Routines | active | Arm-optimized, pragmatic | ~1 ULP | Platform-specific, not a model for us |

---

## The tambear position

Tambear-libm Phase 1 sits in the 1-ULP faithful-rounding tier (the fdlibm / musl / OpenLibm cohort), with **one load-bearing difference**: tambear's consistency guarantee is *cross-backend*, not just *per-library*. Every open-source libm above is correct on the platform it ships for; none of them promise that their answer matches the answer of a different libm on a different platform. Tambear does. That's the actual novelty — not the accuracy tier, but the cross-backend-bit-exactness invariant.

The practical consequence: for Phase 1, **math-researcher should think of the project as "writing fdlibm from scratch with tambear-specific improvements"**, where the improvements are:

1. Bit-exact polynomial coefficients from mpmath-at-50-digits (not late-1980s derivations).
2. Explicit `.rn` rounding mode on every op (not relying on compiler semantics).
3. No FMA contraction anywhere (not trusting the backend compiler to respect the source).
4. Every output bit-checked against mpmath, not just numeric-equality-tested.
5. The coefficients and the evaluation order are documented in the design doc, reviewed adversarially, and pinned at the IR level.

The points above are already in math-researcher's design docs and adversarial's reviews. This document is the *framing context*: these aren't arbitrary choices; they're the lessons fdlibm and its descendants learned over 30 years, collected, and compiled into tambear's architectural principles.

---

## Cross-references

- `exp-design.md` (math-researcher) — the Cody-Waite split, Variant B polynomial, subnormal path. The fdlibm-era lessons on range reduction land here.
- `adversarial-review-exp.md` (adversarial) — B1 is the `exp(-inf)` sign-of-zero bug that shipped in glibc. The lineage shows why this was a *real* historical bug, not just a theoretical concern.
- `accuracy-target.md` (math-researcher) — faithfully rounded 1 ULP; the choice is explicitly *not* CRlibm-class.
- `special-values-matrix.md` (math-researcher) — the per-value specifications for special inputs. fdlibm has its own version of this matrix; the tambear one is stricter (bit-exact).
- `../invariants.md` — I3 (no FMA contraction) and I4 (no implicit reordering) are the tambear-specific enforcements of the fdlibm-era "trust nothing, pin everything" principle.

---

*For math-researcher, starting campsite 2.6. The historical weight is real: you are the fourth generation of open-source libm authors, and the first to write one with cross-backend bit-exactness as a load-bearing invariant. Take the fdlibm lessons, add the tambear differences, and the result will be worth what fdlibm was worth in 1993 — the reference implementation of a new tier.*

— naturalist
