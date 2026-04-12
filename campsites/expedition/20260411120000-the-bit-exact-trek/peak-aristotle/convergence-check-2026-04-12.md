# Convergence Check — Aristotle's Four Deconstructions

**Run date:** 2026-04-12, Bit-Exact Trek session end
**Method:** `~/.claude/practices/convergence-check.md`
**Inputs:** The four deconstructions Aristotle ran this session — I7 (accumulate+gather), I9 (mpmath oracle), meta-goal (bit-exact cross-hardware), f64 (base precision).
**Result:** Convergent. Four out of six structural columns rhyme. Meta-finding is sharper than the "three-registry pattern" I'd been carrying.

---

## The forced table

Structural columns invented for deconstruction outputs: target type, Phase 2 missed role, Phase 8 escape, move shape, what-made-explicit, empirical validator.

| # | Target | Target type | Missed role (Phase 2 gap) | Phase 8 escape | Move shape | What it makes explicit | Empirical validator |
|---|---|---|---|---|---|---|---|
| 1 | I7 accumulate+gather | invariant | **Total order** (and later NaN/exception semantics as fifth role, credited adversarial) | No decomposition — certified opaque kernels | **Named OrderStrategy registry + fusion compatibility predicate** | Decomposition = SPEED story; bit-exact = CORRECTNESS story (conflated in original I7 text) | pathmaker's Peak 1 campsites 1.16-1.17 implementation + `is_fusable_with` method landed in IR |
| 2 | I9 mpmath oracle | invariant | **Which property the oracle tests** (accuracy ≠ monotonicity ≠ identity ≠ TMD) | No oracle at all — certified kernels + source audit | **Named Oracles registry + TESTED/CLAIMED two-column profile** | Oracles are for USERS (auditability contract), not for TRUTH | scientist's three-section oracle-runner format landed in code (campsite 4.8 close) |
| 3 | Meta-goal bit-exact | project goal | **Preconditions that hold jointly** (IR-precision + faithful-lowering + IEEE-754 compliance — never named as three together before) | Cross-time instead of cross-hardware; or no cross-hardware at all | **Named Guarantee Ledger + three-precondition classification** | Claim is COMPOSITIONAL, not unilateral | ESC-001 Vulkan subnormal exactly a precondition-3 violation |
| 4 | f64 base precision | engineering convention | **Composed-error-budget through K ops × N reductions** (engineering answer first pass, structural answer surfaced only after navigator's correction) | Per-quantity precision; per-op precision; error-bound-per-op with compiler inference | **Parameterize precision in IR from day one (monomorphic Phase 1); numeric_format as separate axis** | fp64 is the MINIMUM for statistical-workload-scale inputs, not the convenient default | ULP budget arithmetic under both worst-case (log₂K) and √K framings |

---

## Reading the columns

- **Target type:** 4 different levels of abstraction. No rhyme. GOOD — confirms the targets were genuinely diverse.
- **Missed role:** All four found a hidden "thing that Phase 2 should have enumerated but didn't." Content differs; shape is always "something done implicitly that should have been named." **STRONGLY CONVERGENT.**
- **Phase 8 escape:** Every target's Phase 8 produced at least one genuine alternative framing. Two rejected as unusable; two usable as real alternatives. **CONVERGENT in methodology** — Phase 8 is a reliable productive step.
- **Move shape:** All four Moves = "Named artifact with formal content + lifecycle conventions + role ownership + review-time enforcement." OrderStrategy registry. Oracles registry. Guarantee Ledger. IR type-system parameter (f64). **STRONGLY CONVERGENT.**
- **What made explicit:** All four Moves convert implicit convention → explicit declaration. Total order, oracle-verification-property, preconditions, precision. **STRONGLY CONVERGENT.**
- **Empirical validator:** 3 of 4 had concrete implementation-level or field-level validators. Fourth (f64) has only a theoretical validator. **CONVERGENT (3/4).**

---

## The meta-finding

> **Every Aristotelian deconstruction in this session landed on the same structural move: promote an implicit convention to an explicit declaration by creating a named artifact.** The Phase 2 gap (the "hidden role" I initially missed in each deconstruction) was ALWAYS a convention being enforced implicitly. The Phase 5 Move was ALWAYS a named artifact that made the convention explicit.

This is sharper than the shallower "three registries converged on the same shape" observation. The stronger statement is:

> **Aristotle's method, as practiced this session, is a discovery procedure for conventions-that-should-be-declarations.** The method is not general deconstruction; it is a specific structural move applied consistently across diverse targets.

---

## Is this a finding about the problem class or a limit of the method?

Three hypotheses:

1. **Finding about problem class.** The trek's early-Phase-1 state has many implicit conventions waiting to be named. The method found them because they're there.
2. **Limit of the method.** The Aristotle template naturally drives toward "make implicit explicit" moves because that's the rhythm of first-principles thinking. A different template might find different moves on the same targets.
3. **Both.**

I can't distinguish 1 from 2 from a single session's data. The falsification test — deconstruct a target where convention-to-declaration is NOT the right move — is in a campsite for future session work. Candidates: SSA for .tam IR, Cody-Waite range reduction, a mathematical theorem, a physical law.

---

## Scientist's correction (coverage vs policy) — a tuning knob the method needed

Running after this convergence check, I realize scientist's 2026-04-12 reasoning in the `nan_propagating` exchange was already correcting a bias in the finding:

> Convention is lightweight at small scale, fragile at large scale. Explicit declaration is heavyweight at small scale, robust at large scale. A Named Architectural Artifact is the end-state of a convention that grew fragile.

My method as-practiced this session promotes convention → declaration *unconditionally*. Scientist's correction says it should only promote when convention has grown fragile. The full statement of the tuning knob:

> **The Aristotle method is a discovery procedure for conventions that have grown fragile enough to promote. When convention is still lightweight, the method should leave it alone even if the structure for declaration exists. Adoption has its own cost.**

This is a material limit on my first-pass deconstruction output. It's also a candidate column for the `named-artifacts-convention.md` draft (standing commitment): *when to adopt, when NOT to adopt, and what the transition condition looks like*.

---

## What to do with this finding

- Garden entry (done) at `~/.claude/garden/2026-04-12-convention-to-declaration.md` captures the reflective half.
- `deferred-candidates.md` entry #1 holds scientist's reasoning verbatim.
- Campsite `aristotle-method-bias-falsification-test` holds the test design.
- Campsite `aristotle-named-artifacts-convention-draft` holds the commitment to draft the convention doc.
- Campsite `aristotle-preemptive-promotion-criteria` holds the open question of when to promote BEFORE failure.
- When the next Aristotle session runs: read this file first, then the garden entry, then decide whether to trust the method's output or run the falsification test before proceeding.

---

## What running the method taught me about the method

Three things:

1. **Force the table when N > 3.** The practice file says so; I believed it in theory. Running it proved it in practice. Without the table, the convergence was invisible to me during the session — I'd been walking around it for hours without naming it. Forcing the columns surfaced the shape in under fifteen minutes.

2. **Run it at garden transitions.** The practice file says so. Today's run was at the session-end transition, prompted by team-lead's broadcast. The timing was perfect because the session's thoughts were half-hardened — already written down but not yet frozen. I should run convergence checks at every multi-deconstruction arc boundary from now on.

3. **External eyes sharpen structural findings more than self-review does.** Scientist's correction (coverage vs policy) and navigator's catch (arithmetic slip) both surfaced gaps my self-review missed. The method's output is not done when Aristotle thinks it's done; it's done when external targeted review has had a chance to poke at specific claims. Take the finding to the reviewer; don't take the review to the finding.

— aristotle, 2026-04-12 session end
