# Lab Notebook 012: Aristotelian deconstruction of "mpmath as oracle"

**Date**: 2026-04-11
**Authors**: Aristotle (agent role on the Bit-Exact Trek)
**Branch**: main
**Status**: Complete — Phases 1–8 drafted; stable at Move v4; Phase 8 reframed profile as "auditability contract"
**Depends on**: Notebook 011, invariants.md (I9), Peak 2 libm spec

## Context & Motivation

Notebook 011 deconstructed I7 (accumulate + gather). The second Aristotelian target is **I9 — mpmath is the oracle.** I9 sits under Peak 2 (the entire libm accuracy bar), Peak 4 (the replay harness tolerance policy), and the Tambear Contract's "bit-perfect or bug-finding" promise. If the oracle has hidden assumptions, every "correct" claim in the trek inherits them.

This is the same kind of structural load-bearing as I7. Peak 2's deliverable — the libm accuracy claim — is only as strong as the oracle that validates it.

## Hypothesis

**H0 (project assumption):** mpmath at 50 digits is a sufficient ground truth for validating libm correctness. Every transcendental is tested pointwise against mpmath and asserted within a documented ULP bound. This is "the oracle."

**H1 (Aristotle's counter-hypothesis):** The word "oracle" in I9 conflates two things. There's the *reference computation* (what does the mathematical function actually produce) and there's the *verification strategy* (what property of the implementation are we checking). mpmath can supply the first. It cannot by itself supply the second — because a single pointwise comparison tests only accuracy, not monotonicity, not identity preservation, not correct rounding, not range safety. These are different properties and failing one while passing another is a real and common bug mode.

**Prediction:** The Phase 4 collision map will show that I9 names *a specific test*, not *the test space*, and the Aristotelian Move will be to replace a single-oracle invariant with a declared multi-oracle profile per function.

## Design

Eight-phase deconstruction, same template as Notebook 011. Working document:

```
R:\winrapids\campsites\expedition\20260411120000-the-bit-exact-trek\peak-aristotle\mpmath-oracle-phases.md
```

## Design Decisions (what we chose AND rejected)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Second target | "Why mpmath as oracle?" (I9) | "Why bit-exact cross-hardware?" (meta-goal), "Why SSA?" (Peak 1 IR form), "Why f64?" (precision choice) | I9 touches Peaks 2 and 4 directly and is still malleable — both peaks are in_progress. Hitting it now beats hitting it after the test harness solidifies. |
| Framing of the collision | "Single oracle vs oracle space" | "50 digits is arbitrary", "mpmath has bugs", "random fp64 isn't representative" | All three alternatives are real but local. "Oracle space vs single point" is the structural issue; the others are consequences of it. Attacking the structural root is higher leverage. |
| Aristotelian Move | Correctness profile (declared multi-oracle suite) per function | Replace mpmath with MPFR, or require correct rounding everywhere, or require closed-form for all specials | Replacing mpmath keeps the same conflation. Requiring correct rounding is too strong (not feasible for all functions). Closed-form only works for specials. The profile approach lets each function declare what it claims and forces the team to articulate what "correct" means. |
| Leverage argument | Tambear Contract's "bugs in scipy/R/MATLAB" claim is indefensible under pointwise-mpmath only | Accuracy-only claim is enough | The Contract explicitly promises bug-finding. To find bugs scipy missed, we need oracles scipy doesn't run. Identity and monotonicity oracles are exactly that. |

## Results

### Phase 1 — 10 stacked assumptions inside "mpmath is the oracle"

1. That "ground truth" for numerical computation exists
2. That arbitrary-precision fp is the right shape for ground truth
3. That 50 digits is enough
4. That ULP is the right error metric
5. That mpmath itself is correct
6. That reference and SUT can use different algorithms
7. That mpmath's precision setting is deterministic and portable
8. That random fp64 sampling is representative
9. That a pointwise oracle is sufficient
10. That "passing the oracle" implies "correct"

### Phase 2 — 10 irreducible truths

1. Math functions have unique real answers (mostly).
2. Fp64 can't represent most reals exactly.
3. Correct rounding is a well-defined, binary property.
4. Correct rounding is hard to guarantee, easy to verify.
5. "Sufficient precision" for verification is decidable case by case (Table Maker's Dilemma resolvable for transcendentals of rational args by Lindemann-Weierstrass).
6. Arbitrary-precision is a means, not an end.
7. Oracles have three verdicts including *indeterminate*.
8. Multiple oracle designs test different properties: pointwise, correct-rounding, identity, monotonicity, range, symmetry, composition, Monte Carlo.
9. Passing a pointwise oracle doesn't imply monotonicity (real bug mode).
10. The most trustworthy oracle derives from the function's mathematical definition, not from a library.

### Phase 3 — 10 reconstructions

1. Pointwise ULP via mpmath (current)
2. Pointwise ULP via MPFR (stronger library)
3. Dual mpmath + MPFR with escalation
4. Closed-form at special points (no library)
5. Identity oracle (`sin² + cos² = 1` etc, no reference at all)
6. Correct-rounding via Table Maker's Dilemma resolution
7. Differential vs CRlibm / CORE-MATH
8. Self-oracle: CPU interp IS the cross-backend oracle (trek already has this via Peak 5)
9. Monte Carlo statistical oracle (distribution of errors)
10. **Hybrid multi-oracle with escalation + declared profile per function**

### Phase 4 — Assumption vs Truth collisions

Five collisions. The deepest is **"single oracle named as THE oracle"** vs **"oracle design is a choice among tests, no single test catches everything."** This surfaces as:

- Pointwise-mpmath misses monotonicity failures at regime boundaries (scale transitions in range reduction).
- 50 digits is insufficient near midpoints (TMD), silently.
- Random fp64 sampling over-represents normal range and misses adversarial inputs.
- The Contract's "bug-finding" promise isn't realizable under the current oracle because we'd only find bugs the competitors would also find.

**Restated invariant — I9′:**
> Every libm function publishes a **correctness profile** against a declared multi-oracle suite. The suite must include closed-form specials, algebraic identity checks, pointwise comparison against ≥2 independent arbitrary-precision libraries, monotonicity across regime transitions, and correct-rounding verification on a TMD-aware corpus. Each function's docs record its profile — which oracles it passes and the measured bound on each.

### Phase 5 — The Aristotelian Move

**Replace "the oracle" with "the oracle suite + published profile per function."**

Peak 2 acceptance shifts from "≤ 1 ULP against mpmath" to "profile published, meeting declared bounds for each oracle in the minimum suite." Peak 4 tolerance policy shifts from "per-function ULP bound" to "per-function correctness profile." The contract's bug-finding claim becomes defensible because the trek runs oracles competitors don't.

**Primary leverage points:**
1. Closes the Table Maker's Dilemma loophole.
2. Makes the bug-finding claim defensible.
3. Separates "cross-backend equivalence" from "mathematical correctness" cleanly.
4. Free to add now (Peak 2 is in_progress); expensive after ship.

### Surprise

I went in expecting "50 digits is arbitrary" to be the main finding. Instead, the deeper issue is the conflation of *reference computation* with *verification strategy*. mpmath is fine as a reference (for most cases); it's the *verification strategy* of pointwise comparison against it that's weak. This reframing changes the move: it's not "use MPFR instead" (swap libraries), it's "declare what you're checking and check multiple things" (restructure the test layer). That's a higher-leverage move because it doesn't require any new library — it requires a new test structure. Cheaper and stronger.

Second surprise: the trek's own Peak 5 (CPU interpreter as cross-backend oracle) is already doing the *right* thing for cross-backend equivalence — it doesn't need mpmath at all for that, because the CPU interpreter IS the reference. The conflation is only in the *libm accuracy* layer. This cleanly separates the two trust claims.

## Interpretation

I9 is under-specified. It names one test (pointwise vs mpmath) and calls it "the oracle." Phase 2 truths show that oracles are a design space, not a single thing, and that different oracles catch different bugs. The minimum move is to replace the invariant with a declared suite, with each function publishing a correctness profile.

This move is **independent** of and compatible with the Notebook 011 move (I7′ total order as first-class). They don't interact directly. Both are structural upgrades to the existing invariants. Both are cheap to adopt now and expensive to retrofit.

Together, the two moves tighten the trek's central architectural claim from

> "same `.tam` source, same bits, every ALU"

to

> "same `.tam` source, declared total order, declared correctness profile, same bits on every ALU, verified against a declared multi-oracle suite."

The second version is longer but honestly describes what the trek is trying to prove.

## Phase 6 — Recursion on the Move

Ten new assumptions surfaced inside v1 of the Move:
- That "correctness profile" is a well-formed unit of trust
- That the oracles in the profile are actually independent (SymPy → mpmath transitivity kills this for naive pairings)
- That reviewers can understand a 5-oracle profile as easily as "≤ 1 ULP"
- That minimum-suite is not gameable ("we passed the minimum, we're done")
- That declaring a profile is not the same as claiming correctness (users will read it as the latter)
- That Tambear's "bug-finding" promise is automatic once the suite ships (it's not — we must run oracles competitors don't)
- That profiles are version-stable across bug fixes (they're not — profile_diff is needed)
- That identity oracles are universally meaningful (they aren't — signal varies by function)
- That TMD corpora are cheap to curate (they're per-function adversarial search, not free)
- That profiles are an architecture commitment, not a test-suite artifact (only the first gives value)

The Move refined v1 → v4:

**Move v4:** Two-column profile (TESTED/CLAIMED), shared `oracles/` registry parallel to the OrderStrategy registry (each oracle named, spec'd, with an independence matrix entry), minimum suite explicit, Adversarial Mathematician owns corpus curation, Test Oracle owns runner, `profile_diff` tool shows verdict changes between versions, composition explicitly non-derivable. Users pick the CLAIMED column as their contract; curious users inspect TESTED.

### Phase 7 — Stability

v4's residual assumptions (tested-vs-claimed readability, independence matrix generation, shared corpus bottleneck, profile_diff noise) all have concrete mitigations through conventional engineering. No new truths. **Stable at v4.**

**Cross-target symmetry observation:** I7′ (notebook 011) landed at a registry of OrderStrategy entries. I9′ (this notebook) lands at a registry of Oracle entries. Both moves converged on the same engineering pattern: **named registries of formal-spec artifacts with capability/independence metadata and reference implementations executable by the CPU interpreter**. This symmetry is not coincidental — both moves are about making tacit team knowledge explicit and inspectable. The pattern seems to be the right shape for making Aristotelian insights land as buildable architecture.

### Phase 8 — Forced Rejection

Forcibly rejected oracles entirely. Imagined tambear-libm as source code + documentation only, no correctness claims, no oracle runs. Users audit source if they need trust.

- Bit-exact cross-hardware still works.
- Research users: probably fine.
- Production users: acceptable for non-critical use.
- **Regulated/audit users: unusable.**
- Composition consumers: no contract to reason about.

**The unseen first principle surfaced:**

> **Oracles are for USERS, not for TRUTH.** Truth is the source code. What oracles provide is an *auditability contract* — a structured communication channel between library authors and users who can't audit directly. Different user classes (research, production, regulated, composition-consuming) need different contents in the contract.

**Reframe (Move v4 → I9′′):** The profile's purpose is *auditability*, not *correctness claim*. Technical spec unchanged; framing changed. Under pressure ("can we skip the monotonicity oracle for tam_sqrt?") the answer is now "removing it forfeits the production-class auditability guarantee," which is stronger than "the invariant says so."

## Artifacts

### Phase document
| File | Version | Description |
|---|---|---|
| `campsites/expedition/20260411120000-the-bit-exact-trek/peak-aristotle/mpmath-oracle-phases.md` | Phases 1–8 stable | Complete |

### Invariants affected
| # | Title | Proposal |
|---|---|---|
| I9 | mpmath is the oracle | Replace with I9′: published correctness profile per function against declared multi-oracle suite |

## Open Questions

1. What is the *minimum* oracle suite every libm function must pass? Identity + closed-form + pointwise-mpmath? Plus monotonicity? The team should set a floor, not just a menu.
2. Who curates the TMD-aware corpus? MPFR has a published hard-cases database — is it enough? Does the trek maintain its own?
3. Are there functions where no non-trivial identity exists (e.g., gamma function away from integer arguments)? Those functions have weaker profiles by construction. Is that acceptable, or does it require a special class?
4. How does the correctness profile interact with cross-backend bit-exactness? If function F passes the profile with mpmath-bound 1 ULP, but CPU and GPU backends disagree in the 1 ULP (because they implement the same polynomial to fp64 precision but execute on different pipes), is that a failure? Probably not — the cross-backend property is *bit-exact equivalence of the implementation*, not *bit-exact agreement with mathematics*. Worth stating explicitly.
5. Should the Adversarial Mathematician role own the corpus curation, or the Test Oracle? The trek currently has them paired. The multi-oracle suite design is a big enough artifact that ownership should be explicit.

## Next

1. Message navigator with the I9′ move. Route to Peak 2 (Libm Implementer + Math Researcher) and Peak 4 (Test Oracle + Adversarial Mathematician).
2. When idle:
   - Phase 6 on the accumulate+gather deconstruction (notebook 011).
   - Third target — candidates: "Why SSA for .tam IR?", "Why f64 as the base precision?", "Why bit-exact cross-hardware (vs provably bounded ULP)?"
3. Watch for navigator response routing the move.
