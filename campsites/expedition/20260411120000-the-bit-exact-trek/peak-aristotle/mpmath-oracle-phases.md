# Target: Why mpmath as oracle?

**Deconstructor:** Aristotle
**Date opened:** 2026-04-11
**Status:** Phases 1–8 drafted. Deconstruction stable at Move v4.

Invariant under deconstruction: **I9 — mpmath (or equivalent arbitrary-precision reference) is the oracle.**

Connected load-bearing claims:
- CLAUDE.md Tambear Contract clause 10: "Proven correct under the tests that matter… Gold-standard oracles for every published result where one exists… verified against mpmath/SymPy/closed-form analytical reference at 50-digit precision."
- Peak 2 (libm) accuracy bar: "Reference. mpmath at ≥ 50-digit precision… compute reference to 50 digits, compute ours, measure max and mean ULP error, assert max ≤ target."
- Peak 4 (replay harness) tolerance policy: transcendental ULP bounds are defined *relative to mpmath*.
- The entire "bit-perfect or bug-finding" claim: we compare against peers, but the ground truth — the one we *trust* — is mpmath.

---

## Phase 1 — Assumption Autopsy

"mpmath is the oracle" compresses ten assumptions into three words. Naming them:

1. **That there exists a "ground truth" for numerical computation.** This is the deepest assumption, and it's not obvious. For `sin(x)` on a real number, the mathematical truth is a real number — an infinite object — and any computer representation is an approximation. "Ground truth" in the computer sense means "the best approximation we can afford." Assuming such a thing is even well-defined is an assumption.

2. **That arbitrary-precision floating-point IS the best approximation we can afford.** Alternatives exist: symbolic computation (SymPy/Mathematica/Maple), interval arithmetic (MPFI), exact-real arithmetic (iRRAM, RealLib), computable reals, lazy infinite-precision libraries. Arbitrary-precision is one of several approaches to "make the approximation arbitrarily good."

3. **That 50 digits is enough.** Why 50? Not 34 (quad precision, the next IEEE level), not 100, not 1000. 50 is a cultural number — it feels round, it's well above fp64's 17, and mpmath lets you set it cheaply. But 50 digits in the reference means a ULP error of ~10⁻³⁴ in the reference, which is ~30 orders of magnitude below fp64's 2⁻⁵³ ≈ 10⁻¹⁶. That's "a lot of margin." Why this much? Why not less?

4. **That ULP-distance is the right error metric.** ULP assumes the comparison point is fp64 itself — "the nearest fp64 value to the true answer." Other metrics: relative error, absolute error, signed relative error (with direction), correctly-rounded indicator (binary: is our answer the single correct fp64 value or not?). ULP is the conventional choice, but it hides directional bias.

5. **That mpmath's implementations are themselves correct.** mpmath is software. It has bugs. It has algorithmic choices (e.g., for `sin(x)` on large arguments, what reduction does mpmath use? Does it agree with MPFR? With SymPy's `sin`?). We treat it as the ground truth, but it is in fact a particular *implementation* of a numerical library, and the trek explicitly forbids treating implementations as ground truth. The invariant I9 parenthetical "or equivalent arbitrary-precision reference" hints at this: mpmath is a specific stand-in.

6. **That the *reference* is allowed to use a different algorithm than the *system under test*.** mpmath's `exp` may use Taylor series on reduced range with high-precision constants; tambear-libm's `exp` will use a specific Remez polynomial at fp64 coefficients. We compare outputs, not algorithms. That's fine — as long as both algorithms converge to the same mathematical limit. But if we're testing whether our *specific* polynomial implementation is correct, mpmath (which uses a different polynomial) tests a different claim: "does our answer equal the mathematical truth?" rather than "does our polynomial implement the Remez coefficients we chose?"

7. **That running mpmath on the reference inputs is itself deterministic and reproducible.** mpmath's precision is configurable globally; different sessions might set it differently; some mpmath functions have internal heuristics that change at higher precision. "Run mpmath at 50 digits" is not a single well-defined operation — it's a call into a library whose behavior depends on version, precision setting, and function choice.

8. **That the test corpus is a representative sample.** "Generate 1M random fp64 inputs in each function's domain" is stated as the protocol. But random sampling of fp64 is not uniform on ℝ — 97% of fp64 values are in magnitudes ≥ 2⁻¹⁰²² — so random sampling massively over-represents normal and denormal regimes and under-represents small arguments. "Random fp64" is not "representative input."

9. **That the oracle is a single-point comparison.** We compute our answer, we compute mpmath's answer, we diff. We don't compare *distributions* or *error profiles* or *relative error vs argument magnitude*. A single-point comparison masks systematic bias.

10. **That the oracle catches bugs.** We compare outputs, so we catch cases where our output differs from mpmath's. We do NOT catch cases where our implementation has a semantic bug that produces the right answer on the test corpus but wrong answers elsewhere. "Passing the oracle" is not the same as "being correct."

Sub-assumptions I also flag:
- "Equivalent arbitrary-precision reference" as a category — MPFR, mpmath, Mathematica, Arb, Sage — are all *different* references. They disagree at the last digit for some inputs. The trek treats "any arbitrary-precision library" as interchangeable, which is empirically false.
- "The oracle is outside the system under test" — but mpmath is Python, tambear is Rust, and we compare Python floats to Rust floats. Both are IEEE fp64 at the reporting boundary, so that's fine, BUT: any tooling in between (JSON round-trip, string formatting, Python's `float()` constructor) must be bit-exact. That's a silent dependency.

---

## Phase 2 — Irreducible Truths

What survives if I strip everything back to what's undeniable?

1. **A mathematical function `f: ℝ → ℝ` has, for most inputs, a unique real-valued answer.** Exceptions: partial functions (log at zero), branch points (sqrt of negative), singularities. Otherwise, there is *a* truth.

2. **A fp64 computer cannot represent most real numbers exactly.** For any transcendental function at a generic input, neither the input nor the output is exactly representable. What's representable is the nearest fp64 value to the real answer.

3. **"Correct rounding" is a well-defined property.** Given a real answer `y`, the correctly-rounded fp64 value is the single fp64 value nearest to `y` (with round-to-nearest-even for ties). Either an implementation produces this value or it doesn't. It's a binary property.

4. **Correct rounding is hard to guarantee but easy to verify.** Producing correctly-rounded transcendentals is a research problem (CRlibm, CORE-MATH). Verifying whether a given `(input, output)` pair is correctly rounded is a straightforward computation: compute the real answer to sufficient precision, round to fp64, compare.

5. **"Sufficient precision" for verification is decidable case by case.** For any real number `y` that is not exactly half-way between two fp64 values, there is some precision `p` such that computing `y` to `p` bits unambiguously determines which fp64 value is nearest. The halfway case is the Table Maker's Dilemma, and it's always decidable for algebraic functions; for transcendentals, Lindemann-Weierstrass guarantees no transcendental value of a rational argument is exactly an fp64 binary fraction.

6. **Arbitrary-precision arithmetic is a means, not an end.** Its purpose is to compute `y` to enough bits to decide the rounding question. 50 digits is enough iff the true answer isn't within `10⁻³⁴` of an fp64 midpoint — which is almost always true, but not always. For sin(x) where x is near an integer multiple of π, the midpoint case can require thousands of digits.

7. **A test oracle is a function from (input, output) to verdict ∈ {correct, incorrect, indeterminate}.** "Indeterminate" is a real verdict — it means the oracle itself cannot decide at its current precision. Most "oracles" silently collapse indeterminate into correct or incorrect. That's a bug in the oracle design.

8. **Multiple oracle designs exist and they test different properties:**
   - **Pointwise ULP oracle** ("our answer is within N ULPs of truth"): tests accuracy.
   - **Correct-rounding oracle** ("our answer equals the correctly-rounded truth"): tests exactness.
   - **Monotonicity oracle** ("if x₁ < x₂ then f(x₁) ≤ f(x₂)"): tests sign-preservation and rank-preservation.
   - **Range oracle** ("our answer is in the declared range of the function"): tests domain safety.
   - **Symmetry oracle** ("f(-x) = -f(x) for odd f"): tests algebraic identities.
   - **Composition oracle** ("f(g(x)) = h(x) for known identity"): tests chain correctness.
   - **Monte Carlo oracle** ("the distribution of outputs on random inputs matches the theoretical distribution"): tests statistical correctness.

   These are *different tests*. "The oracle" in I9 currently names only the first one.

9. **An implementation can pass a pointwise oracle and fail a monotonicity oracle.** A polynomial approximation that's within 1 ULP but flips sign slightly in the wrong place violates monotonicity. The test corpus might not catch this because monotonicity failures often occur at scale-transition boundaries (between range-reduction regimes), and random sampling rarely hits them.

10. **The most trustworthy oracle is the one derived from the function's mathematical definition directly.** If `sin(x)` is defined by the Taylor series `Σ (-1)^k x^(2k+1) / (2k+1)!`, then the most trustworthy oracle computes that series to arbitrary precision on the *exact* input and rounds to fp64. That's what mpmath *does*, but the reason it's trustworthy is the series, not the library.

---

## Phase 3 — Reconstruction from Zero

Given only Phase 2's truths, what are the plausible oracle designs for tambear-libm?

### 1. Pointwise ULP oracle via mpmath (current choice).
Compute reference with mpmath at 50 digits, compare to our answer, measure ULP distance, assert ≤ bound.
- **Pro:** Easy to implement. Fast enough. Well-understood.
- **Con:** Trusts mpmath. Doesn't detect monotonicity failures. 50-digit constant is arbitrary. Single-point comparison.

### 2. Pointwise ULP oracle via MPFR.
Same shape, but use MPFR (the GNU library, wrapped via a Rust crate) instead of mpmath.
- **Pro:** MPFR is *itself* correctly rounded to within 0.5 ULP at its configured precision. It has stronger guarantees than mpmath.
- **Con:** MPFR is a C library with its own history and bugs. It's more trustworthy than mpmath statistically, but philosophically it's still just another implementation.

### 3. Dual oracle: mpmath AND MPFR must agree.
Compute the reference with both libraries. If they disagree at 50 digits, escalate to higher precision; if they still disagree, report the input as "oracle indeterminate" and exclude it from the test corpus, investigating separately.
- **Pro:** Catches mpmath bugs. The disagreement itself is evidence of something interesting.
- **Con:** Double the oracle cost. Still doesn't tell us the *true* answer in the indeterminate cases.

### 4. Closed-form symbolic oracle where one exists.
For identities like `sin(π/6) = 1/2`, `exp(0) = 1`, `log(1) = 0`, `tan(π/4) = 1`, compute the exact fp64 representation directly from the closed form — no approximation needed. These are checked bit-exactly, not within a tolerance.
- **Pro:** No dependency on any reference library. The closed form IS the truth.
- **Con:** Covers only special points. Doesn't test the polynomial at most inputs.

### 5. Identity oracle (no reference library at all).
Test algebraic relationships: `sin²(x) + cos²(x) = 1`, `exp(a+b) = exp(a) * exp(b)`, `log(exp(x)) = x`, `sin(-x) = -sin(x)`, `atan(tan(x)) = x` on the principal branch. No ground truth required — the identity IS the test.
- **Pro:** No reference library. Catches many subtle bugs (e.g., sign errors, phase errors, regime-switching errors) that a pointwise oracle misses. Exercises function composition.
- **Con:** Doesn't measure absolute accuracy. Identities can hold to 1 ULP while each side is 10 ULPs off the true value (if the errors cancel).

### 6. Correct-rounding oracle via Table Maker's Dilemma resolution.
For each test input, compute the real answer to as many bits as needed to resolve the rounding — potentially thousands of digits near midpoints. Assert our answer equals the correctly-rounded truth bit-exactly.
- **Pro:** The strongest possible oracle. If we pass this on a test corpus, we've shown correct rounding on that corpus.
- **Con:** Very expensive for hard cases. Requires a library that can detect midpoint cases (MPFR can; mpmath with 50 digits cannot).

### 7. Differential oracle: compare against CRlibm / CORE-MATH.
CRlibm is a research library that provides correctly-rounded implementations of many transcendentals. Assert our answer equals CRlibm's.
- **Pro:** CRlibm is designed for exactly this purpose. If we match CRlibm, we're correctly rounded.
- **Con:** CRlibm is incomplete (not every function, not every precision). It's also "borrowed" in the I8 sense — its implementation carries assumptions.

### 8. Self-oracle: our CPU interpreter IS the oracle for our GPU backends.
The trek already has this at Peak 5. The CPU interpreter runs our own libm on our own IR, and every GPU backend must match it bit-exactly. The oracle for libm *correctness* is separate from the oracle for cross-backend *equivalence*.
- **Pro:** Clean separation. Cross-backend equivalence is a structural property that doesn't need an external reference.
- **Con:** Doesn't validate the CPU interpreter's libm against anything. CPU interpreter agrees with itself by definition.

### 9. Monte Carlo statistical oracle.
Generate N inputs, run our function and the reference, compare the *distribution* of errors (mean ULP, max ULP, direction bias, histogram). Assert distribution properties, not per-point properties.
- **Pro:** Catches systematic bias that pointwise oracles miss.
- **Con:** Rare event bugs are invisible. Statistical properties are weaker than pointwise properties.

### 10. Hybrid multi-oracle with escalation.
Run ALL of: closed-form (where applicable), identity, pointwise-vs-mpmath, pointwise-vs-MPFR, correct-rounding TMD, and monotonicity. Each produces a verdict. The aggregate is a *profile* of the implementation. "Correct" means passing a declared subset of oracles, documented per function.
- **Pro:** Every oracle catches something the others miss. The profile is the most honest description of what's been verified.
- **Con:** Expensive. Requires defining what "correct" means per-function (which is actually a feature: we're forced to say what we're claiming).

---

## Phase 4 — Assumption vs Truth Map

| Assumption | Matching truth | Where they collide |
|---|---|---|
| "mpmath at 50 digits is ground truth" | T5, T6: arbitrary precision is a *means* of resolving rounding; 50 digits is sufficient iff the true value isn't near an fp64 midpoint | Near midpoints (Table Maker's Dilemma), 50 digits is NOT enough. The oracle silently gives wrong verdicts on those inputs. The trek doesn't know how many of those there are in its test corpus. |
| "Compare within N ULPs" | T8: multiple oracle designs test different properties | Pointwise ULP tests only accuracy, not monotonicity, not identities, not correct rounding. Our claim "bounded ULP error" is smaller than our intent "our math is correct." |
| "The reference library is the truth" | T10: the most trustworthy oracle comes from the function's mathematical definition | mpmath is a library, not a definition. The series that defines `exp` is the truth; mpmath is an implementation of the series that we happen to trust. |
| "1M random fp64 inputs" | T7: an oracle has a verdict space including "indeterminate" | Random sampling doesn't target the hard cases (midpoints, catastrophic-cancellation zones, argument-reduction boundaries). It over-samples where mpmath and tambear trivially agree and under-samples where they might disagree. "1M random" is a weak corpus. |
| "The oracle catches bugs" | T7, T9: passing a pointwise oracle doesn't imply monotonicity, range safety, or identity correctness | Bugs that are invisible to the pointwise oracle will ship. The trek's Tambear Contract claims "we find bugs in scipy/R/MATLAB" — but this is only findable if we run oracles they don't. With only a pointwise ULP oracle, we'd find only the bugs they'd also find. |

**The deepest collision:** I9 names a *single* oracle and treats it as *the* oracle. But the irreducible truths say that oracle design is a *choice among multiple tests*, and no single test catches everything. Our invariant is underspecified.

**Restated invariant (I9′):**
> **Every libm function is tested against a declared multi-oracle suite. The suite must include: (a) closed-form at special points, (b) algebraic identity checks, (c) pointwise reference comparison against ≥2 independent arbitrary-precision libraries, (d) monotonicity across regime transitions, (e) correct-rounding verification on a curated TMD-aware corpus. Each function's "correctness claim" names the subset it passes and records it in the tambear-libm docs.**

This replaces "mpmath is the oracle" with "the oracle is a suite, and we publish which parts of the suite each function passes."

---

## Phase 5 — The Aristotelian Move

The highest-leverage action:

**Define the oracle as a declared multi-test suite, and require every libm function to publish its *correctness profile* — not a single pass/fail, but a profile of which properties it's been verified against.**

Concretely:

- The Test Oracle role (currently paired with Adversarial Mathematician) builds a `libm_oracle_suite` with at minimum: `closed_form_specials`, `identity_checks`, `mpmath_pointwise`, `mpfr_pointwise`, `monotonicity_sweep`, `tmd_aware_corpus`. More oracles can be added; removing any is an escalation.
- Every libm function (starting with `tam_sqrt`, `tam_exp`, `tam_log`, `tam_sin`, `tam_cos`) is tested against the full suite.
- Each function's docs record a table: oracle × verdict × measured bound. A function that's within 1 ULP on mpmath but off-by-2-ULP on MPFR is HONESTLY reported as "mpmath: 1 ULP, MPFR: 2 ULP." Not masked.
- The trek's Peak 2 acceptance criterion shifts from "≤ 1 ULP against mpmath" to "correctness profile published and meeting a declared bound for each oracle."
- Peak 4's tolerance policy changes from "per-function ULP bound from tambear-libm's published accuracy" to "per-function correctness profile from tambear-libm's published profile."
- The "bit-perfect or bug-finding" claim from the Tambear Contract becomes STRONGER: we find bugs in competitors via the identity and monotonicity oracles that their pointwise-mpmath validation misses, which is the whole point.

**Why this is high leverage:**

1. It aligns the oracle with what we actually want to know. We don't want to know "are we within N ULPs of mpmath"; we want to know "is our math correct." Those are different questions.
2. It closes the Table Maker's Dilemma loophole. The TMD-aware corpus is built from MPFR's pre-computed hard cases (public, curated) plus adversarial search for new ones.
3. It makes the "bugs in scipy/R/MATLAB" claim defensible. The trek explicitly promises this — but pointwise mpmath comparison is weaker than what those libraries use, so we wouldn't find bugs they'd miss. Identity and monotonicity oracles find different bugs.
4. It makes the cross-backend equivalence claim cleaner. Cross-backend is "all backends match each other bit-exactly"; correctness is "each backend matches mathematics within declared oracles." Conflating them (as "the oracle is mpmath") hides a design choice.
5. It's free now and expensive later. Adding a monotonicity oracle in Peak 2 is one more test file. Adding it after we've shipped "1 ULP against mpmath" as a promise is a specification change.

**Why this is the first-principles move:**

Because Phase 2 truth 8 says "oracles are a design space, not a single thing," and Phase 2 truth 9 says "passing one oracle can fail another." I9 currently picks one point in that design space and names it "the oracle." First-principles thinking says: pick the whole space, declare the profile, test the profile.

**Recursion check — what does this move assume?**

1. That "declared profile" is a sensible unit of trust. It is, but it creates a new risk: the profile can be weak (few oracles, generous bounds) and still look credible. The trek needs a minimum profile — a set of oracles every libm function MUST pass, not just a list of possible ones.
2. That the identity oracle's identities are known. For sin/cos they're famous. For tanh/sinh they exist but are less-exercised. For special functions (erf, gamma) the identities are subtler. A libm function whose identities are unknown has a weaker profile.
3. That "independent arbitrary-precision libraries" exist for every operation. For exp, log, sin, cos — yes. For the full catalog (Bessel, elliptic, hypergeometric) — less so. SymPy can do symbolic evaluation but its numerical routines often call mpmath internally, so "independent" is a claim to verify.

---

## Phase 6 — Recursion: challenge the Phase-5 Move itself

The Phase-5 Move is:
> I9′ — Every libm function publishes a correctness profile against a declared multi-oracle suite. Minimum suite: closed-form at special points, algebraic identity checks, pointwise vs ≥2 arbitrary-precision libraries, monotonicity across regime transitions, correct-rounding on a TMD-aware corpus.

Adding this Move to the assumption list and running the Aristotle loop again.

### Phase 6.1 — Assumption autopsy on the Move

**M1.** That "correctness profile" is a well-formed unit of trust. "Profile" is a formal methods concept (software contracts, property-based testing suites). Importing it assumes profiles compose well — that a caller can inspect the profile of a function and reason about the profile of a composition.

**M2.** That the oracles in the profile are *independent*. If closed-form, identity, and pointwise-mpmath oracles all use mpmath under the hood (through transitivity — SymPy calls mpmath, some identity oracles use mpmath to evaluate the identity expression), the "independent" claim is violated and the profile is weaker than advertised.

**M3.** That profile reviewers (humans reading `tam_exp`'s docs) can *understand* the profile. Five oracle bounds is already cognitively heavier than "≤ 1 ULP." The profile risks being ignored by everyone except the few users who care deeply.

**M4.** That the minimum suite is REALLY minimum. A profile of "identity + monotonicity + pointwise-mpmath" might let a function ship without the TMD-aware corpus — and TMD is where the subtle bugs actually live. If "minimum" is gameable, the profile becomes a floor that protects against worst-case but not expected-case bugs.

**M5.** That declaring a profile is not itself a *correctness claim*. It's a test-passage claim: "this function passed these tests at these bounds." That's a weaker thing than "this function is mathematically correct." Users may read "profile published" as "certified correct" and be wrong.

**M6.** That the Tambear Contract's "bit-perfect or bug-finding" promise is satisfied by the refined oracle. It's not automatic. The promise requires running oracles competitors DON'T run. Which oracles are actually novel vs which are "we also run mpmath plus something" matters enormously for whether we'd find bugs they missed.

**M7.** That the profile is stable as the function evolves. A bug-fix that improves identity correctness by 0.3 ULP might worsen pointwise ULP by 0.1 ULP — the profile would need to be re-published and downstream consumers notified. Profiles are artifacts that need versioning, migration stories, diff tooling.

**M8.** That "identity oracle" covers the interesting space. Identities for sin/cos/exp are famous. For less-exercised functions (asinh, expm1, cbrt) the useful identities are fewer and weaker. The identity oracle's signal-to-noise is very function-dependent.

**M9.** That TMD-aware corpora can be curated per function. MPFR has pre-computed hard cases for SOME functions. For many functions, the hard case corpus has to be generated via our own adversarial search — which is a nontrivial Adversarial Mathematician project per function.

**M10.** That the profile IS an architecture-level commitment, not a test-suite-level one. A test suite is local (a single file); an architecture commitment shapes how downstream consumers query the library ("does this function pass the monotonicity oracle for my range?"). The second is expensive; the first is cheap. Phase 5 proposed the first but the *value* depends on the second.

### Phase 6.2 — Irreducible truths visible at this level

1. **A test is not a claim.** A test says "at these inputs, at this bound, our answer and the oracle agree." It does NOT say "our answer is correct on all inputs." The profile needs to distinguish TESTED properties from CLAIMED properties.

2. **Independence of oracles is a property, not a label.** Two oracles are independent iff they can disagree. If they can't disagree (because they share an implementation), they're one oracle wearing two costumes. The profile must *measure* independence, not assume it.

3. **The profile's trust value scales with its *distinguishing* power.** A profile that every libm passes with a perfect score is worthless — it's not distinguishing tambear from scipy. A profile where tambear passes and scipy fails (or vice versa) is the evidence of a quality difference.

4. **Curated corpora are a shared asset.** MPFR's TMD hard cases exist because someone did the work. Our curated corpus can be contributed back to the public pool, and new curated corpora from us are a strong artifact for the Tambear Contract's "publishable claims" deliverable.

5. **The Adversarial Mathematician's role is NOT a test-writing role — it's a bug-generation role.** Their value is not running oracles, it's generating the inputs that break oracles. Those inputs become part of the curated corpus. Separating "generate hard inputs" from "check if function handles them" is important because the first is creative, the second is mechanical.

6. **Oracle composition is a separate question.** If libm function A passes profile P_A and function B passes profile P_B, what does A∘B (as a numerical composition) satisfy? NO general answer. Composition of numerical functions loses precision; the profile of the composition has to be tested directly, not derived. This is a real limitation of the profile framework.

7. **"Correct-rounding on TMD corpus" is the highest-value single oracle.** Because it catches the bugs that only happen near fp64 midpoints, and because those bugs are rare but consequential (a single digit wrong in the last bit can flip a comparison). If the team can only add ONE oracle beyond pointwise-mpmath, the TMD corpus oracle is the answer.

8. **The profile is a *separator*, not a *certificate*.** It separates functions that pass into "good enough for X user class" buckets. It does NOT certify correctness. Users who need certificates should use formally verified libm (CRlibm + verification work), not tambear. Tambear can be honest about this.

### Phase 6.3 — Reconstructions of the Move

**Move v1 (original).** Declared multi-oracle suite. Profile per function. Minimum set listed.

**Move v2.** Same, but with *measured oracle independence* — the profile documents which oracles are actually independent of each other, so users can reason about what agreement between them implies.

**Move v3.** v2 plus *explicit distinction between TESTED and CLAIMED*. The profile has two sections: "tested against these oracles at these bounds" (empirical) and "claimed to satisfy these properties on these domains" (contractual). The contractual claim is the weaker of the empirical results and the declared domain limits.

**Move v4 (recommend).** v3 plus:
- A **shared corpus infrastructure** — `tests/libm_corpora/` with per-function TMD hard-case files, adversarial inputs, closed-form specials, identity test vectors. The Adversarial Mathematician owns the corpora. The Test Oracle owns the runner. They cooperate but their artifacts are separable.
- A **TestedVsClaimed** section in every function's rustdoc, with the two columns plus an `independence_matrix` showing which oracle pairs are genuinely independent for this function.
- A **profile_diff tool** — shows the profile delta between two versions of the same function. Makes "fixed bug in tam_exp regime boundary" a quantitative change, not a vague claim.
- An **oracle registry** mirroring the v5 OrderStrategy registry from Notebook 011 — each oracle has a name, a formal description, a reference implementation, and an independence table against other registry oracles. Uniformity between the I7′ move and the I9′ move helps the team maintain both.

### Phase 6.4 — The refined Aristotelian Move

> **I9′ (refined, v4):** Every libm function publishes a **correctness profile** with two columns — TESTED and CLAIMED. The TESTED column records each oracle run, its bound, and the corpus used. The CLAIMED column is the contractual promise (always weaker than TESTED by a declared safety margin on a declared domain). Oracles are registered in a shared `oracles/` registry parallel to the OrderStrategy registry; each entry names its reference implementation and its independence relations with other registered oracles. The minimum suite for a Phase-1 libm function is: closed-form specials + algebraic identity + pointwise vs mpmath + pointwise vs a genuinely independent arbitrary-precision reference (MPFR or Arb, NOT SymPy-via-mpmath) + monotonicity sweep across regime boundaries. Correct-rounding on a TMD-aware corpus is a *target*, function-by-function, not a minimum. The Adversarial Mathematician owns corpus curation; the Test Oracle owns the runner. A `profile_diff` tool makes profile changes between versions diffable.

This is the version to route to Libm Implementer + Test Oracle + Adversarial Mathematician.

---

## Phase 7 — Stability check

Run one more pass, adding v4's structure to the assumption list.

**New assumptions introduced by v4:**
- That "tested vs claimed" is a distinction users will read and understand.
- That independence matrices can be generated without excessive manual work.
- That the shared corpus infrastructure doesn't become a bottleneck.
- That profile_diff produces useful output (not just noise).

**Autopsy findings:**

- **Tested-vs-claimed readability.** Users who read rustdocs read declarative statements; a two-column table is not harder than a prose paragraph. Docs should default-show the CLAIMED column and let curious users expand to TESTED. Manageable.
- **Independence matrix generation.** For the minimum suite (5 oracles), the matrix is 5x5 = 10 pairs. For each pair, you ask: "if oracle A used mpmath or mpmath-derived data, and oracle B also did, they're dependent." Static analysis of the oracle registry can compute most of this automatically. Manual for edge cases. Acceptable.
- **Shared corpus bottleneck.** MPFR has ~100KB of TMD hard cases publicly; our curated corpus starts as a copy plus per-function additions. The bottleneck is CPU time for the nightly run, not storage or code. Profile-aware sampling (run full corpus weekly, subset nightly) mitigates.
- **profile_diff noise.** The tool should report CHANGES in oracle verdicts, not values. "Went from 0.8 ULP to 0.9 ULP on pointwise-mpmath" is noise. "Identity oracle started failing at x = 2π" is signal. The tool's value is in the filter, not the diff.

**Stability verdict:** v4 is a genuine engineering proposal with concrete costs and concrete mitigations. No new truths emerge. The deconstruction is **stable at v4**. No further recursion warranted.

One observation worth flagging: the I7′ move (Notebook 011) landed at v5, and the I9′ move (this document) lands at v4. Both moves converged on the same shape — **a named registry of formal-spec artifacts with capability metadata and per-operation declarations**. The I7′ registry is for order strategies; the I9′ registry is for oracles. The symmetry is not coincidental; it reflects the fact that both moves are about making implicit knowledge explicit and inspectable. Worth mentioning to navigator.

---

## Phase 8 — Forced Rejection

Forcibly reject the entire deconstruction. What would it mean to have NO oracles, NO correctness claims, NO profile — to simply publish the implementation as-is and let users decide?

### The void: no oracles at all

Suppose tambear-libm ships functions with no accompanying oracle, no published accuracy bound, no correctness promise. Each function is documented with its algorithm ("uses Cody-Waite argument reduction + degree-10 Remez polynomial + table-based reconstruction") and its source code. Users are told: "here's what we wrote, audit it if you care."

- **Users who care about correctness** read the source, run their own tests, decide whether to trust the function. Sophisticated users end up with more confidence than with a published "≤ 1 ULP" promise, because they've verified directly. Unsophisticated users either skip the check or trust us implicitly.
- **Bit-exact cross-hardware still works.** If every backend implements the same source, they produce the same bits, regardless of whether we claim mathematical correctness.
- **The Tambear Contract's "bit-perfect or bug-finding" promise becomes unverifiable.** We'd have to generate bugs in competitors ad-hoc rather than through a systematic oracle run. Still possible (the Adversarial Mathematician can still find bugs), but less scalable.
- **Regulatory/audit users** cannot use tambear. They need a paper trail, a test suite, a published bound. Without that, the library is unacceptable for audit-dependent work.
- **Research users** probably fine. They check what they care about and cite the source.

### What forced rejection reveals

**The oracle is a user-facing artifact, not a truth-seeking one.** When I proposed the multi-oracle profile, I framed it as "making the correctness claim honest." Phase 8 shows it's also — maybe primarily — **making the claim AUDITABLE by users who can't run their own oracles**. The profile is a trust interface.

**Implication:** The profile's DESIGN should optimize for auditability by specific user classes:
- **Research users** need: source + reference implementation + simple accuracy summary.
- **Production users** need: tested bounds + regression detection (profile_diff).
- **Regulated users** need: independence matrix + CLAIMED-column promise + versioning + diff tooling.
- **Downstream library consumers** need: composition semantics ("if I call `tam_exp(tam_log(x))`, what's my combined profile?").

The v4 profile serves all four, but unevenly. The weakest is composition — Phase 6.2 Truth 6 pointed out that composition of numerical functions loses precision, and no general derivation rule exists. That's a structural limit, not a design flaw. The profile framework should be honest about it: composition profiles are NOT derivable from component profiles; they must be tested directly.

### The unseen first principle surfaced by forced rejection

**Oracles are for USERS, not for TRUTH.** Truth is what the function computes, full stop. That's knowable from the source. What oracles provide is a *communication channel* between the library authors and users who can't audit directly. The multi-oracle profile is the right communication channel for a multi-class user base, but its purpose is communication, not verification.

This reframes I9′ slightly:

> **I9′′ (after Phase 8):** Every libm function publishes a correctness profile as its **auditability contract** — the structured communication channel that lets users of different rigor classes (research, production, regulated, composition-consuming) understand what the function guarantees for their use case. The profile's content is the TESTED/CLAIMED two-column structure from v4, plus explicit composition-safety notes. Its *purpose* is to make the function's mathematical behavior inspectable from outside the source code.

The renaming of the purpose — from "correctness claim" to "auditability contract" — changes nothing in the technical spec. But it clarifies WHY the profile exists and what it protects. Under pressure ("can we skip the monotonicity oracle for tam_sqrt since it's just hardware?"), the answer isn't "the invariant says so" — it's "removing it forfeits the production-class auditability guarantee, which the Tambear Contract promises."

---

## Status as of 2026-04-12

- Phases 1–8 drafted.
- **Final Move (v4):** Two-column profile (TESTED/CLAIMED), shared oracle registry paralleling the OrderStrategy registry, minimum suite defined, Adversarial Mathematician owns corpus curation, profile_diff tool, composition explicitly called out as non-derivable.
- **Phase 8 reframe:** The profile is an *auditability contract* for user classes, not a truth-seeking device. Changes the *purpose framing* without changing the technical spec.
- **Cross-target observation:** I7′ landed at a registry-of-OrderStrategy; I9′ landed at a registry-of-Oracles. Both moves converged on the same engineering pattern (named registry with formal specs + capability/independence metadata + reference implementations). The symmetry suggests this is the right shape for making tacit knowledge explicit across the trek.
- Second pass (Phase 7) found stability. No further recursion warranted.
- Ready to communicate the refinement to navigator.
