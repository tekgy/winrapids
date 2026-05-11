# Defaults are claims — Phases 1-8 deconstruction

**Created:** 2026-05-08
**Author:** aristotle (tambear-formalize team, post-survey)
**Inputs:**
- math-researcher's SURVEY.md at `R:\winrapids\campsites\tambear-formalize\survey\20260508123003-math-researcher\SURVEY.md` (lines 70-92, 140-205 specifically)
- math-researcher's assumption-docs/ (cody-waite-payne-hanek-crossover.md, trig-reduce-sharing.md)
- The Tambear Contract (CLAUDE.md §"The Tambear Contract")
- vocabulary.md Part II (behavioral stances)
- atoms-primitives-recipes.md ("Lowering strategies" + "Precision tag semantics")
- F1-F11.2 series (the formalization-process gap analysis from earlier today)

**Stress-test question (from math-researcher's task):** Is there a coherent
reading of "default" that ISN'T a claim? If yes, the proposed hard floor —
"the default declared in spec.toml MUST have a working implementation that
meets the claim that default makes" — is too strict. If no, F4 (zero
existing recipes ship the triplet) becomes a contract-violation audit, not
a stance-classification gap.

**The corollary under pressure-test:** "Aliasing collapses the claim space
silently. The default declared in spec.toml MUST have a working
implementation that meets the claim that default makes."

---

## Phase 1 — Assumption Autopsy

What the "defaults are claims" recognition silently assumes:

1. **A1.** `default` and `claim` are well-defined separately. (Claims live in
   `methods_template` ULP budgets and `description`; defaults live in
   `parameters.precision.default`. The recognition collapses them.)

2. **A2.** Every spec.toml has a `methods_template` field that makes ULP
   commitments. (math-researcher: yes — "Worst-case error budget: ≤ 2 ulps
   strict, ≤ 1 ulp correctly_rounded" is the recurring pattern.)

3. **A3.** The user calling `sin(x)` without an explicit `using(...)` causes
   the default-named strategy to fire. (Otherwise the default is decorative.)

4. **A4.** The strategy-named function (`sin_compensated(x)`) is bound to a
   specific implementation, not a redirection target. (Aliasing is exactly
   the violation of this assumption.)

5. **A5.** The user has a way to verify which strategy actually ran. (Tambear
   today: stack trace, which is opaque to a TBS-level user. So users may
   never know they got `_strict` when they paid for `_compensated`.)

6. **A6.** The cost of meeting the claim equals the cost of the
   spec'd strategy. (For Tang-1989 exp: strict = polynomial Horner;
   compensated = compensated Horner; correctly_rounded = DD-throughout. Real
   cost differential ~3-8x.)

7. **A7.** The promised tolerance bound is achievable by the spec'd
   strategy. (`_compensated` spec: ≤1.5 ulps. Tang-1989 compensated Horner
   achieves this on `[-π/4, π/4]`. So the promise is meetable in principle —
   but only if real compensated arithmetic is run, not aliased.)

8. **A8.** Defaults can be neutral. ("If you don't pick, here's what runs"
   sounds neutral — but the choice of WHICH "what runs" is itself a
   commitment. There is no coordinate-free "default.")

9. **A9.** Three different defaults shipping = three different products. (Or
   are they three different *configurations* of one product? Big difference
   for what gets versioned, what gets benchmarked, what gets audited.)

10. **A10.** Tests cover the claim, not the implementation. ("If
    `sin_compensated` is aliased to `_strict`, and `_strict` passes the
    `_compensated` ULP-bound oracle, is anything wrong?" — tests-serve-reality
    question.)

11. **A11.** Library callers care about the strategy choice. (Otherwise
    aliasing is fine and the only contract is "the function returns a number
    in the documented ULP envelope of *some* strategy.")

12. **A12.** The spec.toml's `methods_template` is enforced at build/CI time.
    (math-researcher's lint proposal: "verify the recipe ACTUALLY ships
    everything that section claims" — currently this is aspiration, not
    infrastructure.)

13. **A13.** The recognition/design distinction (F11) applies here. (A
    `default = "compensated"` declaration is recognition or design? See
    Phase 5.)

---

## Phase 2 — Irreducible Truths

Strip the assumptions. What survives?

1. **T1.** A spec.toml is a public artifact. Anyone can read what's
   committed. (T1 from F-series survey carries.)

2. **T2.** The Tambear Contract item 10 demands: "publication-grade rigor...
   benchmarked against every competing implementation," with bit-perfect or
   bug-finding outcomes. This applies to every recipe.

3. **T3.** The locked vocabulary defines "recipe" as Tier 4 with an oracle
   test, declared Kingdom, every-parameter-tunable, no-vendor-wrap. Each
   recipe is a single named composition with declared properties.

4. **T4.** A function call `sin(x)` produces some f64. That f64 either is
   within the ULP envelope of *some* mathematically-defined function or
   isn't. Independently of which strategy ran.

5. **T5.** Two distinct mathematical commitments — "≤2 ulps under strict
   evaluation" vs. "≤1.5 ulps under compensated evaluation" vs. "≤1 ulp
   under correctly_rounded evaluation" — can be simultaneously true for the
   same number on the same input. (E.g., a `correctly_rounded` answer
   trivially satisfies the looser bounds too.)

6. **T6.** A spec.toml that declares `default = "compensated"` is a
   public-readable artifact promising the call `sin(x)` selects the
   compensated path. The user reasonably reads this as "compensated runs."

7. **T7.** A unit test that asserts `(result - expected).abs() < 1.5e-16` on
   the implementation under `_compensated` does not, by itself, prove that
   compensated arithmetic was used. It proves the result was within bound,
   nothing more. The bound is a downstream observable; the strategy is
   upstream.

8. **T8.** Tests serve reality (CLAUDE.md). A passing test that doesn't
   detect aliasing is a test that doesn't validate the claim made by the
   spec.

---

## Phase 3 — Reconstruction from Zero (10 paths)

What does "default" actually mean? Ten readings of the same word, from
narrowest to widest:

1. **R1 — Default-as-defaulted-value.** `default` means the literal string
   value of the precision parameter when no `using()` overrides. No
   commitment about behavior. Just a string. The string happens to match a
   strategy name; that's a coincidence the SDK exploits at call time.
   (Defensible as the textbook reading of TOML parameter defaults — but see
   Phase 8.)

2. **R2 — Default-as-strategy-pointer.** `default = "compensated"` means
   "calling `sin(x)` invokes whichever fn the type system maps to the
   `compensated` enum branch." If that fn is aliased to `_strict`, the type
   system permits it; the contract is satisfied because the contract is a
   pointer, not a guarantee about the pointee's substance.

3. **R3 — Default-as-tolerance-budget.** `default = "compensated"` means
   "the result is within the `_compensated` ULP envelope." Strategy is
   implementation choice; what's promised is the bound. Aliasing to a
   stricter strategy (`_strict` with worst-case 2 ulps used to fulfill a
   `_compensated` 1.5-ulp commitment) is a *bound violation*, but aliasing
   to a looser strategy that happens to satisfy the bound on real inputs
   passes.

4. **R4 — Default-as-strategy-claim.** `default = "compensated"` means the
   compensated-arithmetic *implementation path* runs. The user's mental
   model is "I asked for compensated; compensated arithmetic happened."
   Aliasing is a violation regardless of whether bounds are met.

5. **R5 — Default-as-multi-claim.** `default = "compensated"` makes
   simultaneous claims: (a) about tolerance budget; (b) about
   implementation-path; (c) about cost (compensated is ~3x slower than
   strict — users may benchmark expecting that ratio); (d) about
   reproducibility across hardware (compensated arithmetic has different
   intermediate-state behavior under FMA than non-compensated). Aliasing
   collapses (b), (c), (d) silently.

6. **R6 — Default-as-recipe-identity.** `default = "compensated"` means the
   recipe's identity at this call site IS the compensated variant. Three
   defaults = three recipes (per F11.2's graph view: three distinct
   recognition-claim nodes with potentially distinct lineage and forcing
   chains).

7. **R7 — Default-as-tested-thing.** `default = "compensated"` means the CI
   harness tests this strategy by default, against this oracle, with this
   tolerance. Defaults define the test surface; aliasing makes one test
   surface fold onto another.

8. **R8 — Default-as-stance-declaration.** `default = "compensated"` is the
   recipe's *invocation stance* (per F2 + math-researcher's [stances]
   refinement): the "implements vs invoked-as-default" distinction. The
   default IS the invocation-stance commitment. Aliasing is a stance-claim
   violation.

9. **R9 — Default-as-publication-commitment.** `default = "compensated"`
   means a published paper using tambear's sin would cite the compensated
   strategy as the one used unless overridden. Aliasing is an academic
   misrepresentation — the published artifact says compensated; the run
   actually says strict.

10. **R10 — Default-as-budget-anchor.** `default = "compensated"` is a
    tambear-team-level commitment to where the rigor floor sits for this
    recipe. Aliasing-as-tolerated lowers the floor to "strict at the
    weakest"; the budget anchor is whatever the strictest strategy with a
    real implementation is. Default value is the anchor's name.

---

## Phase 4 — Assumption vs Truth Map

| Assumption | Survives deconstruction? | Replaced by truth |
|---|---|---|
| A1: `default` and `claim` are separate | NO under R3-R10 | T6: spec.toml IS the claim; default IS one of its components |
| A2: every spec.toml has methods_template ULP commitments | YES | T2 + T6 |
| A3: calling `sin(x)` without `using()` fires the default strategy | YES | T6 (the contract reading) |
| A4: strategy-named fn is bound to specific impl | NO under R2 | T7: aliasing is permitted by the language; banned only by contract-shape choice |
| A5: user has a way to verify which strategy ran | NO | T7: bound is observable; strategy is internal |
| A6: cost of meeting claim = cost of spec'd strategy | YES under R3-R5 (cost is part of the claim) | T2 (publication-grade includes benchmarks) |
| A7: promised tolerance is achievable | YES (verified in math-researcher's analysis for Tang 1989 + others) | T2 |
| A8: defaults can be neutral | NO under R3-R10 | T6: defaults select what to commit to |
| A9: three defaults = three products | YES under R6-R10 | T3: each default is a distinct recipe-instance |
| A10: tests cover claim not implementation | TENSION (this is the empirical question) | T7+T8: a test can pass without validating the claim shape |
| A11: callers care about strategy choice | YES under R3-R10; NO under R1-R2 | T2: publication-grade rigor implies callers SHOULD care |
| A12: methods_template is CI-enforced | NO (currently aspirational) | T8: tests-serve-reality demands they be enforced |
| A13: recognition/design lens applies | YES — see Phase 5 | F11 |

The biggest tensions:
- R1-R2 vs R3-R10: is a default a string or a commitment? This is the core
  question. Phase 8 (forced rejection) tests R1-R2 directly.
- T7 (bound observable, strategy internal) vs T8 (tests serve reality):
  current test infrastructure under-validates the claim space.

---

## Phase 5 — The Aristotelian Move

Apply the recognition/design lens (F11) to the question.

A `default = "compensated"` declaration is *which kind* of claim?

If it's a **design-claim** ("we, the authors, propose that compensated
should be the canonical strategy at this call site; alternatives considered
were strict and correctly_rounded; we picked compensated because..."), then
defaults *are* configuration, alternatives-analysis-shaped. The hard floor
is wrong; aliasing is permitted as long as the design rationale is
documented.

If it's a **recognition-claim** ("the structure of this recipe forces
compensated as the default — anything else either fails the bound on a
documented region of the input domain, or pays unnecessary cost on the
common case"), then defaults *are* claims, structurally-forced. The hard
floor is right; aliasing is a violation because the structural-forcing
argument that justified the recognition-claim wasn't honored in
implementation.

**Most spec.toml `default = X` declarations in the libm tree are
recognition-claims, not design-claims.** Evidence:

- The methods_template makes a ULP-budget claim. The default selects the
  strategy that meets that budget at the right cost-trade-off. The choice
  is forced by (Tang 1989 polynomial error budget) + (compensated Horner
  literature ULP table) + (real-input distribution), not by author
  preference.

- Three defaults aren't three equally-defensible alternatives — they're
  three points on a tolerance/cost Pareto frontier. The default selects the
  point. The Pareto frontier is a structural-forcing artifact.

- math-researcher's observation: `default = "compensated"` for sin is "a
  middle-budget commitment, neither the cheapest nor the strictest." That's
  a recognition (the optimum sits in the middle, given the rigor floor +
  cost ceiling).

So under F11's recognition/design lens, defaults in libm spec.tomls are
recognition-claims. Recognition-claims must anchor to structural-forcing.
The structural forcing here is: tolerance budget × cost × hardware-FMA
availability × DD-cost-amortization. The argument is real and walkable.

**The Aristotelian move:** treat every `default = X` declaration as a
recognition-claim, and require the structural-forcing argument as part of
the spec.toml. Concretely:

```toml
[parameters.precision]
default = "compensated"
default_recognition = """
Forcing: methods_template requires ≤1.5 ulps; strict polynomial
worst-case 2 ulps fails this. Correctly_rounded path costs ~3x more for
~0.5 ulp gain on 99.7% of input domain (mpmath survey, 1e7 samples).
Compensated Horner sits at the Pareto-optimal point: meets bound,
minimum cost. Forcing chain: claim_id `tang_1989_exp_polynomial_bound`
→ external `Muller HoFA ch.11.2.3 fig 11-7`.
"""
```

Under this move, the hard floor follows by force: aliasing the implementation
of the compensated strategy to strict means the Pareto-optimal point's
cost claim is silently violated, AND the bound-satisfaction is now
provided by a different strategy than the one whose forcing-argument the
spec.toml cited. The recognition-claim is invalidated; the spec.toml is
lying.

So: **defaults ARE claims, and aliasing IS a contract violation** —
provided the spec.toml is being read as a recognition-claim artifact under
F11's discipline.

The question of whether the tambear team is willing to commit to that
discipline is the real decision. Phase 5 says: yes, they should, because
the structural-forcing argument is genuinely walkable for libm and the
alternative is silent claim-collapse (the failure mode F4 already
documents).

---

## Phase 6 — Recursive Challenge

Add Phase 1-5 to the assumption pile. What did Phase 5 silently assume?

- **B1.** The structural-forcing argument is always available. (Counter:
  for some recipes the choice between strategies might be genuinely
  arbitrary — three defensible options with no structural reason to prefer
  one. In those cases, the default is a design-claim, not a
  recognition-claim, and the hard floor doesn't apply.)

- **B2.** Spec.toml authors will write the `default_recognition` field
  faithfully. (Counter: in practice, this field could become rote
  ritual — copy-pasted from sibling recipes without genuine forcing
  analysis. The discipline gets weaker over time.)

- **B3.** Tools enforce the discipline. (Currently no lint exists that
  cross-checks "the cited forcing chain actually walks to leaves" or
  "the claimed strategy actually runs.")

- **B4.** Recognition vs design is a binary. (Counter: most claims are
  mixed — recognition-substance with design-shape. F11.1 already noted
  this for documents; same principle here for individual claim entries.
  A `default = X` declaration might be recognition-justified at the
  budget level but design-justified at the variable-name level.)

- **B5.** All recipes have a clear Pareto frontier. (Counter: for some
  recipes, "compensated" might be strictly dominated by "strict" on
  every axis — same accuracy, lower cost. In that case, the default
  should obviously be strict; if compensated is offered at all, it's a
  design-claim about completeness-of-API, not a recognition-claim about
  optimality.)

Returning to T1-T8:
- T2 (publication-grade rigor) stands.
- T3 (recipe definition) stands.
- T6 (spec.toml IS the claim) stands.
- T7 + T8 (aliasing is permitted by language; banned by contract; tests
  serve reality) — these together generate the enforcement requirement
  (B3).

After Phase 6 the picture sharpens: **the hard floor IS too strict if
applied uniformly.** It's correct for recipes whose default is a
recognition-claim with a walkable forcing argument. It's wrong for
recipes whose default is a design-claim about API completeness or a
rote-pasted ritual.

The refined floor: *every recipe whose spec.toml default is presented as
a recognition-claim must have an implementation that actually meets the
claim that default makes; aliasing is permitted only when explicitly
declared as a stub in the spec.toml itself, with the design-claim
rationale.*

This is what the rare-trig case (versin/haversin/gudermannian aliasing)
should look like under correct discipline: the alias should be DECLARED
in spec.toml as a design-claim ("compensated path stubbed because
real-world inputs to versin are dominated by small-x where strict already
hits the bound; DD-throughout offers no measurable accuracy gain") with
the design rationale spelled out.

---

## Phase 7 — Recursive Process

Add B1-B5 to the assumption pile. Re-run.

- B1: counter-example acknowledged. The framework now distinguishes
  recognition-defaults (hard floor applies) from design-defaults (declared
  stubs permitted). Stable.
- B2: rote-ritual erosion is a real risk. Mitigation: same as F11.2 —
  forcing chains as graphs, with cycle-detection and external-citation
  density as quality metrics. A `default_recognition` field whose forcing
  chain has zero external citations within four hops triggers a CI
  warning. Stable.
- B3: enforcement requires tooling. Tooling cost is real but bounded
  (math-researcher's lint proposal for [stances] table extends naturally).
  Stable.
- B4: recognition/design is not binary. Confirmed by Phase 5's nested
  reading. The `default_recognition` field captures the recognition
  substance; if the substance is partial, the field documents what's
  recognition vs what's design. Stable.
- B5: dominated alternatives are real. The spec.toml should be allowed to
  declare "this strategy is dominated; provided for API completeness only"
  — that's a design-claim with structural-forcing about *the strategy
  family's API completeness*, not the strategy's optimality. Stable.

After Phase 7 no new principles emerge that aren't already in F1-F11.2 or
the Tambear Contract. STABLE.

The refined claim, ready to ship as F12:

> **F12.** Every recipe's spec.toml default selection is either a
> recognition-claim or a design-claim. Recognition-defaults must have
> implementations that meet the claim the default makes; aliasing is a
> contract violation under T8 (tests serve reality). Design-defaults
> (rare; e.g., dominated-strategy stubs for API completeness) must be
> explicitly declared as such in the spec.toml with their design
> rationale. Mixing the two without declaration is the failure mode
> visible in F4 (zero shipped recipes have non-aliased triplets) — and
> is a debt-audit, not a stance-classification gap.

---

## Phase 8 — Forced Rejection

Forcibly reject Phases 1-7. What if defaults are NEVER claims?

**Rejection 1 — defaults are pure configuration.** The spec.toml is just
a config file; `default = "compensated"` is the value picked when the
caller doesn't specify. That's all. Tests pass; bounds are met; nobody
is misled. The Pareto frontier framing is post-hoc rationalization.

What does the void look like?

- The methods_template stops making ULP commitments per-strategy. Either
  it makes no commitment (the recipe ships f64 numbers and the user
  decides if they trust them) or it makes one whole-recipe commitment
  ("this recipe satisfies ≤1 ulp under at-least-one strategy"). Either
  removes the per-strategy claim space.
- Spec.toml stops being a public truth artifact. It's the build-config
  for the recipe; users don't read it; it shapes CI and nothing else.
- Aliasing is fine because it's within-config; the user-facing surface
  is the function returning a number, and that number satisfies whatever
  bound the recipe-as-a-whole commits to.

Is this coherent? *Sort of.* It mirrors how scipy and numpy ship f64
math: there's a `numpy.sin`, you call it, you get f64s; numpy has a CI
suite that checks broad ULP envelopes; nobody reads numpy's internal
strategy tables. If tambear's spec.toml were treated this way, defaults
would be pure config.

**But** — and this is the load-bearing rejection — tambear's filter
test §10 demands "publication-grade rigor... benchmarked against every
competing implementation." Publication-grade rigor MEANS the methods
section says which strategy ran. A paper using tambear's sin must
report: which precision strategy was active. A paper that says "we
used `tambear::sin(x).using(precision='compensated')`" makes a claim
about compensated arithmetic. If aliasing happens, the paper is
incorrect — not in spirit, in fact.

So: defaults are NEVER claims is incoherent under filter test §10.
It IS coherent under a no-rigor-floor library (numpy, scipy). Tambear
chose the high-floor stance. The hard floor follows.

**Rejection 2 — claims live elsewhere; defaults live in toml.** Maybe
the claim about "compensated arithmetic happens" lives in the
methods_template prose, and the toml `default` field is just a binding
mechanism. So when spec.toml says `default = "compensated"`, the claim
about compensated isn't *made by the default field* — it's made by the
methods_template's "the default uses compensated Horner" sentence,
which the default field implements.

Under this rejection, the hard floor is misattributed: aliasing isn't a
default-field violation, it's a methods_template violation. The
default-as-claim framing is wrong; the methods-template-as-claim
framing is right.

What does this void look like?

- Default-field becomes mechanical (just a string).
- Methods_template becomes the locus of substance.
- The lint/audit shifts: instead of checking "default has a working
  implementation," we check "methods_template's described strategy
  matches the actual implementation."

This is a reasonable refactoring of the framing, but it doesn't change
the substance. The claim still exists; its location moves. F12 stands
with `methods_template` as the operative claim site instead of `default`,
or — more precisely — the spec.toml as a *whole* is the claim artifact
and individual fields are not claims independently.

Refined F12:

> **F12 (refined).** A spec.toml is a public claim artifact. The pair
> (methods_template prose + parameters.precision.default value) jointly
> commits the recipe to a strategy at the default call site. Aliasing
> the strategy's implementation to a different strategy violates this
> joint commitment. Whether to call this a "default field" violation or
> a "methods_template" violation is a framing choice; the substance is
> identical: the recipe at its default invocation runs the strategy the
> spec describes.

**Rejection 3 — the spec.toml lies. So what?** What if we accept that
the spec.toml is aspirational, that aliasing is a known temporary state
during incremental implementation, that the discipline is "be honest
about where you are" not "ship with full triplets always"?

Under this rejection, aliasing is fine if it's documented somewhere
(commit message, GAP register, TODO comment). The hard floor is
softened to "either ship the triplet OR ship a tracked GAP that the
team is working on."

This is actually how DEC-002 (no tech debt) interacts with reality: tech
debt that's tracked and being worked on is acceptable in transit. The
question is whether spec.toml-aliasing-without-disclosure crosses the
line into actual debt.

For the rare-trig cases (versin/haversin/gudermannian), math-researcher's
SURVEY notes the alias state is "honest stubs (no silent precision
claim)." That's the in-transit-with-disclosure case. F12 should
acknowledge it: aliasing is permitted iff disclosed (in-spec.toml
declaration that this strategy is stubbed/aliased + rationale).

For sin/cos/asin/etc. (which math-researcher's survey marks as having
real `_compensated` and `_correctly_rounded`), the question doesn't
arise. Aliasing isn't happening there.

**Rejection 4 — defaults shouldn't exist.** Make every call explicit:
no `default = X`; users always pick. Then there's no claim made by
absence; all claims are at call sites.

What does the void look like? `sin(x)` becomes a compile error;
`sin(x).using(precision="compensated")` is required. This is the
maximally-explicit world. It's defensible (every call-site documents
its assumption) but ergonomically heavy.

Under this rejection, tambear ships with no call-site default
behavior, which means recipes are always invocation-explicit. That's a
substantive design choice. It would dissolve the entire problem (no
default → no default-as-claim question). It's also a substantial UX
shift from current API.

What MUST exist if Rejection 4 is taken seriously? Every TBS-level
caller would need to make precision explicit. Every spec.toml's
parameters.precision section would lose `default = X` and gain
`required = true`. The IDE/REPL would prompt at first use. This is
heavyweight but coherent.

Worth flagging as a real alternative the team could pick. If picked,
F12 rewrites: "since defaults don't exist, the question doesn't arise;
every call carries its own claim."

**Rejection 5 — claims aren't binary.** Maybe aliasing isn't a
violation OR non-violation; it's a degree-of-faithfulness question.
A `_compensated` aliased to `_strict` is "less compensated" than a
real implementation but "more compensated" than no implementation at
all. The world admits a continuum.

Under this rejection, the hard floor becomes a soft floor with a
quality metric: "claim-fidelity" = (degree to which actual
implementation realizes the claimed strategy). Range: 0 (alias to
unrelated strategy) to 1 (full compensated arithmetic throughout).
Spec.toml gets a `claim_fidelity` field; CI tracks it; the audit is
quantitative not boolean.

This is structurally appealing — it admits incremental progress
(fidelity moves from 0.6 to 0.9 over time) and doesn't force a hard
gate. It also leaves room for the F12 hard-floor reading at the
limit (`claim_fidelity = 1.0` required for production claims).

Worth keeping in the toolkit. If F12's binary framing turns out to be
too strict in practice, the continuous reading is the soft-fallback.

---

## What MUST exist under these rejections

Walking through what each rejection forces:

- **Rejection 1** (defaults never claims): tambear should NOT call itself
  publication-grade. Either soften the contract or accept the hard floor.
- **Rejection 2** (claims live elsewhere): F12 stands; locus of claim
  moves to spec.toml-as-whole, not just default field.
- **Rejection 3** (spec.toml lies are tracked debt): F12 amended —
  aliasing is permitted with explicit declaration; hidden aliasing is the
  violation.
- **Rejection 4** (defaults shouldn't exist): F12 dissolves; every call
  is invocation-explicit. This is a real alternative; team should
  consider.
- **Rejection 5** (claims are continuous): F12 becomes a fidelity metric
  with hard floor at fidelity = 1.0 for production claims.

The intersection of rejections that survive: **F12 is correct in
substance. Its precise wording should incorporate Rejections 2 and 3 —
the claim is borne by the spec.toml as a whole (not just the default
field); declared aliasing is permitted; undeclared aliasing is the
violation.**

Final F12, ready to ship:

> **F12 (final).** A spec.toml is a public claim artifact. The
> declaration `parameters.precision.default = X` together with the
> `methods_template` ULP commitments jointly commits the recipe at
> default invocation to running strategy X with X's documented
> tolerance budget and X's documented cost characteristics. The
> implementation MUST realize this commitment for every recipe whose
> spec.toml does not explicitly declare otherwise. Permitted exception:
> a spec.toml may declare a strategy as `aliased_to = Y` or
> `stubbed_pending = true` with rationale, in which case the alias is
> documented design-claim rather than hidden recognition violation.
> Undeclared aliasing is a contract violation under T8 (tests serve
> reality) and the Tambear Contract item 10 (publication-grade rigor).

---

## Direct answer to math-researcher's stress-test question

> "Is there a coherent reading of 'default' that ISN'T a claim?"

**There are exactly two coherent readings under which `default` is not
a claim:**

1. **Tambear softens the contract.** If the publication-grade-rigor
   filter test floor is lowered to scipy/numpy-style "broad envelope per
   recipe, internals opaque," then defaults become pure config. This
   requires changing the contract; it's coherent but contradicts what
   tambear-the-product is trying to be.

2. **Tambear removes defaults.** If every call must specify
   `using(precision=X)`, defaults don't exist; the question dissolves.
   This is coherent and substantive; it shifts UX heavily but
   eliminates the silent-claim-collapse class entirely.

Outside these two, every coherent reading has defaults carrying
substantive claims (Rejections 2, 3, 5 each refine the claim's
location, declaration discipline, or boundary-shape — but not its
existence). Under tambear's current contract, with defaults present,
they ARE claims. The corollary stands: **the default declared in
spec.toml MUST have a working implementation that meets the claim that
default makes** — provided that "must" admits documented exception via
explicit declaration of aliasing as a design-claim rather than as a
hidden state.

For F4: the existing recipes that ship undeclared aliases (the rare-trig
cases that math-researcher's SURVEY identifies) ARE in contract
violation. The fix is straightforward: either implement the aliased
strategies for real, OR add a spec.toml declaration documenting the
alias as a design-claim with rationale. Both are acceptable; the
silent in-between is not.

---

## Cross-references

- F11 — recognition vs design as core distinction; F12 is its
  specific application to the default-field question.
- F8 — claim-shape spec docs; F12 makes the spec.toml's claim
  explicit and falsifiable.
- F4 — zero shipped recipes have non-aliased triplets; F12 reframes
  this as a contract-violation audit, not a stance-classification gap.
- math-researcher's [stances] proposal — F12 lives in the
  `[stances.override_transparency]` block: each strategy in the
  `strategies` array must have a real implementation OR an
  `aliased_to`/`stubbed_pending` declaration.
- math-researcher's lint proposal (CI cross-check that recipe
  ships everything spec.toml claims) — F12 is the substance the lint
  enforces.

---

## Status

Phases 1-8 walked once; recursion converged at Phase 7. Forced
rejection at Phase 8 surfaced two coherent alternative framings
(Rejections 1 and 4) and three internal refinements (Rejections 2, 3,
5).

The refined claim F12 carries forward to navigator's synthesis as the
twelfth headline finding, completing the F-series at F12 unless
follow-on dialog surfaces F13.

Final disposition of math-researcher's two cited statements:

- **"Defaults encode tolerance-budget commitments. Tolerance-budget is
  a claim. So defaults ARE claims, not configuration."** — Confirmed.
  Survives Phase 8.

- **"Aliasing collapses the claim space silently. The default declared
  in spec.toml MUST have a working implementation that meets the claim
  that default makes."** — Refined. The "MUST" is correct for
  undeclared aliasing; explicit declaration of aliasing as a
  design-claim is permitted. The corollary stands with this
  amendment.

Cross-link from math-researcher's SURVEY.md (per their request).
Available for further deconstruction on demand.
