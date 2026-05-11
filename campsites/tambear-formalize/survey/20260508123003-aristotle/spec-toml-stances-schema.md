# Spec.toml Stances Schema — F12 Operationalization

**Created:** 2026-05-08
**Author:** aristotle
**Status:** draft for math-researcher review + navigator routing
**Co-authors-in-substance:** math-researcher (F12 stress-test, audit, schema seed; flag 2 = F12.1 prose audit; flag 3 = sweep-prerequisite framing); naturalist (F11 recognition/design discipline; F11.1 sub-document granularity)
**Inputs:**
- F11/F11.1/F11.2/F11.3 (recognition/design + granularity + graph-form + cross-team coordination)
- F12 deconstruction at `default-is-a-claim.md` (Phases 1-8 with adversarial Phase 8 forced rejection)
- Math-researcher's F4 audit (~48 of 50 libm recipes ship undeclared aliases)
- Math-researcher's SURVEY.md per-recipe stance audit
- Reference spec.toml at `R:\winrapids\crates\tambear\src\recipes\libm\asin.spec.toml` (worked exemplar)
- Tambear Contract item 10 (publication-grade rigor) + DEC-002 (no tech debt) + DEC-006 (locked vocabulary)

---

## 1. Purpose

This document operationalizes F12 ("defaults are claims") into a mechanical CI-checkable schema for tambear spec.toml files. It serves three audiences:

1. **Pathmaker** running the libm formalization sweep — defines the `[stances]` block every formalized recipe must include.
2. **Math-researcher and adjacent recipe authors** — defines what `aliased_to` and `stubbed_pending` declarations look like, with rationale requirements.
3. **CI tooling (lint to be built)** — defines the validation rules that catch undeclared aliasing as a contract violation per F12.

**This is a sweep-prerequisite document, not a sweep-deliverable.** The schema must exist before the libm formalization sweep starts; pulling recipes into `R:\tambear` without the schema in place re-imports the F4-audit-discovered violations intact (30 of 32 libm recipes — math-researcher's corrected count post-grep — claim strategies that don't exist; only `exp` and `log` ship real triplets).

---

## 2. Recognition the schema operationalizes

Per F12 (final, post-Phase-8): a spec.toml is a public claim artifact. The declaration `parameters.precision.default = X` together with the `methods_template` ULP commitments jointly commits the recipe at default invocation to running strategy X with X's documented tolerance budget and X's documented cost characteristics. The implementation MUST realize this commitment for every recipe whose spec.toml does not explicitly declare otherwise. Permitted exception: a spec.toml may declare a strategy as `aliased_to = Y` or `stubbed_pending = true` with rationale.

Per F12.1 (extending F12 from math-researcher's flag 2): every claim-shaped statement in the spec.toml is bound by F12, including methods_template prose ULP bounds, parameter description claims, output guarantees, and oracle-comparison pledges — not just the `default` field.

Per F11's recognition/design discipline: defaults in libm spec.tomls are recognition-claims (not design-claims) because the choice of strategy is structurally forced by tolerance budget × cost × hardware-FMA availability × DD-cost-amortization, not by author preference. Recognition-claims must anchor to structural-forcing arguments; the schema captures this by requiring `aliased_to` and `stubbed_pending` declarations to carry rationale.

---

## 3. Schema specification

### 3.1 Required block: `[stances]`

Every spec.toml that declares a `precision` parameter MUST include a `[stances]` block. The block enumerates which stances the recipe IMPLEMENTS (the static capability declaration, per math-researcher's "implements vs invoked-as-default" distinction) and what strategies each stance offers.

```toml
[stances]
implements = ["pure_math", "override_transparency"]
invoked_default = "override_transparency"
```

- `implements` — list of stances the recipe ships. At minimum `["pure_math"]`. May include `["override_transparency"]` if the strategy triplet is exposed via `precision` parameter. May include `["diagnostic"]` or `["discovery"]` for richer stance shapes (rare for libm; common for workflow recipes).
- `invoked_default` — the stance fired at the default call site (`recipe(x)` with no `using()`). Must be present in `implements`.

### 3.2 Per-strategy state declarations

Every strategy listed in the `precision` parameter's domain MUST have a corresponding `[stances.override_transparency.strategy.<name>]` entry. (The schema generalizes for non-precision strategies — see §3.5 — but the libm case is the canonical use.)

```toml
[stances.override_transparency.strategy.strict]
state = "real"

[stances.override_transparency.strategy.compensated]
state = "aliased_to"
target = "strict"
rationale = """
The strict polynomial already meets the design budget on the recipe's
input domain. Forcing chain: the Tang-1989 polynomial evaluation with
80-digit-mpmath-Remez coefficients on [-π/4, π/4] satisfies ≤ 2 ulps;
compensated arithmetic on the same coefficients gains ~0.5 ulps for ~3x
cost. For consumers without 1-ulp pressure, this trade-off does not
favor compensated. Real DD path is a follow-on sweep (DD_libm) priority
ranked by callsite-precision audit.
"""
fidelity_target = 1.0   # what real_fidelity should be when implemented
sweep_blocking = "DD_libm"  # if stubbed, which sweep ratifies the real impl

[stances.override_transparency.strategy.correctly_rounded]
state = "stubbed_pending"
sweep = "DD_libm"
rationale = """
1-ulp guarantee requires DD-throughout polynomial evaluation. Not yet
implemented. See DD_libm sweep for landing schedule. Currently a
declared stub: callers that explicitly request correctly_rounded will
receive an error, NOT silent fallback to strict.
"""
```

### 3.3 The three states

```
state ∈ {real, aliased_to, stubbed_pending}
```

**`real`** — the recipe ships an actual implementation of this strategy. The pub fn `<recipe>_<strategy>(x)` exists and contains the actual algorithm (not a forwarder). CI validates by checking the function body is non-trivial and references compensated/DD primitives appropriate to the strategy.

**`aliased_to`** — the recipe declares this strategy alias to another strategy that IS implemented. Required fields:
- `target` — the strategy alias points to. Must be a real-state strategy in the same recipe, OR a real-state strategy in another recipe (cross-recipe aliasing, rare).
- `rationale` — structural-forcing argument for why the alias is acceptable. Must include: (a) what design budget the target strategy meets that satisfies this strategy's claim; (b) why implementation of this strategy independently is not currently warranted; (c) reference to where the cost-benefit decision was made (typically a sweep doc or a prior decision).
- `fidelity_target` (optional) — the `real_fidelity` value when the alias is replaced by a real implementation. Used for migration tracking.
- `sweep_blocking` (optional) — which sweep is gating implementation. Used for CI to surface "this alias is on the migration path."

**`stubbed_pending`** — the recipe declares this strategy is not yet implemented. Calling it returns an error (NOT silent fallback). Required fields:
- `sweep` — which sweep will land the real implementation. Must reference an existing sweep in `R:\tambear\sweeps\` or equivalent.
- `rationale` — why the stub exists rather than full implementation now. May reference design considerations (the strategy is rarely-needed) or implementation considerations (DD substrate not yet ready in the locked-vocabulary library).

### 3.4 Strict default: undeclared = real

If a strategy is listed in the `precision` parameter's domain but has NO `[stances.override_transparency.strategy.<name>]` entry, CI defaults to `state = "real"` and validates the implementation. **This is the critical default for catching the 48-of-50 silent-collapse failure mode.** A pre-lock recipe with `precision` parameter declaring `["strict", "compensated", "correctly_rounded"]` and no `[stances]` block will fail lint on the (non-existent) compensated and correctly_rounded implementations, surfacing the violation.

### 3.5 Generalization to non-precision strategies

The schema generalizes for stances beyond `override_transparency`. Diagnostic recipes use `[stances.diagnostic.method.<name>]` blocks; discovery recipes use `[stances.discovery.method.<name>]`. Each method has the same `state ∈ {real, aliased_to, stubbed_pending}` requirement. Out of scope for this document; libm exclusively uses `override_transparency`.

---

## 4. F12.1 prose-audit rules

Every prose claim in the spec.toml that mentions a strategy or commits to a tolerance MUST match the actual strategy capabilities. CI extracts and validates the following claim sites:

### 4.1 `[parameters.<name>] description` claims

Description prose for the precision parameter typically says:

```
• strict: ... ≤ 2 ulps.
• compensated: ... ≤ 1 ulp.
• correctly_rounded: ... ≤ 1 ulp on tested samples.
```

The lint rule: every per-strategy bullet that asserts a ULP bound MUST be backed by the strategy's actual implementation OR by an explicit `aliased_to`/`stubbed_pending` declaration that surfaces the discrepancy.

If a description prose says "compensated: ≤ 1 ulp" and the strategy is declared `aliased_to = "strict"` (which has ≤ 2 ulps), the prose MUST be updated to reflect the alias. Possible options:
- `compensated: aliased to strict; ≤ 2 ulps. (Real DD path follows in DD_libm sweep.)`
- `compensated: not currently differentiated from strict; ≤ 2 ulps.`

The CI's prose-audit rule does not need NLP — it parses bullet points keyed by strategy names and matches the ULP claim against the strategy's state-block bound.

### 4.2 `[writeup.methods_template]` prose

The methods_template prose is what gets included in published reports / papers using this recipe. Per F12, this prose IS a public claim. CI validates that every `{precision}`-templated reference in the prose has corresponding documented tolerance for the actual strategy that fires.

For libm recipes, `methods_template` typically reads:

```
"Precision strategy: {precision}. ... [implicitly carries the bullet-list claim]"
```

If `parameters.default.using = "compensated"` and `compensated` is `aliased_to = "strict"`, then the methods_template's `{precision}` substitution renders as `"compensated"` in the report — but the actual computation ran strict. Per F12, this is a published-claim violation. The lint rule: methods_template MUST reference the *real* strategy fired, not the alias declaration. Possible updates:
- Render the alias chain: `"Precision strategy: compensated (aliased to strict; ≤ 2 ulps)."`
- Force `parameters.default.using = "strict"` if compensated is aliased.

This second option is structurally cleaner: **a recipe whose default-strategy is aliased should change its default to point at the real strategy.** That brings F12 compliance with minimal changes — the prose stops claiming a strategy that doesn't fire. The cost: the recipe now defaults to strict instead of compensated, which is a UX-visible change. Most callers won't notice; some who reasoned about default = compensated may be surprised.

### 4.3 `[[outputs]] description` claims

Output description prose may say e.g., "asin(x) ∈ [−π/2, π/2]" with implicit precision. If the recipe's output range tightens for one strategy versus another, the description must call this out per-strategy. (Rare for libm; more common for workflow recipes.)

---

## 5. Dual fidelity metrics

Per math-researcher's refinement, `claim_fidelity` is a dual metric, not a single number:

### 5.1 `declared_fidelity`

```
declared_fidelity = (count of strategies with explicit state block)
                  / (count of strategies in domain)
```

A recipe is F12-compliant iff `declared_fidelity = 1.0`. The CI hard-fails on `declared_fidelity < 1.0`.

### 5.2 `real_fidelity`

```
real_fidelity = (count of strategies with state = "real")
              / (count of strategies in domain)
```

A recipe at `(declared_fidelity = 1.0, real_fidelity = 1.0)` ships a complete, transparent triplet (e.g., exp.rs + log.rs in winrapids today).

A recipe at `(declared_fidelity = 1.0, real_fidelity = 0.33)` is transparent but incomplete (e.g., post-migration sin.rs with one real strategy and two declared aliases).

A recipe at `(declared_fidelity = 0.0, real_fidelity = 0.33)` is the F12 violation case — same headline 0.33 but completely opaque about which strategy is real.

### 5.3 CI output

For every recipe, the CI emits a record:

```yaml
recipe: sin
declared_fidelity: 1.00
real_fidelity: 0.33
strategies:
  strict: real
  compensated: aliased_to(strict)
  correctly_rounded: stubbed_pending(DD_libm)
prose_audit: PASS
issues: []
```

Aggregate metrics across the catalog:
- Mean `declared_fidelity` (must be 1.0 for the locked-vocabulary library to be F12-compliant)
- Mean `real_fidelity` (engineering progress signal; goes up over time as DD paths land)
- Count of strategies in each state (snapshot of completeness)

---

## 6. Migration paths for the 48 violations

Every libm recipe currently in pre-lock violation has three resolution paths under F12. The migration table maps each recipe to one path:

### 6.1 Path (a) — implement the aliased strategy

Build the actual DD-arithmetic implementation; flip state to `real`. High cost (weeks per recipe in some cases); only warranted for recipes where downstream consumers need the precision strategy.

**Selection criterion:** any recipe with a downstream caller in winrapids that uses `using(precision="compensated")` or `using(precision="correctly_rounded")` explicitly.

### 6.2 Path (b) — declare the alias

Add `[stances.override_transparency.strategy.<name>] state = "aliased_to" target = "strict" rationale = "..."`; update prose to reflect alias. Mechanical (~10 minutes per recipe). F12-compliant.

**Selection criterion:** the recipe has no consumers requesting the aliased strategy explicitly, and the strict polynomial design budget meets the recipe's stated tolerance.

### 6.3 Path (c) — narrow the parameter domain

Remove the `precision` parameter declaration entirely. The recipe ships one strategy; no triplet claim is made. Calls like `recipe(x).using(precision="compensated")` become errors at the parameter validation layer (compensated isn't a domain value), surfacing the lack of triplet to the caller as an explicit choice.

**Selection criterion:** the recipe is so simple that triplet exposure adds no value (e.g., the rare `versin` family — the integer-landmark cases are already exact; the general path delegates to sin/cos which carry their own precision contract; offering versin's own precision parameter is API noise).

### 6.4 Recommended migration table for libm (math-researcher's audit data, finalized)

Math-researcher's audit (post-draft) provided the two pieces of data §6.4 was waiting for:

**Q1 — Which spec.tomls declare a `precision` parameter at all?** ALL 32 libm spec.tomls in `R:\winrapids\crates\tambear\src\recipes\libm\` declare the precision triplet `["strict", "compensated", "correctly_rounded"]`. None are exempt at the spec-declaration level.

The 32 spec.tomls (corrected count, replacing math-researcher's earlier 44):

```
acos, acosh, acospi, acot, acsc, asec, asin, asinh,
asinpi, atan, atan2, atanh, atanpi, cos, cosh, cospi,
cot, csc, exp, gudermannian, haversin, inv_gudermannian,
sec, sin, sincos, sincospi, sinh, sinpi, tan, tanh,
tanpi, versin
```

**Q2 — Which downstream consumers use `using(precision=...)` explicitly?** Effectively zero. No production callsite in winrapids' `recipes/statistics/` or `recipes/pipelines/` references `_compensated` or `_correctly_rounded` libm functions explicitly. The only matches are libm-internal cross-strategy comparison tests in `exp.rs` and `log.rs` (the only recipes with real triplets) and one default-declaration test in `pipelines/schema.rs`.

**Implication:** path (a) is currently speculative-precision-supply (no consumer demand); path (b) is justified for every aliased recipe; path (c) is theoretically available everywhere but might over-narrow the future-extension surface.

Finalized migration table:

| Recipe family | Recommended path | Rationale |
|---|---|---|
| `exp`, `log` (2) | Path (a) — already real | Phase C1-C2 shipped real triplets; declare `state = "real"` for all three |
| `sin`, `cos`, `sincos`, `tan` (4) | Path (b) for v1; path (a) eventual | No current consumer; future path-a candidates as kernel composers (versin/haversin/atan2pi) — flag for DD_libm sweep when demand surfaces |
| `asin`, `acos`, `atan`, `atan2` (4) | Path (b) | No current consumer; strict polynomial meets design budget per refit-Remez evidence |
| `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` (6) | Path (b) | Same |
| `sec`, `csc`, `cot`, `asec`, `acsc`, `acot` (3 in libm + 3 in inv_recip; ~6) | Path (b) | Composed from sin/cos/tan/asin/acos/atan; precision flows through composition |
| `sinpi`, `cospi`, `tanpi`, `atan2pi`, `asinpi`, `acospi`, `atanpi` (~7) | Path (b) | Exact landmarks bit-exact regardless; general path delegates |
| `versin`, `haversin`, `gudermannian`, `inv_gudermannian` (4) | Path (b) | Composed; same logic |
| `sincos`, `sincospi` (fused pairs) | Path (b) | Both members alias the same way |

**Bulk: 30 of 32 recipes → path (b).** Cost per recipe: one `[stances]` block with rationale. ~10 minutes mechanical work per spec.toml, or one well-tested template applied to all 30 with per-recipe rationale customization. **Total path-b migration is hours, not weeks.** Eligible to be done as a single pre-sweep substream.

The 2 exceptions (`exp`, `log`) ship real triplets already; their migration is just adding the `[stances]` declaration to surface what's already real. No code work; spec-declaration-only.

**Path (a) and path (c) are not currently selected for any recipe.** Path (a) is speculative-precision-supply with no consumer demand to justify the implementation cost. Path (c) (narrowing the parameter domain) is too aggressive — losing the triplet shape removes the future-extension surface, and path (b) achieves transparency without giving up the API.

---

## 7. CI lint contract (mechanical enforcement)

A `cargo tambear-stance-lint` (or equivalent) tool reads every `*.spec.toml` and validates:

### 7.1 Schema validation

For every recipe with a `precision` parameter:
- `[stances]` block exists.
- `[stances.implements]` is non-empty.
- `[stances.invoked_default]` is present in `[stances.implements]`.
- For every strategy in `parameters.precision.domain.values`, a `[stances.override_transparency.strategy.<name>]` block either exists or defaults to `state = "real"`.
- Every state-block has `state ∈ {real, aliased_to, stubbed_pending}`.
- `aliased_to` blocks have `target` (must be a real strategy in the same recipe) + `rationale` (non-empty).
- `stubbed_pending` blocks have `sweep` (must reference an existing sweep dir) + `rationale`.

### 7.2 Implementation validation

For every strategy declared `state = "real"`:
- The pub fn `<recipe>_<strategy>(x)` exists.
- The function body is non-trivial (more than a single forwarding call to `<recipe>_<other>(x)`).
- The function references compensated/DD primitives appropriate to the strategy:
  - `compensated`: must reference `compensated_horner`, `two_sum`, `two_product_fma`, or similar compensated primitives.
  - `correctly_rounded`: must reference `dd_*` primitives (DoubleDouble arithmetic).

### 7.3 Prose audit validation

For every per-strategy bullet in `parameters.<precision-param>.description`:
- The asserted ULP bound matches the strategy's actual capability per its state.
- For `state = "aliased_to"`, the prose must surface the alias (e.g., "compensated: aliased to strict; ≤ 2 ulps").
- For `state = "stubbed_pending"`, the prose must surface the stub (e.g., "correctly_rounded: not currently implemented (DD_libm sweep)").

For `[writeup.methods_template]`:
- Any strategy referenced must be in a real or transparently-disclosed state.
- If the default-fired strategy is aliased, methods_template should render the alias chain OR the spec.toml should change `parameters.default.using` to point at the real strategy.

### 7.4 Aggregate metrics

The lint emits the dual fidelity metrics per recipe and aggregates across the catalog. Recipe records persist in a `target/tambear-stance-report.toml` artifact, used by:
- The formalize sweep's pathmaker as scope-of-debt input.
- The synthesis-doc author for tracking F12-compliance progress over time.
- Any future paper's methods-section auto-generator (the rendered methods prose for each recipe at any point in time).

---

## 8. Worked example: asin.spec.toml under F12

Reference spec.toml at `R:\winrapids\crates\tambear\src\recipes\libm\asin.spec.toml` currently declares:

```toml
[parameters.default]
using = "compensated"

[parameters.domain]
kind = "enum"
values = ["strict", "compensated", "correctly_rounded"]
```

with description prose:

```
• strict: single-FMA Horner polynomial, standard sqrt. ≤ 2 ulps.
• compensated: compensated Horner + DD-corrected sqrt. ≤ 1 ulp.
• correctly_rounded: DD working precision throughout. ≤ 1 ulp on tested samples.
```

Per math-researcher's F4 audit, `asin_compensated` and `asin_correctly_rounded` are aliased to `asin_strict`. **F12 violation: 2 of 3 strategies undeclared aliased. Prose claims ≤ 1 ulp for the compensated path that isn't honored.**

Applying path (b) under this schema:

```toml
[stances]
implements = ["pure_math", "override_transparency"]
invoked_default = "override_transparency"

[stances.override_transparency.strategy.strict]
state = "real"

[stances.override_transparency.strategy.compensated]
state = "aliased_to"
target = "strict"
rationale = """
The fdlibm-lineage rational P/Q polynomial on |x| ≤ 0.5 is fit at 80-digit
mpmath-Remez precision; the strict implementation worst-case is ≤ 2 ulps.
Compensated arithmetic on the same coefficients gains ~0.5 ulps for ~3x
cost; for asin's typical use (geographic, geometric, angle-of-incidence
calculation), the additional precision rarely matters relative to the input
data quality. Real DD path follows in DD_libm sweep when consumer demand
is established.
"""
fidelity_target = 1.0
sweep_blocking = "DD_libm"

[stances.override_transparency.strategy.correctly_rounded]
state = "stubbed_pending"
sweep = "DD_libm"
rationale = """
1-ulp guarantee requires full DD-throughout for the rational P/Q evaluation
plus DD-corrected sqrt for the |x| > 0.5 half-angle path. Not yet
implemented. Calling asin(x).using(precision='correctly_rounded') currently
returns an error.
"""
```

And the parameter description must be updated:

```
• strict: single-FMA Horner polynomial, standard sqrt. ≤ 2 ulps.
• compensated: aliased to strict; ≤ 2 ulps. (DD path follows in DD_libm.)
• correctly_rounded: not currently implemented; DD_libm sweep schedules.
```

And `parameters.default.using = "compensated"` should change to `"strict"`, OR the methods_template prose should render the alias chain. The cleaner option: change the default to `"strict"` to keep methods_template simple. (This is a UX-visible change; documented in the migration table.)

After this migration, asin's CI record reads:

```yaml
recipe: asin
declared_fidelity: 1.00
real_fidelity: 0.33
strategies:
  strict: real
  compensated: aliased_to(strict)
  correctly_rounded: stubbed_pending(DD_libm)
prose_audit: PASS
issues: []
```

F12-compliant. Transparent. Path-(b) migration complete.

---

## 9. Why this is sweep-prerequisite, not sweep-deliverable

The schema MUST exist before pathmaker pulls libm into `R:\tambear`. Reasons:

1. **Without the schema, the lint doesn't exist.** Pathmaker pulling 30 violations (math-researcher's audit count: 30 of 32 libm recipes) into the locked-vocabulary library means 30 silent F12 violations bake in. Zero CI coverage to flag them.

2. **The migration choices (path a/b/c per recipe) shape the implementation work.** A recipe taking path (a) needs DD primitives + algorithm work. A recipe taking path (b) needs only spec.toml updates. A recipe taking path (c) needs parameter-domain narrowing AND any downstream consumer fixes. Pathmaker can't plan the sweep without knowing the migration table.

3. **The dual fidelity metrics are needed for sweep progress tracking.** Pathmaker's commit-per-substream discipline (per SWEEP_PLAYBOOK.md) requires tests-green-at-each-commit. Without `declared_fidelity = 1.0` as the F12-compliance bar, "green tests" doesn't capture F12 compliance.

4. **The recognition/design boundary (F11) demands operationalization.** Per math-researcher's meta-principle ("recognition lands with operationalization"), F12 floats as principle if the schema doesn't exist. F12-as-floating-recognition will erode within sweeps as ad-hoc choices accumulate. F12-with-schema is enforceable.

The proposed sequence:

1. **This document is reviewed** by math-researcher (consumer, expert on libm specifics) and navigator (synthesis owner).
2. **Math-researcher provides per-recipe migration data** (which spec.tomls have `precision` parameter; which call sites use explicit `using(precision=...)`) — fills in §6.4.
3. **Team-lead/Tekgy ratifies F11+F12+schema** as one piece, OR rejects with reasoning.
4. **If ratified:** schema becomes a sweep-prerequisite; lint substrate is built (incremental — basic schema validation first, prose audit later); libm formalization sweep starts with path-classified recipes.
5. **If rejected:** schema is filed as substrate for future ratification; the F-series stack gets archived with the survey output; team continues with whatever the synthesis prioritizes.

---

## 10. Cross-references

- **F1-F10:** philosophical-survey findings (formalization process gaps, claim-shape, self-observation layer)
- **F11:** recognition/design distinction; F12 is its specific application to spec.toml defaults
- **F11.1:** sub-document granularity; this schema's claim-tagging at per-strategy granularity is the F11.1 pattern at the spec.toml level
- **F11.2:** graph-form forcing chains; future extension — `rationale` fields can carry `claim_id` pointers to other tambear recognition-claims, building the recognition-graph
- **F11.3:** cross-team coordination; antigen team's `#[descended_from]` propagation tooling is the structural neighbor for the lint walker
- **F12 (final):** defaults are claims with declared-exception amendment; this schema mechanizes the exception declaration
- **F12.1:** prose is part of the claim; this schema's §4 prose-audit rules operationalize
- **Math-researcher's [stances] proposal:** this schema is the fully-specified version
- **Math-researcher's claim_fidelity dual metric:** this schema's §5 formalizes
- **F4 audit:** this schema is the fix path

Reference exemplar: `R:\winrapids\crates\tambear\src\recipes\libm\asin.spec.toml` (pre-F12) and §8 above (post-F12 path-b migration).

---

## 11. Status

**Synthesis-ready** post math-researcher's review. §6.4 migration data filled in (30 of 32 recipes → path (b); 2 exceptions are exp/log already real). §11 open questions answered with reasoned positions (below).

**Reviewers:**
- math-researcher (consumer, libm domain expert; confirmed schema is ship-ready, supplied audit data, opined on three open questions — see below)
- navigator (synthesis owner; routes to team-lead with F-series stack)

**Open questions resolved post-draft** (math-researcher's positions, accepted as schema defaults):

- **Should `parameters.default.using` be REQUIRED to point at a `state = "real"` strategy?** YES. The cleaner option is the requirement. Per F12.1 (prose-as-claim), if the default fires an aliased strategy, the methods_template will render an awkward alias chain ("Precision strategy: compensated (aliased to strict; ≤ 2 ulps)"). Forcing default to point at a real strategy keeps the prose simple. Cost: the libm catalog needs to change `parameters.default.using = "compensated"` to `"strict"` for all 30 path-b recipes. UX-visible change, but documented in the migration. Most consumers don't pin the default; for those who reasoned about default = compensated, the change fails-loudly (default now strict; output ULPs the same since compensated was aliased anyway). Cleanest semantics. **The lint MUST enforce this.**

- **Should cross-recipe aliasing (`target = "other_recipe::strict"`) be permitted?** YES, with the lint walking the cross-recipe reference transitively. Use case: `versin_compensated` aliasing to `1 - cos_compensated(x)` (when cos eventually has a real compensated path in DD_libm). Without cross-recipe aliasing, every composed recipe has to ship its own DD path, which double-counts work. The lint's transitive-real check is mechanical (BFS through the aliased_to graph; assert all leaves are `state = "real"` strategies; detect cycles). **Add to §7's lint contract.**

- **Should the lint emit warnings for `real_fidelity < some_threshold`?** NO — keeps the metric clean. If we want signal, surface `real_fidelity` in the CI report (already in §5.3) without making it a gate. The danger of normalizing-low-fidelity is real; the safer rule is "report the number, gate only on declared_fidelity." Engineering judgment, not warning-noise.

These positions are now schema defaults. If team-lead/Tekgy want to override any of them at ratification time, that's their call — but the schema ships with these answers baked in.

---

## 12. Closing

The schema is the operationalization F12 was waiting for. Without it, F12 floats as principle and 48 recipes' violations bake into the locked-vocabulary library. With it, every recipe's claim space is mechanically auditable; aliasing is permitted iff disclosed; the migration table from pre-lock-debt to F12-compliant-substrate is enumerable.

The recognition was math-researcher's stress-test; the design is this schema; the deployment ordering is sweep-prerequisite. F11 closes the loop: recognition-with-operationalization-with-deployment-ordering is the full piece. Anything less floats.

Schema draft complete. Routing to math-researcher and navigator.
