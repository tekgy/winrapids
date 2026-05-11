---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: assumption document — methods_template + precision-description prose audit per F12.1
status: complete
audience: pathmaker (path-(b) migration input), aristotle (F12.1 verification), navigator (synthesis)
sources:
  - All 32 spec.toml files in R:\winrapids\crates\tambear\src\recipes\libm\
  - aristotle's F12 deconstruction at survey/20260508123003-aristotle/default-is-a-claim.md
  - aristotle's spec-toml-stances-schema.md
  - F4 audit (math-researcher SURVEY.md per-recipe stance audit, corrected version)
---

# F12.1 Methods_template + Precision-Description Prose Audit

> **Purpose.** Per aristotle's F12.1 ("every claim-shaped statement in the spec.toml is bound by F12, including methods_template prose ULP bounds, parameter description claims, output guarantees, and oracle-comparison pledges — not just the `default` field"), this document enumerates every claim site across the 32 libm spec.tomls. It serves as the audit-input for pathmaker's path-(b) declared-aliasing migration: per-recipe table of what each spec.toml currently asserts about strategy behavior, what the implementation actually does (per F4 audit), and which prose lines need updating.

---

## 1. Population — every spec.toml in scope

All 32 libm spec.tomls in `R:\winrapids\crates\tambear\src\recipes\libm\`:

| spec.toml | strategy domain | default | F4 status |
|---|---|---|---|
| acos | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| acosh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| acospi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| acot | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| acsc | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| asec | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| asin | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| asinh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| asinpi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| atan | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| atan2 | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| atanh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| atanpi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| cos | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict (in sin.rs) |
| cosh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| cospi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| cot | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| csc | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| **exp** | {strict, compensated, correctly_rounded} | compensated | **all three real** |
| gudermannian | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| haversin | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| inv_gudermannian | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sec | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sin | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sincos | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sincospi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sinh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| sinpi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| tan | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| tanh | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| tanpi | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |
| versin | {strict, compensated, correctly_rounded} | compensated | comp/cr alias to strict |

**Universal pattern**: 32/32 spec.tomls declare the triplet domain, default to `compensated`, and have a methods_template that references `{precision}`. Only `exp` (and by analogous Phase C2 commit, `log`) ship real triplets. The other 30 silently fire `_strict` when `precision = "compensated"` or `"correctly_rounded"`.

`log.spec.toml` does not appear in the list above because it does not exist as a separate file (`log.rs` is implemented but no `.spec.toml` companion was authored). That's a separate drift item — `log` should have a spec.toml. For the audit purposes, log is in the F4-real category but lacks the spec surface to violate.

## 2. Two prose-shape classes — terse vs verbose

The 32 spec.tomls split into two prose-shape classes by how they describe per-strategy ULP bounds.

### 2.1 Class A — Verbose claims (6 spec.tomls)

Six spec.tomls write detailed per-strategy bullets making explicit ULP commitments. These are the "loudest violations" under F12.1.

**asin.spec.toml**:
```
• strict: single-FMA Horner polynomial, standard sqrt. ≤ 2 ulps.
• compensated: compensated Horner + DD-corrected sqrt. ≤ 1 ulp.
• correctly_rounded: DD working precision throughout. ≤ 1 ulp on tested samples.
```
**Reality**: only `_strict` is real. Compensated and correctly_rounded both alias `_strict` (which is `≤ 2 ulps`). The "compensated: ≤ 1 ulp" and "correctly_rounded: ≤ 1 ulp" claims are not honored.

**atan.spec.toml**:
```
• strict: four-constant reconstruction + single-FMA Horner. ≤ 2 ulps.
• compensated: DD-precision constants + compensated Horner. ≤ 1 ulp.
• correctly_rounded: DD throughout. ≤ 1 ulp on tested samples.
```
**Reality**: same — alias to strict. "DD-precision constants + compensated Horner" prose is fictional.

**sin.spec.toml**:
```
• strict: single-FMA Horner path, ≤ 2 ulps worst case.
• compensated: compensated Horner for polynomial phase, ≤ 1–2 ulps.
• correctly_rounded: DD working precision throughout, ≤ 1 ulp on tested samples.
```
**Reality**: same.

**cos.spec.toml**:
```
• strict: single-FMA Horner path, ≤ 2 ulps worst case.
• compensated: compensated Horner, ≤ 1–2 ulps.
• correctly_rounded: DD working precision, ≤ 1 ulp on tested samples.
```
**Reality**: same.

**exp.spec.toml**:
```
• strict: fast single-FMA path, ≤ 4 ulps worst case.
• compensated: DD range reduction + compensated Horner, ≤ 2 ulps worst case.
• correctly_rounded: full DD working precision, ≤ 1 ulp worst case (essentially publication grade).
```
**Reality**: all three real per Phase C1. **F12-compliant if a `[stances]` block declares `state = "real"` for all three.** Only verbose-class spec.toml that doesn't violate F12 today (modulo the missing declaration). Sanity-check: exp's strict bound `≤ 4 ulps` is more conservative than asin/atan/sin/cos's `≤ 2 ulps` — this is honest about the cost-precision-tradeoff at the strict floor. The other verbose specs may have over-promised their strict tier as well.

**log** is implemented per Phase C2 but lacks a spec.toml. No prose to audit; F12 doesn't apply yet.

### 2.2 Class B — Terse shorthand (26 spec.tomls)

The other 26 spec.tomls write the precision parameter description as something close to:

```
description = "strict / compensated / correctly_rounded."
```

(Some variants: `"strict / compensated / correctly_rounded. Same semantics as tan."` for `cot`; `"strict / compensated / correctly_rounded. Applied to both kernels."` for `sincos`; etc.)

**No explicit ULP claim per strategy.** No "compensated: ≤ 1 ulp." prose. The terseness is itself ambiguous under F12: does "strict / compensated / correctly_rounded" assert that all three are *available*, or just that they're labels? The only thing the terseness implies is that the user can pick from the three.

**But the methods_template prose still references `{precision}`** for every Class-B spec.toml. So Class-B violations are subtler: the description prose doesn't lie explicitly, but the methods_template renders `Precision: compensated.` (when the default fires) into reports, while the implementation runs strict.

The Class-B issue is that **the implication of the triplet domain is inherited from the parent template**. Users reading Class-B spec.tomls assume the same per-strategy ULP commitments as Class-A — they're using the same parameter shape. Under F12.1 the inheritance is itself a claim: by reusing the triplet domain, Class-B spec.tomls inherit the contract that the verbose specs make explicit. That's the contract that's silently violated.

## 3. methods_template prose patterns

Three patterns in the methods_template prose. None is F12.1-clean.

### 3.1 Pattern α — explicit ULP claim within methods_template (3 spec.tomls)

**sin.spec.toml** + **cos.spec.toml** + **exp.spec.toml**:
```
{precision} precision strategy was selected, with a worst-case error
budget of {target_ulps} ULPs. ...
```
or:
```
Worst-case error budget: ≤ 2 ulps (strict/compensated)
or ≤ 1 ulp (correctly_rounded) against the IEEE 754 rounded result.
```

**Class-α prose makes the claim TWICE** — once in parameters.precision.description and once in methods_template. Both must be updated for F12.1 compliance. exp is the only Class-α spec for which the claim is honored (real triplet); sin and cos make published-paper-grade claims their implementations can't keep.

### 3.2 Pattern β — single-line `Precision: {precision}.` (24 spec.tomls)

The bulk of the catalog. Templates like:
- `Precision: {precision}. Output unit: {angle_unit}.`
- `Precision: {precision}. Out-of-range: {out_of_range}.`
- `methods_template = "Inverse hyperbolic cosine. Precision: {precision}. Out-of-range: {out_of_range}."`

The substitution `{precision}` renders as the actual strategy fired (e.g., "compensated") in published reports. **For the 24 Class-β recipes whose compensated path aliases strict, this prose lies in a published artifact**: the report says "Precision: compensated" while the implementation ran strict. Per F12.1, the claim site is the methods_template's `{precision}` substitution.

### 3.3 Pattern γ — domain-specific elaboration (5 spec.tomls)

asin / atan / sincos / cot / etc. extend β with one more sentence:
- asin: `Precision strategy: {precision}. Output unit: {angle_unit}.`
- atan: `Precision: {precision}. Output unit: {angle_unit}.`
- sincos: `Precision strategy: {precision}. Range reduction: {range_reduction}. Angle unit: {angle_unit}.`
- cot: `Range reduction: {range_reduction}. Precision strategy: {precision}. Angle unit: {angle_unit}.`

Same F12.1 issue as β. The extra sentence doesn't change the analysis.

## 4. Per-spec.toml fix recommendations

For each of the 30 violating spec.tomls, the fix path under F12 (per aristotle's schema) is path-(b): declared aliasing. Two prose updates per spec.toml:

### 4.1 parameters.precision.description update

**Class-A (5 verbose spec.tomls minus exp)**: rewrite the per-strategy bullets to surface the alias:

For asin (example template):
```toml
description = """
• strict: single-FMA Horner polynomial + DD-corrected sqrt reconstruction. ≤ 2 ulps.
• compensated: aliased to strict; ≤ 2 ulps. (Real DD path follows in DD_libm sweep.)
• correctly_rounded: aliased to strict; ≤ 2 ulps. (1-ulp guarantee follows in DD_libm sweep.)
"""
```

**Class-B (26 terse spec.tomls)**: add a parenthetical surfacing the alias:

```toml
description = """
strict / compensated / correctly_rounded.
(compensated and correctly_rounded currently aliased to strict; ≤ 2 ulps.
DD paths follow in DD_libm sweep.)
"""
```

Or, cleaner: **drop the triplet from the description text and let the [stances] block speak for itself**:

```toml
description = """
strict implementation; ≤ 2 ulps. compensated and correctly_rounded
strategies are declared aliases per [stances] block.
"""
```

The latter is cleaner and matches the F12-compliant shape better.

### 4.2 methods_template update — change the default

**The cleaner structural fix is to change `parameters.default.using = "compensated"` to `"strict"` for all 30 path-(b) recipes.** This satisfies aristotle's schema §4.2 open question ("default should point at a real strategy"). The methods_template's `{precision}` substitution then renders "Precision: strict." in reports — which matches the actual strategy fired. No alias-chain prose required.

UX-visibility: changing the default is observable to users who relied on the implicit "compensated is a real default." However:
- Per the F4 audit, NO downstream consumer in winrapids currently calls `using(precision="compensated")` explicitly.
- The aliased compensated produces strict output anyway. Changing default to strict produces identical output.
- Any consumer who DID rely on compensated semantically (rather than just letting the default fire) will get an honest error if `compensated` is later removed from the domain (path c) or remains aliased.

The cost is symbolic, not numerical. The benefit is F12.1 compliance without prose-rewriting beyond default-line change.

### 4.3 Add the `[stances]` block per aristotle's schema

Per the schema, every spec.toml gets a fully-declared `[stances]` block. Template for path-(b) recipes (the 30 aliasing ones):

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
The strict path uses {recipe-specific algorithm description with ULP budget}.
Compensated arithmetic on the same coefficients gains negligible precision
relative to the strict polynomial residual. No current downstream consumer
requires the compensated tier; future consumers can request the DD_libm
sweep to land a real implementation.
"""
fidelity_target = 1.0
sweep_blocking = "DD_libm"

[stances.override_transparency.strategy.correctly_rounded]
state = "stubbed_pending"
sweep = "DD_libm"
rationale = """
1-ulp guarantee requires DD-throughout polynomial evaluation. Not yet
implemented. Calling with this strategy currently aliases to strict;
future DD_libm sweep will replace alias with real implementation.
"""
```

The 30 path-(b) recipes share this template; only the rationale's recipe-specific algorithm description varies.

For the 2 path-(a) recipes (exp + log) that ship real triplets:

```toml
[stances]
implements = ["pure_math", "override_transparency"]
invoked_default = "override_transparency"

[stances.override_transparency.strategy.strict]
state = "real"

[stances.override_transparency.strategy.compensated]
state = "real"

[stances.override_transparency.strategy.correctly_rounded]
state = "real"
```

Plus the methods_template can keep referencing `{precision}` honestly because the rendered string matches the strategy fired.

## 5. Specific over-claims to flag

Beyond the structural F12.1 violation, three spec.tomls make claims that are likely incorrect even given a real implementation:

### 5.1 sin.spec.toml — "≤ 1 ulp on tested samples"

> "correctly_rounded: DD working precision throughout, ≤ 1 ulp on tested samples"

The phrase "on tested samples" is honest weasel-wording — the claim is only about samples that have been tested. **But there's no actual `_correctly_rounded` implementation today**, only an alias to strict. So the prose claims a property of an implementation that doesn't exist. Even if the DD path lands later, "on tested samples" needs a corpus-design claim per aristotle's "static corpus is itself a claim about input-region coverage" principle (see asin assumption doc §5).

### 5.2 atan.spec.toml — "≤ 1 ulp" (no qualifier)

> "compensated: DD-precision constants + compensated Horner. ≤ 1 ulp."

This is a stronger claim than sin's (no "on tested samples" qualifier). Today it's pure fiction. If the DD path eventually lands, "≤ 1 ulp" without a domain qualifier is worth verifying per-region — atan's four-interval reduction has different ULP characteristics in each sub-interval. This is the kind of claim corpus-design-as-claim catches.

### 5.3 exp.spec.toml — `≤ 4 ulps worst case` for strict

> "strict: fast single-FMA path, ≤ 4 ulps worst case."

This is the LOOSEST strict claim in the catalog. asin/atan/sin/cos/etc. claim `≤ 2 ulps` for strict. exp acknowledges that the fast strict path can't do better than 4 ULPs because the range-reduction is single-precision. **This is honest, but the inconsistency suggests the other recipes' "≤ 2 ulps strict" claims may be over-confident.** A separate audit pass should verify each strict path against its claimed budget on the adversarial corpus. (Out of scope for this F12.1 audit; flagged for future work.)

## 6. Migration scope summary

For pathmaker's path-(b) sweep:

| Action | Recipe count | Per-recipe effort |
|---|---:|---|
| Add `[stances]` block per template | 32 (incl. exp/log path-a declaration) | ~5-10 minutes (template + recipe-specific rationale) |
| Update `parameters.precision.description` to surface alias OR move ULP claims into `[stances]` rationale | 30 | ~3 minutes |
| Change `parameters.default.using = "compensated"` to `"strict"` | 30 | trivial line edit |
| Update `methods_template` for 6 verbose-class spec.tomls to remove explicit ULP claims about non-existent strategies | 5 (exp is fine) | ~5 minutes |
| Add `log.spec.toml` (currently missing, drift item) | 1 | ~30 minutes from-scratch |

**Total path-(b) work**: hours, not weeks. Single sweep deliverable. Mechanical with the template; per-recipe rationale customization is the only non-mechanical part.

## 7. CI lint validation surface (per aristotle's schema §7)

The lint must validate, for each spec.toml:

1. **§7.1 schema**: `[stances]` block exists; every strategy in `parameters.precision.domain.values` has a `[stances.override_transparency.strategy.<name>]` block.
2. **§7.2 implementation**: every `state = "real"` strategy has a real pub fn (non-trivially-different body from `_strict`).
3. **§7.3 prose audit**:
   - per-strategy bullets in `parameters.precision.description` match the strategy's actual capability.
   - aliased strategies' bullet text contains the alias-disclosure phrase.
   - methods_template's `{precision}` substitution doesn't render an aliased strategy name (i.e., `parameters.default.using` points at a `state = "real"` strategy).
4. **§7.4 metrics**: emit `(declared_fidelity, real_fidelity)` per recipe; aggregate across catalog.

This audit document is the input data for §7.3 — the lint can read these tables and verify the prose-update rules were applied per recipe.

## 8. Status

Audit complete. Path-(b) migration ready to execute when pathmaker pulls libm into `R:\tambear`. The sweep prerequisites are now:

1. ✅ Schema document (aristotle's spec-toml-stances-schema.md)
2. ✅ Migration data (math-researcher's F4 audit + this audit)
3. Pending team-lead/Tekgy ratification of F11+F12+schema as one piece (per aristotle's §9 sequence)

Once ratified, the libm formalization sweep can pull recipes into the locked-vocabulary library WITH F12-compliant `[stances]` blocks from commit-1, avoiding the silent-violation re-import.

## 9. Cross-references

- aristotle's `default-is-a-claim.md` — F12 deconstruction (recognition source)
- aristotle's `spec-toml-stances-schema.md` — F12 schema (operationalization)
- math-researcher's SURVEY.md F4-corrected stance audit (substrate scope)
- math-researcher's `assumption-docs/asin-rational-kernel.md` §5 (corpus-design-as-claim case study)
- math-researcher's `assumption-docs/ieee-754-2019-pi-scaled-exactness.md` §7 (recursive precision dispatch alternative for path-a recipes)
