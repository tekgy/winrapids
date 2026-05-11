---
campsite: tambear-sweep31-finish/naturalist
role: naturalist
date: 2026-05-08
sweep: 31 (cross-stance taxonomy — override-stance verification)
audience: math-researcher (back-reference to their diagnostic-stance-verification.md and pure-math-stance-skeleton.md); future-team-lead/Tekgy at sweep 35+ scoping
trigger: math-researcher's diagnostic-stance-verification.md §6 listed override as the next stance to verify (~2 hours, smallest of the three remaining). Their Observation B established the two-identifier pattern (signature step + enforcement step). This doc applies that pattern to the override stance.
purpose: apply 3-instance verification to the override stance. Test whether (a) step 0 (capture-and-divide) is structurally forced as the signature, (b) step 4 (record both + warning) is structurally forced as the enforcement, (c) single-path dispatch (just running the user's choice) is mis-classified as override-stance. If all three hold, the methodology is now triple-validated and the remaining two stances (workflow + discovery) can be picked up by anyone cycling back.
status: load-bearing for sweep 35+ test-infrastructure planning. Not load-bearing for sweep 31 mission. Filed in naturalist's campsite (created for this doc per the team-shape pattern of math-researcher having their own).
inputs:
  - math-researcher's diagnostic-stance-verification.md (esp. §0 + §4 Observations A/B/C)
  - naturalist's cross-stance-signature-steps.md and every-stance-has-two-identifiers.md
  - CLAUDE.md "Recipes by Behavioral Stance" §"Override-transparency stance" definition
  - winrapids/CLAUDE.md memory: feedback_expose_all_outputs.md (recipes-return-all-outputs)
---

# Override Stance — 3-Instance Verification

> **What this verifies**. Naturalist's draft (cross-stance-signature-steps.md) +
> math-researcher's two-identifier sharpening: every override-stance recipe has
> step 0 (capture-and-divide) as signature step + step 4 (record both results +
> warning when divergent) as enforcement step. Single-path dispatch where only
> the user's choice runs doesn't qualify as override-stance — that's just
> polymorphism with a `using()` knob.
>
> **Method**. Take three override-stance recipes from CLAUDE.md or its natural
> extension; decompose each against the two-identifier shape; verify both
> identifiers are non-trivial in each instance; verify single-path dispatch
> would mis-classify.

---

## 0. The override skeleton restated

Per cross-stance-signature-steps.md + every-stance-has-two-identifiers.md, the
override-transparency stance has a six-step skeleton:

| Step | What override-stance owns |
|---|---|
| **0. Capture-and-divide** | **Capture user's forced choice + compute tambear's recommendation in parallel; both paths set up before either runs** |
| 1. Special-value dispatch | Same shape as pure-math step 1 (NaN/Zero/Inf inputs propagate or short-circuit) |
| 2. (delegated) | Inner pure-math sub-recipes own their precision discipline |
| 3. (run both sub-recipes) | The chosen sub-recipe AND the recommended sub-recipe both execute |
| **4. Record both results + warning** | **Output struct holds `user_result`, `recommended_result`, `numerical_difference`, `warning_text` if results diverge** |
| 5. Canonicalize / return-all-outputs | Output struct fully populated with both sub-recipes' full outputs + the override audit trail |

**Signature claim**: step 0 is non-trivial in every override recipe.
**Enforcement claim**: step 4 records both results and a warning-when-divergent.
**Negation claim**: a recipe that only runs the user's choice (single path)
does NOT qualify as override-stance — it's just `pure_math.using(knob)`.

All three claims must hold for the verification to validate.

---

## 1. Test instance 1 — `pearson_r.using(method="kendall")` (forced rank-correlation when diagnostic would pick Pearson)

**Recipe**: a user calls `correlation(x, y).using(method="kendall")` over data
that's bivariate-normal-with-no-ties. Diagnostic would pick Pearson (both
continuous + both normal + no ties → Pearson). User forces Kendall.

**Skeleton fit**:

| Step | What the recipe does | Non-trivial? |
|---|---|---|
| **0. Capture-and-divide** | Capture user's `using(method="kendall")` → `user_choice = "kendall"`. Run diagnostic in parallel: variable types, normality, ties → `recommended = "pearson"`. Both choices captured. | **YES — load-bearing**. Without this step, the recipe doesn't know that kendall is the override and pearson is the recommendation; the warning at step 4 has nothing to compare. |
| 1. Special-value dispatch | NaN propagates; n < 3 → undefined | Yes |
| 3. Run BOTH sub-recipes | Both `pearson_r(x, y)` and `kendall_tau(x, y)` execute in parallel. Each is a pure-math recipe with own step 2. | **Both must run** — only running kendall would mean we have nothing to compare. |
| **4. Record both + warning** | Output: `OverrideResult { user_method: "kendall", user_value: 0.74, recommended_method: "pearson", recommended_value: 0.91, numerical_diff: 0.17, warning: "Forcing Kendall when Pearson would apply: Kendall's τ-b is the rank-based transform of Pearson; on bivariate-normal data with no ties, expect τ ≈ (2/π)·arcsin(r) so 0.74 vs 0.91 is consistent with rank-vs-value semantics rather than a bug. The diagnostic recommendation is pearson because all four diagnostic axes (continuity, normality, ties, outliers) favor it" }` | **YES — load-bearing**. The user must see both numbers and understand the difference. |
| 5. Canonicalize / return-all-outputs | Both sub-recipes' full output structs in addition to the override audit trail | Yes |

**Signature-step (0) verdict**: non-trivial. Without capture-and-divide, the recipe can't run both paths; the override claim is empty.

**Enforcement-step (4) verdict**: non-trivial. Without recording both + warning, the user has no transparency.

**Negation check**: if `pearson_r.using(method="kendall")` simply ran kendall and returned its result without computing pearson_r in parallel, that would be polymorphism — `kendall_tau` selected via dispatch — NOT override. The override claim "user sees what they chose AND what tambear would have chosen" is unfulfilled. Single-path dispatch fails the override classification.

✓ Override stance verified for `pearson_r.using(method="kendall")`.

---

## 2. Test instance 2 — `regression(data).using(estimator="ols")` (forced OLS when diagnostic would pick robust)

**Recipe**: a user calls `regression(data).using(estimator="ols")` over data
with heavy-tailed residuals. Diagnostic would pick robust regression (Huber
M-estimator) because Cook's-distance is high. User forces OLS.

**Skeleton fit**:

| Step | What the recipe does | Non-trivial? |
|---|---|---|
| **0. Capture-and-divide** | Capture user's `using(estimator="ols")` → `user_choice = "ols"`. Run diagnostic in parallel: residual heavy-tailedness (Cook's distance, leverage) → `recommended = "huber"`. Both captured. | **YES — load-bearing**. Different domain (regression rather than correlation), same shape. |
| 1. Special-value dispatch | Singular X^T X → error; under-determined system → error | Yes |
| 3. Run BOTH sub-recipes | Both `fit_ols(X, y)` and `fit_huber(X, y)` execute. Each is pure-math with own step 2 (compensated arithmetic for OLS sums; iterative reweighted least squares for Huber). | **Both must run.** |
| **4. Record both + warning** | Output: `OverrideRegressionResult { user_estimator: "ols", user_coefs: [β_ols], user_se: [se_ols], recommended_estimator: "huber", recommended_coefs: [β_huber], recommended_se: [se_huber], coefficient_diffs: [δ_β], warning: "Forcing OLS when Huber would apply: 12% of observations have Cook's > 4/n indicating high-leverage outliers. β coefficients differ by up to 23% between estimators (β_3: ols=0.41, huber=0.32). OLS standard errors are likely too narrow under heavy-tailed residuals (DW=1.2, BP p=0.003)." }` | **YES — load-bearing**. The user must see how much OLS differs from huber under their data. |
| 5. Canonicalize / return-all-outputs | Both regression result structs (coefs, SE, R², residuals, fit statistics) in addition to override audit trail | Yes |

**Signature-step (0) verdict**: non-trivial.

**Enforcement-step (4) verdict**: non-trivial. The numerical difference between estimators IS the transparency content — without it, the user can't see whether the override matters.

**Negation check**: if the recipe just ran OLS because the user asked for it, no warning, no parallel huber computation — that's polymorphism. The override claim is unfulfilled. Single-path dispatch fails.

✓ Override stance verified for `regression(data).using(estimator="ols")`.

---

## 3. Test instance 3 — `clustering(data).using(method="kmeans", k=5)` (forced kmeans+k=5 when diagnostic would pick HDBSCAN+auto-k)

**Recipe**: a user calls `clustering(data).using(method="kmeans", k=5)` over
data with non-spherical clusters and noise points. Diagnostic would pick
HDBSCAN with auto-detected k (looking at silhouette + density estimate).
User forces kmeans+k=5.

This is the most-different domain (clustering rather than correlation or
regression) and the only instance where the override has *two* parameters
(method + k), not one.

**Skeleton fit**:

| Step | What the recipe does | Non-trivial? |
|---|---|---|
| **0. Capture-and-divide** | Capture user's `using(method="kmeans", k=5)` → `user_choice = ("kmeans", k=5)`. Run diagnostic in parallel: cluster-shape estimate (gap statistic), noise estimate, density distribution → `recommended = ("hdbscan", auto_k=8)`. Both captured. | **YES — load-bearing**. Multi-parameter override doesn't change the shape — both choices are captured as tuples. |
| 1. Special-value dispatch | n < k → error; all-NaN → propagate; constant data → degenerate clustering | Yes |
| 3. Run BOTH sub-recipes | Both `kmeans(data, k=5)` and `hdbscan(data, auto_k=true)` execute. Each is pure-math (kmeans uses Lloyd's algorithm with compensated centroid arithmetic; HDBSCAN uses density-based exact-distance arithmetic). | **Both must run.** |
| **4. Record both + warning** | Output: `OverrideClusteringResult { user_method: "kmeans", user_k: 5, user_labels: [...], user_silhouette: 0.31, user_inertia: 87.4, recommended_method: "hdbscan", recommended_k: 8, recommended_labels: [...], recommended_silhouette: 0.62, recommended_n_noise: 12, agreement_rand_index: 0.41, warning: "Forcing kmeans+k=5 when HDBSCAN+auto_k=8 would apply: silhouette differs (0.31 vs 0.62 — kmeans clusters are less well-separated). Rand index 0.41 indicates moderate disagreement; HDBSCAN identified 12 noise points that kmeans assigned to clusters. The diagnostic chose HDBSCAN because the gap-statistic shape suggests non-spherical clusters and density variations." }` | **YES — load-bearing**. The clustering domain has TWO axes of disagreement (number of clusters + assignment of points), and both are exposed. The Rand-index agreement metric is the structural-fingerprint version of "numerical_diff" — same shape, scaled to clustering. |
| 5. Canonicalize / return-all-outputs | Both clustering result structs (labels, silhouette, inertia/density, parameters) in addition to override audit trail | Yes |

**Signature-step (0) verdict**: non-trivial.

**Enforcement-step (4) verdict**: non-trivial. The Rand-index agreement is a richer enforcement than scalar-numerical-difference because clustering disagreements are richer than scalar disagreements. The pattern generalizes — *the enforcement step adapts to the domain's natural form of disagreement*.

**Negation check**: if the recipe ran kmeans+k=5 only, no parallel HDBSCAN, no Rand-index — polymorphism. The override claim is unfulfilled.

✓ Override stance verified for `clustering(data).using(method="kmeans", k=5)`.

---

## 4. Cross-instance summary — what's structurally forced

Three test instances across three different domains (correlation, regression,
clustering). The same skeleton holds. Three observations make this stronger
than coincidence:

**Observation D — Step 0 (capture-and-divide) is ALWAYS multi-path setup before any computation runs.**

In every instance, step 0 captured both the user's forced choice and tambear's
recommendation BEFORE either path ran. This is what distinguishes override from
diagnostic: diagnostic runs the diagnostic FIRST, then picks one path; override
runs BOTH paths and shows both. The temporal ordering matters — capture-and-
divide is a *parallel-execution-setup step*, not a sequential decision step.

**Single-path-with-knob (e.g., user calls `kmeans(data, k=5)` directly without
the `using()` interface) is just a parameterized pure-math recipe**. It has no
override-stance content. This sharpens math-researcher's polymorphism-vs-
diagnosis boundary at a different layer: **single-path dispatch is polymorphism;
parallel-dual-execution is override**. The two are structurally distinct.

**Observation E — Step 4 (record both + warning) is ALWAYS rich enough to expose the domain's natural form of disagreement.**

In each instance, the enforcement step recorded:
- Correlation: two scalars + numerical_diff + warning (scalar disagreement)
- Regression: two coefficient vectors + per-coefficient diffs + warning (vector disagreement)
- Clustering: two label assignments + Rand-index + per-cluster comparisons + warning (structural disagreement)

The shape is always "user_X / recommended_X / structured_diff / warning" but
the structured_diff is whatever exposes disagreement at the right granularity
for the domain. A scalar numerical_diff would be insufficient for clustering
because two clusterings with the same silhouette can disagree wildly in
labels — the disagreement is structural, not scalar.

**This is publication-grade rigor at the override layer**: the user must see
disagreement in the form most useful for THEIR domain, not in some uniform-but-
inadequate form. The enforcement step adapts; it's not a fixed schema.

**Observation F — The override stance is the recognition-collaboration form of method dispatch.**

This is the philosophical layer. Math-researcher's Observation A established
that diagnostic-stance is "convention-to-declaration of polymorphism." Override-
stance is one step further — it's "convention-to-declaration of diagnostic
itself." When a user disagrees with tambear's diagnostic, the failure mode of
implicit-trust is: silently accept user's choice and hide tambear's better
recommendation. The convention form is "the user is always right; just run
their choice." The declaration form is "show the user both choices, let them
disagree with eyes open."

This is exactly the recognition-collaboration pattern from the earlier garden
entries — *neither tambear nor the user silently wins*. Both contributions
are preserved; the disagreement is exposed; the user makes the final call
with full information.

The override stance is the F-stance taxonomy's instance of *substrate-honest
dual-execution*. The discipline is the same as the substrate-over-memory
discipline at every other scope: don't silently override, don't silently
accept; expose both states and let the user choose with eyes open.

---

## 5. The methodology is now triple-validated

Three stances verified (pure-math + diagnostic + override) across nine total
test instances. Two stances remaining (workflow + discovery). The methodology
holds.

For sweep 35+, the trait family from naturalist's draft can now be implemented
with high confidence:

```rust
trait OverrideInvariants {
    /// Step 0 signature: both user-choice and recommendation are captured
    /// before any computation runs. The signature MUST be parallel-setup,
    /// not sequential decide-then-run.
    fn signature_test_capture_and_divide() -> Result;

    /// Step 4 enforcement: output struct contains both results, the
    /// domain-appropriate structural-difference metric, and the
    /// warning text. Must adapt to the domain's natural form of
    /// disagreement (scalar diff for scalar, vector diff for vector,
    /// structural metric for structured outputs).
    fn enforcement_test_dual_results_with_warning() -> Result;

    /// Negation: single-path dispatch (only user's choice runs) MUST
    /// fail this trait. Verifies the boundary between override and
    /// polymorphism.
    fn negation_test_single_path_fails_classification() -> Result;
}
```

The negation test is a new structural beat: each stance's trait should include
a negation check that distinguishes the stance from the next-simpler form
(override-vs-polymorphism, diagnostic-vs-polymorphism, pure-math-vs-empty,
workflow-vs-sequential-call-without-routing, discovery-vs-diagnostic). The
negation tests are F13 antibodies for stance mis-classification.

---

## 6. What remains for whoever cycles back — workflow + discovery

Per math-researcher's §6, two stances remain:

**Workflow stance** — three candidates from CLAUDE.md:
- `two_group_comparison`
- `regression_diagnostics`
- `survival_analysis`

For each, verify: step k.5 (between-step routing) is non-trivial between every pair of sub-recipes; the recipe is a *composition* of sub-recipes from other stances.

**Discovery stance** — three candidates from project memory `project_discover_superposition`:
- `discover_correlation`
- `discover_clustering`
- `discover_distance_metric`

For each, verify: step 2' (fan-out) runs every plausible method in parallel; step 4' (agreement-fingerprint) computes a non-trivial structural summary.

Estimated budget: ~2 hours per stance, 4 hours total to fully verify the cross-stance taxonomy.

After full verification, the trait family is buildable, the per-stance gauntlet is automatic, and the F-stance taxonomy becomes a small queryable structure for sweep 35+ scoping.

---

## 7. Provenance + what closed today

- Authored 2026-05-08 by naturalist in team `tambear-sweep31-finish` after math-researcher's diagnostic-stance-verification.md established the two-identifier pattern (Observation B) and named override as the next stance to verify.
- Three test instances drawn from natural extensions of CLAUDE.md's recipe-catalog: `pearson_r.using(method="kendall")` (correlation domain), `regression(data).using(estimator="ols")` (regression domain), `clustering(data).using(method="kmeans", k=5)` (clustering domain — multi-parameter override).
- Cross-checked: every step 0 is parallel-setup-before-computation (Observation D); every step 4 records domain-appropriate structural disagreement (Observation E); single-path dispatch fails classification (negation check).
- Conjecture holds for override stance. Methodology now triple-validated (pure-math + diagnostic + override).
- Not load-bearing for sweep 31. Filed for sweep 35+ scoping.

**What closed in this thread today (math-researcher + naturalist together)**:
1. Pure-math stance skeleton verified (math-researcher's pure-math-stance-skeleton.md)
2. Cross-stance taxonomy drafted (naturalist's cross-stance-signature-steps.md)
3. Diagnostic stance verified (math-researcher's diagnostic-stance-verification.md)
4. Two-identifier pattern named (naturalist's every-stance-has-two-identifiers.md)
5. **Override stance verified (this doc)**
6. F11.2 graph-form forcing for trait-promotion-now (math-researcher + naturalist via navigator)
7. Real silent bug found in newton_reciprocal fallback (math-researcher)

Seven findings from one naturalist's first-day BZ-skeleton noticing. The compounding kept compounding.

Two stances remaining (workflow + discovery). Whoever picks them up has a triple-validated methodology to apply.

— naturalist
