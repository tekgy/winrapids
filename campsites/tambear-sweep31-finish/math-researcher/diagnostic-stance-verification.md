---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (cross-stance taxonomy — diagnostic-stance verification)
audience: naturalist (back-reference to their cross-stance-signature-steps.md draft); future-team-lead/Tekgy at sweep 35+ scoping
trigger: naturalist's draft cross-stance taxonomy proposed signature-step-per-stance: pure-math = step 2 (precision), diagnostic = step 2.5 (method-selection), override = step 0 (capture-and-divide), workflow = step k.5 (between-step routing), discovery = steps 2'/4' (fan-out + agreement-fingerprint). Pure-math case verified with 3 instances in pure-math-stance-skeleton.md. The other four stances need their own 3-instance verification.
purpose: apply the same 3-instance verification to the diagnostic stance. Test whether step 2.5 is structurally forced and whether non-diagnostic stances delegate to it. If verification succeeds, the methodology is validated (workflow + override + discovery can use the same shape). If it fails, the boundary of the cross-stance hypothesis is found.
status: load-bearing for sweep 35+ test-infrastructure planning. Not load-bearing for sweep 31 mission. Filed while the connection is fresh per substrate-over-memory discipline.
inputs:
  - naturalist's `~/.claude/garden/2026-05-08-cross-stance-signature-steps.md` §"Verification candidates" + signature-step claim for diagnostic
  - pure-math-stance-skeleton.md (this campsite) — methodology + verification structure
  - CLAUDE.md "Recipes by Behavioral Stance" §"Diagnostic / auto-selection stance" definition + examples
  - CLAUDE.md "Recipes Are Compositions" example table: kendall_tau as composition; diagnostic recipes appear later in the workflow examples
---

# Diagnostic Stance — 3-Instance Verification

> **What this verifies**. Naturalist's draft claim: every diagnostic-stance recipe has step 2.5 (sub-recipe selection based on diagnostics) as a structurally-forced beat in its implementation. The signature-step is the implementation-locus of the stance's ownership claim ("tambear knows which sub-recipe to pick for this data").
>
> **Method**. Take three diagnostic recipes from CLAUDE.md or its natural extension; decompose each against naturalist's draft skeleton (step 0 setup, step 1 special-value, step 2.5 diagnose-and-select-sub-recipe, step 3 run sub-recipe, step 4 record decision-and-rationale, step 5 return-all-outputs); ask if step 2.5 is non-trivial in each, and if step 2 (precision discipline) is *delegated* to the inner pure-math sub-recipe rather than owned at the diagnostic layer. If both, the stance-signature claim holds.

---

## 0. The diagnostic skeleton restated

Per naturalist's draft, the diagnostic stance has six steps (one more than pure-math because step 2.5 is inserted before step 3):

| Step | What diagnostic-stance owns at this step |
|---|---|
| 1. Special-value dispatch | Same shape as pure-math step 1 (NaN/Zero/Inf inputs propagate or short-circuit) |
| **2.5. Diagnose-and-select** | **Run diagnostic checks; pick the sub-recipe; record the choice + rationale** |
| 3. Run the selected sub-recipe | The sub-recipe is itself a pure-math recipe with its own five steps internally (including its own step 2 precision-extension) |
| 4. Record decision + rationale in output struct | The output carries `recommended` (sub-recipe-result), `user_override` (None for diagnostic, populated for override-stance), `diagnostics` (the values that drove the choice) |
| 5. Canonicalize / return-all-outputs | Same shape as pure-math step 5; the output struct is fully populated |

**The signature claim**: step 2.5 is non-trivial in every diagnostic recipe. Without step 2.5, the recipe is just a hard-coded call to one sub-recipe and the diagnostic claim is empty.

**The delegation claim**: step 2 (precision discipline) lives *inside* the chosen sub-recipe (which is pure-math), not at the diagnostic layer. The diagnostic layer is *precision-discipline-agnostic* — it routes; it doesn't compute.

Both claims must hold for each instance.

---

## 1. Test instance 1 — `correlation(x, y)` (CLAUDE.md's canonical example)

**Recipe**: `correlation(x: &[f64], y: &[f64]) -> CorrelationResult`. The auto-selecting correlation: tambear inspects the data and picks Pearson, Spearman, Kendall, point-biserial, polychoric, tetrachoric, or distance correlation based on (a) variable types (continuous, ordinal, dichotomous, mixed), (b) normality, (c) outlier influence, (d) ties.

**Skeleton fit**:

| Step | What correlation does | Non-trivial? |
|---|---|---|
| 1. Special-value dispatch | NaN propagates; `n < 3` → NaN; mismatched lengths → error | Yes — same as pearson_r's step 1 |
| **2.5. Diagnose-and-select** | **(a) Classify variable types**: continuous/ordinal/dichotomous via heuristic (cardinality, value-set sparsity). **(b) If both continuous**: run Shapiro-Wilk normality on both; check outlier influence via Cook's distance or a simpler robust-z test; check ties (Pearson assumes no ties; Kendall handles ties via τ-b). **(c) Decision tree**: both continuous + both normal + few outliers + no ties → Pearson; both continuous + non-normal OR many outliers → Spearman; either ordinal OR many ties → Kendall τ-b; one dichotomous + one continuous → point-biserial; both ordinal-with-known-categories + assumed-underlying-continuous → polychoric; etc. **(d) Record the decision**: which sub-recipe + which diagnostic values triggered which branch | **YES — load-bearing**. The whole point of `correlation()` vs `pearson_r()` is that this step exists. Without it, the call is just `pearson_r(x, y)` with extra ceremony. |
| 3. Run the selected sub-recipe | `pearson_r(x, y)` OR `spearman_r(x, y)` OR `kendall_tau(x, y)` OR ... — each is a pure-math recipe with its own internal step 2 (precision discipline). The diagnostic layer DELEGATES to the sub-recipe and returns its output unchanged | The math runs in the pure-math sub-recipe; the diagnostic layer doesn't touch precision. |
| 4. Record decision + rationale | `CorrelationResult { recommended_method: "spearman", recommended_value: 0.71, diagnostics: { shapiro_x_p: 0.02, shapiro_y_p: 0.18, ties_x: 3, ties_y: 0 }, decision_path: ["both_continuous", "x_non_normal", "spearman" ] }` | Yes — this is the transparency claim. |
| 5. Canonicalize / return-all-outputs | All sub-recipe outputs + diagnostic struct + decision path returned together | Yes — recipes-return-all-outputs. |

**Signature-step (2.5) verdict**: non-trivial. The diagnostic decision tree is the entire reason `correlation()` exists as a separate recipe from `pearson_r()`.

**Precision-delegation verdict**: yes. Step 2 (precision discipline) is owned by `pearson_r` / `spearman_r` / `kendall_tau`, NOT by `correlation`. The diagnostic layer is precision-agnostic.

✓ Conjecture holds for `correlation`.

---

## 2. Test instance 2 — `regression_diagnostics(model, data)` (CLAUDE.md "Workflow stance" examples)

**Recipe**: `regression_diagnostics(model: &OlsModel, data: &Data) -> RegressionDiagnosticsResult`. Note: CLAUDE.md lists this under "workflow recipes" alongside `two_group_comparison` and `survival_analysis`. But the diagnostic *step* of regression_diagnostics is itself a diagnostic-stance recipe inside the workflow — workflow contains diagnostic contains pure-math, three nested layers.

For verification, focus on the diagnostic inner-step: the recipe that *picks which corrective action to recommend* based on diagnostic outputs (heteroskedasticity → recommend robust SE; multicollinearity → recommend ridge; outliers → recommend robust regression; non-normal residuals → recommend bootstrap CIs).

Call this inner recipe `recommend_correction(diagnostic_summary) -> CorrectionRecommendation`.

**Skeleton fit**:

| Step | What recommend_correction does | Non-trivial? |
|---|---|---|
| 1. Special-value dispatch | If diagnostic_summary is empty (no observations) → return None recommendation; if model is degenerate → return "model is unfit, no correction valid" | Yes |
| **2.5. Diagnose-and-select** | **(a) Examine each diagnostic value**: BP test p-value, VIF max, Cook's-distance max, Shapiro-Wilk p-value on residuals, Durbin-Watson stat. **(b) Threshold-driven decision**: BP p < 0.05 → "use HC3 robust SE"; VIF > 10 → "ridge regularization with λ chosen via CV"; Cook's > 4/n → "robust regression via M-estimator"; SW p < 0.05 → "bootstrap CIs". Multiple may apply; aggregate per a fixed precedence rule. **(c) Record which thresholds fired and what action was selected**. | **YES — load-bearing**. The whole point of recommend_correction vs hard-coding "always use HC3" is that this step exists. |
| 3. Run the selected sub-recipe | The sub-recipe is the *correction itself* — `compute_hc3_se(model)`, `fit_ridge(model, lambda)`, `fit_huber(model)`, etc. — each a pure-math recipe with own step 2. | Yes — math is in sub-recipe. |
| 4. Record decision + rationale | `CorrectionRecommendation { selected_action: "hc3_se", reason: "BP p=0.012 indicates heteroskedasticity", alternatives_considered: [...] }` | Yes |
| 5. Canonicalize / return-all-outputs | Recommendation struct + computed correction + diagnostic values | Yes |

**Signature-step (2.5) verdict**: non-trivial. Threshold-driven multi-axis dispatch among 4-5 corrective actions IS the diagnostic content of the recipe.

**Precision-delegation verdict**: yes. The corrective sub-recipes own their own precision discipline. recommend_correction is precision-agnostic.

✓ Conjecture holds for `recommend_correction`.

---

## 3. Test instance 3 — `two_group_test(group1, group2)` (the diagnostic step inside two_group_comparison)

**Recipe**: `two_group_test(g1: &[f64], g2: &[f64]) -> TwoGroupTestResult`. Auto-picks Welch's t-test, Student's t-test, or Mann-Whitney U based on (a) sample sizes, (b) normality, (c) variance equality.

This is naturalist's third example shape — different domain (hypothesis testing) from correlation (correlation coefficient) and regression (model selection). Tests the conjecture across the breadth of diagnostic-recipes, not just a single domain.

**Skeleton fit**:

| Step | What two_group_test does | Non-trivial? |
|---|---|---|
| 1. Special-value dispatch | If either group has `n < 2` → undefined; if all-NaN → propagate; if g1 == g2 (degenerate identical) → t = 0, p = 1, no real test | Yes |
| **2.5. Diagnose-and-select** | **(a) Normality check**: Shapiro-Wilk on each group (or D'Agostino's K² for n > 50). **(b) Variance equality check**: Levene's test (or Bartlett's if both pass normality). **(c) Decision tree**: both normal + equal variance → Student's t; both normal + unequal variance → Welch's t; either non-normal → Mann-Whitney U; rank-based with small-n → exact Mann-Whitney via permutation. **(d) Record which checks fired and which test was selected**. | **YES — load-bearing**. The decision tree is exactly what makes `two_group_test()` more useful than calling Welch's directly. |
| 3. Run the selected sub-recipe | `welch_t(g1, g2)` OR `student_t(g1, g2)` OR `mann_whitney_u(g1, g2)` — each a pure-math recipe with own step 2 (compensated arithmetic for sample stats; exact-permutation arithmetic for small-n MWU). | Yes — math in sub-recipes. |
| 4. Record decision + rationale | `TwoGroupTestResult { test_name: "welch", t_stat: 2.34, df: 47.3, p_value: 0.024, diagnostics: { shapiro_g1_p: 0.31, shapiro_g2_p: 0.18, levene_p: 0.04 }, decision_path: ["both_normal", "unequal_variance", "welch"] }` | Yes |
| 5. Canonicalize / return-all-outputs | All test stats + diagnostics + effect size (Cohen's d or rank-biserial) + CI + decision path | Yes |

**Signature-step (2.5) verdict**: non-trivial. Three-way dispatch among Student/Welch/MWU based on Shapiro + Levene IS the diagnostic content.

**Precision-delegation verdict**: yes. The Welch/Student/MWU sub-recipes each own their own precision discipline (compensated stats for the parametric tests; exact rank arithmetic for MWU). two_group_test is precision-agnostic.

✓ Conjecture holds for `two_group_test`.

---

## 4. Cross-instance summary — what's structurally forced

After three test instances across three different domains (correlation coefficient, regression diagnostics, hypothesis testing), the same skeleton holds. Three observations make this stronger than coincidence:

**Observation A — Step 2.5's content is ALWAYS a multi-input decision tree.**

In every instance, step 2.5 examined multiple diagnostic outputs (variable type + normality + outliers + ties; BP + VIF + Cook + SW + DW; Shapiro + Levene; etc.) and ran a decision tree to pick a sub-recipe. **Single-axis dispatch (e.g., "if x is numeric, use Pearson") doesn't qualify as diagnostic-stance** — that's just polymorphism. Diagnostic-stance demands real multi-axis decision-making. Naturalist's "tambear knows which sub-recipe to pick" recognition-claim is honored only when the picking is non-trivial.

This sharpens the stance definition: **diagnostic-stance recipes have multi-axis decision trees at step 2.5**.

**Observation B — Step 4 (record decision) is ALWAYS load-bearing.**

In every instance, the output included not just the sub-recipe's result but also `(selected_method, diagnostic_values, decision_path)`. Without this, the user can't audit why tambear made the choice it did. The transparency claim of the stance is enforced at step 4.

**Step 4 is the second load-bearing identifier of the diagnostic stance**, alongside step 2.5. Same shape as pure-math (where step 2 + step 5 are the two identifiers).

**Observation C — Diagnostic recipes are precision-discipline-AGNOSTIC, not precision-discipline-OWNING.**

In every instance, the precision discipline lives in the inner pure-math sub-recipe (Pearson's compensated arithmetic, ridge's QR factorization, MWU's exact permutation arithmetic). The diagnostic layer doesn't add or alter precision discipline; it just routes.

**This is the cross-stance distinction made sharp**: pure-math owns precision; diagnostic delegates precision. The two stances are distinguished by *who owns step 2*. Pure-math owns it directly; diagnostic owns step 2.5 instead and delegates step 2 to its inner sub-recipe.

---

## 5. The methodology is validated

Three instances confirm the diagnostic stance has its own structurally-forced signature step (2.5) plus a second load-bearing identifier (step 4 = decision-recording). The verification methodology naturalist asked for in §7 of pure-math-stance-skeleton.md and naturalist applied to the cross-stance taxonomy in their draft works.

**This validates the broader hypothesis**: each of the five F-stances has its own signature-step. Workflow's step k.5, override's step 0, discovery's steps 2'/4' — these can be verified by the same 3-instance methodology. Naturalist's draft of those signatures stands until tested.

For sweep 35+, this means: **the test-infrastructure trait family naturalist sketched in §"What this gives us if it survives verification" is buildable**. Each stance gets its own InvariantTrait; recipes declare their stance; the gauntlet for that stance fires automatically.

The classification rule (§"The classification rule" in naturalist's draft):
1. Has non-trivial step 2 → pure-math
2. Has non-trivial multi-axis decision tree at step 2.5 → diagnostic
3. (workflow / override / discovery — to be verified)

The first two rungs are now verified. Three more remain.

---

## 6. What remains for naturalist (or future-someone) — the three other stances

**Override stance** verification candidates (3 needed):
1. `pearson_r.using(method="kendall")` — user forces Kendall when diagnostic would pick Pearson.
2. `regression(data).using(estimator="ols")` — user forces OLS when diagnostic would pick robust.
3. `clustering(data).using(method="kmeans", k=5)` — user forces kmeans-with-k=5 when diagnostic would auto-pick HDBSCAN with auto-k.

For each, verify: step 0 (capture-and-divide) runs both paths; step 4 records `recommended` + `user_override` + warning when they differ; step 5 returns both results.

**Workflow stance** verification candidates (3 named in CLAUDE.md):
1. `two_group_comparison` — full workflow including diagnostic, t-test, effect size, CI, robustness checks.
2. `regression_diagnostics` — full workflow including model fit, residual analysis, recommend_correction, bootstrap CIs.
3. `survival_analysis` — full workflow including KM estimator, log-rank test, Cox PH, proportional-hazards check.

For each, verify: step k.5 (between-step routing) is non-trivial between every pair of sub-recipes; the recipe is a *composition* of sub-recipes from other stances (mostly pure-math + diagnostic).

**Discovery stance** verification candidates (3 from project memory `project_discover_superposition`):
1. `discover_correlation` — runs every correlation method, reports Pearson + Spearman + Kendall + ... plus structural-fingerprint of agreement.
2. `discover_clustering` — runs every clustering method (K-means, HDBSCAN, GMM, spectral), reports each clustering plus inter-method agreement (Rand index).
3. `discover_distance_metric` — runs every distance metric (Euclidean, Manhattan, Mahalanobis, Wasserstein), reports each + structural-fingerprint.

For each, verify: step 2' (fan-out) runs every plausible method in parallel; step 4' (agreement-fingerprint) computes a non-trivial structural summary of where methods agree/disagree.

**Recommended verification budget**: ~2 hours per stance for whoever picks this up. Total 6 hours to fully verify the cross-stance taxonomy. After that, the F-stance trait family from naturalist's §"trait family" sketch can be implemented at sweep 35+.

---

## 7. Provenance + what closed today

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish` after naturalist's draft cross-stance taxonomy invited the same 3-instance verification I applied to pure-math.
- Three test instances drawn from CLAUDE.md's recipe-catalog + natural extensions: `correlation`, `recommend_correction` (the diagnostic step inside `regression_diagnostics`), `two_group_test` (the diagnostic step inside `two_group_comparison`). Three different domains; same skeleton; same forced signature-step.
- Cross-checked: every step 2.5 is multi-axis decision tree (Observation A); every step 4 is load-bearing decision-recording (Observation B); every diagnostic recipe is precision-discipline-AGNOSTIC and delegates precision to inner pure-math sub-recipes (Observation C).
- Conjecture holds for diagnostic stance. Methodology validated (now twice — pure-math + diagnostic). Three stances remain (override, workflow, discovery).
- Not load-bearing for sweep 31. Filed for sweep 35+ scoping per substrate-over-memory: future-team-lead at sweep 35+ won't have to re-derive the diagnostic-stance verification, just pick up where this leaves off and verify the other three.

**What closed today (the compounding naturalist named)**:
1. Pure-math stance skeleton verified (math-researcher's pure-math-stance-skeleton.md).
2. Cross-stance taxonomy drafted (naturalist's cross-stance-signature-steps.md).
3. Diagnostic stance verified (this doc).
4. F11.2 graph-form forcing for trait-promotion-now (math-researcher + naturalist via navigator).
5. Real silent bug found in newton_reciprocal fallback (math-researcher's BZ-text-match review).

Five findings closed today. The compounding kept compounding.

— math-researcher
