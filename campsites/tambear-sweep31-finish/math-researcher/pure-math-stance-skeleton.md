---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (cross-domain rhyme — pure-math stance generalization)
audience: future-team-lead/Tekgy at sweep 35+ scoping; future math-researcher when pearson_r or efa or any pure-math recipe gets test-infrastructure shaped; naturalist (back-reference)
trigger: naturalist's bz-skeleton-converges.md §"A different angle — recipes by stance" claimed pearson_r has the same five-step shape as BigFloat add/mul/div/sqrt. This doc tests that claim with three more pure-math recipes drawn from CLAUDE.md's recipe-catalog tour.
status: not load-bearing for this session (sweep 31 mission is the multi-limb arith unstub). Loading-bearing for sweep 35+ planning. Filed now while the connection is fresh; future-context-self won't have to re-derive.
inputs:
  - naturalist's `~/.claude/garden/2026-05-08-bz-skeleton-converges.md` §"A different angle — recipes by stance"
  - op-invariant-catalog.md (this campsite) — six invariant families F-A through F-H
  - CLAUDE.md "Recipes Are Compositions" + "Recipes by Behavioral Stance" sections
  - winrapids/CLAUDE.md "What This System IS" — the dimensional-ladder framing
purpose: test the claim that the BZ five-step skeleton extends across ALL pure-math-stance recipes (not just arithmetic primitives). If the claim holds, the OpInvariants trait shape generalizes to a RecipeInvariants pattern usable for every pure-math recipe in the F-stance taxonomy.
---

# The BZ Skeleton Generalizes — Pure-Math Recipes Are All Five-Step

> **The conjecture** (naturalist's): the five-step skeleton math-researcher named for
> BZ Algorithm 3.1/3.3/3.5/3.10 isn't an arithmetic-specific shape. It's the
> *pure-math-stance's structural shape*. Every pure-math recipe — from the
> arithmetic primitives up through pearson_r, kendall_tau, kaplan_meier, garch_fit —
> has exactly the same five steps; only step 3 differs.
>
> **Why this matters**. If the conjecture holds, the `OpInvariants` trait math-researcher
> sketched in op-invariant-catalog.md §7 isn't just for arithmetic. It's the test-skeleton
> for the entire pure-math catalog. Sweep 35+ (transcendentals, then statistical recipes,
> then ML primitives) inherits the same gauntlet shape. The 2000-cell test surface for
> add/sub/mul/div/sqrt becomes the *seed* for tens of thousands of cells across the
> catalog; the trait makes them all fall out of the same skeleton.
>
> **What this doc does**. Tests the conjecture against three pure-math recipes drawn
> from CLAUDE.md's table. If the five-step skeleton fits each, that's three independent
> instances of the pattern (graph-form forcing per F11.2, not chain). If it doesn't
> fit one, that's a finding worth investigating — either the pattern is BigFloat-specific,
> or the misfit-recipe is mis-classified.

---

## 1. The skeleton, restated for cross-domain use

The five steps as naturalist named them for the BZ algorithms:

1. **Special-value dispatch** — handle NaN/Inf/Zero/sentinel inputs by classification before touching the math.
2. **Precision extension** — bring inputs into a working representation that absorbs the rounding error of the core math.
3. **Core math** — the operation-specific computation. THIS is what differs across recipes.
4. **Final round** — bring the working result back to the requested output precision per RoundingMode (or its analog in the recipe's domain).
5. **Canonicalize** — enforce post-condition invariants on the result so it satisfies the next consumer's preconditions.

For the conjecture to extend, each pure-math recipe should have all five steps as *non-trivial* phases of its implementation, not just step 3 (the math). If a recipe collapses steps 1, 2, 4, 5 to no-ops, the skeleton is BigFloat-specific. If they're all non-trivial, the skeleton is structural.

---

## 2. Test instance 1 — `pearson_r` (correlation coefficient)

Naturalist's example. Restated more carefully:

**Recipe**: pearson_r(x: &[f64], y: &[f64]) → f64. Returns Pearson's product-moment correlation coefficient.

Mathematical definition: `r = cov(x, y) / (sx · sy)` where `cov` is the sample covariance and `sx`, `sy` are sample standard deviations.

**Five-step decomposition**:

| Step | What pearson_r does | Why this step is non-trivial |
|---|---|---|
| 1. Special-value dispatch | If `x.len() != y.len()` → error (or NaN per the recipe's policy). If any input is NaN → result is NaN (per F-stance's NaN-propagation default). If `n < 2` → undefined (recipe's choice: NaN or panic). If all-x-equal or all-y-equal (zero variance) → undefined → NaN. | Several distinct boundary cases need explicit handling; not just "compute the formula." |
| 2. Precision extension | Compute `cov(x, y)`, `var(x)`, `var(y)` using **compensated arithmetic** (Welford-style streaming, or two-pass with Kahan summation) rather than naive sum-then-divide. The naive form has catastrophic cancellation when `mean(x)` is large relative to `std(x)`. | The "precision extension" of arithmetic's `p+50 guard bits` becomes "use compensated streaming statistics" in the statistical-recipe domain. Same SHAPE — bring the computation into a higher-precision intermediate that absorbs rounding. |
| 3. Core math | `r = cov / (sx * sy)`. One division, two square roots already done as part of sx/sy. | The op-specific bit. |
| 4. Final round | Return as f64 (the input dtype). Handles the case where `cov / (sx · sy)` rounds slightly outside [-1, 1] due to floating-point error. | The "round to RoundingMode" of arithmetic becomes "match the input dtype + boundary correction." Step exists; non-trivial. |
| 5. Canonicalize | Recipes-return-all-outputs (per CLAUDE.md feedback memory): pearson_r returns a struct with `(r, n_used, n_dropped, residual_norm)`, NOT just the headline scalar. The canonicalize step packages all outputs into the shape consumers expect. | This step would be missing in a "naive" implementation that returns just `r: f64`; in tambear's pure-math-stance contract, it's mandatory. |

**Verdict for pearson_r**: the five-step skeleton fits cleanly. All five steps are non-trivial in tambear's contract. The "arithmetic-specific" interpretations (special-value tag, p+50 guard bits, RoundingMode round, top-bit-set canonical form) all have natural analogs in the statistical-recipe domain.

✓ Conjecture holds for pearson_r.

---

## 3. Test instance 2 — `kendall_tau` (rank correlation)

**Recipe**: kendall_tau(x: &[f64], y: &[f64]) → f64. Returns Kendall's τ-b rank correlation coefficient.

Mathematical definition: τ-b = (concordant - discordant) / sqrt((n_pairs - tx) · (n_pairs - ty)) where tx and ty count tied pairs.

Per CLAUDE.md, kendall_tau decomposes into recipes: `sort`, `inversion_count`, `tie_count`, plus the τ-b formula.

**Five-step decomposition**:

| Step | What kendall_tau does | Non-trivial? |
|---|---|---|
| 1. Special-value dispatch | NaN inputs propagate; n < 2 → undefined; ties handling needs the τ-b variant (vs τ-a) | Yes, multiple cases. |
| 2. Precision extension | Sort x and y to ranked indices; for ties, use mid-rank averaging (per F-stance default). Inversion count is exact integer arithmetic — no floating-point precision concern. | The "precision extension" here is the *exact-integer* form: rank-based inversions are computed in integer arithmetic, sidestepping the float-precision question entirely. The step is still here — it's "convert to a representation in which the math is exact" — but it's exact-integer rather than higher-precision-float. Same SHAPE, different domain. |
| 3. Core math | inversion_count + tie_count → τ-b formula | The op-specific bit. |
| 4. Final round | Pack result back to f64. Same boundary-correction concern as pearson_r ([-1, 1] clamping). | Yes. |
| 5. Canonicalize | Return struct with (τ, n_used, n_concordant, n_discordant, n_tied_x, n_tied_y) per recipes-return-all-outputs. | Yes. |

**Verdict for kendall_tau**: skeleton fits. Step 2's "precision extension" generalizes from "more bits" to "exact representation in which the math is exact" — both are forms of *computing in a domain where rounding doesn't bite the answer*.

✓ Conjecture holds for kendall_tau.

---

## 4. Test instance 3 — `kaplan_meier` (survival function estimator)

**Recipe**: kaplan_meier(durations: &[f64], events: &[bool]) → KaplanMeierResult. Returns the empirical survival function.

Mathematical definition: S(t) = ∏_{t_i ≤ t} (1 - d_i/n_i) where t_i are unique event times, d_i is event count at t_i, n_i is at-risk count at t_i.

Per CLAUDE.md, kaplan_meier decomposes into recipes: `sort`, `prefix_product`, plus the conditional-survival formula.

**Five-step decomposition**:

| Step | What kaplan_meier does | Non-trivial? |
|---|---|---|
| 1. Special-value dispatch | NaN durations propagate; negative durations are invalid; events.len() must equal durations.len() | Yes. |
| 2. Precision extension | Sort by duration. The prefix-product chain `∏(1 - d_i/n_i)` is run in **log-space** for numerical stability when n_i is large and d_i/n_i is small (so `1 - d_i/n_i` is near 1, accumulating float error in the linear product). Sum-of-logs is the precision-stable form. | "Precision extension" here is "compute in log-space to avoid float underflow on the product chain" — a form of *higher-precision intermediate that absorbs rounding error*. Same SHAPE. |
| 3. Core math | prefix_sum-of-log followed by exp. | The op-specific bit. |
| 4. Final round | Convert log-space result back to linear S(t). | Yes. |
| 5. Canonicalize | Return KaplanMeierResult with full survival curve, confidence bands per Greenwood, censoring info, plus the input data record. | Yes — this is publication-grade rigor at the recipe level. |

**Verdict for kaplan_meier**: skeleton fits. Step 2's "precision extension" generalizes again — to "compute in log-space to absorb numerical error." Same shape, different domain implementation.

✓ Conjecture holds for kaplan_meier.

---

## 5. The skeleton is what "pure-math-stance" MEANS structurally

After three test instances, the conjecture survives. Two structural observations make it stronger than "it just happens to fit three recipes":

**Observation A — Step 2 is *the* signature of the pure-math stance.**

Diagnostic-stance and discovery-stance recipes don't have step 2. They CALL pure-math recipes (which have step 2 internally), but at the diagnostic/discovery layer, the call is just `pure_recipe(args)` — no precision-extension at the outer level. Workflow-stance recipes are *compositions of pure-math recipes*; the precision-extension is inherited from the inner pure-math recipes, not produced at the workflow layer.

**Step 2 is what a pure-math recipe DOES that other stances don't**: own its own precision discipline. The diagnostic stance owns method-selection. The override stance owns transparency. The workflow stance owns sequencing. The discovery stance owns superposition. The pure-math stance owns *precision discipline* — and step 2 is where that ownership manifests.

**Observation B — Step 5 is what *publication-grade rigor* means at the implementation layer.**

CLAUDE.md's recipes-return-all-outputs feedback (filed as a memory: "Recipes return ALL computed outputs in a struct, not just the headline scalar — R² alongside Kyle λ, trough_bucket alongside max_drawdown_pct, etc.") IS step 5. Every pure-math recipe MUST have step 5; otherwise it's not satisfying tambear's contract. Step 5 isn't optional polish — it's structural.

These two observations together make the five-step skeleton not a coincidence but a *definition*. **A pure-math recipe is a recipe with these five steps**, where step 2 owns precision and step 5 owns full-output disclosure. Recipes lacking step 2 are diagnostic/workflow/discovery stances; recipes lacking step 5 violate the contract.

---

## 6. The implication for sweep 35+

Sweep 35 ships BigFloat transcendentals (`exp`, `log`, `sin`, `cos`, etc.). Sweep 36+ extends the catalog (every measure in every family per the Tambear Contract §5).

**The op-invariant catalog generalizes**:
- Each new pure-math recipe gets the same six invariant families (F-A through F-H from op-invariant-catalog.md §0).
- The shared invariants (NaN propagation, identity-when-applicable, fast-path agreement, cross-precision consistency, mpmath bit-exact, canonical form) are inherited from the trait.
- The op-specific invariants are 1-3 per recipe, just like add/sub/mul/div/sqrt.

**The trait shape generalizes**:
- `RecipeInvariants` (renamed from `OpInvariants` to reflect the broader scope) is implemented by every pure-math recipe.
- Each implementor specifies the op-specific invariants via the `op_specific_invariants(p)` method.
- The shared methods (`nan_propagation_test`, `cross_precision_consistency_test`, `mpmath_oracle_test`, `canonical_form_post_condition`) are inherited.

**The gauntlet count grows linearly, not quadratically**:
- 2000 cells for add/sub/mul/div/sqrt at sweep 31.
- ~400 cells per new recipe (1 recipe × ~70 invariants × 6 precisions × 5 rounding modes ≈ 2100 / 5 ops × 1 op = 420). This is a *per-recipe* number; the catalog's total scales linearly.
- For ~30 statistical recipes in sweep 36+ (mean, var, cov, std, skewness, kurtosis, correlation variants, etc.) → ~12,000 cells. Still tractable with the trait shape.
- Without the trait: 12,000 nearly-identical test functions. Catastrophic.

**This is the F11.2 graph-form forcing for trait-promotion-now restated at scale**: the silent-failure surface naturalist named at 2000 cells is 6x worse at sweep 36 (12,000 cells), 30x worse at sweep 40+ (60,000 cells if we hit ~150 recipes per Tambear Contract §5). The promote-now case strengthens monotonically as the catalog grows.

---

## 7. What's load-bearing vs not

**Load-bearing** (worth preserving across sessions):
- The five-step skeleton is the structural definition of pure-math stance, not BigFloat-specific.
- Step 2 (precision extension) and step 5 (canonicalize / return-all-outputs) are the load-bearing identifiers — recipes lacking these aren't pure-math stance.
- The `RecipeInvariants` trait shape (generalized from `OpInvariants`) is the test-infrastructure-pattern for the entire pure-math catalog at sweep 35+.

**Not load-bearing** (specific to this session's analysis):
- The three test instances (pearson_r, kendall_tau, kaplan_meier) — they verified the conjecture but the conjecture-itself is what generalizes, not the specific recipes.
- The cell-count estimates are order-of-magnitude.

**Open for naturalist if they continue this thread**: the diagnostic/override/workflow/discovery stances probably have their own structural-skeletons that aren't five-step. Worth a parallel analysis when the catalog reaches those stances. The cross-stance taxonomy isn't this doc's scope.

---

## 8. Provenance + cross-references

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish` after naturalist's bz-skeleton-converges.md §"A different angle — recipes by stance" planted the conjecture.
- Three test-instances drawn from CLAUDE.md's recipe-catalog table (kendall_tau, pearson_r, kaplan_meier).
- Cross-checked: the F-stance taxonomy in CLAUDE.md "Recipes by Behavioral Stance" → step 2 (precision extension) IS what makes pure-math stance distinct from diagnostic/override/workflow/discovery stances. Step 5 (canonicalize / return-all-outputs) IS the recipes-return-all-outputs feedback from project memory operationalized at the implementation layer.
- This doc is filed in the campsite for sweep 31 because the connection is fresh now; future-context-self won't have to re-derive when sweep 35 starts. The artifact is the conjecture-survives-three-instances finding, not action-this-session.
- Reciprocal cross-reference to naturalist's garden: their bz-skeleton entry §3 raised the conjecture; this doc is the worked verification across three more recipes. Naturalist may want to reciprocate-back into garden if they continue the thread; not asking.

— math-researcher
