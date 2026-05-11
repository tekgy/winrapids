# Recipes by Behavioral Stance — Pure / Diagnostic / Override / Workflow / Discovery

> **Source**: this document was extracted from CLAUDE.md (2026-05-08) so the project root file stays under context-budget. CLAUDE.md retains the directive principle plus the five stance names with one-line summaries; this document is the elaboration.
>
> **Vocabulary**: see `docs/architecture/vocabulary.md` (canonical, locked 2026-04-17). All five stances live at vocabulary tier 4 (recipes); they are not separate tiers.

Tambear's recipes (locked vocabulary Tier 4) are pure math. They do one thing, do it correctly, and return a result. That's it — no opinions, no diagnostics, no method-switching, no workflow assembly. A recipe like `spearman_correlation(x, y)` just computes Spearman's ρ.

The *smart* parts of tambear — the parts that make it feel like a $10M/year quant is sitting next to the user — live in **other recipes that compose the pure ones**. These higher-stance recipes pick methods, tune parameters, run diagnostics, surface superposition. They are first-class products in their own right, but they are *also recipes* — same vocabulary tier, distinguished only by their *behavioral stance*. This separation is what keeps the pure recipes clean, reusable, and composable.

> **Vocabulary note.** Earlier versions of this material called these "Layer 0 / Layer 1 / Layer 2 / Layer 3 / Layer 4." Under the locked vocabulary (`docs/architecture/vocabulary.md`), there are no separate "layers" — every named composition is a recipe. What follows is a description of *behavioral stances* recipes can take, all at the same vocabulary tier (Tier 4 — Recipes).

## The five behavioral stances

**Pure-math stance** (the bulk of the recipe catalog).
A recipe like `spearman_correlation(x, y) -> f64`. One operation, one result. Atom + primitive composition. Shareable intermediates. Benchmarked, oracled, adversarial-tested. No knowledge of pipelines, no knowledge of diagnostics, no knowledge of other recipes' decision-making.

**Diagnostic / auto-selection stance**.
A recipe that knows *which other recipe to call for which data*. When a user asks for the generic thing (`correlation(x, y)`), this stance runs the diagnostics a senior statistician would run (normality check, variable-type check, outlier influence, ties), picks the appropriate sub-recipe (Pearson / Spearman / Kendall τ-b / polychoric / tetrachoric / point-biserial / distance correlation), calls it, and reports the decision + rationale in a structured output (`TbsStepAdvice`: `recommended`, `user_override`, `diagnostics`). It's built *out of* pure-math recipes — it calls `shapiro_wilk`, `pearson_r`, `spearman_r`, etc. — but those pure-math recipes know nothing about being inside a diagnostic recipe.

**Override-transparency stance**.
A diagnostic recipe wrapped in a transparency layer. When a user forces a specific method via `using(method="pearson")`, this stance still runs the diagnostic recommendation *and* the user's choice, and writes both results into the output alongside a warning if the override is statistically questionable. The user sees what they chose, what tambear would have chosen, and the numerical difference — so they disagree with their eyes open. Neither silently wins. The underlying pure-math recipes are called twice from the same diagnostic-backed infrastructure.

**Workflow stance** (named multi-step recipes).
Curated workflows shipped as first-class recipes — `two_group_comparison`, `regression_diagnostics`, `time_series_analysis`, `clustering_workflow`, `survival_analysis`, `efa`, `garch_fit`, `sip_signal_bundle`, etc. Each is a composition of pure-math recipes orchestrated through diagnostic and override-transparency recipes, tuned as if a decade-experienced domain expert was hired to build that one workflow from scratch. Users load a workflow recipe and run it; inside, it walks the decision tree (normality → variance homogeneity → t-test variant → effect size → power → CI → robustness) calling sub-recipes as needed. Each workflow recipe has its own version, its own test suite against published benchmark datasets, and its own oracle against whatever the gold-standard workflow in that domain looks like. Workflow recipes are what most users reach for; the pure-math toolkit stays available for people who want to assemble their own.

**Discovery / superposition stance**.
A recipe that runs *every plausible method simultaneously*, keeps all results in superposition, and reports structural fingerprints of agreement/disagreement across methods. When the user wants exploration rather than a single answer, this stance surfaces "which views agree, which disagree, and what does that tell us about the data?" It's the opposite philosophical stance from auto-selection (auto-selection picks; discovery refuses to pick) but uses the same pure-math recipes underneath. Both stances exist because both are valid depending on the user's question.

## Why separating stances matters

- **Pure-math recipes stay reusable**. `spearman_correlation` can be called from a diagnostic recipe (auto-picked it), an override-transparency recipe (user forced it), a workflow recipe (regression workflow needs a rank correlation on residuals), a discovery recipe (superposition ensemble), or a user's own custom code. Zero coupling to any one consumer.
- **Stances can be swapped or extended**. A new diagnostic recipe that prefers Bayesian reasoning over frequentist can be built without touching a single pure-math recipe. A new workflow genre (e.g. experimental-design workflows) is just a new workflow recipe that calls existing pure-math recipes.
- **Testing is cleaner**. A pure-math recipe is tested against its math. A diagnostic recipe is tested against whether it picks the right sub-recipe. A workflow recipe is tested against whether it produces the expert's workflow. Each stance has its own oracle.
- **The Contract stays scoped**. The Filter Test (numerical correctness, oracle comparison, adversarial coverage) applies to pure-math recipes. Diagnostic / workflow / discovery recipes have their own separate quality gates — they don't need to be bit-perfect vs scipy (because they're workflow orchestration, not math), but they do need to make defensible methodological choices, match expert judgment on benchmark scenarios, and preserve override transparency.

**The rule**: if you find yourself wanting a pure-math recipe to "know about" normality checks, method switching, expert defaults, or diagnostic wiring — stop. That knowledge belongs in a recipe of a different stance that wraps yours. The pure-math recipe stays pure. The wrapping recipe composes it.
