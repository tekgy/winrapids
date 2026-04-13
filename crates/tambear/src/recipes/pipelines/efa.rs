//! Exploratory Factor Analysis (EFA) — the pilot pipeline.
//!
//! This is the first "real" pipeline built on the Phase D0/D0b pipeline
//! layer. It demonstrates how a familiar multivariate-statistics workflow
//! decomposes into an ordered list of recipe invocations with composable
//! per-parameter bindings.
//!
//! # The workflow
//!
//! EFA as taught in any multivariate-stats textbook (Tabachnick & Fidell,
//! Hair et al., Fabrigar & Wegener) is conventionally a six-move dance:
//!
//! 1. **Compute the correlation matrix** from the raw observation data.
//!    This is the input to everything downstream. Pearson is the
//!    standard default; the `correlation_matrix` recipe keeps its normality
//!    autodiscover probe active as advisory, so the user is told if
//!    Spearman/Kendall would be safer on non-normal data.
//! 2. **KMO sampling adequacy.** Kaiser-Meyer-Olkin answers "is this
//!    correlation matrix even factorable?" by comparing the magnitudes of
//!    observed correlations to partial correlations. The result is a
//!    scalar plus a per-variable MSA vector — variables with MSA below
//!    ~0.5 are candidates for removal before factoring.
//! 3. **Bartlett's test of sphericity.** Complements KMO from the other
//!    side: tests the null hypothesis that the correlation matrix equals
//!    the identity (i.e. no meaningful correlations at all). If Bartlett
//!    fails to reject, factoring is pointless.
//! 4. **Factor analysis.** The main event: extract latent factors from
//!    the correlation matrix, using `minres` / `ml` / `pa` and applying a
//!    rotation (`varimax` by default). Wired to [`FACTOR_ANALYSIS_SCHEMA`]
//!    which declares the full parameter set with auto-select defaults for
//!    `n_factors` (parallel analysis / Kaiser) and factorability-probe
//!    advice for `extraction`.
//! 5. **Factor scores.** Turn the loadings back into per-observation
//!    quantities so the scores can be used downstream (clustering,
//!    regression, visualization). Regression/Thurstone is the default;
//!    Bartlett and Anderson-Rubin are alternatives.
//! 6. **Model fit diagnostics.** RMSEA, TLI, CFI, and the residual
//!    correlation Frobenius norm — the standard structural-equation-model
//!    fit indices, used here to sanity-check the extracted factor
//!    solution.
//!
//! # Pipeline vs monolithic recipe
//!
//! The IDE's `factor_analysis` method is a single-call surface with seven
//! sub-parameters — the user ticks boxes and gets everything at once.
//! This pipeline is the *decomposed* form: the same workflow expressed as
//! six addressable steps so each intermediate (correlation matrix, KMO,
//! Bartlett, factor solution, scores, fit) is a first-class value with
//! its own output columns, its own per-parameter bindings, and its own
//! IDE label. Both forms coexist — users pick whichever surface matches
//! their task.
//!
//! # Still pure specification
//!
//! Nothing in this file touches data or runs math. The result of
//! [`exploratory_factor_analysis`] is a [`Pipeline`] value that describes
//! *what would run* if a user bound data to it — the orchestration /
//! execution crate is responsible for actually running the steps.

use super::schema::step_from_schema;
use super::types::{Pipeline, PipelineStep, RecipeRef};

/// Build the six-step EFA pilot pipeline.
///
/// Every step is populated from its recipe schema via [`step_from_schema`]
/// so all parameter defaults — across the `using`, `autodiscover`,
/// `sweep`, and `superposition` dimensions — come from the recipes
/// themselves. The pipeline owns only the sequencing and the step keys.
///
/// Step keys are short and stable:
///
/// | index | key                  | recipe                                        |
/// |-------|----------------------|-----------------------------------------------|
/// | 0     | `cor`                | `Expr("correlation_matrix")`                  |
/// | 1     | `kmo`                | `Expr("kmo")`                                 |
/// | 2     | `bartlett`           | `Expr("bartlett_sphericity")`                 |
/// | 3     | `fa`                 | `Recipe("factor_analysis")`                   |
/// | 4     | `scores`             | `Expr("factor_scores")`                       |
/// | 5     | `fit`                | `Expr("model_fit_diagnostics")`               |
pub fn exploratory_factor_analysis() -> Pipeline {
    Pipeline::new("exploratory_factor_analysis")
        .with_description(
            "Exploratory factor analysis with explicit diagnostics. Computes the correlation \
             matrix, checks factorability (KMO + Bartlett), extracts factors (minres + varimax \
             by default), produces per-observation factor scores, and reports fit indices \
             (RMSEA, TLI, CFI, residual norm).",
        )
        .with_step(build_correlation_matrix_step())
        .with_step(build_kmo_step())
        .with_step(build_bartlett_step())
        .with_step(build_factor_analysis_step())
        .with_step(build_factor_scores_step())
        .with_step(build_model_fit_step())
}

fn build_correlation_matrix_step() -> PipelineStep {
    step_from_schema("cor", RecipeRef::Expr("correlation_matrix".into()))
        .expect("correlation_matrix schema must be registered")
}

fn build_kmo_step() -> PipelineStep {
    step_from_schema("kmo", RecipeRef::Expr("kmo".into()))
        .expect("kmo schema must be registered")
}

fn build_bartlett_step() -> PipelineStep {
    step_from_schema("bartlett", RecipeRef::Expr("bartlett_sphericity".into()))
        .expect("bartlett_sphericity schema must be registered")
}

fn build_factor_analysis_step() -> PipelineStep {
    step_from_schema("fa", RecipeRef::Recipe("factor_analysis".into()))
        .expect("factor_analysis schema must be registered")
}

fn build_factor_scores_step() -> PipelineStep {
    step_from_schema("scores", RecipeRef::Expr("factor_scores".into()))
        .expect("factor_scores schema must be registered")
}

fn build_model_fit_step() -> PipelineStep {
    step_from_schema("fit", RecipeRef::Expr("model_fit_diagnostics".into()))
        .expect("model_fit_diagnostics schema must be registered")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::pipelines::schema::{effective_binding, schema_for};
    use crate::recipes::pipelines::types::Value;

    // ── Structural tests ───────────────────────────────────────────────────

    #[test]
    fn pipeline_has_expected_name_and_description() {
        let p = exploratory_factor_analysis();
        assert_eq!(p.name, "exploratory_factor_analysis");
        assert!(!p.description.is_empty());
    }

    #[test]
    fn pipeline_has_six_steps() {
        let p = exploratory_factor_analysis();
        assert_eq!(p.steps.len(), 6);
    }

    #[test]
    fn steps_are_in_expected_order() {
        let p = exploratory_factor_analysis();
        let keys: Vec<&str> = p.steps.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["cor", "kmo", "bartlett", "fa", "scores", "fit"]
        );
    }

    #[test]
    fn steps_have_expected_recipe_refs() {
        let p = exploratory_factor_analysis();
        assert_eq!(
            p.steps[0].recipe,
            RecipeRef::Expr("correlation_matrix".into())
        );
        assert_eq!(p.steps[1].recipe, RecipeRef::Expr("kmo".into()));
        assert_eq!(
            p.steps[2].recipe,
            RecipeRef::Expr("bartlett_sphericity".into())
        );
        assert_eq!(
            p.steps[3].recipe,
            RecipeRef::Recipe("factor_analysis".into())
        );
        assert_eq!(p.steps[4].recipe, RecipeRef::Expr("factor_scores".into()));
        assert_eq!(
            p.steps[5].recipe,
            RecipeRef::Expr("model_fit_diagnostics".into())
        );
    }

    // ── Schema coverage ────────────────────────────────────────────────────

    #[test]
    fn every_step_has_a_registered_schema() {
        let p = exploratory_factor_analysis();
        for step in &p.steps {
            assert!(
                schema_for(&step.recipe).is_some(),
                "step {} has no schema registered for {:?}",
                step.key,
                step.recipe
            );
        }
    }

    #[test]
    fn every_step_has_at_least_one_output() {
        // step_from_schema populates outputs from the schema's OutputSpec
        // list. A zero-output step would indicate a schema bug we want
        // to catch here.
        let p = exploratory_factor_analysis();
        for step in &p.steps {
            assert!(
                !step.outputs.is_empty(),
                "step {} has zero outputs",
                step.key
            );
        }
    }

    // ── Step-by-step binding checks ────────────────────────────────────────

    #[test]
    fn correlation_step_default_method_is_pearson_with_normality_probe() {
        let p = exploratory_factor_analysis();
        let step = p.find_step("cor").unwrap();
        let effective = effective_binding(&step.recipe, step, "method")
            .expect("method should resolve");
        assert_eq!(effective.using, Some(Value::Method("pearson".into())));
        assert!(effective.autodiscover.is_some());
        assert_eq!(
            effective.autodiscover.as_ref().unwrap().0,
            "normality_probe"
        );
    }

    #[test]
    fn kmo_step_has_no_parameters() {
        // KMO's canonical form is parameter-free — the test exists so we
        // notice if someone accidentally adds a knob to the schema.
        let p = exploratory_factor_analysis();
        let step = p.find_step("kmo").unwrap();
        assert!(step.choices.is_empty());
    }

    #[test]
    fn bartlett_step_default_alpha_is_0_05() {
        let p = exploratory_factor_analysis();
        let step = p.find_step("bartlett").unwrap();
        let effective = effective_binding(&step.recipe, step, "alpha")
            .expect("alpha should resolve");
        assert_eq!(effective.using, Some(Value::Float(0.05)));
    }

    #[test]
    fn factor_analysis_step_default_is_minres_varimax_auto_nfactors() {
        let p = exploratory_factor_analysis();
        let step = p.find_step("fa").unwrap();

        // n_factors has no using default — parallel analysis / Kaiser
        // auto-discovers at runtime.
        let n_factors = effective_binding(&step.recipe, step, "n_factors").unwrap();
        assert!(n_factors.using.is_none());
        assert!(n_factors.autodiscover.is_some());
        assert_eq!(
            n_factors.autodiscover.as_ref().unwrap().0,
            "parallel_analysis_or_kaiser"
        );

        // extraction defaults to minres (matches IDE default, not the
        // previous `pa` placeholder).
        let extraction = effective_binding(&step.recipe, step, "extraction").unwrap();
        assert_eq!(extraction.using, Some(Value::Method("minres".into())));
        assert!(extraction.autodiscover.is_some());
        assert_eq!(
            extraction.autodiscover.as_ref().unwrap().0,
            "factorability_probe"
        );

        // rotation defaults to varimax (no autodiscover advice yet).
        let rotation = effective_binding(&step.recipe, step, "rotation").unwrap();
        assert_eq!(rotation.using, Some(Value::Method("varimax".into())));

        // missing defaults to listwise.
        let missing = effective_binding(&step.recipe, step, "missing").unwrap();
        assert_eq!(missing.using, Some(Value::Method("listwise".into())));

        // fm_se defaults to false.
        let fm_se = effective_binding(&step.recipe, step, "fm_se").unwrap();
        assert_eq!(fm_se.using, Some(Value::Bool(false)));
    }

    #[test]
    fn factor_scores_step_default_method_is_regression() {
        let p = exploratory_factor_analysis();
        let step = p.find_step("scores").unwrap();
        let method = effective_binding(&step.recipe, step, "method").unwrap();
        assert_eq!(method.using, Some(Value::Method("regression".into())));
    }

    #[test]
    fn model_fit_step_default_rmsea_threshold_is_conventional() {
        let p = exploratory_factor_analysis();
        let step = p.find_step("fit").unwrap();
        let threshold = effective_binding(&step.recipe, step, "rmsea_threshold").unwrap();
        assert_eq!(threshold.using, Some(Value::Float(0.06)));
    }

    // ── Pipeline-level operations work ─────────────────────────────────────

    #[test]
    fn force_deterministic_preserves_shape() {
        // The reproducibility lockdown on a freshly-built EFA pipeline
        // shouldn't change step count or key order.
        let mut p = exploratory_factor_analysis();
        let keys_before: Vec<String> =
            p.steps.iter().map(|s| s.key.clone()).collect();
        p.force_deterministic();
        let keys_after: Vec<String> =
            p.steps.iter().map(|s| s.key.clone()).collect();
        assert_eq!(keys_before, keys_after);
        assert_eq!(p.steps.len(), 6);
    }
}
