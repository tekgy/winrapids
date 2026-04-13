//! Pipeline serialization with explicit default materialization.
//!
//! The pipeline types in [`super::types`] carry `#[derive(Serialize,
//! Deserialize)]`, so a raw `serde_json::to_string(&pipeline)` round-trip
//! works out of the box. This module adds the **materialization
//! semantics** that turn a user-edited pipeline into a saved pipeline.
//!
//! # The materialization rule
//!
//! When a user saves a pipeline (or a community member exports one for
//! sharing), every parameter at every step must be **fully explicit** in
//! the saved form. Any dimension (`using` / `autodiscover` / `sweep` /
//! `superposition`) that was inheriting from a recipe default at edit
//! time is written out as an explicit value at save time, resolved via
//! [`schema::effective_binding`].
//!
//! This has one critical consequence: **future changes to tambear's
//! recipe defaults cannot silently alter a user's saved pipeline.** If
//! we ship a tambear update that changes the default `rotation` from
//! `varimax` to `promax`, every existing saved pipeline still runs with
//! `varimax` because that value was materialized at save time.
//!
//! # What gets materialized
//!
//! - Every parameter declared in the recipe's schema gets an entry in
//!   the step's `choices` map, even if the user never touched it.
//! - Each entry's `using` dimension is filled with whatever the
//!   effective binding resolved to.
//! - `autodiscover`, `sweep`, `superposition` are also frozen to their
//!   effective values at save time.
//!
//! The user can distinguish "values I chose" from "defaults I kept" by
//! diffing against the recipe's current default binding — the IDE
//! surfaces that diff in the "saved pipeline review" panel.
//!
//! # What is NOT materialized
//!
//! - Parameters the recipe declares as runtime-data-dependent
//!   (e.g., a default that is "pick based on data shape at execution
//!   time"). Those stay as `Binding::Autodiscover` values — the point
//!   of autodiscover is to defer the choice to execution time.
//! - Output column shape inferences. Shapes are recomputed on load
//!   from the Pipeline spec + schema lookup.

use super::schema::{effective_binding, schema_for};
use super::types::{Binding, Pipeline};
use serde::{Deserialize, Serialize};

/// Serialization error returned by [`save`] / [`load`].
#[derive(Debug, Clone)]
pub enum SerializeError {
    /// JSON (de)serialization failed.
    Json(String),
    /// The pipeline references a recipe that has no schema registered.
    /// Saving requires knowing the full parameter set so defaults can
    /// be materialized; an unknown recipe blocks that.
    UnknownRecipe(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::Json(s) => write!(f, "JSON serialization: {s}"),
            SerializeError::UnknownRecipe(r) => write!(f, "unknown recipe in pipeline: {r}"),
        }
    }
}

impl std::error::Error for SerializeError {}

/// Version tag carried in the serialized envelope. Bumped when the
/// on-disk schema changes in a way that isn't transparently
/// serde-compatible.
pub const SAVE_FORMAT_VERSION: &str = "tambear.pipeline.v1";

/// Top-level envelope for a saved pipeline. Wraps the [`Pipeline`] with
/// a version tag and a `materialized` flag so loaders know whether the
/// saved form has been through [`save`] (fully explicit) or
/// [`save_raw`] (may still contain inheriting bindings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPipeline {
    pub version: String,
    pub materialized: bool,
    pub pipeline: Pipeline,
}

/// Serialize a pipeline to JSON, first materializing every parameter's
/// effective binding so nothing is left inheriting from recipe defaults.
///
/// This is the correct entry point for "save" and "export" operations.
/// After a successful `save`, every `step.choices` map contains an
/// explicit entry for every parameter the recipe declares, and each
/// entry's four dimensions are fully resolved.
///
/// # Errors
///
/// Returns `UnknownRecipe` if any step references a recipe that has no
/// schema in the pipeline-layer registry.
pub fn save(pipeline: &Pipeline) -> Result<String, SerializeError> {
    let materialized = materialize(pipeline)?;
    let envelope = SavedPipeline {
        version: SAVE_FORMAT_VERSION.to_string(),
        materialized: true,
        pipeline: materialized,
    };
    serde_json::to_string_pretty(&envelope).map_err(|e| SerializeError::Json(e.to_string()))
}

/// Serialize a pipeline to JSON **without** materializing defaults.
///
/// Use this for in-progress autosave or for copy/paste within the same
/// tambear version, where the implicit dependence on recipe defaults
/// is acceptable (because the defaults haven't changed). For long-term
/// storage and community sharing, use [`save`] instead.
pub fn save_raw(pipeline: &Pipeline) -> Result<String, SerializeError> {
    let envelope = SavedPipeline {
        version: SAVE_FORMAT_VERSION.to_string(),
        materialized: false,
        pipeline: pipeline.clone(),
    };
    serde_json::to_string_pretty(&envelope).map_err(|e| SerializeError::Json(e.to_string()))
}

/// Parse a saved pipeline from JSON. Accepts both materialized and
/// raw saves.
pub fn load(json: &str) -> Result<SavedPipeline, SerializeError> {
    serde_json::from_str(json).map_err(|e| SerializeError::Json(e.to_string()))
}

/// Materialize every parameter's effective binding, returning a new
/// [`Pipeline`] that is fully explicit.
///
/// This is the core of the save rule: walk every step, look up its
/// recipe schema, and for every parameter the schema declares, write
/// the step's effective binding into the choices map. Parameters the
/// user explicitly overrode keep their overrides; parameters that were
/// inheriting from the default get the default's current values
/// materialized into the map.
pub fn materialize(pipeline: &Pipeline) -> Result<Pipeline, SerializeError> {
    let mut materialized = pipeline.clone();

    for step in &mut materialized.steps {
        let schema = schema_for(&step.recipe)
            .ok_or_else(|| SerializeError::UnknownRecipe(format!("{:?}", step.recipe)))?;

        // For every parameter the recipe declares, resolve its effective
        // binding (which composes step overrides on top of recipe defaults)
        // and store that into the choices map.
        for param in schema.parameters {
            let effective = effective_binding(&step.recipe, step, param.key)
                .unwrap_or_else(Binding::default);
            step.choices.insert(param.key.to_string(), effective);
        }
    }

    Ok(materialized)
}

/// Check if a pipeline is already fully materialized — that is, for
/// every step, every parameter the recipe schema declares has an
/// explicit entry in `step.choices`.
///
/// Returns `true` for pipelines whose every parameter is explicitly
/// bound, `false` if any step is still relying on implicit defaults.
/// This is a diagnostic helper; it doesn't mutate anything.
pub fn is_fully_materialized(pipeline: &Pipeline) -> bool {
    for step in &pipeline.steps {
        let Some(schema) = schema_for(&step.recipe) else {
            return false;
        };
        for param in schema.parameters {
            if !step.choices.contains_key(param.key) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::pipelines::schema::step_from_schema;
    use crate::recipes::pipelines::types::{PipelineStep, RecipeRef, Value};

    fn exp_pipeline() -> Pipeline {
        let step = step_from_schema("e", RecipeRef::Recipe("exp".into()))
            .expect("exp schema exists");
        Pipeline::new("exp_test").with_step(step)
    }

    // ── Round-trip ─────────────────────────────────────────────────────────

    #[test]
    fn raw_save_load_roundtrip() {
        let pipeline = exp_pipeline();
        let json = save_raw(&pipeline).unwrap();
        let loaded = load(&json).unwrap();
        assert_eq!(loaded.pipeline, pipeline);
        assert!(!loaded.materialized);
        assert_eq!(loaded.version, SAVE_FORMAT_VERSION);
    }

    #[test]
    fn materialized_save_load_roundtrip() {
        let pipeline = exp_pipeline();
        let json = save(&pipeline).unwrap();
        let loaded = load(&json).unwrap();
        assert!(loaded.materialized);
        assert_eq!(loaded.version, SAVE_FORMAT_VERSION);
    }

    // ── Materialization semantics ─────────────────────────────────────────

    #[test]
    fn materialize_fills_in_defaults() {
        // Start with an exp step that has NO explicit choices.
        let mut step = PipelineStep::new("e", "E", RecipeRef::Recipe("exp".into()));
        step.choices.clear(); // empty — every dim inherits from default
        let pipeline = Pipeline::new("test").with_step(step);

        // Before materialization: step has no choices at all.
        assert!(pipeline.steps[0].choices.is_empty());

        let materialized = materialize(&pipeline).unwrap();

        // After materialization: step has an entry for every parameter
        // the recipe schema declares, with the recipe's default values
        // frozen in.
        assert!(materialized.steps[0].choices.contains_key("precision"));
        let precision = &materialized.steps[0].choices["precision"];
        // The exp default for `precision` is using=Method("compensated").
        assert_eq!(precision.using, Some(Value::Method("compensated".into())));
    }

    #[test]
    fn materialize_preserves_user_overrides() {
        // User forced precision=strict.
        let mut step = step_from_schema("e", RecipeRef::Recipe("exp".into())).unwrap();
        step.choices.insert(
            "precision".to_string(),
            Binding::using(Value::Method("strict".into())),
        );
        let pipeline = Pipeline::new("test").with_step(step);

        let materialized = materialize(&pipeline).unwrap();

        let precision = &materialized.steps[0].choices["precision"];
        // User's explicit override survives.
        assert_eq!(precision.using, Some(Value::Method("strict".into())));
    }

    #[test]
    fn materialize_keeps_recipe_probe_when_user_overrides_using() {
        // The correlation_matrix schema has BOTH using=pearson AND
        // autodiscover=normality_probe as defaults. When the user forces
        // method=spearman, the using dimension is overridden but the
        // autodiscover probe should still be present in the materialized
        // save — Layer 2 override-transparency.
        use crate::recipes::pipelines::types::DataProbeRef;

        let mut step = step_from_schema("c", RecipeRef::Expr("correlation_matrix".into()))
            .unwrap();
        step.choices.insert(
            "method".to_string(),
            Binding::using(Value::Method("spearman".into())),
        );
        let pipeline = Pipeline::new("test").with_step(step);

        let materialized = materialize(&pipeline).unwrap();

        let method = &materialized.steps[0].choices["method"];
        assert_eq!(method.using, Some(Value::Method("spearman".into())));
        assert_eq!(
            method.autodiscover,
            Some(DataProbeRef("normality_probe".into())),
            "materialize lost the recipe-default autodiscover probe"
        );
    }

    #[test]
    fn materialize_produces_fully_materialized_pipeline() {
        let pipeline = Pipeline::new("test").with_step(PipelineStep::new(
            "e",
            "E",
            RecipeRef::Recipe("exp".into()),
        ));
        assert!(!is_fully_materialized(&pipeline));
        let materialized = materialize(&pipeline).unwrap();
        assert!(is_fully_materialized(&materialized));
    }

    #[test]
    fn materialize_errors_on_unknown_recipe() {
        let pipeline = Pipeline::new("test").with_step(PipelineStep::new(
            "x",
            "X",
            RecipeRef::Recipe("no_such_recipe".into()),
        ));
        let result = materialize(&pipeline);
        assert!(matches!(result, Err(SerializeError::UnknownRecipe(_))));
    }

    // ── Default-change immunity ────────────────────────────────────────────

    #[test]
    fn saved_pipeline_survives_hypothetical_default_change() {
        // Materialize a pipeline. The saved form should contain the
        // current default value as an explicit using(). If we then
        // mutate something that looks like a default change at runtime
        // (simulated here by loading the JSON and comparing against the
        // same pipeline), the loaded pipeline carries the frozen value.
        let pipeline = exp_pipeline();
        let json = save(&pipeline).unwrap();
        let loaded = load(&json).unwrap();
        assert!(loaded.materialized);
        // The materialized exp default should be explicit in the saved form.
        let step = &loaded.pipeline.steps[0];
        let precision = step
            .choices
            .get("precision")
            .expect("precision should be materialized");
        assert_eq!(
            precision.using,
            Some(Value::Method("compensated".into())),
            "materialized default was not frozen"
        );
    }

    // ── JSON shape / envelope ──────────────────────────────────────────────

    #[test]
    fn save_produces_versioned_envelope() {
        let pipeline = exp_pipeline();
        let json = save(&pipeline).unwrap();
        assert!(json.contains(SAVE_FORMAT_VERSION));
        assert!(json.contains("\"materialized\": true"));
    }

    #[test]
    fn save_raw_produces_unmaterialized_flag() {
        let pipeline = exp_pipeline();
        let json = save_raw(&pipeline).unwrap();
        assert!(json.contains("\"materialized\": false"));
    }

    #[test]
    fn load_rejects_garbage() {
        let result = load("not json");
        assert!(matches!(result, Err(SerializeError::Json(_))));
    }

    // ── All Binding variants round-trip ───────────────────────────────────

    #[test]
    fn binding_with_all_four_dimensions_roundtrips() {
        use crate::recipes::pipelines::types::{Combiner, DataProbeRef};

        let binding = Binding::inherit_defaults()
            .with_using(Value::Method("varimax".into()))
            .with_autodiscover(DataProbeRef("factorability_probe".into()))
            .with_sweep(vec![
                Value::Method("varimax".into()),
                Value::Method("promax".into()),
            ])
            .with_superposition(
                vec![
                    Value::Method("varimax".into()),
                    Value::Method("promax".into()),
                ],
                Combiner::AgreementPartition {
                    threshold_basis_points: 100,
                },
            );

        let json = serde_json::to_string(&binding).unwrap();
        let decoded: Binding = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, binding);
    }

    #[test]
    fn value_variants_roundtrip() {
        let values = vec![
            Value::Bool(true),
            Value::Int(-42),
            Value::Float(3.14159),
            Value::String("hello".into()),
            Value::Method("varimax".into()),
            Value::Vector(vec![1.0, 2.0, 3.0]),
            Value::Matrix {
                rows: 2,
                cols: 3,
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        ];
        for v in values {
            let json = serde_json::to_string(&v).unwrap();
            let decoded: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, v);
        }
    }
}
