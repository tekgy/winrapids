//! Pre-execution chain shape inference.
//!
//! Given a [`Pipeline`] spec (with no data attached), walk it step by
//! step and report what each step's output columns will look like at
//! runtime. This is what the IDE reads to render the chain graph
//! before the user has loaded any data or clicked "run".
//!
//! # What is known at spec time
//!
//! - The recipe at each step (from `PipelineStep.recipe`).
//! - The recipe's declared [`OutputShape`] for each output column
//!   (from the schema).
//! - The effective binding for every parameter at each step (via
//!   `schema::effective_binding`), including any
//!   [`DimensionSource::FromChoice`] resolutions.
//!
//! # What is NOT known at spec time
//!
//! - Data row/column counts. A shape that depends on
//!   [`DimensionSource::InputRows`] or [`DimensionSource::InputCols`]
//!   is reported with those sources intact — the execution layer
//!   fills them in when data is attached. The IDE can also render them
//!   as "rows = N" or "cols = p" placeholders in the chain graph.
//!
//! # Error model
//!
//! Shape inference is **best-effort**. If a step references a recipe
//! with no schema, or a `DimensionSource::FromChoice` points at a choice
//! that isn't present, we produce a [`ResolvedColumn`] with an
//! [`UnresolvedDimension`] marker rather than failing the whole
//! operation. The IDE surfaces these as warnings without blocking the
//! user's edit flow — they fix the root cause and shape inference
//! re-runs automatically.

use super::schema::{effective_binding, schema_for};
use super::types::{
    DimensionSource, Dtype, OutputShape, Pipeline, PipelineStep, Value,
};
use serde::{Deserialize, Serialize};

/// A resolved chain shape: one [`ResolvedStep`] per step in the
/// pipeline, each with its output columns' shapes computed as far as
/// possible at spec time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainShape {
    pub steps: Vec<ResolvedStep>,
}

/// Shape info for one step after inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedStep {
    /// Copied from `PipelineStep.key`.
    pub key: String,
    /// Display name copied from `PipelineStep.display_name`.
    pub display_name: String,
    /// Column-by-column resolved shapes. One entry per
    /// `PipelineStep.outputs` (or per schema output if the step's own
    /// outputs list is empty).
    pub columns: Vec<ResolvedColumn>,
    /// Any inference warnings that apply to this step. These don't
    /// block the pipeline; the IDE surfaces them as soft warnings.
    pub warnings: Vec<ShapeWarning>,
}

/// A single output column with its resolved shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedColumn {
    pub semantic_name: String,
    pub description: String,
    pub shape: ResolvedShape,
    pub dtype: Dtype,
    pub has_v_column: bool,
}

/// The resolved shape: same variants as [`OutputShape`] but with
/// [`ResolvedDimension`] in place of [`DimensionSource`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolvedShape {
    Scalar,
    Vector {
        length: ResolvedDimension,
    },
    Matrix {
        rows: ResolvedDimension,
        cols: ResolvedDimension,
    },
    Table {
        columns: Vec<(String, Dtype)>,
    },
}

/// A dimension after inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolvedDimension {
    /// Fixed at spec time.
    Fixed(usize),
    /// Will equal the number of rows in the attached input data.
    InputRows,
    /// Will equal the number of columns in the attached input data.
    InputCols,
    /// Resolved from a choice key to a concrete integer value.
    FromChoiceInt(String, i64),
    /// The dimension references a choice key that wasn't present or
    /// wasn't integer-valued. Rendered as a warning in the IDE.
    Unresolved(UnresolvedDimension),
}

/// Reason a dimension couldn't be resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnresolvedDimension {
    pub choice_key: String,
    pub reason: String,
}

/// A warning emitted during shape inference. Non-fatal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeWarning {
    /// The step references a recipe with no schema in the registry.
    UnknownRecipe(String),
    /// A `FromChoice` dimension references a choice key that isn't
    /// present in the step's effective bindings.
    MissingChoice {
        column: String,
        choice_key: String,
    },
    /// A `FromChoice` dimension pointed at a choice whose `using` value
    /// wasn't an integer type.
    NonIntegerChoice {
        column: String,
        choice_key: String,
        actual_type: String,
    },
}

/// Walk a pipeline and produce its resolved chain shape.
pub fn infer(pipeline: &Pipeline) -> ChainShape {
    let mut steps = Vec::with_capacity(pipeline.steps.len());
    for step in &pipeline.steps {
        steps.push(resolve_step(step));
    }
    ChainShape { steps }
}

fn resolve_step(step: &PipelineStep) -> ResolvedStep {
    let mut warnings = Vec::new();
    let schema = schema_for(&step.recipe);

    // Use step.outputs if the user declared them, otherwise fall back
    // to the schema's declared outputs.
    let source_columns: Vec<super::types::OutputColumn> = if !step.outputs.is_empty() {
        step.outputs.clone()
    } else if let Some(schema) = schema {
        schema
            .outputs
            .iter()
            .map(|o| o.to_output_column())
            .collect()
    } else {
        warnings.push(ShapeWarning::UnknownRecipe(format!("{:?}", step.recipe)));
        Vec::new()
    };

    let columns = source_columns
        .into_iter()
        .map(|col| resolve_column(step, col, &mut warnings))
        .collect();

    ResolvedStep {
        key: step.key.clone(),
        display_name: step.display_name.clone(),
        columns,
        warnings,
    }
}

fn resolve_column(
    step: &PipelineStep,
    col: super::types::OutputColumn,
    warnings: &mut Vec<ShapeWarning>,
) -> ResolvedColumn {
    let resolved_shape = match col.shape {
        OutputShape::Scalar => ResolvedShape::Scalar,
        OutputShape::Vector { length } => ResolvedShape::Vector {
            length: resolve_dimension(step, &col.semantic_name, length, warnings),
        },
        OutputShape::Matrix { rows, cols } => ResolvedShape::Matrix {
            rows: resolve_dimension(step, &col.semantic_name, rows, warnings),
            cols: resolve_dimension(step, &col.semantic_name, cols, warnings),
        },
        OutputShape::Table { columns } => ResolvedShape::Table { columns },
    };

    ResolvedColumn {
        semantic_name: col.semantic_name,
        description: col.description,
        shape: resolved_shape,
        dtype: col.dtype,
        has_v_column: col.has_v_column,
    }
}

fn resolve_dimension(
    step: &PipelineStep,
    column_name: &str,
    source: DimensionSource,
    warnings: &mut Vec<ShapeWarning>,
) -> ResolvedDimension {
    match source {
        DimensionSource::Fixed(n) => ResolvedDimension::Fixed(n),
        DimensionSource::InputRows => ResolvedDimension::InputRows,
        DimensionSource::InputCols => ResolvedDimension::InputCols,
        DimensionSource::FromChoice(key) => {
            let Some(binding) = effective_binding(&step.recipe, step, &key) else {
                warnings.push(ShapeWarning::MissingChoice {
                    column: column_name.to_string(),
                    choice_key: key.clone(),
                });
                return ResolvedDimension::Unresolved(UnresolvedDimension {
                    choice_key: key,
                    reason: "choice not declared by recipe schema".to_string(),
                });
            };

            // We look at the `using` dimension of the effective binding.
            // Autodiscover / sweep / superposition values can't be
            // resolved to a single integer at spec time — the IDE can
            // still display a "depends on sweep" placeholder for those
            // cases, but the resolved shape marks them as unresolved.
            let Some(ref value) = binding.using else {
                warnings.push(ShapeWarning::MissingChoice {
                    column: column_name.to_string(),
                    choice_key: key.clone(),
                });
                return ResolvedDimension::Unresolved(UnresolvedDimension {
                    choice_key: key,
                    reason: "no using() value — will be chosen by autodiscover/sweep at runtime"
                        .to_string(),
                });
            };

            match value {
                Value::Int(n) => ResolvedDimension::FromChoiceInt(key, *n),
                other => {
                    let type_name = value_type_name(other).to_string();
                    warnings.push(ShapeWarning::NonIntegerChoice {
                        column: column_name.to_string(),
                        choice_key: key.clone(),
                        actual_type: type_name.clone(),
                    });
                    ResolvedDimension::Unresolved(UnresolvedDimension {
                        choice_key: key,
                        reason: format!("choice value is {type_name}, not an integer"),
                    })
                }
            }
        }
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::String(_) => "string",
        Value::Method(_) => "method",
        Value::Vector(_) => "vector",
        Value::Matrix { .. } => "matrix",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::pipelines::schema::step_from_schema;
    use crate::recipes::pipelines::types::{Binding, PipelineStep, RecipeRef};

    fn exp_pipeline() -> Pipeline {
        let step = step_from_schema("e", RecipeRef::Recipe("exp".into()))
            .expect("exp schema exists");
        Pipeline::new("exp_test").with_step(step)
    }

    // ── Basic inference ───────────────────────────────────────────────────

    #[test]
    fn infer_exp_pipeline_has_one_step() {
        let pipeline = exp_pipeline();
        let chain = infer(&pipeline);
        assert_eq!(chain.steps.len(), 1);
        assert_eq!(chain.steps[0].key, "e");
    }

    #[test]
    fn infer_exp_step_has_one_column() {
        let pipeline = exp_pipeline();
        let chain = infer(&pipeline);
        let step = &chain.steps[0];
        assert_eq!(step.columns.len(), 1);
        assert_eq!(step.columns[0].semantic_name, "result");
    }

    #[test]
    fn infer_exp_output_shape_is_vector_input_rows() {
        let pipeline = exp_pipeline();
        let chain = infer(&pipeline);
        let col = &chain.steps[0].columns[0];
        match &col.shape {
            ResolvedShape::Vector {
                length: ResolvedDimension::InputRows,
            } => {}
            other => panic!("expected Vector<InputRows>, got {other:?}"),
        }
    }

    // ── Factor analysis: fromchoice resolution ───────────────────────────

    fn fa_pipeline_with_n_factors(n: i64) -> Pipeline {
        let mut step = step_from_schema("fa", RecipeRef::Recipe("factor_analysis".into()))
            .expect("factor_analysis schema exists");
        step.choices.insert(
            "n_factors".to_string(),
            Binding::using(Value::Int(n)),
        );
        Pipeline::new("fa").with_step(step)
    }

    #[test]
    fn fa_with_n_factors_resolves_matrix_cols() {
        let pipeline = fa_pipeline_with_n_factors(3);
        let chain = infer(&pipeline);
        let step = &chain.steps[0];

        // Find the loadings column.
        let loadings = step
            .columns
            .iter()
            .find(|c| c.semantic_name == "loadings")
            .expect("loadings column should exist");

        match &loadings.shape {
            ResolvedShape::Matrix {
                rows: ResolvedDimension::InputCols,
                cols: ResolvedDimension::FromChoiceInt(key, n),
            } => {
                assert_eq!(key, "n_factors");
                assert_eq!(*n, 3);
            }
            other => panic!("expected Matrix<InputCols, FromChoice(3)>, got {other:?}"),
        }
    }

    #[test]
    fn fa_without_n_factors_reports_unresolved() {
        // The recipe default for n_factors is autodiscover, not using —
        // so without explicit override, the FromChoice dimension is
        // unresolved (will be decided at runtime).
        let pipeline = Pipeline::new("fa").with_step(
            step_from_schema("fa", RecipeRef::Recipe("factor_analysis".into())).unwrap(),
        );
        let chain = infer(&pipeline);
        let step = &chain.steps[0];

        let loadings = step
            .columns
            .iter()
            .find(|c| c.semantic_name == "loadings")
            .unwrap();

        match &loadings.shape {
            ResolvedShape::Matrix {
                rows: ResolvedDimension::InputCols,
                cols: ResolvedDimension::Unresolved(_),
            } => {}
            other => panic!("expected Unresolved cols, got {other:?}"),
        }

        // A warning should have been emitted for the missing choice.
        assert!(
            !step.warnings.is_empty(),
            "expected at least one shape warning"
        );
    }

    // ── Unknown recipe ─────────────────────────────────────────────────────

    #[test]
    fn infer_unknown_recipe_emits_warning() {
        let pipeline = Pipeline::new("test").with_step(PipelineStep::new(
            "x",
            "X",
            RecipeRef::Recipe("no_such_recipe".into()),
        ));
        let chain = infer(&pipeline);
        assert_eq!(chain.steps.len(), 1);
        assert!(chain.steps[0].columns.is_empty());
        assert!(matches!(
            chain.steps[0].warnings.first(),
            Some(ShapeWarning::UnknownRecipe(_))
        ));
    }

    // ── Multi-step pipeline ────────────────────────────────────────────────

    #[test]
    fn infer_multi_step_preserves_order() {
        let pipeline = Pipeline::new("multi")
            .with_step(step_from_schema("a", RecipeRef::Recipe("exp".into())).unwrap())
            .with_step(
                step_from_schema("b", RecipeRef::Recipe("factor_analysis".into())).unwrap(),
            );
        let chain = infer(&pipeline);
        assert_eq!(chain.steps.len(), 2);
        assert_eq!(chain.steps[0].key, "a");
        assert_eq!(chain.steps[1].key, "b");
    }

    // ── JSON round-trip of the shape graph ─────────────────────────────────

    #[test]
    fn chain_shape_roundtrips_through_json() {
        let pipeline = fa_pipeline_with_n_factors(5);
        let chain = infer(&pipeline);

        let json = serde_json::to_string(&chain).unwrap();
        let decoded: ChainShape = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, chain);
    }

    #[test]
    fn fa_variance_explained_resolves_from_n_factors() {
        let pipeline = fa_pipeline_with_n_factors(4);
        let chain = infer(&pipeline);
        let step = &chain.steps[0];

        let var = step
            .columns
            .iter()
            .find(|c| c.semantic_name == "variance_explained")
            .unwrap();

        match &var.shape {
            ResolvedShape::Vector {
                length: ResolvedDimension::FromChoiceInt(_, 4),
            } => {}
            other => panic!("expected Vector<FromChoice(4)>, got {other:?}"),
        }
    }
}
