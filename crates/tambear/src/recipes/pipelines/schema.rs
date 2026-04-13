//! Recipe schemas — pipeline-layer view of what a recipe's parameters
//! and defaults look like.
//!
//! # Reader, not definer
//!
//! **The pipeline layer READS schemas; it does not DEFINE them.** Each
//! recipe is the authoritative source of its own parameter schema,
//! including the four-dimension defaults (using / autodiscover / sweep /
//! superposition) for every parameter. The pipeline layer queries that
//! schema to:
//!
//! - Prepopulate a fresh `PipelineStep` with the recipe's declared
//!   parameter set and defaults.
//! - Render `.tbs` syntax, formal notation, and IDE dropdowns with the
//!   effective binding shown as the initial selection.
//! - Produce a self-describing output for the bridge protocol so the
//!   tambear-ide can build its GUI without recipe-specific code.
//! - Materialize defaults into explicit `using()` values at save time so
//!   saved user pipelines don't silently inherit future tambear
//!   default changes.
//!
//! # Registration model
//!
//! For now, [`schema_for`] is a hand-maintained match statement that
//! dispatches a [`RecipeRef`] to a static [`RecipeSchema`] const. As
//! more recipes declare schemas, this file grows; eventually a
//! compile-time macro will auto-generate the match from per-recipe
//! `const RECIPE_SCHEMA: RecipeSchema = …;` declarations.
//!
//! New recipes are responsible for registering their own schema in this
//! file until the macro lands. That is a one-line addition.

use super::types::{
    Binding, Combiner, DataProbeRef, DimensionSource, Dtype, OutputColumn, OutputShape,
    RecipeRef, SuperpositionSpec, SweepSpec, Value,
};

/// Schema for one recipe: what parameters it takes, with per-parameter
/// defaults spanning all four hook dimensions, plus the output column
/// specification.
#[derive(Debug, Clone, PartialEq)]
pub struct RecipeSchema {
    /// Human-readable recipe name (e.g. `"correlation_matrix"`).
    pub name: &'static str,
    /// Short description shown in the IDE.
    pub description: &'static str,
    /// Parameter specs in display order.
    pub parameters: &'static [ParameterSpec],
    /// Output column spec — what the IDE should show for this recipe's
    /// outputs in the chain graph.
    pub outputs: &'static [OutputSpec],
}

/// Schema for one parameter of a recipe: its name, description, default
/// binding across all four hook dimensions, and optional value domain
/// for IDE dropdown population.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpec {
    /// Parameter key (matches [`ChoiceKey`] entries in `PipelineStep.choices`).
    pub key: &'static str,
    /// Display name shown in the IDE.
    pub display_name: &'static str,
    /// Prose description shown in hover/help text.
    pub description: &'static str,
    /// Default binding with any or all of the four hook dimensions
    /// populated. This is the "baked-in recipe default" — when a
    /// `PipelineStep` doesn't override a parameter, these values take
    /// effect.
    pub default: DefaultBinding,
    /// Optional value domain for IDE dropdown/slider/validator
    /// population. `None` means the recipe accepts free-form values.
    pub domain: Option<ValueDomain>,
}

/// Static-friendly version of [`Binding`] that can live in `const`
/// contexts. The pipeline layer converts it to a live [`Binding`] when
/// populating a step.
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultBinding {
    pub using: Option<DefaultValue>,
    pub autodiscover: Option<&'static str>,
    pub sweep: Option<&'static [DefaultValue]>,
    pub superposition: Option<DefaultSuperposition>,
}

/// Default superposition spec.
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultSuperposition {
    pub values: &'static [DefaultValue],
    pub combiner: DefaultCombiner,
}

/// Const-friendly version of [`Combiner`].
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultCombiner {
    KeepAll,
    WeightedSum,
    AgreementPartition { threshold_basis_points: u32 },
    Custom(&'static str),
}

/// Const-friendly version of [`Value`]. Only covers the cases a recipe
/// might reasonably use for a const default. Matrix/Vector defaults are
/// rare and can be constructed at runtime by the recipe if needed.
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(&'static str),
    Method(&'static str),
}

/// Output column spec (const-friendly version of [`OutputColumn`]).
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    pub semantic_name: &'static str,
    pub description: &'static str,
    pub shape: OutputShapeSpec,
    pub dtype: Dtype,
    pub has_v_column: bool,
}

/// Const-friendly version of [`OutputShape`].
#[derive(Debug, Clone, PartialEq)]
pub enum OutputShapeSpec {
    Scalar,
    Vector {
        length: DimensionSourceSpec,
    },
    Matrix {
        rows: DimensionSourceSpec,
        cols: DimensionSourceSpec,
    },
    Table {
        columns: &'static [(&'static str, Dtype)],
    },
}

/// Const-friendly version of [`DimensionSource`].
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionSourceSpec {
    Fixed(usize),
    InputRows,
    InputCols,
    FromChoice(&'static str),
}

/// Value domain hints for IDE dropdown / slider / validator population.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueDomain {
    /// Finite set of allowed values. IDE renders as dropdown.
    Enum(&'static [DefaultValue]),
    /// Numeric range (inclusive).
    Range { min: f64, max: f64 },
    /// Integer range (inclusive).
    IntRange { min: i64, max: i64 },
    /// Free-form string.
    FreeString,
}

// ── Conversion helpers ────────────────────────────────────────────────────

impl DefaultValue {
    /// Convert to a runtime [`Value`].
    pub fn to_value(&self) -> Value {
        match self {
            DefaultValue::Bool(b) => Value::Bool(*b),
            DefaultValue::Int(i) => Value::Int(*i),
            DefaultValue::Float(f) => Value::Float(*f),
            DefaultValue::String(s) => Value::String((*s).to_string()),
            DefaultValue::Method(m) => Value::Method((*m).to_string()),
        }
    }
}

impl DefaultCombiner {
    pub fn to_combiner(&self) -> Combiner {
        match self {
            DefaultCombiner::KeepAll => Combiner::KeepAll,
            DefaultCombiner::WeightedSum => Combiner::WeightedSum,
            DefaultCombiner::AgreementPartition {
                threshold_basis_points,
            } => Combiner::AgreementPartition {
                threshold_basis_points: *threshold_basis_points,
            },
            DefaultCombiner::Custom(tag) => Combiner::Custom((*tag).to_string()),
        }
    }
}

impl DefaultBinding {
    /// Realize this const default as a runtime [`Binding`].
    pub fn to_binding(&self) -> Binding {
        Binding {
            using: self.using.as_ref().map(|v| v.to_value()),
            autodiscover: self
                .autodiscover
                .map(|s| DataProbeRef(s.to_string())),
            sweep: self.sweep.map(|vs| SweepSpec {
                values: vs.iter().map(|v| v.to_value()).collect(),
            }),
            superposition: self.superposition.as_ref().map(|sp| SuperpositionSpec {
                values: sp.values.iter().map(|v| v.to_value()).collect(),
                combiner: sp.combiner.to_combiner(),
            }),
        }
    }
}

impl DimensionSourceSpec {
    pub fn to_dimension_source(&self) -> DimensionSource {
        match self {
            DimensionSourceSpec::Fixed(n) => DimensionSource::Fixed(*n),
            DimensionSourceSpec::InputRows => DimensionSource::InputRows,
            DimensionSourceSpec::InputCols => DimensionSource::InputCols,
            DimensionSourceSpec::FromChoice(k) => DimensionSource::FromChoice((*k).to_string()),
        }
    }
}

impl OutputShapeSpec {
    pub fn to_output_shape(&self) -> OutputShape {
        match self {
            OutputShapeSpec::Scalar => OutputShape::Scalar,
            OutputShapeSpec::Vector { length } => OutputShape::Vector {
                length: length.to_dimension_source(),
            },
            OutputShapeSpec::Matrix { rows, cols } => OutputShape::Matrix {
                rows: rows.to_dimension_source(),
                cols: cols.to_dimension_source(),
            },
            OutputShapeSpec::Table { columns } => OutputShape::Table {
                columns: columns
                    .iter()
                    .map(|(name, dt)| ((*name).to_string(), *dt))
                    .collect(),
            },
        }
    }
}

impl OutputSpec {
    pub fn to_output_column(&self) -> OutputColumn {
        OutputColumn {
            semantic_name: self.semantic_name.to_string(),
            description: self.description.to_string(),
            shape: self.shape.to_output_shape(),
            dtype: self.dtype,
            has_v_column: self.has_v_column,
        }
    }
}

// ── Registry / lookup ─────────────────────────────────────────────────────

/// Look up a recipe schema by reference.
///
/// This is the pipeline layer's READ API for recipe defaults. New
/// recipes register themselves by adding a match arm here that points
/// to their static `RECIPE_SCHEMA` const.
pub fn schema_for(recipe: &RecipeRef) -> Option<&'static RecipeSchema> {
    match recipe {
        RecipeRef::Expr(name) => match name.as_str() {
            "correlation_matrix" => Some(&CORRELATION_MATRIX_SCHEMA),
            _ => None,
        },
        RecipeRef::Recipe(name) => match name.as_str() {
            "factor_analysis" => Some(&FACTOR_ANALYSIS_SCHEMA),
            "varimax_rotation" => Some(&VARIMAX_ROTATION_SCHEMA),
            "exp" => Some(&EXP_SCHEMA),
            _ => None,
        },
        _ => None,
    }
}

// ── Placeholder recipe schemas ────────────────────────────────────────────
//
// These are stubs that demonstrate the shape. Once recipes declare their
// own schemas inline, each const below will move to its recipe's source
// file and be re-exported from here.

/// Placeholder schema for `exp` — single-parameter recipe for the
/// three-strategy lowering pattern. Used to validate that the pipeline
/// layer can read schemas end-to-end.
pub static EXP_SCHEMA: RecipeSchema = RecipeSchema {
    name: "exp",
    description: "Natural exponential e^x with three lowering strategies.",
    parameters: &[ParameterSpec {
        key: "precision",
        display_name: "Precision strategy",
        description:
            "strict: fast single-FMA path (<= 4 ulps). compensated: DD range reduction + compensated Horner (<= 2 ulps). correctly_rounded: full DD working precision (<= 1 ulp).",
        default: DefaultBinding {
            using: Some(DefaultValue::Method("compensated")),
            autodiscover: None,
            sweep: None,
            superposition: None,
        },
        domain: Some(ValueDomain::Enum(&[
            DefaultValue::Method("strict"),
            DefaultValue::Method("compensated"),
            DefaultValue::Method("correctly_rounded"),
        ])),
    }],
    outputs: &[OutputSpec {
        semantic_name: "result",
        description: "exp(x) evaluated element-wise across the input column.",
        shape: OutputShapeSpec::Vector {
            length: DimensionSourceSpec::InputRows,
        },
        dtype: Dtype::F64,
        has_v_column: false,
    }],
};

/// Placeholder schema for `correlation_matrix`. Will be filled in with
/// real defaults once the correlation recipe is wired to this module.
pub static CORRELATION_MATRIX_SCHEMA: RecipeSchema = RecipeSchema {
    name: "correlation_matrix",
    description: "Pearson/Spearman/Kendall correlation matrix with auto-detection.",
    parameters: &[ParameterSpec {
        key: "method",
        display_name: "Correlation method",
        description:
            "pearson: linear association assuming normality. spearman: rank-based, robust to outliers and non-linearity. kendall: tau-b, strongest small-sample behavior.",
        default: DefaultBinding {
            using: Some(DefaultValue::Method("pearson")),
            autodiscover: Some("normality_probe"),
            sweep: None,
            superposition: None,
        },
        domain: Some(ValueDomain::Enum(&[
            DefaultValue::Method("pearson"),
            DefaultValue::Method("spearman"),
            DefaultValue::Method("kendall"),
        ])),
    }],
    outputs: &[OutputSpec {
        semantic_name: "correlation_matrix",
        description: "Pairwise correlation coefficients across input columns.",
        shape: OutputShapeSpec::Matrix {
            rows: DimensionSourceSpec::InputCols,
            cols: DimensionSourceSpec::InputCols,
        },
        dtype: Dtype::F64,
        has_v_column: false,
    }],
};

/// Placeholder schema for `factor_analysis`. Matches the existing
/// `crates/tambear/src/factor_analysis.rs` surface at a high level.
pub static FACTOR_ANALYSIS_SCHEMA: RecipeSchema = RecipeSchema {
    name: "factor_analysis",
    description:
        "Exploratory factor analysis: extracts latent factors from a correlation matrix, \
         optionally applying a rotation to improve interpretability.",
    parameters: &[
        ParameterSpec {
            key: "n_factors",
            display_name: "Number of factors",
            description:
                "How many latent factors to extract. If omitted, tambear auto-selects via \
                 parallel analysis (Horn) or Kaiser criterion depending on sample size.",
            default: DefaultBinding {
                using: None,
                autodiscover: Some("parallel_analysis_or_kaiser"),
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::IntRange { min: 1, max: 20 }),
        },
        ParameterSpec {
            key: "extraction",
            display_name: "Extraction method",
            description:
                "pa: principal axis factoring. ml: maximum likelihood. minres: minimum residual. \
                 Default auto-selects based on factorability diagnostics (KMO, Bartlett).",
            default: DefaultBinding {
                using: Some(DefaultValue::Method("pa")),
                autodiscover: Some("factorability_probe"),
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::Enum(&[
                DefaultValue::Method("pa"),
                DefaultValue::Method("ml"),
                DefaultValue::Method("minres"),
            ])),
        },
        ParameterSpec {
            key: "rotation",
            display_name: "Rotation method",
            description:
                "varimax: orthogonal, simple structure. promax: oblique, allows correlated factors. \
                 oblimin: oblique, slightly different criterion. none: no rotation.",
            default: DefaultBinding {
                using: Some(DefaultValue::Method("varimax")),
                autodiscover: None,
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::Enum(&[
                DefaultValue::Method("none"),
                DefaultValue::Method("varimax"),
                DefaultValue::Method("promax"),
                DefaultValue::Method("oblimin"),
            ])),
        },
    ],
    outputs: &[
        OutputSpec {
            semantic_name: "loadings",
            description: "Factor loadings matrix (n_variables × n_factors).",
            shape: OutputShapeSpec::Matrix {
                rows: DimensionSourceSpec::InputCols,
                cols: DimensionSourceSpec::FromChoice("n_factors"),
            },
            dtype: Dtype::F64,
            has_v_column: true,
        },
        OutputSpec {
            semantic_name: "communalities",
            description:
                "Communality per variable — how much variance is explained by the extracted factors.",
            shape: OutputShapeSpec::Vector {
                length: DimensionSourceSpec::InputCols,
            },
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "eigenvalues",
            description: "Eigenvalues of the reduced correlation matrix.",
            shape: OutputShapeSpec::Vector {
                length: DimensionSourceSpec::FromChoice("n_factors"),
            },
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "variance_explained",
            description: "Proportion of total variance explained by each factor.",
            shape: OutputShapeSpec::Vector {
                length: DimensionSourceSpec::FromChoice("n_factors"),
            },
            dtype: Dtype::F64,
            has_v_column: false,
        },
    ],
};

/// Placeholder schema for `varimax_rotation`. Stub until the rotation
/// recipe is wired.
pub static VARIMAX_ROTATION_SCHEMA: RecipeSchema = RecipeSchema {
    name: "varimax_rotation",
    description: "Kaiser's varimax orthogonal rotation of a loadings matrix.",
    parameters: &[
        ParameterSpec {
            key: "max_iterations",
            display_name: "Maximum iterations",
            description: "Iteration cap for the rotation convergence loop.",
            default: DefaultBinding {
                using: Some(DefaultValue::Int(1000)),
                autodiscover: None,
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::IntRange { min: 10, max: 100000 }),
        },
        ParameterSpec {
            key: "tolerance",
            display_name: "Convergence tolerance",
            description: "Stop when change in rotation angle is below this threshold.",
            default: DefaultBinding {
                using: Some(DefaultValue::Float(1e-6)),
                autodiscover: None,
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::Range { min: 1e-12, max: 1e-2 }),
        },
    ],
    outputs: &[OutputSpec {
        semantic_name: "rotated_loadings",
        description: "Varimax-rotated loadings matrix.",
        shape: OutputShapeSpec::Matrix {
            rows: DimensionSourceSpec::InputCols,
            cols: DimensionSourceSpec::InputCols,
        },
        dtype: Dtype::F64,
        has_v_column: false,
    }],
};

// ── Pipeline-level prepopulation helper ───────────────────────────────────

/// Build a fresh [`PipelineStep`] from a recipe's schema, with all
/// parameters bound to `Binding::inherit_defaults()` (meaning the IDE
/// will display the recipe-declared defaults as the initial state).
///
/// The caller can then override any parameter via `step.with_choice(...)`
/// to layer user choices on top.
pub fn step_from_schema(
    key: impl Into<String>,
    recipe: RecipeRef,
) -> Option<super::types::PipelineStep> {
    let schema = schema_for(&recipe)?;
    let mut step = super::types::PipelineStep::new(key, schema.name.to_string(), recipe);
    for param in schema.parameters {
        step.choices
            .insert(param.key.to_string(), Binding::inherit_defaults());
    }
    for out in schema.outputs {
        step.outputs.push(out.to_output_column());
    }
    Some(step)
}

/// Resolve a step's effective binding for a given choice key, falling
/// back to recipe-declared defaults for any dimension the step doesn't
/// override.
///
/// The returned [`Binding`] has every dimension filled in (either by the
/// step override or by the recipe default), so saved/serialized forms
/// can materialize it directly without implicit inheritance.
pub fn effective_binding(
    recipe: &RecipeRef,
    step: &super::types::PipelineStep,
    choice_key: &str,
) -> Option<Binding> {
    let schema = schema_for(recipe)?;
    let param = schema.parameters.iter().find(|p| p.key == choice_key)?;
    let default_binding = param.default.to_binding();

    let step_binding = step.choices.get(choice_key).cloned().unwrap_or_default();

    // Compose: step overrides take precedence, but any dimension the
    // step leaves as `None` falls back to the recipe default.
    Some(Binding {
        using: step_binding.using.or(default_binding.using),
        autodiscover: step_binding.autodiscover.or(default_binding.autodiscover),
        sweep: step_binding.sweep.or(default_binding.sweep),
        superposition: step_binding
            .superposition
            .or(default_binding.superposition),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::pipelines::types::{PipelineStep, Value};

    #[test]
    fn schema_lookup_finds_exp() {
        let schema = schema_for(&RecipeRef::Recipe("exp".into()));
        assert!(schema.is_some());
        let schema = schema.unwrap();
        assert_eq!(schema.name, "exp");
        assert_eq!(schema.parameters.len(), 1);
        assert_eq!(schema.parameters[0].key, "precision");
    }

    #[test]
    fn schema_lookup_finds_factor_analysis() {
        let schema = schema_for(&RecipeRef::Recipe("factor_analysis".into()));
        assert!(schema.is_some());
        let schema = schema.unwrap();
        assert_eq!(schema.parameters.len(), 3);
        let keys: Vec<&str> = schema.parameters.iter().map(|p| p.key).collect();
        assert!(keys.contains(&"n_factors"));
        assert!(keys.contains(&"extraction"));
        assert!(keys.contains(&"rotation"));
    }

    #[test]
    fn schema_lookup_returns_none_for_unknown() {
        let schema = schema_for(&RecipeRef::Recipe("nonexistent".into()));
        assert!(schema.is_none());
    }

    #[test]
    fn step_from_schema_populates_choices() {
        let step = step_from_schema("fa", RecipeRef::Recipe("factor_analysis".into()))
            .expect("factor_analysis schema should exist");
        assert_eq!(step.key, "fa");
        assert_eq!(step.choices.len(), 3);
        assert!(step.choices.contains_key("n_factors"));
        assert!(step.choices.contains_key("extraction"));
        assert!(step.choices.contains_key("rotation"));
        // Every choice starts as inherit_defaults.
        for (_, binding) in &step.choices {
            assert!(binding.is_fully_default());
        }
    }

    #[test]
    fn step_from_schema_populates_outputs() {
        let step = step_from_schema("fa", RecipeRef::Recipe("factor_analysis".into()))
            .expect("factor_analysis schema should exist");
        assert_eq!(step.outputs.len(), 4);
        let names: Vec<&str> = step.outputs.iter().map(|o| o.semantic_name.as_str()).collect();
        assert!(names.contains(&"loadings"));
        assert!(names.contains(&"communalities"));
        assert!(names.contains(&"eigenvalues"));
        assert!(names.contains(&"variance_explained"));
    }

    #[test]
    fn effective_binding_falls_back_to_default_when_step_empty() {
        let recipe = RecipeRef::Recipe("factor_analysis".into());
        let step = step_from_schema("fa", recipe.clone()).unwrap();

        let effective = effective_binding(&recipe, &step, "extraction")
            .expect("extraction should resolve");
        // The default for extraction is using=Method("pa"), autodiscover=factorability_probe.
        assert_eq!(effective.using, Some(Value::Method("pa".into())));
        assert!(effective.autodiscover.is_some());
        assert_eq!(
            effective.autodiscover.as_ref().unwrap().0,
            "factorability_probe"
        );
    }

    #[test]
    fn effective_binding_honors_step_override_on_using_only() {
        // User forces extraction=ml but leaves other dimensions alone.
        // The recipe-declared autodiscover probe should still be present
        // in the effective binding so advice can surface it.
        let recipe = RecipeRef::Recipe("factor_analysis".into());
        let mut step = step_from_schema("fa", recipe.clone()).unwrap();
        step.choices.insert(
            "extraction".to_string(),
            Binding::using(Value::Method("ml".into())),
        );

        let effective = effective_binding(&recipe, &step, "extraction")
            .expect("extraction should resolve");
        assert_eq!(effective.using, Some(Value::Method("ml".into())));
        assert!(effective.autodiscover.is_some(), "advice probe lost on override");
    }

    #[test]
    fn effective_binding_composes_autodiscover_from_step() {
        // Opposite case: step sets only autodiscover, default provides
        // the using value. The effective binding has both dimensions.
        let recipe = RecipeRef::Recipe("factor_analysis".into());
        let mut step = step_from_schema("fa", recipe.clone()).unwrap();
        step.choices.insert(
            "extraction".to_string(),
            Binding::autodiscover(DataProbeRef("custom_probe".into())),
        );

        let effective = effective_binding(&recipe, &step, "extraction")
            .expect("extraction should resolve");
        // Step-level autodiscover wins.
        assert_eq!(
            effective.autodiscover.as_ref().unwrap().0,
            "custom_probe"
        );
        // But using still comes from recipe default.
        assert_eq!(effective.using, Some(Value::Method("pa".into())));
    }

    #[test]
    fn exp_schema_default_is_compensated() {
        let schema = schema_for(&RecipeRef::Recipe("exp".into())).unwrap();
        let precision = &schema.parameters[0];
        assert_eq!(precision.key, "precision");
        let binding = precision.default.to_binding();
        assert_eq!(binding.using, Some(Value::Method("compensated".into())));
    }

    #[test]
    fn correlation_matrix_has_autodiscover_default() {
        let schema = schema_for(&RecipeRef::Expr("correlation_matrix".into())).unwrap();
        let method = &schema.parameters[0];
        let binding = method.default.to_binding();
        // Default has BOTH using (pearson) AND autodiscover (normality probe) —
        // the "compose, don't pick" pattern from the user's design.
        assert_eq!(binding.using, Some(Value::Method("pearson".into())));
        assert!(binding.autodiscover.is_some());
        assert_eq!(
            binding.autodiscover.as_ref().unwrap().0,
            "normality_probe"
        );
    }

    #[test]
    fn value_domain_carries_enum_choices() {
        let schema = schema_for(&RecipeRef::Recipe("exp".into())).unwrap();
        let precision = &schema.parameters[0];
        match &precision.domain {
            Some(ValueDomain::Enum(choices)) => {
                assert_eq!(choices.len(), 3);
            }
            _ => panic!("expected enum domain"),
        }
    }
}
