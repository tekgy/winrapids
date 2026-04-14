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

use super::toml_schema::{self, OwnedRecipeSchema};
use super::types::{
    Binding, Combiner, DataProbeRef, DimensionSource, Dtype, OutputColumn, OutputShape,
    RecipeRef, SuperpositionSpec, SweepSpec, Value,
};
use std::sync::OnceLock;

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
            "kmo" => Some(&KMO_SCHEMA),
            "bartlett_sphericity" => Some(&BARTLETT_SPHERICITY_SCHEMA),
            "factor_scores" => Some(&FACTOR_SCORES_SCHEMA),
            "model_fit_diagnostics" => Some(&MODEL_FIT_DIAGNOSTICS_SCHEMA),
            _ => None,
        },
        RecipeRef::Recipe(name) => match name.as_str() {
            "factor_analysis" => Some(&FACTOR_ANALYSIS_SCHEMA),
            "varimax_rotation" => Some(&VARIMAX_ROTATION_SCHEMA),
            // First recipe to be migrated to .spec.toml. The toml file
            // at `src/recipes/libm/exp.spec.toml` is the single source
            // of truth; both the Rust tests and the tambear-ide read it.
            "exp" => Some(exp_schema_from_toml()),
            _ => None,
        },
        _ => None,
    }
}

// ── TOML-loaded schemas ───────────────────────────────────────────────────
//
// Schemas declared in `.spec.toml` files next to their recipe's `.rs`
// implementation. These are loaded on first access, parsed once, and
// cached via `OnceLock`. The parsed `OwnedRecipeSchema` is converted
// to a static `RecipeSchema` by leaking its strings, which is fine
// because each schema loads exactly once per process lifetime.
//
// New recipes should follow this pattern rather than adding hardcoded
// consts below.

static EXP_SCHEMA_CELL: OnceLock<&'static RecipeSchema> = OnceLock::new();

fn exp_schema_from_toml() -> &'static RecipeSchema {
    EXP_SCHEMA_CELL.get_or_init(|| {
        let toml_str = include_str!("../libm/exp.spec.toml");
        let owned = toml_schema::parse_spec_toml(toml_str)
            .expect("exp.spec.toml must parse — this is a build-time invariant");
        Box::leak(Box::new(leak_into_static(&owned)))
    })
}

/// Convert an [`OwnedRecipeSchema`] to a [`RecipeSchema`] by leaking
/// every owned String/Vec as `&'static`. The caller is responsible for
/// ensuring this happens at most once per schema (via `OnceLock`).
fn leak_into_static(owned: &OwnedRecipeSchema) -> RecipeSchema {
    RecipeSchema {
        name: Box::leak(owned.name.clone().into_boxed_str()),
        description: Box::leak(owned.description.clone().into_boxed_str()),
        parameters: Box::leak(
            owned
                .parameters
                .iter()
                .map(leak_parameter)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        outputs: Box::leak(
            owned
                .outputs
                .iter()
                .map(leak_output)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
    }
}

fn leak_parameter(p: &toml_schema::OwnedParameterSpec) -> ParameterSpec {
    ParameterSpec {
        key: Box::leak(p.key.clone().into_boxed_str()),
        display_name: Box::leak(p.display_name.clone().into_boxed_str()),
        description: Box::leak(p.description.clone().into_boxed_str()),
        default: leak_default_binding(&p.default),
        domain: p.domain.as_ref().map(leak_domain),
    }
}

fn leak_default_binding(b: &toml_schema::OwnedDefaultBinding) -> DefaultBinding {
    DefaultBinding {
        using: b.using.as_ref().map(leak_default_value),
        autodiscover: b
            .autodiscover
            .as_ref()
            .map(|s| &*Box::leak(s.clone().into_boxed_str())),
        sweep: b.sweep.as_ref().map(|vs| {
            let leaked: Vec<DefaultValue> = vs.iter().map(leak_default_value).collect();
            &*Box::leak(leaked.into_boxed_slice())
        }),
        superposition: b.superposition.as_ref().map(|sp| DefaultSuperposition {
            values: {
                let leaked: Vec<DefaultValue> =
                    sp.values.iter().map(leak_default_value).collect();
                Box::leak(leaked.into_boxed_slice())
            },
            combiner: leak_combiner(&sp.combiner),
        }),
    }
}

fn leak_default_value(v: &toml_schema::OwnedDefaultValue) -> DefaultValue {
    match v {
        toml_schema::OwnedDefaultValue::Bool(b) => DefaultValue::Bool(*b),
        toml_schema::OwnedDefaultValue::Int(i) => DefaultValue::Int(*i),
        toml_schema::OwnedDefaultValue::Float(f) => DefaultValue::Float(*f),
        toml_schema::OwnedDefaultValue::String(s) => {
            DefaultValue::String(Box::leak(s.clone().into_boxed_str()))
        }
        toml_schema::OwnedDefaultValue::Method(m) => {
            DefaultValue::Method(Box::leak(m.clone().into_boxed_str()))
        }
    }
}

fn leak_combiner(c: &toml_schema::OwnedDefaultCombiner) -> DefaultCombiner {
    match c {
        toml_schema::OwnedDefaultCombiner::KeepAll => DefaultCombiner::KeepAll,
        toml_schema::OwnedDefaultCombiner::WeightedSum => DefaultCombiner::WeightedSum,
        toml_schema::OwnedDefaultCombiner::AgreementPartition {
            threshold_basis_points,
        } => DefaultCombiner::AgreementPartition {
            threshold_basis_points: *threshold_basis_points,
        },
        toml_schema::OwnedDefaultCombiner::Custom(tag) => {
            DefaultCombiner::Custom(Box::leak(tag.clone().into_boxed_str()))
        }
    }
}

fn leak_domain(d: &toml_schema::OwnedValueDomain) -> ValueDomain {
    match d {
        toml_schema::OwnedValueDomain::Enum(vs) => ValueDomain::Enum({
            let leaked: Vec<DefaultValue> = vs.iter().map(leak_default_value).collect();
            Box::leak(leaked.into_boxed_slice())
        }),
        toml_schema::OwnedValueDomain::Range { min, max } => ValueDomain::Range {
            min: *min,
            max: *max,
        },
        toml_schema::OwnedValueDomain::IntRange { min, max } => ValueDomain::IntRange {
            min: *min,
            max: *max,
        },
        toml_schema::OwnedValueDomain::FreeString => ValueDomain::FreeString,
    }
}

fn leak_output(o: &toml_schema::OwnedOutputSpec) -> OutputSpec {
    OutputSpec {
        semantic_name: Box::leak(o.semantic_name.clone().into_boxed_str()),
        description: Box::leak(o.description.clone().into_boxed_str()),
        shape: leak_output_shape(&o.shape),
        dtype: o.dtype,
        has_v_column: o.has_v_column,
    }
}

fn leak_output_shape(s: &toml_schema::OwnedOutputShape) -> OutputShapeSpec {
    match s {
        toml_schema::OwnedOutputShape::Scalar => OutputShapeSpec::Scalar,
        toml_schema::OwnedOutputShape::Vector { length } => OutputShapeSpec::Vector {
            length: leak_dimension(length),
        },
        toml_schema::OwnedOutputShape::Matrix { rows, cols } => OutputShapeSpec::Matrix {
            rows: leak_dimension(rows),
            cols: leak_dimension(cols),
        },
        toml_schema::OwnedOutputShape::Table { columns } => OutputShapeSpec::Table {
            columns: {
                let leaked: Vec<(&'static str, Dtype)> = columns
                    .iter()
                    .map(|(name, dt)| (&*Box::leak(name.clone().into_boxed_str()), *dt))
                    .collect();
                Box::leak(leaked.into_boxed_slice())
            },
        },
    }
}

fn leak_dimension(d: &toml_schema::OwnedDimensionSource) -> DimensionSourceSpec {
    match d {
        toml_schema::OwnedDimensionSource::Fixed(n) => DimensionSourceSpec::Fixed(*n),
        toml_schema::OwnedDimensionSource::InputRows => DimensionSourceSpec::InputRows,
        toml_schema::OwnedDimensionSource::InputCols => DimensionSourceSpec::InputCols,
        toml_schema::OwnedDimensionSource::FromChoice(k) => {
            DimensionSourceSpec::FromChoice(Box::leak(k.clone().into_boxed_str()))
        }
    }
}

// ── Placeholder recipe schemas ────────────────────────────────────────────
//
// These are stubs that demonstrate the shape. Once recipes declare their
// own schemas inline, each const below will move to its recipe's source
// file and be re-exported from here.

// EXP_SCHEMA migrated to src/recipes/libm/exp.spec.toml.
// Access via `schema_for(&RecipeRef::Recipe("exp".to_string()))` which
// routes through `exp_schema_from_toml()` above. First recipe piloted
// on the .spec.toml pattern; all future recipes should follow suit.

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
                "minres: minimum residual (IDE default — balances speed and robustness). \
                 ml: maximum likelihood. pa: principal axis factoring. \
                 Default auto-selects based on factorability diagnostics (KMO, Bartlett).",
            default: DefaultBinding {
                using: Some(DefaultValue::Method("minres")),
                autodiscover: Some("factorability_probe"),
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::Enum(&[
                DefaultValue::Method("minres"),
                DefaultValue::Method("ml"),
                DefaultValue::Method("pa"),
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
        ParameterSpec {
            key: "missing",
            display_name: "Missing data handling",
            description:
                "listwise: drop any row with NaN (the IDE default — conservative). \
                 pairwise: use all available pairs per covariance entry (larger effective N but \
                 can produce non-positive-definite correlation matrices).",
            default: DefaultBinding {
                using: Some(DefaultValue::Method("listwise")),
                autodiscover: None,
                sweep: None,
                superposition: None,
            },
            domain: Some(ValueDomain::Enum(&[
                DefaultValue::Method("listwise"),
                DefaultValue::Method("pairwise"),
            ])),
        },
        ParameterSpec {
            key: "fm_se",
            display_name: "Compute standard errors",
            description:
                "If true, compute asymptotic standard errors for factor loadings. Adds cost but \
                 lets the IDE overlay uncertainty on the loadings table.",
            default: DefaultBinding {
                using: Some(DefaultValue::Bool(false)),
                autodiscover: None,
                sweep: None,
                superposition: None,
            },
            domain: None,
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

// ── Diagnostic schemas used by the EFA pipeline ───────────────────────────

/// Schema for `kmo` — Kaiser-Meyer-Olkin measure of sampling adequacy.
///
/// KMO is a purely diagnostic recipe with no tunable parameters in its
/// canonical form. It takes a correlation matrix and produces an overall
/// scalar plus a per-variable MSA vector. The output is consumed by
/// downstream steps as an advisory signal on whether the data is
/// "factorable" at all.
pub static KMO_SCHEMA: RecipeSchema = RecipeSchema {
    name: "kmo",
    description:
        "Kaiser-Meyer-Olkin measure of sampling adequacy. Overall score and per-variable MSA. \
         Rule of thumb: > 0.9 marvellous, 0.8 meritorious, 0.7 middling, 0.6 mediocre, \
         < 0.5 unacceptable for factor analysis.",
    parameters: &[],
    outputs: &[
        OutputSpec {
            semantic_name: "kmo_overall",
            description: "Overall KMO sampling adequacy score in [0, 1].",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "kmo_per_variable",
            description: "Per-variable measure of sampling adequacy (MSA).",
            shape: OutputShapeSpec::Vector {
                length: DimensionSourceSpec::InputCols,
            },
            dtype: Dtype::F64,
            has_v_column: false,
        },
    ],
};

/// Schema for `bartlett_sphericity` — Bartlett's test of sphericity.
///
/// The canonical form has one tunable: the significance level `alpha` used
/// to flag the test as rejecting or not rejecting the identity-correlation
/// null hypothesis. The chi-squared statistic and p-value are emitted
/// regardless; `alpha` is advisory.
pub static BARTLETT_SPHERICITY_SCHEMA: RecipeSchema = RecipeSchema {
    name: "bartlett_sphericity",
    description:
        "Bartlett's test of sphericity: H0 = correlation matrix is identity. Emits chi-squared \
         statistic, degrees of freedom, and p-value. Rejecting H0 (low p-value) is a necessary \
         condition for factor analysis to be meaningful.",
    parameters: &[ParameterSpec {
        key: "alpha",
        display_name: "Significance level",
        description:
            "Threshold for advisory 'rejects H0' flag. The underlying chi-squared statistic and \
             p-value are always returned; alpha only drives the rendered 'reject/fail to reject' \
             label in the IDE.",
        default: DefaultBinding {
            using: Some(DefaultValue::Float(0.05)),
            autodiscover: None,
            sweep: None,
            superposition: None,
        },
        domain: Some(ValueDomain::Range {
            min: 1e-6,
            max: 0.25,
        }),
    }],
    outputs: &[
        OutputSpec {
            semantic_name: "chi_squared",
            description: "Bartlett chi-squared test statistic.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "degrees_of_freedom",
            description: "Degrees of freedom = p(p-1)/2 where p = number of variables.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::I64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "p_value",
            description: "Right-tail p-value under the chi-squared null.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
    ],
};

/// Schema for `factor_scores` — per-observation factor scores.
///
/// Turns the loadings matrix from a `factor_analysis` step back into
/// per-observation quantities: one score per observation per extracted
/// factor. Two common methods are offered — regression (Thurstone) and
/// Bartlett — plus Anderson-Rubin for a decorrelated variant.
pub static FACTOR_SCORES_SCHEMA: RecipeSchema = RecipeSchema {
    name: "factor_scores",
    description:
        "Per-observation factor scores computed from a fitted loadings matrix. Regression \
         (Thurstone) is the standard default; Bartlett gives unbiased estimates at the cost of \
         higher variance; Anderson-Rubin produces orthogonal standardized scores.",
    parameters: &[ParameterSpec {
        key: "method",
        display_name: "Scoring method",
        description:
            "regression: Thurstone's weighted-least-squares estimator, correlated with true \
             scores but biased. bartlett: maximum-likelihood unbiased estimator. \
             anderson_rubin: orthogonal, unit-variance standardized scores.",
        default: DefaultBinding {
            using: Some(DefaultValue::Method("regression")),
            autodiscover: None,
            sweep: None,
            superposition: None,
        },
        domain: Some(ValueDomain::Enum(&[
            DefaultValue::Method("regression"),
            DefaultValue::Method("bartlett"),
            DefaultValue::Method("anderson_rubin"),
        ])),
    }],
    outputs: &[OutputSpec {
        semantic_name: "scores",
        description: "Factor scores: (n_observations × n_factors) matrix.",
        shape: OutputShapeSpec::Matrix {
            rows: DimensionSourceSpec::InputRows,
            cols: DimensionSourceSpec::FromChoice("n_factors"),
        },
        dtype: Dtype::F64,
        has_v_column: true,
    }],
};

/// Schema for `model_fit_diagnostics` — RMSEA/TLI/CFI plus residual norm.
///
/// Takes the fitted loadings and the original correlation matrix, computes
/// the implied correlation matrix, and reports standard structural-equation
/// fit indices. This is the final sanity-check step in the EFA pipeline.
pub static MODEL_FIT_DIAGNOSTICS_SCHEMA: RecipeSchema = RecipeSchema {
    name: "model_fit_diagnostics",
    description:
        "Standard fit indices for an extracted factor model: RMSEA (root mean square error of \
         approximation), TLI (Tucker-Lewis index), CFI (comparative fit index), and the Frobenius \
         norm of the residual correlation matrix.",
    parameters: &[ParameterSpec {
        key: "rmsea_threshold",
        display_name: "RMSEA acceptance threshold",
        description:
            "Advisory cutoff for rendering 'acceptable fit' in the IDE. The underlying RMSEA \
             value is always returned.",
        default: DefaultBinding {
            using: Some(DefaultValue::Float(0.06)),
            autodiscover: None,
            sweep: None,
            superposition: None,
        },
        domain: Some(ValueDomain::Range {
            min: 0.01,
            max: 0.20,
        }),
    }],
    outputs: &[
        OutputSpec {
            semantic_name: "rmsea",
            description: "Root mean square error of approximation. Lower is better; < 0.06 is conventionally good.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "tli",
            description: "Tucker-Lewis non-normed fit index. Higher is better; > 0.95 is good.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "cfi",
            description: "Comparative fit index. Higher is better; > 0.95 is good.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
        OutputSpec {
            semantic_name: "residual_norm",
            description:
                "Frobenius norm of (observed - implied) correlation matrix. Zero = perfect fit.",
            shape: OutputShapeSpec::Scalar,
            dtype: Dtype::F64,
            has_v_column: false,
        },
    ],
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
        assert_eq!(schema.parameters.len(), 5);
        let keys: Vec<&str> = schema.parameters.iter().map(|p| p.key).collect();
        assert!(keys.contains(&"n_factors"));
        assert!(keys.contains(&"extraction"));
        assert!(keys.contains(&"rotation"));
        assert!(keys.contains(&"missing"));
        assert!(keys.contains(&"fm_se"));
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
        assert_eq!(step.choices.len(), 5);
        assert!(step.choices.contains_key("n_factors"));
        assert!(step.choices.contains_key("extraction"));
        assert!(step.choices.contains_key("rotation"));
        assert!(step.choices.contains_key("missing"));
        assert!(step.choices.contains_key("fm_se"));
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
        // The default for extraction is using=Method("minres"), autodiscover=factorability_probe.
        assert_eq!(effective.using, Some(Value::Method("minres".into())));
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
        assert_eq!(effective.using, Some(Value::Method("minres".into())));
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
