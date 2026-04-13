//! Core types for the pipeline layer.
//!
//! The vocabulary:
//!
//! - [`Pipeline`] — ordered sequence of [`PipelineStep`]s.
//! - [`PipelineStep`] — one recipe invocation with choice bindings and an
//!   output column spec.
//! - [`Binding`] — how one parameter/choice at a step is filled in. Has
//!   four composable dimensions: `using`, `autodiscover`, `sweep`,
//!   `superposition`. Any subset can be present simultaneously — they
//!   are NOT mutually exclusive.
//! - [`OutputColumn`] — the pre-execution shape info the IDE reads to
//!   render the chain graph before the pipeline runs.
//!
//! # The composable bindings model
//!
//! Each recipe declares, for each of its parameters, a *default binding*
//! — which of the four hook dimensions are active by default and with
//! what values. Example: a `correlation()` recipe might declare:
//!
//! ```text
//! parameter: method
//!   default.using           = Some(Pearson)
//!   default.autodiscover    = Some(NormalityProbe)  // advisory only
//!   default.sweep           = None
//!   default.superposition   = None
//! ```
//!
//! When a pipeline step invokes this recipe and doesn't touch the
//! `method` parameter, all four defaults apply: `using=Pearson` sets the
//! effective value, `autodiscover=NormalityProbe` still runs and can
//! emit advice ("your data is not normal; Spearman would be safer"). If
//! the user writes `.using(method=Spearman)` in their `.tbs` script, the
//! `using` dimension is overridden but the `autodiscover` probe still
//! runs for advice, so the user sees BOTH choices side by side — this
//! is the Layer 2 override-transparency pattern from CLAUDE.md.
//!
//! The four dimensions are independent and each has its own per-recipe
//! default. The saved/community form of a pipeline freezes every
//! dimension into an explicit value, so future tambear default changes
//! never silently alter a user's work.
//!
//! All types here are pure data with no execution semantics. The
//! execution/orchestration crate reads them and runs them; this crate
//! only describes what to run.

use std::collections::BTreeMap;

/// An ordered pipeline of steps.
#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
    /// Human-readable name (e.g. `"exploratory_factor_analysis"`).
    pub name: String,
    /// Optional longer description shown alongside the academic writeup.
    pub description: String,
    /// Ordered list of steps. Execution order is vector order.
    pub steps: Vec<PipelineStep>,
}

/// One step of a pipeline: a recipe invocation plus parameter bindings
/// plus output column spec.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStep {
    /// Short stable identifier, unique within the pipeline. The IDE uses
    /// this as the invariant key when reordering steps; the A####/B####
    /// display labels are derived from step index at render time.
    pub key: String,
    /// Display-friendly name shown in the IDE.
    pub display_name: String,
    /// Which recipe / expr / primitive / atom this step invokes.
    pub recipe: RecipeRef,
    /// Per-parameter bindings. Absence from this map means "use all four
    /// recipe-declared defaults for this parameter." Presence with a
    /// partial [`Binding`] (some dimensions set, others `None`) means
    /// "override the set dimensions, use defaults for the rest."
    pub choices: BTreeMap<ChoiceKey, Binding>,
    /// What columns this step produces, known at spec time. The IDE
    /// reads this to render the chain graph.
    pub outputs: Vec<OutputColumn>,
}

/// A reference to a callable in the tambear catalog.
///
/// Pipelines can invoke at any layer — atoms, primitives, exprs,
/// recipes. That's how the "force anything anywhere" invariant is
/// preserved: a pipeline step can target a primitive-level choice, not
/// only a recipe-level one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecipeRef {
    Atom(String),
    Primitive(String),
    Expr(String),
    Recipe(String),
}

/// A choice key is the name of a parameter/tunable at a step.
pub type ChoiceKey = String;

/// A per-parameter binding.
///
/// The four fields are **composable dimensions**, not mutually-exclusive
/// variants. Any subset can be present:
///
/// - `using` — user-forced value for this parameter.
/// - `autodiscover` — probe that runs at execution time. Can SET the
///   value (if `using` is `None`) or just emit advice (if `using` is
///   set, the probe's result is reported alongside but doesn't override).
/// - `sweep` — grid of values to iterate over. Each sweep value produces
///   a separate result row.
/// - `superposition` — set of values held in parallel and combined
///   without collapsing, per a specified combiner.
///
/// When a dimension is `None`, the recipe's declared default for that
/// dimension takes effect.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Binding {
    /// User-forced value, or `None` to use the recipe default.
    pub using: Option<Value>,
    /// Data probe for autodiscovery/advice, or `None` to use the recipe
    /// default.
    pub autodiscover: Option<DataProbeRef>,
    /// Sweep grid, or `None` for no sweep (use recipe default).
    pub sweep: Option<SweepSpec>,
    /// Superposition spec, or `None` for no superposition (use recipe
    /// default).
    pub superposition: Option<SuperpositionSpec>,
}

impl Binding {
    /// Empty binding — all four dimensions default to recipe declarations.
    pub fn inherit_defaults() -> Self {
        Self::default()
    }

    /// Binding with only `using` set.
    pub fn using(value: Value) -> Self {
        Self {
            using: Some(value),
            ..Self::default()
        }
    }

    /// Binding with only `autodiscover` set.
    pub fn autodiscover(probe: DataProbeRef) -> Self {
        Self {
            autodiscover: Some(probe),
            ..Self::default()
        }
    }

    /// Binding with only `sweep` set.
    pub fn sweep(values: Vec<Value>) -> Self {
        Self {
            sweep: Some(SweepSpec { values }),
            ..Self::default()
        }
    }

    /// Binding with only `superposition` set.
    pub fn superposition(values: Vec<Value>, combiner: Combiner) -> Self {
        Self {
            superposition: Some(SuperpositionSpec { values, combiner }),
            ..Self::default()
        }
    }

    // ── Composable builders ──────────────────────────────────────────────

    pub fn with_using(mut self, value: Value) -> Self {
        self.using = Some(value);
        self
    }

    pub fn with_autodiscover(mut self, probe: DataProbeRef) -> Self {
        self.autodiscover = Some(probe);
        self
    }

    pub fn with_sweep(mut self, values: Vec<Value>) -> Self {
        self.sweep = Some(SweepSpec { values });
        self
    }

    pub fn with_superposition(mut self, values: Vec<Value>, combiner: Combiner) -> Self {
        self.superposition = Some(SuperpositionSpec { values, combiner });
        self
    }

    /// True if this binding fully inherits from recipe defaults (no
    /// dimension is set).
    pub fn is_fully_default(&self) -> bool {
        self.using.is_none()
            && self.autodiscover.is_none()
            && self.sweep.is_none()
            && self.superposition.is_none()
    }
}

/// Sweep grid: run the step once per value and return a result-per-value.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepSpec {
    pub values: Vec<Value>,
}

/// Superposition: hold multiple values in parallel and combine without
/// collapsing.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperpositionSpec {
    pub values: Vec<Value>,
    pub combiner: Combiner,
}

/// A parameter value. Pipelines carry these opaquely and pass them to
/// recipes; recipes know how to interpret each key.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// A tag like `"varimax"`, `"ml"`, `"minres"`. Semantically a string,
    /// kept separate so IDE can render as a dropdown.
    Method(String),
    /// Fixed-length vector.
    Vector(Vec<f64>),
    /// Matrix (row-major).
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
}

/// Reference to a named runtime probe for Layer 1 autodiscovery.
///
/// Maps into [`primitives::oracle::algorithm_properties::DataProbe`] and
/// recipe-specific probe registries. The string is the probe name; the
/// execution layer resolves it at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataProbeRef(pub String);

/// How to combine multiple results in a superposition without collapsing.
///
/// The set is open — recipes can register their own via the `Custom`
/// variant. Standard ones cover the `.discover()` patterns in CLAUDE.md.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Combiner {
    /// Keep every result, report them all separately. The "superposition
    /// fingerprint" from the garden entries.
    KeepAll,
    /// Weighted sum of per-component results (Bayesian model averaging).
    WeightedSum,
    /// Partition results into "agreeing" and "disagreeing" buckets by
    /// comparing against each other with the given threshold.
    AgreementPartition { threshold_basis_points: u32 },
    /// Custom combiner named by a string tag that a recipe registers.
    Custom(String),
}

// ── Output shape info ─────────────────────────────────────────────────────

/// What a step's output column looks like, known at spec time.
///
/// The IDE reads these to render the chain graph and to let the user
/// pick downstream references without running anything. Every field is
/// pre-execution metadata — the data doesn't exist yet.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputColumn {
    /// Semantic name in the recipe's own vocabulary ("loadings",
    /// "eigenvalues", "scores", "residuals"). The IDE combines this
    /// with the step index to produce the A####/B#### display label.
    pub semantic_name: String,
    /// Short human description shown in hover/tooltip.
    pub description: String,
    /// Shape (scalar / vector / matrix / table) with dimension sources.
    pub shape: OutputShape,
    /// Storage dtype.
    pub dtype: Dtype,
    /// Does this output carry a confidence/validity V column per the
    /// CLAUDE.md DO/V convention?
    pub has_v_column: bool,
}

/// Dimensionality and dynamism of an output.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputShape {
    Scalar,
    Vector {
        length: DimensionSource,
    },
    Matrix {
        rows: DimensionSource,
        cols: DimensionSource,
    },
    /// Table with a fixed schema of named columns.
    Table {
        columns: Vec<(String, Dtype)>,
    },
}

/// Where a dimension value comes from.
///
/// A dimension can be known at spec time (fixed), known once input data
/// is attached (rows/cols of the input), or derived from a parameter
/// value (`FromChoice`). This lets the IDE reason about chain shapes
/// before any data is loaded.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimensionSource {
    /// Known at pipeline spec time.
    Fixed(usize),
    /// Number of rows in the input data.
    InputRows,
    /// Number of columns in the input data.
    InputCols,
    /// Equal to a named choice key's value (must resolve to `Value::Int`).
    FromChoice(ChoiceKey),
}

/// Numeric/logical dtype of a stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F64,
    I32,
    I64,
    Bool,
    /// Opaque string (categorical label, method name, etc.).
    String,
}

// ── Builders ──────────────────────────────────────────────────────────────

impl Pipeline {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            steps: Vec::new(),
        }
    }

    pub fn with_step(mut self, step: PipelineStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn find_step(&self, key: &str) -> Option<&PipelineStep> {
        self.steps.iter().find(|s| s.key == key)
    }

    pub fn step_index(&self, key: &str) -> Option<usize> {
        self.steps.iter().position(|s| s.key == key)
    }
}

impl PipelineStep {
    pub fn new(
        key: impl Into<String>,
        display_name: impl Into<String>,
        recipe: RecipeRef,
    ) -> Self {
        Self {
            key: key.into(),
            display_name: display_name.into(),
            recipe,
            choices: BTreeMap::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_choice(mut self, key: impl Into<ChoiceKey>, binding: Binding) -> Self {
        self.choices.insert(key.into(), binding);
        self
    }

    pub fn with_output(mut self, output: OutputColumn) -> Self {
        self.outputs.push(output);
        self
    }
}

impl OutputColumn {
    pub fn new(
        semantic_name: impl Into<String>,
        shape: OutputShape,
        dtype: Dtype,
    ) -> Self {
        Self {
            semantic_name: semantic_name.into(),
            description: String::new(),
            shape,
            dtype,
            has_v_column: false,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_v_column(mut self) -> Self {
        self.has_v_column = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pipeline construction ──────────────────────────────────────────────

    #[test]
    fn pipeline_construction() {
        let step = PipelineStep::new(
            "cor",
            "Correlation Matrix",
            RecipeRef::Expr("correlation_matrix".into()),
        )
        .with_choice("method", Binding::using(Value::Method("pearson".into())))
        .with_output(OutputColumn::new(
            "correlation_matrix",
            OutputShape::Matrix {
                rows: DimensionSource::InputCols,
                cols: DimensionSource::InputCols,
            },
            Dtype::F64,
        ));

        let pipeline = Pipeline::new("test")
            .with_description("Test pipeline")
            .with_step(step);

        assert_eq!(pipeline.name, "test");
        assert_eq!(pipeline.steps.len(), 1);
        assert_eq!(pipeline.steps[0].key, "cor");
    }

    #[test]
    fn pipeline_find_and_index_by_key() {
        let pipeline = Pipeline::new("p")
            .with_step(PipelineStep::new(
                "a",
                "A",
                RecipeRef::Recipe("foo".into()),
            ))
            .with_step(PipelineStep::new(
                "b",
                "B",
                RecipeRef::Recipe("bar".into()),
            ));

        assert!(pipeline.find_step("a").is_some());
        assert!(pipeline.find_step("missing").is_none());
        assert_eq!(pipeline.step_index("a"), Some(0));
        assert_eq!(pipeline.step_index("b"), Some(1));
    }

    // ── Composable bindings ────────────────────────────────────────────────

    #[test]
    fn empty_binding_inherits_all_defaults() {
        let b = Binding::inherit_defaults();
        assert!(b.is_fully_default());
        assert!(b.using.is_none());
        assert!(b.autodiscover.is_none());
        assert!(b.sweep.is_none());
        assert!(b.superposition.is_none());
    }

    #[test]
    fn using_with_autodiscover_both_active() {
        // Forcing a value while still requesting the probe for advice.
        let b = Binding::using(Value::Method("pearson".into()))
            .with_autodiscover(DataProbeRef("normality_probe".into()));

        assert!(b.using.is_some());
        assert!(b.autodiscover.is_some());
        assert!(b.sweep.is_none());
        assert!(b.superposition.is_none());
        assert!(!b.is_fully_default());
    }

    #[test]
    fn sweep_with_using_composes() {
        // User forces a nominal value AND requests a sweep around it —
        // maybe for sensitivity analysis. The recipe sees both and can
        // decide how to reconcile (typically: run the sweep and mark the
        // using-value as the "selected" one).
        let b = Binding::using(Value::Float(0.05))
            .with_sweep(vec![Value::Float(0.01), Value::Float(0.05), Value::Float(0.1)]);

        assert!(b.using.is_some());
        assert!(b.sweep.is_some());
        if let Some(ref s) = b.sweep {
            assert_eq!(s.values.len(), 3);
        }
    }

    #[test]
    fn all_four_dimensions_simultaneously() {
        let b = Binding::inherit_defaults()
            .with_using(Value::Method("varimax".into()))
            .with_autodiscover(DataProbeRef("factorability_probe".into()))
            .with_sweep(vec![
                Value::Method("varimax".into()),
                Value::Method("promax".into()),
                Value::Method("oblimin".into()),
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

        assert!(b.using.is_some());
        assert!(b.autodiscover.is_some());
        assert!(b.sweep.is_some());
        assert!(b.superposition.is_some());
    }

    // ── Sweep and superposition specs ──────────────────────────────────────

    #[test]
    fn sweep_spec_holds_multiple_values() {
        let sweep = SweepSpec {
            values: vec![Value::Int(1), Value::Int(2), Value::Int(3)],
        };
        assert_eq!(sweep.values.len(), 3);
    }

    #[test]
    fn superposition_spec_has_combiner() {
        let sp = SuperpositionSpec {
            values: vec![Value::Method("a".into()), Value::Method("b".into())],
            combiner: Combiner::KeepAll,
        };
        assert_eq!(sp.values.len(), 2);
        assert_eq!(sp.combiner, Combiner::KeepAll);
    }

    // ── Output shape ───────────────────────────────────────────────────────

    #[test]
    fn output_column_with_v() {
        let col = OutputColumn::new(
            "loadings",
            OutputShape::Matrix {
                rows: DimensionSource::InputCols,
                cols: DimensionSource::FromChoice("n_factors".into()),
            },
            Dtype::F64,
        )
        .with_v_column()
        .with_description(
            "Factor loadings: how much each variable loads onto each factor",
        );
        assert!(col.has_v_column);
        assert_eq!(col.dtype, Dtype::F64);
    }

    #[test]
    fn dimension_source_variants_distinct() {
        let fixed = DimensionSource::Fixed(10);
        let rows = DimensionSource::InputRows;
        let cols = DimensionSource::InputCols;
        let from_choice = DimensionSource::FromChoice("k".into());
        assert_ne!(fixed, rows);
        assert_ne!(rows, cols);
        assert_ne!(cols, from_choice);
    }

    #[test]
    fn recipe_ref_variants_are_distinct() {
        let atom = RecipeRef::Atom("accumulate".into());
        let prim = RecipeRef::Primitive("fmadd".into());
        let expr = RecipeRef::Expr("mean".into());
        let recipe = RecipeRef::Recipe("exp".into());
        for r in [atom, prim, expr, recipe] {
            match r {
                RecipeRef::Atom(_)
                | RecipeRef::Primitive(_)
                | RecipeRef::Expr(_)
                | RecipeRef::Recipe(_) => {}
            }
        }
    }
}
