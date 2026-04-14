//! Parse `spec.toml` files into [`OwnedRecipeSchema`] instances.
//!
//! This is the bridge between the user-editable `.toml` spec format and the
//! runtime schema type. Each recipe ships a `*.spec.toml` file next to its
//! `.rs` implementation; the toml is embedded into the binary via
//! `include_str!` and parsed on first access via `OnceLock` so we only pay
//! the parse cost once per process.
//!
//! # Format conventions
//!
//! Each parameter declares its `kind` (bool / int / float / string / method).
//! Default values and enum domain values are written as bare toml scalars
//! and coerced at parse time according to the parameter's kind. So for a
//! `kind = "method"` parameter, `using = "compensated"` produces
//! `Method("compensated")`, not `String("compensated")`.
//!
//! This keeps the .toml files ergonomic for humans (no tag noise) while
//! preserving type safety on the Rust side.
//!
//! # Why owned types here
//!
//! Parsed toml is owned data (`String` / `Vec`). [`OwnedRecipeSchema`]
//! mirrors [`super::schema::RecipeSchema`] with owned fields so parsed
//! data can populate it directly. The pipeline-layer accessor uses
//! `OnceLock` + `Box::leak` to produce a `&'static` reference for
//! long-lived consumer use.

use super::types::{
    Binding, Combiner, DataProbeRef, DimensionSource, Dtype, OutputColumn, OutputShape,
    SuperpositionSpec, SweepSpec, Value,
};
use serde::Deserialize;

// ── The owned analogue of RecipeSchema ──────────────────────────────────────

/// Fully-owned schema, loaded from a `spec.toml` file.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedRecipeSchema {
    pub name: String,
    pub description: String,
    pub parameters: Vec<OwnedParameterSpec>,
    pub outputs: Vec<OwnedOutputSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedParameterSpec {
    pub key: String,
    pub display_name: String,
    pub description: String,
    pub kind: ParamKind,
    pub advanced: bool,
    pub default: OwnedDefaultBinding,
    pub domain: Option<OwnedValueDomain>,
}

/// Scalar type of a parameter. Determines how bare toml values are coerced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Bool,
    Int,
    Float,
    String,
    Method,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OwnedDefaultBinding {
    pub using: Option<OwnedDefaultValue>,
    pub autodiscover: Option<String>,
    pub sweep: Option<Vec<OwnedDefaultValue>>,
    pub superposition: Option<OwnedDefaultSuperposition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedDefaultSuperposition {
    pub values: Vec<OwnedDefaultValue>,
    pub combiner: OwnedDefaultCombiner,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedDefaultCombiner {
    KeepAll,
    WeightedSum,
    AgreementPartition { threshold_basis_points: u32 },
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedDefaultValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Method(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedOutputSpec {
    pub semantic_name: String,
    pub description: String,
    pub shape: OwnedOutputShape,
    pub dtype: Dtype,
    pub has_v_column: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedOutputShape {
    Scalar,
    Vector {
        length: OwnedDimensionSource,
    },
    Matrix {
        rows: OwnedDimensionSource,
        cols: OwnedDimensionSource,
    },
    Table {
        columns: Vec<(String, Dtype)>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedDimensionSource {
    Fixed(usize),
    InputRows,
    InputCols,
    FromChoice(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedValueDomain {
    Enum(Vec<OwnedDefaultValue>),
    Range { min: f64, max: f64 },
    IntRange { min: i64, max: i64 },
    FreeString,
}

// ── Conversion to runtime Binding / OutputColumn types ────────────────────

impl OwnedDefaultValue {
    pub fn to_value(&self) -> Value {
        match self {
            OwnedDefaultValue::Bool(b) => Value::Bool(*b),
            OwnedDefaultValue::Int(i) => Value::Int(*i),
            OwnedDefaultValue::Float(f) => Value::Float(*f),
            OwnedDefaultValue::String(s) => Value::String(s.clone()),
            OwnedDefaultValue::Method(m) => Value::Method(m.clone()),
        }
    }
}

impl OwnedDefaultCombiner {
    pub fn to_combiner(&self) -> Combiner {
        match self {
            OwnedDefaultCombiner::KeepAll => Combiner::KeepAll,
            OwnedDefaultCombiner::WeightedSum => Combiner::WeightedSum,
            OwnedDefaultCombiner::AgreementPartition {
                threshold_basis_points,
            } => Combiner::AgreementPartition {
                threshold_basis_points: *threshold_basis_points,
            },
            OwnedDefaultCombiner::Custom(tag) => Combiner::Custom(tag.clone()),
        }
    }
}

impl OwnedDefaultBinding {
    pub fn to_binding(&self) -> Binding {
        Binding {
            using: self.using.as_ref().map(|v| v.to_value()),
            autodiscover: self
                .autodiscover
                .as_ref()
                .map(|s| DataProbeRef(s.clone())),
            sweep: self.sweep.as_ref().map(|vs| SweepSpec {
                values: vs.iter().map(|v| v.to_value()).collect(),
            }),
            superposition: self
                .superposition
                .as_ref()
                .map(|sp| SuperpositionSpec {
                    values: sp.values.iter().map(|v| v.to_value()).collect(),
                    combiner: sp.combiner.to_combiner(),
                }),
        }
    }
}

impl OwnedDimensionSource {
    pub fn to_dimension_source(&self) -> DimensionSource {
        match self {
            OwnedDimensionSource::Fixed(n) => DimensionSource::Fixed(*n),
            OwnedDimensionSource::InputRows => DimensionSource::InputRows,
            OwnedDimensionSource::InputCols => DimensionSource::InputCols,
            OwnedDimensionSource::FromChoice(k) => DimensionSource::FromChoice(k.clone()),
        }
    }
}

impl OwnedOutputShape {
    pub fn to_output_shape(&self) -> OutputShape {
        match self {
            OwnedOutputShape::Scalar => OutputShape::Scalar,
            OwnedOutputShape::Vector { length } => OutputShape::Vector {
                length: length.to_dimension_source(),
            },
            OwnedOutputShape::Matrix { rows, cols } => OutputShape::Matrix {
                rows: rows.to_dimension_source(),
                cols: cols.to_dimension_source(),
            },
            OwnedOutputShape::Table { columns } => OutputShape::Table {
                columns: columns.iter().cloned().collect(),
            },
        }
    }
}

impl OwnedOutputSpec {
    pub fn to_output_column(&self) -> OutputColumn {
        OutputColumn {
            semantic_name: self.semantic_name.clone(),
            description: self.description.clone(),
            shape: self.shape.to_output_shape(),
            dtype: self.dtype,
            has_v_column: self.has_v_column,
        }
    }
}

// ── TOML wire format ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct TomlRoot {
    recipe: TomlRecipeMeta,
    #[serde(default)]
    parameters: Vec<TomlParameter>,
    #[serde(default)]
    outputs: Vec<TomlOutput>,
}

#[derive(Debug, Deserialize)]
struct TomlRecipeMeta {
    name: String,
    description: String,
    // Other metadata (layer, family, long_description, references,
    // decomposition, sharing, writeup) is accepted-but-ignored on the
    // Rust side. The IDE will consume the rest; Rust doesn't need it
    // for execution yet.
}

#[derive(Debug, Deserialize)]
struct TomlParameter {
    key: String,
    display_name: String,
    description: String,
    kind: ParamKind,
    #[serde(default)]
    advanced: bool,
    #[serde(default)]
    default: TomlDefaultBinding,
    #[serde(default)]
    domain: Option<TomlDomain>,
}

#[derive(Debug, Deserialize, Default)]
struct TomlDefaultBinding {
    #[serde(default)]
    using: Option<toml::Value>,
    #[serde(default)]
    autodiscover: Option<String>,
    #[serde(default)]
    sweep: Option<Vec<toml::Value>>,
    #[serde(default)]
    superposition: Option<TomlSuperposition>,
}

#[derive(Debug, Deserialize)]
struct TomlSuperposition {
    values: Vec<toml::Value>,
    combiner: TomlCombiner,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TomlCombiner {
    KeepAll,
    WeightedSum,
    AgreementPartition { threshold_basis_points: u32 },
    Custom { tag: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TomlDomain {
    Enum { values: Vec<toml::Value> },
    Range { min: f64, max: f64 },
    IntRange { min: i64, max: i64 },
    FreeString,
}

#[derive(Debug, Deserialize)]
struct TomlOutput {
    semantic_name: String,
    description: String,
    shape: TomlOutputShape,
    dtype: String,
    #[serde(default)]
    has_v_column: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TomlOutputShape {
    Scalar,
    Vector {
        #[serde(default)]
        length: Option<usize>,
        #[serde(default)]
        length_from: Option<String>,
    },
    Matrix {
        rows: TomlDim,
        cols: TomlDim,
    },
    Table {
        columns: Vec<(String, String)>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum TomlDim {
    Fixed(usize),
    Keyword(String),
}

// ── Coercion: toml::Value → OwnedDefaultValue by parameter kind ────────────

fn coerce_value(v: &toml::Value, kind: ParamKind) -> Result<OwnedDefaultValue, String> {
    match (kind, v) {
        (ParamKind::Bool, toml::Value::Boolean(b)) => Ok(OwnedDefaultValue::Bool(*b)),
        (ParamKind::Int, toml::Value::Integer(i)) => Ok(OwnedDefaultValue::Int(*i)),
        (ParamKind::Float, toml::Value::Float(f)) => Ok(OwnedDefaultValue::Float(*f)),
        // Allow integer literals where a float is expected — ergonomic.
        (ParamKind::Float, toml::Value::Integer(i)) => Ok(OwnedDefaultValue::Float(*i as f64)),
        (ParamKind::String, toml::Value::String(s)) => Ok(OwnedDefaultValue::String(s.clone())),
        (ParamKind::Method, toml::Value::String(s)) => Ok(OwnedDefaultValue::Method(s.clone())),
        (kind, val) => Err(format!(
            "parameter value {val:?} does not match declared kind {kind:?}"
        )),
    }
}

fn parse_dim_keyword(s: &str) -> Result<OwnedDimensionSource, String> {
    match s {
        "input_rows" => Ok(OwnedDimensionSource::InputRows),
        "input_cols" => Ok(OwnedDimensionSource::InputCols),
        other if other.starts_with("choice:") => Ok(OwnedDimensionSource::FromChoice(
            other.strip_prefix("choice:").unwrap().to_string(),
        )),
        other => Err(format!(
            "unknown dimension keyword: {other:?}. Expected `input_rows`, `input_cols`, or `choice:<key>`"
        )),
    }
}

fn parse_length(
    length: &Option<usize>,
    length_from: &Option<String>,
) -> Result<OwnedDimensionSource, String> {
    match (length, length_from) {
        (Some(n), None) => Ok(OwnedDimensionSource::Fixed(*n)),
        (None, Some(s)) => parse_dim_keyword(s),
        (Some(_), Some(_)) => Err(
            "vector shape: specify exactly one of `length` or `length_from`".to_string(),
        ),
        (None, None) => Err(
            "vector shape: must specify `length` (fixed usize) or `length_from` (keyword)"
                .to_string(),
        ),
    }
}

fn parse_dtype(s: &str) -> Result<Dtype, String> {
    match s {
        "f32" => Ok(Dtype::F32),
        "f64" => Ok(Dtype::F64),
        "i32" => Ok(Dtype::I32),
        "i64" => Ok(Dtype::I64),
        "bool" => Ok(Dtype::Bool),
        "string" => Ok(Dtype::String),
        other => Err(format!("unknown dtype: {other:?}")),
    }
}

impl TomlDim {
    fn to_owned_dim(&self) -> Result<OwnedDimensionSource, String> {
        match self {
            TomlDim::Fixed(n) => Ok(OwnedDimensionSource::Fixed(*n)),
            TomlDim::Keyword(s) => parse_dim_keyword(s),
        }
    }
}

impl TomlCombiner {
    fn to_owned_combiner(&self) -> OwnedDefaultCombiner {
        match self {
            TomlCombiner::KeepAll => OwnedDefaultCombiner::KeepAll,
            TomlCombiner::WeightedSum => OwnedDefaultCombiner::WeightedSum,
            TomlCombiner::AgreementPartition {
                threshold_basis_points,
            } => OwnedDefaultCombiner::AgreementPartition {
                threshold_basis_points: *threshold_basis_points,
            },
            TomlCombiner::Custom { tag } => OwnedDefaultCombiner::Custom(tag.clone()),
        }
    }
}

fn to_owned(root: TomlRoot) -> Result<OwnedRecipeSchema, String> {
    let parameters = root
        .parameters
        .into_iter()
        .map(|p| {
            let kind = p.kind;
            let default = OwnedDefaultBinding {
                using: p
                    .default
                    .using
                    .as_ref()
                    .map(|v| coerce_value(v, kind))
                    .transpose()?,
                autodiscover: p.default.autodiscover,
                sweep: p
                    .default
                    .sweep
                    .as_ref()
                    .map(|vs| vs.iter().map(|v| coerce_value(v, kind)).collect())
                    .transpose()?,
                superposition: p
                    .default
                    .superposition
                    .as_ref()
                    .map(|sp| {
                        let values: Result<Vec<_>, _> =
                            sp.values.iter().map(|v| coerce_value(v, kind)).collect();
                        Ok::<_, String>(OwnedDefaultSuperposition {
                            values: values?,
                            combiner: sp.combiner.to_owned_combiner(),
                        })
                    })
                    .transpose()?,
            };
            let domain = match p.domain {
                Some(TomlDomain::Enum { values }) => {
                    let vs: Result<Vec<_>, _> =
                        values.iter().map(|v| coerce_value(v, kind)).collect();
                    Some(OwnedValueDomain::Enum(vs?))
                }
                Some(TomlDomain::Range { min, max }) => {
                    Some(OwnedValueDomain::Range { min, max })
                }
                Some(TomlDomain::IntRange { min, max }) => {
                    Some(OwnedValueDomain::IntRange { min, max })
                }
                Some(TomlDomain::FreeString) => Some(OwnedValueDomain::FreeString),
                None => None,
            };
            Ok(OwnedParameterSpec {
                key: p.key,
                display_name: p.display_name,
                description: p.description,
                kind,
                advanced: p.advanced,
                default,
                domain,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let outputs = root
        .outputs
        .into_iter()
        .map(|o| {
            let shape = match o.shape {
                TomlOutputShape::Scalar => OwnedOutputShape::Scalar,
                TomlOutputShape::Vector {
                    length,
                    length_from,
                } => OwnedOutputShape::Vector {
                    length: parse_length(&length, &length_from)?,
                },
                TomlOutputShape::Matrix { rows, cols } => OwnedOutputShape::Matrix {
                    rows: rows.to_owned_dim()?,
                    cols: cols.to_owned_dim()?,
                },
                TomlOutputShape::Table { columns } => OwnedOutputShape::Table {
                    columns: columns
                        .iter()
                        .map(|(name, dtype)| {
                            let dt = parse_dtype(dtype)?;
                            Ok((name.clone(), dt))
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                },
            };
            let dtype = parse_dtype(&o.dtype)?;
            Ok(OwnedOutputSpec {
                semantic_name: o.semantic_name,
                description: o.description,
                shape,
                dtype,
                has_v_column: o.has_v_column,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(OwnedRecipeSchema {
        name: root.recipe.name,
        description: root.recipe.description,
        parameters,
        outputs,
    })
}

/// Parse a spec.toml string into an [`OwnedRecipeSchema`].
pub fn parse_spec_toml(toml_str: &str) -> Result<OwnedRecipeSchema, String> {
    let root: TomlRoot =
        toml::from_str(toml_str).map_err(|e| format!("toml parse error: {e}"))?;
    to_owned(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_exp_spec_toml() {
        let exp_toml = include_str!("../libm/exp.spec.toml");
        let schema = parse_spec_toml(exp_toml).expect("valid exp spec");
        assert_eq!(schema.name, "exp");
        assert_eq!(schema.description.contains("Natural exponential"), true);
        assert_eq!(schema.parameters.len(), 1);

        let precision = &schema.parameters[0];
        assert_eq!(precision.key, "precision");
        assert_eq!(precision.display_name, "Precision strategy");
        assert_eq!(precision.kind, ParamKind::Method);
        assert!(!precision.advanced);

        // Bare "compensated" in the toml should coerce to Method("compensated")
        // because the parameter's kind is "method".
        match &precision.default.using {
            Some(OwnedDefaultValue::Method(m)) => assert_eq!(m, "compensated"),
            other => panic!("expected Method('compensated'), got {other:?}"),
        }

        // Domain values should all be Method("…") for the same reason.
        match &precision.domain {
            Some(OwnedValueDomain::Enum(vals)) => {
                assert_eq!(vals.len(), 3);
                for v in vals {
                    assert!(
                        matches!(v, OwnedDefaultValue::Method(_)),
                        "expected Method variant, got {v:?}"
                    );
                }
            }
            other => panic!("expected enum domain, got {other:?}"),
        }

        assert_eq!(schema.outputs.len(), 1);
        assert_eq!(schema.outputs[0].semantic_name, "result");
        assert_eq!(schema.outputs[0].dtype, Dtype::F64);
        assert!(!schema.outputs[0].has_v_column);
    }

    #[test]
    fn coerce_int_to_float() {
        let v = toml::Value::Integer(3);
        let coerced = coerce_value(&v, ParamKind::Float).unwrap();
        assert_eq!(coerced, OwnedDefaultValue::Float(3.0));
    }

    #[test]
    fn coerce_rejects_mismatched_kind() {
        let v = toml::Value::String("hello".to_string());
        let err = coerce_value(&v, ParamKind::Int).unwrap_err();
        assert!(err.contains("does not match declared kind"));
    }

    #[test]
    fn parse_dim_keyword_variants() {
        assert_eq!(
            parse_dim_keyword("input_rows").unwrap(),
            OwnedDimensionSource::InputRows
        );
        assert_eq!(
            parse_dim_keyword("input_cols").unwrap(),
            OwnedDimensionSource::InputCols
        );
        assert_eq!(
            parse_dim_keyword("choice:n_factors").unwrap(),
            OwnedDimensionSource::FromChoice("n_factors".to_string())
        );
        assert!(parse_dim_keyword("bogus").is_err());
    }

    #[test]
    fn default_binding_to_binding_roundtrip() {
        let owned = OwnedDefaultBinding {
            using: Some(OwnedDefaultValue::Method("varimax".to_string())),
            autodiscover: Some("probe_a".to_string()),
            sweep: None,
            superposition: None,
        };
        let binding = owned.to_binding();
        assert_eq!(binding.using, Some(Value::Method("varimax".to_string())));
        assert_eq!(
            binding.autodiscover,
            Some(DataProbeRef("probe_a".to_string()))
        );
    }
}
