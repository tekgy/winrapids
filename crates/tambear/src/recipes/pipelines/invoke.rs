//! Bridge surface: `invoke` — the single function that every non-Rust
//! consumer (Tauri, Python, CLI, web worker) calls into tambear through.
//!
//! # What this is
//!
//! One function. Takes a JSON-serializable request, returns a
//! JSON-serializable response. No Rust types leak across the boundary.
//!
//! # What this is NOT
//!
//! - **Not Tauri-specific.** A Tauri app wraps this in a `#[tauri::command]`
//!   with one line: `fn tambear_invoke(req: String) -> String { ... }`.
//!   A Python binding wraps it with PyO3. A CLI wraps it with stdio. A
//!   web worker wraps it with postMessage. Same function, different
//!   transport.
//! - **Not an execution engine.** The request describes WHAT to do; a
//!   separate execution/orchestration crate (not yet built) actually
//!   runs the computation. This module validates, resolves schemas,
//!   and produces the dispatch plan. When execution lands, it hooks
//!   into the stub `execute()` below.
//! - **Not stateful.** No session, no connection. The caller is
//!   responsible for passing in whatever data handles / TamSession
//!   references the execution layer needs. This keeps the bridge
//!   trivially parallelizable.
//!
//! # The request shape
//!
//! ```json
//! {
//!   "op": "schema_for" | "infer_shape" | "materialize" | "execute",
//!   "payload": { ...op-specific... }
//! }
//! ```
//!
//! # The response shape
//!
//! Always the same envelope for every op:
//!
//! ```json
//! {
//!   "status": "ok" | "err",
//!   "result_type": "schema" | "chain_shape" | "saved_pipeline" | "step_outputs",
//!   "payload": { ... },
//!   "advice": [...],      // Layer-1 recommendations, user-facing
//!   "diagnostics": [...], // technical diagnostics (timing, memory, etc.)
//!   "lints": [...]        // static warnings (schema-level issues)
//! }
//! ```
//!
//! The envelope is stable across ops; only `payload.result_type` +
//! `payload.contents` vary.

use super::{schema_for, types::RecipeRef, Pipeline};
use serde::{Deserialize, Serialize};

// ── The request envelope ──────────────────────────────────────────────────

/// Top-level request from any bridge consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "payload", rename_all = "snake_case")]
pub enum InvokeRequest {
    /// Return the schema for a recipe. Used by the IDE to render step
    /// cards without knowing anything recipe-specific.
    SchemaFor { recipe: RecipeRef },

    /// Given a Pipeline (with no data attached), return the chain shape:
    /// what each step will produce, including dimension-source resolution
    /// via effective bindings. The IDE uses this to draw the chain graph
    /// before the user clicks Run.
    InferShape { pipeline: Pipeline },

    /// Materialize the given pipeline by freezing every parameter's
    /// effective binding into explicit values. Used when the user saves
    /// a pipeline for later or for community sharing — the saved form is
    /// immune to future tambear default changes.
    Materialize { pipeline: Pipeline },

    /// Execute the pipeline. Currently returns a stub error — the
    /// execution/orchestration layer is not yet wired. When it lands,
    /// this op will dispatch to it and return step outputs.
    Execute {
        pipeline: Pipeline,
        /// Caller-provided opaque handle to data. The execution layer
        /// interprets these — tambear at this layer doesn't care whether
        /// they're file paths, Arrow columns, or session references.
        data_handles: Vec<DataHandle>,
    },
}

/// An opaque reference to data the caller has made available to the
/// execution layer. The bridge doesn't interpret these; they pass
/// through to the execution crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataHandle {
    /// Caller-chosen identifier (e.g., "column:age", "file:data.csv").
    pub handle_id: String,
    /// Optional human-readable name (the IDE's column label).
    pub display_name: Option<String>,
    /// Free-form metadata the caller wants to attach.
    #[serde(default)]
    pub meta: serde_json::Value,
}

// ── The response envelope ─────────────────────────────────────────────────

/// Universal response envelope. Every `invoke` call returns one of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeResponse {
    pub status: Status,
    pub result: ResultPayload,
    #[serde(default)]
    pub advice: Vec<Advice>,
    #[serde(default)]
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub lints: Vec<Lint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Err,
}

/// Discriminated union of result payloads. The IDE switches on
/// `result_type` to pick a renderer; no recipe-specific knowledge
/// required.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultPayload {
    /// Response to SchemaFor.
    Schema {
        // We keep this as serde_json::Value rather than typed because
        // the RecipeSchema type uses &'static str and doesn't round-trip
        // through serde cleanly. The actual content is an object matching
        // the schema shape.
        schema: serde_json::Value,
    },
    /// Response to InferShape. See `shape::ChainShape`.
    ChainShape {
        chain: super::shape::ChainShape,
    },
    /// Response to Materialize.
    SavedPipeline {
        saved: super::serialize::SavedPipeline,
    },
    /// Response to Execute (stub).
    StepOutputs {
        /// Per-step outputs. Currently always empty because execute
        /// is not wired.
        steps: Vec<StepOutput>,
    },
    /// Bridge-level error (not a recipe error, a transport or schema
    /// error).
    Error {
        message: String,
    },
}

/// Per-step output. Placeholder until execution lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    /// Step key from the pipeline.
    pub step_key: String,
    /// Output columns produced by this step.
    pub columns: Vec<StepOutputColumn>,
}

/// One output column from a step, with the actual data (or a reference
/// to it). Currently a placeholder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutputColumn {
    pub semantic_name: String,
    /// A handle the caller can use to fetch the column data via a
    /// separate call. Keeps the response small for large outputs.
    pub handle_id: String,
    /// Summary statistics or a preview the IDE renders immediately
    /// without needing a separate data-fetch round-trip.
    #[serde(default)]
    pub preview: serde_json::Value,
}

/// User-facing advice from Layer 1 auto-dispatch and algorithm_properties
/// weakness detection. Goes to the "Tambear recommends…" panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advice {
    pub step_key: Option<String>,
    pub parameter_key: Option<String>,
    pub severity: Severity,
    pub message: String,
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Advisory,
    Warning,
    Error,
}

/// Technical diagnostic (timing, memory, backend dispatch). IDE renders
/// these in the developer/profiler view, not the main user surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub step_key: Option<String>,
    pub category: String,
    pub key: String,
    pub value: serde_json::Value,
}

/// Static schema-level warning (e.g., unresolved dimension from shape
/// inference, unknown recipe reference). Never comes from executing
/// data — it comes from analyzing the pipeline spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lint {
    pub step_key: Option<String>,
    pub code: String,
    pub message: String,
}

// ── The dispatch function ─────────────────────────────────────────────────

/// The single entry point for every non-Rust consumer. Takes a request
/// object (ideally deserialized from JSON by the caller), returns a
/// response object (ideally serialized back to JSON).
///
/// Tauri binding:
///
/// ```ignore
/// #[tauri::command]
/// fn tambear_invoke(req: String) -> String {
///     let req: InvokeRequest = serde_json::from_str(&req).unwrap_or_else(|e| {
///         // Return a transport-error response
///         todo!()
///     });
///     let resp = tambear::recipes::pipelines::invoke::invoke(req);
///     serde_json::to_string(&resp).unwrap()
/// }
/// ```
pub fn invoke(request: InvokeRequest) -> InvokeResponse {
    match request {
        InvokeRequest::SchemaFor { recipe } => handle_schema_for(&recipe),
        InvokeRequest::InferShape { pipeline } => handle_infer_shape(pipeline),
        InvokeRequest::Materialize { pipeline } => handle_materialize(pipeline),
        InvokeRequest::Execute {
            pipeline,
            data_handles,
        } => handle_execute(pipeline, data_handles),
    }
}

fn handle_schema_for(recipe: &RecipeRef) -> InvokeResponse {
    match schema_for(recipe) {
        Some(schema) => {
            // Serialize the schema as a JSON object by extracting its
            // fields. Can't just serde_json::to_value(schema) because
            // RecipeSchema uses &'static str which doesn't have a clean
            // Serialize impl without custom wrapping.
            let schema_json = serde_json::json!({
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters.iter().map(|p| serde_json::json!({
                    "key": p.key,
                    "display_name": p.display_name,
                    "description": p.description,
                    // default / domain are complex enums — we expose the
                    // most-common display fields. A full schema export is
                    // planned once the IDE renders need it.
                    "has_default_using": p.default.using.is_some(),
                    "has_default_autodiscover": p.default.autodiscover.is_some(),
                    "has_default_sweep": p.default.sweep.is_some(),
                    "has_default_superposition": p.default.superposition.is_some(),
                })).collect::<Vec<_>>(),
                "outputs": schema.outputs.iter().map(|o| serde_json::json!({
                    "semantic_name": o.semantic_name,
                    "description": o.description,
                    "dtype": format!("{:?}", o.dtype).to_lowercase(),
                    "has_v_column": o.has_v_column,
                })).collect::<Vec<_>>(),
            });
            InvokeResponse {
                status: Status::Ok,
                result: ResultPayload::Schema { schema: schema_json },
                advice: Vec::new(),
                diagnostics: Vec::new(),
                lints: Vec::new(),
            }
        }
        None => err_response(format!("unknown recipe: {recipe:?}")),
    }
}

fn handle_infer_shape(pipeline: Pipeline) -> InvokeResponse {
    let chain = super::shape::infer(&pipeline);
    // Pull any warnings from the chain into the lints field.
    let lints = chain
        .steps
        .iter()
        .flat_map(|step| {
            step.warnings.iter().map(|w| Lint {
                step_key: Some(step.key.clone()),
                code: format!("{w:?}").split_whitespace().next().unwrap_or("").to_string(),
                message: format!("{w:?}"),
            })
        })
        .collect();

    InvokeResponse {
        status: Status::Ok,
        result: ResultPayload::ChainShape { chain },
        advice: Vec::new(),
        diagnostics: Vec::new(),
        lints,
    }
}

fn handle_materialize(pipeline: Pipeline) -> InvokeResponse {
    match super::serialize::materialize(&pipeline) {
        Ok(materialized) => {
            let saved = super::serialize::SavedPipeline {
                version: super::serialize::SAVE_FORMAT_VERSION.to_string(),
                materialized: true,
                pipeline: materialized,
            };
            InvokeResponse {
                status: Status::Ok,
                result: ResultPayload::SavedPipeline { saved },
                advice: Vec::new(),
                diagnostics: Vec::new(),
                lints: Vec::new(),
            }
        }
        Err(e) => err_response(format!("materialize failed: {e}")),
    }
}

fn handle_execute(_pipeline: Pipeline, _handles: Vec<DataHandle>) -> InvokeResponse {
    // Execution is in a separate crate that doesn't exist yet. When it
    // lands, this will dispatch there and return actual StepOutputs.
    err_response(
        "execute is not yet wired — the orchestration/execution crate is under construction. \
         Shape inference, materialization, and schema queries all work today; come back for \
         execute when the exec layer ships."
            .to_string(),
    )
}

fn err_response(message: String) -> InvokeResponse {
    InvokeResponse {
        status: Status::Err,
        result: ResultPayload::Error { message },
        advice: Vec::new(),
        diagnostics: Vec::new(),
        lints: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::pipelines::{PipelineStep, step_from_schema};

    #[test]
    fn schema_for_exp_returns_schema_payload() {
        let req = InvokeRequest::SchemaFor {
            recipe: RecipeRef::Recipe("exp".to_string()),
        };
        let resp = invoke(req);
        assert_eq!(resp.status, Status::Ok);
        match &resp.result {
            ResultPayload::Schema { schema } => {
                assert_eq!(schema["name"], "exp");
                assert!(schema["parameters"].is_array());
                assert_eq!(schema["parameters"].as_array().unwrap().len(), 1);
            }
            other => panic!("expected Schema payload, got {other:?}"),
        }
    }

    #[test]
    fn schema_for_unknown_returns_error() {
        let req = InvokeRequest::SchemaFor {
            recipe: RecipeRef::Recipe("no_such_recipe".to_string()),
        };
        let resp = invoke(req);
        assert_eq!(resp.status, Status::Err);
        match &resp.result {
            ResultPayload::Error { message } => {
                assert!(message.contains("unknown recipe"));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn infer_shape_works_on_exp_pipeline() {
        let step = step_from_schema("e", RecipeRef::Recipe("exp".into())).unwrap();
        let pipeline = Pipeline::new("test").with_step(step);
        let req = InvokeRequest::InferShape { pipeline };
        let resp = invoke(req);
        assert_eq!(resp.status, Status::Ok);
        match &resp.result {
            ResultPayload::ChainShape { chain } => {
                assert_eq!(chain.steps.len(), 1);
                assert_eq!(chain.steps[0].key, "e");
            }
            _ => panic!("expected ChainShape"),
        }
    }

    #[test]
    fn materialize_roundtrips_exp_pipeline() {
        let step = step_from_schema("e", RecipeRef::Recipe("exp".into())).unwrap();
        let pipeline = Pipeline::new("test").with_step(step);
        let req = InvokeRequest::Materialize { pipeline };
        let resp = invoke(req);
        assert_eq!(resp.status, Status::Ok);
        match &resp.result {
            ResultPayload::SavedPipeline { saved } => {
                assert!(saved.materialized);
                assert_eq!(saved.pipeline.steps.len(), 1);
            }
            _ => panic!("expected SavedPipeline"),
        }
    }

    #[test]
    fn execute_stub_returns_err() {
        let step = step_from_schema("e", RecipeRef::Recipe("exp".into())).unwrap();
        let pipeline = Pipeline::new("test").with_step(step);
        let req = InvokeRequest::Execute {
            pipeline,
            data_handles: Vec::new(),
        };
        let resp = invoke(req);
        assert_eq!(resp.status, Status::Err);
        match &resp.result {
            ResultPayload::Error { message } => {
                assert!(message.contains("orchestration") || message.contains("not yet wired"));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn request_roundtrips_through_json() {
        let req = InvokeRequest::SchemaFor {
            recipe: RecipeRef::Recipe("exp".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: InvokeRequest = serde_json::from_str(&json).unwrap();
        match decoded {
            InvokeRequest::SchemaFor { recipe } => match recipe {
                RecipeRef::Recipe(name) => assert_eq!(name, "exp"),
                _ => panic!("expected Recipe variant"),
            },
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn response_envelope_roundtrips() {
        let resp = InvokeResponse {
            status: Status::Ok,
            result: ResultPayload::Error {
                message: "test".to_string(),
            },
            advice: vec![Advice {
                step_key: Some("step_a".to_string()),
                parameter_key: Some("method".to_string()),
                severity: Severity::Advisory,
                message: "Data non-normal; consider Spearman".to_string(),
                recommendation: Some("using(method=spearman)".to_string()),
            }],
            diagnostics: Vec::new(),
            lints: Vec::new(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: InvokeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, Status::Ok);
        assert_eq!(decoded.advice.len(), 1);
        assert_eq!(decoded.advice[0].severity, Severity::Advisory);
    }

    #[test]
    fn unknown_recipe_in_infer_shape_yields_lints() {
        let step = PipelineStep::new(
            "bad",
            "Bad",
            RecipeRef::Recipe("not_a_recipe".into()),
        );
        let pipeline = Pipeline::new("test").with_step(step);
        let req = InvokeRequest::InferShape { pipeline };
        let resp = invoke(req);
        // Should succeed at the bridge level — the lint carries the warning.
        assert_eq!(resp.status, Status::Ok);
        assert!(!resp.lints.is_empty(), "expected a lint for unknown recipe");
    }
}
