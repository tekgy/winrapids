//! Pipelines — Layer 4 of the atoms/primitives/recipes/pipelines architecture.
//!
//! A pipeline is an ordered sequence of recipe invocations, each with its
//! own parameter bindings. Pipelines are the unit of user-facing work:
//! "run exploratory factor analysis," "run clustering with diagnostics,"
//! "run the full time-series health check."
//!
//! # What belongs in the pipeline layer
//!
//! - **Sequencing**: the list of steps and their order.
//! - **Parameter bindings**: which choices at each step are forced
//!   (`Using`), auto-selected (`Autodiscover`), swept (`Sweep`), held in
//!   superposition (`Superposition`), or taken from tambear's default
//!   (`Default`).
//! - **Shape inference**: given a pipeline spec (without any data), compute
//!   what each step's output columns look like. This lets the IDE render
//!   the chain graph before the user runs the pipeline — it knows the
//!   semantic names, types, and dimensions of every intermediate result.
//! - **Serialization**: save a pipeline in a round-trippable format. When
//!   a user saves their own pipeline, any parameter they *kept from the
//!   tambear default* is explicitly materialized as a `Using` binding, so
//!   nothing is implicit and later tambear default changes cannot silently
//!   alter the user's saved work.
//!
//! # What does NOT belong here
//!
//! The pipeline types in this module are pure **specification**. They
//! describe what to do; they do not run anything, touch hardware,
//! allocate device memory, or bind to columns in a user's data. Three
//! separate crates own the layers around this one:
//!
//! - **`tambear-ide`** — the data binding and GUI layer. Owns the
//!   A####/B#### labeling of intermediate outputs, the relabel-on-reorder
//!   behavior, the graph visualization, the live results panel, the
//!   dropdowns and sliders rendered for each [`Binding`] variant, the
//!   drag-and-drop pipeline editor, the academic writeup attachment.
//!   Reads this module's shape info ([`OutputColumn`]) to know what
//!   labels and references are possible, but doesn't extend the types
//!   with widget metadata.
//!
//! - **Execution / orchestration crate(s)** — compile, run, report,
//!   storage, memory, cross-hardware scheduling, TAM above the Fock
//!   boundary. Takes a [`Pipeline`] spec and user data, produces results.
//!   Handles intermediate caching, Kingdom A/B/C/D lifting, GPU/CPU/NPU
//!   placement, wavefront scheduling. None of that appears in this
//!   module.
//!
//! - **Widget metadata / per-parameter UI rules** — the IDE decides how
//!   to render a `Binding::Using(Value::Method("varimax"))`: dropdown,
//!   radio buttons, autocomplete textbox. The pipeline type just says
//!   "this is the binding"; the rendering choice belongs to the IDE.
//!
//! The rule of thumb: if a concept requires hardware, network, or UI
//! knowledge, it does not belong in this crate at all.
//!
//! # The four hooks plus default
//!
//! Every choice point in every step has a `Binding`:
//!
//! - [`Binding::Using`]: user-forced value. Saved explicitly, survives
//!   tambear default changes.
//! - [`Binding::Autodiscover`]: auto-select based on a data probe at
//!   execution time. The IDE surfaces this as "Tambear picked X because
//!   the data had property Y."
//! - [`Binding::Sweep`]: run the step once per value and return a
//!   result-per-value. Used for grid search, hyperparameter exploration.
//! - [`Binding::Superposition`]: run the step across every value in
//!   parallel, combine results via a specified combiner function without
//!   collapsing. Used for ensemble / Bayesian model averaging / discover
//!   workflows.
//! - [`Binding::Default`]: take the tambear default. At save time, this
//!   is materialized into whichever of the above four types the default
//!   resolves to, so the saved pipeline is explicit about every choice.

pub mod efa;
pub mod schema;
pub mod types;

pub use schema::{
    effective_binding, schema_for, step_from_schema, DefaultBinding, DefaultValue,
    DimensionSourceSpec, OutputShapeSpec, OutputSpec, ParameterSpec, RecipeSchema, ValueDomain,
};
pub use types::{
    Binding, ChoiceKey, Combiner, DataProbeRef, DimensionSource, Dtype, OutputColumn, OutputShape,
    Pipeline, PipelineStep, RecipeRef, SuperpositionSpec, SweepSpec, Value,
};
