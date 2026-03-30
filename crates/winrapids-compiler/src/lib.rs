//! # winrapids-compiler
//!
//! Pipeline compiler: spec → primitive DAG → CSE → execution plan.
//!
//! The sharing optimizer's brain. Every design decision asks:
//! "does this increase sharing?"
//!
//! - CSE finds shared primitives across specialist boundaries (2x)
//! - Provenance reuse skips already-computed results (865x)
//! - Fusion reduces kernel launches (2.3x fewer)
//!
//! # Usage
//!
//! ```no_run
//! use winrapids_compiler::{PipelineSpec, SpecialistCall, plan};
//! use winrapids_compiler::registry::build_e04_registry;
//! use winrapids_store::world::NullWorld;
//!
//! let spec = PipelineSpec {
//!     calls: vec![
//!         SpecialistCall { specialist: "rolling_zscore".into(), data_var: "price".into(), window: 20 },
//!         SpecialistCall { specialist: "rolling_std".into(), data_var: "price".into(), window: 20 },
//!     ],
//! };
//! let registry = build_e04_registry();
//! let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
//! // CSE eliminated 2 of 6 nodes (33%)
//! assert_eq!(exec_plan.cse_stats.eliminated, 2);
//! ```

pub mod ir;
pub mod registry;
pub mod plan;
pub mod topo;
pub mod execute;
pub mod cuda_dispatch;

pub use ir::{Arena, NodeId, PrimitiveOp, Node};
pub use plan::{PipelineSpec, SpecialistCall, ExecutionPlan, ExecStep, CseStats, plan};
pub use registry::{SpecialistRecipe, PrimitiveStep, build_e04_registry};
pub use execute::{execute, KernelDispatcher, MockDispatcher, StepResult, ExecuteStats};
pub use cuda_dispatch::CudaKernelDispatcher;
