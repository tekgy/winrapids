//! Specialist registry.
//!
//! Each specialist is a recipe: a name, a primitive DAG, and metadata.
//! The compiler decomposes specialist calls into primitive nodes using these recipes.
//!
//! Registry is a HashMap<String, SpecialistRecipe> — add specialists by inserting.

use std::collections::HashMap;

/// One step in a specialist's primitive DAG.
#[derive(Clone, Debug)]
pub struct PrimitiveStep {
    /// Output name within this specialist (e.g. "cs", "out").
    pub output_name: String,
    /// Primitive operation name (e.g. "scan", "fused_expr").
    pub op: String,
    /// Input names within this specialist (e.g. ["data"], ["data", "cs", "cs2"]).
    pub input_names: Vec<String>,
    /// Parameters (e.g. [("agg", "add"), ("formula", "rolling_zscore")]).
    pub params: Vec<(String, String)>,
}

/// One row in the specialist registry.
#[derive(Clone, Debug)]
pub struct SpecialistRecipe {
    pub name: String,
    /// Ordered list of primitive steps.
    pub primitive_dag: Vec<PrimitiveStep>,
    /// Whether fused_expr nodes can be kernel-fused.
    pub fusion_eligible: bool,
    /// Row count crossover for fusion (anti-YAGNI slot).
    pub fusion_crossover_rows: u64,
    /// True if the specialist has no scan/sort/reduce (purely element-wise).
    pub independent: bool,
    /// Parameters that constitute canonical identity.
    pub identity_params: Vec<String>,
}

/// Build the E04 minimal registry: rolling_mean, rolling_std, rolling_zscore,
/// and kalman_filter (steady-state Kalman via affine scan).
///
/// kalman_filter uses a single `scan` with `agg=kalman_affine`. The DARE
/// constants (F, H, Q, R) are baked into the kernel at construction time,
/// so no fused_expr step is needed — the scan output IS the filtered signal.
pub fn build_e04_registry() -> HashMap<String, SpecialistRecipe> {
    let mut reg = HashMap::new();

    reg.insert("rolling_mean".into(), SpecialistRecipe {
        name: "rolling_mean".into(),
        primitive_dag: vec![
            PrimitiveStep {
                output_name: "cs".into(),
                op: "scan".into(),
                input_names: vec!["data".into()],
                params: vec![("agg".into(), "add".into())],
            },
            PrimitiveStep {
                output_name: "out".into(),
                op: "fused_expr".into(),
                input_names: vec!["data".into(), "cs".into()],
                params: vec![("formula".into(), "rolling_mean".into())],
            },
        ],
        fusion_eligible: true,
        fusion_crossover_rows: u64::MAX,
        independent: false,
        identity_params: vec!["data_identity".into(), "window".into()],
    });

    reg.insert("rolling_std".into(), SpecialistRecipe {
        name: "rolling_std".into(),
        primitive_dag: vec![
            PrimitiveStep {
                output_name: "cs".into(),
                op: "scan".into(),
                input_names: vec!["data".into()],
                params: vec![("agg".into(), "add".into())],
            },
            PrimitiveStep {
                output_name: "cs2".into(),
                op: "scan".into(),
                input_names: vec!["data_sq".into()],
                params: vec![("agg".into(), "add".into())],
            },
            PrimitiveStep {
                output_name: "out".into(),
                op: "fused_expr".into(),
                input_names: vec!["data".into(), "cs".into(), "cs2".into()],
                params: vec![("formula".into(), "rolling_std".into())],
            },
        ],
        fusion_eligible: true,
        fusion_crossover_rows: u64::MAX,
        independent: false,
        identity_params: vec!["data_identity".into(), "window".into()],
    });

    reg.insert("rolling_zscore".into(), SpecialistRecipe {
        name: "rolling_zscore".into(),
        primitive_dag: vec![
            PrimitiveStep {
                output_name: "cs".into(),
                op: "scan".into(),
                input_names: vec!["data".into()],
                params: vec![("agg".into(), "add".into())],
            },
            PrimitiveStep {
                output_name: "cs2".into(),
                op: "scan".into(),
                input_names: vec!["data_sq".into()],
                params: vec![("agg".into(), "add".into())],
            },
            PrimitiveStep {
                output_name: "out".into(),
                op: "fused_expr".into(),
                input_names: vec!["data".into(), "cs".into(), "cs2".into()],
                params: vec![("formula".into(), "rolling_zscore".into())],
            },
        ],
        fusion_eligible: true,
        fusion_crossover_rows: u64::MAX,
        independent: false,
        identity_params: vec!["data_identity".into(), "window".into()],
    });

    reg.insert("kalman_filter".into(), SpecialistRecipe {
        name: "kalman_filter".into(),
        primitive_dag: vec![
            PrimitiveStep {
                output_name: "out".into(),
                op: "scan".into(),
                input_names: vec!["data".into()],
                params: vec![
                    ("agg".into(), "kalman_affine".into()),
                    ("F".into(), "0.98".into()),
                    ("H".into(), "1.0".into()),
                    ("Q".into(), "0.01".into()),
                    ("R".into(), "0.1".into()),
                ],
            },
        ],
        fusion_eligible: false,
        fusion_crossover_rows: u64::MAX,
        independent: false,
        identity_params: vec![
            "data_identity".into(),
            "F".into(), "H".into(), "Q".into(), "R".into(),
        ],
    });

    reg
}
