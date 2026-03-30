//! Pipeline compiler: spec → execution plan.
//!
//! Five phases, matching E04:
//!   1. Decompose — expand specialist calls to primitive nodes
//!   2. Bind — substitute data variable names into input refs
//!   3. CSE — deduplicate by identity hash (built into Arena)
//!   4. Sort — topological order for execution
//!   5. Check — probe world state (provenance/residency/dirty)
//!
//! Injectable world state: provenance cache, dirty bitmap, residency map.
//! NullWorld by default (E04 baseline: compute everything).
//!
//! Connection 1 (Phase 4): real provenance flows from input data buffers
//! through the DAG via `provenance_hash(input_tags, node.identity)`.

use std::collections::{HashMap, HashSet};

use winrapids_store::world::WorldState;
use winrapids_store::provenance::{provenance_hash, data_provenance};

use crate::ir::{Arena, NodeId, PrimitiveOp};
use crate::registry::SpecialistRecipe;
use crate::topo::topo_sort;

/// One specialist call in the user's pipeline.
#[derive(Clone, Debug)]
pub struct SpecialistCall {
    /// Registry key (e.g. "rolling_zscore").
    pub specialist: String,
    /// Data variable name (identity proxy).
    pub data_var: String,
    /// Window size.
    pub window: u32,
}

/// User's pipeline specification.
#[derive(Clone, Debug)]
pub struct PipelineSpec {
    pub calls: Vec<SpecialistCall>,
}

/// One step in the execution plan.
#[derive(Clone, Debug)]
pub struct ExecStep {
    /// The node in the arena.
    pub node_id: NodeId,
    /// Input binding: local name → identity hash of the input.
    pub binding: HashMap<String, String>,
    /// Whether world state says we can skip this (provenance hit + clean + resident).
    pub skip: bool,
    /// The provenance tag for this step's output. Used by execute.rs
    /// to register results in the store and by downstream nodes to
    /// build their own provenance.
    pub provenance: [u8; 16],
}

/// CSE statistics.
#[derive(Clone, Debug)]
pub struct CseStats {
    pub original_nodes: usize,
    pub after_cse: usize,
    pub eliminated: usize,
}

/// Compiler output: ordered unique primitives + result mapping.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// The IR arena (owns all nodes).
    pub arena: Arena,
    /// Ordered execution steps (topological order).
    pub steps: Vec<ExecStep>,
    /// Output mapping: (call_idx, "out") → node identity.
    pub outputs: HashMap<(usize, String), String>,
    /// CSE statistics.
    pub cse_stats: CseStats,
}

/// Map string op names to PrimitiveOp enum.
fn parse_op(op: &str) -> PrimitiveOp {
    match op {
        "scan" => PrimitiveOp::Scan,
        "sort" => PrimitiveOp::Sort,
        "reduce" => PrimitiveOp::Reduce,
        "tiled_reduce" => PrimitiveOp::TiledReduce,
        "scatter" => PrimitiveOp::Scatter,
        "gather" => PrimitiveOp::Gather,
        "search" => PrimitiveOp::Search,
        "compact" => PrimitiveOp::Compact,
        "fused_expr" => PrimitiveOp::FusedExpr,
        other => panic!("Unknown primitive op: {}", other),
    }
}

/// Core compiler pass: spec → execution plan.
///
/// Phases:
///   1. Decompose specialist calls into primitive nodes
///   2. Bind data variable names into input references
///   3. CSE via Arena::add_or_dedup (identity hash deduplication)
///   4. Topological sort for correct execution order
///   5. World state probe (provenance/dirty/residency check)
///
/// `input_provenances` maps data variable names (e.g. "price") to their
/// real provenance tags from the data source. When None, provenance is
/// derived from the variable name string (E04 baseline behavior).
pub fn plan(
    spec: &PipelineSpec,
    registry: &HashMap<String, SpecialistRecipe>,
    world: &mut dyn WorldState,
    input_provenances: Option<&HashMap<String, [u8; 16]>>,
) -> ExecutionPlan {
    let mut arena = Arena::new();
    let mut output_map: HashMap<(usize, String), String> = HashMap::new();

    // Track all nodes before CSE (for stats)
    let mut total_nodes_before_cse = 0usize;

    // Track bindings per identity
    let mut bindings: HashMap<String, HashMap<String, String>> = HashMap::new();

    // --- Phase 1 + 2: Decompose and bind ---
    for (call_idx, call) in spec.calls.iter().enumerate() {
        let recipe = registry.get(&call.specialist)
            .unwrap_or_else(|| panic!("Unknown specialist: {}", call.specialist));

        // Within-specialist name → resolved identity
        let mut local_identity: HashMap<String, String> = HashMap::new();

        // Implicit inputs: "data" and "data_sq" are identity-based on the data variable
        let data_id = format!("data:{}", call.data_var);
        let data_sq_id = format!("data_sq:{}", call.data_var);
        local_identity.insert("data".into(), data_id);
        local_identity.insert("data_sq".into(), data_sq_id);

        for step in &recipe.primitive_dag {
            // Resolve input identities
            let input_ids: Vec<String> = step.input_names.iter()
                .map(|n| local_identity.get(n)
                    .unwrap_or_else(|| panic!("Unresolved input '{}' in specialist '{}'", n, call.specialist))
                    .clone())
                .collect();

            // Canonical params: recipe params + window
            let mut canonical_params: Vec<(String, String)> = step.params.clone();
            canonical_params.push(("window".into(), call.window.to_string()));
            canonical_params.sort();

            total_nodes_before_cse += 1;

            let node_id = arena.add_or_dedup(
                parse_op(&step.op),
                input_ids,
                canonical_params,
                step.output_name.clone(),
            );

            let node_identity = arena.get(node_id).identity.clone();

            // Store binding for this node
            let binding: HashMap<String, String> = step.input_names.iter()
                .map(|n| (n.clone(), local_identity[n].clone()))
                .collect();
            bindings.entry(node_identity.clone())
                .or_insert(binding);

            // Register output identity for downstream use within this specialist
            local_identity.insert(step.output_name.clone(), node_identity);
        }

        output_map.insert(
            (call_idx, "out".into()),
            local_identity["out"].clone(),
        );
    }

    let after_cse = arena.len();

    // --- Phase 4: Topological sort ---
    let all_identities: Vec<String> = arena.nodes.iter()
        .map(|n| n.identity.clone())
        .collect();

    let mut dep_graph: HashMap<String, HashSet<String>> = HashMap::new();
    for node in &arena.nodes {
        let mut deps = HashSet::new();
        for inp_id in &node.input_identities {
            if arena.get_by_identity(inp_id).is_some() {
                deps.insert(inp_id.clone());
            }
        }
        dep_graph.insert(node.identity.clone(), deps);
    }

    let ordered = topo_sort(&all_identities, &dep_graph);

    // --- Phase 5: World state probe with real provenance chain ---
    //
    // Build provenance tags that flow from input data through the DAG.
    // data_var "price" → data_provenance("price") or real tag from input_provenances.
    // Each computed node: provenance_hash([input_provs...], node.identity).

    // Resolve data variable provenance tags
    let mut data_var_provs: HashMap<String, [u8; 16]> = HashMap::new();
    for call in &spec.calls {
        if !data_var_provs.contains_key(&call.data_var) {
            let prov = match input_provenances {
                Some(map) => map.get(&call.data_var)
                    .copied()
                    .unwrap_or_else(|| data_provenance(&call.data_var)),
                None => data_provenance(&call.data_var),
            };
            data_var_provs.insert(call.data_var.clone(), prov);
        }
    }

    // Map input string identities (e.g. "data:price", "data_sq:price") to provenance tags
    let mut identity_provs: HashMap<String, [u8; 16]> = HashMap::new();
    for (var, prov) in &data_var_provs {
        let data_id = format!("data:{}", var);
        let data_sq_id = format!("data_sq:{}", var);
        identity_provs.insert(data_id.clone(), *prov);
        // data_sq has its own provenance derived from the data provenance
        identity_provs.insert(data_sq_id, provenance_hash(&[*prov], "square"));
    }

    let steps: Vec<ExecStep> = ordered.iter()
        .filter_map(|identity| {
            let node_id = arena.get_by_identity(identity)?;
            let binding = bindings.get(identity).cloned().unwrap_or_default();
            let node = arena.get(node_id);

            // Build provenance from real input provenance tags
            let input_provs: Vec<[u8; 16]> = node.input_identities.iter()
                .map(|inp_id| {
                    // Look up the provenance for this input:
                    // either a data leaf or a previously computed node
                    identity_provs.get(inp_id)
                        .copied()
                        .unwrap_or_else(|| {
                            // Fallback: hash the identity string (shouldn't happen
                            // if topo order is correct)
                            data_provenance(inp_id)
                        })
                })
                .collect();

            let prov = provenance_hash(&input_provs, &node.identity);

            // Register this node's provenance for downstream use
            identity_provs.insert(node.identity.clone(), prov);

            // Probe world state: can we skip?
            // Navigator's simplified hot path: just provenance_get.
            // If it returns Some, the result is on GPU (Phase 3 invariant).
            let skip = world.provenance_get(&prov).is_some();

            Some(ExecStep { node_id, binding, skip, provenance: prov })
        })
        .collect();

    ExecutionPlan {
        arena,
        steps,
        outputs: output_map,
        cse_stats: CseStats {
            original_nodes: total_nodes_before_cse,
            after_cse,
            eliminated: total_nodes_before_cse - after_cse,
        },
    }
}
