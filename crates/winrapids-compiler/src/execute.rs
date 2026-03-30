//! Execution engine: walks the plan, dispatches kernels, routes pointers.
//!
//! This is where the 865x lives. The hot path:
//!
//! ```text
//! let prov = step.provenance;
//! if let Some(ptr) = world.provenance_get(&prov) {
//!     // HIT: pointer handoff, zero computation
//! } else {
//!     // MISS: dispatch kernel, register result
//! }
//! ```
//!
//! Per-node overhead: BLAKE3 (~100ns) + HashMap lookup (O(1)) + branch.
//! On a provenance hit, no kernel launches — pointer routing only.
//!
//! Phase 3 invariant: the store only holds GPU-resident entries, so
//! `provenance_get` returning Some means the result is on GPU.
//! When spill-to-pinned is added (future), lookup will need to check
//! `is_resident` before using the raw device pointer.

use std::collections::HashMap;
use std::time::Instant;

use winrapids_store::header::BufferPtr;
use winrapids_store::world::WorldState;

use crate::ir::PrimitiveOp;
use crate::plan::ExecutionPlan;

/// Result of executing a plan step.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// The provenance tag for this result.
    pub provenance: [u8; 16],
    /// The device pointer + size for this result.
    pub ptr: BufferPtr,
    /// Whether this was a store hit (skipped computation).
    pub was_hit: bool,
    /// Compute time in microseconds (0 for hits).
    pub compute_us: f32,
}

/// Execution statistics.
#[derive(Clone, Debug, Default)]
pub struct ExecuteStats {
    pub hits: u64,
    pub misses: u64,
    pub total_compute_us: f64,
    pub total_steps: u64,
}

impl ExecuteStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Trait for kernel dispatch. The executor calls this for each MISS step.
///
/// Different backends implement this:
/// - GPU backend: cudarc kernel launch
/// - CPU backend: reference implementation (testing)
/// - Mock backend: identity function (validation)
pub trait KernelDispatcher {
    /// Dispatch a single computation step.
    ///
    /// Args:
    ///   - op: which primitive operation
    ///   - params: canonical parameters (e.g. [("agg", "add"), ("window", "20")])
    ///   - input_ptrs: device pointers for each input (in input_identities order)
    ///   - input_sizes: byte sizes for each input
    ///
    /// Returns: the output buffer pointer.
    fn dispatch(
        &mut self,
        op: &PrimitiveOp,
        params: &[(String, String)],
        input_ptrs: &[BufferPtr],
    ) -> Result<BufferPtr, Box<dyn std::error::Error>>;
}

/// Execute a compiled plan against a world state and kernel dispatcher.
///
/// Walks steps in topological order. For each step:
/// 1. Check provenance in world state (store lookup)
/// 2. HIT → route pointer (zero computation)
/// 3. MISS → dispatch kernel → register result in store
///
/// Returns: per-step results keyed by node identity, plus stats.
pub fn execute(
    plan: &ExecutionPlan,
    world: &mut dyn WorldState,
    dispatcher: &mut dyn KernelDispatcher,
    data_ptrs: &HashMap<String, BufferPtr>,
) -> Result<(HashMap<String, StepResult>, ExecuteStats), Box<dyn std::error::Error>> {
    let mut results: HashMap<String, StepResult> = HashMap::new();
    let mut stats = ExecuteStats::default();

    // Seed data leaf pointers into results (these are the raw inputs)
    for (data_id, ptr) in data_ptrs {
        results.insert(data_id.clone(), StepResult {
            provenance: [0; 16], // Data leaves have their own provenance from plan()
            ptr: *ptr,
            was_hit: true, // Data is always "available"
            compute_us: 0.0,
        });
    }

    for step in &plan.steps {
        let node = plan.arena.get(step.node_id);
        let prov = step.provenance;
        stats.total_steps += 1;

        // Probe world state: do we already have this result?
        if let Some(ptr) = world.provenance_get(&prov) {
            // HIT: pointer handoff, zero computation
            results.insert(node.identity.clone(), StepResult {
                provenance: prov,
                ptr,
                was_hit: true,
                compute_us: 0.0,
            });
            stats.hits += 1;
            continue;
        }

        // MISS: gather input pointers, dispatch kernel
        let input_ptrs: Vec<BufferPtr> = node.input_identities.iter()
            .map(|inp_id| {
                results.get(inp_id)
                    .map(|r| r.ptr)
                    .unwrap_or_else(|| {
                        // Try data_ptrs directly (for data leaf identities like "data:price")
                        data_ptrs.get(inp_id)
                            .copied()
                            .unwrap_or_else(|| panic!(
                                "Missing input '{}' for node '{}' ({:?})",
                                inp_id, node.output_name, node.op
                            ))
                    })
            })
            .collect();

        let t0 = Instant::now();
        let ptr = dispatcher.dispatch(&node.op, &node.params, &input_ptrs)?;
        let compute_us = t0.elapsed().as_micros() as f32;

        // Register in store
        world.provenance_put(prov, ptr, compute_us);

        results.insert(node.identity.clone(), StepResult {
            provenance: prov,
            ptr,
            was_hit: false,
            compute_us,
        });
        stats.misses += 1;
        stats.total_compute_us += compute_us as f64;
    }

    Ok((results, stats))
}

/// Mock dispatcher for testing: returns dummy pointers with incrementing addresses.
/// Validates execution flow without GPU.
pub struct MockDispatcher {
    next_addr: u64,
    pub dispatch_log: Vec<(PrimitiveOp, Vec<(String, String)>)>,
}

impl MockDispatcher {
    pub fn new() -> Self {
        Self {
            next_addr: 0x1000,
            dispatch_log: Vec::new(),
        }
    }
}

impl KernelDispatcher for MockDispatcher {
    fn dispatch(
        &mut self,
        op: &PrimitiveOp,
        params: &[(String, String)],
        _input_ptrs: &[BufferPtr],
    ) -> Result<BufferPtr, Box<dyn std::error::Error>> {
        self.dispatch_log.push((op.clone(), params.to_vec()));
        let addr = self.next_addr;
        self.next_addr += 0x1000;
        Ok(BufferPtr {
            device_ptr: addr,
            byte_size: 8000, // 1000 f64s
        })
    }
}
