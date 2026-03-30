//! PyO3 bindings for WinRapids.
//!
//! The 10-line pipeline API: lazy graph building in Python,
//! compiled execution in Rust.
//!
//! ```python
//! import winrapids as wr
//!
//! pipe = wr.Pipeline()
//! pipe.add("rolling_zscore", data="price", window=20)
//! pipe.add("rolling_std",    data="price", window=20)
//! plan = pipe.compile()
//!
//! print(plan.cse_stats)
//! # {'original_nodes': 6, 'after_cse': 4, 'eliminated': 2}
//!
//! # Execute with mock dispatcher (GPU dispatch comes with cudarc integration)
//! results = pipe.execute({"price": (0x100, 8000)})
//! print(results)  # {'rolling_zscore_out': (ptr, size), ...}
//! ```

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use winrapids_compiler::plan::{self, PipelineSpec, SpecialistCall};
use winrapids_compiler::execute::{self, MockDispatcher};
use winrapids_compiler::registry::build_e04_registry;
use winrapids_store::header::BufferPtr;
use winrapids_store::provenance::data_provenance;
use winrapids_store::store::GpuStore;
use winrapids_store::world::NullWorld;

/// A lazy pipeline builder. Add specialist calls, then compile or execute.
#[pyclass]
#[derive(Clone)]
struct Pipeline {
    calls: Vec<SpecialistCall>,
}

#[pymethods]
impl Pipeline {
    #[new]
    fn new() -> Self {
        Pipeline { calls: Vec::new() }
    }

    /// Add a specialist call to the pipeline.
    #[pyo3(signature = (specialist, data, window=20))]
    fn add(&mut self, specialist: String, data: String, window: u32) -> PyResult<()> {
        self.calls.push(SpecialistCall {
            specialist,
            data_var: data,
            window,
        });
        Ok(())
    }

    /// Compile the pipeline. Returns an execution plan with CSE optimization.
    fn compile(&self) -> PyResult<Plan> {
        let spec = PipelineSpec { calls: self.calls.clone() };
        let registry = build_e04_registry();
        let exec_plan = plan::plan(&spec, &registry, &mut NullWorld, None);
        Ok(Plan::from_exec_plan(&exec_plan))
    }

    /// Execute the pipeline with mock dispatch (validates the full path).
    ///
    /// Args:
    ///     data: dict mapping data variable names to (device_ptr, byte_size) tuples.
    ///           e.g. {"price": (0x100, 8000), "volume": (0x200, 8000)}
    ///
    /// Returns a dict with execution results:
    ///     - "plan": the compiled Plan
    ///     - "stats": {"hits": N, "misses": N, "hit_rate": float}
    ///     - "outputs": list of (call_idx, output_name, device_ptr, byte_size)
    #[pyo3(signature = (data, use_store=false))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyDict>,
        use_store: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let spec = PipelineSpec { calls: self.calls.clone() };
        let registry = build_e04_registry();

        // Parse data dict → BufferPtrs + provenance tags
        let mut data_ptrs: HashMap<String, BufferPtr> = HashMap::new();
        let mut input_provs: HashMap<String, [u8; 16]> = HashMap::new();

        for (key, value) in data.iter() {
            let name: String = key.extract()?;
            let (ptr, size): (u64, u64) = value.extract()?;

            // Register both data:name and data_sq:name
            let data_id = format!("data:{}", name);
            let data_sq_id = format!("data_sq:{}", name);
            data_ptrs.insert(data_id, BufferPtr { device_ptr: ptr, byte_size: size });
            // data_sq pointer: in real execution this would be computed;
            // for mock, just use a placeholder offset
            data_ptrs.insert(data_sq_id, BufferPtr { device_ptr: ptr + size, byte_size: size });

            input_provs.insert(name, data_provenance(&format!("data:{}:{}", key.str()?, ptr)));
        }

        // Choose world state: GpuStore (persistent) or NullWorld (compute everything)
        let result_dict = PyDict::new(py);

        if use_store {
            let mut store = GpuStore::new(60_000_000_000); // 60GB VRAM ceiling
            let exec_plan = plan::plan(&spec, &registry, &mut store, Some(&input_provs));
            result_dict.set_item("plan", Plan::from_exec_plan(&exec_plan))?;

            let mut dispatcher = MockDispatcher::new();
            let (results, stats) = execute::execute(&exec_plan, &mut store, &mut dispatcher, &data_ptrs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            Self::populate_results(py, &result_dict, &exec_plan, &results, &stats)?;
        } else {
            let exec_plan = plan::plan(&spec, &registry, &mut NullWorld, Some(&input_provs));
            result_dict.set_item("plan", Plan::from_exec_plan(&exec_plan))?;

            let mut dispatcher = MockDispatcher::new();
            let (results, stats) = execute::execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            Self::populate_results(py, &result_dict, &exec_plan, &results, &stats)?;
        }

        Ok(result_dict)
    }

    fn __repr__(&self) -> String {
        format!("Pipeline({} calls)", self.calls.len())
    }

    fn __len__(&self) -> usize {
        self.calls.len()
    }
}

impl Pipeline {
    fn populate_results<'py>(
        py: Python<'py>,
        result_dict: &Bound<'py, PyDict>,
        exec_plan: &winrapids_compiler::ExecutionPlan,
        results: &HashMap<String, execute::StepResult>,
        stats: &execute::ExecuteStats,
    ) -> PyResult<()> {
        // Stats
        let stats_dict = PyDict::new(py);
        stats_dict.set_item("hits", stats.hits)?;
        stats_dict.set_item("misses", stats.misses)?;
        stats_dict.set_item("hit_rate", stats.hit_rate())?;
        stats_dict.set_item("total_compute_us", stats.total_compute_us)?;
        result_dict.set_item("stats", stats_dict)?;

        // Outputs
        let mut outputs = Vec::new();
        for ((call_idx, out_name), node_id) in &exec_plan.outputs {
            if let Some(result) = results.get(node_id) {
                outputs.push((*call_idx, out_name.clone(), result.ptr.device_ptr, result.ptr.byte_size, result.was_hit));
            }
        }
        result_dict.set_item("outputs", outputs)?;

        Ok(())
    }
}

/// One step in the compiled execution plan.
#[pyclass]
#[derive(Clone)]
struct StepInfo {
    #[pyo3(get)]
    op: String,
    #[pyo3(get)]
    output_name: String,
    #[pyo3(get)]
    identity: String,
    #[pyo3(get)]
    inputs: Vec<String>,
    #[pyo3(get)]
    skip: bool,
}

#[pymethods]
impl StepInfo {
    fn __repr__(&self) -> String {
        format!("[{:12}] {:<6} id={} skip={}",
            self.op, self.output_name, &self.identity[..8.min(self.identity.len())], self.skip)
    }
}

/// Compiled execution plan -- the compiler's output.
#[pyclass]
struct Plan {
    #[pyo3(get)]
    original_nodes: usize,
    #[pyo3(get)]
    after_cse: usize,
    #[pyo3(get)]
    eliminated: usize,
    #[pyo3(get)]
    steps: Vec<StepInfo>,
    #[pyo3(get)]
    outputs: Vec<(usize, String, String)>,
}

impl Plan {
    fn from_exec_plan(exec_plan: &winrapids_compiler::ExecutionPlan) -> Self {
        let steps: Vec<StepInfo> = exec_plan.steps.iter()
            .map(|step| {
                let node = exec_plan.arena.get(step.node_id);
                StepInfo {
                    op: format!("{:?}", node.op),
                    output_name: node.output_name.clone(),
                    identity: node.identity.clone(),
                    inputs: node.input_identities.clone(),
                    skip: step.skip,
                }
            })
            .collect();

        Plan {
            original_nodes: exec_plan.cse_stats.original_nodes,
            after_cse: exec_plan.cse_stats.after_cse,
            eliminated: exec_plan.cse_stats.eliminated,
            steps,
            outputs: exec_plan.outputs.iter()
                .map(|((idx, name), identity)| (*idx, name.clone(), identity.clone()))
                .collect(),
        }
    }
}

#[pymethods]
impl Plan {
    /// CSE statistics as a dict.
    #[getter]
    fn cse_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("original_nodes", self.original_nodes)?;
        dict.set_item("after_cse", self.after_cse)?;
        dict.set_item("eliminated", self.eliminated)?;
        dict.set_item("elimination_pct",
            if self.original_nodes > 0 {
                100 * self.eliminated / self.original_nodes
            } else { 0 }
        )?;
        Ok(dict)
    }

    fn __len__(&self) -> usize {
        self.steps.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Plan({} steps, CSE: {} -> {} ({} eliminated))",
            self.steps.len(), self.original_nodes, self.after_cse, self.eliminated
        )
    }
}

/// List available specialists in the E04 registry.
#[pyfunction]
fn list_specialists() -> Vec<String> {
    let reg = build_e04_registry();
    let mut names: Vec<String> = reg.keys().cloned().collect();
    names.sort();
    names
}

/// Get primitive decomposition for a specialist.
#[pyfunction]
fn specialist_dag(name: &str) -> PyResult<Vec<(String, String, Vec<String>)>> {
    let reg = build_e04_registry();
    let recipe = reg.get(name)
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
            format!("Unknown specialist: {}", name)))?;

    Ok(recipe.primitive_dag.iter()
        .map(|step| (
            step.output_name.clone(),
            step.op.clone(),
            step.input_names.clone(),
        ))
        .collect())
}

/// The WinRapids native module.
#[pymodule]
fn _winrapids_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    m.add_class::<Plan>()?;
    m.add_class::<StepInfo>()?;
    m.add_function(wrap_pyfunction!(list_specialists, m)?)?;
    m.add_function(wrap_pyfunction!(specialist_dag, m)?)?;
    Ok(())
}
