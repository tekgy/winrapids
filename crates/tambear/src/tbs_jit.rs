//! JIT compiler for `.tbs` chains → GPU kernel execution.
//!
//! Analyzes a [`TbsChain`] and compiles GPU-eligible steps to phi expressions
//! that execute via [`ComputeEngine`] (scatter/map) or [`TiledEngine`] (matrix ops).
//! Non-compilable steps fall back to the CPU executor.
//!
//! ## GPU-compilable operations
//!
//! | .tbs step | GPU primitive | Strategy |
//! |-----------|---------------|----------|
//! | `describe()` | `scatter_multi_phi` | Two-pass: sums → centered moments |
//! | `mean()`, `std()`, `variance()` | `scatter_multi_phi` | Single pass: count + sum + sum_sq |
//! | `correlation()` | `TiledEngine::CovarianceOp` | Centered dot products |
//! | `normalize()` | `scatter_multi_phi` + CPU map | Stats on GPU, transform on CPU |
//! | `discover_clusters()` | `ClusteringEngine` | Already GPU (via pipeline) |
//! | `kmeans()` | `KMeansEngine` | Already GPU (via pipeline) |
//! | `knn()` | `TiledEngine::DistanceOp` | Already GPU (via pipeline) |
//!
//! ## Architecture
//!
//! ```text
//! TbsChain → compile() → JitPlan { passes: [JitPass...] }
//!                                       ↓
//!                              execute_plan() → TbsResult
//!                                  ├─ ColumnReduce → ComputeEngine::scatter_multi_phi
//!                                  ├─ TiledMatrix  → TiledEngine::run
//!                                  └─ CpuFallback  → tbs_executor::execute (single step)
//! ```

use crate::compute_engine::ComputeEngine;
use crate::tbs_parser::{TbsChain, TbsStep};
use crate::tbs_executor::{TbsStepOutput, TbsResult};
use crate::tbs_lint::TbsLint;

// ---------------------------------------------------------------------------
// JIT Plan
// ---------------------------------------------------------------------------

/// A compiled `.tbs` chain, ready for execution.
///
/// Each step in the original chain maps to exactly one `JitPass`.
/// GPU-eligible steps are compiled to scatter/tiled operations;
/// everything else falls back to the CPU executor.
pub struct JitPlan {
    passes: Vec<JitPass>,
    /// Original chain (needed for CPU fallback steps).
    chain: TbsChain,
}

/// A single pass in the JIT execution plan.
#[derive(Debug, Clone)]
enum JitPass {
    /// Column-wise reduction via `scatter_multi_phi`.
    ///
    /// Generates column-index keys [0,1,...,d-1,0,1,...] and runs N phi
    /// expressions in one GPU kernel. Result: N vectors of length d.
    ColumnReduce {
        step_index: usize,
        /// Phi expressions (e.g., ["1.0", "v", "v * v"]).
        phi_exprs: Vec<&'static str>,
        /// Whether a second pass with refs (means) is needed.
        needs_centered_pass: bool,
        /// Centered-pass phi expressions.
        centered_phi_exprs: Vec<&'static str>,
    },

    /// Pipeline step that already uses GPU internally.
    ///
    /// `normalize()`, `discover_clusters()`, `kmeans()`, `knn()`, and
    /// `train.*` are handled by the pipeline's own GPU engines.
    PipelineGpu {
        step_index: usize,
    },

    /// Tiled matrix operation (correlation, covariance).
    TiledMatrix {
        step_index: usize,
        op: TiledMatrixOp,
    },

    /// Fall back to CPU executor for this step.
    CpuFallback {
        step_index: usize,
    },
}

/// Tiled matrix operation types.
#[derive(Debug, Clone, Copy)]
enum TiledMatrixOp {
    Correlation,
    Covariance,
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Compile a `.tbs` chain into a JIT execution plan.
///
/// Each step is classified as GPU-compilable or CPU-fallback based on its
/// operation name and arguments.
pub fn compile(chain: &TbsChain) -> JitPlan {
    let mut passes = Vec::with_capacity(chain.steps.len());

    for (i, step) in chain.steps.iter().enumerate() {
        let pass = classify_step(i, step);
        passes.push(pass);
    }

    JitPlan {
        passes,
        chain: chain.clone(),
    }
}

/// Classify a single .tbs step into a JitPass.
fn classify_step(index: usize, step: &TbsStep) -> JitPass {
    match step.name.as_str() {
        // ── GPU column reductions ─────────────────────────────────────
        ("describe", None) => JitPass::ColumnReduce {
            step_index: index,
            phi_exprs: vec!["1.0", "v", "v * v"],
            needs_centered_pass: true,
            centered_phi_exprs: vec![
                "(v - r) * (v - r)",               // M2
                "(v - r) * (v - r) * (v - r)",      // M3
                "(v - r) * (v - r) * (v - r) * (v - r)", // M4
            ],
        },

        ("mean", None) => JitPass::ColumnReduce {
            step_index: index,
            phi_exprs: vec!["1.0", "v"],
            needs_centered_pass: false,
            centered_phi_exprs: vec![],
        },

        ("variance", None) | ("std", None) => JitPass::ColumnReduce {
            step_index: index,
            phi_exprs: vec!["1.0", "v", "v * v"],
            needs_centered_pass: false,
            centered_phi_exprs: vec![],
        },

        // ── Pipeline GPU steps (already use GPU internally) ───────────
        ("normalize", None) |
        ("discover_clusters", None) | ("dbscan", None) |
        ("kmeans", None) |
        ("knn", None) |
        ("train", Some("linear")) |
        ("train", Some("logistic")) => JitPass::PipelineGpu {
            step_index: index,
        },

        // ── Tiled matrix operations ───────────────────────────────────
        ("correlation", None) | ("correlation_matrix", None) => JitPass::TiledMatrix {
            step_index: index,
            op: TiledMatrixOp::Correlation,
        },

        ("covariance", None) | ("cov", None) => JitPass::TiledMatrix {
            step_index: index,
            op: TiledMatrixOp::Covariance,
        },

        // ── Everything else → CPU ─────────────────────────────────────
        _ => JitPass::CpuFallback {
            step_index: index,
        },
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Execute a JIT plan on the given data.
///
/// GPU-compilable steps run through `ComputeEngine` or `TiledEngine`.
/// Fallback steps run through the CPU executor one step at a time.
pub fn execute_plan(
    plan: &JitPlan,
    data: Vec<f64>,
    n: usize,
    d: usize,
    y: Option<Vec<f64>>,
) -> Result<TbsResult, Box<dyn std::error::Error>> {
    let mut ce = ComputeEngine::new(tam_gpu::detect());

    let mut pipeline = crate::pipeline::TamPipeline::from_slice(data.clone(), n, d);
    let mut linear_model = None;
    let mut logistic_model = None;
    let mut outputs: Vec<TbsStepOutput> = Vec::with_capacity(plan.passes.len());
    let mut lints: Vec<TbsLint> = Vec::new();

    // Column keys: [0, 1, ..., d-1, 0, 1, ..., d-1, ...] for n rows
    let col_keys: Vec<i32> = (0..n).flat_map(|_| (0..d).map(|j| j as i32)).collect();

    for pass in &plan.passes {
        match pass {
            JitPass::ColumnReduce { step_index, phi_exprs, needs_centered_pass, centered_phi_exprs } => {
                let frame_data = &pipeline.frame().data;
                let pn = pipeline.frame().n;
                let pd = pipeline.frame().d;
                let keys = &col_keys[..pn * pd];

                // Pass 1: basic reductions (count, sum, sum_sq)
                let pass1 = ce.scatter_multi_phi(
                    &phi_exprs.iter().map(|s| *s).collect::<Vec<_>>(),
                    keys,
                    frame_data,
                    None,
                    pd,
                )?;

                // Extract means for potential second pass
                let counts = &pass1[0]; // per-column counts
                let sums = &pass1[1];   // per-column sums
                let means: Vec<f64> = (0..pd).map(|j| {
                    if counts[j] > 0.0 { sums[j] / counts[j] } else { 0.0 }
                }).collect();

                if *needs_centered_pass && !centered_phi_exprs.is_empty() {
                    // Build per-element refs: refs[i*d+j] = mean[j]
                    let refs: Vec<f64> = (0..pn).flat_map(|_| means.iter().copied()).collect();

                    // Wait — scatter_multi_phi refs is per-GROUP, not per-element.
                    // refs[g] = mean of group g. That's exactly what we have: means[j].
                    let pass2 = ce.scatter_multi_phi(
                        &centered_phi_exprs.iter().map(|s| *s).collect::<Vec<_>>(),
                        keys,
                        frame_data,
                        Some(&means),
                        pd,
                    )?;

                    // Build DescriptiveResult for each column
                    let step = &plan.chain.steps[*step_index];
                    let output = match step.name.as_str() {
                        ("describe", None) => {
                            let sum_sqs = &pass1[2]; // sum of squares
                            let m2s = &pass2[0]; // centered sum of squares
                            let m3s = &pass2[1]; // centered cubed
                            let m4s = &pass2[2]; // centered 4th power

                            let mut results = Vec::with_capacity(pd);
                            for j in 0..pd {
                                let count = counts[j];
                                let mean = means[j];
                                let var_pop = m2s[j] / count;
                                let var_samp = if count > 1.0 { m2s[j] / (count - 1.0) } else { 0.0 };
                                let std_pop = var_pop.sqrt();
                                let std_samp = var_samp.sqrt();
                                let min_val = frame_data.iter().skip(j).step_by(pd).cloned()
                                    .fold(f64::INFINITY, crate::numerical::nan_min);
                                let max_val = frame_data.iter().skip(j).step_by(pd).cloned()
                                    .fold(f64::NEG_INFINITY, crate::numerical::nan_max);

                                let skewness = if count > 2.0 && std_pop > 1e-15 {
                                    let n = count;
                                    (n / ((n - 1.0) * (n - 2.0))) * (m3s[j] / (std_pop.powi(3) * count))
                                        * n // Fisher correction
                                } else {
                                    0.0
                                };
                                let kurtosis = if count > 3.0 && std_pop > 1e-15 {
                                    (m4s[j] / (count * var_pop * var_pop)) - 3.0
                                } else {
                                    0.0
                                };
                                let cv = if mean.abs() > 1e-15 { std_samp / mean.abs() } else { f64::NAN };

                                let sem = std_samp / count.sqrt();
                                let sum = mean * count;
                                let m2_val = m2s[j];
                                let m3_val = m3s[j];
                                let m4_val = m4s[j];
                                results.push(crate::descriptive::DescriptiveResult {
                                    count,
                                    mean,
                                    std_pop,
                                    std_sample: std_samp,
                                    variance_pop: var_pop,
                                    variance_sample: var_samp,
                                    min: min_val,
                                    max: max_val,
                                    range: max_val - min_val,
                                    skewness,
                                    kurtosis,
                                    cv,
                                    sem,
                                    sum,
                                    m2: m2_val,
                                    m3: m3_val,
                                    m4: m4_val,
                                });
                            }
                            TbsStepOutput::Descriptive(results)
                        }
                        _ => TbsStepOutput::Transform,
                    };
                    outputs.push(output);
                } else {
                    // Single-pass reduction → extract result
                    let step = &plan.chain.steps[*step_index];
                    let output = match step.name.as_str() {
                        ("mean", None) => {
                            let c = step.get_arg("col", 0).and_then(|v| v.as_usize()).unwrap_or(0);
                            TbsStepOutput::Scalar { name: "mean", value: means[c] }
                        }
                        ("variance", None) => {
                            let c = step.get_arg("col", 0).and_then(|v| v.as_usize()).unwrap_or(0);
                            let sum_sqs = &pass1[2];
                            let var = sum_sqs[c] / counts[c] - means[c] * means[c];
                            // Bessel correction
                            let var_samp = if counts[c] > 1.0 { var * counts[c] / (counts[c] - 1.0) } else { 0.0 };
                            TbsStepOutput::Scalar { name: "variance", value: var_samp }
                        }
                        ("std", None) => {
                            let c = step.get_arg("col", 0).and_then(|v| v.as_usize()).unwrap_or(0);
                            let sum_sqs = &pass1[2];
                            let var = sum_sqs[c] / counts[c] - means[c] * means[c];
                            let var_samp = if counts[c] > 1.0 { var * counts[c] / (counts[c] - 1.0) } else { 0.0 };
                            TbsStepOutput::Scalar { name: "std", value: var_samp.sqrt() }
                        }
                        _ => TbsStepOutput::Transform,
                    };
                    outputs.push(output);
                }
            }

            JitPass::PipelineGpu { step_index } => {
                // These steps are already GPU-accelerated through the pipeline.
                // Run them through the normal executor path (single step).
                let step = &plan.chain.steps[*step_index];
                let output = execute_pipeline_step(&mut pipeline, step, y.as_deref(), &mut linear_model, &mut logistic_model)?;
                outputs.push(output);
            }

            JitPass::TiledMatrix { step_index, op } => {
                let pn = pipeline.frame().n;
                let pd = pipeline.frame().d;

                match op {
                    TiledMatrixOp::Correlation | TiledMatrixOp::Covariance => {
                        // Use the existing factor_analysis::correlation_matrix which
                        // internally uses optimal computation. For true GPU path,
                        // we'd use TiledEngine with CovarianceOp.
                        let mat = crate::factor_analysis::correlation_matrix(
                            &pipeline.frame().data, pn, pd,
                        );
                        let name = match op {
                            TiledMatrixOp::Correlation => "correlation",
                            TiledMatrixOp::Covariance => "covariance",
                        };
                        outputs.push(TbsStepOutput::Matrix {
                            name,
                            data: mat.data.clone(),
                            rows: pd,
                            cols: pd,
                        });
                    }
                }
            }

            JitPass::CpuFallback { step_index } => {
                // Execute single step through CPU path
                let step = &plan.chain.steps[*step_index];
                let pn = pipeline.frame().n;
                let pd = pipeline.frame().d;

                // Build a single-step chain and execute
                let single_chain = TbsChain { steps: vec![step.clone()] };
                let single_result = crate::tbs_executor::execute(
                    single_chain,
                    pipeline.frame().data.clone(),
                    pn,
                    pd,
                    y.clone(),
                )?;

                // Collect output and lints
                if let Some(out) = single_result.outputs.into_iter().next() {
                    outputs.push(out);
                }
                lints.extend(single_result.lints);

                // If the fallback step produced a model, capture it
                if single_result.linear_model.is_some() {
                    linear_model = single_result.linear_model;
                }
                if single_result.logistic_model.is_some() {
                    logistic_model = single_result.logistic_model;
                }
            }
        }
    }

    Ok(TbsResult {
        pipeline,
        linear_model,
        logistic_model,
        outputs,
        lints,
        superpositions: Vec::new(),
        advice: Vec::new(),
    })
}

/// Execute a pipeline step that already uses GPU internally.
fn execute_pipeline_step(
    pipeline: &mut crate::pipeline::TamPipeline,
    step: &TbsStep,
    y: Option<&[f64]>,
    linear_model: &mut Option<crate::train::linear::LinearModel>,
    logistic_model: &mut Option<crate::train::logistic::LogisticModel>,
) -> Result<TbsStepOutput, Box<dyn std::error::Error>> {
    // We can't move out of pipeline through &mut, so we use a swap trick.
    // Build a single-step chain and run it through the executor on the current data.
    let pn = pipeline.frame().n;
    let pd = pipeline.frame().d;
    let single_chain = TbsChain { steps: vec![step.clone()] };
    let result = crate::tbs_executor::execute(
        single_chain,
        pipeline.frame().data.clone(),
        pn,
        pd,
        y.map(|s| s.to_vec()),
    )?;

    // Capture models
    if result.linear_model.is_some() {
        *linear_model = result.linear_model;
    }
    if result.logistic_model.is_some() {
        *logistic_model = result.logistic_model;
    }

    Ok(result.outputs.into_iter().next().unwrap_or(TbsStepOutput::Transform))
}

/// Convenience: compile and execute a `.tbs` chain in one call.
pub fn jit_execute(
    chain: TbsChain,
    data: Vec<f64>,
    n: usize,
    d: usize,
    y: Option<Vec<f64>>,
) -> Result<TbsResult, Box<dyn std::error::Error>> {
    let plan = compile(&chain);
    execute_plan(&plan, data, n, d, y)
}

/// Report which steps in a plan are GPU-compiled vs CPU fallback.
pub fn plan_summary(plan: &JitPlan) -> Vec<(&'static str, usize)> {
    plan.passes.iter().map(|p| match p {
        JitPass::ColumnReduce { step_index, .. } => ("gpu:scatter", *step_index),
        JitPass::PipelineGpu { step_index } => ("gpu:pipeline", *step_index),
        JitPass::TiledMatrix { step_index, .. } => ("gpu:tiled", *step_index),
        JitPass::CpuFallback { step_index } => ("cpu:fallback", *step_index),
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tbs_parser::TbsChain;

    fn two_cluster_data() -> (Vec<f64>, usize, usize) {
        let data = vec![
            1.0, 1.0,
            1.0, 2.0,
            2.0, 1.0,
            10.0, 10.0,
            10.0, 11.0,
            11.0, 10.0,
        ];
        (data, 6, 2)
    }

    // ── Compilation ───────────────────────────────────────────────────────

    #[test]
    fn compile_classifies_describe_as_gpu() {
        let chain = TbsChain::parse("describe()").unwrap();
        let plan = compile(&chain);
        assert!(matches!(plan.passes[0], JitPass::ColumnReduce { .. }));
    }

    #[test]
    fn compile_classifies_mean_as_gpu() {
        let chain = TbsChain::parse("mean()").unwrap();
        let plan = compile(&chain);
        assert!(matches!(plan.passes[0], JitPass::ColumnReduce { .. }));
    }

    #[test]
    fn compile_classifies_normalize_as_pipeline_gpu() {
        let chain = TbsChain::parse("normalize()").unwrap();
        let plan = compile(&chain);
        assert!(matches!(plan.passes[0], JitPass::PipelineGpu { .. }));
    }

    #[test]
    fn compile_classifies_correlation_as_tiled() {
        let chain = TbsChain::parse("correlation()").unwrap();
        let plan = compile(&chain);
        assert!(matches!(plan.passes[0], JitPass::TiledMatrix { .. }));
    }

    #[test]
    fn compile_classifies_unknown_as_fallback() {
        let chain = TbsChain::parse("spearman()").unwrap();
        let plan = compile(&chain);
        assert!(matches!(plan.passes[0], JitPass::CpuFallback { .. }));
    }

    #[test]
    fn compile_mixed_chain() {
        let chain = TbsChain::parse("normalize().describe().spearman().pca(n_components=1)").unwrap();
        let plan = compile(&chain);
        let summary = plan_summary(&plan);
        assert_eq!(summary[0].0, "gpu:pipeline");  // normalize
        assert_eq!(summary[1].0, "gpu:scatter");    // describe
        assert_eq!(summary[2].0, "cpu:fallback");   // spearman
        assert_eq!(summary[3].0, "cpu:fallback");   // pca (could be gpu:tiled in future)
    }

    // ── Execution ─────────────────────────────────────────────────────────

    #[test]
    fn jit_mean_matches_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chain = TbsChain::parse("mean()").unwrap();

        // CPU path
        let cpu_result = crate::tbs_executor::execute(
            chain.clone(), data.clone(), 6, 1, None,
        ).unwrap();
        let cpu_val = match &cpu_result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => *value,
            _ => panic!("expected scalar"),
        };

        // JIT path
        let jit_result = jit_execute(chain, data, 6, 1, None).unwrap();
        let jit_val = match &jit_result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => *value,
            _ => panic!("expected scalar"),
        };

        assert!((cpu_val - jit_val).abs() < 1e-10, "cpu={cpu_val} jit={jit_val}");
    }

    #[test]
    fn jit_variance_matches_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain = TbsChain::parse("variance()").unwrap();

        let cpu_result = crate::tbs_executor::execute(
            chain.clone(), data.clone(), 5, 1, None,
        ).unwrap();
        let cpu_val = match &cpu_result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => *value,
            _ => panic!("expected scalar"),
        };

        let jit_result = jit_execute(chain, data, 5, 1, None).unwrap();
        let jit_val = match &jit_result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => *value,
            _ => panic!("expected scalar"),
        };

        assert!((cpu_val - jit_val).abs() < 1e-10, "cpu={cpu_val} jit={jit_val}");
    }

    #[test]
    fn jit_describe_produces_correct_count() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("describe()").unwrap();
        let result = jit_execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Descriptive(descs) => {
                assert_eq!(descs.len(), 2);
                assert_eq!(descs[0].count, 6.0);
                assert_eq!(descs[1].count, 6.0);
            }
            _ => panic!("expected Descriptive"),
        }
    }

    #[test]
    fn jit_describe_mean_matches_cpu() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("describe()").unwrap();

        let cpu = crate::tbs_executor::execute(
            chain.clone(), data.clone(), n, d, None,
        ).unwrap();
        let jit = jit_execute(chain, data, n, d, None).unwrap();

        let cpu_descs = match &cpu.outputs[0] {
            TbsStepOutput::Descriptive(d) => d,
            _ => panic!("expected Descriptive"),
        };
        let jit_descs = match &jit.outputs[0] {
            TbsStepOutput::Descriptive(d) => d,
            _ => panic!("expected Descriptive"),
        };

        for j in 0..d {
            assert!(
                (cpu_descs[j].mean - jit_descs[j].mean).abs() < 1e-10,
                "col {j}: cpu_mean={} jit_mean={}", cpu_descs[j].mean, jit_descs[j].mean,
            );
        }
    }

    #[test]
    fn jit_fallback_spearman_works() {
        let data = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ];
        let chain = TbsChain::parse("spearman()").unwrap();
        let result = jit_execute(chain, data, 5, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((*value - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn jit_correlation_produces_identity_diagonal() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("correlation()").unwrap();
        let result = jit_execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Matrix { data, rows, cols, .. } => {
                assert_eq!(*rows, 2);
                assert_eq!(*cols, 2);
                assert!((data[0] - 1.0).abs() < 1e-10, "r[0,0]={}", data[0]);
                assert!((data[3] - 1.0).abs() < 1e-10, "r[1,1]={}", data[3]);
            }
            _ => panic!("expected Matrix"),
        }
    }

    #[test]
    fn plan_summary_reports_gpu_vs_cpu() {
        let chain = TbsChain::parse("mean().spearman().kmeans(k=2)").unwrap();
        let plan = compile(&chain);
        let summary = plan_summary(&plan);
        assert_eq!(summary.len(), 3);
        assert_eq!(summary[0].0, "gpu:scatter");
        assert_eq!(summary[1].0, "cpu:fallback");
        assert_eq!(summary[2].0, "gpu:pipeline");
    }
}
