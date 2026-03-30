//! Compiler validation — reproduces E04's CSE test.
//!
//! Pipeline: rolling_zscore(price, w=20) + rolling_std(price, w=20)
//!
//! Without CSE: 6 primitive nodes (2 scans + 1 fused_expr per specialist)
//! With CSE:    4 primitive nodes (shared scan(price,add) and scan(price_sq,add))
//! Eliminated:  2 nodes (33%)
//!
//! This is the sharing optimizer finding shared computation automatically.

use std::collections::HashMap;

use winrapids_compiler::ir::PrimitiveOp;
use winrapids_compiler::plan::{PipelineSpec, SpecialistCall, plan};
use winrapids_compiler::execute::{execute, MockDispatcher};
use winrapids_compiler::cuda_dispatch::CudaKernelDispatcher;
use winrapids_scan::DevicePtr;
use winrapids_compiler::registry::build_e04_registry;
use winrapids_store::header::BufferPtr;
use winrapids_store::store::GpuStore;
use winrapids_store::world::NullWorld;
use winrapids_store::provenance::data_provenance;

fn main() {
    println!("{}", "=".repeat(70));
    println!("winrapids-compiler validation");
    println!("{}", "=".repeat(70));

    test_cse_sharing();
    test_single_specialist();
    test_independent_data();
    test_topo_order();
    test_world_state_probe();
    test_execute_mock();
    test_provenance_reuse();
    test_cuda_dispatch();
    test_fused_expr_numerics();

    println!("\n{}", "=".repeat(70));
    println!("ALL COMPILER TESTS PASSED");
    println!("{}", "=".repeat(70));
}

fn test_cse_sharing() {
    println!("\n--- Test 1: CSE sharing (E04 reproduction) ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
            SpecialistCall {
                specialist: "rolling_std".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    println!("  Pipeline: rolling_zscore(price, w=20) + rolling_std(price, w=20)");
    println!("  CSE stats:");
    println!("    Original nodes : {}", exec_plan.cse_stats.original_nodes);
    println!("    After CSE      : {}", exec_plan.cse_stats.after_cse);
    println!("    Eliminated     : {}", exec_plan.cse_stats.eliminated);

    assert_eq!(exec_plan.cse_stats.original_nodes, 6,
        "Should have 6 original nodes (3 per specialist)");
    assert_eq!(exec_plan.cse_stats.after_cse, 4,
        "Should have 4 nodes after CSE (2 shared scans)");
    assert_eq!(exec_plan.cse_stats.eliminated, 2,
        "Should eliminate 2 nodes (scan(price,add) + scan(price_sq,add))");

    println!("\n  Execution steps ({} total):", exec_plan.steps.len());
    for step in &exec_plan.steps {
        let node = exec_plan.arena.get(step.node_id);
        let binding_vals: Vec<&str> = step.binding.values().map(|s| s.as_str()).collect();
        println!("    [{:12?}]  {:<6}  inputs={:?}  id={}  skip={}",
            node.op, node.output_name, binding_vals, &node.identity[..8], step.skip);
    }

    // Verify structure
    assert_eq!(exec_plan.steps.len(), 4, "Should have 4 execution steps");
    println!("  PASS");
}

fn test_single_specialist() {
    println!("\n--- Test 2: Single specialist (no sharing opportunity) ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_mean".into(),
                data_var: "price".into(),
                window: 10,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    assert_eq!(exec_plan.cse_stats.original_nodes, 2, "rolling_mean has 2 nodes");
    assert_eq!(exec_plan.cse_stats.eliminated, 0, "No sharing with single specialist");
    assert_eq!(exec_plan.steps.len(), 2);

    println!("  rolling_mean(price, w=10): {} nodes, {} eliminated  PASS",
        exec_plan.cse_stats.original_nodes, exec_plan.cse_stats.eliminated);
}

fn test_independent_data() {
    println!("\n--- Test 3: Different data variables (no sharing) ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "volume".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    // Different data variables → different identities → no CSE
    assert_eq!(exec_plan.cse_stats.original_nodes, 6);
    assert_eq!(exec_plan.cse_stats.eliminated, 0,
        "Different data vars should not share");
    assert_eq!(exec_plan.steps.len(), 6);

    println!("  rolling_zscore(price) + rolling_zscore(volume): {} nodes, {} eliminated  PASS",
        exec_plan.cse_stats.original_nodes, exec_plan.cse_stats.eliminated);
}

fn test_topo_order() {
    println!("\n--- Test 4: Topological ordering ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    // Verify scans come before fused_expr
    let ops: Vec<&PrimitiveOp> = exec_plan.steps.iter()
        .map(|s| &exec_plan.arena.get(s.node_id).op)
        .collect();

    let fused_idx = ops.iter().position(|op| **op == PrimitiveOp::FusedExpr)
        .expect("Should have a fused_expr node");
    let scan_indices: Vec<usize> = ops.iter().enumerate()
        .filter(|(_, op)| ***op == PrimitiveOp::Scan)
        .map(|(i, _)| i)
        .collect();

    for scan_idx in &scan_indices {
        assert!(*scan_idx < fused_idx,
            "Scan at {} should come before FusedExpr at {}", scan_idx, fused_idx);
    }

    println!("  Scans at {:?}, FusedExpr at {}  PASS", scan_indices, fused_idx);
}

fn test_world_state_probe() {
    println!("\n--- Test 5: World state probe (NullWorld = all skip=false) ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    // NullWorld: everything should need computation (skip = false)
    for step in &exec_plan.steps {
        assert!(!step.skip, "NullWorld should never skip: {:?}",
            exec_plan.arena.get(step.node_id).output_name);
    }

    println!("  All {} steps have skip=false (NullWorld baseline)  PASS",
        exec_plan.steps.len());
}

fn test_execute_mock() {
    println!("\n--- Test 6: Execute with MockDispatcher ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
            SpecialistCall {
                specialist: "rolling_std".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);

    // Provide data leaf pointers
    let mut data_ptrs = HashMap::new();
    data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: 0x100, byte_size: 8000 });
    data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: 0x200, byte_size: 8000 });

    let mut dispatcher = MockDispatcher::new();
    let (results, stats) = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
        .expect("Execute failed");

    // NullWorld = all misses = all dispatched
    assert_eq!(stats.misses, 4, "4 steps should all be misses");
    assert_eq!(stats.hits, 0, "No hits with NullWorld");
    assert_eq!(dispatcher.dispatch_log.len(), 4, "4 kernels dispatched");

    // Check outputs exist
    for ((call_idx, _), node_id) in &exec_plan.outputs {
        let result = results.get(node_id).unwrap_or_else(|| panic!(
            "Missing result for output call_idx={}", call_idx));
        assert!(!result.was_hit, "Should be computed, not hit");
    }

    println!("  4 steps dispatched, 0 hits  PASS");
}

fn test_provenance_reuse() {
    println!("\n--- Test 7: Provenance reuse (the 865x case) ---");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    // Use real provenance tags
    let mut input_provs = HashMap::new();
    input_provs.insert("price".into(), data_provenance("price:AAPL:2026-03-30:1s"));

    // Run 1: GpuStore is empty, everything misses
    let mut store = GpuStore::new(1_000_000_000); // 1GB budget
    let exec_plan = plan(&spec, &registry, &mut store, Some(&input_provs));

    let mut data_ptrs = HashMap::new();
    data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: 0x100, byte_size: 8000 });
    data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: 0x200, byte_size: 8000 });

    let mut dispatcher = MockDispatcher::new();
    let (_results1, stats1) = execute(&exec_plan, &mut store, &mut dispatcher, &data_ptrs)
        .expect("Execute run 1 failed");

    println!("  Run 1: {} hits, {} misses (cold store)",
        stats1.hits, stats1.misses);
    assert_eq!(stats1.misses, 3, "3 steps should miss on cold store");
    assert_eq!(stats1.hits, 0);

    // Run 2: Same plan, same inputs — everything should hit
    let exec_plan2 = plan(&spec, &registry, &mut store, Some(&input_provs));
    let mut dispatcher2 = MockDispatcher::new();
    let (_results2, stats2) = execute(&exec_plan2, &mut store, &mut dispatcher2, &data_ptrs)
        .expect("Execute run 2 failed");

    println!("  Run 2: {} hits, {} misses (warm store)",
        stats2.hits, stats2.misses);
    assert_eq!(stats2.hits, 3, "All 3 steps should hit on warm store");
    assert_eq!(stats2.misses, 0, "Zero misses on identical rerun");
    assert_eq!(dispatcher2.dispatch_log.len(), 0, "Zero kernel dispatches on hit");

    // Run 3: Different data → different provenance → all miss
    let mut input_provs3 = HashMap::new();
    input_provs3.insert("price".into(), data_provenance("price:AAPL:2026-03-31:1s"));
    let exec_plan3 = plan(&spec, &registry, &mut store, Some(&input_provs3));
    let mut dispatcher3 = MockDispatcher::new();
    let (_results3, stats3) = execute(&exec_plan3, &mut store, &mut dispatcher3, &data_ptrs)
        .expect("Execute run 3 failed");

    println!("  Run 3: {} hits, {} misses (new date = new provenance)",
        stats3.hits, stats3.misses);
    assert_eq!(stats3.misses, 3, "New data should miss");
    assert_eq!(stats3.hits, 0);

    let store_stats = store.stats();
    println!("  Store: {} entries, {:.0}% hit rate overall",
        store_stats.entries, store_stats.hit_rate() * 100.0);
    println!("  PASS: provenance reuse eliminates all computation on identical rerun");
}

fn test_cuda_dispatch() {
    println!("\n--- Test 8: CudaKernelDispatcher (Scan + FusedExpr on real GPU) ---");

    // rolling_mean = scan(data, agg=add) + fused_expr(rolling_mean)
    // Both primitives now dispatch to real GPU kernels.
    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_mean".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
    println!("  Plan: {} steps", exec_plan.steps.len());

    for step in &exec_plan.steps {
        let node = exec_plan.arena.get(step.node_id);
        println!("    [{:12?}] {:<6}  inputs={}",
            node.op, node.output_name, node.input_identities.len());
    }

    // Create real GPU data: 1000 doubles = 8000 bytes
    let n = 1000usize;
    let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();
    let host_data_sq: Vec<f64> = host_data.iter().map(|x| x * x).collect();

    let mut dispatcher = CudaKernelDispatcher::new()
        .expect("Failed to create CudaKernelDispatcher (is a GPU available?)");

    // Upload data to GPU. Clone the Arc<CudaStream> to avoid borrowing dispatcher.
    let stream = dispatcher.engine().stream().clone();
    let input_dev = stream.clone_htod(&host_data)
        .expect("Failed to upload data to GPU");
    let sq_dev = stream.clone_htod(&host_data_sq)
        .expect("Failed to upload squared data");

    // Extract raw device pointers (SyncOnDrop guards drop at end of block)
    let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };
    let sq_ptr = { let (p, _g) = sq_dev.device_ptr(&stream); p };
    let byte_size = (n * 8) as u64;

    // Build data_ptrs map
    let mut data_ptrs = HashMap::new();
    data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
    data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: sq_ptr, byte_size });

    // Execute — both Scan and FusedExpr dispatch to real GPU kernels.
    let (results, stats) = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
        .expect("Execute failed");

    println!("  Execute succeeded: {} steps, {} misses", stats.total_steps, stats.misses);
    for (identity, step_result) in &results {
        if identity.starts_with("data:") || identity.starts_with("data_sq:") {
            continue;
        }
        println!("    Result '{}': ptr=0x{:x}, size={}B, was_hit={}",
            identity, step_result.ptr.device_ptr, step_result.ptr.byte_size, step_result.was_hit);
    }

    assert_eq!(stats.total_steps, 2, "rolling_mean has 2 steps");
    assert_eq!(stats.misses, 2, "NullWorld: both steps should miss");
    println!("  PASS: Scan + FusedExpr both dispatched on real GPU");
}

fn test_fused_expr_numerics() {
    println!("\n--- Test 9: FusedExpr numerical validation ---");

    // x = [1.0, 2.0, ..., 100.0], window = 3
    // For k >= 2 (0-indexed): mean[k] = k exactly (x[k-2]+x[k-1]+x[k])/3 = (3k)/3 = k)
    // For k >= 2: std[k] = sqrt(2/3) = 0.8164965...
    // For k >= 2: zscore[k] = 1/sqrt(2/3) = sqrt(3/2) = 1.2247448...

    let n = 100usize;
    let window = 3usize;
    let host_x: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();
    let host_x_sq: Vec<f64> = host_x.iter().map(|x| x * x).collect();

    let expected_mean_at_k = |k: usize| -> f64 {
        if k < window { (0..=k).map(|i| host_x[i]).sum::<f64>() / (k + 1) as f64 }
        else { (k - window + 1..=k).map(|i| host_x[i]).sum::<f64>() / window as f64 }
    };
    let expected_std = (2.0f64 / 3.0).sqrt();   // constant for k >= 2 with window=3
    let expected_zscore = (3.0f64 / 2.0).sqrt(); // constant for k >= 2

    let mut dispatcher = CudaKernelDispatcher::new()
        .expect("CudaKernelDispatcher failed");
    let stream = dispatcher.engine().stream().clone();

    let x_dev = stream.clone_htod(&host_x).expect("htod x");
    let xsq_dev = stream.clone_htod(&host_x_sq).expect("htod x_sq");
    let (x_ptr, _gx) = x_dev.device_ptr(&stream);
    let (xsq_ptr, _gxsq) = xsq_dev.device_ptr(&stream);
    let byte_size = (n * 8) as u64;

    // --- rolling_mean ---
    {
        let registry = build_e04_registry();
        let spec = PipelineSpec {
            calls: vec![SpecialistCall {
                specialist: "rolling_mean".into(),
                data_var: "price".into(),
                window: window as u32,
            }],
        };
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: x_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: xsq_ptr, byte_size });

        let (results, _stats) = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
            .expect("rolling_mean execute failed");

        // Find the "out" result
        let out_identity = exec_plan.outputs.get(&(0, "out".to_string())).expect("no output");
        let out_result = results.get(out_identity).expect("no result for output");
        let out_host = unsafe { dispatcher.copy_to_host(out_result.ptr) }.expect("dtoh");

        let max_err = out_host.iter().enumerate()
            .map(|(k, &v)| (v - expected_mean_at_k(k)).abs())
            .fold(0.0f64, f64::max);

        println!("  rolling_mean: max_err={:.2e}  out[2]={:.4}  out[99]={:.4}",
            max_err, out_host[2], out_host[99]);
        assert!(max_err < 1e-9, "rolling_mean max_err={} exceeds tolerance", max_err);
        assert!((out_host[2] - 2.0).abs() < 1e-12, "mean[2] should be 2.0");
        assert!((out_host[99] - 99.0).abs() < 1e-9, "mean[99] should be 99.0");
    }

    // --- rolling_std ---
    {
        let registry = build_e04_registry();
        let spec = PipelineSpec {
            calls: vec![SpecialistCall {
                specialist: "rolling_std".into(),
                data_var: "price".into(),
                window: window as u32,
            }],
        };
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: x_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: xsq_ptr, byte_size });

        let (results, _stats) = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
            .expect("rolling_std execute failed");

        let out_identity = exec_plan.outputs.get(&(0, "out".to_string())).expect("no output");
        let out_result = results.get(out_identity).expect("no result");
        let out_host = unsafe { dispatcher.copy_to_host(out_result.ptr) }.expect("dtoh");

        // For k >= window-1=2: std should be sqrt(2/3) exactly
        let max_err_steady = out_host[window-1..].iter()
            .map(|&v| (v - expected_std).abs())
            .fold(0.0f64, f64::max);

        println!("  rolling_std:  max_err={:.2e}  out[2]={:.8}  expected={:.8}",
            max_err_steady, out_host[2], expected_std);
        assert!(max_err_steady < 1e-9, "rolling_std max_err={} exceeds tolerance", max_err_steady);
    }

    // --- rolling_zscore ---
    {
        let registry = build_e04_registry();
        let spec = PipelineSpec {
            calls: vec![SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: window as u32,
            }],
        };
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: x_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: xsq_ptr, byte_size });

        let (results, _stats) = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs)
            .expect("rolling_zscore execute failed");

        let out_identity = exec_plan.outputs.get(&(0, "out".to_string())).expect("no output");
        let out_result = results.get(out_identity).expect("no result");
        let out_host = unsafe { dispatcher.copy_to_host(out_result.ptr) }.expect("dtoh");

        // For k >= window-1=2: zscore = (x[k] - mean[k]) / std[k]
        //   = (k+1 - k) / sqrt(2/3) = sqrt(3/2) = 1.2247...
        let max_err_steady = out_host[window-1..].iter()
            .map(|&v| (v - expected_zscore).abs())
            .fold(0.0f64, f64::max);

        println!("  rolling_zscore: max_err={:.2e}  out[2]={:.8}  expected={:.8}",
            max_err_steady, out_host[2], expected_zscore);
        assert!(max_err_steady < 1e-9, "rolling_zscore max_err={} exceeds tolerance", max_err_steady);
    }

    println!("  PASS: rolling_mean, rolling_std, rolling_zscore numerically correct on GPU");
}
