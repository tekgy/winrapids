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
