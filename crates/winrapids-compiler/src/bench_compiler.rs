//! Observer benchmark: compiler plan() time and CSE verification.
//!
//! Measures:
//! 1. CSE elimination ratio against E04 Python baseline (33%)
//! 2. plan() compilation time for E04 pipeline
//! 3. plan() scaling: 2 specialists → 10 → 50 (all same data)
//! 4. Topological sort cost
//!
//! Standing methodology: 3 warmup, 20 timed, p50/p99/mean.

use std::time::Instant;
use std::collections::HashMap;
use winrapids_compiler::plan::{PipelineSpec, SpecialistCall, plan};
use winrapids_compiler::registry::build_e04_registry;
use winrapids_store::world::NullWorld;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: winrapids-compiler Performance");
    println!("{}", "=".repeat(70));

    let registry = build_e04_registry();

    bench_e04_cse(&registry);
    bench_plan_time(&registry);
    bench_scaling(&registry);

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

fn bench_e04_cse(registry: &HashMap<String, winrapids_compiler::registry::SpecialistRecipe>) {
    println!("\n--- E04 CSE Verification ---\n");

    // E04 pipeline: rolling_zscore + rolling_std on same data
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall { specialist: "rolling_zscore".into(), data_var: "price".into(), window: 20 },
            SpecialistCall { specialist: "rolling_std".into(), data_var: "price".into(), window: 20 },
        ],
    };

    let exec_plan = plan(&spec, &registry, &mut NullWorld);

    println!("  Pipeline: rolling_zscore(price,20) + rolling_std(price,20)");
    println!("  Original nodes: {}", exec_plan.cse_stats.original_nodes);
    println!("  After CSE:      {}", exec_plan.cse_stats.after_cse);
    println!("  Eliminated:     {} ({:.0}%)",
        exec_plan.cse_stats.eliminated,
        100.0 * exec_plan.cse_stats.eliminated as f64 / exec_plan.cse_stats.original_nodes as f64);
    println!("  E04 Python baseline: 2 eliminated (33%)");
    println!("  Match: {}", if exec_plan.cse_stats.eliminated == 2 { "YES" } else { "NO" });
}

fn bench_plan_time(registry: &HashMap<String, winrapids_compiler::registry::SpecialistRecipe>) {
    println!("\n--- plan() Compilation Time (E04 Pipeline) ---\n");

    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall { specialist: "rolling_zscore".into(), data_var: "price".into(), window: 20 },
            SpecialistCall { specialist: "rolling_std".into(), data_var: "price".into(), window: 20 },
        ],
    };

    // Warmup
    for _ in 0..3 {
        let _ = plan(&spec, &registry, &mut NullWorld);
    }

    // Timed
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        let t0 = Instant::now();
        let _ = plan(&spec, &registry, &mut NullWorld);
        times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[10];
    let p99 = times[19];
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    println!("  p50={:.1} us  p99={:.1} us  mean={:.1} us", p50, p99, mean);
    println!("  E04 Python baseline: ~15,000 us (E05 measurement)");
}

fn bench_scaling(registry: &HashMap<String, winrapids_compiler::registry::SpecialistRecipe>) {
    println!("\n--- plan() Scaling: Specialists on Same Data ---\n");
    println!("  All specialists operate on 'price' with window=20.");
    println!("  CSE should increase as more specialists share primitives.\n");

    let specialist_names = ["rolling_mean", "rolling_std", "rolling_zscore"];

    for &n_calls in &[2usize, 3, 6, 10, 30, 50] {
        let calls: Vec<SpecialistCall> = (0..n_calls)
            .map(|i| SpecialistCall {
                specialist: specialist_names[i % 3].into(),
                data_var: "price".into(),
                window: 20,
            })
            .collect();
        let spec = PipelineSpec { calls };

        // Warmup
        for _ in 0..3 {
            let _ = plan(&spec, &registry, &mut NullWorld);
        }

        // Timed
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let exec_plan = plan(&spec, &registry, &mut NullWorld);
            let elapsed = t0.elapsed().as_nanos() as f64 / 1000.0;
            times.push(elapsed);

            // Use result from last run for stats
            if times.len() == 20 {
                let cse = &exec_plan.cse_stats;
                let pct = 100.0 * cse.eliminated as f64 / cse.original_nodes as f64;
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = times[10];
                println!("  n_calls={:>3}: orig={:>3}  after_cse={:>3}  elim={:>3} ({:4.0}%)  plan_p50={:7.1} us",
                    n_calls, cse.original_nodes, cse.after_cse, cse.eliminated, pct, p50);
            }
        }
    }
}
