//! End-to-end recipe tests through the tam-IR harness (campsite 4.6).
//!
//! These are the harness-side translations of the tests in
//! `tambear-primitives/tests/gpu_end_to_end.rs`. The originals run on the
//! vendor CUDA/NVRTC path; these run on the CPU interpreter via the
//! `TamBackend` trait, with documented tolerance policies.
//!
//! ## Tolerance policy
//!
//! | Recipe | Current | Rationale |
//! |--------|---------|-----------|
//! | sum | xfail_nondeterministic → bit_exact post-Peak-6 | Pure arithmetic; atomicAdd non-det |
//! | variance_pass | bit_exact (accumulation only, no catastrophic cancel) | Three independent sums |
//! | pearson_r | bit_exact (accumulation pass) | Five independent sums + fsqrt at gather |
//! | Σ\|ln x\| | xfail — requires Peak 2 tam_ln | Transcendental chain |
//!
//! ## Invariants
//!
//! - I5: reduction tests with non-deterministic ordering are marked
//!   `#[ignore = "xfail_nondeterministic"]` until Peak 6 (deterministic
//!   tree-reduce) removes the annotation.  Only the scientist removes these.
//! - I1/I8: `tam_ln` tests are `#[ignore]` until tambear-libm lands.

use tambear_tam_test_harness::{
    TamProgram, Inputs, ToleranceSpec,
    TamBackend,
    CpuInterpreterBackend,
    run_all_backends, assert_cross_backend_agreement_named, compare_backends,
};
use tambear_tam_test_harness::backend::{BackendRegistry, NullBackend};

// ---------------------------------------------------------------------------
// Program loaders
// ---------------------------------------------------------------------------

fn sum_all_add_program() -> TamProgram {
    TamProgram::from_source(include_str!(
        "../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/sum_all_add.tam"
    )).expect("sum_all_add.tam must parse and verify")
}

fn variance_pass_program() -> TamProgram {
    TamProgram::from_source(include_str!(
        "../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/variance_pass.tam"
    )).expect("variance_pass.tam must parse and verify")
}

fn pearson_r_pass_program() -> TamProgram {
    TamProgram::from_source(include_str!(
        "../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/pearson_r_pass.tam"
    )).expect("pearson_r_pass.tam must parse and verify")
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

/// Σ 1..=10_000 = n*(n+1)/2 = 50_005_000 — exactly representable in f64.
/// Translated from `gpu_sum_matches_cpu` (exact value assertion only).
#[test]
fn cpu_sum_exact_value_1_to_10000() {
    let cpu = CpuInterpreterBackend::new();
    let program = sum_all_add_program();
    let data: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let inputs = Inputs::new().with_buf("data", data);
    let out = cpu.run(&program, &inputs);
    // n*(n+1)/2 = 50_005_000.0 — exactly representable; expect bit-exact.
    assert_eq!(out.slots[0], 50_005_000.0,
        "CPU interpreter sum[1..=10000] must be exactly 50_005_000.0");
}

/// Harness cross-backend agreement for sum.
///
/// Tagged xfail_nondeterministic: `reduce_block_add` uses sequential_left ordering
/// on the CPU interpreter (deterministic on CPU, but not guaranteed on GPU).
/// Post-Peak-6 this becomes a strict bit_exact cross-backend test.
#[test]
#[ignore = "xfail_nondeterministic — remove after Peak 6 (deterministic tree reduce)"]
fn harness_sum_cross_backend_bit_exact() {
    let mut registry = BackendRegistry::new();
    registry.push(Box::new(CpuInterpreterBackend::new()));
    // registry.push(Box::new(CudaPtxRawBackend::new())); // when Peak 3 lands
    // registry.push(Box::new(VulkanSpirvBackend::new())); // when Peak 7 lands
    let program = sum_all_add_program();
    let data: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let inputs = Inputs::new().with_buf("data", data);
    let results = run_all_backends(&registry, &program, &inputs);
    assert_cross_backend_agreement_named(&results, "sum_all_add", ToleranceSpec::bit_exact());
}

// ---------------------------------------------------------------------------
// Variance (accumulation pass only — gathering is host-side)
// ---------------------------------------------------------------------------

/// Translated from `gpu_variance_matches_cpu`.
///
/// The variance_pass kernel accumulates three independent sums: sum(x), sum(x^2),
/// count. These are pure arithmetic — no catastrophic cancellation in the kernel
/// itself (the subtraction for variance happens at gather time on the host).
/// So: bit-exact from the CPU interpreter, no xfail needed for the accumulation pass.
#[test]
fn cpu_variance_pass_sums_nice_data() {
    let cpu = CpuInterpreterBackend::new();
    let program = variance_pass_program();
    // Same data as gpu_variance_matches_cpu: sin wave centered near 5.
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64).sin() * 10.0 + 5.0).collect();
    let inputs = Inputs::new().with_buf("data", data.clone());
    let out = cpu.run(&program, &inputs);

    // Verify slot layout: [sum, sum_sq, count].
    let n = data.len() as f64;
    let sum_x: f64 = data.iter().sum();
    let sum_sq: f64 = data.iter().map(|x| x * x).sum();

    assert_eq!(out.slots.len(), 3, "variance_pass must produce 3 slots");
    assert_eq!(out.slots[2], n, "slot[2] = count must equal n");

    // sum and sum_sq are sequential left-to-right sums in the interpreter,
    // identical to the Rust iterator sum — so bit-exact agreement is expected.
    assert_eq!(out.slots[0], sum_x,
        "slot[0] = sum(x): interpreter must match sequential Rust sum");
    assert_eq!(out.slots[1], sum_sq,
        "slot[1] = sum(x^2): interpreter must match sequential Rust sum");
}

/// The variance trap: data near 1e9 with small variance.
/// One-pass formula catastrophically cancels. The variance_pass kernel is safe
/// (it accumulates sum and sum_sq independently), so only the HOST gather is
/// at risk. This test pins the accumulation-only output as correct.
#[test]
fn cpu_variance_pass_sums_trap_data() {
    let cpu = CpuInterpreterBackend::new();
    let program = variance_pass_program();
    // Mean ~1e9, small spread: the one-pass host formula would give 0 variance.
    // The KERNEL is fine; the gather formula is the problem.
    let data: Vec<f64> = (0..1000).map(|i| 1.0e9 + i as f64).collect();
    let inputs = Inputs::new().with_buf("data", data.clone());
    let out = cpu.run(&program, &inputs);

    let n = data.len() as f64;
    let sum_x: f64 = data.iter().sum();
    let sum_sq: f64 = data.iter().map(|x| x * x).sum();

    assert_eq!(out.slots[2], n, "count");
    assert_eq!(out.slots[0], sum_x, "sum(x) — accumulation must be exact");
    assert_eq!(out.slots[1], sum_sq, "sum(x^2) — accumulation must be exact");

    // Demonstrate the gather hazard (not the kernel's fault):
    // one-pass formula on these sums gives wrong variance.
    let mean = out.slots[0] / out.slots[2];
    let one_pass_var = (out.slots[1] / out.slots[2]) - mean * mean;
    // True population variance = sum((i - 499.5)^2) / 1000 = 83333.25
    let true_var = 83_333.25;
    let one_pass_rel_err = ((one_pass_var - true_var) / true_var).abs();
    // The one-pass formula may lose precision entirely on this data (known P12 bug).
    // We document the error rather than asserting a specific wrong value,
    // since the degree of cancellation depends on the exact fp64 path.
    eprintln!(
        "one-pass formula relative error on trap data: {one_pass_rel_err:.3e} \
         (expected ~1.0 or infinite; kernel sums are correct regardless)"
    );
}

// ---------------------------------------------------------------------------
// Pearson r accumulation pass
// ---------------------------------------------------------------------------

/// Translated from `gpu_pearson_matches_cpu_two_input`.
///
/// The pearson_r_pass kernel accumulates six sums: count, sum_x, sum_y, sum_x2,
/// sum_y2, sum_xy. For y = 2.5*x + 7 with integer x, r = 1.0 exactly.
///
/// Pearson formula (host-side gather):
///   r = (n * sum_xy - sum_x * sum_y)
///       / sqrt((n * sum_x2 - sum_x^2) * (n * sum_y2 - sum_y^2))
///
/// With perfect linear data, numerator = denominator, so r = 1.0.
/// IEEE 754 fsqrt is correctly rounded, so the division should also be exact.
#[test]
fn cpu_pearson_perfect_linear_is_one() {
    let cpu = CpuInterpreterBackend::new();
    let program = pearson_r_pass_program();
    let n = 1024usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.5 * xi + 7.0).collect();

    let inputs = Inputs::new()
        .with_buf("xs", x.clone())
        .with_buf("ys", y.clone());
    let out = cpu.run(&program, &inputs);

    assert_eq!(out.slots.len(), 6, "pearson_r_pass must produce 6 slots");

    // Host gather: compute r from the 6 accumulated sums.
    let count  = out.slots[0];
    let sum_x  = out.slots[1];
    let sum_y  = out.slots[2];
    let sum_x2 = out.slots[3];
    let sum_y2 = out.slots[4];
    let sum_xy = out.slots[5];

    let num = count * sum_xy - sum_x * sum_y;
    let den = ((count * sum_x2 - sum_x * sum_x) * (count * sum_y2 - sum_y * sum_y)).sqrt();
    let r = num / den;

    // Perfect linear relationship → r = 1.0 exactly.
    // In practice, floating-point rounding in the intermediates means we may
    // be off by 1-2 ULPs. We accept bit-exact 1.0 as the target; document
    // the actual ULP distance if it isn't.
    let ulp_err = if r == 1.0 {
        0u64
    } else {
        r.to_bits().abs_diff(1.0_f64.to_bits())
    };
    eprintln!("pearson r = {r}, ULP distance from 1.0: {ulp_err}");
    assert!(
        ulp_err <= 2,
        "Pearson r on perfect linear data must be within 2 ULP of 1.0; got r={r} ({ulp_err} ULP off)"
    );
}

/// Harness cross-backend check for pearson accumulation pass.
/// Tagged xfail_nondeterministic (same reasoning as sum — reduce_block_add
/// ordering not guaranteed across backends until Peak 6).
#[test]
#[ignore = "xfail_nondeterministic — remove after Peak 6 (deterministic tree reduce)"]
fn harness_pearson_cross_backend_bit_exact() {
    let mut registry = BackendRegistry::new();
    registry.push(Box::new(CpuInterpreterBackend::new()));
    let n = 256usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.5 * xi + 7.0).collect();
    let inputs = Inputs::new()
        .with_buf("xs", x)
        .with_buf("ys", y);
    let program = pearson_r_pass_program();
    let results = run_all_backends(&registry, &program, &inputs);
    assert_cross_backend_agreement_named(&results, "pearson_r_pass", ToleranceSpec::bit_exact());
}

// ---------------------------------------------------------------------------
// Σ|ln x| — transcendental chain (blocked until Peak 2)
// ---------------------------------------------------------------------------

/// Translated from `gpu_dispatches_custom_expression`.
///
/// This recipe requires `tam_ln`, which does not exist in tambear-libm yet.
/// The CPU interpreter will panic on any TamLn op. Blocked until:
///   - Peak 2 campsite 2.6+ (tambear-libm exp/ln implementation)
///   - tambear-libm wired into the CPU interpreter (campsite 5.2)
///
/// When unblocked: tolerance is `within_ulp(1)` per the parity table
/// (tam_ln ≤ 1 ULP; sum of n ULP errors bounds total by n·1 ULP, but
/// the cross-backend diff uses bit_exact since both backends use the same
/// first-principles tam_ln).
#[test]
#[ignore = "requires tambear-libm tam_ln (Peak 2 campsite 2.6+) + CPU interpreter wiring (Peak 5 campsite 5.2)"]
fn harness_sum_abs_ln_cross_backend() {
    let mut registry = BackendRegistry::new();
    registry.push(Box::new(CpuInterpreterBackend::new()));
    // A .tam program for Σ|ln x| will be written when tam_ln exists.
    // For now: placeholder to document the expected test shape.
    let _program = TamProgram::pure("sum_abs_ln");  // placeholder
    let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    let _inputs = Inputs::new().with_buf("data", data);
    // When unblocked:
    //   let results = run_all_backends(&registry, &_program, &_inputs);
    //   assert_cross_backend_agreement_named(&results, "sum_abs_ln",
    //       ToleranceSpec::bit_exact());  // same libm → bit_exact cross-backend
    todo!("activate when tam_ln is implemented in tambear-libm and wired into CPU interpreter")
}

// ---------------------------------------------------------------------------
// Harness correctness: CPU vs NullBackend detects disagreement
// ---------------------------------------------------------------------------

/// Integration test for the harness's violation-detection path.
/// CPU produces the real sum; NullBackend produces zeros. The report must flag a
/// disagreement. This is the same test as `harness_detects_cpu_vs_null_disagreement`
/// in `cpu_backend.rs`, but expressed as an integration test to verify the full path.
#[test]
fn e2e_harness_detects_real_vs_null_disagreement() {
    let mut registry = BackendRegistry::new();
    registry.push(Box::new(CpuInterpreterBackend::new()));
    registry.push(Box::new(NullBackend::new(1)));
    let program = sum_all_add_program();
    let inputs = Inputs::new().with_buf("data", vec![1.0, 2.0, 3.0]);
    let results = run_all_backends(&registry, &program, &inputs);
    let report = compare_backends(&results, "sum_all_add", ToleranceSpec::bit_exact());
    // CPU gives 6.0; NullBackend gives 0.0. Must disagree.
    assert!(!report.all_agree(),
        "Harness must detect CPU (6.0) vs Null (0.0) disagreement");
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].value_a, 6.0);
    assert_eq!(report.violations[0].value_b, 0.0);
}
