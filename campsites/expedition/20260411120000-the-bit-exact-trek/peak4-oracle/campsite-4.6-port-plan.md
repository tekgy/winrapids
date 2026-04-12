# Campsite 4.6 — Port Plan: gpu_end_to_end.rs → harness

**Status:** Ready to execute once Peak 5 (CpuInterpreterBackend) lands.  
**Author:** Test Oracle  
**Date:** 2026-04-11

---

## What 4.6 does

Take every test in `crates/tambear-primitives/tests/gpu_end_to_end.rs` and
re-express it through the harness in `tambear-tam-test-harness`. The new tests:

1. Run the same program on every registered backend
2. Compare with `ToleranceSpec::bit_exact()` for pure arithmetic (post-Peak 6)
3. Compare with `ToleranceSpec::within_ulp(bound)` for transcendental chains
4. Mark reduction-result tests `#[ignore = "xfail_nondeterministic"]` until Peak 6

---

## Current tests and their harness translation

### `gpu_sum_matches_cpu`

**Current:** `assert!((gpu_v - cpu_v).abs() < 1e-6)` — relative tolerance.

**Ported:** 
```rust
// Pre-Peak-6: reduction is non-deterministic
#[test]
#[ignore = "xfail_nondeterministic — remove after Peak 6"]
fn harness_sum_cross_backend() {
    let mut reg = BackendRegistry::new();
    reg.push(Box::new(CpuInterpreterBackend::new()));
    // reg.push(Box::new(CudaPtxRawBackend::new())); // when Peak 3 lands
    
    let program = TamProgram::pure("sum_all");
    let data: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let inputs = Inputs::new().with_buf("x", data);
    
    let results = run_all_backends(&reg, &program, &inputs);
    // Post-Peak-6: bit_exact. Pre-Peak-6: this stays ignored.
    assert_cross_backend_agreement_named(&results, "sum", ToleranceSpec::bit_exact());
}

// Separate test for the scalar value — does NOT need xfail, runs on CPU only
#[test]
fn cpu_sum_exact_value() {
    // n*(n+1)/2 = 50,005,000 — exactly representable in f64
    let cpu = CpuInterpreterBackend::new();
    let program = TamProgram::pure("sum_all");
    let data: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let inputs = Inputs::new().with_buf("x", data);
    let out = cpu.run(&program, &inputs);
    assert_eq!(out.slots[0], 50_005_000.0);
}
```

### `gpu_variance_matches_cpu`

**Critical note:** This test uses `(0..10_000).map(|i| (i as f64).sin() * 10.0 + 5.0)` — data centered near 5.0. The one-pass formula doesn't catastrophically fail here because the mean is small. **But** on the adversarial one-pass-variance-trap data (mean near 1e9), it will. The port must include both the "nice data" test AND the adversarial trap.

**Current tolerance:** `rel < 1e-10` — uses relative error, which masks ULP-level disagreement.

**Ported:**
```rust
#[test]
#[ignore = "xfail_nondeterministic — remove after Peak 6"]
fn harness_variance_cross_backend_nice_data() {
    // This test passes even with the buggy one-pass formula
    // because the mean is small. It's testing reduction correctness,
    // not numerical stability.
    // ...
}

#[test]
fn cpu_variance_trap_two_pass_required() {
    // This is the acceptance test for campsite 1.4 (two-pass variance)
    // Must pass BEFORE we accept the variance implementation.
    // Data: Normal(μ=1e9, σ=0.1), n=10000
    // Expected: var ≈ 0.01, |ε| < 1e-6
    // One-pass formula: FAILS (gives 0.0)
    // Two-pass formula: PASSES
    todo!("activate when two-pass variance .tam kernel lands (pathmaker 1.4)");
}
```

### `gpu_pearson_matches_cpu_two_input`

**Current tolerance:** `abs < 1e-10` on both CPU and GPU.

**Ported:** Pearson R on a perfect linear relationship = 1.0 exactly. CPU interpreter should return exactly 1.0 if the arithmetic is right. This is a synthetic ground truth test.

```rust
#[test]
fn cpu_pearson_perfect_linear_is_one() {
    // y = 2.5*x + 7 → r = 1.0 exactly
    // This is a synthetic ground truth test.
    // The computation involves fdiv and fsqrt, so
    // we expect bit-exact 1.0 from a correct implementation.
    // (Actually: sqrt(1.0) = 1.0 exactly. So this should be bit-exact.)
    todo!("activate when CpuInterpreterBackend exists");
}
```

### `gpu_dispatches_custom_expression` (Σ|ln x|)

**Critical:** This uses `tam_ln` — a transcendental. Before Peak 2 (tambear-libm) lands, the CPU interpreter will panic on `tam_ln` (campsite 1.13: "not yet in libm"). So this test MUST be `#[ignore]` until both Peak 2 AND Peak 5 are done.

**Ported:**
```rust
#[test]
#[ignore = "requires tambear-libm tam_ln (Peak 2) + CpuInterpreterBackend (Peak 5)"]
fn harness_sum_abs_ln_cross_backend() {
    // Uses tam_ln — within_ulp(1) tolerance for the ln chain
    // ...
    assert_cross_backend_agreement_named(&results, "sum_abs_ln", 
        ToleranceSpec::within_ulp(1));  // tam_ln ≤1 ULP → n*1 ULP budget per chain
}
```

---

## Tolerance assignment per recipe

| Recipe | Current tolerance | Harness tolerance | Rationale |
|--------|------------------|-------------------|-----------|
| sum | `abs < 1e-6` | `bit_exact` (post-Peak-6) | Pure arithmetic. atomicAdd non-det until Peak 6 |
| mean | `abs < 1e-9` | `bit_exact` (post-Peak-6) | Pure arithmetic + division |
| variance | `rel < 1e-10` | `bit_exact` (post-Peak-6, two-pass) | Pure arithmetic once two-pass lands |
| rms | `rel < 1e-12` | `bit_exact` (post-Peak-6) | Pure arithmetic |
| sum_sq | `rel < 1e-11` | `bit_exact` (post-Peak-6) | Pure arithmetic |
| l1_norm | `rel < 1e-12` | `bit_exact` (post-Peak-6) | Pure arithmetic |
| pearson_r | `abs < 1e-12` | `bit_exact` (post-Peak-6) | Pure arithmetic + fsqrt (IEEE exact) |
| Σ\|ln x\| | `rel < 1e-11` | `within_ulp(n_elements * 1)` | tam_ln ≤1 ULP per call; error accumulates |

**Key observation:** Every current test uses relative or absolute tolerance. The harness replaces all of these with either `bit_exact` (post-Peak-6) or documented ULP bounds. The current tests are *not wrong* — they're just not as strict as they should be, and they hide backend bugs that would only show up at the ULP level.

---

## Pre-conditions for 4.6 execution

- [ ] Peak 5 campsite 5.1 — `CpuInterpreterBackend` implementing `TamBackend`
- [ ] Peak 1 campsite 1.13 — `tam_ln` stub (panics until Peak 2) exists in AST
- [ ] Placeholder `TamProgram` types replaced with real `tambear-tam-ir` types

Optional (needed for xfail removal):
- [ ] Peak 6 complete — deterministic reductions
- [ ] Peak 2 complete — `tam_ln` implemented in tambear-libm

---

## The one test that doesn't port

`gpu_device_name_printed` — this is a backend introspection test, not a computation test. It stays in `gpu_end_to_end.rs` as-is. The harness doesn't care about device names.

---

## File location for the ported tests

New file: `crates/tambear-tam-test-harness/tests/e2e_recipes.rs`

Not in `src/` — these are integration tests that depend on backends being registered.
Each test file can be gated with `#[cfg(feature = "...")]` if needed to keep
CPU-only machines from failing.
