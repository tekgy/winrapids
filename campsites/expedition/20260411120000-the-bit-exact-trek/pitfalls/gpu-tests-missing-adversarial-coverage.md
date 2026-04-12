# Pitfall: GPU End-to-End Tests Have No Adversarial Coverage

**Status: DOCUMENTATION — no failing test yet, but a structural gap**

## What the GPU tests cover

`crates/tambear-primitives/tests/gpu_end_to_end.rs` (9 tests):
- `sum` of 1..10000 (nice integers, no fp pathology)
- `mean_arithmetic` of 1..1000 (closed-form expected: 500.5)
- `variance` of 10000 sin-wave values (bounded, small-ish mean, no catastrophic cancellation)
- `mean_quadratic` / `sum_of_squares` / `l1_norm` of bounded data
- `pearson_r` with perfect linear y=2.5x+7 (x in 0..1024)
- Custom `Σ|ln x|` recipe
- Device name detection

## What the GPU tests do NOT cover

- **NaN in input data** — no test
- **Inf in input data** — no test
- **Empty input** — no test (would crash: `dispatch` with n=0)
- **Large-mean data** — `gpu_variance_matches_cpu` only checks CPU-GPU agreement, not correctness. Both CPU and GPU use the same one-pass formula and both return garbage on large-mean data. The test passes while both return a plausible-but-wrong number.
- **Subnormal values** — no test
- **Extreme overflow** — no test

## The structural problem: "CPU matches GPU" ≠ "both are correct"

`gpu_variance_matches_cpu` asserts `|gpu - cpu| / |cpu| < 1e-10`. This is a
**relative agreement test**, not a **correctness test**. When the CPU result
is wrong (e.g., variance = -4592 for large-mean data), and the GPU result
happens to also be approximately -4592 (same formula, same catastrophic
cancellation), the test passes. Both are wrong, but they agree.

This is the hardest class of silent failure: two independent paths agreeing on
the wrong answer. The agreement gives false confidence.

The fix requires an oracle — a reference value computed by a different method
(two-pass, or high-precision via Python mpmath). Cross-backend agreement is
necessary but not sufficient for correctness.

## What GPU tests need

As the adversarial input suite is built, the same pathological inputs should
be dispatched on GPU with an oracle reference, not just CPU comparison:

```rust
#[test]
fn gpu_variance_large_mean_against_oracle() {
    // Same adversarial data from adversarial_baseline.rs
    let data: Vec<f64> = (0..1000).map(|i| 1.0e9 + (i as f64) * 1.0e-6).collect();
    let true_var = 1000.0 * 1001.0 / 12.0 * 1.0e-12;  // oracle

    let gpu_v = run_recipe_gpu(&gpu, &variance(), &data, &[], 0.0);
    let rel = (gpu_v - true_var).abs() / true_var;
    assert!(rel < 0.01, "gpu variance catastrophic cancellation: got {}, true={}", gpu_v, true_var);
}
```

This test will fail until variance is fixed to use a stable algorithm on both
CPU and GPU paths.

## Why this matters for I10 (cross-backend diff is continuous)

I10 says: don't defer cross-backend validation to a "final audit." But the
current GPU tests effectively only validate "GPU ≈ CPU on nice inputs." They
don't validate "GPU is correct on hard inputs." The adversarial inputs need to
run through the GPU path too — and before the PTX assembler (Peak 3) replaces
NVRTC, so we have a NVRTC baseline to compare the new assembler against.

## Action items

- After `variance` is fixed (two-pass), add GPU adversarial tests that check
  variance against the analytic oracle on large-mean data.
- Add `gpu_variance_nan_propagation` — dispatch NaN-containing data, verify GPU
  returns NaN (not a silently-wrong float). PTX NaN handling may differ from CPU.
- Add `gpu_empty_input_graceful` — dispatch empty buffer, verify no crash and
  correct identity/NaN result.

These tests are the I10 enforcement mechanism — not a final audit, a per-bug check.
