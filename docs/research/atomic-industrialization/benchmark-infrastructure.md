# Benchmark Infrastructure: Principle 10 Harness

**Author**: scientist
**Date**: 2026-04-10
**Status**: framework documented; first workup (pearson_r) complete

---

## Purpose

Every primitive workup requires three types of validation infrastructure:

1. **Oracle harness** — compare tambear against mpmath/SymPy at high precision
2. **Competitor harness** — compare tambear against scipy/R/Julia/MATLAB
3. **Scale sweep** — measure time and accuracy at n = 10², 10³, 10⁴, 10⁵, 10⁶, 10⁷, 10⁸

These are separate concerns and should be separate test files.

---

## Oracle Harness Pattern

The oracle harness lives in Python (mpmath) and compares against tambear's
output embedded as constants. The workflow is:

1. Run the Python oracle script to generate expected values at 50 dp
2. Extract the f64-representable portion of each expected value
3. Embed those constants in the Rust parity test file (`workup_<name>.rs`)
4. The Rust tests assert agreement to ≤ 1 ULP (relative error < 2.2e-16)

**Why embed constants rather than calling Python from Rust tests?**
- Tests run without any Python dependency
- Expected values are explicit and reviewable in the PR diff
- Oracle generation is a one-time computation, not a per-run cost

### Oracle script template

```python
#!/usr/bin/env python3
"""
Oracle script for workup: <primitive_name>
Produces mpmath reference values at 50 decimal digits.
Run once to generate constants for the Rust parity test file.
"""
import mpmath
mpmath.mp.dps = 50

def <primitive>_mp(args):
    """First-principles implementation at 50 dp."""
    # ...

# --- test cases ---
cases = [
    ("case name", args, expected_f64_approx),
    ...
]

for name, args, _ in cases:
    result = <primitive>_mp(*args)
    print(f"# {name}")
    print(f"# 50dp: {mpmath.nstr(result, 50)}")
    print(f"# f64:  {float(result)!r}")
    print()
```

---

## Competitor Harness Pattern

The competitor harness also lives in Python. It calls scipy, R (via rpy2 or
subprocess), and records agreement/disagreement.

### Competitor harness template

```python
#!/usr/bin/env python3
"""Competitor comparison for workup: <primitive_name>"""
import mpmath
mpmath.mp.dps = 50
from scipy import stats
# from rpy2.robjects import r as R  # optional

def run_case(name, args, tambear_result, oracle_result):
    scipy_result = scipy_function(*args)
    oracle_f64 = float(oracle_result)
    
    print(f"\n=== {name} ===")
    print(f"  tambear:  {tambear_result}")
    print(f"  scipy:    {scipy_result}")
    print(f"  oracle:   {oracle_f64}")
    print(f"  |tambear - oracle| = {abs(tambear_result - oracle_f64):.2e}")
    print(f"  |scipy   - oracle| = {abs(scipy_result   - oracle_f64):.2e}")
    
    if abs(tambear_result - oracle_f64) > 1e-12:
        print(f"  *** TAMBEAR DISCREPANCY ***")
    if abs(scipy_result - oracle_f64) > 1e-12:
        print(f"  *** SCIPY DISCREPANCY ***")
```

---

## Scale Sweep Pattern

The scale sweep lives in Rust (`tests/scale_ladder_<primitive>.rs`). It uses
`#[ignore]` so it only runs when explicitly requested with `--release --ignored`.

```rust
//! Scale ladder: `<primitive>`
//!
//! Run: cargo test --test scale_ladder_<primitive> --release -- --ignored --nocapture

use std::time::Instant;
use tambear::<module>::<primitive>;

fn gen_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use tambear::rng::{Xoshiro256, TamRng};
    let mut rng = Xoshiro256::new(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    (x, y)
}

#[test]
#[ignore]
fn scale_sweep_<primitive>() {
    let sizes: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
    
    println!("\n{:<12} {:>12} {:>15}", "n", "time_us", "throughput");
    for &n in sizes {
        let (x, y) = gen_data(n, 42);
        let t0 = Instant::now();
        let r = <primitive>(&x, &y);
        let elapsed = t0.elapsed().as_micros();
        let throughput = (n as f64 / elapsed as f64) as u64;
        println!("{:<12} {:>12} {:>14}M/s  r={:.6}", n, elapsed, throughput, r);
    }
}
```

### Documented scale results for pearson_r

Benchmarks **not yet run** — pearson_r is O(n) so expected:
- n=10²: < 1 µs (constant overhead)
- n=10⁶: ~1 ms (RAM bandwidth bound at 16n bytes)
- n=10⁸: ~100 ms (feasible)

GPU crossover: not useful for single (x, y) pair; useful for batch of k columns
against one reference column (k×n matrix correlation).

---

## Workup Completion Checklist

For each primitive, the workup is complete when:

- [ ] Workup .md file at `docs/research/atomic-industrialization/<name>.md`
- [ ] Rust parity test at `crates/tambear/tests/workup_<name>.rs`
- [ ] Oracle Python script embedded in workup Appendix B
- [ ] All parity tests green (`cargo test --test workup_<name>`)
- [ ] Competitor comparison section filled with scipy/mpmath agreement
- [ ] Scale ladder `#[ignore]` test written (run pending hardware session)
- [ ] Known bugs documented with severity in Section 10
- [ ] Sign-off checklist updated

---

## Primitives Queue

Ordered by impact (how many downstream methods depend on this primitive):

| Priority | Primitive | File | Depends-on | Used-by |
|----------|-----------|------|------------|---------|
| 1 | `pearson_r` | nonparametric.rs | mean | OLS, CCC, concordance, PCA |
| 2 | `normal_cdf` | special_functions.rs | erfc | t-test, z-test, every p-value |
| 3 | `erfc` | special_functions.rs | — | normal_cdf, everything |
| 4 | `cholesky` | linear_algebra.rs | — | OLS, PCA, Mahalanobis |
| 5 | `pelt` | time_series.rs | pelt_segment_cost | changepoint analysis |
| 6 | `log_gamma` | special_functions.rs | — | beta, gamma, every test |
| 7 | `mean` | descriptive.rs | — | nearly everything |
| 8 | `spearman_r` | nonparametric.rs | rank | nonparametric tests |

### Status

| Primitive | Oracle | Competitor | Scale ladder | Signed off |
|-----------|--------|------------|--------------|------------|
| kendall_tau | ✓ | partial | pending | no |
| pearson_r | ✓ | ✓ (scipy+mpmath) | pending | no |
| all others | — | — | — | — |
