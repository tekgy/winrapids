# RefCenteredStatsOp: Navigator Notes

## The numerical stability split

Scout's `GroupByResult::variances()` uses naive formula:
```
var = (sum_sq - sum²/n) / (n-1)
```

This is NUMERICALLY UNSTABLE for large values with small variance — catastrophic
cancellation. Example: if prices ~ $150.00 ± $0.01, then sum_sq ≈ 150² × n and
sum² / n ≈ same large number. The subtraction loses all significant bits.

RefCenteredStatsOp fixes this by centering BEFORE scatter:
```
centered = value - ref  (where ref ≈ group mean, a priori estimate)
scatter sum(centered), sum(centered²)
var = sum(centered²)/n - (sum(centered)/n)²
```

Centered values are ~$0.01, not ~$150. No catastrophic cancellation.

## Two tiers, one kernel

**Fast tier (Scout's kernel)**: sum + sum_sq + count raw values.
- Use when: values are already small (normalized, z-scored, log-returns)
- Use when: precision requirements are moderate
- Cost: 3 atomicAdds per element, no center lookup

**Accurate tier (Naturalist's kernel)**: sum + sum_sq + count CENTERED values.
- Use when: values are large with small variance (prices, volumes)
- Use when: precision matters (volatility calculations, risk models)
- Cost: 3 atomicAdds + 1 read of ref value per element

The ref value can be: 0.0 (no centering), group mean from prior run (provenance!),
or a fast estimate (first element in each group).

## Connection to provenance

The killer insight: on the SECOND groupby run, we have the per-group means from
the FIRST run (stored via provenance). Use them as ref values for RefCentered
variance on the second run.

First run:  scatter_stats (naive) → get approximate means → store in provenance
Second run: scatter_stats_centered (RefCentered, ref=means from provenance) → exact

The first run pays the naive cost. Every subsequent run pays the RefCentered cost
(one extra read of ref value per element) but gets exact results.

## Naturalist's implementation target

`RefCenteredStatsEngine` in `crates/tambear/src/stats.rs`:

```cuda
extern "C" __global__ void scatter_stats_centered(
    const int* keys, const double* values, const double* refs,
    double* sums, double* sum_sqs, double* counts, int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int g = keys[gid];
        double c = values[gid] - refs[g];   // center around group ref
        atomicAdd(&sums[g],     c);
        atomicAdd(&sum_sqs[g],  c * c);
        atomicAdd(&counts[g],   1.0);
    }
}
```

Post-scatter (O(n_groups)):
```
actual_mean[g] = refs[g] + sums[g] / counts[g]
var[g] = sum_sqs[g]/counts[g] - (sums[g]/counts[g])²
```

## The "fast tier" naming

From the manuscript: this is "RefCentered = fast tier variance."
- "Fast" refers to O(n) scatter with no per-group iterations
- "RefCentered" refers to the centering that gives numerical stability
- It's BOTH fast AND accurate — the name "fast tier" means it uses the
  scatter-add tier (not the scan tier like WelfordOp)

WelfordOp is the scan tier: exact, running, one pass, sequential dependency structure.
RefCenteredStatsOp is the scatter tier: exact (after first run), parallel, no dependencies.
Scout's naive scatter: fast, parallel, but numerically unstable for real prices.

The right tool for each job:
- Rolling window variance on a time series → WelfordOp (scan)
- Per-group variance across a DataFrame → RefCenteredStatsOp (scatter)
