# The Two Primitives: Scan and Scatter

## The claim

Every WinRapids signal computation is a composition of exactly two primitives:
scan and scatter. Nothing else. The entire Kingdom system is built from these.

This isn't a design choice. It's a structural consequence of what the market
data looks like and what GPU parallelism allows.

## The two primitives defined

**Scan** (prefix operation, sequential dependency):
```
output[i] = f(input[0], input[1], ..., input[i])
```
Each output depends on ALL previous inputs. Temporal order preserved.
Examples: EWM, Kalman, Welford running variance, cumulative sum.

```
output = [f(x[0]),
          f(x[0], x[1]),
          f(x[0], x[1], x[2]),
          ...]
```

**Scatter** (group aggregation, no ordering dependency):
```
output[g] = reduce({ input[i] : key[i] == g })
```
Each output depends only on elements in its group. Order within groups
doesn't matter. Temporal structure discarded.
Examples: sum-by-ticker, variance-by-minute-bin, count-by-day.

```
keys    = [0, 1, 0, 2, 1]
values  = [a, b, c, d, e]
output  = [a+c, b+e, d]     // group sums
```

## Why they're dual

| Property | Scan | Scatter |
|---|---|---|
| Dependency | Input[i] depends on Input[0..i] | None between elements |
| Order | Temporal order is essential | Order within group is irrelevant |
| Output size | Same as input (n → n) | Smaller (n → n_groups) |
| Parallelism | O(log n) passes (Blelloch) | O(1) pass (fully parallel) |
| GPU primitive | parallel prefix tree | atomicAdd |
| Liftable? | Yes, if combine is associative | Always (commutativity for free) |

Scan and scatter are the only two operations you can do on an array
that are:
1. Computable in O(n) work
2. Parallelizable on GPU
3. Expressible as reductions over ordered/unordered subsequences

Everything else — sort, join, unique — reduces to one of these or requires
a higher constant factor (O(n log n)) that makes it non-competitive on GPU.

## Every Kingdom transition is a scan, a scatter, or both

```
K01 (raw ticks, per-ticker time series):
  Signal: EWM(α, prices)         → SCAN  (AffineOp with a=1-α, b=α)
  Signal: Kalman filter           → SCAN  (SarkkaOp — parallel scan on state)
  Signal: Rolling Welford std     → SCAN  (WelfordOp — running variance)

K01 → K02 (minute bars from ticks):
  price_mean_per_minute          → SCATTER (group by ticker × minute)
  volume_sum_per_minute          → SCATTER (group by ticker × minute)

K02 (minute bar time series):
  Rolling EWM on minute bars     → SCAN  (same AffineOp, different cadence)
  Cross-cadence correlation       → SCAN  (scan over pairs of cadences)

K02 → K03 (cross-ticker via shared time):
  Cross-ticker correlation        → SCATTER + SCAN
  Market beta                     → SCATTER (group by day) + SCAN (rolling)
```

There are no other operation types. Sort doesn't appear.
Matrix multiply doesn't appear (it's O(n²) and wrong for time series).
The compiler has exactly two kernel families to generate.

## The universal scan: AffineOp

From manuscript 009 and the winrapids-scan work:
```
state[t] = a * state[t-1] + b * input[t]
```

With (a, b) = (1-α, α): EWM.
With (a, b) = (F, H): Kalman state transition.
With (a, b) = (decay, 1): AR(1) leaky integrator.

AffineOp is the parameterization of the universal scan: one combine function,
infinitely many operations via parameter choice.

## The universal scatter: mapped atomicAdd

```
output[g] += φ(input[i])   for all i where key[i] == g
```

With φ(x) = x: sum scatter (scatter_sum).
With φ(x) = (x, x², 1): three-stat scatter (scatter_stats).
With φ(x) = (x - ref[g], (x - ref[g])², 1): centered scatter (RefCenteredStatsOp).
With φ(x) = log(x): log-sum scatter.
With φ(x) = max(x, 0): positive-sum scatter (for drawdown analysis).

The scatter is parameterized by φ (element transform), not by the reduce itself.
The reduce is always sum (because atomicAdd). The variation lives in φ.

Claim: every scatter needed for financial signals is a mapped scatter.
The compiler only needs to JIT the φ function, not the scatter mechanism.

## The composition rules

The compiler doesn't need N² rules. It needs 4:

```
scatter ∘ scatter → depends on key identity:
  same key: merge into one scatter (different φ, same groups)
  different keys: sequential (can't fuse, different output sizes)

scan ∘ scan → fuse into one scan:
  (scan f) ∘ (scan g) = scan (f ∘ g) if state types compatible

scan ∘ scatter → pipeline:
  scatter first (n → n_groups), then scan on smaller output

scatter ∘ scan → pipeline:
  scan first (n → n, same size), then scatter on transformed values
```

The most valuable fusion is `scatter ∘ scatter` with the same key:
computing price_sum, price_sum_sq, and volume_sum in ONE scatter pass.
This is exactly scatter_stats + scatter_sum composed. Three for the price of one.

## What this means for the compiler

The compiler's kernel library is O(|φ functions|) not O(|signals|).
Every signal is: map φ over input → scatter OR scan with AffineOp parameters.

The JIT generator has two templates:
1. `scatter_template(phi_expr: &str) -> CUDA source`
2. `scan_template(a_expr: &str, b_expr: &str, state_init: &str) -> CUDA source`

With these two templates and runtime parameter substitution, the compiler can
generate any K01 → K02 signal (scatter) or K01 rolling signal (scan).

The "135 specialists" in the specialist registry are:
- ~60 scatter specialists: different φ functions
- ~60 scan specialists: different (a, b) parameterizations
- ~15 compositions: specific scan ∘ scatter patterns (like EWM + groupby)

The primitives don't proliferate. The parameters do.

## The open question: compositions beyond two operations

What about signals that require three operations?

Example: "rolling volatility of per-minute returns"
1. Scatter: ticks → per-minute means (K01 → K02)
2. Scan: per-minute means → log returns (scan with shift)
3. Scan: log returns → rolling std (Welford scan)

This is scatter ∘ scan ∘ scan. The fusion rule: fuse the two scans first
(scan ∘ scan = fused scan), then compose with scatter.

Or equivalently: K01 → K02 (scatter), K02 → K02 log returns (scan), K02 → K02 rolling std (scan). Three operations, two primitives, one composition.

The claim remains: everything is scan and scatter. The nesting depth grows for
complex signals, but the primitives don't change.

## The connection to liftability

The manuscript on liftability (Pith's principle) shows that scan is liftable
when the combine is a semigroup homomorphism. This is Blelloch's theorem.

Scatter is ALWAYS liftable: each element contributes independently to its group.
No semigroup condition needed — independence is stronger than semigroup structure.

Scan has a liftability condition; scatter has none.
This is why scatter is 17x faster than sort-based groupby:
sort requires a global ordering (not liftable), scatter requires none.

The compiler's parallelism guarantee:
- Any scatter: always O(n) on GPU.
- Any scan: O(n log n) work but O(log n) depth on GPU, IF the combine is associative.

Associativity of the scan combine is the only condition the compiler needs to verify.
Everything else parallelizes for free.
