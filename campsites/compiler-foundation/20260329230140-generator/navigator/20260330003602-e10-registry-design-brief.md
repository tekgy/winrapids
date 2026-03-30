# E10 Design Brief: Full Specialist Registry

Created: 2026-03-30T00:36:02-05:00
By: navigator

---

## What E10 Is

Full implementation of the specialist registry in Rust. ~135 specialists, each as a
5-field recipe of the 9 primitives. E04 proved the loop with 3 specialists in Python;
E10 fills the registry.

The loop is proven. The registry is the work.

## Architecture Constraints (non-negotiable, from Phase 2)

### The 9 Primitives (closed set)

| Primitive    | Identity Shape                              | CSE Granularity     |
|--------------|---------------------------------------------|---------------------|
| scan         | (op, data_id, op_type)                      | sequence-level      |
| sort         | (data_id, key_cols, order)                  | order-level         |
| reduce       | (data_id, agg_op)                           | dataset-level       |
| tiled_reduce | (left_id, right_id, op)                     | block-level matrix  |
| scatter      | (indices_id, values_id, target_id, op)      | index-write-level   |
| gather       | (indices_id, source_id)                     | index-read-level    |
| search       | (sorted_data_id, query_id, variant)         | lookup-level        |
| compact      | (mask_id, data_id)                          | density-level       |
| fused_expr   | (expression_tree_hash, input_ids...)        | expression-level    |

**Primitive admission test** (formal): A proposed primitive is accepted if and only if
its minimal CSE identity structure cannot be expressed as an existing primitive type
without losing block-level sharing. Any algorithm that DOES fit an existing primitive
type (even with a different kernel template) is a VARIANT, not a new primitive.

**The primitives are the sharing granularities**, not algorithm categories. Adding a
10th primitive requires showing a new sharing granularity that none of the 9 covers.

### 5-Field Specialist Entry (from E04, validated)

```rust
pub struct SpecialistRecipe {
    pub name: &'static str,
    pub primitive_dag: &'static [PrimitiveStep],  // ordered, named outputs
    pub fusion_eligible: bool,
    pub fusion_crossover_rows: u64,  // default u64::MAX (always fuse, anti-YAGNI)
    pub independent: bool,           // false if dag contains scan/sort/reduce/tiled_reduce
    pub identity: &'static [&'static str],  // params that determine canonical identity
}

pub struct PrimitiveStep {
    pub output_name: &'static str,
    pub op: PrimitiveOp,
    pub inputs: &'static [&'static str],
    pub params: &'static [(&'static str, ParamValue)],
}
```

**Static registry, not runtime-loaded**: the registry is known at compile time.
`&'static` slices, `const` where possible. Zero runtime overhead on lookup.
Compile-time verification that all recipes reference valid primitive types.

### Injectable World State API (from E04, validated)

```rust
pub fn plan(
    spec: &PipelineSpec,
    registry: &SpecialistRegistry,
    provenance: Option<&dyn ProvenanceCache>,  // None = always miss
    dirty_bitmap: Option<&dyn DirtyBitmap>,    // None = everything dirty
    residency: Option<&dyn ResidencyMap>,       // None = nothing warm
) -> ExecutionPlan
```

All four slots present from day one. Post-E07 integration: inject live provenance.

## Priority Order for Specialist Decompositions

Priority = sharing value (how many other specialists share the same primitive nodes?).

### Tier 1: Scan-heavy (highest sharing value)

All generate scan(data, add) and scan(data_sq, add). Every pair shares both scans.
Most FinTek pipeline sharing lives here.

```
rolling_mean(data, w)       -> cs = scan(data, add); out = fused_expr(cs, "rolling_mean")
rolling_std(data, w)        -> cs, cs2 = scan(data,add), scan(data_sq,add); fused_expr(...)
rolling_zscore(data, w)     -> shares cs, cs2 with rolling_std
rolling_var(data, w)        -> shares cs, cs2 with rolling_std
bollinger_bands(data, w)    -> shares cs, cs2 with rolling_std; fused_expr for bands
ewm(data, alpha)            -> scan(data, ewm_op)  -- different op, may share with kalman
cumsum(data)                -> scan(data, add)  -- base primitive, always shared
cumprod(data)               -> scan(data, mul)
cummax(data)                -> scan(data, max)
diff_n(data, lags)          -> one scan per lag (multi-lag, independent passes)
```

K04 pattern: each ticker needs rolling_std -> scan sharing is within-ticker.
Across tickers: tiled_reduce(stats_A, stats_B, correlation_op) for pairwise correlation.

### Tier 2: Sort-heavy (second highest; groupby dominates)

```
sort(data, keys)                     -> sort primitive
groupby_sum(data, keys)              -> sort(keys) + reduce(sorted, sum, segments)
groupby_mean(data, keys)             -> shares sort(keys) with groupby_sum
groupby_std(data, keys)              -> shares sort(keys); adds reduce variants
groupby_count(data, keys)            -> shares sort(keys) with any groupby
rank(data)                           -> sort(data) + fused_expr(ranks)
value_counts(data)                   -> sort(data) + reduce(sorted, count, segments)
drop_duplicates(data, keys)          -> sort(keys) + compact(dedup_mask)
merge_sort_join(left, right, keys)   -> sort(left.keys) + sort(right.keys) + gather(...)
```

### Tier 3: DataFrame core (high volume, moderate sharing)

```
filter(data, pred)          -> fused_expr(pred) + compact(mask, data)
fillna_const(data, val)     -> fused_expr(fill_formula)
fillna_ffill(data)          -> scan(data, last_valid_op)  [shares with ewm!]
concat(arrays)              -> scatter(ranges, arrays, target)
select(data, cols)          -> gather(col_indices, data)
standardscaler(data)        -> reduce(data, mean_std) + fused_expr(scale)  [shares with rolling_std!]
minmax_scaler(data)         -> reduce(data, min_max) + fused_expr(scale)
```

StandardScaler shares reduce(data, sum) and reduce(data, sum_sq) with any rolling_std
on full-dataset window. CSE catches this automatically.

### Tier 4: ML algorithms (complex; highest tiled_reduce sharing)

```
pca(data)             -> reduce(data, mean) + tiled_reduce(centered, centered.T, matmul)
kmeans(data, k)       -> tiled_reduce(data, centroids, distance) + reduce(distances, argmin)
knn(query, data, k)   -> tiled_reduce(query, data, distance) + sort(distances, partial_k)
linear_regression     -> tiled_reduce(X.T, X, matmul) + tiled_reduce(X.T, y, matmul)
```

PCA and KNN both need tiled_reduce(data, data.T, matmul) -- they share the covariance
matrix automatically via CSE once both are registered.

### FinTek Domain (first-class, not P1)

```
vwap(price, volume)                    -> reduce(price*volume, sum) / reduce(volume, sum)
log_returns(price)                     -> fused_expr(log(p[i]/p[i-1]))
pairwise_correlation(stats_A, stats_B) -> tiled_reduce(stats, stats.T, correlation_op)
online_covariance(data)                -> scan(data, welford_op)  [scan variant]
bin_stats(data, bins)                  -> search(bin_edges, data) + reduce(bucketed, multi_stat)
```

## CSE Hash: BLAKE3 (consistent with PTX cache)

E04 prototype used MD5. Production: BLAKE3 (already in use for PTX cache, E09).

```rust
fn node_identity(op: PrimitiveOp, inputs: &[&str], params: &[(&str, &str)]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(format!("{:?}", op).as_bytes());
    for inp in inputs { h.update(inp.as_bytes()); }
    for (k, v) in params { h.update(k.as_bytes()); h.update(v.as_bytes()); }
    h.finalize().into()
}
```

CSE map: HashMap<[u8; 32], PrimitiveNode>. One hash algorithm across the entire compiler.

## What E10 Is NOT

- NOT a port of E04 Python to Rust. Start fresh, Rust-native design.
- NOT a runtime registry loaded from YAML. Static, compile-time.
- NOT the persistent store integration (E07, separately validated).
- NOT a complete WinRapids implementation.

## What E10 IS

- Full static registry (~135 specialists, ~9 primitives)
- plan() in Rust: CSE pass + topological sort + world state injection
- Codegen for fused_expr: NVRTC from Rust (E09 validated)
- Execution via cudarc: launch plan steps (E08 validated)
- PyO3 boundary: Python -> Rust -> plan -> execute -> Python (E13 validates the roundtrip)

After E10: the compiler loop exists in Rust with full registry. Next: wire to PyO3,
then persistent store integration, then FinTek farm migration to the Rust compiler.

## The Invariant That Cannot Change

**The registry is type inference for operations.** The user writes `rolling_std(price, 20)`.
The compiler infers the primitive decomposition. The user never sees primitives.

If any E10 decision forces the user to specify primitives explicitly, that is a registry
design failure. The domain-level API is the invariant. Primitive-level execution is the
mechanism. They must never be exposed to each other.
