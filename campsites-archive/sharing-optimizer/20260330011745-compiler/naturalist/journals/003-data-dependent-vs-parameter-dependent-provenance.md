# Data-Dependent vs Parameter-Dependent Provenance

*Naturalist journal — 2026-03-30*

---

## The scout's finding

MobiusKalmanOp computes the covariance sequence P_t from noise parameters (q, r) and initial condition P_0. It doesn't read the observations. The scan element `[[r+q, qr], [1, r]]` is the same regardless of which ticker's data is being processed.

Two tickers with the same noise model → same P sequence. This is cross-ticker sharing. The 865x provenance system should catch it. But it doesn't — because provenance flows through the data leaf unconditionally.

## Why the current system misses it

In plan.rs, every node's provenance chains through data leaf provenances:

```rust
// Data leaf provenance
identity_provs.insert(data_id, *prov);  // "data:AAPL" → hash of AAPL data

// Node provenance
let input_provs: Vec<[u8; 16]> = node.input_identities.iter()
    .map(|inp_id| identity_provs.get(inp_id).copied().unwrap_or(...))
    .collect();
let prov = provenance_hash(&input_provs, &node.identity);
```

If a scan node takes "data:AAPL" as input, its provenance includes AAPL's data hash. The same scan on "data:MSFT" gets MSFT's data hash → different provenance → no sharing.

For MobiusKalmanOp, this is wrong. The P scan doesn't READ the data. Its result depends on (q, r, P_0, N) — parameters, not data. AAPL and MSFT with the same (q, r) model produce identical P sequences. The provenance should be `hash(q, r, P_0, N)`, not `hash(AAPL_data, computation_id)`.

## The new axis

| Operator type | Provenance depends on | Example |
|---|---|---|
| Data-dependent | data + params | AddOp, WelfordOp, EWMOp |
| Parameter-dependent | params only | MobiusKalmanOp (P scan) |
| Co-input-dependent | data + other stage output | AffineKalmanOp (x scan, uses P from stage 1) |

The current provenance system handles type 1 correctly. Type 2 is the new class — provenance should exclude data leaves. Type 3 is handled correctly IF the co-input's provenance is correctly computed (which it is, via the DAG's topological order).

## What would fix it

Two approaches:

### Approach A: `is_data_independent` flag on operators

```rust
pub trait AssociativeOp: Send + Sync {
    // ... existing methods ...

    /// If true, this operator's output depends only on parameters,
    /// not on the data input. Provenance should exclude data leaf hashes.
    fn is_data_independent(&self) -> bool { false }
}
```

Plan.rs would check this flag and substitute a parameter-derived provenance for the data leaf:

```rust
let input_provs = if op_is_data_independent {
    // Use only parameter provenances, not data provenances
    vec![data_provenance(&op_params_key)]
} else {
    // Current behavior: chain through data leaf provenances
    node.input_identities.iter()
        .map(|id| identity_provs[id])
        .collect()
};
```

Pro: minimal change. One flag, one branch.
Con: it's a boolean on a continuum. What about operators that are partially data-independent? (The co-input case — data-independent for SOME inputs, not others.)

### Approach B: Per-input data-dependence in the registry

The specialist recipe marks each input as "data" or "parameter":

```rust
pub struct PrimitiveStep {
    pub op: String,
    pub input_names: Vec<String>,
    pub input_kinds: Vec<InputKind>,  // NEW: Data or Parameter
    pub params: Vec<(String, String)>,
    pub output_name: String,
}

enum InputKind { Data, Parameter }
```

Plan.rs uses `InputKind` to decide whether to include each input's data provenance:

```rust
let input_provs: Vec<[u8; 16]> = step.input_names.iter()
    .zip(&step.input_kinds)
    .map(|(name, kind)| {
        match kind {
            InputKind::Data => identity_provs[name],       // hash data
            InputKind::Parameter => data_provenance(name),  // hash name only
        }
    })
    .collect();
```

Pro: handles the co-input case naturally. Stage 2's "P_t input" is marked as Data (its provenance matters), while stage 1's "observation input" is marked as Parameter (ignored).
Con: more registry complexity. Every specialist recipe must annotate its inputs.

## The sharing opportunity

At K04 (cross-ticker), this becomes load-bearing. If 100 tickers share the same noise model, the Möbius P scan runs ONCE and the result is shared 100 ways. Without parameter-dependent provenance, it runs 100 times — each producing identical results that the store can't recognize as identical because the data leaf hashes differ.

This is exactly the K04 super-linear sharing the Phase 2 expedition log predicted. But it requires the provenance system to distinguish data-dependent from parameter-dependent inputs. The family observation explains WHY: we only had data-dependent family members until now, so the provenance system was designed for them. The Kalman family introduces the first parameter-dependent member.

## The gradient update

The scout suggested a fourth point on the liftability gradient:

```
exactly liftable → exactly liftable (co-input) → approximately liftable → unliftable
AddOp              AffineKalmanOp                  EKFOp                   Nonlinear
                   (needs P from stage 1)
```

The co-input case is NOT an approximation and NOT a Fock boundary violation. It's a structural dependency between liftable stages. Each stage is individually liftable. The dependency creates a scheduling constraint (stage 1 before stage 2) but not a liftability constraint. The parallel scan within each stage is exact.

This is different from the Fock boundary in a subtle way. The Fock boundary says "this stage ITSELF can't be parallelized." The co-input pattern says "this stage CAN be parallelized, but only after another stage completes." Intra-stage parallelism is preserved. Inter-stage ordering is added. The compiler's topological sort already handles this — it's just a DAG edge.

---

## Refinement: parameter-independence propagates for free

The scout pointed out something I missed: you only need the `InputKind::Parameter` annotation at the LEAF connection — the single point where raw observation data would incorrectly pull in data provenance. Once the Möbius P scan's provenance is `BLAKE3(q, r, N, P_0)` (no data hash), everything downstream inherits the data-independence automatically:

```
P scan: provenance = hash(param_prov, "mobius_scan")     ← annotated here
K extraction: provenance = hash(P_prov, "kalman_gain")   ← inherits, no annotation needed
```

K extraction's provenance is `hash(P_provenance, computation_id)`. P_provenance has no data hash in it. So K's provenance has no data hash either. The parameter-independence flows through the DAG without special-casing at each node.

This means the annotation is NARROWER than Approach B suggested. Not per-input-per-step, but per-leaf-entry-point. For Möbius Kalman, that's ONE flag on ONE input connection.

## The K04 sharing breakdown (concrete)

For 100 tickers with the same (q, r) model:

| Node | Data-dependent? | Runs | Shared? |
|---|---|---|---|
| Möbius P scan | No (parameter-only) | 1 | Yes — all 100 tickers hit same cache entry |
| K extraction fused_expr | No (derived from P) | 1 | Yes — inherits parameter-independence |
| Affine x scan | Yes (prices differ) | 100 | No — per-ticker |

Without InputKind: 300 dispatches. With InputKind: 102 dispatches. The P and K nodes are computed once and shared 100 ways. The affine x scan is per-ticker because it reads actual price data.

This is computation-level cross-ticker sharing — the first concrete K04 sharing surface. The provenance cache doesn't need to know about K04 structure. It just sees "same hash → cache hit" regardless of which ticker asked.

---

*The provenance system was designed for data-dependent operators because those were the only operators that existed. The Kalman family reveals the first parameter-dependent operator. The fix is narrower than I first proposed — one flag at one leaf connection, propagation for free. The opportunity is large (cross-ticker sharing at K04). The theory already supports it (the provenance hash doesn't NEED data leaves for parameter-only inputs — it was just unconditionally including them).*
