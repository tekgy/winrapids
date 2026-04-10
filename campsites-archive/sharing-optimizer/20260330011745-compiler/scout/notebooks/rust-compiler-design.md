# Rust Compiler Design: E04 Port

*Scout notebook — 2026-03-30 (updated with navigator correction)*

Read: `E04-pipeline-generator/pipeline_generator.py` (full), `winrapids-scan/src/ops.rs`,
`winrapids-scan/src/cache.rs`, `winrapids-store/src/lib.rs`, `winrapids-store/src/world.rs`,
`winrapids-store/src/provenance.rs`, `winrapids-store/src/header.rs`, `docs/vision.md`.

---

## The One Thing

The CSE pass IS the compiler. The rest is bookkeeping.

```python
# E04 Python CSE pass — the entire value of the compiler in 5 lines:
seen: dict[str, tuple] = {}
for (node, binding) in all_nodes_raw:
    if node.identity not in seen:
        seen[node.identity] = (node, binding)
```

Every eliminated node is a GPU computation that doesn't happen. The type signatures
should make this prominent. `run_cse()` is the star function; `plan()` is the
orchestrator that feeds it.

---

## Decision 1: PrimitiveNode and NodeId

### Python
```python
@dataclass(frozen=True)
class PrimitiveNode:
    op: str
    inputs: tuple        # of identity strings
    params: tuple        # sorted (k, v) pairs
    output_name: str     # NOT in identity

    @property
    def identity(self) -> str:
        raw = f"{self.op}:{self.inputs}:{self.params}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
```

### Rust translation

```rust
/// The canonical identity of a primitive computation.
/// BLAKE3 of (op_bytes | input_ids | sorted_params) → first 16 bytes → u128.
/// 128 bits: no collisions at any scale WinRapids will reach.
/// u128: zero-copy hash map key, fits in two registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u128);

impl NodeId {
    /// Short hex prefix for display/debug (8 chars, like E04's 12-char MD5).
    pub fn short(&self) -> String {
        format!("{:016x}", self.0)[..8].to_string()
    }
}

#[derive(Debug, Clone)]
pub struct PrimitiveNode {
    pub op: PrimitiveOp,
    pub inputs: SmallVec<[NodeId; 4]>,  // usually 1-3 inputs; stack-allocated
    pub params: Vec<Param>,             // sorted canonical — same as Python's tuple
    pub label: &'static str,           // debug only: "cs", "cs2", "out"
}
```

**Key change from Python**: `output_name` → `label`. Renaming signals that this field is
NOT part of identity. The Python's `identity` property already excludes it; Rust makes
the exclusion structural.

**Hash uses BLAKE3** (already in winrapids-scan/cache.rs — same pattern):

```rust
impl PrimitiveNode {
    pub fn compute_id(&self) -> NodeId {
        let mut h = blake3::Hasher::new();
        h.update(&self.op.discriminant_bytes());
        for input in &self.inputs {
            h.update(&input.0.to_be_bytes());
        }
        for p in &self.params {
            h.update(p.key.as_bytes());
            h.update(&p.value.canonical_bytes());
        }
        // label is intentionally excluded — two nodes with the same computation
        // but different names ARE the same computation for CSE purposes.
        let bytes: [u8; 16] = h.finalize().as_bytes()[..16].try_into().unwrap();
        NodeId(u128::from_be_bytes(bytes))
    }
}
```

**MD5 → BLAKE3**: no concern. The identity is used for CSE deduplication, not
cryptographic security. BLAKE3 is faster than MD5 in Rust anyway.

**Frozen dataclass → Rust struct**: `PrimitiveNode` is not `Copy` (Vec inside), but
`NodeId` is `Copy`. The plan stores `NodeId`s as edges; nodes are owned by the plan.

---

## Decision 2: SpecialistRegistry

### Python
```python
def build_registry() -> dict[str, SpecialistRecipe]:
    return {
        "rolling_mean": SpecialistRecipe(...),
        "rolling_std": SpecialistRecipe(...),
    }
```

### Decision: `HashMap` in `OnceLock`, NOT `phf::Map`

`phf::Map` is elegant for a closed static set. But the full vision has ~135 specialists,
and the architecture needs to allow domain-specific specialists (FinTek, bio, graph) to
register without a full recompile. `OnceLock<HashMap>` costs one allocation at startup
and nothing at lookup. It also allows test injection.

```rust
pub type Registry = HashMap<&'static str, SpecialistRecipe>;

static BUILTIN_REGISTRY: OnceLock<Registry> = OnceLock::new();

pub fn builtin_registry() -> &'static Registry {
    BUILTIN_REGISTRY.get_or_init(|| {
        let mut m = Registry::new();
        register_rolling_specialists(&mut m);
        // register_ml_specialists(&mut m);   // Phase 4+
        m
    })
}
```

The `plan()` function takes `&Registry` — no global reference required. Tests can inject
a minimal registry. Production uses `builtin_registry()`. Anti-YAGNI: the full registry
is always initialized, but individual test cases only need the specialists they test.

**Ownership**: `SpecialistRecipe` is `Clone` + `Send + Sync`. The global is `'static`.
References to it are `&'static SpecialistRecipe`.

---

## Decision 3: CSE Pass Type

### Python result
```python
seen: dict[str, (PrimitiveNode, binding)]   # identity → step
# then topo-sorted into:
steps: list[(PrimitiveNode, binding)]
```

### Decision: `IndexMap<NodeId, PlanStep>` for CSE, `Vec<PlanStep>` for execution

Two-phase internal representation:

**Phase 1 — CSE deduplication** (the value creation):
```rust
pub struct PlanStep {
    pub id: NodeId,
    pub node: PrimitiveNode,
    pub binding: HashMap<&'static str, NodeId>,  // input_name → resolved NodeId
}

// CSE pass: identity deduplication
// IndexMap = insertion-ordered HashMap — preserves processing order for stable output
fn run_cse(raw: Vec<(PrimitiveNode, Binding)>) -> IndexMap<NodeId, PlanStep> {
    let mut seen: IndexMap<NodeId, PlanStep> = IndexMap::new();
    for (node, binding) in raw {
        let id = node.compute_id();
        seen.entry(id).or_insert_with(|| PlanStep { id, node, binding });
    }
    seen
}
```

**Phase 2 — Topological sort** (Kahn's algorithm on NodeId dependency edges):
```rust
fn topo_sort(cse: IndexMap<NodeId, PlanStep>) -> Vec<PlanStep> {
    // build dep_graph: NodeId → set of input NodeIds that are computed (not raw data)
    // Kahn: start from nodes with no computed deps, emit, decrement neighbors
    // ...
}
```

**Phase 3 — World state probe** (eliminates cross-session duplicates):
```rust
fn prune_warm(steps: Vec<PlanStep>, world: &WorldState) -> Vec<PlanStep> {
    steps.into_iter()
         .filter(|s| !world.residency.is_resident(s.id) && !world.provenance.has(s.id))
         .collect()
}
```

**Final type**:
```rust
pub struct ExecutionPlan {
    pub steps: Vec<PlanStep>,                        // topo-sorted, CSE-deduped, world-pruned
    pub outputs: HashMap<OutputKey, NodeId>,          // (call_idx, "out") → NodeId
    pub cse_stats: CseStats,
}

pub struct CseStats {
    pub original_nodes: usize,
    pub after_cse: usize,
    pub eliminated_by_cse: usize,
    pub pruned_by_world: usize,   // new: tracks world-state eliminations separately
}
```

**Why ordered Vec, not a DAG type?**: The topo sort is done once during planning. The
executor just walks the Vec sequentially. The NodeId references inside each `PlanStep`'s
`binding` ARE the DAG edges — no separate graph structure needed. Adding a DAG type would
be complexity without benefit (YAGNI, and YAGNI is wrong here because we'll definitely
need sequential execution, and may not need random-access into the graph).

---

## Decision 4: Codegen → Scan Delegation

### Python
```python
if node.op == "scan":
    inp_f64 = inp_arr.astype(cp.float64)
    cs = cp.cumsum(inp_f64)  # dispatches to CuPy
elif node.op == "fused_expr":
    kernel = kernels[formula]  # dispatches to KERNEL_TEMPLATES
    kernel(...)
```

### Rust: direct crate dependency, not a trait abstraction

The dependency is one-directional: `winrapids-compiler` → `winrapids-scan`. No abstraction
layer needed. The compiler calls `winrapids_scan::engine::generate_scan_kernel(op)` directly.

The bridge: `PrimitiveOp::Scan { agg: AggType }` → `Box<dyn AssociativeOp>`:

```rust
// In winrapids-compiler/src/ops.rs:

#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveOp {
    Scan { agg: AggType },
    FusedExpr { formula: FusedFormula },
    Reduce { agg: AggType },
    Sort { order: SortOrder },
    // Gather, Scatter, Search, Compact — future primitives
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggType {
    Add,
    Mul,
    Max,
    Min,
    Welford,       // → WelfordOp (mean + variance)
    EWM { alpha: f64 },
}

impl AggType {
    /// Convert to a winrapids-scan AssociativeOp.
    /// This is the bridge: AggType → scan kernel codegen.
    pub fn as_scan_op(&self) -> Box<dyn winrapids_scan::ops::AssociativeOp> {
        use winrapids_scan::ops::*;
        match self {
            AggType::Add => Box::new(AddOp),
            AggType::Mul => Box::new(MulOp),
            AggType::Max => Box::new(MaxOp),
            AggType::Min => Box::new(MinOp),
            AggType::Welford => Box::new(WelfordOp),
            AggType::EWM { alpha } => Box::new(EWMOp { alpha: *alpha }),
        }
    }
}
```

**In the codegen pass**:
```rust
pub fn codegen(plan: &ExecutionPlan, kernel_cache: &mut KernelCache) -> Vec<CompiledStep> {
    plan.steps.iter().map(|step| {
        match &step.node.op {
            PrimitiveOp::Scan { agg } => {
                let scan_op = agg.as_scan_op();
                let ptx = kernel_cache.get_or_compile(scan_op.as_ref())?;
                CompiledStep::Scan { id: step.id, ptx, binding: step.binding.clone() }
            }
            PrimitiveOp::FusedExpr { formula } => {
                let cuda_src = render_fused_expr(formula, &step.node.params);
                let ptx = compile_fused_expr(cuda_src, kernel_cache)?;
                CompiledStep::FusedExpr { id: step.id, ptx, binding: step.binding.clone() }
            }
            // Reduce, Sort, etc. — future
        }
    }).collect()
}
```

**KernelCache**: the one from winrapids-scan already handles BLAKE3 keying + NVRTC +
disk cache. The compiler reuses it for both scan and fused_expr kernels (same pattern,
different source strings).

---

## Decision 5: Injection Points (WorldState)

### Correction from navigator: the traits already exist in winrapids-store

After reading `winrapids-store/src/world.rs`, `provenance.rs`, and `header.rs`:

**The WorldState traits are already defined in winrapids-store.** My original design had the
compiler defining its own traits. That was wrong — winrapids-compiler should DEPEND ON
winrapids-store, not the other way around.

```rust
// winrapids-store/src/world.rs — already exists:
pub trait ProvenanceCache {
    fn provenance_get(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr>;
    fn provenance_put(&mut self, provenance: [u8; 16], ptr: BufferPtr, cost_us: f32);
}
pub trait DirtyBitmap { fn is_clean(&self, provenance: &[u8; 16]) -> bool; }
pub trait ResidencyMap {
    fn is_resident(&self, provenance: &[u8; 16]) -> bool;
    fn resident_pointer(&self, provenance: &[u8; 16]) -> Option<BufferPtr>;
}
pub trait WorldState: ProvenanceCache + DirtyBitmap + ResidencyMap {}
pub struct NullWorld; // impl all three — always misses
```

**The provenance key is `[u8; 16]`** — not `NodeId`. It's the output of `provenance_hash()`:
```rust
// winrapids-store/src/provenance.rs — already exists:
pub fn provenance_hash(input_provenances: &[[u8; 16]], computation_id: &str) -> [u8; 16]
pub fn data_provenance(data_identity: &str) -> [u8; 16]
```

**`BufferPtr` is the zero-translation cache entry** — just `(device_ptr, byte_size)`.
`BufferHeader` (the 64-byte self-describing header) carries `provenance: [u8; 16]` and
`cost_us`, `access_count`, etc.

### The Three Concentric Identities (navigator's observation)

Reading the codebase confirms the nesting:

```
kernel_cache_key = BLAKE3(CUDA source)
                 ⊂
NodeId           = BLAKE3(op_bytes | input_NodeIds | params)    ← NEW in winrapids-compiler
                 ⊂
provenance_tag   = BLAKE3(input_data_prov_chain | NodeId_bytes) ← existing [u8; 16]
```

- `kernel_cache_key` asks: "is this kernel compiled?" — pure code identity, data-agnostic, computation-agnostic
- `NodeId` asks: "what computation is this?" — pure structural identity, data-agnostic
- `provenance_tag` asks: "was this computation done on this data?" — combines data + computation

`NodeId` is the `computation_id` parameter in `provenance_hash`. Currently in winrapids-store,
`computation_id` is a string like `"scan:add:w=20"`. `NodeId` makes that canonical: instead of
ad-hoc strings, the computation identity is a BLAKE3 hash of the full structural description.

### NodeId → ProvenanceTag bridge

The compiler needs to produce provenance tags to probe the world state. Given:
- `NodeId` for this computation
- `[u8; 16]` tags for each input (carried by the buffers from prior steps)

```rust
impl NodeId {
    /// Compute the provenance tag for this computation applied to these inputs.
    /// This is what the ProvenanceCache is keyed by.
    ///
    /// Uses winrapids_store::provenance_hash — the NodeId IS the computation_id.
    pub fn to_provenance_tag(&self, input_tags: &[[u8; 16]]) -> [u8; 16] {
        // NodeId is the computation_id — format as hex for the string parameter
        let computation_id = format!("{:032x}", self.0);
        winrapids_store::provenance_hash(input_tags, &computation_id)
    }
}
```

During execution, the executor tracks `ProvenanceTag` per step:
1. Compute `step_tag = step.node.id.to_provenance_tag(&input_tags)`
2. Check `world.is_clean(&step_tag)` — skip if clean
3. Check `world.is_resident(&step_tag)` — pointer handoff if resident
4. Check `world.provenance_get(&step_tag)` — return if cached
5. Execute → `world.provenance_put(step_tag, result_ptr, cost_us)`

### GpuBufferHandle: does it carry NodeId?

The navigator asked whether `GpuBufferHandle` (= `BufferPtr` in winrapids-store) should
carry the `NodeId` that produced it. Looking at `BufferHeader`:

The 64-byte header carries `provenance: [u8; 16]` — the full `provenance_tag` — plus
`cost_us`, `access_count`, `location`, `dtype`, `len`, etc. It does NOT separately carry
`NodeId` (since `NodeId` is baked into the provenance_tag via `computation_id`).

Adding `NodeId` to `BufferHeader` would be redundant IF the computation_id format is
stable. But for the cost model (eviction decisions), knowing WHICH computation produced
a buffer is useful beyond just its provenance hash. The navigator is right: `GpuBufferHandle`
as used by the compiler's executor should carry `NodeId` for cost estimation.

**Practical resolution**: don't change `BufferHeader` (it's a fixed 64-byte struct with
no room for a `NodeId`). Instead, the compiler's executor maintains a separate
`HashMap<ProvenanceTag, NodeId>` that maps "this provenance → which computation produced it."
The cost model queries this table to decide eviction priority, not the BufferHeader.

### Updated WorldState usage in compiler

```rust
// winrapids-compiler/src/plan.rs:
use winrapids_store::{WorldState, NullWorld, provenance_hash, data_provenance};

pub fn plan(
    spec: &PipelineSpec,
    registry: &Registry,
    world: &mut dyn WorldState,
) -> ExecutionPlan {
    let raw = decompose_and_bind(spec, registry);
    let cse_result = run_cse(raw);           // THE SHARING PASS
    let topo = topo_sort(cse_result);
    ExecutionPlan { steps: topo, ... }
}
```

`NullWorld` from winrapids-store is the E04 baseline. `GpuStore` (also winrapids-store)
is the live implementation. The compiler just takes `&mut dyn WorldState` — no new trait
definitions needed.

### Updated crate dependency

```
winrapids-compiler
  ├── winrapids-scan    (for AssociativeOp, generate_scan_kernel, KernelCache)
  └── winrapids-store   (for WorldState traits, BufferPtr, provenance_hash, NullWorld)
```

Not the other way around. This was the key correction.

---

## Module Layout for winrapids-compiler

```
crates/winrapids-compiler/
├── Cargo.toml
│   deps: blake3, smallvec, indexmap, winrapids-scan, winrapids-store
└── src/
    ├── lib.rs           — pub use {plan, ExecutionPlan, PipelineSpec, ...}
    ├── node.rs          — PrimitiveNode, NodeId, PrimitiveOp, AggType, Param
    │                      + NodeId::to_provenance_tag(&[input_tags]) → [u8; 16]
    ├── recipe.rs        — SpecialistRecipe, PrimitiveStep, Registry, builtin_registry()
    ├── spec.rs          — PipelineSpec, SpecialistCall
    ├── plan.rs          — plan(spec, registry, world: &mut dyn WorldState) orchestrator
    ├── cse.rs           — run_cse() — THE STAR FUNCTION
    ├── topo.rs          — topo_sort() (Kahn's algorithm)
    ├── codegen.rs       — codegen() → CompiledStep; bridges to winrapids-scan
    └── execute.rs       — execute() — walks CompiledStep, dispatches via cudarc
```

No `world.rs` — WorldState traits live in winrapids-store, imported here.
`NullWorld` from winrapids-store is the E04 baseline.
`cse.rs` gets its own file. It's the center of the compiler.

---

## The Sharing Hierarchy in Types

The four CSE scopes use two related but distinct identity types:

| Scope | Where | Key type | What it captures |
|---|---|---|---|
| Within-plan CSE | `cse.rs:run_cse()` | `NodeId` (u128) | structural computation identity |
| Dirty tracking | `winrapids-store:DirtyBitmap` | `[u8; 16]` provenance_tag | data + computation |
| Residency | `winrapids-store:ResidencyMap` | `[u8; 16]` provenance_tag | data + computation |
| Cross-plan provenance | `winrapids-store:ProvenanceCache` | `[u8; 16]` provenance_tag | data + computation |

`NodeId` is the structural key — data-agnostic, used for within-plan deduplication.
`provenance_tag` = `provenance_hash(input_data_chain, NodeId_hex)` — data-aware, used for
cross-plan and cross-session reuse.

The three identities are concentric:
```
kernel_cache_key ⊂ NodeId ⊂ provenance_tag
     (code)        (structure)   (data + structure)
```

`NodeId` IS the value number for CSE. `provenance_tag` IS the value number for the store.
`NodeId` is how you build `provenance_tag`.

---

## Flag for Navigator: EWMOp Segment-Length Bug

`winrapids-scan/src/ops.rs` — `EWMOp::cuda_combine`:
```rust
format!(r#"({
    double decay = pow(1.0 - {alpha}, (double)1);  // <-- exponent is ALWAYS 1
    ...
})"#, alpha = self.alpha)
```

`pow(1.0 - alpha, 1)` = `1 - alpha` regardless of segment b's length. A correct parallel
EWM merge needs `pow(1.0 - alpha, len_b)` where `len_b` is the number of elements in
segment b. Without this, the parallel EWM result diverges from the sequential result when
segment lengths > 1.

This doesn't block the compiler port (EWM isn't in the E04 registry), but it should be
fixed before EWMOp is used in production. The EWM state struct probably needs to carry
the segment length: `struct EWMState { weight: f64, value: f64, count: i64 }` so that
the combine function has access to `b.count`.

---

## What the Pathmaker Needs

1. Create `crates/winrapids-compiler/` with the module layout above
2. `node.rs` + `recipe.rs` + `spec.rs` first — pure data types, no GPU dependencies
3. `cse.rs` as a standalone function that takes `Vec<(PrimitiveNode, Binding)>` and returns
   `IndexMap<NodeId, PlanStep>` — testable without GPU
4. `topo.rs` — Kahn's algorithm on NodeId dep edges
5. `world.rs` — traits + null impls (ships in compiler; store implements the traits later)
6. `plan.rs` — wire phases 1-5 together
7. `codegen.rs` — bridge to winrapids-scan via `AggType::as_scan_op()`
8. `execute.rs` — the cudarc dispatch loop (last, needs GPU)

Phases 1-6 are testable without a GPU. The test suite can verify CSE logic, topo sort
correctness, and world-state integration purely in Rust. Phase 7 (execute) needs the
scan GPU launch work from Task #2.

---

## What This Looks Like in Use

```rust
// The 3-line compiler invocation:
let spec = PipelineSpec::new()
    .call("rolling_zscore", "price", w=20)
    .call("rolling_std",    "price", w=20);

let plan = plan(&spec, builtin_registry(), &WorldState::default());

// CSE found it: 4 primitive nodes → 2 (cs and cs2 shared)
assert_eq!(plan.cse_stats.eliminated_by_cse, 2);
```
