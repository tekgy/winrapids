# tambear-primitives Architecture

## What This Crate IS

The flat catalog of ALL mathematical primitives. One folder per primitive. The filesystem IS the inventory. Every primitive decomposes into accumulate+gather steps expressed as a Recipe that TAM compiles and executes on any ALU.

## What This Crate Is NOT

- Not a runtime engine (that's tambear-tam)
- Not a language (that's tambear-tbs)
- Not a GPU backend (that's tam-gpu / tambear-wgpu)
- Not compositions/methods (that's tambear-methods)

## The Dependency Chain

```
tambear-primitives  → declares recipes (what math to compute)
        ↓
tambear-methods     → compositions: select primitives by parameter
        ↓
tambear-tbs         → language: express compositions as scripts
        ↓
tambear-tam         → compiler: fuse recipes → single-pass execution
        ↓
tam-gpu / wgpu      → vendor doors: CPU / CUDA / Vulkan / Metal
```

Each layer depends ONLY downward. Never upward. Never sideways.

## Adding a New Primitive

### Step 1: Identify the Math

What does it compute? Write the formula. Is it ONE thing? If it does multiple things, it's a method, not a primitive. Break it down further.

A primitive is: one symbol, one operation, one result.

### Step 2: Write the Recipe

Decompose into accumulate+gather steps. Ask:
- What gets accumulated? (the Expr: Value, ValueSq, Ln, Reciprocal, etc.)
- How does it group? (All, ByKey, Prefix, Tiled, etc.)
- What operation combines? (Add, Max, Min, etc.)
- What does the gather compute from the accumulated values?

Example — mean_arithmetic:
```rust
Step::Accumulate { grouping: All, expr: Value, op: Add, output: "sum" }
Step::Accumulate { grouping: All, expr: One,   op: Add, output: "count" }
Step::Gather { expr: "sum / count", output: "mean" }
```

### Step 3: Check for Sharing

Look at your accumulate steps. Do ANY match an existing primitive's steps?
- Same (Grouping, Expr, Op) = shared. TAM deduplicates automatically.
- If your primitive accumulates (All, Value, Add) → that's "sum" → shared with mean_arithmetic, variance, and everything else that sums.

### Step 4: Create the Folder

```
src/primitives/<name>/
├── mod.rs           — implementation + recipe constant
├── params.toml      — metadata for IDE/search
├── README.md        — human docs
└── tests/
    └── mod.rs       — tests
```

### Step 5: Register

Add one line to `src/primitives/mod.rs`:
```rust
pub mod <name>;
```
And one re-export line.

Add one entry to `src/catalog/mod.rs` in the CATALOG array.

### Step 6: Verify

```bash
cargo test -p tambear-primitives
cargo test -p tambear-tam
```

Both must pass. The recipe must produce correct results through TAM execution.

## Rules

### Every primitive has a Recipe

No exceptions. The recipe is how TAM knows what to compute. A primitive without a recipe can't participate in fusion, can't share intermediates, can't compile to GPU. The recipe IS the primitive's contract with the execution engine.

### The Recipe must decompose into accumulate+gather

If your math can't be expressed as accumulate+gather steps, one of three things is true:
1. You need a new ExprKind (common — add it to recipe.rs)
2. You need a new GroupingKind (rare — means a new parallelism pattern)
3. It's genuinely Kingdom B/C/D — declare it honestly and document why

### Recipes share automatically

Two recipes with the same (GroupingKind, ExprKind, OpKind) triple share that accumulate. You don't wire sharing manually. You don't call TamSession::register. The recipe structure IS the sharing specification. TAM reads it at compile time.

### One pass is the goal

All accumulates with the same (GroupingKind, OpKind) fuse into one data pass. Different ExprKind values become multiple outputs of the same loop. The number of data passes = the number of unique (GroupingKind, OpKind) pairs, NOT the number of accumulate steps.

### Gather expressions are cheap

The gather phase operates on accumulated scalars (or per-group arrays), not on the raw data. Gather is O(1) or O(k) where k = number of groups. The cost is always in the accumulate phase. Design accordingly.

## The Recipe Types

### ExprKind — what to compute per element

| ExprKind | Formula | Used by |
|----------|---------|---------|
| Value | v | sum, max, min, cumsum |
| ValueSq | v² | sum of squares, variance, RMS |
| One | 1.0 | count |
| Ln | ln(v) | geometric mean, entropy |
| Reciprocal | 1/v | harmonic mean |
| Pow | v^p | power mean, moments |
| CrossRef | v × ref | covariance, weighted sums |
| AbsDev | |v - ref| | MAD, MAE |
| SqDev | (v - ref)² | variance (centered), MSE |

Adding a new ExprKind is cheap — one variant + one match arm in TAM's executor.

### GroupingKind — how to partition

| GroupingKind | Pattern | Data passes |
|-------------|---------|-------------|
| All | N → 1 | Fuses with all other All+same-Op |
| ByKey | N → K | Fuses with all other ByKey+same-Op |
| Prefix | N → N (scan) | Each is its own pass (serial dependency) |
| Segmented | N → N (scan with resets) | Like Prefix but with boundaries |
| Windowed | N → N (rolling) | Via prefix subtraction trick |
| Tiled | M×K → M×N | Matrix operations |
| Graph | Adjacency-based | Graph operations |

### OpKind — how to combine

| OpKind | Operation | Identity |
|--------|-----------|----------|
| Add | a + b | 0.0 |
| Max | max(a, b) | -∞ |
| Min | min(a, b) | +∞ |
| DotProduct | Σ aᵢbᵢ | 0.0 |
| Distance | Σ (aᵢ-bᵢ)² | 0.0 |
| Semiring(name) | custom algebra | custom |
