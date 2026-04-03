# Tam

**Tam is the self-aware computation engine. Everything below Tam is lifted. Everything Tam touches becomes the most efficient version of itself. Tam knows.**

---

## The Truths

### 1. Global Device

Tam runs on anything with compute. NVIDIA, AMD, Intel, Apple, Qualcomm. Desktop, laptop, server, phone, embedded. CUDA, Vulkan, Metal, DX12, CPU fallback. If it has an OS and you can get data onto it, Tam runs.

Tam doesn't care what GPU you have. Tam cares that you have cores. Any cores. The kernel driver is the only dependency. Everything above the driver is Tam.

**Born configured.** No toolkit install. No driver version matrix. No conda environment. No Docker container. `pip install tambear` and you have the full stack. Tam detects your hardware at first import and compiles for it. The user never sees a configuration screen.

**Scales to what you have.** Pregnancy test screen with a microcontroller? Small dataset, CPU fallback, still correct. RTX PRO 6000 Blackwell with 96GB VRAM? Full pipeline, fused kernels, provenance store. Heterogeneous cluster with NVIDIA + AMD + Apple? Same code, every GPU contributes. Resources join live, resources leave live, Tam adapts.

### 2. Global Sharing

Sharing is the architecture. Not an optimization. Not a cache layer. The ARCHITECTURE.

Tam's primary job is to find what can be shared. Computation is the fallback when sharing fails. Eight dimensions of sharing, seven of which work on first run with cold caches:

| Sharing | What it eliminates | Needs warm cache? |
|---|---|---|
| **Structural** | Redundant computation across consumers | No |
| **Fusion** | Redundant memory traffic between operations | No |
| **Layout** | Data format transformation | No |
| **Buffer** | Memory allocation overhead | No |
| **Dispatch** | Kernel launch overhead | No |
| **Preprocessing** | Redundant data preparation | No |
| **Cross-algorithm** | Redundant computation across algorithm boundaries | No |
| **Provenance** | Redundant computation across runs | Yes |

Even without provenance, even on first run, Tam produces the most efficient possible execution through the seven structural sharing dimensions. Provenance is the cherry on top for repeated runs.

**The intermediate marketplace.** Every step in a pipeline both produces and consumes intermediates. The compiler matches producers to consumers. Distance computed by KMeans is FREE for KNN. Statistics computed by normalization are FREE for Z-scores. Masks produced by filters are FREE for masked operations. No intermediate is computed twice. No intermediate is stored unnecessarily. Every value flows from its single producer to all its consumers through registers, not memory.

**Tam Knows.** Each composable step does ONE thing — the thing that makes it unique. KNN computes neighbors, not distance. Normalization computes scaled values, not statistics. Clustering computes labels, not density. Everything else — distance, statistics, density, masks, embeddings — Tam already has. Not cached. Not preprocessed. Just there. Because the compiler planned the entire pipeline before launching a single kernel.

When something ISN'T already there, Tam inserts the cheapest valid version automatically. The user never writes a preprocessing step. There ARE no preprocessing steps. There's just the pipeline, and Tam knows what each step needs.

### 3. Global Abstraction

The most complicated computations humanity has — neural network training, Kalman filtering, spectral analysis, geometric embedding, gradient descent — composable by a first-year high school student in a rural library computer lab. 10-20 lines of Python. No expertise beyond knowing which pipeline to use.

```python
import tambear as tb
model = tb.train.gpt(my_data)
```

One line. A state-of-the-art transformer. On any GPU. The student doesn't know about attention heads, learning rates, gradient computation, kernel launches, or GPU architecture. They don't need to. Tam knows.

Tam reads the data, discovers its structure, selects the architecture for YOUR data size and YOUR hardware, compiles the entire training pipeline into fused kernels for YOUR specific GPU, runs training with real-time stats streaming to your terminal, and saves the model with full documentation of what it did and why.

```python
tb.train.gpt(data)      # transformer training, best of class
tb.train.fmm(data)      # multi-surface geometric model
tb.train.forest(data)   # random forest
tb.train.cluster(data)  # discover structure, no k needed
```

Each one: data in, results out, real-time stats, documentation on your system. No config files. No YAML. No hyperparameter tuning. Simpler than writing a Markdown file.

**For those who want to compose:**

```python
data = tb.read("my_data.csv")
clusters = data.select("feature1", "feature2").discover_clusters()
model = data.train(tb.forest(trees=100), target="label")
predictions = model.predict(new_data)
```

Four lines. Clustering + training + prediction. Every operation chains with every other operation. But composing is OPTIONAL. The pre-built pipelines cover every common task. You compose only when you're inventing something new.

**The abstraction is not hiding complexity.** The abstraction IS the computation. `tb.train.gpt(data)` is not a wrapper around PyTorch. It's a pipeline of `accumulate + gather` operations compiled into fused kernels. The abstraction and the implementation are the same thing — there is no layer between what the user writes and what the GPU executes.

### 4. Global Efficiency

Single pass. Single kernel. Single buffer. One read from memory, all computation in registers, one write of final results.

Traditional frameworks: 10 operations = 10 kernel launches = 10 memory round trips. The GPU sits idle 90% of the time waiting for memory. The compute takes nanoseconds; the memory traffic takes microseconds.

Tam: 10 operations = 1 kernel = 1 memory round trip. The compiler sees the entire pipeline — forward pass, backward pass, optimizer step, cross-validation — as ONE computation graph. It fuses everything that touches the same data into one kernel. Values live in registers, flow between operations without hitting memory. No intermediate buffers. No unnecessary memory traffic.

**JIT-compiled for YOUR hardware.** Not pre-compiled for a generic GPU. Compiled at first run for the specific GPU you have, with the specific block sizes and shared memory your hardware supports. Cached permanently. Second run: zero compilation overhead.

**The theoretical minimum.** After the sharing optimizer eliminates every redundant computation, every unnecessary memory access, every duplicate intermediate — what's left is the minimum possible work. The fused kernel executes that minimum. You literally cannot do less computation or less I/O. This is what "most efficient by nature" means: not optimized to be fast, but constructed so that waste is structurally impossible.

### 5. Global Discovery

Every composable has a `.discover()` variant. Superposition → measurement → collapse.

```python
data.normalize.discover()     # tries every normalization in parallel
    .cluster.discover()       # tries every clustering method in parallel
    .project.discover()       # tries every manifold in parallel
```

At each `.discover()`, the pipeline enters superposition — all possible states exist simultaneously, sharing everything except their unique leaf. Tam evaluates each branch (quality, overfitting, fit), picks the winner, purges the losers from VRAM, and continues in a single definite state.

The cost is almost nothing: all branches share data loading, encoding, grouping. They differ only in the tiny expr that makes each option unique. Exploring 10 options costs barely more than running 1.

The output is KNOWLEDGE, not just a result. "Your data is mildly hyperbolic (curvature -0.3). Poincare overfits at full curvature. Learned manifold wins." The user discovers something about their data they didn't know before.

### 6. Global Provenance

Every run is automatically versioned, documented, and preserved. The user never manages this.

Before Tam runs: auto-commit the script. After Tam runs: auto-commit the results, models, documentation. Every run gets a lab notebook (what was tried, what worked, comparisons to prior runs), benchmarks (timing, memory, GPU utilization per step), and output files. Nothing overwrites — run_001, run_002, run_003 all exist, all comparable.

Tam auto-benchmarks. Tam auto-sciences (lab notebook of results). Tam auto-reports to terminal and to persistent documentation. The user thinks, runs, reads results, thinks again. The workflow overhead is zero.

### 7. Global Competence

Tam doesn't just run your pipeline. Tam understands it.

**Tam discovers structure.** `tb.discover_clusters()` doesn't need k. Tam reads the density structure of the data and finds the natural clusters. No user parameter required. The data knows its own structure; Tam reads it.

**Tam discovers sharing.** The compiler analyzes the full pipeline graph and finds every sharing opportunity — across operations, across algorithms, across the forward and backward pass. Sharing the user didn't know existed. Sharing that's invisible in imperative frameworks because each operation dispatches before the next is known.

**Tam discovers types.** When KNN needs distance and KMeans already computed distance, Tam sees the match. When normalization needs statistics and the groupby already computed statistics, Tam sees the match. The intermediate marketplace operates on semantic types (Distance, Statistics, Mask, Embedding), not tensor shapes.

**Tam adapts.** New GPU joins the cluster? Tam redistributes. GPU leaves? Tam rebalances. Dataset grows? Tam re-tiles. User changes the pipeline? Tam recompiles only what changed — provenance identifies what's still valid.

### 6. The Fock Boundary

In every other framework, self-reference is distributed. Every PyTorch tensor knows its own grad_fn. Every module tracks its own parameters. Every operation checks autograd state. The Fock boundary — the limit of what can be parallelized — is everywhere. Nothing can be fully optimized across operation boundaries.

In Tam, the Fock boundary is Tam himself. One self-aware entity. The compiler that sees the whole graph. The sharing optimizer that knows all intermediates. The provenance store that remembers all prior computation.

Everything below Tam — every primitive, every specialist, every operation — is fully liftable. No self-reference. No awareness of what else exists. They declare what they need. They declare what they produce. That's it.

**Tam Raises the Fock.** The Fock boundary can't be eliminated (self-reference is fundamental). But it can be raised as high as possible. Tam raises it to the compiler itself — the ONE entity that SHOULD be self-aware. Everything below: pure, liftable, shareable, fusible. One kernel. One pass.

### 7. The Inversion

Tambear is the dual of isolation-based architectures.

In systems where each node is independent (signal farms, leaf computations, embarrassingly parallel workloads), performance comes from doing things INDEPENDENTLY. The architecture prevents sharing because sharing would create dependencies that slow things down.

In tambear, performance comes from NOT doing things. From finding what's already computed, what's shared across algorithms, what can be reused. The architecture maximizes sharing because sharing eliminates redundant work.

**The compiler handles both.** Give it independent signals: it finds no sharing, parallelizes everything. Give it a training pipeline: it finds sharing everywhere, eliminates every redundancy. Give it a mixed workload: it discovers which parts are independent and which are interdependent, and optimizes each appropriately.

Same compiler. Same primitives. Same `accumulate + gather`. The DATA TOPOLOGY determines whether the optimizer discovers isolation or connection. The compiler doesn't assume either — it discovers the truth.

### 8. Two Operations

All of computation — every ML algorithm, every DataFrame operation, every signal processing pipeline — decomposes to two operations:

1. **`accumulate(data, grouping, expr, op)`** — THE computation primitive
2. **`gather(indices, source)`** — THE read primitive

Every algorithm is a choice from four menus:
- **Addressing** (how to read): Direct, Strided, MultiOffset, Broadcast, Masked, Tiled
- **Grouping** (how to partition): All, ByKey, Prefix, Windowed, Tiled, Segmented, Masked
- **Expression** (what to compute): any element-wise function
- **Operator** (how to combine): Add, Welford, Affine, Max, ArgMin, SoftmaxWeighted, Custom

The specialist library is the menu. Each specialist = one choice from each menu. The compiler compiles menu choices into fused kernels. The sharing optimizer finds when two choices share a menu item and eliminates the redundancy.

### 9. Tam Is Not a Library

Tam is not a faster DataFrame library. Tam is not a better ML framework. Tam is not a GPU abstraction layer.

Tam is a computation platform. The computation platform. The one that makes every other framework unnecessary — not by being better at what they do, but by being a fundamentally different thing.

Libraries execute operations. Tam compiles pipelines. Libraries optimize individual kernels. Tam optimizes entire computation graphs. Libraries run on one GPU vendor. Tam runs on anything with cores.

**Tam replaces:**
- The CUDA engineers ($500K/yr) — the pipeline compiles itself
- The ML infrastructure engineers ($500K/yr) — zero config, any hardware
- The DevOps for GPU clusters ($200K/yr) — heterogeneous auto-discovery
- Months of optimization work — specialists + sharing + fusion = already optimized
- Hardware vendor lock-in — any GPU, any mix
- Framework lock-in — one API, every language
- The entire "set up CUDA" workflow — born configured

**Tam provides:**
- 10-20 lines of composable pipelines for ANY computation
- Pre-optimized specialist library that grows with usage
- JIT compilation tuned for YOUR specific GPU
- Eight dimensions of sharing that eliminate redundant work
- Single-pass execution for any pipeline
- From pregnancy test to supercluster — same code, same API

---

## The One-Sentence Version

**Tam is the self-aware compiler that sees your entire computation, finds every sharing opportunity, fuses everything into one pass, runs on any device, and makes the most powerful computations on Earth accessible to anyone who can write 10 lines of code.**

---

*Tam knows. Tam shares. Tam compiles. Tam runs everywhere. Tam raises the Fock.*
