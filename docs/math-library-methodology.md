# Tambear Math Library: Methodology

**Goal**: Every algorithm in mathematics, statistics, and signal processing — individually verified, audited, composable, on any GPU. Replaces SPSS, SAS, Mplus, MATLAB, cuBLAS, cuFFT, cuML, cuDNN, and every other numerical library. Not wrappers. Native implementations from tambear primitives.

---

## The Standard: Per-Algorithm

Every algorithm goes through this pipeline:

### 1. DECOMPOSE
- What IS the math? Not how people implement it. What does it compute?
- Read the original paper. Read the textbook definition.
- Identify: inputs, outputs, assumptions, edge cases, numerical stability requirements.
- What existing tambear primitives does it use? (distance, scatter, scan, tiled)
- What's missing? Does it need a new primitive or operator?

### 2. IMPLEMENT (tambear-native)
- Build from accumulate + gather + fused_expr. No vendor libraries.
- Both versions:
  - **Full**: computes from raw data, exact, no shortcuts
  - **Sufficient**: extracts from already-accumulated MSR fields (free when Tam has stats)
- Must work on ALL backends (CUDA, Vulkan, Metal, CPU)
- Must handle: NaN, Inf, empty input, single element, all-same values, all-zeros

### 3. VALIDATE (gold standard parity)
- Compare output against established implementations:
  - R (stats, psych, lavaan, forecast)
  - Python (scipy, sklearn, statsmodels, numpy)
  - NVIDIA (cuBLAS, cuFFT, cuML) where applicable
- **Bit-perfect** where possible (f64). Document any f32/f64 differences.
- Test on:
  - **Synthetic data**: known ground truth (e.g., generate from known distribution, verify recovered parameters)
  - **Real data**: AAPL ticks, findmcp features, public datasets
  - **Edge cases**: n=1, n=2, all identical, all NaN, extreme values, denormals
  - **Adversarial cases**: data designed to break the algorithm

### 4. BENCHMARK (fair comparison)
- Same hardware, same data, same precision.
- Measure:
  - Tambear standalone (just this algorithm)
  - Tambear composable (with Tam's sharing — distance/stats from session)
  - R equivalent
  - Python equivalent
  - cu* equivalent (where applicable)
- **Fair means**: if our version uses shared distance from a prior step, benchmark BOTH:
  - Our algorithm alone (including distance computation)
  - Our algorithm with shared distance (what the user actually gets)
  - Compare both against standalone R/Python/cu*

### 5. DOCUMENT (lab notebook per algorithm)
- Hypothesis → Design → Results → Surprise → Discussion
- .tbs script saved alongside notebook
- Synthetic test results with ground truth
- Real data results with R/Python/cu* comparison
- Benchmark numbers with hardware specs
- Edge case audit with pass/fail per case
- Mathematical assumptions listed and verified

### 6. COMPOSE (integrate into tambear)
- Wire into .tbs executor vocabulary
- Wire into TamSession (what does it produce? what does it consume?)
- Wire into .discover() (what flavors exist? how to evaluate?)
- Test in pipeline context (does sharing work? does fusion work?)

---

## Build Order

### Phase 1: Individual Flavors
Each algorithm implemented, validated, benchmarked, documented individually.
Composed manually in .tbs scripts. No .discover() yet. No superposition.

### Phase 2: Tambear-Custom-Best
For algorithms where our first-principles approach produces something NEW
(like hash scatter for groupby, density clustering without k),
document the novel approach alongside the standard implementation.

### Phase 3: Superposition + .discover()
Add .discover() for families with multiple flavors.
Superposition of flavors, auto-evaluate, auto-collapse, auto-document.

### Phase 4: Full Pipelines
Pre-built pipelines for common workflows:
- tb.eda(data) — full exploratory data analysis
- tb.factor_analysis(data) — best-practice FA pipeline
- tb.time_series(data) — full TS analysis
- tb.train.X(data) — model training pipelines

---

## Team Composition

### Pathmaker (implementer)
- Reads the math, builds the accumulate decomposition
- Writes the Rust implementation from primitives
- Runs initial tests

### Navigator (architect)
- Designs the sharing surface (what goes in TamSession)
- Identifies MSR for each algorithm family
- Maps cross-algorithm sharing opportunities

### Scientist Observer
- Validates correctness against R/Python/cu* gold standards
- Runs synthetic ground-truth tests
- Writes validation sections of lab notebooks
- Ensures numerical stability (Kahan summation, log-sum-exp, etc.)

### Adversarial Mathematician
- Tries to break every algorithm
- Designs adversarial test cases (near-singular matrices, ill-conditioned data)
- Challenges every assumption ("does this REALLY need sorted input?")
- Finds silent failures (algorithm returns plausible-but-wrong answer)

### Naturalist ("what-if" explorer)
- Questions whether the standard approach is even the right math
- Proposes tambear-native alternatives
- Connects insights across algorithm families
- "What if skewness doesn't need the third moment?"
- "What if PCA is just a tiled accumulate?"

---

## The Composability Contract

Every algorithm must declare:

```toml
[algorithm]
name = "skewness_fisher"
family = "descriptive.shape"
version = "1.0.0"

[inputs]
required = ["numeric_array"]         # what it needs
optional = ["weights"]               # what it can use

[outputs]
primary = "skewness_g1"              # the result
secondary = ["se", "z_score"]        # bonus outputs

[sufficient_stats]
consumes = ["n", "sum_x", "sum_x2", "sum_x3"]   # from MSR
produces = ["n", "sum_x", "sum_x2", "sum_x3"]   # deposits to MSR

[sharing]
provides_to_session = ["MomentStats(order=3)"]
consumes_from_session = ["MomentStats(order>=3)"]
auto_insert_if_missing = "accumulate(All, [x, x^2, x^3, 1], Add)"

[assumptions]
requires_sorted = false
requires_positive = false
requires_no_nan = false              # handles NaN internally
minimum_n = 3                        # needs at least 3 values
```

This is the spec.toml for ALGORITHMS, not leaves. Same pattern as fintek's leaf spec.
The compiler reads these to plan sharing, insert missing prerequisites, and verify assumptions.
