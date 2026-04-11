# Migration Map: crates/tambear → tambear/ workspace

## tambear/crates/tbs/ — THE language
From old crate:
- tbs_parser.rs → tbs/src/parser.rs (TbsChain, TbsStep, TbsArg, TbsName, TbsValue)
- using.rs → tbs/src/using.rs (UsingBag, UsingValue)
- tbs_advice.rs → tbs/src/advice.rs (TbsStepAdvice)
- tbs_lint.rs → tbs/src/lint.rs (science warnings)

From tambear-primitives (tonight's work):
- tbs/mod.rs → tbs/src/expr.rs (Expr AST — the universal expression type)

## tambear/crates/primitives/ — transforms × accumulates × gathers + recipes
From tambear-primitives (tonight's work):
- transforms/mod.rs → primitives/src/transforms.rs
- accumulates/mod.rs → primitives/src/accumulates.rs
- gathers/mod.rs → primitives/src/gathers.rs
- recipes/mod.rs → primitives/src/recipes.rs

From old crate (ALL the math implementations — these become recipes):
- descriptive.rs → recipes source for: mean, variance, skewness, kurtosis, quantile, etc.
- nonparametric.rs → recipes: Kendall, Spearman, Mann-Whitney, KS test, etc.
- hypothesis.rs → recipes: t-test, ANOVA, chi-square, etc.
- time_series.rs → recipes: ACF, PACF, AR, ARMA, ADF, KPSS, etc.
- linear_algebra.rs → recipes: Cholesky, QR, SVD, eigendecomposition, OLS, etc.
- information_theory.rs → recipes: entropy, KL divergence, MI, etc.
- special_functions.rs → recipes: erf, gamma, beta, Bessel, etc.
- volatility.rs → recipes: GARCH, EGARCH, realized vol, etc.
- complexity.rs → recipes: sample entropy, DFA, Lyapunov, etc.
- graph.rs → recipes: BFS, DFS, Dijkstra, MST, community detection, etc.
- clustering.rs → recipes: DBSCAN, K-means, hierarchical, spectral, etc.
- multivariate.rs → recipes: PCA, LDA, CCA, MANOVA, etc.
- bayesian.rs → recipes
- causal.rs → recipes
- hmm.rs → recipes
- kalman.rs → recipes
- state_space.rs → recipes
- survival.rs → recipes
- panel.rs → recipes
- mixture.rs → recipes
- irt.rs → recipes
- robust.rs → recipes
- neural.rs → recipes (activation functions, backprop, layers)
- optimization.rs → recipes
- signal_processing.rs → recipes
- interpolation.rs → recipes
- numerical.rs → recipes (root finding, quadrature, ODE solvers)
- number_theory.rs → recipes
- physics.rs → recipes
- stochastic.rs → recipes
- spatial.rs → recipes
- spectral.rs → recipes
- factor_analysis.rs → recipes
- dim_reduction.rs → recipes
- tda.rs → recipes
- distributional_distances.rs → recipes

## tambear/crates/tam/ — the compiler
From old crate:
- accumulate.rs → tam/src/accumulate.rs (AccumulateEngine, Grouping, Expr, Op)
- scatter_engine.rs → tam/src/scatter.rs (ScatterEngine)
- scatter_jit.rs → tam/src/scatter_jit.rs (ScatterJit — JIT kernel compilation)
- filter_jit.rs → tam/src/filter_jit.rs (FilterJit)
- gather_op.rs → tam/src/gather.rs (GatherOp)
- reduce_op.rs → tam/src/reduce.rs (ReduceOp)
- codegen.rs → tam/src/codegen.rs
- tbs_jit.rs → tam/src/jit.rs
- spec_compiler.rs → tam/src/spec_compiler.rs
- tam.rs → tam/src/tam.rs (the orchestrator)

From tambear-tam (tonight's work):
- lib.rs → tam/src/plan.rs (Plan, compile, fuse, execute)

## tambear/crates/session/ — sharing + state
From old crate:
- intermediates.rs → session/src/intermediates.rs (TamSession, IntermediateTag, DataId)
- superposition.rs → session/src/superposition.rs (Superposition, SuperpositionView)
- pipeline.rs → session/src/pipeline.rs
- tbs_executor.rs → session/src/executor.rs (dispatch chain)
- tbs_autodetect.rs → session/src/autodetect.rs (Layer 1 diagnostics)
- scoring.rs → session/src/scoring.rs
- hazard.rs → session/src/hazard.rs
- predictive.rs → session/src/predictive.rs

## tambear/crates/runtime/ — vendor doors
From old crate:
- compute_engine.rs → runtime/src/compute.rs (ComputeEngine)

From tam-gpu crate:
- lib.rs → runtime/src/backend.rs (TamGpu trait, CpuBackend, CudaBackend)
- Buffer, Kernel, ShaderLang types

From tambear-wgpu crate:
- lib.rs → runtime/src/wgpu.rs (WgpuBackend)

## tambear/crates/frame/ — data containers
From old crate:
- frame.rs → frame/src/frame.rs (Frame, Column, DType)
- group_index.rs → frame/src/group_index.rs (GroupIndex)
- dictionary.rs → frame/src/dictionary.rs
- nan_guard.rs → frame/src/nan_guard.rs (NanPolicy, guards)
- rng.rs → frame/src/rng.rs (Xoshiro256)

## tambear/crates/io/ — file formats
From old crate:
- tb_io.rs → io/src/tb_io.rs (TbFile)
- format.rs → io/src/format.rs

From mkt-rs crate:
- MKTF/MKTC format handling

## NOT migrated (experiments, demos, collatz, benchmarks)
- experiment0.rs, experiment1.rs, experiment1b.rs, experiment2.rs
- collatz_parallel.rs, collatz_search.rs, collatz_structural.rs
- extremal_orbit.rs, fold_irreversibility.rs, multi_adic.rs
- layer_bijection.rs, equipartition.rs, spectral_gap.rs
- beal_search.rs, proof.rs, naturalist_observation.rs
- scatter_bench.rs, io_bench.rs, compile_budget.rs
- tb_demo.rs, train_demo.rs, real_data.rs, main.rs
- copa.rs (goes to session or primitives depending)
- stats.rs (probably merged into other modules)
- hash_scatter.rs (implementation detail of scatter_engine)
- manifold.rs (goes to primitives/recipes)
- knn.rs (goes to primitives/recipes)
- kmeans.rs (goes to primitives/recipes)
- parallel.rs, sketches.rs (utilities)
- bigint.rs, bigfloat.rs (goes to primitives or frame)
- series_accel.rs (goes to primitives/recipes)

## Key architectural decisions

1. tbs/ has ZERO dependencies on anything else. It's the language. Pure.
2. primitives/ depends ONLY on tbs/ (uses Expr for transforms and gathers).
3. tam/ depends on primitives/ + tbs/ (compiles recipes to plans).
4. session/ depends on tbs/ (UsingBag, IntermediateTag types).
5. runtime/ depends on tam/ + session/ (executes plans with sharing).
6. frame/ has ZERO dependencies. Data containers are independent.
7. io/ depends on frame/ only.

No circular dependencies. No upward dependencies. Clean DAG.
