# .tbs Science Lint Specification

Mathematical correctness rules for the .tbs chain language. These lints run at parse-time (static) or at the start of execution (dynamic). They produce warnings, not errors — the user can suppress them.

---

## Static Lints (parse-time, no data needed)

### L001: Missing normalization before distance-based operations

**Trigger**: Chain contains `dbscan`, `kmeans`, `knn`, or any distance-based op without a preceding `normalize()`.

**Why**: Distance-based algorithms are scale-sensitive. Features with large range dominate Euclidean distance. Without normalization, clustering results are artifacts of units, not structure.

**Message**: `"⚠ L001: {op} uses Euclidean distance — consider normalize() first to avoid scale artifacts"`

**Exception**: If user passes `metric="cosine"` or `metric="correlation"`, suppress (these are scale-invariant).

### L002: Supervised step without clustering

**Trigger**: Chain goes directly to `train.linear` or `train.logistic` without any exploratory step.

**Why**: Mild — linear regression doesn't require clustering. But the .tbs philosophy is exploration-first. This is a suggestion, not a warning.

**Message**: `"ℹ L002: consider exploratory steps (describe, discover_clusters) before supervised training"`

**Severity**: Info, not warning.

### L003: Chained predictions without validation

**Trigger**: `train.linear().predict()` or `train.logistic().predict()` on the same data used for training.

**Why**: Evaluating on training data gives optimistic performance estimates.

**Message**: `"⚠ L003: predicting on training data — results are optimistically biased"`

### L004: Distribution assumption mismatch

**Trigger**: Chain contains a distribution shape step (`skewness`, `describe`, `moments`) followed by a normality-assuming test (`t_test`, `anova`, `f_test`, `pearson_r`, `one_sample_t`, `paired_t`, `welch_t`).

**Why**: The chain structure implies the user is aware of distribution shape but proceeds with normality-assuming tests. If |skewness| > 1.0 (Bulmer 1979: "highly skewed"), parametric tests produce unreliable p-values. The chain *pattern* is sufficient to flag — actual skewness is checked dynamically by L109.

**Message**: `"ℹ L004: {op} assumes normality — chain includes distribution shape analysis. Check |skewness| < 1 before trusting results."`

**Severity**: Info.

**Companion**: L109 (dynamic) fires when actual |skewness| > 1.0.

---

## Dynamic Lints (at execution, require data inspection)

### L101: Naive variance warning

**Trigger**: `var()` or `std()` called on data where `max / min > 1e6` or `mean² / var > 1e10` (condition number of the centering operation).

**Why**: The two-pass naive formula `Σ(x - x̄)² / n` loses precision when values are large relative to their spread. Welford's one-pass algorithm is numerically stable. Tambear's `moments_ungrouped` uses Welford internally — but if the user manually computes variance, the naive formula is dangerous.

**Message**: `"⚠ L101: data range ({max:.2e}/{min:.2e}) suggests naive variance may lose precision — use moments() instead of manual Σ(x-x̄)²"`

**Implementation**: Check after data is loaded, before any variance computation.

### L102: Heywood case in factor analysis

**Trigger**: `principal_axis_factoring` returns `heywood = true`.

**Why**: Communality > 1 means the extracted factors account for MORE than the total variance of a variable — mathematically impossible for standardized data. Indicates rank deficiency or too many factors.

**Message**: `"⚠ L102: Heywood case detected — communality > 1 for {n} variables. Check for multicollinearity or reduce n_factors."`

**Implementation**: Check `FaResult.heywood` after PAF runs.

### L103: Bipolar factor in reliability

**Trigger**: `mcdonalds_omega` returns `bipolar = true`.

**Why**: ω assumes unidirectional factor loadings. Mixed-sign loadings cancel in the numerator, giving artificially low ω despite strong factor structure. Tambear auto-corrects with |loadings|, but the user should know.

**Message**: `"ℹ L103: bipolar factor detected — loadings have mixed signs. ω computed with |loadings| (auto-corrected). Consider reverse-scoring items for interpretability."`

### L104: Near-IGARCH in GARCH estimation

**Trigger**: `garch11_fit` returns `alpha + beta > 0.99`.

**Why**: At α+β = 1 (IGARCH), the unconditional variance is infinite and the MLE landscape degenerates. Estimates near this boundary are unreliable.

**Message**: `"⚠ L104: α + β = {sum:.4} ≈ 1 (near IGARCH boundary). Estimates may be unreliable. Consider fitting IGARCH directly."`

### L105: Too few clusters for clustered SE

**Trigger**: `panel_fe` with `n_units < 30`.

**Why**: Clustered standard errors have a degrees-of-freedom correction that performs poorly with few clusters. With < 30 clusters, the correction inflates SEs (conservative but inefficient). With 1 cluster, the correction produces Inf (which tambear already guards against).

**Message**: `"⚠ L105: only {n} clusters for clustered SE. Results may be conservative (recommend ≥ 30 clusters for valid inference)."`

### L106: Constant or near-constant column

**Trigger**: Any input column with `std / mean < 1e-10` (or `std < 1e-15` if mean ≈ 0).

**Why**: A near-constant column contributes nothing to distance-based methods and can cause numerical issues in regression (singular X'X). Should be dropped or flagged.

**Message**: `"⚠ L106: column {j} is near-constant (σ = {std:.2e}). Consider removing before analysis."`

### L107: Small sample for bootstrap

**Trigger**: `bootstrap_percentile` or `bootstrap_bca` with `n < 20`.

**Why**: Bootstrap confidence intervals require enough samples to capture the distribution shape. Below ~20, the percentile method is unreliable.

**Message**: `"⚠ L107: bootstrap with n={n} — small samples give unreliable bootstrap CIs (recommend n ≥ 30)."`

### L108: R̂ not computed or > 1.1

**Trigger**: MCMC chain used for inference without checking R̂, or R̂ > 1.1.

**Why**: Gelman-Rubin diagnostic < 1.1 is the standard convergence criterion. Inference from non-converged chains is unreliable.

**Message**: `"⚠ L108: R̂ = {rhat:.3} > 1.1 — chains have not converged. Results are unreliable."`

### L109: High skewness before normality-assuming test

**Trigger**: `describe()` produces |skewness| > 1.0 in any column AND the chain includes a normality-assuming test (`t_test`, `anova`, `f_test`, `pearson_r`).

**Why**: Dynamic companion to L004. Threshold |skew| > 1.0 from Bulmer (1979): "highly skewed" distributions invalidate the CLT-based assumptions of parametric tests at typical sample sizes. With n > 500, the CLT provides some protection, but not enough for |skew| > 2.

**Message**: `"⚠ L109: column {j} has |skewness| = {skew:.2} > 1.0 — normality-assuming tests may be unreliable. Consider nonparametric alternatives (mann_whitney, kruskal_wallis)."`

**Implementation**: Check `DescribeResult` columns after `describe()` runs, before any normality-assuming test executes.

---

## Kingdom-Aware Lints (structural)

### L201: Wrong accumulation type

**Trigger**: Using a Kingdom A (single-pass accumulate) operation where Kingdom C (iterative convergence) is needed, or vice versa.

**Why**: Each algorithm family has a natural kingdom. Forcing the wrong kingdom either loses information (A where C needed) or wastes computation (C where A suffices).

**Examples**:
- Using manual mean+std for normalization (Kingdom A ✓) — fine
- Using iterative EM for simple mean (Kingdom C for a Kingdom A problem) — wasteful
- Using one-pass PCA approximation for a problem requiring convergence (Kingdom A for Kingdom C) — incorrect

**Message**: `"ℹ L201: {op} is Kingdom {actual}; the {next_op} step expects Kingdom {expected} input."`

### L202: Dimensional mismatch in chained operations

**Trigger**: Output dimensionality of one step doesn't match input requirement of the next.

**Why**: E.g., PCA reducing to 2 dimensions followed by an operation requiring the original feature space.

**Message**: `"⚠ L202: {prev_op} outputs {d_out}D data but {next_op} expects {d_in}D input."`

---

## Kingdom A Subproblem Table (for L201-L202)

The L201/L202 linter needs to know, for each Kingdom C algorithm, what Kingdom A subproblem it depends on. Two adjacent C steps share a Kingdom A subproblem IF they depend on the same accumulate expression (same grouping, same expression, same operator). When detected, the linter can suggest: "extract shared intermediate" or "fuse A passes."

### Kingdom C → Kingdom A Dependencies

| Algorithm | Kingdom | A Subproblem | Accumulate Signature |
|---|---|---|---|
| t-SNE | C | Pairwise distance matrix | `accumulate(AllPairs, (x_i - x_j)², Sum)` |
| NMF | C | Weighted covariance per iteration | `accumulate(ByColumn, w_i × x_i, Sum)` |
| GMM (EM) | C | Weighted means + covariances (M-step) | `accumulate(ByCluster, r_ik × x_i, Sum)` |
| LME (mixed effects) | C | Henderson equation matrices | `accumulate(Global, X'X + Z'Z, Sum)` |
| CFA (confirmatory FA) | C | Covariance matrix + gradient | `accumulate(AllPairs, x_i × x_j, Sum)` |
| IRT (2PL/3PL) | C | Item-person cross-product Hessian | `accumulate(ByItem, p_ij × (1-p_ij), Sum)` |
| Cox PH | C | Partial likelihood risk set sums | `accumulate(RiskSet, exp(β'x_i), Sum)` |
| GARCH(1,1) | B+C | Conditional variance recursion + MLE | `accumulate(Sequential, σ²_t, Recursive)` |
| Panel RE | C | LME internals | Same as LME |
| Variational inference | C | ELBO gradient | `accumulate(Global, ∇log q(z), Sum)` |

### Shared Subproblem Detection Rules

**Rule 1: Distance matrix sharing.** If two adjacent steps both need `accumulate(AllPairs, dist, Sum)` — e.g., `t-SNE` followed by `dbscan`, or `kmeans` followed by `knn` — the distance matrix is computed once.

**Rule 2: Covariance matrix sharing.** If two steps both need `accumulate(AllPairs, x_i × x_j, Sum)` — e.g., `pca` followed by `factor_analysis`, or `cfa` followed by `omega` — the covariance/correlation matrix is shared.

**Rule 3: No sharing across kingdom B boundaries.** If a Kingdom B step (sequential) intervenes between two Kingdom A/C steps, the shared subproblem breaks — the B step may have transformed the data. Emit L201 if the user chains `pca().exponential_smoothing().factor_analysis()` — the FA operates on smoothed data, not on the original covariance.

**Rule 4: Redundant C detection.** `kmeans(k=3).kmeans(k=5)` — both perform full distance + assignment passes. Warn: "adjacent clustering steps each compute distances independently; consider `discover_with(k=3, k=5)` for shared distance matrix."

### Kingdom Annotation Table

The linter consults this at parse time. Each function maps to `(Kingdom, SharedSubproblem)`:

```rust
fn kingdom_of(op: &str) -> (Kingdom, Option<SharedSubproblem>) {
    match op {
        // Kingdom A — single-pass accumulate
        "normalize" | "describe" | "moments" | "var" | "std"
            => (A, None),
        "pca" | "efa" | "varimax" | "manova" | "lda" | "cca"
            => (A, Some(Covariance)),
        "train.linear" | "panel_fe" | "two_sls" | "hausman"
            => (A, Some(GramMatrix)),
        "adf_test" | "ar" | "yule_walker"
            => (A, Some(Autocorrelation)),

        // Kingdom B — sequential
        "kaplan_meier" | "log_rank" | "exp_smoothing"
            => (B, None),
        "panel_fd"
            => (B, Some(LagScan)),

        // Kingdom C — iterative convergence
        "kmeans" | "dbscan" | "discover_clusters" | "knn"
            => (C, Some(DistanceMatrix)),
        "tsne" | "umap"
            => (C, Some(DistanceMatrix)),
        "gmm" | "mixture"
            => (C, Some(WeightedCovariance)),
        "lme" | "panel_re"
            => (C, Some(Henderson)),
        "cfa"
            => (C, Some(Covariance)),
        "irt"
            => (C, Some(CrossProduct)),
        "cox_ph"
            => (C, Some(RiskSet)),
        "garch"
            => (BC, Some(ConditionalVariance)),
        "train.logistic"
            => (C, Some(GramMatrix)),

        _ => (A, None), // default: assume single-pass
    }
}
```

---

## Implementation Notes

- Lints should be collected into a `Vec<TbsLint>` and returned alongside `TbsResult`.
- Each lint has a code (L001..L202), severity (info/warning/error), and human-readable message.
- The IDE should display lints inline with the .tbs source.
- Lints are NEVER errors — they are V columns (confidence metadata) on the computation itself.

```rust
pub struct TbsLint {
    pub code: &'static str,
    pub severity: LintSeverity,
    pub message: String,
    pub step_index: Option<usize>,  // which step triggered it
}

pub enum LintSeverity {
    Info,
    Warning,
}
```
