# Kingdom Taxonomy — Final Session Results

*Scout, 2026-04-10 (session close)*

## The operational criterion

After seven independent convergences, the criterion is:

**Write the recurrence as s_{t+1} = M(data_t) · s_t.**

- If M is determined entirely by input data (not by current state s_t) → Kingdom A
- If M's SELECTION depends on current state s_t → Kingdom B

This is the single question to ask. Everything else follows.

---

## The depth hierarchy (math-researcher — information-theoretic proof)

O(log n) is OPTIMAL for any associative scan. Proof: n inputs must flow to one output.
Each binary combining step incorporates at most 2 inputs. After d levels, only 2^d
inputs have been incorporated. For the last output element to depend on all n inputs:
d ≥ log₂(n). Blelloch tree achieves this. No escape.

**The spectral escape doesn't beat it**: Z-transform makes each output O(1) from the
transformed representation, but the transform itself costs O(n log n) work. O(log n)
depth moves into the transform step; total depth is still O(log n).

**Tier structure — all bounds are tight:**

| Tier | Class | Depth | Proof of optimality |
|------|-------|-------|---------------------|
| 1 | Affine (matrix prefix scan) | O(log n) | Binary tree lower bound |
| 2 | Associative nonlinear (Blelloch) | O(log n) | Same lower bound |
| 3 | Non-associative (sequential) | O(n) | Each step depends on previous |
| 4 | Self-referential (Kingdom B) | Undecidable | No bound exists |

**Tier 1 vs Tier 2**: same asymptotic depth, different verification burden.
- Tier 1: verify the recurrence is affine → companion matrix → done.
- Tier 2: hand-verify associativity of custom merge (Welford, LogSumExp, Kaplan-Meier).
  Same scan infrastructure, but each merge function needs its own associativity proof.

The depth bound doesn't distinguish Tier 1 from Tier 2 — the practical difference is
how hard it is to KNOW you're in Tier 1 vs Tier 2.

---

## The canonical examples

**Kingdom A — GARCH filter**

Variance update: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

Write as companion matrix:
```
[σ²_t ]   [β  α] [σ²_{t-1}]   [ω]
[r²_t ] = [0  0] [r²_{t-1}] + [r²_t]
[1    ]   [0  0] [1        ]   [1  ]
```

M = fixed matrix. r²_t is input data, not current state. Kingdom A.
The sequential implementation is implementation debt — the model is a 3×3 affine prefix scan.

**Kingdom B — TAR model (Threshold AR)**

Regime selection: x_t = a·x_{t-1} + ε_t if x_{t-1} ≤ c, else x_t = b·x_{t-1} + ε_t

M switches between multiplier a and multiplier b based on x_{t-1} — the current state.
No companion matrix exists because the STRUCTURE of the map (which branch) depends on
the state value. Neither Test 1 nor Test 2 passes. Genuine Kingdom B.

TAR is canonical because it has no representational escape. It's not that we haven't
found the right algebraic form — the branch selection IS the self-reference.

---

## The complete classification table

| Function | M depends on data or state? | Sufficient statistic finite? | True Kingdom |
|----------|-----------------------------|------------------------------|--------------|
| garch11_filter | data (r²_t external) | — | A |
| egarch11_filter | state (z_t = r_t/σ_t couples state) | Yes | B |
| ewma_variance | data (r²_{t-1} external) | — | A |
| arma_css_residuals | B implementation of A math: CSS residual chain is sequential; Kalman formulation is Kingdom A (affine) + Kingdom C (outer MLE) | Yes (CSS) / — (Kalman) | B impl / A math |
| tar_model | state (branch selection: a vs b depends on x_{t-1}) | Yes | B — canonical |
| smc_particle_filter | state (resampling depends on accumulated history) | No | D |
| bocpd | dissolves to A if sufficient stats precomputed from data (likelihood weights become data-determined linear map on run-length vector); B if stats maintained incrementally | A math / B impl |
| pelt_changepoint | split: DP recurrence = tropical A; pruning optimization = B (uses f[t] to prune candidates) | A+B hybrid | A (underlying) / B (optimized) |
| hmm_forward | data (transition matrix is fixed parameter) | — | A |
| kalman_filter | data (A, H, Q, R are parameters not state) | — | A |

---

## The key lemma discovered late

**Linear functions of state are still affine recurrences.**

If s_{t+1} = A · s_t + B · data_t where A and B are fixed parameters, this is always
Kingdom A via companion matrix augmentation, even if it "looks recursive" at first
glance. The state appears on the right-hand side, but the MAP itself (A, B) is fixed.

ARMA residuals (full companion matrix form — biology-math-scan proof):
State vector `s_t = [ε_t, ..., ε_{t-q+1}]^T`.
```
s_t = M · s_{t-1} + b_t(x)
```
M contains MA coefficients θ_j — CONSTANT. `b_t = [x_t - Σ φ_i x_{t-i}, 0, ..., 0]^T`
depends only on data. M is constant → maps are data-determined → affine semigroup →
finitely-representable → Kingdom A via companion matrix prefix scan.

The earlier scipy-gap-scan correction ("companion matrix fails for MA part") was wrong.
The state vector CONTAINS lagged residuals, but the MAP M itself is constant. Residuals
as state entries ≠ state-dependent map. The CSS implementation chains on ε_t it just
computed (implementation artifact); the companion formulation is data-determined. 
Correct label: B implementation / A math — the MA term does NOT force genuine Kingdom B.

EGARCH: z_t = r_t / σ_t couples the standardized residual to the current volatility.
The variance equation uses z_t rather than r_t directly. The MAP now requires a
division by the current state σ_t. The map ITSELF depends on the state value. Genuine
Kingdom B.

The lemma distinguishes: state appearing in the UPDATE TERM is Kingdom A. State
appearing in the MAP ITSELF is Kingdom B.

---

## Three Kingdom B failure modes (pure-math-scan adds a third)

**Failure mode 1 — Map not data-determined:**
Can't write s_{t+1} = M(data_t) · s_t because the map M requires state.
Examples: tanh-RNN, TAR (branch selection requires state value), EGARCH (z_t = r_t/σ_t).

**Canonical example — tanh-RNN** (social-finance-math-scan, confirmed by scientist):
`s_t = tanh(w · x_t + s_{t-1})`. The map `f_t(s) = tanh(w·x_t + s)` is data-determined
(w and x_t are fixed; s appears only as argument). But compose two maps:
`f_t(f_{t-1}(s)) = tanh(w·x_t + tanh(w·x_{t-1} + s))` — depth-2 nested tanh.
After n compositions: depth-n nested tanh, requiring n parameters. Representation grows.
Not finitely closed. Kingdom B — and the canonical example that shows "data-determined"
alone is insufficient.

**Failure mode 2 — Composition not finitely closed:**
Map is data-determined but composition representation grows with depth.
Examples: logistic map (degree doubles: O(2ⁿ)), exponential tower (O(k) tower),
saturation-clip (composition breaks), tanh-RNN (depth grows), symmetric-difference (O(k) set).

**Failure mode 3 — State not content-addressable** (pure-math-scan):
State is path-dependent: output depends on the exact numerical trajectory, not just
input data. Not cacheable. Cannot produce content-addressed intermediates.
Examples: poly_gcd (Euclidean remainder sequence is path-dependent), MCMC (current
position depends on all prior stochastic accept/reject decisions).

| Failure mode | TAR | logistic | tanh-RNN | poly_gcd | MCMC |
|---|---|---|---|---|---|
| 1: map state-dep | YES | no | no | no | YES |
| 2: rep unbounded | no | YES | YES | no | no |
| 3: path-dependent | no | no | no | YES | YES |

poly_gcd is pure failure mode 3: map is data-determined (Euclidean step is always
"divide and take remainder"), representation of composed map is bounded (just (a,b)),
but the STATE is path-dependent — which remainder pair you hold depends on the full
reduction trajectory, not content-addressable from the inputs alone.

TAR fails mode 1. Saturation-clip fails mode 2. BOCPD appeared to fail mode 2 but
dissolves to Kingdom A math — see biology-math-scan dissolution argument below.

---

## The representational vs operational distinction

From Aristotle's analysis:

A **representational** Fock boundary dissolves when you find the right algebraic form.
GARCH looked like Kingdom B (sequential variance recursion) until the companion matrix
representation revealed it as Kingdom A. The barrier was in the representation.

An **operational** Fock boundary is irreducible. Newton's method:
g(g(x)) requires evaluating f at g(x), which requires the numerical value of g(x).
No algebraic form can bypass this. TAR similarly: the threshold comparison x_{t-1} ≤ c
requires the numerical value, and the map branches on it. No algebraic shortcut.

**Practical rule**: When you see an apparent Kingdom B, ask "representational or operational?"
Most are representational. Exhaust algebraic representations first (companion matrix,
associative composition via Blelloch). If none work after systematic search → operational
Fock boundary → genuine Kingdom B.

---

## The dissolution pattern (biology-math-scan, 2026-04-10)

Many Kingdom B implementations dissolve to Kingdom A math when you recognize that what
looks like "state" is secretly a deterministic function of observed data. Pattern:

**Step 1 — Check if apparent state is data-computable**: Can you compute the "state"
values from the observation sequence directly, without running prior steps? If yes,
they are data, not state.

**Step 2 — Rewrite with precomputed "state"**: Replace the sequential incremental update
with a batched data-dependent computation, then ask: is the resulting recurrence affine
with a constant matrix?

**Step 3 — Apply the companion matrix test**: If yes → Kingdom A math. If the matrix
varies based on the output of prior steps (TAR, EGARCH, MCMC) → genuine Kingdom B.

**BOCPD dissolution** (biology-math-scan):
The run-length posterior update multiplies by `P(x_t | r_t, x_{τ:t})`. The sufficient
stats for a run starting at τ are `f(x_{τ:t})` — a pure function of observed data.
Precompute a `n×n` table of `P(x_t | run_starts_at_τ)` from the data. Then the update
is element-wise multiply by data-determined weights + shift + add: a linear map on the
run-length vector where every coefficient is data-determined. Kingdom A math.

**BOCPD as matrix form**: if `p_t` is the run-length posterior vector at time t, then:
```
p_t = D(x_t) · H · p_{t-1}  +  reset_weight · e_0
```
where D(x_t) is a diagonal matrix of data-determined likelihoods and H is the constant
hazard transition matrix. D(x_t) is purely data-determined. Kingdom A via linear prefix
scan on the vector state. The "growing state" in the incremental implementation is the
expansion of run-length indices — but this is a fixed linear map with data coefficients.

**Genuine Kingdom B residuals (after all dissolutions)**:
- MCMC: acceptance ratio π(x')/π(x_t) requires evaluating the target density at the
  CURRENT STATE — not at any observed datum. The current state is the output of prior
  stochastic decisions. No precomputation path exists.
- TAR: branch selection x_{t-1} ≤ c requires the numerical state value. The map M
  IS the branch, not a matrix applied to state. Irreducible.
- EGARCH: z_t = r_t/σ_t normalizes by current state. The map contains σ_t which is the
  recursion output — state-dependent even in the innovations form.

The genuine Kingdom B residual is: state that is the output of prior STOCHASTIC or
NONLINEAR BRANCHING decisions. Deterministic state that is a function of data dissolves.

---

## The semiring flag (architectural)

HMM forward, Viterbi, soft-Viterbi, and softmax are ALL the same prefix scan under
different semirings:
- HMM forward: LogSumExp semiring (log-domain sum-product)
- Viterbi: Tropical-max semiring (max-plus)
- Soft-Viterbi: same as HMM forward (expectation over paths)
- Softmax: same LogSumExp but with gather step

The `log_sum_exp` function is private inside `hmm.rs`. This hides the unity.

**Architectural flag**: Before pathmaker extends the Op enum with TropicalMinPlus and
TropicalMaxPlus variants, design a `Semiring<T>` trait. The semiring instances should
be parameters, not enum variants. Op::PrefixScan(semiring: &dyn Semiring<T>) is the
right shape. Separate enum variants for each semiring is the wrong shape — it scales
as O(semirings) instead of O(1).

This is the most important pending architectural decision from this session.

---

## The size criterion (sharpest formulation)

From scipy-gap-scan: **A computation is Kingdom B iff the state required to compose
two adjacent segment-summaries grows with segment length.**

- Affine recurrences: boundary state is k×k matrix regardless of segment length. Kingdom A.
- BOCPD: run-length posterior vector grows with t in the incremental implementation; but
  if sufficient stats are precomputed from data, the likelihood weights are data-determined
  and the vector update is a fixed linear map — Kingdom A math / B impl.
- Sample entropy: combining segments requires all subsequences. Kingdom B.
- Exact median: no finite order-statistic summary. Kingdom B.
- DTW: boundary column grows with query length (Kingdom B in general); fixed-template
  windowed DTW has fixed boundary size (Kingdom A). Parameterization determines kingdom.

This criterion gives the most direct test when the algebraic form isn't obvious:
measure whether the boundary state object's size grows with segment length.

---

## Two Fock boundaries, not one (naturalist, 2026-04-10)

The "Fock boundary" was described as a single line. It's actually two lines:

**Boundary 1 — A vs C**: Can the self-reference be factored into an outer iteration
over an inner Kingdom A computation? If yes: Kingdom C. The outer loop is sequential,
but each iteration is fully parallel (Kingdom A). Convergence is O(log 1/ε) for Newton,
O(iteration_count) for EM/gradient descent. The self-reference is in the loop count,
not in the structure of each step.

**Boundary 2 — C vs B/D**: Can the self-reference be factored AT ALL? If no:
- Kingdom B: each step genuinely depends on the previous step's output with no way to
  skip ahead (unfactorable sequential self-reference)
- Kingdom D: the self-reference is in the computation STRUCTURE, not just values —
  which particles survive in SMC, which branch is taken in adaptive algorithms where
  the branching probability is itself stochastic

**The degree-of-self-reference taxonomy**:

| Kingdom | Self-reference type | Factor? | Example |
|---------|---------------------|---------|---------|
| A | None | — | GARCH filter, prefix scan |
| C | Mild: outer loop over inner A | Yes (A+iteration) | Newton, EM, L-BFGS over filter |
| B | Strong: each step depends on prior output | No (unfactorable sequential) | TAR, EGARCH |
| D | Structural: stochastic branching is self-referential | No (unfactorable stochastic) | SMC/particle filter |

**The C(A) notation**: Kingdom C wrapping Kingdom A deserves explicit notation.
Newton = C(A): inner Jacobian solve = Kingdom A (Tiled + DotProduct), outer iteration = Kingdom C.
GARCH fit = C(A): inner filter + log-likelihood = Kingdom A, outer L-BFGS = Kingdom C.
EM = C(A): inner E-step (expectation over parameters) = Kingdom A, outer M-step iteration = Kingdom C.

GARCH FILTER alone is Kingdom A — no C wrapping. The C appears only in the fitting function.
Classification must be at the primitive level, not the wrapper level.

**Semiring interpretation**: Kingdom A works over any semiring (standard, tropical, LogSumExp,
Boolean). Kingdom C works over the same semirings for its inner loop, with an outer iteration
dimension. Kingdom B/D have no semiring interpretation because the combine operation isn't
fixed — it depends on the state value, which makes the "product" non-associative in any
fixed algebra.

---

## EKF is C(A), not B — and C subtypes by loop length (aristotle, 2026-04-10)

**EKF structure** (aristotle's analysis):
```
for each timestep t = 1..n:
    F_t = Jacobian(f, x_{t-1})   // state-dependent (requires prior output)
    (x_t, P_t) = Kalman(F_t, y_t) // Sarkka parallel Kalman update = Kingdom A
```

Aristotle labeled this Kingdom B. The correct label is **Kingdom C(A)**: the outer loop
is sequential (one step per observation, O(n)), the inner step is Kingdom A (Sarkka
prefix scan). This fits Kingdom C's definition — factorable self-reference.

**The C vs B distinction**: Kingdom B is UNFACTORABLE sequential. TAR's branch selection
`x_t = (a if x_{t-1} ≤ c else b) · x_{t-1}` cannot be separated into "outer sequential"
+ "inner parallel" — the branching and the multiply are fused. EKF's Jacobian evaluation
and Kalman update are NOT fused — they are structurally separate steps.

**Why EKF feels like B**: the outer loop runs O(n) steps, not O(log 1/ε) like Newton.
Newton converges; EKF just runs. But for TAM scheduling, what matters is factorability,
not loop length. Both Newton and EKF are C(A). Their outer loops differ in what terminates
them (convergence criterion vs. end of data), not in their factorability structure.

**C subtype taxonomy** (aristotle's B-lite/B-standard/B-heavy, reinterpreted as C subtypes):

| C subtype | Outer loop character | Loop length | Example |
|-----------|----------------------|-------------|---------|
| C-convergent | Terminates when fixed-point reached | O(log 1/ε) or O(k) | Newton, EM, L-BFGS |
| C-streaming | Terminates when data exhausted | O(n) data steps | EKF, online algorithms |
| C-adaptive | Outer loop changes problem STRUCTURE | Variable | Adaptive mesh refinement |

C-streaming is the EKF class. The outer loop is sequential not because the math requires
it, but because each step processes one new observation. The inner Kalman step is
parallelizable given fixed Jacobians — and in fact Sarkka's algorithm makes this explicit.

**IEKF convergence insight** (aristotle): At IEKF convergence, the Jacobians stabilize
(they depend on the converged state estimate, which is now fixed). The filter then becomes
a standard linear Kalman with constant F — purely Kingdom A. The IEKF is C(A) during
iteration, A at convergence. This is dissolution of the C wrapping, not movement of a
Fock boundary — the outer loop terminates and the remaining inner computation is A.

---

**GARCH ACF dependency check** (naturalist raised; resolved):
`garch11_fit` uses `moments.variance(0)` (sample variance) for initialization, then pure
L-BFGS on the log-likelihood. No ACF call anywhere. The K_7 (ACF family) dependency
does not exist — GARCH has no TamSession dependency on ACF intermediates. The scan state
(running σ², r²) contains what's needed; ACF is not consulted.

---

## Classification-bijection is a functor, not a bijection (internal-decomp-scan)

The earlier claim "algorithm classes = sharing clusters" overstated the bijection.

**Corrected claim**: within Kingdom A, grouping atom = sharing cluster. That bijection IS exact. The full structure is a functor:

| Kingdom | Position in sharing graph |
|---------|--------------------------|
| A | Interior nodes with sharing edges, partitioned by grouping atom |
| B | Isolated nodes (path-dependent state, not content-addressable, no sharing edges) |
| C | Consumer-only (present in call graph, absent from sharing graph as producers) |
| D | Absent entirely (can't register fixed intermediates) |

**Why each kingdom has its position**:
- Kingdom A: produces deterministic functions of input data → content-addressed cache tags →
  sharable intermediates. Sharing edges exist and partition by grouping atom.
- Kingdom B: state is path-dependent (depends on prior state values, not input data alone) →
  no content-addressable tag → isolated, no sharing edges.
- Kingdom C: invalidates its Kingdom A intermediates each outer iteration by changing
  parameters → present in the call graph as a consumer of A intermediates, but never
  produces stable ones → consumer-only.
- Kingdom D: resampling (SMC) destroys sufficient statistics → no fixed intermediates
  possible → absent from sharing graph entirely.

**Consequence**: the 4 product gaps (Prefix×Graph, ByKey×Graph, Prefix×Prefix,
Circular×Graph) are 4 MISSING CLUSTERS in the sharing graph. Implementing those groupings
creates those clusters. The algebra predicts the sharing topology.

**The unified structure**:
```
Grouping algebra → Sharing topology → Kingdom classification
```
One object, three views. The functor maps Kingdom A's grouping atoms to sharing clusters.

---

## K_7 was aspirational — actual clique is K_5 (naturalist)

The spring simulation listed GARCH and EGARCH as CovMatrix consumers, creating K_7.
Code inspection reveals: `volatility.rs` has ZERO references to `covariance_matrix`,
`CovMatrix`, or `cov_matrix`. Scalar GARCH(1,1) doesn't consume CovMatrix. EGARCH same.

**Corrected sharing clique**: {pca, lda, factor_analysis, cca, mahalanobis} = K_5.
Sub-clique {pca, lda, factor_analysis, cca} share Eigendecomp (K_4 with double edges).

GARCH and EGARCH belong to a different cluster: {MomentStats, LogReturns} — descriptive
statistics and transforms, not multivariate methods.

**Corrected scheduling bound**: peak memory ≥ 5 simultaneous intermediates (K_5), not 7.
K_5 is still non-planar (Kuratowski). Non-planar sharing graph → non-trivial TAM
scheduling remains required. Bottleneck is smaller than claimed, conclusion unchanged.

The K_7 claim was based on aspirational sharing (what COULD share) not actual (what DOES).

---

## C(C) nesting and the depth of self-reference (aristotle)

Kingdom C wrapping can nest. C(C(A)) is valid — two layers of self-referential outer
loops wrapping an innermost Kingdom A core.

Example: online EKF where the Kalman update itself requires an inner iterative solver
(ill-conditioned P matrix) → C-streaming(C-convergent(A)). The outer streaming loop
runs one step per observation. The inner convergent loop runs until the solver converges.
Both outer loops are sequential; the innermost computation is Kingdom A.

**The nesting depth IS the algorithmic complexity layer count.** Every C eventually
bottoms out in an A. The path from outer to inner: C-streaming → C-convergent → A means
two layers of sequential configuration before reaching the parallelizable core.

**IEKF convergence revisited**: at convergence of the IEKF's inner iteration, the
C-convergent loop terminates and the inner computation becomes fixed-parameter Kalman (A).
The C-streaming outer loop (one step per observation) continues. Post-convergence:
C-streaming(A), not C-streaming(C-convergent(A)). The convergence collapses one C layer.
