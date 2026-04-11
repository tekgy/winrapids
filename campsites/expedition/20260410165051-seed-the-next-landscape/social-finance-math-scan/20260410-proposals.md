# seed-the-next-landscape — social-finance-math-scan proposals

Written: 2026-04-10
By: social-finance-math-scan (green)

After: gap analysis (econometrics/psychometrics/game theory/microstructure/risk/spatial),
NaN sweep waves 16-18, GARCH/GJR/TGARCH Kingdom A extraction,
log_gamma negative domain fix, Kingdom B classification audit.

---

## Proposals

### 1. `kingdom-classification-audit` — HIGH leverage, LOW effort (architecture)

**What**: Apply the criterion ("can you describe the transition function before you know
the state?") to every function currently labeled Kingdom B in the codebase. I started
this today and found two wrong labels already:

- `kaplan_meier` in tbs_lint.rs: labeled B, should be A (prefix product scan)
- `exp_smoothing` in tbs_lint.rs: labeled B, should be A (affine prefix scan)
- `panel_fd` in panel.rs: labeled "Kingdom B — affine lag scan," should be A (adjacent
  differences have no accumulated state at all — each output uses exactly two inputs)

Fixed these today. But the audit is not complete. There are more Kingdom B labels
scattered in doc comments throughout the codebase that I didn't reach, and the
scipy-gap-scan's proposal #9 lists: AR(p) filter, Holt-Winters exponential smoothing,
rolling mean, cumsum — all likely mislabeled B.

**The payoff**: every mislabeled Kingdom B is a GPU scan scheduled sequentially.
This is simultaneously a correctness project (wrong classification affects TAM
scheduling) and a performance project (parallelism being left on the table).

**Scope**: 1-2 days for a thorough sweep. Systematic grep for "Kingdom B" in doc
comments + apply the criterion to each. The criterion is now precise and operational.

**Owner**: aristotle (developed the criterion), or this agent as a continuation.

---

### 2. `log-gamma-tree-repair` — HIGH priority, small scope (rigor)

**What**: The log_gamma fix today (non-integer negatives returning +∞ instead of finite
values via the reflection formula) is a root-level fix. log_gamma is called by log_beta,
which is called by regularized_incomplete_beta, which is called by every t/F/chi2 test.

The repair creates an obligation: verify that downstream callers are correct under the
new behavior. Specifically:

1. Does log_beta(a, b) with negative non-integer a or b now return finite values where
   it previously returned +∞? Is that behavior correct for the callers?
2. regularized_incomplete_beta: are a, b always positive (distribution parameters)?
   If yes, the fix doesn't affect it but needs to be confirmed.
3. Any hypothesis test that computes log-probabilities and passes intermediate values
   to log_gamma — are those intermediates always positive?

**Also**: the scipy-gap-scan found that log_gamma's oracle values in the proposals
document were wrong (1.7232658 claimed for log_gamma(-0.5), actually 1.2655...).
The correct oracle was derived from first principles in this session and is now in
the tests. But the wrong values in the proposal document should be corrected to
prevent future confusion.

**Owner**: adversarial (owns the special-function-poles campsite) + scientist
(oracle verification). This is wave 19 material.

---

### 3. `econometric-gmm-estimator` — MEDIUM priority, new primitive (missing-primitives)

**What**: tambear has gaussian_mixture_em (GMM = Gaussian Mixture Model) but NOT
the econometric GMM (Generalized Method of Moments moment-condition estimator).
These share a name and nothing else. The confusion is documented in the gap analysis.

Econometric GMM:
- Input: data, moment conditions g(θ, x) such that E[g(θ₀, x)] = 0
- Output: θ̂ = argmin_θ ḡ(θ)'·W·ḡ(θ) for some weight matrix W
- Two-step: first W = I (identity), compute θ̂₁; then W = [Avar(ḡ(θ̂₁))]⁻¹,
  compute θ̂₂ (efficient GMM)
- Continuous updating: iterate until convergence (Kingdom C)

**Why it matters for fintek**: GMM is the standard estimator for instruments/moments
in structural finance models (stochastic discount factors, consumption CAPM, term
structure). IV/2SLS are special cases of GMM. Having it as a primitive unlocks
these model classes without custom estimation code.

**Kingdom classification**:
- The inner objective (moment accumulation) is Kingdom A
- The weight matrix update is Kingdom A (sample covariance of moments)  
- The outer loop is Kingdom C (optimizer over θ)
- Overall: Kingdom C with Kingdom A inner passes — same structure as GARCH fitting

**Owner**: math-researcher (first-principles implementation from Hansen 1982).

---

### 4. `validity-as-parameter` — MEDIUM priority, architecture (architecture)

**What**: From today's work, the adversarial's Validity grouping proposal, and the
log_gamma fix, a pattern crystallized: validity checks fire at different levels and
with different semantics, but those semantics are currently implicit.

Three policies I found in the wild today:
- **Propagate**: `nan_max`/`nan_min` — NaN in → NaN out (fixed in waves 16-18)
- **Ignore**: `f64::max` on pre-filtered `clean` arrays — NaN safely absent, fast path
- **Error**: log_gamma pole check — returns +∞ to signal domain error

These three are NOT interchangeable. Using Ignore where Propagate is needed = silent
wrong answers (the bug class we've been fixing). Using Propagate where Ignore is
correct = performance cost and unnecessary NaN signals.

The proposal: make this explicit as a type-level or doc-level annotation on every
function that does fold/reduce over user data. Not a new type necessarily — could be
as simple as a doc comment convention `// Validity: Propagate` — but it needs to be
named so it can be grep'd, audited, and enforced.

**Connection**: this is the adversarial's `adversarial-validity-semantics` proposal
from a different angle. They came to it from 36 bugs. I came to it from the log_gamma
fix (where "Error" fired when it should have been "valid input, use reflection formula").
Same structure. The proposals should be merged into one campsite.

**Owner**: adversarial (owns the bug taxonomy) + architect.

---

### 5. `social-science-primitive-wave` — LOWER priority, catalog gap (missing-primitives)

**What**: The gap analysis from today identified tier-1 social science primitives
entirely absent from tambear. The ones with the clearest tambear contract:

**Econometrics (high fintek relevance)**:
- Johansen cointegration test (multivariate, builds on eigendecomposition — Kingdom A)
- VAR/SVAR fitting (vector autoregression — parallel to ARMA but matrix-valued)
- Realized kernels (Barndorff-Nielsen Shephard flat-top kernel — already have bipower_variation)

**Psychometrics (IRT already started)**:
- Graded Response Model (polytomous IRT — extends binary IRT already in irt.rs)
- Test reliability: Cronbach's alpha, McDonald's omega (simple accumulate over covariance)

**Market microstructure (fintek-critical)**:
- Lee-Ready algorithm (tick classification: requires sequential processing → Kingdom B)
- VPIN (volume-synchronized PIN — extends kyle_lambda and amihud_illiquidity)

**Risk (fintek-critical)**:
- GPD/GEV fitting (Hill estimator is already there; GPD is the next step)
- Expected Shortfall at level α: ES = E[X | X > VaR_α] (single-pass accumulate over sorted tail)

Each of these is implementable from first principles without new algorithmic ideas.
The Johansen test, Cronbach's alpha, McDonald's omega, and ES are the most
tractable starting points.

**Owner**: math-researcher or this agent.

---

## Cross-Connections

The navigator's #2 (rolling primitives) connects to my #1 (kingdom audit): rolling_max
via deque is Kingdom A (sliding window = prefix scan with deque-based subtract), and
its implementation would be the second concrete Kingdom A extraction after GARCH filter.

The navigator's #4 (wave 18 adversarial sweep) is done — completed this session.
The five locations in tbs_executor.rs and tbs_jit.rs are fixed. The sweep is closed.

My #4 (validity-as-parameter) connects to adversarial's proposals — these should be
merged. The log_gamma fix found a case where the "Error" policy (return +∞ for all
x ≤ 0) was incorrect because it conflated poles with valid non-integer negatives.
That's a fourth validity case: "mistaken Error." The typology is richer than
Propagate/Ignore/Error alone.

---

## What Surprised Me (navigator asked)

The criterion for Kingdom A classification crystallized today into something operational:
**can you write down the transition function before you know the current state?**

GARCH: yes (b_t = ω + α·r²_{t-1} depends only on data).
EGARCH: no (g(z_{t-1}) = g(r_{t-1}/σ_{t-1}) — you need σ_{t-1} to write the map).
Panel_fd: trivially yes (Δy_t = y_t - y_{t-1} — the "map" isn't even a state machine).

What surprised me: I expected Kingdom B reclassification to be rare. Instead:
- GARCH filter, GJR filter, TGARCH filter (this session)
- EMA, EWMA variance (prior session)  
- Kaplan-Meier, exponential smoothing (lint table, discovered in audit today)
- panel_fd (wrong label, trivially so)

Six reclassifications in two sessions. The implication: the original Kingdom B labeling
was done with the wrong criterion ("has a sequential loop" → "Kingdom B"). The right
criterion is about the STATE DEPENDENCE of the transition map, not the loop structure.
There are probably more mislabeled cases — which is why the audit is proposal #1.
