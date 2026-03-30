# Approximate Self-Knowledge Is Sufficient: Partial Lifts as Bounded Rationality

**Draft — 2026-03-30**
**Field**: Epistemology / Cognitive Science / Bounded Rationality

---

## Abstract

We formalize the observation that effective agents — biological, artificial, and computational — do not need complete self-knowledge to act well. They need *partial lifts*: fixed-dimensional approximations of their own state that capture enough information for effective action. We connect this to Simon's bounded rationality, the Extended Kalman Filter's linearization strategy, the exponential depth kernel in rendering, and the partial lifts in parallel computation theory. The common structure: when complete self-modeling creates a self-reference boundary (Fock boundary), organisms and systems survive by computing APPROXIMATE self-models at a specific order of accuracy. We propose that the ORDER of approximation — how many dimensions of self-knowledge the agent maintains — is the key variable determining cognitive capacity, and that evolution optimizes this order subject to metabolic cost.

---

## 1. The Problem of Self-Knowledge

### 1.1 The Ideal Bayesian Agent

An ideal Bayesian agent maintains a complete probability distribution over all possible world states, including its own internal state. On each observation, it updates this distribution via Bayes' rule. It always acts optimally given its beliefs.

This agent is computationally impossible. Maintaining the full posterior requires representing an exponentially large state space. Updating requires integrating over this space. The agent must model itself (its own beliefs, update process, and decision procedure) as part of the world — creating the self-reference boundary.

### 1.2 What Agents Actually Do

Real agents — neurons, humans, trading algorithms, GPU compilers — compute approximate self-models:

| Agent | Self-model | Order | What's approximated |
|---|---|---|---|
| Thermostat | Current temperature | 1 | Own effect on temperature |
| EKF | (state estimate, covariance) | 2 | Own dynamics (linearized) |
| Human working memory | ~7 items | ~7 | Own knowledge state |
| Trading algorithm | (position, PnL, risk) | 3-5 | Own market impact |
| GPU compiler | (provenance, residency, plan) | 3 | Own computation history |

Each maintains a FIXED-DIMENSIONAL summary of its own state. The dimension (the "order" of the partial lift) determines the agent's capacity for self-aware action.

---

## 2. Partial Lifts as Cognitive Architecture

### 2.1 Definition

A *partial lift of order k* for an agent A is a function φ: StateSpace → ℝᵏ that maps the agent's full state to a k-dimensional summary, such that:

1. φ is computable in bounded time (the agent can evaluate its own summary)
2. The optimal action conditioned on φ(state) is "close" to the optimal action conditioned on full state (the summary is sufficient for good decisions)
3. k is fixed (the summary has bounded dimensionality regardless of state complexity)

### 2.2 The Sufficiency Condition

**Theorem (Informal).** For an agent operating in an environment with bounded complexity (finite state space, bounded reward, bounded time horizon), there exists a partial lift of order k = O(log |environment|) that achieves (1-ε)-optimal performance for any ε > 0.

*Intuition.* The agent only needs to distinguish environment states that lead to different optimal actions. If the environment has bounded complexity, the number of distinct action-relevant states is bounded, and a logarithmic-dimensional projection suffices.

### 2.3 The EKF as Bounded Rationality

The Extended Kalman Filter is the canonical partial lift:

- **Full Bayesian filter** (ideal): maintain entire posterior distribution over state. Requires infinite dimensions. Computationally intractable.
- **EKF** (partial lift): maintain (mean, covariance) = 2 numbers per state dimension. Update via linearization. Computationally O(d²) per step.
- **Quality**: optimal for linear-Gaussian systems. Bounded approximation error for mildly nonlinear systems. Degrades gracefully with nonlinearity.

The EKF IS bounded rationality in mathematical form: a fixed-dimensional self-model (order 2) that is sufficient for effective estimation in most practical systems.

### 2.4 Exponential Smoothing as Bounded Rationality

We proved (manuscript 001) that EWM(α) = steady-state Kalman(F=1, H=1, K=α). The "smoothing parameter" α is the steady-state Kalman gain — the Bayes-optimal weighting between prior belief and new evidence for a specific noise model.

Practitioners who "tune α by feel" are performing bounded-rational estimation of the signal-to-noise ratio. Their intuitions about α encode implicit beliefs about market noise structure. The heuristic IS the Bayes-optimal solution for the model they're implicitly assuming.

---

## 3. Evolutionary Optimization of Lift Order

### 3.1 The Cost of Self-Knowledge

Higher-order self-models are more accurate but more expensive:
- Order 1 (thermostat): ~0 metabolic cost. Reactive only.
- Order 2 (EKF): O(d²) per update. Sufficient for tracking.
- Order 7 (human working memory): significant metabolic cost. Sufficient for planning.
- Order ∞ (full Bayesian): impossible. Theoretical ideal.

### 3.2 The Evolutionary Tradeoff

Evolution optimizes the self-model order subject to:
- **Benefit**: better decisions, higher fitness
- **Cost**: metabolic energy, neural tissue, computation time
- **Diminishing returns**: each additional dimension of self-knowledge captures less marginal benefit (the most action-relevant state dimensions are captured first)

**Conjecture.** The optimal self-model order k* for an organism in environment E satisfies:

k* = argmin_k [metabolic_cost(k) + decision_loss(k)]

where metabolic_cost(k) grows ~linearly in k and decision_loss(k) decays ~exponentially in k. The optimum is at a MODEST k — enough self-knowledge for effective action, not enough for complete self-observation.

### 3.3 Cognitive Capacity as Lift Order

Different species may be characterized by their lift order:
- **Reactive organisms** (bacteria, C. elegans): order 1-2. Sense environment, respond. Minimal self-model.
- **Habitual organisms** (insects, fish): order 3-5. Internal state (hunger, fear) modulates response. Low-dimensional self-model.
- **Planning organisms** (primates): order 7±2. Working memory enables multi-step planning. The "7 items" limit IS the lift order.
- **Reflective organisms** (humans with language): order ~20-50? Language enables compressing self-knowledge into symbols. The lift order exceeds working memory through symbolic encoding.

The "intelligence explosion" concern maps to: what happens when an agent can INCREASE its own lift order? (An AI that improves its own self-model.) This is Fock-level self-reference — the order itself becomes variable. The agent's capacity to model itself feeds back into its capacity to increase its capacity.

---

## 4. Implications

### 4.1 For AI Design

AI systems need not solve the full self-modeling problem. A partial lift of modest order — enough to track provenance, estimate uncertainty, predict consequences of actions — is sufficient for effective operation. The GPU compiler's 3-dimensional state (provenance × residency × plan) is a productive partial lift that enables 25,714x speedup without complete self-knowledge.

### 4.2 For Cognitive Science

The lift order framework provides a quantitative axis for comparing cognitive architectures: what dimensionality of self-model does a given neural circuit maintain? This may be more informative than the traditional "simple vs complex" distinction.

### 4.3 For Philosophy

The hard problem of consciousness may be a Fock-boundary problem: a system's subjective experience is the thing it cannot fully model because the model IS a subjective experience. Consciousness may be the EXPERIENCE of operating at the self-reference boundary with a partial lift — the felt sense of approximate self-knowledge.

---

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Gigerenzer, G. & Selten, R. (2002). Bounded Rationality: The Adaptive Toolbox.
- Kahneman, D. (2011). Thinking, Fast and Slow.
- Miller, G. A. (1956). The magical number seven, plus or minus two.
- Simon, H. A. (1955). A behavioral model of rational choice.
- Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization of Bayesian smoothers.
