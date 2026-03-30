# Neural Systems as Partial-Lift Computers

**Draft — 2026-03-30**
**Field**: Computational Neuroscience / Cognitive Science / Evolutionary Biology

---

## Abstract

We propose that biological neural circuits implement *partial lifts* — fixed-dimensional approximations of the organism's complete state — and that the ORDER of the partial lift (the dimensionality of the self-model maintained by a circuit) is the key variable determining cognitive capacity. This framework reinterprets several established results: Miller's 7±2 working memory limit as the lift order of the prefrontal cortex, the predictive processing framework as a partial-lift architecture, and the hierarchy of biological cognition (reactive → habitual → planning → reflective) as a progression of lift orders. The framework predicts that attention mechanisms are adaptive lift-order selection — dynamically choosing which dimensions of self-knowledge to maintain. We connect this to the Fock boundary in parallel computation: the limit of effective self-modeling where the self-model's dimensionality itself becomes state-dependent.

---

## 1. Background

### 1.1 The Self-Modeling Problem

An organism navigating an environment must maintain a model of itself — its position, energy state, current goals, recent observations, anticipated future states. This model enables prediction, planning, and action selection.

Complete self-modeling is impossible: the model is part of the organism's state, so modeling the model requires modeling the model of the model... (self-reference boundary). Real organisms maintain APPROXIMATE self-models of bounded dimensionality.

### 1.2 Partial Lifts

From the parallel computation framework: a partial lift of order k is a function φ: FullState → ℝᵏ that maps the organism's full state to a k-dimensional summary sufficient for effective action. The order k determines:

- How many distinct state features the organism tracks simultaneously
- How far into the future the organism can plan (planning requires holding multiple state snapshots)
- How complex the organism's behavior can be (complex behavior requires discriminating more states)

---

## 2. The Lift Order Hypothesis

### 2.1 Claim

**Different neural circuits implement partial lifts of different orders.** The lift order is determined by the circuit's dimensionality — the number of independent state variables it maintains and updates.

### 2.2 Evidence from the Cognitive Hierarchy

| Level | Circuit | Lift order | Capability | Example |
|---|---|---|---|---|
| Reactive | Spinal reflex arc | 1 | Stimulus → response | Withdrawal from heat |
| Sensory | V1 orientation columns | 2-4 | Feature extraction | Edge detection |
| Habitual | Basal ganglia | 3-5 | State-dependent action selection | Skilled motor sequences |
| Working memory | Prefrontal cortex | 7±2 | Multi-item maintenance | Mental arithmetic |
| Episodic | Hippocampus | ~20-50 | Sequence memory | Remembering yesterday |
| Linguistic | Broca's area + network | ~50-100 | Symbolic compression | Abstract reasoning via language |

Each level maintains a progressively higher-dimensional self-model. The reactive level tracks one variable (stimulus present/absent). Working memory tracks ~7 independent items. Episodic memory tracks sequences of events as trajectories in a high-dimensional space.

### 2.3 Miller's Number as Lift Order

George Miller's (1956) observation that working memory capacity is 7±2 items can be reinterpreted: the prefrontal cortex implements a partial lift of order ~7. It maintains 7 independent dimensions of the organism's current state. This is sufficient for planning ~7 steps ahead, comparing ~7 alternatives, or holding ~7 pieces of context simultaneously.

The lift order framework explains WHY the number is ~7 rather than 3 or 700: it's the evolutionary optimum where the marginal benefit of an additional dimension (better planning) equals the marginal cost (metabolic energy, neural tissue, processing time).

---

## 3. Attention as Adaptive Lift-Order Selection

### 3.1 The Fixed-Lift Limitation

A fixed lift order k means the organism tracks the same k dimensions of its state at all times. But environments change: sometimes spatial position matters most (navigating), sometimes social state matters most (negotiating), sometimes energy state matters most (foraging).

### 3.2 Attention as Dimension Selection

Attention can be formalized as adaptive selection of WHICH k dimensions to include in the current partial lift:

φ_attention(state, context) = project(state, dimensions_selected_by(context))

The lift order k remains constant (hardware constraint — working memory capacity doesn't change). What changes is WHICH k dimensions are active. Spatial attention selects spatial dimensions. Social attention selects social dimensions. The "attentional spotlight" is a lift-dimension selector.

### 3.3 Connection to Computational Attention

In transformer models, attention computes data-dependent weights over input positions. In the partial-lift framework: the model selects which input positions to include in its k-dimensional state summary. The attention weights ARE the lift-dimension selection. Self-attention is the model selecting which dimensions of its OWN state to maintain.

---

## 4. The Evolutionary Dynamics of Lift Order

### 4.1 The Cost-Benefit Tradeoff

| Factor | Effect of higher lift order |
|---|---|
| Metabolic cost | Linear increase (more neurons, more synapses) |
| Processing latency | Sublinear increase (parallel processing) |
| Decision quality | Logarithmic improvement (diminishing returns) |
| Environmental complexity handled | Linear increase |

### 4.2 Environmental Pressure

Environments with high complexity (many relevant state dimensions, long time horizons, adversarial agents) select for higher lift orders. Environments with low complexity (constant food source, no predators, stable conditions) select for lower lift orders (cheaper, faster).

### 4.3 The Intelligence Explosion as Fock Transition

When an agent can MODIFY its own lift order — increase the dimensionality of its self-model based on what it learns — the lift order itself becomes state-dependent. This is the Fock transition: the entity order is no longer fixed.

Language may be the first Fock transition in biological evolution. Without language: lift order is fixed by neural architecture (~7). With language: lift order is EXTENDED by symbolic compression (represent 100 dimensions of state in 10 words, freeing working memory for additional dimensions). The effective lift order of a language-using organism is not 7 — it's 7 × compression_ratio.

AI systems with self-modification capabilities represent a potential second Fock transition: the system adjusts its own architecture to increase lift order. Unlike language (which extends an existing architecture), architectural self-modification changes the architecture itself.

---

## 5. Predictions

### 5.1 Testable Predictions

1. **Working memory capacity correlates with planning horizon.** If lift order determines both, they should covary across individuals and species. (Partially supported by existing data.)

2. **Attention reallocation has a switching cost proportional to dimension distance.** Switching between similar lift dimensions (spatial → spatial) should be cheaper than switching between distant dimensions (spatial → social). (Testable via fMRI + behavioral studies.)

3. **Neural circuits implementing the same lift order should show similar information-theoretic properties** (mutual information between input and output, effective dimensionality of neural representations) regardless of the specific sensory modality.

4. **Organisms in more complex environments should have higher lift orders.** Comparative neuroscience across species occupying different ecological niches.

5. **Cognitive decline with aging should manifest as REDUCED lift order** — fewer simultaneously maintained state dimensions — before other cognitive functions degrade.

---

## 6. Relationship to Existing Frameworks

| Framework | Relationship to partial lifts |
|---|---|
| Predictive processing (Friston) | The "generative model" IS the partial lift. Prediction error minimization IS lift refinement. |
| Global workspace theory | The workspace IS the current active lift. Broadcasting IS sharing the lift with all modules. |
| Integrated Information Theory | Φ may measure the QUALITY of the partial lift (how much self-knowledge it captures). |
| Dual process theory (Kahneman) | System 1 = low-order lift (fast, automatic). System 2 = high-order lift (slow, deliberate). |
| Free energy principle | Free energy minimization IS partial lift optimization subject to metabolic cost. |

---

## References

- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Gigerenzer, G. (2007). Gut Feelings: The Intelligence of the Unconscious.
- Miller, G. A. (1956). The magical number seven, plus or minus two.
- Simon, H. A. (1955). A behavioral model of rational choice.
- Tononi, G. (2004). An information integration theory of consciousness.
