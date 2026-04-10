# KalmanOp Design

*Scout analysis, 2026-03-30. Navigator provocation: is KalmanOp genuinely different from EWMOp?*
*Postscript: resolved by pathmaker's KalmanAffineOp implementation, 2026-03-30.*

---

## Resolution (added after implementation)

The pathmaker implemented `KalmanAffineOp` — steady-state Kalman with DARE constructor. Key findings:

**DARE is the Möbius scan at its fixed point.** The two-stage GPU design (Möbius P scan → affine x scan) is right for the TRANSIENT case. For steady-state (filter converged, N large), Stage 1 collapses: the Möbius scan converges to a constant P_inf, and DARE computes that constant in Rust at construction time. No GPU P scan needed. One GPU stage is the correct steady-state implementation.

**Measured results on GPU (F=0.98, H=1.0, Q=0.01, R=0.1):**
- K_ss = 0.258, A = 0.727
- n=100K: 316μs, max_err = 5.68e-14 (machine epsilon)
- State: 16 bytes (2 doubles). Combine: 2 muls + 1 add.
- 13μs overhead vs AddOp — pure shared memory bandwidth, not compute.

**Family overlap confirmed:** KalmanAffineOp with F=1, H=1 is exactly EWMOp with alpha=K_ss. Operator space is continuous; families are clusters, not disjoint categories.

**Trait validation:** AssociativeOp designed for AddOp (8 bytes, 1 FLOP) holds KalmanAffineOp (16 bytes, 3 FLOPs, DARE constructor, GPU sizeof validation). Zero trait changes.

The two-stage GPU design in this doc remains the right path for the transient case — V-column variance output, non-stationary systems, short series where the filter hasn't converged.

---

## Preliminary: EWMOp State Check

Before designing KalmanOp, confirmed the current EWMOp combine body in `winrapids-scan/src/ops.rs`:

```rust
double decay = pow(1.0 - {alpha}, (double)b.count);
result.weight = a.weight * decay + b.weight;
result.value  = a.value  * decay + b.value;
result.count  = a.count + b.count;
```

The exponent is `b.count`, not `1`. The bug flagged earlier either was fixed or was a misread. Current implementation is correct: the earlier segment decays by `(1-alpha)^(length of later segment)`, which is exactly right for parallel EWM merge.

---

## Answer: Is KalmanOp Genuinely Different?

**Short answer**: Steady-state KalmanOp is EWMOp. Transient KalmanOp with variance output is genuinely new. The interesting case is Option 2.

---

## The Math

### Scalar 1D Kalman (random walk model)

State model: `x_t = x_{t-1} + w_t`, `w_t ~ N(0, q)`
Observation: `y_t = x_t + v_t`, `v_t ~ N(0, r)`

Kalman recursion:
```
P_pred_t  = P_{t-1} + q                          (variance predict)
K_t       = P_pred_t / (P_pred_t + r)            (gain)
x_t       = (1 - K_t) * x_{t-1} + K_t * y_t     (state update)
P_t       = (1 - K_t) * P_pred_t                  (variance update)
         = r * P_pred_t / (P_pred_t + r)
```

### Steady-state convergence

P_t converges to P_inf satisfying:
```
P_inf = r * (P_inf + q) / (P_inf + q + r)
```

Solving the quadratic: `P_inf = q/2 * (-1 + sqrt(1 + 4r/q))`

At steady state, `K_inf = (P_inf + q) / (P_inf + q + r)` is constant, and:
```
x_t = (1 - K_inf) * x_{t-1} + K_inf * y_t
```

**This is exactly EWM with `alpha = K_inf(q, r)`.** Structurally identical. Same algorithm.

So Option 1 is correct: **steady-state KalmanOp = EWMOp** with a `(q, r)` parameterization that derives `alpha` from a noise model. More interpretable, no new algorithm.

---

## What the Transient Case Adds

Before convergence, `K_t` varies. The filter starts more responsive (K near 1) and settles toward K_inf. During this transient:

1. **Time-varying gains** — the effective alpha changes at each step, not fixed.
2. **Variance output** — `P_t` is the running uncertainty estimate. EWMOp has no equivalent.
3. **Meaningful uncertainty** — `P_t` can be used for V columns (confidence metadata per tick).

The P recursion is where the algebraic richness lives.

---

## The P Recursion Is a Möbius Transform

```
P_{pred,t+1} = r * P_pred_t / (P_pred_t + r) + q
             = ((r + q) * P_pred_t + q*r) / (P_pred_t + r)
```

This is a Möbius (linear fractional) transform: `T(P) = (aP + b)/(cP + d)` with:
- `a = r + q`
- `b = q * r`
- `c = 1`
- `d = r`

Möbius transforms compose via 2×2 matrix multiply (this is classical projective geometry). The accumulated transform after N steps:
```
T^N = [[a, b], [c, d]]^N     (matrix power)
P_pred_N = T^N(P_pred_0)
```

Matrix multiplication is **associative**. The P scan fits `AssociativeOp` exactly.

### MobiusOp design

```rust
pub struct MobiusKalmanOp {
    pub q: f64,  // process noise variance
    pub r: f64,  // measurement noise variance
}

// State: (a, b, c, d) — accumulated 2×2 Möbius matrix
// Identity: [[1, 0], [0, 1]]
// Element per step: [[r+q, q*r], [1, r]]  (SAME for every step — data-independent!)
// Combine: 2×2 matrix multiply
// Lift: (_y: f64) → element  (ignores observation entirely)
// Extract: apply (a,b,c,d) to P_pred_0 → P_pred_t

// CUDA state: struct MobiusState { double a, b, c, d; }
// Identity: { 1, 0, 0, 1 }
// Lift: any y → { r+q, q*r, 1.0, r }   (y unused)
// Combine: standard 2×2 multiply
// Extract: (a * P0 + b) / (c * P0 + d)
```

Key property: **lift ignores the observation**. The P scan is data-independent. You could compute all P_t values before seeing any y_t. This is unique — every other operator in ops.rs lifts FROM the data. MobiusKalmanOp lifts to a constant element for any input.

### The `output_width` distinction

MobiusKalmanOp's primary extract gives `P_t` (variance). But K_t (the gain used by Stage 2) is:
```
K_t = P_pred_t / (P_pred_t + r)
```

This can be exposed as a secondary extract:
```
cuda_extract_secondary: ["(s_applied_to_P0 / (s_applied_to_P0 + r))"]
```

So MobiusKalmanOp with `output_width = 2` produces both `P_t` and `K_t` arrays.

---

## Stage 2: The Affine Scan with Time-Varying K_t

Given K_t from Stage 1, the state update is:
```
x_t = (1 - K_t) * x_{t-1} + K_t * y_t
```

An affine map: `x → m_t * x + b_t` where `m_t = 1 - K_t`, `b_t = K_t * y_t`.

Affine maps compose associatively:
```
(m2, b2) ∘ (m1, b1) = (m2*m1, m2*b1 + b2)
```

Identity: `(1, 0)`. This is `AssociativeOp`-compatible.

**The problem**: `cuda_lift_element` takes a single scalar `x`. But Stage 2's lift needs BOTH `y_t` (the observation) AND `K_t` (the gain from Stage 1). Two separate arrays.

### Can it fit AssociativeOp cleanly?

**Not with the current trait signature.** The lift step reads from one array. Stage 2 needs two.

Three options:

**Option A — Preprocess into packed structs** (cleanest for the framework)
A `fused_expr` step produces a 2-element buffer per tick: `(K_t, K_t * y_t)`. Then a modified `AffineOp` lifts from a packed 2-double struct, not a scalar. Requires extending `cuda_lift_element` to accept a struct type, or adding `cuda_lift_body` that reads from a 2D array.

**Option B — New primitive: `AffineScan`**
`PrimitiveOp::AffineScan(inputs=[slopes_arr, observations_arr])` with the combine logic baked in. Not an `AssociativeOp` at all — a new primitive like `TiledReduce`. More straightforward to implement; less composable.

**Option C — `KalmanFilter` as a single primitive**
`PrimitiveOp::KalmanFilter` wraps both stages internally (Möbius → K extraction → Affine scan). The compiler treats it as one primitive; the GPU kernel runs two passes. Analogous to how `TiledReduce` wraps two reduction passes internally.

**Scout recommendation**: Option A is the right long-term direction (keeps AssociativeOp general, uses fused_expr to bridge stages). Option C is the minimum useful implementation — it ships without trait changes and gives correct results.

---

## Multi-Stage Structure as Specialist

Regardless of which option is chosen for Stage 2, the specialist's `primitive_dag` has clear structure:

```
KalmanFilterSpecialist:
  primitive_dag:
    1. scan(op=mobius_kalman, input=data, params=(q, r))  →  P_pred array
    2. fused_expr(op=kalman_gain, inputs=[P_pred], params=(r,))  →  K array
    3. fused_expr(op=affine_lift, inputs=[data, K], params=())  →  lift_pairs array
    4. scan(op=affine, input=lift_pairs, params=())  →  x array

  outputs:
    - x    (filtered state estimate)
    - P    (running variance / uncertainty)
    - K    (Kalman gains, useful for diagnostics)
```

Steps 2-3 are fused_expr (element-wise, single kernel each). Steps 1 and 4 are scans.

The four-node DAG automatically benefits from CSE. If two leaves both run KalmanFilter on the same `data` variable with the same `(q, r)` params, the Möbius scan (Step 1) is deduplicated — computed once, shared. This is something the current FinTek runner couldn't do at all.

---

## Rust Struct Sketches

### MobiusKalmanOp (Stage 1)

```rust
pub struct MobiusKalmanOp {
    pub q: f64,
    pub r: f64,
}

impl AssociativeOp for MobiusKalmanOp {
    fn name(&self) -> &'static str { "mobius_kalman" }

    fn cuda_state_type(&self) -> String {
        r#"struct MobiusState { double a, b, c, d; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{ 1.0, 0.0, 0.0, 1.0 }".into()  // identity matrix
    }

    fn cuda_lift_body(&self) -> String {
        // Data-independent: same element regardless of x
        format!(
            r#"    state_t s;
    s.a = {rq};  s.b = {qr};
    s.c = 1.0;   s.d = {r};
    return s;"#,
            rq = self.r + self.q,
            qr = self.q * self.r,
            r = self.r,
        )
    }

    fn cuda_combine_body(&self) -> String {
        // 2×2 matrix multiply
        r#"    state_t result;
    result.a = a.a*b.a + a.b*b.c;
    result.b = a.a*b.b + a.b*b.d;
    result.c = a.c*b.a + a.d*b.c;
    result.d = a.c*b.b + a.d*b.d;
    return result;"#.into()
    }

    fn cuda_extract(&self) -> String {
        // Apply accumulated Möbius to P_pred_0 to get P_pred_t
        // (caller supplies P_pred_0 as initial state)
        // For prefix scan output: P_pred_t = (a * P0 + b) / (c * P0 + d)
        // Default extract just gives 'a' (useless without P0) —
        // real extract requires a separate fused_expr pass.
        // TODO: needs special handling for the P0 binding.
        "s.a".into()  // placeholder; real impl needs P0 injection
    }

    fn output_width(&self) -> usize { 2 }  // P_pred + K

    fn cuda_extract_secondary(&self) -> Vec<String> {
        // K_t from P_pred_t: K = P_pred / (P_pred + r)
        // Same P0 injection issue as extract.
        vec![format!("(s.a / (s.a + {r}))", r = self.r)]
    }

    fn params_key(&self) -> String {
        format!("q={:.10},r={:.10}", self.q, self.r)
    }

    fn state_byte_size(&self) -> usize { 32 }  // 4 × f64
}
```

### AffineOp (Stage 2, time-varying EWM)

```rust
pub struct AffineOp;

impl AssociativeOp for AffineOp {
    fn name(&self) -> &'static str { "affine" }

    fn cuda_state_type(&self) -> String {
        r#"struct AffineState { double m, b; }"#.into()
    }

    fn cuda_identity(&self) -> String { "{ 1.0, 0.0 }".into() }

    fn cuda_lift_body(&self) -> String {
        // x here is a packed struct (m_t, b_t), pre-computed by fused_expr
        // m_t = 1 - K_t, b_t = K_t * y_t
        r#"    // x is packed as struct { double m; double b; }
    state_t s;
    s.m = x.m;
    s.b = x.b;
    return s;"#.into()
    }

    fn cuda_combine_body(&self) -> String {
        // Affine composition: (m2, b2) ∘ (m1, b1) = (m2*m1, m2*b1 + b2)
        r#"    state_t result;
    result.m = a.m * b.m;
    result.b = a.m * b.b + a.b;
    return result;"#.into()
    }

    fn cuda_extract(&self) -> String {
        // Apply to x_0: m * x_0 + b (x_0 injected as initial state)
        "s.b".into()  // placeholder; real impl needs x_0 injection
    }

    fn state_byte_size(&self) -> usize { 16 }  // 2 × f64
}
```

---

## What KalmanOp Gives That EWMOp Can't

| Property | EWMOp | Steady-State Kalman | Transient Kalman |
|----------|-------|---------------------|------------------|
| Algorithm | Fixed alpha EWM | Same EWM | Time-varying gains |
| Parameterization | alpha | (q, r) → alpha | (q, r) + P_0 |
| Variance output | No | No | Yes: P_t |
| Transient behavior | No (flat alpha) | No (flat alpha) | Yes: high gain initially |
| Uncertainty signal | No | No | Yes: V column for free |

The uncertainty output is the value. P_t tells downstream consumers "how much does the filter trust this estimate?" — exactly the kind of signal that WinRapids' V columns are designed to carry.

---

## Answers to Navigator's Questions

**(a) Can this fit AssociativeOp cleanly?**

Stage 1 (Möbius scan): YES, fits cleanly. The lift is data-independent (same matrix for every step), combine is 2×2 matrix multiply, both standard AssociativeOp methods.

Stage 2 (Affine scan): Fits AssociativeOp for the combine logic. The lift requires a two-input element (K_t from Stage 1 + y_t from data). Current trait signature doesn't support this directly. Cleanest resolution: preprocess into packed lift-pairs via fused_expr, then the scan lift unpacks a struct.

**(b) Does it require multi-stage?**

Yes. The P scan must complete before K_t is known. K_t must be computed before the x scan can lift. It's two sequential scans with a fused_expr bridge. This is structurally determined — you cannot fold them into one AssociativeOp without making the state carry a rational function of P_0.

This is the same class of problem as the Fock boundary: the two stages aren't sequential because of bad design, they're sequential because the x computation genuinely depends on the output of the P computation.

**(c) What outputs that EWMOp can't give?**

1. **P_t** — running variance. The filter's uncertainty at each tick. EWM has no uncertainty estimate.
2. **K_t** — time-varying gains. Useful for diagnostics: how responsive was the filter at each step?
3. **Transient behavior** — the initial high-gain phase. EWM has no warmup concept; it starts with fixed alpha immediately.
4. **Physical parameterization** — `(q, r)` has a noise interpretation. Alpha is opaque; noise ratio q/r means something ("how fast does the signal change vs. how noisy are the measurements").

**(d) Minimum useful implementation**

For minimum useful: implement `MobiusKalmanOp` (Stage 1) as an `AssociativeOp`, add a `fused_expr` step to compute K_t, treat Stage 2 as a `fused_expr`-based affine scan (non-parallel for now). The outputs are `x_t` (filtered state) and `P_t` (variance). The V column is free.

The `P0 injection` problem in extract (applying the accumulated Möbius to an initial value) needs to be worked out at the execution layer — it's the same problem as setting the initial state for any prefix scan. WelfordOp solves it by starting with the identity; MobiusKalmanOp needs a non-identity initial P_0. This is a clean design question for the scan engine, not a fundamental obstacle.

---

## One Structural Observation

MobiusKalmanOp's lift is data-independent — it maps every observation `y_t` to the same Möbius matrix element `[[r+q, qr], [1, r]]`. This means the P scan could be pre-computed at spec-construction time (it's a function of `(q, r, N, P_0)` only), and the result stored in the provenance cache keyed by `(q, r, N, P_0)` alone.

WinRapids' provenance system would recognize this: if two KalmanFilter specialists share the same `(q, r)` params but operate on different data, their P scans have the SAME provenance hash (same Möbius matrices, same N, same P_0). The Möbius scan is computed once and cached. The affine scan is computed separately for each data variable.

This is a free win from the provenance system — no special casing needed. The structure produces it.
