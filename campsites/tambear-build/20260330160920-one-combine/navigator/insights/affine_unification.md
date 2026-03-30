# one-combine: Navigator Notes

## The unification I see

`KalmanAffineOp` in winrapids-scan/src/ops.rs IS the universal AffineOp — it just has Riccati logic baked in. The generalization is clean:

```rust
pub struct AffineOp {
    pub a: f64,  // fixed decay constant: x[t] = a * x[t-1] + b[t]
}
```

State: `{a_acc, b_acc}` — unchanged from KalmanAffineOp.
Identity: `{1.0, 0.0}` — unchanged.
Lift(x): `{a, x}` — b[t] IS the raw input.
Combine: `{b.a_acc * a.a_acc, b.a_acc * a.b_acc + b.b_acc}` — unchanged.
Extract: `b_acc` — unchanged.

That's it. The Riccati solver becomes a named constructor.

## The named constructors

```rust
impl AffineOp {
    /// AR(1) process: x[t] = phi * x[t-1] + noise[t]
    pub fn ar1(phi: f64) -> Self { Self { a: phi } }

    /// Leaky integrator: output[t] = decay * output[t-1] + input[t]
    pub fn leaky(decay: f64) -> Self { Self { a: decay } }

    /// Kalman steady-state: solves Riccati, returns AffineOp with a = (1-K_ss*H)*F
    pub fn kalman_ss(f: f64, h: f64, q: f64, r: f64) -> Self { ... }

    /// EWM approximation: a = 1-alpha, b[t] = alpha * z[t].
    /// NOTE: NOT equivalent to EWMOp for small t (initialization differs).
    /// EWMOp normalizes by weight sum; AffineOp assumes x_0 = 0.
    pub fn ewm_approx(alpha: f64) -> Self { Self { a: 1.0 - alpha } }
}
```

And the lift needs to fold in the pre-multiplier: for Kalman-SS, b[t] = K_ss * z[t], not z[t]. So lift = `{a, k_ss * x}`. For the general case, the lift just uses x directly.

## The generalized lift

Actually, the lift needs a coefficient:

```rust
pub struct AffineOp {
    pub a: f64,           // recurrence decay: x[t] = a * x[t-1] + b_coeff * input[t]
    pub b_coeff: f64,     // input scaling: b[t] = b_coeff * input[t]
}
```

Lift(x) = `{a, b_coeff * x}`. With b_coeff=1.0, input IS b. With b_coeff=K_ss, input is the observation. This is one extra multiply in the lift (no cost in the combine — which is the hot path).

## EWMOp vs AffineOp::ewm_approx

The divergence for small t is a genuine mathematical difference, not a bug:
- EWMOp: `x[t] = value_sum / weight_sum` where weights decay exponentially
- AffineOp(1-alpha): `x[t] = (1-alpha)^t * 0 + alpha * sum_k (1-alpha)^{t-k} * z[k]`

The AffineOp version is unnormalized — the weight sum is `alpha * (1 - (1-alpha)^t) / alpha = 1 - (1-alpha)^t`, which converges to 1 as t→∞ but is small for small t. EWMOp explicitly divides by this weight sum, giving the normalized version.

Naturalist should benchmark both and confirm the divergence matches theory.
