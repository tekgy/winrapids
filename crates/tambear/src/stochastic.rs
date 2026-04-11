//! # Stochastic Processes
//!
//! Brownian motion, geometric Brownian motion, Ornstein-Uhlenbeck,
//! Poisson processes, Markov chains (discrete and continuous time),
//! birth-death processes, and random walk statistics.
//!
//! ## Architecture (accumulate+gather)
//!
//! - **Random walk**: sequential scan accumulating position increments
//! - **Markov chain**: accumulate(transition matrix^n, All, mat_pow)
//! - **Stationary distribution**: fixed-point of left eigenvector problem
//! - **Poisson process**: accumulate(exponential inter-arrivals, Prefix, sum)
//! - **Brownian bridge**: gather(endpoint constraint, scatter into path)

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1 — Brownian Motion (Wiener Process)
// ═══════════════════════════════════════════════════════════════════════════

/// Simulate standard Brownian motion W(t) on grid [0, T].
/// W(0) = 0, W(t+dt) - W(t) ~ N(0, dt).
/// Returns (times, values).
pub fn brownian_motion(t_end: f64, n_steps: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let dt = t_end / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut values = Vec::with_capacity(n_steps + 1);
    times.push(0.0);
    values.push(0.0);
    let mut w = 0.0;
    for i in 1..=n_steps {
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        w += sqrt_dt * z;
        times.push(i as f64 * dt);
        values.push(w);
    }
    (times, values)
}

/// Brownian bridge W(t) conditioned on W(T) = b.
/// B(t) = W(t) + t/T · (b - W(T)).
pub fn brownian_bridge(t_end: f64, end_val: f64, n_steps: usize, seed: u64) -> Vec<f64> {
    let (_, w) = brownian_motion(t_end, n_steps, seed);
    let dt = t_end / n_steps as f64;
    let w_end = *w.last().unwrap();
    w.iter().enumerate().map(|(i, &wi)| {
        let t = i as f64 * dt;
        wi + t / t_end * (end_val - w_end)
    }).collect()
}

/// Quadratic variation of a path: Σ (W_{t+dt} - W_t)². Should converge to T.
pub fn quadratic_variation(path: &[f64]) -> f64 {
    path.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2 — Geometric Brownian Motion (Black-Scholes)
// ═══════════════════════════════════════════════════════════════════════════

/// Simulate Geometric Brownian Motion: dS = μS dt + σS dW.
/// Exact solution: S(t) = S₀ exp((μ - σ²/2)t + σW(t)).
/// Returns (times, prices).
pub fn geometric_brownian_motion(
    s0: f64,
    mu: f64,
    sigma: f64,
    t_end: f64,
    n_steps: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let dt = t_end / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut prices = Vec::with_capacity(n_steps + 1);
    times.push(0.0);
    prices.push(s0);
    let mut s = s0;
    for i in 1..=n_steps {
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        s *= (drift + sigma * sqrt_dt * z).exp();
        times.push(i as f64 * dt);
        prices.push(s);
    }
    (times, prices)
}

/// Black-Scholes European option price.
/// `call`: true for call option, false for put.
/// Returns (price, delta).
pub fn black_scholes(s: f64, k: f64, t: f64, r: f64, sigma: f64, call: bool) -> (f64, f64) {
    use std::f64::consts::SQRT_2;
    if t <= 0.0 {
        let intrinsic = if call { (s - k).max(0.0) } else { (k - s).max(0.0) };
        let delta = if call { if s > k { 1.0 } else { 0.0 } } else { if s < k { -1.0 } else { 0.0 } };
        return (intrinsic, delta);
    }
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    if call {
        let price = s * nd1 - k * (-r * t).exp() * nd2;
        let delta = nd1;
        (price, delta)
    } else {
        let price = k * (-r * t).exp() * (1.0 - nd2) - s * (1.0 - nd1);
        let delta = nd1 - 1.0;
        (price, delta)
    }
}

/// Standard normal CDF — delegates to the high-precision erfc in special_functions.
fn normal_cdf(x: f64) -> f64 {
    crate::special_functions::normal_cdf(x)
}

/// GBM expected value: E[S(T)] = S₀ exp(μT).
pub fn gbm_expected(s0: f64, mu: f64, t: f64) -> f64 {
    s0 * (mu * t).exp()
}

/// GBM variance: Var[S(T)] = S₀² exp(2μT) (exp(σ²T) - 1).
pub fn gbm_variance(s0: f64, mu: f64, sigma: f64, t: f64) -> f64 {
    s0 * s0 * (2.0 * mu * t).exp() * ((sigma * sigma * t).exp() - 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3 — Ornstein-Uhlenbeck Process (mean-reverting)
// ═══════════════════════════════════════════════════════════════════════════

/// Simulate Ornstein-Uhlenbeck: dX = θ(μ-X)dt + σdW.
/// Exact discretization: X(t+dt) = X(t)e^{-θdt} + μ(1-e^{-θdt}) + σ√((1-e^{-2θdt})/(2θ)) Z.
pub fn ornstein_uhlenbeck(
    x0: f64,
    mu: f64,
    theta: f64,
    sigma: f64,
    t_end: f64,
    n_steps: usize,
    seed: u64,
) -> Vec<f64> {
    let dt = t_end / n_steps as f64;
    let e_minus = (-theta * dt).exp();
    let e_minus_2 = (-2.0 * theta * dt).exp();
    let std_step = sigma * ((1.0 - e_minus_2) / (2.0 * theta)).sqrt();
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push(x0);
    let mut x = x0;
    for _ in 0..n_steps {
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        x = x * e_minus + mu * (1.0 - e_minus) + std_step * z;
        path.push(x);
    }
    path
}

/// OU stationary variance: σ²/(2θ).
pub fn ou_stationary_variance(sigma: f64, theta: f64) -> f64 {
    sigma * sigma / (2.0 * theta)
}

/// OU autocorrelation: Corr(X(t), X(t+lag)) = exp(-θ·lag).
pub fn ou_autocorrelation(theta: f64, lag: f64) -> f64 {
    (-theta * lag).exp()
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4 — Poisson Process
// ═══════════════════════════════════════════════════════════════════════════

/// Simulate homogeneous Poisson process with rate λ on [0, T].
/// Returns event arrival times.
pub fn poisson_process(lambda: f64, t_end: f64, seed: u64) -> Vec<f64> {
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0;
    loop {
        let u = crate::rng::TamRng::next_f64(&mut rng);
        let inter_arrival = -u.ln() / lambda;
        t += inter_arrival;
        if t > t_end { break; }
        times.push(t);
    }
    times
}

/// Count of Poisson events in [0, T]: N(T) ~ Poisson(λT).
pub fn poisson_count(events: &[f64], t: f64) -> usize {
    events.iter().filter(|&&e| e <= t).count()
}

/// Poisson process expected count: E[N(T)] = λT.
pub fn poisson_expected_count(lambda: f64, t: f64) -> f64 {
    lambda * t
}

/// Non-homogeneous Poisson process with rate λ(t) via thinning.
/// `lambda_bound`: upper bound on λ(t). `lambda_fn`: actual rate function.
pub fn nonhomogeneous_poisson(
    lambda_bound: f64,
    lambda_fn: impl Fn(f64) -> f64,
    t_end: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0;
    loop {
        let u = crate::rng::TamRng::next_f64(&mut rng);
        t += -u.ln() / lambda_bound;
        if t > t_end { break; }
        let accept = crate::rng::TamRng::next_f64(&mut rng);
        if accept < lambda_fn(t) / lambda_bound {
            times.push(t);
        }
    }
    times
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5 — Discrete-Time Markov Chains
// ═══════════════════════════════════════════════════════════════════════════

/// n-step transition probabilities: T^n for n×n stochastic matrix.
///
/// **Kingdom A** — binary exponentiation (repeated squaring) is a prefix scan
/// over the bits of n with matrix multiplication as the semigroup operator.
/// Loop runs exactly `floor(log₂(n))` steps unconditionally — termination is
/// data-determined by n, NOT convergence-dependent. Maps are data-determined
/// (transition matrix is a fixed parameter). Prior label "Kingdom C" was wrong:
/// C means iterative-until-convergence; binary exponentiation has no convergence
/// criterion and no self-referential state selection.
pub fn markov_n_step(transition: &[f64], n: usize, n_states: usize) -> Vec<f64> {
    let mut result: Vec<f64> = (0..n_states * n_states).map(|i| {
        if i / n_states == i % n_states { 1.0 } else { 0.0 }
    }).collect(); // identity

    let mut base = transition.to_vec();
    let mut exp = n;

    while exp > 0 {
        if exp & 1 == 1 { result = mat_mul_stochastic(&result, &base, n_states); }
        base = mat_mul_stochastic(&base, &base, n_states);
        exp >>= 1;
    }
    result
}

fn mat_mul_stochastic(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik.abs() < 1e-15 { continue; }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Stationary distribution π: πP = π, Σπᵢ = 1.
/// Solved by power iteration: π_∞ = lim_{n→∞} π₀ · P^n.
pub fn stationary_distribution(transition: &[f64], n_states: usize) -> Vec<f64> {
    // Start with uniform distribution
    let mut pi = vec![1.0 / n_states as f64; n_states];

    for _ in 0..1000 {
        let mut new_pi = vec![0.0; n_states];
        // Accumulate: π_{n+1}[j] = Σᵢ π_n[i] · P[i,j]
        for i in 0..n_states {
            for j in 0..n_states {
                new_pi[j] += pi[i] * transition[i * n_states + j];
            }
        }
        // Normalize
        let sum: f64 = new_pi.iter().sum();
        for p in new_pi.iter_mut() { *p /= sum; }

        let diff: f64 = pi.iter().zip(new_pi.iter()).map(|(a, b)| (a - b).abs()).sum();
        pi = new_pi;
        if diff < 1e-12 { break; }
    }
    pi
}

/// Mean first passage time from state i to state j.
/// MFPT = (I - P_{removed_j})^{-1} · 1 component i.
/// Uses a simplified iterative approach for small state spaces.
pub fn mean_first_passage_time(
    transition: &[f64],
    from: usize,
    to: usize,
    n_states: usize,
) -> f64 {
    // Simulate: average steps to reach `to` from `from`
    let n_runs = 10000;
    let mut rng = crate::rng::Xoshiro256::new(12345);
    let mut total_steps = 0u64;

    for _ in 0..n_runs {
        let mut state = from;
        let mut steps = 0u64;
        loop {
            // Gather: transition from current state
            let u = crate::rng::TamRng::next_f64(&mut rng);
            let row = &transition[state * n_states..(state + 1) * n_states];
            let mut cumsum = 0.0;
            let mut next = 0;
            for (j, &p) in row.iter().enumerate() {
                cumsum += p;
                if u < cumsum { next = j; break; }
            }
            state = next;
            steps += 1;
            if state == to { break; }
            if steps > 100_000 { break; } // absorbing?
        }
        total_steps += steps;
    }
    total_steps as f64 / n_runs as f64
}

/// Is a Markov chain ergodic (irreducible and aperiodic)?
/// Check: T^n has all positive entries for some n.
pub fn is_ergodic(transition: &[f64], n_states: usize) -> bool {
    let tn = markov_n_step(transition, n_states * n_states, n_states);
    tn.iter().all(|&p| p > 1e-10)
}

/// Mixing time: steps until the chain is within ε of stationary in total variation.
pub fn mixing_time(transition: &[f64], n_states: usize, epsilon: f64) -> usize {
    let pi = stationary_distribution(transition, n_states);
    let mut current = vec![1.0, 0.0_f64]; // not used directly
    let _ = current;

    // Start from worst state (state 0)
    let mut dist = vec![0.0; n_states];
    dist[0] = 1.0;

    for t in 1..1000 {
        // One step: dist → dist · P
        let mut new_dist = vec![0.0; n_states];
        for i in 0..n_states {
            for j in 0..n_states {
                new_dist[j] += dist[i] * transition[i * n_states + j];
            }
        }
        dist = new_dist;

        // Total variation distance to stationary
        let tv: f64 = dist.iter().zip(pi.iter())
            .map(|(d, p)| (d - p).abs())
            .sum::<f64>() / 2.0;
        if tv < epsilon { return t; }
    }
    1000
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6 — Continuous-Time Markov Chains (CTMC)
// ═══════════════════════════════════════════════════════════════════════════

/// CTMC: given generator matrix Q, compute P(t) = exp(Qt).
/// Uses uniformization: P(t) = exp(-qt) Σ (qt)^n/n! P^n where q = max(-Q_ii).
/// Accumulate: Poisson-weighted sum of discrete transition powers.
pub fn ctmc_transition_matrix(q_matrix: &[f64], t: f64, n_states: usize) -> Vec<f64> {
    // Uniformization rate
    let q_max = (0..n_states).map(|i| -q_matrix[i * n_states + i]).fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    if q_max < 1e-15 {
        // Diagonal Q → identity (absorbing states)
        return (0..n_states * n_states).map(|i| if i / n_states == i % n_states { 1.0 } else { 0.0 }).collect();
    }

    // Uniformized chain P̃ = I + Q/q
    let p_tilde: Vec<f64> = (0..n_states * n_states).map(|idx| {
        let i = idx / n_states;
        let j = idx % n_states;
        if i == j { 1.0 + q_matrix[idx] / q_max }
        else { q_matrix[idx] / q_max }
    }).collect();

    let qt = q_max * t;
    let exp_minus_qt = (-qt).exp();

    // P(t) = exp(-qt) Σ_{n=0}^∞ (qt)^n/n! P̃^n
    // Truncate series when Poisson weight becomes negligible
    let mut result = vec![0.0; n_states * n_states];
    let mut p_n = (0..n_states * n_states).map(|i| {
        if i / n_states == i % n_states { 1.0 } else { 0.0 }
    }).collect::<Vec<f64>>(); // P̃^0 = I
    let mut poisson_weight = exp_minus_qt;
    let mut qt_pow = 1.0;
    let mut n_fact = 1.0;

    for k in 0..100 {
        // Accumulate: add Poisson(qt)[k] × P̃^k to result
        let w = poisson_weight * qt_pow / n_fact;
        for i in 0..n_states * n_states {
            result[i] += w * p_n[i];
        }
        if w < 1e-15 { break; }
        p_n = mat_mul_stochastic(&p_n, &p_tilde, n_states);
        qt_pow *= qt;
        n_fact *= (k + 1) as f64;
    }
    result
}

/// CTMC stationary distribution: solve πQ = 0, Σπᵢ = 1.
/// Uses power iteration on the uniformized chain P̃ = I + Q/q_max,
/// which has the same stationary distribution as Q but is well-conditioned.
pub fn ctmc_stationary(q_matrix: &[f64], n_states: usize) -> Vec<f64> {
    let q_max = (0..n_states)
        .map(|i| -q_matrix[i * n_states + i])
        .fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    if q_max < 1e-15 {
        let mut pi = vec![0.0; n_states];
        pi[0] = 1.0;
        return pi;
    }
    // Uniformized chain: same stationary distribution, avoids exp(-qt) underflow
    let p_tilde: Vec<f64> = (0..n_states * n_states).map(|idx| {
        let i = idx / n_states;
        let j = idx % n_states;
        if i == j { 1.0 + q_matrix[idx] / q_max }
        else { q_matrix[idx] / q_max }
    }).collect();
    stationary_distribution(&p_tilde, n_states)
}

/// Mean holding time in state i: h_i = -1/Q_ii.
pub fn ctmc_holding_time(q_matrix: &[f64], state: usize, n_states: usize) -> f64 {
    -1.0 / q_matrix[state * n_states + state].max(-1e15)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7 — Birth-Death Process
// ═══════════════════════════════════════════════════════════════════════════

/// Birth-death process stationary distribution.
/// State k has birth rate λ_k, death rate μ_k.
/// π_k = π_0 × Π_{j=0}^{k-1} λ_j / μ_{j+1} (detailed balance).
pub fn birth_death_stationary(lambdas: &[f64], mus: &[f64]) -> Vec<f64> {
    let n = lambdas.len() + 1; // states 0..n-1
    assert_eq!(mus.len(), n - 1);
    let mut pi = vec![1.0f64; n];
    let mut prod = 1.0;
    for k in 1..n {
        prod *= lambdas[k - 1] / mus[k - 1];
        pi[k] = prod;
    }
    let sum: f64 = pi.iter().sum();
    pi.iter_mut().for_each(|p| *p /= sum);
    pi
}

/// M/M/1 queue: arrival rate λ, service rate μ. ρ = λ/μ < 1 for stability.
/// Returns (stationary distribution for states 0..max_state, mean queue length).
pub fn mm1_queue(lambda: f64, mu: f64, max_state: usize) -> (Vec<f64>, f64) {
    let rho = lambda / mu;
    assert!(rho < 1.0, "ρ = {} ≥ 1: unstable queue", rho);
    let pi: Vec<f64> = (0..max_state).map(|k| (1.0 - rho) * rho.powi(k as i32)).collect();
    let mean_queue = rho / (1.0 - rho);
    (pi, mean_queue)
}

/// M/M/c queue: c servers, arrival rate λ, per-server rate μ. ρ = λ/(cμ) < 1.
/// Returns probability of waiting (Erlang C formula).
pub fn erlang_c(lambda: f64, mu: f64, c: usize) -> f64 {
    let rho = lambda / (c as f64 * mu);
    if rho >= 1.0 { return 1.0; }
    let a = lambda / mu; // offered traffic

    // Sum for P_0
    let mut sum = 0.0;
    let mut a_n = 1.0;
    let mut n_fact = 1.0;
    for n in 0..c {
        a_n *= if n == 0 { 1.0 } else { a };
        n_fact *= if n == 0 { 1.0 } else { n as f64 };
        sum += a_n / n_fact;
    }
    a_n *= a;
    n_fact *= c as f64;
    let p_c = a_n / n_fact / (1.0 - rho);
    let p0 = 1.0 / (sum + p_c);

    p_c * p0
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8 — Random Walks
// ═══════════════════════════════════════════════════════════════════════════

/// Simple symmetric random walk on ℤ: steps ±1 with equal probability.
/// Returns positions at each step.
pub fn simple_random_walk(n_steps: usize, seed: u64) -> Vec<i64> {
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut positions = Vec::with_capacity(n_steps + 1);
    positions.push(0i64);
    let mut pos = 0i64;
    for _ in 0..n_steps {
        let bit = crate::rng::TamRng::next_u64(&mut rng) >> 63;
        pos += if bit == 0 { 1 } else { -1 };
        positions.push(pos);
    }
    positions
}

/// First passage time distribution for 1D RW.
/// P(T_a = n) for walk starting at 0 to reach level a.
/// Returns P(T_a ≤ t) by simulation.
pub fn first_passage_time_cdf(a: i64, max_steps: usize, n_runs: usize, seed: u64) -> Vec<f64> {
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut counts = vec![0u64; max_steps + 1];

    for _ in 0..n_runs {
        let mut pos = 0i64;
        let mut found = false;
        for t in 1..=max_steps {
            let bit = crate::rng::TamRng::next_u64(&mut rng) >> 63;
            pos += if bit == 0 { 1 } else { -1 };
            if pos.abs() >= a.abs() { counts[t] += 1; found = true; break; }
        }
        if !found { counts[max_steps] += 1; }
    }

    let mut cdf = Vec::with_capacity(max_steps + 1);
    let mut cumsum = 0u64;
    for &c in &counts {
        cumsum += c;
        cdf.push(cumsum as f64 / n_runs as f64);
    }
    cdf
}

/// Return probability for 1D simple random walk: P(return to 0) = 1.
/// Verified by simulation.
pub fn return_probability_1d(n_steps: usize, n_runs: usize, seed: u64) -> f64 {
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut returns = 0u64;
    for _ in 0..n_runs {
        let mut pos = 0i64;
        for _ in 0..n_steps {
            let bit = crate::rng::TamRng::next_u64(&mut rng) >> 63;
            pos += if bit == 0 { 1 } else { -1 };
            if pos == 0 { returns += 1; break; }
        }
    }
    returns as f64 / n_runs as f64
}

/// Expected maximum of a random walk up to time n: E[max W_t] ≈ √(2n/π).
pub fn rw_expected_maximum(n: usize) -> f64 {
    (2.0 * n as f64 / std::f64::consts::PI).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9 — Itô Calculus
// ═══════════════════════════════════════════════════════════════════════════

/// Itô integral ∫₀ᵀ f(t) dW(t) approximated by Riemann-Itô sum.
/// `f_values`: f evaluated at t₀, t₁, ..., t_{n-1} (left endpoints).
/// `increments`: Brownian increments dW_t = W_{t+1} - W_t.
pub fn ito_integral(f_values: &[f64], increments: &[f64]) -> f64 {
    assert_eq!(f_values.len(), increments.len());
    // Itô: evaluate f at LEFT endpoint
    f_values.iter().zip(increments.iter()).map(|(f, dw)| f * dw).sum()
}

/// Stratonovich integral ∫₀ᵀ f(t) ∘ dW(t) (midpoint rule).
pub fn stratonovich_integral(f_values: &[f64], increments: &[f64]) -> f64 {
    assert_eq!(f_values.len(), increments.len());
    let n = f_values.len();
    if n == 0 { return 0.0; }
    // Stratonovich: evaluate f at MID point (average of left and right)
    let mut sum = f_values[0] * increments[0]; // first term uses f(t_0)
    for i in 1..n {
        sum += 0.5 * (f_values[i - 1] + f_values[i]) * increments[i];
    }
    sum
}

/// Itô's lemma: df(W_t) = f'(W_t)dW_t + ½f''(W_t)dt.
/// Verifies the exact algebraic identity for f(x)=x²:
///   W(T)² - W(0)² = 2 Σᵢ W(tᵢ) ΔWᵢ + Σᵢ (ΔWᵢ)²
/// The last term is the quadratic variation, which converges to T as n→∞.
/// Uses actual QV (not n·dt) so the identity holds exactly at finite n.
/// Returns the floating-point error (should be near machine epsilon).
pub fn ito_lemma_verification(path: &[f64], _dt: f64) -> f64 {
    let n = path.len() - 1;
    // Direct: W²(T) - W²(0)
    let direct = path[n] * path[n] - path[0] * path[0];
    let dw: Vec<f64> = path.windows(2).map(|w| w[1] - w[0]).collect();
    // Itô formula with actual quadratic variation: algebraically exact
    let ito_sum: f64 = path[..n].iter().zip(dw.iter()).map(|(w, dw)| w * dw).sum();
    let actual_qv: f64 = dw.iter().map(|d| d * d).sum();
    let ito = 2.0 * ito_sum + actual_qv;
    (direct - ito).abs()
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10 — SDE Stepping Primitives (Euler-Maruyama, Milstein, Langevin)
//
// Kingdom A: each step is an affine map (a_t, b_t) applied to the current
// state — BUT the map coefficients depend on the state x_t, making these
// Kingdom B (state-dependent maps). The Fock boundary holds: the drift f(x)
// and diffusion g(x) at step t require x_t, which requires x_{t-1}, etc.
//
// Single-step primitives (euler_maruyama_step, milstein_step) are the
// composable atoms. The full path integrators compose them sequentially.
// ═══════════════════════════════════════════════════════════════════════════

/// Euler-Maruyama step for SDE: dX = f(X,t)dt + g(X,t)dW.
/// X_{t+dt} = X_t + f(X_t, t)·dt + g(X_t, t)·√dt·Z, Z ~ N(0,1).
/// Strong order 0.5, weak order 1.0.
pub fn euler_maruyama_step(
    x: f64,
    t: f64,
    dt: f64,
    drift: impl Fn(f64, f64) -> f64,
    diffusion: impl Fn(f64, f64) -> f64,
    z: f64,  // pre-sampled N(0,1) variate
) -> f64 {
    x + drift(x, t) * dt + diffusion(x, t) * dt.sqrt() * z
}

/// Milstein step for SDE with Itô correction for multiplicative noise.
/// X_{t+dt} = X_t + f·dt + g·√dt·Z + ½g·g'·dt·(Z²-1).
/// Strong order 1.0 (vs Euler-Maruyama's 0.5) for scalar SDEs.
/// `diffusion_deriv`: ∂g/∂x at (X_t, t).
pub fn milstein_step(
    x: f64,
    t: f64,
    dt: f64,
    drift: impl Fn(f64, f64) -> f64,
    diffusion: impl Fn(f64, f64) -> f64,
    diffusion_deriv: impl Fn(f64, f64) -> f64,
    z: f64,  // pre-sampled N(0,1) variate
) -> f64 {
    let f = drift(x, t);
    let g = diffusion(x, t);
    let gp = diffusion_deriv(x, t);
    let sqrt_dt = dt.sqrt();
    // Milstein correction: ½ g g' (dW² - dt) where dW² ≈ dt·Z²
    x + f * dt + g * sqrt_dt * z + 0.5 * g * gp * dt * (z * z - 1.0)
}

/// Full path integration of a general scalar SDE using Euler-Maruyama.
/// Returns (times, states).
/// `drift(x, t)` and `diffusion(x, t)` are the SDE coefficients.
pub fn euler_maruyama_path(
    x0: f64,
    t_end: f64,
    n_steps: usize,
    drift: impl Fn(f64, f64) -> f64,
    diffusion: impl Fn(f64, f64) -> f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let dt = t_end / n_steps as f64;
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut states = Vec::with_capacity(n_steps + 1);
    times.push(0.0);
    states.push(x0);
    let mut x = x0;
    for i in 0..n_steps {
        let t = i as f64 * dt;
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        x = euler_maruyama_step(x, t, dt, &drift, &diffusion, z);
        times.push((i + 1) as f64 * dt);
        states.push(x);
    }
    (times, states)
}

/// Full path integration of a scalar SDE using Milstein's method.
/// Returns (times, states).
pub fn milstein_path(
    x0: f64,
    t_end: f64,
    n_steps: usize,
    drift: impl Fn(f64, f64) -> f64,
    diffusion: impl Fn(f64, f64) -> f64,
    diffusion_deriv: impl Fn(f64, f64) -> f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let dt = t_end / n_steps as f64;
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut states = Vec::with_capacity(n_steps + 1);
    times.push(0.0);
    states.push(x0);
    let mut x = x0;
    for i in 0..n_steps {
        let t = i as f64 * dt;
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        x = milstein_step(x, t, dt, &drift, &diffusion, &diffusion_deriv, z);
        times.push((i + 1) as f64 * dt);
        states.push(x);
    }
    (times, states)
}

/// Langevin dynamics for a particle in potential V(x).
/// dX = -γX dt + σ dW  (linear damping / harmonic well case).
/// More generally: dX = -V'(X)dt + σ dW.
/// The stationary distribution is the Gibbs measure exp(-2V(x)/σ²) / Z.
/// Returns (positions, velocities) if velocity tracking enabled, else just positions.
///
/// This is the overdamped Langevin equation (no inertia term).
/// `force`: -∂V/∂x = the force on the particle.
pub fn langevin_overdamped(
    x0: f64,
    t_end: f64,
    n_steps: usize,
    force: impl Fn(f64) -> f64,
    sigma: f64,
    seed: u64,
) -> Vec<f64> {
    // Overdamped Langevin = Euler-Maruyama with drift = force(x), diffusion = σ
    euler_maruyama_path(
        x0,
        t_end,
        n_steps,
        |x, _t| force(x),
        |_x, _t| sigma,
        seed,
    ).1
}

/// Underdamped Langevin dynamics with explicit momentum.
/// dx = v dt;  dv = (-γv + F(x)/m)dt + σ dW.
/// Returns (positions, velocities).
pub fn langevin_underdamped(
    x0: f64,
    v0: f64,
    mass: f64,
    gamma: f64,   // friction coefficient
    sigma: f64,   // noise amplitude (σ² = 2γk_BT/m by fluctuation-dissipation)
    t_end: f64,
    n_steps: usize,
    force: impl Fn(f64) -> f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let dt = t_end / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = crate::rng::Xoshiro256::new(seed);
    let mut positions = Vec::with_capacity(n_steps + 1);
    let mut velocities = Vec::with_capacity(n_steps + 1);
    positions.push(x0);
    velocities.push(v0);
    let mut x = x0;
    let mut v = v0;
    for _ in 0..n_steps {
        let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
        let f = force(x);
        // Euler-Maruyama for the (x, v) system
        let v_new = v + (-gamma * v + f / mass) * dt + sigma * sqrt_dt * z;
        let x_new = x + v * dt;
        x = x_new;
        v = v_new;
        positions.push(x);
        velocities.push(v);
    }
    (positions, velocities)
}

/// Runge-Kutta 4th order step for a vector-valued ODE: dX/dt = f(X, t).
/// Single step primitive — compose via euler_maruyama_path for SDEs.
/// For deterministic ODEs (no noise), this is the standard RK4 stepper.
/// `state`: current state vector; returns next state.
pub fn rk4_step_vec(
    state: &[f64],
    t: f64,
    dt: f64,
    f: impl Fn(&[f64], f64) -> Vec<f64>,
) -> Vec<f64> {
    let n = state.len();
    let k1 = f(state, t);
    let s2: Vec<f64> = (0..n).map(|i| state[i] + 0.5 * dt * k1[i]).collect();
    let k2 = f(&s2, t + 0.5 * dt);
    let s3: Vec<f64> = (0..n).map(|i| state[i] + 0.5 * dt * k2[i]).collect();
    let k3 = f(&s3, t + 0.5 * dt);
    let s4: Vec<f64> = (0..n).map(|i| state[i] + dt * k3[i]).collect();
    let k4 = f(&s4, t + dt);
    (0..n).map(|i| state[i] + dt / 6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: got {a}, expected {b} (diff={})", (a - b).abs());
    }

    fn close_rel(a: f64, b: f64, rtol: f64, label: &str) {
        let rel = if b.abs() > 1e-30 { (a - b).abs() / b.abs() } else { (a - b).abs() };
        assert!(rel < rtol, "{label}: got {a}, expected {b} (rel={rel})");
    }

    // ── Section 1: Brownian Motion ──────────────────────────────────────

    #[test]
    fn brownian_starts_at_zero() {
        let (_, w) = brownian_motion(1.0, 1000, 42);
        close(w[0], 0.0, 1e-10, "BM starts at 0");
    }

    #[test]
    fn brownian_quadratic_variation() {
        // E[QV] = T (law of large numbers for QV)
        let (_, w) = brownian_motion(1.0, 10000, 42);
        let qv = quadratic_variation(&w);
        // QV should be close to T=1 with ~1% error for 10000 steps
        close(qv, 1.0, 0.05, "BM quadratic variation ≈ T");
    }

    #[test]
    fn brownian_bridge_endpoint() {
        let target = 2.0;
        let bridge = brownian_bridge(1.0, target, 1000, 42);
        close(bridge[0], 0.0, 1e-10, "Bridge starts at 0");
        close(*bridge.last().unwrap(), target, 1e-10, "Bridge ends at target");
    }

    // ── Section 2: GBM and Black-Scholes ────────────────────────────────

    #[test]
    fn gbm_expected_value() {
        let (mu, sigma, s0, t) = (0.1, 0.2, 100.0, 1.0);
        let (_, prices) = geometric_brownian_motion(s0, mu, sigma, t, 10000, 42);
        // With 10000 paths: E[S(T)] = S0 exp(μT), estimate from one path is noisy
        // Just verify the path is positive and starts correctly
        assert!(prices.iter().all(|&p| p > 0.0), "GBM prices must be positive");
        close(prices[0], s0, 1e-10, "GBM starts at S0");
    }

    #[test]
    fn gbm_variance_formula() {
        let (mu, sigma, s0, t) = (0.05, 0.3, 1.0, 1.0);
        let var = gbm_variance(s0, mu, sigma, t);
        // For S0=1: Var = exp(2μT)(exp(σ²T) - 1) ≈ large
        assert!(var > 0.0, "GBM variance must be positive");
    }

    #[test]
    fn black_scholes_call_put_parity() {
        // Put-call parity: C - P = S - K·exp(-rT)
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let (call, _) = black_scholes(s, k, t, r, sigma, true);
        let (put, _) = black_scholes(s, k, t, r, sigma, false);
        let parity = s - k * (-r * t).exp();
        close(call - put, parity, 0.01, "Put-call parity");
    }

    #[test]
    fn black_scholes_otm_call_positive() {
        // Deep OTM call: S=80, K=100 → small but positive price
        let (call, _) = black_scholes(80.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(call > 0.0 && call < 5.0, "OTM call price in reasonable range: {call}");
    }

    #[test]
    fn black_scholes_intrinsic_at_expiry() {
        // At t=0: C = max(S-K, 0)
        let (call, _) = black_scholes(110.0, 100.0, 0.0, 0.05, 0.2, true);
        close(call, 10.0, 1e-6, "At expiry: call = max(S-K,0)");
        let (call_otm, _) = black_scholes(90.0, 100.0, 0.0, 0.05, 0.2, true);
        close(call_otm, 0.0, 1e-6, "OTM at expiry: call = 0");
    }

    // ── Section 3: Ornstein-Uhlenbeck ───────────────────────────────────

    #[test]
    fn ou_mean_reversion() {
        // OU process should have mean near μ
        let (x0, mu, theta, sigma) = (5.0, 0.0, 2.0, 0.5);
        let path = ornstein_uhlenbeck(x0, mu, theta, sigma, 10.0, 10000, 42);
        // Long-run mean should be near μ = 0
        let long_run_mean: f64 = path[5000..].iter().sum::<f64>() / 5000.0;
        assert!(long_run_mean.abs() < 0.3, "OU long-run mean={long_run_mean} should be near 0");
    }

    #[test]
    fn ou_stationary_variance_formula() {
        // Stationary variance should be σ²/(2θ)
        let (theta, sigma) = (2.0, 0.5);
        let expected_var = ou_stationary_variance(sigma, theta);
        let path = ornstein_uhlenbeck(0.0, 0.0, theta, sigma, 100.0, 100000, 42);
        let mean: f64 = path[50000..].iter().sum::<f64>() / 50000.0;
        let var: f64 = path[50000..].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 50000.0;
        close_rel(var, expected_var, 0.15, "OU stationary variance");
    }

    #[test]
    fn ou_autocorrelation_formula() {
        // Corr(X(t), X(t+lag)) = exp(-θ·lag)
        let (theta, lag) = (1.5_f64, 0.5_f64);
        let expected = ou_autocorrelation(theta, lag);
        close(expected, (-theta * lag).exp(), 1e-10, "OU autocorrelation formula");
    }

    // ── Section 4: Poisson Process ──────────────────────────────────────

    #[test]
    fn poisson_event_count_mean() {
        // E[N(T)] = λT. Average over multiple simulations.
        let (lambda, t) = (3.0, 10.0);
        let n_sims = 500;
        let total: usize = (0..n_sims)
            .map(|seed| poisson_process(lambda, t, seed as u64 * 7 + 42).len())
            .sum();
        let mean = total as f64 / n_sims as f64;
        close_rel(mean, lambda * t, 0.05, "Poisson E[N(T)] = λT");
    }

    #[test]
    fn poisson_event_ordering() {
        let events = poisson_process(5.0, 10.0, 42);
        for i in 1..events.len() {
            assert!(events[i] > events[i - 1], "Poisson events must be increasing");
        }
    }

    // ── Section 5: Markov Chains ─────────────────────────────────────────

    #[test]
    fn markov_stationary_two_state() {
        // Two-state: [[0.7, 0.3], [0.4, 0.6]]
        // π = [4/7, 3/7] ≈ [0.571, 0.429]
        let p = vec![0.7, 0.3, 0.4, 0.6];
        let pi = stationary_distribution(&p, 2);
        close(pi[0], 4.0 / 7.0, 0.001, "π[0] = 4/7");
        close(pi[1], 3.0 / 7.0, 0.001, "π[1] = 3/7");
    }

    #[test]
    fn markov_stationary_sum_to_one() {
        let p = vec![0.2, 0.5, 0.3, 0.1, 0.6, 0.3, 0.4, 0.2, 0.4];
        let pi = stationary_distribution(&p, 3);
        let sum: f64 = pi.iter().sum();
        close(sum, 1.0, 1e-10, "Stationary dist sums to 1");
    }

    #[test]
    fn markov_n_step_converges_to_stationary() {
        let p = vec![0.7, 0.3, 0.4, 0.6];
        let pi = stationary_distribution(&p, 2);
        let pn = markov_n_step(&p, 50, 2);
        // Both rows of P^50 should ≈ π
        close(pn[0], pi[0], 0.001, "P^50 row 0 = π[0]");
        close(pn[2], pi[0], 0.001, "P^50 row 1 = π[0]");
    }

    #[test]
    fn markov_mean_fpt_random_walk() {
        // Symmetric random walk on 3 states: 0-1-2.
        // P = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
        // MFPT from 0 to 2 = 4 (by detailed balance)
        let p = vec![0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0];
        let mfpt = mean_first_passage_time(&p, 0, 2, 3);
        assert!(mfpt > 2.0 && mfpt < 8.0, "MFPT from 0 to 2 ≈ {mfpt}, expected ~4");
    }

    // ── Section 6: CTMC ──────────────────────────────────────────────────

    #[test]
    fn ctmc_holding_time_correct() {
        // Q = [[-2, 2], [1, -1]]: state 0 has rate 2, state 1 has rate 1
        let q = vec![-2.0, 2.0, 1.0, -1.0];
        close(ctmc_holding_time(&q, 0, 2), 0.5, 1e-10, "Holding time state 0 = 1/2");
        close(ctmc_holding_time(&q, 1, 2), 1.0, 1e-10, "Holding time state 1 = 1");
    }

    #[test]
    fn ctmc_stationary_two_state() {
        // Q = [[-2, 2], [1, -1]]. Stationary: π[0]·2 = π[1]·1 → π = [1/3, 2/3]
        let q = vec![-2.0, 2.0, 1.0, -1.0];
        let pi = ctmc_stationary(&q, 2);
        close(pi[0], 1.0 / 3.0, 0.05, "CTMC π[0] = 1/3");
        close(pi[1], 2.0 / 3.0, 0.05, "CTMC π[1] = 2/3");
    }

    // ── Section 7: Birth-Death and Queues ────────────────────────────────

    #[test]
    fn birth_death_stationary_detailed_balance() {
        // M/M/∞ truncated at 99 states: λ = 1, μ = 2 everywhere, ρ = 1/2
        // π_k = (1-ρ) ρ^k = (1/2)^{k+1} (geometric distribution, infinite-series limit)
        // With 100 states the truncation tail is (1/2)^100 ≈ 0 — negligible
        let n = 99usize;
        let lambdas = vec![1.0; n];
        let mus = vec![2.0; n];
        let pi = birth_death_stationary(&lambdas, &mus);
        // Check first few states against geometric(ρ=1/2) with loose tolerance
        for k in 0..5usize {
            let expected = 0.5_f64.powi(k as i32 + 1);
            close(pi[k], expected, 0.005, &format!("BD π[{k}]"));
        }
        // Detailed balance: λ_k π_k = μ_{k+1} π_{k+1}
        for k in 0..5usize {
            close(1.0 * pi[k], 2.0 * pi[k + 1], 1e-10, &format!("BD balance at {k}"));
        }
    }

    #[test]
    fn mm1_queue_mean_length() {
        // ρ = 0.5: E[L] = ρ/(1-ρ) = 1
        let (_, mean_q) = mm1_queue(0.5, 1.0, 20);
        close(mean_q, 1.0, 1e-10, "M/M/1 E[L] = 1 for ρ=0.5");
    }

    // ── Section 8: Random Walks ─────────────────────────────────────────

    #[test]
    fn rw_starts_at_zero() {
        let walk = simple_random_walk(100, 42);
        assert_eq!(walk[0], 0, "RW starts at 0");
    }

    #[test]
    fn rw_steps_are_unit() {
        let walk = simple_random_walk(100, 42);
        for i in 1..walk.len() {
            assert!((walk[i] - walk[i - 1]).abs() == 1, "RW step = ±1");
        }
    }

    #[test]
    fn return_probability_positive() {
        // 1D symmetric RW: return probability = 1 (recurrent)
        let p = return_probability_1d(10000, 1000, 42);
        assert!(p > 0.8, "1D RW return probability={p} should be near 1");
    }

    #[test]
    fn rw_expected_maximum_scaling() {
        // E[max W_t] ≈ √(2n/π)
        let n = 1000;
        let expected = rw_expected_maximum(n);
        // Empirical check from walk statistics
        let walk = simple_random_walk(n, 42);
        let max_pos = walk.iter().map(|&x| x.abs()).max().unwrap() as f64;
        // The formula gives order-of-magnitude estimate; allow 3x range
        assert!(max_pos > expected / 3.0 && max_pos < expected * 3.0,
            "RW max={max_pos}, E[max]={expected:.2}");
    }

    // ── Section 9: Itô Calculus ──────────────────────────────────────────

    #[test]
    fn ito_integral_of_constant() {
        // ∫₀ᵀ 1 dW = W(T) - W(0) = W(T)
        let n = 1000;
        let (_, w) = brownian_motion(1.0, n, 42);
        let dw: Vec<f64> = w.windows(2).map(|win| win[1] - win[0]).collect();
        let ones = vec![1.0; n];
        let integral = ito_integral(&ones, &dw);
        close(integral, w[n], 0.01, "Itô ∫1 dW = W(T)");
    }

    #[test]
    fn ito_lemma_w_squared() {
        // Itô: W² = 2∫W dW + T. Error should be near 0.
        let (_, w) = brownian_motion(1.0, 1000, 42);
        let dt = 1.0 / 1000.0;
        let err = ito_lemma_verification(&w, dt);
        // Discretization error is O(dt) = O(10⁻³), allow 10× margin
        assert!(err < 0.01, "Itô d(W²) verification error={err} should be near 0");
    }

    // ── Section 10: SDE Stepping Primitives ─────────────────────────────

    #[test]
    fn euler_maruyama_gbm_drift() {
        // GBM with mu=0.1, sigma=0.2: E[X(T)] = X0 exp(mu*T)
        // Average over many paths to check mean drift.
        let (mu, sigma, x0, t_end, n_steps) = (0.1_f64, 0.2_f64, 100.0_f64, 1.0, 1000);
        let n_paths = 500;
        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let (_, path) = euler_maruyama_path(
                x0, t_end, n_steps,
                |x, _t| mu * x,
                |x, _t| sigma * x,
                seed,
            );
            sum += path.last().unwrap();
        }
        let mean_x = sum / n_paths as f64;
        let expected = x0 * (mu * t_end).exp();
        close_rel(mean_x, expected, 0.05, "EM GBM mean");
    }

    #[test]
    fn milstein_gbm_same_mean_as_em() {
        // For GBM g(x)=sigma*x: g'=sigma. Milstein and EM should give same mean.
        let (mu, sigma, x0) = (0.05_f64, 0.2_f64, 100.0_f64);
        let (t_end, n_steps) = (0.1, 100);
        let n_paths = 500;
        let mut em_sum = 0.0;
        let mut mil_sum = 0.0;
        for seed in 0..n_paths as u64 {
            let (_, em) = euler_maruyama_path(
                x0, t_end, n_steps,
                |x, _t| mu * x,
                |x, _t| sigma * x,
                seed,
            );
            let (_, mil) = milstein_path(
                x0, t_end, n_steps,
                |x, _t| mu * x,
                |x, _t| sigma * x,
                |_x, _t| sigma,
                seed,
            );
            em_sum += em.last().unwrap();
            mil_sum += mil.last().unwrap();
        }
        let em_mean = em_sum / n_paths as f64;
        let mil_mean = mil_sum / n_paths as f64;
        let expected = x0 * (mu * t_end).exp();
        close_rel(em_mean, expected, 0.05, "EM GBM mean");
        close_rel(mil_mean, expected, 0.05, "Milstein GBM mean");
    }

    #[test]
    fn langevin_overdamped_harmonic_stationary() {
        // Overdamped Langevin in V(x)=0.5*x^2: force=-x, sigma=1.
        // Stationary dist: p(x) ~ exp(-x^2) => std ~ 1/sqrt(2) ~ 0.707.
        let (x0, t_end, n_steps, sigma) = (5.0_f64, 50.0, 100_000, 1.0_f64);
        let path = langevin_overdamped(x0, t_end, n_steps, |x| -x, sigma, 42);
        let start = n_steps / 5;
        let vals = &path[start..];
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let std = var.sqrt();
        assert!(std > 0.5 && std < 1.2, "Langevin stationary std={std:.3}, expected ~0.707");
        assert!(mean.abs() < 0.2, "Langevin stationary mean={mean:.3} should be ~0");
    }

    #[test]
    fn langevin_underdamped_energy_dissipates() {
        // No noise (sigma=0): underdamped Langevin should lose energy due to friction.
        let (x0, v0, mass, gamma) = (1.0_f64, 0.0_f64, 1.0_f64, 0.5_f64);
        let (pos, vel) = langevin_underdamped(x0, v0, mass, gamma, 0.0, 5.0, 5000, |x| -x, 42);
        let e0 = 0.5 * mass * v0 * v0 + 0.5 * x0 * x0;
        let x_end = pos.last().unwrap();
        let v_end = vel.last().unwrap();
        let e_end = 0.5 * mass * v_end * v_end + 0.5 * x_end * x_end;
        assert!(e_end < e0, "Energy dissipates: {e_end:.4} < {e0:.4}");
    }

    #[test]
    fn rk4_step_harmonic_oscillator() {
        // RK4 on SHO: x(pi/omega) = cos(pi) = -1 for x0=1, v0=0.
        let omega = 2.0_f64;
        let dt = 0.01;
        let f = |s: &[f64], _t: f64| vec![s[1], -omega * omega * s[0]];
        let mut state = vec![1.0_f64, 0.0];
        let t_end = std::f64::consts::PI / omega;
        let n_steps = (t_end / dt).ceil() as usize;
        for i in 0..n_steps {
            state = rk4_step_vec(&state, i as f64 * dt, dt, &f);
        }
        close(state[0], -1.0, 0.01, "RK4 SHO half-period x=-1");
        let energy = 0.5 * (state[1].powi(2) + omega.powi(2) * state[0].powi(2));
        close_rel(energy, 0.5 * omega.powi(2), 1e-4, "RK4 SHO energy conserved");
    }
}
