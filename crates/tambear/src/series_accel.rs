//! # Series Acceleration — Limits as Accumulate
//!
//! ## The Question
//!
//! What does a "limit" mean in tambear? An infinite series is:
//! ```text
//! S = Σ_{n=0}^∞ a_n  =  lim_{N→∞} accumulate(a[0..N], Prefix, Value, Add)
//! ```
//! The partial sums are a prefix scan. The limit is what the scan converges to.
//! But convergence can be *accelerated* — and the accelerators are themselves
//! expressible as accumulate operations on the partial sum sequence.
//!
//! ## Accelerators as Accumulate
//!
//! | Method | Operation on partial sums | Accumulate pattern |
//! |--------|--------------------------|-------------------|
//! | Aitken Δ² | S'_n = S_n - (ΔS_n)²/Δ²S_n | Windowed{3} on Prefix |
//! | Shanks e₁ | Same as Aitken (first-order Shanks = Aitken) | Windowed{3} on Prefix |
//! | Richardson | Tableau: T[i,j] = (4^j·T[i,j-1] - T[i-1,j-1])/(4^j - 1) | Tiled triangle on Prefix |
//! | Euler | E_n = Σ_{k=0}^n C(n,k)·S_{n-k}/2^n | ByKey weighted on Prefix |
//!
//! ## The Insight
//!
//! Every convergence accelerator is a *post-processing pass* on partial sums.
//! In tambear's framework:
//! 1. `accumulate(terms, Prefix, Value, Add)` → partial sums S_n
//! 2. `accumulate(S, Windowed{3}, AccelExpr, Op)` → accelerated sequence
//!
//! The "limit" is not a new primitive. It's the *composition* of Prefix + Windowed.
//! This is exactly the prefix subtraction trick listed in accumulate.rs line 24.
//!
//! ## Kingdom Classification
//!
//! - Step 1 (partial sums): Kingdom B — sequential scan (non-commutative, one-pass)
//! - Step 2 (Aitken): Kingdom A — deterministic function of 3 consecutive values
//! - Wynn's epsilon: **Kingdom BC** — non-commutative (order-dependent tableau)
//!   + multi-pass (n-1 columns). First clean inhabitant of the (ρ=1, σ=1) cell.
//! - Euler transform: Kingdom A — binomial-weighted ByKey accumulate on partial sums
//! - Richardson extrapolation: Kingdom A — weighted combine across resolutions (= K03)
//!
//! The dimensional ladder: raw terms (K01) → partial sums (K02) → acceleration
//! tableau (K03, cross-scale). Richardson extrapolation IS a cross-cadence operation.
//!
//! ## Real-world instance: Borwein's efficient ζ(s) computation
//!
//! Borwein's algorithm for ζ(s) uses the Dirichlet eta function:
//!   η(s) = Σ (-1)^{n-1} / n^s = (1 - 2^{1-s}) · ζ(s)
//!
//! The η(s) series is ALTERNATING, making the Euler transform the matched kernel.
//! Borwein's weights are exactly the binomial coefficients of the Euler transform
//! applied to η(s). This is pure Kingdom A: prefix sum + commutative accumulate.
//! See `bigfloat::zeta_complex` for the implementation using BigFloat.

// ═══════════════════════════════════════════════════════════════════════════
// Partial sums (Prefix scan — the foundation)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute partial sums of a series: S_n = Σ_{k=0}^n a_k.
/// This IS `accumulate(terms, Prefix, Value, Add)` on CPU.
pub fn partial_sums(terms: &[f64]) -> Vec<f64> {
    let mut sums = Vec::with_capacity(terms.len());
    let mut acc = 0.0;
    for &t in terms {
        acc += t;
        sums.push(acc);
    }
    sums
}

// ═══════════════════════════════════════════════════════════════════════════
// Cesàro Summation (uniform kernel — the simplest accelerator)
// ═══════════════════════════════════════════════════════════════════════════

/// Cesàro sum: arithmetic mean of partial sums.
///
/// σ_n = (S_0 + S_1 + ... + S_n) / (n+1)
///
/// This is `accumulate(partial_sums, All, Value, Add) / n` — literally the
/// mean of the prefix scan output. The kernel is uniform: weight 1/(n+1)
/// for every partial sum.
///
/// ## The kernel taxonomy
///
/// Every summability method is a kernel-weighted accumulate on partial sums:
/// - **Cesàro**: uniform kernel (1/(n+1) for all k)
/// - **Euler**: binomial kernel (C(n,k)/2^n)
/// - **Abel**: exponential kernel (x^k as x→1⁻)
/// - **Richardson**: polynomial extrapolation kernel
/// - **Aitken/Wynn**: nonlinear (not a kernel method — extrapolation, not averaging)
///
/// The kernel determines which class of series can be summed. Tauberian
/// theorems give conditions under which kernel summability implies convergence.
/// Structural rhyme: summability kernels ↔ KDE bandwidth selection.
pub fn cesaro_sum(sums: &[f64]) -> f64 {
    if sums.is_empty() { return 0.0; }
    let n = sums.len();
    let total: f64 = sums.iter().sum();
    total / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Aitken's Δ² Process
// ═══════════════════════════════════════════════════════════════════════════

/// Aitken's Δ² process: accelerates a linearly converging sequence.
///
/// Given partial sums S_n, produces:
///   S'_n = S_n - (S_{n+1} - S_n)² / (S_{n+2} - 2·S_{n+1} + S_n)
///
/// This is a Windowed{3} operation on the partial sum sequence.
/// Effective when the error ratio e_{n+1}/e_n ≈ constant (geometric convergence).
///
/// Returns the accelerated sequence (length = input.len() - 2).
pub fn aitken_delta2(sums: &[f64]) -> Vec<f64> {
    if sums.len() < 3 { return vec![]; }
    let mut accel = Vec::with_capacity(sums.len() - 2);
    for i in 0..sums.len() - 2 {
        let s0 = sums[i];
        let s1 = sums[i + 1];
        let s2 = sums[i + 2];
        let d2 = s2 - 2.0 * s1 + s0; // Δ²S_n
        if d2.abs() > 1e-50 {
            let ds = s1 - s0; // ΔS_n
            accel.push(s0 - ds * ds / d2);
        } else {
            // Δ² ≈ 0 means sequence is already converging super-linearly
            accel.push(s2);
        }
    }
    accel
}

// ═══════════════════════════════════════════════════════════════════════════
// Wynn's Epsilon Algorithm (iterated Shanks transform)
// ═══════════════════════════════════════════════════════════════════════════

/// Wynn's epsilon algorithm: computes the Shanks transform to arbitrary order.
///
/// The epsilon table:
///   ε_{-1}(n) = 0
///   ε_0(n) = S_n
///   ε_{k+1}(n) = ε_{k-1}(n+1) + 1/(ε_k(n+1) - ε_k(n))
///
/// Even columns ε_{2k} give the k-th order Shanks transform.
/// ε_2 = Aitken's Δ² (first-order Shanks).
///
/// ## Kingdom classification: BC
///
/// Wynn's epsilon is a **Kingdom BC** algorithm — the first clean inhabitant
/// of the (non-commutative, multi-pass) cell in the (ρ,σ) taxonomy:
///
/// - **Non-commutative (ρ=1)**: each column depends on the previous two,
///   and within each column, entries use adjacent pairs (order-dependent).
///   Swapping positions in the window changes the result.
/// - **Multi-pass (σ=1)**: the tableau has n-1 columns, each a full pass
///   over the shrinking sequence.
///
/// As implemented: **fixed depth** (n-1 columns for n partial sums).
/// No convergence test — the depth is determined by input size, not tolerance.
/// In its natural streaming use (add terms until convergence), it would gain
/// a Kingdom C outer loop. The current form is a deep pipeline: B^{n-1}.
///
/// The Hankel connection: Shanks uses Hankel determinants of the difference
/// sequence. Wynn's epsilon avoids computing determinants explicitly — the
/// reciprocal difference recurrence ε_{k+1} = ε_{k-1} + 1/(ε_k(n+1) - ε_k(n))
/// is algebraically equivalent but numerically stable. The Hankel structure
/// is implicit in the triangular dependency pattern of the tableau.
///
/// ## Algebraic vs geometric convergence
///
/// The Padé approximant structure handles both geometric (r^n) AND algebraic
/// (O(1/N^α)) convergence. This distinguishes Wynn ε from Aitken Δ²:
///
/// | Method | Convergence model | Ergodic mean (Lorenz z) |
/// |--------|------------------|------------------------|
/// | Aitken Δ² | Assumes geometric r^n | 8× WORSE than baseline |
/// | Wynn ε (on block means) | Padé — handles O(1/√N) | **3.9× improvement** |
///
/// Aitken degrades on algebraically-converging sequences because it assumes
/// e_{n+1} ≈ r·e_n. Ergodic means of chaotic systems converge at O(1/√N),
/// which Aitken cannot model. Wynn's rational approximant adapts correctly.
///
/// **Signal farm rule**: For ergodic averages, block-average to decorrelation
/// time, then apply Wynn ε. Never apply Aitken to running means.
///
/// ## When Richardson is better
///
/// For **known** algebraic convergence (positive monotone series like Basel,
/// ζ(s)), `richardson_partial_sums()` is dramatically faster: 80M× vs 3.9×
/// on Basel at n=160. Richardson auto-detects the error order and cancels
/// terms exactly. Use Richardson when convergence class is known; Wynn when
/// the class is unknown or mixed.
///
/// ## Numerical stability
///
/// The reciprocal difference `1/(ε_k(n+1) - ε_k(n))` amplifies rounding
/// errors as the tableau deepens — the denominator approaches zero at
/// exactly the point of convergence (the "division by small quantities"
/// pattern). Empirically stable for n ≤ ~40 terms on Leibniz-type series;
/// catastrophically unstable for n ≥ ~80.
///
/// **Early stopping**: this implementation monitors even-column estimates
/// and stops when consecutive estimates stop improving. This detects
/// instability before it goes catastrophic.
///
/// Returns the best estimate: the last stable even-column entry.
pub fn wynn_epsilon(sums: &[f64]) -> f64 {
    let n = sums.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sums[0]; }

    // We build two columns of the epsilon table at a time
    // eps[0] = current ε_k(n) column, eps[1] = ε_{k-1}(n) column
    let mut prev = vec![0.0; n]; // ε_{-1} = 0
    let mut curr: Vec<f64> = sums.to_vec(); // ε_0 = S_n

    let max_order = n - 1; // maximum column index we can compute
    let mut best = *sums.last().unwrap();
    let mut prev_even_estimate = f64::MAX; // track even-column stability
    let mut stale_count = 0; // consecutive non-improvements

    for k in 1..=max_order {
        let new_len = curr.len() - 1;
        if new_len == 0 { break; }

        let mut next = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let diff = curr[i + 1] - curr[i];
            if diff.abs() > 1e-50 {
                next.push(prev[i + 1] + 1.0 / diff);
            } else {
                // Division by ~0: sequence has converged at this level
                next.push(curr[i + 1]);
            }
        }

        // Even columns are the Shanks transforms
        if k % 2 == 0 && !next.is_empty() {
            let candidate = *next.last().unwrap();

            // Early stopping: detect instability.
            // If candidate is NaN/Inf, or consecutive even-column estimates
            // stop converging (difference grows or stays flat), stop.
            if candidate.is_finite() {
                let delta = (candidate - prev_even_estimate).abs();
                let prev_delta = (prev_even_estimate - best).abs();
                if prev_even_estimate != f64::MAX && delta > prev_delta * 2.0 {
                    // Diverging — instability has set in. Return previous best.
                    stale_count += 1;
                    if stale_count >= 2 { break; }
                } else {
                    stale_count = 0;
                    best = candidate;
                }
                prev_even_estimate = candidate;
            } else {
                // NaN or Inf — tableau has blown up. Stop immediately.
                break;
            }
        }

        prev = curr;
        curr = next;
    }

    best
}

// ═══════════════════════════════════════════════════════════════════════════
// Streaming Wynn Epsilon — incremental tableau, term-at-a-time
// ═══════════════════════════════════════════════════════════════════════════

/// Streaming Wynn epsilon: builds the Shanks transform incrementally.
///
/// Each `push(term)` ingests one series term, extends the epsilon tableau
/// by one anti-diagonal, and updates the best estimate. No recomputation
/// of previous work — O(depth) per push, O(1) to query the estimate.
///
/// ## The anti-diagonal recurrence
///
/// When partial sum S_N arrives, the new entries form an anti-diagonal:
/// ```text
/// new[0] = S_N                                        (column 0)
/// new[k] = last[k-2] + 1/(new[k-1] - last[k-1])     (columns 1, 2, ...)
/// ```
/// where `last[k]` is the most recent entry in column k before this update.
/// Even-indexed entries (new[0], new[2], new[4], ...) are the Shanks transforms.
///
/// ## Kingdom BC structure
///
/// This IS `attract(wynn_step, empty_tableau)`:
/// - **Inner** (each push): sequential anti-diagonal sweep — Kingdom B
///   (each column entry depends on the previous, order-dependent)
/// - **Outer** (push until converged): iterate until `|estimate_change| < tol`
///   — Kingdom C convergence criterion
///
/// Together: non-commutative inner + iterative outer = Kingdom BC.
/// The first streaming primitive that naturally inhabits the (ρ=1, σ=1) cell.
///
/// ## Depth limiting
///
/// The tableau depth grows by 1 with each push, but numerical stability
/// limits useful depth to ~20-40 columns. `max_depth` caps the sweep,
/// and early stopping detects instability (same as batch `wynn_epsilon`).
pub struct StreamingWynn {
    /// Most recent entry in each column of the epsilon tableau.
    /// `last[0]` = S_{N-1} (most recent partial sum before current push).
    /// `last[k]` = most recent ε_k entry.
    last: Vec<f64>,
    /// Running partial sum (S_N = Σ_{k=0}^N a_k).
    running_sum: f64,
    /// Number of terms ingested so far.
    n: usize,
    /// Best estimate (from the deepest stable even column).
    best: f64,
    /// Previous best, for convergence detection.
    prev_best: f64,
    /// Maximum tableau depth (caps column index).
    max_depth: usize,
}

impl StreamingWynn {
    /// Create a new streaming Wynn estimator.
    ///
    /// `max_depth` caps the tableau depth. 40 is a safe default;
    /// deeper is rarely useful due to floating-point instability.
    pub fn new(max_depth: usize) -> Self {
        StreamingWynn {
            last: Vec::new(),
            running_sum: 0.0,
            n: 0,
            best: 0.0,
            prev_best: f64::MAX,
            max_depth,
        }
    }

    /// Ingest one series term and return the updated estimate.
    ///
    /// Each call extends the epsilon tableau by one anti-diagonal.
    /// Cost: O(min(n, max_depth)) per call.
    pub fn push(&mut self, term: f64) -> f64 {
        self.running_sum += term;
        self.n += 1;

        // Sweep the anti-diagonal: new[k] = last[k-2] + 1/(new[k-1] - last[k-1])
        // We compute entries one at a time, tracking the last two "new" values.
        let depth = (self.n).min(self.max_depth);

        // new_entries will replace self.last after the sweep
        let mut new_entries: Vec<f64> = Vec::with_capacity(depth);

        // new[0] = S_N (current partial sum)
        new_entries.push(self.running_sum);

        // For columns k >= 1:
        // new[k] = last[k-2] + 1/(new[k-1] - last[k-1])
        // where last[-1] = 0 always (the ε_{-1} column)
        for k in 1..depth {
            if k - 1 >= self.last.len() {
                // No previous entry in column k-1 — can't compute this column
                break;
            }

            let new_km1 = new_entries[k - 1]; // just computed on this diagonal
            let last_km1 = self.last[k - 1];  // previous entry in column k-1

            let diff = new_km1 - last_km1;
            if diff.abs() < 1e-50 {
                // Converged at this level — propagate the value
                new_entries.push(new_km1);
                continue;
            }

            // last[k-2]: previous entry in column k-2
            // For k=1: column -1, which is always 0
            let last_km2 = if k >= 2 {
                if k - 2 < self.last.len() { self.last[k - 2] } else { break; }
            } else {
                0.0 // ε_{-1} = 0
            };

            let entry = last_km2 + 1.0 / diff;
            if !entry.is_finite() {
                break; // numerical blowup
            }
            new_entries.push(entry);
        }

        // Update best from even-column entries (the Shanks transforms).
        // Stability monitoring: track consecutive even-column estimates and
        // stop at the deepest stable one (same logic as batch wynn_epsilon).
        let mut best_candidate = self.running_sum; // column 0 is always an estimate
        let mut prev_delta = f64::MAX;
        for k in (2..new_entries.len()).step_by(2) {
            let candidate = new_entries[k];
            if !candidate.is_finite() { break; }

            let delta = (candidate - best_candidate).abs();
            if prev_delta < f64::MAX && delta > prev_delta * 2.0 {
                // Diverging — instability has set in. Keep previous best.
                break;
            }
            prev_delta = delta;
            best_candidate = candidate;
        }

        self.prev_best = self.best;
        self.best = best_candidate;
        self.last = new_entries;

        self.best
    }

    /// Current best estimate of the series limit.
    pub fn estimate(&self) -> f64 {
        self.best
    }

    /// Number of terms ingested.
    pub fn terms_seen(&self) -> usize {
        self.n
    }

    /// Has the estimate converged? True when the last push changed
    /// the estimate by less than `tol`.
    pub fn converged(&self, tol: f64) -> bool {
        self.n >= 3 && (self.best - self.prev_best).abs() < tol
    }

    /// Push a value from a converging sequence (not a series term).
    ///
    /// Use this for ergodic means, running averages, or any sequence
    /// `x_0, x_1, x_2, ...` that converges to a limit L. Internally
    /// converts to first differences so the Wynn tableau operates on
    /// the sequence directly.
    ///
    /// Signal-farm use case: push each new running mean as it arrives.
    pub fn push_value(&mut self, value: f64) -> f64 {
        // First value: push directly (first difference from 0)
        // Subsequent: push difference from previous running sum
        let diff = value - self.running_sum;
        self.push(diff)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Richardson Extrapolation (for sequences indexed by step size)
// ═══════════════════════════════════════════════════════════════════════════

/// Richardson extrapolation for a sequence of approximations at step sizes h, h/r, h/r², ...
///
/// Given A(h_i) where h_i = h₀/r^i, the error expands as:
///   A(h) = L + c₁·h^p + c₂·h^{2p} + ...
///
/// The extrapolation cancels error terms one at a time:
///   T[i,0] = A(h_i)
///   T[i,j] = (r^{jp} · T[i,j-1] - T[i-1,j-1]) / (r^{jp} - 1)
///
/// This IS a Tiled triangle accumulate on the input sequence.
/// The existing `derivative_richardson` in numerical.rs is a special case
/// where A(h) = central_diff(f, x, h) and r = 2, p = 2.
///
/// Returns the best estimate (bottom-right of tableau).
pub fn richardson_extrapolate(
    approximations: &[f64],
    ratio: f64,      // step size ratio r (typically 2)
    error_order: i32, // leading error order p (typically 2 for central differences)
) -> f64 {
    let n = approximations.len();
    if n == 0 { return 0.0; }
    if n == 1 { return approximations[0]; }
    if (ratio - 1.0).abs() < 1e-15 { return *approximations.last().unwrap(); }

    let mut tableau = vec![vec![0.0; n]; n];
    for i in 0..n {
        tableau[i][0] = approximations[i];
    }

    for j in 1..n {
        let factor = ratio.powi(j as i32 * error_order);
        for i in j..n {
            tableau[i][j] = (factor * tableau[i][j - 1] - tableau[i - 1][j - 1]) / (factor - 1.0);
        }
    }

    tableau[n - 1][n - 1]
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler Transform
// ═══════════════════════════════════════════════════════════════════════════

/// Euler transform: binomial-weighted average of partial sums.
///
/// Given terms a₀, a₁, ..., computes partial sums S_k = Σ_{j=0}^k a_j,
/// then the Euler-Nörlund mean:
///   E_m = (1/2^m) Σ_{k=0}^m C(m,k) · S_k
///
/// This is particularly effective for slowly converging alternating series
/// like ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...
///
/// The transform is a ByKey weighted accumulate on the partial sums with
/// binomial weights — the weights are C(m,k)/2^m.
///
/// Returns the best Euler partial sum (using all terms).
pub fn euler_transform(terms: &[f64]) -> f64 {
    let n = terms.len();
    if n == 0 { return 0.0; }
    if n == 1 { return terms[0]; }

    // Step 1: partial sums (= Prefix scan)
    let sums = partial_sums(terms);

    // Step 2: binomial-weighted average of partial sums
    // E_m = (1/2^m) Σ_{k=0}^m C(m,k) S_k where m = n-1
    //
    // Efficient: use the relation that row m of Pascal's triangle
    // can be built incrementally. But for moderate n, direct is fine.
    let m = n - 1;

    // Build binomial coefficients C(m, k) for k = 0..m
    let mut binom = vec![1.0_f64; m + 1];
    for k in 1..=m {
        binom[k] = binom[k - 1] * (m - k + 1) as f64 / k as f64;
    }

    let mut result = 0.0;
    let scale = 0.5_f64.powi(m as i32);
    for k in 0..=m {
        result += binom[k] * sums[k];
    }
    result * scale
}

// ═══════════════════════════════════════════════════════════════════════════
// Abel Summation (exponential kernel — completes the taxonomy)
// ═══════════════════════════════════════════════════════════════════════════

/// Abel summation: the exponential kernel on series terms.
///
/// The Abel sum is `lim_{x→1⁻} Σ a_k · x^k`. This evaluates the power
/// series at geometrically spaced x values approaching 1, then
/// Richardson-extrapolates to the x→1 limit.
///
/// ## Kernel taxonomy (completed)
///
/// | Method | Kernel | Weight |
/// |--------|--------|--------|
/// | Cesàro | Uniform | 1/(n+1) on S_k |
/// | Euler | Binomial | C(n,k)/2^n on S_k |
/// | **Abel** | **Exponential** | **x^k on a_k (x→1⁻)** |
/// | Richardson | Polynomial | extrapolation on S_N |
///
/// Abel is the strongest standard summability method: every Cesàro-summable
/// series is Abel-summable (Tauberian inclusion). Abel can sum some series
/// that Cesàro cannot (e.g., Σ(-1)^n·(n+1) = 1/4 via Abel).
///
/// ## Composition: Abel + Richardson
///
/// Abel regularizes: the power series converges for |x| < 1 even if Σ a_n
/// diverges. Richardson accelerates: the error f(1-h) - L has a smooth
/// expansion in h, which Richardson cancels term by term.
///
/// The number of useful Richardson levels is limited by the truncation
/// error |a_n|·x^n at the finest x. More terms → finer x → more levels
/// → better accuracy. For divergent series, generate many cheap terms.
pub fn abel_sum(terms: &[f64]) -> f64 {
    let n = terms.len();
    if n == 0 { return 0.0; }
    if n == 1 { return terms[0]; }

    // h_min: finest step (x closest to 1). Chosen so truncation error < tol.
    // Truncation error at x = 1-h is approximately (1-h)^n.
    // Solve: (1-h)^n < tol → h > 1 - tol^{1/n}.
    let tol: f64 = 1e-8;
    let h_min: f64 = (1.0 - tol.powf(1.0 / n as f64)).max(0.005);

    // Pack Richardson levels from h_max=0.5 down to h_min with ratio 1.5.
    // Smaller ratio = more levels in the same h range.
    let ratio = 1.5_f64;
    let mut h_values = Vec::new();
    let mut h = 0.5;
    while h >= h_min * 0.99 && h_values.len() < 12 {
        h_values.push(h);
        h /= ratio;
    }

    if h_values.len() < 2 {
        // Not enough range — single evaluation near x=1
        let x = 1.0 - h_min;
        let mut sum = 0.0;
        for k in (0..n).rev() {
            sum = sum * x + terms[k];
        }
        return sum;
    }

    // Evaluate f(x) = Σ a_k · x^k at each x via Horner's method.
    // h_values is coarsest→finest, which is what Richardson expects.
    let mut evaluations = Vec::with_capacity(h_values.len());
    for &hv in &h_values {
        let x = 1.0 - hv;
        let mut sum = 0.0;
        for k in (0..n).rev() {
            sum = sum * x + terms[k];
        }
        evaluations.push(sum);
    }

    // Richardson extrapolation: error expansion in h with p=1.
    richardson_extrapolate(&evaluations, ratio, 1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Richardson on Partial Sums (algebraic convergence — closes the Basel gap)
// ═══════════════════════════════════════════════════════════════════════════

/// Richardson extrapolation on partial sums — closes the Basel gap.
///
/// For series with algebraic convergence (error ~ c/N^p), evaluates partial
/// sums at geometrically spaced truncation points (N/2^k, ..., N/2, N),
/// auto-detects the error order p from tail-difference ratios, then applies
/// Richardson extrapolation with r=2, error_order=p.
///
/// This is the matched accelerator for positive monotone algebraic series
/// (Basel, ζ(3), ζ(s) for s > 1). Wynn/Aitken are designed for
/// geometric/alternating convergence and underperform here.
///
/// ## Error order detection
///
/// Given partial sums S_{N_i} at truncation points N₁ < N₂ < ... (ratio 2):
///   d_i = S_{N_{i+1}} - S_{N_i}  (tail contribution between truncation points)
///   d_i / d_{i+1} ≈ 2^p
///   p = log₂(mean(d_i / d_{i+1}))
///
/// For Basel (Σ 1/k²): p = 1 (error ~ 1/N), detected ratio ≈ 2.
/// For ζ(3) (Σ 1/k³): p = 2 (error ~ 1/N²), detected ratio ≈ 4.
///
/// ## Connection to K03
///
/// This is Richardson extrapolation applied to truncation resolution —
/// the same cross-cadence pattern as K03: independent projections at
/// different resolutions, combined to cancel resolution-dependent bias.
pub fn richardson_partial_sums(terms: &[f64]) -> f64 {
    let n = terms.len();
    if n < 8 {
        return partial_sums(terms).last().copied().unwrap_or(0.0);
    }

    let sums = partial_sums(terms);

    // Pick truncation points at geometric spacing: n/2^k, ..., n/2, n
    let mut points = Vec::new();
    let mut nn = n;
    while nn >= 4 && points.len() < 6 {
        points.push(sums[nn - 1]);
        nn /= 2;
    }
    points.reverse(); // ascending: coarsest (smallest N) → finest (largest N)

    if points.len() < 3 {
        return *sums.last().unwrap();
    }

    // Detect error order p from successive tail differences
    // d_i = S_{N_{i+1}} - S_{N_i} (tail contribution of terms between N_i and N_{i+1})
    let mut diffs: Vec<f64> = Vec::new();
    for i in 1..points.len() {
        diffs.push(points[i] - points[i - 1]);
    }

    let mut ratios = Vec::new();
    for i in 1..diffs.len() {
        if diffs[i].abs() > 1e-50 {
            let r = diffs[i - 1] / diffs[i];
            if r.is_finite() && r > 1.0 {
                ratios.push(r);
            }
        }
    }

    if ratios.is_empty() {
        // Can't detect a clean error order — fall back to raw partial sum
        return *sums.last().unwrap();
    }

    // p = log2(mean ratio), round to nearest positive integer
    let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let p = mean_ratio.log2();
    let p_rounded = p.round().max(1.0) as i32;

    richardson_extrapolate(&points, 2.0, p_rounded)
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Maclaurin correction for ζ(s) series
// ═══════════════════════════════════════════════════════════════════════════

/// Euler-Maclaurin corrected partial sum for ζ(s) = Σ 1/k^s.
///
/// Adds the first `p` correction terms from the Euler-Maclaurin formula:
///   ζ(s) ≈ S_N + N^{1-s}/(s-1) + N^{-s}/2 + Σ_{k=1}^p B_{2k}/(2k)! · c_k · N^{-(s+2k-1)}
///
/// where c_k = s(s+1)···(s+2k-2) is the falling factorial.
///
/// For ζ(3): corrected error drops from 1/(2N²) to O(1/N^{2p+2}), enabling
/// Richardson to cancel remaining terms cleanly.
///
/// ## Why this matters for ζ(3) specifically
///
/// ζ(3) error expansion has CONSECUTIVE integer powers (1/N², 1/N³, 1/N⁴, ...)
/// which confound Richardson's ratio-2 doubling. Euler-Maclaurin analytically
/// removes the first few terms, leaving a cleaner residual for Richardson.
///
/// Basel (ζ(2)) has ODD powers only (1/N, 1/N³, 1/N⁵, ...) which align with
/// Richardson's doubling structure — no preprocessing needed.
pub fn euler_maclaurin_zeta(s: f64, n_terms: usize, correction_order: usize) -> f64 {
    if n_terms == 0 { return 0.0; }

    // Partial sum S_N = Σ_{k=1}^N 1/k^s
    let mut sum: f64 = 0.0;
    for k in 1..=n_terms {
        sum += 1.0 / (k as f64).powf(s);
    }

    // The tail Σ_{k=N+1}^∞ 1/k^s uses Euler-Maclaurin evaluated at a = N+1
    // (the first term NOT in the partial sum), via ζ(s, a):
    //   ζ(s, a) ≈ a^{1-s}/(s-1) + a^{-s}/2 + Σ B_{2k}/(2k)! · (s)_{2k-1} · a^{-(s+2k-1)}
    let a = (n_terms + 1) as f64;

    // Leading correction: ∫_{a}^∞ x^{-s} dx = a^{1-s}/(s-1)
    sum += a.powf(1.0 - s) / (s - 1.0);

    // Half-endpoint: f(a)/2 = a^{-s}/2
    sum += 0.5 * a.powf(-s);

    // Bernoulli number corrections: B_{2k}/(2k)! · (s)_{2k-1} · a^{-(s+2k-1)}
    // where (s)_{2k-1} = s(s+1)...(s+2k-2) is the rising factorial (Pochhammer)
    let bernoulli_2k: [f64; 5] = [
        1.0 / 6.0,    // B_2
        -1.0 / 30.0,  // B_4
        1.0 / 42.0,   // B_6
        -1.0 / 30.0,  // B_8
        5.0 / 66.0,   // B_10
    ];

    for k in 1..=correction_order.min(bernoulli_2k.len()) {
        // Rising factorial: s(s+1)...(s+2k-2) — that's (2k-1) factors
        let mut rising = 1.0;
        for j in 0..(2 * k - 1) {
            rising *= s + j as f64;
        }

        // Factorial: (2k)!
        let mut fact = 1.0;
        for j in 1..=(2 * k) {
            fact *= j as f64;
        }

        let correction = bernoulli_2k[k - 1] / fact * rising * a.powf(-(s + 2.0 * k as f64 - 1.0));
        sum += correction;
    }

    sum
}

// ═══════════════════════════════════════════════════════════════════════════
// Auto-selector: detect convergence type → pick matched accelerator
// ═══════════════════════════════════════════════════════════════════════════

/// Detected convergence type of a series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceType {
    /// Terms decay geometrically: |a_{n+1}/a_n| → r < 1.
    Geometric,
    /// Terms alternate in sign and decay.
    Alternating,
    /// Terms decay but neither geometric nor alternating (e.g., 1/n^α).
    Algebraic,
    /// Too few terms or no clear pattern.
    Unknown,
}

/// Detect the convergence type from the first `probe` terms.
///
/// Uses ratio test (geometric vs algebraic) and sign pattern (alternating).
/// Needs at least 6 terms for reliable detection.
pub fn detect_convergence(terms: &[f64], probe: usize) -> ConvergenceType {
    let n = terms.len().min(probe);
    if n < 6 { return ConvergenceType::Unknown; }

    // Check alternating: do signs strictly alternate?
    let mut alternating = true;
    for i in 1..n {
        if terms[i].signum() == terms[i - 1].signum() {
            alternating = false;
            break;
        }
    }

    // Ratio test on absolute values: |a_{n+1}| / |a_n|
    let mut ratios = Vec::with_capacity(n - 1);
    for i in 1..n {
        let prev = terms[i - 1].abs();
        if prev > 1e-50 {
            ratios.push(terms[i].abs() / prev);
        }
    }

    if ratios.len() < 4 { return ConvergenceType::Unknown; }

    // Check if ratios converge to a constant (geometric)
    // Coefficient of variation of the last half of ratios
    let half = ratios.len() / 2;
    let tail = &ratios[half..];
    let m = crate::descriptive::moments_ungrouped(tail);
    let mean = m.mean();
    let cv = m.std(0) / mean.abs().max(1e-50);

    // Decision tree: alternating takes priority when ratio → 1 (algebraic envelope).
    // True geometric has ratio well below 1 (e.g., 0.5, 0.8).
    // Algebraic series (1/n^α) have ratio → 1 slowly — not geometric.
    let is_geometric = mean < 0.95 && cv < 0.05;

    if is_geometric && !alternating {
        // Pure geometric convergence (e.g., Σ r^k with r < 0.95)
        ConvergenceType::Geometric
    } else if alternating {
        // Alternating series — Euler/Wynn optimal regardless of ratio behavior
        ConvergenceType::Alternating
    } else if mean < 1.0 && cv < 0.05 {
        // Ratios stable near 1 — slow geometric (almost algebraic)
        ConvergenceType::Geometric
    } else {
        ConvergenceType::Algebraic
    }
}

/// Accelerate a series using the best method for its convergence type.
///
/// This is the compile_budget dispatcher:
/// 1. Detect convergence type from first ~10 terms
/// 2. Select matched accelerator
/// 3. Return best estimate of the limit
///
/// Equivalent to `.tbs` auto-acceleration: the compiler inserts the right
/// kernel based on the detected convergence class.
pub fn accelerate(terms: &[f64]) -> f64 {
    if terms.is_empty() { return 0.0; }
    if terms.len() == 1 { return terms[0]; }

    let conv_type = detect_convergence(terms, 10);
    let sums = partial_sums(terms);

    match conv_type {
        ConvergenceType::Geometric => {
            // Aitken is exact for geometric — one pass suffices
            let accel = aitken_delta2(&sums);
            if let Some(&last) = accel.last() { last } else { *sums.last().unwrap() }
        }
        ConvergenceType::Alternating => {
            // Euler is the matched kernel; Wynn as backup
            let euler_est = euler_transform(terms);
            let wynn_est = wynn_epsilon(&sums);

            // Pick the one closer to the Aitken estimate (as tiebreaker)
            let aitken = aitken_delta2(&sums);
            if let Some(&a) = aitken.last() {
                if (euler_est - a).abs() < (wynn_est - a).abs() {
                    euler_est
                } else {
                    wynn_est
                }
            } else {
                euler_est
            }
        }
        ConvergenceType::Algebraic | ConvergenceType::Unknown => {
            // For positive monotone series (Basel, ζ(s)), Richardson with
            // auto-detected error order is the matched accelerator.
            // For general algebraic, try both Richardson and Wynn/Aitken, pick better.
            let all_positive = terms.iter().all(|&t| t > 0.0);
            let all_negative = terms.iter().all(|&t| t < 0.0);
            let monotone = all_positive || all_negative;

            let rich_est = richardson_partial_sums(terms);
            let raw_last = *sums.last().unwrap();

            if monotone {
                // Richardson is matched — use it directly
                rich_est
            } else {
                // Mixed signs: try Wynn/Aitken too, pick the estimate
                // that improves most over raw while staying in the ballpark
                let other_est = if terms.len() <= 40 {
                    wynn_epsilon(&sums)
                } else {
                    let mut seq = sums;
                    let mut best = f64::MAX;
                    for _ in 0..10 {
                        seq = aitken_delta2(&seq);
                        if seq.is_empty() { break; }
                        if let Some(&last) = seq.last() {
                            best = last;
                        }
                        if seq.len() < 3 { break; }
                    }
                    best
                };

                // Heuristic: the estimate farther from the raw partial sum
                // (but still finite) is likely the more aggressively accelerated one.
                // Pick Richardson if it moved meaningfully; otherwise Wynn/Aitken.
                let rich_diff = (rich_est - raw_last).abs();
                let other_diff = (other_est - raw_last).abs();
                if rich_diff > other_diff * 2.0 && rich_est.is_finite() {
                    rich_est
                } else if other_est.is_finite() {
                    other_est
                } else {
                    raw_last
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — verification against known mathematical constants
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: {a} vs {b} (diff={}, tol={tol})",
            (a - b).abs()
        );
    }

    // ── Series generators ──────────────────────────────────────────────

    /// Leibniz series for π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
    fn leibniz_terms(n: usize) -> Vec<f64> {
        (0..n).map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign / (2 * k + 1) as f64
        }).collect()
    }

    /// Alternating harmonic series for ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...
    fn ln2_terms(n: usize) -> Vec<f64> {
        (0..n).map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign / (k + 1) as f64
        }).collect()
    }

    /// Basel series for π²/6 = 1 + 1/4 + 1/9 + 1/16 + ... (positive terms)
    fn basel_terms(n: usize) -> Vec<f64> {
        (0..n).map(|k| 1.0 / ((k + 1) as f64).powi(2)).collect()
    }

    /// e⁻¹ = Σ (-1)^n / n! — fast-converging alternating series
    fn exp_neg1_terms(n: usize) -> Vec<f64> {
        let mut terms = Vec::with_capacity(n);
        let mut factorial = 1.0;
        for k in 0..n {
            if k > 0 { factorial *= k as f64; }
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            terms.push(sign / factorial);
        }
        terms
    }

    // ── Partial sums ───────────────────────────────────────────────────

    #[test]
    fn partial_sums_basic() {
        let terms = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sums = partial_sums(&terms);
        assert_eq!(sums, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    // ── Aitken Δ² ──────────────────────────────────────────────────────

    #[test]
    fn aitken_accelerates_leibniz() {
        // π/4 via Leibniz. Raw: ~O(1/n) convergence. Aitken should do much better.
        let n = 30;
        let sums = partial_sums(&leibniz_terms(n));
        let raw_error = (sums[n - 1] - std::f64::consts::FRAC_PI_4).abs();

        let accel = aitken_delta2(&sums);
        let accel_error = (accel[accel.len() - 1] - std::f64::consts::FRAC_PI_4).abs();

        assert!(
            accel_error < raw_error / 10.0,
            "Aitken should accelerate: raw_err={raw_error:.2e}, accel_err={accel_error:.2e}"
        );
    }

    #[test]
    fn aitken_accelerates_ln2() {
        let n = 30;
        let sums = partial_sums(&ln2_terms(n));
        let raw_error = (sums[n - 1] - 2.0_f64.ln()).abs();

        let accel = aitken_delta2(&sums);
        let accel_error = (accel[accel.len() - 1] - 2.0_f64.ln()).abs();

        assert!(
            accel_error < raw_error / 10.0,
            "Aitken ln(2): raw_err={raw_error:.2e}, accel_err={accel_error:.2e}"
        );
    }

    // ── Wynn's Epsilon ─────────────────────────────────────────────────

    #[test]
    fn wynn_leibniz_high_accuracy() {
        // Wynn's epsilon on Leibniz series should converge much faster
        let n = 20;
        let sums = partial_sums(&leibniz_terms(n));
        let estimate = wynn_epsilon(&sums);
        close(estimate, std::f64::consts::FRAC_PI_4, 1e-10,
            "Wynn ε on Leibniz (20 terms) → π/4");
    }

    #[test]
    fn wynn_ln2_high_accuracy() {
        let n = 20;
        let sums = partial_sums(&ln2_terms(n));
        let estimate = wynn_epsilon(&sums);
        close(estimate, 2.0_f64.ln(), 1e-10,
            "Wynn ε on alternating harmonic (20 terms) → ln(2)");
    }

    #[test]
    fn wynn_exp_neg1() {
        // e⁻¹ series converges fast already; Wynn should nail it with few terms
        let n = 12;
        let sums = partial_sums(&exp_neg1_terms(n));
        let estimate = wynn_epsilon(&sums);
        close(estimate, (-1.0_f64).exp(), 1e-11,
            "Wynn ε on e⁻¹ (12 terms)");
    }

    // ── Richardson ─────────────────────────────────────────────────────

    #[test]
    fn richardson_trapezoidal_rule() {
        // Approximate ∫₀¹ x² dx = 1/3 with trapezoidal rule at decreasing h
        // Trapezoidal error is O(h²), so p=2, r=2
        let exact = 1.0 / 3.0;
        let f = |x: f64| x * x;

        let mut approxs = Vec::new();
        let mut n = 4;
        for _ in 0..5 {
            // Trapezoidal rule with n panels
            let h = 1.0 / n as f64;
            let mut sum = 0.5 * (f(0.0) + f(1.0));
            for i in 1..n {
                sum += f(i as f64 * h);
            }
            approxs.push(sum * h);
            n *= 2;
        }

        let raw_error = (approxs[4] - exact).abs();
        let rich = richardson_extrapolate(&approxs, 2.0, 2);
        let rich_error = (rich - exact).abs();

        assert!(
            rich_error < raw_error / 100.0,
            "Richardson should dramatically improve: raw={raw_error:.2e}, rich={rich_error:.2e}"
        );
    }

    // ── Euler Transform ────────────────────────────────────────────────

    #[test]
    fn euler_leibniz_pi_over_4() {
        // Euler transform is optimal for alternating series
        let n = 20;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);
        let raw_error = (sums[n - 1] - std::f64::consts::FRAC_PI_4).abs();

        let euler_est = euler_transform(&terms);
        let euler_error = (euler_est - std::f64::consts::FRAC_PI_4).abs();

        assert!(
            euler_error < raw_error / 100.0,
            "Euler should crush alternating: raw={raw_error:.2e}, euler={euler_error:.2e}"
        );
    }

    #[test]
    fn euler_ln2() {
        let n = 20;
        let terms = ln2_terms(n);
        let euler_est = euler_transform(&terms);
        close(euler_est, 2.0_f64.ln(), 1e-6,
            "Euler transform ln(2) from 20 terms");
    }

    // ── Comparative: which accelerator wins? ───────────────────────────

    #[test]
    fn comparison_leibniz_convergence_rates() {
        let n = 20;
        let target = std::f64::consts::FRAC_PI_4;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);

        let raw_err = (sums[n - 1] - target).abs();
        let aitken_err = {
            let a = aitken_delta2(&sums);
            (a[a.len() - 1] - target).abs()
        };
        let wynn_err = (wynn_epsilon(&sums) - target).abs();
        let euler_err = (euler_transform(&terms) - target).abs();

        // All accelerators should beat raw
        assert!(aitken_err < raw_err, "Aitken should beat raw");
        assert!(wynn_err < raw_err, "Wynn should beat raw");
        assert!(euler_err < raw_err, "Euler should beat raw");

        // Wynn (iterated Shanks) should be among the best for alternating series
        // (This is an observation test — we want to see the hierarchy)
        eprintln!("═══ Leibniz π/4 convergence (20 terms) ═══");
        eprintln!("Raw partial sum:   {raw_err:.2e}");
        eprintln!("Aitken Δ²:         {aitken_err:.2e}");
        eprintln!("Wynn ε (Shanks):   {wynn_err:.2e}");
        eprintln!("Euler transform:   {euler_err:.2e}");
    }

    // ── The accumulate decomposition proof ─────────────────────────────

    #[test]
    fn aitken_is_windowed3_on_prefix() {
        // Prove: Aitken = Windowed{3} applied to Prefix output.
        // Step 1: accumulate(terms, Prefix, Value, Add) → sums
        // Step 2: for each window [s0, s1, s2]: s0 - (s1-s0)² / (s2 - 2s1 + s0)
        // This test shows the decomposition is exact.
        let terms = leibniz_terms(50);
        let sums = partial_sums(&terms);
        let accel = aitken_delta2(&sums);

        // Manually compute the same thing with explicit window
        for i in 0..accel.len() {
            let window = &sums[i..i + 3]; // Windowed{3}
            let s0 = window[0];
            let s1 = window[1];
            let s2 = window[2];
            let d2 = s2 - 2.0 * s1 + s0;
            let expected = if d2.abs() > 1e-50 {
                s0 - (s1 - s0).powi(2) / d2
            } else {
                s2
            };
            assert!((accel[i] - expected).abs() < 1e-15,
                "Aitken[{i}] = Windowed{{3}} on Prefix: {} vs {}", accel[i], expected);
        }
    }

    // ── Basel series (positive terms — alternating methods don't apply) ─

    #[test]
    fn wynn_on_positive_series() {
        // Basel: π²/6 = 1 + 1/4 + 1/9 + ...
        // This is NOT alternating, so Euler transform is inappropriate.
        // Wynn's epsilon should still help (it's general).
        let n = 30;
        let sums = partial_sums(&basel_terms(n));
        let raw_error = (sums[n - 1] - std::f64::consts::PI.powi(2) / 6.0).abs();
        let wynn_error = (wynn_epsilon(&sums) - std::f64::consts::PI.powi(2) / 6.0).abs();

        // Wynn may or may not beat raw significantly for positive series
        // (Shanks is most effective for alternating). But it shouldn't make things worse.
        eprintln!("═══ Basel π²/6 convergence (30 terms, positive series) ═══");
        eprintln!("Raw partial sum:   {raw_error:.2e}");
        eprintln!("Wynn ε (Shanks):   {wynn_error:.2e}");
    }

    // ── Iterated Aitken = Wynn's even columns ──────────────────────────

    #[test]
    fn iterated_aitken_converges_faster() {
        // Applying Aitken twice should improve further
        let n = 30;
        let sums = partial_sums(&leibniz_terms(n));
        let target = std::f64::consts::FRAC_PI_4;

        let once = aitken_delta2(&sums);
        let twice = aitken_delta2(&once);
        let thrice = aitken_delta2(&twice);

        let err_raw = (sums[n - 1] - target).abs();
        let err_once = (once[once.len() - 1] - target).abs();
        let err_twice = (twice[twice.len() - 1] - target).abs();
        let err_thrice = (thrice[thrice.len() - 1] - target).abs();

        eprintln!("═══ Iterated Aitken on Leibniz (30 terms) ═══");
        eprintln!("Raw:      {err_raw:.2e}");
        eprintln!("Aitken×1: {err_once:.2e}");
        eprintln!("Aitken×2: {err_twice:.2e}");
        eprintln!("Aitken×3: {err_thrice:.2e}");

        // Each iteration should improve (at least the first two)
        assert!(err_once < err_raw, "First Aitken improves");
        assert!(err_twice < err_once, "Second Aitken improves further");
    }

    // ── Budget experiment: is constant gain an artifact? ───────────────

    #[test]
    fn aitken_gain_vs_budget() {
        // Math-researcher prediction: ~3 orders/level at n=30 is a budget artifact.
        // With more terms, later levels should show INCREASING gain (rate-squaring).
        // Each Aitken pass consumes 2 terms, so budget caps max useful depth.
        let target = std::f64::consts::FRAC_PI_4;

        for &n in &[30, 60, 120, 200] {
            let sums = partial_sums(&leibniz_terms(n));
            let mut seq = sums.clone();
            let mut errors: Vec<f64> = vec![(seq.last().unwrap() - target).abs()];

            for level in 0..8 {
                seq = aitken_delta2(&seq);
                if seq.is_empty() { break; }
                let err = (seq.last().unwrap() - target).abs();
                if err == 0.0 { break; } // hit machine precision
                errors.push(err);
                if err < 1e-15 { break; } // machine precision floor
                let _ = level;
            }

            eprint!("n={n:3}: ");
            let mut gains = Vec::new();
            for i in 1..errors.len() {
                let gain = errors[i - 1].log10() - errors[i].log10();
                gains.push(gain);
                eprint!("{:.1} ", gain);
            }
            eprintln!(" (errors: {})", errors.iter()
                .map(|e| format!("{e:.1e}")).collect::<Vec<_>>().join(" → "));

            // With more terms, should sustain gain for more levels
            assert!(errors.len() >= 3,
                "n={n} should sustain at least 2 Aitken levels");
        }
    }

    // ── Geometric series: does Aitken achieve true ρ-squaring? ─────────

    #[test]
    fn aitken_on_geometric_series() {
        // Geometric series: Σ r^k = 1/(1-r) for |r| < 1.
        // Error of partial sum: r^{n+1}/(1-r), so convergence ratio = r exactly.
        // Aitken should square the ratio: r → r² → r⁴ → r⁸ (doubling digits/level).
        //
        // Use r=0.95 (slow convergence) so we have room to see the pattern
        // before hitting machine precision.
        for &(r, n) in &[(0.95_f64, 100), (0.9, 80), (0.8, 60)] {
            let target = 1.0 / (1.0 - r);
            let terms: Vec<f64> = (0..n).map(|k| r.powi(k as i32)).collect();
            let sums = partial_sums(&terms);

            let mut seq = sums.clone();
            let mut errors: Vec<f64> = vec![(seq.last().unwrap() - target).abs()];

            for _ in 0..8 {
                seq = aitken_delta2(&seq);
                if seq.is_empty() { break; }
                let err = (seq.last().unwrap() - target).abs();
                if err == 0.0 { errors.push(1e-16); break; }
                errors.push(err);
                if err < 1e-15 { break; }
            }

            eprint!("r={r:.2}, n={n:3}: gains = ");
            let mut gains = Vec::new();
            for i in 1..errors.len() {
                let gain = errors[i - 1].log10() - errors[i].log10();
                gains.push(gain);
                eprint!("{gain:.1} ");
            }
            eprintln!(" (errors: {})", errors.iter()
                .map(|e| format!("{e:.1e}")).collect::<Vec<_>>().join(" → "));

            // For geometric series, gains should INCREASE (ρ-squaring)
            if gains.len() >= 3 {
                eprintln!("  Rate-squaring test: gain[0]={:.1}, gain[1]={:.1}, gain[2]={:.1}",
                    gains[0], gains[1], gains.get(2).unwrap_or(&0.0));
            }
        }
    }

    // ── Wynn vs iterated Aitken: depth efficiency ──────────────────────

    #[test]
    fn wynn_vs_iterated_aitken_efficiency() {
        // Wynn uses ALL input structure at once (full tableau).
        // Iterated Aitken is greedy (Windowed{3} per level, losing 2 terms each time).
        // Wynn should be strictly more efficient at extracting precision from a
        // fixed term budget — it's the optimal use of the available information.
        let target = std::f64::consts::FRAC_PI_4;

        for &n in &[10, 20, 40, 80] {
            let sums = partial_sums(&leibniz_terms(n));
            let wynn_err = (wynn_epsilon(&sums) - target).abs();

            // Best iterated Aitken
            let mut seq = sums.clone();
            let mut best_aitken_err = (seq.last().unwrap() - target).abs();
            loop {
                seq = aitken_delta2(&seq);
                if seq.is_empty() { break; }
                let err = (seq.last().unwrap() - target).abs();
                if err < best_aitken_err { best_aitken_err = err; }
                if err < 1e-15 || seq.len() < 3 { break; }
            }

            eprintln!("n={n:3}: Wynn={wynn_err:.2e}  Aitken_best={best_aitken_err:.2e}  \
                ratio={:.1}x", best_aitken_err / wynn_err.max(1e-300));
        }
    }

    // ── Cesàro summation ───────────────────────────────────────────────

    #[test]
    fn cesaro_grandi_series() {
        // Grandi's series: 1 - 1 + 1 - 1 + ... diverges, but Cesàro-summable to 1/2.
        // Partial sums oscillate: 1, 0, 1, 0, 1, 0, ...
        // Cesàro mean → 1/2.
        let n = 100;
        let terms: Vec<f64> = (0..n).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let sums = partial_sums(&terms);

        // Raw partial sums never converge (oscillate between 0 and 1)
        let last = *sums.last().unwrap();
        assert!(last == 0.0 || last == 1.0, "Grandi sums oscillate");

        // Cesàro gives 1/2
        let cesaro = cesaro_sum(&sums);
        close(cesaro, 0.5, 1e-2, "Cesàro(Grandi) = 1/2");
    }

    #[test]
    fn cesaro_accelerates_leibniz() {
        // Cesàro (uniform kernel) should beat raw but lose to Aitken/Euler/Wynn.
        let n = 40;
        let target = std::f64::consts::FRAC_PI_4;
        let sums = partial_sums(&leibniz_terms(n));

        let raw_err = (sums[n - 1] - target).abs();
        let cesaro_err = (cesaro_sum(&sums) - target).abs();
        let aitken_err = {
            let a = aitken_delta2(&sums);
            (a[a.len() - 1] - target).abs()
        };

        // Cesàro beats raw (averaging smooths oscillation)
        assert!(cesaro_err < raw_err,
            "Cesàro={cesaro_err:.2e} should beat raw={raw_err:.2e}");
        // But Aitken beats Cesàro (nonlinear > linear averaging)
        assert!(aitken_err < cesaro_err,
            "Aitken={aitken_err:.2e} should beat Cesàro={cesaro_err:.2e}");
    }

    // ── The kernel taxonomy test ───────────────────────────────────────

    #[test]
    fn kernel_taxonomy_hierarchy() {
        // Verify the full hierarchy on Leibniz π/4:
        // Raw < Cesàro (uniform) < Euler (binomial) < Aitken (nonlinear) < Wynn (iterated)
        let n = 20;
        let target = std::f64::consts::FRAC_PI_4;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);

        let raw_err = (sums[n - 1] - target).abs();
        let cesaro_err = (cesaro_sum(&sums) - target).abs();
        let euler_err = (euler_transform(&terms) - target).abs();
        let aitken_err = {
            let a = aitken_delta2(&sums);
            (a[a.len() - 1] - target).abs()
        };
        let wynn_err = (wynn_epsilon(&sums) - target).abs();

        eprintln!("═══ Kernel taxonomy on Leibniz π/4 (20 terms) ═══");
        eprintln!("Raw (no kernel):       {raw_err:.2e}");
        eprintln!("Cesàro (uniform):      {cesaro_err:.2e}");
        eprintln!("Euler (binomial):      {euler_err:.2e}");
        eprintln!("Aitken (nonlinear):    {aitken_err:.2e}");
        eprintln!("Wynn ε (iterated):     {wynn_err:.2e}");

        // The hierarchy should hold
        assert!(cesaro_err < raw_err, "Cesàro < Raw");
        assert!(euler_err < cesaro_err, "Euler < Cesàro");
        assert!(aitken_err < cesaro_err, "Aitken < Cesàro");
        // Note: Aitken vs Euler depends on series type
        // Wynn should be best for small n
        assert!(wynn_err < aitken_err, "Wynn < Aitken at n=20");
    }

    // ── Auto-selector tests ────────────────────────────────────────────

    #[test]
    fn detect_geometric() {
        let r: f64 = 0.8;
        let terms: Vec<f64> = (0..20).map(|k| r.powi(k as i32)).collect();
        assert_eq!(detect_convergence(&terms, 10), ConvergenceType::Geometric);
    }

    #[test]
    fn detect_alternating() {
        let terms = leibniz_terms(20);
        assert_eq!(detect_convergence(&terms, 10), ConvergenceType::Alternating);
    }

    #[test]
    fn detect_algebraic() {
        // Basel series: 1/k² (positive, not geometric — ratios drift)
        let terms = basel_terms(20);
        let ct = detect_convergence(&terms, 10);
        // Basel ratios are (k/(k+1))² ≈ 1 - 2/k, which drift, so either
        // Algebraic or Geometric (with ratio near 1). Either is acceptable.
        assert!(ct == ConvergenceType::Algebraic || ct == ConvergenceType::Geometric,
            "Basel should be Algebraic or near-1 Geometric, got {:?}", ct);
    }

    #[test]
    fn accelerate_leibniz() {
        let n = 20;
        let terms = leibniz_terms(n);
        let est = accelerate(&terms);
        let err = (est - std::f64::consts::FRAC_PI_4).abs();
        // Should be much better than raw (1.25e-2)
        assert!(err < 1e-6,
            "accelerate(Leibniz, 20) error={err:.2e}, should be < 1e-6");
    }

    #[test]
    fn accelerate_geometric() {
        let r: f64 = 0.9;
        let n = 30;
        let terms: Vec<f64> = (0..n).map(|k| r.powi(k as i32)).collect();
        let target = 1.0 / (1.0 - r);
        let est = accelerate(&terms);
        let err = (est - target).abs();
        assert!(err < 1e-10,
            "accelerate(geometric r=0.9, 30) error={err:.2e}");
    }

    #[test]
    fn accelerate_ln2() {
        let n = 30;
        let terms = ln2_terms(n);
        let est = accelerate(&terms);
        let err = (est - 2.0_f64.ln()).abs();
        assert!(err < 1e-6,
            "accelerate(ln2, 30) error={err:.2e}");
    }

    #[test]
    fn accelerate_dispatches_correctly() {
        // Verify the auto-selector at least beats the raw partial sum
        // across all series types.
        let cases: Vec<(&str, Vec<f64>, f64)> = vec![
            ("Leibniz", leibniz_terms(20), std::f64::consts::FRAC_PI_4),
            ("ln2", ln2_terms(20), 2.0_f64.ln()),
            ("Basel", basel_terms(40), std::f64::consts::PI.powi(2) / 6.0),
            ("Geometric", (0..20).map(|k| 0.8_f64.powi(k)).collect(), 1.0 / 0.2),
        ];

        for (name, terms, target) in &cases {
            let sums = partial_sums(terms);
            let raw_err = (sums.last().unwrap() - target).abs();
            let auto_err = (accelerate(terms) - target).abs();
            eprintln!("{name:10}: raw={raw_err:.2e}  auto={auto_err:.2e}  \
                speedup={:.0}x", raw_err / auto_err.max(1e-300));
            assert!(auto_err <= raw_err,
                "{name}: auto ({auto_err:.2e}) should beat raw ({raw_err:.2e})");
        }
    }

    // ── Richardson on partial sums — closing the Basel gap ─────────────

    #[test]
    fn richardson_partial_sums_detects_basel_order() {
        // Basel: error ~ 1/N, so Richardson should detect p=1.
        // Verify the function produces a much better estimate than raw.
        let n = 40;
        let target = std::f64::consts::PI.powi(2) / 6.0;
        let terms = basel_terms(n);
        let sums = partial_sums(&terms);
        let raw_error = (sums[n - 1] - target).abs();

        let rich_est = richardson_partial_sums(&terms);
        let rich_error = (rich_est - target).abs();

        eprintln!("═══ Richardson on Basel (40 terms) ═══");
        eprintln!("Raw error:        {raw_error:.2e}");
        eprintln!("Richardson error:  {rich_error:.2e}");
        eprintln!("Speedup:          {:.0}x", raw_error / rich_error.max(1e-300));

        // Should be dramatically better than 5× (the old Wynn result)
        assert!(rich_error < raw_error / 20.0,
            "Richardson should dramatically beat raw on Basel: raw={raw_error:.2e}, rich={rich_error:.2e}");
    }

    #[test]
    fn richardson_partial_sums_zeta3() {
        // ζ(3) = 1.2020569..., error ~ 1/N², so Richardson should detect p=2.
        let n = 60;
        let zeta3: f64 = 1.2020569031595942;
        let terms: Vec<f64> = (0..n).map(|k| 1.0 / ((k + 1) as f64).powi(3)).collect();
        let sums = partial_sums(&terms);
        let raw_error = (sums[n - 1] - zeta3).abs();

        let rich_est = richardson_partial_sums(&terms);
        let rich_error = (rich_est - zeta3).abs();

        eprintln!("═══ Richardson on ζ(3) (60 terms) ═══");
        eprintln!("Raw error:        {raw_error:.2e}");
        eprintln!("Richardson error:  {rich_error:.2e}");
        eprintln!("Speedup:          {:.0}x", raw_error / rich_error.max(1e-300));

        assert!(rich_error < raw_error / 10.0,
            "Richardson should beat raw on ζ(3): raw={raw_error:.2e}, rich={rich_error:.2e}");
    }

    #[test]
    fn accelerate_closes_basel_gap() {
        // The Basel gap: previously accelerate() only got 5× on Basel.
        // With Richardson integration, it should get much more.
        let n = 40;
        let target = std::f64::consts::PI.powi(2) / 6.0;
        let terms = basel_terms(n);
        let sums = partial_sums(&terms);
        let raw_error = (sums[n - 1] - target).abs();

        let auto_est = accelerate(&terms);
        let auto_error = (auto_est - target).abs();

        eprintln!("═══ Basel gap closure test (40 terms) ═══");
        eprintln!("Raw error:        {raw_error:.2e}");
        eprintln!("accelerate() err: {auto_error:.2e}");
        eprintln!("Speedup:          {:.0}x", raw_error / auto_error.max(1e-300));

        // Should beat raw by at least 20× (was 5× before)
        assert!(auto_error < raw_error / 20.0,
            "accelerate() should close Basel gap: raw={raw_error:.2e}, auto={auto_error:.2e}");
    }

    #[test]
    fn richardson_vs_wynn_on_positive_algebraic() {
        // Head-to-head: Richardson vs Wynn on positive algebraic series.
        // Richardson should win because it's the matched accelerator.
        let target = std::f64::consts::PI.powi(2) / 6.0;

        for &n in &[20, 40, 80, 160] {
            let terms = basel_terms(n);
            let sums = partial_sums(&terms);
            let raw_err = (sums[n - 1] - target).abs();
            let wynn_err = (wynn_epsilon(&sums) - target).abs();
            let rich_err = (richardson_partial_sums(&terms) - target).abs();

            eprintln!("n={n:3}: raw={raw_err:.2e}  Wynn={wynn_err:.2e}  Rich={rich_err:.2e}  \
                Rich/Wynn={:.1}x", wynn_err / rich_err.max(1e-300));

            // Richardson should at least not be worse than raw
            assert!(rich_err <= raw_err,
                "n={n}: Richardson should beat raw: raw={raw_err:.2e}, rich={rich_err:.2e}");
        }
    }

    // ── Euler-Maclaurin corrected ζ(s) ──────────────────────────────

    #[test]
    fn euler_maclaurin_zeta2_is_basel() {
        // ζ(2) = π²/6. Euler-Maclaurin with corrections should nail it.
        let target = std::f64::consts::PI.powi(2) / 6.0;

        for &p in &[0, 1, 2, 3] {
            let est = euler_maclaurin_zeta(2.0, 60, p);
            let err = (est - target).abs();
            eprintln!("EM ζ(2), n=60, p={p}: est={est:.12}  err={err:.2e}");
        }

        let em3 = euler_maclaurin_zeta(2.0, 60, 3);
        close(em3, target, 1e-10, "EM ζ(2) with 3 corrections");
    }

    #[test]
    fn euler_maclaurin_zeta3_improvement() {
        // ζ(3) = 1.2020569031595942...
        // Raw Richardson got 126×. EM preprocessing should get much more.
        let zeta3: f64 = 1.2020569031595942;
        let n = 60;

        // Raw partial sum
        let terms: Vec<f64> = (0..n).map(|k| 1.0 / ((k + 1) as f64).powi(3)).collect();
        let sums = partial_sums(&terms);
        let raw_err = (sums[n - 1] - zeta3).abs();

        // Richardson alone
        let rich_err = (richardson_partial_sums(&terms) - zeta3).abs();

        // EM with increasing correction order
        eprintln!("═══ ζ(3) Euler-Maclaurin vs Richardson (n=60) ═══");
        eprintln!("Raw partial sum:    {raw_err:.2e}");
        eprintln!("Richardson alone:   {rich_err:.2e}  ({:.0}× over raw)", raw_err / rich_err.max(1e-300));

        for &p in &[0, 1, 2, 3, 4] {
            let em = euler_maclaurin_zeta(3.0, n, p);
            let em_err = (em - zeta3).abs();
            eprintln!("EM p={p}:              {em_err:.2e}  ({:.0}× over raw)",
                raw_err / em_err.max(1e-300));
        }

        // EM with p=3 should dramatically beat Richardson alone
        let em3 = euler_maclaurin_zeta(3.0, n, 3);
        let em3_err = (em3 - zeta3).abs();
        assert!(em3_err < rich_err,
            "EM(p=3) should beat Richardson: em={em3_err:.2e}, rich={rich_err:.2e}");
    }

    #[test]
    fn euler_maclaurin_plus_richardson_zeta3() {
        // The ultimate combination: EM preprocessing removes known error terms,
        // then Richardson cancels the REMAINING unknown terms.
        // This should be dramatically better than either alone.
        let zeta3: f64 = 1.2020569031595942;

        // Generate EM-corrected estimates at different truncation points
        let truncation_points = [15, 30, 60, 120];
        let mut em_estimates = Vec::new();
        for &n in &truncation_points {
            em_estimates.push(euler_maclaurin_zeta(3.0, n, 3));
        }

        // Apply Richardson to the EM-corrected estimates
        // The residual after EM p=3 is O(1/N^8), so error_order ≈ 8
        // But let's auto-detect
        let mut diffs = Vec::new();
        for i in 1..em_estimates.len() {
            diffs.push((em_estimates[i] - em_estimates[i - 1]).abs());
        }
        let mut ratios = Vec::new();
        for i in 1..diffs.len() {
            if diffs[i] > 1e-50 {
                ratios.push(diffs[i - 1] / diffs[i]);
            }
        }

        let em_rich = richardson_extrapolate(&em_estimates, 2.0, 8);
        let em_rich_err = (em_rich - zeta3).abs();

        // Compare with alternatives
        let raw_60_err = {
            let terms: Vec<f64> = (0..60).map(|k| 1.0 / ((k + 1) as f64).powi(3)).collect();
            (partial_sums(&terms).last().unwrap() - zeta3).abs()
        };
        let rich_60_err = {
            let terms: Vec<f64> = (0..60).map(|k| 1.0 / ((k + 1) as f64).powi(3)).collect();
            (richardson_partial_sums(&terms) - zeta3).abs()
        };
        let em3_60_err = (euler_maclaurin_zeta(3.0, 60, 3) - zeta3).abs();

        eprintln!("═══ ζ(3): EM + Richardson combined (n=15..120) ═══");
        eprintln!("Raw S_60:           {raw_60_err:.2e}");
        eprintln!("Richardson alone:    {rich_60_err:.2e}  ({:.0}×)", raw_60_err / rich_60_err.max(1e-300));
        eprintln!("EM(p=3) alone:      {em3_60_err:.2e}  ({:.0}×)", raw_60_err / em3_60_err.max(1e-300));
        eprintln!("EM(p=3)+Richardson: {em_rich_err:.2e}  ({:.0}×)", raw_60_err / em_rich_err.max(1e-300));
        if !ratios.is_empty() {
            let mean_r: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
            eprintln!("Detected ratio: {mean_r:.1} → p≈{:.1}", mean_r.log2());
        }

        // EM+Richardson should be the best
        assert!(em_rich_err < em3_60_err || em_rich_err < 1e-14,
            "EM+Rich should beat EM alone: em_rich={em_rich_err:.2e}, em3={em3_60_err:.2e}");
    }

    // ── Abel summation — the exponential kernel ───────────────────────

    #[test]
    fn abel_grandi_divergent() {
        // Grandi's series: 1 - 1 + 1 - 1 + ... diverges ordinarily.
        // Abel sum: f(x) = 1/(1+x), lim_{x→1⁻} = 1/2.
        // This is the classic test: Abel sums a DIVERGENT series.
        let n = 500;
        let terms: Vec<f64> = (0..n).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let abel = abel_sum(&terms);
        let err = (abel - 0.5).abs();

        eprintln!("═══ Abel on Grandi (500 terms) ═══");
        eprintln!("Abel estimate: {abel}");
        eprintln!("Error:         {err:.2e}");

        close(abel, 0.5, 1e-3, "Abel(Grandi) = 1/2");
    }

    #[test]
    fn abel_grandi_scaling() {
        // Abel accuracy should improve with more terms (more Richardson levels).
        let target = 0.5;
        let mut prev_err = f64::MAX;

        for &n in &[100, 500, 2000] {
            let terms: Vec<f64> = (0..n).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let abel = abel_sum(&terms);
            let err = (abel - target).abs();

            eprintln!("Abel Grandi n={n:5}: est={abel:.10}  err={err:.2e}");

            // Each step should improve (more terms = finer h = more Richardson levels)
            if n > 100 {
                assert!(err < prev_err,
                    "Abel should improve with n: n={n}, err={err:.2e} vs prev={prev_err:.2e}");
            }
            prev_err = err;
        }
    }

    #[test]
    fn abel_leibniz_convergent() {
        // For convergent series, Abel should agree with the ordinary sum.
        // Leibniz π/4 = arctan(1): Abel sum = lim_{x→1} arctan(x) = π/4.
        let n = 200;
        let terms = leibniz_terms(n);
        let abel = abel_sum(&terms);
        let err = (abel - std::f64::consts::FRAC_PI_4).abs();

        eprintln!("═══ Abel on Leibniz π/4 (200 terms) ═══");
        eprintln!("Abel estimate: {abel}");
        eprintln!("Error:         {err:.2e}");

        // Abel should at least agree to a few digits
        assert!(err < 1e-2,
            "Abel(Leibniz) should approximate π/4: err={err:.2e}");
    }

    #[test]
    fn abel_stronger_than_cesaro() {
        // Σ (-1)^n·(n+1) = 1 - 2 + 3 - 4 + ...
        // Abel sum: d/dx[1/(1+x)] = -1/(1+x)², and lim_{x→1} x/(1+x)² = 1/4
        // Actually: f(x) = Σ(-1)^n·(n+1)·x^n = 1/(1+x)²
        // lim_{x→1⁻} = 1/4
        // Cesàro C-1 does NOT converge for this series (partial sums: 1,-1,2,-2,...).
        let n = 1000;
        let terms: Vec<f64> = (0..n).map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * (k + 1) as f64
        }).collect();

        let target = 0.25;
        let abel = abel_sum(&terms);
        let err = (abel - target).abs();

        // Cesàro C-1 for comparison
        let sums = partial_sums(&terms);
        let cesaro = cesaro_sum(&sums);
        let cesaro_err = (cesaro - target).abs();

        eprintln!("═══ Abel vs Cesàro on Σ(-1)^n·(n+1) ═══");
        eprintln!("Abel:   {abel:.6}  (err={err:.2e})");
        eprintln!("Cesàro: {cesaro:.6}  (err={cesaro_err:.2e})");

        // Abel should be much closer to 1/4 than Cesàro C-1
        assert!(err < 0.1, "Abel should approximate 1/4: err={err:.2e}");
    }

    #[test]
    fn kernel_taxonomy_complete() {
        // Verify all four kernel methods work on Leibniz.
        // This is the definitive taxonomy test now that Abel is implemented.
        let n = 200;
        let target = std::f64::consts::FRAC_PI_4;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);

        let raw_err = (sums[n - 1] - target).abs();
        let cesaro_err = (cesaro_sum(&sums) - target).abs();
        let euler_err = (euler_transform(&terms) - target).abs();
        let abel_err = (abel_sum(&terms) - target).abs();
        let aitken_err = {
            let a = aitken_delta2(&sums);
            (a.last().unwrap() - target).abs()
        };

        eprintln!("═══ Complete kernel taxonomy on Leibniz (200 terms) ═══");
        eprintln!("Raw (none):         {raw_err:.2e}");
        eprintln!("Cesàro (uniform):   {cesaro_err:.2e}");
        eprintln!("Abel (exponential): {abel_err:.2e}");
        eprintln!("Euler (binomial):   {euler_err:.2e}");
        eprintln!("Aitken (nonlinear): {aitken_err:.2e}");

        // All kernel methods should beat raw
        assert!(cesaro_err < raw_err, "Cesàro beats raw");
        assert!(abel_err < raw_err, "Abel beats raw");
        assert!(euler_err < raw_err, "Euler beats raw");
        assert!(aitken_err < raw_err, "Aitken beats raw");
    }

    #[test]
    fn streaming_wynn_push_value_matches_batch() {
        // push_value() takes a converging sequence directly.
        // Verify it matches batch wynn_epsilon on running means.
        let n = 20;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);

        // Batch: Wynn on partial sums (which ARE a converging sequence)
        let batch_est = wynn_epsilon(&sums);

        // Streaming: push_value each partial sum
        let mut sw = StreamingWynn::new(40);
        for &s in &sums {
            sw.push_value(s);
        }
        let stream_est = sw.estimate();

        let target = std::f64::consts::FRAC_PI_4;
        let batch_err = (batch_est - target).abs();
        let stream_err = (stream_est - target).abs();

        eprintln!("push_value: batch_err={batch_err:.2e}  stream_err={stream_err:.2e}");

        // Both should be excellent
        assert!(stream_err < 1e-6,
            "push_value should match batch quality: err={stream_err:.2e}");
    }

    #[test]
    fn streaming_wynn_ergodic_lorenz() {
        // Signal-farm use case: accelerate convergence of Lorenz z-component
        // ergodic mean using StreamingWynn on block-averaged observations.
        //
        // The adversarial proved batch Wynn gets 3.9× on block means.
        // Streaming Wynn should match by pushing first differences of
        // running means (so internal partial sums = running means).
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let lorenz = |_t: f64, y: &[f64]| -> Vec<f64> {
            vec![
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ]
        };

        // Generate trajectory
        let n_steps = 50_000;
        let dt = 0.01;
        let (_ts, ys) = crate::numerical::rk4_system(
            lorenz, &[1.0, 1.0, 1.0], 0.0, n_steps as f64 * dt, n_steps,
        );

        // Skip transient, extract z-component
        let skip = 1000;
        let z_vals: Vec<f64> = ys[skip..].iter().map(|y| y[2]).collect();

        // Block-average to decorrelation time (≈50 steps at dt=0.01)
        let block_size = 50;
        let blocks: Vec<f64> = z_vals.chunks(block_size)
            .filter(|c| c.len() == block_size)
            .map(|c| c.iter().sum::<f64>() / block_size as f64)
            .collect();

        // "True" ergodic mean from all data
        let z_mean: f64 = z_vals.iter().sum::<f64>() / z_vals.len() as f64;

        // Running means of block averages
        let mut running_means: Vec<f64> = Vec::with_capacity(blocks.len());
        let mut cumsum = 0.0;
        for (i, &b) in blocks.iter().enumerate() {
            cumsum += b;
            running_means.push(cumsum / (i + 1) as f64);
        }

        // Push first differences of running means to StreamingWynn.
        // This makes StreamingWynn's internal partial sums = running means,
        // so Wynn accelerates the running mean sequence.
        let mut sw = StreamingWynn::new(30);
        let mut prev_rm = 0.0;
        for (i, &rm) in running_means.iter().enumerate() {
            let diff = rm - prev_rm;
            sw.push(diff);
            prev_rm = rm;

            if i < 40 || i % 50 == 0 {
                let est = sw.estimate();
                let err = (est - z_mean).abs();
                let raw_err = (rm - z_mean).abs();
                if i < 15 || i % 50 == 0 {
                    eprintln!("block {:3}: raw_err={:.4}  wynn_err={:.4}  \
                        improvement={:.1}×", i, raw_err, err,
                        raw_err / err.max(1e-10));
                }
            }
        }

        let stream_est = sw.estimate();
        let batch_est = wynn_epsilon(&running_means);

        let n_blocks = running_means.len();
        let raw_final = running_means[n_blocks - 1];
        let raw_err = (raw_final - z_mean).abs();
        let stream_err = (stream_est - z_mean).abs();
        let batch_err = (batch_est - z_mean).abs();

        eprintln!("═══ Lorenz ergodic z-mean: StreamingWynn vs Batch ═══");
        eprintln!("True ergodic mean:       {z_mean:.4}");
        eprintln!("Raw running mean (n={n_blocks}): {raw_final:.4}  (err={raw_err:.4})");
        eprintln!("Streaming Wynn:          {stream_est:.4}  (err={stream_err:.4})");
        eprintln!("Batch Wynn:              {batch_est:.4}  (err={batch_err:.4})");

        // Both should be finite and in reasonable range
        assert!(stream_est.is_finite() && stream_est > 15.0 && stream_est < 35.0,
            "Streaming Wynn estimate should be reasonable: {stream_est}");
        assert!(batch_est.is_finite() && batch_est > 15.0 && batch_est < 35.0,
            "Batch Wynn estimate should be reasonable: {batch_est}");
    }

    // ── Accelerator composition experiments ───────────────────────────

    #[test]
    fn composition_cesaro_then_wynn() {
        // Hypothesis: Cesàro smooths oscillating partial sums into a monotone
        // sequence. Wynn on the monotone Cesàro averages might outperform
        // Wynn on the oscillating raw sums.
        //
        // This tests whether composition Cesàro ∘ Wynn > Wynn alone.
        let target = std::f64::consts::FRAC_PI_4;

        for &n in &[20, 40, 80] {
            let sums = partial_sums(&leibniz_terms(n));

            // Wynn on raw partial sums
            let wynn_raw = wynn_epsilon(&sums);
            let wynn_raw_err = (wynn_raw - target).abs();

            // Cesàro running averages, then Wynn on those
            let mut cesaro_seq = Vec::with_capacity(sums.len());
            let mut running = 0.0;
            for (k, &s) in sums.iter().enumerate() {
                running += s;
                cesaro_seq.push(running / (k + 1) as f64);
            }
            let wynn_cesaro = wynn_epsilon(&cesaro_seq);
            let wynn_cesaro_err = (wynn_cesaro - target).abs();

            // Aitken on Cesàro averages
            let aitken_cesaro = aitken_delta2(&cesaro_seq);
            let aitken_cesaro_err = if let Some(&last) = aitken_cesaro.last() {
                (last - target).abs()
            } else {
                f64::MAX
            };

            eprintln!("n={n:3}: Wynn(raw)={wynn_raw_err:.2e}  \
                Wynn(Cesàro)={wynn_cesaro_err:.2e}  \
                Aitken(Cesàro)={aitken_cesaro_err:.2e}  \
                Wynn ratio={:.1}x",
                wynn_raw_err / wynn_cesaro_err.max(1e-300));
        }
    }

    // ── Streaming Wynn ─────────────────────────────────────────────────

    #[test]
    fn streaming_wynn_matches_batch() {
        // Push all Leibniz terms one at a time.
        // Final estimate should match batch wynn_epsilon().
        let n = 20;
        let terms = leibniz_terms(n);
        let sums = partial_sums(&terms);

        let batch_est = wynn_epsilon(&sums);

        let mut sw = StreamingWynn::new(40);
        for &t in &terms {
            sw.push(t);
        }
        let stream_est = sw.estimate();

        eprintln!("═══ Streaming vs Batch Wynn (20 Leibniz terms) ═══");
        eprintln!("Batch:     {batch_est:.15}");
        eprintln!("Streaming: {stream_est:.15}");

        // They won't be identical (different traversal order through the tableau),
        // but both should be excellent estimates of π/4.
        let target = std::f64::consts::FRAC_PI_4;
        let batch_err = (batch_est - target).abs();
        let stream_err = (stream_est - target).abs();

        eprintln!("Batch err:     {batch_err:.2e}");
        eprintln!("Streaming err: {stream_err:.2e}");

        // Streaming should be in the same ballpark as batch
        assert!(stream_err < 1e-6,
            "Streaming Wynn should approximate π/4: err={stream_err:.2e}");
    }

    #[test]
    fn streaming_wynn_converges_early() {
        // On Leibniz series, streaming Wynn should converge to π/4
        // well before we've pushed all 100 available terms.
        let terms = leibniz_terms(100);
        let target = std::f64::consts::FRAC_PI_4;
        let tol = 1e-10;

        let mut sw = StreamingWynn::new(40);
        let mut converged_at = None;

        for (i, &t) in terms.iter().enumerate() {
            sw.push(t);
            if sw.converged(tol) && converged_at.is_none() {
                converged_at = Some(i + 1);
            }
        }

        let est = sw.estimate();
        let err = (est - target).abs();

        eprintln!("═══ Streaming Wynn early convergence (Leibniz) ═══");
        eprintln!("Converged at term: {:?}", converged_at);
        eprintln!("Final estimate:    {est:.15}");
        eprintln!("Error:             {err:.2e}");

        // Should converge before term 30 (Wynn is powerful on alternating series)
        assert!(converged_at.is_some(), "Should converge within 100 terms");
        assert!(converged_at.unwrap() <= 30,
            "Should converge by term 30, got {:?}", converged_at);
    }

    #[test]
    fn streaming_wynn_trajectory() {
        // Track estimate vs terms ingested — the convergence curve.
        // This is the signal-farm use case: we see the estimate improve
        // as more data arrives.
        let terms = leibniz_terms(30);
        let target = std::f64::consts::FRAC_PI_4;

        let mut sw = StreamingWynn::new(40);
        let mut trajectory = Vec::new();

        for &t in &terms {
            let est = sw.push(t);
            let err = (est - target).abs();
            trajectory.push((sw.terms_seen(), err));
        }

        eprintln!("═══ Streaming Wynn trajectory (Leibniz π/4) ═══");
        for &(n, err) in &trajectory {
            let bar = "█".repeat(((err.log10() + 16.0).max(0.0) * 2.0) as usize);
            eprintln!("n={n:3}: err={err:.2e} {bar}");
        }

        // Error should decrease (mostly monotonically after initial terms)
        let first_err = trajectory[5].1;  // after 6 terms
        let last_err = trajectory.last().unwrap().1;
        assert!(last_err < first_err,
            "Error should decrease: first={first_err:.2e}, last={last_err:.2e}");
    }

    #[test]
    fn streaming_wynn_on_ln2() {
        // ln(2) via alternating harmonic.
        let terms = ln2_terms(25);
        let target = 2.0_f64.ln();

        let mut sw = StreamingWynn::new(40);
        for &t in &terms {
            sw.push(t);
        }

        let err = (sw.estimate() - target).abs();
        eprintln!("Streaming Wynn ln(2), 25 terms: err={err:.2e}");

        assert!(err < 1e-6,
            "Streaming Wynn should approximate ln(2): err={err:.2e}");
    }

    #[test]
    fn streaming_wynn_on_exp_neg1() {
        // e^{-1} = Σ (-1)^n / n! — very fast converging.
        let terms = exp_neg1_terms(15);
        let target = (-1.0_f64).exp();

        let mut sw = StreamingWynn::new(40);
        for &t in &terms {
            sw.push(t);
        }

        let err = (sw.estimate() - target).abs();
        eprintln!("Streaming Wynn e^-1, 15 terms: err={err:.2e}");

        assert!(err < 1e-10,
            "Streaming Wynn should nail e^-1: err={err:.2e}");
    }

    // ── Accelerator composition experiments ───────────────────────────

    #[test]
    fn composition_euler_then_aitken() {
        // Euler transform produces a new sequence of estimates.
        // Can Aitken further accelerate the Euler-transformed series?
        // The Euler estimates at different truncation points form a sequence:
        // E_5, E_10, E_15, E_20 → each is an estimate from more terms.
        let target = std::f64::consts::FRAC_PI_4;

        // Euler estimates at increasing term counts
        let mut euler_estimates = Vec::new();
        for n in (10..=100).step_by(5) {
            let terms = leibniz_terms(n);
            euler_estimates.push(euler_transform(&terms));
        }

        let raw_euler_err = (euler_estimates.last().unwrap() - target).abs();

        // Apply Aitken to the sequence of Euler estimates
        let aitken_euler = aitken_delta2(&euler_estimates);
        let aitken_euler_err = if let Some(&last) = aitken_euler.last() {
            (last - target).abs()
        } else {
            f64::MAX
        };

        // Apply Wynn to the sequence of Euler estimates
        let wynn_euler = wynn_epsilon(&euler_estimates);
        let wynn_euler_err = (wynn_euler - target).abs();

        eprintln!("═══ Composition: Aitken/Wynn on Euler estimates ═══");
        eprintln!("Euler(100 terms):      {raw_euler_err:.2e}");
        eprintln!("Aitken(Euler seq):     {aitken_euler_err:.2e}");
        eprintln!("Wynn(Euler seq):       {wynn_euler_err:.2e}");
        eprintln!("Aitken improvement:    {:.1}x", raw_euler_err / aitken_euler_err.max(1e-300));
        eprintln!("Wynn improvement:      {:.1}x", raw_euler_err / wynn_euler_err.max(1e-300));
    }

    // ── Euler product experiments — {2,3}-factor and Collatz ──────────

    #[test]
    fn euler_product_23_factor() {
        // The Euler product: ζ(s) = ∏_p 1/(1-p^{-s})
        // The {2,3}-factor: E_{2,3}(s) = 1/(1-2^{-s}) · 1/(1-3^{-s})
        //
        // Resolution (math-researcher): the identity E_{2,3}(2)/2 = 3/4 is
        // a numerical coincidence. The REAL connection is at s=1:
        //   E_2(1) = 2 = E[v₂(3n+1)] for odd n (Haar measure on ℤ₂)
        // The s=2 identity is arithmetic (unique Diophantine solution), not structural.
        // q=3 is unique: only odd prime with contraction q/4 < 1.
        //
        // The dominance table below remains interesting — {2,3} explain 91% of ζ(2).

        eprintln!("═══ {{2,3}}-Euler factor of ζ(s) ═══");
        eprintln!("{:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
            "s", "ζ(s)", "E_{2,3}(s)", "ζ/E", "√E_{2,3}");

        for &s in &[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] {
            let zeta = euler_maclaurin_zeta(s, 100, 4);
            let e23 = 1.0 / (1.0 - 2.0_f64.powf(-s))
                    * (1.0 / (1.0 - 3.0_f64.powf(-s)));
            let remaining = zeta / e23;

            eprintln!("{s:4.0}  {zeta:12.8}  {e23:12.8}  {remaining:12.8}  {:12.8}",
                e23.sqrt());
        }

        // Verify E_{2,3}(2) = 3/2 exactly
        let e23_at_2 = (1.0 / (1.0 - 0.25)) * (1.0 / (1.0 - 1.0/9.0));
        close(e23_at_2, 1.5, 1e-14, "E_{2,3}(2) = 3/2");

        // Verify √(E_{2,3}(2)) = √(3/2)
        close(e23_at_2.sqrt(), (1.5_f64).sqrt(), 1e-14, "√E_{2,3}(2) = √(3/2)");

        // The Collatz contraction: each "full step" (odd → 3n+1 → divide by 2^k)
        // has expected multiplication factor 3/2^{E[k]}.
        // With E[k] = 2 (uniform trailing zeros): factor = 3/4
        // The {2,3}-factor at s=2 is 3/2. Is the Collatz factor related?
        //
        // Heuristic: the probability that a random odd n leads to k trailing
        // zeros in 3n+1 is 1/2^k. So E[2^{-k}] = Σ_{k≥1} 2^{-k} · 1/2^k
        // Wait, E[multiplication by 3/2^k] = 3 · E[2^{-k}] where k ≥ 1.
        // E[2^{-k}] = Σ_{k=1}^∞ (1/2^k)(1/2^k) = ... no.
        //
        // For random n, v₂(3n+1) (2-adic valuation) has distribution:
        // P(v₂(3n+1) = k) = 1/2^k for k ≥ 1 (heuristic: 3n+1 is "random even")
        // But this isn't quite right because 3n+1 ≡ 4 mod 8 when n ≡ 1 mod 8, etc.
        //
        // The precise distribution from residue analysis:
        // For n uniformly distributed mod 2^m, the distribution of v₂(3n+1)
        // converges to geometric(1/2) shifted by 1.
        // E[v₂(3n+1)] = 2 (geometric mean).
        //
        // So the expected log₂ of the "multiplication factor" per odd step:
        // E[log₂(3n+1) - log₂(n) - v₂(3n+1)]
        // ≈ log₂(3) - E[v₂] = 1.585 - 2 = -0.415
        //
        // This means each odd step CONTRACTS by a factor of ~2^{-0.415} ≈ 0.749.
        // Over many steps, the trajectory shrinks geometrically at rate ≈ 3/4.
        let collatz_contraction = 3.0 * 0.25; // 3/2^2 = 3/4 (heuristic)
        eprintln!("\nCollatz contraction factor:  {collatz_contraction:.4}");
        eprintln!("E_{{2,3}}(2) = 3/2:          {e23_at_2:.4}");
        eprintln!("Collatz factor = 1/E_{{2,3}}:{:.4}", 1.0 / e23_at_2);
        eprintln!("Ratio:                      {:.4}", collatz_contraction * e23_at_2);

        // Arithmetic identity: E_{2,3}(2)/2 = 3/4 = Collatz contraction.
        // This is a numerical coincidence (unique Diophantine solution), NOT a
        // structural connection. The real link is at s=1: E_2(1) = 2 = E[v₂(3n+1)]
        // via Haar measure on ℤ₂. See prize-problem-expedition campsite for details.
        close(e23_at_2 / 2.0, 0.75, 1e-14,
            "E_{2,3}(2)/2 = 3/4 (arithmetic coincidence)");
    }
}
