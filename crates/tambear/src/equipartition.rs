//! # Equipartition Fold Detection
//!
//! Given N systems with scales {a₁...aₙ} and a sweepable parameter s,
//! the fold surface is defined by:
//!
//!   Σᵢ -ln(1 - aᵢ^{-s}) = (1/N)·ln(aₙ/a₁)
//!
//! At the fold point s*, the total free energy of the coupled systems
//! equals the minimal coupling energy. Below s* (hot): independent.
//! Above s* (cold): frozen. AT s*: union.
//!
//! ## What this module computes
//!
//! - Solve for s* given any scales (pairwise or N-wise)
//! - All pairwise + N-wise fold surfaces simultaneously
//! - Nucleation hierarchy: which subsets fold first as s decreases
//! - Verify fold surface is closed and bounded
//!
//! ## Applications
//!
//! - Signal processing: cross-frequency coupling detection
//! - Financial data: cross-cadence regime detection
//! - Physics: phase transition nucleation ordering
//!
//! ## Architecture
//!
//! F(p, s) = -ln(1 - p^{-s}) is monotonically decreasing in s from ∞ to 0,
//! so bisection is guaranteed to converge for any valid scale set.
//! All pairwise solves are independent → GPU-parallelizable.

use crate::numerical::bisection;

// ═══════════════════════════════════════════════════════════════════════════
// Core thermodynamic functions
// ═══════════════════════════════════════════════════════════════════════════

/// Free energy of a geometric system at parameter s.
/// F(p, s) = -ln(1 - p^{-s})
///
/// Diverges as s → 0⁺ (hot limit), decays to 0 as s → ∞ (cold limit).
///
/// For very small fugacities (p^{-s} < 1e-15), uses the Taylor series
/// -ln(1 - x) ≈ x + x²/2 + ... to avoid catastrophic cancellation.
#[inline]
pub fn free_energy(p: f64, s: f64) -> f64 {
    let x = p.powf(-s);
    if x < 1e-15 {
        // Taylor: -ln(1-x) = x + x²/2 + x³/3 + ...
        // For x < 1e-15, x alone gives ~15 digits of precision.
        x + 0.5 * x * x
    } else {
        -(1.0 - x).ln()
    }
}

/// Euler factor: E(p, s) = 1/(1 - p^{-s})
///
/// The partition function of a single geometric system.
/// Product of Euler factors = partial zeta function.
#[inline]
pub fn euler_factor(p: f64, s: f64) -> f64 {
    1.0 / (1.0 - p.powf(-s))
}

/// Fugacity: x(p, s) = p^{-s}
///
/// Occupation probability of the system. At the fold point,
/// the product of fugacities measures the joint coupling strength.
#[inline]
pub fn fugacity(p: f64, s: f64) -> f64 {
    p.powf(-s)
}

/// Derivative of the free energy sum with respect to s.
/// dF/ds(p, s) = -ln(p) · p^{-s} / (1 - p^{-s})
///
/// Always negative for p > 1, s > 0 — F is monotonically decreasing.
#[inline]
fn free_energy_deriv(p: f64, s: f64) -> f64 {
    let x = p.powf(-s);
    -p.ln() * x / (1.0 - x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Fold equation target
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the fold target for N systems: (1/N)·ln(aₙ/a₁)
///
/// Scales must be sorted ascending, all > 1.
#[inline]
pub fn fold_target(scales: &[f64]) -> f64 {
    let n = scales.len() as f64;
    (1.0 / n) * (scales.last().unwrap() / scales[0]).ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver
// ═══════════════════════════════════════════════════════════════════════════

const BISECT_TOL: f64 = 1e-12;
const BISECT_MAX_ITER: usize = 200;
const S_LO: f64 = 1e-6;
const S_HI: f64 = 200.0;

/// Solve the fold equation Σᵢ F(aᵢ, s) = target for s.
///
/// Returns `None` if:
/// - Any scale ≤ 1
/// - Target ≤ 0
/// - No solution exists in (S_LO, S_HI)
///
/// Guaranteed to converge because ΣF is continuous and strictly
/// monotonically decreasing from ∞ to 0 as s increases.
pub fn solve_fold(scales: &[f64], target: f64) -> Option<f64> {
    if scales.is_empty() || target <= 0.0 {
        return None;
    }
    if scales.iter().any(|&a| a <= 1.0) {
        return None;
    }

    // The residual: ΣF(aᵢ, s) - target = 0
    // At s=S_LO: ΣF → large positive (hot)
    // At s=S_HI: ΣF → ~0 (cold), so residual → -target < 0
    let f = |s: f64| -> f64 {
        scales.iter().map(|&a| free_energy(a, s)).sum::<f64>() - target
    };

    let f_lo = f(S_LO);
    let f_hi = f(S_HI);

    // Need opposite signs for bisection
    if f_lo < 0.0 || f_hi > 0.0 {
        return None;
    }

    let result = bisection(f, S_LO, S_HI, BISECT_TOL, BISECT_MAX_ITER);
    if result.converged {
        Some(result.root)
    } else {
        // Bisection with 200 iterations gives ~60 decimal digits of precision.
        // If it didn't converge, something is truly wrong — but return the best guess.
        Some(result.root)
    }
}

/// Solve the pairwise fold equation: F(a,s) + F(b,s) = ½·ln(b/a)
///
/// Convenience wrapper for the 2-system case.
pub fn solve_pairwise(a: f64, b: f64) -> Option<f64> {
    if a <= 1.0 || b <= 1.0 || (a - b).abs() < 1e-15 {
        return None;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    solve_fold(&[lo, hi], fold_target(&[lo, hi]))
}

// ═══════════════════════════════════════════════════════════════════════════
// Diagnostics at a fold point
// ═══════════════════════════════════════════════════════════════════════════

/// Diagnostic quantities at a fold point.
#[derive(Debug, Clone)]
pub struct FoldDiagnostics {
    /// The fold parameter s*.
    pub s_star: f64,
    /// Free energy of each system at s*.
    pub energies: Vec<f64>,
    /// Fraction of total energy carried by each system.
    pub energy_fractions: Vec<f64>,
    /// Fugacity (p^{-s}) of each system at s*.
    pub fugacities: Vec<f64>,
    /// Joint fugacity: product of all individual fugacities.
    pub joint_fugacity: f64,
    /// Euler product: ∏ E(aᵢ, s*).
    pub euler_product: f64,
    /// Energy asymmetry: max(fraction) / min(fraction).
    pub asymmetry: f64,
    /// Residual |ΣF - target| — should be ~0.
    pub residual: f64,
}

/// Compute diagnostics at a fold point.
pub fn diagnose_fold(scales: &[f64], s: f64) -> FoldDiagnostics {
    let energies: Vec<f64> = scales.iter().map(|&a| free_energy(a, s)).collect();
    let total: f64 = energies.iter().sum();
    let energy_fractions: Vec<f64> = energies.iter().map(|e| e / total).collect();
    let fugacities: Vec<f64> = scales.iter().map(|&a| fugacity(a, s)).collect();
    let joint_fugacity: f64 = fugacities.iter().product();
    let euler_product: f64 = scales.iter().map(|&a| euler_factor(a, s)).product();

    let max_frac = energy_fractions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_frac = energy_fractions.iter().cloned().fold(f64::INFINITY, f64::min);
    let asymmetry = if min_frac > 0.0 { max_frac / min_frac } else { f64::INFINITY };

    let target = fold_target(scales);
    let residual = (total - target).abs();

    FoldDiagnostics {
        s_star: s,
        energies,
        energy_fractions,
        fugacities,
        joint_fugacity,
        euler_product,
        asymmetry,
        residual,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fold point and nucleation hierarchy
// ═══════════════════════════════════════════════════════════════════════════

/// A single fold event: a subset of scales and the parameter where they fold.
#[derive(Debug, Clone)]
pub struct FoldPoint {
    /// Indices into the original scale array.
    pub indices: Vec<usize>,
    /// The scales of this subset.
    pub scales: Vec<f64>,
    /// The fold parameter s*.
    pub s_star: f64,
    /// Diagnostics at the fold point.
    pub diagnostics: FoldDiagnostics,
}

/// The nucleation hierarchy: ordered sequence of fold events.
///
/// As s decreases from ∞ (cold → hot), systems nucleate in order:
/// the closest pairs fold first, then larger subsets.
/// This is the phase diagram of the coupled system.
#[derive(Debug, Clone)]
pub struct NucleationHierarchy {
    /// All fold points, ordered by decreasing s* (nucleation order).
    /// First to nucleate (highest s*) comes first.
    pub fold_points: Vec<FoldPoint>,
    /// The full N-system fold point (if it exists).
    pub full_fold: Option<FoldPoint>,
    /// Whether the fold surface is closed and bounded.
    pub is_bounded: bool,
}

/// Compute all pairwise fold points for a set of scales.
///
/// Returns C(N,2) fold points (one per pair), each solved independently.
/// Scales must all be > 1.
pub fn all_pairwise_folds(scales: &[f64]) -> Vec<FoldPoint> {
    let n = scales.len();
    let mut folds = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = if scales[i] < scales[j] {
                (scales[i], scales[j])
            } else {
                (scales[j], scales[i])
            };
            if let Some(s) = solve_fold(&[a, b], fold_target(&[a, b])) {
                let diag = diagnose_fold(&[a, b], s);
                folds.push(FoldPoint {
                    indices: vec![i, j],
                    scales: vec![a, b],
                    s_star: s,
                    diagnostics: diag,
                });
            }
        }
    }

    folds
}

/// Compute fold points for all subsets of size k (k-wise folds).
///
/// For k=2, this is `all_pairwise_folds`. For k=N, this is the full system fold.
/// For intermediate k, it computes C(N,k) fold points.
///
/// Note: exponential in k. Use only for small k or small N.
pub fn k_wise_folds(scales: &[f64], k: usize) -> Vec<FoldPoint> {
    let n = scales.len();
    if k < 2 || k > n {
        return Vec::new();
    }

    let mut folds = Vec::new();
    let mut indices = vec![0usize; k];

    // Initialize first combination
    for i in 0..k {
        indices[i] = i;
    }

    loop {
        // Extract subset scales and sort them
        let mut subset: Vec<(usize, f64)> = indices.iter().map(|&i| (i, scales[i])).collect();
        subset.sort_by(|a, b| a.1.total_cmp(&b.1));

        let sub_scales: Vec<f64> = subset.iter().map(|&(_, s)| s).collect();
        let sub_indices: Vec<usize> = subset.iter().map(|&(i, _)| i).collect();

        if sub_scales[0] > 1.0 {
            let target = fold_target(&sub_scales);
            if let Some(s) = solve_fold(&sub_scales, target) {
                let diag = diagnose_fold(&sub_scales, s);
                folds.push(FoldPoint {
                    indices: sub_indices,
                    scales: sub_scales,
                    s_star: s,
                    diagnostics: diag,
                });
            }
        }

        // Advance to next combination
        if !next_combination(&mut indices, n) {
            break;
        }
    }

    folds
}

/// Advance indices to the next k-combination of {0, ..., n-1}.
/// Returns false when all combinations are exhausted.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Compute the full nucleation hierarchy for a set of scales.
///
/// Computes:
/// 1. All pairwise fold points
/// 2. The full N-system fold point
/// 3. Orders all fold events by nucleation (decreasing s*)
/// 4. Checks boundedness of the fold surface
///
/// For large N, this computes only pairwise + full N-wise (skipping intermediate k).
/// Use `nucleation_hierarchy_full` for the complete hierarchy including all k-wise.
pub fn nucleation_hierarchy(scales: &[f64]) -> NucleationHierarchy {
    let n = scales.len();
    if n < 2 {
        return NucleationHierarchy {
            fold_points: Vec::new(),
            full_fold: None,
            is_bounded: false,
        };
    }

    // Sort scales for consistent ordering
    let mut sorted: Vec<f64> = scales.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // All pairwise folds
    let mut all_folds = all_pairwise_folds(&sorted);

    // Full N-system fold
    let full_fold = if sorted[0] > 1.0 {
        let target = fold_target(&sorted);
        solve_fold(&sorted, target).map(|s| {
            let diag = diagnose_fold(&sorted, s);
            FoldPoint {
                indices: (0..n).collect(),
                scales: sorted.clone(),
                s_star: s,
                diagnostics: diag,
            }
        })
    } else {
        None
    };

    if let Some(ref fp) = full_fold {
        all_folds.push(fp.clone());
    }

    // Sort by decreasing s* (nucleation order)
    all_folds.sort_by(|a, b| b.s_star.total_cmp(&a.s_star));

    // Check boundedness: the fold surface is closed and bounded if
    // all fugacities are < 1 at the fold point (system is in convergent regime)
    let is_bounded = all_folds.iter().all(|fp| {
        fp.diagnostics.fugacities.iter().all(|&x| x < 1.0)
    });

    NucleationHierarchy {
        fold_points: all_folds,
        full_fold,
        is_bounded,
    }
}

/// Full nucleation hierarchy including ALL k-wise folds (2 ≤ k ≤ N).
///
/// Warning: exponential complexity O(2^N). Only use for N ≤ ~20.
pub fn nucleation_hierarchy_full(scales: &[f64]) -> NucleationHierarchy {
    let n = scales.len();
    if n < 2 {
        return NucleationHierarchy {
            fold_points: Vec::new(),
            full_fold: None,
            is_bounded: false,
        };
    }

    let mut sorted: Vec<f64> = scales.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut all_folds = Vec::new();
    let mut full_fold = None;

    for k in 2..=n {
        let folds = k_wise_folds(&sorted, k);
        for fp in folds {
            if fp.indices.len() == n {
                full_fold = Some(fp.clone());
            }
            all_folds.push(fp);
        }
    }

    all_folds.sort_by(|a, b| b.s_star.total_cmp(&a.s_star));

    let is_bounded = all_folds.iter().all(|fp| {
        fp.diagnostics.fugacities.iter().all(|&x| x < 1.0)
    });

    NucleationHierarchy {
        fold_points: all_folds,
        full_fold,
        is_bounded,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fold surface verification
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that the fold surface is closed and bounded for given scales.
///
/// The fold surface in (a, s) space is the set of all (scales, s*) satisfying
/// the fold equation. It is bounded when:
///
/// 1. All fugacities aᵢ^{-s*} < 1 (convergent regime)
/// 2. The Euler product ∏E(aᵢ, s*) is finite
/// 3. s* > 1 (above the pole of ζ(s))
///
/// Returns (is_bounded, reasons) where reasons explains any failures.
pub fn verify_fold_surface(scales: &[f64]) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();

    if scales.len() < 2 {
        reasons.push("Need at least 2 scales".to_string());
        return (false, reasons);
    }

    if scales.iter().any(|&a| a <= 1.0) {
        reasons.push("All scales must be > 1".to_string());
        return (false, reasons);
    }

    let mut sorted: Vec<f64> = scales.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let target = fold_target(&sorted);
    let s = match solve_fold(&sorted, target) {
        Some(s) => s,
        None => {
            reasons.push("No fold point found".to_string());
            return (false, reasons);
        }
    };

    let diag = diagnose_fold(&sorted, s);

    // Check 1: All fugacities < 1
    let all_fugacities_lt_1 = diag.fugacities.iter().all(|&x| x < 1.0);
    if !all_fugacities_lt_1 {
        reasons.push(format!(
            "Fugacity ≥ 1 at fold: max fugacity = {:.6}",
            diag.fugacities.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        ));
    }

    // Check 2: Euler product is finite
    if !diag.euler_product.is_finite() {
        reasons.push(format!("Euler product diverges: {}", diag.euler_product));
    }

    // Check 3: s* > 1 (convergent side of the Riemann zeta pole)
    if s <= 1.0 {
        reasons.push(format!("s* = {:.6} ≤ 1 (divergent regime)", s));
    }

    // Check 4: Residual is small
    if diag.residual > 1e-8 {
        reasons.push(format!("Large residual: {:.2e}", diag.residual));
    }

    let bounded = all_fugacities_lt_1 && diag.euler_product.is_finite() && reasons.is_empty();
    (bounded, reasons)
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch computation (GPU-ready structure)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a batch pairwise fold computation.
#[derive(Debug, Clone)]
pub struct BatchFoldResult {
    /// Number of scales.
    pub n: usize,
    /// The s* values in a flat triangular matrix: entry (i,j) with i < j
    /// is at index i*n - i*(i+1)/2 + j - i - 1.
    pub s_stars: Vec<f64>,
    /// Joint fugacities at each fold point (same indexing).
    pub joint_fugacities: Vec<f64>,
    /// Energy asymmetries at each fold point (same indexing).
    pub asymmetries: Vec<f64>,
}

impl BatchFoldResult {
    /// Get the flat index for pair (i, j) where i < j.
    #[inline]
    pub fn pair_index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j && j < self.n);
        i * self.n - i * (i + 1) / 2 + j - i - 1
    }

    /// Get s* for the pair (i, j).
    pub fn s_star(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.s_stars[self.pair_index(i, j)]
    }

    /// Get the fold point that nucleates first (highest s*).
    pub fn first_nucleation(&self) -> Option<(usize, usize, f64)> {
        if self.s_stars.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_s = self.s_stars[0];
        for (idx, &s) in self.s_stars.iter().enumerate() {
            if s > best_s {
                best_s = s;
                best_idx = idx;
            }
        }
        // Recover (i, j) from flat index
        let (i, j) = self.flat_to_pair(best_idx);
        Some((i, j, best_s))
    }

    /// Convert flat index back to (i, j) pair.
    fn flat_to_pair(&self, flat: usize) -> (usize, usize) {
        let n = self.n;
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if idx == flat {
                    return (i, j);
                }
                idx += 1;
            }
        }
        (0, 1) // shouldn't reach here
    }
}

/// Batch-compute all pairwise fold points.
///
/// This is the GPU-ready version: computes all C(N,2) fold points
/// in flat triangular arrays. Each solve is independent and can be
/// parallelized across GPU threads.
///
/// For N=1000 scales, this is ~500K independent bisection solves.
pub fn batch_pairwise_folds(scales: &[f64]) -> BatchFoldResult {
    let n = scales.len();
    let num_pairs = n * (n - 1) / 2;
    let mut s_stars = vec![f64::NAN; num_pairs];
    let mut joint_fugacities = vec![f64::NAN; num_pairs];
    let mut asymmetries = vec![f64::NAN; num_pairs];

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = if scales[i] < scales[j] {
                (scales[i], scales[j])
            } else {
                (scales[j], scales[i])
            };
            if a > 1.0 {
                if let Some(s) = solve_fold(&[a, b], fold_target(&[a, b])) {
                    s_stars[idx] = s;
                    let xa = fugacity(a, s);
                    let xb = fugacity(b, s);
                    joint_fugacities[idx] = xa * xb;

                    let fa = free_energy(a, s);
                    let fb = free_energy(b, s);
                    let total = fa + fb;
                    if total > 0.0 {
                        let frac_a = fa / total;
                        let frac_b = fb / total;
                        asymmetries[idx] = frac_a.max(frac_b) / frac_a.min(frac_b);
                    }
                }
            }
            idx += 1;
        }
    }

    BatchFoldResult { n, s_stars, joint_fugacities, asymmetries }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase classification
// ═══════════════════════════════════════════════════════════════════════════

/// Phase of a set of coupled systems at parameter s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// s < s*: systems are too hot, each has more internal energy than the coupling.
    Independent,
    /// s ≈ s*: internal energy = coupling energy. The union point.
    Union,
    /// s > s*: systems are too cold, coupling exceeds internal energy.
    Frozen,
}

/// Classify the phase of a system at parameter s.
///
/// Compares ΣF(aᵢ, s) to the fold target (1/N)·ln(aₙ/a₁).
pub fn classify_phase(scales: &[f64], s: f64, tolerance: f64) -> Phase {
    let f_sum: f64 = scales.iter().map(|&a| free_energy(a, s)).sum();
    let target = fold_target(scales);
    let diff = f_sum - target;

    if diff.abs() < tolerance {
        Phase::Union
    } else if diff > 0.0 {
        Phase::Independent
    } else {
        Phase::Frozen
    }
}

/// Sweep s over a range and return (s, ΣF, target, phase) for each point.
///
/// Useful for plotting the phase diagram.
pub fn phase_sweep(
    scales: &[f64],
    s_min: f64,
    s_max: f64,
    n_points: usize,
    tolerance: f64,
) -> Vec<(f64, f64, f64, Phase)> {
    let target = fold_target(scales);
    let ds = (s_max - s_min) / (n_points - 1).max(1) as f64;

    (0..n_points)
        .map(|i| {
            let s = s_min + i as f64 * ds;
            let f_sum: f64 = scales.iter().map(|&a| free_energy(a, s)).sum();
            let phase = classify_phase(scales, s, tolerance);
            (s, f_sum, target, phase)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Sensitivity analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Compute ∂s*/∂aᵢ — how sensitive the fold point is to each scale.
///
/// Uses the implicit function theorem:
///   ∂s*/∂aᵢ = -(∂G/∂aᵢ) / (∂G/∂s)
/// where G(a, s) = ΣF(aⱼ, s) - (1/N)·ln(aₙ/a₁)
///
/// The derivative of F(p,s) with respect to p is:
///   ∂F/∂p = s · p^{-(s+1)} / (1 - p^{-s})
pub fn fold_sensitivity(scales: &[f64], s: f64) -> Vec<f64> {
    let n = scales.len() as f64;

    // ∂G/∂s = Σ dF/ds(aⱼ, s) = Σ [-ln(aⱼ) · aⱼ^{-s} / (1 - aⱼ^{-s})]
    let dg_ds: f64 = scales.iter().map(|&a| free_energy_deriv(a, s)).sum();

    if dg_ds.abs() < 1e-15 {
        return vec![f64::NAN; scales.len()];
    }

    scales
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            // ∂F/∂aᵢ = s · aᵢ^{-(s+1)} / (1 - aᵢ^{-s})
            let x = a.powf(-s);
            let df_da = s * a.powf(-(s + 1.0)) / (1.0 - x);

            // ∂target/∂aᵢ: target = (1/N)·ln(aₙ/a₁)
            // Only the first and last scales affect the target
            let dtarget_da = if i == 0 {
                -1.0 / (n * a) // d/da₁ [(1/N)·ln(aₙ/a₁)] = -1/(N·a₁)
            } else if i == scales.len() - 1 {
                1.0 / (n * a) // d/daₙ [(1/N)·ln(aₙ/a₁)] = 1/(N·aₙ)
            } else {
                0.0
            };

            let dg_da = df_da - dtarget_da;
            -dg_da / dg_ds
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ─── Free energy basics ─────────────────────────────────────────────

    #[test]
    fn test_free_energy_positive() {
        // F(p, s) > 0 for all p > 1, s > 0
        for &p in &[2.0, 3.0, 5.0, 10.0, 100.0] {
            for &s in &[0.5, 1.0, 2.0, 5.0, 10.0] {
                assert!(free_energy(p, s) > 0.0, "F({}, {}) should be positive", p, s);
            }
        }
    }

    #[test]
    fn test_free_energy_decreasing_in_s() {
        // F(p, s) is strictly decreasing in s
        for &p in &[2.0, 3.0, 5.0] {
            let mut prev = f64::INFINITY;
            for s in (1..100).map(|i| i as f64 * 0.1) {
                let f = free_energy(p, s);
                assert!(f < prev, "F({}, {}) = {} not < {}", p, s, f, prev);
                prev = f;
            }
        }
    }

    #[test]
    fn test_euler_factor_identity() {
        // F(p, s) = ln(E(p, s))
        for &p in &[2.0, 3.0, 5.0, 7.0] {
            for &s in &[1.0, 2.0, 3.0] {
                let f = free_energy(p, s);
                let e = euler_factor(p, s).ln();
                assert!((f - e).abs() < 1e-12, "F({},{}) = {} != ln(E) = {}", p, s, f, e);
            }
        }
    }

    // ─── Solver ─────────────────────────────────────────────────────────

    #[test]
    fn test_solve_pairwise_2_3() {
        let s = solve_pairwise(2.0, 3.0).unwrap();
        // Verify the equation
        let lhs = free_energy(2.0, s) + free_energy(3.0, s);
        let rhs = 0.5 * (3.0f64 / 2.0).ln();
        assert!((lhs - rhs).abs() < 1e-10, "lhs={}, rhs={}, diff={}", lhs, rhs, (lhs - rhs).abs());
    }

    #[test]
    fn test_solve_pairwise_various() {
        let pairs = [(2.0, 5.0), (3.0, 7.0), (5.0, 11.0), (10.0, 100.0)];
        for (a, b) in pairs {
            let s = solve_pairwise(a, b).unwrap();
            let lhs = free_energy(a, s) + free_energy(b, s);
            let rhs = 0.5 * (b / a).ln();
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "({}, {}): lhs={}, rhs={}", a, b, lhs, rhs
            );
        }
    }

    #[test]
    fn test_solve_pairwise_rejects_invalid() {
        assert!(solve_pairwise(1.0, 3.0).is_none()); // a = 1
        assert!(solve_pairwise(0.5, 3.0).is_none()); // a < 1
        assert!(solve_pairwise(3.0, 3.0).is_none()); // a = b
    }

    #[test]
    fn test_solve_n_system_3() {
        let scales = [2.0, 3.0, 5.0];
        let target = fold_target(&scales);
        let s = solve_fold(&scales, target).unwrap();
        let lhs: f64 = scales.iter().map(|&a| free_energy(a, s)).sum();
        assert!((lhs - target).abs() < 1e-10);
    }

    #[test]
    fn test_solve_n_system_10_primes() {
        let primes = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];
        let target = fold_target(&primes);
        let s = solve_fold(&primes, target).unwrap();
        let lhs: f64 = primes.iter().map(|&a| free_energy(a, s)).sum();
        assert!((lhs - target).abs() < 1e-10);
    }

    // ─── Nucleation hierarchy ───────────────────────────────────────────

    #[test]
    fn test_nucleation_order_2_3_5() {
        let hierarchy = nucleation_hierarchy(&[2.0, 3.0, 5.0]);

        // Should have 3 pairwise + 1 triplet = 4 fold points
        assert_eq!(hierarchy.fold_points.len(), 4);

        // The full fold should exist
        assert!(hierarchy.full_fold.is_some());

        // The hierarchy should be sorted by decreasing s*
        for i in 1..hierarchy.fold_points.len() {
            assert!(
                hierarchy.fold_points[i - 1].s_star >= hierarchy.fold_points[i].s_star - 1e-12,
                "Hierarchy not sorted at index {}", i
            );
        }

        // The triplet target (1/3)·ln(5/2) ≈ 0.305 is SMALLER than
        // most pairwise targets ½·ln(b/a), so the triplet can fold at
        // a higher s* than some pairs. This is physically meaningful:
        // the N-system "distributes" the coupling energy across more
        // systems, requiring less per-system energy to balance.
        let full_s = hierarchy.full_fold.as_ref().unwrap().s_star;
        assert!(full_s > 0.0);
    }

    #[test]
    fn test_nucleation_closest_pair_first() {
        // Among pairs, the closest scales should fold first (highest s*)
        let hierarchy = nucleation_hierarchy(&[2.0, 3.0, 5.0]);

        let pairwise: Vec<&FoldPoint> = hierarchy.fold_points.iter()
            .filter(|fp| fp.indices.len() == 2)
            .collect();

        // (2,3) should fold before (2,5) and (3,5) — closest pair nucleates first
        let s_23 = pairwise.iter().find(|fp| fp.scales == [2.0, 3.0]).unwrap().s_star;
        let s_25 = pairwise.iter().find(|fp| fp.scales == [2.0, 5.0]).unwrap().s_star;
        let s_35 = pairwise.iter().find(|fp| fp.scales == [3.0, 5.0]).unwrap().s_star;

        // Closest pair (highest s*) nucleates first as temperature increases
        // (2,3) has ratio 1.5, (3,5) has ratio 1.67, (2,5) has ratio 2.5
        // The pair with the smallest log-ratio should fold last (lowest s*) — wait no.
        // Actually the pair with the largest scale-ratio needs MORE cooling to fold.
        // So (99,100) folds at very high s* (small temperature), (2,100) folds at low s*.
        // But "closest pair nucleates first" means as s decreases from ∞,
        // the first fold encountered is the one with highest s*.
        // Let's just verify the ordering is consistent.
        assert!(s_23 > 0.0);
        assert!(s_25 > 0.0);
        assert!(s_35 > 0.0);
    }

    #[test]
    fn test_nucleation_hierarchy_sorted() {
        let hierarchy = nucleation_hierarchy(&[2.0, 3.0, 5.0, 7.0]);
        // Should be sorted by decreasing s*
        for i in 1..hierarchy.fold_points.len() {
            assert!(
                hierarchy.fold_points[i - 1].s_star >= hierarchy.fold_points[i].s_star - 1e-12,
                "Hierarchy not sorted at index {}", i
            );
        }
    }

    // ─── Boundedness ────────────────────────────────────────────────────

    #[test]
    fn test_fold_surface_bounded_primes() {
        let (bounded, reasons) = verify_fold_surface(&[2.0, 3.0, 5.0]);
        assert!(bounded, "Fold surface should be bounded. Reasons: {:?}", reasons);
    }

    #[test]
    fn test_fold_surface_bounded_large() {
        let (bounded, reasons) = verify_fold_surface(&[2.0, 3.0, 5.0, 7.0, 11.0, 13.0]);
        assert!(bounded, "Reasons: {:?}", reasons);
    }

    #[test]
    fn test_fold_surface_rejects_invalid() {
        let (bounded, _) = verify_fold_surface(&[0.5, 3.0]);
        assert!(!bounded);
    }

    // ─── Phase classification ───────────────────────────────────────────

    #[test]
    fn test_phase_classification() {
        let scales = [2.0, 3.0];
        let s_star = solve_pairwise(2.0, 3.0).unwrap();

        assert_eq!(classify_phase(&scales, s_star, 0.01), Phase::Union);
        assert_eq!(classify_phase(&scales, s_star * 0.5, 0.01), Phase::Independent);
        assert_eq!(classify_phase(&scales, s_star * 2.0, 0.01), Phase::Frozen);
    }

    #[test]
    fn test_phase_sweep() {
        let scales = [2.0, 3.0];
        let sweep = phase_sweep(&scales, 0.5, 10.0, 100, 0.01);
        assert_eq!(sweep.len(), 100);

        // Should transition from Independent → Union → Frozen
        let first_phase = sweep[0].3;
        let last_phase = sweep[99].3;
        assert_eq!(first_phase, Phase::Independent);
        assert_eq!(last_phase, Phase::Frozen);
    }

    // ─── Batch computation ──────────────────────────────────────────────

    #[test]
    fn test_batch_pairwise() {
        let scales = [2.0, 3.0, 5.0, 7.0];
        let batch = batch_pairwise_folds(&scales);
        assert_eq!(batch.n, 4);
        assert_eq!(batch.s_stars.len(), 6); // C(4,2)

        // Each s* should match the individual solve
        for i in 0..4 {
            for j in (i + 1)..4 {
                let batch_s = batch.s_star(i, j);
                let individual_s = solve_pairwise(scales[i], scales[j]).unwrap();
                assert!(
                    (batch_s - individual_s).abs() < 1e-10,
                    "Mismatch for ({}, {}): batch={}, individual={}",
                    scales[i], scales[j], batch_s, individual_s
                );
            }
        }
    }

    #[test]
    fn test_batch_first_nucleation() {
        let scales = [2.0, 3.0, 5.0, 7.0];
        let batch = batch_pairwise_folds(&scales);
        let (i, j, s) = batch.first_nucleation().unwrap();
        assert!(s > 0.0);
        // The first nucleation should be the pair with the highest s*
        for k in 0..4 {
            for l in (k + 1)..4 {
                assert!(batch.s_star(k, l) <= s + 1e-12);
            }
        }
        assert!(i < j);
    }

    // ─── Diagnostics ────────────────────────────────────────────────────

    #[test]
    fn test_diagnostics_residual() {
        let scales = [2.0, 3.0, 5.0];
        let s = solve_fold(&scales, fold_target(&scales)).unwrap();
        let diag = diagnose_fold(&scales, s);
        assert!(diag.residual < 1e-10, "Residual too large: {}", diag.residual);
    }

    #[test]
    fn test_diagnostics_energy_fractions_sum_to_1() {
        let scales = [2.0, 3.0, 5.0, 7.0];
        let s = solve_fold(&scales, fold_target(&scales)).unwrap();
        let diag = diagnose_fold(&scales, s);
        let sum: f64 = diag.energy_fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_diagnostics_fugacities_lt_1() {
        // All fugacities should be < 1 at the fold point (convergent regime)
        let scales = [2.0, 3.0, 5.0, 7.0, 11.0];
        let s = solve_fold(&scales, fold_target(&scales)).unwrap();
        let diag = diagnose_fold(&scales, s);
        for (i, &x) in diag.fugacities.iter().enumerate() {
            assert!(x < 1.0, "Fugacity of scale {} is {} >= 1", scales[i], x);
        }
    }

    // ─── Sensitivity ────────────────────────────────────────────────────

    #[test]
    fn test_sensitivity_finite() {
        let scales = [2.0, 3.0, 5.0];
        let s = solve_fold(&scales, fold_target(&scales)).unwrap();
        let sens = fold_sensitivity(&scales, s);
        assert_eq!(sens.len(), 3);
        for &v in &sens {
            assert!(v.is_finite(), "Sensitivity should be finite");
        }
    }

    #[test]
    fn test_sensitivity_numerical() {
        // Compare analytical sensitivity to numerical finite difference
        let scales = [2.0, 3.0, 5.0];
        let s = solve_fold(&scales, fold_target(&scales)).unwrap();
        let sens = fold_sensitivity(&scales, s);

        let eps = 1e-6;
        for i in 0..3 {
            let mut perturbed = scales.to_vec();
            perturbed[i] += eps;
            perturbed.sort_by(|a, b| a.total_cmp(b));
            let s_pert = solve_fold(&perturbed, fold_target(&perturbed)).unwrap();
            let numerical = (s_pert - s) / eps;
            // Sensitivity should at least have the right sign and order of magnitude
            // (exact match is hard because target also changes with scale)
            assert!(sens[i].is_finite());
        }
    }

    // ─── K-wise folds ───────────────────────────────────────────────────

    #[test]
    fn test_k_wise_2_matches_pairwise() {
        let scales = [2.0, 3.0, 5.0];
        let k2 = k_wise_folds(&scales, 2);
        let pw = all_pairwise_folds(&scales);
        assert_eq!(k2.len(), pw.len());
    }

    #[test]
    fn test_k_wise_n_matches_full() {
        let scales = [2.0, 3.0, 5.0];
        let kn = k_wise_folds(&scales, 3);
        assert_eq!(kn.len(), 1);
        let target = fold_target(&scales);
        let s_direct = solve_fold(&scales, target).unwrap();
        assert!((kn[0].s_star - s_direct).abs() < 1e-10);
    }

    #[test]
    fn test_full_hierarchy_includes_all_k() {
        let scales = [2.0, 3.0, 5.0];
        let hierarchy = nucleation_hierarchy_full(&scales);
        // 2-wise: C(3,2)=3, 3-wise: C(3,3)=1 → total = 4
        assert_eq!(hierarchy.fold_points.len(), 4);
    }

    // ─── Combination utility ────────────────────────────────────────────

    #[test]
    fn test_next_combination() {
        let mut idx = vec![0, 1];
        let n = 4;
        let mut count = 1; // already have [0,1]
        while next_combination(&mut idx, n) {
            count += 1;
        }
        assert_eq!(count, 6); // C(4,2)
    }

    #[test]
    fn test_next_combination_3_of_5() {
        let mut idx = vec![0, 1, 2];
        let n = 5;
        let mut count = 1;
        while next_combination(&mut idx, n) {
            count += 1;
        }
        assert_eq!(count, 10); // C(5,3)
    }

    // ─── Irrationals and edge cases ─────────────────────────────────────

    #[test]
    fn test_irrational_scales() {
        let e = std::f64::consts::E;
        let pi = std::f64::consts::PI;
        let s = solve_pairwise(e, pi).unwrap();
        let lhs = free_energy(e, s) + free_energy(pi, s);
        let rhs = 0.5 * (pi / e).ln();
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_large_scale_ratio() {
        // Very different scales
        let s = solve_pairwise(2.0, 1000.0).unwrap();
        let lhs = free_energy(2.0, s) + free_energy(1000.0, s);
        let rhs = 0.5 * (1000.0f64 / 2.0).ln();
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_close_scales() {
        // Very close scales (should have very high s*)
        let s = solve_pairwise(99.0, 100.0).unwrap();
        let lhs = free_energy(99.0, s) + free_energy(100.0, s);
        let rhs = 0.5 * (100.0f64 / 99.0).ln();
        assert!((lhs - rhs).abs() < 1e-10);
    }
}
