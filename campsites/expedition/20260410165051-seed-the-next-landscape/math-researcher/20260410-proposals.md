# Math-Researcher's Next-Landscape Proposals

Written: 2026-04-10

## From mathematical families still thin

**1. Hilbert transform + analytic signal**
Blocks phase-amplitude coupling, instantaneous frequency, envelope detection.
One FFT-based primitive unlocks an entire signal processing sub-family.
Requires complex output — ties directly to ComplexMat/complex f64 foundation gap.
Build order: ComplexMat → FFT complex output → iFFT → Hilbert (~10 lines).

**2. Escort distribution as universal tilting**
`escort(probs, alpha)` primitive unifies Rényi/power-mean/generalized-dimension.
Potential paper: universal structure of parameterized information measures.
Low implementation cost, high theoretical leverage.

**3. Sinkhorn algorithm**
Entropy-regularized optimal transport. O(n²/eps²) vs O(n³) exact.
Needed for Wasserstein-p (p > 1), Wasserstein barycenters, distribution interpolation.
Kingdom C (iterative fixed-point), converges fast in practice.

**4. Copula families**
Gaussian, Clayton, Gumbel, Frank, t-copula.
Currently zero copula support. Copulas separate marginals from dependence structure.
Natural next step for the financial correlation family.

**5. Riemann zero portrait experiment**
Run the full complexity toolkit on 1000+ zeta zeros.
Compare to market eigenvalue spacings measure by measure (MFDFA, permutation entropy, RQA).
All primitives exist. Need compute time, not new math.
If measures match across multiple dimensions: universality result, not coincidence.

## From gaps noticed during paper verification

**6. Schur decomposition**
Needed for numerically robust matrix_log and matrix functions generally.
Current direct approach less stable for ill-conditioned matrices.
Standard intermediate in numerical linear algebra (LAPACK uses it for matrix_log).

**7. False nearest neighbors promotion**
Full Kennel 1992 spec already written by math-researcher.
Private implementation in family15 is correct.
Just needs extraction to tambear::complexity as a public primitive.
Scout has confirmed this as a Type A extraction — lowest-effort, high-value.

**8. SVD workup (Principle 10)**
Our SVD (Golub-Kahan bidiagonalization + QR iteration) is the most complex factorization.
Foundation for pinv, lstsq, rank, effective_rank.
Needs full Principle 10 workup against LAPACK's dgesvd at multiple scales.
Gold-standard oracle: mpmath at 50+ digits for small cases.
