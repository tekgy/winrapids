# Paper 6: Numerical Stability by Construction — The Centered-Basis Principle

## Target
Numerical computing: TOMS, SISC, or Numerical Algorithms.

## Core Claim
The naive formula E[x²]-E[x]² silently produces wrong answers for common real-world data. We prove this with adversarial test tables, identify 7 instances in a production codebase, and show that centering-first eliminates the entire bug class.

## Outline
1. The naive formula: ubiquitous, taught in textbooks, BROKEN at offset ≥ 1e8
2. Adversarial proof tables: variance at offsets 1e4→1e14 (12% error → negative variance → NaN cascade)
3. Higher moments: kurtosis breaks at offset 1e4 (!), skewness at 1e6
4. The centered-basis fix: RefCentered two-pass, safe through 1e12+
5. Pebay parallel merge formulas for distributed computation
6. The Poincaré boundary: conformal factor amplifies rounding by 4/(1-r²)² (adversarial proof table)
7. V-column confidence: report reliability alongside results (the tambear answer to boundary conditions)
8. The adversarial methodology: how to systematically break numerical code (4 bug classes, grep patterns)

## Evidence
- Adversarial: ALL proof tables (variance destruction gradient, Poincaré conformal, Cholesky Hilbert, SVD κ²)
- Observer: cancellation canary test, gold standard comparisons showing centered matches scipy
- Pathmaker: descriptive.rs (correct) vs hash_scatter.rs (broken) — same codebase, two approaches
- Math researcher: Pebay algorithm documentation, edge case tables
