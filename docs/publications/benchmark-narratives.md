# Five Benchmark Narratives + Fair Methodology

Each narrative is proven by the SAME benchmark suite at scale. One experiment, four papers' worth of evidence.

## Narrative 1: MSR (One Pass > N Passes)
**Paper 2 (MSR Principle)**

The win at 1B isn't "scipy can't allocate." NumPy with enough RAM CAN allocate. The win is: tambear computes mean + variance + skewness + kurtosis + CV + SEM + range + z-scores in ONE scan. scipy computes each separately — 8 scans vs 1 scan. At 1B elements: 8×8GB = 64GB of memory traffic vs 8GB. The MSR one-pass principle is 8× less I/O.

This is not a memory story. It's a SCAN ECONOMY story.

## Narrative 2: Algorithmic Inversion (O(n) Where Others Are O(n²))
**Paper 8 (Five GPU Primitives)**

KDE via FFT convolution = O(n + m log m). sklearn pairwise KDE = O(n²). At n=100M: seconds vs hours. The primitives unlock complexity class inversions. Autocorrelation via Wiener-Khinchin = O(n log n) vs naive O(n²). These inversions aren't clever tricks — they're consequences of having FFT as a primitive.

CHECK: does RAPIDS/cuML have GPU KDE? If so, the comparison is tambear (any GPU) vs cuML (NVIDIA only), not tambear vs sklearn. Different headline.

## Narrative 3: Sharing Wins (N Algorithms For The Cost Of 1)
**Paper 3 (Cross-Algorithm Sharing)**

DBSCAN + KNN + silhouette + Davies-Bouldin: 4 algorithms, 1 distance matrix, 1 GPU pass. Traditional: 4 independent GPU passes. The cross-algorithm sharing via TamSession is genuinely novel — no other framework does this.

This is the ONLY narrative where tambear has zero competitors. RAPIDS doesn't share across algorithms. PyTorch doesn't. The sharing infrastructure IS the moat.

## Narrative 4: Correctness At Scale
**Paper 6 (Numerical Stability)**

At 1B elements with offset data (financial prices ~$100K+): scipy's variance is WRONG. Not slow — WRONG. The naive formula E[x²]-E[x]² produces 12% error at offset 1e6, negative variance at 1e12. tambear's RefCentered is correct through 1e12+.

The benchmark isn't just "faster" — it's "faster AND correct where the competitor is wrong." At the absurd scale, numerical stability isn't optional. It's the difference between right and wrong answers.

## The Benchmark Suite

At each scale (10, 1K, 100K, 1M, 10M, 100M, 1B, 10B):
- **Time**: wall clock for tambear vs scipy vs RAPIDS vs faer
- **Memory**: peak allocation
- **Accuracy**: relative error vs analytical ground truth (for synthetic data with known parameters)
- **Sharing**: time for algorithm A alone vs A+B with session sharing

One suite. Four narratives. Four papers. Each scale tier adds evidence to all four simultaneously.

## Narrative 5: Every Algorithm, Fair Fight, We Win
**Paper 9 or universal evidence section**

The cleanest story. Head-to-head. Same machine. Same data. Competitor gets EVERY advantage (best implementation, optimal settings, warm caches, $1M/yr engineer tuning). Tambear uses the GENERAL PURPOSE implementation — the same .tbs a teenager writes.

If the competitor wins on a specific algorithm: REPORT IT HONESTLY. Then show MSR one-pass gives 4 for the price of 1. Or sharing gives the second algorithm free.

The honesty IS the credibility. Nobody believes "fastest at everything." Everyone believes "competitive individually, dominant collectively."

## Three Benchmark Modes (for EVERY algorithm)

**Mode 1: Fair fight** — tambear's general-purpose vs their expert-optimized best.
This is the main paper evidence. Fair methodology. Competitor gets every advantage.

**Mode 2: Apples to apples** — tambear computing ONLY what they compute.
If numpy.mean is faster than tambear MomentStats (which computes mean+var+skew+kurt),
test tambear.mean_only() against numpy.mean(). Isolate WHERE the overhead is.
Every gap found → optimization backlog entry. Chip away over time.

**Mode 3: Their algorithm, our platform** — implement their exact approach in tambear.
If they have a faster narrow-case algorithm, STEAL IT. Offer as a flavor:
  data.mean()                    # tambear-native MSR (best overall)
  data.mean(flavor="simd")       # hand-tuned narrow case
  data.mean(flavor="numpy")      # their algorithm, our GPU

NO REASON TO LEAVE TAMBEAR FOR ANY COMPUTATION.
Not "we're best at everything today." But "if we're not best today, we will be,
and meanwhile you get sharing, MSR, any GPU, .tbs, science linting that nobody else offers."

Every benchmark loss → optimization target. Losses compound into wins over time.

## MEASURED RESULTS (2026-04-01)

### Descriptive Stats at Scale (CPU, single-threaded)
| Scale | tambear (s) | scipy (s) | Speedup |
|-------|:-----------:|:---------:|:-------:|
| 100K | 0.001 | 0.004 | 7x |
| 1M | 0.007 | 0.060 | 9x |
| 10M | 0.059 | 0.735 | 12.6x |
| 100M | 0.59 | 7.72 | 13.1x |
| **1B** | **6.08** | **79.76** | **13.1x** |

### Linear Algebra vs faer (release, 100×100)
| Operation | tambear | faer | Winner |
|-----------|---------|------|--------|
| QR | 0.11ms | 2.4ms | **tambear 20x** |
| LU | 0.97ms | 25.6ms | **tambear 25x** |
| Cholesky | 0.12ms | 2.7ms | **tambear 25x** |
| SVD (n=10) | 0.015ms | 0.012ms | faer 1.3x |
| SVD (n=200) | 1754ms | 38ms | faer 46x |
| SVD (tall 200×50) | 10ms | 35ms | **tambear 3x** |

## Open Items
- KDE via FFT: ~50 lines to build, infrastructure ready. BUILD ITEM.
- RAPIDS comparison: verified — they're Linux-only, no TamSession, broken KDE at 200K.
- Streaming/chunked accumulate: needed for 10B+ scale. MomentStats.merge() exists but pipeline not wired.
- Mode 2/3 benchmarks: not yet started. Need narrow-case implementations for each algorithm.
