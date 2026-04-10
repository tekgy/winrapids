# Ten Groupings From Zero: Phase 3 Reconstruction on Grouping Patterns

*2026-04-10 — Aristotle, responding to team-lead directive*

## Summary

10 maximally different approaches to grouping, classified along three axes (structure, determinism, arity). The current 8 patterns are all ALGEBRAIC groupings — group actions on index sets. The missing approaches are topological, probabilistic, adaptive, relational, and stochastic.

## The 10 Approaches

1. **Algebraic** (what we have): All, ByKey, Prefix, Segmented, Windowed, Tiled, Masked. Defined by group actions on indices.
2. **Topological/Tree**: Elements contribute to ancestors. Hierarchical, O(log n) outputs per element. Euler tour reduces to segmented scan but hides structure.
3. **Graph/Network**: Elements contribute to graph neighbors. Irregular topology. GNN, belief propagation, PageRank. The strongest candidate for a NEW first-class grouping.
4. **Probabilistic/Soft**: Elements contribute to ALL outputs with weights. Attention, soft clustering, KDE. Already partially covered by Tiled+SoftmaxWeighted. This IS quantum grouping with decoherence.
5. **Recursive/Fractal**: Self-similar at multiple scales. Hierarchical tiling. Already implicit in Blelloch tree.
6. **Temporal/Causal**: Respects causal structure (light cones, event ordering). IS the Kingdom B/C boundary.
7. **Dual/Fourier**: Group in transform domain, not original domain. FFT+ByKey+IFFT. Multi-resolution analysis.
8. **Adaptive/Data-Dependent**: Grouping determined by data (clustering, changepoint detection). Self-referential = Fock boundary. Correctly externalized to higher layers.
9. **Relational/Join**: Connects two datasets. Tiled is the special case. General joins have variable-size groups. The kingdom number IS the join arity.
10. **Stochastic/Sampling**: Random assignment. Bootstrap, MCMC, SGD mini-batches. Non-deterministic.

## The Informal Theorem

The grouping patterns are in 1-1 correspondence with finitely-presentable group actions on index sets that admit efficient SIMT implementation. This is why there are 8 and not 80. The constraint space is small.

New patterns emerge from: (a) new algebraic structures with SIMT scheduling (graph Laplacian), (b) new hardware with irregular-pattern support (sparse tensor cores), (c) relaxing the O(1) parameter constraint (data-dependent groupings).

## Key Finding

Graph grouping (Approach 3) is the strongest candidate for a new first-class pattern. It has algebraic structure (Laplacian) and a major workload (GNNs). Everything else is either already covered by composition or correctly externalized.

## Correction (from naturalist cross-reference, same session)

**Phyla are `(grouping, op)` pairs, not grouping alone.** FFT and MomentStats both use `All` grouping but different ops (ButterflyMul vs Add) and share no intermediates. The four-menu decomposition has two levels:
- Level 1 (structural identity): `(grouping, op)` = phylum = sharing pattern
- Level 2 (instance parameters): `(addressing, expr)` = specific computation within phylum

**Missing grouping priority** (confirmed by naturalist's codebase evidence): Tree > Graph > Circular > Adaptive. Tree most urgent (Euler tour is most information-destroying). Circular subsumes FFT: `accumulate(data, Circular(n), ComplexExpWeight(k)*x[j], Add)` IS the DFT, and the butterfly is the compiler optimization of this circular accumulate.

## Connection to Team-Lead's Insight

"Discovering new grouping patterns = discovering new algorithms." YES. Each approach above corresponds to an ENTIRE CLASS of algorithms:
- Algebraic → most of classical statistics, linear algebra, signal processing
- Tree → hierarchical methods, segment trees, recursive neural nets
- Graph → GNNs, message passing, network analysis
- Probabilistic → attention, mixture models, kernel methods
- Causal → time series, event systems, physics
- Spectral → multi-resolution, frequency analysis
- Adaptive → clustering, changepoint, regime detection
- Relational → database operations, cross-dataset analysis

The grouping IS the algorithm class. The operator is just the specific formula within the class.

Full analysis: `~/.claude/garden/2026-04-10-ten-groupings-from-zero.md`
