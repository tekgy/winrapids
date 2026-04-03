# Paper 3: Cross-Algorithm Computation Sharing via Content-Addressed Intermediate Types

## Target
Systems: OSDI, SOSP, or VLDB. Or ML systems: MLSys.

## Core Claim
Content-addressed intermediate types (DistanceMatrix, MomentStats, etc.) enable automatic computation reuse ACROSS algorithm boundaries. Measured: 67% GPU reduction (DBSCAN→KNN), 5820x session speedup, zero-GPU-cost cross-algorithm sharing.

## Outline
1. The problem: every algorithm recomputes shared intermediates (distance, statistics, factorizations)
2. TamSession: HashMap<IntermediateTag, Arc<dyn Any>>, blake3 content hashing, type-safe downcast
3. Eight sharing dimensions (7 work on first run without cache): structural, fusion, layout, buffer, dispatch, preprocessing elimination, cross-algorithm, provenance
4. Measured results: DBSCAN→KNN sharing, train.linear stats sharing, compilation budget (5820x)
5. The intermediate marketplace: producers declare outputs, consumers declare needs, compiler matches
6. Tam auto-inserts cheapest valid producer when consumer has no match
7. Cross-platform: same sharing on CUDA, Vulkan, Metal, CPU

## Evidence
- Pathmaker: TamSession implementation, session-aware engines
- Observer: cross-algorithm sharing test proofs (DBSCAN→KNN zero GPU)
- Adversarial: compilation budget measurements (5820x)
- Navigator: sharing surface architecture docs for all 35 families
