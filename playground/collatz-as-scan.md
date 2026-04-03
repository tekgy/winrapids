# Collatz as Affine Scan — The Tambear Approach

## The Insight

The Collatz map on modular classes is an AFFINE transformation.
For each residue class mod 2^k, k steps of the Collatz map produce:

    output = (a · n + b) / 2^c

where (a, b, c) depend only on the k-bit suffix of n.

Affine composition IS associative:
    (a2, b2, c2) ∘ (a1, b1, c1) = (a2·a1, a2·b1+b2, c1+c2)

Therefore: the entire Collatz trajectory IS a prefix scan
over the bit representation of the starting number.

## The Algorithm

1. Precompute: for every k-bit pattern p (0..2^k),
   run k Collatz steps and record the Affine transform (a_p, b_p, c_p).
   This is a lookup table of 2^k entries.

2. Given an n-bit starting value, break it into n/k chunks
   of k bits each: [chunk_0, chunk_1, ..., chunk_{n/k-1}]

3. Look up the Affine transform for each chunk:
   transforms = [T(chunk_0), T(chunk_1), ..., T(chunk_{n/k-1})]

4. Compose all transforms via prefix scan:
   total_transform = accumulate(transforms, Prefix, identity, AffineCompose)

5. Apply the total transform to the starting value:
   result = total_transform(n)

## Complexity

- Precomputation: O(2^k) one-time cost
- Per starting value: O(n/k) lookups + O(n/k) Affine compositions
- For k=16, n=71: ~5 lookups + ~5 compositions = ~10 operations

Compare: standard Collatz on a 71-bit number averages ~400 steps.
Speedup: ~40× from the scan structure alone.

## The Tambear Connection

This is LITERALLY our AffineOp scan from the scan engine.
The same operator that powers EWM, Kalman, ARIMA, Adam.
The Collatz map is Kingdom B — a sequential Affine scan over bits.

In .tbs:
    n.bits(chunk_size=16)
      .map(collatz_affine_lookup)
      .scan(AffineCompose)
      .extract_final_value()

## Sharing

Multiple starting values with the same SUFFIX share their initial
Affine compositions. Like how DBSCAN and KNN share the distance matrix.

For all n in [N, N+2^k): the last k bits are the same.
Their trajectories share the last Affine transform.
Precompute it once, share across the block.

## The Backward Tree Connection

The backward Collatz tree inverts the affine map:
given (a, b, c), find all n such that (a·n + b) / 2^c = target.

This is: n = (target · 2^c - b) / a.

If a divides (target · 2^c - b), there's a predecessor.
This is a MODULAR ARITHMETIC check — exactly what the
hash scatter and filter primitives compute.

## What This Means

The Collatz conjecture verification is NOT embarrassingly parallel.
It's STRUCTURALLY parallel — the map has Affine structure that
our scan primitive already exploits.

Tam doesn't iterate the Collatz map.
Tam SCANS the bit pattern and reads the result.
