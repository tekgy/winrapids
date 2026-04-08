# Challenge 33 — Tridiagonal Solve Is a 3×3 Matrix Prefix Scan

**Date**: 2026-04-06  
**Type A: Representation Challenge — sequential algorithm concealing a parallel structure**  
**Found in**: `interpolation.rs` natural_cubic_spline / clamped_cubic_spline (lines 274-288)

---

## The Traditional Assumption

Cubic spline fitting requires sequential Thomas algorithm (forward elimination + back substitution). GPU parallelization requires a different approach.

## Why It Dissolves

The Thomas forward sweep IS a 3×3 matrix prefix scan in homogeneous coordinates. The algorithm is already a Blelloch scan — it just doesn't know it yet.

---

## The Derivation

The Thomas forward sweep for a tridiagonal system `(a_i, b_i, c_i) x = d_i`:
```
d'_1 = b_1, r'_1 = d_1
d'_i = b_i - (a_i / d'_{i-1}) * c_{i-1}   ← nonlinear
r'_i = d_i - (a_i / d'_{i-1}) * r'_{i-1}   ← nonlinear
```

Nonlinear because of division by `d'_{i-1}`. **Lift to homogeneous coordinates:**

Let `d'_i = p_i / q_i` and `r'_i = s_i / q_i`. Then:
```
p_i = b_i · p_{i-1} − a_i · c_{i-1} · q_{i-1}
s_i = d_i · p_{i-1} − a_i · s_{i-1}
q_i = p_{i-1}   (just lags p by one step)
```

This is a **3×3 linear recurrence**:
```
[p_i]   [b_i,  0,    −a_i·c_{i-1}] [p_{i-1}]
[s_i] = [d_i,  −a_i, 0           ] [s_{i-1}]
[q_i]   [1,    0,    0            ] [q_{i-1}]
```

Matrix multiplication is associative → Blelloch scan applies directly.

---

## Connection to Challenge 32 (Op::AffineCompose)

Challenge 32 adds `Op::AffineCompose` for 2×2 state. This challenge requires 3×3.

Two implementation paths:

**Path A** (simpler): Recognize that the back-substitution step is also a linear recurrence (suffix/reverse scan). Implement both passes using `accumulate(Prefix, ..., Op::MatMul3x3)` and `accumulate(Suffix, ..., Op::AffineCompose)`.

**Path B** (more general): The cyclic reduction algorithm (Wang 1981) fuses both passes into a single O(log n) tree reduction. Requires no auxiliary state — each node carries a 2×2 matrix representing the "contribution" from its subtree.

---

## Impact

Every algorithm using the Thomas algorithm becomes GPU-parallel:

| Algorithm | Where | State Size |
|---|---|---|
| Natural cubic spline | `interpolation.rs:234-302` | 3-vector |
| Clamped cubic spline | `interpolation.rs:305-375` | 3-vector |
| Monotone Hermite | Already local (no tridiagonal needed) | n/a |
| Akima | Already local | n/a |
| 1D FD heat equation | (future) | 3-vector |
| 1D BVP on uniform grid | (future) | 3-vector |
| B-spline fitting with knots | (future) | 3-vector |

**All of these are the same 3×3 matrix prefix scan.** No new kernel types. Just the matrix bigger than `Op::AffineCompose`'s 2×2.

---

## The Degeneration Hierarchy

```
3×3 MatPrefix (Thomas tridiagonal)
     ↓
2×2 AffineCompose (GARCH, EWMA, AR, Kalman)
     ↓
1×1 Scalar (Add, Mul, Max, Min — the current Op enum)
```

The natural extension of challenge 32 is: add the 3×3 case. Or, better: make the matrix size a parameter of `Op::MatMulPrefix(n)` and cover the entire hierarchy uniformly.

---

## The Bonus: B-spline Cox-de Boor Recursion

`bspline_basis` (line 936) uses a recursive formula with shared subproblems:
```
B_{k,p}(x) = w1·B_{k,p-1}(x) + w2·B_{k+1,p-1}(x)
```

This is a binary tree reduction — exactly `accumulate(Windowed, ..., Op::BsplineMix)`. The window size is `p+1` (local support of each basis function). Once there are enough Op variants, B-spline evaluation also becomes a windowed accumulate.

---

## Bonus: IIR Biquad Filter Is the Same 3×3 Matrix Prefix Scan

The biquad filter (`signal_processing.rs:634-645`, Direct Form II Transposed):
```rust
let y = self.b0 * x + z1;
z1 = self.b1 * x - self.a1 * y + z2;
z2 = self.b2 * x - self.a2 * y;
```

State (z1, z2) evolves as:
```
[new_z1]   [-a1, 1] [z1]   [(b1-a1·b0)·x]
[new_z2] = [-a2, 0] [z2] + [(b2-a2·b0)·x]
```

In homogeneous coordinates — identical 3×3 structure:
```
M_t = [[-a1, 1, (b1-a1·b0)·x_t],
        [-a2, 0, (b2-a2·b0)·x_t],
        [0,   0, 1              ]]
```

Thomas algorithm and IIR biquad are the SAME primitive. One `Op::MatMulPrefix(3)` implementation covers both. A Butterworth 8th-order filter (4 cascaded biquads) = 4 sequential prefix scans, or equivalently 1 prefix scan with composed matrices.

Note: Savitzky-Golay IS implemented at `signal_processing.rs:732` as `savgol_filter` (test at line 1420). Correction: the documentation gap in this module is NOT Savitzky-Golay. The documentation gap is spatial.rs claiming "compressed row format" while using adjacency lists — that remains real (challenge 30).

---

## Most Actionable Next Step

After `Op::AffineCompose` (challenge 32) exists:

1. Add `Op::MatMulPrefix(n: usize)` — generalizes to any square matrix size
2. The Thomas forward sweep is `accumulate(Prefix, tridiag_element(i), Op::MatMulPrefix(3))`
3. Back-substitute: `accumulate(Suffix, back_elem(i), Op::AffineCompose)` using homogeneous coords
4. Test against sequential `natural_cubic_spline` — should match to machine precision

The spline coefficients are then parallel-ready for any number of concurrent spline fits — which is exactly what the signal farm needs when fitting hundreds of instruments simultaneously.
