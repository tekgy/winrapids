"""Verify that the two DimHint partial orders satisfy partial-order axioms.

Two relations on DimHint × DimHint -> bool exist at shape.rs:331 and 413:

1. is_result_share_compatible_with — "producer's RESULT satisfies consumer"
   - Static(a) <= Static(b) iff a == b
   - Static(a) <= Dynamic always
   - Dynamic <= Dynamic
   - Dynamic <= Static never
   - UpTo(a) <= UpTo(b) iff a <= b
   - UpTo(a) <= Dynamic always  (KEY FIX per GAP-UPTO-DYNAMIC)
   - UpTo(a) <= Static never
   - Static(n) <= UpTo(bound) iff n <= bound
   - Dynamic <= UpTo never
   - Anything <= Adaptive always (permissive consumer)
   - Adaptive <= non-Adaptive never (unresolved blocks sharing)

2. is_kernel_share_compatible_with — "producer's KERNEL serves consumer"
   - Static(a) <= Static(b) iff a == b
   - Static(a) <= Dynamic never (kernel may constant-fold N)
   - Dynamic <= Dynamic
   - Dynamic <= Static never
   - UpTo(a) <= UpTo(b) iff a ≥ b  (INVERTED from result-share!)
   - UpTo(a) <= Dynamic never (consumer might exceed bound)
   - UpTo(bound) <= Static(n) iff n <= bound
   - Static(n) <= UpTo(bound): NEVER (constant kernel cannot serve variable consumer)
   - Dynamic <= UpTo never
   - Anything <= Adaptive always
   - Adaptive <= non-Adaptive never

For each relation, verify:
  (i) Reflexivity: for all x, x <= x
  (ii) Antisymmetry: x <= y AND y <= x ⇒ x = y (modulo the Adaptive special case)
  (iii) Transitivity: x <= y AND y <= z ⇒ x <= z

Also verify consistency: the two relations should NOT simultaneously claim
the same reuse is correct if the underlying math contradicts.
"""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Static:
    n: int


@dataclass(frozen=True)
class Dynamic:
    pass


@dataclass(frozen=True)
class UpTo:
    bound: int


@dataclass(frozen=True)
class Adaptive:
    pass


def result_share(a, b) -> bool:
    """is_result_share_compatible_with: can b consume a's result?"""
    # Adaptive consumer permissive
    if isinstance(b, Adaptive):
        return True
    # Adaptive producer blocked
    if isinstance(a, Adaptive):
        return False
    # Static-to-Static
    if isinstance(a, Static) and isinstance(b, Static):
        return a.n == b.n
    # Static-to-Dynamic
    if isinstance(a, Static) and isinstance(b, Dynamic):
        return True
    # Dynamic-to-Dynamic
    if isinstance(a, Dynamic) and isinstance(b, Dynamic):
        return True
    # Dynamic-to-Static
    if isinstance(a, Dynamic) and isinstance(b, Static):
        return False
    # UpTo-to-UpTo
    if isinstance(a, UpTo) and isinstance(b, UpTo):
        return a.bound <= b.bound
    # UpTo-to-Dynamic (KEY FIX)
    if isinstance(a, UpTo) and isinstance(b, Dynamic):
        return True
    # UpTo-to-Static
    if isinstance(a, UpTo) and isinstance(b, Static):
        return False
    # Static-to-UpTo
    if isinstance(a, Static) and isinstance(b, UpTo):
        return a.n <= b.bound
    # Dynamic-to-UpTo
    if isinstance(a, Dynamic) and isinstance(b, UpTo):
        return False
    raise RuntimeError(f"unreachable: {a}, {b}")


def kernel_share(a, b) -> bool:
    """is_kernel_share_compatible_with: can b's consumer use a's kernel directly?"""
    if isinstance(b, Adaptive):
        return True
    if isinstance(a, Adaptive):
        return False
    if isinstance(a, Static) and isinstance(b, Static):
        return a.n == b.n
    if isinstance(a, Static) and isinstance(b, Dynamic):
        return False  # kernel may constant-fold N
    if isinstance(a, Dynamic) and isinstance(b, Dynamic):
        return True
    if isinstance(a, Dynamic) and isinstance(b, Static):
        return False
    if isinstance(a, UpTo) and isinstance(b, UpTo):
        return a.bound >= b.bound  # INVERTED
    if isinstance(a, UpTo) and isinstance(b, Dynamic):
        return False
    if isinstance(a, UpTo) and isinstance(b, Static):
        return b.n <= a.bound
    # Static -> UpTo: a constant-specialized kernel cannot serve a variable-size
    # UpTo consumer — the consumer might send any count up to bound, not just n.
    # (Contrast with result-sharing: a Static result IS usable by an UpTo consumer
    # that will accept exactly n elements. Kernel-sharing is stricter.)
    if isinstance(a, Static) and isinstance(b, UpTo):
        return False
    if isinstance(a, Dynamic) and isinstance(b, UpTo):
        return False
    raise RuntimeError(f"unreachable: {a}, {b}")


def test_axioms(relation, name, samples):
    """Test reflexivity, antisymmetry, transitivity on a sample set."""
    print(f"\n{'=' * 70}")
    print(f"Testing partial-order axioms for: {name}")
    print(f"{'=' * 70}")

    # Reflexivity
    refl_fail = []
    for x in samples:
        if not relation(x, x):
            refl_fail.append(x)
    if refl_fail:
        print(f"  REFLEXIVITY FAILS: {refl_fail}")
    else:
        print(f"  Reflexivity OK on {len(samples)} samples")

    # Antisymmetry: x <= y AND y <= x ⇒ x = y
    # Modulo Adaptive special case: Anything <= Adaptive and Adaptive <= nothing.
    # So (x, Adaptive) where x ≠ Adaptive would have x <= Adaptive but not
    # Adaptive <= x — antisymmetry is not violated. Good.
    antisym_fail = []
    for x, y in product(samples, repeat=2):
        if x == y:
            continue
        if relation(x, y) and relation(y, x):
            antisym_fail.append((x, y))
    if antisym_fail:
        print(f"  ANTISYMMETRY FAILS on {len(antisym_fail)} pairs:")
        for x, y in antisym_fail[:5]:
            print(f"    {x} <= {y} AND {y} <= {x} but they are distinct")
    else:
        print(f"  Antisymmetry OK on {len(samples)**2} pair comparisons")

    # Transitivity: x <= y AND y <= z ⇒ x <= z
    trans_fail = []
    for x, y, z in product(samples, repeat=3):
        if relation(x, y) and relation(y, z):
            if not relation(x, z):
                trans_fail.append((x, y, z))
    if trans_fail:
        print(f"  TRANSITIVITY FAILS on {len(trans_fail)} triples:")
        for x, y, z in trans_fail[:5]:
            print(f"    {x} <= {y} AND {y} <= {z}, but NOT {x} <= {z}")
    else:
        print(f"  Transitivity OK on {len(samples)**3} triple comparisons")

    return not (refl_fail or antisym_fail or trans_fail)


# Sample set covering boundary conditions
samples = [
    Static(0),
    Static(1),
    Static(512),
    Static(1024),
    Static(2048),
    Dynamic(),
    UpTo(0),
    UpTo(1),
    UpTo(512),
    UpTo(1024),
    UpTo(2048),
    Adaptive(),
]

result_ok = test_axioms(result_share, "is_result_share_compatible_with", samples)
kernel_ok = test_axioms(kernel_share, "is_kernel_share_compatible_with", samples)

# ============================================================
# Cross-verify: do the two relations AGREE where they should?
# ============================================================
print(f"\n{'=' * 70}")
print("DIVERGENCES between the two relations")
print(f"{'=' * 70}")
divergences = []
for x, y in product(samples, repeat=2):
    r = result_share(x, y)
    k = kernel_share(x, y)
    if r != k:
        divergences.append((x, y, r, k))

print(f"\n  {len(divergences)} pairs where result_share != kernel_share:")
for x, y, r, k in divergences:
    print(f"    ({x}, {y}): result={r}, kernel={k}")

print()
print("SANITY CHECK: the INVERSE on UpTo-to-UpTo is the signature divergence.")
print("  result_share(UpTo(A), UpTo(B)) iff A <= B")
print("  kernel_share(UpTo(A), UpTo(B)) iff A >= B")
print()
print("Equal iff A == B. So:")
print("  UpTo(512) result-shares UpTo(1024)? ", result_share(UpTo(512), UpTo(1024)))
print("  UpTo(512) kernel-shares UpTo(1024)? ", kernel_share(UpTo(512), UpTo(1024)))
print("  UpTo(1024) result-shares UpTo(512)? ", result_share(UpTo(1024), UpTo(512)))
print("  UpTo(1024) kernel-shares UpTo(512)? ", kernel_share(UpTo(1024), UpTo(512)))

# ============================================================
# Bonus check: does Static(N) <= UpTo(N) hold in both? (boundary)
# ============================================================
print()
print("Static(N) vs UpTo(N) at boundary:")
for n in [0, 1, 512, 1024]:
    s = Static(n)
    u = UpTo(n)
    print(f"  Static({n}) -> UpTo({n}): result={result_share(s, u)}, kernel={kernel_share(s, u)}")
    # Also off-by-one
    u_minus = UpTo(n - 1) if n > 0 else UpTo(0)
    print(f"  Static({n}) -> UpTo({n-1}): result={result_share(s, u_minus)}, kernel={kernel_share(s, u_minus)}")

if result_ok and kernel_ok:
    print(f"\n{'=' * 70}")
    print("CONCLUSION: BOTH relations satisfy partial-order axioms.")
    print("The divergence on UpTo-to-UpTo is INTENTIONAL (two distinct semantics).")
    print("The design is mathematically sound.")
    print(f"{'=' * 70}")
else:
    print(f"\n{'=' * 70}")
    print("CONCLUSION: partial-order axioms VIOLATED — report to aristotle.")
    print(f"{'=' * 70}")
