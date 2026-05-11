"""Analyze whether the kernel_share antisymmetry/transitivity failures
are reachable failure modes or artifacts of degenerate edge cases.

The shape.rs code claims `is_kernel_share_compatible_with` is a partial
order ("conservative kernel-reuse partial order"). The sweep_8_dim_adaptive
test file at U3 describes it as a reuse-decision.

If this relation is used to answer "can I use A's kernel for B's needs?",
then transitivity matters: if the system ever CHAINS decisions (use X's
kernel for Y, and Y's kernel for Z, therefore X's kernel works for Z),
then transitivity violations produce actual wrong code.

If the relation is only queried PAIRWISE (never chained), then the
failures are "just" pointwise wrong.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Static:
    n: int


@dataclass(frozen=True)
class Dynamic:
    pass


@dataclass(frozen=True)
class UpTo:
    bound: int


def kernel_share(a, b) -> bool:
    if isinstance(a, Static) and isinstance(b, Static):
        return a.n == b.n
    if isinstance(a, Static) and isinstance(b, Dynamic):
        return False
    if isinstance(a, Dynamic) and isinstance(b, Dynamic):
        return True
    if isinstance(a, Dynamic) and isinstance(b, Static):
        return False
    if isinstance(a, UpTo) and isinstance(b, UpTo):
        return a.bound >= b.bound
    if isinstance(a, UpTo) and isinstance(b, Dynamic):
        return False
    if isinstance(a, UpTo) and isinstance(b, Static):
        return b.n <= a.bound
    if isinstance(a, Static) and isinstance(b, UpTo):
        return a.n <= b.bound
    if isinstance(a, Dynamic) and isinstance(b, UpTo):
        return False
    raise RuntimeError(f"unreachable: {a}, {b}")


# ============================================================
# Transitivity failure case trace
# ============================================================
print("=" * 70)
print("TRANSITIVITY FAILURE — deep trace")
print("=" * 70)
x = Static(0)
y = UpTo(1)
z = Static(1)

xy = kernel_share(x, y)
yz = kernel_share(y, z)
xz = kernel_share(x, z)
print(f"  x = Static(0), y = UpTo(1), z = Static(1)")
print(f"  kernel_share(x, y) = {xy}  (Static(0)→UpTo(1): n=0 ≤ bound=1, TRUE)")
print(f"  kernel_share(y, z) = {yz}  (UpTo(1)→Static(1): n=1 ≤ bound=1, TRUE)")
print(f"  kernel_share(x, z) = {xz}  (Static(0)→Static(1): n=0 != n=1, FALSE)")
print()
print("  Interpretation: If the sharing infrastructure uses this relation")
print("  to decide 'producer kernel X can serve consumer kernel Z',")
print("  AND it chains — 'X can serve Y, Y can serve Z, therefore X can")
print("  serve Z' — then Static(0)'s kernel would be routed to a Static(1)")
print("  consumer. The Static(0) kernel produces a 0-element buffer; the")
print("  Static(1) consumer expects a 1-element buffer. CORRECTNESS BUG.")
print()

# ============================================================
# Interpretation: the relation is NOT a partial order, but a
# pairwise 'check'. It's still useful, but the "partial order"
# claim in the code is mathematically false.
# ============================================================
print("=" * 70)
print("ANTISYMMETRY FAILURE — deep trace")
print("=" * 70)
x = Static(0)
y = UpTo(5)
xy = kernel_share(x, y)
yx = kernel_share(y, x)
print(f"  x = Static(0), y = UpTo(5)")
print(f"  kernel_share(x, y) = {xy}  (n=0 ≤ bound=5)")
print(f"  kernel_share(y, x) = {yx}  (n=0 ≤ bound=5)")
print(f"  Both True, but x != y. Antisymmetry violated.")
print()
print("  Interpretation: the 'kernel binary from Static(0)' and the")
print("  'kernel binary from UpTo(5)' are structurally different binaries,")
print("  yet the relation marks them mutually substitutable. If a cache")
print("  keyed on 'shapes-up-to-kernel-equivalence' uses the quotient by")
print("  this relation's equivalence closure, distinct kernels collapse")
print("  into one bucket. The result would be wrong kernel selection on")
print("  the 'wrong' side of the bucket.")

# ============================================================
# But: if the relation is ONLY queried pairwise (artifact vs
# consumer, once, per dispatch), not chained, the violations
# above don't manifest as bugs — they're just "this one pairwise
# answer is subtly over-permissive/contradictory."
# ============================================================
print()
print("=" * 70)
print("WHEN DOES THIS ACTUALLY BITE?")
print("=" * 70)
print()
print("  Scenario 1 — PAIRWISE ONLY (probably what the code does today):")
print("    Dispatcher asks 'can this cached kernel serve the request?'")
print("    Static(0) request → asks cache 'is there a kernel compatible")
print("    with Static(0)?' Cache iterates entries, checks each pairwise.")
print("    Finds UpTo(5) entry; kernel_share(UpTo(5), Static(0)) = TRUE.")
print("    Cache returns the UpTo(5) kernel.")
print("    Is this correct? UpTo(5) kernel handles sizes 0..=5. Request")
print("    is size 0. Runs fine. This pairwise answer is OK in isolation.")
print()
print("  Scenario 2 — TRANSITIVE CLOSURE:")
print("    System computes 'equivalence classes' of shapes under this")
print("    relation's reflexive-symmetric-transitive closure, caches by")
print("    class. Classes collapse: Static(0), UpTo(1), Static(1) all in")
print("    one bucket (via transitivity). Dispatcher receives Static(1)")
print("    request, pulls ANY kernel from that bucket — could get the")
print("    Static(0) kernel (produces 0-element buffer). BUG.")
print()
print("  Scenario 3 — SYMMETRIC REUSE:")
print("    System treats 'compatible' as symmetric and re-routes both")
print("    directions. Static(0) ⇄ UpTo(5) swap would let UpTo(5)")
print("    consumer get a Static(0) kernel, which only produces 0")
print("    elements when the consumer might need up to 5. BUG.")
print()
print("  Current code at shape.rs uses this relation pairwise ONLY (via")
print("  is_kernel_share_compatible_with method invocation at cache")
print("  dispatch). No transitive closure. No symmetric-re-routing.")
print("  So the math violations are LATENT — they would manifest if the")
print("  relation were used more aggressively.")
print()
print("  PROPER FIX: rename the method to NOT claim 'partial order' and")
print("  document that it answers a pairwise compatibility question, NOT")
print("  an order relation. Adding:")
print("    /// WARNING: this relation is NOT a partial order —")
print("    /// it is NOT antisymmetric and NOT transitive. It answers")
print("    /// a pairwise 'can producer kernel serve consumer request?'")
print("    /// question. Callers MUST NOT chain decisions through it")
print("    /// (x ≤ y ≤ z does NOT imply x ≤ z).")
print("  OR: restrict the relation to a true partial order by")
print("  redefining the Static/UpTo cross-edges.")
print()

# ============================================================
# PROPOSAL: true partial order version
# ============================================================
print("=" * 70)
print("PROPOSAL — a TRUE partial order for kernel reuse")
print("=" * 70)
print()
print("The current Static-to-UpTo bidirectional edges cause the failures.")
print("A clean partial order that preserves the intent is:")
print()
print("  (a) Static(a) ≤ Static(b) iff a == b")
print("  (b) Dynamic ≤ Dynamic only (Dynamic is incomparable to all else)")
print("  (c) UpTo(a) ≤ UpTo(b) iff a >= b")
print("  (d) Static(a) ≤ UpTo(b) iff a <= b  (Static is 'point-shape';")
print("                                       UpTo(b) kernel covers 0..=b)")
print("  (e) UpTo(b) NOT ≤ Static(a) (block this direction entirely;")
print("                               UpTo's kernel is parameterized and")
print("                               the Static consumer expects a fixed")
print("                               N-iteration kernel)")
print("  (f) Reflexivity: every x ≤ x")
print()
print("The current code has (e) going both directions (UpTo ≤ Static via")
print("the 'n ≤ bound' check), which is what breaks antisymmetry and")
print("transitivity. Removing it — UpTo kernel CANNOT serve Static consumer —")
print("would fix the math. Is that too restrictive?")
print()

def kernel_share_fixed(a, b) -> bool:
    # Same as before but blocks UpTo -> Static.
    if isinstance(a, Static) and isinstance(b, Static):
        return a.n == b.n
    if isinstance(a, Static) and isinstance(b, Dynamic):
        return False
    if isinstance(a, Dynamic) and isinstance(b, Dynamic):
        return True
    if isinstance(a, Dynamic) and isinstance(b, Static):
        return False
    if isinstance(a, UpTo) and isinstance(b, UpTo):
        return a.bound >= b.bound
    if isinstance(a, UpTo) and isinstance(b, Dynamic):
        return False
    if isinstance(a, UpTo) and isinstance(b, Static):
        return False  # CHANGED
    if isinstance(a, Static) and isinstance(b, UpTo):
        return a.n <= b.bound
    if isinstance(a, Dynamic) and isinstance(b, UpTo):
        return False
    raise RuntimeError(f"unreachable: {a}, {b}")


# Re-run axioms on the fixed version
from itertools import product
samples = [
    Static(0), Static(1), Static(512), Static(1024), Static(2048),
    Dynamic(),
    UpTo(0), UpTo(1), UpTo(512), UpTo(1024), UpTo(2048),
]
refl = all(kernel_share_fixed(x, x) for x in samples)
antisym = all(not (kernel_share_fixed(x, y) and kernel_share_fixed(y, x) and x != y)
              for x, y in product(samples, repeat=2))
trans = all(
    (not kernel_share_fixed(x, y)) or (not kernel_share_fixed(y, z))
    or kernel_share_fixed(x, z)
    for x, y, z in product(samples, repeat=3))
print(f"  Fixed version (UpTo ↛ Static blocked):")
print(f"    Reflexive: {refl}")
print(f"    Antisymmetric: {antisym}")
print(f"    Transitive: {trans}")
print()
print("  Impact on real use case: code that dispatches a Static(N) consumer")
print("  against a cache of UpTo kernels loses the ability to find a direct")
print("  'UpTo(N+k) handles my Static(N) request' match. Instead, the caller")
print("  must materialize the UpTo kernel's semantics differently, OR")
print("  Sweep 23's artifact-level `compiled_for` field is queried instead")
print("  of this Shape-level relation (which is what the docstring already")
print("  recommends).")
