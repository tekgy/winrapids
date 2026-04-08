# Abstract Algebra Design — tambear-native
*2026-04-06, navigator*

## Why this module matters

Abstract algebra provides the foundation for:
- Cryptography (groups over elliptic curves, rings mod n, Galois fields)
- Coding theory (linear codes over GF(2), Reed-Solomon over GF(q))
- Number theory (Euler totient = multiplicative group of ℤ/nℤ)
- Quantum mechanics (symmetry groups, representation theory)
- Signal processing (Fourier transform = characters of ℤ/nℤ)

Without abstract_algebra.rs, all these fields are floating without foundation.

## The tambear-native representation

### Finite groups via Cayley table

```rust
pub struct FiniteGroup {
    n: usize,                        // order
    table: Vec<u32>,                 // Cayley table: table[i*n + j] = i op j
}
```

The Cayley table IS a `gather` operation: `table[i*n + j] = gather(i*n + j, elements)`.

Group composition of two elements: `gather(i*n + j, table)`.

### Group operations as accumulate

```rust
// Is H a subgroup of G?
// Test: for all (h1, h2) in H × H, h1 * h2 in H
accumulate(H × H, All, |h1, h2| table[h1 * n + h2], And)  // closure
accumulate(H, All, |h| inverse(h) in H, And)              // inverse closure

// Coset decomposition: partition G by g * H
accumulate(elements, ByKey(|g| representative_of_coset(g, H)), identity, Collect)

// Orbit of element x under group action
accumulate(generators, Prefix, |x| apply_generator(x), Union)

// Conjugacy classes
accumulate(elements, ByKey(|g| canonical_conjugate(g)), identity, Collect)

// Center Z(G) = {g : gx = xg for all x}
accumulate(elements, Masked(|g| forall(x, commutes(g, x))), identity, Collect)
```

## Module structure

### Part 1: Algebraic structure verification (using proof.rs)

```rust
/// Check axioms and return proof object
pub fn verify_group(op: impl Fn(u32, u32) -> u32, elements: &[u32]) -> Result<ProofOf<Group>, AxiomViolation>
pub fn verify_ring(add: impl Fn(u32, u32) -> u32, mul: impl Fn(u32, u32) -> u32, elements: &[u32]) -> Result<ProofOf<Ring>, AxiomViolation>
pub fn verify_field(add: impl Fn(u32, u32) -> u32, mul: impl Fn(u32, u32) -> u32, elements: &[u32]) -> Result<ProofOf<Field>, AxiomViolation>
```

This is where proof.rs becomes essential. `verify_group` should return a `ProofOf<Group>` that downstream code can require — you can't use group operations without a proof that they're actually a group.

### Part 2: Finite group operations

```rust
pub struct FiniteGroup { /* Cayley table */ }

impl FiniteGroup {
    pub fn cyclic(n: usize) -> Self         // Z/nZ
    pub fn symmetric(n: usize) -> Self      // S_n (permutations)
    pub fn alternating(n: usize) -> Self    // A_n
    pub fn dihedral(n: usize) -> Self       // D_n
    pub fn klein_4() -> Self                // V_4
    
    pub fn order(&self) -> usize
    pub fn compose(&self, a: u32, b: u32) -> u32
    pub fn inverse(&self, a: u32) -> u32
    pub fn identity(&self) -> u32
    
    pub fn is_abelian(&self) -> bool
    pub fn center(&self) -> Vec<u32>
    pub fn subgroups(&self) -> Vec<Vec<u32>>
    pub fn cosets(&self, subgroup: &[u32]) -> Vec<Vec<u32>>
    pub fn conjugacy_classes(&self) -> Vec<Vec<u32>>
    pub fn orbit(&self, element: u32) -> Vec<u32>
    
    // Cayley table as gather: the group operation IS a lookup
    pub fn cayley_table(&self) -> &[u32]
}
```

### Part 3: Polynomial rings and finite fields

```rust
/// Polynomial ring F[x] / (irreducible polynomial)
pub struct GaloisField {
    p: u32,   // characteristic (prime)
    k: u32,   // extension degree
    // GF(p^k) via polynomial representation
}

impl GaloisField {
    pub fn new(p: u32, k: u32) -> Result<Self, Error>  // checks p is prime
    pub fn add(&self, a: u32, b: u32) -> u32
    pub fn mul(&self, a: u32, b: u32) -> u32
    pub fn inv(&self, a: u32) -> Option<u32>
    pub fn pow(&self, a: u32, n: u32) -> u32
    pub fn is_primitive_root(&self, a: u32) -> bool
    pub fn primitive_root(&self) -> u32
}

// Polynomial operations in GF(p)[x]
pub fn poly_add_gf(a: &[u32], b: &[u32], p: u32) -> Vec<u32>
pub fn poly_mul_gf(a: &[u32], b: &[u32], p: u32) -> Vec<u32>
pub fn poly_gcd(a: &[u32], b: &[u32], p: u32) -> Vec<u32>
pub fn poly_is_irreducible(p: &[u32], char_p: u32) -> bool
```

### Part 4: Ring operations

```rust
/// Integer ring Z/nZ
pub struct ZnZ {
    n: u64,
}

impl ZnZ {
    pub fn add(&self, a: u64, b: u64) -> u64  { (a + b) % self.n }
    pub fn mul(&self, a: u64, b: u64) -> u64  { (a * b) % self.n }
    pub fn units(&self) -> Vec<u64>             // elements with inverse
    pub fn is_field(&self) -> bool              // iff n is prime
}
```

## Connection to number_theory.rs

The units of Z/nZ form a group. Its order = φ(n) (Euler totient). This directly connects abstract_algebra.rs and number_theory.rs:

```rust
// In a test:
let zn = ZnZ::new(15);
let units = zn.units();
assert_eq!(units.len(), euler_totient(15));  // φ(15) = φ(3)×φ(5) = 2×4 = 8
```

## Connection to proof.rs

The proof.rs `Structure` type already has vocabulary for algebraic structures. The abstract algebra module should use it:

```rust
// proof.rs already has:
// Structure::commutative_monoid(op, identity)
// Structure::group(op, identity, inverse)
// Structure::ring(add, mul, zero, one)

// abstract_algebra.rs extends this:
// FiniteGroup::cyclic(n).verify() -> ProofOf<Group>
// This is the induction-as-prefix-scan pattern from the garden!
```

## The induction connection

From the garden (induction-is-a-prefix-scan): the subgroup tower in group theory IS induction.

If H₀ = {e} ⊂ H₁ ⊂ H₂ ⊂ ... ⊂ G is a composition series, each step is an inductive step: "if H_k has property P, then H_{k+1}/H_k has property Q."

This means `FiniteGroup::subgroups()` + `ProofOf<Group>` + prefix scan = a Lean4-style proof that G has property P.

The abstract algebra module should demonstrate this explicitly.

## Implementation order (by leverage)

1. `FiniteGroup` with cyclic, symmetric, dihedral constructors (tests against known group orders and properties)
2. `ZnZ` (integer ring mod n) — trivial to implement, huge leverage for crypto
3. `GaloisField` (GF(p) first, then GF(p^k)) — needed for coding theory
4. `verify_group`, `verify_ring`, `verify_field` using proof.rs
5. Polynomial rings over GF(p) (for Reed-Solomon)

## Test quality notes

Every test should verify actual algebraic properties:
- `group_has_unique_identity()` — verify identity is unique
- `cyclic_group_n_has_order_n()` — |Z/nZ| = n
- `symmetric_group_3_has_6_elements()` — |S_3| = 3! = 6
- `s3_is_not_abelian()` — S_3 is the smallest non-abelian group
- `gf4_has_characteristic_2()` — GF(4) has char 2
- `units_of_zn_form_group()` — verify multiplicative group structure
- `lagrange_theorem()` — |H| divides |G| for all subgroups H
