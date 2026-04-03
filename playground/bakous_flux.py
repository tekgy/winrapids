"""
Bakous xi-Field Flux Equation — PLAYGROUND
Λ_β = xi integral_S Tr(M x grad_tau) dσ

Unverified, unvalidated internet math. We're playing with it
because it forces us to think about surface integrals as accumulates.

The tambear decomposition:
  Λ_β = xi * accumulate(faces, All, "trace(M x grad_tau) * area", Add)

This is Kingdom A: one reduce over mesh faces.
"""

import numpy as np

# ── Build a mesh (deformed sphere) ──────────────────────────

def make_sphere_mesh(n_lat=20, n_lon=30, deform=0.3):
    """Generate a triangle mesh of a deformed sphere."""
    vertices = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            # Base sphere
            r = 1.0
            # Deformation: random bumps via spherical harmonics approximation
            r += deform * np.sin(3*theta) * np.cos(2*phi)
            r += deform * 0.5 * np.sin(5*theta) * np.sin(3*phi)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Triangulate
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + (j + 1) % n_lon
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return vertices, np.array(faces)


# ── Compute the Bakous flux ─────────────────────────────────

def bakous_flux(vertices, faces, xi=1.0, M_func=None, tau_func=None):
    """
    Λ_β = xi integral_S Tr(M x grad_tau) dσ

    Through tambear: accumulate(faces, All, trace_tensor_expr, Add) * xi

    M_func: vertex position → 3×3 matrix (the field M)
    tau_func: vertex position → scalar (the field tau)
    """
    if M_func is None:
        # Default: M = outer product of position with itself (stress-like)
        M_func = lambda p: np.outer(p, p) / (np.linalg.norm(p) + 1e-10)

    if tau_func is None:
        # Default: tau = distance from a point (temperature-like)
        source = np.array([0.5, 0.3, 0.7])
        tau_func = lambda p: np.linalg.norm(p - source)

    total_flux = 0.0
    per_face_flux = []

    for face in faces:
        # Face vertices
        p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        centroid = (p0 + p1 + p2) / 3.0

        # Face area (half cross product magnitude)
        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2.0

        # M at centroid
        M = M_func(centroid)

        # grad_tau approximated by finite differences on the face
        tau0, tau1, tau2 = tau_func(p0), tau_func(p1), tau_func(p2)
        # Gradient in face plane using barycentric coordinates
        # Simplified: use edge differences
        grad_tau = np.array([
            (tau1 - tau0) / (np.linalg.norm(p1 - p0) + 1e-10),
            (tau2 - tau0) / (np.linalg.norm(p2 - p0) + 1e-10),
            (tau2 - tau1) / (np.linalg.norm(p2 - p1) + 1e-10),
        ])
        # Use first 3 components as approximate gradient direction
        grad_tau_3d = grad_tau[0] * edge1 / (np.linalg.norm(edge1) + 1e-10) + \
                      grad_tau[1] * edge2 / (np.linalg.norm(edge2) + 1e-10)

        # M x grad_tau (tensor product: 3×3 x 3×1 → 3×3×3, but Tr collapses it)
        # Tr(M x v) = M @ v (trace of outer product = matrix-vector product)
        # Actually: Tr(A x B) for A∈ℝ^{m×n}, B∈ℝ^{p×q} requires m=p, n=q
        # For M (3×3) x grad_tau (3-vector), interpret as Tr(M) * (grad_tau · n̂)
        # where n̂ is the surface normal — this gives a scalar flux
        n_hat = normal / (np.linalg.norm(normal) + 1e-10)
        integrand = np.trace(M) * np.dot(grad_tau_3d, n_hat) * area

        per_face_flux.append(integrand)
        total_flux += integrand

    return xi * total_flux, np.array(per_face_flux)


# ── Run it ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Bakous xi-Field Flux Equation — PLAYGROUND")
    print("  Λ_β = xi integral_S Tr(M x grad_tau) dσ")
    print("=" * 60)

    vertices, faces = make_sphere_mesh(n_lat=30, n_lon=40)
    print(f"\nMesh: {len(vertices)} vertices, {len(faces)} faces")

    # Sweep xi
    print("\n── xi sweep ──")
    for xi in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        flux, per_face = bakous_flux(vertices, faces, xi=xi)
        print(f"  xi={xi:5.1f}  →  Λ_β = {flux:12.6f}  (max face flux: {np.max(np.abs(per_face)):8.4f})")

    # Different M fields
    print("\n── M field variations ──")

    # M = identity (divergence theorem test)
    flux_id, _ = bakous_flux(vertices, faces, xi=1.0,
                              M_func=lambda p: np.eye(3))
    print(f"  M=I (identity):     Λ_β = {flux_id:12.6f}")

    # M = rotation matrix
    def rotation_M(p):
        c, s = np.cos(p[2]), np.sin(p[2])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    flux_rot, _ = bakous_flux(vertices, faces, xi=1.0, M_func=rotation_M)
    print(f"  M=R_z(z) (rotation): Λ_β = {flux_rot:12.6f}")

    # M = stress tensor (symmetric, positive definite)
    def stress_M(p):
        return np.array([[2+p[0], p[1], 0], [p[1], 3+p[2], p[0]], [0, p[0], 1+p[1]]])
    flux_stress, _ = bakous_flux(vertices, faces, xi=1.0, M_func=stress_M)
    print(f"  M=σ(p) (stress):    Λ_β = {flux_stress:12.6f}")

    # Different tau fields
    print("\n── tau field variations ──")

    flux_r, _ = bakous_flux(vertices, faces, xi=1.0,
                             tau_func=lambda p: np.linalg.norm(p))
    print(f"  tau=|r| (radial):     Λ_β = {flux_r:12.6f}")

    flux_z, _ = bakous_flux(vertices, faces, xi=1.0,
                             tau_func=lambda p: p[2])
    print(f"  tau=z (height):       Λ_β = {flux_z:12.6f}")

    flux_quad, _ = bakous_flux(vertices, faces, xi=1.0,
                                tau_func=lambda p: p[0]**2 + p[1]**2)
    print(f"  tau=x²+y² (cylinder): Λ_β = {flux_quad:12.6f}")

    print("\n── Tambear decomposition ──")
    print("  accumulate(faces, All, 'Tr(M) * dot(grad_tau, normal) * area', Add)")
    print("  Kingdom A. One reduce. Each face is independent.")
    print(f"  {len(faces)} faces × 1 trace × 1 dot × 1 area = {len(faces)} fused_exprs")
    print(f"  On GPU: one kernel launch. Done.")

    print("\n  The exotic physics equation is a simple scatter-add.")
    print("  Tam knows.")
