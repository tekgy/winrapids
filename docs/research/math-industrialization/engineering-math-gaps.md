# Engineering Math Gaps — What tambear Doesn't Have

**Date**: 2026-04-10
**Purpose**: Identify mathematical primitives that engineering disciplines need and tambear does not yet implement.

---

## Methodology

Surveyed tambear source via `pub fn` grep across all modules. Cross-referenced against 7 engineering domains. Focus on **cross-cutting primitives** — math that serves multiple disciplines from one implementation.

---

## What tambear already has (relevant to engineering)

| Family | What's there |
|--------|-------------|
| Signal processing | FFT (radix-2), IFFT, RFFT, 2D FFT, STFT, Welch, periodogram, FIR (windowed sinc: LP/HP/BP), IIR biquad cascade, **Butterworth LP/HP** (biquad design), Hann/Hamming/Blackman/Bartlett/Kaiser/Flat-top windows, Hilbert transform, analytic signal, envelope, instantaneous frequency, cepstrum, Morlet CWT, Haar DWT/IDWT, Daubechies-4, EMD (empirical mode decomposition), path signatures, ICA, WVD, Savitzky-Golay, Goertzel, zero-crossing rate, median filter, regularization/resampling |
| Linear algebra | LU, Cholesky, QR, SVD, pseudoinverse, symmetric eigensolver, power iteration, conjugate gradient, GMRES, tridiagonal solver (direct + parallel scan), matrix exp/log/sqrt, condition number, rank |
| State-space / Kalman | Linear Gaussian SSM, Kalman filter (scalar + matrix), RTS smoother, EM for SSM, particle filter, HMM (forward-backward, Viterbi, Baum-Welch) |
| Physics | N-body (Velocity-Verlet), SHO/DHO exact solutions, Kepler orbits, double pendulum (RK4), Ising 1D/2D, Schrödinger 1D (split-step), Euler 1D (Lax-Friedrichs), Poisson SOR, vorticity step, relativistic kinematics, thermodynamics (ideal gas, van der Waals, Carnot, Otto), Boltzmann/partition function |
| Numerical | Bisection, Newton, Secant, Brent, RK4, RK45, adaptive Simpson, Gauss-Legendre, trapezoid, fixed-point, Richardson extrapolation, Brusselator |
| Optimization | Gradient descent variants, BFGS/L-BFGS, CG variants, trust-region, NM, PSO, CMA-ES, Bayesian opt, LP simplex |

---

## Confirmed Gaps by Domain

### 1. Control Systems

**What's missing:**

| Primitive | Why it matters |
|-----------|---------------|
| **Dare / Riccati equation solver** (discrete + continuous) | Core of LQR, LQG, H-infinity synthesis. `state_space.rs` has Kalman but zero mention of Riccati or `dare`/`care`. Every control synthesis method needs this. |
| **LQR gain computation** (`lqr(A, B, Q, R)`) | Optimal state feedback. Requires solving DARE. Missing entirely from `state_space.rs`. |
| **LQG controller** | Combination of LQR + Kalman estimator. Neither synthesized as a primitive. |
| **PID controller** (with tuning rules: Ziegler-Nichols, Cohen-Coon, IMC, relay feedback) | Ubiquitous. Nothing in tambear — not even a simple P/I/D biquad representation. |
| **Pole placement** (Ackermann, Bass-Gura) | State feedback by eigenvalue assignment. Missing. |
| **H-infinity synthesis** (gamma-iteration, LMI formulation) | Robust control. Requires Riccati solver + bisection loop. Missing. |
| **Bode / Nyquist plot data** (frequency response of state-space or transfer function) | Frequency-domain analysis primitive. `signal_processing.rs` has FFT but no H(jω) evaluator for rational transfer functions. |
| **Root locus computation** | Eigenvalue paths as gain varies. Missing. |
| **Model predictive control (MPC)** — QP formulation + solve | Requires QP solver. Optimization has general solvers but no MPC-specific formulation (condensed/sparse horizon, terminal constraint). |
| **Stability metrics**: Lyapunov equation solver (`lyap(A, Q)`, `dlyap`), spectral abscissa, stability margin | Needed for certificate generation. Missing. |
| **Transfer function ↔ state-space conversion** (controllable/observable canonical forms) | Realization theory. Missing. |
| **Discretization** (`c2d`: ZOH, Tustin, Euler, matched) | Converting continuous-time models to discrete. Missing. |
| **Controllability / observability Gramians** | Structural analysis of SSMs. Missing. |

**Cross-cutting note**: Riccati solver is the single highest-value primitive here — it unlocks LQR, LQG, H-infinity, and balanced truncation (model reduction) from one implementation.

---

### 2. Signal Processing

**What's present**: Butterworth (LP/HP biquad cascade). FIR windowed sinc.

**What's missing:**

| Primitive | Why it matters |
|-----------|---------------|
| **Chebyshev Type I/II IIR filter design** | Equiripple passband (Type I) or stopband (Type II). Standard filter-design trio alongside Butterworth. |
| **Elliptic (Cauer) filter design** | Equiripple in both bands. Minimum order for given spec. Used in communications and precision measurement. |
| **Bessel filter design** | Maximally flat group delay. Needed wherever phase linearity matters (audio, medical instruments). |
| **Butterworth HP/BP/BS cascade design** | Only LP and HP biquad exist; no arbitrary-order bandpass/bandstop design path. |
| **Bilinear transform** (s → z) | The primitive underlying analog-prototype-to-digital conversion for all IIR designs. Currently Butterworth is hardcoded; a general bilinear transform unlocks Chebyshev/Elliptic/Bessel without new derivations. |
| **Analog prototype poles** (Butterworth, Chebyshev, Elliptic, Bessel in s-domain) | Intermediate needed by the filter design pipeline. |
| **Frequency warping for bilinear transform** | Pre-warping the corner frequency to correct for frequency axis compression. Missing from current Butterworth implementation too. |
| **Kaiser filter order estimation** (`kaiserord`) | Computes minimum FIR order from ripple/transition-band spec. Missing (Kaiser window exists but not the order estimator). |
| **Polyphase resampling** (rational ratio up/downsampling with anti-alias filter) | The `regularize_*` functions are simple; proper sample-rate conversion requires polyphase filterbank. |
| **Multitaper spectral estimation** (Thomson) | Superior to Welch for short sequences. Missing. |
| **Lomb-Scargle periodogram** | Spectral analysis of unevenly sampled data. Missing from `signal_processing.rs` (exists in `spectral.rs` — verify). |
| **Filter frequency response** (`freqz`, `freqs`) | H(z) at arbitrary frequencies. Missing — no way to evaluate a designed filter's response. |
| **Matched filter** | Cross-correlation with known waveform template. Trivial from existing FFT + cross-correlate, but not named. |

**Cross-cutting note**: The bilinear transform + analog prototype poles is the kernel that unlocks the entire IIR design family. One implementation of `bilinear_transform(poles, zeros, gain, fs)` lets Chebyshev, Elliptic, and Bessel follow immediately.

---

### 3. Structural / Civil Engineering

**What's entirely absent: Finite Element Method**

| Primitive | Why it matters |
|-----------|---------------|
| **Element stiffness matrix** — bar/truss (2-node, 1D/3D), Euler-Bernoulli beam (2-node, 6 DOF), Timoshenko beam (shear-corrected) | Atom of FEM. Every structural analysis needs this. |
| **Element mass matrix** (consistent + lumped) | Needed for dynamic FEM (modal analysis). |
| **Global assembly** (`assemble(K_elements, connectivity)`) | Scatter-accumulate over element contributions. This IS tambear's accumulate primitive — natural fit. |
| **Boundary condition application** (elimination, penalty, Lagrange multiplier) | Required to solve the constrained system. |
| **FEM solve pipeline** (assemble → apply BCs → `K u = f`) | End-to-end structural solve. |
| **Plate element** (Kirchhoff, Mindlin-Reissner) | 2D structural analysis. Needed for shell-like structures. |
| **Shell element** (flat-facet composite of membrane + plate) | 3D thin-walled structures. Superposition of plate + plane-stress stiffness. |
| **Gaussian quadrature on reference elements** | Numerical integration over triangles, quads, tets, hexes. `gauss_legendre_5` is 1D; 2D/3D versions are missing. |
| **Isoparametric mapping** (reference → physical element) | Required for curved elements and non-unit-aspect-ratio meshes. |
| **Seismic response spectrum** (Newmark integration, response spectrum generation) | Structural engineering under dynamic loads. |
| **Modal analysis** (generalized eigenproblem `K v = ω² M v`) | Natural frequencies and mode shapes. Requires generalized eigensolver — `sym_eigen` only handles standard eigenproblem. |
| **Lanczos algorithm** (large sparse symmetric eigensolver) | The practical solver for modal analysis at scale. Missing — `sym_eigen` is dense. |
| **Rayleigh damping** | Classical structural damping model. |
| **Structural reliability indices** (FORM, SORM, Monte Carlo) | Probability of failure calculation. |

**Cross-cutting note**: Gaussian quadrature on 2D/3D reference elements and isoparametric mapping are the two cross-cutting primitives — they underpin FEM for structural, thermal, electromagnetic, and fluid domains simultaneously. The generalized eigensolver (`K v = λ M v`) is the second highest-value primitive here: it unlocks modal analysis, stability analysis in fluid mechanics, and quantum chemistry simultaneously.

**What exists**: Schrödinger 1D solver uses symmetric tridiagonal eigensolver (`sym_tridiag_eigvals`) — this is a start but not the general case.

---

### 4. Electrical Engineering / Circuit Simulation

**What's entirely absent:**

| Primitive | Why it matters |
|-----------|---------------|
| **Stamp functions** — resistor, capacitor, inductor, voltage source, current source, VCVS, CCCS etc. | Each element stamps into the MNA conductance matrix G and RHS vector. The atomic operation of SPICE-like simulation. |
| **Modified Nodal Analysis (MNA) assembly** (`G v = i` setup) | The framework for DC circuit analysis. Requires accumulate over element stamps. |
| **LU factorization with partial pivoting for sparse systems** | The solve at the heart of DC analysis. Dense LU exists; sparse LU does not. |
| **Newton-Raphson for nonlinear DC** | Iterative solve for circuits with transistors/diodes. Requires Jacobian stamps for nonlinear elements. |
| **Transient analysis** (GEAR/BDF integration, companion models for L/C) | Time-domain simulation. L and C become resistors + sources at each timestep (companion model). |
| **AC small-signal analysis** (complex-valued MNA at each frequency) | Frequency response of circuits. |
| **Power flow (Newton-Raphson AC load flow)** | Power grid analysis. Jacobian of P/Q w.r.t. V/δ. Missing. |
| **Fast decoupled load flow** (simplified Jacobian, alternating P/δ and Q/V) | More practical power flow variant. Missing. |
| **Diode / BJT / MOSFET device models** | Shockley, Ebers-Moll, Level-1 MOSFET. Needed for nonlinear DC/transient. |

**Cross-cutting note**: MNA stamp accumulation is structurally identical to FEM assembly — both are `accumulate(element_contributions, scatter_to_global_matrix)`. One accumulate primitive serves both domains.

---

### 5. Communications Engineering

**What's entirely absent:**

| Primitive | Why it matters |
|-----------|---------------|
| **Constellation mapping/demapping** — BPSK, QPSK, 8-PSK, QAM-16/64/256 | Symbol encoding/decoding. Fundamental to any digital comm chain. |
| **Gray coding/decoding** | Bit-to-symbol mapping that minimizes bit errors at symbol boundaries. |
| **I/Q modulation / demodulation** | Baseband ↔ passband conversion. Uses existing `hilbert` but no named primitive. |
| **OFDM modulator** (IFFT + cyclic prefix) and demodulator (remove CP + FFT) | Builds directly on existing FFT. High-value because it IS just FFT + simple operations. |
| **Channel models**: AWGN, Rayleigh fading, Rician, multipath delay spread | Statistical channel simulation for BER analysis. |
| **BER/SER computation** — closed-form and Monte Carlo | Performance metric for comm systems. Requires Q-function (`erfc` exists as special function). |
| **Viterbi decoder** for convolutional codes | Soft-decision decoding. Trellis search — different from HMM Viterbi (though structurally related). |
| **Reed-Solomon encoder/decoder** | Finite-field polynomial arithmetic. Galois field GF(2^m) multiply/divide. |
| **LDPC decoder** (belief propagation / sum-product) | Factor graph message passing. Missing. |
| **Turbo decoder** (iterative BCJR / MAP) | Builds on BCJR algorithm (generalization of forward-backward). Structurally related to HMM. |
| **Polar code encoder/decoder** (successive cancellation) | 5G standard. Recursive structure — natural accumulate form. |
| **Channel equalizers**: ZF, MMSE, decision-feedback | Linear/nonlinear equalization to undo ISI. |
| **Spread spectrum**: PN sequence generation, correlation receiver, CDMA | Pseudorandom sequences over GF(2). |
| **CRC computation** | Polynomial division over GF(2). Widely used across domains (storage, networking). |
| **Galois field GF(2^m) arithmetic** | The atom underlying RS codes, CRC, LDPC, and all algebraic coding. |

**Cross-cutting note**: GF(2^m) arithmetic (multiply, divide, log, exp via lookup) is the single primitive that unlocks RS, CRC, BCH, and Reed-Muller codes from one implementation. OFDM demodulation is just FFT + CP removal — essentially free given existing FFT.

---

### 6. Robotics

**What's absent:**

| Primitive | Why it matters |
|-----------|---------------|
| **Homogeneous transformation matrices** (SE(3): rotation + translation as 4×4) | The atom of robotics geometry. Every kinematics computation uses this. |
| **SO(3) / SE(3) exponential map** (axis-angle → rotation matrix, twist → transform) | Lie group operations for smooth interpolation. |
| **Denavit-Hartenberg (DH) parameter table → forward kinematics chain** | Standard robot description. Compose 4×4 transforms down the kinematic chain. |
| **Jacobian computation** (geometric + analytic) | Velocity mapping from joint space to task space. Required for IK and control. |
| **Inverse kinematics**: Newton-Raphson / Jacobian pseudoinverse / damped least squares | Numerical IK for arbitrary chains. Requires Jacobian + pseudoinverse — both building blocks exist. |
| **Quaternion arithmetic** (multiply, conjugate, normalize, slerp, exp, log) | Compact rotation representation. Missing — `manifold.rs` has some geometry but not robotics-specific quaternion algebra. |
| **Rotation conversions** (Euler angles ↔ rotation matrix ↔ quaternion ↔ axis-angle) | Required for interfacing between representations. |
| **Trajectory planning**: cubic/quintic polynomial, trapezoidal velocity profile, B-spline | Joint-space and task-space smooth trajectories. |
| **SLAM primitives**: scan matching (ICP), pose graph, landmark EKF | Uses existing Kalman and optimization — but no named robotics primitives. |
| **Rigid body dynamics**: recursive Newton-Euler (RNE), articulated body algorithm (ABA) | Forward/inverse dynamics for robot simulation. Recursive — Kingdom B. |

**Cross-cutting note**: Homogeneous transforms (SE(3) 4×4 matrices) and quaternion algebra are foundational. They also appear in computer vision, aerospace, and AR/VR. Quaternion slerp is a single function that serves robotics, animation, and aerospace attitude control simultaneously.

---

### 7. Materials Science / Structural Mechanics

**What's absent:**

| Primitive | Why it matters |
|-----------|---------------|
| **Stress/strain tensor operations** (Voigt notation, principal stresses, invariants) | Atom of continuum mechanics. Everything in structural FEM produces stress tensors. |
| **Constitutive models**: linear elastic (Hooke's law), von Mises yield, isotropic hardening, Drucker-Prager | Material behavior laws. Plugged into FEM integration points. |
| **Stress transformation** (rotation of stress tensor, Mohr's circle) | 2D/3D stress analysis. |
| **Fatigue life estimation**: S-N curve interpolation, Basquin law, Coffin-Manson (low-cycle) | Cycle-based life prediction from a given stress amplitude. |
| **Rainflow cycle counting** | Converts irregular load history to amplitude-mean cycles. Input to S-N life calculation. |
| **Miner's rule** (linear damage accumulation) | Sum of D_i = n_i / N_i over cycles until D=1. Trivial once rainflow counting exists. |
| **Fracture mechanics**: stress intensity factor K_I, K_II, K_III; J-integral; Paris law (crack growth rate) | Damage tolerance analysis. `da/dN = C ΔK^m`. |
| **Creep models**: Norton power law, Bailey-Norton, sinh law | Time-dependent deformation at high temperature. |

**Cross-cutting note**: Rainflow counting is the highest-value standalone primitive here — it appears in structural fatigue, electronics reliability (solder joint life), and wind turbine blade life. It is a single pass algorithm over a stress-time series.

---

## Cross-Cutting Primitives — Priority Order

These serve ≥3 engineering domains from a single implementation:

| Primitive | Domains served | Existence in tambear |
|-----------|---------------|---------------------|
| **Riccati equation solver** (DARE, CARE) | Control (LQR, LQG, H-inf), model reduction, stochastic control | ABSENT |
| **Generalized eigensolver** `K v = λ M v` | Structural modal analysis, stability, quantum chemistry, FEM | Only standard `sym_eigen` — MISSING for generalized |
| **Bilinear transform** + analog prototype poles | All IIR filter design (Chebyshev, Elliptic, Bessel, Butterworth) | Butterworth hardcoded; bilinear primitive ABSENT |
| **GF(2^m) arithmetic** | Reed-Solomon, CRC, BCH, LDPC, polar codes | ABSENT |
| **SE(3) / SO(3) operations** (4×4 homogeneous, quaternion) | Robotics, aerospace, computer vision, animation | ABSENT |
| **FEM element stiffness + 2D/3D Gaussian quadrature** | Structural, thermal, electromagnetic FEM | ABSENT |
| **MNA stamp accumulation** | Circuit simulation, power flow | ABSENT (but structurally same as FEM assembly) |
| **Rainflow cycle counting** | Fatigue, reliability, wind energy | ABSENT |
| **Pole/zero ↔ transfer function ↔ state-space conversions** | All control and filter design | ABSENT |
| **CRC (GF(2) polynomial division)** | Communications, storage, networking | ABSENT |

---

## What's Partially There

| Domain | What exists | What's missing |
|--------|------------|----------------|
| Control | Kalman filter (full), RTS smoother, particle filter, EM-SSM | Riccati, LQR, PID, pole placement, H-inf, MPC, Bode/Nyquist |
| Signal processing | FFT family, FIR windowed-sinc, Butterworth LP/HP biquad, all window functions | Chebyshev/Elliptic/Bessel design, bilinear transform, freqz, polyphase resampling, multitaper |
| Physics | N-body, SHO, Schrödinger 1D, Ising, Euler 1D, thermodynamics | FEM (entirely), continuum mechanics, fatigue |
| Linear algebra | Dense LU/QR/SVD/Cholesky, symmetric eigensolver, CG, GMRES | Sparse LU, Lanczos, generalized eigensolver |

---

## Implementation Notes

**Riccati (DARE/CARE)**: Implement via Schur decomposition of the Hamiltonian matrix. Requires `QR` and `sym_eigen` — both present. Sequential priority.

**Generalized eigensolver**: Extend `sym_eigen` to handle `K v = λ M v` via Cholesky factorization of M: `L^{-1} K L^{-T} w = λ w`, then `v = L^{-T} w`. All building blocks in `linear_algebra.rs`.

**Bilinear transform**: Pure mathematics — maps poles/zeros from s-domain to z-domain with frequency pre-warping. About 30 lines. Unlocks the entire IIR design family.

**GF(2^m) arithmetic**: Lookup-table implementation for GF(256) (m=8, the standard for RS and CRC-32). `primitive_polynomial`, `gf_mul`, `gf_div`, `gf_pow`, `gf_log`, `gf_exp`. About 60 lines.

**SE(3) / Quaternion**: 4×4 matrix operations already in `linear_algebra`. Quaternion multiply is 4 multiplications and 3 additions. `slerp` is 15 lines. High value-to-effort ratio.

**Rainflow counting**: O(n) pass with a 4-point counting rule. About 50 lines. Standalone primitive with broad applicability.

**FEM assembly**: The `accumulate(element_stiffness, scatter_to_global)` pattern is structurally identical to tambear's existing accumulate primitive. The hard part is element stiffness matrices; assembly is essentially free.
