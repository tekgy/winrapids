//! # Physics Engines
//!
//! Classical mechanics, statistical mechanics, quantum mechanics,
//! thermodynamics, and fluid dynamics — all from first principles.
//!
//! ## Architecture
//!
//! Every physics computation decomposes into accumulate + gather:
//! - N-body: accumulate(pairwise forces, gather by target particle)
//! - Statistical mechanics: accumulate(Boltzmann weights, gather by observable)
//! - Ising model: accumulate(local energy contributions, gather by spin)
//! - Fluid dynamics: accumulate(pressure/viscous fluxes, gather by cell)
//!
//! All quantities in SI units unless noted.

use std::f64::consts::{PI, TAU};

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants (SI)
// ─────────────────────────────────────────────────────────────────────────────

/// Boltzmann constant k_B (J/K)
pub const K_BOLTZMANN: f64 = 1.380649e-23;
/// Planck constant ℏ = h/(2π) (J·s)
pub const H_BAR: f64 = 1.0545718e-34;
/// Speed of light c (m/s)
pub const SPEED_OF_LIGHT: f64 = 2.99792458e8;
/// Gravitational constant G (m³ kg⁻¹ s⁻²)
pub const G_GRAV: f64 = 6.67430e-11;
/// Elementary charge e (C)
pub const ELEM_CHARGE: f64 = 1.602176634e-19;
/// Electron mass m_e (kg)
pub const MASS_ELECTRON: f64 = 9.1093837015e-31;
/// Bohr radius a_0 (m)
pub const BOHR_RADIUS: f64 = 5.29177210903e-11;
/// Avogadro number N_A (mol⁻¹)
pub const AVOGADRO: f64 = 6.02214076e23;
/// Gas constant R = k_B · N_A (J mol⁻¹ K⁻¹)
pub const GAS_CONSTANT: f64 = 8.314462618;
/// Vacuum permittivity ε_0 (F/m)
pub const EPSILON_0: f64 = 8.8541878128e-12;
/// Hydrogen ground state energy (eV)
pub const HYDROGEN_GROUND_EV: f64 = -13.605693;

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1 — Classical Mechanics
// ═══════════════════════════════════════════════════════════════════════════

// ── 1.1 Particle state ──────────────────────────────────────────────────────

/// State of a classical particle in 3D.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub mass: f64,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

impl Particle {
    pub fn new(mass: f64, pos: [f64; 3], vel: [f64; 3]) -> Self {
        Self { mass, pos, vel }
    }

    /// Kinetic energy ½mv².
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * (self.vel[0].powi(2) + self.vel[1].powi(2) + self.vel[2].powi(2))
    }

    /// Linear momentum mv.
    pub fn momentum(&self) -> [f64; 3] {
        [self.mass * self.vel[0], self.mass * self.vel[1], self.mass * self.vel[2]]
    }

    /// Angular momentum r × p around origin.
    pub fn angular_momentum(&self) -> [f64; 3] {
        let p = self.momentum();
        let r = &self.pos;
        [
            r[1] * p[2] - r[2] * p[1],
            r[2] * p[0] - r[0] * p[2],
            r[0] * p[1] - r[1] * p[0],
        ]
    }
}

// ── 1.2 N-body simulation (Velocity-Verlet) ──────────────────────────────────

/// N-body simulation result.
#[derive(Debug, Clone)]
pub struct NBodyResult {
    /// Final particle states.
    pub particles: Vec<Particle>,
    /// Total energy at each step.
    pub total_energy: Vec<f64>,
}

/// Gravitational N-body simulation using Velocity-Verlet integration.
/// `dt`: timestep (s), `steps`: number of integration steps.
///
/// Accumulate+gather: for each particle i, accumulate forces from all j≠i,
/// then gather (update velocity + position of i).
pub fn nbody_gravity(particles: &[Particle], dt: f64, steps: usize) -> NBodyResult {
    let n = particles.len();
    let mut state: Vec<Particle> = particles.to_vec();
    let mut energies = Vec::with_capacity(steps);

    // Compute accelerations (accumulate pairwise forces, gather by target)
    let compute_acc = |ps: &[Particle]| -> Vec<[f64; 3]> {
        let mut acc = vec![[0.0f64; 3]; ps.len()];
        for i in 0..ps.len() {
            for j in 0..ps.len() {
                if i == j { continue; }
                let dx = ps[j].pos[0] - ps[i].pos[0];
                let dy = ps[j].pos[1] - ps[i].pos[1];
                let dz = ps[j].pos[2] - ps[i].pos[2];
                let r2 = dx * dx + dy * dy + dz * dz + 1e-20; // softening
                let r = r2.sqrt();
                let f = G_GRAV * ps[j].mass / (r2 * r); // F/m = GM_j/r³ · r̂·r
                acc[i][0] += f * dx;
                acc[i][1] += f * dy;
                acc[i][2] += f * dz;
            }
        }
        acc
    };

    let total_energy = |ps: &[Particle]| -> f64 {
        let ke: f64 = ps.iter().map(|p| p.kinetic_energy()).sum();
        let mut pe = 0.0;
        for i in 0..ps.len() {
            for j in (i + 1)..ps.len() {
                let dx = ps[j].pos[0] - ps[i].pos[0];
                let dy = ps[j].pos[1] - ps[i].pos[1];
                let dz = ps[j].pos[2] - ps[i].pos[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-10);
                pe -= G_GRAV * ps[i].mass * ps[j].mass / r;
            }
        }
        ke + pe
    };

    let mut acc = compute_acc(&state);

    for _ in 0..steps {
        energies.push(total_energy(&state));

        // Velocity-Verlet: x_{n+1} = x_n + v_n·dt + ½a_n·dt²
        for i in 0..n {
            for k in 0..3 {
                state[i].pos[k] += state[i].vel[k] * dt + 0.5 * acc[i][k] * dt * dt;
            }
        }

        let new_acc = compute_acc(&state);

        // v_{n+1} = v_n + ½(a_n + a_{n+1})·dt
        for i in 0..n {
            for k in 0..3 {
                state[i].vel[k] += 0.5 * (acc[i][k] + new_acc[i][k]) * dt;
            }
        }
        acc = new_acc;
    }

    NBodyResult { particles: state, total_energy: energies }
}

// ── 1.3 Simple Harmonic Oscillator ──────────────────────────────────────────

/// Simple harmonic oscillator: x''= -ω²x.
/// Returns (position, velocity) at time t given initial conditions.
pub fn sho_exact(x0: f64, v0: f64, omega: f64, t: f64) -> (f64, f64) {
    // x(t) = x0 cos(ωt) + (v0/ω) sin(ωt)
    let x = x0 * (omega * t).cos() + (v0 / omega) * (omega * t).sin();
    let v = -x0 * omega * (omega * t).sin() + v0 * (omega * t).cos();
    (x, v)
}

/// SHO energy: E = ½m(v² + ω²x²).
pub fn sho_energy(mass: f64, omega: f64, x: f64, v: f64) -> f64 {
    0.5 * mass * (v * v + omega * omega * x * x)
}

/// Damped harmonic oscillator: x'' + 2γx' + ω₀²x = 0.
/// Returns position at time t. Assumes underdamped (γ < ω₀).
pub fn dho_underdamped(x0: f64, v0: f64, omega0: f64, gamma: f64, t: f64) -> f64 {
    assert!(gamma < omega0, "Must be underdamped");
    let omega_d = (omega0 * omega0 - gamma * gamma).sqrt();
    let a = x0;
    let b = (v0 + gamma * x0) / omega_d;
    (-gamma * t).exp() * (a * (omega_d * t).cos() + b * (omega_d * t).sin())
}

// ── 1.4 Kepler problem (2-body gravitational) ────────────────────────────────

/// Kepler orbit result: orbital elements.
#[derive(Debug, Clone)]
pub struct KeplerOrbit {
    /// Semi-major axis (m).
    pub semi_major_axis: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Orbital period (s).
    pub period: f64,
    /// Specific orbital energy (J/kg).
    pub energy: f64,
    /// Specific angular momentum magnitude (m²/s).
    pub angular_momentum: f64,
}

/// Compute Kepler orbit parameters from state vectors.
/// `r`: position vector (m), `v`: velocity vector (m/s), `M`: central mass (kg).
pub fn kepler_orbit(r: [f64; 3], v: [f64; 3], m_central: f64) -> KeplerOrbit {
    let mu = G_GRAV * m_central; // gravitational parameter

    let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    // Specific orbital energy: ε = ½v² - μ/r
    let energy = 0.5 * v_mag * v_mag - mu / r_mag;

    // Semi-major axis: a = -μ/(2ε)
    let sma = if energy.abs() > 1e-30 { -mu / (2.0 * energy) } else { f64::INFINITY };

    // Angular momentum: h = r × v
    let h = [
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ];
    let h_mag = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();

    // Eccentricity: e = √(1 + 2εh²/μ²)
    let e_sq = 1.0 + 2.0 * energy * h_mag * h_mag / (mu * mu);
    let eccentricity = if e_sq > 0.0 { e_sq.sqrt() } else { 0.0 };

    // Period: T = 2π√(a³/μ) for bound orbits
    let period = if energy < 0.0 && sma > 0.0 {
        TAU * (sma * sma * sma / mu).sqrt()
    } else {
        f64::INFINITY
    };

    KeplerOrbit { semi_major_axis: sma, eccentricity, period, energy, angular_momentum: h_mag }
}

/// Vis-viva equation: v² = μ(2/r - 1/a).
/// Returns orbital speed at distance r.
pub fn vis_viva(mu: f64, r: f64, sma: f64) -> f64 {
    (mu * (2.0 / r - 1.0 / sma)).sqrt()
}

// ── 1.5 Lagrangian/Hamiltonian mechanics ─────────────────────────────────────

/// Double pendulum state: (θ₁, θ₂, dθ₁/dt, dθ₂/dt).
#[derive(Debug, Clone)]
pub struct DoublePendulumState {
    pub theta1: f64,
    pub theta2: f64,
    pub omega1: f64,
    pub omega2: f64,
}

/// Double pendulum equations of motion (equal masses m, equal lengths L).
/// Derived from Lagrangian mechanics. Returns (dθ₁/dt, dθ₂/dt, d²θ₁/dt², d²θ₂/dt²).
///
/// Equations from Lagrangian with equal m, L:
/// (2m)L θ₁'' + mL θ₂'' cos δ + mL θ₂'² sin δ = -(2m)g sin θ₁
/// mL θ₂'' + mL θ₁'' cos δ - mL θ₁'² sin δ = -mg sin θ₂
/// where δ = θ₁ - θ₂. Solving: denom = 2 - cos²δ.
pub fn double_pendulum_derivs(s: &DoublePendulumState, g: f64, l: f64) -> (f64, f64, f64, f64) {
    let (t1, t2, w1, w2) = (s.theta1, s.theta2, s.omega1, s.omega2);
    let delta = t1 - t2;
    let sin_d = delta.sin();
    let cos_d = delta.cos();
    let gl = g / l;
    // Denominator from 2×2 Cramer rule: det([[2, cosδ],[cosδ, 1]]) = 2 - cos²δ
    let denom = 2.0 - cos_d * cos_d;

    // θ₁'' = [(-2g/l sinθ₁ - ω₂² sinδ) - (-(g/l)sinθ₂ + ω₁² sinδ)cosδ] / (2-cos²δ)
    let a1 = (-2.0 * gl * t1.sin()
        + gl * t2.sin() * cos_d
        - w2 * w2 * sin_d
        - w1 * w1 * sin_d * cos_d) / denom;

    // θ₂'' = [2(-(g/l)sinθ₂ + ω₁² sinδ) - (-2g/l sinθ₁ - ω₂² sinδ)cosδ] / (2-cos²δ)
    let a2 = (-2.0 * gl * t2.sin()
        + 2.0 * w1 * w1 * sin_d
        + 2.0 * gl * t1.sin() * cos_d
        + w2 * w2 * sin_d * cos_d) / denom;

    (w1, w2, a1, a2)
}

/// Integrate double pendulum using RK4 for `steps` timesteps of `dt`.
pub fn double_pendulum_rk4(
    init: &DoublePendulumState,
    g: f64,
    l: f64,
    dt: f64,
    steps: usize,
) -> Vec<DoublePendulumState> {
    let mut states = Vec::with_capacity(steps + 1);
    let mut s = init.clone();
    states.push(s.clone());

    for _ in 0..steps {
        let rk4_step = |s: &DoublePendulumState| -> DoublePendulumState {
            let (dt1, dt2, da1, da2) = double_pendulum_derivs(s, g, l);
            let k1 = (dt1, dt2, da1, da2);

            let s2 = DoublePendulumState {
                theta1: s.theta1 + 0.5 * dt * k1.0,
                theta2: s.theta2 + 0.5 * dt * k1.1,
                omega1: s.omega1 + 0.5 * dt * k1.2,
                omega2: s.omega2 + 0.5 * dt * k1.3,
            };
            let (dt1, dt2, da1, da2) = double_pendulum_derivs(&s2, g, l);
            let k2 = (dt1, dt2, da1, da2);

            let s3 = DoublePendulumState {
                theta1: s.theta1 + 0.5 * dt * k2.0,
                theta2: s.theta2 + 0.5 * dt * k2.1,
                omega1: s.omega1 + 0.5 * dt * k2.2,
                omega2: s.omega2 + 0.5 * dt * k2.3,
            };
            let (dt1, dt2, da1, da2) = double_pendulum_derivs(&s3, g, l);
            let k3 = (dt1, dt2, da1, da2);

            let s4 = DoublePendulumState {
                theta1: s.theta1 + dt * k3.0,
                theta2: s.theta2 + dt * k3.1,
                omega1: s.omega1 + dt * k3.2,
                omega2: s.omega2 + dt * k3.3,
            };
            let (dt1, dt2, da1, da2) = double_pendulum_derivs(&s4, g, l);
            let k4 = (dt1, dt2, da1, da2);

            DoublePendulumState {
                theta1: s.theta1 + dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
                theta2: s.theta2 + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
                omega1: s.omega1 + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
                omega2: s.omega2 + dt / 6.0 * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3),
            }
        };
        s = rk4_step(&s);
        states.push(s.clone());
    }
    states
}

/// Double pendulum total energy E = T + V.
/// T = ½mL²(2ω₁² + ω₂² + 2ω₁ω₂cos(θ₁-θ₂))
/// V = -mgL(2cosθ₁ + cosθ₂)
pub fn double_pendulum_energy(s: &DoublePendulumState, m: f64, l: f64, g: f64) -> f64 {
    let dt = s.theta1 - s.theta2;
    let t = 0.5 * m * l * l * (2.0 * s.omega1.powi(2) + s.omega2.powi(2)
        + 2.0 * s.omega1 * s.omega2 * dt.cos());
    let v = -m * g * l * (2.0 * s.theta1.cos() + s.theta2.cos());
    t + v
}

// ── 1.6 Rigid body rotation ──────────────────────────────────────────────────

/// Euler's rotation equations for a torque-free rigid body.
/// `I`: principal moments of inertia [I₁, I₂, I₃].
/// `omega`: angular velocity components, `dt`: timestep, `steps`: count.
/// Returns sequence of angular velocities (conservation of angular momentum).
pub fn euler_rotation(omega0: [f64; 3], i_moi: [f64; 3], dt: f64, steps: usize) -> Vec<[f64; 3]> {
    let [i1, i2, i3] = i_moi;
    let mut w = omega0;
    let mut result = Vec::with_capacity(steps + 1);
    result.push(w);

    for _ in 0..steps {
        // Euler: I_k dω_k/dt = (I_j - I_i)ω_i ω_j (cyclic)
        let dw1 = (i2 - i3) / i1 * w[1] * w[2];
        let dw2 = (i3 - i1) / i2 * w[2] * w[0];
        let dw3 = (i1 - i2) / i3 * w[0] * w[1];
        w[0] += dt * dw1;
        w[1] += dt * dw2;
        w[2] += dt * dw3;
        result.push(w);
    }
    result
}

/// Rotational kinetic energy: T = ½ Σ I_k ω_k².
pub fn rotational_kinetic_energy(omega: [f64; 3], i_moi: [f64; 3]) -> f64 {
    0.5 * (i_moi[0] * omega[0].powi(2)
        + i_moi[1] * omega[1].powi(2)
        + i_moi[2] * omega[2].powi(2))
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2 — Thermodynamics
// ═══════════════════════════════════════════════════════════════════════════

// ── 2.1 Ideal gas ────────────────────────────────────────────────────────────

/// Ideal gas pressure: P = nRT/V.
pub fn ideal_gas_pressure(n_mol: f64, temperature: f64, volume: f64) -> f64 {
    n_mol * GAS_CONSTANT * temperature / volume
}

/// Ideal gas temperature: T = PV/(nR).
pub fn ideal_gas_temperature(pressure: f64, volume: f64, n_mol: f64) -> f64 {
    pressure * volume / (n_mol * GAS_CONSTANT)
}

/// Ideal gas internal energy: U = f/2 · nRT, where f is degrees of freedom.
/// Monatomic: f=3, diatomic: f=5, polyatomic: f=6.
pub fn ideal_gas_internal_energy(n_mol: f64, temperature: f64, dof: u32) -> f64 {
    0.5 * dof as f64 * n_mol * GAS_CONSTANT * temperature
}

/// Ideal gas entropy change: ΔS = nR ln(V₂/V₁) + n·C_V ln(T₂/T₁).
/// C_V = (f/2)R per mole.
pub fn ideal_gas_entropy_change(
    n_mol: f64, t1: f64, t2: f64, v1: f64, v2: f64, dof: u32,
) -> f64 {
    let cv = 0.5 * dof as f64 * GAS_CONSTANT;
    n_mol * (GAS_CONSTANT * (v2 / v1).ln() + cv * (t2 / t1).ln())
}

// ── 2.2 van der Waals gas ─────────────────────────────────────────────────

/// van der Waals pressure: P = nRT/(V-nb) - an²/V².
/// `a`: attraction parameter (Pa·m⁶/mol²), `b`: volume parameter (m³/mol).
pub fn vdw_pressure(n_mol: f64, temperature: f64, volume: f64, a: f64, b: f64) -> f64 {
    n_mol * GAS_CONSTANT * temperature / (volume - n_mol * b) - a * n_mol * n_mol / (volume * volume)
}

/// van der Waals critical constants: T_c, P_c, V_c.
pub fn vdw_critical(a: f64, b: f64) -> (f64, f64, f64) {
    let t_c = 8.0 * a / (27.0 * GAS_CONSTANT * b);
    let p_c = a / (27.0 * b * b);
    let v_c = 3.0 * b; // per mole
    (t_c, p_c, v_c)
}

// ── 2.3 Thermodynamic cycles ──────────────────────────────────────────────

/// Carnot efficiency: η = 1 - T_cold/T_hot.
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> f64 {
    assert!(t_hot > t_cold && t_cold > 0.0);
    1.0 - t_cold / t_hot
}

/// Otto cycle efficiency: η = 1 - r^{1-γ}, where r = compression ratio.
/// γ = C_P/C_V = (f+2)/f for ideal gas.
pub fn otto_efficiency(compression_ratio: f64, gamma: f64) -> f64 {
    1.0 - compression_ratio.powf(1.0 - gamma)
}

/// Clausius inequality: ΔS_universe ≥ 0. Returns entropy change of system.
/// `q`: heat absorbed by system, `t`: temperature of surroundings.
pub fn entropy_change_isothermal(q: f64, t: f64) -> f64 {
    q / t
}

// ── 2.4 Heat transfer ─────────────────────────────────────────────────────

/// Fourier's law: heat flux q = -k·∇T.
/// `k`: thermal conductivity, `dt_dx`: temperature gradient.
pub fn heat_flux_fourier(k: f64, dt_dx: f64) -> f64 {
    -k * dt_dx
}

/// Newton's cooling: dT/dt = -h(T - T_env).
/// Temperature at time t: T(t) = T_env + (T0 - T_env)·exp(-ht).
pub fn newton_cooling(t0: f64, t_env: f64, h: f64, t: f64) -> f64 {
    t_env + (t0 - t_env) * (-h * t).exp()
}

/// Stefan-Boltzmann radiation: P = ε·σ·A·T⁴.
/// σ = 5.670374419e-8 W m⁻² K⁻⁴.
pub fn stefan_boltzmann(emissivity: f64, area: f64, temperature: f64) -> f64 {
    const SIGMA: f64 = 5.670374419e-8;
    emissivity * SIGMA * area * temperature.powi(4)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3 — Statistical Mechanics
// ═══════════════════════════════════════════════════════════════════════════

// ── 3.1 Canonical ensemble ────────────────────────────────────────────────

/// Compute partition function Z = Σ exp(-β·E_i) for discrete energy levels.
/// `beta` = 1/(k_B·T).
pub fn partition_function(energies: &[f64], beta: f64) -> f64 {
    energies.iter().map(|&e| (-beta * e).exp()).sum()
}

/// Mean energy ⟨E⟩ = -∂ln(Z)/∂β = Σ E_i exp(-βE_i) / Z.
pub fn mean_energy(energies: &[f64], beta: f64) -> f64 {
    let z = partition_function(energies, beta);
    let num: f64 = energies.iter().map(|&e| e * (-beta * e).exp()).sum();
    num / z
}

/// Heat capacity C_V = k_B β² ⟨(ΔE)²⟩.
pub fn heat_capacity_canonical(energies: &[f64], beta: f64) -> f64 {
    let z = partition_function(energies, beta);
    let e_mean = mean_energy(energies, beta);
    let e2_mean: f64 = energies.iter().map(|&e| e * e * (-beta * e).exp()).sum::<f64>() / z;
    let var = e2_mean - e_mean * e_mean;
    K_BOLTZMANN * beta * beta * var
}

/// Helmholtz free energy: F = -k_B T ln(Z).
pub fn helmholtz_free_energy(energies: &[f64], beta: f64) -> f64 {
    let z = partition_function(energies, beta);
    -z.ln() / beta  // = -k_B T ln(Z)
}

/// Boltzmann probability: p_i = exp(-βE_i) / Z.
pub fn boltzmann_probabilities(energies: &[f64], beta: f64) -> Vec<f64> {
    let z = partition_function(energies, beta);
    energies.iter().map(|&e| (-beta * e).exp() / z).collect()
}

/// Gibbs entropy: S = -k_B Σ p_i ln(p_i).
pub fn gibbs_entropy(probabilities: &[f64]) -> f64 {
    -K_BOLTZMANN * probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

// ── 3.2 Quantum harmonic oscillator statistics ────────────────────────────

/// Quantum harmonic oscillator energy levels: E_n = ℏω(n + ½).
pub fn qho_energy(n: u32, omega: f64) -> f64 {
    H_BAR * omega * (n as f64 + 0.5)
}

/// Mean occupation number (Bose-Einstein): ⟨n⟩ = 1/(exp(βℏω) - 1).
pub fn bose_einstein_occupation(omega: f64, beta: f64) -> f64 {
    let x = beta * H_BAR * omega;
    1.0 / (x.exp() - 1.0)
}

/// Planck distribution: energy spectral density for blackbody radiation.
/// u(ω) = ℏω³/(π²c³) · 1/(exp(βℏω) - 1).
pub fn planck_spectral_energy(omega: f64, temperature: f64) -> f64 {
    let beta = 1.0 / (K_BOLTZMANN * temperature);
    let x = beta * H_BAR * omega;
    if x > 700.0 { return 0.0; } // prevent overflow
    H_BAR * omega.powi(3) / (PI * PI * SPEED_OF_LIGHT.powi(3) * (x.exp() - 1.0))
}

/// Wien displacement: peak wavelength λ_max = b/T, b = 2.898e-3 m·K.
pub fn wien_displacement(temperature: f64) -> f64 {
    2.897771955e-3 / temperature
}

// ── 3.3 1D Ising model (exact transfer matrix) ────────────────────────────

// Non-recursive helper: compute 1D Ising transfer-matrix largest eigenvalue.
fn ising1d_lam_plus(j_coupling: f64, h_field: f64, beta: f64) -> f64 {
    let t11 = (beta * (j_coupling + h_field)).exp();
    let t12 = (-beta * j_coupling).exp();
    let t22 = (beta * (j_coupling - h_field)).exp();
    let tr_half = (t11 + t22) / 2.0;
    let disc = ((t11 - t22) / 2.0).powi(2) + t12 * t12;
    tr_half + disc.sqrt()
}

/// 1D Ising model with periodic boundary conditions.
/// H = -J Σ sᵢsᵢ₊₁ - h Σ sᵢ, spins ∈ {+1, -1}.
///
/// Exact solution via transfer matrix. Works in units where k_B = 1.
/// Returns (free energy per spin, magnetization per spin, heat capacity per spin).
pub fn ising1d_exact(j_coupling: f64, h_field: f64, beta: f64) -> (f64, f64, f64) {
    let lp = ising1d_lam_plus(j_coupling, h_field, beta);
    let free_energy_per_spin = -lp.ln() / beta;

    // Magnetization: m = -∂f/∂h via centered finite difference (no recursion)
    let dh = 1e-6 * (j_coupling.abs().max(1.0));
    let lp_ph = ising1d_lam_plus(j_coupling, h_field + dh, beta);
    let lp_mh = ising1d_lam_plus(j_coupling, h_field - dh, beta);
    let f_ph = -lp_ph.ln() / beta;
    let f_mh = -lp_mh.ln() / beta;
    let magnetization = -(f_ph - f_mh) / (2.0 * dh);

    // Mean energy per spin: E = -∂ln(λ+)/∂β = ∂(β f)/∂β
    // Use centered difference on β
    let db = beta * 0.001_f64.max(1e-10);
    let beta_p = beta + db;
    let beta_m = (beta - db).max(1e-15);
    let lp_pb = ising1d_lam_plus(j_coupling, h_field, beta_p);
    let lp_mb = ising1d_lam_plus(j_coupling, h_field, beta_m);
    let e_mean_p = -lp_pb.ln() / beta_p; // f at β+db
    let e_mean_m = -lp_mb.ln() / beta_m; // f at β-db
    // E = ∂(βf)/∂β ≈ [(β+db)f(β+db) - (β-db)f(β-db)] / (2 db)
    let e_mean = ((beta_p * e_mean_p) - (beta_m * e_mean_m)) / (2.0 * db);
    let e2_mean_p = {
        let db2 = db * 0.1;
        let beta_pp = beta_p + db2;
        let beta_pm = (beta_p - db2).max(1e-15);
        let lpp = ising1d_lam_plus(j_coupling, h_field, beta_pp);
        let lpm = ising1d_lam_plus(j_coupling, h_field, beta_pm);
        let fp = -lpp.ln() / beta_pp;
        let fm = -lpm.ln() / beta_pm;
        ((beta_pp * fp) - (beta_pm * fm)) / (2.0 * db2)
    };
    // C_V = k_B β² Var(E) = k_B β² (⟨E²⟩ - ⟨E⟩²)
    // Approximate by finite difference: C_V = -∂⟨E⟩/∂T = k_B β² ∂⟨E⟩/∂β
    let de_dbeta = (e2_mean_p - e_mean) / db;
    let heat_capacity = K_BOLTZMANN * beta * beta * de_dbeta.abs();

    (free_energy_per_spin, magnetization, heat_capacity.max(0.0))
}

/// 2D Ising model Metropolis Monte Carlo.
/// `n`: lattice size (n×n), `j_coupling`: exchange coupling, `h_field`: external field,
/// `temperature`: T in units where k_B=1, `n_sweep`: Monte Carlo sweeps.
/// Returns (magnetization_per_spin, energy_per_spin).
pub fn ising2d_metropolis(
    n: usize,
    j_coupling: f64,
    h_field: f64,
    temperature: f64,
    n_sweep: usize,
    seed: u64,
) -> (f64, f64) {
    let beta = 1.0 / temperature;
    let mut spins = vec![1i8; n * n]; // start ferromagnetic

    // LCG random number generator
    let mut rng_state = seed;
    let lcg_next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s as f64 / u64::MAX as f64
    };

    let spin = |i: usize, j: usize, spins: &[i8]| -> i8 {
        spins[(i % n) * n + j % n]
    };

    for _ in 0..n_sweep {
        // Accumulate flip attempts for all spins (single-spin-flip dynamics)
        for i in 0..n {
            for j in 0..n {
                let s = spins[i * n + j];
                // Local field: sum of neighbors
                let h_local = j_coupling * (
                    spin(i.wrapping_add(1), j, &spins) as f64
                    + spin(i.wrapping_sub(1).wrapping_add(n), j, &spins) as f64
                    + spin(i, j.wrapping_add(1), &spins) as f64
                    + spin(i, j.wrapping_sub(1).wrapping_add(n), &spins) as f64
                ) + h_field;
                let delta_e = 2.0 * s as f64 * h_local;
                // Metropolis criterion
                if delta_e <= 0.0 || lcg_next(&mut rng_state) < (-beta * delta_e).exp() {
                    spins[i * n + j] = -s;
                }
            }
        }
    }

    // Compute observables: accumulate magnetization and energy
    let total_m: i64 = spins.iter().map(|&s| s as i64).sum();
    let mut total_e = 0.0;
    for i in 0..n {
        for j in 0..n {
            let s = spins[i * n + j] as f64;
            let right = spins[i * n + (j + 1) % n] as f64;
            let down = spins[((i + 1) % n) * n + j] as f64;
            total_e -= j_coupling * s * (right + down) + h_field * s;
        }
    }
    let nn = (n * n) as f64;
    (total_m as f64 / nn, total_e / nn)
}

// ── 3.4 Chemical equilibrium / reaction rates ─────────────────────────────

/// Arrhenius rate constant: k = A·exp(-Ea/(RT)).
pub fn arrhenius(pre_exponential: f64, activation_energy: f64, temperature: f64) -> f64 {
    pre_exponential * (-activation_energy / (GAS_CONSTANT * temperature)).exp()
}

/// Equilibrium constant from Gibbs free energy: K = exp(-ΔG°/(RT)).
pub fn equilibrium_constant(delta_g: f64, temperature: f64) -> f64 {
    (-delta_g / (GAS_CONSTANT * temperature)).exp()
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4 — Quantum Mechanics
// ═══════════════════════════════════════════════════════════════════════════

// ── 4.1 Energy levels ─────────────────────────────────────────────────────

/// Hydrogen atom energy levels: E_n = -13.6 eV / n².
pub fn hydrogen_energy_ev(n: u32) -> f64 {
    assert!(n >= 1);
    HYDROGEN_GROUND_EV / (n * n) as f64
}

/// Hydrogen atom wavelength for n₁→n₂ transition (Rydberg formula), in meters.
/// Returns emission wavelength if n2 < n1, absorption if n2 > n1.
pub fn hydrogen_wavelength(n1: u32, n2: u32) -> f64 {
    // R_H = 1.0974e7 m⁻¹ (Rydberg constant)
    const R_H: f64 = 1.097373157e7;
    let inv_lambda = R_H * (1.0 / (n2 * n2) as f64 - 1.0 / (n1 * n1) as f64);
    1.0 / inv_lambda.abs()
}

/// Particle in a 1D box: energy levels E_n = n²π²ℏ²/(2mL²).
pub fn particle_in_box_energy(n: u32, mass: f64, length: f64) -> f64 {
    let n = n as f64;
    n * n * PI * PI * H_BAR * H_BAR / (2.0 * mass * length * length)
}

/// Particle in a 1D box: wavefunction ψ_n(x) = √(2/L) sin(nπx/L).
pub fn particle_in_box_wf(n: u32, x: f64, length: f64) -> f64 {
    let n = n as f64;
    (2.0 / length).sqrt() * (n * PI * x / length).sin()
}

/// Quantum tunneling transmission probability (rectangular barrier).
/// T = 1/(1 + V₀² sinh²(κL) / (4E(V₀-E)))
/// for E < V₀. Returns 1.0 if E >= V₀ (classical transmission).
pub fn tunneling_transmission(energy: f64, v0: f64, mass: f64, barrier_width: f64) -> f64 {
    if energy >= v0 { return 1.0; }
    let kappa = (2.0 * mass * (v0 - energy)).sqrt() / H_BAR;
    let kl = kappa * barrier_width;
    let sinh_kl = kl.sinh();
    let denom = 1.0 + v0 * v0 * sinh_kl * sinh_kl / (4.0 * energy * (v0 - energy));
    1.0 / denom
}

// ── 4.2 Quantum state representation ─────────────────────────────────────

/// Complex number (Re, Im) for quantum amplitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amplitude(pub f64, pub f64);

impl Amplitude {
    pub fn new(re: f64, im: f64) -> Self { Amplitude(re, im) }
    pub fn re(&self) -> f64 { self.0 }
    pub fn im(&self) -> f64 { self.1 }
    pub fn norm_sq(&self) -> f64 { self.0 * self.0 + self.1 * self.1 }
    pub fn norm(&self) -> f64 { self.norm_sq().sqrt() }
    pub fn conj(&self) -> Self { Amplitude(self.0, -self.1) }

    pub fn mul(&self, other: &Amplitude) -> Amplitude {
        Amplitude(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }
    pub fn add(&self, other: &Amplitude) -> Amplitude {
        Amplitude(self.0 + other.0, self.1 + other.1)
    }
    pub fn scale(&self, s: f64) -> Amplitude {
        Amplitude(self.0 * s, self.1 * s)
    }
    /// exp(iφ)
    pub fn phase(phi: f64) -> Amplitude {
        Amplitude(phi.cos(), phi.sin())
    }
}

/// Normalize quantum state: divide each amplitude by √Σ|ψ_i|².
pub fn normalize_state(psi: &mut Vec<Amplitude>) {
    let norm_sq: f64 = psi.iter().map(|a| a.norm_sq()).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-15 {
        for a in psi.iter_mut() { *a = a.scale(1.0 / norm); }
    }
}

/// Time evolution of a state under Hamiltonian with eigenvalues `energies`.
/// ψ(t) = Σ cₙ exp(-iEₙt/ℏ) |n⟩, where input amplitudes cₙ are in eigenbasis.
pub fn time_evolve_state(amplitudes: &[Amplitude], energies: &[f64], t: f64) -> Vec<Amplitude> {
    assert_eq!(amplitudes.len(), energies.len());
    amplitudes.iter().zip(energies.iter())
        .map(|(c, &e)| {
            let phase = -e * t / H_BAR;
            c.mul(&Amplitude::phase(phase))
        })
        .collect()
}

/// Expectation value ⟨A⟩ = Σᵢ pᵢ aᵢ for observable A with eigenvalues `a`.
/// `probs`: |ψ_i|².
pub fn expectation_value(probs: &[f64], eigenvalues: &[f64]) -> f64 {
    probs.iter().zip(eigenvalues.iter()).map(|(p, a)| p * a).sum()
}

/// Uncertainty: Δ²A = ⟨A²⟩ - ⟨A⟩².
pub fn uncertainty(probs: &[f64], eigenvalues: &[f64]) -> f64 {
    let mean = expectation_value(probs, eigenvalues);
    let mean_sq = expectation_value(probs, &eigenvalues.iter().map(|a| a * a).collect::<Vec<_>>());
    (mean_sq - mean * mean).max(0.0).sqrt()
}

/// Heisenberg uncertainty product ΔxΔp ≥ ℏ/2.
/// Checks for Gaussian wavepacket: σ_x · σ_p = ℏ/2.
pub fn heisenberg_uncertainty_product(sigma_x: f64, sigma_p: f64) -> f64 {
    sigma_x * sigma_p
}

// ── 4.3 Density matrix ────────────────────────────────────────────────────

/// Density matrix trace: Tr(ρ) = Σᵢ ρᵢᵢ.
pub fn density_matrix_trace(rho: &[f64], dim: usize) -> f64 {
    (0..dim).map(|i| rho[i * dim + i]).sum()
}

/// Purity: Tr(ρ²) = Σᵢⱼ |ρᵢⱼ|².
pub fn density_matrix_purity(rho: &[f64], dim: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            sum += rho[i * dim + j] * rho[j * dim + i]; // = Σ ρᵢⱼ ρⱼᵢ = Tr(ρ²) for real ρ
        }
    }
    sum
}

/// Von Neumann entropy: S = -Tr(ρ ln ρ) = -Σ λᵢ ln λᵢ where λᵢ = eigenvalues.
/// For diagonal density matrix (mixed state of pure basis states).
pub fn von_neumann_entropy_diagonal(diagonal_probs: &[f64]) -> f64 {
    -diagonal_probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

// ── 4.4 Schrödinger equation (1D, finite difference) ─────────────────────

/// Solve 1D time-independent Schrödinger equation via shooting method.
/// -ℏ²/(2m) ψ'' + V(x) ψ = E ψ on grid.
/// Returns (eigenvalues, eigenvectors) for the first `n_states` states.
pub fn schrodinger1d(
    v_potential: &[f64],
    dx: f64,
    mass: f64,
    n_states: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = v_potential.len();
    // Build tridiagonal Hamiltonian matrix H = T + V
    // T_ii = ℏ²/(m dx²), T_{i,i±1} = -ℏ²/(2m dx²)
    let t_diag = H_BAR * H_BAR / (mass * dx * dx);
    let t_off = -0.5 * H_BAR * H_BAR / (mass * dx * dx);

    // Power iteration for lowest eigenvalues — not practical for full diag
    // Use tridiagonal eigenvalue via QR-like algorithm for symmetric tridiag

    let diag: Vec<f64> = (0..n).map(|i| t_diag + v_potential[i]).collect();
    let off = vec![t_off; n - 1];

    // Lanczos / symmetric tridiagonal eigendecomposition
    // For small systems, use the power iteration approach
    // For the full tridiagonal case, use the QR algorithm for symmetric tridiagonals

    sym_tridiag_eigvals(&diag, &off, n_states)
}

/// Symmetric tridiagonal matrix eigenvalues via bisection (Sturm sequence).
/// Returns the `n_states` lowest eigenvalues and their eigenvectors.
pub fn sym_tridiag_eigvals(
    diag: &[f64],
    off: &[f64],
    n_states: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = diag.len();
    let n_states = n_states.min(n);

    // Gershgorin bounds for eigenvalue range
    let mut e_min = f64::INFINITY;
    let mut e_max = f64::NEG_INFINITY;
    for i in 0..n {
        let r = if i == 0 { off[0].abs() }
            else if i == n - 1 { off[n - 2].abs() }
            else { off[i - 1].abs() + off[i].abs() };
        e_min = e_min.min(diag[i] - r);
        e_max = e_max.max(diag[i] + r);
    }
    e_min -= 1.0;
    e_max += 1.0;

    // Count eigenvalues ≤ x using Sturm sequence
    let sturm_count = |x: f64| -> usize {
        let mut count = 0;
        let mut q = diag[0] - x;
        if q < 0.0 { count += 1; }
        for i in 1..n {
            let prev_q = q;
            q = (diag[i] - x) - if prev_q.abs() > 1e-300 {
                off[i - 1] * off[i - 1] / prev_q
            } else {
                off[i - 1].abs().signum() * 1e300
            };
            if q < 0.0 { count += 1; }
        }
        count
    };

    // Bisect for each eigenvalue
    let mut eigenvalues = Vec::with_capacity(n_states);
    for k in 0..n_states {
        // Find interval [lo, hi] with exactly k eigenvalues < lo, k+1 <= hi
        let mut lo = e_min;
        let mut hi = e_max;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if sturm_count(mid) <= k { lo = mid; } else { hi = mid; }
            if hi - lo < 1e-10 { break; }
        }
        eigenvalues.push((lo + hi) / 2.0);
    }

    // Compute eigenvectors via inverse iteration
    let eigenvectors = eigenvalues.iter().map(|&ev| {
        // Inverse iteration: (H - λI)^{-1} v
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..30 {
            // Solve (H - (λ+shift)I)w = v via Thomas algorithm (tridiagonal)
            let shift = ev + 1e-8;
            let d: Vec<f64> = (0..n).map(|i| diag[i] - shift).collect();
            let o: Vec<f64> = off.to_vec();

            // Forward elimination
            let mut c = vec![0.0; n - 1];
            let mut d2 = d.clone();
            let mut v2 = v.clone();
            c[0] = o[0] / d2[0];
            for i in 1..n - 1 {
                let m = d2[i] - o[i - 1] * c[i - 1];
                d2[i] = m;
                c[i] = o[i] / m;
                v2[i] -= o[i - 1] * v2[i - 1] / (d2[i - 1] + 1e-300);
            }
            // Back substitution
            let mut w = vec![0.0; n];
            w[n - 1] = v2[n - 1] / (d2[n - 1] + 1e-300);
            for i in (0..n - 1).rev() {
                w[i] = (v2[i] - o[i] * w[i + 1]) / (d2[i] + 1e-300);
            }

            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 { v = w.iter().map(|x| x / norm).collect(); }
        }
        v
    }).collect();

    (eigenvalues, eigenvectors)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5 — Fluid Dynamics
// ═══════════════════════════════════════════════════════════════════════════

// ── 5.1 Dimensionless numbers ─────────────────────────────────────────────

/// Reynolds number: Re = ρvL/μ.
pub fn reynolds_number(density: f64, velocity: f64, length: f64, viscosity: f64) -> f64 {
    density * velocity * length / viscosity
}

/// Mach number: Ma = v/c_s where c_s = √(γRT/M) for ideal gas.
pub fn mach_number(velocity: f64, gamma: f64, temperature: f64, molar_mass: f64) -> f64 {
    let c_s = (gamma * GAS_CONSTANT * temperature / molar_mass).sqrt();
    velocity / c_s
}

/// Prandtl number: Pr = μ C_P / k.
pub fn prandtl_number(viscosity: f64, heat_capacity: f64, thermal_conductivity: f64) -> f64 {
    viscosity * heat_capacity / thermal_conductivity
}

/// Nusselt number via Dittus-Boelter correlation (turbulent pipe flow):
/// Nu = 0.023 Re^0.8 Pr^n, n=0.4 (heating) or 0.3 (cooling).
pub fn nusselt_dittus_boelter(re: f64, pr: f64, heating: bool) -> f64 {
    let n = if heating { 0.4 } else { 0.3 };
    0.023 * re.powf(0.8) * pr.powf(n)
}

// ── 5.2 Bernoulli and pipe flow ──────────────────────────────────────────

/// Bernoulli equation: P + ½ρv² + ρgh = const.
/// Given two points, returns velocity at point 2.
/// `p1/p2`: pressures (Pa), `v1`: velocity 1 (m/s), `h1/h2`: heights (m).
pub fn bernoulli_velocity(p1: f64, p2: f64, v1: f64, h1: f64, h2: f64, density: f64) -> f64 {
    let g = 9.80665;
    let v2_sq = v1 * v1 + 2.0 * (p1 - p2) / density + 2.0 * g * (h1 - h2);
    if v2_sq > 0.0 { v2_sq.sqrt() } else { 0.0 }
}

/// Poiseuille flow: volumetric flow rate Q = π r⁴ ΔP / (8 μ L).
pub fn poiseuille_flow_rate(radius: f64, pressure_drop: f64, viscosity: f64, length: f64) -> f64 {
    PI * radius.powi(4) * pressure_drop / (8.0 * viscosity * length)
}

/// Poiseuille velocity profile: v(r) = (P_drop/(4μL)) (R²-r²).
pub fn poiseuille_velocity_profile(r: f64, r_max: f64, pressure_drop: f64, viscosity: f64, length: f64) -> f64 {
    pressure_drop / (4.0 * viscosity * length) * (r_max * r_max - r * r)
}

// ── 5.3 1D Euler equations (compressible flow) ────────────────────────────

/// 1D compressible flow state.
#[derive(Debug, Clone, Copy)]
pub struct FlowState {
    pub density: f64,
    pub momentum: f64, // ρu
    pub energy: f64,   // total energy E = ρ(e + u²/2)
}

impl FlowState {
    pub fn velocity(&self) -> f64 { self.momentum / self.density.max(1e-15) }
    pub fn pressure(&self, gamma: f64) -> f64 {
        (gamma - 1.0) * (self.energy - 0.5 * self.momentum * self.momentum / self.density.max(1e-15))
    }
    pub fn sound_speed(&self, gamma: f64) -> f64 {
        (gamma * self.pressure(gamma) / self.density.max(1e-15)).sqrt()
    }
}

/// 1D Euler equations with Lax-Friedrichs scheme.
/// Returns updated state array after `dt` timestep.
/// `gamma`: adiabatic index (5/3 for monatomic, 7/5 for diatomic).
pub fn euler1d_lax_friedrichs(states: &[FlowState], dx: f64, dt: f64, gamma: f64) -> Vec<FlowState> {
    let n = states.len();
    let mut new_states = states.to_vec();

    let flux = |s: &FlowState| -> (f64, f64, f64) {
        let p = s.pressure(gamma);
        let u = s.velocity();
        (s.momentum, s.density * u * u + p, (s.energy + p) * u)
    };

    for i in 1..n - 1 {
        let (fl0, fl1, fl2) = flux(&states[i - 1]);
        let (fr0, fr1, fr2) = flux(&states[i + 1]);

        new_states[i].density = 0.5 * (states[i - 1].density + states[i + 1].density)
            - 0.5 * dt / dx * (fr0 - fl0);
        new_states[i].momentum = 0.5 * (states[i - 1].momentum + states[i + 1].momentum)
            - 0.5 * dt / dx * (fr1 - fl1);
        new_states[i].energy = 0.5 * (states[i - 1].energy + states[i + 1].energy)
            - 0.5 * dt / dx * (fr2 - fl2);

        // Ensure physical state
        new_states[i].density = new_states[i].density.max(1e-15);
    }

    new_states
}

/// CFL timestep: dt = CFL · dx / max(|u| + c).
pub fn cfl_timestep(states: &[FlowState], dx: f64, cfl: f64, gamma: f64) -> f64 {
    let max_wave: f64 = states.iter().map(|s| {
        s.velocity().abs() + s.sound_speed(gamma)
    }).fold(0.0_f64, f64::max);
    if max_wave > 1e-15 { cfl * dx / max_wave } else { f64::MAX }
}

// ── 5.4 Navier-Stokes 2D vorticity-streamfunction ─────────────────────────

/// Solve 2D Poisson equation ∇²ψ = -ω for streamfunction given vorticity.
/// Uses successive over-relaxation (SOR).
/// `omega`: vorticity field (n×n), returns streamfunction.
pub fn poisson_sor(omega: &[f64], n: usize, dx: f64, n_iter: usize, omega_sor: f64) -> Vec<f64> {
    let mut psi = vec![0.0f64; n * n];
    let dx2 = dx * dx;

    for _ in 0..n_iter {
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let psi_nb = psi[(i - 1) * n + j] + psi[(i + 1) * n + j]
                    + psi[i * n + j - 1] + psi[i * n + j + 1];
                let psi_new = (psi_nb + dx2 * omega[i * n + j]) / 4.0;
                psi[i * n + j] = psi[i * n + j] + omega_sor * (psi_new - psi[i * n + j]);
            }
        }
    }
    psi
}

/// 2D vorticity advection step (explicit upwind).
/// dω/dt + u·∇ω = ν·∇²ω.
pub fn vorticity_step(
    omega: &[f64],
    psi: &[f64],
    n: usize,
    dx: f64,
    dt: f64,
    nu: f64,
) -> Vec<f64> {
    let mut omega_new = omega.to_vec();

    for i in 1..n - 1 {
        for j in 1..n - 1 {
            // Velocity from streamfunction: u = ∂ψ/∂y, v = -∂ψ/∂x
            let u = (psi[i * n + j + 1] - psi[i * n + j - 1]) / (2.0 * dx);
            let v = -(psi[(i + 1) * n + j] - psi[(i - 1) * n + j]) / (2.0 * dx);

            // Upwind advection
            let domega_dx = if u > 0.0 {
                omega[i * n + j] - omega[i * n + j - 1]
            } else {
                omega[i * n + j + 1] - omega[i * n + j]
            } / dx;
            let domega_dy = if v > 0.0 {
                omega[i * n + j] - omega[(i - 1) * n + j]
            } else {
                omega[(i + 1) * n + j] - omega[i * n + j]
            } / dx;

            // Viscous diffusion (Laplacian)
            let lap_omega = (omega[i * n + j - 1] + omega[i * n + j + 1]
                + omega[(i - 1) * n + j] + omega[(i + 1) * n + j]
                - 4.0 * omega[i * n + j]) / (dx * dx);

            omega_new[i * n + j] = omega[i * n + j]
                + dt * (-u * domega_dx - v * domega_dy + nu * lap_omega);
        }
    }
    omega_new
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6 — Special relativity
// ═══════════════════════════════════════════════════════════════════════════

/// Lorentz factor γ = 1/√(1 - v²/c²).
pub fn lorentz_factor(velocity: f64) -> f64 {
    let beta = velocity / SPEED_OF_LIGHT;
    1.0 / (1.0 - beta * beta).sqrt()
}

/// Relativistic kinetic energy: K = (γ - 1)mc².
pub fn relativistic_kinetic_energy(mass: f64, velocity: f64) -> f64 {
    (lorentz_factor(velocity) - 1.0) * mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT
}

/// Relativistic momentum: p = γmv.
pub fn relativistic_momentum(mass: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * mass * velocity
}

/// Mass-energy equivalence: E = mc².
pub fn mass_energy(mass: f64) -> f64 {
    mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT
}

/// Time dilation: Δt' = γ Δt (proper time dilates for moving clock).
pub fn time_dilation(proper_time: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * proper_time
}

/// Length contraction: L = L₀/γ (proper length contracts in direction of motion).
pub fn length_contraction(proper_length: f64, velocity: f64) -> f64 {
    proper_length / lorentz_factor(velocity)
}

/// Relativistic velocity addition: u' = (u + v)/(1 + uv/c²).
pub fn relativistic_velocity_addition(u: f64, v: f64) -> f64 {
    let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    (u + v) / (1.0 + u * v / c2)
}

/// Relativistic Doppler effect: f' = f√((1-β)/(1+β)) for recession.
pub fn relativistic_doppler(f0: f64, velocity: f64, receding: bool) -> f64 {
    let beta = velocity / SPEED_OF_LIGHT;
    if receding {
        f0 * ((1.0 - beta) / (1.0 + beta)).sqrt()
    } else {
        f0 * ((1.0 + beta) / (1.0 - beta)).sqrt()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: got {a}, expected {b} (diff={})", (a - b).abs());
    }

    fn close_rel(a: f64, b: f64, rtol: f64, label: &str) {
        let rel = if b.abs() > 1e-30 { (a - b).abs() / b.abs() } else { (a - b).abs() };
        assert!(rel < rtol, "{label}: got {a}, expected {b} (rel_err={rel})");
    }

    // ── Section 1: Classical Mechanics ──────────────────────────────────

    #[test]
    fn sho_energy_conservation() {
        // SHO: total energy = ½m(v² + ω²x²) is conserved
        let (x0, v0, omega, mass) = (1.0, 0.0, 2.0, 1.0);
        let e0 = sho_energy(mass, omega, x0, v0);
        for &t in &[0.1, 0.5, 1.0, 2.0, 10.0] {
            let (x, v) = sho_exact(x0, v0, omega, t);
            let e = sho_energy(mass, omega, x, v);
            close(e, e0, 1e-10, &format!("SHO energy at t={t}"));
        }
    }

    #[test]
    fn sho_period() {
        // Period T = 2π/ω
        let (x0, v0, omega) = (1.0, 0.0, 3.0);
        let period = TAU / omega;
        let (x1, v1) = sho_exact(x0, v0, omega, period);
        close(x1, x0, 1e-10, "SHO x after period");
        close(v1, v0, 1e-10, "SHO v after period");
    }

    #[test]
    fn dho_decays() {
        // Underdamped oscillator should decay exponentially
        let (x0, v0, omega0, gamma) = (1.0, 0.0, 5.0, 0.5);
        let x1 = dho_underdamped(x0, v0, omega0, gamma, 1.0);
        let x_max = (-gamma * 1.0_f64).exp(); // envelope
        assert!(x1.abs() <= x_max * 1.01, "DHO should be within envelope");
    }

    #[test]
    fn kepler_circular_orbit() {
        // Circular orbit: E = -GMm/(2a), e = 0
        let r = 1.5e11; // ~1 AU
        let m_sun = 1.989e30;
        let v_circ = (G_GRAV * m_sun / r).sqrt();
        let orbit = kepler_orbit([r, 0.0, 0.0], [0.0, v_circ, 0.0], m_sun);
        close_rel(orbit.semi_major_axis, r, 1e-6, "Circular orbit SMA = r");
        close(orbit.eccentricity, 0.0, 1e-6, "Circular orbit e = 0");
        assert!(orbit.period > 0.0, "Positive period");
    }

    #[test]
    fn nbody_2body_energy_conserved() {
        // 2-body: energy should be approximately conserved with small dt
        let particles = vec![
            Particle::new(1e30, [0.0, 0.0, 0.0], [0.0, 1000.0, 0.0]),
            Particle::new(1e20, [1e10, 0.0, 0.0], [0.0, -1e7, 0.0]),
        ];
        let res = nbody_gravity(&particles, 1000.0, 100);
        let e0 = res.total_energy[0];
        let e_last = *res.total_energy.last().unwrap();
        // Energy should be conserved within ~1% for small timestep
        let rel_err = (e_last - e0).abs() / e0.abs();
        assert!(rel_err < 0.05, "N-body energy drift={rel_err} should be < 5%");
    }

    #[test]
    fn rigid_body_kinetic_energy_conserved() {
        // Torque-free rigid body: kinetic energy is conserved
        let omega0 = [1.0, 0.5, 0.1];
        let i_moi = [1.0, 2.0, 3.0];
        let e0 = rotational_kinetic_energy(omega0, i_moi);
        let traj = euler_rotation(omega0, i_moi, 0.001, 1000);
        let e_last = rotational_kinetic_energy(*traj.last().unwrap(), i_moi);
        close_rel(e_last, e0, 0.01, "Rigid body KE conserved");
    }

    #[test]
    fn double_pendulum_energy_conserved() {
        // Use small dt=0.001 for 200 steps (0.2 s). RK4 should conserve to <0.1%.
        let s0 = DoublePendulumState { theta1: 0.5, theta2: 0.3, omega1: 0.0, omega2: 0.0 };
        let (m, l, g) = (1.0, 1.0, 9.81);
        let e0 = double_pendulum_energy(&s0, m, l, g);
        let traj = double_pendulum_rk4(&s0, g, l, 0.001, 200);
        let e_last = double_pendulum_energy(traj.last().unwrap(), m, l, g);
        close_rel(e_last, e0, 0.002, "Double pendulum energy conserved with dt=0.001");
    }

    // ── Section 2: Thermodynamics ───────────────────────────────────────

    #[test]
    fn ideal_gas_law() {
        // PV = nRT: check round-trip
        let (n, t, v) = (1.0, 300.0, 0.025);
        let p = ideal_gas_pressure(n, t, v);
        let t_back = ideal_gas_temperature(p, v, n);
        close(t_back, t, 1e-6, "Ideal gas round-trip T");
    }

    #[test]
    fn carnot_efficiency_range() {
        let eta = carnot_efficiency(500.0, 300.0);
        close(eta, 0.4, 1e-10, "Carnot efficiency = 1 - 300/500");
        assert!(eta < 1.0 && eta > 0.0, "Efficiency in (0,1)");
    }

    #[test]
    fn newton_cooling_limit() {
        // As t→∞, T → T_env
        let t_inf = newton_cooling(100.0, 20.0, 0.1, 1000.0);
        close(t_inf, 20.0, 1e-6, "Newton cooling converges to T_env");
    }

    #[test]
    fn vdw_critical_point() {
        // For CO₂: a ≈ 0.3658 Pa·m⁶/mol², b ≈ 4.286e-5 m³/mol
        let a = 0.3658;
        let b = 4.286e-5;
        let (t_c, _p_c, _v_c) = vdw_critical(a, b);
        // Known T_c ≈ 304.2 K
        close_rel(t_c, 304.2, 0.05, "CO₂ critical temperature");
    }

    // ── Section 3: Statistical Mechanics ───────────────────────────────

    #[test]
    fn partition_function_two_level() {
        // Two-level system: E = {0, ε}. Z = 1 + exp(-βε)
        let eps = 1e-21; // ~small energy
        let beta = 1.0 / (K_BOLTZMANN * 300.0);
        let z = partition_function(&[0.0, eps], beta);
        let z_exact = 1.0 + (-beta * eps).exp();
        close_rel(z, z_exact, 1e-10, "Two-level Z");
    }

    #[test]
    fn boltzmann_probabilities_sum_to_one() {
        let energies = vec![0.0, 1e-21, 2e-21, 3e-21];
        let beta = 1.0 / (K_BOLTZMANN * 300.0);
        let probs = boltzmann_probabilities(&energies, beta);
        let sum: f64 = probs.iter().sum();
        close(sum, 1.0, 1e-10, "Boltzmann probs sum to 1");
    }

    #[test]
    fn mean_energy_two_level() {
        // Two-level: ⟨E⟩ = ε exp(-βε) / (1 + exp(-βε))
        let eps = 1e-21;
        let beta = 1.0 / (K_BOLTZMANN * 300.0);
        let e_mean = mean_energy(&[0.0, eps], beta);
        let e_exact = eps * (-beta * eps).exp() / (1.0 + (-beta * eps).exp());
        close_rel(e_mean, e_exact, 1e-8, "Two-level mean energy");
    }

    #[test]
    fn ising1d_zero_field_symmetry() {
        // At h=0, magnetization should be 0 (symmetric)
        let (j, h, beta) = (1.0, 0.0, 1.0);
        let (_, m, _) = ising1d_exact(j, h, beta);
        close(m, 0.0, 1e-8, "1D Ising m=0 at h=0");
    }

    #[test]
    fn ising1d_high_t_low_m() {
        // At high temperature, magnetization should be small even with small field
        let (j, h, beta) = (1.0, 0.01, 0.01); // very high T
        let (_, m, _) = ising1d_exact(j, h, beta);
        assert!(m.abs() < 0.5, "High-T 1D Ising should have low |m|={m}");
    }

    #[test]
    fn ising2d_ferromagnetic_phase() {
        // Below critical T (~2.27J for J=1), should have |m| > 0 starting ferromagnetic
        let (n, j, h, t, sweeps) = (10, 1.0, 0.0, 1.5, 500);
        let (m, _e) = ising2d_metropolis(n, j, h, t, sweeps, 42);
        // Starting ferromagnetic and below T_c, should stay ordered
        assert!(m.abs() > 0.3, "2D Ising below T_c: |m|={m} should be > 0.3");
    }

    #[test]
    fn ising2d_paramagnetic_phase() {
        // Above critical T (~2.27J), should have near-zero magnetization
        let (n, j, h, t, sweeps) = (10, 1.0, 0.0, 5.0, 2000);
        let (m, _e) = ising2d_metropolis(n, j, h, t, sweeps, 42);
        assert!(m.abs() < 0.5, "2D Ising above T_c: |m|={m} should be small");
    }

    #[test]
    fn qho_energy_levels() {
        // QHO: E_n = ℏω(n + ½)
        let omega = 1e14; // optical frequency
        for n in 0..5 {
            let e = qho_energy(n, omega);
            let expected = H_BAR * omega * (n as f64 + 0.5);
            close(e, expected, 1e-40, &format!("QHO E_{n}"));
        }
    }

    #[test]
    fn boltzmann_temperature_equilibrium() {
        // At high T: Boltzmann → uniform. At low T: ground state dominates.
        let energies = vec![0.0, 1e-21, 2e-21];
        let beta_high = 1.0 / (K_BOLTZMANN * 1e10); // very high T
        let probs_high = boltzmann_probabilities(&energies, beta_high);
        // Should be nearly uniform
        for p in &probs_high { assert!((p - 1.0 / 3.0).abs() < 0.1, "High-T uniform: {p}"); }

        let beta_low = 1.0 / (K_BOLTZMANN * 0.001); // very low T
        let probs_low = boltzmann_probabilities(&energies, beta_low);
        // Ground state should dominate
        assert!(probs_low[0] > 0.99, "Low-T ground state dominates: p0={}", probs_low[0]);
    }

    // ── Section 4: Quantum Mechanics ────────────────────────────────────

    #[test]
    fn hydrogen_energy_levels() {
        // E_1 = -13.606 eV
        close(hydrogen_energy_ev(1), HYDROGEN_GROUND_EV, 1e-3, "H ground state energy");
        // E_2 = E_1/4
        close(hydrogen_energy_ev(2), HYDROGEN_GROUND_EV / 4.0, 1e-3, "H n=2 energy");
    }

    #[test]
    fn hydrogen_balmer_wavelength() {
        // Hα: 3→2 transition ≈ 656.3 nm
        let lambda = hydrogen_wavelength(3, 2);
        close_rel(lambda, 656.3e-9, 0.001, "H-alpha wavelength");
    }

    #[test]
    fn particle_in_box_energy_levels() {
        // E_n ∝ n²
        let (m, l) = (MASS_ELECTRON, 1e-9);
        let e1 = particle_in_box_energy(1, m, l);
        let e2 = particle_in_box_energy(2, m, l);
        close_rel(e2, 4.0 * e1, 1e-10, "PIB E_2 = 4 E_1");
    }

    #[test]
    fn tunneling_below_barrier() {
        // Transmission should be < 1 for E < V0.
        // Use SI: E = 0.5 eV = 8e-20 J, V0 = 1 eV = 1.6e-19 J, L = 2 Å = 2e-10 m.
        // κ = √(2m(V0-E))/ℏ ≈ 3.6e9 m⁻¹, κL ≈ 0.72 → T ≈ 0.62 (measurable tunneling).
        let e = 8.0e-20;
        let v0 = 1.6e-19;
        let l = 2.0e-10;
        let t = tunneling_transmission(e, v0, MASS_ELECTRON, l);
        assert!(t > 0.0 && t < 1.0, "Tunneling T={t} should be in (0,1)");
    }

    #[test]
    fn tunneling_above_barrier() {
        // E > V0: classical transmission = 1
        let t = tunneling_transmission(2.0e-19, 1.6e-19, MASS_ELECTRON, 1e-9);
        close(t, 1.0, 1e-10, "E > V0: T = 1");
    }

    #[test]
    fn time_evolution_probability_conserved() {
        // |ψ|² should sum to 1 after time evolution
        let amps = vec![Amplitude(1.0 / 2.0_f64.sqrt(), 0.0), Amplitude(0.0, 1.0 / 2.0_f64.sqrt())];
        let energies = vec![1e-20, 2e-20];
        let evolved = time_evolve_state(&amps, &energies, 1e-14);
        let prob: f64 = evolved.iter().map(|a| a.norm_sq()).sum();
        close(prob, 1.0, 1e-10, "Time evolution preserves probability");
    }

    #[test]
    fn von_neumann_entropy_pure_state() {
        // Pure state: S = 0
        let probs = vec![1.0, 0.0, 0.0];
        close(von_neumann_entropy_diagonal(&probs), 0.0, 1e-10, "Pure state S=0");
    }

    #[test]
    fn von_neumann_entropy_maximally_mixed() {
        // Maximally mixed: S = ln(d)
        let probs = vec![1.0 / 3.0; 3];
        let s = von_neumann_entropy_diagonal(&probs);
        close(s, (3.0_f64).ln(), 1e-10, "Max mixed S = ln(3)");
    }

    // ── Section 5: Fluid Dynamics ────────────────────────────────────────

    #[test]
    fn reynolds_laminar_turbulent() {
        // Re = ρvL/μ. Water: ρ=1000 kg/m³, μ=0.001 Pa·s
        // Laminar: v=0.001 m/s, L=0.01 m → Re = 1000×0.001×0.01/0.001 = 10
        let re_lam = reynolds_number(1000.0, 0.001, 0.01, 0.001);
        assert!(re_lam < 2300.0, "Low Re={re_lam} should be laminar");
        // Turbulent: v=5 m/s, L=0.1 m → Re = 1000×5×0.1/0.001 = 500000
        let re_turb = reynolds_number(1000.0, 5.0, 0.1, 0.001);
        assert!(re_turb > 4000.0, "High Re={re_turb} should be turbulent");
    }

    #[test]
    fn bernoulli_venturi() {
        // Venturi effect: faster flow has lower pressure
        // v₂ > v₁ when A₁ > A₂ (continuity: A₁v₁ = A₂v₂)
        let (p1, v1, h): (f64, f64, f64) = (101325.0, 1.0, 0.0);
        let p2 = p1 - 0.5 * 1000.0 * (4.0_f64.powi(2) - v1.powi(2)); // v₂=4 m/s → lower P
        let v2 = bernoulli_velocity(p1, p2, v1, h, h, 1000.0);
        close(v2, 4.0, 0.01, "Bernoulli Venturi: v₂=4 m/s");
    }

    #[test]
    fn poiseuille_flow_rate_scaling() {
        // Q ∝ r⁴: doubling radius → 16x flow rate
        let (r1, r2) = (0.01, 0.02);
        let q1 = poiseuille_flow_rate(r1, 1000.0, 0.001, 0.1);
        let q2 = poiseuille_flow_rate(r2, 1000.0, 0.001, 0.1);
        close_rel(q2 / q1, 16.0, 1e-10, "Poiseuille Q ∝ r⁴");
    }

    // ── Section 6: Special Relativity ────────────────────────────────────

    #[test]
    fn lorentz_factor_low_velocity() {
        // At v << c: γ ≈ 1
        let gamma = lorentz_factor(1000.0); // 1 km/s
        close(gamma, 1.0, 1e-8, "γ ≈ 1 at v << c");
    }

    #[test]
    fn lorentz_factor_high_velocity() {
        // At v = 0.99c: γ ≈ 7.09
        let gamma = lorentz_factor(0.99 * SPEED_OF_LIGHT);
        close_rel(gamma, 7.0888, 0.001, "γ at 0.99c");
    }

    #[test]
    fn mass_energy_electron() {
        // m_e c² ≈ 0.511 MeV = 8.187e-14 J
        let e = mass_energy(MASS_ELECTRON);
        close_rel(e, 8.187e-14, 0.001, "Electron rest energy");
    }

    #[test]
    fn relativistic_velocity_addition_below_c() {
        // Adding two velocities should not exceed c
        let v1 = 0.8 * SPEED_OF_LIGHT;
        let v2 = 0.9 * SPEED_OF_LIGHT;
        let v_combined = relativistic_velocity_addition(v1, v2);
        assert!(v_combined < SPEED_OF_LIGHT, "Combined velocity must be < c: v={v_combined}");
    }

    #[test]
    fn time_dilation_factor() {
        // Proper time is shorter for moving clock: Δt' = γ Δt
        let v = 0.6 * SPEED_OF_LIGHT;
        let gamma = lorentz_factor(v); // = 1.25
        let dilated = time_dilation(1.0, v);
        close(dilated, gamma, 1e-10, "Time dilation = γ");
    }

    #[test]
    fn length_contraction_factor() {
        let v = 0.6 * SPEED_OF_LIGHT;
        let gamma = lorentz_factor(v);
        let contracted = length_contraction(1.0, v);
        close(contracted, 1.0 / gamma, 1e-10, "Length contraction = 1/γ");
    }

    #[test]
    fn particle_in_box_wf_normalization() {
        // ∫₀ᴸ |ψ_n(x)|² dx ≈ 1
        let (n_state, m_box, l) = (1u32, MASS_ELECTRON, 1.0);
        let n_pts = 10000;
        let dx = l / n_pts as f64;
        let norm: f64 = (0..n_pts)
            .map(|i| {
                let x = (i as f64 + 0.5) * dx;
                particle_in_box_wf(n_state, x, l).powi(2) * dx
            })
            .sum();
        close(norm, 1.0, 0.001, "PIB wavefunction normalization");
        let _ = m_box;
    }

    #[test]
    fn arrhenius_temperature_dependence() {
        // Higher T → faster rate
        let k1 = arrhenius(1e13, 80000.0, 300.0);
        let k2 = arrhenius(1e13, 80000.0, 400.0);
        assert!(k2 > k1, "Arrhenius: higher T gives larger k");
    }

    #[test]
    fn carnot_is_maximum_efficiency() {
        // Carnot efficiency must be < 1
        let eta = carnot_efficiency(1000.0, 300.0);
        assert!(eta > 0.0 && eta < 1.0, "Carnot η={eta} in (0,1)");
        close_rel(eta, 0.7, 1e-10, "Carnot 1 - 300/1000 = 0.7");
    }

    #[test]
    fn stefan_boltzmann_scaling() {
        // Power scales as T⁴
        let p1 = stefan_boltzmann(1.0, 1.0, 1000.0);
        let p2 = stefan_boltzmann(1.0, 1.0, 2000.0);
        close_rel(p2 / p1, 16.0, 1e-6, "Stefan-Boltzmann T⁴ scaling");
    }

    #[test]
    fn schrodinger1d_ordering() {
        // Verify eigenvalue ordering E_0 < E_1 < E_2 for a harmonic potential.
        // Use mass = H_BAR² to make T_diag = 1/dx² (dimensionless-scale kinetic energy).
        // V(x) = ½ x² (dimensionless harmonic) — this forces a non-trivial potential.
        let n = 60;
        let l = 4.0; // box [-4, 4] in natural units
        let dx = 2.0 * l / n as f64;
        let x_grid: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
        // mass = H_BAR^2 makes T_diag = ℏ²/(ℏ² dx²) = 1/dx² (order 1 for dx~0.1)
        let mass = H_BAR * H_BAR;
        let v: Vec<f64> = x_grid.iter().map(|&x| 0.5 * x * x).collect();
        let (evals, _evecs) = schrodinger1d(&v, dx, mass, 4);
        assert_eq!(evals.len(), 4);
        assert!(evals[0] < evals[1], "E_0={} < E_1={}", evals[0], evals[1]);
        assert!(evals[1] < evals[2], "E_1={} < E_2={}", evals[1], evals[2]);
        assert!(evals[2] < evals[3], "E_2={} < E_3={}", evals[2], evals[3]);
        // All eigenvalues should be positive (V ≥ 0, kinetic ≥ 0)
        assert!(evals[0] >= 0.0, "Ground state energy ≥ 0: E_0={}", evals[0]);
    }
}
