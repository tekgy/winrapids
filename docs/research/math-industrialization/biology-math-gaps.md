# Biology/Medicine Math Gaps — Tambear Coverage Analysis

**Date**: 2026-04-10
**Scope**: What mathematical primitives biology, medicine, and life sciences need that tambear doesn't yet have.
**Method**: Grepped tambear/src/ for existing implementations; web-searched each subfield for canonical primitives.

---

## Executive Summary

Tambear has strong foundations in the underlying math that biology uses: ODE solvers (euler, rk4, rk45, rk4_system), stochastic processes (Brownian motion, Poisson process, CTMC), HMM (forward, Viterbi, Baum-Welch), survival analysis (Kaplan-Meier, Cox PH, log-rank), Markov chains, and basic graph algorithms. It does NOT have the domain-specific biology primitives that compose these into life-science computations. The gap is a second layer: biology-specific data structures, scoring matrices, and algorithms built from primitives we already have.

**Key insight**: The highest-value additions are primitives that are shared across multiple biology subfields. Pairwise distance matrices (already have), dynamic programming (ODE solvers exist, but DP table machinery is missing), exponential-family rate functions, and sequence statistics are the cross-cutting atoms.

---

## What Tambear Already Has (Biology-Relevant)

| Primitive | Location | Biology use |
|-----------|----------|-------------|
| rk4, rk45, rk4_system | numerical.rs | ODEs for all compartmental models |
| euler | numerical.rs | Euler-Maruyama for SDEs |
| brownian_motion, ornstein_uhlenbeck, geometric_brownian_motion | stochastic.rs | Langevin dynamics, diffusion models |
| nonhomogeneous_poisson, poisson_process | stochastic.rs | Gillespie reaction times |
| ctmc_transition_matrix, ctmc_stationary | stochastic.rs | Gene regulatory networks |
| markov_n_step, stationary_distribution | stochastic.rs | Population Markov chains |
| HMM (forward/backward/Viterbi/Baum-Welch) | hmm.rs + kalman.rs | Sequence profiles, gene finding |
| kaplan_meier, cox_ph, log_rank_test | survival.rs | Clinical trials, time-to-event |
| pairwise_dists, knn_adjacency | graph.rs | Sequence distance, BLAST |
| rips_h0, rips_h1, persistence_entropy | tda.rs | TDA on biological networks |
| morans_i, ordinary_kriging | spatial.rs | Spatial epidemiology |
| partition_function, boltzmann_probabilities | physics.rs | Statistical mechanics for folding |
| ising2d_metropolis | physics.rs | Monte Carlo, spin glass models |
| arrhenius | physics.rs | Reaction kinetics |
| double_pendulum_rk4, nbody_gravity | physics.rs | Velocity-Verlet (partial) |

---

## Gaps by Subfield

### 1. Bioinformatics — Sequence Alignment

**Missing primitives** (all are DP table algorithms — the primitive is the recurrence + traceback pattern):

- `needleman_wunsch(seq_a, seq_b, scoring_matrix, gap_open, gap_extend)` — global alignment, O(mn) DP. Accumulate: fill scoring matrix. Gather: traceback.
- `smith_waterman(seq_a, seq_b, scoring_matrix, gap_open, gap_extend)` — local alignment, same DP with floor at 0.
- `affine_gap_alignment(seq_a, seq_b, scoring_matrix, gap_open, gap_extend)` — Gotoh's O(mn) extension. Three DP tables (M, X, Y).
- `substitution_matrix_blosum62()`, `substitution_matrix_pam250()`, `substitution_matrix_nuc44()` — standard scoring tables, not derived from code.
- `edit_distance(a, b)` — Levenshtein, degenerate case of NW with unit costs.
- `longest_common_subsequence(a, b)` — DP, primitive used by diff algorithms too.

**Cross-field note**: The DP recurrence pattern (fill matrix from top-left, traceback from terminal cell) is shared with RNA secondary structure (Nussinov, CYK parser), protein threading, and many OR problems. The primitive is: `dp_table_2d(m, n, recurrence_fn, traceback_fn)` — worth extracting as a general combinator.

---

### 2. Bioinformatics — Phylogenetics

**Missing primitives**:

- `upgma(distance_matrix, labels)` — hierarchical clustering with averaging linkage, returns a rooted dendrogram. Accumulate+gather: at each step find min off-diagonal, merge two clusters. We have `kruskal` and general hierarchical linkage in clustering.rs — UPGMA is a specialized variant with average-linkage and ultrametric output.
- `neighbor_joining(distance_matrix, labels)` — O(n³), unrooted tree. Compute Q-matrix (transformed distances), find minimum, merge. Additive rather than ultrametric output. Different from UPGMA — produces better trees when molecular clock fails.
- `jukes_cantor_distance(p_diff)`, `kimura_2parameter(transitions, transversions)`, `tajima_nei_distance(p_diff, base_freqs)` — substitution model distances that convert raw sequence similarity to evolutionary distance. Input to UPGMA/NJ.
- `maximum_parsimony_fitch(tree, sequences)` — Fitch algorithm, two-pass DP on a tree. Uses our existing graph primitives.
- `newick_serialize(tree)`, `newick_parse(string)` — tree format I/O.

**Cross-field note**: UPGMA is just average-linkage hierarchical clustering (which we have). The new primitive here is specifically the ultrametric output format and phylogenetic interpretation. NJ is genuinely new — the Q-matrix transformation has no analog elsewhere.

---

### 3. Reaction Kinetics / Systems Biology

**Missing primitives** (this is the biggest gap — we have `arrhenius` and Poisson but not the kinetic simulation machinery):

- `mass_action_rhs(species_counts, stoich_matrix, rate_constants)` — computes dx/dt for deterministic ODE system. Accumulate: for each reaction, multiply reactant concentrations by rate constant. Gather: net stoichiometric flux per species. This is the kernel that drives all ODE-based kinetics (SIR, Lotka-Volterra, Michaelis-Menten systems, gene circuits). Currently we have `arrhenius` for computing a single rate constant but not the ODE right-hand side machinery.
- `michaelis_menten_rhs(substrate, vmax, km)` — single-enzyme kinetics: v = Vmax·S/(Km + S). Simple scalar but ubiquitous. Accumulate+gather: straightforward.
- `hill_function(substrate, n, k_half)` — cooperative binding: f = S^n / (K^n + S^n). Used everywhere in gene regulation.
- `gillespie_ssa(stoich_matrix, rate_constants, initial_counts, t_end, seed)` — stochastic simulation algorithm. State: species counts (integers). Event: exponential waiting time `τ ~ Exp(a_total)`, reaction selection proportional to `a_j`. We have `nonhomogeneous_poisson` but not the multi-reaction SSA with stoichiometry updates.
- `tau_leaping(stoich_matrix, propensities, initial_counts, tau, t_end, seed)` — approximate SSA: batch multiple reactions in time step tau. Poisson-distributed reaction counts. 10-100x faster than SSA for large systems.
- `chemical_master_equation_tridiag(stoich, rates, max_n)` — probability distribution P(n,t) for simple birth-death systems. Solvable with our CTMC machinery.
- `lotka_volterra_rhs(x, y, alpha, beta, gamma, delta)` — predator-prey two-species ODE. Composes with rk4_system. Pure formula but named and needed.
- `sir_rhs(s, i, r, beta, gamma)` — SIR epidemic three-compartment ODE. Composes with rk4_system.
- `seir_rhs(s, e, i, r, beta, sigma, gamma)` — SEIR, adds exposed compartment.
- `seird_rhs(s, e, i, r, d, beta, sigma, gamma, mu)` — with mortality.
- `r0_from_sir(beta, gamma)` — basic reproduction number: R0 = β/γ.

**Cross-field note**: `mass_action_rhs` + `gillespie_ssa` are the two highest-value primitives here. `mass_action_rhs` is the kernel for ALL deterministic biological ODE systems (SIR, Lotka-Volterra, gene circuits, SEIR, pharmacokinetics, metabolic flux). `gillespie_ssa` unlocks stochastic versions of all of them. Both are missing and share no code with anything we have.

---

### 4. Population Genetics

**Missing primitives**:

- `wright_fisher_generation(allele_freq, pop_size, selection_coeff, mutation_rates, seed)` — binomial sampling of next generation. Uses our binomial (or can be done with our rng). Accumulate: sample. Gather: new frequency.
- `coalescent_times(n_lineages, pop_size, seed)` — Kingman coalescent: waiting time to next merger is `Exp(C(k,2)/N)`. Iterates from k=n_lineages down to 1. Returns merge times and topology.
- `hardy_weinberg_chi2(counts_aa, counts_ab, counts_bb)` — tests HWE departure. Uses our chi-squared test.
- `fst_weir_cockerham(allele_counts_by_pop)` — fixation index, measures population differentiation. Composes from moments.
- `tajimas_d(segregating_sites, n_sequences, pairwise_differences)` — neutrality test. Formula applied to sequence summary stats.
- `linkage_disequilibrium_r2(haplotype_counts_2x2)` — LD between two loci. Composes from joint frequencies.
- `nucleotide_diversity_pi(sequences)` — average pairwise differences. Accumulate: all pairs. Gather: mean.

**Cross-field note**: Wright-Fisher is just repeated binomial sampling — we have the binomial primitive. Coalescent is just exponential waiting times — we have the exponential distribution. The new content is the biological interpretation and the specific simulation loop. Lower primitivity than Gillespie.

---

### 5. Molecular Dynamics

**Missing primitives** (we have velocity-Verlet in `nbody_gravity` but only for gravity — the MD-specific extensions are missing):

- `verlet_md_step(positions, velocities, forces, masses, dt)` — general velocity-Verlet step for any force field (not gravity-specific). The `nbody_gravity` function computes both force AND integrates; what's missing is the decomposed primitive: `compute_forces(positions, masses, pair_potential_fn)` + `verlet_step(...)`.
- `lennard_jones_pair(r, epsilon, sigma)` — LJ potential: 4ε[(σ/r)¹² - (σ/r)⁶]. Single most common MD force. Accumulate: sum over pairs. Gather: force vectors.
- `nose_hoover_thermostat(velocities, kinetic_energy, target_temp, tau, xi, dt)` — deterministic thermostat coupling. Adds a friction variable ξ to the equations. More accurate than velocity rescaling.
- `langevin_thermostat_step(velocities, forces, masses, gamma_friction, target_temp, dt, seed)` — stochastic thermostat: adds friction -γv and random kicks. Composes with our Brownian motion.
- `minimum_image_convention(dr, box_length)` — periodic boundary conditions. Maps displacement to nearest image in periodic box.
- `ewald_summation(charges, positions, box, alpha, k_max)` — long-range electrostatics in periodic systems. Split into real-space (short-range) + reciprocal-space (Fourier). Uses our FFT.
- `radial_distribution_function(positions, box, n_bins, r_max)` — g(r) structure factor. Accumulate: pair distance histogram. Normalize by ideal gas density. Key observable.
- `mean_squared_displacement(trajectories)` — MSD vs time lag. Accumulate: for each time lag τ, mean of |r(t+τ)-r(t)|². Used for diffusion coefficient.
- `free_energy_perturbation(E_A, E_B, beta)` — ΔF = -kT ln⟨exp(-β(E_B-E_A))⟩_A. Bennett acceptance ratio is more accurate: `bar_estimator(delta_E_AB, delta_E_BA, beta)`.

**Cross-field note**: `verlet_md_step` + `lennard_jones_pair` + `minimum_image_convention` are the three atoms for any MD simulation. They compose with our existing rng and rk4 infrastructure. `radial_distribution_function` is the histogram accumulate+gather pattern applied to pair distances.

---

### 6. Neuroscience

**Missing primitives**:

- `hodgkin_huxley_rhs(V, m, h, n, I_ext, params)` — four coupled ODEs for membrane voltage and gating variables m, h, n. Composes with rk4_system. Ion channel conductances: gNa·m³h(V-VNa), gK·n⁴(V-VK), gL(V-VL).
- `fitzhugh_nagumo_rhs(V, w, I_ext, a, b, tau)` — two-variable simplification of HH. Same structure.
- `alpha_beta_gates(V, channel_type)` — voltage-dependent rate functions α(V) and β(V) for HH gating variables. Accumulate: not applicable (scalar functions). The formula is the primitive.
- `spike_detect(voltage_trace, threshold, refractory_ms, dt)` — threshold crossing detection with refractory period. Accumulate: scan for crossings. Gather: spike times.
- `spike_train_isi(spike_times)` — inter-spike intervals: differences of sorted spike times. Composes from our diff primitive.
- `firing_rate_histogram(spike_times, bin_width, t_end)` — PSTH. Accumulate: count spikes per bin. Gather: rates. Composes from our histogram primitive.
- `phase_locking_value(phase_a, phase_b)` — PLV = |mean(exp(i(φ_A - φ_B)))|. Measures phase synchrony. Uses complex exponential.
- `coherence_spectrum(x, y, fs, nperseg)` — magnitude-squared coherence Cxy(f) = |Pxy(f)|²/(Pxx(f)Pyy(f)). Uses our FFT/PSD.
- `granger_causality_ols(x, y, max_lag)` — whether past of x predicts y beyond y's own past. Ratio of residual variances from bivariate vs univariate AR. Composes from our linear regression.
- `neural_field_1d(u, w_kernel, input, dt, dx, tau, nonlinearity)` — continuum neural field equation. ∂u/∂t = -u + W*σ(u) + I. W* is convolution. Uses our FFT for efficient convolution.

**Cross-field note**: `hodgkin_huxley_rhs` + `spike_detect` are the two new atoms. Everything else (ISI, PSTH, coherence, Granger) composes from primitives we already have (diff, histogram, FFT, linear regression). HH is genuinely new because the specific gating variable equations have no analog elsewhere.

---

### 7. Ecology

**Missing primitives** (mostly ODE RHS formulas — composable with rk4_system):

- `lotka_volterra_nspecies_rhs(pop, growth_rates, interaction_matrix)` — N-species generalization. Each species: dx_i/dt = r_i·x_i + x_i·Σ_j a_ij·x_j. Accumulate: matrix-vector product. Gather: per-species rates.
- `logistic_growth_rhs(n, r, k)` — single-species: dN/dt = rN(1 - N/K). One line but needed by name.
- `allee_effect_rhs(n, r, k, a)` — strong Allee: dN/dt = rN(N/A - 1)(1 - N/K). Extinction below threshold.
- `age_structured_leslie_matrix(survival_rates, fecundities)` — discrete-time population projection. Matrix power applied to age vector. Composes from our linear algebra.
- `species_area_curve(area, c, z)` — S = cA^z. Biogeography. Trivial formula.
- `shannon_diversity(counts)` — H' = -Σ p_i ln p_i. Already have `shannon_entropy` — this is the same function applied to count data.
- `simpson_diversity(counts)` — D = 1 - Σ(n_i(n_i-1)) / (N(N-1)). New formula.
- `chao1_richness(n_observed, f1, f2)` — nonparametric species richness estimator: S_chao1 = S_obs + f1²/(2f2).

**Cross-field note**: Most ecology primitives compose trivially from what we have. The genuinely new content is `lotka_volterra_nspecies_rhs` as a named primitive (the ODE RHS formula, composing with rk4_system), and the diversity/richness estimators. Leslie matrix is already doable with our matrix multiplication.

---

### 8. Medical Imaging

**Missing primitives** (none of these are in tambear — this is a fully new subfield):

- `radon_transform(image, thetas)` — line integrals through an image at each angle. Accumulate: for each (θ, t) integrate along the line. The sinogram. Core of CT reconstruction.
- `filtered_back_projection(sinogram, thetas, filter_type)` — inverse Radon. Apply ramp filter (or Hann, Shepp-Logan) to sinogram, back-project. Uses our FFT. Compose: FFT each projection, multiply by filter, iFFT, back-project.
- `ramp_filter(n, filter_type)` — frequency-domain filter for FBP. Types: Ram-Lak (ideal ramp), Shepp-Logan, Hann, Cosine. Uses our FFT frequency grid.
- `back_project(filtered_sinogram, thetas, image_size)` — accumulate weighted contributions from each projection angle.
- `k_space_to_image(kspace)` — 2D iFFT of complex k-space data (MRI). Uses our FFT.
- `sense_reconstruction(kspace, coil_sensitivities)` — multi-coil MRI reconstruction. Combines coil-weighted projections.
- `compressed_sensing_recovery(measurements, sensing_matrix, lambda, max_iter)` — L1-minimization for under-sampled MRI. Composes from our ADMM/proximal gradient (which we have in optimization.rs).
- `hounsfield_to_attenuation(hu)` — linear conversion: μ = HU/1000 + 1. CT calibration.
- `point_spread_function(sigma, size)` — Gaussian PSF for blur modeling. Composes from our Gaussian.

**Cross-field note**: `radon_transform` + `filtered_back_projection` + `ramp_filter` are the three genuine new atoms. They compose from FFT (which we have) but the specific geometric accumulation pattern (line integral through an image) is new. This is the highest-novelty gap in the whole list.

---

### 9. Drug Discovery

**Missing primitives**:

- `tanimoto_coefficient(fp_a, fp_b)` — Jaccard similarity for binary fingerprints: |A∩B|/|A∪B|. Accumulate: bitwise AND/OR counts. Gather: ratio. The standard molecular similarity metric.
- `morgan_fingerprint(smiles, radius)` — circular fingerprint generation. Requires SMILES parser (out of scope for tambear, but the hashing step is primitive: hash neighborhood of each atom).
- `dice_coefficient(fp_a, fp_b)` — 2|A∩B|/(|A|+|B|). Alternative to Tanimoto.
- `pk_one_compartment(dose, kel, ka, vd, t)` — one-compartment PK model: C(t) = (F·D·ka)/(Vd(ka-kel))·(e^{-kel·t} - e^{-ka·t}). Oral absorption.
- `pk_two_compartment_rhs(c1, c2, dose_rate, k12, k21, kel)` — two-compartment IV PK ODE. Composes with rk4_system.
- `auc_trapezoid(times, concentrations)` — area under the pharmacokinetic curve. Already have `trapezoid` in numerical.rs — this is just an application.
- `hill_equation(dose, emax, ed50, n)` — sigmoidal dose-response: E = Emax·D^n/(ED50^n + D^n). Same as `hill_function` above.

**Cross-field note**: `tanimoto_coefficient` and `pk_one_compartment` / `pk_two_compartment_rhs` are the new atoms. Tanimoto is the fingerprint similarity primitive used in all cheminformatics. PK models are ODE compositions.

---

## Priority Ranking — Highest-Value Additions

Ranked by: (a) shared across multiple subfields, (b) genuinely new code (not just formula application), (c) frequency of use in scientific literature.

### Tier 1 — Cross-field atoms, genuinely new code

1. **`gillespie_ssa`** — serves reaction kinetics, gene regulation, epidemiology, ecology. Uses Poisson + stoichiometry update. Missing.
2. **`mass_action_rhs`** — the ODE kernel for ALL deterministic biological models. Single function drives SIR, Lotka-Volterra, gene circuits, metabolic flux. Missing.
3. **`needleman_wunsch` + `smith_waterman`** — with affine gap (Gotoh). DP on sequence pairs. The atom for all sequence alignment. Missing. The general `dp_table_2d` combinator is worth extracting.
4. **`radon_transform` + `filtered_back_projection`** — CT reconstruction. Line integral geometry is new; uses our FFT. Missing.
5. **`verlet_md_step` + `lennard_jones_pair` + `minimum_image_convention`** — MD triple, composes with our existing ODE infrastructure. Missing.

### Tier 2 — Domain-specific but high-frequency

6. **`hodgkin_huxley_rhs`** — gating variable equations, foundation of all computational neuroscience. Composes with rk4_system.
7. **`neighbor_joining`** — better phylogenetic tree reconstruction than UPGMA, no analog elsewhere.
8. **`nose_hoover_thermostat`** — deterministic thermostat, needed for NVT ensemble MD. Composes with verlet.
9. **`tau_leaping`** — approximate SSA, 10-100x faster than exact Gillespie for large systems.
10. **`tanimoto_coefficient`** — fingerprint similarity, foundation of all cheminformatics virtual screening.

### Tier 3 — Compositions of existing primitives (low new-code cost)

11. `upgma` — average-linkage hierarchical clustering, already almost present.
12. `sir_rhs`, `seir_rhs`, `lotka_volterra_rhs` — ODE formulas for named models. One-liners composing from mass_action_rhs.
13. `wright_fisher_generation` — repeated binomial sampling.
14. `spike_detect` — threshold crossing scan, uses our series scan infrastructure.
15. `phase_locking_value` — complex exponential average, composes from our FFT/signal processing.

---

## Structural Observation

The biology gap is not primarily about missing primitives — it is about **missing domain-specific kernels** that sit between the math and the application. Tambear has RK4, Poisson processes, HMM, FFT, and distance matrices. Biology needs:

1. **Scoring matrices and gap penalties** (alignment) — parametric tables + DP machinery
2. **Stoichiometric update rules** (kinetics) — structured sparse linear form: reactant-product pairs
3. **Line integral geometry** (medical imaging) — accumulate along parameterized lines through a grid
4. **Pair potentials + PBC** (MD) — the periodic boundary + pair summation pattern
5. **Gating variable equations** (neuroscience) — voltage-dependent rate functions

These five patterns, once in the catalog, would unlock the majority of biology computational math.

---

*Sources consulted*:
- [Gillespie algorithm — Wikipedia](https://en.wikipedia.org/wiki/Gillespie_algorithm)
- [Stochastic simulation in systems biology — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262058/)
- [Needleman-Wunsch algorithm — Wikipedia](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
- [Developments in Algorithms for Sequence Alignment — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9024764/)
- [Wright-Fisher Model — ScienceDirect](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/wright-fisher-model)
- [Mathematical structure of the Wright-Fisher model — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4269093/)
- [Hodgkin-Huxley Model — Wikipedia](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)
- [Compartmental models in epidemiology — Wikipedia](https://en.wikipedia.org/wiki/Compartmental_models_(epidemiology))
- [Radon transform — Wikipedia](https://en.wikipedia.org/wiki/Radon_transform)
- [Neighbor joining — Wikipedia](https://en.wikipedia.org/wiki/Neighbor_joining)
- [Thermostat Algorithms for MD — IISc lecture notes](https://physics.iisc.ac.in/~maiti/course_website/Huenberger_thermostat.pdf)
