//! # tambear
//!
//! Sort-free GPU DataFrame engine.
//!
//! Tam doesn't sort. Tam knows where everything is.
//!
//! The core insight: groupby, deduplication, join, and top-k selection
//! all traditionally use sort as an implementation convenience. On GPU,
//! hash scatter is 17x faster — O(n) with one pass versus O(n log n)
//! with random-access memory writes.
//!
//! ## Four architectural invariants
//!
//! Each eliminates an operation by carrying information forward instead of
//! recovering it later.
//!
//! **Sort-free**: GroupBy/Dedup/Join/TopK → hash ops, never sort.
//! Sort emitted only for `sort_values`, `rank`. 17x on GPU.
//! "Tam doesn't sort. Tam knows."
//!
//! **Mask-not-filter**: Filter sets bits in `Frame::row_mask` (1 bit/row,
//! packed u64). Downstream ops are mask-aware. No compaction, no new array.
//! "Tam doesn't filter. Tam knows which rows matter."
//!
//! **Dictionary strings**: String columns are int codes at ingestion.
//! Dictionary lives in the `.tb` header. GroupBy on strings = GroupBy on ints.
//! Decode only at output.
//! "Tam doesn't do strings. Tam knows the dictionary."
//!
//! **Types once**: `tb.pipeline(dtype=tb.float64)` — single declared dtype.
//! JIT kernels generated for the declared type. Zero runtime dispatch.
//! "Tam doesn't check types. Tam knows the schema."
//!
//! **NaN-free**: Data is cleaned at the boundary (ingestion). Inside tambear,
//! NaN does not exist. Every function trusts every other function.
//! Policy is configurable via `.lock(nan="trim"|"kill"|"fill:0.0")`.
//! "Tam doesn't check for NaN. Tam knows the data is clean."
//!
//! ## Quick Start
//!
//! ```no_run
//! use tambear::scatter_engine::ScatterEngine;
//!
//! let engine = ScatterEngine::new(tam_gpu::detect());
//! // Sum by group — runs on any backend (CUDA, CPU)
//! ```

pub mod accumulate;
pub mod clustering;
pub mod codegen;
pub mod compute_engine;
pub mod data_quality;
pub use data_quality::{
    // Size / coverage
    tick_count, nyquist_bins, coverage_ratio, coverage_ratio_slice,
    effective_sample_size, size_effective_n_bin,
    // Diversity
    unique_prices, symbolic_diversity, unique_ordinal_3,
    // Sampling regularity (timestamp-based)
    sampling_regularity_cv, longest_gap_ratio,
    // Variability / dispersion
    price_cv, split_variance_ratio,
    // Structure
    trend_r2, has_trend, acf_decay_exponent,
    lag1_autocorrelation, lag_k_autocorrelation, acf_lag10, acf_abs_lag1,
    // Stationarity / jump / volatility
    is_stationary_adf_05, jump_ratio_proxy, has_vol_clustering,
    // Summary struct
    DataQualitySummary,
    // Composed family-level validity predicates
    fft_is_valid, garch_is_valid, rank_based_is_valid,
    permutation_entropy_3_is_valid, sample_entropy_is_valid,
    wavelet_is_valid, time_series_is_valid, fractal_is_valid,
    entropy_complexity_is_valid, chaos_is_valid, manifold_is_valid,
    distance_is_valid, causality_is_valid, regime_detection_is_valid,
    continuous_time_is_valid, decomposition_is_valid,
    statistical_tests_is_valid, correlation_is_valid,
};
pub mod data_quality_catalog;
pub mod descriptive;
pub mod sketches;
pub mod spectral_clustering;
pub mod complexity;
pub mod hypothesis;
pub mod information_theory;
pub mod nonparametric;
pub mod numerical;
pub mod optimization;
pub mod robust;
pub mod linear_algebra;
pub mod intermediates;
pub mod dictionary;
#[cfg(feature = "legacy-cudarc")]
pub mod filter_jit;
pub mod format;
pub mod graph;
pub mod nan_guard;
pub mod using;
pub mod gather_op;
pub mod frame;
pub mod group_index;
#[cfg(feature = "legacy-cudarc")]
pub mod hash_scatter;
pub mod reduce_op;
pub mod rng;
#[cfg(feature = "legacy-cudarc")]
pub mod scatter_jit;
pub mod signal_processing;
pub mod spatial;
pub mod special_functions;
pub mod stats;
pub mod tb_io;

pub use dictionary::Dictionary;
pub use format::{
    TileColumnStats, TbColumnDescriptor, TbFileHeader,
    global_min, global_max, global_sum, global_count, global_mean,
    tile_skip_mask_gt, tile_skip_mask_lt,
    TB_MAGIC, TB_VERSION, TB_TILE_SIZE_DEFAULT, FILE_HEADER_SIZE, TB_MAX_COLUMNS,
};
pub use frame::{Column, ColumnEncoding, DType, Frame};
pub use group_index::GroupIndex;
#[cfg(feature = "legacy-cudarc")]
pub use hash_scatter::{HashScatterEngine, GroupByResult};
#[cfg(feature = "legacy-cudarc")]
pub use filter_jit::{
    FilterJit, MaskOp,
    mask_popcount, mask_and, mask_or, mask_not, mask_xor,
    PRED_NOT_NAN, PRED_FINITE, PRED_POSITIVE,
};
pub use gather_op::GatherOp;
pub use accumulate::{AccumulateEngine, Grouping, Expr, Op, AccResult};
pub use clustering::{
    ClusteringEngine, ClusterResult, uf_new, uf_find, uf_union,
    // Cluster validation primitives
    ClusterCentroids, ClusterValidation,
    cluster_centroids, cluster_validation,
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
    // Cluster tendency / model selection
    hopkins_statistic,
    gap_statistic, GapStatisticResult,
    bic_score, aic_score,
    kmeans_f64,
};
pub use intermediates::{DistanceMatrix, Metric, SufficientStatistics, IntermediateTag, TamSession, DataId};
pub use reduce_op::ReduceOp;
#[cfg(feature = "legacy-cudarc")]
pub use scatter_jit::{ScatterJit, PHI_SUM, PHI_SUM_SQ, PHI_CENTERED_SUM, PHI_CENTERED_SUM_SQ, PHI_COUNT};
#[cfg(not(feature = "legacy-cudarc"))]
pub use scatter_engine::{PHI_SUM, PHI_SUM_SQ, PHI_CENTERED_SUM_SQ, PHI_COUNT};
pub use information_theory::{
    InformationEngine,
    shannon_entropy, shannon_entropy_from_counts, renyi_entropy, tsallis_entropy,
    kl_divergence, js_divergence, cross_entropy,
    mutual_information, normalized_mutual_information, variation_of_information,
    conditional_entropy, probabilities,
    mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    entropy_histogram,
    mutual_info_miller_madow, fisher_information_histogram,
    histogram_entropy,
    // Bias-corrected estimators
    grassberger_entropy,
    // Atoms — public primitives for composing new information-theoretic measures
    p_log_p, p_log_p_over_q,
    contingency_from_labels, expected_mutual_info,
    // f-divergence family
    hellinger_distance_sq, hellinger_distance,
    total_variation_distance,
    chi_squared_divergence,
    renyi_divergence,
    bhattacharyya_coefficient, bhattacharyya_distance,
    f_divergence,
    // Joint entropy and PMI
    joint_entropy, pointwise_mutual_information,
    // Sample-based divergences
    wasserstein_1d, mmd_rbf, energy_distance,
    // Histogram building blocks
    histogram, joint_histogram,
    // Transfer entropy
    transfer_entropy,
    // Text/vector similarity
    TfidfResult, tfidf,
    cosine_similarity, cosine_similarity_matrix,
};
pub use descriptive::{
    DescriptiveEngine, DescriptiveResult, MomentStats, GroupedMomentStats,
    moments_ungrouped, moments_session, quantile, median, quartiles, iqr,
    geometric_mean, harmonic_mean, trimmed_mean, winsorized_mean,
    coefficient_of_variation,
    sorted_nan_free, mad, gini, bowley_skewness, pearson_first_skewness,
    QuantileMethod,
    PHI_CENTERED_CU, PHI_CENTERED_QU,
    // Forecast error metrics
    mae, rmse, mape, mase,
    // Scipy gap closure (2026-04-10)
    mode, sem, percentileofscore, lmoment, lmoment_ratios,
};
pub use tb_io::{TbFile, TbColumnWrite, write_tb};
pub use special_functions::{
    erf, erfc, log_gamma, gamma, log_beta, digamma, trigamma,
    regularized_incomplete_beta, regularized_gamma_p, regularized_gamma_q,
    normal_cdf, normal_sf, normal_quantile, t_cdf, f_cdf, chi2_cdf, chi2_sf,
    normal_two_tail_p, t_two_tail_p, f_right_tail_p, chi2_right_tail_p,
    weibull_cdf, weibull_pdf, weibull_quantile,
    pareto_cdf, pareto_pdf, pareto_quantile,
    exponential_cdf, exponential_pdf, exponential_quantile,
    lognormal_cdf, lognormal_pdf, lognormal_quantile,
    beta_pdf, beta_cdf,
    gamma_pdf, gamma_cdf,
    poisson_pmf, poisson_cdf,
    binomial_pmf, binomial_cdf,
    neg_binomial_pmf, neg_binomial_cdf,
    cauchy_cdf, cauchy_pdf, cauchy_quantile,
    // Orthogonal polynomials
    chebyshev_t, chebyshev_u, chebyshev_series,
    legendre_p, legendre_p_deriv, gauss_legendre_nodes_weights,
    hermite_he, laguerre_l,
    // Bessel functions
    bessel_j0, bessel_j1, bessel_jn, bessel_i0, bessel_i1,
    // Random matrix theory
    marchenko_pastur_pdf, marchenko_pastur_bounds, marchenko_pastur_classify,
    chebyshev_outlier,
    // Logistic family
    logistic, logit,
    // Normal PDF
    normal_pdf, normal_pdf_standard,
    // Stirling approximation
    stirling_approx, stirling_approx_corrected,
};
pub use hypothesis::{
    TestResult, AnovaResult, ChiSquareResult, HypothesisEngine,
    one_sample_t, two_sample_t, welch_t, paired_t,
    one_way_anova,
    chi2_goodness_of_fit, chi2_independence,
    one_proportion_z, two_proportion_z,
    cohens_d, glass_delta, hedges_g, point_biserial_r,
    odds_ratio, log_odds_ratio, log_odds_ratio_se,
    bonferroni, holm, benjamini_hochberg,
    BreuschPaganResult, breusch_pagan,
    InfluenceResult, cooks_distance,
    TukeyComparison, tukey_hsd,
    LogisticRegressionResult, logistic_regression,
    MediationResult, mediation,
    ModerationResult, moderation,
    BayesFactorResult, bayes_factor_t_one_sample, bayes_factor_correlation,
    // Primitives (flat catalog)
    interpret_bf,
    ols_simple, ols_two_predictor,
};
pub use nonparametric::{
    rank, spearman, pearson_on_ranks, kendall_tau,
    NonparametricResult,
    mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis,
    ks_test_normal, ks_test_normal_standardized, ks_test_two_sample, ks_p_value,
    shapiro_wilk, dagostino_pearson, jarque_bera, anderson_darling, friedman_test,
    DunnComparison, dunn_test,
    partial_correlation, PartialCorrelationResult, partial_correlation_full,
    bootstrap_percentile, BootstrapResult,
    permutation_test_mean_diff,
    kde, kde_fft, silverman_bandwidth, KernelType, kernel_eval,
    runs_test, runs_test_numeric, sign_test,
    level_spacing_r_stat,
    GutenbergRichterResult, gutenberg_richter_fit,
    OmoriResult, omori_fit,
    BathResult, bath_law,
    SdeResult, sde_estimate,
    inversion_count, inversion_count_mergesort,
    pearson_r,
    // Tie counting primitive
    TieInfo, tie_count,
    // Primitives (flat catalog)
    shapiro_wilk_coefficients,
    // Missing flavors (added 2026-04-10)
    hoeffdings_d, blomqvist_beta,
    // Correlation primitives (flat catalog)
    phi_coefficient, point_biserial, biserial_correlation, rank_biserial,
    tetrachoric, cramers_v, eta_squared, distance_correlation, concordance_correlation,
    // Histogram / ECDF
    BinRule, Histogram, histogram_auto,
    scott_bandwidth, sturges_bins, scott_bins, freedman_diaconis_bins, doane_bins,
    Ecdf, ecdf, ecdf_confidence_band,
    // Sequence similarity
    dtw, dtw_banded, levenshtein, quantile_symbolize, edit_distance_on_series,
};
pub use complexity::{
    sample_entropy, approx_entropy, permutation_entropy, normalized_permutation_entropy,
    hurst_rs, dfa, higuchi_fd, lempel_ziv_complexity,
    correlation_dimension, largest_lyapunov,
    LyapunovSpectrum, lyapunov_spectrum,
    // Nonlinear/structural complexity
    RqaResult, rqa, MfdfaResult, mfdfa, CcmResult, ccm,
    PhaseTransitionResult, phase_transition,
    harmonic_r_stat, hankel_r_stat,
    // Atoms — promoted from private: composable sub-operations
    count_matches, phi_func, pattern_to_index, factorial,
    linear_fit_segment, estimate_mean_period,
};
pub use numerical::{
    RootResult, bisection, newton, secant, brent, fixed_point,
    derivative_central, derivative2_central, derivative_richardson,
    simpson, gauss_legendre_5, adaptive_simpson, trapezoid,
    OdeSolution, euler, rk4, rk45, rk4_system,
    // NaN-propagating min/max (use for folds over raw/unfiltered data)
    nan_min, nan_max,
    // Geometry
    stereographic_project, stereographic_project_inverse,
    // Dynamical systems
    BrusselatorAnalysis, brusselator_rhs, brusselator_jacobian,
    brusselator_bifurcation, brusselator_simulate,
};
pub use linear_algebra::{
    Mat, mat_mul, mat_add, mat_sub, mat_scale, mat_vec, dot, vec_norm, outer,
    LuResult, lu, lu_solve, det, inv,
    cholesky, cholesky_solve,
    QrResult, qr, qr_solve, lstsq,
    SvdResult, svd, pinv,
    sym_eigen, power_iteration,
    cond, solve, solve_spd,
    solve_tridiagonal, solve_tridiagonal_scan,
    tridiagonal_scan_element, tridiagonal_scan_compose,
    // Global primitives (flat catalog)
    SimpleRegressionResult, simple_linear_regression, ols_slope,
    ols_normal_equations, ols_residuals,
    forward_solve, back_solve_transpose,
    effective_rank_from_sv,
    // Orthogonalization
    gram_schmidt, gram_schmidt_modified,
    // Matrix functions
    matrix_exp, matrix_log, matrix_sqrt,
    // Iterative solvers
    IterativeSolverResult, conjugate_gradient, gmres,
    // Utilities
    log_det,
};
pub use graph::{
    Edge, Graph, MstResult,
    bfs, dfs, topological_sort, connected_components,
    dijkstra, bellman_ford, floyd_warshall, reconstruct_path,
    kruskal, prim,
    degree_centrality, closeness_centrality, pagerank,
    label_propagation, modularity,
    max_flow, diameter, density, clustering_coefficient,
    // Dense matrix graph primitives
    pairwise_dists, knn_adjacency, graph_laplacian,
};
pub use optimization::{
    OptResult,
    backtracking_line_search, golden_section,
    gradient_descent, adam, adagrad, rmsprop, lbfgs,
    nelder_mead, coordinate_descent, projected_gradient,
};
pub use robust::{
    huber_weight, bisquare_weight, hampel_weight,
    MEstimateResult, huber_m_estimate, bisquare_m_estimate, hampel_m_estimate,
    qn_scale, sn_scale, tau_scale,
    LtsResult, lts_simple,
    McdResult2D, mcd_2d,
    medcouple,
};
pub use rng::{
    TamRng, SplitMix64, Xoshiro256, Lcg64,
    normal_pair, sample_normal, sample_exponential, sample_gamma, sample_beta,
    sample_chi2, sample_t, sample_f, sample_cauchy, sample_lognormal,
    sample_bernoulli, sample_poisson, sample_binomial, sample_geometric,
    shuffle, sample_without_replacement, sample_weighted,
    fill_uniform, fill_normal, randn, randu,
};
pub mod bigint;
pub mod bigfloat;
pub mod copa;
pub mod interpolation;
pub mod multivariate;
pub use multivariate::{
    // Primitives (flat catalog)
    covariance_matrix, col_means, sscp_matrices,
    // Multivariate tests
    HotellingResult, hotelling_one_sample, hotelling_two_sample,
    ManovaResult, manova,
    LdaResult, lda,
    CcaResult, cca,
    MardiaNormalityResult, mardia_normality,
    vif, mahalanobis_distances,
    RegularizedResult, ridge, lasso, elastic_net,
    // Weighted least squares (canonical: beta/wrss/r2 field names)
    WlsResult, wls,
};
pub mod causal;
pub mod spectral;
pub mod mixed_effects;
pub use mixed_effects::{
    LmeResult, lme_random_intercept,
    icc_oneway, IccResult, icc_twoway_random, icc_twoway_mixed,
    design_effect,
    // Primitives (flat catalog)
    twoway_anova_ms,
};
pub mod panel;
pub mod survival;
pub mod mixture;
pub mod bayesian;
pub mod kalman;
pub use kalman::{
    KalmanFilterResult, kalman_filter_scalar, rts_smoother_scalar,
    KalmanFilterMatrixResult, kalman_filter_matrix,
    HmmParams, HmmForwardBackwardResult,
    hmm_forward_backward, hmm_viterbi, hmm_baum_welch,
};
pub mod time_series;
pub use time_series::{
    // Toeplitz / Levinson primitives
    levinson_durbin,
    // Phase-space primitives
    delay_embed,
    StlResult, stl_decompose,
    ArmaResult, arma_fit, ArimaResult, arima_fit, arima_forecast, auto_arima,
    undifference,
    // Stationarity / unit root
    PhillipsPerronResult, phillips_perron_test,
    // HAC / long-run variance
    newey_west_lrv,
    // White noise / dependence
    box_pierce,
    // White noise / serial dependence
    breusch_godfrey, turning_point_test, rank_von_neumann_ratio,
    // Spectral post-processing
    spectral_flatness, spectral_rolloff, spectral_centroid, spectral_bandwidth,
    spectral_skewness, spectral_kurtosis, spectral_crest, spectral_slope,
    spectral_fwhm, spectral_q_factor,
    spectral_flux, spectral_decrease, spectral_contrast,
    dominant_frequency, dominant_frequency_power, peak_to_average_power_ratio,
    spectral_peak_count,
    // Primitives (flat catalog)
    arma_css_residuals, mackinnon_adf_critical_values,
    log_returns,
};
pub mod volatility;
pub mod factor_analysis;
pub mod irt;
pub mod dim_reduction;
pub mod tda;
pub mod kmeans;
pub mod knn;
pub mod manifold;
pub mod neural;
pub mod pipeline;
pub mod tbs_executor;
pub mod tbs_jit;
pub mod tbs_lint;
pub mod tbs_parser;
pub mod train;
pub mod proof;
pub mod extremal_orbit;
pub mod layer_bijection;
pub mod series_accel;
pub mod parallel;
pub use parallel::{parallel_range_reduce, parallel_slice_reduce};
pub mod spectral_gap;
pub use spectral_gap::{
    ArnoldiResult, arnoldi_eigenvalues,
    SparseDeterministicMap, CycleStructure,
};
/// Type alias: a functional graph (every node has exactly one out-edge)
/// is the same structure as a deterministic state-transition map. Use this
/// name when the application is graph-theoretic (cycle detection on a digraph)
/// rather than dynamical (sparse transition operator on a state space).
pub type FunctionalGraph = spectral_gap::SparseDeterministicMap;
pub mod fold_irreversibility;
pub mod spec_compiler;
pub mod superposition;
pub mod scatter_engine;
pub mod tam;
pub mod equipartition;
pub mod multi_adic;
pub mod collatz_parallel;
pub mod stochastic;
pub mod hmm;
pub mod state_space;
pub use state_space::{
    LinearGaussianSsm, KalmanFilterResult as StateSpaceKalmanResult,
    RtsSmootherResult, SsmEmResult,
    kalman_filter as kalman_filter_lgssm_matrix, rts_smoother as rts_smoother_matrix,
    ssm_em,
    ParticleFilterResult, particle_filter, particle_filter_lgssm,
    particle_filter_log_likelihood, systematic_resample,
};
pub use stochastic::{
    // Brownian motion
    brownian_motion, brownian_bridge, quadratic_variation,
    // GBM / Black-Scholes
    geometric_brownian_motion, black_scholes, gbm_expected, gbm_variance,
    // Ornstein-Uhlenbeck
    ornstein_uhlenbeck, ou_stationary_variance, ou_autocorrelation,
    // Poisson processes
    poisson_process, poisson_count, poisson_expected_count, nonhomogeneous_poisson,
    // Discrete-time Markov chains
    markov_n_step, stationary_distribution, mean_first_passage_time,
    is_ergodic, mixing_time,
    // CTMC
    ctmc_transition_matrix, ctmc_stationary, ctmc_holding_time,
    // Birth-death / queues
    birth_death_stationary, mm1_queue, erlang_c,
    // Random walks
    simple_random_walk, first_passage_time_cdf, return_probability_1d, rw_expected_maximum,
    // Itô calculus
    ito_integral, stratonovich_integral, ito_lemma_verification,
};
pub mod number_theory;
pub use number_theory::{
    sieve, segmented_sieve, is_prime, next_prime, prime_count,
    mul_mod, mod_pow, gcd, lcm, extended_gcd, mod_inverse,
    crt, legendre, jacobi, sqrt_mod,
    euler_totient, mobius, factorize, num_divisors, sum_divisors, divisors,
    sieve_totients, sieve_spf,
    primitive_root, discrete_log,
    continued_fraction, convergents, best_rational, cf_period,
    pollard_rho, factorize_complete,
    isqrt, perfect_square, sum_of_two_squares, pell_fundamental,
    partition_count, euler_product_approx, basel_sum_exact,
    rsa_keygen, rsa_encrypt, rsa_decrypt, dh_public_key, dh_shared_secret,
    LinearRecurrenceStep, compose_steps,
};
pub mod physics;
pub use physics::{
    // Constants
    K_BOLTZMANN, H_BAR, SPEED_OF_LIGHT, G_GRAV, ELEM_CHARGE, MASS_ELECTRON,
    BOHR_RADIUS, AVOGADRO, GAS_CONSTANT, EPSILON_0, HYDROGEN_GROUND_EV,
    // Classical mechanics
    Particle, NBodyResult, nbody_gravity,
    sho_exact, sho_energy, dho_underdamped,
    KeplerOrbit, kepler_orbit, vis_viva,
    DoublePendulumState, double_pendulum_rk4, double_pendulum_energy,
    euler_rotation, rotational_kinetic_energy,
    // Thermodynamics
    ideal_gas_pressure, ideal_gas_temperature, ideal_gas_internal_energy, ideal_gas_entropy_change,
    vdw_pressure, vdw_critical,
    carnot_efficiency, otto_efficiency, entropy_change_isothermal,
    heat_flux_fourier, newton_cooling, stefan_boltzmann,
    // Statistical mechanics
    partition_function, mean_energy, heat_capacity_canonical,
    helmholtz_free_energy, boltzmann_probabilities, gibbs_entropy,
    qho_energy, bose_einstein_occupation, planck_spectral_energy, wien_displacement,
    ising1d_exact, ising2d_metropolis,
    arrhenius, equilibrium_constant,
    // Quantum mechanics
    hydrogen_energy_ev, hydrogen_wavelength,
    particle_in_box_energy, particle_in_box_wf,
    tunneling_transmission,
    Amplitude, normalize_state, time_evolve_state,
    expectation_value, uncertainty, heisenberg_uncertainty_product,
    density_matrix_trace, density_matrix_purity, von_neumann_entropy_diagonal,
    schrodinger1d, sym_tridiag_eigvals,
    // Fluid dynamics
    reynolds_number, mach_number, prandtl_number, nusselt_dittus_boelter,
    bernoulli_velocity, poiseuille_flow_rate, poiseuille_velocity_profile,
    FlowState, euler1d_lax_friedrichs, cfl_timestep,
    poisson_sor, vorticity_step,
    // Special relativity
    lorentz_factor, relativistic_kinetic_energy, relativistic_momentum,
    mass_energy, time_dilation, length_contraction,
    relativistic_velocity_addition, relativistic_doppler,
};
pub use equipartition::{
    free_energy, euler_factor, fugacity, fold_target,
    solve_fold, solve_pairwise, diagnose_fold,
    FoldDiagnostics, FoldPoint, NucleationHierarchy, Phase,
    all_pairwise_folds, k_wise_folds, batch_pairwise_folds, BatchFoldResult,
    nucleation_hierarchy, nucleation_hierarchy_full,
    verify_fold_surface, classify_phase, phase_sweep, fold_sensitivity,
};
pub use spatial::{
    SpatialPoint, euclidean_2d, haversine,
    VariogramBin, VariogramModel, empirical_variogram,
    spherical_variogram, exponential_variogram, gaussian_variogram,
    KrigingResult, ordinary_kriging,
    SpatialWeights, morans_i, gearys_c,
    ripleys_k, ripleys_l, nn_distances, clark_evans_r,
    convex_hull_2d, polygon_area, polygon_perimeter,
};
pub use interpolation::{
    lagrange, newton_divided_diff, newton_eval, neville, lerp, nearest,
    SplineSegment, CubicSpline, natural_cubic_spline, clamped_cubic_spline,
    monotone_hermite, akima, pchip,
    chebyshev_nodes, chebyshev_coefficients, chebyshev_eval, chebyshev_approximate,
    PolyFit, polyfit,
    RbfKernel, RbfInterpolant, rbf_interpolate,
    BarycentricRational, barycentric_rational,
    bspline_basis, bspline_eval, uniform_knots,
    GpResult, gp_regression,
    PadeApproximant, pade,
};
pub use signal_processing::{
    Complex, next_pow2, fft, ifft, rfft, irfft, fft2d,
    window_hann, window_hamming, window_blackman, window_bartlett, window_kaiser, window_flat_top,
    periodogram, welch, stft, spectrogram,
    convolve, cross_correlate, autocorrelation,
    dct2, dct3,
    fir_lowpass, fir_highpass, fir_bandpass, fir_filter,
    Biquad, biquad_cascade, butterworth_lowpass_cascade,
    affine_prefix_scan, moving_average, ema, ema_period, savgol_filter,
    hilbert, envelope, instantaneous_frequency,
    real_cepstrum,
    haar_dwt, haar_idwt, haar_wavedec, haar_waverec,
    db4_dwt, db4_idwt,
    goertzel, goertzel_mag,
    zero_crossing_rate, median_filter,
    path_signature_2d, log_signature_2d,
    regularize_interp, regularize_bin_mean, regularize_subsample,
    WvdResult, wvd_features,
    IcaResult, fast_ica,
    EmdResult, emd,
};
pub use pipeline::{TamFrame, TamPipeline, ClusterSpec, ClusterView, ColumnDescribe, DescribeResult, DiscoveryResult};
pub use manifold::{Manifold, ManifoldMixture, ManifoldDistanceOp};
pub use neural::{
    relu, relu_vec, relu_backward,
    leaky_relu, leaky_relu_vec, leaky_relu_backward,
    elu, elu_vec, elu_backward,
    selu, selu_vec, SELU_ALPHA, SELU_LAMBDA,
    gelu, gelu_vec, gelu_backward,
    swish, swish_vec, swish_backward,
    mish, mish_vec,
    sigmoid, sigmoid_vec, sigmoid_backward,
    tanh_vec, tanh_backward,
    softmax, log_softmax, softmax_backward,
    softplus, softplus_vec, softsign, softsign_vec,
    hard_sigmoid, hard_sigmoid_vec, hard_swish, hard_swish_vec,
    conv1d, conv1d_multi, conv2d, conv2d_transpose,
    max_pool1d, avg_pool1d, max_pool2d, avg_pool2d,
    global_avg_pool2d, adaptive_avg_pool1d,
    BatchNormResult, batch_norm, layer_norm, rms_norm, group_norm, instance_norm,
    dropout, dropout_backward,
    linear, linear_backward, bilinear,
    embedding, positional_encoding, rope,
    AttentionResult, scaled_dot_product_attention, multi_head_attention,
    mse_loss, mse_loss_backward,
    bce_loss, bce_loss_backward,
    cross_entropy_loss, cross_entropy_loss_backward,
    huber_loss, huber_loss_backward,
    cosine_similarity_loss, hinge_loss, focal_loss,
    flatten_shape, residual_add,
    clip_grad_norm, clip_grad_value,
    label_smooth, temperature_scale,
    top_k_logits, top_p_logits,
};
pub use bigint::{U256, BigInt};
pub use bigfloat::{BigFloat, BigComplex, zeta_complex, hardy_z, find_zeta_zero, euler_factor_complex, euler_product_complex};
pub use superposition::{Superposition, SuperpositionView};
pub use tam::{tam, TamValue, TamResult, ConvergeResult, Diagnostic, EmergentDepth, tam_f64};

// Survival analysis
pub use survival::{
    KmStep, kaplan_meier, km_median,
    LogRankResult, log_rank_test,
    CoxResult, cox_ph,
    GrambschTherneauResult, grambsch_therneau_test,
};

// Panel / econometrics
pub use panel::{
    FeResult, panel_fe, panel_re,
    FdResult, panel_fd,
    HausmanResult, hausman_test, hausman_test_full,
    panel_twfe, breusch_pagan_re, re_theta,
    TwoSlsResult, two_sls,
    DidResult, did,
};

// Bayesian inference
pub use bayesian::{
    McmcChain, metropolis_hastings,
    BayesLinearResult, bayesian_linear_regression,
    effective_sample_size as mcmc_effective_sample_size, r_hat,
};

// Item Response Theory
pub use irt::{
    rasch_prob, prob_2pl, prob_3pl,
    ItemParams, IrtFitConfig, fit_2pl,
    ability_mle, ability_eap,
    item_information, test_information, sem,
    mantel_haenszel_dif,
};

// Series acceleration (Aitken, Shanks, Wynn, Richardson, Euler, Abel)
pub use series_accel::{
    partial_sums, cumsum, cesaro_sum,
    aitken_delta2, wynn_epsilon, StreamingWynn,
    richardson_extrapolate, euler_transform, abel_sum,
    richardson_partial_sums, euler_maclaurin_zeta,
    ConvergenceType, detect_convergence, accelerate,
};

// Naive Bayes classifier
pub use train::naive_bayes::{
    GaussianNB, gaussian_nb_fit, gaussian_nb_predict, gaussian_nb_predict_proba,
};

// Dimensionality reduction
pub use dim_reduction::{
    PcaResult, pca,
    MdsResult, classical_mds,
    TsneResult, tsne,
    NmfResult, nmf,
};

// Factor analysis and reliability
pub use factor_analysis::{
    FaResult, principal_axis_factoring, varimax,
    cronbachs_alpha,
    OmegaResult, mcdonalds_omega,
    scree_elbow, kaiser_criterion,
    KmoBartlettResult, kmo_bartlett,
};

// Spectral clustering
pub use spectral_clustering::{
    SpectralClusterParams, SpectralClusterResult,
    AffinityKind, LaplacianKind,
    spectral_cluster, spectral_embedding,
    build_affinity, build_laplacian,
};

// Topological Data Analysis
pub use tda::{
    PersistencePair, PersistenceDiagram,
    rips_h0, rips_h1,
    bottleneck_distance, wasserstein_distance,
    persistence_statistics, persistence_entropy, betti_curve,
};

// Volatility estimation and market microstructure
pub use volatility::{
    GarchResult, garch11_fit, garch11_forecast,
    EgarchResult, egarch11_fit,
    GjrGarchResult, gjr_garch11_fit,
    TgarchResult, tgarch11_fit,
    ewma_variance, realized_variance, realized_volatility,
    bipower_variation, jump_test_bns,
    roll_spread, kyle_lambda, amihud_illiquidity,
    annualize_vol,
    parkinson_variance, garman_klass_variance,
    rogers_satchell_variance, yang_zhang_variance,
    hill_estimator, hill_tail_alpha,
    tripower_quarticity,
    ArchLmResult, arch_lm_test,
    VpinResult, vpin_bvc,
    nvg_degree, hvg_degree, nvg_mean_degree, hvg_mean_degree,
};

// Causal inference
pub use causal::{
    propensity_scores,
    MatchResult, psm_match,
    IpwResult, ipw,
    DidResult as CausalDidResult, did as causal_did,
    RddResult, rdd_sharp,
    e_value,
    doubly_robust_ate,
};
