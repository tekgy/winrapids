# Optimization — Complete Variant Catalog

## What Exists (tambear::optimization)

### Line Search
- `backtracking_line_search(f, x, grad, d, alpha_init, rho, c)` — Armijo

### 1D Optimization
- `golden_section(f, a, b, tol)` — golden ratio search

### Gradient-Based
- `gradient_descent(f, grad, x0, lr, tol, max_iter)` — fixed step
- `adam(f, grad, x0, lr, beta1, beta2, eps, tol, max_iter)` — Adam
- `adagrad(f, grad, x0, lr, eps, tol, max_iter)` — AdaGrad
- `rmsprop(f, grad, x0, lr, alpha, eps, tol, max_iter)` — RMSProp
- `lbfgs(f, grad, x0, m, tol, max_iter)` — L-BFGS

### Derivative-Free
- `nelder_mead(f, x0, tol, max_iter)` — Nelder-Mead simplex
- `nelder_mead_with_params(f, x0, alpha, gamma, rho, sigma, tol, max_iter)`
- `coordinate_descent(f, x0, step, tol, max_iter)`

### Constrained
- `projected_gradient(f, grad, proj, x0, lr, tol, max_iter)` — projected gradient

---

## What's MISSING — Complete Catalog

### A. Line Search Improvements

1. **Wolfe conditions line search** — strong Wolfe conditions
   - Sufficient decrease + curvature condition
   - Parameters: `f`, `grad`, `x`, `d`, `c1`, `c2`
   - More robust convergence than pure Armijo

2. **Zoom phase** — Moré-Thuente 1994
   - Bracket + zoom for finding step satisfying strong Wolfe
   - Used internally by L-BFGS implementations

### B. Second-Order Methods

3. **Newton's method** (for optimization) — uses Hessian
   - x_{k+1} = x_k - H⁻¹∇f(x_k)
   - Parameters: `f`, `grad`, `hess`, `x0`, `tol`, `max_iter`
   - Quadratic convergence near optimum
   - Primitives: solve(H, -grad) → descent direction

4. **Modified Newton** — with Hessian regularization
   - Add multiple of identity to ensure positive definiteness
   - Parameters: same as Newton + `beta` (regularization)

5. **BFGS** (full) — Broyden-Fletcher-Goldfarb-Shanno
   - Dense Hessian approximation, O(n²) storage
   - Parameters: `f`, `grad`, `x0`, `tol`, `max_iter`
   - Already have L-BFGS (limited memory version)
   - Full BFGS better for n < 1000

6. **DFP** — Davidon-Fletcher-Powell
   - First quasi-Newton method
   - Parameters: same as BFGS
   - Generally less robust than BFGS

7. **SR1** — Symmetric Rank-1 update
   - Parameters: same as BFGS
   - Can approximate indefinite Hessian (BFGS/DFP cannot)
   - Useful for saddle-point problems

8. **Conjugate gradient** (nonlinear) — for optimization
   - Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel, Dai-Yuan
   - Parameters: `f`, `grad`, `x0`, `variant`, `tol`, `max_iter`
   - O(n) storage, better than gradient descent

### C. Trust Region Methods

9. **Trust region Newton-CG** — Steihaug-Toint
   - Solve trust region subproblem via conjugate gradient
   - Parameters: `f`, `grad`, `hess_vec_product`, `x0`, `delta_max`, `tol`, `max_iter`
   - More robust than line search Newton

10. **Trust region dogleg** — Powell's dogleg
    - Combines Cauchy point and Newton step
    - Parameters: `f`, `grad`, `hess`, `x0`, `delta_max`, `tol`, `max_iter`

11. **Trust region truncated CG** — Steihaug 1983
    - CG on trust region subproblem
    - Parameters: same as trust region Newton-CG

### D. Global / Derivative-Free Optimization

12. **Simulated annealing** — Kirkpatrick et al. 1983
    - Parameters: `f`, `x0`, `temp_init`, `cooling_rate`, `max_iter`
    - Accept worse solutions with probability exp(-Δf/T)

13. **Differential evolution** — Storn & Price 1997
    - Population-based: mutation + crossover + selection
    - Parameters: `f`, `bounds`, `pop_size`, `F` (mutation), `CR` (crossover), `max_gen`
    - Best general-purpose global optimizer for continuous domains

14. **Particle swarm optimization** (PSO) — Kennedy & Eberhart 1995
    - Swarm of particles with velocity and position
    - Parameters: `f`, `bounds`, `n_particles`, `w` (inertia), `c1`, `c2`, `max_iter`

15. **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy
    - Hansen 2006
    - Adapts covariance matrix of search distribution
    - Parameters: `f`, `x0`, `sigma0`, `pop_size`, `max_gen`
    - Best for: medium-dimensional (n < 100) non-convex problems

16. **Bayesian optimization** — Mockus 1975, Jones et al. 1998
    - Surrogate model (GP) + acquisition function (EI, UCB, PI)
    - Parameters: `f`, `bounds`, `n_init`, `n_iter`, `acquisition`
    - Best for: expensive black-box functions, few evaluations

17. **Powell's method** — Powell 1964
    - Direction set method, no derivatives
    - Parameters: `f`, `x0`, `tol`, `max_iter`

18. **COBYLA** — Powell 1994
    - Constrained Optimization BY Linear Approximations
    - Parameters: `f`, `x0`, `constraints`, `tol`, `max_iter`
    - Derivative-free constrained optimization

19. **BOBYQA** — Powell 2009
    - Bound Optimization BY Quadratic Approximation
    - Parameters: `f`, `x0`, `lower`, `upper`, `tol`, `max_iter`

### E. Constrained Optimization

20. **Augmented Lagrangian** (ALM) — Hestenes 1969, Powell 1969
    - L_A(x,λ,ρ) = f(x) + λᵀg(x) + (ρ/2)||g(x)||²
    - Parameters: `f`, `grad`, `constraints`, `x0`, `rho_init`, `tol`, `max_iter`
    - Equality and inequality constraints

21. **Sequential Quadratic Programming** (SQP)
    - Solve QP subproblem at each step
    - Parameters: `f`, `grad`, `hess`, `constraints`, `x0`, `tol`, `max_iter`
    - Gold standard for smooth nonlinear constrained optimization

22. **Interior point** (barrier method) — for convex optimization
    - Parameters: `f`, `grad`, `hess`, `constraints`, `x0`, `mu_init`, `tol`
    - Logarithmic barrier: min f(x) - μ Σ log(-gᵢ(x))

23. **ADMM** — Alternating Direction Method of Multipliers
    - Boyd et al. 2011
    - For: min f(x) + g(z) s.t. Ax + Bz = c
    - Parameters: `prox_f`, `prox_g`, `A`, `B`, `c`, `rho`, `max_iter`
    - Widely used in: LASSO, basis pursuit, consensus optimization

24. **Proximal gradient** (ISTA/FISTA) — Beck & Teboulle 2009
    - For: min f(x) + g(x) where f is smooth, g has easy proximal
    - FISTA: accelerated with Nesterov momentum
    - Parameters: `grad_f`, `prox_g`, `x0`, `step_size`, `tol`, `max_iter`
    - Use case: LASSO (prox_g = soft threshold), constrained optimization

25. **Frank-Wolfe** (conditional gradient)
    - For: min f(x) over convex set C
    - Each step: minimize linear approximation over C
    - Parameters: `grad`, `lmo` (linear minimization oracle), `x0`, `tol`, `max_iter`

26. **Mirror descent** — Nemirovski & Yudin 1983
    - Generalization of gradient descent to non-Euclidean geometry
    - Parameters: `grad`, `mirror_map`, `x0`, `step_sizes`, `max_iter`

### F. Convex Optimization

27. **Linear programming** — simplex or interior point
    - min cᵀx s.t. Ax ≤ b, x ≥ 0
    - Simplex method: O(2^n) worst case, fast in practice
    - Parameters: `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds`

28. **Quadratic programming**
    - min 0.5 xᵀQx + cᵀx s.t. Ax ≤ b
    - Active set or interior point
    - Parameters: `Q`, `c`, `A`, `b`

29. **Second-order cone programming** (SOCP)
    - min cᵀx s.t. ||Aᵢx + bᵢ|| ≤ cᵢᵀx + dᵢ
    - Parameters: `c`, `cone_constraints`

30. **Semidefinite programming** (SDP)
    - min trace(CX) s.t. X ⪰ 0, trace(AᵢX) = bᵢ
    - Parameters: `C`, `A_constraints`, `b`

### G. Multi-Objective

31. **Pareto frontier** — non-dominated sorting
    - Parameters: `objectives: &[Fn]`, `pop`, `max_gen`
    - NSGA-II: Deb et al. 2002

32. **Weighted sum method** — min Σ wᵢ fᵢ(x) for multiple objectives
    - Parameters: `objectives`, `weights`, `x0`

### H. Stochastic

33. **SGD with momentum** — Polyak 1964
    - v_t = μ v_{t-1} + lr × ∇f
    - Parameters: `f`, `grad`, `x0`, `lr`, `momentum`, `max_iter`

34. **Nesterov accelerated gradient** — Nesterov 1983
    - Look-ahead gradient: evaluate at x + μv, not x
    - Parameters: same as momentum SGD

35. **AdamW** — Loshchilov & Hutter 2019
    - Adam with decoupled weight decay
    - Parameters: `f`, `grad`, `x0`, `lr`, `betas`, `eps`, `weight_decay`, `max_iter`

36. **LAMB** — You et al. 2020
    - Layer-wise Adaptive Moments for large batch training
    - Parameters: similar to Adam + trust ratio

37. **Lookahead** — Zhang et al. 2019
    - Wraps any optimizer: slow weights = average of fast weights
    - Parameters: `inner_optimizer`, `k` (sync period), `alpha` (interpolation)

---

## Priority

**Tier 1** — Most impactful missing:
1. `bfgs(f, grad, x0)` — full quasi-Newton (have L-BFGS; need full)
2. `conjugate_gradient_opt(f, grad, x0, variant)` — nonlinear CG
3. `trust_region(f, grad, hess_vec, x0)` — more robust than line search
4. `differential_evolution(f, bounds)` — best global optimizer
5. `augmented_lagrangian(f, grad, constraints, x0)` — general constrained
6. `proximal_gradient / fista(grad_f, prox_g, x0)` — convex composite

**Tier 2**:
7. `cma_es(f, x0, sigma0)` — derivative-free global
8. `simulated_annealing(f, x0)` — global, simple
9. `sqp(f, grad, hess, constraints, x0)` — constrained
10. `admm(prox_f, prox_g, A, B, c, rho)` — splitting
11. `bayesian_optimization(f, bounds)` — expensive functions
12. `adamw` / `nesterov` / `sgd_momentum` — neural network variants

**Tier 3**:
13-37: LP, QP, SOCP, SDP, PSO, multi-objective, etc.
