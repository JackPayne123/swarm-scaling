# Optimisation routes in published AlgoTune solvers - convex/fitting batch

Tasks: qp, chebyshev_center, robust_linear_program, quantile_regression, lasso, least_squares, robust_kalman_filter.

Sources: published final `solver.py` per model, `agent_summary.json` speedups, and the harbor task code (`tests/evaluator.py`, `tests/test_outputs.py`, `task.toml`, `environment/Dockerfile`). No code was run. Every speedup below is a published number. Every "worth Nx" is a ratio of two published numbers from different models' runs.

## Summary

| Task | Class | Reason |
|---|---|---|
| qp | S | Every valid solver (1.38-1.89) gets its gain from calling OSQP directly instead of through CVXPY. Turning polish off adds about 1.2x. Looser tolerances add at most 1.1x. |
| chebyshev_center | M (one dominant switch) | Swapping CVXPY+Clarabel for HiGHS gives about 4.75x in a tight cluster of 7 solvers. Presolve off and bypassing `linprog` each add 1.2-1.35x, and Gemini 3.1 combines both for 7.07. |
| robust_linear_program | S | Building the cone matrix with numpy and calling `ecos.solve` directly gives 6.4-7.0x. Keeping CVXPY with any solver stays at 1.0-1.07. Vectorized assembly adds 1.05x and loose tolerance adds nothing. |
| quantile_regression | A | Three ways of solving the LP each replace sklearn's split-variable HiGHS LP: free-beta primal HiGHS (1.15-1.20), dual LP with HiGHS (1.37-1.41), or a primal IPM via CVXPY+ECOS (1.51-1.69). No solver combines them. |
| lasso | M | float32 data (about 1.5-1.8x) and a lean CD loop without duality-gap checks (about 1.35-1.6x) act on different costs. Gemini 3 Pro stacks both plus wrapper bypass for 3.19, close to the product of the parts. |
| least_squares | U (leaning S) | All but one solver above 1.3 use linear least squares for the polynomial family. Code that matches line for line scored anywhere from 1.33 to 2.29, so cross-model natural experiments cannot resolve the secondary components. |
| robust_kalman_filter | A | Two different whole-solve replacements on the reduced w-only problem top the list: IRLS + Riccati (GPT-5.4, 20.9) and L-BFGS-B + adjoint gradient (Opus 4.5, 18.8). Inside the CVXPY route, vectorizing the objective and the constraints stack (2.1x and 3.5x), but that route stops near 9x. |

## Caveats that apply to every task

1. **Cross-run noise is large.** The published speedups come from separate runs per model. Pairs with near-identical code differ by up to 1.33x (least_squares Gemini 2.5 Pro 1.33 vs GPT-5 Pro 1.77) and 1.22x (chebyshev Opus 4 4.70 vs Opus 4.5 5.74, which differ only in bounds format). Natural-experiment ratios below about 1.25x are within noise.
2. **Library versions differ between runs and from the harbor image.**
   - Qwen3 Coder's chebyshev solver uses `linprog(method='interior-point')`, which SciPy removed in 1.11, and its published result is valid at 6.16. Its run therefore had an older SciPy.
   - Gemini 3.1's lasso passes `do_screening=1` to sklearn's `cd_fast`, so its run had a newer sklearn, whose reference Lasso may itself be faster.
   - The harbor Dockerfile installs unpinned latest packages (scipy, osqp, cvxpy, sklearn, ecos, highspy), so some published solvers may break or change speed locally. Qwen's chebyshev solver will raise there.
3. **Problem sizes.** `task.toml` carries `algotune_problem_size` equal to `PROBLEM_SIZE`. I assumed the published runs used the same n. The sizes are qp 287, chebyshev 192 (n+1 = 193 dims), robust LP 12 (m = 120 cones), quantile 319, lasso 383 (X is 1915 x 3830), least_squares 98304, and robust Kalman 12 (N = 24, state 6).
4. **The harbor harness validates only the warmup call.** `test_outputs.py` checks `is_solution` once per instance (warmup), then times the minimum of 10 unchecked repeats of the same instance. State carried across calls (warm starts, cached solver objects) is therefore rewarded and never re-validated. No published solver in this batch caches results keyed on problem content. These carry state:
   - qp: Gemini 2.5 Pro (cached OSQP object, `update` + `warm_start=True`), GPT-5.2 (warm start from the last x), GPT-5.4 (cached OSQP solver with `update` and `warm_start=True`).
   - chebyshev: Gemini 3.1 keeps one `highspy.Highs()` object and calls `passModel` each time. I could not tell whether `passModel` discards the previous basis. If it keeps it, timed repeats warm-start from the optimal basis.
   - robust_kalman: GPT-5.2 caches a parametrized CVXPY problem.
5. N/A is sometimes run-level rather than code-level. Two examples:
   - GPT-5 Pro's chebyshev code is the same `linprog(method="highs")` LP as five valid solvers, yet it is N/A.
   - Opus 4.1's robust LP `solver.py` contains pasted log text, not code.

---

## qp

**Reference and verifier.** The reference builds the QP in CVXPY (`quad_form(psd_wrap(P))`) and solves it with OSQP at eps 1e-8 (CVXPY's OSQP interface turns polish on by default). The verifier requires `G x - h <= 1e-6`, `np.allclose(A x, b, atol=1e-6)` (default rtol 1e-5 also applies), and `0.5 x'Px + q'x <= ref_obj + 1e-4 (1 + |ref_obj|)`. The objective it compares against comes from re-running the reference. It does not check the returned "objective" field. Dense data, n = 287, m = p = 143.

| Model | Speedup | Components |
|---|---|---|
| GPT-5 | 1.892 | Direct OSQP call. Phase 1 at eps 1e-6 with polish off and max_iter 2000, accepted after a 1e-6 feasibility self-check. Phase 2 refine at 1e-8 only on failure. |
| Claude Opus 4.1 | 1.740 | Direct OSQP, eps 1e-8, polish default (off). |
| Gemini 3.1 Pro | 1.734 | Direct OSQP, eps 1e-8, `polish=False`, `adaptive_rho=False`. |
| Gemini 3 Pro | 1.728 | Direct OSQP, eps 1e-8, polish off. Cython builds the CSC arrays from lists. KKT closed form only when there are no inequalities (never here). |
| Claude Opus 4 | 1.703 | Direct OSQP, eps 1e-8, polish off. |
| Gemini 2.5 Pro | 1.677 | Direct OSQP, eps 1e-8, polish off. Caches the OSQP object by sparsity pattern and calls `update` + warm start. |
| GPT-OSS-120b | 1.661 | Direct OSQP, eps 1e-8, polish off. |
| Claude Opus 4.5 | 1.659 | Direct OSQP (upper-triangular P), eps 1e-8, polish off. SLSQP path only for n <= 20. |
| GPT-5.2 | 1.655 | Direct OSQP, eps 1e-8, polish off, warm start from the previous call. Dense IPM only for n <= 60, so it never runs here. |
| R1 | 1.636 | Direct OSQP, eps 1e-8, polish off. |
| Qwen3 Coder | 1.565 | Direct OSQP, eps 5e-5, polish on, max_iter 200, refine on `solved_inaccurate`. |
| GLM-4.5 | 1.538 | Direct OSQP, eps 1e-6, polish on, max_iter 500. |
| o4-mini | 1.437 | Direct OSQP, eps 1e-4, polish on. |
| GPT-5.4 | 1.381 | Cached OSQP solver with `update`, eps 1e-8, polish on, `check_termination=25`, plus an extra active-set KKT re-solve. Brute-force active-set enumeration only for n <= 8. |
| GPT-5 Pro | 1.381 | Direct OSQP, eps 1e-8, polish on. |
| Claude Opus 4.6 | 1.379 | Direct OSQP, eps 1e-7, polish on. |

**Natural experiments.**
- Polish on vs off at tight eps: Opus 4 (1.703, polish off) vs Opus 4.6 (1.379, polish on, eps 1e-7) gives 1.24x. The groups separate cleanly. Polish-off solvers have median 1.69 (n = 10) and polish-on solvers have median 1.41 (n = 6), a ratio of about 1.20. Polish costs a second factorization.
- Looser eps with polish on: o4-mini (eps 1e-4) 1.437 vs Opus 4.6 (eps 1e-7) 1.379 gives 1.04. A 1000x looser tolerance buys almost nothing, so OSQP iterations are not the main cost. Setup/factorization is.
- Looser eps with polish off: GPT-5 (1e-6) 1.892 vs Opus 4 (1e-8) 1.703 gives 1.11.
- `adaptive_rho=False`: Gemini 3.1 1.734 vs Opus 4 1.703 gives 1.02.
- Cached solver + `update` + warm start: Gemini 2.5 1.677 vs Opus 4 1.703 gives 0.98.
- Cython matrix assembly: Gemini 3 Pro 1.728 vs Opus 4 1.703 gives 1.01.

**Class S.** Every valid solver is within 1.5x of the best, and all of them get their gain from the same change, OSQP called directly with no CVXPY canonicalization. Polish off adds about 1.2x on top. Nothing else adds more than 1.1x.

**Validity.**
- o4-mini (eps 1e-4) and Qwen (eps 5e-5, max_iter 200) rely on polish succeeding. If polish fails, OSQP keeps the unpolished iterate, whose constraint violation (about 1e-4) exceeds the 1e-6 check. These are fragile on some instances, though they were valid in their runs.
- GLM checks status `'solved inaccurate'` with a space. OSQP returns `'solved_inaccurate'`, so an inaccurate solve raises.
- GPT-5's phase 2 calls `prob.update(eps_abs=...)`, which older osqp versions reject. The error is caught and falls back to CVXPY, so it costs only time.
- No exploit seen. The verifier ignores the returned objective field.

**Single change to test.** Replace the CVXPY block with `osqp.OSQP().setup(P=triu(P) csc, q, A=vstack(G, A) csc, l=[-inf; b], u=[h; b], eps_abs=1e-8, eps_rel=1e-8, polish=False)` and return `res.x`. Expect about 1.7x. This is untested, but GPT-5.2 already wrote a dense primal-dual IPM (gated at n <= 60). An n = 287 dense IPM using LAPACK Cholesky might beat OSQP's single-threaded LDL of a dense KKT stored as sparse. This needs to be measured.

---

## chebyshev_center

**Reference and verifier.** The reference solves the LP `max r s.t. a x + r ||a_i|| <= b` in CVXPY with Clarabel. The verifier re-solves, computes the inscribed radius `min((b - a x)/||a_i||)` for both points, and accepts if `np.allclose(r_ref, r_prop, atol=1e-6)` (rtol 1e-5 also applies). It checks only the radius at the returned x, not the returned r, and has no feasibility check beyond that. Dense a is 192 x 193. The generator builds `a = [A; -A]` with `b = a x_p + U(0,1)`, so every constraint comes in a parallel pair (a slab).

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro | 7.070 | highspy `passModel` with a cached `HighsLp` structure, presolve off, dual simplex (`simplex_strategy=1`), persistent Highs object. |
| GPT-5.4 | 6.560 | `linprog(method='highs-ds', presolve=False)`. Numba vertex enumeration only for n <= 4, so it never runs here. |
| Gemini 3 Pro | 6.425 | highspy `addCols`/`addRows` with dense row arrays, presolve default (on). |
| GPT-5.2 | 6.337 | `linprog(highs-ds, presolve=False)`. Numba enumeration only for n <= 3. |
| Qwen3 Coder | 6.156 | `linprog(method='interior-point')` (legacy SciPy IPM, needs SciPy < 1.11). |
| Claude Opus 4.6 | 5.948 | highspy `addVars` + `addRows` from CSR, presolve on. |
| Claude Opus 4.5 | 5.736 | `linprog(highs)` with bounds as an ndarray. |
| o4-mini | 5.646 | `linprog(highs-ds, presolve=False)`. |
| Gemini 2.5 Pro | 4.911 | `linprog(highs-ds)`, presolve on. |
| Claude Opus 4.1 | 4.873 | `linprog(highs)`, bounds as a list of tuples. |
| GPT-5 | 4.805 | Same as Opus 4.1. |
| GLM-4.5 | 4.748 | Same as Opus 4.1 (presolve on). |
| GPT-OSS-120b | 4.748 | Same as Opus 4.1. |
| GPT-5-mini | 4.744 | `linprog(highs)` on the row-normalized LP. |
| Claude Opus 4 | 4.698 | `linprog(highs)`, list bounds. |
| R1 | 3.693 | highspy, but builds 192 rows with Python loops of (index, value) tuples and per-row `addRow`. |

**Natural experiments.** The `linprog(highs)` list-bounds cluster is tight (4.70-4.91, n = 7, median about 4.75), so effects here are easier to see than on other tasks.
- `linprog` highs-ds presolve on vs off: Gemini 2.5 (4.91) vs o4-mini (5.65) gives 1.15. Against GPT-5.2 or GPT-5.4 (6.34 / 6.56, same configuration as o4-mini) the ratio is 1.29-1.34. o4-mini vs GPT-5.4 is itself 1.16 apart on identical configuration.
- Bypassing `linprog` with direct highspy, presolve still on: Gemini 3 Pro 6.43 vs cluster 4.75 gives 1.35. Opus 4.6 gives 1.25.
- Bounds ndarray vs list of tuples: Opus 4.5 5.74 vs Opus 4 4.70 gives 1.22. This is at the noise threshold, but it suggests `linprog` input cleaning is visible at this scale.
- highspy + presolve off + dual simplex + cached LP: Gemini 3.1 7.07 vs Gemini 3 Pro 6.43 gives 1.10. Against the cluster it is 1.49.
- How the model is built matters. R1 also uses highspy but feeds it with Python loops and scores 3.69, 0.78x of `linprog`.

**Class M (one dominant switch).** Replacing CVXPY+Clarabel with HiGHS is about 4.75x of the best 7.07x. Two further components, presolve off (1.2-1.35x) and skipping the `linprog` wrapper (1.25-1.35x), each act on a different part (solver configuration vs Python input handling). They exceed the 1.2x S threshold, and Gemini 3.1 stacks them. If the local test shows presolve-off below 1.2x, reclassify as S.

**Validity.** HiGHS simplex returns a vertex, so the radius is exact. Qwen's legacy IPM (tol 1e-9) is accurate but will raise on current SciPy. Gemini 3.1's persistent Highs object may warm-start timed repeats in the harbor harness (caveat 4). No exploit seen.

**Single changes to test.**
1. Evidence-backed: replace the CVXPY block with `linprog(c, A_ub=[a | ||a_i||], b_ub=b, bounds=..., method='highs-ds', options={'presolve': False})`, or the same LP through highspy `passModel`. Expect 5-7x.
2. Untested, and not used by any published solver: the generator's paired rows give the LP a closed form. Write a = [A; -A] and b = [b1; b2]. Summing each pair gives `r <= (b1_i + b2_i) / (2 ||A_i||)` for every i. All of these bounds are reached at once by any x solving the 96 x 193 underdetermined system `A x = (b1 - b2)/2`, for example `x = lstsq(A, (b1 - b2)/2)`. That x is an exact Chebyshev center, so the verifier's radius check passes. With a structure check (`np.allclose(a[m:], -a[:m])`) and an LP fallback, this should cost about one small lstsq plus list conversion. It depends on the generator's slab structure rather than on the verifier, but the gain needs to be measured.

---

## robust_linear_program

**Reference and verifier.** The reference builds a CVXPY problem with 120 `cp.SOC(b_i - q_i'x, P_i' x)` constraints in a Python loop (n = 12, P_i a random 12 x 12 matrix) and solves it with Clarabel. The verifier checks that the returned objective matches `c'x` (rtol 1e-5) and that every SOC constraint holds within **1e-8 absolute**. It accepts an objective within rtol 1e-2 of the reference's. It also handles unboundedness via inf. `x` must be an ndarray (`.shape` is accessed).

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 7.008 | Direct `ecos.solve`. Vectorized G fill into a cached CSC buffer. LP/scalar special cases unused here. |
| Gemini 3.1 Pro | 6.838 | Direct ECOS, vectorized G (3-D numpy fill), CSC built without validation. |
| Gemini 3 Pro | 6.759 | Direct ECOS, vectorized G, unvalidated CSC. **Tightens b by 1e-6** so the solution satisfies the 1e-8 feasibility check. |
| Claude Opus 4.6 | 6.631 | Direct ECOS, per-constraint Python loop to stack G. |
| Claude Opus 4.5 | 6.623 | Direct ECOS, loop-filled dense G, tolerances 1e-8 (ECOS defaults). |
| Gemini 2.5 Pro | 6.507 | Direct ECOS, loop-filled dense G. |
| o4-mini | 6.491 | Direct ECOS, `vstack` loop. |
| GPT-5.2 | 6.442 | Direct ECOS with cached buffers. First try: tolerances 3e-6 and max_iters 35, then a violation check (<= 1e-8). Retry at 1e-9 if the check fails, CVXPY last. |

All other valid solvers (Opus 4 1.056, GPT-5-mini 1.061, GLM 1.058, GPT-OSS 1.058, Qwen 1.064, GPT-5 1.008, R1 0.998) keep the per-constraint CVXPY loop and only change the solver.

**Natural experiments.**
- Switching the solver inside CVXPY (Clarabel to ECOS): Opus 4 / GPT-5-mini about 1.06.
- Dropping CVXPY, same ECOS solver: about 6.5 / 1.06, which is about 6.2x.
- Vectorized G assembly vs Python-loop assembly: (6.76, 6.84, 7.01) vs (6.49-6.63) gives about 1.05.
- Loose ECOS tolerance with a feasibility check: GPT-5.2 6.44 vs the vectorized group's 6.76-7.01 gives 0.92-0.95, so no gain.

**Class S.** CVXPY canonicalization of 120 looped SOC constraints is nearly the whole reference cost. Every solver within 1.5x of the best uses a direct ECOS call, and no other component exceeds 1.05x.

**Validity.** The 1e-8 absolute SOC feasibility check is tight. Only two solvers guard against it:
- Gemini 3 Pro solves a tightened problem (b - 1e-6). The objective tolerance (1e-2) absorbs the shift.
- GPT-5.2 checks the violation and re-solves if it is too large.

The others (Opus 4.5, Opus 4.6, Gemini 2.5, Gemini 3.1, GPT-5.4, o4-mini) trust ECOS's interior iterate. They passed in their runs but could fail an instance where ECOS lands a hair outside a cone. No exploit seen.

**Single change to test.** Assemble G (120·13 x 12) and h with numpy and call `ecos.solve(c, G_csc, h, {'l': 0, 'q': [13]*120})`. Return `x` and `c @ x`. Optionally use Gemini 3 Pro's b - 1e-6 tightening. Expect about 6.5-7x.

---

## quantile_regression

**Reference and verifier.** The reference is sklearn `QuantileRegressor(quantile=0.5, alpha=0, solver='highs')`. That is a HiGHS LP with coefficients split into positive and negative parts, so it has 2(p+1) + 2n columns, with n = 319 and p = 106. The verifier re-fits and requires coef, intercept and predictions within atol 1e-5 of the reference. That works only because the LP optimum is unique. n = 319 is odd, so tau·n is not an integer, and I believe median regression is then unique for continuous data. If the published runs used an even n, an IPM and a vertex method could return different points on a non-singleton optimal face. Nothing is random at solve time.

| Model | Speedup | Components |
|---|---|---|
| GLM-4.5 | 1.687 | CVXPY + **ECOS** (IPM) on the free-beta primal with an intercept column. |
| Claude Opus 4.5 | 1.515 | CVXPY + ECOS on the free-beta primal (nonneg u, v). |
| Gemini 2.5 Pro | 1.411 | **Dual LP** (n box variables, p+1 equalities) via `linprog(highs)`. Coefficients read from `eqlin.marginals`. |
| Gemini 3.1 Pro | 1.371 | Dual LP via `linprog(highs)`, bounds (0, 1), coefficients from marginals. |
| GPT-5 | 1.198 | Free-beta primal via `linprog(highs)`, COO-built sparse A_eq. |
| GPT-5-mini | 1.182 | Free-beta primal, sparse `hstack`, `linprog(highs)`. |
| R1 | 1.177 | Free-beta primal, sparse, `linprog(highs)`. |
| o4-mini | 1.175 | Free-beta primal, sparse, `linprog(highs)`. |
| GPT-OSS-120b | 1.172 | Free-beta primal, sparse, `linprog(highs)`. |
| Claude Opus 4.1 | 1.170 | Free-beta primal, dense A_eq, `linprog(highs-ds)`. |
| Claude Opus 4 | 1.149 | Free-beta primal, dense A_eq, `linprog(highs)`. |

Below 1.1: GPT-5.4 1.048 (sklearn's split formulation fed straight to `_linprog_highs`), GPT-5 Pro 0.989 (the reference itself), GPT-5.2 0.987, Qwen 1.001 (sklearn with `highs-ds`). Two invalid solvers used direct highspy on the primal: Opus 4.6 and Gemini 3 Pro (Cython helper). I could not tell why they failed.

**Natural experiments.**
- Dropping sklearn's wrapper but keeping its split formulation: GPT-5.4 1.05.
- Free beta instead of the +/- split, `linprog` HiGHS: about 1.17 (n = 7, range 1.15-1.20), about 1.12x over the split formulation.
- Dual LP instead of primal, both on `linprog` HiGHS: 1.37-1.41 vs 1.17 gives about 1.2x.
- IPM (ECOS) instead of HiGHS on the free-beta primal, despite CVXPY overhead: 1.51-1.69 vs 1.17 gives about 1.3-1.45x.
- The two ECOS solvers differ by 1.11x with near-identical code (noise).

**Class A.** The dual-LP route (about 1.4) and the IPM route (about 1.5-1.7) both replace the same computation, the LP solve, and reach similar speedups (their gap is within noise). No published solver combines them. Whether "dual LP solved by an IPM" or "primal via a direct `ecos.solve` without CVXPY" stacks is untested. If it does, this becomes M.

**Validity.** ECOS returns an interior-point iterate. It matches the HiGHS vertex within 1e-5 only because the optimum is unique and ECOS converges to about 1e-8. Both ECOS solvers are fragile to degenerate instances (see the even-n note above). The dual-LP solvers read coefficients from HiGHS dual marginals. These are exact at a vertex but are another place where degeneracy would bite. No exploit seen.

**Single change to test.** Replace `QuantileRegressor` with the free-beta primal solved by ECOS. The evidence-backed version goes through CVXPY (GLM's code). The untested, likely faster version calls `ecos.solve` directly with G = [-I] on u, v. Also time `linprog` on the dual with `method='highs-ipm'`. Expect 1.4-1.7x for the evidence-backed version.

---

## lasso

**Reference and verifier.** The reference is sklearn `Lasso(alpha=0.1, fit_intercept=False)` with defaults (tol 1e-4, max_iter 1000, cyclic CD with duality-gap stopping) on X 1915 x 3830 (float64, an ndarray in harbor). The verifier accepts any w with `obj(w) <= obj(w_sklearn) + 1e-5`, where `obj = ||y - Xw||^2/(2n) + 0.1||w||_1`. It never compares coefficients.

The absolute 1e-5 is small next to an objective of order tens. The real slack is sklearn's own suboptimality at its stop. Its gap bound is `tol·||y||^2/n`, roughly 0.04 in these units if var(y) ≈ s = 383. The actual gap at stopping is smaller and unknown. Any solver at least as converged as sklearn passes.

| Model | Speedup | Components |
|---|---|---|
| Gemini 3 Pro | 3.193 | **float32** X and y. Custom Cython CD using BLAS `sdot`/`saxpy`. Stops when max \|dw\| < 1e-4 (absolute), with **no duality-gap computation**. Skips the residual update when \|dw\| <= 1e-7, a small inconsistency between w and r. No sklearn wrapper. |
| Qwen3 Coder | 1.732 | sklearn Lasso on **float32** inputs, `max_iter=150` cap, tol 1e-5. |
| R1 | 1.567 | Numba CD in float64. One dot plus one axpy per changed coordinate (sklearn uses up to two axpys). Stops when max \|dw\| < 1e-4 absolute, with a cheap objective check every 10 sweeps and no duality gap. |
| o4-mini | 1.181 | Calls sklearn's private `enet_coordinate_descent` directly (wrapper bypass), float64, same tol. |
| GPT-5.2 | 1.180 | Cached Lasso with `copy_X=False`, X built Fortran-ordered. |
| Claude Opus 4.5 | 1.136 | Direct `enet_coordinate_descent` (wrapper bypass). |
| GPT-5.4 | 1.130 | Direct `enet_coordinate_descent` (wrapper bypass). |

Below 1.1: GPT-5 1.0998, Gemini 3.1 1.073 (direct `cd_fast` with `do_screening=1`), GLM 1.037, GPT-5-mini 1.044, GPT-OSS 0.996, Gemini 2.5 0.92. Opus 4 / 4.1 / Sonnet were 0.21-0.33. Opus 4.6 (numba active-set CD, absolute 1e-4 stop) and GPT-5 Pro (numba CD with tol 1e-8) are N/A.

**Natural experiments.**
- Wrapper bypass alone (same algorithm and precision): o4-mini, Opus 4.5, GPT-5.4 and GPT-5.2 give 1.13-1.18, mean about 1.16.
- float32, still inside sklearn (confounded with the max_iter 150 cap and tol 1e-5): Qwen 1.73, about 1.5x over the bypass group.
- Lean CD loop without gap checks, float64: R1 1.57, about 1.35x over the bypass group.
- float32 on top of a lean CD loop: Gemini 3 Pro 3.19 vs R1 1.57 gives 2.04. This is consistent with halving the bytes read per sweep in a memory-bound loop (X is 59 MB in float64).
- Lean CD + wrapper bypass on top of float32: Gemini 3 Pro 3.19 vs Qwen 1.73 gives 1.84.
- Product check: 1.73 (float32) x 1.57 (lean CD) x 1.16 (bypass) / 1.16 (bypass counted in R1 already) ≈ 2.7-3.2, against 3.19 observed.

**Class M.** Precision (bytes per sweep) and the loop/stopping structure (passes per sweep and gap evaluations) are different costs. Each is worth more than 1.3x alone, and the best solver stacks them with a near-multiplicative result. Wrapper bypass adds about 1.16x.

**Validity.**
- Everything above 1.5x relies on verifier slack that comes from sklearn's suboptimality, not from the 1e-5 constant.
- Gemini 3 Pro (float32, absolute stop, skipped tiny residual updates), R1 (absolute stop) and Qwen (float32 plus a 150-sweep cap) pass only if they end at least as close to optimal as sklearn did. They passed in their runs. Opus 4.6's numba CD with the same absolute-1e-4 stop is N/A, which shows this family can fail.
- Qwen's `max_iter=150` is the most fragile. Any instance needing more than 150 sweeps returns an under-converged w.
- No exploit beyond tolerance reliance.

**Single change to test.** `Lasso(alpha=0.1, fit_intercept=False).fit(X.astype(np.float32), y.astype(np.float32))` with defaults otherwise. Expect about 1.5-1.7x if float32 alone carries Qwen's gain. Second test: a numba float32 CD with dot + axpy and an absolute max-dw stop, Gemini 3 Pro's design, for about 3x. Check the objective margin on all 100 instances, not only pass/fail.

---

## least_squares

**Reference and verifier.** Each instance picks one of five model families with equal weight at this n: polynomial (degree 1-8), exponential, logarithmic, sigmoid and sinusoidal. There are n = 98304 points with x in [0, 9831], plus optional seasonality (p = 0.5) and outliers (p = 0.4). x is linspace 70% of the time and sorted uniform otherwise. The reference fits with `scipy.optimize.leastsq` (MINPACK LM with a finite-difference Jacobian) from a fixed start with maxfev 10000. It then computes y_fit, `residuals.tolist()` (98304 floats), mse and a convergence dict.

The verifier reads only `params`. It recomputes MSE and accepts if `mse <= 1.05 · mse(reference params)`. The other output fields are never read, so dropping them is allowed. The comparison is against a local optimum reached from the reference's start, not against the global optimum.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 3.810 | Polynomial: **closed-form linear LS**, centred normal equations for degree <= 3, `vander` + `lstsq` above that. Others: the reference `leastsq` with the same start and maxfev. **Output trimmed** to params only. |
| R1 | 2.317 | Polynomial: iterative Vandermonde + `lstsq`. Others: `scipy.least_squares` (TRF) with bounds and **new starts** (FFT frequency for sinusoid, log-fit for log), run in a forked subprocess with a 9 s timeout. Falls back to `leastsq` maxfev 20000. Output trimmed. |
| GPT-5 | 2.290 | Polynomial: `vander` + `lstsq`. Others: identical reference `leastsq`. Output trimmed. (Its custom LM code is never called.) |
| Gemini 3 Pro | 2.280 | Polynomial: `polyfit`. Others: `leastsq` from the same start with **numba residuals and numba analytic Jacobians** (`Dfun`). Output trimmed. |
| GPT-5.2 | 2.017 | Polynomial: `polyfit`. Others: same start, **numpy analytic Jacobian** for exp, sigmoid and sinusoid (log keeps finite differences). Output trimmed. |
| Claude Opus 4 | 2.015 | Polynomial: `polyfit`. Others: `curve_fit` (same LM, same start) with **maxfev 5000**. Output trimmed. |
| Claude Opus 4.6 | 1.910 | Polynomial: `polyfit`. Others: `leastsq` with **new starts** for sigmoid (range/min) and sinusoid (FFT frequency, mean offset). Output trimmed. |
| GPT-5 Pro | 1.771 | Polynomial: `polyfit`. Others: identical reference `leastsq`. Output trimmed. |
| Qwen3 Coder | 1.427 | **No polynomial change.** Reference `leastsq` with **maxfev 2000**. Output trimmed. |
| Gemini 2.5 Pro | 1.325 | Polynomial: `polyfit`. Others: identical reference `leastsq`. Output trimmed. |
| GPT-OSS-120b | 1.295 | Polynomial: `vander` + `lstsq`. Others: `curve_fit` maxfev 5000, which **returns p0 on RuntimeError**. **Output kept.** |
| GLM-4.5 | 1.211 | Polynomial: `polyfit`. Others: `curve_fit` maxfev 10000. Output kept. |

Below 1.1: Opus 4.1 1.090 (`curve_fit` for every family including polynomial, output kept, effectively the reference). Invalid: Opus 4.5, Sonnet 4.5, Gemini 3.1, GPT-5-mini, o4-mini (not read).

**Natural experiments.**
- **Same route, different runs.** GPT-5 (2.29), GPT-5 Pro (1.77) and Gemini 2.5 (1.33) all combine polynomial closed form, the identical reference path elsewhere, and trimmed output. Their spread is 1.73x. GPT-5 uses `vander` + `lstsq` while the other two use `polyfit`, which cannot explain a 1.3x gap between GPT-5 Pro and Gemini 2.5, whose code is line-for-line equivalent.
- Cheap polynomial solve (normal equations for degree <= 3) vs `lstsq`: GPT-5.4 3.81 vs GPT-5 2.29 gives 1.66. This is real only if polynomial instances remain a large share of solver time.
- Analytic Jacobian: GPT-5.2 2.02 vs GPT-5 Pro 1.77 gives 1.14, but vs GPT-5 2.29 it gives 0.88.
- Numba residuals + Jacobians: Gemini 3 Pro 2.28 vs GPT-5 Pro 1.77 gives 1.29, but vs GPT-5 it gives 1.00.
- Output kept vs trimmed (polynomial closed form in both): GLM 1.21 and GPT-OSS 1.29 vs the trimmed group 1.33-2.29, a ratio of 1.03-1.77.
- maxfev cap without the polynomial change: Qwen 1.43 vs Opus 4.1 1.09 gives 1.31. This includes output trimming, and suggests some reference fits run past 2000 evaluations. Opus 4's `curve_fit` raises when maxfev is hit and Opus 4 was valid, so in its run no non-polynomial fit needed more than 5000 evaluations. The long fits are therefore either polynomial or between 2000 and 5000.

**Class U, leaning S.** Every solver at 1.3 or above except Qwen uses linear least squares for the polynomial family. Every component secondary to that has a natural-experiment ratio inside the 1.33-1.73x noise band seen for identical routes:
- trimming the output,
- a cheaper polynomial solve,
- analytic or numba Jacobians,
- maxfev caps.

The decisive local test is a 2x2 on the reference: polynomial closed form on/off x output trimming on/off. Add a third arm with normal equations vs `lstsq` for the polynomial solve, and log per-family reference time to see which family dominates. If output trimming or the polynomial-solve method is worth 1.3x or more on its own, the class becomes M (they act on different instances or stages).

**Validity.** The MSE criterion is relative to the reference's local optimum from a fixed start.
- **Safe:** solvers whose non-polynomial path is the reference's LM from the same start with the same maxfev (GPT-5.4, GPT-5, GPT-5 Pro, Gemini 2.5, GLM) reproduce the reference exactly on those families. Their polynomial path is the global LS optimum, so MSE <= reference up to rounding.
  - Mild rounding risk: unscaled `vander` + `lstsq` at degree 7-8 with x up to 9831 (columns about 1e32) is rank-truncated at rcond about 2e-11 (GPT-5, GPT-OSS, R1). GPT-5.4's normal equations square the condition number. They passed in their runs, and noise at 0.25·std(y) makes the MSE ratio insensitive.
- **Fragile, different optimizer path:** GPT-5.2 and Gemini 3 Pro (analytic Jacobians change the LM iterates). On the multimodal sinusoid, a different iterate sequence can settle in a different local minimum, worse or better than the reference's.
- **Most fragile, different starts or algorithm:** Opus 4.6 and R1. An FFT frequency guess assumes uniform spacing, which fails for the 30% of instances with random x. With seasonality (50% of instances) the FFT peak can be the seasonal frequency. Either way LM can land in a worse basin than the reference's b = 1 start. R1 also uses TRF with bounds and a forked subprocess per call (fork start method required).
- **Fragile via caps:**
  - Qwen (maxfev 2000) returns an earlier iterate if the reference needed more.
  - Opus 4 (`curve_fit` maxfev 5000) raises if hit.
  - GPT-OSS (`curve_fit` maxfev 5000) returns the initial guess p0 on failure, which would almost certainly fail the 1.05 check.
- **Not an exploit, but verifier-dependent:** dropping residuals, mse and convergence_info is accepted only because `is_solution` reads `params` alone.

**Single change to test.** For `model_type == "polynomial"`, return `np.polyfit(x, y, deg)` (or scaled-Vandermonde `lstsq`) and skip LM. Separately, return only `params`. Expect 1.3-2.3x combined from published numbers. GPT-5.4's 3.81 suggests the cheap normal-equation solve for degree <= 3 is worth testing as a third arm.

---

## robust_kalman_filter

**Reference and verifier.** The reference is CVXPY with 24 per-timestep `huber(norm(v_t), M)` terms and 48 per-timestep equality constraints, solved with the default solver. n = 12 gives N = 24, a 6-dim state, and 3-dim w and v. The verifier requires `||x_hat[0] - x0|| <= 1e-5`, every dynamics and measurement residual norm <= 1e-5, finite values and correct shapes. It accepts an objective `J <= 1.01 · J_ref`, which is generous. If the reference solve fails, it accepts on feasibility alone.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 20.912 | **Different algorithm: IRLS / half-quadratic** for the Huber-of-norm. Each iteration solves the weighted quadratic problem exactly with a backward **Riccati (LQR) recursion** and a forward rollout. Up to 20 iterations, tol 3e-6 on the weights. x and v come from the rollout, so constraints are exact. Pure numpy. Returns ndarrays (no `tolist`). |
| Claude Opus 4.5 | 18.811 | **Eliminates x and v** (rollout), giving an unconstrained smooth problem in w (72 variables). **L-BFGS-B** with an **adjoint gradient in numba**. ftol 1e-9, gtol 1e-7. |
| Gemini 3.1 Pro | 13.030 | Same route as Opus 4.5 (numba adjoint, L-BFGS-B, preallocated buffers, fastmath) with **tighter tolerances** (ftol 1e-12, gtol 1e-8, maxiter 1000). |
| Gemini 3 Pro | 8.821 | CVXPY, **fully vectorized** (dynamics as one matrix constraint, v eliminated, `huber(norm(axis=1))`), Clarabel at tol_gap 1e-4. |
| Gemini 2.5 Pro | 8.630 | CVXPY fully vectorized, v eliminated, default solver. |
| GPT-5 Pro | 7.191 | CVXPY fully vectorized, v eliminated, ECOS. |
| o4-mini | 7.048 | CVXPY fully vectorized (transposed variables), ECOS. |
| Qwen3 Coder | 6.635 | CVXPY fully vectorized (v kept), OSQP requested. OSQP cannot take the SOC, so it falls back to ECOS with max_iters 50. |
| GPT-5.2 | 6.448 | CVXPY fully vectorized with **cp.Parameter caching** by shape, ECOS at 1e-6 and max_iters 50. |
| Claude Opus 4 | 3.474 | CVXPY with **constraints vectorized**, objective still 24 separate norms stacked with `vstack`, Clarabel. |
| GPT-OSS-120b | 2.825 | CVXPY with the **objective vectorized** and v eliminated, dynamics still a 24-step loop, ECOS. |
| GLM-4.5 | 2.141 | CVXPY with the **objective vectorized**, constraints still looped, SCS (eps 1e-5, max_iters 500). |
| R1 | 2.061 | CVXPY with the objective vectorized, constraints looped, SCS (eps 1e-5). |

Below 1.1: GPT-5 0.935, Opus 4.1 1.008, Sonnet 1.000. Opus 4.6 and GPT-5-mini are N/A.

**Natural experiments.**
- Inside CVXPY:
  - Objective vectorized only gives about 2.1x (R1, GLM).
  - Constraints vectorized only gives 3.47x (Opus 4).
  - Both gives 6.4-8.8x. The product 2.1 x 3.5 = 7.3 matches, so the two stack.
  - v elimination with the dynamics loop kept: GPT-OSS 2.83 vs R1/GLM 2.1 gives 1.35.
  - Loose Clarabel tolerance: Gemini 3 Pro 8.82 vs Gemini 2.5 8.63 gives 1.02.
  - Parameter caching: GPT-5.2 6.45 vs other ECOS-vectorized solvers 6.6-7.2, so no gain.
- L-BFGS tolerance: Opus 4.5 (1e-9 / 1e-7) 18.8 vs Gemini 3.1 (1e-12 / 1e-8) 13.0 gives 1.44. Within this route, iteration count is a real component.
- IRLS + Riccati vs L-BFGS + numba adjoint: 20.9 vs 18.8 gives 1.11.
- Leaving CVXPY for the reduced problem: about 18.8-20.9 vs about 8.8 gives 2.1-2.4x over the best CVXPY formulation.

**Class A.** The top two solvers (both within 1.5x of 20.9) use different algorithms, IRLS + Riccati and quasi-Newton on the reduced problem. Each replaces the whole CVXPY solve, so they do not stack. Two secondary structures exist. CVXPY vectorization is genuinely M internally (objective and constraints each about 2-3.5x) but stops near 9x. The L-BFGS route has a stackable tolerance component (about 1.4x) and presumably a numba component; there is no natural experiment for numba alone. One combination is untested and might stack: numba for GPT-5.4's Python-loop Riccati recursion (24 steps x up to 20 IRLS iterations).

**Validity.**
- The rollout-based solvers (GPT-5.4, Opus 4.5, Gemini 3.1) satisfy the constraints to rounding by construction. The 1% objective slack is generous, so they are robust.
- GPT-5.4 stops IRLS after 20 iterations or at tol 3e-6. IRLS for Huber is a majorization-minimization method with a monotone objective, so an early stop is still close.
- The CVXPY solvers that use the solver's own v and dynamics (R1 and GLM with SCS eps 1e-5, Qwen with ECOS max_iters 50) are exposed to the 1e-5 per-step residual check. SCS at eps 1e-5 is the most fragile. Those that recompute v = y - Cx after the solve make the measurement constraint exact.
- No exploit seen.

**Single changes to test.**
1. Inside CVXPY: vectorize to `x[1:] == x[:-1] @ A.T + w @ B.T` and `cp.sum(cp.huber(cp.norm(y - x[:-1] @ C.T, axis=1), M))`, then recompute v afterwards. Expect about 7-9x.
2. Whole-solve replacement: IRLS with weights `q_t = tau / max(1, ||v_t||/M)`, solved by a Riccati recursion (GPT-5.4), or L-BFGS-B on w with an adjoint gradient (Opus 4.5 settings). Expect about 19-21x.

---

## Follow-ups noticed (not acted on)

- Harbor protocol (caveat 4): validity is checked only on the warmup call, so timed repeats of the same instance are never re-validated and cross-call state is rewarded. Any locally evolved solver should be re-validated on every call, or the harness changed, before trusting its speedups.
- Unpinned harbor image: some published solvers will fail or change speed locally. Qwen's chebyshev `interior-point` will raise. osqp 1.x changed status strings and `update` keywords. `cd_fast` signatures vary across sklearn versions. Freezing versions before local tests would make results comparable.
- chebyshev_center's generator makes the LP decomposable (paired slabs). If the benchmark is meant to test LP solving, that structure is a generator weakness worth recording.
