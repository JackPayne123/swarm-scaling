# v7 task screen and candidate set (2026-10-14)

Read-only on the swarm-scaling repo. No cloud, no scorer, no API spend. Every local measurement ran on the Mac (arm64, 1 BLAS thread, other jobs on the machine) in a scratch venv (Python 3.12, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, cvxpy 1.9.3, numba 0.68.0), not in the task image, against the Harbor task code as scored (all 154 Harbor AlgoTune tasks downloaded into the scratchpad; content hashes match the project cache for the 32 it holds). The four input reports had been wiped from the scratchpad (the machine restarted at about 08:18); I recovered them verbatim from the subagent transcripts into `scratchpad/recovered/`.

## Summary

- 29 tasks pass the frozen PLAN.md rules plus the local half of the screen. 19 pass outright, 8 pass only with a named fix (a 1,000-instance validity pass at selection, a generator patch or an upstream verifier port), and 2 are weak but not excludable on measurement grounds. The v6 seven are among them.
- Beyond the frozen rules, the screen excluded 6 tasks on local evidence. sparse_pca, voronoi_diagram and randomized_svd accept trivial outputs. spectral_clustering's reference time is so concentrated that one instance carries 79% of it (btsp, already excluded, fails the same check). kmeans gives different verdicts for the same output. sparse_lowest_eigenvectors_posdef duplicates the eigenvalues task.
- The strata are uneven. On published code plus local one-change ablations, 14 of the 29 are "single decisive switch suspected", 10 "multiple composable routes" (provisional until each gets the same ablation), 3 "alternative routes" and 2 "unclear".
- I recommend running every task that also passes the scorer checks, with no draw. That should be 25-29 tasks x 2 repeats, about 50-58 groups, $33-38 at v6's measured $0.653 per group. Power for +20% is about 0.76-0.82 at v6's task-level sd.
- AlgoTune on Harbor is a weak testbed for team vs best-of-N. A single selected solver plus mostly single-switch tasks makes a team a best-of-N with transfer. A composable-by-construction variant ("AlgoTune bundles") would test division of labour directly with the same harness.

## What the screen measured, and where

| Evidence | Source | Measured or cited |
|---|---|---|
| Published per-model speedups (18 models, 154 tasks) | `reports/agent_summary.json`, AlgoTune repo HEAD dff9914 (2026-06-24), identical sha256 to `research/task-data/algotune_agent_summary.json` | cited |
| Published final solver code (17 models) | `results/<model>/<task>/solver.py`, same repo | read by 5 Opus subagents told not to open the swarm-scaling repo, so the route classes are blind to v6 outcomes |
| AlgoTune baseline timing reproducibility (3 baseline runs per task) | `reports/generation.json` | cited |
| Harbor vs upstream code drift | AST comparison of `generate_problem`, `solve` and `is_solution` for all 154 tasks | measured |
| Reference-time concentration, determinism, self-validity, trivial-output battery | `screen.py`, 100 instances (battery on 30), seeds 9,000,000+ | measured locally |
| Degenerate instances | `gencheck.py`, 1,000 instances, seeds 8,000,000+ (fewer where the 600 s cap hit) | measured locally |
| Rare invalidity of published solvers, local Harbor-format speedup | `runpub.py`, top 3 single-file published solvers per task, 400 instances, seeds 7,000,000+ | measured locally |
| Task-specific verifier probes, upstream verifier check | `probes.py`, `upcheck.py` | measured locally |
| List-to-array conversion share of the reference | `convfloor.py`, 10 instances | measured locally |
| One-change ablations of the reference | `ablate.py`, 25 instances | measured locally |
| Memory of 100 generated problems | `memcheck.py`, max RSS delta | measured locally |

In the AlgoTune paper (arXiv 2507.15887 v4, 24 Oct 2025), each task's size targets 100 ms of reference time on one core, speedup is reference over solver time with 1x for invalid or slower, and the aggregate is the harmonic mean. The authors call model speedups "mostly surface level" and describe one validator gamed with a trivial output (vector quantization). The leaderboard at algotune.io lists the same 18 models (GPT-5.2 2.05x top) and has no newer entries. I found no "AlgoTune Lite" definition in the repo or on the page I could read.

Two facts limit how far published evidence transfers to Harbor. First, the published harness passed large inputs as ndarrays and timed the materialisation of list outputs, while Harbor passes Python lists and times `solve()` only (subagent reading of `AlgoTuner/utils/serialization.py` and `isolated_benchmark.py`). Second, Harbor problem sizes differ from AlgoTune's own for 141 of 154 tasks. Published speedups that came from output handling shrink in Harbor: the best published least_squares solver (3.81x) runs at 1.09x locally, l0_pruning 24.7x at 4.9x, minimum_spanning_tree 27.0x at 5.8x, pde_heat1d 37.8x at 3.6x. Harbor also creates routes the published runs never needed (v6 Haiku agents wrote C list readers for sinkhorn).

## 1. The screen

### 1a. Hard exclusions (measurement validity)

Apply in order to every task in the pool. "Local" means model-free and runnable now; "scorer" needs the remote scorer.

| ID | Exclude when | Procedure | Where | Results now |
|---|---|---|---|---|
| A0 | Frozen PLAN.md rules: pilot task; reference runs a combinatorial search solver (CP-SAT, SAT, and, as my reading of rule (a)'s rationale, MILP branch and bound); reference is one library call on a fixed linear or bilinear transform | Read the reference `solve` | local | excluded kcenters (SAT), capacitated_facility_location (MILP), cumulative_simpson_1d (fixed transform), queens_with_obstacles and max_weighted_independent_set (CP-SAT); the v6 exclusions stand |
| A0d | Task duplicates another eligible task (same computation, same verifier) | Compare reference and verifier code | local | sparse_lowest_eigenvectors_posdef returns and checks eigenvalues, the same computation as sparse_lowest_eigenvalues_posdef. Keep the latter |
| A1 | Upstream AlgoTune has since closed a validity hole in this task's `is_solution` or `generate_problem`, and the fix is not ported into the v7 task before launch | AST diff, Harbor vs upstream HEAD; port or exclude, then rerun A2-A5 on the ported task | local | 27 of 154 Harbor tasks differ from upstream. Screened ones that matter: randomized_svd (Harbor accepts the trivial output below, upstream 0/10), pca (Harbor accepts any rotation of the top-k subspace 20/20, upstream 0/10, upstream still accepts the reference 10/10), optimal_advertising (upstream fixed the NaN generator), spectral_clustering |
| A2 | Any degenerate instance in 1,000 fresh instances at the Harbor size: non-finite input, non-finite reference output, generator exception, or reference output failing its own `is_solution`. Also exclude (or patch before launch) a generator whose output depends on global RNG state | `gencheck.py`; `gendet.py` generates the same seed under two different global numpy and `random` seeds | local | optimal_advertising (pilot, positive control) 6/1,000 non-finite inputs, all 6 reference-invalid, so a 100-instance set contains one with probability 0.45; spectral_clustering reference-invalid 9/1,000; all other 37 screened tasks 0. Non-deterministic generators: sparse_lowest_eigenvalues_posdef and _eigenvectors_ (`scipy.sparse.random` without `random_state`), so their instance sets change at every scoring |
| A3 | Reference time concentrated on few instances: effective number of instances ESS = (sum t)^2 / sum t^2 below 50 per 100, or one instance above 10% of total reference time | 100 fresh instances, warmup plus min of 3; cross-check on the 1,000-instance run | local | btsp ESS 2.1, one instance 50% of total (max/median 14,094); spectral_clustering ESS 1.6, one instance 79%. Lowest passing task least_squares at 61.6 (59 per 100 on 1,000), p99/median 5.9. Every other task 77 or more |
| A4 | Verifier accepts a trivial output on any of 30 instances | Battery: the reference output with every number zeroed; another instance's reference output; trivial outputs read off the verifier's code (identity factors with zero spectrum for randomized_svd, the uniform plan for sinkhorn) | local | sparse_pca zeros 30/30 and other-instance 29/29 (X = 0 is the exact answer on this generator); voronoi_diagram zeros 30/30 (verifier checks counts and shapes only); randomized_svd U = I, S = 0, V = I 20/20 under Harbor's tolerance (1.24 at this size). All other tasks rejected every trivial output |
| A5i | The same output gets different verdicts across repeated `is_solution` calls | Reference output, and one degraded output, 10 calls on each of 5 instances | local | kmeans (the verifier re-runs an unseeded KMeans and allows 5% slack): a 1-iteration Lloyd labelling was accepted in 1, 6, 0, 1 and 3 of 10 calls. Reference outputs never flipped in any task (100 instances each) |
| A5ii | A published solver that AlgoTune scored valid fails on 1-20 of 400 fresh instances, and v7 selection has no validity pass of at least 1,000 fresh instances per shortlisted candidate. With the pass in place, the task is admitted | `runpub.py`, up to 3 top single-file published solvers | local | sparse_lowest_eigenvalues_posdef 4/400 and 1/400; two_eigenvalues_around_0 1/400; robust_linear_program 1/400 for two solvers (1e-8 feasibility check); l1_pruning 1/400; pde_heat1d 1/400 for two solvers; sinkhorn 1/400. least_squares 0/400 for its top 3, but v6's subsampling solver failed 1.245% of 20,000 instances (v6-anomalies), so I flag it. A 100-instance check misses a 1.2% failure rate 29% of the time; 1,000 instances miss it with probability 4e-6 |
| A5iii | Validity pass unaffordable: 1,000 instances of `is_solution` above 30 core-minutes | Median `is_solution` time per instance | local | generalized_eigenvalues_complex 1.48 s per instance (25 core-minutes per candidate, under the bar but the costliest); psd_cone_projection 0.30 s; every other candidate 0.02-0.15 s |
| A6i | Reference against itself: the 3 speedups differ by more than 5% (max/min) | 3 score jobs of the reference, 100 instances, the v7 verifier copy | scorer | not run |
| A6ii | A known fast solver is unstable: 3 scorings differ by more than 10%, or any job has reference_inflation outside [0.87, 1.15] or first_call_ratio above 5 | The best published solver that is valid locally (or the one-change variant), 3 score jobs | scorer | not run |
| A6iii | Two timing modes: if A6ii's spread exceeds 10%, 3 more jobs; exclude if the six form two clusters more than 1.25x apart | as A6ii | scorer | not run; v6 pca is the pattern (7.06-7.25 vs 12.72-13.08 for one source) |
| A6iv | Flag only: the same solver's 20-instance stage-1 dev score differs from its hidden median by more than 1.28x (log 0.25) | 1 score job on finalize's dev instances | scorer | not run |

Notes on the thresholds. ESS 50, the 10% single-instance share, the 1-20 of 400 band, 5% and 10% spreads and the 1.25x two-mode gap are my choices, written with knowledge of the v6 anomalies (so the v6 tasks were not screened blind). Each targets a mechanism seen in v6 or the pilots, not an arm outcome. AlgoTune's own baseline reproducibility (spread of 3 baseline runs in `generation.json`) is a cheap pre-scorer warning; among candidates it is 5% or more only for sinkhorn (16.2%), stable_matching (8.3%), lasso and linear_system_solver (5.0%).

Some verifiers accept a cheaper computation than the reference. That is a legitimate route under AlgoTune's rules, so these are recorded, not excluded. sinkhorn accepts a 1-iteration Sinkhorn plan (5/5) but not the uniform plan (0/5); sparse_lowest_eigenvalues_posdef accepts ARPACK tol 1e-3 (10/10) but not 1e-2 (0/10); clustering_outliers checks summary statistics and ARI at least 0.5; lasso passes anything at least as converged as sklearn. A float32 cast of the reference output passes 30/30 in most tasks and is rejected in generalized_eigenvalues_complex, generalized_eigenvectors_complex and minimum_spanning_tree (0/30), qp (16/30) and robust_linear_program (7/30).

Memory is operational, not an exclusion. Memory of 100 generated problems (max RSS delta, Mac) exceeds 5 GiB for sinkhorn 8.9, stable_matching 8.5, linear_system_solver 6.3, lasso 5.5 and two_eigenvalues_around_0 5.2 GiB. These need the m-family agent boxes v6 used for sinkhorn; every other candidate is under 3 GiB.

### 1b. Strata (never exclusions)

Assign each task one stratum from model-free evidence, frozen before any v7 run:

1. Published route class, from the published solvers' code with this rubric. **S**: every published solver within 1.5x of the best published speedup gets most of its gain from one change, and natural experiments (pairs of published solvers differing by about one component) show no other component worth more than about 1.2x. **M**: two or more components each worth about 1.3x or more, acting on different parts of the computation, so they stack. **A**: two or more routes reach similar speedups by replacing the same computation, so they do not stack. **U**: none of these. The full readings are in `scratchpad/routes-*.md`.
2. Harbor overrides to S. (a) The task is conversion-dominated, meaning that converting the input lists to arrays takes at least 50% of the reference time, so the decisive component is the same fast list reader in every such task. (b) A one-change variant of the reference reaches at least 75% of the best published solver's local log speedup in Harbor format.
3. Report every result for all tasks, and pre-register two sensitivity analyses, one without stratum S and one with M tasks only. Report the S-vs-M difference as exploratory; with about 10 M tasks it will be imprecise.

Rule 2(b) has been applied to 7 tasks so far. Before launch it must be run on every M task, so that the override applies uniformly. The route reports name a candidate single change for each.

One-change ablations run so far (Mac, 1 thread, 25 instances, all variants valid 25/25):

| Task | Change to the reference | Local speedup | Best published solver, local | Share of best log speedup |
|---|---|---|---|---|
| psd_cone_projection | `eig` to `eigh` | 7.76 | 7.77 | 100% |
| eigenvalues_real | `eigh` to `eigvalsh` | 1.87 | 1.94 | 95% |
| lasso | sklearn Lasso on float32 | 1.43 | 1.39 | above 100% |
| pca | sklearn PCA to `eigh` of X^T X | 2.60 | 2.95 | 88% |
| sparse_lowest_eigenvalues_posdef | ARPACK tol 5e-4 | 2.53 | 2.93 | 87% |
| generalized_eigenvalues_complex | QZ to `eigvals(solve(B, A))`, float64 | 6.56 | 10.41 | 80% |
| min_weight_assignment | dense `linear_sum_assignment` | 1.22 | 1.78 | 34% |

The float32 variant of the generalized_eigenvalues_complex change gave 6.56, no gain over float64, on the Mac's arm64 BLAS at 1 thread; the published x86 runs put float32 at 1.40x. The precision component is hardware-dependent and has to be measured on the scorer.

## 2. Applying the screen to the full pool

Pool. The frozen headroom rule (best published speedup in [1.5, 10], median at least 1.05) gives 32 tasks; 6 are pilot tasks, and rules (a) and (b) leave 20 of the 26 held-out ones. That is too few for the power target, so I propose one pre-registered extension: raise the upper bound to 50x, everything else unchanged. It adds 20 tasks, of which 5 fall to A0, 15 enter the screen and 11 survive it. The bound is fixed from published data only. In Harbor format the published speedups above 10x mostly shrink (above), so the dynamic-range worry behind the 10x cap is smaller than it looks. Tasks in (50, 100] are not screened.

Exclusions applied (beyond A0):

| Task | Pool | Rule | Evidence |
|---|---|---|---|
| sparse_pca | ext | A4 | zeros accepted 30/30, another instance's answer 29/29 |
| voronoi_diagram | ext | A4 | zeros accepted 30/30; the top four published solvers (8-32x) never compute the diagram |
| randomized_svd | held-out | A4 (A1 port would readmit it for rescreening) | U = I, S = 0, V = I accepted 20/20 by Harbor, 0/10 by upstream |
| spectral_clustering | ext | A3, A2 | ESS 1.6, one instance 79% of reference time; reference invalid 9/1,000 |
| kmeans | ext | A5i | verdict for one fixed output varied 0-6 of 10 calls |
| sparse_lowest_eigenvectors_posdef | held-out | A0d | duplicate of sparse_lowest_eigenvalues_posdef |
| btsp | held-out | A0 (already), A3 | ESS 2.1 |

### Ranked candidates

Ranked by tier, then by number of risks, with no regard to stratum. "v6" = one of the seven v6 held-out tasks. No candidate ran in a pilot (pilot tasks cvar_projection, dst_type_II_scipy_fftpack, generalized_eigenvalues_real and optimal_advertising are barred from the held-out set; eigenvalues_real never ran). Speedups in the evidence column are published best / local best published solver in Harbor format. Every candidate showed 0 degenerate instances (A2, 1,000 instances, or 360-883 where the time cap hit) and ESS of at least 61 (A3).

**Tier 1: passes without conditions**

| # | Task | Family | Pool | Stratum and evidence | Risks |
|---|---|---|---|---|---|
| 1 | chebyshev_center | convex programs | held-out | M: CVXPY to HiGHS about 4.75x, presolve off and wrapper bypass 1.2-1.35x each, stacked in the best (7.07 / 9.26) | generator's paired slabs may give a closed form no published solver used |
| 2 | generalized_eigenvectors_complex | dense nonsymmetric generalized eigen | held-out | M: B^-1 A reduction 1.2-2.2x and float32 1.27-1.37x stack (3.36 / 6.21); not yet ablated | verifier rejects a float32 cast of the reference |
| 3 | ode_lorenz96_nonchaotic | ODE integration | ext | M: compiled RHS about 3x, compiled RK45 stepper about 3.3x more (11.29 / 8.69) | none found |
| 4 | robust_kalman_filter | convex programs | ext | A: IRLS + Riccati (20.9) and L-BFGS + adjoint (18.8) replace the same solve (local 10.5-22.7) | none found |
| 5 | graph_global_efficiency | graph algorithms | ext (50x) | M: compiled BFS about 15x, threading over sources about 2.7x (44.3 / 12.8) | 2 of 3 published solvers crashed locally (recursion, assertion) |
| 6 | pagerank | graph algorithms | ext (50x) | M: drop the networkx build 4-8x, compiled power iteration 2.5-4x (31.0 / 33.5) | stopping rule must match networkx exactly |
| 7 | minimum_spanning_tree | graph algorithms | ext | M: plain Kruskal about 10x, C conversion plus union-find about 2.7x (27.0 / 5.8) | verifier rejects float32 weights; the published 27x does not transfer |
| 8 | shortest_path_dijkstra | graph algorithms | v6 | M: output via `tolist` about 2.2x, threaded numba Dijkstra about 2.3x (4.95 / 3.99) | none found |
| 9 | l0_pruning | sort-based projection | ext | M on published code (partition 1.4-2.7x, ndarray output 7-13x), but the output component does not exist in Harbor (24.7 / 4.9) | conversion 14% of reference |
| 10 | quantile_regression | convex programs | held-out | A: three LP formulations, none combined (1.69 / 1.54) | low headroom |
| 11 | clustering_outliers | density-based clustering | v6 | A: DBSCAN or sklearn HDBSCAN replace the clustering call (2.53 / 2.79) | weak verifier (summary statistics, ARI at least 0.5) |
| 12 | generalized_eigenvectors_real | dense symmetric eigen | held-out | U: 10 solvers at 1.36-1.48 with one sygvd call, the best bundles several changes (4.96 / 2.72) | none found |
| 13 | eigenvalues_real | dense symmetric eigen | held-out | S: values only; `eigvalsh` gives 95% (2.52 / 1.94) | low headroom |
| 14 | qp | convex programs | held-out | S: OSQP called directly (1.89 / 1.91) | verifier rejects float32 cast 14/30; Harbor validates only the warmup call, so warm starts across timed repeats go unchecked |
| 15 | psd_cone_projection | dense symmetric eigen | ext | S: `eig` to `eigh` gives 100% (12.02 / 7.77) | `is_solution` 0.30 s per instance |
| 16 | unit_simplex_projection | sort-based projection | held-out | S: O(n) threshold search replaces the sort (6.50 / 2.79) | conversion 34% of reference |
| 17 | min_weight_assignment | graph algorithms | held-out | S on published code (dense `linear_sum_assignment`), but the one-change variant gave only 34% locally (1.71 / 1.78) | conversion 29%; 4.3 GiB per 100 problems |
| 18 | generalized_eigenvalues_complex | dense nonsymmetric generalized eigen | v6 | M on published code (2x2 factorial, 1.95x and 1.40x), S by override 2(b) (one change gives 80%) (5.59 / 10.41) | `is_solution` 1.48 s per instance; float32 gain hardware-dependent |
| 19 | lasso | iterative regression fitting | held-out | M on published code (float32 1.5-1.8x, lean CD 1.35-1.6x), S by override 2(b) (float32 alone matches the best local) (3.19 / 1.39) | 5.5 GiB per 100 problems; passes only by being as converged as sklearn |

**Tier 2: admitted only with the named fix**

| # | Task | Family | Pool | Stratum and evidence | Condition and risks |
|---|---|---|---|---|---|
| 20 | two_eigenvalues_around_0 | dense symmetric eigen | held-out | S: shift-invert Lanczos at 0, tuning adds 1.10-1.22x (3.45 / 2.28) | A5ii (1/400); conversion 27%; 5.2 GiB |
| 21 | robust_linear_program | convex programs | held-out | S: `ecos.solve` directly 6.4-7.0x, CVXPY variants about 1.06 (7.01 / 4.46) | A5ii (1/400 for two solvers, 1e-8 feasibility); float32 cast fails 23/30 |
| 22 | l1_pruning | sort-based projection | ext | M: vectorise the loop, drop the sort, ndarray output (17.69 / 9.15) | A5ii (1/400) |
| 23 | pde_heat1d | ODE integration | ext (50x) | M: compiled RHS 2-3.9x, compiled stepper about 9.7x (37.8 / 3.55) | A5ii (1/400 for two solvers); published 37.8 does not transfer |
| 24 | sparse_lowest_eigenvalues_posdef | sparse iterative eigensolvers | v6 | S: ARPACK tol; tol 5e-4 alone gives 87% (2.73 / 2.93) | A5ii (4/400); A2 generator patch (pass the rng to `sparse.random`) |
| 25 | least_squares | iterative regression fitting | v6 | U leaning S: closed-form polynomial fit; published routes give 1.01-1.09 locally (3.81 / 1.09) | A5ii (v6 shortcut 1.245%); ESS 59-62, lowest admitted |
| 26 | pca | SVD and low-rank | v6 | S: `eigh` of X^T X gives 88% (6.47 / 2.95) | A1: decide Harbor or upstream verifier (rotation accepted 20/20 vs 0/10); must pass A6iii, since v6 saw two timing modes for numpy-LAPACK solvers here; conversion 25% |
| 27 | sinkhorn | entropic optimal transport | v6 | A on published code, S by override 2(a): list conversion is 76-81% of reference time (2.25 / 1.16) | A5ii (1/400); 8.9 GiB; AlgoTune baseline spread 16% |

**Tier 3: eligible, weak**

| # | Task | Family | Pool | Stratum and evidence | Risks |
|---|---|---|---|---|---|
| 28 | stable_matching | combinatorial matching | ext | S by 2(a): converting the inputs takes 123% of reference time; the published 11-21x come from Cython solvers I did not compile; pure-Python published solvers run at 0.91 locally | Harbor headroom undemonstrated; 8.5 GiB |
| 29 | linear_system_solver | dense linear solve | ext | S by 2(a): conversion 58%; 16 published solvers at most 1.13; Gemini 3 Pro's 27.4x has no code explanation | headroom undemonstrated; 6.3 GiB |

Stratum counts: S 14 (13-21, 24, 26-29), M 10 (1-3, 5-9, 22, 23), A 3 (4, 10, 11), U 2 (12, 25). Families: convex programs 5, graph algorithms 5, dense symmetric eigen 4, sort-based projection 3, nonsymmetric generalized eigen 2, regression fitting 2, ODE integration 2, and one each for the other 6, so 13 families in 29 tasks.

### The v6 seven

Keep every v6 task that passes the same screen, label it a replication, and pre-register a "new tasks only" sensitivity analysis. Excluding them for what v6 showed would be outcome-dependent; screening them forward on measurement validity is not. I expect shortest_path_dijkstra, clustering_outliers and generalized_eigenvalues_complex to pass outright; sparse_lowest_eigenvalues_posdef, least_squares and sinkhorn pass with the validity pass (sparse_lowest also needs the generator patch); pca depends on A1 and A6iii. Porting the upstream pca verifier makes v7 pca a different task from v6 pca, so it is no longer a strict replication. Use a fresh hidden seed offset for v7, because v6's hidden instances have been used for diagnosis, and close the finalize-seed overlap (dev_eval seed 50,000 reaches finalize's instances).

### Draw rule

No draw is needed if 30 or fewer tasks pass A0-A6; run them all. If more pass, keep every passing v6 task, then fill to 30 from the remaining passing tasks with `random.Random(20261014)`, stratified by stratum in proportion to counts (largest-remainder rounding), each stratum's list sorted by name, drawn in the order S, M, A, U. No replacement after A6: a task failing a scorer check is gone, and the pool is not refilled.

## 3. Size, repeats, and scorer work before launch

Use as many tasks as pass, at 2 repeats. From v6's 14 groups, sd(log ratio) is 0.401 per group and 0.329 for 2-repeat task means (Codex), which splits into about 0.056 between tasks and 0.105 within. At a fixed number of groups G, the variance of the mean is sigma_task^2 / T + sigma_within^2 / G, so more tasks beats more repeats. Two repeats is the minimum that estimates within-task variance for the nested bootstrap.

| Tasks x repeats | Groups | Model spend at $0.653 per group | Power at +20% (sd as v6) |
|---|---|---|---|
| 20 x 3 | 60 | $39 | 0.73 |
| 26 x 2 | 52 | $34 | 0.78 |
| 29 x 2 | 58 | $38 | 0.82 |
| 30 x 2 | 60 | $39 | 0.83 |

Power is from a noncentral t over task means, two-sided 5%. It is optimistic if tasks within a family are correlated, and pessimistic if the harness fixes cut the sd (at sd 0.25, 26 tasks give 0.95). $0.653 is v6's measured spend per (team of 4 + 4 diversified independents) at the 80% submit gate ($4.554 + $4.591 over 14 groups). If v7 runs every agent to exhaustion of its $0.10 cap, a group costs at least $0.80, and $40 buys about 50 groups, so cap at 25 tasks x 2. Report bootstrap intervals by task (pre-registered) and, as a sensitivity, by family, since 29 tasks span 13 families.

Before launch, in order:

1. Locally, before any scorer time, port upstream verifiers or exclude (randomized_svd, pca); patch the sparse_lowest generator; rerun A2-A5 on the patched tasks; run the one-change ablation on every M task (rule 2(b)); compile and run the Cython published solvers for stable_matching; extend `data/algotune_split.json` (the harness loads only pilot and held-out tasks from it) and size agent boxes for the 5 tasks above 5 GiB.
2. On the scorer, run per task A6i (3 jobs), A6ii (3 jobs), A6iv (1 dev job), plus 3 more for any task that trips A6iii. That is about 7-10 jobs per task. v6 score jobs on the c7a.24xlarge took a median of 200-610 s (generalized_eigenvalues_complex 405, pca 606, sinkhorn 279, most others about 310), so about 40-70 slot-minutes per task. For 29 tasks that is 20-34 slot-hours, 2-3 hours of wall time on an 11-slot c7a.24xlarge. At $4.93 per hour, which I scaled linearly from PLAN.md's c7a.8xlarge price without looking it up, that is about $10-17.
3. On the scorer, once, time a 1,000-instance validity pass for the costliest candidates (generalized_eigenvalues_complex, psd_cone_projection) to fix the selection budget. Run `scripts/slot_interference.py` on the multi-slot scorer, which has never been done (HARNESS.md). If v7 adopts reference-alone scoring as primary (Fable's proposal), run A6 under both scores.

## 4. Is AlgoTune a good testbed for team vs best-of-N?

For the question as posed it is a weak one, for three reasons.

- **The outcome caps what a team can show.** The score is one selected solver, and the selector already takes the best of the independents' pool. A team wins only by combining components that different members found, or by refining a found trick further than any single agent does. Both second opinions found the v6 teams did neither.
- **The tasks rarely need combination.** By published code plus local ablation, about half of the eligible tasks are S, where one change sets the score, so a team at best transfers a switch that best-of-N also finds once in N tries. Only about 10 tasks have stacking components.
- **Harbor adds artefacts.** Its list inputs add a component shared across tasks, since fast list parsing dominates sinkhorn, stable_matching and linear_system_solver and is 25-34% of the reference in four more. Its verifiers predate upstream fixes in 27 of 154 tasks. Timing artefacts include the pca modes and the thread state left by a solver.

AlgoTune's strengths remain. Progress is verifiable mid-run, the score is continuous, runs are cheap enough for 50+ groups, and the published per-model code gives a model-free route map for every task. The M stratum is the place to look for a team effect, but 10 tasks cannot estimate it precisely.

A better test of collaboration inside this harness is **AlgoTune bundles**. One `solver.py` must solve K unrelated AlgoTune tasks (say K = 4, one from each of four families), and the score is the geometric mean of the K speedups. Sub-gains compose by construction, a team can divide the bundle, and each independent must cover all K on the same per-agent budget. Report the per-subtask merge of the independents' best solvers as a labelled oracle, since that is exactly the combination a team can perform and best-of-N cannot. The same generators, verifiers and scorer apply. The cost is that the design builds in a division-of-labour advantage, so it answers "do agents divide and merge work when it pays" rather than "does communication help on natural tasks". The external option with naturally composable sub-gains is ProgramBench (fraction of hidden tests passed; features compose), already phase 2 in TASKS.md and $2-25+ per task.

## Could not determine

- Scorer-side noise, two-mode timing and dev-vs-hidden agreement for any task not run in v6 (A6 needs the scorer).
- Whether the float32 components matter on the x86 scorer for tasks other than generalized_eigenvalues_complex (the Mac shows no float32 gain there at 1 thread).
- Cython published solvers (76 solver directories ship a setup.py) and the torch-based generalized_eigenvalues_complex solver were not run; quantile_regression and robust_linear_program ran after installing ecos.
- Local validity used library versions newer than or different from the task image, which installs unpinned latest packages at build time.
- Rule 2(b) for the 10 M tasks not yet ablated, so their stratum is provisional.
- Why Gemini 3 Pro's linear_system_solver scored 27.4x and pde_heat1d's published 37.8x falls to 3.6x locally.
- The c7a.24xlarge price (derived, not looked up). Tasks with best published speedup in (50, 100] (not screened).

## Follow-ups noticed (pre-existing, not fixed)

1. sparse_lowest_eigenvalues_posdef and _eigenvectors_ build the sparsity pattern from numpy's global RNG, so v6's dev, stage-2 and hidden "seeds" did not fix those instances, and every scoring used a different set. The v6 audit's 3% agreement between hidden scorings suggests the effect on v6 was small.
2. The Harbor snapshot predates upstream fixes. randomized_svd (in v6's eligible pool, not drawn) accepts a no-computation answer. pca accepts rotated subspaces, so v6 pca scores are on a looser verifier than upstream AlgoTune. optimal_advertising's generator yields a NaN instance with probability 0.45 per 100-instance set, which may explain the all-1.0 v6 pilot record Fable flagged. I did not check the hidden offset to confirm it.
3. Harbor's verifier checks validity only on each instance's warmup call. Published qp solvers (Gemini 2.5 Pro, GPT-5.2, GPT-5.4) warm-start across the timed repeats, which is rewarded and never re-validated.
4. sparse_lowest_eigenvectors_posdef returns and checks eigenvalues; it is a duplicate task, not an eigenvector task.

## Files

All in `/private/tmp/claude-501/-Users-jack-projects-pa/cb767837-0539-44e7-8d8b-4bf3c9abe4f2/scratchpad/`.

- Route classifications from published code: `routes-linalg.md`, `routes-lowrank.md`, `routes-convex.md`, `routes-graph.md`, `routes-misc.md`.
- Local results: `screen_summary.json`, `screen_out/`, `gen_out/`, `pub_out/`, `probes.json`, `ablate.json`, `gendet.json`, `harbor_info.json` (Harbor vs upstream diffs), `alltasks.json` (published stats).
- Scripts: `screen.py`, `gencheck.py`, `runpub.py`, `probes.py`, `upcheck.py`, `convfloor.py`, `ablate.py`, `memcheck.py`, `gendet.py`.
- Inputs: `harbor_cache/` (all 154 Harbor tasks), `ext/algotune/` (upstream repo), `recovered/` (the four v6 reports).
