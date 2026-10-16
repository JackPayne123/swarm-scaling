# One-change variants, fixed before any timing (2026-10-16)

Source of each variant is the "Single change to test" paragraph of the route report named per task (scratchpad `routes-*.md`). Each variant changes the Harbor reference `solve` in one place; alternatives are tested separately and never combined. Denominator = best `local_speedup` among published solvers with 0 exceptions in `pub_out/<task>.json` (the same rule the 7 earlier rows used: sparse_lowest_eigenvalues_posdef's 2.93 is Qwen3 Coder with 4/400 invalid and 0 exceptions).

| Task | Variant id | Change (reading chosen) | Denominator solver (pub_out) |
|---|---|---|---|
| chebyshev_center | C1 | CVXPY block replaced by `scipy.optimize.linprog(c, A_ub=[a, norm(a_i)], b_ub=b, method="highs-ds", options={"presolve": False})`, bounds x free and r >= 0 (the report elides `bounds=...`; r >= 0 is the docstring's LP and cannot bind here because b - a x_pseudo > 0) | Gemini 3.1 Pro Preview 9.26 |
| chebyshev_center | C2 | The same LP and the same options (dual simplex, presolve off) through highspy `Highs.passModel(HighsLp)`, a fresh Highs object per call. Reading: "the same LP" in the report means the linprog call just named, with only the interface changed | same |
| chebyshev_center | C3 | Closed form: if `np.allclose(a[m:], -a[:m])`, x = `lstsq(A, (b1 - b2)/2)`, else fall back to the reference LP | same |
| generalized_eigenvectors_complex | G1 | `la.eig(A_scaled, B_scaled)` replaced by `np.linalg.eig(np.linalg.solve(B, A))` on the unscaled A, B exactly as the report writes it; reference post-processing (Python normalisation loop, zip sort, `list(vec)`) kept | R1 6.21 |
| generalized_eigenvectors_complex | G2 | Reference QZ call kept unchanged; post-processing replaced by GPT-5.2's (vectorised column norms, `lexsort`, `.T.tolist()`). GPT-5.2's `check_finite=False` and overwrite flags are not part of the post-processing and are not used | same |
| ode_lorenz96_nonchaotic | O1 | Numba RHS (explicit wraparound, Gemini 3 Pro's form) inside the same `solve_ivp` RK45 call, rtol = atol = 1e-8 | Gemini 3.1 Pro Preview 8.69 |
| graph_global_efficiency | E1 | networkx replaced by Opus 4.6's kernel: `fromiter(chain)` CSR and a numba `prange` BFS over sources summing 1/d (serial here, NUMBA_NUM_THREADS=1) | Gemini 3.1 Pro Preview 12.81 |
| pagerank | P1 | DiGraph build and `nx.pagerank` replaced by a `fromiter(chain)` CSR and a numba push power iteration with nx's update (x0 = 1/N, uniform dangling term, stop when L1 change < N * tol, no renormalisation) | GPT-5.4 33.49 |
| pagerank | P2 | Non-compiled: the same `fromiter(chain)` CSR as a scipy `csr_array` of row-normalised weights and nx's own numpy loop (`x = alpha * (x @ A + sum(x[dangling]) / N) + (1 - alpha) / N`). Reading: the CSR is built the same way as in P1, so only the compiled loop differs | same |
| minimum_spanning_tree | T1 | `np.asarray(edges)`, `argsort(a[:, 2], kind="stable")`, numba union-find over that order stopping at n - 1 edges, final sort of chosen edges by (u, v) | GPT-5.4 5.83 |
| minimum_spanning_tree | T2 | scipy `minimum_spanning_tree` on a symmetric CSR from `np.asarray(edges)` (Opus 4.5's code, the evidence the report cites) | same |
| minimum_spanning_tree | T3 | Plain Python Kruskal: `sorted` by weight, union-find with early exit (Opus 4.1's code) | same |
| shortest_path_dijkstra | D1 | Output comprehension replaced by `dist_matrix.tolist()` (inf kept; the verifier maps None to inf) | Gemini 3.1 Pro Preview 3.99 |
| l0_pruning | L0a | Mergesort argsort replaced by `np.argpartition(np.abs(v), n - k)[n - k:]`; list output kept | GPT-5.4 4.90 |
| l0_pruning | L0b | Return the ndarray instead of `.tolist()`; argsort kept. The report also asks for both together, which rule 2(b) excludes, so that is not run | same |
| l1_pruning | L1a | Return the ndarray instead of `.tolist()` | o4-mini 9.15 (1/400 invalid, 0 exceptions) |
| l1_pruning | L1b | Python loop vectorised: `cumsum`, thresholds `1/(j+1) * (cumsum - z)`, first index where `mu < threshold` via `argmax` (Opus 4.5's rule); sort and list output kept | same |
| l1_pruning | L1c | Sort plus loop replaced by o4-mini's numba Michelot elimination computing theta; the rest of the reference (soft threshold, sign, `.tolist()`) kept | same |
| pde_heat1d | H1 | `np.pad` replaced by a preallocated zero-padded buffer (`buf[1:-1] = u`), RHS expression otherwise unchanged | GPT-5.4 3.55 (1/400 invalid, 0 exceptions) |
| pde_heat1d | H2 | Numba RHS inside the same `solve_ivp` call (Opus 4.5's form) | same |
| randomized_svd | R1 | `n_iter=0` in the reference `randomized_svd` call; validity against the patched verifier (tol capped at 0.9). float32 is the report's second change and is not run | best of the top 3 published, rerun against the patched verifier (400 instances) |

Not tested, because the report names them only as additions on top of another change: the compiled RK45 steppers for ode_lorenz96_nonchaotic and pde_heat1d (GPT-5.4's pde solver is itself the denominator), float32 on top of the generalized eigen reduction, float32 on top of `n_iter=0`.
