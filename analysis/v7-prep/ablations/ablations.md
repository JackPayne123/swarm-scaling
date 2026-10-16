# Rule 2(b) one-change ablations for the provisional M stratum

Run 2026-10-16 on the Mac, model-free, with the method of scratchpad `ablate.py`: 1 thread (OMP, OPENBLAS, MKL, NUMEXPR and NUMBA set to 1), Harbor input format, 25 instances on seeds 3,000,000+i at each task's PROBLEM_SIZE, warmup plus min of 2 for the reference and for each arm, speedup = sum of reference minima / sum of arm minima, validity = `Task.is_solution`. Variants were fixed in `preregistered-variants.md` before any timing. Share = log(variant speedup) / log(best published local speedup in `pub_out`, 0 exceptions). Stratum is S when a variant reaches 75% and is valid 25/25, otherwise the route class (M for all 11).

All variants valid 25/25, no exceptions. Same-run = the denominator solver timed in this harness alongside the variants.

| Task | Change to the reference | Local speedup | Best published solver, local (same run) | Share of best log speedup | Stratum |
|---|---|---|---|---|---|
| chebyshev_center | C1 CVXPY to `linprog(highs-ds, presolve off)` | 13.15 | 9.26, Gemini 3.1 Pro Preview (14.82) | 116% | S |
| | C2 same LP through highspy `passModel` | 13.56 | | 117% | |
| | C3 slab closed form (`lstsq`), LP fallback | 79.68 | | 197% | |
| generalized_eigenvectors_complex | G1 QZ to `eig(solve(B, A))`, reference post-processing | 4.57 | 6.21, R1 (9.23) | 83% | S |
| | G2 QZ kept, GPT-5.2's vectorised post-processing | 1.02 | | 1% | |
| ode_lorenz96_nonchaotic | O1 numba RHS inside `solve_ivp` | 4.12 | 8.69, Gemini 3.1 Pro Preview (10.69) | 66% | M |
| graph_global_efficiency | E1 networkx to `fromiter` CSR + numba BFS summing 1/d | 17.88 | 12.80, Gemini 3.1 Pro Preview (19.17) | 113% | S |
| pagerank | P1 networkx to `fromiter` CSR + numba power iteration | 32.63 | 33.49, GPT-5.4 (47.17) | 99% | S |
| | P2 same CSR, scipy matvec + networkx's numpy loop | 40.52 | | 105% | |
| minimum_spanning_tree | T1 `asarray` + stable argsort + numba union-find | 8.01 | 5.83, GPT-5.4 (7.46) | 118% | S |
| | T2 scipy `minimum_spanning_tree` | 5.06 | | 92% | |
| | T3 plain Python Kruskal | 10.60 | | 134% | |
| shortest_path_dijkstra | D1 output via `dist_matrix.tolist()` | 2.33 | 3.99, Gemini 3.1 Pro Preview (5.77) | 61% | M |
| l0_pruning | L0a mergesort `argsort` to `argpartition` | 3.35 | 4.90, GPT-5.4 (4.91) | 76% | S |
| | L0b ndarray output | 1.12 | | 7% | |
| l1_pruning | L1a ndarray output | 1.00 | 9.15, o4-mini (9.57) | 0% | M |
| | L1b Python loop vectorised | 1.57 | | 20% | |
| | L1c sort + loop to o4-mini's numba Michelot | 4.84 | | 71% | |
| pde_heat1d | H1 `np.pad` to preallocated buffer | 2.22 | 3.55, GPT-5.4 (3.00) | 63% | S |
| | H2 numba RHS inside `solve_ivp` | 3.80 | | 105% | |
| randomized_svd | R1 `n_iter=0` (patched verifier, tol capped at 0.9) | 2.20 | 4.63, Gemini 3.1 Pro Preview (4.89) | 51% | M |

Result: 7 of the 11 move to S (chebyshev_center, generalized_eigenvectors_complex, graph_global_efficiency, pagerank, minimum_spanning_tree, l0_pruning, pde_heat1d). 4 stay M (ode_lorenz96_nonchaotic, shortest_path_dijkstra, l1_pruning, randomized_svd).

## Notes

- Denominators follow the earlier table's rule, best `local_speedup` among solvers with 0 exceptions regardless of invalid count. Two of them failed rarely in `pub_out` (o4-mini on l1_pruning and GPT-5.4 on pde_heat1d, 1/400 each); both were 25/25 here.
- randomized_svd denominator. `runpub.py` against the patched copy (`randomized_svd_patched/`, 400 instances, top 3 from `pub_plan.tsv`) gave Gemini 3.1 Pro Preview 4.63, Gemini 3 Pro Preview 4.06 and GPT-5.2 3.96, all 0/400 invalid (unpatched `pub_out`: 4.61, 4.24, 3.81). The cap rejects none of them, and R1 (`n_iter=0`) is valid 25/25 under it. randomized_svd is still excluded by A4 in v7-tasks.md; this row assumes the A1 port readmits it.
- Machine load was 29 to 55 on 18 cores. Reference calls ran at 0.50 to 0.84 CPU seconds per wall second, the fast arms at 0.70 to 0.99, so wall-time speedups in this run are inflated against an idle machine. Same-run published speedups were 0.85x to 1.60x of `pub_out` (chebyshev_center 14.82 vs 9.26, pagerank 47.17 vs 33.49). CPU-time speedups (process time of the same calls) of the published solvers came out at 0.87x to 1.08x of `pub_out`, so they track the denominator better. Every verdict is the same when the variants' CPU-time speedups are used (G1 79%, L0a 78%, C2 89%, H2 94%, P1 100%, E1 104%, T1 107%; O1 62%, D1 58%, L1c 69%, R1 41%). Both checks are in `results.json`; the rule uses wall time and `pub_out` as specified.
- generalized_eigenvectors_complex is the one verdict that depends on the check. Against `pub_out` G1 reaches 83% (86% in a replicate run), with CPU time 79%, but against the same-run published speedup (R1, 9.23) it reaches only 68% (72% in the replicate).
- l0_pruning passes narrowly, 76% in the primary run and 76% in a replicate (CPU time 78%).
- Replicate. l0_pruning, generalized_eigenvectors_complex and l1_pruning (primary shares within 0.05 of 75%) were rerun once with the same seeds (`replicate/`), giving L0a 3.34 (76%), G1 4.76 (86%), L1c 4.54 (68%). Verdicts unchanged. The table uses the primary run.
- Readings chosen. C1 bounds x free and r >= 0 as an ndarray. C2 reads "the same LP" as the same solver options as C1 (dual simplex, presolve off) with a fresh Highs object per call, so Gemini 3.1's cached model is not part of it. G1 solves the unscaled `solve(B, A)` as the report writes it; the reference's scaling lines stay but are unused. P2 builds the CSR with `fromiter(chain)` like P1, so only the compiled loop differs. That reading gives 40.5 (30.1 in CPU time), not the report's expected 4 to 6x, which came from Opus 4's per-edge Python list building. P1 passes either way. L1c keeps o4-mini's `fastmath=True` and the reference's soft-threshold, sign and `.tolist()`. T2 and T3 are the report's "cheaper intermediate steps", tested as alternatives.
- E1 ran Opus 4.6's kernel with `range` in place of `prange`. Under numba 0.68 its `parallel=True` build raises `AssertionError: unexpected cycle in lookup()`, the same error that kept Opus 4.6's solver from loading in `runpub.py`. At NUMBA_NUM_THREADS=1 the two are the same computation.
- Not run, because the reports name them only on top of another change or rule 2(b) forbids combining: the compiled RK45 steppers (ode, pde), float32 on top of G1 or R1, and l0's "both together".
- No invalid variant. L1c computes the textbook L1-ball theta rather than the reference's first-break theta and still passed 25/25, as the route report predicted for k above about 100.
- Surprises. On minimum_spanning_tree a plain Python Kruskal (T3, 10.60) beats every published solver in the same run (GPT-5.4 7.46), because in Harbor format the list-to-array conversion costs more than it saves. On chebyshev_center C3 (80x wall, 40x CPU) uses the generator's paired slabs, which no published solver did; it is valid 25/25 and does not decide the verdict, since C1 and C2 already pass. On pde_heat1d a numba RHS (3.80) beats the denominator solver's compiled stepper in the same run (3.00). G2 settles the route report's 1.1x versus 2.1x question at 1.02 (1.08 in the replicate).
- Accelerate BLAS ignores the OMP, OPENBLAS and MKL variables. A 1500 x 1500 matmul used 0.73 to 0.81 CPU seconds per wall second with or without VECLIB_MAXIMUM_THREADS=1, so BLAS did not multithread. The environment matches `ablate.py` and `runpub.py`.
- Not determined. The load during the original `pub_out` runs is unknown, so I cannot say whether the denominator and these variants were inflated equally. Scorer hardware (x86 BLAS, 8 threads, numba threads uncapped) was not measured, and the float32 and multithreading components that matter there are absent from these ablations.
- Run log. The first full run crashed on pagerank, l1_pruning and pde_heat1d while loading `cache=True` published solvers from a numba cache my smoke test had written ("No module named '<dynamic>'"). They were rerun with a fresh NUMBA_CACHE_DIR per task (`run_all.sh`), and the JSONs are from the reruns.

## Follow-ups (not fixed)

- Opus 4.6's graph_global_efficiency solver does not compile under numba 0.68 (`parallel=True` kernel). If the scorer image has numba 0.68, solvers of this shape fail there too.
- A numba `cache=True` function in a module loaded with `spec_from_file_location` cannot be reloaded from its own cache. Any harness that loads solvers by path and reuses a NUMBA_CACHE_DIR across processes will crash on the second load.
- chebyshev_center has a closed form through the generator's slab structure (C3, 25/25 valid, about 80x). v7-tasks.md lists this as a risk; it is now measured.

## Files

`preregistered-variants.md` (variant list written before timing), `ablate_m.py` (harness and variants), `run_all.sh`, `summarize.py`, `results.json` (per task: variants with speedup, validity, share, CPU-time and same-run checks, denominator and its solver, same-run published speedup), `<task>.json` (per-instance times), `randomized_svd_patched/` and `pub_randomized_svd_patched.json` (patched verifier and its denominator rerun), `replicate/`, `logs/`.
