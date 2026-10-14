# AlgoTune optimisation routes for seven graph and geometry tasks

Sources are the published final solvers under `ext/algotune/results/<model>/<task>/` (including their `.pyx` and `setup.py` helpers), the published speedups in `ext/algotune/reports/agent_summary.json`, and the harbor task code (`tests/evaluator.py`, `tests/test_outputs.py`). No code was run, so every timing statement below is either a published speedup or a derivation from code, and derivations are labelled as such.

## Summary

| Task | Class | Reason |
|---|---|---|
| shortest_path_dijkstra | M | Every solver gains about 2.2x by replacing the reference's per-element `np.isinf` output loop with `tolist()`; the best (Gemini 3.1, 4.95) stacks a multithreaded numba Dijkstra on top (about 2.3x more). |
| min_weight_assignment | S | Every solver at 1.2x or more swaps `min_weight_full_bipartite_matching` for dense `linear_sum_assignment` with inf for missing entries (1.24-1.71); no other component shows a measurable effect. |
| minimum_spanning_tree | M | Dropping networkx for a plain Kruskal on the input order gives about 10x; doing the conversion, stable sort and union-find in C after one `np.asarray(edges)` gives about 2.7x more (GPT-5.4 27.0, Gemini 3 Pro 25.8). |
| pagerank | M | Dropping the networkx graph build gives 4-8x with Python-level conversion; bulk conversion plus a compiled power iteration gives another 2.5-4x (Gemini 2.5 Pro 31.0, Gemini 3.1 29.1). The stopping rule must match networkx exactly. |
| graph_global_efficiency | M | A compiled BFS that sums 1/d on the fly gives about 15x; multithreading over sources gives about 2.7x more (Opus 4.6 44.3, Gemini 3.1 44.0, GPT-5.4 41.2). |
| stable_matching | S | The reference's cost is the n^2 receiver rank table built in Python; filling it in Cython straight from the Python lists gives 11-21x, while every route that first converts to numpy stays at 1.2-1.7x. |
| voronoi_diagram | U (scored results are exploit-dominated) | The top four (8-32x) never compute a Voronoi diagram and pass because the verifier checks only counts and shapes; the legitimate 2.3-4.2x results contain an unexplained 2.3x gap between two near-identical solvers. |

## Reading choices and general caveats

- A solver counts as "valid with speedup at least 1.1" when its `final_speedup` in `agent_summary.json` is numeric and at least 1.1. `claude-sonnet-4-5-20250929` has speedups but no code folder, so it is excluded. Solvers below 1.1 and N/A solvers are cited where they serve as natural experiments.
- "Within 1.5x of best" uses the best published valid speedup for the task, including exploits. For voronoi I report both the exploit and the legitimate view.
- Published speedups for functionally identical or near-identical code differ by up to about 1.5x (pagerank GPT-5.4 20.28 vs Gemini 2.5 Pro 30.97; global efficiency GPT-5.2 10.87 vs GPT-5 16.37; shortest path scipy-plus-`tolist` solvers 2.12-2.50), and in one voronoi pair by 2.3x. Natural-experiment ratios below about 1.5x cannot be separated from this spread.
- Some N/A entries are not caused by the code. Opus 4's minimum_spanning_tree solver is functionally identical to Opus 4.1's (9.72) yet is N/A.
- The published numbers came from AlgoTune's own harness, and I assume it used the same problem sizes as the harbor `PROBLEM_SIZE`. For min_weight_assignment the GPT-5.2 result is predicted to within 0.01 from n = 767 (see that section), which supports the assumption there. For the other tasks I could not check it.
- The harbor harness calls `solve` 11 times on the same problem object and checks validity only on the warmup call, so input mutation and per-problem answer caching would be exploitable. Of the solvers read here, GLM-4.5's MST solver mutates its input, and none caches answers.
- Several top solvers are Cython extensions with a `setup.py`, so local reproduction needs the build step. Gemini 3 Pro's shortest-path and global-efficiency extensions use `prange`, but their `setup.py` never passes `-fopenmp` to the Extension, so they probably ran single-threaded.

## shortest_path_dijkstra

#### Reference and verifier

The reference runs `scipy.sparse.csgraph.shortest_path(method="D", directed=False)` on an n = 319 CSR graph with integer weights 1-100, then builds the output with a per-element list comprehension that calls `np.isinf` on every entry (about 100k numpy scalar calls). The generator draws density 0.2, but `graph.minimum(graph.T)` keeps an entry only where both directions were drawn, so the stored density should be about 0.04 and the average degree about 13 (derived from the code, not measured). The verifier requires a Python list of n lists of length n, a zero diagonal, symmetry and non-negative values, then compares with a recomputed reference using `np.allclose` (rtol 1e-5, atol 1e-8). It accepts `inf` in place of `None`. Nothing is random.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro Preview | 4.95 | scipy replaced by a numba all-pairs Dijkstra with an indexed binary heap; multithreading (`prange` over sources); fastmath; `tolist()` keeping inf |
| GPT-5.4 | 3.29 | different algorithm: Cython Floyd-Warshall updating the upper triangle and mirroring; float32 (exact here because weights are small integers); `tolist()` |
| Gemini 3 Pro Preview | 2.54 | Cython C++ priority-queue Dijkstra over G and G^T; output list built in Cython; probably serial (see caveats) |
| Claude Opus 4.5 | 2.50 | scipy `dijkstra`; `tolist()`, using an object array only if inf is present; its dense Floyd-Warshall branch needs density above 0.1 and should not trigger |
| GPT-5.2 | 2.49 | scipy `shortest_path` D; `tolist()` |
| Gemini 2.5 Pro | 2.44 | `connected_components` then scipy `auto` (which resolves to Dijkstra); object-array `None` conversion; the graph-tool path is probably unavailable |
| R1 | 2.33 | numba O(n^2) linear-scan Dijkstra per source on a ThreadPoolExecutor (nogil); `np.where` to `None` plus `tolist()` |
| Qwen3 Coder | 2.26 | scipy Floyd-Warshall; object-array `None` conversion |
| o4-mini | 2.18 | scipy D; returns a lazy `list` subclass, so `tolist()` runs inside the verifier rather than in the timed call |
| Claude Opus 4.6 | 2.18 | scipy D; `tolist()` plus a Python pass replacing inf |
| GPT-5-mini | 2.15 | scipy `dijkstra`; object-array `None` conversion |
| GPT-5 | 2.15 | scipy `dijkstra`; `tolist()` keeping inf |
| GPT-5 Pro (medium) | 2.12 | scipy `dijkstra`; `tolist()` keeping inf |
| GPT-OSS-120b | 2.09 | scipy `auto`; `np.where` to `None` plus `tolist()` |
| Claude Opus 4.1 | 1.97 | scipy `dijkstra`; `tolist()` plus a Python inf pass |
| GLM-4.5 | 1.96 | scipy D; NaN placeholder plus a Python pass |

Claude Opus 4 (1.00) is the reference copied verbatim.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| Opus 4 (1.00) to GPT-5 Pro (2.12) / GPT-5 (2.15) | 2.1x | output conversion only, from the per-element `np.isinf` comprehension to `ndarray.tolist()` |
| GPT-5 Pro (2.12) to o4-mini (2.18) | 1.03x | moving `tolist()` itself out of the timed call, so `tolist()` is a small cost |
| GLM (1.96) / Opus 4.1 (1.97) vs GPT-5 Pro (2.12) | 1.08x | a Python pass over n^2 entries to write `None` |
| Qwen FW (2.26) vs Opus 4.6 Dijkstra (2.18) | about 1.0x | Floyd-Warshall vs Dijkstra inside scipy |
| GPT-5 Pro (2.12) to Gemini 3.1 (4.95) | 2.3x | compute kernel replaced by a multithreaded numba Dijkstra |
| GPT-5.2 (2.49) to GPT-5.4 (3.29) | 1.3x | compute kernel replaced by a serial Cython float32 Floyd-Warshall |
| Gemini 3 Pro (2.54) vs scipy group | about 1.0x | a serial compiled Dijkstra gains nothing over scipy's |

#### Class M

The output-conversion fix (about 2.1-2.2x, present in every solver) and a faster all-pairs kernel (multithreaded numba Dijkstra about 2.3x, or float32 Floyd-Warshall about 1.3x) act on different parts of the reference and multiply. The only solver within 1.5x of the best, Gemini 3.1 (4.95), combines the conversion fix with the parallel kernel. GPT-5.4 (3.29) sits at the boundary and combines the conversion fix with the float32 Floyd-Warshall. Parallelism is what makes the kernel faster, because a serial compiled Dijkstra (Gemini 3 Pro, R1) does no better than scipy.

#### Validity

- GPT-5.4's float32 is exact for this generator, because path lengths are small integers far below 2^24. It would be approximate for non-integer weights.
- Gemini 3.1 uses `fastmath=True` while comparing against an inf-initialised distance matrix. LLVM's no-infs flag permits miscompiling such comparisons in principle. It passed the published validation.
- o4-mini's lazy list moves the conversion out of the timed call, which is a timing exploit, though here it is worth only about 1.03x.
- Returning inf instead of `None` is valid, because the verifier maps `None` to inf and `allclose` treats inf as equal to inf.

#### Single change to test

Replace the output comprehension with `dist_matrix.tolist()`. Expected about 2.1-2.2x. Reaching the best needs a multithreaded compiled Dijkstra as well.

## min_weight_assignment

#### Reference and verifier

The cost matrix holds squared Euclidean distances between two sets of n = 767 integer points in [1, 999]^2. It is passed as CSR lists, and since CSR drops zeros, coincident points become missing entries. The reference calls `min_weight_full_bipartite_matching` (LAPJVsp). The verifier requires `row_ind` and `col_ind` to be permutations, sums the proposed cost via `mat[i, j]` (0 for a missing entry), and compares it with the recomputed optimum using `isclose` (rtol 1e-5). Any optimal assignment passes. The expected number of missing entries per instance is 767^2 / 999^2 ≈ 0.59 (derived).

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| GPT-OSS-120b | 1.71 | sparse matcher replaced by dense `linear_sum_assignment` (LSA); inf fill by a per-row Python slice loop |
| R1 | 1.70 | dense LSA; numba fill; float32 matrix with a finite big-M for missing entries |
| Claude Opus 4.1 | 1.56 | dense LSA; numba fill with inf |
| GPT-5 Pro (medium) | 1.54 | dense LSA; direct reshape when nnz = n^2, otherwise inf fill |
| GPT-5 | 1.53 | dense LSA; vectorised flat-index scatter with inf |
| GLM-4.5 | 1.53 | dense LSA; vectorised fancy-index fill with inf |
| GPT-5-mini | 1.31 | dense LSA via a `tocoo` fill; Python loops mapping back and checking the permutation |
| GPT-5.2 | 1.24 | dense LSA only when nnz = n^2, otherwise the reference matcher; returns numpy arrays |
| Claude Opus 4.5 | 1.24 | dense LSA; numba fill with a finite 1e15 sentinel |

Below 1.1, Gemini 2.5 Pro (0.99), o4-mini (0.99), Qwen (1.00), Gemini 3 Pro (1.03), Gemini 3.1 (1.02) and GPT-5.4 (0.98) all keep the sparse matcher. Opus 4 (1.01) has a dense path gated on n ≤ 500, which is never taken. Opus 4.6 (N/A) uses dense LSA with a 1e18 sentinel.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| reference-equivalent (0.99-1.03) vs GLM / GPT-5 (1.53) | about 1.5x | sparse matcher to dense LSA |
| GPT-OSS slice loop (1.71) vs Opus 4.1 numba (1.56) vs GLM vectorised (1.53) | within spread | how the dense matrix is filled |
| R1 float32 big-M (1.70) vs Opus 4.1 float64 inf (1.56) | within spread | precision and sentinel |
| GPT-5.2 (1.24) | consistent with the switch | applies dense LSA only on instances with no missing entry |
| Opus 4.5 1e15 (1.24) vs Opus 4.1 inf (1.56) | 1.26x, unexplained | sentinel value; could be measurement spread |

The GPT-5.2 case can be predicted. The fraction of instances with no missing entry is exp(-0.59) ≈ 0.55. Applying 1.55x to 55% of instances and 1.0x to the rest gives 1 / (0.55/1.55 + 0.45) ≈ 1.25, against a published 1.24. This supports the dense switch being the whole effect, and published n ≈ 767.

#### Class S

All nine solvers at 1.2x or more owe their gain to one library swap, sparse LAPJVsp to dense LSA. The fill method, the precision and the output type show no effect beyond the spread.

#### Validity

- The inf-filled dense LSA is exact.
- R1's big-M, (max - min)(n + 1) + max + 1, exceeds any feasible total, so it is exact.
- Opus 4.6's 1e18 sentinel is N/A. A plausible cause is precision loss in the dual updates, since the float64 spacing at 1e18 is 128, but I did not confirm it.
- A plain `toarray()` (missing entries as 0) would be invalid on roughly 45% of instances, because the zero-cost non-edge lets LSA undercut the reference optimum. No valid solver relies on it.

#### Single change to test

Build `dense = np.full((n, n), np.inf)`, set `dense[np.repeat(np.arange(n), np.diff(indptr)), indices] = data`, and replace the matcher call with `scipy.optimize.linear_sum_assignment(dense)`. Expected about 1.5x, which is the best published.

## minimum_spanning_tree

#### Reference and verifier

The input is n = 575 nodes with about 49.5k edges (p = 0.3) in lexicographic (u < v) order. Weights are rounded to 3 decimals in [0.1, 10], so ties are common. The reference builds an `nx.Graph` edge by edge, runs `nx.minimum_spanning_edges` (Kruskal with a stable sort over the `G.edges` order), normalises each edge to u < v and sorts by (u, v). The verifier recomputes the reference and requires exact list equality, so tie-breaking must match networkx's, which is a stable weight sort over the generator's input order. Nothing is random.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 27.03 | networkx dropped; one `np.asarray(edges)`; numba hash dedup reproducing the nx edge order; numpy stable argsort; numba union-find with early exit at n - 1 edges |
| Gemini 3 Pro Preview | 25.84 | networkx dropped; one `np.array(edges)`; Cython adjacency rebuild for the nx order; numpy stable argsort; Cython union-find with early exit |
| Claude Opus 4.5 | 15.15 | networkx replaced by `scipy.sparse.csgraph.minimum_spanning_tree` on a symmetric CSR built from `np.asarray(edges)`; vectorised output sort |
| Claude Opus 4.6 | 13.53 | same as Opus 4.5, with a Python output loop |
| GLM-4.5 | 12.78 | networkx dropped; Python Kruskal on the input list; sorts `problem["edges"]` in place (input mutation); early exit |
| GPT-OSS-120b | 10.07 | Python Kruskal on a `sorted()` copy; early exit |
| o4-mini | 9.90 | numba Kruskal fed by three `np.fromiter` generator passes; stable argsort; early exit |
| Qwen3 Coder | 9.77 | Python Kruskal with recursive find; early exit |
| Claude Opus 4.1 | 9.72 | Python Kruskal; early exit |
| R1 | 9.09 | scipy MST on an upper-triangle COO built from three list comprehensions |
| Gemini 3.1 Pro Preview | 8.30 | Cython Kruskal reading each edge through Python object indexing; `qsort` with an index tiebreak; early exit |
| GPT-5 | 5.71 | Python Kruskal after rebuilding the nx edge order with per-node dicts (handles duplicates); numpy argsort |
| GPT-5.2 | 3.61 | numba Kruskal and numpy argsort, but per-edge Python dict dedup and per-element numpy writes |
| GPT-5-mini | 2.93 | Python Kruskal after a per-edge dict adjacency rebuild with `int()` and `float()` |
| GPT-5 Pro (medium) | 1.30 | Python Kruskal with tuple-keyed dicts and no early exit |

Opus 4 and Gemini 2.5 Pro are N/A.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| reference vs Opus 4.1 / Qwen / GPT-OSS (9.7-10.1) | about 10x | networkx graph build and nx Kruskal replaced by a Python sort plus union-find with early exit |
| Opus 4.1 (9.72) vs GPT-5-mini (2.93) | 3.3x slower | same Kruskal, plus a per-edge dict rebuild of the nx order |
| Opus 4.1 (9.72) to GLM (12.78) | 1.3x | in-place sort and `itemgetter`; part of this comes from timed repeat calls seeing pre-sorted input |
| R1 (9.09) to Opus 4.5 (15.15) | 1.7x | both scipy MST; three list comprehensions vs one `np.asarray` |
| o4-mini (9.90) to GPT-5.4 (27.03) | 2.7x | both numba Kruskal with stable argsort; three generator `fromiter` passes vs one `np.asarray` |
| GPT-5.2 (3.61) to GPT-5.4 (27.03) | 7.5x | both numba Kruskal; per-edge Python dedup vs numba dedup after `np.asarray` |
| Opus 4.5 (15.15) to GPT-5.4 (27.03) | 1.8x | scipy MST vs numba Kruskal after the same `np.asarray` |

#### Class M

Two changes act on different costs and stack. The first, dropping the networkx object graph and nx's Python union-find for a sort of the input plus union-find with early exit, is worth about 10x. The second, doing every per-edge step in C after a single `np.asarray(edges)` (stable numpy argsort and a compiled union-find), is worth about 2.7x on top. Both solvers within 1.5x of the best (GPT-5.4 27.0, Gemini 3 Pro 25.8) combine the two. scipy's MST, called after `np.asarray`, replaces the same computation as the second change and reaches about half the best (13.5-15.2). The natural experiments show that any per-edge Python work (dicts, `int()`, per-element numpy writes) cancels most of the gain.

#### Validity

- All valid fast solvers rely on the generator emitting edges in lexicographic order without duplicates, so a stable weight sort of the input reproduces networkx's tie order. This holds on every generated instance. GPT-5, GPT-5.2, GPT-5.4 and Gemini 3 Pro also handle arbitrary order and duplicates.
- The scipy route is exact only because scipy's MST breaks ties in CSR row-major order. That is the same order here, and three scipy solvers passed the published validation.
- GLM-4.5 sorts `problem["edges"]` in place. The reference's answer does not change, because equal weights keep lexicographic order, but timed repeat calls get pre-sorted input for both the solver and the baseline. Gemini 2.5 Pro does the same and is N/A for a reason I could not find.
- GPT-OSS outputs float node ids, which pass because `0.0 == 0` in list equality.

#### Single change to test

Replace the reference body with `a = np.asarray(edges)`, `order = np.argsort(a[:, 2], kind="stable")`, a numba union-find over `order` that stops at n - 1 edges, and a final sort of the chosen edges by (u, v). Expected about 25x. Cheaper intermediate steps are scipy MST (about 14x) or plain Python Kruskal (about 10x).

## pagerank

#### Reference and verifier

The graph has n = 4329 nodes with average out-degree 8 (about 35k edges). The reference builds an `nx.DiGraph` and calls `nx.pagerank` (alpha 0.85, tol 1e-6, max_iter 100). In networkx 3.x that is a scipy power iteration starting from 1/N, with dangling mass spread uniformly, stopping when the L1 change falls below N * tol ≈ 4.3e-3, and without renormalisation. The verifier requires n non-negative finite floats summing to 1 (rtol 1e-5), then `np.allclose` against the recomputed reference (rtol 1e-5, atol 1e-8). The reference stops far from the fixed point. By my derivation the per-node distance is around 1e-6 to 1e-5, against a per-node tolerance around 1e-8 (not measured). A solver therefore has to reproduce the same iterate, with the same start, update and stopping rule, and converging further fails.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| Gemini 2.5 Pro | 30.97 | networkx dropped; CSR via `np.fromiter(chain.from_iterable(adj))`; numba push power iteration with nx's exact rule; fastmath |
| Gemini 3.1 Pro Preview | 29.11 | networkx dropped; Cython power iteration reading the Python adjacency lists directly on every iteration (no conversion) |
| GPT-5.4 | 20.28 | networkx dropped; `fromiter(chain)` CSR; numba push iteration |
| GPT-5.2 | 13.03 | networkx dropped; CSR via per-row slice assignment; numba transpose and pull iteration; final renormalisation |
| GPT-5 | 8.37 | networkx dropped; Python edge loop with dedup into lists; numpy `bincount` iteration |
| Claude Opus 4.5 | 6.13 | networkx dropped; Python per-edge writes into numpy arrays; scipy CSR matvec iteration |
| Claude Opus 4 | 4.22 | networkx dropped; Python edge lists; scipy CSC matvec; final renormalisation |
| GPT-5 Pro (medium) | 4.22 | networkx dropped; Python dedup and per-row `fromiter`; numpy `bincount` iteration recomputing weights each iteration |

Below 1.1, GLM (1.08, `from_dict_of_lists`), Gemini 3 Pro (1.05), R1 (1.04, `add_edges_from`), o4-mini (1.01) and GPT-5-mini (0.99) all keep networkx. Opus 4.1, Qwen and GPT-OSS are N/A, and each uses a tighter stopping rule (L1 < tol, or tol 1e-8 without the N factor). Opus 4.6 is N/A even though it uses the N * tol rule.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| networkx kept, graph built differently (GLM 1.08, R1 1.04) | at most 1.08x | graph-construction tweaks within networkx |
| reference vs Opus 4 / GPT-5 Pro (4.22), Opus 4.5 (6.13), GPT-5 (8.37) | 4-8x | networkx removed, with Python-level conversion and a numpy or scipy iteration |
| GPT-5 (8.37) to GPT-5.4 (20.28) | 2.4x | bulk `fromiter` conversion plus a compiled iteration |
| GPT-5.2 (13.03) to GPT-5.4 (20.28) | 1.56x | per-row slice CSR plus pull vs `fromiter(chain)` plus push, both numba |
| GPT-5.4 (20.28) vs Gemini 2.5 Pro (30.97) | 1.53x | near-identical code; Gemini adds fastmath; treat as spread unless measured |
| Opus 4.1, Qwen, GPT-OSS | invalid | stopping later than networkx fails the verifier |

#### Class M

Two changes stack. Removing the networkx graph construction and dict output is worth 4-8x with plain Python conversion. Bulk conversion (`fromiter` over `chain`, or Cython reading the lists directly) plus a compiled power iteration is worth about 2.5-4x on top. The two solvers within 1.5x of the best (Gemini 2.5 Pro 31.0, Gemini 3.1 29.1) combine both.

#### Validity

- Every valid fast solver replicates nx's iteration exactly. GPT-5.2's and Opus 4's final renormalisation changes values by about 1e-16 relative, because the iteration already preserves the sum.
- Differences in summation order could change the stopping iteration only when the L1 change lands within rounding of N * tol, which should be rare (not measured).
- I could not find why Opus 4.6 is N/A.
- No published solver uses an approximation, because the verifier tolerance is tighter than the reference's own convergence error.

#### Single change to test

Replace the DiGraph build and `nx.pagerank` with a `fromiter(chain)` CSR and a numba loop implementing nx's exact update: x0 = 1/N, a uniform dangling term, stopping when L1 < N * tol, and no renormalisation. Expected 20-30x. A non-compiled version that builds a scipy CSR from the lists and runs the same loop in numpy should give about 4-6x (Opus 4 evidence).

## graph_global_efficiency

#### Reference and verifier

The graph has n = 448 nodes with average degree 8. The reference builds an `nx.Graph` and calls `nx.global_efficiency`, which runs an all-pairs BFS in Python and sums 1/d over ordered pairs divided by n(n - 1). The verifier compares the scalar with a recomputed reference using `math.isclose` (rtol 1e-5, atol 1e-8). Nothing is random.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| Claude Opus 4.6 | 44.33 | networkx dropped; `fromiter(chain)` CSR; numba BFS per source adding 1/d level by level; multithreading (`prange` over sources) |
| Gemini 3.1 Pro Preview | 44.02 | same; fastmath; per-level count times 1/d |
| GPT-5.4 | 41.20 | same; per-thread marker arrays; precomputed 1/d table; parallel path taken because n ≥ 384 |
| Gemini 3 Pro Preview | 21.30 | Cython BFS per source with a malloc per BFS; probably serial (see caveats) |
| Gemini 2.5 Pro | 16.61 | serial numba BFS summing inside the BFS; per-row slice CSR build |
| GPT-5 | 16.37 | serial numba layered BFS; marker array; reciprocal table |
| Claude Opus 4.5 | 16.11 | serial numba BFS; per-element Python CSR build |
| Claude Opus 4.1 | 15.85 | serial numba BFS; a separate O(n) pass summing 1/d per source |
| R1 | 15.65 | serial numba BFS; separate pass |
| Claude Opus 4 | 14.19 | serial numba BFS over a padded 2D adjacency |
| GPT-5.2 | 10.87 | serial numba BFS; `fromiter` CSR; fastmath |
| Qwen3 Coder | 2.60 | scipy `shortest_path` (unweighted) dense APSP plus a numpy reciprocal sum |
| GLM-4.5 | 2.31 | scipy `shortest_path` unweighted plus a numba upper-triangle sum |
| GPT-5-mini | 2.30 | scipy `shortest_path` unweighted plus a numpy mask and sum |

o4-mini (1.07) and GPT-5 Pro (1.00) are pure-Python BFS. GPT-OSS is N/A; its visited-token array is `np.empty`, so it starts uninitialised, which can produce wrong results.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| reference vs pure-Python BFS without networkx (o4-mini 1.07, GPT-5 Pro 1.00) | at most 1.07x | leaving networkx without compiling |
| scipy dense APSP (2.3-2.6) vs serial numba BFS (14-17) | about 6x | summing inside a compiled BFS instead of building an n x n matrix with scipy |
| Opus 4.5 (16.11) to Opus 4.6 (44.33) | 2.75x | multithreading over sources |
| Gemini 2.5 Pro (16.61) to Gemini 3.1 (44.02) | 2.65x | multithreading over sources |
| GPT-5 on-the-fly sum (16.37) vs Opus 4.1 / R1 separate pass (15.7-15.9) | about 1.0x | where the 1/d sum is taken |
| GPT-5.2 (10.87) vs GPT-5 (16.37) | 1.5x | nearly the same kernel; spread |

#### Class M

A compiled BFS that accumulates 1/d (about 15x over the reference) and multithreading over source nodes (about 2.7x more) stack. All three solvers within 1.5x of the best combine both.

#### Validity

- The computation is exact. A parallel reduction changes the sum by about 1e-16 relative, far inside rtol 1e-5.
- The multithreaded gain depends on the core count of the timing machine. numba uses `NUMBA_NUM_THREADS`, which defaults to all cores. The harbor harness caps OMP, OPENBLAS, MKL and NUMEXPR at 8 threads but not numba.
- No published solver samples sources, and with rtol 1e-5 on a single scalar sampling would fail.

#### Single change to test

Replace networkx with Opus 4.6's kernel, a numba `prange` BFS over sources that sums 1/d inside the BFS. Expected about 40x on a multi-core machine and about 15x single-threaded.

## stable_matching

#### Reference and verifier

There are n = 1151 proposers and receivers, with preferences given as dicts of Python lists. The reference builds the receiver rank table as a Python list of lists, which takes n^2 ≈ 1.3M Python assignments with a double index each. It then runs proposer-proposing Gale-Shapley with a FIFO `list.pop(0)` queue, making about n ln n ≈ 8k proposals (derived). Proposer preferences are read lazily, only for proposals actually made. The verifier checks that the output is a permutation with no blocking pair. It does not check proposer-optimality, so any stable matching passes. No solver uses that gap.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro Preview | 20.56 | Cython Gale-Shapley; receiver rank table filled in C straight from the Python lists via `PyList_GET_ITEM`; proposer preferences read lazily as in the reference; int16 rank table; LIFO stack |
| Gemini 3 Pro Preview | 11.91 | Cython Gale-Shapley; both preference tables copied into int32 memoryviews by typed-list loops; LIFO |
| Claude Opus 4.6 | 10.80 | Cython Gale-Shapley; both tables copied into malloc'd int arrays by typed-list loops; LIFO |
| o4-mini | 1.73 | `np.array` of both preference sets plus numba Gale-Shapley (rank table built in numba) |
| GPT-5 Pro (medium) | 1.67 | `np.asarray` of both plus numba Gale-Shapley |
| GPT-5 | 1.64 | `np.asarray` of both plus Cython Gale-Shapley (int16 ranks, chained proposals) |
| Qwen3 Coder | 1.64 | `np.array` of both (int16) plus numba Gale-Shapley |
| Gemini 2.5 Pro | 1.58 | `np.fromiter(chain)` for both (int16); numpy fancy-index rank table; numba Gale-Shapley |
| GLM-4.5 | 1.55 | pure Python; `array.array` rank rows with the row lookup hoisted; deque |
| R1 | 1.54 | `np.array` of both plus numba Gale-Shapley |
| Claude Opus 4 | 1.49 | `np.array` of both plus numba Gale-Shapley |
| Claude Opus 4.1 | 1.30 | `np.array` of both; numpy fancy-index rank table; Python Gale-Shapley loop over numpy scalars |
| Claude Opus 4.5 | 1.28 | `np.array` of both plus numba Gale-Shapley |
| GPT-5.2 | 1.19 | `np.asarray` of both; numpy per-row rank table; numba Gale-Shapley |
| GPT-5.4 | 1.12 | pure Python; stack instead of `pop(0)`; local bindings |

GPT-5-mini (0.56) is a Cython version that indexes untyped `object` preference lists (or falls back to a Python flattening of both tables) and ends up slower than the reference.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| GPT-5 (1.64) to Opus 4.6 (10.80) | 6.6x | the same C Gale-Shapley; inputs converted with `np.asarray` vs read from the lists in a typed Cython loop, so conversion is the cost and the matching loop is not |
| Gemini 2.5 Pro `fromiter` (1.58) vs o4-mini `np.array` (1.73) | about 1.0x | the numpy conversion route makes no difference |
| Opus 4.6 (10.80) to Gemini 3.1 (20.56) | 1.9x | Opus 4.6 also copies the whole proposer table (another n^2 reads) that the reference never touches; Gemini 3.1 also uses int16 and the raw list macro |
| reference vs GPT-5.4 (1.12) / GLM (1.55) | 1.1-1.55x | pure-Python tweaks (stack, hoisted row lookup) |

#### Class S

The reference spends its time in one place, the n^2 Python loop that builds the receiver rank table. The only solver within 1.5x of the best (Gemini 3.1, 20.56) moves that loop, together with the matching loop, into Cython that reads the list items directly, and keeps the reference's lazy reads of proposer preferences. The 1.9x gap to Opus 4.6 and Gemini 3 Pro is not a second component over the reference. It is a cost those two solvers added by copying the proposer table, which the reference never does. The int16 table may add a little, but no published pair isolates it. Every route through numpy conversion stays at 1.2-1.7x, so this change cannot be made with a numpy library call. It needs compiled code that reads Python lists.

#### Validity

- All solvers produce the proposer-optimal matching, which is the same matching whether a stack or a queue is used, so they are exact.
- Gemini 3.1's int16 rank table is safe for n ≤ 32767.
- Gemini 2.5 Pro reads `dict.values()` in insertion order, which is 0..n-1 for this generator.
- None depends on the verifier's missing proposer-optimality check.

#### Single change to test

Replace the rank-table build and the matching loop with a Cython function, as in Gemini 3.1. It should fill a C rank array from `receiver_prefs[r][k]` through typed list access, read `proposer_prefs[p][next[p]]` lazily, and use a LIFO stack. Expected about 15-20x. A numba or numpy version will not reproduce this, as the 1.5-1.7x numpy group shows.

## voronoi_diagram

#### Reference and verifier

At n = 12288 the generator mostly draws uniform or clustered points (6k-25k of them), sometimes noisy grids or circles, and rarely small special cases. The reference calls `scipy.spatial.Voronoi(points)`, converts the vertices and ridge points with `tolist`, copies every region list, and reorders the regions by `point_region`. The verifier checks key presence, array shapes, that the vertices are finite, and index ranges. It then compares only two numbers with a recomputed reference, the vertex count and `len(point_region)`. It checks no coordinates, no region contents, and neither the contents nor the count of the ridges.

#### Solvers

| Model | Speedup | Components |
|---|---|---|
| GPT-5.2 | 32.05 | exploit: no Voronoi; vertex count 2n - 2 - h from a numba monotone-chain hull on lexsorted points (falls back to scipy if any triple is exactly collinear); zero-filled broadcast vertices, empty regions and ridges |
| GPT-5 | 12.62 | exploit: same count formula with a pure-Python hull over Python-sorted tuples; dummy outputs |
| o4-mini | 9.28 | exploit: scipy `Delaunay` plus vectorised circumcentres as vertices (count = number of triangles); placeholder regions and ridges |
| GPT-5.4 | 8.16 | exploit: `Delaunay` plus circumcentres, rounded and deduplicated to count merged vertices; cached zero arrays |
| Gemini 3 Pro Preview | 4.21 | legitimate: scipy Voronoi; numpy arrays returned; Cython region reorder and `ridge_vertices` conversion |
| Gemini 2.5 Pro | 3.35 | legitimate: qhull options "Qbb Qz" (Qc dropped), with "Qbb Qc" as a fallback on exception; `tolist` conversions kept |
| Gemini 3.1 Pro Preview | 2.58 | legitimate: private `scipy.spatial._qhull._Qhull(b"v")` and `get_voronoi_diagram`; raw outputs |
| Qwen3 Coder | 2.49 | legitimate: Voronoi; raw outputs, no reorder |
| Claude Opus 4.5 | 2.48 | legitimate: `_Qhull` direct; regions reordered; raw arrays |
| GPT-5 Pro (medium) | 2.47 | legitimate: Voronoi; raw outputs |
| Claude Opus 4.6 | 2.42 | legitimate: Voronoi; raw arrays; regions reordered |
| GPT-OSS-120b | 2.41 | legitimate: Voronoi; raw outputs |
| Claude Opus 4 | 2.27 | legitimate: Voronoi plus the same `tolist` and list-copy work as the reference, plus copies of `ridge_vertices` |
| GLM-4.5 | 1.17 | legitimate: float32 round trip; options "Qbb Qc Qz Qx Qt"; `tolist`; dead code paths |

Below 1.1, Opus 4.1 (0.99) runs Voronoi with `tolist` and region reindexing, R1 (1.09) uses "Q0 Qbb Qc" with `tolist`, and GPT-5-mini (0.82) adds extra conversions.

#### Natural experiments

| Pair | Ratio | Component isolated |
|---|---|---|
| Opus 4.1 (0.99) vs Opus 4 (2.27) | 2.3x, unexplained | near-identical work (Voronoi plus `tolist` plus region copies) |
| GPT-5 (12.62) to GPT-5.2 (32.05) | 2.5x | same exploit; Python hull vs numba hull on `lexsort` |
| Delaunay count (8.2-9.3) vs hull-formula count (12.6-32.1) | 1.4-3.9x | avoiding Qhull entirely |
| Opus 4.1 (0.99) vs Gemini 2.5 Pro (3.35) | 3.4x | both keep `tolist`; Gemini drops Qc; confounded by the Opus 4 anomaly |

#### Class U for legitimate routes; the scored top is exploit-dominated

The four best published results (GPT-5.2 32.05, GPT-5 12.62, o4-mini 9.28, GPT-5.4 8.16) answer only the vertex count, which the verifier is limited to. Within the exploit route, the best solver stacks the switch with a compiled hull (2.5x over the Python hull). For the legitimate routes the evidence does not support a class. A cluster of solvers that return raw arrays sits at 2.4-2.6, which would point to skipping the output conversion. But Opus 4 does the reference's full conversion work and scores 2.27 while Opus 4.1 scores 0.99. From the code I would expect the conversions to be a small fraction of a Qhull run on about 15k points (not measured).

Local timing on the harbor instances would decide it. The runs needed are (a) the reference, (b) the reference returning `vor.*` without `tolist` or list copies, (c) the reference with `qhull_options="Qbb Qz"`, (d) b and c together, and (e) `_Qhull` direct. If b and c each give at least 1.3x and stack, the class is M. If only one matters, it is S.

#### Validity

- GPT-5.2, GPT-5, o4-mini and GPT-5.4 do not produce a Voronoi diagram. Any verifier that compared vertices or ridges would reject them.
- The hull formula assumes general position, meaning no cocircular quadruples that Qhull would merge into one vertex. GPT-5 also counts collinear boundary points, and GPT-5.2 falls back to scipy only on exactly collinear triples.
- o4-mini's triangle count would exceed Qhull's merged vertex count on exactly cocircular input, such as a grid with no noise. The generator adds noise, so this should be rare.
- Whether these four pass all 100 harbor instances (seeds 0-99) needs a local check.
- Gemini 2.5 Pro's "Qbb Qz" without Qc is harmless for continuous random points. A duplicated point could get an invalid region index without raising, so the fallback would not catch it.

#### Single change to test

For a legitimate solver, return `vor.vertices`, `vor.regions`, `vor.point_region`, `vor.ridge_points` and `vor.ridge_vertices` unconverted. The published cluster suggests about 2.4x, but the Opus 4 anomaly leaves that unverified. Also test `qhull_options="Qbb Qz"`. For the exploit, which is not a legitimate change, the hull-count dummy output reaches 32x.

## Follow-ups noticed, not acted on

- Opus 4's MST solver is functionally identical to Opus 4.1's (9.72) but is published N/A, so published N/A flags are not reliable evidence of invalid code. I could not find the cause of N/A for Opus 4.6 on pagerank or Gemini 2.5 Pro on MST.
- The voronoi verifier in the harbor `evaluator.py` checks only counts and shapes, and it is exploitable as shown.
- The harbor harness validates only the warmup call and reuses the same problem object for the 10 timed calls, so input mutation (GLM-4.5 MST) and answer caching are possible in principle.
