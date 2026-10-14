# AlgoTune optimisation routes: unit_simplex_projection, l0_pruning, l1_pruning, sinkhorn, clustering_outliers, ode_lorenz96_nonchaotic, pde_heat1d

Sources used: published solver.py files under ext/algotune/results, speedups from reports/agent_summary.json, task code from harbor_cache tests/evaluator.py and test_outputs.py, and the AlgoTune harness source in ext/algotune/AlgoTuner (read only to explain list-vs-array effects). Nothing was executed. "Valid >= 1.1" solvers were all read in full; solvers below 1.1 were read where they serve as a natural experiment. Speedups are quoted to two decimals.

## Harness facts that change what a component is worth

| Fact | Evidence | Consequence for local tests |
|---|---|---|
| Published AlgoTune hands large numeric lists to solve() as ndarrays | `externalize_large_arrays` converts any numeric list over 100 KB to .npy (AlgoTuner/utils/serialization.py ~L196-210); `dataset_decoder` loads it with `np.load(mmap_mode="r")` (~L381-416) | Harbor's test_outputs.py passes the generator's Python lists, so the reference and every solver pay a list-to-array conversion that published runs did not. For simplex (786k), l0 (655k), l1 (393k) and sinkhorn (1535x1535 nested lists) this is a fixed cost on both sides that will compress ratios. Needs measuring. |
| Published timing includes `deep_materialize_fast(result)` inside the timed region | AlgoTuner/utils/isolated_benchmark.py ~L1003-1007, L1163-1174, L2294-2299; it recurses into lists with one Python call per element and passes ndarrays through `np.asarray` | A list output costs far more in the published numbers than `.tolist()` alone. GPT-OSS-120b's simplex solver is the reference plus a final `.tolist()` and scored 0.29x. Harbor times `solve()` only, so "return ndarray instead of list" will be worth much less locally. |
| Harbor checks `is_solution` only on the warmup output of each instance | test_outputs.py `_measure_interleaved_runtime` | Nondeterministic or buffer-reusing solvers are checked once per instance. |
| Problem sizes differ | reports/generation.json vs PROBLEM_SIZE | simplex 982,958 vs 786,432; l0 695,029 vs 655,360; l1 473,085 vs 393,216; sinkhorn 1813 vs 1535; clustering 2457 vs 2303; lorenz96 7856 vs 6144; heat1d 8 vs 8. |
| Published numbers for near-identical code differ by up to ~1.7x | l0 Gemini 2.5 Pro 2.48 vs Gemini 3.1 Pro 1.45 (same np.array, argpartition, scatter, tolist); lorenz96 numba RHS solvers 1.78 to 3.39 | Ratios under about 1.5x between two models are within run-to-run or harness-version noise. I could not tell from the files which harness version or machine each model batch used. |
| Harbor installs unpinned numpy (2.x) | environment/Dockerfile | `np.array(list, copy=False)` raises ValueError under NumPy 2 when a copy is needed. Gemini 3.1 Pro's l1 solver and GPT-5 Pro's sinkhorn solver use it and will fail on Harbor's list inputs. |

## unit_simplex_projection

**Reference and verifier.** The reference sorts y descending, takes the last index rho where u_j > (cumsum_j - 1)/j, sets theta = (cumsum_rho - 1)/(rho+1) and returns max(y - theta, 0) as an ndarray (the standard, correct Duchi formula). The verifier recomputes the reference and requires `np.allclose(..., atol=1e-6)` with the default rtol 1e-5. Inputs are y ~ randn(n), deterministic per seed. Nothing beyond closeness is checked.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.2 | 6.50 | different algorithm: sort-free active set (Michelot/Condat) with in-place index compaction; numba; in-place output |
| o4-mini | 3.61 | different algorithm: 25-step bisection on theta; numba fastmath; approximation that relies on verifier tolerance |
| R1 | 3.53 | different algorithm: repeated max-selection until the simplex condition fails, O(n*rho); numba |
| GPT-5 | 2.70 | different algorithm: three-way partition selection for theta, expected O(n); numba |
| GLM-4.5 | 2.69 | argpartition top 3% then sort only those (relies on rho < 0.03n) |
| GPT-5 Pro (medium) | 2.25 | different algorithm: Michelot active set in vectorised numpy |
| Gemini 3 Pro Preview | 1.11 | sort kept (ascending, no reverse); post-sort scan in numba with early break |
| Gemini 3.1 Pro Preview | 1.11 | same as Gemini 3 Pro |

**Natural experiments.**
- Sort kept, cumsum/where replaced by a numba early-exit scan (Gemini 3 Pro and 3.1 Pro vs reference) gives 1.11x, so the post-sort work is about 10% of the reference.
- The same Michelot algorithm in numpy masks (GPT-5 Pro 2.25) vs numba with in-place compaction (GPT-5.2 6.50) differs by 2.9x.
- Sort-based Cython (GPT-5.4 1.09, Claude Opus 4.6 1.07) and `-np.sort(-y)` (Claude Opus 4.5 1.02) gain nothing, so sort implementation tweaks are worthless.
- Reference plus `.tolist()` (GPT-OSS-120b 0.29) shows the published harness penalty for list output.

**Class: S.** Every solver above 1.2x gets its gain from one change, replacing the O(n log n) sort with an O(n) threshold search. Bisection, selection, partial partition and active-set searches all replace the same sort, so they are alternatives and do not stack, and nothing else in the computation is worth more than ~1.1x. My reading: I count "numpy vs compiled in-place implementation of the threshold search" as part of the same switch because it acts on the same step; under a stricter reading that counts compilation as its own component this becomes A. Only GPT-5.2 is within 1.5x of the best, so the top of the distribution rests on one solver.

**Validity.**
- o4-mini's 25 halvings of [min - 1, max] (width ~11) leave a theta error of at most ~1.6e-7, inside atol 1e-6 on all instances, but only because of the tolerance.
- GLM-4.5 is correct only when the active set lies inside the top 3% of entries. For randn inputs rho is a handful of entries, so it is safe on this generator and wrong for other inputs.
- The rest are exact up to rounding. Gemini 2.5 Pro (randomised Duchi pivot in numba) is N/A and the reason is not visible in the code.

**Single change to test.** Replace the sort/cumsum/where block with GPT-5.2's active-set loop (theta = (sum - 1)/m over entries above theta, repeat until the count stops changing) compiled with numba. The numpy version of the same loop is a smaller edit and gave 2.25x. In Harbor the `np.array(list)` of 786k floats is a fixed cost on both sides and will cap the ratio.

## l0_pruning

**Reference and verifier.** The reference does `np.argsort(np.abs(v), kind="mergesort")`, keeps the last k indices, zeroes the rest and returns `pruned.tolist()`. The verifier requires allclose to the reference with atol 1e-6. Stable tie-breaking (keep the highest index among equal |v|) would matter only with ties, which have probability ~0 for randn inputs. k = random.randint(1, n/2) per seed.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 24.70 | argsort -> np.partition (values) + threshold mask; ndarray output (`v * keep`) |
| Gemini 3 Pro Preview | 21.87 | Cython std::nth_element on an |v| copy, in-place zeroing of the converted v; ndarray output |
| GPT-5.2 | 19.41 | argsort -> argpartition; reused preallocated buffers; ndarray output |
| Claude Opus 4 | 2.71 | argsort -> argpartition; tolist output |
| Gemini 2.5 Pro | 2.48 | argsort -> argpartition; tolist |
| Claude Opus 4.6 | 1.46 | argpartition, zero in place; tolist |
| Gemini 3.1 Pro Preview | 1.45 | argpartition; tolist |
| Claude Opus 4.5 | 1.45 | np.partition threshold + numba masking loop; tolist |
| Claude Opus 4.1 | 1.42 | numba quickselect; tolist |
| GLM-4.5 | 1.41 | argpartition inside numba; tolist |
| Qwen3 Coder | 1.39 | argpartition; tolist |
| GPT-5-mini | 1.38 | np.partition threshold + tie handling; tolist |
| R1 | 1.38 | argpartition inside numba; tolist |
| GPT-5 | 1.38 | argpartition + tie handling; tolist |
| GPT-5 Pro (medium) | 1.22 | np.partition threshold + tie handling; tolist |

**Natural experiments.**
- argsort -> (arg)partition with list output kept gives 1.38-1.46x in most runs and 2.48-2.71x for Claude Opus 4 and Gemini 2.5 Pro. Gemini 2.5 Pro and Gemini 3.1 Pro have near-identical code and differ by 1.7x, which sets the noise floor for this task.
- Same selection, list vs ndarray output: GPT-5.2 19.41 vs Gemini 3.1 Pro 1.45 is 13.4x; vs Claude Opus 4 2.71 it is 7.2x.
- Value partition + mask vs argpartition + scatter, both returning ndarrays (GPT-5.4 24.70 vs GPT-5.2 19.41), differ by 1.27x, within noise.
- Compiled selection (R1, GLM-4.5, Claude Opus 4.1, all 1.38-1.42) adds nothing over numpy's partition.
- No published solver keeps argsort and returns an ndarray, so "drop tolist alone" is not observed. Modelling the reference as sort cost S plus list cost L and fitting the two observed ratios gives L between 0.5 and 1.8 times S, so dropping tolist alone would give roughly 1.5-2.8x in the published harness. This is inference, not observation.

**Class: M.** Two components act on different parts (index selection and output conversion), each is worth at least ~1.4x alone, and they multiply. All three solvers above 19x combine both.

**Validity.** All are exact for continuous inputs; several ignore tie-breaking, which is measure-zero here. GPT-5.2 returns a reused buffer that the next call overwrites (harmless in both harnesses, since each result is checked before the next call). Gemini 3 Pro mutates the array it converted itself, which is safe because `np.array` copies. o4-mini is N/A because its Cython code permutes the |v| buffer with nth_element and then indexes it as if unpermuted.

**Single change to test.** Two lines: `np.argpartition(np.abs(v), n - k)[n - k:]` in place of the mergesort argsort, and return the ndarray instead of `.tolist()`. Test each alone and both together. In Harbor the list cost is only `.tolist()` (no per-element materialisation) and the input conversion is paid on both sides, so expect the output component to shrink and the partition component to carry more of the gain.

## l1_pruning

**Reference and verifier.** The reference sorts |v| descending (mergesort), runs a Python for-loop that accumulates the cumulative sum and breaks at the first j where mu_j < (cumsum_j - k)/(j+1), uses that value as theta, and returns sign(v) * max(|v| - theta, 0) as a list. This first-break theta is not the textbook L1-ball theta. My derivation: it equals (rho*theta* + mu_{rho+1})/(rho+1), below the true theta* by (theta* - mu_{rho+1})/(rho+1). The verifier requires allclose with atol 1e-6 and default rtol 1e-5. k = random.randint(1, n/2).

Exact-projection solvers therefore differ from the reference by that gap on every nonzero entry. My estimate is 1e-10 to 1e-8 for typical k, reaching ~1e-6 only when k is below roughly 100 at these n (about 0.05% of draws, about a 5% chance that at least one of 100 instances is affected). This needs checking against the actual Harbor k values.

| Model | Speedup | Components |
|---|---|---|
| o4-mini | 17.69 | different algorithm: Michelot active set, no sort; numba; ndarray output; computes theta* rather than the reference's first-break theta |
| GPT-5.2 | 6.79 | partition + sort only the top m (doubling m); numba first-break scan replicating the reference rule; ndarray output; in-place copysign |
| GPT-5.4 | 5.62 | full sort kept; numba first-break scan fused with soft-threshold; ndarray output; Cython list path (used only for list input) |
| Gemini 3.1 Pro Preview | 4.58 | sort inside numba; numba first-break scan + soft-threshold; ndarray output |
| Gemini 3 Pro Preview | 1.97 | different algorithm: Michelot/Newton in Cython with prange (4 threads); tolist |
| R1 | 1.85 | different algorithm: bisection on theta with 1e-6 stopping; numba; tolist; approximation that relies on verifier tolerance |
| Gemini 2.5 Pro | 1.79 | Michelot/Newton in numba; tolist |
| GLM-4.5 | 1.73 | bisection, 50 halvings, numba; tolist |
| GPT-5 | 1.68 | Michelot/Newton iterations in vectorised numpy with sort fallback; tolist |
| Qwen3 Coder | 1.47 | sort kept; numba first-break loop; tolist |
| GPT-5 Pro (medium) | 1.47 | sort kept; binary search for rho (textbook rule); tolist |
| GPT-OSS-120b | 1.44 | sort kept; vectorised textbook rule; tolist |
| GPT-5-mini | 1.43 | sort kept; vectorised textbook rule; tolist |
| Claude Opus 4.5 | 1.39 | sort kept; vectorised first-break rule (argmax of mask); tolist |
| Claude Opus 4 | 1.39 | heapsort; vectorised first-break rule; tolist |
| Claude Opus 4.1 | 1.29 | sort and textbook scan in numba; tolist |

**Natural experiments.**
- Vectorise or compile the Python loop only (sort and tolist kept): 1.29-1.47x across seven solvers.
- Also remove the sort (tolist kept): 1.68-1.97x, about 1.25x over the previous group.
- Keep the sort, return an ndarray: 4.58-6.79x, about 3.9x over the first group (GPT-5.4 5.62 vs median 1.43).
- Remove the sort and return an ndarray: o4-mini 17.69, which is 3.1x over GPT-5.4 and 9.9x over Gemini 2.5 Pro (the same Michelot-in-numba algorithm with tolist).
- Partial vs full sort with ndarray output (GPT-5.2 6.79 vs GPT-5.4 5.62) is 1.2x. That is expected because rho is typically 30-90% of n, so GPT-5.2's doubling ends close to a full sort.

**Class: M.** Three components act on different parts (the Python threshold loop, the sort, the list output). Each is worth at least ~1.25x alone in the observed pairs, and the best solver (o4-mini) stacks all three.

**Validity.**
- o4-mini, Gemini 2.5 Pro, Gemini 3 Pro, GPT-5, R1, GLM-4.5, GPT-5 Pro, GPT-OSS-120b, GPT-5-mini and Claude Opus 4.1 compute the textbook projection, not the reference's first-break theta. They pass only because the gap stays under atol except for very small k (see above).
- R1's bisection stops when the bracket is narrower than 1e-6, so its theta error can approach atol; borderline.
- GPT-5.2, GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.5, Claude Opus 4 and Qwen3 Coder replicate the reference's rule and are exact up to rounding.
- Gemini 3.1 Pro calls `np.array(..., copy=False)`, which raises on list input under NumPy 2 (Harbor).

**Single change to test.** No single change reproduces 17.7x in the published numbers. In order of published size, try returning an ndarray, vectorising the loop (the cheapest exact change, ~1.4x), and replacing sort plus loop with o4-mini's numba Michelot kernel. Locally, measure each alone, because the output component is inflated in published numbers and the input conversion is new.

## sinkhorn

**Reference and verifier.** The reference converts a, b, M and calls POT `ot.sinkhorn` (sinkhorn_knopp: K = exp(-M/reg), Kp = K/a, up to 1000 iterations, stopThr 1e-9 on the L2 norm of the column-marginal error, checked every 10 iterations with an `einsum` over n^2). It returns G as an ndarray. Instances are uniform a and b, M = Euclidean distances between uniform points in the unit square, reg = 1, deterministic per seed. The verifier requires allclose to `ot.sinkhorn` with rtol 1e-6 and atol 1e-8. The mean entry of G is 1/n^2, about 3.0e-7 at n = 1813 and 4.2e-7 at Harbor's n = 1535, so atol lets a typical entry deviate by about 3.3% (published) or 2.4% (Harbor). That slack accepts a single Sinkhorn update.

| Model | Speedup | Components |
|---|---|---|
| Qwen3 Coder | 2.25 | own loop (numba-wrapped matvecs); at most 2 iterations (max_iter=2, tol 0.1) |
| Gemini 2.5 Pro | 2.23 | own numpy loop (CuPy path absent); 1 iteration, because its check norm(u*Kv - a) is zero by construction |
| R1 | 1.86 | own numpy loop; 1 iteration (check max|u*(K@v) - a| is zero by construction); extra matvec per iteration |
| GLM-4.5 | 1.84 | POT with numItermax=6, stopThr=1.0, so it stops at the first check (1 iteration) |
| Gemini 3.1 Pro Preview | 1.77 | in-place exp on the converted M; BLAS dot with out=; 1 iteration (check KTu*v - b is zero by construction) |
| GPT-5.2 | 1.76 | own loop with in-place buffers, no Kp and no einsum; genuine check every 10 iterations (L1 of row marginal <= 1e-9); returns a reused buffer |
| Gemini 3 Pro Preview | 1.66 | scipy BLAS gemv; 1 iteration (unconditional `break` inside the first convergence check) |
| o4-mini | 1.62 | own loop with matmul out=; genuine check every iteration, looser tolerance (max abs change of u < 1e-7) |
| Claude Opus 4.5 | 1.37 | own loop; genuine check every iteration (max abs change of u < 1e-9); numba plan formation |

Below 1.1: GPT-5 1.06 and GPT-5.4 1.04 (both call POT at this size), GPT-5 Pro 1.01 (POT); GPT-OSS-120b 1.01 and GPT-5-mini 1.00 not read.

**Natural experiments.**
- POT with iterations cut to one (GLM-4.5 1.84 vs reference) gives 1.84x.
- A lean loop without POT's Kp array and einsum checks at full convergence (GPT-5.2) gives 1.76x.
- A lean loop with 1-2 iterations (Qwen3 Coder 2.25, Gemini 2.5 Pro 2.23) is 1.22x over GLM-4.5 and 1.27x over GPT-5.2.
- Gemini 3.1 Pro has the same two components as Gemini 2.5 Pro and scores 1.77 vs 2.23; I cannot attribute the gap.
- Looser stopping tolerance alone (o4-mini 1e-7 vs Claude Opus 4.5 1e-9, both checking every iteration) gives 1.18x.

**Class: A.** Cutting the iteration count and cutting per-iteration overhead both shrink the same loop. Each reaches ~1.8x alone and together they add only ~1.25x, because exp(-M/reg), plan formation and the input copy are fixed n^2 passes that no published solver reduced (no float32, no threaded exp, no fused exp-and-scale).

**Validity.** Six of the nine solvers above 1.1x run 1-2 iterations. Five do so by accident (a convergence test that is satisfied by construction, or a misplaced `break`) and one (Qwen3 Coder) by design. They pass only because atol 1e-8 is a few percent of a typical entry. Published runs accepted them on all instances at n = 1813; Harbor's n = 1535 leaves less relative slack, so whether one iteration still passes there needs measuring. GPT-5.2, o4-mini and Claude Opus 4.5 converge genuinely (o4-mini with a looser stop). GPT-5.2 returns a reused buffer. GPT-5 Pro's `np.array(..., copy=False)` raises on Harbor's list inputs.

**Single change to test.** `ot.sinkhorn(a, b, M, reg, stopThr=1.0)` (or `numItermax=1`) reproduces GLM-4.5's 1.84x and relies on the tolerance. GPT-5.2's lean loop is the genuine alternative. Harbor caveat: the cost matrix arrives as 1535 nested lists (2.36M floats), and `np.ascontiguousarray` of that is a fixed cost in reference and solver alike that may dominate both. Measure it first.

## clustering_outliers

**Reference and verifier.** The reference runs `hdbscan.HDBSCAN(min_cluster_size=n//20, min_samples=min_cluster_size//2).fit` on n x 10 data (8 Gaussian clusters plus 10% outliers) and returns labels, probabilities and persistence as lists. The verifier recomputes the reference and checks required keys, label length n and labels >= -1, probabilities finite and within [0, 1] (length not checked), cluster count within 30% of the reference, noise fraction within 0.20 absolute, and ARI >= 0.5. `cluster_persistence` is never read. Any reasonable clustering of these separated blobs passes.

| Model | Speedup | Components |
|---|---|---|
| Qwen3 Coder | 2.53 | different algorithm: sklearn DBSCAN, eps = 0.5 on standardised data; constant probabilities 0.5; dummy persistence |
| GPT-OSS-120b | 1.93 | different algorithm: DBSCAN with eps = median k-NN distance from an unseeded 1000-point sample; float32; n_jobs=-1; probabilities = [] |
| Gemini 3.1 Pro Preview | 1.75 | library swap: sklearn.cluster.HDBSCAN with n_jobs=8; persistence [] (sklearn has no such attribute) |
| Claude Opus 4.5 | 1.46 | library swap: sklearn HDBSCAN, kd_tree, n_jobs=-1; persistence [] |
| o4-mini | 1.32 | hdbscan on a 2000-point subsample with faiss 1-NN label transfer; core_dist_n_jobs=-1 |
| R1 | 1.16 | hdbscan prims_kdtree, core_dist_n_jobs=-1 |
| Gemini 3 Pro Preview | 1.16 | hdbscan prims_kdtree, float32, leaf cluster selection, physical-core jobs |
| Claude Opus 4.1 | 1.16 | hdbscan prims_kdtree, float32, core_dist_n_jobs=-1 |
| Claude Opus 4 | 1.13 | hdbscan prims_kdtree, leaf_size 20, core_dist_n_jobs=-1 |
| GLM-4.5 | 1.13 | hdbscan prims_kdtree, float32, leaf cluster selection |

GPT-5, GPT-5.2 and GPT-5.4 (hdbscan boruvka_kdtree, float32, adaptive jobs) score 0.995-1.004.

**Natural experiments.**
- Tuning hdbscan (prims instead of boruvka, all cores, float32): 1.13-1.16x. Boruvka with more jobs (GPT-5 family) gives ~1.0x.
- Swapping to sklearn's HDBSCAN: 1.46-1.75x.
- Swapping to DBSCAN: 1.93-2.53x.
- Subsampling plus label transfer (o4-mini): 1.32x, below the plain library swap.

**Class: A.** The two routes above 1.5x (DBSCAN substitution and the sklearn HDBSCAN swap) each replace the entire clustering call, so they cannot stack. Parameter tuning adds ~1.15x and applies only to the hdbscan library that both swaps remove.

**Validity.** Qwen3 Coder and GPT-OSS-120b do not compute HDBSCAN and pass only because the verifier checks summary statistics and ARI >= 0.5. GPT-OSS-120b returns an empty probabilities list and sets eps from an unseeded random sample, so its output varies between calls; Harbor checks only the warmup call. o4-mini subsamples (n = 2457 and 2303 both exceed its 2000 threshold). Both sklearn swaps return empty persistence, which the verifier never reads. Gemini 3 Pro and GLM-4.5 use leaf cluster selection, a different clustering from the reference's eom, and are still accepted.

**Single change to test.** Replace `hdbscan.HDBSCAN` with `sklearn.cluster.HDBSCAN(..., n_jobs=8)` and guard the missing `cluster_persistence_` (exact algorithm, 1.75x published). DBSCAN is faster but depends on the verifier's looseness.

## ode_lorenz96_nonchaotic

**Reference and verifier.** The reference integrates Lorenz-96 (F = 2, N = n + 4, y0 = 2 + 0.01*U) with `solve_ivp` RK45, rtol = atol = 1e-8, over [0, 10]. Its RHS rebuilds three `np.roll(np.arange(N))` index arrays on every call and fancy-indexes x. It returns the final state as a list. The verifier requires allclose to the reference with rtol 1e-5 and atol 1e-8. Deterministic.

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro Preview | 11.29 | entire RK45 (Dormand-Prince with scipy's initial-step and step-size rules) compiled in numba, fastmath; numba RHS with explicit wraparound |
| Gemini 3 Pro Preview | 3.39 | numba RHS (explicit wraparound) inside solve_ivp |
| GLM-4.5 | 3.15 | numba RHS (modular indices) inside solve_ivp |
| R1 | 2.86 | numba RHS (closure, % N indices) inside solve_ivp |
| Claude Opus 4.5 | 2.67 | numba RHS with precomputed index arrays inside solve_ivp |
| GPT-5.4 | 2.63 | slicing RHS (no fancy indexing); scipy RK45 class stepped directly |
| GPT-5 | 2.33 | slicing RHS with a scratch buffer and in-place ops; solve_ivp |
| Gemini 2.5 Pro | 1.78 | numba RHS with index arrays passed as args; solve_ivp |
| GPT-5.2 | 1.75 | hoisted index arrays; solve_ivp (a fixed-step RK4 numba kernel is compiled but unused) |
| GPT-5-mini | 1.73 | hoisted index arrays |
| Qwen3 Coder | 1.73 | cached index arrays |
| Claude Opus 4 | 1.70 | hoisted index arrays |
| GPT-OSS-120b | 1.69 | np.roll on x instead of index arrays |
| o4-mini | 1.67 | hoisted index arrays |
| Claude Opus 4.1 | 1.66 | hoisted index arrays |
| GPT-5 Pro (medium) | 1.24 | hoisted index arrays, vectorized=True |

Claude Sonnet 4.5 (1.87) has no published code.

**Natural experiments.**
- Hoisting the index construction out of the RHS gives 1.66-1.73x (six solvers).
- Slicing instead of fancy indexing (GPT-5 2.33) is ~1.35x over hoisting.
- A compiled RHS inside solve_ivp gives 2.67-3.39x. Gemini 2.5 Pro's 1.78 with a compiled RHS is an outlier I cannot explain from the code.
- A compiled stepper on top of a compiled RHS (Gemini 3.1 Pro 11.29 vs Gemini 3 Pro 3.39) adds 3.3x.
- Stepping the RK45 class directly instead of solve_ivp, with the same slicing RHS (GPT-5.4 2.63 vs GPT-5 2.33), adds 1.13x, within noise.
- GPT-5 Pro's `vectorized=True` (1.24 vs ~1.7 for the same RHS) is either noise or its extra ndim branch; RK45 does not use the flag.

**Class: M.** The RHS (index rebuild and fancy indexing) and scipy's per-step Python overhead are different parts of the computation. Compiling the RHS gives ~3x, compiling the stepper gives a further ~3.3x, and the best solver has both.

**Validity.** Gemini 3.1 Pro keeps RK45 and the reference tolerances and follows scipy's control law closely, though not necessarily step for step. Its global error at rtol 1e-8 is far inside the verifier's rtol 1e-5 for this non-chaotic system. No solver above 1.1x loosens the tolerance or changes the method. Claude Opus 4.6's compiled DOPRI5 with a PI step controller is N/A, and the reason is not visible in the code.

**Single change to test.** A numba RHS inside solve_ivp (~3x). The 11x needs the stepper compiled as well, for example by porting Gemini 3.1 Pro's `rk45_solve`.

## pde_heat1d

**Reference and verifier.** The reference solves the 1D heat equation by the method of lines on 20n = 160 interior points (n = 8 in both harnesses) with zero Dirichlet boundaries. The RHS calls `np.pad` on every evaluation and `solve_ivp` RK45 runs with rtol = atol = 1e-6 over [0, 2]. With alpha ~ 0.01 and dx = 1/161 the explicit step is stability-limited, so the solver takes many small steps of a tiny system. It returns the final state as a list. The verifier requires allclose to the reference with rtol 1e-5 and atol 1e-8. Deterministic.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 37.79 | entire RK45 compiled in numba, replicating scipy's constants, initial-step selection and step control; numba RHS; DST closed-form solution computed only as a cross-check, with scipy fallback |
| Claude Opus 4.5 | 3.91 | numba RHS inside solve_ivp |
| Gemini 3.1 Pro Preview | 3.77 | numba RHS with a preallocated pad buffer; scipy RK45 class stepped directly |
| Gemini 3 Pro Preview | 3.50 | numba RHS |
| GLM-4.5 | 3.09 | numba RHS, fastmath; max_step=0.1 (no effect) |
| Qwen3 Coder | 2.95 | numba RHS that allocates its pad buffer per call |
| Claude Opus 4.6 | 2.68 | np.pad -> np.empty + slice assignment |
| GPT-5 | 2.27 | preallocated pad buffer, in-place ops |
| GPT-5 Pro (medium) | 2.02 | slicing RHS without padding |
| GPT-5-mini | 1.98 | np.empty pad |
| R1 | 1.92 | np.zeros pad |
| o4-mini | 1.80 | slicing RHS without padding |

GPT-OSS-120b (1.08) keeps np.pad. Claude Sonnet 4.5 (3.14) has no published code.

**Natural experiments.**
- Replacing np.pad with manual padding or slicing in numpy gives 1.80-2.68x, against 1.08 for GPT-OSS-120b, which keeps np.pad.
- A numba RHS gives 2.95-3.91x, about 1.6x over numpy slicing.
- A compiled stepper on top of a numba RHS (GPT-5.4 37.79 vs Claude Opus 4.5 3.91) adds 9.7x. Stepping the scipy RK45 class directly (Gemini 3.1 Pro 3.77) adds nothing.

**Class: M.** Same structure as Lorenz-96 with a larger stepper share, because at N = 160 scipy's per-step overhead dominates once the RHS is cheap.

**Validity.** GPT-5.4 reproduces scipy's RK45 path, so it should match the reference to rounding. No published solver returns the exact DST solution. GPT-OSS-120b's docstring describes one but its code runs RK45, and GPT-5.4 uses it only as a guard, which suggests the closed form was tried and not trusted. It would differ from the reference by the reference's own global error, which may exceed atol 1e-8 where |u| is small near the boundaries. Whether it passes is unknown and worth one local test.

**Single change to test.** np.pad -> preallocated buffer is the one-line change (~2-2.7x); a numba RHS gives ~3.5x; the 38x needs GPT-5.4's compiled stepper.

## Follow-ups noticed, not acted on

- The l1_pruning reference's first-break theta is not the textbook L1-ball projection. Exact solvers pass because the difference is below atol except for very small k. A Harbor instance with k below ~100 would fail them.
- The sinkhorn verifier tolerance (atol 1e-8, about 3% of the mean entry) accepts one Sinkhorn update. Most of the top published sinkhorn solvers stop after one iteration because their convergence check is true by construction, not by design.
- The clustering verifier never checks the length of probabilities or reads cluster_persistence.
- GPT-5.2 returns reused buffers on l0_pruning and sinkhorn. That is harmless in both harnesses but would break any harness that keeps results and validates them later.
