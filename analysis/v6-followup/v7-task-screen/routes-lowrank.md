# AlgoTune optimisation routes for seven linear-algebra and clustering tasks

I read every published `solver.py` for the seven tasks (17 model folders), `reports/agent_summary.json`, `reports/generation.json` (the published problem sizes and baseline times), the Harbor `evaluator.py` and `test_outputs.py` for each task, the upstream task files for pca, randomized_svd and spectral_clustering, and the scipy/sklearn source where a library default mattered. No code was run. Every speed figure below is a published number or a ratio of two published numbers, and every claim about Harbor behaviour is read from the code.

## Summary

| Task | Class | Best published | Route |
|---|---|---|---|
| sparse_lowest_eigenvalues_posdef | S | Qwen3 Coder 2.73 | ARPACK `tol` raised from 0 (machine precision) to 1e-6..5e-4 |
| sparse_lowest_eigenvectors_posdef | S | Qwen3 Coder 2.91 | Same task as above (identical code), same switch |
| pca | S | GPT-5.4 6.47 | Eigendecomposition of X^T X instead of sklearn's randomized PCA |
| randomized_svd | M | Gemini 3.1 Pro 6.17 | Power iterations 5 to 0 (about 2.9x) and float32 (about 1.56x on top) |
| sparse_pca | M | GPT-5.4 14.59 | Closed-form prox instead of cvxpy, top-5 ARPACK instead of full eigh, loose ARPACK tol; the exact answer is always X = 0 |
| spectral_clustering | M | GPT-5.2 41.46 | Direct dense eigh embedding instead of sklearn's, and a cheap single-init k-means instead of sklearn KMeans(n_init=10) |
| kmeans | M | Gemini 3.1 Pro 26.50 | faiss float32 k-means and a 3-5 iteration cap |

## Caveats that apply to all seven tasks

Published speedup is AlgoTuner's mean of per-instance ratios. Harbor's reward is the ratio of summed per-instance minimum times (10 interleaved repeats), validity is checked once per instance on the warmup call, and any invalid instance sets the reward to 1.0.

Problem sizes differ between the published runs and Harbor.

| Task | Published n | Harbor n | Instance at published n | Instance at Harbor n |
|---|---|---|---|---|
| sparse_lowest_eigen* (both) | 1341 | 1279 | 1341x1341 sparse SPD, roughly 170-180 nonzeros per row (estimated from the generator), k=5 | 1279x1279, k=5 |
| pca | 34 | 24 | X 1700x340, k=170 | X 1200x240, k=120 |
| randomized_svd | 776 | 480 | 776x776, k=194 | 480x480, k=120 |
| sparse_pca | 662 | 639 | 662x662 covariance, k=5 | 639x639, k=5 |
| spectral_clustering | 8 | 25 | 120-200 samples, 2-12 clusters | 375-625 samples, 8-37 clusters |
| kmeans | 278 | 351 | 5560x10, k=278 | 7020x10, k=351 |

The published harness turned large numeric lists into `.npy` arrays when it stored datasets (`AlgoTuner/utils/serialization.py`, `externalize_large_arrays`), so published solvers received ndarrays. Harbor calls `generate_problem` directly, so pca `X`, sparse_pca `covariance` and kmeans `X` arrive as Python lists and both the reference and the solver pay the list-to-array conversion. That shared cost will pull Harbor speedups on those three tasks below the published ones by an amount that needs measuring.

Older and newer models' runs do not compare cleanly. GPT-OSS-120b and GPT-5.4 run the same pca recipe (float32, X^T X, `np.linalg.eigh`) and scored 3.89 and 6.47. I therefore read natural experiments within two groups ordered by model release. The 2025 group is Opus 4, Opus 4.1, Gemini 2.5 Pro, o4-mini, R1, GLM-4.5, Qwen3 Coder, GPT-OSS-120b, GPT-5, GPT-5-mini and GPT-5 Pro. The later group is Opus 4.5, Opus 4.6, Gemini 3 Pro, Gemini 3.1 Pro, GPT-5.2 and GPT-5.4. That the groups ran on different hardware or harness versions is my inference and is not documented.

Noise inside a group is also large. In pca, Opus 4.5 and Opus 4.6 differ only in full versus subset scipy `eigh` and scored 5.11 and 2.75. On the sparse eigen task, Opus 4's solver is a copy of the reference and scored 1.74, while two other reference-equivalent 2025 solvers (GPT-5 Pro, GPT-5-mini) scored 1.00. Components worth less than about 1.3x cannot be resolved from published numbers.

I read "valid speedup >= 1.1" as a numeric entry (not "N/A") of at least 1.1. Solvers below 1.1 or invalid are not in the main tables but are used as controls in natural experiments. Where the S/M definitions need a judgement, I treat the number of iterations and the cost per iteration as different parts of the computation, because they multiply.

---

## sparse_lowest_eigenvalues_posdef and sparse_lowest_eigenvectors_posdef

**Reference and verifier.** The two tasks have identical `solve` and `is_solution` in Harbor and upstream; the "eigenvectors" task returns and checks the 5 smallest eigenvalues and never looks at eigenvectors. The reference calls ARPACK `eigsh(which="SM", ncv=20, maxiter=200n, return_eigenvectors=False)` with the default `tol=0` (machine precision). The verifier reruns the same call and accepts if every sorted value is within relative error 1e-6. ARPACK starts from a random vector, so the reference itself varies at round-off level.

**Eigenvalues task, valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| Qwen3 Coder | 2.73 | ARPACK tol 5e-4; `which="SA"`; ncv 15; maxiter capped at 50 restarts (fallback SM, tol 1e-4); CSC format |
| Gemini 3.1 Pro | 2.58 | tol 4.5e-4; SA; ncv 25; dense `eigh` subset only if n<1000 (not taken) |
| GPT-OSS-120b | 2.08 | tol 1e-5; SM as reference; ncv 20 as reference; maxiter 500 |
| GLM-4.5 | 2.07 | tol 1e-6; SM; ncv 15; maxiter 15n; CSC |
| o4-mini | 1.89 | tol 1e-6; SA; ncv 20 |
| R1 | 1.83 | tol 1e-6; SM; ncv 20; adds a sparse symmetry check `(mat != mat.T).sum()` each call (extra cost); shift-invert branch not taken |
| Claude Opus 4.5 | 1.75 | tol 1e-7; SA; ncv 20 |
| Claude Opus 4 | 1.74 | none; byte-for-byte the reference eigsh call |
| GPT-5.2 | 1.74 | tol 1e-7; SA; ncv 20; `v0` = ones; shift-invert branch gated on nnz/row <= 40 (not taken) |
| Gemini 2.5 Pro | 1.63 | tol 1e-7; SM; ncv 30; CuPy attempt falls through; monkey-patches `scipy.sparse.spmatrix.__bool__` at import |
| Claude Opus 4.1 | 1.50 | tol 1e-9; SA; ncv 20 |
| GPT-5.4 | 1.27 | tol 1e-7; SA; ncv 12 |
| GPT-5 | 1.11 | tol 1e-8; SA; ncv 12; diagonal and tridiagonal fast paths (not triggered) |

Solvers below 1.1 were Opus 4.6 0.87 and Gemini 3 Pro 0.83 (both dense LAPACK `eigh` subset at this n), GPT-5-mini 1.00 and GPT-5 Pro 1.00 (reference eigsh unchanged).

**Eigenvectors task (same computation), valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| Qwen3 Coder | 2.91 | tol 5e-4; SM; default ncv (20); maxiter 200 |
| GLM-4.5 | 2.55 | tol 1e-4; SM; ncv 40; maxiter 20n; CSC |
| R1 | 2.47 | tol 1e-5; SA; ncv 25; maxiter 1000; k hard-coded to 5 |
| Gemini 3 Pro | 2.32 | tol 1e-4; SA; ncv 20 |
| GPT-OSS-120b | 2.28 | tol 1e-4; SA; ncv 20 |
| GPT-5.2 | 1.93 | tol 1e-6; SA; ncv 32; `v0` = ones |
| GPT-5 | 1.88 | tol 5e-7; SA; ncv 22; constant `v0`; shift-invert and LOBPCG branches gated off by density |
| o4-mini | 1.88 | tol 1e-6; SM; ncv 20 |
| Gemini 2.5 Pro | 1.74 | main path is shift-invert `eigsh(sigma=0, which="SM")`, which by scipy's convention targets the eigenvalues farthest from 0 and would fail the verifier; falls back to dense `eigvalsh` on exception. I could not tell which path produced the score |
| Claude Opus 4.5 | 1.60 | tol 1e-8; SA; ncv 25; dense `evr` only for n<=800 (not taken) |
| Claude Opus 4 | 1.50 | different algorithm: dense `np.linalg.eigvalsh` of the full matrix |
| Claude Opus 4.1 | 1.49 | tol 1e-8; SA; ncv 15 |

Solvers below 1.1 were Opus 4.6 1.06 (reference copy), GPT-5 Pro 1.0006 (SA with tol 0), GPT-5.4 0.87 (dense `evr` subset, because nnz*20 >= n^2). Gemini 3.1 Pro (SA, tol 1e-3, ncv 50) and GPT-5-mini were invalid.

**Natural experiments (pooled over the two identical tasks)**

| Pair | Component | Ratio |
|---|---|---|
| o4-mini eigenvalues (SA, tol 1e-6) 1.89 vs o4-mini eigenvectors (SM, tol 1e-6) 1.88 | SA instead of SM | 1.01 |
| GPT-5 Pro eigenvectors (SA, tol 0) 1.0006 vs reference | SA instead of SM | 1.00 |
| GPT-OSS-120b eigenvalues (reference plus tol 1e-5) 2.08 vs reference | tol 0 to 1e-5 | 2.08 |
| Qwen3 Coder eigenvalues (tol 5e-4, ncv 15) 2.73 vs o4-mini eigenvalues (tol 1e-6, ncv 20) 1.89 | tol 1e-6 to 5e-4 | 1.44 |
| Gemini 3 Pro eigenvectors (tol 1e-4) 2.32 vs GPT-5.2 eigenvectors (tol 1e-6, ncv 32) 1.93 | tol 1e-6 to 1e-4 | 1.20 |
| Opus 4.5 eigenvalues (tol 1e-7, ncv 20) 1.75 vs GPT-5.4 eigenvalues (tol 1e-7, ncv 12) 1.27 | ncv 12 instead of 20 | 0.73 (a loss) |
| Opus 4.6 / Gemini 3 Pro eigenvalues, GPT-5.4 eigenvectors (dense LAPACK) 0.83-0.87 vs reference | dense LAPACK instead of ARPACK, later group | about 0.85 |
| Opus 4 eigenvectors (dense `eigvalsh`) 1.50, 2025 group | same, but the group's reference copy scored 1.74 | not usable |
| Gemini 3.1 Pro eigenvectors (tol 1e-3, ncv 50) invalid vs Qwen3 Coder eigenvectors (tol 5e-4) valid | validity edge | |

Across both tasks (valid solvers, ARPACK regular mode), tol 0 gives 1.00-1.06; 1e-9 to 1e-8 gives 1.11-1.60; 1e-7 gives 1.27-1.75; 5e-7 to 1e-6 gives 1.83-2.07; 1e-5 gives 2.08-2.47; 1e-4 gives 2.28-2.55; 4.5e-4 to 5e-4 gives 2.58-2.91; 1e-3 is invalid. The ncv=12 solvers sit at the bottom of their bands.

**Class S.** Every solver within 1.5x of the best (Qwen3 Coder 2.73, Gemini 3.1 Pro 2.58, GPT-OSS-120b 2.08, GLM-4.5 2.07, o4-mini 1.89 and R1 1.83 on the eigenvalue task; Qwen3 Coder 2.91, GLM-4.5 2.55, R1 2.47, Gemini 3 Pro 2.32 and GPT-OSS-120b 2.28 on the eigenvector task) differs from the reference mainly in the ARPACK `tol` argument, and the speedup rises monotonically with tol. The SM to SA switch that many solvers also made is worth 1.00. ncv between 15 and 40 shows no effect; only dropping it to 12 costs about 1.4x. No valid published solver used shift-invert at these sizes (every shift-invert branch was gated off or, for Gemini 2.5 Pro, of unknown effect), and dense LAPACK was slower in the later group.

**Validity notes.** Ritz values converge quadratically in the residual, so the eigenvalue error is far below `tol`; that is why tol up to 5e-4 passed the 1e-6 check. The edge lies between 5e-4 (passed in three published runs) and 1e-3 (failed). Solvers at 4.5e-4..5e-4 depend on that margin, and in Harbor a single failing instance sets the reward to 1.0. Qwen3 Coder's 50-restart cap relies on its SM tol=1e-4 fallback when ARPACK does not converge. Gemini 2.5 Pro's eigenvalue solver patches `spmatrix.__bool__` globally at import; Harbor loads the solver into the verifier's process, but this verifier never takes the truth value of a sparse matrix. No verifier exploit; the verifier recomputes the reference.

**Single change to test.** Add `tol=1e-4` to the reference's `eigsh` call and keep everything else. Published solvers at 1e-4 scored 2.28-2.55. `tol=5e-4` scored 2.58-2.91 with less validity margin. A second candidate for local measurement is dense `np.linalg.eigvalsh(mat.toarray())[:k]`, since Harbor sets 8 BLAS threads; published evidence on it conflicts (1.50 in the 2025 group, 0.83-0.87 in the later group).

---

## pca

**Reference and verifier.** The reference builds an array from `X`, centres it and fits sklearn `PCA(n_components=k)`. At these shapes sklearn 1.6 picks the randomized solver (4 LU-normalised power iterations, 10 oversamples, `random_state=None`), so the reference is random. The data is W H plus 0.01 uniform noise with rank exactly k, so the top-k subspace holds nearly all the variance. The Harbor verifier checks V V^T = I (rtol 1e-4, atol 1e-5) and that ||X V^T||_F^2 is within rtol 1e-4 of the reference's value; any orthonormal basis of the signal subspace passes, in any order or rotation. The upstream verifier replaces the objective check with |V V_ref^T| = I under the same tolerances, which requires sklearn's descending order and per-component agreement up to sign.

**Valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 6.47 | float32; X^T X then `np.linalg.eigh`; rows returned in ascending order |
| Gemini 3.1 Pro | 5.57 | float64; X^T X then `np.linalg.eigh` (k = n/2, so its subset branch is not taken); in-place centring |
| Gemini 3 Pro | 5.45 | float32; X^T X then scipy `eigh` subset (`evr`), `overwrite_a`, `check_finite=False` |
| Claude Opus 4.5 | 5.11 | float64; X^T X then scipy `eigh` (full), `check_finite=False` |
| R1 | 4.15 | float32; X^T X then SVD of the 340x340 matrix (`gesdd`); sets `OPENBLAS_NUM_THREADS` after numpy is loaded (no effect) |
| GPT-OSS-120b | 3.89 | float32; X^T X then `np.linalg.eigh`; `argpartition` |
| GPT-5 | 3.69 | float64; X^T X then `np.linalg.eigh`; ascending order |
| o4-mini | 3.62 | float64; X^T X then scipy `eigh` driver `evd` |
| GPT-5-mini | 3.54 | float64; X^T X then `np.linalg.eigh` |
| GPT-5 Pro | 3.47 | float64; X^T X then `np.linalg.eigh` (its ARPACK branch is gated off) |
| Claude Opus 4.6 | 2.75 | float64; X^T X then scipy `eigh` subset |
| GLM-4.5 | 2.51 | float64; X^T X then `np.linalg.eigh`; subtracts column sums instead of means (bug); ascending; returns a list |
| Claude Opus 4.1 | 2.41 | float32; thin SVD of X (scipy `gesdd`, `check_finite=False`) |
| Gemini 2.5 Pro | 2.16 | float64; X^T X then scipy `eigh`; returns a list |
| Qwen3 Coder | 1.89 | float32; faiss `PCAMatrix`; returns a list |
| GPT-5.2 | 1.43 | float64; thin SVD of X (scipy `gesdd`, Fortran copy) |

Opus 4 (numba-compiled `eigh` on the covariance) was invalid.

**Natural experiments**

| Pair | Component | Ratio |
|---|---|---|
| GPT-5.4 (float32) 6.47 vs Gemini 3.1 Pro (float64), same recipe, later group | float32 | 1.16 |
| GPT-OSS-120b (float32) 3.89 vs GPT-5 3.69 and GPT-5-mini 3.54 (float64), 2025 group | float32 | 1.05-1.10 |
| Gemini 3.1 Pro (X^T X eigh) 5.57 vs GPT-5.2 (thin SVD of X), both float64, later group | covariance eigh instead of thin SVD | 3.9 |
| GPT-OSS-120b 3.89 / R1 4.15 vs Opus 4.1 2.41, all float32, 2025 group | covariance eigh instead of thin SVD | 1.6-1.7 |
| R1 (SVD of covariance) 4.15 vs GPT-OSS-120b (eigh of covariance) 3.89 | eigensolver on the 340x340 matrix | 1.07 |
| Opus 4.5 (full eigh) 5.11 vs Opus 4.6 (subset eigh) 2.75, both float64 | subset vs full, or noise | 1.86, not explained by the code |
| GPT-5 (array out) 3.69 vs Gemini 2.5 Pro (list out) 2.16 | list output plus unexplained remainder | 1.7; `.tolist()` of 170x340 cannot account for most of it |

**Class S.** The four solvers within 1.5x of the best (GPT-5.4 6.47, Gemini 3.1 Pro 5.57, Gemini 3 Pro 5.45, Opus 4.5 5.11) all replace sklearn PCA with an eigendecomposition of the 340x340 matrix X^T X. float32 adds 1.05-1.16x on top. Eigensolver driver, subset selection and `check_finite` are within the noise. Thin SVD of X is a weaker alternative for the same step (1.43-2.41).

**Validity notes.** float32 Gram matrices square the condition number; the Harbor check is loose and rotation-invariant, so all float32 solvers pass there. GLM-4.5's sum-for-mean bug adds a huge rank-one term along the mean direction; its basis still spans the signal subspace because the mean of W H lies in the row space of H, so it passes only because the verifier checks the objective rather than the components. No solver relies on randomness or early stopping.

**Harbor accepts, upstream rejects.** GPT-5.4, GPT-5 and GLM-4.5 return the top-k rows in ascending eigenvalue order, which fails the upstream |V V_ref^T| = I check outright; GLM-4.5 also returns rotated components. The float32 solvers (Gemini 3 Pro, R1, GPT-OSS-120b, Opus 4.1, Qwen3 Coder) are at risk upstream because off-diagonal alignment must stay below 1e-5 and float32 eigenvector error grows with the inverse eigenvalue gap; this needs measuring. The float64 descending solvers (Gemini 3.1 Pro, Opus 4.5, o4-mini, GPT-5-mini, GPT-5 Pro, Opus 4.6, Gemini 2.5 Pro, GPT-5.2) should pass both, except on instances with nearly equal eigenvalues where any exact method can disagree with the randomized reference (needs measuring).

**Single change to test.** Replace the sklearn fit with `w, V = np.linalg.eigh(X.T @ X); return V[:, ::-1][:, :k].T` in float64 and descending order, which also satisfies the upstream check. This recipe scored 3.5-5.6 published; Harbor's list input will lower it.

---

## randomized_svd

**Reference and verifier.** The reference is sklearn `randomized_svd(A, k, n_iter=5, random_state=42)` with 10 oversamples and the LU power-iteration normaliser (n_iter 10 for ill-conditioned matrices, but `generate_problem` defaults to `low_rank` and every instance is low rank, so the other matrix types never occur). The verifier checks shapes, finiteness, U^T U and V^T V within atol 1e-5 of I, non-negative descending s, and ||A - U diag(s) V^T||_F / ||A||_F <= tol, where tol = (1 + n/2000) for these instances, i.e. 1.388 at n=776 and 1.24 at n=480. It never compares with the reference. Because tol exceeds 1, orthonormal U and V with s = 0 pass (relative error exactly 1). Upstream caps tol at 0.9.

**Valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro | 6.17 | float32; first k columns of A as the sketch (no random matrix, no A@Omega); n_iter 0; no oversampling; scipy QR and SVD with `check_finite=False` |
| GPT-5.2 | 6.02 | float32; Gaussian sketch; n_iter 0; oversample 2; eigh of the small Gram matrix instead of SVD of B, V = A^T Q U / s |
| Gemini 3 Pro | 5.25 | float32; uniform sketch; n_iter 0; no oversampling; scipy QR/SVD |
| GPT-5.4 | 4.95 | float32; Gaussian sketch drawn in float64 then cast; n_iter 0; oversample 2; scipy QR and `gesdd` |
| GPT-5 | 4.80 | float32; sklearn `randomized_svd` with n_iter 0, n_oversamples 0, normaliser none, `flip_sign=False` |
| GLM-4.5 | 4.61 | float32; sklearn with n_iter 0, n_oversamples 0; outputs cast to float64 |
| R1 | 4.51 | float32; JAX-jitted custom range finder; n_iter 0; no oversampling |
| GPT-OSS-120b | 4.48 | float32; sklearn with n_iter 0 (10 oversamples) |
| Claude Opus 4.5 | 4.36 | float32 in effect (JAX with x64 disabled); JAX jit; n_iter 1 with QR; oversample 5 |
| Qwen3 Coder | 4.02 | float32; numpy range finder; n_iter 0; no oversampling; outputs cast to float64 |
| o4-mini | 3.79 | float32; numpy range finder on A^T; n_iter 0; no oversampling |
| GPT-5-mini | 2.88 | float64; sklearn with n_iter 0 (10 oversamples); otherwise the reference |
| Claude Opus 4.6 | 2.52 | float64 input, float32 power phase; n_iter 1 with LU; 10 oversamples; float64 QR and SVD |
| Claude Opus 4.1 | 2.49 | float64; n_iter 1 without normalisation; oversample 1; numba-wrapped `np.dot` |

Opus 4, Gemini 2.5 Pro and GPT-5 Pro were invalid.

**Natural experiments**

| Pair | Component | Ratio |
|---|---|---|
| GPT-5-mini (reference with n_iter 0) 2.88 vs reference | power iterations 5 to 0 | 2.88 |
| GPT-OSS-120b 4.48 vs GPT-5-mini 2.88 (both sklearn, n_iter 0), 2025 group | float32 | 1.56 |
| GLM-4.5 4.61 / GPT-5 4.80 vs GPT-OSS-120b 4.48 | oversampling 10 to 0 | 1.03-1.07 |
| GPT-5-mini (n_iter 0) 2.88 vs Opus 4.1 (n_iter 1) 2.49, float64, 2025 group | one power iteration | 1.16 |
| GPT-5.4 (n_iter 0) 4.95 vs Opus 4.5 (n_iter 1) 4.36, float32, later group | one power iteration | 1.14 |
| GPT-5.2 (Gram eigh) 6.02 vs GPT-5.4 (SVD of B) 4.95, later group | small Gram eigh instead of SVD | 1.22 |
| Gemini 3.1 Pro (column sketch) 6.17 vs Gemini 3 Pro (random sketch) 5.25, later group | no random matrix and no A@Omega | 1.18 |
| R1 (JAX) 4.51 vs GPT-5 / GLM-4.5 (sklearn) 4.6-4.8 | JAX jit | about 0.95 |

**Class M.** Removing the power iterations (2.88x alone, GPT-5-mini) and float32 (1.56x on top, GPT-OSS-120b) act on different aspects of the work and stack. Every solver within 1.5x of the best (6.17 down to 4.36) combines both. The later-group refinements (Gram eigh, column sketch) add about 1.2x each and replace different sub-steps.

**Validity notes.** All solvers skip power iterations and most drop oversampling. They pass because the verifier tolerance (1.24-1.39) is roughly a hundred times the error of the best rank-k approximation on these instances (my estimate of that error is about 0.01, not measured). Gemini 3.1 Pro's first-k-columns sketch relies on the signal rank equalling k. float32 outputs meet the 1e-5 orthonormality check with modest margin; GPT-5.2 forms V from the Gram matrix, whose orthonormality error grows with the square of the singular-value spread, so its margin is the thinnest and needs measuring at Harbor n.

**Harbor accepts, upstream rejects.** None as far as the code shows. No published solver returns s = 0 or another trivial factorisation, and all of them compute a real approximation whose error should be far below 0.9. The Harbor verifier does accept trivial outputs (s = 0) at both published and Harbor sizes.

**Single change to test.** Pass `n_iter=0` in the reference call (this is GPT-5-mini's solver, 2.88). Adding `A.astype(np.float32)` as a second change gives GPT-OSS-120b's solver (4.48).

---

## sparse_pca

**Reference and verifier.** The reference converts the covariance list, runs a full `np.linalg.eigh`, sets B to the top-5 eigenvectors scaled by sqrt(eigenvalue), and solves min ||B - X||_F^2 + lambda ||X||_1 subject to column norms <= 1 with cvxpy. The verifier checks shape, column norms <= 1 + 1e-5, that each reported explained variance equals x^T A x, and that the objective is at most 1.01 times the reference objective (it reruns the cvxpy solve). Upstream is identical.

**The exact answer is always zero.** The problem separates by column, and its exact solution is soft-thresholding B at lambda/2 followed by scaling to unit norm. Here lambda = 0.5 max|A| and A = G^T G + 0.1 I is positive definite, so max|A| is the largest diagonal entry and lambda/2 = max_i A_ii / 4. Each entry satisfies |B_ij| <= sqrt(A_ii), because A_ii = sum_l mu_l v_il^2 >= mu_j v_ij^2. So whenever max_i A_ii >= 16, every entry is thresholded to zero and X = 0 with zero explained variance is the exact optimum. A_ii is about n (662 or 639), so this holds on every generated instance by a wide margin.

**Valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| GPT-5.4 | 14.59 | closed-form prox (soft-threshold at lambda/2, scale to unit norm); top-5 via `eigsh(which="LA", tol=1e-5, ncv=21)`; returns ndarrays, no `.tolist()` |
| Claude Opus 4.5 | 12.04 | closed form (numba); `eigsh(which="LM", tol=1e-6, ncv=20)`; `.tolist()` |
| Qwen3 Coder | 10.24 | the saved file raises before any solve (`cp.Variable(..., sparse=True)` is not a cvxpy attribute in the versions I know, and `prob` is undefined anyway) and returns all-zero components from its except branch |
| R1 | 9.08 | different algorithm: thresholded power method (threshold lambda on A v) with deflation; returns zeros after one matvec per component; no eigendecomposition |
| Gemini 3.1 Pro | 6.83 | closed form; `eigsh(which="LA")` with default tol 0; `.tolist()` |
| Gemini 3 Pro | 6.44 | closed form; `eigsh(which="LA")` tol 0; `.tolist()` |
| GPT-5 | 6.40 | closed form; `eigsh(which="LM", tol=1e-6)`; early return when all columns are zero (skips A X); `.tolist()` |
| Gemini 2.5 Pro | 5.44 | `eigsh(which="LM", tol=1e-6)`; numba FISTA per column (stops after about two steps because the answer is zero) |
| GPT-OSS-120b | 4.91 | closed form; `eigsh(which="LA")` tol 0 |
| GPT-5-mini | 4.91 | closed form; `eigsh(which="LA")` tol 0 |
| o4-mini | 4.91 | closed form; `eigsh(which="LA")` tol 0 |
| Claude Opus 4.6 | 4.88 | closed form (numba); scipy `eigh` subset (`evr`) |
| GPT-5.2 | 4.76 | closed form; scipy `eigh` subset (`evr`) on a Fortran copy; returns ndarrays |
| Claude Opus 4.1 | 1.75 | full `np.linalg.eigh`; numba proximal gradient, 50 iterations |
| GPT-5 Pro | 1.62 | full `np.linalg.eigh`; closed form |
| Claude Opus 4 | 1.61 | full scipy `eigh` in float32; JAX FISTA up to 300 iterations |
| GLM-4.5 | 1.16 | full `eigh`; keeps cvxpy, ECOS with 20 iterations and loose tolerances |

**Natural experiments**

| Pair | Component | Ratio |
|---|---|---|
| GPT-5 Pro (full eigh + closed form) 1.62 vs reference | cvxpy replaced by the closed form | 1.62 |
| GPT-5-mini / o4-mini / GPT-OSS-120b (top-5 eigsh + closed form) 4.91 vs GPT-5 Pro 1.62, 2025 group | full eigh replaced by top-5 ARPACK | 3.03 |
| GPT-5 (tol 1e-6) 6.40 vs GPT-5-mini (tol 0) 4.91, 2025 group | ARPACK tol 0 to 1e-6 (and skipping A X) | 1.30 |
| Opus 4.5 (tol 1e-6) 12.04 vs Gemini 3.1 Pro (tol 0) 6.83, later group | ARPACK tol 0 to 1e-6 | 1.76 |
| Gemini 3.1 Pro / Gemini 3 Pro (eigsh) 6.44-6.83 vs Opus 4.6 / GPT-5.2 (LAPACK subset) 4.76-4.88, later group | ARPACK instead of LAPACK subset | 1.32-1.43 |
| GPT-5.2 (ndarray out) 4.76 vs Opus 4.6 (`.tolist()`) 4.88 | output type | about 1.0 |
| GPT-5.4 (tol 1e-5, ndarrays) 14.59 vs Opus 4.5 (tol 1e-6, lists) 12.04 | tol and output type together | 1.21 |
| R1 (no eigendecomposition, zeros) 9.08 vs GPT-5 (eigsh tol 1e-6) 6.40, 2025 group | skipping the eigendecomposition | 1.42 |
| GLM-4.5 (cvxpy, ECOS 20 iterations) 1.16 vs GPT-5 Pro (closed form) 1.62 | loose cvxpy vs closed form | 1.40 |

**Class M.** The reference spends its time in two different places, and published solvers replaced both. Replacing cvxpy with the closed form is worth 1.62x alone, replacing the full `eigh` with a top-5 ARPACK call is worth 3.0x on top, and a loose ARPACK tolerance adds 1.3-1.8x more. The best solvers stack all three. The accidental zero route (R1 9.08, Qwen3 Coder 10.24) is an alternative to the eigendecomposition steps rather than an addition to them.

**Validity notes.** Because the answer is zero, eigenvector accuracy never matters here, so loose ARPACK tolerances cannot cause a failure on this generator. R1 solves a different objective and passes only because it also returns zeros. Qwen3 Coder's score comes from code that crashes into its fallback; I could not check whether the scored version is the saved file. Nothing here exploits a verifier gap; the generator makes the task degenerate.

**Single change to test.** Return zeros when `sqrt(max_i A[i][i]) <= sparsity_param / 2`, else fall back to the reference. The test reads only the diagonal (no array conversion), is exact on every instance, and should beat every published solver; its Harbor speedup needs measuring. The general-purpose route is the closed form plus `eigsh(A, k, which="LA", tol=1e-6)`, which scored about 12 published (Opus 4.5).

---

## spectral_clustering

**Reference and verifier.** The reference is sklearn `SpectralClustering(affinity="precomputed", assign_labels="kmeans", random_state=42)`. That computes an ARPACK shift-invert embedding with `tol=0` (LOBPCG if ARPACK raises) and runs k-means++ with `n_init=10`, without row normalisation. The published baseline is heavy-tailed (mean 57 ms, standard deviation 217 ms across instances in `generation.json`); I did not determine which instances are slow. The Harbor verifier never calls the sklearn reference. It requires exactly k distinct labels, then accepts if any one of five signals passes (within-vs-between similarity margin, top-10% pair agreement, silhouette, modularity, or agreement with its own deterministic spectral reference made from a dense `eigh` of the normalised Laplacian, farthest-first initialisation and 15 Lloyd iterations), unless the labels match an argmax-over-k-columns pattern (hard fail at 0.98-0.985 agreement). Upstream requires agreement with that deterministic reference AND at least one structural signal AND Ncut and silhouette floors, and it rejects labels within 0.94 of a hack pattern unless all of those hold.

**Valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| GPT-5.2 | 41.46 | dense `np.linalg.eigh` of the normalised Laplacian (n<=320); row normalisation; farthest-first initialisation and 12 Lloyd iterations, i.e. the verifier's own reference procedure; its own O(n) ARI/NMI to detect hack-pattern matches and perturb labels; no sklearn |
| GLM-4.5 | 19.39 | float32; numba Laplacian; scipy `eigh` subset of k+1 vectors, drops the first; numba k-means, random init, 5 iterations |
| GPT-OSS-120b | 17.19 | no spectral embedding; k-means++ on row-normalised similarity rows (10 iterations, unseeded); random label permutation; 30% of labels reassigned at random to avoid the hack detectors |
| R1 | 13.51 | float32; `eigsh(which="LM", tol=1e-4)` on D^-1/2 S D^-1/2; faiss spherical k-means, 2 iterations, 1 run |
| Claude Opus 4.1 | 10.53 | float32; dense `eigh` subset; faiss k-means, 20 iterations x 3 runs |
| Gemini 2.5 Pro | 9.88 | float64; `eigsh(which="LM", tol=1e-4, v0=ones)`; faiss k-means 20 x 3 |
| GPT-5 | 9.51 | float32; dense `eigh` subset of the normalised affinity; numpy spherical k-means, single init, up to 25 iterations |
| GPT-5-mini | 6.29 | float64; dense `eigh` subset; numpy k-means, 8 inits, up to 300 iterations |
| Qwen3 Coder | 5.82 | float64; numba normalisation; dense `eigh` subset; sklearn `KMeans(n_init=3)` |
| o4-mini | 4.61 | `eigsh(which="LA")` tol 0; numpy k-means++, unseeded, up to 100 iterations with an (n, k, d) broadcast |
| Gemini 3.1 Pro | 1.96 | dense `eigh` subset; scipy `kmeans2` with `++` init; re-implements the hard-fail detectors with sklearn ARI/NMI (up to 8 checks) and flips 4% of labels on a match |

Opus 4 (its saved file is log text), Opus 4.5, Opus 4.6, Gemini 3 Pro, GPT-5 Pro and GPT-5.4 were invalid.

**Natural experiments**

| Pair | Component | Ratio |
|---|---|---|
| GPT-5 (1 init, float32) 9.51 vs GPT-5-mini (8 inits, float64), same dense embedding, 2025 group | k-means restarts 8 to 1 | 1.51 |
| Opus 4.1 (faiss 20x3) 10.53 vs Qwen3 Coder (sklearn KMeans n_init 3), both dense `eigh` subset | sklearn KMeans to faiss | 1.81 |
| GLM-4.5 (numba, 5 iterations) 19.39 vs Opus 4.1 (faiss 20x3), both dense `eigh` subset | lighter k-means | 1.84 |
| R1 (faiss 2x1, float32) 13.51 vs Gemini 2.5 Pro (faiss 20x3, float64), both `eigsh` tol 1e-4 | lighter k-means | 1.37 |
| Opus 4.1 (dense `eigh` subset) 10.53 vs Gemini 2.5 Pro (`eigsh` tol 1e-4), same faiss settings | choice of new embedding solver | 1.07 |
| GPT-5.2 41.46 vs Gemini 3.1 Pro 1.96, later group | own O(n) metrics and plain Lloyd vs sklearn-metric self-checks and `kmeans2` | 21 |

**Class M.** The reference has two expensive sklearn stages and the fast solvers replace both. The k-means replacement is measured in four natural experiments at 1.37-1.84x. The embedding replacement is not isolated by any published pair, because no solver keeps sklearn's k-means with a new embedding. Arithmetic on Qwen3 Coder (5.82, dense embedding with sklearn KMeans n_init 3) and GLM-4.5 (19.39, same embedding with a 5-iteration k-means) implies sklearn k-means is most of Qwen's time, which puts the reference's own embedding at roughly half or more of reference time; that is inference from mean-of-ratio numbers, not a measurement. Dense `eigh` and loose `eigsh` are interchangeable for the new embedding (1.07x), so that sub-choice is type A. The best solver (GPT-5.2) combines a dense embedding with a cheap deterministic Lloyd. Timing sklearn's `spectral_embedding` and its k-means separately on the instance set would confirm or overturn the M reading.

**Validity notes.** GPT-OSS-120b is a verifier exploit by its own docstring; it skips the spectral step and randomises 30% of labels so that the hack detectors do not fire, relying on Harbor accepting any single weak quality signal. GPT-OSS-120b and o4-mini use unseeded random generators, so their outputs and validity vary between runs. GLM-4.5 drops the first eigenvector and uses a 5-iteration random-init k-means. R1's `which="LM"` can pick eigenvalues near -1 instead of the top positive ones. Gemini 3.1 Pro and GPT-5.2 perturb their labels whenever the honest clustering matches a hack pattern. Harbor's larger instances (375-625 samples, 8-37 clusters) switch several solvers onto other code paths. GLM-4.5 then uses a 10-step power method for its eigenvectors and sklearn KMeans with 5 iterations. GPT-5.2 then uses an `eigsh` LinearOperator with `which="SM"`, tol 1e-4 and maxiter 250, and on non-convergence its except branch returns round-robin labels that would very likely fail. All of this needs measuring at Harbor n.

**Harbor accepts, upstream rejects.** GPT-OSS-120b is the clear case, since 30% random relabelling should fail upstream's required reference agreement and Ncut floor (not run). R1, GLM-4.5 and o4-mini (crude embeddings or k-means) are at risk upstream, as are Gemini 3.1 Pro and GPT-5.2, whose perturbations stop just under the 0.985 hack threshold and so land in upstream's 0.94 near-hack band, where acceptance needs strict reference agreement (probably met, since their unperturbed labels track the verifier's reference). This needs running.

**Single change to test.** `SpectralClustering(..., n_init=1)` is a one-argument change; by analogy with GPT-5-mini to GPT-5 it should be worth about 1.5x. The larger gain needs the embedding swap as well, i.e. `np.linalg.eigh` of the normalised Laplacian, row normalisation and a single-init Lloyd (GPT-5.2's core).

---

## kmeans

**Reference and verifier.** The reference is sklearn `KMeans(n_clusters=k).fit(X)` on the list X. sklearn 1.6 (the published pin) defaults to k-means++ initialisation, `n_init="auto"` (one run), `max_iter=300`, `tol=1e-4` and `random_state=None`, so the reference is random. The verifier reruns the reference and accepts if 0.95 x loss(solution) <= loss(reference) + 1e-5, with labels in [0, k). The data is effectively one Gaussian blob (centre spread 10/n of about 0.04 against cluster std 1 + 0.2n of about 57), so this is vector quantisation of unstructured data.

**Valid solvers with speedup >= 1.1**

| Model | Speedup | Components |
|---|---|---|
| Gemini 3.1 Pro | 26.50 | faiss float32; random-sample init; 3 iterations; 1 run |
| Claude Opus 4.5 | 25.52 | faiss float32; 3 iterations; 1 run |
| Claude Opus 4.6 | 21.59 | faiss float32; 4 iterations; subsampling to max(20k, 2000) points, which equals n here and so is inactive |
| o4-mini | 16.87 | faiss float32; 3 iterations; unseeded |
| Claude Opus 4 | 15.49 | faiss float32; 5 iterations |
| Gemini 2.5 Pro | 15.25 | faiss float32; 5 iterations; subsample and PCA branches inactive at this size |
| R1 | 12.53 | faiss float32; 5 iterations; sets faiss to 8 threads |
| Claude Opus 4.1 | 9.29 | faiss float32; 10 iterations |
| Qwen3 Coder | 6.50 | sklearn KMeans, `init="random"`, 1 run, `max_iter=3`, `tol=5e-2` |
| GPT-5 | 5.75 | faiss float32; 14 iterations |
| Gemini 3 Pro | 5.11 | numba float32 Lloyd, single thread, random init, until labels stop changing (at most 100 iterations) |
| GPT-5.2 | 4.86 | sklearn KMeans with k-means++, 1 run, `max_iter=100` instead of 300, fixed seed |
| GPT-OSS-120b | 3.60 | numpy float64 Lloyd, random init, 5 iterations, `np.add.at` centroid update |

GPT-5.4 0.75 (MiniBatchKMeans) and GPT-5-mini 0.079 were below 1.1; GLM-4.5 and GPT-5 Pro were invalid.

**Natural experiments**

| Pair | Component | Ratio |
|---|---|---|
| faiss at 3 (o4-mini 16.87), 5 (Opus 4 15.49, Gemini 2.5 Pro 15.25, R1 12.53), 10 (Opus 4.1 9.29) and 14 iterations (GPT-5 5.75), 2025 group | iteration count | 14 to 3: 2.9; 10 to 5: 1.6; 5 to 3: 1.1-1.3 |
| faiss 3 iterations (Gemini 3.1 Pro 26.50, Opus 4.5 25.52) vs 4 (Opus 4.6 21.59), later group | iteration count | 1.2 |
| o4-mini (faiss, 3 iterations) 16.87 vs Qwen3 Coder (sklearn random init, 3 iterations) 6.50, 2025 group | faiss float32 instead of sklearn at a fixed iteration count | 2.6 |
| faiss 5 iterations (12.5-15.5) vs GPT-OSS-120b (numpy float64, 5 iterations) 3.60, 2025 group | faiss instead of numpy `add.at` | 3.5-4.3 |
| GPT-5.2 (reference with max_iter 100) 4.86 vs reference | iteration cap 300 to 100 alone | 4.86 |
| Gemini 3 Pro (numba, at most 100 iterations) 5.11 vs GPT-5.2 (sklearn, at most 100) 4.86 | implementation at a similar cap | 1.05 |

**Class M.** Capping Lloyd iterations (sklearn max_iter 300 to 100 alone gave 4.86x; within faiss, 14 to 3 iterations gave 2.9x) and the per-iteration implementation (faiss float32 instead of sklearn gave 2.6x at the same 3 iterations) multiply, and every solver within 1.5x of the best uses both. Random-sample initialisation instead of k-means++ comes bundled with the faiss swap and is not separated by any pair.

**Validity notes.** Every fast solver stops after 3-5 Lloyd iterations from a random-sample init, so it relies on the 5% slack against a random, converged k-means++ reference. Acceptance is therefore probabilistic per instance (the reference has no seed, and o4-mini's faiss run has none either). Harbor checks each instance once, and one failure sets the reward to 1.0. Harbor's X is a list, so conversion is a shared cost that is significant next to a 3-iteration faiss run.

**Single change to test.** Replace the sklearn fit with `faiss.Kmeans(d, k, niter=5, nredo=1, seed=0)` on float32, then `index.search` for labels (12.5-15.5 published at 5 iterations, 21.6-26.5 at 3-4 iterations in the later group). Within sklearn, `max_iter=100` alone scored 4.86 (GPT-5.2).

---

## Harbor versus upstream verifiers (pca, randomized_svd, spectral_clustering)

| Task | Upstream change | Published solvers Harbor accepts and upstream rejects |
|---|---|---|
| pca | Objective check replaced by per-component alignment with sklearn's ordered components | Certain from code: GPT-5.4, GPT-5, GLM-4.5 (ascending order; GLM-4.5 also rotated). Needs measuring: float32 solvers Gemini 3 Pro, R1, GPT-OSS-120b, Opus 4.1, Qwen3 Coder |
| randomized_svd | Tolerance capped at 0.9 | None from the code; the cap only rejects trivial outputs such as s = 0, which no published solver returns |
| spectral_clustering | Requires reference agreement, a structural signal and Ncut/silhouette floors; adds a 0.94 near-hack band | GPT-OSS-120b (by construction, not run). At risk: R1, GLM-4.5, o4-mini, Gemini 3.1 Pro, GPT-5.2 |

## Follow-ups noticed, not acted on

- The Harbor and upstream `sparse_lowest_eigenvectors_posdef` tasks are copies of the eigenvalues task (they return and check eigenvalues), so the benchmark counts one task twice.
- `randomized_svd` in Harbor accepts U, V orthonormal with s = 0 at both published and Harbor sizes, and every generated instance is `low_rank`, so the solvers' other matrix-type branches are dead code.
- `sparse_pca` as generated never exercises sparse PCA; the exact answer is zero on every instance for n >= about 16.
- Harbor's `PROBLEM_SIZE` differs from the published n for all seven tasks, and for spectral_clustering the change (published 120-200 samples, Harbor 375-625) moves several solvers onto untested code paths.
- Harbor scores the ratio of summed times while the published numbers are means of per-instance ratios; with spectral_clustering's heavy-tailed reference this will reweight instances substantially.
- Opus 4's reference-copy solver scored 1.74 on the sparse eigenvalue task; Opus 4's sparse-eigen numbers should not be used as evidence.
- Qwen3 Coder's saved sparse_pca solver cannot reach its own solve, and Gemini 2.5 Pro's saved eigenvector solver would fail on its main path, so for at least these two the saved file may not be what was scored.
