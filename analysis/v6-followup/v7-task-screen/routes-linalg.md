# Optimisation routes in published AlgoTune solvers, linear-algebra tasks

| Task | Class | Decisive evidence |
|---|---|---|
| eigenvalues_real | S | All 17 solvers sit at 1.79-2.52x and all make one change, eigh with vectors to eigenvalues only. Driver and library choices move it by 1.04x or less. |
| two_eigenvalues_around_0 | S | Every solver within 1.5x of the best (3.45) replaces the full dense spectrum with shift-invert Lanczos around 0. ARPACK tuning adds 1.10-1.22x on top. |
| generalized_eigenvectors_real | U | Ten solvers cluster at 1.36-1.48x with one sygvd call. The best (GPT-5.2, 4.96) bundles several changes, and two near-identical float32 solvers differ by 1.89x. |
| generalized_eigenvalues_complex | M | A clean 2x2 factorial. QZ to B^-1 A + standard eig is worth 1.95x and float32 is worth 1.40x, and they multiply. Both sit on top of eig to eigvals (1.95x). |
| generalized_eigenvectors_complex | M | QZ to B^-1 A + eig is worth 2.2x (earlier-model runs) and float32 is worth 1.27-1.37x. R1 combines both and is best at 3.36. |
| psd_cone_projection | S | eig to eigh accounts for 8-9x in every solver. Subspace, syrk and driver changes add at most 1.08x. |
| linear_system_solver | U | 16 solvers reach at most 1.13x. Gemini 3 Pro's 27.4x has no code explanation, because Gemini 3.1's near-identical solver scored 1.08x. |

## Reading choices and data caveats

- I read "do not run any code" as a ban on executing solvers, references or benchmarks. I used read-only shell commands (ls, cat, jq, grep, diff) to view the files, and I read SciPy's installed source to confirm default drivers.
- The tables list solvers with a valid published speedup of 1.1 or more. Rows below 1.1 or marked N/A appear in the text only where they serve as a control. claude-sonnet-4-5 has speedups but no code, so it is quoted only as a number.
- The task code is identical in the published repo (`AlgoTuneTasks/<task>/<task>.py`) and the harbor `tests/evaluator.py`. The diff is empty apart from registration lines.
- Problem sizes differ. The published sizes come from `reports/generation.json` and the harbor sizes from `PROBLEM_SIZE`.

| Task | Published n | Harbor n | Published baseline (ms) |
|---|---|---|---|
| eigenvalues_real | 875 | 960 | 124 |
| two_eigenvalues_around_0 | 1123 (+2) | 1151 (+2) | 100 |
| generalized_eigenvectors_real | 574 | 639 | 106 |
| generalized_eigenvalues_complex | 272 | 224 | 99 |
| generalized_eigenvectors_complex | 269 | 224 | 99 |
| psd_cone_projection | 349 | 447 | 102 |
| linear_system_solver | 1450 | 1279 | 102 |

- Input types differ between the two harnesses. When AlgoTune saves a dataset, `AlgoTuner/utils/serialization.py` converts nested numeric lists over 100 KB to ndarrays and reloads them with `np.load(mmap_mode="r")`. In the published runs, `problem["matrix"]` (two_eigenvalues_around_0) and `problem["A"]` (linear_system_solver) were therefore ndarrays. Harbor calls `generate_problem` in process, so both stay Python lists. The list-to-array conversion is then timed in both the baseline and the solver, so harbor speedups on these two tasks should come out lower than the published ones. The size of that effect needs to be measured.
- The harbor harness checks `is_solution` only on the warm-up output for each problem. It then takes the minimum of 10 interleaved repeats on the same problem object. A solver that cached its result by object identity would pass and look very fast, but no published solver does this.
- Run-to-run noise in the published numbers is large. Some pairs use the same route but score differently.

| Task | Pair with the same route | Ratio |
|---|---|---|
| eigenvalues_real | GPT-5.2 1.79 vs Opus 4.5 2.42 | 1.35x |
| psd_cone_projection | Gemini 3 Pro 12.02 vs GPT-5 8.77 | 1.37x |
| generalized_eigenvectors_real | R1 1.68 vs Gemini 2.5 Pro 3.19 | 1.89x |
| generalized_eigenvectors_complex | Opus 4.1 1.12 vs GPT-5.2 2.13 | 1.9x |
| linear_system_solver | Gemini 3.1 1.08 vs Gemini 3 Pro 27.4 | 25x |

  I treat single-pair ratios below about 1.3x as unresolved, and I rely on clusters where they exist. On two_eigenvalues_around_0, later models (Opus 4.5/4.6, Gemini 3/3.1, GPT-5.4) score 2.7-3.4x with the same shift-invert route on which earlier models score 1.7-1.9x. On generalized_eigenvalues_complex, the two groups agree to within 5%. I cannot tell whether this is a change of hardware, a change of library or noise.
- JAX solvers (R1 and Opus 4.1 on some tasks) call `jnp.asarray(..., dtype=jnp.float64)` without enabling `jax_enable_x64`. JAX then silently computes in float32, so I label them float32.

---

## eigenvalues_real

**Reference and verifier.** The reference calls `np.linalg.eigh(A)[0]`, which computes eigenvectors it then discards, and sorts descending with Python `sorted`. The verifier recomputes `eigh`, compares sorted values with a per-value relative error of at most 1e-6 (floor 1e-12), and checks length, finiteness and descending order with a 1e-6 slack. The check is strict, because float32 would fail on eigenvalues near zero.

| Model | Speedup | Components |
|---|---|---|
| GPT-5 | 2.524 | eigvals only (scipy eigh eigvals_only, driver evr); F-order copy + overwrite_a; check_finite=False; closed form n<=3 |
| R1 | 2.516 | eigvals only (scipy eigh, evr) |
| Opus 4.1 | 2.508 | eigvals only (scipy eigvalsh, default evr); reverse slice instead of sort |
| GLM-4.5 | 2.507 | `eigsh(k=n)`, which SciPy redirects to dense `eigh(eigvals_only=True)` (evr) |
| Qwen3 Coder | 2.495 | scipy eigvalsh (evr) + np.sort |
| GPT-OSS-120b | 2.490 | np.linalg.eigvalsh (syevd) |
| GPT-5-mini | 2.473 | np.linalg.eigvalsh |
| o4-mini | 2.466 | scipy eigvalsh, driver evr |
| GPT-5 Pro | 2.439 | np.linalg.eigvalsh |
| Opus 4 | 2.425 | scipy eigh eigvals_only (evr) + np.sort |
| Opus 4.6 | 2.422 | np.linalg.eigvalsh; closed form n<=3 |
| Opus 4.5 | 2.419 | np.linalg.eigvalsh |
| Gemini 2.5 Pro | 2.415 | scipy eigvalsh evr, overwrite_a, check_finite=False; `sklearnex.patch_sklearn()`, which patches scikit-learn only and does not affect SciPy |
| Gemini 3.1 Pro | 2.383 | direct LAPACK dsyevr, compute_v=0 |
| Gemini 3 Pro | 2.162 | torch.linalg.eigvalsh |
| GPT-5.4 | 1.889 | picks among numpy/scipy/dsyevd/dsyevr by microbenchmark on n<=64 test matrices; tridiagonal check |
| GPT-5.2 | 1.791 | np.linalg.eigvalsh; closed form n<=3 |

**Natural experiments**
- evr vs evd: R1 2.516 (scipy evr) vs Opus 4.5 2.419 (numpy syevd) gives 1.04x.
- Direct LAPACK vs SciPy wrapper: Gemini 3.1 2.383 (dsyevr) vs o4-mini 2.466 (eigvalsh evr) gives 0.97x.
- torch vs numpy: Gemini 3 Pro 2.162 vs Opus 4.5 2.419 gives 0.89x.
- np.sort vs reverse slice: Opus 4 2.425 vs Opus 4.1 2.508 gives 1.03x.
- Identical code: GPT-5.2 1.791 vs Opus 4.5 2.419 differ by 1.35x. This is noise, not a component.

**Class S.** Every solver gets its gain from computing eigenvalues without eigenvectors. No secondary component exceeds 1.04x.

**Validity.** All solvers compute exact float64 results. Gemini 3.1 passes `overwrite_a=1` on the problem array itself. This is safe only because the array is C-contiguous, so f2py copies it. An F-contiguous input would be overwritten.

**Single change to test.** Replace `np.linalg.eigh(problem)[0]` with `np.linalg.eigvalsh(problem)` and return `w[::-1].tolist()`.

---

## two_eigenvalues_around_0

**Reference and verifier.** The reference builds an array with `np.array(problem["matrix"])` (a list in harbor), calls dense `np.linalg.eigvalsh`, and sorts all n values by |x| in Python. The verifier recomputes `eigvalsh` and checks that the two values match the two nearest zero with `np.allclose(rtol=1e-5, atol=1e-8)`. Only these two values are checked.

| Model | Speedup | Components |
|---|---|---|
| Gemini 3 Pro | 3.448 | shift-invert Lanczos with an explicit in-place dgetrf (sigma=1e-6) and dgetrs as LinearOperator; ARPACK k=2, ncv=5, tol=1e-3, random v0; F-order input |
| Gemini 3.1 Pro | 3.317 | `eigsh(sigma=0, k=2, tol=1e-3, ncv=5, v0=ones)` |
| GPT-5.4 | 2.991 | `eigsh(sigma=0, k=2, tol=1e-5, ncv=8, v0=ones)` |
| Opus 4.6 | 2.812 | `eigsh(sigma=0, k=4, tol=1e-7)`, no vectors |
| Opus 4.5 | 2.708 | `eigsh(sigma=0, k=2, ncv=20, tol=1e-7)`, no vectors |
| o4-mini | 1.920 | explicit in-place `lu_factor` + LinearOperator; eigsh k=2, ncv=8, tol=1e-4 |
| GPT-5-mini | 1.826 | `eigsh(sigma=0, k=2, tol=1e-8)` |
| Opus 4.1 | 1.798 | `eigsh(sigma=0, k=2, tol=1e-8)` |
| Qwen3 Coder | 1.791 | `eigsh(sigma=0, k=2)`, default tol |
| GLM-4.5 | 1.763 | `eigsh(sigma=0, k=2, tol=1e-6)` |
| Gemini 2.5 Pro | 1.749 | `eigsh(sigma=0, k=2)`, default tol |
| R1 | 1.698 | `eigsh(sigma=0, k=2, ncv=34, tol=1e-4)` |
| Opus 4 | 1.666 | `eigsh(sigma=0, k=4)` with eigenvectors; numba selection |

Below 1.1 (controls): GPT-5 1.061, GPT-5.2 1.025, GPT-5 Pro 1.022 and GPT-OSS 1.002. All four keep dense eigvalsh and change only the selection (searchsorted or argpartition) or the driver (scipy evd).

**Natural experiments**
- Faster selection after the dense solve: GPT-5 gives 1.06x over the reference. Selection is not a route.
- ARPACK tuning in the later-model group: Gemini 3.1 (tol 1e-3, ncv 5, v0=ones) 3.317 vs Opus 4.5 (tol 1e-7, ncv 20) 2.708 gives 1.22x. GPT-5.4 (tol 1e-5, ncv 8) vs Opus 4.5 gives 1.10x.
- ARPACK tuning in the earlier-model group: o4-mini (in-place LU, tol 1e-4, ncv 8) 1.920 vs Gemini 2.5 (defaults) 1.749 gives 1.10x. Tolerance alone has no visible effect, since GLM (1e-6) scores 1.763, Opus 4.1 (1e-8) scores 1.798 and Qwen (default) scores 1.791.
- In-place LU instead of eigsh's internal copy: Gemini 3 Pro 3.448 vs Gemini 3.1 3.317 gives 1.04x.
- k=4 with eigenvectors vs k=2 without: Opus 4 1.666 vs Opus 4.1 1.798 gives 1.08x.
- The same route scores differently across model groups. Opus 4.5 2.708 vs Opus 4.1 1.798 is 1.5x apart with near-identical calls.

**Class S.** All five solvers within 1.5x of the best use the same switch, an LU factorisation plus a handful of triangular solves (shift-invert Lanczos around 0) in place of a full tridiagonal reduction. Tuning adds at most 1.22x. How much the switch is worth depends on which group's runs you trust, at about 1.8x (earlier) or 2.7x (later).

**Validity**
- Gemini 3 Pro and Gemini 3.1 use ARPACK tol=1e-3. GPT-5.4 uses 1e-5, and o4-mini and R1 use 1e-4. Lanczos Ritz values are usually far more accurate than the residual tolerance, so these normally meet rtol 1e-5. This is not guaranteed, though, and ncv=5 also raises the chance of converging to the wrong second eigenvalue on a near-tie.
- These solvers rely on the verifier's tolerance. Whether they pass all 100 harbor instances at n=1153 needs checking.
- Gemini 3 Pro's random v0 makes the iteration count, and so the timing, vary between repeats. Harbor checks only the warm-up output.
- In harbor the matrix arrives as a list of about 1.33M floats. Every solver and the baseline pay for the conversion, so the published 3.4x will compress by an amount that needs measuring.

**Single change to test.** Replace `np.linalg.eigvalsh(matrix)` with `scipy.sparse.linalg.eigsh(matrix, k=2, sigma=0, which='LM', return_eigenvectors=False)`. Test a second variant that adds `tol=1e-3, ncv=5, v0=np.ones(n)`.

---

## generalized_eigenvectors_real

**Reference and verifier.** The reference runs Cholesky, then `inv(L)` (a general inverse), then `Linv @ A @ Linv.T` (two GEMMs), then `np.linalg.eigh` with vectors, then `Linv.T @ V` (a third GEMM). It then runs a Python loop over n columns that B-normalises each with a `B @ v` matvec, and finally builds the output with per-column `tolist`. The verifier checks the B-norm with `np.isclose(B_norm, 1, atol=1e-6)` (default rtol 1e-5, so the effective tolerance is about 1.1e-5), checks each residual with `||Av - lam Bv|| / (||A||_F + ||B||_F) <= 1e-6`, and checks descending order. It does not compare eigenvalues with the true spectrum and does not check B-orthogonality or distinctness, so n copies of one valid eigenpair would pass. No published solver exploits this.

| Model | Speedup | Components |
|---|---|---|
| GPT-5.2 | 4.958 | scipy Cholesky + BLAS dtrsm x2 for L^-1 A L^-T (no inverse, no GEMMs) + np.linalg.eigh + dtrsm back-transform; no normalization loop; `.T.tolist()` |
| Gemini 2.5 Pro | 3.185 | float32 + scipy `eigh(A, B)` (sygvd), overwrite, check_finite=False, `.T.tolist()`; CuPy branch presumably inactive (no GPU) |
| Opus 4 | 1.919 | scipy `eigh(A, B)` (sygvd); per-column tolist |
| Gemini 3 Pro | 1.726 | float32 ssygvd, through Cython that also builds the lists, or through the SciPy wrapper as fallback |
| R1 | 1.682 | float32 + scipy eigh, driver gvd, overwrite |
| Gemini 3.1 Pro | 1.475 | scipy eigh gvd, overwrite, check_finite=False |
| Opus 4.6 | 1.474 | scipy `eigh(A, B)`, overwrite, check_finite=False |
| Opus 4.5 | 1.455 | direct dsygvd |
| GPT-5.4 | 1.452 | direct dsygvd on F-order copies |
| o4-mini | 1.429 | direct sygvd (get_lapack_funcs) |
| GPT-5 | 1.421 | scipy `eigh(A, B)` on F-order copies, overwrite |
| GLM-4.5 | 1.420 | scipy `eigh(A, B)`, check_finite=False |
| GPT-OSS-120b | 1.415 | scipy `eigh(A, B)` |
| Qwen3 Coder | 1.408 | scipy `eigh(A, B)` |
| Opus 4.1 | 1.361 | scipy `eigh(A, B)`, check_finite=False |
| GPT-5-mini | 1.278 | scipy `eigh(A, B)` + per-element `float()` list building |

GPT-5 Pro is N/A. Its code has a bug: `np.linalg.solve(L.T, X.T).T` equals L^-1 A L^-1, not L^-1 A L^-T, so the eigenpairs are wrong.

**Natural experiments**
- sygvd through a wrapper vs a direct LAPACK call: Opus 4.5 1.455 and GPT-5.4 1.452 (direct) vs Opus 4.6 1.474 and Gemini 3.1 1.475 (eigh) differ by at most 1.02x.
- float32 on sygvd: R1 1.682 vs GPT-5 1.421 gives 1.18x, and Gemini 3 Pro 1.726 vs Opus 4.5 1.455 gives 1.19x. Gemini 2.5 3.185 vs GPT-5 gives 2.24x, but R1 runs nearly the same float32 call and scored 1.89x lower.
- Manual reduction (TRSM + numpy eigh) vs sygvd: GPT-5.2 4.958 vs the later-model sygvd cluster (1.45-1.48) gives 3.4x. This rests on one sample, and that sample changes several things at once.
- Output conversion: GPT-5-mini (per-element float) 1.278 vs GPT-5 1.421 gives 1.11x. Opus 4 (per-column tolist, nominally slower) 1.919 vs Opus 4.1 (`.T.tolist()`) 1.361 points the wrong way, so it is noise.

**Class U.** The one-call sygvd route is consistently worth about 1.45x. The best solver gets about 3.4x more by keeping the reference's own numpy eigh and changing everything around it. It deletes the per-column normalization loop (n BLAS-2 matvecs plus Python overhead, which scales as n^3 at memory speed), replaces inv plus three GEMMs with three TRSMs, and builds the output with one `.T.tolist()`. The published data cannot say which of these matters. Float32 adds 1.18-1.19x in two of three samples.

What would decide it is timing these variants at harbor n=639:
- (a) the reference with only the normalization loop deleted
- (b) the reference with inv and GEMMs replaced by TRSM
- (c) scipy `eigh(A, B)`
- (d) GPT-5.2's solver
- (e) (d) in float32

If (a) alone gets most of (d), the class is S. If (a) and (b) are each worth 1.3x or more, it is M. The output must be a list of n lists, so every solver also pays for n^2 Python floats, which sets a floor on the speedup.

**Validity.** The float32 solvers (Gemini 2.5, R1, Gemini 3 Pro) rely on the roughly 1.1e-5 B-norm tolerance. Their B-orthonormality error grows with n, so check them on all 100 instances at n=639. Their residuals sit comfortably inside 1e-6 relative. Deleting the normalization loop is exact in principle, because x = L^-T v is B-normalised whenever v is orthonormal.

**Single change to test.** Delete the normalization loop and return `eigenvectors.T.tolist()`. Next, replace `inv(L)` and the GEMMs with `scipy.linalg.blas.dtrsm` or `solve_triangular`, as GPT-5.2 does.

---

## generalized_eigenvalues_complex

**Reference and verifier.** The reference scales A and B by sqrt(||B||_F) and calls `scipy.linalg.eig(A, B)`, which is QZ (ggev) with right eigenvectors that it discards. It then sorts with a Python key. The verifier checks length, finiteness and order (by re-sorting). For each value it checks that the smallest singular value of A - lam B is at most 1e-6 (||A||_F + ||B||_F). That is a backward-error test, not a comparison with reference eigenvalues. It does not check multiplicity or coverage, so one eigenvalue repeated n times would pass. No published solver exploits this.

| Model | Speedup | Components |
|---|---|---|
| Opus 4.1 | 5.389 | B^-1 A via JAX inv + standard eigvals; float32 (implicit, x64 off); lexsort |
| R1 | 5.264 | JAX lu_factor/lu_solve + eigvals; float32 (implicit) |
| Gemini 3.1 Pro | 4.131 | torch solve + eigvals, float64 |
| Opus 4.5 | 3.863 | scipy lu_factor/lu_solve + la.eigvals, float64 |
| GPT-5.2 | 3.849 | np.linalg.solve + np.linalg.eigvals; lexsort |
| GPT-5.4 | 3.840 | np.linalg.solve + eigvals |
| GPT-5-mini | 3.716 | np.linalg.solve + eigvals |
| o4-mini | 3.655 | np.linalg.solve + eigvals |
| GPT-OSS-120b | 3.631 | la.solve(check_finite=False) + np eigvals |
| Opus 4 | 3.490 | eigvals(A_s @ inv(B_s)) (similar matrix), numba norm |
| Gemini 3 Pro | 2.751 | float32 QZ without vectors (Cython sggev), preallocated F buffers |
| GLM-4.5 | 2.706 | float32 `la.eigvals(A, B)` (QZ, no vectors) |
| Gemini 2.5 Pro | 1.962 | `la.eig(right=False)` QZ, no vectors. The JAX branch depends on the removed `jax.config` import, and its speedup matches the QZ cluster, so it was probably inactive. |
| Opus 4.6 | 1.952 | `la.eigvals(A_s, B_s)` QZ, no vectors |
| Qwen3 Coder | 1.938 | `la.eigvals(A, B)`, no scaling |
| GPT-5 | 1.937 | `la.eigvals(A, B)` (n>64) |
| GPT-5 Pro | 1.933 | `la.eig(left=False, right=False)` |

**Natural experiments** (all solvers drop the eigenvectors)

| | float64 | float32 |
|---|---|---|
| QZ (ggev, no vectors) | 1.93-1.96 (GPT-5, GPT-5 Pro, Qwen, Gemini 2.5, Opus 4.6) | 2.71-2.75 (GLM, Gemini 3 Pro) |
| B^-1 A + standard eig (geev) | 3.49-4.13, median about 3.78 (9 solvers) | 5.26-5.39 (R1, Opus 4.1; Sonnet 4.5 5.59) |

- eig to eigvals (no vectors) is worth about 1.95x over the reference.
- QZ to reduction is worth 1.95x in float64 (3.78/1.94) and 1.95x in float32 (5.33/2.73).
- float64 to float32 is worth 1.40x on QZ (2.73/1.94) and 1.41x on the reduction (5.33/3.78).
- inv vs solve: Opus 4 3.490 vs o4-mini 3.655 gives 1.05x. torch vs numpy: Gemini 3.1 4.131 vs GPT-5.4 3.840 gives 1.08x.
- Both model groups agree here. Opus 4.6 1.952 matches GPT-5 1.937, and Opus 4.5 3.863 matches o4-mini 3.655.

**Class M.** Three components multiply, eig to eigvals (1.95x), QZ to B^-1 A + geev (1.95x) and float32 (1.40x). Their product is 5.3x, which matches the best solvers.

**Validity.** The verifier measures backward error. At harbor n=224 the absolute threshold on the smallest singular value is roughly 3e-5, and a backward-stable float32 eig gives errors of about eps32 times the norm. Float32 should therefore pass on essentially all instances, but this should still be confirmed locally. The reduction is safe because B's singular values lie in [1, 2] by construction. The float32 in the JAX solvers is accidental, since the code asks for float64.

**Single change to test.** Replace `eigenvalues, _ = la.eig(A_scaled, B_scaled)` with `eigenvalues = np.linalg.eigvals(np.linalg.solve(B, A))`, which should give about 3.8x published. A second step casts A and B to float32 first, which should reach about 5.3x.

---

## generalized_eigenvectors_complex

**Reference and verifier.** The reference calls `la.eig(A_s, B_s)` (QZ with right vectors), normalises the vectors in a Python loop, sorts zipped pairs with a key, and builds the output with `[list(vec) ...]`, which creates n^2 numpy complex scalars. The verifier checks unit norm (`np.isclose`, atol 1e-6), order, and each residual against `||Av - lam Bv|| / (||A||_F + ||B||_F) <= 1e-6`. It does not check coverage or distinctness. No published solver exploits this.

| Model | Speedup | Components |
|---|---|---|
| R1 | 3.361 | JAX float32 (implicit) `solve(B, A)` + eig, lexsort inside jit; relies on geev's unit-norm vectors |
| Gemini 3.1 Pro | 2.730 | np.linalg.solve + np.linalg.eig; vectorized normalization; argsort; `.T.tolist()` |
| Gemini 3 Pro | 2.699 | float32 QZ `la.eig(A, B)`; vectorized normalization; lexsort; `.T.tolist()` |
| Gemini 2.5 Pro | 2.695 | la.solve + la.eig(C); vectorized |
| GPT-5.4 | 2.547 | np.linalg.solve + eig; lexsort; no renormalization |
| Opus 4.5 | 2.538 | la.solve + la.eig(C); numba normalization |
| GPT-OSS-120b | 2.490 | np solve + eig; no renormalization |
| Qwen3 Coder | 2.455 | np solve + eig; vectorized |
| GPT-5 | 2.452 | lu_factor/lu_solve + la.eig(C); vectorized |
| o4-mini | 2.449 | np solve + eig; no renormalization |
| GPT-5-mini | 2.412 | np solve + eig; sampled symmetry checks |
| GPT-5.2 | 2.127 | QZ float64; vectorized normalization, lexsort, `.T.tolist()` |
| GLM-4.5 | 1.122 | QZ float64; allclose symmetry check; numba normalization; per-row tolist |
| Opus 4.1 | 1.115 | QZ float64; vectorized normalization, lexsort, `.T.tolist()` |
| GPT-5 Pro | 1.103 | QZ float64 with copies and scaling; vectorized post-processing |

Controls: Opus 4 scored 1.016 (QZ float64 with vectorized post-processing). Opus 4.6 is N/A (a direct dggev solver whose failure is not visible in the code).

**Natural experiments**
- Vectorizing the post-processing while keeping QZ64: Opus 4.1 1.115 and GPT-5 Pro 1.103 (earlier group) vs GPT-5.2 2.127 (later group). The route is the same but the scores are 1.9x apart. The reference's extra Python work is O(n^2) element handling plus an O(n) loop, which should be small next to O(n^3) QZ at n=269. That fits the 1.10-1.12 cluster better than 2.13, but it needs measuring.
- QZ64 to reduction (both vectorized): the earlier-group median of 6 solvers is 2.45 against 1.11, giving 2.2x. In the later group, Opus 4.5 2.538, GPT-5.4 2.547 and Gemini 3.1 2.730 against GPT-5.2 2.127 give 1.20-1.28x. The reduction scores agree across groups, and only the QZ baseline differs.
- float32 on the reduction: R1 3.361 vs the earlier-group reduction64 median 2.45 gives 1.37x, with JAX also in the mix.
- float32 on QZ: Gemini 3 Pro 2.699 vs GPT-5.2 2.127 gives 1.27x.
- Skipping renormalization (geev already returns unit vectors): o4-mini 2.449 and GPT-OSS 2.490 vs Qwen 2.455 show no difference.

**Class M.** The QZ-to-reduction switch (2.2x on earlier-group numbers, at least 1.2x on later) and float32 (1.27-1.37x) act on different things, the factorisation versus the precision. The best solver, R1, stacks both. Vectorizing the Python post-processing is a third, separate component of uncertain size (1.1x or 2.1x, depending on which run you trust).

**Validity.** R1's float32 relies on the residual and norm tolerances. The residual check is a backward-error test with an absolute threshold of roughly 3e-5 at n=224, and geev normalizes in float32 to about 1e-7. It should pass, but confirm on all instances.

**Single change to test.** Replace `la.eig(A_scaled, B_scaled)` with `np.linalg.eig(np.linalg.solve(B, A))` and keep the reference's post-processing. Separately, test the reference with only the post-processing vectorized (GPT-5.2's code) to settle the 1.1x versus 2.1x question.

---

## psd_cone_projection

**Reference and verifier.** The reference calls `np.linalg.eig(A)`, a general non-symmetric eig with vectors (geev), on a symmetric matrix. It clips negative eigenvalues and reconstructs with `V @ np.diag(w) @ V.T`, two full GEMMs. The verifier checks symmetry (allclose rtol 1e-5), PSD (eigvals at least -1e-5) and the objective ||A - X||_F^2 (isclose rtol 1e-5 vs the reference). The objective test is loose, so an X with a Frobenius error of order sqrt(1e-5 times the objective) passes. Approximate or float32 projections would be accepted.

| Model | Speedup | Components |
|---|---|---|
| Gemini 3 Pro | 12.023 | np eigh; keep positive-eigenvalue columns; `(V*w) @ V.T`. The folder has `fast_reconstruct.pyx` but no setup.py, so the Cython path was probably not built. If built, it is a naive triple loop. |
| GPT-5.2 | 9.345 | np eigh; positive subspace; sqrt-scaled columns + dsyrk upper triangle + mirror |
| Opus 4.6 | 9.058 | scipy eigh evd, overwrite, check_finite=False; positive subspace |
| o4-mini | 8.960 | np eigh; sqrt scaling; full `V @ V.T` |
| GPT-5 Pro | 8.883 | np eigh; positive subspace |
| Opus 4.5 | 8.822 | scipy eigh evd; positive subspace |
| Gemini 3.1 Pro | 8.811 | numba-jitted eigh + sqrt scaling + dot |
| GPT-5 | 8.768 | np eigh; positive subspace; early exits |
| GPT-5.4 | 8.688 | np eigh; positive subspace |
| GLM-4.5 | 8.649 | numba eigh + broadcast reconstruction |
| Opus 4 | 8.457 | np eigh; full `(V*w) @ V.T` |
| GPT-OSS-120b | 8.264 | np eigh; full |
| Gemini 2.5 Pro | 8.106 | np eigh; full; returns `X.tolist()` |
| Qwen3 Coder | 7.959 | scipy eigh (evr); full |
| R1 | 2.879 | JAX eigh, float32 (implicit); symmetrize twice; tolist |
| GPT-5-mini | 2.520 | symmetrize; np eigh; positive subspace; symmetrize; tolist |
| Opus 4.1 | 2.460 | np eigh; `V @ (w[:,None] * V.T)`; tolist |

**Natural experiments**
- Positive subspace vs full reconstruction: GPT-5 8.768 vs Opus 4 8.457 gives 1.04x, and GPT-5 Pro 8.883 vs GPT-OSS 8.264 gives 1.07x.
- syrk vs GEMM on the subspace: GPT-5.2 9.345 vs GPT-5.4 8.688 gives 1.08x.
- Driver choice: scipy evd (Opus 4.5 8.822) vs numpy (GPT-5 Pro 8.883) gives 0.99x, and scipy evr (Qwen 7.959) vs numpy evd (Opus 4 8.457) gives 0.94x. Numba eigh (GLM 8.649) vs numpy (Opus 4) gives 1.02x.
- Gemini 3 Pro 12.023 vs GPT-5 8.768 use effectively the same route, so the 1.37x gap is noise.
- Returning nested lists: Opus 4.1 (2.460), GPT-5-mini (2.520) and R1 (2.879) all return lists and score about 3x lower than ndarray returns. Gemini 2.5 also returns `X.tolist()` and scores 8.106, so the published data do not establish a list-return penalty. Harbor times inside the process, so this needs measuring there.

**Class S.** Every solver within 1.5x of the best gets its gain from replacing eig with eigh. No other component adds more than 1.08x. No published solver isolates the reconstruction change (np.diag GEMM to broadcasting), because all of them made it together with eigh.

**Validity.** All solvers are exact float64 except R1 (float32), which passes under the loose objective tolerance.

**Single change to test.** Replace `np.linalg.eig(A)` with `np.linalg.eigh(A)`. As a second step, replace `eigvecs @ np.diag(eigvals) @ eigvecs.T` with `(eigvecs * eigvals) @ eigvecs.T`, which removes one n^3 GEMM. Its effect at n=447 is unmeasured.

---

## linear_system_solver

**Reference and verifier.** The reference calls `np.array(A)` (a nested list in harbor, an mmap ndarray in the published runs), then `np.linalg.solve`, then `tolist()`. The verifier checks `||Ax - b|| <= 1e-6 (1 + ||b||)`.

| Model | Speedup | Components |
|---|---|---|
| Gemini 3 Pro | 27.428 | Cython: `np.array(A, order='F')` + LAPACK dgesv through scipy cython_lapack + tolist |
| GPT-5 | 1.133 | F-order array + direct dgesv with overwrite (the pure-Python elimination for n<=64 is inactive) |
| o4-mini | 1.122 | F-order arrays + np.linalg.solve |
| Opus 4.1 | 1.111 | scipy lu_factor/lu_solve, check_finite=False |
| GLM-4.5 | 1.106 | np.asarray + np.linalg.solve |

Controls below 1.1:
- GPT-5.4 1.097 (F-order + dgesv, returns ndarray)
- Gemini 2.5 1.089 (F-order dgesv, overwrite)
- Gemini 3.1 1.076 (Cython dgesv, the same design as Gemini 3 Pro)
- R1 1.065, GPT-OSS 1.056, Opus 4 1.039, GPT-5.2 1.003, Opus 4.5 0.993 (all np.linalg.solve variants)
- Opus 4.6 N/A

**Natural experiments**
- F-order with a direct dgesv vs np.linalg.solve: GPT-5 1.133 vs R1 1.065 gives about 1.06x.
- Gemini 3 Pro 27.428 vs Gemini 3.1 1.076 use the same design (F-order copy, Cython dgesv, tolist) but are 25x apart. 27.4x on the published 101.7 ms baseline means 3.7 ms per solve, or about 550 GFLOP/s for the 2.0 GFLOP LU at n=1450. The baseline itself implies about 21 GFLOP/s for the same LAPACK routine.

**Class U.** No published route beyond about 1.13x reproduces, and the 27.4x outlier cannot be attributed to any code component. Rebuilding and timing Gemini 3 Pro's `solver_impl.pyx` and Gemini 3.1's `solver_cy.pyx` side by side would decide it. If both come out near 1.1x, the task has no meaningful route under this harness, because LU is already LAPACK and harbor also times the conversion of a nested list of about 1.6M floats.

**Validity.** Nothing in the published solvers relies on tolerance. GPT-5.4 returns an ndarray instead of a list, which the verifier accepts.

**Single change to test.** None reproduces the best number. The reproducible ceiling is `np.array(A, order='F')` + `scipy.linalg.lapack.dgesv(..., overwrite_a=1, overwrite_b=1)`, which scored about 1.1x published.

---

## Follow-ups noticed (not acted on)

- Three verifiers (generalized_eigenvalues_complex, generalized_eigenvectors_complex, generalized_eigenvectors_real) check each eigenpair in isolation and never check coverage or distinctness. One valid eigenpair repeated n times would pass. Harbor checks only the warm-up output and repeats the same problem object, so result caching keyed on the problem would also pass. No published solver does either, but a local experiment should screen for both.
- An idea not seen in any published solver, for linear_system_solver, is LU in float32 followed by one or two float64 iterative-refinement steps. It could beat dgesv on the factorisation, but harbor's list conversion would bound any gain.
- Several solvers pass `overwrite_a=True` on the caller's problem arrays (Gemini 3.1 on eigenvalues_real, Opus 4.5 on psd_cone_projection, Gemini 2.5 on generalized_eigenvalues_complex). They are safe only because the inputs are C-contiguous, so f2py copies them. A harness that passed F-contiguous arrays would corrupt the problem before verification.
