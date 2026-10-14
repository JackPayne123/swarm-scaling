# v6 anomalies and selection regret (analysis 2026-10-13)

Read-only analysis. No scorer jobs, no cloud resources, no API spend. Local checks ran the Harbor least_squares generator and verifier (`evaluator.py` from the inspect_harbor cache) in a scratch venv on the Mac (Python 3.12, numpy 2.5.4, scipy 1.18.1, numba 0.68.0), not in the task image. Scratch scripts and outputs are in `scratchpad/v6/`.

## 1. least_squares r1, loose arm: why the 40x pick was invalid on hidden

**The pick.** sha256 `2bd60aced442...`, published by `v6-lsq-r1-loose-i0` agent_0 (also its final workspace). Stage-1 dev 60.17x (20 instances), stage 2 39.99x valid (100 instances), hidden 1.00 invalid in all three scorings.

**What it does.** A numba Levenberg-Marquardt fitter with a grid of starting points. The shortcut is at the top of `solve`: when the problem has more than 2,000 points it keeps every `ceil(n/2000)`-th point and fits only that subsample, with no pass over the full data. The verifier uses n = 98,304, so the solver fits 1,967 points (stride 50) and returns those parameters. The same agent's earlier candidate `a2cf9473ae83` has identical fitting code without the input subsampling and scored 1.16x.

**Why it fails.** `is_solution` accepts parameters whose full-data MSE is at most 1.05 times the MSE of `scipy.optimize.leastsq` started from a fixed guess. At this size x runs from 0 to 9,831 in steps of 0.1, so the stride-50 subsample has spacing 5.0005 and a Nyquist frequency of 0.628, while the generator draws the sinusoid frequency b from U(0.1, 2). On the subsample, every sinusoid has exact aliases (b' = 2π/5.0005 - b and so on), and an alias fits the subsample as well as the true frequency does. When the solver's frequency grid (120 values spaced 0.021) lands within roughly 3e-4 of an alias, LM converges to it and returns a full-amplitude sinusoid at the wrong frequency. On the full data that fit is uncorrelated with the signal, so its MSE is about var(y) + a²/2. The reference never fits sinusoids at this size either: it converges near b = 1 with amplitude near 0, a flat line with MSE about var(y). The failure ratio is therefore about 1.9, far from the 1.05 tolerance. Example: the failing hidden instance (index 35 of the 100) has true b = 1.0067; the solver returned b = 0.2498 (= 1.2565 - 1.0067), amplitude 3.69 against a true 3.67, MSE ratio 1.81. A few sigmoid instances fail the same way when the transition is narrower than the subsample spacing (one fit returned a = 4.4e9).

**Classification.** A deliberate shortcut that is invalid on a measurable fraction of instances, not a tolerance edge and not a coding bug in the usual sense. Valid instances sit far from the tolerance (MSE ratio quantiles over 20,000 instances: median 1.001, 99th percentile 1.010, max 1.046), and failures sit at 1.3-1.97. It passes most instances because the subsample fit of smooth models loses under 1% MSE and because the reference's sinusoid fit is no better than a flat line.

**Failure rate (measured locally).**

| Instance set | Instances | Invalid |
|---|---|---|
| Finalize dev (seeds 60,000-60,019) | 20 | 0 |
| Stage 2 (seeds 500,000-500,099) | 100 | 0 |
| Hidden (the 100 verifier seeds) | 100 | 1 (index 35, sinusoidal) |
| Fresh seeds 2,000,000-2,019,999 | 20,000 | 249 (1.245%) |

By model type on the 20,000: sinusoidal 223/3,954 (5.6%), sigmoid 25/4,024, logarithmic 1/4,069, polynomial and exponential 0. The local harness reproduced every scorer outcome I could check: this pick valid on stage 2 and invalid on hidden, plus the stage-2 and hidden validity of seven other shortlisted least_squares sources (section 3 lists them). With p = 1.245% per instance, passing the 120 dev instances had probability 0.22, and failing at least one of the 100 hidden instances given that had probability 0.71. Stage 2 at 100 instances catches a solver with this rate 71% of the time.

**Other arms.** I ran all 54 least_squares sources (all arms, both repeats) on 1,000 fresh instances plus the 249 failing ones. Only four sources fit on a strided subsample of the input with no full-data pass: this pick, and three from one diversified-independent agent, `v6-lsq-r2-div-i0` (`24becded15e7`, `15223374a8e6`, `593dde616361`, cap 4,096 points, stride 25). `24becded15e7` failed 14/1,000 (13 sigmoid, 1 sinusoidal); it was rank 1 in lsq-r2-div at 28.42x dev, and stage 2 caught it (3 of 100 invalid locally, 1.00 on the scorer), so the div pick fell to rank 2 (7.70 hidden). Its siblings failed 1/1,000 and 0/1,000 and were not shortlisted. No team source used the trick. Most other sources failed 0-2 of 1,000. Separate weaker failure modes exist: `121a6bce325b` (lsq-r1 team, rank 2) fails logarithmic instances (9/1,000) and was invalid on stage 2 and hidden; four r1 sources that never reached a shortlist fail 130-219 exponential instances per 1,000. `fb0ffab8927b` (lsq-r2 team final workspace) hangs on some sinusoidal instances; finalize recorded it as killed, so it never entered stage 1.

## 2. pca r2, team arm: pick 13.08 on stage 2, 7.25 hidden median

All scorings use n = 24 (X is 1,200 x 240, 120 components), so instance size does not vary between stage 2 and hidden. The gap is a two-state timing behaviour of this source.

| Job | Source | Score | Reference total (s) | Reference alone (s) | Solver total (s) | Reference inflation | First-call ratio |
|---|---|---|---|---|---|---|---|
| stage 2 | pick `7599d2996614` | 13.08 | 12.41 | 11.20 | 0.949 | 1.108 | 1.34 |
| hidden 1 | pick | 7.25 | 9.09 | 11.97 | 1.254 | 0.760 | 8.10 |
| hidden 2 | pick | 12.72 | 12.07 | 11.68 | 0.949 | 1.033 | 1.26 |
| hidden 3 | pick | 7.06 | 8.89 | 11.29 | 1.258 | 0.787 | 7.45 |
| stage 2 | `7340074b6bb1` | 12.12 | 11.55 | 11.42 | 0.953 | 1.012 | 1.26 |
| hidden | `7340074b6bb1` | 12.75 | 11.71 | 11.75 | 0.918 | 0.997 | 1.31 |
| stage 2 | `950d6c396eed` | 7.48 | 8.92 | 12.10 | 1.191 | 0.737 | 8.35 |
| hidden | `950d6c396eed` | 7.16 | 8.73 | 11.82 | 1.219 | 0.739 | 7.86 |

"Reference alone" is reference total / inflation, the thread guard's child-process timing. It stays at 11.2-12.1 s in every job, so the machine was not slower. In two of the pick's three hidden runs the interleaved reference ran 24% faster than alone and the solver 32% slower, with a first call about 8 times its best timed call. Those two runs look the same as `950d6c396eed` in both of its runs.

**Not ordinary noise.** Other pca picks repeat within about ±6% (`0ee54d1d6fe6` 20.15-21.70, `70b781604490` 22.58-25.26). The two modes differ by 1.8x. Slot does not explain it: slots 2 and 6 gave the slow mode for the pick and the normal mode for other sources.

**Thread behaviour.** The slow mode follows how a solver calls numpy's own OpenBLAS-backed LAPACK. In the finalize records (20 dev instances, one scoring per source), all 14 pca sources that call `np.linalg` LAPACK routines (eigh, qr, cholesky) with no thread limit and no scipy path ran in the slow mode: inflation 0.66-0.86, first-call ratio 5.3-9.3. The sources that use `scipy.linalg`, or limit every OpenBLAS library around the solve, ran at inflation 0.83-1.27. The pick calls `np.linalg.eigh` and limits threads with threadpoolctl, but only on the first controller whose `internal_api` is openblas. numpy and scipy each bundle their own OpenBLAS, so this limits one of the two. `600e74831c24` (also team r2) limits only the first BLAS library and ran slow at finalize. `7340074b6bb1` limits every OpenBLAS library, has the same Gram plus eigh core, and ran normal in all three of its scorings. The pick ran normal at finalize, on stage 2 and in one hidden run, and slow in two hidden runs (2 of 5 scorings).

**Not determined.** (a) Why the pick switches mode between runs of the same container image. One candidate explanation is that the limit sometimes lands on scipy's library instead of numpy's, but threadpoolctl's library order should be fixed by import order, so I could not confirm this. (b) Why numpy's multithreaded LAPACK makes the next reference call faster. Testing either needs the scorer's x86 image and CPU, which this analysis did not use.

**Selection consequence.** In its normal mode the pick is no faster than `7340074b6bb1` (solver 0.949 s vs 0.953 s on stage 2). Its stage-2 lead of 13.08 vs 12.12 came from the reference side: inflation 1.108 vs 1.012. The pick won stage 2 on reference timing noise, then landed in the slow mode twice on hidden.

## 3. Selection regret across all 42 portfolios

Sources: the 36 surviving records in `analysis/portfolio/20261012*.jsonl`. The six overwritten portfolios (generalized_eigenvalues_complex r2 and pca r2, three arms each) were rebuilt from the eval logs (stage 1, via `select_portfolio.load_portfolios`/`stage1`) and `logs/_scorer/i-0d30a2ddff1fb1a5d/jobs.jsonl` (stage 2 and hidden). The rebuilt picks, stage-2 scores, hidden medians and oracle bounds match the tables in `logs/_plans/reselect-v6-*.log` exactly, and all 42 pick medians match `analysis/v6-main-results.md`. The oracle bound compares the pick's median of 3 with single scorings of the other shortlisted sources, as `select_portfolio.py` labels it.

### Per arm

| Measure | Loose indep. | Diversified indep. | Team |
|---|---|---|---|
| Portfolios where the pick was not the hidden-best in its shortlist | 3/14 | 2/14 | 1/14 |
| Geometric-mean regret (oracle / pick hidden) | 1.125 | 1.001 | 1.041 |
| Largest regret | 3.89 (lsq r1) | 1.006 | 1.76 (pca r2) |
| Geometric mean of pick hidden / pick stage 2 | 0.756 (min 0.025) | 0.970 (0.83-1.03) | 0.939 (min 0.55) |
| Shortlist pairs ordered the same by stage 2 and hidden | 31 of 41 (1 tie) | 39 of 42 | 39 of 42 |
| Shortlisted sources invalid on stage 2 / on hidden | 2 / 1 | 1 / 1 | 2 / 1 |
| Distinct valid candidates, mean (median) | 6.9 (6.5) | 8.6 (9) | 8.6 (8) |

The loose regrets are lsq r1 (3.89: invalid pick), lsq r2 (1.33: the stage-1 ranks 1 and 2 were invalid on stage 2, each with one failing sigmoid instance that I reproduced locally, but valid on hidden at 2.19 and 2.05, so the pick was rank 3 at 1.65) and pca r1 (1.012). The div regrets are 1.006 and 1.001. The team regret is pca r2 (section 2). All six validity disagreements between stage 2 and hidden are least_squares sources. Selection hurt the loose arm most, mostly through least_squares validity on 100-instance samples. It hurt the team arm once, through the pca timing modes.

### Per group (hidden pick / oracle bound / best stage-1 dev / distinct valid)

| Group | Loose | Diversified | Team |
|---|---|---|---|
| clustering_outliers r1 | 7.52 / 7.52 / 7.31 / 5 | 5.39 / 5.39 / 5.21 / 10 | 6.41 / 6.41 / 6.22 / 13 |
| clustering_outliers r2 | 2.69 / 2.69 / 2.63 / 6 | 4.54 / 4.55 / 4.44 / 10 | 6.58 / 6.58 / 6.44 / 8 |
| generalized_eigenvalues_complex r1 | 3.57 / 3.57 / 3.85 / 6 | 6.44 / 6.48 / 6.41 / 10 | 6.35 / 6.35 / 6.65 / 4 |
| generalized_eigenvalues_complex r2 | 4.77 / 4.77 / 4.83 / 8 | 3.95 / 3.95 / 4.41 / 8 | 6.43 / 6.43 / 6.33 / 6 |
| least_squares r1 | 1.00 / 3.89 / 60.17 / 8 | 5.39 / 5.39 / 8.29 / 5 | 3.47 / 3.47 / 4.73 / 7 |
| least_squares r2 | 1.65 / 2.19 / 3.32 / 7 | 7.70 / 7.70 / 28.42 / 12 | 6.43 / 6.43 / 6.77 / 7 |
| pca r1 | 20.95 / 21.20 / 20.40 / 10 | 23.95 / 23.95 / 23.57 / 9 | 20.42 / 20.42 / 19.80 / 7 |
| pca r2 | 23.20 / 23.20 / 25.19 / 8 | 20.96 / 20.96 / 20.51 / 6 | 7.25 / 12.75 / 12.86 / 8 |
| sinkhorn r1 | 6.89 / 6.89 / 6.71 / 4 | 7.67 / 7.67 / 7.55 / 10 | 7.34 / 7.34 / 6.99 / 11 |
| sinkhorn r2 | 7.46 / 7.46 / 7.29 / 6 | 8.17 / 8.17 / 8.10 / 7 | 9.72 / 9.72 / 9.57 / 9 |
| sparse_lowest_eigenvalues_posdef r1 | 4.51 / 4.51 / 4.61 / 6 | 3.70 / 3.70 / 3.74 / 7 | 6.16 / 6.16 / 6.15 / 8 |
| sparse_lowest_eigenvalues_posdef r2 | 6.12 / 6.12 / 6.30 / 4 | 8.13 / 8.13 / 8.39 / 7 | 7.58 / 7.58 / 7.81 / 12 |
| shortest_path_dijkstra r1 | 34.67 / 34.67 / 47.68 / 10 | 44.06 / 44.06 / 65.45 / 11 | 39.11 / 39.11 / 56.20 / 10 |
| shortest_path_dijkstra r2 | 40.60 / 40.60 / 60.91 / 9 | 38.25 / 38.25 / 55.61 / 9 | 38.74 / 38.74 / 57.81 / 10 |

### Paired comparisons over the 14 groups

Wins, geometric mean of ratios, and the same t interval on log ratios as `v6-main-results.md` (it treats the two repeats of a task as independent).

| Measure | Team vs diversified | Team vs loose | Diversified vs loose |
|---|---|---|---|
| Pick hidden (headline) | 6/14, 0.97 (0.77-1.23) | 10/14, 1.33 (0.92-1.90) | 9/14, 1.36 (0.96-1.94) |
| Oracle bound | 6/14, 1.01 (0.85-1.21) | 9/14, 1.23 (0.96-1.57) | 9/14, 1.21 (0.95-1.53) |
| Best stage-1 dev | 7/14, 0.91 (0.69-1.21) | 9/14, 1.00 (0.62-1.60) | 8/14, 1.09 (0.67-1.78) |
| Pick stage 2 | 7/14, 1.01 (0.85-1.19) | 9/14, 1.07 (0.66-1.72) | 8/14, 1.06 (0.70-1.60) |
| Distinct valid candidates | 8/14, 0.98 (0.78-1.23) | 7/14 + 3 ties, 1.24 (0.93-1.65) | 9/14 + 2 ties, 1.26 (1.00-1.59) |
| Pick hidden, least_squares excluded (12 groups) | 6/12, 1.02 (0.79-1.33) | 8/12, 1.12 (0.82-1.53) | 7/12, 1.09 (0.91-1.31) |
| Oracle, least_squares excluded (12 groups) | 6/12, 1.07 (0.89-1.29) | 8/12, 1.17 (0.92-1.49) | 7/12, 1.09 (0.91-1.31) |

**Would conclusions change?** Team vs diversified independents shows no difference under every measure (ratios 0.91-1.07, wins 6-8 of 14). On oracle bounds, the diversified arms' lead over loose falls from 1.33-1.36 to 1.21-1.23, so selection regret on loose (the two least_squares groups) accounts for about a third of the headline log gap. On best stage-1 dev scores the lead nearly vanishes (1.00 and 1.09). That measure includes hidden-invalid sources (60.17 loose lsq r1, 28.42 div lsq r2), and on shortest_path_dijkstra the 20-instance dev scores run 35-60% above hidden, so I would not use it as an arm measure. On candidate counts, both diversified arms produced more distinct valid candidates than loose (about 1.25x); team and diversified independents produced the same number.

## Not determined

- The pca mechanism (section 2): why numpy-LAPACK solvers make the interleaved reference faster and themselves slower, and why the pick switches between modes.
- The scorer's job logs for the original main-run scorer (98.83.234.191) are not on disk, so the verifier output of the least_squares hidden failure was not available. The failing instance was found by local reproduction, which agreed with the scorer on every validity outcome I compared.
- Local validity used the Mac's numpy/scipy/numba builds, not the task image. Agreement with the scorer on 16 stage-2/hidden validity outcomes across 8 sources suggests the failure rates carry over, but they were not measured in the image.
- Hidden instances were generated locally using the seed offset in `data/.algotune_seed_offset`. Results stay in the session scratchpad.

## Follow-ups (not fixed)

- The thread guard flags only reference inflation above 1.15. Fourteen or more pca sources run in a mode where the interleaved reference is 15-35% faster than alone. This lowers those solvers' own scores (it is not an exploit), but it puts pca scores on two scales depending on which LAPACK a solver calls, and it decided the team pca r2 result.
- Stage 2 with 100 instances misses a solver that fails 1.2% of instances 29% of the time. least_squares produced six validity disagreements between stage 2 and hidden among 42 shortlisted sources.
- The AlgoTune mercy score turns one invalid instance into 1.0 for the whole run, which is what made lsq r1 loose 1.00 rather than 40.
