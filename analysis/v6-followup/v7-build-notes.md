# v7 harness build notes (2026-10-14)

Worktree `~/projects/swarm-scaling-v7`, branch `v7` (pushed to origin), 9 commits on top of main `482cb6d`:

| Commit | What |
|---|---|
| 1d50cb5 | swarm/team: `--end-on-budget`, `--elicit diversified-v7`, `--publish-notices`, `--tool-fixes` (items 2, 3, 7, 11) |
| 71e2a6c | per-agent boxes `--isolate-agents --agent-memory-mb` (item 1) |
| e365189 | seed ranges + `--disjoint-dev-seeds` (4), `--experiment v7` offset (10), alone score + `--primary-score` (6), validity job + selection pass (5), request ids, network retries, per-portfolio output (8b, 8d) |
| 34e8273 | cloud ops: `scorer.sh pull`/`allow-me`, retrying `down`, driver pulls every 10 min, operator IP refresh on SSH/scorer failures (8a, 8c), `--group-start` (9), experiment plumbing in scorer.sh/run_sample/vm_run (10) |
| f043a52 | `--validity-n-task TASK=N` (per-task validity size, after the timing below) |
| 6bd9655, f7a9b00 | PLAN.md v7 section + 11 decisions-log rows (2026-10-14); HARNESS.md v7 bullet |
| 81bbe1b, 36a0b14 | run_sample header doc; alone-score test made robust to timing noise |

Tests: 135 pass (119 before). Full suite run 7 times at the end: 5 clean, 2 with one failure each (once `test_thread_guard::test_the_alone_score...` before I widened its timings; once `test_scorer_service::test_a_job_past_its_timeout_has_only_its_own_container_killed`, which passed 5/5 alone and 5/5 in later full runs; it uses a 0.5 s timeout against a fake docker script, so I read it as load-sensitive, not caused by this change, but I did not prove that).

## v7 row flags

`--experiment v7 --elicit diversified-v7 --end-on-budget --isolate-agents --agent-memory-mb M --disjoint-dev-seeds --tool-fixes` (+ `--priority-index k` for independents; `--publish-notices` and `--primary-score alone` follow by default). Same M on every row of a comparison. Driver: `EXPERIMENT=v7 bash scripts/v6.sh <plan> <cap> <ref> --group-start --validity-n 1000 [--validity-n-task generalized_eigenvalues_complex=300]`.

M must fit the row's VM: `box_resources` refuses n x M + 2,048 MiB headroom (+ 8,192 for a local checker) above Docker's memory. An independent on c7a.xlarge (8 GiB) leaves roughly 5.5 GiB; sinkhorn needed m7a rows in v6 (9.6 GiB peak in the agent box), so M will differ by task and the plan must give each task's rows one value.

## Verified in the mock end-to-end runs (local Docker, mockllm, no API spend)

Script `scratchpad/v7_mock_e2e.py` (scripted agents, algotune_task + swarm wired as the runner does, local checker, pilot task dst_type_II_scipy_fftpack, token budget 10,000, dummy seed offset via ALGOTUNE_SEED_OFFSET so no secret was used or logged). Logs: `scratchpad/e2e2/{indep,team}/`, `scratchpad/e2e3/indep/`; reader `scratchpad/inspect_e2e.py`.

- Item 1: independent and team of 2 (2 CPUs, 4,096 MiB each). Every box: nproc 2, `os.cpu_count()` 2, `multiprocessing.cpu_count()` 2, affinity {0,1} (agent_1: {2,3}), OpenBLAS 2 threads in numpy's and scipy's libraries, numba 2 threads, memory.max 4 GiB, cpu.max 200000/100000, cpuset 0-1 / 2-3, same env. agent_1 saw agent_0's working dir and registry entry through the volumes. agent_0's `sleep 900 &` was killed in agent_0's box at cleanup ("killed 1"), agent_1's box had nothing to kill ("killed 0"). Volumes were gone after the run (`docker volume ls`). The affinity sets differ in which CPUs (disjoint cpusets), not in size: my reading of "the same cpuset".
- Item 2: no `submit` tool offered; a text-only turn got the continue message; both team agents and the independent ended with end_reason `token`, error None; finalize ran and installed the published candidate; scoring ran; no finish notice reached agent_1; wait_for_message after agent_0 ended said "... Continue working." (no "or finish").
- Item 3: agent_1's next tool result after agent_0's publish carried `[registry] agent_0 published agent_0-0 at /app/registry/agent_0-0 (dev_eval 1.01x on n=20, seed 0). Note: reference wrapper measured on dev_eval`; `swarm.registry_notices` logged it with `to: [agent_1]` and its delivery time.
- Item 7: SendMessage without summary was delivered (envelope `summary=""`); check_budget printed the percent remaining. The 4-decimal dollar display has a unit test only (mock runs use token budgets; mockllm is unpriced).
- Item 11: v7 prompt blocks present in every arm, V7_TEAM in the team, no TEAM_SHARING; run metadata `elicit diversified-v7`.
- Items 4, 6, 10 (e2e3, `experiment="v7"` with the dummy offset): dev_eval's seed description reads "seed + n below 50,000"; seed 49,990 n 20 was refused with "seed + n must be below 50,000 (got seed 49,990, n 20); use a smaller seed"; finalize recorded `alone_speedup` 1.0016 beside the interleaved 0.9974; the verifier printed "Alone raw speedup: 1.0119 x", "Alone score: 1.0119"; the run's Score.value was 1.0119 (alone, the v7 default) with `algotune_score` 1.0 (AlgoTune's 0.9965 raw, mercy 1.0) in metadata; task metadata recorded experiment v7, primary_score alone.

## Unit tests only (no live run)

- Validity job kind and the selection pass with backfill (fake scorer); validity_check.py itself never ran in a container. The timing below used my own measure script, not validity_check.py.
- Request-id dedup on a lost POST answer (real HTTP on localhost, fake runner); selection resume across a restart (by construction of the ids, not tested end to end).
- Alone-score ranking in stage 2 and hidden (fake scorer).
- Per-portfolio incremental writes (fake scorer).
- Group-start scheduling (fake launches).
- Per-experiment offset files and the 100-seed separation.
- Nothing cloud-side ran: `scorer.sh pull`/`allow-me`, the retrying `down`, `operator_ip_refresh` in `vm_exec`/`vm_get`, run_sample's experiment check and vm_run's offset install are untested beyond `bash -n`. The scorer's `--experiment` flag and /health field have no live run either.

## Validity pass cost (item 5)

Measured 2026-10-14 on the Mac's Docker (arm64 task image, `--cpuset-cpus 0-7`, the verifier's thread settings and TIMING_ENV), the reference standing in for a candidate, 10 instances per held-out task on seeds 600,000+. Estimate = 1,000 x (generate + solve + is_solution). Not measured on the scorer's x86 c7a CPUs; container start and the solver-vs-reference speed difference are not included (a fast candidate makes "solve" smaller).

| Task | generate ms | solve ms | is_solution ms | 1,000 instances, min |
|---|---|---|---|---|
| btsp | 1.1 | 2.7 | 2.2 | 0.1 |
| chebyshev_center | 1.3 | 255.8 | 242.8 | 8.3 |
| clustering_outliers | 1.5 | 85.5 | 90.0 | 3.0 |
| eigenvalues_real | 16.9 | 106.3 | 104.7 | 3.8 |
| fft_cmplx_scipy_fftpack | 61.1 | 104.3 | 145.9 | 5.2 |
| fft_real_scipy_fftpack | 50.4 | 105.8 | 339.2 | 8.3 |
| generalized_eigenvalues_complex | 19.9 | 99.1 | 5409.9 | 92.1 |
| generalized_eigenvectors_complex | 21.4 | 108.9 | 50.7 | 3.0 |
| generalized_eigenvectors_real | 89.1 | 451.8 | 243.9 | 13.1 |
| lasso | 110.9 | 88.0 | 101.8 | 5.0 |
| least_squares | 8.0 | 161.6 | 180.7 | 5.8 |
| min_dominating_set | 0.1 | 60.0 | 62.4 | 2.0 |
| min_weight_assignment | 33.2 | 112.6 | 123.1 | 4.5 |
| pca | 12.0 | 179.7 | 210.8 | 6.7 |
| qp | 4.6 | 113.2 | 119.9 | 4.0 |
| quantile_regression | 1.2 | 98.3 | 98.2 | 3.3 |
| randomized_svd | 12.1 | 296.2 | 22.5 | 5.5 |
| robust_linear_program | 0.8 | 116.6 | 109.5 | 3.8 |
| set_cover_conflicts | 0.5 | 40.5 | 49.3 | 1.5 |
| shortest_path_dijkstra | 2.5 | 74.6 | 34.7 | 1.9 |
| sinkhorn | 62.9 | 111.3 | 111.0 | 4.8 |
| sparse_lowest_eigenvalues_posdef | 36.3 | 125.2 | 98.1 | 4.3 |
| sparse_lowest_eigenvectors_posdef | 34.7 | 128.1 | 99.6 | 4.4 |
| tsp | 0.9 | 72.9 | 73.1 | 2.4 |
| two_eigenvalues_around_0 | 51.3 | 123.1 | 120.1 | 4.9 |
| unit_simplex_projection | 23.8 | 48.5 | 51.8 | 2.1 |

generalized_eigenvalues_complex's is_solution takes one SVD per eigenvalue, so 1,000 instances (about 92 min) pass the validity job's 3,600 s limit and every source would be dropped; use `--validity-n-task generalized_eigenvalues_complex=300` (about 28 min) or fewer, or raise `TIMEOUTS_S["validity"]`. Per portfolio the pass checks at least 3 sources (more when some fail), concurrently on scorer slots.

## Assumptions (readings taken)

- "Same cpuset as an independent": same size, disjoint CPUs per agent. "Same memory": an explicit `--agent-memory-mb` that the plan gives every row; required with `--isolate-agents`.
- `os.cpu_count()` is set to the box's CPU count in every agent box under `--isolate-agents` (independents included), by a sitecustomize written at setup; without it Python 3.12 reports the host's CPUs (4 on an independent VM, 16 on a team VM). Agents can still read /proc/cpuinfo.
- Disjoint seeds keep v6's refusal rule (seed + n at the cap is refused) with the cap at 50,000; the v6 default cap (400,000) stays unless `--disjoint-dev-seeds`.
- The validity pass uses one instance set (600,000 + i) for every source and arm ("its own seed range" read as the pass's own range). A failed validity job (crash, timeout) drops the source.
- Alone primary: stage 1 still ranks by finalize's interleaved dev score, and the dev_eval report agents read is unchanged (alone_speedup is recorded only), because the task named only stage 2 and hidden scoring.
- Publish notice dev score: the latest completed dev_eval of that exact file by any agent of the sample.
- `--group-start` groups by the first three dash-separated parts of the row name, as v6_drive did.
- The v7 offset was created once in `~/projects/swarm-scaling-v7/data/.algotune_seed_offset.v7` (mode 600, gitignored, checked at least 100 seeds from v6's). It must be copied to the main checkout's `data/` before running v7 from there; deleting the worktree deletes it.

## Follow-ups (found, not fixed)

1. `tasks.task_image_digest` has no retry; Docker Desktop answered `docker image inspect hb__...` with "No such image" for about a minute (the first e2e start failed on it; `docker inspect --type image` and fully qualified names worked). The scorer already wraps it (`image_digest_with_retry`); the runner path does not. Inspect's own prebuilt-image check uses the same call.
2. `tasks._offsets_in_logs` globs `logs/*.eval`, but runs write `logs/<name>/*.eval`, so restoring a missing offset from logs never finds one (the experiment filter I added inherits this).
3. `scripts/cpu_report.py` reads only `swarm.cpu`, which under `--isolate-agents` is agent_0's box; the other boxes are in `swarm.cpu_boxes`.
4. The task image has no `ps`, so agents' `ps` calls fail (seen in the mock run); not a harness bug, may matter for agents debugging background jobs.
5. Not built (deferred by the brief): non-blocking dev_eval.
