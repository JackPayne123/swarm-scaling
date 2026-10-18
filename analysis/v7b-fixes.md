# v7b fixes after the v7 calibration (2026-10-18)

Branch `v7b` (worktree `~/projects/swarm-scaling-v7b`), 15 commits on main `907a000`, pushed, not merged. It fixes the harness problems that `analysis/v7-calibration.md` found. Per Jack's instruction of 2026-10-18, PLAN.md's pre-registered rules no longer bind, and the two calibration groups are pilot data, not main-run data (decisions-log row of that date). Tests: 150 pass (140 before; `uv run --group dev pytest tests/ -q`). The final code passed four full runs with no flaky failure.

## What changed

| # | Change | Where | Verified |
|---|---|---|---|
| 1 | Every agent box is m7a (AMD EPYC 9R14, Zen 4, AVX-512, the scorer's CPU), the same in every arm. `make_v7_plan.py --boxes m7a` is the default: m7a.large per independent, m7a.2xlarge per team of 4, m7a.xlarge / m7a.4xlarge for big-memory tasks. An m row falls back only to r of the same generation (m7a -> r7a). `fleet.py` prices a VM only at its own region's rate (`HOURLY_USD_BY_REGION`, us-east-1 from the AWS Price List API, queried 2026-10-18) | 6cdb9d6 | Live: both smoke agent VMs reported AMD EPYC 9R14 (cloud-status `host` line and each agent's lscpu). The r7a fallback was not triggered |
| 2 | Every agent's prompt carries one timing-hardware paragraph, the same in every arm, built from the scorer's /health at run start. It says that dev_eval, the final selection and the official scoring run on a separate machine, gives the CPUs per timed run and threads per core, the CPU model, family and Zen generation, and the ISA extensions. /health gains `session`, `machine_type`, `slot_cpus`, `exclusive_slot`, `arch`, `cpu_vendor`, `cpu_family`, `cpu_model_number`, `microarchitecture`, `threads_per_core` and `cpu_flags`. Logged as run metadata `scorer_hardware` | d1153c9, 78919e1, 578db3c | Live: present in all 5 smoke prompts, identical in both arms. Text below |
| 3 | v7 selection starts only after every row of the plan has ended. Then one `select_portfolio.py` selects every group (`v6_drive.py run --select-after plan`, the default for plans with v7 rows) | ba81eab, 081430f | Live: the team row ended 16:32:58 and the independent 16:33:07; selection started 16:33:22, and its first job was queued 91 s after the last agent-run job. The single-process form (081430f) came after the smoke and has a unit test and the local mock run only |
| 4 | Hidden scoring runs as exclusive scorer jobs, one at a time with every other slot idle, on one fixed slot. Every hidden job must come from the scorer session the selection started on. Hidden jobs start once every pick is fixed, and a group's arms alternate repeat by repeat. The diagnosis is in HARNESS.md, "Scorer noise", and summarised below | d1153c9, 147cefd, 081430f | Live: 21 exclusive jobs, all on slot 0, none overlapping any other job, one session. Alternation is unit-tested only |
| 5 | v7 is_solution rejects non-concrete solutions with upstream AlgoTune's `find_nonconcrete_solution`, which `tasks.patch_task` appends to every v7 evaluator copy. That covers the toolkit, dev_eval, finalize, the validity pass and the verifier | 7680182 | Live: the ode team's pick failed a validity job 20 of 20 with "Non-concrete solution rejected: Solution contains non-list iterable at solution: array"; the mixed pick passed 20 of 20. All 30 v7 reference outputs pass the guard (task image, Mac Docker) |
| 6 | The operator's public-IP lookup retries with backoff (5 s doubling to 60 s) for up to 15 min, and rejects a non-IPv4 answer. allow-me, `operator_ip_refresh` and `aws_ensure_access` all use it, and select_portfolio waits up to 1,200 s for allow-me | 1bd4cf7 | Unit test with an injected lookup and sleep. Not tested on a real outage |
| 7 | `analysis/v7-calibration/analyse.py` (now tracked) counts only completed model calls | 90723bd | Rerun on the calibration logs gave 2,042 calls before and 2,039 after; the 3 removed calls are the report's 3 `ReadError` calls |
| - | `scripts/smoke_v7b.py` runs the runner with scripted mockllm agents. Each agent records its first prompt and a probe of its box (`report` prints them). `v6_drive.py run --script` passes a script through to fleet.py. v6_drive prices all-mockllm rows at $0 | d175cba, 8cf2061, 7019040 | Used for the local and AWS smoke runs |

The smoke scorer's paragraph, as every agent received it:

> Timing hardware: dev_eval, the final selection and the official scoring all run on a separate scoring machine, not in your own box. Each timed run there gets 8 dedicated CPUs (1 hardware thread per core). CPU: AMD EPYC 9R14 (x86_64, AMD family 25 model 17, Zen 4). Instruction-set extensions: sse4_2, avx, avx2, fma, bmi2, avx512f, avx512dq, avx512cd, avx512bw, avx512vl, avx512ifma, avx512vbmi, avx512_vbmi2, avx512_vnni, avx512_bitalg, avx512_vpopcntdq, avx512_bf16, gfni, vaes, vpclmulqdq.

The paragraph is on by default for `--experiment v7 --checker remote` (`--no-scorer-hardware` turns it off). A scorer whose /health lacks a field is refused. v6 rows are unchanged.

## Scorer noise

The calibration's 1.16x gap was in the solver's time, not the reference's. Across the 42 score jobs the reference timed alone varied by at most 1.3% between scorings of one source and 2.4% across a task. All jobs ran on one VM type with the same CPU, slots, timing environment and alone measurement. The noisy sources are 8-thread ode solvers at about 0.5 ms per solve. stable_matching's picks varied by 1.01-1.02x, and 57 of 60 Phase B sources by at most 1.07x. Over 18 repeated ode scorings, solver time relative to the source's fastest averaged 1.075 on slot 0, 1.006 on slot 1 and 1.059 on slot 2.

The smoke scorer (c7a.8xlarge, 3 slots; `analysis/v7b-smoke/measure_scorer.py`, dev_eval n=20 with 10 repeats; rows in `measure.jsonl`) split this into three effects on the mixed team's pick (`mix_pool`) and the independents' pick (`indep_pick`).

| Effect | Measurement | mix_pool | indep_pick |
|---|---|---|---|
| Slot | Run alone, one at a time; mean solver ms by slot (0, 1, 2) | 10.49, 11.22, 10.49 (max/min of 6 runs 1.12) | 17.99, 18.66, 18.12 (1.05) |
| Load | All 3 slots busy, against alone | +5% (11.31 against 10.73 ms) | +3% (18.80 against 18.25 ms) |
| Drift | Exclusive jobs on slot 0, round 2 against round 1 (5 sources alternating) | +6.5% | +5.5%; the other 3 sources +2.9 to +6.4% |
| Host pinning | dockerd, containerd, the latency holder and movable IRQs on the service CPUs, irqbalance stopped; alone again | 1.15x across slots, slot 1 still slowest | 1.06x |

The reference stayed within 1.8% in every phase. The guest's L3 domains coincide with the slots. On the smoke VM slot 1 was the slowest, and it held containerd and an NVMe interrupt the kernel would not move. On the calibration VM slot 1 was the fastest. The slot pattern therefore differs between VMs, and pinning host processes did not remove it, so I did not adopt pinning. Fixes 3 and 4 remove the slot, load and session effects from the scores a comparison uses, and alternating the arms makes drift fall on both. A residual per-job spread of 3-9% for these solvers remains within one slot (3 exclusive runs per source); the median of 3 repeats reduces it.

Exclusive scoring is serial and holds the whole scorer. A smoke unit_simplex_projection score job took 4.0 min, and calibration ode and stable_matching score jobs took 3.7 and 6.2 min. The calibration's 30 hidden jobs would take about 2.3 h serially, against about 45 min on 3 slots.

## Output-trick audit

| Trick | Evidence | Verdict |
|---|---|---|
| ode team pick returns `array.array('d')` | Harbor's is_solution calls `np.asarray(solution, dtype=float)`, which accepts any buffer. Upstream AlgoTune's validator rejects any solution that is not built from plain values before calling is_solution (`find_nonconcrete_solution`, commit 32f9e06, 2026-01-19, hardened in d1decc3). array.array falls under "non-list iterable". The trick skips building the list of Python floats that the reference returns | Exploit by upstream's rule. Closed in the v7 copies (fix 5) with a test and a live validity job |
| Mixed pick reuses its output lists and their float objects | It returns a plain list of Python floats, which passes upstream's guard. It keeps 4 lists and writes new values into a list or float only when CPython's reference count shows nothing else holds it. Every value is the computed final state: 1,000 of 1,000 calibration validity instances and 20 of 20 in the smoke passed. Measured gain: the same source with `USE_POOL = 0` took 13.40 against 11.72 ms (1.14x, 3 exclusive rounds); the team pick's own float-reusing list against `out.tolist()` took 12.84 against 14.39 ms (1.12x) | Legitimate optimisation (object pooling). It relies on CPython internals, and its gain needs callers to drop earlier outputs; the verifier's loop does so, and 4 pooled lists cover a caller that keeps the last 3. Left alone |
| Mixed pick approximates the step-size factor `0.9 * err^-0.2` with a cubic | The cubic applies for err between 0.568 and 0.612, where its largest relative error is 1.25e-7 (checked on 200,001 points), and `pow` is used elsewhere. It changes step sizes, not the method's error control (rtol = atol = 1e-8). is_solution compares against scipy's RK45 with rtol 1e-5, which is the task's stated contract, and 1,000 of 1,000 calibration validity instances passed | Legitimate numerical change within the stated tolerance. Left alone |

The report also listed the stable_matching team pick reading CPython 3.12 object memory directly. That affects only how the input is read, and the output is a plain list, so by the same criteria it is legitimate and was not changed.

## Live AWS smoke test (mockllm only)

Run on 2026-10-18 from 16:22 to 17:38 Sydney time (driver log `~/.cache/bgjob/jobs/20261018-162222-v7b-aws-98c4/log`; scorer logs `logs/_scorer/i-083d0b297a4e58bb7/`). It used one c7a.8xlarge scorer with 3 slots at ref v7b 7019040, and `plans/v7b-smoke.tsv`. That plan has a team of 4 on m7a.2xlarge and an independent on m7a.large, in two groups, with the v7 flags and a 60,000-token budget. `v6_drive.py run` drove it with `--group-start --validity-n 20 --script scripts/smoke_v7b.py`. The measurements followed, then `scorer.sh down`.

- All 5 agents (`analysis/v7b-smoke/agents-report.txt`) reported AMD EPYC 9R14, nproc 2, `os.cpu_count()` 2, an affinity of 2 CPUs (team 0-1, 2-3, 4-5, 6-7), OpenBLAS 2 threads in numpy's and scipy's libraries, numba 2 threads, memory.max 5,376 MiB and cpu.max 2 CPUs. All 5 prompts held the hardware paragraph.
- /health reported every new field (`analysis/v7b-smoke/scorer-health.json`), with machine c7a.8xlarge, 8 CPUs per slot, Zen 4 and exclusive slot 0.
- Both portfolios recorded the exclusive hidden jobs, the session `34aa61f8e4db4f4e` and the machine type (`portfolio-*.jsonl`). The picks were the reference wrapper, scoring 1.0.
- The scorer was terminated at 06:38:44 UTC and both agent VMs at 05:32:55 and 05:33:04. Afterwards `describe-instances` listed no pending, running, stopping, stopped or shutting-down instance in us-east-1, and no tagged volume remained.

AWS cost at list prices was about $2.17. That is the scorer (1.27 h at $1.64224/h, $2.08), the team VM (0.115 h at $0.46368, $0.05), the independent VM (0.114 h at $0.11592, $0.01), about $0.01 of gp3 and about $0.01 of public IPv4. The volume-initialization rate charge is not priced here, and none of this is checked against billing. There was no model spend.

The local mock run (Mac Docker, one container at a time, 2 CPUs each, about 90 s per run, four runs) used a team of 2 and an independent against an in-process scorer. dev_eval and final_eval ran for real; score, score_dev and validity returned canned results. It showed the hardware paragraph in every prompt, no selection before the last row ended, all hidden jobs exclusive, and in the last run (081430f) both groups in one selection.

## Left alone

- Queue waits (3 slots for 21 agents), the text-only tail before the budget ends, and the Sol agent reading the continue message as a user instruction. These apply to both arms alike.
- The mixed team's Sol-only hours and team run-to-run variance. Both are design questions (Phase D, number of groups), not harness bias.
- Stage-1 (finalize, during the run) and stage-2 (score_dev) timings are not exclusive, so their ranking carries the slot and load noise above. Making them exclusive would make the run's dev_eval queue serial or lengthen selection further.
- The runs' own hidden scores (finalize's pick, scored at the end of each run) stay non-exclusive. Making them exclusive would stall every other agent's dev_eval for 4-6 min at each run's end.

## Follow-ups

1. Serial hidden scoring is the main cost of fix 4. 30-40 groups at 10 hidden jobs of about 4 min come to about 20-27 h of a whole scorer. Options are a small scorer for the selection phase only (still one session and one machine type for all hidden scores) or fewer oracle-bound scorings.
2. 8-thread sub-millisecond solvers still vary by 3-9% between exclusive runs on one slot. Consider 5 repeats per pick on such tasks, or report the per-pick spread.
3. Any analysis that resamples the runs' own scores (best-of-k independents, per-run spread) should first rescore those solvers exclusively in one session. `rescore.py` has no exclusive mode yet.
4. `/proc/cpuinfo` and lscpu still show the VM's CPU count, 8 on a team VM and 2 on an independent's. Every box's nproc, `os.cpu_count()` and affinity show 2.
5. A c row still falls back to m6a, c6a, m7i or c7i, and an r row to r6a or r7i, which are other CPUs. `make_v7_plan.py --boxes t3a` still exists and puts the arms on different CPUs for big-memory tasks.
6. Upstream's other validator hardening is not ported. Upstream validates the timed output rather than an untimed first call, and checks validator dependencies for tampering. A solver could return a correct first output and different later ones, and the first-call ratio only catches a speed difference.
7. v6_drive's spend cap reads `--budget` as dollars even for token budgets, so the smoke needed a cap of 1e9.
8. The calibration report's AWS table priced the m7a.4xlarge at Sydney's rate. At the us-east-1 rate it is $1.98 (2.13 h at $0.92736), not $2.46, which makes the total about $13.37.
9. Main's working tree holds an untracked `analysis/v7-calibration/analyse.py`, which this branch now tracks (fixed), so git will refuse the merge until that untracked copy is moved. `analysis/portfolio/` is also untracked in both trees.
10. Carried over from `analysis/v6-followup/v7-build-notes.md`: `task_image_digest` has no retry in the runner, `_offsets_in_logs` uses the wrong glob, `cpu_report.py` reads only agent_0's box, and the task image has no `ps`.

## Readings taken

- I read "same CPU generation and memory per vCPU" as the same generation with at least as much memory per vCPU (the existing rule), so m7a may fall back to r7a, which has the same CPU and more memory.
- For "one scorer session", I took a session to be one service run on one VM. Each job's view carries the service's session id, and selection checks it.
- I relied on the brief's statement that Jack approved the AWS smoke with a $5 cap. The run cost about $2.17.
