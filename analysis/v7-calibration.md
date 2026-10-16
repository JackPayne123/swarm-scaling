# v7 calibration results (2026-10-16)

Read-only analysis of the 12 rows in `plans/v7-calibration.tsv` (commit 9f7ddf5, all exited 0) and their selection records, `analysis/portfolio/20261016T191712-10624.jsonl` (stable_matching) and `analysis/portfolio/20261016T210428-31499.jsonl` (ode_lorenz96_nonchaotic, "ode" below). Every agent had a $5 cost budget, ran to budget with no submit tool, worked in its own 2-CPU box (`--isolate-agents`) and got the diversified-v7 prompt; teams also got publish notices. Opus 5.5 ran at effort xhigh, GPT-6.1 Sol at high. The scorer was one c7a.8xlarge with 3 slots. Times are Sydney local. No scorer job, cloud call or model call was made for this report.

## Headline

- The diversified team of 4 beat the 4 diversified independents by 1.58x on ode (152.5 against 96.3) and 1.60x on stable_matching (184.0 against 114.9), on the primary alone score. Both team picks also beat every independent run's own score (best 107.0 and 117.8). In v6 (Haiku 5.5, $0.10 per agent, 14 groups) the same contrast was 0.97.
- On ode the mixed team (2 Opus, 2 Sol) scored 177.0, 1.84x the Opus independents and 1.16x the Opus team. One Sol agent alone scored 102.6. These are one run each.
- Both Opus-team picks combine work from all four teammates. Agents shared findings from the first minutes (first message at 0.00-0.09 of the sender's budget), claimed separate parts of the problem, and kept building on each other's code until their budgets ran out. Nobody stopped early to defer to a teammate.
- Every agent ended by spending its budget ($5.001-5.046). Opus agents spent a median 15% (independents) and 11% (team agents) of their budget after their last published candidate, against 37-39% for v6 Haiku. About half of that tail was text-only turns at the end, when too little budget was left to edit and test a change.
- Harness caveats that bear on the comparison. On stable_matching the team box was a Zen 4 m7a (the scorer's CPU family) while the independents had Zen 1 t3a boxes. Nine agents probed the scorer's hardware by encoding CPU counts into `solve()` run time through dev_eval; on stable_matching the only independent that found the scorer's 8 cores scored twice the other three. On ode, the same source on the same hidden instances scored up to 1.16x apart in different scorer sessions.
- The calibration cost $105.42 in model calls ($90.35 Anthropic, $15.07 OpenAI) and about $13.9 of AWS at list prices.
- At a per-group SD of log ratios of 0.34-0.40, a main Opus run needs about 30-40 groups for 80% power at +20% and 50-67 at +15%. Two groups say almost nothing about that SD.

## 1. Results per group and arm

Scores are speedups over the reference. "Alone" is the v7 primary score (reference timed alone); "interleaved" is AlgoTune's. Stage 2 is the pick's score on the 100 stage-2 dev instances. The oracle bound is the best hidden score in the shortlist, chosen with hidden scores (the pick's median, one scoring for each other source).

| Group, arm | Pick (author, stage-1 rank) | Hidden median, alone | Hidden median, interleaved | Pick's 3 hidden scorings, alone | Stage 2, alone (interleaved) | Oracle bound | Distinct valid sources (all candidates) | Best stage-1 dev |
|---|---|---|---|---|---|---|---|---|
| ode, 4 diversified independents | i0 agent_0-7 (2) | **96.30** | 96.71 | 96.04, 96.30, 104.62 | 102.28 (103.02) | 100.13 | 29 (33) | 113.58 |
| ode, diversified team of 4 | agent_3-4 (2) | **152.55** | 151.75 | 157.99, 152.55, 150.38 | 152.92 (154.62) | 164.53 | 28 (33) | 208.97 |
| ode, mixed team (agents 0, 2 Opus; 1, 3 Sol) | agent_1-17, Sol (2) | **176.95** | 178.67 | 163.12, 183.94, 176.95 | 170.21 (172.39) | 177.26 | 44 (50) | 203.85 |
| ode, 1 Sol independent | agent_0-16 (2) | **102.60** | 104.35 | 101.66, 102.60, 112.46 | 112.68 (112.62) | 109.74 | 21 (22) | 117.82 |
| stable_matching, 4 diversified independents | i0 agent_0-5 (1) | **114.89** | 115.50 | 117.32, 114.89, 113.89 | 119.04 (118.67) | 116.71 | 19 (23) | 116.55 |
| stable_matching, diversified team of 4 | agent_1-5 (2) | **184.01** | 178.10 | 183.52, 184.57, 184.01 | 185.51 (179.93) | 186.23 | 18 (23) | 166.36 |

The headline figures in the brief (96.3, 152.5, 177.0, 102.6, 114.9, 184.0) match the records. Interleaved ratios tell the same story (team over independents 1.57 on ode, 1.54 on stable_matching). All 18 validity checks (1,000 fresh instances each) passed with 0 invalid, so no source was dropped or backfilled. The stage-2 pick was stage-1 rank 2 in 5 of 6 portfolios. The oracle bound exceeded the pick by 0.2-7.9%; the largest gap was the ode team (agent_0-9, 164.53 on one scoring). The independents' portfolios each picked a source from the run that also had the best own score (i0 on both tasks).

### What the winning solvers did

**ode.** The reference integrates Lorenz-96 (N = 6,148) with scipy's RK45 at rtol = atol = 1e-8. Every ode agent decided within its first minute or two to reproduce scipy's step sequence exactly, and the team agents measured why. The reference is itself about 1e-4 from the true solution, so a more accurate integrator fails is_solution near zero-valued components. All four independents and all teams therefore built exact RK45 replicas and spread each step over 8 threads.

- Independents' pick (i0, agent_0-7). C exact replica compiled with gcc at `__init__`, cache tiles of 160, a persistent pthread pool with min(8, CPUs) threads, and Python C-API list input and output. The other three independents built numba variants of the same design and reached 83-108 on stage 1.
- Team pick (agent_3-4). The same exact replica with pieces from all four agents. agent_2 found that the scorer has 8 cores (probe at 417 s, f = 0.23) and introduced halo-only exchange with pointer swaps (950 s) and a fully parallel start-up (1,752 s). agent_0 introduced the pool with early wake and a per-thread sequence-flag barrier (919 s) and array.array output (919 s). agent_1's register-shuffle AVX kernel (911 s) was the team's best kernel until agent_3 replaced it, and agent_1 handed agent_3 the idea of writing the output into array.array directly when it ran out of budget (1,914 s). agent_3 wrote the lane-transposed kernel (1,761 s), in which each thread's segment is stored across the vector lanes so every stencil neighbour is the adjacent row and the inner loop needs no shuffles, and reimplemented the pool, barrier and halo exchange around it. That kernel took the team's best stage-1 score from 162 to 180, then 193. The pick is agent_3's kernel with workers writing the final state straight into an `array.array('d')` buffer, plus agent_2's start-up patch.
- Mixed team pick (Sol agent_1-17, published at 11,006 s). Its base came from the Opus agents, namely agent_0's single-region OpenMP replica (458 s) and agent_2's strict-list float-object pool, slot allreduce and L1 tiling (908-1,580 s). Sol agents added the SSE2 pipeline with hoisted coefficients and resident-interior halo exchange (agent_1, 677-786 s), the AVX2 sliding stencil (agent_3, 1,088 s), compact scratch rows and 64-byte row alignment (agent_3, 2,530-2,919 s). After both Opus agents had spent their budgets (2,474 s and 3,321 s), the Sol agents added aligned stage starts, AVX-512VL VALIGNQ shifts, a no-unroll build and a cubic approximation of the step-size factor `0.9 * err^-0.2` (error below 1.3e-7, with a fallback to `pow`).
- Sol independent pick (agent_0-16). C++ exact replica, 8-thread pool, specialised 192/256-element tiles, distributed step control, direct CPython list interface and cache-line-aligned state.

**stable_matching.** The reference is pure-Python Gale-Shapley on n = 1,151.

- Independents' pick (i0, agent_0-5). C extension that overlaps Gale-Shapley with a 6-thread on-demand conversion of the first 40% of each receiver's list into a rank table, deferring proposals to unconverted rows. i0 learned from a sleep probe at f = 0.28 that the scorer slot has 8 cores. i1, i2 and i3 each found receiver-list truncation (if Gale-Shapley on lists cut to each receiver's top K gives a perfect matching, it is stable in the full instance) and raw CPython memory reads. i1 and i3 stayed single-threaded and i2 parallelised only its data build; they reached 45-48 on stage 1 and 55-58 on their own hidden scorings.
- Team pick (agent_1-5). numba with llvmlite intrinsics that read CPython list, dict and int memory directly (agent_3, 217 s); receiver lists truncated to the top K (agent_1, 650 s, K later tuned to 160); an 8-thread prescan building uint8 rank tables and a bitset (agent_0, 825 s, after its own probe found 8 cores); a two-phase design in which each proposer collects its first 16 acceptable entries in parallel (agent_1, 1,224 s); software prefetch, packed rows and a single numba dispatch (agent_3, 1,303-1,830 s); phase B overlapped with the sequential matching in one prange region using atomics (agent_0, 2,137 s); and prefetch-distance tuning (agent_1, 4,304-4,592 s). agent_3's first combination of truncation with the parallel prescan (1,047 s, 97.7x on its own dev_eval) raised the team's best stage-1 score from 69 to 100. By 1,224 s (agent_1 at f = 0.52) the team had passed every independent's final stage-1 score.

## 2. Behaviour against v6 Haiku and pilot 5 Opus

v7 rows are measured from the logs (`analysis/v7-calibration/analyse.py` plus scratch scripts); pilot 5 and v6 rows are copied from `analysis/v6-followup/padding-check.md`. Pilot 5 and v6 had an 80% submit gate, so their after-last-publish shares are of a spend that stopped at about 81% of budget. v7 agents ran to 100%.

| Setting | Agents | Spend / budget | Model calls, median (range) | Distinct candidates per agent | Spend after last publish, median (range) | Calls after last publish, median share | How runs ended |
|---|---|---|---|---|---|---|---|
| v7 Opus diversified independents | 8 | 100-101% | 85.5 (57-95) | 6 (3-9) | 14.8% (4.4-18.2%) | 23% | budget, all 8 |
| v7 Opus team agents (ode and stable_matching teams) | 8 | 100-101% | 81.5 (67-96) | 5 (3-10) | 11.0% (3.7-23.1%) | 16% | budget, all 8 |
| v7 Opus agents in the mixed team (agent_2, agent_0) | 2 | 100-101% | 74, 86 | 8, 4 | 4.5%, 54.9% | 7%, 53% | budget |
| v7 Sol (independent; two in the mixed team) | 3 | 100-101% | 215; 188, 190 | 21; 16, 16 | 2.3%; 1.7%, 23.7% | 1%; 1%, 15% | budget |
| pilot 5 Opus single, dst, $5 | 9 | 81% | 53 (38-89) | 5.2 | 8% | 13% | accepted submit at 80-91% |
| pilot 5 Opus team of 4, dst | 4 | 81% | 65 (59-71) | 2.2 | 7% | not tabulated | accepted submit |
| v6 Haiku diversified independents, $0.10 | 56 | 81% | 54 (19-119) | 2.1 | 37% | 58-65% (three v6 arms) | accepted submit |
| v6 Haiku diversified team | 56 | 81% | 61 (34-129) | 2.0 | 39% | 58-65% | accepted submit |

- **Spend.** Measured per-call spend equals each agent's metered total to 4 decimals for all 21 agents. Overshoot past $5 was $0.001-0.046, at most one call.
- **The tail.** For Opus independents about half of the spend after the last publish (50% pooled, $0.11-0.42 per agent) was text-only turns. In them the agent declined to start a change it could not also test, and each one drew the harness's continue message. The rest went to check_budget, a few bash calls and 1-4 dev_evals. Trailing text-only turns per agent were 0-6. Mixed-team Opus agent_0 is the exception. It spent 55% after its last publish on real experiments (a butterfly allreduce, a split-phase barrier, robustness runs of teammates' candidates) that did not beat what teammates had published.
- **Re-timing.** dev_eval of a source already evaluated (same sha256) was 0-3 calls per Opus agent (at most 17% of its dev_evals), against 56% of v6 dev_eval calls that re-timed an unchanged file. The checker served 2 exact repeats from its record.
- **First publish.** Median 0.31 of budget for Opus independents and 0.21 for the 10 Opus team agents (0.08-0.56). Sol agents published at 0.01-0.02 and then published 16-21 times.
- **Wall time.** Opus agents ran 1,946-5,842 s (median 3,604 s); Sol agents 10,377-12,783 s. Per-call latency was similar (median 5-11 s Opus, 7 s Sol), but Sol made 2.2-2.5x as many calls at $0.023-0.027 each against Opus's $0.052-0.086, and 65-90 dev_evals against Opus's 8-30. Sol spent 7,500-9,070 s inside dev_eval, about 6,000 s of it queueing. For v6 Haiku the mean wall time per agent was 4,991 s (independents) and 6,295 s (teams) in the groups-A report.
- **Output tokens.** Opus 100,760-135,118 per agent, Sol 97,766-131,784, similar to pilot 5 Opus (102-116k median).

## 3. Team mechanics

| | ode team | stable_matching team | mixed team |
|---|---|---|---|
| Distinct messages (per-recipient deliveries; dropped because the recipient had finished) | 28 (63; 7) | 21 (53; 11) | about 150 (281; 14) |
| Messages per agent | 6-8 | 4-7 | Opus 9 and 10, Sol 61 and 70 |
| First message: time, sender's budget fraction | 92 s, 0.06 | 217 s, 0.09 | 10 s, 0.00 (Sol); first Opus message 82 s, 0.04 |
| Publish notices (deliveries) | 29 (75) | 19 (48) | 46 (100) |
| Notice delivery latency, median (p90, max) | 24 s (98, 280) | 38 s (115, 205) | 26 s (111, 235) |
| Deliveries followed by a recipient tool call naming the entry or its registry path | 22 of 75 | 9 of 48 | 13 of 100 |
| Tool calls touching a teammate's directory or registry entry | 57 | 53 | 93 (45 of them cross-model) |
| Published notes crediting a teammate | 20 | 5 | 13 (10 cross-model) |
| Spend on coordination tools (SendMessage, wait_for_message, list_candidates, TaskList, send_file) | 4.7% | 4.5% | 15.9% |
| SendMessage calls that failed | 0 | 0 | 0 |

- **Delivery.** Every notice reached every running teammate. The "used" count is a lower bound, because messages usually pointed teammates at working directories (`/app/agents/agent_k/...`), not registry paths. The candidate notes and messages show the adoption directly.
- **When sharing started.** In v6, the first message came at a team mean of 0.13-0.55 of budget, or never (groups-A report), and the first substantive share at 0.18-0.52 (groups-B report). In v7 the ode team's first messages (92-209 s) were findings and territory, for example "agent_3 IMPORTANT accuracy finding ... the tolerance is NOT loose in practice" and "To avoid overlap I'm taking the 'alternative implementation' route" (agent_0, f = 0.13). In the mixed team the Sol agents divided the methods between themselves before writing any code (10-22 s).
- **Adoption chains.** Both Opus-team picks are combinations that passed through every agent (section 1). Typical notes are "agent_0-3 = agent_3-3 (154x) + phase B and GS overlapped in ONE prange(T) region" and "agent_0: agent_3-2 (lanet2 lane-transposed kernel) + array.array('d') output". Twice two agents published byte-identical files (stable_matching agent_1-3 and agent_0-4, both agent_0-3 with K = 160; ode agent_2-9 and agent_0-8, both agent_0-7 with agent_2's start-up patch). Two agents handed off an untested idea when out of budget (ode agent_1 to agent_3 at f = 0.95, the array.array output; stable_matching agent_2 to everyone at f = 1.00, K = 192), and in both cases a teammate ran it.
- **Deferral.** Every "I'm stopping" statement came at 0.87-1.00 of the agent's budget and cited the money left. In v6, 13 of 24 team agents (groups-B report) announced stopping or deferring to a teammate at 0.21-0.65. wait_for_message was called once per team, against 5.8% of v6 team spend.
- **Mixed team.** No agent mentioned a model name; the prompt did not reveal teammates' models (`reveal_model_family` is off). Opus and Sol agents read each other's files about as often as same-model teammates' (45 against 48 tool calls) and credited each other more often (10 against 3 notes). Sol sent 109 deliveries to Opus agents and Opus 35 to Sol. Opus messages were longer (median 866 characters against 272). The Opus agents spent their $5 in 41 and 55 minutes; the two Sol agents kept working for another 2 h, so 54 of the messages and 15 of the 34 Sol publishes came after no Opus agent was running. The pick was published by a Sol agent and built on an Opus framework (section 1). The shortlist holds only Sol-published sources, and no candidate from the Opus period was scored on hidden instances, so I could not measure what the Sol-only period added on hidden. On stage 1 it raised the team's best from 196.2 (agent_2-6) to 203.9.

## 4. Harness checks

- **Box configuration.** Every agent box had 2 CPUs on its own cpuset (team agents 0-1, 2-3, 4-5, 6-7) and 5,376 MiB (ode) or 13,056 MiB (stable_matching), per `agent_boxes` and the compose file. CPU telemetry reports a 2-core limit on every box. No box was flagged (saturated intervals at most 10%, throttled periods at most 9.4%, ode team agent_3). Mean box use was 1-15% of 2 cores.
- **Live readings by the agents.** `nproc` printed 2 for all 18 Opus agents that ran it. Where agents printed them, `os.cpu_count()`, affinity and numba's thread count were 2 (stable_matching i2 "2 2 2", stable_matching team agent_1 "omp 2 2 2", ode i2 "nthreads=2 aff=[0, 1]", mixed agent_3 affinity {6, 7}). No agent printed BLAS thread counts or its memory limit, so those are verified only as configuration. `lscpu` reports the VM's CPUs (2 on t3a.large, 8 on t3a.2xlarge, 16 on m7a.4xlarge), so the VM size was visible; I saw no agent size threads from it.
- **CPU model differs between arms on stable_matching.** The stable_matching team ran on m7a.4xlarge (AMD EPYC 9R14, Zen 4, AVX-512, the scorer's family), its independents on t3a.xlarge (EPYC 7571, Zen 1). Every ode row ran on EPYC 7571. The team's local timings were therefore on the scorer's microarchitecture and the independents' were not. ode team agent_0 named the problem for the Zen 1 boxes ("our containers are Zen1, so local timings mislead on SIMD"). I could not determine how much of the stable_matching gain this explains.
- **Agents probed the scorer.** Nine of 21 agents wrote a solver that slept for a time encoding `sched_getaffinity`, `os.cpu_count()` or CPU flags and read the value back from dev_eval's timing (ode i1, i3, team agents 0 and 2, mixed agents 0 and 1, stable_matching i0, i3, team agent_0). They learned that the dev_eval slot has 8 usable cores on a 32-CPU host with AVX-512, while their own box has 2. Every pick runs 6-8 threads on the scorer. Teams shared the fact by message (stable_matching agent_0 at f = 0.37, ode agent_2 at f = 0.23). On stable_matching, i0 probed at f = 0.28 and went multi-threaded; i3 probed only at f = 0.76, none of i1-i3 sized its solver to the 8 cores, and they scored about half of i0.
- **Scorer timing noise on multi-threaded ode solvers.** The picks' 3 hidden scorings spread 1.05-1.13x on ode and 1.006-1.03x on stable_matching; Phase B's ode fast solver (12.2x) spread 1.003. The same source on the same hidden instances scored differently in different sessions. ode team agent_0-9 scored 190.85 in its run-level scoring (19:08, other slots running dev_evals) against 164.53 in selection (21:29, all three slots running 8-thread ode scorings), a ratio of 1.16; ode i0 agent_0-8 1.10; the mixed and Sol runs' own picks 1.04; stable_matching 0.96-1.03. The ode team pick fell from 205.8 at stage 1 to 152.5 on hidden (1.35x, beyond the 1.28 level Phase B's A6iv would flag); the other ode portfolios fell 1.14-1.16x and stable_matching 0.90-1.01x. Selection scored all four ode portfolios in one session under the same load, so the within-group comparison is like for like, but I could not determine the cause or whether it treats solver designs differently.
- **Queue waits.** 3 slots for 21 agents. The slots were 95.5% busy from 17:09 to 19:17. dev_eval waits had a median of 69 s and a mean of 82 s over 527 jobs (p90 168 s, max 835 s), against 161-186 s mean in v6. The stable_matching selection (19:17-20:13) ran while the mixed team and the Sol independent were still working, and their 31 dev_evals in that hour waited a mean of 221 s (max 835 s), against 76 s before 19:17.
- **Scorer errors.** None. All 764 jobs (527 dev_eval, 159 final_eval, 42 score, 18 score_dev, 18 validity) finished with status done and no job error. 15 dev_eval results reported invalid solver outputs, all agent bugs. Selection stage-2 and hidden jobs had reference inflation 0.959-1.024 and first-call ratios at most 2.03.
- **Validity pass.** 18 sources checked, 0 invalid, no drops. Runs took 95-98 s per ode source, 369-372 s per stable_matching independent source and 845-862 s per stable_matching team source (Phase B's reference took 582 s). All are inside the 3,600 s limit.
- **Selection pause.** The driver log prints one "scorer unreachable; allow-me: could not read this machine's public IP" between the 19:49 and 19:55 log pulls, then "scorer port and SSH open to 103.134.170.9" between 19:55 and 20:00. These lines carry no timestamps, so I could not date the block more precisely. The scorer's job log shows no submission gap other than the selection's limit of 6 jobs in flight. Every job was queued within 4 s of a predecessor ending or a slot freeing. The independents' portfolio was recorded at 19:55:06, 39 s after its last hidden job ended. Request ids prevented duplicates, and the counts are exactly six portfolios' worth (18 validity, 18 stage-2, 30 hidden). Earlier, from about 17:57 to 18:28, the scorer-log pulls failed ("Network is unreachable", "Operation timed out"). Agents run on AWS and kept working (113 scorer jobs finished in that window).
- **Other.** 3 bash timeouts in 21 agents (v6 teams 56). 3 model calls failed with `ReadError('')` and were retried (no usage recorded; I excluded them from call counts). Background-process cleanup found nothing to kill in any box. One Sol agent read the harness's continue message as a user instruction ("User asks continue until budget spent").
- **For the exploit audit (not judged here).** The ode team pick returns `array.array('d')` instead of a list. The mixed pick returns a list whose float objects are reused across calls when CPython reference counts show nothing else holds them, and approximates the step-size factor. The stable_matching team pick reads CPython 3.12 object memory directly. All passed 1,000-instance validity.

## 5. Spend

Model spend from `scripts/run_cost.py` (per call from logged usage, 1-hour cache writes pinned). It matches the driver's "$105.42".

| Row | Model | Spend |
|---|---|---|
| ode div-i0, i1, i2, i3 | Opus 5.5 | $5.002, $5.020, $5.031, $5.012 |
| ode divteam | Opus 5.5 | $20.087 |
| ode mixteam | Opus 5.5 / Sol | $10.048 / $10.039 |
| ode soldiv-i0 | Sol | $5.030 |
| stable_matching div-i0, i1, i2, i3 | Opus 5.5 | $5.025, $5.022, $5.017, $5.021 |
| stable_matching divteam | Opus 5.5 | $20.068 |
| Total | Anthropic $90.35, OpenAI $15.07 | $105.42 |

AWS, estimated from instance hours (creation to deletion in each row's `cloud-status`) and `scripts/cloud/fleet.py` HOURLY_USD:

| Item | Hours | Rate | Estimate |
|---|---|---|---|
| 5 x t3a.large (ode independents, Sol independent) | 1.63-3.92 each, 10.83 total | $0.0752/h | $0.81 |
| 2 x t3a.2xlarge (ode team, mixed team) | 2.11, 3.73 | $0.3008/h | $1.76 |
| 4 x t3a.xlarge (stable_matching independents) | 1.32-1.96 each, 6.23 total | $0.1504/h | $0.94 |
| 1 x m7a.4xlarge (stable_matching team) | 2.13 | $1.1592/h | $2.46 |
| Scorer c7a.8xlarge, 17:03:45 to 21:51:46 | 4.80 | $1.64224/h | $7.88 |
| Total | | | about $13.85 |

fleet.py's m7a rate is the ap-southeast-2 price, but this VM ran in us-east-1c; I did not look up the us-east-1 rate, so that row may be overstated. The estimate leaves out EBS volumes, data transfer, the GCS log bucket and t3a unlimited-mode CPU charges (box CPU averaged 1-15% of 2 cores; host-level averages were not measured). Model plus AWS is about $119. Per group, an Opus group cost $40.15 in model calls plus $1.15 (ode) or $3.40 (stable_matching) of agent VMs, with the scorer shared.

## 6. What it implies

**Spread of the independents.** Each independent run's own hidden score (its finalize pick, one scoring during the run) was:

| Task | i0 | i1 | i2 | i3 | SD of log | max / min |
|---|---|---|---|---|---|---|
| ode | 106.99 | 75.91 | 94.79 | 79.23 | 0.160 | 1.41 |
| stable_matching | 117.83 | 58.47 | 55.16 | 57.69 | 0.363 | 2.14 |

Pooled over the two tasks the SD of a single run's log score is 0.28 (6 degrees of freedom, 95% interval 0.18-0.62). The stable_matching spread comes almost entirely from i0, the one independent that used the scorer's 8 cores.

**Team over independents, as log ratios.** ode 0.460 (1.58x), stable_matching 0.471 (1.60x); mean 0.466. The mixed team over the Opus independents was 0.608 (1.84x) and over the Opus team 0.148 (1.16x). The Sol independent over the 4-agent Opus portfolio was 0.063 (1.07x). That is one $5 Sol agent against a $20 Opus portfolio, so it is not a matched comparison.

**Power for a main Opus run.** I assumed a one-sample t-test on the per-group log ratio (team over diversified independents), two-sided alpha 0.05, 80% power, independent groups, and that the pre-registered task-level bootstrap behaves like this test at these sizes. The answer depends on the SD of per-group log ratios, which two groups cannot estimate.

| Source of the SD | SD | Groups for +20% | Groups for +15% |
|---|---|---|---|
| The two calibration ratios (1 df; they differ by 0.011) | 0.008 (95% interval 0.0035-0.25) | 2 (17 at the interval's top) | 3 (28 at the top) |
| Components from the calibration's run spread: team portfolio noise = one run's (0.28), independents' portfolio = 0.7 of that (best of 4), no task-to-task variation in the effect | 0.34 (0.22-0.75 from the run-spread interval) | 30 (14-137) | 50 (22-231) |
| v6 Haiku team over diversified independents, 14 groups (13 df) | 0.40 (0.29-0.65) | 40 (22-101) | 67 (36-170) |

- The two calibration ratios agree closely by chance as far as I can tell; with 1 degree of freedom the SD's 95% interval spans a factor of 70. Planning on an SD of 0.34-0.40 gives 30-40 groups for +20% and 50-67 for +15%.
- With an SD of 0.34-0.40, 2 groups put a 95% interval of roughly 0.9-2.8x around the observed 1.59x (known-SD normal approximation). If the true effect were as large as observed, 7-9 groups would give 80% power. At an SD of 0.34-0.40, 4 groups have 10-12% power at +20%.
- Hidden-score noise is part of the SD. On ode, same-source scorings in different sessions differed by up to 0.15 in log (section 4), a sizeable share of a +20% effect (0.18).
- Cost at the measured $40.15 of model calls per Opus group is about $1,200-1,600 for 30-40 groups. PLAN.md's 2026-10-16 row describes a $200 Anthropic credit; if that is the only source, about $110 remains after this run's measured $90.35, enough for 2 more groups. I did not check the balance against billing.

**PLAN.md's rule.** The amended calibration rule (2026-10-16) makes these 2 groups (ode and stable_matching, Opus team against diversified independents) the first groups of the main Opus run if no harness, prompt, selection or budget change follows the calibration. If any change is made, both tasks become pilot tasks and leave the held-out set. Section 4 records several findings a change might target, namely the CPU-model mismatch on stable_matching rows, agents probing the scorer for hardware they cannot see in their own box, selection overlapping running agents, and session-dependent scoring of multi-threaded solvers. Changing any of them would trigger the rule. The mixed team and the Sol independent are Phase D rows outside the main design either way.

## What I could not determine

- Why the same ode source scored up to 1.16x apart in different scorer sessions, and whether that noise favours some solver designs. Concurrent multi-threaded jobs in the other slots are a candidate, not tested.
- How much of the stable_matching team's gain comes from its Zen 4 box against the independents' Zen 1 boxes.
- What the mixed team's two Sol-only hours added on hidden instances (no Opus-period candidate was hidden-scored).
- BLAS thread counts and memory limits as seen inside the boxes (configuration only; no agent printed them).
- The exact start and end of the operator-IP block during selection (the driver's lines carry no timestamps).
- Run-to-run variance of a team, since each task has one team run. The 0.34 component SD assumes it equals an independent run's.
- Opus reasoning is encrypted; I read reasoning summaries, visible text, tool calls, messages and candidate notes.
- AWS figures are list-price estimates, not billing.

## Method and files

Per-agent metrics (spend, calls, last publish, distinct sources, end reason, queue waits, notice use) come from `analysis/v7-calibration/analyse.py`, which was already in the working tree and untracked. Its call counts include the 3 failed calls, so I subtracted them. Transcript timelines, candidate notes, messages, CPU telemetry, scorer jobs and the power calculations come from scratch scripts that read `logs/v7c-*/*.eval` with the Inspect API, `logs/_scorer/i-0b09e81e911fb9ccd/jobs.jsonl` and `queue.jsonl`, the driver log `~/.cache/bgjob/jobs/20261016-170342-v7cal-4a95/log`, each row's `cloud-status`, and `analysis/v7-prep/scorer-checks/verdicts.json` for the Phase B comparison. Budget fractions (f) are an agent's own spend at that moment over $5; times (t) are seconds since the run started.
