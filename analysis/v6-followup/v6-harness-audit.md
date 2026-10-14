# v6 main: harness audit across 126 rows

Scope: every row of `plans/v6-main.tsv` (`logs/v6-*`, 126 rows, 168 agents), the selection records in `analysis/portfolio/`, the re-selection scorer log `logs/_scorer/i-0d30a2ddff1fb1a5d/`, and the fleet log. Read-only; no scorer or model calls. The least_squares r1 loose invalid pick and the pca r2 team stage-2/hidden drop are covered by another agent and appear here only where a harness fact bears on them. Scratch extraction scripts and outputs: `scratchpad/audit/` (`rows.jsonl` holds one record per row).

Headline check: all 42 cells of the results table in `analysis/v6-main-results.md` reproduce from the selection records (36 portfolios) and the re-selection job log (6 portfolios whose records were overwritten): the pick is the fastest valid stage-2 entry and its hidden median matches to 0.006 in every cell.

## Findings ranked by how much they could bias team vs independents

### 1. generalized_eigenvalues_complex is decided by one precision switch (float32 eigvals)

Every geneigc pick computes `X = B^-1 A` and then either `sgeev` in float32 or `dgeev`/QZ in float64. The score follows that choice, not the arm:

| Portfolio | Pick's eigenvalue precision | Hidden median |
|---|---|---|
| r1 loose | float64 (getrf/getrs/dgeev) | 3.57 |
| r1 diversified | float32 | 6.44 |
| r1 team | float32 (sgesv + sgeev) | 6.35 |
| r2 loose | float32, plus `threadpool_limits(1)` held at module level | 4.77 |
| r2 diversified | float64 (lu_solve + eigvals) | 3.95 |
| r2 team | float32 | 6.43 |

Every float32 pick without the thread limit lands at 6.35-6.44; both float64 picks land at 3.57-3.95. The team chose float32 in both repeats, each independent arm in one. The team's r2 win over diversified independents (+0.49 log) is this single switch. This is the dst/float32 pattern the v6 pre-registration tried to keep out of the comparison: the result on this task measures whether a portfolio adopted one precision trick, which sharing can propagate but which says little about collaboration in general. Recommend reporting the paired ratios with and without geneigc.

The r2 loose pick also shows a harness side effect worth knowing: its module-level `threadpool_limits(1)` persists in the verifier process, so the interleaved reference also runs single-threaded. Its reference_inflation was 0.63-0.65 (reference faster interleaved than alone), which lowers its speedup. The thread guard only gates inflation above 1.15, so process-global side effects can move the denominator down freely and up by up to 15%.

### 2. Team rows ran 1.4-2.8 hours later than their group's independents in 10 of 14 groups

The fleet backfilled 4-vCPU independent VMs as quota freed, so a 16-vCPU team VM waited until 16 vCPUs were free at once. Five of the ten late team starts fall on the same minute as another team row's exit in the fleet log; the other five began once the r2 independents had drained. Because they started after 01:28 UTC, all seven r2 team rows checked out commit 6472d4c (finding 8).

| Group | Team start (UTC) | Independents' median start | Team delay (min) |
|---|---|---|---|
| clust r1, geneigc r1, sparseig r1, spd r1 | 21:30-21:37 | 21:31-21:38 | -1 |
| lsq r1 / pca r1 / sinkhorn r1 | 23:35 / 23:32 / 01:06 | 22:12 / 21:53 / 22:20 | 83 / 99 / 166 |
| r2 groups (7) | 01:37-02:31 | 22:53-01:05 | 86-164 |

What this changed, measured from the client-side job records:

| | Loose | Diversified | Team |
|---|---|---|---|
| Scorer jobs queued or running when a dev_eval was requested (mean) | 31.4 | 31.2 | 25.2 |
| Mean queue wait per dev_eval (s) | 187 | 178 | 172 |
| Agent's total queue wait as share of its wall time (mean) | 47% | 53% | 38% |
| dev_eval jobs that overlapped a selection job | 24% | 22% | 52% |

Hour by hour, the three arms saw nearly the same wait (for example 01:00 UTC: 299 / 299 / 316 s; 02:00: 369 / 363 / 390 s; 03:00: 85 / 74 / 87 s). The team simply ran more of its calls after 02:00 UTC, when independents had finished and the queue drained. The budget is metered in dollars and agents ran with a 1-hour prompt cache, so queue waits cost no tokens; the effect is on wall time and on when feedback arrived, not on spend. Per-row mean queue wait does not predict the row's best finalize speedup (Spearman -0.02 over 112 independent rows, task-centred). Selection is time-matched: all three arms of a group were selected in one process, with their stage-2 and hidden jobs starting within 0-13 minutes of each other. I rate the bias as small and of unknown sign.

### 3. least_squares: validity on 20 dev instances is unreliable, and the Haiku long-prompt price tier ended three independents early

Six of the 18 shortlisted lsq sources were fast on finalize's 20 instances and invalid on 100 (stage 2) or on hidden: r1 loose 60.2x stage 1 (invalid on hidden, the other agent's anomaly), r2 diversified 28.4x (invalid at stage 2), and one or two each in r1 team, r2 team and r2 loose (stage 2 caught these). No other task had an invalid shortlisted source. lsq scores therefore depend on whether a portfolio's best-looking candidate happens to be invalid, and on whether stage 2 catches it.

All three agents that ended on budget instead of an accepted submit were lsq independents, and each crossed the 100,000-token prompt threshold where Haiku 5.5's price rises 5x (`prices.LONG_PROMPT`): lsq r2 loose i0 ($0.1139, ended after 12 min, 10 model calls, one $0.0525 call at a 105,680-token prompt; it published nothing and left only a 0.59x workspace candidate), lsq r1 loose i3 ($0.1010) and lsq r1 diversified i3 ($0.1047). No team agent crossed the threshold (team max prompt 95,143). Long-tier calls were 2.1% of loose spend, 1.4% of diversified spend and 0% of team spend. The lsq r2 loose portfolio effectively had three working agents. This mostly bears on diversified vs loose and team vs loose, where lsq supplies the two lowest loose cells (1.00 and 1.65).

### 4. Team agents share one 8-CPU box; independents get 2 CPUs each

Team containers had 8 CPUs for 4 agents with no per-agent pinning; independents had 2 each. Nothing on the agent box is scored, but local experiments differ: a team agent can burst to 8 cores when teammates idle and is slowed when they run benchmarks or background jobs. Bash/python tool timeouts (300 s): team 61 of 1,713 calls (3.6%), loose 8 of 1,482 (0.5%), diversified 2 of 1,448 (0.1%). Most team timeouts are self-inflicted: 27 come from one geneigc r1 team agent running `nohup ... ; sleep 580` to wait for its own background seed sweeps. CPU flags (container saturated or throttled): team 7 of 14 rows, loose 16 of 56, diversified 10 of 56. Small bias, direction unknown.

### 5. Three groups were selected on a different scorer

geneigc r1, geneigc r2 and pca r2 were selected on a fresh c7a.24xlarge (`54.173.73.233`, scorer code 6472d4c, identical scorer-side code and the same task image digest) after the Mac lost network. All arms of each of these groups ran there, so within-group comparisons stay matched. reference_inflation on geneigc was about 0.8 on both scorers (task-intrinsic: dev_eval median 0.795 on the main scorer), but pca moved: main scorer stage 2 / hidden medians 1.02 / 1.04, fresh scorer 0.98 / 0.98 with a range of 0.74-1.11. The pca r2 team pick's bimodal hidden scores (7.25, 12.72, 7.06, with reference_inflation 0.76 / 1.03 / 0.79 and first_call_ratio 8.1 / 1.3 / 7.4) were measured there. That pick also embeds a compiled shared object as a base64 string (`_SO_B64`). Passed to the agent reading that anomaly.

### 6. The channel's cost is charged to the team's budget, as designed

Spend per agent was matched: loose $0.0823, diversified $0.0820, team $0.0813 (totals $4.610 / $4.591 / $4.554, $13.755 in all, matching the $13.76 the driver reported). Per portfolio: $0.329 / $0.328 / $0.325. The live meter matches per-call event pricing to within $0.0000005 for every agent, every model call sits inside an agent span, and no call escaped metering. Team agents pay for the channel. Their first prompt is about 1,790 tokens longer (team tool definitions and protocol; mean 8,194 vs 6,407 for diversified), which at cache-read prices over about 66 calls I estimate at roughly $0.0015 per agent, about 2% of spend (computed from the first-prompt difference, not measured per call). Responses that called a messaging or task-list tool accounted for 4.8% of team output tokens (an upper bound on the messaging output, since those responses also carry reasoning). Received messages are charged as input to the recipient, as the team prompt states.

### 7. Five dev_eval calls touched the stage-1 (finalize) instances

dev_eval seed s times instances 10,000 + s, and finalize times 60,000-60,019, so dev_eval seed 50,000 reaches the stage-1 instances. The seed cap (400,000) protects only stage 2 and hidden. Five calls did this: geneigc r1 div i0, geneigc r2 div i3, geneigc r2 loose i1 and i2, sparseig r1 div i3 (3 diversified, 2 loose, 0 team). Negligible bias, but it lets independents see the instances that rank their candidates.

### 8. Code version: agent-side code identical

| Commit | Rows |
|---|---|
| 9708fb2 | all 112 independents; team r1 rows of all 7 tasks |
| 6472d4c | team r2 rows of all 7 tasks (spd, clust, sparseig, geneigc, pca, lsq, sinkhorn) |

`git diff 9708fb2 6472d4c` touches only `scripts/select_portfolio.py` (the claim lock), `scripts/select_ready.py` (new) and a test. 482cb6d adds only the pid to the selector's output filename. Selection logic (stage1, pick, select) is unchanged across all three. The main scorer reported version 9708fb2 to every row; seed offset id and task image digest are identical in all 126 rows and both scorers.

## Checks that came back clean

- End reasons: 165 of 168 agents submitted after the 80% gate; 3 ended on budget (finding 3). No agent errored, no sample errored, no time limit was hit, every eval status is success, every VM exited 0. Overshoot past $0.10: only the 3 budget-ended agents, by $0.0010-0.0139 (the limit is checked after each call).
- Submit refusals per agent: loose 2.7, diversified 3.2, team 3.4; budget notices at 50% and 75% fired for every agent that reached them.
- Model API: 10,606 model calls, 0 recorded retries, 0 model errors, 0 `max_tokens` or length stops, 0 compaction events. 17 turns ended without a tool call (13 in geneigc r1 loose i0, 4 in diversified rows); react's continue prompt handled them. Max prompt 106,808 tokens.
- Prompts: all 168 agents' first-turn prompts checked against `swarm.py` constants. Loose agents have no exploration block, no priority and no sharing line; diversified independent `i<k>` has priority k; team `agent_i` has priority i; the sharing line appears only in team prompts; "up to you" appears nowhere. Exactly one prompt text per (task, arm) after normalising agent ids and priority names. The loose protocol is byte-identical to the v6 pilot's and differs from haiku2's only in the budget amounts. Tool lists: independents 8 tools; team adds SendMessage, send_file, read_message, wait_for_message, TaskCreate/Get/List/Update.
- Team tools: 203 peer messages delivered across the 14 teams (0 to 37 per team; lsq r2 team sent none), 0 dropped, 6 finish notices per team as expected, delivery lag median 1-165 s by team (max 564 s; lag is time to the recipient's next tool result). 27 SendMessage calls failed schema validation for a missing `summary`; I did not check whether each was re-sent. send_file used twice (pca r1 team, 4 deliveries). The task list was used twice (TaskList), TaskCreate never. Registry reads of teammates' candidates: 0 to 26 per team.
- dev_eval dedupe: 21 cached repeats of 2,103 calls (loose 0.8%, diversified 1.3%, team 0.8%), versus 56% re-timing in the pilot. Agents now vary the seed instead (seed > 0 on 82% / 78% / 69% of calls, max seed 395,000). Seed-cap refusals: loose 2, diversified 1, team 2.
- Candidate pools: mean distinct valid candidates per portfolio: loose 6.9, diversified 8.6, team 8.6.
- Stage 1 vs stage 2: finalize's 20-instance scores were all taken while 5-30 other jobs overlapped on the scorer. log(stage 2 / stage 1) for shortlisted valid sources: team mean -0.059 (sd 0.17), diversified -0.053 (0.18), loose -0.102 (0.31). No arm difference. Load at finalize time vs this ratio: per-task correlations from -0.43 to 0.23 on 9-18 points; not separable from task differences.
- Hidden timing noise: the pick's 3 hidden scorings agree within 3% in 40 of 42 portfolios; the exceptions are pca r2 team (7.06-12.72) and pca r2 loose (22.6-25.3).
- reference_inflation above 1.15 or scoring timeouts: none in any stage-2 or hidden job. In agents' dev_eval: inflation > 1.15 in 10 / 13 / 8 calls (loose / diversified / team). first_call_ratio > 5 in dev_eval: 88 / 64 / 20 calls; among picks it reached 5-8 only for sinkhorn r1 diversified (5.9-6.6, a solver that compiles C at first use) and pca r2 diversified and team (4.2-5.6 and 7.4-8.1). first_call_ratio is reported, never scored.

## Scorer job errors (5)

All five come from agent solvers, and none touched a pick or a shortlisted source.

| Row | Arm | Job | Error |
|---|---|---|---|
| lsq r1 div i3 | diversified | dev_eval, n 20 | exit 137 after 7.2 s |
| lsq r2 loose i1 | loose | dev_eval, n 30 | exit 137 after 7.3 s |
| lsq r2 divteam agent_3 | team | dev_eval, n 20 | exit 137 after 7.8 s |
| lsq r2 divteam agent_3 | team | finalize (final workspace) | exit 137 ("Killed") after 28.0 s |
| sparseig r1 loose i3 | loose | dev_eval, n 20, seed 61,000 | timed out after 900 s |

Exit 137 after a few seconds looks like the container's memory limit killing the solver. I could not confirm this because the main scorer's own logs were not copied (below).

## Task-level noise and headroom

| Task | Mean repeat-to-repeat abs(log r1/r2) per arm | Team vs diversified (log, r1, r2) | Diversified vs loose (log, r1, r2) | Max/min of the 6 cells |
|---|---|---|---|---|
| clust | 0.41 | +0.17, +0.37 | -0.33, +0.52 | 2.80 |
| geneigc | 0.26 | -0.01, +0.49 | +0.59, -0.19 | 1.80 |
| lsq | 0.49 | -0.44, -0.18 | +1.68, +1.54 | 7.70 |
| pca | 0.42 (1.04 in team) | -0.16, -1.06 | +0.13, -0.10 | 3.30 |
| sinkhorn | 0.14 | -0.04, +0.17 | +0.11, +0.09 | 1.41 |
| sparseig | 0.43 | +0.51, -0.07 | -0.20, +0.28 | 2.20 |
| spd | 0.10 | -0.12, +0.01 | +0.24, -0.06 | 1.27 |

sinkhorn and spd sit near a shared ceiling (all six cells within 1.27-1.41x), so they add little discrimination. On clust, lsq and sparseig, repeat-to-repeat variation within an arm is as large as the between-arm differences. Hidden-timing noise is small (above), so this variation comes from the agents and the selection, not from the scorer.

Per task and arm (means per agent):

| Task | Arm | Spend $ | dev_eval calls | Instances timed | Queue wait min | Wait share | Wall min (median) | Ended on budget |
|---|---|---|---|---|---|---|---|---|
| clust | loose / div / team | 0.0811 / 0.0813 / 0.0810 | 7.4 / 8.4 / 8.1 | 162 / 152 / 201 | 21.7 / 20.4 / 23.1 | 42% / 50% / 39% | 27 / 23 / 45 | 0 / 0 / 0 |
| geneigc | loose / div / team | 0.0808 / 0.0817 / 0.0820 | 29.8 / 31.6 / 8.5 | 964 / 1232 / 169 | 91.0 / 101.5 / 16.9 | 41% / 48% / 12% | 202 / 193 / 187 | 0 / 0 / 0 |
| lsq | loose / div / team | 0.0882 / 0.0846 / 0.0810 | 5.0 / 7.2 / 3.5 | 158 / 155 / 79 | 14.3 / 20.6 / 19.2 | 45% / 47% / 58% | 22 / 31 / 33 | 2 / 1 / 0 |
| pca | loose / div / team | 0.0813 / 0.0819 / 0.0817 | 11.2 / 12.9 / 10.0 | 276 / 242 / 184 | 38.2 / 37.4 / 34.0 | 40% / 50% / 34% | 90 / 55 / 85 | 0 / 0 / 0 |
| sinkhorn | loose / div / team | 0.0813 / 0.0816 / 0.0814 | 8.4 / 9.9 / 10.6 | 166 / 218 / 208 | 31.3 / 36.9 / 40.3 | 63% / 66% / 45% | 34 / 57 / 89 | 0 / 0 / 0 |
| sparseig | loose / div / team | 0.0818 / 0.0813 / 0.0813 | 17.9 / 9.0 / 10.2 | 501 / 178 / 197 | 54.5 / 25.8 / 30.3 | 56% / 55% / 38% | 77 / 35 / 68 | 0 / 0 / 0 |
| spd | loose / div / team | 0.0818 / 0.0815 / 0.0809 | 9.9 / 17.0 / 26.4 | 212 / 413 / 758 | 26.3 / 38.7 / 56.3 | 38% / 51% / 44% | 22 / 58 / 116 | 0 / 0 / 0 |

Across all tasks, team agents timed about 30% fewer dev instances per agent (257 vs 348 loose and 370 diversified) and used 690 scorer-seconds per agent vs 1,139 and 1,040. On geneigc, independents ran about 30 dev_eval calls each (seed sweeps checking float32 validity), while the team ran its sweeps locally in background processes. These are behaviour differences, not harness asymmetries, but they mean the team got less scorer feedback per agent.

## Could not determine

- The main scorer's own logs (`logs/_scorer/i-09115118c00b5f745/` is empty; the fleet log shows "ssh: Network is unreachable ... could not copy the scorer logs"). Queue lengths, slot busy fractions and the cause of the four exit-137 kills come only from client-side records. Server-side errors the client never saw cannot be counted.
- Stage-2 jobs that the geneigc r1/r2 and pca r2 selections may have run on the main scorer before the 17:20 network loss are not in any record.
- Whether concurrent jobs in other scorer slots shifted speedups during the run. `scripts/slot_interference.py` has not been run on a multi-slot scorer (HARNESS.md), and the earlier scoring investigation measured a 7-10% slowdown of reference and solver alike on generalized_eigenvalues_real with four busy slots. Selection is time-matched within each group, which limits any effect on arm comparisons.
- Whether each of the 27 SendMessage calls that failed validation was re-sent.
- Exact token cost of received messages per team agent (not separated in the logs).

## Follow-ups (not fixed)

- The dev_eval seed cap does not keep agents off the finalize (stage-1) instances; a range check on [50,000, 50,020) would close it.
- `fleet.py` backfills small VMs ahead of a waiting team VM, which delays team rows by hours; a future run that wants time-matched arms needs to reserve quota for the team row or start each group's rows together.
- The thread guard is one-sided: a solver that makes the interleaved reference faster (process-global thread limits) loses speedup and is never flagged.
