# Selection scorer and agent-scorer size (2026-10-18)

Branch `selscorer` (worktree `~/projects/swarm-scaling-sel`), on main `82ba244`, pushed, not merged. It makes two changes to the v7 run driver for the next batch, both approved by Jack on 2026-10-18. The agent scorer gets one slot per 2 running agents, and selection runs on a second scorer that comes up when the first group ends. Tests: 154 pass (150 before; `uv run --group dev pytest tests/ -q`).

A live AWS smoke with scripted mockllm agents in two groups that ended at different times confirmed the sequence. The selection scorer came up only after the first group's last row ended, both groups were selected on it in one session, the agent VMs used only the agent scorer, and each scorer came down when it was no longer needed. Nothing was left in us-east-1. AWS cost at list prices was about $1.89, with no model spend.

## What changed

| # | Change | Where | Verified |
|---|---|---|---|
| 1 | v7 agent scorer: one slot per 2 agents running at once (`V7_AGENTS_PER_SLOT`), at most 11 slots; v6 plans keep 8. 16 agents get a c7a.24xlarge (11 slots), 8 a c7a.12xlarge (5). `SCORER_MACHINE=<type>` (v6.sh; `v6_drive.py size --machine`) sets the type instead, and the slots and the agent cap (slots x 2) follow from it. The quota check is kept and now also counts the selection scorer's vCPUs | `scripts/v6_drive.py` size, `scripts/v6.sh` | Unit test (`test_v7_agent_scorer_gets_a_slot_per_two_agents_...`). `size` against the live quota (read-only) printed `11 c7a.24xlarge 22` for `plans/v7-batch1.tsv`. In the smoke, `size` with `SCORER_MACHINE=c7a.8xlarge` printed 3 slots and a cap of 6, and fleet reported 448 free vCPUs with 32 kept for the selection scorer. No c7a.24xlarge agent scorer was launched |
| 2 | Selection scorer: for plans with a v7 row, `v6_drive.py run` brings up a second scorer (`--selection-scorer`, default c7a.8xlarge with 3 slots; `none` keeps the old `--select-after` modes) when the first group's rows have all ended. It gets the agent scorer's experiment, so its secret offset, and the same ref. Each group is selected on it as soon as its own rows have ended. When no selection is running, one `select_portfolio.py` takes every group waiting, so all arms of a group run validity, stage 2 and exclusive hidden scoring in one session. The agent scorer goes down when the last row has ended and the selection scorer after the last selection; v6.sh takes both down again on exit. `fleet.py plan --reserve-vcpus` keeps the selection scorer's vCPUs free until it starts. After 3 failed `scorer.sh up` attempts the remaining groups are left unselected and the driver exits 1 | `scripts/v6_drive.py` run, `scripts/v6.sh`, `scripts/cloud/fleet.py` | Unit test with faked scorer, fleet and select calls (`test_v7_groups_select_as_they_end_...`), mutation-checked: each of five deliberate breaks failed it. Live in the smoke (below). The give-up path was not exercised |
| 3 | Per-scorer state: `SCORER_NAME` (`common.sh scorer_use`) is agent (default; `~/.cache/swarm-scaling/scorer.json`, VM `ss-scorer`, security group `swarm-scorer`, all as before) or select (`scorer-select.json`, `ss-scorer-select`, `swarm-scorer-select`). `scorer.sh up/down/pull/allow-me/status` act on that scorer. `down` terminates only instances with its own VM name and revokes only its own group's rules. Before, it terminated every `role=scorer` instance and revoked every rule of the one shared group, which would have cut off the other scorer. run_sample.sh sets `SCORER_NAME=agent` itself, so agent VMs only ever get the agent scorer's URL, token and port. `operator_ip_refresh` re-opens the operator's IP on every scorer that has a state file. `fleet.py cleanup` and `status` cover both scorers, v6_drive pulls logs from every scorer that is up, and `select_portfolio.py`'s allow-me acts on the scorer it was started for | `scripts/cloud/common.sh`, `scorer.sh`, `run_sample.sh`, `fleet.py`, `scripts/select_portfolio.py` | Unit tests with stubbed network calls and a fake `aws` on PATH (two tests in `tests/test_fleet.py`), mutation-checked. Live in the smoke: agent VM IPs were only ever in `swarm-scorer`, and the agent scorer's teardown left the selection scorer's rule in place. `operator_ip_refresh` after a real IP change and `fleet.py cleanup` were not exercised |
| - | Docs and records: HARNESS.md bullet "Selection scorer and agent-scorer size", a PLAN.md decisions-log row, v6.sh and v6_drive usage text. `analysis/selection-scorer/queue_model.py` computes the load figures and the queue model below. Smoke plan `plans/v7-selscorer-smoke.tsv`, wrapper `analysis/selection-scorer/aws_smoke.sh`, records in `analysis/selection-scorer/smoke/` | | |

## The sizing rule

Batch 1's agent scorer (`logs/_scorer/i-0b9a6d925d49b1c1d/jobs.jsonl` in main, agents' jobs only, `queue_model.py`):

- 16 agents, 213 dev_eval jobs over 2.45 h on 3 slots, which were 89% busy.
- A dev_eval job ran 67 s on average and waited 128 s in the queue.
- An agent queued its next dev_eval a mean 136 s (median 78 s) after its previous one ended.
- So one agent keeps about 67 / (67 + 136) = 0.33 of a slot busy when nothing queues. The v7 calibration's 21 agents (30 s jobs, 75 s gaps) give 0.28.

16 agents therefore offer about 5.3 slots of work. Three slots were saturated, and agents spent 43% of their wall time in dev_eval.

I chose one slot per 2 running agents. The c7a types come in 1, 3, 5, 7 and 11 slots, so 16 agents (8 slots wanted) get the c7a.24xlarge with 11 and 8 agents a c7a.12xlarge with 5. The two tasks differ: pagerank's dev_eval jobs ran 81 s with 118 s gaps (0.41 of a slot per agent), quantile_regression's 47 s with 163 s gaps (0.22). An exact mean-value analysis of the closed queue (agents alternate think time and one job at the scorer, exponential service; `queue_model.py`) gives these mean waits for 16 agents:

| Slots | 3 | 5 | 7 | 11 |
|---|---|---|---|---|
| Batch 1 average (67 s jobs, 136 s gaps) | 153 s (128 s measured), 100% busy | 28 s, 92% | 4 s, 74% | under 1 s, 48% |
| Pagerank (81 s jobs, 118 s gaps) | 232 s, 100% | 64 s, 98% | 13 s, 87% | under 1 s, 59% |

That is a model, not a measurement. It assumes an agent's think time does not depend on how long it waited, and it overestimated batch 1's mean wait by about 20%. One slot per 3 agents would give 16 agents 7 slots, which the model puts at 87% busy on a pagerank-like task, and 8 agents 3 slots (19 s, 80% on the batch average). One per 2 keeps the slots below about 60% busy at 16 agents on either task. At 22 agents, the most an 11-slot scorer serves at that ratio, the model gives 1 s on the batch average and 3 s (80% busy) on pagerank.

Cost: the c7a.24xlarge is $4.93/h against $1.64/h for batch 1's c7a.8xlarge. Over batch 1's 2.5 h agent phase that is about $8.20 more per batch, against agents that spent 43% of their wall time in dev_eval. Selection moves to a c7a.8xlarge, the type that held it in batch 1, so its cost changes only by the selection scorer's idle time between groups.

## Live AWS smoke test (mockllm only)

Run on 2026-10-19 from 00:49 to 01:50 Sydney time, after batch 1's driver had exited (00:49:20, exit 0). Driver log `analysis/selection-scorer/smoke/driver.log`. It used `analysis/selection-scorer/aws_smoke.sh`, which runs the real `scripts/v6.sh` on `plans/v7-selscorer-smoke.tsv` at ref `selscorer` fbaf245, with `SCORER_MACHINE=c7a.8xlarge`, the default c7a.8xlarge selection scorer, `--group-start --validity-n 20 --script scripts/smoke_v7b.py --max-agents 5`. Group r1 was a team of 4 (m7a.2xlarge) and an independent (m7a.large). Group r2 was one independent. With the cap of 5, r2 could start only once an r1 row had ended, so the groups ended 7 minutes apart. Every 60 s the wrapper sampled both scorers' security-group rules and the instances (`smoke/aws-samples.txt`, operator IP masked).

| Time (Sydney) | Event |
|---|---|
| 00:49:48 | Agent scorer `i-076fb5ecb0eaa8668` (ss-scorer) launched |
| 00:52:27, 00:52:42 | r1 team and r1 independent started |
| 00:59:48 | r1 team ended; r2 independent started |
| 01:00:06 | r1 independent ended, so group r1 was complete |
| 01:00:21 | Driver: "r1 is ready: selection scorer c7a.8xlarge up"; instance `i-0359debc675a57ccc` (ss-scorer-select) launched 01:00:34 |
| 01:03:30 | Selection scorer up; selection of r1 started on it |
| 01:07:10 | r2 independent ended (the last row) |
| 01:07:30 | Agent scorer down (instance terminated 01:07:42) |
| 01:27:35, 01:31:35 | r1's two portfolios done |
| 01:31:37 | Selection of r2 started on the same selection scorer |
| 01:47:44 | r2's portfolio done; selection scorer down (terminated 01:47:57) |
| 01:48:42 | v6.sh's second pair of downs found nothing; exit 0 |

Checks:

- The selection scorer came up only after the first group ended. Its instance launched 28 s after r1's last row ended, while r2's agent was working. Before 01:00:21 no `ss-scorer-select` instance and no `swarm-scorer-select` rule appear in any sample.
- Both groups selected on it, in one session. All 15 selection jobs (`smoke/selection-scorer-jobs.jsonl`) are `select_portfolio` jobs from session `2fb4369b2f2c4fe7`: 3 validity, 3 score_dev and 9 exclusive hidden score jobs, the hidden ones all on slot 0 and none overlapping another job. r1's two arms alternated repeat by repeat. All three portfolio rows (`smoke/portfolios.jsonl`) record that session, machine type c7a.8xlarge and exclusive slot 0. The picks were the reference wrapper, scoring 1.0.
- Agent VMs used the agent scorer only. Its 12 jobs (`smoke/agent-scorer-jobs.jsonl`: 6 dev_eval, 3 final_eval, 3 score) all come from the three runs, in session `b8e819afb5e740cb`. Each run's eval log names that session in its /health, its dev_eval calls and its final score. The two agent VMs of r1 and the one of r2 only ever appear in `swarm-scorer`'s rules; `swarm-scorer-select` only ever held the operator's rule. r2's own score job ran on the agent scorer from 01:01:58 to 01:05:57, while r1's validity and stage 2 ran on the selection scorer from 01:03:32.
- Both scorers came down. The agent scorer's teardown removed its own rules and left the selection scorer's operator rule in place (samples 01:09-01:47). The final check (`smoke/final-aws.txt`, 01:49:43) listed no pending, running, stopping, stopped or shutting-down instance in us-east-1, and no volume tagged tool=swarm-runner.

AWS cost at list prices was about $1.89. That is the agent scorer (0.298 h at $1.64224/h, $0.49), the selection scorer (0.790 h, $1.30), the team VM (0.117 h at $0.46368, $0.05), the two independent VMs (0.117 h each at $0.11592, $0.03), about $0.01 of gp3 and about $0.01 of public IPv4. The volume-initialization rate charge is not priced, and none of this is checked against billing. There was no model spend.

## Batch 1's own selection, for comparison

Batch 1 selected after its last row ended, on its agent scorer (c7a.8xlarge, 3 slots), from 21:35:54 to 00:48: 3.2 h after its agents had finished. Its 12 validity jobs at n=1000, 6 per task, ran 3 at a time and took about 1,600 s each on pagerank and 108 s on quantile_regression. A pagerank hidden score job took 446 s.

## Readings taken

- "Slots per running agent": the rule counts the agents that can run at once (the plan's agents, or as many as fit beside both scorers in the quota), as the old rule did, and the cap on agents at once stays slots x ratio. With an explicit `SCORER_MACHINE` the cap also stays slots x 2, so a type too small for a group holds that group back rather than lifting the cap.
- "Select groups as they become ready": a group is ready when every one of its rows has ended. When the selection scorer is free, one `select_portfolio.py` takes every group waiting; while one runs, newly ready groups wait for it. I did not run selections of different groups at once. Their exclusive hidden jobs would then be spread out between other groups' long validity jobs, and the scorer would finish at about the same time either way.
- The selection scorer stays up between groups until the last selection, as the brief says, so it idles while it waits for later groups.
- The selection scorer gets the agent scorer's experiment (read from its state file when the driver starts), the run's ref, and v6.sh's HOURS and SCORER_MEMORY.
- Agent VMs are kept off the selection scorer by separate security groups and by run_sample.sh pinning `SCORER_NAME=agent`, not only by the URL they are given.
- The smoke used `--max-agents 5` so that the second group started after the first. Batch 1 is the evidence that real groups end at different times (21:07 and 21:35).

## Follow-ups

1. Selection scorer size. On 3 slots a group's 6 validity jobs (more with backfills) take two rounds, and so does stage 2. In batch 1 that was about 54 min of validity per pagerank group. `SELECTION_SCORER=c7a.16xlarge` (7 slots, $3.28/h; `SELECT_CONCURRENCY` allows 6 jobs in flight) would do each in one round. Exclusive hidden scoring stays serial either way. I kept the brief's c7a.8xlarge default. This is a cost-against-time call for Jack.
2. One selection at a time. A group that ends while another is being selected waits for that selection to finish, including its serial hidden scoring. In the smoke, r2 waited 24 min (01:07:10 to 01:31:37). The scorer is busy throughout, so starting the second selection earlier would not finish sooner unless the selection scorer has spare slots for the non-exclusive parts (follow-up 1).
3. The selection scorer idles between groups, at $1.64/h, when groups end far apart. Taking it down between groups would put each group in its own session, which the comparison allows, at about 3 min of start-up per group.
4. v6.sh has no signal trap. `bgjob stop` on a v6.sh job kills the shell before its two `scorer.sh down` lines, so the scorers then end only at the HOURS backstop. This was already true for the agent scorer and now applies to the selection scorer too. `fleet.py cleanup --yes` takes both down.
5. With an explicit `SCORER_MACHINE`, a cap of slots x 2 below a group's agent count keeps that group from ever starting under `--group-start`. For example, a c7a.8xlarge has a cap of 6, and a batch-1 group has 8 agents. fleet.py then reports the group as "does not fit in the free vCPUs", which names the wrong cause. Nothing guards against it.
6. `SCORER_NAME=select scorer.sh down` creates the empty `swarm-scorer-select` group when it does not exist yet, because `aws_scorer_sg` creates on lookup. v6.sh runs that down for v6 plans too. The group costs nothing, but a lookup-only path would avoid creating it.
7. `fleet.py scorer up --plan` keeps its own sizing rule (one slot per 4 remote agents), separate from v6_drive's and unchanged. v6.sh does not use it.
8. The queue model's assumption that think time does not depend on waits is unmeasured. Batch 2's agent-scorer job log will show the real waits on 11 slots, and `queue_model.py` reruns on any job log.
9. Not exercised live: an explicit type different from the rule's pick (the smoke's c7a.8xlarge is what the rule picks for 6 agents), the 3-attempt give-up, `operator_ip_refresh` after an IP change, and `fleet.py cleanup` over both scorers.
