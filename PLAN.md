# Swarm Scaling at Matched Compute - Plan

Status: pre-pilot. Created 2026-10-05 from a scoping session (PA repo). Owner: Jack (independent side project, not Lyptus).

## Goal

A one-week independent study, written up as a LessWrong post, that follows up Toby Ord's curated post "Swarm Scaling" (22 Sep 2026). It can stand on its own or be phase 1 of a larger project.

## Research question

At a matched compute budget, when does a team of communicating agents beat one agent given the same total budget, and beat the same number of agents working independently? Does mixing model families (e.g. an OpenAI model with a GLM or Claude model) change the answer?

Three sub-questions, each answered by a planned contrast (see Analysis):

1. Swarm scaling: how score scales with team size N at fixed per-agent budget, versus one agent with N times the budget. Ord's λ: N agents achieve what one agent achieves with N^λ times the budget.
2. Communication: how much of a team's gain comes from communication rather than parallel sampling (communicating team minus independent team of the same composition).
3. Heterogeneity: whether mixed-family teams gain more than same-family teams, beyond what a mixed independent portfolio already gives.

A fourth axis is built in through task choice. On AlgoTune, agents can verify progress mid-run; on Harvey LAB data rooms they cannot. Park et al. predict communication pays mainly when progress is verifiable. Jack's view is that sharing ideas and dividing work has value without verification. Running both families tests this directly instead of assuming it.

## Why this is worth doing (evidence summary, detail in research/EVIDENCE.md)

- Every published multi-agent result we found comes from a lab (OpenAI GPT-5.6 Sol launch charts, Claude Opus 5.5 / Sonnet 5.5 system cards) and covers same-model teams. Where compute is matched, the single agent usually wins on tokens and the team wins on wall-clock. Exceptions exist (Anthropic's async subagents lead at low token budgets; Park et al. report favourable token curves), so the post must not claim "single agents always win".
- Ord's λ values (0.68 BrowseComp, 0.57 SEC-Bench Pro, 0.48 Terminal-Bench) come from launch-chart data. Our replication (research/REPLICATION.md) reproduces his point estimates (0.650 / 0.565 / 0.481) but not his BrowseComp interval: ours is 0.45-0.85, and BrowseComp violates the equal-slope assumption (step λ 0.82 for 1→4 agents, 0.47 for 4→16). Terminal-Bench is a 1→4 measure only. Budgeting in API dollars instead of output tokens raises λ by 0.06-0.08.
- Noam Brown (Dwarkesh, 17 Sep 2026): "we don't have very good science on multi-agent scaling"; OpenAI measured up to 16 agents; swarms tend to collapse into each agent solving the problem independently.
- No public compute-matched swarm study on open-ended optimisation or large-document legal work exists that we found. Mixed-model teams have been studied on debate/QA tasks (Beyond Symmetric Agents, Sept 2026, negative result vs sampling controls; Self-MoA 2025) but not on iterative agentic work at matched budget.

## Design

### Arms (per task, per repeat)

| Arm | What runs | Budget units (b = one base per-agent budget) |
|---|---|---|
| Solo duration, model A | 1 agent at b, 2b, 4b, 8b | 15 |
| Solo duration, model B | same | 15 |
| Independent pools | 8 independent runs at b per model, resampled into teams of N (no trajectory reused within a team) | ~14 extra |
| Communicating, same-family A | N = 2, 4, 8 at b each | 14 |
| Communicating, same-family B | same | 14 |
| Communicating, mixed A+B | N = 2, 4, 8, half each family | 14 |
| Total | | ~86 b per task per repeat |

N = 8 is the minimum Jack considers publishable (decided 2026-10-05). N = 1, 2, 4, 8 restores a λ curve comparable to OpenAI's 1/4/16 and Anthropic's 1/10/30/100.

### Models

- Main grid, cheap pair (revised 2026-10-06, choice not final): Gemini 3.8 Flash ($0.75 / $3.75 per 1M in/out, AA intelligence 40.9, agentic 40.2, ~243 tok/s) + DeepSeek V4.1 Flash via OpenRouter ($0.30 / $1.20, intelligence 39.5, ~222 tok/s). Both pass the tool-calling smoke test. Alternative second model: GPT-6 Luna via OpenRouter ($0.10 / $0.50, 38.1; ~96 s to first token). Not price-matched (about 3x apart); budgets are in output tokens, so matching holds, and dollars are reported per family.
- Dropped: GLM-5.3-Flash. Measured ~20-38 output tok/s with ~92-95% of output spent on reasoning; in two pilot runs it never wrote a solver.py (Jack: the Z.ai API is also unreliable). GPT-6 Luna direct: OpenAI account has no credit (2026-10-05); use OpenRouter if Luna is wanted.
- Frontier grid (2026-10-06, Jack: the study should run state-of-the-art Anthropic and OpenAI models): Claude Sonnet 5.5 ($2 / $10, intelligence 56) + GPT-6.1 Sol ($2 / $10, 51.8), price-matched, full N = 1, 2, 4, 8 design on ~15 AlgoTune tasks, plus an Opus 5.5 ($4 / $20, 57.6) single-family check at N = 1 and 8. Est. ~$1.9-2.5 per base run for Sonnet/Sol, scaled from the Gemini pilot's token profile (unmeasured for these models). Both have long time-to-first-token at default effort (AA: Sonnet 452 s, Sol 274 s), so set reasoning effort explicitly and measure per-call latency in a pilot. Originally planned as a small check:
- Frontier check, price-matched pair: Claude Sonnet 5.5 ($2 / $10, 56) + GPT-6.1 Sol ($2 / $10, 51.8), at N = 1 and 8 only, ~6 tasks. This tests whether the cheap-tier result holds for the Claude + OpenAI pairing Jack's hypothesis is about. Sonnet 5.5 is the cheapest permitted Claude tier (no Haiku, per Jack's global rules).
- Prices from `aai`, fetched 2026-10-05. Re-check before spending.

### Team protocol

- Fixed N, all agents start together on the full task. No spawning, so N is fixed and compute matching holds.
- No designated lead. A lead's model identity would confound the mixed-model result (GPT-6.1 Sol review). Agents may self-organise through messages.
- Real-time messaging: `send_message(to, text)` with `to` = one agent or `all`, delivered into the recipient's context before its next model call, plus `wait_for_message(timeout)`. This matches OpenAI's and Anthropic's lab designs (push messages), not Park's shared-directory-only design.
- Private writable workspace per agent, plus a shared read-only candidate registry where agents publish immutable candidates with their dev scores. Prevents agents overwriting each other's work.
- Incoming messages count against the recipient's token budget.
- Prompt is loose: facts only (own id, teammate ids, the tools, how the final answer is chosen, the budget, and "how you work together, if at all, is up to you"). No rules on message content, adoption, roles or approach assignment. The independent arm gets the same text minus teammates and messaging. Exact text in research/PROTOCOL.md. Rationale: the labs' designs bake in as little structure as possible, and whether agents form a conductor, split the work or work alone is a result to measure, not something to dictate.
- Side check: the structured protocol in research/PROTOCOL.md (Park-style anti-herding, approach declaration, adoption bar) runs at N = 4 on pilot tasks with the cheap model only. If it beats the loose prompt by a wide margin, the post reports that these models under-coordinate by default; the loose prompt stays the main design either way.
- Full detail: HARNESS.md.

### Final selection (applies to every arm identically)

- AlgoTune: the fastest correct candidate on development inputs, chosen deterministically. Legitimate per the Sol review; using held-out scores to choose would be oracle selection. No voting (it would handicap the independent arm).
- Harvey LAB: a merge step writes the task's named `.docx` deliverables in every arm, including solo, and its cost counts against the budget. Harbor's format forces a single deliverable, so this is no longer optional. Scoring the union of independent findings remains possible as a secondary analysis, labelled as an upper bound.

### Compute accounting

- Allocate and enforce budget per agent in OUTPUT tokens (incl. reasoning), the unit of Ord's λ (decision 2026-10-06). The price-matched pairs keep output-token and dollar matching close. Log input, output, cached and reasoning tokens, dollars, verifier calls, CPU time and wall-clock for every agent, and report λ under output tokens (primary), dollars with caching, and total tokens.
- Report realised spend and plot outcomes against it, but do not condition on realised spend (an agent stopping early is part of the treatment).

## Tasks (detail and rejected options in research/TASKS.md)

1. AlgoTune (primary). 154 CPU numerical-optimisation tasks, continuous speedup score. Use the ~32 tasks with real headroom (best model 1.5x-10x), split into pilot and held-out sets before looking at results. Per-task metric: log speedup; also report the official harmonic mean for comparability. Available as `inspect_harbor/algotune`. As shipped on Harbor, agents get no generator or eval command, so we install a dev toolkit (generator, validator, timing script) in every arm to make it the "verifiable mid-run" family, and turn agent network off. Details in HARNESS.md.
2. Harvey LAB data rooms (second family, if the pilot passes). The 11 `diligence/*` tasks: 2,600-4,060 files each, 438-1,114 criteria, recall of red flags located in specific documents. No mid-run verification (rubric never mounted). Standard Harvey tasks are unusable (all-pass at the floor, criterion pass at 88-95%). The Harbor dataset is the v1.0 launch snapshot and does NOT contain the data rooms, so we convert the 11 tasks into local Harbor tasks (template in `research/harbor-examples/harvey_task`). Its scorer keeps only all-pass; we need a custom scorer to keep the per-criterion fraction.
3. ProgramBench: phase 2. Good fit and a published Anthropic baseline, but registry data shows $2-25+ per task under mini-swe-agent and it needs Linux x86 Docker images (~1 GB each). A 20-task subset is ready in `research/task-data/core_subset.json`.

## Analysis (freeze before the main run)

- Unit of resampling: task. Repeats nested within task. Never treat resampled independent teams or correlated criteria as independent observations.
- λ: fit score against log total budget for the solo curve and the team curve within overlapping budget ranges; λ = slope ratio. This equals the replication's λ = 1 + c/s (score = a + s ln T + s(λ-1) ln N) for this design, because team runs hold per-agent budget b fixed, so ln T = ln N + ln b and the team curve's slope in ln T is sλ. State the fitted model. Bootstrap over tasks for CIs. Report per family. Do not assume λ is constant across N: also report step λ (1→2, 2→4, 4→8) and test equal slopes, since OpenAI's BrowseComp data violates it (REPLICATION.md). Report λ under both token and dollar budgets.
- Planned contrasts, at each N:
  1. Communication benefit: C_mixed - I_mixed, and C_A - I_A, C_B - I_B.
  2. Portfolio benefit: I_mixed versus I_A and versus I_B separately (never a per-task "best homogeneous" chosen with held-out results, unless labelled as an oracle bound).
  3. Heterogeneity-communication interaction: (C_mixed - I_mixed) - 0.5 * [(C_A - I_A) + (C_B - I_B)].
- Adoption: record artifact lineage (which agent's candidate or finding another agent built on). Report "no observed cross-agent adoption" counts, not a "collapse rate". Inspect a few adoption chains by hand.
- Emergent structure (from message and registry logs, no extra cost), per team, then compared across N, task family and same-family vs mixed teams:
  - message graph: who messages whom, in-degree and out-degree; an agent receiving or sending most traffic, or assigning work, is an emergent lead; count teams with one;
  - division of labour: how many distinct approaches the team's candidates represent, and whether agents claimed different parts of the task;
  - teams with no coordination at all (no messages read and acted on, no adoption);
  - a short hand-coded label per team (conductor-led, peer split, independent, other) on a sample, with the coding rule written before reading the logs.
- Wall-clock speedup per arm.
- CPU contention (teams of N share the one 8-CPU AlgoTune container a solo agent gets): every Docker run records cgroup CPU use and throttling (`state.metadata["swarm"]["cpu"]`, sampled every 15 s) and flags runs where the agents saturated the container (90%+ of the CPU limit in 25%+ of intervals, or 10%+ of periods throttled). `scripts/cpu_report.py` tabulates flagged share, mean utilisation and throttling per N and arm. Decision rule for the pilot, fixed now: if team runs are flagged materially more often than solo runs, raise `override_cpus` for team arms (up to Docker's 16) or move to the AWS box, and report CPU per agent alongside tokens. Verified in a real container with `scripts/check_cpu_telemetry.py` (8 busy loops: mean util 0.996, flagged).
- Candidate pool size per agent and per arm (published + final). Pool parity is by mechanism, not by count; if teams publish far more per agent, report it and check the communication benefit holds when selection is restricted to each agent's final directory.
- Exploit audit: check winning AlgoTune solutions for validator shortcuts (AlgoTune has patched several, e.g. PCA rotated-subspace). Track whether an exploit spreads through the team's messages.

## Week plan

| Day | Work | Stop condition |
|---|---|---|
| 1 | λ replication from `research/sources/openai/multiagent_charts.json` (check latency units, they look like minutes). Local Docker (VM only if the timing gate fails). Install inspect-ai + inspect-harbor. Run the AlgoTune oracle and default agent on 2 tasks with GLM-5.3-Flash. Build the AlgoTune dev toolkit. Start the multi-agent solver (HARNESS.md). | Scoring works end to end |
| 2 | Finish the solver with mockllm tests (termination, budgets, delivery). Pilot N = 1, 2, 4, 8 on 3 AlgoTune pilot tasks with the loose prompt; measure cost per b, message delivery, adoption, emergent structure. Side check: structured vs loose prompt at N = 4 on the pilot tasks. Harvey: convert one data room to a local Harbor task with a criterion-fraction scorer; one GLM-5.3-Flash solo run; check recall is well below ceiling and measure agent + judge cost. | If no credible progress or timing is untrustworthy, fall back (below) |
| 3 | Freeze prompts, budgets, task list, selector and analysis. Start AlgoTune main grid. | |
| 4 | Finish AlgoTune grid. Harvey grid if the day-2 pilot passed. Rerun infrastructure failures under a rule fixed in advance. | |
| 5 | Analysis. Manual checks of the largest gains and adoption chains. | |
| 6-7 | Write the post. Package prompts, logs and exclusions. | |

Fallback (from the Sol review): if the pilot shows no progress or untrustworthy timing by end of day 2, publish the λ replication plus pilot findings and defer the heterogeneity claim.

## Budget

- Measured pilot runs (AlgoTune cvar_projection, one agent, b = 60k output tokens, 2026-10-06; logs/pilot-*):

| Model | Output tok | Input tok (uncached) | Cache-read tok | Wall | Result | Est. cost |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 60,167 (55,367 reasoning) | 128,464 | 88,576 | 31.6 min | no solver.py, 0 | ~$0.05 |
| Gemini 3.8 Flash | 60,879 (23,000 reasoning) | 417,176 | 2,139,449 | 12.5 min | 22.9x speedup | ~$0.70-0.94 |

  Cost estimate = list prices x logged tokens, with cache reads priced at 10-25% of input (the Gemini cache rate was not checked; confirm against the provider's billing before relying on it). Output is ~2% of all tokens: the agent re-sends its growing context every call, so input and cache reads dominate cost even though the budget meters output.
- Grid cost at b = 60k (~86 base-run units per task per repeat, assuming cost scales with output tokens; the 8b solo run will cost more per unit because its context grows longer):
  - Gemini 3.8 Flash: ~$60-81 per task-repeat; 20 tasks x 2 repeats ~ $2,400-3,200. Over the ~$2k ceiling.
  - DeepSeek V4.1 Flash (same token profile assumed, unmeasured): ~$0.26 per base run, ~$22 per task-repeat, ~$900 for 20 x 2.
  - Mixed Gemini + DeepSeek: between the two.
  - Ways to fit the budget: DeepSeek-heavy main grid; fewer tasks (12 x 2 repeats on Gemini ~ $1.4-1.9k); smaller b; or drop the 8b solo point and fit λ on 1b-4b. Decide after one DeepSeek pilot run measures its real cost.
- Harvey adds judge cost (up to ~1,100 criteria per data-room run; batch criteria and cache the shared prefix, one judge from a third family validated against the default dual judge).
- Compute: run locally first (M5 Pro, Docker Desktop 16 CPUs / 24 GB since 2026-10-05; AlgoTune containers capped at 8 CPUs / 12 GB by default in `algotune_task`; AlgoTune and Harvey images run natively on arm64). Gate: score the reference solver against itself repeatedly in the pilot; if the speedup spread is more than a few percent, move timing to a box. All arms must run on the same machine. Score final solutions in a separate serial pass. Keep the laptop plugged in and awake (`caffeinate -dimsu`).
- Timing gate result (2026-10-05, `scripts/timing_noise.py`, reference vs itself, dev_eval n=20, 5 runs, host load 3-10): cvar_projection range 2.1% (an earlier run with other containers busy: 0.13%); job_shop_scheduling 12.2% (earlier 7% and 11%). Decision: run locally. The spread on job_shop persists on a quiet Docker, and its reference uses OR-Tools CP-SAT, whose multi-worker search time varies run to run, so a cloud box would likely not fix it (likely, untested). 5 of the 32 selected tasks use CP-SAT (job_shop_scheduling, vehicle_routing, min_dominating_set, set_cover_conflicts, tsp). The verifier's 100 instances average some of this out; the analysis reports every result with and without these 5 tasks.
- Fallback / ProgramBench box (prices fetched 2026-10-05): AWS c7a.8xlarge on-demand, us-east-1 or us-west-2, 32 physical cores (SMT off), 64 GiB, $1.642/h, plus 400 GB gp3 ($32/month while stopped). About $114 for 60 h of use over 2 weeks; $567 if left on. Spot (~$0.62/h) only for agent rollouts, never for timing. Hetzner AX162 (48-core bare metal, ~$690 for 2 weeks plus EUR 304 setup, cannot be stopped) wins only above ~17 days of continuous use. If c7a timings are noisy: c7a.metal-48xl ($9.85/h) for timing runs only. A new AWS account may need a vCPU quota increase for 32+ vCPU.

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | Drop BrowseComp-style search tasks | Jack finds search results uninteresting; Ord already has BrowseComp λ |
| 2026-10-05 | Cut cyber tasks | Cost, containment needs, Lyptus-only infrastructure, publication sensitivity |
| 2026-10-05 | N up to 8 minimum | Jack: N = 2 is not publishable; overrides Sol's pairs-only design |
| 2026-10-05 | Verification is a measured axis, not a task filter | Jack disagrees that only verifiable tasks matter |
| 2026-10-05 | Real-time messaging plus candidate registry, no lead | Matches lab designs; lead identity would confound mixed result |
| 2026-10-05 | Token-vs-dollar comparability across vendors is not a blocker | Jack's call; log both anyway |
| 2026-10-05 | Drop Lean, CritPt, SciCode | Lean: no public milestone benchmark; CritPt: rate-limited aggregate-only grader and mostly grader error (arXiv 2609.13009); SciCode: score barely moves with capability, no mid-run verifier, sequential |
| 2026-10-05 | Harness = Inspect + inspect_harbor + custom multi-agent solver | Both chosen evals exist as inspect_harbor tasks; one harness for all families |
| 2026-10-05 | Build the multi-agent solver ourselves (300-500 lines est.) | Survey found no off-the-shelf option with flat peers, mid-loop message injection, mixed providers and hard per-agent budgets; native lab teams are single-vendor and hierarchical |
| 2026-10-05 | Every arm ends with a step that writes the single deliverable (AlgoTune selector, Harvey merge) | Harbor tasks score one `/app/solver.py` or fixed `.docx` paths |
| 2026-10-05 | AlgoTune gets a dev toolkit and no agent network | Harbor copy gives agents no generator or eval command; public network exposes the AlgoTune repo |
| 2026-10-05 | Convert the 11 Harvey data rooms to local Harbor tasks; custom scorer keeps criterion fraction | Harbor `harveyai/lab` is the launch snapshot without them; its scorer keeps all-pass only |
| 2026-10-05 | Every arm can save candidates (solo and independent runs: private `publish_candidate`; teams: shared registry). Independent arm = separate N = 1 samples, never N agents in one container (`swarm()` refuses it) | Otherwise teams get a larger pick pool and checkpoints, inflating the "communication benefit"; agents in one container share a filesystem, so they would not be independent |
| 2026-10-06 | Drop GLM-5.3-Flash; candidate cheap pair Gemini 3.8 Flash + DeepSeek V4.1 Flash | GLM measured ~20-38 tok/s, ~95% reasoning, no solver.py in two pilot runs, and its API is unreliable (Jack). Gemini solved cvar_projection (22.9x) in 12.5 min at the same b |
| 2026-10-08 | Pilot 2 team arms use the Claude-Code-style tools (SendMessage with teammate-message envelopes, shared task list) and budget warnings at 50/75/90%; no default-vs-Claude-Code A/B | Jack's call. Pilot 1 analysis: agents messaged but did not coordinate (proposal to split work ignored; better approach never saved, budget never checked). Tools shaped like Claude Code's agent-team tools may elicit trained coordination behaviour; budget warnings apply to every arm |
| 2026-10-08 | Pilot 2 adds explicit sharing: `send_file` tool and a prompt line saying teammates' folders are readable (pilot 1 had sharing only implicitly, via registry and readable folders). Compare cross-agent adoption between pilot 1 and pilot 2 | Jack: agents should be able to pass files, folders and other information directly, not only text messages and finished candidates |
| 2026-10-08 | Primary comparison becomes communicating team vs the same number of independent agents, same per-agent cap, same final selector; agents are told their budget, can check it (`check_budget`) and may stop whenever they choose. Team-vs-long-solo λ becomes secondary. Cost-matched check: resample best-of-k from independent runs with k chosen to match the team's realised spend. Pilot model: Claude Opus 5.5 (effort high) on Anthropic credits | Jack's hypothesis is that 2-4 agents that can bounce off each other beat the same number working independently. Forcing agents to use their whole budget is out of distribution; comparing on realised spend alone is hard to read when a solo agent uses a tenth of a team's tokens. Sonnet 5.5 probe submitted itself after 2 min, 8% of a 2M budget, $0.14 measured |
| 2026-10-08 | SUPERSEDES the next row: budgets meter TOTAL tokens (input incl. cached + output), base unit b about 2M per agent (`token_limit(type="all")`, runner default) | Jack: models differ in how verbose they are and how much they re-read, so a budget should capture input and output. Consequences, accepted: the longest solo arm pays for re-reading its growing history, which favours teams somewhat; λ is no longer in Ord's unit, so an output-token λ is also computed post hoc from the logged token types for comparison |
| 2026-10-06 | Budgets meter OUTPUT tokens (incl. reasoning), enforced per agent by `token_limit(type="output")`; λ reported primarily in output tokens, secondarily in dollars (with caching) and total tokens | Ord's λ and OpenAI's charts are in output tokens, so ours is directly comparable. Metering all tokens would charge a long-running solo agent for re-sending its growing history on every call, so at "matched tokens" teams would look better than they are and λ would be inflated |
| 2026-10-05 | Loose, facts-only team prompt; emergent structure measured as an outcome; structured protocol only as an N = 4 side check | Jack: let agents decide whether to form a conductor, split work or work alone. Matches the labs' minimal-structure designs. Wording-only anti-herding prompts have weak evidence (research/PROTOCOL.md) |
| 2026-10-10 | Run to budget: `submit` is refused until an agent has spent 80% of its own budget (runner `--min-spend-frac`, default 0.8), in every arm. Replaces free stopping (decided 2026-10-08). Agents end by budget, time limit or an accepted submit, and finalize picks the best candidate | Pilot 4 agents stopped at 4-18% of their budget, so larger budgets bought nothing and budget scaling was unmeasurable. Matches AlgoTune's protocol ("continuously queries the LM to improve its solution until the budget runs out, at which point we submit its best code") |
| 2026-10-10 | Scorer slots = 1 per 4 agents, on one scorer (one global queue) | Pilot 4: a team of 2 waited 12.4 min of queue time over 8 dev_eval calls on a 2-slot scorer shared by 16 agents |
| 2026-10-10 | Agent time limit is a uniform 16 h backstop for every row (runner default); VM backstop 18 h, scorer 30 h | Jack: under run-to-budget the budget should end runs, not the clock |
| 2026-10-10 | Agents run at reasoning effort xhigh from the next pilot | Jack's call. Checked offline that Inspect sends `output_config.effort = xhigh` for Opus 5.5 |
| 2026-10-10 | The scorer caches the reference-alone baseline (`--cache-alone`, default on); the interleaved speedup timing is unchanged | Jack: re-timing the reference alone in every job roughly doubles a score job. Kept only if the slot-interference pre-flight shows slots do not slow each other; otherwise `--no-cache-alone` |

## Must fix before the Harvey pilot

- Not implemented yet (found in review 2026-10-06): the runner only builds AlgoTune tasks, and there is no Harvey merge/finalize step that writes the single `red-flags-report.md` from a team's work. Add a Harvey path to the runner and a finalize that runs in every arm, with its cost counted against the budget.

Status 2026-10-05 (details in HARNESS.md, Harvey LAB). All fixes are covered by stubbed tests (`tests/test_harvey_grader.py`); none has run against a real judge or container yet.

- Rubric exposure: fixed. Grading runs on the host (`harvey_grader`, rubric read from the host task dir) and task.toml sets `network_mode = "no-network"`; `harvey_task` asserts it per sample. The original rewardkit verifier remains as `harvey_task(rewardkit_parity=True)`, which turns network back on (fixed-deliverable parity checks only).
- Judge fragility: fixed. 4 tries per call with exponential backoff; a still-failing criterion is non-pass and counted in `n_errored`, other results kept. All criteria errored -> the sample errors.
- Judge cost: fixed in code, not measured. Shared instruction + report prefix with Anthropic `cache_control` (automatic prefix caching for OpenAI/Gemini); optional `batch_size` (default 1, Harvey's protocol). Per-scoring tokens (input, cached, cache-write, output) and cost land in `Score.metadata["judge_usage"]`. TODO(pilot): measure cost per scoring and the cache hit rate; decide whether batching is worth a parity check.
- Headline metric: fixed. `Score.value` is `criterion_fraction`; metrics are `mean` (mean criterion_fraction) and `all_pass_rate`.
- Timeouts: raised up front (2026-10-05). Sample limit 4 h (`harvey_task(time_limit=14400)`), so scoring gets 2 h (Inspect gives scoring half the sample limit); agents keep 2 h each via `swarm(time_limit=7200)`; judge concurrency 16, 300 s per call. TODO(pilot): check measured agent and scoring times fit, and tighten.
- Parity: TODO(pilot) grade a few saved deliverables with both scorers (`rewardkit_parity=True`) and check criterion-level agreement, since our message order differs from rewardkit's.

## Pilot findings so far (2026-10-06)

- End-to-end works with a real model (Gemini): dev toolkit used (42 python calls), 2 candidates published, selector picked the published 23.8x candidate over the agent's final working copy (1.0x), so checkpoints matter.
- Finalize is slow: one candidate's dev evaluation hit the 900 s timeout. With N = 8 publishing many candidates, finalize needs a lower per-candidate timeout, fewer dev instances, or parallel evaluation.
- The team (N = 2) real-model run was stopped before completing; messaging with a real model is still untested.
- CPU stayed near idle (agents are API-bound), and the container used ~4 MB of RAM, so running several samples concurrently is feasible; score in a separate serial pass for clean timing.
- Next: one DeepSeek V4.1 Flash solo run to measure its cost and behaviour, one N = 2 team run, then choose the pair, b and task count against the budget.

## Open questions

- Harvey judge: the default (Claude Sonnet 4.6) is outside the cheap pair's families; the frontier check with Sonnet 5.5 needs a swapped judge (Gemini via `harvey_task(judge=...)`; request shape checked offline, no live call yet).
- Whether to reveal teammates' model family in the mixed arm's system prompt.
- Whether the frontier pair check fits the budget after day-2 cost measurement.
- Whether to add a cheap control at N = 4 on pilot tasks where agents are told teammates exist but cannot reach them (separates "knowing peers exist" from communication; research/PROTOCOL.md risk 3).
- Whether to keep a messaging-off (registry only) ablation on a subset. Sol cut it; it would isolate push messaging versus shared files.
- Repeats per cell (2 planned; more if variance is high and budget allows).
- Exact AlgoTune task split (pilot vs held-out), fixed before results are seen.
- Side project, not this week: addressee-identity propensity eval (does behaviour change when the counterpart is labelled human / other-model agent / self-copy?). Appears unpublished; see research/EVIDENCE.md.
