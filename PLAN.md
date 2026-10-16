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

## v6: elicitation redesign (pre-registered 2026-10-11, before any held-out run)

Question: can a diversified, sharing team of 4 beat equally budgeted, equally diversified independent search on held-out AlgoTune tasks, and does diversification alone beat loose independents? Jack approved the design on 2026-10-11; the decisions log row of that date gives the reasons.

- Model and budget: Claude Haiku 5.5, effort xhigh, $0.10 per agent, run to budget (80% submit gate), 1 h prompt cache, 2 CPUs per agent (agent boxes only run local experiments; every timing is on the remote scorer), remote scorer. Groups of N = 4, 2 repeats per task.
- Arms per task and repeat: (1) 4 loose independents, today's loose solo prompt; (2) 4 diversified independents, no messaging, agent priority index 0-3 set per plan row (runner `--elicit diversified --priority-index i`); (3) 1 diversified team of 4 (`--arm team --n 4 --elicit diversified`), messaging and registry as before, agent i gets priority i. The pilot adds (4) 1 two-phase diversified team of 4 (`--blind-frac 0.5`): the same prompt blocks and priorities, but each agent works blind until it has spent half its own budget (messaging tools refused, nothing delivered to it, list_candidates shows only its own candidates; the prompt states the rule from the start), then its channel opens with a notice. Its candidates go into the shared registry throughout and selection is unchanged. The shared filesystem is not access-controlled, so blind-phase tool calls that name the registry or a teammate's directory are logged per agent (`blind_peeks`). Main keeps one team arm, chosen after the pilot. Motivation: The Interaction Tax (arXiv 2608.23541, research/EVIDENCE.md) found full-solution exchange collapses diversity within one round.
- Prompt text: `swarm.EXPLORATION_BLOCK`, `swarm.PRIORITIES`, `swarm.priority_text` and `swarm.TEAM_SHARING`, added to the default protocol. Both diversified arms get the same exploration block and the same priority for the same index; the team's sharing line replaces "How you work together, if at all, is up to you." No wording on precision, number formats or risk appetite. Each run records the variant (`protocol` in run metadata, `elicit` in swarm metadata) and each agent its `priority_index` and `priority`.
- Selection, identical for every arm (`scripts/select_portfolio.py`): a portfolio is one team run or one group of 4 independent runs (`<group>-i<k>`). Stage 1 ranks every candidate in the portfolio by its recorded 20-instance dev score (distinct sources). Stage 2 re-times the top 3 distinct sources with the final verifier on 100 dev instances (seeds 500,000-500,099, scorer job `score_dev`) and picks the fastest valid one. The pick is scored 3 times on the hidden instances; the median is the portfolio's result. The other two shortlisted sources are scored once each, and the best hidden score in the shortlist (the pick's median, the others' single score) is reported only as a labelled oracle bound. Every candidate's source is kept in the log (finalize `sources`).
- Analysis: per-task mean log speedup per arm, contrasts within task (diversified team minus diversified independents isolates communication; diversified minus loose independents isolates the prompt), bootstrap over tasks with repeats nested, each portfolio one observation, failures included. Claims are pre-registered as "a 20% or larger gain, or no evidence of one"; a claim of "no gain" is not available at this sample size.
- Plans: `plans/v6-pilot.tsv` (the two pilot tasks generalized_eigenvalues_real and optimal_advertising, 1 repeat of each of the four arms, 32 agents, max $3.20) and `plans/v6-main.tsv` (the 7 tasks below x 2 repeats x 3 arms, 168 agents, max $16.80; SPEND_CAP $16). Rows are `<prefix>-<task>-r<k>-<arm>[-i<j>]`, repeat 1 of every task before repeat 2, team row first in each group. Independents run on c7a.xlarge, teams of 4 on c7a.4xlarge; sinkhorn rows on m7a.xlarge / m7a.4xlarge, because its 100 verifier problems hold 7.0 GiB and a c7a.xlarge agent box has about 5.6 GiB (measured problem sizes: every other task, pilot tasks included, at most 0.9 GiB per 100 problems).
- Driver: `scripts/v6.sh <plan> <SPEND_CAP>` (`scripts/v6_drive.py`). It reads the AWS quota at launch and sizes the scorer: at most 11 slots (c7a.24xlarge), at least one slot per 8 agents running at once, room left for agent VMs; at 300 vCPUs that is 11 slots with at most 88 agents at once (51 fit), for the pilot 5 slots. Scorer slots get 16 GB (the 100 sinkhorn problems). The whole plan runs as one fleet plan, so tasks run in parallel as the quota allows. Each task-repeat group's portfolios are selected as soon as its runs have ended, one selection at a time at concurrency 6. No further row starts once measured spend of ended runs plus the full budget of running agents plus the largest row would pass SPEND_CAP.

### Held-out task selection

Rule, fixed before any held-out run: start from the 26 held-out tasks in `data/algotune_split.json`. Exclude (a) tasks whose reference runs a combinatorial search solver: OR-Tools CP-SAT (multi-worker search time varies run to run; see the timing gate under Budget) or SAT search and (b) tasks whose reference is a single library call on a fixed linear transform (the dst-like case, where one library or precision swap sets the ceiling). Eigen-decompositions and other single calls that compute a decomposition or solve an optimisation are not fixed transforms and stay in. Assign each remaining task to an algorithmic family by what its reference computes, then draw 7 families with `random.Random(20261011).sample(sorted(families), 7)` and, in the drawn order, one task per family with `rng.choice(sorted(tasks))` from the same generator. Family assignments were written from the reference `solve` code only.

| Task | Reference | Decision |
|---|---|---|
| min_dominating_set | OR-Tools CP-SAT | excluded (a) |
| set_cover_conflicts | OR-Tools CP-SAT | excluded (a) |
| tsp | OR-Tools CP-SAT AddCircuit | excluded (a) |
| fft_cmplx_scipy_fftpack | `scipy.fftpack.fftn` | excluded (b) |
| fft_real_scipy_fftpack | `scipy.fftpack.fftn` | excluded (b) |
| eigenvalues_real | `np.linalg.eigh` | family: dense symmetric eigenproblems |
| generalized_eigenvectors_real | Cholesky reduction + `eigh` | family: dense symmetric eigenproblems |
| two_eigenvalues_around_0 | `np.linalg.eigvalsh` | family: dense symmetric eigenproblems |
| generalized_eigenvalues_complex | `scipy.linalg.eig(A, B)` (QZ) | family: dense nonsymmetric generalized eigenproblems |
| generalized_eigenvectors_complex | `scipy.linalg.eig(A, B)` (QZ) | family: dense nonsymmetric generalized eigenproblems |
| sparse_lowest_eigenvalues_posdef | `scipy.sparse.linalg.eigsh` | family: sparse iterative eigensolvers |
| sparse_lowest_eigenvectors_posdef | `scipy.sparse.linalg.eigsh` | family: sparse iterative eigensolvers |
| pca | sklearn PCA | family: SVD and low-rank factorisation |
| randomized_svd | sklearn `randomized_svd` | family: SVD and low-rank factorisation |
| chebyshev_center | cvxpy LP (Clarabel) | family: convex programs through a general solver |
| qp | cvxpy QP (OSQP) | family: convex programs through a general solver |
| quantile_regression | sklearn QuantileRegressor (HiGHS LP) | family: convex programs through a general solver |
| robust_linear_program | cvxpy SOCP (Clarabel) | family: convex programs through a general solver |
| lasso | sklearn Lasso (coordinate descent) | family: iterative regression fitting |
| least_squares | `scipy.optimize.leastsq` (Levenberg-Marquardt) | family: iterative regression fitting |
| min_weight_assignment | `min_weight_full_bipartite_matching` | family: graph algorithms |
| shortest_path_dijkstra | `scipy.sparse.csgraph.shortest_path` | family: graph algorithms |
| btsp | pysat SAT search + networkx | excluded (a): SAT search; reference solve times on 5 instances at the task's size were 1 ms to 19.9 s (Mac, 2026-10-11) |
| clustering_outliers | HDBSCAN | family: density-based clustering |
| sinkhorn | POT `ot.sinkhorn` | family: entropic optimal transport |
| unit_simplex_projection | sort-based projection in numpy | family: sort-based projection |

Draw (10 families, 7 drawn): iterative regression fitting -> least_squares; entropic optimal transport -> sinkhorn; sparse iterative eigensolvers -> sparse_lowest_eigenvalues_posdef; dense nonsymmetric generalized eigenproblems -> generalized_eigenvalues_complex; graph algorithms -> shortest_path_dijkstra; SVD and low-rank factorisation -> pca; density-based clustering -> clustering_outliers. Families not drawn: dense symmetric eigenproblems, convex programs through a general solver, sort-based projection.

The v6 held-out tasks: clustering_outliers, generalized_eigenvalues_complex, least_squares, pca, shortest_path_dijkstra, sinkhorn, sparse_lowest_eigenvalues_posdef. Reference solve times on 5 dev instances each (Mac): 14-182 ms, no heavy tail.

Revision history (all before any held-out run, 2026-10-11): the first version drew 8 tasks with btsp eligible (btsp, clustering_outliers, least_squares, qp, shortest_path_dijkstra, sinkhorn, sparse_lowest_eigenvalues_posdef, unit_simplex_projection). The budget then fixed 7 tasks, and rule (a) was widened to SAT search after btsp's reference timing was measured; the draw was rerun with the same seed.

## v7: harness changes (2026-10-14, branch v7)

Built after the v6 reviews (four transcript reports, the harness audit, the anomalies and selection-regret analysis, and the Fable and Sol second opinions, 2026-10-13). The experiment design and task set are decided separately. Every change is a runner flag or a selector option, so v6 plans reproduce unchanged; the only new defaults are the alone primary score under `--experiment v7` and publish notices for `--arm team --elicit diversified-v7`. A v7 row passes `--experiment v7 --elicit diversified-v7 --end-on-budget --isolate-agents --agent-memory-mb M --disjoint-dev-seeds --tool-fixes` (independents add `--priority-index k`; every row of a comparison gets the same M). The driver runs `EXPERIMENT=v7 scripts/v6.sh <plan> <cap> <ref> --group-start --validity-n 1000`, plus `--validity-n-task` for slow-checking tasks (below).

1. Per-agent boxes (`--isolate-agents --agent-memory-mb M`): team agent i gets its own container, the box an independent gets: `--cpus-per-agent` CPUs on its own cpuset (i x c .. i x c + c - 1), M MiB, the same image and environment. The registry and every agent's working directory (with its inbox) are named volumes mounted in every agent box, never in the checker. Each agent's bash/python, publish, send_file and dev_eval file reads run in its own box, so background processes stay there; every box is CPU-monitored and cleaned up. A sitecustomize sets `os.cpu_count()` to the box's CPU count (the task image's Python 3.12 reports the host's, 16 on a team VM against 4 on an independent's). Mock run on the Mac (2 CPUs, 4,096 MiB): independent and both team agents read nproc 2, `os.cpu_count()` 2, affinity of 2 CPUs (0-1, and 2-3 for agent_1), OpenBLAS 2 threads in numpy's and scipy's copies, numba 2 threads, memory.max 4 GiB, cpu.max 2 CPUs; a background process agent_0 started was killed in agent_0's box at cleanup and agent_1's box had none.
2. Run to budget (`--end-on-budget`): no submit tool, no submit gate, no "agent_k has finished" notices. An agent ends when its budget is used up or at the time limit, both ordinary ends (end_reason `cost`/`token`/`time`), and finalize runs on published candidates and final workspaces as before. After a turn with no tool call the agent is told: "Your run continues until your budget is used up. What is the next concrete improvement you will make? Start it now with a tool call."
3. Publish notices (`--publish-notices`, default on for team + diversified-v7): every publish adds one line to each running teammate's next tool result (and ends a teammate's wait_for_message): publisher, entry id and registry path, the latest dev_eval of that exact file if any ("dev_eval 3.21x on n=20, seed 0"), and the note. `swarm.registry_notices` logs each notice with its recipients and when it reached each.
4. Disjoint seed ranges, written down once in `algotune_devkit.seed_ranges` (generate_problem seeds):

| Use | Seeds |
|---|---|
| Agents' dev_eval | 10,000 + s for s + n below 50,000 with `--disjoint-dev-seeds` (10,000-59,999); v6: below 400,000 |
| Finalize (stage 1) | 60,000-60,019 |
| Stage 2 (score_dev) | 500,000-500,099 |
| Validity pass | 600,000 + i, i < n (range reserved to 699,999) |
| Hidden | offset + 0..99, a secret offset in 1,000,000-2,147,482,647, one per experiment |

5. Validity pass (`select_portfolio.py --validity-n 1000`, scorer job kind `validity`, `validity_check.py`): stage-1 sources, in rank order, are checked untimed with is_solution on the same fresh instances, one instance in memory at a time, until 3 pass; a source with any invalid instance or a failed job is dropped and the next stage-1 source backfills, identically for every arm. Each portfolio row records every check (n, invalid count, first invalid instances and errors). Measured cost per source at 1,000 instances (2026-10-14, Mac Docker arm64 on 8 CPUs with the verifier's thread settings, the reference standing in for a candidate, 10 instances per held-out task; not measured on the scorer's x86 CPUs): 0.1-8.3 min for 24 of the 26 held-out tasks (v6 tasks: clustering_outliers 3.0, least_squares 5.8, pca 6.7, shortest_path_dijkstra 1.9, sinkhorn 4.8, sparse_lowest_eigenvalues_posdef 4.3 min), generalized_eigenvectors_real 13.1 min and generalized_eigenvalues_complex 92 min. That task's is_solution takes an SVD per eigenvalue (5.4 s per instance), so 1,000 instances pass the job's 3,600 s limit and every source would be dropped; give it `--validity-n-task generalized_eigenvalues_complex=300` (about 28 min) or fewer.
6. Alone-reference speedup as the primary score: the verifier also prints "Alone score" (total of the reference's alone minimums, from the thread guard's alone timer, over the solver's total on the same instances, with AlgoTune's rules: invalid or below 1 scores 1), and dev_eval records `alone_speedup`. With `--primary-score alone` (default for `--experiment v7`) a run's Score.value is the alone score and `algotune_score` keeps AlgoTune's; selection stage 2 and hidden rank by the logs' primary score and record both. The 1.15 reference-inflation guard stays; there is no lower-bound guard. Stage 1 still ranks by the recorded interleaved dev score, and dev_eval's printed report is unchanged.
7. `--tool-fixes`: SendMessage's summary is optional; check_budget shows dollars to 4 decimals and the percent remaining.
8. Operations: `scorer.sh pull` copies the scorer logs to `logs/_scorer/<instance>/`, the driver pulls every 10 min and `down` retries 5 times before terminating; the scorer client gives every job a request id that the service deduplicates, so a retried POST never starts a second job, and selection ids are fixed by the runs, portfolio, source and job place, so a restarted selection on the same scorer gets its jobs back (`--fresh` re-times); network errors are retried for up to 30 min; a failing connection from the Mac re-opens SSH and the scorer port to its current IP (`scorer.sh allow-me`, `common.sh operator_ip_refresh`); select_portfolio writes each portfolio as soon as it is done.
9. Group starts (`fleet.py plan --group-start`, `v6_drive.py run --group-start`): a task-repeat group's rows start only when all fit, then together in file order (team row first), with backfill by whole groups.
10. Fresh hidden instances: `--experiment v7` scores with `data/.algotune_seed_offset.v7` (gitignored, chosen once on 2026-10-14, at least 100 seeds from v6's; it lives in the v7 worktree's `data/` until copied to the main checkout); v6 keeps `data/.algotune_seed_offset`. The scorer serves one experiment (`scorer.sh up --experiment`, reported in /health), and a run or a selection refuses a scorer serving another.
11. `--elicit diversified-v7`: EXPLORATION_BLOCK and the priority line unchanged, plus `swarm.V7_PERSISTENCE` in every arm, and `swarm.V7_TEAM` in place of TEAM_SHARING for teams; refused without `--end-on-budget`; recorded as `protocol` (run) and `elicit` (swarm).

Not built: non-blocking dev_eval (deferred).

## v7: task screen, strata and scorer checks (pre-registered 2026-10-16, before any v7 run or scorer check)

Written and committed before any Phase B scorer job or any v7 agent run. The screen's evidence and reasoning are in `analysis/v6-followup/v7-tasks.md` (2026-10-14); local measurements made for this section are in `analysis/v7-prep/`.

**Pool.** The frozen headroom rule (best published speedup in [1.5, 10], median at least 1.05) extended once: the upper bound is raised to 50x, everything else unchanged, tasks in (50, 100] not screened. Pilot tasks are barred. Rules (a) and (b) of the v6 selection apply (rule (a) read to include MILP branch and bound). Duplicates are dropped (sparse_lowest_eigenvectors_posdef).

**Local screen (A1-A5, done).** A1 upstream fixes ported or the task excluded; A2 no degenerate instance in 1,000 and no generator that reads global RNG state; A3 reference-time ESS at least 50 per 100 and no instance above 10% of the total; A4 no trivial output accepted on 30 instances; A5i the same output never gets different verdicts; A5ii rare failures admitted only with the selection validity pass of 1,000 instances; A5iii the pass must cost at most 30 core-minutes per candidate. Results: v7-tasks.md, plus:

- Task fixes (`tasks.TASK_PATCHES`, experiments after v6 only; v6 keeps Harbor's code): pca and randomized_svd get upstream AlgoTune's verifier (HEAD dff9914), verbatim, in the agents' toolkit, the instruction, the scorer's toolkit and the verifier; sparse_lowest_eigenvalues_posdef's generator gets `random_state=rng` (our patch; upstream has the same bug). optimal_advertising (upstream's NaN generator fix) is a pilot task, outside the pool, and is not patched.
- Rescreen of the three patched tasks with the same scripts (`analysis/v7-prep/rescreen/`): A2 0 degenerate of 1,000 each; A3 ESS 89-96 per 100; A4 zeros and another instance's output rejected 30/30 and 29/29; the trivial randomized_svd output (U = I, S = 0, V = I) 0/20 and a rotated pca subspace 0/20; A5i the pca reference and its eigh variant accepted 10 of 10 calls on each of 5 instances (the verifier re-runs a randomized sklearn PCA); sparse_lowest instances now identical under different global seeds. Published solvers under the ports: pca's float32 and ascending-order solvers fail 400/400, the float64 descending ones pass 0/400 invalid (best GPT-5 Pro 2.77 locally); randomized_svd's top 3 pass 0/400 (best 4.63); sparse_lowest's top 3 pass 0/400 on the seeded generator (4/400 and 1/400 before). randomized_svd is readmitted.
- stable_matching: the published Cython solvers, compiled locally, run at 24.7x (Gemini 3.1 Pro), 12.5x and 12.1x in Harbor format, 0/400 invalid; Gemini 3.1 Pro's also runs as one solver.py that compiles its .pyx with pyximport at import. Its Harbor headroom is demonstrated.
- Memory: tasks whose 100 generated problems take more than 4 GiB on the Mac (sinkhorn 8.9, stable_matching 8.5, linear_system_solver 6.3, lasso 5.5, two_eigenvalues_around_0 5.2, min_weight_assignment 4.3) run on m7a with 13,056 MiB per agent box; every other task on c7a with 5,376 MiB (`scripts/make_v7_plan.py`). Scorer slots get 16 GB.

Candidates after the local screen: 30 tasks, `data/algotune_split.json` key `v7` (runner `--split v7`).

**Strata (frozen now; never exclusions).** Rule 1, the published route class from the published solvers' code: S (every solver within 1.5x of the best gets most of its gain from one change; no other component above about 1.2x), M (two or more components each about 1.3x or more, acting on different parts, stacking), A (several routes replace the same computation), U (none of these). Rule 2, overrides to S: (a) converting the input lists takes at least 50% of the reference's time; (b) a one-change variant of the reference reaches at least 75% of the best published solver's local log speedup in Harbor format (1 thread, 25 instances, variants named in the route reports before timing, each alternative tested alone). Rule 2(b) was applied to every task of route class M (11 tasks, `analysis/v7-prep/ablations/ablations.md`; 7 moved to S) and earlier to 7 others; it was not applied to A and U tasks. Result, per task in `data/v7_tasks.json`: S 21, M 4 (ode_lorenz96_nonchaotic, shortest_path_dijkstra, l1_pruning, randomized_svd), A 3 (robust_kalman_filter, quantile_regression, clustering_outliers), U 2 (generalized_eigenvectors_real, least_squares).

**Scorer checks (Phase B), one c7a.24xlarge scorer serving experiment v7, `scripts/v7_scorer_checks.py`.** For every candidate: 3 score jobs of the reference against itself (A6i), 3 score jobs of a known fast solver (A6ii; the fastest published single-file solver with 0 invalid and 0 exceptions in the local run, or for stable_matching the one-file Cython solver; `analysis/v7-prep/fast_solvers/`), 1 final_eval job of that solver on finalize's stage-1 instances (A6iv), and 1 validity job of the reference on 1,000 fresh instances (300 for generalized_eigenvalues_complex). Scores are the primary alone scores. A task is excluded when:

| Check | Exclude when |
|---|---|
| A6i | the reference's 3 alone raw speedups differ by more than 5% (max/min) |
| A6ii | any fast-solver job has reference inflation outside [0.87, 1.15], or a first-call ratio above 5 from a solver that does not compile on its first call (numba or pyximport solvers above 5 are flagged and read instead) |
| A6iii | the fast solver's 3 alone scores differ by more than 10%: 3 more jobs, and exclude when the 6 differ by more than 1.25x (two timing modes 1.25x apart, or noise as wide); whether the six look bimodal is reported |
| A6iv | never excludes: flag when the stage-1 dev score and the hidden interleaved median differ by more than 1.28x (log 0.25) |
| Reference validity | the reference fails any of its validity instances |

A job that fails (crash, timeout, out of memory) is rerun once before any verdict; a task whose job fails twice is excluded. The reference's validity run time sets the main run's `--validity-n-task`: 1,000 (300 for generalized_eigenvalues_complex) when it takes at most 1,800 s, else cut in proportion to a multiple of 100. Excluded tasks are not replaced and the pool is not refilled. Since 30 or fewer pass, there is no draw: every passing task is in the final list. Once, on the same scorer: `scripts/slot_interference.py` (never run on a multi-slot scorer) with one repeat.

**Calibration tasks (Phase C), fixed now.** From the final list, `rng = random.Random(20261016)`, `m = rng.choice(sorted(final M tasks))`, then `s = rng.choice(sorted(final S tasks))` from the same generator. If no M task survives Phase B, the M draw uses the A tasks instead and the report says so. The calibration runs at the main design (Opus 5.5 xhigh, $5 per agent, diversified team of 4 vs 4 diversified independents). Amended 2026-10-16 before any calibration run (Jack): if no harness, prompt, selection or budget change is made after the calibration, its 2 groups are the first groups of the main Opus run and their tasks stay in the main set; if any such change is made, the two calibration tasks become pilot tasks and leave the main run's held-out set (pilot and held-out tasks never overlap).

**Analysis of the main run (fixed now, in addition to the v6 section's).** Every result for every final task; the bootstrap resamples tasks with repeats nested, and as a sensitivity resamples families. Pre-registered sensitivity analyses: without stratum S; M only (exploratory, few tasks); new tasks only (the v6 seven are labelled a replication; v7 pca is a different task from v6 pca because of the verifier port).

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
| 2026-10-10 | Scoring fixes: glibc heap trimming off in every timed process (MALLOC_* in the container environment); scorer containers and the local checker use the cpuset only, without a CPU quota; the scorer VM holds /dev/cpu_dma_latency at 0; the alone-baseline cache is off by default. The official speedup stays AlgoTune's (interleaved reference) and the reference_inflation gate stays at 1.15 | Scoring investigation: the reference check on generalized_eigenvalues_real measured heap trimming (reference vs itself 0.91; 4 of 8 pilot-5b solvers invalid at 1.14-1.20). With trimming off all 8 read 0.99-1.02. The quota throttled about 1% of a busy run. C2 exits cost short multi-threaded calls 800 us. On BLAS-bound tasks the idle-measured cached baseline read about 0.09 higher under load |
| 2026-10-11 | v6 pivot: Haiku 5.5 at $0.10 per agent, groups of 4, three arms per task (loose independents, diversified independents, diversified team of 4), two-stage dev selection for every arm, 8 held-out tasks x 2 repeats chosen by a written rule (section "v6: elicitation redesign"). plans/haiku3-dst.tsv cancelled, not run | Both pivot reviews (Codex Sol and Fable, 2026-10-11): an elicited team against a loose solo prompt cannot attribute a gain to communication, so a diversified-independent control with the same exploration text and priorities isolates the channel. Eliciting the strongest team protocol is legitimate under METR's elicitation guidelines when the comparison arm gets comparable effort. The draft's risk-taking wording was removed because on dst it pushes the one float32 decision that sets the Haiku score, a propensity change rather than collaboration. Task breadth over repeats: the unit of resampling is the task, and Haiku team scores on dst spread 4.1-11.8. haiku3 on dst would sit at the float32 ceiling at N = 8 and 16 whatever the agents did. Pre-registered claim: "a 20%+ gain, or no evidence of one", never "no gain" |
| 2026-10-11 | v6 revisions before any run: 7 held-out tasks (budget: about $20 of credit including the pilot; main SPEND_CAP $17); rule (a) also excludes SAT search (btsp); 2 CPUs per agent for every v6 row; main as one plan with all tasks interleaved and per-group selection while other rows run; a two-phase team arm (blind until 50% of budget) in the pilot only | Jack. btsp's reference took 1 ms to 19.9 s per instance (5 instances, Mac), the timing-noise reason behind rule (a); agent boxes time nothing, so 2 CPUs suffice and let independents run on 4-vCPU VMs; the two-phase arm tests whether delaying exchange keeps the diversity that full-solution exchange collapsed in The Interaction Tax (arXiv 2608.23541) |
| 2026-10-12 | After the v6 pilot, before main: main keeps the diversified team (no two-phase rows); dev_eval refuses seed + n >= 400,000 and returns an agent's identical repeat (same source hash, n, seed, reps, size) from the record without a scorer job; m- and r-family rows fall back only to families with as much memory per vCPU; selection concurrency 6; main SPEND_CAP $16 | Pilot: the diversified team beat diversified independents on both pilot tasks, while the two-phase team (channel open only from 50% to the 80% submit gate) fell to the loose level on optimal_advertising; 56% of dev_eval calls re-timed an unchanged file and seeds reached 104,000 while agents padded to the gate; sinkhorn's 100 problems peak at 9.6 GiB in the agent box; serial selection at concurrency 3 would take about 13 h for main; about $16.4 of credit left after the pilot's measured $2.62 |
| 2026-10-14 | v7 per-agent boxes (`--isolate-agents --agent-memory-mb`): each team agent gets an independent's container, files shared on volumes, `os.cpu_count()` set to the box's CPUs | v6 team agents shared one 8-CPU box: 56 local bash timeouts against 1, both wrong negatives came from contended local timings, and sinkhorn agents read `os.cpu_count()` = 16 and chose 16 OpenMP threads (1.2-1.4x dev against 6.8-7.6x for independents on 2 CPUs) |
| 2026-10-14 | v7 run to budget (`--end-on-budget`): no submit, no gate, no finish notices, a continue message after text-only turns | Agents published their last candidate at about half their budget and spent about 40% padding to the 80% gate; finish notices told teammates that stopping was normal (Fable review) |
| 2026-10-14 | v7 publish notices to teammates, default on for team + diversified-v7 | The channel opened when an agent chose to write (mean spend 0.45, or never) while first publishes came at 0.25-0.36; scores and method notes are the low-bandwidth signal the Interaction Tax paper recommends |
| 2026-10-14 | v7 disjoint seed ranges (`--disjoint-dev-seeds`, `algotune_devkit.seed_ranges`) | dev_eval seeds 50,000+ reached finalize's stage-1 instances; five v6 calls did |
| 2026-10-14 | v7 validity pass at selection (`--validity-n 1000`, per-task `--validity-n-task`), dropped sources backfilled from stage 1 | least_squares r1: a solver failing 1.2% of instances passed 120 dev instances and scored 1.00 on hidden; 100 stage-2 instances miss such a solver 29% of the time |
| 2026-10-14 | v7 primary score against the reference timed alone (default for `--experiment v7`); AlgoTune's interleaved score kept beside it; 1.15 guard kept; no lower-bound guard | pca r2: thread state the pick left behind moved the interleaved reference, so one source scored 12.72, 7.25 and 7.06 on identical instances; a lower bound would invalidate legitimate numpy-LAPACK solvers (14 pca sources read 0.66-0.86) |
| 2026-10-14 | v7 `--tool-fixes`: SendMessage summary optional, check_budget to 4 decimals with percent remaining | 11-16 SendMessage calls failed on the required summary; padding agents polled a 2-decimal display that looked stuck |
| 2026-10-14 | v7 operations: scorer logs pulled every 10 min and retried at teardown, deduplicated job request ids, 30 min network retries, operator IP refresh, per-portfolio selection output | The main scorer's jobs.jsonl was lost at teardown; the Mac's IP changed twice and cut selection off |
| 2026-10-14 | v7 group starts (`--group-start`) | Teams started 83-166 min after their group's independents in 10 of 14 v6 groups |
| 2026-10-14 | v7 secret offset (`--experiment v7`, `data/.algotune_seed_offset.v7`); v6 keeps its own | The v6 hidden instances informed the v6 diagnosis (Sol review); v6 must still reproduce |
| 2026-10-14 | v7 elicitation `--elicit diversified-v7` (persistence block in every arm, v7 team block in place of TEAM_SHARING), only with `--end-on-budget` | Fable's tightened draft with the Sol review's "smallest implementation" wording; the text says the run ends at the budget, which is true only without submit |
| 2026-10-16 | v7 task screen, strata, scorer-check thresholds and calibration-task rule pre-registered (section "v7: task screen, strata and scorer checks"); 30 candidates in `--split v7`; pca and randomized_svd on upstream's verifiers, sparse_lowest's generator seeded (v7 only) | Jack approved the scorer checks and an Opus calibration (2026-10-16). Harbor's pca accepts rotated subspaces and randomized_svd a no-computation answer; sparse_lowest's instances changed at every scoring |
| 2026-10-16 | v7 agent memory 5,376 MiB per box on c7a, 13,056 MiB on m7a for tasks whose 100 problems exceed 4 GiB | The same value in every row of a task; both fit the Docker hosts v6 measured with headroom |
| 2026-10-16 | Calibration groups count toward the main Opus run when nothing changes after them (amends the calibration-task rule above, before any calibration result) | Jack: about $80 of the $200 Anthropic credit; as pilot-only data the main run would get about 2-3 groups |
| 2026-10-16 | AlgoTune bundles (one solver for K unrelated tasks, geometric-mean score) rejected | Jack: the packaging builds in a team advantage. Open question kept: what counts as a task; look for setups closer to real-world agent use than one task, one solver, where any decomposability comes from the task itself |
| 2026-10-16 | Agent boxes on t3a (on-demand, unlimited credits): t3a.large per independent, t3a.2xlarge per team of 4, t3a.xlarge / m7a.4xlarge for big-memory tasks; scorer on-demand c7a.12xlarge | AWS credits exhausted (real spend from 12 Oct); v6 agent boxes averaged 2-7% CPU; spot rejected (c7a spot placement scores 1-3, interruption bands 10-20%) |

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
