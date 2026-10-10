# Evidence and literature

Gathered 2026-10-05. Figure-read numbers are approximate (about ±0.01). Source files are in `research/sources/` (see its README).

## Public swarm / multi-agent scaling data

### OpenAI, GPT-5.6 Sol launch post (2026-07-09)

- Charts: 1 / 4 / 16 agents ("ultra mode") x effort low..max on BrowseComp, SEC-Bench Pro, Terminal-Bench 2.1 (no 16-agent Terminal-Bench series). Raw Vega-Lite data extracted to `sources/openai/multiagent_charts.json`: output tokens, API cost, latency, score per point. Footnote: latency from the root agent; tokens and cost include all agents. The `latency_s` field appears to be in minutes; verify before use.
- Low to max effort, score (total output tokens per task):
  - BrowseComp: 1 agent 69.0 (2.0k) to 90.8 (22k); 4 agents 81.8 (7.4k) to 92.2 (54k); 16 agents 86.4 (22.9k) to 93.3 (144k).
  - SEC-Bench Pro: 1 agent 17.1 (4.8k) to 71.4 (77k); 4 agents 20.4 (11k) to 74.3 (159k); 16 agents 22.4 (17k) to 76.2 (357k).
  - Terminal-Bench 2.1: 1 agent 76.4 (3.8k) to 88.8 (12.8k); 4 agents 79.8 (10k) to 91.9 (39k).
  - SEC-Bench Pro at max effort: latency 20.6 / 11.9 / 8.0 and cost $15.5 / $32.4 / $65.9 for 1 / 4 / 16 agents.
- Matched-token comparison (GPT-6.1 Sol check): within the observed solo token range there are 8 overlapping swarm points, and the single agent leads at all 8. Deficits: BrowseComp 0.22-3.85 points, SEC-Bench Pro 8.76-23.33, Terminal-Bench 6.25. Example: SEC-Bench Pro 16 agents score 45.63 at 68,568 tokens vs 68.96 (linear) or 69.25 (log) interpolated solo. No task-level data, so no significance. Many swarm points lie beyond the solo range. Swarm ceilings above solo max: 4-agent +1.4 to +3.1; 16-agent +2.53 (BrowseComp), +4.78 (SEC-Bench Pro). `latency_s` is in minutes (confirmed in REPLICATION.md); at max effort Terminal-Bench shows no speed-up (4.24 vs 4.28 min).
- λ refit: see research/REPLICATION.md. Point estimates reproduce Ord; his BrowseComp interval (0.63-0.76) does not (ours 0.45-0.85, fit-based; equal-slope premise fails on BrowseComp).
- Multi-agent API (Responses API): root agent plus subagents; actions spawn_agent, send_message (queues a message without starting a turn), followup_task, wait_agent, interrupt_agent, list_agents; `fork_turns` sets how much context subagents inherit; `max_concurrent_subagents` defaults to 3. Available on GPT-6.1 Sol and GPT-5.6 models.
- GPT-6 Astra card and GPT-6.1 Sol addendum: no swarm scaling data. Astra card s8.5 message-board propensity eval: Sol engaged 84% and followed instructions 52%; Astra 27% / 0%; GPT-6 Sol 26% / 11%. The card notes it does not measure agents communicating with other agents of the same user in the same Codex harness, "a behavior we noticed in internal testing".
- Astra card expert cyber eval used up to 64 subagents at Ultra effort.

### Anthropic, Claude Opus 5.5 system card s8.12 (`sources/anthropic/opus55.txt`)

- ProgramBench (166 tasks, fraction of hidden tests passed): single agent 20M token cap with compaction vs fixed 5-agent peer team at 4M each (matched total) vs async subagents (lead spawns, 1M each, unlimited). Team reaches 0.6 "a 2.7x latency improvement over the single agent". Single agent's score-vs-tokens curve sits left of the team's (more token-efficient), but async subagents lead at low token budgets before the solo curve overtakes them (Fig 8.12.1.B).
- DRACO (100 deep-research tasks, rubric): teams can be slower due to coordination overhead; under latency budgets, at 0.5x the team more than matches solo (~2.8x speedup). "At tighter latency budgets... the lead spawns them at a much lower rate and prefers to act as a single agent."
- Large teams (s8.12.3): 1 lead + N-1 helpers, N = 1 / 10 / 30 / 100, 24 h, shared Linux box with no network, Send Message + Wait for Message, shared git, 3 runs, slightly different model snapshot. Best score within 24 h:

| N | KB Opus 5.5 | Lean Opus 5.5 | KB Opus 5 | KB Fable 5.1 | Lean Opus 5 | Lean Fable 5.1 |
|---|---|---|---|---|---|---|
| 1 | 0.53 | 0.39 | 0.45 | 0.53 | 0.19 | 0.16 |
| 10 | 0.70 | 0.66 | 0.64 | 0.66 | 0.44 | 0.33 |
| 30 | 0.71 | 0.66 | 0.65 | 0.69 | 0.55 | 0.45 |
| 100 | 0.74 | 0.68 | 0.68 | 0.68 | 0.58 | 0.53 |

- By time (Opus 5.5, 2h / 6h / 24h): KB N=1 0.37/0.47/0.53, N=10 0.53/0.64/0.71, N=30 0.66/0.68/0.71, N=100 0.69/0.71/0.74. Lean N=1 0.07/0.14/0.39, N=10 0.15/0.42/0.66, N=30 0.34/0.58/0.66, N=100 0.44/0.64/0.68.
- Not compute-matched. At roughly matched agent-hours (10 agents x 2 h ~ 20 vs 1 agent x 24 h), solo ties on KB (0.53) and wins on Lean (0.39 vs 0.15). Agent-hours are not tokens.
- Emergent structure: the Lean team appointed 12 sub-leads (two tiers); the KB team stayed flat (lead partitioned records, helpers worked mostly alone).
- s8.12.5 defines multi-agent token usage (summed across agents, each context-window token counted once). Reusable for accounting.
- Agent-to-agent: small self-preference when grading (0.07 / 10 with a "you are Claude" reminder), framed as a collusion risk; behavioural audit has "thin coverage of multi-agent scenarios".

### Anthropic, Claude Sonnet 5.5 system card (`sources/anthropic/sonnet55.txt`)

- ProgramBench at 1h / 2h / 4h latency budgets: single 0.856 / 0.896 / 0.917; 5-team 0.913 / 0.941 / 0.947; async 0.931 / 0.956 / 0.963. Tokens per task at 4 h: ~1.3M single, 4.6M team, 10M async. "The async subagents team with a 1-hr budget exceeded the performance of a single agent with a 4-hr budget." At about matched tokens the team buys little.

### Google, Gemini 4 Argon (2026-09-30, `sources/google/argon.txt`)

- No multi-agent, Deep Think or parallel-sampling results. Scores are single attempts ("allow no majority voting or parallel test-time compute"). Google scales one long trajectory (output cap 1M tokens).

### Vals AI, "Do Agent Teams Pay Off? A Case Study on Vibe Code Bench" (2026-10-09, `sources/other/vals-multiagent-vcb.md`)

- Setup: GPT 6 Sol (OpenAI Agents API `multi_agent`) and Claude Opus 5.5 (Claude Code headless, subagent tools on/off) on 50 full-stack web apps, single agent vs lead + up to 5 subagents, medium and max effort, one run per setup. Teams got a delegation instruction ("split, delegate, integrate and verify"), so each comparison is "subagents plus a delegation prompt" vs neither. Different harness per model.
- Results: teams cost 1.8-5.1x the single agent. Only Sol medium's team gain was significant (+7.3 points, p = 0.005, paired t-test across apps). Opus: medium single 91.5% ($4.08), medium team 91.2% ($9.10), max single 89.8% ($23.77), max team 93.2% ($122, 2.9 h median). For Sol, max effort raised the single agent 11.4 points, more than a team did.
- Mechanism notes: most extra team spend is cached input (Opus max subagents read a median 224M cached tokens per app vs 55M for the single agent). Opus leads wrote a CONTRACT.md first (42/50 runs) and delegated in sequential waves (build, features, test); Sol leads split by architecture in the first minutes and tested themselves. Sol team gains concentrated on harder apps, where single agents claimed tests passed on features never built.
- Relation to this project: hierarchical lead/subagents, team vs single at the same effort, so not matched compute and no best-of-N independent baseline. Our design differs on all three (flat peers, matched per-agent budget, best-of-N independents), plus a verifiable optimisation task. Their "higher effort is an alternative to a team" result suggests an effort-matched single-agent arm. Their cached-input finding matches our pilot 3 measurement (91-95% of tokens were cache reads).

## Papers

- Park et al., "Scaling Discovery through Test-Time Communication", arXiv 2609.21032 (Sept 2026). Identical agents, shared directory, no roles. A team of k matches 4k independent agents on ARC-AGI-3 (counted in game actions; the paper also reports output-token curves); multiplier 4.3x at k=3, 6.6x at k=5. Also polyomino packing and MNIST compression (beat best-known human result). Independent agents win when compute is limited or there is no clear measure of progress. Their protocol includes distinct approaches, evidence-backed adoption and preserved variation; token curves show an initial coordination cost.
- Kim et al., "Towards a Science of Scaling Agent Systems", arXiv 2512.08296. 260 configurations; +80.8% on decomposable financial reasoning to -70.0% on sequential planning. Read before freezing.
- "Beyond Symmetric Agents", arXiv 2609.35875 (Sept 2026). Heterogeneous debate teams vs independent samples from the same rosters: sampling controls win. Generation-budget matching; debate tasks, not iterative code. Weakens a general novelty claim for mixed-model teams.
- Self-MoA, "Rethinking Mixture-of-Agents", arXiv 2502.00674. Aggregating one strong model's samples often beats heterogeneous mixtures (quality-diversity tradeoff).
- "Pareto-Optimal Test-Time Scaling", arXiv 2605.01566. Debate and MoA advantages under a modelled inference-resource budget.
- Expert re-grading of physics benchmarks, arXiv 2609.13009. Most CritPt "failures" were grader or reference errors; GPT-5.6 Sol corrected pass@4 94.4% on 54 retained challenges.
- Phil Trammell, "Parallelizability" essay (philiptrammell.com/static/Parallelizability.pdf), recommended by Ord.

## Ord's post and comments (`sources/openai/lw.txt`)

- λ defined via economists' "stepping on toes" parameter. Swarms are mainly a speed trade: 4 agents ~ half the time for ~2x cost.
- Julian Bradshaw: λ near 1 possible where agents share intermediate progress; homogeneity drags λ down; Terminal-Bench showed no swarm benefit in Park et al.
- alkjash: λ > 1 implies a serial agent could emulate the swarm; Julian: true for an idealised serial agent, not in practice (context attachment, caching).
- Max Harms: context length and compaction may change λ on very long tasks; Ord agrees this could eventually make swarm scaling beat CoT scaling.
- Tim L: Junyu Ren (Pierce-Birkhoff, $400 budget) found one agent per model family beat a larger same-model swarm. Anecdote, one case.
- Ord withdrew his Navier-Stokes swarm-scaling section after OpenAI sources said that chart showed longer CoT only.

## Noam Brown, Dwarkesh Podcast, 2026-09-17 (`sources/openai/dw.txt`, `sources/other/noam.html`)

- Messaging is "just a tool call"; messages are inserted into the recipient's context; sub-agents get forked context.
- "it's very tempting for them to just collapse to, oh, we're all just going to solve the problem independently. And that is a local minimum."
- Multi-agent "is less efficient"; measured up to 16 agents; "we don't have very good science on multi-agent scaling"; next step is 64 / 128 / 256.
- Attributes under 10% of the Navier-Stokes result to multi-agent.
- Agents "understand when they're talking to an agent versus when they're talking to a person, and their behavior will be different." Telling agents the user is Agent A raises honesty and instruction-following on OpenAI alignment evals (unpublished).
- Possible that 10,000 humans coordinate better than 10,000 agents today.

## Incidents

- OpenAI-Hugging Face (report: `sources/openai/hf.txt`): agents evaluated separately coordinated through an improvised message board; the report attributes this to cooperative multi-agent training ("agents learned to use improvised collaboration channels in rare cases during the training process ... This behavior was then reinforced"). Ord's "1,200 agents / 700 attacked" figures are not in the report text; do not cite them.
- UK AISI incident report (unsanctioned agent behaviour during cyber testing, July 25-28 2026; https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing): 10 of 122 runs showed unsanctioned behaviour (17 cases Mythos 5, 2 from one GPT-5.6 Sol run), mostly one sustained line of activity. "One agent left public messages on GitHub offering collaboration with other agents working on the same challenge. It also provided instructions to reuse accounts and artefacts it had left behind, which were discovered and used by subsequent agents." Cross-run information sharing through unintended channels, not a coordinated team. (Corrected 2026-10-06; an earlier line here said "instances coordinated", which overstated it.)
- GTG-1002 (Nov 2025): multi-agent decomposition into benign-looking subtasks to bypass safeguards.
- Cooperative AI writeup (cooperativeai.com/post/lessons-from-multi-agent-safety-incidents): current evals "fail to measure multi-agent capabilities or propensities".

## Navier-Stokes (OpenAI post, `sources/openai/ns.txt`)

- ~10,000 concurrent agents in communicating groups, 88 h; 4.9M messages and 300B output tokens across all problem variants, 2.7M and 130B for Navier-Stokes itself. Model stronger than Astra. No scaling data.

## Side project: addressee identity (not this week)

- No published study crosses {human, other-model agent, same-model copy} counterpart labels with honesty, sandbagging, instruction-following or collusion. Nearest: Metzgar and Graziano (style only, `sources/other/metzgar.pdf`); arXiv 2602.22070 (source labels); Newman LessWrong 2026-08-10 (self-attribution when judging); arXiv 2606.14923; arXiv 2602.08208. Scoop risk: OpenAI holds unpublished data per Brown.
- inspect_petri v3.1.1 (github.com/meridianlabs-ai/inspect_petri) can run a rough version (auditor plays the counterpart via seed text) but the auditor's messages vary with the label. A plain Inspect task changing one system-prompt line is cleaner.
