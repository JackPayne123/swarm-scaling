I’ll read the reports, prompts, selection code and evidence notes, then check the prompting guidance and power calculation. I’ll keep this read-only and won’t launch resources or spend API money.

I’m using the OpenAI Docs and documentation lookup skills to check the persistence claim and Anthropic’s guidance against primary sources.

The reports point to two distinct issues: agents often stopped useful exploration while budget remained, and team agents had a different local benchmarking environment. I wouldn’t rate the CPU confound as small: the sinkhorn thread choices and pca’s discarded branch give concrete paths from the environment to the outcome.

Context7 failed to fetch Anthropic’s docs, so I’m checking the official pages directly.

The oracle result is near parity too, so the final selector alone doesn’t explain the missing team gain. I’m also checking the statistical unit: there are seven tasks with two repeats each, which supports a narrower claim than fourteen unrelated tasks would.

I would keep “one best solver” as the outcome. It matches the stated question. Scoring breadth instead would test a different claim about team capability.

One proposed messaging fix is already implemented: pending messages are appended as soon as a successful `dev_eval` returns. The reported delay is time spent inside the blocking call. Reducing it would require asynchronous evaluation or interruptible tool calls; changing the delivery wrapper wouldn’t help.

I’m using the Inspect log-reading skill to spot-check the decisive transcript claims and recalculate uncertainty from the recorded results.

The recalculation reproduces the reported ratio and interval: 0.975, with 95% CI 0.773-1.228. Averaging the two repeats within each task and using a t interval across seven task means gives 0.719-1.321. That supports “no evidence of a gain,” not evidence that a 15-20% gain is absent.

Replacing pca r2’s pick with its shortlist oracle raises the aggregate ratio only to 1.015. That sensitivity check supports fixing the scorer, but it doesn’t reveal a large team advantage.

**Fix the benchmarking environment and test whether agents can sustain useful search before buying larger teams or an Opus comparison.** My reading is that v6 found no demonstrated advantage for this elicited, flat Haiku team over equally elicited independent search. It did not establish that communication has no value.

I read all four reports, the results, prompt and selection code, the earlier transcript summaries, and the cited literature. I also spot-checked the pca and sinkhorn messages in the eval logs. My recommendations assume model dollars are the primary resource constraint, with cloud and scorer costs outside the $30-60 allowance.

**(a) What explains the result, ranked**

| Rank | Explanation | My assessment |
|---|---|---|
| 1 | **Teams seldom extended the best discovery.** Agents explored, then deferred rather than refining or combining. | The strongest explanation. Sharing a winning solver adds little when the independent arm’s selector already finds its best solver. Communication must improve discovery, correctness or subsequent refinement to beat that baseline. |
| 2 | **Useful search stopped early in both arms.** | A major limitation on what was tested. Remaining dollars mostly bought compliance with the spend gate. More budget alone is unlikely to help; the earlier $0.25 Haiku runs already exhibited this. |
| 3 | **Local experimentation differed between arms.** | A material confound. The sinkhorn transcript directly connects the larger visible CPU count to a 16-thread implementation and an “avoid OMP” broadcast. Remote final timing does not undo decisions made using misleading local experiments. |
| 4 | **Several tasks reward discovering a dominant technique, with limited subsequent refinement.** | This favours the strong best-of-N baseline. But “every group is decided by one trick” is too broad: a full HDBSCAN pipeline and a precision cast are substantially different discoveries. |
| 5 | **Scoring and validity anomalies distort individual comparisons.** | Important to repair, insufficient to explain away the aggregate null. Replacing pca r2’s team pick with its shortlist oracle changes team/diversified from 0.975 to only 1.015. |
| 6 | **The experiment is small.** | Explains uncertainty, rather than the point estimate. A meaningful positive effect remains compatible with the evidence. |

I would rank raw communication-token overhead below these. The reports do not show it consuming enough productive search to explain the result.

There is also a useful correction to the pca interpretation. The actual negative message tested **float32 eigh plus QR re-orthogonalisation**, explicitly saying float32 eigh without QR was unmeasured. A later message rejected float32 subspace iteration and broadly dropped float32. That is overgeneralisation from particular implementations, alongside timing uncertainty. CPU isolation will help, but evidence discipline matters too.

The literature supports these mechanisms without settling your hypothesis. *The Interaction Tax* principally examines different model families and full-solution exchange; it does not demonstrate that homogeneous flat peers should fail. Its diversity result is also sensitive to one task. Conversely, Park et al. provide evidence that communication can beat independent attempts under sufficient compute and useful feedback. [Interaction Tax](https://arxiv.org/html/2608.23541v1), [Scaling Discovery through Test-Time Communication](https://arxiv.org/abs/2609.21032).

**(b) Which fixes matter**

| Proposal | Priority | Recommendation |
|---|---|---|
| **1. CPU isolation** | Essential | Give each agent an enforced, disjoint two-CPU allocation, including spawned/background processes. Match memory allowances and default thread settings too. Check actual affinity and thread behaviour; `nproc = 2` alone is insufficient. |
| **5. Thread/reference behaviour** | Essential | Prefer reference and solver timing in separate, controlled processes. A two-sided diagnostic is useful, but automatically making a solver invalid below 0.87 converts a measurement problem into a correctness penalty. |
| **6. Wider validity** | Essential | Add a fresh, selection-only validity pass in the task image before choosing the winner. Stream instances rather than holding 1,000 in memory. Backfill the shortlist when candidates fail. Measure its operational cost before calling it cheap. |
| **4. Seed separation** | Essential, easy | Enforce disjoint **instance ranges**, accounting for offsets and `n`. Finalize starts at dev seed 50,000; the present 400,000 cap protects stage 2 but not finalize. |
| **3. Messaging and budget display** | Worth doing | Make `summary` optional and show four decimals plus percentage remaining. Delivery immediately after successful tool completion already exists. |
| **7. Operations** | Required hygiene | Preserve logs continuously, persist selection results incrementally, and make retries resume existing job IDs. An ambiguous POST retry must not silently launch another scoring job. |
| **2. Scheduling** | Useful, secondary | Reserve capacity for matched blocks and pin commits/images before launching. Randomise selection job order across arms. Current evidence does not make start-time skew a leading explanation. |
| **8. Task screen** | Revise substantially | Screen for measurement reliability. Do not automatically exclude bimodal or single-switch tasks. |

The delivery wrapper already drains pending messages after the blocking tool returns: [team.py:657](/Users/jack/projects/swarm-scaling/src/swarm_scaling/team.py:657). Reducing delays during `dev_eval` requires asynchronous evaluation or interruptible calls. I would defer that architectural change unless earlier messaging remains ineffective after the simpler repairs.

The current CPU allocation multiplies CPUs by team size and exposes one shared cpuset: [tasks.py:418](/Users/jack/projects/swarm-scaling/src/swarm_scaling/tasks.py:418). The reference guard is explicitly one-sided: [thread_guard.py:113](/Users/jack/projects/swarm-scaling/src/swarm_scaling/algotune_assets/thread_guard.py:113).

For validity, using the report’s measured failure probability of 1.245%:

| Validation instances | Probability of missing every failure |
|---|---:|
| 100 | 28.6% |
| 1,000 | 0.00036% |

That calculation assumes independent draws from the same distribution. At a 0.1% failure rate, 1,000 instances still miss every failure 36.8% of the time. Thus “passes 1,000” is stronger evidence, not a guarantee. The three hidden timings reuse the hidden instance set; they are not 300 independent validity trials.

I disagree with excluding least_squares simply because agents produced invalid shortcuts. Those are legitimate search failures. Exclude it if the **verifier or measurement cannot support trustworthy comparisons**, not because correctness is difficult. Likewise, a dominant trick can be exactly what communication should spread. Excluding such tasks changes the target population. Retain them as a declared stratum, with an all-task primary result and predeclared sensitivity analyses.

What is missing:

- **A calibration criterion for productive continuation.** Inspect whether additional budget produces implementation changes and discriminating experiments, rather than just more calls.
- **Controlled scorer-load validation.** Disjoint cpusets do not eliminate shared memory bandwidth or machine-wide effects.
- **Evidence provenance in messages.** Distinguish local timing, remote timing, correctness failures and untested conjectures.
- **Fresh hidden seeds for the follow-up.** The existing hidden cases have now informed diagnosis; freeze a new hidden set before further development.
- **Task-level inference.** The pre-registration specifies task resampling with nested repeats, while the published interval treats fourteen groups as independent: [PLAN.md:139](/Users/jack/projects/swarm-scaling/PLAN.md:139), [v6-main-results.md:22](/Users/jack/projects/swarm-scaling/analysis/v6-main-results.md:22).

**The one-solver outcome is fair for your stated question.** Every arm can preserve candidates, and the selector applies the same rule. Breadth matters insofar as it produces a better selected solver. A union-of-findings or coverage outcome would be a separate experiment. Also describe the treatment accurately: team membership, shared visibility, tools and collaboration guidance form a package; this is not an isolated causal estimate of messaging alone.

**(c) Prompt critique**

Add persistence guidance. It is legitimate elicitation if applied symmetrically and frozen before evaluation. Your draft is too long and contains instructions that could waste search:

| Wording | Problem | Alternative |
|---|---|---|
| “Implement it fully” | Forces expensive completion before rejecting weak ideas. | Build the smallest experiment that can fairly assess the idea; complete promising ones. |
| “A single disappointing measurement is not proof…” | Can encourage repeated testing until a favourable result appears. | Check measurement quality and use a controlled comparison. |
| “Not … repeating measurements of unchanged code” | Forbids useful correctness stress tests and uncertainty checks. | Require a stated purpose for repetition. |
| “Effort is not duplicated” | Reinforces the observed tendency to stop once someone claims territory. | Avoid unnecessary duplication; allow independent confirmation and competing approaches. |
| “Keep at least one alternative alive” | Can become ceremonial maintenance. | Preserve useful alternatives and explore ones with a plausible advantage. |

OpenAI’s reported near-20% SWE-bench improvement came from the **combined persistence, tool-use and planning reminders**, on GPT-4.1. It is not an estimate of persistence alone, nor evidence of a comparable gain on Haiku optimisation. [OpenAI GPT-4.1 prompting guide](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide).

Anthropic recommends specific unattended-run completion instructions. Its Haiku early-stopping guidance concerns long prompts at low effort, whereas this study uses xhigh. Both sources justify trying the intervention and evaluating it, not assuming it works. [Haiku guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5), [Opus guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5).

I would use this shared block:

> Your objective is to find the fastest solver that satisfies the task’s correctness requirements within the available budget. Keep a working baseline and useful alternatives saved. A working candidate is a baseline to improve: continue testing plausible improvements and materially different approaches while budget remains. Give promising ideas a fair test using the smallest implementation that can assess them, then complete and repair those that warrant further work. Use controlled comparisons and the remote evaluator for performance decisions. Repeat measurements when they answer a specific correctness or timing question. After a failed approach, choose and test the next plausible improvement rather than stopping or spending calls only to reach the spend threshold.

Team addition, identical for every member:

> Share your intended direction early, and share useful partial results and artifacts as they become available. State what you tested, where you measured it, and what remains uncertain; distinguish failure of one implementation from failure of an approach. A teammate’s strong result is a baseline to improve. You can refine it, combine useful parts with your work, or pursue a competing approach. Avoid unnecessary duplication, while allowing independent checks. You may divide work by approach or component, or balance refinement with exploration; choose whatever helps produce a better final solver.

This provides suggestions without prescribed roles.

Strong encouragement changes the intervention, but does not inherently bias the comparison. The bias risk comes from repeatedly tuning teams on evaluation outcomes while holding independent elicitation fixed. Use separate calibration tasks, comparable development effort for both arms, and then freeze everything. Call the result the effect of the **elicited team package**.

**(d) Recommended next experiment**

**Use the next $30-60 for a repaired Haiku experiment at N = 4. Defer Opus and N = 8.**

| Component | Design | Allocated model budget |
|---|---|---:|
| Calibration | Two existing pilot tasks; team of four versus four diversified independents; $0.25 per agent | $4 |
| Main | Twelve frozen tasks × two repeats × those two arms; $0.25 per agent | $48 |
| Reserve | Per-call overshoot and operational allowance | $8 |
| Total | | **$60** |

At roughly $30, run one main repeat per task: $24 main + $4 calibration, leaving $2 reserve. That is exploratory.

The $0.25 level is a **design choice, not a measured optimum**. It is affordable, previously exercised, and provides room for the revised continuation behaviour. Calibration should assess whether that room is used. If agents still pad, stop scaling the experiment and revise the continuation mechanism.

For this run:

- Keep xhigh, cache policy, initial priorities and the 80% gate fixed across arms. Keep the gate provisionally because removing it previously destroyed budget utilisation. Replace the purely numerical refusal with a symmetric reminder to choose a useful next experiment.
- Fix CPU isolation, seed separation, validity selection and scorer behaviour first.
- Keep the same two-stage selector, incorporating the validity screen identically in both arms. Picks must remain fixed before hidden scoring: [select_portfolio.py:162](/Users/jack/projects/swarm-scaling/scripts/select_portfolio.py:162).
- Sample twelve tasks across the frozen algorithmic families. Include dominant-switch and fragile-validity cases as declared strata. Label previously inspected tasks as replications.
- Pre-register the task-weighted mean log ratio, task-resampled interval, failure treatment and sensitivity analyses.
- Record first sharing, post-sharing implementation changes, useful continuation, candidate lineage and realised spend as mechanism diagnostics.

Do not pool this with v6 as though the harness and prompt were unchanged.

**Power is the limiting factor.** From the rounded result table, I calculated log-ratio SD = 0.401. Approximate two-sided paired-t planning at 5% significance gives:

| True team improvement | Independent paired blocks for 80% power | For 90% power |
|---|---:|---:|
| 15% | 67 | 89 |
| 20% | 40 | 53 |

These assume the observed variance persists and blocks are independent. For generalisation across tasks, averaging each task’s two repeats gives SD = 0.329. Using that estimate requires approximately **28 tasks with two repeats for 80% power at 20%**, or **46 tasks at 15%**. Seven tasks are insufficient to estimate those requirements precisely.

As a diagnostic, a t interval over the seven task means is **0.719-1.321**, compared with the reported fourteen-group interval of **0.773-1.228**. Neither establishes equivalence. The recommended twelve-task experiment has only about **42% power for a 20% effect** under that task-mean variance assumption. It can expose a large gain and test the mechanisms; it cannot reliably settle a modest gain.

| Follow-up axis | My recommendation |
|---|---|
| **Opus versus Haiku** | Scientifically worthwhile later. Compare each model’s team/independent contrast under the repaired harness. A model interaction requires both contrasts; “Opus significant, Haiku not significant” is insufficient. Equal dollars also do not isolate coordination capability from different trajectory lengths. |
| **N = 8** | Defer until useful sharing and refinement occur at N = 4. Eight increases the independent baseline’s discovery coverage too. |
| **80% gate** | Retain provisionally, with productive continuation verified in calibration. Do not describe enforced expenditure as enforced search. |

Vals supplies no strong reason to expect Opus coordination to rescue this setup: its Opus gains were not significant despite substantially higher team costs, and it used hierarchical teams versus a single agent, not matched best-of-N controls. [Vals study](https://www.vals.ai/blogs/multi-agent-vibe-code-bench).

**Separate implementation follow-up:** diversified N = 8 currently fails the four-priority index check in [swarm.py:214](/Users/jack/projects/swarm-scaling/src/swarm_scaling/swarm.py:214). An eight-agent study needs an explicit symmetric priority policy before launch. I would repeat the four priorities twice in both arms rather than invent four new roles.
