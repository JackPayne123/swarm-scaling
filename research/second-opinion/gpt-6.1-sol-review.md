I’ll check the chart data and benchmark papers first, then assess the design and cut it down to a defensible one-week study. I’ll use the second-opinion and documentation skills, and leave the repo unchanged.

Two findings change the framing already. Ord explicitly treats swarms as a speed-versus-cost tradeoff, so “single agents win at matched tokens” may confirm his argument rather than challenge it. Park’s paper also reports token-efficiency curves, not just action-budget comparisons. I’m checking the chart arithmetic and the mixed-model literature next.

The chart check confirms the narrow claim: single-agent interpolation beats every swarm point within the observed single-agent token range. But BrowseComp’s closest gap is only 0.22 percentage points, and aggregate chart data cannot establish statistical significance.

I also found a September 2026 paper testing mixed-model teams against matched-budget sampling. It reports a negative result on debate tasks. That weakens the novelty claim, while leaving room for a study of teams that share verifiable code improvements.

I would make the headline question: “Does sharing verified improvements beat independent search at the same API budget, and does mixing model families change that advantage?”

For one week, I’d use one task family and one team size. The original plan asks the same small study to estimate a scaling law, compare model families, diagnose coordination, and measure creative optimisation. That would spread the evidence too thinly.

The main design issue is the selector. For AlgoTune, choosing the fastest correct candidate on development inputs is a legitimate baseline and should be available to every arm. Choosing it using held-out test scores would be oracle selection. Voting would make the independent arm unnecessarily weak.

I’d also remove the designated lead and give each worker its own writable files. That avoids confounding a mixed-model result with the lead’s identity or with agents overwriting each other’s code.

The Anthropic figure adds another qualification: the async-subagent curve is above the single-agent curve at low token budgets, before the single agent overtakes it. So “every public compute-matched comparison favours the single agent” is too strong even for these lab plots. The post should name the budget ranges where the single agent leads.

**Narrow this to one task family, two models, and one team size. Make the question: “At a fixed API budget, does sharing verified improvements outperform independent search, and does mixing model families change that advantage?”**

That is a defensible empirical follow-up to Ord. Estimating a general swarm-scaling exponent, testing heterogeneity, comparing two benchmarks, and diagnosing coordination is too much for one week.

I read the supplied chart JSON, card excerpts and Inspect docs, inspected the Anthropic figures, and checked the benchmark and relevant literature. I have not run benchmark agents, so affordability and throughput still need a pilot.

Several premises need correcting before you write the post.

| Premise | What I verified | Consequence |
|---|---|---|
| Ord eyeballed the charts | His current post explicitly reports regressions and 90% CIs | Reanalysis is a replication and sensitivity check, not a replacement for his eyeballing |
| Single agents win at every overlapping OpenAI token budget | True under both linear-token and log-token interpolation of the supplied JSON | Preserve the interpolation qualification |
| Every public compute-matched comparison favours single agents | Too strong. Anthropic’s async curve leads at low token budgets; Park reports favourable token-efficiency comparisons | Drop this headline |
| Park counts actions rather than tokens | Actions are one budget measure, but the paper also reports cumulative output-token curves | Incorporate those results into the literature comparison |
| Cross-vendor teams are an unpublished independent angle | Heterogeneous teams have already been studied, including a September 2026 paper with sampling controls | Novelty must come from the task, protocol and controls |

Ord already makes the token-efficiency and latency argument. Your contribution should test a proposed exception to it. His [current post](https://www.lesswrong.com/posts/6cb7qd3RSkgnviCpf/swarm-scaling) also reports the regressions and CIs.

My independent calculation from the [supplied JSON](/private/tmp/claude-501/-Users-jack-projects-pa/649ef890-38c9-4edc-a65b-1abdc50aca03/scratchpad/oai/multiagent_charts.json:1) gives:

| Benchmark | Swarm points inside the observed solo token range | Swarm deficit against linear interpolation |
|---|---:|---:|
| BrowseComp | 2 | 0.22–3.85 percentage points |
| SEC-Bench Pro | 5 | 8.76–23.33 points |
| Terminal-Bench 2.1 | 1 | 6.25 points |

The SEC example checks out: 16 agents score **45.63% at 68,568 output tokens**, versus **68.96%** from linear interpolation or **69.25%** from log-token interpolation.

That is eight overlapping comparisons. Many swarm points lie beyond the observed solo range. The 0.22-point BrowseComp difference also cannot establish a statistically meaningful disadvantage without task-level results.

A simple regression of score on log total tokens and log agent count produces λ values of **0.650, 0.565 and 0.481**. Using logit score gives **0.676, 0.572 and 0.494**. These broadly reproduce Ord’s estimates; they do not reveal a major correction.

The Anthropic evidence needs more careful wording. In [Figure 8.12.1.B](/private/tmp/claude-501/-Users-jack-projects-pa/649ef890-38c9-4edc-a65b-1abdc50aca03/scratchpad/fig191-191.png), async subagents outperform the solo curve at low token budgets before the solo curve overtakes them. Also, their token measure counts tokens once per context window, and their latency is reconstructed from reference serving rates plus tool time. Those are different quantities from cumulative billed tokens and observed API wall-clock time. See [opus55.txt:6565](/private/tmp/claude-501/-Users-jack-projects-pa/649ef890-38c9-4edc-a65b-1abdc50aca03/scratchpad/opus55.txt:6565).

Similarly, comparing 10 agents at two hours with one agent at 24 hours is approximately **20 versus 24 allocated agent-hours**. It does not match actual inference work or active agent time.

**The interesting question is whether useful intermediate discoveries change the economics of collaboration.**

Park provides motivation, but not a guarantee that cheap, short runs will show the effect. Its protocol includes distinct approaches, evidence-backed adoption and preserved variation after adoption. It is more specified than “identical agents sharing a directory.” Its token curves also show an initial coordination cost before gains emerge. [Scaling Discovery through Test-Time Communication](https://arxiv.org/html/2609.21032v1).

Your study could test whether that mechanism works at an independent researcher’s budget. That is useful even if the answer is negative. It would measure these models, this protocol and this budget range. It would not estimate frontier-lab swarm efficiency or directly update an intelligence-explosion parameter.

For compute matching, choose one primary resource and name it precisely.

| Resource | Recommended use |
|---|---|
| API dollars | Primary matching variable for mixed-model comparisons |
| Generated tokens, including reasoning | Secondary accounting; useful for homogeneous scaling |
| Billed input, output and cache tokens | Required cost ledger |
| CPU time, verifier calls and memory | Separate tool-resource accounting |
| Observed wall-clock time | Secondary operational outcome |
| FLOPs | Do not claim to match them across proprietary models |

Equal prices would not establish equal compute. Your cheap pair also has different input prices, and the Artificial Analysis lookup shows different cache rates. More fundamentally, tokenizers, reasoning behaviour, context handling and hidden inference architectures differ.

Match **allocated budget**, report **realized expenditure**, and plot outcomes against expenditure. Do not condition the experiment solely on realized token usage: an agent’s stopping behaviour is itself part of the treatment.

Inspect is suitable infrastructure, but its limits require deliberate configuration. It distinguishes total-token limits from output-only limits and supports scoped limits and model-specific cost accounting. Verify that per-worker and sample-level limits behave correctly during concurrent execution. [Inspect limits documentation](https://inspect.aisi.org.uk/setting-limits.html).

The first methodological attacks will be:

| Attack | Concrete defence |
|---|---|
| “Your ensemble baseline is deliberately weak” | Select candidates using the available development verifier |
| “Communication also changed files, prompts and resource access” | Define the treatment as the complete collaboration protocol; keep other resources equivalent |
| “The mixed team won because the better model was lead” | Use flat peers and a deterministic final selector |
| “Concurrent timing corrupted optimisation feedback” | Isolate or serialize development timing, not just final scoring |
| “You selected tasks where collaboration looked promising” | Separate pilot tasks from evaluation tasks; freeze selection rules |
| “Your CI counts correlated tests or recycled samples as independent” | Resample tasks, with repetitions nested within tasks |
| “The custom harness under-elicited collaboration” | Validate delivery, adoption and solo competence before evaluation; limit the conclusion to that harness |
| “You tuned communication after seeing evaluation results” | Freeze prompts and protocol before the main run |

**Use verifier-based selection for the independent arm.** For AlgoTune, choosing the fastest correct candidate on development inputs is legitimate. Choosing using held-out test scores is oracle selection. Apply the same final selection rule to communicating teams and solos.

Voting is poorly suited to executable optimisation. An LLM selector would introduce another capability and budget confound.

Resampling independent teams from a solo pool is acceptable, but the pool must contain enough independently generated trajectories. Thousands of resampled combinations do not become thousands of experimental observations. Account for pool uncertainty, and do not reuse the same trajectory twice within a simulated team.

The communication-minus-independent contrast estimates the **net effect of your collaboration protocol**. It does not, by itself, identify how much benefit came from messaging, code exchange, avoiding duplicated work or earlier access to promising candidates.

I would choose tasks as follows:

| Task family | Assessment | Decision for this week |
|---|---|---|
| **AlgoTune** | Development correctness and timing feedback support verified progress sharing. The original study used $1/task budgets | Best practical choice |
| **Frontier-CS algorithmic track** | Open-ended, continuous-scored optimisation; closer to Park’s research framing | Strong alternative if setup and cheap-model headroom pass the pilot |
| **ProgramBench** | Useful reconstruction benchmark, but hidden final scoring supplies weaker progress feedback and setup is heavier | Cut from the main experiment |
| ARC-AGI-3 | Relevant to communication, but adds interaction, visual representation and action-budget complications | Cut |

AlgoTune’s published protocol already selects the best development candidate. Its official aggregate is the **harmonic mean of speedups**, with invalid or slower solutions assigned 1×. Retain that metric and separately report correctness, invalid submissions and tasks with no improvement. Its historical $1 results establish that short-budget optimisation is possible, not that your proposed models will collaborate effectively. [AlgoTune paper](https://arxiv.org/html/2507.15887v3).

Serial final scoring alone is insufficient. Noisy development timing can cause agents to discard good ideas or adopt bad ones. Control cores, threading, warmup, compilation and candidate/reference comparisons during search too.

Frontier-CS is the strongest public alternative I found. Its current repository provides continuous scoring, locally available algorithmic tests and agent infrastructure. Use CPU algorithmic tasks, and keep evaluation inputs outside agent-visible files. Public tests are not automatically uncontaminated held-out tests. [Frontier-CS repository](https://github.com/FrontierCS/Frontier-CS).

ProgramBench’s 0.92 Sonnet score does not prove a ceiling for your models and budgets. More importantly, its official tests remain unavailable to the task worker: a continuous final score is not the same as reliable intermediate feedback. Its provided images also require Linux x86-64; that adds provisioning work on Jack’s Mac. [ProgramBench paper](https://arxiv.org/html/2605.03546v1), [usage guide](https://github.com/facebookresearch/programbench/blob/main/docs/README.md).

**The mixed-model hypothesis is worth testing, but I would not expect diversity alone to explain most swarm gains.**

Different families may have complementary strengths. They may also communicate less effectively, dilute stronger proposals or simply provide a better independent portfolio. Similar Artificial Analysis scores do not establish similar performance on these tasks.

Two relevant precedents:

- **Self-MoA** found that aggregating samples from one strong model often beat heterogeneous mixtures. It explicitly studies the quality–diversity tradeoff. [Rethinking Mixture-of-Agents](https://arxiv.org/abs/2502.00674).
- **Beyond Symmetric Agents**, submitted September 2026, compares heterogeneous debate teams with independent samples from the same rosters and reports that the sampling controls win. Its matching is principally generation-budget matching, rather than equal total billed tokens, and its tasks differ from iterative code optimisation. It weakens the general novelty claim without answering your specific question. [Paper](https://arxiv.org/html/2609.35875v1).

There are positive results too: another 2026 study reports debate and MoA advantages under a modeled inference-resource budget. The literature does not support a universal “matched compute always favours solo” conclusion. [Pareto-Optimal Test-Time Scaling](https://arxiv.org/html/2605.01566v1).

“Labs cannot publish cross-vendor results” is not a sound premise. Their incentives may favour native systems, but cross-vendor studies already exist. The independent angle is a transparent, controlled experiment they have not answered.

For an interpretable mixed-model experiment, use these eight configurations at total API budget \(B\):

| Configuration | Workers | Allocation |
|---|---|---|
| Solo A | A | \(B\) |
| Solo B | B | \(B\) |
| Independent AA | A + A | \(B/2\) each |
| Independent BB | B + B | \(B/2\) each |
| Independent AB | A + B | \(B/2\) each |
| Communicating AA | A + A | \(B/2\) each |
| Communicating BB | B + B | \(B/2\) each |
| Communicating AB | A + B | \(B/2\) each |

Use private writable workspaces. Communicating workers can publish immutable candidates, development scores and findings, and inspect their partner’s publications. Independent workers cannot inspect them during search. All arms use the same deterministic final selector.

Predeclare three contrasts:

1. **Communication benefit:** \(C_{AB}-I_{AB}\).
2. **Portfolio benefit:** compare \(I_{AB}\) separately with \(I_{AA}\) and \(I_{BB}\).
3. **Heterogeneity–communication interaction:**
   \[
   (C_{AB}-I_{AB})-\tfrac12[(C_{AA}-I_{AA})+(C_{BB}-I_{BB})].
   \]

A mixed communicating team beating homogeneous communicating teams does not establish that heterogeneity improved communication. The mixed independent team might already enjoy the same advantage.

Compare against both homogeneous alternatives explicitly. Do not construct a per-task “best homogeneous model” comparator using held-out results unless you label it as an oracle bound.

Cut “collapse rate” as a headline metric. No visible reuse is better called **no observed cross-agent adoption**. Record artifact lineage, reproductions and adoption events; manually inspect a few examples. Message counts and self-reported usefulness are weak proxies.

Also cut the LLM classification of “algorithmic versus surface-level” improvements. Manual analysis of a few important solutions will be more credible than another uncertain aggregate.

Keep λ in the public-data appendix. Aggregate chart coordinates cannot provide task-level uncertainty: bootstrapping those coordinates measures a different uncertainty. A slope ratio also assumes a shared response shape and becomes unstable near saturation. If you later estimate λ experimentally, use overlapping budgets and explicitly state the fitted model. A two-worker study should not advertise a scaling law.

I would structure the post around the experiment:

1. Ord’s result and the proposed verified-progress exception.
2. The exact budget, protocol, selector and preregistered comparisons.
3. Results with every task visible, including failures.
4. A few demonstrated adoption or harmful-convergence examples.
5. Limits on generalisation.
6. Appendix: public-chart replication and sensitivity checks.

A suitable working title is **“Do mixed-model teams beat independent search at the same cost?”** Let the findings determine the final title.

The revised week plan:

| Day | Work and stopping condition |
|---|---|
| **1** | Finish the bounded chart replication. Choose AlgoTune and provision one reproducible Linux environment |
| **2** | Pilot three development tasks on both models, solos, independent pairs and communicating pairs. Verify scoring, message delivery, adoption, accounting and controlled timing |
| **3** | Freeze prompts, model settings, task-selection rule and analysis. Choose budgets from measured cost and progress; start evaluation |
| **4** | Finish the frozen evaluation and rerun infrastructure failures under a predetermined rule |
| **5** | Paired analysis, task-level uncertainty and manual checks of the largest gains |
| **6** | Write the post and prepare prompts, logs, artifacts and exclusions for review |
| **7** | Check claims against artifacts and revise |

A concrete initial evaluation is **12 held-out tasks × eight configurations × two budgets × two repetitions = 384 system runs**. Those numbers define scope, not adequate statistical power. If the pilot cannot support that workload, retain all eight controls and use one budget: **192 runs**. Prefer additional tasks over an \(N=8\) extension.

The API allocation ceiling for the two-budget design is:

\[
192(B_{\text{low}}+B_{\text{high}})
\]

plus pilots, permitted reruns and infrastructure. Choose the budgets after measurement. If the pilot shows no credible progress or cannot produce trustworthy timing by the end of day two, publish the chart replication and pilot findings, and defer the heterogeneity claim.

Cut ProgramBench, the second model tier, hosted native-agent comparison, messaging ablation and full \(N=1,2,4,8\) grid. **One week is realistic for this bounded study; the original plan is not.**

Separate follow-up: the supplied JSON’s `latency_s` values appear to be expressed in minutes, matching labels such as “0.88 minutes.” Verify that field’s units before latency analysis. I did not modify it or any repo files.
