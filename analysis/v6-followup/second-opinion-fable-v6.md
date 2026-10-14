# Second opinion on v6 (Fable 5.1, claude-fable-5-1), 2026-10-13

Read: the brief, the four reader reports, `analysis/v6-main-results.md`, PLAN.md (v6 section and decisions log), `swarm.py` prompt code, `select_portfolio.py`, `prices.py`, `research/PROTOCOL.md`, `research/EVIDENCE.md`, the pilot 5 and haiku2 result and transcript notes, the v6 pilot portfolio records, and the Sol pivot review. Looked up: OpenAI's GPT-4.1 prompting guide, Anthropic's current prompting best practices and the Haiku 5.5 page, the Interaction Tax paper, Park et al., Tran and Kiela (arXiv 2604.02460), ExComm (arXiv 2605.22102), Anthropic's multi-agent research write-up, and the LessWrong swarm research-priorities post. No files changed, no cloud, no API spend. The power numbers below were computed from the 14 paired groups in the results table.

## Bottom line

The v6 result is not a null; it is an underpowered estimate centred slightly below 1 on every measure (pick 0.97, oracle 1.01, stage 2 1.01, best dev 0.91). The pre-registered statement, "no evidence of a 20%+ gain", is exactly what the data supports and is what the post should say. What explains it is mostly structural and behavioural, not the harness: on these tasks the score is set by one or two tricks, the final artifact is one solver, so a team of 4 is a best-of-4 over what its members publish plus a few percent of transfer, and Haiku's members stop working once a lead appears and then pad to the submit gate. The harness confounds are real but second order and mostly fixable for free.

The three decisions I would make differently from the brief:

1. Remove the submit tool and the 80% gate altogether. Run every agent to budget exhaustion, as AlgoTune's own protocol does. The gate created the stop-then-pad phase (40% of spend in both arms) and the finish notices that tell teammates stopping is normal. No prompt text will fix a structure that rewards waiting.
2. Do not spend this round on a model contrast or on N = 8. $30-60 cannot answer either. It can fund a properly powered Haiku rerun of the one comparison that matters (diversified team vs diversified independents) at about 60 groups, which is the first time this project would have the power to see a 15-20% effect.
3. Shorten the prompt draft by half and drop the organisational menu. The persistence evidence the brief cites does not transfer (details in section c), and the one line in the team block that targets the measured failure, "do not stop when a teammate leads", is the part worth keeping.

## (a) What explains the result, ranked

1. **Task structure plus a single-solver output makes a team a best-of-4 with a small transfer bonus.** Every reader found the same thing: each group turned on whether some agent in the arm found the decisive technique in a fast, valid form (float32 sgeev, full numba HDBSCAN, prange LM, int16 Floyd-Warshall, f32 shift-invert). Under a "fastest valid published candidate" selector, a team can beat best-of-N only through (i) propagating a found trick into a better-engineered version (observed twice, +6-7%), (ii) combining parts from different members (observed twice, +2-6%; never across method families), or (iii) parallel refinement of one lineage after the trick is found (not observed, because members stopped). The harness allows all three. The agents used almost none of them. Expected gain under this structure is well below the 20% the claim needs, and that would hold for a stronger model unless it combines or refines more than Haiku did. Pilot 5 on Opus showed the same convergence pattern and the same best-of-N loss on generalized_eigenvalues_real, so I would not treat the model-capability hypothesis as supported by anything in hand.

2. **Stop-then-pad, caused by the gate.** Agents in both arms published their last candidate at about half their budget, then spent 40% of spend on refused submits, check_budget polls, repeat dev_evals and wait_for_message. The marginal dollar bought nothing, so "matched budget" was really "matched to the first stop". This is why a larger budget cannot help (haiku2 at $0.25 showed the same), and it removes the one phase where a team could have shown an advantage, refinement of the lead's lineage with four agents' turns. Anthropic's own Haiku 5.5 guidance describes early stopping as a low-effort, long-system-prompt phenomenon where the model "hands the task back"; these agents run at xhigh and do not hand back, they declare done and wait. The observed behaviour is a response to a goal ("submit") blocked by a rule (80%), not a persistence failure. Change the structure, not the wording.

3. **Deference after a lead.** 13 of 24 agents on three tasks announced they would stop or defer once a teammate led, at 0.21-0.65 of budget; 18 recipients published 15 more candidates between them; the team pool was 60 distinct candidates against the independents' 70 on the four group-A tasks. Communication shrank the pool. This is the mechanism by which the channel is net negative: it converts a best-of-4 into a best-of-fewer. It is consistent with Park's warning that teams lose when agents cannot or do not verify before adopting, and with the Interaction Tax finding that full-solution exposure makes agents stay near the first good solution they see. Note the agents mostly did not copy the leader's code; they stopped. That is worse than copying.

4. **Late and low-bandwidth channel use.** In four of eight group-A teams the first message came after mean spend of 0.45, or never (lsq r2). By then every agent had committed to its lineage. Messages can only redirect work not yet done, and with about 24 productive calls per agent there was little left. The shared registry did its job silently (clust r2's win came from a registry read before any message). This is a protocol-timing issue the prompt alone will not fix; see the auto-notice fix below.

5. **Wrong negatives from the shared box.** Three broadcast negatives, two from noisy local timing on the contended 8-core box, one of which (pca r1 "float32 eigh gave NO gain") pruned the branch the independents won with. The sinkhorn thread-count effect (team agents tuned for 8-16 threads from `os.cpu_count()`, 1.2-1.4x, against 6.8-7.6x for independents on 2 cores) is the same confound. This is the one harness asymmetry that demonstrably flipped groups. It is team-arm only, so it biases against teams, and it is cheap to remove.

6. **Noise.** sd of the paired log ratio is 0.40 over 14 groups, so the 95% interval spans 0.77-1.23 and a true 15% effect would be invisible. Two of the 14 groups were decided by measurement accidents (lsq r1 loose invalid pick, pca r2 team timing mode), and those belong to item 5's family, not to the team hypothesis.

Not an explanation, despite the brief's prior: "Haiku lacks coordination as a capability". Haiku's messages were competent findings reports with numbers and territory claims, deliveries worked, and the two wrong negatives were corrected by teammates re-measuring. What Haiku lacks is the propensity to keep working after a lead and to combine. Opus in pilot 5 showed the same propensity at one run per cell. Treat the model question as open, not as the likely answer.

Is the comparison unfair to teams? No, for the question Ord asks. The score is capability on one task, and breadth only counts if it reaches the final artifact. The team's structural costs (about 2% for tool definitions and messages, plus whatever attention goes to re-verifying teammates, 47% of one agent's spend in geneigc r1) are part of the treatment and should stay. One measurement asymmetry worth reporting: stage 1 shortlists the top 3 distinct sources, and a converged team's shortlist is three near-copies (sparseig r2, spd r2), so stage 2 protects a team less against a noisy rank-1. Keep the rule identical across arms, report it.

## (b) The proposed fixes, ranked, plus what is missing

Do these, in this order of payoff:

1. **Remove submit; run to budget (missing from the brief, biggest lever).** No gate, no refused submits, no finish notices to pad against. The run ends when the budget is used; the best published candidate is taken. This is AlgoTune's protocol ("continuously queries the LM ... until the budget runs out"). Risk: text-only idle turns (haiku2 saw one agent burn $0.21 that way). Mitigate with the react continue prompt saying the run continues until the budget is used and asking for the next concrete improvement, and with the persistence text below. Symmetric across arms. Measure "spend after last publish" as the check that it worked; v6's 40% is the baseline to beat.

2. **CPU isolation (brief fix 1), implemented as one container per team agent.** A per-agent cpuset inside a shared container needs every bash call wrapped in `taskset`, and a solver can call `os.sched_setaffinity` or read `os.cpu_count()` (sinkhorn r1 agent_2 did exactly this and chose 16 threads). Give each team agent its own container on the team VM with the registry and inbox directories on a shared volume, so `nproc`, `os.cpu_count()` and BLAS defaults match the independents. This removes items 5 above and most of the 56 bash timeouts.

3. **Non-blocking dev_eval (missing).** Most messages (53 of 85 in group B) waited behind a blocking dev_eval, median 163 s. Return a job id and let the agent poll or be told in the next tool result. Symmetric, and it also lets independents work during the queue wait. The brief's "deliver held messages as soon as dev_eval returns" already happens; the delay is the blocking call itself.

4. **Validity pass on about 1,000 instances at stage 2 (brief fix 6).** Yes. Untimed, cheap, and it turns the lsq rows from a lottery (29% miss rate for a 1.2% failure solver) into a measurement. Also treat a hidden-invalid pick as a selection failure to report, not as a 1.0 that enters the paired ratio; the 1,000-instance pass should make this moot.

5. **Score against the reference measured alone (replaces brief fix 5).** A two-sided flag at 0.87 would invalidate legitimate numpy-LAPACK solvers (14 pca sources sat at 0.66-0.86). The pca problem is that the denominator depends on the thread state the solver leaves behind. The AloneTimer child already measures the reference in a clean process per instance; make speedup = alone reference / solver the study's primary score and keep AlgoTune's interleaved number as a secondary column for comparability. The 1.15 guard stays as the anti-exploit check. This fixes pca r2 (13.08 stage 2 against 7.25 hidden) and the r2 loose geneigc `threadpool_limits(1)` side effect in one change.

6. **Seed range cap (brief fix 4), SendMessage summary optional, check_budget to 4 decimals (brief fix 3).** Trivial, do them. The 4-decimal display matters less once the gate is gone, because the polling loop was gate-driven.

7. **Start teams with their independents (brief fix 2).** Do it because it is cheap (launch the team VM first), but the audit's hour-matched waits show it changed little. Low priority.

8. **Operations (brief fix 7).** Copy scorer logs before teardown, retry selection network calls, timestamp-plus-pid output names (already done). Yes.

Add one protocol change that I think is the most important team-side fix:

9. **Auto-notices from the registry.** When any team agent publishes, every teammate gets a one-line system notice in its next tool result: agent, dev score, note, path. The channel then opens at the first publish (f = 0.25-0.36 in v6) instead of when an agent decides to write a message (f = 0.45 or never). It is exactly the low-bandwidth signal the Interaction Tax paper recommends (scores and method descriptions, not full solutions), it costs the team a few tokens per publish, and it does not require the agent to choose to communicate. Independents cannot have it by definition; that is the treatment. Log it separately so adoption chains can be traced.

Disagree with:

- **Brief fix 8, dropping "single-switch" tasks.** Dropping geneigc and lsq after seeing that they moved the ratio is outcome-dependent exclusion, whichever way it moves. A forward screen is legitimate only if written before the next run and applied by a pilot of independents, and it narrows the claim to "tasks where collaboration could plausibly matter", which the post must then say. My preference: keep the 7 tasks (known behaviour, pre-registered), add 3 held-out tasks from the three families the draw did not reach (dense symmetric eigenproblems, convex programs through a general solver, sort-based projection), and handle lsq-type instability with the validity fix rather than exclusion. Report with and without geneigc as a sensitivity column, because its bimodality is a property of the task and the pre-registration tried to keep it out.

- **Larger Haiku budget.** Median prompt was 57k tokens at $0.10. Haiku's long-prompt tier starts at 100k and multiplies input and output prices by 5. A $0.25-0.50 Haiku agent crosses it and then pays 5x per call for the rest of the run, which is why haiku2 at $0.25 bought padding. Without compaction, more Haiku dollars do not buy more turns. Keep $0.10.

- **N = 8 now.** haiku2 showed the team deficit against best-of-N growing from team of 2 (about 0) to team of 4 (-21%), and v6 found the deference mechanism. A team of 8 under the same behaviour is a best-of-fewer with more messages, and it doubles the group cost. Run it only after the no-stop protocol has been shown to remove deference at N = 4.

## (c) The prompt draft

Two general points first.

**The persistence evidence does not transfer.** OpenAI's "close to 20%" is from the GPT-4.1 guide, a non-reasoning model whose failure was ending its turn and handing control to the user; the reminders fix a turn-boundary behaviour. Anthropic's Haiku 5.5 page says the same symptom appears "with a long coding-agent system prompt at low effort" and that raising effort halves it. v6 agents run at xhigh, have no user to hand back to, and did not stop; they declared done and waited for a gate. Wording-only persistence has weak support in exactly the literature PROTOCOL.md already summarises (Cho et al. 2025, Chen et al. 2026: tone and "be stubborn" moved nothing; structure moved diversity). Expect the all-arms block to add little on its own, and do not let it substitute for removing the gate. Measure its effect directly: publishes after the first "done" statement, and spend after last publish.

**Symmetry.** The all-arms block is symmetric in text and probably not in effect, which is fine as long as it is reported. It will lengthen the productive phase for independents too, and an independent has one lineage to refine, so it may raise the independent arm more than the team. That is the honest comparison and the pre-registered claim survives it. The real asymmetry to remove is the finish notice ("agent_k has finished and will not read further messages"), which only team agents receive and which cues stopping; removing submit removes it. The brief's worry that "strong encouragement" biases the comparison is only a worry if the encouragement differs between arms; the team block is the treatment and is meant to differ.

Line by line on the draft:

- "nobody will step in to help or end the run early" - keep, it is true and removes the hand-back frame.
- "treat it as a baseline to beat, not a finish line" - keep; this is the one sentence aimed at the measured failure in both arms.
- "Give each promising idea enough time ... implement it fully, measure it, and fix the obvious problems before you decide it does not work" - keep but shorten; "measure it with dev_eval" is more useful than "measure it", because the wrong negatives came from local timings.
- "A single disappointing measurement is not proof ... check whether the measurement was sound" - keep, it targets wrong negatives in both arms.
- "When you are unsure how to proceed, choose the most reasonable approach and test it rather than stopping" - cut; nothing in the transcripts shows indecision as a failure mode.
- "not on waiting or on repeating measurements of unchanged code" - keep; it names both arms' padding behaviours.
- Team: "the team's result is the best solution any of you produces" - keep and sharpen. The agents behaved as if the goal were "the team has a good solver"; the selector's actual objective is "the fastest valid candidate in the pool", so four different fast candidates beat four copies of one. Say that, it is the direct counter to deference.
- "tell teammates early what you are trying and what you have measured, including partial results and dead ends" - keep "early" and "measured"; qualify "dead ends" with "after you have confirmed them with dev_eval", because unconfirmed dead ends were the three wrong negatives.
- "do not stop your own line of work; check whether your approach can beat it or be combined with it, and keep at least one alternative alive" - the best sentence in the draft. Add "re-measure it yourself first", which is how both corrected negatives happened.
- "Useful ways to organise include agreeing who explores which direction, splitting ..., or having some members refine ... choose whatever suits the task" - cut. Priorities were mentioned once across 32 agents, pilot 1's split proposal was ignored, the task list was opened twice and never used. Prose menus of organisational options are not followed by these models, and the sentence lengthens a prompt that Haiku's own guidance says should stay short.

A tightened version, both blocks together about half the draft's length:

All arms:

> You work autonomously on a fixed budget. Your run ends when the budget is used up; nobody will intervene or end it earlier. Treat every working candidate as a baseline to beat, not a finish. Before you decide an idea does not work, implement it fully, measure it with dev_eval, and check that the measurement was sound; one local timing is not evidence. Spend every remaining dollar on attempts to improve your best candidate or on approaches you have not tried, never on waiting or on re-measuring unchanged code.

Team only (replaces the sharing line):

> The team's result is the single fastest valid candidate anyone publishes, so four different fast candidates are worth more than four copies of one. Publish early and tell teammates what you measured and how, so they can use it while they still have budget. When a teammate reports a strong result, do not stop: re-measure it yourself, then beat it, combine it with your own work, or push a different approach. Share a negative finding only after dev_eval confirms it.

Keep EXPLORATION_BLOCK and the priority line as they are (the diversified-independent control depends on them being identical), and record the prompt variant in run metadata as now.

## (d) The next experiment within $30-60

**Power.** From the 14 v6 groups, sd(log team/div) = 0.40. For 80% power at two-sided 5%: about 38 groups to detect +20%, 64 for +15%, 18 for +30%. Dropping geneigc and lsq does not reduce the sd (0.43 on 10 groups); dropping pca as well gives 0.22 on 8 groups, which says the measurement accidents (lsq validity, pca timing modes) are a large share of the noise and the fixes above should pull the sd toward 0.3, where 40 groups detect +15%. Caveat: the unit of resampling is the task, repeats within a task are correlated, and v6 cannot separate between-task from within-task variance with two repeats. Prefer more tasks over more repeats where the budget allows, and bootstrap over tasks as PLAN.md says.

**Cost.** One group of (diversified team of 4 + 4 diversified independents) is 8 agents at $0.082 measured, $0.66 in model spend. 60 groups is $40. Drop the loose arm; it has answered its question (diversification helps, mostly via lsq) and costs a third of every group.

**Recommended run (v7).** Haiku 5.5, xhigh, $0.10 per agent, no submit tool, run to budget, fixes 2-6 and 9 above, the tightened prompt in both arms (team block in the team only). 10 tasks (the v6 seven plus one each from the three undrawn families, chosen by the same seeded rule) x 6 repeats = 60 groups, about $40 model spend, selection and VM costs extra (not estimated here; v6's were not in the brief). Pre-register the same claim ("20%+ gain or no evidence"), the primary score (alone-reference speedup), the sensitivity columns (interleaved score; without geneigc), and three behavioural checks that the fixes worked: spend after last publish under 15%, first registry notice before f = 0.4 in every team, and distinct valid candidates per team at least equal to the independents'. If any of the three fails, the result is a harness result, not a team result, and the post should say so.

Why this and not the alternatives:

- **Model contrast.** Opus 5.5 at the pilot-5 budget is $5 x 8 = $40 per group; $60 buys 1.5 groups. Even if Opus's paired sd were 0.15 (pilot 5 independents were tighter than Haiku's, 25.1-33.6 on dst, but that is one task and six runs), 20% power needs about 12 groups, $480. Sonnet 5.5 at $1.50 per agent is $12 per group, 20 groups for $240 at an assumed sd of 0.2. Neither fits. State the price and defer; do not run two or three Opus groups and read anything from them.
- **N = 8.** Doubles the cost per group and tests the hypothesis in the configuration where the measured deference mechanism bites hardest. After v7, if the no-stop protocol removes deference, N = 8 at the same per-agent budget is the right second step and costs about $1.30 per group.
- **A blind-first (two-phase) arm.** The v6 pilot's two-phase team scored 2.41 on geneig against the diversified team's 2.14 and lost on optimal_advertising. With auto-notices opening the channel at first publish, the two-phase design is less needed; keep it in reserve.

If the budget must stay at $30, run the same design at 7 tasks x 6 repeats = 42 groups ($28), which still clears the 20% detection threshold.

## Follow-ups outside the question (not fixed, not folded in)

- The v6 pilot portfolio record for optimal_advertising shows every arm at 1.0 (`analysis/portfolio/20261011T223022.jsonl`); either all picks were invalid or the record predates a fix. Check before citing pilot numbers.
- Stage 1 ranks distinct sources, so a team that converges fills its shortlist with one lineage and loses the stage-2 protection other portfolios get (sparseig r2, spd r2). Not a bug, but report it alongside the oracle bound.
- `analysis/v6-main-results.md`'s t interval treats repeats as independent; the post should show the task-level bootstrap PLAN.md pre-registered, which will be wider.
- The main scorer's logs were never copied and four exit-137 kills are unexplained; the audit's guess (container memory limit) is testable on the next scorer by recording cgroup memory events per job.

## Assumptions and readings taken

- "Null result" in the brief is read as "point estimate near 1 with a wide interval", which is what the pre-registration allows the post to say; I did not treat it as evidence of no effect.
- Power figures use the v6 paired sd as if groups were independent; they are optimistic by an unknown factor because of task clustering.
- Cost figures are model spend only, per the brief; VM and scorer costs are not included.
- I did not open eval logs directly; the four reader reports and the audit reproduce the headline numbers and agree with each other on every fact I relied on.

## Sources

- OpenAI GPT-4.1 prompting guide (persistence reminders, "close to 20%" on internal SWE-bench Verified): https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide
- Anthropic, Prompting Claude Haiku 5.5 (early stopping at low effort with long prompts; effort halves it): https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5
- Anthropic, Prompting best practices (long-horizon persistence snippet, context awareness): https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
- Anthropic, multi-agent research system (effort scaling, duplicated work, token usage explains 80% of variance): https://www.anthropic.com/engineering/multi-agent-research-system
- The Interaction Tax, arXiv 2608.23541 (local copy `research/sources/other/`)
- Park et al., Scaling Discovery through Test-Time Communication, arXiv 2609.21032
- Tran and Kiela, single-agent vs multi-agent under equal thinking-token budgets, arXiv 2604.02460
- ExComm, exploration-stage communication, arXiv 2605.22102
- Research priorities for multi-agent swarms (LessWrong): https://www.lesswrong.com/posts/XDybLbLbDsnEhnEWi/research-priorities-for-multi-agent-swarms-1
