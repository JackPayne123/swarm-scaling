# v6 team transcripts, groups A (clustering_outliers, generalized_eigenvalues_complex, least_squares, pca)

Reader: Opus 5.5 subagent, 2026-10-13. Read-only analysis of 8 team runs (`logs/v6-<task>-r<k>-divteam`) and their 32 diversified independents (`logs/v6-<task>-r<k>-div-i0..3`). No scorer calls, no spend. Scripts and per-run JSON extracts are in the session scratchpad under `v6a/` (`extract_a.py`, `spend_a.py`, `cands_a.py`, `families_a.py`, `walltime_a.py`, `delivery_a.py`, `lost_a.py`; per-agent timelines from `scripts/analyse_team_run.py` in `v6a/<run>/`). Haiku reasoning is encrypted, so everything below comes from readable reasoning summaries, tool calls, messages, published notes and candidate sources. Times are seconds since the run started; f is the agent's own spend as a fraction of its $0.10 budget at that moment.

## Bottom line

In these 8 groups the team won 3 and lost 5 (geometric mean ratio 0.90). Every gap traces to which arm contained an agent that found the task's one decisive technique in a fast, valid form, plus one selection accident (pca r2). Communication produced two small measured gains (+6% and +7% hidden, both on groups the team already led) and one combination that was lost to the reference-interference check. It also produced two wrong negative findings that teammates broadcast, one of which pruned the branch the independents won with (pca r1).

Coordination was cheap in dollars. It took 2.1% of team spend before an agent's last publish and 12.6% overall, and most of that overall share was padding toward the 80% gate. Messages reached the recipient model in a median 46 s. What limited the team was timing and information quality, not budget. In 4 of 8 teams the first message came after the team had spent 45-55% of its budget, or never came (lsq r2 sent zero messages). Each agent got about 24 model turns before its last publish, and the assigned priorities did not shape what agents built. A team-arm-only harness factor, four agents sharing 8 vCPUs with default multithreaded BLAS, caused 56 local bash timeouts (independents had 1) and the noisy local timings behind both wrong negatives.

## Scores and what decided each group

Hidden scores are the pick's median of 3 (from `analysis/v6-main-results.md`; geneigc r2 and pca r2 reconstructed from `logs/_scorer/i-0d30a2ddff1fb1a5d/jobs.jsonl` by matching candidate sources).

| Group | Team | Div. indep. | What decided it |
|---|---|---|---|
| clust r1 | 6.41 | 5.39 | Team had two agents (agent_0, agent_3) with full numba HDBSCAN pipelines (own condense/EOM) at 5.4-6.2 dev, built independently before the first message (472 s, team mean f=0.45). Best independent was i0's torch float32 core distances (5.21 dev). Member effect, not sharing. |
| clust r2 | 6.58 | 4.54 | agent_3 built the only full numba pipeline (5.38 dev). agent_0 found it through the registry, copied it and added a float32 distance matrix plus SoA layout (pick, 6.58). agent_3 reached its own float32 version 67 s later (6.18 hidden). Sharing gain about +6%. No independent built a full pipeline. |
| geneigc r1 | 6.35 | 6.44 | Single decisive trick, float32 sgeev on B^-1 A. Two team agents and three independents published it. Tie at the trick's ceiling. |
| geneigc r2 | 6.43 | 3.95 | agent_2 found float32 itself (reasoning before any teammate mentioned it), resisted agent_3's broadcast "float32 is dead", and later took agent_1's `la.eigvals` call form (optimal lwork), raising its hidden score from 6.02 to 6.43. The only independent with float32 (i1) used `lapack.sgeev` with the default minimal workspace and scored 3.15. |
| lsq r1 | 3.47 | 5.39 | Only independent i2 built a parallel (prange) numba LM with grid multistart (8.29 dev). No team agent wrote a prange LM. Team's subsampled LM (agent_3) lost validity on hidden in one version (stage 2 = 1.0). |
| lsq r2 | 6.43 | 7.70 | The team sent zero messages, so this is four independents in one box. Independent i3's subsample-then-polish VarPro LM beat team agent_1's parallel LM. |
| pca r1 | 20.42 | 23.95 | Independents i1 and i3 used float32 Gram plus float32 eigh (ssyevd) with a C list converter and single-thread BLAS. The team pruned float32 eigh after agent_3's noisy local test, and its one float32 candidate (agent_0-0, agent_1's design with a float32 Gram, 22.96 dev) failed the reference-interference check at finalize (1.172x against a 1.15x limit). |
| pca r2 | 7.25 | 20.96 | Selection picked agent_0-1 on stage 2 (13.08) but its three hidden scorings were bimodal (12.72, 7.25, 7.06, reference_inflation 1.03 vs 0.76-0.79). Shortlisted agent_3-1 scored 12.75. Separately, no team agent combined the C converter with a float32 Gram in a form that passed the interference check. Independent i2 did (20.96). |

## 1. Spend breakdown

Each model call's dollars (priced with `swarm_scaling.prices.usage_cost`, 1-hour writes, as metered) were split evenly across the tool calls it emitted. Categories are own work (bash/python not touching a teammate's files, publish), dev_eval calls, messaging and registry (SendMessage, wait_for_message, list_candidates, send_file), peer reads (bash touching `/app/registry` or another agent's directory), and gate (check_budget, submit). "Spend after last publish" is the share of dollars spent after the agent's final publish_candidate. dev_eval waiting costs no dollars, so queue time is reported as wall time.

| Arm (32 agents each) | Calls/agent | Own work | dev_eval calls | Messaging+registry | Peer reads | Gate | After last publish | dev_evals/agent | Queue wait s/agent | Queue share of wall | Wall s/agent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Team | 67 | 55% | 9% | 9.7% | 2.9% | 23% | 42% | 7.5 | 1,398 | 36% | 6,295 |
| Div. indep. | 65 | 60% | 15% | 0.3% | 0.5% | 25% | 39% | 15.0 | 2,698 | 49% | 4,991 |

Split by phase, coordination is mostly padding. Before an agent's last publish, team spend was 44.5% own work, 5.0% dev_eval, 1.7% messaging, 0.4% peer reads and 3.3% gate (independents 50.4%, 6.3%, 0.0%, 0.2%, 4.0%). After it, team spend was 10.9% own work, 4.1% dev_eval, 8.0% messaging, 2.5% peer reads and 19.4% gate. Team agents padded with wait_for_message, and independents padded with repeat dev_evals (geneigc r1 i0 ran 96 dev_evals and waited 15,875 s in the queue). Carrying teammates' message text in context cost an estimated 0.0-1.1% of each agent's spend (chars/4 tokens at the cache-read price for every later call).

| Group | Arm | Calls/agent | Own work | dev_eval | Msg+registry | Peer reads | Gate | After last publish | dev_evals/agent | Queue wait s/agent | Queue share of wall | Wall s/agent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clust r1 | team | 56 | 59% | 10% | 6.7% | 0.3% | 24% | 35% | 6.8 | 528 | 22% | 2,550 |
| clust r1 | indep | 44 | 66% | 12% | 0.7% | 0.0% | 21% | 29% | 6.5 | 460 | 37% | 1,170 |
| clust r2 | team | 46 | 58% | 13% | 6.5% | 3.1% | 19% | 34% | 9.5 | 2,240 | 57% | 4,042 |
| clust r2 | indep | 49 | 63% | 14% | 0.9% | 0.6% | 21% | 30% | 10.2 | 1,991 | 63% | 2,859 |
| geneigc r1 | team | 117 | 51% | 5% | 18.0% | 3.0% | 23% | 59% | 5.8 | 245 | 2% | 19,507 |
| geneigc r1 | indep | 105 | 50% | 22% | 0.0% | 0.0% | 28% | 65% | 36.8 | 6,185 | 36% | 12,872 |
| geneigc r2 | team | 94 | 34% | 10% | 16.2% | 12.9% | 27% | 61% | 11.2 | 1,780 | 22% | 8,658 |
| geneigc r2 | indep | 107 | 38% | 23% | 0.0% | 0.3% | 38% | 70% | 26.5 | 5,999 | 60% | 10,030 |
| lsq r1 | team | 43 | 72% | 6% | 1.4% | 0.0% | 21% | 27% | 3.2 | 870 | 49% | 1,780 |
| lsq r1 | indep | 34 | 69% | 12% | 0.3% | 0.7% | 18% | 18% | 11.0 | 1,693 | 51% | 2,653 |
| lsq r2 | team | 42 | 70% | 9% | 0.9% | 0.3% | 20% | 24% | 3.8 | 1,439 | 66% | 2,129 |
| lsq r2 | indep | 39 | 73% | 7% | 0.2% | 1.6% | 18% | 23% | 3.5 | 775 | 42% | 1,790 |
| pca r1 | team | 66 | 52% | 6% | 18.3% | 2.1% | 21% | 47% | 4.8 | 1,075 | 30% | 3,648 |
| pca r1 | indep | 68 | 57% | 16% | 0.4% | 0.3% | 26% | 43% | 16.0 | 2,457 | 46% | 4,916 |
| pca r2 | team | 74 | 48% | 14% | 9.6% | 1.3% | 27% | 46% | 15.2 | 3,004 | 38% | 8,048 |
| pca r2 | indep | 72 | 63% | 11% | 0.0% | 0.3% | 26% | 34% | 9.8 | 2,028 | 54% | 3,640 |

The highest single coordination share was geneigc r1 agent_2 (40% messaging, 7% peer reads, 16 dev_evals). It published one candidate at f=0.12 and spent the rest running tail-risk checks on teammates' candidates and waiting (25,627 s in wait_for_message). Mean queue wait per dev_eval was the same in both arms (team 186 s, independents 180 s). The full per-agent table for the 32 team agents is in the appendix.

Wall time is dominated by dev_eval and waiting, not model calls. Model calls took 1-17% of a team agent's wall time. dev_eval took 2-92%, wait_for_message 0-90%, and bash 2-90%. Bash reached 90% for geneigc r1 agent_0 because of 300 s timeouts, covered in section 5.

## 2. When findings were shared, and what followed

| Team | First message (s) | Sender f | Team mean f | Messages (unique) | Copies or derivatives of a teammate's candidate | Agents who ended endorsing a teammate's candidate |
|---|---|---|---|---|---|---|
| clust r1 | 472 | 0.41 | 0.45 | 3 | 0 of 11 published | 1 (agent_2) |
| clust r2 | 1,663 | 0.64 | 0.55 | 3 | 1 (agent_0-1 from agent_3-0, similarity 0.97) | 2 (agent_1, agent_2 recommend agent_0-1) |
| geneigc r1 | 856 | 0.15 | 0.13 | 7 | 0 published, agent_1's final file identical to agent_0-0 | 3 (agent_1, agent_2, agent_3 back agent_0-0) |
| geneigc r2 | 939 | 0.14 | 0.13 | 9 | 2 same-method rewrites (agent_0-0, agent_1-1 of agent_2-0) | 2 (agent_0, agent_3 back agent_2-1) |
| lsq r1 | 1,162 | 0.58 | 0.55 | 2 | 0 | 0 |
| lsq r2 | none | | | 0 | 0 | 0 |
| pca r1 | 1,025 | 0.25 | 0.25 | 15 (+3 send_file) | 2 (agent_3-1 byte-identical copy of agent_1-0, agent_0-0 at 0.98) | 2 (agent_2, agent_3 back agent_0-0) |
| pca r2 | 2,479 | 0.19 | 0.32 | 5 | 0 | 0 |

Similarity is a line-based difflib ratio between candidate sources. Copies by registry did not need messages. In clust r2 agent_0 (f=0.34) reasoned that "a quick directory scan of other agents' work could reveal an existing solution", dev_eval'd `/app/registry/agent_3-0` and built on it before any message had been sent.

**Convergence (the Interaction Tax pattern) appeared in the two geneigc teams and pca r1, and it came after each agent's own first solver.** In geneigc r1 four messages between 856 s and 1,294 s settled on agent_0's all-float32 path, and agent_3 stopped at f=0.18: "I'm not publishing a competing candidate; agent_0's all-f32 sgesv+sgeev (6.3x, 0 invalid) is the fastest I've seen ... I'm stopping to conserve budget" (agent_3, 1,429 s). The team then had 4 distinct valid candidates against the independents' 10. It cost nothing measurable, because the independents' best used the same trick and scored 6.44 against 6.35. In pca r1 the team converged on agent_1's design (C converter, Gram, `Z = G(GR)` subspace iteration) between 1,513 s and 2,409 s. That lineage was the pick, and the convergence removed the float32 eigh branch discussed below.

**The assigned priorities did not shape the work, so there was nothing to abandon.** Across 32 team agents the reasoning summaries mention the agent's own assigned priority once (clust r1 agent_2 at f=0.24, "my assigned priority is memory layout and execution parameters"). Every agent in both arms started from the same baseline family whatever its priority. All pca agents opened with Gram plus eigh (f=0.02-0.05), all geneigc agents with `solve(B, A)` plus eigvals, and all clust agents with a numba HDBSCAN reimplementation. Teams did divide work, but by pipeline stage and through messages. Examples are "I will NOT do eigen-step work (agent_0 owns randomized/Ritz)" (pca r1 agent_2, 1,233 s) and "Ideas I'm NOT doing: scipy QZ, threadpool tweaks" (geneigc r1 agent_1, 868 s).

**Method families.** I tagged code that agents wrote in bash/python calls, and the published candidate sources, with per-task keyword families (`families_a.py`, for example float32, C converter, thread control, subspace iteration and eigh for pca). The tagging is coarse because a fallback branch can trip a keyword.

| Measure (mean over 8 groups) | Team | Div. indep. |
|---|---|---|
| Families an agent touched in code | 4.2 | 4.5 |
| Families touched by the arm by f=0.25 | 5.0 | 5.5 |
| Families touched by the arm by the end | 5.6 | 6.0 |
| Families present in the arm's published candidates | 4.4 | 5.25 |
| Distinct valid candidates entering selection (sum over the 8 portfolios) | 60 | 70 |

Exploration breadth was about equal. The team published fewer families, most visibly in geneigc (2-3 against 4-5). The diversity lost was in finished candidates, which matches the pilot 5 finding.

## 3. Combinations, unique finds, picks and lost ideas

**Combinations better than any member alone.** I found two, both small.
- geneigc r2. agent_1 (2,334 s): "published agent_1-1 = same f32 path as agent_2-0 but calls la.eigvals(C.astype(float32)) (scipy does the optimal lwork query)". agent_2 benchmarked both call forms, found the `la.eigvals` form 11% faster locally, and published agent_2-1, the pick. agent_2-0 scored 6.02 on hidden and agent_2-1 scored 6.43.
- clust r2. agent_0-1 is agent_3-0's pipeline plus agent_0's float32 distance matrix and SoA loop. It scored 6.58, against 6.18 for agent_3's own float32 version, which agent_3 reached independently during padding.
- pca r1. agent_0-0 combined agent_1's no-eigh design with a float32 Gram (22.96 dev, best in the team). It was flagged invalid at finalize stage 1 (reference ran 1.172x slower interleaved, limit 1.15x) and never reached the shortlist. agent_0 had already seen the flag once ("seed 500 23.8x but reference-interference check flagged 1.151x", 3,172 s). Whether it would have beaten 20.42 on hidden is unknown. It failed the check in 2 of its 4 measurements, and a failed hidden scoring scores 1.0.

**Techniques an independent found that the team never used.**
- lsq r1. Parallel prange numba LM with grid-only multistart (i2, the pick). No team agent used prange.
- pca r1. Float32 eigh (ssyevd or `eigh(driver="evd")` on a float32 Gram) together with a C converter and thread control (i1 23.95, i3 22.68). The team pruned it twice. agent_3 (1,417 s): "float32 eigh ... gave NO gain at size 24 (10.9 ms vs 9.3)", measured on the contended box. agent_3 (1,752 s): "float32 Gram and float32 subspace iteration FAILS is_solution ... so I'm dropping float32", which is true only for the subspace-iteration variant.
- pca r2. C converter, float32 ssyrk and ssyevd in one candidate that passed the interference check (i2, 20.96). The team had the parts but did not assemble them. agent_1 (3,319 s) asked for it: "If your C-ext conversion lands, please combine it with this float32 eigh/Gram path and dev_eval it." agent_3 (3,414 s) dropped its own version: "my f32-Gram + scipy evd variant got ~25-29x but was flagged 'reference ran 1.18-1.35x slower interleaved' twice". agent_0 tested a float32 evd variant (candidate C) and kept the float64 one after a single same-seed comparison.
- clust r1. torch cdist and kthvalue in float32 (i0). The team did not need it.

**Techniques the team found that no independent used.** In clust r1 and r2, full numba pipelines above 5x dev (independents peaked at 5.21 and 4.44). In geneigc r2, a fast float32 call form. These account for the team's three wins.

**Was the pick from the best lineage?** In 7 of 8 teams the pick was the best hidden score in the shortlist (oracle = pick). The exception is pca r2, where the pick (7.25) lost to shortlisted agent_3-1 (12.75). Its three hidden scorings were 12.72, 7.25 and 7.06, and each tracked reference_inflation (1.03 against 0.76-0.79). The solver sets OpenBLAS to 1 thread during solve and restores it, so its score depends on the thread state the reference runs in. In pca r1, the best team candidate (agent_0-0) was removed by the interference flag before selection.

**Good ideas found and then dropped.** The pca float32 eigh branch in r1 and r2 (above). In geneigc r2, agent_0's torch float32 path, dropped after an interference flag at 1.30x, which was correct because it was no faster than scipy float32. In lsq r1, agent_3-0's 2,048-point subsample (8.08x on agent_3's own dev_eval, 6.73 at finalize but invalid), dropped after one MSE-ratio failure (1.088) out of 40, which was correct. In geneigc r1, agent_2 said "float32 not viable (tol 1e-6)" at 856 s and agent_1 corrected it 12 s later ("float32 is viable, measured"). In geneigc r2, agent_3 said "Float32 sgeev is NOT faster here ... so float32 is dead" (1,009 s), and agent_2 settled it with dev_eval instead ("Agent_3's numbers conflict with mine ... run dev_eval on solver_f32 to get an authoritative answer", then 5.74x).

## 4. Does $0.10 of Haiku leave room for coordination to pay off?

- **Turns.** Team agents made 67 model calls on average (median 63.5, range 34-129). Independents made 65 (median 54.5, range 21-119). Before their last publish, agents made 23.5 calls (team) and 25.6 (independents). Median call cost was $0.00089 (mean $0.00124) at a median prompt of 57k tokens. A typical agent got one to three solver versions.
- **Dollars do not bind.** Agents published their first candidate at f=0.36 (team) and 0.37 (independents) and their last at about f=0.58. They then spent 42% and 39% of budget padding to the 80% gate, with 113 and 116 refused submits. Coordination before the last publish was 2.1% of spend. No team agent crossed Haiku's 100k-token long-prompt tier (3 of 32 independents did).
- **Timing binds.** A message can only redirect an agent while it is still choosing. In clust r1, clust r2, lsq r1 and lsq r2 the channel opened after the team's mean f was 0.45 or later, or never, when every agent had committed to its lineage. In geneigc r1 and r2 it opened at f=0.13, but by then the decisive trick had been found in code by the agents who would publish it (geneigc r1 agent_0 at f=0.09 and agent_1 at f=0.13), so the messages mostly spread verification and stopped further search.
- **Task structure binds.** On these four tasks the score is set by one or two tricks per task (float32 sgeev, float32 Gram plus C fill plus thread control, parallel or subsampled LM, full numba HDBSCAN). With about 24 productive turns per agent, a team of 4 behaves like a best-of-4 over what its agents publish, as Haiku 2 dst found. Sharing then helps only through small call-form tweaks (+6-7%), and hurts when a noisy negative prunes a branch.

My answer is that the $0.10 budget pays for coordination in dollars many times over, but the agents use it late and the measurement noise in the shared box makes what they share less reliable than what an independent measures for itself. A larger budget alone would mostly buy more padding unless the extra turns are spent before the first publish.

## 5. Harness and measurement issues (not agent behaviour)

1. **Shared CPU in the team box, team arm only.** Teams ran four agents in one container limited to 8 vCPUs (cpuset 0-7). Independents ran alone with 2 vCPUs. The per-agent core count matches, but every team agent's numpy and scipy saw 8 cores and started 8 BLAS threads, so the container ran 32 or more threads on 8 cores. Agents reported it themselves, "load avg 14 on 8 cores" (geneigc r2 agent_1, 939 s) and "A load average of 32 on 8 cores explains it" (geneigc r1 agent_3, reasoning at f=0.24). Team agents hit 56 bash timeouts at the 300 s tool limit (geneigc r1 35, geneigc r2 12, pca r1 4, pca r2 3, clust r1 2). Independents hit 1. That cost wall time, not dollars, but it corrupted local timings. Both wrong negatives above (geneigc r2 "float32 is dead", pca r1 "float32 eigh gave NO gain") came from local measurements on the contended box. This confounds the communication contrast on BLAS-bound tasks.
2. **The reference-interference check and per-call thread toggling (both arms, decisive on pca).** Solvers that set OpenBLAS to 1 thread during solve land near the 1.15x interference limit (1.151, 1.172, 1.18-1.35, 1.271 for indep pca r2 i2 agent_0-1). Hidden scores of such solvers are bimodal with reference_inflation (pca r2 pick 12.72 at 1.03 against 7.06-7.25 at 0.76-0.79). Across the geneigc and pca scorer jobs, reference_inflation ranged 0.63-1.11, so the reference is often faster when interleaved than alone. This decided pca r2 and removed pca r1's top team candidate. I count it as a measurement-design issue, not a bug.
3. **SendMessage schema.** 16 SendMessage calls failed because `summary` is required (clust r1 2, geneigc r1 1, geneigc r2 6, pca r1 4, pca r2 3). At the median call cost that is about $0.014, roughly 0.5% of team spend. It slowed the first broadcast in clust r1 by one call.
4. **Message delivery works.** All 118 message records reached the recipient's model input. Median latency was 46 s, 75th percentile 144 s, maximum 565 s. Latency is bounded by the recipient's in-flight tool call (a queued dev_eval or a 300 s bash), not by the channel.
5. **Gate padding and budget display.** check_budget reports dollars to 2 decimals, so agents polled it repeatedly while padding. clust r1 agent_2 reasoned that "The budget display is stuck at $0.06". Gate calls were 23-25% of spend in both arms. Team agents padded with long wait_for_message calls (up to 25,627 s in one agent). Independents padded with repeat dev_evals, which doubled their scorer load (15.0 against 7.5 calls per agent).
6. **Budget overshoot.** lsq r1 i3 spent $0.1047. Its context crossed the 100k-token tier and the second-to-last call cost $0.0225, five times the base price. One case only.
7. **Finalize kill.** lsq r2 team agent_3's final workspace scored "no result, exit 137: Killed". Its published candidates were scored normally, so selection was unaffected. The cause was not established.
8. **Lost portfolio records** for geneigc r2 and pca r2 are already logged in `v6-main-results.md`. I rebuilt the shortlists from `jobs.jsonl` by matching candidate sources to finalize `sources`. Two stage-2 jobs and three hidden jobs per group did not match by prefix, which is why some rows in `lost_a.py` output show None.

## 6. Method notes and limits

- One team run per group, 8 groups. These are observations, not estimates of an effect size.
- Spend categories split a call's cost evenly over the tools it emitted. A call that issues wait_for_message costs a full context read, so "messaging" includes the context cost of the waiting turn as well as the message itself.
- Lineage similarity is textual (difflib on stripped lines). Two agents can reach the same method independently with low similarity (geneigc float32 paths are 0.38-0.60 similar).
- Family tags are keyword matches. I used them only for arm-level counts and checked the conclusions against the published notes.
- Message counts in the appendix are per-recipient records (a broadcast to 3 counts 3).

## Appendix: per-agent spend, team runs

| Run | Agent | Priority | Calls | Own work | dev_eval | Msg+registry | Peer reads | Gate | After last publish | Msg records sent/recv | dev_evals | Queue wait s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clust r1 | 0 | 0 | 57 | 50% | 16% | 6% | 0% | 28% | 35% | 0/2 | 12 | 1,471 |
| clust r1 | 1 | 1 | 51 | 64% | 7% | 9% | 0% | 21% | 39% | 0/2 | 4 | 108 |
| clust r1 | 2 | 2 | 65 | 55% | 9% | 8% | 1% | 27% | 37% | 4/1 | 5 | 175 |
| clust r1 | 3 | 3 | 51 | 69% | 9% | 4% | 0% | 18% | 30% | 3/2 | 6 | 357 |
| clust r2 | 0 | 0 | 63 | 43% | 30% | 4% | 6% | 17% | 45% | 3/2 | 25 | 4,663 |
| clust r2 | 1 | 1 | 40 | 65% | 6% | 8% | 3% | 18% | 28% | 3/2 | 3 | 1,059 |
| clust r2 | 2 | 2 | 38 | 63% | 9% | 1% | 4% | 22% | 36% | 0/3 | 6 | 2,079 |
| clust r2 | 3 | 3 | 41 | 61% | 8% | 13% | 0% | 18% | 28% | 3/2 | 4 | 1,159 |
| geneigc r1 | 0 | 0 | 115 | 71% | 2% | 5% | 0% | 22% | 74% | 6/5 | 3 | 447 |
| geneigc r1 | 1 | 1 | 129 | 54% | 2% | 19% | 2% | 22% | 78% | 3/6 | 3 | 334 |
| geneigc r1 | 2 | 2 | 120 | 17% | 13% | 40% | 7% | 24% | 85% | 9/4 | 16 | 83 |
| geneigc r1 | 3 | 3 | 104 | 62% | 2% | 8% | 3% | 25% | no publish | 3/6 | 1 | 116 |
| geneigc r2 | 0 | 0 | 101 | 30% | 2% | 24% | 22% | 22% | 65% | 6/6 | 4 | 1,225 |
| geneigc r2 | 1 | 1 | 100 | 31% | 29% | 12% | 4% | 24% | 61% | 6/7 | 34 | 2,910 |
| geneigc r2 | 2 | 2 | 93 | 44% | 5% | 11% | 0% | 40% | 52% | 4/7 | 4 | 1,579 |
| geneigc r2 | 3 | 3 | 80 | 30% | 4% | 17% | 26% | 23% | 67% | 9/5 | 3 | 1,405 |
| lsq r1 | 0 | 0 | 54 | 68% | 6% | 4% | 0% | 22% | 46% | 3/1 | 3 | 759 |
| lsq r1 | 1 | 1 | 39 | 70% | 7% | 2% | 0% | 20% | 21% | 3/1 | 3 | 865 |
| lsq r1 | 2 | 2 | 34 | 71% | 4% | 0% | 0% | 25% | 29% | 0/2 | 3 | 749 |
| lsq r1 | 3 | 3 | 46 | 79% | 6% | 0% | 0% | 16% | 13% | 0/2 | 4 | 1,107 |
| lsq r2 | 0 | 0 | 41 | 78% | 5% | 0% | 0% | 17% | 21% | 0/0 | 3 | 1,309 |
| lsq r2 | 1 | 1 | 40 | 78% | 3% | 2% | 0% | 16% | 20% | 0/0 | 2 | 905 |
| lsq r2 | 2 | 2 | 43 | 62% | 11% | 1% | 0% | 26% | 23% | 0/0 | 4 | 1,500 |
| lsq r2 | 3 | 3 | 43 | 60% | 18% | 0% | 1% | 21% | 32% | 0/0 | 6 | 2,041 |
| pca r1 | 0 | 0 | 55 | 59% | 14% | 11% | 0% | 16% | 39% | 9/10 | 9 | 1,549 |
| pca r1 | 1 | 1 | 72 | 59% | 4% | 19% | 0% | 19% | 42% | 7/10 | 4 | 1,071 |
| pca r1 | 2 | 2 | 64 | 48% | 5% | 23% | 4% | 20% | 58% | 9/9 | 4 | 1,025 |
| pca r1 | 3 | 3 | 72 | 43% | 2% | 20% | 4% | 30% | 48% | 12/8 | 2 | 655 |
| pca r2 | 0 | 0 | 70 | 54% | 19% | 6% | 0% | 20% | 19% | 3/4 | 17 | 3,128 |
| pca r2 | 1 | 1 | 77 | 26% | 18% | 4% | 0% | 53% | 62% | 4/3 | 31 | 4,133 |
| pca r2 | 2 | 2 | 83 | 58% | 7% | 10% | 5% | 20% | 73% | 3/3 | 7 | 2,435 |
| pca r2 | 3 | 3 | 64 | 53% | 12% | 18% | 0% | 16% | 32% | 3/3 | 6 | 2,321 |
