# v6 diversified teams vs diversified independents: sinkhorn, sparse_lowest_eigenvalues_posdef, shortest_path_dijkstra (2026-10-13)

Reader: Opus subagent, read-only. Six team runs (`logs/v6-<task>-r<k>-divteam`) and their 24 diversified independents (`-div-i0..3`). Haiku reasoning is encrypted, but each model event carries a readable reasoning summary, which `scripts/analyse_team_run.py` prints; quotes below come from those summaries, messages and candidate notes. Spend is recomputed per call with `swarm_scaling.prices.usage_cost` (1-hour writes pinned) and matches each agent's metered total to within $0.0000005 on all 48 agents. Scratch data and scripts are in this scratchpad (`x-*.json`, `spend.json`, `teamsb/`).

## Summary

- In these six groups the team won 3 of 6, geometric-mean ratio 1.08 (sinkhorn r2 1.19, sparseig r1 1.67, spd r2 1.01; sinkhorn r1 0.96, sparseig r2 0.93, spd r1 0.89). The arm-level 0.97 comes from the other four tasks.
- Each team's result was set by one agent's lineage. Teams won when one member found a technique that no independent in the group found (sparseig r1 float32 shift-invert, sinkhorn r2 pthread reader with fused exp). They lost when an independent pushed one lineage further than any team member (spd r1 int16 multi-pivot Floyd-Warshall, sparseig r2 strtri inverse).
- No early diversity collapse. Teams published more method families than the independents, both by 50% of budget (20 vs 12 across the six groups) and by the end (23 vs 16).
- What did happen after a lead appeared (at 0.31-0.62 of the leader's budget) was stopping. 13 of 24 team agents announced that they were stopping or deferring to a teammate, at 0.21-0.65 of their budget. When the lead finding reached them, recipients still had 17-73 model calls left (fractions 0.32-0.63), but published only 15 more candidates between 18 recipients. Two teams converged on the leader's code (sparseig r2, spd r2) and gained at most 3% on hidden.
- Coordination overhead is modest but larger than in haiku2. Messaging, registry tools and reading peers' files took 5.7% of team spend, and wait_for_message turns another 5.8%. Most wait_for_message use is padding to the 80% gate, which independents do with check_budget instead. Gate and wait spend is 26.1% in teams and 25.6% in independents.
- Haiku at $0.10 gets 43-103 model calls per agent (median 60.5 team, 54 independent, about $0.0013 per call), of which a median 27 (team) and 20 (independent) come before the agent's last publish. The binding constraint is behaviour, not turns. Both arms stop publishing at about half their budget (median last publish at 0.45 team, 0.49 independent) and pad for the rest.
- Harness confounds. Team agents share 8 cores, independents get 2 each. 3 of 6 team boxes were flagged CPU-throttled against 1 of 24 independent boxes, and the larger visible core count led team agents to OpenMP thread counts that collapsed on the scorer. Teams started 101-175 min after their group's independents in 4 of 6 groups. 53 of 85 peer messages arrived while the recipient was blocked in dev_eval, median delay 163 s. 11 of 53 SendMessage calls failed schema validation.

## Scores and selection

Hidden score is the pick's median of 3 (from `analysis/portfolio/*.jsonl`). "Lineage" means the code line an agent built across versions.

| Group | Team | Div. indep. | Ratio | Team pick (stage-1 rank) | Team shortlist lineages | Div. pick | Div. shortlist lineages |
|---|---|---|---|---|---|---|---|
| sinkhorn r1 | 7.34 | 7.67 | 0.96 | agent_2-3, serial C fused sweep (1) | 2 (agent_2 x2, agent_0) | i2 OpenMP-2-thread C (1) | 3 |
| sinkhorn r2 | 9.72 | 8.17 | 1.19 | agent_1-0, pthread reader + fused exp + BLAS loop (1) | 2 | i3 pthread fill + BLAS loop (1) | 2 |
| sparseig r1 | 6.16 | 3.70 | 1.67 | agent_0-1, f32 Cholesky inverse + ARPACK + f64 Rayleigh-Ritz (1) | 2 (agent_0 x2, agent_3) | i3 f32 dense eigh + Krylov RR (1) | 3 |
| sparseig r2 | 7.58 | 8.13 | 0.93 | agent_2-1, agent_0-2 with NCV=12 (2) | 1 (all three are agent_0-2 or its variants) | i1 f32 Cholesky + strtri inverse + ARPACK (1) | 2 |
| spd r1 | 39.11 | 44.06 | 0.89 | agent_0-0, numba Dial + int tolist (1) | 3 (Dial, FW, bitset) | i2 int16 4-pivot FW (1) | 1 (i2 x3) |
| spd r2 | 38.74 | 38.25 | 1.01 | agent_2-3, Dial + int tolist (1) | 2 code lineages, all three with agent_0's int-tolist output | i0 Dial + witness edge pruning (1) | 2 |

The pick was always from the team's best lineage. Within the shortlist the oracle bound equalled the pick in all 12 portfolios, so no group lost score at stage 2 or on hidden scoring. Where losses happened they came earlier, in what got built.

## Spend breakdown

Each model call's dollars are split evenly over the tool calls it issued. "Own work" is bash/python (excluding reads of peers' files), publish_candidate, and for independents everything that is not dev_eval or the gate. "Coordination" is SendMessage, wait_for_message, list_candidates, TaskList and bash reads of the registry or another agent's directory. No call was text-only. Own work also contains some padding (file listings, re-reads), so it overstates productive spend.

| Arm | Own work | dev_eval | Send | wait_for_message | Registry tools + peer file reads | check_budget | submit |
|---|---|---|---|---|---|---|---|
| Team (24 agents) | 52.1% | 16.0% | 2.5% | 5.8% | 3.2% | 14.9% | 5.4% |
| Diversified indep. (24) | 60.9% | 13.3% | 0 | 0 | 0.3% (own registry) | 20.8% | 4.8% |

Spend after each agent's last publish: 40% team, 42% independents. 70% of team dev_eval calls (266 of 378) and 62% of independent ones (179 of 287) came after the caller's last publish, mostly re-timing on new seeds. 109 team dev_eval calls (29%) timed a teammate's registry entry. spd r2 agent_0 alone ran 51 such calls (agent_2-3 x28, agent_3-1 x23, 29 seeds) after saying at 1013 s "I'm not pursuing further".

Per team agent:

| Run | Agent (priority) | Calls | Own work | dev_eval | Coordination | Gate | Spent after last publish | First / last publish (budget fraction) | dev_eval calls | Queue wait s | wait_for_message s | Active s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sinkhorn-r1 | agent_0 (p0) | 56 | 62% | 8% | 6% | 23% | 55% | 0.35 / 0.36 | 7 | 2351 | 900 | 4412 |
| sinkhorn-r1 | agent_1 (p1) | 71 | 48% | 27% | 5% | 20% | 45% | 0.22 / 0.45 | 24 | 6208 | 0 | 7872 |
| sinkhorn-r1 | agent_2 (p2) | 51 | 68% | 6% | 10% | 16% | 22% | 0.27 / 0.63 | 5 | 1592 | 1710 | 4305 |
| sinkhorn-r1 | agent_3 (p3) | 50 | 49% | 13% | 21% | 17% | 34% | 0.38 / 0.54 | 4 | 1246 | 6025 | 7888 |
| sinkhorn-r2 | agent_0 (p0) | 49 | 67% | 9% | 10% | 14% | 24% | 0.42 / 0.63 | 5 | 1522 | 540 | 2792 |
| sinkhorn-r2 | agent_1 (p1) | 70 | 66% | 3% | 11% | 19% | 57% | 0.35 / 0.35 | 3 | 1320 | 3383 | 5334 |
| sinkhorn-r2 | agent_2 (p2) | 72 | 35% | 38% | 7% | 19% | 75% | 0.20 / 0.20 | 34 | 3797 | 34 | 5406 |
| sinkhorn-r2 | agent_3 (p3) | 50 | 63% | 4% | 13% | 20% | 31% | 0.16 / 0.56 | 3 | 1309 | 3466 | 5464 |
| sparseig-r1 | agent_0 (p0) | 50 | 69% | 6% | 7% | 19% | 31% | 0.24 / 0.56 | 3 | 316 | 1010 | 2303 |
| sparseig-r1 | agent_1 (p1) | 61 | 54% | 6% | 20% | 21% | (never published) | - | 3 | 310 | 1038 | 2327 |
| sparseig-r1 | agent_2 (p2) | 61 | 64% | 7% | 16% | 13% | 28% | 0.30 / 0.59 | 5 | 568 | 2472 | 3844 |
| sparseig-r1 | agent_3 (p3) | 78 | 35% | 22% | 14% | 29% | 51% | 0.23 / 0.40 | 21 | 3310 | 0 | 5084 |
| sparseig-r2 | agent_0 (p0) | 60 | 61% | 5% | 18% | 16% | 26% | 0.34 / 0.60 | 5 | 2122 | 420 | 3678 |
| sparseig-r2 | agent_1 (p1) | 57 | 63% | 6% | 13% | 19% | 22% | 0.21 / 0.64 | 5 | 2133 | 1043 | 4326 |
| sparseig-r2 | agent_2 (p2) | 53 | 64% | 7% | 14% | 14% | 15% | 0.14 / 0.70 | 9 | 1799 | 1104 | 5368 |
| sparseig-r2 | agent_3 (p3) | 59 | 51% | 23% | 6% | 21% | 37% | 0.25 / 0.51 | 31 | 4004 | 0 | 6890 |
| spd-r1 | agent_0 (p0) | 72 | 49% | 12% | 15% | 24% | 64% | 0.29 / 0.29 | 11 | 1463 | 2318 | 5142 |
| spd-r1 | agent_1 (p1) | 72 | 41% | 14% | 26% | 19% | 49% | 0.25 / 0.41 | 11 | 374 | 4255 | 5555 |
| spd-r1 | agent_2 (p2) | 103 | 29% | 29% | 14% | 28% | 64% | 0.10 / 0.30 | 32 | 5812 | 5020 | 13801 |
| spd-r1 | agent_3 (p3) | 58 | 40% | 25% | 10% | 25% | 55% | 0.16 / 0.36 | 20 | 2373 | 45 | 4388 |
| spd-r2 | agent_0 (p0) | 82 | 22% | 35% | 17% | 26% | 61% | 0.32 / 0.32 | 60 | 1842 | 7203 | 13295 |
| spd-r2 | agent_1 (p1) | 43 | 59% | 15% | 4% | 21% | 30% | 0.54 / 0.57 | 10 | 3690 | 780 | 5176 |
| spd-r2 | agent_2 (p2) | 87 | 40% | 37% | 4% | 20% | 46% | 0.08 / 0.44 | 38 | 5520 | 607 | 8380 |
| spd-r2 | agent_3 (p3) | 77 | 45% | 29% | 2% | 25% | 44% | 0.33 / 0.45 | 29 | 5963 | 900 | 9003 |

Independents per group (mean of 4): calls 42-65, gate 20-30%, spent after last publish 27-55%. The full per-run table is in `spend.json`.

Wall time. dev_eval queue waits averaged 161 s per call in teams (378 calls, median 130, p90 394, max 719) and 170 s in independents (287 calls, median 154, p90 302, max 535). As a share of active time, team agents spent 43% waiting in the dev_eval queue, 57% inside dev_eval and 31% blocked in wait_for_message. Independents spent 64% in the queue and 82% inside dev_eval. Team agents ran 1.3-2.3x longer in wall time (mean active 3389-8964 s against 2158-3866 s) because wait_for_message blocks for up to 600 s per call. That costs wall time, not dollars. The 1-hour cache held across every wait (0 calls in 30 runs re-wrote over half their input after a dev_eval).

## Idea flow, timeline per team

Budget fraction (bf) is the sender's own spend at the time.

| Team | First substantive share | Lead finding broadcast (became the pick's lineage) | Recipients' bf on delivery | Stop or defer announcements |
|---|---|---|---|---|
| sinkhorn r1 | 624 s, agent_1 bf 0.23 (C list reader, 5.6x) | 1563 s, agent_2 bf 0.56 (single-thread C sweep, avoid OMP) | 0.42-0.52 | agent_1 bf 0.47, agent_3 bf 0.56 |
| sinkhorn r2 | 1322 s, agent_2 bf 0.22 | 1483 s, agent_1 bf 0.37 (9.66x) | 0.32-0.44 | agent_2 bf 0.40, agent_0 bf 0.65 |
| sparseig r1 | 1111 s, agent_2 bf 0.52 | 1234 s, agent_0 bf 0.57 (f32 Cholesky, 6.23x) | 0.39-0.57 | agent_1 bf 0.52, agent_3 bf 0.42 |
| sparseig r2 | 702 s, agent_1 bf 0.20 | 2661 s, agent_0 bf 0.62 (7.12x) | 0.44-0.56 | agent_2 bf 0.47, agent_1 bf 0.59 |
| spd r1 | 307 s, agent_1 bf 0.26 (int output matters, 49x) | 613 s, agent_0 bf 0.31 (Dial, 56.8x) | 0.32-0.46 | agent_1 bf 0.49, agent_3 bf 0.45 (agent_2 deferred at 628 s, bf 0.34) |
| spd r2 | 939 s, agent_2 bf 0.18 | 2388 s, agent_0 bf 0.32 (int tolist on agent_3-0, 54.5x) | 0.41-0.63 | agent_0 bf 0.21 |

31 distinct messages (85 deliveries) across six teams, 3-8 per team. Messages are findings reports with numbers and a claim of territory ("Don't duplicate FW", "the Dijkstra inner loop... it's yours"). The task list was opened twice and never used.

After the lead finding, recipients' remaining budget went mainly to the gate. The announcing agents' post-announcement spend (18-59% of budget each) went 32-54% to check_budget/submit, 0-60% to dev_eval re-timing, and 0-57% to wait_for_message. Two reasoning summaries state the intent directly:

- spd r1 agent_1, 867 s: "I think it's reasonable to conclude: submit solver4 as my final candidate, then make a few cheap calls (list_candidates, maybe wait_for_message) just to reach the minimum spend before finalizing." At 1176 s: "I'll spend the gap doing several wait_for_message calls of 600s each... a reasonable way to burn through the required budget."
- spd r2 agent_2, 2381 s: "I should just wait for incoming messages using wait_for_message, which costs nothing extra and lets others' activity accumulate usage." Each wait_for_message call is a model turn, about $0.0013.

Abandonment of the assigned priority is hard to separate from ordinary behaviour, because the priority text is rarely mentioned in reasoning (one sinkhorn r1 agent cites "my assigned priority" once). What the transcripts do show is that agents abandoned their own lineage once a teammate's was ahead. They did not copy the leader's lineage, with two exceptions.

- Deferral without copying (4 teams). spd r1 agent_1 stopped Floyd-Warshall at 53.3x against Dial's 57.5x on the same seeds ("I'm stopping FW tuning... I'm not working on Dijkstra to avoid overlap"). sinkhorn r2 agents 0, 2 and 3 all declared they would not try to beat agent_1-0. sparseig r1 agent_3 at bf 0.42: "Agent_0-1 at 6.23x is the best I've seen, so I'm not publishing more variants."
- Convergence onto the leader's code (2 teams). In sparseig r2, agent_3 (3668 s) said "the team is converging on agent_0's approach as the best one, so duplicating that effort probably isn't worth it" and then built an f32 LU variant of agent_0-2 anyway. agent_2 built an NCV=12 variant. All three shortlisted sources are that one lineage, with hidden scores 7.32, 7.58 (pick) and 7.34. In spd r2, three agents adopted agent_0's int64-tolist output, and all three shortlisted sources carry it (hidden 38.74, 37.74, 38.04).

Method families published (corrected by hand from the candidate notes; one family per candidate):

| Group | Team by bf 0.5 | Team at end | Div. by bf 0.5 | Div. at end | Team families |
|---|---|---|---|---|---|
| sinkhorn r1 | 4 | 4 | 1 | 2 | C fused serial sweep, C reader + BLAS loop, numba loop, C OpenMP loop |
| sinkhorn r2 | 2 | 3 | 2 | 2 | BLAS loop, C OpenMP loop, C fused serial |
| sparseig r1 | 4 | 4 | 1 | 4 | f64 shift-invert, tuned eigsh, f64 inverse Lanczos, f32 shift-invert + RR |
| sparseig r2 | 4 | 5 | 3 | 3 | tuned eigsh, dense eigh subset, f64 LU shift-invert, numba thick-restart Lanczos, f32 shift-invert + RR |
| spd r1 | 4 | 4 | 2 | 2 | heap Dijkstra, Dial, Floyd-Warshall, bitset BFS |
| spd r2 | 2 | 3 | 3 | 3 | heap Dijkstra, Dial, multi-source Bellman-Ford |
| Total | 20 | 23 | 12 | 16 | |

Teams published 2.29 candidates per agent against 2.0, and their first publish came earlier (median bf 0.25 vs 0.34). Family counts in the 3-source shortlists were equal (10 vs 10).

## Combination, independent-only finds, lost ideas

Combinations that reached the pick.

- spd r1. agent_1's 307 s message ("Float64 output tolist costs ~1.7ms vs int ~0.45ms... int output matters") preceded agent_0's switch from a float object table (code at 410 s) to int64 `out.tolist()` (code at 520 s). Its summary at 522 s says it is "confident the grader's is_solution check works with integer values since other agents' solutions used ints successfully". agent_0's resulting candidate was the pick.
- spd r2. agent_0 combined agent_3-0's Dial kernel with int tolist (54.5x against 44.5x on the same seeds). agent_2 then adopted the int tolist for its own kernel (57.8x, the pick). Hidden gain over agent_0's version was 1.8%.
- sparseig r2. The NCV tweak on agent_0's lineage was picked and scored 7.58 on hidden, against 7.34 for agent_0-2. That gain of 3.3% is within the spread of the three-scoring medians elsewhere.

No team combined two members' different methods into a new solver. In sinkhorn r1 the r2 winner's recipe (threaded reader with fused exp, BLAS loop) was split across agent_0 (pthread fill, sent to agent_1 only at 650 s) and agent_1 (BLAS loop, OpenMP fill). agent_1-2 self-reported 8.10x but recorded 6.41 on the selection dev set, ranked 4th, and was never scored on hidden.

Found by an independent, never by the team in the same group.

- spd r1. int16 storage with multi-pivot Floyd-Warshall (i1 36.1x, i2 65.45x dev, 44.06 hidden). The team's FW agent used int32 with K-blocking and stopped at 50.1x. Team agent_3 had considered an int16 sentinel for Dial at 172 s and did not build it.
- sparseig r2. f32 Cholesky with strtri explicit triangular inverse (i1, 8.39 dev, 8.13 hidden). The team used strtrs solves or LU.
- spd r2. 2/3/4-hop witness edge pruning before Dial (i0, 38.25 hidden). The team's best scored higher without it.
- sinkhorn r1. A working OpenMP loop with 2 threads (i0 6.81, i2 7.55 dev) and vectorised polynomial exp in C (i0, i2, i3). Every team OpenMP-loop variant collapsed (1.2-1.4x dev).

Found by the team, never by an independent in the same group. Examples are f32 shift-invert with Rayleigh-Ritz (sparseig r1, worth +66%), the pthread reader with fused exp and a per-iteration marginal check (sinkhorn r2, 9.57 dev against i3's 8.10), bitset level BFS (spd r1) and multi-source Bellman-Ford (spd r2).

Wrong negative findings sent by message. spd r2 agent_0 at 1013 s wrote "Int64 tolist (returning ints) is no faster than the object gather", then reversed it at 2388 s with a 54.5x vs 44.5x head-to-head. sparseig r2 agent_1 (702 s, "Dense/sparse-LU shift-invert looks no faster") and agent_2 (924 s, "Shift-invert with dense Cholesky + eigsh (sigma=0) was SLOWER") both dismissed the family that won, in its float64 form. agent_0 went on to build the f32 version regardless. The sinkhorn "avoid OMP" consensus (r1 agent_2, r2 agents 0, 2 and 3) was correct for the thread counts the team used, but not for the 2-thread version that worked for independents (next section).

## Harness and environment

- Arm-dependent agent box. A team's 4 agents share `cpuset 0-7` (8 cores, 29-61 GB), while each independent gets `cpuset 0-1`. In 3 of 6 team boxes the CPU monitor's flag is set (sparseig r1 and r2 37% of periods throttled, spd r1 20%), against 1 of 24 independent boxes (sinkhorn r2 i2). Team agents repeatedly called their local timings unreliable ("my container is noisy", spd r1 agent_0 measured a parallel speedup of "only 1.8x under load"). The box also changed what agents chose. sinkhorn r1 agent_2 saw "os.cpu_count()=16 but nproc=8" and ran 16 OpenMP threads (1.2x on dev), and r2 agent_3 used 8 threads (1.35x). Independents on 2-core boxes picked 2 threads and got 6.8-7.55x from the same family. The scorer slot is 8 cores for both arms, so the scorer itself is not the asymmetry. Local experimentation differs, and this is a confound on the sinkhorn comparison.
- Start-time skew. In 4 of 6 groups the team started 101-175 min after its group's independents (sinkhorn r1 170, sinkhorn r2 101, sparseig r2 142, spd r2 175). This is probably the wait for a 16-vCPU VM, which I did not check. Mean queue waits were close (161 vs 170 s), so the skew changed scorer load little on average.
- Message delivery waits for a turn boundary. Of 85 peer messages, 53 arrived while the recipient was in dev_eval (median delay 163 s, max 546 s) and 19 while it was in wait_for_message (delivered within 1 ms). No message was dropped.
- SendMessage schema. 11 of 53 SendMessage calls (21%) failed with "'summary' is a required property" and had to be re-issued, one wasted turn each.
- Gate. 80 submits were refused in teams and 65 in independents. dev_eval refused seed 410,000 twice, so the seed guard works.
- Budget accounting is consistent. Metered spend equals the per-call recomputation, every agent ended between $0.080 and $0.084, and all ended by submit.
- Telemetry gap (follow-up, not fixed). `run_telemetry.py` reports every scorer slot as 0.0 busy for these runs. The main-run scorer's local `logs/_scorer/i-0132786452a94bb5b/jobs.jsonl` holds only its first 12 jobs (138 s), and `find_scorer_dir` matches it on 3 run ids. The per-sample checker metadata is complete, and all queue numbers above come from it.
- Final-workspace candidates sometimes differ from what the agent says it submitted. Examples are spd r1 div-i0 ("published as agent_0-2", but a different sha scoring 2.96 against 37.79) and sparseig r1 team agent_1 (0.54). Two of these final files are invalid (spd r2 div-i2, sparseig r2 div-i2). Selection was unaffected because the registry copies rank above them.

## Is $0.10 of Haiku enough for coordination to pay off?

- Turns. 43-103 model calls per team agent (median 60.5) and 19-98 per independent (median 54), at $0.00126 and $0.00145 per call. 27 (team) and 20 (independent) calls, as medians, came before the agent's last publish.
- Timing. The lead finding reached teammates at bf 0.32-0.63 with 17-73 calls left (median about 40). The calls were available, but recipients spent them on stopping and padding (15 publishes from 18 recipients; 2 of 6 teams built on the lead).
- Overhead. Information exchange (send, registry, peer reads) cost 5.7% of team spend, against 1.3-3.0% in haiku2 dst and 7.6-23.3% for Opus in pilot 5. Most of the extra wait_for_message spend (5.8%) replaces check_budget padding rather than displacing solver work, since gate plus wait is 26.1% in teams and 25.6% in independents.
- Productive window. Haiku produces two or three versions of one idea in about 20-27 calls and then stops. Communication can redirect that small amount of work but does not lengthen it. A larger dollar budget would mostly buy more padding unless the gate or the prompt changes what agents do after their first stop, which matches haiku2. Whether more turns would let teams refine a shared lead (the pattern that beat teams in spd r1 and sparseig r2) is not tested here.

## Limits

- One team run per group and six groups, so these are observations, not effect sizes.
- Families were hand-classified from candidate notes. Unpublished experiments (for example sparseig r1 agent_1's block shift-invert at 0.52x) appear only in messages and are not counted.
- "Own work" includes some padding bash calls. Productive spend is overstated in both arms by an unknown amount.
- Only the three shortlisted sources per portfolio have hidden scores. Lost candidates (for example sinkhorn r1 agent_1-2) cannot be valued.
