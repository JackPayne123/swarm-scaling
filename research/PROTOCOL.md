# Team protocol wording

**Status (2026-10-05, decision in PLAN.md):** the main arms use the loose, facts-only prompt in the next section. The structured protocol further down ("Structured variant") is a side check only: N = 4, pilot tasks, cheap model. Nothing here has been run on a model yet. The loose prompt is implemented as `swarm.default_protocol_prompt` (2026-10-05); render it there for the exact current text.

## Main-arm prompt (loose)

Communicating arm (`{rule}` is the task's selection rule, e.g. "the fastest correct candidate on the dev inputs"):

```text
You are {agent_id}, one of {n} agents working on this task at the same time. The other agents are {peer_ids}.

- send_message(to, text) sends a message to one agent or to "all". Messages sent to you appear after your next tool result. wait_for_message(timeout_s) waits for one.
- publish_candidate(path, note) adds a solution to a shared registry that every agent can read. list_candidates() lists it.
- Your private working directory is {private_dir}.

How you work together, if at all, is up to you.

When every agent has finished, one final answer is chosen from all published candidates and every agent's final working directory by a fixed rule: {rule}.

You have a budget of {budget} tokens. Messages you receive count against it.
```

Independent arm and single agent (separate N = 1 runs; same save-a-candidate mechanism as teams, so the pick pool does not differ by arm):

```text
You are working on this task alone.
- Your private working directory is {private_dir}.
- publish_candidate(path, note) saves a copy of a solution as a candidate; list_candidates() lists your saved candidates.

When you finish, the final answer is chosen from your saved candidates and your final working directory by a fixed rule: {rule}.

You have a budget of {budget} tokens.
```

Design notes: no capability claims (false in mixed teams), no roles, no message-content rules, no adoption bar, no anti-herding text. The one neutral sentence ("How you work together, if at all, is up to you.") states the absence of instructions so agents do not infer a hidden expected structure. Emergent structure is measured from logs (PLAN.md Analysis).

Sources read 2026-10-05: Park repo (commit 173dc13, `multiagent_protocols/*.py`, README, `agent.py`) and paper PDF; Opus 5.5 card s8.12.3-8.12.4 (it gives no verbatim tool descriptions or lead/helper prompts, only the semantics quoted below); OpenAI multi-agent guide (`ma.txt` and `ma2.txt` differ only in page navigation). Nothing proposed here has been run.

## Side by side

| | Park et al. | Anthropic (Opus 5.5 card) | OpenAI (Responses) | Ours now |
|---|---|---|---|---|
| Topology | Flat peers, slot ownership by atomic `mkdir slots/slot-$i` | Peers with identical tools and the full task. Lead on DRACO and large-team tasks, none on ProgramBench. "how the work is divided is left to the agents" | Root plus spawned subagents | Flat, no lead |
| Channel | Append-only `findings.log`, `disconfirmations.log`, `adoptions.log`, score log, `leaderboard/`. Pull, "at natural boundaries" | Send Message "inserted following the recipient's next tool result". Wait for Message "blocks sampling until an incoming message arrives" | `send_message` "without starting a new turn". `wait_agent` "Wait for an update in the calling agent's mailbox". Envelope `Message Type: MESSAGE \| FINAL_ANSWER / Task name / Sender / Payload` | Push at next tool result, wait tool, registry |
| Team framing | "{n} agents share this container and work the same task in parallel, all with this identical prompt. Search widely without herding, coordinate as you go, and keep improving until time runs out." | Not quoted. Agents told to check the time and plan | "All agents in the team ... are equally intelligent and capable, and have access to the same set of tools." | "one of N agents ... no leader" |
| Against independent collapse | Declare an approach in a slot and `cat` the others first. Log each attempt's score. "Every turn must run a command." "NEVER STOP WHILE TIME REMAINS." | Not described. The flat knowledge-base team had helpers "communicate relatively little with one another" | Nothing in the injected text. Brown says collapse is overcome by training | Nothing. "keep them short" discourages sending |
| Against herding | "Your job is not to agree early." Adopt only on a better measured outcome, replication, a blocked approach or finalisation. "Do not write persuasive prose that asks everyone to copy you." Disconfirm. "keep one meaningful variation." Plateau rule "pick a family no active peer is on" | None | None | None |

**Park's method.** The one-agent arm gets only the task instruction for ARC and polyomino. Only MNIST has a dedicated solo prompt (`SOLO_BODY`, with "do not herd to your OWN first idea") unless `--task-prompt-only`. Non-ARC continuation prompts (plateau means a structurally different approach) go to both arms. ARC best@k is exact from 64 solo trials, 1 - C(n-s,k)/C(n,k), per level. Polyomino compares the best run among 60 solo and 20 team@3 trials (12 and 2 at 72 h). MNIST best@4 is four solo runs against one team run. All use the test score, so best@k is an oracle selection. The paper has no prompt ablation. Its Appendix A.2 prints the `anti_plateau` text while the README defaults to `antiherding`, so which protocol produced which result is unverified. Team@k trails below roughly 400K output tokens per agent, and team@5 at 0.2x budget per agent loses to one agent.

**Other tested work (weaker evidence).** Cho et al. 2025 (arXiv 2505.21588) found the system prompt "Please be stubborn" left flip rate (0.55) and entropy (0.43) at baseline on MMLU-Pro, while peer presentation format and order moved herding (fixed-answer questions, one revision round). Chen et al. 2026 (2604.18005) found directive versus exploratory tone had no effect on diversity (F=1.90, p=0.172), dense topology and authority lowered it, and blind writing before discussion gave the highest early diversity. Zhu et al. (2410.12428) found one extra dissenting answer in context reduced conformity. Pappu et al. 2026 (2602.01011) found GEPA-tuned deference prompts did not close the gap. Wording alone has little support. Structure has more (independent first move, evidence-bearing records, deterministic tie-break, scores to adopt on).

Park's verifier condition applies here. With weak feedback on Terminal-Bench, team@2 (60.67%) did not beat pass@2 (62.36%). AlgoTune with the dev toolkit has dense feedback. Harvey has none, so expect no communication gain and nothing to adopt on.

## Risks for the experiment

1. Diversify, plateau and never-stop text changes solo scores. It goes into both arms word for word, and only peer-dependent text stays in the communicating arm.
2. The id tie-break makes approaches non-overlapping, a coordination gain that is part of the treatment. Separating it from shared evidence needs another arm (independent with pre-assigned approach seeds). Not planned.
3. Knowing peers exist may change effort (Brown). A control arm told peers exist but unreachable is cheap at N=4 on pilot tasks.
4. "Equally capable" is false in the mixed arm, so the text makes no capability claim in any arm.
5. "Submit only when no untried approach remains" departs from PLAN's "stopping early is part of the treatment". It applies to both arms, and realised spend is logged by arm.
6. Below the coordination-tax budget the prompt cannot rescue the communicating arm. Report that as a result.

## Structured variant (side check only)

Originally proposed as the default; demoted 2026-10-05. Used only for the N = 4 structured-vs-loose comparison on pilot tasks. Risks 1-6 above apply to this variant; the loose prompt avoids 1, 2, 4 and 5 by not including the corresponding text.

Shared block, both arms (`{budget_scope}` is " per agent" in teams):

```text
## Working protocol

You are {agent_id}. Your private working directory is {private_dir}. Do your work there.

- Put a correct solution in your directory early, then improve it. Measure every change with the dev toolkit and trust only those measurements.
- Keep notes.md in your directory with one line per attempt giving what you changed and the measured result. Read it before choosing your next change.
- If your best score has not improved over three consecutive attempts, stop tuning and switch to a structurally different approach, not a variant of the current one.
- Before you invest in an idea, run the cheapest test that could show it is wrong.
- You have a fixed token budget{budget_scope}. When it runs out you stop at once, so keep your best work in a usable state. Submit only when no untried approach you expect to help remains, or the budget is nearly used.
```

Independent arm adds only `You are {agent_id}, the only agent working on this task.` and `- The final answer is taken from your working directory by a fixed rule, so leave your best solution there.` It names no peers, registry or messages.

Communicating arm adds:

```text
You are one of {n} agents working on this task at the same time. Your teammates are {others}. No agent is in charge and none is assumed to be more capable. Judge an idea by its measured result, not by who proposed it. Do not modify other agents' directories.

- Before you start work, send_message to "all" one line naming the approach you will try first. If a teammate with a lower id has declared the same approach, choose a different one. Do not wait for replies.
- list_candidates() shows what teammates have published. Check it before you start a new approach and whenever your own progress stalls. Registry entries are read-only.
- Call publish_candidate(path, note) every time your dev score improves, not only at the end. In the note give the score, the command that produced it, and the id of any published candidate yours builds on (for example "based on agent_2-1").
- Send a message only when a teammate can act on it without replying, such as a measured result, an approach that failed and why, or a fault in the dev toolkit or a candidate. Do not send status updates.
- Switch to a teammate's approach only if its measured score beats yours or your approach is blocked. If you build on a teammate's candidate, change something it does not, such as an optimisation, a parameter or a fallback.
- Try to disprove the leading candidate and report what you find.
- Use wait_for_message only when you have nothing left to run.
- The final answer is chosen from the published candidates by a fixed rule. Unpublished work is not considered, so publish your best candidate before you finish.
```

Tool descriptions (replace the docstrings in `team.py`):

```text
send_message(to, text): Send a message to one teammate (for example "agent_2") or to "all". It is appended to the recipient's next tool result and does not interrupt them. Send results they can act on, not status updates. Each message uses part of the recipient's token budget.
  to: Recipient agent id, or "all".
  text: The message, one to three sentences.

wait_for_message(timeout_s): Pause until a teammate message arrives or timeout_s seconds pass. Teammates keep working while you wait, so use this only when you have nothing left to run. Returns at once if every teammate has finished or is also waiting.
  timeout_s: Maximum seconds to wait.

publish_candidate(path, note): Copy the file or directory at path into a new read-only registry entry that teammates can read and the final selection considers. Call it each time your dev score improves. Later edits to path do not change the entry.
  path: Absolute path.
  note: Dev score, the command that produced it, and the id of the candidate it builds on, if any.

list_candidates(): List every published candidate with its id, author and note. Check it before starting a new approach and whenever your own progress stalls.
```

"Dev toolkit" and "dev score" need per-family wording, fixed before freezing and identical across arms. Harvey has no score to measure or adopt on and needs its own text.

## Needs a model run

1. Uptake at N=4 and 8 on the pilot tasks. Count first-approach messages, id tie-break compliance, `list_candidates` calls and tool results carrying messages, for GLM-5.3-Flash and Luna.
2. Wording ablation at N=4. Communicating with full text against mechanics only, and independent with the shared block against the bare solo prompt. This sizes how much of C minus I is wording.
3. Whether the shared block shifts solo scores and realised spend.
4. Whether "uses part of the recipient's token budget" suppresses sending (omit it in one pilot cell).
5. The tie-break under simultaneous starts at N=8.
