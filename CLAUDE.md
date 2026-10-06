# Swarm Scaling

Jack's independent one-week research project (not Lyptus): matched-compute swarm scaling (Toby Ord's λ), communication vs independent agents, mixed-model teams. Output is a LessWrong post.

## Read first

- PLAN.md - question, design, arms, analysis, week plan, decisions log, open questions. Source of truth; update it when a decision changes.
- HARNESS.md - Inspect + inspect_harbor + custom multi-agent solver; per-family setup notes and day-1 checks.
- research/EVIDENCE.md - public swarm data, papers, incidents, quotes.
- research/TASKS.md - task families chosen, phase 2, rejected (with reasons).
- research/second-opinion/ - GPT-6.1 Sol's review of the plan (2026-10-05).

## Rules for this project

- Never guess costs or token counts. Measure in the pilot; write the measured value and date into PLAN.md.
- Freeze prompts, budgets, task split, selector and analysis before the main run. Log any post-freeze change in PLAN.md's decisions log with the reason.
- Pilot tasks and held-out evaluation tasks never overlap.
- Resample by task for every CI. Repeats are nested within tasks.
- The same final selector applies to every arm. Selection on held-out scores is an oracle and must be labelled as one.
- Separate what was run from what was verified in every writeup. Figure-read numbers from system cards are approximate (±0.01).
- Use `uv` for Python. No Haiku anywhere (judges, agents, pilots).
- Never write or run code that kills processes unless it is provably inside a container (positive check). `swarm.kill_agent_processes` SIGKILLed every user process on Jack's Mac twice on 2026-10-05: Inspect's `sandbox()` returns a `SandboxEnvironmentProxy`, so the local-sandbox isinstance guard never matched, and macOS `ps` has no `etimes`, so every process qualified. It is disabled; see the comment in swarm.py. Inspect's local sandbox runs real shell commands on the host, so treat local-sandbox tests as host execution.
- Docker Desktop: you may start it (`open -g -a Docker`) only if it is not running, then wait with a bounded `docker info` loop. Never start it twice, and never stop, restart or kill it. The `guard-docker-desktop.sh` hook enforces this.
- Run at most one Docker build or one sandboxed eval at a time on the Mac unless Jack says otherwise.
- Judges must not share a model family with the agents being compared (self-preference bias).
- Do not cite Ord's "1,200 agents / 700 attacked" Hugging Face figures; they are not in OpenAI's report.
