# Swarm scaling at matched compute

When does a team of communicating AI agents beat one agent given the same total budget, and beat the same number of agents working independently? Does mixing model families change the answer?

This is an independent study by Jack Payne, following Toby Ord's [Swarm Scaling](https://www.lesswrong.com/posts/6cb7qd3RSkgnviCpf/swarm-scaling) (LessWrong, Sept 2026). Ord defines λ: N agents achieve what one agent achieves with N^λ times the budget. Ord's estimates are read from OpenAI launch charts of same-model teams, and the other published comparisons (lab system cards, Park et al. 2026) also use one model family per team. This project measures λ directly at matched output-token budgets on two task types, one where agents can check their progress mid-run (AlgoTune) and one where they cannot (Harvey LAB legal data rooms), and adds teams that mix model families.

Status: harness built and verified, pilot runs under way. No results on the main question yet.

## What exists so far

- **Replication of Ord's λ** from the raw chart data behind OpenAI's GPT-5.6 Sol launch post ([research/REPLICATION.md](research/REPLICATION.md)). Point estimates match his (0.65 BrowseComp, 0.57 SEC-Bench Pro, 0.48 Terminal-Bench). His BrowseComp interval does not reproduce (ours is 0.45-0.85), and on BrowseComp λ is not constant across team sizes (0.82 from 1 to 4 agents, 0.47 from 4 to 16), which the constant-λ model assumes away.
- **A multi-agent harness** on [Inspect](https://inspect.aisi.org.uk/) and [inspect_harbor](https://meridianlabs-ai.github.io/inspect_harbor/): N concurrent agents per sample, flat peers (no designated lead), real-time messaging delivered into each recipient's context, a shared candidate registry, hard per-agent output-token budgets, mixed model families in one team, and per-run CPU-contention telemetry. Every arm can checkpoint candidates, so teams do not get a larger pick pool than solo agents.
- **Task setup**: AlgoTune with a dev toolkit so agents can verify progress mid-run, a pinned dataset digest and a secret verifier seed offset (agents hold the problem generator, so the public seeds would allow lookup tables); the 11 Harvey LAB diligence data rooms (2,600-4,060 documents each) converted to local Harbor tasks with a host-side judge, so agents never have network access to the public rubrics.
- **Pilot**: a single Gemini 3.8 Flash agent solved an AlgoTune task at a 22.9x speedup within a 60k output-token budget. Measured cost and throughput are in [PLAN.md](PLAN.md).

## Layout

| Path | Contents |
|---|---|
| [PLAN.md](PLAN.md) | Question, arms, models, analysis plan, budget, decisions log |
| [HARNESS.md](HARNESS.md) | Harness design, per-task setup, known pitfalls |
| [research/](research/) | Evidence review, task assessments, protocol wording, λ replication, an external review of the plan |
| [src/swarm_scaling/](src/swarm_scaling/) | Solver (`swarm.py`, `team.py`), tasks, AlgoTune toolkit, Harvey grader, runner |
| [scripts/](scripts/) | Container checks, timing-noise and CPU reports, model smoke test |
| [tests/](tests/) | Unit tests (`uv run --group dev pytest`) |

## Running

Requires Docker, [uv](https://docs.astral.sh/uv/) and provider API keys.

```bash
uv sync
uv run --group dev pytest tests/ -q
# one arm = one Inspect eval, logs under logs/<name>/
uv run python -m swarm_scaling.runner --arm solo --models google/gemini-3.8-flash \
    --budget 60000 --sample algotune/cvar-projection --name my-run
```

On cloud VMs (one sample per VM, GCP `swarm-scaling-jp` and AWS `us-east-1`, or `ap-southeast-2` with `AWS_REGION`): `scripts/cloud/build_image.sh publish` builds the task image once and pins its digest, then `build_image.sh <gcp|aws> [ref]` builds each cloud's VM image around it; `scripts/cloud/run_sample.sh <name> <machine-type> <ref> -- <runner args>` runs one invocation on a fresh VM, copies `logs/<name>/` back and deletes the VM; `uv run python scripts/cloud/fleet.py plan|status|cleanup|scorer` runs a plan across both clouds within their vCPU quotas, shows live VMs with estimated cost, deletes stray VMs, and starts or stops the remote scorer VM (`scripts/cloud/scorer.sh`) that `--checker remote` runs use. Usage is in each script's header. The ref must be on GitHub.

Third-party source texts used during the review (system cards, posts, transcripts) are not redistributed; [research/sources/README.md](research/sources/README.md) links to the originals.
