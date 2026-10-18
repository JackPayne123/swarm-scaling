# Swarm scaling at matched compute

Does a team of communicating AI agents beat the same number of agents working independently, at the same dollar budget and with the same final selection rule? Does it beat one agent given the whole budget, and does mixing model families change the answer?

This is an independent study by Jack Payne, following Toby Ord's [Swarm Scaling](https://www.lesswrong.com/posts/6cb7qd3RSkgnviCpf/swarm-scaling) (LessWrong, Sept 2026). Ord defines λ: N agents achieve what one agent achieves with N^λ times the budget. This project runs controlled comparisons on AlgoTune code-optimisation tasks: a team of 4 agents that can message each other and share files, against 4 independent agents whose best candidate is taken by the same selection rule. Both arms get the same per-agent dollar budget, the same exploration prompt (the team's adds a paragraph on sharing), identical CPUs, and scoring on a separate machine against hidden instances.

**Status (2026-10-19).** Claude Haiku 5.5 teams showed no clear advantage over independents (0.97x over 14 groups). Claude Opus 5.5 teams beat the best of 4 independents on both tasks run on the final harness (1.21x, 1.12x) and on an earlier pilot task re-scored after the fixes (1.74x). Three groups are not enough to establish an effect: the 95% interval is 0.74x to 2.38x. Results, caveats and record paths: [analysis/v7-results.md](analysis/v7-results.md). Design and decisions log: [PLAN.md](PLAN.md) (v7 sections).

## What exists so far

- **Replication of Ord's λ** from the raw chart data behind OpenAI's GPT-5.6 Sol launch post ([research/REPLICATION.md](research/REPLICATION.md)). Point estimates match his (0.65 BrowseComp, 0.57 SEC-Bench Pro, 0.48 Terminal-Bench). His BrowseComp interval does not reproduce (ours is 0.45-0.85), and on BrowseComp λ is not constant across team sizes (0.82 from 1 to 4 agents, 0.47 from 4 to 16), which the constant-λ model assumes away.
- **A multi-agent harness** on [Inspect](https://inspect.aisi.org.uk/) and [inspect_harbor](https://meridianlabs-ai.github.io/inspect_harbor/): N concurrent agents per sample, flat peers (no designated lead), messaging and file sharing between agents, a shared candidate registry with publish notices, per-agent dollar budgets with runs that end when the budget is spent, isolated per-agent boxes with identical CPUs and memory, mixed model families in one team, a remote scorer that does all timing on dedicated cores, and a two-stage selection rule applied identically to every arm. Every arm can checkpoint candidates, so teams do not get a larger pick pool than solo agents.
- **Task setup**: AlgoTune with a dev toolkit so agents can verify progress mid-run, a pinned dataset digest and a secret verifier seed offset (agents hold the problem generator, so the public seeds would allow lookup tables); the 11 Harvey LAB diligence data rooms (2,600-4,060 documents each) converted to local Harbor tasks with a host-side judge, so agents never have network access to the public rubrics.
- **Runs so far**: pilots on Opus 5.5 and Haiku 5.5, the Haiku v6 main run (126 runs, 168 agents; [analysis/v6-main-results.md](analysis/v6-main-results.md)), the Opus calibration ([analysis/v7-calibration.md](analysis/v7-calibration.md)), the harness fixes it led to ([analysis/v7b-fixes.md](analysis/v7b-fixes.md)) and Opus batch 1 ([analysis/v7-results.md](analysis/v7-results.md)). Selection records are in `analysis/portfolio/`.

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
