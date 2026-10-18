"""Run one experiment arm as one Inspect eval, tagged so analysis can group the logs.

Arms (PLAN.md):
  solo         one agent at budget mult x b (duration scaling); mult in {1, 2, 4, 8}
  independent  one agent at budget b, repeated (--epochs) and resampled into teams afterwards
  team         N agents in one container (one each with --isolate-agents) at budget b each, messaging + shared registry
  registry     N agents, shared registry but no messaging (ablation)

Every arm can save candidates (registry=True); only team/registry arms share them.

Usage (keys from scripts/env.sh):
  source scripts/env.sh
  uv run python -m swarm_scaling.runner --arm solo --models openai-api/zai/glm-5.3-flash \\
      --budget 5 --sample algotune/cvar-projection --name pilot-solo-glm   # $5 per agent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path

from inspect_ai import eval

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import (
    DEV_SEED_CAP, DISJOINT_DEV_SEED_CAP, SOLVER_PATH, algotune_agent_tools, algotune_finalize, known_dev_score,
    scorer_hardware_text,
)
from swarm_scaling.swarm import ELICITS, PRIORITIES, swarm
from swarm_scaling.tasks import EXPERIMENTS, LOG_DIR, algotune_task

ALGOTUNE_RULE = "the fastest correct candidate on the dev inputs (measured as the dev_eval tool measures)"
ALGOTUNE_DELIVERABLE = (
    "a file named solver.py defining class Solver with a solve method, as the task describes "
    "(a published file is saved as solver.py; in a published folder or your working directory, "
    "only a file named solver.py is considered)"
)
ARMS = ("solo", "independent", "team", "registry")
# A cost budget above this per agent is refused: a token-sized number (e.g. 2000000) passed without
# --budget-type all would otherwise run as dollars.
MAX_COST_PER_AGENT = 100.0


def arm_config(arm: str, models: list[str], n: int, budget: float, mult: int) -> dict:
    """Swarm settings for one arm; validates the combination."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if arm in ("solo", "independent"):
        if n != 1 or len(models) != 1:
            raise ValueError(f"{arm} runs one agent; use --epochs for repeats")
        if arm == "independent" and mult != 1:
            raise ValueError("independent runs use budget b (mult 1); duration scaling is the solo arm")
        return {"models": models, "per_agent_tokens": budget * mult, "messaging": False, "registry": True}
    if mult != 1:
        raise ValueError("team arms use budget b per agent (mult 1)")
    team = (models * n)[:n] if len(models) < n else models[:n]
    if len(models) > 1 and n % len(models) != 0:
        raise ValueError(f"a mixed team of {n} cannot split evenly across {len(models)} models")
    return {"models": team, "per_agent_tokens": budget, "messaging": arm == "team", "registry": True}


def wait_for_docker_images(timeout_s: int = 300) -> None:
    """Block until Docker lists the prebuilt AlgoTune images (hb__*).

    Docker Desktop's Resource Saver stops the engine after idle minutes. A run that wakes it can reach
    `compose up` before the image store is back, and compose then tries to pull the local-only image:
    "pull access denied for hb__...". That killed two runs on 2026-10-08 (pilot2-team2-cvar, a smoke run).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        out = subprocess.run(
            ["docker", "image", "ls", "-q", "--filter", "reference=hb__*"], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"Docker did not list any hb__* image within {timeout_s}s: {out.stderr.strip()}")
        time.sleep(3)


def effort_by_model(spec: str | None, models: list[str]) -> dict[str, str] | None:
    """--reasoning-effort as {model: effort}: one effort for every model, or `model=effort,...` (a mixed team, 2026-10-16:
    Opus at xhigh, GPT-6.1 Sol at high). Every model of the run must get one when the per-model form is used."""
    if not spec:
        return None
    if "=" not in spec:
        return {m: spec for m in models}
    efforts = dict(part.split("=", 1) for part in spec.split(","))
    if missing := [m for m in models if m not in efforts]:
        raise SystemExit(f"--reasoning-effort gives no effort for {missing}")
    return {m: efforts[m] for m in models}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--models", required=True, help="comma-separated; a mixed team cycles through them")
    p.add_argument("--n", type=int, default=1, help="agents in one container (team/registry arms)")
    p.add_argument("--budget", type=float, required=True,
                   help="base per-agent budget b: US dollars with --budget-type cost (default), else tokens")
    p.add_argument("--mult", type=int, default=1, help="solo arm: budget multiplier (1, 2, 4, 8)")
    p.add_argument("--split", default="pilot", choices=("pilot", "heldout", "v7"))
    p.add_argument("--sample", action="append", help="sample id(s); default: the whole split")
    p.add_argument("--epochs", type=int, default=1, help="repeats per sample")
    p.add_argument("--time-limit", type=int, default=57600,
                   help="per-agent wall-clock seconds (default 16 h: a backstop; agents end by budget)")
    p.add_argument("--name", required=True, help="run name; logs go to logs/<name>/")
    p.add_argument("--budget-type", default="cost",
                   help='what --budget meters: "cost" (default: dollars at swarm_scaling.prices), '
                        '"all" (input incl. cached + output tokens) or "output"')
    p.add_argument("--reasoning-effort", default=None,
                   help="explicit reasoning effort for every model (e.g. high), or per model: model=effort,model=effort")
    p.add_argument("--tool-style", default="default", choices=("default", "claude_code"),
                   help="claude_code: SendMessage + shared task list instead of send_message (team arm only)")
    p.add_argument("--parallel", type=int, default=1,
                   help="samples (incl. epochs) run at once; timed checker runs stay one at a time, agent boxes get less memory")
    p.add_argument("--checker", default="local", choices=("local", "remote"),
                   help="remote: every timed run goes to the scorer service at SCORER_URL (token SCORER_TOKEN)")
    p.add_argument("--cpus-per-agent", type=int, default=4,
                   help="agent box CPUs = this x agents in the sample; the checker has 8 more (the Docker host must have both)")
    p.add_argument("--isolate-agents", action="store_true",
                   help="v7: one box per agent (team agents too), each with --cpus-per-agent CPUs on its own cpuset and "
                        "--agent-memory-mb, sharing the registry and working directories on volumes")
    p.add_argument("--agent-memory-mb", type=int, default=None,
                   help="with --isolate-agents: memory of every agent box; give every row of a comparison the same value")
    p.add_argument("--min-spend-frac", type=float, default=None,
                   help="submit is refused until an agent has used this fraction of its own budget; 0 = stop freely "
                        "(default 0.8; 0 with --end-on-budget)")
    p.add_argument("--end-on-budget", action="store_true",
                   help="v7 run to budget: no submit tool and no submit gate; an agent ends when its budget is used "
                        "up or at the time limit, and teammates get no finish notices")
    p.add_argument("--elicit", default="loose", choices=ELICITS,
                   help="diversified: the protocol plus the v6 exploration block and an initial method priority "
                        "(team agent i gets priority i; teams also get the sharing instruction); diversified-v7: "
                        "that plus the v7 persistence block, and for teams the v7 team block (needs --end-on-budget)")
    p.add_argument("--publish-notices", action=argparse.BooleanOptionalAction, default=None,
                   help="team arm: tell every teammate about each publish in its next tool result "
                        "(default: on for --arm team --elicit diversified-v7, else off)")
    p.add_argument("--tool-fixes", action="store_true",
                   help="v7: SendMessage's summary is optional; check_budget shows 4 decimals and the percent remaining")
    p.add_argument("--priority-index", type=int, default=None,
                   help=f"diversified independent runs: which initial priority (0-{len(PRIORITIES) - 1}) this agent gets")
    p.add_argument("--blind-frac", type=float, default=0.0,
                   help="team arm, two-phase: each agent works blind until it has spent this fraction of its budget, "
                        "then its team channel opens (0 = open throughout)")
    p.add_argument("--disjoint-dev-seeds", action="store_true",
                   help="v7: dev_eval needs seed + n below 50,000, off the finalize (stage-1) instances, so every seed "
                        "range is disjoint (algotune_devkit.seed_ranges); default: below 400,000 as in v6")
    p.add_argument("--experiment", default="v6", choices=EXPERIMENTS,
                   help="whose secret hidden-seed offset scores the run (one per experiment; a remote scorer must serve it)")
    p.add_argument("--primary-score", default=None, choices=("interleaved", "alone"),
                   help="the run's Score.value: AlgoTune's interleaved speedup or the speedup against the reference "
                        "timed alone (default: alone for --experiment v7, else interleaved)")
    p.add_argument("--scorer-hardware", action=argparse.BooleanOptionalAction, default=None,
                   help="state the scoring machine's CPU, ISA extensions and CPUs per timed run in every agent's prompt, "
                        "from the scorer's /health at run start (default: on for --experiment v7 with --checker remote)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--request-timeout", type=int, default=900, help="seconds per model request")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.budget_type != "cost" and not args.budget.is_integer():
        p.error("a token budget must be a whole number")
    budget = args.budget if args.budget_type == "cost" else int(args.budget)
    config = arm_config(args.arm, models, args.n, budget, args.mult)
    if args.budget_type == "cost" and config["per_agent_tokens"] > MAX_COST_PER_AGENT:
        p.error(
            f"cost budget ${config['per_agent_tokens']:,.2f} per agent is above ${MAX_COST_PER_AGENT:.0f}; "
            "for a token budget pass --budget-type all"
        )
    if args.priority_index is not None and (args.elicit == "loose" or args.arm != "independent"):
        p.error("--priority-index is for --arm independent with a diversified --elicit (team agent i gets priority i)")
    if args.arm == "independent" and args.elicit != "loose" and args.priority_index is None:
        p.error("a diversified independent run needs --priority-index")
    if args.blind_frac and args.arm != "team":
        p.error("--blind-frac delays a team's channel; it is for --arm team")
    if args.end_on_budget and args.min_spend_frac:
        p.error("--end-on-budget has no submit tool, so no submit gate: drop --min-spend-frac")
    min_spend_frac = 0.0 if args.end_on_budget else (0.8 if args.min_spend_frac is None else args.min_spend_frac)
    if (args.agent_memory_mb is not None) != args.isolate_agents:
        p.error("--isolate-agents needs --agent-memory-mb (the same in every row), and only it takes one")
    if args.elicit == "diversified-v7" and not args.end_on_budget:
        p.error("--elicit diversified-v7 tells agents the run ends when the budget is used up: it needs --end-on-budget")
    publish_notices = args.publish_notices
    if publish_notices is None:
        publish_notices = args.arm == "team" and args.elicit == "diversified-v7"
    if publish_notices and args.arm != "team":
        p.error("--publish-notices tells teammates about each publish: it is for --arm team")
    scorer_hardware = args.scorer_hardware
    if scorer_hardware is None:
        scorer_hardware = args.experiment == "v7" and args.checker == "remote"
    if scorer_hardware and args.checker != "remote":
        p.error("--scorer-hardware describes the remote scorer: it needs --checker remote")
    # the same /health fields give every row on one scorer the same text, whatever its arm
    hardware = scorer_hardware_text(scorer_client.health()) if scorer_hardware else None
    metadata = {
        "arm": args.arm,
        "n": args.n,
        "models": config["models"],
        "budget_b": budget,
        "mult": args.mult,
        "per_agent_tokens": config["per_agent_tokens"],
        "budget_type": args.budget_type,
        "messaging": config["messaging"],
        "registry": config["registry"],
        "family": "algotune",
        "split": args.split,
        "protocol": args.elicit,  # the prompt variant (--elicit); "loose" before v6
        "priority_index": args.priority_index,
        "blind_frac": args.blind_frac,
        "tool_style": args.tool_style,
        "reasoning_effort": args.reasoning_effort,
        "time_limit": args.time_limit,
        "min_spend_frac": min_spend_frac,
        "end_on_budget": args.end_on_budget,
        "publish_notices": publish_notices,
        "tool_fixes": args.tool_fixes,
        "scorer_hardware": hardware,  # the prompt's timing-hardware paragraph (None: not stated)
        "parallel": args.parallel,
        "cpus_per_agent": args.cpus_per_agent,
        "isolate_agents": args.isolate_agents,
        "agent_memory_mb": args.agent_memory_mb,
        "checker": args.checker,  # cpusets and memory limits: algotune_task adds them to the log metadata
        "dev_seed_cap": DISJOINT_DEV_SEED_CAP if args.disjoint_dev_seeds else DEV_SEED_CAP,
        # experiment and primary_score: algotune_task records them in the log metadata
        "launched_at": time.time(),
    }
    log_dir = LOG_DIR / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.json").write_text(json.dumps({**metadata, "argv": vars(args)}, indent=2))

    os.environ["SWARM_RUN_ID"] = args.name  # labels remote scorer jobs
    wait_for_docker_images()
    logs = eval(
        algotune_task(split=args.split, cpus_per_agent=args.cpus_per_agent, n_agents=len(config["models"]),
                      parallel=args.parallel, checker_backend=args.checker, isolate_agents=args.isolate_agents,
                      memory_per_agent_mb=args.agent_memory_mb, experiment=args.experiment,
                      primary_score=args.primary_score),
        solver=swarm(
            models=config["models"],
            per_agent_tokens=config["per_agent_tokens"],
            messaging=config["messaging"],
            registry=config["registry"],
            finalize=algotune_finalize,
            selection_rule=ALGOTUNE_RULE,
            time_limit=args.time_limit,
            budget_type=args.budget_type,
            deliverable=ALGOTUNE_DELIVERABLE,
            tool_style=args.tool_style,
            agent_tools=partial(algotune_agent_tools, seed_cap=metadata["dev_seed_cap"]),
            final_path=SOLVER_PATH,
            candidate_file="solver.py",
            min_spend_frac=min_spend_frac,
            elicit=args.elicit,
            priority_index=args.priority_index,
            blind_frac=args.blind_frac,
            end_on_budget=args.end_on_budget,
            publish_notices=publish_notices,
            known_score=known_dev_score,
            tool_fixes=args.tool_fixes,
            reasoning_effort=effort_by_model(args.reasoning_effort, config["models"]),
            hardware=hardware,
        ),
        model=config["models"][0],
        sample_id=args.sample,
        epochs=args.epochs,
        log_dir=str(log_dir),
        tags=[args.arm, f"n{args.n}", f"mult{args.mult}"],
        metadata=metadata,
        max_retries=args.max_retries,
        timeout=args.request_timeout,
        max_samples=args.parallel,
        max_sandboxes=args.parallel,
        display="none",
    )
    for log in logs:
        print(f"status={log.status} log={log.location}")
        for s in log.samples or []:
            sw = (s.metadata or {}).get("swarm", {})
            score = {k: v.value for k, v in (s.scores or {}).items()}
            tokens = {a: r.get("tokens", {}).get("metered") for a, r in sw.get("agents", {}).items()}
            print(f"  {s.id} epoch={s.epoch} score={score} tokens={tokens} cpu_flag={sw.get('cpu', {}).get('flag')}")


if __name__ == "__main__":
    main()
