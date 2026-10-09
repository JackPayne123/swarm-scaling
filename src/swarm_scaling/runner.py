"""Run one experiment arm as one Inspect eval, tagged so analysis can group the logs.

Arms (PLAN.md):
  solo         one agent at budget mult x b (duration scaling); mult in {1, 2, 4, 8}
  independent  one agent at budget b, repeated (--epochs) and resampled into teams afterwards
  team         N agents in one container at budget b each, messaging + shared registry
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
from pathlib import Path

from inspect_ai import eval

from swarm_scaling.algotune_devkit import SOLVER_PATH, algotune_agent_tools, algotune_finalize
from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import LOG_DIR, algotune_task

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--models", required=True, help="comma-separated; a mixed team cycles through them")
    p.add_argument("--n", type=int, default=1, help="agents in one container (team/registry arms)")
    p.add_argument("--budget", type=float, required=True,
                   help="base per-agent budget b: US dollars with --budget-type cost (default), else tokens")
    p.add_argument("--mult", type=int, default=1, help="solo arm: budget multiplier (1, 2, 4, 8)")
    p.add_argument("--split", default="pilot", choices=("pilot", "heldout"))
    p.add_argument("--sample", action="append", help="sample id(s); default: the whole split")
    p.add_argument("--epochs", type=int, default=1, help="repeats per sample")
    p.add_argument("--time-limit", type=int, default=3600, help="per-agent wall-clock seconds")
    p.add_argument("--name", required=True, help="run name; logs go to logs/<name>/")
    p.add_argument("--budget-type", default="cost",
                   help='what --budget meters: "cost" (default: dollars at swarm_scaling.prices), '
                        '"all" (input incl. cached + output tokens) or "output"')
    p.add_argument("--reasoning-effort", default=None, help="explicit reasoning effort for every model (e.g. high)")
    p.add_argument("--tool-style", default="default", choices=("default", "claude_code"),
                   help="claude_code: SendMessage + shared task list instead of send_message (team arm only)")
    p.add_argument("--parallel", type=int, default=1,
                   help="samples (incl. epochs) run at once; timed checker runs stay one at a time, agent boxes get less memory")
    p.add_argument("--checker", default="local", choices=("local", "remote"),
                   help="remote: every timed run goes to the scorer service at SCORER_URL (token SCORER_TOKEN)")
    p.add_argument("--cpus-per-agent", type=int, default=4,
                   help="agent box CPUs = this x agents in the sample; the checker has 8 more (the Docker host must have both)")
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
        "protocol": "loose",
        "tool_style": args.tool_style,
        "reasoning_effort": args.reasoning_effort,
        "time_limit": args.time_limit,
        "parallel": args.parallel,
        "cpus_per_agent": args.cpus_per_agent,
        "checker": args.checker,  # cpusets and memory limits: algotune_task adds them to the log metadata
        "launched_at": time.time(),
    }
    log_dir = LOG_DIR / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.json").write_text(json.dumps({**metadata, "argv": vars(args)}, indent=2))

    os.environ["SWARM_RUN_ID"] = args.name  # labels remote scorer jobs
    wait_for_docker_images()
    logs = eval(
        algotune_task(split=args.split, cpus_per_agent=args.cpus_per_agent, n_agents=len(config["models"]),
                      parallel=args.parallel, checker_backend=args.checker),
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
            agent_tools=algotune_agent_tools,
            final_path=SOLVER_PATH,
            candidate_file="solver.py",
            reasoning_effort={m: args.reasoning_effort for m in config["models"]} if args.reasoning_effort else None,
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
