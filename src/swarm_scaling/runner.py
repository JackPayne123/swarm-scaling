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
      --budget 300000 --sample algotune/cvar-projection --name pilot-solo-glm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from inspect_ai import eval

from swarm_scaling.algotune_devkit import algotune_finalize
from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import LOG_DIR, algotune_task

ALGOTUNE_RULE = "the fastest correct candidate on the dev inputs (dev_eval.py)"
ALGOTUNE_DELIVERABLE = (
    "a file named solver.py defining class Solver with a solve method, as the task describes "
    "(only files named solver.py are considered, in a published candidate or your working directory)"
)
ARMS = ("solo", "independent", "team", "registry")


def arm_config(arm: str, models: list[str], n: int, budget: int, mult: int) -> dict:
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--models", required=True, help="comma-separated; a mixed team cycles through them")
    p.add_argument("--n", type=int, default=1, help="agents in one container (team/registry arms)")
    p.add_argument("--budget", type=int, required=True, help="base per-agent token budget b")
    p.add_argument("--mult", type=int, default=1, help="solo arm: budget multiplier (1, 2, 4, 8)")
    p.add_argument("--split", default="pilot", choices=("pilot", "heldout"))
    p.add_argument("--sample", action="append", help="sample id(s); default: the whole split")
    p.add_argument("--epochs", type=int, default=1, help="repeats per sample")
    p.add_argument("--time-limit", type=int, default=3600, help="per-agent wall-clock seconds")
    p.add_argument("--name", required=True, help="run name; logs go to logs/<name>/")
    p.add_argument("--budget-type", default="output", help='what --budget meters: "output" (default) or "all"')
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--request-timeout", type=int, default=900, help="seconds per model request")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    config = arm_config(args.arm, models, args.n, args.budget, args.mult)
    metadata = {
        "arm": args.arm,
        "n": args.n,
        "models": config["models"],
        "budget_b": args.budget,
        "mult": args.mult,
        "per_agent_tokens": config["per_agent_tokens"],
        "budget_type": args.budget_type,
        "messaging": config["messaging"],
        "registry": config["registry"],
        "family": "algotune",
        "split": args.split,
        "protocol": "loose",
        "time_limit": args.time_limit,
        "launched_at": time.time(),
    }
    log_dir = LOG_DIR / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.json").write_text(json.dumps({**metadata, "argv": vars(args)}, indent=2))

    logs = eval(
        algotune_task(split=args.split),
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
        ),
        model=config["models"][0],
        sample_id=args.sample,
        epochs=args.epochs,
        log_dir=str(log_dir),
        tags=[args.arm, f"n{args.n}", f"mult{args.mult}"],
        metadata=metadata,
        max_retries=args.max_retries,
        timeout=args.request_timeout,
        max_samples=1,
        max_sandboxes=1,
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
