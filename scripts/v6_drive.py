"""The v6 driver's two steps (scripts/v6.sh runs them around scorer.sh up/down).

  size <plan>    prints "<slots> <machine> <max_agents>": the scorer for the plan, read against the AWS quota now
  run <plan> <SPEND_CAP> --ref R --scorer URL --max-agents M
                 runs the whole plan through fleet.py (rows start as the quota allows), runs select_portfolio.py on each
                 task-repeat group as soon as its runs have ended, and stops starting rows near the spend cap

Scorer size: the plan's agents use about 4 vCPUs each (agent VMs at 2 CPUs per agent + 2). The scorer gets the
smallest machine (then every slot that machine holds, at most MAX_SLOTS) with at least one slot per AGENTS_PER_SLOT
agents that can run alongside it in the free quota; when even MAX_SLOTS is too few, the agents running at once are
capped at MAX_SLOTS x AGENTS_PER_SLOT instead. max_agents is always slots x AGENTS_PER_SLOT.

Spend cap: once measured spend of ended runs (run_cost.py's pricing) + the full budget of every running agent + the
budget of the largest row would exceed SPEND_CAP, no further row starts (fleet.py --stop-file); running rows finish.

Groups: rows named <prefix>-<task>-r<k>-<arm>[-i<j>] (plans/v6-*.tsv); a group is <prefix>-<task>-r<k>, so its
portfolios (one per arm) are selected together. Selections run one at a time with --concurrency 3, so live agents
keep most scorer slots.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/cloud"), str(ROOT / "scripts")]
import fleet  # noqa: E402
from run_cost import call_costs  # noqa: E402

MAX_SLOTS = 11  # c7a.24xlarge: 96 vCPUs = 11 slots x 8 + 8 for the service
AGENTS_PER_SLOT = 8
POLL_S = 60
SELECT_CONCURRENCY = 3


def arg(row: fleet.Row, flag: str, default: float) -> float:
    a = row.args
    return float(a[a.index(flag) + 1]) if flag in a else default


def size(rows: list[fleet.Row], free: int) -> tuple[int, str, int]:
    """(slots, machine, max_agents) for the plan's agents in `free` AWS vCPUs (see the module docstring)."""
    total = sum(fleet.agents(r) for r in rows)
    per_agent = max(fleet.vcpus(fleet.machine_for(r, "aws")) / fleet.agents(r) for r in rows)
    for want in range(1, MAX_SLOTS + 1):
        machine = fleet.scorer_machine(want)
        slots = min(MAX_SLOTS, (fleet.vcpus(machine) - fleet.SCORER_SERVICE_VCPUS) // fleet.SLOT_VCPUS)
        concurrent = min(total, int((free - fleet.vcpus(machine)) // per_agent))
        if slots >= math.ceil(concurrent / AGENTS_PER_SLOT):
            break
    if free - fleet.vcpus(machine) < max(fleet.vcpus(fleet.machine_for(r, "aws")) for r in rows):
        raise SystemExit(f"{free} free vCPUs leave no room for an agent VM beside a {machine} scorer")
    return slots, machine, slots * AGENTS_PER_SLOT


def group_of(name: str) -> str:
    return "-".join(name.split("-")[:3])


def state(name: str) -> str | None:
    """The last state in logs/<name>/cloud-status, None if the row has not started."""
    path = fleet.status_path(name)
    lines = path.read_text().split("\n") if path.exists() else []
    lines = [ln for ln in lines if ln.strip()]
    return lines[-1].split()[1] if lines else None


def run_cost(name: str) -> float:
    return sum(m["cost"] for f in sorted((ROOT / "logs" / name).rglob("*.eval")) for m in call_costs(f).values())


def run(args: argparse.Namespace) -> int:
    rows = fleet.parse_plan(Path(args.plan).read_text())
    budget = {r.name: fleet.agents(r) * arg(r, "--budget", 0.0) for r in rows}
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(group_of(r.name), []).append(r.name)
    stop = ROOT / "logs" / "_plans" / f"{Path(args.plan).stem}.stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    plan = subprocess.Popen([sys.executable, "-u", "scripts/cloud/fleet.py", "plan", args.plan, "--ref", args.ref,
                             "--max-agents", str(args.max_agents), "--stop-file", str(stop)], cwd=ROOT)  # fmt: skip
    costs: dict[str, float] = {}
    selected: set[str] = set()
    queue: list[str] = []
    selecting: tuple[str, subprocess.Popen] | None = None

    def select(group: str) -> subprocess.Popen | None:
        dirs = [f"logs/{n}" for n in groups[group] if (ROOT / "logs" / n).is_dir()]
        if not dirs:
            return None
        print(f"[v6] {time.strftime('%H:%M:%S')} selecting {group}", flush=True)
        return subprocess.Popen([sys.executable, "-u", "scripts/select_portfolio.py", *dirs, "--scorer", args.scorer,
                                 "--concurrency", str(SELECT_CONCURRENCY)], cwd=ROOT)  # fmt: skip

    def step(plan_done: bool) -> None:
        nonlocal selecting
        states = {r.name: state(r.name) for r in rows}
        for name, st in states.items():
            if st == "exit" and name not in costs:
                costs[name] = run_cost(name)
        if not stop.exists():
            running = sum(budget[n] for n, st in states.items() if st not in (None, "exit"))
            projected = sum(costs.values()) + running + max(budget.values())
            if projected > args.spend_cap:
                stop.touch()
                print(f"[v6] spend cap: measured ${sum(costs.values()):.2f} + running ${running:.2f} + next row "
                      f"> ${args.spend_cap:.2f}; no further rows start", flush=True)  # fmt: skip
        for g, names in groups.items():
            if g not in selected and (plan_done or all(states[n] == "exit" for n in names)):
                selected.add(g)
                queue.append(g)
        if selecting is not None and selecting[1].poll() is not None:
            selecting = None
        while selecting is None and queue:
            g = queue.pop(0)
            if (p := select(g)) is not None:
                selecting = (g, p)

    while plan.poll() is None:
        step(plan_done=False)
        time.sleep(POLL_S)
    step(plan_done=True)
    while selecting is not None or queue:
        time.sleep(POLL_S / 6)
        step(plan_done=True)
    print(f"[v6] measured spend of ended runs: ${sum(costs.values()):.2f} (cap ${args.spend_cap:.2f})", flush=True)
    return plan.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("size")
    ps.add_argument("plan")
    pr = sub.add_parser("run")
    pr.add_argument("plan")
    pr.add_argument("spend_cap", type=float)
    pr.add_argument("--ref", required=True)
    pr.add_argument("--scorer", required=True)
    pr.add_argument("--max-agents", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "size":
        free = fleet.aws_free()
        slots, machine, max_agents = size(fleet.parse_plan(Path(args.plan).read_text()), free)
        print(f"free AWS vCPUs now: {free}", file=sys.stderr)
        print(slots, machine, max_agents)
    else:
        if not os.environ.get("SCORER_TOKEN"):
            sys.exit("set SCORER_TOKEN")
        sys.exit(run(args))


if __name__ == "__main__":
    main()
