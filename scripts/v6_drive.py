"""The v6 driver's two steps (scripts/v6.sh runs them around scorer.sh up/down).

  size <plan>    prints "<slots> <machine> <max_agents>": the scorer for the plan, read against the AWS quota now
  run <plan> <SPEND_CAP> --ref R --scorer URL --max-agents M [--select-after group|plan]
                 runs the whole plan through fleet.py (rows start as the quota allows), runs select_portfolio.py on the
                 task-repeat groups (below), and stops starting rows near the spend cap

Scorer size: the plan's agents use about 4 vCPUs each (agent VMs at 2 CPUs per agent + 2). The scorer gets the
smallest machine (then every slot that machine holds, at most MAX_SLOTS) with at least one slot per AGENTS_PER_SLOT
agents that can run alongside it in the free quota; when even MAX_SLOTS is too few, the agents running at once are
capped at MAX_SLOTS x AGENTS_PER_SLOT instead. max_agents is always slots x AGENTS_PER_SLOT.

Spend cap: once measured spend of ended runs (run_cost.py's pricing) + the full budget of every running agent + the
budget of the largest row would exceed SPEND_CAP, no further row starts (fleet.py --stop-file); running rows finish.

Groups: rows named <prefix>-<task>-r<k>-<arm>[-i<j>] (plans/v6-*.tsv); a group is <prefix>-<task>-r<k>, so its
portfolios (one per arm) are selected together. --select-after group (v6): each group's selection starts as soon as its
own runs have ended, one selection at a time with --concurrency 6 (main's scorer has 11 slots), so live agents keep
most scorer slots. --select-after plan (default for plans with a v7 row, 2026-10-18): no selection starts until every
row of the plan has ended, then one select_portfolio.py selects every group at once (portfolios concurrently, and with
v7's exclusive hidden scoring a group's hidden jobs next to each other, arms alternating): in the v7 calibration the
stable_matching selection ran beside agents still working, whose dev_eval waits rose to a mean of 221 s (max 835 s)
from 76 s.

v7 options: --group-start (fleet.py: a group's rows start together), --validity-n N (select_portfolio.py's validity
pass). Every PULL_S the scorer's logs are copied to logs/_scorer/<instance>/ (scorer.sh pull), so a connection lost at
teardown loses at most that much of jobs.jsonl.
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
PULL_S = 600  # scorer log copies (v6: the main scorer's jobs.jsonl was lost when the one copy at teardown failed)
SELECT_CONCURRENCY = 6  # 2026-10-12, after the pilot: at 3, main's 14 groups needed about 13 h of serial selection


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


group_of = fleet.group_of


def ready_groups(groups: dict[str, list[str]], states: dict[str, str | None], plan_done: bool, after_plan: bool) -> list[str]:
    """The groups whose selection may start now: each once its own rows have ended (state "exit"), or with after_plan
    every group, but only once the whole plan has ended, so no selection job shares the scorer with a running agent."""
    if after_plan:
        return list(groups) if plan_done else []
    return [g for g, names in groups.items() if plan_done or all(states[n] == "exit" for n in names)]


def experiment_of(row: fleet.Row) -> str:
    a = row.args
    return a[a.index("--experiment") + 1] if "--experiment" in a else "v6"


def state(name: str) -> str | None:
    """The last state in logs/<name>/cloud-status, None if the row has not started."""
    path = fleet.status_path(name)
    lines = path.read_text().split("\n") if path.exists() else []
    lines = [ln for ln in lines if ln.strip()]
    return lines[-1].split()[1] if lines else None


def run_cost(name: str, models: list[str]) -> float:
    if all(m.startswith("mockllm/") for m in models):  # scripted smoke runs: free, and run_cost has no price for them
        return 0.0
    return sum(m["cost"] for f in sorted((ROOT / "logs" / name).rglob("*.eval")) for m in call_costs(f).values())


def run(args: argparse.Namespace) -> int:
    rows = fleet.parse_plan(Path(args.plan).read_text())
    budget = {r.name: fleet.agents(r) * arg(r, "--budget", 0.0) for r in rows}
    models = {r.name: r.args[r.args.index("--models") + 1].split(",") for r in rows}
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(group_of(r.name), []).append(r.name)
    stop = ROOT / "logs" / "_plans" / f"{Path(args.plan).stem}.stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    plan = subprocess.Popen([sys.executable, "-u", "scripts/cloud/fleet.py", "plan", args.plan, "--ref", args.ref,
                             "--max-agents", str(args.max_agents), "--stop-file", str(stop),
                             *(["--group-start"] if args.group_start else []),
                             *(["--script", args.script] if args.script else [])], cwd=ROOT)  # fmt: skip
    costs: dict[str, float] = {}
    selected: set[str] = set()
    queue: list[str] = []
    selecting: list[subprocess.Popen] = []
    after_plan = (args.select_after or ("plan" if any(experiment_of(r) != "v6" for r in rows) else "group")) == "plan"
    print(f"[v6] selections start {'once every row has ended, all groups in one selection' if after_plan else 'per group, one at a time'}",
          flush=True)  # fmt: skip

    def select(names: list[str]) -> subprocess.Popen | None:
        dirs = [f"logs/{n}" for g in names for n in groups[g] if (ROOT / "logs" / n).is_dir()]
        if not dirs:
            return None
        print(f"[v6] {time.strftime('%H:%M:%S')} selecting {', '.join(names)}", flush=True)
        return subprocess.Popen([sys.executable, "-u", "scripts/select_portfolio.py", *dirs, "--scorer", args.scorer,
                                 "--concurrency", str(SELECT_CONCURRENCY), "--validity-n", str(args.validity_n),
                                 *(f"--validity-n-task={x}" for x in args.validity_n_task)], cwd=ROOT)  # fmt: skip

    last_pull = time.monotonic()

    def step(plan_done: bool) -> None:
        nonlocal last_pull
        if time.monotonic() - last_pull >= PULL_S:
            last_pull = time.monotonic()
            try:
                subprocess.run(["bash", "scripts/cloud/scorer.sh", "pull"], cwd=ROOT, timeout=300)
            except subprocess.TimeoutExpired:
                print("[v6] scorer.sh pull timed out", flush=True)
        states = {r.name: state(r.name) for r in rows}
        for name, st in states.items():
            if st == "exit" and name not in costs:
                costs[name] = run_cost(name, models[name])
        if not stop.exists():
            running = sum(budget[n] for n, st in states.items() if st not in (None, "exit"))
            projected = sum(costs.values()) + running + max(budget.values())
            if projected > args.spend_cap:
                stop.touch()
                print(f"[v6] spend cap: measured ${sum(costs.values()):.2f} + running ${running:.2f} + next row "
                      f"> ${args.spend_cap:.2f}; no further rows start", flush=True)  # fmt: skip
        for g in ready_groups(groups, states, plan_done, after_plan):
            if g not in selected:
                selected.add(g)
                queue.append(g)
        selecting[:] = [p for p in selecting if p.poll() is None]
        while not selecting and queue:
            # after_plan: every group in one select_portfolio, whose portfolios run concurrently and whose hidden jobs it
            # orders group by group, arms alternating; v6: one group at a time
            names = queue[:] if after_plan else queue[:1]
            del queue[: len(names)]
            if (p := select(names)) is not None:
                selecting.append(p)

    while plan.poll() is None:
        step(plan_done=False)
        time.sleep(POLL_S)
    step(plan_done=True)
    while selecting or queue:
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
    pr.add_argument("--group-start", action="store_true", help="start each task-repeat group's rows together (v7)")
    pr.add_argument("--validity-n", type=int, default=0, help="select_portfolio.py --validity-n (v7: 1000)")
    pr.add_argument("--validity-n-task", action="append", default=[], metavar="TASK=N", help="select_portfolio.py's")
    pr.add_argument("--script", help="fleet.py plan --script: run this script instead of the runner (smoke tests)")
    pr.add_argument("--select-after", choices=("group", "plan"), default=None,
                    help="start a group's selection when its rows end (group) or every selection once the whole plan "
                         "has ended (plan); default: plan when a row is a v7 row, else group")
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
