"""The v6 driver's two steps (scripts/v6.sh runs them around scorer.sh up/down).

  size <plan> [--machine TYPE] [--selection-scorer TYPE|none]
                 prints "<slots> <machine> <max_agents>": the agent scorer for the plan, read against the AWS quota now
  run <plan> <SPEND_CAP> --ref R --scorer URL --max-agents M [--select-after group|plan] [--selection-scorer TYPE|none]
                 runs the whole plan through fleet.py (rows start as the quota allows), runs select_portfolio.py on the
                 task-repeat groups (below), and stops starting rows near the spend cap

Agent scorer size: each agent counts the most VM vCPUs per agent of any row (v7's m7a boxes: 2). The scorer gets the
smallest machine (then every slot that machine holds, at most MAX_SLOTS) with at least one slot per agents_per_slot
agents that can run alongside it, and beside the selection scorer, in the free quota; when even MAX_SLOTS is too few,
the agents running at once are capped at MAX_SLOTS x agents_per_slot instead. max_agents is always slots x agents_per_slot.
agents_per_slot is AGENTS_PER_SLOT (8) for v6 plans and V7_AGENTS_PER_SLOT (2) for plans with a v7 row: a v7 Opus
agent keeps about a third of a slot busy (v7 batch 1: 16 agents on 3 slots, 89% busy, mean dev_eval queue wait 128 s;
HARNESS.md, "Selection scorer"), so 16 agents get a c7a.24xlarge (11 slots). --machine sets the agent scorer's type
instead (its slots and the cap follow from it); the quota check still applies.

Spend cap: once measured spend of ended runs (run_cost.py's pricing) + the full budget of every running agent + the
budget of the largest row would exceed SPEND_CAP, no further row starts (fleet.py --stop-file); running rows finish.

Groups: rows named <prefix>-<task>-r<k>-<arm>[-i<j>] (plans/v6-*.tsv); a group is <prefix>-<task>-r<k>, so its
portfolios (one per arm) are selected together, in one select_portfolio.py, so on one scorer session.
Selection scorer (default for plans with a v7 row, 2026-10-18; --selection-scorer TYPE, default SELECT_MACHINE, or
none): v7 selection runs on a second scorer of its own (scorer.sh with SCORER_NAME=select), with the agent scorer's
experiment, so its secret offset, and the same ref. It comes up when the first group's rows have all ended (fleet.py
plan --reserve-vcpus keeps its vCPUs free until then), and each group is selected on it as soon as its own rows have
ended: when no selection is running, one select_portfolio.py takes every group waiting (validity pass, stage 2 and
exclusive hidden scoring). Agents never queue behind selection jobs: their VMs only reach the agent scorer. The agent
scorer goes down when the last row has ended, the selection scorer after the last selection (v6.sh takes both down
again on exit, whatever happened). In v7 batch 1 selection after the plan, on the agent scorer, added 1.5-2 h to the
batch after the last agent ended.
Without a selection scorer: --select-after group (v6): each group's selection starts as soon as its own runs have
ended, one selection at a time with --concurrency 6 (main's scorer has 11 slots), so live agents keep most scorer
slots. --select-after plan (v7 with --selection-scorer none): no selection starts until every row of the plan has
ended, then one select_portfolio.py selects every group at once (portfolios concurrently, and with v7's exclusive
hidden scoring a group's hidden jobs next to each other, arms alternating): in the v7 calibration the stable_matching
selection ran beside agents still working, whose dev_eval waits rose to a mean of 221 s (max 835 s) from 76 s.

v7 options: --group-start (fleet.py: a group's rows start together), --validity-n N (select_portfolio.py's validity
pass). Every PULL_S each scorer's logs are copied to logs/_scorer/<instance>/ (scorer.sh pull), so a connection lost
at teardown loses at most that much of jobs.jsonl.
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
AGENTS_PER_SLOT = 8  # v6 (Haiku agents)
V7_AGENTS_PER_SLOT = 2  # 2026-10-18, after v7 batch 1 (module docstring)
SELECT_MACHINE = f"{fleet.AWS_FAMILY}.16xlarge"  # the selection scorer: 7 slots, so validity passes (pagerank: ~1,600 s each) run in parallel; exclusive hidden scoring is serial anyway (Jack, 2026-10-19)
SELECT_UP_TRIES = 3  # scorer.sh up attempts for the selection scorer before its groups are left unselected
POLL_S = 60
PULL_S = 600  # scorer log copies (v6: the main scorer's jobs.jsonl was lost when the one copy at teardown failed)
SELECT_CONCURRENCY = 6  # 2026-10-12, after the pilot: at 3, main's 14 groups needed about 13 h of serial selection


def arg(row: fleet.Row, flag: str, default: float) -> float:
    a = row.args
    return float(a[a.index(flag) + 1]) if flag in a else default


def slots_of(machine: str) -> int:
    return min(MAX_SLOTS, (fleet.vcpus(machine) - fleet.SCORER_SERVICE_VCPUS) // fleet.SLOT_VCPUS)


def agents_per_slot(rows: list[fleet.Row]) -> int:
    return AGENTS_PER_SLOT if all(experiment_of(r) == "v6" for r in rows) else V7_AGENTS_PER_SLOT


def size(rows: list[fleet.Row], free: int, machine: str | None = None, reserve: int = 0) -> tuple[int, str, int]:
    """(slots, machine, max_agents) for the plan's agents in `free` AWS vCPUs, `reserve` of them kept for the selection
    scorer; `machine` fixes the agent scorer's type (see the module docstring)."""
    free -= reserve
    ratio = agents_per_slot(rows)
    total = sum(fleet.agents(r) for r in rows)
    per_agent = max(fleet.vcpus(fleet.machine_for(r, "aws")) / fleet.agents(r) for r in rows)
    if machine is None:
        for want in range(1, MAX_SLOTS + 1):
            machine = fleet.scorer_machine(want)
            concurrent = min(total, int((free - fleet.vcpus(machine)) // per_agent))
            if slots_of(machine) >= math.ceil(concurrent / ratio):
                break
    slots = slots_of(machine)
    if slots < 1:
        raise SystemExit(f"a {machine} scorer has no room for a slot ({fleet.SLOT_VCPUS} vCPUs, "
                         f"{fleet.SCORER_SERVICE_VCPUS} more for the service)")  # fmt: skip
    if free - fleet.vcpus(machine) < max(fleet.vcpus(fleet.machine_for(r, "aws")) for r in rows):
        raise SystemExit(f"{free + reserve} free vCPUs leave no room for an agent VM beside a {machine} scorer"
                         + (f" and {reserve} vCPUs for the selection scorer" if reserve else ""))  # fmt: skip
    return slots, machine, slots * ratio


def selection_machine(rows: list[fleet.Row], choice: str | None) -> str | None:
    """The selection scorer's machine type, or None to select on the agent scorer: `choice` (--selection-scorer;
    "none" = None), by default SELECT_MACHINE for a plan with a v7 row and None for a v6 plan."""
    if choice is not None:
        return None if choice == "none" else choice
    return None if all(experiment_of(r) == "v6" for r in rows) else SELECT_MACHINE


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


def start(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    """A child process of the driver (fleet.py plan, scorer.sh up, select_portfolio.py), run from the repo root."""
    return subprocess.Popen(cmd, cwd=ROOT, env=env)


def scorer_env(name: str) -> dict[str, str]:
    return {**os.environ, "SCORER_NAME": name}


def scorer_field(name: str, key: str) -> str:
    """A field of scorer `name`'s state file (common.sh scorer_field; name agent or select)."""
    out = subprocess.run(["bash", "-c", 'source scripts/cloud/common.sh && scorer_field "$1"', "_", key], cwd=ROOT,
                         env=scorer_env(name), capture_output=True, text=True, check=True)  # fmt: skip
    return out.stdout.strip()


def pull(name: str) -> None:
    """scorer.sh pull on scorer `name`: its logs to logs/_scorer/<instance>/."""
    try:
        subprocess.run(["bash", "scripts/cloud/scorer.sh", "pull"], cwd=ROOT, env=scorer_env(name), timeout=300)
    except subprocess.TimeoutExpired:
        print(f"[v6] scorer.sh pull ({name}) timed out", flush=True)


def scorer_down(name: str) -> None:
    print(f"[v6] {time.strftime('%H:%M:%S')} {name} scorer down", flush=True)
    if subprocess.run(["bash", "scripts/cloud/scorer.sh", "down"], cwd=ROOT, env=scorer_env(name)).returncode:
        print(f"[v6] {name} scorer down failed; v6.sh takes it down again when the driver exits", flush=True)


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
    sel_machine = selection_machine(rows, args.selection_scorer)
    if sel_machine and args.select_after:
        raise SystemExit("--select-after applies only without a selection scorer (--selection-scorer none)")
    # the selection scorer serves the agent scorer's experiment (its secret offset); read while the agent scorer is up
    sel = {"experiment": scorer_field("agent", "experiment") if sel_machine else None, "up": None, "tries": 0,
           "url": None, "token": None, "gave_up": False}  # fmt: skip
    plan = start([sys.executable, "-u", "scripts/cloud/fleet.py", "plan", args.plan, "--ref", args.ref,
                  "--max-agents", str(args.max_agents), "--stop-file", str(stop),
                  *(["--group-start"] if args.group_start else []),
                  *(["--reserve-vcpus", str(fleet.vcpus(sel_machine))] if sel_machine else []),
                  *(["--script", args.script] if args.script else [])])  # fmt: skip
    costs: dict[str, float] = {}
    selected: set[str] = set()
    queue: list[str] = []
    unselected: list[str] = []
    selecting: list[subprocess.Popen] = []
    agent_up = True
    v7 = any(experiment_of(r) != "v6" for r in rows)
    after_plan = sel_machine is None and (args.select_after or ("plan" if v7 else "group")) == "plan"
    when = (f"per group as each ends, on a {sel_machine} selection scorer" if sel_machine else
            "once every row has ended, all groups in one selection" if after_plan else "per group, one at a time")  # fmt: skip
    print(f"[v6] selections start {when}", flush=True)

    def has_logs(g: str) -> bool:
        return any((ROOT / "logs" / n).is_dir() for n in groups[g])

    def select(names: list[str]) -> subprocess.Popen | None:
        dirs = [f"logs/{n}" for g in names for n in groups[g] if (ROOT / "logs" / n).is_dir()]
        if not dirs:
            return None
        print(f"[v6] {time.strftime('%H:%M:%S')} selecting {', '.join(names)}", flush=True)
        # on the selection scorer: its token, and allow-me (select_portfolio's on_unreachable) acts on it
        env = None if sel_machine is None else {**scorer_env("select"), "SCORER_TOKEN": sel["token"]}
        return start([sys.executable, "-u", "scripts/select_portfolio.py", *dirs, "--scorer", sel["url"] or args.scorer,
                      "--concurrency", str(SELECT_CONCURRENCY), "--validity-n", str(args.validity_n),
                      *(f"--validity-n-task={x}" for x in args.validity_n_task)], env)  # fmt: skip

    def selection_scorer() -> None:
        """Start the selection scorer, or see whether it has come up; give up after SELECT_UP_TRIES failures."""
        if sel["up"] is None:
            sel["tries"] += 1
            print(f"[v6] {time.strftime('%H:%M:%S')} {queue[0]} is ready: selection scorer {sel_machine} up "
                  f"(attempt {sel['tries']} of {SELECT_UP_TRIES})", flush=True)  # fmt: skip
            sel["up"] = start(["bash", "scripts/cloud/scorer.sh", "up", "--ref", args.ref, "--machine", sel_machine,
                               "--hours", str(args.scorer_hours), "--experiment", sel["experiment"], "--",
                               "--slots", str(slots_of(sel_machine)), "--memory", args.scorer_memory], scorer_env("select"))  # fmt: skip
        elif (rc := sel["up"].poll()) is not None:
            sel["up"] = None
            if rc == 0:
                sel["url"], sel["token"] = scorer_field("select", "url"), scorer_field("select", "token")
                print(f"[v6] {time.strftime('%H:%M:%S')} selection scorer up at {sel['url']}", flush=True)
            elif sel["tries"] >= SELECT_UP_TRIES:
                sel["gave_up"] = True
                print(f"[v6] the selection scorer did not come up in {SELECT_UP_TRIES} attempts; groups waiting or ending "
                      "later are left unselected", flush=True)  # fmt: skip

    last_pull = time.monotonic()

    def step(plan_done: bool) -> None:
        nonlocal last_pull, agent_up
        if time.monotonic() - last_pull >= PULL_S:
            last_pull = time.monotonic()
            for name, up in (("agent", agent_up), ("select", bool(sel["url"]))):
                if up:
                    pull(name)
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
        if plan_done and agent_up and sel_machine:  # the last row has ended: nothing needs the agent scorer any more
            scorer_down("agent")
            agent_up = False
        for g in ready_groups(groups, states, plan_done, after_plan):
            if g not in selected:
                selected.add(g)
                if has_logs(g):
                    queue.append(g)
        selecting[:] = [p for p in selecting if p.poll() is None]
        if sel_machine and queue and not sel["url"]:
            if sel["gave_up"]:
                unselected.extend(queue)
                queue.clear()
            else:
                selection_scorer()
        while not selecting and queue and (sel_machine is None or sel["url"]):
            # after_plan, or on the selection scorer: every waiting group in one select_portfolio, whose portfolios run
            # concurrently and whose hidden jobs it orders group by group, arms alternating; v6: one group at a time
            names = queue[:] if after_plan or sel_machine else queue[:1]
            del queue[: len(names)]
            if (p := select(names)) is not None:
                selecting.append(p)
        if sel["url"] and plan_done and not queue and not selecting:  # the last selection has ended
            scorer_down("select")
            sel["url"] = None

    while plan.poll() is None:
        step(plan_done=False)
        time.sleep(POLL_S)
    step(plan_done=True)
    while selecting or queue:
        time.sleep(POLL_S / 6)
        step(plan_done=True)
    print(f"[v6] measured spend of ended runs: ${sum(costs.values()):.2f} (cap ${args.spend_cap:.2f})", flush=True)
    if unselected:
        print(f"[v6] not selected (no selection scorer): {', '.join(unselected)}", flush=True)
    return plan.returncode or (1 if unselected else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("size")
    ps.add_argument("plan")
    ps.add_argument("--machine", help="the agent scorer's machine type (default: sized from the plan)")
    ps.add_argument("--selection-scorer", help="as run's: its vCPUs are kept free beside the agent scorer")
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
                    help="without a selection scorer: start a group's selection when its rows end (group) or every "
                         "selection once the whole plan has ended (plan); default: plan when a row is a v7 row, else group")
    pr.add_argument("--selection-scorer", metavar="TYPE|none", default=None,
                    help=f"select each group as it ends on a second scorer of this type (default: {SELECT_MACHINE} when a "
                         "row is a v7 row, else none: select on the agent scorer)")
    pr.add_argument("--scorer-hours", type=int, default=30, help="the selection scorer's backstop (scorer.sh up --hours)")
    pr.add_argument("--scorer-memory", default="16g", help="the selection scorer's slot memory (scorer_service --memory)")
    args = p.parse_args()
    if args.cmd == "size":
        free = fleet.aws_free()
        rows = fleet.parse_plan(Path(args.plan).read_text())
        sel_machine = selection_machine(rows, args.selection_scorer)
        slots, machine, max_agents = size(rows, free, args.machine, fleet.vcpus(sel_machine) if sel_machine else 0)
        print(f"free AWS vCPUs now: {free}", file=sys.stderr)
        print(slots, machine, max_agents)
    else:
        if not os.environ.get("SCORER_TOKEN"):
            sys.exit("set SCORER_TOKEN")
        sys.exit(run(args))


if __name__ == "__main__":
    main()
