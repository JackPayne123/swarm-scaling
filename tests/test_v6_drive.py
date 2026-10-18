"""scripts/v6_drive.py: when selections start. No cloud calls."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/cloud"), str(ROOT / "scripts")]
_spec = importlib.util.spec_from_file_location("v6_drive", ROOT / "scripts/v6_drive.py")
v6_drive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6_drive)


def test_v7_selections_wait_for_every_row_of_the_plan_then_start_together():
    """v7 calibration: stable_matching's selection ran while other rows' agents worked; their dev_eval waits rose to a
    mean of 221 s (max 835 s) from 76 s. v6 kept selecting each group as soon as its own rows ended."""
    groups = {"v7-ode-r1": ["v7-ode-r1-divteam", "v7-ode-r1-div-i0"], "v7-sm-r1": ["v7-sm-r1-divteam", "v7-sm-r1-div-i0"]}
    states = {"v7-ode-r1-divteam": "exit", "v7-ode-r1-div-i0": "exit", "v7-sm-r1-divteam": "running", "v7-sm-r1-div-i0": "exit"}
    assert v6_drive.ready_groups(groups, states, plan_done=False, after_plan=False) == ["v7-ode-r1"]  # v6
    assert v6_drive.ready_groups(groups, states, plan_done=False, after_plan=True) == []
    states["v7-sm-r1-divteam"] = "exit"
    assert v6_drive.ready_groups(groups, states, plan_done=False, after_plan=True) == []  # the fleet plan has not ended
    assert v6_drive.ready_groups(groups, states, plan_done=True, after_plan=True) == ["v7-ode-r1", "v7-sm-r1"]
    rows = v6_drive.fleet.parse_plan("a aws m7a.large --arm independent --experiment v7\nb aws m7a.large --arm independent\n")
    assert [v6_drive.experiment_of(r) for r in rows] == ["v7", "v6"]


def v7_rows(groups: int, experiment: str = "--experiment v7") -> list:
    """Batch-1 groups: a team of 4 on m7a.2xlarge and 4 independents on m7a.large (2 vCPUs per agent)."""
    lines = []
    for g in range(groups):
        lines.append(f"v7b-t{g}-r1-divteam aws m7a.2xlarge --arm team --n 4 --models m --budget 5 {experiment}")
        lines += [f"v7b-t{g}-r1-div-i{k} aws m7a.large --arm independent --models m --budget 5 {experiment}" for k in range(4)]
    return v6_drive.fleet.parse_plan("\n".join(lines))


def test_v7_agent_scorer_gets_a_slot_per_two_agents_beside_the_selection_scorer_and_an_explicit_type_wins():
    """v7 batch 1: 16 Opus agents on the v6 rule's 3 slots kept them 89% busy and agents spent 43% of their wall time
    in dev_eval (mean 202 s a call). A v7 agent keeps about a third of a slot busy, so v7 sizes 2 agents per slot."""
    sel = v6_drive.fleet.vcpus(v6_drive.SELECT_MACHINE)
    assert v6_drive.SELECT_MACHINE == f"{v6_drive.fleet.AWS_FAMILY}.8xlarge" and sel == 32
    fam = v6_drive.fleet.AWS_FAMILY
    assert v6_drive.size(v7_rows(2), 512, reserve=sel) == (11, f"{fam}.24xlarge", 22)  # 16 agents
    assert v6_drive.size(v7_rows(1), 512, reserve=sel) == (5, f"{fam}.12xlarge", 10)  # 8 agents
    assert v6_drive.size(v7_rows(2, experiment=""), 512) == (3, f"{fam}.8xlarge", 24)  # v6 rows: 8 per slot, as before
    assert v6_drive.size(v7_rows(2), 512, f"{fam}.8xlarge", sel) == (3, f"{fam}.8xlarge", 6)  # explicit: ratio kept
    # the quota check counts the selection scorer: 7 vCPUs left beside both scorers is no room for a team VM
    assert v6_drive.size(v7_rows(2), 96 + 7 + sel, f"{fam}.24xlarge") == (11, f"{fam}.24xlarge", 22)
    with pytest.raises(SystemExit):
        v6_drive.size(v7_rows(2), 96 + 7 + sel, f"{fam}.24xlarge", sel)
    assert v6_drive.selection_machine(v7_rows(1), None) == v6_drive.SELECT_MACHINE
    assert v6_drive.selection_machine(v7_rows(1, experiment=""), None) is None  # v6 selects on the agent scorer
    assert v6_drive.selection_machine(v7_rows(1), "none") is None


class Proc:
    """A fake child process that is still running for its first `polls` polls."""

    def __init__(self, polls: int, rc: int = 0):
        self.polls, self.rc, self.returncode = polls, rc, None

    def poll(self):
        if self.polls:
            self.polls -= 1
            return None
        self.returncode = self.rc
        return self.rc


def test_v7_groups_select_as_they_end_on_a_selection_scorer_that_comes_up_with_the_first_and_scorers_go_down_when_done(
    tmp_path, monkeypatch
):
    """v7 batch 1 waited for every row, then ran selection with exclusive hidden scoring, one job at a time, on the
    agents' scorer for another 1.5-2 h. Now group 1 is selected on a second scorer while the other groups' agents still
    work, and agents never share a scorer with selection jobs. Scorer and fleet calls are faked."""
    rows = ["v7x-qr-r1-divteam aws m7a.2xlarge --arm team --n 2 --models mockllm/model --budget 1 --experiment v7",
            "v7x-qr-r1-div-i0 aws m7a.large --arm independent --models mockllm/model --budget 1 --experiment v7",
            "v7x-pr-r1-div-i0 aws m7a.large --arm independent --models mockllm/model --budget 1 --experiment v7",
            "v7x-sm-r1-div-i0 aws m7a.large --arm independent --models mockllm/model --budget 1 --experiment v7"]  # fmt: skip
    plan = tmp_path / "plan.tsv"
    plan.write_text("\n".join(rows) + "\n")
    names = [r.split()[0] for r in rows]
    for n in names:
        (tmp_path / "logs" / n).mkdir(parents=True)
    exits = {names[0]: 3, names[1]: 3, names[2]: 10, names[3]: 10}  # the step each row's cloud-status reaches "exit"
    step, events = [0], []

    def state(name):
        if name == names[0]:
            step[0] += 1  # step() reads every row's state once, this one first
        return "exit" if step[0] >= exits[name] else "running"

    def start(cmd, env=None):
        if "scripts/cloud/fleet.py" in cmd:
            events.append(("fleet", cmd))
            return Proc(polls=10)  # the fleet plan ends after step 10, when the last rows have ended
        if cmd[:3] == ["bash", "scripts/cloud/scorer.sh", "up"]:
            events.append(("up", env["SCORER_NAME"], step[0], cmd))
            return Proc(polls=1)
        assert cmd[2] == "scripts/select_portfolio.py"
        events.append(("select", step[0], cmd[3 : cmd.index("--scorer")], cmd[cmd.index("--scorer") + 1],
                       env["SCORER_NAME"], env["SCORER_TOKEN"]))  # fmt: skip
        return Proc(polls=2)

    fields = {("agent", "experiment"): "v7", ("select", "url"): "http://sel:8770", ("select", "token"): "sel-token"}
    monkeypatch.setattr(v6_drive, "ROOT", tmp_path)
    monkeypatch.setattr(v6_drive, "POLL_S", 0)
    monkeypatch.setattr(v6_drive, "PULL_S", 0)  # pull every step
    monkeypatch.setattr(v6_drive, "state", state)
    monkeypatch.setattr(v6_drive, "start", start)
    monkeypatch.setattr(v6_drive, "scorer_field", lambda name, key: fields[(name, key)])
    monkeypatch.setattr(v6_drive, "pull", lambda name: events.append(("pull", name)))
    monkeypatch.setattr(v6_drive, "scorer_down", lambda name: events.append(("down", name, step[0])))
    args = argparse.Namespace(plan=str(plan), spend_cap=1e9, ref="r", scorer="http://agent:8770", max_agents=22,
                              group_start=True, validity_n=20, validity_n_task=[], script=None, select_after=None,
                              selection_scorer=None, scorer_hours=4, scorer_memory="16g")  # fmt: skip
    assert v6_drive.run(args) == 0

    fleet_cmd = events[0][1]
    assert fleet_cmd[fleet_cmd.index("--reserve-vcpus") + 1] == "32"  # room kept for the selection scorer
    life = [e[:3] if e[0] != "select" else ("select", e[1], e[2]) for e in events if e[0] not in ("fleet", "pull")]
    assert life == [
        ("up", "select", 3),  # when group 1's rows have ended, not at launch
        ("select", 5, [f"logs/{n}" for n in names[:2]]),  # once it is up (the fake up takes a step); it ends at step 8
        ("select", 10, [f"logs/{n}" for n in names[2:]]),  # still up at step 10: groups 2 and 3 together
        ("down", "agent", 11),  # the step after the plan's last rows ended
        ("down", "select", 13),  # after the last selection
    ]
    up = next(e for e in events if e[0] == "up")[3]
    assert up[up.index("--experiment") + 1] == "v7" and up[up.index("--slots") + 1] == "3"  # the agent scorer's offset
    assert {e[3:] for e in events if e[0] == "select"} == {("http://sel:8770", "select", "sel-token")}
    order = [e[:2] for e in events if e[0] in ("pull", "down", "up")]
    agent_down, sel_up, sel_down = order.index(("down", "agent")), order.index(("up", "select")), order.index(("down", "select"))
    assert all(i < agent_down for i, e in enumerate(order) if e == ("pull", "agent"))
    assert all(sel_up < i < sel_down for i, e in enumerate(order) if e == ("pull", "select"))
    assert ("pull", "select") in order
