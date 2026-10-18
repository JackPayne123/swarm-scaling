"""scripts/v6_drive.py: when selections start. No cloud calls."""

import importlib.util
import sys
from pathlib import Path

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
