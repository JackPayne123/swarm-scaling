"""scripts/cloud/fleet.py: plan parsing, VM sizing and which cloud a row starts on. No cloud calls."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fleet", ROOT / "scripts/cloud/fleet.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)

PLAN = """
# name  cloud  machine  args
solo    gcp    auto     --arm independent --models mockllm/model --budget 10
team2   any    auto     --arm team --n 2 --models mockllm/model --budget 10
team4   aws    auto     --arm team --n 4 --cpus-per-agent=4 --models mockllm/model --budget 10
fixed   aws    m7a.4xlarge --arm solo --models mockllm/model --budget 10
"""


def test_auto_sizes_by_agents_times_cpus_plus_two():
    """Agent VMs are sized n x cpus_per_agent + 2, smallest type that fits: 6 -> 8, 10 -> 16, 18 -> 32 vCPUs."""
    rows = {r.name: r for r in fleet.parse_plan(PLAN)}
    assert fleet.machine_for(rows["solo"], "gcp") == "t2d-standard-8"
    assert fleet.machine_for(rows["team2"], "gcp") == "t2d-standard-16"
    assert fleet.machine_for(rows["team4"], "aws") == "m7a.8xlarge"
    assert fleet.machine_for(rows["fixed"], "aws") == "m7a.4xlarge"
    assert rows["team2"].args[:4] == ["--arm", "team", "--n", "2"]


def test_any_fills_gcp_first_then_aws():
    row = fleet.parse_plan(PLAN)[1]
    assert fleet.choose(row, {"gcp": 24, "aws": 32}) == ("gcp", "t2d-standard-16")
    assert fleet.choose(row, {"gcp": 8, "aws": 32}) == ("aws", "m7a.4xlarge")
    assert fleet.choose(row, {"gcp": 8, "aws": 0}) is None  # e.g. the scorer holds the whole AWS quota


def test_explicit_cloud_never_spills_over():
    solo = fleet.parse_plan(PLAN)[0]
    assert fleet.choose(solo, {"gcp": 4, "aws": 32}) is None


@pytest.mark.parametrize(
    "line",
    ["x gcp m7a.4xlarge --n 1", "x azure auto --n 1", "x gcp auto --arm team --n 8 --models m", "a gcp auto\na aws auto"],
)
def test_bad_plans_fail_before_anything_starts(line):
    """Wrong-cloud machine, unknown cloud, no type big enough (8 x 4 + 2 = 34 vCPUs), duplicate names."""
    with pytest.raises(SystemExit):
        fleet.parse_plan(line)


def test_schedule_respects_free_vcpus_and_starts_queued_rows_as_room_frees():
    """24 free GCP vCPUs fit one 16-vCPU team plus one 8-vCPU solo at a time; the rest wait, none overlap beyond."""
    import threading
    import time

    rows = fleet.parse_plan("t1 gcp auto --n 2\nt2 gcp auto --n 2\ns1 gcp auto\ns2 gcp auto\n")
    lock, live, peak = threading.Lock(), [0], [0]

    def launch(row, machine):
        with lock:
            live[0] += fleet.vcpus(machine)
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= fleet.vcpus(machine)
        return 0

    free = {"gcp": 24, "aws": 0}
    assert fleet.schedule(rows, free, launch) == {"t1": 0, "t2": 0, "s1": 0, "s2": 0}
    assert peak[0] <= 24
    assert free == {"gcp": 24, "aws": 0}


def test_schedule_fails_rows_that_never_fit():
    rows = fleet.parse_plan("big aws auto --n 4\n")  # needs 32 vCPUs; the scorer holds the AWS quota
    assert fleet.schedule(rows, {"gcp": 24, "aws": 0}, lambda row, machine: 0) == {"big": 1}
