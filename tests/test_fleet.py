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
fixed   aws    c7a.4xlarge --arm solo --models mockllm/model --budget 10
"""


def test_auto_sizes_by_agents_times_cpus_plus_two():
    """Agent VMs are sized n x cpus_per_agent + 2, smallest type that fits: 6 -> 8, 10 -> 16, 18 -> 32 vCPUs."""
    rows = {r.name: r for r in fleet.parse_plan(PLAN)}
    assert fleet.machine_for(rows["solo"], "gcp") == "t2d-standard-8"
    assert fleet.machine_for(rows["team2"], "gcp") == "t2d-standard-16"
    assert fleet.machine_for(rows["team4"], "aws") == "c7a.8xlarge"
    assert fleet.machine_for(rows["fixed"], "aws") == "c7a.4xlarge"
    assert rows["team2"].args[:4] == ["--arm", "team", "--n", "2"]


def test_any_fills_gcp_first_then_aws():
    row = fleet.parse_plan(PLAN)[1]
    assert fleet.choose(row, {"gcp": 24, "aws": 32}) == ("gcp", "t2d-standard-16")
    assert fleet.choose(row, {"gcp": 8, "aws": 32}) == ("aws", "c7a.4xlarge")
    assert fleet.choose(row, {"gcp": 8, "aws": 0}) is None  # e.g. the scorer holds the whole AWS quota


def test_explicit_cloud_never_spills_over():
    solo = fleet.parse_plan(PLAN)[0]
    assert fleet.choose(solo, {"gcp": 4, "aws": 32}) is None


@pytest.mark.parametrize(
    "line",
    ["x gcp c7a.4xlarge --n 1", "x azure auto --n 1", "x gcp auto --arm team --n 8 --models m", "a gcp auto\na aws auto"],
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


def test_gcp_room_is_the_global_quota_only_when_every_row_uses_the_remote_scorer(monkeypatch):
    """Remote-checker agent VMs may fall back to any x86 family and US region; local-checker runs time on T2D."""
    region = {"quotas": [{"metric": "T2D_CPUS", "limit": 24, "usage": 0}, {"metric": "CPUS", "limit": 200, "usage": 0}]}
    project = {"quotas": [{"metric": "CPUS_ALL_REGIONS", "limit": 32, "usage": 0}]}
    monkeypatch.setattr(fleet, "run_json", lambda cmd: region if "regions" in cmd else project)
    rows = fleet.parse_plan("a gcp auto --checker remote\nb any auto --checker=remote\n")
    assert fleet.gcp_free(t2d_only=not all(fleet.is_remote(r.args) for r in rows)) == 32
    rows.append(fleet.parse_plan("c gcp auto --arm solo\n")[0])
    assert fleet.gcp_free(t2d_only=not all(fleet.is_remote(r.args) for r in rows)) == 24


def test_dry_run_fills_gcp_then_aws_and_queues_the_rest():
    rows = fleet.parse_plan("t4 any auto --n 4\nt2 any auto --n 2\ns1 any auto\ns2 any auto\ns3 gcp auto\n")
    plan = fleet.dry_run(rows, {"gcp": 32, "aws": 16})
    assert plan == [
        ("t4", "gcp", "t2d-standard-32", "at launch"),
        ("t2", "aws", "c7a.4xlarge", "at launch"),
        ("s1", "gcp", "t2d-standard-8", "after t4"),
        ("s2", "gcp", "t2d-standard-8", "after t4"),
        ("s3", "gcp", "t2d-standard-8", "after t4"),
    ]


def test_scorer_gets_one_slot_per_four_remote_agents_on_the_smallest_type_that_fits(tmp_path):
    """Pilot 4 put 16 agents on 2 slots and they queued; 4 slots x 8 vCPUs + 8 for the service = 40 -> 12xlarge."""
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "t4 aws auto --arm team --n 4 --checker remote\nt2 aws auto --arm team --n 2 --checker remote\n"
        + "".join(f"i{k} aws auto --arm independent --checker=remote\n" for k in range(10))
        + "local gcp auto --arm team --n 4\n"  # times on its own checker: not counted
    )
    fam = fleet.AWS_FAMILY
    assert fleet.scorer_args(["up", "--ref", "r", "--plan", str(plan)]) == [
        "up", "--ref", "r", "--machine", f"{fam}.12xlarge", "--", "--slots", "4"
    ]
    # overrides: an explicit slot count sizes the machine; an explicit machine is kept
    assert fleet.scorer_args(["up", "--plan", str(plan), "--", "--slots", "7"])[1:] == ["--machine", f"{fam}.16xlarge", "--", "--slots", "7"]
    assert fleet.scorer_args(["up", "--machine", "x.y", "--plan", str(plan)])[1:3] == ["--machine", "x.y"]
    assert fleet.scorer_args(["status"]) == ["status"]
