"""scripts/cloud/fleet.py: plan parsing, VM sizing and which cloud a row starts on. No cloud calls."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fleet", ROOT / "scripts/cloud/fleet.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)
_spec = importlib.util.spec_from_file_location("make_v7_plan", ROOT / "scripts/make_v7_plan.py")
make_v7_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_v7_plan)

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


def test_schedule_caps_running_agents_and_starts_nothing_once_the_stop_file_exists():
    """v6 keeps at most 8 agents per scorer slot running, and stops launching at the spend cap; running rows finish."""
    import threading
    import time

    rows = fleet.parse_plan("t1 aws auto --n 4\ni1 aws auto\ni2 aws auto\ni3 aws auto\ni4 aws auto\ni5 aws auto\n")
    lock, live, peak, started = threading.Lock(), [0], [0], []

    def launch(row, machine):
        with lock:
            started.append(row.name)
            live[0] += fleet.agents(row)
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= fleet.agents(row)
        return 0

    stop = lambda: len(started) >= 2  # noqa: E731 (the cap is passed once the first wave has started)
    results = fleet.schedule(rows, {"gcp": 0, "aws": 300}, launch, max_agents=5, stop=stop)
    assert started == ["t1", "i1"] and peak[0] == 5  # i2 would have made 6 agents
    assert results == {"t1": 0, "i1": 0, "i2": 3, "i3": 3, "i4": 3, "i5": 3}


def test_independents_and_teams_of_four_at_two_cpus_per_agent_get_the_small_c7a_types():
    rows = fleet.parse_plan("i aws auto --cpus-per-agent 2\nt aws auto --n 4 --cpus-per-agent 2\n")
    assert [fleet.machine_for(r, "aws") for r in rows] == [f"{fleet.AWS_FAMILY}.xlarge", f"{fleet.AWS_FAMILY}.4xlarge"]


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



def test_collect_pulls_finished_uploads_by_run_name_for_vms_on_either_cloud():
    """Logs survive an expired operator login: every VM uploads to gs://<LOG_BUCKET>/runs/<name>/, collect pulls them."""
    import subprocess

    url = subprocess.run(["bash", "-c", "source scripts/cloud/common.sh && log_url r1"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    assert url == f"gs://{fleet.COMMON['LOG_BUCKET']}/runs/r1"
    rows = fleet.parse_plan("on-aws aws auto\non-gcp gcp auto\nstill-running any auto\n")
    tried = []

    def fetch(name):
        tried.append(name)
        return name != "still-running"

    assert fleet.collect_rows(rows, fetch) == {"on-aws": True, "on-gcp": True, "still-running": False}
    assert tried == ["on-aws", "on-gcp", "still-running"]


@pytest.mark.parametrize("mt,expected", [
    ("m7a.large", "m7a.large r7a.large"),
    ("m7a.4xlarge", "m7a.4xlarge r7a.4xlarge"),
    ("c7a.xlarge", "c7a.xlarge m7a.xlarge m6a.xlarge c6a.xlarge m7i.xlarge c7i.xlarge"),
    ("t3a.large", "t3a.large"),
])
def test_a_memory_driven_type_never_falls_back_to_less_memory_per_vcpu_or_another_cpu(mt, expected):
    """sinkhorn rows ask for m7a (4 GiB per vCPU) because 100 problems peak at 9.6 GiB in the agent box; a c7a.xlarge
    fallback (8 GiB, about 5.6 GiB for the box) would run out of memory. v7 agent boxes are m7a so that every arm times
    locally on the scorer's Zen 4: m6a (Zen 3) or m7i (Intel) would put one arm on another CPU, r7a has the same one."""
    import subprocess

    out = subprocess.run(["bash", "-c", f"source scripts/cloud/common.sh && fallback_types {mt}"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert out == expected.split()


def test_schedule_spaces_vm_starts_so_a_wave_does_not_hit_awss_volume_and_request_limits():
    """v6 main started ~50 VMs at once: AWS throttled RunInstances and its volume initialization cap ended 11 new VMs."""
    import time

    starts = []
    rows = fleet.parse_plan("a aws auto\nb aws auto\nc aws auto\n")
    fleet.schedule(rows, {"gcp": 0, "aws": 300}, lambda row, machine: starts.append(time.monotonic()) or 0, start_gap=0.2)
    assert len(starts) == 3 and all(b - a >= 0.19 for a, b in zip(starts, starts[1:]))


def test_group_start_starts_a_groups_team_with_its_independents_instead_of_after_them():
    """v6: 4-vCPU independents backfilled freed quota and a group's 16-vCPU team waited 83-166 min in 10 of 14 groups."""
    import threading
    import time

    plan = "".join(f"v7-qp-r{k}-team aws auto --n 4 --cpus-per-agent 2\n"
                   + "".join(f"v7-qp-r{k}-div-i{j} aws auto --cpus-per-agent 2\n" for j in range(4)) for k in (1, 2))  # fmt: skip
    rows = fleet.parse_plan(plan)
    lock, started = threading.Lock(), []

    def launch(row, machine):
        with lock:
            started.append(row.name)
        time.sleep(0.1 if "team" in row.name else 0.02)
        return 0

    # 40 free vCPUs: one group (16 + 4 x 4 = 32) at a time. Row by row, group 2's independents would fill the 8 left.
    fleet.schedule(rows, {"gcp": 0, "aws": 40}, launch, group=fleet.group_of)
    assert started == [r.name for r in rows]  # group 2 started only once group 1 ended, all at once, team first
    assert fleet.group_of("v7-qp-r2-div-i3") == "v7-qp-r2"
    # without group-start, group 2's independents start while group 1's team still runs and its team waits
    started.clear()
    fleet.schedule(rows, {"gcp": 0, "aws": 40}, launch)
    assert started[5:7] == ["v7-qp-r2-div-i0", "v7-qp-r2-div-i1"]


def test_vms_are_priced_at_their_own_regions_rate_and_every_default_v7_box_has_a_us_east_1_price():
    """The calibration's cost estimate priced a us-east-1 m7a.4xlarge at Sydney's $1.1592/h (us-east-1: $0.92736/h)."""
    assert fleet.region_of("us-east-1c") == "us-east-1" and fleet.region_of("us-central1-a") == "us-central1"
    assert fleet.hourly_usd("m7a.4xlarge", "us-east-1c") == 0.92736
    assert fleet.hourly_usd("m7a.4xlarge", "ap-southeast-2b") == 1.1592
    assert fleet.hourly_usd("c7a.8xlarge", "ap-southeast-2a") is None  # no c7a in Sydney: unpriced, never borrowed
    assert fleet.cost("c7a.8xlarge", "ap-southeast-2a", 1.0) == "unpriced in ap-southeast-2"
    import subprocess

    boxes = {t for (family, _), pair in make_v7_plan.BOXES.items() if family == "m7a" for t in pair}  # the default
    fallbacks = {f for t in boxes for f in subprocess.run(["bash", "-c", f"source scripts/cloud/common.sh && fallback_types {t}"],
                                                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()}  # fmt: skip
    assert [t for t in sorted(fallbacks) if fleet.hourly_usd(t, "us-east-1a") is None] == []


def test_v7_plans_put_every_arm_on_the_scorers_cpu_by_default():
    """stable_matching calibration: the team's m7a box had the scorer's Zen 4 (AVX-512), the independents' t3a Zen 1."""
    for big in (False, True):
        lines = make_v7_plan.rows("p", "qp", 1, "m", "high", 5, big)
        machines = {line.split()[0].rsplit("-", 1)[-1]: line.split()[2] for line in lines}
        assert machines == ({"divteam": "m7a.4xlarge", **{f"i{k}": "m7a.xlarge" for k in range(4)}} if big else
                            {"divteam": "m7a.2xlarge", **{f"i{k}": "m7a.large" for k in range(4)}})  # fmt: skip
        assert len({line.split("--agent-memory-mb ")[1].split()[0] for line in lines}) == 1  # one box memory per task


def test_the_public_ip_lookup_retries_with_backoff_until_the_network_returns_and_gives_up_after_its_bound(tmp_path):
    """v7 calibration: allow-me failed once with "could not read this machine's public IP" while the network was down,
    and selection stalled until a human re-ran it. The lookup and sleep are injected; no network, no waiting."""
    import subprocess

    def my_ip(failures: int, wait_s: int) -> tuple[int, str, list[str]]:
        script = f"""source scripts/cloud/common.sh
            _my_ip_lookup() {{  # runs in $(...): count calls in a file
              echo x >> {tmp_path}/calls
              [ "$(wc -l < {tmp_path}/calls)" -gt {failures} ] && echo 203.0.113.7 || echo '<html>captive</html>'
            }}
            sleep() {{ echo "$1" >> {tmp_path}/slept; }}
            MY_IP_WAIT_S={wait_s} my_ip"""
        (tmp_path / "slept").write_text("")
        (tmp_path / "calls").write_text("")
        out = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True)
        return out.returncode, out.stdout.strip(), (tmp_path / "slept").read_text().split()

    assert my_ip(0, 900) == (0, "203.0.113.7", [])
    assert my_ip(4, 900) == (0, "203.0.113.7", ["5", "10", "20", "40"])  # a non-IP answer counts as a failure
    rc, out, slept = my_ip(1000, 900)  # never returns: about 15 min of backoff, capped at 60 s per wait, then fails
    assert rc == 1 and out == "" and slept[:6] == ["5", "10", "20", "40", "60", "60"] and sum(map(int, slept)) in range(900, 961)
