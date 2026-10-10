"""Run a plan of samples on GCP and AWS VMs (one VM each, via run_sample.sh), show live VMs and cost, clean up.

Usage:
  uv run python scripts/cloud/fleet.py plan <plan.tsv> --ref <git-ref> [--script <path>] [--max-runs N] [--dry-run]
  uv run python scripts/cloud/fleet.py status
  uv run python scripts/cloud/fleet.py cleanup [--yes]
  uv run python scripts/cloud/fleet.py scorer up|down|status [args]   (scripts/cloud/scorer.sh)
  uv run python scripts/cloud/fleet.py scorer up --plan <plan.tsv> --ref <git-ref> [scorer.sh up args]

Plan file: one run per line, `name  cloud  machine  runner args...` (tab or space separated; the args are split like
a shell would), blank lines and # comments ignored.
  cloud    gcp | aws | any (any: GCP first, AWS when GCP has no room)
  machine  a machine type of that cloud (t2d-standard-16, c7a.4xlarge) or `auto`: the smallest type with at least
           n x cpus_per_agent + 2 vCPUs (`--n`, default 1; `--cpus-per-agent`, default 4; +2 for the OS and Inspect)
Free vCPUs per cloud are read once at launch: GCP CPUS_ALL_REGIONS left when every row is `--checker remote`
(their agent VMs may fall back to other x86 families and US regions, see run_sample.sh), else min(T2D_CPUS, CPUS
in us-central1, CPUS_ALL_REGIONS) left; AWS the on-demand standard quota (service-quotas) minus every running
instance's vCPUs (a scorer VM included). Machine types are what a row asks for; cloud-status records what it got.
Rows start as soon as their cloud has room, in file order with backfill; a row that cannot fit even in an
empty cloud fails at once. `plan` warns about rows whose --time-limit is below their dollar budget / SPEND_USD_PER_H. Each run's output goes to logs/<name>/driver.log and its progress to
logs/<name>/cloud-status. Run a long plan under bgjob (--grace 60) and caffeinate.

cleanup lists every VM tagged/labelled tool=swarm-runner on both clouds and, with --yes, deletes them (running
plans included) and runs `scorer.sh down`.

scorer up --plan sizes the scorer from the plan's `--checker remote` rows: slots = ceil(agents / AGENTS_PER_SLOT)
(agents = the sum of each row's --n) and the smallest <AWS_FAMILY> type with slots x 8 + 8 vCPUs (8 per slot, 8 for
the service). An explicit --machine or `-- --slots S` overrides either.
"""

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = dict(re.findall(r"^(\w+)=([^\s$]+)", (ROOT / "scripts/cloud/common.sh").read_text(), re.M))
GCLOUD = ["gcloud", f"--account={COMMON['GCP_ACCOUNT']}", f"--project={COMMON['GCP_PROJECT']}", "--quiet"]
AWS_REGION, AWS_FAMILY = subprocess.run(  # common.sh decides both (env AWS_REGION overrides its default)
    ["bash", "-c", 'source scripts/cloud/common.sh && echo "$AWS_REGION $AWS_FAMILY"'], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.split()
AWS = ["aws", "--profile", COMMON["AWS_PROFILE_NAME"], "--region", AWS_REGION, "--output", "json"]
TOOL = COMMON["TOOL_LABEL"]
GCP_REGION = "us-central1"
AWS_STANDARD_QUOTA = "L-1216C47A"  # Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances, vCPUs

# Machine types `auto` picks from, smallest first. AWS: c7a in us-east-1, m7a in ap-southeast-2 (no c7a there).
MACHINES = {
    "gcp": ["t2d-standard-8", "t2d-standard-16", "t2d-standard-32"],
    "aws": [f"{AWS_FAMILY}.2xlarge", f"{AWS_FAMILY}.4xlarge", f"{AWS_FAMILY}.8xlarge"],
}
AWS_SIZES = {"large": 2, "xlarge": 4, "2xlarge": 8, "4xlarge": 16, "8xlarge": 32, "12xlarge": 48, "16xlarge": 64,
             "24xlarge": 96, "32xlarge": 128, "48xlarge": 192}
AGENTS_PER_SLOT = 4  # one scorer slot per 4 agents (decided 2026-10-10)
SLOT_VCPUS, SCORER_SERVICE_VCPUS = 8, 8  # as scorer_service.py's SLOT_CPUS and SERVICE_CPUS
# Pilot 4 (2026-10-10): the fastest-spending independent Opus 5.5 agent spent about $1.9 per hour. A row whose time
# limit is below its per-agent dollar budget / this rate may end by time before it can spend its budget.
SPEND_USD_PER_H = 1.9

# On-demand compute USD per hour. Estimates for the cost column only; check billing. They leave out disks
# (60 GB pd-balanced / gp3, about $0.01/h) and public IPs (about $0.005/h).
HOURLY_USD = {
    # GCP us-central1, third-party trackers July 2026; -8 is half of -16 (T2D is priced per vCPU and GB)
    "t2d-standard-8": 0.34, "t2d-standard-16": 0.68, "t2d-standard-32": 1.35,
    "e2-standard-8": 0.27,  # unverified
    # AWS Linux on-demand, AWS public price map published 2026-10-08: ap-southeast-2
    "m7a.2xlarge": 0.5796, "m7a.4xlarge": 1.1592, "m7a.8xlarge": 2.3184,
    # AWS us-east-1 Linux on-demand, same source
    "c7a.2xlarge": 0.41056, "c7a.4xlarge": 0.82112, "c7a.8xlarge": 1.64224,
}


@dataclass
class Row:
    name: str
    cloud: str  # gcp | aws | any
    machine: str  # a machine type or "auto"
    args: list[str]


def cloud_of(machine: str) -> str:
    return "aws" if "." in machine else "gcp"


def vcpus(machine: str) -> int:
    if cloud_of(machine) == "aws":
        return AWS_SIZES[machine.split(".", 1)[1]]
    return int(machine.rsplit("-", 1)[1])


def arg_value(args: list[str], flag: str, default, cast=int):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return cast(args[i + 1])
        if a.startswith(flag + "="):
            return cast(a.split("=", 1)[1])
    return default


def needed_vcpus(args: list[str]) -> int:
    return arg_value(args, "--n", 1) * arg_value(args, "--cpus-per-agent", 4) + 2


def machine_for(row: Row, cloud: str) -> str | None:
    """The machine type `row` would use on `cloud`, or None if no listed type is big enough."""
    if row.machine != "auto":
        return row.machine if cloud_of(row.machine) == cloud else None
    need = needed_vcpus(row.args)
    return next((m for m in MACHINES[cloud] if vcpus(m) >= need), None)


def candidates(row: Row) -> list[str]:
    return ["gcp", "aws"] if row.cloud == "any" else [row.cloud]


def choose(row: Row, free: dict[str, int]) -> tuple[str, str] | None:
    """(cloud, machine) for the first candidate cloud with room for this row now, else None."""
    for cloud in candidates(row):
        machine = machine_for(row, cloud)
        if machine and vcpus(machine) <= free[cloud]:
            return cloud, machine
    return None


def parse_plan(text: str) -> list[Row]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, cloud, machine, *rest = line.split(None, 3)
        if cloud not in ("gcp", "aws", "any"):
            raise SystemExit(f"{name}: cloud must be gcp, aws or any, not {cloud!r}")
        row = Row(name, cloud, machine, shlex.split(rest[0]) if rest else [])
        if machine != "auto" and cloud != "any" and cloud_of(machine) != cloud:
            raise SystemExit(f"{name}: {machine} is not a {cloud} machine type")
        if not any(machine_for(row, c) for c in candidates(row)):
            raise SystemExit(f"{name}: no {row.cloud} machine type has {needed_vcpus(row.args)} vCPUs")
        rows.append(row)
    names = [r.name for r in rows]
    if dupes := {n for n in names if names.count(n) > 1}:
        raise SystemExit(f"duplicate run names: {sorted(dupes)}")
    return rows


def run_json(cmd: list[str]) -> dict | list:
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def is_remote(args: list[str]) -> bool:
    return "--checker=remote" in args or any(a == "--checker" and b == "remote" for a, b in zip(args, args[1:]))


def time_limit_warnings(rows: list[Row]) -> list[str]:
    """Rows with a dollar budget whose per-agent time limit (runner default 3600 s) is below budget / SPEND_USD_PER_H."""
    out = []
    for r in rows:
        budget = arg_value(r.args, "--budget", None, float)
        if budget is None or arg_value(r.args, "--budget-type", "cost", str) != "cost":
            continue
        budget *= arg_value(r.args, "--mult", 1)
        limit_h = arg_value(r.args, "--time-limit", 3600) / 3600
        if limit_h < budget / SPEND_USD_PER_H:
            out.append(f"{r.name}: time limit {limit_h:.2f} h is below ${budget:g} / ${SPEND_USD_PER_H}/h = "
                       f"{budget / SPEND_USD_PER_H:.2f} h; agents may end by time before spending their budget")
    return out


def scorer_size(rows: list[Row]) -> tuple[int, str]:
    """(slots, machine) for the plan's remote-checker agents: one slot per AGENTS_PER_SLOT, smallest type that fits."""
    agents = sum(arg_value(r.args, "--n", 1) for r in rows if is_remote(r.args))
    if not agents:
        raise SystemExit("no --checker remote rows in the plan: nothing uses the scorer")
    slots = math.ceil(agents / AGENTS_PER_SLOT)
    return slots, scorer_machine(slots)


def scorer_machine(slots: int) -> str:
    need = slots * SLOT_VCPUS + SCORER_SERVICE_VCPUS
    size = next((s for s, v in sorted(AWS_SIZES.items(), key=lambda kv: kv[1]) if v >= need), None)
    if size is None:
        raise SystemExit(f"no {AWS_FAMILY} type has {need} vCPUs for {slots} slots")
    return f"{AWS_FAMILY}.{size}"


def scorer_args(args: list[str]) -> list[str]:
    """scorer.sh arguments with `--plan <tsv>` replaced by the --machine and `-- --slots` it implies (`up` only)."""
    if "--plan" not in args:
        return args
    i = args.index("--plan")
    plan_file, args = args[i + 1], args[:i] + args[i + 2 :]
    own, service = (args[: args.index("--")], args[args.index("--") + 1 :]) if "--" in args else (args, [])
    slots, machine = scorer_size(parse_plan(Path(plan_file).read_text()))
    if "--slots" in service or any(a.startswith("--slots=") for a in service):
        slots = arg_value(service, "--slots", slots)
        machine = scorer_machine(slots)
    else:
        service = ["--slots", str(slots), *service]
    if "--machine" not in own:
        own = [*own, "--machine", machine]
    print(f"scorer for {plan_file}: {' '.join(own[1:])} -- {' '.join(service)}", flush=True)
    return [*own, "--", *service]


def gcp_free(t2d_only: bool) -> int:
    """Free GCP vCPUs. Remote-checker agent VMs may fall back to any x86 family and US region, so only the global
    CPU quota binds them; local-checker runs need T2D in us-central1."""
    region = run_json([*GCLOUD, "compute", "regions", "describe", GCP_REGION, "--format=json"])
    project = run_json([*GCLOUD, "compute", "project-info", "describe", "--format=json"])
    quotas = {q["metric"]: q["limit"] - q["usage"] for q in region["quotas"]}
    glob = {q["metric"]: q["limit"] - q["usage"] for q in project["quotas"]}
    if not t2d_only:
        return int(glob["CPUS_ALL_REGIONS"])
    return int(min(quotas["T2D_CPUS"], quotas["CPUS"], glob["CPUS_ALL_REGIONS"]))


def aws_free() -> int:
    quota = run_json([*AWS, "service-quotas", "get-service-quota", "--service-code", "ec2", "--quota-code", AWS_STANDARD_QUOTA])
    cpus = run_json([*AWS, "ec2", "describe-instances", "--filters", "Name=instance-state-name,Values=pending,running",
                     "--query", "Reservations[].Instances[].CpuOptions"])
    return int(quota["Quota"]["Value"]) - sum(c["CoreCount"] * c["ThreadsPerCore"] for c in cpus)


def status_path(name: str) -> Path:
    return ROOT / "logs" / name / "cloud-status"


def last_state(name: str) -> str:
    path = status_path(name)
    if not path.exists():
        return "no VM was created, see driver.log"
    lines = path.read_text().splitlines()
    return lines[-1].split(None, 1)[1] if lines else "?"


def schedule(rows: list[Row], free: dict[str, int], launch, max_runs: int = 0) -> dict[str, int]:
    """Start each row (in file order, with backfill) once a candidate cloud has room; release its vCPUs when it
    ends. `launch(row, machine)` runs one row and returns its exit code. Rows that still do not fit once nothing
    is running get exit code 1 without being run."""
    results: dict[str, int] = {}
    pending = list(rows)
    running = {}  # future -> (row, cloud, vcpus)
    with ThreadPoolExecutor(max(len(rows), 1)) as pool:
        while pending or running:
            for row in list(pending):
                if max_runs and len(running) >= max_runs:
                    break
                if pick := choose(row, free):
                    cloud, machine = pick
                    free[cloud] -= vcpus(machine)
                    pending.remove(row)
                    running[pool.submit(launch, row, machine)] = (row, cloud, vcpus(machine))
                    print(f"{datetime.now():%H:%M:%S} {row.name}: start on {cloud} {machine}", flush=True)
            if not running:  # nothing left that could free room
                for row in pending:
                    results[row.name] = 1
                    print(f"{row.name}: does not fit in the free vCPUs ({free}); not run", flush=True)
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                row, cloud, n = running.pop(f)
                free[cloud] += n
                results[row.name] = f.result()
                print(f"{datetime.now():%H:%M:%S} {row.name}: exit {results[row.name]} ({last_state(row.name)})", flush=True)
    return results


def dry_run(rows: list[Row], free: dict[str, int], max_runs: int = 0) -> list[tuple[str, str, str, str]]:
    """What `schedule` would do, assuming runs end in the order they started (durations are unknown):
    (name, cloud, machine, when) per row in start order; when = "at launch", "after <name>" or "never fits"."""
    free = dict(free)
    out, pending, running, after = [], list(rows), [], "at launch"
    while pending:
        for row in list(pending):
            if max_runs and len(running) >= max_runs:
                break
            if pick := choose(row, free):
                cloud, machine = pick
                free[cloud] -= vcpus(machine)
                pending.remove(row)
                running.append((row.name, cloud, vcpus(machine)))
                out.append((row.name, cloud, machine, after))
        if not running:
            out += [(row.name, "-", "-", "never fits") for row in pending]
            break
        name, cloud, n = running.pop(0)
        free[cloud] += n
        after = f"after {name}"
    return out


def plan(args: argparse.Namespace) -> None:
    rows = parse_plan(Path(args.plan).read_text())
    for warning in time_limit_warnings(rows):
        print(f"warning: {warning}", flush=True)
    if args.dry_run:
        free = {"gcp": gcp_free(t2d_only=not all(is_remote(r.args) for r in rows)), "aws": aws_free()}
        print(f"{len(rows)} runs; free vCPUs now: GCP {free['gcp']}, AWS {free['aws']} (nothing is launched)")
        for i, (name, cloud, machine, when) in enumerate(dry_run(rows, free, args.max_runs), 1):
            print(f"  {i:2} {name:30} {cloud:4} {machine:16} {when}")
        print("Order after the first wave assumes runs end in the order they started.")
        return
    if used := [r.name for r in rows if status_path(r.name).exists()]:
        raise SystemExit(f"run names already used (logs/<name>/cloud-status exists): {used}")
    script = ["--script", args.script] if args.script else []
    free = {"gcp": gcp_free(t2d_only=not all(is_remote(r.args) for r in rows)), "aws": aws_free()}
    print(f"{len(rows)} runs, ref {args.ref}; free vCPUs now: GCP {free['gcp']}, AWS {free['aws']}", flush=True)

    def launch(row: Row, machine: str) -> int:
        log = ROOT / "logs" / row.name / "driver.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", "scripts/cloud/run_sample.sh", *script, row.name, machine, args.ref, "--", *row.args]
        with log.open("a") as f:
            return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode

    results = schedule(rows, free, launch, args.max_runs)
    failed = [n for n, rc in results.items() if rc != 0]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed {failed}")
    if left := [vm for vm in live_vms() if vm["role"] != "scorer"]:
        print(f"agent VMs still up: {[vm['name'] for vm in left]}; check `fleet.py status` / `fleet.py cleanup`")
        sys.exit(1)
    sys.exit(1 if failed else 0)


def live_vms() -> list[dict]:
    """Every tool-labelled VM on both clouds: name, cloud, zone, machine, state, created (aware datetime), role."""
    vms = []
    for vm in run_json([*GCLOUD, "compute", "instances", "list", f"--filter=labels.tool={TOOL}", "--format=json"]):
        vms.append({"name": vm["name"], "id": vm["name"], "cloud": "gcp", "zone": vm["zone"].rsplit("/", 1)[1],
                    "machine": vm["machineType"].rsplit("/", 1)[1], "state": vm["status"],
                    "created": datetime.fromisoformat(vm["creationTimestamp"]),
                    "role": vm.get("labels", {}).get("role", "")})
    instances = run_json([*AWS, "ec2", "describe-instances", "--filters", f"Name=tag:tool,Values={TOOL}",
                          "Name=instance-state-name,Values=pending,running,stopping,stopped",
                          "--query", "Reservations[].Instances[]"])
    for i in instances:
        tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
        vms.append({"name": tags.get("Name", i["InstanceId"]), "id": i["InstanceId"], "cloud": "aws",
                    "zone": i["Placement"]["AvailabilityZone"], "machine": i["InstanceType"], "state": i["State"]["Name"],
                    "created": datetime.fromisoformat(i["LaunchTime"]), "role": tags.get("role", "")})
    return vms


def hours_since(start: datetime, end: datetime | None = None) -> float:
    return ((end or datetime.now(timezone.utc)) - start).total_seconds() / 3600


def cost(machine: str, hours: float) -> str:
    return f"${HOURLY_USD[machine] * hours:.2f}" if machine in HOURLY_USD else "unpriced"


def stamp(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def status(_: argparse.Namespace) -> None:
    vms = live_vms()
    print(f"live VMs tagged tool={TOOL}: {len(vms)}")
    total = 0.0
    for vm in vms:
        h = hours_since(vm["created"])
        total += HOURLY_USD.get(vm["machine"], 0) * h
        print(f"  {vm['cloud']} {vm['name']:34} {vm['zone']:16} {vm['machine']:16} {vm['role']:11} {vm['state']:10} up {h:5.2f} h  ~{cost(vm['machine'], h)}")
    print(f"  estimated compute so far: ~${total:.2f} (on-demand estimates, see HOURLY_USD)", flush=True)
    subprocess.run(["bash", "scripts/cloud/scorer.sh", "status"], cwd=ROOT)

    rows = []
    for f in sorted((ROOT / "logs").glob("*/cloud-status")):
        lines = [line.split(None, 2) for line in f.read_text().splitlines()]
        created = next((ln for ln in lines if ln[1] == "created"), None)
        if not created:
            continue
        machine = re.search(r"machine=(\S+)", created[2]).group(1)
        deleted = next((ln for ln in lines if ln[1] in ("deleted", "delete-requested")), None)
        h = hours_since(stamp(created[0]), stamp(deleted[0]) if deleted else None)
        rows.append((f.parent.name, machine, " ".join(lines[-1][1:])[:30], f"{h:.2f}", cost(machine, h)))
    if rows:
        print("runs (logs/*/cloud-status): name, machine, last state, VM-hours, estimated compute")
        for row in rows:
            print(f"  {row[0]:36} {row[1]:16} {row[2]:30} {row[3]:>6} {row[4]}")


def cleanup(args: argparse.Namespace) -> None:
    vms = live_vms()
    for vm in vms:
        print(f"  {vm['cloud']} {vm['name']} ({vm['id']}, {vm['zone']}, {vm['role']}, {vm['state']})")
    if not args.yes:
        print(f"{len(vms)} VMs; rerun with --yes to delete them and take the scorer down")
        return
    by_zone: dict[str, list[str]] = {}
    for vm in vms:
        if vm["cloud"] == "gcp":
            by_zone.setdefault(vm["zone"], []).append(vm["id"])
    for zone, names in by_zone.items():
        subprocess.run([*GCLOUD, "compute", "instances", "delete", *names, "--zone", zone, "--delete-disks=all"], check=True)
    if ids := [vm["id"] for vm in vms if vm["cloud"] == "aws" and vm["role"] != "scorer"]:
        subprocess.run([*AWS, "ec2", "terminate-instances", "--instance-ids", *ids], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["bash", "scripts/cloud/scorer.sh", "down"], cwd=ROOT, check=True)
    print(f"left: {[vm['name'] for vm in live_vms() if vm['state'] not in ('shutting-down', 'terminated')]}")


def main() -> None:
    if sys.argv[1:2] == ["scorer"]:  # passed through as is
        sys.exit(subprocess.run(["bash", "scripts/cloud/scorer.sh", *scorer_args(sys.argv[2:])], cwd=ROOT).returncode)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("plan")
    pp.add_argument("--ref", required="--dry-run" not in sys.argv, help="git branch, tag or commit on GitHub")
    pp.add_argument("--script", help="run this script instead of the runner (see run_sample.sh)")
    pp.add_argument("--max-runs", type=int, default=0, help="cap on runs at once (default: only vCPU room)")
    pp.add_argument("--dry-run", action="store_true", help="print where and in which order rows would start; launch nothing")
    pp.set_defaults(fn=plan)
    sub.add_parser("status").set_defaults(fn=status)
    pc = sub.add_parser("cleanup")
    pc.add_argument("--yes", action="store_true")
    pc.set_defaults(fn=cleanup)
    sub.add_parser("scorer", help="up|down|status, see scripts/cloud/scorer.sh")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
