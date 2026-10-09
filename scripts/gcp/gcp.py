"""Run a plan of samples on GCE VMs (one VM each, via run_sample.sh), show live VMs and their cost, clean up.

Usage:
  uv run python scripts/gcp/gcp.py plan <plan.tsv> --ref <git-ref> [--concurrency 4] [--script <path>]
  uv run python scripts/gcp/gcp.py status
  uv run python scripts/gcp/gcp.py cleanup [--yes]

Plan file: one run per line, `name  machine-type  runner args...` (tab or space separated; the args are split
like a shell would), blank lines and # comments ignored. Every run is launched with
`run_sample.sh [--script <path>] <name> <machine-type> <ref> -- <args>`, at most --concurrency at once; its
output goes to logs/<name>/gcp-driver.log and its progress to logs/<name>/gcp-status. Keep --concurrency within
the T2D vCPU quota (24 vCPUs = one t2d-standard-16 on 2026-10-09). Run a long plan under bgjob.

cleanup lists every VM labelled tool=swarm-gcp-runner and, with --yes, deletes them (running plans included).
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = dict(re.findall(r"^(\w+)=([^\s$]+)", (ROOT / "scripts/gcp/common.sh").read_text(), re.M))
GCLOUD = ["gcloud", f"--account={COMMON['GCP_ACCOUNT']}", f"--project={COMMON['GCP_PROJECT']}", "--quiet"]
LABEL = f"labels.tool={COMMON['TOOL_LABEL']}"

# On-demand compute USD per hour in us-central1, from third-party price trackers (July 2026). Estimates only:
# they leave out the 60 GB pd-balanced disk (about $0.008/h) and the external IP (about $0.005/h). Check billing.
HOURLY_USD = {"t2d-standard-16": 0.68, "t2d-standard-32": 1.35, "e2-standard-8": 0.27}


def gcloud_json(*args: str) -> list[dict]:
    out = subprocess.run([*GCLOUD, *args, "--format=json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def live_vms() -> list[dict]:
    return gcloud_json("compute", "instances", "list", f"--filter={LABEL}")


def parse_plan(path: Path) -> list[tuple[str, str, list[str]]]:
    runs = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, machine, *rest = line.split(None, 2)
        runs.append((name, machine, shlex.split(rest[0]) if rest else []))
    names = [r[0] for r in runs]
    if dupes := {n for n in names if names.count(n) > 1}:
        raise SystemExit(f"duplicate run names: {sorted(dupes)}")
    if used := [n for n in names if (ROOT / "logs" / n / "gcp-status").exists()]:
        raise SystemExit(f"run names already used (logs/<name>/gcp-status exists): {used}")
    return runs


def last_state(name: str) -> str:
    path = ROOT / "logs" / name / "gcp-status"
    if not path.exists():
        return "no VM was created, see gcp-driver.log"
    lines = path.read_text().splitlines()
    return lines[-1].split(None, 1)[1] if lines else "?"


def plan(args: argparse.Namespace) -> None:
    runs = parse_plan(Path(args.plan))
    script = ["--script", args.script] if args.script else []

    def launch(name: str, machine: str, runner_args: list[str]) -> int:
        log = ROOT / "logs" / name / "gcp-driver.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", "scripts/gcp/run_sample.sh", *script, name, machine, args.ref, "--", *runner_args]
        with log.open("a") as f:
            return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode

    print(f"{len(runs)} runs, ref {args.ref}, concurrency {args.concurrency}", flush=True)
    results = {}
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(launch, *run): run[0] for run in runs}
        try:
            for f in as_completed(futures):
                name = futures[f]
                results[name] = f.result()
                print(f"{datetime.now():%H:%M:%S} {name}: exit {results[name]} ({last_state(name)})", flush=True)
        except KeyboardInterrupt:
            pool.shutdown(cancel_futures=True)  # started runs get the SIGINT too and delete their VMs
            raise
    failed = [n for n, rc in results.items() if rc != 0]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed {failed}")
    names = {r[0] for r in runs}
    if left := [vm["name"] for vm in live_vms() if vm.get("labels", {}).get("run", "") in {vm_name(n) for n in names}]:
        print(f"VMs still up for this plan: {left}; run `gcp.py cleanup --yes`")
        sys.exit(1)
    sys.exit(1 if failed else 0)


def vm_name(run: str) -> str:
    out = subprocess.run(["bash", "-c", 'source scripts/gcp/common.sh && vm_name "$1"', "_", run],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def hours_since(stamp: str, end: datetime | None = None) -> float:
    start = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return ((end or datetime.now(timezone.utc)) - start).total_seconds() / 3600


def cost(machine: str, hours: float) -> str:
    return f"${HOURLY_USD[machine] * hours:.2f}" if machine in HOURLY_USD else "unpriced"


def status(_: argparse.Namespace) -> None:
    vms = live_vms()
    print(f"live VMs labelled tool={COMMON['TOOL_LABEL']}: {len(vms)}")
    total = 0.0
    for vm in vms:
        machine = vm["machineType"].rsplit("/", 1)[1]
        h = hours_since(vm["creationTimestamp"])
        total += HOURLY_USD.get(machine, 0) * h
        print(f"  {vm['name']:40} {vm['zone'].rsplit('/', 1)[1]:15} {machine:16} {vm['status']:11} up {h:5.2f} h  ~{cost(machine, h)}")
    print(f"  estimated compute so far: ~${total:.2f} (on-demand estimates, see HOURLY_USD)")

    rows = []
    for f in sorted((ROOT / "logs").glob("*/gcp-status")):
        lines = [line.split(None, 2) for line in f.read_text().splitlines()]
        if not lines:
            continue
        created = next((ln for ln in lines if ln[1] == "created"), None)
        if not created:
            rows.append((f.parent.name, lines[-1][1], "-", "-"))
            continue
        machine = re.search(r"machine=(\S+)", created[2]).group(1)
        deleted = next((ln for ln in lines if ln[1] in ("deleted", "delete-requested")), None)
        end = datetime.fromisoformat(deleted[0].replace("Z", "+00:00")) if deleted else None
        h = hours_since(created[0], end)
        rows.append((f.parent.name, " ".join(lines[-1][1:]), f"{h:.2f}", cost(machine, h)))
    if rows:
        print("runs (logs/*/gcp-status): name, last state, VM-hours, estimated compute")
        for row in rows:
            print(f"  {row[0]:40} {row[1]:30} {row[2]:>6} {row[3]}")


def cleanup(args: argparse.Namespace) -> None:
    vms = live_vms()
    for vm in vms:
        print(f"  {vm['name']} ({vm['zone'].rsplit('/', 1)[1]}, {vm['status']})")
    if not vms:
        print("no VMs to clean up")
        return
    if not args.yes:
        print(f"{len(vms)} VMs; rerun with --yes to delete them")
        return
    by_zone: dict[str, list[str]] = {}
    for vm in vms:
        by_zone.setdefault(vm["zone"].rsplit("/", 1)[1], []).append(vm["name"])
    for zone, names in by_zone.items():
        subprocess.run([*GCLOUD, "compute", "instances", "delete", *names, "--zone", zone, "--delete-disks=all"], check=True)
    print(f"left: {[vm['name'] for vm in live_vms()]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("plan")
    pp.add_argument("--ref", required=True, help="git branch, tag or commit on GitHub")
    pp.add_argument("--concurrency", type=int, default=1)
    pp.add_argument("--script", help="run this script instead of the runner (see run_sample.sh)")
    pp.set_defaults(fn=plan)
    sub.add_parser("status").set_defaults(fn=status)
    pc = sub.add_parser("cleanup")
    pc.add_argument("--yes", action="store_true")
    pc.set_defaults(fn=cleanup)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
