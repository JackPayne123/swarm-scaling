"""Do a scorer's slots slow each other down? Times the same score job alone, then on every slot at once.

Usage (against a live scorer; never one that a running pilot is using):
  SCORER_URL=http://<ip>:8770 SCORER_TOKEN=... uv run python scripts/slot_interference.py \\
      [--task dst_type_II_scipy_fftpack] [--solver solver.py] [--repeats 3] [--out result.json]

Each repeat submits one score job and waits for it (alone), then submits one per slot at once (all slots busy).
Without --solver the job scores the task's reference solver (loaded from /app/dev/reference_task.py, which the
scorer mounts in every job), so the run is reference against reference. The verifier prints "Total Baseline Time"
(the reference) and "Total Solver Time"; both are reported per slot as concurrent / mean alone time. A ratio near
1.0 means the slots do not interfere. Each job's reference inflation (the reference check's ratio) and alone-baseline
mode are printed too: with the scorer's --cache-alone on, an inflation that rises when every slot is busy means the
cached baseline is not safe under load. The service's /stats are printed at the end.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from swarm_scaling import scorer_client

REFERENCE_SOLVER = '''import importlib.util
_spec = importlib.util.spec_from_file_location("reference_task", "/app/dev/reference_task.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


class Solver:
    def __init__(self):
        self.task = _ref.Task()

    def solve(self, problem):
        return self.task.solve(problem)
'''
TIMES = {"reference_s": r"^Total Baseline Time: ([\d.]+)s", "solver_s": r"^Total Solver Time:\s+([\d.]+)s"}


def timings(rec: dict) -> dict:
    """Slot, run time and the verifier's reference and solver totals of a finished score job."""
    if rec["status"] != "done":
        raise RuntimeError(f"job {rec['job_id']} ended {rec['status']}: {rec.get('error')}")
    out = rec["result"]["verifier_stdout"]
    found = {k: re.search(p, out, re.M) for k, p in TIMES.items()}
    if not all(found.values()):
        raise RuntimeError(f"job {rec['job_id']}: no timing totals in the verifier output: {out[-1000:]}")
    return {"slot": rec["slot"], "run_s": rec["run_s"], "score": rec["result"]["score"],
            "reference_inflation": rec["result"].get("reference_inflation"),
            "alone_baseline": (rec.get("alone_baseline") or {}).get("mode"),
            **{k: float(m.group(1)) for k, m in found.items()}}  # fmt: skip


def wait_all(job_ids: list[str]) -> list[dict]:
    recs: dict[str, dict] = {}
    while len(recs) < len(job_ids):
        time.sleep(scorer_client.POLL_S)
        for j in job_ids:
            if j not in recs and (rec := scorer_client.get(j))["status"] in ("done", "error"):
                recs[j] = {**rec, "job_id": j}
    return [recs[j] for j in job_ids]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="dst_type_II_scipy_fftpack")
    p.add_argument("--solver", type=Path, help="solver.py to score (default: the reference solver)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, help="write every job's timings here as JSON")
    args = p.parse_args()

    health = scorer_client.health()
    slots = health["slots"]
    source = args.solver.read_text() if args.solver else REFERENCE_SOLVER
    job = {"kind": "score", "task": args.task, "solver_source": source, "run_id": "slot-interference",
           "agent_id": "", "sample_id": ""}  # fmt: skip
    print(f"scorer {os.environ['SCORER_URL']}: {slots} slots, {health['cpu_model']}, version {health['version']}", flush=True)

    alone, together = [], []
    for r in range(args.repeats):
        (a,) = [timings(x) for x in wait_all([scorer_client.submit(job)])]
        alone.append({"repeat": r, **a})
        print(f"repeat {r} alone: slot {a['slot']} reference {a['reference_s']:.3f}s solver {a['solver_s']:.3f}s "
              f"inflation {a['reference_inflation']} ({a['alone_baseline']})", flush=True)
        batch = [timings(x) for x in wait_all([scorer_client.submit(job) for _ in range(slots)])]
        together += [{"repeat": r, **b} for b in batch]
        for b in sorted(batch, key=lambda b: b["slot"]):
            print(f"repeat {r} all slots: slot {b['slot']} reference {b['reference_s']:.3f}s solver {b['solver_s']:.3f}s "
                  f"inflation {b['reference_inflation']} ({b['alone_baseline']})", flush=True)

    base = {k: sum(a[k] for a in alone) / len(alone) for k in TIMES}
    print(f"\nalone (mean of {len(alone)}, slots {sorted({a['slot'] for a in alone})}): "
          f"reference {base['reference_s']:.3f}s, solver {base['solver_s']:.3f}s")  # fmt: skip
    print("all slots busy, mean time / mean alone time:")
    by_slot = defaultdict(list)
    for b in together:
        by_slot[b["slot"]].append(b)
    for slot, rows in sorted(by_slot.items()):
        ratios = {k: sum(x[k] for x in rows) / len(rows) / base[k] for k in TIMES}
        print(f"  slot {slot}: reference x{ratios['reference_s']:.3f}, solver x{ratios['solver_s']:.3f} ({len(rows)} jobs)")
    print(f"/stats: {json.dumps(scorer_client.request('GET', '/stats'))}")
    if args.out:
        args.out.write_text(json.dumps({"health": health, "alone": alone, "together": together}, indent=2))


if __name__ == "__main__":
    main()
