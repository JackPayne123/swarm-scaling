"""Scorer measurements for the v7b fixes (analysis/v7b-fixes.md), against a live scorer nothing else is using.

    SCORER_URL=... SCORER_TOKEN=... uv run python analysis/v7b-smoke/measure_scorer.py <logs dir with the v7c-* runs> <out.jsonl>

Sources are the v7 calibration's ode picks, read from the runs' finalize sources by sha256, plus output-handling
variants made by one textual edit each:
  mix_pool      the mixed team's pick (8 threads; returns a pooled list whose float objects it rewrites in place)
  mix_nopool    the same with USE_POOL = 0 (a fresh list of fresh floats on every call)
  team_array    the ode team's pick (returns array.array('d'))
  team_tolist   the same returning out.tolist()
  team_ring     the same returning its C make_list (a list whose cached float objects it rewrites in place)
  indep_pick    the independents' pick (8 threads, fresh list)
Phases (dev_eval n=20, reps=10, seed 0 unless stated):
  validity   validity jobs (n=20) of team_array and mix_pool: is the array.array output now rejected?
  alone      mix_pool and indep_pick, one job at a time: the service hands consecutive jobs to its slots in turn
  loaded     the same, three jobs at once (one per slot), twice per source
  exclusive  every source 3 times as exclusive jobs, interleaved by source
  pinned     then dockerd, containerd and the latency holder are moved to the service's CPUs and every device IRQ with
             them (irqbalance stopped), over SSH as scorer.sh would, and alone and loaded are repeated (alone2,
             loaded2): does keeping the host's own work off the slots remove the slot differences?
Each finished job is one JSON line: phase, source, slot, session, start/end, total_solver_s, total_reference_s,
speedup, alone_speedup, valid, n_invalid and the first error.
"""

import asyncio
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from inspect_ai.log import read_eval_log

from swarm_scaling import scorer_client

PICKS = {  # the calibration's portfolio picks (analysis/v7-calibration.md, section 1)
    "mix_pool": ("v7c-ode_lorenz96_nonchaotic-r1-mixteam", "992dd7889286"),
    "team_array": ("v7c-ode_lorenz96_nonchaotic-r1-divteam", "59246960f8ad"),
    "indep_pick": ("v7c-ode_lorenz96_nonchaotic-r1-div-i0", "af484b4c20e8"),
}
TEAM_RETURN = '''            if lib.solve_into(y0, float(problem["F"]), float(problem["t0"]), float(problem["t1"]), 1e-8, 1e-8, self.nt,
                              out.buffer_info()[0]) == 0:
                return out'''
TEAM_RING = '''            res = lib.solve_py(y0, float(problem["F"]), float(problem["t0"]), float(problem["t1"]), 1e-8, 1e-8, self.nt)
            if res is not None:
                return res'''


def sources(logs: Path) -> dict[str, str]:
    out = {}
    for name, (run, prefix) in PICKS.items():
        for path in sorted(glob.glob(f"{logs}/{run}/*.eval"), key=os.path.getmtime, reverse=True):
            found = {k: v for k, v in read_eval_log(path).samples[0].metadata["finalize"]["sources"].items() if k.startswith(prefix)}
            if found:
                out[name] = next(iter(found.values()))
                break
    assert out["mix_pool"].count("USE_POOL = 1\n") == 1 and out["team_array"].count(TEAM_RETURN) == 1
    out["mix_nopool"] = out["mix_pool"].replace("USE_POOL = 1\n", "USE_POOL = 0\n")
    out["team_tolist"] = out["team_array"].replace(TEAM_RETURN, TEAM_RETURN.replace("return out", "return out.tolist()"))
    out["team_ring"] = out["team_array"].replace(TEAM_RETURN, TEAM_RING)
    return out


async def main() -> None:
    src = sources(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    health = scorer_client.health()
    print(f"scorer {health.get('machine_type')} {health['cpu_model']} slots {health['slots']} session {health.get('session')}", flush=True)

    async def one(phase: str, name: str, kind: str = "dev_eval", exclusive: bool = False, **args) -> dict:
        body = {"kind": kind, "task": "ode_lorenz96_nonchaotic", "solver_source": src[name], "run_id": "v7b-measure",
                "agent_id": name, "sample_id": phase, "exclusive": exclusive, **args}
        rec = await scorer_client.run_job(body)
        res = rec.get("result") or {}
        row = {"phase": phase, "source": name, "kind": kind, "slot": rec.get("slot"), "session": rec.get("session"),
               "exclusive": rec.get("exclusive"), "started_at": rec.get("started_at"), "ended_at": rec.get("ended_at"),
               "status": rec["status"], "error": rec.get("error"), "valid": res.get("valid"), "n_invalid": res.get("n_invalid"),
               "first_error": (res.get("errors") or [None])[0], "total_solver_s": res.get("total_solver_s"),
               "total_reference_s": res.get("total_reference_s"), "speedup": res.get("speedup"),
               "alone_speedup": res.get("alone_speedup"), "output_tail": (rec.get("output") or "")[-600:]}  # fmt: skip
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {phase:9} {name:11} slot {row['slot']} valid {row['valid']} solver {row['total_solver_s']} "
              f"alone {row['alone_speedup']} {row['first_error'] or ''}"[:220], flush=True)  # fmt: skip
        return row

    dev = {"n": 20, "seed": 0, "reps": 10}
    await asyncio.gather(*(one("validity", s, kind="validity", n=20, seed=600_000) for s in ("team_array", "mix_pool")))
    for s in ("mix_pool", "indep_pick"):
        for _ in range(6):
            await one("alone", s, **dev)
        for _ in range(2):
            await asyncio.gather(*(one("loaded", s, **dev) for _ in range(3)))
    for _ in range(3):
        for s in ("mix_pool", "mix_nopool", "team_tolist", "team_ring", "indep_pick"):
            await one("exclusive", s, exclusive=True, **dev)
    service = "24-31"  # c7a.8xlarge with 3 slots: slots 0-7, 8-15, 16-23
    pin = (f"sudo systemctl stop irqbalance; for f in /proc/irq/*/smp_affinity_list; do echo {service} | sudo tee $f >/dev/null 2>&1; done; "
           f"sudo systemctl set-property --runtime docker.service AllowedCPUs={service}; "
           f"sudo systemctl set-property --runtime containerd.service AllowedCPUs={service}; "
           f"for p in $(pgrep -f hold_cpu_dma_latency); do sudo taskset -apc {service} $p >/dev/null; done; "
           "ps -eo psr,comm | grep -E 'dockerd|containerd|python' ; grep -E 'nvme|ena|enp' /proc/interrupts | head -4")  # fmt: skip
    ssh = subprocess.run(["bash", "-c", 'source scripts/cloud/common.sh && ssh "${AWS_SSH_OPTS[@]}" "ubuntu@$(scorer_field ip)" "$1"', "_", pin],
                         capture_output=True, text=True, timeout=120)  # fmt: skip
    print("pinned:", ssh.returncode, ssh.stdout[-1500:], ssh.stderr[-500:], flush=True)
    for s in ("mix_pool", "indep_pick"):
        for _ in range(6):
            await one("alone2", s, **dev)
        for _ in range(2):
            await asyncio.gather(*(one("loaded2", s, **dev) for _ in range(3)))


if __name__ == "__main__":
    asyncio.run(main())
