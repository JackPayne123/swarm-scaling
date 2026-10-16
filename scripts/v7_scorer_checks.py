"""v7 scorer checks (PLAN.md, "v7: task screen and strata, pre-registered"): the A6 checks and a reference validity pass
for every candidate task, on a live scorer serving experiment v7, with the pre-registered verdicts.

Per task, all submitted at once (the scorer's FIFO queue orders them):
  ref-0..2   score jobs of the reference solver (A6i, reference against itself)
  fast-0..2  score jobs of the task's known fast solver, <solvers>/<task>.py (A6ii)
  dev        a final_eval job of the fast solver on finalize's stage-1 instances (n 20, seed FINAL_DEV_SEED) (A6iv)
  validity   a validity job of the reference on VALIDITY_SEED + i, i < 1,000 (300 for generalized_eigenvalues_complex)
and then fast-3..5 for every task whose fast-solver spread exceeds A6II_SPREAD (A6iii).

Verdicts (the pre-registered thresholds below; scores are the primary alone scores, raw where noted):
  A6i    exclude if the reference's 3 alone raw speedups differ by more than A6I_SPREAD (max/min)
  A6ii   exclude if any fast-solver job has reference_inflation outside INFLATION_BAND, or a first_call_ratio above
         FIRST_CALL_MAX (flagged for reading instead when the solver compiles on its first call: numba, pyximport)
  A6iii  the fast solver's 3 scores within A6II_SPREAD pass; otherwise 3 more jobs, and exclude if the 6 differ by more
         than A6III_SPREAD (this covers two timing modes 1.25x apart; whether they looked bimodal is reported)
  A6iv   flag (never exclude) when the dev score and the hidden interleaved median differ by more than DEV_HIDDEN_LOG
  ref validity  exclude if the reference fails any validity instance; its run time sets the main run's validity size
         (VALIDITY_RUN_S: n is cut in proportion, to a multiple of 100, when 1,000 instances take longer)

A failed job is rerun once. Every finished job is appended to <out>/jobs.jsonl; request ids are fixed per task and job, so a rerun against the same
scorer gets its jobs back and skips the ones already recorded. <out>/verdicts.json and a table on stdout at the end.

  SCORER_URL=http://<ip>:8770 SCORER_TOKEN=... uv run python scripts/v7_scorer_checks.py \\
      --solvers analysis/v7-prep/fast_solvers --out analysis/v7-prep/scorer-checks [--tasks a,b]
"""

import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import anyio

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import FINAL_DEV_N, FINAL_DEV_SEED, VALIDITY_SEED
from swarm_scaling.tasks import PROJECT_ROOT

A6I_SPREAD = 1.05
A6II_SPREAD = 1.10
A6III_SPREAD = 1.25
INFLATION_BAND = (0.87, 1.15)
FIRST_CALL_MAX = 5.0
DEV_HIDDEN_LOG = 0.25
VALIDITY_N = 1000
VALIDITY_N_TASK = {"generalized_eigenvalues_complex": 300}  # 1,000 would pass the job's 3,600 s limit (Mac, 2026-10-14)
VALIDITY_RUN_S = 1800  # half the validity job's limit: a candidate's checks must fit with room to spare
CONCURRENCY = 40  # jobs in flight from this client (each polls every 2 s)
COMPILES_ON_FIRST_CALL = re.compile(r"numba|pyximport|@jit|njit", re.I)

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


def raw(rec: dict, pattern: str) -> float | None:
    m = re.search(pattern, (rec.get("result") or {}).get("verifier_stdout", ""), re.M)
    return float(m.group(1)) if m else None


def summary(rec: dict) -> dict:
    """The fields a verdict reads from one finished job."""
    r = rec.get("result") or {}
    out = {"status": rec["status"], "error": rec.get("error"), "run_s": rec.get("run_s"), "slot": rec.get("slot")}
    if rec["kind"] in ("score",):
        out |= {"score": r.get("score"), "alone_score": r.get("alone_score"), "valid": r.get("valid"),
                "raw": raw(rec, r"^Raw Speedup:\s+([\d.]+) x"), "alone_raw": raw(rec, r"^Alone raw speedup: ([\d.]+) x"),
                "reference_inflation": r.get("reference_inflation"), "first_call_ratio": r.get("first_call_ratio"),
                "scoring_timeout": r.get("scoring_timeout")}  # fmt: skip
    elif rec["kind"] == "final_eval":
        out |= {k: r.get(k) for k in ("valid", "speedup", "alone_speedup", "reference_inflation", "first_call_ratio")}
    elif rec["kind"] == "validity":
        out |= {k: r.get(k) for k in ("n", "checked", "n_invalid", "valid", "errors")}
    return out


def spread(values: list[float]) -> float:
    return max(values) / min(values)


def verdict(task: str, jobs: dict[str, dict], fast_source: str) -> dict:
    """The pre-registered verdict for one task from its jobs (label -> summary)."""
    v: dict = {"task": task, "exclude": [], "flags": []}
    refs = [jobs.get(f"ref-{k}") for k in range(3)]
    fasts = [jobs[f"fast-{k}"] for k in range(6) if f"fast-{k}" in jobs]
    failed = [lab for lab, j in jobs.items() if j["status"] != "done"]
    if failed:
        v["exclude"].append(f"jobs failed: {failed}")
        return v
    alone_raw = [r["alone_raw"] for r in refs]
    v["a6i"] = {"alone_raw": alone_raw, "raw": [r["raw"] for r in refs], "spread": spread(alone_raw)}
    if v["a6i"]["spread"] > A6I_SPREAD:
        v["exclude"].append(f"A6i: reference against itself spread {v['a6i']['spread']:.3f}")
    scores = [f["alone_score"] for f in fasts]
    v["a6ii"] = {"alone_score": scores, "score": [f["score"] for f in fasts], "spread3": spread(scores[:3]),
                 "inflation": [f["reference_inflation"] for f in fasts], "first_call": [f["first_call_ratio"] for f in fasts]}  # fmt: skip
    bad_inflation = [x for x in v["a6ii"]["inflation"] if x is None or not INFLATION_BAND[0] <= x <= INFLATION_BAND[1]]
    if bad_inflation:
        v["exclude"].append(f"A6ii: reference inflation {bad_inflation} outside {INFLATION_BAND}")
    high = [x for x in v["a6ii"]["first_call"] if x is not None and x > FIRST_CALL_MAX]
    if high:
        if COMPILES_ON_FIRST_CALL.search(fast_source):
            v["flags"].append(f"A6ii: first-call ratio {max(high):.1f} from a solver that compiles on its first call")
        else:
            v["exclude"].append(f"A6ii: first-call ratio {max(high):.1f}")
    if v["a6ii"]["spread3"] > A6II_SPREAD:
        if len(scores) < 6:
            v["needs_a6iii"] = True
            return v
        v["a6iii"] = {"spread6": spread(scores), "sorted": sorted(scores)}
        if v["a6iii"]["spread6"] > A6III_SPREAD:
            v["exclude"].append(f"A6iii: fast solver spread {v['a6iii']['spread6']:.3f} over 6 scorings")
    dev, hidden = jobs["dev"], statistics.median(f["score"] for f in fasts[:3])
    v["a6iv"] = {"dev_speedup": dev["speedup"], "hidden_median": hidden,
                 "log_ratio": None if not dev["speedup"] else math.log(dev["speedup"] / hidden)}  # fmt: skip
    if v["a6iv"]["log_ratio"] is None or abs(v["a6iv"]["log_ratio"]) > DEV_HIDDEN_LOG:
        v["flags"].append(f"A6iv: dev {dev['speedup']} against hidden median {hidden:.2f}")
    val = jobs["validity"]
    v["validity"] = {"n": val["n"], "n_invalid": val["n_invalid"], "run_s": val["run_s"]}
    if val["n_invalid"]:
        v["exclude"].append(f"reference invalid on {val['n_invalid']} of {val['n']} validity instances")
    n = VALIDITY_N_TASK.get(task, VALIDITY_N)
    v["validity_n_main"] = n if val["run_s"] <= VALIDITY_RUN_S else max(100, int(n * VALIDITY_RUN_S / val["run_s"]) // 100 * 100)
    return v


async def main_async(args: argparse.Namespace) -> None:
    health = scorer_client.health()
    if health.get("experiment") != "v7":
        raise SystemExit(f"the scorer serves {health.get('experiment')}, not v7")
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "jobs.jsonl"
    done = {(j["task"], j["label"]): j for j in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    tasks = args.tasks.split(",") if args.tasks else sorted(p.stem for p in args.solvers.glob("*.py"))
    sources = {t: (args.solvers / f"{t}.py").read_text() for t in tasks}
    limit = anyio.Semaphore(CONCURRENCY)

    async def run(task: str, label: str, kind: str, source: str, **job_args) -> None:
        if (task, label) in done:
            return
        for attempt in (1, 2):  # a failed job (crash, timeout, out of memory) is rerun once
            async with limit:
                rec = await scorer_client.run_job({
                    "kind": kind, "task": task, "solver_source": source, "run_id": "v7-scorer-checks", "agent_id": label,
                    "sample_id": task, "request_id": f"v7chk-{health['seed_offset_id']}-{task}-{label}-a{attempt}", **job_args,
                })  # fmt: skip
            if rec["status"] == "done":
                break
        row = {"task": task, "label": label, "kind": kind, "attempts": attempt, **summary({**rec, "kind": kind})}
        done[(task, label)] = row
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{task} {label}: {row['status']} {json.dumps({k: row.get(k) for k in ('alone_score', 'score', 'speedup', 'n_invalid', 'run_s')})}", flush=True)

    async with anyio.create_task_group() as tg:
        for t in tasks:
            for k in range(3):
                tg.start_soon(run, t, f"ref-{k}", "score", REFERENCE_SOLVER)
            for k in range(3):
                tg.start_soon(run, t, f"fast-{k}", "score", sources[t])
            tg.start_soon(lambda t=t: run(t, "dev", "final_eval", sources[t], n=FINAL_DEV_N, seed=FINAL_DEV_SEED, reps=10))
            tg.start_soon(lambda t=t: run(t, "validity", "validity", REFERENCE_SOLVER,
                                          n=VALIDITY_N_TASK.get(t, VALIDITY_N), seed=VALIDITY_SEED))  # fmt: skip
    jobs = {t: {lab: j for (tt, lab), j in done.items() if tt == t} for t in tasks}
    again = [t for t in tasks if verdict(t, jobs[t], sources[t]).get("needs_a6iii")]
    async with anyio.create_task_group() as tg:
        for t in again:
            for k in range(3, 6):
                tg.start_soon(run, t, f"fast-{k}", "score", sources[t])
    jobs = {t: {lab: j for (tt, lab), j in done.items() if tt == t} for t in tasks}
    verdicts = {t: verdict(t, jobs[t], sources[t]) for t in tasks}
    (args.out / "verdicts.json").write_text(json.dumps({"health": health, "verdicts": verdicts}, indent=1))
    for t, v in verdicts.items():
        state = "EXCLUDE" if v["exclude"] else "pass"
        print(f"{t:36s} {state:7s} ref spread {v.get('a6i', {}).get('spread', float('nan')):.3f} fast "
              f"{[round(x, 2) for x in v.get('a6ii', {}).get('alone_score', [])]} {'; '.join(v['exclude'] + v['flags'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solvers", type=Path, required=True, help="directory of <task>.py fast solvers; one task each")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", help="comma-separated subset (default: every solver in --solvers)")
    args = p.parse_args()
    scorer_client.on_unreachable = lambda: subprocess.run(["bash", "scripts/cloud/scorer.sh", "allow-me"], cwd=PROJECT_ROOT)
    anyio.run(main_async, args)


if __name__ == "__main__":
    main()
