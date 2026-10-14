"""Run published AlgoTune solvers against the Harbor task code: per-instance validity on many fresh
instances (rare-failure check) and a rough local speedup on the first instances.
usage: python -I runpub.py <harbor dir> <task> <out.json> <n_valid> <n_time> <cap_s> <solver.py>...
"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
harbor_dir, task_name, out_path = sys.argv[1:4]
N_VALID, N_TIME, CAP = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
solver_paths = sys.argv[7:]

import importlib.util
import json
import logging
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
spec = importlib.util.spec_from_file_location("evaluator", os.path.join(harbor_dir, "tests", "evaluator.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
task = mod.Task()
SIZE = int(open(os.path.join(harbor_dir, "tests", "test_outputs.py")).read().split("PROBLEM_SIZE = ")[1].split()[0])

solvers = {}
load_errors = {}
for k, p in enumerate(solver_paths):
    label = p.split("/results/")[1].split("/")[0]
    try:
        s = importlib.util.spec_from_file_location(f"pub_solver_{k}", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        solvers[label] = m.Solver()
    except Exception as e:
        load_errors[label] = f"{type(e).__name__}: {e}"[:300]

stats = {lab: {"n": 0, "invalid": [], "exceptions": [], "t_ref": 0.0, "t_sol": 0.0, "n_timed": 0} for lab in solvers}
t0 = time.time()
n_done = 0
for i in range(N_VALID):
    if time.time() - t0 > CAP:
        break
    seed = 7_000_000 + i
    problem = task.generate_problem(n=SIZE, random_seed=seed)
    timed = i < N_TIME
    if timed:
        task.solve(problem)
        a = time.perf_counter(); task.solve(problem); b = time.perf_counter(); task.solve(problem); c = time.perf_counter()
        t_ref = min(b - a, c - b)
    for lab, sol in solvers.items():
        st = stats[lab]
        st["n"] += 1
        try:
            out = sol.solve(problem)
            ok = bool(task.is_solution(problem, out))
        except Exception as e:
            st["exceptions"].append([seed, f"{type(e).__name__}: {e}"[:160]])
            continue
        if not ok:
            st["invalid"].append(seed)
        if timed:
            try:
                a = time.perf_counter(); sol.solve(problem); b = time.perf_counter(); sol.solve(problem); c = time.perf_counter()
                st["t_sol"] += min(b - a, c - b); st["t_ref"] += t_ref; st["n_timed"] += 1
            except Exception:
                pass
    n_done += 1

out = {"task": task_name, "n_instances": n_done, "load_errors": load_errors, "elapsed_s": time.time() - t0, "solvers": {}}
for lab, st in stats.items():
    out["solvers"][lab] = {
        "n": st["n"], "n_invalid": len(st["invalid"]), "n_exceptions": len(st["exceptions"]),
        "invalid_seeds": st["invalid"][:50], "exceptions": st["exceptions"][:5],
        "local_speedup": (st["t_ref"] / st["t_sol"]) if st["t_sol"] > 0 else None, "n_timed": st["n_timed"],
    }
json.dump(out, open(out_path, "w"), indent=0)
print(task_name, n_done, {k: (v["n_invalid"], v["n_exceptions"], v["local_speedup"] and round(v["local_speedup"], 2)) for k, v in out["solvers"].items()}, load_errors)
