"""Model-free screen of one Harbor AlgoTune task: reference-time tail, degenerate instances,
reference self-validity, determinism, and a trivial-output / tolerance battery.

usage: python -I screen.py <harbor task dir> <task> <out.json> [n_instances] [threads] [cap_s]
"""
import os
import sys

harbor_dir, task_name, out_path = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
THREADS = sys.argv[5] if len(sys.argv) > 5 else "1"
CAP_S = float(sys.argv[6]) if len(sys.argv) > 6 else 900
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = THREADS

import copy
import importlib.util
import json
import logging
import math
import time
import traceback

import numpy as np

logging.disable(logging.CRITICAL)
spec = importlib.util.spec_from_file_location("evaluator", os.path.join(harbor_dir, "tests", "evaluator.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
task = mod.Task()
src = open(os.path.join(harbor_dir, "tests", "test_outputs.py")).read()
SIZE = int(src.split("PROBLEM_SIZE = ")[1].split()[0])
SEED0 = 9_000_000


def leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float, complex, np.number)) and not isinstance(x, bool) for x in obj):
            yield np.asarray(obj)
        else:
            for v in obj:
                yield from leaves(v)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for v in obj.ravel():
                yield from leaves(v)
        else:
            yield obj
    elif isinstance(obj, (float, complex, np.floating, np.complexfloating, int, np.integer)) and not isinstance(obj, bool):
        yield np.asarray(obj)


def finite(obj):
    for a in leaves(obj):
        if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
            return False
    return True


def fmap(obj, f, ints=False):
    """Apply f to float (and optionally int) numeric content, keeping container types."""
    if isinstance(obj, dict):
        return {k: fmap(v, f, ints) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(fmap(v, f, ints) for v in obj)
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float, complex, np.number)) and not isinstance(x, bool) for x in obj):
            a = np.asarray(obj)
            if a.dtype.kind in "fc" or (ints and a.dtype.kind in "iu"):
                return f(a).tolist()
            return list(obj)
        return [fmap(v, f, ints) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return np.array([fmap(v, f, ints) for v in obj.ravel()], dtype=object).reshape(obj.shape)
        if obj.dtype.kind in "fc" or (ints and obj.dtype.kind in "iu"):
            return f(obj)
        return obj
    if isinstance(obj, (float, np.floating, complex, np.complexfloating)) or (ints and isinstance(obj, (int, np.integer)) and not isinstance(obj, bool)):
        r = f(np.asarray(obj))
        return type(obj)(r.item()) if not isinstance(obj, np.generic) else r[()]
    return obj


rng = np.random.default_rng(0)


def perturb(eps):
    def f(a):
        a = np.asarray(a)
        s = rng.choice([-1.0, 1.0], size=a.shape)
        if a.dtype.kind == "c":
            return a * (1 + eps * s)
        return (a * (1 + eps * s)).astype(a.dtype if a.dtype.kind == "f" else float)
    return f


def f32(a):
    a = np.asarray(a)
    if a.dtype.kind == "c":
        return a.astype(np.complex64).astype(np.complex128)
    return a.astype(np.float32).astype(np.float64)


def zeros(a):
    return np.zeros_like(np.asarray(a))


def check(problem, sol):
    try:
        return bool(task.is_solution(problem, sol))
    except Exception:
        return False


def same(a, b):
    la, lb = list(leaves(a)), list(leaves(b))
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        if x.shape != y.shape:
            return False
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            if not np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


res = {"task": task_name, "size": SIZE, "threads": THREADS, "n_target": N, "instances": []}
t_start = time.time()
prev_out = None
battery = {"zeros": zeros, "f32": f32, "rel1e-6": perturb(1e-6), "rel1e-3": perturb(1e-3), "rel1e-1": perturb(1e-1)}
for i in range(N):
    if time.time() - t_start > CAP_S:
        res["capped"] = True
        break
    rec = {"seed": SEED0 + i}
    try:
        tg = time.perf_counter()
        problem = task.generate_problem(n=SIZE, random_seed=SEED0 + i)
        rec["gen_s"] = time.perf_counter() - tg
        rec["input_finite"] = finite(problem)
        pcopy = copy.deepcopy(problem)
        out = task.solve(problem)  # warmup
        times = []
        for _ in range(3):
            t0 = time.perf_counter_ns()
            out2 = task.solve(problem)
            times.append(time.perf_counter_ns() - t0)
        rec["t_ref"] = min(times) / 1e9
        rec["t_ref_max"] = max(times) / 1e9
        rec["output_finite"] = finite(out)
        rec["deterministic"] = same(out, out2)
        tv = time.perf_counter()
        rec["self_valid"] = check(problem, out)
        rec["t_check"] = time.perf_counter() - tv
        rec["self_valid_again"] = check(problem, out)
        rec["self_valid_out2"] = check(problem, out2)
        rec["input_unchanged"] = same(problem, pcopy)
        bat = {}
        if i < 30:
            for name, f in battery.items():
                try:
                    cand = fmap(out, f, ints=(name == "zeros"))
                    bat[name] = check(problem, cand)
                except Exception as e:
                    bat[name] = f"err:{type(e).__name__}"
            if prev_out is not None:
                bat["other_instance"] = check(problem, prev_out)
        rec["battery"] = bat
        prev_out = out
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["tb"] = traceback.format_exc()[-600:]
    res["instances"].append(rec)

ts = np.array([r["t_ref"] for r in res["instances"] if "t_ref" in r])
if len(ts):
    med = float(np.median(ts))
    res["summary"] = {
        "n": int(len(ts)),
        "median_s": med,
        "max_over_median": float(ts.max() / med),
        "p95_over_median": float(np.percentile(ts, 95) / med),
        "ess": float(ts.sum() ** 2 / (ts ** 2).sum()),
        "top_share": float(ts.max() / ts.sum()),
        "cv": float(ts.std() / ts.mean()),
        "rep_spread_median": float(np.median([r["t_ref_max"] / r["t_ref"] for r in res["instances"] if "t_ref" in r])),
    }
res["elapsed_s"] = time.time() - t_start
json.dump(res, open(out_path, "w"), indent=0, default=str)
print(task_name, json.dumps(res.get("summary")), "elapsed", round(res["elapsed_s"]))
