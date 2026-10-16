"""pca under the upstream verifier port (v7 TASK_PATCHES): A5i verdict stability, the rotated-subspace probe, and the
rule 2(b) one-change ablation (sklearn PCA -> eigh of X^T X, float64, descending), with v7-tasks.md's methods.
usage: python -I pca_checks.py <patched harbor task dir> <out.json>"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import importlib.util
import json
import logging
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
d, out_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("evaluator", os.path.join(d, "tests", "evaluator.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
T = mod.Task()
SIZE = int(open(os.path.join(d, "tests", "test_outputs.py")).read().split("PROBLEM_SIZE = ")[1].split()[0])


def pca_eigh(p):  # ablate.py's variant, as the route report names it
    X = np.array(p["X"], dtype=float)
    X = X - X.mean(axis=0)
    k = p["n_components"]
    w, V = np.linalg.eigh(X.T @ X)
    return V[:, ::-1][:, :k].T


res = {"size": SIZE}
# A5i: the same output, 10 is_solution calls on each of 5 instances (the verifier re-runs a randomized sklearn PCA)
flips = {}
for label, f in (("reference", T.solve), ("eigh_f64_desc", pca_eigh)):
    counts = []
    for i in range(5):
        p = T.generate_problem(n=SIZE, random_seed=9_000_000 + i)
        out = f(p)
        counts.append(sum(bool(T.is_solution(p, out)) for _ in range(10)))
    flips[label] = counts
res["a5i_accepted_of_10_per_instance"] = flips
# rotated reference subspace (probes.py's probe)
rng = np.random.default_rng(1)
acc = 0
for i in range(20):
    p = T.generate_problem(n=SIZE, random_seed=6_000_000 + i)
    V = np.asarray(T.solve(p))
    Q, _ = np.linalg.qr(rng.standard_normal((V.shape[0], V.shape[0])))
    acc += bool(T.is_solution(p, (Q @ V).tolist()))
res["rotated_accepted_of_20"] = acc
# rule 2(b) ablation, ablate.py's method: 25 instances, seeds 3,000,000+i, warmup + min of 2
tot_ref = tot_var = 0.0
valid = 0
for i in range(25):
    p = T.generate_problem(n=SIZE, random_seed=3_000_000 + i)
    T.solve(p)
    best = float("inf")
    for _ in range(2):
        a = time.perf_counter(); T.solve(p); best = min(best, time.perf_counter() - a)
    tot_ref += best
    pca_eigh(p)
    best = float("inf")
    for _ in range(2):
        a = time.perf_counter(); out = pca_eigh(p); best = min(best, time.perf_counter() - a)
    tot_var += best
    valid += bool(T.is_solution(p, out))
res["ablation_eigh_f64_desc"] = {"speedup": tot_ref / tot_var, "valid": f"{valid}/25"}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res))
