"""Large-sample degenerate-instance check: generate many instances, check inputs are finite,
and check the reference's own output is finite and passes is_solution.
usage: python -I gencheck.py <harbor dir> <task> <out.json> <n> <cap_s>"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[v]="1"
harbor_dir, task_name, out_path, N, CAP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
import importlib.util, json, logging, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
spec = importlib.util.spec_from_file_location("evaluator", os.path.join(harbor_dir, "tests", "evaluator.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
task = mod.Task()
SIZE = int(open(os.path.join(harbor_dir, "tests", "test_outputs.py")).read().split("PROBLEM_SIZE = ")[1].split()[0])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_util import finite
bad_in, bad_out, invalid, errors, times = [], [], [], [], []
t0 = time.time(); n = 0
for i in range(N):
    if time.time() - t0 > CAP: break
    seed = 8_000_000 + i
    try:
        p = task.generate_problem(n=SIZE, random_seed=seed)
        if not finite(p): bad_in.append(seed)
        ts = time.perf_counter(); out = task.solve(p); times.append(time.perf_counter() - ts)
        if not finite(out): bad_out.append(seed)
        if not task.is_solution(p, out): invalid.append(seed)
    except Exception as e:
        errors.append([seed, f"{type(e).__name__}: {e}"[:200]])
    n += 1
ts = np.array(times)
res = dict(task=task_name, n=n, bad_in=bad_in, bad_out=bad_out, ref_invalid=invalid, errors=errors[:20], n_errors=len(errors),
           ess=float(ts.sum()**2/(ts**2).sum()) if len(ts) else None, ess_per100=float(100*ts.sum()**2/(ts**2).sum()/len(ts)) if len(ts) else None,
           max_over_median=float(ts.max()/np.median(ts)) if len(ts) else None, p99_over_median=float(np.percentile(ts,99)/np.median(ts)) if len(ts) else None,
           elapsed=time.time()-t0)
json.dump(res, open(out_path, "w"), indent=0)
print(task_name, n, len(bad_in), len(bad_out), len(invalid), len(errors), res['ess_per100'], res['max_over_median'])
