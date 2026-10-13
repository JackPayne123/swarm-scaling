"""Validity pass for selection (scorer job kind "validity"): is_solution only, no timing. Standard library only.

    python /app/dev/validity_check.py /job/solver.py --n 1000 --seed 600000 [--size N] [--json-out FILE]

Generates instance seed + i for i < n one at a time (nothing is held), runs the solver on it once and checks the
output with the reference is_solution, the check the final evaluation applies. An exception counts as invalid. Loads
the solver as the final evaluation does (no toolkit directory, working directory or solver directory on sys.path).
Prints a progress line every 100 instances and writes the counts as JSON.
"""

import os

# Same thread limits as the final evaluation, set before numerical libraries load.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parent
KEEP = 20  # invalid instances listed by index; the first 3 errors are kept in full


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(solver_path: Path, n: int, seed: int, size: int) -> dict:
    result = {"solver": str(solver_path), "n": n, "seed": seed, "size": size, "checked": 0, "n_invalid": 0,
              "invalid_instances": [], "errors": [], "valid": False, "run_s": None}  # fmt: skip
    start = time.perf_counter()
    task = load_module("reference_task", DEV_DIR / "reference_task.py").Task()
    hidden = {DEV_DIR, Path.cwd().resolve(), solver_path.resolve().parent}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in hidden]
    try:
        solver = load_module("solver", solver_path).Solver()
    except Exception:
        result["errors"].append("could not load Solver from the file:\n" + traceback.format_exc())
        result["n_invalid"] = n
        return result
    for i in range(n):
        problem = task.generate_problem(n=size, random_seed=seed + i)
        try:
            ok = task.is_solution(problem, solver.solve(problem))
            why = f"instance {i} (seed {seed + i}): is_solution returned False"
        except Exception:
            ok, why = False, f"instance {i} (seed {seed + i}): exception\n" + traceback.format_exc()
        result["checked"] += 1
        if not ok:
            result["n_invalid"] += 1
            if len(result["invalid_instances"]) < KEEP:
                result["invalid_instances"].append(i)
            if len(result["errors"]) < 3:
                result["errors"].append(why)
        if (i + 1) % 100 == 0:
            print(f"checked {i + 1}/{n}, invalid {result['n_invalid']}, {time.perf_counter() - start:.1f}s", flush=True)
    result["valid"] = result["n_invalid"] == 0
    result["run_s"] = round(time.perf_counter() - start, 3)
    return result


def main() -> None:
    config = json.loads((DEV_DIR / "config.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solver", type=Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, default=config["problem_size"])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = check(args.solver, args.n, args.seed, args.size)
    print(f"validity: {result['n_invalid']} invalid of {result['checked']} checked (seeds {args.seed}+, size {args.size})")
    for error in result["errors"]:
        print(error.rstrip(), file=sys.stderr)
    if args.json_out:
        args.json_out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
