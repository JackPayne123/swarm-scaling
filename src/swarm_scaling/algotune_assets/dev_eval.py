"""Dev evaluator for this AlgoTune task. Runs inside the task container, standard library plus numpy only.

    python /app/dev/dev_eval.py /app/solver.py [--n 20] [--seed 0] [--size N] [--reps 10] [--alone-cache FILE]

Mirrors the final evaluation: the same instance generator, the same is_solution check, the same
interleaved timing (one untimed warmup of the reference and the solver, then `reps` alternating timed
calls of each, minimum per instance, speedup = total reference time / total solver time), and the same
reference check (thread_guard.py: invalid when the reference, timed with the solver, is more than 15% slower
than when timed alone on the same instances in a process without the solver).
Dev instances come from seeds the final evaluation does not use.
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

from thread_guard import ThreadGuard, make_alone_timer, reference_inflation, reference_inflation_error  # next to this file

DEV_DIR = Path(__file__).resolve().parent
SEED_OFFSET = 10_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed_ns(fn, problem) -> int:
    start = time.perf_counter_ns()
    fn(problem)
    return time.perf_counter_ns() - start


def evaluate(solver_path: Path, n: int, seed: int, size: int, reps: int, alone_cache: Path | None = None) -> dict:
    result = {"solver": str(solver_path), "n": n, "seed": seed, "size": size, "reps": reps,
              "valid": False, "speedup": None, "n_invalid": None, "total_solver_s": None,
              "total_reference_s": None, "thread_check": None, "first_call_ratio": None,
              "reference_inflation": None, "alone_baseline": None, "errors": []}
    task = load_module("reference_task", DEV_DIR / "reference_task.py").Task()
    # Started before the solver is imported: times the reference alone on each instance, in its own process (or reads
    # the scorer's cached alone timings and deletes the file).
    alone_timer = make_alone_timer(str(DEV_DIR / "reference_task.py"), reps, alone_cache and str(alone_cache))
    result["alone_baseline"] = alone_timer.mode
    # One untimed reference call before the solver is imported, so the threads the reference starts are its own.
    task.solve(task.generate_problem(n=size, random_seed=SEED_OFFSET + seed))
    guard = ThreadGuard()
    # Load the solver exactly as the final evaluation does: plain `pytest /tests/test_outputs.py`
    # puts neither this directory, the working directory nor the solver's own directory on sys.path.
    # Otherwise a solver that imports itself by name (e.g. numba cache=True) passes here and fails
    # in the real evaluation, as an Opus 5.5 pilot run did on 2026-10-08.
    hidden = {DEV_DIR, Path.cwd().resolve(), solver_path.resolve().parent}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in hidden]
    try:
        solver = load_module("solver", solver_path).Solver()
    except Exception:
        alone_timer.close()
        result["errors"].append("could not load Solver from the file:\n" + traceback.format_exc())
        return result

    total_solver = total_reference = total_first = 0
    with_solver, alone = [], []  # reference mins, timed with the solver and alone, for the timed instances
    for i in range(n):
        problem = task.generate_problem(n=size, random_seed=SEED_OFFSET + seed + i)
        try:
            guard.before_reference()
            task.solve(problem)
            guard.after_reference(None)
            first_ns = time.perf_counter_ns()
            solution = solver.solve(problem)
            first_ns = time.perf_counter_ns() - first_ns
            if not task.is_solution(problem, solution):
                result["errors"].append(f"instance {i}: is_solution returned False")
                continue
            solver_times, reference_times = [], []
            for _ in range(reps):
                guard.before_reference()
                reference_times.append(timed_ns(task.solve, problem))
                guard.after_reference(reference_times[-1])
                solver_times.append(timed_ns(solver.solve, problem))
        except Exception:
            result["errors"].append(f"instance {i}: exception\n" + traceback.format_exc())
            continue
        total_solver += min(solver_times)
        total_reference += min(reference_times)
        total_first += first_ns
        with_solver.append(min(reference_times))
        alone.append(alone_timer.time(n=size, random_seed=SEED_OFFSET + seed + i))
    alone_timer.close()

    result["n_invalid"] = len(result["errors"])
    result["thread_check"] = guard.summary()
    result["reference_inflation"] = reference_inflation(with_solver, alone)
    inflation_error = reference_inflation_error(with_solver, alone)
    if inflation_error is not None:
        result["errors"].insert(0, inflation_error)
    result["valid"] = not result["errors"]
    if total_solver > 0:
        result["speedup"] = total_reference / total_solver
        # untimed first call / min timed call: large for a solver that caches results by problem identity
        result["first_call_ratio"] = total_first / total_solver
        result["total_solver_s"] = total_solver / 1e9
        result["total_reference_s"] = total_reference / 1e9
    return result


def report(result: dict) -> None:
    for error in result["errors"][:3]:
        print(error.rstrip(), file=sys.stderr)
    if len(result["errors"]) > 3:
        print(f"... and {len(result['errors']) - 3} more invalid instances", file=sys.stderr)
    print(f"instances: {result['n']} (size {result['size']}, seeds {SEED_OFFSET + result['seed']}+), "
          f"invalid: {result['n_invalid']}")
    if result["speedup"] is None:
        print("speedup: n/a (no valid instance was timed)")
        return
    print(f"reference total {result['total_reference_s']:.4f}s, solver total {result['total_solver_s']:.4f}s")
    print(f"speedup: {result['speedup']:.3f}x" + ("" if result["valid"] else "  (valid instances only)"))
    if not result["valid"]:
        print("INVALID: the final evaluation scores an invalid run as 1.0")
    elif result["speedup"] < 1.0:
        print("NOTE: the final evaluation scores a solver slower than the reference as 1.0")


def main() -> None:
    config = json.loads((DEV_DIR / "config.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solver", type=Path, help="path to a solver.py defining class Solver")
    parser.add_argument("--n", type=int, default=20, help="number of dev instances (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="first dev seed offset, >= 0 (default 0)")
    parser.add_argument("--size", type=int, default=config["problem_size"],
                        help="problem size n passed to generate_problem (default: the final evaluation's)")
    parser.add_argument("--reps", type=int, default=10, help="timed repeats per instance (default 10)")
    parser.add_argument("--json-out", type=Path, help="also write the result as JSON to this path")
    parser.add_argument("--alone-cache", type=Path, help="the scorer's cached alone timings (read, then deleted)")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be >= 0")

    result = evaluate(args.solver, args.n, args.seed, args.size, args.reps, args.alone_cache)
    report(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
