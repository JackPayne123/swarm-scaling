"""The reference check and thread-CPU metadata (algotune_assets/thread_guard.py) and the first-call ratio of
dev_eval.py and our verifier copy.

Threads are faked (thread id -> CPU seconds, as /proc/self/task reports them), so this runs on macOS;
scripts/smoke_harness_v4.py runs real solvers through dev_eval in the task image.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from swarm_scaling.algotune_devkit import ASSETS
from swarm_scaling.tasks import seed_with_offset, with_first_call_ratio, with_thread_guard

spec = importlib.util.spec_from_file_location("thread_guard", ASSETS / "thread_guard.py")
thread_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thread_guard)

VERIFIER = Path(__file__).resolve().parents[1] / "research/harbor-examples/algotune_task/tests/test_outputs.py"
MAIN, REFERENCE_S = 1, 0.05


def interleaved(threads: dict[int, float], solver_call, during_reference, timings: int = 20) -> "thread_guard.ThreadGuard":
    """dev_eval's order: guard after an untimed reference call, then a reference warmup, then timed pairs."""
    guard = thread_guard.ThreadGuard(read=lambda: dict(threads))
    solver_call(threads)  # import and Solver(); may start threads
    for timed in [False] + [True] * timings:
        guard.before_reference()
        threads[MAIN] += REFERENCE_S
        during_reference(threads)
        guard.after_reference(int(REFERENCE_S * 1e9) if timed else None)
        solver_call(threads)
    return guard


def test_thread_cpu_metadata_counts_a_spinning_helper_and_not_a_clean_solver() -> None:
    # Logged with every run (not scored): pilot 3's helper threads that spun through the reference's timing.
    clean = interleaved({MAIN: 0.0}, solver_call=lambda t: None, during_reference=lambda t: None)
    assert clean.solver_cpu_s == 0

    def start_helpers(t):
        t.setdefault(99, 0.0)

    def helper_spins(t):
        t[99] += REFERENCE_S  # one thread busy for the whole reference call

    spinning = interleaved({MAIN: 0.0}, solver_call=start_helpers, during_reference=helper_spins)
    assert spinning.solver_cpu_s == pytest.approx(20 * REFERENCE_S)


def test_thread_cpu_metadata_does_not_count_library_pools_the_reference_uses() -> None:
    # numpy's OpenBLAS pool (threads 2-8) exists after the reference's untimed call, before the solver is imported.
    pool = {tid: 0.0 for tid in range(2, 9)}

    # The solver shuts the pool down when imported (blas_thread_shutdown_); the reference's next call restarts
    # it with new thread ids, and the reference keeps using those threads in every timed call.
    imported = []

    def shut_down_pool(t):
        if not imported:
            imported.append(True)
            for tid in [tid for tid in t if tid != MAIN]:
                del t[tid]

    def reference_restarts_and_uses_pool(t):
        if len(t) == 1:
            t.update({tid: 0.0 for tid in range(100, 107)})
        for tid in t:
            t[tid] += REFERENCE_S

    shutdown = interleaved({MAIN: 0.0, **pool}, shut_down_pool, reference_restarts_and_uses_pool)
    assert shutdown.solver_cpu_s == 0

    # The solver's own OpenMP workers spin ~3 ms each after its parallel region (libgomp's default wait).
    def openmp_region(t):
        for tid in range(30, 37):
            t.setdefault(tid, 0.0)

    def workers_finish_spinning(t):
        for tid in range(30, 37):
            t[tid] += 0.003

    openmp = interleaved({MAIN: 0.0, **pool}, openmp_region, workers_finish_spinning)
    assert openmp.solver_cpu_s == pytest.approx(20 * 7 * 0.003)


def test_the_scoring_verifier_runs_the_same_check() -> None:
    patched = with_thread_guard(seed_with_offset(VERIFIER.read_text()))
    compile(patched, "test_outputs.py", "exec")
    # the reference timed alone on every problem, before the solver fixture imports the solver
    assert "    _ALONE = time_reference(task.solve, problems, NUM_REPEATS)\n    _GUARD = ThreadGuard()\n    return problems\n" in patched
    assert patched.count("_GUARD.before_reference()") == 2  # warmup and every timed reference call
    assert "        _WITH_SOLVER.append(t_baseline)\n" in patched
    assert "        logger.error(reference_inflation_error(_WITH_SOLVER, _alone))\n        validity = False\n" in patched
    with pytest.raises(ValueError):
        with_thread_guard(VERIFIER.read_text().replace("            _ = baseline_func(problem)\n", "", 1))


def test_first_call_ratio_exposes_a_solver_that_caches_results_by_problem_identity(tmp_path, monkeypatch) -> None:
    # Only the untimed first call's output is checked; a solver that remembers each problem object it has seen is
    # slow once and near-free in every timed repeat. The ratio is reported for review, never scored.
    monkeypatch.syspath_prepend(str(ASSETS))
    monkeypatch.setattr(sys, "path", list(sys.path))  # evaluate() edits sys.path
    spec = importlib.util.spec_from_file_location("dev_eval", ASSETS / "dev_eval.py")
    dev_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev_eval)
    (tmp_path / "reference_task.py").write_text(
        "import random\n"
        "class Task:\n"
        "    def generate_problem(self, n, random_seed):\n"
        "        return [random.Random(random_seed).random() for _ in range(n)]\n"
        "    def solve(self, problem):\n"
        "        return sorted(problem)\n"
        "    def is_solution(self, problem, solution):\n"
        "        return solution == sorted(problem)\n"
    )
    monkeypatch.setattr(dev_eval, "DEV_DIR", tmp_path)
    monkeypatch.setattr(dev_eval, "ThreadGuard", lambda: thread_guard.ThreadGuard(read=lambda: {MAIN: 0.0}))  # no /proc
    honest = tmp_path / "honest.py"
    honest.write_text("class Solver:\n    def solve(self, problem):\n        return sorted(problem)\n")
    caching = tmp_path / "caching.py"
    caching.write_text(
        "import time\n"
        "class Solver:\n"
        "    def __init__(self):\n"
        "        self.seen = {}\n"
        "    def solve(self, problem):\n"
        "        if id(problem) not in self.seen:\n"
        "            time.sleep(0.01)\n"
        "            self.seen[id(problem)] = sorted(problem)\n"
        "        return self.seen[id(problem)]\n"
    )
    clean = dev_eval.evaluate(honest, n=5, seed=0, size=20_000, reps=5)
    cached = dev_eval.evaluate(caching, n=5, seed=0, size=20_000, reps=5)
    assert clean["valid"] and cached["valid"]  # neither is invalidated
    assert clean["first_call_ratio"] < 5 < cached["first_call_ratio"]
    # the scoring verifier copy reports the same ratio
    patched = with_first_call_ratio(with_thread_guard(seed_with_offset(VERIFIER.read_text())))
    compile(patched, "test_outputs.py", "exec")
    assert '    print(f"First-call ratio: {_first_call_ratio()}")\n' in patched



def load_dev_eval(tmp_path, monkeypatch, reference_solve: str):
    """dev_eval.py with a toy reference task in tmp_path and the thread reader faked (macOS has no /proc)."""
    monkeypatch.syspath_prepend(str(ASSETS))
    monkeypatch.setattr(sys, "path", list(sys.path))  # evaluate() edits sys.path
    spec = importlib.util.spec_from_file_location("dev_eval", ASSETS / "dev_eval.py")
    dev_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev_eval)
    (tmp_path / "reference_task.py").write_text(
        "import builtins, random, time\n"
        "class Task:\n"
        "    def generate_problem(self, n, random_seed):\n"
        "        return [random.Random(random_seed).random() for _ in range(n)]\n"
        f"    def solve(self, problem):\n{reference_solve}"
        "    def is_solution(self, problem, solution):\n"
        "        return solution == sorted(problem)\n"
    )
    monkeypatch.setattr(dev_eval, "DEV_DIR", tmp_path)
    monkeypatch.setattr(dev_eval, "ThreadGuard", lambda: thread_guard.ThreadGuard(read=lambda: {MAIN: 0.0}))
    return dev_eval


def test_a_solver_that_slows_the_reference_is_invalid_and_one_that_does_not_passes(tmp_path, monkeypatch) -> None:
    # The criterion is the harm: the reference, timed interleaved with the solver, against the same reference timed
    # alone before the solver was imported. Here the "slowdown" is a flag the solver sets when imported (on Linux it
    # is CPU contention from threads it leaves running), so the test does not depend on this machine's CPUs.
    monkeypatch.delattr("builtins._slow_reference", raising=False)
    dev_eval = load_dev_eval(tmp_path, monkeypatch, (
        "        time.sleep(0.004 if getattr(builtins, '_slow_reference', False) else 0.002)\n"
        "        return sorted(problem)\n"
    ))  # fmt: skip
    clean = tmp_path / "clean.py"
    clean.write_text("class Solver:\n    def solve(self, problem):\n        return sorted(problem)\n")
    slowing = tmp_path / "slowing.py"
    slowing.write_text(
        "import builtins\n"
        "builtins._slow_reference = True  # stands in for helper threads competing with the reference\n"
        "class Solver:\n    def solve(self, problem):\n        return sorted(problem)\n"
    )

    ok = dev_eval.evaluate(clean, n=3, seed=0, size=100, reps=3)
    assert ok["valid"] and ok["reference_inflation"] == pytest.approx(1.0, abs=0.15)

    monkeypatch.setattr("builtins._slow_reference", False, raising=False)  # removed again at teardown
    bad = dev_eval.evaluate(slowing, n=3, seed=0, size=100, reps=3)
    assert not bad["valid"] and bad["reference_inflation"] > 1.5
    assert bad["errors"][0].startswith("the reference ran") and "timed before the solver was imported" in bad["errors"][0]
