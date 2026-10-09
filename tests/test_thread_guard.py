"""The thread check (algotune_assets/thread_guard.py) and first-call ratio of dev_eval.py and our verifier copy.

Threads are faked (thread id -> CPU seconds, as /proc/self/task reports them), so this runs on macOS;
scripts/smoke_thread_guard.py runs real solvers through dev_eval in the task image.
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


def test_a_spinning_helper_thread_is_flagged_and_a_clean_solver_is_not() -> None:
    # Pilot 3: helper threads left spinning slowed the reference 20-28% and dev_eval reported 3837x.
    clean = interleaved({MAIN: 0.0}, solver_call=lambda t: None, during_reference=lambda t: None)
    assert clean.error() is None and clean.solver_cpu_s == 0

    def start_helpers(t):
        t.setdefault(99, 0.0)

    def helper_spins(t):
        t[99] += REFERENCE_S  # one thread busy for the whole reference call

    spinning = interleaved({MAIN: 0.0}, solver_call=start_helpers, during_reference=helper_spins)
    assert spinning.solver_cpu_s == pytest.approx(20 * REFERENCE_S)
    assert spinning.error().startswith("solver threads kept running during reference timing")


def test_library_thread_pools_a_legitimate_solver_uses_are_not_flagged() -> None:
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
    assert shutdown.error() is None and shutdown.solver_cpu_s == 0

    # The solver's own OpenMP workers spin ~3 ms each after its parallel region (libgomp's default wait).
    def openmp_region(t):
        for tid in range(30, 37):
            t.setdefault(tid, 0.0)

    def workers_finish_spinning(t):
        for tid in range(30, 37):
            t[tid] += 0.003

    openmp = interleaved({MAIN: 0.0, **pool}, openmp_region, workers_finish_spinning)
    assert openmp.solver_cpu_s == pytest.approx(20 * 7 * 0.003)
    assert openmp.error() is None


def test_the_scoring_verifier_runs_the_same_check() -> None:
    patched = with_thread_guard(seed_with_offset(VERIFIER.read_text()))
    compile(patched, "test_outputs.py", "exec")
    # created after an untimed reference call on the first problem, before the solver fixture imports the solver
    assert "    task.solve(problems[0])\n    _GUARD = ThreadGuard()\n    return problems\n" in patched
    assert patched.count("_GUARD.before_reference()") == 2  # warmup and every timed reference call
    assert "_GUARD.after_reference(end_b - start_b)" in patched
    assert "    if _GUARD.error() is not None:\n        logger.error(_GUARD.error())\n        validity = False\n" in patched
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
