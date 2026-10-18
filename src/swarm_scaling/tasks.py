"""Task factories for the swarm-scaling experiments."""

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import yaml
from inspect_ai import Task, task, task_with
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, mean, metric, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec, sandbox, sandbox_default
from inspect_harbor import algotune, harbor_scorer

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import (
    ASSETS,
    CHECKER,
    CHECKER_LOCK,
    SEED_OFFSET_RANGE,
    SOLVER_PATH,
    TIMING_ENV,
    VERIFIER_INSTANCES,
    algotune_setup,
    end_checker_processes,
    remote,
    remote_timing,
    scorer_job,
)

SPLIT_PATH = Path(__file__).resolve().parents[2] / "data" / "algotune_split.json"
# Task lists in SPLIT_PATH: pilot and heldout (the frozen 2026-10-05 rule), v7 (the v7 screen's candidates, PLAN.md)
SPLITS = ("pilot", "heldout", "v7")

# Harbor's verifier scores on generate_problem(random_seed=i) for i in 0..99, and agents hold generate_problem.
# The offset is added to those seeds. Dev instances use seeds 10,000+ and selection 60,000+; np.random.seed needs < 2**32.
SEED_OFFSET_ENV = "ALGOTUNE_SEED_OFFSET"
# algotune_devkit.seed_ranges lists every seed range in one place; the hidden offset's is SEED_OFFSET_RANGE.
_VERIFIER_SEED_CALL = "task.generate_problem(n=PROBLEM_SIZE, random_seed=i)"
_VERIFIER_IMPORT = "from evaluator import Task\n"
_VERIFIER_FIXTURE_ORDER = "def performance_results(solver_instance, problem_set) -> dict:"

# Pinned hub dataset (was `latest` on 2026-10-05), so a hub update cannot change the verifier or image mid-experiment.
ALGOTUNE_REF = "sha256:69d264f15f717a20841c30fe3e91306925a7df1fd08830b3129d92e1b2c4958e"

PROJECT_ROOT = SPLIT_PATH.parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
# Patched verifier copies (they hold the secret offset file); reused across runs instead of temp dirs. Gitignored.
VERIFIER_DIR = SPLIT_PATH.parent / ".algotune_verifier"
# Generated compose files, one per sample (see algotune_task). Gitignored.
COMPOSE_DIR = SPLIT_PATH.parent / ".algotune_compose"
CHECKER_CPUS = 8
CHECKER_MEMORY_MB = 8192  # the verifier's measured peak is 4.2 GiB (dst-type-ii-scipy-fftpack, 2026-10-09)
# Left for the Docker host itself (970 MiB measured on the Mac's Docker VM with no containers, 2026-10-09).
HOST_HEADROOM_MB = 2048
# Per-agent boxes (isolate_agents, v7): every agent's box mounts these named volumes, so the registry, inboxes and
# teammates' working directories are where swarm() puts them (workspace_root /app) in every box; the checker gets none.
SHARED_VOLUMES = {"swarm_agents": "/app/agents", "swarm_registry": "/app/registry"}
# Written into every per-agent box's site-packages: Python 3.12's os.cpu_count() reports the host's CPUs (16 on a team
# VM, 4 on an independent's), while nproc, OpenBLAS and numba already follow the box's cpuset. v6 sinkhorn agents read
# os.cpu_count()=16 and chose 16 OpenMP threads on 8 CPUs.
CPU_COUNT_SITECUSTOMIZE = """\
# swarm-scaling isolate_agents: os.cpu_count() is the CPUs this box may run on, as nproc reports.
import os

os.cpu_count = lambda: len(os.sched_getaffinity(0))
"""

# Inserted into the verifier copy. The offset travels as a file in /tests (copied in only at scoring), not as an
# environment variable (visible in /proc/self/environ to the solver), and the file is read and deleted while the
# problems are generated, before the solver module is imported (see the fixture reordering in seed_with_offset).
_SEED_HELPER = '''
_OFFSET_PATH = Path(__file__).with_name("seed_offset")
_OFFSET = None


def _take_seed_offset() -> int:
    """Read the secret seed offset once and delete its file, before the solver is imported."""
    global _OFFSET
    if _OFFSET is None:
        _OFFSET = int(_OFFSET_PATH.read_text())
        _OFFSET_PATH.unlink()
    return _OFFSET
'''


def _speedup(reward: float) -> float:
    """AlgoTune scores an invalid or slower solver as 1.0. A reward of 0 (no /app/solver.py) is scored 1.0 too."""
    return max(reward, 1.0)


@metric
def harmonic_mean_speedup() -> Metric:
    """AlgoTune's official aggregate."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:  # Inspect calls metrics with no scores at the start of a run
            return float("nan")
        speedups = [_speedup(s.score.as_float()) for s in scores]
        return len(speedups) / sum(1 / s for s in speedups)

    return compute


@metric
def mean_log_speedup() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return float("nan")
        return sum(math.log(_speedup(s.score.as_float())) for s in scores) / len(scores)

    return compute


async def _copy_solver_to_checker() -> None:
    """Put the agent box's /app/solver.py (the selected candidate, or none) at the same path in the checker."""
    checker = sandbox(CHECKER)
    removed = await checker.exec(["rm", "-f", SOLVER_PATH])
    if not removed.success:
        raise RuntimeError(f"could not clear {SOLVER_PATH} in the checker: {removed.stderr}")
    try:
        source = await sandbox().read_file(SOLVER_PATH, text=False)
    except FileNotFoundError:
        return  # nothing was selected: the verifier finds no solver and scores 0, as in the agent box
    await checker.write_file(SOLVER_PATH, source)


@scorer(metrics=[mean(), harmonic_mean_speedup(), mean_log_speedup()])
def algotune_scorer() -> Scorer:
    """Harbor's scorer run in the checker container (Score.value stays the raw reward), plus the log speedup.

    The selected /app/solver.py is copied from the agent box into the checker, then Harbor's scorer runs with
    the checker as the default sandbox, so its /tests copy (with the seed_offset file), verifier run and reward
    read all happen there, on the checker's CPUs. CHECKER_LOCK is held throughout, so no other timed run in
    the process overlaps the verifier. Every checker process started since setup is killed first, so nothing a
    solver left running competes with the verifier or sees /tests. Queue wait, run times and the cleanup
    result go in metadata["checker"].
    The selected solver was already evaluated in the checker by finalize, also as a single file.
    A verifier that times out scores 1.0 (no speedup, as an invalid run) with metadata["scoring_timeout"] True,
    instead of failing the eval. metadata["first_call_ratio"] is the verifier's first-call ratio
    (`with_first_call_ratio`; None when nothing was timed).
    """
    verify = harbor_scorer()

    async def score(state: TaskState, target: Target) -> Score:
        if remote(state):
            return await _remote_score(state)
        requested = time.time()
        async with CHECKER_LOCK:
            started = time.time()
            cleanup = await end_checker_processes(state)
            await _copy_solver_to_checker()
            with sandbox_default(CHECKER):
                try:
                    result = await verify(state, target)
                    timed_out = False
                except TimeoutError:  # Inspect raises it when the verifier exceeds verifier_timeout_sec
                    timeout_s = state.metadata.get("verifier_timeout_sec", 600)  # harbor_scorer's default
                    result = Score(value=1.0, explanation=f"verifier timeout after {timeout_s:g} s")
                    timed_out = True
            ended = time.time()
        ratio = re.search(r"^First-call ratio: (\S+)$", result.explanation or "", re.M)
        inflation = re.search(r"^Reference inflation: (\S+)$", result.explanation or "", re.M)
        alone = re.search(r"^Alone score: (\S+)$", result.explanation or "", re.M)
        result.metadata = {
            **(result.metadata or {}),
            "algotune_score": result.value,
            "alone_score": float(alone.group(1)) if alone else None,
            "scoring_timeout": timed_out,
            "first_call_ratio": float(ratio.group(1)) if ratio and ratio.group(1) != "None" else None,
            "reference_inflation": float(inflation.group(1)) if inflation and inflation.group(1) != "None" else None,
            "log_speedup": math.log(_speedup(result.as_float())),
            "checker": {
                "queue_wait_s": round(started - requested, 3), "started_at": started, "ended_at": ended, "cleanup": cleanup,
                "alone_baseline": {"mode": "child"}, "timing_env": TIMING_ENV,
            },
        }
        return _primary(state, result)

    return score


def _primary(state: TaskState, score: Score) -> Score:
    """Score.value is the sample's primary score (metadata "primary_score"): AlgoTune's (interleaved, the raw reward,
    always kept as metadata "algotune_score") or the alone score (v7). A run without a timing (no solver, a scoring
    timeout) keeps its value; log_speedup follows the primary."""
    alone = (score.metadata or {}).get("alone_score")
    if state.metadata.get("primary_score") == "alone" and alone is not None:
        score.value = alone
        score.metadata["log_speedup"] = math.log(_speedup(alone))
    return score


async def _remote_score(state: TaskState) -> Score:
    """algotune_scorer for the remote checker: one `score` job on the scorer service, scored as the local path does.

    The service runs the same seeded verifier in a fresh container on a scorer slot and applies the same timeout
    semantics (1.0 with scoring_timeout). No /app/solver.py scores 0 without a job, as the verifier would.
    """
    try:
        source = await sandbox().read_file(SOLVER_PATH, text=False)
    except FileNotFoundError:
        return Score(value=0.0, explanation=f"no {SOLVER_PATH}", metadata={
            "log_speedup": 0.0, "scoring_timeout": False, "first_call_ratio": None, "reference_inflation": None,
            "checker": None,
        })  # fmt: skip
    try:
        rec = await scorer_client.run_job(scorer_job(state, "score", source))
    except TimeoutError as ex:  # the service did not end the job in time: score as a verifier timeout
        return Score(value=1.0, explanation=str(ex), metadata={
            "log_speedup": 0.0, "scoring_timeout": True, "first_call_ratio": None, "reference_inflation": None,
            "checker": None,
        })  # fmt: skip
    if rec["status"] != "done":
        raise RuntimeError(f"scorer job {rec['job_id']} failed: {rec['error']}")
    r = rec["result"]
    return _primary(state, Score(
        value=r["score"],
        explanation=r["verifier_stdout"],
        metadata={
            "log_speedup": math.log(_speedup(r["score"])),
            "algotune_score": r["score"],
            "alone_score": r.get("alone_score"),
            "scoring_timeout": r["scoring_timeout"],
            "first_call_ratio": r["first_call_ratio"],
            "reference_inflation": r.get("reference_inflation"),
            "checker": remote_timing(rec),
        },
    ))  # fmt: skip


def seed_with_offset(test_outputs: str) -> str:
    """Harbor's AlgoTune test_outputs.py with its instance seeds read as offset + i.

    The offset is read from a `seed_offset` file next to the verifier and deleted on first use, and the
    problems are generated before the solver is imported, so the solver's import-time code cannot read the
    offset to precompute answers. Raises unless every patched anchor occurs exactly once, so a changed hub
    copy cannot silently keep seeds 0..99 or the old import order.
    """
    for anchor in (_VERIFIER_SEED_CALL, _VERIFIER_IMPORT, _VERIFIER_FIXTURE_ORDER):
        if test_outputs.count(anchor) != 1:
            raise ValueError(f"expected exactly one {anchor!r} in the AlgoTune verifier")
    seeded = _VERIFIER_SEED_CALL.replace("random_seed=i", "random_seed=_take_seed_offset() + i")
    return (
        test_outputs.replace(_VERIFIER_SEED_CALL, seeded)
        .replace(_VERIFIER_IMPORT, _VERIFIER_IMPORT + _SEED_HELPER)
        # pytest sets up fixtures in argument order: problems (offset read and deleted) before solver import
        .replace(_VERIFIER_FIXTURE_ORDER, "def performance_results(problem_set, solver_instance) -> dict:")
    )


# with_thread_guard: anchor -> replacement in the verifier, each anchor required exactly once.
_GUARD_PATCHES = {
    _VERIFIER_IMPORT: _VERIFIER_IMPORT
    + "from thread_guard import ThreadGuard, make_alone_timer, reference_inflation, reference_inflation_error\n\n"
    + "_GUARD = _ALONE_TIMER = None  # set in problem_set, before the solver is imported\n"
    + "_ALONE = []  # per timed problem: the reference's min timed call alone, in the AloneTimer process (ns)\n"
    + "_WITH_SOLVER = []  # per timed problem: the reference's min timed call interleaved with the solver (ns)\n",
    '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n    return problems\n': (
        '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n'
        "    global _GUARD, _ALONE_TIMER\n"
        "    _ALONE_TIMER = make_alone_timer(str(Path(__file__).with_name('evaluator.py')), NUM_REPEATS,\n"
        "                                    str(Path(__file__).with_name('alone_cache.json')))\n"
        "    task.solve(problems[0])\n"
        "    _GUARD = ThreadGuard()\n"
        "    return problems\n"
    ),
    "        _ = baseline_func(problem)\n\n        # Warmup Solver": (
        "        _GUARD.before_reference()\n"
        "        _ = baseline_func(problem)\n"
        "        _GUARD.after_reference(None)\n\n"
        "        # Warmup Solver"
    ),
    (
        "            start_b = time.perf_counter_ns()\n"
        "            _ = baseline_func(problem)\n"
        "            end_b = time.perf_counter_ns()\n"
    ): (
        "            _GUARD.before_reference()\n"
        "            start_b = time.perf_counter_ns()\n"
        "            _ = baseline_func(problem)\n"
        "            end_b = time.perf_counter_ns()\n"
        "            _GUARD.after_reference(end_b - start_b)\n"
    ),
    "        total_time_baseline += t_baseline\n": (
        "        total_time_baseline += t_baseline\n"
        "        _WITH_SOLVER.append(t_baseline)\n"
        "        _ALONE.append(_ALONE_TIMER.time(n=PROBLEM_SIZE, random_seed=_take_seed_offset() + i))\n"
    ),
    "    if validity and total_time_solver > 0:\n": (
        "    _ALONE_TIMER.close()\n"
        "    print(f\"Thread check: {_GUARD.summary()}\")\n"
        "    print(f\"Alone baseline: {_ALONE_TIMER.mode}\")\n"
        "    print(f\"Reference inflation: {reference_inflation(_WITH_SOLVER, _ALONE)}\")\n"
        "    if reference_inflation_error(_WITH_SOLVER, _ALONE) is not None:\n"
        "        logger.error(reference_inflation_error(_WITH_SOLVER, _ALONE))\n"
        "        validity = False\n"
        "    if validity and total_time_solver > 0:\n"
    ),
}


# with_first_call_ratio: anchor -> replacement, each anchor required exactly once.
_FIRST_CALL_PATCHES = {
    "        warmup_sol = solver_func(problem)\n": (
        "        _first_start = time.perf_counter_ns()\n"
        "        warmup_sol = solver_func(problem)\n"
        "        _first_ns = time.perf_counter_ns() - _first_start\n"
    ),
    "    return min(solver_timings), min(baseline_timings)\n": (
        "    _FIRST_CALLS.append((_first_ns, min(solver_timings)))\n"
        "    return min(solver_timings), min(baseline_timings)\n"
    ),
    "def _write_reward(speedup: float) -> None:\n": (
        "_FIRST_CALLS = []  # (untimed first solver call, min timed solver call) per instance, in ns\n\n\n"
        "def _first_call_ratio():\n"
        "    timed = sum(m for _, m in _FIRST_CALLS)\n"
        "    return sum(f for f, _ in _FIRST_CALLS) / timed if timed else None\n\n\n"
        "def _write_reward(speedup: float) -> None:\n"
    ),
    '    print(f"Final Reward (Score): {final_score:.4f}")\n': (
        '    print(f"Final Reward (Score): {final_score:.4f}")\n'
        '    print(f"First-call ratio: {_first_call_ratio()}")\n'
    ),
}


# with_alone_score: anchor -> replacement, required exactly once; needs with_thread_guard's _ALONE first.
_ALONE_SCORE_PATCH = {
    "    _write_reward(final_score)\n": (
        "    _alone_raw = sum(_ALONE) / total_time_solver if validity and total_time_solver > 0 else 0.0\n"
        "    _alone_score = _alone_raw if validity and _alone_raw >= 1.0 else 1.0\n"
        '    print(f"Alone raw speedup: {_alone_raw:.4f} x")\n'
        '    print(f"Alone score: {_alone_score:.4f}")\n'
        "    _write_reward(final_score)\n"
    ),
}


# v7 task fixes (experiments after v6; v6 keeps the Harbor code): per task, (anchor, replacement, shown in the
# instruction) applied to evaluator.py (the reference Task: generate_problem, solve, is_solution) in the agents' toolkit,
# the scorer's toolkit and the verifier copy, and to the instruction when it shows that code. Ports are verbatim from
# upstream AlgoTune (github.com/oripress/AlgoTune HEAD dff9914, 2026-06-24); the Harbor snapshot predates them.
TASK_PATCHES: dict[str, list[tuple[str, str, bool]]] = {
    # Upstream port. Harbor's is_solution compares only the captured variance, so any rotation of the top-k subspace
    # passes (20/20 rotated outputs locally; upstream 0/10). Upstream matches sklearn's components up to sign.
    "pca": [
        (
            '            V = np.array(solution)\n            X = np.array(problem["X"])\n',
            '            V = np.array(solution, dtype=float)\n            X = np.array(problem["X"], dtype=float)\n',
            True,
        ),
        (
            "            # check objective\n"
            "            res = self.solve(problem)\n"
            "            V_solver = np.array(res)\n"
            "\n"
            "            obj_solver = np.linalg.norm(X @ V_solver.T) ** 2\n"
            "            obj_sol = np.linalg.norm(X @ V.T) ** 2\n"
            "            if np.allclose(obj_sol, obj_solver, rtol=tol, atol=tol / 10):\n"
            "                return True\n"
            "            return False\n",
            "            # Match sklearn's ordered principal components up to per-row sign flips.\n"
            "            # A rotated basis for the same top-k subspace is not equivalent to the\n"
            "            # reference PCA output, even if it preserves the aggregate objective.\n"
            "            V_solver = np.array(self.solve(problem), dtype=float)\n"
            "            if V_solver.shape != V.shape:\n"
            "                return False\n"
            "\n"
            "            component_alignment = np.abs(V @ V_solver.T)\n"
            "            return np.allclose(\n"
            "                component_alignment,\n"
            "                np.eye(n_components),\n"
            "                rtol=tol,\n"
            "                atol=tol / 10,\n"
            "            )\n",
            True,
        ),
    ],
    # Upstream port. Harbor's tolerance exceeds 1 at this size (1.24), so U = I, S = 0, V = I passes (20/20 locally).
    "randomized_svd": [
        (
            "        tol *= 1 + (1 - k / min(n, m)) * 2\n",
            "        tol *= 1 + (1 - k / min(n, m)) * 2\n"
            "        tol = min(tol, 0.9)  # Cap tolerance to reject trivial solutions\n",
            True,
        ),
    ],
    # Ours (upstream has the same code). scipy.sparse.random without random_state draws the sparsity pattern from
    # numpy's global RNG, so a seed did not fix the instance and every scoring used a different set.
    "sparse_lowest_eigenvalues_posdef": [
        (
            "            data_rvs=rng.standard_normal,\n",
            "            data_rvs=rng.standard_normal,\n            random_state=rng,\n",
            False,
        ),
    ],
}


# Appended to every task's evaluator.py after v6 (not to the instruction): upstream's non-concrete solution guard
# around Task.is_solution (algotune_assets/concrete_guard.py, 2026-10-18).
CONCRETE_GUARD = (ASSETS / "concrete_guard.py").read_text() + "\n\nguard_is_solution(Task)\n"


def patch_task(name: str, text: str, experiment: str, instruction: bool = False) -> str:
    """The task's code (evaluator.py, or with instruction=True the instruction) with its TASK_PATCHES for `experiment`,
    and for the code CONCRETE_GUARD at the end.

    v6 is never patched. Each anchor must occur exactly once, unless its replacement already does (the text was
    patched before: a seeded verifier copy holds the patched evaluator.py), so a changed Harbor copy raises instead
    of silently keeping the old code.
    """
    if experiment == "v6":
        return text
    for anchor, replacement, shown in TASK_PATCHES.get(name, []):
        if instruction and not shown:
            continue
        if text.count(replacement) == 1:
            continue
        if text.count(anchor) != 1:
            raise ValueError(f"{name}: expected exactly one {anchor!r} in the task's {'instruction' if instruction else 'code'}")
        text = text.replace(anchor, replacement)
    if not instruction and CONCRETE_GUARD not in text:
        text = text.rstrip("\n") + "\n\n\n" + CONCRETE_GUARD
    return text


def with_alone_score(test_outputs: str) -> str:
    """The verifier also prints the speedup against the reference timed alone ("Alone score: <x>", v7's primary score).

    AlgoTune's score divides the reference's time interleaved with the solver by the solver's, so a solver that leaves
    process-wide thread state behind moves its own denominator (v6 pca r2: one source scored 12.72, 7.25 and 7.06 on the
    same hidden instances). The alone score uses the same instances, the same per-instance minimum of the same solver
    timings and the same rules (invalid or below 1.0 scores 1.0), with the reference's time from the thread guard's
    alone timer instead. reward.txt keeps AlgoTune's score. Needs with_thread_guard; raises unless the anchor occurs once.
    """
    for anchor, replacement in _ALONE_SCORE_PATCH.items():
        if test_outputs.count(anchor) != 1 or "_ALONE = []" not in test_outputs:
            raise ValueError(f"expected exactly one {anchor!r} and the thread guard's _ALONE in the AlgoTune verifier")
        test_outputs = test_outputs.replace(anchor, replacement)
    return test_outputs


def with_first_call_ratio(test_outputs: str) -> str:
    """The verifier also reports the first-call ratio: sum of untimed first solver calls / sum of min timed calls.

    A solver that caches results by problem identity is slow on the first (checked) call and fast on the timed
    repeats, so its ratio is large. The ratio is only reported (verifier output, score metadata), never scored.
    Raises unless every anchor occurs exactly once.
    """
    for anchor, replacement in _FIRST_CALL_PATCHES.items():
        if test_outputs.count(anchor) != 1:
            raise ValueError(f"expected exactly one {anchor!r} in the AlgoTune verifier")
        test_outputs = test_outputs.replace(anchor, replacement)
    return test_outputs


def with_thread_guard(test_outputs: str) -> str:
    """The verifier with thread_guard.py's reference check: invalid when the reference, timed with the solver, is
    more than 15% slower than timed alone on the same instances before the solver was imported.

    Pilot 3: a solver that left spin-waiting threads running scored 3837x on dev_eval by slowing the reference.
    Starts thread_guard's AloneTimer before the solver fixture imports the solver; after each instance's
    interleaved timing it times the reference alone on that instance (one untimed and NUM_REPEATS timed calls, in
    the timer's process, the two processes never running at once). Prints "Reference inflation: <ratio>" and the
    thread-CPU summary, and sets validity False when the check fails, which the verifier scores 1.0 like any
    invalid run. Timing and the speedup formula are unchanged. Needs thread_guard.py and evaluator.py next to the
    verifier. Raises unless every anchor occurs exactly once.
    """
    for anchor, replacement in _GUARD_PATCHES.items():
        if test_outputs.count(anchor) != 1:
            raise ValueError(f"expected exactly one {anchor!r} in the AlgoTune verifier")
        test_outputs = test_outputs.replace(anchor, replacement)
    return test_outputs


SEED_OFFSET_FILE = SPLIT_PATH.parent / ".algotune_seed_offset"  # gitignored; one secret for the whole experiment
# One secret hidden-seed offset per experiment (runner/scorer --experiment). v7's hidden instances are new because the
# v6 ones informed the v6 diagnosis (Sol review, 2026-10-13); v6 keeps its own so v6 runs still reproduce.
EXPERIMENTS = ("v6", "v7")


def seed_offset_file(experiment: str = "v6") -> Path:
    """The experiment's offset file: v6 keeps data/.algotune_seed_offset, later ones data/.algotune_seed_offset.<name>."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")
    return SEED_OFFSET_FILE if experiment == "v6" else SEED_OFFSET_FILE.with_name(f"{SEED_OFFSET_FILE.name}.{experiment}")


def _offsets_in_logs(log_dir: Path, experiment: str = "v6") -> set[int]:
    """Seed offsets recorded in existing AlgoTune eval logs (eval.metadata["algotune_seed_offset"]) of this experiment
    (eval.metadata["experiment"]; logs without one are v6)."""
    from inspect_ai.log import read_eval_log

    found: set[int] = set()
    for log in sorted(log_dir.glob("*.eval")) if log_dir.is_dir() else []:
        try:
            header = read_eval_log(str(log), header_only=True)
        except Exception:
            continue
        meta = header.eval.metadata or {}
        value = meta.get("algotune_seed_offset")
        if value is not None and meta.get("experiment", "v6") == experiment:
            found.add(int(value))
    return found


def _seed_offset(path: Path = SEED_OFFSET_FILE, log_dir: Path = LOG_DIR, experiment: str = "v6") -> int:
    """One experiment-wide secret offset, so every arm and repeat is scored on the same instances.

    ALGOTUNE_SEED_OFFSET overrides. Otherwise the offset is read from `path`. If `path` is missing it is
    restored from the offset recorded in existing eval logs of `experiment`; only when there are none is a new random
    offset written, at least 100 seeds from every other experiment's offset file next to `path`, so no hidden instance
    is shared. Creation is atomic, so two runs started at once cannot write different offsets.
    A fresh offset per run would score separately launched arms on different instances and break
    paired per-task comparisons.
    """
    low, high = SEED_OFFSET_RANGE
    if os.environ.get(SEED_OFFSET_ENV):
        offset = int(os.environ[SEED_OFFSET_ENV])
    else:
        if not path.exists():
            logged = _offsets_in_logs(log_dir, experiment)
            if len(logged) > 1:
                raise ValueError(f"{path} is missing and the logs in {log_dir} record several offsets: {sorted(logged)}")
            others = [int(f.read_text().strip()) for f in path.parent.glob(f"{SEED_OFFSET_FILE.name}*") if f != path]
            new = logged.pop() if logged else None
            while new is None or any(abs(new - o) < VERIFIER_INSTANCES for o in others):
                new = low + secrets.randbelow(high - low)
            try:
                with path.open("x") as f:  # atomic: fails if another run created it first
                    f.write(f"{new}\n")
            except FileExistsError:
                pass
        offset = int(path.read_text().strip())
    if not low <= offset <= high:
        raise ValueError(f"seed offset {offset} (from {SEED_OFFSET_ENV} or {path}) is outside {low}..{high}")
    return offset


def docker_host() -> tuple[int, int]:
    """The Docker host's CPU count and memory in MiB (`docker info`), read when the task is built."""
    out = subprocess.run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"docker info failed: {out.stderr.strip()}")
    cpus, memory = out.stdout.split()
    return int(cpus), int(memory) // 2**20


def task_image_digest(image: str) -> str | None:
    """The image's registry digest (sha256:...) from `docker image inspect`, or None for a locally built image.

    Cloud VMs pull the task image by digest, so agents' and scorer's logs can show they ran the identical image.
    """
    out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"docker image inspect {image} failed: {out.stderr.strip()}")
    digests = json.loads(out.stdout) or []
    return digests[0].split("@", 1)[1] if digests else None


def box_resources(
    cpus_per_agent: int, n_agents: int, parallel: int, host_cpus: int, host_memory_mb: int, checker: bool = True,
    isolate: bool = False, memory_per_agent_mb: int | None = None,
) -> dict:
    """CPUs, cpusets and memory limits of the agent box and the checker on a Docker host of this size.

    Every agent gets the same CPUs: the agent box has cpus_per_agent x n_agents CPUs (0..A-1), the checker
    CHECKER_CPUS (A..A+7), so agent work cannot disturb the checker's timings. Only one checker times at a
    time (CHECKER_LOCK), so the `parallel` agent boxes share the host memory left after one checker and
    HOST_HEADROOM_MB. With parallel > 1 the agent boxes share cpuset 0..A-1 and the checkers A..A+7.
    checker=False (remote scorer): no checker on this host, so nothing is reserved for it.
    isolate (v7): one box per agent instead (`agent_boxes`): agent i's box has cpus_per_agent CPUs, its own cpuset
    (i x c .. i x c + c - 1) and memory_per_agent_mb, the same as an independent run's box given the same arguments.
    """
    if min(cpus_per_agent, n_agents, parallel) < 1:
        raise ValueError("cpus_per_agent, n_agents and parallel must be >= 1")
    if isolate != (memory_per_agent_mb is not None):
        raise ValueError("per-agent boxes need memory_per_agent_mb (the same in every row), and only they take it")
    if isolate:
        return _isolated_resources(cpus_per_agent, n_agents, parallel, host_cpus, host_memory_mb, checker, memory_per_agent_mb)
    agent_cpus = cpus_per_agent * n_agents
    if not checker:
        if agent_cpus > host_cpus:
            raise ValueError(f"{n_agents} agents x {cpus_per_agent} CPUs = {agent_cpus} CPUs, but the Docker host has {host_cpus}")
        agent_memory_mb = (host_memory_mb - HOST_HEADROOM_MB) // parallel
        return {
            "cpus_per_agent": cpus_per_agent,
            "agent_cpus": agent_cpus,
            "agent_cpuset": f"0-{agent_cpus - 1}",
            "agent_memory_mb": agent_memory_mb,
            "docker_host_cpus": host_cpus,
            "docker_host_memory_mb": host_memory_mb,
        }
    if agent_cpus + CHECKER_CPUS > host_cpus:
        raise ValueError(
            f"{n_agents} agents x {cpus_per_agent} CPUs + {CHECKER_CPUS} checker CPUs = {agent_cpus + CHECKER_CPUS} "
            f"CPUs, but the Docker host has {host_cpus}"
        )
    agent_memory_mb = (host_memory_mb - CHECKER_MEMORY_MB - HOST_HEADROOM_MB) // parallel
    if agent_memory_mb < 1024:
        raise ValueError(f"only {agent_memory_mb} MiB per agent box on a host with {host_memory_mb} MiB")
    return {
        "cpus_per_agent": cpus_per_agent,
        "agent_cpus": agent_cpus,
        "agent_cpuset": f"0-{agent_cpus - 1}",
        "checker_cpus": CHECKER_CPUS,
        "checker_cpuset": f"{agent_cpus}-{agent_cpus + CHECKER_CPUS - 1}",
        "agent_memory_mb": agent_memory_mb,
        "checker_memory_mb": CHECKER_MEMORY_MB,
        "docker_host_cpus": host_cpus,
        "docker_host_memory_mb": host_memory_mb,
    }


def _isolated_resources(
    c: int, n: int, parallel: int, host_cpus: int, host_memory_mb: int, checker: bool, memory_mb: int
) -> dict:
    agent_cpus = c * n
    boxes = [{"service": "default" if i == 0 else f"agent_{i}", "cpus": c, "cpuset": f"{i * c}-{i * c + c - 1}",
              "memory_mb": memory_mb} for i in range(n)]  # fmt: skip
    need_cpus = agent_cpus + (CHECKER_CPUS if checker else 0)
    need_mb = memory_mb * n * parallel + (CHECKER_MEMORY_MB if checker else 0) + HOST_HEADROOM_MB
    if need_cpus > host_cpus or need_mb > host_memory_mb:
        raise ValueError(
            f"{n} agent boxes x {c} CPUs and {memory_mb} MiB (x {parallel} samples)"
            + (f" + the checker's {CHECKER_CPUS} CPUs and {CHECKER_MEMORY_MB} MiB" if checker else "")
            + f" + {HOST_HEADROOM_MB} MiB headroom need {need_cpus} CPUs and {need_mb} MiB; the Docker host has "
            f"{host_cpus} and {host_memory_mb}"
        )
    out = {
        "isolate_agents": True,
        "cpus_per_agent": c,
        "agent_cpus": agent_cpus,
        "agent_cpuset": f"0-{agent_cpus - 1}",
        "agent_memory_mb": memory_mb,
        "agent_boxes": boxes,
        "docker_host_cpus": host_cpus,
        "docker_host_memory_mb": host_memory_mb,
    }
    if checker:
        out |= {"checker_cpus": CHECKER_CPUS, "checker_cpuset": f"{agent_cpus}-{agent_cpus + CHECKER_CPUS - 1}",
                "checker_memory_mb": CHECKER_MEMORY_MB}  # fmt: skip
    return out


def add_checker(config: ComposeConfig, resources: dict) -> None:
    """Add the `checker` service, a copy of `default` that only the dev_eval tool, final selection and scoring use.

    Pins both boxes to their cpusets from `box_resources` and sets their memory limits; the agent box also gets a CPU
    quota, the checker none: a quota equal to its cpuset throttled fully busy timed runs about 1% of the time
    (scoring investigation, 2026-10-10). Every process in the checker (dev_eval, the verifier, the alone-timer child)
    gets TIMING_ENV from the service environment. ComposeService has no cpuset field; extras set after construction
    reach the generated YAML.
    """
    default = _limit_agent_box(config, resources)
    checker = default.model_copy(deep=True)
    checker.cpus = None
    checker.volumes = None  # per-agent boxes share the agents' files on volumes; the checker sees none of them
    env = checker.environment or {}
    if isinstance(env, list):
        env = dict(e.split("=", 1) if "=" in e else (e, None) for e in env)
    checker.environment = {**env, **TIMING_ENV}
    checker.mem_limit = f"{resources['checker_memory_mb']}m"
    checker.__pydantic_extra__["cpuset"] = resources["checker_cpuset"]
    config.services[CHECKER] = checker


def _limit_agent_box(config: ComposeConfig, resources: dict):
    """Set the agent box's CPUs, memory and cpuset from `box_resources`; returns the service.

    Per-agent boxes (resources["agent_boxes"]): `default` is agent_0's box and `agent_<i>` a copy for each other agent,
    each with its own CPUs, cpuset and memory, and all with SHARED_VOLUMES mounted.
    """
    default = config.services["default"]
    boxes = resources.get("agent_boxes")
    if boxes is None:
        default.cpus = float(resources["agent_cpus"])
        default.mem_limit = f"{resources['agent_memory_mb']}m"
        default.__pydantic_extra__["cpuset"] = resources["agent_cpuset"]
        return default
    config.volumes = {**(config.volumes or {}), **{name: {} for name in SHARED_VOLUMES}}
    default.volumes = [*(default.volumes or []), *(f"{name}:{path}" for name, path in SHARED_VOLUMES.items())]
    for box in boxes:
        service = default if box["service"] == "default" else default.model_copy(deep=True)
        service.cpus = float(box["cpus"])
        service.mem_limit = f"{box['memory_mb']}m"
        service.__pydantic_extra__["cpuset"] = box["cpuset"]
        config.services[box["service"]] = service
    return default


def seed_offset_id(offset: int) -> str:
    """A short fingerprint of the seed offset for logs: shows which offset scored a run without stating it.

    The offset range is small enough to brute-force from this, so it only keeps the offset out of plain sight.
    """
    return hashlib.sha256(f"algotune-seed-offset:{offset}".encode()).hexdigest()[:16]


def seed_verifier(sample, offset: int, experiment: str = "v6") -> Path:
    """Write our patched copy of the sample's Harbor tests (seed offset, thread check, first-call ratio, alone score,
    the experiment's TASK_PATCHES in evaluator.py) with the offset file, point the sample's test_path and tests_dir at
    it, and return it. The hub cache is shared between runs, so the seeded verifier is a copy."""
    tests_dir = Path(sample.metadata["tests_dir"])
    seeded_dir = VERIFIER_DIR / str(sample.id).replace("/", "_")
    shutil.rmtree(seeded_dir, ignore_errors=True)
    shutil.copytree(tests_dir, seeded_dir)
    evaluator = seeded_dir / "evaluator.py"
    evaluator.write_text(patch_task(task_names()[str(sample.id)], evaluator.read_text(), experiment))
    test_outputs = seeded_dir / "test_outputs.py"
    test_outputs.write_text(with_alone_score(with_first_call_ratio(with_thread_guard(seed_with_offset(test_outputs.read_text())))))
    shutil.copy(ASSETS / "thread_guard.py", seeded_dir)
    (seeded_dir / "seed_offset").write_text(f"{offset}\n")
    sample.metadata["test_path"] = str(seeded_dir / Path(sample.metadata["test_path"]).relative_to(tests_dir))
    sample.metadata["tests_dir"] = str(seeded_dir)
    return seeded_dir


def task_names() -> dict[str, str]:
    """Harbor sample id (algotune/dst-type-ii-scipy-fftpack) -> AlgoTune task name (dst_type_II_scipy_fftpack)."""
    split = json.loads(SPLIT_PATH.read_text())
    return {_harbor_name(n): n for key in SPLITS for n in split[key]}


def _harbor_name(task_name: str) -> str:
    """AlgoTune's snake_case name to the Harbor hub slug."""
    return "algotune/" + task_name.replace("_", "-").lower()


@task
def algotune_task(
    split: Literal["pilot", "heldout", "v7"],
    cpus_per_agent: int = 4,
    n_agents: int = 1,
    parallel: int = 1,
    checker_backend: Literal["local", "remote"] = "local",
    isolate_agents: bool = False,
    memory_per_agent_mb: int | None = None,
    experiment: str = "v6",
    primary_score: Literal["interleaved", "alone"] | None = None,
) -> Task:
    """The inspect_harbor AlgoTune task restricted to one split of data/algotune_split.json.

    Every sample gets the dev toolkit (`algotune_setup`, a Task.setup step that still runs when
    `--solver` replaces the default agent) and a sandbox without network, plus a `checker` service
    (`add_checker`) on its own CPUs for the dev_eval tool, final selection and scoring. The scorer is
    Harbor's, run in the checker, with the AlgoTune metrics added. CPUs and memory come from `box_resources`
    for the Docker host this runs against (cpus_per_agent x n_agents CPUs for the agent box, 8 for the
    checker, `parallel` samples at once) and are recorded in the task metadata.

    The verifier scores on seeds `offset + i` instead of `i` (`seed_with_offset`) and marks a run invalid when
    solver threads use CPU while the reference is timed (`with_thread_guard`). The offset reaches the
    container only at scoring time, as a `seed_offset` file in the copied /tests that the verifier deletes
    before importing the solver, and is recorded in the task metadata. The hub dataset is pinned (ALGOTUNE_REF).

    checker_backend="remote": every timed run goes to the scorer service at SCORER_URL (scorer_service.py), which
    holds the seed offset and the seeded verifier. The sample gets no checker service, nothing is seeded here,
    and the task metadata records the service's /health (CPU model, version, seed offset fingerprint) instead of
    the offset.

    isolate_agents (v7): one box per agent (`box_resources`), each with cpus_per_agent CPUs on its own cpuset and
    memory_per_agent_mb, the agents' files on shared volumes; sample metadata "agent_boxes" lists the services by agent
    index, and algotune_setup installs the toolkit and CPU_COUNT_SITECUSTOMIZE in each.

    experiment: whose secret offset scores the hidden instances (`seed_offset_file`); a remote scorer must serve the
    same experiment. primary_score: what Score.value is (algotune_scorer); default "alone" for v7, else "interleaved".
    """
    names = json.loads(SPLIT_PATH.read_text())[split]
    is_remote = checker_backend == "remote"
    resources = box_resources(cpus_per_agent, n_agents, parallel, *docker_host(), checker=not is_remote,
                              isolate=isolate_agents, memory_per_agent_mb=memory_per_agent_mb)  # fmt: skip
    base = algotune(
        ref=ALGOTUNE_REF,
        dataset_task_names=[_harbor_name(n) for n in names],
        override_cpus=resources["agent_cpus"],
        override_memory_mb=resources["agent_memory_mb"],
    )
    if primary_score is None:
        primary_score = "alone" if experiment == "v7" else "interleaved"
    if is_remote:
        backend_meta = {"checker_backend": "remote", "scorer": scorer_client.health()}
        backend_meta["seed_offset_id"] = backend_meta["scorer"]["seed_offset_id"]
        served = backend_meta["scorer"].get("experiment", "v6")  # scorers before v7 served only v6's offset
        if served != experiment:
            raise ValueError(f"this run is experiment {experiment}, but the scorer at SCORER_URL serves {served}'s offset")
    else:
        offset = _seed_offset(seed_offset_file(experiment), experiment=experiment)
        backend_meta = {"checker_backend": "local", "algotune_seed_offset": offset, "seed_offset_id": seed_offset_id(offset)}
    backend_meta |= {"experiment": experiment, "primary_score": primary_score}
    names_by_id = task_names()
    images = set()
    for sample in base.dataset:
        config = sample.sandbox.config
        assert isinstance(config, ComposeConfig)
        # inspect_harbor reads network_mode from task.toml only (all 154 say "public") and has no override.
        service = config.services["default"]
        service.network_mode = "none"
        # Use the already-built image and never rebuild: a rebuild needs PyPI, and a pip read timeout
        # broke one on 2026-10-08. One fixed image for the whole experiment is also more reproducible.
        # Build it once with: docker build -t <image> <task>/environment
        images.add(service.image)
        if service.image:
            service.build = None
            service.__pydantic_extra__["x-local"] = True
        if is_remote:
            _limit_agent_box(config, resources)
        else:
            add_checker(config, resources)
        sample.metadata["checker_backend"] = checker_backend
        sample.metadata["algotune_task_name"] = names_by_id[str(sample.id)]
        sample.metadata["primary_score"] = primary_score
        sample.metadata["experiment"] = experiment  # algotune_devkit's toolkit applies the experiment's TASK_PATCHES
        sample.input = patch_task(names_by_id[str(sample.id)], sample.input, experiment, instruction=True)
        if isolate_agents:
            sample.metadata["agent_boxes"] = [b["service"] for b in resources["agent_boxes"]]
        # Passed as a file, not inline: logs store each sample's sandbox config, and Inspect re-validates an
        # inline ComposeConfig when reading them, which rejects cpuset (only x- extras are allowed), so every
        # log would be unreadable. The YAML is what Inspect would generate; it is also kept in sample metadata.
        compose_yaml = yaml.dump(
            config.model_dump(mode="json", by_alias=True, exclude_none=True), default_flow_style=False, sort_keys=False
        )
        compose_file = COMPOSE_DIR / f"{str(sample.id).replace('/', '_')}-compose.yaml"
        compose_file.parent.mkdir(parents=True, exist_ok=True)
        compose_file.write_text(compose_yaml)
        sample.sandbox = SandboxEnvironmentSpec(sample.sandbox.type, str(compose_file))
        sample.metadata["compose_yaml"] = compose_yaml
        if not is_remote:  # only the local scorer reads it
            seed_verifier(sample, offset, experiment)
    image_meta = {"task_images": {image: task_image_digest(image) for image in sorted(images)}}
    if len(images) == 1:
        (image,) = images
        image_meta = {"task_image": image, "task_image_digest": image_meta["task_images"][image]}
    return task_with(
        base,
        setup=algotune_setup(),
        scorer=algotune_scorer(),
        metadata={**(base.metadata or {}), **backend_meta, **resources, **image_meta},
    )
