"""Task factories for the swarm-scaling experiments."""

import json
import math
import os
import secrets
import shutil
import time
from pathlib import Path
from typing import Literal

import yaml
from inspect_ai import Task, task, task_with
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, mean, metric, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec, sandbox, sandbox_default
from inspect_harbor import algotune, harbor_scorer

from swarm_scaling.algotune_devkit import (
    ASSETS,
    CHECKER,
    CHECKER_LOCK,
    SOLVER_PATH,
    algotune_setup,
    end_checker_processes,
)

SPLIT_PATH = Path(__file__).resolve().parents[2] / "data" / "algotune_split.json"

# Harbor's verifier scores on generate_problem(random_seed=i) for i in 0..99, and agents hold generate_problem.
# The offset is added to those seeds. Dev instances use seeds 10,000+ and selection 60,000+; np.random.seed needs < 2**32.
SEED_OFFSET_ENV = "ALGOTUNE_SEED_OFFSET"
SEED_OFFSET_RANGE = (1_000_000, 2**31 - 1000)
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
DOCKER_VM_CPUS = 16  # Docker Desktop's VM on this Mac; the agent box and the checker each get half
CHECKER_MEMORY_MB = 8192
AGENT_MEMORY_MB = 12288  # agent box when one sample runs at a time (task.toml says 16 GB)
DOCKER_VM_MEMORY_MB = 23_492  # MemTotal of the Docker VM (24,056,148 kB, measured 2026-10-09)
# The VM's own use with no container running (970 MiB measured 2026-10-09), idle checkers and margin.
DOCKER_VM_RESERVE_MB = 1536

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
    """
    verify = harbor_scorer()

    async def score(state: TaskState, target: Target) -> Score:
        requested = time.time()
        async with CHECKER_LOCK:
            started = time.time()
            cleanup = await end_checker_processes(state)
            await _copy_solver_to_checker()
            with sandbox_default(CHECKER):
                result = await verify(state, target)
            ended = time.time()
        result.metadata = {
            **(result.metadata or {}),
            "log_speedup": math.log(_speedup(result.as_float())),
            "checker": {
                "queue_wait_s": round(started - requested, 3), "started_at": started, "ended_at": ended, "cleanup": cleanup
            },
        }
        return result

    return score


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
    + "from thread_guard import ThreadGuard\n\n"
    + "_GUARD = None  # set in problem_set: after an untimed reference call, before the solver is imported\n",
    '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n    return problems\n': (
        '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n'
        "    global _GUARD\n"
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
    "    if validity and total_time_solver > 0:\n": (
        "    logger.info(f\"Thread check: {_GUARD.summary()}\")\n"
        "    if _GUARD.error() is not None:\n"
        "        logger.error(_GUARD.error())\n"
        "        validity = False\n"
        "    if validity and total_time_solver > 0:\n"
    ),
}


def with_thread_guard(test_outputs: str) -> str:
    """The verifier with thread_guard.py's check: a solver whose threads use CPU during reference timing is invalid.

    Pilot 3: a solver that left spin-waiting threads running scored 3837x on dev_eval by slowing the reference.
    Adds one untimed reference call (on the first problem, before the solver is imported, so the threads the
    reference starts count as its own), wraps every reference call in the guard, and sets validity False when
    the check fails, which the verifier scores 1.0 like any invalid run. Timing and the speedup formula are
    unchanged. Needs thread_guard.py next to the verifier. Raises unless every anchor occurs exactly once.
    """
    for anchor, replacement in _GUARD_PATCHES.items():
        if test_outputs.count(anchor) != 1:
            raise ValueError(f"expected exactly one {anchor!r} in the AlgoTune verifier")
        test_outputs = test_outputs.replace(anchor, replacement)
    return test_outputs


SEED_OFFSET_FILE = SPLIT_PATH.parent / ".algotune_seed_offset"  # gitignored; one secret for the whole experiment


def _offsets_in_logs(log_dir: Path) -> set[int]:
    """Seed offsets recorded in existing AlgoTune eval logs (eval.metadata["algotune_seed_offset"])."""
    from inspect_ai.log import read_eval_log

    found: set[int] = set()
    for log in sorted(log_dir.glob("*.eval")) if log_dir.is_dir() else []:
        try:
            header = read_eval_log(str(log), header_only=True)
        except Exception:
            continue
        value = (header.eval.metadata or {}).get("algotune_seed_offset")
        if value is not None:
            found.add(int(value))
    return found


def _seed_offset(path: Path = SEED_OFFSET_FILE, log_dir: Path = LOG_DIR) -> int:
    """One experiment-wide secret offset, so every arm and repeat is scored on the same instances.

    ALGOTUNE_SEED_OFFSET overrides. Otherwise the offset is read from `path`. If `path` is missing it is
    restored from the offset recorded in existing eval logs; only when there are none is a new random
    offset written. Creation is atomic, so two runs started at once cannot write different offsets.
    A fresh offset per run would score separately launched arms on different instances and break
    paired per-task comparisons.
    """
    low, high = SEED_OFFSET_RANGE
    if os.environ.get(SEED_OFFSET_ENV):
        offset = int(os.environ[SEED_OFFSET_ENV])
    else:
        if not path.exists():
            logged = _offsets_in_logs(log_dir)
            if len(logged) > 1:
                raise ValueError(f"{path} is missing and the logs in {log_dir} record several offsets: {sorted(logged)}")
            new = logged.pop() if logged else low + secrets.randbelow(high - low)
            try:
                with path.open("x") as f:  # atomic: fails if another run created it first
                    f.write(f"{new}\n")
            except FileExistsError:
                pass
        offset = int(path.read_text().strip())
    if not low <= offset <= high:
        raise ValueError(f"seed offset {offset} (from {SEED_OFFSET_ENV} or {path}) is outside {low}..{high}")
    return offset


def add_checker(config: ComposeConfig) -> None:
    """Add the `checker` service: a copy of `default` that only the dev_eval tool and final selection use.

    The two are pinned to disjoint CPUs of the Docker VM (default 0..n-1, checker n..2n-1, n = the
    default's cpus), so agent work in the default box cannot disturb the checker's timings. Final
    scoring runs in the checker too (algotune_scorer), on the same CPUs and thread settings.
    ComposeService has no cpuset field; extras set after construction reach the generated YAML.
    """
    default = config.services["default"]
    n = int(default.cpus or 0)
    if not 0 < 2 * n <= DOCKER_VM_CPUS:
        raise ValueError(f"cannot pin two boxes of {default.cpus} CPUs inside {DOCKER_VM_CPUS} CPUs")
    checker = default.model_copy(deep=True)
    # Two 12 GB boxes would fill the 24.6 GB Docker VM; one dev_eval process needs far less.
    checker.mem_limit = f"{CHECKER_MEMORY_MB}m"
    default.__pydantic_extra__["cpuset"] = f"0-{n - 1}"
    checker.__pydantic_extra__["cpuset"] = f"{n}-{2 * n - 1}"
    config.services[CHECKER] = checker


def agent_memory_mb(parallel: int) -> int:
    """Agent box memory limit (MiB) when `parallel` samples run at once.

    Only one checker runs a timed evaluation at a time (CHECKER_LOCK), so the agent boxes share what is
    left of the VM after its own use and one busy checker at its full limit.
    """
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    return min(AGENT_MEMORY_MB, (DOCKER_VM_MEMORY_MB - DOCKER_VM_RESERVE_MB - CHECKER_MEMORY_MB) // parallel)


def _harbor_name(task_name: str) -> str:
    """AlgoTune's snake_case name to the Harbor hub slug."""
    return "algotune/" + task_name.replace("_", "-").lower()


@task
def algotune_task(
    split: Literal["pilot", "heldout"],
    override_cpus: int | None = 8,
    override_memory_mb: int | None = AGENT_MEMORY_MB,  # task.toml says 16 GB; Docker Desktop has 24 GB
) -> Task:
    """The inspect_harbor AlgoTune task restricted to one split of data/algotune_split.json.

    Every sample gets the dev toolkit (`algotune_setup`, a Task.setup step that still runs when
    `--solver` replaces the default agent) and a sandbox without network, plus a `checker` service
    (`add_checker`) on its own CPUs for the dev_eval tool, final selection and scoring. The scorer is
    Harbor's, run in the checker, with the AlgoTune metrics added. Defaults to 8 CPUs and 12 GB (task.toml
    asks for 16 GB; Docker has 24 GB).

    The verifier scores on seeds `offset + i` instead of `i` (`seed_with_offset`) and marks a run invalid when
    solver threads use CPU while the reference is timed (`with_thread_guard`). The offset reaches the
    container only at scoring time, as a `seed_offset` file in the copied /tests that the verifier deletes
    before importing the solver, and is recorded in the task metadata. The hub dataset is pinned (ALGOTUNE_REF).
    """
    names = json.loads(SPLIT_PATH.read_text())[split]
    base = algotune(
        ref=ALGOTUNE_REF,
        dataset_task_names=[_harbor_name(n) for n in names],
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
    )
    offset = _seed_offset()
    for sample in base.dataset:
        config = sample.sandbox.config
        assert isinstance(config, ComposeConfig)
        # inspect_harbor reads network_mode from task.toml only (all 154 say "public") and has no override.
        service = config.services["default"]
        service.network_mode = "none"
        # Use the already-built image and never rebuild: a rebuild needs PyPI, and a pip read timeout
        # broke one on 2026-10-08. One fixed image for the whole experiment is also more reproducible.
        # Build it once with: docker build -t <image> <task>/environment
        if service.image:
            service.build = None
            service.__pydantic_extra__["x-local"] = True
        add_checker(config)
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
        # The hub cache is shared between runs, so the seeded verifier is a copy; only the scorer reads it.
        tests_dir = Path(sample.metadata["tests_dir"])
        seeded_dir = VERIFIER_DIR / str(sample.id).replace("/", "_")
        shutil.rmtree(seeded_dir, ignore_errors=True)
        shutil.copytree(tests_dir, seeded_dir)
        test_outputs = seeded_dir / "test_outputs.py"
        test_outputs.write_text(with_thread_guard(seed_with_offset(test_outputs.read_text())))
        shutil.copy(ASSETS / "thread_guard.py", seeded_dir)
        (seeded_dir / "seed_offset").write_text(f"{offset}\n")
        sample.metadata["test_path"] = str(seeded_dir / Path(sample.metadata["test_path"]).relative_to(tests_dir))
        sample.metadata["tests_dir"] = str(seeded_dir)
    return task_with(
        base, setup=algotune_setup(), scorer=algotune_scorer(), metadata={**(base.metadata or {}), "algotune_seed_offset": offset}
    )
