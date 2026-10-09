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
    SOLVER_PATH,
    algotune_setup,
    end_checker_processes,
    remote,
    remote_timing,
    scorer_job,
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
CHECKER_CPUS = 8
CHECKER_MEMORY_MB = 8192  # the verifier's measured peak is 4.2 GiB (dst-type-ii-scipy-fftpack, 2026-10-09)
# Left for the Docker host itself (970 MiB measured on the Mac's Docker VM with no containers, 2026-10-09).
HOST_HEADROOM_MB = 2048

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
        result.metadata = {
            **(result.metadata or {}),
            "scoring_timeout": timed_out,
            "first_call_ratio": float(ratio.group(1)) if ratio and ratio.group(1) != "None" else None,
            "reference_inflation": float(inflation.group(1)) if inflation and inflation.group(1) != "None" else None,
            "log_speedup": math.log(_speedup(result.as_float())),
            "checker": {
                "queue_wait_s": round(started - requested, 3), "started_at": started, "ended_at": ended, "cleanup": cleanup
            },
        }
        return result

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
    return Score(
        value=r["score"],
        explanation=r["verifier_stdout"],
        metadata={
            "log_speedup": math.log(_speedup(r["score"])),
            "scoring_timeout": r["scoring_timeout"],
            "first_call_ratio": r["first_call_ratio"],
            "reference_inflation": r.get("reference_inflation"),
            "checker": remote_timing(rec),
        },
    )


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
    + "from thread_guard import ThreadGuard, reference_inflation, reference_inflation_error, time_reference\n\n"
    + "_GUARD = None  # set in problem_set, after the reference was timed alone and before the solver is imported\n"
    + "_ALONE = []  # per problem: the reference's min timed call before the solver is imported (ns)\n"
    + "_WITH_SOLVER = []  # per timed problem: the reference's min timed call interleaved with the solver (ns)\n",
    '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n    return problems\n': (
        '    logger.info(f"All {NUM_TEST_INSTANCES} problems generated.")\n'
        "    global _GUARD, _ALONE\n"
        "    _ALONE = time_reference(task.solve, problems, NUM_REPEATS)\n"
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
    ),
    "    if validity and total_time_solver > 0:\n": (
        "    _alone = _ALONE[: len(_WITH_SOLVER)]\n"
        "    print(f\"Thread check: {_GUARD.summary()}\")\n"
        "    print(f\"Reference inflation: {reference_inflation(_WITH_SOLVER, _alone)}\")\n"
        "    if reference_inflation_error(_WITH_SOLVER, _alone) is not None:\n"
        "        logger.error(reference_inflation_error(_WITH_SOLVER, _alone))\n"
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
    Times the reference on every problem before the solver fixture imports the solver (one untimed and
    NUM_REPEATS timed calls each, as the verifier times it), records the reference's per-instance minimum during
    the interleaved timing, prints "Reference inflation: <ratio>" and the thread-CPU summary, and sets validity
    False when the check fails, which the verifier scores 1.0 like any invalid run. Timing and the speedup
    formula are unchanged. Needs thread_guard.py next to the verifier. Raises unless every anchor occurs once.
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
    cpus_per_agent: int, n_agents: int, parallel: int, host_cpus: int, host_memory_mb: int, checker: bool = True
) -> dict:
    """CPUs, cpusets and memory limits of the agent box and the checker on a Docker host of this size.

    Every agent gets the same CPUs: the agent box has cpus_per_agent x n_agents CPUs (0..A-1), the checker
    CHECKER_CPUS (A..A+7), so agent work cannot disturb the checker's timings. Only one checker times at a
    time (CHECKER_LOCK), so the `parallel` agent boxes share the host memory left after one checker and
    HOST_HEADROOM_MB. With parallel > 1 the agent boxes share cpuset 0..A-1 and the checkers A..A+7.
    checker=False (remote scorer): no checker on this host, so nothing is reserved for it.
    """
    if min(cpus_per_agent, n_agents, parallel) < 1:
        raise ValueError("cpus_per_agent, n_agents and parallel must be >= 1")
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


def add_checker(config: ComposeConfig, resources: dict) -> None:
    """Add the `checker` service, a copy of `default` that only the dev_eval tool, final selection and scoring use.

    Pins both boxes to their cpusets from `box_resources` and sets their CPU and memory limits.
    ComposeService has no cpuset field; extras set after construction reach the generated YAML.
    """
    default = _limit_agent_box(config, resources)
    checker = default.model_copy(deep=True)
    checker.cpus = float(resources["checker_cpus"])
    checker.mem_limit = f"{resources['checker_memory_mb']}m"
    checker.__pydantic_extra__["cpuset"] = resources["checker_cpuset"]
    config.services[CHECKER] = checker


def _limit_agent_box(config: ComposeConfig, resources: dict):
    """Set the agent box's CPUs, memory and cpuset from `box_resources`; returns the service."""
    default = config.services["default"]
    default.cpus = float(resources["agent_cpus"])
    default.mem_limit = f"{resources['agent_memory_mb']}m"
    default.__pydantic_extra__["cpuset"] = resources["agent_cpuset"]
    return default


def seed_offset_id(offset: int) -> str:
    """A short fingerprint of the seed offset for logs: shows which offset scored a run without stating it.

    The offset range is small enough to brute-force from this, so it only keeps the offset out of plain sight.
    """
    return hashlib.sha256(f"algotune-seed-offset:{offset}".encode()).hexdigest()[:16]


def seed_verifier(sample, offset: int) -> Path:
    """Write our patched copy of the sample's Harbor tests (seed offset, thread check, first-call ratio) with the
    offset file, point the sample's test_path and tests_dir at it, and return it. The hub cache is shared between
    runs, so the seeded verifier is a copy."""
    tests_dir = Path(sample.metadata["tests_dir"])
    seeded_dir = VERIFIER_DIR / str(sample.id).replace("/", "_")
    shutil.rmtree(seeded_dir, ignore_errors=True)
    shutil.copytree(tests_dir, seeded_dir)
    test_outputs = seeded_dir / "test_outputs.py"
    test_outputs.write_text(with_first_call_ratio(with_thread_guard(seed_with_offset(test_outputs.read_text()))))
    shutil.copy(ASSETS / "thread_guard.py", seeded_dir)
    (seeded_dir / "seed_offset").write_text(f"{offset}\n")
    sample.metadata["test_path"] = str(seeded_dir / Path(sample.metadata["test_path"]).relative_to(tests_dir))
    sample.metadata["tests_dir"] = str(seeded_dir)
    return seeded_dir


def task_names() -> dict[str, str]:
    """Harbor sample id (algotune/dst-type-ii-scipy-fftpack) -> AlgoTune task name (dst_type_II_scipy_fftpack)."""
    split = json.loads(SPLIT_PATH.read_text())
    return {_harbor_name(n): n for key in ("pilot", "heldout") for n in split[key]}


def _harbor_name(task_name: str) -> str:
    """AlgoTune's snake_case name to the Harbor hub slug."""
    return "algotune/" + task_name.replace("_", "-").lower()


@task
def algotune_task(
    split: Literal["pilot", "heldout"],
    cpus_per_agent: int = 4,
    n_agents: int = 1,
    parallel: int = 1,
    checker_backend: Literal["local", "remote"] = "local",
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
    """
    names = json.loads(SPLIT_PATH.read_text())[split]
    is_remote = checker_backend == "remote"
    resources = box_resources(cpus_per_agent, n_agents, parallel, *docker_host(), checker=not is_remote)
    base = algotune(
        ref=ALGOTUNE_REF,
        dataset_task_names=[_harbor_name(n) for n in names],
        override_cpus=resources["agent_cpus"],
        override_memory_mb=resources["agent_memory_mb"],
    )
    if is_remote:
        backend_meta = {"checker_backend": "remote", "scorer": scorer_client.health()}
        backend_meta["seed_offset_id"] = backend_meta["scorer"]["seed_offset_id"]
    else:
        offset = _seed_offset()
        backend_meta = {"checker_backend": "local", "algotune_seed_offset": offset, "seed_offset_id": seed_offset_id(offset)}
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
            seed_verifier(sample, offset)
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
