"""AlgoTune dev toolkit and final selection (HARNESS.md, AlgoTune section).

`algotune_setup()` runs before any agent and installs /app/dev in the agent box and the checker.
`algotune_agent_tools()` gives every agent the dev_eval tool, which runs dev_eval.py in the
checker container (tasks.py).
`algotune_finalize()` runs after the agents, evaluates every candidate solver on fixed dev
instances in the checker and installs the fastest correct one as /app/solver.py in the agent box.
Every timed run in a checker (dev_eval calls, finalize evaluations, final scoring in tasks.py) holds
CHECKER_LOCK, so across all samples running in this process only one is timed at any moment, and first
ends every checker process started since setup (`end_checker_processes`).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import anyio
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError
from inspect_ai.util import sandbox

from swarm_scaling import scorer_client
from swarm_scaling.swarm import Candidate, agent_box, kill_agent_processes, snapshot_container_pids

ASSETS = Path(__file__).parent / "algotune_assets"
DEV_DIR = "/app/dev"
SOLVER_PATH = "/app/solver.py"

# Compose service (tasks.py) where every evaluation runs. Agents' bash and python tools reach only the default box.
CHECKER = "checker"
CHECK_DIR = "/app/checks"  # in the checker: a fresh directory per evaluation
# state.metadata key: the checker's PIDs at setup (its init and keepalive), spared by end_checker_processes
CHECKER_PIDS = "checker_baseline_pids"
DEV_EVAL_TIMEOUT_S = 900  # per dev_eval tool call, run time only (not the queue wait)

# One queue for the whole process: parallel samples' checkers share CPUs 8-15, so their timings must not overlap.
CHECKER_LOCK = anyio.Lock()  # waiters are served in arrival order (FIFO)

# glibc heap trimming off in every timed process (2026-10-10, scoring investigation): with the defaults the
# reference's temporaries were either recycled heap (0 page faults, 53-55 ms on generalized_eigenvalues_real) or
# trimmed and refaulted (4,600-5,800 faults, 63-72 ms) depending on the heap layout the solver left, so the
# reference check measured the allocator. glibc reads these when the process starts, so they go in the container's
# environment (scorer `docker run -e`, the local checker's compose environment), never from inside Python; the
# alone-timer child inherits them from dev_eval or the verifier.
TIMING_ENV = {
    "MALLOC_MMAP_THRESHOLD_": "33554432",
    "MALLOC_TRIM_THRESHOLD_": "68719476736",
    "MALLOC_TOP_PAD_": "268435456",
}

# Selection runs on its own dev instances: dev_eval's default seeds start at 0, these start far away.
FINAL_DEV_SEED = 50_000
FINAL_DEV_N = 20
FINAL_TIMEOUT_S = 900  # per candidate
# Stage 2 of scripts/select_portfolio.py: the shortlist is re-timed by the final verifier (scorer job kind score_dev) on
# instances SELECT_SEED + i, i < 100: away from dev_eval's (10,000+), finalize's (60,000+) and the hidden ones (1,000,000+).
SELECT_SEED = 500_000
# dev_eval refuses seed + n at or above this, so an agent's seed sweep never reaches the stage-2 instances (pilot agents
# swept seeds up to 104,000 while waiting out the submit gate).
DEV_SEED_CAP = 400_000
# v7 (runner --disjoint-dev-seeds): seed + n below FINAL_DEV_SEED, so dev_eval also stays off the finalize (stage-1)
# instances; under DEV_SEED_CAP five v6 dev_eval calls timed them.
DISJOINT_DEV_SEED_CAP = FINAL_DEV_SEED
DEV_SEED_BASE = 10_000  # dev_eval.py's SEED_OFFSET: dev_eval and finalize instance i is seed DEV_SEED_BASE + seed + i
# Validity pass (scripts/select_portfolio.py --validity-n, scorer job kind validity): untimed is_solution checks of
# each shortlisted source on instances VALIDITY_SEED + i, the same instances for every source and arm.
VALIDITY_SEED = 600_000
VALIDITY_MAX_N = 100_000
# The hidden instances are offset + i, i < 100, for the experiment's secret offset in this range (tasks._seed_offset).
SEED_OFFSET_RANGE = (1_000_000, 2**31 - 1000)
VERIFIER_INSTANCES = 100  # Harbor's AlgoTune verifier: NUM_TEST_INSTANCES (stage 2 and hidden)


def seed_ranges(dev_seed_cap: int = DEV_SEED_CAP) -> dict[str, tuple[int, int]]:
    """Every range of generate_problem seeds the experiment uses, [first, last + 1), by purpose: the one place they
    are written down. Disjoint with dev_seed_cap = DISJOINT_DEV_SEED_CAP (tests/test_algotune_seeds.py); the v6 cap
    lets dev_eval reach the finalize instances.

      dev_eval   agents' dev_eval tool: DEV_SEED_BASE + seed + i, with seed >= 0 and seed + n below dev_seed_cap
      finalize   stage 1, every candidate at the end of a run: DEV_SEED_BASE + FINAL_DEV_SEED + i, i < FINAL_DEV_N
      score_dev  stage 2 of selection: SELECT_SEED + i, i < 100
      validity   selection's validity pass: VALIDITY_SEED + i, i < the pass's n (at most VALIDITY_MAX_N)
      hidden     scoring: offset + i, i < 100, offset secret and in SEED_OFFSET_RANGE (one offset per experiment)
    """
    return {
        "dev_eval": (DEV_SEED_BASE, DEV_SEED_BASE + dev_seed_cap - 1),
        "finalize": (DEV_SEED_BASE + FINAL_DEV_SEED, DEV_SEED_BASE + FINAL_DEV_SEED + FINAL_DEV_N),
        "score_dev": (SELECT_SEED, SELECT_SEED + VERIFIER_INSTANCES),
        "validity": (VALIDITY_SEED, VALIDITY_SEED + VALIDITY_MAX_N),
        "hidden": (SEED_OFFSET_RANGE[0], SEED_OFFSET_RANGE[1] + VERIFIER_INSTANCES),
    }

DEV_TOOLKIT_NOTE = (
    "A dev toolkit is installed in /app/dev (read /app/dev/README.md first). It contains the full "
    "reference Task class with generate_problem. The dev_eval tool checks a solver for correctness and "
    "measures its speedup the way the final evaluation does. It runs on a separate, dedicated machine, one "
    "evaluation at a time: calls from every agent working on this task wait in one queue, and each result "
    "says how long it waited. Timings you take in your own container are affected by whatever else is "
    "running there. A run in which the reference, timed alongside your solver, is more than 15% slower than when "
    "timed alone on the same instances is scored invalid."
)


async def _install_toolkit(state: TaskState) -> None:
    from swarm_scaling.tasks import patch_task

    # Only evaluator.py is read from the verifier files: no test_outputs.py, no test instances.
    evaluator = patch_task(
        state.metadata.get("algotune_task_name", ""), (Path(state.metadata["tests_dir"]) / "evaluator.py").read_text(),
        state.metadata.get("experiment", "v6"),
    )  # fmt: skip
    problem_size = state.metadata["harbor_config"]["metadata"]["algotune_problem_size"]

    for name in state.metadata.get("agent_boxes") or [None]:  # per-agent boxes: each gets its own copy
        box = sandbox(name)
        await box.write_file(f"{DEV_DIR}/reference_task.py", evaluator)
        await box.write_file(f"{DEV_DIR}/README.md", (ASSETS / "README.md").read_text())
    if remote(state):  # the scorer service holds its own copy
        return

    checker = sandbox(CHECKER)
    await checker.write_file(f"{DEV_DIR}/reference_task.py", evaluator)
    await checker.write_file(f"{DEV_DIR}/dev_eval.py", (ASSETS / "dev_eval.py").read_text())
    await checker.write_file(f"{DEV_DIR}/thread_guard.py", (ASSETS / "thread_guard.py").read_text())
    await checker.write_file(f"{DEV_DIR}/config.json", json.dumps({"problem_size": problem_size}))


async def _install_cpu_count(name: str) -> None:
    """Per-agent boxes: os.cpu_count() reports the box's CPUs (tasks.CPU_COUNT_SITECUSTOMIZE) in every Python run there."""
    from swarm_scaling.tasks import CPU_COUNT_SITECUSTOMIZE

    box = sandbox(name)
    where = await box.exec(["python3", "-c", "import site; print(site.getsitepackages()[0])"])
    if not where.success:
        raise RuntimeError(f"could not find site-packages in {name}: {where.stderr}")
    path = f"{where.stdout.strip()}/sitecustomize.py"
    if (await box.exec(["test", "-e", path])).success:
        raise RuntimeError(f"{name} already has {path}; os.cpu_count() cannot be set without replacing it")
    await box.write_file(path, CPU_COUNT_SITECUSTOMIZE)


@solver
def algotune_setup() -> Solver:
    """Install the dev toolkit in /app/dev and point the task prompt at it."""

    async def setup(state: TaskState, generate: Generate) -> TaskState:
        await _install_toolkit(state)
        for name in state.metadata.get("agent_boxes") or []:
            await _install_cpu_count(name)
        if not remote(state):
            # before any solver has run, so only the checker's own processes are spared later
            state.metadata[CHECKER_PIDS] = await snapshot_container_pids(CHECKER)
        state.user_prompt.text += f"\n\n{DEV_TOOLKIT_NOTE}"
        return state

    return setup


@dataclass
class DevResult:
    candidate: Candidate
    valid: bool
    speedup: float | None  # reference time / solver time on the dev instances; None if nothing was timed
    error: str = ""
    detail: dict = field(default_factory=dict)


def pick_best(results: list[DevResult]) -> DevResult | None:
    """Fastest correct candidate; ties go to the earliest published, then published over final workspace."""
    correct = [r for r in results if r.valid and r.speedup is not None]
    if not correct:
        return None
    return min(
        correct,
        key=lambda r: (
            -r.speedup,
            r.candidate.published_at,
            r.candidate.kind != "published",
            r.candidate.agent_id,
            r.candidate.path,
        ),
    )


def remote(state: TaskState) -> bool:
    """True when this sample's timed runs go to the remote scorer service (algotune_task checker_backend)."""
    return state.metadata.get("checker_backend", "local") == "remote"


def scorer_job(state: TaskState, kind: str, source: bytes, agent_id: str = "", **args: Any) -> dict:
    """A scorer_service job for this sample."""
    return {
        "kind": kind,
        "task": state.metadata["algotune_task_name"],
        "solver_source": source.decode(errors="replace"),
        "run_id": os.environ.get("SWARM_RUN_ID", ""),
        "agent_id": agent_id,
        "sample_id": f"{state.sample_id}/e{state.epoch}",
        **args,
    }


def remote_timing(rec: dict) -> dict:
    """Telemetry of a finished scorer job, in the shape of the local checker's plus where it ran."""
    keep = ("queue_wait_s", "started_at", "ended_at", "run_s", "slot", "cpu_model", "scorer_host", "job_id", "alone_baseline",
            "timing_env", "session", "exclusive")
    return {k: rec.get(k) for k in keep}


VENDORS = {"AuthenticAMD": "AMD", "GenuineIntel": "Intel"}


def scorer_hardware_text(health: dict) -> str:
    """The paragraph every agent's prompt gets about the machine that times its candidates (v7, 2026-10-18), from the
    scorer's /health at run start. In the calibration 9 of 21 agents learned the scorer's 8 cores and AVX-512 only by
    encoding them into solve() run times through dev_eval, and the one independent on stable_matching that did scored
    twice the other three. Raises when /health lacks a field (a scorer from before this text), rather than guess."""
    need = ("cpu_model", "arch", "cpu_vendor", "cpu_family", "cpu_model_number", "threads_per_core", "cpu_flags", "slot_cpus")
    if missing := [k for k in need if health.get(k) is None]:
        raise ValueError(f"the scorer's /health has no {missing}: it predates the timing-hardware text")
    family = f"{VENDORS.get(health['cpu_vendor'], health['cpu_vendor'])} family {health['cpu_family']} model {health['cpu_model_number']}"
    if health.get("microarchitecture"):
        family += f", {health['microarchitecture']}"
    per_core = health["threads_per_core"]
    return (
        "Timing hardware: dev_eval, the final selection and the official scoring all run on a separate scoring machine, "
        f"not in your own box. Each timed run there gets {health['slot_cpus']} dedicated CPUs ({per_core} hardware "
        f"thread{'s' if per_core != 1 else ''} per core). CPU: {health['cpu_model']} ({health['arch']}, {family}). "
        f"Instruction-set extensions: {', '.join(health['cpu_flags']) or 'none of the vector extensions'}."
    )


async def timed_dev_eval(
    state: TaskState, kind: str, source: bytes, run_dir: str, agent_id: str, n: int, seed: int, reps: int = 10,
    size: int | None = None, timeout: int = DEV_EVAL_TIMEOUT_S,
) -> tuple[dict | None, str, dict]:
    """One dev_eval run of `source`, locally in the checker or on the remote scorer: (result, report, timing).

    Local: holds CHECKER_LOCK and first kills leftover checker processes. Remote: the service's FIFO queue and a
    fresh container per job do both; the timing then also says which slot, host and CPU model ran it.
    """
    if remote(state):
        rec = await scorer_client.run_job(scorer_job(state, kind, source, agent_id, n=n, seed=seed, reps=reps, size=size))
        if rec["status"] == "done":
            return rec["result"], rec["output"], remote_timing(rec)
        return None, rec["error"], remote_timing(rec)
    args = ["--n", str(n), "--seed", str(seed), "--reps", str(reps)] + (["--size", str(size)] if size is not None else [])
    requested = time.time()
    async with CHECKER_LOCK:
        started = time.time()
        cleanup = await end_checker_processes(state)
        detail, report = await run_dev_eval(source, run_dir, args, timeout)
        ended = time.time()
    timing = {
        "queue_wait_s": round(started - requested, 3), "started_at": started, "ended_at": ended,
        "run_s": round(ended - started, 3), "cleanup": cleanup, "alone_baseline": {"mode": "child"},
        "timing_env": TIMING_ENV,
    }  # fmt: skip
    return detail, report, timing


async def end_checker_processes(state: TaskState) -> str:
    """Before a timed run: kill every checker process not running at setup (e.g. a solver's daemon or helper).

    A solver evaluated earlier can leave processes that would compete with the next timing or read
    /tests during scoring. Uses swarm.kill_agent_processes on the checker, so nothing runs unless the
    checker is positively a Docker container (DockerSandboxEnvironment and an in-container probe), and
    the PIDs snapshotted at setup (the container's init and keepalive) are spared. Returns what happened.
    """
    return await kill_agent_processes(state.metadata.get(CHECKER_PIDS), CHECKER)


async def run_dev_eval(source: bytes, run_dir: str, args: list[str], timeout: int) -> tuple[dict | None, str]:
    """Run the checker's dev_eval.py on `source` (a solver.py), copied into the fresh directory `run_dir`.

    Returns dev_eval's JSON result and its printed report, or (None, why there is no result).
    `timeout` inside the container ends dev_eval itself, and the run is shielded from cancellation,
    so an agent stopped mid-call cannot leave an evaluation running under the next one.
    """
    box = sandbox(CHECKER)
    solver, out = f"{run_dir}/solver.py", f"{run_dir}/result.json"
    await box.write_file(solver, source)
    cmd = [
        "timeout", "-k", "10", str(timeout),
        "python", f"{DEV_DIR}/dev_eval.py", solver, *args, "--json-out", out,
    ]  # fmt: skip
    with anyio.CancelScope(shield=True):
        try:
            proc = await box.exec(cmd, timeout=timeout + 60, timeout_retry=False)
        except TimeoutError:
            return None, f"timed out after {timeout}s"
    if proc.returncode == 124:  # coreutils timeout
        return None, f"timed out after {timeout}s"
    try:
        detail = json.loads(await box.read_file(out))
    except FileNotFoundError:  # the process died before writing a result (crash, OOM kill)
        return None, f"no result, exit {proc.returncode}: {proc.stderr[-500:]}"
    return detail, (proc.stderr + proc.stdout).strip()


def _first_errors(detail: dict) -> str:
    return "; ".join(e.splitlines()[-1] for e in detail["errors"][:3])


async def _dev_eval(state: TaskState, candidate: Candidate, source: bytes, index: int) -> DevResult:
    detail, report, timing = await timed_dev_eval(
        state, "final_eval", source, f"{CHECK_DIR}/final_{index}", candidate.agent_id,
        n=FINAL_DEV_N, seed=FINAL_DEV_SEED, timeout=FINAL_TIMEOUT_S,
    )  # fmt: skip
    if detail is None:
        return DevResult(candidate, False, None, error=report, detail=timing)
    return DevResult(
        candidate,
        detail["valid"],
        detail["speedup"],
        error=_first_errors(detail),
        detail={**{k: detail[k] for k in ("n_invalid", "total_solver_s", "total_reference_s", "thread_check", "first_call_ratio", "reference_inflation")},
                "alone_speedup": detail.get("alone_speedup"), **timing},  # fmt: skip
    )


class Checker:
    """The dev_eval tool for every agent of one sample: one evaluation at a time in the checker, in call order.

    Calls queue on CHECKER_LOCK, shared with every other sample in the process, so the queue wait
    includes other samples' dev_eval calls, finalize evaluations and final scoring.
    Every call is recorded in state.metadata["checker"]["calls"] (who, when, queue wait, run time, result).
    An agent's repeat of a completed call (same file contents, n, seed, reps and size) returns the recorded result
    without a new evaluation (v6 pilot: 56% of dev_eval calls re-timed an unchanged file); it is recorded with
    `cached` True, `same_as` (the index of the call it repeats), zero queue wait and zero run time.
    """

    def __init__(self, state: TaskState, seed_cap: int = DEV_SEED_CAP) -> None:
        self.state = state
        self.seed_cap = seed_cap
        self.calls: list[dict[str, Any]] = []
        self.done: dict[tuple, tuple[int, str]] = {}  # (agent, sha256, n, seed, reps, size) -> (call index, report)
        state.metadata["checker"] = {"calls": self.calls, "seed_cap": seed_cap}

    def tool(self, agent_id: str) -> Tool:
        async def execute(path: str, n: int = 20, seed: int = 0, reps: int = 10, size: int | None = None) -> str:
            """Check a solver for correctness and measure its speedup over the reference, the way the final evaluation does.

            Runs /app/dev's dev_eval on a separate, dedicated machine that holds the toolkit and a copy of this
            one file (as solver.py). Evaluations run one at a time, in the order they are requested by any
            agent working on this task; the result says how long this call waited in the queue.

            Args:
                path: Absolute path of the solver file in your container. Only this file is copied, so it cannot import sibling files.
                n: Number of dev instances.
                seed: First dev seed offset (>= 0, and seed + n below 400,000). Dev instances never use the final evaluation's seeds.
                reps: Timed repeats per instance.
                size: Problem size passed to generate_problem. Default: the final evaluation's.

            Repeating a call with an unchanged file and the same n, seed, reps and size returns the earlier result
            without a new evaluation.
            """
            if not path.startswith("/"):
                raise ToolError(f"path must be absolute, got {path!r}")
            if seed < 0:
                raise ToolError("seed must be >= 0")
            if seed >= self.seed_cap or seed + n >= self.seed_cap:
                raise ToolError(f"seed + n must be below {self.seed_cap:,} (got seed {seed:,}, n {n}); use a smaller seed")
            try:
                source = await sandbox(agent_box(self.state, agent_id)).read_file(path, text=False)
            except (FileNotFoundError, IsADirectoryError):
                raise ToolError(f"no such file: {path}")
            key = (agent_id, hashlib.sha256(source).hexdigest(), n, seed, reps, size)
            if key in self.done:
                i, report = self.done[key]
                same = {k: self.calls[i][k] for k in ("valid", "speedup", "error", "first_call_ratio", "reference_inflation")}
                self.calls.append({"agent_id": agent_id, "path": path, "started_at": time.time(), "queue_wait_s": 0.0,
                                   "run_s": 0.0, **same, "n": n, "seed": seed, "cached": True, "same_as": i,
                                   "sha256": key[1]})  # fmt: skip
                return (
                    "[dev_eval] same file contents and arguments as an earlier call of yours: this is its recorded "
                    f"result; no new evaluation was run.\n{report}"
                )
            requested = time.time()
            run_dir = f"{CHECK_DIR}/{agent_id}-{time.time_ns()}"
            detail, report, timing = await timed_dev_eval(
                self.state, "dev_eval", source, run_dir, agent_id, n=n, seed=seed, reps=reps, size=size
            )
            self.calls.append({
                "agent_id": agent_id,
                "path": path,
                **timing,
                "started_at": requested,  # when the call was made, before queueing
                "valid": None if detail is None else detail["valid"],
                "speedup": None if detail is None else detail["speedup"],
                "error": report if detail is None else _first_errors(detail),
                "thread_check": None if detail is None else detail["thread_check"],
                "first_call_ratio": None if detail is None else detail["first_call_ratio"],
                "reference_inflation": None if detail is None else detail["reference_inflation"],
                "n": n,
                "seed": seed,
                "cached": False,
                "sha256": key[1],
            })  # fmt: skip
            if detail is not None:  # a completed evaluation; errors (e.g. a timeout) may be retried
                self.done[key] = (len(self.calls) - 1, report)
            head = f"[dev_eval] waited {timing['queue_wait_s']:.1f}s in the queue, ran {timing['run_s']:.1f}s."
            return f"{head}\n{report}"

        execute.__doc__ = execute.__doc__.replace("400,000", f"{self.seed_cap:,}")  # the agent sees its own cap
        return ToolDef(execute, name="dev_eval").as_tool()


def queue_telemetry(calls: list[dict], agents: dict[str, dict], time_limit: float | None) -> dict[str, dict]:
    """Per agent: dev_eval calls, total and max queue wait, wait as a fraction of its active (wall) time, time limit.

    `calls` is state.metadata["checker"]["calls"], `agents` state.metadata["swarm"]["agents"].
    """
    out = {}
    for agent_id, rec in agents.items():
        waits = [c["queue_wait_s"] for c in calls if c["agent_id"] == agent_id and c.get("queue_wait_s") is not None]
        active = rec.get("wall_s")
        out[agent_id] = {
            "dev_eval_calls": len(waits),
            "queue_wait_total_s": round(sum(waits), 3),
            "queue_wait_max_s": max(waits, default=0.0),
            "active_s": active,
            "queue_wait_frac": round(sum(waits) / active, 4) if active else None,
            "time_limit_s": time_limit,
        }
    return out


def known_dev_score(state: TaskState) -> Callable[[bytes], str | None]:
    """swarm(known_score=...): a file's latest completed dev_eval by any agent of this sample, as a publish notice
    quotes it ("dev_eval 3.21x on n=20, seed 0"), or None if that exact file was never evaluated."""

    def lookup(source: bytes) -> str | None:
        digest = hashlib.sha256(source).hexdigest()
        done = [c for c in state.metadata.get("checker", {}).get("calls", [])
                if c.get("sha256") == digest and not c["cached"] and c["valid"] is not None]  # fmt: skip
        if not done:
            return None
        c = done[-1]
        result = "no valid instance timed" if c["speedup"] is None else f"{c['speedup']:.2f}x"
        return f"dev_eval {result}{'' if c['valid'] else ', invalid'} on n={c['n']}, seed {c['seed']}"

    return lookup


def algotune_agent_tools(state: TaskState, seed_cap: int = DEV_SEED_CAP) -> Callable[[str], list[Tool]]:
    """swarm(agent_tools=...): one Checker (queue) per sample, and its dev_eval tool for each agent."""
    checker = Checker(state, seed_cap)
    return lambda agent_id: [checker.tool(agent_id)]


async def algotune_finalize(state: TaskState, candidates: list[Candidate]) -> None:
    """Evaluate every candidate, copy the fastest correct solver to /app/solver.py, record all results.

    Evaluations run in the checker container after every agent has stopped, so no agent work shares
    their CPUs, and each holds CHECKER_LOCK, so no other sample's timed run overlaps it.
    Reinstalls the toolkit first so the check is the packaged one.
    Candidates with identical solver.py text share one evaluation. Any /app/solver.py an agent wrote is
    removed first (agents are told it has no effect), so nothing is installed when no candidate is correct.
    Each agent's dev_eval queue waits are summarised first (`queue_telemetry`, state.metadata["checker"]["agents"]).
    Every candidate's solver.py is kept in the log, once per distinct text: each candidate row carries its `sha256`
    and finalize's `sources` maps it to the text (scripts/select_portfolio.py selects across runs from these).
    """
    sw = state.metadata.get("swarm", {})
    checker = state.metadata.setdefault("checker", {"calls": []})
    checker["agents"] = queue_telemetry(checker.get("calls", []), sw.get("agents", {}), sw.get("time_limit"))
    await _install_toolkit(state)
    box = sandbox()
    removed = await box.exec(["rm", "-f", SOLVER_PATH])
    if not removed.success:
        raise RuntimeError(f"could not clear {SOLVER_PATH}: {removed.stderr}")
    cache: dict[str, DevResult] = {}
    sources: dict[str, str] = {}
    results, digests = [], []
    for i, c in enumerate(candidates):
        try:
            source = await box.read_file(f"{c.path}/solver.py", text=False)
        except FileNotFoundError:
            results.append(DevResult(c, False, None, error="no solver.py"))
            digests.append(None)
            continue
        digest = hashlib.sha256(source).hexdigest()
        if digest not in cache:
            cache[digest] = await _dev_eval(state, c, source, i)
            sources[digest] = source.decode(errors="replace")
        results.append(replace(cache[digest], candidate=c))
        digests.append(digest)
    best = pick_best(results)
    selected_source = None
    if best is not None:
        copied = await box.exec(["cp", f"{best.candidate.path}/solver.py", SOLVER_PATH])
        if not copied.success:
            raise RuntimeError(f"could not install the selected solver: {copied.stderr}")
        # kept in the log so every final solution can be re-scored later in a clean serial timing pass
        selected_source = await box.read_file(SOLVER_PATH)

    state.metadata["finalize"] = {
        "dev_n": FINAL_DEV_N,
        "dev_seed": FINAL_DEV_SEED,
        "selected": None if best is None else {"agent_id": best.candidate.agent_id, "path": best.candidate.path},
        "selected_source": selected_source,
        "sources": sources,
        "candidates": [
            {
                "agent_id": r.candidate.agent_id,
                "model": r.candidate.model,
                "kind": r.candidate.kind,
                "path": r.candidate.path,
                "published_at": r.candidate.published_at,
                "sha256": digest,
                "valid": r.valid,
                "speedup": r.speedup,
                "error": r.error,
                **r.detail,
            }
            for r, digest in zip(results, digests)
        ],
    }
