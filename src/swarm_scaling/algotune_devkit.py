"""AlgoTune dev toolkit and final selection (HARNESS.md, AlgoTune section).

`algotune_setup()` runs before any agent and installs /app/dev in the agent box and the checker.
`algotune_agent_tools()` gives every agent the dev_eval tool, which runs dev_eval.py in the
checker container (tasks.py), one evaluation at a time per sample.
`algotune_finalize()` runs after the agents, evaluates every candidate solver on fixed dev
instances in the checker and installs the fastest correct one as /app/solver.py in the agent box.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import anyio
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError
from inspect_ai.util import sandbox

from swarm_scaling.swarm import Candidate

ASSETS = Path(__file__).parent / "algotune_assets"
DEV_DIR = "/app/dev"
SOLVER_PATH = "/app/solver.py"

# Compose service (tasks.py) where every evaluation runs. Agents' bash and python tools reach only the default box.
CHECKER = "checker"
CHECK_DIR = "/app/checks"  # in the checker: a fresh directory per evaluation
DEV_EVAL_TIMEOUT_S = 900  # per dev_eval tool call, run time only (not the queue wait)

# Selection runs on its own dev instances: dev_eval's default seeds start at 0, these start far away.
FINAL_DEV_SEED = 50_000
FINAL_DEV_N = 20
FINAL_TIMEOUT_S = 900  # per candidate

DEV_TOOLKIT_NOTE = (
    "A dev toolkit is installed in /app/dev (read /app/dev/README.md first). It contains the full "
    "reference Task class with generate_problem. The dev_eval tool checks a solver for correctness and "
    "measures its speedup the way the final evaluation does. It runs on a separate, dedicated machine, one "
    "evaluation at a time: calls from every agent working on this task wait in one queue, and each result "
    "says how long it waited. Timings you take in your own container are affected by whatever else is "
    "running there."
)


async def _install_toolkit(state: TaskState) -> None:
    # Only evaluator.py is read from the verifier files: no test_outputs.py, no test instances.
    evaluator = (Path(state.metadata["tests_dir"]) / "evaluator.py").read_text()
    problem_size = state.metadata["harbor_config"]["metadata"]["algotune_problem_size"]

    box = sandbox()
    await box.write_file(f"{DEV_DIR}/reference_task.py", evaluator)
    await box.write_file(f"{DEV_DIR}/README.md", (ASSETS / "README.md").read_text())

    checker = sandbox(CHECKER)
    await checker.write_file(f"{DEV_DIR}/reference_task.py", evaluator)
    await checker.write_file(f"{DEV_DIR}/dev_eval.py", (ASSETS / "dev_eval.py").read_text())
    await checker.write_file(f"{DEV_DIR}/config.json", json.dumps({"problem_size": problem_size}))


@solver
def algotune_setup() -> Solver:
    """Install the dev toolkit in /app/dev and point the task prompt at it."""

    async def setup(state: TaskState, generate: Generate) -> TaskState:
        await _install_toolkit(state)
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


async def _dev_eval(candidate: Candidate, source: bytes, index: int) -> DevResult:
    args = ["--n", str(FINAL_DEV_N), "--seed", str(FINAL_DEV_SEED)]
    detail, report = await run_dev_eval(source, f"{CHECK_DIR}/final_{index}", args, FINAL_TIMEOUT_S)
    if detail is None:
        return DevResult(candidate, False, None, error=report)
    return DevResult(
        candidate,
        detail["valid"],
        detail["speedup"],
        error=_first_errors(detail),
        detail={k: detail[k] for k in ("n_invalid", "total_solver_s", "total_reference_s")},
    )


class Checker:
    """The dev_eval tool for every agent of one sample: one evaluation at a time in the checker, in call order.

    Every call is recorded in state.metadata["checker"]["calls"] (who, when, queue wait, run time, result).
    """

    def __init__(self, state: TaskState) -> None:
        self.lock = anyio.Lock()  # waiters are served in arrival order (FIFO)
        self.calls: list[dict[str, Any]] = []
        state.metadata["checker"] = {"calls": self.calls}

    def tool(self, agent_id: str) -> Tool:
        async def execute(path: str, n: int = 20, seed: int = 0, reps: int = 10, size: int | None = None) -> str:
            """Check a solver for correctness and measure its speedup over the reference, the way the final evaluation does.

            Runs /app/dev's dev_eval on a separate, dedicated machine that holds the toolkit and a copy of this
            one file (as solver.py). Evaluations run one at a time, in the order they are requested by any
            agent working on this task; the result says how long this call waited in the queue.

            Args:
                path: Absolute path of the solver file in your container. Only this file is copied, so it cannot import sibling files.
                n: Number of dev instances.
                seed: First dev seed offset (>= 0). Dev instances never use the final evaluation's seeds.
                reps: Timed repeats per instance.
                size: Problem size passed to generate_problem. Default: the final evaluation's.
            """
            if not path.startswith("/"):
                raise ToolError(f"path must be absolute, got {path!r}")
            if seed < 0:
                raise ToolError("seed must be >= 0")
            try:
                source = await sandbox().read_file(path, text=False)
            except (FileNotFoundError, IsADirectoryError):
                raise ToolError(f"no such file: {path}")
            args = ["--n", str(n), "--seed", str(seed), "--reps", str(reps)]
            if size is not None:
                args += ["--size", str(size)]
            requested = time.time()
            async with self.lock:
                started = time.time()
                run_dir = f"{CHECK_DIR}/{agent_id}-{time.time_ns()}"
                detail, report = await run_dev_eval(source, run_dir, args, DEV_EVAL_TIMEOUT_S)
                ended = time.time()
                self.calls.append({
                    "agent_id": agent_id,
                    "path": path,
                    "started_at": requested,  # when the call was made, before queueing
                    "queue_wait_s": round(started - requested, 3),
                    "run_s": round(ended - started, 3),
                    "valid": None if detail is None else detail["valid"],
                    "speedup": None if detail is None else detail["speedup"],
                    "error": report if detail is None else _first_errors(detail),
                    "n": n,
                    "seed": seed,
                })  # fmt: skip
            head = f"[dev_eval] waited {started - requested:.1f}s in the queue, ran {ended - started:.1f}s."
            return f"{head}\n{report}"

        return ToolDef(execute, name="dev_eval").as_tool()


def algotune_agent_tools(state: TaskState) -> Callable[[str], list[Tool]]:
    """swarm(agent_tools=...): one Checker (queue) per sample, and its dev_eval tool for each agent."""
    checker = Checker(state)
    return lambda agent_id: [checker.tool(agent_id)]


async def algotune_finalize(state: TaskState, candidates: list[Candidate]) -> None:
    """Evaluate every candidate, copy the fastest correct solver to /app/solver.py, record all results.

    Evaluations run in the checker container after every agent has stopped, so no agent work shares
    their CPUs. Reinstalls the toolkit first so the check is the packaged one.
    Candidates with identical solver.py text share one evaluation. Any /app/solver.py an agent wrote is
    removed first (agents are told it has no effect), so nothing is installed when no candidate is correct.
    """
    await _install_toolkit(state)
    box = sandbox()
    removed = await box.exec(["rm", "-f", SOLVER_PATH])
    if not removed.success:
        raise RuntimeError(f"could not clear {SOLVER_PATH}: {removed.stderr}")
    cache: dict[str, DevResult] = {}
    results = []
    for i, c in enumerate(candidates):
        try:
            source = await box.read_file(f"{c.path}/solver.py", text=False)
        except FileNotFoundError:
            results.append(DevResult(c, False, None, error="no solver.py"))
            continue
        digest = hashlib.sha256(source).hexdigest()
        if digest not in cache:
            cache[digest] = await _dev_eval(c, source, i)
        results.append(replace(cache[digest], candidate=c))
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
        "candidates": [
            {
                "agent_id": r.candidate.agent_id,
                "model": r.candidate.model,
                "kind": r.candidate.kind,
                "path": r.candidate.path,
                "published_at": r.candidate.published_at,
                "valid": r.valid,
                "speedup": r.speedup,
                "error": r.error,
                **r.detail,
            }
            for r in results
        ],
    }
