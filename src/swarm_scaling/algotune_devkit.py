"""AlgoTune dev toolkit and final selection (HARNESS.md, AlgoTune section).

`algotune_setup()` runs before any agent and installs /app/dev in the sample sandbox.
`algotune_finalize()` runs after the agents, evaluates every candidate solver on fixed dev
instances and installs the fastest correct one as /app/solver.py.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from swarm_scaling.swarm import Candidate

ASSETS = Path(__file__).parent / "algotune_assets"
DEV_DIR = "/app/dev"
SOLVER_PATH = "/app/solver.py"

# Selection runs on its own dev instances: dev_eval's default seeds start at 0, these start far away.
FINAL_DEV_SEED = 50_000
FINAL_DEV_N = 20
FINAL_TIMEOUT_S = 900  # per candidate

DEV_TOOLKIT_NOTE = (
    "A dev toolkit is installed in /app/dev (read /app/dev/README.md first). It contains the full "
    "reference Task class with generate_problem, and `python /app/dev/dev_eval.py <solver.py>`, which "
    "checks your solver for correctness and measures its speedup the way the final evaluation does."
)


async def _install_toolkit(state: TaskState) -> None:
    # Only evaluator.py is read from the verifier files: no test_outputs.py, no test instances.
    evaluator = (Path(state.metadata["tests_dir"]) / "evaluator.py").read_text()
    problem_size = state.metadata["harbor_config"]["metadata"]["algotune_problem_size"]

    box = sandbox()
    await box.write_file(f"{DEV_DIR}/reference_task.py", evaluator)
    await box.write_file(f"{DEV_DIR}/dev_eval.py", (ASSETS / "dev_eval.py").read_text())
    await box.write_file(f"{DEV_DIR}/README.md", (ASSETS / "README.md").read_text())
    await box.write_file(f"{DEV_DIR}/config.json", json.dumps({"problem_size": problem_size}))


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


async def _dev_eval(candidate: Candidate, index: int) -> DevResult:
    box = sandbox()
    out = f"/tmp/dev_eval_{index}.json"
    cmd = [
        "python", f"{DEV_DIR}/dev_eval.py", f"{candidate.path}/solver.py",
        "--n", str(FINAL_DEV_N), "--seed", str(FINAL_DEV_SEED), "--json-out", out,
    ]  # fmt: skip
    try:
        proc = await box.exec(cmd, timeout=FINAL_TIMEOUT_S, timeout_retry=False)
    except TimeoutError:
        return DevResult(candidate, False, None, error=f"timed out after {FINAL_TIMEOUT_S}s")
    try:
        detail = json.loads(await box.read_file(out))
    except FileNotFoundError:  # the process died before writing a result (crash, OOM kill)
        return DevResult(candidate, False, None, error=f"no result, exit {proc.returncode}: {proc.stderr[-500:]}")
    return DevResult(
        candidate,
        detail["valid"],
        detail["speedup"],
        error="; ".join(e.splitlines()[-1] for e in detail["errors"][:3]),
        detail={k: detail[k] for k in ("n_invalid", "total_solver_s", "total_reference_s")},
    )


async def algotune_finalize(state: TaskState, candidates: list[Candidate]) -> None:
    """Evaluate every candidate, copy the fastest correct solver to /app/solver.py, record all results.

    Run it after every agent process has been stopped: timings are meaningless under load.
    Reinstalls the toolkit first so the check is the packaged one, not an agent-edited copy.
    Candidates with identical solver.py text share one evaluation. Copies nothing when no candidate is correct.
    """
    await _install_toolkit(state)
    box = sandbox()
    cache: dict[str, DevResult] = {}
    results = []
    for i, c in enumerate(candidates):
        try:
            digest = hashlib.sha256(await box.read_file(f"{c.path}/solver.py", text=False)).hexdigest()
        except FileNotFoundError:
            results.append(DevResult(c, False, None, error="no solver.py"))
            continue
        if digest not in cache:
            cache[digest] = await _dev_eval(c, i)
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
