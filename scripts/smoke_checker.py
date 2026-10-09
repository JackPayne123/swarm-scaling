"""Live smoke test for the checker container (one AlgoTune sample, two scripted mockllm agents, no model spend).

Run from the worktree, with nothing else using Docker:
    cd ~/projects/swarm-scaling-pilot2 && uv run python <this file>

Remote scorer (scorer_service.py running, SCORER_URL and SCORER_TOKEN set):
    uv run python scripts/smoke_checker.py --checker remote

Checks: both containers start with their cpusets; the agent box has no dev_eval.py; both agents' dev_eval
calls run in the checker (if they overlap, the later one reports a queue wait); telemetry lands in metadata; finalize runs in the
checker and installs /app/solver.py; the sample scores; the log reads back.
"""

import os
import sys

from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.util import sandbox

from swarm_scaling.algotune_devkit import SOLVER_PATH, algotune_agent_tools, algotune_finalize
from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import algotune_task

SOLVER = '''import importlib.util
_spec = importlib.util.spec_from_file_location("ref", "/app/dev/reference_task.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


class Solver:
    def __init__(self):
        self.task = _ref.Task()

    def solve(self, problem):
        return self.task.solve(problem)
'''
PROBE = "nproc; cat /sys/fs/cgroup/cpuset.cpus.effective; ls /app/dev"


def scripted(*calls: tuple[str, dict]):
    return get_model("mockllm/model", custom_outputs=[ModelOutput.for_tool_call("mockllm/model", t, a) for t, a in calls])


async def finalize(state, candidates):
    if state.metadata.get("checker_backend") != "remote":  # remote: no checker container to probe
        probe = await sandbox("checker").exec(["sh", "-c", PROBE])
        state.metadata["smoke_checker_probe"] = probe.stdout
    await algotune_finalize(state, candidates)


if __name__ == "__main__":
    # --checker remote: dev_eval, finalize and scoring go to the scorer service (SCORER_URL, SCORER_TOKEN)
    checker = sys.argv[sys.argv.index("--checker") + 1] if "--checker" in sys.argv else "local"
    os.environ.setdefault("SWARM_RUN_ID", f"smoke-checker-{checker}")
    a0 = "/app/agents/agent_0/solver.py"
    a1 = "/app/agents/agent_1/solver.py"
    write = lambda path: ("python", {"code": f"open({path!r}, 'w').write({SOLVER!r})"})  # noqa: E731
    agents = [
        scripted(write(a0), ("dev_eval", {"path": a0, "n": 3}), ("submit", {"answer": "done"})),
        scripted(write(a1), ("bash", {"command": PROBE}), ("dev_eval", {"path": a1, "n": 3}), ("submit", {"answer": "done"})),
    ]
    (log,) = eval(
        algotune_task(split="pilot", n_agents=2, checker_backend=checker),
        solver=swarm(models=agents, per_agent_tokens=200_000, finalize=finalize, agent_tools=algotune_agent_tools,
                     final_path=SOLVER_PATH, budget_warnings=()),
        model="mockllm/model",
        sample_id="algotune/cvar-projection",
        max_samples=1,
        max_sandboxes=1,
        display="none",
    )
    print("status:", log.status, log.error or "")
    s = read_eval_log(log.location).samples[0]  # also proves the log reads back
    for e in s.events:
        if e.event == "tool" and e.function in ("bash", "dev_eval"):
            print(f"--- {e.function}:\n{str(e.result)[:800]}")
    print("checker probe:", s.metadata.get("smoke_checker_probe"))
    print("checker calls:", s.metadata.get("checker"))
    fin = s.metadata.get("finalize", {})
    print("finalize selected:", fin.get("selected"), [(c["agent_id"], c["valid"], c["speedup"], c["error"]) for c in fin.get("candidates", [])])
    print("score:", {k: v.value for k, v in (s.scores or {}).items()})
    print("score metadata:", {k: v.metadata for k, v in (s.scores or {}).items()})
