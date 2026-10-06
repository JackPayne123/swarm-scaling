"""Timing-noise gate (PLAN.md Budget): time the reference solver against itself with dev_eval.

Installs the reference as /app/solver.py via inspect_harbor's oracle, then runs
`dev_eval.py /app/solver.py` REPEATS times inside the container. A clean machine should give
speedups close to 1.0 with a small spread. Run with nothing else busy on the machine.

Usage: uv run python scripts/timing_noise.py [sample_id ...]
"""

import re
import sys

from inspect_ai import eval
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.util import sandbox
from inspect_harbor import oracle

from swarm_scaling.tasks import algotune_task

REPEATS = 5


@solver
def time_reference_against_itself() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        speedups = []
        for _ in range(REPEATS):
            result = await sandbox().exec(["python", "/app/dev/dev_eval.py", "/app/solver.py"], timeout=1800)
            match = re.search(r"speedup[^0-9]*([0-9.]+)x", result.stdout)
            speedups.append(float(match.group(1)) if match else None)
        state.metadata["timing_noise"] = speedups
        return state

    return solve


if __name__ == "__main__":
    sample_ids = sys.argv[1:] or ["algotune/cvar-projection", "algotune/job-shop-scheduling"]
    for sample_id in sample_ids:
        (log,) = eval(
            algotune_task(split="pilot"),
            solver=chain(oracle(), time_reference_against_itself()),
            model="mockllm/model",
            sample_id=sample_id,
            score=False,
            max_samples=1,
            max_sandboxes=1,
            display="none",
        )
        sample = log.samples[0]
        values = sample.metadata.get("timing_noise", [])
        ok = [v for v in values if v is not None]
        spread = (max(ok) - min(ok)) / min(ok) * 100 if ok else float("nan")
        print(f"{sample_id}: {values}  range {spread:.2f}%  status {log.status}")
