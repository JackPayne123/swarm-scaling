"""Live smoke test for parallel samples (one AlgoTune sample, 4 epochs at once, scripted mockllm agents, no model spend).

Run from the worktree, with nothing else using Docker:
    cd ~/projects/swarm-scaling-pilot2 && uv run python scripts/smoke_parallel.py

Settings match `runner --parallel 4`: max_samples = max_sandboxes = 4, agent boxes at box_resources' memory for 4.
Each agent writes a solver wrapping the reference (it prints its CPU affinity and hostname when imported),
runs the reference on dev problems in its own box, and calls dev_eval. Checks from the log: the four samples
ran concurrently; no two timed checker runs (dev_eval calls, finalize evaluations, final scoring) overlapped;
final scoring ran in the checker (the verifier's output shows the checker's hostname and CPUs 8-15) and
produced scores; the log reads back.
"""

import sys

from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.util import sandbox

from swarm_scaling.algotune_devkit import SOLVER_PATH, algotune_agent_tools, algotune_finalize
from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import algotune_task

PARALLEL = 4
SOLVER = '''import importlib.util, os, socket
print("SOLVER_IMPORTED affinity=%s host=%s" % (sorted(os.sched_getaffinity(0)), socket.gethostname()), flush=True)
_spec = importlib.util.spec_from_file_location("ref", "/app/dev/reference_task.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


class Solver:
    def __init__(self):
        self.task = _ref.Task()

    def solve(self, problem):
        return self.task.solve(problem)
'''
# What an agent does in its own box: generate dev problems at the task's size and time the reference.
LOCAL_WORK = '''
import importlib.util, time
spec = importlib.util.spec_from_file_location("ref", "/app/dev/reference_task.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
t = ref.Task()
problems = [t.generate_problem(n=9, random_seed=10_000 + i) for i in range(20)]
start = time.perf_counter()
for p in problems:
    t.solve(p)
print(f"local reference run: {time.perf_counter() - start:.2f}s for 20 problems")
'''
PROBE = "hostname; cat /sys/fs/cgroup/cpuset.cpus.effective; cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/memory.peak"


def scripted(*calls: tuple[str, dict]):
    """One model for all epochs: each conversation's next step is picked by how many turns it already has."""

    def output(messages, tools, tool_choice, config) -> ModelOutput:
        tool, args = calls[sum(m.role == "assistant" for m in messages)]
        return ModelOutput.for_tool_call("mockllm/model", tool, args)

    return get_model("mockllm/model", custom_outputs=output)


async def finalize(state, candidates):
    for name in ("default", "checker"):
        probe = await sandbox(name).exec(["sh", "-c", PROBE])
        state.metadata[f"smoke_probe_{name}"] = probe.stdout.split()
    await algotune_finalize(state, candidates)


def timed_runs(sample) -> list[tuple[float, float, str]]:
    """Every timed checker run in one sample's log: (start, end, label)."""
    tag = f"epoch{sample.epoch}"
    runs = []
    for c in sample.metadata["checker"]["calls"]:
        start = c["started_at"] + c["queue_wait_s"]
        runs.append((start, start + c["run_s"], f"{tag} dev_eval (waited {c['queue_wait_s']:.2f}s)"))
    for c in sample.metadata["finalize"]["candidates"]:
        if "started_at" in c:
            runs.append((c["started_at"], c["ended_at"], f"{tag} finalize {c['agent_id']} (waited {c['queue_wait_s']:.2f}s)"))
    for score in sample.scores.values():
        t = score.metadata["checker"]
        runs.append((t["started_at"], t["ended_at"], f"{tag} scoring (waited {t['queue_wait_s']:.2f}s)"))
    return sorted(set(runs))  # candidates with identical source share one evaluation


def report(location: str) -> None:
    samples = read_eval_log(location).samples  # also proves the log reads back
    print(f"samples read back: {len(samples)}")
    spans = []
    for s in samples:
        sw = s.metadata["swarm"]
        spans.append((sw["started_at"], sw["ended_at"]))
        score = {k: v.value for k, v in s.scores.items()}
        default, checker = s.metadata["smoke_probe_default"], s.metadata["smoke_probe_checker"]
        print(f"epoch {s.epoch}: agents {sw['started_at']:.1f}-{sw['ended_at']:.1f} score={score} cleanup={sw['process_cleanup']!r}")
        for name, (host, cpus, mem_max, mem_peak) in (("agent box", default), ("checker", checker)):
            print(f"  {name}: host {host}, cpus {cpus}, memory.max {int(mem_max) >> 20} MiB, memory.peak before finalize {int(mem_peak) >> 20} MiB")
        explanation = next(iter(s.scores.values())).explanation or ""
        seen = [line.split("SOLVER_IMPORTED ")[1] for line in explanation.splitlines() if "SOLVER_IMPORTED affinity=[" in line]  # not the cat of the source
        print(f"  verifier imported the solver with: {seen}; in the checker: {bool(seen) and seen[0].endswith(checker[0])}")
    latest_start = max(a for a, _ in spans)
    print("agent phases running at the latest start:", sum(a <= latest_start <= b for a, b in spans))

    runs = sorted(r for s in samples for r in timed_runs(s))
    t0 = runs[0][0]
    for start, end, label in runs:
        print(f"  {start - t0:8.2f} - {end - t0:8.2f}  {label}")
    # dev_eval times are stored rounded to the millisecond, so allow 2 ms
    overlaps = [(a, b) for a, b in zip(runs, runs[1:]) if a[1] > b[0] + 0.002]
    print(f"timed checker runs: {len(runs)}, overlapping pairs: {len(overlaps)}", overlaps)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # report on an existing log
        report(sys.argv[1])
        sys.exit()
    path = "/app/agents/agent_0/solver.py"
    write = ("python", {"code": f"open({path!r}, 'w').write({SOLVER!r})"})
    agents = [
        scripted(write, ("python", {"code": LOCAL_WORK}), ("dev_eval", {"path": path, "n": 3}), ("submit", {"answer": "done"}))
    ]
    (log,) = eval(
        algotune_task(split="pilot", parallel=PARALLEL),
        solver=swarm(models=agents, per_agent_tokens=200_000, finalize=finalize, agent_tools=algotune_agent_tools,
                     final_path=SOLVER_PATH, budget_warnings=()),
        model="mockllm/model",
        sample_id="algotune/cvar-projection",
        epochs=PARALLEL,
        max_samples=PARALLEL,
        max_sandboxes=PARALLEL,
        log_dir="logs/smoke-parallel",
        display="none",
    )
    print("status:", log.status, log.error or "")
    report(log.location)
