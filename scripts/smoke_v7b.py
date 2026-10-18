"""v7b smoke test: swarm_scaling.runner with every mockllm/model agent scripted (no model spend).

    uv run python scripts/smoke_v7b.py <runner args, with --models mockllm/model --budget-type all --budget N --name X>
    uv run python scripts/smoke_v7b.py report logs/<name> [...]

Each scripted agent probes its own box (CPU model, nproc, os.cpu_count(), affinity, BLAS and numba threads, cgroup
memory limit, AVX-512), writes a solver that wraps the reference, dev_evals and publishes it, then answers with text
until its token budget is used up (v7 rows run to budget). The agent's first prompt and its probe output are saved as
logs/<name>/prompt-<agent>.txt and probe-<agent>.txt, so they travel with the run's logs from a cloud VM
(run_sample.sh --script scripts/smoke_v7b.py, or v6_drive.py run --script). `report` prints them per agent beside
the run's scorer_hardware (run.json) and whether every prompt holds that paragraph.
"""

import json
import sys
from pathlib import Path

from inspect_ai.model import ModelOutput, get_model

from swarm_scaling import runner

PROBE = r"""lscpu | sed -n 's/^Model name: *//p'; echo "nproc $(nproc)"; echo "avx512f $(grep -c -w avx512f /proc/cpuinfo)"
python - <<'EOF'
import json, os
import numba, numpy, scipy.linalg, threadpoolctl
print("PROBE " + json.dumps({
    "cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
    "blas": [(d["internal_api"], d["filepath"].rsplit("/", 1)[-1], d["num_threads"]) for d in threadpoolctl.threadpool_info()],
    "numba_threads": numba.config.NUMBA_NUM_THREADS,
    "memory_max": open("/sys/fs/cgroup/memory.max").read().strip(),
    "cpus_max": open("/sys/fs/cgroup/cpu.max").read().strip(),
}))
EOF"""
SOLVER = '''import importlib.util

_spec = importlib.util.spec_from_file_location("ref", "/app/dev/reference_task.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


class Solver:
    def __init__(self):
        self.task = _ref.Task()

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
'''


def text_of(message) -> str:
    content = message.content
    return content if isinstance(content, str) else "\n".join(getattr(c, "text", "") or "" for c in content)


def scripted(agent: str, log_dir: Path):
    """An agent that probes its box, then writes, dev_evals and publishes the reference wrapper, then only talks."""
    path = f"/app/agents/{agent}/solver.py"
    steps = [
        ("bash", {"command": PROBE}),
        ("python", {"code": f"open({path!r}, 'w').write({SOLVER!r})"}),
        ("dev_eval", {"path": path, "n": 3, "reps": 3}),
        ("publish_candidate", {"path": path, "note": "reference wrapper (smoke test)"}),
    ]

    def output(messages, tools, tool_choice, config) -> ModelOutput:
        turn = sum(m.role == "assistant" for m in messages)
        if turn == 0:
            (log_dir / f"prompt-{agent}.txt").write_text("\n\n".join(text_of(m) for m in messages if m.role == "user"))
        if turn == 1:
            (log_dir / f"probe-{agent}.txt").write_text(text_of(messages[-1]))
        if turn < len(steps):
            return ModelOutput.for_tool_call("mockllm/model", *steps[turn])
        return ModelOutput.from_content("mockllm/model", "Smoke agent: nothing further to try.")

    return get_model("mockllm/model", custom_outputs=output)


def run(argv: list[str]) -> None:
    log_dir = runner.LOG_DIR / argv[argv.index("--name") + 1]
    log_dir.mkdir(parents=True, exist_ok=True)
    real_swarm = runner.swarm

    def swarm(models, **kwargs):
        assert all(m == "mockllm/model" for m in models), "the smoke test scripts mockllm/model agents only"
        return real_swarm(models=[scripted(f"agent_{i}", log_dir) for i in range(len(models))], **kwargs)

    runner.swarm = swarm
    sys.argv = ["runner", *argv]
    runner.main()


def report(dirs: list[str]) -> None:
    for d in map(Path, dirs):
        hardware = json.loads((d / "run.json").read_text()).get("scorer_hardware")
        print(f"{d.name}: scorer_hardware = {hardware!r}")
        for prompt in sorted(d.glob("prompt-*.txt")):
            agent = prompt.stem.split("-", 1)[1]
            probe = (d / f"probe-{agent}.txt").read_text() if (d / f"probe-{agent}.txt").exists() else ""
            print(f"  {agent}: hardware paragraph in prompt: {bool(hardware) and hardware in prompt.read_text()}")
            for line in probe.splitlines():
                if line.strip() and not line.startswith("[budget]"):
                    print(f"    {line.strip()[:300]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(sys.argv[2:])
    else:
        run(sys.argv[1:])
