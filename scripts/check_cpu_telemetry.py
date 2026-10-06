"""Check the swarm CPU telemetry in a real AlgoTune container (mockllm, no API calls).

One scripted agent saturates the container's 8 CPUs for ~40 s; the summary must flag it.
Usage: uv run python scripts/check_cpu_telemetry.py
"""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import algotune_task

BURN = "for i in 1 2 3 4 5 6 7 8; do timeout 40 sh -c 'while :; do :; done' & done; wait; echo burned"


def scripted(*calls: tuple[str, dict]):
    return get_model("mockllm/model", custom_outputs=[ModelOutput.for_tool_call("mockllm/model", t, a) for t, a in calls])


if __name__ == "__main__":
    burner = scripted(("bash", {"command": BURN}), ("submit", {"answer": "done"}))
    quiet = scripted(("submit", {"answer": "done"}))
    (log,) = eval(
        algotune_task(split="pilot"),
        solver=swarm(models=[burner, quiet], per_agent_tokens=6000, cpu_sample_interval=5),
        model="mockllm/model",
        sample_id="algotune/cvar-projection",
        score=False,
        max_samples=1,
        max_sandboxes=1,
        display="none",
    )
    cpu = log.samples[0].metadata["swarm"]["cpu"]
    print("status:", log.status)
    print({k: v for k, v in cpu.items() if k != "samples"})
