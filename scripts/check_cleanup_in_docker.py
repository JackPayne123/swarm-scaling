"""Run the swarm solver with mockllm inside one AlgoTune Docker sample to check that
kill_agent_processes runs inside the container (and only there).

Usage: uv run python scripts/check_cleanup_in_docker.py
"""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from swarm_scaling.algotune_devkit import algotune_finalize
from swarm_scaling.swarm import swarm
from swarm_scaling.tasks import algotune_task


def scripted(*calls: tuple[str, dict]):
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.for_tool_call("mockllm/model", t, a) for t, a in calls],
    )


if __name__ == "__main__":
    task = algotune_task(split="pilot", n_agents=2)
    # agent_0 leaves a background process behind; cleanup must kill it inside the
    # container and leave the container itself running for scoring.
    leaver = scripted(
        ("bash", {"command": "nohup sleep 600 >/dev/null 2>&1 & echo started"}),
        ("submit", {"answer": "done"}),
    )
    quiet = scripted(("submit", {"answer": "done"}))
    logs = eval(
        task,
        solver=swarm(
            models=[leaver, quiet],
            per_agent_tokens=6000,
            finalize=algotune_finalize,  # algotune_task already runs algotune_setup via Task.setup
        ),
        model="mockllm/model",
        sample_id="algotune/cvar-projection",
        max_samples=1,
        max_sandboxes=1,
        display="none",
    )
    sample = logs[0].samples[0]
    swarm_meta = sample.metadata.get("swarm", {})
    print("status:", logs[0].status)
    print("process_cleanup:", swarm_meta.get("process_cleanup"))
    print("score:", {k: v.value for k, v in (sample.scores or {}).items()})
