"""End-to-end wiring check: two mockllm agents on a trivial local-sandbox task.

    uv run inspect eval scripts/smoke_swarm.py --model mockllm/model
    uv run python scripts/smoke_swarm.py

The default mockllm output never calls a tool, so react keeps prompting it to
continue until each agent's token budget runs out. Both agents must end with
limit_hit == "token" and the swarm record must land in sample metadata.
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample

from swarm_scaling.swarm import swarm


@task
def smoke(per_agent_tokens: int = 3000, workspace_root: str = "/tmp/swarm-smoke") -> Task:
    return Task(
        dataset=[Sample(input="Write hello to a file in your working directory.")],
        solver=swarm(models=["mockllm/model"] * 2, per_agent_tokens=per_agent_tokens, workspace_root=workspace_root),
        sandbox="local",
    )


if __name__ == "__main__":
    (log,) = eval(smoke(), model="mockllm/model", display="plain")
    assert log.status == "success", log.error
    agents = log.samples[0].metadata["swarm"]["agents"]
    for agent_id, rec in agents.items():
        print(agent_id, rec["model"], rec["end_reason"], rec["tokens"])
        assert rec["limit_hit"] == "token", rec
    print("smoke ok")
