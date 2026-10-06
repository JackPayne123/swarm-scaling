"""Task factory for the Harvey LAB diligence data rooms (tasks/harvey_diligence, built by scripts/convert_harvey_diligence.py)."""

from pathlib import Path

from inspect_ai import Task, task, task_with
from inspect_ai.util import ComposeConfig
from inspect_harbor import harbor, harbor_scorer

from swarm_scaling.harvey_grader import DEFAULT_JUDGE, harvey_grader

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks" / "harvey_diligence"

# Harvey's published agent timeout. inspect_harbor ignores task.toml's [agent].timeout_sec, so it is applied
# as the sample time limit, and Inspect then caps scoring at half of it. Unmeasured: set from the pilot (PLAN.md).
AGENT_TIME_LIMIT_SEC = 7200  # per agent; enforce via swarm(time_limit=...), not the sample limit
# Inspect gives scoring half the sample time limit (inspect_ai/_eval/task/run.py). 1,114 criteria at
# 16 concurrent judge calls is ~70 waves, so 4 h for the sample leaves 2 h for scoring. Unmeasured; set from the pilot.
SAMPLE_TIME_LIMIT_SEC = 4 * 3600
JUDGE_TIMEOUT_SEC = 300  # per judge call, rewardkit's default


@task
def harvey_task(
    names: list[str] | None = None,
    judge: str = DEFAULT_JUDGE,
    batch_size: int = 1,
    judge_concurrency: int = 16,
    judge_timeout: int = JUDGE_TIMEOUT_SEC,
    time_limit: int = SAMPLE_TIME_LIMIT_SEC,
    rewardkit_parity: bool = False,
) -> Task:
    """The diligence tasks with no container network, graded on the host by `harvey_grader`.

    Args:
        names: Task directory names or globs, e.g. ["diligence-media-recap"]. Default: all 11.
        judge: LiteLLM model string for the judge (e.g. "gemini/gemini-3.1-pro-preview" for the frontier check).
        batch_size: Criteria per judge call. 1 is Harvey's protocol; see harvey_grader for the tradeoff.
        judge_concurrency: Judge calls in flight across all samples.
        judge_timeout: Seconds per judge call.
        time_limit: Sample wall-clock limit in seconds. Scoring gets half of it, so keep it well above the
            agents' own limit (AGENT_TIME_LIMIT_SEC, passed to swarm(time_limit=...)).
        rewardkit_parity: Also run the original in-container rewardkit verifier as a second scorer. It
            pip-installs and calls the judge from inside the container, so this turns container network on,
            which lets an agent fetch the public rubric. Use it only to grade a fixed deliverable, never for
            a scored agent run.
    """
    base = harbor(path=TASKS_DIR, dataset_task_names=names)
    for sample in base.dataset:
        assert sample.sandbox is not None
        config = sample.sandbox.config
        assert isinstance(config, ComposeConfig)
        service = config.services["default"]
        # task.toml sets network_mode = "no-network"; refuse to run if a regenerated task lost it.
        assert service.network_mode == "none", f"{sample.id}: agent container has network ({service.network_mode})"
        if rewardkit_parity:
            service.network_mode = "bridge"
    grader = harvey_grader(judge=judge, batch_size=batch_size, max_concurrency=judge_concurrency, judge_timeout=judge_timeout)
    return task_with(base, scorer=[grader, harbor_scorer()] if rewardkit_parity else grader, time_limit=time_limit)
