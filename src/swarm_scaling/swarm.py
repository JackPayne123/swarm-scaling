"""Multi-agent swarm solver: N flat peers on one task, each with its own model and budget.

Design and rationale: HARNESS.md ("Solver design"). Each agent is inspect_harbor's
default react() scaffold with its own model, token budget and optional messaging
and registry tools; agents run concurrently in one sandbox.
"""

from __future__ import annotations

import logging
import time

import anyio
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Literal

from inspect_ai.agent import AgentState, AgentSubmit, react, run
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    CompactionEdit,
    ContentText,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, bash, python, update_plan
from inspect_ai.util import LimitExceededError, collect, sandbox, token_limit
from inspect_ai.util import time_limit as wall_clock_limit
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

from swarm_scaling.team import (
    Team,
    list_candidates_tool,
    publish_candidate_tool,
    read_message_tool,
    send_file_tool,
    send_message_tool,
    wait_for_message_tool,
    with_delivery,
)

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    agent_id: str
    model: str
    path: str
    note: str
    published_at: float
    kind: Literal["published", "final_workspace"]


Finalize = Callable[[TaskState, list[Candidate]], Awaitable[None]]

DEFAULT_SELECTION_RULE = "a rule fixed before the run"


@solver
def swarm(
    models: list[str | Model],
    per_agent_tokens: int,
    messaging: bool = True,
    registry: bool = True,
    workspace_root: str = "/app",
    setup: Solver | None = None,
    finalize: Finalize | None = None,
    protocol_prompt: str | None = None,
    reasoning_effort: dict[str, str] | None = None,
    time_limit: int | None = None,
    reveal_model_family: bool = False,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    cpu_sample_interval: float = 15.0,
    budget_type: str = "all",
    deliverable: str | None = None,
) -> Solver:
    """Run N concurrent agents on the sample, then hand their candidates to `finalize`.

    Arms: solo and independent runs are separate N = 1 samples with registry=True (saved
    candidates), so every arm can checkpoint; independent teams are resampled from those
    runs. N agents in one container share a filesystem, so N > 1 with both messaging and
    registry off is refused: it would not be independent.

    Args:
        models: One model id (or Model) per agent; len(models) = N. N = 1 is the solo arm.
        per_agent_tokens: Hard per-agent token budget (Inspect token_limit, all tokens).
        messaging: Give agents send_message / wait_for_message and inject replies.
        registry: Give agents publish_candidate / list_candidates and a shared registry.
        workspace_root: Private dirs at {root}/agents/agent_{i}, registry at {root}/registry.
        setup: Task-specific solver run once before the agents start.
        finalize: Writes the task's single deliverable from the collected candidates.
        protocol_prompt: Replaces the default team-protocol text appended to the instruction.
        reasoning_effort: Explicit reasoning effort per model id.
        time_limit: Per-agent wall-clock limit in seconds.
        reveal_model_family: Include the sender's model id in message envelopes.
        selection_rule: How the final answer is chosen, stated to agents (e.g. AlgoTune:
            "the fastest correct candidate on the dev inputs").
        budget_type: What per_agent_tokens meters: "all" (default since 2026-10-08: input incl. cached
            plus output, so models that think or re-read differently are matched on what they consume),
            "output" (Ord's unit), or an Inspect formula over input/output.
        deliverable: What counts as a solution, stated to agents as a fact (task-specific), e.g.
            AlgoTune: a file named solver.py. Candidates are found by this name, so state it.
        cpu_sample_interval: Seconds between container CPU samples (Docker only); the summary in
            state.metadata["swarm"]["cpu"] flags samples where the agents saturated the container's CPUs.
    """
    if not models:
        raise ValueError("swarm() needs at least one model")
    n = len(models)
    if n > 1 and not messaging and not registry:
        raise ValueError(
            "independent arm: run N separate single-agent samples and resample them into teams; "
            "agents in one container share a filesystem and are not independent"
        )
    agent_ids = [f"agent_{i}" for i in range(n)]
    model_names = [m if isinstance(m, str) else m.name for m in models]
    messaging = messaging and n > 1

    def resolve_model(m: str | Model) -> Model:
        if isinstance(m, Model):
            return m
        effort = (reasoning_effort or {}).get(m)
        config = GenerateConfig(reasoning_effort=effort) if effort else GenerateConfig()  # type: ignore[arg-type]
        return get_model(m, config=config)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if setup is not None:
            state = await setup(state, generate)

        team = Team(agent_ids, model_names, workspace_root, reveal_model_family)
        dirs = [team.registry_dir] + [a.private_dir for a in team.agents.values()]
        mk = await sandbox().exec(["mkdir", "-p", *dirs])
        if not mk.success:
            raise RuntimeError(f"could not create swarm workspace: {mk.stderr}")

        started = time.time()
        baseline_pids = await snapshot_container_pids()

        async def run_agent(agent_id: str, model: str | Model) -> AgentState | None:
            rec = team.agents[agent_id]
            tokens = token_limit(per_agent_tokens, type=budget_type)
            limits = [tokens]
            if time_limit is not None:
                limits.append(wall_clock_limit(time_limit))

            def submitted(answer: str) -> None:
                rec.submitted = True

            agent = react(
                name=agent_id,
                tools=_agent_tools(team, agent_id, messaging, registry, budget_tool(tokens, budget_type)),
                model=resolve_model(model),
                submit=AgentSubmit(tool=_submit_tool(submitted)),
                compaction=CompactionEdit(),
            )
            protocol = protocol_prompt or default_protocol_prompt(
                team,
                agent_id,
                messaging=messaging,
                registry=registry,
                selection_rule=selection_rule,
                budget_tokens=per_agent_tokens,
                budget_type=budget_type,
                deliverable=deliverable,
            )
            rec.started_at = time.time()
            agent_state: AgentState | None = None
            try:
                agent_state, limit_error = await run(
                    agent,
                    _with_protocol(state.messages, protocol),
                    limits=limits,
                    name=agent_id,
                )
                if limit_error is not None:
                    rec.limit_hit = limit_error.type
                    rec.end_reason = limit_error.type
                else:
                    rec.end_reason = "submitted" if rec.submitted else "stopped"
            except LimitExceededError:
                # a sample-level limit, not one of ours: let it end the sample
                raise
            except Exception as ex:  # one agent's failure must not cancel its peers
                rec.end_reason = "error"
                rec.error = f"{type(ex).__name__}: {ex}"
                logger.warning("swarm %s failed: %s", agent_id, rec.error)
            finally:
                rec.ended_at = time.time()
                usage = getattr(tokens, "_usage", None)
                rec.tokens = {"metered": int(tokens.usage)}
                if usage is not None:
                    rec.tokens.update(usage.model_dump(exclude_none=True))
                team.set_status(agent_id, "finished")
            return agent_state

        cpu = CpuMonitor(cpu_sample_interval) if baseline_pids is not None else None
        async with anyio.create_task_group() as tg:
            if cpu is not None:
                tg.start_soon(cpu.run)
            try:
                results = await collect(*(run_agent(aid, m) for aid, m in zip(agent_ids, models)))
            finally:
                if cpu is not None:
                    cpu.stop()

        cleanup = await kill_agent_processes(baseline_pids)

        candidates = [
            Candidate(e.agent_id, e.model, e.path, e.note, e.published_at, "published")
            for e in team.entries
        ]
        for rec, result in zip(team.agents.values(), results):
            note = result.output.completion if result is not None else ""
            candidates.append(
                Candidate(
                    rec.agent_id,
                    rec.model,
                    rec.private_dir,
                    note[:2000],
                    rec.ended_at or time.time(),
                    "final_workspace",
                )
            )

        state.metadata["swarm"] = {
            "n": n,
            "messaging": messaging,
            "registry": registry,
            "workspace_root": workspace_root,
            "per_agent_tokens": per_agent_tokens,
            "time_limit": time_limit,
            "started_at": started,
            "ended_at": time.time(),
            "process_cleanup": cleanup,
            "cpu": cpu.summary() if cpu is not None else {"skipped": "not a Docker container"},
            "candidates": [asdict(c) for c in candidates],
            "agents": {aid: team.agents[aid].as_log() for aid in agent_ids},
        }

        if n == 1 and results[0] is not None:
            state.messages = results[0].messages
            state.output = results[0].output

        if finalize is not None:
            await finalize(state, candidates)
        return state

    return solve


def budget_tool(limit, budget_type: str) -> Tool:
    """check_budget(): tokens used and remaining under this agent's own limit (no effect on the budget)."""

    async def execute() -> str:
        """Show how much of your token budget you have used and how much remains.

        You may stop at any time by submitting; you do not have to use the whole budget.
        """
        used = int(limit.usage)
        cap = int(limit.limit) if limit.limit is not None else None
        unit = "output tokens" if budget_type == "output" else "tokens"
        if cap is None:
            return f"Used {used:,} {unit}; no limit."
        return f"Used {used:,} of {cap:,} {unit}; {max(cap - used, 0):,} remaining."

    return ToolDef(execute, name="check_budget").as_tool()


def _agent_tools(
    team: Team, agent_id: str, messaging: bool, registry: bool, budget: Tool | None = None
) -> list[Tool]:
    tools: list[Tool] = [bash(timeout=300), python(timeout=300), update_plan()]
    if budget is not None:
        tools.append(budget)
    if registry:
        tools += [publish_candidate_tool(team, agent_id), list_candidates_tool(team, agent_id)]
    if not messaging:
        return tools
    # every tool result is a delivery point; wait_for_message drains its own inbox
    tools = [with_delivery(t, team, agent_id) for t in tools]
    tools.append(with_delivery(send_message_tool(team, agent_id), team, agent_id))
    tools.append(with_delivery(read_message_tool(team, agent_id), team, agent_id))
    tools.append(with_delivery(send_file_tool(team, agent_id), team, agent_id))
    tools.append(wait_for_message_tool(team, agent_id))
    return tools


def _submit_tool(on_submit: Callable[[str], None]) -> Tool:
    async def execute(answer: str) -> str:
        """Submit an answer for evaluation.

        Args:
            answer: Submitted answer
        """
        on_submit(answer)
        return answer

    return ToolDef(execute, name="submit").as_tool()


def _with_protocol(messages: list[ChatMessage], protocol: str) -> list[ChatMessage]:
    """Copy the sample messages with the protocol appended to the last user message."""
    out = [m.model_copy(deep=True) for m in messages]
    for m in reversed(out):
        if isinstance(m, ChatMessageUser):
            if isinstance(m.content, str):
                m.content = f"{m.content}\n\n{protocol}"
            else:
                m.content = [*m.content, ContentText(text=f"\n\n{protocol}")]
            return out
    out.append(ChatMessageUser(content=protocol))
    return out


def default_protocol_prompt(
    team: Team,
    agent_id: str,
    *,
    messaging: bool,
    registry: bool,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    budget_tokens: int = 0,
    budget_type: str = "all",
    deliverable: str | None = None,
) -> str:
    # The loose, facts-only prompt (research/PROTOCOL.md, PLAN.md decision 2026-10-05):
    # no roles, no message rules, no anti-herding text. Every arm can save candidates, so
    # the size of the pool the final rule picks from does not depend on the arm; only
    # whether agents can see each other's candidates and messages does.
    rec = team.agents[agent_id]
    others = [a.agent_id for a in team.others(agent_id)]
    lines = ["## Working protocol", ""]
    if others:
        lines.append(
            f"You are {agent_id}, one of {len(team.agents)} agents working on this task "
            f"at the same time. The other agents are {', '.join(others)}."
        )
    else:
        lines.append("You are working on this task alone.")
    lines.append(f"- Your private working directory is {rec.private_dir}.")
    if deliverable:
        lines.append(f"- A solution is {deliverable}. Keep your current best one in your private working directory.")
    if messaging:
        lines.append(
            '- send_message(to, text) sends a message to one agent ("agent_i") or to "all". '
            "Messages sent to you appear after your next tool result (long ones as a preview; "
            "read_message(id) shows the full text). wait_for_message(timeout_s) waits for one."
        )
        lines.append(
            "- send_file(to, path, note) sends a copy of any file or folder to one agent or to \"all\"; "
            f"it lands in their {team.root}/agents/<id>/inbox folder and they get a message saying where."
        )
    if registry and others:
        lines.append(
            "- publish_candidate(path, note) adds a solution to a shared registry "
            f"({team.registry_dir}) that every agent can read; list_candidates() lists it."
        )
    elif registry:
        lines.append(
            "- publish_candidate(path, note) saves a copy of a solution as a candidate; "
            "list_candidates() lists your saved candidates."
        )
    if others:
        lines.append(
            f"- Other agents' working directories are readable at {team.root}/agents/<agent_id>. "
            "You can share any file or folder by publishing it"
            + (", sending it with send_file, or sending its path in a message." if messaging else ".")
        )
    if others and (messaging or registry):
        lines.append("")
        lines.append("How you work together, if at all, is up to you.")
    lines.append("")
    pool = "all published candidates" if others else "your saved candidates"
    if not registry:
        pool = None
    where = "every agent's final working directory" if others else "your final working directory"
    lines.append(
        ("When every agent has finished, one final answer" if others else "When you finish, the final answer")
        + " is chosen from "
        + (f"{pool} and {where}" if pool else where)
        + f" by a fixed rule: {selection_rule}."
    )
    lines.append("")
    if budget_type == "output":
        lines.append(f"You have a budget of {budget_tokens:,} output tokens (everything you generate, including reasoning).")
    else:
        lines.append(
            f"You have a budget of {budget_tokens:,} tokens, counting everything sent to and generated by "
            "the model on each call (your conversation so far, tool results"
            + (", messages you receive" if messaging else "")
            + " and your output)."
        )
    lines.append(
        "check_budget() shows how much you have used. You may stop at any time by submitting; "
        "you do not have to use the whole budget."
    )
    return "\n".join(lines)


# Runs inside the sandbox before any kill. Prints "ok" only from a Linux Docker
# container: /.dockerenv exists, /proc is mounted, and PID 1 is not launchd.
_CONTAINER_PROBE = r"""
[ "$(uname -s 2>/dev/null)" = Linux ] || { echo "not linux"; exit 1; }
[ -f /.dockerenv ] || { echo "no /.dockerenv"; exit 1; }
[ -r /proc/1/comm ] || { echo "no /proc"; exit 1; }
case "$(cat /proc/1/comm)" in launchd*) echo "pid 1 is launchd"; exit 1;; esac
echo ok
"""

# Lists every PID in the container's PID namespace (one per line).
_PID_SNAPSHOT = r"""
for d in /proc/[0-9]*; do echo "${d#/proc/}"; done
"""

# Kills every PID in the container's PID namespace that was not running before the
# agents started ($1 = space-separated baseline PIDs), except PID 1, this shell and
# its ancestors. The baseline protects the container's own keepalive process (its
# PID 1 can be an init whose child keeps the container up; killing that child
# stopped the container in the first Docker test, 2026-10-05). Walks /proc instead
# of ps, so it does not depend on ps flags.
_KILL_SCRIPT = r"""
excl=" 1 $$ $1 "
p=$$
while [ "$p" -gt 1 ]; do
  p=$(awk '/^PPid:/ {print $2}' "/proc/$p/status" 2>/dev/null)
  [ -n "$p" ] || break
  excl="$excl$p "
done
pids=""
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$excl" in *" $pid "*) continue;; esac
  pids="$pids $pid"
done
killed=0
for pid in $pids; do
  kill -9 "$pid" 2>/dev/null && killed=$((killed+1))
done
echo "killed $killed"
"""


# Read-only: the container's cgroup v2 CPU accounting and limit.
_CPU_PROBE = "cat /sys/fs/cgroup/cpu.stat; echo; echo max_line $(cat /sys/fs/cgroup/cpu.max)"

# A sample interval counts as saturated when the container used at least this share of its CPU limit.
CPU_SATURATION_UTIL = 0.9
# A run is flagged when this share of intervals were saturated, or this share of CFS periods were throttled.
CPU_FLAG_SATURATED_FRAC = 0.25
CPU_FLAG_THROTTLED_FRAC = 0.10


def parse_cpu_probe(text: str) -> dict[str, float]:
    """cgroup cpu.stat fields plus `limit_cores` from cpu.max ("max" means unlimited -> 0)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = float(parts[1])
        elif parts[:1] == ["max_line"] and len(parts) == 3:
            quota, period = parts[1], parts[2]
            out["limit_cores"] = 0.0 if quota == "max" else float(quota) / float(period)
    return out


def summarise_cpu(samples: list[dict[str, float]]) -> dict[str, object]:
    """Utilisation per interval (CPU-seconds used / (wall seconds x limit)) and throttling, with a flag."""
    usable = [s for s in samples if "usage_usec" in s and "t" in s]
    if len(usable) < 2:
        return {"n_samples": len(usable), "flag": None, "note": "too few samples"}
    limit = usable[-1].get("limit_cores") or 0.0
    utils = []
    for a, b in zip(usable, usable[1:]):
        wall = b["t"] - a["t"]
        if wall > 0 and limit > 0:
            utils.append((b["usage_usec"] - a["usage_usec"]) / 1e6 / (wall * limit))
    first, last = usable[0], usable[-1]
    periods = last.get("nr_periods", 0) - first.get("nr_periods", 0)
    throttled = last.get("nr_throttled", 0) - first.get("nr_throttled", 0)
    throttled_frac = throttled / periods if periods > 0 else 0.0
    saturated_frac = sum(u >= CPU_SATURATION_UTIL for u in utils) / len(utils) if utils else 0.0
    return {
        "n_samples": len(usable),
        "limit_cores": limit,
        "mean_util": sum(utils) / len(utils) if utils else None,
        "max_util": max(utils) if utils else None,
        "saturated_interval_frac": saturated_frac,
        "throttled_period_frac": throttled_frac,
        "throttled_s": (last.get("throttled_usec", 0) - first.get("throttled_usec", 0)) / 1e6,
        "flag": saturated_frac >= CPU_FLAG_SATURATED_FRAC or throttled_frac >= CPU_FLAG_THROTTLED_FRAC,
        "samples": usable,
    }


class CpuMonitor:
    """Samples the container's cgroup CPU accounting while the agents run (Docker containers only)."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._done = anyio.Event()

    async def sample(self) -> None:
        try:
            result = await sandbox().exec(["sh", "-c", _CPU_PROBE], timeout=30)
        except Exception:
            return
        if result.success:
            self.samples.append({"t": time.time(), **parse_cpu_probe(result.stdout)})

    async def run(self) -> None:
        await self.sample()
        while not self._done.is_set():
            with anyio.move_on_after(self.interval):
                await self._done.wait()
            await self.sample()

    def stop(self) -> None:
        self._done.set()

    def summary(self) -> dict[str, object]:
        return summarise_cpu(self.samples)


async def _confirmed_docker_container() -> str | None:
    """Return None if the sandbox is positively a Linux Docker container, else why not."""
    sb = sandbox()
    real = getattr(sb, "_sandbox", sb)
    if not isinstance(real, DockerSandboxEnvironment):
        return f"not a Docker sandbox ({type(real).__name__})"
    probe = await sb.exec(["sh", "-c", _CONTAINER_PROBE], timeout=30)
    if not probe.success or probe.stdout.strip() != "ok":
        return f"container probe failed: {(probe.stdout + probe.stderr).strip()}"
    return None


async def snapshot_container_pids() -> list[int] | None:
    """PIDs running in the container before agents start; None outside a container."""
    try:
        if await _confirmed_docker_container() is not None:
            return None
        result = await sandbox().exec(["sh", "-c", _PID_SNAPSHOT], timeout=30)
    except Exception:
        return None
    if not result.success:
        return None
    return [int(p) for p in result.stdout.split() if p.isdigit()]


async def kill_agent_processes(baseline_pids: list[int] | None = None) -> str:
    """Kill processes started during the swarm, but only inside a Docker container.

    Scoring runs in the same container and AlgoTune scoring is timing-sensitive,
    so leftover agent processes must go. On 2026-10-05 an earlier version of this
    ran on the host (sandbox() returns a SandboxEnvironmentProxy, so an isinstance
    check on it never matched) and SIGKILLed every user process on Jack's Mac.
    Two positive checks now gate the kill, and nothing is executed unless both pass:

    1. the environment behind the proxy is a DockerSandboxEnvironment;
    2. a probe run inside the sandbox confirms a Linux Docker container
       (/.dockerenv, /proc, PID 1 not launchd).

    Processes in `baseline_pids` (snapshotted before the agents started) are spared,
    so the container's own processes survive. Without a baseline nothing is killed.
    """
    sb = sandbox()
    try:
        reason = await _confirmed_docker_container()
        if reason is not None:
            return f"skipped: {reason}"
        if baseline_pids is None:
            return "skipped: no baseline PID snapshot"
        baseline = " ".join(str(p) for p in baseline_pids)
        result = await sb.exec(["sh", "-c", _KILL_SCRIPT, "sh", baseline], timeout=60)
    except Exception as ex:
        return f"failed: {type(ex).__name__}: {ex}"
    if not result.success:
        return f"failed: {result.stderr.strip()}"
    return result.stdout.strip()
