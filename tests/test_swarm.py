"""Behavioural tests for the swarm solver, driven by scripted mockllm agents.

Each agent gets its own mockllm Model whose outputs are a fixed script of tool
calls, so the tests never touch a real model or Docker (sandbox = local).
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.event import ToolEvent
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolDef
from inspect_ai.util import ExecResult
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment

from swarm_scaling import swarm as swarm_module
from swarm_scaling.swarm import Candidate, kill_agent_processes, snapshot_container_pids, swarm

MODEL = "mockllm/model"


def call(tool: str, usage: ModelUsage | None = None, **args: Any) -> ModelOutput:
    out = ModelOutput.for_tool_call(MODEL, tool, args)
    out.usage = usage
    return out


def submit(answer: str = "done") -> ModelOutput:
    return call("submit", answer=answer)


def scripted(*outputs: ModelOutput):
    return get_model(MODEL, custom_outputs=list(outputs))


def run_swarm(tmp_path: Path, solver, **eval_kwargs: Any) -> EvalLog:
    task = Task(dataset=[Sample(input="Solve the task.")], solver=solver, sandbox="local")
    (log,) = eval(task, model=MODEL, display="none", log_dir=str(tmp_path / "logs"), **eval_kwargs)
    assert log.status == "success", log.error
    return log


def swarm_meta(log: EvalLog) -> dict[str, Any]:
    return log.samples[0].metadata["swarm"]


def tool_events(log: EvalLog, function: str) -> list[ToolEvent]:
    return [e for e in log.samples[0].events if isinstance(e, ToolEvent) and e.function == function]


def test_solo_runs_and_calls_finalize(tmp_path: Path) -> None:
    seen: list[list[Candidate]] = []

    async def finalize(state, candidates):
        seen.append(candidates)

    log = run_swarm(
        tmp_path,
        swarm(
            models=[scripted(call("bash", command="echo hi"), submit("answer"))],
            per_agent_tokens=100_000,
            workspace_root=str(tmp_path / "ws"),
            finalize=finalize,
        ),
    )
    agents = swarm_meta(log)["agents"]
    assert list(agents) == ["agent_0"]
    assert agents["agent_0"]["end_reason"] == "submitted"
    assert seen and [c.kind for c in seen[0]] == ["final_workspace"]
    assert seen[0][0].path == str(tmp_path / "ws" / "agents" / "agent_0")
    assert log.samples[0].output.completion.endswith("answer")
    assert swarm_meta(log)["process_cleanup"] == "skipped: not a Docker sandbox (LocalSandboxEnvironment)"


def test_message_delivered_in_recipients_next_tool_result(tmp_path: Path) -> None:
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(call("send_message", to="agent_1", text="hello from 0"), submit()),
                scripted(call("bash", command="sleep 0.5"), call("bash", command="echo later"), submit()),
                scripted(submit()),
            ],
            per_agent_tokens=100_000,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    bash_results = [str(e.result) for e in tool_events(log, "bash")]
    delivered = [r for r in bash_results if "hello from 0" in r and "from agent_0" in r]
    assert len(delivered) == 1, bash_results
    agents = swarm_meta(log)["agents"]
    assert agents["agent_0"]["messages_sent"][0]["to"] == "agent_1"
    assert agents["agent_1"]["messages_received"][0]["text"] == "hello from 0"


def test_independent_mode_exposes_no_message_or_registry_tools(tmp_path: Path) -> None:
    seen_tools: set[str] = set()
    prompts: list[str] = []

    def script(input, tools, tool_choice, config):
        seen_tools.update(t.name for t in tools)
        prompts.append(input[-1].text)
        return submit()

    # An independent run is a single agent that can save candidates (same pool mechanism as
    # teams) but has no peers or messaging.
    log = run_swarm(
        tmp_path,
        swarm(
            models=[get_model(MODEL, custom_outputs=script)],
            per_agent_tokens=100_000,
            messaging=False,
            registry=True,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    assert seen_tools == {"bash", "python", "update_plan", "check_budget", "submit", "publish_candidate", "list_candidates"}
    for prompt in prompts:
        assert "other agents" not in prompt and "message" not in prompt and "shared" not in prompt
        assert "your saved candidates" in prompt
    assert swarm_meta(log)["messaging"] is False


def test_independent_agents_in_one_container_are_refused() -> None:
    # They would share a filesystem, so they would not be independent.
    with pytest.raises(ValueError, match="separate single-agent samples"):
        swarm(models=[MODEL, MODEL], per_agent_tokens=1000, messaging=False, registry=False)


def test_token_limit_stops_one_agent_while_others_continue(tmp_path: Path) -> None:
    heavy = ModelUsage(input_tokens=900, output_tokens=100, total_tokens=1000)
    light = ModelUsage(input_tokens=9, output_tokens=1, total_tokens=10)
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(*[call("bash", usage=heavy, command="true") for _ in range(10)]),
                # every scripted output carries usage: mockllm charges real token
                # counts (prompt + tool JSON, far over 1500) for outputs without it
                scripted(
                    call("bash", usage=light, command="true"),
                    call("bash", usage=light, command="true"),
                    call("submit", usage=light, answer="done"),
                ),
            ],
            per_agent_tokens=1500,
            budget_type="all",  # the scripted usage is mostly input; this tests the limit mechanism
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    agents = swarm_meta(log)["agents"]
    assert agents["agent_0"]["limit_hit"] == "token"
    assert agents["agent_0"]["tokens"]["metered"] == 2000
    assert agents["agent_1"]["end_reason"] == "submitted"
    assert agents["agent_1"]["limit_hit"] is None


def test_wait_for_message_returns_when_all_others_finished(tmp_path: Path) -> None:
    log = run_swarm(
        tmp_path,
        swarm(
            models=[scripted(submit()), scripted(call("wait_for_message", timeout_s=60), submit())],
            per_agent_tokens=100_000,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    (wait,) = tool_events(log, "wait_for_message")
    assert "finished or is also waiting" in str(wait.result)
    assert swarm_meta(log)["agents"]["agent_1"]["wall_s"] < 10


def test_publish_candidate_is_immutable_and_listed(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    sol = ws / "agents" / "agent_0" / "sol"
    seen: list[list[Candidate]] = []

    async def finalize(state, candidates):
        seen.append(candidates)

    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(
                    call("bash", command=f"mkdir -p {sol} && echo v1 > {sol}/a.txt"),
                    call("publish_candidate", path=str(sol), note="first version"),
                    call("bash", command=f"echo v2 > {sol}/a.txt"),
                    submit(),
                ),
                scripted(call("bash", command="sleep 0.5"), call("list_candidates"), submit()),
            ],
            per_agent_tokens=100_000,
            workspace_root=str(ws),
            finalize=finalize,
        ),
    )
    entry = ws / "registry" / "agent_0-0"
    assert (entry / "a.txt").read_text() == "v1\n"
    assert not os.access(entry / "a.txt", os.W_OK)
    (listing,) = tool_events(log, "list_candidates")
    assert "agent_0-0" in str(listing.result) and "first version" in str(listing.result)
    published = [c for c in seen[0] if c.kind == "published"]
    assert [c.path for c in published] == [str(entry)]
    agents = swarm_meta(log)["agents"]
    assert agents["agent_0"]["candidates_published"] == ["agent_0-0"]
    assert agents["agent_1"]["candidates_read"] == ["agent_0-0"]


def test_one_agent_raising_does_not_cancel_others(tmp_path: Path) -> None:
    def boom(input, tools, tool_choice, config):
        raise RuntimeError("boom")

    log = run_swarm(
        tmp_path,
        swarm(
            models=[get_model(MODEL, custom_outputs=boom), scripted(call("bash", command="true"), submit())],
            per_agent_tokens=100_000,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    agents = swarm_meta(log)["agents"]
    assert agents["agent_0"]["end_reason"] == "error" and "boom" in agents["agent_0"]["error"]
    assert agents["agent_1"]["end_reason"] == "submitted"


def exec_result(stdout: str, success: bool = True) -> ExecResult[str]:
    return ExecResult(success=success, returncode=0 if success else 1, stdout=stdout, stderr="")


@pytest.mark.asyncio
async def test_cleanup_never_executes_on_local_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inspect hands solvers a proxy, so the guard must look through it. The host
    # was SIGKILLed twice when this check matched the proxy instead of the env.
    local = LocalSandboxEnvironment()
    local.exec = AsyncMock(return_value=exec_result("ok"))  # type: ignore[method-assign]
    proxy = SandboxEnvironmentProxy(local)
    proxy.exec = AsyncMock(return_value=exec_result("ok"))  # type: ignore[method-assign]
    monkeypatch.setattr(swarm_module, "sandbox", lambda name=None: proxy)

    assert await snapshot_container_pids() is None
    assert await kill_agent_processes([1, 7]) == "skipped: not a Docker sandbox (LocalSandboxEnvironment)"
    proxy.exec.assert_not_called()
    local.exec.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_probes_inside_the_container_before_killing(monkeypatch: pytest.MonkeyPatch) -> None:
    docker = DockerSandboxEnvironment.__new__(DockerSandboxEnvironment)
    proxy = SandboxEnvironmentProxy(docker)
    monkeypatch.setattr(swarm_module, "sandbox", lambda name=None: proxy)

    # a Docker-typed env whose probe does not confirm a container: probe only, no kill
    proxy.exec = AsyncMock(return_value=exec_result("no /.dockerenv", success=False))  # type: ignore[method-assign]
    assert (await kill_agent_processes([1, 7])).startswith("skipped: container probe failed")
    assert [c.args[0][2] for c in proxy.exec.call_args_list] == [swarm_module._CONTAINER_PROBE]

    # no baseline snapshot: probe only, no kill
    proxy.exec = AsyncMock(return_value=exec_result("ok"))  # type: ignore[method-assign]
    assert await kill_agent_processes(None) == "skipped: no baseline PID snapshot"
    assert [c.args[0][2] for c in proxy.exec.call_args_list] == [swarm_module._CONTAINER_PROBE]

    # probe confirms a container: the kill script runs second, sparing the baseline
    proxy.exec = AsyncMock(side_effect=[exec_result("ok"), exec_result("killed 3")])  # type: ignore[method-assign]
    assert await kill_agent_processes([1, 7]) == "killed 3"
    calls = [c.args[0] for c in proxy.exec.call_args_list]
    assert [c[2] for c in calls] == [swarm_module._CONTAINER_PROBE, swarm_module._KILL_SCRIPT]
    assert calls[1][3:] == ["sh", "1 7"]


def test_cpu_summary_flags_a_saturated_container() -> None:
    # Teams share one container's CPUs; the run must say when the agents maxed them out.
    probe = "usage_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n\nmax_line 800000 100000"
    assert swarm_module.parse_cpu_probe(probe)["limit_cores"] == 8.0

    def sample(t: float, used_cpu_s: float, periods: int, throttled: int) -> dict[str, float]:
        return {"t": t, "usage_usec": used_cpu_s * 1e6, "nr_periods": periods, "nr_throttled": throttled,
                "throttled_usec": 0.0, "limit_cores": 8.0}

    busy = swarm_module.summarise_cpu([sample(0, 0, 0, 0), sample(10, 78, 100, 40), sample(20, 157, 200, 80)])
    idle = swarm_module.summarise_cpu([sample(0, 0, 0, 0), sample(10, 4, 100, 0), sample(20, 8, 200, 0)])
    assert busy["flag"] is True and busy["mean_util"] > 0.9
    assert idle["flag"] is False and idle["mean_util"] < 0.1


@pytest.mark.asyncio
async def test_check_budget_reports_this_agents_usage_and_remaining() -> None:
    # Agents are told their budget up front and may stop early; they need to see what is left.
    class FakeLimit:
        usage, limit = 1500, 2_000_000

    tool = swarm_module.budget_tool(FakeLimit(), "all")
    out = await tool()
    assert "Used 1,500 of 2,000,000 tokens" in out and "1,998,500 remaining" in out


def test_send_file_copies_into_the_recipients_inbox_and_tells_them_where(tmp_path: Path) -> None:
    # Teams share drafts, data and notes directly, not only text messages and finished candidates.
    ws = tmp_path / "ws"
    sender_dir = ws / "agents" / "agent_0"
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(
                    call("bash", command=f"mkdir -p {sender_dir}/notes && echo idea > {sender_dir}/notes/plan.md"),
                    call("send_file", to="agent_1", path="notes", note="my plan"),
                    submit(),
                ),
                scripted(call("bash", command="sleep 0.5"), call("bash", command="echo after"), submit()),
            ],
            per_agent_tokens=100_000,
            workspace_root=str(ws),
        ),
    )
    copied = ws / "agents" / "agent_1" / "inbox" / "from-agent_0-0" / "notes" / "plan.md"
    assert copied.read_text() == "idea\n"
    assert (sender_dir / "notes" / "plan.md").read_text() == "idea\n"
    delivered = [str(e.result) for e in tool_events(log, "bash") if "[file] agent_0 sent you" in str(e.result)]
    assert len(delivered) == 1 and "my plan" in delivered[0] and str(copied.parent.parent) in delivered[0]
    assert swarm_meta(log)["agents"]["agent_0"]["files_sent"][0]["to"] == "agent_1"


def used(n: int) -> ModelUsage:
    return ModelUsage(input_tokens=n, output_tokens=0, total_tokens=n)


def test_budget_notice_appears_once_per_threshold_on_any_tool(tmp_path: Path) -> None:
    # Pilot: an agent ran out of budget mid-work without saving, never having called check_budget.
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(
                    call("bash", usage=used(400), command="true"),  # 40%
                    call("send_message", usage=used(150), to="agent_1", text="hi"),  # 55%: crosses 50%
                    call("bash", usage=used(50), command="true"),  # 60%
                    call("bash", usage=used(100), command="true"),  # 70%
                    call("bash", usage=used(100), command="true"),  # 80%: crosses 75%
                    call("submit", usage=used(1), answer="done"),
                ),
                scripted(call("submit", usage=used(1), answer="done")),
            ],
            per_agent_tokens=1000,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    results = [str(e.result) for e in tool_events(log, "bash") + tool_events(log, "send_message")]
    notices = sorted(r[r.index("[budget]"):] for r in results if "[budget]" in r)
    assert notices == [
        "[budget] You have used 55% of your token budget (550 of 1,000).",
        "[budget] You have used 80% of your token budget (800 of 1,000).",
    ]
    assert "[budget]" in str(tool_events(log, "send_message")[0].result)
    log_notices = swarm_meta(log)["agents"]["agent_0"]["budget_notices"]
    assert [(n["threshold"], n["used"]) for n in log_notices] == [(0.5, 550), (0.75, 800)]


def test_solo_agent_gets_only_the_highest_of_several_thresholds_passed_at_once(tmp_path: Path) -> None:
    log = run_swarm(
        tmp_path,
        swarm(
            models=[scripted(call("bash", usage=used(920), command="true"), call("submit", usage=used(1), answer="x"))],
            per_agent_tokens=1000,
            messaging=False,
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    (bash_event,) = tool_events(log, "bash")
    assert str(bash_event.result).count("[budget]") == 1 and "used 92%" in str(bash_event.result)
    assert [n["threshold"] for n in swarm_meta(log)["agents"]["agent_0"]["budget_notices"]] == [0.9]


def test_claude_code_style_swaps_the_message_tool_and_adds_a_task_list(tmp_path: Path) -> None:
    seen_tools: set[str] = set()
    prompts: list[str] = []

    def script(input, tools, tool_choice, config):
        seen_tools.update(t.name for t in tools)
        prompts.append(input[-1].text)
        return submit()

    run_swarm(
        tmp_path,
        swarm(
            models=[get_model(MODEL, custom_outputs=script), scripted(submit())],
            per_agent_tokens=100_000,
            tool_style="claude_code",
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    assert {"SendMessage", "TaskCreate", "TaskList", "TaskUpdate", "TaskGet", "read_message", "wait_for_message"} <= seen_tools
    assert "send_message" not in seen_tools
    assert "SendMessage(to, message, summary)" in prompts[0] and "TaskCreate(subject, description)" in prompts[0]
    assert "send_message" not in prompts[0] and "send_file(to, path, note)" in prompts[0]
    assert "How you work together, if at all, is up to you." in prompts[0]


def test_claude_code_message_arrives_as_a_teammate_message_block(tmp_path: Path) -> None:
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(call("SendMessage", to="agent_1", message="hello from 0", summary="greeting"), submit()),
                scripted(call("bash", command="sleep 0.5"), call("bash", command="echo later"), submit()),
            ],
            per_agent_tokens=100_000,
            tool_style="claude_code",
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    delivered = [str(e.result) for e in tool_events(log, "bash") if "hello from 0" in str(e.result)]
    assert len(delivered) == 1
    assert '<teammate-message teammate_id="agent_0" summary="greeting" message_id="m0">\nhello from 0\n</teammate-message>' in delivered[0]


def test_task_created_by_one_agent_is_claimed_and_completed_by_another(tmp_path: Path) -> None:
    log = run_swarm(
        tmp_path,
        swarm(
            models=[
                scripted(
                    call("TaskCreate", subject="profile solver", description="find the hot loop"),
                    call("bash", command="sleep 1.5"),
                    call("TaskGet", taskId="1"),
                    submit(),
                ),
                scripted(
                    call("bash", command="sleep 0.5"),
                    call("TaskList"),
                    call("TaskUpdate", taskId="1", owner="agent_1", status="in_progress"),
                    call("TaskUpdate", taskId="1", status="completed"),
                    submit(),
                ),
            ],
            per_agent_tokens=100_000,
            tool_style="claude_code",
            workspace_root=str(tmp_path / "ws"),
        ),
    )
    (listing,) = tool_events(log, "TaskList")
    assert "#1 [pending] profile solver" in str(listing.result) and "created by: agent_0" in str(listing.result)
    (got,) = tool_events(log, "TaskGet")
    assert all(s in str(got.result) for s in ("status: completed", "owner: agent_1", "find the hot loop"))
    events = swarm_meta(log)["tasks"]
    assert [(e["event"], e["agent"]) for e in events] == [("create", "agent_0"), ("update", "agent_1"), ("update", "agent_1")]


def test_task_tools_reach_every_agent_in_every_arm_and_the_prompt_says_where_the_answer_goes(tmp_path: Path) -> None:
    # Parity: solo and team agents get the same task tools (AlgoTune: dev_eval). Agents are told the
    # chosen answer is written to the deliverable path, so writing it themselves has no effect.
    def probe_tools(state):
        def for_agent(agent_id: str):
            async def execute() -> str:
                """Probe."""
                return agent_id

            return [ToolDef(execute, name="probe").as_tool()]

        return for_agent

    for n in (1, 2):
        seen: list[set[str]] = []
        prompts: list[str] = []

        def script(input, tools, tool_choice, config):
            seen.append({t.name for t in tools})
            prompts.append(input[-1].text)
            return submit()

        run_swarm(
            tmp_path / f"n{n}",
            swarm(
                models=[get_model(MODEL, custom_outputs=script) for _ in range(n)],
                per_agent_tokens=100_000,
                workspace_root=str(tmp_path / f"ws{n}"),
                agent_tools=probe_tools,
                final_path="/app/solver.py",
            ),
        )
        assert len(seen) == n and all("probe" in tools for tools in seen)
        assert all(
            "The chosen answer is then written to /app/solver.py, so writing to /app/solver.py yourself has no effect."
            in p
            for p in prompts
        )
