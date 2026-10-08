"""The AlgoTune checker container: dev_eval and final selection run there, never in the agents' box.

Sandboxes are in-memory fakes (no Docker): "default" is the agents' box, "checker" fakes dev_eval.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from inspect_ai.util import ComposeConfig, ComposeService, ExecResult

from swarm_scaling import algotune_devkit as devkit
from swarm_scaling.swarm import Candidate
from swarm_scaling.tasks import add_checker


def ok(stdout: str = "") -> ExecResult[str]:
    return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")


class FakeBox:
    def __init__(self, files: dict[str, bytes | str] | None = None) -> None:
        self.files: dict[str, bytes | str] = dict(files or {})
        self.execs: list[list[str]] = []

    async def write_file(self, path: str, contents: bytes | str) -> None:
        self.files[path] = contents

    async def read_file(self, path: str, text: bool = True) -> bytes | str:
        if path not in self.files:
            raise FileNotFoundError(path)
        value = self.files[path]
        if text:
            return value if isinstance(value, str) else value.decode()
        return value if isinstance(value, bytes) else value.encode()

    async def exec(self, cmd: list[str], **kwargs) -> ExecResult[str]:
        self.execs.append(cmd)
        if cmd[0] == "rm":
            self.files.pop(cmd[-1], None)
        elif cmd[0] == "cp":
            self.files[cmd[2]] = self.files[cmd[1]]
        return ok()


class FakeChecker(FakeBox):
    """Fakes dev_eval.py: a solver file 'speedup=2.5' is valid with that speedup, 'bad' is invalid."""

    def __init__(self) -> None:
        super().__init__()
        self.running = self.max_running = 0
        self.ran: list[str] = []

    async def exec(self, cmd: list[str], **kwargs) -> ExecResult[str]:
        self.execs.append(cmd)
        solver = cmd[cmd.index(f"{devkit.DEV_DIR}/dev_eval.py") + 1]
        out = cmd[cmd.index("--json-out") + 1]
        source = (await self.read_file(solver)).strip()
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.ran.append(source)
        await anyio.sleep(0.05)
        self.running -= 1
        valid = source != "bad"
        speedup = float(source.split("=")[1]) if valid else None
        errors = [] if valid else ["instance 0: is_solution returned False"]
        self.files[out] = json.dumps({
            "valid": valid, "speedup": speedup, "n_invalid": len(errors), "errors": errors,
            "total_solver_s": 1.0, "total_reference_s": speedup or 1.0,
        })  # fmt: skip
        return ok(f"speedup: {speedup}x")


@pytest.fixture
def boxes(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeBox]:
    boxes = {"default": FakeBox(), devkit.CHECKER: FakeChecker()}
    monkeypatch.setattr(devkit, "sandbox", lambda name=None: boxes[name or "default"])
    return boxes


def test_checker_is_a_twin_of_the_agent_box_on_its_own_cpus() -> None:
    # Agent work must not share CPUs with the checker's timings, and the checker must time on as
    # many CPUs as final scoring, in the same image, with no network.
    default = ComposeService(image="hb__x", cpus=8.0, mem_limit="12288m", network_mode="none")
    default.__pydantic_extra__["x-local"] = True
    config = ComposeConfig(services={"default": default})
    add_checker(config)
    dumped = config.model_dump(mode="json", by_alias=True, exclude_none=True)["services"]
    assert dumped["default"]["cpuset"] == "0-7" and dumped["checker"]["cpuset"] == "8-15"
    assert {k: v for k, v in dumped["checker"].items() if k != "cpuset"} == {
        k: v for k, v in dumped["default"].items() if k != "cpuset"
    }
    with pytest.raises(ValueError):
        add_checker(ComposeConfig(services={"default": ComposeService(image="hb__x", cpus=12.0)}))


@pytest.mark.asyncio
async def test_agent_box_gets_no_evaluator(boxes, tmp_path: Path) -> None:
    # Agents time only in their own box; dev_eval.py lives in the checker, reached through the tool.
    (tmp_path / "evaluator.py").write_text("class Task: ...")
    state = SimpleNamespace(metadata={"tests_dir": str(tmp_path), "harbor_config": {"metadata": {"algotune_problem_size": 7}}})
    await devkit._install_toolkit(state)
    assert set(boxes["default"].files) == {"/app/dev/reference_task.py", "/app/dev/README.md"}
    assert set(boxes["checker"].files) == {"/app/dev/reference_task.py", "/app/dev/dev_eval.py", "/app/dev/config.json"}


@pytest.mark.asyncio
async def test_dev_eval_calls_run_one_at_a_time_in_call_order_and_report_their_wait(boxes) -> None:
    for i in range(3):
        boxes["default"].files[f"/app/agents/agent_{i}/try.py"] = f"speedup={i + 1}"
    state = SimpleNamespace(metadata={})
    tools_for = devkit.algotune_agent_tools(state)
    results: dict[str, str] = {}

    async def call(agent_id: str, delay: float) -> None:
        await anyio.sleep(delay)
        results[agent_id] = await tools_for(agent_id)[0](path=f"/app/agents/{agent_id}/try.py")

    async with anyio.create_task_group() as tg:
        for i in range(3):
            tg.start_soon(call, f"agent_{i}", 0.01 * i)

    checker = boxes["checker"]
    assert checker.max_running == 1
    assert checker.ran == ["speedup=1", "speedup=2", "speedup=3"]
    calls = state.metadata["checker"]["calls"]
    assert [c["agent_id"] for c in calls] == ["agent_0", "agent_1", "agent_2"]
    assert calls[0]["queue_wait_s"] < 0.04 < calls[2]["queue_wait_s"]
    assert [c["speedup"] for c in calls] == [1.0, 2.0, 3.0] and all(c["valid"] for c in calls)
    assert all(r.startswith("[dev_eval] waited ") for r in results.values())
    # each call copied only the named file, as solver.py, into its own fresh directory
    solvers = [cmd[cmd.index(f"{devkit.DEV_DIR}/dev_eval.py") + 1] for cmd in checker.execs]
    assert len(set(solvers)) == 3 and all(s.endswith("/solver.py") for s in solvers)
    assert boxes["default"].execs == []  # nothing ran in the agents' box


@pytest.mark.asyncio
async def test_finalize_times_in_the_checker_and_only_the_selected_solver_reaches_app_solver(boxes, tmp_path) -> None:
    (tmp_path / "evaluator.py").write_text("class Task: ...")
    meta = {"tests_dir": str(tmp_path), "harbor_config": {"metadata": {"algotune_problem_size": 7}}}
    agent_box = boxes["default"]
    agent_box.files.update({
        "/app/agents/agent_0/solver.py": "speedup=1.5",
        "/app/agents/agent_1/solver.py": "speedup=3.0",
        "/app/solver.py": "speedup=9.0",  # an agent wrote it directly; it is not a candidate
    })  # fmt: skip
    candidates = [Candidate(f"agent_{i}", "m", f"/app/agents/agent_{i}", "", float(i), "final_workspace") for i in range(2)]

    state = SimpleNamespace(metadata=dict(meta))
    await devkit.algotune_finalize(state, candidates)
    assert agent_box.files["/app/solver.py"] == "speedup=3.0"
    assert sorted(boxes["checker"].ran) == ["speedup=1.5", "speedup=3.0"]
    assert state.metadata["finalize"]["selected"]["agent_id"] == "agent_1"

    # nothing correct: an agent-written /app/solver.py must not be scored in place of the rule's choice
    agent_box.files.update({"/app/agents/agent_0/solver.py": "bad", "/app/agents/agent_1/solver.py": "bad"})
    agent_box.files["/app/solver.py"] = "speedup=9.0"
    state = SimpleNamespace(metadata=dict(meta))
    await devkit.algotune_finalize(state, candidates)
    assert "/app/solver.py" not in agent_box.files
    assert state.metadata["finalize"]["selected"] is None
