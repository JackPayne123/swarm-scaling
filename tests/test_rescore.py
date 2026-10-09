"""scripts/rescore.py: the serial final-score pass over logged selected solvers. Fakes only, no Docker."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target
from inspect_ai.util import ExecResult
from inspect_ai.util._sandbox.context import sandbox_default_context_var, sandbox_environments_context_var

from swarm_scaling.algotune_devkit import CHECKER, SOLVER_PATH
from swarm_scaling.tasks import algotune_scorer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rescore", ROOT / "scripts/rescore.py")
rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore)


def fake_log(metadata: dict, samples: list[tuple[str, int, str | None, float]]):
    return SimpleNamespace(
        eval=SimpleNamespace(metadata=metadata),
        samples=[
            SimpleNamespace(
                id=task, epoch=epoch, metadata={"finalize": {"selected_source": source}},
                scores={"algotune_scorer": SimpleNamespace(value=score)},
            )  # fmt: skip
            for task, epoch, source, score in samples
        ],
    )


def test_rows_carry_the_run_and_unselected_samples_are_skipped(monkeypatch):
    # A run where no candidate was correct has nothing to re-time; it must be reported, not silently dropped.
    logs = {
        "logs/pilot3-team4-x/a.eval": fake_log(
            {"arm": "team", "n": 4, "split": "pilot", "algotune_seed_offset": 7},
            [("algotune/x", 1, "class Solver: ...", 3.5), ("algotune/x", 2, None, 0.0)],
        )
    }
    monkeypatch.setattr(rescore, "read_eval_log", lambda path: logs[path])
    rows, skipped, offsets = rescore.collect([Path(p) for p in logs])
    assert rows == [{
        "log": "logs/pilot3-team4-x/a.eval", "run": "pilot3-team4-x", "arm": "team", "n": 4, "split": "pilot",
        "task": "algotune/x", "epoch": 1, "original_score": 3.5, "source": "class Solver: ...",
    }]  # fmt: skip
    assert [(r["task"], r["epoch"]) for r in skipped] == [("algotune/x", 2)]
    assert offsets == {7}


def test_rescore_samples_are_the_tasks_own_and_other_offsets_are_refused():
    # Rescores are only comparable to the original scores if they use the same verifier instances (seed offset).
    base = Sample(input="i", id="algotune/x", metadata={"tests_dir": "/v/x"})
    task_for = lambda split: SimpleNamespace(metadata={"algotune_seed_offset": 7}, dataset=[base])  # noqa: E731
    rows = [{"split": "pilot", "task": "algotune/x", "run": f"r{i}", "epoch": 1, "source": "s"} for i in range(2)]
    _, samples = rescore.build(rows, {7}, task_for)
    assert len({s.id for s in samples}) == 2
    assert all(s.metadata["tests_dir"] == "/v/x" and s.metadata["rescore"]["source"] == "s" for s in samples)
    assert "rescore" not in base.metadata  # copies, so one task sample can be rescored for several runs
    with pytest.raises(ValueError):
        rescore.build(rows, {7, 8}, task_for)


class Box:
    def __init__(self, files: dict | None = None) -> None:
        self.files = dict(files or {})

    async def write_file(self, path, contents) -> None:
        self.files[path] = contents if isinstance(contents, bytes) else contents.encode()

    async def read_file(self, path, text: bool = True):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path].decode() if text else self.files[path]

    async def exec(self, cmd, **kwargs) -> ExecResult[str]:
        if cmd[0] == "rm":
            self.files.pop(cmd[-1], None)
        return ExecResult(success=True, returncode=0, stdout="", stderr="")


class Verifier(Box):
    async def exec(self, cmd, **kwargs) -> ExecResult[str]:
        if cmd[:2] == ["sh", "-c"] and "/tests/test.sh" in cmd[2]:
            speedup = self.files[SOLVER_PATH].decode().split("=")[1]
            self.files["/logs/verifier/reward.txt"] = speedup.encode()
            return ExecResult(success=True, returncode=0, stdout=f"Validity: True\nFinal Reward (Score): {speedup}\n", stderr="")
        return await super().exec(cmd, **kwargs)


@pytest.mark.asyncio
async def test_the_logged_solver_is_timed_by_the_checkers_verifier(tmp_path):
    # The rescore must time exactly the solver finalize selected, through the same scoring path as a real run.
    (tmp_path / "test.sh").write_text("#!/bin/bash\n")
    agent_box, checker = Box(), Verifier()
    sandbox_environments_context_var.set({"default": agent_box, CHECKER: checker})
    sandbox_default_context_var.set("default")
    row = {"run": "pilot3-team2-x", "task": "algotune/x", "epoch": 1, "original_score": 3.0, "source": "speedup=2.75"}
    state = SimpleNamespace(metadata={
        "tests_dir": str(tmp_path), "test_path": str(tmp_path / "test.sh"), "verifier_timeout_sec": 60, "rescore": row,
    })  # fmt: skip

    await rescore.install_selected()(state, None)
    score = await algotune_scorer()(state, Target(""))
    out = rescore.result_row(SimpleNamespace(metadata=state.metadata, scores={"algotune_scorer": score}, epoch=2, error=None))

    assert checker.files[SOLVER_PATH] == b"speedup=2.75"
    assert out["rescore"] == 2.75 and out["valid"] is True and out["repeat"] == 2
    assert out["original_score"] == 3.0 and "source" not in out
    assert "Final Reward (Score): 2.75" in out["verifier_output"]


def test_refuses_to_run_while_other_containers_run(monkeypatch):
    # Another eval's containers would share CPUs with the rescore's timings.
    monkeypatch.setattr(rescore.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="abc123\n"))
    monkeypatch.setattr(rescore, "collect", lambda files: pytest.fail("read logs while Docker was busy"))
    monkeypatch.setattr("sys.argv", ["rescore.py", "logs/pilot3-*"])
    with pytest.raises(SystemExit, match="running"):
        rescore.main()
