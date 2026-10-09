"""The remote scorer service and the harness's remote checker backend. Real HTTP on localhost, fake job runners
and a fake `docker` executable: no containers."""

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Target
from inspect_ai.util._sandbox.context import sandbox_default_context_var, sandbox_environments_context_var

from swarm_scaling import algotune_devkit as devkit
from swarm_scaling import scorer_client
from swarm_scaling import swarm as swarm_module
from swarm_scaling.scorer_service import Job, JobError, Scorer, TaskAssets, docker_runner, handler
from swarm_scaling.swarm import Candidate
from swarm_scaling.tasks import algotune_scorer

TOKEN = "test-token"


def assets(tmp_path: Path) -> TaskAssets:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")
    (tests / "seed_offset").write_text("1234567\n")
    return TaskAssets("cvar_projection", "algotune/cvar-projection", "hb__img", tmp_path / "dev", tests)


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Start a Scorer with the given fake runner behind the real HTTP handler; point the client at it."""
    servers = []

    def start(run, slots: int = 1) -> Scorer:
        scorer = Scorer({"cvar_projection": assets(tmp_path)}, [f"{8 * k}-{8 * k + 7}" for k in range(slots)], run,
                        "Test CPU", "abc123", "offsetid")  # fmt: skip
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(scorer, TOKEN))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        monkeypatch.setenv("SCORER_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("SCORER_TOKEN", TOKEN)
        monkeypatch.setattr(scorer_client, "POLL_S", 0.01)
        return scorer

    yield start
    for server in servers:
        server.shutdown()


def job(source: str = "s", agent_id: str = "", kind: str = "dev_eval") -> dict:
    return {"kind": kind, "task": "cvar_projection", "solver_source": source, "run_id": "r", "agent_id": agent_id, "sample_id": "x"}


@pytest.mark.asyncio
async def test_jobs_from_every_caller_run_in_one_fifo_order(serve) -> None:
    # Timings are only comparable if no caller can jump the queue: one global FIFO, whoever submits.
    gate, ran = threading.Event(), []

    def run(job: Job, cpuset: str, timeout_s: float):
        gate.wait(5)  # hold the slot until every job is queued
        ran.append(job.meta["agent_id"])
        return {"valid": True}, ""

    serve(run)
    order = ["a-1", "b-1", "a-2", "c-1", "b-2"]  # three callers, interleaved
    ids = [scorer_client.submit(job(agent_id=a)) for a in order]
    gate.set()
    last = await scorer_client.run_job(job(agent_id="last"))
    assert ran == order + ["last"]
    views = [scorer_client.get(i) for i in ids]
    assert all(v["status"] == "done" and v["slot"] == 0 and v["cpu_model"] == "Test CPU" for v in views)
    assert [v["started_at"] for v in views] == sorted(v["started_at"] for v in views)
    assert last["status"] == "done" and last["queue_wait_s"] >= 0


def test_requests_without_the_token_are_rejected(serve, monkeypatch) -> None:
    # The scorer holds the seed offset and runs arbitrary solver code: nothing without the bearer token.
    scorer = serve(lambda job, cpuset, timeout_s: pytest.fail("ran a job without auth"))
    monkeypatch.setenv("SCORER_TOKEN", "wrong")
    for call in (scorer_client.health, lambda: scorer_client.submit(job())):
        with pytest.raises(scorer_client.ScorerError, match="HTTP 401"):
            call()
    assert scorer.jobs == {}
    monkeypatch.setenv("SCORER_TOKEN", TOKEN)
    assert scorer_client.health()["seed_offset_id"] == "offsetid"


def fake_docker(tmp_path: Path) -> Path:
    """A `docker` that records its argv; `run` blocks until `kill` of the same container name."""
    script = tmp_path / "docker"
    script.write_text(f"""#!{sys.executable}
import json, pathlib, sys, time
log = pathlib.Path({str(tmp_path / "docker.log")!r})
with log.open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "kill":
    pathlib.Path({str(tmp_path)!r}, "killed-" + sys.argv[2]).touch()
    sys.exit(0)
name = sys.argv[sys.argv.index("--name") + 1]
sys.stdin.buffer.read()
while not pathlib.Path({str(tmp_path)!r}, "killed-" + name).exists():
    time.sleep(0.05)
print("partial verifier output")
sys.exit(137)
""")
    script.chmod(0o755)
    return script


def test_a_job_past_its_timeout_has_only_its_own_container_killed(tmp_path) -> None:
    # Scoring keeps the harness's timeout semantics (1.0, flagged); dev_eval reports a timeout error.
    run = docker_runner(docker=str(fake_docker(tmp_path)), work_dir=tmp_path)
    task = assets(tmp_path)
    score = run(Job("j1", "score", task, "class Solver: ...", {}, {}), "8-15", 0.5)
    assert score[0]["score"] == 1.0 and score[0]["scoring_timeout"] is True
    with pytest.raises(JobError, match="timed out after 0.5s"):
        run(Job("j2", "dev_eval", task, "x", {"n": 1, "seed": 0, "reps": 1, "size": None}, {}), "0-7", 0.5)

    calls = [json.loads(line) for line in (tmp_path / "docker.log").read_text().splitlines()]
    kills = [c for c in calls if c[0] == "kill"]
    assert kills == [["kill", "scorer-j1"], ["kill", "scorer-j2"]]
    first = calls[0]
    assert first[:3] == ["run", "--rm", "-i"] and first[first.index("--network") + 1] == "none"
    assert first[first.index("--cpuset-cpus") + 1] == "8-15" and first[first.index("--cpus") + 1] == "8"
    assert first[first.index("--memory") + 1] == "8g"
    # the seeded verifier (with the offset file) goes in through stdin, never as a mount a solver could read
    assert not any(str(task.tests_dir) in arg for arg in first)


@pytest.mark.asyncio
async def test_remote_backend_sends_dev_eval_finalize_and_scoring_to_the_service(serve, monkeypatch, tmp_path) -> None:
    # With checker_backend="remote" no checker container exists: every timed run must be a scorer job.
    def run(job: Job, cpuset: str, timeout_s: float):
        speedup = float(job.solver_source.split("=")[1])
        if job.kind == "score":
            return {"score": speedup, "valid": True, "verifier_stdout": f"Validity: True\nFinal Reward (Score): {speedup}",
                    "first_call_ratio": 1.1, "scoring_timeout": False}, ""  # fmt: skip
        result = {"valid": True, "speedup": speedup, "n_invalid": 0, "errors": [], "total_solver_s": 1.0,
                  "total_reference_s": speedup, "thread_check": None, "first_call_ratio": 1.2}  # fmt: skip
        return result, f"speedup: {speedup}x"

    scorer = serve(run, slots=2)
    agent_box = devkit_box({"/app/agents/agent_0/solver.py": b"speedup=1.5", "/app/agents/agent_1/solver.py": b"speedup=3.0"})
    boxes = {"default": agent_box}  # no "checker": any local checker access raises KeyError
    monkeypatch.setattr(devkit, "sandbox", lambda name=None: boxes[name or "default"])
    monkeypatch.setattr(swarm_module, "sandbox", lambda name=None: boxes[name or "default"])
    sandbox_environments_context_var.set(boxes)
    sandbox_default_context_var.set("default")
    (tmp_path / "evaluator.py").write_text("class Task: ...")
    state = SimpleNamespace(sample_id="algotune/cvar-projection", epoch=1, metadata={
        "checker_backend": "remote", "algotune_task_name": "cvar_projection", "tests_dir": str(tmp_path),
        "harbor_config": {"metadata": {"algotune_problem_size": 9}},
    })  # fmt: skip

    report = await devkit.algotune_agent_tools(state)("agent_0")[0](path="/app/agents/agent_0/solver.py", n=3)
    assert "speedup: 1.5x" in report
    (call,) = state.metadata["checker"]["calls"]
    assert call["slot"] in (0, 1) and call["scorer_host"] == "127.0.0.1" and call["cpu_model"] == "Test CPU"
    assert call["speedup"] == 1.5 and call["first_call_ratio"] == 1.2

    candidates = [Candidate(f"agent_{i}", "m", f"/app/agents/agent_{i}", "", float(i), "final_workspace") for i in range(2)]
    await devkit.algotune_finalize(state, candidates)
    assert agent_box.files[devkit.SOLVER_PATH] == b"speedup=3.0"
    assert all(c["slot"] in (0, 1) for c in state.metadata["finalize"]["candidates"])

    score = await algotune_scorer()(state, Target(""))
    assert score.value == 3.0 and score.metadata["first_call_ratio"] == 1.1 and score.metadata["scoring_timeout"] is False
    assert score.metadata["checker"]["cpu_model"] == "Test CPU"
    kinds = [j.kind for j in sorted(scorer.jobs.values(), key=lambda j: j.queued_at)]
    assert kinds == ["dev_eval", "final_eval", "final_eval", "score"]
    assert all(j.task.name == "cvar_projection" for j in scorer.jobs.values())


def devkit_box(files: dict[str, bytes]):
    class Box:
        def __init__(self) -> None:
            self.files = dict(files)

        async def write_file(self, path, contents) -> None:
            self.files[path] = contents if isinstance(contents, bytes) else contents.encode()

        async def read_file(self, path, text: bool = True):
            if path not in self.files:
                raise FileNotFoundError(path)
            return self.files[path].decode() if text else self.files[path]

        async def exec(self, cmd, **kwargs):
            if cmd[0] == "rm":
                self.files.pop(cmd[-1], None)
            elif cmd[0] == "cp":
                self.files[cmd[2]] = self.files[cmd[1]]
            return SimpleNamespace(success=True, returncode=0, stdout="", stderr="")

    return Box()
