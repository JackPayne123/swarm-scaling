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
from swarm_scaling.algotune_devkit import TIMING_ENV
from swarm_scaling.scorer_service import (
    AloneCache, Job, JobError, Scorer, TaskAssets, docker_alone_measure, docker_runner, handler, image_digest_with_retry,
    job_inputs, job_script, parser, slot_cpus,
)
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


@pytest.mark.asyncio
async def test_stats_report_queue_waits_and_per_slot_busy_time(serve) -> None:
    # Pilot 4: a team of 2 waited 12.4 min over 8 checks on 2 slots shared by 16 agents; the service must show it.
    gate = threading.Event()

    def run(job: Job, cpuset: str, timeout_s: float):
        gate.wait(5)
        return {"valid": True}, ""

    scorer = serve(run, slots=2)
    ids = [scorer_client.submit(job(agent_id=f"a{i}")) for i in range(3)]  # 2 run at once, 1 waits
    stats = scorer_client.request("GET", "/stats")
    assert stats["running"] == 2 and stats["queue_len"] == 1 and stats["jobs_done"] == 0
    gate.set()
    for i in ids:
        while scorer_client.get(i)["status"] != "done":
            pass
    stats = scorer_client.request("GET", "/stats")
    waits = sorted(scorer.jobs[i].started_at - scorer.jobs[i].queued_at for i in ids)
    assert stats["jobs_done"] == 3 and stats["queue_len"] == 0 and stats["running"] == 0
    assert stats["queue_wait_p95_s"] == round(waits[-1], 3)
    assert stats["queue_wait_mean_s"] == pytest.approx(sum(waits) / 3, abs=1e-3)
    assert len(stats["slot_busy_frac"]) == 2 and all(0 < b <= 1 for b in stats["slot_busy_frac"])


def test_slots_never_overlap_the_services_cpus() -> None:
    # Each slot times on its own 8 CPUs; the service, Docker and the OS keep at least 8 others.
    cpusets, service = slot_cpus(4, 0, 64)  # c7a.16xlarge
    assert cpusets == ["0-7", "8-15", "16-23", "24-31"] and service == set(range(32, 64))
    assert slot_cpus(1, 8, 16) == (["8-15"], set(range(8)))  # the Mac's Docker, slot above the agent box
    for slots, first, host in ((4, 0, 32), (4, 0, 36), (2, 16, 24)):  # too few CPUs left, or past the host's
        with pytest.raises(ValueError):
            slot_cpus(slots, first, host)


def test_the_alone_baseline_is_measured_once_per_instance_and_sent_into_later_jobs(tmp_path) -> None:
    # Only the reference-alone timing is cached; each job still times reference and solver interleaved itself.
    import io
    import tarfile
    import time

    task = assets(tmp_path)
    task.dev_dir.mkdir()
    (task.dev_dir / "config.json").write_text('{"problem_size": 9}')
    (task.tests_dir / "test_outputs.py").write_text("PROBLEM_SIZE = 7\nNUM_TEST_INSTANCES = 4\nNUM_REPEATS = 10\n")
    measured, sent = [], []

    def measure(job, seeds, size, reps, cpuset):
        measured.append((seeds, size, reps, cpuset))
        return [1000 + s for s in seeds]

    def run(job, cpuset, timeout_s):
        with tarfile.open(fileobj=io.BytesIO(job_inputs(job))) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        sent.append((job.alone_baseline, files, job_script(job)))
        return {"valid": True}, ""

    def scorer(cache: bool, measure=measure) -> Scorer:
        alone = AloneCache(tmp_path / "alone", measure, "sha256:img", "Test CPU") if cache else None
        return Scorer({"cvar_projection": task}, ["8-15"], run, "Test CPU", "v", "id", alone_cache=alone)

    def do(sc: Scorer, kind="dev_eval", **args) -> tuple:
        job_id = sc.submit({**job(kind=kind), **args})
        while sc.view(job_id)["status"] not in ("done", "error"):
            time.sleep(0.01)
        return sent[-1]

    sc = scorer(cache=True)
    info, files, script = do(sc, n=3, reps=2)
    assert measured == [([10_000, 10_001, 10_002], 9, 2, "8-15")] and info["mode"] == "cached" and info["measured_now"] == 3
    assert json.loads(files["job/alone.json"]) == {"size": 9, "reps": 2, "ns": {"10000": 11_000, "10001": 11_001, "10002": 11_002}}
    assert "--alone-cache /job/alone.json" in script
    info, files, _ = do(sc, n=5, reps=2)  # only the two new instances are measured
    assert measured[-1][0] == [10_003, 10_004] and len(json.loads(files["job/alone.json"])["ns"]) == 5
    info, files, _ = do(scorer(cache=True), kind="score")  # a restarted service keeps the dev cache; scoring: offset + i
    assert measured[-1] == ([1_234_567 + i for i in range(4)], 7, 10, "8-15")
    assert json.loads(files["tests/alone_cache.json"])["reps"] == 10 and "job/alone.json" not in files
    info, _, _ = do(scorer(cache=True), n=3, reps=2)
    assert len(measured) == 3 and info["measured_now"] == 0 and info["age_s"] >= 0  # nothing re-measured

    info, files, script = do(scorer(cache=False), n=3, reps=2)  # --no-cache-alone: every job times it alone itself
    assert info == {"mode": "child"} and "job/alone.json" not in files and "--alone-cache" not in script

    def broken(job, seeds, size, reps, cpuset):
        raise JobError("alone baseline timed out")

    info, files, _ = do(scorer(cache=True, measure=broken), n=1, seed=99, reps=2)  # a failed measurement: fall back
    assert info["mode"] == "child" and "timed out" in info["cache_error"] and "job/alone.json" not in files


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
    assert first[:4] == ["run", "--rm", "-i", "--init"] and first[first.index("--network") + 1] == "none"
    # cpuset only, no --cpus quota (it throttled ~1% of a fully busy job's periods on an 8-CPU cpuset)
    assert first[first.index("--cpuset-cpus") + 1] == "8-15" and "--cpus" not in first
    assert first[first.index("--memory") + 1] == "8g"
    # glibc heap trimming off in the process environment of the verifier (score) and of dev_eval
    runs = [c for c in calls if c[0] == "run"]
    for argv in runs:
        assert {argv[i + 1] for i, a in enumerate(argv) if a == "-e"} >= {f"{k}={v}" for k, v in TIMING_ENV.items()}
    assert "dev_eval.py" in runs[1][-1] and "test.sh" in runs[0][-1]
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
                  "total_reference_s": speedup, "thread_check": None, "first_call_ratio": 1.2, "reference_inflation": 1.0}  # fmt: skip
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


def test_startup_retries_an_intermittent_image_inspect_failure_and_gives_up_after_the_last_try() -> None:
    # Docker Desktop sometimes answers "No such image" for a listed image; the next call a few seconds later works.
    calls = []

    def flaky(image):
        calls.append(image)
        if len(calls) < 3:
            raise RuntimeError(f"docker image inspect {image} failed: No such image")
        return "sha256:abc"

    assert image_digest_with_retry("hb__x", flaky, wait_s=0) == "sha256:abc" and len(calls) == 3
    def missing(image):
        raise RuntimeError("No such image")

    with pytest.raises(RuntimeError, match="No such image"):
        image_digest_with_retry("hb__x", missing, tries=2, wait_s=0)


def test_the_alone_measurement_container_gets_the_allocator_settings_and_no_cpu_quota(tmp_path) -> None:
    # The cached alone baseline must be timed under the same allocator and CPU conditions as the jobs it is compared to.
    docker = tmp_path / "docker"
    docker.write_text(f"""#!{sys.executable}
import json, sys
open({str(tmp_path / "argv.json")!r}, "w").write(json.dumps(sys.argv[1:]))
for line in sys.stdin:
    print("ALONE 1000")
""")
    docker.chmod(0o755)
    task = assets(tmp_path)
    measure = docker_alone_measure(docker=str(docker))
    assert measure(Job("j", "dev_eval", task, "", {}, {}), [1, 2], 9, 3, "8-15") == [1000, 1000]
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--cpus" not in argv and argv[argv.index("--cpuset-cpus") + 1] == "8-15"
    assert {argv[i + 1] for i, a in enumerate(argv) if a == "-e"} == {f"{k}={v}" for k, v in TIMING_ENV.items()}


def test_the_alone_baseline_cache_is_off_unless_asked_for() -> None:
    # On BLAS-bound tasks an idle-measured cached baseline read ~0.09 higher than the busy-slot interleaved timing.
    assert parser().parse_args([]).cache_alone is False
    assert parser().parse_args(["--cache-alone"]).cache_alone is True
