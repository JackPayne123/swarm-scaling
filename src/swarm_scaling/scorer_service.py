"""Remote scorer: every timed AlgoTune run (dev_eval, finalize evaluations, final scoring) on dedicated CPU slots.

Spec: scorer-api.md (2026-10-09); HARNESS.md "Remote scorer". Run on the scorer machine, from the repo root:

    SCORER_TOKEN=... scripts/scorer_service.sh --slots 2            # slot 0 = CPUs 0-7, slot 1 = 8-15

API (JSON over HTTP, bearer token on every request):
    GET  /health      -> {ok, slots, queue_len, cpu_model, version, seed_offset_id, tasks}
    POST /jobs        {kind: dev_eval|final_eval|score, task, solver_source, run_id, agent_id, sample_id,
                       n, seed, reps, size}  -> {job_id}
    GET  /jobs/<id>   -> {status: queued|running|done|error, queued_at, started_at, ended_at, queue_wait_s, run_s,
                          slot, cpu_model, result, output, error, kind, task}

One FIFO queue for all callers; each slot's worker takes the next job. Every job runs in a fresh container
(`docker run --rm`, the task's image, no network, the slot's cpuset, 8 CPUs, 8 GB), so nothing a solver starts
survives into the next job. Inputs (the solver, and for scoring the seeded verifier with the seed offset file) go in
through stdin as a tar, so the offset file is never mounted where a solver could read it after the verifier
deleted it. A job past its timeout has its own container killed by name (`docker kill scorer-<job id>`); nothing
else is ever killed. The seed offset and the seeded verifier copies exist only on this machine.
"""

from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import platform
import queue
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from swarm_scaling.algotune_devkit import ASSETS, FINAL_DEV_N, FINAL_DEV_SEED

KINDS = ("dev_eval", "final_eval", "score")
TIMEOUTS_S = {"dev_eval": 900, "final_eval": 900, "score": 3600}
OUTPUT_CHARS = 20_000  # verifier_stdout / dev_eval output kept per job (the end, where the summary is)
KILL_GRACE_S = 60  # after docker kill, how long to wait for the docker CLI to return


@dataclass
class TaskAssets:
    name: str  # AlgoTune task name, e.g. dst_type_II_scipy_fftpack
    slug: str  # Harbor sample id, e.g. algotune/dst-type-ii-scipy-fftpack
    image: str
    dev_dir: Path  # reference_task.py, dev_eval.py, thread_guard.py, config.json (mounted read-only)
    tests_dir: Path  # the seeded verifier copy, with the seed_offset file


@dataclass
class Job:
    id: str
    kind: str
    task: TaskAssets
    solver_source: str
    args: dict
    meta: dict
    status: str = "queued"
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    slot: int | None = None
    result: dict | None = None
    output: str = ""
    error: str | None = None

    def view(self, cpu_model: str) -> dict:
        return {
            "job_id": self.id, "kind": self.kind, "task": self.task.name, "status": self.status,
            "queued_at": self.queued_at, "started_at": self.started_at, "ended_at": self.ended_at,
            "queue_wait_s": None if self.started_at is None else round(self.started_at - self.queued_at, 3),
            "run_s": None if self.ended_at is None or self.started_at is None else round(self.ended_at - self.started_at, 3),
            "slot": self.slot, "cpu_model": cpu_model, "result": self.result, "output": self.output, "error": self.error,
        }  # fmt: skip


class JobError(Exception):
    """The job ends with status "error" and this message."""


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def job_inputs(job: Job) -> bytes:
    """The tar a job's container extracts at /: the solver, plus the seeded verifier for scoring."""
    if job.kind != "score":
        return _tar({"job/solver.py": job.solver_source.encode()})
    files = {"app/solver.py": job.solver_source.encode()}
    for path in sorted(job.task.tests_dir.rglob("*")):
        if path.is_file():
            files[f"tests/{path.relative_to(job.task.tests_dir)}"] = path.read_bytes()
    return _tar(files)


def job_script(job: Job) -> str:
    """The shell command the container runs. Outputs go to /out, a per-job directory on this host."""
    if job.kind == "score":
        # as Harbor's scorer runs it: test.sh writes reward 0 first, then pytest on the verifier
        return (
            "tar -x -C / && mkdir -p /logs/verifier && { bash /tests/test.sh; rc=$?; "
            "cp /logs/verifier/reward.txt /out/reward.txt 2>/dev/null; exit $rc; }"
        )
    a = job.args
    argv = ["--n", a["n"], "--seed", a["seed"], "--reps", a["reps"]] + (["--size", a["size"]] if a.get("size") is not None else [])
    return "tar -x -C / && python /app/dev/dev_eval.py /job/solver.py " + " ".join(str(x) for x in argv) + " --json-out /out/result.json"


def parse_score(stdout: str, reward_text: str | None) -> dict:
    validity = re.search(r"^Validity: (True|False)$", stdout, re.M)
    ratio = re.search(r"^First-call ratio: (\S+)$", stdout, re.M)
    if reward_text is None:
        raise JobError("the verifier wrote no reward.txt")
    return {
        "score": float(reward_text.strip()),
        "valid": None if validity is None else validity.group(1) == "True",
        "verifier_stdout": stdout[-OUTPUT_CHARS:],
        "first_call_ratio": float(ratio.group(1)) if ratio and ratio.group(1) != "None" else None,
        "scoring_timeout": False,
    }


def docker_runner(docker: str = "docker", memory: str = "8g", cpus: int = 8, work_dir: Path | None = None):
    """Returns run(job, cpuset, timeout_s) -> (result, output) running the job in a fresh container."""

    def run(job: Job, cpuset: str, timeout_s: float) -> tuple[dict | None, str]:
        name = f"scorer-{job.id}"
        with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            out.chmod(0o777)  # the container writes as root
            cmd = [
                docker, "run", "--rm", "-i", "--name", name, "--network", "none", "--cpuset-cpus", cpuset,
                "--cpus", str(cpus), "--memory", memory, "-w", "/app", "-v", f"{out}:/out",
                # the toolkit, as in the local checker (solvers may load /app/dev/reference_task.py); no secrets
                "-v", f"{job.task.dev_dir}:/app/dev:ro",
            ]  # fmt: skip
            if job.kind == "score":
                cmd += ["-e", "TEST_DIR=/tests"]
            cmd += [job.task.image, "sh", "-c", job_script(job)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timed_out = False
            try:
                stdout, stderr = proc.communicate(job_inputs(job), timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                # only this job's own container, by its unique name; the docker CLI then returns by itself
                subprocess.run([docker, "kill", name], capture_output=True, timeout=KILL_GRACE_S)
                try:
                    stdout, stderr = proc.communicate(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    raise JobError(f"timed out after {timeout_s:g}s and the container did not stop") from None
            stdout_s, stderr_s = stdout.decode(errors="replace"), stderr.decode(errors="replace")
            output = (stderr_s + stdout_s).strip()[-OUTPUT_CHARS:]
            if job.kind == "score":
                if timed_out:  # the harness's timeout semantics: no speedup, flagged
                    return {"score": 1.0, "valid": None, "verifier_stdout": (stdout_s + stderr_s)[-OUTPUT_CHARS:],
                            "first_call_ratio": None, "scoring_timeout": True}, output  # fmt: skip
                reward = out / "reward.txt"
                return parse_score(stdout_s + stderr_s, reward.read_text() if reward.exists() else None), output
            if timed_out:
                raise JobError(f"timed out after {timeout_s:g}s")
            result = out / "result.json"
            if not result.exists():  # the process died before writing a result (crash, OOM kill)
                raise JobError(f"no result, exit {proc.returncode}: {stderr_s[-500:]}")
            return json.loads(result.read_text()), output

    return run


class Scorer:
    """The job store, the FIFO queue and one worker thread per slot."""

    def __init__(
        self,
        tasks: dict[str, TaskAssets],
        cpusets: list[str],
        run: Callable[[Job, str, float], tuple[dict | None, str]],
        cpu_model: str,
        version: str,
        seed_offset_id: str,
        timeouts_s: dict[str, float] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.tasks = {**tasks, **{a.slug: a for a in tasks.values()}}  # accept the task name or the sample id
        self.cpusets, self.run = cpusets, run
        self.cpu_model, self.version, self.seed_offset_id = cpu_model, version, seed_offset_id
        self.timeouts_s = timeouts_s or TIMEOUTS_S
        self.log_path = log_path
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()  # FIFO across every caller
        self.lock = threading.Lock()
        self.workers = [threading.Thread(target=self._work, args=(slot,), daemon=True) for slot in range(len(cpusets))]
        for w in self.workers:
            w.start()

    def health(self) -> dict:
        return {
            "ok": True, "slots": len(self.cpusets), "queue_len": self.queue.qsize(), "cpu_model": self.cpu_model,
            "version": self.version, "seed_offset_id": self.seed_offset_id,
            "tasks": sorted({a.name for a in self.tasks.values()}),
        }  # fmt: skip

    def submit(self, body: dict) -> str:
        kind = body.get("kind")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        task = self.tasks.get(body.get("task"))
        if task is None:
            raise ValueError(f"unknown task {body.get('task')!r}")
        if not isinstance(body.get("solver_source"), str):
            raise ValueError("solver_source must be the solver.py text")
        args = {}
        if kind != "score":
            final = kind == "final_eval"  # finalize's constants unless given
            args = {
                "n": body.get("n", FINAL_DEV_N if final else 20),
                "seed": body.get("seed", FINAL_DEV_SEED if final else 0),
                "reps": body.get("reps", 10),
                "size": body.get("size"),
            }
            if not all(isinstance(v, int) for v in args.values() if v is not None) or args["seed"] < 0 or min(args["n"], args["reps"]) < 1:
                raise ValueError("n and reps must be integers >= 1, seed an integer >= 0, size an integer or null")
        meta = {k: str(body.get(k, "")) for k in ("run_id", "agent_id", "sample_id")}
        job = Job(uuid.uuid4().hex[:16], kind, task, body["solver_source"], args, meta)
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job.id

    def view(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else job.view(self.cpu_model)

    def _work(self, slot: int) -> None:
        while True:
            job = self.queue.get()
            with self.lock:
                job.status, job.slot, job.started_at = "running", slot, time.time()
            try:
                result, output = self.run(job, self.cpusets[slot], self.timeouts_s[job.kind])
                status, error = "done", None
            except JobError as ex:
                result, output, status, error = None, "", "error", str(ex)
            except Exception as ex:  # an infrastructure failure (docker missing, disk full): report it on the job
                result, output, status, error = None, "", "error", f"{type(ex).__name__}: {ex}"
            with self.lock:
                job.result, job.output, job.error, job.status, job.ended_at = result, output, error, status, time.time()
            if self.log_path is not None:
                with self.lock, self.log_path.open("a") as fh:
                    fh.write(json.dumps({**job.view(self.cpu_model), **job.meta, "output": None}) + "\n")


def handler(scorer: Scorer, token: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            given = self.headers.get("Authorization", "")
            if hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                return True
            self._send(401, {"error": "missing or wrong bearer token"})
            return False

        def do_GET(self) -> None:
            if not self._authorised():
                return
            if self.path == "/health":
                return self._send(200, scorer.health())
            if self.path.startswith("/jobs/"):
                view = scorer.view(self.path[len("/jobs/") :])
                return self._send(200, view) if view else self._send(404, {"error": "no such job"})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorised():
                return
            if self.path != "/jobs":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                job_id = scorer.submit(body)
            except (ValueError, json.JSONDecodeError) as ex:
                return self._send(400, {"error": str(ex)})
            self._send(200, {"job_id": job_id})

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (BaseHTTPRequestHandler's signature)
            pass  # jobs are logged to jobs.jsonl; per-request lines would be every 2 s poll

    return Handler


def cpu_model() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        found = re.search(r"^Model name:\s*(.+)$", out, re.M)
        if found:
            return found.group(1).strip()
    except FileNotFoundError:
        pass
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return platform.processor() or "unknown"


def git_version(root: Path) -> str:
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def prepare_tasks(names: list[str], offset: int, work_dir: Path) -> dict[str, TaskAssets]:
    """Seeded verifier copies and dev_eval assets for each task, from the Harbor task cache (as tasks.py does)."""
    from inspect_harbor import algotune

    from swarm_scaling.tasks import ALGOTUNE_REF, _harbor_name, seed_verifier

    base = algotune(ref=ALGOTUNE_REF, dataset_task_names=[_harbor_name(n) for n in names])
    by_slug = {_harbor_name(n): n for n in names}
    tasks = {}
    for sample in base.dataset:
        name = by_slug[str(sample.id)]
        evaluator = (Path(sample.metadata["tests_dir"]) / "evaluator.py").read_text()  # before seeding repoints it
        dev = work_dir / "assets" / name / "dev"
        dev.mkdir(parents=True, exist_ok=True)
        (dev / "reference_task.py").write_text(evaluator)
        for asset in ("dev_eval.py", "thread_guard.py"):
            (dev / asset).write_text((ASSETS / asset).read_text())
        size = sample.metadata["harbor_config"]["metadata"]["algotune_problem_size"]
        (dev / "config.json").write_text(json.dumps({"problem_size": size}))
        image = sample.sandbox.config.services["default"].image
        tasks[name] = TaskAssets(name, str(sample.id), image, dev, seed_verifier(sample, offset))
    return tasks


def main() -> None:
    from swarm_scaling.tasks import PROJECT_ROOT, SEED_OFFSET_ENV, SEED_OFFSET_FILE, SPLIT_PATH, _seed_offset, seed_offset_id

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--first-cpu", type=int, default=0, help="slot k uses CPUs first + 8k .. first + 8k + 7")
    p.add_argument("--memory", default="8g")
    p.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "data" / ".scorer")
    args = p.parse_args()

    token = os.environ.get("SCORER_TOKEN")
    if not token:
        raise SystemExit("set SCORER_TOKEN")
    # A missing offset file would silently create a new offset and score on different instances than every
    # other run of the experiment: copy data/.algotune_seed_offset here, or set ALGOTUNE_SEED_OFFSET.
    if not os.environ.get(SEED_OFFSET_ENV) and not SEED_OFFSET_FILE.exists():
        raise SystemExit(f"no seed offset: copy the experiment's {SEED_OFFSET_FILE.name} to {SEED_OFFSET_FILE} or set {SEED_OFFSET_ENV}")
    offset = _seed_offset()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text())
    tasks = prepare_tasks([n for names in split.values() for n in names], offset, args.work_dir)
    cpusets = [f"{args.first_cpu + 8 * k}-{args.first_cpu + 8 * k + 7}" for k in range(args.slots)]
    scorer = Scorer(
        tasks, cpusets, docker_runner(memory=args.memory, work_dir=args.work_dir), cpu_model(), git_version(PROJECT_ROOT),
        seed_offset_id(offset), log_path=args.work_dir / "jobs.jsonl",
    )  # fmt: skip
    server = ThreadingHTTPServer((args.host, args.port), handler(scorer, token))
    print(f"scorer on {args.host}:{args.port}: slots {cpusets}, {len(tasks)} tasks, cpu {scorer.cpu_model}, "
          f"version {scorer.version}, seed offset id {scorer.seed_offset_id}", flush=True)  # fmt: skip
    server.serve_forever()


if __name__ == "__main__":
    main()
