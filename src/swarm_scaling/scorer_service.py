"""Remote scorer: every timed AlgoTune run (dev_eval, finalize evaluations, final scoring) on dedicated CPU slots.

Spec: scorer-api.md (2026-10-09); HARNESS.md "Remote scorer". Run on the scorer machine, from the repo root:

    SCORER_TOKEN=... scripts/scorer_service.sh --slots 2            # slot 0 = CPUs 0-7, slot 1 = 8-15

Slots must fit in Docker's CPUs with at least SERVICE_CPUS left over; on Linux the service pins itself to the CPUs
outside every slot, so its threads never run on a slot's CPUs.

API (JSON over HTTP, bearer token on every request):
    GET  /health      -> {ok, slots, queue_len, cpu_model, version, seed_offset_id, tasks}
    GET  /stats       -> {queue_len, running, jobs_done, queue_wait_mean_s, queue_wait_p95_s, slot_busy_frac, ...}
    POST /jobs        {kind: dev_eval|final_eval|score, task, solver_source, run_id, agent_id, sample_id,
                       n, seed, reps, size}  -> {job_id}
    GET  /jobs/<id>   -> {status: queued|running|done|error, queued_at, started_at, ended_at, queue_wait_s, run_s,
                          slot, cpu_model, result, output, error, kind, task}

Alone baseline (--cache-alone, default off since 2026-10-10): the reference check compares the reference timed interleaved with the
solver against the reference timed alone on the same instances. Instead of re-timing it alone in every job, the
service times it once per (task, instance seeds, repeats, size, task image, CPU model), in a fresh container with no
solver on the slot of the first job that needs it, keeps it in <work dir>/alone_cache/, and sends it into each job,
whose dev_eval.py / verifier copy reads it (thread_guard.CachedAlone) and only times interleaved. The speedup timing
is unchanged. Each job records `alone_baseline` (cached + age, or child). Off by default: on BLAS-bound tasks four
busy slots slow the reference 7-10%, so an idle-measured cached baseline read about 0.09 higher under load (scoring
investigation); without the cache each job times it alone itself, alternating with the interleaved timing.

One FIFO queue for all callers; each slot's worker takes the next job. Every QUEUE_SAMPLE_S the queue length and
running job count are appended to <work dir>/queue.jsonl. Every job runs in a fresh container
(`docker run --rm`, the task's image, no network, the slot's cpuset and no CPU quota, 8 GB, glibc heap trimming off
through TIMING_ENV), so nothing a solver starts
survives into the next job. Inputs (the solver, and for scoring the seeded verifier with the seed offset file) go in
through stdin as a tar, so the offset file is never mounted where a solver could read it after the verifier
deleted it. A job past its timeout has its own container killed by name (`docker kill scorer-<job id>`); nothing
else is ever killed. The seed offset and the seeded verifier copies exist only on this machine.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import os
import platform
import queue
import re
import struct
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

from swarm_scaling.algotune_devkit import ASSETS, FINAL_DEV_N, FINAL_DEV_SEED, TIMING_ENV

KINDS = ("dev_eval", "final_eval", "score")
TIMEOUTS_S = {"dev_eval": 900, "final_eval": 900, "score": 3600}
OUTPUT_CHARS = 20_000  # verifier_stdout / dev_eval output kept per job (the end, where the summary is)
KILL_GRACE_S = 60  # after docker kill, how long to wait for the docker CLI to return
DEV_SEED_OFFSET = 10_000  # dev_eval.py's SEED_OFFSET: dev instance i of a job is seed DEV_SEED_OFFSET + seed + i
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")  # set to 8 by default
SLOT_CPUS = 8
SERVICE_CPUS = 8  # CPUs left outside every slot for this service, Docker and the OS
QUEUE_SAMPLE_S = 10


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
    alone_payload: dict | None = None  # cached alone timings sent into the job, or None (the job times them itself)
    alone_baseline: dict | None = None  # what was sent: {"mode": "cached", "age_s", ...} or {"mode": "child"}

    def view(self, cpu_model: str) -> dict:
        return {
            "job_id": self.id, "kind": self.kind, "task": self.task.name, "status": self.status,
            "queued_at": self.queued_at, "started_at": self.started_at, "ended_at": self.ended_at,
            "queue_wait_s": None if self.started_at is None else round(self.started_at - self.queued_at, 3),
            "run_s": None if self.ended_at is None or self.started_at is None else round(self.ended_at - self.started_at, 3),
            "slot": self.slot, "cpu_model": cpu_model, "result": self.result, "output": self.output, "error": self.error,
            "alone_baseline": self.alone_baseline, "timing_env": TIMING_ENV,
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
    alone = {} if job.alone_payload is None else {"alone": json.dumps(job.alone_payload).encode()}
    if job.kind != "score":
        return _tar({"job/solver.py": job.solver_source.encode(), **{"job/alone.json": v for v in alone.values()}})
    files = {"app/solver.py": job.solver_source.encode()}
    for path in sorted(job.task.tests_dir.rglob("*")):
        if path.is_file():
            files[f"tests/{path.relative_to(job.task.tests_dir)}"] = path.read_bytes()
    return _tar({**files, **{"tests/alone_cache.json": v for v in alone.values()}})


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
    argv += ["--alone-cache", "/job/alone.json"] if job.alone_payload is not None else []
    return "tar -x -C / && python /app/dev/dev_eval.py /job/solver.py " + " ".join(str(x) for x in argv) + " --json-out /out/result.json"


def parse_score(stdout: str, reward_text: str | None) -> dict:
    validity = re.search(r"^Validity: (True|False)$", stdout, re.M)
    ratio = re.search(r"^First-call ratio: (\S+)$", stdout, re.M)
    inflation = re.search(r"^Reference inflation: (\S+)$", stdout, re.M)
    if reward_text is None:
        raise JobError("the verifier wrote no reward.txt")
    return {
        "score": float(reward_text.strip()),
        "valid": None if validity is None else validity.group(1) == "True",
        "verifier_stdout": stdout[-OUTPUT_CHARS:],
        "first_call_ratio": float(ratio.group(1)) if ratio and ratio.group(1) != "None" else None,
        "reference_inflation": float(inflation.group(1)) if inflation and inflation.group(1) != "None" else None,
        "scoring_timeout": False,
    }


def timing_env_args() -> list[str]:
    """`docker run -e` arguments for TIMING_ENV: in the process environment before Python starts, as glibc needs."""
    return [arg for k, v in TIMING_ENV.items() for arg in ("-e", f"{k}={v}")]


def docker_runner(docker: str = "docker", memory: str = "8g", work_dir: Path | None = None):
    """Returns run(job, cpuset, timeout_s) -> (result, output) running the job in a fresh container."""

    def run(job: Job, cpuset: str, timeout_s: float) -> tuple[dict | None, str]:
        name = f"scorer-{job.id}"
        with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            out.chmod(0o777)  # the container writes as root
            cmd = [
                # cpuset only: a --cpus quota equal to the cpuset throttles a fully busy job about 1% of the time
                docker, "run", "--rm", "-i", "--init", "--name", name, "--network", "none", "--cpuset-cpus", cpuset,
                "--memory", memory, *timing_env_args(), "-w", "/app", "-v", f"{out}:/out",
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
                            "first_call_ratio": None, "reference_inflation": None, "scoring_timeout": True}, output  # fmt: skip
                reward = out / "reward.txt"
                return parse_score(stdout_s + stderr_s, reward.read_text() if reward.exists() else None), output
            if timed_out:
                raise JobError(f"timed out after {timeout_s:g}s")
            result = out / "result.json"
            if not result.exists():  # the process died before writing a result (crash, OOM kill)
                raise JobError(f"no result, exit {proc.returncode}: {stderr_s[-500:]}")
            return json.loads(result.read_text()), output

    return run


def alone_request(job: Job) -> tuple[list[int], int, int]:
    """The instances (generate_problem seeds), problem size and repeats a job times the reference alone on."""
    if job.kind == "score":  # the verifier's constants and the seed offset of this scorer's seeded copy
        text = (job.task.tests_dir / "test_outputs.py").read_text()
        size, n, reps = (int(re.search(rf"^{k} = (\d+)", text, re.M).group(1)) for k in ("PROBLEM_SIZE", "NUM_TEST_INSTANCES", "NUM_REPEATS"))
        offset = int((job.task.tests_dir / "seed_offset").read_text())
        return [offset + i for i in range(n)], size, reps
    a = job.args
    size = a["size"] if a.get("size") is not None else json.loads((job.task.dev_dir / "config.json").read_text())["problem_size"]
    return [DEV_SEED_OFFSET + a["seed"] + i for i in range(a["n"])], size, a["reps"]


class AloneCache:
    """The reference's alone timing (min ns) per instance seed, per (task, size, repeats, image, CPU model).

    Missing instances are measured by `measure(job, seeds, size, reps, cpuset)` on the slot of the job that first
    needs them, one key at a time, and kept in <dir>/<task>-<key hash>.json with when each was measured.
    """

    def __init__(self, dir: Path, measure: Callable, image_digest: str | dict | None, cpu_model: str) -> None:
        self.dir, self.measure = dir, measure
        self.image_digest, self.cpu_model = image_digest, cpu_model
        dir.mkdir(parents=True, exist_ok=True)
        self.locks: dict[Path, threading.Lock] = {}
        self.guard = threading.Lock()

    def get(self, job: Job, cpuset: str) -> tuple[dict, dict]:
        """(the alone timings the job's dev_eval / verifier reads, what to record about them)."""
        seeds, size, reps = alone_request(job)
        key = {"task": job.task.name, "size": size, "reps": reps, "image": job.task.image,
               "image_digest": json.dumps(self.image_digest), "cpu_model": self.cpu_model}  # fmt: skip
        path = self.dir / f"{job.task.name}-{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]}.json"
        with self.guard:
            lock = self.locks.setdefault(path, threading.Lock())
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {"key": key, "ns": {}, "at": {}, "measured": []}
            missing = [s for s in seeds if str(s) not in data["ns"]]
            measure_s = None
            if missing:
                start = time.time()
                values = self.measure(job, missing, size, reps, cpuset)
                measure_s = round(time.time() - start, 3)
                for seed, ns in zip(missing, values, strict=True):
                    data["ns"][str(seed)], data["at"][str(seed)] = ns, start
                data["measured"].append({"at": start, "instances": len(missing), "measure_s": measure_s, "cpuset": cpuset, "job_id": job.id})
                path.with_suffix(".tmp").write_text(json.dumps(data))
                path.with_suffix(".tmp").replace(path)
                print(f"alone baseline for {job.task.name} (size {size}, {reps} repeats): measured {len(missing)} instances "
                      f"on CPUs {cpuset} in {measure_s:.1f}s", flush=True)  # fmt: skip
        oldest = min(data["at"][str(s)] for s in seeds)
        payload = {"size": size, "reps": reps, "ns": {str(s): data["ns"][str(s)] for s in seeds}}
        return payload, {"mode": "cached", "age_s": round(time.time() - oldest, 1), "measured_now": len(missing), "measure_s": measure_s}


def docker_alone_measure(docker: str = "docker", memory: str = "8g", timeout_s: float = TIMEOUTS_S["score"]):
    """AloneCache's measure: thread_guard.py's alone-timing child (the code the per-job check runs) on each seed, in a
    fresh container with the job's limits, the toolkit and no solver, with the verifier's thread settings."""

    def measure(job: Job, seeds: list[int], size: int, reps: int, cpuset: str) -> list[int]:
        name = f"scorer-{job.id}-alone"
        threads = " ".join(f'{v}="${{{v}:-8}}"' for v in THREAD_VARS)
        script = f"export {threads}; exec python /app/dev/thread_guard.py /app/dev/reference_task.py {reps} 0"
        cmd = [
            docker, "run", "--rm", "-i", "--init", "--name", name, "--network", "none", "--cpuset-cpus", cpuset,
            "--memory", memory, *timing_env_args(), "-w", "/app", "-v", f"{job.task.dev_dir}:/app/dev:ro",
            job.task.image, "sh", "-c", script,
        ]  # fmt: skip
        stdin = "".join(json.dumps({"n": size, "random_seed": s}) + "\n" for s in seeds)
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run([docker, "kill", name], capture_output=True, timeout=KILL_GRACE_S)
            raise JobError(f"alone baseline timed out after {timeout_s:g}s") from None
        values = [int(line.split()[1]) for line in proc.stdout.splitlines() if line.startswith("ALONE ")]
        if proc.returncode != 0 or len(values) != len(seeds):
            raise JobError(f"alone baseline: {len(values)} of {len(seeds)} timings, exit {proc.returncode}: {proc.stderr[-500:]}")
        return values

    return measure


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
        task_image_digest: str | dict | None = None,
        timeouts_s: dict[str, float] | None = None,
        log_path: Path | None = None,
        queue_log_path: Path | None = None,
        queue_sample_s: float = QUEUE_SAMPLE_S,
        alone_cache: AloneCache | None = None,
    ) -> None:
        self.tasks = {**tasks, **{a.slug: a for a in tasks.values()}}  # accept the task name or the sample id
        self.cpusets, self.run = cpusets, run
        self.cpu_model, self.version, self.seed_offset_id = cpu_model, version, seed_offset_id
        self.task_image_digest = task_image_digest
        self.timeouts_s = timeouts_s or TIMEOUTS_S
        self.log_path = log_path
        self.alone_cache = alone_cache
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()  # FIFO across every caller
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.busy_s = [0.0] * len(cpusets)  # per slot: run time of finished jobs
        self.workers = [threading.Thread(target=self._work, args=(slot,), daemon=True) for slot in range(len(cpusets))]
        if queue_log_path is not None:
            self.workers.append(threading.Thread(target=self._sample_queue, args=(queue_log_path, queue_sample_s), daemon=True))
        for w in self.workers:
            w.start()

    def health(self) -> dict:
        return {
            "ok": True, "slots": len(self.cpusets), "queue_len": self.queue.qsize(), "cpu_model": self.cpu_model,
            "version": self.version, "seed_offset_id": self.seed_offset_id, "task_image_digest": self.task_image_digest,
            "cache_alone": self.alone_cache is not None, "timing_env": TIMING_ENV,
            "cpu_dma_latency_us": cpu_dma_latency(),
            "tasks": sorted({a.name for a in self.tasks.values()}),
        }  # fmt: skip

    def stats(self) -> dict:
        """Queue length now, jobs done, queue wait (mean, nearest-rank p95) of started jobs, per-slot busy fraction."""
        now = time.time()
        with self.lock:
            jobs = list(self.jobs.values())
            busy = list(self.busy_s)
            for j in jobs:
                if j.status == "running":
                    busy[j.slot] += now - j.started_at
        waits = sorted(j.started_at - j.queued_at for j in jobs if j.started_at is not None)
        up = now - self.started_at
        return {
            "queue_len": self.queue.qsize(), "running": sum(j.status == "running" for j in jobs),
            "jobs_done": sum(j.status in ("done", "error") for j in jobs), "jobs_error": sum(j.status == "error" for j in jobs),
            "queue_wait_mean_s": round(sum(waits) / len(waits), 3) if waits else None,
            "queue_wait_p95_s": round(waits[math.ceil(0.95 * len(waits)) - 1], 3) if waits else None,
            "uptime_s": round(up, 3), "slot_busy_frac": [round(b / up, 4) if up > 0 else 0.0 for b in busy],
        }  # fmt: skip

    def _sample_queue(self, path: Path, every_s: float) -> None:
        while True:
            with self.lock:
                running = sum(j.status == "running" for j in self.jobs.values())
            with path.open("a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), "queue_len": self.queue.qsize(), "running": running}) + "\n")
            time.sleep(every_s)

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
            job.alone_baseline = {"mode": "child"}
            if self.alone_cache is not None:
                try:
                    job.alone_payload, job.alone_baseline = self.alone_cache.get(job, self.cpusets[slot])
                except Exception as ex:  # this job falls back to timing the reference alone itself
                    job.alone_baseline = {"mode": "child", "cache_error": f"{type(ex).__name__}: {ex}"}
            try:
                result, output = self.run(job, self.cpusets[slot], self.timeouts_s[job.kind])
                status, error = "done", None
            except JobError as ex:
                result, output, status, error = None, "", "error", str(ex)
            except Exception as ex:  # an infrastructure failure (docker missing, disk full): report it on the job
                result, output, status, error = None, "", "error", f"{type(ex).__name__}: {ex}"
            with self.lock:
                job.result, job.output, job.error, job.status, job.ended_at = result, output, error, status, time.time()
                self.busy_s[slot] += job.ended_at - job.started_at
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
            if self.path == "/stats":
                return self._send(200, scorer.stats())
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


def cpu_dma_latency(path: str = "/dev/cpu_dma_latency") -> int | None:
    """The CPU latency limit in effect (µs; 0 = no C-state exits), read from the PM QoS device; None where unreadable.

    scorer.sh starts scripts/cloud/hold_cpu_dma_latency.py on the scorer VM, which holds it at 0 for the VM's life.
    """
    try:
        with open(path, "rb") as f:
            return struct.unpack("i", f.read(4))[0]
    except (OSError, struct.error):
        return None


def docker_cpus(docker: str = "docker") -> int:
    out = subprocess.run([docker, "info", "--format", "{{.NCPU}}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"docker info failed: {out.stderr.strip()}")
    return int(out.stdout)


def slot_cpus(slots: int, first_cpu: int, host_cpus: int) -> tuple[list[str], set[int]]:
    """Each slot's cpuset (slot k = first + 8k .. first + 8k + 7) and the CPUs outside every slot, for the service.

    Refuses slots that run past the host's CPUs or leave fewer than SERVICE_CPUS for the service.
    """
    used = set(range(first_cpu, first_cpu + SLOT_CPUS * slots))
    if slots < 1 or first_cpu < 0 or max(used) >= host_cpus:
        raise ValueError(f"{slots} slots from CPU {first_cpu} need CPUs up to {first_cpu + SLOT_CPUS * slots - 1}; "
                         f"the Docker host has {host_cpus} (0-{host_cpus - 1})")  # fmt: skip
    service = set(range(host_cpus)) - used
    if len(service) < SERVICE_CPUS:
        raise ValueError(f"{slots} slots leave {len(service)} of {host_cpus} CPUs for the service; it needs {SERVICE_CPUS}")
    cpusets = [f"{first_cpu + SLOT_CPUS * k}-{first_cpu + SLOT_CPUS * k + SLOT_CPUS - 1}" for k in range(slots)]
    return cpusets, service


def image_digest_with_retry(image: str, digest_of: Callable[[str], str | None], tries: int = 5, wait_s: float = 10) -> str | None:
    """digest_of(image), retried: on the Mac, Docker Desktop intermittently answers `image inspect <name>` with "No such
    image" while `image ls` lists it (2026-10-10, three failed starts); a few seconds later the same call works."""
    for attempt in range(1, tries + 1):
        try:
            return digest_of(image)
        except RuntimeError as ex:
            if attempt == tries:
                raise
            print(f"{ex} (attempt {attempt} of {tries}; retrying in {wait_s:g}s)", flush=True)
            time.sleep(wait_s)


def git_version(root: Path) -> str:
    out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"], capture_output=True, text=True)
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


def parser() -> argparse.ArgumentParser:
    from swarm_scaling.tasks import PROJECT_ROOT

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--first-cpu", type=int, default=0, help="slot k uses CPUs first + 8k .. first + 8k + 7")
    p.add_argument("--memory", default="8g")
    p.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "data" / ".scorer")
    p.add_argument("--cache-alone", action=argparse.BooleanOptionalAction, default=False,
                   help="time the reference alone once per instance set and reuse it (default off: every job times it)")
    return p


def main() -> None:
    from swarm_scaling.tasks import (
        PROJECT_ROOT, SEED_OFFSET_ENV, SEED_OFFSET_FILE, _seed_offset, seed_offset_id, task_image_digest, task_names,
    )  # fmt: skip

    args = parser().parse_args()

    token = os.environ.get("SCORER_TOKEN")
    if not token:
        raise SystemExit("set SCORER_TOKEN")
    # A missing offset file would silently create a new offset and score on different instances than every
    # other run of the experiment: copy data/.algotune_seed_offset here, or set ALGOTUNE_SEED_OFFSET.
    if not os.environ.get(SEED_OFFSET_ENV) and not SEED_OFFSET_FILE.exists():
        raise SystemExit(f"no seed offset: copy the experiment's {SEED_OFFSET_FILE.name} to {SEED_OFFSET_FILE} or set {SEED_OFFSET_ENV}")
    try:
        cpusets, service_cpus = slot_cpus(args.slots, args.first_cpu, docker_cpus())
    except ValueError as ex:
        raise SystemExit(str(ex)) from None
    if hasattr(os, "sched_setaffinity"):  # Linux, where Docker's CPUs are the host's: keep the service off the slots
        os.sched_setaffinity(0, service_cpus)  # before any thread starts, so every worker inherits it
    offset = _seed_offset()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tasks = prepare_tasks(sorted(task_names().values()), offset, args.work_dir)
    # one string when every task shares one image (AlgoTune does), else {image: digest}; None = built locally
    digests = {a.image: image_digest_with_retry(a.image, task_image_digest) for a in tasks.values()}
    digest = next(iter(digests.values())) if len(digests) == 1 else digests
    cpu = cpu_model()
    alone = AloneCache(args.work_dir / "alone_cache", docker_alone_measure(memory=args.memory), digest, cpu) if args.cache_alone else None
    scorer = Scorer(
        tasks, cpusets, docker_runner(memory=args.memory, work_dir=args.work_dir), cpu, git_version(PROJECT_ROOT),
        seed_offset_id(offset), digest, log_path=args.work_dir / "jobs.jsonl", queue_log_path=args.work_dir / "queue.jsonl",
        alone_cache=alone,
    )  # fmt: skip
    server = ThreadingHTTPServer((args.host, args.port), handler(scorer, token))
    print(f"scorer on {args.host}:{args.port}: slots {cpusets} (service on {len(service_cpus)} other CPUs), {len(tasks)} tasks, cpu {scorer.cpu_model}, "
          f"version {scorer.version}, seed offset id {scorer.seed_offset_id}, cache alone {args.cache_alone}, "
          f"cpu_dma_latency {cpu_dma_latency()}", flush=True)  # fmt: skip
    server.serve_forever()


if __name__ == "__main__":
    main()
