"""Client for the remote scorer service (scorer_service.py), configured by SCORER_URL and SCORER_TOKEN.

Jobs are asynchronous: submit, then poll every POLL_S seconds, so a long timing run does not depend on one
HTTP connection staying open.
"""

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import anyio

POLL_S = 2.0
# Run-time limits the service enforces (it kills the job's container after these), plus a margin for the poll.
TIMEOUTS_S = {"dev_eval": 900, "final_eval": 900, "score": 3600}
MARGIN_S = 120
MAX_CONSECUTIVE_FAILURES = 5  # connection failures while polling before giving up


class ScorerError(RuntimeError):
    pass


def config() -> tuple[str, str]:
    url, token = os.environ.get("SCORER_URL"), os.environ.get("SCORER_TOKEN")
    if not url or not token:
        raise ScorerError("the remote checker needs SCORER_URL and SCORER_TOKEN in the environment")
    return url.rstrip("/"), token


def scorer_host() -> str:
    return urlparse(config()[0]).hostname or ""


def request(method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
    url, token = config()
    req = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as ex:  # a definite answer from the service: do not retry
        raise ScorerError(f"{method} {path}: HTTP {ex.code}: {ex.read().decode(errors='replace')[:500]}") from ex


def health() -> dict:
    return request("GET", "/health")


def submit(job: dict) -> str:
    return request("POST", "/jobs", job)["job_id"]


def get(job_id: str) -> dict:
    return request("GET", f"/jobs/{job_id}")


async def run_job(job: dict) -> dict:
    """Submit a job and poll it to the end, without blocking the event loop.

    Shielded from cancellation like the local checker run, so a caller that is stopped mid-call still waits
    for (and records) a job that already occupies a scorer slot.
    """
    with anyio.CancelScope(shield=True):
        job_id = await anyio.to_thread.run_sync(submit, job)
        rec = None
        running_since = None
        failures = 0
        while True:
            await anyio.sleep(POLL_S)
            try:
                rec = await anyio.to_thread.run_sync(get, job_id)
            except (urllib.error.URLError, OSError):
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise
                continue
            failures = 0
            if rec["status"] in ("done", "error"):
                return {**rec, "job_id": job_id, "scorer_host": scorer_host()}
            if rec["status"] == "running":
                running_since = running_since or time.monotonic()
                limit = TIMEOUTS_S[job["kind"]] + MARGIN_S
                if time.monotonic() - running_since > limit:
                    raise TimeoutError(f"scorer job {job_id} still running after {limit} s")
