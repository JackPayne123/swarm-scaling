"""Client for the remote scorer service (scorer_service.py), configured by SCORER_URL and SCORER_TOKEN.

Jobs are asynchronous: submit, then poll every POLL_S seconds, so a long timing run does not depend on one
HTTP connection staying open. Every job carries a request_id, so a POST retried after a network error (it may have
reached the service) returns the job it already made instead of a duplicate; polling then follows that job id. Network
errors are retried for up to NETWORK_GRACE_S, calling `on_unreachable` (if set) at most every UNREACHABLE_EVERY_S:
the operator's selection sets it to re-open the scorer port to this machine's current IP (v6: the Mac's IP changed
twice and cut selection off).
"""

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Callable
from urllib.parse import urlparse

import anyio

POLL_S = 2.0
# Run-time limits the service enforces (it kills the job's container after these), plus a margin for the poll.
TIMEOUTS_S = {"dev_eval": 900, "final_eval": 900, "score": 3600, "score_dev": 3600, "validity": 3600}
MARGIN_S = 120
NETWORK_GRACE_S = 1800.0  # consecutive seconds of network errors on one job before giving up
MAX_BACKOFF_S = 30.0
UNREACHABLE_EVERY_S = 60.0
on_unreachable: Callable[[], None] | None = None  # e.g. scripts/select_portfolio.py: scorer.sh allow-me
_last_unreachable = float("-inf")


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


async def _network_retry(call: Callable[[], dict | str], failing_since: float | None, attempt: int):
    """call() once; on a network error, wait (backing off, after on_unreachable) and return None, or raise once errors
    have lasted NETWORK_GRACE_S. Returns (result or None, failing_since, attempt)."""
    global _last_unreachable
    try:
        return await anyio.to_thread.run_sync(call), None, 0
    except (urllib.error.URLError, OSError):  # incl. timeouts and resets; an HTTPError answer raised ScorerError
        now = time.monotonic()
        failing_since = failing_since if failing_since is not None else now
        if now - failing_since >= NETWORK_GRACE_S:
            raise
        if on_unreachable is not None and now - _last_unreachable >= UNREACHABLE_EVERY_S:
            _last_unreachable = now
            await anyio.to_thread.run_sync(on_unreachable)
        await anyio.sleep(min(POLL_S * 2**attempt, MAX_BACKOFF_S))
        return None, failing_since, attempt + 1


async def run_job(job: dict) -> dict:
    """Submit a job and poll it to the end, without blocking the event loop.

    Shielded from cancellation like the local checker run, so a caller that is stopped mid-call still waits
    for (and records) a job that already occupies a scorer slot. A job without a request_id gets a fresh one, so its
    own retried POSTs never start a second job; a caller that passes a stable one (select_portfolio.py) also gets the
    same job back after a restart, as long as the service is the same.
    """
    job = {**job, "request_id": job.get("request_id") or uuid.uuid4().hex}
    with anyio.CancelScope(shield=True):
        job_id, failing_since, attempt = None, None, 0
        while job_id is None:
            job_id, failing_since, attempt = await _network_retry(lambda: submit(job), failing_since, attempt)
        rec = None
        running_since = None
        failing_since, attempt = None, 0
        while True:
            await anyio.sleep(POLL_S)
            rec, failing_since, attempt = await _network_retry(lambda: get(job_id), failing_since, attempt)
            if rec is None:
                continue
            if rec["status"] in ("done", "error"):
                return {**rec, "job_id": job_id, "scorer_host": scorer_host()}
            if rec["status"] == "running":
                running_since = running_since or time.monotonic()
                limit = TIMEOUTS_S[job["kind"]] + MARGIN_S
                if time.monotonic() - running_since > limit:
                    raise TimeoutError(f"scorer job {job_id} still running after {limit} s")
