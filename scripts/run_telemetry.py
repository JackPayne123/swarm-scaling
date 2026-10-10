"""Waits, scorer slot use and cache rewrites for any pilot log (team or single agent).

Usage: uv run python scripts/run_telemetry.py <log.eval | dir> [--scorer-dir logs/_scorer/<instance>]

Per sample: each agent's end reason, spend, dev_eval queue waits (total, max, share of its active time) and time
limit; model calls right after a dev_eval result whose cache write exceeds 50% of their input (the prompt cache
expired during the wait and the context was written again at the write price); and, when the scorer's job log is
found (--scorer-dir, else any logs/_scorer/*/jobs.jsonl holding jobs of this run), each slot's busy fraction over
the sample's run and the queue length samples (queue.jsonl) in that window. The run is the log's directory name
(the runner's --name, which labels its scorer jobs).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from swarm_scaling.algotune_devkit import queue_telemetry

ROOT = Path(__file__).resolve().parents[1]
REWRITE_FRAC = 0.5
WAIT_RE = re.compile(r"^\[dev_eval\] waited ([\d.]+)s", re.M)


def agent_sequences(sample) -> dict[str, list[tuple]]:
    """Per agent, in event order: ("model", t, usage dict) and ("tool", t, function, result text)."""
    parent: dict[str, str | None] = {}
    agent_of_span: dict[str, str] = {}
    for e in sample.events:
        if e.event == "span_begin":
            parent[e.id] = e.parent_id
            if re.fullmatch(r"agent_\d+", e.name or ""):
                agent_of_span[e.id] = e.name

    def agent_for(span_id):
        for _ in range(200):
            if not span_id or span_id in agent_of_span:
                break
            span_id = parent.get(span_id)
        return agent_of_span.get(span_id) if span_id else None

    seqs = defaultdict(list)
    for e in sample.events:
        agent = agent_for(getattr(e, "span_id", None))
        if agent is None:
            continue
        t = e.timestamp.timestamp()
        if e.event == "model":
            u = e.output.usage if e.output else None
            seqs[agent].append(("model", t, u.model_dump(exclude_none=True) if u else {}))
        elif e.event == "tool":
            seqs[agent].append(("tool", t, e.function, str(e.result)))
    return dict(seqs)


def cache_rewrites(seq: list[tuple], frac: float = REWRITE_FRAC) -> list[dict]:
    """Model calls made right after a dev_eval result (the previous turn's tools included dev_eval) whose cache
    write is above `frac` of their input (fresh + cache write + cache read)."""
    out, waits, t0 = [], [], seq[0][1] if seq else 0.0
    for item in seq:
        if item[0] == "tool":
            if item[2] == "dev_eval":
                m = WAIT_RE.search(item[3])
                waits.append(float(m.group(1)) if m else None)
            continue
        usage = item[2]
        write = usage.get("input_tokens_cache_write") or 0
        total = usage.get("input_tokens", 0) + write + (usage.get("input_tokens_cache_read") or 0)
        if waits and total and write > frac * total:
            known = [w for w in waits if w is not None]
            out.append({"t_s": round(item[1] - t0, 1), "dev_eval_wait_s": max(known) if known else None,
                        "cache_write": write, "input_total": total})  # fmt: skip
        waits = []
    return out


def slot_use(jobs: list[dict], start: float, end: float, slots: int) -> list[float]:
    """Each slot's busy fraction over [start, end] (any run's jobs: they share the slots)."""
    busy = [0.0] * slots
    for j in jobs:
        if j.get("slot") is None or j.get("started_at") is None or j.get("ended_at") is None:
            continue
        busy[j["slot"]] += max(0.0, min(j["ended_at"], end) - max(j["started_at"], start))
    return [round(b / (end - start), 3) if end > start else 0.0 for b in busy]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def find_scorer_dir(run_id: str) -> Path | None:
    for path in sorted((ROOT / "logs" / "_scorer").glob("*/jobs.jsonl")):
        if any(j.get("run_id") == run_id for j in read_jsonl(path)):
            return path.parent
    return None


def sample_report(sample, run_id: str, eval_metadata: dict | None = None, scorer_dir: Path | None = None) -> list[str]:
    sw = sample.metadata.get("swarm", {})
    agents = sw.get("agents", {})
    checker = sample.metadata.get("checker") or {}
    waits = checker.get("agents") or queue_telemetry(checker.get("calls", []), agents, sw.get("time_limit"))
    lines = [f"== telemetry: {sample.id} epoch {sample.epoch} (run {run_id}) ==",
             f"budget {sw.get('per_agent_tokens')} ({sw.get('budget_type')}), min_spend_frac {sw.get('min_spend_frac', 0.0)}"]  # fmt: skip
    for aid, rec in agents.items():
        w = waits.get(aid, {})
        frac = f"{w['queue_wait_frac']:.1%}" if w.get("queue_wait_frac") is not None else "-"
        lines.append(
            f"  {aid}: end={rec.get('end_reason')} spent={rec.get('tokens', {}).get('metered')} wall={rec.get('wall_s')}s "
            f"time_limit={w.get('time_limit_s')}s submit_refusals={len(rec.get('submit_refusals') or [])} | "
            f"dev_eval {w.get('dev_eval_calls', 0)} calls, queue wait total {w.get('queue_wait_total_s', 0):.0f}s "
            f"max {w.get('queue_wait_max_s', 0):.0f}s ({frac} of active time)"
        )
    flagged = {a: cache_rewrites(seq) for a, seq in agent_sequences(sample).items()}
    n = sum(len(v) for v in flagged.values())
    lines.append(f"calls after a dev_eval result with cache write > {REWRITE_FRAC:.0%} of input: {n}")
    for aid, calls in sorted(flagged.items()):
        for c in calls:
            lines.append(f"  {aid} at {c['t_s']}s: wait {c['dev_eval_wait_s']}s, cache write {c['cache_write']:,} of {c['input_total']:,} input")

    scorer_dir = scorer_dir or find_scorer_dir(run_id)
    start, end = sw.get("started_at"), sw.get("ended_at")
    if scorer_dir is None or start is None:
        lines.append("scorer slots: no scorer job log found" if start is not None else "scorer slots: no run window")
        return lines
    jobs = read_jsonl(scorer_dir / "jobs.jsonl")
    slots = (eval_metadata or {}).get("scorer", {}).get("slots") or 1 + max((j["slot"] for j in jobs if j.get("slot") is not None), default=0)
    mine = [j for j in jobs if j.get("run_id") == run_id and j.get("queue_wait_s") is not None]
    lines.append(f"scorer slots ({scorer_dir}): busy over the agents' run {slot_use(jobs, start, end, slots)}")
    if mine:
        qw = [j["queue_wait_s"] for j in mine]
        lines.append(f"  this run's jobs: {len(mine)}, queue wait mean {sum(qw) / len(qw):.0f}s max {max(qw):.0f}s")
    queue = [q for q in read_jsonl(scorer_dir / "queue.jsonl") if start <= q["t"] <= end]
    if queue:
        lens = [q["queue_len"] for q in queue]
        lines.append(f"  queue length over the run: mean {sum(lens) / len(lens):.1f}, max {max(lens)} ({len(queue)} samples)")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--scorer-dir", type=Path, help="directory holding the scorer's jobs.jsonl (and queue.jsonl)")
    args = p.parse_args()
    target = Path(args.log)
    path = max(target.glob("*.eval"), key=os.path.getmtime) if target.is_dir() else target
    log = read_eval_log(str(path))
    for sample in log.samples or []:
        print("\n".join(sample_report(sample, path.parent.name, log.eval.metadata, args.scorer_dir)))


if __name__ == "__main__":
    sys.exit(main())
