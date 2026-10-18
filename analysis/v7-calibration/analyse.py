"""Per-agent and per-run measurements for the v7 Opus calibration (analysis/v7-calibration.md). Read-only on the logs.

    uv run python analysis/v7-calibration/analyse.py 'v7c-*' > analysis/v7-calibration/agents.json

Per agent: spend (every model call priced with prices.usage_cost, as run_cost.py does) and the metered value, model
calls (completed calls only: a failed call, e.g. ReadError(''), has no usage and is retried as a new call; the
calibration report subtracted 3 such calls by hand before this was fixed, 2026-10-18), last publish (its share of the
agent's own spend), spend and calls after it, distinct published sources, end
reason, dev_eval calls and queue waits. Per team: publish notices, their deliveries, and whether each recipient used one
(a later tool call of its own names the entry's id or path). Per run: the run's own hidden score (primary alone and
AlgoTune's interleaved).
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

from swarm_scaling.prices import usage_cost

ROOT = Path(__file__).resolve().parents[2]


def agent_spans(sample) -> dict:
    parent, agent_of = {}, {}
    for e in sample.events:
        if e.event == "span_begin":
            parent[e.id] = e.parent_id
            if re.fullmatch(r"agent_\d+", e.name or ""):
                agent_of[e.id] = e.name

    def find(span_id):
        for _ in range(200):
            if not span_id or span_id in agent_of:
                break
            span_id = parent.get(span_id)
        return agent_of.get(span_id) if span_id else None

    return find


def run_rows(run_dir: Path) -> tuple[list[dict], dict]:
    path = max(glob.glob(f"{run_dir}/*.eval"), key=os.path.getmtime)
    log = read_eval_log(path, resolve_attachments=True)
    (s,) = log.samples
    sw, fz = s.metadata["swarm"], s.metadata.get("finalize") or {}
    score = next(iter(s.scores.values())) if s.scores else None
    find = agent_spans(s)
    calls: dict[str, list] = {}
    tools: dict[str, list] = {}
    for e in s.events:
        a = find(getattr(e, "span_id", None))
        if a is None:
            continue
        if e.event == "model":
            if e.error is not None or not (e.output and e.output.usage):  # a failed call: no usage, no spend
                continue
            u = e.output.usage.model_dump(exclude_none=True)
            ttl1h = sw["agents"][a].get("cache_ttl") == "1h"
            cost = usage_cost(e.model, u, (u.get("input_tokens_cache_write") or 0) if ttl1h else 0)
            calls.setdefault(a, []).append({"t": e.timestamp.timestamp(), "cost": cost, "out": u.get("output_tokens", 0)})
        elif e.event == "tool":
            tools.setdefault(a, []).append({"t": e.timestamp.timestamp(), "fn": e.function, "args": json.dumps(e.arguments)})
    published = {}
    for c in fz.get("candidates", []):
        if c["kind"] == "published" and c.get("sha256"):
            published.setdefault(c["agent_id"], []).append(c)
    queue = (s.metadata.get("checker") or {}).get("agents", {})
    rows = []
    for aid, rec in sw["agents"].items():
        mine = sorted(calls.get(aid, []), key=lambda c: c["t"])
        pubs = sorted(published.get(aid, []), key=lambda c: c["published_at"])
        spend = sum(c["cost"] for c in mine)
        last = pubs[-1]["published_at"] if pubs else None
        after = [c for c in mine if last is not None and c["t"] > last]
        before_spend = spend - sum(c["cost"] for c in after)
        rows.append({
            "run": run_dir.name, "agent": aid, "priority_index": rec.get("priority_index"),
            "spend": round(spend, 4), "metered": rec["tokens"].get("metered"), "calls": len(mine),
            "output_tokens": sum(c["out"] for c in mine), "end_reason": rec["end_reason"], "error": rec["error"],
            "wall_s": rec["wall_s"], "publishes": len(pubs), "distinct_sources": len({c["sha256"] for c in pubs}),
            "valid_published": sum(1 for c in pubs if c.get("valid")),
            "last_publish_spend_frac": None if last is None else round(before_spend / spend, 4),
            "spend_after_last_publish": None if last is None else round(spend - before_spend, 4),
            "spend_after_last_publish_frac": None if last is None else round((spend - before_spend) / spend, 4),
            "calls_after_last_publish": None if last is None else len(after),
            "best_dev": max((c["speedup"] for c in pubs if c.get("valid") and c.get("speedup")), default=None),
            "messages_sent": len(rec["messages_sent"]),
            "messages_received": sum(1 for m in rec["messages_received"] if m.get("from") != "system"),
            **{f"q_{k}": v for k, v in queue.get(aid, {}).items()},
        })  # fmt: skip
    notices = []
    for n in sw.get("registry_notices", []):
        used = {}
        for to, at in n["delivered"].items():
            later = [t for t in tools.get(to, []) if t["t"] >= at and (n["entry"] in t["args"] or n["path"] in t["args"])]
            used[to] = round(later[0]["t"] - at, 1) if later else None
        notices.append({"entry": n["entry"], "publisher": n["publisher"], "dev_score": n["dev_score"], "to": n["to"],
                        "delivered": len(n["delivered"]), "used_after_s": used})  # fmt: skip
    run = {
        "run": run_dir.name, "arm": log.eval.metadata.get("arm"), "task": str(s.id),
        "score_primary": None if score is None else score.value,  # v7: the alone score
        "score_interleaved": None if score is None else (score.metadata or {}).get("algotune_score"),
        "reference_inflation": None if score is None else (score.metadata or {}).get("reference_inflation"),
        "selected": fz.get("selected"), "notices": notices, "cpu_boxes": sw.get("cpu_boxes"), "cpu": sw.get("cpu"),
        "sample_error": s.error.message if s.error else None,
    }  # fmt: skip
    return rows, run


def main() -> None:
    out = {"agents": [], "runs": []}
    for d in sorted(glob.glob(str(ROOT / "logs" / sys.argv[1]))):
        rows, run = run_rows(Path(d))
        out["agents"] += rows
        out["runs"].append(run)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
