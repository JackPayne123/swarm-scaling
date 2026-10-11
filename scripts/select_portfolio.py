"""Two-stage portfolio selection (v6, PLAN.md), applied identically to every arm.

A portfolio is one team run, or one group of independent runs named `<group>-i<k>` (plans/v6-*.tsv), so the group's
key is the run name without its `-i<k>` suffix.

Stage 1: every candidate from the portfolio's runs, ranked by the dev score finalize recorded for it (FINAL_DEV_N
instances), one entry per distinct source (sha256; a source seen in several places keeps its best score).
Stage 2: the top SHORTLIST distinct sources are re-timed by the final verifier path (problems generated and held in
memory, then timed, in the scorer's timing environment) on 100 dev instances, SELECT_SEED + i (scorer job kind
score_dev). The pick is the fastest valid source on stage 2.
Hidden: each shortlisted source is then scored on the hidden instances REPEATS times (score jobs, as rescore.py does)
and its median kept; the pick's median is the portfolio's result. Every pick is fixed before any hidden job is
submitted, so hidden scores never choose. The best hidden median in the shortlist is reported as an oracle bound.

    SCORER_TOKEN=... uv run python scripts/select_portfolio.py 'logs/v6p-*' --scorer http://<scorer>:8770

Needs logs whose finalize kept every candidate's source (`sources`, since v6). Refuses logs scored with another seed
offset than the scorer's. Writes analysis/portfolio/<timestamp>.jsonl (one row per portfolio) and prints a table.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

import anyio
from inspect_ai.log import read_eval_log

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import SELECT_SEED, remote_timing
from swarm_scaling.tasks import PROJECT_ROOT, task_names

sys.path.insert(0, str(Path(__file__).parent))
from rescore import eval_files  # noqa: E402

SHORTLIST = 3
REPEATS = 3
ORACLE_LABEL = "oracle bound: the best hidden median among the shortlist (chosen with hidden scores)"

RunJob = Callable[[dict], Awaitable[dict]]


def portfolio_key(run: str, arm: str) -> str:
    """A team run is its own portfolio; independent runs `<group>-i<k>` belong to `<group>`."""
    if arm == "team":
        return run
    if arm != "independent":
        raise ValueError(f"{run}: arm {arm!r} is not a v6 portfolio arm")
    group = re.sub(r"-i\d+$", "", run)
    if group == run:
        raise ValueError(f"{run}: an independent run's name must end in -i<k> to place it in a portfolio")
    return group


def load_portfolios(files: list[Path]) -> tuple[dict[str, dict], set]:
    """Portfolios from eval logs (candidates with dev scores, sources) and the seed offset ids the logs were scored with."""
    portfolios: dict[str, dict] = {}
    offsets = set()
    for f in files:
        log = read_eval_log(str(f))
        meta = log.eval.metadata or {}
        offsets.add(meta.get("seed_offset_id"))
        run, arm = f.parent.name, meta.get("arm")
        for s in log.samples or []:
            fin = (s.metadata or {}).get("finalize") or {}
            if not fin:  # the sample errored before finalize: its runs contribute nothing, and the row's `runs` shows it
                print(f"skipped (no finalize): {run} {s.id} epoch {s.epoch} {s.error.message if s.error else ''}")
                continue
            if "sources" not in fin:
                raise ValueError(f"{run}: finalize kept no candidate sources (log predates v6)")
            key = portfolio_key(run, arm) + (f"/e{s.epoch}" if s.epoch > 1 else "")
            p = portfolios.setdefault(key, {
                "portfolio": key, "task": str(s.id), "arm": arm, "protocol": meta.get("protocol"), "runs": [],
                "priority_indexes": [], "candidates": [], "sources": {},
            })  # fmt: skip
            if p["task"] != str(s.id) or p["arm"] != arm or p["protocol"] != meta.get("protocol"):
                raise ValueError(f"{key}: runs differ in task, arm or prompt variant")
            p["runs"].append(run)
            p["priority_indexes"] += [a.get("priority_index") for a in s.metadata["swarm"]["agents"].values()]
            p["sources"].update(fin["sources"])
            p["candidates"] += [{**c, "run": run} for c in fin["candidates"]]
    return portfolios, offsets


def stage1(candidates: list[dict]) -> list[dict]:
    """Valid candidates ranked by recorded dev speedup, one entry per distinct source, rank 1 = fastest."""
    best: dict[str, dict] = {}
    for c in candidates:
        if not c.get("valid") or c.get("speedup") is None or not c.get("sha256"):
            continue
        if c["sha256"] not in best or c["speedup"] > best[c["sha256"]]["dev_speedup"]:
            best[c["sha256"]] = {"sha256": c["sha256"], "dev_speedup": c["speedup"], "run": c["run"],
                                 "agent_id": c["agent_id"], "kind": c["kind"]}  # fmt: skip
    ranked = sorted(best.values(), key=lambda e: (-e["dev_speedup"], e["sha256"]))
    return [{"rank": i, **e} for i, e in enumerate(ranked, 1)]


def pick(stage2: list[dict]) -> dict | None:
    """The fastest valid stage-2 entry (ties: the better stage-1 rank); None if none is valid."""
    ok = [e for e in stage2 if e["valid"] and e["score"] is not None and not e["scoring_timeout"]]
    return min(ok, key=lambda e: (-e["score"], e["stage1_rank"])) if ok else None


def _result(rec: dict) -> dict:
    res = rec["result"] if rec["status"] == "done" else None
    return {
        "score": res["score"] if res else None, "valid": res["valid"] if res else None,
        "scoring_timeout": res["scoring_timeout"] if res else None,
        "reference_inflation": res.get("reference_inflation") if res else None,
        "first_call_ratio": res["first_call_ratio"] if res else None,
        "error": None if res else rec.get("error"), "job": remote_timing(rec),
    }  # fmt: skip


async def select(
    portfolios: list[dict], names: dict[str, str], run_job: RunJob = scorer_client.run_job,
    shortlist: int = SHORTLIST, repeats: int = REPEATS, seed: int = SELECT_SEED, concurrency: int = 8,
) -> list[dict]:
    """One output row per portfolio. Stage 2 runs for every portfolio and every pick is fixed before the hidden jobs."""
    limiter = anyio.CapacityLimiter(concurrency)

    def job(p: dict, sha: str, kind: str, **extra) -> dict:
        return {"kind": kind, "task": names[p["task"]], "solver_source": p["sources"][sha], "run_id": "select_portfolio",
                "agent_id": "", "sample_id": f"{p['portfolio']}/{sha[:12]}", **extra}  # fmt: skip

    async def run_all(calls: list[tuple[dict, Callable[[dict], None]]]) -> None:
        async def one(body: dict, done: Callable[[dict], None]) -> None:
            async with limiter:
                done(await run_job(body))

        async with anyio.create_task_group() as tg:
            for body, done in calls:
                tg.start_soon(one, body, done)

    rows = []
    for p in portfolios:
        ranked = stage1(p["candidates"])
        rows.append({
            "portfolio": p["portfolio"], "task": p["task"], "arm": p["arm"], "protocol": p["protocol"], "runs": p["runs"],
            "priority_indexes": p["priority_indexes"], "n_candidates": len(p["candidates"]), "n_distinct_valid": len(ranked),
            "stage1": ranked, "stage2_seed": seed, "stage2": [], "pick": None, "hidden": {},
        })  # fmt: skip

    calls = []
    for p, row in zip(portfolios, rows):
        for e in row["stage1"][:shortlist]:
            entry = {"stage1_rank": e["rank"], "sha256": e["sha256"], "dev_speedup": e["dev_speedup"]}
            row["stage2"].append(entry)
            calls.append((job(p, e["sha256"], "score_dev", seed=seed), lambda rec, entry=entry: entry.update(_result(rec))))
    await run_all(calls)

    for row in rows:  # fixed here, before any hidden score exists
        row["stage2"].sort(key=lambda e: e["stage1_rank"])
        chosen = pick(row["stage2"])
        row["pick"] = None if chosen is None else {k: chosen[k] for k in ("sha256", "stage1_rank", "score")}

    calls = []
    for p, row in zip(portfolios, rows):
        for e in row["stage2"]:
            row["hidden"][e["sha256"]] = {"scores": []}
            for _ in range(repeats):
                calls.append((job(p, e["sha256"], "score"),
                              lambda rec, h=row["hidden"][e["sha256"]]: h["scores"].append(_result(rec))))  # fmt: skip
    await run_all(calls)

    for row in rows:
        for h in row["hidden"].values():
            scores = [r["score"] for r in h["scores"] if r["score"] is not None]
            h["median"] = statistics.median(scores) if scores else None
        row["pick_hidden_median"] = None if row["pick"] is None else row["hidden"][row["pick"]["sha256"]]["median"]
        scored = [(h["median"], sha) for sha, h in row["hidden"].items() if h["median"] is not None]
        best = max(scored, default=None)
        row["oracle"] = {"label": ORACLE_LABEL, "sha256": best[1] if best else None, "hidden_median": best[0] if best else None}
    return rows


def fmt(value: float | None) -> str:
    return f"{value:>8.3f}" if value is not None else f"{'-':>8}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="log dirs, .eval files or globs (quote them)")
    p.add_argument("--scorer", required=True, help="the remote scorer service URL (token: SCORER_TOKEN)")
    p.add_argument("--shortlist", type=int, default=SHORTLIST)
    p.add_argument("--repeats", type=int, default=REPEATS, help="hidden scorings per shortlisted source")
    p.add_argument("--concurrency", type=int, default=8, help="jobs in flight at once")
    args = p.parse_args()
    os.environ["SCORER_URL"] = args.scorer

    portfolios, offsets = load_portfolios(eval_files(args.logs))
    if not portfolios:
        sys.exit("no portfolios in the given logs")
    service = scorer_client.health()
    if offsets != {service["seed_offset_id"]}:
        sys.exit(f"logs were scored with seed offset id(s) {sorted(offsets, key=str)}, the scorer uses {service['seed_offset_id']}")
    rows = anyio.run(lambda: select(list(portfolios.values()), task_names(), shortlist=args.shortlist,
                                    repeats=args.repeats, concurrency=args.concurrency))  # fmt: skip

    out = PROJECT_ROOT / "analysis" / "portfolio" / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{'portfolio':40} {'arm':12} {'prompt':12} {'runs':>4} {'distinct':>8} {'s1 rank':>7} {'stage 2':>8} "
          f"{'hidden':>8} {'oracle*':>8}")  # fmt: skip
    for r in sorted(rows, key=lambda r: r["portfolio"]):
        pk = r["pick"] or {}
        print(f"{r['portfolio']:40} {r['arm']:12} {str(r['protocol']):12} {len(r['runs']):>4} {r['n_distinct_valid']:>8} "
              f"{pk.get('stage1_rank', '-'):>7} {fmt(pk.get('score'))} {fmt(r['pick_hidden_median'])} "
              f"{fmt(r['oracle']['hidden_median'])}")  # fmt: skip
    print(f"* {ORACLE_LABEL}\nwrote {out} ({len(rows)} portfolios)")


if __name__ == "__main__":
    main()
