"""Two-stage portfolio selection (v6, PLAN.md), applied identically to every arm.

A portfolio is one team run, or one group of independent runs named `<group>-i<k>` (plans/v6-*.tsv: `<prefix>-<task>-r<k>-<arm>-i<j>`), so the group's
key is the run name without its `-i<k>` suffix.

Stage 1: every candidate from the portfolio's runs, ranked by the dev score finalize recorded for it (FINAL_DEV_N
instances), one entry per distinct source (sha256; a source seen in several places keeps its best score).
Validity pass (v7, --validity-n N): sources in stage-1 order are checked untimed on N instances, VALIDITY_SEED + i
(scorer job kind validity, is_solution only, streamed), until SHORTLIST of them pass; a source with any invalid
instance (or a failed job, e.g. past the service's 3,600 s limit) is dropped and the next stage-1 source backfills,
identically for every arm. --validity-n-task TASK=N sets a task's N, for tasks whose is_solution is slow (measured
2026-10-14 on the Mac: generalized_eigenvalues_complex about 5.5 s per instance, so 1,000 would pass the limit).
Stage 2: the top SHORTLIST distinct sources (with the validity pass: those that passed it) are re-timed by the final
verifier path (problems generated and held in memory, then timed, in the scorer's timing environment) on 100 dev
instances, SELECT_SEED + i (scorer job kind score_dev). The pick is the fastest valid source on stage 2.
Hidden: the pick is then scored on the hidden instances REPEATS times (score jobs, as rescore.py does) and its median
is the portfolio's result; every other shortlisted source is scored once, only for the oracle bound. A portfolio's
pick is fixed before any of its hidden jobs is submitted, so hidden scores never choose. The best hidden score in the
shortlist (the pick's median, the others' single score) is reported as an oracle bound.
Scores: stage 2 and hidden use the logs' primary score (run metadata primary_score: "alone", against the reference
timed alone, for v7; AlgoTune's interleaved "score" before), unless --primary-score overrides; both are recorded.
Exclusive hidden scoring (--exclusive-hidden, default on for v7 logs; 2026-10-18): every hidden job runs as an
exclusive scorer job, with every other slot idle and always on the same slot's CPUs, because in the v7 calibration the
same multi-threaded ode solver scored up to 1.16x apart by slot and load (HARNESS.md, "Scorer noise"). Every hidden
job must come from the scorer session /health named when the selection started (one service run on one machine
type); a selection whose hidden jobs ran in another session fails instead of writing a mixed row.

    SCORER_TOKEN=... uv run python scripts/select_portfolio.py 'logs/v6p-*' --scorer http://<scorer>:8770

Portfolios are selected concurrently, each written to analysis/portfolio/<timestamp>-<pid>.jsonl as soon as it is
done. Every job has a request_id fixed by these runs, the portfolio, the source and the job's place, so a selection
restarted on the same scorer gets its jobs back instead of submitting them again (--fresh re-times instead); network
errors are retried, and a failing connection re-opens the scorer port to this machine (scorer.sh allow-me).
Needs logs whose finalize kept every candidate's source (`sources`, since v6). Refuses logs scored with another seed
offset or experiment than the scorer's. Prints a table at the end.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

import anyio
from inspect_ai.log import read_eval_log

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import SELECT_SEED, VALIDITY_SEED, remote_timing
from swarm_scaling.tasks import PROJECT_ROOT, task_names

sys.path.insert(0, str(Path(__file__).parent))
from rescore import eval_files  # noqa: E402

SHORTLIST = 3
REPEATS = 3
ORACLE_LABEL = (
    "oracle bound: the best hidden score among the shortlist (the pick's median of its repeats, one scoring for each "
    "other source; chosen with hidden scores)"
)

RunJob = Callable[[dict], Awaitable[dict]]
SCORE_KEYS = {"interleaved": "score", "alone": "alone_score"}  # primary score -> the result field it reads


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


def load_portfolios(files: list[Path]) -> tuple[dict[str, dict], set, set]:
    """Portfolios from eval logs (candidates with dev scores, sources), the seed offset ids the logs were scored with,
    and their (experiment, primary score) pairs (logs before v7: v6, interleaved)."""
    portfolios: dict[str, dict] = {}
    offsets, kinds = set(), set()
    for f in files:
        log = read_eval_log(str(f))
        meta = log.eval.metadata or {}
        offsets.add(meta.get("seed_offset_id"))
        kinds.add((meta.get("experiment", "v6"), meta.get("primary_score", "interleaved")))
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
    return portfolios, offsets, kinds


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


def pick(stage2: list[dict], key: str = "score") -> dict | None:
    """The fastest valid stage-2 entry by `key` (ties: the better stage-1 rank); None if none is valid."""
    ok = [e for e in stage2 if e["valid"] and e.get(key) is not None and not e["scoring_timeout"]]
    return min(ok, key=lambda e: (-e[key], e["stage1_rank"])) if ok else None


def _validity(rec: dict) -> dict:
    """A validity job's counts; a job that ended in an error (crash, timeout) is invalid too."""
    res = rec["result"] if rec["status"] == "done" else None
    if res is None:
        return {"valid": False, "n": None, "checked": None, "n_invalid": None, "invalid_instances": [], "errors": [],
                "error": rec.get("error"), "job": remote_timing(rec)}  # fmt: skip
    keep = ("n", "checked", "n_invalid", "invalid_instances", "errors", "run_s")
    return {"valid": bool(res["valid"]), **{k: res.get(k) for k in keep}, "error": None, "job": remote_timing(rec)}


def _result(rec: dict) -> dict:
    res = rec["result"] if rec["status"] == "done" else None
    return {
        "score": res["score"] if res else None, "alone_score": res.get("alone_score") if res else None,
        "valid": res["valid"] if res else None,
        "scoring_timeout": res["scoring_timeout"] if res else None,
        "reference_inflation": res.get("reference_inflation") if res else None,
        "first_call_ratio": res["first_call_ratio"] if res else None,
        "error": None if res else rec.get("error"), "job": remote_timing(rec),
    }  # fmt: skip


async def select(
    portfolios: list[dict], names: dict[str, str], run_job: RunJob = scorer_client.run_job,
    shortlist: int = SHORTLIST, repeats: int = REPEATS, seed: int = SELECT_SEED, concurrency: int = 8,
    primary: str = "interleaved", validity_n: int = 0, validity_seed: int = VALIDITY_SEED,
    on_row: Callable[[dict], None] | None = None, request_prefix: str = "", validity_n_by_task: dict[str, int] | None = None,
    exclusive_hidden: bool = False, scorer: dict | None = None,
) -> list[dict]:
    """One output row per portfolio. Portfolios run concurrently (jobs share `concurrency`); within one, the validity
    pass (validity_n > 0) and stage 2 come first and the pick is fixed before its hidden jobs. on_row(row) gets each
    row as soon as it is final. Every job's request_id starts with request_prefix. exclusive_hidden: hidden jobs run
    as exclusive scorer jobs. scorer: the service's /health at the start; every hidden job must carry its session."""
    limiter = anyio.CapacityLimiter(concurrency)
    key = SCORE_KEYS[primary]

    def job(p: dict, sha: str, kind: str, place: str, **extra) -> dict:
        return {"kind": kind, "task": names[p["task"]], "solver_source": p["sources"][sha], "run_id": "select_portfolio",
                "agent_id": "", "sample_id": f"{p['portfolio']}/{sha[:12]}",
                "request_id": f"{request_prefix}/{p['portfolio']}/{sha}/{kind}/{place}", **extra}  # fmt: skip

    async def run_all(calls: list[tuple[dict, Callable[[dict], None]]]) -> None:
        async def one(body: dict, done: Callable[[dict], None]) -> None:
            async with limiter:
                done(await run_job(body))

        async with anyio.create_task_group() as tg:
            for body, done in calls:
                tg.start_soon(one, body, done)

    async def select_one(p: dict, row: dict) -> None:
        passed = row["stage1"][:shortlist]
        n_check = row["validity_n"]
        if n_check:  # stage-1 order, backfilling each dropped source with the next one
            passed, tried = [], 0
            while len(passed) < shortlist and tried < len(row["stage1"]):
                batch = row["stage1"][tried : tried + shortlist - len(passed)]
                tried += len(batch)
                found: dict[str, dict] = {}
                await run_all([(job(p, e["sha256"], "validity", f"n{n_check}-s{validity_seed}", n=n_check,
                                    seed=validity_seed), lambda rec, e=e: found.update({e["sha256"]: _validity(rec)}))
                               for e in batch])  # fmt: skip
                for e in batch:
                    row["validity"].append({"stage1_rank": e["rank"], "sha256": e["sha256"], **found[e["sha256"]]})
                    if found[e["sha256"]]["valid"]:
                        passed.append(e)

        calls = []
        for e in passed:
            entry = {"stage1_rank": e["rank"], "sha256": e["sha256"], "dev_speedup": e["dev_speedup"]}
            row["stage2"].append(entry)
            calls.append((job(p, e["sha256"], "score_dev", f"s{seed}", seed=seed), lambda rec, entry=entry: entry.update(_result(rec))))
        await run_all(calls)
        row["stage2"].sort(key=lambda e: e["stage1_rank"])
        chosen = pick(row["stage2"], key)  # fixed here, before any of this portfolio's hidden scores exists
        row["pick"] = None if chosen is None else {k: chosen[k] for k in ("sha256", "stage1_rank", "score", "alone_score")}

        for e in row["stage2"]:
            row["hidden"][e["sha256"]] = {"scores": []}

    def finish(p: dict, row: dict) -> None:
        if scorer is not None:
            sessions = {r["job"].get("session") for h in row["hidden"].values() for r in h["scores"]}
            if sessions - {scorer["session"]}:
                raise RuntimeError(f"{p['portfolio']}: hidden jobs ran in scorer sessions {sorted(sessions, key=str)}, not "
                                   f"only {scorer['session']}: rerun the selection on one scorer")  # fmt: skip
        for h in row["hidden"].values():
            for name, field in (("median", key), ("median_algotune", "score"), ("median_alone", "alone_score")):
                scores = [r[field] for r in h["scores"] if r[field] is not None]
                h[name] = statistics.median(scores) if scores else None
        row["pick_hidden_median"] = None if row["pick"] is None else row["hidden"][row["pick"]["sha256"]]["median"]
        scored = [(h["median"], sha) for sha, h in row["hidden"].items() if h["median"] is not None]
        best = max(scored, default=None)
        row["oracle"] = {"label": ORACLE_LABEL, "sha256": best[1] if best else None, "hidden_median": best[0] if best else None}
        if on_row is not None:
            on_row(row)

    rows = []
    for p in portfolios:
        ranked = stage1(p["candidates"])
        rows.append({
            "portfolio": p["portfolio"], "task": p["task"], "arm": p["arm"], "protocol": p["protocol"], "runs": p["runs"],
            "priority_indexes": p["priority_indexes"], "n_candidates": len(p["candidates"]), "n_distinct_valid": len(ranked),
            "primary_score": primary, "stage1": ranked,
            "validity_n": (validity_n_by_task or {}).get(names[p["task"]], validity_n) if validity_n else 0,
            "validity_seed": validity_seed if validity_n else None, "validity": [], "stage2_seed": seed, "stage2": [],
            "pick": None, "hidden": {}, "exclusive_hidden": exclusive_hidden,
            "scorer": None if scorer is None else {k: scorer.get(k) for k in ("session", "machine_type", "cpu_model", "slots", "exclusive_slot")},
        })  # fmt: skip
    async with anyio.create_task_group() as tg:
        for p, row in zip(portfolios, rows):
            tg.start_soon(select_one, p, row)
    # Hidden scoring once every pick is fixed: per task-repeat group, the picks' repeat r of all its portfolios next to
    # each other, then the other shortlisted sources once each, so a slow drift in machine speed (about 5% over tens
    # of minutes on the v7b smoke scorer, HARNESS.md "Scorer noise") does not fall on one arm. Exclusive jobs run one at
    # a time in that order; otherwise they share `concurrency` as before.
    by_group: dict[str, list[tuple[dict, dict]]] = {}
    for p, row in zip(portfolios, rows):
        by_group.setdefault(group_of(p["portfolio"]), []).append((p, row))
    left = {row["portfolio"]: sum(repeats if pick_of(row) == e["sha256"] else 1 for e in row["stage2"]) for row in rows}
    calls = []
    for members in by_group.values():
        order = [(p, row, row["pick"]["sha256"], r) for r in range(repeats) for p, row in members if row["pick"] is not None]
        order += [(p, row, e["sha256"], 0) for p, row in members for e in row["stage2"] if e["sha256"] != pick_of(row)]
        for p, row, sha, r in order:
            def done(rec: dict, p=p, row=row, sha=sha) -> None:
                row["hidden"][sha]["scores"].append(_result(rec))
                left[row["portfolio"]] -= 1
                if not left[row["portfolio"]]:
                    finish(p, row)
            calls.append((job(p, sha, "score", f"r{r}", **({"exclusive": True} if exclusive_hidden else {})), done))
    for p, row in zip(portfolios, rows):
        if not left[row["portfolio"]]:  # nothing to score (no valid stage-2 source)
            finish(p, row)
    if exclusive_hidden:
        for body, done in calls:  # in order: the service runs them one at a time anyway
            done(await run_job(body))
    else:
        await run_all(calls)
    return rows


def group_of(portfolio: str) -> str:
    """The task-repeat group of a portfolio: `<prefix>-<task>-r<k>` of `<prefix>-<task>-r<k>-<arm>` (plans/v6-*.tsv)."""
    return "-".join(portfolio.split("/")[0].split("-")[:3]) + ("/" + portfolio.split("/", 1)[1] if "/" in portfolio else "")


def pick_of(row: dict) -> str | None:
    return None if row["pick"] is None else row["pick"]["sha256"]


def fmt(value: float | None) -> str:
    return f"{value:>8.3f}" if value is not None else f"{'-':>8}"


CLAIMS = PROJECT_ROOT / "analysis" / "portfolio" / ".claims"


def claim_path(runs: list[str]) -> Path:
    return CLAIMS / hashlib.sha256("\n".join(sorted(runs)).encode()).hexdigest()[:16]


def allow_me() -> None:
    """scorer_client.on_unreachable: re-open the scorer and SSH ports to this machine's current public IP. allow-me
    retries the IP lookup for up to 15 min (common.sh my_ip) while the network is down, hence the long timeout. It acts
    on the scorer SCORER_NAME names (v6_drive.py sets select when it selects on the selection scorer)."""
    out = subprocess.run(["bash", "scripts/cloud/scorer.sh", "allow-me"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                         timeout=1200)  # fmt: skip
    print(f"[select] scorer unreachable; allow-me: {(out.stdout + out.stderr).strip()[-300:]}", flush=True)


def claim(logs: list[str]) -> bool:
    """Claim the selection of exactly these runs, so it runs once although the v6 driver (one group at a time) and
    scripts/select_ready.py (several at once, 2026-10-12) both start it. A second caller waits for the claimant's
    process to end, then returns False. Delete analysis/portfolio/.claims/<key> to select the same runs again."""
    path = claim_path([Path(x).name for x in logs])
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = int(path.read_text().strip() or 0)
        print(f"selection of these runs was claimed by pid {pid}; waiting for it to end", flush=True)
        while pid and _alive(pid):
            time.sleep(30)
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check only, sends nothing
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="log dirs, .eval files or globs (quote them)")
    p.add_argument("--scorer", required=True, help="the remote scorer service URL (token: SCORER_TOKEN)")
    p.add_argument("--shortlist", type=int, default=SHORTLIST)
    p.add_argument("--repeats", type=int, default=REPEATS, help="hidden scorings of the pick (the others: one each)")
    p.add_argument("--concurrency", type=int, default=8, help="jobs in flight at once")
    p.add_argument("--primary-score", choices=sorted(SCORE_KEYS), default=None,
                   help="what stage 2 and hidden scoring rank by (default: the logs' primary_score)")
    p.add_argument("--validity-n", type=int, default=0,
                   help="v7: check every shortlisted source untimed on this many fresh instances first (0 = off; v7: 1000)")
    p.add_argument("--validity-n-task", action="append", default=[], metavar="TASK=N",
                   help="a task's own validity-pass size (repeatable), e.g. generalized_eigenvalues_complex=100")
    p.add_argument("--fresh", action="store_true", help="new request ids: re-time instead of resuming earlier jobs")
    p.add_argument("--exclusive-hidden", action=argparse.BooleanOptionalAction, default=None,
                   help="hidden jobs run with every other scorer slot idle, on one slot (default: on for v7 logs)")
    args = p.parse_args()
    os.environ["SCORER_URL"] = args.scorer
    if not claim(args.logs):
        return
    scorer_client.on_unreachable = allow_me

    portfolios, offsets, kinds = load_portfolios(eval_files(args.logs))
    if not portfolios:
        sys.exit("no portfolios in the given logs")
    if len(kinds) != 1:
        sys.exit(f"logs mix experiments or primary scores {sorted(kinds)}: select each experiment on its own")
    ((experiment, logged_primary),) = kinds
    service = scorer_client.health()
    if offsets != {service["seed_offset_id"]}:
        sys.exit(f"logs were scored with seed offset id(s) {sorted(offsets, key=str)}, the scorer uses {service['seed_offset_id']}")
    if service.get("experiment", "v6") != experiment:
        sys.exit(f"logs are experiment {experiment}, the scorer serves {service.get('experiment', 'v6')}")
    primary = args.primary_score or logged_primary
    exclusive = args.exclusive_hidden if args.exclusive_hidden is not None else experiment == "v7"
    if exclusive and "exclusive_slot" not in service:
        sys.exit("the scorer predates exclusive jobs: it would run hidden jobs beside others (--no-exclusive-hidden to accept that)")
    prefix = claim_path([Path(x).name for x in args.logs]).name + (f"-{time.time_ns()}" if args.fresh else "")

    out = PROJECT_ROOT / "analysis" / "portfolio" / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(row: dict) -> None:  # each portfolio as soon as it is final, so a crash keeps the finished ones
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[select] {time.strftime('%H:%M:%S')} {row['portfolio']} done -> {out}", flush=True)

    rows = anyio.run(lambda: select(list(portfolios.values()), task_names(), shortlist=args.shortlist,
                                    repeats=args.repeats, concurrency=args.concurrency, primary=primary,
                                    validity_n=args.validity_n, on_row=write, request_prefix=prefix,
                                    validity_n_by_task={k: int(v) for k, v in (x.split("=", 1) for x in args.validity_n_task)},
                                    exclusive_hidden=exclusive, scorer=service if "session" in service else None))  # fmt: skip
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
