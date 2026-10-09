"""Re-time every run's selected AlgoTune solver in one consistent serial pass.

For each sample in the given logs, takes state.metadata["finalize"]["selected_source"] and scores it again
through the normal scoring path: `algotune_task` (same patched verifier, same secret seed offset) and
`algotune_scorer` (copies /app/solver.py into the sample's checker and runs Harbor's verifier there, on its 8 CPUs).
One sample at a time (max_samples = max_sandboxes = 1), each in fresh containers; refuses to start while any
other container is running. Samples with no selected solver are skipped and listed.

    cd ~/projects/swarm-scaling-pilot2 && uv run python scripts/rescore.py '../swarm-scaling/logs/pilot3-*' --repeats 2

With --scorer URL (and SCORER_TOKEN set) every scoring is a job on the remote scorer service instead
(scorer_service.py): all are queued at once and run on its slots; no local Docker and no eval log are used.

Writes analysis/rescore/<timestamp>.jsonl (one row per sample and repeat, with the raw verifier output),
the rescore eval log under logs/rescore/<timestamp>/, and prints a table.
The table flags rows whose first-call ratio (verifier: untimed first solver call / min timed call, summed
over instances) exceeds 5 for manual review: caching results by problem identity shows up there, as can a
solver that compiles on its first call. Verifier timeouts score 1.0 and are flagged too.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import anyio
from inspect_ai import Task, eval, task_with
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from swarm_scaling import scorer_client
from swarm_scaling.algotune_devkit import SOLVER_PATH, remote_timing
from swarm_scaling.tasks import LOG_DIR, PROJECT_ROOT, algotune_task, seed_offset_id, task_names

# Untimed first solver call / min timed call, summed over instances. A solver that caches results by problem
# identity shows a large ratio; so can one that compiles or initialises lazily on its first call. Review, not scored.
REVIEW_FIRST_CALL_RATIO = 5.0


@solver
def install_selected() -> Solver:
    """Put the logged selected solver at /app/solver.py in the agent box, where finalize left it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file(SOLVER_PATH, state.metadata["rescore"]["source"])
        return state

    return solve


def eval_files(patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            files.update(path.glob("*.eval") if path.is_dir() else [path] if path.suffix == ".eval" else [])
    return sorted(files)


def validity(explanation: str) -> bool | None:
    found = re.search(r"^Validity: (True|False)$", explanation, re.M)
    return None if found is None else found.group(1) == "True"


def fmt(value: float | None) -> str:
    return f"{value:>10.4f}" if value is not None else f"{'-':>10}"


def collect(files: list[Path]) -> tuple[list[dict], list[dict], set]:
    """Rows to rescore (with the selected source), samples skipped for having none, seed offsets the logs used."""
    rows, skipped, offsets = [], [], set()
    for f in files:
        log = read_eval_log(str(f))
        meta = log.eval.metadata or {}
        # local runs record the offset; remote-scorer runs only its fingerprint
        offsets.add(meta.get("algotune_seed_offset", meta.get("seed_offset_id")))
        for s in log.samples or []:
            source = (s.metadata.get("finalize") or {}).get("selected_source")
            info = {
                "log": str(f), "run": f.parent.name, "arm": meta.get("arm"), "n": meta.get("n"),
                "split": meta.get("split", "pilot"), "task": s.id, "epoch": s.epoch,
                "original_score": {k: v.value for k, v in (s.scores or {}).items()}.get("algotune_scorer"),
            }  # fmt: skip
            (rows if source else skipped).append({**info, "source": source})
    return rows, skipped, offsets


def build(rows: list[dict], offsets: set, task_for: Callable = algotune_task) -> tuple[Task, list[Sample]]:
    """One rescore sample per row, copied from the task's own sample (verifier, seed offset, compose file).

    Raises if the logs were scored with a different seed offset, since their scores would not be comparable.
    """
    samples: list[Sample] = []
    for split in sorted({r["split"] for r in rows}):
        task = task_for(split=split)
        offset = task.metadata["algotune_seed_offset"]
        if offsets - {offset}:
            raise ValueError(f"logs were scored with seed offset(s) {sorted(offsets - {offset}, key=str)}, not this experiment's")
        by_id = {s.id: s for s in task.dataset}
        for r in (x for x in rows if x["split"] == split):
            sample = by_id[r["task"]].model_copy(deep=True)
            sample.id = f"{len(samples):03d}-{r['run']}-e{r['epoch']}"
            sample.metadata["rescore"] = r
            samples.append(sample)
    return task, samples


def result_row(sample) -> dict:
    """One output row from a rescored sample: the input row (without source) plus score, validity, raw output."""
    row = {k: v for k, v in sample.metadata["rescore"].items() if k != "source"}
    score = (sample.scores or {}).get("algotune_scorer")
    explanation = score.explanation if score else str(sample.error)
    meta = (score.metadata or {}) if score else {}
    return {
        **row, "repeat": sample.epoch, "rescore": score.value if score else None, "valid": validity(explanation or ""),
        "first_call_ratio": meta.get("first_call_ratio"), "scoring_timeout": meta.get("scoring_timeout"),
        "checker": meta.get("checker"), "verifier_output": explanation,
    }  # fmt: skip


def remote_results(rows: list[dict], offsets: set, repeats: int) -> list[dict]:
    """Score every row `repeats` times as jobs on the remote scorer (SCORER_URL / SCORER_TOKEN), all queued at once.

    Refuses logs scored with another seed offset than the service's (compared by fingerprint).
    """
    service = scorer_client.health()
    ids = {o if isinstance(o, str) else seed_offset_id(o) for o in offsets}
    if ids != {service["seed_offset_id"]}:
        raise ValueError(f"logs were scored with seed offset id(s) {sorted(ids, key=str)}, the scorer uses {service['seed_offset_id']}")
    names = task_names()
    results: list[dict] = []

    async def one(r: dict, repeat: int) -> None:
        job = {
            "kind": "score", "task": names[r["task"]], "solver_source": r["source"], "run_id": "rescore",
            "agent_id": "", "sample_id": f"{r['run']}/{r['task']}/e{r['epoch']}",
        }  # fmt: skip
        rec = await scorer_client.run_job(job)
        res = rec["result"] if rec["status"] == "done" else None
        row = {k: v for k, v in r.items() if k != "source"}
        results.append({
            **row, "repeat": repeat, "rescore": res["score"] if res else None, "valid": res["valid"] if res else None,
            "first_call_ratio": res["first_call_ratio"] if res else None,
            "scoring_timeout": res["scoring_timeout"] if res else None, "checker": remote_timing(rec),
            "verifier_output": res["verifier_stdout"] if res else rec["error"],
        })  # fmt: skip

    async def run_all() -> None:
        async with anyio.create_task_group() as tg:
            for r in rows:
                for repeat in range(1, repeats + 1):
                    tg.start_soon(one, r, repeat)

    anyio.run(run_all)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="log dirs, .eval files or globs (quote them)")
    p.add_argument("--repeats", type=int, default=1, help="scorings per solver (to measure rescore noise)")
    p.add_argument("--scorer", help="score on the remote scorer service at this URL (token: SCORER_TOKEN)")
    args = p.parse_args()

    if args.scorer:
        os.environ["SCORER_URL"] = args.scorer
    else:
        running = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
        if running:
            sys.exit(f"{len(running)} container(s) running; rescore needs Docker otherwise idle")

    rows, skipped, offsets = collect(eval_files(args.logs))
    for r in skipped:
        print(f"skipped (no selected solver): {r['run']} {r['task']} epoch {r['epoch']}")
    if not rows:
        sys.exit("nothing to rescore")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if args.scorer:
        results = remote_results(rows, offsets, args.repeats)
    else:
        task, samples = build(rows, offsets)
        (log,) = eval(
            task_with(task, dataset=MemoryDataset(samples), solver=install_selected()),
            model="mockllm/model",
            epochs=args.repeats,
            max_samples=1,
            max_sandboxes=1,
            log_dir=str(LOG_DIR / "rescore" / stamp),
            display="none",
        )
        print("status:", log.status, log.error or "", log.location)
        results = [result_row(s) for s in read_eval_log(log.location).samples or []]

    out = PROJECT_ROOT / "analysis" / "rescore" / f"{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'run':48} {'arm':12} {'n':>2} {'ep':>3} {'rep':>3} {'original':>10} {'rescore':>10} {'ratio':>7} "
          f"{'1st-call':>9} valid")  # fmt: skip
    for r in sorted(results, key=lambda r: (r["run"], r["epoch"], r["repeat"])):
        orig, new = r["original_score"], r["rescore"]
        ratio = f"{new / orig:7.3f}" if orig and new is not None else f"{'':>7}"
        first = r["first_call_ratio"]
        flags = ["REVIEW: first-call ratio > 5"] if first is not None and first > REVIEW_FIRST_CALL_RATIO else []
        flags += ["SCORING TIMEOUT"] if r["scoring_timeout"] else []
        first_s = f"{first:9.2f}" if first is not None else f"{'-':>9}"
        print(f"{r['run']:48} {r['arm']:12} {r['n']:>2} {r['epoch']:>3} {r['repeat']:>3} {fmt(orig)} {fmt(new)} {ratio} "
              f"{first_s} {r['valid']} {' '.join(flags)}")  # fmt: skip
    print(f"\nwrote {out} ({len(results)} rows; {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
