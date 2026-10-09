"""Re-time every run's selected AlgoTune solver in one consistent serial pass.

For each sample in the given logs, takes state.metadata["finalize"]["selected_source"] and scores it again
through the normal scoring path: `algotune_task` (same patched verifier, same secret seed offset) and
`algotune_scorer` (copies /app/solver.py into the sample's checker and runs Harbor's verifier there, CPUs 8-15).
One sample at a time (max_samples = max_sandboxes = 1), each in fresh containers; refuses to start while any
other container is running. Samples with no selected solver are skipped and listed.

    cd ~/projects/swarm-scaling-pilot2 && uv run python scripts/rescore.py '../swarm-scaling/logs/pilot3-*' --repeats 2

Writes analysis/rescore/<timestamp>.jsonl (one row per sample and repeat, with the raw verifier output),
the rescore eval log under logs/rescore/<timestamp>/, and prints a table.
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from inspect_ai import Task, eval, task_with
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from swarm_scaling.algotune_devkit import SOLVER_PATH
from swarm_scaling.tasks import LOG_DIR, PROJECT_ROOT, algotune_task


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
        offsets.add(meta.get("algotune_seed_offset"))
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
    return {
        **row, "repeat": sample.epoch, "rescore": score.value if score else None, "valid": validity(explanation or ""),
        "checker": (score.metadata or {}).get("checker") if score else None, "verifier_output": explanation,
    }  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="log dirs, .eval files or globs (quote them)")
    p.add_argument("--repeats", type=int, default=1, help="scorings per solver (to measure rescore noise)")
    args = p.parse_args()

    running = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
    if running:
        sys.exit(f"{len(running)} container(s) running; rescore needs Docker otherwise idle")

    rows, skipped, offsets = collect(eval_files(args.logs))
    for r in skipped:
        print(f"skipped (no selected solver): {r['run']} {r['task']} epoch {r['epoch']}")
    if not rows:
        sys.exit("nothing to rescore")
    task, samples = build(rows, offsets)

    stamp = time.strftime("%Y%m%dT%H%M%S")
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

    out = PROJECT_ROOT / "analysis" / "rescore" / f"{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = [result_row(s) for s in read_eval_log(log.location).samples or []]
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'run':48} {'arm':12} {'n':>2} {'ep':>3} {'rep':>3} {'original':>10} {'rescore':>10} {'ratio':>7} valid")
    for r in sorted(results, key=lambda r: (r["run"], r["epoch"], r["repeat"])):
        orig, new = r["original_score"], r["rescore"]
        ratio = f"{new / orig:7.3f}" if orig and new is not None else f"{'':>7}"
        print(f"{r['run']:48} {r['arm']:12} {r['n']:>2} {r['epoch']:>3} {r['repeat']:>3} {fmt(orig)} {fmt(new)} {ratio} {r['valid']}")
    print(f"\nwrote {out} ({len(results)} rows; {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
