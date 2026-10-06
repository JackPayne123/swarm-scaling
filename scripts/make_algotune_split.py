"""Select AlgoTune tasks with headroom and split them into pilot and held-out sets.

Rule (fixed before any run of ours): from AlgoTune's published per-model results, keep a task when
the best model's final speedup is in [1.5, 10] and the median across models is >= 1.05. Tasks whose
best model never exceeds 1.1x are excluded. A model with no valid result ("N/A") counts as 1.0,
AlgoTune's own score for an invalid or missing solution. The kept tasks are sorted by name and a
fixed-seed shuffle draws the pilot set; the rest are held out.

Usage: uv run python scripts/make_algotune_split.py
"""

import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "research/task-data/algotune_agent_summary.json"
HARBOR_META = ROOT / "research/task-data/algotune_harbor_meta.json"
OUT = ROOT / "data/algotune_split.json"

BEST_RANGE = (1.5, 10.0)
MIN_MEDIAN = 1.05
NEVER_ABOVE = 1.1
PILOT_SIZE = 6
SEED = 20261005  # fixed once, never re-rolled


def speedup(entry: dict) -> float:
    try:
        return float(entry["final_speedup"])
    except ValueError:  # "N/A"
        return 1.0


def headroom_stats(summary: dict) -> dict[str, tuple[float, float]]:
    """task -> (best, median) final speedup across every model in the summary."""
    models = sorted({m for task in summary.values() for m in task})
    stats = {}
    for task, per_model in summary.items():
        values = [speedup(per_model[m]) if m in per_model else 1.0 for m in models]
        stats[task] = (max(values), statistics.median(values))
    return stats


def select_tasks(stats: dict[str, tuple[float, float]]) -> list[str]:
    lo, hi = BEST_RANGE
    return sorted(
        task
        for task, (best, median) in stats.items()
        if lo <= best <= hi and median >= MIN_MEDIAN and best > NEVER_ABOVE
    )


def split_tasks(tasks: list[str], pilot_size: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic split: the same task list and seed always give the same pilot set."""
    ordered = sorted(tasks)
    pilot = sorted(random.Random(seed).sample(ordered, pilot_size))
    return pilot, [t for t in ordered if t not in pilot]


def harbor_name(task: str) -> str:
    return task.replace("_", "-").lower()


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    stats = headroom_stats(summary)
    selected = select_tasks(stats)
    pilot, heldout = split_tasks(selected, PILOT_SIZE, SEED)

    harbor_names = {t["task_name"] for t in json.loads(HARBOR_META.read_text())}
    missing = [t for t in selected if harbor_name(t) not in harbor_names]
    assert not missing, f"no Harbor task for {missing}"

    result = {
        "rule": (
            f"best-model final speedup in {list(BEST_RANGE)} and median across models >= {MIN_MEDIAN}; "
            f"tasks whose best model never exceeds {NEVER_ABOVE}x are excluded; "
            "N/A (no valid result) counts as 1.0; selected tasks sorted by name, "
            f"pilot = random.Random(seed).sample(sorted_selected, {PILOT_SIZE}), held-out = the rest"
        ),
        "seed": SEED,
        "source": {
            "file": str(SUMMARY.relative_to(ROOT)),
            "sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
            "origin": "github.com/oripress/AlgoTune reports/agent_summary.json",
            "n_models": len({m for task in summary.values() for m in task}),
        },
        "n_selected": len(selected),
        "never_above_1.1x": sorted(t for t, (best, _) in stats.items() if best <= NEVER_ABOVE),
        "pilot": pilot,
        "heldout": heldout,
        "stats": {t: {"best": round(stats[t][0], 3), "median": round(stats[t][1], 3)} for t in selected},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected {len(selected)}: pilot {pilot}")


if __name__ == "__main__":
    main()
