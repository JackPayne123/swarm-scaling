import importlib.util
import json
from pathlib import Path

from swarm_scaling.algotune_devkit import DevResult, pick_best
from swarm_scaling.swarm import Candidate

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_algotune_split", ROOT / "scripts/make_algotune_split.py")
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)


def summary(**per_task: list[str]) -> dict:
    return {t: {f"m{i}": {"final_speedup": v} for i, v in enumerate(vals)} for t, vals in per_task.items()}


def selected(**per_task: list[str]) -> list[str]:
    return split.select_tasks(split.headroom_stats(summary(**per_task)))


def test_best_model_range_is_inclusive_and_caps_heavy_tails():
    # ode_seirs reached 3,084x for one model; a single outlier must not make a task a headroom task.
    s = {"low": ["1.49"] * 3, "lo_edge": ["1.5", "1.2", "1.2"], "hi_edge": ["10", "1.2", "1.2"], "too_high": ["10.01", "1.2", "1.2"]}
    assert selected(**s) == ["hi_edge", "lo_edge"]


def test_median_must_show_that_models_generally_gain():
    # One model at 3x with the rest at 1.0 is a lucky solution, not headroom.
    assert selected(lucky=["3.0", "1.0", "1.0"], broad=["3.0", "1.06", "1.06"]) == ["broad"]


def test_missing_results_count_as_one():
    # N/A is an invalid or missing solution, which AlgoTune scores 1.0, so it drags the median down.
    assert selected(mostly_na=["2.0", "N/A", "N/A"], solved=["2.0", "1.1", "N/A"]) == ["solved"]


def test_split_is_deterministic_disjoint_and_independent_of_input_order():
    tasks = [f"t{i:02d}" for i in range(32)]
    pilot, heldout = split.split_tasks(tasks, 6, seed=7)
    assert split.split_tasks(list(reversed(tasks)), 6, seed=7) == (pilot, heldout)
    assert len(pilot) == 6 and not set(pilot) & set(heldout)
    assert sorted(pilot + heldout) == tasks
    assert split.split_tasks(tasks, 6, seed=8)[0] != pilot  # the seed is what fixes the split


def test_committed_split_is_what_the_rule_produces():
    # The split is frozen before the main run; this fails if the data, the rule or the file drift apart.
    committed = json.loads(split.OUT.read_text())
    stats = split.headroom_stats(json.loads(split.SUMMARY.read_text()))
    chosen = split.select_tasks(stats)
    assert (committed["pilot"], committed["heldout"]) == split.split_tasks(chosen, split.PILOT_SIZE, committed["seed"])
    assert committed["seed"] == split.SEED and committed["n_selected"] == len(chosen)


def candidate(agent_id: str, published_at: float, kind: str = "published") -> Candidate:
    return Candidate(agent_id, "m", f"/workspace/{agent_id}", "", published_at, kind)


def result(c: Candidate, speedup: float | None, valid: bool = True) -> DevResult:
    return DevResult(c, valid, speedup)


def test_selector_takes_the_fastest_correct_candidate():
    a, b, c = candidate("a", 1), candidate("b", 2), candidate("c", 3)
    # c is fastest but incorrect: an invalid solver scores 1.0 in the final evaluation, so it must never win.
    assert pick_best([result(a, 2.0), result(b, 3.0), result(c, 9.0, valid=False)]).candidate is b


def test_selector_ties_go_to_the_earliest_published():
    early, late = candidate("early", 1), candidate("late", 2)
    assert pick_best([result(late, 2.0), result(early, 2.0)]).candidate is early


def test_selector_ties_prefer_published_over_final_workspace():
    final, published = candidate("a", 5, "final_workspace"), candidate("b", 5)
    assert pick_best([result(final, 2.0), result(published, 2.0)]).candidate is published


def test_selector_returns_none_when_nothing_is_correct():
    a = candidate("a", 1)
    assert pick_best([result(a, 5.0, valid=False), result(a, None)]) is None
    assert pick_best([]) is None
