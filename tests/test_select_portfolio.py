"""scripts/select_portfolio.py: two-stage selection over a portfolio of runs, with a fake scorer (no service)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("select_portfolio", ROOT / "scripts/select_portfolio.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def test_runs_are_grouped_into_portfolios_by_plan_name() -> None:
    assert sp.portfolio_key("v6-qp-r1-div-i2", "independent") == "v6-qp-r1-div"
    assert sp.portfolio_key("v6-qp-r1-divteam", "team") == "v6-qp-r1-divteam"
    with pytest.raises(ValueError, match="-i<k>"):
        sp.portfolio_key("v6-qp-r1-div", "independent")


@pytest.mark.asyncio
async def test_the_pick_comes_from_stage_two_and_hidden_scores_only_inform_the_labelled_oracle() -> None:
    # A wins stage 1 but is slower on the 100-instance verifier pass; C is fastest there but invalid; B is the pick.
    # C has the best hidden score: that is reported as the oracle bound, never chosen.
    def cand(source: str, speedup: float | None, run: str, valid: bool = True) -> dict:
        return {"sha256": f"sha-{source}", "valid": valid, "speedup": speedup, "run": run, "agent_id": "agent_0",
                "kind": "published"}  # fmt: skip

    portfolio = {
        "portfolio": "v6-qp-r1-div", "task": "algotune/qp", "arm": "independent", "protocol": "diversified",
        "runs": ["v6-qp-r1-div-i0", "v6-qp-r1-div-i1"], "priority_indexes": [0, 1],
        "sources": {f"sha-{x}": x for x in "ABCDE"},
        "candidates": [
            cand("A", 5.0, "v6-qp-r1-div-i0"), cand("A", 4.5, "v6-qp-r1-div-i1"),  # one source, two runs
            cand("B", 4.0, "v6-qp-r1-div-i1"), cand("C", 3.0, "v6-qp-r1-div-i0"), cand("D", 2.0, "v6-qp-r1-div-i1"),
            cand("E", None, "v6-qp-r1-div-i0", valid=False),
        ],
    }  # fmt: skip
    stage2 = {"A": (2.0, True), "B": (3.5, True), "C": (4.0, False)}
    hidden = {"A": [6.0, 6.0, 6.0], "B": [3.0, 2.0, 4.0], "C": [9.0, 9.0, 9.0]}
    submitted = []

    async def run_job(job: dict) -> dict:
        src = job["solver_source"]
        submitted.append((job["kind"], src, job.get("seed")))
        if job["kind"] == "score_dev":
            score, valid = stage2[src]
        else:
            score, valid = hidden[src].pop(0), True
        result = {"score": score, "valid": valid, "scoring_timeout": False, "first_call_ratio": 1.0, "reference_inflation": 1.0}
        return {"status": "done", "result": result, "job_id": "j"}

    (row,) = await sp.select([portfolio], {"algotune/qp": "qp"}, run_job=run_job, seed=500_000)

    assert [(e["rank"], e["sha256"], e["dev_speedup"]) for e in row["stage1"]] == [
        (1, "sha-A", 5.0), (2, "sha-B", 4.0), (3, "sha-C", 3.0), (4, "sha-D", 2.0)]  # fmt: skip
    kinds = [k for k, _, _ in submitted]
    assert kinds == ["score_dev"] * 3 + ["score"] * 9  # every pick is fixed before any hidden job
    assert {(s, seed) for k, s, seed in submitted if k == "score_dev"} == {("A", 500_000), ("B", 500_000), ("C", 500_000)}
    assert "D" not in {s for _, s, _ in submitted}  # only the top 3 distinct sources go to stage 2
    assert row["pick"] == {"sha256": "sha-B", "stage1_rank": 2, "score": 3.5}
    assert row["pick_hidden_median"] == 3.0
    assert row["oracle"] == {"label": sp.ORACLE_LABEL, "sha256": "sha-C", "hidden_median": 9.0}
    assert row["oracle"]["label"].startswith("oracle bound")
