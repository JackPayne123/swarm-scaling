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
    hidden = {"A": [6.0], "B": [3.0, 2.0, 4.0], "C": [9.0]}
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
    assert kinds == ["score_dev"] * 3 + ["score"] * 5  # every pick is fixed before any hidden job
    assert sorted(s for k, s, _ in submitted if k == "score") == ["A", "B", "B", "B", "C"]  # pick 3x, the others once
    assert {(s, seed) for k, s, seed in submitted if k == "score_dev"} == {("A", 500_000), ("B", 500_000), ("C", 500_000)}
    assert "D" not in {s for _, s, _ in submitted}  # only the top 3 distinct sources go to stage 2
    assert row["pick"] == {"sha256": "sha-B", "stage1_rank": 2, "score": 3.5, "alone_score": None}
    assert row["pick_hidden_median"] == 3.0
    assert row["oracle"] == {"label": sp.ORACLE_LABEL, "sha256": "sha-C", "hidden_median": 9.0}
    assert row["oracle"]["label"].startswith("oracle bound")


def test_a_selection_runs_once_when_the_driver_and_select_ready_both_start_it(tmp_path, monkeypatch):
    """Selecting the same runs twice would double the scorer load and write two competing results."""
    import subprocess

    from select_portfolio import claim

    monkeypatch.setattr("select_portfolio.CLAIMS", tmp_path)
    assert claim(["logs/r-i1", "logs/r-i0"]) is True
    ended = subprocess.Popen(["true"])
    ended.wait()
    (next(tmp_path.iterdir())).write_text(str(ended.pid))  # the claimant has finished
    assert claim(["logs/r-i0", "logs/r-i1"]) is False  # same runs, any order: no second selection
    assert claim(["logs/r-i0"]) is True  # different runs: their own claim


def _portfolio(name: str, scores: dict[str, float]) -> dict:
    return {
        "portfolio": name, "task": "algotune/qp", "arm": "team", "protocol": "diversified-v7", "runs": [name],
        "priority_indexes": [0], "sources": {f"sha-{x}": x for x in scores},
        "candidates": [{"sha256": f"sha-{x}", "valid": True, "speedup": s, "run": name, "agent_id": "agent_0",
                        "kind": "published"} for x, s in scores.items()],
    }  # fmt: skip


def _done(score: float, alone: float | None = None, valid: bool = True) -> dict:
    return {"status": "done", "job_id": "j", "result": {"score": score, "alone_score": alone, "valid": valid,
            "scoring_timeout": False, "first_call_ratio": 1.0, "reference_inflation": 1.0}}  # fmt: skip


@pytest.mark.asyncio
async def test_a_source_invalid_on_the_validity_pass_is_dropped_and_the_next_stage_one_source_backfills() -> None:
    # v6 least_squares: a subsampling solver failed 1.2% of instances, passed 120 dev instances and scored 1.00 hidden.
    portfolio = _portfolio("v7-qp-r1-team", {"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0, "E": 5.0})
    invalid = {"A": 14, "C": 0}  # A fails 14 of 1,000; every other source passes
    seen = []

    async def run_job(job: dict) -> dict:
        src = job["solver_source"]
        seen.append((job["kind"], src, job.get("n"), job.get("seed")))
        if job["kind"] == "validity":
            bad = invalid.get(src, 0)
            return {"status": "done", "job_id": "v", "result": {"valid": bad == 0, "n": job["n"], "checked": job["n"],
                    "n_invalid": bad, "invalid_instances": list(range(bad)), "errors": [], "run_s": 1.0}}  # fmt: skip
        return _done({"B": 4.0, "C": 5.0, "D": 3.0}[src])

    (row,) = await sp.select([portfolio], {"algotune/qp": "qp"}, run_job=run_job, validity_n=1000)
    assert [(v["sha256"], v["valid"], v["n_invalid"]) for v in row["validity"]] == [
        ("sha-A", False, 14), ("sha-B", True, 0), ("sha-C", True, 0), ("sha-D", True, 0)]  # fmt: skip
    assert {(n, seed) for k, _, n, seed in seen if k == "validity"} == {(1000, 600_000)}  # one instance set for all
    assert [e["sha256"] for e in row["stage2"]] == ["sha-B", "sha-C", "sha-D"]  # A never timed, D backfilled, E untried
    assert row["pick"]["sha256"] == "sha-C"
    assert seen.index(next(s for s in seen if s[0] == "score_dev")) > max(i for i, s in enumerate(seen) if s[0] == "validity")
    seen.clear()  # a task whose is_solution is slow gets its own size, the same for every arm
    (row,) = await sp.select([portfolio], {"algotune/qp": "qp"}, run_job=run_job, validity_n=1000, validity_n_by_task={"qp": 200})
    assert row["validity_n"] == 200 and {n for k, _, n, _ in seen if k == "validity"} == {200}


@pytest.mark.asyncio
async def test_v7_ranks_stage_two_and_hidden_by_the_alone_score_and_keeps_algotunes_score_beside_it() -> None:
    # v6 pca r2: the pick won stage 2 on reference inflation (13.08 vs 12.12) and then scored 7.25 and 7.06 on hidden.
    portfolio = _portfolio("v7-pca-r2-team", {"A": 9.0, "B": 8.0})
    interleaved, alone = {"A": 13.08, "B": 12.12}, {"A": 11.9, "B": 12.0}

    async def run_job(job: dict) -> dict:
        src = job["solver_source"]
        return _done(interleaved[src], alone[src])

    (row,) = await sp.select([portfolio], {"algotune/qp": "qp"}, run_job=run_job, primary="alone")
    assert row["primary_score"] == "alone" and row["pick"]["sha256"] == "sha-B"
    assert row["pick_hidden_median"] == 12.0
    assert row["hidden"]["sha-B"]["median_algotune"] == 12.12 and row["hidden"]["sha-B"]["median_alone"] == 12.0
    (row,) = await sp.select([portfolio], {"algotune/qp": "qp"}, run_job=run_job)  # v6 logs: AlgoTune's score
    assert row["pick"]["sha256"] == "sha-A" and row["pick_hidden_median"] == 13.08


@pytest.mark.asyncio
async def test_each_portfolio_is_written_when_it_is_done_and_every_job_has_a_stable_request_id() -> None:
    # v6: selection results were written only at the end, and a lost network killed whole selections.
    import anyio

    slow, fast = _portfolio("slow", {"A": 2.0}), _portfolio("fast", {"B": 2.0})
    written, ids = [], []

    async def run_job(job: dict) -> dict:
        ids.append(job["request_id"])
        await anyio.sleep(0.3 if job["solver_source"] == "A" else 0.0)
        return _done(2.0)

    rows = await sp.select([slow, fast], {"algotune/qp": "qp"}, run_job=run_job, on_row=lambda r: written.append(r["portfolio"]),
                           request_prefix="claim")  # fmt: skip
    assert written == ["fast", "slow"] and [r["portfolio"] for r in rows] == ["slow", "fast"]
    assert len(ids) == len(set(ids)) == 8  # 2 x (1 stage-2 + 3 hidden repeats)
    assert "claim/fast/sha-B/score/r2" in ids


@pytest.mark.asyncio
async def test_hidden_jobs_run_exclusively_alternating_arms_and_all_in_the_scorer_session_selection_started_on() -> None:
    # v7 calibration: the same ode solver scored up to 1.17x apart by slot and load in one session; a comparison's hidden
    # scores now run with the other slots idle, on one slot, in one session (one machine type).
    def portfolio(name: str) -> dict:
        return {"portfolio": name, "task": "algotune/qp", "arm": "team", "protocol": "diversified-v7", "runs": [name],
                "priority_indexes": [0], "sources": {"sha-A": "A"},
                "candidates": [{"sha256": "sha-A", "valid": True, "speedup": 2.0, "run": name, "agent_id": "agent_0",
                                "kind": "published"}]}  # fmt: skip

    jobs = []

    def fake(session_of_hidden: str):
        async def run_job(job: dict) -> dict:
            jobs.append(job)
            result = {"score": 2.0, "alone_score": 2.0, "valid": True, "scoring_timeout": False, "first_call_ratio": 1.0,
                      "reference_inflation": 1.0}  # fmt: skip
            return {"status": "done", "result": result, "job_id": "j",
                    "session": session_of_hidden if job["kind"] == "score" else "s1"}  # fmt: skip

        return run_job

    scorer = {"session": "s1", "machine_type": "c7a.8xlarge", "cpu_model": "AMD EPYC 9R14", "slots": 3, "exclusive_slot": 0}
    team, indep = portfolio("v7-qp-r1-divteam"), portfolio("v7-qp-r1-div")
    rows = await sp.select([team, indep], {"algotune/qp": "qp"}, run_job=fake("s1"), exclusive_hidden=True, scorer=scorer)
    assert {j["kind"]: j.get("exclusive", False) for j in jobs} == {"score_dev": False, "score": True}
    assert all(r["exclusive_hidden"] is True and r["scorer"] == scorer for r in rows)
    assert [r["job"]["session"] for r in rows[0]["hidden"]["sha-A"]["scores"]] == ["s1"] * 3
    # smoke scorer: machine speed drifted ~5% over tens of minutes, so the two arms' repeats alternate
    assert [j["sample_id"].split("/")[0] for j in jobs if j["kind"] == "score"] == ["v7-qp-r1-divteam", "v7-qp-r1-div"] * 3
    with pytest.raises(RuntimeError, match="scorer sessions"):  # a restarted service: fail, write no mixed row
        await sp.select([portfolio("p2")], {"algotune/qp": "qp"}, run_job=fake("s2"), exclusive_hidden=True, scorer=scorer)
