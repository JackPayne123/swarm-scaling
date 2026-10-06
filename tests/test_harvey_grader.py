"""Host-side Harvey grader, with the LiteLLM judge call and the sandbox stubbed (no network, no Docker)."""

import asyncio
import json
import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm fetches its price map at import otherwise

import pytest

from swarm_scaling import harvey_grader as hg
from swarm_scaling.harvey_grader import Criterion, judge_report

REPORT_PATH = "/workspace/output/red-flags-report.md"
REPORT = "# Red flags\n\nOIBDA is overstated by $303 million."


def reply(content: dict) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=1000, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=900, cache_creation_tokens=0)
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))], usage=usage)


def verdict(passed: bool) -> dict:
    return {"score": "yes" if passed else "no", "reasoning": "r"}


def criterion_names(kwargs: dict) -> list[str]:
    """The criterion ids named in a call (they appear only after the cached report block)."""
    text = kwargs["messages"][1]["content"][1]["text"]
    return [line.split("'")[1] for line in text.splitlines() if line.startswith("- '")]


@pytest.fixture(autouse=True)
def no_cost_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hg.litellm, "completion_cost", lambda completion_response: 0.01)


def judge(criteria: list[Criterion], **kwargs) -> tuple[dict, hg.JudgeUsage]:
    defaults = {"model": "anthropic/claude-sonnet-4-6", "semaphore": asyncio.Semaphore(4), "backoff_sec": 0}
    return asyncio.run(judge_report(REPORT, REPORT_PATH, criteria, **(defaults | kwargs)))


def test_transient_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rate limit or a malformed reply on one of ~1,000 calls must not cost the criterion.
    outcomes = [RuntimeError("429"), reply({"wrong": "shape"}), reply(verdict(True))]

    async def flaky(**kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(hg.litellm, "acompletion", flaky)
    results, usage = judge([Criterion("c-001", "d")])
    assert results == {"c-001": {"passed": True, "reasoning": "r"}}
    assert usage.calls == 2 and usage.cost_usd == pytest.approx(0.02)  # the unparseable reply was paid for too


def test_persistent_failure_is_errored_non_pass_and_others_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    # rewardkit aborted the whole scoring on one failed call. Here only that criterion is lost, as a non-pass.
    attempts = {"c-002": 0}

    async def stub(**kwargs):
        (name,) = criterion_names(kwargs)
        if name == "c-002":
            attempts[name] += 1
            raise RuntimeError("overloaded")
        return reply(verdict(True))

    monkeypatch.setattr(hg.litellm, "acompletion", stub)
    results, _ = judge([Criterion(f"c-00{i}", "d") for i in (1, 2, 3)], attempts=4)
    assert attempts["c-002"] == 4
    assert results["c-002"] == {"passed": False, "reasoning": "", "error": "RuntimeError: overloaded"}
    assert results["c-001"]["passed"] and results["c-003"]["passed"]


class FakeSandbox:
    """Serves only the deliverable and records every path read; holds a decoy rubric that must never be read."""

    def __init__(self) -> None:
        self.files = {REPORT_PATH: REPORT, "/tests/judge.toml": "decoy"}
        self.reads: list[str] = []

    async def read_file(self, path: str, text: bool = True) -> str:
        self.reads.append(path)
        return self.files[path]


def score_sample(monkeypatch: pytest.MonkeyPatch, tmp_path, verdicts: dict[str, bool]) -> tuple:
    rubric = "".join(
        f'[[criterion]]\nname = "{name}"\ndescription = "d"\ntype = "binary"\nfiles = ["{REPORT_PATH}"]\n\n' for name in verdicts
    )
    (tmp_path / "judge.toml").write_text(rubric)
    fake = FakeSandbox()
    monkeypatch.setattr(hg, "sandbox", lambda: fake)

    async def stub(**kwargs):
        (name,) = criterion_names(kwargs)
        return reply(verdict(verdicts[name]))

    monkeypatch.setattr(hg.litellm, "acompletion", stub)
    state = SimpleNamespace(metadata={"tests_dir": str(tmp_path)})
    return asyncio.run(hg.harvey_grader(backoff_sec=0)(state, None)), fake


def test_score_is_the_criterion_fraction_and_all_pass_is_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The all-pass reward is 0 for nearly every run on a 438-1,114 criterion rubric, so the headline is the fraction.
    score, _ = score_sample(monkeypatch, tmp_path, {"c-001": True, "c-002": False, "c-003": True, "c-004": True})
    assert score.value == 0.75
    assert (score.metadata["n_passed"], score.metadata["n_criteria"], score.metadata["n_errored"]) == (3, 4, 0)
    assert score.metadata["all_pass"] is False
    assert score.metadata["judge_usage"]["calls"] == 4

    perfect, _ = score_sample(monkeypatch, tmp_path, {"c-001": True, "c-002": True})
    assert perfect.value == 1.0 and perfect.metadata["all_pass"] is True
    samples = [SimpleNamespace(score=s) for s in (score, perfect)]
    assert hg.all_pass_rate()(samples) == 0.5


def test_rubric_comes_from_the_host_never_the_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    score, fake = score_sample(monkeypatch, tmp_path, {"c-001": True})
    assert fake.reads == [REPORT_PATH]
    assert list(score.metadata["criteria"]) == ["c-001"]


def test_cached_prefix_is_byte_identical_across_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prompt caching only pays if everything up to the cache breakpoint is the same for every criterion.
    calls: list[dict] = []

    async def stub(**kwargs):
        calls.append(kwargs)
        return reply(verdict(True))

    monkeypatch.setattr(hg.litellm, "acompletion", stub)
    judge([Criterion("c-001", "first"), Criterion("c-002", "second"), Criterion("c-003", "third")])
    prefixes = {json.dumps([c["messages"][0], c["messages"][1]["content"][0]]) for c in calls}
    assert len(calls) == 3 and len(prefixes) == 1
    (prefix,) = prefixes
    assert json.dumps(REPORT)[1:-1] in prefix and '"cache_control": {"type": "ephemeral"}' in prefix
    assert not any(word in prefix for word in ("first", "second", "third"))
    assert len({json.dumps(c["response_format"]) for c in calls}) == 1  # one structured-output grammar to compile

    calls.clear()  # LiteLLM turns cache_control into an explicit Gemini context cache; Gemini caches prefixes itself
    judge([Criterion("c-001", "first")], model="gemini/gemini-3.1-pro-preview")
    assert "cache_control" not in json.dumps(calls[0]["messages"])


def test_batches_map_each_result_to_its_criterion(monkeypatch: pytest.MonkeyPatch) -> None:
    # The judge may list results in any order; each verdict must land on the id it names.
    expected = {"c-001": True, "c-002": False, "c-003": False, "c-004": True, "c-005": True}
    batches: list[list[str]] = []

    async def stub(**kwargs):
        names = criterion_names(kwargs)
        batches.append(names)
        if len(names) == 1:  # a batch of one uses the individual-mode schema
            return reply(verdict(expected[names[0]]))
        return reply({"results": [{"id": n, **verdict(expected[n])} for n in reversed(names)]})

    monkeypatch.setattr(hg.litellm, "acompletion", stub)
    results, _ = judge([Criterion(n, "d") for n in expected], batch_size=2)
    assert sorted(batches) == [["c-001", "c-002"], ["c-003", "c-004"], ["c-005"]]
    assert {n: r["passed"] for n, r in results.items()} == expected
