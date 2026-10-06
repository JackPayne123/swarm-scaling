"""Host-side grader for the Harvey LAB diligence tasks.

The deliverable is read out of the sandbox, the rubric is read from the task's tests/judge.toml on the
host, and the LLM judge runs on the host, so the agent container needs no network and never holds the
rubric.

The judge reproduces harbor-rewardkit 0.1.4's LLM judge (rewardkit/judges.py and prompts/llm.md,
Apache-2.0; the prompt text, criterion line, example, response schema, score normalisation, LiteLLM call
parameters and the 1 MB file limit are copied from it). Differences, each so the score survives or costs less:
- Message order. rewardkit puts the criteria in the system prompt and the file in the user turn, so no two
  calls share a prefix. Here the system prompt is the template's instruction line, then the user turn holds
  the report and then the criterion. The prefix (instruction + report) is byte-identical across criteria,
  so provider prompt caching applies: explicit cache_control for Anthropic; OpenAI and Gemini cache
  prefixes automatically (cache_control would make LiteLLM create a Gemini context cache, so it is not sent).
- Failures. rewardkit retries only unparseable replies, and any API error aborts the whole run. Here every
  call gets `attempts` tries with exponential backoff; a criterion that still fails scores as non-pass and is
  counted as errored, as Harvey's native harness does.
- Batching (`batch_size` > 1, ours). k criteria per call with a fixed schema (an array of {id, score,
  reasoning}). rewardkit's batched schema keys on criterion names, so every call would compile a new
  grammar and Anthropic limits compilations to 20 per minute. Batching cuts calls and report re-reads
  k-fold, but criteria judged together can influence each other and Harvey's protocol is one call per
  criterion, so batch_size = 1 is the default and any other value needs a parity check against it.
- A missing deliverable or one over 1 MB scores every criterion non-pass without calling the judge
  (rewardkit judges a "[not found]" / "[skipped: file too large]" placeholder).
"""

import asyncio
import json
import logging
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import litellm
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, mean, metric, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

logger = logging.getLogger(__name__)

DEFAULT_JUDGE = "anthropic/claude-sonnet-4-6"  # as published in Harvey's judge.toml
SYSTEM_PROMPT = "You are an evaluation judge. Evaluate the provided file contents against the following criteria."
MAX_FILE_BYTES = 1024 * 1024

_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "string", "enum": ["yes", "no"]}, "reasoning": {"type": "string"}},
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str


def load_rubric(tests_dir: Path) -> tuple[str, list[Criterion]]:
    """The judged file's container path and the criteria, from a task's tests/judge.toml on the host."""
    rubric = tomllib.loads((tests_dir / "judge.toml").read_text())["criterion"]
    files = {tuple(c["files"]) for c in rubric}
    assert len(files) == 1 and len(next(iter(files))) == 1, f"expected one judged file shared by every criterion, got {files}"
    assert all(c["type"] == "binary" for c in rubric)
    return files.pop()[0], [Criterion(c["name"], c["description"]) for c in rubric]


def criteria_block(criteria: list[Criterion]) -> str:
    lines = [f"- '{c.name}': {c.description} (score: \"yes\" or \"no\")" for c in criteria]
    lines += ["", "Respond with a JSON object. Example:"]
    if len(criteria) == 1:
        example: dict[str, Any] = {"score": 1, "reasoning": "..."}
    else:
        example = {"results": [{"id": c.name, "score": 1, "reasoning": "..."} for c in criteria]}
    lines.append(json.dumps(example, indent=2))
    return "\n".join(lines)


def build_messages(path: str, report: str, criteria: list[Criterion], cache: bool) -> list[dict[str, Any]]:
    """Shared prefix (system prompt + report block, cache breakpoint on the report), then the criteria."""
    report_block: dict[str, Any] = {"type": "text", "text": f"--- {path} ---\n{report}"}
    if cache:
        report_block["cache_control"] = {"type": "ephemeral"}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [report_block, {"type": "text", "text": criteria_block(criteria)}]},
    ]


def response_schema(n_criteria: int) -> dict[str, Any]:
    if n_criteria == 1:
        return _ENTRY_SCHEMA
    item = {
        **_ENTRY_SCHEMA,
        "properties": {"id": {"type": "string"}, **_ENTRY_SCHEMA["properties"]},
        "required": ["id", "score", "reasoning"],
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
        "additionalProperties": False,
    }


def _passed(raw: Any) -> bool:
    """rewardkit's Binary.normalize."""
    if isinstance(raw, str):
        return raw.strip().lower() in ("yes", "true", "1")
    return bool(raw)


def parse_response(text: str, criteria: list[Criterion]) -> dict[str, dict[str, Any]]:
    """One {passed, reasoning} per criterion; raises if any criterion is missing, duplicated or unknown."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if not (fenced or braces):
        raise ValueError(f"Could not parse JSON from judge response: {text[:200]}")
    data = json.loads((fenced or braces).group(1 if fenced else 0))
    if len(criteria) == 1:
        entries = {criteria[0].name: data}
    else:
        ids = [e["id"] for e in data["results"]]
        if sorted(ids) != sorted(c.name for c in criteria):
            raise ValueError(f"judge returned ids {ids}, expected {[c.name for c in criteria]}")
        entries = {e["id"]: e for e in data["results"]}
    return {c.name: {"passed": _passed(entries[c.name]["score"]), "reasoning": entries[c.name]["reasoning"]} for c in criteria}


@dataclass
class JudgeUsage:
    calls: int = 0  # responses received, including ones whose content failed to parse
    input_tokens: int = 0  # includes cached and cache-write tokens
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None once LiteLLM's price map cannot price a call

    def add(self, response: Any) -> None:
        usage = response.usage
        details = usage.prompt_tokens_details
        self.calls += 1
        self.input_tokens += usage.prompt_tokens
        self.cached_input_tokens += (details and details.cached_tokens) or 0
        self.cache_write_tokens += (details and details.cache_creation_tokens) or 0
        self.output_tokens += usage.completion_tokens
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # model missing from the price map; tokens are still logged
            cost = None
        self.cost_usd = None if cost is None or self.cost_usd is None else self.cost_usd + cost


async def judge_report(
    report: str,
    path: str,
    criteria: list[Criterion],
    *,
    model: str,
    semaphore: asyncio.Semaphore,
    batch_size: int = 1,
    attempts: int = 4,
    backoff_sec: float = 2.0,
    timeout: int = 300,
    reasoning_effort: str = "medium",
) -> tuple[dict[str, dict[str, Any]], JudgeUsage]:
    """Judge every criterion; a criterion whose call fails `attempts` times gets passed=False and an error."""
    usage = JudgeUsage()
    cache = model.startswith("anthropic/")

    async def judge_chunk(chunk: list[Criterion]) -> dict[str, dict[str, Any]]:
        messages = build_messages(path, report, chunk, cache)
        response_format = {"type": "json_schema", "json_schema": {"name": "judge_response", "schema": response_schema(len(chunk)), "strict": True}}
        error = ""
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(backoff_sec * 2 ** (attempt - 1))
            try:
                async with semaphore:
                    response = await litellm.acompletion(
                        model=model, messages=messages, response_format=response_format, timeout=timeout, reasoning_effort=reasoning_effort
                    )
                usage.add(response)
                return parse_response(response.choices[0].message.content, chunk)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.warning("judge call for %s failed (attempt %d/%d): %s", [c.name for c in chunk], attempt + 1, attempts, error)
        return {c.name: {"passed": False, "reasoning": "", "error": error} for c in chunk}

    chunks = [criteria[i : i + batch_size] for i in range(0, len(criteria), batch_size)]
    results: dict[str, dict[str, Any]] = {}
    for chunk_results in await asyncio.gather(*(judge_chunk(chunk) for chunk in chunks)):
        results |= chunk_results
    return results, usage


@metric
def all_pass_rate() -> Metric:
    """Share of samples where every criterion passed (rewardkit's all_pass reward)."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:  # Inspect calls metrics with no scores at the start of a run
            return float("nan")
        return sum(s.score.metadata["all_pass"] for s in scores) / len(scores)

    return compute


@scorer(metrics=[mean(), all_pass_rate()])
def harvey_grader(
    judge: str = DEFAULT_JUDGE,
    batch_size: int = 1,
    max_concurrency: int = 8,
    attempts: int = 4,
    backoff_sec: float = 2.0,
    judge_timeout: int = 300,
    reasoning_effort: str = "medium",
) -> Scorer:
    """Score = criterion_fraction (passed / all criteria, errored counted as non-pass); details in metadata.

    `max_concurrency` bounds in-flight judge calls across every sample this scorer grades. `judge_timeout`
    is per call (rewardkit's default). Inspect caps the whole scoring step at half the sample time limit.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        path, criteria = load_rubric(Path(state.metadata["tests_dir"]))
        try:
            report = await sandbox().read_file(path)
        except FileNotFoundError:
            report = None
        if report is None or len(report.encode()) > MAX_FILE_BYTES:
            problem = "missing" if report is None else "over 1 MB"
            results = {c.name: {"passed": False, "reasoning": f"deliverable {problem}, not judged"} for c in criteria}
            usage = JudgeUsage()
        else:
            results, usage = await judge_report(
                report, path, criteria, model=judge, semaphore=semaphore, batch_size=batch_size,
                attempts=attempts, backoff_sec=backoff_sec, timeout=judge_timeout, reasoning_effort=reasoning_effort,
            )  # fmt: skip

        n = len(criteria)
        n_passed = sum(r["passed"] for r in results.values())
        errors = [r["error"] for r in results.values() if "error" in r]
        if len(errors) == n:  # a bad key or a provider outage, not a report that failed every criterion
            raise RuntimeError(f"all {n} judge calls failed; last error: {errors[-1]}")
        fraction = n_passed / n
        logger.info("harvey judge %s: %d/%d passed, %d errored, usage %s", judge, n_passed, n, len(errors), asdict(usage))
        return Score(
            value=fraction,
            answer="PASS" if n_passed == n else "FAIL",
            explanation=f"{n_passed}/{n} criteria passed, {len(errors)} errored (judge {judge}, batch_size {batch_size})",
            metadata={
                "criterion_fraction": fraction,
                "all_pass": n_passed == n,
                "n_criteria": n,
                "n_passed": n_passed,
                "n_errored": len(errors),
                "judge": judge,
                "batch_size": batch_size,
                "judge_usage": asdict(usage),
                "criteria": results,
            },
        )

    return score
