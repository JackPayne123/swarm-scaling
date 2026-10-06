"""One tiny tool-calling run per experiment model, plus one LiteLLM call per judge.

Usage: source scripts/env.sh && uv run python scripts/model_smoke.py
Costs well under a cent in total.
"""

import litellm
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import tool

AGENT_MODELS = [
    "openai-api/zai/glm-5.3-flash",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-sol",
    "anthropic/claude-sonnet-5-5",
    "google/gemini-3.8-flash",
    "openrouter/deepseek/deepseek-v4.1-flash",
    "openrouter/openai/gpt-6-luna",
]
JUDGES = ["anthropic/claude-sonnet-4-6", "gemini/gemini-3.1-pro-preview"]


@tool
def add():
    async def execute(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        return a + b

    return execute


if __name__ == "__main__":
    import sys

    only = sys.argv[1:]
    if only:
        AGENT_MODELS = [m for m in AGENT_MODELS if m in only]
        JUDGES = [j for j in JUDGES if j in only]
    task = Task(
        dataset=[Sample(input="Use the add tool to compute 17 + 25, then submit just the number.", target="42")],
        solver=react(tools=[add()]),
        scorer=includes(),
        message_limit=8,
    )
    for model in AGENT_MODELS:
        # fail fast: an exhausted quota (429) otherwise retries silently for many minutes
        (log,) = eval(task, model=model, display="none", log_dir="logs/dev-2026-10-05", max_retries=2, timeout=600)
        s = log.samples[0] if log.samples else None
        used_tool = bool(s) and any(m.role == "tool" for m in s.messages)
        score = s.scores["includes"].value if s and s.scores else None
        usage = {k: (v.input_tokens, v.output_tokens) for k, v in (log.stats.model_usage or {}).items()}
        err = log.error.message[:200] if log.error else ""
        print(f"{model}: status={log.status} tool_call={used_tool} correct={score} tokens(in,out)={usage} {err}")
    for judge in JUDGES:
        try:
            r = litellm.completion(model=judge, messages=[{"role": "user", "content": "Reply with the word yes."}], max_tokens=200)
            print(f"judge {judge}: {r.choices[0].message.content!r} tokens={r.usage.total_tokens}")
        except Exception as ex:
            print(f"judge {judge}: FAILED {type(ex).__name__}: {str(ex)[:200]}")
