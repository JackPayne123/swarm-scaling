import pytest
from inspect_ai.model import ModelUsage
from inspect_ai.model._model import compute_model_cost

from swarm_scaling.prices import budget_cost, usage_cost


def test_opus_5_5_cache_reads_cost_a_twentieth_of_input() -> None:
    # Anthropic prices Opus 5.5 cache reads at $0.20/MTok (0.05x input), not 0.1x; agents re-read their
    # context on every call, so a 0.1x rate would overstate an Opus agent's spend and end its budget early.
    model = "anthropic/claude-opus-5-5"
    usage = ModelUsage(input_tokens=1_000, input_tokens_cache_read=1_000_000, output_tokens=1_000)
    expected = (1_000 * 4.0 + 1_000_000 * 0.20 + 1_000 * 20.0) / 1e6  # $0.224
    assert compute_model_cost(budget_cost(model), usage) == pytest.approx(expected)  # the live budget
    assert usage_cost(model, usage.model_dump(exclude_none=True)) == pytest.approx(expected)  # scripts/run_cost.py
