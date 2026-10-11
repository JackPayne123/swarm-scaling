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


def test_haiku_5_5_prices_each_call_at_the_rate_card_its_prompt_size_selects() -> None:
    # Anthropic bills a Haiku 5.5 request over 100,000 prompt tokens (input + cache read + cache write) at 5x the
    # per-token rates; pricing an agent's calls at the short-prompt card would understate its spend once its context grows.
    model = "anthropic/claude-haiku-5-5"
    below = {"input_tokens": 1_000, "input_tokens_cache_read": 97_999, "input_tokens_cache_write": 1_000,
             "output_tokens": 1_000}  # prompt 99,999
    above = {**below, "input_tokens_cache_read": 98_001}  # prompt 100,001
    assert usage_cost(model, below, cache_write_1h=1_000) == pytest.approx(
        (1_000 * 0.10 + 97_999 * 0.01 + 1_000 * 0.20 + 1_000 * 0.50) / 1e6)
    assert usage_cost(model, above, cache_write_1h=1_000) == pytest.approx(
        (1_000 * 0.50 + 98_001 * 0.05 + 1_000 * 1.00 + 1_000 * 2.50) / 1e6)
    assert usage_cost(model, above) == pytest.approx((1_000 * 0.50 + 98_001 * 0.05 + 1_000 * 0.625 + 1_000 * 2.50) / 1e6)
