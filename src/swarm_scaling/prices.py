"""Model prices in USD per 1M tokens, shared by the dollar budget (swarm.py) and scripts/run_cost.py.

List prices from Artificial Analysis, 2026-10-06. Check against the provider's billing before relying on totals.
"""

from inspect_ai.model import ModelCost

PRICES = {  # model substring -> (input, output)
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "deepseek-v4.1-flash": (0.30, 1.20),
    "glm-5.3-flash": (0.15, 0.50),
}

# The dollar budget prices every provider's cache the same way: writes at 1.25x input, reads at 0.1x input
# (Anthropic's published 5-minute rates). Output includes reasoning tokens. Inspect raises a write to 2x input
# when it sent the call with Anthropic's 1-hour TTL (after a gap of over 5 minutes in a sample), as Anthropic bills it.
BUDGET_CACHE_WRITE, BUDGET_CACHE_READ = 1.25, 0.10


def budget_cost(model: str) -> ModelCost:
    """The per-token prices the dollar budget meters `model` at. An unpriced or ambiguous model is an error."""
    keys = [k for k in PRICES if k in model]
    if len(keys) != 1:
        raise ValueError(f"no single price for {model!r} in swarm_scaling.prices.PRICES (matches: {keys})")
    p_in, p_out = PRICES[keys[0]]
    return ModelCost(
        input=p_in,
        output=p_out,
        input_cache_write=p_in * BUDGET_CACHE_WRITE,
        input_cache_read=p_in * BUDGET_CACHE_READ,
    )
