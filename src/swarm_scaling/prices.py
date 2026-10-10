"""Model prices in USD per 1M tokens, shared by the dollar budget (swarm.py) and scripts/run_cost.py.

List prices from Artificial Analysis, 2026-10-06; Anthropic base and cache-read prices from Anthropic's
prompt-caching pricing page (2026-10-09). Check against the provider's billing before relying on totals.
"""

from inspect_ai.model import ModelCost

PRICES = {  # model substring -> (input, output, cache read as a multiple of input)
    "claude-fable-5-1": (10.0, 50.0, 0.025),
    "claude-opus-5-5": (4.0, 20.0, 0.05),
    "claude-sonnet-5-5": (2.0, 10.0, 0.05),
    "claude-sonnet-4-6": (3.0, 15.0, 0.10),
    "gpt-6.1-sol": (2.0, 10.0, 0.10),
    "gpt-6-luna": (0.10, 0.50, 0.10),
    "gemini-3.8-flash": (0.75, 3.75, 0.10),
    "deepseek-v4.1-flash": (0.30, 1.20, 0.10),
    "glm-5.3-flash": (0.15, 0.50, 0.10),
}

# Cache writes: 5-minute TTL at 1.25x input, 1-hour TTL at 2x input (Anthropic's rates, applied to every provider).
# Output includes reasoning tokens. The live budget (Inspect's cost_limit) prices a write by the TTL the request
# was sent with; since 2026-10-10 swarm pins the 1-hour TTL on every Anthropic agent call (swarm anthropic_cache_ttl),
# so those writes are metered at 2x. (Inspect's default switches to 1 hour only after a gap of over 5 minutes.)
CACHE_WRITE_5M, CACHE_WRITE_1H = 1.25, 2.0


def budget_cost(model: str) -> ModelCost:
    """The per-token prices the dollar budget meters `model` at. An unpriced or ambiguous model is an error."""
    keys = [k for k in PRICES if k in model]
    if len(keys) != 1:
        raise ValueError(f"no single price for {model!r} in swarm_scaling.prices.PRICES (matches: {keys})")
    p_in, p_out, read = PRICES[keys[0]]
    return ModelCost(input=p_in, output=p_out, input_cache_write=p_in * CACHE_WRITE_5M, input_cache_read=p_in * read)


def usage_cost(model: str, usage: dict, cache_write_1h: int = 0) -> float:
    """Dollars for token counts in Inspect ModelUsage fields, of which `cache_write_1h` cache writes were 1-hour."""
    c = budget_cost(model)
    writes = usage.get("input_tokens_cache_write") or 0
    return (
        usage.get("input_tokens", 0) * c.input
        + (writes - cache_write_1h) * c.input_cache_write
        + cache_write_1h * c.input * CACHE_WRITE_1H
        + (usage.get("input_tokens_cache_read") or 0) * c.input_cache_read
        + usage.get("output_tokens", 0) * c.output
    ) / 1e6
