"""Cost of swarm eval logs from the token usage each provider reported.

Usage: uv run python scripts/run_cost.py <log.eval | dir> [...]

Prices are USD per 1M tokens (list prices from Artificial Analysis, 2026-10-06). Cache multipliers:
Anthropic's published 0.1x input for cache reads and 1.25x for 5-minute cache writes; Google and
DeepSeek cache-read rates are not checked and use CACHE_READ_FALLBACK (stated in the output).
Check against the provider's billing before relying on totals.
"""

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

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
ANTHROPIC_CACHE_READ, ANTHROPIC_CACHE_WRITE = 0.10, 1.25
CACHE_READ_FALLBACK = 0.25  # unchecked providers: upper end of the 10-25% range used in PLAN.md


def model_cost(model: str, u: dict) -> tuple[float, str]:
    key = next((k for k in PRICES if k in model), None)
    if key is None:
        return float("nan"), f"no price for {model}"
    p_in, p_out = PRICES[key]
    anthropic = "claude" in model
    read_mult = ANTHROPIC_CACHE_READ if anthropic else CACHE_READ_FALLBACK
    write_mult = ANTHROPIC_CACHE_WRITE if anthropic else 1.0
    cost = (
        u.get("input_tokens", 0) * p_in
        + u.get("input_tokens_cache_read", 0) * p_in * read_mult
        + u.get("input_tokens_cache_write", 0) * p_in * write_mult
        + u.get("output_tokens", 0) * p_out
    ) / 1e6
    note = "" if anthropic else f"cache read assumed {read_mult:.0%} of input"
    return cost, note


if __name__ == "__main__":
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(p.rglob("*.eval")) if p.is_dir() else [p]
    total = 0.0
    for path in paths:
        log = read_eval_log(str(path), header_only=True)
        for model, usage in (log.stats.model_usage or {}).items():
            u = usage.model_dump(exclude_none=True)
            cost, note = model_cost(model, u)
            total += cost if cost == cost else 0.0
            print(f"{path.parent.name}/{path.name[:19]} {model}: ${cost:.3f}  "
                  f"in={u.get('input_tokens', 0):,} cache_r={u.get('input_tokens_cache_read', 0):,} "
                  f"cache_w={u.get('input_tokens_cache_write', 0):,} out={u.get('output_tokens', 0):,} {note}")
    print(f"TOTAL ${total:.2f}")
