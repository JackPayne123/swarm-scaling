"""Cost of swarm eval logs from the token usage each provider reported.

Usage: uv run python scripts/run_cost.py <log.eval | dir> [...]

Prices and cache multipliers come from swarm_scaling.prices, the table the dollar budget meters with (cache reads
at each model's multiple of input, cache writes at 1.25x input, 1-hour writes at 2x). The log's per-model totals do
not split cache writes by TTL, so writes count as 5-minute except the 1-hour writes reported in the raw responses
the log kept (Anthropic's usage.cache_creation; Inspect keeps the raw call for only some model events, and the
output says for how many). Check against the provider's billing before relying on totals.
"""

import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

from swarm_scaling.prices import usage_cost


def one_hour_writes(path: Path) -> tuple[Counter, Counter, Counter]:
    """Per model: 1-hour cache-write tokens seen in logged raw responses, model calls, calls with a raw response."""
    writes, calls, seen = Counter(), Counter(), Counter()
    for sample in read_eval_log(str(path)).samples or []:
        for e in sample.events:
            if e.event != "model":
                continue
            calls[e.model] += 1
            usage = ((e.call.response or {}) if e.call is not None else {}).get("usage") or {}
            if "cache_creation" in usage:
                seen[e.model] += 1
                writes[e.model] += (usage["cache_creation"] or {}).get("ephemeral_1h_input_tokens") or 0
    return writes, calls, seen


if __name__ == "__main__":
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(p.rglob("*.eval")) if p.is_dir() else [p]
    total = 0.0
    for path in paths:
        log = read_eval_log(str(path), header_only=True)
        writes_1h, calls, seen = one_hour_writes(path)
        for model, usage in (log.stats.model_usage or {}).items():
            u = usage.model_dump(exclude_none=True)
            try:
                cost, note = usage_cost(model, u, writes_1h[model]), ""
            except ValueError:
                cost, note = float("nan"), f"no price for {model}"
            total += cost if cost == cost else 0.0
            print(f"{path.parent.name}/{path.name[:19]} {model}: ${cost:.3f}  "
                  f"in={u.get('input_tokens', 0):,} cache_r={u.get('input_tokens_cache_read', 0):,} "
                  f"cache_w={u.get('input_tokens_cache_write', 0):,} (1h seen {writes_1h[model]:,}; TTL split known "
                  f"for {seen[model]} of {calls[model]} calls) out={u.get('output_tokens', 0):,} {note}")
    print(f"TOTAL ${total:.2f}")
