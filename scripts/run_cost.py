"""Cost of swarm eval logs from the token usage each provider reported.

Usage: uv run python scripts/run_cost.py <log.eval | dir> [...]

Each model call is priced on its own, from its model event's usage, with swarm_scaling.prices.usage_cost, the function
the dollar budget meters each call with: cache reads at each model's multiple of input, cache writes at 1.25x input,
1-hour writes at 2x, and a model with a long-prompt rate card (Haiku 5.5) at the tier the call's prompt size selects.
A model whose agents had the 1-hour TTL pinned (swarm metadata `cache_ttl`, since 2026-10-10) has every write counted
as 1-hour. Otherwise a call's writes count as 5-minute unless its raw response, which Inspect keeps for only some calls,
reports 1-hour writes (Anthropic's usage.cache_creation); the output says for how many calls the split was known.
The per-call token sums are checked against the log's per-model totals. Check against the provider's billing before
relying on totals.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from swarm_scaling.prices import LONG_PROMPT, usage_cost

FIELDS = ("input_tokens", "input_tokens_cache_read", "input_tokens_cache_write", "output_tokens")


def call_costs(path: Path) -> dict[str, dict]:
    """Per model: dollars, token sums, calls, calls whose TTL split was known, and calls (and dollars) above the
    model's long-prompt threshold, summed over every model call in the log."""
    out: dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "tokens": Counter(), "calls": 0, "seen": 0,
                                                "long_calls": 0, "long_cost": 0.0})
    for sample in read_eval_log(str(path)).samples or []:
        pinned = {a["model"] for a in ((sample.metadata or {}).get("swarm") or {}).get("agents", {}).values()
                  if a.get("cache_ttl") == "1h"}
        for e in sample.events:
            if e.event != "model" or e.output is None or e.output.usage is None or e.model.startswith("mockllm/"):
                continue  # mockllm (tests, smoke runs) costs nothing and has no price
            u = e.output.usage.model_dump(exclude_none=True)
            raw = ((e.call.response or {}) if e.call is not None else {}).get("usage") or {}
            m = out[e.model]
            if e.model in pinned:
                writes_1h = u.get("input_tokens_cache_write") or 0
            else:
                writes_1h = (raw.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
            m["seen"] += e.model in pinned or "cache_creation" in raw
            cost = usage_cost(e.model, u, writes_1h)
            m["cost"] += cost
            m["calls"] += 1
            m["tokens"].update({f: u.get(f) or 0 for f in FIELDS})
            threshold = next((t for k, (t, _) in LONG_PROMPT.items() if k in e.model), None)
            if threshold is not None and sum(u.get(f) or 0 for f in FIELDS[:3]) > threshold:
                m["long_calls"] += 1
                m["long_cost"] += cost
    return out


if __name__ == "__main__":
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(p.rglob("*.eval")) if p.is_dir() else [p]
    total = 0.0
    for path in paths:
        log = read_eval_log(str(path), header_only=True)
        try:
            per_model = call_costs(path)
        except ValueError as ex:  # a model with no price
            print(f"{path.parent.name}/{path.name[:19]}: {ex}")
            continue
        for model, usage in (log.stats.model_usage or {}).items():
            u = usage.model_dump(exclude_none=True)
            m = per_model.get(model)
            if m is None:
                print(f"{path.parent.name}/{path.name[:19]} {model}: no model events with usage")
                continue
            total += m["cost"]
            t = m["tokens"]
            mismatch = [f for f in FIELDS if t[f] != (u.get(f) or 0)]
            tiered = any(k in model for k in LONG_PROMPT)
            long = f" long-prompt calls {m['long_calls']} (${m['long_cost']:.3f})" if tiered else ""
            print(f"{path.parent.name}/{path.name[:19]} {model}: ${m['cost']:.3f}  "
                  f"in={t['input_tokens']:,} cache_r={t['input_tokens_cache_read']:,} "
                  f"cache_w={t['input_tokens_cache_write']:,} out={t['output_tokens']:,} calls={m['calls']} "
                  f"(TTL split known for {m['seen']}){long}"
                  + (f"  WARNING per-call sums differ from the log totals in {mismatch}" if mismatch else ""))
    print(f"TOTAL ${total:.2f}")
