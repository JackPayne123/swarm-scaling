"""Summarise one swarm eval log: agent records, finalize results, tool calls, last commands.

Usage: uv run python scripts/inspect_run.py <log.eval | log_dir>
"""

import collections
import json
import os
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

if __name__ == "__main__":
    target = Path(sys.argv[1])
    path = max(target.glob("*.eval"), key=os.path.getmtime) if target.is_dir() else target
    log = read_eval_log(str(path))
    for s in log.samples or []:
        sw = s.metadata.get("swarm", {})
        print(f"== {s.id} epoch {s.epoch} score={ {k: v.value for k, v in (s.scores or {}).items()} }")
        for aid, a in sw.get("agents", {}).items():
            keep = ("end_reason", "limit_hit", "wall_s", "submitted", "candidates_published", "error")
            print(f"  {aid}: {json.dumps({k: a.get(k) for k in keep})}")
            print(f"    tokens {a.get('tokens')}")
            print(f"    sent {len(a.get('messages_sent') or [])} received {len(a.get('messages_received') or [])}")
        fin = s.metadata.get("finalize") or {}
        for c in fin.get("candidates", []):
            print(f"  candidate {c.get('agent_id')} {c.get('kind')} {c.get('path')} valid={c.get('valid')} "
                  f"speedup={c.get('speedup')} err={c.get('error')}")
        print(f"  selected: {fin.get('selected')}")
        calls = [(tc.function, str(tc.arguments)) for m in s.messages for tc in (getattr(m, 'tool_calls', None) or [])]
        print(f"  tool calls (main transcript): {dict(collections.Counter(f for f, _ in calls))}")
        for f, a in calls[-6:]:
            print(f"    {f}: {a[:200].replace(chr(10), ' ')}")
