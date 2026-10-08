"""Deep dump of one swarm team run: per-agent timelines, concurrency, messages, reasoning about teammates.

Usage: uv run python scripts/analyse_team_run.py <log.eval | dir> [out_dir]
Writes timeline-<agent>.txt (every model call and tool call, with reasoning), messages.txt, summary.txt.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

TEAM_WORDS = re.compile(
    r"\b(agent_\d|teammate|other agent|send_message|wait_for_message|message|registry|publish|"
    r"collaborat|coordinat|split|divide|duplicate|together|team)\w*",
    re.I,
)


def text_of(content) -> tuple[str, str]:
    """(visible text, reasoning text) from a message content field."""
    if isinstance(content, str):
        return content, ""
    vis, rea = [], []
    for c in content or []:
        t = getattr(c, "type", "")
        if t == "reasoning":
            # Anthropic returns full thinking encrypted (redacted=True) plus a readable summary
            if getattr(c, "redacted", False):
                rea.append(getattr(c, "summary", "") or "[redacted, no summary]")
            else:
                rea.append(getattr(c, "reasoning", "") or getattr(c, "summary", "") or "")
        elif t == "text":
            vis.append(c.text)
    return "\n".join(vis), "\n".join(rea)


def main() -> None:
    target = Path(sys.argv[1])
    path = max(target.glob("*.eval"), key=os.path.getmtime) if target.is_dir() else target
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis/runs") / path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    log = read_eval_log(str(path), resolve_attachments=True)
    s = log.samples[0]
    t0 = s.events[0].timestamp.timestamp()

    # map every span to the agent span it sits under
    parent: dict[str, str | None] = {}
    agent_of_span: dict[str, str] = {}
    for e in s.events:
        if e.event == "span_begin":
            parent[e.id] = e.parent_id
            if re.fullmatch(r"agent_\d+", e.name or ""):
                agent_of_span[e.id] = e.name

    def agent_for(span_id):
        seen = 0
        while span_id and seen < 200:
            if span_id in agent_of_span:
                return agent_of_span[span_id]
            span_id = parent.get(span_id)
            seen += 1
        return None

    lines = defaultdict(list)
    model_intervals = defaultdict(list)
    tool_counts = defaultdict(lambda: defaultdict(int))
    tool_errors = []
    model_errors = []
    team_reasoning = defaultdict(list)
    for e in s.events:
        agent = agent_for(getattr(e, "span_id", None))
        if agent is None:
            continue
        t = e.timestamp.timestamp() - t0
        if e.event == "model":
            end = (e.completed.timestamp() - t0) if getattr(e, "completed", None) else t
            model_intervals[agent].append((t, end))
            msg = e.output.message if e.output and e.output.choices else None
            vis, rea = text_of(msg.content) if msg else ("", "")
            calls = [f"{tc.function}({str(tc.arguments)[:300]})" for tc in (msg.tool_calls or [])] if msg else []
            if e.error:
                model_errors.append((agent, round(t), str(e.error)[:300]))
            u = e.output.usage if e.output else None
            lines[agent].append(
                f"\n[{t:7.0f}s-{end:7.0f}s] MODEL out={getattr(u, 'output_tokens', None)} "
                f"reasoning={getattr(u, 'reasoning_tokens', None)}"
                + (f"\n  REASONING: {rea.strip()}" if rea.strip() else "")
                + (f"\n  TEXT: {vis.strip()}" if vis.strip() else "")
                + "".join(f"\n  CALL: {c}" for c in calls)
            )
            for chunk in (rea, vis):
                for sent in re.split(r"(?<=[.!?])\s+", chunk):
                    if TEAM_WORDS.search(sent):
                        team_reasoning[agent].append(f"[{t:5.0f}s] {sent.strip()[:400]}")
        elif e.event == "tool":
            tool_counts[agent][e.function] += 1
            res = str(e.result)
            if e.error:
                tool_errors.append((agent, round(t), e.function, str(e.error)[:300]))
            lines[agent].append(f"[{t:7.0f}s] TOOL {e.function} -> {res[:600]}")

    for agent, ls in lines.items():
        (out / f"timeline-{agent}.txt").write_text("\n".join(ls))

    sw = s.metadata.get("swarm", {})
    msg_lines = []
    for aid, a in sw.get("agents", {}).items():
        for m in a.get("messages_sent") or []:
            msg_lines.append((m["at"], f"{aid} -> {m['to']} at {m['at'] - t0:6.0f}s:\n{m['text']}\n"))
        for f in a.get("files_sent") or []:
            msg_lines.append((f["at"], f"{aid} sent file {f['source']} -> {f['dest']} at {f['at'] - t0:6.0f}s"))
    (out / "messages.txt").write_text("\n".join(x for _, x in sorted(msg_lines)))

    # concurrency: share of each agent's active span during which the other agents were also active
    spans = {a: (rec.get("started_at", 0) - t0, rec.get("ended_at", 0) - t0) for a, rec in sw.get("agents", {}).items()}
    summ = [f"log: {path}", f"score: { {k: v.value for k, v in (s.scores or {}).items()} }", ""]
    for a, (st, en) in spans.items():
        busy = sum(e2 - s2 for s2, e2 in model_intervals[a])
        rec = sw["agents"][a]
        summ.append(
            f"{a}: active {st:.0f}-{en:.0f}s ({en - st:.0f}s); model-call time {busy:.0f}s over "
            f"{len(model_intervals[a])} calls; end={rec.get('end_reason')}; tools={dict(tool_counts[a])}; "
            f"published={rec.get('candidates_published')}; msgs sent={len(rec.get('messages_sent') or [])} "
            f"recv={len(rec.get('messages_received') or [])}"
        )
    if len(spans) > 1:
        (a1, (s1, e1)), (a2, (s2, e2)) = list(spans.items())[:2]
        overlap = max(0, min(e1, e2) - max(s1, s2))
        summ.append(f"overlap {a1}/{a2}: {overlap:.0f}s")
    checker_calls = (s.metadata.get("checker") or {}).get("calls", [])
    per_agent = defaultdict(lambda: [0, 0.0])
    for c in checker_calls:
        per_agent[c["agent_id"]][0] += 1
        per_agent[c["agent_id"]][1] += c["queue_wait_s"]
    summ += ["", f"checker (dev_eval) calls: {len(checker_calls)}"]
    summ += [f"  {a}: {k} calls, {w:.0f}s total queue wait" for a, (k, w) in sorted(per_agent.items())]
    summ += ["", f"model errors: {model_errors}", f"tool errors ({len(tool_errors)}):"]
    summ += [f"  {x}" for x in tool_errors[:40]]
    summ += ["", "TEAM-RELATED SENTENCES IN REASONING/TEXT:"]
    for a, sents in team_reasoning.items():
        summ.append(f"--- {a} ({len(sents)})")
        summ += [f"  {x}" for x in sents[:80]]
    (out / "summary.txt").write_text("\n".join(summ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
