"""Shared team state for the swarm solver: inboxes, candidate registry, tools.

One ``Team`` lives per sample, in memory, and is shared by the N concurrent
agents through closures. Everything here runs on one event loop, so plain
lists are safe; ``anyio.Event`` is used only to wake blocked waiters.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import anyio
from inspect_ai.model import ContentText
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolResult
from inspect_ai.util import sandbox

AgentStatus = Literal["running", "waiting", "finished"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S") + "Z"


# Incoming messages are appended to the recipient's next tool result. With 8 agents broadcasting,
# full texts would flood every context, so deliveries show a preview per message and stop listing
# bodies past a total cap; read_message(id) returns the full text (Orazio, Inspect community Slack).
MESSAGE_PREVIEW_CHARS = 2000
DELIVERY_CHARS = 8000


@dataclass
class Message:
    sender: str
    sender_model: str | None
    to: str
    text: str
    sent_at: float
    id: str = ""

    def render(self, max_chars: int | None = None) -> str:
        who = self.sender
        if self.sender_model:
            who += f" ({self.sender_model})"
        body = self.text
        if max_chars is not None and len(body) > max_chars:
            body = (
                body[:max_chars]
                + f"\n[... truncated: {len(self.text) - max_chars} more characters. "
                f'Call read_message("{self.id}") for the full text.]'
            )
        return f"--- message {self.id} from {who} to {self.to} at {_iso(self.sent_at)} ---\n{body}"


@dataclass
class RegistryEntry:
    id: str
    agent_id: str
    model: str
    path: str
    source: str
    note: str
    published_at: float


@dataclass
class AgentRecord:
    agent_id: str
    model: str
    private_dir: str
    status: AgentStatus = "running"
    inbox: list[Message] = field(default_factory=list)
    sent: list[dict[str, Any]] = field(default_factory=list)
    received: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    list_candidates_calls: int = 0
    candidates_read: set[str] = field(default_factory=set)
    submitted: bool = False
    started_at: float | None = None
    ended_at: float | None = None
    end_reason: str | None = None
    limit_hit: str | None = None
    error: str | None = None
    tokens: dict[str, Any] = field(default_factory=dict)

    def as_log(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "private_dir": self.private_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_s": (
                round(self.ended_at - self.started_at, 3)
                if self.started_at is not None and self.ended_at is not None
                else None
            ),
            "end_reason": self.end_reason,
            "limit_hit": self.limit_hit,
            "error": self.error,
            "submitted": self.submitted,
            "tokens": self.tokens,
            "messages_sent": self.sent,
            "messages_received": self.received,
            "messages_dropped": self.dropped,
            "candidates_published": self.published,
            "list_candidates_calls": self.list_candidates_calls,
            "candidates_read": sorted(self.candidates_read),
        }


class Team:
    def __init__(
        self,
        agent_ids: list[str],
        model_names: list[str],
        workspace_root: str,
        reveal_model_family: bool = False,
    ) -> None:
        self.root = workspace_root.rstrip("/")
        self.registry_dir = f"{self.root}/registry"
        self.reveal_model_family = reveal_model_family
        self.agents: dict[str, AgentRecord] = {
            aid: AgentRecord(aid, name, f"{self.root}/agents/{aid}")
            for aid, name in zip(agent_ids, model_names, strict=True)
        }
        self.entries: list[RegistryEntry] = []
        self.messages: dict[str, Message] = {}
        self._change = anyio.Event()
        self._read_re = re.compile(
            re.escape(self.registry_dir) + r"/(agent_\d+-\d+)"
        )

    # -- status and wakeups -------------------------------------------------

    def _notify(self) -> None:
        ev, self._change = self._change, anyio.Event()
        ev.set()

    def set_status(self, agent_id: str, status: AgentStatus) -> None:
        self.agents[agent_id].status = status
        self._notify()

    def others(self, agent_id: str) -> list[AgentRecord]:
        return [a for a in self.agents.values() if a.agent_id != agent_id]

    # -- messaging ----------------------------------------------------------

    def send(self, sender: str, to: str, text: str) -> str:
        rec = self.agents[sender]
        if to == "all":
            targets = [a.agent_id for a in self.others(sender)]
        elif to == sender:
            raise ToolError("You cannot send a message to yourself.")
        elif to in self.agents:
            targets = [to]
        else:
            raise ToolError(
                f"Unknown recipient {to!r}. Use one of "
                f"{', '.join(a.agent_id for a in self.others(sender))} or 'all'."
            )
        now = time.time()
        msg = Message(
            sender=sender,
            sender_model=rec.model if self.reveal_model_family else None,
            to=to,
            text=text,
            sent_at=now,
            id=f"m{len(self.messages)}",
        )
        self.messages[msg.id] = msg
        lines: list[str] = []
        for target in targets:
            trec = self.agents[target]
            entry = {"to": target, "text": text, "at": now}
            if trec.status == "finished":
                rec.dropped.append(entry)
                lines.append(f"{target} has already finished; message not delivered.")
            else:
                trec.inbox.append(msg)
                rec.sent.append(entry)
                lines.append(f"Delivered to {target}.")
        self._notify()
        return "\n".join(lines)

    def drain(self, agent_id: str) -> list[Message]:
        rec = self.agents[agent_id]
        msgs, rec.inbox = rec.inbox, []
        now = time.time()
        for m in msgs:
            rec.received.append(
                {"from": m.sender, "text": m.text, "sent_at": m.sent_at, "delivered_at": now}
            )
        return msgs

    @staticmethod
    def render(msgs: list[Message]) -> str:
        """Previews of each message, and headers only once the delivery cap is reached."""
        parts = [f"[team messages: {len(msgs)} new]"]
        used = 0
        for i, m in enumerate(msgs):
            if used >= DELIVERY_CHARS:
                rest = msgs[i:]
                listed = ", ".join(f"{x.id} from {x.sender} ({len(x.text)} chars)" for x in rest)
                parts.append(
                    f"[{len(rest)} more message(s) not shown here: {listed}. "
                    "Read each with read_message(id).]"
                )
                break
            block = m.render(MESSAGE_PREVIEW_CHARS)
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    def read(self, agent_id: str, message_id: str) -> str:
        msg = self.messages.get(message_id)
        if msg is None or msg.sender == agent_id or msg.to not in (agent_id, "all"):
            raise ToolError(f"No message {message_id!r} was sent to you.")
        return msg.render()

    async def wait(self, agent_id: str, timeout_s: float) -> str:
        rec = self.agents[agent_id]
        deadline = time.monotonic() + max(0.0, timeout_s)
        self.set_status(agent_id, "waiting")
        try:
            while True:
                # capture the event before checking, so a notify between the
                # check and the wait is not lost
                ev = self._change
                if rec.inbox:
                    return self.render(self.drain(agent_id))
                if all(o.status != "running" for o in self.others(agent_id)):
                    return (
                        "No messages: every teammate has finished or is also waiting, "
                        "so no message can arrive right now. Continue working or finish."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return f"No messages received within {timeout_s} seconds."
                with anyio.move_on_after(remaining):
                    await ev.wait()
        finally:
            self.set_status(agent_id, "running")

    # -- registry -----------------------------------------------------------

    async def publish(self, agent_id: str, path: str, note: str) -> str:
        rec = self.agents[agent_id]
        entry_id = f"{agent_id}-{len(rec.published)}"
        dest = f"{self.registry_dir}/{entry_id}"
        script = (
            "set -e\n"
            f"src={shlex.quote(path)}; dest={shlex.quote(dest)}\n"
            '[ -e "$src" ] || { echo "no such path: $src" >&2; exit 2; }\n'
            '[ -e "$dest" ] && { echo "registry entry exists: $dest" >&2; exit 3; }\n'
            'mkdir -p "$dest"\n'
            'if [ -d "$src" ]; then cp -R "$src"/. "$dest"/; else cp "$src" "$dest"/; fi\n'
            'chmod -R a-w "$dest"\n'
        )
        result = await sandbox().exec(["sh", "-c", script], timeout=120)
        if not result.success:
            raise ToolError(f"publish_candidate failed: {result.stderr.strip()}")
        entry = RegistryEntry(
            id=entry_id,
            agent_id=agent_id,
            model=rec.model,
            path=dest,
            source=path,
            note=note,
            published_at=time.time(),
        )
        self.entries.append(entry)
        rec.published.append(entry_id)
        await sandbox().write_file(
            f"{self.registry_dir}/{entry_id}.json", json.dumps(asdict(entry), indent=2)
        )
        return f"Published {entry_id} at {dest} (read-only copy of {path})."

    def list_entries(self, agent_id: str) -> str:
        rec = self.agents[agent_id]
        rec.list_candidates_calls += 1
        if not self.entries:
            return "No candidates have been published yet."
        rec.candidates_read.update(e.id for e in self.entries)
        lines = [f"Published candidates ({len(self.entries)}):"]
        for e in self.entries:
            lines.append(
                f"- {e.id}  by {e.agent_id}  at {_iso(e.published_at)}  path: {e.path}\n"
                f"    note: {e.note}"
            )
        return "\n".join(lines)

    def note_registry_reads(self, agent_id: str, arguments: dict[str, Any]) -> None:
        known = {e.id for e in self.entries}
        for m in self._read_re.finditer(json.dumps(arguments, default=str)):
            if m.group(1) in known:
                self.agents[agent_id].candidates_read.add(m.group(1))


# -- tools ---------------------------------------------------------------------


def send_message_tool(team: Team, agent_id: str) -> Tool:
    async def execute(to: str, text: str) -> str:
        """Send a message to a teammate. It is delivered with the recipient's next tool result.

        Args:
            to: Recipient agent id (for example "agent_1"), or "all" for every teammate.
            text: The message. Recipients see up to 2,000 characters inline and can read the rest with read_message.
        """
        return team.send(agent_id, to, text)

    return ToolDef(execute, name="send_message").as_tool()


def read_message_tool(team: Team, agent_id: str) -> Tool:
    async def execute(message_id: str) -> str:
        """Read the full text of a message sent to you (shown truncated or listed without its text).

        Args:
            message_id: The message id, for example "m3".
        """
        return team.read(agent_id, message_id)

    return ToolDef(execute, name="read_message").as_tool()


def wait_for_message_tool(team: Team, agent_id: str) -> Tool:
    async def execute(timeout_s: float) -> str:
        """Block until a teammate sends you a message, or the timeout passes.

        Returns immediately if every teammate has finished or is also waiting, so
        it never deadlocks the team.

        Args:
            timeout_s: Maximum seconds to wait.
        """
        return await team.wait(agent_id, timeout_s)

    return ToolDef(execute, name="wait_for_message").as_tool()


def publish_candidate_tool(team: Team, agent_id: str) -> Tool:
    async def execute(path: str, note: str) -> str:
        """Publish a candidate solution to the shared registry.

        The file or directory at `path` is copied into a new read-only registry
        entry; later edits to `path` do not change the entry. The final answer is
        chosen from published candidates by a fixed rule, so publish anything
        worth keeping.

        Args:
            path: Absolute path of the file or directory to publish.
            note: What this candidate is and any measured result (for example a dev score).
        """
        return await team.publish(agent_id, path, note)

    return ToolDef(execute, name="publish_candidate").as_tool()


def list_candidates_tool(team: Team, agent_id: str) -> Tool:
    async def execute() -> str:
        """List every candidate published to the shared registry, with its path and note."""
        return team.list_entries(agent_id)

    return ToolDef(execute, name="list_candidates").as_tool()


def _append_messages(result: ToolResult, block: str) -> ToolResult:
    if isinstance(result, str):
        return f"{result}\n\n{block}" if result else block
    if isinstance(result, ContentText):
        return [result, ContentText(text=block)]
    if isinstance(result, list):
        return [*result, ContentText(text=block)]
    return f"{result}\n\n{block}"


def with_delivery(tool: Tool, team: Team, agent_id: str) -> Tool:
    """Wrap a tool so pending inbox messages are appended to its result."""
    tdef = ToolDef(tool)
    inner = tdef.tool

    async def execute(**kwargs: Any) -> ToolResult:
        team.note_registry_reads(agent_id, kwargs)
        # a ToolError leaves the inbox intact for the next successful result
        result = await inner(**kwargs)
        pending = team.drain(agent_id)
        if not pending:
            return result
        return _append_messages(result, Team.render(pending))

    return ToolDef(
        execute,
        name=tdef.name,
        description=tdef.description,
        parameters=tdef.parameters,
        parallel=tdef.parallel,
        viewer=tdef.viewer,
        model_input=tdef.model_input,
        max_output=tdef.max_output,
        options=tdef.options,
    ).as_tool()
