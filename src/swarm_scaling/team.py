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
from typing import Any, Callable, Literal

import anyio
from inspect_ai.model import ContentText
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolResult
from inspect_ai.util import sandbox

AgentStatus = Literal["running", "waiting", "finished"]
ToolStyle = Literal["default", "claude_code"]
TaskStatus = Literal["pending", "in_progress", "completed"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S") + "Z"


# Incoming messages are appended to the recipient's next tool result. With 8 agents broadcasting,
# full texts would flood every context, so deliveries show a preview per message and stop listing
# bodies past a total cap; read_message(id) returns the full text (Orazio, Inspect community Slack).
MESSAGE_PREVIEW_CHARS = 2000
DELIVERY_CHARS = 8000

# Longest single wait_for_message: under run-to-budget an agent could otherwise idle until its 16 h time limit
# without spending (2026-10-10). Each further wait is another model call.
WAIT_MAX_S = 600

# Why an agent ended (its end_reason: "submitted" or the limit type), as its teammates are told.
FINISH_REASONS = {"submitted": "submitted", "token": "budget used", "cost": "budget used", "time": "time limit"}

# Publish notices (v7): the note is cut to this many characters so a notice stays one line.
NOTICE_NOTE_CHARS = 300


@dataclass
class Message:
    sender: str
    sender_model: str | None
    to: str
    text: str
    sent_at: float
    id: str = ""
    summary: str | None = None

    def render(self, max_chars: int | None = None, style: ToolStyle = "default") -> str:
        body = self.text
        if max_chars is not None and len(body) > max_chars:
            body = (
                body[:max_chars]
                + f"\n[... truncated: {len(self.text) - max_chars} more characters. "
                f'Call read_message("{self.id}") for the full text.]'
            )
        if style == "claude_code":
            # the envelope Claude Code uses for agent-team messages, plus the id read_message takes
            attrs = f'teammate_id="{self.sender}"'
            if self.sender_model:
                attrs += f' model="{self.sender_model}"'
            attrs += f' summary="{self.summary or ""}" message_id="{self.id}"'
            return f"<teammate-message {attrs}>\n{body}\n</teammate-message>"
        who = self.sender
        if self.sender_model:
            who += f" ({self.sender_model})"
        return f"--- message {self.id} from {who} to {self.to} at {_iso(self.sent_at)} ---\n{body}"


@dataclass
class TeamTask:
    id: str
    subject: str
    description: str
    created_by: str
    status: TaskStatus = "pending"
    owner: str | None = None


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
class PublishNotice:
    """One publish, as the publisher's teammates are told it (swarm publish_notices). `delivered` maps each teammate
    to when the notice reached its tool result, so adoption chains can be traced from the log."""

    id: str
    entry: str
    publisher: str
    path: str
    dev_score: str | None
    note: str
    at: float
    to: list[str] = field(default_factory=list)
    delivered: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        note = " ".join(self.note.split())
        if len(note) > NOTICE_NOTE_CHARS:
            note = note[:NOTICE_NOTE_CHARS] + "..."
        score = self.dev_score or "no dev score known"
        return f"[registry] {self.publisher} published {self.entry} at {self.path} ({score}). Note: {note}"


@dataclass
class AgentRecord:
    agent_id: str
    model: str
    private_dir: str
    status: AgentStatus = "running"
    inbox: list[Message] = field(default_factory=list)
    notices: list[PublishNotice] = field(default_factory=list)  # not yet delivered
    sent: list[dict[str, Any]] = field(default_factory=list)
    received: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    files_sent: list[dict[str, Any]] = field(default_factory=list)
    budget_notices: list[dict[str, Any]] = field(default_factory=list)
    submit_refusals: list[dict[str, Any]] = field(default_factory=list)
    list_candidates_calls: int = 0
    candidates_read: set[str] = field(default_factory=set)
    submitted: bool = False
    started_at: float | None = None
    ended_at: float | None = None
    end_reason: str | None = None
    limit_hit: str | None = None
    error: str | None = None
    tokens: dict[str, Any] = field(default_factory=dict)
    cache_ttl: str | None = None  # prompt-cache TTL pinned on this agent's model calls (None: the provider's default)
    # two-phase teams (swarm blind_frac): False until the agent has spent its blind fraction; while False, messaging
    # tools are refused, nothing is delivered to it and list_candidates shows only its own candidates
    channel_open: bool = True
    channel_opened_at: float | None = None
    blind_peeks: list[dict[str, Any]] = field(default_factory=list)

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
            "cache_ttl": self.cache_ttl,
            "messages_sent": self.sent,
            "messages_received": self.received,
            "messages_dropped": self.dropped,
            "candidates_published": self.published,
            "files_sent": self.files_sent,
            "budget_notices": self.budget_notices,
            "submit_refusals": self.submit_refusals,
            "list_candidates_calls": self.list_candidates_calls,
            "candidates_read": sorted(self.candidates_read),
            "channel_opened_at": self.channel_opened_at,
            "blind_peeks": self.blind_peeks,
        }


class Team:
    def __init__(
        self,
        agent_ids: list[str],
        model_names: list[str],
        workspace_root: str,
        reveal_model_family: bool = False,
        tool_style: ToolStyle = "default",
        candidate_file: str | None = None,
        publish_notices: bool = False,
        known_score: Callable[[bytes], str | None] | None = None,
        end_on_budget: bool = False,
        boxes: dict[str, str | None] | None = None,
    ) -> None:
        self.root = workspace_root.rstrip("/")
        self.registry_dir = f"{self.root}/registry"
        self.reveal_model_family = reveal_model_family
        self.tool_style = tool_style
        # a published single file is saved under this name (AlgoTune: solver.py, the name finalize reads)
        self.candidate_file = candidate_file
        self.publish_notices = publish_notices
        self.known_score = known_score
        self.end_on_budget = end_on_budget  # agents cannot finish early, so nothing tells them to
        self.notices: list[PublishNotice] = []
        # each agent's sandbox (per-agent boxes; None = the shared default box): its paths are resolved there
        self.boxes = boxes or {}
        self.agents: dict[str, AgentRecord] = {
            aid: AgentRecord(aid, name, f"{self.root}/agents/{aid}")
            for aid, name in zip(agent_ids, model_names, strict=True)
        }
        self.entries: list[RegistryEntry] = []
        self.messages: dict[str, Message] = {}
        self.tasks: dict[str, TeamTask] = {}
        self.task_events: list[dict[str, Any]] = []
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

    def finish(self, agent_id: str, end_reason: str | None, notify: bool) -> None:
        """Mark an agent finished; with `notify` (messaging arms), tell every teammate still running.

        The notice goes through the normal message channel (sender "system"), so it arrives after a
        teammate's next tool result or ends its wait_for_message. Finished agents get nothing, as with
        any message.
        """
        if notify:
            why = FINISH_REASONS.get(end_reason or "", "stopped")
            msg = Message(
                sender="system",
                sender_model=None,
                to="all",
                text=f"{agent_id} has finished ({why}) and will not read further messages.",
                sent_at=time.time(),
                id=f"m{len(self.messages)}",
                summary=f"{agent_id} finished",
            )
            self.messages[msg.id] = msg
            for rec in self.others(agent_id):
                if rec.status != "finished":
                    rec.inbox.append(msg)
        self.set_status(agent_id, "finished")

    def others(self, agent_id: str) -> list[AgentRecord]:
        return [a for a in self.agents.values() if a.agent_id != agent_id]

    # -- messaging ----------------------------------------------------------

    def send(self, sender: str, to: str, text: str, summary: str | None = None) -> str:
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
            summary=summary,
        )
        self.messages[msg.id] = msg
        lines: list[str] = []
        for target in targets:
            trec = self.agents[target]
            entry = {"to": target, "text": text, "at": now}
            if summary is not None:
                entry["summary"] = summary
            if trec.status == "finished":
                rec.dropped.append(entry)
                lines.append(f"{target} has already finished; message not delivered.")
            else:
                trec.inbox.append(msg)
                rec.sent.append(entry)
                if trec.channel_open:
                    lines.append(f"Delivered to {target}.")
                else:
                    lines.append(f"Queued for {target}: its team channel is not open yet, so it sees this when it opens.")
        self._notify()
        return "\n".join(lines)

    def drain(self, agent_id: str) -> list[Message]:
        rec = self.agents[agent_id]
        if not rec.channel_open:  # held until the agent's channel opens
            return []
        msgs, rec.inbox = rec.inbox, []
        now = time.time()
        for m in msgs:
            rec.received.append(
                {"from": m.sender, "text": m.text, "sent_at": m.sent_at, "delivered_at": now}
            )
        return msgs

    def drain_notices(self, agent_id: str) -> list[PublishNotice]:
        rec = self.agents[agent_id]
        if not rec.channel_open:  # held until the agent's channel opens, as messages are
            return []
        notices, rec.notices = rec.notices, []
        now = time.time()
        for n in notices:
            n.delivered[agent_id] = now
        return notices

    def pending_block(self, agent_id: str) -> str | None:
        """Everything waiting for the agent, as appended to its next tool result: publish notices, then messages."""
        notices, msgs = self.drain_notices(agent_id), self.drain(agent_id)
        parts = [n.render() for n in notices] + ([self.render(msgs)] if msgs else [])
        return "\n".join(parts) if parts else None

    def render(self, msgs: list[Message]) -> str:
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
            block = m.render(MESSAGE_PREVIEW_CHARS, self.tool_style)
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    def read(self, agent_id: str, message_id: str) -> str:
        msg = self.messages.get(message_id)
        if msg is None or msg.sender == agent_id or msg.to not in (agent_id, "all"):
            raise ToolError(f"No message {message_id!r} was sent to you.")
        return msg.render(style=self.tool_style)

    async def wait(self, agent_id: str, timeout_s: float) -> str:
        rec = self.agents[agent_id]
        deadline = time.monotonic() + max(0.0, timeout_s)
        self.set_status(agent_id, "waiting")
        try:
            while True:
                # capture the event before checking, so a notify between the
                # check and the wait is not lost
                ev = self._change
                if rec.inbox or rec.notices:
                    return self.pending_block(agent_id) or ""
                if all(o.status != "running" for o in self.others(agent_id)):
                    return (
                        "No messages: every teammate has finished or is also waiting, "
                        "so no message can arrive right now. "
                        + ("Continue working." if self.end_on_budget else "Continue working or finish.")
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return f"No messages received within {timeout_s} seconds."
                with anyio.move_on_after(remaining):
                    await ev.wait()
        finally:
            self.set_status(agent_id, "running")

    # -- direct file sharing -------------------------------------------------

    def inbox_dir(self, agent_id: str) -> str:
        return f"{self.agents[agent_id].private_dir}/inbox"

    async def share_file(self, sender: str, to: str, path: str, note: str) -> str:
        """Copy a file or folder into each recipient's inbox folder and message them its location.

        The copy belongs to the recipient (writable); the sender's original is untouched. Inbox
        folders sit below each agent's working directory, so their solver.py files are not
        selection candidates unless the recipient moves one up.
        """
        if to == "all":
            targets = [a.agent_id for a in self.others(sender)]
        elif to == sender:
            raise ToolError("You cannot send a file to yourself.")
        elif to in self.agents:
            targets = [to]
        else:
            raise ToolError(
                f"Unknown recipient {to!r}. Use one of "
                f"{', '.join(a.agent_id for a in self.others(sender))} or 'all'."
            )
        rec = self.agents[sender]
        if not path.startswith("/"):
            path = f"{rec.private_dir}/{path}"
        k = len(rec.files_sent)
        name = path.rstrip("/").split("/")[-1] or "file"
        lines: list[str] = []
        for target in targets:
            if self.agents[target].status == "finished":
                lines.append(f"{target} has already finished; file not sent.")
                continue
            dest = f"{self.inbox_dir(target)}/from-{sender}-{k}/{name}"
            script = (
                "set -e\n"
                f"src={shlex.quote(path)}; dest={shlex.quote(dest)}\n"
                '[ -e "$src" ] || { echo "no such path: $src" >&2; exit 2; }\n'
                'mkdir -p "$(dirname "$dest")"\n'
                'cp -R "$src" "$dest"\n'
            )
            result = await sandbox(self.boxes.get(sender)).exec(["sh", "-c", script], timeout=120)
            if not result.success:
                raise ToolError(f"send_file failed: {result.stderr.strip()}")
            rec.files_sent.append({"to": target, "source": path, "dest": dest, "note": note, "at": time.time()})
            body = f"[file] {sender} sent you {path} -> copied to {dest}" + (f"\n{note}" if note else "")
            self.send(sender, target, body, summary=f"sent you {name}")
            lines.append(f"Sent to {target}: {dest}")
        return "\n".join(lines)

    # -- shared task list (claude_code tool style) ------------------------------

    def _task(self, task_id: str) -> TeamTask:
        task = self.tasks.get(str(task_id).lstrip("#"))
        if task is None:
            raise ToolError(f"No task {task_id!r}. TaskList() lists the task ids.")
        return task

    def create_task(self, agent_id: str, subject: str, description: str) -> str:
        task = TeamTask(str(len(self.tasks) + 1), subject, description, created_by=agent_id)
        self.tasks[task.id] = task
        self.task_events.append(
            {"event": "create", "task_id": task.id, "agent": agent_id, "at": time.time(),
             "subject": subject, "description": description}
        )
        return f"Task #{task.id} created: {subject}"

    def update_task(
        self, agent_id: str, task_id: str, status: TaskStatus | None, owner: str | None
    ) -> str:
        task = self._task(task_id)
        if status is None and owner is None:
            raise ToolError("Nothing to update: give a status, an owner, or both.")
        if owner is not None and owner not in self.agents:
            raise ToolError(f"Unknown owner {owner!r}. Use one of {', '.join(self.agents)}.")
        changes: dict[str, Any] = {}
        if status is not None:
            task.status = changes["status"] = status
        if owner is not None:
            task.owner = changes["owner"] = owner
        self.task_events.append(
            {"event": "update", "task_id": task.id, "agent": agent_id, "at": time.time(), **changes}
        )
        return f"Task #{task.id} updated: " + ", ".join(f"{k}={v}" for k, v in changes.items())

    def list_tasks(self) -> str:
        if not self.tasks:
            return "No tasks yet."
        lines = [f"Tasks ({len(self.tasks)}):"]
        for t in self.tasks.values():
            lines.append(
                f"- #{t.id} [{t.status}] {t.subject}  owner: {t.owner or '-'}  created by: {t.created_by}"
            )
        return "\n".join(lines)

    def get_task(self, task_id: str) -> str:
        t = self._task(task_id)
        return (
            f"Task #{t.id}: {t.subject}\nstatus: {t.status}\nowner: {t.owner or '-'}\n"
            f"created by: {t.created_by}\n\n{t.description}"
        )

    # -- registry -----------------------------------------------------------

    async def publish(self, agent_id: str, path: str, note: str) -> str:
        rec = self.agents[agent_id]
        entry_id = f"{agent_id}-{len(rec.published)}"
        dest = f"{self.registry_dir}/{entry_id}"
        file_dest = f'"$dest"/{shlex.quote(self.candidate_file)}' if self.candidate_file else '"$dest"/'
        script = (
            "set -e\n"
            f"src={shlex.quote(path)}; dest={shlex.quote(dest)}\n"
            '[ -e "$src" ] || { echo "no such path: $src" >&2; exit 2; }\n'
            '[ -e "$dest" ] && { echo "registry entry exists: $dest" >&2; exit 3; }\n'
            'mkdir -p "$dest"\n'
            f'if [ -d "$src" ]; then cp -R "$src"/. "$dest"/; else cp "$src" {file_dest}; echo file; fi\n'
            'chmod -R a-w "$dest"\n'
        )
        box = sandbox(self.boxes.get(agent_id))  # the path is the publisher's; the registry is shared
        result = await box.exec(["sh", "-c", script], timeout=120)
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
        await box.write_file(f"{self.registry_dir}/{entry_id}.json", json.dumps(asdict(entry), indent=2))
        renamed = (
            self.candidate_file is not None
            and result.stdout.strip() == "file"
            and path.rstrip("/").rsplit("/", 1)[-1] != self.candidate_file
        )
        saved_as = f", saved as {dest}/{self.candidate_file}" if renamed else ""
        if self.publish_notices:
            await self._notify_publish(entry)
        return f"Published {entry_id} at {dest} (read-only copy of {path}{saved_as})."

    async def _notify_publish(self, entry: RegistryEntry) -> None:
        """Queue a publish notice for every teammate still running; quote the candidate file's known dev score."""
        score = None
        if self.known_score is not None and self.candidate_file is not None:
            try:
                box = sandbox(self.boxes.get(entry.agent_id))
                score = self.known_score(await box.read_file(f"{entry.path}/{self.candidate_file}", text=False))
            except FileNotFoundError:  # a published folder without the candidate file
                pass
        notice = PublishNotice(f"n{len(self.notices)}", entry.id, entry.agent_id, entry.path, score, entry.note,
                               entry.published_at)  # fmt: skip
        self.notices.append(notice)
        for rec in self.others(entry.agent_id):
            if rec.status != "finished":
                notice.to.append(rec.agent_id)
                rec.notices.append(notice)
        self._notify()

    def visible_entries(self, agent_id: str) -> list[RegistryEntry]:
        """Every entry once the agent's channel is open; before that, only its own."""
        if self.agents[agent_id].channel_open:
            return self.entries
        return [e for e in self.entries if e.agent_id == agent_id]

    def list_entries(self, agent_id: str) -> str:
        rec = self.agents[agent_id]
        rec.list_candidates_calls += 1
        entries = self.visible_entries(agent_id)
        if not entries:
            return "No candidates have been published yet."
        rec.candidates_read.update(e.id for e in entries)
        lines = [f"Published candidates ({len(entries)}):"]
        for e in entries:
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


def send_message_cc_tool(team: Team, agent_id: str, summary_optional: bool = False) -> Tool:
    """Claude Code's SendMessage. summary_optional (v7 tool_fixes): v6 agents failed 11-16 calls on a missing summary."""

    async def execute(to: str, message: str, summary: str) -> str:
        """Send a message to a teammate. It is delivered with the recipient's next tool result.

        Args:
            to: Recipient teammate id (for example "agent_1"), or "all" for every teammate.
            message: The message. Recipients see up to 2,000 characters inline and can read the rest with read_message.
            summary: A 5-10 word preview of the message, shown to the recipient with it.
        """
        return team.send(agent_id, to, message, summary=summary)

    async def execute_optional(to: str, message: str, summary: str | None = None) -> str:
        """Send a message to a teammate. It is delivered with the recipient's next tool result.

        Args:
            to: Recipient teammate id (for example "agent_1"), or "all" for every teammate.
            message: The message. Recipients see up to 2,000 characters inline and can read the rest with read_message.
            summary: Optional 5-10 word preview of the message, shown to the recipient with it.
        """
        return team.send(agent_id, to, message, summary=summary)

    return ToolDef(execute_optional if summary_optional else execute, name="SendMessage").as_tool()


def task_create_tool(team: Team, agent_id: str) -> Tool:
    async def execute(subject: str, description: str) -> str:
        """Create a task in the team's shared task list. New tasks are pending with no owner.

        Args:
            subject: A brief title for the task.
            description: What needs to be done.
        """
        return team.create_task(agent_id, subject, description)

    return ToolDef(execute, name="TaskCreate").as_tool()


def task_list_tool(team: Team, agent_id: str) -> Tool:
    async def execute() -> str:
        """List every task in the team's shared task list with its id, subject, status, owner and creator."""
        return team.list_tasks()

    return ToolDef(execute, name="TaskList").as_tool()


def task_get_tool(team: Team, agent_id: str) -> Tool:
    async def execute(taskId: str) -> str:  # noqa: N803 (Claude Code's parameter name)
        """Show one task from the shared task list in full, including its description.

        Args:
            taskId: The task id, for example "1".
        """
        return team.get_task(taskId)

    return ToolDef(execute, name="TaskGet").as_tool()


def task_update_tool(team: Team, agent_id: str) -> Tool:
    async def execute(  # noqa: N803 (Claude Code's parameter name)
        taskId: str, status: TaskStatus | None = None, owner: str | None = None
    ) -> str:
        """Update a task in the shared task list. Any agent may update any task.

        Args:
            taskId: The task id, for example "1".
            status: New status: pending, in_progress or completed.
            owner: New owner, an agent id (for example "agent_1"); set it to your own id to claim the task.
        """
        return team.update_task(agent_id, taskId, status, owner)

    return ToolDef(execute, name="TaskUpdate").as_tool()


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
        """Block until a teammate sends you a message, or the timeout passes (at most 600 seconds per call).

        Returns immediately if every teammate has finished or is also waiting, so
        it never deadlocks the team.

        Args:
            timeout_s: Maximum seconds to wait; values above 600 wait 600.
        """
        return await team.wait(agent_id, min(timeout_s, WAIT_MAX_S))

    return ToolDef(execute, name="wait_for_message").as_tool()


def send_file_tool(team: Team, agent_id: str) -> Tool:
    async def execute(to: str, path: str, note: str = "") -> str:
        """Send a copy of a file or folder to a teammate (or "all"). It lands in their inbox folder and they get a message with its location.

        Args:
            to: Recipient agent id (for example "agent_1"), or "all" for every teammate.
            path: File or folder to send (absolute path, or relative to your working directory).
            note: Optional message to go with it.
        """
        return await team.share_file(agent_id, to, path, note)

    return ToolDef(execute, name="send_file").as_tool()


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
        pending = team.pending_block(agent_id)
        if pending is None:
            return result
        return _append_messages(result, pending)

    return _rewrap(tdef, execute)


# Team tools refused while an agent's channel is closed (two-phase teams).
CHANNEL_TOOLS = {
    "send_message", "SendMessage", "read_message", "send_file", "wait_for_message",
    "TaskCreate", "TaskList", "TaskGet", "TaskUpdate",
}  # fmt: skip


def with_channel_gate(tool: Tool, team: Team, agent_id: str, ready: Any, closed_text: Any) -> Tool:
    """Two-phase teams: refuse channel tools until `ready()` (the agent has spent its blind fraction), then open.

    The channel opens at the start or the end of whichever tool call first finds `ready()` true; that call's
    result gets a notice that the channel is open, followed by every message held while it was closed.
    `closed_text()` is the refusal. While closed, calls whose arguments name the registry or another agent's
    directory are logged in `blind_peeks` (the shared filesystem itself is not access-controlled).
    """
    tdef = ToolDef(tool)
    inner = tdef.tool
    rec = team.agents[agent_id]
    peek = re.compile(re.escape(team.registry_dir) + "|" + re.escape(team.root) + r"/agents/(agent_\d+)")

    def open_if_ready() -> bool:
        if rec.channel_open or not ready():
            return False
        rec.channel_open, rec.channel_opened_at = True, time.time()
        return True

    async def execute(**kwargs: Any) -> ToolResult:
        opened = open_if_ready()
        if not rec.channel_open:
            if any(m.group(1) in (None, *(a.agent_id for a in team.others(agent_id)))
                   for m in peek.finditer(json.dumps(kwargs, default=str))):  # fmt: skip
                rec.blind_peeks.append({"tool": tdef.name, "arguments": kwargs, "at": time.time()})
            if tdef.name in CHANNEL_TOOLS:
                raise ToolError(closed_text())
        result = await inner(**kwargs)
        opened = open_if_ready() or opened
        if not opened:
            return result
        notice = ("[team] The team channel is now open: you can use the messaging tools, list_candidates shows "
                  "every published candidate, and you may read the registry and teammates' working directories.")  # fmt: skip
        pending = team.pending_block(agent_id)
        return _append_messages(result, notice + (f"\n\n{pending}" if pending else ""))

    return _rewrap(tdef, execute)


def budget_amount(value: float, budget_type: str, decimals: int = 2) -> str:
    """A budget quantity as agents see it: dollars for a cost budget ("$1.25"), else a token count ("1,250")."""
    return f"${value:,.{decimals}f}" if budget_type == "cost" else f"{int(value):,}"


def with_budget_notices(
    tool: Tool, limit: Any, rec: AgentRecord, thresholds: tuple[float, ...], budget_type: str = "all"
) -> Tool:
    """Wrap a tool so a notice is appended to its result when the agent's usage passes a threshold.

    `limit` is the agent's own token or cost limit (``.usage``, ``.limit``). Each threshold is announced
    once; when several are passed at once only the highest is announced.
    """
    tdef = ToolDef(tool)
    inner = tdef.tool

    async def execute(**kwargs: Any) -> ToolResult:
        result = await inner(**kwargs)
        if limit.limit is None:
            return result
        used, cap = limit.usage, limit.limit
        # usage only grows, so every threshold up to the last one announced is done
        done = max((n["threshold"] for n in rec.budget_notices), default=0.0)
        passed = [t for t in thresholds if t > done and used >= t * cap]
        if not passed:
            return result
        rec.budget_notices.append({"threshold": max(passed), "used": used, "at": time.time()})
        unit = {"output": "output-token ", "cost": ""}.get(budget_type, "token ")
        amounts = f"{budget_amount(used, budget_type)} of {budget_amount(cap, budget_type)}"
        notice = f"[budget] You have used {int(used * 100 // cap)}% of your {unit}budget ({amounts})."
        return _append_messages(result, notice)

    return _rewrap(tdef, execute)


def _rewrap(tdef: ToolDef, execute: Any) -> Tool:
    """A tool with `execute` as its body and everything else (name, schema, options) from `tdef`."""
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
