import pytest
from inspect_ai.tool import ToolError

from swarm_scaling.team import DELIVERY_CHARS, MESSAGE_PREVIEW_CHARS, Team


def team() -> Team:
    return Team(["agent_0", "agent_1", "agent_2"], ["m", "m", "m"], "/app")


def test_long_message_arrives_as_a_preview_with_a_pointer_to_the_full_text():
    # With 8 agents broadcasting, full texts would flood every recipient's context.
    t = team()
    long = "x" * (MESSAGE_PREVIEW_CHARS + 500)
    t.send("agent_0", "agent_1", long)
    shown = t.render(t.drain("agent_1"))
    assert "x" * MESSAGE_PREVIEW_CHARS in shown and long not in shown
    assert 'read_message("m0")' in shown
    assert long in t.read("agent_1", "m0")


def test_delivery_cap_lists_the_rest_by_header_only():
    t = team()
    for i in range(10):
        t.send("agent_0", "all", f"{i}" * MESSAGE_PREVIEW_CHARS)
    shown = t.render(t.drain("agent_2"))
    assert len(shown) < DELIVERY_CHARS + 2 * MESSAGE_PREVIEW_CHARS
    assert "more message(s) not shown here" in shown and "m9 from agent_0" in shown
    assert "9" * MESSAGE_PREVIEW_CHARS in t.read("agent_2", "m9")


def test_agents_can_only_read_messages_sent_to_them():
    t = team()
    t.send("agent_0", "agent_1", "private")
    with pytest.raises(ToolError):
        t.read("agent_2", "m0")
    with pytest.raises(ToolError):
        t.read("agent_0", "m0")
