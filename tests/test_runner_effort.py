import pytest

from swarm_scaling.runner import effort_by_model

OPUS, SOL = "anthropic/claude-opus-5-5", "openai/gpt-6.1-sol"


def test_a_mixed_team_gets_each_models_own_effort_and_a_missing_one_is_refused():
    """A mixed Opus + Sol team runs each model at its own tier; silently giving Sol Opus's effort would change the arm."""
    assert effort_by_model("xhigh", [OPUS, SOL]) == {OPUS: "xhigh", SOL: "xhigh"}
    assert effort_by_model(f"{OPUS}=xhigh,{SOL}=high", [OPUS, SOL]) == {OPUS: "xhigh", SOL: "high"}
    with pytest.raises(SystemExit):
        effort_by_model(f"{OPUS}=xhigh", [OPUS, SOL])
