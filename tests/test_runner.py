import sys

import pytest

from swarm_scaling import runner
from swarm_scaling.runner import arm_config

GLM, LUNA = "openai-api/zai/glm-5.3-flash", "openai/gpt-6-luna"


def test_arms_share_the_save_mechanism_and_differ_only_in_sharing():
    solo = arm_config("solo", [GLM], 1, 1000, 4)
    team = arm_config("team", [GLM, LUNA], 4, 1000, 1)
    assert solo == {"models": [GLM], "per_agent_tokens": 4000, "messaging": False, "registry": True}
    assert team == {"models": [GLM, LUNA, GLM, LUNA], "per_agent_tokens": 1000, "messaging": True, "registry": True}
    assert arm_config("registry", [GLM], 2, 1000, 1)["messaging"] is False


@pytest.mark.parametrize(
    "arm,models,n,mult",
    [("independent", [GLM], 1, 2), ("solo", [GLM], 2, 1), ("team", [GLM], 4, 2), ("team", [GLM, LUNA], 3, 1)],
)
def test_invalid_arm_settings_are_refused(arm, models, n, mult):
    with pytest.raises(ValueError):
        arm_config(arm, models, n, 1000, mult)


def test_a_token_sized_budget_is_refused_as_dollars(monkeypatch, tmp_path, capsys):
    # The default budget type is dollars: a pilot script's --budget 2000000 must not become $2,000,000 per agent.
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path)
    argv = ["runner", "--arm", "solo", "--models", GLM, "--budget", "2000000", "--name", "x"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        runner.main()
    assert "for a token budget pass --budget-type all" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()  # refused before anything is written or started
