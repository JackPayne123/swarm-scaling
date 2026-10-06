import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("harvey_criterion_fraction", ROOT / "scripts/harvey_criterion_fraction.py")
fraction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fraction)


def criterion(name: str, value: float) -> dict:
    return {"name": name, "value": value, "raw": "yes" if value else "no", "weight": 1.0, "description": name}


def test_fraction_survives_next_to_the_all_pass_reward(tmp_path):
    # rewardkit's all_pass reward is 0.0 when any criterion fails, so on a 500-criterion data room it
    # cannot tell a run that found 99% of the red flags from one that found none. harbor_scorer reads the
    # first key of reward.json as the score and deletes /logs/verifier, so the fraction must be added to
    # reward.json itself and the all-pass key must stay first.
    criteria = [criterion("c-001", 1.0), criterion("c-002", 0.0), criterion("c-003", 1.0), criterion("c-004", 1.0)]
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 0.0}))
    (tmp_path / "reward-details.json").write_text(json.dumps({"reward": {"score": 0.0, "criteria": criteria, "kind": "llm"}}))

    fraction.add_criterion_fraction(tmp_path)

    reward = json.loads((tmp_path / "reward.json").read_text())
    assert next(iter(reward.items())) == ("reward", 0.0)
    assert reward["criterion_fraction"] == 0.75
    assert (reward["n_passed"], reward["n_criteria"]) == (3, 4)
    assert [reward[c["name"]] for c in criteria] == [1.0, 0.0, 1.0, 1.0]
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in reward.values())  # harbor_scorer rejects anything else
