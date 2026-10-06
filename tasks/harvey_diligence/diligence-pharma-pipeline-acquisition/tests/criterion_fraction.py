"""Add the per-criterion pass fraction to rewardkit's reward.json.

harbor_scorer keeps only /logs/verifier/reward.json and then deletes /logs/verifier, and rewardkit's
all-pass reward hides how many criteria passed. This reads reward-details.json (written next to
reward.json) and adds criterion_fraction, n_criteria, n_passed and a 0/1 entry per criterion id to
reward.json. The existing keys stay first, so harbor_scorer's Score.value is still the all-pass reward
and the rest lands in Score.metadata["reward_dict"].

convert_harvey_diligence.py copies this file into every task's tests/ directory, where test.sh runs it:
    python3 /tests/criterion_fraction.py [/logs/verifier]
It must stay stdlib-only and do nothing at import, because rewardkit imports every .py file in /tests.
"""

import json
import sys
from pathlib import Path


def add_criterion_fraction(verifier_dir: Path) -> dict:
    details = json.loads((verifier_dir / "reward-details.json").read_text())
    (detail,) = details.values()  # one judge.toml, so one reward
    passed = {c["name"]: float(c["value"] > 0) for c in detail["criteria"]}
    n_passed = int(sum(passed.values()))
    reward = json.loads((verifier_dir / "reward.json").read_text())
    reward |= {"criterion_fraction": n_passed / len(passed), "n_criteria": len(passed), "n_passed": n_passed} | passed
    (verifier_dir / "reward.json").write_text(json.dumps(reward, indent=2))
    return reward


if __name__ == "__main__":
    result = add_criterion_fraction(Path(sys.argv[1] if len(sys.argv) > 1 else "/logs/verifier"))
    print(f"criterion_fraction: {result['criterion_fraction']} ({result['n_passed']}/{result['n_criteria']})")
