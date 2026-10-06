"""Convert the Harvey LAB diligence data rooms into local Harbor task directories.

The Harbor hub dataset harveyai/lab is the v1.0 launch snapshot and has no diligence/* tasks, so this
rebuilds them from github.com/harveyai/harvey-labs the way Harvey's published Harbor tasks are built
(research/harbor-examples/harvey_task, from the native tasks/employment-labor/draft-markup-of-settlement-agreement):
    task.json criterion {id, title, match_criteria, deliverables}  ->  tests/judge.toml [[criterion]]
        name = id.lower(), description = "<title>\n\n<match_criteria>", files = /workspace/output/<deliverable>
    task.json instructions                                         ->  instruction.md, deliverables as /workspace/output/<name>
    documents/                                                     ->  environment/documents, COPYed to /workspace/documents
task.toml sets [environment] network_mode = "no-network". Grading runs on the host (src/swarm_scaling/harvey_grader.py
reads the rubric from tests/judge.toml there); the tests/ directory (judge.toml, test.sh, criterion_fraction.py) is
Harvey's original rewardkit verifier, copied into the container only by harvey_task(rewardkit_parity=True).

The clone is shallow at tasks/_src/harvey-labs and the copied documents are gitignored; never commit them.

Usage: uv run python scripts/convert_harvey_diligence.py [task-name ...] [--agent-timeout S] [--verifier-timeout S]
(default: all 11)
"""

import argparse
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tasks/_src/harvey-labs"
OUT = ROOT / "tasks/harvey_diligence"
FRACTION_SCRIPT = Path(__file__).with_name("harvey_criterion_fraction.py")

# Same image as Harvey's published Harbor tasks (their Dockerfile is lab_core/sandbox/Dockerfile in the repo).
SANDBOX_IMAGE = "ghcr.io/harveyai/lab-sandbox@sha256:cf4dac013b4cbae0fe417074a0180109fe1a3cccb6ec8b149909e3f8b87b4cb3"
REWARDKIT = "harbor-rewardkit[documents]==0.1.4"
JUDGE = "anthropic/claude-sonnet-4-6"
# Defaults for --verifier-timeout and --agent-timeout. Both are unmeasured; set them from the pilot (PLAN.md).
# The verifier timeout only applies to the in-container rewardkit parity check. The published tasks use 1800 s
# for 52 criteria; a diligence rubric has 438-1,114 criteria, one judge call each.
VERIFIER_TIMEOUT_SEC = 14400.0
# As published. inspect_harbor ignores [agent].timeout_sec; harvey_task applies the same value as its time_limit.
AGENT_TIMEOUT_SEC = 7200.0

TEST_SH = f"""#!/bin/bash
set -euo pipefail
# lab-sandbox already has python3.12 + pip + markitdown. Just install harbor-rewardkit.
pip install --quiet --no-cache-dir '{REWARDKIT}'
rewardkit /tests --max-concurrent-llm "${{JUDGE_CONCURRENCY:-8}}"
python3 /tests/criterion_fraction.py
"""

DOCKERFILE = f"""# Base: https://github.com/harveyai/harvey-labs/blob/main/lab_core/sandbox/Dockerfile

FROM {SANDBOX_IMAGE}

COPY documents/ /workspace/documents/
"""


def q(value) -> str:
    """A TOML basic string or array of them (JSON escaping is valid TOML for these)."""
    return json.dumps(value, ensure_ascii=False)


def task_toml(slug: str, task: dict, commit: str, agent_timeout: float, verifier_timeout: float) -> str:
    keywords = ["legal", "diligence", *(re.sub(r"\s+", "-", tag.lower()) for tag in task["tags"])]
    artifacts = [f"/workspace/output/{name}" for name in task["deliverables"]]
    return f"""schema_version = "1.0"
artifacts = {q(artifacts)}

[task]
name = {q(f"harveyai/diligence-{slug}")}
authors = [{{ name = "Harvey AI" }}]
keywords = {q(keywords)}

[metadata]
work_type = {q(task["work_type"])}
source_commit = {q(f"harveyai/harvey-labs@{commit}")}

[environment]
network_mode = "no-network"

[verifier]
timeout_sec = {verifier_timeout}

[verifier.env]
ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
JUDGE_CONCURRENCY = "${{JUDGE_CONCURRENCY:-8}}"
REWARDKIT_JUDGE = "${{REWARDKIT_JUDGE:-}}"

[agent]
timeout_sec = {agent_timeout}
"""


def instruction(task: dict) -> str:
    text = task["instructions"]
    for name in task["deliverables"]:
        assert f"`{name}`" in text, f"deliverable {name} is not named in the instructions"
        text = text.replace(f"`{name}`", f"`/workspace/output/{name}`")
    return f"{text}\n\nInput `/workspace/documents`\n"


def judge_toml(task: dict) -> str:
    parts = [f'[judge]\njudge = {q(JUDGE)}\nmode = "individual"\n\n[scoring]\naggregation = "all_pass"\n']
    for c in task["criteria"]:
        files = [f"/workspace/output/{name}" for name in c["deliverables"]]
        parts.append(
            f'[[criterion]]\nname = {q(c["id"].lower())}\n'
            f'description = {q(c["title"] + chr(10) * 2 + c["match_criteria"])}\n'
            f'type = "binary"\nfiles = {q(files)}\n'
        )
    return "\n".join(parts)


def convert(name: str, commit: str, agent_timeout: float, verifier_timeout: float) -> Path:
    src = SRC / "tasks/diligence" / name
    task = json.loads((src / "task.json").read_text())
    out = OUT / f"diligence-{name}"
    (out / "environment").mkdir(parents=True, exist_ok=True)
    (out / "tests").mkdir(exist_ok=True)

    (out / "task.toml").write_text(task_toml(name, task, commit, agent_timeout, verifier_timeout))
    (out / "instruction.md").write_text(instruction(task))
    (out / "environment/Dockerfile").write_text(DOCKERFILE)
    (out / "tests/judge.toml").write_text(judge_toml(task))
    (out / "tests/test.sh").write_text(TEST_SH)
    (out / "tests/test.sh").chmod(0o755)
    shutil.copy(FRACTION_SCRIPT, out / "tests/criterion_fraction.py")
    shutil.copytree(src / "documents", out / "environment/documents", dirs_exist_ok=True)

    # The judge file must carry every criterion unchanged; TOML escaping is the only risk.
    parsed = tomllib.loads((out / "tests/judge.toml").read_text())["criterion"]
    assert [(p["name"], p["description"]) for p in parsed] == [
        (c["id"].lower(), f'{c["title"]}\n\n{c["match_criteria"]}') for c in task["criteria"]
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="diligence task directory names (default: all)")
    parser.add_argument("--agent-timeout", type=float, default=AGENT_TIMEOUT_SEC)
    parser.add_argument("--verifier-timeout", type=float, default=VERIFIER_TIMEOUT_SEC)
    args = parser.parse_args()
    names = args.names or sorted(p.name for p in (SRC / "tasks/diligence").iterdir() if p.is_dir())
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    for name in names:
        print(convert(name, commit, args.agent_timeout, args.verifier_timeout))


if __name__ == "__main__":
    main()
