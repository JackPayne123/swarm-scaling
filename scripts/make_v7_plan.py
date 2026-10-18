"""Write a v7 plan (fleet.py's TSV) to stdout: per task and repeat, one diversified team of 4 and 4 diversified
independents (priority index 0-3), every row with the v7 flags (PLAN.md, "v7: harness changes"), team row first.

Agent boxes (2 CPUs each) get the same memory in every row of a task: AGENT_MEMORY_MB, or for tasks whose 100
generated problems take more than 4 GiB (data/v7_tasks.json big_memory) BIG_AGENT_MEMORY_MB. Both fit the Docker
hosts v6 measured: c7a.xlarge 7,731 MiB and c7a.4xlarge 31,379; m7a.xlarge 15,615 and m7a.4xlarge 62,915
(tasks.box_resources keeps 2,048 MiB of headroom; these leave 300-500 MiB more on the xlarge hosts). m7a.large and
m7a.2xlarge have the memory of c7a.xlarge and c7a.4xlarge.

--boxes m7a (default since 2026-10-18): every agent box on m7a, AMD EPYC 9R14 (Zen 4, AVX-512), the scorer's c7a CPU,
in every arm: m7a.large (2 vCPUs, 8 GiB) per independent and m7a.2xlarge (8 vCPUs, 32 GiB) per team of 4;
big-memory tasks m7a.xlarge / m7a.4xlarge. In the calibration the stable_matching team ran on m7a (Zen 4) and its
independents on t3a (EPYC 7571, Zen 1, no AVX-512), so only the team timed locally on the scorer's microarchitecture.

  uv run python scripts/make_v7_plan.py --prefix v7c --tasks pagerank,l1_pruning --model anthropic/claude-opus-5-5 \\
      --effort xhigh --budget 2 > plans/v7-calibration.tsv
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_MEMORY_MB = 5376
BIG_AGENT_MEMORY_MB = 13056
V7_FLAGS = ("--experiment v7 --elicit diversified-v7 --end-on-budget --isolate-agents --disjoint-dev-seeds --tool-fixes "
            "--tool-style claude_code --checker remote --cpus-per-agent 2 --split v7 --budget-type cost --time-limit 57600")


# Box types (team row, independent row) per --boxes and memory need. m7a: the scorer's CPU in every arm (above). t3a
# (2026-10-16, Jack: cheapest boxes, more memory where needed): t3a.large and t3a.2xlarge have the same memory as
# c7a.xlarge and c7a.4xlarge; t3a stops at 32 GiB, so a big-memory team of 4 goes on m7a.4xlarge, the same box as with
# --boxes c7a. t3a is Zen 1 and leaves big-memory teams on another CPU than their independents.
BOXES = {
    ("m7a", False): ("m7a.2xlarge", "m7a.large"), ("m7a", True): ("m7a.4xlarge", "m7a.xlarge"),
    ("c7a", False): ("c7a.4xlarge", "c7a.xlarge"), ("c7a", True): ("m7a.4xlarge", "m7a.xlarge"),
    ("t3a", False): ("t3a.2xlarge", "t3a.large"), ("t3a", True): ("m7a.4xlarge", "t3a.xlarge"),
}


def rows(prefix: str, task: str, repeat: int, model: str, effort: str, budget: float, big: bool,
         boxes: str = "m7a") -> list[str]:
    team_box, indep_box = BOXES[(boxes, big)]
    memory = BIG_AGENT_MEMORY_MB if big else AGENT_MEMORY_MB
    common = (f"--models {model} --reasoning-effort {effort} --budget {budget:g} --sample algotune/{task.replace('_', '-')} "
              f"{V7_FLAGS} --agent-memory-mb {memory}")
    group = f"{prefix}-{task}-r{repeat}"
    out = [f"{group}-divteam aws {team_box} --arm team --n 4 {common}"]
    out += [f"{group}-div-i{k} aws {indep_box} --arm independent {common} --priority-index {k}" for k in range(4)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--tasks", required=True, help="comma-separated v7 task names")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--budget", type=float, required=True, help="dollars per agent")
    p.add_argument("--boxes", choices=("m7a", "c7a", "t3a"), default="m7a",
                   help="agent box family (default m7a: the scorer's Zen 4 CPU in every arm; t3a: burstable, cheapest, Zen 1)")
    args = p.parse_args()
    meta = json.loads((ROOT / "data/v7_tasks.json").read_text())["tasks"]
    tasks = args.tasks.split(",")
    if unknown := [t for t in tasks if t not in meta]:
        raise SystemExit(f"not v7 candidates: {unknown}")
    for r in range(1, args.repeats + 1):  # repeat 1 of every task before repeat 2
        for t in tasks:
            print("\n".join(rows(args.prefix, t, r, args.model, args.effort, args.budget, meta[t]["big_memory"], args.boxes)))


if __name__ == "__main__":
    main()
