"""Select several finished v6 groups at once, beside a running scripts/v6_drive.py (which selects one at a time).

Every POLL_S it starts scripts/select_portfolio.py for each group of the plan whose rows have all ended and that has
no claim yet (analysis/portfolio/.claims), keeping at most --parallel selections running; it ends when every group
is claimed and its own selections have ended. The selector's claim makes each group run once: when the driver
reaches a group selected here, it waits for that selection to end, so v6.sh still takes the scorer down only after
the last one (2026-10-12: one group at a time would have run past the scorer's 30 h backstop).

    SCORER_TOKEN=... uv run python scripts/select_ready.py plans/v6-main.tsv --scorer http://<scorer>:8770
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "cloud"))
from select_portfolio import claim_path  # noqa: E402
from v6_drive import ROOT, SELECT_CONCURRENCY, fleet, group_of, state  # noqa: E402

POLL_S = 60


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--scorer", required=True)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--validity-n", type=int, default=0, help="select_portfolio.py --validity-n (v7: 1000)")
    args = p.parse_args()
    groups: dict[str, list[str]] = {}
    for r in fleet.parse_plan(Path(args.plan).read_text()):
        groups.setdefault(group_of(r.name), []).append(r.name)
    running: dict[str, subprocess.Popen] = {}
    while True:
        running = {g: pr for g, pr in running.items() if pr.poll() is None}
        for g, names in groups.items():
            if len(running) >= args.parallel:
                break
            names = [n for n in names if (ROOT / "logs" / n).is_dir()]
            if g in running or not names or claim_path(names).exists():
                continue
            if all(state(n) == "exit" for n in groups[g]):
                print(f"[select_ready] {time.strftime('%H:%M:%S')} selecting {g}", flush=True)
                running[g] = subprocess.Popen(
                    [sys.executable, "-u", "scripts/select_portfolio.py", *[f"logs/{n}" for n in names],
                     "--scorer", args.scorer, "--concurrency", str(SELECT_CONCURRENCY), "--validity-n", str(args.validity_n)],
                    cwd=ROOT)  # fmt: skip
        unclaimed = [g for g, ns in groups.items() if not claim_path([n for n in ns if (ROOT / "logs" / n).is_dir()]).exists()]
        if not unclaimed and not running:
            print(f"[select_ready] {time.strftime('%H:%M:%S')} every group is claimed; done", flush=True)
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
