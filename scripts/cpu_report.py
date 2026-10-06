"""Per-arm CPU contention from swarm eval logs (state.metadata["swarm"]["cpu"]).

Teams share one container's CPUs; this shows whether larger teams saturate it more than solo runs.
Usage: uv run python scripts/cpu_report.py [log_dir]   (default: logs)
"""

import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

if __name__ == "__main__":
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
    arms: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            meta = (sample.metadata or {}).get("swarm")
            if not meta or "cpu" not in meta or meta["cpu"].get("flag") is None:
                continue
            key = (len(meta.get("agents", {})), "comm" if meta.get("messaging") else "solo/indep")
            arms[key].append(meta["cpu"])
    if not arms:
        print(f"no swarm CPU telemetry in {log_dir}")
        sys.exit(0)
    print(f"{'N':>3} {'arm':<11} {'samples':>7} {'flagged':>8} {'mean util':>10} {'throttled':>10}")
    for (n, arm), cpus in sorted(arms.items()):
        flagged = sum(bool(c["flag"]) for c in cpus) / len(cpus)
        util = [c["mean_util"] for c in cpus if c.get("mean_util") is not None]
        thr = [c["throttled_period_frac"] for c in cpus]
        print(
            f"{n:>3} {arm:<11} {len(cpus):>7} {flagged:>8.0%} "
            f"{(sum(util) / len(util) if util else float('nan')):>10.2f} {sum(thr) / len(thr):>10.2%}"
        )
    print("\nflagged = share of samples where the container's CPUs were saturated (see swarm.summarise_cpu).")
