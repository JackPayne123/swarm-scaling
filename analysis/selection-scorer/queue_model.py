"""The agent scorer's dev_eval load from a scorer job log, and the queue wait a closed-network model predicts by slots.

    uv run python analysis/selection-scorer/queue_model.py ~/projects/swarm-scaling/logs/_scorer/i-0b9a6d925d49b1c1d/jobs.jsonl

Per job log: dev_eval run time and queue wait, the gap between an agent's dev_eval ending and its next one being
queued (the agent's "think time"), and how busy the slots were over the span of the agents' jobs (selection jobs
left out). The model is exact mean-value analysis of a closed network: N agents, each thinking Z seconds, then one job
at a multi-server station (c slots, exponential service of mean S). It is a model, not a measurement (v7 batch 1: 153 s
modelled on 3 slots, 128 s measured).
"""

import collections
import json
import statistics
import sys


def mva(n_agents: int, think_s: float, service_s: float, slots: int) -> tuple[float, float]:
    """(mean queue wait, slot utilisation) for the closed network above."""
    p = [1.0]  # p[j]: probability of j agents at the scorer, for the current population
    resp = 0.0
    for n in range(1, n_agents + 1):
        resp = sum(j * service_s / min(j, slots) * p[j - 1] for j in range(1, n + 1))
        x = n / (think_s + resp)
        q = [0.0] + [x * service_s / min(j, slots) * p[j - 1] for j in range(1, n + 1)]
        q[0] = 1 - sum(q[1:])
        p = q
    return resp - service_s, x * service_s / slots


def main() -> None:
    jobs = [json.loads(line) for line in open(sys.argv[1])]
    jobs = [j for j in jobs if j.get("run_id") != "select_portfolio"]  # the agents' jobs only, not a later selection
    dev = [j for j in jobs if j["kind"] == "dev_eval" and j.get("run_s") is not None]
    by_agent = collections.defaultdict(list)
    for j in dev:
        by_agent[(j["run_id"], j["agent_id"])].append(j)
    gaps = []
    for js in by_agent.values():
        js.sort(key=lambda j: j["queued_at"])
        gaps += [b["queued_at"] - a["ended_at"] for a, b in zip(js, js[1:])]
    span = max(j["ended_at"] for j in jobs if j.get("ended_at")) - min(j["queued_at"] for j in jobs)
    slots = len({j["slot"] for j in jobs if j.get("slot") is not None})
    busy = sum(j["run_s"] for j in jobs if j.get("run_s")) / (span * slots)
    run_s, wait_s = statistics.mean(j["run_s"] for j in dev), statistics.mean(j["queue_wait_s"] for j in dev)
    think_s = statistics.mean(gaps)
    print(f"{len(dev)} dev_eval jobs from {len(by_agent)} agents over {span / 3600:.2f} h on {slots} slots ({busy:.0%} busy)")
    print(f"dev_eval run {run_s:.0f} s, queue wait {wait_s:.0f} s; gap to the next call mean {think_s:.0f} s, "
          f"median {statistics.median(gaps):.0f} s; one agent keeps {run_s / (run_s + think_s):.2f} of a slot busy")
    for n in (8, 16, 24):
        print(f"model, {n} agents: " + ", ".join(
            f"{c} slots {w:.0f} s ({u:.0%})" for c in (3, 5, 7, 11) for w, u in [mva(n, think_s, run_s, c)]))


if __name__ == "__main__":
    main()
