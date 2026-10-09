"""Live checks for harness-v4's thread check and checker cleanup (Docker, no model spend). Not run yet.

Run from this worktree when nothing else is using Docker (each part starts containers one at a time):
    uv run python scripts/smoke_harness_v4.py                  # all parts, about 20-40 min (not measured)
    uv run python scripts/smoke_harness_v4.py --part threads   # or verifier, cleanup

threads   dev_eval.py, in each pilot-3 task image with the checker's limits (8 CPUs 8-15, 8 GB, no network), on:
          the reference solver (expect valid, not flagged), the reference plus 4 native spinning threads
          (expect "solver threads kept running"), and every solver pilot 3 selected for that task, read from
          the eval logs (expect valid; their solver-thread CPU is printed to calibrate thread_guard's threshold).
verifier  our patched verifier (seed offset + thread check) for generalized-eigenvalues-real on the reference
          solver (expect Validity: True) and the spinning one (expect the thread error and Validity: False).
cleanup   one AlgoTune sample through Inspect with a scripted mockllm agent whose solver starts a detached
          `sleep 600` when imported: two dev_eval calls, a publish of daemon.py, finalize and scoring. Expect
          each timed checker run after the first to report "killed 1", and the publish to say "saved as".
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from swarm_scaling.algotune_devkit import ASSETS
from swarm_scaling.tasks import COMPOSE_DIR, VERIFIER_DIR, with_thread_guard

TASKS = ("cvar-projection", "dst-type-ii-scipy-fftpack", "generalized-eigenvalues-real")
CHECKER_LIMITS = ["--network", "none", "--cpus", "8", "--cpuset-cpus", "8-15", "--memory", "8g"]

REFERENCE_SOLVER = '''import importlib.util
_spec = importlib.util.spec_from_file_location("ref", "/work/dev/reference_task.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


class Solver:
    def __init__(self):
        self.task = _ref.Task()

    def solve(self, problem):
        return self.task.solve(problem)
'''

# Pilot 3's 3837x: helper threads that keep spinning after the solver returns. Native threads (no GIL), as there.
SPINNERS = r'''
import ctypes, os, subprocess, tempfile
_C = "#include <pthread.h>\nstatic void *spin(void *a) { volatile unsigned long x = 0; for (;;) x++; return 0; }\n" \
     "void start(int n) { for (int i = 0; i < n; i++) { pthread_t t; pthread_create(&t, 0, spin, 0); pthread_detach(t); } }\n"
_d = tempfile.mkdtemp()
open(os.path.join(_d, "spin.c"), "w").write(_C)
subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-pthread", os.path.join(_d, "spin.c"), "-o", os.path.join(_d, "spin.so")], check=True)
ctypes.CDLL(os.path.join(_d, "spin.so")).start(4)
'''

DAEMON = '''
import subprocess
subprocess.Popen(["sleep", "600"], start_new_session=True)  # outlives this process
'''


def image(task: str) -> str:
    compose = yaml.safe_load((COMPOSE_DIR / f"algotune_{task}-compose.yaml").read_text())
    return compose["services"]["checker"]["image"]


def problem_size(task: str) -> int:
    text = (VERIFIER_DIR / f"algotune_{task}" / "test_outputs.py").read_text()
    return int(re.search(r"^PROBLEM_SIZE = (\d+)", text, re.M).group(1))


def selected_solvers(logs: Path, task: str) -> dict[str, str]:
    from inspect_ai.log import read_eval_log

    found = {}
    for path in sorted(logs.glob(f"pilot3-*-{task}/*.eval")):
        for s in read_eval_log(str(path)).samples or []:
            source = (s.metadata or {}).get("finalize", {}).get("selected_source")
            if source:
                found[f"{path.parent.name}-e{s.epoch}"] = source
    return found


def docker(task: str, work: Path, *cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    run = ["docker", "run", "--rm", *CHECKER_LIMITS, "-v", f"{work}:/work", image(task), *cmd]
    return subprocess.run(run, capture_output=True, text=True, timeout=timeout)


def threads_part(logs: Path, n: int) -> None:
    for task in TASKS:
        work = Path(tempfile.mkdtemp(prefix=f"smoke-v4-{task}-"))
        (work / "dev").mkdir()
        (work / "solvers").mkdir()
        shutil.copy(VERIFIER_DIR / f"algotune_{task}" / "evaluator.py", work / "dev" / "reference_task.py")
        for name in ("dev_eval.py", "thread_guard.py"):
            shutil.copy(ASSETS / name, work / "dev" / name)
        (work / "dev" / "config.json").write_text(json.dumps({"problem_size": problem_size(task)}))
        solvers = {"reference": REFERENCE_SOLVER, "reference+spinners": REFERENCE_SOLVER + SPINNERS}
        solvers.update(selected_solvers(logs, task))
        print(f"\n== {task}: {len(solvers)} solvers, n={n}")
        for name, source in solvers.items():
            (work / "solvers" / f"{name}.py").write_text(source)
            proc = docker(task, work, "python", "/work/dev/dev_eval.py", f"/work/solvers/{name}.py",
                          "--n", str(n), "--json-out", f"/work/{name}.json")  # fmt: skip
            try:
                r = json.loads((work / f"{name}.json").read_text())
            except FileNotFoundError:
                print(f"  {name}: NO RESULT (exit {proc.returncode}) {proc.stderr[-400:]}")
                continue
            expect = "flagged" if name == "reference+spinners" else "not flagged"
            flagged = any(e.startswith("solver threads kept running") for e in r["errors"])
            verdict = "OK" if flagged == (expect == "flagged") else "UNEXPECTED"
            print(f"  {verdict} {name}: expect {expect}; valid={r['valid']} speedup={r['speedup']} "
                  f"thread_check={r['thread_check']} errors={[e.splitlines()[-1] for e in r['errors'][:2]]}")  # fmt: skip


def verifier_part() -> None:
    task = "generalized-eigenvalues-real"
    for name, source in (("reference", REFERENCE_SOLVER), ("reference+spinners", REFERENCE_SOLVER + SPINNERS)):
        work = Path(tempfile.mkdtemp(prefix="smoke-v4-verifier-"))
        tests = work / "tests"
        tests.mkdir()
        seeded = VERIFIER_DIR / f"algotune_{task}"
        shutil.copy(seeded / "evaluator.py", tests)
        shutil.copy(ASSETS / "thread_guard.py", tests)
        # the cached copy already has the seed patch; a made-up offset, not the experiment's secret one
        (tests / "test_outputs.py").write_text(with_thread_guard((seeded / "test_outputs.py").read_text()))
        (tests / "seed_offset").write_text("1234567\n")
        (work / "dev").mkdir()
        shutil.copy(seeded / "evaluator.py", work / "dev" / "reference_task.py")
        (work / "solver.py").write_text(source)
        cmd = "mkdir -p /app /logs/verifier && cp /work/solver.py /app/solver.py && cp -r /work/tests /tests && " \
              "pytest /tests/test_outputs.py -rA -s; cat /logs/verifier/reward.txt"  # fmt: skip
        proc = docker(task, work, "sh", "-c", cmd, timeout=3600)
        lines = [ln for ln in (proc.stdout + proc.stderr).splitlines()
                 if "Thread check" in ln or "solver threads" in ln or ln.startswith(("Validity", "Raw Speedup", "Final Reward"))]  # fmt: skip
        print(f"\n== verifier, {name} (expect Validity: {name == 'reference'}):")
        print("\n".join(f"  {ln}" for ln in lines) or proc.stdout[-1500:])


def cleanup_part() -> None:
    from inspect_ai import eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import ModelOutput, get_model

    from swarm_scaling.algotune_devkit import SOLVER_PATH, algotune_agent_tools, algotune_finalize
    from swarm_scaling.swarm import swarm
    from swarm_scaling.tasks import algotune_task

    path = "/app/agents/agent_0/daemon.py"
    source = REFERENCE_SOLVER.replace("/work/dev/", "/app/dev/") + DAEMON
    calls = [
        ("python", {"code": f"open({path!r}, 'w').write({source!r})"}),
        ("dev_eval", {"path": path, "n": 3}),
        ("dev_eval", {"path": path, "n": 3}),
        ("publish_candidate", {"path": path, "note": "daemon"}),
        ("submit", {"answer": "done"}),
    ]
    agent = get_model("mockllm/model", custom_outputs=[ModelOutput.for_tool_call("mockllm/model", t, a) for t, a in calls])
    (log,) = eval(
        algotune_task(split="pilot"),
        solver=swarm(models=[agent], per_agent_tokens=200_000, finalize=algotune_finalize, agent_tools=algotune_agent_tools,
                     final_path=SOLVER_PATH, candidate_file="solver.py", budget_warnings=()),  # fmt: skip
        model="mockllm/model",
        sample_id="algotune/cvar-projection",
        max_samples=1,
        max_sandboxes=1,
        display="none",
    )
    print("\n== cleanup: status", log.status, log.error or "")
    s = read_eval_log(log.location).samples[0]
    print("  checker baseline PIDs:", s.metadata.get("checker_baseline_pids"))
    for e in s.events:
        if e.event == "tool" and e.function == "publish_candidate":
            print("  publish:", e.result)
    print("  dev_eval cleanups (expect killed 0, killed 1):", [c["cleanup"] for c in s.metadata["checker"]["calls"]])
    fin = s.metadata.get("finalize", {})
    print("  finalize (expect killed 1 for the published one):",
          [(c["kind"], c["valid"], c["error"][:60], c.get("cleanup")) for c in fin.get("candidates", [])])  # fmt: skip
    for k, v in (s.scores or {}).items():
        print(f"  score {k}={v.value} cleanup (expect killed 1): {(v.metadata or {}).get('checker', {}).get('cleanup')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=("threads", "verifier", "cleanup", "all"), default="all")
    p.add_argument("--logs", type=Path, default=Path.home() / "projects/swarm-scaling/logs", help="pilot-3 eval logs")
    p.add_argument("--n", type=int, default=10, help="dev instances per dev_eval run")
    args = p.parse_args()
    if args.part in ("threads", "all"):
        threads_part(args.logs, args.n)
    if args.part in ("verifier", "all"):
        verifier_part()
    if args.part in ("cleanup", "all"):
        cleanup_part()
