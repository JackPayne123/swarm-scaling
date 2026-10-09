"""Flags a solver that slows the reference while it is timed. Linux (/proc), standard library only.

AlgoTune times the reference and the solver interleaved in one process, so whatever a solver leaves running
(spin-waiting threads, a busy thread pool) can slow the reference and inflate the speedup. In pilot 3 a solver
that left 4 spin-waiting worker threads running scored 3837x on dev_eval, with the reference about 20-28% slower.
Used by dev_eval.py and by our copy of the verifier (tasks.with_thread_guard), which mark such a run invalid.

The criterion is the harm itself. After each instance's interleaved timing, `AloneTimer` times the reference on
the same instance alone (same warmup, repeats, thread settings and CPUs) in a child process started before the
solver was imported, so no solver code exists there. `reference_inflation_error` marks the run invalid when the
interleaved reference totals more than MAX_REFERENCE_INFLATION times the alone reference over the timed instances.
Inside a container the two never run at the same time: the child pauses this process (SIGSTOP, then SIGCONT)
while it times, so nothing the solver left running here competes with it (nor the reference's own OpenBLAS pool,
whose idle threads keep spinning for a while), and this process pauses the child while it times. Alternating per
instance cancels drift in the machine's speed, which a single "alone" pass before the solver did not: on a busy
Mac it made legitimate solvers read 0.78x and 1.45x (2026-10-09). The first version of this check invalidated on
CPU used by solver threads, and it flagged two legitimate pilot-3 eigenvalue solvers whose `blas_thread_shutdown_`
made OpenBLAS restart its pool (2026-10-09).

`ThreadGuard` still measures the CPU of solver threads during the reference's timed calls, for the logs only:
the reference's own threads are those alive after an untimed reference call made before the solver is imported,
plus any thread that first appears during a reference call. CPU of a thread that exits during a call is not seen.
"""

import json
import os
import signal
import subprocess
import sys
import time

# Invalid when the reference, timed interleaved with the solver, takes more than this times its time measured alone.
MAX_REFERENCE_INFLATION = 1.15

_TICKS_PER_S = os.sysconf("SC_CLK_TCK")


def thread_cpu_seconds() -> dict[int, float]:
    """User + system CPU seconds of each thread of this process, by thread id (/proc/self/task/<tid>/stat)."""
    cpu = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                stat = f.read()
        except (FileNotFoundError, ProcessLookupError):  # the thread exited after the listing
            continue
        fields = stat[stat.rindex(")") + 2 :].split()  # fields after "pid (comm) "; comm may contain spaces
        cpu[int(tid)] = (int(fields[11]) + int(fields[12])) / _TICKS_PER_S  # utime, stime (stat fields 14, 15)
    return cpu


class ThreadGuard:
    """Create after an untimed reference call and before the solver is imported; wrap every reference call."""

    def __init__(self, read=thread_cpu_seconds) -> None:
        self._read = read
        self._reference_threads = set(read())
        self._start: dict[int, float] = {}
        self.solver_cpu_s = 0.0  # CPU used by solver threads during timed reference calls
        self.reference_s = 0.0  # total time of the timed reference calls
        self.timings = 0

    def before_reference(self) -> None:
        self._start = self._read()

    def after_reference(self, timed_ns: int | None) -> None:
        """Call after each reference call with its timed duration, or None for an untimed (warmup) call."""
        end = self._read()
        self._reference_threads |= end.keys() - self._start.keys()
        if timed_ns is None:
            return
        self.solver_cpu_s += sum(
            end[tid] - self._start[tid] for tid in end.keys() & self._start.keys() if tid not in self._reference_threads
        )
        self.reference_s += timed_ns / 1e9
        self.timings += 1

    def summary(self) -> dict:
        return {
            "solver_thread_cpu_s": round(self.solver_cpu_s, 3),
            "reference_timed_s": round(self.reference_s, 3),
            "reference_timings": self.timings,
        }


def time_reference(solve, problems: list, reps: int) -> list[int]:
    """Per problem, the minimum of `reps` timed reference calls after one untimed call, in ns (as the verifier
    times the reference, without the solver calls in between)."""
    mins = []
    for problem in problems:
        solve(problem)
        times = []
        for _ in range(reps):
            start = time.perf_counter_ns()
            solve(problem)
            times.append(time.perf_counter_ns() - start)
        mins.append(min(times))
    return mins


def reference_inflation(with_solver_ns: list[int], alone_ns: list[int]) -> float | None:
    """Total reference time timed with the solver / timed alone, over the same instances; None if none."""
    alone = sum(alone_ns)
    return sum(with_solver_ns) / alone if with_solver_ns and alone else None


def reference_inflation_error(with_solver_ns: list[int], alone_ns: list[int]) -> str | None:
    """Why the run is invalid, or None."""
    ratio = reference_inflation(with_solver_ns, alone_ns)
    if ratio is None or ratio <= MAX_REFERENCE_INFLATION:
        return None
    return (
        f"the reference ran {ratio:.3f}x slower while timed with the solver: {sum(with_solver_ns) / 1e9:.4f} s "
        f"interleaved with the solver vs {sum(alone_ns) / 1e9:.4f} s timed before the solver was imported, "
        f"over the same {len(with_solver_ns)} instances (allowed {MAX_REFERENCE_INFLATION}x)"
    )


def in_container() -> bool:
    """Positive check for a Linux Docker container, the only place a process is ever paused by this module."""
    return sys.platform.startswith("linux") and os.path.exists("/.dockerenv")


class AloneTimer:
    """Times the reference alone, one instance at a time, in a child process. Create it before the solver is imported.

    The child loads the reference module from `reference_path` and generates each problem itself from the
    generate_problem arguments it is sent (untimed, while this process runs). Inside a container (in_container())
    the child pauses this process while it times and this process pauses the child between requests; signals go
    only to these two processes. Elsewhere (unit tests) the child times while this process waits for its answer.
    """

    def __init__(self, reference_path: str, reps: int) -> None:
        self.pause = in_container()
        self.proc = subprocess.Popen(
            [sys.executable, __file__, reference_path, str(reps), "1" if self.pause else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )  # fmt: skip

    def time(self, **generate_kwargs) -> int:
        """The reference's minimum timed call on generate_problem(**generate_kwargs), timed alone, in ns."""
        if self.pause:
            os.kill(self.proc.pid, signal.SIGCONT)
        self.proc.stdin.write(json.dumps(generate_kwargs) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:  # the reference module may print; answers are tagged
            if line.startswith("ALONE "):
                if self.pause:
                    os.kill(self.proc.pid, signal.SIGSTOP)  # its idle BLAS threads must not compete with ours
                return int(line.split()[1])
        raise RuntimeError(f"the reference timing process exited ({self.proc.wait()})")

    def close(self) -> None:
        if self.pause:
            os.kill(self.proc.pid, signal.SIGCONT)
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


def _pause(pid: int) -> None:
    """SIGSTOP `pid` and wait until the kernel reports it stopped. Fails loudly if it never stops: the init process
    of a PID namespace (e.g. `docker run image python ...` without --init) ignores SIGSTOP from inside it, and the
    alone timing would then silently compete with whatever the solver left running."""
    os.kill(pid, signal.SIGSTOP)
    for _ in range(200):
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        if stat[stat.rindex(")") + 2] in "Tt":
            return
        time.sleep(0.005)
    raise RuntimeError(f"process {pid} did not stop (is it PID 1 of its namespace? run the container with --init)")


def _alone_child(reference_path: str, reps: int, pause_parent: bool) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("reference_alone", reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task, parent = module.Task(), os.getppid()
    for line in sys.stdin:
        problem = task.generate_problem(**json.loads(line))
        if pause_parent:
            _pause(parent)
        try:
            (ns,) = time_reference(task.solve, [problem], reps)
        finally:
            if pause_parent:
                os.kill(parent, signal.SIGCONT)
        print(f"ALONE {ns}", flush=True)


if __name__ == "__main__":
    _alone_child(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1")
