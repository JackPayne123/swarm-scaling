"""Flags a solver that slows the reference while it is timed. Linux (/proc), standard library only.

AlgoTune times the reference and the solver interleaved in one process, so whatever a solver leaves running
(spin-waiting threads, a busy thread pool) can slow the reference and inflate the speedup. In pilot 3 a solver
that left 4 spin-waiting worker threads running scored 3837x on dev_eval, with the reference about 20-28% slower.
Used by dev_eval.py and by our copy of the verifier (tasks.with_thread_guard), which mark such a run invalid.

The criterion is the harm itself: `time_reference` times the reference on every instance before the solver is
imported (same warmup, repeats, thread settings and CPUs; nothing of the solver exists yet), and
`reference_inflation_error` marks the run invalid when the reference, timed interleaved with the solver, totals
more than MAX_REFERENCE_INFLATION times that over the instances both timed. The first version of this check
invalidated on CPU used by solver threads, and it flagged two legitimate pilot-3 eigenvalue solvers whose
`blas_thread_shutdown_` made OpenBLAS restart its pool (reference time normal, 2026-10-09).

`ThreadGuard` still measures the CPU of solver threads during the reference's timed calls, for the logs only:
the reference's own threads are those alive after an untimed reference call made before the solver is imported,
plus any thread that first appears during a reference call. CPU of a thread that exits during a call is not seen.
"""

import os
import time

# Invalid when the reference, timed interleaved with the solver, takes more than this times its time measured
# before the solver was imported. The legitimate pilot-3 solvers this must pass are listed in HARNESS.md.
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
    times the reference, without the solver calls in between). Call before the solver is imported."""
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
