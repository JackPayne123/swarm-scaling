"""Flags a solver whose threads keep using CPU while the reference is timed. Linux (/proc), standard library only.

AlgoTune times the reference and the solver interleaved in one process, so threads the solver starts can still
run while the reference is timed, slow it down and inflate the speedup. In pilot 3 a solver that left 4
spin-waiting worker threads running scored 3837x on dev_eval, with the reference about 20-28% slower.
Used by dev_eval.py and by our copy of the verifier (tasks.with_thread_guard), which mark such a run invalid.

Which threads count: every thread that exists at the start of a timed reference call and is not the
reference's own. The reference's own are the threads alive after one untimed reference call made before the
solver is imported (the main thread and the reference's BLAS pools), plus any thread that first appears during
a reference call (e.g. OpenBLAS restarting its pool after a solver shut it down). So a solver that calls LAPACK
through numpy's own OpenBLAS, or shuts that pool down, is not counted. CPU of a thread that exits during a
reference call is not seen.
"""

import os

# Allowed solver-thread CPU per run: MAX_SHARE of the reference's total timed seconds, plus SPIN_TAIL_S per timed
# reference call. MAX_SHARE = 0.1 is a tenth of one CPU on average; one helper thread spinning throughout uses 1.0.
# SPIN_TAIL_S covers OpenMP's default wait: libgomp workers spin for GOMP_SPINCOUNT = 300,000 iterations (about
# 3 ms by libgomp's own estimate) after a parallel region before sleeping, so a solver's OpenMP loop can leave up
# to 7 workers (OMP_NUM_THREADS=8) spinning for ~3 ms into the next reference call: 21 ms, rounded up to 25 ms.
# Not measured in our containers; dev_eval and the verifier report the measured CPU (scripts/smoke_thread_guard.py).
MAX_SHARE = 0.1
SPIN_TAIL_S = 0.025

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

    def allowed_s(self) -> float:
        return MAX_SHARE * self.reference_s + SPIN_TAIL_S * self.timings

    def summary(self) -> dict:
        return {
            "solver_thread_cpu_s": round(self.solver_cpu_s, 3),
            "reference_timed_s": round(self.reference_s, 3),
            "reference_timings": self.timings,
            "allowed_s": round(self.allowed_s(), 3),
        }

    def error(self) -> str | None:
        """Why the run is invalid, or None."""
        if self.solver_cpu_s <= self.allowed_s():
            return None
        return (
            f"solver threads kept running during reference timing: they used {self.solver_cpu_s:.3f} s of CPU "
            f"while the reference was timed for {self.reference_s:.3f} s in {self.timings} calls "
            f"(allowed {self.allowed_s():.3f} s)"
        )
