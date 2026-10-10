"""scripts/run_telemetry.py: cache rewrites after dev_eval waits, scorer slot use. Synthetic events, no logs."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_telemetry", ROOT / "scripts/run_telemetry.py")
telemetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry)


def model(t: float, fresh: int, write: int, read: int) -> tuple:
    return ("model", t, {"input_tokens": fresh, "input_tokens_cache_write": write, "input_tokens_cache_read": read})


def test_only_a_call_after_a_dev_eval_result_that_rewrites_most_of_its_input_is_flagged():
    # A long scorer queue wait lets the prompt cache expire, so the next call writes the whole context again.
    seq = [
        model(0, 10, 40_000, 0),  # first call writes the cache: no dev_eval before it
        ("tool", 5, "dev_eval", "[dev_eval] waited 400.0s in the queue, ran 12.0s.\nspeedup 2.0x"),
        model(420, 10, 44_000, 0),  # flagged
        ("tool", 430, "dev_eval", "[dev_eval] waited 3.0s in the queue, ran 12.0s."),
        model(450, 10, 1_000, 44_000),  # cache still warm: not flagged
        ("tool", 460, "bash", "ok"),
        model(470, 10, 45_000, 0),  # a rewrite, but not after a dev_eval: not flagged
    ]
    assert telemetry.cache_rewrites(seq) == [
        {"t_s": 420.0, "dev_eval_wait_s": 400.0, "cache_write": 44_000, "input_total": 44_010}
    ]


def test_slot_busy_fraction_counts_only_the_part_of_each_job_inside_the_window():
    jobs = [
        {"slot": 0, "started_at": 50, "ended_at": 150},  # half inside [100, 200]
        {"slot": 1, "started_at": 100, "ended_at": 200},
        {"slot": 1, "started_at": 300, "ended_at": 400},  # after the window
    ]
    assert telemetry.slot_use(jobs, 100, 200, 2) == [0.5, 1.0]
