import ast
from pathlib import Path

import pytest

from swarm_scaling.tasks import SEED_OFFSET_ENV, _SEED_HELPER, _seed_offset, seed_with_offset

VERIFIER = Path(__file__).resolve().parents[1] / "research/harbor-examples/algotune_task/tests/test_outputs.py"


def test_verifier_scores_on_offset_seeds_not_0_to_99(tmp_path):
    # Agents hold generate_problem, so a lookup table for seeds 0..99 would fake any speedup.
    patched = seed_with_offset(VERIFIER.read_text())
    call = next(
        n for n in ast.walk(ast.parse(patched)) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "generate_problem"
    )
    seed = ast.Expression(next(k.value for k in call.keywords if k.arg == "random_seed"))
    (tmp_path / "seed_offset").write_text("908272966\n")
    namespace = {"Path": Path, "__file__": str(tmp_path / "test_outputs.py")}
    exec(_SEED_HELPER, namespace)
    seeds = [eval(compile(seed, "<seed>", "eval"), {**namespace, "i": i}) for i in range(100)]
    assert seeds == list(range(908272966, 908272966 + 100))


def test_offset_is_gone_before_the_solver_is_imported(tmp_path):
    # The solver's import-time code must not be able to read the offset: it is not in the environment,
    # its file is deleted on first use, and problems are generated before the solver fixture runs.
    patched = seed_with_offset(VERIFIER.read_text())
    assert SEED_OFFSET_ENV not in patched
    assert "def performance_results(problem_set, solver_instance)" in patched
    (tmp_path / "seed_offset").write_text("1234567\n")
    namespace = {"Path": Path, "__file__": str(tmp_path / "test_outputs.py")}
    exec(_SEED_HELPER, namespace)
    assert namespace["_take_seed_offset"]() == 1234567
    assert not (tmp_path / "seed_offset").exists()
    assert namespace["_take_seed_offset"]() == 1234567


def test_a_changed_verifier_is_refused_rather_than_left_on_seeds_0_to_99():
    with pytest.raises(ValueError):
        seed_with_offset("problem = task.generate_problem(n=PROBLEM_SIZE, random_seed=j)")


def test_one_offset_for_the_whole_experiment(tmp_path, monkeypatch):
    # Arms run as separate evals; a fresh offset per run would score them on different instances.
    monkeypatch.delenv(SEED_OFFSET_ENV, raising=False)
    path = tmp_path / ".algotune_seed_offset"
    first = _seed_offset(path, log_dir=tmp_path / "no-logs")
    assert path.read_text().strip() == str(first)
    assert _seed_offset(path, log_dir=tmp_path / "no-logs") == first
    monkeypatch.setenv(SEED_OFFSET_ENV, "1234567")
    assert _seed_offset(path) == 1234567


def test_with_the_v7_dev_cap_every_seed_range_is_disjoint():
    # v6: dev_eval seed s timed instance 10,000 + s, so seeds 50,000+ reached finalize's stage-1 instances (5 calls did).
    from swarm_scaling.algotune_devkit import DEV_SEED_CAP, DISJOINT_DEV_SEED_CAP, seed_ranges

    ranges = sorted(seed_ranges(DISJOINT_DEV_SEED_CAP).items(), key=lambda kv: kv[1])
    assert [name for name, _ in ranges] == ["dev_eval", "finalize", "score_dev", "validity", "hidden"]
    assert all(a[1][1] <= b[1][0] for a, b in zip(ranges, ranges[1:]))
    v6 = seed_ranges(DEV_SEED_CAP)
    assert v6["dev_eval"][1] > v6["finalize"][0]  # why v7 needs --disjoint-dev-seeds


def test_each_experiment_has_its_own_offset_at_least_100_seeds_from_any_other(tmp_path, monkeypatch):
    # v7's hidden instances must be new: the v6 ones informed the v6 diagnosis. v6 keeps its own to reproduce v6.
    from swarm_scaling import tasks

    monkeypatch.delenv(SEED_OFFSET_ENV, raising=False)
    monkeypatch.setattr(tasks, "SEED_OFFSET_FILE", tmp_path / ".algotune_seed_offset")
    tasks.SEED_OFFSET_FILE.write_text("5000000\n")
    draws = iter([5_000_050 - tasks.SEED_OFFSET_RANGE[0], 7_000_000 - tasks.SEED_OFFSET_RANGE[0]])
    monkeypatch.setattr(tasks.secrets, "randbelow", lambda _: next(draws))  # the first draw is 50 from v6's: redrawn
    v7 = tasks._seed_offset(tasks.seed_offset_file("v7"), log_dir=tmp_path / "no-logs", experiment="v7")
    assert v7 == 7_000_000 and tasks.seed_offset_file("v7").name == ".algotune_seed_offset.v7"
    assert tasks._seed_offset(tasks.seed_offset_file("v6"), log_dir=tmp_path / "no-logs") == 5_000_000
    assert tasks._seed_offset(tasks.seed_offset_file("v7"), log_dir=tmp_path / "no-logs", experiment="v7") == v7
    with pytest.raises(ValueError, match="experiment must be one of"):
        tasks.seed_offset_file("v8")


def test_v7_task_fixes_reach_the_pinned_harbor_code_and_v6_keeps_harbors():
    # The fixes (upstream verifier ports, a seeded generator) must reach the agents' toolkit, the instruction and the
    # verifier for v7, or a v7 pca would still accept any rotated subspace; v6 must still score on Harbor's code.
    from inspect_harbor import algotune

    from swarm_scaling.tasks import ALGOTUNE_REF, TASK_PATCHES, _harbor_name, patch_task

    base = algotune(ref=ALGOTUNE_REF, dataset_task_names=[_harbor_name(n) for n in TASK_PATCHES])
    for sample in base.dataset:
        name = next(n for n in TASK_PATCHES if _harbor_name(n) == str(sample.id))
        code = (Path(sample.metadata["tests_dir"]) / "evaluator.py").read_text()
        assert patch_task(name, code, "v6") == code
        fixed = patch_task(name, code, "v7")
        assert all(fixed.count(new) == 1 for _, new, _ in TASK_PATCHES[name])
        assert patch_task(name, fixed, "v7") == fixed  # a seeded verifier copy is already fixed
        shown = patch_task(name, sample.input, "v7", instruction=True)
        assert all(shown.count(new) == 1 for _, new, in_instruction in TASK_PATCHES[name] if in_instruction)


def test_a_task_fix_whose_code_changed_is_refused_rather_than_skipped(monkeypatch):
    from swarm_scaling import tasks

    monkeypatch.setitem(tasks.TASK_PATCHES, "t", [("old line\n", "new line\n", False)])
    with pytest.raises(ValueError):
        tasks.patch_task("t", "something else\n", "v7")


def test_v7_is_solution_rejects_a_non_concrete_output_as_upstream_does_and_v6_keeps_accepting_it():
    # v7 calibration: the ode team's pick returned array.array('d'); Harbor's is_solution converts it with np.asarray and
    # passes it, upstream AlgoTune's validator rejects any non-list iterable before is_solution is called.
    import array

    from swarm_scaling.tasks import CONCRETE_GUARD, patch_task

    code = ("import numpy as np\n\n\nclass Task:\n"
            "    def is_solution(self, problem, solution, atol=1e-8):\n"
            "        return np.allclose(np.asarray(solution, dtype=float), problem['y'], atol=atol)\n")  # fmt: skip

    def task(experiment: str):
        namespace: dict = {}
        exec(patch_task("any_task", code, experiment), namespace)
        return namespace["Task"]()

    problem = {"y": [1.0, 2.0, 3.0]}
    v7, v6 = task("v7"), task("v6")
    assert v7.is_solution(problem, [1.0, 2.0, 3.0]) and v7.is_solution(problem, (1.0, 2.0, 3.0))
    assert v7.is_solution(problem, [1.0, 2.0, 3.1], atol=0.2)  # a task's own keyword arguments still reach it
    assert not v7.is_solution(problem, array.array("d", problem["y"]))
    assert not v7.is_solution(problem, iter(problem["y"]))
    assert v6.is_solution(problem, array.array("d", problem["y"]))  # Harbor's code, unchanged for v6
    assert patch_task("any_task", patch_task("any_task", code, "v7"), "v7").count(CONCRETE_GUARD) == 1
    assert CONCRETE_GUARD not in patch_task("any_task", code, "v7", instruction=True)
