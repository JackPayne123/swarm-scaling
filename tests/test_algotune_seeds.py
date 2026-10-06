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
