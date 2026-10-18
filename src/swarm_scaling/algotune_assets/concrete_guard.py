# --- v7 non-concrete solution guard (swarm_scaling/algotune_assets/concrete_guard.py; tasks.patch_task appends this file
# and a call of guard_is_solution(Task) to every task's evaluator.py for experiments after v6) ---
# Upstream AlgoTune's validator rejects a solution that is not built from plain values before it calls is_solution
# (AlgoTuner/utils/evaluator/solution_checks.py and runner.py, github.com/oripress/AlgoTune HEAD dff9914; added
# 2026-01-19, hardened 2026-03-10); the Harbor snapshot's verifier, which predates it, has no such check. In the v7
# calibration the ode team's pick returned array.array('d'), which skips building the list of floats the reference
# returns and only passes because is_solution converts any buffer with np.asarray. The two functions below are
# upstream's, verbatim; guard_is_solution is ours. Every is_solution call goes through it: dev_eval, finalize, the
# validity pass and the verifier.
import logging as _guard_logging
import numbers
from typing import Any

import numpy as np


def _has_container_proxy_overrides(
    value: Any,
    base_type: type,
    method_names: tuple[str, ...],
) -> bool:
    """Detect builtin container subclasses that override access/materialization hooks."""
    cls = type(value)
    if cls is base_type:
        return False

    for method_name in method_names:
        base_attr = getattr(base_type, method_name, None)
        cls_attr = getattr(cls, method_name, None)
        if cls_attr is not base_attr:
            return True

    return False


def find_nonconcrete_solution(solution: Any) -> str | None:
    stack = [("solution", solution)]
    seen: set[int] = set()
    nodes_seen = 0
    max_nodes = 1_000_000

    while stack:
        path, value = stack.pop()
        nodes_seen += 1
        if nodes_seen > max_nodes:
            return "Solution too large to validate safely"

        if value is None:
            continue
        if isinstance(value, (str, bytes, bool, numbers.Number, np.generic)):
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                obj_id = id(value)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                for idx, elem in enumerate(value.flat):
                    stack.append((f"{path}[{idx}]", elem))
            continue
        if isinstance(value, (list, tuple)):
            if isinstance(value, list) and _has_container_proxy_overrides(
                value,
                list,
                ("__iter__", "__getitem__", "__len__"),
            ):
                return f"Solution contains list-like proxy at {path}: {type(value).__name__}"
            if isinstance(value, tuple) and _has_container_proxy_overrides(
                value,
                tuple,
                ("__iter__", "__getitem__", "__len__"),
            ):
                return f"Solution contains tuple-like proxy at {path}: {type(value).__name__}"
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for idx, elem in enumerate(value):
                stack.append((f"{path}[{idx}]", elem))
            continue
        if isinstance(value, dict):
            if _has_container_proxy_overrides(
                value,
                dict,
                ("__iter__", "__getitem__", "__len__", "items", "keys", "values", "get"),
            ):
                return f"Solution contains dict-like proxy at {path}: {type(value).__name__}"
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for key, elem in value.items():
                if not isinstance(key, (str, bytes, bool, numbers.Number, np.generic)):
                    return (
                        f"Solution has non-primitive dict key at {path}: {type(key).__name__}"
                    )
                stack.append((f"{path}[{key!r}]", elem))
            continue
        if hasattr(value, "__array__"):
            return f"Solution contains array-like proxy at {path}: {type(value).__name__}"
        if hasattr(value, "__iter__"):
            return f"Solution contains non-list iterable at {path}: {type(value).__name__}"
        return f"Solution contains unsupported type at {path}: {type(value).__name__}"

    return None


def guard_is_solution(task_class: type) -> None:
    """Make task_class.is_solution reject a non-concrete solution (logged, False) before running its own checks."""
    checks = task_class.is_solution

    def is_solution(self, problem, solution, *args, **kwargs):
        reason = find_nonconcrete_solution(solution)
        if reason:
            _guard_logging.error(f"Non-concrete solution rejected: {reason}")
            return False
        return checks(self, problem, solution, *args, **kwargs)

    task_class.is_solution = is_solution
