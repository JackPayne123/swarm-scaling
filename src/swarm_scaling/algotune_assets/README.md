# Dev toolkit

The task description gives only the reference `solve` and `is_solution`. This directory and the `dev_eval` tool add what is missing to check your work before you finish.

- `reference_task.py` - the full reference `Task` class: `generate_problem(n, random_seed)`, `solve` (the reference being timed against), `is_solution`, with its imports and helpers. Use it for your own correctness checks.
- The `dev_eval` tool generates dev instances, checks a solver with `is_solution`, and reports the speedup over the reference. It copies the one file you name (as `solver.py`) to a separate, dedicated machine and runs the check there.

```
dev_eval(path="<your working directory>/solver.py")              # 20 instances, default seeds
dev_eval(path="<your working directory>/try2.py", n=5, reps=3)   # quicker, noisier
dev_eval(path="<your working directory>/solver.py", seed=500)    # different dev instances
```

Other option: `size` (problem size, default the final evaluation's; the output shows the size used).

The dedicated machine runs one evaluation at a time. Calls from every agent working on this task wait in one queue, in the order they were made, and each result says how long the call waited.

## How it matches the final evaluation

- Same generator and the same problem size. Dev instances use different seeds from the final instances.
- Same check: your output goes to `is_solution` exactly as `solve` returns it, with no conversion. Return the types `is_solution` expects.
- Same timing: per instance, one untimed warmup call of the reference and of your solver, then alternating timed calls of each (10 by default), minimum of each. Speedup is total reference time divided by total solver time.
- Same thread check: a solver whose threads keep using CPU while the reference is being timed is scored invalid.
- The final evaluation uses 100 instances. An invalid output or a solver slower than the reference scores 1.0, and a missing `/app/solver.py` scores 0.

## Notes

- Only `/app/solver.py` is evaluated, loaded as a single file. Keep it self-contained; it cannot import sibling files.
- Timings you take in your own container are affected by whatever else is running there. The `dev_eval` tool's machine runs nothing else while it times a solver.
- The container has no network access.
