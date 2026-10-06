# Dev toolkit

The task description gives only the reference `solve` and `is_solution`. This directory adds what is missing to check your work before you finish.

- `reference_task.py` - the full reference `Task` class: `generate_problem(n, random_seed)`, `solve` (the reference being timed against), `is_solution`, with its imports and helpers.
- `dev_eval.py` - generates dev instances, checks your solver with `is_solution`, and reports the speedup over the reference.
- `config.json` - the problem size `n` the final evaluation uses.

```
python /app/dev/dev_eval.py /app/solver.py            # 20 instances, default seeds
python /app/dev/dev_eval.py my_try/solver.py --n 5 --reps 3   # quicker, noisier
python /app/dev/dev_eval.py /app/solver.py --seed 500 # different dev instances
```

Other options: `--size N` (problem size, default the final evaluation's), `--json-out PATH`. Exit code 0 means every instance was valid.

## How it matches the final evaluation

- Same generator and the same problem size. Dev instances use different seeds from the final instances.
- Same check: your output goes to `is_solution` exactly as `solve` returns it, with no conversion. Return the types `is_solution` expects.
- Same timing: per instance, one untimed warmup call of the reference and of your solver, then alternating timed calls of each (10 by default), minimum of each. Speedup is total reference time divided by total solver time.
- The final evaluation uses 100 instances. An invalid output or a solver slower than the reference scores 1.0, and a missing `/app/solver.py` scores 0.

## Notes

- Only `/app/solver.py` is evaluated, loaded as a single file. Keep it self-contained; it cannot import sibling files.
- Anything else running in the container (including other `dev_eval.py` runs) distorts timings. Treat speedups measured under load as rough.
- The container has no network access.
