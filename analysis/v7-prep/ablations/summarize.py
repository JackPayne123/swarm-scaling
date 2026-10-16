"""Collect the per-task JSONs from ablate_m.py into results.json and print the ablations.md table rows.
Rule 2(b): share = log(variant speedup) / log(best published local speedup in pub_out, 0 exceptions);
override to S when some variant has share >= 0.75 and is valid 25/25.
Also reported as checks (not used by the rule): share against the same-run published speedup, and
CPU-time speedups (process_time of the same min-wall calls), which are less sensitive to machine load.
usage: python3 summarize.py"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["chebyshev_center", "generalized_eigenvectors_complex", "ode_lorenz96_nonchaotic", "graph_global_efficiency", "pagerank",
         "minimum_spanning_tree", "shortest_path_dijkstra", "l0_pruning", "l1_pruning", "pde_heat1d", "randomized_svd"]


def summarize(d):
    rc = d["ref_cpu_over_wall"]
    den = d["denominator"]["local_speedup"]
    pub = [v for v in d["arms"].values() if v["is_published"]][0]
    pub_cpu = pub["speedup"] * rc / pub["cpu_over_wall"]
    variants = {}
    for name, v in d["arms"].items():
        if v["is_published"]:
            continue
        sp = v["speedup"]
        cpu_sp = sp * rc / v["cpu_over_wall"] if v["n_exceptions"] == 0 else None
        variants[name] = {
            "speedup": sp, "valid": v["valid"], "share_of_best_log_speedup": math.log(sp) / math.log(den),
            "share_vs_same_run_published": math.log(sp) / math.log(pub["speedup"]),
            "cpu_time_speedup": cpu_sp, "cpu_share_vs_pub_out": math.log(cpu_sp) / math.log(den) if cpu_sp else None,
            "cpu_over_wall": v["cpu_over_wall"], "n_exceptions": v["n_exceptions"],
        }
    hit = [k for k, v in variants.items() if v["share_of_best_log_speedup"] >= 0.75 and v["valid"] == f"{d['n_done']}/{d['n_requested']}" and d["n_done"] == 25]
    return {
        "problem_size": d["problem_size"], "n_instances": d["n_done"], "elapsed_s": d["elapsed_s"], "ref_cpu_over_wall": rc,
        "denominator": {"solver": d["denominator"]["solver"], "local_speedup": den, "n_invalid_of_400": d["denominator"]["n_invalid_of_400"],
                        "source": os.path.basename(d["denominator"]["source"])},
        "same_run_published": {"solver": d["denominator"]["solver"], "speedup": pub["speedup"], "valid": pub["valid"],
                               "cpu_time_speedup": pub_cpu, "cpu_over_wall": pub["cpu_over_wall"]},
        "variants": variants, "variants_reaching_75pct_valid": hit,
        "route_class": "M", "stratum": "S" if hit else "M",
    }


res = {t: summarize(json.load(open(f"{HERE}/{t}.json"))) for t in TASKS}
rep = {}
for t in TASKS:
    p = f"{HERE}/replicate/{t}.json"
    if os.path.exists(p):
        rep[t] = summarize(json.load(open(p)))
out = {"rule": "2(b): log(variant)/log(best pub_out local speedup, 0 exceptions) >= 0.75 with 25/25 valid -> S",
       "method": "ablate_m.py: Harbor input, 1 thread (OMP/OPENBLAS/MKL/NUMEXPR/NUMBA=1), seeds 3,000,000+i, 25 instances, warmup + min of 2",
       "tasks": res, "replicate_borderline": rep}
json.dump(out, open(f"{HERE}/results.json", "w"), indent=1)

for t, r in res.items():
    pub_out = f"{r['denominator']['local_speedup']:.2f} ({r['denominator']['solver']})"
    same = f"{r['same_run_published']['speedup']:.2f}"
    for i, (name, v) in enumerate(r["variants"].items()):
        share = v["share_of_best_log_speedup"]
        cells = [t if i == 0 else "", name, f"{v['speedup']:.2f}", v["valid"], pub_out if i == 0 else "", same if i == 0 else "",
                 f"{100 * share:.0f}%", r["stratum"] if i == 0 else ""]
        print("| " + " | ".join(cells) + " |")
for t, r in rep.items():
    print("replicate", t, {k: (round(v["speedup"], 2), v["valid"], round(v["share_of_best_log_speedup"], 3), round(v["share_vs_same_run_published"], 3)) for k, v in r["variants"].items()},
          "same-run pub", round(r["same_run_published"]["speedup"], 2))
