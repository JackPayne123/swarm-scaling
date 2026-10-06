"""Replicate Toby Ord's swarm-scaling lambda from the GPT-5.6 Sol launch-post chart data.

Run: uv run --group analysis python analysis/replicate_lambda.py

Definition: N agents achieve what one agent achieves with N**lam times the per-agent budget b
(total tokens T = N*b). With a solo curve score = a + s*ln(b), N agents score
a + s*(lam*ln N + ln b) = a + s*ln T + s*(lam - 1)*ln N.
So regressing score on ln T and ln N (coefficients s and c) gives lam = 1 + c/s.

Equivalent horizontal reading: at equal score the N-agent curve sits ln(T_N/T_1) = (1 - lam)*ln N to
the right of the solo curve in log total tokens (lam = 0.5 and N = 4 means 2x the total tokens).

The chart points are aggregates over tasks. There is no task-level data, so every interval below is a
fit-based interval (OLS t-interval via the delta method, and a residual bootstrap), not a task-level one.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/sources/openai/multiagent_charts.json"
FIGDIR = Path(__file__).resolve().parent / "figures"
ORD = {"BrowseComp": (0.68, 0.63, 0.76), "SEC-Bench Pro": (0.57, 0.52, 0.61), "Terminal-Bench 2.1": (0.48, 0.40, 0.57)}
SOL_CHECK = {"BrowseComp": (0.650, 0.676), "SEC-Bench Pro": (0.565, 0.572), "Terminal-Bench 2.1": (0.481, 0.494)}
SERIES = re.compile(r"GPT-5\.6 Sol · (\d+) agents?")
B_BOOT = 10_000
LEVEL = 0.90


def load():
    """Per benchmark, a dict of arrays (N, idx, tokens, cost, latency, score). Non-Sol series are returned apart."""
    out = {}
    for chart, rows in json.load(open(DATA)).items():
        name = chart.removesuffix(" (Multi-Agent)")
        pts = {}
        for r in rows:
            p = pts.setdefault((r["model"], r["juice_index"]), {"model": r["model"], "idx": r["juice_index"], "score": r["score"]})
            assert p["score"] == r["score"], "score differs across the three x_metric rows of one point"
            p[r["x_metric"]] = r["x_value"]
            p[r["x_metric"] + "_label"] = r["x_label"]
        sol = [p for p in pts.values() if SERIES.fullmatch(p["model"])]
        ref = [p for p in pts.values() if not SERIES.fullmatch(p["model"])]
        sol.sort(key=lambda p: (int(SERIES.fullmatch(p["model"]).group(1)), p["idx"]))
        d = {"N": np.array([int(SERIES.fullmatch(p["model"]).group(1)) for p in sol]),
             "idx": np.array([p["idx"] for p in sol]), "score": np.array([p["score"] for p in sol]),
             "tokens": np.array([p["output_tokens"] for p in sol]), "cost": np.array([p["api_cost_usd"] for p in sol]),
             "latency": np.array([p["latency_s"] for p in sol]), "latency_label": [p["latency_s_label"] for p in sol],
             "ref": ref}
        out[name] = d
    return out


def sub(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items() if k not in ("ref", "latency_label")}


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    cov = (resid @ resid / df) * np.linalg.inv(X.T @ X)
    return beta, cov, df, resid


def delta_ci(g, beta, cov, df):
    eps = 1e-6
    grad = np.array([(g(beta + eps * e) - g(beta - eps * e)) / (2 * eps) for e in np.eye(len(beta))])
    half = stats.t.ppf(0.5 + LEVEL / 2, df) * np.sqrt(grad @ cov @ grad)
    est = g(beta)
    return est - half, est + half


def boot_ci(g, X, beta, resid, df, rng):
    """Residual bootstrap, residuals rescaled by sqrt(n/df). The design (which N and efforts exist) is held fixed."""
    e = rng.choice(resid * np.sqrt(len(resid) / df), size=(len(resid), B_BOOT))
    bs = np.linalg.pinv(X) @ (X @ beta)[:, None] + np.linalg.pinv(X) @ e
    lam = g(bs)
    return tuple(np.quantile(lam, [0.5 - LEVEL / 2, 0.5 + LEVEL / 2]))


def models(d, x):
    """Design matrices and lambda functions. x is the log budget column (ln tokens or ln cost)."""
    N, u = d["N"], np.log(d[x])
    levels = sorted(set(N))
    k = len(levels)
    dummies = np.column_stack([(N == n).astype(float) for n in levels])
    pooled = (np.column_stack([np.ones_like(u), u, np.log(N)]), {"pooled": lambda b: 1 + b[2] / b[1]})
    # common slope, one intercept per N: horizontal offset between N-curves in ln tokens, converted to N**lam
    common = {f"{levels[i]}->{levels[j]}": (lambda b, i=i, j=j: 1 + (b[j] - b[i]) / (b[-1] * np.log(levels[j] / levels[i])))
              for i in range(k) for j in range(i + 1, k)}
    # separate slopes: lam at per-agent budget b0, reading the swarm curve at N*b0 against the reference curve's slope
    sep = {}
    for i in range(k):
        for j in range(i + 1, k):
            for tag, b0 in (("low", d["_b_low"]), ("mid", d["_b_mid"])):
                sep[f"{levels[i]}->{levels[j]} @{tag}"] = (
                    lambda b, i=i, j=j, ub=np.log(b0):
                    ((b[j] + b[k + j] * (np.log(levels[j]) + ub)) - (b[i] + b[k + i] * (np.log(levels[i]) + ub)))
                    / (b[k + i] * np.log(levels[j] / levels[i])))
    return {"pooled": pooled,
            "common": (np.column_stack([dummies, u]), common),
            "separate": (np.column_stack([dummies, dummies * u[:, None]]), sep)}


def transform(score, kind):
    return np.log(score / (1 - score)) if kind == "logit" else score


def fit(d, x="tokens", kind="linear", family="pooled", rng=None):
    """Returns {label: (lam, delta_lo, delta_hi, boot_lo, boot_hi)}, plus the fit's R^2 and residual SD."""
    solo = d["N"] == 1
    d = dict(d, _b_low=d[x][solo].min(), _b_mid=np.sqrt(d[x][solo].min() * d[x][solo].max()))
    X, gs = models(d, x)[family]
    y = transform(d["score"], kind)
    beta, cov, df, resid = ols(X, y)
    rng = rng or np.random.default_rng(0)
    res = {}
    for label, g in gs.items():
        res[label] = (g(beta), *delta_ci(g, beta, cov, df), *boot_ci(g, X, beta, resid, df, rng))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return res, r2, np.sqrt(resid @ resid / df)


def f(r):
    return f"{r[0]:.3f} [{r[1]:.2f}, {r[2]:.2f}]"


def audit(data):
    print("== Data audit")
    for name, d in data.items():
        series = {int(n): int((d["N"] == n).sum()) for n in sorted(set(d["N"]))}
        print(f"{name}: Sol points per N {series}; excluded non-Sol series: "
              f"{sorted({p['model'] for p in d['ref']}) or 'none'} ({len(d['ref'])} points)")


def main_table(data):
    print("\n== (a) Pooled fit, all efforts, x = total output tokens. lam = 1 + c/s. 90% interval: delta method | residual bootstrap")
    rng = np.random.default_rng(0)
    for name, d in data.items():
        for kind in ("linear", "logit"):
            res, r2, sd = fit(d, kind=kind, rng=rng)
            e = res["pooled"]
            print(f"{name:20s} {kind:6s} lam {e[0]:.3f}  delta [{e[1]:.2f}, {e[2]:.2f}]  boot [{e[3]:.2f}, {e[4]:.2f}]  R2 {r2:.3f}  resid SD {sd:.4f}")
        print(f"{'':20s} Ord {ORD[name][0]:.2f} [{ORD[name][1]:.2f}, {ORD[name][2]:.2f}]; Sol check {SOL_CHECK[name][0]:.3f} linear, {SOL_CHECK[name][1]:.3f} logit")
    print("\n== (a) Common slope, one intercept per N (Ord's same-slope lines). lam from the horizontal offset, delta-method 90% interval")
    for name, d in data.items():
        for kind in ("linear", "logit"):
            res, r2, _ = fit(d, kind=kind, family="common", rng=rng)
            print(f"{name:20s} {kind:6s} " + "  ".join(f"{k}: {f(v)}" for k, v in res.items()) + f"  R2 {r2:.3f}")


def sensitivity(data):
    print("\n== (b) Sensitivity, pooled lam (linear | logit), x = tokens unless stated")
    subsets = {"all efforts": lambda d: d["idx"] > 0, "drop max": lambda d: d["idx"] < 5, "drop low": lambda d: d["idx"] > 1,
               "drop low+max": lambda d: (d["idx"] > 1) & (d["idx"] < 5)}
    for name, d in data.items():
        for label, m in subsets.items():
            vals = [fit(sub(d, m(d)), kind=k)[0]["pooled"] for k in ("linear", "logit")]
            print(f"{name:20s} {label:13s} n={int(m(d).sum()):2d}  linear {f(vals[0])} | logit {f(vals[1])}")
        vals = [fit(d, x="cost", kind=k)[0]["pooled"] for k in ("linear", "logit")]
        print(f"{name:20s} {'x = api cost':13s} n={len(d['N']):2d}  linear {f(vals[0])} | logit {f(vals[1])}")
    print("\n== (b) Per-N separate slopes (OLS interaction model, common residual variance)")
    for name, d in data.items():
        for kind in ("linear", "logit"):
            x = np.log(d["tokens"])
            slopes = {n: np.polyfit(x[d["N"] == n], transform(d["score"], kind)[d["N"] == n], 1)[0] for n in sorted(set(d["N"]))}
            ratio = "  ".join(f"slope(N={n})/slope(solo) {s / slopes[1]:.2f}" for n, s in slopes.items() if n > 1)
            res, _, _ = fit(d, kind=kind, family="separate")
            print(f"{name:20s} {kind:6s} {ratio}")
            print(f"{'':27s}" + "  ".join(f"{k}: {f(v)}" for k, v in res.items()))
    print("(@low: per-agent budget b0 = lowest solo token count, Ord's starting point; @mid: geometric mean of the solo token range.)")


def matched_tokens(data):
    print("\n== (d) Matched total tokens: solo curve interpolated at each swarm point inside the solo token range")
    n_pts = n_lead = 0
    for name, d in data.items():
        solo = d["N"] == 1
        T, S = d["tokens"][solo], d["score"][solo]
        for i in np.flatnonzero(~solo):
            if not T.min() <= d["tokens"][i] <= T.max():
                continue
            lin = np.interp(d["tokens"][i], T, S)
            log = np.interp(np.log(d["tokens"][i]), np.log(T), S)
            n_pts += 1
            n_lead += lin > d["score"][i] and log > d["score"][i]
            print(f"{name:20s} N={d['N'][i]:2d} effort {d['idx'][i]} T={d['tokens'][i]:9,.0f} swarm {100 * d['score'][i]:.2f}  "
                  f"solo lin {100 * lin:.2f} log {100 * log:.2f}  deficit lin {100 * (lin - d['score'][i]):.2f} log {100 * (log - d['score'][i]):.2f}")
        beyond = (~solo) & (d["tokens"] > T.max())
        ceil = {n: 100 * (d["score"][d["N"] == n].max() - S.max()) for n in sorted(set(d["N"])) if n > 1}
        print(f"{'':20s} swarm points beyond solo max tokens: {int(beyond.sum())} of {int((~solo).sum())}; max-score gap to solo max (points): "
              + ", ".join(f"N={n} {v:+.2f}" for n, v in ceil.items()))
    print(f"Single agent leads on both interpolations at {n_lead} of {n_pts} overlapping swarm points.")


def latency_units(data):
    print("\n== (e) Latency field. Key is 'latency_s', labels read 'minutes'")
    for name, d in data.items():
        labels_ok = all(l.endswith("minutes") and abs(float(l.split()[0]) - v) < 0.005 for l, v in zip(d["latency_label"], d["latency"]))
        solo = d["N"] == 1
        tps_min = d["tokens"][solo] / (d["latency"][solo] * 60)
        tps_sec = d["tokens"][solo] / d["latency"][solo]
        by_n = {int(n): float(d["latency"][(d["N"] == n) & (d["idx"] == 5)][0].round(2)) for n in sorted(set(d["N"]))}
        print(f"{name:20s} labels all 'minutes' and match values: {labels_ok}. Solo output tokens/s if minutes: {tps_min.min():.0f}-{tps_min.max():.0f}; "
              f"if seconds: {tps_sec.min():.0f}-{tps_sec.max():.0f}. Max-effort latency by N {by_n}")
        cpt = d["cost"][solo] / d["tokens"][solo] * 1000
        print(f"{'':20s} solo api cost per 1k output tokens: ${cpt.min():.3f} to ${cpt.max():.3f}")


def figures(data):
    FIGDIR.mkdir(exist_ok=True)
    ink, grid, surface = "#0b0b0b", "#e4e3de", "#fcfcfb"
    colors = {1: "#2a78d6", 4: "#eb6834", 16: "#1baf7a"}
    for name, d in data.items():
        beta, *_ = ols(*[models(dict(d, _b_low=1, _b_mid=1), "tokens")["common"][0], d["score"]])
        levels = sorted(set(d["N"]))
        lam = fit(d, family="pooled")[0]["pooled"][0]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=surface)
        ax.set_facecolor(surface)
        for i, n in enumerate(levels):
            m = d["N"] == n
            T, S = d["tokens"][m], 100 * d["score"][m]
            xs = np.linspace(np.log(T.min()), np.log(T.max()), 50)
            ax.plot(np.exp(xs), 100 * (beta[i] + beta[-1] * xs), color=colors[n], lw=1.6, zorder=2)
            ax.scatter(T, S, s=48, color=colors[n], edgecolor=surface, linewidth=1.5, zorder=3,
                       label=f"{n} agent{'s' if n > 1 else ''}")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("total output tokens per task, all agents (log scale)", color=ink)
        ax.set_ylabel("score (%)", color=ink)
        ax.set_title(f"{name}: score vs total output tokens", color=ink, fontsize=11, loc="left")
        fig.text(0.01, 0.01, f"lines: same-slope fit, one intercept per agent count (linear score). pooled lam = {lam:.2f}", fontsize=8, color="#52514e")
        ax.grid(color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors="#52514e")
        ax.legend(loc="lower right", frameon=False, labelcolor=ink, fontsize=9)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = FIGDIR / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".png")
        fig.savefig(path, dpi=200, facecolor=surface)
        plt.close(fig)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    data = load()
    audit(data)
    main_table(data)
    sensitivity(data)
    matched_tokens(data)
    latency_units(data)
    print()
    figures(data)
