"""Rule 2(b) one-change ablations for the provisional M stratum, extending scratchpad ablate.py's
method: Harbor input format (generator output as is), 1 BLAS/OpenMP/numba thread, seeds 3_000_000+i,
per instance one reference warmup + min of 2 timed reference calls, then per variant one warmup +
min of 2 timed calls; speedup = sum ref minima / sum variant minima; validity = Task.is_solution on
the variant's output. The best published solver (rule 2(b) denominator) is timed alongside as a check.
usage: python -I ablate_m.py <scratchpad dir> <task> <n instances> <cap seconds> <out dir>
"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import glob
import importlib.util
import json
import logging
import time
import warnings
from itertools import chain

import numba
import numpy as np
import scipy.linalg as la
import scipy.sparse
import scipy.sparse.csgraph
from numba import njit
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
D, TASK, N, CAP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
SLUG = TASK.replace("_", "-")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_task():
    d = glob.glob(f"{D}/harbor_cache/hub/algotune/{SLUG}/*/")[0]
    if TASK == "randomized_svd":  # upstream A1 port: tol capped at 0.9 (patched copy, cache untouched)
        d = f"{HERE}/randomized_svd_patched/"
    s = importlib.util.spec_from_file_location("hb_" + TASK, d + "tests/evaluator.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.Task(), int(open(d + "tests/test_outputs.py").read().split("PROBLEM_SIZE = ")[1].split()[0])


T, SIZE = load_task()


def denominator():
    """Best local_speedup among published solvers with 0 exceptions, and its solver path."""
    src = f"{HERE}/pub_randomized_svd_patched.json" if TASK == "randomized_svd" else f"{D}/pub_out/{TASK}.json"
    sol = json.load(open(src))["solvers"]
    ok = {k: v for k, v in sol.items() if v["n_exceptions"] == 0 and v["local_speedup"]}
    best = max(ok, key=lambda k: ok[k]["local_speedup"])
    paths = [p for line in open(f"{D}/pub_plan.tsv") if line.split("\t")[0] == TASK for p in line.rstrip("\n").split("\t")[1:]]
    path = [p for p in paths if p.split("/results/")[1].split("/")[0] == best][0]
    return best, ok[best]["local_speedup"], ok[best]["n_invalid"], src, path


# ---------------------------------------------------------------- chebyshev_center
def cheb_linprog(p):  # C1
    from scipy.optimize import linprog
    a = np.array(p["a"]); b = np.array(p["b"]); n = a.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    bounds = np.full((n + 1, 2), [-np.inf, np.inf]); bounds[-1, 0] = 0.0
    res = linprog(c, A_ub=np.hstack([a, np.linalg.norm(a, axis=1)[:, None]]), b_ub=b, bounds=bounds,
                  method="highs-ds", options={"presolve": False})
    assert res.status == 0
    return {"solution": res.x[:n].tolist()}


def cheb_highspy(p):  # C2
    import highspy
    a = np.array(p["a"]); b = np.array(p["b"]); m, n = a.shape
    lp = highspy.HighsLp()
    lp.num_col_ = n + 1; lp.num_row_ = m
    cost = np.zeros(n + 1); cost[-1] = -1.0
    lo = np.full(n + 1, -np.inf); lo[-1] = 0.0
    lp.col_cost_ = cost; lp.col_lower_ = lo; lp.col_upper_ = np.full(n + 1, np.inf)
    lp.row_lower_ = np.full(m, -np.inf); lp.row_upper_ = b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(0, (n + 2) * m, m, dtype=np.int32)
    lp.a_matrix_.index_ = np.tile(np.arange(m, dtype=np.int32), n + 1)
    lp.a_matrix_.value_ = np.concatenate((a.ravel(order="F"), np.linalg.norm(a, axis=1)))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False); h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex"); h.setOptionValue("simplex_strategy", 1)
    h.passModel(lp); h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return {"solution": list(h.getSolution().col_value[:n])}


def cheb_closed(p):  # C3
    a = np.array(p["a"]); b = np.array(p["b"]); m = a.shape[0] // 2
    if a.shape[0] == 2 * m and np.allclose(a[m:], -a[:m]):
        x = np.linalg.lstsq(a[:m], (b[:m] - b[m:]) / 2, rcond=None)[0]
        return {"solution": x.tolist()}
    return T.solve(p)


# ---------------------------------------------------------------- generalized_eigenvectors_complex
def _gec_ref_post(A, eigenvalues, eigenvectors):  # the reference's post-processing, verbatim
    n = A.shape[0]
    for i in range(n):
        v = eigenvectors[:, i]
        norm = np.linalg.norm(v)
        if norm > 1e-15:
            eigenvectors[:, i] = v / norm
    pairs = list(zip(eigenvalues, [eigenvectors[:, i] for i in range(n)]))
    pairs.sort(key=lambda pair: (-pair[0].real, -pair[0].imag))
    sorted_eigenvalues, sorted_eigenvectors = zip(*pairs)
    return (list(sorted_eigenvalues), [list(vec) for vec in sorted_eigenvectors])


def gec_reduction(problem):  # G1
    A, B = problem
    scale_B = np.sqrt(np.linalg.norm(B))
    B_scaled = B / scale_B  # noqa: F841 (kept: only the eig line changes)
    A_scaled = A / scale_B  # noqa: F841
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(B, A))
    return _gec_ref_post(A, eigenvalues, eigenvectors)


def gec_vecpost(problem):  # G2
    A, B = problem
    scale_B = np.sqrt(np.linalg.norm(B))
    w, v = la.eig(A / scale_B, B / scale_B)
    norms = np.linalg.norm(v, axis=0)
    norms = np.where(norms > 1e-300, norms, 1.0)
    v = v / norms
    idx = np.lexsort((-w.imag, -w.real))
    return (w[idx].tolist(), v[:, idx].T.tolist())


# ---------------------------------------------------------------- ode_lorenz96_nonchaotic
@njit
def _l96_rhs(t, x, F):
    N = len(x)
    dxdt = np.empty(N)
    dxdt[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0] + F
    dxdt[1] = (x[2] - x[N - 1]) * x[0] - x[1] + F
    dxdt[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1] + F
    for i in range(2, N - 1):
        dxdt[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt


def ode_numba_rhs(problem):  # O1
    y0 = np.array(problem["y0"])
    t0, t1 = float(problem["t0"]), float(problem["t1"])
    F = float(problem["F"])
    sol = solve_ivp(_l96_rhs, [t0, t1], y0, method="RK45", rtol=1e-8, atol=1e-8, t_eval=None, dense_output=False, args=(F,))
    if sol.success:
        return sol.y[:, -1].tolist()
    raise RuntimeError(sol.message)


# ---------------------------------------------------------------- graph_global_efficiency
@njit
def _eff_kernel(indices, indptr, n):
    # Opus 4.6's kernel with prange -> range: under numba 0.68 its parallel=True build raises
    # "AssertionError: unexpected cycle in lookup()" (the same error that stopped Opus 4.6's solver
    # loading in runpub); at NUMBA_NUM_THREADS=1 prange would run these iterations serially anyway.
    total = 0.0
    for source in range(n):
        visited = np.zeros(n, dtype=np.int8)
        queue = np.empty(n, dtype=np.int32)
        visited[source] = 1
        head = 0
        tail = 1
        queue[0] = source
        local_sum = 0.0
        d = 0
        while head < tail:
            d += 1
            inv_d = 1.0 / d
            level_end = tail
            while head < level_end:
                u = queue[head]
                head += 1
                for idx in range(indptr[u], indptr[u + 1]):
                    v = indices[idx]
                    if visited[v] == 0:
                        visited[v] = 1
                        local_sum += inv_d
                        queue[tail] = v
                        tail += 1
        total += local_sum
    return total / (n * (n - 1))


def eff_numba(problem):  # E1
    adj_list = problem["adjacency_list"]
    n = len(adj_list)
    if n <= 1:
        return {"global_efficiency": 0.0}
    indptr = np.empty(n + 1, dtype=np.int32); indptr[0] = 0
    np.cumsum(np.array([len(x) for x in adj_list], dtype=np.int32), out=indptr[1:])
    if indptr[n] == 0:
        return {"global_efficiency": 0.0}
    indices = np.fromiter(chain.from_iterable(adj_list), dtype=np.int32, count=indptr[n])
    return {"global_efficiency": float(_eff_kernel(indices, indptr, n))}


# ---------------------------------------------------------------- pagerank
def _pr_csr(adj_list):
    n = len(adj_list)
    outdeg = np.fromiter((len(x) for x in adj_list), dtype=np.int64, count=n)
    indptr = np.empty(n + 1, dtype=np.int64); indptr[0] = 0
    np.cumsum(outdeg, out=indptr[1:])
    indices = np.fromiter(chain.from_iterable(adj_list), dtype=np.int64, count=indptr[n])
    return n, outdeg, indptr, indices


@njit
def _pr_push(indptr, indices, outdeg, alpha, max_iter, tol):
    n = outdeg.shape[0]
    x = np.full(n, 1.0 / n)
    for it in range(max_iter):
        dsum = 0.0
        for i in range(n):
            if outdeg[i] == 0:
                dsum += x[i]
        xnew = np.full(n, alpha * dsum / n + (1.0 - alpha) / n)
        for i in range(n):
            if outdeg[i] != 0:
                c = alpha * x[i] / outdeg[i]
                for q in range(indptr[i], indptr[i + 1]):
                    xnew[indices[q]] += c
        err = 0.0
        for i in range(n):
            err += abs(xnew[i] - x[i])
        x = xnew
        if err < n * tol:
            return x, it + 1
    return x, -1


def pr_numba(problem):  # P1
    adj_list = problem["adjacency_list"]
    if len(adj_list) <= 1:
        return T.solve(problem)
    n, outdeg, indptr, indices = _pr_csr(adj_list)
    x, it = _pr_push(indptr, indices, outdeg, T.alpha, T.max_iter, T.tol)
    return {"pagerank_scores": x.tolist() if it > 0 else [0.0] * n}


def pr_numpy(problem):  # P2
    adj_list = problem["adjacency_list"]
    if len(adj_list) <= 1:
        return T.solve(problem)
    N, outdeg, indptr, indices = _pr_csr(adj_list)
    A = scipy.sparse.csr_array((np.repeat(1.0 / np.maximum(outdeg, 1), outdeg), indices, indptr), shape=(N, N))
    alpha, tol = T.alpha, T.tol
    dangling_nodes = outdeg == 0
    x = np.repeat(1.0 / N, N)
    p = np.repeat(1.0 / N, N)
    for _ in range(T.max_iter):  # networkx 3.7 _pagerank_scipy loop
        xlast = x
        x = alpha * (x @ A + np.sum(x[dangling_nodes]) * p) + (1 - alpha) * p
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return {"pagerank_scores": x.tolist()}
    return {"pagerank_scores": [0.0] * N}


# ---------------------------------------------------------------- minimum_spanning_tree
@njit
def _kruskal(u, v, order, n):
    parent = np.arange(n)
    chosen = np.empty(max(n - 1, 0), dtype=np.int64)
    cnt = 0
    for idx in order:
        a = u[idx]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        b = v[idx]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[a] = b
            chosen[cnt] = idx
            cnt += 1
            if cnt == n - 1:
                break
    return chosen[:cnt]


def mst_numba(problem):  # T1
    a = np.asarray(problem["edges"])
    n = problem["num_nodes"]
    order = np.argsort(a[:, 2], kind="stable")
    u = a[:, 0].astype(np.int64); v = a[:, 1].astype(np.int64)
    ch = _kruskal(u, v, order, n)
    ch = ch[np.lexsort((v[ch], u[ch]))]
    return {"mst_edges": [list(e) for e in zip(u[ch].tolist(), v[ch].tolist(), a[ch, 2].tolist())]}


def mst_scipy(problem):  # T2, Opus 4.5's solver body
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    num_nodes = problem["num_nodes"]; edges = problem["edges"]
    if num_nodes == 0 or len(edges) == 0:
        return {"mst_edges": []}
    edges_arr = np.asarray(edges)
    row = edges_arr[:, 0].astype(np.int32); col = edges_arr[:, 1].astype(np.int32); data = edges_arr[:, 2]
    graph = csr_matrix((np.concatenate((data, data)), (np.concatenate((row, col)), np.concatenate((col, row)))), shape=(num_nodes, num_nodes))
    mst_coo = minimum_spanning_tree(graph).tocoo()
    u = mst_coo.row; v = mst_coo.col; w = mst_coo.data
    if len(w) == 0:
        return {"mst_edges": []}
    swap = u > v
    u_new = np.where(swap, v, u); v_new = np.where(swap, u, v)
    s = np.lexsort((v_new, u_new))
    ul, vl, wl = u_new[s].tolist(), v_new[s].tolist(), w[s].tolist()
    return {"mst_edges": [[ul[i], vl[i], wl[i]] for i in range(len(wl))]}


def mst_python(problem):  # T3, Opus 4.1's solver body
    num_nodes = problem["num_nodes"]; edges = problem["edges"]
    sorted_edges = sorted(edges, key=lambda x: x[2])
    parent = list(range(num_nodes)); rank = [0] * num_nodes

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
        return True

    mst_edges = []
    for u, v, w in sorted_edges:
        if union(u, v):
            if u > v:
                u, v = v, u
            mst_edges.append([u, v, w])
            if len(mst_edges) == num_nodes - 1:
                break
    mst_edges.sort(key=lambda x: (x[0], x[1]))
    return {"mst_edges": mst_edges}


# ---------------------------------------------------------------- shortest_path_dijkstra
def sp_tolist(problem):  # D1
    g = scipy.sparse.csr_matrix((problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"])
    dist = scipy.sparse.csgraph.shortest_path(csgraph=g, method=T.method, directed=T.directed)
    return {"distance_matrix": dist.tolist()}


# ---------------------------------------------------------------- l0_pruning
def l0_argpartition(problem):  # L0a
    v = np.array(problem.get("v")); k = problem.get("k")
    v = v.flatten()
    pruned = np.zeros_like(v)
    n = v.size
    remaining_indx = np.argpartition(np.abs(v), n - k)[n - k:]
    pruned[remaining_indx] = v[remaining_indx]
    return {"solution": pruned.tolist()}


def l0_ndarray(problem):  # L0b
    v = np.array(problem.get("v")); k = problem.get("k")
    v = v.flatten()
    pruned = np.zeros_like(v)
    indx = np.argsort(np.abs(v), kind="mergesort")
    remaining_indx = indx[-k:]
    pruned[remaining_indx] = v[remaining_indx]
    return {"solution": pruned}


# ---------------------------------------------------------------- l1_pruning
def _l1_ref_sub(vn, z):  # reference subproblem, verbatim
    mu = np.sort(vn, kind="mergesort")[::-1]
    cumsum = 0
    theta = 0
    for j in range(len(mu)):
        cumsum += mu[j]
        if mu[j] < 1 / (j + 1) * (cumsum - z):
            theta = 1 / (j + 1) * (cumsum - z)
            break
    return np.maximum(vn - theta, 0)


def _l1_vec_sub(vn, z):  # L1b: the loop vectorised, same first-break rule and arithmetic
    mu = np.sort(vn, kind="mergesort")[::-1]
    thr = 1 / np.arange(1, len(mu) + 1) * (np.cumsum(mu) - z)
    mask = mu < thr
    theta = thr[np.argmax(mask)] if mask.any() else 0
    return np.maximum(vn - theta, 0)


@njit(fastmath=True, nogil=True)
def _michelot_theta(u, k):  # o4-mini's Michelot elimination, theta only
    n = u.shape[0]
    total = 0.0
    for i in range(n):
        total += u[i]
    if total <= k:
        return 0.0
    rho = total - k
    m = n
    while True:
        lam = rho / m
        s = 0.0
        cnt = 0
        for i in range(n):
            if u[i] > lam:
                s += u[i]
                cnt += 1
        if cnt == m:
            return lam
        rho = s - k
        m = cnt


def _l1_mich_sub(vn, z):  # L1c
    return np.maximum(vn - _michelot_theta(vn, float(z)), 0)


def _l1(problem, sub, as_list=True):
    v = np.array(problem.get("v")); k = problem.get("k")
    v = v.flatten()
    u = np.abs(v)
    b = sub(u, k)
    new_v = b * np.sign(v)
    remaining_indx = new_v != 0
    pruned = np.zeros_like(v)
    pruned[remaining_indx] = new_v[remaining_indx]
    return {"solution": pruned.tolist() if as_list else pruned}


# ---------------------------------------------------------------- pde_heat1d
def heat_prealloc(problem):  # H1
    y0 = np.array(problem["y0"])
    t0, t1 = problem["t0"], problem["t1"]
    params = problem["params"]
    u_padded = np.zeros(y0.size + 2)

    def heat_equation(t, u):
        alpha = params["alpha"]
        dx = params["dx"]
        u_padded[1:-1] = u
        u_xx = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx**2)
        return alpha * u_xx

    sol = solve_ivp(heat_equation, [t0, t1], y0, method="RK45", rtol=1e-6, atol=1e-6, t_eval=None, dense_output=False)
    if sol.success:
        return sol.y[:, -1].tolist()
    raise RuntimeError(sol.message)


@njit
def _heat_rhs(u, alpha_dx2):  # Opus 4.5's form
    n = len(u)
    du = np.empty(n)
    du[0] = alpha_dx2 * (u[1] - 2 * u[0])
    for i in range(1, n - 1):
        du[i] = alpha_dx2 * (u[i + 1] - 2 * u[i] + u[i - 1])
    du[n - 1] = alpha_dx2 * (-2 * u[n - 1] + u[n - 2])
    return du


def heat_numba(problem):  # H2
    y0 = np.array(problem["y0"])
    t0, t1 = problem["t0"], problem["t1"]
    params = problem["params"]
    c = float(params["alpha"]) / (float(params["dx"]) ** 2)
    sol = solve_ivp(lambda t, u: _heat_rhs(u, c), [t0, t1], y0, method="RK45", rtol=1e-6, atol=1e-6, t_eval=None, dense_output=False)
    if sol.success:
        return sol.y[:, -1].tolist()
    raise RuntimeError(sol.message)


# ---------------------------------------------------------------- randomized_svd
def rsvd_niter0(problem):  # R1
    from sklearn.utils.extmath import randomized_svd
    U, s, Vt = randomized_svd(problem["matrix"], n_components=problem["n_components"], n_iter=0, random_state=42)
    return {"U": U, "S": s, "V": Vt.T}


PLAN = {
    "chebyshev_center": {"C1 linprog highs-ds, presolve off": cheb_linprog, "C2 highspy passModel, dual simplex, presolve off": cheb_highspy,
                         "C3 slab closed form (lstsq) with LP fallback": cheb_closed},
    "generalized_eigenvectors_complex": {"G1 QZ to eig(solve(B, A)), reference post-processing": gec_reduction,
                                         "G2 QZ kept, GPT-5.2 vectorised post-processing": gec_vecpost},
    "ode_lorenz96_nonchaotic": {"O1 numba RHS in solve_ivp": ode_numba_rhs},
    "graph_global_efficiency": {"E1 fromiter CSR + numba BFS summing 1/d (Opus 4.6 kernel)": eff_numba},
    "pagerank": {"P1 fromiter CSR + numba power iteration": pr_numba, "P2 same CSR, scipy matvec + numpy loop": pr_numpy},
    "minimum_spanning_tree": {"T1 asarray + stable argsort + numba union-find": mst_numba, "T2 scipy minimum_spanning_tree": mst_scipy,
                              "T3 plain Python Kruskal": mst_python},
    "shortest_path_dijkstra": {"D1 output via dist_matrix.tolist()": sp_tolist},
    "l0_pruning": {"L0a argsort to argpartition": l0_argpartition, "L0b ndarray output": l0_ndarray},
    "l1_pruning": {"L1a ndarray output": lambda p: _l1(p, _l1_ref_sub, as_list=False), "L1b loop vectorised": lambda p: _l1(p, _l1_vec_sub),
                   "L1c sort + loop to numba Michelot": lambda p: _l1(p, _l1_mich_sub)},
    "pde_heat1d": {"H1 np.pad to preallocated buffer": heat_prealloc, "H2 numba RHS in solve_ivp": heat_numba},
    "randomized_svd": {"R1 n_iter=0": rsvd_niter0},
}


def tmin(f, p, k=2):
    best, cpu, out = float("inf"), 0.0, None
    for _ in range(k):
        c = time.process_time(); a = time.perf_counter()
        out = f(p)
        w = time.perf_counter() - a
        if w < best:
            best, cpu = w, time.process_time() - c
    return best, cpu, out


best_label, denom, denom_invalid, denom_src, pub_path = denominator()
spec = importlib.util.spec_from_file_location("pub_solver", pub_path)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)
pub_solver = pm.Solver()
arms = dict(PLAN[TASK])
PUB = f"published {best_label}"
arms[PUB] = pub_solver.solve

st = {k: {"t": [], "cpu": [], "t_ref": [], "valid": 0, "n": 0, "exceptions": []} for k in arms}
ref_t, ref_cpu = [], []
t0 = time.time()
n_done = 0
for i in range(N):
    if time.time() - t0 > CAP:
        break
    p = T.generate_problem(n=SIZE, random_seed=3_000_000 + i)
    T.solve(p)
    tr, cr, _ = tmin(T.solve, p)
    ref_t.append(tr); ref_cpu.append(cr)
    for name, f in arms.items():
        s = st[name]
        s["n"] += 1
        try:
            f(p)
            tv, cv, out = tmin(f, p)
        except Exception as e:
            s["exceptions"].append([3_000_000 + i, f"{type(e).__name__}: {e}"[:200]])
            continue
        s["t"].append(tv); s["cpu"].append(cv); s["t_ref"].append(tr)
        try:
            s["valid"] += bool(T.is_solution(p, out))
        except Exception as e:
            s["exceptions"].append([3_000_000 + i, f"is_solution {type(e).__name__}: {e}"[:200]])
    n_done += 1
    print(f"{TASK} {i + 1}/{N} {time.time() - t0:.0f}s ref {tr * 1e3:.1f}ms " +
          " ".join(f"[{k.split()[0]} {tr / st[k]['t'][-1] if st[k]['t'] else float('nan'):.2f}x]" for k in arms), flush=True)

res = {"task": TASK, "problem_size": SIZE, "n_requested": N, "n_done": n_done, "elapsed_s": time.time() - t0, "cap_s": CAP,
       "ref_total_s": sum(ref_t), "ref_cpu_over_wall": sum(ref_cpu) / sum(ref_t),
       "denominator": {"solver": best_label, "local_speedup": denom, "n_invalid_of_400": denom_invalid, "source": denom_src, "path": pub_path},
       "arms": {}}
for name, s in st.items():
    sp = sum(s["t_ref"]) / sum(s["t"]) if s["t"] else None
    res["arms"][name] = {
        "speedup": sp, "valid": f"{s['valid']}/{s['n']}", "n_valid": s["valid"], "n": s["n"], "n_exceptions": len(s["exceptions"]),
        "exceptions": s["exceptions"][:5], "cpu_over_wall": sum(s["cpu"]) / sum(s["t"]) if s["t"] else None,
        "share_of_best_log_speedup": (np.log(sp) / np.log(denom)) if sp else None, "is_published": name == PUB,
        "per_instance_ref_s": s["t_ref"], "per_instance_s": s["t"],
    }
json.dump(res, open(f"{OUT}/{TASK}.json", "w"), indent=1)
print("DONE", TASK, json.dumps({k: (round(v["speedup"], 3) if v["speedup"] else None, v["valid"]) for k, v in res["arms"].items()}), flush=True)
