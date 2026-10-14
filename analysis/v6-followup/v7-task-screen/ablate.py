"""One-change ablations of the reference, in Harbor input format: does a single change reproduce
most of the best published solver's local speedup? Local, 1 thread, Mac. Speedup = sum ref / sum variant.
usage: python -I ablate.py <scratchpad dir> <n instances>"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import glob
import importlib.util
import json
import logging
import time
import warnings

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
D, N = sys.argv[1], int(sys.argv[2])


def hb(slug):
    d = glob.glob(f"{D}/harbor_cache/hub/algotune/{slug}/*/")[0]
    s = importlib.util.spec_from_file_location("hb_" + slug.replace("-", "_"), d + "tests/evaluator.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.Task(), int(open(d + "tests/test_outputs.py").read().split("PROBLEM_SIZE = ")[1].split()[0])


def eig_sort(ev):
    return sorted(ev, key=lambda z: (-z.real, -z.imag))


def geneig_f64(p):
    A, B = p
    A = np.asarray(A); B = np.asarray(B)
    return eig_sort(np.linalg.eigvals(np.linalg.solve(B, A)))


def geneig_f32(p):
    A, B = p
    A = np.asarray(A, dtype=np.float32); B = np.asarray(B, dtype=np.float32)
    return eig_sort(np.linalg.eigvals(np.linalg.solve(B, A)).astype(np.complex128))


def geneig_qz_novec(p):
    A, B = p
    s = np.sqrt(np.linalg.norm(B))
    return eig_sort(la.eigvals(np.asarray(A) / s, np.asarray(B) / s))


def psd_eigh(p):
    A = np.array(p["A"])
    w, V = np.linalg.eigh(A)
    return {"X": (V * np.maximum(w, 0)) @ V.T}


def eigvals_only(p):
    return sorted(np.linalg.eigvalsh(p), reverse=True)


def pca_eigh(p):
    X = np.array(p["X"], dtype=float)
    X = X - X.mean(axis=0)
    k = p["n_components"]
    w, V = np.linalg.eigh(X.T @ X)
    return V[:, ::-1][:, :k].T


def mwa_dense(p):
    import scipy.sparse
    mat = scipy.sparse.csr_matrix((p["data"], p["indices"], p["indptr"]), shape=p["shape"])
    dense = np.full(mat.shape, np.inf)
    coo = mat.tocoo()
    dense[coo.row, coo.col] = coo.data
    r, c = linear_sum_assignment(dense)
    return {"assignment": {"row_ind": r.tolist(), "col_ind": c.tolist()}}


def sparse_tol(tol):
    def f(p):
        mat = p["matrix"].asformat("csr"); k = int(p["k"]); n = mat.shape[0]
        vals = eigsh(mat, k=k, which="SM", return_eigenvectors=False, maxiter=n * 200, ncv=min(n - 1, max(2 * k + 1, 20)), tol=tol)
        return [float(v) for v in np.sort(np.real(vals))]
    return f


def lasso_f32(p):
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
    clf.fit(np.asarray(p["X"], dtype=np.float32), np.asarray(p["y"], dtype=np.float32))
    return clf.coef_.astype(float).tolist()


PLAN = {
    "generalized-eigenvalues-complex": {"QZ, no eigenvectors": geneig_qz_novec, "B^-1 A + eigvals, float64": geneig_f64, "B^-1 A + eigvals, float32": geneig_f32},
    "psd-cone-projection": {"eig -> eigh": psd_eigh},
    "eigenvalues-real": {"eigh -> eigvalsh": eigvals_only},
    "pca": {"sklearn PCA -> eigh of X^T X": pca_eigh},
    "min-weight-assignment": {"dense linear_sum_assignment": mwa_dense},
    "sparse-lowest-eigenvalues-posdef": {"eigsh tol 1e-4": sparse_tol(1e-4), "eigsh tol 5e-4": sparse_tol(5e-4)},
    "lasso": {"sklearn Lasso on float32": lasso_f32},
}


def tmin(f, p, k=2):
    best = float("inf")
    for _ in range(k):
        a = time.perf_counter(); out = f(p); best = min(best, time.perf_counter() - a)
    return best, out


res = {}
for slug, variants in PLAN.items():
    T, size = hb(slug)
    tot_ref = 0.0
    tot = {k: 0.0 for k in variants}
    valid = {k: 0 for k in variants}
    for i in range(N):
        p = T.generate_problem(n=size, random_seed=3_000_000 + i)
        T.solve(p)
        tr, _ = tmin(T.solve, p)
        tot_ref += tr
        for name, f in variants.items():
            f(p)
            tv, out = tmin(f, p)
            tot[name] += tv
            try:
                valid[name] += bool(T.is_solution(p, out))
            except Exception:
                pass
    res[slug] = {name: {"speedup": tot_ref / tot[name], "valid": f"{valid[name]}/{N}"} for name in variants}
    print(slug, json.dumps(res[slug]), flush=True)
json.dump(res, open(f"{D}/ablate.json", "w"), indent=1)
