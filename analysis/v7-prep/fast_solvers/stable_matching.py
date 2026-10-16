# Gemini 3.1 Pro Preview's published Cython stable_matching solver (AlgoTune results/), as one file: the .pyx is
# compiled with pyximport when this module is imported (untimed), then the published solver.py runs unchanged.
import os
import sys
import tempfile

import pyximport

_PYX = r'''# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free

cdef extern from "Python.h":
    ctypedef struct PyObject
    PyObject* PyList_GET_ITEM(PyObject*, Py_ssize_t)
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject* list, Py_ssize_t index, PyObject* item)
    PyObject* PyLong_FromLong(long v)

def solve_cython(list proposer_prefs, list receiver_prefs, int n):
    cdef int i, r, p, rank, cur
    cdef short* recv_rank = <short*>malloc(n * n * sizeof(short))
    cdef int* next_prop = <int*>malloc(n * sizeof(int))
    cdef int* recv_match = <int*>malloc(n * sizeof(int))
    cdef int* free_list = <int*>malloc(n * sizeof(int))
    cdef PyObject** prop_prefs_arr = <PyObject**>malloc(n * sizeof(PyObject*))
    cdef int free_count = n

    cdef PyObject* prop_ptr = <PyObject*>proposer_prefs
    cdef PyObject* recv_ptr = <PyObject*>receiver_prefs

    for i in range(n):
        next_prop[i] = 0
        recv_match[i] = -1
        free_list[i] = n - 1 - i
        prop_prefs_arr[i] = PyList_GET_ITEM(prop_ptr, i)

    cdef PyObject* prefs_obj
    cdef short* r_rank
    for r in range(n):
        prefs_obj = PyList_GET_ITEM(recv_ptr, r)
        r_rank = recv_rank + r * n
        for rank in range(n):
            p = <int><object>PyList_GET_ITEM(prefs_obj, rank)
            r_rank[p] = rank

    while free_count > 0:
        free_count -= 1
        p = free_list[free_count]
        
        r = <int><object>PyList_GET_ITEM(prop_prefs_arr[p], next_prop[p])
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            r_rank = recv_rank + r * n
            if r_rank[p] < r_rank[cur]:
                recv_match[r] = p
                free_list[free_count] = cur
                free_count += 1
            else:
                free_list[free_count] = p
                free_count += 1

    cdef PyObject* matching_obj = PyList_New(n)
    for r in range(n):
        p = recv_match[r]
        PyList_SET_ITEM(matching_obj, p, PyLong_FromLong(r))

    free(recv_rank)
    free(next_prop)
    free(recv_match)
    free(free_list)
    free(prop_prefs_arr)

    return <object>matching_obj'''
_DIR = tempfile.mkdtemp()
with open(os.path.join(_DIR, "fast_solver.pyx"), "w") as _f:
    _f.write(_PYX)
pyximport.install(build_dir=os.path.join(_DIR, "build"), language_level=3)
sys.path.insert(0, _DIR)
from fast_solver import solve_cython  # noqa: E402

from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        if type(prop_raw) is dict:
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = prop_raw
            n = len(proposer_prefs)

        if type(recv_raw) is dict:
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = recv_raw

        matching = solve_cython(proposer_prefs, receiver_prefs, n)
        return {"matching": matching}