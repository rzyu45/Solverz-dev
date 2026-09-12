"""Structural sparsity of LoopEqn Jacobian blocks, issue #181.

``compute_loop_jac_sparsity`` keeps every pattern fragment as a pair of
``int64`` arrays and takes their union once with ``np.unique``, where it used
to collect Python tuples in a set. These tests pin its output on one body per
fragment kind against the pattern written out from the definitions: ``int64``
arrays, sorted column-major, without duplicates, and exactly the union of the
fragments.
"""
import numpy as np
import sympy as sp
from scipy.sparse import csc_array

from Solverz import Idx, LoopEqn, Model, Param, Set, Sum, Var, sin
from Solverz.equation.loop_jac import (
    canonicalize_kronecker, compute_loop_jac_sparsity,
)

N = 7


def _band(n, offsets):
    a = np.zeros((n, n))
    for o in offsets:
        a += np.diag(np.full(n - abs(o), 1.0 + abs(o)), o)
    return csc_array(a)


def _stored(a):
    coo = a.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def _pattern(le, var_name, n_diff):
    k = sp.Idx('_sz_loop_dk')
    canonical = canonicalize_kronecker(
        sp.diff(le.body, sp.IndexedBase(var_name)[k]), le.outer_index, k)
    return compute_loop_jac_sparsity(canonical, le.outer_index, k,
                                     le.var_map, le.n_outer, n_diff)


def _assert_pattern(rows, cols, expected):
    assert rows.dtype == np.int64 and cols.dtype == np.int64
    got = list(zip(cols.tolist(), rows.tolist()))
    # Column-major and without duplicates: strictly increasing in (col, row).
    assert got == sorted(set(got))
    assert {(r, c) for c, r in got} == expected


def test_diagonal_and_param_fragments_are_merged():
    """``δ(i, k) * Sum`` gives the diagonal and ``A[i, k]`` the stored
    entries of ``A``; the two overlap on the main diagonal."""
    A = _band(N, [-1, 0, 2])
    m = Model()
    m.x = Var('x', np.linspace(0.1, 0.2, N))
    m.A = Param('A', A, dim=2, sparse=True)
    m.b = Param('b', np.ones(N))
    i, j = Idx('i', N), Idx('j', N)
    m.f = LoopEqn('f', outer_index=i,
                  body=m.x[i] * Sum(m.A[i, j] * sin(m.x[j]), j) - m.b[i], model=m)
    rows, cols = _pattern(m.f, 'x', N)
    _assert_pattern(rows, cols, {(r, r) for r in range(N)} | _stored(A))


def test_rows_selected_by_a_set_from_a_sparse_param():
    """With a ``Set`` outer index the diagonal becomes ``(g, S[g])`` and
    the Param term reads the stored entries of row ``S[g]``."""
    rows_of = np.array([1, 4, 5, 6])
    A = _band(N, [-2, 0, 1])
    m = Model()
    m.x = Var('x', np.linspace(0.1, 0.2, N))
    m.A = Param('A', A, dim=2, sparse=True)
    m.S = Set('S', rows_of)
    g = m.S.idx('g')
    j = Idx('j', N)
    m.f = LoopEqn('f', outer_index=g,
                  body=m.x[g] * Sum(m.A[g, j] * sin(m.x[j]), j), model=m)
    rows, cols = _pattern(m.f, 'x', N)
    stored = _stored(A)
    expected = {(p, int(r)) for p, r in enumerate(rows_of)}
    expected |= {(p, c) for p, r in enumerate(rows_of) for rr, c in stored if rr == r}
    _assert_pattern(rows, cols, expected)


def test_sum_kd_columns_through_a_map_filtered_by_a_sparse_param():
    """``Sum_j A[i, j] * y[cmap[j]]`` differentiates to a ``Sum`` whose
    ``δ(k, cmap[j])`` sends each stored ``A[i, j]`` to column ``cmap[j]``.
    Map values outside ``y`` are dropped, and two stored entries of a row
    that map to one column give one position."""
    n_y = 4
    cmap = np.array([0, 2, 2, 5, 1, 3, -1])
    A = _band(N, [-1, 0, 1])
    m = Model()
    m.y = Var('y', np.linspace(0.1, 0.2, n_y))
    m.A = Param('A', A, dim=2, sparse=True)
    m.cmap = Param('cmap', cmap, dtype=int)
    i, j = Idx('i', N), Idx('j', N)
    m.f = LoopEqn('f', outer_index=i,
                  body=Sum(m.A[i, j] * sin(m.y[m.cmap[j]]), j), model=m)
    rows, cols = _pattern(m.f, 'y', n_y)
    expected = {(r, int(cmap[c])) for r, c in _stored(A) if 0 <= cmap[c] < n_y}
    _assert_pattern(rows, cols, expected)


def test_sum_kd_without_a_sparse_param_reaches_every_row():
    """Without a 2-D Param in the ``Sum`` every row reaches every column
    the map sends into ``y``."""
    n_y = 4
    cmap = np.array([3, 0, 7, 3, -2])
    m = Model()
    m.x = Var('x', np.linspace(0.1, 0.2, N))
    m.y = Var('y', np.linspace(0.3, 0.4, n_y))
    m.w = Param('w', np.linspace(1.0, 2.0, cmap.size))
    m.cmap = Param('cmap', cmap, dtype=int)
    i, j = Idx('i', N), Idx('j', cmap.size)
    m.f = LoopEqn('f', outer_index=i,
                  body=m.x[i] * Sum(m.w[j] * sin(m.y[m.cmap[j]]), j), model=m)
    rows, cols = _pattern(m.f, 'y', n_y)
    _assert_pattern(rows, cols, {(r, c) for r in range(N) for c in (0, 3)})
