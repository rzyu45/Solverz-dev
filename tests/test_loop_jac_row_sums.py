"""A sparse LoopEqn Jacobian kernel evaluates a ``Sum`` that only its diagonal
uses once per row, and reads a sparse Param at a position computed with the
pattern instead of searching the row for it (issue #179).

The model is the reproduction of the issue,
``f_i = x_i * Sum_j A[i, j] sin(x_j) - b_i``. Its derivative is
``delta(i, k) * Sum_j A[i, j] sin(x_j) + x_i * A[i, k] * cos(x_k)``, so the
``Sum`` is needed only where ``k == i``. Row 3 of ``A`` stores no diagonal
entry, so the kernel also meets a nonzero at which ``A[i, k]`` is not stored.
"""
import importlib
import re
import sys
import uuid

import numpy as np
import sympy as sp
from numba import njit
from scipy.sparse import csc_array, csr_array, diags

from Solverz import LoopEqn, Model, Param, Set, Sum, Var, module_printer, sin
from Solverz.equation.loop_jac import csr_point_positions

N = 8

# The kernel that Solverz 0.11.1 generated for this model, with the helper that
# searches row i of A for column k. Where the delta holds, the kernel of the fix
# performs the same operations in the same order, so the values must agree bit
# for bit.
KERNEL_0111 = '''
def _sz_csr_A_point(row, col):
    for _sz_pk in range(_sz_csr_A_indptr[row], _sz_csr_A_indptr[row + 1]):
        if _sz_csr_A_indices[_sz_pk] == col:
            return _sz_csr_A_data[_sz_pk]
    return 0.0

def kernel_0111(x, _sz_row_arr, _sz_col_arr):
    data = np.empty(_sz_row_arr.size)
    for _sz_idx in range(_sz_row_arr.size):
        i = _sz_row_arr[_sz_idx]
        _sz_loop_dk = _sz_col_arr[_sz_idx]
        _sz_loop_acc_0 = 0.0
        for _sz_kk_0 in range(_sz_csr_A_indptr[i], _sz_csr_A_indptr[i + 1]):
            j = _sz_csr_A_indices[_sz_kk_0]
            _sz_loop_acc_0 += (_sz_csr_A_data[_sz_kk_0] * np.sin(x[j]))
        data[_sz_idx] = (((_sz_loop_acc_0) if _sz_loop_dk == i else 0.0) + (x[i] * _sz_csr_A_point(i, _sz_loop_dk) * np.cos(x[_sz_loop_dk])))
    return data
'''


def _matrix():
    A = diags([np.full(N - 1, 0.7), np.linspace(1.0, 2.0, N), np.full(N - 1, 0.4)],
              [-1, 0, 1], shape=(N, N), format='lil')
    A[3, 3] = 0.0
    A = csc_array(A)
    A.eliminate_zeros()
    assert A[3, 3] == 0 and A.nnz == 3 * N - 3
    return A


def _model(A):
    m = Model()
    m.x = Var('x', np.linspace(0.1, 0.8, N))
    m.A = Param('A', A, dim=2, sparse=True)
    m.b = Param('b', np.ones(N))
    m.S = Set('S', N)
    i, j = m.S.idx('i'), m.S.idx('j')
    m.f = LoopEqn('f', outer_index=i, body=m.x[i] * Sum(m.A[i, j] * sin(m.x[j]), j) - m.b[i], model=m)
    return m.create_instance()


def _reference(A, jit):
    csr = A.tocsr()
    ns = {'np': np,
          '_sz_csr_A_data': np.ascontiguousarray(csr.data, dtype=float),
          '_sz_csr_A_indices': np.ascontiguousarray(csr.indices, dtype=np.int64),
          '_sz_csr_A_indptr': np.ascontiguousarray(csr.indptr, dtype=np.int64)}
    exec(KERNEL_0111, ns)
    if not jit:
        return ns['kernel_0111']
    ns['_sz_csr_A_point'] = njit(ns['_sz_csr_A_point'])
    return njit(ns['kernel_0111'])


def _pattern(J):
    """The rows and columns of the stored entries of ``J`` in storage order."""
    J = csc_array(J)
    cols = np.repeat(np.arange(J.shape[1], dtype=np.int64), np.diff(J.indptr))
    return J.indices.astype(np.int64), cols


def _points(y):
    rng = np.random.default_rng(179)
    x0 = np.array(y['x'], dtype=float)
    return [x0, x0 + rng.uniform(-0.5, 0.5, N)]


def _assert_same_bits(a, b):
    np.testing.assert_array_equal(np.ascontiguousarray(a, dtype=float).view(np.int64),
                                  np.ascontiguousarray(b, dtype=float).view(np.int64))


def _check(mdl, y, reference):
    for x in _points(y):
        y['x'] = x
        J = csc_array(mdl.J(y, mdl.p))
        rows, cols = _pattern(J)
        # the diagonal is stored at row 3 too, where A has no entry
        assert {(r, r) for r in range(N)} <= set(zip(rows.tolist(), cols.tolist()))
        _assert_same_bits(J.data, reference(x, rows, cols))


def _kernel_diff(spf):
    (ed,) = [ed for eqn in spf.EQNs.values() for ed in eqn.derivatives.values()
             if hasattr(ed, 'kernel_source')]
    return ed


def test_inline_kernel_runs_the_sum_only_on_the_diagonal():
    A = _matrix()
    spf, y0 = _model(A)
    ed = _kernel_diff(spf)
    src = ed.kernel_source
    lines = src.splitlines()
    # at the level of the loop over the nonzeros, the row loop of the Sum
    # appears only under the delta condition
    body = [ln[8:] for ln in lines if ln.startswith(' ' * 8) and not ln.startswith(' ' * 9)]
    assert not [ln for ln in body if ln.startswith('for ')]
    k_if = next(n for n, ln in enumerate(lines) if ln.strip().startswith('if ') and '==' in ln)
    k_for = next(n for n, ln in enumerate(lines) if ln.strip().startswith('for _sz_kk_'))
    assert k_if < k_for
    # A[i, k] is read at a precomputed position, with -1 for row 3
    assert ed.helper_sources == []
    assert '_sz_csr_A_point' not in src
    assert re.search(r'\(_sz_csr_A_data\[_sz_pos_0\[_sz_idx\]\] if _sz_pos_0\[_sz_idx\] >= 0 else 0\.0\)', src)
    (pos,) = ed.point_pos_arrays
    missing = pos < 0
    assert missing.sum() == 1
    assert (ed._sparsity_row[missing], ed._sparsity_col[missing]) == (3, 3)

    # the kernel that FormJac evaluates, in plain Python
    reference = _reference(A, jit=False)
    for x in _points(y0):
        _assert_same_bits(ed.NUM_EQN(x), reference(x, ed._sparsity_row, ed._sparsity_col))


def test_module_kernel_equals_0111_bit_for_bit(tmp_path):
    A = _matrix()
    spf, y0 = _model(A)
    name = f'sz_test_rowsum_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=str(tmp_path), jit=True).render()
    src = (tmp_path / name / 'num_func.py').read_text()
    assert '_sz_csr_A_point' not in src
    assert '_sz_loop_jac_pos_0_0 = setting["_sz_loop_jac_pos_0_0"]' in src
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(name)
        _check(mod.mdl, mod.y, _reference(A, jit=True))
    finally:
        sys.path.remove(str(tmp_path))
        for k in [k for k in sys.modules if k.startswith(name)]:
            del sys.modules[k]


def test_entry_that_depends_on_a_sum_dummy_keeps_the_search():
    """The generator computes a position only for an entry whose row and
    column follow from the nonzero. In ``x[i] A[i, k] + Sum_j A[i, j] B[j, k]
    x[j]`` the walker of the ``Sum`` is ``A``, and ``B[j, k]`` depends on the
    dummy ``j``, so it is still read through the search helper, in the same
    kernel as ``A[i, k]``, which is read at its position. Without the pattern
    the generator searches every entry, as before issue #179."""
    from Solverz.equation.loop_jac import build_loop_jac_kernel_source

    rng = np.random.default_rng(3)
    dense_a = np.where(rng.random((N, N)) < 0.4, rng.uniform(0.5, 1.5, (N, N)), 0.0) + np.eye(N)
    dense_a[2, 5] = 0.0
    dense_b = np.where(rng.random((N, N)) < 0.4, rng.uniform(0.5, 1.5, (N, N)), 0.0)
    x_var = Var('x', np.linspace(0.2, 0.9, N))
    params = {'A': Param('A', csc_array(dense_a), dim=2, sparse=True),
              'B': Param('B', csc_array(dense_b), dim=2, sparse=True)}
    var_map = {'x': x_var, **params}
    i, j, k = sp.Idx('i'), sp.Idx('j'), sp.Idx('k')
    x, A, B = sp.IndexedBase('x'), sp.IndexedBase('A'), sp.IndexedBase('B')
    canonical = x[i] * A[i, k] + sp.Sum(A[i, j] * B[j, k] * x[j], (j, 0, N - 1))
    rows, cols = (a.ravel().astype(np.int64) for a in np.meshgrid(np.arange(N), np.arange(N), indexing='ij'))
    csr_arrays = {}
    for nm, p in params.items():
        csr = p.v.tocsr()
        csr_arrays[f'_sz_csr_{nm}_data'] = np.ascontiguousarray(csr.data, dtype=float)
        csr_arrays[f'_sz_csr_{nm}_indices'] = np.ascontiguousarray(csr.indices, dtype=np.int64)
        csr_arrays[f'_sz_csr_{nm}_indptr'] = np.ascontiguousarray(csr.indptr, dtype=np.int64)

    src, helpers, positions = build_loop_jac_kernel_source(
        'kernel', canonical, i, k, rows.size, ['x'], var_map,
        row_arr=rows, col_arr=cols, csr_arrays=csr_arrays)
    assert len(positions) == 1 and (positions[0] < 0).any()
    assert '_sz_csr_A_point' not in src
    assert '_sz_csr_B_point(j, k)' in src
    assert [h.split('(')[0] for h in helpers] == ['def _sz_csr_B_point']
    ns = {'np': np, **csr_arrays}
    for h in helpers:
        exec(h, ns)
    exec(src, ns)
    xv = np.array(x_var.value, dtype=float)
    expected = xv[:, None] * dense_a + dense_a @ (xv[:, None] * dense_b)
    np.testing.assert_allclose(ns['kernel'](xv, rows, cols, *positions), expected.ravel(), rtol=1e-14, atol=1e-14)

    old_src, old_helpers, old_positions = build_loop_jac_kernel_source(
        'kernel', canonical, i, k, rows.size, ['x'], var_map)
    assert old_positions == []
    assert '_sz_csr_A_point(i, k)' in old_src and '_sz_csr_B_point(j, k)' in old_src
    assert len(old_helpers) == 2


def test_csr_point_positions_matches_the_search():
    """The first stored entry, as the search helper returns it, including a
    CSR matrix with a duplicate entry, and None out of range."""
    indptr = np.array([0, 3, 3, 5])
    indices = np.array([2, 0, 2, 1, 0])
    shape = (3, 4)
    rows = np.array([0, 0, 0, 1, 2, 2, 2])
    cols = np.array([0, 1, 2, 0, 0, 1, 3])

    def search(r, c):
        for p in range(indptr[r], indptr[r + 1]):
            if indices[p] == c:
                return p
        return -1

    expected = [search(r, c) for r, c in zip(rows, cols)]
    assert expected == [1, -1, 0, -1, 4, 3, -1]
    np.testing.assert_array_equal(csr_point_positions(indptr, indices, shape, rows, cols), expected)
    assert csr_point_positions(indptr, indices, shape, np.array([3]), np.array([0])) is None
    assert csr_point_positions(indptr, indices, shape, np.array([0]), np.array([4])) is None
    empty = csr_array((3, 4))
    np.testing.assert_array_equal(
        csr_point_positions(empty.indptr, empty.indices, shape, rows, cols), -np.ones(rows.size))
