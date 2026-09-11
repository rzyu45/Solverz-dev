"""A LoopEqn kernel reads a transcendental function of one entry of a Var from
a vector computed once per call, and, unless ``expand_trig=False``, evaluates
the sine and cosine of a difference of two entries of a Var through those
vectors (issue #183).

The vectors hold the same function of the same numbers, so hoisting alone
keeps F and J bit for bit those of Solverz 0.11.1. The expansion changes them
at the level of rounding.
"""
import importlib
import re
import sys
import uuid

import numpy as np
import sympy as sp
from numba import njit
from scipy.sparse import csc_array

from Solverz import LoopEqn, Model, Param, Set, Sum, Var, cos, module_printer, sin
from Solverz.equation.eqn import expand_trig_differences
from Solverz.sym_algebra.functions import cos as sol_cos, sin as sol_sin

from .test_loop_jac_row_sums import _assert_same_bits, _matrix, _model, _pattern, _points, _reference

# The residual kernel that Solverz 0.11.1 generated for the model of issue #179.
F_KERNEL_0111 = '''
def f_0111(b, x):
    out = np.empty(x.size)
    for i in range(x.size):
        _sz_loop_acc_0 = 0.0
        for _sz_kk_0 in range(_sz_csr_A_indptr[i], _sz_csr_A_indptr[i + 1]):
            j = _sz_csr_A_indices[_sz_kk_0]
            _sz_loop_acc_0 += (_sz_csr_A_data[_sz_kk_0] * np.sin(x[j]))
        out[i] = ((-1 * b[i]) + (x[i] * _sz_loop_acc_0))
    return out
'''
TRIG = re.compile(r'np\.(?:sin|cos)\(')


def _functions(src):
    """Name → source of every top-level function of a rendered ``num_func.py``."""
    out = {}
    for part in re.split(r'(?m)^(?=@njit|def )', src):
        m = re.search(r'(?m)^def (\w+)\(', part)
        if m:
            out[m.group(1)] = part
    return out


def _kernels(functions):
    return {k: v for k, v in functions.items()
            if re.fullmatch(r'inner_F\d+|_sz_loop_jac_kernel_\d+', k)}


def _render(tmp_path, spf, y0, prefix):
    name = f'{prefix}_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=str(tmp_path), jit=True).render()
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))
    return mod, (tmp_path / name / 'num_func.py').read_text(), name


def _forget(name):
    for k in [k for k in sys.modules if k.startswith(name)]:
        del sys.modules[k]


def test_expand_trig_differences_rewrites_the_difference_of_two_var_entries():
    var_map = {'x': Var('x', np.ones(3)), 'p': Param('p', np.ones(3))}
    x, p = sp.IndexedBase('x'), sp.IndexedBase('p')
    i, j = sp.Idx('i'), sp.Idx('j')
    for c, s in ((sp.cos, sp.sin), (sol_cos, sol_sin)):
        expanded = expand_trig_differences(p[i] * c(x[i] - x[j]) + s(x[i] - x[j]), var_map)
        assert expanded == (p[i] * (c(x[i]) * c(x[j]) + s(x[i]) * s(x[j]))
                            + s(x[i]) * c(x[j]) - c(x[i]) * s(x[j]))
        # nothing but the difference of two entries of one Var
        for kept in (c(x[i] - p[j]), c(x[i] + x[j]), c(x[i] - x[j] + 1), c(2 * x[i] - x[j]), c(x[i])):
            assert expand_trig_differences(kept, var_map) == kept
    e = sp.cos(x[i] - x[j]) + sp.sin(x[j] - x[i])
    at = {x[i]: 0.3, x[j]: -1.1}
    assert abs(float(expand_trig_differences(e, var_map).subs(at)) - float(e.subs(at))) < 1e-15


def test_hoisting_keeps_the_minimal_model_bit_for_bit(tmp_path):
    """``f_i = x_i Sum_j A[i, j] sin(x_j) - b_i``: the kernels read ``sin`` and
    ``cos`` of ``x`` from vectors that ``inner_F`` and ``inner_J`` compute, and
    F and J equal the kernels of 0.11.1 bit for bit."""
    A = _matrix()
    spf, y0 = _model(A)
    mod, src, name = _render(tmp_path, spf, y0, 'sz_test_hoist')
    try:
        functions = _functions(src)
        assert '_sz_h_sin_x = np.sin(x)' in functions['inner_F']
        assert '_sz_h_sin_x = np.sin(x)' in functions['inner_J']
        assert '_sz_h_cos_x = np.cos(x)' in functions['inner_J']
        kernels = _kernels(functions)
        assert sorted(kernels) == ['_sz_loop_jac_kernel_0', 'inner_F0']
        assert not [k for k, v in kernels.items() if TRIG.search(v)]

        csr = A.tocsr()
        ns = {'np': np,
              '_sz_csr_A_data': np.ascontiguousarray(csr.data, dtype=float),
              '_sz_csr_A_indices': np.ascontiguousarray(csr.indices, dtype=np.int64),
              '_sz_csr_A_indptr': np.ascontiguousarray(csr.indptr, dtype=np.int64)}
        exec(F_KERNEL_0111, ns)
        f_0111, j_0111 = njit(ns['f_0111']), _reference(A, jit=True)
        mdl, y = mod.mdl, mod.y
        b = np.asarray(mdl.p['b'], dtype=float)
        for x in _points(y):
            y['x'] = x
            _assert_same_bits(mdl.F(y, mdl.p), f_0111(b, x))
            J = csc_array(mdl.J(y, mdl.p))
            rows, cols = _pattern(J)
            _assert_same_bits(J.data, j_0111(x, rows, cols))
    finally:
        _forget(name)


def _polar_model(expand_trig):
    """The two injection equations of a polar power flow on a random network,
    each row sum over ``G`` and over ``B`` a Sum of its own, as in SolUtil."""
    n = 7
    rng = np.random.default_rng(183)
    dense = np.where(rng.random((n, n)) < 0.35, rng.uniform(-5, 5, (n, n)), 0.0)
    dense = dense + dense.T + np.diag(rng.uniform(5, 10, n))
    m = Model()
    m.Va = Var('Va', rng.uniform(-0.3, 0.3, n))
    m.Vm = Var('Vm', rng.uniform(0.95, 1.05, n))
    m.G = Param('G', csc_array(0.1 * dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(dense), dim=2, sparse=True)
    m.P = Param('P', rng.uniform(-1, 1, n))
    m.Q = Param('Q', rng.uniform(-1, 1, n))
    m.N = Set('N', n)
    i, j = m.N.idx('i'), m.N.idx('j')
    kw = {} if expand_trig is None else {'expand_trig': expand_trig}
    m.p_eqn = LoopEqn('p_eqn', outer_index=i, model=m, **kw,
                      body=m.Vm[i] * Sum(m.G[i, j] * m.Vm[j] * cos(m.Va[i] - m.Va[j]), j)
                      + m.Vm[i] * Sum(m.B[i, j] * m.Vm[j] * sin(m.Va[i] - m.Va[j]), j) - m.P[i])
    m.q_eqn = LoopEqn('q_eqn', outer_index=i, model=m, **kw,
                      body=m.Vm[i] * Sum(m.G[i, j] * m.Vm[j] * sin(m.Va[i] - m.Va[j]), j)
                      - m.Vm[i] * Sum(m.B[i, j] * m.Vm[j] * cos(m.Va[i] - m.Va[j]), j) - m.Q[i])
    return m.create_instance()


def test_expanded_trig_agrees_with_the_difference_to_rounding(tmp_path):
    """By default the kernels of a polar power flow evaluate no sine or cosine;
    with ``expand_trig=False`` they evaluate that of the difference, as in
    0.11.1, and F and J agree to rounding."""
    values = {}
    for expand in (None, False):
        spf, y0 = _polar_model(expand)
        assert all(eqn.expand_trig is (expand is None) for eqn in spf.EQNs.values())
        mod, src, name = _render(tmp_path, spf, y0, f'sz_test_expand_{expand}')
        try:
            functions = _functions(src)
            kernels = _kernels(functions)
            assert len(kernels) == 6
            if expand is None:
                assert not [k for k, v in kernels.items() if TRIG.search(v)]
                for dispatcher in ('inner_F', 'inner_J'):
                    assert '_sz_h_cos_Va = np.cos(Va)' in functions[dispatcher]
                    assert '_sz_h_sin_Va = np.sin(Va)' in functions[dispatcher]
            else:
                assert '_sz_h_' not in src
                assert all(re.search(r'np\.(?:sin|cos)\(\(\(-1 \* Va\[', v) for v in kernels.values())
            mdl, y = mod.mdl, mod.y
            rng = np.random.default_rng(7)
            shift = {v: rng.uniform(-0.2, 0.2, np.size(y[v])) for v in ('Va', 'Vm')}
            out = []
            for point in range(2):
                for v in ('Va', 'Vm'):
                    y[v] = np.asarray(y0[v], dtype=float) + point * shift[v]
                J = csc_array(mdl.J(y, mdl.p))
                out.append((np.array(mdl.F(y, mdl.p)), J.indptr.copy(), J.indices.copy(), J.data.copy()))
            values[expand] = out
        finally:
            _forget(name)
    for (F1, p1, i1, J1), (F0, p0, i0, J0) in zip(values[None], values[False]):
        np.testing.assert_array_equal(p1, p0)
        np.testing.assert_array_equal(i1, i0)
        assert np.abs(F1 - F0).max() <= 1e-13 * np.abs(F0).max()
        assert np.abs(J1 - J0).max() <= 1e-13 * np.abs(J0).max()
