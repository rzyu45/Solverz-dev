"""An indexed Solverz symbol such as ``m.x[i]`` carries its own index (issue #168).

``IdxSymBasic`` is a SymPy ``Symbol``. Its printed name drops what an index
does not print, namely the token of a ``Set`` index and the bounds of an
``Idx``, so two constructions on different indices used to compare equal and
share one cached instance. SymPy's caches, keyed on that equality, then handed
a later model an expression that held an earlier model's index.
"""
import warnings

import numpy as np
import sympy as sp
from scipy.sparse import csc_array

from Solverz import LoopEqn, Model, Param, Set, Sum, Var, sin
from Solverz.sym_algebra.symbols import Para


def _set_index(members):
    m = Model()
    m.x = Var('x', np.zeros(10))
    m.S = Set('S', np.array(members))
    return m, m.S.idx('i')


def test_a_second_construction_does_not_rewrite_the_first():
    m1, i1 = _set_index([1, 2, 3])
    a1 = m1.x[i1]
    m2, i2 = _set_index([4, 5, 6])
    a2 = m2.x[i2]
    assert a1.index.token == i1.token
    assert a2.index.token == i2.token
    assert a1 != a2
    assert a1 == m1.x[i1] and hash(a1) == hash(m1.x[i1])


def test_sympy_caches_do_not_return_another_models_index():
    m1, i1 = _set_index([1, 2, 3])
    e1 = sin(m1.x[i1])
    m2, i2 = _set_index([4, 5, 6])
    e2 = sin(m2.x[i2])
    assert e2 is not e1
    assert e2.args[0].index.token == i2.token


def test_equality_follows_the_index_and_not_its_printed_form():
    m = Model()
    m.x = Var('x', np.zeros(10))
    assert m.x[sp.Idx('j', 4)] != m.x[sp.Idx('j', 9)]
    assert m.x[0] == m.x[np.int64(0)]
    assert m.x[1:3] == m.x[1:3] and m.x[1:3] != m.x[1:4]
    assert m.x[[1, 2]] == m.x[[1, 2]] and hash(m.x[[1, 2]]) == hash(m.x[[1, 2]])


def test_equality_follows_the_base_symbol():
    """``obtain_dim`` reads the dim of an indexed symbol, and
    ``Eqn.obtain_symbols`` registers its base symbol, whose value
    ``Equations.add_eqn`` then reads (issue #175)."""
    assert Para('A')[0] != Para('A', dim=2)[0]
    assert Para('p', value=[1.0])[0] != Para('p', value=[2.0])[0]
    assert Para('A', dim=2)[0, 1] == Para('A', dim=2)[0, 1]


def test_loop_eqn_jacobian_after_a_symbol_cache_eviction():
    """The cookbook shape: two power-flow-shaped builds in one process, with
    enough symbols created in between to evict the first build's indexed
    symbols from SymPy's Symbol cache while its ``sin(...)`` stays cached."""
    n = 9
    ring = np.zeros((n, n))
    for k in range(n):
        ring[k, (k + 1) % n] = ring[(k + 1) % n, k] = 1.0
        ring[k, k] = -2.0

    def unsound_after_build():
        m = Model()
        m.Vm = Var('Vm', np.ones(n))
        m.Va = Var('Va', np.linspace(0.0, 0.2, n))
        m.Bbus = Param('Bbus', csc_array(ring), dim=2, sparse=True)
        m.Bus = Set('Bus', n)
        m.PQ = Set('PQ', np.array([3, 4, 5, 6, 7, 8]))
        i_q = m.PQ.idx('i_q')
        j = m.Bus.idx('j')
        body = m.Vm[i_q] * Sum(m.Vm[j] * m.Bbus[i_q, j] * sin(m.Va[i_q] - m.Va[j]), j)
        m.Q_eqn = LoopEqn('Q_eqn', outer_index=i_q, body=body, model=m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            spf, y0 = m.create_instance()
            spf.FormJac(y0)
        return [str(w.message) for w in caught if 'structurally unsound' in str(w.message)]

    assert unsound_after_build() == []
    for k in range((sp.core.cache.SYMPY_CACHE_SIZE or 0) + 100):
        sp.Symbol(f'_sz_evict_{k}')
    assert unsound_after_build() == []
