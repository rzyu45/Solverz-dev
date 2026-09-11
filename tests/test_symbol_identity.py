"""A Solverz symbol such as ``Para('A', dim=2)`` carries its own dim and value (issue #175).

``SolSymBasic``, the base of ``Para``, ``iVar``, ``iAliasVar`` and ``idx``, is
a SymPy ``Symbol``. Its name says neither whether the symbol is a vector or a
matrix nor which value it carries, yet SymPy cached and compared it by that
name alone. A second construction of one name therefore rewrote the first, and
once SymPy had evicted the first, SymPy's other caches, keyed on that
equality, handed a later model an expression that held an earlier model's
symbol.
"""
import copy
import pickle
import warnings

import numpy as np
import sympy as sp

from Solverz import Eqn, MatVecMul, Mat_Mul, Model, Param, Var, made_numerical
from Solverz.equation.equations import AE
from Solverz.sym_algebra.symbols import Para, iVar, idx


def _evict_symbol_cache():
    for k in range((sp.core.cache.SYMPY_CACHE_SIZE or 0) + 100):
        sp.Symbol(f'_sz_evict_{k}')


def _matrix_model():
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', m.b - Mat_Mul(m.A, m.x))
    return m


def test_a_second_construction_does_not_rewrite_the_first():
    early = Mat_Mul(Para('A'), iVar('x'))
    _matrix_model()
    assert early.args[0].dim == 1
    p = Para('p', value=[1.0])
    Para('p', value=[2.0])
    np.testing.assert_array_equal(p.value, [1.0])


def test_equality_follows_dim_and_value():
    assert Para('A') != Para('A', dim=2)
    assert Para('A', dim=2) == Para('A', dim=2)
    assert hash(Para('A', dim=2)) == hash(Para('A', dim=2))
    assert Para('p', value=[1.0]) != Para('p', value=[2.0])
    assert idx('i', value=[0, 1]) != idx('i', value=[0, 2])
    # The printers and loop_jac rebuild a symbol from its name and dim, which
    # must still give the model's own symbol.
    m = _matrix_model()
    assert Para('A', dim=2) == m.A.symbol
    assert iVar('x') == m.x.symbol


def test_sympy_caches_do_not_return_another_models_symbol():
    """The CI failure: an earlier test built ``Mat_Mul`` on a 1-D ``A``, a
    later test evicted the Symbol cache, and the ``Mat_Mul`` cache then gave a
    model with a 2-D ``A`` the earlier expression."""
    early = Mat_Mul(Para('A'), iVar('x'))
    _evict_symbol_cache()
    smdl, y0 = _matrix_model().create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)
    np.testing.assert_allclose(mdl.J(y0.array, mdl.p).toarray(), [[-1, -3], [1, -2]])
    assert early.args[0].dim == 1


def test_an_equation_system_registers_its_own_symbol_value():
    x = iVar('x', value=[0.0])
    Eqn('g1', x - Para('q', value=[1.0]))
    _evict_symbol_cache()
    ae = AE([Eqn('g2', x - Para('q', value=[2.0]))])
    np.testing.assert_array_equal(ae.PARAM['q'].v, [2.0])


def test_matvec_derivative_is_the_two_dimensional_matrix():
    A, x = Para('A', dim=2), iVar('x')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        derivative = MatVecMul(A, x).diff(x)
    assert derivative == A


def test_a_copied_or_pickled_symbol_equals_the_original():
    for s in (Para('A', dim=2), Para('p', value=[1.0, 2.0]), idx('i', value=[0, 1]),
              iVar('_F_', internal_use=True)):
        assert copy.deepcopy(s) == s
        assert pickle.loads(pickle.dumps(s)) == s
