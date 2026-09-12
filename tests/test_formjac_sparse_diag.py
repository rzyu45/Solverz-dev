"""``FormJac`` evaluates a derivative that holds a ``Diag`` and depends on a
Var with the sparse ``SpDiag`` only. It evaluated it as ``np.diagflat`` too, a
dense n-by-n matrix, only to learn that the block is a matrix, which cost
memory and time in n squared (issue #185)."""
import importlib
import sys
import uuid

import numpy as np
import pytest
from scipy.sparse import csc_array, diags

from Solverz import Eqn, Mat_Mul, Model, Param, Var, module_printer

N = 40


def _matrix():
    return csc_array(diags([np.full(N - 1, -1.0), np.full(N, 4.0), np.full(N - 1, -1.0)],
                           [-1, 0, 1], shape=(N, N), format='csc'))


def _mutable_model():
    """``f = x * (A @ x) - b``, with the derivative ``Diag(A @ x) + Diag(x) @ A``."""
    m = Model()
    m.x = Var('x', np.linspace(0.5, 1.5, N))
    m.A = Param('A', _matrix(), dim=2, sparse=True)
    m.b = Param('b', np.ones(N))
    m.f = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)
    return m.create_instance()


def _constant_model():
    """``f = p * (A @ x) - b``: the derivative ``Diag(p) @ A`` holds no Var."""
    m = Model()
    m.x = Var('x', np.linspace(0.5, 1.5, N))
    m.A = Param('A', _matrix(), dim=2, sparse=True)
    m.p = Param('p', np.linspace(1.0, 2.0, N))
    m.b = Param('b', np.ones(N))
    m.f = Eqn('f', m.p * Mat_Mul(m.A, m.x) - m.b)
    return m.create_instance()


@pytest.fixture
def diagflat_calls(monkeypatch):
    """Count the calls of ``np.diagflat``, which the lambdified derivatives
    look up on the NumPy module at every call."""
    calls = []
    dense = np.diagflat

    def counted(v, *args, **kwargs):
        calls.append(np.size(v))
        return dense(v, *args, **kwargs)
    monkeypatch.setattr(np, 'diagflat', counted)
    return calls


def test_mutable_diag_block_is_not_evaluated_densely(diagflat_calls):
    spf, y0 = _mutable_model()
    spf.FormJac(y0)
    assert diagflat_calls == []
    (jb,) = [jb for row in spf.jac.blocks.values() for jb in row.values()]
    assert jb.is_mutable_matrix
    A = _matrix().toarray()
    x = np.asarray(y0['x'])
    expected = np.diag(A @ x) + np.diag(x) @ A
    # the pattern of the block is that of the derivative
    assert set(zip(jb.CooRow.tolist(), jb.CooCol.tolist())) == set(zip(*np.nonzero(expected)))


def test_constant_diag_block_keeps_its_evaluation(diagflat_calls):
    spf, y0 = _constant_model()
    spf.FormJac(y0)
    assert diagflat_calls == [N]


def test_rendered_jacobian_of_the_mutable_block(tmp_path):
    spf, y0 = _mutable_model()
    name = f'sz_test_sparse_diag_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=str(tmp_path), jit=True).render()
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(name)
        rng = np.random.default_rng(185)
        A = _matrix().toarray()
        for shift in (0.0, 1.0):
            x = np.linspace(0.5, 1.5, N) + shift * rng.uniform(-0.3, 0.3, N)
            mod.y['x'] = x
            J = csc_array(mod.mdl.J(mod.y, mod.mdl.p)).toarray()
            np.testing.assert_allclose(J, np.diag(A @ x) + np.diag(x) @ A, rtol=1e-14, atol=1e-14)
    finally:
        sys.path.remove(str(tmp_path))
        for k in [k for k in sys.modules if k.startswith(name)]:
            del sys.modules[k]
