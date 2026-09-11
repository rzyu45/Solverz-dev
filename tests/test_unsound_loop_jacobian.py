"""A structurally unsound canonical LoopEqn Jacobian fails the build (issue #177).

``check_canonical_invariants`` used to warn, so a model whose canonical
Jacobian carried an escaped ``Sum`` dummy still built and ran on a wrong
Jacobian. It now raises ``UnsoundLoopJacobianError``, and
``LoopEqn.derive_derivative`` names the equation and the variable.
"""
import numpy as np
import pytest
from scipy.sparse import csc_array

import Solverz.equation.loop_jac as loop_jac
from Solverz import LoopEqn, Model, Param, Set, Sum, Var, sin
from Solverz.equation.loop_jac import UnsoundLoopJacobianError

# The ring has one equation per PQ bus and two variables per bus, which
# create_instance reports; the size is not what these tests check.
pytestmark = pytest.mark.filterwarnings('ignore:Equation size')


def _ring_model(n=9):
    ring = np.zeros((n, n))
    for k in range(n):
        ring[k, (k + 1) % n] = ring[(k + 1) % n, k] = 1.0
        ring[k, k] = -2.0
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
    return m


def test_an_escaped_sum_dummy_fails_create_instance(monkeypatch):
    """Switching off both label comparisons of a ``KroneckerDelta`` with
    the ``Sum`` dummy, the match and the test for a reference, reproduces
    the cookbook CI failure fixed in #163. Both used SymPy equality then,
    and a ``SetIdx`` met a plain ``Idx`` of the same label. The delta on
    the dummy is then lifted out of its ``Sum``, and the build must stop.
    With the match alone switched off, the delta stays inside its ``Sum``
    as a reference to the dummy, and the canonical form is sound."""
    monkeypatch.setattr(loop_jac, '_index_matches', lambda expr, idx: False)
    monkeypatch.setattr(loop_jac, '_references_index', lambda expr, idx: False)
    with pytest.raises(UnsoundLoopJacobianError, match=r'for Q_eqn w\.r\.t\. V') as info:
        _ring_model().create_instance()
    assert any('outside every Sum' in p for p in info.value.problems)


def test_a_sound_model_builds_without_error():
    spf, y0 = _ring_model().create_instance()
    spf.FormJac(y0)
