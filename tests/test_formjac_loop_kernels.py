"""``FormJac`` does not evaluate the LoopEqn Jacobian kernels (issue #180).

A ``LoopEqnDiff`` block holds the sparsity pattern that
``compute_loop_jac_sparsity`` derived from its canonical expression, and the
generated ``J_`` writes the data of the block at every call, so ``FormJac``
needs the pattern only. It used to run every kernel twice in plain Python,
which took most of the time of ``render()`` on a large model.
"""
import importlib
import uuid

import numpy as np
from scipy.sparse import csc_array, diags

from Solverz import Eqn, LoopEqn, Model, Param, Set, Sum, Var, module_printer, sin
from Solverz.equation.eqn import LoopEqnDiff


def _rowsum_model(n=8, r=3):
    """The model of issue #179 and a scalar ``Eqn``, so that ``Fy`` returns a
    ``LoopEqnDiff`` block and an ordinary derivative."""
    offsets = list(range(-(r // 2), r // 2 + 1))
    A = diags([np.full(n - abs(o), 1.0 + 0.1 * abs(o)) for o in offsets], offsets,
              shape=(n, n), format='csc')
    m = Model()
    m.x = Var('x', np.linspace(0.1, 0.2, n))
    m.z = Var('z', 0.5)
    m.A = Param('A', csc_array(A), dim=2, sparse=True)
    m.b = Param('b', np.ones(n))
    m.N = Set('N', n)
    i, j = m.N.idx('i'), m.N.idx('j')
    m.f = LoopEqn('f', outer_index=i, body=m.x[i] * Sum(m.A[i, j] * sin(m.x[j]), j) - m.b[i], model=m)
    m.g = Eqn('g', m.z ** 2 - 1)
    return m.create_instance()


def _loop_diffs(spf):
    diffs = [ed for eqn in spf.EQNs.values() for ed in eqn.derivatives.values()
             if isinstance(ed, LoopEqnDiff)]
    assert diffs, 'the model must reach the Phase J3 kernel'
    return diffs


def _forbid_kernel_calls(monkeypatch, diffs):
    def fail(*args):
        raise AssertionError('a LoopEqn Jacobian kernel was evaluated')

    for ed in diffs:
        monkeypatch.setattr(ed, 'NUM_EQN', fail)


def test_formjac_keeps_the_pattern_without_running_the_kernel(monkeypatch):
    spf, y0 = _rowsum_model()
    diffs = _loop_diffs(spf)
    _forbid_kernel_calls(monkeypatch, diffs)
    spf.FormJac(y0)
    blocks = {id(jb._loop_eqn_diff): jb for jbs in spf.jac.blocks.values()
              for jb in jbs.values() if hasattr(jb, '_loop_eqn_diff')}
    for ed in diffs:
        jb = blocks[id(ed)]
        np.testing.assert_array_equal(jb.CooRow, ed._sparsity_row)
        np.testing.assert_array_equal(jb.CooCol, ed._sparsity_col)
        assert jb.SpEleSize == ed._nnz


def test_fy_skips_only_loop_kernels_and_only_when_asked():
    spf, y0 = _rowsum_model()
    spf.assign_eqn_var_address(y0)
    for fy in spf.Fy(y0):
        assert fy[3] is not None
    for fy in spf.Fy(y0, eval_loop_kernels=False):
        assert (fy[3] is None) == isinstance(fy[2], LoopEqnDiff)


def test_rendered_jacobian_holds_the_kernel_values(monkeypatch, tmp_path):
    spf, y0 = _rowsum_model()
    (ed,) = _loop_diffs(spf)
    expected = ed.NUM_EQN(*spf.obtain_eqn_args(ed, y0))
    _forbid_kernel_calls(monkeypatch, [ed])
    name = f'_sz_formjac_180_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=str(tmp_path), jit=False).render()
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module(name)
    J = mod.mdl.J(mod.y, mod.mdl.p).toarray()
    # ``Value0`` of the block held ones, and ``J_`` must overwrite them.
    rows = spf.a['f'].start + ed._sparsity_row
    cols = spf.var_address['x'].start + ed._sparsity_col
    np.testing.assert_array_equal(J[rows, cols], expected)
