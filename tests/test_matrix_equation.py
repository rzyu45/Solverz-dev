import copy
import os
import shutil
import sys
import warnings

import numpy as np
import scipy.sparse.linalg
from Solverz import (MatVecMul, Mat_Mul, Var, Param, made_numerical, Model, Eqn,
                     nr_method, module_printer)


# %%
def test_matrix_equation1():
    """A@x-b=0 using legacy MatVecMul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m.eqnf = Eqn('eqnf', m.b - MatVecMul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_matrix_equation2():
    """-A@x+b=0 using legacy MatVecMul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m.eqnf = Eqn('eqnf', - m.b + MatVecMul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


# --- Mat_Mul tests (new unified interface) ---

def test_mat_mul_inline():
    """A@x-b=0 using Mat_Mul in inline mode"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', m.b - Mat_Mul(m.A, m.x))

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_mat_mul_negative():
    """-A@x+b=0 using Mat_Mul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', -m.b + Mat_Mul(m.A, m.x))

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_mat_mul_nonlinear():
    """A@x + x^2 - b = 0: mutable matrix Jacobian (A + diag(2x))"""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.A, m.x) + m.x ** 2 - m.b)

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([1.0, 1.0]), atol=1e-5)


def _build_mutable_matrix_model():
    """x * (A@x) - b = 0: Jacobian = diag(A@x) + diag(x)@A (mutable matrix)."""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.b = Param('b', [4.0, 5.0])
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)
    return m


def test_mat_mul_mutable_jac(tmp_path):
    """Mutable matrix Jacobian: inline, module (jit=False), module (jit=True)
    must produce identical Jacobians at every Newton step.
    """
    m = _build_mutable_matrix_model()
    spf, y0 = m.create_instance()

    # --- 1. Inline ---
    mdl_inline = made_numerical(spf, y0, sparse=True)

    # --- 2. Module (jit=False) ---
    dir_nojit = str(tmp_path / 'nojit')
    printer = module_printer(spf, y0, 'mut_nojit', directory=dir_nojit, jit=False)
    printer.render()
    sys.path.insert(0, dir_nojit)
    from mut_nojit import mdl as mdl_mod, y as y_mod

    # --- 3. Module (jit=True) ---
    dir_jit = str(tmp_path / 'jit')
    printer_jit = module_printer(spf, y0, 'mut_jit', directory=dir_jit, jit=True)
    printer_jit.render()
    sys.path.insert(0, dir_jit)
    from mut_jit import mdl as mdl_jit, y as y_jit

    # --- Compare Jacobian at each Newton step (drive iteration via inline) ---
    y_test = copy.deepcopy(y0)
    for step in range(6):
        J_i = mdl_inline.J(y_test, mdl_inline.p)
        J_m = mdl_mod.J(y_test, mdl_mod.p)
        J_j = mdl_jit.J(y_test, mdl_jit.p)
        np.testing.assert_allclose(
            J_i.toarray(), J_m.toarray(), atol=1e-12,
            err_msg=f"module(jit=False) J mismatch at step {step}")
        np.testing.assert_allclose(
            J_i.toarray(), J_j.toarray(), atol=1e-12,
            err_msg=f"module(jit=True) J mismatch at step {step}")
        F_val = mdl_inline.F(y_test, mdl_inline.p)
        dy = scipy.sparse.linalg.spsolve(J_i, -F_val)
        y_test.array[:] = y_test.array + dy

    # --- Three modes must converge to the same solution ---
    sol_inline = nr_method(mdl_inline, y0)
    sol_mod = nr_method(mdl_mod, y_mod)
    sol_jit = nr_method(mdl_jit, y_jit)
    np.testing.assert_allclose(sol_inline.y.array, sol_mod.y.array, atol=1e-10)
    np.testing.assert_allclose(sol_inline.y.array, sol_jit.y.array, atol=1e-10)


# ---- Regression tests for review findings ----

def test_multi_diag_accumulation(tmp_path):
    """Finding 4: multiple independent ``Diag(...)`` terms must
    accumulate on the diagonal, not overwrite. The derivative of
    ``x*(A@x) + x*(B@x) - b`` w.r.t. ``x`` is
    ``diag(A@x) + diag(x)@A + diag(B@x) + diag(x)@B`` — two diag
    terms both land on ``(i,i)`` positions and their values must sum.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.b = Param('b', [4.0, 5.0])
    m.B = Param('B', [[1.5, 0.5], [0.5, 2.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) + m.x * Mat_Mul(m.B, m.x) - m.b)

    smdl, y0 = m.create_instance()
    mdl_inline = made_numerical(smdl, y0, sparse=True)

    dir_mod = str(tmp_path / 'multi_diag_mod')
    printer = module_printer(smdl, y0, 'multi_diag_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    sys.path.insert(0, dir_mod)
    import importlib
    if 'multi_diag_mod' in sys.modules:
        del sys.modules['multi_diag_mod']
    mod = importlib.import_module('multi_diag_mod')
    mdl_mod = mod.mdl
    y_mod = mod.y

    # Drive by a non-trivial iterate so every diag term has a distinct,
    # non-zero value at (i,i). The bug manifests as module J missing
    # one of the diagonal contributions.
    rng = np.random.default_rng(20260413)
    y_seed = rng.uniform(0.5, 1.5, size=y0.array.shape[0])
    y_test = copy.deepcopy(y0)
    y_test.array[:] = y_seed
    y_test_mod = copy.deepcopy(y_mod)
    y_test_mod.array[:] = y_seed
    J_i = mdl_inline.J(y_test, mdl_inline.p)
    J_m = mdl_mod.J(y_test_mod, mdl_mod.p)
    np.testing.assert_allclose(
        J_i.toarray(), J_m.toarray(), rtol=1e-10, atol=1e-12,
        err_msg='module J misses accumulation across multiple Diag terms')


def test_reserved_prefix_rejected():
    """Findings 5,6: user symbols starting with reserved internal
    prefixes must be rejected at construction time so the code
    generator can freely emit helper names of the form
    ``_sz_mm_<int>`` and ``_sz_mb_<int>_...`` without shadowing."""
    import pytest
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Var('_sz_mm_0', [1.0])
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Var('_sz_mb_0_u0', [1.0])
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Param('_sz_mm_42', [1.0])


def test_triggerable_vector_next_to_matmul_allowed():
    """Finding 2 (narrowed in review pass R1): a triggerable *vector*
    parameter sitting next to a ``Mat_Mul`` — but **not** inside its
    matrix operand — must be allowed. The vector's current value is
    read from ``p_`` on every Newton step, so the Layer-2 scatter-add
    cache never sees a stale copy of it. Rejecting this case was the
    original R1 complaint.
    """
    import scipy.sparse.linalg
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [5.0, 6.0])
    m.K = Param('K', [1.0, 1.0], triggerable=True,
                trigger_var=['x'], trigger_fun=lambda x: 2 * x)
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.eqn = Eqn('f', m.K * m.x + Mat_Mul(m.A, m.x) - m.b)

    smdl, y0 = m.create_instance()
    # The narrow check must NOT fire for a triggerable scalar vector.
    smdl.FormJac(y0)
    mdl = made_numerical(smdl, y0, sparse=True)
    # And the model must still compute a correct Newton step. Only
    # sanity-check the residual at y0 (not convergence, since the
    # triggerable term has no dK/dx in the analytical Jacobian).
    F0 = mdl.F(y0, mdl.p)
    J0 = mdl.J(y0, mdl.p)
    assert F0.shape == (2,)
    assert J0.shape == (2, 2)


def test_triggerable_sparse_matrix_rejected_at_construction():
    """Time-varying sparse ``dim=2`` matrices are unsupported in
    Solverz: every downstream code path (``MatVecMul``, ``Mat_Mul``
    fast path, mutable-matrix Jacobian scatter-add) caches the
    matrix's CSC fields at model-build time, so a runtime trigger
    update would silently be ignored and produce wrong Newton
    steps. Construction must fail at the ``Param(...)`` line
    itself — not deep inside ``FormJac`` — so the error points at
    the offending declaration in the user's source.
    """
    import pytest
    from scipy.sparse import csc_array
    with pytest.raises(NotImplementedError,
                       match=r"sparse 2-D ``Param``.*triggerable"):
        Param('A', csc_array(np.array([[2., 1.], [1., 3.]])),
              dim=2, sparse=True,
              triggerable=True, trigger_var=['x'],
              trigger_fun=lambda x: csc_array(
                  np.array([[2., 1.], [1., 3.]])))


def test_timeseries_sparse_matrix_rejected_at_construction():
    """Same policy for ``TimeSeriesParam``: a sparse ``dim=2``
    time-series param is rejected at construction, independently
    of whether it's ever used in a ``Mat_Mul``.
    """
    import pytest
    from Solverz import TimeSeriesParam
    from scipy.sparse import csc_array
    I = csc_array(np.eye(2))
    with pytest.raises(NotImplementedError,
                       match=r"sparse 2-D ``TimeSeriesParam``"):
        TimeSeriesParam('A', v_series=[I, I],
                        time_series=[0.0, 1.0],
                        dim=2, sparse=True)


def test_formjac_backstop_rejects_tainted_triggerable_sparse_param():
    """Backstop check: even if a triggerable sparse ``dim=2`` Param
    slips past ``Param.__init__`` (e.g. via ``__new__`` and
    attribute assignment, or post-hoc flag mutation), the
    ``FormJac`` backstop ``_check_no_timevar_sparse_matrices`` must
    still raise ``NotImplementedError``.

    Tested directly at the method level, rather than by trying to
    coerce a full ``Model`` / ``create_instance`` flow into a
    tainted state — the intervening ``trigger_param_updater``
    machinery fights back when ``triggerable`` is flipped after
    construction, and the point here is that the backstop catches
    a tainted ``PARAM`` dict regardless of how it got that way.
    """
    import pytest
    from scipy.sparse import csc_array
    from Solverz.equation.equations import Equations as SymEquations

    # Build a legal Equations-like container and stuff a tainted
    # Param into its PARAM dict by bypassing ``Param.__init__``.
    eqs = SymEquations.__new__(SymEquations)
    eqs.PARAM = {}

    tainted = Param.__new__(Param)
    tainted.name = 'A'
    tainted.dim = 2
    tainted.sparse = True
    tainted.triggerable = True
    tainted.trigger_var = None
    tainted.trigger_fun = None
    tainted.is_alias = False
    tainted.dtype = float
    tainted._ParamBase__v = csc_array(np.eye(2))
    eqs.PARAM['A'] = tainted

    with pytest.raises(NotImplementedError,
                       match=r"triggerable sparse ``dim=2`` parameter"):
        eqs._check_no_timevar_sparse_matrices()


def test_dense_dim2_param_warning():
    """Finding 3: dense ``dim=2`` parameters used in Mat_Mul must
    produce a warning pointing at the performance cost. They should
    still work correctly (via the fallback path)."""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    m.A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]),
                dim=2, sparse=False)
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)

    smdl, y0 = m.create_instance()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        smdl.FormJac(y0)
        dense_warnings = [x for x in w
                          if issubclass(x.category, UserWarning)
                          and 'dense 2-D' in str(x.message)]
        assert len(dense_warnings) >= 1, \
            'expected a UserWarning about dense dim=2 in Mat_Mul'


def test_dense_dim2_param_jacobian_numerical(tmp_path):
    """Finding 3 (R6.1): dense ``dim=2`` ``Param(sparse=False)`` must
    not just warn — it must produce a *numerically correct* Jacobian
    via the ``MutableMatJacDataModule`` fallback path. Inline mode
    already works (it always re-evaluates the expression); the
    regression risk is the module printer, which until the R1 fix
    for the dense-vs-sparse decomposition would crash or silently
    mis-index the fallback. Compare the two paths block-for-block.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    m.A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]),
                dim=2, sparse=False)
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)

    spf, y0 = m.create_instance()

    # --- 1. Inline ---
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mdl_inline = made_numerical(spf, y0, sparse=True)

    # --- 2. Module (jit=True) ---
    dir_mod = str(tmp_path / 'dense_mat')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        printer = module_printer(spf, y0, 'dense_mat_mod',
                                 directory=dir_mod, jit=True)
        printer.render()
    sys.path.insert(0, dir_mod)
    from dense_mat_mod import mdl as mdl_mod, y as y_mod

    # Drive by a non-trivial iterate so every block gets a distinct
    # numeric value. 5 steps is plenty for NR on 2 unknowns.
    y_test = copy.deepcopy(y0)
    y_test_mod = copy.deepcopy(y_mod)
    rng = np.random.default_rng(20260413)
    seed = rng.uniform(0.5, 1.5, size=y0.array.shape[0])
    y_test.array[:] = seed
    y_test_mod.array[:] = seed
    for step in range(5):
        J_i = mdl_inline.J(y_test, mdl_inline.p)
        J_m = mdl_mod.J(y_test_mod, mdl_mod.p)
        np.testing.assert_allclose(
            J_i.toarray(), J_m.toarray(), rtol=1e-10, atol=1e-12,
            err_msg=f'dense dim=2 fallback J mismatch at step {step}')
        F_val = mdl_inline.F(y_test, mdl_inline.p)
        dy = scipy.sparse.linalg.spsolve(J_i, -F_val)
        y_test.array[:] = y_test.array + dy
        y_test_mod.array[:] = y_test_mod.array + dy


def test_timeseries_param_next_to_matmul_allowed():
    """Finding 2 / R1 (R6.2): a ``TimeSeriesParam`` (always 1-D) can
    never be a ``Mat_Mul`` matrix operand, so the narrow R1
    triggerable check never fires for it. A time-series scalar
    sitting next to a ``Mat_Mul`` must therefore be allowed to build
    and the J_ wrapper must be able to compute F and J without error.
    Locks in that R1's narrowing also covers TimeSeriesParam.
    """
    from Solverz import TimeSeriesParam
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [5.0, 6.0])
    # Scalar-valued time series — 1-D is the only shape TimeSeriesParam
    # supports, which is also exactly why R1 narrowing is safe for it.
    m.t_scale = TimeSeriesParam('t_scale',
                                v_series=[1.0, 2.0],
                                time_series=[0.0, 1.0])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    # TimeSeriesParam modulates the forcing term, not the variable
    # being differentiated. This keeps the mutable-matrix Jacobian
    # independent of t_scale while still making t_scale a free symbol
    # of an equation that also contains Mat_Mul — exactly the scenario
    # the narrow R1 check is meant to accept.
    m.eqn = Eqn('f', Mat_Mul(m.A, m.x) - m.t_scale * m.b)

    smdl, y0 = m.create_instance()
    # The sole R1 assertion: the narrow check must NOT reject a
    # TimeSeriesParam that sits in a Mat_Mul equation as long as it
    # isn't the Mat_Mul matrix operand (which TimeSeriesParam can
    # never be anyway — it's 1-D only). Downstream AE evaluation of
    # TimeSeriesParam needs a ``t`` in the wrapper signature that
    # plain AE models don't expose; that's orthogonal to R1 and is
    # intentionally not exercised here.
    smdl.FormJac(y0)
    # The Jacobian block for ∂f/∂x is just ``A`` (constant), so we
    # can inspect it structurally without calling F_ / J_.
    assert len(smdl.jac.blocks) == 1
    [jbs_row] = smdl.jac.blocks.values()
    assert all(jb.DeriType == 'matrix' and not jb.is_mutable_matrix
               for jb in jbs_row.values())


def test_selective_njit_gating_element_wise_happy_path(tmp_path):
    """Finding 1 (R6.3 positive case): a pure element-wise model with
    NO sparse params must still produce ``@njit``-decorated ``inner_F``
    / ``inner_J``. This locks in that the selective gating in
    ``_has_sparse_in_param_list`` does not accidentally strip @njit
    from the happy path.
    """
    m = Model()
    m.x = Var('x', [1.0, 1.0])
    m.a = Param('a', 2.0)
    m.eqn = Eqn('f', m.a * m.x - 1.0)
    smdl, y0 = m.create_instance()

    dir_mod = str(tmp_path / 'njit_happy')
    printer = module_printer(smdl, y0, 'njit_happy_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    num_func_path = os.path.join(dir_mod, 'njit_happy_mod', 'num_func.py')
    with open(num_func_path) as f:
        source = f.read()
    assert '@njit(cache=True)\ndef inner_F(' in source, \
        'Expected @njit on inner_F for pure element-wise model'
    assert '@njit(cache=True)\ndef inner_J(' in source, \
        'Expected @njit on inner_J for pure element-wise model'


# NOTE: the 0.8.1 R6.3 negative test (sparse triggerable Param →
# @njit skipped) has been removed because sparse ``dim=2``
# triggerable / TimeSeriesParam instances are now rejected at
# ``Param(...)`` construction time (see
# ``test_triggerable_sparse_matrix_rejected_at_construction``). Any
# equation system that previously relied on the @njit gating for
# a sparse 2-D time-varying param is now a construction error; the
# ``_has_sparse_in_param_list`` function is still kept as defence
# in depth for sparse 1-D time-varying params (an edge case that
# is neither rejected nor common in practice).


def test_nested_matmul_smoke(tmp_path):
    """R6.4: ``Mat_Mul(A, Mat_Mul(B, x))`` — nested Mat_Mul — must
    render (the ``extract_matmuls`` helper creates one placeholder
    per nested level) and solve to the analytical solution.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    # f = A @ B @ x - b
    m.eqn = Eqn('f', Mat_Mul(m.A, Mat_Mul(m.B, m.x)) - m.b)

    spf, y0 = m.create_instance()

    # Inline solve
    mdl_inline = made_numerical(spf, y0, sparse=True)
    sol_inline = nr_method(mdl_inline, y0)

    # Module solve (exercise the precompute deduplication + rendering)
    dir_mod = str(tmp_path / 'nested_mm')
    printer = module_printer(spf, y0, 'nested_mm_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    sys.path.insert(0, dir_mod)
    from nested_mm_mod import mdl as mdl_mod, y as y_mod
    sol_mod = nr_method(mdl_mod, y_mod)

    # Analytical: AB @ x = b → x = solve(AB, b)
    AB = np.array([[1.0, 0.5], [0.5, 1.0]]) @ np.array([[2.0, 0.0], [0.0, 2.0]])
    expected = np.linalg.solve(AB, np.array([3.0, 4.0]))
    np.testing.assert_allclose(sol_inline.y.array, expected, atol=1e-10)
    np.testing.assert_allclose(sol_mod.y.array, expected, atol=1e-10)


def test_nested_matmul_outer_fallback_demotes_inner(tmp_path):
    """Review R2: when the outer of a nested ``Mat_Mul`` cannot hit
    the fast path (e.g. ``Mat_Mul(-A, Mat_Mul(B, x))`` — the outer
    matrix is ``-A``, not a bare ``Para``), the inner placeholder
    must be **demoted** from the fast path to the fallback. The
    wrapper needs to define ``_sz_mm_0`` (the inner B@x) before the
    outer ``_sz_mm_1 = (-A) @ _sz_mm_0`` scipy SpMV runs. A classifier
    that looks only at ``matrix_arg`` leaves ``_sz_mm_0`` on the
    fast path (B is a bare sparse Para) and the generated wrapper
    references an undefined local.

    Locks in the dependency-aware classification added after the
    ``csc_matvec`` review pass.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    # Outer ``-A`` is Mul(-1, A) — not a bare Para — so it falls
    # back. Inner Mat_Mul(B, x) looks fast in isolation but must be
    # demoted because the outer fallback references it.
    m.eqn = Eqn('f', Mat_Mul(-m.A, Mat_Mul(m.B, m.x)) - m.b)

    spf, y0 = m.create_instance()

    # Inline solve (independent correctness reference)
    mdl_inline = made_numerical(spf, y0, sparse=True)
    sol_inline = nr_method(mdl_inline, y0)

    # Module solve — this is the path that had the R2 bug. If the
    # inner placeholder is still fast-path-classified, the wrapper
    # emits ``_sz_mm_1 = (-A) @ _sz_mm_0`` with ``_sz_mm_0`` never
    # defined anywhere and ``F_`` raises ``NameError`` on entry.
    dir_mod = str(tmp_path / 'nested_neg')
    printer = module_printer(spf, y0, 'nested_neg_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    sys.path.insert(0, dir_mod)
    from nested_neg_mod import mdl as mdl_mod, y as y_mod
    sol_mod = nr_method(mdl_mod, y_mod)

    # Analytical: -A @ B @ x = b → x = solve(-AB, b)
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected = np.linalg.solve(-A @ B, np.array([3.0, 4.0]))
    np.testing.assert_allclose(sol_inline.y.array, expected, atol=1e-10)
    np.testing.assert_allclose(sol_mod.y.array, expected, atol=1e-10)

    # Also inspect the generated source to make sure the wrapper
    # actually emits both ``_sz_mm_0`` and ``_sz_mm_1`` in
    # topological order. A regression that re-introduces the R2
    # bug would skip the ``_sz_mm_0`` assignment.
    num_func_path = os.path.join(dir_mod, 'nested_neg_mod', 'num_func.py')
    with open(num_func_path) as f:
        source = f.read()
    # The F_ wrapper body must contain both precompute assignments,
    # and _sz_mm_0 must come before _sz_mm_1 (topological order).
    f_body_start = source.index('def F_(')
    f_body_end = source.index('@njit', f_body_start)
    f_body = source[f_body_start:f_body_end]
    pos_mm0 = f_body.find('_sz_mm_0 = ')
    pos_mm1 = f_body.find('_sz_mm_1 = ')
    assert pos_mm0 >= 0, (
        'F_ wrapper missing _sz_mm_0 assignment — inner fast path '
        'was not demoted even though outer fallback depends on it')
    assert pos_mm1 >= 0, 'F_ wrapper missing _sz_mm_1 assignment'
    assert pos_mm0 < pos_mm1, (
        'F_ wrapper emits _sz_mm_1 before _sz_mm_0 — non-topological '
        'precompute order would break at runtime')


def test_print_F_keeps_sparse_params_in_param_list():
    """Review R1: ``print_F`` must not drop a wrapper assignment for
    a sparse ``dim=2`` ``Param`` whose name ends up in
    ``param_list``. ``print_param``'s ``TimeSeriesParam`` branch
    appends every time-series parameter to ``param_list``
    unconditionally, so a sparse time-series parameter is (a)
    loaded in the wrapper as ``A = p_["A"].get_v_t(t)`` and (b)
    passed through to ``inner_F`` as an argument. A filter that
    checks only ``dim == 2 and sparse`` would drop the wrapper
    load but still emit ``inner_F(..., A, ...)``, producing a
    ``NameError`` on the first call.

    We test the filter directly via ``print_F`` with a constructed
    ``PARAM`` dict — a sparse 2-D ``TimeSeriesParam`` is hard to
    build end-to-end through ``Model``/``create_instance`` because
    its ``v_series`` is forced to 1-D, but the filter logic is
    what the reviewer flagged and it is fully testable in
    isolation.
    """
    from scipy.sparse import csc_array
    from Solverz.code_printer.python.module.module_printer import print_F
    from Solverz.equation.param import TimeSeriesParam
    from Solverz.utilities.address import Address

    # Construct a 2-D sparse ``TimeSeriesParam`` by reaching past
    # the ``Array(v_series, dim=1)`` normalisation: set the
    # attributes the filter actually inspects. This is the minimum
    # shape the R1 regression reasons about.
    ts = TimeSeriesParam.__new__(TimeSeriesParam)
    ts.name = 'A'
    ts.dim = 2
    ts.sparse = True
    ts.is_alias = False
    ts.triggerable = False
    ts.trigger_var = None
    ts.trigger_fun = None
    ts.dtype = float
    # Stash a csc_array so the wrapper could actually use it if it
    # wanted to. The filter never dereferences v, but parse_p might.
    ts._ParamBase__v = csc_array(np.array([[1.0, 0.0], [0.0, 1.0]]))

    PARAM = {'A': ts}
    var_addr = Address()
    var_addr.add('x', 2)

    source = print_F('AE', var_addr, PARAM, nstep=0, precompute_info=None)
    # Before the R1 fix, the filter would strip the wrapper load
    # even though ``A`` is still in ``param_list`` — so ``inner_F``
    # received ``A`` as an argument with no prior definition.
    assert 'A = p_["A"].get_v_t' in source, (
        'print_F dropped the TimeSeriesParam wrapper load; inner_F '
        'will reference an undefined local A')
    # And ``inner_F`` is still called with A in its argument list
    # (we care about the symmetry).
    assert 'inner_F(_F_, x, A)' in source, (
        'print_F dropped A from the inner_F call site — the filter '
        'is now too aggressive in the other direction')


def test_matmul_three_arg_operand_falls_back(tmp_path):
    """Review R3: ``Mat_Mul(A, B, x)`` — Solverz allows 3+-argument
    ``Mat_Mul`` nodes, and ``extract_matmuls`` folds them into a
    single placeholder with ``matrix_arg = A`` and
    ``operand_arg = Mat_Mul(B, x)`` (the fresh inner ``Mat_Mul`` is
    *not* re-walked). A classifier that looks only at ``matrix_arg``
    would route this into the ``SolCF.csc_matvec`` fast path and
    emit a call whose operand is ``B @ x`` — but ``B`` is a sparse
    ``dim=2`` ``Para``, not available inside ``inner_F`` by name,
    so the generated code would fail.

    The fix is the ``_shape_is_fast`` operand check: fast path
    requires the operand to contain neither ``Mat_Mul`` nodes nor
    sparse ``dim=2`` ``Para`` references.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    # 3-arg Mat_Mul: A @ B @ x
    m.eqn = Eqn('f', Mat_Mul(m.A, m.B, m.x) - m.b)

    spf, y0 = m.create_instance()
    mdl_inline = made_numerical(spf, y0, sparse=True)
    sol_inline = nr_method(mdl_inline, y0)

    dir_mod = str(tmp_path / 'three_arg')
    printer = module_printer(spf, y0, 'three_arg_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    sys.path.insert(0, dir_mod)
    from three_arg_mod import mdl as mdl_mod, y as y_mod
    sol_mod = nr_method(mdl_mod, y_mod)

    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected = np.linalg.solve(A @ B, np.array([3.0, 4.0]))
    np.testing.assert_allclose(sol_inline.y.array, expected, atol=1e-10)
    np.testing.assert_allclose(sol_mod.y.array, expected, atol=1e-10)

    # The generated F_ wrapper must keep B loaded (it appears in the
    # fallback operand) and must NOT invoke ``SolCF.csc_matvec`` with
    # a ``(B@x)`` operand — a regression would do either.
    num_func_path = os.path.join(dir_mod, 'three_arg_mod', 'num_func.py')
    with open(num_func_path) as f:
        source = f.read()
    f_body_start = source.index('def F_(')
    f_body_end = source.index('@njit', f_body_start)
    f_body = source[f_body_start:f_body_end]
    assert 'B = p_["B"]' in f_body, (
        'F_ wrapper filter removed B load even though B is needed by '
        'the fallback Mat_Mul(A, B, x) evaluation')
    # And check that the csc_matvec fast path is NOT used for this
    # placeholder — a regression would emit
    # ``SolCF.csc_matvec(A_data, ..., (B@x))`` inside inner_F.
    inner_body_start = source.index('def inner_F(')
    inner_body_end = source.index('def inner_F0', inner_body_start)
    inner_body = source[inner_body_start:inner_body_end]
    assert 'SolCF.csc_matvec' not in inner_body, (
        'inner_F routed a Mat_Mul(A, B, x) placeholder through the '
        'csc_matvec fast path — operand is not a clean vector')


# --- Mat_Mul fallback diagnostic warnings (0.8.3) ---
#
# These tests pin the user-facing UserWarning that the module printer
# emits whenever a ``Mat_Mul(A, x)`` placeholder is forced onto the
# slower scipy.sparse fallback path. Each warning must (a) name the
# placeholder, (b) print the matrix expression that broke the fast
# path, and (c) suggest a concrete rewrite.
#
# Layer 1 (this section) covers ``Mat_Mul`` precompute placeholders.
# Layer 2 (mutable Jacobian fallback) is covered separately.

def test_matmul_negation_warns_with_rewrite_suggestion():
    """``Mat_Mul(-A, x)`` is on the fallback path because ``-A`` is
    not a bare sparse Param. The Layer 1 classifier must emit a
    UserWarning that:

    1. Names the placeholder (e.g. ``_sz_mm_0``).
    2. Describes the actual matrix expression as a negation.
    3. Suggests moving the negation outside Mat_Mul, i.e.
       ``-Mat_Mul(A, x)``.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.0, 0.0])
    m.b = Param('b', [1.0, 2.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(-m.A, m.x) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert '_sz_mm_' in msg, \
        f'warning should name the placeholder, got: {msg}'
    assert '-A' in msg or 'negation' in msg.lower(), (
        f'warning should describe the matrix expression as a '
        f'negation, got: {msg}')
    assert '-Mat_Mul' in msg or 'outside' in msg.lower(), (
        f'warning should suggest moving the negation outside, got: {msg}')


def test_matmul_sum_warns_with_distribute_suggestion():
    """``Mat_Mul(A+B, x)`` falls back because ``A+B`` is not a bare
    sparse Param. End-to-end test: drives ``print_F`` and asserts the
    resulting ``UserWarning`` names the placeholder, identifies the sum,
    and suggests distributing the product.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[0.5, 0.0], [0.0, 0.5]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.A + m.B, m.x) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert '_sz_mm_' in msg, \
        f'warning should name the placeholder, got: {msg}'
    assert 'sum' in msg.lower(), \
        f'warning should mention "sum", got: {msg}'
    assert 'A' in msg and 'B' in msg, \
        f'warning should name both matrices, got: {msg}'
    assert 'distribute' in msg.lower(), \
        f'warning should suggest distribution, got: {msg}'
    assert 'Mat_Mul(A' in msg and 'Mat_Mul(B' in msg, \
        f'suggestion should show both distributed Mat_Muls, got: {msg}'


def test_matmul_scalar_mul_warns_with_factor_suggestion():
    """``Mat_Mul(2*A, x)`` falls back because ``2*A`` is not a bare
    sparse Param. The Layer 1 classifier must emit a UserWarning that:

    1. Names the placeholder.
    2. Identifies the scalar coefficient.
    3. Suggests factoring: ``2 * Mat_Mul(A, x)``.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.0, 0.0])
    m.b = Param('b', [1.0, 2.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(2 * m.A, m.x) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert '_sz_mm_' in msg, \
        f'warning should name the placeholder, got: {msg}'
    assert ('scalar' in msg.lower()
            or 'coefficient' in msg.lower()
            or 'factor' in msg.lower()), (
        f'warning should mention scalar/factor, got: {msg}')
    assert 'outside' in msg.lower() or 'c *' in msg or '2 *' in msg, (
        f'warning should suggest moving scalar outside, got: {msg}')


def test_matmul_three_arg_operand_warns_with_nesting_suggestion():
    """``Mat_Mul(A, B, x)`` falls back because the operand contains an
    unresolved ``Mat_Mul`` (R3 multi-arg fold). The Layer 1 classifier
    must emit a UserWarning that:

    1. Names the placeholder.
    2. Identifies that the operand contains Mat_Mul.
    3. Suggests explicit two-level nesting:
       ``Mat_Mul(A, Mat_Mul(B, x))``.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.A, m.B, m.x) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert '_sz_mm_' in msg, \
        f'warning should name the placeholder, got: {msg}'
    assert ('nested' in msg.lower()
            or ('Mat_Mul' in msg and 'operand' in msg.lower()
                and 'nest' in msg.lower())), (
        f'warning should identify Mat_Mul in operand and suggest nesting, '
        f'got: {msg}')


def test_matmul_sparse_para_in_operand_warns_with_precompute_suggestion():
    """``Mat_Mul(A, f(B, x))`` where ``B`` is a sparse ``dim=2`` Param
    falls back because sparse ``dim=2`` symbols are not available by
    name inside ``inner_F`` (only their CSC flat fields are). The Layer 1
    classifier must return a precompute suggestion *and* the emission
    path must produce a ``UserWarning`` carrying that suggestion.

    No equation hits this case naturally (a sparse ``dim=2`` Param can
    only enter the operand of a ``Mat_Mul`` placeholder by way of
    another ``Mat_Mul``, which is the R3 case). So this test exercises
    the classifier and the emission entry point directly with a
    synthesised ``(name, mat, op)`` triple — the same triple shape
    ``_classify_matmul_placeholders`` would feed in production.
    """
    from Solverz.code_printer.python.module.module_printer import (
        _classify_l1_fallback_reason, _emit_l1_fallback_warnings)
    from Solverz.sym_algebra.symbols import iVar

    m = Model()
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    op = iVar('x') * m.B
    spf, y0 = m.create_instance()
    PARAM = spf.PARAM

    reason, expr_str, suggestion = _classify_l1_fallback_reason(m.A, op, PARAM)
    assert 'references' in reason.lower() and 'sparse' in reason.lower(), (
        f'reason should mention operand references a sparse Param, '
        f'got: {reason}')
    assert 'B' in reason, \
        f'reason should name the offending Param B, got: {reason}'
    assert ('precompute' in suggestion.lower()
            or 'vector' in suggestion.lower()), (
        f'suggestion should recommend precomputing as a vector, '
        f'got: {suggestion}')

    # Verify the emission path wraps the classifier output into a
    # ``UserWarning`` with the placeholder name and full message body.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _emit_l1_fallback_warnings(
            [('_sz_mm_test', m.A, op)],
            fast_candidates=set(),
            PARAM=PARAM,
        )
    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)]
    assert len(fb) == 1, (
        f'expected exactly one Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert "'_sz_mm_test'" in msg, \
        f'warning should name the placeholder, got: {msg}'
    assert 'precompute' in msg.lower() or 'vector' in msg.lower(), \
        f'warning should carry the precompute suggestion, got: {msg}'


def test_matmul_nested_demotion_warns_with_root_cause():
    """``Mat_Mul(-A, Mat_Mul(B, x))`` — the outer placeholder
    ``_sz_mm_0 = Mat_Mul(-A, _sz_mm_1)`` falls back (negation), and
    the inner ``_sz_mm_1 = Mat_Mul(B, x)`` is demoted because the
    outer fallback consumes it. The warning for the demoted inner
    placeholder must:

    1. Name which upstream placeholder caused the demotion.
    2. Explain the dependency chain.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.B = Param('B', [[2.0, 0.0], [0.0, 2.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(-m.A, Mat_Mul(m.B, m.x)) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)]
    assert len(fb) >= 2, (
        f'expected warnings for both outer fallback and demoted inner, '
        f'captured: {[str(x.message) for x in w]}')
    # Find the demotion warning — it should mention "demoted" and the
    # upstream placeholder that caused the demotion.
    demoted = [x for x in fb if 'demoted' in str(x.message).lower()]
    assert len(demoted) >= 1, (
        f'expected at least one demotion warning, '
        f'captured fallback warnings: {[str(x.message) for x in fb]}')
    assert '_sz_mm_' in str(demoted[0].message), (
        f'demotion warning should name placeholders, '
        f'got: {demoted[0].message}')
    # Verify the demoted placeholder name and upstream both appear.
    msg = str(demoted[0].message)
    assert 'consumes' in msg.lower() or 'references' in msg.lower() or 'depend' in msg.lower(), (
        f'demotion warning should explain the dependency, got: {msg}')


# --- Layer 2 (mutable Jacobian fallback) diagnostic warnings ---


def test_mutable_jac_fallback_warns_with_eqn_var_context():
    """When a mutable Jacobian block contains a term that doesn't match
    any supported fast-path shape (Diag / row-scale / col-scale /
    biscale), the module printer must emit a UserWarning per fallback
    piece that:

    1. Names the equation and variable.
    2. Prints the term that broke the fast path.
    3. Identifies the specific structural mismatch (here: element-wise
       ``Mul`` with ``Diag`` factors, not a matrix-product).
    4. Suggests rewriting as ``Mat_Mul``.

    Uses ``Diag(x) * A * Diag(y)`` (a raw ``Mul``, not wrapped in
    ``Mat_Mul``) as the trigger — it doesn't match any fast-path shape
    and falls through to ``fallback_pieces``.
    """
    from Solverz.code_printer.python.module.module_printer import (
        _emit_l2_fallback_warnings)
    from Solverz.sym_algebra.functions import Diag

    A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]), dim=2, sparse=True)
    x = Param('x', [0.5, 0.5])
    y = Param('y', [0.3, 0.7])
    term = Diag(x) * A * Diag(y)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _emit_l2_fallback_warnings('f', 'x', [term])

    l2 = [m for m in w
          if issubclass(m.category, UserWarning)
          and 'Mutable Jacobian block' in str(m.message)]
    assert len(l2) >= 1, (
        f'expected a Layer 2 Jacobian fallback warning, '
        f'captured: {[str(m.message) for m in w]}')
    msg = str(l2[0].message)
    assert 'eqn' in msg.lower() and "'f'" in msg, (
        f'L2 warning should name the equation, got: {msg}')
    assert 'var' in msg.lower() and "'x'" in msg, (
        f'L2 warning should name the variable, got: {msg}')
    assert 'element-wise' in msg.lower() or "mul" in msg.lower(), (
        f'L2 warning should diagnose the element-wise Mul mistake, '
        f'got: {msg}')
    assert 'mat_mul' in msg.lower(), (
        f'L2 warning should suggest rewriting as Mat_Mul, got: {msg}')


def test_mutable_jac_l2_classifier_distinguishes_shapes():
    """The Layer 2 classifier must produce *different* diagnostic text
    for structurally different fallback shapes — biscale/single-Diag/
    no-Diag Mat_Mul should not all degrade to the same message.

    Regression guard for the previous classifier, which fell through
    to "multiple Diag nodes" for any term containing ``Diag``,
    misdiagnosing single-Diag and bare-matrix cases.
    """
    from Solverz.code_printer.python.module.module_printer import (
        _classify_l2_fallback_reason)
    from Solverz.sym_algebra.functions import Diag
    from Solverz.sym_algebra.symbols import iVar

    A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]), dim=2, sparse=True)
    B = Param('B', np.array([[1.0, 0.0], [0.0, 1.0]]), dim=2, sparse=True)
    u = iVar('u')
    v = iVar('v')

    # 1) Mat_Mul of two matrices, no Diag at either end.
    no_diag_term = Mat_Mul(A, B)
    r_no_diag, s_no_diag = _classify_l2_fallback_reason(no_diag_term)
    assert 'without' in r_no_diag.lower() and 'diag' in r_no_diag.lower(), (
        f'no-Diag case should flag missing Diag wrapper, got: {r_no_diag}')
    assert 'wrap' in s_no_diag.lower() or 'split' in s_no_diag.lower(), (
        f'no-Diag suggestion should propose wrapping or splitting, '
        f'got: {s_no_diag}')

    # 2) Single-Diag Mat_Mul where the matrix factor is a sum (the
    # analyzer rejects ``A + B`` as a constant sparse matrix).
    single_diag_term = Mat_Mul(Diag(u), A + B)
    r_single, s_single = _classify_l2_fallback_reason(single_diag_term)
    assert ('single-diag' in r_single.lower()
            or "diag(v) @ m" in r_single.lower()
            or "m @ diag(v)" in r_single.lower()), (
        f'single-Diag case should be identified explicitly, got: {r_single}')

    # 3) Biscale shape but the middle is non-materialisable.
    biscale_term = Mat_Mul(Diag(u), A + B, Diag(v))
    r_bi, s_bi = _classify_l2_fallback_reason(biscale_term)
    assert 'biscale' in r_bi.lower(), (
        f'biscale case should be identified explicitly, got: {r_bi}')

    # 4) Bare ``Para`` — no Diag, no Mat_Mul. Extract the Para symbol
    # from a Mat_Mul (the analyzer's Param→Para conversion is what the
    # classifier actually sees in production).
    A_para = Mat_Mul(A, iVar('z')).args[0]
    r_bare, s_bare = _classify_l2_fallback_reason(A_para)
    assert 'bare' in r_bare.lower() and 'param' in r_bare.lower(), (
        f'bare Para case should be identified explicitly, got: {r_bare}')

    # 5) Two-argument ``Mat_Mul(Diag(u), Diag(v))`` — degenerate biscale.
    # Should NOT report "biscale shape Diag(u) @ M @ Diag(v) where M ..."
    # because there is no middle factor M.
    diag_diag_term = Mat_Mul(Diag(u), Diag(v))
    r_dd, s_dd = _classify_l2_fallback_reason(diag_diag_term)
    assert 'two' in r_dd.lower() or 'no middle' in r_dd.lower(), (
        f'2-arg Mat_Mul(Diag, Diag) should NOT be reported as a "biscale '
        f"with bad middle M\" (there is no middle), got: {r_dd}")
    assert 'diag(u * v)' in s_dd.lower() or 'compose' in s_dd.lower(), (
        f"suggestion should propose Diag(u*v), got: {s_dd}")

    # All five reasons should be pairwise distinct — this is the
    # regression guard.
    reasons = {r_no_diag, r_single, r_bi, r_bare, r_dd}
    assert len(reasons) == 5, (
        f'each fallback shape should produce a distinct reason, '
        f'got: {reasons}')


def test_matmul_transpose_warns_with_predeclare_suggestion():
    """``Mat_Mul(transpose(A), x)`` falls back because the @njit fast
    path reads A's CSC flat fields directly and can't transpose at
    runtime. The Layer 1 classifier must surface this and suggest
    predeclaring ``A_T`` as a separate sparse Param.

    Drives the classifier directly because ``transpose`` doesn't yet
    have a Python printer (``PrintMethodNotImplementedError`` from
    ``PythonCodePrinter``) so end-to-end ``render_modules`` can't
    materialise the equation.
    """
    from Solverz.code_printer.python.module.module_printer import (
        _classify_l1_fallback_reason)
    from Solverz.sym_algebra.functions import transpose
    from Solverz.sym_algebra.symbols import iVar

    m = Model()
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    spf, y0 = m.create_instance()
    PARAM = spf.PARAM

    mat = transpose(m.A)
    op = iVar('x')
    reason, expr_str, suggestion = _classify_l1_fallback_reason(
        mat, op, PARAM)
    assert 'transpose' in reason.lower(), (
        f'reason should mention transpose, got: {reason}')
    assert 'A' in expr_str, (
        f'expression should reference matrix A, got: {expr_str}')
    assert ('predeclare' in suggestion.lower()
            or 'A_T' in suggestion), (
        f'suggestion should propose predeclaring A.T as a separate '
        f'Param, got: {suggestion}')


def test_matmul_scalar_param_coeff_warns_with_factor_suggestion():
    """``Mat_Mul(c*A, x)`` where ``c`` is a *Solverz scalar Param* (not
    a numeric literal) must hit the scalar-mul branch, not fall through
    to the generic message.

    Regression guard for codex review finding: the previous classifier
    only matched ``isinstance(a, Number)`` for the coefficient, so
    ``Param('c', 2.0) * A`` silently degraded to "matrix operand is not
    a bare sparse Param" with no actionable suggestion.
    """
    from Solverz.code_printer.python.module.module_printer import (
        print_F, print_sub_inner_F)

    m = Model()
    m.x = Var('x', [0.0, 0.0])
    m.b = Param('b', [1.0, 2.0])
    m.c = Param('c', 2.0)
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.c * m.A, m.x) - m.b)

    spf, y0 = m.create_instance()
    spf.FormJac(y0)
    _, precompute_info = print_sub_inner_F(spf.EQNs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        print_F('AE', spf.var_address, spf.PARAM, nstep=0,
                precompute_info=precompute_info)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert ('scalar' in msg.lower() and 'multiple' in msg.lower()), (
        f'warning should diagnose scalar*matrix shape, got: {msg}')
    assert 'A' in msg and 'c' in msg, (
        f'warning should name both factors, got: {msg}')
    assert 'factor' in msg.lower() and 'outside' in msg.lower(), (
        f'warning should suggest factoring scalar outside Mat_Mul, '
        f'got: {msg}')


def test_matmul_elementwise_matrix_product_warns_with_nesting_suggestion():
    """``Mat_Mul(A*B, x)`` uses Python ``*`` (element-wise) between two
    sparse matrix Params instead of ``Mat_Mul`` (matrix product). The
    Layer 1 classifier must distinguish this from "scalar * matrix" and
    suggest either nesting Mat_Muls or distributing.

    Verifies stacklevel points at user code: the warning's filename
    should NOT be inside ``Solverz/code_printer/`` (which would mean
    stacklevel was off and the warning is blaming library internals).
    """
    import tempfile
    from Solverz.code_printer.python.module.module_generator import (
        render_modules)

    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[2.0, 1.0], [1.0, 3.0]], dim=2, sparse=True)
    m.B = Param('B', [[1.0, 0.0], [0.0, 1.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.A * m.B, m.x) - m.b)
    spf, y0 = m.create_instance()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with tempfile.TemporaryDirectory() as d:
            render_modules(spf, y0, name='probe_em', directory=d)

    fb = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mat_Mul placeholder' in str(x.message)
          and 'falls back' in str(x.message)]
    assert len(fb) >= 1, (
        f'expected a Mat_Mul fallback warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(fb[0].message)
    assert 'element-wise' in msg.lower(), (
        f'warning should diagnose element-wise Mul, got: {msg}')
    assert 'A * B' in msg, (
        f'warning should show the element-wise product, got: {msg}')
    # stacklevel guard: the warning's reported filename must NOT be
    # inside the code_printer subtree (which would mean stacklevel was
    # too small and the warning blamed library internals).
    assert 'code_printer' not in fb[0].filename, (
        f'stacklevel should point at user code, not at '
        f'Solverz/code_printer/. Got: {fb[0].filename}')


def test_mutable_jac_l2_fallback_via_analyzer_integration():
    """End-to-end coverage of the Layer 2 pipeline:

    1. ``analyze_mutable_mat_expr`` correctly identifies a
       fallback-shape term and surfaces it via
       ``mapping.has_fallback`` / ``mapping.fallback_pieces``.
    2. The same fallback pieces feed cleanly into
       ``_emit_l2_fallback_warnings``, producing a UserWarning with
       the L2 classifier's structural diagnosis.

    Together these two assertions guarantee the analyzer→emitter
    contract holds. The only remaining integration step is the 3-line
    ``if mapping.has_fallback: emit(...)`` glue inside
    ``print_inner_J`` (line 350-353).
    """
    from Solverz.code_printer.python.module.module_printer import (
        _emit_l2_fallback_warnings)
    from Solverz.code_printer.python.module.mutable_mat_analyzer import (
        analyze_mutable_mat_expr)
    from Solverz.sym_algebra.functions import Diag
    from Solverz.sym_algebra.symbols import iVar

    # Build a real PARAM dict via create_instance() so the analyzer
    # has a registered ``A`` to inspect.
    m = Model()
    m.A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]),
                dim=2, sparse=True)
    spf, _ = m.create_instance()
    PARAM = spf.PARAM

    # Element-wise Mul of Diag(u) * A * Diag(v) — falls through
    # ``handle()`` to ``fallback_pieces`` because it isn't Mat_Mul.
    u = iVar('u')
    v = iVar('v')
    expr = Diag(u) * m.A * Diag(v)

    # Stage 1: analyzer correctly classifies as fallback.
    row = np.array([0, 1])
    col = np.array([0, 1])
    mapping = analyze_mutable_mat_expr(expr, row, col, PARAM, eqn_size=2)
    assert mapping.has_fallback, (
        'analyzer must mark this expression as has_fallback=True')
    assert len(mapping.fallback_pieces) == 1, (
        f'expected exactly one fallback piece, got '
        f'{mapping.fallback_pieces}')

    # Stage 2: emitter formats the warning correctly from the
    # analyzer's output.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _emit_l2_fallback_warnings('myeqn', 'myvar',
                                   mapping.fallback_pieces)
    l2 = [x for x in w
          if issubclass(x.category, UserWarning)
          and 'Mutable Jacobian block' in str(x.message)]
    assert len(l2) == 1, (
        f'expected exactly one L2 warning, captured: '
        f'{[str(x.message) for x in w]}')
    msg = str(l2[0].message)
    assert "'myeqn'" in msg and "'myvar'" in msg, (
        f'warning should name the equation and variable, got: {msg}')
    assert 'element-wise' in msg.lower(), (
        f'warning should diagnose element-wise Mul, got: {msg}')
    assert 'Mat_Mul' in msg, (
        f'warning should suggest rewriting as Mat_Mul, got: {msg}')


def test_no_matmul_fallback_warnings_in_inline_mode():
    """Layer 1 / Layer 2 fallback warnings fire only from the module
    printer — inline mode (``made_numerical``) has no fast/fallback
    split, so none of the ``Mat_Mul placeholder`` / ``Mutable Jacobian
    block`` warnings should appear.

    ``_warn_dense_matmul_params`` (from ``FormJac``) still fires in
    both modes — that's expected and not filtered here.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [3.0, 4.0])
    m.A = Param('A', [[1.0, 0.5], [0.5, 1.0]], dim=2, sparse=True)
    # Same model that triggers L1 warnings in module mode
    m.eqn = Eqn('f', Mat_Mul(-m.A, m.x) - m.b)

    spf, y0 = m.create_instance()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        made_numerical(spf, y0, sparse=True)

    l1_l2 = [x for x in w
             if issubclass(x.category, UserWarning)
             and ('Mat_Mul placeholder' in str(x.message)
                  or 'Mutable Jacobian block' in str(x.message))]
    assert len(l1_l2) == 0, (
        f'L1/L2 fallback warnings must not fire in inline mode, '
        f'captured: {[str(x.message) for x in l1_l2]}')
