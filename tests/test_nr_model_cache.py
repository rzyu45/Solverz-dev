"""``nr_method`` keeps the KLU symbolic ordering on the ``nAE`` it solves, so
a repeated solve of the same equations does not pay the symbolic analysis
and the row matching again, and the per-step implicit solvers hand their
throwaway ``nAE`` the ordering of the model they integrate."""
import numpy as np
import scipy.sparse as sps

from Solverz import nr_method, Opt, backward_euler, implicit_trapezoid
from Solverz.num_api.num_eqn import nAE, nDAE
from Solverz.solvers.klu_backend import KLU_AVAILABLE
from Solverz.solvers.laesolver import resolve_backend


def _scrambled(n, seed):
    """A diagonally dominant matrix with its rows permuted, so the structural
    diagonal is empty, as for a LoopEqn Jacobian."""
    rng = np.random.default_rng(seed)
    A = (sps.random(n, n, density=3.0 / n, format='csc', random_state=rng) + sps.eye(n) * 8).tocsc()
    return sps.csc_array(A[rng.permutation(n)]), rng


def _linear_ae(A, b):
    return nAE(lambda y, p: A @ y - b, lambda y, p: A, {})


def test_nr_method_keeps_the_ordering_on_the_nae():
    A, rng = _scrambled(400, 0)
    b = rng.standard_normal(A.shape[0])
    ae = _linear_ae(A, b)
    assert not hasattr(ae, '_klu_cache')
    sol1 = nr_method(ae, np.zeros(A.shape[0]), Opt(ite_tol=1e-10))
    assert np.allclose(A @ sol1.y, b, atol=1e-8)
    cache = getattr(ae, '_klu_cache', None)
    assert cache is not None                       # attached by the first call
    sym = cache.symbolic
    sol2 = nr_method(ae, np.ones(A.shape[0]), Opt(ite_tol=1e-10))
    assert np.allclose(A @ sol2.y, b, atol=1e-8)
    assert ae._klu_cache is cache                  # the same holder, not a new one
    if KLU_AVAILABLE and resolve_backend() == 'klu':
        assert sym is not None                     # the first call analysed
        assert cache.symbolic is sym               # the second call reused it


def test_a_pattern_change_on_the_same_nae_is_detected():
    A, rng = _scrambled(300, 1)
    B, _ = _scrambled(300, 2)
    b = rng.standard_normal(300)
    box = {'A': A}
    ae = nAE(lambda y, p: box['A'] @ y - b, lambda y, p: box['A'], {})
    nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10))
    sym = ae._klu_cache.symbolic
    box['A'] = B                                   # a different sparsity pattern
    sol = nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10))
    assert np.allclose(B @ sol.y, b, atol=1e-8)
    if KLU_AVAILABLE and resolve_backend() == 'klu':
        assert ae._klu_cache.symbolic is not sym   # re-analysed, not reused


def test_per_step_solvers_share_the_model_ordering():
    # y' = -y, z = y: a stiff-free linear DAE whose iteration matrix keeps one pattern
    n = 6
    M = sps.diags([1.0] * (n // 2) + [0.0] * (n // 2), format='csc')
    K = sps.csc_array(sps.diags([-1.0] * n) + sps.eye(n, k=n // 2) * 0.5)

    def F(t, y, p):
        return K @ y

    def J(t, y, p):
        return K

    for solver in (backward_euler, implicit_trapezoid):
        dae = nDAE(M, F, J, {})
        sol = solver(dae, [0, 1], np.ones(n), Opt(step_size=0.1, fix_h=True))
        cache = getattr(dae, '_klu_cache', None)
        assert cache is not None                   # every step's nAE used the dae's holder
        assert np.all(np.isfinite(sol.Y))
