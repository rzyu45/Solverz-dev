"""SuperLU wrapper of ``lu_decomposition``: the factor copies are read on demand
(issue #159), and the column ordering of a pattern is computed once and reused
(issue #182)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from Solverz import Opt, nr_method
from Solverz.num_api.num_eqn import nAE
from Solverz.solvers.klu_backend import KLUCache
from Solverz.solvers.laesolver import SuperLUOrdering, linsolver, lu_decomposition, sp_decomposition


def _system(n, seed):
    rng = np.random.default_rng(seed)
    A = (sp.random(n, n, density=0.05, format="csc", random_state=rng) + sp.eye(n) * 5).tocsc()
    return A, rng.standard_normal(n)


def _factorization_holds(dec, A):
    """``Pr A Pc = L U`` with the permutations the decomposition reports."""
    n = A.shape[0]
    P = sp.csc_array((np.ones(n), (dec.perm_r, np.arange(n))))
    Q = sp.csc_array((np.ones(n), (np.arange(n), dec.perm_c)))
    return abs((dec.L @ dec.U) - (P @ A @ Q)).max() < 1e-10


def test_sp_decomposition_reads_the_factors_lazily():
    A, b = _system(200, 0)
    dec = lu_decomposition(A, backend="superlu")
    assert isinstance(dec, sp_decomposition)
    assert 'L' not in dec.__dict__ and 'U' not in dec.__dict__ and 'nnz' not in dec.__dict__
    assert np.allclose(A @ dec.solve(b), b, atol=1e-10)
    assert 'L' not in dec.__dict__                       # solving does not build the copies
    ref = splu(A)
    assert dec.nnz == ref.nnz
    assert np.array_equal(dec.perm_r, ref.perm_r) and np.array_equal(dec.perm_c, ref.perm_c)
    # the factors are still available, and L U reproduces the permuted matrix
    assert _factorization_holds(dec, A)
    assert 'L' in dec.__dict__ and dec.L is dec.L          # read once, then kept


def test_superlu_reuses_the_column_ordering_of_a_pattern():
    A, b = _system(300, 1)
    cache = KLUCache()
    dec1 = lu_decomposition(A, backend="superlu", cache=cache)
    ordering = cache.superlu
    assert isinstance(ordering, SuperLUOrdering)
    assert np.array_equal(dec1.solve(b), splu(A).solve(b))      # the first factorization is scipy's own
    A2 = A.copy()                                                # new values on the same pattern
    A2.data = A2.data * np.random.default_rng(2).uniform(0.5, 2.0, A2.nnz)
    dec2 = lu_decomposition(A2, backend="superlu", cache=cache)
    assert cache.superlu is ordering                             # reused, not computed again
    ref = splu(A2)
    assert abs(ordering.permute(A2) - A2[:, np.argsort(ref.perm_c)]).max() == 0   # A Pc
    assert np.array_equal(dec2.perm_c, ref.perm_c)               # the column order COLAMD chooses
    assert _factorization_holds(dec2, A2)                        # reported in terms of A, not A Pc
    x = dec2.solve(b)
    assert np.allclose(x, ref.solve(b), rtol=0, atol=1e-12 * np.abs(x).max())
    B = np.random.default_rng(3).standard_normal((300, 3))
    assert np.allclose(A2 @ dec2.solve(B), B, atol=1e-10)        # several right-hand sides


def test_a_pattern_change_computes_the_superlu_ordering_again():
    A, b = _system(300, 4)
    cache = KLUCache()
    lu_decomposition(A, backend="superlu", cache=cache)
    ordering = cache.superlu
    B, _ = _system(300, 5)                                       # another pattern of the same shape
    x = lu_decomposition(B, backend="superlu", cache=cache).solve(b)
    assert cache.superlu is not ordering and cache.superlu.matches(B)
    assert np.allclose(B @ x, b, atol=1e-10)


def test_the_superlu_pattern_check_reads_the_row_indices():
    # Every row index shifted by one, cyclically: each column keeps its count,
    # so indptr is the same, and the pattern is another one.
    A, b = _system(300, 6)
    C = sp.csc_matrix((A.data.copy(), (A.indices + 1) % 300, A.indptr.copy()), shape=A.shape)
    C.sort_indices()
    assert np.array_equal(C.indptr, A.indptr) and not np.array_equal(C.indices, A.indices)
    cache = KLUCache()
    lu_decomposition(A, backend="superlu", cache=cache)
    ordering = cache.superlu
    x = lu_decomposition(C, backend="superlu", cache=cache).solve(b)
    assert cache.superlu is not ordering and cache.superlu.matches(C)
    assert np.allclose(C @ x, b, atol=1e-10)


def test_nr_method_keeps_the_superlu_ordering_on_the_nae():
    A, b = _system(300, 7)
    A = sp.csc_array(A)
    ae = nAE(lambda y, p: A @ y - b, lambda y, p: A, {})
    with linsolver("superlu"):
        sol1 = nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10))
        ordering = ae._klu_cache.superlu
        sol2 = nr_method(ae, np.ones(300), Opt(ite_tol=1e-10))
    assert isinstance(ordering, SuperLUOrdering)
    assert ae._klu_cache.superlu is ordering                     # the second solve reused it
    assert ae._klu_cache.symbolic is None                        # KLU did not run
    assert np.allclose(A @ sol1.y, b, atol=1e-8) and np.allclose(A @ sol2.y, b, atol=1e-8)
