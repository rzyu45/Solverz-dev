import os
import contextvars
import functools
from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, linalg as sla

# from scikits import umfpack
# umfpack is slow compared with superlu on Apple M4 with MACOS 15.6.1, scikits-umfpack 0.4.2, and suite-sparse 7.10.3.
# Also, it was found that umfpack was not accurate enough, causing non-convergence issues.
#
# KLU (SuiteSparse) is the default backend when libklu is installed: see
# klu_backend.py. Unlike umfpack it is BTF+AMD-ordered for circuit/network
# matrices and is roughly 2x faster than superlu on the IEGS Jacobians. superlu
# is the fallback (used when libklu is absent, the matrix is complex/dense, or
# the linsolver is set to 'superlu'). KLU is fastest when its symbolic ordering
# is reused across steps (pass a KLUCache via lu_decomposition's cache arg); the
# same cache keeps superlu's column ordering, so superlu computes COLAMD once
# per pattern instead of at every factorization (issue #182).

from Solverz.solvers.klu_backend import KLU_AVAILABLE, klu_decomposition, KLUCache  # noqa: F401

splu = sla.splu

# --------------------------------------------------------------------------- #
# Global linear-solver selection.
#
# The backend is read from a ContextVar so it can be set once for a whole script
# (set_linsolver), scoped to a block (the ``linsolver`` context manager), or
# overridden per-solve via Opt(linsolver=...). The default is 'klu', overridable
# at import via the SOLVERZ_LINSOLVER environment variable. Resolution always
# degrades to 'superlu' when libklu is unavailable, so the default is safe on any
# machine. The ContextVar (not a plain global) keeps this thread- and async-safe
# and correctly scoped for nested solves.
# --------------------------------------------------------------------------- #
_LINSOLVER = contextvars.ContextVar(
    'solverz_linsolver',
    default=os.environ.get('SOLVERZ_LINSOLVER', 'klu').lower())


def _check_name(name):
    name = str(name).lower()
    if name not in ('klu', 'superlu'):
        raise ValueError(f"linsolver must be 'klu' or 'superlu', got {name!r}")
    return name


def set_linsolver(name):
    """Set the default linear-solver backend for the rest of the script."""
    _LINSOLVER.set(_check_name(name))


def get_linsolver():
    """Return the currently selected backend name ('klu' or 'superlu')."""
    return _LINSOLVER.get()


class linsolver:
    """Context manager scoping the linear-solver backend for a block::

        with linsolver('superlu'):
            sol = Rodas(mdl, tspan, y0)
    """

    def __init__(self, name):
        self.name = _check_name(name)
        self._token = None

    def __enter__(self):
        self._token = _LINSOLVER.set(self.name)
        return self

    def __exit__(self, *exc):
        _LINSOLVER.reset(self._token)
        return False


def resolve_backend(backend=None):
    """Resolve an explicit backend (or None -> the global selection) to the
    effective backend, degrading 'klu' to 'superlu' when libklu is absent."""
    b = _check_name(backend) if backend is not None else _LINSOLVER.get()
    if b == 'klu' and not KLU_AVAILABLE:
        return 'superlu'
    return b


def model_cache(obj):
    """Return a KLUCache attached to ``obj`` (a model/solver object), creating
    it on first use. Lets a solver reuse the KLU symbolic ordering, or the
    superlu column ordering, across its factorizations (the iteration-matrix
    pattern is fixed for a model) with a single ``cache=model_cache(dae)`` at
    the factorization site, rather than threading a cache object through the
    loop."""
    c = getattr(obj, '_klu_cache', None)
    if c is None:
        c = KLUCache()
        try:
            obj._klu_cache = c
        except (AttributeError, TypeError):
            pass
    return c


def solve(A, b, backend=None, cache=None):
    """Single linear solve. ``cache`` (a KLUCache) reuses the KLU symbolic
    ordering, or the superlu column ordering, across calls of the same
    pattern, e.g. across Newton iterations where the Jacobian structure is
    fixed. A complex system goes to ``spsolve`` without the cache."""
    if isinstance(A, (csc_array, csc_matrix, csr_array, csr_matrix)):
        real = not np.iscomplexobj(A.data) and not np.iscomplexobj(b)
        if resolve_backend(backend) == 'klu' and real:
            try:
                sym = cache.symbolic if cache is not None else None
                dec = klu_decomposition(A, symbolic=sym)
                if cache is not None:
                    cache.symbolic = dec.symbolic
                return dec.solve(b)
            except (NotImplementedError, OverflowError, RuntimeError):
                pass
        if cache is not None and real:
            return sp_decomposition(A, cache=cache).solve(b)
        return sla.spsolve(A, b)
    else:
        return np.linalg.solve(A, b)


def lu_decomposition(A: Union[np.ndarray, csc_array, csc_matrix],
                     backend: str = None,
                     cache: 'KLUCache' = None):
    """Factorize ``A`` and return an object exposing ``.solve(b)``.

    backend : None (default) | 'klu' | 'superlu'.
        None defers to the global selection (see set_linsolver / linsolver).
        'klu' silently falls back to superlu when libklu is unavailable or
        ``A`` is complex/dense.
    cache : KLUCache, optional
        Holds the reusable KLU symbolic ordering, or the superlu column
        ordering, across calls of the same sparsity pattern. Owned by the
        model and threaded in by the solver.
    """
    if isinstance(A, np.ndarray):
        return dense_decomposition(A)
    if resolve_backend(backend) == 'klu' and not np.iscomplexobj(A.data):
        sym = cache.symbolic if cache is not None else None
        try:
            dec = klu_decomposition(A, symbolic=sym)
        except (NotImplementedError, OverflowError, RuntimeError):
            return sp_decomposition(A, cache=cache)
        if cache is not None:
            cache.symbolic = dec.symbolic
        return dec
    return sp_decomposition(A, cache=cache)


class dense_decomposition:
    def __init__(self,
                 A: np.ndarray):
        self.A = A

    def solve(self, b):
        return solve(self.A, b)


class SuperLUOrdering:
    """The column ordering of a SuperLU factorization, kept so that the next
    factorization of the same sparsity pattern skips COLAMD (issue #182).

    SciPy's factors satisfy ``Pr A Pc = L U`` with ``Pc[j, perm_c[j]] = 1``,
    so ``A Pc = A[:, argsort(perm_c)]``; indexing with ``perm_c`` itself
    scrambles the columns and the fill explodes. The ordering depends only on
    the pattern, which a model's iteration matrix keeps from one step to the
    next, so once it is known :class:`sp_decomposition` factorizes ``A Pc``
    with ``permc_spec='NATURAL'``. SuperLU then eliminates the same columns
    in the same order, and partial pivoting still runs at every
    factorization.

    SuperLU prefers the diagonal entry of a column when it ties with the
    largest entry, and it finds the diagonal through the column ordering it
    computed itself. With the ordering switched off it takes entry ``(j, j)``
    of ``A Pc`` for the diagonal of column ``j``, so an exact tie can be
    broken on another row than under COLAMD. Both are partial-pivoting
    factorizations of ``A`` in the same column order, and their solutions
    agree to rounding.

    The pattern this ordering belongs to is kept as ``shape``, ``indptr`` and
    ``indices`` of the canonical matrix, and ``gather``, ``indptr_q`` and
    ``indices_q`` build ``A Pc`` from ``A.data`` without a sort.
    """

    __slots__ = ("shape", "indptr", "indices", "perm_c", "gather", "indptr_q", "indices_q")

    def __init__(self, A, perm_c):
        n = A.shape[1]
        self.shape = A.shape
        self.indptr = A.indptr.copy()
        self.indices = A.indices.copy()
        self.perm_c = np.asarray(perm_c, dtype=np.intp)
        q = np.argsort(self.perm_c)
        counts = np.diff(self.indptr)[q]
        self.indptr_q = np.zeros(n + 1, dtype=np.intc)
        np.cumsum(counts, out=self.indptr_q[1:])
        # Column j of A Pc is column q[j] of A, the slice indptr[q[j]]:indptr[q[j] + 1].
        self.gather = (np.repeat(self.indptr[q] - self.indptr_q[:-1], counts)
                       + np.arange(self.indptr_q[-1], dtype=np.intp))
        self.indices_q = np.ascontiguousarray(self.indices[self.gather], dtype=np.intc)

    def matches(self, A):
        """Whether the canonical matrix ``A`` has the pattern of this ordering."""
        return (A.shape == self.shape
                and np.array_equal(A.indptr, self.indptr)
                and np.array_equal(A.indices, self.indices))

    def permute(self, A):
        """``A Pc`` for a canonical ``A`` of this pattern."""
        Aq = csc_array((A.data[self.gather], self.indices_q, self.indptr_q), shape=self.shape)
        Aq.has_canonical_format = True      # the columns of a canonical matrix, reordered
        return Aq


class sp_decomposition:
    """SuperLU factorization behind the ``.solve(b)`` interface of
    :class:`klu_decomposition`.

    ``L``, ``U`` and ``nnz`` are read from the SuperLU object on first
    access and kept, since each of ``L`` and ``U`` builds a scipy sparse
    matrix from the factor, a copy of the factor's size that most callers
    never read (issue #159); ``perm_r`` and ``perm_c`` are cheap and eager.

    With a ``cache``, the first factorization of a pattern stores its column
    ordering in ``cache.superlu``, and every later factorization of that
    pattern factorizes ``A Pc`` with SuperLU's ordering switched off, see
    :class:`SuperLUOrdering` (issue #182). ``splu`` is then the factorization
    of ``A Pc``, while ``solve``, ``perm_r``, ``perm_c``, ``L`` and ``U``
    refer to ``A`` as they do without the cache: ``Pr A Pc = L U``.
    """

    def __init__(self,
                 A: Union[(csc_array, csc_matrix)],
                 cache: 'KLUCache' = None):
        ordering = None
        if cache is not None:
            A = A.tocsc()
            A.sum_duplicates()      # as splu does, so that the pattern compares canonically
            ordering = cache.superlu
            if ordering is not None and not ordering.matches(A):
                ordering = None
        if ordering is None:
            self.splu = splu(A)
            self.perm_c = self.splu.perm_c
            self._perm_c = None
            if cache is not None:
                cache.superlu = SuperLUOrdering(A, self.perm_c)
        else:
            self.splu = splu(ordering.permute(A), permc_spec='NATURAL')
            # Column i of A is column perm_c[i] of A Pc, which SuperLU puts at
            # its own perm_c; with the ordering off that is the identity.
            self.perm_c = self.splu.perm_c[ordering.perm_c]
            self._perm_c = ordering.perm_c
        self.perm_r = self.splu.perm_r

    @functools.cached_property
    def L(self):
        return self.splu.L

    @functools.cached_property
    def U(self):
        return self.splu.U

    @functools.cached_property
    def nnz(self):
        return self.splu.nnz

    def solve(self, b):
        z = self.splu.solve(b)
        # (A Pc) z = b gives x = Pc z, that is x[i] = z[perm_c[i]].
        return z if self._perm_c is None else z[self._perm_c]
