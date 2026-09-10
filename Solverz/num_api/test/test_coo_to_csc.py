"""``CooToCsc`` reproduces ``coo_array(...).tocsc()`` for a fixed pattern (issue #160)."""
import numpy as np
import scipy.sparse as sps

from Solverz.num_api.custom_function import CooToCsc


def _check(row, col, shape, data):
    ref = sps.coo_array((data, (row, col)), shape=shape).tocsc()
    conv = CooToCsc(row, col, shape)
    out = conv(data)
    assert out.shape == ref.shape and out.nnz == ref.nnz
    assert np.array_equal(out.indptr, ref.indptr) and np.array_equal(out.indices, ref.indices)
    assert np.allclose(out.data, ref.data)
    assert out.has_canonical_format
    return conv


def test_pattern_without_duplicates_is_a_pure_gather():
    rng = np.random.default_rng(0)
    A = sps.random(60, 45, density=0.08, format='coo', random_state=rng)
    perm = rng.permutation(A.nnz)                     # generation order is not the CSC order
    row, col = A.row[perm], A.col[perm]
    conv = _check(row, col, A.shape, rng.standard_normal(row.size))
    assert not conv.has_duplicates and conv.order is not None
    _check(row, col, A.shape, rng.standard_normal(row.size))   # fresh values, same pattern


def test_duplicate_entries_are_summed_like_scipy():
    rng = np.random.default_rng(1)
    row = rng.integers(0, 30, 400)
    col = rng.integers(0, 25, 400)
    conv = _check(row, col, (30, 25), rng.standard_normal(400))
    assert conv.has_duplicates
    _check(row, col, (30, 25), rng.standard_normal(400) + 1j * rng.standard_normal(400))


def test_empty_pattern():
    conv = CooToCsc(np.zeros(0, int), np.zeros(0, int), (4, 3))
    out = conv(np.zeros(0))
    assert out.shape == (4, 3) and out.nnz == 0
