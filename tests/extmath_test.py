# encoding: utf-8

from __future__ import division, print_function

import numpy as np
import pytest as pt
from mpnum import _testing as mptest
from mpnum import factory, utils
from mpnum.utils import extmath as em
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)
from scipy.linalg import block_diag
from six.moves import range


def test_block_diag_simple(rgen):
    rows = (4, 7)
    cols = (3, 6)
    summands = [factory._zrandn((rows[i], cols[i]), randstate=rgen)
                for i in range(len(rows))]
    blockdiag_sum = utils.block_diag(summands)
    blockdiag_sum_scipy = block_diag(*summands)
    assert_array_almost_equal(blockdiag_sum, blockdiag_sum_scipy)


def test_block_diag(rgen):
    leftmargin = 3
    rightmargin = 5
    rows = (4, 7)
    cols = (3, 6)
    nr_blocks = len(rows)
    nr_summands = 3
    leftvecs = factory._zrandn((nr_blocks, nr_summands, leftmargin),
                               randstate=rgen)
    middlematrices = [factory._zrandn((nr_summands, rows[i], cols[i]),
                                      randstate=rgen)
                      for i in range(nr_blocks)]
    rightvecs = factory._zrandn((nr_blocks, nr_summands, rightmargin),
                                randstate=rgen)
    blockdiag_summands = []
    for i in range(nr_blocks):
        summand = np.zeros(
            (leftmargin, rows[i], cols[i], rightmargin), dtype=complex)
        for j in range(nr_summands):
            summand += np.outer(
                np.outer(leftvecs[i, j, :], middlematrices[i][j, :, :]),
                rightvecs[i, j, :]).reshape(summand.shape)
        blockdiag_summands.append(summand)
    blockdiag_sum = utils.block_diag(blockdiag_summands, axes=(1, 2))
    blockdiag_sum_explicit = np.zeros(
        (leftmargin, sum(rows), sum(cols), rightmargin), dtype=complex)

    for i in range(nr_blocks):
        for j in range(nr_summands):
            summands = [middlematrices[i2][j]
                        if i2 == i else np.zeros_like(middlematrices[i2][j])
                        for i2 in range(nr_blocks)]
            middle = block_diag(*summands)
            blockdiag_sum_explicit += \
                np.outer(np.outer(leftvecs[i][j], middle), rightvecs[i][j]) \
                .reshape(blockdiag_sum_explicit.shape)

    assert_array_almost_equal(blockdiag_sum, blockdiag_sum_explicit)


TESTARGS_MATRIXDIMS = [(50, 50), (100, 50), (50, 75)]
TESTARGS_RANKS = [1, 10, 'fullrank']


@pt.mark.parametrize('rows, cols', TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.MP_TEST_DTYPES)
@pt.mark.parametrize('piter_normalizer', [None, 'qr', 'lu', 'auto'])
def test_approximate_range_finder(rows, cols, rank, dtype, piter_normalizer, rgen):
    # only guaranteed to work for low-rank matrices
    if rank == 'fullrank':
        return

    rf_size = rank + 10
    assert min(rows, cols) > rf_size

    A = mptest.random_lowrank(rows, cols, rank, randstate=rgen, dtype=dtype)
    A /= np.linalg.norm(A, ord='fro')
    Q = em.approx_range_finder(A, rf_size, 7, randstate=rgen,
                               piter_normalizer=piter_normalizer)

    assert Q.shape == (rows, rf_size)
    normdist = np.linalg.norm(A - Q @ (Q.T.conj() @ A), ord='fro')
    assert normdist < 1e-7


@pt.mark.parametrize('rows, cols', TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.MP_TEST_DTYPES)
@pt.mark.parametrize('transpose', [False, True, 'auto'])
@pt.mark.parametrize('n_iter, target_gen', [(7, mptest.random_lowrank),
                                            (20, mptest.random_fullrank)])
def test_randomized_svd(rows, cols, rank, dtype, transpose, n_iter, target_gen,
                        rgen):
    rank = min(rows, cols) - 2 if rank == 'fullrank' else rank
    A = target_gen(rows, cols, rank=rank, randstate=rgen, dtype=dtype)

    U_ref, s_ref, V_ref = utils.truncated_svd(A, k=rank)
    U, s, V = em.randomized_svd(A, rank, transpose=transpose, randstate=rgen,
                                n_iter=n_iter)

    error_U = np.abs(U.conj().T.dot(U_ref)) - np.eye(rank)
    assert_allclose(np.linalg.norm(error_U), 0, atol=1e-3)
    error_V = np.abs(V.dot(V_ref.conj().T)) - np.eye(rank)
    assert_allclose(np.linalg.norm(error_V), 0, atol=1e-3)
    assert_allclose(s.ravel() - s_ref, 0, atol=1e-3)
    # Check that singular values are returned in descending order
    assert_array_equal(s, np.sort(s)[::-1])
