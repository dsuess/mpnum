# encoding: utf-8
"""Additional math functions for dealing with dense arrays"""

from __future__ import division, print_function

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import aslinearoperator
from six.moves import range, zip

__all__ = ['block_diag', 'matdot', 'mkron', 'partial_trace',
           'truncated_svd', 'randomized_svd']


def partial_trace(array, traceout):
    """Return the partial trace of an array over the sites given in traceout.

    :param np.ndarray array: Array in global form (see :func:`global_to_local`
        above) with exactly 2 legs per site
    :param traceout: List of sites to trace out, must be in _ascending_ order
    :returns: Partial trace over input array

    """
    if len(traceout) == 0:
        return array
    sites, rem = divmod(array.ndim, 2)
    assert rem == 0, 'ndim={} is not a multiple of 2'.format(array.ndim)

    # Recursively trace out the last site given
    array_red = np.trace(array, axis1=traceout[-1], axis2=traceout[-1] + sites)
    return partial_trace(array_red, traceout=traceout[:-1])


def matdot(A, B, axes=((-1,), (0,))):
    """np.tensordot with sane defaults for matrix multiplication"""
    return np.tensordot(A, B, axes=axes)


def mkron(*args):
    """np.kron() with an arbitrary number of n >= 1 arguments"""
    if len(args) == 1:
        return args[0]
    return mkron(np.kron(args[0], args[1]), *args[2:])


def block_diag(summands, axes=(0, 1)):
    """Block-diagonal sum for n-dimensional arrays.

    Perform something like a block diagonal sum (if len(axes) == 2)
    along the specified axes. All other axes must have identical
    sizes.

    :param axes: Along these axes, perform a block-diagonal sum. Can
        be negative.

    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> b = np.arange(8, 16).reshape((2, 2, 2))
    >>> a
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> b
    array([[[ 8,  9],
            [10, 11]],
    <BLANKLINE>
           [[12, 13],
            [14, 15]]])
    >>> block_diag((a, b), axes=(1, -1))
    array([[[ 0,  1,  0,  0],
            [ 2,  3,  0,  0],
            [ 0,  0,  8,  9],
            [ 0,  0, 10, 11]],
    <BLANKLINE>
           [[ 4,  5,  0,  0],
            [ 6,  7,  0,  0],
            [ 0,  0, 12, 13],
            [ 0,  0, 14, 15]]])

    """
    axes = np.array(axes)
    axes += (axes < 0) * summands[0].ndim

    nr_axes = len(axes)
    axes_order = list(axes)
    axes_order += [i for i in range(summands[0].ndim)
                   if i not in axes]
    summands = [array.transpose(axes_order) for array in summands]

    shapes = np.array([array.shape[:nr_axes] for array in summands])
    res = np.zeros(tuple(shapes.sum(axis=0)) + summands[0].shape[nr_axes:],
                   dtype=summands[0].dtype)
    startpos = np.zeros(nr_axes, dtype=int)

    for array, shape in zip(summands, shapes):
        endpos = startpos + shape
        pos = tuple(slice(start, end) for start, end in zip(startpos, endpos))
        res[pos] += array
        startpos = endpos

    old_axes_order = np.argsort(axes_order)
    res = res.transpose(old_axes_order)
    return res


def truncated_svd(A, k):
    """Compute the truncated SVD of the matrix `A` i.e. the `k` largest
    singular values as well as the corresponding singular vectors. It might
    return less singular values/vectors, if one dimension of `A` is smaller
    than `k`.

    In the background it performs a full SVD. Therefore, it might be
    inefficient when `k` is much smaller than the dimensions of `A`.

    :param A: A real or complex matrix
    :param k: Number of singular values/vectors to compute
    :returns: u, s, v, where
        u: left-singular vectors
        s: singular values in descending order
        v: right-singular vectors

    """
    u, s, v = np.linalg.svd(A)
    k_prime = min(k, len(s))
    return u[:, :k_prime], s[:k_prime], v[:k_prime]


####################
#  Randomized SVD  #
####################

def _standard_normal(shape, randstate=np.random, dtype=np.float_):
    """Generates a standard normal numpy array of given shape and dtype, i.e.
    this function is equivalent to `randstate.randn(*shape)` for real dtype and
    `randstate.randn(*shape) + 1.j * randstate.randn(shape)` for complex dtype.

    :param tuple shape: Shape of array to be returned
    :param randstate: An instance of :class:`numpy.random.RandomState` (default is
        ``np.random``))
    :param dtype: ``np.float_`` (default) or `np.complex_`

    Returns
    -------

    A: An array of given shape and dtype with standard normal entries

    """
    if dtype == np.float_:
        return randstate.randn(*shape)
    elif dtype == np.complex_:
        return randstate.randn(*shape) + 1.j * randstate.randn(*shape)
    else:
        raise ValueError('{} is not a valid dtype.'.format(dtype))


def approx_range_finder(A, sketch_size, n_iter, piter_normalizer='auto',
                        randstate=np.random):
    """Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    :param A: The input data matrix, can be any type that can be converted
        into a :class:`scipy.linalg.LinarOperator`, e.g. :class:`numpy.ndarray`,
        or a sparse matrix.
    :param int sketch_size: Size of the return array
    :param int n_iter: Number of power iterations used to stabilize the result
    :param str piter_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``,
        ``'none'``.  Whether the power iterations are normalized with
        step-by-step QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable but
        can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.
    :param randstate: An instance of :class:`numpy.random.RandomState` (default is
        ``np.random``))

    Returns
    -------
    :returns: :class:`numpy.ndarray`
        A (A.shape[0] x sketch_size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3/4.4 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014

    Original implementation from scikit-learn.

    """
    A = aslinearoperator(A)

    # note that real normal vectors might actually be sufficient
    Q = _standard_normal((A.shape[1], sketch_size), randstate=randstate, dtype=A.dtype)

    # Deal with "auto" mode
    if piter_normalizer == 'auto':
        if n_iter <= 2:
            piter_normalizer = 'none'
        else:
            piter_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if piter_normalizer == 'none':
            Q = A * Q
            Q = A.H * Q
        elif piter_normalizer == 'LU':
            Q, _ = linalg.lu(A * Q, permute_l=True)
            Q, _ = linalg.lu(A.H * Q, permute_l=True)
        elif piter_normalizer == 'QR':
            Q, _ = linalg.qr(A * Q, mode='economic')
            Q, _ = linalg.qr(A.H * Q, mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(A * Q, mode='economic')
    return Q


def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   piter_normalizer='auto', transpose='auto', randstate=np.random):
    """Computes a truncated randomized SVD. Uses the same convention as
    :func:`scipy.sparse.linalg.svds`. However, we guarantee to return the
    singular values in descending order.

    :param M: The input data matrix, can be any type that can be converted
        into a :class:`scipy.linalg.LinarOperator`, e.g. :class:`numpy.ndarray`,
        or a sparse matrix.
    :param int n_components: Number of singular values and vectors to extract.
    :param int n_oversamples: Additional number of random vectors to sample the
        range of `M` so as to ensure proper conditioning. The total number of
        random vectors used to find the range of M is ``n_components +
        n_oversamples``.  Smaller number can improve speed but can negatively
        impact the quality of approximation of singular vectors and singular
        values. (default 10)
    :param n_iter: Number of power iterations. It can be used to deal with very
        noisy problems. When ``'auto'``, it is set to 4, unless
        ``n_components`` is small (``< .1 * min(X.shape)``). Then,
        ``n_iter`` is set to 7.  This improves precision with few
        components. (default ``'auto'``)
    :param str piter_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``,
        ``'none'``.  Whether the power iterations are normalized with
        step-by-step QR factorization (the slowest but most accurate),
        ``'none'`` (the fastest but numerically unstable when `n_iter` is
        large, e.g.  typically 5 or larger), or ``'LU'`` factorization
        (numerically stable but can lose slightly in accuracy). The 'auto' mode
        applies no normalization if ``n_iter <= 2`` and switches to LU
        otherwise.
    :param transpose: ``True``, ``False`` or ``'auto'``
        Whether the algorithm should be applied to ``M.T`` instead of ``M``.
        The result should approximately be the same. The ``'auto'`` mode will
        trigger the transposition if ``M.shape[1] > M.shape[0]`` since then
        the computational overhead in the randomized SVD is generally smaller.
        (default ``'auto'``).
    :param randstate: An instance of :class:`numpy.random.RandomState` (default is
        ``np.random``))

    .. rubric:: Notes

    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, ``n_iter`` can be set <=2 (at the cost of
    loss of precision).

    .. rubric:: References

    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    M = aslinearoperator(M)
    sketch_size = n_components + n_oversamples

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitely specified
        # Adjust n_iter. 7 was found a good compromise for PCA.
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.H

    Q = approx_range_finder(M, sketch_size, n_iter, piter_normalizer, randstate)
    # project M to the (k + p) dimensional space using the basis vectors
    # B = Q.H * M
    B = (M.H * Q).conj().T

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, Uhat)
    sel = slice(None, n_components, 1)

    if transpose:
        # transpose back the results according to the input convention
        return (V[sel].conj().T, s[sel], U[:, sel].conj().T)
    else:
        return U[:, sel], s[sel], V[sel, :]
