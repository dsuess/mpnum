# encoding: utf-8
"""Module to create random test instances of matrix product arrays"""

from __future__ import division, print_function

import functools as ft
import itertools as it
from collections.abc import Iterable

import numpy as np
from scipy.linalg import qr

from six.moves import range

from . import mparray as mp
from . import mpsmpo
from .utils import global_to_local, matdot
from .mpstruct import LocalTensors


__all__ = ['eye', 'random_local_ham', 'random_mpa', 'random_mpdo',
           'random_mps', 'random_mpo', 'zero', 'diagonal_mpa']


def _zrandn(shape, randstate=None):
    """Shortcut for :code:`np.random.randn(*shape) + 1.j *
    np.random.randn(*shape)`

    :param randstate: Instance of np.radom.RandomState or None (which yields
        the default np.random) (default None)

    """
    randstate = randstate if randstate is not None else np.random
    return randstate.randn(*shape) + 1.j * randstate.randn(*shape)


def _randn(shape, randstate=None):
    """Shortcut for :code:`np.random.randn(*shape)`

    :param randstate: Instance of np.radom.RandomState or None (which yields
        the default np.random) (default None)

    """
    randstate = randstate if randstate is not None else np.random
    return randstate.randn(*shape)


_randfuncs = {np.float_: _randn, np.complex_: _zrandn}


def _random_vec(sites, ldim, randstate=None, dtype=np.complex_):
    """Returns a random complex vector (normalized to ||x||_2 = 1) of shape
    (ldim,) * sites, i.e. a pure state with local dimension `ldim` living on
    `sites` sites.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim,) * sites

    >>> psi = _random_vec(5, 2); psi.shape
    (2, 2, 2, 2, 2)
    >>> np.abs(np.vdot(psi, psi) - 1) < 1e-6
    True
    """
    shape = (ldim, ) * sites
    psi = _randfuncs[dtype](shape, randstate=randstate)
    psi /= np.linalg.norm(psi)
    return psi


def _random_op(sites, ldim, hermitian=False, normalized=False, randstate=None,
               dtype=np.complex_):
    """Returns a random operator  of shape (ldim,ldim) * sites with local
    dimension `ldim` living on `sites` sites in global form.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param hermitian: Return only the hermitian part (default False)
    :param normalized: Normalize to Frobenius norm=1 (default False)
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim,ldim) * sites

    >>> A = _random_op(3, 2); A.shape
    (2, 2, 2, 2, 2, 2)
    """
    op = _randfuncs[dtype]((ldim**sites,) * 2, randstate=randstate)
    if hermitian:
        op += np.transpose(op).conj()
    if normalized:
        op /= np.linalg.norm(op)
    return op.reshape((ldim,) * 2 * sites)


def _random_state(sites, ldim, randstate=None):
    """Returns a random positive semidefinite operator of shape (ldim, ldim) *
    sites normalized to Tr rho = 1, i.e. a mixed state with local dimension
    `ldim` living on `sites` sites. Note that the returned state is positive
    semidefinite only when interpreted in global form (see
    :func:`tools.global_to_local`)

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim, ldim) * sites

    >>> from numpy.linalg import eigvalsh
    >>> rho = _random_state(3, 2).reshape((2**3, 2**3))
    >>> all(eigvalsh(rho) >= 0)
    True
    >>> np.abs(np.trace(rho) - 1) < 1e-6
    True
    """
    shape = (ldim**sites, ldim**sites)
    mat = _zrandn(shape, randstate=randstate)
    rho = np.conj(mat.T).dot(mat)
    rho /= np.trace(rho)
    return rho.reshape((ldim,) * 2 * sites)


####################################
#  Factory functions for MPArrays  #
####################################

def _generate(sites, ldim, rank, func, force_rank):
    """Returns a matrix product operator with identical number and dimensions
    of the physical legs. The local tensors are generated using `func`

    :param sites: Number of sites

    :param ldim: Physical legs, depending on the type passed:

        * scalar: Single physical leg for each site with given dimension
        * iterable of scalar: Same physical legs for all sites
        * iterable of iterable: Generated MPA will have exactly this
          as `ndims`

    :param rank: rank, depending on the type passed:

        * scalar: Same rank everywhere
        * iterable of length :code:`sites - 1`: Generated MPA will
          have exactly this as `ranks`

    :param func: Generator function for local tensors, should accept
        shape as tuple in first argument and should return
        numpy.ndarray of given shape
    :param force_rank: If True, the rank is exaclty `rank`.
        Otherwise, it might be reduced if we reach the maximum sensible rank.
    :returns: randomly choosen matrix product array

    """
    # If ldim is passed as scalar, make it 1-element tuple.
    ldim = tuple(ldim) if isinstance(ldim, Iterable) else (ldim,)
    # If ldim[0] is not iterable, we want the same physical legs on
    # all sites.
    if not isinstance(ldim[0], Iterable):
        ldim = (ldim,) * sites
    # If rank is not iterable, we want the same rank
    # everywhere.
    if not isinstance(rank, Iterable):
        rank = (rank,) * (sites - 1)
    else:
        rank = tuple(rank)

    if not force_rank:
        rank = tuple(min(b1, b2) for b1, b2 in zip(rank, mp.full_rank(ldim)))

    assert len(ldim) == sites
    assert len(rank) == sites - 1

    rank = (1,) + rank + (1,)
    ltens = (func((rank[n],) + tuple(ld) + (rank[n + 1],))
             for n, ld in enumerate(ldim))
    return mp.MPArray(ltens)


def random_mpa(sites, ldim, rank, randstate=None, normalized=False,
               force_rank=False, dtype=np.float_):
    """Returns an MPA with randomly choosen local tensors (real by default)

    :param sites: Number of sites
    :param ldim: Physical legs, depending on the type passed:

        * scalar: Single physical leg for each site with given dimension
        * iterable of scalar: Same physical legs for all sites
        * iterable of iterable: Generated MPA will have exactly this
          as `ndims`

    :param rank: Desired rank, depending on the type passed:

        * scalar: Same rank everywhere
        * iterable of length :code:`sites - 1`: Generated MPA will
          have exactly this as `ranks`

    :param randstate: numpy.random.RandomState instance or None
    :param normalized: Resulting `mpa` has `mp.norm(mpa) == 1`
    :param force_rank: If True, the rank is exaclty `rank`.
        Otherwise, it might be reduced if we reach the maximum sensible rank.
    :param dtype: Type of the returned MPA. Currently only
        ``np.float_`` and ``np.complex_`` are implemented (default:
        ``np.float_``, i.e. real values).

    :returns: Randomly choosen matrix product array

    Entries of local tensors are drawn from a normal distribution of
    unit variance.  For complex values, the real and imaginary parts
    are independent and have unit variance.

    >>> mpa = random_mpa(4, 2, 10, force_rank=True)
    >>> mpa.ranks, mpa.shape
    ((10, 10, 10), ((2,), (2,), (2,), (2,)))

    >>> mpa = random_mpa(4, (1, 2), 10, force_rank=True)
    >>> mpa.ranks, mpa.shape
    ((10, 10, 10), ((1, 2), (1, 2), (1, 2), (1, 2)))

    >>> mpa = random_mpa(4, [(1, ), (2, 3), (4, 5), (1, )], 10, force_rank=True)
    >>> mpa.ranks, mpa.shape
    ((10, 10, 10), ((1,), (2, 3), (4, 5), (1,)))

    The following doctest verifies that we do not change how random
    states are generated, ensuring reproducible results. In addition,
    it verifies the returned dtype:

    >>> rng = np.random.RandomState(seed=3208886881)
    >>> random_mpa(2, 2, 3, rng).to_array()
    array([[-0.7254321 ,  3.44263486],
           [-0.17262967,  2.4505633 ]])
    >>> random_mpa(2, 2, 3, rng, dtype=np.complex_).to_array()
    array([[-0.53552415+1.39701566j, -2.12128866+0.57913253j],
           [-0.32652114+0.51490923j, -0.32222320-0.32675463j]])

    """
    randfun = ft.partial(_randfuncs[dtype], randstate=randstate)
    mpa = _generate(sites, ldim, rank, randfun, force_rank)
    if normalized:
        mpa /= mp.norm(mpa.copy())
    return mpa


def zero(sites, ldim, rank, force_rank=False):
    """Returns a MPA with localtensors beeing zero (but of given shape)

    :param sites: Number of sites
    :param ldim: Depending on the type passed (checked in the following order)

        * iterable of iterable: Detailed list of physical dimensions,
          retured mpa will have exactly this for mpa.shape
        * iterable of scalar: Same physical dimension for each site
        * scalar: Single physical leg for each site with given
          dimension

    :param rank: Rank
    :param force_rank: If True, the rank is exaclty `rank`.
        Otherwise, it might be reduced if we reach the maximum sensible rank.
    :returns: Representation of the zero-array as MPA

    """
    return _generate(sites, ldim, rank, np.zeros, force_rank)


def eye(sites, ldim):
    """Returns a MPA representing the identity matrix

    :param sites: Number of sites
    :param ldim: Int-like local dimension or iterable of local dimensions
    :returns: Representation of the identity matrix as MPA

    >>> I = eye(4, 2)
    >>> I.ranks, I.shape
    ((1, 1, 1), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> I = eye(3, (3, 4, 5))
    >>> I.shape
    ((3, 3), (4, 4), (5, 5))
    """
    if isinstance(ldim, Iterable):
        ldim = tuple(ldim)
        assert len(ldim) == sites
    else:
        ldim = it.repeat(ldim, sites)
    return mp.MPArray.from_kron(map(np.eye, ldim))


def diagonal_mpa(entries, sites):
    """Returns an MPA with ``entries`` on the diagonal and zeros otherwise.

    :param numpy.ndarray entries: one-dimensional array
    :returns: :class:`~mpnum.mparray.MPArray` with rank ``len(entries)``.

    """
    assert sites > 0

    if entries.ndim != 1:
        raise NotImplementedError("Currently only supports diagonal MPA with "
                                  "one leg per site.")

    if sites < 2:
        return mp.MPArray.from_array(entries)

    ldim = len(entries)
    leftmost_ltens = np.eye(ldim).reshape((1, ldim, ldim))
    rightmost_ltens = np.diag(entries).reshape((ldim, ldim, 1))
    center_ltens = np.zeros((ldim,) * 3)
    np.fill_diagonal(center_ltens, 1)
    ltens = it.chain((leftmost_ltens,), it.repeat(center_ltens, sites - 2),
                     (rightmost_ltens,))

    return mp.MPArray(LocalTensors(ltens, cform=(sites - 1, sites)))


#########################
#  More physical stuff  #
#########################
def random_mpo(sites, ldim, rank, randstate=None, hermitian=False,
               normalized=True, force_rank=False):
    """Returns an hermitian MPO with randomly choosen local tensors

    :param sites: Number of sites
    :param ldim: Local dimension
    :param rank: Rank
    :param randstate: numpy.random.RandomState instance or None
    :param hermitian: Is the operator supposed to be hermitian
    :param normalized: Operator should have unit norm
    :param force_rank: If True, the rank is exaclty `rank`.
        Otherwise, it might be reduced if we reach the maximum sensible rank.
    :returns: randomly choosen matrix product operator

    >>> mpo = random_mpo(4, 2, 10, force_rank=True)
    >>> mpo.ranks, mpo.shape
    ((10, 10, 10), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> mpo.canonical_form
    (0, 4)

    """
    mpo = random_mpa(sites, (ldim,) * 2, rank, randstate=randstate,
                     force_rank=force_rank, dtype=np.complex_)

    if hermitian:
        # make mpa Herimitan in place, without increasing rank:
        ltens = (l + l.swapaxes(1, 2).conj() for l in mpo.lt)
        mpo = mp.MPArray(ltens)
    if normalized:
        # we do this with a copy to ensure the returned state is not
        # normalized
        mpo /= mp.norm(mpo.copy())

    return mpo


def random_mps(sites, ldim, rank, randstate=None, force_rank=False):
    """Returns a randomly choosen normalized matrix product state

    :param sites: Number of sites
    :param ldim: Local dimension
    :param rank: Rank
    :param randstate: numpy.random.RandomState instance or None
    :param force_rank: If True, the rank is exaclty `rank`.
        Otherwise, it might be reduced if we reach the maximum sensible rank.
    :returns: randomly choosen matrix product (pure) state

    >>> mps = random_mps(4, 2, 10, force_rank=True)
    >>> mps.ranks, mps.shape
    ((10, 10, 10), ((2,), (2,), (2,), (2,)))
    >>> mps.canonical_form
    (0, 4)
    >>> round(abs(1 - mp.inner(mps, mps)), 10)
    0.0

    """
    return random_mpa(sites, ldim, rank, normalized=True, randstate=randstate,
                      force_rank=force_rank, dtype=np.complex_)


def random_mpdo(sites, ldim, rank, randstate=np.random):
    """Returns a randomly choosen matrix product density operator (i.e.
    positive semidefinite matrix product operator with trace 1).

    :param sites: Number of sites
    :param ldim: Local dimension
    :param rank: Rank
    :param randstate: numpy.random.RandomState instance
    :returns: randomly choosen classicaly correlated matrix product density op.

    >>> rho = random_mpdo(4, 2, 4)
    >>> rho.ranks, rho.shape
    ((4, 4, 4), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> rho.canonical_form
    (0, 4)

    """
    # generate density matrix as a mixture of `rank` pure product states
    psis = [random_mps(sites, ldim, 1, randstate=randstate) for _ in range(rank)]
    weights = (lambda x: x / np.sum(x))(randstate.rand(rank))
    rho = mp.sumup(mpsmpo.mps_to_mpo(psi) * weight
                   for weight, psi in zip(weights, psis))

    # Scramble the local tensors
    for n, rank in enumerate(rho.ranks):
        unitary = _unitary_haar(rank, randstate)
        rho.lt[n] = matdot(rho.lt[n], unitary)
        rho.lt[n + 1] = matdot(np.transpose(unitary).conj(), rho.lt[n + 1])

    rho /= mp.trace(rho)
    return rho


def random_local_ham(sites, ldim=2, intlen=2, randstate=None):
    """Generates a random Hamiltonian on `sites` sites with local dimension
    `ldim`, which is a sum of local Hamiltonians with interaction length
    `intlen`.

    :param sites: Number of sites
    :param ldim: Local dimension
    :param intlen: Interaction length of the local Hamiltonians
    :returns: MPA representation of the global Hamiltonian

    """
    def get_local_ham():
        op = _random_op(intlen, ldim, hermitian=True, normalized=True)
        op = global_to_local(op, sites=intlen)
        return mp.MPArray.from_array(op, ndims=2)

    assert sites >= intlen
    local_hams = [get_local_ham() for _ in range(sites + 1 - intlen)]
    return mp.local_sum(local_hams)


def _unitary_haar(dim, randstate=None):
    """Returns a sample from the Haar measure of the unitary group of given
    dimension.

    :param int dim: Dimension
    :param randn: Function to create real N(0,1) distributed random variables.
        It should take the shape of the output as numpy.random.randn does
        (default: numpy.random.randn)
    """
    z = _zrandn((dim, dim), randstate) / np.sqrt(2.0)
    q, r = qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph
