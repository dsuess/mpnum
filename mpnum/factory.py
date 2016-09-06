# encoding: utf-8
"""Module to create random test instances of matrix product arrays"""

from __future__ import division, print_function

import functools as ft
import itertools as it
import collections

import numpy as np
from scipy.linalg import qr

from six.moves import range

from . import mparray as mp
from . import mpsmpo
from ._tools import global_to_local, matdot


__all__ = ['eye', 'random_local_ham', 'random_mpa', 'random_mpdo',
           'random_mps', 'random_mpo', 'random_op', 'random_state',
           'random_vec', 'zero']


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


def random_vec(sites, ldim, randstate=None, dtype=np.complex_):
    """Returns a random complex vector (normalized to ||x||_2 = 1) of shape
    (ldim,) * sites, i.e. a pure state with local dimension `ldim` living on
    `sites` sites.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim,) * sites

    >>> psi = random_vec(5, 2); psi.shape
    (2, 2, 2, 2, 2)
    >>> np.abs(np.vdot(psi, psi) - 1) < 1e-6
    True
    """
    shape = (ldim, ) * sites
    psi = _randfuncs[dtype](shape, randstate=randstate)
    psi /= np.linalg.norm(psi)
    return psi


def random_op(sites, ldim, hermitian=False, normalized=False, randstate=None,
              dtype=np.complex_):
    """Returns a random operator  of shape (ldim,ldim) * sites with local
    dimension `ldim` living on `sites` sites in global form.

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param hermitian: Return only the hermitian part (default False)
    :param normalized: Normalize to Frobenius norm=1 (default False)
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim,ldim) * sites

    >>> A = random_op(3, 2); A.shape
    (2, 2, 2, 2, 2, 2)
    """
    op = _randfuncs[dtype]((ldim**sites,) * 2, randstate=randstate)
    if hermitian:
        op += np.transpose(op).conj()
    if normalized:
        op /= np.linalg.norm(op)
    return op.reshape((ldim,) * 2 * sites)


def random_state(sites, ldim, randstate=None):
    """Returns a random positive semidefinite operator of shape (ldim, ldim) *
    sites normalized to Tr rho = 1, i.e. a mixed state with local dimension
    `ldim` living on `sites` sites. Note that the returned state is positive
    semidefinite only when interpreted in global form (see
    :func:`_tools.global_to_local`)

    :param sites: Number of local sites
    :param ldim: Local ldimension
    :param randstate: numpy.random.RandomState instance or None
    :returns: numpy.ndarray of shape (ldim, ldim) * sites

    >>> from numpy.linalg import eigvalsh
    >>> rho = random_state(3, 2).reshape((2**3, 2**3))
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


def _generate(sites, ldim, bdim, func):
    """Returns a matrix product operator with identical number and dimensions
    of the physical legs. The local tensors are generated using `func`

    :param sites: Number of sites

    :param ldim: Physical legs, depending on the type passed:

        * scalar: Single physical leg for each site with given dimension
        * iterable of scalar: Same physical legs for all sites
        * iterable of iterable: Generated MPA will have exactly this
          as `plegs`

    :param bdim: Bond dimension, depending on the type passed:

        * scalar: Same bond dimension everywhere
        * iterable of length :code:`sites - 1`: Generated MPA will
          have exactly this as `bdims`

    :param func: Generator function for local tensors, should accept
        shape as tuple in first argument and should return
        numpy.ndarray of given shape

    :returns: randomly choosen matrix product array

    """
    # If ldim is passed as scalar, make it 1-element tuple.
    ldim = tuple(ldim) if isinstance(ldim, collections.Iterable) else (ldim,)
    # If ldim[0] is not iterable, we want the same physical legs on
    # all sites.
    if not isinstance(ldim[0], collections.Iterable):
        ldim = (ldim,) * sites
    # If bdim is not iterable, we want the same bond dimension
    # everywhere.
    if not isinstance(bdim, collections.Iterable):
        bdim = (bdim,) * (sites - 1)
    else:
        bdim = tuple(bdim)
    bdim_l = (1,) + bdim
    bdim_r = bdim + (1,)
    assert len(ldim) == sites
    assert len(bdim) == sites - 1

    ltens = (
        func((b_l,) + tuple(ld) + (b_r,))
        for b_l, ld, b_r in zip(bdim_l, ldim, bdim_r))
    return mp.MPArray(ltens)


def random_mpa(sites, ldim, bdim, randstate=None, normalized=False,
               dtype=np.complex_):
    """Returns a MPA with randomly choosen local tensors

    :param sites: Number of sites
    :param ldim: Depending on the type passed (checked in the following order)

        * iterable of iterable: Detailed list of physical dimensions,
          retured mpa will have exactly this for mpa.pdims
        * iterable of scalar: Same physical dimension for each site
        * scalar: Single physical leg for each site with given
          dimension

    :param bdim: Bond dimension
    :param randn: Function used to generate random local tensors
    :param randstate: numpy.random.RandomState instance or None
    :param normalized: Resulting `mpa` has `mp.norm(mpa) == 1`
    :param dtype: Whicht type the returned array should have. Currently only
        `np.real_` and `np.complex_` is implemented (default: complex)
    :returns: randomly choosen matrix product array

    >>> mpa = random_mpa(4, 2, 10)
    >>> mpa.bdims, mpa.pdims
    ((10, 10, 10), ((2,), (2,), (2,), (2,)))

    >>> mpa = random_mpa(4, (1, 2), 10)
    >>> mpa.bdims, mpa.pdims
    ((10, 10, 10), ((1, 2), (1, 2), (1, 2), (1, 2)))

    >>> mpa = random_mpa(4, [(1, ), (2, 3), (4, 5), (1, )], 10)
    >>> mpa.bdims, mpa.pdims
    ((10, 10, 10), ((1,), (2, 3), (4, 5), (1,)))

    """
    randfun = ft.partial(_randfuncs[dtype], randstate=randstate)
    mpa = _generate(sites, ldim, bdim, randfun)
    if normalized:
        mpa /= mp.norm(mpa.copy())
    return mpa


def zero(sites, ldim, bdim):
    """Returns a MPA with localtensors beeing zero (but of given shape)

    :param sites: Number of sites
    :param ldim: Depending on the type passed (checked in the following order)

        * iterable of iterable: Detailed list of physical dimensions,
          retured mpa will have exactly this for mpa.pdims
        * iterable of scalar: Same physical dimension for each site
        * scalar: Single physical leg for each site with given
          dimension

    :param bdim: Bond dimension
    :returns: Representation of the zero-array as MPA

    """
    return _generate(sites, ldim, bdim, np.zeros)


def eye(sites, ldim):
    """Returns a MPA representing the identity matrix

    :param sites: Number of sites
    :param ldim: Int-like local dimension or iterable of local dimensions
    :returns: Representation of the identity matrix as MPA

    >>> I = eye(4, 2)
    >>> I.bdims, I.pdims
    ((1, 1, 1), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> I = eye(3, (3, 4, 5))
    >>> I.pdims
    ((3, 3), (4, 4), (5, 5))
    """
    if isinstance(ldim, collections.Iterable):
        ldim = tuple(ldim)
        assert len(ldim) == sites
    else:
        ldim = it.repeat(ldim, sites)
    return mp.MPArray.from_kron(map(np.eye, ldim))


#########################
#  More physical stuff  #
#########################
def random_mpo(sites, ldim, bdim, randstate=None, hermitian=False,
               normalized=True):
    """Returns an hermitian MPO with randomly choosen local tensors

    :param sites: Number of sites
    :param ldim: Local dimension
    :param bdim: Bond dimension
    :param randstate: numpy.random.RandomState instance or None
    :param hermitian: Is the operator supposed to be hermitian
    :param normalized: Operator should have unit norm
    :returns: randomly choosen matrix product operator

    >>> mpo = random_mpo(4, 2, 10)
    >>> mpo.bdims, mpo.pdims
    ((10, 10, 10), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> mpo.normal_form
    (0, 4)

    """
    mpo = random_mpa(sites, (ldim,) * 2, bdim, randstate=randstate)

    if hermitian:
        # make mpa Herimitan in place, without increasing bond dimension:
        for lten in mpo:
            lten += lten.swapaxes(1, 2).conj()
    if normalized:
        # we do this with a copy to ensure the returned state is not
        # normalized
        mpo /= mp.norm(mpo.copy())

    return mpo


def random_mps(sites, ldim, bdim, randstate=None):
    """Returns a randomly choosen normalized matrix product state

    :param sites: Number of sites
    :param ldim: Local dimension
    :param bdim: Bond dimension
    :param randstate: numpy.random.RandomState instance or None
    :returns: randomly choosen matrix product (pure) state

    >>> mps = random_mps(4, 2, 10)
    >>> mps.bdims, mps.pdims
    ((10, 10, 10), ((2,), (2,), (2,), (2,)))
    >>> mps.normal_form
    (0, 4)
    >>> round(abs(1 - mp.inner(mps, mps)), 10)
    0.0

    """
    mps = random_mpa(sites, ldim, bdim, randstate=randstate)
    mps /= mp.norm(mps.copy())
    return mps


def random_mpdo(sites, ldim, bdim, randstate=np.random):
    """Returns a randomly choosen matrix product density operator (i.e.
    positive semidefinite matrix product operator with trace 1).

    :param sites: Number of sites
    :param ldim: Local dimension
    :param bdim: Bond dimension
    :param randstate: numpy.random.RandomState instance
    :returns: randomly choosen classicaly correlated matrix product density op.

    >>> rho = random_mpdo(4, 2, 4)
    >>> rho.bdims, rho.pdims
    ((4, 4, 4), ((2, 2), (2, 2), (2, 2), (2, 2)))
    >>> rho.normal_form
    (0, 4)

    """
    # generate density matrix as a mixture of `bdim` pure product states
    psis = [random_mps(sites, ldim, 1, randstate=randstate) for _ in range(bdim)]
    weights = (lambda x: x / np.sum(x))(randstate.rand(bdim))
    rho = ft.reduce(mp.MPArray.__add__, (mpsmpo.mps_to_mpo(psi) * weight
                                         for weight, psi in zip(weights, psis)))

    # Scramble the local tensors
    for n, bdim in enumerate(rho.bdims):
        unitary = _unitary_haar(bdim, randstate)
        rho[n] = matdot(rho[n], unitary)
        rho[n + 1] = matdot(np.transpose(unitary).conj(), rho[n + 1])

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
        op = random_op(intlen, ldim, hermitian=True, normalized=True)
        op = global_to_local(op, sites=intlen)
        return mp.MPArray.from_array(op, plegs=2)

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
