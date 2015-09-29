# encoding: utf-8


r"""Matrix Product State (MPS) and Operator (MPO) functions

.. _mpsmpo-definitions:

Definitions
-----------

We consider a linear chain of :math:`n` sites with associated Hilbert
spaces \mathcal H_k = \C^{d_k}, :math:`d_k`, :math:`k \in [1..n] :=
\{1, 2, \ldots, n\}`. The set of linear operators :math:`\mathcal H_k
\to \mathcal H_k` is denoted by :math:`\mathcal B_k`. We write
:math:`\mathcal H = \mathcal H_1 \otimes \cdots \otimes \mathcal H_n`
and the same for :math:`\mathcal B`.

We use the following three representations:

* Matrix product state (MPS): Vector :math:`\lvert \psi \rangle \in
  \mathcal H`

* Matrix product operator (MPO): Operator :math:`M \in \mathcal B`

* Locally purified matrix product state (PMPS): Positive semidefinite
  operator :math:`\rho \in \mathcal B`

All objects are represented by :math:`n` local tensors.

Matrix product state (MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a vector :math:`\lvert \psi \rangle \in \mathcal H` as

.. math::

   \langle i_1 \ldots i_n \vert \psi \rangle 
   = A^{(1)}_{i_1} \cdots A^{(n)}_{i_n}, 
   \quad A^{(k)}_{i_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,r} =
(A^{(k)}_i)_{l,r}`.

The vector :math:`\lvert \psi \rangle` can be a state, with the
density matrix given by :math:`\rho = \lvert \psi \rangle \langle \psi
\rvert \in \mathcal B`.

Matrix product operator (MPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent an operator :math:`M \in \mathcal B` as

.. math::

  \langle i_1 \ldots i_n \vert M \vert j_1 \ldots j_n \rangle
  = A^{(1)}_{i_1 j_1} \cdots A^{(n)}_{i_n j_n},
  \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,j,r} = (A^{(k)}_{i
j})_{l,r}`.

This representation can be used to represent a mixed state :math:`\rho
= M`, but it is not limited to positive semidefinite :math:`M`.

Locally purified matrix product state (PMPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a positive semidefinite operator :math:`\rho \in \mathcal B`
as follows: Let :math:`\mathcal H_k' = \mathbb C^{d'_k}` with suitable
:math:`d'_k` and :math:`\mathcal P = \mathcal H_1 \otimes \mathcal
H'_1 \otimes \cdots \otimes \mathcal H_n \otimes \mathcal H'_n`. Find
:math:`\vert \Phi \rangle \in \mathcal P` such that

.. math::

   \rho = \operatorname{tr}_{\mathcal H'_1, \ldots, \mathcal H'_n}
   (\lvert \Phi \rangle \langle \Phi \rvert)

and represent :math:`\lvert \Phi \rangle` as

.. math::

   \langle i_1 i'_1 \ldots i_n i'_n \vert \Phi \rangle
   = A^{(1)}_{i_1 i'_1} \cdots A^{(n)}_{i_n i'_n},
   \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k}, 
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,i',r} = (A^{(k)}_{i
i'})_{l,r}`.

The ancillary dimensions :math:`d'_i` are not determined by the
:math:`d_i` but depend on the state. E.g. if :math:`\rho` is pure, one
can set all :math:`d_i = 1`.


.. todo:: Add references.

.. todo:: Are derived classes MPO/MPS/PMPS of any help?

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_equal

import mpnum.mparray as mp
from mpnum._tools import matdot
from six.moves import range


def _check_reductions_args(nr_sites, width, startsites, stopsites):
    """Expand the arguments of :func:`reductions_mpo()` et al.

    The arguments are documented in :func:`reductions_mpo()`.

    """
    if stopsites is None:
        assert width is not None
        if startsites is None:
            startsites = range(nr_sites - width + 1)
        stopsites = (start + width for start in startsites)
    else:
        assert width is None
        assert startsites is not None
    return startsites, stopsites


def reductions_mpo(mpa, width=None, startsites=None, stopsites=None):
    """Iterate over MPO partial traces of an MPO

    The support of the i-th result is :code:`range(startsites[i],
    stopsites[i])`.

    :param mpnum.mparray.MPArray mpa: An MPO

    :param startsites: Defaults to :code:`range(len(mpa) - width +
        1)`.

    :param stopsites: Defaults to :code:`[ start + width for start in
        startsites ]`. If specified, we require `startsites` to be
        given and `width` to be None.

    :param width: Number of sites in support of the results. Default
        `None`. Must be specified if one or both of `startsites` and
        `stopsites` are not given.

    :returns: Iterator over partial traces as MPO

    """
    startsites, stopsites = \
        _check_reductions_args(len(mpa), width, startsites, stopsites)

    assert_array_equal(mpa.plegs, 2)
    rem_left = {0: np.array(1, ndmin=2)}
    rem_right = rem_left.copy()

    def get_remainder(rem_cache, num_sites, end):
        """Obtain the vectors resulting from tracing over
        the left or right end of a Matrix Product Operator.

        :param rem_cache: Save remainder terms with smaller num_sites here
        :param num_sites: Number of sites from left or right that have been
            traced over.
        :param end: +1 or -1 for tracing over the left or right end
        """
        try:
            return rem_cache[num_sites]
        except KeyError:
            rem = get_remainder(rem_cache, num_sites - 1, end)
            last_pos = num_sites - 1 if end == 1 else -num_sites
            add = np.trace(mpa[last_pos], axis1=1, axis2=2)
            if end == -1:
                rem, add = add, rem

            rem_cache[num_sites] = matdot(rem, add)
            return rem_cache[num_sites]

    num_sites = len(mpa)
    for start, stop in zip(startsites, stopsites):
        # FIXME we could avoid taking copies here, but then in-place
        # multiplication would have side effects. We could make the
        # affected arrays read-only to turn unnoticed side effects into
        # errors.
        # Is there something like a "lazy copy" or "copy-on-write"-copy?
        # I believe not.
        ltens = [lten.copy() for lten in mpa[start:stop]]
        rem = get_remainder(rem_left, start, 1)
        ltens[0] = matdot(rem, ltens[0])
        rem = get_remainder(rem_right, num_sites - stop, -1)
        ltens[-1] = matdot(ltens[-1], rem)
        yield mp.MPArray(ltens)


def reductions_pmps(pmps, width=None, startsites=None, stopsites=None):
    """Iterate over PMPS partial traces of a PMPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param pmps: Mixed state in locally purified MPS representation
        (PMPS, see :ref:`mpsmpo-definitions`)
    :returns: Iterator over reduced states as PMPS

    """
    startsites, stopsites = \
        _check_reductions_args(len(pmps), width, startsites, stopsites)

    for start, stop in zip(startsites, stopsites):
        pmps.normalize(left=start, right=stop)

        # leftmost site
        lten = pmps[start]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (1, system, left_bd * ancilla, right_bd)
        ltens = [lten.swapaxes(0, 1).copy().reshape(newshape)]

        # central sites and last site
        ltens += (lten.copy() for lten in pmps[start + 1:stop])

        # fix up the last site -- may be the same as the first site
        left_bd, system, ancilla, right_bd = ltens[-1].shape
        newshape = (left_bd, system, ancilla * right_bd, 1)
        ltens[-1] = ltens[-1].reshape(newshape)

        reduced_mps = mp.MPArray(ltens)
        yield reduced_mps


def reductions_mps_as_pmps(mps, width=None, startsites=None, stopsites=None):
    """Iterate over PMPS reduced states of an MPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param mps: Pure state as MPS
    :returns: Iterator over reduced states as PMPS

    """
    pmps = mps_to_pmps(mps)
    return reductions_pmps(pmps, width, startsites, stopsites)


def reductions_mps_as_mpo(mps, width=None, startsites=None, stopsites=None):
    """Iterate over MPO reduced states of an MPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param mps: Pure state as MPS
    :returns: Iterator over reduced states as MPO

    """
    return map(pmps_to_mpo, reductions_mps_as_pmps(mps, width, startsites, stopsites))


def pmps_to_mpo(pmps):
    """Convert a local purification MPS to a mixed state MPO.

    A mixed state on n sites is represented in local purification MPS
    form by a MPA with n sites and two physical legs per site. The
    first physical leg is a 'system' site, while the second physical
    leg is an 'ancilla' site.

    :param MPArray pmps: An MPA with two physical legs (system and ancilla)
    :returns: An MPO (density matrix as MPA with two physical legs)

    """
    return mp.dot(pmps, pmps.adj())


def mps_to_pmps(mps):
    """Convert a pure MPS into a local purification MPS mixed state.

    The ancilla legs will have dimension one, not increasing the
    memory required for the MPS.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPA with two physical legs (system and ancilla)

    """
    assert_array_equal(mps.plegs, 1)
    ltens = (lten.reshape(lten.shape[0:2] + (1, lten.shape[2])) for lten in mps)
    return mp.MPArray(ltens)


def mps_to_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    return pmps_to_mpo(mps_to_pmps(mps))
