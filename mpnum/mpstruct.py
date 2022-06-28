# encoding: utf-8
"""Core data structure & routines to manage local tensors"""
from __future__ import absolute_import, division, print_function

import itertools as it
from collections.abc import Sequence

from six.moves import range, zip


__all__ = ['LocalTensors']


def _roview(array):
    """Creates a read only view of the numpy array `view`."""
    view = array.view()
    view.setflags(write=False)
    return view


class LocalTensors(object):
    r"""Core data structure to manage the local tensors of a
    :class:`~mpnum.mparray.MPArray`\ .

    The local tensors are kept in ``_ltens``\ , a list of
    :class:`numpy.ndarray`\ s such that ``_ltens[i]`` corresponds to the
    local tensor of site `i`\ .

    If there are :math:`k` (non-virtual) indices at site :math:`i`, the
    corresponding local tensor is a ndarray with ``ndim == k + 2``. The two
    additional indices of the local tensor correspond to the virtual legs. We
    reserve the `0`\ th index of the local tensor for the virtal leg coupling
    to site :math:`i - 1` and the last index for the virtual leg coupling to
    site :math:`i + 1`.

    Therefore, if the physical legs at site :math:`i` have dimensions
    :math:`d_1, \ldots, d_k`, the corresponding local tensor has shape
    :math:`(r_{i-1}, d_1, \ldots, d_k, r_{i})`. Here, :math:`r_{i-1}` and
    :math:`r_i` denote the rank between sites :math:`(i - 1, i)` and
    :math:`(i, i + 1)`, respectively.

    To keep the data structure consistent, we include the left virutal leg of
    the leftmost local tensor as well as the right virtual leg of the rightmost
    local tensor as dummy indices of dimension 1.

    """

    def __init__(self, ltens, cform=(None, None)):
        """
        :param ltens: List of local tensor according to the data structure
            described at :class:`LocalTensors`.
        :param tuple cform: Canoncial form of the local tensors passed in.
            Should follow the conventions of :py:attr:`canonical_form`.
            The following values are equivalent:
            - for ``cform[0]``: ``None`` and ``0`` (i.e. no left-canonical form
              is assumed)
            - for ``cform[1]``: ``None`` and ``len(ltens)`` (i.e. no
              right-canonical form is assumed)

        """
        self._ltens = list(ltens)
        lcanonical, rcanonical = cform
        self._lcanonical = lcanonical or 0
        self._rcanonical = rcanonical or len(self._ltens)

        assert len(self._ltens) > 0
        assert 0 <= self._lcanonical < len(self._ltens)
        assert 0 < self._rcanonical <= len(self._ltens)

        if __debug__:
            for i, (ten, nten) in enumerate(zip(self._ltens[:-1],
                                                self._ltens[1:])):
                assert ten.shape[-1] == nten.shape[0]

    def _update(self, index, tens, canonicalization=None):
        """Update the local tensor at site ``index`` to the new value ``tens``.
        Keeps track of canonical form during the update.

        This function only performs the update and does not check if the
        resulting MPA is consistent (e.g. non-matching virtual legs in two
        neighboring local tensors)

        For a safe update function and the parameters see :func:`update`. In
        contrast to :func:`update`, this function only accepts a single index
        and NO slices.
        """
        self._ltens[index] = tens
        # If a canonical tensor is set next to a slice in canonical form,
        # the size of the canonical slice will increase by one
        # (equality case; first argument to max/min). If a canoical
        # tensor is set inside a canoncial slice, its size will
        # remain the same (inequality case; second argument).
        if canonicalization == 'left' and self._lcanonical - index >= 0:
            self._lcanonical = max(index + 1, self._lcanonical)
        elif canonicalization == 'right' and index - self._rcanonical >= -1:
            self._rcanonical = min(index, self._rcanonical)
        else:
            # If a non-canonical tensor is provided, the sizes of the
            # canoical slices may decrease.
            self._lcanonical = min(index, self._lcanonical)
            self._rcanonical = max(index + 1, self._rcanonical)

    def update(self, index, tens, canonicalization=None):
        """Update the local tensor at site ``index`` to the new value ``tens``.
        Checks the rank and shape of the new values to keep the MPA consistent.
        Therefore, some actions such as changing the rank between two sites
        require to update both sites at the same time, which can be done by
        passing in multiple values as arguments.

        :param index: Integer/slice. Site index/indices of the local tensor/
            tensors to be updated.
        :param tens: New local tensor as ``numpy.ndarray``. Alternatively,
            sequence over multiple ndarrays (in case ``index`` is a slice).
        :param canonicalization: If ``tens`` is left-/right-normalized, pass
            ``'left'``/``'right'``, respectively. Otherwise, pass ``None``
            (default ``None``). In case ``index`` is a slice, either pass a
            sequence of the corresponding values or a single value, which is
            repeated for each site updated.

        """
        if isinstance(index, slice):
            indices = index.indices(len(self))
            # In Python 3, we can do range(*indices).start etc. Python 2 compat:
            start, stop, step = indices
            # Allow rank changes if multiple consecutive local tensors change.
            tens = list(tens)
            assert self[start].shape[0] == tens[0].shape[0]
            assert all(t.shape[-1] == u.shape[0] for t, u in zip(
                tens[:-1], tens[1:]))
            assert self[stop - 1].shape[-1] == tens[-1].shape[-1]

            if not isinstance(canonicalization, Sequence):
                canonicalization = it.repeat(canonicalization)

            for ten, pos, norm in zip(tens, range(*indices), canonicalization):
                self._update(pos, ten, canonicalization=norm)

        else:
            current = self._ltens[index]
            assert tens.ndim >= 2
            assert current.shape[0] == tens.shape[0]
            assert current.shape[-1] == tens.shape[-1]
            self._update(index, tens, canonicalization=canonicalization)

    def __len__(self):
        """Number of sites"""
        return len(self._ltens)

    def __iter__(self):
        """Use only for read-only access! Do not change arrays in place!

        Subclasses should not override this method because it will
        break basic MPA functionality such as :func:`dot`.

        """
        for ltens in self._ltens:
            yield _roview(ltens)

    def __getitem__(self, index):
        """Return a read-only view on the local tensor at site ``index``"""
        if isinstance(index, slice):
            return (_roview(lten) for lten in self._ltens[index])
        else:
            return _roview(self._ltens[index])

    def __setitem__(self, index, value):
        """Updates the local tensor at site ``index`` with ``value`` while
        assuming ``value`` is not in canonical form.

        Shorthand for ``self.update(index, value)``
        """
        self.update(index, value)

    @property
    def canonical_form(self):
        """Tensors which are currently in left/right-canonical form.

        Returns tuple ``(left, right)`` such that

        - :code:`self[:left]` are left-normalized
        - :code:`self[right:]` are right-normalized.

        """
        return self._lcanonical, self._rcanonical

    @property
    def shape(self):
        """List of tuples with the dimensions of each tensor leg at each site"""
        return tuple(m.shape for m in self._ltens)

    def copy(self):
        """Returns a deep copy of the local tensors"""
        ltens = (lt.copy() for lt in self._ltens)
        return type(self)(ltens, cform=self.canonical_form)
