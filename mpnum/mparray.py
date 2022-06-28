# encoding: utf-8
""" Core MPArray data structure & general purpose functions

  .. todo:: single site MPAs -- what is left?
  .. todo:: Local tensor ownership -- see MPArray class comment
  .. todo:: Possible optimization:

     - replace integer-for loops with iterator (not obviously possible
       everwhere)
     - replace internal structure as list of arrays with lazy generator of
       arrays (might not be possible, since we often iterate both ways!)
     - more in place operations for addition, subtraction, multiplication
  .. todo:: Replace all occurences of self._ltens with self[...] or similar &
      benchmark. This will allow easier transition to lazy evaluation of
      local tensors

"""
from __future__ import absolute_import, division, print_function

import itertools as it
from collections.abc import Iterable, Iterator, Sequence

import mpnum as mp
import numpy as np
from numpy.linalg import qr, svd
from numpy.testing import assert_array_equal
from six.moves import range, zip, zip_longest

from ._named_ndarray import named_ndarray
from .mpstruct import LocalTensors
from .utils import (block_diag, global_to_local, local_to_global, matdot,
                    truncated_svd)

__all__ = ['MPArray', 'dot', 'inject', 'inner', 'local_sum', 'localouter',
           'norm', 'normdist', 'chain', 'partialdot', 'partialtrace',
           'prune', 'regular_slices', 'sandwich', 'embed_slice',
           'trace', 'diag', 'sumup', 'full_rank']


class MPArray(object):
    r"""Efficient representation of a general N-partite array :math:`A` in
    matrix product form with open boundary conditions:

    .. math::
       :label: mpa

       A_{i_1, \ldots, i_N} = A^{[1]}_{i_1} \ldots A^{[N]}_{i_N}

    where the :math:`A^{[k]}` are local tensors (with N legs/dimensions).
    The matrix products in :eq:`mpa` are taken with respect to the left-
    and right-most legs (virtual indices) and the multi-index :math:`i_k`
    corresponds to the true local legs. Open boundary conditions imply that
    :math:`A^{[1]}` is 1-by-something and :math:`A^{[N]}` is something-by-1.

    For the details on the data model used for storing the local tensors see
    :class:`.mpstruct.LocalTensors`.

    .. todo:: As it is now, e.g. :func:`~__imul__()` modifies
              items from ``self._ltens``.  This requires
              e.g. :func:`chain()` to take copies of the local
              tensors.  The data model seems to be that an `MPArray`
              instance owns its local tensors and everyone else,
              including each new `MPArray` instance, must take
              copies. Is this correct?

    .. automethod:: __init__
    .. automethod:: __len__

    """

    def __init__(self, ltens):
        """
        :param ltens: local tensors as instance of
            :class:`.mpstruct.LocalTensors` or simply as a list of
            :class:`numpy.ndarray` in the format described at
            :class:`.mpstruct.LocalTensors`

        """
        self._lt = ltens if isinstance(ltens, LocalTensors) \
            else LocalTensors(ltens)

    def copy(self):
        """Returns a deep copy of the MPA"""
        return type(self)(self._lt.copy())

    def __len__(self):
        """Returns the number of sites"""
        return len(self._lt)

    def get(self, indices, astype=None):
        """Returns the current MPA but with the first index at each sites
        evaluated at the corresponding value of ``indices``

        :param indices: Length ``len(self)`` sequence of index values for
            first physical leg at each site
        :returns: ``type(self)`` object

        """
        assert len(indices) == len(self)
        if astype is None:
            astype = type(self)
        return astype(lt[:, i, ..., :] for i, lt in zip(indices, self._lt))

    @property
    def lt(self):
        return self._lt

    @property
    def size(self):
        """Returns the number of floating point numbers used to represent the
        MPArray

        >>> from .factory import zero
        >>> zero(sites=3, ldim=4, rank=3).lt.shape
        ((1, 4, 3), (3, 4, 3), (3, 4, 1))
        >>> zero(sites=3, ldim=4, rank=3).size
        60

        """
        return sum(lt.size for lt in self._lt)

    @property
    def dtype(self):
        """Returns the dtype that should be returned by ``to_array``"""
        return np.common_type(*tuple(self._lt))

    @property
    def ranks(self):
        """Tuple of ranks"""
        return tuple(m.shape[0] for m in self._lt[1:])

    @property
    def shape(self):
        """List of tuples with the dimensions of each tensor leg at each site"""
        return tuple((m.shape[1:-1]) for m in self._lt)

    @property
    def ndims(self):
        """Tuple of number of legs per site"""
        return tuple(lten.ndim - 2 for lten in self._lt)

    @property
    def canonical_form(self):
        """See :py:attr:`.mpstruct.LocalTensors.canonical_form`"""
        return self._lt.canonical_form

    def dump(self, target):
        """Serializes MPArray to :code:`h5py.Group`. Recover using
        :func:`~load`.

        :param target: :code:`h5py.Group` the instance should be saved to or
            path to h5 file (it's then serialized to /)

        """
        if isinstance(target, str):
            import h5py
            with h5py.File(target, 'w') as outfile:
                return self.dump(outfile)

        for prop in ('ranks', 'shape'):
            # these are only saved for convenience
            target.attrs[prop] = str(getattr(self, prop))

        # these are actually used in MPArray.load
        target.attrs['len'] = len(self)
        target.attrs['canonical_form'] = self.canonical_form

        for site, lten in enumerate(self._lt):
            target[str(site)] = lten

    @classmethod
    def load(cls, source):
        """Deserializes MPArray from :code:`h5py.Group`. Serialize using
        :func:`~dump`.

        :param target: :code:`h5py.Group` containing serialized MPArray or
            path to a single h5 File containing serialized MPArray under /

        """
        if isinstance(source, str):
            import h5py
            with h5py.File(source, 'r') as infile:
                return cls.load(infile)

        ltens = [source[str(i)].value for i in range(source.attrs['len'])]
        return cls(LocalTensors(ltens, cform=source.attrs['canonical_form']))

    @classmethod
    def from_array_global(cls, array, ndims=None, has_virtual=False):
        """Create MPA from array in global form.

        See :func:`mpnum.tools.global_to_local()` for global vs. local form.

        Parameters and return value: See ``from_array()``.
            ``has_virtual=True`` is not supported yet.

        """
        if has_virtual:
            raise NotImplementedError('has_virtual=True not implemeted')
        ndims = ndims if ndims is not None else array.ndim
        assert array.ndim % ndims == 0, \
            "ndims invalid: {} is not multiple of {}".format(array.ndim, ndims)
        sites = array.ndim // ndims
        return cls.from_array(global_to_local(array, sites), ndims, has_virtual)

    @classmethod
    def from_array(cls, array, ndims=None, has_virtual=False):
        """Create MPA from array in local form.

        See :func:`mpnum.tools.global_to_local()` for global
        vs. local form.

        Computes the (exact up to numerical accuracy) representation of `array`
        as MPA with open boundary conditions, i.e. rank 1 at the boundary. This
        is done by factoring off the left and the "physical" legs from the
        rest of the tensor via QR decomposition and working its way through
        the tensor from the left. This yields a left-canonical representation
        of `array`. [:ref:`Sch11 <Sch11>`, Sec. 4.3.1]

        The result is a chain of local tensors with ``ndims`` physical legs at
        each location and has ``array.ndim // ndims`` number of sites (assuming
        ``ndims`` has the same value for each site)

        ``has_virtual`` allows to treat a part of the linear chain of an MPA
        as MPA as well. The rank on the left and right can be different from
        one and different from each other in that case.  This is useful to
        apply SVD compression only to part of an MPA.

        :param np.ndarray array: Dense array with global structure
            ``array[(i0), ..., (iN)]``, i.e. the legs which are factorized into
            the same factor are already adjacent. (For me details see
            :func:`.tools.global_to_local`)
        :param ndims: Number of physical legs per site (default array.ndim)
            or iterable over number of physical legs
        :param bool has_virtual: ``True`` if array already has indices for
            the left and right virtual legs

        """

        ndims = ndims if ndims is not None else array.ndim
        ndims = iter(ndims) if isinstance(ndims, Iterable) else ndims

        if not has_virtual:
            array = array[None, ..., None]
        ltens = _extract_factors(array, ndims=ndims)
        return cls(LocalTensors(ltens, cform=(len(ltens) - 1, len(ltens))))

    @classmethod
    def from_kron(cls, factors):
        """Returns the (exact) representation of an n-fold  Kronecker (tensor)
        product as MPA with ranks 1 and n sites.

        :param factors: A list of arrays with arbitrary number of physical legs
        :returns: The kronecker product of the factors as MPA
        """
        # FIXME Do we still need this or shall we prefer mp.chain?
        return cls(a[None, ..., None] for a in factors)

    def to_array(self):
        """Return MPA as array in local form.

        See :func:`mpnum.tools.global_to_local()` for global
        vs. local form.

        :returns: ndarray of shape :code:`sum(self.shape, ())`

        .. note:: Full arrays can require much more memory than
                  MPAs. (That's why you are using MPAs, right?)

        """
        return _ltens_to_array(iter(self._lt))[0, ..., 0]

    def to_array_global(self):
        """Return MPA as array in global form.

        See :func:`mpnum.tools.global_to_local()` for global
        vs. local form.

        :returns: ndarray of shape :code:`sum(zip(*self.shape, ()))`

        See :func:`to_array()` for more details.

        """
        return local_to_global(self.to_array(), sites=len(self))

    def axis_iter(self, axes=0):
        """Returns an iterator yielding Sub-MPArrays of ``self`` by iterating
        over the specified physical axes.

        **Example:** If ``self`` represents a bipartite (i.e. length 2)
        array with 2 physical dimensions on each site ``A[(k,l), (m,n)]``,
        ``self.axis_iter(0)`` is equivalent to::

            (A[(k, :), (m, :)] for m in range(...) for k in range(...))

        :param axes: Iterable or int specifiying the physical axes to iterate
            over (default 0 for each site)
        :returns: Iterator over :class:`.MPArray`

        """
        if not isinstance(axes, Iterable):
            axes = it.repeat(axes, len(self))

        ltens_iter = it.product(*(iter(np.rollaxis(lten, i + 1))
                                  for i, lten in zip(axes, self.lt)))
        return (MPArray(ltens) for ltens in ltens_iter)

    ##########################
    #  Algebraic operations  #
    ##########################
    @property
    def T(self):
        """Transpose (=reverse order of) physical legs on each site. See also
        :func:`transpose` for more fine grained control.

        """
        ltens = LocalTensors((_local_transpose(tens) for tens in self.lt),
                             cform=self.canonical_form)
        return type(self)(ltens)

    def transpose(self, axes=None):
        """Transpose (=reverse order of) physical legs on each site

        :param axes: New order of the physical axes. If ``None`` is passed,
            we reverse the order of the legs on each site. (default ``None``)

        >>> from .factory import random_mpa
        >>> mpa = random_mpa(2, (2, 3, 4), 2)
        >>> mpa.shape
        ((2, 3, 4), (2, 3, 4))
        >>> mpa.transpose((2, 0, 1)).shape
        ((4, 2, 3), (4, 2, 3))

        """
        ltens = LocalTensors((_local_transpose(tens, axes) for tens in self.lt),
                             cform=self.canonical_form)
        return type(self)(ltens)

    def adj(self):
        """Hermitian adjoint. Equivalent to self.T.conj()."""
        return type(self)([_local_transpose(tens).conjugate()
                           for tens in self.lt])

    def conj(self):
        """Complex conjugate"""
        return type(self)(LocalTensors((ltens.conj() for ltens in self._lt),
                                       cform=self.canonical_form))

    def __add__(self, summand):
        assert len(self) == len(summand), \
            "Length is not equal: {} != {}".format(len(self), len(summand))
        if len(self) == 1:
            # The code below assumes at least two sites.
            return MPArray((self._lt[0] + summand.lt[0],))

        ltens = [np.concatenate((self._lt[0], summand.lt[0]), axis=-1)]
        ltens += [_local_add((l, r))
                  for l, r in zip(self._lt[1:-1], summand.lt[1:-1])]
        ltens += [np.concatenate((self._lt[-1], summand.lt[-1]), axis=0)]
        return MPArray(ltens)

    def __sub__(self, subtr):
        return self + (-1) * subtr

    def __mul__(self, fact):
        """Multiply ``MPArray`` by a scalar.
          .. todo::  These could be made more stable by rescaling all
            non-normalized tens
        """
        if np.isscalar(fact):
            lcanon, rcanon = self.canonical_form
            ltens = self._lt
            ltens_new = it.chain(ltens[:lcanon], [fact * ltens[lcanon]],
                                 ltens[lcanon + 1:])
            return type(self)(LocalTensors(ltens_new, cform=(lcanon, rcanon)))

        raise NotImplementedError("Multiplication by non-scalar not supported")

    def __imul__(self, fact):
        if np.isscalar(fact):
            lcanon, _ = self.canonical_form
            # FIXME TEMPORARY FIX
            #  self._lt[lcanon] *= fact
            self._lt.update(lcanon, self._lt[lcanon] * fact)
            return self

        raise NotImplementedError("Multiplication by non-scalar not supported")

    def __rmul__(self, fact):
        return self.__mul__(fact)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __truediv__(self, divisor):
        if np.isscalar(divisor):
            return self.__mul__(1 / divisor)
        raise NotImplementedError("Division by non-scalar not supported")

    def __itruediv__(self, divisor):
        if np.isscalar(divisor):
            return self.__imul__(1 / divisor)
        raise NotImplementedError("Division by non-scalar not supported")

    def sum(self, axes=None):
        """Element-wise sum over physical legs

        :param axes: Physical legs to sum over

        axes can have the following values:

        * Sequence of length zero: Sum over nothing

        * Sequence of (sequences or None): ``axes[i]`` specifies the
          physical legs to sum over at site ``i``; ``None`` sums over all
          physical legs at a site
        * Sequence of integers: ``axes`` specifies the physical legs to
          sum over at each site
        * Single integer: Sum over physical leg ``axes`` at each site
        * ``None``: Sum over all physical legs at each site

        To not sum over any axes at a certain site, specify the empty
        sequence for that site.

        """
        if axes is None:
            axes = it.repeat(axes)
        else:
            if not hasattr(axes, '__iter__'):
                axes = (axes,)  # Single integer
            axes = tuple(axes)
            if len(axes) == 0 or not (axes[0] is not None
                                      and hasattr(axes[0], '__iter__')):
                axes = it.repeat(axes)  # Sum over same physical legs everywhere
            else:
                assert len(axes) == len(self)
        axes = (tuple(range(1, ndims + 1)) if ax is None
                else tuple(a + 1 for a in ax)
                for ax, ndims in zip(axes, self.ndims))
        out = type(self)(lt.sum(ax) if ax else lt
                         for ax, lt in zip(axes, self.lt))
        if sum(out.ndims) == 0:
            out = out.to_array()
        return out

    ################################
    #  Shape changes, conversions  #
    ################################
    def reshape(self, newshapes):
        """Reshape physical legs in place.

        Use :py:attr:`~shape` to obtain the shape of the physical legs.

        :param newshapes: A single new shape or a list of new shape.
            Alternatively, you can pass 'prune' to get rid of all legs
            of dimension 1.
        :returns: Reshaped MPA

        .. todo:: Why is this here? What's wrong with the purne function?

        """
        if newshapes == 'prune':
            newshapes = (tuple(s for s in pdim if s > 1) for pdim in self.shape)

        newshapes = tuple(newshapes)
        if not isinstance(newshapes[0], Iterable):
            newshapes = it.repeat(newshapes, times=len(self))

        ltens = [_local_reshape(lten, newshape)
                 for lten, newshape in zip(self._lt, newshapes)]
        return MPArray(LocalTensors(ltens, cform=self.canonical_form))

    def ravel(self):
        """Flatten the MPA to an MPS, shortcut for ``self.reshape((-1,))``"""
        return self.reshape((-1,))

    def group_sites(self, sites_per_group):
        """Group several MPA sites into one site.

        The resulting MPA has length ``len(self) // sites_per_group`` and
        ``sites_per_group * self.ndims[i]`` physical legs on site ``i``. The
        physical legs on each sites are in local form.

        :param int sites_per_group: Number of sites to be grouped into one
        :returns: An MPA with ``sites_per_group`` fewer sites and more ndims

        """
        if (len(self) % sites_per_group) != 0:
            raise ValueError('Cannot group: {} not a multiple of {}'
                             .format(len(self), sites_per_group))

        if sites_per_group == 1:
            return self
        ltens = [_ltens_to_array(self._lt[i:i + sites_per_group])
                 for i in range(0, len(self), sites_per_group)]
        return MPArray(ltens)

    def split_sites(self, sites_per_group):
        """Split MPA sites into several sites.

        The resulting MPA has length ``len(self) * sites_per_group`` and
        ``self.ndims[i] // sites_per_group`` indices on site i.

        :param int sites_per_group: Split each site in that many sites
        :returns: An mpa with ``sites_per_group`` more sites and fewer
            ``ndims``

        """
        ltens = []
        for i in range(len(self)):
            ndims = self.ndims[i]
            assert (ndims % sites_per_group) == 0, \
                'ndims not a multiple of sites_per_group'
            ltens += _extract_factors(self._lt[i], ndims // sites_per_group)
        return MPArray(ltens)

    def vleg2leg(self, pos):
        """Transforms the virtual leg between site ``pos`` and ``pos + 1`` into
        local legs at those sites. The new leg will be the rightmost one
        at site ``pos`` and the leftmost one at site ``pos + 1``. The new
        rank is 1.

        Also see :func:`leg2vleg`.

        :param pos: Number of the virtual to perform the transformation
        :returns: MPA with transformed virtual

        .. todo:: More appropriate naming for this functions?

        """
        ltens = list(self._lt)
        ltens[pos] = ltens[pos][..., None]
        ltens[pos + 1] = ltens[pos + 1][None]

        lcanon, rcanon = self.canonical_form
        new_normal_form = min(lcanon, pos), max(rcanon, pos + 2)
        return MPArray(LocalTensors(ltens, cform=new_normal_form))

    def leg2vleg(self, pos):
        """Performs the inverse operation to :func:`vleg2leg`.

        :param pos: Number of the virtual to perform the transformation
        :returns: read-only MPA with transformed virtual

        .. todo:: More appropriate naming for this functions?

        """
        ltens = list(self._lt)
        assert ltens[pos].shape[-1] == 1
        assert ltens[pos + 1].shape[0] == 1
        ltens[pos] = ltens[pos][..., 0]
        ltens[pos + 1] = ltens[pos + 1][0]

        lcanon, rcanon = self.canonical_form
        new_normal_form = min(lcanon, pos), max(rcanon, pos + 1)
        return MPArray(LocalTensors(ltens, cform=new_normal_form))

    def split(self, pos):
        """Splits the MPA into two by transforming the virtual legs into
        local legs according to :func:`vleg2leg`.

        :param pos: Number of the virtual to perform the transformation
        :returns: (mpa_left, mpa_right)

        """
        if pos < 0:
            return None, self
        elif pos >= len(self):
            return self, None

        mpa_t = self.vleg2leg(pos)
        lnorm, rnorm = mpa_t.canonical_form

        ltens_l = LocalTensors(it.islice(mpa_t.lt, 0, pos + 1),
                               cform=(min(lnorm, pos), min(rnorm, pos + 1)))
        ltens_r = LocalTensors(it.islice(mpa_t.lt, pos + 1, len(mpa_t)),
                               cform=(max(0, lnorm - pos), max(0, rnorm - pos - 1)))
        return type(self)(ltens_l), type(self)(ltens_r)

    def reverse(self):
        return type(self)(
            lt.transpose([lt.ndim - 1] + list(range(1, lt.ndim - 1)) + [0])
            for lt in self.lt[::-1]
        )

    ################################
    #  Normalizaton & Compression  #
    ################################
    def canonicalize(self, left=None, right=None):
        """Brings the MPA to canonical form in place
        [:ref:`Sch11 <Sch11>`, Sec. 4.4]

        Note that we do not support full left- or right-canonicalization. In
        general, the right- (left- resp.)most local tensor cannot be in a
        canonical form since at least one local tensor must be non-normalized.

        The following values for `left` and `right` will be needed
        most frequently:

        +---------------+--------------+-----------------------+
        | Left-/Right-  | Do Nothing   | To canonicalize       |
        | canonicalize: |              | maximally             |
        +===============+==============+=======================+
        | `left`        | :code:`None` | :code:`'afull'`,      |
        |               |              | :code:`len(self) - 1` |
        +---------------+--------------+-----------------------+
        | `right`       | :code:`None` | :code:`'afull'`,      |
        |               |              | :code:`1`             |
        +---------------+--------------+-----------------------+

        :code:`'afull'` is short for "almost full" (we do not support
        normalizing the outermost sites).

        Arbitrary integer values of ``left`` and ``right`` have the
        following meaning:

        - :code:`self[:left]` will be left-normalized

        - :code:`self[right:]` will be right-normalized

        In accordance with the last table, the special values
        :code:`None` and :code:`'afull'` will be replaced by the
        following integers:

        +---------+-------------------+-----------------------+
        |         | :code:`None`      | :code:`'afull'`       |
        +---------+-------------------+-----------------------+
        | `left`  | :code:`0`         | :code:`len(self) - 1` |
        +---------+-------------------+-----------------------+
        | `right` | :code:`len(self)` | :code:`1`             |
        +---------+-------------------+-----------------------+

        Exceptions raised:

        - Integer argument too large or small: ``IndexError``

        - Matrix would be both left- and right-normalized: ``ValueError``

        """
        current_lcanon, current_rcanon = self.canonical_form
        if left is None and right is None:
            if current_lcanon < len(self) - current_rcanon:
                self._lcanonicalize(1)
            else:
                self._rcanonicalize(len(self) - 1)
            return

        # Fill the special values for `None` and 'afull'.
        target_lcanon = {None: 0, 'afull': len(self) - 1}.get(left, left)
        target_rcanon = {None: len(self), 'afull': 1}.get(right, right)
        # Support negative indices.
        if target_lcanon < 0:
            target_lcanon += len(self)
        if target_rcanon < 0:
            target_rcanon += len(self)
        # Perform range checks.
        if not 0 <= target_lcanon <= len(self):
            raise IndexError('len={!r}, left={!r}'.format(len(self), left))
        if not 0 <= target_rcanon <= len(self):
            raise IndexError('len={!r}, right={!r}'.format(len(self), right))

        if not target_lcanon < target_rcanon:
            raise ValueError("Canonicalization {}:{} invalid"
                             .format(target_lcanon, target_rcanon))
        if current_lcanon < target_lcanon:
            self._rcanonicalize(target_lcanon)
        if current_rcanon > target_rcanon:
            self._lcanonicalize(target_rcanon)

    def _rcanonicalize(self, to_site):
        """Left-canonicalizes all local tensors _ltens[:to_site] in place

        :param to_site: Index of the site up to which canonicalization is to be
            performed

        """
        assert 0 <= to_site < len(self), 'to_site={!r}'.format(to_site)

        lcanon, rcanon = self._lt.canonical_form
        for site in range(lcanon, to_site):
            ltens = self._lt[site]
            q, r = qr(ltens.reshape((-1, ltens.shape[-1])))
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting rank here
            newtens = (q.reshape(ltens.shape[:-1] + (-1,)),
                       matdot(r, self._lt[site + 1]))
            self._lt.update(slice(site, site + 2), newtens,
                            canonicalization=('left', None))

    def _lcanonicalize(self, to_site):
        """Right-canonicalizes all local tensors _ltens[to_site:] in place

        :param to_site: Index of the site up to which canonicalization is to be
            performed

        """
        assert 0 < to_site <= len(self), 'to_site={!r}'.format(to_site)

        lcanon, rcanon = self.canonical_form
        for site in range(rcanon - 1, to_site - 1, -1):
            ltens = self._lt[site]
            q, r = qr(ltens.reshape((ltens.shape[0], -1)).T)
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting rank here
            newtens = (matdot(self._lt[site - 1], r.T),
                       q.T.reshape((-1,) + ltens.shape[1:]))
            self._lt.update(slice(site - 1, site + 1), newtens,
                            canonicalization=(None, 'right'))

    def compress(self, method='svd', **kwargs):
        r"""Compress ``self``, modifying it in-place.

        Let :math:`\vert u \rangle` the original vector and let
        :math:`\vert c \rangle` the compressed vector. The
        compressions we return have the property (cf. [:ref:`Sch11 <Sch11>`,
        Sec. 4.5.2])

        .. math::

           \langle u \vert c \rangle = \langle c \vert c \rangle \in (0, \infty).

        It is a useful property because it ensures

        .. math::

           \min_{\phi \in \mathbb R} \| u - r e^{i \phi} c \| &= \| u - r c \|,
           \quad r > 0, \\
           \min_{\mu \in \mathbb C} \| u - \mu c \| &= \| u - c \|

        for the vector 2-norm. Users of this function can compute norm
        differences between u and a normalized c via

        .. math::

           \| u - r c \|^2 = \| u \|^2 + r (r - 2) \langle u \vert c \rangle,
           \quad r \ge 0.

        In the special case of :math:`\|u\| = 1` and :math:`c_0 = c/\| c
        \|` (pure quantum states as MPS), we obtain

        .. math::

           \| u - c_0 \|^2 = 2(1 - \sqrt{\langle u \vert c \rangle})

        :returns: Inner product :math:`\langle u \vert c \rangle \in
            (0, \infty)` of the original u and its compression c.

        :param method: ``'svd'`` or ``'var'``

        .. rubric:: Parameters for ``'svd'``:

        :param rank: Maximal rank of the result. (default: ``None``)

        :param relerr: Maximal fraction of discarded singular values.
            Default ``0``.  If both rank and relerr are given, the
            smaller resulting rank is used.

        :param direction: ``'right'`` (sweep from left to right), ``'left'``
            (inverse) or ``None`` (choose depending on
            canonicalization). (default: ``None``)

        :param canonicalize: SVD compression works best when the MPA is
            brought into full left-/right-cannonical form first. This variable
            determines whether cannonical form is enforced before compression
            (default: ``True``)

        :param svdfunc: Which SVD function to use during the compression.
            It should follow the conventios of
            :func:`~.utils.extmath.truncated_svd()`, which is also the
            default choice. In some circumstances, a partial SVD as provided
            by :func:`scipy.sparse.linalg.svds()` or a randomized SVD such as
            :func:`~.utils.extmath.randomized_svd()` might speed up
            computations with no or little loss of accuracy.

        .. rubric:: Parameters for ``'var'``:

        :param rank: Maximal rank for the result. Either
            ``startmpa`` or ``rank`` is required.

        :param num_sweeps: Number of variational sweeps (required).

        :param startmpa: Start vector, also fixes the rank
            of the result. Default: Random, with same norm as self.

        :param randstate: ``numpy.random.RandomState`` instance used for
            random start vector. (default: ``numpy.random``).

        :param var_sites: Number of connected sites to be varied
            simultaneously (default 1)

        Increasing ``var_sites`` makes it less likely to get stuck in a
        local minimum but is generally slower.

        References:

        * ``'svd'``: Singular value truncation, [:ref:`Sch11 <Sch11>`, Sec. 4.5.1]
        * ``'var'``: Variational compression, [:ref:`Sch11 <Sch11>`, Sec. 4.5.2]

        """
        if method == 'svd':
            return self._compress_svd(**kwargs)
        elif method == 'var':
            compr, overlap = self._compression_var(**kwargs)
            self._lt = compr._lt
            return overlap
        else:
            raise ValueError('{!r} is not a valid method'.format(method))

    def compression(self, method='svd', **kwargs):
        """Return a compression of ``self``. Does not modify ``self``.

        Parameters: See :func:`~compress()`.

        :returns: ``(compressed_mpa, overlap)`` where ``overlap`` is the inner
            product returned by :func:`~compress()`.

        """
        if method == 'svd':
            target = self.copy()
            overlap = target._compress_svd(**kwargs)
            return target, overlap
        elif method == 'var':
            return self._compression_var(**kwargs)
        else:
            raise ValueError('{!r} is not a valid method'.format(method))

    def _compress_svd(self, rank=None, relerr=None, direction=None,
                      canonicalize=True, svdfunc=truncated_svd):
        """Compress `self` using SVD [:ref:`Sch11 <Sch11>`, Sec. 4.5.1]

        Parameters: See :func:`~compress()`.

        """
        if len(self) == 1:
            # Cannot do anything. Return perfect overlap.
            return norm(self)**2

        ln, rn = self.canonical_form
        default_direction = 'left' if len(self) - rn > ln else 'right'
        direction = default_direction if direction is None else direction
        rank = max(self.ranks) if rank is None else rank

        if direction == 'right':
            if canonicalize:
                self.canonicalize(right=1)
            for item in self._compress_svd_r(rank, relerr, svdfunc):
                pass
            return item
        elif direction == 'left':
            if canonicalize:
                self.canonicalize(left=len(self) - 1)
            for item in self._compress_svd_l(rank, relerr, svdfunc):
                pass
            return item

        raise ValueError('{} is not a valid direction'.format(direction))

    def _compression_var(self, num_sweeps, startmpa=None, rank=None,
                         randstate=np.random, var_sites=2):
        """Return a compression of ``self`` using variational compression
        [:ref:`Sch11 <Sch11>`, Sec. 4.5.2]

        Parameters and return value: See :func:`~compression()`.

        """
        if len(self) == 1:
            # Cannot do anything. We make a copy, see below.
            copy = self.copy()
            return copy, norm(copy)**2

        if startmpa is not None:
            rank = max(startmpa.ranks)
        elif rank is None:
            raise ValueError('You must provide startmpa or rank')
        if rank > max(self.ranks):
            # The caller expects that the result is independent from
            # `self`. Take a copy. If we are called from .compress()
            # instead of .compression(), we could avoid the copy and
            # return self.
            copy = self.copy()
            return copy, norm(copy)**2

        if startmpa is None:
            from mpnum.factory import random_mpa
            compr = random_mpa(len(self), self.shape, rank, randstate=randstate,
                               dtype=self.dtype)
        else:
            compr = startmpa.copy()
            assert all(d1 == d2 for d1, d2 in zip(self.shape, compr.shape))

        # flatten the array since MPS is expected & bring back
        shape = self.shape
        compr = compr.ravel()
        overlap = compr._adapt_to(self.ravel(), num_sweeps, var_sites)
        compr = compr.reshape(shape)
        return compr, overlap

    def _compress_svd_l(self, rank, relerr, svdfunc):
        """Compresses the MPA in place from right to left using SVD;
        yields a right-canonical state

        See :func:`~compress` for more details and arguments.

        """
        assert rank > 0, "Cannot compress to rank={}".format(rank)
        assert (relerr is None) or ((0. <= relerr) and (relerr <= 1.)), \
            "relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1, 0, -1):
            ltens = self._lt[site]
            matshape = (ltens.shape[0], -1)
            if relerr is None:
                u, sv, v = svdfunc(ltens.reshape(matshape), rank)
                rank_t = len(sv)
            else:
                u, sv, v = svd(ltens.reshape(matshape))
                svsum = np.cumsum(sv) / np.sum(sv)
                rank_relerr = np.searchsorted(svsum, 1 - relerr) + 1
                rank_t = min(ltens.shape[0], v.shape[0], rank, rank_relerr)

            yield sv, rank_t

            newtens = (matdot(self._lt[site - 1], u[:, :rank_t] * sv[None, :rank_t]),
                       v[:rank_t, :].reshape((rank_t, ) + ltens.shape[1:]))
            self._lt.update(slice(site - 1, site + 1), newtens,
                            canonicalization=(None, 'right'))

        yield np.sum(np.abs(self._lt[0])**2)

    def _compress_svd_r(self, rank, relerr, svdfunc):
        """Compresses the MPA in place from left to right using SVD;
        yields a left-canonical state

        See :func:`~compress` for parameters
        """
        assert rank > 0, "Cannot compress to rank={}".format(rank)
        assert (relerr is None) or ((0. <= relerr) and (relerr <= 1.)), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1):
            ltens = self._lt[site]
            matshape = (-1, ltens.shape[-1])
            if relerr is None:
                u, sv, v = svdfunc(ltens.reshape(matshape), rank)
                rank_t = len(sv)
            else:
                u, sv, v = svd(ltens.reshape(matshape))
                svsum = np.cumsum(sv) / np.sum(sv)
                rank_relerr = np.searchsorted(svsum, 1 - relerr) + 1
                rank_t = min(ltens.shape[-1], u.shape[1], rank, rank_relerr)

            yield sv, rank_t

            newtens = (u[:, :rank_t].reshape(ltens.shape[:-1] + (rank_t, )),
                       matdot(sv[:rank_t, None] * v[:rank_t, :], self._lt[site + 1]))
            self._lt.update(slice(site, site + 2), newtens,
                            canonicalization=('left', None))

        yield np.sum(np.abs(self._lt[-1])**2)

    def singularvals(self):
        """Return singular values of ``self`` for all bipartitions

        :returns: Iterate over bipartitions with 1, 2, ... len(self) -
            1 sites on the left hand side. Yields a ``np.ndarray`` containing
            singular values for each bipartition.

        .. note:: May decrease the rank (without changing the represented
            tensor).

        """
        if len(self) == 1:
            return  # No bipartitions with two non-empty parts for a single site
        self.canonicalize(right=1)
        iterator = self._compress_svd_r(max(self.ranks), None, truncated_svd)
        # We want everything from the iterator except for the last element.
        for _, (sv, rank) in zip(range(len(self) - 1), iterator):
            # We could verify that `rank` did not decrease but it may
            # decrease because of zero singular values -- let's trust
            # that relerr=0.0 behaves as intended.
            # assert old_rank == rank
            yield sv

    def pad_ranks(self, rank=None, force_rank=False):
        """Increase rank by padding with zeros

        This function is useful to prepare initial states for
        variational compression. E.g. for a five-qubit pure state with
        ranks `(2, 2, 4, 2)` it is desirable to increase the
        ranks to `(2, 4, 4, 2)` before using it as an
        initial state for variational compression.

        :param int rank: Increase rank to this value, use ``max(self.rank)``
            if ``None`` (default: ``None``)
        :param force_rank: Use full rank even at the beginning and
            end of the MPS. See :func:`~mpnum.mparray.full_rank` for more
            details. (default: ``False``)
        :returns: MPA representation of the same array with padded rank

        """
        if rank is None:
            rank = max(self.ranks)
        ranks = it.repeat(rank)
        if not force_rank:
            ranks = [min(f, b) for f, b in zip(full_rank(self.shape), ranks)]
        pad = [max(s, b) - s for s, b in zip(self.ranks, ranks)]
        lt = (np.pad(lt, [(0, lp)] + [(0, 0)] * (lt.ndim - 2) + [(0, rp)],
                     'constant')
              for lp, rp, lt in zip([0] + pad, pad + [0], self.lt))
        return mp.MPArray(lt)

    def _adapt_to(self, target, num_sweeps, var_sites):
        """Iteratively minimize the l2 distance between `self` and `target`.
        This is especially important for variational compression, where `self`
        is the initial guess and target the MPA to be compressed.

        :param target: MPS to compress; i.e. MPA with only one
            physical leg per site

        Other parameters and references: See
        :func:`~compress()`.

        .. todo:: Possible improvements:
            - Can we refactor this function into several shorter functions?
            - track overlap between 'compr' and 'target' and stop sweeping if it
            is small
            - maybe increase rank of given error cannot be reached
            - Shall we track the error in the SVD truncation for multi-site
            updates? [Sch11]_ says it turns out to be useful in actual DMRG.
            - return these details for tracking errors in larger computations

        """
        # For
        #
        #   pos in range(nr_sites - var_sites),
        #
        # we increase the overlap between `self` and `target` by
        # varying the local tensors of `self` on sites
        #
        #   range(pos, pos_end),  pos_end = pos + var_sites
        #
        # lvecs[pos] and rvecs[pos] contain the vectors needed to
        # obtain the new local tensors. Therefore, lvecs[pos] is
        # constructed from matrices on
        #
        #   range(0, pos - 1)
        #
        # and rvecs[pos] is constructed from matrices on
        #
        #   range(pos_end, nr_sites),  pos_end = pos + var_sites
        assert_array_equal(self.ndims, 1, "Self is not a MPS")
        assert_array_equal(target.ndims, 1, "Target is not a MPS")

        nr_sites = len(target)
        lvecs = [np.array(1, ndmin=2)] + [None] * (nr_sites - var_sites)
        rvecs = [None] * (nr_sites - var_sites) + [np.array(1, ndmin=2)]
        self.canonicalize(right=1)
        for pos in reversed(range(nr_sites - var_sites)):
            pos_end = pos + var_sites
            rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self._lt[pos_end],
                                         target.lt[pos_end])

        # Example: For `num_sweeps = 3`, `nr_sites = 3` and `var_sites
        # = 1`, we want the following sequence for `pos`:
        #
        #    0    1    2    1    0    1    2    1    0    1    2    1    0
        #    \_________/    \____/    \____/    \____/    \____/    \____/
        #        LTR         RTL       LTR       RTL       LTR       RTL
        #    \___________________/    \______________/    \______________/
        #     num_sweep = 0            num_sweep = 1       num_sweep = 1

        max_rank = max(self.ranks)
        for num_sweep in range(num_sweeps):
            # Sweep from left to right (LTR)
            for pos in range(nr_sites - var_sites + 1):
                if pos == 0 and num_sweep > 0:
                    # Don't do first site again if we are not in the first sweep.
                    continue
                if pos > 0:
                    self.canonicalize(left=pos)
                    rvecs[pos - 1] = None
                    lvecs[pos] = _adapt_to_add_l(lvecs[pos - 1], self._lt[pos - 1],
                                                 target.lt[pos - 1])
                pos_end = pos + var_sites
                new_ltens = _adapt_to_new_lten(lvecs[pos], target.lt[pos:pos_end],
                                               rvecs[pos], max_rank)
                self._lt[pos:pos_end] = new_ltens

            # Sweep from right to left (RTL; don't do `pos = nr_sites
            # - var_sites` again)
            for pos in reversed(range(nr_sites - var_sites)):
                pos_end = pos + var_sites
                if pos < nr_sites - var_sites:
                    # We always do this, because we don't do the last site again.
                    self.canonicalize(right=pos_end)
                    lvecs[pos + 1] = None
                    rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self._lt[pos_end],
                                                 target.lt[pos_end])

                new_ltens = _adapt_to_new_lten(lvecs[pos], target.lt[pos:pos_end],
                                               rvecs[pos], max_rank)
                self._lt[pos:pos_end] = new_ltens

        # Let u the uncompressed vector and c the compression which we
        # return. c satisfies <c|c> = <u|c> (mentioned more or less in
        # [Sch11]). We compute <c|c> to get <u|c> and use the
        # canonicalization of the state to compute <c|c> (e.g. [Sch11,
        # Fig. 24]).
        return norm(self)**2


#############################################
#  General functions to deal with MPArrays  #
#############################################
def dot(mpa1, mpa2, axes=(-1, 0), astype=None):
    """Compute the matrix product representation of the contraction of ``a``
        and ``b`` over the given axes. [:ref:`Sch11 <Sch11>`, Sec. 4.2]

    :param mpa1, mpa2: Factors as MPArrays

    :param axes: Tuple ``(ax1, ax2)`` where ``ax1`` (``ax2``) is a single
        physical leg number or sequence of physical leg numbers
        referring to ``mpa1`` (``mpa2``). The first (second, etc) entries
        of ``ax1`` and ``ax2`` will be contracted. Very similar to the
        ``axes`` argument for :func:`numpy.tensordot()`.
        (default: ``(-1, 0)``)

    .. note:: Note that the default value of ``axes`` is different compared to
              :func:`numpy.tensordot`.

    :param astype: Return type. If ``None``, use the type of ``mpa1``

    :returns: Dot product of the physical arrays

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))

    # adapt the axes from physical to true legs
    if isinstance(axes[0], Sequence):
        axes = tuple(tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes2)
                     for axes2 in axes)
    else:
        axes = tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes)

    ltens = [_local_dot(l, r, axes) for l, r in zip(mpa1.lt, mpa2.lt)]

    if astype is None:
        astype = type(mpa1)
    return astype(ltens)


def sumup(mpas, weights=None):
    """Returns the sum of the MPArrays in ``mpas``. Same as

    .. code-block:: python

        functools.reduce(mp.MPArray.__add__, mpas)

    but should be faster as we can get rid of intermediate allocations.

    :param mpas: Iterator over :class:`~MPArray`
    :returns: Sum of ``mpas``

    """
    mpas = list(mpas)
    length = len(mpas[0])
    assert all(len(mpa) == length for mpa in mpas)

    if length == 1:
        if weights is None:
            return MPArray([sum(mpa.lt[0] for mpa in mpas)])
        else:
            return MPArray([sum(w * mpa.lt[0] for w, mpa in zip(weights, mpas))])

    ltensiter = [iter(mpa.lt) for mpa in mpas]
    if weights is None:
        ltens = [np.concatenate([next(lt) for lt in ltensiter], axis=-1)]
    else:
        ltens = [np.concatenate([w * next(lt)
                                 for w, lt in zip(weights, ltensiter)], axis=-1)]
    ltens += [_local_add([next(lt) for lt in ltensiter])
              for _ in range(length - 2)]
    ltens += [np.concatenate([next(lt) for lt in ltensiter], axis=0)]

    return MPArray(ltens)


def partialdot(mpa1, mpa2, start_at, axes=(-1, 0)):
    """Partial dot product of two MPAs of inequal length.

    The shorter MPA will start on site ``start_at``. Local dot products
    will be carried out on all sites of the shorter MPA. Other sites
    will remain unmodified.

    mpa1 and mpa2 can also have equal length if ``start_at == 0``. In
    this case, we do the same as :func:`dot()`.

    :param mpa1, mpa2: Factors as MPArrays, length must be inequal.
    :param start_at: The shorter MPA will start on this site.
    :param axes: See ``axes`` argument to :func:`dot()`.
    :returns: MPA with length of the longer MPA.

    """
    # adapt the axes from physical to true legs
    if isinstance(axes[0], Sequence):
        axes = tuple(tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes2)
                     for axes2 in axes)
    else:
        axes = tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes)

    # Make the MPAs equal length (in fact, the shorter one will be
    # infinite length, but that's fine because we use zip()).
    shorter = mpa1 if len(mpa1) < len(mpa2) else mpa2
    shorter = it.chain(it.repeat(None, times=start_at), shorter.lt,
                       it.repeat(None))
    if len(mpa1) < len(mpa2):
        mpa1_lt = shorter
        mpa2_lt = mpa2.lt
    else:
        mpa1_lt = mpa1.lt
        mpa2_lt = shorter

    ltens_new = (
        l if r is None else (r if l is None else _local_dot(l, r, axes))
        for l, r in zip(mpa1_lt, mpa2_lt)
    )
    return MPArray(ltens_new)


# NOTE: I think this is a nice example how we could use Python's generator
#       expression to implement lazy evaluation of the matrix product structure
#       which is the whole point of doing this in the first place
def inner(mpa1, mpa2):
    """Compute the inner product `<mpa1|mpa2>`. Both have to have the same
    physical dimensions. If these represent a MPS, ``inner(...)`` corresponds
    to the canoncial Hilbert space scalar product. If these represent a MPO,
    ``inner(...)`` corresponds to the Frobenius scalar product (with Hermitian
    conjugation in the first argument)

    :param mpa1: MPArray with same number of physical legs on each site
    :param mpa2: MPArray with same physical shape as mpa1
    :returns: <mpa1|mpa2>

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))
    ltens_new = (_local_dot(_local_ravel(l).conj(), _local_ravel(r), axes=(1, 1))
                 for l, r in zip(mpa1.lt, mpa2.lt))
    return _ltens_to_array(ltens_new)[0, ..., 0]


def sandwich(mpo, mps, mps2=None):
    """Compute ``<mps|MPO|mps>`` efficiently

    This function computes the same value as ``mp.inner(mps,
    mp.dot(mpo, mps))`` in a more efficient way. 

    The runtime of this method scales with ``D**3 * Dp + D**2 * Dp**3`` 
    where ``D`` and ``Dp`` are the ranks of ``mps`` and
    ``mpo``. This is more efficient than ``mp.inner(mps, mp.dot(mpo,
    mps))``, whose runtime scales with ``D**4 * Dp**3``, and also more
    efficient than ``mp.dot(mps.conj(), mp.dot(mpo,
    mps)).to_array()``, whose runtime scales with ``D**6 * Dp**3``.

    If ``mps2`` is given, ``<mps2|MPO|mps>`` is computed instead
    (i.e. ``mp.inner(mps2, mp.dot(mpo, mps))``; see also :func:`dot()`).

    """
    # Fortunately, the contraction has been implemented already:
    arr = np.ones((1, 1, 1))
    for mpo_lt, mps_lt, mps2_lt in zip(mpo.lt, mps.lt, (mps2 or mps).lt):
        arr = mp.linalg._eig_leftvec_add(arr, mpo_lt, mps_lt, mps2_lt)
    assert arr.size == 1
    return arr.flat[0]


def chain(mpas, astype=None):
    """Computes the tensor product of MPAs given in ``*args`` by adding more
    sites to the array.

    :param mpas: Iterable of MPAs in the order as they should appear in the
        chain
    :param astype: dtype of the returned MPA. If ``None``, use the type of the
        first MPA.
    :returns: MPA of length ``len(args[0]) + ... + len(args[-1])``

    .. todo:: Make this canonicalization aware
    .. todo:: Raise warning when casting complex to real dtype

    """
    mpas = iter(mpas)
    try:
        first = next(mpas)
    except StopIteration:
        raise ValueError('Argument `mpas` is an empty list')
    rest = (lt for mpa in mpas for lt in mpa.lt)
    if astype is None:
        astype = type(first)
    return astype(it.chain(first.lt, rest))


def diag(mpa, axis=0):
    """Returns the diagonal elements :code:`mpa[i, i, ..., i]`. If :code:`mpa`
    has more than one physical dimension, the result is a numpy array with
    :code:`MPArray` entries, otherwise its a numpy array with floats.

    :param mpa: :class:`MPArray` with shape > :code:`axis`
    :param axis: The physical index to take diagonals over
    :returns: Array containing the diagonal elements (each diagonal
        element is an :class:`MPArray` with the physical dimension
        reduced by one, note that an :class:`MPArray` with
        dimension 0 is a simple number)

    """
    dim = mpa.shape[0][axis]
    # work around http://bugs.python.org/issue21161
    try:
        valid_axis = [d[axis] == dim for d in mpa.shape]
        assert all(valid_axis)
    except NameError:
        pass
    ndims = mpa.ndims[0]
    assert all(p == ndims for p in mpa.ndims)

    slices = ((slice(None),) * (axis + 1) + (i,) for i in range(dim))
    mpas = [MPArray(ltens[s] for ltens in mpa.lt) for s in slices]

    if len(mpa.shape[0]) == 1:
        return np.array([mpa.to_array() for mpa in mpas])
    else:
        return np.array(mpas, dtype=object)


def inject(mpa, pos, num=None, inject_ten=None):
    """Interleaved chain product of an MPA and a rank 1 MPA

    Return the chain product between mpa and ``num`` copies of the local
    tensor ``inject_ten``, but place the copies of ``inject_ten`` before
    site ``pos`` inside or outside ``mpa``. You can also supply ``num =
    None`` and a sequence of local tensors. All legs of the local
    tensors are interpreted as physical legs. Placing the local
    tensors at the beginning or end of ``mpa`` using ``pos = 0`` or ``pos =
    len(mpa)`` is also supported, but :func:`chain()` is preferred for
    that as it is a much simpler function.

    If ``inject_ten`` is omitted, use a square identity matrix of size
    ``mpa.shape[pos][0]``. If ``pos == len(mpa)``, ``mpa.shape[pos - 1][0]``
    will be used for the size of the matrix.

    :param mpa: An MPA.
    :param pos: Inject sites into the MPA before site ``pos``.
    :param num: Inject ``num`` copies. Can be ``None``; in this case
        ``inject_ten`` must be a sequence of values.
    :param inject_ten: Physical tensor to inject (if omitted, an
        identity matrix will be used; cf. above)
    :returns: The chain product

    ``pos`` can also be a sequence of positions. In this case, ``num`` and
    ``inject_ten`` must be either sequences or ``None``, where ``None`` is
    interpreted as ``len(pos) * [None]``. As above, if ``num[i]`` is
    ``None``, then ``inject_ten[i]`` must be a sequence of values.

    """
    if isinstance(pos, Iterable):
        pos = tuple(pos)
        num = (None,) * len(pos) if num is None else tuple(num)
        inject_ten = ((None,) * len(pos)
                      if inject_ten is None else tuple(inject_ten))
    else:
        pos, num, inject_ten = (pos,), (num,), (inject_ten,)
    assert len(pos) == len(num) == len(inject_ten)
    assert not any(n is None and hasattr(tens, 'shape')
                   for n, tens in zip(num, inject_ten)), \
        """num[i] is None requires a list of tensors at inject_ten[i]"""
    assert all(begin < end for begin, end in zip(pos[:-1], pos[1:]))
    pos = (0,) + pos + (len(mpa),)
    pieces = tuple(tuple(mpa.lt[begin:end])
                   for begin, end in zip(pos[:-1], pos[1:]))
    ranks = (l[-1].shape[-1] if l else 1 for l in pieces[:-1])
    pdims = (r[0].shape[1] if r else mpa.lt[-1].shape[1] for r in pieces[1:])
    inject_ten = (
        (
            np.rollaxis(np.tensordot(
                np.eye(pdim) if ten is None else ten,
                np.eye(rank), axes=((), ())), -1)
            for ten in (inj if n is None else (inj,) * n)
        )
        for rank, pdim, n, inj in zip(ranks, pdims, num, inject_ten)
    )
    ltens = (lt for ltens in zip(pieces, inject_ten) for lten in ltens
             for lt in lten)
    ltens = it.chain(ltens, pieces[-1])
    return MPArray(ltens)


def localouter(a, b):
    """Computes the tensor product of :math:`a \otimes b` locally, that is
    when a and b have the same number of sites, the new local tensors are the
    tensorproducts of the original ones.

    :param MPArray a: :class:`MPArray`
    :param MPArray b: :class:`MPArray` of same length as ``a``
    :returns: Tensor product of ``a`` and ``b`` in terms of their local tensors

    """
    assert len(a) == len(b)
    ltens = (_local_dot(t1[:, None], t2[:, None], axes=(1, 1))
             for t1, t2 in zip(a.lt, b.lt))
    return MPArray(ltens)


def norm(mpa):
    """Computes the norm (Hilbert space norm for MPS, Frobenius norm for MPO)
    of the matrix product operator. In contrast to ``mparray.inner``, this can
    take advantage of the canonicalization

    WARNING This also changes the MPA inplace by normalizing.

    :param mpa: MPArray
    :returns: l2-norm of that array

    """
    mpa.canonicalize()
    current_lcanon, current_rcanon = mpa.canonical_form

    if current_rcanon == 1:
        return np.linalg.norm(mpa.lt[0])
    elif current_lcanon == len(mpa) - 1:
        return np.linalg.norm(mpa.lt[-1])
    else:
        raise ValueError("Normalization error in MPArray.norm")


def normdist(mpa1, mpa2):
    """More efficient version of norm(mpa1 - mpa2)

    :param mpa1: MPArray
    :param mpa2: MPArray
    :returns: l2-norm of mpa1 - mpa2

    """
    return norm(mpa1 - mpa2)
    # This implementation doesn't produce an MPA with double rank:
    #
    # return np.sqrt(norm(mpa1)**2 + norm(mpa2)**2
    #                - 2 * np.real(inner(mpa1, mpa2)))
    #
    # However, there are precision issues which show up e.g. in
    # test_project_fused_clusters(). Due to rounding errors, the term
    # inside np.sqrt() can be slightly negative and np.sqrt() will
    # return nan. Even if we replace slightly negative terms by zero,
    # the slightly-positive-instead-of-zero rounding errors are
    # amplified by np.sqrt().
    #
    # On the other hand, the test which fails checks the result to 7
    # decimals. Given that np.sqrt() amplifies errors in small values,
    # this is too strict.


# TODO Convert to iterator
def _prune_ltens(mpa):
    """Contract local tensors with no physical legs.

    By default, contract to the left. At the start of the chain,
    contract to the right.

    If there are no physical legs at all, yield a single scalar.

    :param mpa: MPA to take local tensors from.
    :returns: An iterator over local tensors.

    """
    mpa_iter = iter(mpa)
    last_lten = next(mpa_iter)
    last_lten_ndims = last_lten.ndim - 2
    for lten in mpa_iter:
        num_ndims = lten.ndim - 2
        if num_ndims == 0 or last_lten_ndims == 0:
            last_lten = matdot(last_lten, lten)
            last_lten_ndims = last_lten.ndim - 2
        else:
            # num_ndims > 0 and last_lten_ndims > 0
            yield last_lten
            last_lten = lten
            last_lten_ndims = num_ndims
    # If there are no physical legs at all, we will yield one scalar
    # here.
    yield last_lten


def prune(mpa, singletons=False):
    """Contract sites with zero (physical) legs.

    :param MPArray mpa: :class:`MPArray` or iterator over local tensors
    :param singletons: If True, also contract sites where all physical
        legs have size 1 (default: ``False``)
    :returns: An :class:`MPArray` (of possibly smaller length)

    """
    if singletons and any(np.prod(p) == 1 for p in mpa.shape):
        mpa = mpa.reshape(() if np.prod(p) == 1 else p for p in mpa.shape)
    return MPArray(_prune_ltens(mpa.lt))


def partialtrace(mpa, axes=(0, 1), mptype=None):
    """Computes the trace or partial trace of an MPA.

    This function is most useful for computing traces of an MPO or MPA over
    given physical legs. For obtaining partial traces (i.e., reduced states)
    of an MPO, :func:`mpnum.mpsmpo.reductions_mpo()` will be more convenient.

    By default ``(axes=(0, 1))`` compute the trace and return the value as
    length-one MPA with zero physical legs.

    For ``axes=(m, n)`` with integer ``m``, trace over the given axes at all
    sites and return a length-one MPA with zero physical legs. (Use
    trace() to get the value directly.)

    For ``axes=(axes1, axes2, ...)`` trace over ``axesN`` at site ``N``, with
    ``axesN=(axisN_1, axisN_2)`` tracing the given physical legs and
    ``axesN=None`` leaving the site invariant. Afterwards, :func:`prune()` is
    called to remove sites with zero physical legs from the result.

    :param mpa: MPArray
    :param axes: Axes for trace, (axis1, axis2) or (axes1, axes2, ...)
        with axesN=(axisN_1, axisN_2) or axesN=None.
    :param mptype: Which constructor to call with the new local tensors
        (default: ``type(mpa)``)
    :returns: An MPArray (possibly one site with zero physical legs)

    """
    if axes[0] is not None and not isinstance(axes[0], Iterable):
        axes = (axes,) * len(mpa)
    # Prevent the user from accidentally tracing out bond legs
    assert all(ax is None or all(-n <= a < n for a in ax)
               for n, ax in zip(mpa.ndims, axes)), "Too few legs"
    axes = (None if axesitem is None else tuple(ax + 1 if ax >= 0 else ax - 1
                                                for ax in axesitem)
            for axesitem in axes)
    ltens = (
        lten if ax is None else np.trace(lten, axis1=ax[0], axis2=ax[1])
        for lten, ax in zip(mpa.lt, axes))
    return (mptype or type(mpa))(_prune_ltens(ltens))


def trace(mpa, axes=(0, 1)):
    """Compute the trace of the given MPA.

    If you specify axes (see partialtrace() for details), you must
    ensure that the result has no physical legs anywhere.

    :param mpa: MParray
    :param axes: Axes for trace, ``(axis1, axis2)`` or ``(axes1, axes2, ...)``
        with ``axesN=(axisN_1, axisN_2)`` or ``axesN=None``.
        (default: ``(0, 1)``)
    :returns: A single scalar of type ``mpa.dtype``

    """
    out = partialtrace(mpa, axes)
    out = out.to_array()
    assert out.size == 1, 'trace must return a single scalar'
    return out[None][0]


def regular_slices(length, width, offset):
    """Iterate over regular slices on a linear chain.

    Put slices on a linear chain as follows:

    >>> n = 5
    >>> [tuple(range(*s.indices(n))) for s in regular_slices(n, 3, 2)]
    [(0, 1, 2), (2, 3, 4)]
    >>> n = 7
    >>> [tuple(range(*s.indices(n))) for s in regular_slices(n, 3, 2)]
    [(0, 1, 2), (2, 3, 4), (4, 5, 6)]

    The scheme is illustrated by the following figure:

    .. tabularcolumns:: |c|c|c|

    +--------------------------+------------+
    | ###### ``width`` ####### |            |
    +------------+-------------+------------+
    | ``offset`` | ``overlap`` | ``offset`` |
    +------------+-------------+------------+
    |            | ####### ``width`` ###### |
    +------------+--------------------------+

    .. todo:: This table needs cell borders in the HTML output (->
              CSS) and the tabularcolumns command doesn't work.

    Note that the overlap may be larger than, equal to or smaller than
    zero.

    We enforce that the last slice coincides with the end of the
    chain, i.e. :code:`(length - width) / offset` must be integer.  We
    produce :code:`(length - width) / offset + 1` slices and the i-th
    slice is :code:`slice(offset * i, offset * i + width)`, with
    :code:`i` starting at zero.

    :param int length: The length of the chain.
    :param int width: The width of each slice.
    :param int offset: Difference between starting positions of
        successive slices. First slice starts at 0.
    :returns: Iterator over slices.

    """
    assert ((length - width) % offset) == 0, \
        'length {}, width {}, offset {}'.format(length, width, offset)
    num_slices = (length - width) // offset + 1

    for i in range(num_slices):
        yield slice(offset * i, offset * i + width)


def _embed_ltens_identity(mpa, embed_tensor=None):
    """Embed with identity matrices by default.

    :param embed_tensor: If the MPAs do not have two physical legs or
        have non-square physical dimensions, you must provide an
        embedding tensor. The default is to use the square identity
        matrix, assuming that the size of the two physical legs is the
        same at each site.
    :returns: `embed_tensor` with one size-one virtual leg added at each
        end.

    """
    if embed_tensor is None:
        pdims = mpa.shape[0]
        assert len(pdims) == 2 and pdims[0] == pdims[1], (
            "For ndims != 2 or non-square shape, you must supply a tensor"
            "for embedding")
        embed_tensor = np.eye(pdims[0])
    embed_ltens = embed_tensor[None, ..., None]
    return embed_ltens


def embed_slice(length, slice_, mpa, embed_tensor=None):
    """Embed a local MPA on a linear chain.

    :param int length: Length of the resulting MPA.
    :param slice slice_: Specifies the position of ``mpa`` in the
        result.
    :param MPArray mpa: MPA of length :code:`slice_.stop -
        slice_.start`.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`_embed_ltens_identity` for details)
    :returns: MPA of length `length`

    """
    start, stop, step = slice_.indices(length)
    assert step == 1
    assert len(mpa) == stop - start
    embed_ltens = _embed_ltens_identity(mpa, embed_tensor)
    left = it.repeat(embed_ltens, times=start)
    right = it.repeat(embed_ltens, times=length - stop)
    return MPArray(it.chain(left, mpa.lt, right))


def _local_sum_identity(mpas, embed_tensor=None):
    """Implement a special case of :func:`local_sum`.

    See :func:`local_sum` for a description.  We return an MPA with
    smaller rank than naive embed+MPA-sum.

    ``mpas`` is a list of MPAs. The width 'width' of all the ``mpas[i]`` must
    be the same. ``mpas[i]`` is embedded onto a linear chain on sites ``i,
    ..., i + width - 1``.

    Let `D` the rank of the ``mpas[i]``. Then the MPA we return
    has rank `width * D + 1` instead of `width * D + len(mpas)`.

    The basic idea behind the construction we use is similar to
    [:ref:`Sch11 <Sch11>`, Sec. 6.1].

    :param mpas: A list of MPArrays with the same length.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`_embed_ltens_identity` for details)

    """
    width = len(mpas[0])
    nr_sites = len(mpas) + width - 1
    ltens = []
    embed_ltens = _embed_ltens_identity(mpas[0], embed_tensor)
    assert all(len(mpa) == width for mpa in mpas)

    # The following ASCII art tries to illustrate the
    # construction. The first shows a sum of three width 2 MPAs with
    # local tensors A, B, C. The second shows a sum of four width 1
    # MPAs with local tensors A, B, C, D. The embedding tensor is
    # denoted by `1` (identity matrix). The physical legs of all local
    # tensors have been omitted for simplicity.
    # width == 2:
    #
    #               / A_2         \  /  1       \  /  1  \
    #   ( A_1  1 )  \      B_1  1 /  | B_2      |  | C_2 |
    #                                \      C_1 /  \     /
    # width == 1:
    #               /  1     \  /  1     \  /  1  \
    #   ( A_1  1 )  \ B_1  1 /  \ C_1  1 /  \ D_1 /
    for pos in range(nr_sites):
        # At this position, we local summands mpas[i] starting at the
        # following startsites are present:
        # FIXME Can we do this with slice?
        startsites = range(max(0, pos - width + 1), min(len(mpas), pos + 1))
        mpas_ltens = [mpas[i]._lt[pos - i] for i in startsites]
        # The embedding tensor embed_ltens has to be added if
        # - we need an embedding tensor on the right of an mpas[i]
        #   (pos is large enough)
        # - we need an embedding tensor on the left of an mpas[i]
        #   (pos is small enough)
        if pos >= width:
            # Construct (1 \\ B_2), (1 \\ C_2), etc.
            mpas_ltens[0] = np.concatenate((embed_ltens, mpas_ltens[0]), axis=0)
        if pos < len(mpas) - 1:
            right_embed = embed_ltens
            if width == 1 and pos >= width:
                assert len(mpas_ltens) == 1
                # Special case: We need (0 \\ 1) instead of just (1).
                right_embed = np.concatenate(
                    (np.zeros_like(embed_ltens), embed_ltens), axis=0)
            # Construct (A_1 1), (B_1 1), etc.
            mpas_ltens[-1] = np.concatenate((mpas_ltens[-1], right_embed), axis=-1)

        lten = block_diag(mpas_ltens, axes=(0, -1))
        ltens.append(lten)

    mpa = MPArray(ltens)
    return mpa


def local_sum(mpas, embed_tensor=None, length=None, slices=None):
    """Embed local MPAs on a linear chain and sum as MPA.

    We return the sum over :func:`embed_slice(length, slices[i],
    mpas[i], embed_tensor) <embed_slice>` as MPA.

    If ``slices`` is omitted, we use :func:`regular_slices(length,
    width, offset) <regular_slices>` with :code:`offset = 1`,
    :code:`width = len(mpas[0])` and :code:`length = len(mpas) + width
    - offset`.

    If ``slices`` is omitted or if the slices just described are given,
    we call :func:`_local_sum_identity()`, which gives a smaller virtual
    dimension than naive embedding and summing.

    :param mpas: List of local MPAs.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`_embed_ltens_identity` for details)
    :param length: Length of the resulting chain, ignored unless
        slices is given.
    :param slices: ``slice[i]`` specifies the position of ``mpas[i]``,
        optional.
    :returns: An MPA.

    """
    # Check whether we can fall back to :func:`_local_sum_identity`
    # even though `slices` is given.
    if slices is not None:
        assert length is not None
        slices = tuple(slices)
        reg = regular_slices(length, slices[0].stop - slices[0].start, offset=1)
        if all(s == t for s, t in zip_longest(slices, reg)):
            slices = None
    # If `slices` is not given, use :func:`_local_sum_identity`.
    if slices is None:
        return _local_sum_identity(tuple(mpas), embed_tensor)

    mpas = (embed_slice(length, slice_, mpa, embed_tensor)
            for mpa, slice_ in zip(mpas, slices))
    return sumup(mpas)


############################################################
#  Functions for dealing with local operations on tensors  #
############################################################
def _extract_factors(tens, ndims):
    """Extract iteratively the leftmost MPO tensor with given number of
    legs by a qr-decomposition

    :param np.ndarray tens: Full tensor to be factorized
    :param ndims: Number of physical legs per site or iterator over number of
        physical legs
    :returns: List of local tensors with given number of legs yielding a
        factorization of tens
    """
    current = next(ndims) if isinstance(ndims, Iterator) else ndims
    if tens.ndim == current + 2:
        return [tens]
    elif tens.ndim < current + 2:
        raise AssertionError("Number of remaining legs insufficient.")
    else:
        unitary, rest = qr(tens.reshape((np.prod(tens.shape[:current + 1]), -1)))

        unitary = unitary.reshape(tens.shape[:current + 1] + rest.shape[:1])
        rest = rest.reshape(rest.shape[:1] + tens.shape[current + 1:])

        return [unitary] + _extract_factors(rest, ndims)


def _local_dot(ltens_l, ltens_r, axes):
    """Computes the local tensors of a dot product `dot(l, r)`.

    Besides computing the normal dot product, this function rearranges the
    virtual legs in such a way that the result is a valid local tensor again.

    :param ltens_l: Array with `ndim > 1`
    :param ltens_r: Array with `ndim > 1`
    :param axes: Axes to compute dot product using the convention of
        :func:`numpy.tensordot()`. Note that these correspond to the true
        (and not the just the physical) legs of the local tensors
    :returns: Correct local tensor representation

    """
    # number of contracted legs need to be the same
    clegs_l = len(axes[0]) if isinstance(axes[0], Sequence) else 1
    clegs_r = len(axes[1]) if isinstance(axes[0], Sequence) else 1
    assert clegs_l == clegs_r, \
        "Number of contracted legs differ: {} != {}".format(clegs_l, clegs_r)
    res = np.tensordot(ltens_l, ltens_r, axes=axes)
    # Rearrange the virtual-dimension legs
    res = np.rollaxis(res, ltens_l.ndim - clegs_l, 1)
    res = np.rollaxis(res, ltens_l.ndim - clegs_l,
                      ltens_l.ndim + ltens_r.ndim - clegs_l - clegs_r - 1)
    return res.reshape((ltens_l.shape[0] * ltens_r.shape[0], ) +
                       res.shape[2:-2] +
                       (ltens_l.shape[-1] * ltens_r.shape[-1],))


def _local_add(ltenss):
    """Computes the local tensors of a sum of MPArrays (except for the boundary
    tensors)

    :param ltenss: List of arrays with `ndim > 1`
    :returns: Correct local tensor representation

    """
    shape = ltenss[0].shape
    # NOTE These are currently disabled due to real speed issues.
    #  if __debug__:
    #      for lt in ltenss[1:]:
    #          assert_array_equal(shape[1:-1], lt.shape[1:-1])

    # FIXME: Find out whether the following code does the same as
    # :func:`block_diag()` used by :func:`_local_sum_identity` and
    # which implementation is faster if so.
    newshape = (sum(lt.shape[0] for lt in ltenss), )
    newshape += shape[1:-1]
    newshape += (sum(lt.shape[-1] for lt in ltenss), )
    res = np.zeros(newshape, dtype=max(lt.dtype for lt in ltenss))

    pos_l, pos_r = 0, 0
    for lt in ltenss:
        pos_l_new, pos_r_new = pos_l + lt.shape[0], pos_r + lt.shape[-1]
        res[pos_l:pos_l_new, ..., pos_r:pos_r_new] = lt
        pos_l, pos_r = pos_l_new, pos_r_new
    return res


def _local_ravel(ltens):
    """Flattens the physical legs of ltens, the virtual-legs remain untouched

    :param ltens: :func:`numpy.ndarray` with :code:`ndim > 1`

    :returns: Reshaped `ltens` with shape :code:`(ltens.shape[0], ...,
        ltens.shape[-1])`, where :code`...` is determined from the
        size of ltens

    """
    return _local_reshape(ltens, (-1, ))


def _local_reshape(ltens, shape):
    """Reshapes the physical legs of ltens, the virtual-legs remain untouched

    :param ltens: numpy.ndarray with `ndim > 1`
    :param shape: New shape of physical legs
    :returns: Reshaped ltens

    """
    full_shape = ltens.shape
    return ltens.reshape((full_shape[0], ) + tuple(shape) + (full_shape[-1], ))


def _local_transpose(ltens, axes=None):
    """Transposes the physical legs of the local tensor ``ltens``

    :param ltens: Local tensor as numpy.ndarray with `ndim >= 2`
    :param axes:
    :returns: Transpose of ltens except for first and last dimension

    """
    # Should we construct `axes` using numpy arrays?
    last = ltens.ndim - 1
    if axes is None:
        axes = tuple(it.chain((0,), reversed(range(1, last)), (last,)))
    else:
        axes = tuple(it.chain((0,), (ax + 1 for ax in axes), (last,)))
    return np.transpose(ltens, axes=axes)


def _ltens_to_array(ltens):
    """Computes the full array representation from an iterator yielding the
    local tensors. Note that it does not get rid of virtual legs.

    :param ltens: Iterator over local tensors
    :returns: numpy.ndarray representing the contracted MPA

    """
    ltens = ltens if isinstance(ltens, Iterator) else iter(ltens)
    res = first = next(ltens)
    for tens in ltens:
        res = matdot(res, tens)
    if res is first:
        # Always return a writable array, even if len(ltens) == 1.
        res = res.copy()
    return res


################################################
#  Helper methods for variational compression  #
################################################
def _adapt_to_add_l(leftvec, compr_lten, tgt_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has two indices: `compr_mps_bond` and `tgt_mps_bond`
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct L from [:ref:`Sch11 <Sch11>`, Fig. 27, p. 48]. We have
    ``compr_lten`` in the top row of the figure without complex conjugation
    and ``tgt_lten`` in the bottom row with complex conjugation.

    .. todo:: Adapt tensor leg names.

    """
    leftvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    compr_lten = named_ndarray(compr_lten, compr_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)

    contract_compr_mps = (('compr_bond', 'compr_left_bond'),)
    leftvec = leftvec.tensordot(compr_lten, contract_compr_mps)

    contract_tgt_mps = (
        ('compr_phys', 'tgt_phys'),
        ('tgt_bond', 'tgt_left_bond'))
    leftvec = leftvec.tensordot(tgt_lten.conj(), contract_tgt_mps)
    rename_mps_mpo = (
        ('compr_right_bond', 'compr_bond'),
        ('tgt_right_bond', 'tgt_bond'))
    leftvec = leftvec.rename(rename_mps_mpo)

    leftvec = leftvec.to_array(leftvec_names)
    return leftvec


def _adapt_to_add_r(rightvec, compr_lten, tgt_lten):
    """Add one column to the right vector.

    :param rightvec: existing right vector
        It has two indices: `compr_mps_bond` and `tgt_mps_bond`

    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS


    Construct R from [:ref:`Sch11 <Sch11>`, Fig. 27, p. 48]. See comments in
    :func:`_adapt_to_add_l()` for further details.

    .. todo:: Adapt tensor leg names.

    """
    rightvec_names = ('compr_bond', 'tgt_bond')
    compr_names = ('compr_left_bond', 'compr_phys', 'compr_right_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec = named_ndarray(rightvec, rightvec_names)
    compr_lten = named_ndarray(compr_lten, compr_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)

    contract_compr_mps = (('compr_bond', 'compr_right_bond'),)
    rightvec = rightvec.tensordot(compr_lten, contract_compr_mps)

    contract_tgt_mps = (
        ('compr_phys', 'tgt_phys'),
        ('tgt_bond', 'tgt_right_bond'))
    rightvec = rightvec.tensordot(tgt_lten.conj(), contract_tgt_mps)
    rename = (
        ('compr_left_bond', 'compr_bond'),
        ('tgt_left_bond', 'tgt_bond'))
    rightvec = rightvec.rename(rename)

    rightvec = rightvec.to_array(rightvec_names)
    return rightvec


def _adapt_to_new_lten(leftvec, tgt_ltens, rightvec, max_rank):
    """Create new local tensors for the compressed MPS.

    :param leftvec: Left vector
        It has two indices: `compr_mps_bond` and `tgt_mps_bond`
    :param tgt_ltens: List of local tensor of the target MPS
    :param rightvec: Right vector
        It has two indices: `compr_mps_bond` and `tgt_mps_bond`
    :param int max_rank: Maximal rank of the result

    Compute the right-hand side of [:ref:`Sch11 <Sch11>`, Fig. 27, p. 48]. We
    have ``compr_lten`` in the top row of the figure without complex
    conjugation and ``tgt_lten`` in the bottom row with complex
    conjugation.

    For len(tgt_ltens) > 1, compute the right-hand side of
    [:ref:`Sch11 <Sch11>`, Fig. 29, p. 49].

    .. todo:: Adapt tensor leg names.

    """
    # Produce one MPS local tensor supported on len(tgt_ltens) sites.
    tgt_ltens = list(tgt_ltens)
    tgt_lten = _ltens_to_array(tgt_ltens)
    tgt_lten_shape = tgt_lten.shape
    tgt_lten = tgt_lten.reshape((tgt_lten_shape[0], -1, tgt_lten_shape[-1]))

    # Contract the middle part with the left and right parts.
    leftvec_names = ('compr_left_bond', 'tgt_left_bond')
    tgt_names = ('tgt_left_bond', 'tgt_phys', 'tgt_right_bond')
    rightvec_names = ('compr_right_bond', 'tgt_right_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    tgt_lten = named_ndarray(tgt_lten, tgt_names)
    rightvec = named_ndarray(rightvec, rightvec_names)

    contract = (('tgt_left_bond', 'tgt_left_bond'),)
    compr_lten = leftvec.tensordot(tgt_lten.conj(), contract)
    contract = (('tgt_right_bond', 'tgt_right_bond'),)
    compr_lten = compr_lten.tensordot(rightvec, contract)

    compr_lten_names = (
        'compr_left_bond', 'tgt_phys', 'compr_right_bond'
    )
    compr_lten = compr_lten.to_array(compr_lten_names).conj()
    s = compr_lten.shape
    compr_lten = compr_lten.reshape((s[0],) + tgt_lten_shape[1:-1] + (s[-1],))

    if len(tgt_ltens) == 1:
        return (compr_lten,)
    else:
        # [Sch11, p. 49] says that we can go with QR instead of SVD
        # here. However, this will generally increase the rank of
        # our compressed MPS, which we do not want.
        compr_ltens = MPArray.from_array(compr_lten, ndims=1, has_virtual=True)
        compr_ltens.compress('svd', rank=max_rank)
        return compr_ltens.lt


def full_rank(ldims):
    """Computes a list of maximal ranks for a tensor with given local dimesions

    :param ldims: Dimensions of the legs of the tensor per site. Can be either
        passed as one number per site (``[2, 5, 2]``) or if there are multiple
        legs per site as a list of tuples similar to :py:attr:`MPArray.shape`
        (e.g. ``[(2,), (3, 4), (5,)])``)
    :returns: Tuple of ranks that are maximal for the local dimensions
        ``ldims``.

    >>> full_rank([3] * 5)
    [3, 9, 9, 3]
    >>> full_rank([2] * 8)
    [2, 4, 8, 16, 8, 4, 2]
    >>> full_rank([(2, 3)] * 4)
    [6, 36, 6]

    """
    # Use arbitrary-size Python integers for dimension
    # computation. Fixed-size numpy integers will overflow quickly
    # (e.g. 2**63 = -9223372036854775808 on 64-bit systems).
    ldims_raveled = np.array([int(np.prod(ldim)) for ldim in ldims],
                             dtype=object)
    return [min(np.prod(ldims_raveled[:cut]), np.prod(ldims_raveled[cut:]))
            for cut in range(1, len(ldims))]
