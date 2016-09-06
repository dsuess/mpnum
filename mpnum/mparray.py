# encoding: utf-8
"""Module containing routines for dealing with general matrix product arrays.

References:

* .. _Sch11:

  [Sch11] Schollwöck, U. (2011). “The density-matrix renormalization
  group in the age of matrix product states”. Ann. Phys. 326(1),
  pp. 96–192. `DOI: 10.1016/j.aop.2010.09.012`_. `arXiv: 1008.3477`_.

  .. _`DOI: 10.1016/j.aop.2010.09.012`:
     http://dx.doi.org/10.1016/j.aop.2010.09.012

  .. _`arXiv: 1008.3477`: http://arxiv.org/abs/1008.3477

"""
# FIXME single site MPAs -- what is left?
# FIXME Local tensor ownership -- see MPArray class comment
# FIXME Possible Optimization:
#   - replace integer-for loops with iterataor (not obviously possible
#     everwhere)
#   - replace internal structure as list of arrays with lazy generator of
#     arrays (might not be possible, since we often iterate both ways!)
#   - more in place operations for addition, subtraction, multiplication
# TODO Replace all occurences of self._ltens with self[...] or similar &
#      benchmark. This will allow easier transition to lazy evaluation of
#      local tensors
from __future__ import absolute_import, division, print_function

import functools as ft
import itertools as it
import collections

import numpy as np
from numpy.linalg import qr, svd
from numpy.testing import assert_array_equal

from six.moves import range, zip, zip_longest

from ._named_ndarray import named_ndarray
from ._tools import block_diag, global_to_local, local_to_global, matdot


__all__ = ['MPArray', 'dot', 'inject', 'inner', 'local_sum', 'louter',
           'norm', 'normdist', 'outer', 'partialdot', 'partialtrace',
           'prune', 'regular_slices', 'embed_slice', 'trace']


class MPArray(object):
    r"""Efficient representation of a general N-partite array A in matrix
    product form with open boundary conditions:

    .. math::
       :label: mpa

       A_{i_1, \ldots, i_N} = A^{[1]}_{i_1} \ldots A^{[N]}_{i_N}

    where the :math:`A^{[k]}` are local tensors (with N legs). The
    matrix products in :eq:`mpa` are taken with respect to the left
    and right leg and the multi-index :math:`i_k` corresponds to the
    physical legs. Open boundary conditions imply that :math:`A^{[1]}`
    is 1-by-something and :math:`A^{[N]}` is something-by-1.

    By convention, the 0th and last dimension of the local tensors are reserved
    for the auxillary legs.

    .. todo:: As it is now, e.g. :func:`MPArray.__imul__()` modifies
              items from `self._ltens`.  This requires
              e.g. :func:`outer()` to take copies of the local
              tensors.  The data model seems to be that an `MPArray`
              instance owns its local tensors and everyone else,
              including each new `MPArray` instance, must take
              copies. Is this correct?

    .. todo:: :func:`MPArray.copy()` adheres to the model from the
              last item. Check whether this is the case everywhere.

    .. todo:: If we enable all special members (e.g. `__len__`) to be
              shown, we get things like `__dict__` with very long
              contents. Therefore, special members are hidden at the
              moment, but we should show the interesting one.

    .. automethod:: __init__

    """

    def __init__(self, ltens, **kwargs):
        """
        :param list ltens: List of local tensors for the MPA. In order
            to be valid the elements of `tens` need to be
            :code:`N`-dimensional arrays with :code:`N > 1` and need
            to fullfill::

                shape(tens[i])[-1] == shape(tens[i])[0].


        :param `**kwargs`: Additional paramters to set protected
            variables, not for use by user

        """
        self._ltens = list(ltens)
        for i, (ten, nten) in enumerate(zip(self._ltens[:-1], self._ltens[1:])):
            if ten.shape[-1] != nten.shape[0]:
                raise ValueError("Shape mismatch on {}: {} != {}"
                                 .format(i, ten.shape[-1], nten.shape[0]))

        # Elements _ltens[m] with m < self._lnorm are in left-canon. form
        self._lnormalized = kwargs.get('_lnormalized', None)
        # Elements _ltens[n] with n >= self._rnorm are in right-canon. form
        self._rnormalized = kwargs.get('_rnormalized', None)

    def copy(self):
        """Makes a deep copy of the MPA"""
        result = type(self)([ltens.copy() for ltens in self._ltens],
                            _lnormalized=self._lnormalized,
                            _rnormalized=self._rnormalized)
        return result

    def __len__(self):
        return len(self._ltens)

    # FIXME Can we return immutable view into array without having to set the
    #   WRITEABLE flag for the local copy?
    def __iter__(self):
        """Use only for read-only access! Do not change arrays in place!

        Subclasses should not override this method because it will
        break basic MPA functionality such as :func:`dot`.

        """
        return iter(self._ltens)

    def __getitem__(self, index):
        """Use only for read-only access! Do not change arrays in place!"""
        if type(index) == tuple:
            assert len(index) == len(self)
            return MPArray(ltens[:, i, ..., :]
                           for i, ltens in zip(index, self._ltens))
        else:
            # FIXME Maybe this should be moved to another Function
            return self._ltens[index]

    def __setitem__(self, index, value):
        """Update a local tensor and keep track of normalization."""
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
        else:
            start = index
            stop = index + 1
        if self._lnormalized is not None:
            self._lnormalized = min(self._lnormalized, start)
        if self._rnormalized is not None:
            self._rnormalized = max(self._rnormalized, stop)
        self._ltens[index] = value

    @property
    def dtype(self):
        """Returns the dtype that should be returned by to_array"""
        return np.common_type(*self._ltens)

    @property
    def dims(self):
        """Tuple of shapes for the local tensors"""
        return tuple(m.shape for m in self._ltens)

    @property
    def bdims(self):
        """Tuple of bond dimensions"""
        return tuple(m.shape[0] for m in self._ltens[1:])

    # FIXME Rremove this function or rname to maxbdim
    @property
    def bdim(self):
        """Largest bond dimension across the chain"""
        return max(self.bdims)

    @property
    def pdims(self):
        """Tuple of physical dimensions"""
        return tuple((m.shape[1:-1]) for m in self._ltens)

    @property
    def legs(self):
        """Tuple of total number of legs per site"""
        return tuple(lten.ndim for lten in self._ltens)

    @property
    def plegs(self):
        """Tuple of number of physical legs per site"""
        return tuple(lten.ndim - 2 for lten in self._ltens)

    @property
    def normal_form(self):
        """Tensors which are currently in left/right-canonical form."""
        return self._lnormalized or 0, self._rnormalized or len(self)

    def dump(self, target):
        """Serializes MPArray to :code:`h5py.Group`. Recover using
        :func:`MPArray.load`.

        :param target: :code:`h5py.Group` the instance should be saved to or
            path to h5 file (it's then serialized to /)

        """
        if isinstance(target, str):
            import h5py
            with h5py.File(target, 'w') as outfile:
                return self.dump(outfile)

        for prop in ('bdims', 'pdims'):
            # these are only saved for convenience
            target.attrs[prop] = str(getattr(self, prop))

        # these are actually used in MPArray.load
        target.attrs['len'] = len(self)
        target.attrs['normal_form'] = self.normal_form

        for site, lten in enumerate(self._ltens):
            target[str(site)] = lten

    @classmethod
    def load(cls, source):
        """Deserializes MPArray from :code:`h5py.Group`. Serialize using
        :func:`MPArray.dump`.

        :param target: :code:`h5py.Group` containing serialized MPArray or
            path to a single h5 File containing serialized MPArray under /

        """
        if isinstance(source, str):
            import h5py
            with h5py.File(source, 'r') as infile:
                return cls.load(infile)

        ltens = [source[str(i)].value for i in range(source.attrs['len'])]
        lnorm, rnorm = source.attrs['normal_form']
        return cls(ltens, _lnormalized=lnorm, _rnormalized=rnorm)

    #FIXME Where is this used? Does it really have to be in here?
    @classmethod
    def from_array_global(cls, array, plegs=None, has_bond=False):
        """Create MPA from array in global form.

        See :func:`mpnum._tools.global_to_local()` for global
        vs. local form.

        Parameters and return value: See
        `from_array()`. `has_bond=True` is not supported yet.

        """
        assert not has_bond, 'not implemented yet'
        plegs = plegs if plegs is not None else array.ndim
        assert array.ndim % plegs == 0, \
            "plegs invalid: {} is not multiple of {}".format(array.ndim, plegs)
        sites = array.ndim // plegs
        return cls.from_array(global_to_local(array, sites), plegs, has_bond)

    @classmethod
    def from_array(cls, array, plegs=None, has_bond=False):
        """Create MPA from array in local form.

        See :func:`mpnum._tools.global_to_local()` for global
        vs. local form.

        Computes the (exact) representation of `array` as MPA with
        open boundary conditions, i.e. bond dimension 1 at the
        boundary. This is done by factoring the off the left and the
        "physical" legs from the rest of the tensor by a QR
        decomposition and working its way through the tensor from the
        left. This yields a left-canonical representation of
        `array`. [Sch11_, Sec. 4.3.1]

        The result is a chain of local tensors with `plegs` physical legs at
        each location and has array.ndim // plegs number of sites.

        has_bond = True allows to treat a part of the linear chain of
        an MPA as MPA as well. The bond dimension on the left and
        right can be different from one and different from each other
        in that case.  This is useful to apply SVD compression only to
        part of an MPA. It is used in
        linalg._mineig_minimize_locally().

        :param np.ndarray array: Array representation with global structure
            array[(i1), ..., (iN)], i.e. the legs which are factorized into
            the same factor are already adjacent. (For me details see
            :func:`_tools.global_to_local`)
        :param plegs: Number of physical legs per site (default array.ndim)
            or iterable over number of physical legs
        :param bool has_bond: True if array already has indices for
            the left and right bond

        """

        plegs = plegs if plegs is not None else array.ndim
        plegs = iter(plegs) if isinstance(plegs, collections.Iterable) else plegs

        if not has_bond:
            array = array[None, ..., None]
        ltens = _extract_factors(array, plegs=plegs)
        return cls(ltens, _lnormalized=len(ltens) - 1)

    @classmethod
    def from_kron(cls, factors):
        """Returns the (exact) representation of an n-fold  Kronecker (tensor)
        product as MPA with bond dimensions 1 and n sites.

        :param factors: A list of arrays with arbitrary number of physical legs
        :returns: The kronecker product of the factors as MPA
        """
        # FIXME Do we still need this or shall we prefer mp.outer?
        return cls(a[None, ..., None] for a in factors)

    def to_array(self):
        """Return MPA as array in local form.

        See :func:`mpnum._tools.global_to_local()` for global
        vs. local form.

        :returns: ndarray of shape :code:`sum(self.pdims, ())`

        .. note:: Full arrays can require much more memory than
                  MPAs. (That's why you are using MPAs, right?)

        """
        return _ltens_to_array(iter(self))[0, ..., 0]

    #FIXME Where is this used? Does it really have to be in here?
    def to_array_global(self):
        """Return MPA as array in global form.

        See :func:`mpnum._tools.global_to_local()` for global
        vs. local form.

        :returns: ndarray of shape :code:`sum(zip(*self.pdims, ()))`

        See :func:`to_array()` for more details.

        """
        return local_to_global(self.to_array(), sites=len(self))

    def paxis_iter(self, axes=0):
        """Returns an iterator yielding Sub-MPArrays of `self` by iterating
        over the specified physical axes.

        **Example:** If `self` represents a bipartite (i.e. length 2)
        array with 2 physical dimensions on each site A[(k,l), (m,n)],
        self.paxis_iter(0) is equivalent to::

            (A[(k, :), (m, :)] for m in range(...) for k in range(...))

        FIXME The previous code is not highlighted because
        :code:`A[(k, :)]` is invalid syntax. Example of working
        highlighting::

            (x**2 for x in range(...))

        :param axes: Iterable or int specifiying the physical axes to iterate
            over (default 0 for each site)
        :returns: Iterator over MPArray

        """
        if not isinstance(axes, collections.Iterable):
            axes = it.repeat(axes, len(self))

        ltens_iter = it.product(*(iter(np.rollaxis(lten, i + 1))
                                  for i, lten in zip(axes, self._ltens)))
        return (MPArray(ltens) for ltens in ltens_iter)

    ##########################
    #  Algebraic operations  #
    ##########################
    @property
    def T(self):
        """Transpose (=reverse order of) physical legs"""
        return type(self)([_local_transpose(tens) for tens in self._ltens])

    def transpose(self, axes=None):
        """Transpose physical legs

        :param axes: New order of the physical axes (default `None` =
            reverse the order).

        >>> from .factory import random_mpa
        >>> mpa = random_mpa(2, (2, 3, 4), 2)
        >>> mpa.pdims
        ((2, 3, 4), (2, 3, 4))
        >>> mpa.transpose((2, 0, 1)).pdims
        ((4, 2, 3), (4, 2, 3))

        """
        return type(self)([_local_transpose(tens, axes) for tens in self._ltens])

    def adj(self):
        """Hermitian adjoint"""
        return type(self)([_local_transpose(tens).conjugate()
                           for tens in self._ltens])

    def conj(self):
        """Complex conjugate"""
        return type(self)(np.conjugate(self._ltens))

    def __add__(self, summand):
        assert len(self) == len(summand), \
            "Length is not equal: {} != {}".format(len(self), len(summand))
        if len(self) == 1:
            # The code below assumes at least two sites.
            return MPArray((self[0] + summand[0],))

        ltens = [np.concatenate((self[0], summand[0]), axis=-1)]
        ltens += [_local_add(l, r) for l, r in zip(self[1:-1], summand[1:-1])]
        ltens += [np.concatenate((self[-1], summand[-1]), axis=0)]
        return MPArray(ltens)

    def __sub__(self, subtr):
        return self + (-1) * subtr

    # TODO These could be made more stable by rescaling all non-normalized tens
    def __mul__(self, fact):
        if np.isscalar(fact):
            lnormal, rnormal = self.normal_form
            ltens = self._ltens[:lnormal] + [fact * self._ltens[lnormal]] + \
                self._ltens[lnormal + 1:]
            return type(self)(ltens, _lnormalized=lnormal,
                              _rnormalized=rnormal)

        raise NotImplementedError("Multiplication by non-scalar not supported")

    def __imul__(self, fact):
        if np.isscalar(fact):
            lnormal, _ = self.normal_form
            self._ltens[lnormal] *= fact
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

    ################################
    #  Shape changes, conversions  #
    ################################
    def reshape(self, newshapes):
        """Reshape physical legs in place.

        Use self.pdims to obtain the shapes of the physical legs.

        :param newshapes: A single new shape or a list of new shapes
        :returns: Reshaped MPA

        """
        newshapes = tuple(newshapes)
        if not isinstance(newshapes[0], collections.Iterable):
            newshapes = it.repeat(newshapes, times=len(self))

        return MPArray([_local_reshape(lten, newshape)
                       for lten, newshape in zip(self._ltens, newshapes)])

    def ravel(self):
        """Flatten the MPA to an MPS, shortcut for self.reshape((-1,))

        """
        return self.reshape((-1,))

    def group_sites(self, sites_per_group):
        """Group several MPA sites into one site.

        The resulting MPA has length len(self) // sites_per_group and
        sites_per_group * self.plegs[i] physical legs on site i. The
        physical legs on each sites are in local form.

        :param int sites_per_group: Number of sites to be grouped into one
        :returns: An MPA with sites_per_group fewer sites and more plegs

        """
        assert (len(self) % sites_per_group) == 0, \
            'Cannot group: {} not a multiple of {}'.format(len(self), sites_per_group)

        if sites_per_group == 1:
            return self
        ltens = [_ltens_to_array(self._ltens[i:i + sites_per_group])
                 for i in range(0, len(self), sites_per_group)]
        return MPArray(ltens)

    def split_sites(self, sites_per_group):
        """Split MPA sites into several sites.

        The resulting MPA has length len(self) * sites_per_group and
        self.plegs[i] // sites_per_group physical legs on site i. The
        physical legs on before splitting must be in local form.

        :param int sites_per_group: Split each site in that many sites
        :returns: An mpa with sites_per_group more sites and fewer plegs

        """
        ltens = []
        for i in range(len(self)):
            plegs = self.plegs[i]
            assert (plegs % sites_per_group) == 0, \
                'plegs not a multiple of sites_per_group'
            ltens += _extract_factors(self[i], plegs // sites_per_group)
        return MPArray(ltens)

    ################################
    #  Normalizaton & Compression  #
    ################################
    def normalize(self, left=None, right=None):
        """Brings the MPA to canonical form in place [Sch11_, Sec. 4.4]

        Note that we do not support full left- or right-normalization. The
        right- (left- resp.)  most local tensor is not normalized since this
        can be done by simply calculating its norm (instead of using SVD).

        The following values for `left` and `right` will be needed
        most frequently:

        +--------------+--------------+-----------------------+
        | Left-/Right- | Do Nothing   | To normalize          |
        | normalize:   |              | maximally             |
        +==============+==============+=======================+
        | `left`       | :code:`None` | :code:`'afull'`,      |
        |              |              | :code:`len(self) - 1` |
        +--------------+--------------+-----------------------+
        | `right`      | :code:`None` | :code:`'afull'`,      |
        |              |              | :code:`1`             |
        +--------------+--------------+-----------------------+

        :code:`'afull'` is short for "almost full" (we do not support
        normalizing the outermost sites).

        Arbitrary integer values of `left` and `right` have the
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

        - Integer argument too large or small: `IndexError`

        - Matrix would be both left- and right-normalized: `ValueError`

        """
        current_lnorm, current_rnorm = self.normal_form
        if left is None and right is None:
            if current_lnorm < len(self) - current_rnorm:
                self._rnormalize(1)
            else:
                self._lnormalize(len(self) - 1)
            return

        # Fill the special values for `None` and 'afull'.
        lnormalize = {None: 0, 'afull': len(self) - 1}.get(left, left)
        rnormalize = {None: len(self), 'afull': 1}.get(right, right)
        # Support negative indices.
        if lnormalize < 0:
            lnormalize += len(self)
        if rnormalize < 0:
            rnormalize += len(self)
        # Perform range checks.
        if not 0 <= lnormalize <= len(self):
            raise IndexError('len={!r}, left={!r}'.format(len(self), left))
        if not 0 <= rnormalize <= len(self):
            raise IndexError('len={!r}, right={!r}'.format(len(self), right))

        if not lnormalize < rnormalize:
            raise ValueError("Normalization {}:{} invalid"
                             .format(lnormalize, rnormalize))
        if current_lnorm < lnormalize:
            self._lnormalize(lnormalize)
        if current_rnorm > rnormalize:
            self._rnormalize(rnormalize)

    def _lnormalize(self, to_site):
        """Left-normalizes all local tensors _ltens[:to_site] in place

        :param to_site: Index of the site up to which normalization is to be
            performed

        """
        assert 0 <= to_site < len(self), 'to_site={!r}'.format(to_site)

        lnormal, rnormal = self.normal_form
        for site in range(lnormal, to_site):
            ltens = self._ltens[site]
            matshape = (np.prod(ltens.shape[:-1]), ltens.shape[-1])
            q, r = qr(ltens.reshape(matshape))
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting bond dimension here
            self._ltens[site] = q.reshape(ltens.shape[:-1] + (-1, ))
            self._ltens[site + 1] = matdot(r, self._ltens[site + 1])

        self._lnormalized = to_site
        self._rnormalized = max(to_site + 1, rnormal)

    def _rnormalize(self, to_site):
        """Right-normalizes all local tensors _ltens[to_site:] in place

        :param to_site: Index of the site up to which normalization is to be
            performed

        """
        assert 0 < to_site <= len(self), 'to_site={!r}'.format(to_site)

        lnormal, rnormal = self.normal_form
        for site in range(rnormal - 1, to_site - 1, -1):
            ltens = self._ltens[site]
            matshape = (ltens.shape[0], np.prod(ltens.shape[1:]))
            q, r = qr(ltens.reshape(matshape).T)
            # if ltens.shape[-1] > prod(ltens.phys_shape) --> trivial comp.
            # can be accounted by adapting bond dimension here
            self._ltens[site] = q.T.reshape((-1, ) + ltens.shape[1:])
            self._ltens[site - 1] = matdot(self._ltens[site - 1], r.T)

        self._lnormalized = min(to_site - 1, lnormal)
        self._rnormalized = to_site

    def compress(self, method='svd', **kwargs):
        r"""Compress `self`, modifying it in-place.

        Let :math:`\vert u \rangle` the original vector and let
        :math:`\vert c \rangle` the compressed vector. The
        compressions we return have the property (cf. [Sch11_,
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

        :param method: 'svd', 'svdsweep' or 'var'

        .. rubric:: Parameters for 'svd':

        :param bdim: Maximal bond dimension of the result. Default
            `None`.

        :param relerr: Maximal fraction of discarded singular values.
            Default `0`.  If both bdim and relerr are given, the
            smaller resulting bond dimension is used.

        :param direction: `right` (sweep from left to right), `left`
            (inverse) or `None` (choose depending on
            normalization). Default `None`.

        .. rubric:: Parameters for 'svdsweep':

        TODO

        .. rubric:: Parameters for 'var':

        :param startmpa: Start vector, also fixes the bond dimension
            of the result. Default: Random, with same norm as self.

        :param bdim: Maximal bond dimension for the result. Either
            `startmpa` or `bdim` is required.

        :param randstate: `numpy.random.RandomState` instance used for
            random start vector. Default: `numpy.random`.

        :param num_sweeps: Maximum number of sweeps to do. Default 5.

        :param var_sites: Number of sites to modify
            simultaneausly. Default 1.

        Increasing `var_sites` makes it less likely to get stuck in a
        local minimum.

        References:

        * 'svd': Singular value truncation, [Sch11_, Sec. 4.5.1]
        * 'var': Variational compression, [Sch11_, Sec. 4.5.2]

        """
        if method == 'svd':
            return self._compress_svd(**kwargs)
        elif method == 'swdsweep':
            pass
        elif method == 'var':
            compr, overlap = self._compression_var(**kwargs)
            self._lnormalized = compr._lnormalized
            self._rnormalized = compr._rnormalized
            self._ltens = compr[:]
            return overlap
        else:
            raise ValueError('{!r} is not a valid method'.format(method))

    def compression(self, method='svd', **kwargs):
        """Return a compression of `self`. Does not modify `self`.

        Parameters: See :func:`MPArray.compress()`.

        :returns: `(compressed_mpa, iprod)` where `iprod` is the inner
            product returned by :func:`MPArray.compress()`.

        """
        if method == 'svd':
            target = self.copy()
            overlap = target._compress_svd(**kwargs)
            return target, overlap
        elif method == 'var':
            return self._compression_var(**kwargs)
        else:
            raise ValueError('{!r} is not a valid method'.format(method))



    def _compress_svd(self, bdim=None, relerr=0.0, direction=None):
        """Compress `self` using SVD [Sch11_, Sec. 4.5.1]

        Parameters: See :func:`MPArray.compress()`.

        """
        if len(self) == 1:
            # Cannot do anything. Return perfect overlap.
            return norm(self)**2

        ln, rn = self.normal_form
        default_direction = 'left' if len(self) - rn > ln else 'right'
        direction = default_direction if direction is None else direction
        bdim = max(self.bdims) if bdim is None else bdim

        if direction == 'right':
            self.normalize(right=1)
            return self._compress_svd_r(bdim, relerr)
        elif direction == 'left':
            self.normalize(left=len(self) - 1)
            return self._compress_svd_l(bdim, relerr)

        raise ValueError('{} is not a valid direction'.format(direction))

    def _compress_svdsweep(self, stages=None, bdim=None, relerr=0.0,
                          num_sweeps=1):
        """Compresses the MPA by sweeping & iterative SVD compression

        :param stages: Iterator of (bdim, relerr) for iterative compression;
            if None is passed, it is created from the following parameters
        :param bdim: Maximal final bond dimension for the compressed MPA
            (default max of current bond dimensions, i.e. no compression)
        :param relerr: Maximal allowed final error for each truncation step,
            that is the fraction of truncated singular values over their sum
            (default 0.0, i.e. no compression)
        :param num_sweeps: Number of distinct stages

        """
        if stages is None:
            bdim = max(self.bdims) if bdim is None else bdim
            bdims = np.linspace(bdim, max(self.bdims), num_sweeps,
                                endpoint=False, dtype=int)
            stages = [(bd, relerr / num_sweeps) for bd in bdims]
        stages = reversed(sorted(stages))
        for bdim, relerr in stages:
            self._compress_svd(bdim, relerr)

    def _compression_var(self, startmpa=None, bdim=None, randstate=np.random,
                         num_sweeps=5, var_sites=1):
        """Return a compression from variational compression [Sch11_,
        Sec. 4.5.2]

        Parameters and return value: See
        :func:`MPArray.compression()`.

        """
        if len(self) == 1:
            # Cannot do anything. We make a copy, see below.
            copy = self.copy()
            return copy, norm(copy)**2

        if startmpa is not None:
            bdim = startmpa.bdim
        elif bdim is None:
            raise ValueError('You must provide startmpa or bdim')
        if bdim > self.bdim:
            # The caller expects that the result is independent from
            # `self`. Take a copy. If we are called from .compress()
            # instead of .compression(), we could avoid the copy and
            # return self.
            copy = self.copy()
            return copy, norm(copy)**2

        if startmpa is None:
            from mpnum.factory import random_mpa
            compr = random_mpa(len(self), self.pdims, bdim, randstate=randstate,
                               dtype=self.dtype)
        else:
            compr = startmpa.copy()
            assert all(d1 == d2 for d1, d2 in zip(self.pdims, compr.pdims))

        # flatten the array since MPS is expected & bring back
        shape = self.pdims
        compr = compr.ravel()
        overlap = compr._adapt_to(self.ravel(), num_sweeps, var_sites)
        compr = compr.reshape(shape)
        return compr, overlap

    def _compress_svd_r(self, bdim, relerr):
        """Compresses the MPA in place from left to right using SVD;
        yields a left-canonical state

        See :func:`MPArray.compress` for parameters
        """
        assert self.normal_form == (0, 1)
        assert bdim > 0, "Cannot compress to bdim={}".format(bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1):
            ltens = self._ltens[site]
            u, sv, v = svd(ltens.reshape((-1, ltens.shape[-1])))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[-1], u.shape[1], bdim, bdim_relerr)

            newshape = ltens.shape[:-1] + (bdim_t, )
            self._ltens[site] = u[:, :bdim_t].reshape(newshape)
            self._ltens[site + 1] = matdot(sv[:bdim_t, None] * v[:bdim_t, :],
                                           self._ltens[site + 1])

        self._lnormalized = len(self) - 1
        self._rnormalized = len(self)
        return np.sum(np.abs(self._ltens[-1])**2)

    def _compress_svd_l(self, bdim, relerr):
        """Compresses the MPA in place from right to left using SVD;
        yields a right-canonical state

        See :func:`MPArray.compress` for parameters

        """
        assert self.normal_form == (len(self) - 1, len(self))
        assert bdim > 0, "Cannot compress to bdim={}".format(bdim)
        assert (0. <= relerr) and (relerr <= 1.), \
            "Relerr={} not allowed".format(relerr)

        for site in range(len(self) - 1, 0, -1):
            ltens = self._ltens[site]
            matshape = (ltens.shape[0], -1)
            u, sv, v = svd(ltens.reshape(matshape))

            svsum = np.cumsum(sv) / np.sum(sv)
            bdim_relerr = np.searchsorted(svsum, 1 - relerr) + 1
            bdim_t = min(ltens.shape[0], v.shape[0], bdim, bdim_relerr)

            newshape = (bdim_t, ) + ltens.shape[1:]
            self._ltens[site] = v[:bdim_t, :].reshape(newshape)
            self._ltens[site - 1] = matdot(self._ltens[site - 1],
                                           u[:, :bdim_t] * sv[None, :bdim_t])

        self._lnormalized = 0
        self._rnormalized = 1
        return np.sum(np.abs(self._ltens[0])**2)

    #  Possible TODOs:
    #
    #  - Can we refactor this function into several shorter functions?
    #  - track overlap between 'compr' and 'target' and stop sweeping if it
    #    is small
    #  - maybe increase bond dimension of given error cannot be reached
    #  - Shall we track the error in the SVD truncation for multi-site
    #    updates? [Sch11_] says it turns out to be useful in actual DMRG.
    #  - return these details for tracking errors in larger computations
    def _adapt_to(self, target, num_sweeps, var_sites):
        """Iteratively minimize the l2 distance between `self` and `target`.
        This is especially important for variational compression, where `self`
        is the initial guess and target the MPA to be compressed.

        :param target: MPS to compress; i.e. MPA with only one
            physical leg per site

        Other parameters and references: See
        :func:`MPArray.compress()`.

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
        assert_array_equal(self.plegs, 1, "Self is not a MPS")
        assert_array_equal(target.plegs, 1, "Target is not a MPS")

        nr_sites = len(target)
        lvecs = [np.array(1, ndmin=2)] + [None] * (nr_sites - var_sites)
        rvecs = [None] * (nr_sites - var_sites) + [np.array(1, ndmin=2)]
        self.normalize(right=1)
        for pos in reversed(range(nr_sites - var_sites)):
            pos_end = pos + var_sites
            rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self[pos_end],
                                         target[pos_end])

        # Example: For `num_sweeps = 3`, `nr_sites = 3` and `var_sites
        # = 1`, we want the following sequence for `pos`:
        #
        #    0    1    2    1    0    1    2    1    0    1    2    1    0
        #    \_________/    \____/    \____/    \____/    \____/    \____/
        #        LTR         RTL       LTR       RTL       LTR       RTL
        #    \___________________/    \______________/    \______________/
        #     num_sweep = 0            num_sweep = 1       num_sweep = 1

        max_bonddim = self.bdim
        for num_sweep in range(num_sweeps):
            # Sweep from left to right (LTR)
            for pos in range(nr_sites - var_sites + 1):
                if pos == 0 and num_sweep > 0:
                    # Don't do first site again if we are not in the first sweep.
                    continue
                if pos > 0:
                    self.normalize(left=pos)
                    rvecs[pos - 1] = None
                    lvecs[pos] = _adapt_to_add_l(lvecs[pos - 1], self[pos - 1],
                                                 target[pos - 1])
                pos_end = pos + var_sites
                self[pos:pos_end] = _adapt_to_new_lten(lvecs[pos],
                                                       target[pos:pos_end],
                                                       rvecs[pos], max_bonddim)

            # Sweep from right to left (RTL; don't do `pos = nr_sites
            # - var_sites` again)
            for pos in reversed(range(nr_sites - var_sites)):
                pos_end = pos + var_sites
                if pos < nr_sites - var_sites:
                    # We always do this, because we don't do the last site again.
                    self.normalize(right=pos_end)
                    lvecs[pos + 1] = None
                    rvecs[pos] = _adapt_to_add_r(rvecs[pos + 1], self[pos_end],
                                                 target[pos_end])

                self[pos:pos_end] = _adapt_to_new_lten(lvecs[pos],
                                                       target[pos:pos_end],
                                                       rvecs[pos], max_bonddim)

        # Let u the uncompressed vector and c the compression which we
        # return. c satisfies <c|c> = <u|c> (mentioned more or less in
        # [Sch11_]). We compute <c|c> to get <u|c> and use the
        # normalization of the state to compute <c|c> (e.g. [Sch11_,
        # Fig. 24]).
        self.normalize(right=1)  # necessary if var_sites > 1
        return (np.abs(self[0])**2).sum()


#############################################
#  General functions to deal with MPArrays  #
#############################################
def dot(mpa1, mpa2, axes=(-1, 0)):
    """Compute the matrix product representation of a.b over the given
    (physical) axes. [Sch11_, Sec. 4.2]

    :param mpa1, mpa2: Factors as MPArrays

    :param axes: Tuple `(ax1, ax2)` where `ax1` (`ax2`) is a single
        physical leg number or sequence of physical leg numbers
        referring to `mpa1` (`mpa2`). The first (second, etc) entries
        of `ax1` and `ax2` will be contracted. Very similar to the
        `axes` argument for `np.tensordot()`, but the default value is
        different.

    :returns: Dot product of the physical arrays

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))

    # adapt the axes from physical to true legs
    if isinstance(axes[0], collections.Sequence):
        axes = tuple(tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes2)
                     for axes2 in axes)
    else:
        axes = tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes)

    ltens = [_local_dot(l, r, axes) for l, r in zip(mpa1, mpa2)]

    return MPArray(ltens)


def partialdot(mpa1, mpa2, start_at, axes=(-1, 0)):
    """Partial dot product of two MPAs of inequal length.

    The shorter MPA will start on site `start_at`. Local dot products
    will be carried out on all sites of the shorter MPA. Other sites
    will remain unmodified.

    mpa1 and mpa2 can also have equal length if `start_at = 0`. In
    this case, we do the same as :func:`dot()`.

    :param mpa1, mpa2: Factors as MPArrays, length must be inequal.
    :param start_at: The shorter MPA will start on this site.
    :param axes: See `axes` argument to :func:`dot()`.
    :returns: MPA with length of the longer MPA.

    """
    # adapt the axes from physical to true legs
    if isinstance(axes[0], collections.Sequence):
        axes = tuple(tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes2)
                     for axes2 in axes)
    else:
        axes = tuple(ax + 1 if ax >= 0 else ax - 1 for ax in axes)

    # Make the MPAs equal length (in fact, the shorter one will be
    # infinite length, but that's fine because we use zip()).
    shorter = mpa1 if len(mpa1) < len(mpa2) else mpa2
    shorter = it.chain(
        it.repeat(None, times=start_at), shorter, it.repeat(None))
    if len(mpa1) < len(mpa2):
        mpa1 = shorter
    else:
        mpa2 = shorter

    ltens_new = (
        l if r is None else (r if l is None else _local_dot(l, r, axes))
        for l, r in zip(mpa1, mpa2)
        )
    return MPArray(ltens_new)


# NOTE: I think this is a nice example how we could use Python's generator
#       expression to implement lazy evaluation of the matrix product structure
#       which is the whole point of doing this in the first place
def inner(mpa1, mpa2):
    """Compute the inner product <mpa1|mpa2>. Both have to have the same
    physical dimensions. If these represent a MPS, inner(...) corresponds to
    the canoncial Hilbert space scalar product, if these represent a MPO,
    inner(...) corresponds to the Frobenius scalar product (with Hermitian
    conjugation in the first argument)

    :param mpa1: MPArray with same number of physical legs on each site
    :param mpa2: MPArray with same physical shape as mpa1
    :returns: <mpa1|mpa2>

    """
    assert len(mpa1) == len(mpa2), \
        "Length is not equal: {} != {}".format(len(mpa1), len(mpa2))
    ltens_new = (_local_dot(_local_ravel(l).conj(), _local_ravel(r), axes=(1, 1))
                 for l, r in zip(mpa1, mpa2))
    return _ltens_to_array(ltens_new)[0, ..., 0]


def outer(mpas):
    """Performs the tensor product of MPAs given in `*args`

    :param mpas: Iterable of MPAs same order as they should appear in the chain
    :returns: MPA of length len(args[0]) + ... + len(args[-1])

    """
    # TODO Make this normalization aware
    # FIXME Is copying here a good idea?
    return MPArray(sum(([ltens.copy() for ltens in mpa] for mpa in mpas), []))


#FXIME Why is outer not a special case of this?
def inject(mpa, pos, num, inject_ten=None):
    """Like outer(), but place second factor somewhere inside mpa.

    Return the outer product between mpa and 'num' copies of the local
    tensor 'inject_ten', but place the copies of 'inject_ten' before
    site 'pos' inside 'mpa'. Placing at the edges of 'mpa' is not
    supported (use outer() for that).

    If 'inject_ten' is omitted, use a square identity matrix of size
    mpa.pdims[pos][0].

    :param mpa: An MPA.
    :param pos: Inject sites into the MPA before site 'pos'.
    :param num: Inject 'num' copies.
    :param inject_ten: Inject this physical tensor (if None use
       np.eye(mpa.pdims[pos][0]))
    :returns: An MPA of length len(mpa) + num

    """
    if inject_ten is None:
        inject_ten = np.eye(mpa.pdims[pos][0])
    bdim = mpa.bdims[pos - 1]
    inject_lten = np.tensordot(np.eye(bdim), inject_ten, axes=((), ()))
    inject_lten = np.rollaxis(inject_lten, 1, inject_lten.ndim)
    ltens = it.chain(
        mpa[:pos], it.repeat(inject_lten, times=num), mpa[pos:])
    return MPArray(ltens)


def louter(a, b):
    """Computes the tensorproduct of :math:`a \otimes b` locally, that is
    when a and b have the same number of sites, the new local tensors are the
    tensorproducts of the original ones.

    :param MPArray a: MPArray
    :param MPArray b: MPArray of same length as `a`
    :returns: Tensor product of `a` and `b` in terms of their local tensors

    """
    assert len(a) == len(b)
    ltens = (_local_dot(t1[:, None], t2[:, None], axes=(1, 1))
             for t1, t2 in zip(a, b))
    return MPArray(ltens)


def norm(mpa):
    """Computes the norm (Hilbert space norm for MPS, Frobenius norm for MPO)
    of the matrix product operator. In contrast to `mparray.inner`, this can
    take advantage of the normalization

    WARNING This also changes the MPA inplace by normalizing.

    :param mpa: MPArray
    :returns: l2-norm of that array

    """
    mpa.normalize()
    current_lnorm, current_rnorm = mpa.normal_form

    if current_rnorm == 1:
        return np.linalg.norm(mpa[0])
    elif current_lnorm == len(mpa) - 1:
        return np.linalg.norm(mpa[-1])
    else:
        raise ValueError("Normalization error in MPArray.norm")


def normdist(mpa1, mpa2):
    """More efficient version of norm(mpa1 - mpa2)

    :param mpa1: MPArray
    :param mpa2: MPArray
    :returns: l2-norm of mpa1 - mpa2

    """
    return norm(mpa1 - mpa2)
    # This implementation doesn't produce an MPA with double bond dimension:
    #
    # return np.sqrt(norm(mpa1)**2 + norm(mpa2)**2 - 2 * np.real(inner(mpa1, mpa2)))
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


#TODO Convert to iterator
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
    last_lten_plegs = last_lten.ndim - 2
    for lten in mpa_iter:
        num_plegs = lten.ndim - 2
        if num_plegs == 0 or last_lten_plegs == 0:
            last_lten = matdot(last_lten, lten)
            last_lten_plegs = last_lten.ndim - 2
        else:
            # num_plegs > 0 and last_lten_plegs > 0
            yield last_lten
            last_lten = lten
            last_lten_plegs = num_plegs
    # If there are no physical legs at all, we will yield one scalar
    # here.
    yield last_lten


def prune(mpa):
    """Contract sites with zero physical legs.

    :param mpa: MPA or iterator over local tensors
    :returns: An MPA of smaller length

    """
    return MPArray(_prune_ltens(mpa))


def partialtrace(mpa, axes=(0, 1)):
    """Computes the trace or partial trace of an MPA.

    By default (axes=(0, 1)) compute the trace and return the value as
    length-one MPA with zero physical legs.

    For axes=(m, n) with integer m, trace over the given axes at all
    sites and return a length-one MPA with zero physical legs. (Use
    trace() to get the value directly.)

    For axes=(axes1, axes2, ...) trace over axesN at site N, with
    axesN=(axisN_1, axisN_2) tracing the given physical legs and
    axesN=None leaving the site invariant. Afterwards, prune() is
    called to remove sites with zero physical legs from the result.

    If you need the reduced state of an MPO on all blocks of k
    consecutive sites, see mpnum.mpsmpa.partialtrace_mpo() for a more
    convenient and faster function.

    :param mpa: MPArray
    :param axes: Axes for trace, (axis1, axis2) or (axes1, axes2, ...)
        with axesN=(axisN_1, axisN_2) or axesN=None.
    :returns: An MPArray (possibly one site with zero physical legs)

    """
    if axes[0] is not None and not isinstance(axes[0], collections.Iterable):
        axes = it.repeat(axes)
    axes = (None if axesitem is None else tuple(ax + 1 if ax >= 0 else ax - 1
                                                for ax in axesitem)
            for axesitem in axes)
    ltens = (
        lten if ax is None else np.trace(lten, axis1=ax[0], axis2=ax[1])
        for lten, ax in zip(mpa, axes))
    return prune(ltens)


def trace(mpa, axes=(0, 1)):
    """Compute the trace of the given MPA.

    By default, just compute the trace.

    If you specify axes (see partialtrace() for details), you must
    ensure that the result has no physical legs anywhere.

    :param mpa: MParray
    :param axes: Axes for trace, (axis1, axis2) or (axes1, axes2, ...)
        with axesN=(axisN_1, axisN_2) or axesN=None.
    :returns: A single scalar (int/float/complex, depending on mpa)

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

    +------------------+--------+
    | #### width ##### |        |
    +--------+---------+--------+
    | offset | overlap | offset |
    +--------+---------+--------+
    |        | ##### width #### |
    +--------+------------------+

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


# FIXME What is this doing here?
def default_embed_ltens(mpa, embed_tensor):
    """Embed with identity matrices by default.

    :param embed_tensor: If the MPAs do not have two physical legs or
        have non-square physical dimensions, you must provide an
        embedding tensor. The default is to use the square identity
        matrix, assuming that the size of the two physical legs is the
        same at each site.
    :returns: `embed_tensor` with one size-one bond leg added at each
        end.

    """
    if embed_tensor is None:
        pdims = mpa.pdims[0]
        assert len(pdims) == 2 and pdims[0] == pdims[1], (
            "For plegs != 2 or non-square pdims, you must supply a tensor"
            "for embedding")
        embed_tensor = np.eye(pdims[0])
    embed_ltens = embed_tensor[None, ..., None]
    return embed_ltens


def embed_slice(length, slice_, mpa, embed_tensor=None):
    """Embed a local MPA on a linear chain.

    :param int length: Length of the resulting MPA.
    :param slice slice_: Specifies the position of `mpa` in the
        result.
    :param MPArray mpa: MPA of length :code:`slice_.stop -
        slice_.start`.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`default_embed_ltens` for details)
    :returns: MPA of length `length`

    """
    start, stop, step = slice_.indices(length)
    assert step == 1
    assert len(mpa) == stop - start
    embed_ltens = default_embed_ltens(mpa, embed_tensor)
    left = it.repeat(embed_ltens, times=start)
    right = it.repeat(embed_ltens, times=length - stop)
    return MPArray(it.chain(left, mpa, right))


def local_sum(mpas, embed_tensor=None, length=None, slices=None):
    """Embed local MPAs on a linear chain and sum as MPA.

    We return the sum over :func:`embed_slice(length, slices[i],
    mpas[i], embed_tensor) <embed_slice>` as MPA.

    If `slices` is omitted, we use :func:`regular_slices(length,
    width, offset) <regular_slices>` with :code:`offset = 1`,
    :code:`width = len(mpas[0])` and :code:`length = len(mpas) + width
    - offset`.

    If `slices` is omitted or if the slices just described are given,
    we call :func:`local_sum_simple()`, which gives a smaller bond
    dimension than naive embedding and summing.

    :param mpas: List of local MPAs.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`default_embed_ltens` for details)
    :param length: Length of the resulting chain, ignored unless
        slices is given.
    :param slices: slice[i] specifies the position of mpas[i],
        optional.
    :returns: An MPA.

    """
    if slices is not None:
        assert length is not None
        slices = tuple(slices)
        reg = regular_slices(length, slices[0].stop - slices[0].start, offset=1)
        if all(s == t for s, t in zip_longest(slices, reg)):
            slices = None
    if slices is None:
        return local_sum_simple(tuple(mpas), embed_tensor)
    mpas = (embed_slice(length, slice_, mpa, embed_tensor)
            for mpa, slice_ in zip(mpas, slices))
    return ft.reduce(MPArray.__add__, mpas)


def local_sum_simple(mpas, embed_tensor=None):
    """Implement a special case of :func:`local_sum`.

    See :func:`local_sum` for a description.  We return an MPA with
    smaller bond dimension than naive embed+MPA-sum.

    mpas is a list of MPAs. The width 'width' of all the mpas[i] must
    be the same. mpas[i] is embedded onto a linear chain on sites i,
    ..., i + width - 1.

    Let D the bond dimension of the mpas[i]. Then the MPA we return
    has bond dimension width * D + 1 instead of width * D + len(mpas).

    The basic idea behind the construction we use is similar to
    [Sch11_, Sec. 6.1].

    :param mpas: A list of MPArrays with the same length.
    :param embed_tensor: Defaults to square identity matrix (see
        :func:`default_embed_ltens` for details)

    """
    width = len(mpas[0])
    nr_sites = len(mpas) + width - 1
    ltens = []
    embed_ltens = default_embed_ltens(mpas[0], embed_tensor)
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
        startsites = range(max(0, pos - width + 1),
                           min(len(mpas), pos + 1))
        mpas_ltens = [mpas[i]._ltens[pos - i] for i in startsites]
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


############################################################
#  Functions for dealing with local operations on tensors  #
############################################################
def _extract_factors(tens, plegs):
    """Extract iteratively the leftmost MPO tensor with given number of
    legs by a qr-decomposition

    :param np.ndarray tens: Full tensor to be factorized
    :param plegs: Number of physical legs per site or iterator over number of
        physical legs
    :returns: List of local tensors with given number of legs yielding a
        factorization of tens
    """
    current = next(plegs) if isinstance(plegs, collections.Iterator) else plegs
    if tens.ndim == current + 2:
        return [tens]
    elif tens.ndim < current + 2:
        raise AssertionError("Number of remaining legs insufficient.")
    else:
        unitary, rest = qr(tens.reshape((np.prod(tens.shape[:current + 1]),
                                         np.prod(tens.shape[current + 1:]))))

        unitary = unitary.reshape(tens.shape[:current + 1] + rest.shape[:1])
        rest = rest.reshape(rest.shape[:1] + tens.shape[current + 1:])

        return [unitary] + _extract_factors(rest, plegs)


def _local_dot(ltens_l, ltens_r, axes):
    """Computes the local tensors of a dot product dot(l, r).

    Besides computing the normal dot product, this function rearranges the
    bond legs in such a way that the result is a valid local tensor again.

    :param ltens_l: Array with ndim > 1
    :param ltens_r: Array with ndim > 1
    :param axes: Axes to compute dot product using the convention of
        np.tensordot. Note that these correspond to the true (and not the
        physical) legs of the local tensors
    :returns: Correct local tensor representation

    """
    # number of contracted legs need to be the same
    clegs_l = len(axes[0]) if isinstance(axes[0], collections.Sequence) else 1
    clegs_r = len(axes[1]) if isinstance(axes[0], collections.Sequence) else 1
    assert clegs_l == clegs_r, \
        "Number of contracted legs differ: {} != {}".format(clegs_l, clegs_r)
    res = np.tensordot(ltens_l, ltens_r, axes=axes)
    # Rearrange the bond-dimension legs
    res = np.rollaxis(res, ltens_l.ndim - clegs_l, 1)
    res = np.rollaxis(res, ltens_l.ndim - clegs_l,
                      ltens_l.ndim + ltens_r.ndim - clegs_l - clegs_r - 1)
    return res.reshape((ltens_l.shape[0] * ltens_r.shape[0], ) +
                       res.shape[2:-2] +
                       (ltens_l.shape[-1] * ltens_r.shape[-1],))


def _local_add(ltens_l, ltens_r):
    """Computes the local tensors of a sum l + r (except for the boundary
    tensors)

    :param ltens_l: Array with ndim > 1
    :param ltens_r: Array with ndim > 1
    :returns: Correct local tensor representation

    """
    assert_array_equal(ltens_l.shape[1:-1], ltens_r.shape[1:-1])

    shape = (ltens_l.shape[0] + ltens_r.shape[0], )
    shape += ltens_l.shape[1:-1]
    shape += (ltens_l.shape[-1] + ltens_r.shape[-1], )
    res = np.zeros(shape, dtype=max(ltens_l.dtype, ltens_r.dtype))

    res[:ltens_l.shape[0], ..., :ltens_l.shape[-1]] = ltens_l
    res[ltens_l.shape[0]:, ..., ltens_l.shape[-1]:] = ltens_r
    return res


def _local_ravel(ltens):
    """Flattens the physical legs of ltens, the bond-legs remain untouched

    :param ltens: :func:`numpy.ndarray` with :code:`ndim > 1`

    :returns: Reshaped `ltens` with shape :code:`(ltens.shape[0], ...,
        ltens.shape[-1])`, where :code`...` is determined from the
        size of ltens

    """
    return _local_reshape(ltens, (-1, ))


def _local_reshape(ltens, shape):
    """Reshapes the physical legs of ltens, the bond-legs remain untouched

    :param ltens: numpy.ndarray with ndim > 1
    :param shape: New shape of physical legs
    :returns: Reshaped ltens

    """
    full_shape = ltens.shape
    return ltens.reshape((full_shape[0], ) + tuple(shape) + (full_shape[-1], ))


def _local_transpose(ltens, axes=None):
    """Transposes the physical legs of the local tensor `ltens`

    :param ltens: Local tensor as numpy.ndarray with ndim >= 2
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
    local tensors. Note that it does not get rid of bond legs.

    :param ltens: Iterator over local tensors
    :returns: numpy.ndarray representing the contracted MPA

    """
    ltens = ltens if isinstance(ltens, collections.Iterator) else iter(ltens)
    res = next(ltens)
    for tens in ltens:
        res = matdot(res, tens)
    return res


################################################
#  Helper methods for variational compression  #
################################################
def _adapt_to_add_l(leftvec, compr_lten, tgt_lten):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct L from [Sch11_, Fig. 27, p. 48]. We have compr_lten in
    the top row of the figure without complex conjugation and tgt_lten
    in the bottom row with complex conjugation.

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
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param compr_lten: Local tensor of the compressed MPS
    :param tgt_lten: Local tensor of the target MPS

    Construct R from [Sch11_, Fig. 27, p. 48]. See comments in
    :func:`_adapt_to_add_l()` for further details.

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


def _adapt_to_new_lten(leftvec, tgt_ltens, rightvec, max_bonddim):
    """Create new local tensors for the compressed MPS.

    :param leftvec: Left vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param tgt_ltens: List of local tensor of the target MPS
    :param rightvec: Right vector
        It has two indices: compr_mps_bond and tgt_mps_bond
    :param int max_bonddim: Maximal bond dimension of the result

    Compute the right-hand side of [Sch11_, Fig. 27, p. 48]. We have
    compr_lten in the top row of the figure without complex
    conjugation and tgt_lten in the bottom row with complex
    conjugation.

    For len(tgt_ltens) > 1, compute the right-hand side of [Sch11_,
    Fig. 29, p. 49].

    """
    # Produce one MPS local tensor supported on len(tgt_ltens) sites.
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
        compr_ltens = (compr_lten,)
    else:
        # [Sch11_, p. 49] says that we can go with QR instead of SVD
        # here. However, this will generally increase the bond dimension of
        # our compressed MPS, which we do not want.
        compr_ltens = MPArray.from_array(compr_lten, plegs=1, has_bond=True)
        compr_ltens.compress('svd', bdim=max_bonddim)
    return compr_ltens
