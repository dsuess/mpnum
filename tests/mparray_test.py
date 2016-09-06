# encoding: utf-8
# FIXME Is there a better metric to compare two arrays/scalars than
#       assert_(array)_almost_equal? Something that takes magnitude into
#       account?

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_almost_equal, assert_equal
from six.moves import range, zip
import h5py as h5

import mpnum.factory as factory
import mpnum.mparray as mp
from mpnum import _tools
from mpnum._tools import global_to_local
from mpnum._testing import assert_mpa_almost_equal, \
    assert_mpa_identical, mpo_to_global


# nr_sites, local_dim, bond_dim
MP_TEST_PARAMETERS = [(1, 7, np.nan), (2, 3, 3), (3, 2, 4), (6, 2, 4),
                      (4, 3, 5), (5, 2, 1)]
# local_dim, bond_dim
MP_TEST_PARAMETERS_INJECT = [(2, 4), (3, 3), (2, 5), (2, 1), (1, 2)]
# nr_sites, local_dim, bond_dim, sites_per_group
MP_TEST_PARAMETERS_GROUPS = [(6, 2, 4, 3), (6, 2, 4, 2), (4, 3, 5, 2)]

MP_TEST_DTYPES = [np.float_, np.complex_]


def update_copy_of(target, newvals):
    new = target.copy()
    new.update(newvals)
    return new


###############################################################################
#                         Basic creation & operations                         #
###############################################################################
@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_from_full(nr_sites, local_dim, _, rgen, dtype):
    psi = factory.random_vec(nr_sites, local_dim, randstate=rgen, dtype=dtype)
    mps = mp.MPArray.from_array(psi, 1)
    assert_array_almost_equal(psi, mps.to_array())
    assert mps.dtype == dtype

    op = factory.random_op(nr_sites, local_dim, randstate=rgen, dtype=dtype)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(op, mpo.to_array())
    assert mpo.dtype == dtype


def test_from_inhomogenous(rgen):
    array = rgen.randn(4, 3, 3, 3)
    mpa = mp.MPArray.from_array(array, plegs=(2, 1, 1))
    assert_array_almost_equal(array, mpa.to_array())
    assert mpa.plegs == (2, 1, 1)
    assert mpa.pdims == ((4, 3), (3,), (3,))


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_from_kron(nr_sites, local_dim, bond_dim, dtype):
    plegs = 2
    randfun = factory._randfuncs[dtype]
    factors = tuple(randfun([nr_sites] + ([local_dim] * plegs)))
    op = _tools.mkron(*factors)
    op.shape = [local_dim] * (plegs * nr_sites)
    mpo = mp.MPArray.from_kron(factors)
    assert_array_almost_equal(op, mpo_to_global(mpo))
    assert mpo.dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_conjugations(nr_sites, local_dim, _, rgen, dtype):
    op = factory.random_op(nr_sites, local_dim, randstate=rgen, dtype=dtype)
    mpo = mp.MPArray.from_array(op, 2)
    assert_array_almost_equal(np.conj(op), mpo.conj().to_array())
    assert mpo.conj().dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_transpose(nr_sites, local_dim, _, rgen, dtype):
    op = factory.random_op(nr_sites, local_dim, randstate=rgen, dtype=dtype)
    mpo = mp.MPArray.from_array(global_to_local(op, nr_sites), 2)

    opT = op.reshape((local_dim**nr_sites,) * 2).T \
        .reshape((local_dim,) * 2 * nr_sites)
    assert_array_almost_equal(opT, mpo_to_global(mpo.T))
    assert mpo.T.dtype == dtype


def test_transpose_axes(rgen):
    ldim = (2, 5, 3)
    axes = (2, 0, 1)
    new_ldim = tuple(ldim[ax] for ax in axes)

    # Easy (to implement) test: One physical site only.
    vec = factory._zrandn(ldim, rgen)
    mps = mp.MPArray.from_array(vec, plegs=len(ldim))
    assert len(mps) == 1

    vec_t = vec.transpose(axes)
    mpa_t = mps.transpose(axes)
    mpa_t_to_vec = mpa_t.to_array()
    assert vec_t.shape == new_ldim
    assert_array_equal(mpa_t_to_vec, vec_t)

    # Test with 3 sites
    nr_sites = 3
    tensor = factory._zrandn(ldim * nr_sites, rgen)  # local form
    mpa = mp.MPArray.from_array(tensor, plegs=len(ldim))
    assert len(mpa) == nr_sites
    assert mpa.pdims == (ldim,) * nr_sites
    # transpose axes in local form
    tensor_axes = tuple(ax + site * len(ldim)
                        for site in range(nr_sites) for ax in axes)
    tensor_t = tensor.transpose(tensor_axes)
    mpa_t = mpa.transpose(axes)
    mpa_t_to_tensor = mpa_t.to_array()
    assert mpa_t.pdims == (new_ldim,) * nr_sites
    assert_array_almost_equal(mpa_t_to_tensor, tensor_t)


def test_dump_and_load(tmpdir):
    mpa = factory.random_mpa(5, [(4,), (2, 3), (1,), (4,), (4, 3)],
                             (4, 7, 1, 3))
    mpa.normalize(left=1, right=3)

    with h5.File(str(tmpdir / 'dump_load_test.h5'), 'w') as buf:
        newgroup = buf.create_group('mpa')
        mpa.dump(newgroup)
    with h5.File(str(tmpdir / 'dump_load_test.h5'), 'r') as buf:
        mpa_loaded = mp.MPArray.load(buf['mpa'])
    assert_mpa_identical(mpa, mpa_loaded)

    mpa.dump(str(tmpdir / 'dump_load_test_str.h5'))
    mpa_loaded = mp.MPArray.load(str(tmpdir / 'dump_load_test_str.h5'))
    assert_mpa_identical(mpa, mpa_loaded)



###############################################################################
#                            Algebraic operations                             #
###############################################################################


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_sum(nr_sites, local_dim, bond_dim, rgen, dtype):
    """Compare mpa.sum() with full array computation"""
    mpa = factory.random_mpa(nr_sites, local_dim, bond_dim, rgen, dtype)
    array_sum = mpa.to_array().sum()
    # Test summation over all indices and different argument values.
    assert_almost_equal(mpa.sum(), array_sum)
    assert_almost_equal(mpa.sum(0), array_sum)
    assert_almost_equal(mpa.sum([0]), array_sum)
    assert_almost_equal(mpa.sum([[0]] * nr_sites), array_sum)

    # Test summation over site-dependent indices
    n_plegs = 3 if nr_sites <= 4 and local_dim <= 2 else 2
    mpa = factory.random_mpa(nr_sites, [local_dim] * n_plegs, bond_dim, rgen, dtype)
    # Pseudo-randomly choose how many physical legs to sum over at each site.
    num_sum = ((rgen.choice(range(plegs + 1)), plegs) for plegs in mpa.plegs)
    # Pseudo-randomly choose which physical legs to sum over.
    axes = tuple(
        rgen.choice(range(plegs), num, replace=False) for num, plegs in num_sum)
    array_axes = tuple(n_plegs * pos + a
                       for pos, ax in enumerate(axes) for a in ax)
    mpa_sum = mpa.sum(axes)
    if hasattr(mpa_sum, 'to_array'):  # possibly, no physical legs are left
        mpa_sum = mpa_sum.to_array()
    array_sum = mpa.to_array().sum(array_axes)
    assert_array_almost_equal(mpa_sum, array_sum)


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_dot(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op1 = mpo_to_global(mpo1)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op2 = mpo_to_global(mpo2)

    # Dotproduct of all 1st physical with 0th physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([1], [0]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(1, 0)))
    assert_array_almost_equal(dot_np, dot_mp)
    assert dot_mp.dtype == dtype
    # this should also be the default axes
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2))
    assert_array_almost_equal(dot_np, dot_mp)

    # Dotproduct of all 0th physical with 1st physical legs = np.dot
    dot_np = np.tensordot(op1.reshape((local_dim**nr_sites, ) * 2),
                          op2.reshape((local_dim**nr_sites, ) * 2),
                          axes=([0], [1]))
    dot_np = dot_np.reshape(op1.shape)
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(0, 1)))
    assert_array_almost_equal(dot_np, dot_mp)
    assert dot_mp.dtype == dtype
    # this should also be the default axes
    dot_mp = mpo_to_global(mp.dot(mpo1, mpo2, axes=(-2, -1)))
    assert_array_almost_equal(dot_np, dot_mp)


def test_dot_multiaxes(rgen):
    ldim1 = (2, 2, 3, 2)
    ldim2 = (3, 2, 4)
    ax1 = (0, 2)
    ax2 = (-2, 0)
    assert len(ax1) == len(ax2)

    # Easy (to implement) test: One physical site.
    vec1 = factory._zrandn(ldim1, rgen)
    vec2 = factory._zrandn(ldim2, rgen)
    mpa1 = mp.MPArray.from_array(vec1, plegs=len(ldim1))
    mpa2 = mp.MPArray.from_array(vec2, plegs=len(ldim2))
    assert len(mpa1) == 1
    assert len(mpa2) == 1

    mpa_prod = mp.dot(mpa1, mpa2, axes=(ax1, ax2)).to_array()
    vec_prod = np.tensordot(vec1, vec2, (ax1, ax2))
    assert_array_almost_equal(mpa_prod, vec_prod)

    # Test with 3 sites
    nr_sites = 3
    vec1 = factory._zrandn(ldim1 * nr_sites, rgen)  # local form
    vec2 = factory._zrandn(ldim2 * nr_sites, rgen)  # local form
    mpa1 = mp.MPArray.from_array(vec1, plegs=len(ldim1))
    mpa2 = mp.MPArray.from_array(vec2, plegs=len(ldim2))
    assert len(mpa1) == nr_sites
    assert len(mpa2) == nr_sites
    mpa_prod = mp.dot(mpa1, mpa2, axes=(ax1, ax2)).to_array()
    vec_ax1, vec_ax2 = (
        tuple(ax + site * nldim
              if ax >= 0 else ax - (nr_sites - site - 1) * nldim
              for site in range(nr_sites) for ax in ax_n)
        for ax_n, nldim in ((ax1, len(ldim1)), (ax2, len(ldim2)))
    )
    vec_prod = np.tensordot(vec1, vec2, (vec_ax1, vec_ax2))
    # The problem with vec_prod is: The order of the indices does not
    # match the order of the indices in mpa_prod. We need to change
    # that order:
    nldim1, nldim2 = (len(ldim1) - len(ax1), len(ldim2) - len(ax2))
    assert vec_prod.ndim == nr_sites * (nldim1 + nldim2)
    perm = tuple(
        offset + site * nldim + ax
        for site in range(nr_sites)
        for offset, nldim in ((0, nldim1), (nr_sites * nldim1, nldim2))
        for ax in range(nldim)
    )
    vec_prod = vec_prod.transpose(perm)
    assert_array_almost_equal(mpa_prod, vec_prod)


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_partialdot(nr_sites, local_dim, bond_dim, rgen, dtype):
    # Only for at least two sites, we can apply an operator to a part
    # of a chain.
    if nr_sites < 2:
        return
    part_sites = nr_sites // 2
    start_at = min(2, nr_sites // 2)

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)
    mpo_part = factory.random_mpa(part_sites, (local_dim, local_dim), bond_dim,
                                  randstate=rgen, dtype=dtype)
    op_part = mpo_to_global(mpo_part).reshape((local_dim**part_sites,) * 2)
    op_part_embedded = np.kron(
        np.kron(np.eye(local_dim**start_at), op_part),
        np.eye(local_dim**(nr_sites - part_sites - start_at)))

    prod1 = np.dot(op, op_part_embedded)
    prod2 = np.dot(op_part_embedded, op)
    prod1_mpo = mp.partialdot(mpo, mpo_part, start_at=start_at)
    prod2_mpo = mp.partialdot(mpo_part, mpo, start_at=start_at)
    prod1_mpo = mpo_to_global(prod1_mpo).reshape((local_dim**nr_sites,) * 2)
    prod2_mpo = mpo_to_global(prod2_mpo).reshape((local_dim**nr_sites,) * 2)

    assert_array_almost_equal(prod1, prod1_mpo)
    assert_array_almost_equal(prod2, prod2_mpo)
    assert prod1_mpo.dtype == dtype
    assert prod2_mpo.dtype == dtype


def test_partialdot_multiaxes(rgen):
    ldim1 = (2, 2, 3, 2)
    ldim2 = (3, 2, 4)
    ax1 = (0, 2)
    ax2 = (-2, 0)
    assert len(ax1) == len(ax2)

    # Easy (to implement) test: One physical site.
    vec1 = factory._zrandn(ldim1, rgen)
    vec2 = factory._zrandn(ldim2, rgen)
    mpa1 = mp.MPArray.from_array(vec1, plegs=len(ldim1))
    mpa2 = mp.MPArray.from_array(vec2, plegs=len(ldim2))
    assert len(mpa1) == 1
    assert len(mpa2) == 1

    mpa_prod = mp.partialdot(mpa1, mpa2, start_at=0, axes=(ax1, ax2)).to_array()
    vec_prod = np.tensordot(vec1, vec2, (ax1, ax2))
    assert_array_almost_equal(mpa_prod, vec_prod)

    # Test with 3 sites
    nr_sites = 3
    nr_sites_shorter = 2
    start_at = 1
    vec1 = factory._zrandn(ldim1 * nr_sites, rgen)  # local form
    vec2 = factory._zrandn(ldim2 * nr_sites_shorter, rgen)  # local form
    mpa1 = mp.MPArray.from_array(vec1, plegs=len(ldim1))
    mpa2 = mp.MPArray.from_array(vec2, plegs=len(ldim2))
    assert len(mpa1) == nr_sites
    assert len(mpa2) == nr_sites_shorter
    mpa_prod = mp.partialdot(mpa1, mpa2, start_at, axes=(ax1, ax2)).to_array()
    vec_ax1, vec_ax2 = (
        tuple(ax + (startsite + site) * nldim
              if ax >= 0 else ax - (nr_sites_shorter - site - 1) * nldim
              for site in range(nr_sites_shorter) for ax in ax_n)
        for ax_n, nldim, startsite in
        ((ax1, len(ldim1), start_at), (ax2, len(ldim2), 0))
    )
    vec_prod = np.tensordot(vec1, vec2, (vec_ax1, vec_ax2))
    # The problem with vec_prod is: The order of the indices does not
    # match the order of the indices in mpa_prod. We need to change
    # that order:
    nldim1, nldim2 = (len(ldim1) - len(ax1), len(ldim2) - len(ax2))
    assert vec_prod.ndim == start_at * len(ldim1) + nr_sites_shorter * (nldim1 + nldim2)
    # For sites before start_at, the axes of `vec1` remain unchanged.
    perm = tuple(range(len(ldim1) * start_at))
    # For site start_at and following sites, we need to fix the order
    # of sites. We use the same scheme as `test_dot_multiaxes` above.
    perm2 = tuple(
        offset + site * nldim + ax
        for site in range(nr_sites_shorter)
        for offset, nldim in ((0, nldim1), (nr_sites_shorter * nldim1, nldim2))
        for ax in range(nldim)
    )
    # Now we displace that permutation by the number of unchanged
    # sites at the beginning:
    perm += tuple(len(perm) + ax for ax in perm2)
    vec_prod = vec_prod.transpose(perm)
    assert_array_almost_equal(mpa_prod, vec_prod)


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_vec(nr_sites, local_dim, bond_dim, rgen, dtype):
    mp_psi1 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen,
                                 dtype=dtype)
    psi1 = mp_psi1.to_array().ravel()
    mp_psi2 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen,
                                 dtype=dtype)
    psi2 = mp_psi2.to_array().ravel()

    inner_np = np.vdot(psi1, psi2)
    inner_mp = mp.inner(mp_psi1, mp_psi2)
    assert_almost_equal(inner_mp, inner_np)
    assert inner_mp.dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inner_mat(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op1 = mpo_to_global(mpo1).reshape((local_dim**nr_sites, ) * 2)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op2 = mpo_to_global(mpo2).reshape((local_dim**nr_sites, ) * 2)

    inner_np = np.trace(np.dot(op1.conj().transpose(), op2))
    inner_mp = mp.inner(mpo1, mpo2)
    assert_almost_equal(inner_mp, inner_np)
    assert inner_mp.dtype == dtype


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_norm(nr_sites, local_dim, bond_dim, rgen):
    mp_psi = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)
    psi = mp_psi.to_array()

    assert_almost_equal(mp.inner(mp_psi, mp_psi), mp.norm(mp_psi)**2)
    assert_almost_equal(np.sum(psi.conj() * psi), mp.norm(mp_psi)**2)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normdist(nr_sites, local_dim, bond_dim, rgen):
    psi1 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)
    psi2 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)

    assert_almost_equal(mp.normdist(psi1, psi2), mp.norm(psi1 - psi2))


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim, keep_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_partialtrace(nr_sites, local_dim, bond_dim, keep_width, rgen, dtype):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo)

    for site in range(nr_sites - keep_width + 1):
        traceout = tuple(range(site)) \
            + tuple(range(site + keep_width, nr_sites))
        axes = [(0, 1) if site in traceout else None for site in range(nr_sites)]
        red_mpo = mp.partialtrace(mpo, axes=axes)
        red_from_op = _tools.partial_trace(op, traceout)
        assert_array_almost_equal(mpo_to_global(red_mpo), red_from_op,
                                  err_msg="not equal at site {}".format(site))
        assert red_mpo.dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_trace(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo).reshape((local_dim**nr_sites,) * 2)

    mpo_trace = mp.trace(mpo)
    assert_almost_equal(np.trace(op), mpo_trace)
    assert np.array(mpo_trace).dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_add_and_subtr(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op1 = mpo_to_global(mpo1)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=dtype)
    op2 = mpo_to_global(mpo2)

    assert_array_almost_equal(op1 + op2, mpo_to_global(mpo1 + mpo2))
    assert_array_almost_equal(op1 - op2, mpo_to_global(mpo1 - mpo2))
    assert (mpo1 + mpo2).dtype == dtype
    assert (mpo1 + mpo2).dtype == dtype

    mpo1 += mpo2
    assert_array_almost_equal(op1 + op2, mpo_to_global(mpo1))
    assert mpo1.dtype == dtype


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', [(3, 2, 2)])
def test_operations_typesafety(nr_sites, local_dim, bond_dim, rgen):
    # create a real MPA
    mpo1 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=np.float_)
    mpo2 = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen, dtype=np.complex_)

    assert mpo1.dtype == np.float_
    assert mpo2.dtype == np.complex_

    assert (mpo1 + mpo1).dtype == np.float_
    assert (mpo1 + mpo2).dtype == np.complex_
    assert (mpo2 + mpo1).dtype == np.complex_

    assert (mpo1 - mpo1).dtype == np.float_
    assert (mpo1 - mpo2).dtype == np.complex_
    assert (mpo2 - mpo1).dtype == np.complex_

    mpo1 += mpo2
    assert mpo1.dtype == np.complex_


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    # FIXME Change behavior of to_array
    # For nr_sites == 1, changing `mpo` below will change `op` as
    # well, unless we call .copy().
    op = mpo_to_global(mpo).copy()
    scalar = rgen.randn()

    assert_array_almost_equal(scalar * op, mpo_to_global(scalar * mpo))

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo_to_global(mpo))
    assert mpo.dtype == dtype
    assert (1.j * mpo).dtype == np.complex_


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_div_mpo_scalar(nr_sites, local_dim, bond_dim, rgen):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate=rgen)
    # FIXME Change behavior of to_array
    # For nr_sites == 1, changing `mpo` below will change `op` as
    # well, unless we call .copy().
    op = mpo_to_global(mpo).copy()
    scalar = rgen.randn()

    assert_array_almost_equal(op / scalar, mpo_to_global(mpo / scalar))

    mpo /= scalar
    assert_array_almost_equal(op / scalar, mpo_to_global(mpo))


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_outer(nr_sites, local_dim, bond_dim, rgen, dtype):
    # This test produces at most `nr_sites` by tensoring two
    # MPOs. This doesn't work for :code:`nr_sites = 1`.
    if nr_sites < 2:
        return

    # NOTE: Everything here is in local form!!!
    mpo = factory.random_mpa(nr_sites // 2, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo.to_array()

    # Test with 2-factors with full form
    mpo_double = mp.outer((mpo, mpo))
    op_double = np.tensordot(op, op, axes=(tuple(), ) * 2)
    assert len(mpo_double) == 2 * len(mpo)
    assert_array_almost_equal(op_double, mpo_double.to_array())
    assert_array_equal(mpo_double.bdims, mpo.bdims + (1,) + mpo.bdims)
    assert mpo.dtype == dtype

    # Test 3-factors iteratively (since full form would be too large!!
    diff = mp.outer((mpo, mpo, mpo)) - mp.outer((mpo, mp.outer((mpo, mpo))))
    diff.normalize()
    assert len(diff) == 3 * len(mpo)
    assert mp.norm(diff) < 1e-6


@pt.mark.parametrize('local_dim, bond_dim', MP_TEST_PARAMETERS_INJECT)
def test_inject(local_dim, bond_dim):
    """mp.inject() vs. computation with full arrays"""
    # bond_dim is np.nan for nr_sites = 1 (first argument,
    # ignored). We require a value for bond_dim.
    if np.isnan(bond_dim):
        return

    # plegs = 3 is hardcoded below (argument to .transpose()).
    # Uniform local dimension is also hardcoded below (arguments to
    # .reshape()).
    plegs = 3
    local_dim = (local_dim,) * plegs

    a, b, c = factory._zrandn((3, 2) + local_dim)
    # We don't use b[1, :]
    b = b[0, :]
    # Here, only global order (as given by np.kron()).
    abbc0 = _tools.mkron(a[0, :], b, b, c[0, :])
    abbc1 = _tools.mkron(a[1, :], b, b, c[1, :])
    abbc = (abbc0 + abbc1).reshape(4 * local_dim)
    ac0 = np.kron(a[0, :], c[0, :])
    ac1 = np.kron(a[1, :], c[1, :])
    ac = (ac0 + ac1).reshape(2 * local_dim)
    ac_mpo = mp.MPArray.from_array(global_to_local(ac, sites=2), plegs)
    abbc_mpo = mp.inject(ac_mpo, pos=1, num=2, inject_ten=b)
    abbc_mpo2 = mp.inject(ac_mpo, pos=[1], num=[2], inject_ten=[b])
    abbc_mpo3 = mp.inject(ac_mpo, pos=[1], num=None, inject_ten=[[b, b]])
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo))
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo2))
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo3))

    # Here, only local order.
    ac = factory._zrandn(local_dim * 2)
    b = factory._zrandn(local_dim)
    acb = np.tensordot(ac, b, axes=((), ()))
    abc = acb.transpose((0, 1, 2, 6, 7, 8, 3, 4, 5))
    ac_mpo = mp.MPArray.from_array(ac, plegs)
    abc_mpo = mp.inject(ac_mpo, pos=1, num=1, inject_ten=b)
    # Keep local order
    abc_from_mpo = abc_mpo.to_array()
    assert_array_almost_equal(abc, abc_from_mpo)

    # plegs = 2 is hardcoded below (argument to .transpose()).
    # Uniform local dimension is also hardcoded below (arguments to
    # .reshape()).
    plegs = 2
    local_dim = (local_dim[0],) * plegs

    a, c = factory._zrandn((2, 2) + local_dim)
    b = np.eye(local_dim[0])
    # Here, only global order (as given by np.kron()).
    abbc0 = _tools.mkron(a[0, :], b, b, c[0, :])
    abbc1 = _tools.mkron(a[1, :], b, b, c[1, :])
    abbc = (abbc0 + abbc1).reshape(4 * local_dim)
    ac0 = np.kron(a[0, :], c[0, :])
    ac1 = np.kron(a[1, :], c[1, :])
    ac = (ac0 + ac1).reshape(2 * local_dim)
    ac_mpo = mp.MPArray.from_array(global_to_local(ac, sites=2), plegs)
    abbc_mpo = mp.inject(ac_mpo, pos=1, num=2, inject_ten=None)
    abbc_mpo2 = mp.inject(ac_mpo, pos=[1], num=[2])
    abbc_mpo3 = mp.inject(ac_mpo, pos=[1], inject_ten=[[None, None]])
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo))
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo2))
    assert_array_almost_equal(abbc, mpo_to_global(abbc_mpo3))

    # Here, only local order.
    ac = factory._zrandn(local_dim * 2)
    b = np.eye(local_dim[0])
    acb = np.tensordot(ac, b, axes=((), ()))
    abc = acb.transpose((0, 1, 4, 5, 2, 3))
    ac_mpo = mp.MPArray.from_array(ac, plegs)
    abc_mpo = mp.inject(ac_mpo, pos=1, num=1, inject_ten=None)
    # Keep local order
    abc_from_mpo = abc_mpo.to_array()
    assert_array_almost_equal(abc, abc_from_mpo)


@pt.mark.parametrize('local_dim, bond_dim', MP_TEST_PARAMETERS_INJECT)
def test_inject_many(local_dim, bond_dim, rgen):
    """Calling mp.inject() repeatedly vs. calling it with sequence arguments"""
    mpa = factory.random_mpa(3, local_dim, bond_dim, rgen, normalized=True)
    inj_lt = [factory._zrandn(s, rgen) for s in [(2, 3), (1,), (2, 2), (3, 2)]]

    mpa_inj1 = mp.inject(mpa, 1, None, [inj_lt[0]])
    mpa_inj1 = mp.inject(mpa_inj1, 2, 1, inj_lt[0])
    mpa_inj1 = mp.inject(mpa_inj1, 4, None, [inj_lt[2]])
    mpa_inj2 = mp.inject(mpa, [1, 2], [2, None], [inj_lt[0], [inj_lt[2]]])
    mpa_inj3 = mp.inject(mpa, [1, 2], [2, 1], [inj_lt[0], inj_lt[2]])
    assert_mpa_almost_equal(mpa_inj1, mpa_inj2, True)
    assert_mpa_almost_equal(mpa_inj1, mpa_inj3, True)

    inj_lt = [inj_lt[:2], inj_lt[2:]]
    mpa_inj1 = mp.inject(mpa, 1, None, inj_lt[0])
    mpa_inj1 = mp.inject(mpa_inj1, 4, inject_ten=inj_lt[1])
    mpa_inj2 = mp.inject(mpa, [1, 2], None, inj_lt)
    assert_mpa_almost_equal(mpa_inj1, mpa_inj2, True)


def test_inject_pdim(rgen):
    """Check that mp.inject() picks up the correct physical dimension"""
    bond_dim = 3
    mpa = factory.random_mpa(3, ([1], [2], [3]), 3, rgen, normalized=True)
    print(mpa.pdims)
    mpa_inj = mp.inject(mpa, [0, 2], [1, 1])
    assert mpa_inj.pdims == ((1, 1), (1,), (2,), (3, 3), (3,))
    mpa_inj = mp.inject(mpa, [1, 3], [1, 1], None)
    assert mpa_inj.pdims == ((1,), (2, 2), (2,), (3,), (3, 3))


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_inject_outer(nr_sites, local_dim, bond_dim, rgen):
    """Compare mp.inject() with mp.outer()"""
    if nr_sites == 1:
        return
    mpa = factory.random_mpa(nr_sites // 2, local_dim, bond_dim, rgen, True)
    pten = [factory._zrandn((local_dim,) * 2) for _ in range(nr_sites // 2)]
    pten_mpa = mp.MPArray.from_kron(pten)

    outer1 = mp.outer((pten_mpa, mpa))
    outer2 = mp.inject(mpa, 0, inject_ten=pten)
    assert_mpa_almost_equal(outer1, outer2, True)

    outer1 = mp.outer((mpa, pten_mpa))
    outer2 = mp.inject(mpa, [len(mpa)], [None], inject_ten=[pten])
    assert_mpa_almost_equal(outer1, outer2, True)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_louter(nr_sites, local_dim, bond_dim, rgen):
    mpa1 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)
    mpa2 = factory.random_mpa(nr_sites, local_dim, bond_dim, randstate=rgen)
    arr1 = mpa1.to_array()
    arr1 = arr1.reshape(arr1.shape + (1, ) * nr_sites)
    arr2 = mpa2.to_array()
    arr2 = arr2.reshape((1, ) * nr_sites + arr2.shape)

    tensor_mp = mp.louter(mpa1, mpa2)
    tensor_np = arr1 * arr2

    assert tensor_mp.plegs == (2,) * nr_sites
    assert tensor_np.shape == (local_dim,) * (2 * nr_sites)

    assert_array_almost_equal(tensor_np, tensor_mp.to_array_global())


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, local_width',
                     [(5, 2, 3, 1), (6, 2, 4, 3), (4, 3, 5, 2)])
def test_local_sum(nr_sites, local_dim, bond_dim, local_width, rgen):
    eye_mpa = factory.eye(1, local_dim)

    def embed_mpa(mpa, startpos):
        mpas = [eye_mpa] * startpos + [mpa] + \
               [eye_mpa] * (nr_sites - startpos - local_width)
        res = mp.outer(mpas)
        return res

    nr_startpos = nr_sites - local_width + 1
    mpas = [factory.random_mpa(local_width, (local_dim,) * 2, bond_dim,
                               randstate=rgen)
            for i in range(nr_startpos)]

    # Embed with mp.outer() and calculate naive MPA sum:
    mpas_embedded = [embed_mpa(mpa, i) for i, mpa in enumerate(mpas)]
    mpa_sum = mpas_embedded[0]
    for mpa in mpas_embedded[1:]:
        mpa_sum += mpa

    # Compare with local_sum: Same result, smaller bond
    # dimension.
    mpa_local_sum = mp.local_sum(mpas)

    # Check that local_sum() is no worse than naive sum
    assert all(d1 <= d2 for d1, d2 in zip(mpa_local_sum.bdims, mpa_sum.bdims))
    # Check that local_sum() is actually better than naive sum because
    # it calls local_sum_simple().
    assert any(d1 < d2 for d1, d2 in zip(mpa_local_sum.bdims, mpa_sum.bdims))
    assert_array_almost_equal(mpa_local_sum.to_array(), mpa_sum.to_array())


###############################################################################
#                         Shape changes, conversions                          #
###############################################################################
@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group',
                     MP_TEST_PARAMETERS_GROUPS)
def test_group_sites(nr_sites, local_dim, bond_dim, sites_per_group, rgen):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    mpa = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim, randstate=rgen)
    grouped_mpa = mpa.group_sites(sites_per_group)
    op = mpa.to_array()
    grouped_op = grouped_mpa.to_array()
    assert_array_almost_equal(op, grouped_op)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim, sites_per_group',
                     MP_TEST_PARAMETERS_GROUPS)
def test_split_sites(nr_sites, local_dim, bond_dim, sites_per_group, rgen):
    assert (nr_sites % sites_per_group) == 0, \
        'nr_sites not a multiple of sites_per_group'
    plegs = (local_dim,) * (2 * sites_per_group)
    mpa = factory.random_mpa(nr_sites // sites_per_group, plegs, bond_dim, randstate=rgen)
    split_mpa = mpa.split_sites(sites_per_group)
    op = mpa.to_array()
    split_op = split_mpa.to_array()
    assert_array_almost_equal(op, split_op)


###############################################################################
#                         Normalization & Compression                         #
###############################################################################
def assert_lcanonical(ltens, msg=''):
    ltens = ltens.reshape((np.prod(ltens.shape[:-1]), ltens.shape[-1]))
    prod = ltens.conj().T.dot(ltens)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_rcanonical(ltens, msg=''):
    ltens = ltens.reshape((ltens.shape[0], np.prod(ltens.shape[1:])))
    prod = ltens.dot(ltens.conj().T)
    assert_array_almost_equal(prod, np.identity(prod.shape[0]),
                              err_msg=msg)


def assert_correct_normalization(mpo, lnormal_target=None, rnormal_target=None):
    lnormal, rnormal = mpo.normal_form

    # If no targets are given, verify that the data matches the
    # information in `mpo.normal_form`.
    lnormal_target = lnormal_target or lnormal
    rnormal_target = rnormal_target or rnormal

    # If targets are given, verify that the information in
    # `mpo.normal_form` matches the targets.
    assert_equal(lnormal, lnormal_target)
    assert_equal(rnormal, rnormal_target)

    for n in range(lnormal):
        assert_lcanonical(mpo[n], msg="Failure left canonical (n={}/{})"
                          .format(n, lnormal_target))
    for n in range(rnormal, len(mpo)):
        assert_rcanonical(mpo[n], msg="Failure right canonical (n={}/{})"
                          .format(n, rnormal_target))


@pt.mark.parametrize('nr_sites, local_dim, _', MP_TEST_PARAMETERS)
def test_normalization_from_full(nr_sites, local_dim, _, rgen):
    op = factory.random_op(nr_sites, local_dim, randstate=rgen)
    mpo = mp.MPArray.from_array(op, 2)
    assert_correct_normalization(mpo, nr_sites - 1, nr_sites)


# FIXME Add counter to normalization functions
@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_incremental(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo)
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    for site in range(1, nr_sites):
        mpo.normalize(left=site)
        assert_correct_normalization(mpo, site, nr_sites)
        assert_array_almost_equal(op, mpo_to_global(mpo))
        assert mpo.dtype == dtype

    for site in range(nr_sites - 1, 0, -1):
        mpo.normalize(right=site)
        assert_correct_normalization(mpo, site - 1, site)
        assert_array_almost_equal(op, mpo_to_global(mpo))
        assert mpo.dtype == dtype


# FIXME Add counter to normalization functions
@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_jump(nr_sites, local_dim, bond_dim, rgen, dtype):
    # This test assumes at least two sites.
    if nr_sites == 1:
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo)
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    assert_correct_normalization(mpo, center - 1, center)
    assert_array_almost_equal(op, mpo_to_global(mpo))
    assert mpo.dtype == dtype


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_full(nr_sites, local_dim, bond_dim, rgen, dtype):
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo)
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    mpo.normalize(right=1)
    assert_correct_normalization(mpo, 0, 1)
    assert_array_almost_equal(op, mpo_to_global(mpo))
    assert mpo.dtype == dtype

    ###########################################################################
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen, dtype=dtype)
    op = mpo_to_global(mpo)
    assert_correct_normalization(mpo, 0, nr_sites)
    assert_array_almost_equal(op, mpo_to_global(mpo))

    mpo.normalize(left=len(mpo) - 1)
    assert_correct_normalization(mpo, len(mpo) - 1, len(mpo))
    assert_array_almost_equal(op, mpo_to_global(mpo))
    assert mpo.dtype == dtype


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_normalization_default_args(nr_sites, local_dim, bond_dim, rgen):
    # The following normalizations assume at least two sites.
    if nr_sites == 1:
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate=rgen)
    assert_correct_normalization(mpo, 0, nr_sites)

    mpo.normalize(left=1)
    mpo.normalize()
    assert_correct_normalization(mpo, nr_sites - 1, nr_sites)

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate=rgen)
    assert_correct_normalization(mpo, 0, nr_sites)

    # The following normalization assumes at least three sites.
    if nr_sites == 2:
        return

    mpo.normalize(left=1)
    mpo.normalize(right=nr_sites - 2)
    mpo.normalize()
    assert_correct_normalization(mpo, 0, 1)


def test_normalization_compression(rgen):
    """If the bond dimension is too large at the boundary, qr decompostion
    in normalization may yield smaller bond dimension"""
    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20, randstate=rgen)
    mpo.normalize(right=1)
    assert_correct_normalization(mpo, 0, 1)
    assert mpo.bdims[0] == 2

    mpo = factory.random_mpa(sites=2, ldim=2, bdim=20, randstate=rgen)
    mpo.normalize(left=1)
    assert_correct_normalization(mpo, 1, 2)
    assert mpo.bdims[0] == 2


@pt.mark.parametrize('nr_sites, local_dim, bond_dim', MP_TEST_PARAMETERS)
def test_mult_mpo_scalar_normalization(nr_sites, local_dim, bond_dim, rgen):
    if nr_sites < 2:
        # Re-normalization has no effect for nr_sites == 1. There is
        # nothing more to test than :func:`test_mult_mpo_scalar`.
        return

    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim, randstate=rgen)
    op = mpo_to_global(mpo)
    scalar = rgen.randn()

    center = nr_sites // 2
    mpo.normalize(left=center - 1, right=center)
    mpo_times_two = scalar * mpo

    assert_array_almost_equal(scalar * op, mpo_to_global(mpo_times_two))
    assert_correct_normalization(mpo_times_two, center - 1, center)

    mpo *= scalar
    assert_array_almost_equal(scalar * op, mpo_to_global(mpo))
    assert_correct_normalization(mpo, center - 1, center)


#####################################
#  SVD and variational compression  #
#####################################

# nr_sites, local_dims, bond_dim
compr_sizes = pt.mark.parametrize(
    # Start with `2*bond_dim` and compress to `bond_dim`.
    'nr_sites, local_dims, bond_dim',
    (
        (4, 2, 3),
        pt.mark.long((2, (2, 3), 5)),
        pt.mark.long((5, 3, 4)),
        # TODO Create a separate marker for very long tests:
        # (4, (2, 3), 5),
        # (6, 2, 3),
        # (5, (2, 2, 2), 20),  # about  2 minutes (Core i5-3380M)
        # (16, 2, 10),         # about  2 minutes
        # (16, 2, 30),         # about 10 minutes
    )
)

compr_settings = pt.mark.parametrize(
    'comparg',
    (
        dict(method='svd', direction='left'),
        dict(method='svd', direction='right'),
        dict(method='svd', direction='left', relerr=1e-6),
        dict(method='svd', direction='right', relerr=1e-6),
        pt.mark.long(dict(method='var', num_sweeps=1, var_sites=1)),
        dict(method='var', num_sweeps=2, var_sites=1),
        pt.mark.long(dict(method='var', num_sweeps=3, var_sites=1)),
        pt.mark.long(dict(method='var', num_sweeps=1, var_sites=2)),
        dict(method='var', num_sweeps=2, var_sites=2),
        pt.mark.long(dict(method='var', num_sweeps=3, var_sites=2)),
        # See :func:`call_compression` below for the meaning of
        # 'fillbelow'.
        dict(method='var', num_sweeps=2, var_sites=1, startmpa='fillbelow'),
    )
)

# Test compression works for different normalizations of the MPA
# before compression.
compr_normalization = pt.mark.parametrize(
    'normalize',
    (dict(left=1, right=-1), dict()) +
        tuple(pt.mark.long(x) for x in (
        None,
        dict(left='afull'),
        dict(right='afull'),
        dict(left=1), dict(left=-1), dict(right=1), dict(right=-1),
        dict(left=1, right=2), dict(left=-2, right=-1),
        dict(left=1, right=-1),
    ))
)


def _chain_decorators(*args):
    def chain_decorator(f):
        for deco in reversed(args):
            f = deco(f)
        return f
    return chain_decorator

compr_test_params = _chain_decorators(compr_sizes, compr_settings,
                                      compr_normalization)


def normalize_if_applicable(mpa, nmz):
    """Check whether the given normalization can be applied.

    :param mp.MPArray mpa: Will call `mpa.normalize()`
    :param nmz: Keyword arguments for `mpa.normalize()` or `None`

    :returns: True if the normalization has been applied.

    `nmz=None` means not to call `mpa.normalize()` at all.

    The test whether the normalization can be applied is not
    comprehensive.

    """
    # Make sure the input is non-normalized. Otherwise, the output can
    # be more normalized than desired for the test.
    assert mpa.normal_form == (0, len(mpa)), "want non-normalized MPA for test"
    if nmz is not None:
        if nmz.get('left') == 1 and nmz.get('right') == -1 and len(mpa) == 2:
            return False
        mpa.normalize(**nmz)
    return True


def call_compression(mpa, comparg, bonddim, rgen, call_compress=False):
    """Call `mpa.compress` or `mpa.compression` with suitable arguments.

    Does not make a copy of `mpa` in any case.

    :param bonddim: Compress to bond dimension `bonddim`.
    :param call_compress: If `True`, call `mpa.compress` instead of
        `mpa.compression` (the default).
    :param comparg: Settings dict for compression.  If `relerr` is not
        present, add `bdim = bonddim`.  If `startmpa` is equal to
        `'fillbelow'`, insert a random MPA.

    :returns: Compressed MPA.

    """
    if not ('relerr' in comparg) and (comparg.get('startmpa') == 'fillbelow'):
        startmpa = factory.random_mpa(len(mpa), mpa.pdims[0], bonddim,
                                      normalized=True, randstate=rgen,
                                      dtype=mpa.dtype)
        comparg = update_copy_of(comparg, {'startmpa': startmpa})
    else:
        comparg = update_copy_of(comparg, {'bdim': bonddim})

    if (comparg.get('method') == 'var') and not ('startmpa' in comparg):
        comparg = update_copy_of(comparg, {'randstate': rgen})

    if call_compress:
        return mpa.compress(**comparg)
    else:
        return mpa.compression(**comparg)


# We want check compression for inputs with norm different from 1.  In the next
# function and below, we do this with a normalized state multiplied with a
# constant with magnitude different from 1.  This is to avoid errors like
# "123456789.1 and 123456789.2 are not equal to six decimals" and is related to
# the fixme at the module start.


@compr_test_params
def test_compression_and_compress(nr_sites, local_dims, bond_dim, normalize, comparg, rgen):
    """Test that .compression() and .compress() produce identical results.

    """
    # See comment above on "4.2 *"
    mpa = 4.2 * factory.random_mpa(nr_sites, local_dims, bond_dim * 2,
                                   normalized=True, randstate=rgen)
    if not normalize_if_applicable(mpa, normalize):
        return

    comparg = comparg.copy()
    if comparg['method'] == 'var':
        # Exact equality between `compr` and `compr2` below requires
        # using the same start vector in both cases.
        comparg['startmpa'] = factory.random_mpa(nr_sites, local_dims, bond_dim,
                                                 randstate=rgen)

    # The results from .compression() and .compress() must match
    # exactly. No numerical difference is allowed.
    compr2 = mpa.copy()
    overlap2 = call_compression(compr2, comparg, bond_dim, rgen, call_compress=True)
    compr, overlap = call_compression(mpa, comparg, bond_dim, rgen)
    assert_almost_equal(overlap, overlap2)
    # FIXME Why do they not agree completely? We are doing the same thing...
    assert_mpa_identical(compr, compr2, decimal=12)


@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@compr_test_params
def test_compression_result_properties(nr_sites, local_dims, bond_dim,
                                        normalize, comparg, rgen, dtype):
    """Test general properties of the MPA coming from a compression.

    * Compare SVD compression against simpler implementation

    * Check that all implementations return the correct overlap

    * Check that the bond dimension has decreased and that it is as
      prescribed

    * Check that the normalization advertised in the result is correct

    * Check that compression doesnt change the dtype

    TODO: The worst case for compression is that all singular values
    have the same size.  This gives a fidelity lower bound for the
    compression result.  Check that lower bound.

    FIXME: Make this test a wrapper around MPArray.compression() to
    reduce code duplication.  This wrapper would replace
    call_compression().  This would also apply more tests
    .compress(). At the moment, we mostly test .compression().

    """
    mpa = 4.2 * factory.random_mpa(nr_sites, local_dims, bond_dim * 2,
                                   normalized=True, randstate=rgen, dtype=dtype)
    if not normalize_if_applicable(mpa, normalize):
        return
    compr, overlap = call_compression(mpa.copy(), comparg, bond_dim, rgen)

    # 'relerr' is currently 1e-6 and no bond_dim is provided, so no
    # compression will occur.
    if 'relerr' not in comparg:
        # Check that the bond dimension has changed.
        assert compr.bdim < mpa.bdim
        # Check that the target bond dimension is satisfied
        assert compr.bdim <= bond_dim

    # Check that the inner product is correct.
    assert_almost_equal(overlap, mp.inner(mpa, compr))

    # SVD: Check that .normal_form is as expected.
    if comparg['method'] == 'svd':
        normtarget = {'left': (0, 1), 'right': (len(compr) - 1, len(compr))}
        assert compr.normal_form == normtarget[comparg['direction']]

    # Check the content of .normal_form is correct.
    assert_correct_normalization(compr)
    assert compr.dtype == dtype

    # SVD: compare with alternative implementation
    if comparg['method'] == 'svd' and 'relerr' not in comparg:
        alt_compr = _svd_compression_full(mpa, comparg['direction'], bond_dim)
        compr = compr.to_array()
        assert_array_almost_equal(alt_compr, compr)


@pt.mark.skip(reason="Depends strongly on initial guess, work in progress")
@pt.mark.parametrize('dtype', MP_TEST_DTYPES)
@compr_test_params
def test_var_no_worse_than_svd(nr_sites, local_dims, bond_dim, normalize,
                                comparg, rgen, dtype):
    """Var: If we perform enough sweeps (enough = empirical value), we
    expect to be at least as good as SVD compression (up to a small
    tolerance)."""
    if (comparg['method'] != 'var'):
        return

    # Below, we want to check that var is at least as good as SVD
    # compression.  This requires a big enough number of sweeps.
    # Because a big number of sweeps is not required in any other
    # test, we override the number of sweeps here.
    comparg = update_copy_of(comparg, {'num_sweeps': 20 // comparg['var_sites']})

    mpa = 4.2 * factory.random_mpa(nr_sites, local_dims, bond_dim * 5,
                                   normalized=True, randstate=rgen, dtype=dtype)
    mpa.compress()
    _, overlap_var = call_compression(mpa.copy(), comparg, bond_dim, rgen)

    for direction in ('left', 'right'):
        _, overlap_svd = mpa.compression(method='svd', bdim=bond_dim)
        assert overlap_var >= overlap_svd * (1 - 1e-6)


@compr_test_params
def test_compression_bonddim_noincrease(nr_sites, local_dims, bond_dim,
                                         normalize, comparg, rgen):
    """Check that bond dimension does not increase if the target bond
    dimension is larger than the MPA bond dimension

    """
    if 'relerr' in comparg:
        return  # Test does not apply
    mpa = 4.2 * factory.random_mpa(nr_sites, local_dims, bond_dim, normalized=True,
                                   randstate=rgen)
    norm = mp.norm(mpa.copy())
    if not normalize_if_applicable(mpa, normalize):
        return

    for factor in (1, 2):
        compr, overlap = call_compression(mpa, comparg, bond_dim * factor, rgen)
        assert_almost_equal(overlap, norm**2)
        assert_mpa_almost_equal(compr, mpa, full=True)
        assert (np.array(compr.bdims) <= np.array(mpa.bdims)).all()


@pt.mark.parametrize('add', ('zero', 'self', 'self2'))
@compr_test_params
def test_compression_trivialsum(nr_sites, local_dims, bond_dim, normalize,
                                comparg, add, rgen):
    """Check that `a + b` compresses exactly to a multiple of `a` if `b`
    is equal to one of `0`, `a` or `-2*a`

    """
    mpa = 4.2 * factory.random_mpa(nr_sites, local_dims, bond_dim, normalized=True,
                                   randstate=rgen)
    norm = mp.norm(mpa.copy())
    if not normalize_if_applicable(mpa, normalize):
        return
    zero = factory.zero(nr_sites, local_dims, bond_dim)
    choices = {'zero': (zero, 1), 'self': (mpa, 2), 'self2': (-2*mpa, -1)}
    add, factor = choices[add]

    msum = mpa + add
    assert_mpa_almost_equal(msum, factor * mpa, full=True)

    # Check that bond dimension has increased (they exactly add)
    for dim1, dim2, sum_dim in zip(mpa.bdims, add.bdims, msum.bdims):
        assert dim1 + dim2 == sum_dim

    compr, overlap = call_compression(msum, comparg, bond_dim, rgen)
    assert_almost_equal(overlap, (norm * factor)**2)
    assert_mpa_almost_equal(compr, factor * mpa, full=True)
    assert (np.array(compr.bdims) <= np.array(mpa.bdims)).all()


#######################################
#  Compression test helper functions  #
#######################################


def _svd_compression_full(mpa, direction, target_bonddim):
    """Re-implement MPArray.compress('svd') but on the level of the full
    matrix representation, i.e. it truncates the Schmidt-decompostion
    on each bipartition sequentially.

    We have two implementations and check that both produce the same
    output.  This is useful because the correctness of the MPA-based
    implementation depends crucially on correct normalization at each
    step, while the implementation here is much simpler.

    :param mpa: The MPA to compress
    :param direction: 'right' means sweep from left to right,
        'left' vice versa
    :param target_bonddim: Compress to this bond dimension
    :returns: Result as numpy.ndarray

    """
    def singlecut(array, nr_left, plegs, target_bonddim):
        array_shape = array.shape
        array = array.reshape((np.prod(array_shape[:nr_left * plegs]), -1))
        u, s, v = svd(array, full_matrices=False)
        u = u[:, :target_bonddim]
        s = s[:target_bonddim]
        v = v[:target_bonddim, :]
        opt_compr = np.dot(u * s, v)
        opt_compr = opt_compr.reshape(array_shape)
        return opt_compr

    array = mpa.to_array()
    plegs = mpa.plegs[0]
    nr_sites = len(mpa)
    if direction == 'right':
        nr_left_values = range(1, nr_sites)
    else:
        nr_left_values = range(nr_sites-1, 0, -1)
    for nr_left in nr_left_values:
        array = singlecut(array, nr_left, plegs, target_bonddim)
    return array
