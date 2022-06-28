"""Code related to physical models

Contents:

- Hamiltonian and analytic ground state energy of the cyclic XY model

References:

.. [LSM61] Lieb, Schultz and Mattis (1961). Two soluble models of an
   antiferromagnetic chain.

"""


import numpy as np
import scipy.sparse as sp

import mpnum as mp


# Pauli operators
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])
pauli_Z = np.diag([1, -1])

# Pauli operators as length-1 MPOs
mpo_X, mpo_Y, mpo_Z = (mp.MPArray.from_array(x, ndims=2)
                       for x in (pauli_X, pauli_Y, pauli_Z))


def cXY_local_terms(nr_sites, gamma):
    r"""Local terms of the cyclic XY model (MPOs)

    :param nr_sites: Number of spin one-half sites
    :param gamma: Asymmetry parameter
    :returns: List :code:`terms` of length :code:`nr_sites` (MPOs)

    The term :code:`terms[i]` acts on spins :code:`(i, i + 1)` and
    spin :code:`nr_sites` is the same as the first spin.

    The Hamiltonian of the cyclic XY model is given by
    [:ref:`LSM61 <LSM61>`, Eq. 2.1]:

    .. math::

       H_\gamma = \sum_{i=1}^{N}   (1+\gamma) S^x_i S^x_{i+1}
                                 + (1-\gamma) S^y_i S^y_{i+1}

    with :math:`S^j_{N+1} = S^j_{1}`. The function :func:`cXY_E0`
    returns the exact ground state energy of this Hamiltonian.

    """
    local = ((1 + gamma) * 0.25 * mp.chain([mpo_X, mpo_X])
             + (1 - gamma) * 0.25 * mp.chain([mpo_Y, mpo_Y]))
    return (local,) * nr_sites


def cXY_E0(nr_sites, gamma):
    r"""Ground state energy of the cyclic XY model

    :param nr_sites: Number of spin one-half sites
    :param gamma: Asymmetry parameter
    :returns: Exact energy of the ground state

    This function is implemented for :code:`nr_sites` which is an odd
    multiple of two.  In this case, the ground state energy of the XY
    model is given by (Eqs. (A-12), (2.20) of [LSM61]_)

    .. math::

       E_0 = -\frac12 \sum_{l=0}^{N-1} \Lambda_{k(l)}

    with (Eqs. (2.18b), (2.18c))

    .. math::

       \Lambda_k^2 = 1 - (1 - \gamma^2) [\sin(k)]^2, \quad
       k(l) = \frac{2\pi}{N} \left( l - \frac N2 \right)

    and :math:`\Lambda_k \ge 0`.

    """
    N = nr_sites
    if (N % 2) != 0 or (N % 4) == 0:
        raise ValueError('nr_sites must be an odd multiple of two')
    l = np.arange(N)
    k = (2 * np.pi / N) * (l - N / 2)  # Eq. (2.18c)
    Lambda2 = 1 - (1 - gamma**2) * (np.sin(k))**2  # Eq. (2.18b)
    Lambda = Lambda2**0.5
    E0 = -0.5 * Lambda.sum()  # Eqs. (2.20), (A-12)
    return E0


def sparse_cH(terms, ldim=2):
    """Construct a sparse cyclic nearest-neighbour Hamiltonian

    :param terms: List of nearst-neighbour terms (square array or MPO,
        see return value of :func:`cXY_local_terms`)
    :param ldim: Local dimension

    :returns: The Hamiltonian as sparse matrix

    """
    H = 0
    N = len(terms)
    for pos, term in enumerate(terms[:-1]):
        if hasattr(term, 'lt'):
            # Convert MPO to regular matrix
            term = term.to_array_global().reshape((ldim**2, ldim**2))
        left = sp.eye(ldim**pos)
        right = sp.eye(ldim**(N - pos - 2))
        H += sp.kron(left, sp.kron(term, right))
    # The last term acts on the first and last site.
    cyc = terms[-1]
    middle = sp.eye(ldim**pos)
    for i in range(cyc.ranks[0]):
        H += sp.kron(cyc.lt[0][0, ..., i], sp.kron(middle, cyc.lt[1][i, ..., 0]))
    return H


def mpo_cH(terms):
    """Construct an MPO cyclic nearest-neighbour Hamiltonian

    :param terms: List of nearst-neighbour terms (MPOs, see return
        value of :func:`cXY_local_terms`)

    :returns: The Hamiltonian as MPO

    .. note::

       It may not be advisable to call
       :func:`mp.MPArray.canonicalize()` on a Hamiltonian, e.g.:

       >>> mpoH = mpo_cH(cXY_local_terms(nr_sites=100, gamma=0))
       >>> abs1 = max(abs(lt).max() for lt in mpoH.lt)
       >>> mpoH.canonicalize()
       >>> abs2 = np.round(max(abs(lt).max() for lt in mpoH.lt), -3)
       >>> print('{:.3f}  {:.2e}'.format(abs1, abs2))
       1.000  2.79e+15

       The Hamiltonian generally has a large Frobenius norm because
       local terms are embedded with identity matrices. This causes
       large tensor entries of canonicalization which will eventually
       overflow the numerical maximum (the overflow happens somewhere
       between 2000 and 3000 sites in this example). One could embed
       local terms with Frobenius-normalized identity matrices
       instead, but this would make the eigenvalues of H exponentially
       (in :code:`nr_sites`) small. This would eventually cause
       numerical underflows.

    """
    H = mp.local_sum(terms[:-1])
    # The last term acts on the first and last site.
    H += mp.inject(terms[-1], pos=1, num=len(H) - 2)
    return H
