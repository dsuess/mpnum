# encoding: utf-8
"""An informationally complete d-level POVM.

The POVM simplifies to measuring Paulis matrices in the case of
qubits.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_almost_equal

from six.moves import range, zip


class POVM(object):
    """Represent a Positive Operator-Valued Measure (POVM).
    """
    def __init__(self, elements, info_complete=False, pinv=np.linalg.pinv):
        """Create a POVM.

        The caller must supply whether the POVM elements are
        informationally complete.

        :param elements: @todo
        :param info_complete: Is the POVM informationally complete (IC)
            (default False)
        :param pinv: Pseudo-inverse function to be used (default
            numpy.linalg.pinv)

        """
        self._elements = np.asarray(elements)
        self._info_complete = info_complete
        self._pinv = pinv

    def __len__(self):
        return len(self._elements)

    def __iter__(self):
        return iter(self._elements)

    def __getitem__(self, index):
        return self._elements[index]

    @classmethod
    def from_vectors(cls, vecs, info_complete=False):
        """Generates a POVM consisting of rank 1 projectors based on the
        corresponding vectors.

        :param vecs: Iterable of np.ndarray with ndim=1 representing the
            vectors for the POVM
        :param info_complete: Is the POVM informationally complete
            (default False)
        :returns:
        """
        povm_elems = np.array([np.outer(vec, vec.conj()) for vec in vecs])
        return cls(povm_elems, info_complete=info_complete)

    @property
    def probability_map(self):
        """Map that takes a raveled density matrix to the POVM probabilities

        The following two return the same::

            probab = np.array([ np.trace(np.dot(elem, rho)) for elem in a_povm ])
            probab = np.dot(a_povm.probability_map, rho.ravel())
        """
        # tr(M rho) = \sum_ij M_ji rho_ij = \sum_ij (M^T)_ij rho_ij
        return self._elements.transpose((0, 2, 1)).reshape(len(self), -1)

    @property
    def linear_inversion_map(self):
        """Map that reconstructs a density matrix with linear inversion.

        Linear inversion is performed by taking the Moore--Penrose
        pseudoinverse of self.probability_map.

        """
        return self._pinv(self.probability_map)

    @property
    def informationally_complete(self):
        return self._info_complete


def x_povm(dim):
    """The X POVM simplifies to measuring Pauli X eigenvectors for dim=2.

    :param dim: Dimension of the system
    :returns: POVM with generalized X measurments
    """
    vectors = np.zeros([dim * (dim - 1), dim])
    k = 0
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            vectors[k, i], vectors[k, j] = 1.0, 1.0
            k += 1
            vectors[k, i], vectors[k, j] = 1.0, -1.0
            k += 1

    vectors /= np.sqrt(2 * (dim - 1))
    return POVM.from_vectors(vectors, info_complete=False)


def y_povm(dim):
    """The Y POVM simplifies to measuring Pauli Y eigenvectors for dim=2.

    :param dim: Dimension of the system
    :returns: POVM with generalized Y measurments
    """
    vectors = np.zeros([dim * (dim - 1), dim], dtype=complex)
    k = 0
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            vectors[k, i], vectors[k, j] = 1.0, 1.0j
            k += 1
            vectors[k, i], vectors[k, j] = 1.0, -1.0j
            k += 1

    vectors /= np.sqrt(2 * (dim - 1))
    return POVM.from_vectors(vectors, info_complete=False)


def z_povm(dim):
    """The Z POVM simplifies to measuring Pauli Z eigenvectors for dim=2.

    :param dim: Dimension of the system
    :returns: POVM with generalized Z measurments
    """
    return POVM.from_vectors(np.eye(dim, dim), info_complete=False)


def pauli_parts(dim):
    """The POVMs used by :func:`pauli_povm` as a list

    For `dim > 3`, :func:`x_povm` and :func:`y_povm` are returned. For
    `dim = 2`, :func:`z_povm` is included as well.

    :param dim: Dimension of the system
    :returns: Tuple of :class:`POVMs <POVM>`

    """
    assert dim > 1, "What do you mean by 1-dim. Pauli measurements?"
    if dim == 2:
        return (x_povm(dim), y_povm(dim), z_povm(dim))
    else:
        return (x_povm(dim), y_povm(dim))


def pauli_povm(dim):
    """An informationally complete d-level POVM that simplifies to measuring
    Pauli matrices in the case d=2.

    :param dim: Dimension of the system
    :returns: :class:`POVM` with (generalized) Pauli measurments
    """
    parts = pauli_parts(dim)
    return concat(parts, (1/len(parts),) * len(parts), info_complete=True)


def concat(povms, weights, info_complete=False):
    """Combines the POVMs given in `povms` according the weights given to a new
    POVM.

    :param povms: Iterable of POVM
    :param weights: Iterable of real numbers, should sum up to one
    :param info_complete: Is the resulting POVM informationally complete
    :returns: POVM

    """
    assert_almost_equal(sum(weights), 1.0)
    elements = sum(([weight * elem for elem in povm]
                    for povm, weight in zip(povms, weights)), [])
    return POVM(elements, info_complete=info_complete)
