import numpy as np
import pytest


def pytest_configure():
    # TODO Move these constants to mpnum._testing or find a better solution.
    # nr_sites, local_dim, rank
    pytest.MP_TEST_PARAMETERS=[(1, 7, np.nan), (2, 3, 3), (3, 2, 4), (6, 2, 4),
                            (4, 3, 5), (5, 2, 1)],
    pytest.MP_TEST_DTYPES=[np.float_, np.complex_]


@pytest.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)
