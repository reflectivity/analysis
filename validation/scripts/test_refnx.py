import itertools
import pytest
import numpy as np
from test_discovery import get_test_data

from refnx.reflect.reflect_model import abeles, use_reflect_backend


# cython backend may or may not be present on all systems
backends = ["c", "python", "cython", "pyopencl"]
tests_backends = list(itertools.product(get_test_data(), backends))
ids = [f"{t[0][0]}-{t[1]}" for t in tests_backends]


@pytest.mark.parametrize("nsd, backend", tests_backends, ids=ids)
def test_refnx(nsd, backend):
    """
    Run validation for refnx.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    backend: {"c", "python", "cython", "pyopencl"}
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    test_name, slabs, data = nsd
    # test no resolution first
    # no resolution data, just test kernel
    if data.shape[1] < 4:
        kernel_test(slabs, data, backend)


def kernel_test(slabs, data, backend):
    """
    Test the reflectivity kernels for refnx.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    backend: {"c", "python", "cython", "pyopencl"}
    """
    with use_reflect_backend(backend) as abeles:
        R = abeles(data[:, 0], slabs)
    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_refnx(nsd, backend)
