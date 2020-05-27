import itertools
import numpy as np
from test_discovery import get_test_data

from refnx.reflect.reflect_model import abeles, use_reflect_backend


def test_refnx():
    # NCOLS:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    # test no resolution first
    for slabs, data in get_test_data():
        # no resolution data, just test kernel
        if data.shape[1] < 4:
            kernel_test(slabs, data)


def kernel_test(slabs, data):
    # cython backend may or may not be present on all systems
    backends = ["c", "python", "cython"]

    for backend in backends:
        with use_reflect_backend(backend) as abeles:
            R = abeles(data[:, 0], slabs)
        assert R.shape == data[:, 1].shape

        np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    test_refnx()
