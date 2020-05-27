import itertools
import numpy as np
from test_discovery import get_test_data

from refnx.reflect.reflect_model import abeles, use_reflect_backend


def run_tests():
    # cython backend may or may not be present on all systems
    backends = ["c", "python", "cython"]

    for backend, test in itertools.product(backends, get_test_data()):
        slabs, data = test
        with use_reflect_backend(backend) as abeles:
            R = abeles(data[:, 0], slabs)
        assert R.shape == data[:, 1].shape

        np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    run_tests()
