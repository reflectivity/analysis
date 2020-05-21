import numpy as np
from test_discovery import get_test_data

from refnx.reflect.reflect_model import abeles


def run_tests():

    for slabs, data in get_test_data():
        R = abeles(data[:, 0], slabs)
        assert R.shape == data[:, 1].shape

        np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    run_tests()
