import numpy as np
from test_discovery import get_test_data

from refl1d import abeles


def run_tests():

    for slabs, data in get_test_data():
        z = abeles.refl(
            data[:, 0] / 2.0,
            slabs[:, 0],
            slabs[:, 1],
            irho=slabs[:, 2],
            sigma=slabs[1:, 3],
        )
        R = z.real ** 2 + z.imag ** 2

        assert R.shape == data[:, 1].shape

        np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    run_tests()
