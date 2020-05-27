import itertools
import numpy as np
from test_discovery import get_test_data

from refl1d import abeles
from refl1d.reflectivity import reflectivity_amplitude


def run_tests():
    # abeles.refl is a Python calculator, reflectivity_amplitude uses
    # a C extension
    f_amplitudes = [abeles.refl, reflectivity_amplitude]

    for f_amplitude, test in itertools.product(f_amplitudes, get_test_data()):
        slabs, data = test

        r = f_amplitude(
            data[:, 0] / 2.0,
            slabs[:, 0],
            slabs[:, 1],
            irho=slabs[:, 2],
            sigma=slabs[1:, 3],
        )
        R = (r * np.conj(r)).real

        assert R.shape == data[:, 1].shape

        np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    run_tests()
