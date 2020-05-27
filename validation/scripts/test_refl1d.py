import itertools
import numpy as np
from test_discovery import get_test_data

from refl1d import abeles
from refl1d.reflectivity import reflectivity_amplitude


def test_refl1d():
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
    # abeles.refl is a Python calculator, reflectivity_amplitude uses
    # a C extension
    f_amplitudes = [abeles.refl, reflectivity_amplitude]

    for f_amplitude in f_amplitudes:
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
    test_refl1d()
