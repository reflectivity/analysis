import itertools
import pytest
import numpy as np
from test_discovery import get_test_data

from refl1d import abeles
from refl1d.reflectivity import reflectivity_amplitude

# abeles.refl is a Python calculator, reflectivity_amplitude uses
# a C extension
backends = [abeles.refl, reflectivity_amplitude]
tests_backends = list(itertools.product(get_test_data(), backends))
ids = [f"{t[0][0]}-{t[1]}" for t in tests_backends]


@pytest.mark.parametrize("nsd, backend", tests_backends, ids=ids)
def test_refl1d(nsd, backend):
    """
    Run validation for refl1d.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
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
    Test the reflectivity kernels for ref1d.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
    """
    r = backend(
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
    for nsd, backend in tests_backends:
        test_refl1d(nsd, backend)
