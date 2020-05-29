import itertools
import warnings
import pytest
import numpy as np
from test_discovery import get_test_data

from refnx.reflect import use_reflect_backend, SLD, ReflectModel, Structure


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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using the SLOW reflectivity calculation.",
            category=UserWarning,
        )
        if data.shape[1] == 4:
            # resolution smeared
            resolution_test(slabs, data, backend)
        elif data.shape[1] < 4:
            # no resolution data, just test kernel
            kernel_test(slabs, data)


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


def resolution_test(slabs, data, backend):
    structure = Structure()
    for i, slab in enumerate(slabs):
        m = SLD(complex(slab[1], slab[2]))
        structure |= m(slab[0], slab[-1])

    with use_reflect_backend(backend):
        model = ReflectModel(structure, bkg=0.0)
        model.quad_order = 17
        R = model.model(
            data[:, 0], x_err=data[:, -1] * 2 * np.sqrt(2 * np.log(2))
        )
        np.testing.assert_allclose(R, data[:, 1], rtol=0.03)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_refnx(nsd, backend)
