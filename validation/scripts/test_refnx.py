import itertools
import warnings
import pytest
import numpy as np
from test_discovery import get_test_data, get_polarised_test_data

from refnx.reflect import (
    use_reflect_backend,
    SLD,
    ReflectModel,
    Structure,
    MagneticSlab,
    PolarisedReflectModel,
)


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


pol_tests = list(get_polarised_test_data())
pol_ids = [f"{t[0]}" for t in pol_tests]


@pytest.mark.parametrize("nsd", pol_tests, ids=pol_ids)
def test_refnx_pol(nsd):
    """
    Run validation for genx.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation
    test_name, slabs, data, AGUIDE, H = nsd
    if H > 0:
        pytest.skip("refnx does not support Zeeman splitting (yet)")
        return

    slabs = np.array(slabs)
    kernel_test_pol(slabs, data, AGUIDE)


def kernel_test_pol(slabs, data, AGUIDE):
    """
    Test the polarised reflectivity kernel for refnx.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays (--, -+, +-, ++)
    AGUIDE: float
        AGUIDE (degrees)
    """
    Q = data[:, 0]
    npts = len(Q)

    layers = []
    for thickness, rsld, isld, theta, sldm, sigma in slabs:
        slab = MagneticSlab(
            thickness, rsld + 1j * isld, sigma, sldm, thetaM=theta
        )
        layers.append(slab)
    s = Structure(components=layers)
    model = PolarisedReflectModel(s, bkgs=0.0, dq=0.0, Aguide=AGUIDE)
    qq = np.full((len(Q) * 4, 4), np.nan)
    qq[0:npts, 0] = Q
    qq[npts : 2 * npts, 1] = Q
    qq[2 * npts : 3 * npts, 2] = Q
    qq[3 * npts : 4 * npts, 3] = Q
    R = model(qq)
    Ruu = R[0:npts]
    Rud = R[npts : 2 * npts]
    Rdu = R[2 * npts : 3 * npts]
    Rdd = R[3 * npts : 4 * npts]
    np.testing.assert_allclose(Ruu, data[:, 4], rtol=8e-5)
    np.testing.assert_allclose(Rdd, data[:, 1], rtol=8e-5)
    np.testing.assert_allclose(Rud, data[:, 3], rtol=8e-5)
    np.testing.assert_allclose(Rdu, data[:, 2], rtol=8e-5)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_refnx(nsd, backend)
    for nsd, backend in pol_tests_backends:
        test_refnx_pol(nsd, backend)
