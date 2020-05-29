import itertools
import pytest
import numpy as np
from test_discovery import get_test_data

from refl1d import abeles
from refl1d.reflectivity import reflectivity_amplitude
from refl1d.names import Stack, QProbe, Experiment, SLD

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

    if data.shape[1] == 4:
        # resolution smeared
        if backend == abeles.refl:
            # no way of setting backend for resolution smearing tests
            pass

        # TODO, when QProbe gets oversampling
        pytest.xfail("refl1d QProbe does not have oversample")
        resolution_test(slabs, data)
    elif data.shape[1] < 4:
        # no resolution data, just test kernel
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


def resolution_test(slabs, data):
    stk = Stack()
    for i, slab in enumerate(slabs[::-1]):
        m = SLD(f"layer {i}", rho=slab[1], irho=slab[2])
        stk |= m(thickness=slab[0], interface=slab[-1])

    probe = QProbe(Q=data[:, 0], dQ=data[:, 3])
    # TODO, oversample when QProbe can do that

    M = Experiment(stk, probe)
    _, R = M.reflectivity()
    np.testing.assert_allclose(R, data[:, 1], rtol=0.03)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_refl1d(nsd, backend)
