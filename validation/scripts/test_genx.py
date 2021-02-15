import itertools
import warnings
import pytest
import numpy as np
from test_discovery import get_test_data

import genx.models.spec_nx as model

# enumerate backends
# 'neutron pol spin-flip' requires fix in GenX to including non-air ambient
# layer
backends = ["neutron pol"]
tests_backends = list(itertools.product(get_test_data(), backends))
ids = [f"{t[0][0]}-{t[1]}" for t in tests_backends]


@pytest.mark.parametrize("nsd, backend", tests_backends, ids=ids)
def test_genx(nsd, backend):
    """
    Run validation for genx.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    backend: {parratt, matrix}
        function for reflectance calculation
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    test_name, slabs, data = nsd

    kernel_test(slabs, data, backend)


def kernel_test(slabs, data, backend):
    """
    Test the reflectivity kernels for genx.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    backend: {parratt, matrix}
        function for reflectance calculation
    """
    Q = data[:, 0]

    layers = []
    for thickness, rsld, isld, sigma in slabs:
        layers.append(
            model.Layer(
                b=(rsld - 1j * isld), dens=0.1, d=thickness, sigma=sigma
            )
        )
    layers.reverse()
    stack = model.Stack(Layers=list(layers[1:-1]), Repetitions=1)
    sample = model.Sample(
        Stacks=[stack], Ambient=layers[-1], Substrate=layers[0]
    )
    # print(sample)

    inst = model.Instrument(
        probe=backend,
        wavelength=1.54,
        coords="q",
        I0=1,
        res=0,
        restype="no conv",
        respoints=5,
        resintrange=2,
        beamw=0.1,
        footype="no corr",
        samplelen=10,
        pol="uu",
    )
    if data.shape[1] == 4:
        dQ = data[:, 3]
        inst.restype = "full conv and varying res."
        inst.res = dQ
        inst.respoints = (
            10001  # try to use same convolution as ref1d when generating
        )
        inst.resintrange = 3.5

    # print(inst)
    R = sample.SimSpecular(Q, inst)

    assert R.shape == data[:, 1].shape
    if data.shape[1] == 4:
        # validation accuracy is reduced for resolution runs, as strongly
        # depends on numerical convolution scheme
        np.testing.assert_allclose(R, data[:, 1], rtol=0.001)
    else:
        np.testing.assert_allclose(R, data[:, 1], rtol=0.001)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_genx(nsd, backend)
