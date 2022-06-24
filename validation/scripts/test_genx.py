import itertools
import warnings
import pytest
import numpy as np
from test_discovery import get_test_data, get_polarised_test_data

import genx.models.spec_nx as model
from genx.models.lib.physical_constants import muB_to_SL

# enumerate backends,
# 'neutron pol' is half-/non-polarized Parratt implementation
# and 'neutron pol spin flip' is full matrix implementation.
# All other options in GenX use the same backends.
backends = ["neutron pol", "neutron pol spin flip"]
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
        if backend == "neutron pol spin flip":
            # memory issues in matrix formalism if too many data points
            inst.respoints = 101
        else:
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
        if backend == "neutron pol spin flip":
            np.testing.assert_allclose(R, data[:, 1], rtol=0.005)
        else:
            np.testing.assert_allclose(R, data[:, 1], rtol=0.001)
    else:
        np.testing.assert_allclose(R, data[:, 1], rtol=0.001)


# "neutron pol" does use Parratt with different SLD
# for spin-up/spin-down, only valid for M||P
pol_backends = ["neutron pol", "neutron pol spin flip"]
pol_tests_backends = list(
    itertools.product(get_polarised_test_data(), pol_backends)
)
pol_ids = [f"{t[0][0]}-{t[1]}" for t in pol_tests_backends]


def angle_between(AGUIDE, theta):
    # calculate angle between magnetization and guide field
    M = np.array(
        [
            np.cos(theta / 180.0 * np.pi),
            np.sin(theta / 180.0 * np.pi),
            0.0 * theta,
        ]
    )
    P = np.array(
        [0.0, np.sin(AGUIDE / 180.0 * np.pi), np.cos(AGUIDE / 180.0 * np.pi)]
    )

    phi = np.arccos(np.clip(np.dot(M.T, P), -1.0, 1.0))
    return phi * 180.0 / np.pi


@pytest.mark.parametrize("nsd, backend", pol_tests_backends, ids=pol_ids)
def test_genx_pol(nsd, backend):
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

    test_name, slabs, data, AGUIDE, H = nsd
    # convert from AGUIDE+theta to relative angle phi
    slabs = np.array(slabs)
    slabs[:, 3] = angle_between(AGUIDE, slabs[:, 3])

    if (
        backend != "neutron pol spin flip"
        and ((np.array(slabs)[:, 3] % 180) != 0).any()
    ):
        pytest.skip(
            "models with spin-flip can't be "
            "described by the 'neutron pol' model"
        )
        return
    if H > 0:
        pytest.skip("GenX does not support Zeeman splitting")
        return
    kernel_test_pol(slabs, data, backend)


def kernel_test_pol(slabs, data, backend):
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
    for thickness, rsld, isld, theta, sldm, sigma in slabs:
        if backend == "neutron pol":
            sldm *= 1 - (theta % 360 > 0).astype(int) * 2
        layers.append(
            model.Layer(
                b=(rsld - 1j * isld),
                dens=0.1,
                d=thickness,
                sigma=sigma,
                magn=sldm / muB_to_SL * 1e-5,
                magn_ang=theta,
            )
        )
    # layers.reverse() # the layer order in pol tests
    # is currently reversed compared to unpol
    stack = model.Stack(Layers=list(layers[1:-1]), Repetitions=1)
    sample = model.Sample(
        Stacks=[stack], Ambient=layers[-1], Substrate=layers[0]
    )

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

    inst.pol = "uu"
    Rup = sample.SimSpecular(Q, inst)
    if backend == "neutron pol spin flip":
        inst.pol = "ud"
        Rflip = sample.SimSpecular(Q, inst)
    else:
        Rflip = 0.0 * Rup
    inst.pol = "dd"
    Rdown = sample.SimSpecular(Q, inst)

    assert Rup.shape == data[:, 1].shape

    np.testing.assert_allclose(Rdown, data[:, 1], rtol=1e-3)
    np.testing.assert_allclose(Rflip, data[:, 2], rtol=1e-3)
    # without Zeeman splitting there is no difference between ud and du
    np.testing.assert_allclose(Rup, data[:, 4], rtol=1e-3)


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_genx(nsd, backend)
    for nsd, backend in pol_tests_backends:
        test_genx_pol(nsd, backend)
