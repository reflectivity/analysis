import pytest
import numpy as np
from test_discovery import get_test_data

import bornagain as ba
from bornagain import angstrom
from bornagain.numpyutil import Arrayf64Converter as dac


def get_sample(slabs):
    """
    Defines sample and returns it. Note that SLD-based materials are used.
    """
    # creating materials
    multi_layer = ba.Sample()

    ambient = ba.MaterialBySLD("ma", slabs[0, 1] * 1e-6, 0)
    layer = ba.Layer(ambient)
    multi_layer.addLayer(layer)

    for slab in slabs[1:-1]:
        material = ba.MaterialBySLD("stuff", slab[1] * 1e-6, slab[2] * 1e-6)

        autocorr = ba.SelfAffineFractalModel(slab[3] * angstrom, 0.7, 250*angstrom)
        roughness = ba.Roughness(autocorr, ba.ErfTransient())

        layer = ba.Layer(material, slab[0] * angstrom, roughness)
        multi_layer.addLayer(layer)

    substrate = ba.MaterialBySLD("msub", slabs[-1, 1] * 1e-6, 0)

    autocorr = ba.SelfAffineFractalModel(slabs[-1, 3] * angstrom, 0.7, 250*angstrom)
    roughness = ba.Roughness(autocorr, ba.ErfTransient())

    layer = ba.Layer(substrate, 0, roughness)

    multi_layer.addLayer(layer)

    return multi_layer


def get_simulation(qzs, sample):
    """
    Defines and returns specular simulation
    with a qz-defined beam
    """
    # bornagain requires Qz in nm
    scan = ba.QzScan(qzs * 10.0)
    simulation = ba.SpecularSimulation(scan, sample)
    # simulation.setScan(scan)
    return simulation


def get_simulation_smeared(qzs, dqzs, sample):
    """
    Defines and returns specular simulation
    with a qz-defined beam
    """
    # 3.5 sigma to sync with refnx
    # n_sig = 3.5
    # n_samples = 21

    distr = ba.DistributionGaussian(0.0, 1.0, 21, 3.5)
    scan = ba.QzScan(qzs * 10.0)
    scan.setVectorResolution(distr, dqzs * 10.0)

    simulation = ba.SpecularSimulation(scan, sample)

    return simulation


tests = list(get_test_data())
ids = [f"{t[0]}" for t in tests]


@pytest.mark.parametrize("nsd", tests, ids=ids)
def test_bornagain(nsd):
    """
    Run validation for BornAgain.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    test_name, slabs, data = nsd

    if data.shape[1] == 4:
        # resolution smeared
        resolution_test(slabs, data)
    elif data.shape[1] < 4:
        # no resolution data, just test kernel
        kernel_test(slabs, data)


def resolution_test(slabs, data):
    sample = get_sample(slabs)
    simulation = get_simulation_smeared(data[:, 0], data[:, -1], sample)

    res = simulation.simulate()
    R = dac.asNpArray(res.dataArray())

    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=0.03)


def kernel_test(slabs, data):
    """
    Test the reflectivity kernel for BornAgain.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    """
    sample = get_sample(slabs)
    simulation = get_simulation(data[:, 0], sample)
    res = simulation.simulate()
    R = dac.asNpArray(res.dataArray())

    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    for nsd in tests:
        test_bornagain(nsd)
