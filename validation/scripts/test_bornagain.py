import numpy as np
from test_discovery import get_test_data

import bornagain as ba
from bornagain import angstrom


def get_sample(slabs):
    """
    Defines sample and returns it. Note that SLD-based materials are used.
    """
    # creating materials
    multi_layer = ba.MultiLayer()

    ambient = ba.MaterialBySLD("ma", slabs[0, 1] * 1e-6, 0)
    layer = ba.Layer(ambient)
    multi_layer.addLayer(layer)

    for slab in slabs[1:-1]:
        material = ba.MaterialBySLD("stuff", slab[1] * 1e-6, slab[2] * 1e-6)
        layer = ba.Layer(material, slab[0] * angstrom)

        roughness = ba.LayerRoughness()
        roughness.setSigma(slab[3] * angstrom)

        multi_layer.addLayerWithTopRoughness(layer, roughness)

    substrate = ba.MaterialBySLD("msub", slabs[-1, 1] * 1e-6, 0)
    layer = ba.Layer(substrate)
    roughness = ba.LayerRoughness()
    roughness.setSigma(slabs[-1, 3] * angstrom)
    multi_layer.addLayerWithTopRoughness(layer, roughness)

    multi_layer.setRoughnessModel(ba.RoughnessModel.NEVOT_CROCE)

    return multi_layer


def get_simulation(qzs):
    """
    Defines and returns specular simulation
    with a qz-defined beam
    """
    # bornagain requires Qz in nm
    scan = ba.QSpecScan(qzs * 10.0)
    simulation = ba.SpecularSimulation()
    simulation.setScan(scan)
    return simulation


def test_bornagain():
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
    simulation = get_simulation(data[:, 0])
    sample = get_sample(slabs)
    simulation.setSample(sample)
    simulation.runSimulation()
    R = simulation.result().array()

    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    test_bornagain()
