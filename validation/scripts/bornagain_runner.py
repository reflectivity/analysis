import os.path
import glob

import numpy as np
from numpy.testing import assert_allclose

import bornagain as ba
from bornagain import angstrom


PTH = os.path.join('..', 'test', 'unpolarised')


def find_tests():
    # find all the unpolarised tests in analysis/validation/test/unpolarised
    tests = glob.glob(os.path.join(PTH, '*.txt'))
    data = [get_data(test) for test in tests]
    return data


def get_data(file):
    # for each of the unpolarised test files figure out where the data and
    # layer parameters are

    with open(file, 'r') as f:
        # ignore comment lines starting with # or space
        while True:
            line = f.readline()
            if line.lstrip(' \t').startswith('#'):
                continue
            elif line == '\n':
                continue
            else:
                layers = line.rstrip('\n')
                data = f.readline().rstrip('\n')
                return (layers, data)


def get_sample(slabs):
    """
    Defines sample and returns it. Note that SLD-based materials are used.
    """
    # creating materials
    multi_layer = ba.MultiLayer()

    ambient = ba.MaterialBySLD('ma', slabs[0, 1] * 1e-6, 0)
    layer = ba.Layer(ambient)
    multi_layer.addLayer(layer)

    for slab in slabs[1:-1]:
        material = ba.MaterialBySLD('stuff', slab[1] * 1e-6, slab[2] * 1e-6)
        layer = ba.Layer(material, slab[0] * angstrom)

        roughness = ba.LayerRoughness()
        roughness.setSigma(slab[3] * angstrom)

        multi_layer.addLayerWithTopRoughness(layer, roughness)

    substrate = ba.MaterialBySLD('msub', slabs[-1, 1] * 1e-6, 0)
    layer = ba.Layer(substrate)
    roughness = ba.LayerRoughness()
    roughness.setSigma(slabs[-1, 3] * angstrom)
    multi_layer.addLayerWithTopRoughness(layer, roughness)

    return multi_layer


def get_simulation(qzs):
    """
    Defines and returns specular simulation
    with a qz-defined beam
    """
    # bornagain requires Qz in nm
    scan = ba.QSpecScan(qzs * 10.)
    simulation = ba.SpecularSimulation()
    simulation.setScan(scan)
    return simulation


def run_tests():
    tests = find_tests()
    for test in tests:
        slabs = np.loadtxt(os.path.join(PTH, test[0]))
        assert slabs.shape[1] == 4

        data = np.loadtxt(os.path.join(PTH, test[1]))
        assert data.shape[1] == 2

        simulation = get_simulation(data[:, 0])
        sample = get_sample(slabs)
        simulation.setSample(sample)
        simulation.runSimulation()
        R = simulation.result().array()

        assert R.shape == data[:, 1].shape

        assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == '__main__':
    run_tests()
