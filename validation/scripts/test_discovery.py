import os.path
import glob

import numpy as np


PTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test', 'unpolarised')


def get_test_data():
    # find all the unpolarised tests in analysis/validation/test/unpolarised
    tests = glob.glob(os.path.join(PTH, '*.txt'))

    # layers/data tuples
    test_files = [get_data(test) for test in tests]

    for layers, data in test_files:
        slabs = np.loadtxt(layers)
        assert slabs.shape[1] == 4

        data = np.loadtxt(data)
        assert data.shape[1] == 2

        yield slabs, data


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

                layers = os.path.join(PTH, layers)
                data = os.path.join(PTH, data)
                return (layers, data)
