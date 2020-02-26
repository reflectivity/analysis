import os.path
import glob

import numpy as np
from numpy.testing import assert_allclose

from refnx.reflect import reflectivity

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


def run_tests():
    tests = find_tests()
    for test in tests:
        slabs = np.loadtxt(os.path.join(PTH, test[0]))
        assert slabs.shape[1] == 4

        data = np.loadtxt(os.path.join(PTH, test[1]))
        assert data.shape[1] == 2

        R = reflectivity(data[:, 0], slabs, dq=0)
        assert R.shape == data[:, 1].shape

        assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == '__main__':
    run_tests()
