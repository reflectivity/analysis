import os.path
import glob

import numpy as np


PTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "unpolarised"
)


def get_test_data():
    """
    A generator yielding (slabs, data) tuples.

    `slabs` are np.ndarrays that specify the layer structure of the test.
    ``slabs.shape = (N + 2, 4)``, where N is the number of layers.

    The layer specification file has the layout:

    ignored     SLD_fronting ignored      ignored
    thickness_1 SLD_1        iSLD_1       rough_fronting1
    thickness_2 SLD_2        iSLD_2       rough_12
    ignored     SLD_backing  iSLD_backing rough_backing2

    `data` contains the test reflectivity data. It's an np.ndarray with
    shape `(M, 2)`, where M is the number of datapoints. The first column
    contains Q points (reciprocal Angstrom), the second column the
    reflectivity data. In future extra columns may be added to represent
    dR and dQ.
    """
    # find all the unpolarised tests in analysis/validation/test/unpolarised
    tests = glob.glob(os.path.join(PTH, "*.txt"))

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

    with open(file, "r") as f:
        # ignore comment lines starting with # or space
        while True:
            line = f.readline()
            if line.lstrip(" \t").startswith("#"):
                continue
            elif line == "\n":
                continue
            else:
                layers = line.rstrip("\n")
                data = f.readline().rstrip("\n")

                layers = os.path.join(PTH, layers)
                data = os.path.join(PTH, data)
                return (layers, data)
