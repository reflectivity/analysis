import os.path
import glob

import numpy as np


PTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "unpolarised"
)


def idfn(val):
    return val


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
    shape `(M, N)`, where M is the number of datapoints, and N in [2, 3, 4].
    The first column contains Q points (reciprocal Angstrom), the second column
    the reflectivity data, the optional third and fourth columns are dR and dQ.
    If present dR and dQ are 1 standard deviation uncertainties on reflectivity
    and Q-resolution (gaussian resolution kernel).
    """
    # find all the unpolarised tests in analysis/validation/test/unpolarised
    tests = glob.glob(os.path.join(PTH, "*.txt"))

    for test in tests:
        # layers/data tuples
        layers, data = get_data(test)

        slabs = np.loadtxt(layers)
        assert slabs.shape[1] == 4

        data = np.loadtxt(data)
        assert data.shape[1] in [2, 3, 4]

        yield os.path.basename(test), slabs, data


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
