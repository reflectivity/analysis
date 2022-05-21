import os.path
import glob

import numpy as np


PTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "unpolarised"
)
POL_PTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "polarised"
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
        layers_file, data_file = get_data(test)

        slabs = np.loadtxt(os.path.join(PTH, layers_file))
        assert slabs.shape[1] == 4

        data = np.loadtxt(os.path.join(PTH, data_file))
        assert data.shape[1] in [2, 3, 4]

        yield os.path.basename(test), slabs, data


def get_data(file):
    # for each of the test files extract the parameters,
    # e.g. the names of the data_file and layers_file

    with open(file, "rt") as f:
        # ignore comment lines starting with #
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                yield line


def get_polarised_test_data():
    """
    A generator yielding (slabs, data) tuples.

    `slabs` are np.ndarrays that specify the layer structure of the test.
    ``slabs.shape = (N + 2, 6)``, where N is the number of layers.

    The layer specification file has the layout:

    # { "AGUIDE": angle_between_Q_and_H }
    ignored     SLDn_fronting ignored      ignored   ignored ignored
    thickness_1 SLDn_1        iSLD_1       thetaM_1  SLDm_1  rough_fronting1
    thickness_2 SLDn_2        iSLD_2       thetaM_2  SLDm_2  rough_12
    ignored     SLDn_backing  iSLD_backing ignored   ignored rough_backing2

    `data` contains the test reflectivity data. It's an np.ndarray with
    shape `(M, 5)`, where M is the number of datapoints (with 5 data columns.)
    The first column contains Q points (reciprocal Angstrom), and the remaining
    columns correspond to R--, R-+, R+- and R++, where the first +/- corresponds
    to the polarisation of the incoming beam, and the second to the scattered
    beam.
    """
    # find all the polarised tests in analysis/validation/test/polarised
    tests = glob.glob(os.path.join(POL_PTH, "*.txt"))

    for test in tests:
        # layers/data tuples
        layers_file, data_file, AGUIDE, H = get_data(test)

        slabs = np.loadtxt(os.path.join(POL_PTH, layers_file))
        assert slabs.shape[1] == 6

        data = np.loadtxt(os.path.join(POL_PTH, data_file))
        assert data.shape[1] == 5

        AGUIDE = float(AGUIDE)
        H = float(H)

        yield os.path.basename(test), slabs, data, AGUIDE, H
