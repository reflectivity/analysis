import os.path
import glob

import numpy as np
from numpy.testing import assert_allclose

from refnx.reflect import reflectivity

PTH = os.path.join('..', 'test', 'unpolarised')


def find_tests():
    tests = glob