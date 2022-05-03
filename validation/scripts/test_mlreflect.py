import itertools
import warnings
import pytest
import numpy as np
from mlreflect.data_generation.reflectivity import multilayer_reflectivity
from test_discovery import get_test_data

# validate the inbuilt reflectivity calculation in mlreflect


def filter_tests(td):
    test_name, slabs, data = td
    if data.shape[1] >= 4:
        # mlreflect doesn't do any resolution smearing
        return False
    # if np.any(slabs[:, 2]):
    #     # mlreflect doesn't handle imaginary part of SLD (I don't think)
    #     return False
    if slabs[0, 1] > 0:
        # mlreflect has to have fronting medium have SLD==0.
        return False
    return True


test_data = [td for td in get_test_data() if filter_tests(td)]


@pytest.mark.parametrize("td", test_data)
def test_mlreflect(td):
    """
    Run validation for mlreflect.

    Parameters
    ----------
    td: tuple
        test_name, slabs, data
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation
    test_name, slabs, data = td
    print(test_name)
    kernel_test(slabs, data)


def kernel_test(slabs, data):
    """
    Test the reflectivity kernels for mlreflect

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    """
    q = data[:, 0]
    # slabs are expected from bottom up.
    # there should be one less thickness than SLD
    # SLD array should not include fronting medium
    # len(roughness) == len(sld) == len(thickness) + 1
    thickness = slabs[-2:0:-1, 0] * 1e-10
    sld = (slabs[:0:-1, 1] + 1j * slabs[:0:-1, 2]) * 1e14
    roughness = slabs[:0:-1, 3] * 1e-10

    # furthermore the units expected by the builtin engine are expected to be
    # in m, m**-1!
    R = multilayer_reflectivity(
        q * 1e10, thickness, roughness, sld, ambient_sld=slabs[0, 1] * 1e14
    )
    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


if __name__ == "__main__":
    for td in test_data:
        test_mlreflect(td)
