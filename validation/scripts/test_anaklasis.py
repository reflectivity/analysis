import pytest

import numpy as np
from test_discovery import get_test_data
from anaklasis import ref


tests = list(get_test_data())
ids = [f"{t[0]}" for t in tests]


@pytest.mark.parametrize("nsd", tests, ids=ids)
def test_anaklasis(nsd):
    """
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
        resolution_test(slabs, data)
        pass
    elif data.shape[1] < 4:
        # no resolution data, just test kernel
        kernel_test(slabs, data)


def kernel_test(slabs, data):
    # test unsmeared reflectivity calculation
    q = data[:, 0]
    R = data[:, 1]

    project = "none"
    patches = [1.0]
    LayerMatrix = []
    for i, layer in enumerate(slabs):
        # layer = thickness, re, im, roughness

        # for anaklasis the roughnesses of a layer is between i and i + 1
        # for refnx the roughness of a layer is between i and i - 1.
        if i == len(slabs) - 1:
            rough = 0
        else:
            rough = slabs[i + 1, -1]

        LayerMatrix.append(
            [
                layer[1] * 1e-6,
                layer[2] * 1e-6,
                layer[0],
                rough,
                0.0,
            ]
        )

    # system = [LayerMatrix]
    # result = ref.calculate(
    #     "none",
    #     [0.0], patches, system, [], [0], [1.0], [np.max(q)]
    # )

    dq = np.zeros_like(q)
    # This is much harder than it should be:
    #   - there is no public API to calculate reflectivity with anaklasis
    #   - not clear what LayerMatrix layout is supposed to be
    #   - Reflectivity function is not properly documented
    tmp = ref.Reflectivity(q, dq, [LayerMatrix], [0], 0, 1, patches, 1)
    np.testing.assert_allclose(tmp[:, 1], R, rtol=8e-5)


def resolution_test(slabs, data):
    pass


if __name__ == "__main__":
    for nsd in tests:
        test_anaklasis(nsd)
