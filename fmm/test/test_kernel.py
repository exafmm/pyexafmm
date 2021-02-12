import numpy as np
import pytest

import fmm.kernel as kernel


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 0]), np.array([0, 0]), 1/(4*np.pi)),
        (np.array([1e-13, 0]), np.array([0, 0]), 0),
    ]
)
def test_identity_kernel_function(x, y, expected):
    result = kernel.laplace(x, y)
    assert result == expected
