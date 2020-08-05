import numpy as np
import pytest

import fmm.kernel as kernel


def test_identity_instantiation():
    obj = kernel.Identity()
    assert isinstance(obj, kernel.Kernel)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 2]), np.array([1, 2]), 5)
    ]
)
def test_identity_kernel_function(x, y, expected):
    obj = kernel.Identity()
    result = obj(x, y)
    assert result == expected


def test_laplace_instantiation():
    obj = kernel.Laplace()
    assert isinstance(obj, kernel.Kernel)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 0]), np.array([0, 0]), 1/(4*np.pi)),
        (np.array([1e-13, 0]), np.array([0, 0]), 0),
    ]
)
def test_identity_kernel_function(x, y, expected):
    obj = kernel.Laplace()
    result = obj(x, y)
    assert result == expected
