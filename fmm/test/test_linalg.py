"""
Test linear algebra routines
"""
import numpy as np
import pytest

from fmm.linalg import pinv, pinv2, EPS


@pytest.mark.parametrize(
    "dtype",
    [
        (np.float32),
        (np.float64)
    ]
)
def test_pinv(dtype):
    m = 10
    n = 10
    x = np.random.rand(m*n).reshape(m, n).astype(dtype)

    result = pinv(x)

    id = x @ result
    expected = np.diag(np.ones(m))

    err = np.mean(id - expected)

    # Test shape
    assert id.shape == expected.shape

    # Test that error is smaller than a constant factor of machine epsilon
    assert (err - EPS[dtype]) < 4*EPS[dtype]

    # Test casting
    assert isinstance(result[0, 0], dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        (np.float32),
        (np.float64)
    ]
)
def test_pinv2(dtype):
    m = 10
    n = 10
    x = np.random.rand(m*n).reshape(m, n).astype(dtype)

    a, b = pinv2(x)

    result = a @ b

    id = x @ result
    expected = np.diag(np.ones(m))

    err = np.mean(id - expected)

    # Test shape
    assert id.shape == expected.shape

    # Test that error is smaller than a constant factor of machine epsilon
    assert (err - EPS[dtype]) < 4*EPS[dtype]

    # Test casting
    assert isinstance(result[0, 0], dtype)
