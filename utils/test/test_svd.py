"""
Tests for SVD compression
"""
import numpy as np
import pytest

import utils.svd as svd


@pytest.mark.parametrize(
    "matrix",
    [
        (
            np.array([
                [1, 10, 21],
                [2, 12, 32],
                [22, 413, 2],
            ])
        )
    ]
)
def test_compress(matrix):
    uk, sk, vhk = svd.compress(matrix)
    reconstructed = np.matmul(uk, np.matmul(np.diag(sk), vhk))

    assert np.allclose(reconstructed, matrix)
