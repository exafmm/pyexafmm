"""
Tests for Hilbert key helper functions.
"""
import numpy as np
import pytest

import fmm.hilbert as hilbert


@pytest.mark.parametrize(
    "level, offset", [
        (0, 0),
        (1, 1),
        (2, 9),
        (3, 73)
    ]
)
def test_level_offset(level, offset):
    assert hilbert.level_offset(level) == offset

@pytest.mark.parametrize(
    "key, level", [
        (0, 0),
        (8, 1),
        (64, 2)
    ]
)
def test_get_level(key, level):
    assert hilbert.get_level(key) == level


@pytest.mark.parametrize(
    "key, index", [
        (0, np.array([0, 0, 0, 0])),
        (8, np.array([1, 1, 1, 1]))
    ]
)
def test_get_4d_index_from_key(key, index):
    assert np.array_equal(
        hilbert.get_4d_index_from_key(key),
        index
    )

@pytest.mark.parametrize(
    "index, center",
    [
        # Root node, expect center to be the same as set
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0]))
    ]
)
def test_get_center_from_4d_index(index, center):

    # Center and side half-length of root node
    x0 = np.array([0, 0, 0])
    r0 = 1
    assert np.array_equal(
        hilbert.get_center_from_4d_index(index, center, r0),
        x0
    )
