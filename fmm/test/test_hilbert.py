"""
Tests for Hilbert key helper functions.
"""
import numpy as np
import pytest

import fmm.hilbert as hilbert


@pytest.mark.parametrize(
    "level, offset",
    [
        (0, 0),
        (1, 1),
        (2, 9),
        (3, 73)
    ]
)
def test_get_level_offset(level, offset):
    assert hilbert.get_level_offset(level) == offset


@pytest.mark.parametrize(
    "key, level",
    [
        (0, 0),
        (8, 1),
        (64, 2)
    ]
)
def test_get_level(key, level):
    assert hilbert.get_level(key) == level


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, 0),
        (1, 1),
        (2, 9),
    ]
)
def test_get_level_offset(level, expected):
    assert hilbert.get_level_offset(level) == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (9, 0),
        (72, 63)
    ]
)
def test_remove_level_offset(key, expected):
    assert hilbert.remove_level_offset(key) == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (0, 0),
        (132, 3),
    ]
)
def test_get_octant(key, expected):
    assert hilbert.get_octant(key) == expected


@pytest.mark.parametrize(
    "index, expected",
    [
        (np.array([1, 1, 1, 1]), 8),
        (np.array([0, 0, 0, 0]), 0)
    ]
)
def test_get_key_from_4d_index(index, expected):
    assert hilbert.get_key_from_4d_index(index) == expected


@pytest.mark.parametrize(
    "key, index",
    [
        (0, np.array([0, 0, 0, 0])),
        (8, np.array([1, 1, 1, 1]))
    ]
)
def test_get_4d_index_from_key(key, index):
    result = hilbert.get_4d_index_from_key(key)
    assert np.array_equal(result, index)

    assert result.shape == (4,)


@pytest.mark.parametrize(
    "point, level, expected",
    [
        (np.array([1, 1, 0]), 1, np.array([1, 1, 0, 1]))
    ]
)
def test_get_4d_index_from_point(point, level, expected):
    x0 = np.array([1, 1, 1])
    r0 = 1

    result = hilbert.get_4d_index_from_point(point, level, x0, r0)
    assert np.array_equal(result,expected)

    assert result.shape == (4,)



@pytest.mark.parametrize(
    "point, level, x0, r0, expected",
    [
        (np.array([0, 0, 0]), 1, np.array([1, 1, 1]), 1, 1),
        (np.array([0.9, 0.9, 0.9]), 1, np.array([1, 1, 1]), 1, 1),
    ]
)
def test_get_key_from_point(point, level, x0, r0, expected):
    result = hilbert.get_key_from_point(point, level, x0, r0)
    assert result == expected


@pytest.mark.parametrize(
    "points, level, x0, r0, expected",
    [
      (np.array([[0, 0, 0]]), 1, np.array([1, 1, 1]), 1, np.array([1]))
    ]
)
def test_get_keys_from_points(points, level, x0, r0, expected):
    result = hilbert.get_keys_from_points(points,level, x0, r0)
    assert np.array_equal(result, expected)

    assert result.shape == (1, )



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

    result = hilbert.get_center_from_4d_index(index, center, r0)
    assert np.array_equal(result, x0)

    assert result.shape == (3,)


@pytest.mark.parametrize(
    "key, expected",
    [
        (1, 0),
        (9, 1),
        (73, 9)
    ]
)
def test_get_parent(key, expected):
    assert hilbert.get_parent(key) == expected


@pytest.mark.parametrize(
    "parent, expected",
    [
        (0, 1+np.arange(8))
    ]
)
def test_get_children(parent, expected):
    assert np.array_equal(hilbert.get_children(parent), expected)


@pytest.mark.parametrize(
    "level, expected",
    [
        (1, 9),
        (2, 73)
    ]
)
def test_get_number_of_all_nodes(level, expected):
    assert hilbert.get_number_of_all_nodes(level) == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (1, np.array([2, 3, 4, 5, 6, 7, 8]))
    ]
)
def test_get_neighbors(key, expected):

    result = hilbert.get_neighbors(key)

    assert np.array_equal(result, expected)

    assert len(result) <= 26


TEST_INTERACTION_LIST = np.array(
    [
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72
        ]
    )

@pytest.mark.parametrize(
    "key, expected",
    [
        (1, np.array([])),
        (9, TEST_INTERACTION_LIST)
    ]
)
def test_get_interaction_list(key, expected):

    result = hilbert.get_interaction_list(key)
    assert np.array_equal(result, expected)

    assert len(result) <= 189
