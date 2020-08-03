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



def test_compute_neighbors():
    pass



def test_compute_interaction_list():
    pass
