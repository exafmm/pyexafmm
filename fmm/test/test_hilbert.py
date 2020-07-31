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
        (-1, -1),
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
        (np.array([1, 3, 1, 2]), 2)
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





# @pytest.mark.parametrize(
#     "key",
#     [
#         9
#     ]
# )
# def test_compute_neighbors(key):

#     neighbors = hilbert.compute_neighbors(key)
#     interaction_list = hilbert.compute_interaction_list(key)

#     # assert len(neighbors) == 26

#     neighbor_coords = np.array([
#         hilbert.get_4d_index_from_key(k)[:3]
#         for k in neighbors
#     ])

#     int_list_coords = np.array([
#         hilbert.get_4d_index_from_key(k)[:3]
#         for k in interaction_list
#     ])

#     node_coords = hilbert.get_4d_index_from_key(key)[:3]

#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(neighbor_coords[:, 0],neighbor_coords[:, 1],neighbor_coords[:, 2])
#     ax.scatter(node_coords[0],node_coords[1],node_coords[2], color='red')
#     ax.scatter(int_list_coords[:, 0],int_list_coords[:, 1],int_list_coords[:, 2], color='green')

#     plt.show()

#     assert False


# @pytest.mark.parametrize(
#     "key",
#     [
#         9,
#     ]
# )
# def test_compute_interaction_list(key):

#     interaction_list = hilbert.compute_interaction_list(key)

#     print(interaction_list, len(interaction_list))

#     assert False