"""
Tests for Octree data structure and its helper methods
"""
import os
import pathlib

import numpy as np
import pytest

import fmm.hilbert as hilbert
import fmm.octree as octree
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent.parent

CONFIG_FILEPATH = ROOT / 'test_config.json'
CONFIG = data.load_json(CONFIG_FILEPATH)

SOURCE_FILENAME = 'sources'
SOURCE_DENSITIES_FILENAME = 'source_densities'
TARGET_FILENAME = 'targets'
DATA_DIRPATH = ROOT / CONFIG['data_dirname']


@pytest.fixture
def tree():
    """Fill up an octree."""
    sources = data.load_hdf5_to_array(SOURCE_FILENAME, SOURCE_FILENAME, DATA_DIRPATH)
    targets = data.load_hdf5_to_array(TARGET_FILENAME, TARGET_FILENAME, DATA_DIRPATH)
    source_densities = data.load_hdf5_to_array(
        SOURCE_DENSITIES_FILENAME, SOURCE_DENSITIES_FILENAME, DATA_DIRPATH
        )

    return octree.Octree(sources, targets, CONFIG['octree_max_level'], source_densities)


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, 1),
        (1, 2)
    ]
)
def test_nodes_per_side(level, expected):
    assert octree.nodes_per_side(level) == expected



@pytest.mark.parametrize(
    "level, expected",
    [
        (0, 8**0),
        (1, 8**1),
        (5, 8**5)
    ]
)
def test_nodes_per_level(level, expected):
    assert octree.nodes_per_level(level) == expected


def test_source_leaf_assignment(tree):
    """Test source leaf assignment."""

    for index, source_node in enumerate(tree.source_leaf_nodes):
        for point_index in tree.sources_by_leafs[
                tree.source_index_ptr[index] : tree.source_index_ptr[1 + index]
        ]:
            expected = source_node
            actual = hilbert.get_key_from_point(
                tree.sources[point_index],
                tree.maximum_level,
                tree.center,
                tree.radius,
            )
            assert expected == actual


def test_target_leaf_assignment(tree):
    """Test target leaf assignment."""

    for index, target_node in enumerate(tree.target_leaf_nodes):
        for point_index in tree.targets_by_leafs[
                tree.target_index_ptr[index] : tree.target_index_ptr[1 + index]
        ]:
            expected = target_node
            actual = hilbert.get_key_from_point(
                tree.targets[point_index],
                tree.maximum_level,
                tree.center,
                tree.radius,
            )
            assert expected == actual


def test_parents(tree):
    """Test that parent computation works."""

    for point in tree.sources:
        key = hilbert.get_key_from_point(
            point, tree.maximum_level, tree.center, tree.radius
        )
        actual = hilbert.get_parent(key)
        expected = hilbert.get_key_from_point(
            point, tree.maximum_level - 1, tree.center, tree.radius
        )
        assert actual == expected


def test_neighbors(tree):
    """Test the correctness of all neighbors."""

    for index, target_node in enumerate(tree.non_empty_target_nodes):
        target_vec = hilbert.get_4d_index_from_key(target_node)
        neighbors = tree.target_neighbors[index]
        for neighbor in neighbors:
            if neighbor == -1:
                continue
            vec = hilbert.get_4d_index_from_key(neighbor)
            assert target_vec[-1] == vec[-1]
            assert np.max(np.abs(target_vec[:3] - vec[:3])) <= 1


def test_correct_keys_assigned_to_leafs(tree):
    """Test that the correct keys were assigned to the leafs."""

    max_dist = 2 * tree.radius / octree.nodes_per_side(CONFIG['octree_max_level'])

    for index, node in enumerate(tree.source_leaf_nodes):
        vec = hilbert.get_4d_index_from_key(node)
        node_center = hilbert.get_center_from_4d_index(
            vec, tree.center, tree.radius
        )
        for source_index in tree.sources_by_leafs[
                tree.source_index_ptr[index] : tree.source_index_ptr[index + 1]
        ]:
            dist = np.max(np.abs(node_center - tree.sources[source_index]))
            assert dist <= max_dist


def test_interaction_list_assignment(tree):
    """Check that the interaction list has been correctly assigned."""

    source_nodes_set = set(tree.non_empty_source_nodes)


    for node_index, node in enumerate(tree.non_empty_target_nodes):
        level = hilbert.get_level(node)
        if level < 2:
            continue
        parent = hilbert.get_parent(node)
        parent_neighbors = tree.target_neighbors[tree.target_node_to_index[parent]]
        node_neighbors = tree.target_neighbors[tree.target_node_to_index[node]]
        for neighbor_index in range(27):
            parent_neighbor = parent_neighbors[neighbor_index]
            if parent_neighbors[neighbor_index] == -1:
                # The corresponding neighbor has no sources.
                assert np.all(tree.interaction_list[node_index, neighbor_index] == -1)
            else:
                # There are sources in the neighbor
                for child_index, child in enumerate(hilbert.get_children(parent_neighbor)):
                    if child in source_nodes_set and child not in set(node_neighbors):
                        assert tree.interaction_list[node_index, neighbor_index, child_index] == child  # pylint: disable=C0301
                    else:
                        assert tree.interaction_list[node_index, neighbor_index, child_index] == -1  # pylint: disable=C0301




def test_enumerate_non_empty_nodes(maximum_level, leaves):
    pass


@pytest.mark.parametrize(
    "keys, expected",
    [
        (
            np.arange(10),
            {
                0: np.array([0]),
                1: np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                2: np.array([9])
            }
        )
    ]
)
def test_sort_keys_by_level(keys, expected):
    result = octree.sort_keys_by_level(keys)

    for key, value in result.items():
        assert np.array_equal(value, expected[key])


@pytest.mark.parametrize(
    "sources, targets, max_bound, min_bound",
    [
        (
            np.array([[0, 4, 0]]),
            np.array([[1, 1, 1]]),
            np.array([1, 4, 1]),
            np.array([0, 1, 0]),
        )
    ]
)
def test_compute_bounds(sources, targets, max_bound, min_bound):
    result = octree.compute_bounds(sources, targets)

    assert np.array_equal(result[0], max_bound)
    assert np.array_equal(result[1], min_bound)


@pytest.mark.parametrize(
    "max_bound, min_bound, expected",
    [
        (np.array([1, 1, 1,]), np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5]))
    ]
)
def test_compute_center(max_bound, min_bound, expected):
    assert np.array_equal(octree.compute_center(max_bound, min_bound), expected)


@pytest.mark.parametrize(
    "center, max_bound, min_bound, expected",
    [
        (np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 0, 0]), 1*(1+1e-5))
    ]
)
def test_compute_radius(center, max_bound, min_bound, expected):
    assert octree.compute_radius(center, max_bound, min_bound) == expected
