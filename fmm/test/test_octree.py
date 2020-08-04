"""
Tests for Octree data structure
"""
import os
import pathlib

import numpy as np
import pytest

import fmm.hilbert as hilbert
import fmm.octree as octree
import utils.data as data

NPOINTS = 10000

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

    max_dist = 2 * tree.radius / tree.nodes_per_side(CONFIG['octree_max_level'])

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


# @pytest.mark.parametrize(
#     "sources, targets, expected",
#     [
#         (
#             np.array([[0, 0, 0]]),
#             np.array([[1, 1, 1]]),
#             (np.array([0.5, 0.5, 0.5]), 0.5*1.00001)
#         )
#     ]
# )
# def test_compute_bounds(sources, targets, expected):
#     center, radius = octree.compute_bounds(sources, targets)
#     assert np.array_equal(center, expected[0])
#     assert radius == expected[1]