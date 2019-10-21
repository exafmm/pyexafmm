import fmm.hilbert as hilbert

import numpy as np
import pytest

NPOINTS = 10000
MAXIMUM_LEVEL = 3


@pytest.fixture
def octree():
    """Fill up an octree."""
    from fmm.octree import Octree

    rand = np.random.RandomState(0)

    sources = rand.rand(NPOINTS, 3)
    targets = rand.rand(NPOINTS, 3)

    return Octree(sources, targets, MAXIMUM_LEVEL)


def test_source_leaf_assignment(octree):
    """Test source leaf assignment."""

    for index, source_node in enumerate(octree.source_leaf_nodes):
        for point_index in octree.sources_by_leafs[
            octree.source_index_ptr[index] : octree.source_index_ptr[1 + index]
        ]:
            expected = source_node
            actual = hilbert.get_key_from_point(
                octree.sources[point_index],
                octree.maximum_level,
                octree.center,
                octree.radius,
            )
            assert expected == actual


def test_target_leaf_assignment(octree):
    """Test target leaf assignment."""

    for index, target_node in enumerate(octree.target_leaf_nodes):
        for point_index in octree.targets_by_leafs[
            octree.target_index_ptr[index] : octree.target_index_ptr[1 + index]
        ]:
            expected = target_node
            actual = hilbert.get_key_from_point(
                octree.targets[point_index],
                octree.maximum_level,
                octree.center,
                octree.radius,
            )
            assert expected == actual


def test_parents(octree):
    """Test that parent computation works."""

    for point in octree.sources:
        key = hilbert.get_key_from_point(
            point, octree.maximum_level, octree.center, octree.radius
        )
        actual = octree.parent(key)
        expected = hilbert.get_key_from_point(
            point, octree.maximum_level - 1, octree.center, octree.radius
        )
        assert actual == expected


def test_neighbors(octree):
    """Test the correctness of all neighbors."""

    for index, target_node in enumerate(octree.non_empty_target_nodes):
        target_vec = hilbert.get_4d_index_from_key(target_node)
        neighbors = octree.target_neighbors[index]
        for neighbor in neighbors:
            if neighbor == -1:
                continue
            vec = hilbert.get_4d_index_from_key(neighbor)
            assert target_vec[-1] == vec[-1]
            assert np.max(np.abs(target_vec[:3] - vec[:3])) <= 1


def test_correct_keys_assigned_to_leafs(octree):
    """Test that the correct keys were assigned to the leafs."""

    max_dist = 2 * octree.radius / octree.nodes_per_side(MAXIMUM_LEVEL)

    for index, node in enumerate(octree.source_leaf_nodes):
        vec = hilbert.get_4d_index_from_key(node)
        node_center = hilbert.get_center_from_4d_index(
            vec, octree.center, octree.radius
        )
        for source_index in octree.sources_by_leafs[
            octree.source_index_ptr[index] : octree.source_index_ptr[index + 1]
        ]:
            dist = np.max(np.abs(node_center - octree.sources[source_index]))
            assert dist <= max_dist

def test_interaction_list_assignment(octree):
    """Check that the interaction list has been correctly assigned."""

    nnodes = octree.interaction_list.shape[0]
    source_nodes_set = set(octree.non_empty_source_nodes)


    for node_index, node in enumerate(octree.non_empty_target_nodes):
        level = hilbert.get_level(node)
        if level < 2: continue
        parent = hilbert.get_parent(node)
        parent_neighbors = octree.target_neighbors[octree.target_node_to_index[parent]]
        node_neighbors = octree.target_neighbors[octree.target_node_to_index[node]]
        for neighbor_index in range(27):
            parent_neighbor = parent_neighbors[neighbor_index]
            if parent_neighbors[neighbor_index] == -1:
                # The corresponding neighbor has no sources.
                assert np.all(octree.interaction_list[node_index, neighbor_index] == -1)
            else:
                # There are sources in the neighbor
                for child_index, child in enumerate(hilbert.get_children(parent_neighbor)):
                    if child in source_nodes_set and child not in set(node_neighbors):
                        assert octree.interaction_list[node_index, neighbor_index, child_index] == child
                    else:
                        assert octree.interaction_list[node_index, neighbor_index, child_index] == -1

