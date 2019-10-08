from fmm import hilbert

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
    center = np.array([0.5, 0.5, 0.5])
    radius = 0.5

    return Octree(sources, targets, MAXIMUM_LEVEL, center, radius)


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
                point,
                octree.maximum_level,
                octree.center,
                octree.radius,
                )
        actual = octree.parent(key)
        expected = hilbert.get_key_from_point(
                point,
                octree.maximum_level - 1,
                octree.center,
                octree.radius,
                )
        assert actual == expected

def test_neighbors(octree):
    """Test the correctness of all neighbors."""

    for index, target_node in enumerate(octree.non_empty_target_nodes):
        target_vec = hilbert.get_4d_index_from_key(target_node)
        neighbors = octree.target_neighbors[index]
        for neighbor in neighbors:
            if neighbor == -1: continue
            vec = hilbert.get_4d_index_from_key(neighbor)
            assert target_vec[-1] == vec[-1]
            assert np.max(np.abs(target_vec[:3] - vec[:3])) <= 1






