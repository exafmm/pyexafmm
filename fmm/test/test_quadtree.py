"""
Tests for Quadtree
"""
import numpy as np
import pytest

from fmm.quadtree import Node, partition

SOURCES = np.array((0, 0)).reshape(1, 2)
TARGETS = np.array((1, 1)).reshape(1, 2)
NODE_BOUNDS = (-0.1, 1.1, -0.1, 1.1)
SIMPLE_BOUNDS = (0, 1, 0, 1)
SIMPLE_PARTITION = (
    (0, 0.5, 0.5, 1),
    (0.5, 1, 0.5, 1),
    (0, 0.5, 0, 0.5),
    (0.5, 1, 0, 0.5)
)

@pytest.mark.parametrize(
    "sources, targets, bounds",
    [
        (SOURCES, TARGETS, NODE_BOUNDS)
    ]
)
def test_node_init(sources, targets, bounds):
    """Test instantiation of Node object"""
    n = Node(sources, targets, bounds)

    assert n.parent is None
    assert np.array_equal(n.sources, sources)
    assert np.array_equal(n.targets, targets)
    assert n.bounds == bounds

@pytest.mark.parametrize(
    "sources, targets, bounds",
    [
        (SOURCES, TARGETS, NODE_BOUNDS)
    ]
)
def test_node_children(sources, targets, bounds):
    """Test the finding of node children"""
    n = Node(sources, targets, bounds)

    # Test that partition takes place
    assert len(n.children) == 4

    # Test that parent is assigned
    assert n.children[0].parent is n
    
    # Check that sources and targets are partitioned amongst children
    north_west, north_east, south_west, south_east = n.children

    # From our choice of targets and sources expect following two nodes to
    # containt targets/sources
    assert np.array_equal(sources, south_west.sources)
    assert np.array_equal(targets, north_east.targets)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (SIMPLE_BOUNDS, SIMPLE_PARTITION)
    ]
)
def test_partition(bounds, expected):
    """Test partition function"""
    p = partition(bounds)

    assert p == expected
