"""
Tests for Quadtree
"""
import numpy as np
import pytest

from fmm.quadtree import Node, partition


@pytest.mark.parametrize(
    'sources, targets',
    [
        (np.random.rand(10, 2), np.random.rand(10, 2))
    ]
)
def test_init(sources, targets):
    """Test that Node object is properly initialised"""

    n = Node(sources, targets)

    assert len(n.sources) == 10
    assert len(n.targets) == 10


@pytest.mark.parametrize(
    'sources, targets',
    [
        (np.random.rand(10, 2), np.random.rand(10, 2))
    ]
)
def test_bounds(sources, targets):
    """Test that bounds make physical sense"""

    n = Node(sources, targets)

    left, right, bottom, top = n.bounds

    assert left <= right
    assert bottom <= top


@pytest.mark.parametrize(
    'sources, targets, expected',
    [
        (
            np.array(((0, 0), (0, 1), (1, 0), (1, 1))),
            np.array(((0, 0), (0, 1), (1, 0), (1, 1))),
            [(0, 0.5, 0.5, 1), (0.5, 1, 0.5, 1), (0, 0.5, 0, 0.5), (0.5, 1, 0, 0.5)]
        )
    ]
)
def test_partition(sources, targets, expected):
    
    n = Node(sources, targets)
    p = partition(n.bounds)
    
    assert p == expected
    assert type(p) == list
