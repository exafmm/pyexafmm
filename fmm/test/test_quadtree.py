"""
Tests for Quadtree
"""
import numpy as np
import pytest

from fmm.quadtree import Quadtree


@pytest.mark.parametrize(
    'sources, targets, max_depth',
    [
        (np.random.rand(10, 2), np.random.rand(10, 2), 1)
    ]
)
def test_init(sources, targets, max_depth):
    """Test that Quadtree object is properly initialised"""

    q = Quadtree(sources, targets, max_depth)

    assert q.max_level == 1
    assert len(q.sources) == 10
    assert len(q.targets) == 10


@pytest.mark.parametrize(
    'sources, targets, max_depth',
    [
        (np.random.rand(10, 2), np.random.rand(10, 2), 1)
    ]
)
def test_bounds(sources, targets, max_depth):
    """Test that bounds make physical sense"""

    q = Quadtree(sources, targets, max_depth)

    left, right, bottom, top = q.bounds

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
    
    q = Quadtree(sources, targets, 1)
    print(q.bounds)
    partition = q.partition(q.bounds)
    
    assert partition == expected
    assert type(partition) == list
