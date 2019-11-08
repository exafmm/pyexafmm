"""
Tests for Quadtree
"""
import numpy as np
import pytest

from fmm.quadtree import Node, partition, find_bounds


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


@pytest.mark.parametrize(
    'sources, targets',
    [
        (
            np.array((0,1)), np.array((0,1))
        )
    ]
)
def test_find_bounds(sources, targets):
    sources = sources.reshape(1, 2)
    targets = targets.reshape(1, 2)
    print(find_bounds(sources, targets))
    assert True


@pytest.mark.parametrize(
    'sources, targets',
    [
        (
            np.array(((0, 0), (0, 1), (1, 0), (1, 1))),
            np.array(((0, 0), (0, 1), (1, 0), (1, 1))),
        )
    ]
)
def test_children(sources, targets):
    n = Node(sources, targets)
    # [print(c.bounds) for c in n.children]
    # [print(c.parent) for c in n.children]
    [print(c.targets) for c in n.children]

    assert False