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

    l, r, b, t = q.bounds

    assert l <= r
    assert b <= t
