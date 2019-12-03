"""
Tests for Quadtree
"""
import numpy as np
import pytest

from fmm.quadtree import (
    point_to_curve, curve_to_point, find_parent, find_children, Quadtree, Points
)


ARRAY = np.ones(shape=(1000,3))
TARGETS = SOURCES = Points(points=ARRAY)


@pytest.mark.parametrize(
    "point, precision, expected",
    [
        ((0, 0), 1, 0),
    ]
)
def test_point_to_curve(point, precision, expected):
    """Test Point to Curve calculation"""
    result = point_to_curve(*point, precision)

    assert result == expected


@pytest.mark.parametrize(
    "key, precision, expected",
    [
        (0, 1, (0, 0)),
    ]
)
def test_curve_to_point(key, precision, expected):
    """Test curve to point calculator"""
    result = curve_to_point(key, precision)

    assert result == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (0, 0)
    ]
)
def test_find_parent(key, expected):

    result = find_parent(key)

    assert result == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (0, [0, 1, 2, 3])
    ]
)
def test_find_children(key, expected):

    result = find_children(key)

    assert result == expected


@pytest.mark.parametrize(
    "sources, targets, precision, expected",
    [
        (
            SOURCES, TARGETS, 2,
            {
                'n_nodes': 21,
                'n_levels': 2,
                'n_leaves': 16,
            }
        )
    ]
)
def test_quadtree(sources, targets, precision, expected):
    tree = Quadtree(sources, targets, precision,)

    assert tree.n_levels == expected['n_levels']
    assert tree.n_nodes == expected['n_nodes']
    assert tree.n_leaves == expected['n_leaves']


@pytest.mark.parametrize(
    "points, expected",
    [
        (
            ARRAY,
            {
                'shape': ARRAY.shape
            }
        )
    ]
)
def test_points(points, expected):

    p = Points(points)
    assert p.shape == expected['shape']
