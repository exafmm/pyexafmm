"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.fmm import Fmm, laplace, surface, p2p, p2m, m2m
from fmm.octree import Octree
import fmm.hilbert as hilbert


@pytest.fixture
def n_points():
    return 500

@pytest.fixture
def max_level():
    return 1

@pytest.fixture
def order():
    return 5

@pytest.fixture
def octree(n_points, max_level):
    """Fill up an octree."""

    rand = np.random.RandomState(0)

    sources = rand.rand(n_points, 3)
    targets = rand.rand(n_points, 3)

    return Octree(sources, targets, max_level)


@pytest.mark.parametrize(
    "order, radius, level, center, alpha, ncoeffs",
    [
        (2, 1, 0, np.array([0, 0, 0]), 1, 8)
    ]
)
def test_surface(order, radius, level, center, alpha, ncoeffs):
    ndim = 3
    surf = surface(order, radius, level, center, alpha)

    # Test correct number of quadrature points
    assert surf.shape == (ncoeffs, ndim)

    # Test radius
    for i in range(ndim):
        assert max(surf[:, i]) - min(surf[:, i]) == 2

    # Test center
    for i in range(ndim):
        assert np.mean(surf[:, i]) == center[i]


@pytest.mark.parametrize(
    "targets, sources, source_densities, expected",
    [
        # 1. Single source, single target, unit density
        (
            np.array([[0, 0, 0]]),
            np.array([[0, 0, 1]]),
            np.array([1]),
            np.array([1/(4*np.pi)]))
    ]
)
def test_p2p(targets, sources, source_densities, expected):
    """
    Test with single-layer Laplace kernel.
    Strategy: test whether equivalent density and direct evaluation are
    approximately equivalent in the far field.
    """

    result = p2p(laplace, targets, sources, source_densities).density

    assert result == expected


def test_p2m(octree, order, max_level):
    """Test with single-layer Laplace kernel"""

    sources = octree.sources
    source_densities = np.ones(shape=(len(sources)))

    result = p2m(
        kernel_function=laplace,
        order=order,
        center=octree.center,
        radius=octree.radius,
        maximum_level=max_level,
        leaf_sources=octree.sources
        )

    distant_point = np.array([[100, 0, 0], [0, 111, 0]])

    direct = p2p(laplace, distant_point, sources, source_densities)
    equivalent = p2p(laplace, distant_point, result.surface, result.density)

    for i in range(len(equivalent.density)):
        a = equivalent.density[i][0]; b = direct.density[i][0]
        assert np.isclose(a, b, rtol=1e-1)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(equivalent.surface, direct.surface)


def test_m2m(octree, order, n_points):
    """
    Test with single-layer Laplace kernel.
    Strategy: Test whether child equivalent density and parent equivalent
    density are approximately equivalent to each other in the far field.
    """

    parent_key = 0
    child_key = 1

    x0 = octree.center; r0 = octree.radius
    parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = hilbert.get_level(parent_key)
    child_level = hilbert.get_level(child_key)

    child_equivalent_density = np.ones(shape=(n_points))
    child_equivalent_surface = surface(
        order=order,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=1.05
    )

    result = m2m(
        kernel_function=laplace,
        order=order,
        parent_center=parent_center,
        child_center=child_center,
        radius=octree.radius,
        child_level=child_level,
        parent_level=parent_level,
        child_equivalent_density=child_equivalent_density
    )

    parent_equivalent_surface, parent_equivalent_density = result.surface, result.density

    distant_point = np.array([[10, 0, 0]])

    child_result = p2p(
        laplace, distant_point, child_equivalent_surface, child_equivalent_density
    )

    parent_result = p2p(
        laplace, distant_point, parent_equivalent_surface, parent_equivalent_density
    )

    assert np.isclose(child_result.density[0], parent_result.density[0], rtol=1e-1)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(parent_result.surface, child_result.surface)