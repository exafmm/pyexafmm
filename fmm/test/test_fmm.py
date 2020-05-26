"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.octree import Octree
from fmm.fmm import Fmm, laplace, surface, p2p, p2m


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
    """Test with single-layer Laplace kernel"""

    result = p2p(laplace, targets, sources, source_densities).density

    assert result == expected


def test_p2m(octree, order, max_level):
    """Test with single-layer Laplace kernel"""

    # Attach unit densities to each source in Octree
    sources = octree.sources
    source_densities = np.ones(shape=(len(sources)))

    # Evaluate p2m with a single-level Octree
    result = p2m(
        kernel_function=laplace,
        order=order,
        center=octree.center,
        radius=octree.radius,
        maximum_level=max_level,
        leaf_sources=octree.sources
        )

    # Evaluate directly using the leaf sources, and compare to evaluating
    # directly using the equivalent surface at a distant point in the far field.

    distant_point = np.array([[100, 0, 0], [0, 111, 0]])

    direct = p2p(laplace, distant_point, sources, source_densities)
    equivalent = p2p(laplace, distant_point, result.surface, result.density)

    # Test that potentials are evaluated to approximately the same value.
    for i in range(len(equivalent.density)):
        a = equivalent.density[i][0]; b = direct.density[i][0]
        assert np.isclose(a, b, rtol=1e-1)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(equivalent.surface, direct.surface)


def test_m2m():
    assert True

# def test_upward_and_downward_pass(order, n_points, octree):
#     """Test the upward pass."""

#     fmm = Fmm(octree, order, laplace)
#     fmm.upward_pass()
#     fmm.downward_pass()

#     for index in range(n_points):
#         assert len(fmm._result_data[index]) == n_points
