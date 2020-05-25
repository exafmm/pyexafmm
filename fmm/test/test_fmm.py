"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.octree import Octree
from fmm.fmm import Fmm, laplace, surface, potential_p2p, p2m


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

@pytest.fixture
def dummy_surface(order, max_level, octree):

    return surface(
        order=order,
        radius=octree.radius,
        level=max_level,
        center=octree.center,
        alpha=2.95
    )

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


def test_potential_p2p(octree, order, max_level): 
    """Test with single-layer Laplace kernel"""

    sources = octree.sources
    targets = octree.targets

    source_densities = np.ones(shape=(len(sources)))

    k, surf, densities = p2m(
        laplace,
        order,
        octree.center,
        octree.radius,
        1,
        octree.sources
    )

    print('condition number', np.linalg.cond(k))

    # assert np.array_equal(result, expected)
    distant_point = np.array([[2009, 0, 0], [0, 2500, 0]])

    direct = potential_p2p(laplace, distant_point, sources, source_densities)
    print(f'direct {direct}')

    surface = potential_p2p(laplace, distant_point, surf, densities)

    print(f'surface {surface}')

    print(surf.shape)

    # Potential p2p needs to be passed the source densities to work
    # properly for the comparison in the far field

    assert False

# def test_p2m(octree, order, max_level):
#     """
#     Compare far-field approximation at a 'distant point'
#     """

#     p2m_density = p2m(
#         kernel=laplace,
#         order=order,
#         center=octree.center,
#         radius=octree.radius,
#         maximum_level=1,
#         leaf_sources=octree.sources,
#     )

#     distant_point = np.array([[1000, 0, 0]])

#     # Calculate effect at distant point
#     direct_result = potential_p2p(laplace, distant_point, octree.sources)
#     p2m_result = potential_p2p(laplace, distant_point, p2m_density)

#     print(f'direct result {direct_result}')
#     print(f'p2m result {p2m_result}')

#     # print(p2m_density)

#     print(p2m_density[0])

#     assert True

def test_m2m():
    assert True

# def test_upward_and_downward_pass(order, n_points, octree):
#     """Test the upward pass."""

#     fmm = Fmm(octree, order, laplace)
#     fmm.upward_pass()
#     fmm.downward_pass()

#     for index in range(n_points):
#         assert len(fmm._result_data[index]) == n_points
