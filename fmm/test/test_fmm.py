"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.octree import Octree
from fmm.fmm import Fmm, laplace, surface, potential_p2p


@pytest.fixture
def n_points():
    return 10

@pytest.fixture
def max_level():
    return 1

@pytest.fixture
def order():
    return 2

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


def test_potential_p2p(octree):
    potential = potential_p2p(laplace, octree.targets, octree.sources)

    assert potential.shape == (len(octree.targets), 1)

def test_p2m():
    assert True

def test_m2m():
    assert True

def test_upward_and_downward_pass(order, n_points, octree):
    """Test the upward pass."""

    fmm = Fmm(octree, order, laplace)
    fmm.upward_pass()
    fmm.downward_pass()

    for index in range(n_points):
        assert len(fmm._result_data[index]) == n_points
