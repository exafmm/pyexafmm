"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.octree import Octree
from fmm.fmm import Fmm, laplace


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

def test_gram_matrix():
    assert True

def test_surface():
    assert True

def test_potential_p2p():
    assert True

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
