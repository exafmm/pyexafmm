"""
Test the FMM
"""
import numpy as np
import pytest

import fmm.hilbert as hilbert
from fmm.octree import Octree
from fmm.fmm import Fmm, cartesian_to_spherical, direct_calculation, p2m

NPOINTS = 1000
MAXIMUM_LEVEL = 3
ORDER = 5


@pytest.fixture
def octree():
    """Fill up an octree."""

    rand = np.random.RandomState(0)

    sources = rand.rand(NPOINTS, 3)
    targets = rand.rand(NPOINTS, 3)

    return Octree(sources, targets, MAXIMUM_LEVEL)

def test_upward_and_donward_pass(octree):
    """Test the upward pass."""

    fmm = Fmm(octree, ORDER)
    fmm.upward_pass()
    fmm.downward_pass()

    for index in range(NPOINTS):
        assert len(fmm._result_data[index]) == NPOINTS


def test_p2m(octree):
    """Test Multipole expansion"""
    sources = octree._sources
    sources_sph_harm = np.apply_along_axis(cartesian_to_spherical, 1, sources)

    # well separated
    target = np.array([10, 10, 10])
    target_sph_harm = cartesian_to_spherical(target)

    direct_result = direct_calculation(sources, target)
    p2m_result = p2m(sources_sph_harm, target_sph_harm, 1)

    assert np.around(direct_result) == np.around(p2m_result)
