"""
Test the FMM
"""
import numpy as np
import pytest

import fmm.hilbert as hilbert
from fmm.octree import Octree
from fmm.fmm import Fmm

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

def test_upward_pass(octree):
    """Test the upward pass."""

    fmm = Fmm(octree, ORDER)
    fmm.upward_pass()
    fmm.downward_pass()
    breakpoint()
