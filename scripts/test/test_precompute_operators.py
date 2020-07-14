"""
Test the precomputed operators
"""

import numpy as np
import pytest

from fmm.octree import Octree
import fmm.hilbert
import utils.data as data
from fmm.fmm import p2p, Laplace, gram_matrix
from fmm.operator import compute_surface, scale_surface


@pytest.fixture
def order():
    return 5


@pytest.fixture
def surface(order):
    return compute_surface(order)


def test_m2m(order, surface):

    sources = data.load_hdf5_to_array('sources', 'random_sources', '../../data')
    targets = data.load_hdf5_to_array('targets', 'random_targets', '../../data')

    octree = Octree(sources, targets, maximum_level=5)

    m2m = data.load_hdf5_to_array(
        'm2m', 'm2m', '../../precomputed_operators_order_5')

    npoints = 6*(order-1)**2 + 2

    parent_key = 0
    child_key = 1

    x0 = octree.center
    r0 = octree.radius

    parent_center = fmm.hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = fmm.hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = fmm.hilbert.get_level(parent_key)
    child_level = fmm.hilbert.get_level(child_key)

    child_equivalent_density = np.ones(shape=(npoints))

    parent_equivalent_density = np.matmul(m2m[0], child_equivalent_density)

    print(parent_equivalent_density)
    laplace = Laplace()

    distant_point = np.array([[1e3, 0, 0]])

    child_equivalent_surface = scale_surface(surface, r0, child_level, child_center, 1.05)
    parent_equivalent_surface = scale_surface(surface, r0, parent_level, parent_center, 1.05)

    parent_direct = p2p(laplace, distant_point, parent_equivalent_surface, parent_equivalent_density)
    child_direct = p2p(laplace, distant_point, child_equivalent_surface, child_equivalent_density)

    print('parent result', parent_direct)
    print('child direct', child_direct)

    assert False


