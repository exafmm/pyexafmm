"""
Test the precomputed operators
"""
import os
import pathlib
import subprocess

import numpy as np
import pytest

from fmm.octree import Octree
import fmm.hilbert
from fmm.kernel import Laplace
from fmm.operator import compute_surface, scale_surface, p2p, M2LOperators
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

ORDER = CONFIG['order']
SURFACE = compute_surface(ORDER)
KERNEL_FUNCTION = Laplace()

OPERATOR_DIRPATH = HERE.parent.parent / CONFIG['operator_dirname']
DATA_DIRPATH = HERE.parent.parent / CONFIG['data_dirname']


@pytest.fixture
def octree():
    sources = data.load_hdf5_to_array('sources', 'sources', DATA_DIRPATH)
    targets = data.load_hdf5_to_array('targets', 'targets', DATA_DIRPATH)

    source_densities = data.load_hdf5_to_array(
        'source_densities', 'source_densities', DATA_DIRPATH)

    return Octree(sources, targets, CONFIG['octree_max_level'], source_densities)


@pytest.fixture
def m2m():
    return data.load_hdf5_to_array('m2m', 'm2m', OPERATOR_DIRPATH)


@pytest.fixture
def l2l():
    return data.load_hdf5_to_array('l2l', 'l2l', OPERATOR_DIRPATH)


@pytest.fixture
def m2l_operators():
    return M2LOperators(CONFIG_FILEPATH)


@pytest.fixture
def npoints():
    return 6*(ORDER-1)**2 + 2


def plot_surfaces(source_surface, target_surface, check_surface=None):
    """
    Plot surfaces for testing purposes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        source_surface[:, 0],
        source_surface[:, 1],
        source_surface[:, 2],
        color='red'
        )

    ax.scatter(
        target_surface[:, 0],
        target_surface[:, 1],
        target_surface[:, 2],
        color='green'
     )

    if check_surface is not None:
        ax.scatter(
            check_surface[:, 0],
            check_surface[:, 1],
            check_surface[:, 2],
        )

    plt.show()


def test_m2m(npoints, octree, m2m):

    parent_key = 0
    child_key = fmm.hilbert.get_children(parent_key)[0]

    x0 = octree.center
    r0 = octree.radius

    parent_center = fmm.hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = fmm.hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = fmm.hilbert.get_level(parent_key)
    child_level = fmm.hilbert.get_level(child_key)

    operator_idx = (child_key % 8) -1

    child_equivalent_density = np.ones(shape=(npoints))

    parent_equivalent_density = np.matmul(m2m[operator_idx], child_equivalent_density)

    distant_point = np.array([[1e3, 0, 0]])

    child_equivalent_surface = scale_surface(SURFACE, r0, child_level, child_center, 1.05)
    parent_equivalent_surface = scale_surface(SURFACE, r0, parent_level, parent_center, 1.05)

    parent_direct = p2p(KERNEL_FUNCTION, distant_point, parent_equivalent_surface, parent_equivalent_density)
    child_direct = p2p(KERNEL_FUNCTION, distant_point, child_equivalent_surface, child_equivalent_density)

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.1)


def test_l2l(npoints, octree, l2l):

    parent_key = 9
    child_key = fmm.hilbert.get_children(parent_key)[-1]

    x0 = octree.center
    r0 = octree.radius

    parent_center = fmm.hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = fmm.hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = fmm.hilbert.get_level(parent_key)
    child_level = fmm.hilbert.get_level(child_key)

    parent_equivalent_density = np.ones(shape=(npoints))

    operator_idx = (child_key % 8) - 1

    child_equivalent_density = np.matmul(l2l[operator_idx], parent_equivalent_density)

    child_equivalent_surface = scale_surface(SURFACE, r0, child_level, child_center, 2.95)
    parent_equivalent_surface = scale_surface(SURFACE, r0, parent_level, parent_center, 2.95)

    local_point = np.array([list(child_center)])

    parent_direct = p2p(KERNEL_FUNCTION, local_point, parent_equivalent_surface, parent_equivalent_density)
    child_direct = p2p(KERNEL_FUNCTION, local_point, child_equivalent_surface, child_equivalent_density)

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.1)


def test_m2l(npoints, octree, m2l_operators):

    # pick a target box on level 2 or below
    x0 = octree.center
    r0 = octree.radius

    target_key = 72

    source_level = target_level = fmm.hilbert.get_level(target_key)

    m2l = m2l_operators.operators[source_level]

    target_index = fmm.hilbert.remove_offset(target_key)
    target_center = fmm.hilbert.get_center_from_key(target_key, x0, r0)
    interaction_list = fmm.hilbert.compute_interaction_list(target_key)

    # pick a source box in target's interaction list
    source_key = interaction_list[2]
    source_center = fmm.hilbert.get_center_from_key(source_key, x0, r0)

    # get the operator index from current level lookup table
    index_to_key = m2l_operators.index_to_key[target_level][target_index]
    source_index = np.where(source_key == index_to_key)[0][0]

    # place unit densities on source box
    # source_equivalent_density = np.ones(shape=(npoints))
    rand = np.random.rand
    source_equivalent_density = rand(npoints)
    source_equivalent_surface = scale_surface(SURFACE, r0, source_level, source_center, 1.05)

    m2l_matrix = m2l[target_index][source_index]

    target_equivalent_density = np.matmul(m2l_matrix, source_equivalent_density)
    target_equivalent_surface = scale_surface(SURFACE, r0, target_level, target_center, 2.95)

    local_point = np.array([list(target_center)])

    target_direct = p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=target_equivalent_surface,
        source_densities=target_equivalent_density
    )

    source_direct = p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=source_equivalent_surface,
        source_densities=source_equivalent_density
    )

    assert np.isclose(target_direct.density, source_direct.density, rtol=0.1)
