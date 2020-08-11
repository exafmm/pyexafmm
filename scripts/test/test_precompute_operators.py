"""
Test the precomputed operators
"""
import os
import pathlib

import numpy as np
import pytest

import fmm.hilbert as hilbert
from fmm.kernel import KERNELS
from fmm.octree import Octree
import fmm.operator as operator
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

ORDER = CONFIG['order']
SURFACE = operator.compute_surface(ORDER)
KERNEL_FUNCTION = KERNELS['laplace']()

OPERATOR_DIRPATH = HERE.parent.parent / CONFIG['operator_dirname']
DATA_DIRPATH = HERE.parent.parent / CONFIG['data_dirname']

RTOL = 1e-1

@pytest.fixture
def octree():
    sources = data.load_hdf5_to_array('sources', 'sources', DATA_DIRPATH)
    targets = data.load_hdf5_to_array('targets', 'targets', DATA_DIRPATH)

    source_densities = data.load_hdf5_to_array(
        'source_densities', 'source_densities', DATA_DIRPATH)

    return Octree(
        sources=sources,
        targets=targets,
        maximum_level=CONFIG['octree_max_level'],
        source_densities=source_densities
        )


@pytest.fixture
def m2m():
    return data.load_hdf5_to_array('m2m', 'm2m', OPERATOR_DIRPATH)


@pytest.fixture
def l2l():
    return data.load_hdf5_to_array('l2l', 'l2l', OPERATOR_DIRPATH)


@pytest.fixture
def m2l_operators():
    return operator.M2L(CONFIG_FILEPATH)


@pytest.fixture
def npoints():
    return 6*(ORDER-1)**2 + 2


def plot_surfaces(source_surface, target_surface, check_surface):
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

    ax.scatter(
        check_surface[:, 0],
        check_surface[:, 1],
        check_surface[:, 2],
    )

    plt.show()


def test_m2m(npoints, octree, m2m):

    parent_key = 0
    child_key = hilbert.get_children(parent_key)[0]

    x0 = octree.center
    r0 = octree.radius

    parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = hilbert.get_level(parent_key)
    child_level = hilbert.get_level(child_key)

    operator_idx = (child_key % 8) -1

    child_equivalent_density = np.ones(shape=(npoints))

    parent_equivalent_density = np.matmul(m2m[operator_idx], child_equivalent_density)

    distant_point = np.array([[1e3, 0, 0]])

    child_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=1.05
        )
    parent_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=parent_level,
        center=parent_center,
        alpha=1.05
        )

    parent_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=distant_point,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density
        )

    child_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=distant_point,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density
        )

    assert np.isclose(parent_direct.density, child_direct.density, rtol=RTOL)


parent_equivalent_density = np.array(
    [
        4.08530528,  4.98335145,  8.79908241, 11.36927637, 19.36460982,  4.95690057,
        6.84512128, 11.34198553, 18.25788085,  4.26111775,  6.83114839, 10.41075108,
        17.35226654, 32.10162004, 37.59156295, 44.76134101, 19.80916225, 17.38448808,
        37.55814296, 47.58635286, 19.78243739, 18.25720914, 38.90247477, 47.56580171,
        21.06584849, 19.32057619
    ]
)

def test_l2l(npoints, octree, l2l):

    parent_key = 9
    child_key = hilbert.get_children(parent_key)[-1]

    x0 = octree.center
    r0 = octree.radius

    parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = hilbert.get_level(parent_key)
    child_level = hilbert.get_level(child_key)

    # parent_equivalent_density = np.ones(shape=(npoints))

    operator_idx = (child_key % 8) - 1

    child_equivalent_density = np.matmul(l2l[operator_idx], parent_equivalent_density)

    child_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=2.95
    )

    parent_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=parent_level,
        center=parent_center,
        alpha=2.95
    )

    local_point = np.array([list(child_center)])

    parent_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density
    )

    child_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density
    )

    assert np.isclose(parent_direct.density, child_direct.density, rtol=RTOL)


f = np.array([0.22073459, 0.35864311,  0.17874369,  0.24654391,  0.15047553,  0.1210559,
  0.13997958,  0.00978738, -0.04337865,  0.32848649,  0.75337007,  0.21722818,
  0.35063859,  0.01678733, -0.03936223,  0.16834922,  0.00866047,  0.11725961,
  0.14546409,  0.20802148,  0.2423115,   0.38121081, -0.03373622, -0.20371675,
  0.01821985, -0.03785398])

def test_m2l(npoints, octree, m2l_operators):

    # pick a target box on level 2 or below
    x0 = octree.center
    r0 = octree.radius

    target_key = 72

    source_level = target_level = hilbert.get_level(target_key)

    m2l = m2l_operators.operators[source_level]

    target_index = hilbert.remove_level_offset(target_key)
    target_center = hilbert.get_center_from_key(target_key, x0, r0)
    interaction_list = hilbert.get_interaction_list(target_key)

    # pick a source box in target's interaction list
    source_key = interaction_list[2]
    source_center = hilbert.get_center_from_key(source_key, x0, r0)

    # get the operator index from current level lookup table
    index_to_key = m2l_operators.index_to_key[target_level][target_index]
    source_index = np.where(source_key == index_to_key)[0][0]

    # place unit densities on source box
    # source_equivalent_density = np.ones(shape=(npoints))
    # source_equivalent_density = np.random.rand(npoints)
    source_equivalent_density = f

    source_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=source_level,
        center=source_center,
        alpha=1.05
    )

    m2l_matrix = m2l[target_index][source_index]

    target_equivalent_density = np.matmul(m2l_matrix, source_equivalent_density)

    target_equivalent_surface = operator.scale_surface(
        surface=SURFACE,
        radius=r0,
        level=target_level,
        center=target_center,
        alpha=2.95
    )

    local_point = np.array([list(target_center)])

    target_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=target_equivalent_surface,
        source_densities=target_equivalent_density
    )

    source_direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=local_point,
        sources=source_equivalent_surface,
        source_densities=source_equivalent_density
    )

    assert np.isclose(target_direct.density, source_direct.density, rtol=RTOL)
