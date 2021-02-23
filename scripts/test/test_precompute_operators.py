"""
Test the precomputed operators
"""
import os
import pathlib

import h5py
import numpy as np
import pytest

import adaptoctree.morton as morton

from fmm.kernel import KERNELS
import fmm.operator as operator
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent.parent
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

RTOL = 1e-1


@pytest.fixture
def db():
    experiment = CONFIG['experiment']
    return h5py.File(ROOT / f'{experiment}.hdf5', 'r')


def test_m2m(db):

    parent_key = 0
    child_key = morton.find_children(parent_key)[0]

    x0 = db['octree']['x0'][...]
    r0 = db['octree']['r0'][...]
    surface = db['surface'][...]
    npoints = len(surface)
    kernel = CONFIG['kernel']
    p2p_function = KERNELS[kernel]['p2p']

    parent_center = morton.find_physical_center_from_key(parent_key, x0, r0)
    child_center = morton.find_physical_center_from_key(child_key, x0, r0)

    parent_level = morton.find_level(parent_key)
    child_level = morton.find_level(child_key)

    operator_idx = 0

    child_equivalent_density = np.ones(shape=(npoints))

    parent_equivalent_density = np.matmul(db['m2m'][operator_idx], child_equivalent_density)

    distant_point = np.array([[1e3, 0, 0]])

    child_equivalent_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=CONFIG['alpha_inner']
    )

    parent_equivalent_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=parent_level,
        center=parent_center,
        alpha=CONFIG['alpha_inner']
    )

    parent_direct = p2p_function(
        targets=distant_point,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density
    )

    child_direct = p2p_function(
        targets=distant_point,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density
    )

    assert np.isclose(parent_direct, child_direct, rtol=RTOL)


# def test_l2l(npoints, octree, l2l):

#     parent_key = 9
#     child_key = hilbert.get_children(parent_key)[-1]

#     x0 = octree.center
#     r0 = octree.radius

#     parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
#     child_center = hilbert.get_center_from_key(child_key, x0, r0)

#     parent_level = hilbert.get_level(parent_key)
#     child_level = hilbert.get_level(child_key)

#     parent_equivalent_density = np.ones(shape=(npoints))

#     operator_idx = (child_key % 8) - 1

#     child_equivalent_density = np.matmul(l2l[operator_idx], parent_equivalent_density)

#     child_equivalent_surface = operator.scale_surface(
#         surface=SURFACE,
#         radius=r0,
#         level=child_level,
#         center=child_center,
#         alpha=2.95
#     )

#     parent_equivalent_surface = operator.scale_surface(
#         surface=SURFACE,
#         radius=r0,
#         level=parent_level,
#         center=parent_center,
#         alpha=2.95
#     )

#     local_point = np.array([list(child_center)])

#     parent_direct = operator.p2p(
#         kernel_function=KERNEL_FUNCTION,
#         targets=local_point,
#         sources=parent_equivalent_surface,
#         source_densities=parent_equivalent_density
#     )

#     child_direct = operator.p2p(
#         kernel_function=KERNEL_FUNCTION,
#         targets=local_point,
#         sources=child_equivalent_surface,
#         source_densities=child_equivalent_density
#     )

#     assert np.isclose(parent_direct.density, child_direct.density, rtol=RTOL)


# def test_m2l(npoints, octree, m2l_operators):

#     # pick a target box on level 2 or below
#     x0 = octree.center
#     r0 = octree.radius

#     target_key = 72

#     source_level = target_level = hilbert.get_level(target_key)

#     m2l = m2l_operators.operators[source_level]

#     target_index = hilbert.remove_level_offset(target_key)
#     target_center = hilbert.get_center_from_key(target_key, x0, r0)
#     interaction_list = hilbert.get_interaction_list(target_key)

#     # pick a source box in target's interaction list
#     source_key = interaction_list[2]
#     source_center = hilbert.get_center_from_key(source_key, x0, r0)

#     # get the operator index from current level lookup table
#     index_to_key = m2l_operators.index_to_key[target_level][target_index]
#     source_index = np.where(source_key == index_to_key)[0][0]

#     # place unit densities on source box
#     # source_equivalent_density = np.ones(shape=(npoints))
#     source_equivalent_density = np.random.rand(npoints)

#     source_equivalent_surface = operator.scale_surface(
#         surface=SURFACE,
#         radius=r0,
#         level=source_level,
#         center=source_center,
#         alpha=1.05
#     )

#     m2l_matrix = m2l[target_index][source_index]

#     target_equivalent_density = np.matmul(m2l_matrix, source_equivalent_density)

#     target_equivalent_surface = operator.scale_surface(
#         surface=SURFACE,
#         radius=r0,
#         level=target_level,
#         center=target_center,
#         alpha=2.95
#     )

#     local_point = np.array([list(target_center)])

#     target_direct = operator.p2p(
#         kernel_function=KERNEL_FUNCTION,
#         targets=local_point,
#         sources=target_equivalent_surface,
#         source_densities=target_equivalent_density
#     )

#     source_direct = operator.p2p(
#         kernel_function=KERNEL_FUNCTION,
#         targets=local_point,
#         sources=source_equivalent_surface,
#         source_densities=source_equivalent_density
#     )

#     assert np.isclose(target_direct.density, source_direct.density, rtol=RTOL)
