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
from fmm.operator import compute_surface, scale_surface, p2p, compute_m2l_operator_index
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

ORDER = CONFIG['order']
SURFACE = compute_surface(ORDER)
KERNEL_FUNCTION = Laplace()

OPERATOR_DIRPATH = HERE.parent.parent / f'precomputed_operators_order_{ORDER}'


def setup_module(module):
    os.chdir(HERE.parent)
    subprocess.run(['python', 'precompute_operators.py', CONFIG_FILEPATH])
    os.chdir('test')


def teardown_module(module):
    os.chdir(HERE.parent.parent)
    subprocess.run(['rm', '-fr', f'precomputed_operators_order_{ORDER}'])


@pytest.fixture
def octree():
    sources = data.load_hdf5_to_array('sources', 'random_sources', '../../data')
    targets = data.load_hdf5_to_array('targets', 'random_targets', '../../data')
    source_densities = data.load_hdf5_to_array(
        'source_densities', 'source_densities', '../../data')

    return Octree(sources, targets, 5, source_densities)


@pytest.fixture
def m2m():
    return data.load_hdf5_to_array('m2m', 'm2m', OPERATOR_DIRPATH)


@pytest.fixture
def l2l():
    return data.load_hdf5_to_array('l2l', 'l2l', OPERATOR_DIRPATH)


@pytest.fixture
def m2l():
    return data.load_hdf5_to_array('m2l', 'm2l', OPERATOR_DIRPATH)


@pytest.fixture
def sources_relative_to_targets():
    return data.load_hdf5_to_array(
        'sources_relative_to_targets', 'sources_relative_to_targets',
        OPERATOR_DIRPATH
    )


@pytest.fixture
def npoints():
    return 6*(ORDER-1)**2 + 2


def test_m2m(npoints, octree, m2m):

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

    distant_point = np.array([[1e3, 0, 0]])

    child_equivalent_surface = scale_surface(SURFACE, r0, child_level, child_center, 1.05)
    parent_equivalent_surface = scale_surface(SURFACE, r0, parent_level, parent_center, 1.05)

    parent_direct = p2p(KERNEL_FUNCTION, distant_point, parent_equivalent_surface, parent_equivalent_density)
    child_direct = p2p(KERNEL_FUNCTION, distant_point, child_equivalent_surface, child_equivalent_density)

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.1)


def test_l2l(npoints, octree, l2l):

    parent_key = 0
    child_key = 2

    x0 = octree.center
    r0 = octree.radius

    parent_center = fmm.hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = fmm.hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = fmm.hilbert.get_level(parent_key)
    child_level = fmm.hilbert.get_level(child_key)

    parent_equivalent_density = np.ones(shape=(npoints))

    operator_idx = (child_key % 8) -1

    child_equivalent_density = np.matmul(l2l[operator_idx], parent_equivalent_density)

    child_equivalent_surface = scale_surface(SURFACE, r0, child_level, child_center, 2.95)
    parent_equivalent_surface = scale_surface(SURFACE, r0, parent_level, parent_center, 2.95)

    local_point = np.array([list(child_center)])

    parent_direct = p2p(KERNEL_FUNCTION, local_point, parent_equivalent_surface, parent_equivalent_density)
    child_direct = p2p(KERNEL_FUNCTION, local_point, child_equivalent_surface, child_equivalent_density)

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.1)

def test_m2l(
    npoints,
    octree,
    m2l,
    sources_relative_to_targets,
    ):

    source_level = target_level = 5
    x0 = octree.center
    r0 = octree.radius

    # pick a source box on level 3 or below
    source_key = fmm.hilbert.get_key_from_point(x0, source_level, x0, r0)
    source_4d_index = fmm.hilbert.get_4d_index_from_key(source_key)
    source_center = fmm.hilbert.get_center_from_key(source_key, x0, r0)
    interaction_list = fmm.hilbert.compute_interaction_list(source_key)

    # pick a target box in source's interaction list
    target_key = interaction_list[3]
    target_center = fmm.hilbert.get_center_from_key(target_key, x0, r0)
    target_4d_index = fmm.hilbert.get_4d_index_from_key(target_key)

    operator_index = compute_m2l_operator_index(
        sources_relative_to_targets, source_4d_index, target_4d_index)

    # place unit densities on source box
    source_equivalent_density = np.ones(shape=(npoints))
    source_equivalent_surface = scale_surface(SURFACE, r0, source_level, source_center, 1.05)

    operator = m2l[operator_index].reshape((npoints, npoints))
    target_equivalent_density = np.matmul(operator, source_equivalent_density)
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
        source_densities=source_equivalent_density,
        sources=source_equivalent_surface
    )

    assert np.isclose(target_direct.density, source_direct.density, rtol=0.1)
