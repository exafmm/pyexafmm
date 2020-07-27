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
DATA_DIRPATH = HERE.parent.parent / CONFIG['data_dirname']


def setup_module(module):
    os.chdir(HERE.parent)
    subprocess.run(['python', 'precompute_operators.py', CONFIG_FILEPATH])
    os.chdir('test')


def teardown_module(module):
    os.chdir(HERE.parent.parent)
    subprocess.run(['rm', '-fr', f'precomputed_operators_order_{ORDER}'])


@pytest.fixture
def octree():
    sources = data.load_hdf5_to_array('random_sources', 'random_sources', DATA_DIRPATH)
    targets = data.load_hdf5_to_array('random_sources', 'random_sources', DATA_DIRPATH)

    source_densities = data.load_hdf5_to_array(
        'source_densities', 'source_densities', DATA_DIRPATH)

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
    child_key = fmm.hilbert.get_children(parent_key)[0]

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

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.05)


def test_l2l(npoints, octree, l2l):

    parent_key = 0
    child_key = fmm.hilbert.get_children(parent_key)[0]

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

    assert np.isclose(parent_direct.density, child_direct.density, rtol=0.05)


def test_m2l(
    npoints,
    octree,
    m2l,
    sources_relative_to_targets,
    ):

