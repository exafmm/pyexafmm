"""
Test the precomputed operators
"""
import os
import pathlib

import h5py
from numba.core.types.npytypes import DType
import numpy as np
import pytest

import adaptoctree.morton as morton


from fmm.dtype import NUMPY
from fmm.kernel import KERNELS
import fmm.surface as surface

import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent.parent
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)
TOL =  1e-5


@pytest.fixture
def db():
    experiment = CONFIG["experiment"]
    return h5py.File(ROOT / f"{experiment}.hdf5", "r")


def test_m2m(db):
    """
    Test the convergence of the multipole expansion of the root node. If this
        as expected, then the M2M translation operators as well as the P2M step
        work.
    """
    parent_key = 0
    child_key = morton.find_children(parent_key)[0]

    x0 = db["octree"]["x0"][...]
    r0 = db["octree"]["r0"][...][0]
    equivalent_surface = db["surface"]["equivalent"][...]

    npoints_equivalent = len(equivalent_surface)

    kernel = CONFIG["kernel"]
    p2p_function = KERNELS[kernel]["p2p"]

    parent_center = morton.find_physical_center_from_key(parent_key, x0, r0)
    child_center = morton.find_physical_center_from_key(child_key, x0, r0)

    parent_level = morton.find_level(parent_key)
    child_level = morton.find_level(child_key)

    operator_idx = 0

    child_equivalent_density = np.ones(shape=npoints_equivalent)

    parent_equivalent_density = np.matmul(
        db["m2m"][operator_idx], child_equivalent_density
    )

    distant_point = parent_center+r0*CONFIG['alpha_outer']*1.1

    child_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=CONFIG["alpha_inner"],
    )

    parent_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=r0,
        level=parent_level,
        center=parent_center,
        alpha=CONFIG["alpha_inner"],
    )

    parent_direct = p2p_function(
        targets=distant_point,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density,
    )

    child_direct = p2p_function(
        targets=distant_point,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density,
    )


    assert np.isclose(parent_direct, child_direct, atol=TOL, rtol=0)


def test_l2l(db):
    """
    Test the convergence of the local expansion after L2L has been performed
        on a given parent/child pair.
    """
    parent_key = 1
    child_key = morton.find_children(parent_key)[-1]

    x0 = db["octree"]["x0"][...]
    r0 = db["octree"]["r0"][...][0]

    equivalent_surface = db["surface"]['equivalent'][...]

    npoints_equivalent = len(equivalent_surface)

    kernel = CONFIG["kernel"]
    p2p_function = KERNELS[kernel]["p2p"]

    parent_center = morton.find_physical_center_from_key(parent_key, x0, r0)
    child_center = morton.find_physical_center_from_key(child_key, x0, r0)

    parent_level = morton.find_level(parent_key)
    child_level = morton.find_level(child_key)

    parent_equivalent_density = np.ones(shape=npoints_equivalent)

    operator_idx = 0

    child_equivalent_density = np.matmul(
        db['l2l'][operator_idx], parent_equivalent_density
    )

    child_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface, radius=r0, level=child_level,
        center=child_center, alpha=CONFIG['alpha_outer']
    )

    parent_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface, radius=r0, level=parent_level,
        center=parent_center, alpha=CONFIG['alpha_outer']
    )

    local_point = child_center

    parent_direct = p2p_function(
        targets=local_point,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density,
    )

    child_direct = p2p_function(
        targets=local_point,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density,
    )

    assert np.isclose(parent_direct, child_direct, atol=TOL, rtol=0)


def test_m2l(db):
    """
    Test convergence of local expansion after M2L translation has been applied,
        as well as SVD compression by computing result from all multipole
        expansions of source nodes for a given target node in its V list, and
        comparing with local expansion result computed with compressed M2L
        translation operator.
    """
    x0 = db["octree"]["x0"][...]
    r0 = db["octree"]["r0"][...][0]
    dc2e_inv = db['dc2e_inv'][...]
    equivalent_surface = db["surface"]["equivalent"][...]
    npoints_equivalent = len(equivalent_surface)
    check_surface = db['surface']['check'][...]
    npoints_check = len(check_surface)
    kernel = CONFIG["kernel"]
    p2p_function = KERNELS[kernel]["p2p"]

    # Pick a target key with a non-empty interaction list
    complete = db['octree']['complete']
    v_lists = db['interaction_lists']['v']

    for i, v_list in enumerate(v_lists):
        if complete[i] != 0:
            if len(v_list[v_list != -1]) > 0:
                target_key = complete[i]
                target_index = i
                break

    v_list = v_lists[target_index]
    v_list = v_list[v_list != -1]

    source_level = target_level = morton.find_level(target_key)

    scale = (1/2)**target_level

    target_center = morton.find_physical_center_from_key(target_key, x0, r0)

    # Construct a vector of source points for all boxes in v_list
    sources = np.zeros(shape=(npoints_equivalent*len(v_list), 3))
    for idx in range(len(v_list)):
        source_key = v_list[idx]
        source_center = morton.find_physical_center_from_key(source_key, x0, r0)

        source_equivalent_surface = surface.scale_surface(
            surf=equivalent_surface,
            radius=r0,
            level=source_level,
            center=source_center,
            alpha=CONFIG['alpha_inner']
        )

        lidx = idx*npoints_equivalent
        ridx = lidx+npoints_equivalent

        sources[lidx:ridx] = source_equivalent_surface

    # place densities on source boxes
    source_equivalent_density = np.random.rand(len(v_list)*npoints_equivalent)

    u = db['m2l'][str(target_level)]['u']
    s = db['m2l'][str(target_level)]['s']
    vt = db['m2l'][str(target_level)]['vt']
    hashes = db['m2l'][str(target_level)]['hashes'][...]

    target_equivalent_density = np.zeros(npoints_equivalent)

    for i in range(len(v_list)):

        source_key = v_list[i]
        transfer_vec = morton.find_transfer_vector(target_key, source_key)
        m2l_idx = np.where(transfer_vec == hashes)[0][0]
        m2l_lidx = m2l_idx*npoints_equivalent
        m2l_ridx = m2l_lidx+npoints_equivalent
        vt_sub = vt[:, m2l_lidx:m2l_ridx]

        lidx = i*npoints_equivalent
        ridx = lidx+npoints_equivalent
        target_equivalent_density_sub = (scale*dc2e_inv) @ (u @ np.diag(s) @ vt_sub) @ source_equivalent_density[lidx:ridx]
        target_equivalent_density += target_equivalent_density_sub

    targets = surface.scale_surface(
        surf=equivalent_surface,
        radius=r0,
        level=target_level,
        center=target_center,
        alpha=CONFIG['alpha_outer']
    )

    local_point = target_center

    target_direct = p2p_function(
        targets=local_point,
        sources=targets,
        source_densities=target_equivalent_density
    )

    source_direct = p2p_function(
        targets=local_point,
        sources=sources,
        source_densities=source_equivalent_density
    )

    assert np.isclose(target_direct, source_direct, atol=0.5, rtol=0)
