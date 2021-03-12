"""
Test the fmm module.

Configured with test data 'test.hdf5' and a test config, 'test_config.json',
bundled with the package.
"""
import numpy as np

import adaptoctree.morton as morton
from adaptoctree.tree import find_dense_v_list

from fmm.fmm import Fmm, _source_to_local
import fmm.surface as surface
from fmm.kernel import KERNELS


RTOL = 1e-1


def test_upward_pass():
    """
    Test that multipole expansion of root node is the same as a direct
        computation for a set of source points, at a target point located
        at a distance.
    """
    fmm = Fmm('test_config')
    fmm.upward_pass()

    upward_equivalent_surface = surface.scale_surface(
        surf=fmm.equivalent_surface,
        radius=fmm.r0,
        level=np.int32(0),
        center=fmm.x0,
        alpha=fmm.alpha_inner
    )

    distant_point = np.array([[1e4, 0, 0]])

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(
        sources=fmm.sources,
        targets=distant_point,
        source_densities=fmm.source_densities
    )

    equivalent = p2p(
        sources=upward_equivalent_surface,
        targets=distant_point,
        source_densities=fmm.multipole_expansions[0]
    )

    assert np.allclose(direct, equivalent, rtol=RTOL)


def test_m2l():
    """
    Test that the local expansions of a given target node correspond to the
        multipole expansions of source nodes in its V list at a local point
        within the target node.
    """
    fmm = Fmm('test_config')

    fmm.run()

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    idxs = fmm.complete_levels == 2
    key = fmm.complete[idxs][0]
    idx = np.where(fmm.complete == key)
    v_list = fmm.v_lists[idx]
    v_list = v_list[v_list != -1]

    target_idxs = key == fmm.targets_to_keys
    target_coordinates = fmm.targets[target_idxs]

    print(target_coordinates.shape)
    level = np.int32(morton.find_level(key))
    center = morton.find_physical_center_from_key(key, fmm.x0, fmm.r0).astype(np.float32)

    local_expansion = fmm.local_expansions[key]

    downward_equivalent_surface = surface.scale_surface(
        surf=fmm.equivalent_surface,
        radius=fmm.r0,
        level=level,
        center=center,
        alpha=fmm.alpha_outer
    )

    equivalent = p2p(
        sources=downward_equivalent_surface,
        targets=target_coordinates,
        source_densities=local_expansion
    )

    direct = np.zeros_like(equivalent)

    for source in v_list:

        source_level = np.int32(morton.find_level(source))
        source_center = morton.find_physical_center_from_key(source, fmm.x0, fmm.r0).astype(np.float32)

        upward_equivalent_surface = surface.scale_surface(
            surf=fmm.equivalent_surface,
            radius=fmm.r0,
            level=source_level,
            center=source_center,
            alpha=fmm.alpha_inner
        )

        tmp = p2p(
            sources=upward_equivalent_surface,
            targets=target_coordinates,
            source_densities=fmm.multipole_expansions[source]
        )

        direct += tmp

    assert np.allclose(direct, equivalent, rtol=RTOL)


def test_fmm():
    """
    End To End Fmm Test.
    """
    fmm = Fmm('test_config')

    fmm.run()

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(
        targets=fmm.targets,
        sources=fmm.sources,
        source_densities=fmm.source_densities
    )

    assert np.allclose(direct, fmm.target_potentials, rtol=0.1)
